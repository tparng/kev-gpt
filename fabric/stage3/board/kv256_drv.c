/* kv256_drv.c — compiled MMIO inner loop for the resident kv256 SQRV engine.
 *
 * The deployed faithful-stream chat (pl_kv256.PLKV256Device) drives the fabric
 * one register at a time from Python (each /dev/mem poke a ~7.75us mmap+numpy
 * round trip). MEASURED: a localhost reply is 17.8 ms of which the fabric is
 * only 7.21 ms — the rest is per-token host tax (the prompt-prime + feedback GO
 * train is ~5-6 pokes/token). This driver moves that GO train into compiled
 * volatile stores/loads, exactly as gemv_axi_drv.c did for the resident GEMV.
 *
 * BIT-EXACT by construction: the register writes and their ORDER are identical
 * to PLKV256Device._go / _stream_gen (TOKID, POS, [SEED on the last prompt GO],
 * CTRL=go, spin STATUS.done, read TOK_OUT). On-chip sampling (seed != 0) or
 * greedy (seed == 0) both land the next token in TOK_OUT, so the whole feedback
 * loop stays in fabric — no host logit readback, so no host sampling here (the
 * host_sample path keeps the Python driver). Same seed => same fabric xorshift
 * => same token stream as Python; verify with kv256_ab_test.py on the board.
 *
 * Targets the EXISTING kv256 bitstream (IDCODE "SQRV" 0x5351_5256 @ 0xA000_0000)
 * — no new bitstream, no Vivado. Register map (see pl_kv256.py):
 *   0x00 CTRL (b0 go)              0x04 STATUS (b0 done)
 *   0x08 TOKID    0x0C POS         0x10 WDATA (weight/emb load stream)
 *   0x14 RDSEL 0x18 RDADDR 0x1C RDLO 0x20 RDHI   (debug readback)
 *   0x24 TOKOUT   0x28 CYCLES      0x2C IDCODE    0x30 SEED (0 => greedy)
 *
 * Build on the board:  cc -O2 -fPIC -shared -o libkv256_drv.so kv256_drv.c
 * ctypes API (see pl_kv256_c.py):
 *   int   kv256_open(unsigned long base);   // mmap + IDCODE check; 0 ok, <0 err
 *   void  kv256_close(void);
 *   unsigned kv256_idcode(void);
 *   long  kv256_run(const int *ids, int L, int gen, unsigned seed, int *out);
 *         // prime ids[0..L-1] from pos 0, then generate `gen` tokens into out[]
 *         // (out[0] = last-prompt TOK_OUT), returns total fabric cycles / -1 hang.
 */
#include <stdint.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define CTRL   0x00
#define STAT   0x04
#define TOKID  0x08
#define POS    0x0C
#define TOKOUT 0x24
#define CYC    0x28
#define IDC    0x2C
#define SEED   0x30

#define IDCODE_SQRV 0x53515256u   /* "SQRV" */
#define MAP_SIZE    0x1000u
#define TOK_MASK    0x1FFu        /* 9-bit token id / position, matches Python &0x1FF */
#define SPIN_MAX    200000000L    /* ~hundreds of ms ceiling per GO (hang guard) */

static int              g_fd  = -1;
static volatile uint8_t *g_map = NULL;

static inline void wr(unsigned off, uint32_t v) {
    *(volatile uint32_t *)(g_map + off) = v;      /* single aligned 32-bit store */
}
static inline uint32_t rd(unsigned off) {
    return *(volatile uint32_t *)(g_map + off);
}

/* mmap the AXI slave window. returns 0 on success, <0 on error. */
int kv256_open(unsigned long base) {
    if (g_map) return 0;                           /* already open */
    g_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_fd < 0) return -1;
    void *m = mmap(NULL, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                   g_fd, (off_t)base);
    if (m == MAP_FAILED) { close(g_fd); g_fd = -1; return -2; }
    g_map = (volatile uint8_t *)m;
    if (rd(IDC) != IDCODE_SQRV) {                  /* wrong/no bitstream */
        munmap(m, MAP_SIZE); close(g_fd); g_fd = -1; g_map = NULL;
        return -3;
    }
    return 0;
}

void kv256_close(void) {
    if (g_map) { munmap((void *)g_map, MAP_SIZE); g_map = NULL; }
    if (g_fd >= 0) { close(g_fd); g_fd = -1; }
}

unsigned kv256_idcode(void) {
    return g_map ? rd(IDC) : 0u;
}

/* One GO: TOKID/POS (+ optional SEED) -> go -> spin done -> accumulate cycles.
 * Returns TOK_OUT (masked), or -1 on hang. *cyc_acc gets the pass cycle count. */
static inline long go_one(int tok, int pos, unsigned seed, int arm, long *cyc_acc) {
    wr(TOKID, (uint32_t)tok & TOK_MASK);
    wr(POS,   (uint32_t)pos & TOK_MASK);
    if (arm) wr(SEED, seed);                       /* arm on-chip sampler once */
    wr(CTRL, 0x1);                                 /* go */
    long spins = 0;
    while (!(rd(STAT) & 0x1u)) {
        if (++spins > SPIN_MAX) return -1;
    }
    *cyc_acc += (long)rd(CYC);
    return (long)(rd(TOKOUT) & TOK_MASK);
}

/* Full request: prime ids[0..L-1] from pos 0 (arming SEED before the last prompt
 * GO iff seed != 0), then generate `gen` tokens. Mirrors _stream_gen exactly:
 *   out[0]   = TOK_OUT of the last prompt GO (pos L-1)
 *   out[g>0] = TOK_OUT of GO(out[g-1], pos L+g-1)
 * so the emitted stream is identical to the Python driver's pre-tidy tokens.
 * Returns total fabric cycles over all passes, or -1 on any hang. */
long kv256_run(const int *ids, int L, int gen, unsigned seed, int *out) {
    if (!g_map || L < 1 || gen < 1) return -1;
    long cyc = 0;
    long am = 0;
    for (int pos = 0; pos < L; ++pos) {
        int arm = (seed != 0u) && (pos == L - 1);
        am = go_one(ids[pos], pos, seed, arm, &cyc);
        if (am < 0) return -1;
    }
    out[0] = (int)am;                              /* 1st emitted token */
    long prev = am;
    for (int g = 1; g < gen; ++g) {
        prev = go_one((int)prev, L + g - 1, 0u, 0, &cyc);
        if (prev < 0) return -1;
        out[g] = (int)prev;
    }
    return cyc;
}
