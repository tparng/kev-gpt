/* gemv_axi_drv.c — compiled MMIO inner loop for the resident PL GEMV.
 *
 * Replaces the per-register Python /dev/mem pokes (each a ~7.75us round trip
 * through mmap + numpy element store) with compiled volatile stores/loads. The
 * math + orchestration stay in Python (pl_kv_chat.py / model.goformer_kv); this
 * accelerates ONLY the per-GEMV AXI traffic: set M/K/W_BASE, soft-reset, stream
 * the K activation bytes, start, spin on done, read back the M int32 results.
 *
 * Targets the EXISTING resident bitstream gemv_axi_resident @ 0xA000_0000 — no
 * new bitstream. Register map (byte offsets, see gemv_axi_resident.v):
 *   0x00 CTRL  (b0 start, b1 ld_rst, b2 soft_reset)
 *   0x04 STATUS(b0 done, b1 busy)   0x08 M_COUNT  0x0C K_COUNT
 *   0x10 W_DATA(load stream)        0x14 X_DATA   0x18 RD_ADDR  0x1C RD_DATA
 *   0x20 CYCLES                     0x24 IDCODE("GEMR" 0x4745_4D52)  0x28 W_BASE
 *
 * Bit-exactness is unchanged: identical register writes in the identical order
 * as fabric/pl_gemv.py PLResident.matmul, just from C. Build: see run_on_board.sh
 *   cc -O2 -fPIC -shared -o libgemv_axi_drv.so gemv_axi_drv.c
 *
 * ctypes API (see pl_resident_c.py):
 *   int   gemv_open(unsigned long base);     // mmap /dev/mem -> regs; check IDCODE
 *   void  gemv_close(void);
 *   unsigned gemv_idcode(void);
 *   long  gemv_run(int M, int K, unsigned w_base,
 *                  const signed char *x,      // K activation bytes (INT8)
 *                  int *y);                   // out: M int32 results
 *                                             // returns cycles, or -1 on timeout
 */
#include <stdint.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define CTRL 0x00
#define STAT 0x04
#define MCNT 0x08
#define KCNT 0x0C
#define WDAT 0x10
#define XDAT 0x14
#define RADD 0x18
#define RDAT 0x1C
#define CYC  0x20
#define IDC  0x24
#define WBASE 0x28

#define IDCODE_RESIDENT 0x47454D52u   /* "GEMR" */
#define MAP_SIZE 0x1000u

static int             g_fd  = -1;
static volatile uint8_t *g_map = NULL;

static inline void wr(unsigned off, uint32_t v) {
    *(volatile uint32_t *)(g_map + off) = v;   /* single aligned 32-bit store */
}
static inline uint32_t rd(unsigned off) {
    return *(volatile uint32_t *)(g_map + off);
}

/* mmap the AXI slave window. returns 0 on success, <0 on error. */
int gemv_open(unsigned long base) {
    if (g_map) return 0;                         /* already open */
    g_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_fd < 0) return -1;
    void *m = mmap(NULL, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                   g_fd, (off_t)base);
    if (m == MAP_FAILED) { close(g_fd); g_fd = -1; return -2; }
    g_map = (volatile uint8_t *)m;
    if (rd(IDC) != IDCODE_RESIDENT) {            /* wrong/no bitstream */
        munmap(m, MAP_SIZE); close(g_fd); g_fd = -1; g_map = NULL;
        return -3;
    }
    return 0;
}

void gemv_close(void) {
    if (g_map) { munmap((void *)g_map, MAP_SIZE); g_map = NULL; }
    if (g_fd >= 0) { close(g_fd); g_fd = -1; }
}

unsigned gemv_idcode(void) {
    return g_map ? rd(IDC) : 0u;
}

/* One resident GEMV: weights already in URAM at byte offset w_base.
 * Mirrors PLResident.matmul exactly (same writes, same order) so the int32
 * result is bit-identical. Returns the PL cycle count, or -1 on timeout. */
long gemv_run(int M, int K, unsigned w_base,
              const signed char *x, int *y) {
    wr(CTRL, 0x4); wr(CTRL, 0x0);                /* soft reset: clear stale done */
    wr(MCNT, (uint32_t)M);
    wr(KCNT, (uint32_t)K);
    wr(WBASE, w_base);
    wr(CTRL, 0x2);                               /* ld_rst: xptr->0, URAM kept */
    for (int k = 0; k < K; ++k)
        wr(XDAT, (uint32_t)((unsigned char)x[k])); /* stream activation (low byte used) */
    wr(CTRL, 0x1);                               /* start */

    /* spin on STATUS.done (b0). bound the wait so a hang can't lock Python. */
    long spins = 0;
    while (!(rd(STAT) & 0x1u)) {
        if (++spins > 200000000L) return -1;     /* ~hundreds of ms ceiling */
    }
    long cycles = (long)rd(CYC);

    for (int m = 0; m < M; ++m) {                /* read back M int32 results */
        wr(RADD, (uint32_t)m);
        y[m] = (int)rd(RDAT);                     /* RDAT is the signed int32 acc */
    }
    return cycles;
}
