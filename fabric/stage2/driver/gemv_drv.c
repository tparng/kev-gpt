// gemv_drv — userspace driver for the Stage 2 PS+PL GEMV demo on the KV260.
//
// mmaps the gemv_axi AXI-Lite register block via /dev/mem, launches one GEMV on
// the fabric, reads back the START->DONE cycle count and the y[] result vector,
// and (optionally) checks it bit-exact against the numpy golden y.mem. This is
// the on-silicon counterpart to the sim gate: bit-exact or it fails.
//
//   sudo ./gemv_drv [base_addr] [M] [golden_y.mem]
//   sudo ./gemv_drv 0xA0000000 256 blocks_0_proj.y.mem
//
// Register map (see gemv_axi.v):
//   0x00 CTRL(W: b0=start b1=softreset)  0x04 STATUS(R: b0=done b1=busy)
//   0x08 RD_ADDR(W)  0x0C RD_DATA(R)  0x10 CYCLES(R)  0x14 IDCODE(R="GEMV")

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define CTRL    0x00
#define STATUS  0x04
#define RD_ADDR 0x08
#define RD_DATA 0x0C
#define CYCLES  0x10
#define IDCODE  0x14
#define IDCODE_EXPECT 0x47454D56u  // "GEMV"

static inline void wr(volatile uint32_t *b, int off, uint32_t v) { b[off/4] = v; }
static inline uint32_t rd(volatile uint32_t *b, int off) { return b[off/4]; }

int main(int argc, char **argv) {
    unsigned long base = (argc > 1) ? strtoul(argv[1], NULL, 0) : 0xA0000000UL;
    int M = (argc > 2) ? atoi(argv[2]) : 256;
    const char *gold = (argc > 3) ? argv[3] : NULL;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (need sudo)"); return 1; }
    volatile uint32_t *r = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, base);
    if (r == MAP_FAILED) { perror("mmap"); return 1; }

    uint32_t id = rd(r, IDCODE);
    printf("IDCODE = 0x%08X (%s)\n", id,
           id == IDCODE_EXPECT ? "GEMV ok" : "UNEXPECTED - bitstream loaded?");
    if (id != IDCODE_EXPECT) { fprintf(stderr, "abort: bad IDCODE\n"); return 2; }

    // soft reset, then launch
    wr(r, CTRL, 0x2); usleep(100); wr(r, CTRL, 0x0); usleep(10);
    wr(r, CTRL, 0x1);                       // start pulse

    // The core's DONE is a 1-cycle pulse (~10 ns) that software polling can't
    // catch; the core latches the START->DONE cycle count and the result, so we
    // wait generously (run is ~41 us @100MHz) and read the latched state.
    usleep(5000);
    uint32_t st = rd(r, STATUS);
    uint32_t cycles = rd(r, CYCLES);
    printf("STATUS=0x%08X (busy=%d done_pulse=%d)\n", st, (st >> 1) & 1, st & 1);

    int32_t *y = malloc(sizeof(int32_t) * M);
    for (int m = 0; m < M; m++) { wr(r, RD_ADDR, m); y[m] = (int32_t)rd(r, RD_DATA); }

    printf("DONE  cycles=%u  (%.2f us @100MHz)\n", cycles, cycles / 100.0);
    printf("y[0..3] = %d %d %d %d\n", y[0], y[1], y[2], y[3]);

    int rc = 0;
    if (gold) {
        FILE *f = fopen(gold, "r");
        if (!f) { perror("open golden"); return 4; }
        int mism = 0, n = 0; char line[64];
        for (int m = 0; m < M && fgets(line, sizeof line, f); m++, n++) {
            int32_t g = (int32_t)strtoul(line, NULL, 16);
            if (g != y[m]) {
                if (mism < 5) printf("  MISMATCH m=%d hw=%d gold=%d\n", m, y[m], g);
                mism++;
            }
        }
        fclose(f);
        printf("A53_PL_GEMV bitexact=%s mismatches=%d/%d cycles=%u\n",
               (mism == 0 && n == M) ? "True" : "False", mism, n, cycles);
        rc = (mism == 0 && n == M) ? 0 : 5;
    } else {
        printf("A53_PL_GEMV cycles=%u (no golden given)\n", cycles);
    }
    free(y);
    return rc;
}
