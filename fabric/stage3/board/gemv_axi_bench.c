/* gemv_axi_bench.c — standalone sanity + micro-benchmark for the C MMIO driver.
 *
 * Links against gemv_axi_drv.c. It does NOT load weights (that is the Python
 * preload path); instead it runs N back-to-back GEMV calls at a host-supplied
 * (M, K, w_base) against whatever weights are already resident in URAM, and times
 * them. Use it AFTER pl_kv_chat/pl_resident has preloaded the model (URAM is
 * non-volatile across processes until the PL is reset), or just to measure raw
 * per-GEMV AXI latency. It also reads back y[0] so the readback path is exercised.
 *
 *   build: cc -O2 -o gemv_axi_bench gemv_axi_bench.c gemv_axi_drv.c
 *   run  : sudo ./gemv_axi_bench [M] [K] [W_BASE] [ITERS]
 *          (defaults: M=768 K=256 w_base=0 iters=1000 — the qkv layer shape)
 *
 * Prints:  GEMV_BENCH idcode=0x47454d52 M=768 K=256 iters=1000 us_per_gemv=12.3
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern int      gemv_open(unsigned long base);
extern void     gemv_close(void);
extern unsigned gemv_idcode(void);
extern long     gemv_run(int M, int K, unsigned w_base,
                         const signed char *x, int *y);

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 768;
    int K = argc > 2 ? atoi(argv[2]) : 256;
    unsigned wb = argc > 3 ? (unsigned)strtoul(argv[3], NULL, 0) : 0u;
    int iters = argc > 4 ? atoi(argv[4]) : 1000;

    if (gemv_open(0xA0000000UL) != 0) {
        fprintf(stderr, "gemv_open failed (root? bitstream loaded?)\n");
        return 1;
    }
    unsigned id = gemv_idcode();

    signed char *x = malloc((size_t)K);
    int *y = malloc((size_t)M * sizeof(int));
    for (int k = 0; k < K; ++k) x[k] = (signed char)((k % 7) - 3);  /* small INT8 */

    long last_cyc = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i)
        last_cyc = gemv_run(M, K, wb, x, y);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double us = secs / iters * 1e6;
    printf("GEMV_BENCH idcode=0x%08x M=%d K=%d iters=%d us_per_gemv=%.2f "
           "last_cycles=%ld y0=%d\n", id, M, K, iters, us, last_cyc, y[0]);

    free(x); free(y);
    gemv_close();
    return 0;
}
