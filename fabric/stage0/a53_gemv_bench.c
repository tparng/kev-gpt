/* a53_gemv_bench — Stage 0 baseline: the bandwidth wall, measured on the A53s.
 *
 * Doc 2's stage 0: run the model's dominant work (streaming the whole weight set
 * once per decode token through INT4xINT8 MACs) on the Arm cores and show it is
 * memory-bandwidth bound, not compute bound. That measured "before" number is
 * what the on-chip fabric (stage 1+) is compared against.
 *
 * This streams the exact packed-INT4 image the fabric uses (fabric/export ->
 * weights.bin via make_blob.py), so both paths move identical bytes. Per decode
 * step it touches all weight bytes once; tok/s and effective GB/s fall straight
 * out. On the KV260 the A53s and PL share one ~20 GB/s DDR controller, so when
 * the 1.5 MB model spills L2 this is DDR-bound and the GB/s number sits near
 * that wall — the whole thesis in one measurement.
 *
 * Portable C99 (stdint/stdio/time). Build + run ON THE BOARD (or any host):
 *   gcc -O3 -march=native a53_gemv_bench.c -o a53_gemv_bench
 *   ./a53_gemv_bench fabric/export/weights.bin 256 2000
 * Confirm bandwidth-bound with: perf stat -d ./a53_gemv_bench ...
 */
#define _POSIX_C_SOURCE 199309L   /* expose clock_gettime/CLOCK_MONOTONIC on glibc */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Wall-clock seconds. Uses the monotonic clock on the board (A53 Linux/glibc);
 * falls back to clock() where CLOCK_MONOTONIC isn't exposed (e.g. a bare-metal
 * syntax-check toolchain), so this compiles everywhere and times correctly on
 * the target. */
static double now_sec(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

/* signed 4-bit two's-complement nibble -> int */
static inline int nib_s(unsigned v) { v &= 0xF; return (v >= 8) ? (int)v - 16 : (int)v; }

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s weights.bin [K=256] [iters=2000]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    int K     = (argc > 2) ? atoi(argv[2]) : 256;     /* reduction length / row */
    long iters = (argc > 3) ? atol(argv[3]) : 2000;   /* decode steps to time   */
    if (K <= 0 || K % 2) { fprintf(stderr, "K must be positive and even\n"); return 2; }

    /* load the packed-INT4 weight blob */
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END); long nbytes = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *w = (uint8_t *)malloc((size_t)nbytes);
    if (!w || fread(w, 1, (size_t)nbytes, f) != (size_t)nbytes) {
        fprintf(stderr, "read failed\n"); return 1;
    }
    fclose(f);

    const int KB = K / 2;                 /* packed bytes per row */
    long rows = nbytes / KB;              /* total output rows across the model */

    /* a fixed pseudo-random INT8 activation vector, reused each row */
    int8_t *x = (int8_t *)malloc((size_t)K);
    uint32_t s = 0x12345678u;
    for (int k = 0; k < K; k++) { s = s * 1103515245u + 12345u; x[k] = (int8_t)(s >> 16); }

    volatile int64_t sink = 0;            /* defeat dead-code elimination */
    double t0 = now_sec();

    for (long it = 0; it < iters; it++) {
        int64_t acc_all = 0;
        const uint8_t *p = w;
        for (long r = 0; r < rows; r++) {
            int32_t acc = 0;
            for (int j = 0; j < KB; j++) {
                uint8_t b = p[j];
                acc += nib_s(b)        * x[2*j];       /* low nibble  = even k */
                acc += nib_s(b >> 4)   * x[2*j + 1];   /* high nibble = odd  k */
            }
            acc_all += acc;
            p += KB;
        }
        sink += acc_all;
    }

    double sec = now_sec() - t0;
    double toks = iters / sec;
    double gbps = (double)nbytes * iters / sec / 1e9;

    printf("model_bytes=%ld rows=%ld K=%d iters=%ld sec=%.3f\n", nbytes, rows, K, iters, sec);
    printf("A53_BASELINE tok_s=%.1f eff_GBps=%.2f (sink=%lld)\n",
           toks, gbps, (long long)sink);
    free(w); free(x);
    return 0;
}
