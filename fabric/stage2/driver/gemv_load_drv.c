// gemv_load_drv — drive the loadable GEMV engine (gemv_axi_load) on the KV260.
// Streams ONE layer's weights+activation over AXI, sets runtime M/K, runs, reads
// y back, and checks bit-exact vs the layer's golden. A shell loop calls this for
// all 17 layers from a single bitstream (weights re-streamed per layer).
//
//   sudo ./gemv_load_drv <base> <M> <K> <w.mem> <x.mem> <y.mem> [name]
//
// Registers (gemv_axi_load): 00 CTRL(b0 start,b1 ld_rst,b2 reset) 04 STATUS(b0 done)
//   08 M_COUNT 0C K_COUNT 10 W_DATA 14 X_DATA 18 RD_ADDR 1C RD_DATA 20 CYCLES 24 IDCODE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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
#define CYCC 0x20
#define IDC  0x24
#define ID_EXPECT 0x47454D4Cu  /* "GEML" */

static volatile uint32_t *R;
static inline void wr(int o, uint32_t v) { R[o/4] = v; }
static inline uint32_t rd(int o) { return R[o/4]; }

/* read up to n hex tokens (one per line) from a .mem file into out[] */
static int read_mem(const char *path, long *out, int n) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    char line[64]; int i = 0;
    while (i < n && fgets(line, sizeof line, f))
        if (line[0] && line[0] != '\n' && line[0] != '/' && line[0] != '@')
            out[i++] = strtol(line, NULL, 16);
    fclose(f);
    return i;
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s base M K w x y [name]\n", argv[0]); return 1; }
    unsigned long base = strtoul(argv[1], NULL, 0);
    int M = atoi(argv[2]), K = atoi(argv[3]);
    const char *wf = argv[4], *xf = argv[5], *yf = argv[6];
    const char *name = argc > 7 ? argv[7] : "layer";
    int nw = M * K / 2;

    long *w = malloc(sizeof(long) * nw), *x = malloc(sizeof(long) * K), *g = malloc(sizeof(long) * M);
    if (read_mem(wf, w, nw) != nw) { fprintf(stderr, "%s: weight count\n", name); return 2; }
    if (read_mem(xf, x, K)  != K)  { fprintf(stderr, "%s: activation count\n", name); return 2; }
    int ng = read_mem(yf, g, M);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem (sudo)"); return 1; }
    R = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
    if (R == MAP_FAILED) { perror("mmap"); return 1; }

    if (rd(IDC) != ID_EXPECT) { fprintf(stderr, "bad IDCODE 0x%08X\n", rd(IDC)); return 3; }

    wr(CTRL, 0x4); usleep(50); wr(CTRL, 0x0);          // soft reset
    wr(MCNT, M); wr(KCNT, K);
    wr(CTRL, 0x2);                                      // ld_rst (rewind load ptrs)
    for (int i = 0; i < nw; i++) wr(WDAT, (uint32_t)(w[i] & 0xff));
    for (int i = 0; i < K;  i++) wr(XDAT, (uint32_t)(x[i] & 0xff));
    wr(CTRL, 0x1);                                      // start

    long spins = 0;
    while (!(rd(STAT) & 0x1)) if (++spins > 100000000L) { fprintf(stderr, "%s timeout\n", name); return 4; }
    uint32_t cyc = rd(CYCC);

    int mism = 0;
    for (int m = 0; m < M; m++) {
        wr(RADD, m);
        int32_t y = (int32_t)rd(RDAT);
        int32_t gold = (int32_t)(uint32_t)g[m];
        if (m < ng && y != gold) { if (mism < 3) printf("  %s m=%d hw=%d gold=%d\n", name, m, y, gold); mism++; }
    }
    int ok = (mism == 0 && ng == M);
    printf("PL_LOAD %-18s M=%4d K=%4d cycles=%7u bitexact=%s mism=%d/%d\n",
           name, M, K, cyc, ok ? "True" : "False", mism, M);
    return ok ? 0 : 5;
}
