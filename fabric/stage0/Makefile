# Build the A53 baseline benchmark. Run ON THE BOARD (native aarch64) or any
# Linux host. -O3 -march=native lets the compiler use the A53's NEON for the
# INT8 MACs, so the baseline is honest (a tuned CPU run, not a strawman).
CC      ?= cc
CFLAGS  ?= -O3 -march=native -std=c99 -Wall

a53_gemv_bench: a53_gemv_bench.c
	$(CC) $(CFLAGS) $< -o $@

.PHONY: run clean
run: a53_gemv_bench
	./a53_gemv_bench ../export/weights.bin 256 2000

clean:
	rm -f a53_gemv_bench
