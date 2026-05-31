# Stage 0 — the A53 baseline and the bandwidth wall

Doc 2's stage 0: run the model's dominant work on the board's Arm cores, measure
tokens/sec and confirm with a profiler that it is **memory-bandwidth bound, not
compute bound**. This is the honest "before" number the whole project is measured
against — and the proof that the only way to win is to leave DDR (stages 1–3).

**Status: portable C reference written and compiles clean (`-Wall`). Runs on the
board (or any Linux host); no board here, so the number below is the roofline
*prediction* to confirm against once it runs on the KV260.**

## The microbenchmark

`a53_gemv_bench.c` streams the whole packed-INT4 weight image once per "decode
step" doing INT4×INT8 MACs — the bandwidth-dominant part of single-stream decode.
It moves the *same bytes* the fabric holds on-chip (`fabric/export/weights.bin`,
built by `make_blob.py`), so stage 0 and stage 1+ are compared on identical data.

```
python -m model.export_fabric data/ckpt.qat.pt   # produces fabric/export/*.w.mem
python -m fabric.stage0.make_blob                 # -> fabric/export/weights.bin (1.5 MB)

# on the board (or any gcc host):
cd fabric/stage0 && make run
#   -> A53_BASELINE tok_s=... eff_GBps=...
```

Confirm it's bandwidth-bound (the load-bearing claim):

```
perf stat -d ./a53_gemv_bench ../export/weights.bin 256 2000
# look for: high LLC-load-misses, IPC well under the core's peak, and
# eff_GBps sitting near the DDR ceiling rather than the core's MAC ceiling.
```

## What to expect — the roofline prediction (from fabric/roofline.py)

At the deployable size (3.15M linear params, **1.5 MB INT4**):

| ceiling | predicted tok/s | why |
|---|---|---|
| DDR bandwidth (20 GB/s shared) | ~12,700 | 20e9 / 1.5 MB per token |
| A53 INT8 compute (~75 GOPS pk) | ~11,900 | 75e9 / (2·params) |
| **realistic A53** | **min of the two, lower in practice** | sustained << peak |

The point: both ceilings are low and close, and the 1.5 MB model spills the A53
L2, so the measured `eff_GBps` should land near ~20 GB/s — DDR-bound. The fabric's
escape is keeping those same 1.5 MB in BRAM/URAM at hundreds of GB/s (stage 1's
measured Fmax × on-chip width feeds the real number).

## Honest scope / next

- This isolates the **weight-streaming bandwidth** — the dominant decode cost and
  the cleanest way to show the wall. The complete stage-0 deliverable doc 2
  describes is the *full* model on the A53 (goformer-on-ARM or a tuned
  `llama.cpp`), which also exercises attention/softmax/norm; this microbench is
  the bandwidth core of that and the quickest honest "before" number.
- Pair the run with **board-level power** (`A53 vs fabric tok/joule`) for doc 3's
  metrics table.
- The crossover plot (`fabric/roofline.py --plot`) turns this measured baseline
  into the model-size axis where the fabric stops winning.
