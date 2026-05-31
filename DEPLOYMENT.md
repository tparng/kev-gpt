# Deployment plan — Kria + full corpus

Operational runbook for the day the Kevinised corpus lands and the KV260 is
connected. Two tracks run in parallel: **A** (corpus → trained INT4 model →
re-export, on the RTX, no board) and **B** (Kria bring-up). They only converge at
the very end, so start both.

Conventions from the project hold: **bit-honest before fast** — every gate below
must pass before any speed number is quoted; honest-first on where it loses.

---

## 0. Status going in (what's already verified)

All on `main`, pushed. Built and checked **without the board**:

- Software→fabric bridge: `fabric/spec.py` (contract), `fabric/pack.py` (INT4
  packing, 12/12 tests), `model/export_fabric.py` (QAT ckpt → packed `.mem` +
  per-channel scales + manifest), `fabric/golden.py` (numpy integer "goformer").
- `fabric/roofline.py`: 3.15M params = **1.5 MB INT4, fits the ~3 MB budget**;
  crossover **~6.3M params**; 10–50× decode band over the 20 GB/s DDR wall.
- `fabric/stage1/` SystemVerilog GEMV: **bit-exact vs golden, all 17 layers,
  `maxabserr=0`** (iverilog).
- `model/validate_goformer.py`: exported int datapath vs Brevitas QAT forward,
  **worst cosine 1.0000000** (rel err ~1e-7).
- `fabric/stage0/` A53 baseline: C compiles clean; runs on the board.

**Caveats to fix tomorrow (known, not hidden):**
1. Current checkpoints (`data/ckpt.fp.pt`, `data/ckpt.qat.pt`) were trained on the
   **2 MB smoke corpus** — placeholders. Replace with the full-corpus model.
2. Vivado OOC synth closes timing at 300 MHz but **utilization is degenerate
   (56 LUT / 0 DSP / 0 URAM)** — the datapath was pruned because `y` is an
   internal reg with no observable output. **Real numbers need the output
   exposed** (see §B2).

---

## Track A — full corpus → trained INT4 model (RTX 3050 Ti, no board)

Trigger: the Mac's `TinyStories-train.kevin.txt` (~1.3 GB) is ready.

### A1. Land the corpus
```
# copy TinyStories-train.kevin.txt into data/ (scp/USB/cloud)
ls -la data/TinyStories-train.kevin.txt          # sanity: ~1.3 GB
# (optional) confirm the compression headline travelled with it:
cat data/tinystories-train.stats.json
```

### A2. Tokenise
```
python -m model.data data/TinyStories-train.kevin.txt data/char
# note the printed vocab_size + train_chars; vocab flows into the ckpt cfg and
# through export, so nothing downstream is hand-set.
```

### A3. Train FP (the coherent model first)
```
python -m model.train --device cuda --compile \
    --corpus data/TinyStories-train.kevin.txt \
    --max-iters 20000 --eval-interval 1000 \
    --batch-size 96 --n-embd 256 --n-layer 4 --out data/ckpt.pt
# watch it climb:
python -m model.evolution data/states.jsonl -o data/evolution.md
python -m model.sample data/ckpt.pt --prompt "once upon time"
```
**Gate A3:** sample reads as coherent telegraphic Kevin, val curve flat/declining
(use `data/ckpts/` snapshots to roll back if it overfits). ~full corpus is ~100×
the smoke data, so 20k iters won't overfit at 4k the way the smoke run would.

### A4. INT4 QAT fine-tune (warm-started off the FP model)
```
python -m model.train --qat --init-from data/ckpt.pt \
    --corpus data/TinyStories-train.kevin.txt \
    --max-iters 4000 --eval-interval 500 --out data/ckpt.qat.pt
```
**Gate A4:** QAT val sits within a few % of the FP val (it inherits, doesn't
restart). QAT throughput is ~15–20× slower than FP (fake-quant) — ~70 min for 4k
iters is expected, not a bug.

### A5. Re-export + re-validate (same commands, new weights)
```
python -m model.export_fabric data/ckpt.qat.pt        # -> fabric/export/*.mem + manifest
python -m fabric.golden --all                          # -> per-layer TB vectors
python -m model.validate_goformer                      # cosine gate
python -m fabric.stage0.make_blob                      # -> fabric/export/weights.bin
python -m fabric.roofline --plot fabric/roofline.png   # refresh the crossover plot
```
**Gate A5 (the bit-honest gate):** `GOFORMER_VALIDATE_PASS`, worst cosine > 0.9999.
Then re-run the RTL sim on the new weights:
```
bash fabric/stage1/sim/run_iverilog.sh blocks_0_proj 16
# -> PYVERDICT bitexact=True maxabserr=0
```
After A5 the full deployable image (`fabric/export/`) is the real model, not the
smoke one. **This is the artifact the board work consumes.**

---

## Track B — Kria KV260 bring-up

Trigger: board powered, on the network. Can start before Track A finishes (uses
the smoke export until A5 lands; swap in the real one when ready).

### B1. Stage 0 — the A53 baseline + the bandwidth-wall proof
On the board (serial console or SSH):
```
# get the repo + weights.bin onto the board (git clone + scp weights.bin,
# or scp the whole fabric/export/)
cd fabric/stage0 && make run
#   -> A53_BASELINE tok_s=... eff_GBps=...
perf stat -d ./a53_gemv_bench ../export/weights.bin 256 2000
```
**Gate B1:** record `tok_s` and `eff_GBps`; confirm **bandwidth-bound** — high
LLC-load-misses, IPC well under peak, `eff_GBps` near ~20 GB/s (the shared DDR
wall), *not* near the core's MAC ceiling. Compare to the roofline prediction
(~12k tok/s ceiling at 1.5 MB). This is the honest "before" number everything is
measured against; capture **board power** alongside it for tok/joule.

### B2. Real synth numbers — fix the output pruning first
The GEMV is functionally proven; we just need a meaningful utilization/Fmax. The
OOC synth pruned the datapath because `y` is unobservable. **Smallest fix:**

- In `fabric/stage1/rtl/gemv_int4.sv`, add a read-out port so the result array is
  observable (forces synthesis to keep the datapath + infer the URAM/DSP):
  ```systemverilog
  input  wire [$clog2(M)-1:0] rd_addr,
  output reg  signed [31:0]   y_out
  // ... in the always block:
  y_out <= y[rd_addr];
  ```
- Point the synth at a **real weight image** so the ROM infers as URAM rather than
  being optimised to constants — pass `WFILE` to a `.w.mem` in `tcl/synth.tcl`
  (`-generic WFILE=...`) or add `(* dont_touch = "true" *)` to `wmem`/`y`.
- Re-run:
  ```
  vivado -mode batch -source fabric/stage1/tcl/synth.tcl -tclargs 256 256 16
  # reports -> fabric/stage1/synth/{util,timing}_256x256_pe16.rpt
  ```
**Gate B2:** util report shows real **DSP / URAM / LUT / FF**, timing closes at
300 MHz (stretch: push the clock to find Fmax). Feed measured on-chip
bandwidth (URAM width × Fmax) back into `fabric/roofline.py` to replace the
assumed 200–1000 GB/s band with a measured number.

> Note: iverilog rejected the unpacked-array output port, which is why `y` was
> made internal. Keep the iverilog sim using the hierarchical `dut.y[m]` read;
> the new `y_out` port is for synthesis/observability and the TB can ignore it.
> If iverilog still complains, simulate this variant with **xsim** (also
> installed) and keep iverilog for the array-internal version.

### B3. (stretch) toward Stage 2/3
Only after B1+B2 and a real model (A5):
- **Stage 2:** PL matmul from on-chip weights + A53 softmax/RMSNorm over AXI;
  measure the ping-pong tax (µs/round-trip, DDR bytes). The honest "here's what
  the easy heterogeneous split costs" number.
- **Stage 3:** fabric-native LUT-softmax (running-max) + RMSNorm + GELU next to
  the GEMV; the zero-DRAM headline + roofline crossover plot.
- **Throughput:** bank the weight memory per lane so PE_LANES weights land per
  cycle (the current core proves correctness; banked + pipelined is the speed
  follow-up).

---

## Verification gates, in one place (don't skip)

| Gate | Command | Must see |
|---|---|---|
| Tokenise | `model.data` | vocab_size + train_chars printed |
| FP coherent | `model.sample data/ckpt.pt` | readable telegraphic Kevin |
| QAT inherits | `model.train --qat` | val ≈ FP val |
| **Bit-honest** | `model.validate_goformer` | `GOFORMER_VALIDATE_PASS`, cosine > 0.9999 |
| RTL == golden | `run_iverilog.sh <layer> 16` | `bitexact=True maxabserr=0` |
| A53 baseline | `make run` + `perf stat -d` | tok/s + bandwidth-bound confirmed |
| Real synth | `synth.tcl` (after B2 fix) | non-degenerate DSP/URAM, 300 MHz closes |

---

## Gotchas / hard-won lessons

- **Layer keys are module names:** `blocks_0_proj` (not `attn_proj`), `qkv`,
  `mlp_fc`, `mlp_proj`, `head`. The sim defaults to `blocks_0_proj`.
- **Verify RTL via the file-based Python compare**, never by reading numbers off
  the console (the TB `$writememh`s and `run_iverilog.sh` diffs in Python).
- **Toolchain:** Vivado 2025.2 (no `vitis_hls` → hand-RTL path is correct);
  iverilog + xsim for sim; **no local C compiler** — Stage 0 C only builds on the
  board / a Linux host. Part: `xck26-sfvc784-2LV-c`.
- **Regenerables are gitignored:** `fabric/export/`, `*.png`, sim/synth scratch.
  Re-run export + golden after any retrain; don't expect them in git.
- **On-chip budget is a hard 3 MB ceiling.** Current model = 1.5 MB INT4, room to
  spare. If you trade up context/vocab, re-check `fabric/roofline.py` for where it
  crosses back over the wall before committing.

---

## Order of the day

1. **First thing:** kick off Track A as soon as the corpus arrives (A2→A4 is the
   long pole — hours of GPU time; start it, then do board work while it runs).
2. **In parallel:** Track B1 (Stage 0 baseline) on the board with the smoke
   `weights.bin` — the bandwidth-bound proof doesn't need the final model.
3. **Then:** B2 synth fix for real PL numbers.
4. **When A5 lands:** swap the real `fabric/export/` onto the board, re-run B1
   with the real model, re-run the RTL sim gate.
5. **Capture for the writeup (doc 3/4):** A53 tok/s + tok/joule, the Keviniser
   compression ratio, on-chip headroom bought, the measured synth Fmax/util, and
   the roofline crossover refreshed with measured bandwidth.
