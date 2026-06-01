# Stage 3 — the fabric throughput core + the CPU-out-of-the-loop dataflow

Stage 3 is the build that takes Kevin from ~100 tok/s (CPU in the per-token loop) to
the **10k–100k** band: the whole forward as a hardware dataflow, the A53 only writing
the prompt and draining tokens. This directory holds the throughput core (built,
synth-proven) and the verified reference designs the rest of the RTL must match.

Project rule throughout: **bit-honest before fast** — every block is gated against a
reference (bit-exact, or cosine > 0.9999 for the transcendentals) before any speed
number is trusted.

## What's built and verified

### 1. The throughput core — `rtl/gemv_banked.sv` (RTL, synth-proven)
The transposed *wider-word banked* GEMV that breaks the PE=64 one-bank-per-lane
ceiling. Weights are stored so one wide URAM word holds the **same column k of LANES
consecutive output rows**, so a single read feeds all LANES lanes sharing the
activation `x[k]` — LANES MACs/cycle, each lane accumulating its own rows (no
reduction tree). A 64-bit word feeds 16 lanes/URAM; PE=256 spans 16 URAM in parallel.

- **Bit-exact** (`run_banked.py`, iverilog): `BANKED_VERDICT bitexact=True
  mismatches=0` across PE = 16…256, every real layer shape (qkv/proj/mlp/head),
  including padded M.
- **OOC synth on `xck26-2LV`** (`tcl/synth_banked.tcl`, constant ~1.5 MB footprint):

  | PE (LANES) | Fmax | LUT | DSP | URAM |
  |---|---|---|---|---|
  | 16  | 198 MHz | 1.4% | 0 | 48 |
  | 64  | 229 MHz | 5.0% | 0 | 48 |
  | **256** | **293 MHz** | **22.7%** | **0** | **45** |

  **Fmax increases with width** — narrow lanes mean a *deeper* URAM cascade (long
  read path); wide words are shallow and fast. 0 DSP (INT4×INT8 fits in LUTs),
  ~104 LUT/lane. Extrapolation: **PE=1152 fits the 64 URAM** (4608-bit word = 64
  blocks wide), MACs split across the 117k LUT + the **1248 idle DSPs** →
  **~90k tok/s; ~100k a small push** (light DSP packing or a clock nudge).

### 2. KV cache — `model/goformer_kv.py` (reference, bit-exact)
Incremental decode (process only the new position, cache per-layer K/V). The
prerequisite for 10k+ — the old forward recomputes the whole context (O(T²), 256× the
work at T=256). Bit-IDENTICAL to full-recompute by causality:
`GOFORMER_KV_PASS` — maxabsdiff = 0 over every prefix, greedy stream identical.

### 3. Fabric non-linears — `model/goformer_fixed.py` (reference, cosine 1.0)
softmax / LayerNorm / GELU / dequant implemented the way the fabric will (LUTs,
fixed-point, Newton steps), with the precision **measured** against the real exported
model — the *precision crossover*, not assumed:

| Op | Requirement | HW cost |
|---|---|---|
| LayerNorm (γ-only, eps 1e-5) | 8-bit-seed rsqrt + 2 Newton → exact | trivial |
| Softmax | exp LUT **Q1.20**, clamp z ∈ **[−16, 0]** | small LUT |
| GELU | **8192-entry** LUT [−8,8] lin-interp (interp leaves a *coherent* bias that accumulates) | ~4 BRAM |
| Dequant | **24-bit** per-channel folded scale | 24-bit mult |

Result: `GOFORMER_FIXED_PASS` — cosine 1.0000000, same argmax everywhere.

### 4. Integrated sequencer reference — `model/goformer_seq.py` (the contract)
Composes (2) + (3) into the exact per-token dataflow the FSM runs and gates the
strongest contract: the fixed-point KV-cached streaming decoder emits the **same
token stream** as the float reference — `GOFORMER_SEQ_PASS`, 48/48 identical,
worst per-step cosine 0.99995. This is the golden stream the RTL sequencer is
validated against.

## Build order — what remains (the FPGA realization)

The references above are the bit-honest specs; these turn them into timing-closed RTL:

1. **Pin the fixed-point datapath Q-format.** The reference carries the residual
   stream in float; the RTL needs a fixed format. Profile the residual range, pick the
   smallest Q(I).(F) that holds cosine > 0.9999 as a fully-integer datapath.
2. **RTL for the three non-linears** (GELU-LUT, LayerNorm, softmax), each iverilog-gated
   bit-true vs `goformer_fixed` in the pinned format.
3. **The sequencer FSM** — embed → 4×[LN, qkv, attn(+KV), proj, +res, LN, mlp(GELU),
   +res] → LN_f → head → LFSR-sample → append-KV → loop, zero CPU between tokens.
   Prompt-id input FIFO, token-id output FIFO. Wire to `gemv_banked`; timing-close.

## Files

```
rtl/gemv_banked.sv      transposed banked GEMV throughput core
pack_banked.py          transposed packer + golden + file-based compare
tb/tb_gemv_banked.sv    testbench (dumps y.out for the Python gate)
run_banked.py           iverilog sim driver  ->  BANKED_VERDICT
tcl/synth_banked.tcl    OOC Fmax/util sweep on xck26-2LV
```
Reference designs live in `model/goformer_{kv,fixed,seq}.py`. Sim/synth scratch goes
to `C:/kevbuild` (outside the repo; OneDrive's cloud-files filter locks project dirs).
