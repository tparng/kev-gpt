# Plan — the P-wide vector datapath (the road to ~10k tok/s)

Status anchor (all MEASURED on a KV260, token-stream bit-exact unless noted):
- baseline sequencer (PE=16, per-matmul reload): **44.3 tok/s** @ 40 MHz, 902,528 cyc/tok
- resident-read + PE=256: **231 tok/s** @ 40 MHz, 173,151 cyc/tok
- + streaming GELU + pipelined-LN-Newton (this branch): **~281 tok/s** (sim, cyc ~142k) and a
  higher Fmax ceiling (LN cascade broken) — see `BOARD-TEST.md` once the rebuild lands.

This doc is the design for the *next* architecture: make the whole datapath **P-wide**, the
way the GEMV already is. It is the single-stream path to ~10k. (100k is a separate, batched
story with a hard on-chip-KV ceiling — see the honest note at the end.)

## Why P-wide — the measured reason

At PE=256 the GEMV MACs are no longer the bottleneck. The **measured** per-token budget
(`run_sequencer --fast --lanes 256 --dbg-phase`, before this branch's GELU/LN work):

| phase | cyc/token | share | nature |
|---|---:|---:|---|
| gemv (act-feed + readback + run) | 55,545 | 32.1% | act-feed & readback are **1 elem/cyc**; run is wide |
| gelu | 38,400 | 22.2% | per-element stall (fixed on this branch → 7,710) |
| attention | 37,890 | 21.9% | KV dot-products **1 elem/cyc**, grows with context |
| layernorm | 17,431 | 10.1% | feed/collect **1 elem/cyc** (+ compute latency) |
| dequant | 17,280 | 10.0% | **1 elem/cyc** |
| residual/pre, embed, emit | ~9,100 | 5.5% | **1 elem/cyc** |

Everything except the GEMV *run* (~12k/tok) and the LN compute is a **1-element/cycle loop
over a D=256 or D_MLP=1024 vector**. Processing P of those elements per cycle collapses the
whole serial floor by ~P. That is the entire idea.

## The target arithmetic

Two independent levers, multiply together:
- **Cycles:** widen every 1-elem/cyc loop to **P lanes** (P = 16–32). Serial floor ÷ P.
- **Clock:** pipeline the deep combinational arith so Fmax rises from ~50 MHz toward ~150–200 MHz.

Estimate at **P = 16, pipelined to ~200 MHz** (per-token, from the 173k budget):

| phase | now | P=16 | note |
|---|---:|---:|---|
| gemv run (already wide) | ~12,000 | ~12,000 | unchanged (MAC array) |
| gemv act-feed + readback | ~43,000 | ~2,700 | quantize/read P/cyc |
| attention | ~38,000 | ~2,400 | P-wide KV dot-products |
| layernorm | ~17,000 | ~1,100 | `layernorm_par.sv` (already P-wide) |
| dequant | ~17,000 | ~1,100 | P parallel dequant units |
| gelu | ~7,700 | ~500 | P parallel `gelu_lut` |
| resid/pre/embed/emit | ~9,000 | ~600 | P-wide adds |
| **total** | **~143,000** | **~20,400** | |

~20k cyc/tok @ 40 MHz ≈ 2,000 tok/s; **@ 200 MHz ≈ 10,000 tok/s.** That is the plan.

## The work, phase by phase

Each item keeps the **token-stream bit-exact** gate green (values unchanged; only lane-width
and pipeline depth change) and is independently simulatable.

1. **Vector scratch banking.** `xres`, `lnout`, `mlpbuf`, `qvec`, `ctxv` become **P-banked**
   (P parallel memories, element `e` lives in bank `e mod P` at row `e div P`), so a whole
   P-slice reads/writes in one cycle. This is the enabler for every phase below — do it first.
2. **P-wide dequant.** Replicate the dequant unit (INT32 × 24-bit mant, shift) P×. The
   per-channel `dq_mant`/`dq_exp` ROMs are read P-at-a-time → bank them P-ways (or replicate).
3. **P-wide act-quant + GEMV feed.** Quantize P activations/cyc (mult by `inv_sact`, round/sat)
   and feed the GEMV P/cyc. The GEMV already consumes one x[k] shared across its 256 output
   lanes; feeding it P columns/cyc needs P x-write ports or a P-deep x-FIFO. **Readback**:
   the GEMV `ymem` is grouped by output row; read P outputs/cyc (it is already 256-wide
   internally — expose a P-wide read).
4. **Integrate `layernorm_par.sv`.** It already does the two-pass mean/var + normalize P-wide;
   wire it in place of the serial `layernorm`. Re-pin its Q-format to the committed one and
   gate against `run_layernorm` + the sequencer token stream.
5. **P-wide GELU.** P parallel `gelu_lut` instances (each is a small BRAM + interp); drive P
   `gl_x`/cyc, capture P `gl_y` with the same 4-deep shift (already streaming on this branch).
6. **P-wide attention.** The score (q·k) and context (Σ p·v) reductions go P-wide: bank the
   KV caches P-ways so P elements of a key/value read per cycle, with a P-input adder tree for
   the dot-product. This is the phase that **grows with context length**, so it matters most
   for real prompts — prioritise it after the easy wins.
7. **Pipeline for clock.** Break the remaining deep combinational paths into stages (done:
   LN Newton; next: the 64×64 act-quant multiply, the 96-bit dequant shift, the LN `S_OUT`
   double-multiply). Target ≥150 MHz, then push toward 200. OOC-sweep `sequencer_fast` after
   each to find the new limiter before committing a full build.

## The resource tension (the honest hard part)

The PE=256 GEMV alone already routed at **87% LUT** (102k/117k). Adding P=16 copies of dequant
/ act-quant / GELU / the attention adder tree **will not fit** on top of PE=256. The realistic
config is a **trade between GEMV width and serial width**:

- Drop the GEMV to **PE=64 or PE=128** (the run phase is only ~12k/tok at PE=256; at PE=64 it
  is ~48k/tok — still small vs the serial floor we are cutting). That frees a large LUT budget.
- Spend it on **P=16–32 serial lanes** + the pipeline registers.

Rough budget target: PE≈128 GEMV (~50% LUT) + P=16 serial datapath (~25% LUT) + pipeline
regs, fitting <90% LUT, 60/64 URAM (the resident weight image is fixed at ~60 regardless of
PE — it is the model size, not the lane count). DSP: the dequant/act-quant/LN multiplies P×
will push DSP usage up (currently 130/1248 — lots of headroom). This balance is an OOC-sweep
exercise, not a guess; **measure PE × P combinations OOC for LUT/DSP/URAM/Fmax before building.**

## Suggested order (each a gated increment)

1. Vector scratch banking + P-wide dequant + P-wide GELU (the easy ÷P wins, ~2× cycles).
2. Integrate `layernorm_par`; P-wide act-quant/feed/readback.
3. P-wide attention (the context-scaling win).
4. Pipeline the act-quant / dequant / LN-out paths; OOC-sweep PE×P for the best Fmax-fit.
5. Build the chosen PE×P @ its closed clock; board-measure; flip the rung MEASURED.

Expect this to land in the **5,000–12,000 tok/s** band depending on the PE×P×Fmax point the
board actually closes — report it as the measured result, not the model.

## Why 100k is not on this line

100k single-stream is latency-bound: a 4-layer autoregressive forward cannot compress to the
~2,000 cyc/token (≈10 µs) it would require even at 200 MHz. 100k can only come from **batched
aggregate** throughput (B streams overlapped through one pipeline). But each stream's KV cache
is ~512 KB and only ~1–1.2 MB of on-chip memory is free after the 1.5 MB resident model →
**B ≈ 2 fits on-chip**, i.e. ~2× the single-stream peak, not 50×. Real 100k needs a much
shorter context, a smaller model, KV spill to DDR (which breaks the bandwidth thesis), or
multiple boards. For this model on one KV260 the honest aggregate ceiling is **~20k**. That is
a crossover the project states out loud, not a bottleneck to be engineered away.
