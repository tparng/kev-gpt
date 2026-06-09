# 100K-REVIEW — code review + full-history audit of the road to 100,000 tok/s

*2026-06-09. Three parallel audits (cycle ledger, clock/fit history, model-scaling arithmetic)
plus an independent re-derivation of the scaling law against the MEASURED record. This is the
honest synthesis; it corrects two errors the individual audits made.*

## TL;DR

**100k tok/s is reachable on this chip at 200 MHz — with zero clock risk — by shrinking the
model, and by no other route.** Every datapath/clock/parallelism lever is now measured to a
wall. The verified cycle law says an L3 d256 ff512 (1.70M param) or L2 d256 ff1024 (1.70M
param) goformer lands at **~98–122k tok/s PROJECTED @200 MHz**, keeping d=256 so the existing
LN/attention/softmax/AQ datapaths are untouched — the cheapest possible RTL delta (loop bounds
and ROM sizes only). The thesis closes on itself: the last lever is a dumber Kevin.

## 1. The verified scaling law (this is the review's load-bearing result)

The GE engine's RUN phase decomposes **bit-exactly** as a sum over GEMV calls of
`rows × ceil(K / LANES)` with LANES=128:

```
per layer: QKV 768r×2b + proj 256r×2b + fc 1024r×2b + mlp_proj 256r×8b = 6,144 cyc
RUN = 4 × 6,144 + head 256r×2b = 25,088  ✓ (matches the doc-6 DERIVED number exactly)
```

Readback reconciles exactly too: `RB = rows × 35/32` → 9,472 × 35/32 = **10,360** ✓.
The remaining 17,916 cyc of the MEASURED 53,364 (= 16 × 200e6 / 59,965.5) is the serial
non-linear chain + FSM idle. So the full cycle model reproduces the silicon record to the
cycle, and projections from it are trustworthy on RUN/RB (serial scaling is the soft part —
tagged below).

**Consequence the scaling audit initially missed: the K dimension is quantized to 128-lane
beats.** `ceil(192/128) = ceil(256/128) = 2`, so a d=192 model pays the same MAC beats per row
as d=256 — 25% of the lanes idle. **d and d_ff must stay multiples of 128** or the shrink is
partially wasted. (The first-pass recommendation of "L4 d192 → 118.8k" was wrong for exactly
this reason; the corrected number is ~81–87k.)

## 2. The dead-lever ledger (do not re-propose these)

| lever | verdict | evidence |
|---|---|---|
| Streams > 16 | **DEAD** — 2.0 MAC/DSP packing wall, 1,187/1,248 DSPs used | doc 6 §18, `dsp3_pack_proof.py` |
| LANES=256 | **DEAD** — needs 2,048 DSPs, part has 1,248 | doc 6 §18 |
| 250 MHz (current design) | **DEAD** — STA closes (5 ns MET +0.074) but route fails at 98–99% LUT density, ~2.9 ns pure routing on the worst path; 4 builds | doc 6 §14–31 |
| Double-pump MAC @clk2x | **DEAD** — bit-exact on silicon but the fabric→clk2x data feed walls at 50 MHz; full N=14 integration violates −3.3 ns at 99% CLB density. `tok/s = 14×clk/41,753`; 80k needs 238.6 MHz > the 200 wall | `DOUBLE-PUMP-100K.md`, blog 14, branch `dp-hw-maconly` |
| P=16 readback/dequant | **DEAD** — LUT budget bust (127.6k LUT at P=8/L=128 already) | doc 6 §139, WIDE-WORD log §4 |
| LN→AQ / attn→PROJ schedule overlap | **DEAD** — lockstep GEMM consume; two agent HONEST-STOPs | doc 6 §78–87 |
| Shared-attention arbitration fixes | **DEAD** — Δ0 measured; it's unit throughput, not scheduling | doc 6 §23–24 |

One audit stacked double-pump stages to a "~94.5k best case" — **that stack is invalid**; it
assumed DP closes 400 MHz on the board, which the silicon measurement (50 MHz data-feed wall)
and the routed-timing report (WNS −4.658 on clk2x) both refute. The record build itself has no
headroom to add anything: URAM 100%, DSP 95.1%, LUT 92.9%, CLB 99.6%.

So at the current model: cycle floor ~51,100 → **~62–64k @200 MHz is the architecture's
ceiling**, exactly as doc 6 concluded. The review confirms doc 6 rather than overturning it.

## 3. The corrected candidate table (the only live road)

Cycle model: verified RUN/RB laws + serial scaled per component (attention ∝ layers·TMAX,
LN/AQ ∝ layers·d, GELU ∝ layers·ff, fixed FSM tail). Serial scaling is the weakest
assumption — but `run_vec_seq` sim gives **exact** cycles before any synth, so every row below
is cheaply falsifiable. All rows N=16, LANES=128, P=8, d kept ≡ 0 (mod 128). PROJECTED.

| config | params | RUN | RB | serial | total cyc | tok/s @200 | +gated cuts¹ @200 |
|---|---|---|---|---|---|---|---|
| **current** L4 d256 ff1024 | 3.28M | 25,088 | 10,360 | 17,916 | 53,364 | **59,966 MEASURED** | 64,069 |
| A: L4 d256 ff512 | 2.23M | 16,896 | 8,120 | ~17,378 | ~42,394 | ~75,481 | ~81,886 |
| B: L3 d256 ff1024 | 2.49M | 18,944 | 7,840 | ~13,974 | ~40,758 | ~78,511 | ~84,007 |
| **C: L3 d256 ff512** | 1.70M | 12,800 | 6,160 | ~13,616 | ~32,576 | **~98,231** | **~106,744** |
| **D: L2 d256 ff1024** | 1.70M | 12,800 | 5,320 | ~10,032 | ~28,152 | **~113,665** | **~121,961** |
| E: L4 d128 ff512 | 0.85M | 6,400 | 5,320 | ~9,495 | ~21,215 | ~150,833 | ~164,919 |
| X: L4 d192 ff768 (lane-wasting — listed to kill it) | 1.87M | 17,408 | 7,840 | ~14,332 | ~39,580 | ~80,847 | ~86,849 |

¹ TMAX=16 (−1,673) + ATT2 per-cohort attention (−1,745), both already gated bit-exact in sim
(doc 6 §25–26, `DOUBLE-PUMP-100K.md` Stage 3), scaled to the config's serial share.

**Bonus, not assumed:** every candidate sheds 31–48% of the weight URAM and a matching slice of
LUT/BRAM. The 250 MHz route died at 98–99% density — a C/D-sized design routes much looser, so
the 250 MHz shot (×1.25 on every row) comes back from the dead *as margin, not as the plan*.
The plan needs only the proven 200 MHz.

## 4. Why C and D specifically

- **d stays 256** → every d-width datapath (LayerNorm, attention, softmax, act-quant, embed,
  residual, head) is bit-for-bit untouched. The RTL delta is layer count (sequencer loop
  bound), fc/mlp_proj dimensions (C only), weight ROM sizes, and `.mem` regeneration. E and X
  would touch everything.
- **C vs D is a quality experiment, same param count (1.70M):** C keeps depth (3 layers, thin
  MLP), D keeps the wide MLP (2 layers). Nobody knows which degrades Kevin less — there is no
  NLL-vs-shape ladder in the repo. Training a config costs ~5–7 min (FP ~4 min + INT4 QAT
  warm-start ~2 min, BUILD-LOG); **train both, plus A/B for the curve, and let val loss pick.**
- Current INT4 QAT val loss is 0.804 (FP 0.780). Expect the 1.70M configs to land noticeably
  higher — the honest gate is sample quality (telegraphic but coherent), not a fixed
  threshold. If both C and D produce mush, B at ~84k is the fallback and 100k waits for the
  250 MHz margin to be proven on the looser build.

## 5. The execution plan (in gate order, bit-honest before fast)

1. **Train the ladder** — A, B, C, D FP runs + INT4 QAT warm-starts (`qgpt.py`), ~30 min total.
   Record val losses + sample text side by side.
2. **Export + gate** — `export_fabric.py` / `validate_goformer.py` per config (cosine gate).
3. **Sim cycle gate** — parameterize `seq_ref.py` + the sequencer for L/ff, run
   `run_vec_seq` on C and D: this turns every PROJECTED cycle count above into an exact
   sim-MEASURED one *before* any synth. If serial scaling was optimistic, we find out here
   for free.
4. **Synth/impl at 200 MHz** (the proven clock; expect easy closure at the lower density),
   board sweep, 16/16 bit-exact gate vs the new reference, then the tok/s number.
5. **Optional margin run** — same bitstream flow targeting 250 MHz on the looser design.

Predicted outcome, honestly hedged: **C lands ~95–107k, D lands ~110–122k @200 MHz**
(PROJECTED → exact at step 3). The whole campaign is bounded by one or two Vivado runs, not by
a new RTL architecture.

## 6. What this review adds beyond doc 6

Doc 6 already said "100k needs the model, not the schedule." This review (a) verified the
cycle law bit-exactly against silicon so the model-shrink projections are quantitative, (b)
found the **lane-granularity constraint** (d ≡ 0 mod 128) that silently kills the most
intuitive shrink (d=192), (c) identified **C/D as the d-preserving configs** that make the RTL
delta nearly free, and (d) killed the temptation to re-stack double-pump into the projections.
