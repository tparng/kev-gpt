> **STATUS (2026-06-06): TARGET BROKEN 2.4×.** The measured ladder: 11,143.9
> (1 stream @200) → 16,969.3 (N=4) → 17,740.6 (ping-pong N=8) → 19,275.6
> (single-pass N=8) → **24,134.0 tok/s aggregate MEASURED** (N=16: 12 DSP-packed
> banks + shared LN/attention, 16/16 bit-exact 3/3 @166.7 MHz). N=16 is the
> proven stream ceiling (3 MACs/DSP impossible — research/dsp3_pack_proof.py);
> the chase to 100k is now cycles (103,879 → ~40k floor) × clock (200/250 MHz):
> 16 × 250 MHz / 40k cyc = 100k. Engineering log: fabric/stage3/WIDE-WORD-DATAPATH-LOG.md §12–18.

# Roadmap to 10,000 tok/s — and the honest ladder to the board's ceiling

The committed target: **≥10,000 tok/s** generating Kevin on the KV260. That number
is comfortably feasible and this is the staged plan to reach it. The second half of
the document answers the bigger question — *how far past 10k can this silicon
actually go?* — because the honest answer (≈200k realistic, ~500k beyond this
board) changes how you'd choose between the small Kevin model and a full-TinyStories
one.

It follows the project rules: **bit-honest before fast** (every stage has a
bit-exact / cosine > 0.9999 gate before any speed number is trusted) and
**honest-first** (the crossover, the dominant risk, and the dev-effort cost are
stated, not buried). Every number below is tagged **MEASURED** (on silicon /
printed by a committed tool), **DERIVED** (arithmetic from the spec), or
**PROJECTED** (needs a Vivado synth or on-board run to confirm). The two analysis
passes behind this doc recomputed every figure adversarially; the corrections are
folded in.

## Where we are (2026-06-01)

Done and verified on real silicon — the whole gate ladder is green:

| Stage | Result | Gate |
|---|---|---|
| Model trained (INT4 QAT) | val 0.804, cosine **1.0000000** vs FP-int | bit-honest ✅ MEASURED |
| Export → fabric | 17 layers, 1560 KB packed, fits 3 MB | cosine 1.0 ✅ MEASURED |
| GEMV on fabric | bit-exact, all 17 layers | maxabserr = 0 ✅ MEASURED |
| Full forward on fabric | generates *"once upon time there be lazy bunny"* | maxdiff = 0 vs numpy ✅ MEASURED |
| Resident URAM weights | **4.6 s/token ≈ 0.22 tok/s** | maxdiff = 0 ✅ MEASURED |

The 4.6 s/token is almost entirely the *driver*, not the silicon. ~71 % of the
per-token traffic is **9,409 single-beat result reads** plus 17 per-layer CPU↔PL
round-trips, all driven from Python over AXI-Lite (~7.75 µs/transaction, calibrated
from the measured run), on top of `PE=1` and the O(T²) full-context recompute. The
fabric sits nearly idle. Our own roofline says this board can do **hundreds of
thousands** of tokens/sec on this model — the gap is implementation, not physics.

## The one fact that shapes everything

**At 10,000 tok/s the budget is 100 µs per token** (DERIVED: 1 s / 10⁴). You cannot
do hundreds of register transactions — let alone a CPU round-trip per layer — in
100 µs. So:

> **Any Python/CPU work in the per-token loop caps you ~2 orders of magnitude below
> 10k.** The interface fixes (resident weights, C, DMA, wider transfers) are worth
> doing, but they top out around **~100 tok/s**. 10k requires the entire forward to
> run as a **hardware dataflow** — the A53 only writes the prompt and drains tokens.

This is the Taalas lesson at hobby scale: weights *in* the silicon, the whole model
running spatially, no software in the hot loop.

## Is 10k feasible on this board? Yes — and it's compute-bound, not bandwidth-bound

Per-token work, re-derived from the spec (DERIVED, all independently confirmed):

| Quantity | Value |
|---|---|
| Linear MACs/token (4 layers) | 3,145,728 |
| Head MACs/token (193×256) | 49,408 |
| Attention MACs/token at T=256 | 524,288 (= 2048·T) |
| **Total, KV-cached incremental decode** | **3,719,424 MAC/token** |
| MAC/s needed for 10k | 37.2 GMAC/s |
| **MAC/cycle needed @100 / 200 / 300 MHz** | **372 / 186 / 124** |
| DSPs for 10k @300 MHz (1 MAC/DSP) | ~124 of 1248 (**~10 %**) |
| On-chip weight reads needed at 10k | 15.7 GB/s |
| URAM read floor (64 banks ×72b @100 MHz) | ~58 GB/s — already **3.7× over** |

So 10k needs ~10 % of the DSP budget at 300 MHz and a fraction of the on-chip read
bandwidth. The compute headroom is real (the Stage-1 GEMV closed **300 MHz OOC at
0 DSP** — INT4×INT8 fits in LUTs). **10k is bound by the architecture (CPU out of
the loop) and the per-lane weight banking, not by raw MACs.**

**The framing correction.** Earlier drafts cited "~12.7k tok/s on-chip ceiling."
That is wrong: **12,658 tok/s is the *DDR* line** (20 GB/s shared ÷ 1.5 MB/token,
MEASURED from `roofline.py`). The *on-chip* band the model's own roofline prints is
**126,582–632,911 tok/s**. 10k sits ~13–64× *below* that band — it is not
bandwidth-limited on-chip at all. (Honest caveat used later: that printed
632,911 high end assumes a 1000 GB/s on-chip rate; real 2-port URAM delivers
~461 GB/s, so the *physical* single-stream ceiling is lower — see "the board's true
ceiling".)

**The one hard dependency: the KV cache.** Today's `goformer_full` full-recomputes
the whole context every token (O(T²)) — at T=256 that is **805M linear
MACs/forward, 256× the incremental work**. 10k is *impossible* in that mode.
Incremental decode (process only the new position, cache per-layer K/V on-chip) is a
**prerequisite, not an optimisation.**

Honest where it loses: this only works because the model is tiny and **fits
on-chip** (1.5 MB INT4, under the ~3 MB URAM wall). Past the **6.29M-param crossover**
it spills to DDR and none of this applies. The dumbness is the price of the speed.

---

## The plan

Three tiers. Tier 1 (resident weights) is **done**. Tier 2 is cheap driver work
that makes the chat *usable* (~100 tok/s) with the CPU still in the loop. Tier 3 is
the real accelerator — the only thing that reaches 10k and beyond.

### Tier 1 — Resident weights ✅ DONE
Whole 1.56 MB model loaded once into **49 of 64 URAM** (64-bit-wide words, RLAT=2
read pipeline), per layer set `W_BASE` + stream only the activation. Closes 100 MHz
(WNS **+0.086 ns**, MEASURED). Killed the ~83 % per-token weight re-streaming:
15 s/token → **4.6 s/token**, still bit-exact (maxdiff = 0).

### Tier 2 — Driver efficiency (CPU still in loop) → ~100 tok/s
Makes the chat usable without the big hardware build. `gcc 11.4` is on the board, so
C builds are native. Ordered by **tok/s per unit effort**:

1. **Software KV cache (do this first).** Pure algorithm, no bitstream change. The
   forward stops recomputing all T positions each token. At a typical chat context
   (avg T≈82) this is an **~82× cut: 0.06 → ~5 tok/s** (PROJECTED). *Gate:* the
   incremental logits must be `np.array_equal` to the full-recompute forward
   (maxabserr = 0). Biggest single near-term win.
2. **C/MMIO driver** (the `gemv_load_drv.c` pattern). Replaces the ~7.75 µs Python
   AXI-Lite poke with a compiled `volatile` store/load. Stacked on the KV cache:
   **~80–160 tok/s** (PROJECTED — needs an on-board read-latency microbenchmark; the
   71 %-of-traffic result read-out is read-stall-bound, so it lands at 80 if reads
   are ~1 µs, 160 if ~0.3 µs).
3. **pack-4** (widen `W_DATA`/`X_DATA` to 4 bytes/write): modest for inference
   (~5–15 %, the read-out dominates once cached); its real payoff is the one-time
   preload (11.3 s → sub-second).
4. **AXI-DMA** (one descriptor/layer): theoretically ~600–3,000 tok/s but bounded by
   17 dependent CPU↔PL round-trips, and it needs an RTL/bitstream change — the
   boundary of this tier.
5. **Wire the webchat "PL" toggle** once the chat is tolerable.

**Effort:** days. **Ceiling: ~100 tok/s** — the hard wall of anything with the CPU
in the per-token loop (9,409 result reads + 17 round-trips per token; no C trick,
packing, or DMA removes that). A perfectly good live demo — but *not* the goal.

### Tier 3 — The fabric accelerator (CPU out of the loop) → 10k and up
The real build. Generation becomes a hardware dataflow; the A53 only writes prompt
token-ids into an input FIFO and burst-drains generated ids (~20 KB/s) from an
output FIFO. Each sub-step is gated bit-exact / cosine vs `goformer_full`.

- **T3.1 — KV cache + incremental decode (in fabric).** The prerequisite. Process
  only the new token's position each step; cache per-layer K/V on-chip (INT8,
  512 KB at full T=256). *Gate:* incremental logits == full-recompute logits,
  bit-exact.
- **T3.2 — PE-banked GEMV (the throughput core).** Restripe the weight matrix so
  each lane reads its own bank. **This is the single biggest open feasibility item.**
  Two regimes (DERIVED):
  - *One URAM bank per lane* is clean but caps at **PE = 64** (64 blocks → 64
    single-read-port banks). PE=64 @300 MHz ≈ **5,200 tok/s** — short of 10k.
  - *Wider-word banking* (the 72-bit URAM word = 18 INT4 nibbles, so 64 blocks
    deliver up to 1,152 nibbles/cycle one-port) can feed **PE ≈ 128–256** if the
    layout is conflict-free. PE=128 @300 MHz ≈ **10,300 tok/s**; PE=256 ≈ 20,600.
    *Conflict-free banking past PE=64 is unproven — a Vivado synth gates the whole
    10k claim.* Output-row padding is free at every PE (M ∈ {768,256,1024,256} all
    divide 256).
- **T3.3 — Fabric-native non-linearities** (the *hard* part — not the GEMV). See
  "the non-linears" below. Each op gated cosine > 0.9999 vs `goformer_full`.
- **T3.4 — Hardware token sequencer.** An FSM/dataflow that runs the whole per-token
  forward — embed → 4×(LN, qkv, attn+KV, proj, +res, LN, mlp, +res) → LN_f → head →
  sample → append-KV — and *loops* with **zero** CPU between tokens. Decode is
  inherently sequential across the 17 sub-stages (each layer needs the prior
  residual), so the win is deep pipelining *inside* each GEMV/non-linear, not across
  layers. Sampling is an in-fabric LFSR + temperature + argmax/top-k. *Gate:* given
  a seed, the fabric emits the **same token stream** as `goformer_full`. (Greedy
  argmax is bit-exact gate-able; stochastic sampling cannot match numpy's Mersenne
  stream from an LFSR, so the fabric LFSR is declared the reference and only the
  *logits* are validated bit-exact.)
- **T3.5 — Clock push.** 100 → 200–300 MHz. **This is the dominant lever and the
  dominant risk** (below). tok/s scales linearly with it.

**Effort:** weeks of FPGA work, multiple long Vivado builds, bit-exact validation at
every step.

---

## Milestone ladder

| Rung | tok/s | CPU in loop? | Binding constraint | Tag |
|---|---|---|---|---|
| Now (resident, Python) | ~0.22 | yes | AXI-Lite pokes + O(T²) recompute | MEASURED |
| + software KV cache | ~5 | yes | Python interpreter overhead | PROJECTED |
| + C/DMA driver | ~80–160 | yes | 9,409 reads + 17 round-trips/token | PROJECTED |
| HW sequencer, PE=64 | ~5,200 | **no** | one-URAM-bank-per-lane (64 banks) | PROJECTED |
| HW sequencer, PE=128 | **~10,300** | **no** | wider-word banking (unproven) + Fmax | PROJECTED |
| HW sequencer, PE=256 @300/400 MHz | ~20,600 / 27,500 | **no** | URAM weight-read bandwidth | PROJECTED |

The jump from ~100 to 10k is the **architecture change** (CPU out of the loop), not
a faster driver — there is no incremental path across it.

---

## How far past 10k? The board's true ceiling

This is where the 500k question lands. Three independent ceilings converge in one
band, and the answer reframes the whole "fast" conversation.

**Single-stream is URAM-bandwidth = compute bound at ~200k tok/s.** KV-cached decode
reads each of the 1.5 MB INT4 weights exactly once per token (no reuse to amortise),
so the rate is on-chip read bandwidth ÷ 1,572,864 bytes. Both URAM ports (the
boot-load port is free at inference) give **345.6 GB/s @300 MHz → ~220k tok/s**,
**460.8 GB/s @400 MHz → ~293k tok/s** (DERIVED). The DSP array, at the *proven*
2 MAC/DSP INT8 packing, is **~201k @300 MHz** — essentially the same wall. (These
aren't independent axes: each linear MAC consumes exactly one INT4 nibble, and
2-port URAM delivers 2,304 nibbles/cycle, so *the weight read **is** the compute
feed*.) **Honest single-stream ceiling: ~200–270k tok/s.** Note `roofline.py`'s
printed 632,911 high end assumes 1000 GB/s on-chip — ~2.2× optimistic vs real URAM.

**500k is beyond this silicon.** It requires *three* things at once (DERIVED):
~4 MAC/DSP INT4 packing (unbuilt — only 2 MAC/DSP is a proven Xilinx trick),
**400 MHz** (the only resident design we've actually built closed **100 MHz** —
4× short on -2LV), **and** batch B≥3 to feed that array. But the KV cache pins the
batch: B concurrent streams need B × 512 KB on-chip, and after weights only ~1.2 MB
is left → **Bmax ≈ 2** at full context. B=3 spills KV to DDR and re-imports the
20 GB/s wall the project exists to escape. The requirements are mutually
contradictory; 500k single-stream needs 786 GB/s of reads against a 461 GB/s URAM
maximum. **Physically over-constrained on this board.**

**300k is a ragged-edge stretch**, not a free number: ~3 MAC/DSP + ~400 MHz + light
batching (B≈1.4). The only measured full-design Fmax (100 MHz) would cap it at
~67–73k, so 300k rests entirely on an unproven 3–4× clock improvement plus unbuilt
packing.

### The full-TinyStories fork (dropping the Kevin gimmick)

Real, and it stays under the crossover — but the 300k figure does not survive
contact with the memory budget:

- The crossover (6,291,456 params, 3 MB INT4) is **exactly 2.0× the Kevin model**.
  A coherent full-English model plausibly wants ~1.5–2× the params, so the *largest*
  that fits is **~5.9–6.3M (2.8–3.0 MB INT4)** — candidate geometries L4 d352
  (5.95M), L6 d288, L8 d256 (6.29M, the exact wall). About double Kevin, no more.
- tok/s scales ~1/params, so a ~6M model runs at **half** the Kevin ceiling:
  **~100–155k tok/s** single-stream (PROJECTED). To get 300k single-stream from a
  6M model, the Kevin model would first have to hit ~600k — above even its
  optimistic 500k. **The 500k→300k trade is internally inconsistent: a 2× model
  halves throughput.**
- Worse: total on-chip fabric memory is **2.88 MB** (2.25 MB URAM + 0.65 MB BRAM) —
  *less* than the 3 MB weight budget. A ~3 MB full model leaves **~0 room for the
  KV cache**, so batching is unavailable and KV partly spills. And a real BPE vocab's
  FP embedding table (~4 MB at 8k vocab) alone exceeds on-chip memory — the fork
  **must stay char-level**.

So the full model lands at **~100–150k tok/s**, not 300k — but that *confirms the
thesis*: dropping Kevin and still doing hundreds-of-thousands of tok/s proves the
order-of-magnitude win is **small-model-on-chip-vs-DDR (10–50×)**, and Kevin was the
**~1.5× garnish** (70.1 % of words, ~67 % of tokens).

### The decision: choose on coherence, not speed

The clinching number (DERIVED): **content-normalized, 500k Kevin ≈ 335k
full-equivalent ≈ 300k full.** Kevin says the same thing in ~0.67 the tokens, so the
"500k vs 300k" gap is **token accounting, not silicon** — both land in the same
hundreds-of-thousands band. Pick Kevin-vs-full on whether you want the joke or a
coherent model; the speed is essentially the same. And Kevin's smaller footprint is
a genuine engineering edge (it leaves room for the KV cache that the full model
can't afford), not just a gag.

**Recommended target for the writeup:** *10k tok/s is the comfortable, defensible
milestone; ~100–250k is the realistic ceiling; 300k+ only at the ragged
INT4-packing / batching / 400 MHz edge; 500k is beyond this part.* Every number
above ~5k is gated on one unmeasured quantity — see the dominant risk.

---

## The non-linears (the hard part, not the GEMV)

The matmuls map cleanly to a systolic array; the non-linears shape the datapath and
are where the gate softens from bit-exact to **cosine > 0.9999** (exp/rsqrt/erf
can't reproduce a float reference to maxabserr = 0). The end-to-end budget is tight
but workable: cos > 0.9999 ⇒ ≤ ~1.4 % relative-L2 drift, ~0.3 % per stage across the
~30 stages if errors add in quadrature (DERIVED). A 16-bit-fractional datapath gives
~1.5e-5 steps — two orders under budget — so **precision is not the wall;
architecture is.**

- **Softmax is the long pole** — the only non-linear that scales with context
  (T=256), the only one chaining three ops (running-max → exp LUT → reciprocal), and
  the reason the KV cache is mandatory (without it softmax is O(T²)). Plan: Q1.15 exp
  LUT over z ∈ [−11.1, 0] (exp(−11.09) ≈ 1 LSB, so deeper values correctly flush to
  0); causal mask is structural (masked → exp = 0); ≥24-bit denominator accumulator.
- **LayerNorm** (gamma-only, eps 1e-5, population variance): streaming mean + E[x²],
  then rsqrt via an 8-bit LUT seed + 1 Newton iteration (2.3e-5 rel — inside budget).
- **GELU** (erf-exact reference): a **256-entry LUT** over [−8,8] gives 3.9e-4 max
  error (512 entries → 9.8e-5), both under the tanh-approx's own 4.7e-4 (DERIVED).
- **Dequant/requant**: fold w_scale·act_scale into one stored per-channel fixed-point
  scale (16-bit mantissa → 0.0015 % error). The GEMV int32 accumulator stays
  bit-exact (MEASURED on silicon, |acc| ≤ 1.05e6 < 2³¹).

The exact LUT/iteration counts that clear cosine > 0.9999 end-to-end need a
fixed-point bit-true sim against `goformer_full` on **real-checkpoint activations**
(not random) — that is where the validation iteration lives, and the required
precision is itself a reportable result.

## Validation discipline (don't skip)

Four gates, innermost first — no speed number is quoted before its gate passes:

1. **Model gate** — `model.validate_goformer`, cosine > 0.9999. *Achieved
   1.0000000.*
2. **Per-layer RTL gate** — testbench vs `fabric.golden`, maxabserr = 0. *Achieved,
   all 17 layers.*
3. **On-board forward gate** — `goformer_full` + PL backend == numpy, maxdiff = 0.
   *Achieved; generates coherent Kevin.*
4. **Sequencer gate (Tier 3, strongest)** — given a seed, the fabric emits the
   **identical token stream** as `goformer_full`. *Pending.* Underwritten by per-op
   cosine > 0.9999 (softmax/LayerNorm/GELU) and the incremental == full-recompute
   logit check.

## Honest risks / where this could fall short

- **Fmax on -2LV is the dominant risk.** The full resident GEMV closed only
  **100 MHz** (cascaded URAM, RLAT=2); the 300 MHz number belongs to the *small,
  non-scaling* Stage-1 shared-ROM core. If the full banked sequencer + non-linears
  land at 150–200 MHz, every projected rung scales down linearly — 10k could become
  5–7k, and the ~200k ceiling could become ~70–130k. **Every number above ~5k is
  gated on this single unmeasured quantity.** A Vivado timing run on the banked
  design is the highest-value next experiment.
- **Per-lane banking past PE=64 is unproven.** Wider-word banking (multiple lanes
  per 72-bit URAM word) is plausible from the nibble budget but may force
  bank-conflict stalls; needs synth + sim.
- **3–4 MAC/DSP INT4 packing is asserted, not built.** Only 2 MAC/DSP is a proven
  trick. The entire 200–540k compute band depends on a packing scheme that has had
  no Vivado run.
- **The non-linears in fixed-point** will need cosine-gate iteration; the required
  precision is a result, not a footnote.
- **It only works on-chip.** Past the ~6.29M crossover (or once long context spills
  the KV cache) the fabric is back behind the 20 GB/s wall and none of this
  transfers. **The batching catch is the same wall in disguise:** B>2 at full context
  spills KV to DDR.
- **Effort is high.** Tier 3 is weeks. Tier 2 (~100 tok/s) is the pragmatic stopping
  point for a *usable demo*; 10k is the *showcase*; the ceiling is a roofline result,
  not a deliverable.

## Immediate next actions

1. **Software KV cache** (Tier 2.1) — pure algorithm, gates bit-exact, 0.06 → ~5
   tok/s, makes the rest measurable. Do this first.
2. **C + KV driver** (Tier 2.2) to make the webchat usable (~80–160 tok/s), then wire
   the PL toggle. Microbenchmark the AXI-Lite read latency to fix the 80-vs-160.
3. **The one Vivado experiment that de-risks everything:** synth a PE-banked
   (PE≥128) URAM GEMV on `xck26-2LV` and report **Fmax + URAM/LUT/DSP fit**. That
   single result turns the entire Tier-3 ladder from PROJECTED to grounded — it
   decides whether 10k is a 300 MHz/PE=128 build or a harder one, and where the real
   ceiling sits.
