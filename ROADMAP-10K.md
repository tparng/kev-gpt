# Roadmap to 10,000 tok/s — taking the CPU out of the loop

The target: **≥10,000 tok/s** generating Kevin on the KV260. This document is the
honest plan to get there from where we are now, what each step buys, and where it
loses. It follows the project rules: *bit-honest before fast* (every stage has a
bit-exact / cosine gate before any speed number is trusted) and *honest-first*
(the crossover and the dev-effort cost are stated, not hidden).

## Where we are (2026-06-01)

Done and verified on real silicon:
- Trained INT4 model (val 0.804), all 17 GEMVs **bit-exact on fabric**, the full
  forward **bit-identical** to the numpy/Brevitas reference, generating coherent
  Kevin (*"once upon time there be lazy bunny"*).
- Measured: **~15 s/token ≈ 0.07 tok/s.**

The 15 s is almost entirely the *driver*, not the silicon: ~83 % is re-streaming
~1.6 MB of weights every forward as one-32-bit-AXI-Lite-write-per-byte from Python;
the rest is `PE=1` and no KV cache. The fabric sits nearly idle.

## The one fact that shapes everything

**At 10,000 tok/s the budget is 100 µs per token.** You cannot do hundreds of
register transactions — let alone a CPU round-trip per layer — in 100 µs. So:

> **Any Python/CPU work in the per-token loop caps you ~2 orders of magnitude
> below 10k.** The interface fixes (resident weights, C, DMA, wider transfers) are
> worth doing, but they top out around ~100 tok/s. 10k requires the entire forward
> to run as a **hardware dataflow** — CPU only sets the prompt and drains tokens.

This is the Taalas lesson at hobby scale: weights *in* the silicon, the whole model
running spatially, no software in the hot loop.

## Is 10k even possible on this board? (yes, for this model)

Feasibility math for the deployable model (~3.2M linear params → ~3.2M MAC/token):

| Quantity | Value |
|---|---|
| MACs/token (linear) | ~3.2 M |
| MAC/s needed for 10k tok/s | **~32 GMAC/s** |
| MAC/cycle needed @100 MHz | ~320 |
| MAC/cycle needed @200 MHz | ~160 |
| DSPs available (used now) | 1248 (**0**) |
| LUTs used now | ~10 % |
| URAM for resident weights | 49 / 64 |

`PE=256 @100 MHz = 25.6 GMAC/s ≈ 8k tok/s`; `PE=256 @200 MHz ≈ 16k tok/s`. The
Stage-1 GEMV already closed **300 MHz** out-of-context, and we use **0 DSPs**. So
the compute headroom for 10k exists — the binding constraints are the **driver
architecture** (CPU in the loop) and the **non-linear ops** (softmax/norm/GELU/
attention), not raw MACs. Our own `roofline.py` ceiling for this on-chip model is
~12k tok/s, consistent with this.

Honest where it loses: this only works because the model is tiny and **fits
on-chip** (1.5 MB INT4, under the ~3 MB URAM wall). Past the ~6.3M-param crossover
it spills to DDR and none of this applies. The dumbness is the price of the speed.

---

## The plan

Three tiers. Tier 1 is the resident-weights win (in flight). Tier 2 is cheap driver
work that makes the chat *usable* (~100 tok/s) while keeping the CPU in the loop.
Tier 3 is the real accelerator — the only thing that reaches 10k.

### Tier 1 — Resident weights *(in progress)*
**What:** whole model loaded once into URAM (`gemv_resident`), per layer set
`W_BASE` + stream only the activation. No per-token weight movement.
**Status:** RTL sim-bit-exact at all offsets/shapes; OOC synth fits **49 URAM**,
closes 100 MHz (WNS +0.086 ns); bitstream building; `PLResident.preload()` driver
written.
**Gate:** on-board PL forward `identical == numpy`, measure tok/s.
**Expected:** ~1–3 s/token (kills the ~83 % weight-streaming).
**Effort:** ~done.

### Tier 2 — Driver efficiency (CPU still in loop) → ~100 tok/s
Make the chat usable without the big hardware build. Each keeps bit-exactness.

- **T2.1 — Drive the forward loop in C, not Python.** Per-poke overhead drops
  ~10–50× (no interpreter). The `a53` C driver pattern already exists.
  *Expected:* ~50–300 ms/token.
- **T2.2 — Pack 4 bytes per 32-bit transfer.** Widen `W_DATA`/`X_DATA` to accept 4
  bytes/write → 4× fewer transactions. *Expected:* stacks on T2.1.
- **T2.3 — AXI-DMA for activation + readback.** Replace byte-poking with a single
  DMA descriptor per layer (PS DMA or an `axi_dma` in PL feeding an AXI-Stream
  port on the engine). *Expected:* **~100 tok/s** — the ceiling of CPU-in-the-loop.
- **T2.4 — Wire the webchat "PL" toggle** once T2.x makes it tolerable.

**Effort:** days. **Ceiling:** ~100 tok/s — useful, but *not* the goal.

### Tier 3 — The fabric accelerator (CPU out of the loop) → 10k
This is the real build. Generation becomes a hardware dataflow; the A53 only writes
the prompt tokens and reads results from a FIFO. Each sub-step is gated bit-exact /
cosine vs `goformer_full`.

- **T3.1 — PE-banked GEMV (PE≈128–256).** Per-lane URAM banking: each lane owns a
  row-stripe and reads its own bank, so `PE_LANES` MACs land per cycle (the URAM
  word width = the lane count — naturally efficient). This is the throughput core.
  *Gate:* bit-exact GEMV, report Fmax/util/URAM. *Buys:* the ~16–256× compute.
- **T3.2 — KV cache + incremental decode.** Process only the *new* token's position
  each step; cache per-layer K/V on-chip. Kills the O(T²) recompute (the current
  forward recomputes the whole context every token). *Gate:* incremental logits ==
  full-recompute logits, bit-exact.
- **T3.3 — Fabric-native non-linearities** (doc 2's Stage 3): LUT-softmax
  (running-max, exp LUT), LayerNorm/RMSNorm, GELU LUT, residual adds, embedding
  lookup, per-channel dequant inline after each GEMV. These *must* be in PL — run
  on the A53 over AXI and the CPU is back in the loop. *Gate:* each op cosine
  > 0.9999 vs `goformer_full`; end-to-end logits cosine vs the model.
- **T3.4 — Hardware token sequencer.** An FSM (or small dataflow) that runs the
  whole per-token forward — embed → 4×(norm, qkv, attn, proj, +res, norm, mlp,
  +res) → norm → head → sample → append-to-KV — and *loops*, with **zero** CPU
  involvement between tokens. Sampling in hardware (LFSR + top-k threshold / argmax).
  A/V interface: A53 writes prompt token-ids into an input FIFO, reads generated
  token-ids from an output FIFO. *Gate:* given a seed, the hardware emits the *same
  token stream* as `goformer_full`; measure tok/s.
- **T3.5 — Clock push.** 100 → 200–300 MHz (the GEMV closed 300 MHz OOC; the full
  design with the sequencer + non-linears will need timing work). Linear on tok/s.

**Effort:** weeks of FPGA work, multiple long Vivado builds, bit-exact validation
at every step. This is essentially Stage 3 + Stage 4's "make it a product."

---

## Milestone ladder (expected tok/s)

| Milestone | tok/s (rough) | CPU in loop? |
|---|---|---|
| Now (Python, re-stream) | ~0.07 | yes |
| T1 resident weights | ~0.3–1 | yes |
| T2.1+T2.2 C + 4-byte | ~3–20 | yes |
| T2.3 DMA | ~100 | yes |
| T3.1 PE-bank (still SW-driven per layer) | ~100–500 | yes |
| **T3.1–T3.5 full sequencer** | **8k–16k** | **no** |

The jump from ~100 to 10k is exactly the CPU-out-of-the-loop step. There is no
incremental path across it — it's a different architecture, not a faster driver.

## Validation discipline (don't skip)

Every tier re-runs the existing gates: `model.validate_goformer` (cosine > 0.9999),
the per-layer RTL sim (`run_*.sh`, maxabserr=0), and the on-board
`identical == numpy` forward check. A speed number is only quoted after its gate
passes. The hardware sequencer's gate is the strongest: *same generated token
stream as the software reference, from the same seed.*

## Honest risks / where this could fall short

- **Non-linears in fabric are the hard part**, not the GEMV. Softmax/LayerNorm in
  bit-exact fixed-point is fiddly; expect cosine-gate iteration, and the crossover
  question (how much precision) is a real result, not a footnote.
- **Timing at 200–300 MHz** with the sequencer + non-linears is not free; may land
  lower, capping tok/s proportionally.
- **It only works on-chip.** This is a tiny, deliberately-dumb model. Scale past the
  ~6.3M crossover and you're back behind the DDR wall — none of this transfers.
- **Effort is high.** Tier 3 is weeks. Tier 2 (~100 tok/s) may be the pragmatic
  stopping point for a *usable demo*; 10k is the *showcase* build.

## Immediate next actions

1. Land Tier 1 (resident build → measure tok/s, confirm bit-exact). *(in flight)*
2. Tier 2.1/2.2 (C + 4-byte driver) to make the webchat usable, then wire the toggle.
3. Scope T3.1 (PE-banked URAM GEMV) — the throughput core and the first real piece
   of the 10k accelerator.
