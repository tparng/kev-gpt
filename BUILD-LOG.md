# Build log — how Kevin got onto the silicon

The narrative spine for the writeup: what we actually did, in order, with the
honest numbers and the findings that mattered. The bumps *are* the story — every
bug we hit on real hardware is a result the design docs predicted or a lesson worth
keeping. Mischief in the title; measurement in the body.

The thesis throughout (from `0-master.md`/`2-llm-on-kria.md`): single-stream decode
is memory-bandwidth bound, so the only way to win on a KV260 is to keep the whole
(tiny, INT4) model **on-chip** and never touch DDR — and Kevinising the corpus makes
the model dumb enough to fit. The dumbness and the speed are the same property.

---

## 1. The data and the model (Track A) — all gates green

- **Keviniser** strips function words by POS (spaCy), compressing TinyStories to
  **70.1 % of words (~67 % of tokens)** — a ~1.5× reduction, not 10× (we kept the
  hedging honest; see §note on what Kevinising actually buys).
- Trained a **3.2M-param** char-level GPT on the full Kevinised corpus: FP best
  **val 0.780**, coherent telegraphic Kevin (*"once upon time there be timmy him
  live big city…"*), ~63 min on a laptop RTX 3050 Ti (eager mode — no Triton on
  Windows, so `torch.compile` was dropped; throughput was fine anyway).
- **INT4 QAT** (Brevitas, warm-started off FP): val **0.804**, within 3 % of FP.
- **Export → bit-honest gate:** the exported integer datapath reconstructs the
  Brevitas forward to **cosine 1.0000000**. Packed weights **1560 KB**, under the
  ~3 MB on-chip ceiling. This is the contract everything downstream checks against.

## 2. Board bring-up + the honest baseline (Stage 0)

KV260 on Ubuntu 22.04, reached over SSH (later Tailscale: `kria-kev`). Stage 0 is
the A53 baseline — the "before" number.

- `A53_BASELINE tok_s=177.8 eff_GBps=0.28`.
- **Honest surprise:** it's **compute/unpack-bound, not bandwidth-bound** — 0.28
  GB/s is ~1.4 % of the 20 GB/s DDR wall. The naïve scalar INT4-nibble unpack on the
  in-order A53 is the bottleneck, ~70× below *both* roofline ceilings. A fair "CPU
  baseline" for the eventual bake-off would need a tuned (NEON) unpack; we recorded
  this rather than flatter the fabric with a strawman.

## 3. The live chatbot (A53) — Kevin talks

A Flask app on the board runs the FP model (PyTorch CPU) and streams tokens with
**TTFT + tok/s** shown. Reached over Tailscale at `kria-kev:8080`. Honest A53 number:
**~11 tok/s** at 150-token context (char-level, so chars/s). A tuning sweep found the
unglamorous truth: **INT8-dynamic-quant and oneDNN both made it *slower*, and
`torch.compile` was flat** — the model is so small it's *dispatch- and O(T²)-recompute
bound, not compute-bound*, so the usual CPU weapons have nothing to bite on. Only a
KV cache would help. This shaped the whole speed story later.

## 4. First PL silicon — and the bug only hardware could catch

Goal: run the model's matmuls on the FPGA fabric. The Stage-1 GEMV core was
sim-bit-exact and OOC-synthesised, but **had never been run on hardware**. We built
the first real bitstream (Zynq MPSoC PS + AXI-Lite shell around the GEMV core),
loaded it via `fpgautil`, and drove it from the A53 over `/dev/mem`.

- **The path worked:** `IDCODE` verified, the GEMV FSM ran **cycle-accurately**
  (4113 cycles) on silicon.
- **But every output was zero.** The diagnosis, dug out of the synth log: the
  weight ROM was a combinational `initial $readmemh` array; the data is read at
  *elaboration* but synthesis **drops it** (*"Net wmem has no driver" → pruned to
  zero*) and the multi-port read won't map to BRAM. **Sim-correct, synth-wrong** —
  exactly what *"bit-honest before fast"* exists to catch, caught on first silicon.
- **Fix:** rewrote the core as a **registered-read BRAM** (`gemv_core`). Validated
  bit-exact in sim, then confirmed *in the synth report* that the weights now land
  in **8 initialised RAMB36s** (no prune), *then* built the bitstream. Result on the
  board: **`bitexact=True, mismatches=0/256`** — the model's math, correct on fabric.

Two findings banked here: (a) URAM **cannot be bitstream-initialised** (UltraScale+
hardware limit), so the real model's weights must be **loaded into URAM at runtime**,
not baked in — the bandwidth thesis survives ("resident on-chip", not "etched"); and
(b) build Vivado outside OneDrive (its cloud-files filter locks the project).

## 5. All 17 layers, one bitstream — the loadable engine

A fixed-weight demo isn't the model. We made the engine **runtime-loadable and
runtime-sized** (`gemv_load` / `gemv_axi_load`): weights stream in over AXI, M/K are
registers, so *one bitstream runs every layer* by re-streaming its weights. Gated in
sim (core + an AXI bus-functional model), then on silicon:

> `STAGE2_ALL_LAYERS pass=17/17` — qkv (768×256), proj (256×256), mlp_fc (1024×256),
> mlp_proj (256×1024), head (193×256), all **bit-exact**.

The entire linear backbone of the model, correct on the fabric, from one image.

## 6. The full forward on fabric — Kevin generated on silicon

The matmuls were on hardware; the rest (dequant, attention, softmax, LayerNorm,
GELU, sampling) still had to be stitched around them. We wrote a **pure-numpy full
forward with a pluggable matmul** (`goformer_full`) and validated it against the
Brevitas model: **cosine 0.9999113, same next-token at every position.** Because the
PL GEMV is bit-exact to numpy's, swapping the backend to the fabric leaves the whole
forward bit-identical — so the numpy validation *is* the proof the PL forward is
correct.

Wiring the Python PL backend surfaced one more hardware-only bug: Python `mmap`
byte-slice writes **decompose into 4 byte-sized AXI transactions**, and each write
to the streaming `W_DATA` port pulses the load again → corrupted weights. Fix: a
numpy `uint32` view so every register write is **one aligned 32-bit access** (the
standard MMIO pattern). After that:

> `PL forward identical == numpy, maxdiff=0` — and the fabric generated
> **"once upon time there be lazy bunny."** The whole model, running on the FPGA.

## 7. The speed reckoning

First on-fabric generation: **304 s / 20 tokens ≈ 15 s/token.** Honest read: ~95 %
of that is *Python poking AXI-Lite registers one byte at a time* (re-streaming
~1.6 MB of weights every forward), plus `PE=1` and no KV cache. The fabric sits
nearly idle. Our own roofline says this board should do **~12,000 tok/s** on this
model — a ~170,000× gap that is *all implementation*, not physics.

This is the same thesis Taalas industrialised (weights *in* the silicon, no CPU in
the loop, full spatial dataflow) — at hobby scale. The plan to close the gap is
`ROADMAP-10K.md`; the one load-bearing fact is that **any CPU/Python in the per-token
loop caps you ~100× below 10k** (100 µs/token budget), so 10k requires the entire
forward as a hardware dataflow.

## 8. Tier 1 — resident weights (done)

The first and biggest single fix: stop re-streaming weights every token. The whole
1.56 MB model now lives in **49 URAMs**, loaded **once**. Getting there meant a real
memory redesign — URAM is efficient only with wide words (a byte-wide 1.6 M-deep
array needs ~390 URAM blocks; we have 64), so weights live in a **64-bit-wide URAM**
with a `w_base` byte-offset per layer, byte-select on the wide word, and a
**multi-cycle read pipeline** (a deep URAM cascade can't do 1-cycle at 100 MHz).
Sim-bit-exact at arbitrary offsets, OOC-confirmed to fit (49/64 URAM, WNS +0.086 ns),
then built.

> Preload 1.56 MB → URAM in 11.3 s (one-time). Forward `identical == numpy, maxdiff=0`.
> **4.6 s/token, down from 15** (~3.3×), still bit-exact.

The remaining 4.6 s/token is now the activation streaming + readback (still Python)
and the O(T²) recompute — the Tier-2/Tier-3 targets.

## Where we are, and the line to 10k

| Stage | Result | Gate |
|---|---|---|
| Model trained (INT4 QAT) | val 0.804, cosine 1.0 vs FP-int | bit-honest ✅ |
| A53 Stage-0 baseline | 177.8 tok/s, compute-bound (honest) | measured ✅ |
| A53 live chatbot | ~11 tok/s, Tailscale-reachable | running ✅ |
| GEMV on fabric | bit-exact, all 17 layers | maxabserr=0 ✅ |
| Full forward on fabric | generates Kevin, == numpy | maxdiff=0 ✅ |
| Resident URAM weights | **4.6 s/tok** (3.3×) | maxdiff=0 ✅ |
| → Tier 2 (C+DMA driver) | ~100 tok/s (target) | — |
| → Tier 3 (HW sequencer) | **8–16k tok/s** (target) | — |

The jump from ~100 to 10k is the architecture change, not a faster driver: a
hardware token sequencer running the whole forward in fabric, CPU only setting the
prompt and draining tokens. See `ROADMAP-10K.md` for the staged plan, the feasibility
math, and — honest-first — where it loses (the non-linears are the hard part, and
none of it transfers past the ~6.3M-param on-chip crossover).

### What Kevinising actually buys (keep honest in the writeup)

A late, important correction we owe the reader: on TinyStories specifically,
Kevinising is **mostly a ~1.5× token/KV reduction**, not the source of the speed.
The order-of-magnitude win is **small-model-on-chip vs DDR**; Kevinising is the
garnish (and the joke), plus a coherence-per-param margin that matters more for
harder corpora than TinyStories. The "dumbness *is* the speed" line is cute but,
quantified, it's the on-chip residency doing the heavy lifting. Say so.
