# Glossary — Kevin on Kria

Terms used across the writeup, grouped by domain. Definitions are scoped to how the
term is used *in this project*, and keep the project's honest-first discipline (where
a term carries a caveat, the caveat is part of the definition).

---

## The project (the framing, and the joke)

- **Kevin** — the deliberately dumb, telegraphic language model at the centre of the
  project. Speaks in clipped "few word do trick" English.
- **Kevin-speak / telegraphic English** — text with function words (articles,
  auxiliaries, most prepositions) stripped and inflection flattened: *"once upon time
  there be lazy bunny."* Fewer tokens to say roughly the same thing.
- **Keviniser** — the spaCy part-of-speech preprocessor that compresses a normal
  English corpus into Kevin-speak by dropping words *by their POS tag* (so it removes
  auxiliary "do" but keeps main-verb "do" — a flat stopword list can't). It strips the
  **training corpus, not the inference input**; the model learns the compressed
  distribution and generates telegraphic text on its own.
- **The fusion ("the joke is the thesis")** — the claim that the model's *dumbness*
  and its *speed* are the same property: fewer tokens → less weight streamed and a
  smaller KV cache → more fits on-chip → faster. **Honest correction:** on TinyStories
  the Keviniser only buys ~1.5× (≈70% of words, ≈67% of tokens); the order-of-magnitude
  win is *small-model-on-chip-vs-DDR*, not the joke. Kevinising is the garnish.
- **goformer** — the project's **golden reference** model: a plain, exact NumPy
  implementation of the forward pass that the FPGA output is validated against. "Be
  bit-honest against goformer" is the recurring discipline.
- **TinyStories** — the public dataset of simple children's stories used as the
  training corpus; small enough that a tiny model stays coherent.
- **Kria KV260** — the target board: a Xilinx development kit built around a Zynq
  UltraScale+ MPSoC, with a quad-core CPU and FPGA fabric on one chip.

---

## The model (ML / transformer)

- **Token** — the unit the model reads and writes. Here it's **char-level**: one token
  ≈ one character, so "tok/s" is literally characters per second.
- **Decode / autoregressive decode** — generating one token at a time, each new token
  conditioned on all previous ones. **Single-stream** decode = one sequence at a time.
- **Prefill** — running the prompt through the model once to populate the KV cache,
  before the first new token is generated. Dominates TTFT.
- **KV cache** — stored per-layer Keys and Values for every past position, so each new
  token only computes its *own* position instead of re-reading the whole context.
- **Incremental decode** — decode *with* a KV cache: O(T) work per token instead of the
  O(T²) of re-running the whole context every step. A prerequisite for any real speed.
- **Attention** — the layer where the new token's Query is compared against all cached
  Keys, softmax-normalised into weights, and used to average the Values.
- **Softmax** — turns a row of scores into a probability distribution (exponentiate,
  then normalise). The fiddliest non-linear to do in hardware, and the one that scales
  with context length.
- **LayerNorm** — normalises a vector to zero mean / unit variance, then scales by a
  learned gain (`gamma`). This project uses the **gamma-only** form (no bias).
- **GELU** — the MLP activation function (a smooth gate). Implemented in fabric as a
  lookup table.
- **MLP / feed-forward** — the per-token two-layer network inside each transformer
  block (expand to `d_mlp`, GELU, project back).
- **Residual stream** — the running vector `x` that each block reads from and adds back
  into; it carries information down the layers. The precision-critical datapath.
- **Embedding** — the table mapping a token id to its initial vector (`tok_emb`), plus
  a per-position vector (`pos_emb`).
- **Logits** — the model's raw output scores over the vocabulary, one per possible next
  token. **Argmax / greedy** picks the highest; **top-k / temperature** sampling adds
  controlled randomness.
- **Vocab / context length (`block_size`)** — number of distinct tokens (here 193) and
  the maximum sequence the model attends over (here 256).
- **GEMV / GEMM** — General Matrix-Vector / Matrix-Matrix multiply. Decode is batch-1,
  so each linear layer is a **GEMV** (matrix × vector). The bulk of the compute.
- **MAC** — Multiply-ACcumulate, the one-multiply-one-add operation a matmul is built
  from. "MACs/token" is the model's compute cost; "MAC/cycle" is the hardware's rate.

---

## Quantisation & numerics

- **INT4 / INT8** — 4-bit / 8-bit integers. Weights are **INT4** (16 levels), activations
  **INT8**. The whole point: 4-bit weights make the model small enough to fit on-chip.
- **QAT (Quantisation-Aware Training)** — training (here fine-tuning) with the
  quantisation simulated in the loop, so the model learns to tolerate INT4 weights.
- **Brevitas** — the PyTorch library used for the INT4 QAT.
- **Dequant / requant** — converting integer accumulator results back to real values
  (`int × scale`) and back down to INT8 for the next layer (`round(x / scale)`).
- **Per-channel scale** — each output channel has its own dequant scale factor; folded
  into a single stored fixed-point number per channel.
- **Q-format (e.g. Q6.25)** — fixed-point notation: *Qi.f* = signed value with `i`
  integer bits and `f` fractional bits. The residual stream is pinned to **Q6.25**
  (32-bit) — the smallest format that keeps the output faithful.
- **Nibble** — 4 bits; one INT4 weight. Two nibbles pack into a byte.
- **Two's complement** — the standard signed-integer encoding; signed INT4 spans
  [−8, 7].
- **Accumulator** — the wide (INT32) register a GEMV sums into so it can't overflow.
- **LUT (lookup table)** — a precomputed table read at runtime to approximate a function
  (GELU, exp). Distinct from "LUT" the FPGA primitive below — context disambiguates.
- **Linear interpolation** — reading *between* two LUT entries for a finer result.
- **rsqrt / Newton-Raphson** — reciprocal-square-root (needed by LayerNorm), computed
  in hardware from a small seed table refined by a couple of Newton-Raphson iterations.

---

## The hardware (FPGA / Kria)

- **FPGA / fabric / PL (Programmable Logic)** — the reconfigurable part of the chip
  where custom digital circuits are built. "Fabric" and "PL" are used interchangeably.
- **PS (Processing System)** — the hard CPU side of the chip: the **A53** cores.
- **A53** — the quad-core Arm Cortex-A53 CPU on the KV260 (≈1.33 GHz). The "baseline"
  the fabric is compared against, and the orchestrator in the CPU-in-the-loop path.
- **SOM (System-on-Module)** — the KV260's plug-in compute module.
- **Zynq UltraScale+ MPSoC** — the chip family: CPU (PS) + FPGA (PL) sharing one
  package and one DDR controller.
- **DDR** — the off-chip DRAM (4 GB), ~20 GB/s, **shared by the PS and PL**. The wall.
- **BRAM / URAM** — on-chip SRAM blocks. **BRAM** (Block RAM, ~5 Mb) is small/flexible;
  **URAM** (UltraRAM, ~18 Mb, 64 blocks × 4096×72-bit) is the big, wide one that holds
  the resident weights. Bandwidth here is hundreds of GB/s — the escape from the wall.
- **DSP48E2** — the FPGA's dedicated hardware multiplier blocks (1248 of them). Notably
  this project uses **0** of them — INT4×INT8 MACs fit in LUTs — leaving all 1248 free
  for the scale-up to 100k.
- **LUT (Look-Up Table) / FF (Flip-Flop)** — the FPGA's basic logic and storage
  primitives; utilisation is reported as a % of these.
- **RTL / SystemVerilog (.sv) / Verilog (.v)** — Register-Transfer-Level hardware
  description; the languages the circuits are written in.
- **Bitstream (.bit / .bit.bin)** — the compiled configuration file that programs the
  FPGA fabric. Loaded on the Kria with `fpgautil`.
- **AXI / AXI-Lite / AXI-Stream** — the on-chip bus protocols. **AXI-Lite** is the
  simple register interface the CPU pokes (one transaction per access — the bottleneck
  in the CPU-in-loop path); **AXI-Stream** is the high-throughput dataflow interface.
- **MMIO / `/dev/mem`** — Memory-Mapped I/O: the CPU reads/writes the fabric's registers
  as if they were memory, via the `/dev/mem` device on Linux.
- **Resident weights** — the whole model preloaded into URAM **once at boot** and kept
  there during inference (URAM can't be initialised by the bitstream itself, so it's
  loaded once, not "baked in"). This is what keeps inference off the DDR wall.
- **`w_base`** — the per-layer offset into the resident weight URAM that selects which
  layer's weights to use, so all layers live in one memory.

---

## The accelerator architecture

- **PE (Processing Element) / lane** — one MAC unit. **PE=256** means 256 multiply-
  accumulates per clock cycle. More lanes → more throughput.
- **Systolic array** — a grid of PEs that stream data through in lockstep; the classic
  matmul-on-silicon structure.
- **Banking / per-lane banking / wide-word banking** — splitting the weight memory so
  many lanes can each read their own weight *every cycle*. The key trick: store weights
  **transposed** so one wide URAM word (e.g. 1024 bits = 256 nibbles) feeds 256 lanes
  the same column, all sharing one activation. Breaks the "one-bank-per-lane → PE=64"
  ceiling.
- **RLAT (read latency)** — the number of pipeline cycles between requesting a URAM word
  and the data arriving; cascaded URAM can't deliver in a single cycle at speed.
- **Pipeline / latency / throughput** — a pipelined unit produces a result every cycle
  (high *throughput*) but each individual result takes several cycles to emerge (its
  *latency*). The distinction drives the single-stream-vs-batched story.
- **Sequencer / FSM (Finite-State Machine)** — the hardware controller that runs the
  *whole* per-token forward (embed → blocks → head → sample → append-KV → loop) with
  **zero CPU in the loop**. The thing that turns ~100 tok/s into ~10k.
- **CPU-in-the-loop / CPU-out-of-the-loop** — whether the A53 participates in every
  token (driving the fabric over AXI, capped ~100 tok/s) or only loads the prompt and
  drains tokens while the fabric runs autonomously (the path to 10k+).
- **LFSR** — Linear-Feedback Shift Register, a cheap hardware pseudo-random generator;
  used for in-fabric sampling.
- **Operand packing** — fitting more than one MAC into a single DSP by placing two (or
  more) small operands side by side in its wide multiplier (the Xilinx INT8 trick).

---

## Performance & analysis

- **tok/s (tokens per second)** — the headline throughput. Char-level here, so it's
  also characters/second.
- **TTFT (Time To First Token)** — latency from prompt submitted to first generated
  token appearing ≈ (prompt length + 1) × per-token latency. Dominated by prefill.
- **Bandwidth-bound vs compute-bound** — whether throughput is limited by how fast data
  can be *moved* (memory bandwidth) or *multiplied* (MACs). The project's founding
  claim: single-stream decode is normally **bandwidth-bound**.
- **The bandwidth wall** — the shared ~20 GB/s DDR controller. Because the PS and PL
  share it, a DDR-resident model gets no speedup from the fabric; the only escape is to
  keep the model on-chip.
- **Roofline** — the analytical plot of achievable throughput vs model size, showing the
  DDR-bound line, the on-chip band, and where they cross. "A result, not a caveat."
- **Crossover** — the model size (~6.3M params / ~3 MB at INT4) where the weights stop
  fitting on-chip and spill to DDR. Past it, the fabric advantage **collapses** back to
  the wall. Stating the crossover is part of the honesty discipline.
- **On-chip residency** — keeping the entire model in BRAM/URAM so inference never
  touches DDR. The real source of the order-of-magnitude speedup.
- **Arithmetic intensity** — MACs performed per byte of weight read. Low for
  single-stream decode (each weight used once per token); **batching raises it**.
- **Batching / batched-aggregate vs single-stream** — running B concurrent streams so
  each weight read is reused across B tokens, and the hardware units stay busy across
  streams. Aggregate throughput is then set by the *busiest unit*, not the sum of all
  stages — the route to 100k. It's a **serving** number (many users at once), not a
  single-user latency improvement.
- **Serving throughput vs inference latency** — the "two ceilings": how many tokens/sec
  the box can produce across all users (throughput) vs how fast one user sees their next
  token (latency). Batching helps the former, not the latter.
- **Cycle budget** — cycles available per token at a target rate (e.g. 100 µs/token =
  ~30k cycles at 300 MHz). The arithmetic that decides feasibility.
- **GMAC/s / TOPS** — billions / trillions of operations per second; the compute-rate
  units.

---

## Toolchain & method

- **Vivado** — Xilinx's FPGA design suite (synthesis, implementation, bitstream gen).
- **Synthesis / implementation / place / route** — the stages compiling RTL into a
  bitstream: synth → logic; place → assign to physical sites; route → wire them; the
  final timing decides whether it works.
- **OOC synth (Out-Of-Context)** — synthesising one module in isolation to get early
  area/timing numbers, before the full design.
- **Fmax / WNS / timing closure** — the maximum clock the design can run at; **WNS**
  (Worst Negative Slack) is the timing margin (positive = meets timing); "closing
  timing" = WNS ≥ 0 at the target clock.
- **iverilog (Icarus Verilog)** — the open-source simulator used for the fast local
  correctness loop (vs Vivado's slower flow).
- **Testbench (tb) / `$readmemh`** — the simulation harness driving a module with
  stimulus; `$readmemh` loads memory contents (weights, LUTs) from a hex file.
- **bootgen / fpgautil / xmutil** — Xilinx tools to convert a `.bit` to the Kria's
  loadable `.bit.bin`, load it, and manage fabric apps on the board.
- **Tailscale** — the mesh VPN used to reach the board remotely (`ssh ubuntu@…`).

---

## Verification & honesty (the project's rules)

- **Bit-honest before fast** — every step is validated for *correctness* before any
  *speed* number is trusted. The load-bearing discipline.
- **Bit-exact** — the hardware result matches the reference to the last bit
  (`maxabserr = 0`). Used for the integer GEMV path.
- **Cosine (similarity) gate** — for the non-linears, which can't be bit-exact to a
  float reference (exp/rsqrt/erf are transcendental), the gate softens to **cosine >
  0.9999** between the fabric output and goformer's.
- **The token-stream gate** — the strongest and most honest check: given a fixed prompt
  (and seed), the fabric must emit the **identical sequence of tokens** as the reference.
  More robust than raw-logit cosine.
- **Brittle proxy / requant half-boundary** — a finding: on this tiny char model the
  raw-logit cosine is *brittle* — a handful of activations sit exactly on an INT8
  requant rounding boundary, so any discrete approximation flips them and the cosine
  drops, even when the generated text is identical. Hence the token-stream gate is the
  one that matters.
- **MEASURED / DERIVED / PROJECTED** — the tagging convention: a number is either
  measured on silicon (or printed by a tool), derived by arithmetic, or projected
  (needs a synth/board run to confirm). Keeps claims honest about their pedigree.
- **Honest-first** — every claim states where it *loses* (DDR-resident models, long
  context, dev effort, the crossover) up front, not in a footnote.

---

## The tok/s ladder (one-line reference)

| Rung | tok/s | Why |
|---|---|---|
| A53 baseline | ~178 (GEMV bench) / ~11 (chat) | CPU, compute/dispatch-bound |
| PL today (resident, Python) | 0.22 | strangled by the Python AXI driver |
| + software KV cache | ~5 | kills the O(T²) recompute |
| + C/MMIO driver | ~40–126 | the CPU-in-the-loop ceiling |
| single-stream sequencer | ~7–13k | CPU out of the loop |
| batched serving | ~30k–124k | + concurrent streams + parallel non-linears + multi-GEMV-engine |
| on-chip roofline ceiling | ~127k–636k | the bandwidth limit, for reference |
