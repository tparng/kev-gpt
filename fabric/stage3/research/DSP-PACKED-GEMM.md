# DSP-packed batch GEMM — the path from 11.1k single-stream to ~70k aggregate

**Status:** RESEARCH / design note. No RTL yet. Numbers tagged MEASURED (committed silicon),
DERIVED (arithmetic from those), PROJECTED (needs OOC/board).

## 0. Where we start (MEASURED, §13 of WIDE-WORD-DATAPATH-LOG)

- Single stream: **11,143.9 tok/s @ 200 MHz, 3/3 bit-exact**, CYCLES = 17,947.
- Cycle floor = GEMV reading **12,800** 1024-bit URAM words/token. `LANES=256` INT4×INT8
  MACs done in **LUTs** (~32k LUT), the wide-word resident core in `gemv_banked_resident_vec.sv`.
- Fit: 74k LUT (63%), BRAM 142.5/144, **URAM 56–60/64**, **DSP 505/1248** (act-quant +
  non-linears; the GEMV MACs are LUT, not DSP).
- KV260 limits: 117k LUT · 144 BRAM tiles · 64 URAM · **1248 DSP**. Silicon/STA factor ≈ **1.6**.

The single stream is **latency-bound**: 12.8k weight reads is the irreducible serial cost of
streaming the whole INT4 image once per token. You cannot beat it by going faster per token —
you beat it by amortising those *same 12.8k reads across multiple streams*. That is batch GEMM:
read each weight word once, multiply it into **N** activation vectors. The weight bandwidth is
already paid; the extra streams are nearly free *if you have MAC units to spend* — and we have
**~740 idle DSPs**.

---

## 1. DSP48E2 INT4×INT8 packing — the multiplier math

The DSP48E2 multiplier is **27 × 18 (signed)** feeding a 48-bit post-adder/accumulator. The
established trick (Xilinx **WP487**, "Deep Learning with INT8 Optimization", and the
DSP-Packing literature, arXiv:2203.11028) is to compute **two products that share one operand**
in a single multiplier by packing the two non-shared operands far enough apart in the wide port
that their products land in non-overlapping fields of the 45-bit product.

### The classic INT8×INT8 = 2-per-DSP result (WP487)
Pack two 8-bit weights `w0, w1` into the 27-bit port with a large shift, share one 8-bit
activation `a` in the 18-bit port:

```
A:D port (27b) = w1 << K  +  w0          (K large enough to separate)
B port  (18b)  = a
product (45b)  = w1*a << K  +  w0*a       # two INT8 MACs, one shared operand
```

The two 16-bit products must not collide; WP487 uses the 27-bit width to give a wide gap and
notes that with the leftover guard bits **up to 7 terms** accumulate before the lower field can
corrupt the upper. One input ≥ 24b and a 32b accumulator field are the stated minimums.

### Adapting to **INT4 weight × INT8 activation** (our case)
Our weights are **INT4** (4-bit), activations **INT8**. This is *more* favourable than INT8×INT8
because the weight is narrower, so two weight-products are smaller and pack with a bigger guard.

- Product of `int4 × int8` is signed **12-bit** (range −1024..1016; |max| < 2^11, fits 12b incl.
  sign). To accumulate K terms without the lower field overflowing into the upper, the field
  needs `12 + ceil(log2 K)` bits. Kevin's longest reduction is **K ≤ 1024** → `+10` bits →
  **22-bit accumulation field** per packed lane.
- Separation gap: place the upper weight's product so its LSB sits **above the top of the lower
  field**. With a 22-bit field, shift the second weight by **K_shift = 22** (round up to a clean
  value; 24 leaves 2 extra guard bits — the "2-bit remaining → 7 terms" margin in the literature,
  but we don't need that headroom since our field is sized for the full K=1024). Layout in the
  27-bit port:

```
A:D (27b) =  w_hi << 24      |  w_lo            # two INT4 weights, gap of (24 - 4)=20 bits clear
B   (18b) =  a               (INT8 activation, sign-extended)
P (≈ 4+8+24 = 36b used)   = (w_hi*a) << 24  +  (w_lo*a)
                              └ upper field ┘    └ lower 24b: 12b prod + 12b accum headroom ┘
```

  24-bit shift comfortably exceeds the 22b the lower field needs at K=1024, so **no carry
  corruption** and **no per-term correction term** is required in the common case (both weights
  positive-or-handled-by-sign-extension). The only correction is the standard **signed-pack
  artifact**: when `w_lo` is negative its sign bits ripple up; the textbook fix is to add a
  one-time correction `−(sign(w_lo) ? a << 24 : 0)` once at accumulation end (a single subtract
  per output, not per MAC), or pre-bias weights to unsigned and subtract the column-sum of `a`
  afterwards. Both are O(1) per output row, negligible against 12.8k reads.

### MACs per DSP per cycle
- **Conservative / recommended: 2 INT4×INT8 MACs per DSP per cycle** (the WP487 weight-share
  scheme above, verbatim but with INT4 weights). This is the number I size the design on — it is
  the proven, low-risk packing that closes timing and needs no exotic pre-adder gymnastics.
- **Aggressive (future): 3 INT4×INT8 per DSP** by also packing two activations using the 27-bit
  pre-adder, as in the DSP-Packing paper (it reports ~3 for INT4×INT8, 6 for INT4×INT4). This
  needs cross-product correction terms and tighter timing; treat as a stretch lever, not the plan.

**Plan assumes 2 MACs/DSP/cycle.**

---

## 2. 740 free DSPs → extra MAC lanes and N for 70k

The batch-GEMM idea: the existing **LANES=256 LUT MACs serve stream 0** (the proven datapath,
untouched). Each *additional* batch stream gets its own **256-lane MAC bank built from DSPs**,
where each DSP does **2 INT4×INT8 MACs/cycle**.

- DSPs to give one extra stream a full 256-lane bank, **if each DSP did 1 MAC**: 256 DSP/stream.
  With **2 MACs/DSP** via weight-sharing we instead share *one weight word across two streams*:
  one DSP holds the INT4 weight nibble and multiplies it by **stream-i act** and **stream-j act**
  packed in the 27-bit port → **one DSP serves the same lane for 2 streams**.

  ⇒ **256 / 2 = 128 DSPs per *pair* of streams per 256-lane bank.**

- Budget after keeping current 505 DSP (act-quant/non-linears stay): **~740 free**.
  - `740 / 128 ≈ 5.8` → **5 stream-pairs = 10 DSP-batch streams** at 128 DSP each = **640 DSP**,
    leaving ~100 DSP slack for fanout/pipeline and the per-stream act-quant/dequant copies.
  - Plus the LUT stream-0 → **N_total = 1 (LUT) + 10 (DSP) = 11 streams**. Round the design knob
    to **N = 8 DSP streams (512 DSP) + 1 LUT = 9**, or push **N = 10 DSP + 1 = 11** if routing
    allows. (DERIVED.)

### Throughput map
Weight reads stay **12.8k/token shared across all N streams** (read once, fan out). The
per-token cycle count is unchanged (~17.9k, dominated by the shared GEMV read + the still-serial
non-linear/attention tail), but it now **emits N tokens per token-time**:

- `aggregate = single_stream_tok_s × N`
- N = 8 → **11,144 × 8 ≈ 89k** (DERIVED, optimistic — assumes tail fully shared)
- More honest: the non-linear/attention tail (~5k cyc) is **not** all shared and may serialise or
  need replication; budget a ~0.7–0.85 efficiency. N=8 × 0.8 → **~71k tok/s aggregate
  (PROJECTED)** — squarely the ~70k target.

**Recommendation: N = 8 DSP-batch streams (512 DSP @ 2 MAC/DSP) + the existing LUT stream → 70k.**
N = 6 is the safe fallback (~53k, ~384 DSP) if routing/BRAM bites; N = 10 the stretch (~89k).

---

## 3. SIMD mode — alternative for the INT4 accumulation tree

DSP48E2 ALU supports **SIMD = FOUR12 (four independent 12-bit add/sub) or TWO24**. This is *not*
a multiplier mode — it splits the **post-adder** into lanes. Relevance to us:

- Our packed product is a signed **12-bit** int4×int8 result → **FOUR12 SIMD** can accumulate
  **four 12-bit partial sums in parallel in one DSP's ALU** without using the multiplier at all.
  Use case: an **INT4 reduction/accumulation tree** — e.g. summing four lanes' pre-computed
  products, or a 4-way adder-tree stage feeding the wide accumulator, at 1 DSP per 4 adds instead
  of LUT carry chains. §10 of the log already folded 9 carry-chain adders into DSPs; FOUR12 is the
  principled way to do more of that.
- Caveat: FOUR12 with only **12-bit lanes overflows after a few adds** (no inter-lane carry), so
  it suits *short* trees / final reduction, not the long K=1024 accumulation (which needs the
  full 48b or the 22b-field packing of §1). **Use FOUR12 for the adder-tree / partial-sum
  merge; use the multiplier-pack of §1 for the MAC core.** Don't conflate the two.

---

## 4. URAM bandwidth — is one read/cycle enough for the batch?

**Yes, and this is the whole reason batch GEMM is free.** The weight image is read **once** and
broadcast to all N streams:

- Weights resident as **60 banks × 72b @ 200 MHz** (the §11 dense URAM packing, 1024-bit wide
  word reconstructed from 15 banks × 4 cascade). One wide word/cycle = **256 INT4 weights/cycle**
  = 12,800 words/token. **MEASURED** as sufficient for one stream.
- Batch streams **consume the same word the same cycle** — the URAM read port is shared, the fan
  is in the *activation* dimension (each stream brings its own INT8 acts from its own small
  scratch BRAM, not from URAM). So URAM bandwidth is **independent of N**: one read/cycle feeds
  N streams.
- URAM raw ceiling: 64 URAM × 72b × 200 MHz ≈ **920 Gb/s**; we use 60 banks → ~864 Gb/s, already
  committed. **Adding streams adds zero URAM traffic.** The constraint moves entirely to **DSP
  count and activation-scratch BRAM** (N copies of the per-stream act/KV buffers — watch the
  142.5/144 BRAM ceiling; this is the likely real limiter, see §5).

---

## 5. Risks

1. **BRAM, not DSP, is the tight resource.** We are at **142.5/144 BRAM tiles**. Each batch
   stream needs its own activation scratch + KV cache. The Kevin short-context thesis is what
   saves us: **TMAX=32** (the Kevin attention window) keeps per-stream KV tiny. Even so, N copies
   of x/qkv/ctx scratch will overflow 144 tiles unless per-stream scratch is shrunk or packed into
   URAM headroom (4 spare URAM) / LUTRAM. **This is the #1 build risk — model it before RTL.**
2. **Routing congestion across DSP columns.** 512+ DSPs spanning multiple DSP columns, all fed by
   the *same* broadcast weight word, is a high-fanout net. Mitigation: **register-replicate the
   weight word per DSP column** (pipeline the broadcast, +1–2 cyc latency, negligible vs 17.9k),
   and floorplan one batch-stream bank per DSP column (`Pblock`s) so each column is local.
3. **200 MHz across DSP columns.** Single-stream already closes 200 MHz silicon (factor 1.6 over
   ~125 MHz STA). DSP MAC pipelines are *easier* to time than LUT MACs (the multiplier + P-reg are
   hard silicon). Keep the WP487 pack fully registered (A/D reg, B reg, M reg, P reg = 4 stages);
   the cost is +4 cyc latency, irrelevant. **Risk is the broadcast net and the wide accumulator
   readback, not the multiply.**
4. **Packing correctness.** The signed-INT4 sign-extension correction (§1) must be gated
   bit-exact against `seq_ref.py` exactly like every prior block — add a `run_vec_dsppack.py`
   gate before any synth. Bit-honest before fast.
5. **Tail amortisation.** If softmax/LN/attention don't share across streams, aggregate efficiency
   drops below N×. Either replicate the (cheap, mostly-LUT) non-linears per stream or interleave
   streams through one non-linear unit. Budgeted as the 0.8 efficiency factor in §2.

---

## 6. Recommendation (one line)

Keep the LUT GEMV as stream 0; add **N = 8 DSP-batch streams** built from **512 DSP48E2 @ 2
INT4×INT8 MACs/DSP/cycle** (WP487 weight-share pack, 24-bit separation, K≤1024 needs a 22-bit
field), broadcasting the **same 12.8k-word/token URAM read** to all streams. Watch **BRAM
(142.5/144)** as the true limiter — lean on TMAX=32. **PROJECTED ~70k tok/s aggregate** at 200 MHz.

## 3-per-DSP: proven scheme (or impossibility)

**Verdict (PROVED, arithmetic): exact 3-INT4×INT8-per-DSP with a shared activation is
IMPOSSIBLE on the DSP48E2 27×18 multiplier. The exact ceiling is 2.0 MACs/DSP.** The
proof is `fabric/stage3/research/dsp3_pack_proof.py` — a bit-accurate DSP model plus
610,756 zero-mismatch verifications. The N=24 target (which needs 3/DSP) is therefore
not reachable; the DSP-batch ceiling at LANES=128 on 1,024 DSPs is **N = 16**.

The brief asked for the shape "one INT8 activation `x` shared by three INT4 weights
`w0,w1,w2`", packed so the 27-bit signed weight operand `w0 + w1·2^g + w2·2^(2g)` holds
all three, with recovery via overlap-correction or periodic drains. It fails on **two
independent walls**, either of which alone is fatal:

### Wall 1 — the operand port is one bit too narrow (blocks even K=1)

A single `int4 × int8` product is **12 bits** (signed: range −1920..1905; a negative
product's sign bits ripple the full field width). To keep field 0 from bleeding into
field 1 in even **one** cycle, the gap must be `g ≥ 12`. But three nibbles at gap `g`
put the top weight's MSB at `2g + 4`, and the 27-bit *signed* operand allows the packed
value `15·(1 + 2^g + 2^{2g}) < 2^26`, i.e. `2g ≤ 22 → g ≤ 11`.

> No-bleed needs `g ≥ 12`. Operand-legal needs `g ≤ 11`. **11 < 12 → no gap works.**
> Equivalently: three 4-bit nibbles with 12-bit gaps need `2·12 + 4 = 28` bits, but the
> port is 27. The A+D pre-adder does not help — it feeds the **same** 27-bit multiplier
> input, so the truncation-to-27 wall is unchanged.

`dsp3_pack_proof.py` demonstrates this concretely against the bit-accurate model at the
largest legal gap `g = 11`: packing `w0'=15`, activation `x = −128`, the field-0 readback
is `128`, not the intended `−1920` (`clean=False`) — the product has bled.

### Wall 2 — three 22-bit neuron sums don't fit a 48-bit accumulator (blocks full K)

Even if the operand fit, the three results `y_i = Σ_k w_i·x_k` are three **distinct
output neurons**, each a 22-bit signed value at K=1024 (`|y| < 1024·1024 < 2^21`).
Three of them carry `3 × 22 = 66` bits of independent information; one 48-bit accumulator
holds 48. **66 > 48** — no overlap-correction can recover them, because a per-*stream*
side scalar (the only cheap side state, like `sum_act`) carries `O(log K)` bits, not the
22 bits each that three *distinct* neurons need. (Side state that *could* separate them
would have to be per-(lane,DSP), which is exactly the cost the pack is meant to avoid.)

### The drain escape, and why it also collapses here

The literature's route to ~3/DSP is **periodic drains**: fold the 48-bit acc into three
separate 24-bit fabric accumulators every `D` cycles, so each field only has to hold a
`D`-cycle window sum. Exactness needs three non-overlapping fields each wide enough for
the window sum (`3825·D < 2^w`), with `3w ≤ 48 → w ≤ 16`. That alone would allow `D ≤ 17`
(already marginal — the brief notes `D ≥ 64` to amortize fabric cost). **But Wall 1 caps
the gap at `g ≤ 11`, so the usable field width is `w = 11`, giving `3825·D < 2^11 = 2048
→ D = 0`.** There is *no* valid drain window. The search over `g ∈ [8,16]` in
`max_drain_period_3()` returns `D = 0`. (If the operand port were 28 bits, the drain
scheme would work at `D = 16` with three 24-bit fabric accs and +1 add/lane every 16
cycles — recorded as a counterfactual only.)

### 2.5-per-DSP hybrid (5 weights / 2 DSPs) — also blocked

- **Independent accumulators:** each DSP exact-packs at most 2 weights (Walls 1+2), so two
  DSPs give 4 weights = **2.0/DSP**, not 2.5.
- **Cascade** (DSP-A `P → DSP-B PCIN`, two multiplies summed into one 48-bit acc): even if
  each multiply 2-packs, that is **4** distinct neuron sums in one 48-bit acc → `4·22 = 88
  > 48` (Wall 2) → not separable at full K; and the per-window drain collapses for the same
  `g ≤ 11 < 12` reason as the 3-pack. No 2.5/DSP path is exact. **Ceiling stays 2.0/DSP.**

### Verification counts (verbatim from the proof run)

```
[2-per-DSP verification against the bit-accurate DSP model]
  (a) exhaustive K=1  (all w0,w1 x sample x) ... mismatches=0/10752
  (b) randomized K=1024 (600000 pairs = 1200000 lanes) . mismatches=0/600000
  (c) adversarial corners ...................... mismatches=0/4

DSP3_PROOF scheme=3-per-DSP-INT4xINT8-shared-act result=IMPOSSIBLE(operand27<28 & info66>48) fallback=2.0-per-DSP(22b-gap,debias) mismatches=0/610756 OK
```

The 2-per-DSP scheme re-verified here uses biased-unsigned weights `w' = w+8` (0..15) and
biased-unsigned activations `x' = x+128` (0..255), packed as `operand = w0' + (w1' << 22)`
(max `15 + 15·2^22 = 6.29e7 < 2^26`, fits 27-bit signed). The two biased field sums split
positionally (no overlap, since each field sum `< 2^22` at K≤1024), then de-bias exactly via
`y_i = SUM_i' − 128·Σw_i − 8·Σx − 1024·K`, where `Σw_i` is one per-lane scalar and `Σx` one
per-stream scalar (the proven `sum_act`-style correction, generalised to also de-bias the
weight). Recovery cost per lane-pair: 1 mask + 1 shift + 3 mul-sub. **No drain, no per-MAC
correction.** This matches the committed bit-exact silicon scheme.

### Implied stream budget (the headline)

| rate (exact) | DSP/stream @ LANES=128 | N on 1,024 DSPs | status |
|---|---|---|---|
| 3/DSP | 128/3 ≈ 43 | **24** | **IMPOSSIBLE** (Walls 1+2) |
| 2.5/DSP | 128/2.5 ≈ 51 | 20 | **IMPOSSIBLE** (cascade info wall) |
| **2/DSP** | **64** | **16** | **PROVEN bit-exact** |

> **The N = 24 plan is not reachable via DSP packing.** At the proven 2.0 MACs/DSP the hard
> ceiling is **N = 16** DSP-batch streams on 1,024 DSPs at LANES=128. To exceed it you must
> either add LUT-MAC streams (as today's stream 0), shorten LANES, or change the operand
> precision (INT4×INT4 *can* reach higher pack rates — but that is a different activation
> precision than Kevin's INT8 acts).

## Sources
- Xilinx WP487, *Deep Learning with INT8 Optimization on Xilinx Devices*
  (https://docs.amd.com/api/khub/documents/z7yAy_aweTmRYkGaTVyhbw/content)
- *DSP-Packing: Squeezing Low-precision Arithmetic into FPGA DSP Blocks*, arXiv:2203.11028
  (https://arxiv.org/pdf/2203.11028)
- *Revealing Untapped DSP Optimization Potentials for FPGA-Based Systolic Matrix Engines*,
  arXiv:2409.03508 (https://arxiv.org/html/2409.03508v1)
- DSP48E2 SIMD modes: AMD/Xilinx UG579 (DSP48E2 User Guide); MicroZed Chronicles SIMD article.
