# KV-to-DDR for 100k tok/s aggregate — bandwidth, burst engine, scheduling, cap

Research note. Anchor (MEASURED, §13): single stream **11,143.9 tok/s @ 200 MHz**, 17,947
cyc/token, P=8 / LANES=256 / TMAX=32, weights resident in 60/64 URAM, BRAM 283/288. The 100k
target is **aggregate** across N≈10–24 keystroke-speculative streams. Weights stay on-chip
(thesis). The new resource that does *not* fit on-chip past ~6 streams is **per-stream KV cache +
scratch** → this note asks whether KV-in-DDR can feed N streams without becoming the new wall.

Every number tagged **MEASURED** / **DERIVED** (arithmetic from a measured figure) / **PROJECTED**
(needs a synth or board run). Today the design uses **no HP port at all** — `build_bd_seq_vec.tcl`
wires only `M_AXI_HPM0_FPD` (a GP-class control master, §BD). A KV-DDR path is **new RTL**: one or
more `S_AXI_HP*` masters from the PL into the FPD DDR controller. That is the central build cost.

---

## 0. KV footprint per stream (DERIVED)

Model: d=256, 4 layers, context window TMAX=32. Per token the cache grows by one position:

- K+V per position per layer = 2 × 256 elements.
- Across 4 layers = 4 × 2 × 256 = **2,048 elements/token**.
- At the activation precision the attention path actually consumes (**INT8** K/V scratch, per
  `vec_attn` ld port = ×32-wide INT8): **2,048 B/token written**, and the *full* resident cache
  at TMAX=32 = 32 × 2,048 = **65,536 B = 64 KB/stream** — matches the task figure.

So per stream: **2 KB/token written** (the new position) and up to **64 KB resident** read-set,
of which attention rereads the live window every token.

KV read traffic per token (what attention must *stream in*, not the steady-state size): each of
the 4 layers reads its K and V for all live positions. At full window that is the same **64 KB
read/token/stream** in the worst case (position 32). Averaged over a 32-token generation it is
~32 KB/token; for a steady-state chat at full window, budget the **64 KB/token** worst case.

---

## 1. Bandwidth budget (DERIVED) vs KV260 HP ceiling

Per-stream KV **read** traffic at full window = 64 KB/token. Aggregate read demand:

| N streams | per-stream tok/s | aggregate tok/s | KV read GB/s (64 KB/tok) | + KV write (2 KB/tok) |
|---|---|---|---|---|
| 10 | 8,000 | 80,000  | **5.12** | +0.16 |
| 10 | 10,000 | 100,000 | **6.40** | +0.20 |
| 16 | 8,000 | 128,000 | **8.19** | +0.26 |
| 16 | 6,250 | 100,000 | **6.40** | +0.20 |
| 24 | 8,000 | 192,000 | **12.29** | +0.39 |
| 24 | ~4,170 | 100,000 | **6.40** | +0.20 |

**Key DERIVED result: 100k aggregate = ~6.4 GB/s of KV read regardless of N** (it is just
100,000 × 64 KB). Write traffic is ~0.2 GB/s — negligible (2 KB/token, the one new position).

**KV260 HP ceiling (realistic figures).** The Zynq UltraScale+ PS exposes the PL→DDR path through
the HP (high-performance) AXI ports on the FPD: up to **4 × S_AXI_HP** + 2 × S_AXI_HPC, each
**128-bit** wide. At a PL-side AXI clock of 250–333 MHz that is a theoretical **4–5.3 GB/s per
port** (128 b × 250 MHz = 4.0 GB/s; × 333 MHz = 5.3 GB/s). The shared DDR4 controller on the
KV260 SOM (32-bit @ ~2400 MT/s) has a raw ceiling of **~9.6 GB/s**, and *measured* sustained PL
read bandwidth across all HP ports on UltraScale+ typically lands at **65–80% of raw** =
**~6–7.5 GB/s aggregate** once refresh, page-miss and PS contention are included (consistent with
the ~20 GB/s LPDDR4 figure CLAUDE.md cites for the higher-end SoM variants; the KV260 commercial
SOM is the 32-bit ~9.6 GB/s part — use the conservative number).

**Verdict on §1:** 100k needs ~6.4 GB/s read; the KV260 DDR delivers ~6–7.5 GB/s sustained
**total**, shared with the PS. **KV-read demand at 100k essentially equals the whole measured DDR
budget.** This is the headline tension: it is feasible only with near-perfect burst efficiency and
a near-idle PS, and it leaves no margin. 80k (N=10 @ 8k) at ~5.1 GB/s is the comfortable target;
100k is the stretch; >128k is bandwidth-impossible on this SOM without shrinking the KV footprint.

**The thesis lever that saves it:** Kevin context is *short*. TMAX=32 is already assumed. The KV
read is linear in window length — a TMAX=16 Kevin window **halves** KV traffic to ~3.2 GB/s at
100k, restoring 2× headroom. The joke (few word do trick) is again the optimisation.

---

## 2. Burst engine: attention as sequential KV reads (PROJECTED RTL)

Attention per head reads its K rows then V rows for the live window — **sequential, contiguous**
if the cache is laid out **[stream][layer][K|V][position][d]** so that one head's window is one
contiguous DDR extent. That is exactly the AXI-burst-friendly access pattern.

- **Burst layout.** Pack each (stream, layer) K block and V block as a contiguous region, row =
  one position (256 INT8 = 256 B = two 128-bit beats). A full TMAX=32 window = 32 positions ×
  256 B = 8 KB per K block, 8 KB per V block → **64-beat INCR bursts** (AXI max burst = 256 beats,
  so a whole K or V block is one burst). 4 layers × 2 (K,V) = **8 bursts/token/stream**, each
  64×128b. This is the ideal AXI profile: long INCR bursts, no per-element address churn.
- **Ping-pong prefetch into BRAM.** Today `vec_attn` reads K/V from a 1-deep BRAM prefetch
  (§9, `qkv_r`/`at_ldv` one register behind the address counter). Extend that to a **double
  buffer**: while head *h* computes from buffer A, the burst engine DMAs head *h+1*'s K/V from DDR
  into buffer B. Attention compute per head is ~5,216 cyc/token / 4 heads ≈ **1,300 cyc/head**
  (§12 profile: load 3,088 + compute 2,128 across the token). A 64-beat HP burst at 250 MHz with
  ~20-cycle first-word latency completes in ~84 PL cycles — **far shorter than the 1,300-cyc
  compute window**, so prefetch fully hides DDR latency *per head*.
- **Latency hiding with 4 heads.** Four heads give four prefetch/compute overlaps per layer; the
  burst engine stays ~1 head ahead. The only un-hidden cost is the **first** head's load per layer
  (cold buffer) ≈ 84 cyc × 4 layers = ~340 cyc/token — small against 17,947. The steady-state
  attention load (currently 3,088 cyc on-chip) is *replaced* by overlapped DDR bursts, so moving
  KV off-chip need not cost cycles if prefetch depth ≥ 2.

**PROJECTED brick:** `kv_dma.sv` — an AXI4 master FSM (addr-gen per (stream,layer,K/V), 64-beat
INCR) + a 2-deep ping-pong BRAM (2 × 256 B × P-wide). Gate it bit-exact against `seq_ref`'s
attention phase by replaying a known KV image from a `.mem` "DDR model" in iverilog before any
board run (the established gate pattern).

---

## 3. Scheduling N streams through one sequencer/GEMM (PROJECTED)

One sequencer, one resident weight image, N streams time-shared. Two viable schedulers:

- **Phase-bucket round-robin (preferred).** The 17,947-cyc token splits into GEMV (12.8k, URAM-
  bound) and non-GEMV (~5.1k: attention, LN, GELU, act-quant, argmax). Run the **GEMV core
  continuously**, and while stream *s* is in GEMV, run stream *s−1*'s non-GEMV phase and stream
  *s+1*'s KV prefetch in parallel. This is software-pipelining the token across streams so the
  expensive shared resource (the L=256 MAC array) never idles. Throughput ceiling = GEMV floor
  (see §4).
- **Simple stagger.** Launch streams offset by ~(17,947 / N) cycles so their DDR bursts and
  BRAM-bank accesses interleave rather than collide. Cheaper to build, leaves the GEMV idle
  between streams — only worth it as the N=2 first step (task #10).

**Extra LUT for stream state.** Per-stream the sequencer must replicate the *scratch* banks
(xres/qkv/ctx/mlp/ln), **not** the weights and **not** the GEMV MAC array. From the fit history
the resident GEMV is cheap (~13k LUT) and the **datapath/scratch is the bulk**. Memory note (§24)
estimates **~32k LUT/stream** if scratch stays in distributed RAM. The fix: push per-stream
scratch into **BRAM banks selected by a stream-id address prefix** (one wide BRAM, stream = high
address bits) — then per-stream cost is *memory depth*, not *logic*, and N scales by BRAM tiles,
not LUTs. With BRAM already at 283/288, **this is the binding on-chip constraint for N**, and it
is why KV (the largest per-stream array) *must* go to DDR — keeping only the live ping-pong window
on-chip frees the BRAM for N-way scratch.

**N ceiling on-chip (DERIVED):** freeing the 64 KB/stream KV from BRAM is what makes N=10–16
feasible; scratch (xres/ctx/mlp ≈ a few KB/stream) fits as BRAM depth. Past ~N=16 the BRAM
displaces the embed ROMs (the 1.5-tile-spare constraint, §9) — that is the on-chip N wall,
independent of bandwidth.

---

## 4. Cap analysis — compute or DDR at 100k? (DERIVED)

- **Compute floor.** GEMV reads 12,800 URAM words/token (irreducible at L=256, §13/status).
  Single-stream cycle floor ~12.8k GEMV + 5.1k rest = 17,947 @ 200 MHz = 11,144 tok/s. The
  **shared GEMV array** is the throughput ceiling for N streams sharing one sequencer: with the
  non-GEMV work overlapped (§3 phase-bucket), aggregate ceiling ≈ 200 MHz / 12,800 cyc =
  **15,600 tok/s × (weight reuse factor 1)**. *That is the problem:* one weight image, read once
  per token per stream, gives **~15.6k aggregate**, not 100k. **To reach 100k the GEMV must
  amortise one weight read across multiple streams** — i.e. **GEMM batching**: read the weight
  word once, MAC it against N stream-activations. The status note's "6.4k cyc / 4 streams (LANES
  128)" is exactly this: batching N=4 into the MAC array cuts effective cyc/token/stream ~4×.
- **With GEMM batch B:** effective aggregate ≈ B × 15,600. B=4 → 62k; **B=8 → ~125k** (the status
  "100k = N=8 @ L=128" line). So **compute reaches 100k only via GEMM batching**, and B≈8 is the
  number. LUT cost = B accumulator sets on the shared weight read (cheap) + B scratch (the §3 BRAM
  problem). DSP: INT4 MACs pack into DSP58s; L=256 used 505 DSP — B parallel accumulators need
  more DSP or more passes (DSP is the next-after-BRAM constraint at high B).
- **DDR floor.** §1: 100k = 6.4 GB/s read ≈ the entire ~6–7.5 GB/s sustained DDR budget.

**Cap verdict:** with GEMM batching B=8 the **compute** ceiling (~125k) clears 100k, so **DDR
bandwidth is the wall, not compute.** The two walls nearly coincide at 100k (6.4 GB/s read ≈ DDR
ceiling), which means the design is **co-limited** — neither side has margin. The honest reachable
number on one KV260 is **80–100k**, with 80k (B=8, N=10 @ 8k, ~5.1 GB/s) as the *reliable* claim
and 100k as the stretch that needs both a near-idle PS and ≥90% burst efficiency, or a TMAX=16
Kevin window to halve KV traffic.

---

## 5. Risks (honest-first)

- **PS contention.** The DDR controller is shared with the A53s. The live chat server
  (`webchat/app.py`, the A53 baseline runs at ~11 tok/s, and the systemd `kevweb` serving traffic)
  competes for the same ~6–7.5 GB/s. At 100k the KV read is *already* the whole budget, so **any
  PS DDR traffic directly subtracts from aggregate tok/s.** Mitigation: pin serving to a QoS-capped
  HP port, or accept that the realistic shared-system number is **80k, not 100k**. This is the
  single biggest reason to claim 80k.
- **DRAM refresh + page misses.** The ~6–7.5 GB/s sustained figure already includes refresh
  (~3–5% overhead) and assumes long bursts. The §2 64-beat contiguous layout is what keeps page-
  miss rate low; a naive per-position or per-head scatter would drop sustained BW to <4 GB/s and
  make 100k impossible. **Burst layout is load-bearing, not an optimisation.**
- **Write bandwidth of new KV.** Only 2 KB/token/stream = ~0.2 GB/s at 100k — negligible, *but*
  the write is on the critical path (the new position must be in DDR before the next token's
  attention reads it). A small write-combine buffer (one position) avoids a read-after-write
  stall; the ping-pong already covers the read side.
- **Burst-efficiency assumption.** The whole §1 budget assumes ~80% sustained. If HP sustained is
  closer to 5 GB/s on this SOM under PS load, the cap is **~78k**, and 100k requires TMAX=16.
- **Bit-honesty.** KV-DDR introduces a DDR round-trip that iverilog must model (a `.mem` "DDR
  image" + latency model) or the async/sync class of bug (§6/§7 of the log) returns. Gate the
  `kv_dma` brick against `seq_ref` attention before any board run.

---

## Go / No-go

**GO — with a calibrated target of 80k aggregate, 100k as a stretch.** The arithmetic closes:
100k aggregate = **~6.4 GB/s** KV read, which sits right at the KV260's realistic ~6–7.5 GB/s
sustained DDR ceiling, and GEMM batching B≈8 pushes the **compute** ceiling to ~125k so DDR is the
single wall. The attention access pattern is naturally burst-friendly (8 × 64-beat INCR
bursts/token/stream), and double-buffered ping-pong prefetch fully hides DDR latency behind the
~1,300-cyc/head compute window, so moving KV off-chip costs ~340 cyc/token, not the 3,088-cyc
on-chip load. The on-chip N ceiling is set by **BRAM** (per-stream scratch displacing embed ROMs
past ~N=16), which is exactly why KV *must* leave BRAM. **The single risk that turns 100k into 80k
is PS contention** — the same DDR feeds the chat server. Recommend: build `kv_dma.sv` + ping-pong,
gate bit-exact vs `seq_ref`, prove **N=2 GEMM batch** (task #10) on-chip first, then add the HP-DDR
KV path at **N=4** and measure sustained HP bandwidth under live PS load *before* committing to the
N=8/100k build. If measured sustained HP < 5 GB/s under load, drop to **TMAX=16** (halves KV
traffic, pure thesis) rather than chasing the bandwidth — the short Kevin context is the escape
hatch.
