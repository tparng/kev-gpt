# Past the stream ceiling: where the levers become cycles and clock

This is doc 6, and it is the chapter the design docs (0–5) could not write, because they were
written before the build. Doc 2 left off at the plan: bake the INT4 transformer fully on-chip,
hand-roll the non-linearities in fabric next to a systolic GEMV, validate to bit-exact before
trusting a number, and report the roofline crossover. That plan is now done and running on
silicon. This doc picks up *after* the model is generating Kevin-speak with the CPU out of the
loop, when the question stopped being "can we get it on-chip at all" and became "how much more N
can the silicon give." Everything below is MEASURED on the KV260, token-stream bit-exact against
the integer reference `seq_ref`, 3/3 deterministic runs, unless tagged DERIVED or PROJECTED. The
primary source is the engineering log, `fabric/stage3/WIDE-WORD-DATAPATH-LOG.md` §12–26; the
section numbers below cite it.

## The stream ceiling

Once the single-stream sequencer was correct and fast, the throughput lever was *batching*.
A single resident-weight GEMM pass reads the whole ~12.6 Mbit INT4 weight image out of URAM
exactly once; running N streams through that one pass amortises the read across N tokens. That is
the entire reason batching wins here — the weight bandwidth is the floor, and more streams divide
it. The ladder followed it: N=4 (16,969.3 tok/s @166.7), N=8 ping-pong (17,740.6), N=8 single-pass
(19,275.6), N=16 (24,134.0), all bit-exact, all on silicon (§14–17).

N=16 is where batching stops. Not for want of trying — it is a proven hard ceiling at LANES=128.
The packing is 2.0 INT4×INT8 MACs per DSP, and three-per-DSP is impossible on two independent
walls: three no-bleed nibbles need 28 bits against the DSP's 27-bit port, and three K=1024 neurons
hold 66 bits of accumulator state against a 48-bit accumulator. `research/dsp3_pack_proof.py`
falsifies the 3.0 scheme over 1.2M randomized K=1024 lane-products, exhaustive K=1, and adversarial
corners — 0 mismatches on the proven 2.0 scheme, and the 3.0/5-per-2 variants die the same way
(§18). LANES=256 would need 2,048 DSPs, also dead. So past the N=16 record — pushed to 25,744.5
tok/s @166.7 by the softmax-latency cut (§18) — **more streams was over.** The two remaining levers
are CYCLES (fewer cycles per token) and CLOCK (more MHz). The rest of this doc is those two.

## Split-brain: the one big cycle play

The single weight-read port serialised everything: one GEMM engine, one pass at a time. But the
KV260's URAM is genuinely dual-port, and an `xpm_memory_tdpram` in "ultra" mode maps both ports
independently — 56 URAM, 0 LUT, 0 BRAM, proven (commit `2f3ba17`). That unlocks the structural
move: run **two independent N=8 cohorts**, each reading the same resident weight image through its
*own* port, sharing only the weight image plus an arbitrated LayerNorm, attention, embed and
dequant channel. A cohort never shares a weight pass, so the whole stream-desync problem that
haunted the single-pass merge simply dissolves (§19).

It paid. The split-brain N=14 variant (NC=7, the one that fit after the LUT campaign) measured
**36,970.7 tok/s @166.7** — a +43.6% record and the first of this era (§20). The N=16 variant
(NC=8) fit later, once the AQ-multiplier was range-narrowed and attention un-evicted from fabric
(§21).

## The lever campaign

After split-brain, progress was a disciplined loop: profile to find the current timing-critical
path or the biggest idle wedge, retire it bit-exact, re-gate, repeat. The MEASURED silicon records,
in order:

| tok/s | clock | cyc (silicon) | config | what cleared it | tag |
|---|---|---|---|---|---|
| 24,134.0 | 166.7 MHz | 110,494 / 16 tok | N=16 merged | N=16 fit campaign | MEASURED §17 |
| 25,744.5 | 166.7 MHz | 103,582 / 16 tok | N=16 | softmax-latency cut | MEASURED §18 |
| 36,970.7 | 166.7 MHz | 63,113 / 14 tok | N=14 split-brain | dual-port URAM cohorts | MEASURED §20 |
| 46,604.4 | 200 MHz | 68,663 / 16 tok | N=16 split-brain | first 200-clean build | MEASURED §21 |
| 56,262.7 | 200 MHz | 56,876 / 16 tok | N=16 split-brain | schedule-pipelining wave | MEASURED §24 |

The two jumps to 200 MHz and beyond are worth naming. The **46,604.4** build (§21) was the first
to close 200 MHz silicon clean, and it took three composed timing levers: un-retiming the LayerNorm
sum-of-squares path so its barrel-shift could not fuse with the Newton squarer (the path that had
been failing at WNS −1.876); a 32×48 range proof on the activation-quant multiplier
(`research/aq_range_proof.py`) that freed enough DSP to **un-evict attention** from fabric back into
DSPs; and an attention operand-register split. The **56,262.7** build (§24) was the
schedule-pipelining wave: AQ/RUN overlap, stream-granular NL/GEMM readback overlap, and the
per-call `vec_attn` cuts, composed.

The cycle campaign ran ahead of silicon, all bit-exact in sim: **71,441 → 66,285** (AQ/RUN overlap:
row-major act-quant plus a stall-guarded early GEMM start) → **61,245** (stream-granular NL/GEMM
readback overlap, §22) → **57,149** (`vec_attn` per-call cuts: V-load overlapped with score+softmax,
ctx-emit fused into the final accumulate, §23) → **53,565** (CTX cross-group streaming, §25) →
**51,892** (TMAX 32→16 + per-cohort attention un-share + CTX cross-group, composed, §26). The 56.3k
record is the last build that reached silicon; the sim cuts below it are gated and queued.

### Name the dead ends

The honesty is the point, so the dead ends get named with the records. Two agent rungs returned
HONEST STOPs that now bound the schedule space (§24). The **LN→AQ** boundary overlap is
arithmetically dead: the lockstep GEMM consume gates RUN-start on the slowest stream's LayerNorm,
so there is no per-stream wavefront to exploit on the consume side. The **attention→PROJ** overlap
is dead for the same shape. And the arbiter-fairness lever measured **exactly Δ0** — the shared-attention
cost is pure serial throughput of the one shared `vec_attn` unit, not a priority artifact, so
re-prioritising it buys nothing (§23, §24). Schedule overlap is exhausted at this topology; the road
on from there is architectural, not a scheduler tweak.

## The silicon margin

STA on the `-2LV` part is pessimistic, and this whole campaign leans on that gap honestly. Designs
that close ~70–85 MHz in static timing have run bit-exact at 125–200 MHz on silicon — an observed
factor of roughly 1.3–1.76× across builds. So the policy is: build at a clock that *closes* in STA,
then find the real ceiling with the board `--fclk` sweep, withholding the tok/s claim unless the
token matches `seq_ref` 3/3. The PLL snaps to 1000/N MHz (…125, 142.9, 166.7, 200, 250…), so a
build either clears a rung or it does not; there is no fractional headroom to claim.

One small, repeatable detail backs the discipline: the **SETTLE signature**. Silicon runs a few
hundred cycles fewer than sim (−297 or −273, depending on call count) because a sim-only settle
state is skipped on the real fabric. That gap has held for five consecutive builds, predicted ahead
of each one (§16–24). It is not a headline; it is the kind of small thing that, when it keeps
coming out exactly right, means the model of the silicon is honest.

## Where the context went

The 53,565 → 51,892 cut (§26) was not free, and the trade is explicit. TMAX dropped from 32 to 16 —
the on-chip attention window halved to 16 positions — and that freed the BRAM tiles that funded
**per-cohort attention** (deleting the shared-`vec_attn` arbiter, which the §23/§24 STOPs had named
as the serial wall). It is a documented context-for-cycles trade, not a silent regression: the
board driver `pl_seq_sb` gained a `--tmax` flag, and the embed upload must match the build TMAX or
positions past 0 corrupt.

The restore path is real and already gated. The KV-DDR stack (`kv_dma` + `kv_prefetch`,
sim-complete, bit-exact, P=8 wide-emit) moves the KV cache to DDR with a double-buffered prefetch
that fully hides DDR latency behind the per-head compute window. The bandwidth math is DERIVED in
`research/KV-DDR-100K.md`: a full-window 100k-aggregate KV read is ~6.4 GB/s; a TMAX=16 Kevin window
halves that to ~3.2 GB/s; K4/V4 + Hadamard KV quantization (MEASURED at +0.72% NLL for 1.78×
compression in `model/exp_kvarn.py`) shrinks it further. All of those sit under the ~6–7.5 GB/s
sustained HP-port ceiling the same doc measures — so DDR is **not** the binding wall at 100k for the
short Kevin context. The thesis survives intact: telegraphic Kevin-speak keeps real prompts inside
small windows, which is exactly what keeps the KV traffic affordable. The dumbness is still the
optimisation.

## The 100k identity, honestly

The target factors cleanly: **100,000 tok/s = 16 streams × 250 MHz / 40,000 cyc.** Three knobs,
each with an honest status:

- **Streams: maxed at 16, proven.** The 2.0/DSP packing is the wall (§18); nothing more to get here.
- **Cycles: 53,565 gated in sim (§25), 51,892 with the architectural wave composed (§26).** The
  path to ~40k is designed and profiled, not walked. The irreducible floor is GE_WAIT, ~19k cycles
  of MAC compute that cannot be removed. The rest of the gap is shrinking readback and the last
  attention residue — both mapped, neither built to silicon yet.
- **Clock: 200 MHz MEASURED (§21, §24); 250 MHz is the open question.** Reaching it needs a
  post-route-MET 5 ns build so silicon's observed ~1.3× margin covers the 1.25× ask. Not guaranteed.

State it plainly: **56,262.7 tok/s @200 MHz is MEASURED and current. 100k is PROJECTED** and needs
*both* the cycle floor near 40k *and* 250 MHz silicon — two uncertainties, neither guaranteed, both
mapped. 100k is not done, and this doc does not claim it.

## Where it stands, what is left

The model runs entirely in fabric, CPU out of the loop, generating bit-exact Kevin-speak across 16
concurrent streams at **56,262.7 tok/s @200 MHz, MEASURED** — about +118% over the 25,744.5 record
in roughly thirty hours of the lever campaign, and far past where doc 2's plan stopped. The stream
ceiling is hit and proven; the remaining road is cycles and clock, both inside the same bit-honest
gate ladder that got us here.

Two genuine uncertainties remain, and they are the whole of the gap to 100k. First, **250 MHz on
silicon** — the cycle records that reach it in PROJECTED tok/s all assume a 250 MHz build the timing
campaign has not yet produced clean. Second, **the last ~13k cycles** from the 53,565 gated floor
down toward ~40k, which is designed and profiled but not yet built to a bitstream. Both are mapped;
neither is walked. That is the honest line: the stream ceiling is behind us, the cycle and clock
floors are in front, and 56.3k is the number that is actually true today.
