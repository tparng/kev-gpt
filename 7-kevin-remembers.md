# Part 7, Kevin remembers — the faithful-stream campaign

**Target: 20,000 tok/s *average* on N=1, T=160, model-faithful decode.** A 160-token
reply — an actual message, with the whole story in attention — generated on the
fabric in under 10 ms. One stream that remembers, instead of sixteen that don't.

## 1. The confession this campaign starts from

The 59,965.5 record (doc 6 §27) is sixteen streams of **attention T=1**. That is not
an implementation accident discovered late; it is hardwired in every N≥4 design:
`nl_engine.sv` drives `at_tcount_o <= 9'd1`, `sequencer_vec.sv` likewise, and the
silicon agrees — `fabric/stage3/board/test_sb_kvwin.py` (MEASURED 2026-06-10) drove
the record bitstream position-by-position and got `match=False` with a constant
53,364 cyc/pass at every position. No KV survives a GO. The chat built on it emits
one faithful character and then degenerates ("he he he he").

The only designs that ever decoded faithfully are the early single-stream sequencers
(`sequencer.sv` / `sequencer_fast.sv`, `sm_tcount <= tpos`): real text, T up to 32,
**751.8 tok/s MEASURED** top. Everything from the vec rung (1,882.7) through the 60k
record bought speed by freezing T=1. Nobody ever had both. This campaign builds both
— at N=1, where the chip has room for it.

## 2. The identity

```
20,000 tok/s average  =  ≤10,000 cyc/tok average @ 200 MHz
                      =  ≤12,500 cyc/tok average @ 250 MHz
```

"Average" is over a 160-token generation with the window growing 1→160, so the mean
attention span is T̄ ≈ 80. Worst case (window full, T=160) is reported separately —
both numbers get MEASURED, neither gets hidden.

## 3. The cost model (DERIVED from measured rungs)

Per token, every weight is read once: **3,195,136 MACs** (4 × 786,432 + head 49,408).

| component | cycles, today | the lever | cycles, after |
|---|---|---|---|
| GEMV (LANES=256) | 12,481 | second URAM port → LANES=512 | ~6,241 |
| attention (P=8, T̄=80) | 20,608 avg / 40,960 worst | widen to P=64 (P=128 stretch) | 2,576 avg / 5,120 worst |
| LN/GELU/AQ/softmax/control | ~5,466 (= 17,947 − 12,481, from the @200 rung) | P-wide non-linears (P=8→32) with the freed fabric | ~2,500 |
| **total avg** | — | — | **~11,317 @ 200 = 17,672 · @ 250 = 22,093 ✓** |

KV cache at T=160, N=1, **INT8**: 2 × 4 layers × 256 × 160 = **327 KB** — fits the
URAM left over after the 1.6 MB weights (2.3 − 1.6 = 0.7 MB) without touching BRAM
scratch. KV write traffic is 2 KB/token — noise.

## 4. The rungs (each ships a MEASURED, bit-exact number)

**R0 — the reference.** `goformer_kv` already proves KV decode ≡ full recompute.
New: an **INT8-KV** fixed-point reference (`goformer_kv8` + the spec pin in the
`goformer_fixed` style) and a model-level NLL gate by the KVarN method — K4/V4 was
already a GO at +0.72% NLL, so K8/V8 must pass with margin or the campaign stops
here. This changes the golden contract; the new reference becomes the gate for
every rung below.

**R1 — make it remember.** Per-layer K/V banks + write-at-`pos` + `tcount = pos+1`
in the vec-lineage datapath; TMAX=160 pos_emb upload. Gate: `run_vec_kv.py`, token
streams equal to R0 over full 160-token generations, then board.
→ ~38.5k cyc avg = **~5,200 tok/s avg @ 200 (PROJECTED)**. The chat becomes real at
this rung — everything after is speed.

**R2 — wide attention.** `vec_attn` P=8→64 (elementwise MAC; INT8 KV halves the
read bandwidth that pays for it). → ~20.5k cyc = **~9,700 (PROJECTED)**.

**R3 — dual-port GEMV.** At N=1 the true-dual-port URAM's second port — the one
split-brain spent on cohort 2 — is free. Stripe weight reads across both ports:
LANES=512, GEMV floor halves. Both ports at 200 MHz are *silicon-proven by the
record itself*. This is **not** the dead clock-domain double-pump
(DOUBLE-PUMP-100K.md); no second clock exists here. → ~14.3k cyc = **~14,000
(PROJECTED)**.

**R4 — P-wide non-linears.** LN/dequant/AQ at P=32; the N=1 design frees ~15/16 of
the MAC fabric, so area is not the constraint it was at 98% density. 5,466 → ~2,500.
→ ~11.3k cyc = **~17,700 @ 200 (PROJECTED)**.

**R5 — the clock.** 250 MHz on a design a fraction of the SB's density (the 6 ns SB
build carried ~1.6× silicon margin; the blocker at 250 was routing density, which
N=1 removes). → **~22,100 avg / ~18,000 worst-case (PROJECTED) — target met.**
*Contingency if 250 won't close:* 214.3 MHz (next PL divisor) + attention P=128
(−1,288 cyc) → ~10.0k cyc = **~21,400 avg** — the target survives losing the
headline clock.

## 5. Already proven dead — do not re-propose (doc 6 / log §19–26)

LN→AQ schedule overlap · attention→PROJ overlap · 3 INT4×INT8 MACs/DSP
(`dsp3_pack_proof.py`) · the 2× clock-domain double-pump. The levers above are
chosen to be none of these: R3 is port-parallelism (proven), R2/R4 are width
(area-for-cycles, and N=1 has the area), R5 is the density argument inverted.

## 6. Serving shape (what the chat does with it)

One faithful stream; queueing on the Precision (the existing 16-slot batch
assembler degenerates cleanly to a depth-N queue). Prompt ingestion is sequential
prefill — ~60 µs/position, so a 32-char prompt costs ~2 ms before generation.
A full 160-token reply: **~8 ms compute**. A hundred users deep, the last one
still waits under a second for a real message. Daemon grows `--engine kv160`;
the t1/SQSB config stays one `fpgautil` load away for the aggregate-throughput
headline.

## 7. Where it loses, said up front

- **Aggregate throughput drops.** One stream at ~20k average is a third of the
  59,965.5 sixteen-stream record. Different product: that one sells tok/s, this
  one sells *messages*. Both stay buildable from the same tree.
- **The bit-exact contract changes.** INT8 KV means a new golden reference (R0);
  numbers before and after are not comparable token-for-token. The NLL gate is
  what keeps it honest.
- **T > 160 still spills.** Past the window it's the KV-DDR road (doc 6), with its
  own bandwidth co-limit (~80k aggregate INT8 / ~125k INT4, DERIVED). This campaign
  deliberately stops at the on-chip window.
- **R5 is PROJECTED.** 250 MHz has never closed on this part; the contingency rung
  exists because of exactly that.

## 8. Logistics

Gate-in-sim before every synth (`run_vec_kv.py` joins the `run_*.py` ladder);
`.mem` dirs and builds under `C:/kevbuild/` (never in the OneDrive tree); every
Vivado run timed and logged; tok/s claims need 3/3 bit-exact board runs; ladder
entries land in `WIDE-WORD-DATAPATH-LOG.md` and `fabric/progress.py` as they're
measured. By this project's cadence: R0–R1 ≈ 2–3 days, R2–R4 ≈ 1–2 days each,
R5 ≈ 1–2 days of builds — **call it two weeks to 20k, with the chat turning real
in the first three days.**
