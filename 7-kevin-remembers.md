# Part 7, Kevin remembers — the faithful-stream campaign

**Target: 20,000 tok/s *average* on N=1 at T=256 — the model's entire trained
context — model-faithful decode.** A 160-token reply that is an actual message,
generated on the fabric in single-digit milliseconds, in a chat that *remembers
the previous exchange* (256 chars ≈ 2–4 short turns). One stream that remembers,
instead of sixteen that don't.

Why 256 and not some smaller window: the window only costs attention cycles, and
once attention is P-wide those are small against the GEMV floor (§3). T=64 would
cap a whole turn at 64 chars; T=160 holds one turn; **T=256 is full trained
context and multi-turn memory for ~1.5k cycles over the T=160 plan** — the only
window where the demo visibly remembers you. The chat UX stopped depending on the
tok/s rungs ago (even 5k tok/s returns 160 chars in ~31 ms); the rungs are the
headline number, the window is the behavior.

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

"Average" is over a full-window generation (window growing 1→256, mean T̄ ≈ 128).
A realistic chat turn (short prompt + 160-char reply, positions ≤ ~180, T̄ ≈ 90)
runs *better* than the headline average. Worst case (window full, T=256) is
reported separately — all three numbers get MEASURED, none gets hidden.

## 3. The cost model (DERIVED from measured rungs)

Per token, every weight is read once: **3,195,136 MACs** (4 × 786,432 + head 49,408).

| component | cycles, today | the lever | cycles, after |
|---|---|---|---|
| GEMV (LANES=256) | 12,481 | second URAM port → LANES=512 | ~6,241 |
| attention (P=8) | 32,896 avg / 65,536 worst | widen to **P=128** | 2,056 avg / 4,096 worst |
| LN/GELU/AQ/softmax/control | ~5,466 (= 17,947 − 12,481, from the @200 rung) | P-wide non-linears + a wide softmax pass (T=256 walks 256 elements/layer) | ~2,500 |
| **total avg** | — | — | **~10,797 @ 200 = 18,524 · @ 250 = 23,155 ✓** |

KV cache at T=256, N=1, **INT8**: 2 × 4 layers × 256 × 256 = **512 KB**. Placement:
KV banks in BRAM + leftover URAM (2.3 − 1.6 = 0.7 MB URAM free; the N=16 per-stream
scratch shrinks 16× at N=1, freeing BRAM). KV write traffic is 2 KB/token — noise.

Memory is the wall that does NOT move with N: N=1 frees *logic* (15 of 16 MAC
banks: LUT/DSP/FF), not URAM/BRAM. The ~3 MB budget and the doc-0 crossover stand.
A larger model fits only to ~L5 d256 (3.95M params, 1.99 MB INT4, −3k tok/s) — and
a non-lemmatised model would spend exactly that margin re-learning the function
words the Keviniser deletes. The campaign keeps the kevinised 3.16M model; an
L5-raw training run is a cheap side experiment (data track, no RTL), not the plan.

## 4. The rungs (each ships a MEASURED, bit-exact number)

**R0 — the reference.** `goformer_kv` already proves KV decode ≡ full recompute.
New: an **INT8-KV** fixed-point reference (`goformer_kv8` + the spec pin in the
`goformer_fixed` style) and a model-level NLL gate by the KVarN method — K4/V4 was
already a GO at +0.72% NLL, so K8/V8 must pass with margin or the campaign stops
here. This changes the golden contract; the new reference becomes the gate for
every rung below.

**R1 — make it remember.** Per-layer K/V banks + write-at-`pos` + `tcount = pos+1`
in the vec-lineage datapath; TMAX=256 pos_emb upload. Gate: `run_vec_kv.py`, token
streams equal to R0 over full-window generations, then board.
→ ~50.8k cyc avg = **~3,900 tok/s avg @ 200 (PROJECTED)**. The chat becomes real at
this rung — a 160-char reply in ~40 ms — everything after is the headline number.

**R2 — wide attention.** `vec_attn` P=8→128 (elementwise MAC; INT8 KV halves the
read bandwidth that pays for it; N=1 has the area). → ~20.0k cyc = **~10,000
(PROJECTED)**.

**R3 — dual-port GEMV.** At N=1 the true-dual-port URAM's second port — the one
split-brain spent on cohort 2 — is free. Stripe weight reads across both ports:
LANES=512, GEMV floor halves. Both ports at 200 MHz are *silicon-proven by the
record itself*. This is **not** the dead clock-domain double-pump
(DOUBLE-PUMP-100K.md); no second clock exists here. → ~13.8k cyc = **~14,500
(PROJECTED)**.

**R4 — P-wide non-linears.** LN/dequant/AQ at P=32 and a widened softmax walk
(T=256 makes the serial softmax visible; the doc-6 radix-4 work is the starting
point). The N=1 design frees ~15/16 of the MAC fabric, so area is not the
constraint it was at 98% density. 5,466 → ~2,500.
→ ~10.8k cyc = **~18,500 @ 200 (PROJECTED)**.

**R5 — the clock.** 250 MHz on a design a fraction of the SB's density (the 6 ns SB
build carried ~1.6× silicon margin; the blocker at 250 was routing density, which
N=1 removes). → **~23,200 avg / ~19,500 worst-case (PROJECTED) — target met.**
At T=256 this rung is load-bearing for the 20k headline (at 200 MHz the average
lands ~18.5k). *Contingency if 250 won't close:* 214.3 MHz (next PL divisor) →
~19,850 avg — within trimming distance (softmax/control), or the realistic-traffic
number (T̄≈90 → ~21,500 @ 214.3) carries the demo honestly.

## 5. Already proven dead — do not re-propose (doc 6 / log §19–26)

LN→AQ schedule overlap · attention→PROJ overlap · 3 INT4×INT8 MACs/DSP
(`dsp3_pack_proof.py`) · the 2× clock-domain double-pump. The levers above are
chosen to be none of these: R3 is port-parallelism (proven), R2/R4 are width
(area-for-cycles, and N=1 has the area), R5 is the density argument inverted.

## 6. Serving shape (what the chat does with it)

One faithful stream; queueing on the Precision (the existing 16-slot batch
assembler degenerates cleanly to a depth-N queue). The window holds the
conversation: prompt = the running transcript tail (≤ 256 − reply budget), so
Kevin remembers the previous turns without any session machinery. Prompt
ingestion is sequential prefill — ~60 µs/position, a full-window refill ~15 ms,
a typical prompt ~2 ms. A 160-token reply: **~8 ms compute**. A hundred users
deep, the last one still waits well under a second for a real message. Daemon
grows `--engine kv256`; the t1/SQSB config stays one `fpgautil` load away for
the aggregate-throughput headline.

## 7. Where it loses, said up front

- **Aggregate throughput drops.** One stream at ~20k average is a third of the
  59,965.5 sixteen-stream record. Different product: that one sells tok/s, this
  one sells *messages*. Both stay buildable from the same tree.
- **The bit-exact contract changes.** INT8 KV means a new golden reference (R0);
  numbers before and after are not comparable token-for-token. The NLL gate is
  what keeps it honest.
- **The 20k headline leans on R5.** At T=256 the 250 MHz rung is load-bearing
  (200 MHz lands ~18.5k average). The contingency ladder is stated in R4/R5;
  every fallback number is still reported as what it is.
- **T > 256 does not exist** — that is the model's trained context, not a window
  choice. Longer conversations truncate to the transcript tail, like every chat
  with a context limit. KV-DDR (doc 6) stays the road for a future bigger-context
  model, not this one.

## 8. Logistics

Gate-in-sim before every synth (`run_vec_kv.py` joins the `run_*.py` ladder);
`.mem` dirs and builds under `C:/kevbuild/` (never in the OneDrive tree); every
Vivado run timed and logged; tok/s claims need 3/3 bit-exact board runs; ladder
entries land in `WIDE-WORD-DATAPATH-LOG.md` and `fabric/progress.py` as they're
measured. By this project's cadence: R0–R1 ≈ 2–3 days, R2–R4 ≈ 1–2 days each,
R5 ≈ 1–2 days of builds — **call it two weeks to the headline, with the chat
turning real in the first three days.**
