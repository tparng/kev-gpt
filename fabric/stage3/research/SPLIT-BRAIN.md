# Split-brain: two independent 8-stream pipelines on dual-ported URAM

**Status:** design note, 2026-06-06. The ONE big play left on the cycle axis —
~1.65× in a single redesign, multiplicative with the clock campaign.

## The observation

The N=16 single-engine design (99,828 cyc/16 tok, MEASURED-equivalent) spends:
WAIT 26.8k (weight stream) + RB 20.7k + AQ 15.7k + AQW 13.5k + IDLE 20.1k +
tails ~3k. Only the 26.8k touches the URAM weight banks — **73k cycles queue
behind a resource that is busy 27% of the time.** And the URAM banks are TDP:
**port A is idle after boot** (loader-only).

## The architecture

Two fully independent 8-stream pipelines ("cohorts"), phase-free — no merge,
no GWAIT, no partner-waiting:

- Cohort k = 8 streams, its own GE FSM (AQ→RUN→RB), its own dequant+GELU,
  its own AQ quantizer, its own readback recovery, its own nl_engine (already
  exists — one per group today), its own MAC banks (already exists — 16 banks
  split 8+8; rebalance ND per cohort: e.g. 6 DSP + 2 LUT each).
- Shared: ONLY the URAM weight banks, read through both TDP ports — cohort 0
  on port B (today's path), cohort 1 on port A (muxed with the boot loader's
  write; loader idle at runtime). Independent addresses — the cohorts are at
  different calls/layers at any moment and that's FINE (no bandwidth shared,
  no merge needed — the whole desync problem dissolves instead of being
  managed).
- Shared LN/attention (today's arbitration) initially KEPT (each cohort = one
  engine, same contention as today); split later only if the profile says so.

## The numbers (from measured components)

- Each cohort = the proven single-pass N=8 flow: 62,704 cyc/8 tok (gated,
  with early-AQ). Two in parallel: **16 tok / ~63k cyc**.
- @166.7: **42.3k** · @200: 50.8k · @250: 63.5k (vs 25.7k measured today).
- Then per-cohort internals (RB/AQ overlap is affordable when each cohort owns
  its dequant) toward ~40-45k → 16×250/40k = the 100k identity.

## Budget (measured deltas)

| add | LUT | DSP |
|---|---|---|
| dequant ×2 (measured standalone) | +7.1k | +16 |
| AQ quantizer ×2 | ~+5k | ~+30 |
| GE FSM + recovery + muxes ×2 | ~+2k | — |
| reclaim: kmem/vmem LUTRAM→BRAM (2 attn copies... shared attn = 1 copy) | −2.5k | — |
| reclaim: gelu p_0_out → BRAM | −1.8k | — |
| **net** | **~+9.8k → ~117k** | +46 → 1,217 |

Tight against 117.1k — the build may need one more reclaim (ymem width, or
ND rebalance to flip a LUT bank per cohort to DSP with the remaining DSPs).
BRAM: +2 gelu/kmem tiles ≈ 136/144.

## Risks (honest)

1. **URAM TDP with both ports reading at speed** — the saga (log §16) proved
   the weight banks are the most synthesis-fragile structure in the design.
   The canonical TDP template change is a 5-minute standalone OOC experiment
   — RUN IT FIRST. If port A refuses, fallback: time-multiplex one port at 2×
   the word rate (needs the clock campaign first) or duplicate-load a second
   URAM image... (32 URAMs spare? No — 64/64 full. Port A must work.)
2. LUT fit at ~117k = 99.9% — see reclaim list; impl congestion at that fill
   regresses Fmax (the 95.6% build needed 1.61×; split-brain MUST NOT cost the
   166.7 floor). If fit busts: drop to 2×7 streams (14 total, −2 banks ≈ −9k
   LUT, 37k @166.7 still).
3. The loader-mux on port A: boot-time writes vs runtime reads — clean enable
   separation, but it's new logic on the fragile structure.

## Sequencing

1. Timing campaign lands (its pipeline stages benefit both cohorts).
2. URAM dual-read micro-experiment (standalone OOC, 5 min).
3. Core: second read port through the bank generate (waddr_b/rd_b per bank),
   second wsel pipeline, MAC banks partitioned per cohort.
4. Sequencer: instantiate the GE machinery per cohort; delete the merge/GWAIT
   logic entirely (cohorts never share a pass); nl_engine k pairs with GE k.
5. Gates at every step (run_gemm16 grows a dual-cohort case; pp gates as ever),
   then build + sweep.
