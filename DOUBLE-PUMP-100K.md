# Double-pump to 100k: running the MAC island at 2× the fabric clock

The honest plan to make **100,000 tok/s the headline** on the KV260 — without a bigger
chip and without a dumber Kevin. This is the one lever left that breaks the MAC floor.

## Why this, and why now

The speed campaign hit a measured ceiling: **59,965.5 tok/s @200 MHz, MEASURED** (N=16
split-brain, 16/16 bit-exact, 3/3). Two walls, both proven:

1. **The cycle floor is ~51,100 cyc** (doc 6). The GE engine is a strictly serial
   FSM — RUN (25,088 MAC cyc at LANES=128) + readback (10,360) + the un-hideable
   non-linear chain. Schedule overlap is exhausted; the floor is MAC-bound.
2. **250 MHz is route-congestion-blocked** (3 builds confirm). Every timing cone was
   retired (LN ×2, embed→xres, u_dq) and the OOC closes 5 ns MET — but at 91–98%
   device fill the post-route critical path carries ~2.9 ns of pure *routing* delay,
   needing a 1.67× silicon margin the part does not give. Freeing 3.9k LUT moved WNS
   +0.06 ns. The general fabric cannot go faster than ~200 MHz here.

But the **DSP48E2 slices can.** They close 400–500+ MHz on this part natively, and the
systolic MAC rides the **dedicated DSP-column cascade** (PCIN/PCOUT), *not* the
congested general routing that blocks 250 MHz. So the one part of the design that can
go fast is exactly the part that dominates the floor. Right now we run it at 200 MHz —
granny computing. Double-pump it.

## The arithmetic (DERIVED — set honest expectations)

Per 16-token pass, the silicon cycle budget (~53,364 @200 MHz = 59,965 tok/s) splits:

| phase | cyc | what it is |
|---|---|---|
| RUN (MAC) | 25,088 | pure DSP multiply-accumulate, LANES=128 — **the prize** |
| readback + dequant | 10,360 | the RB drain, arithmetic-heavy (the u_dq path) |
| non-linear + idle | ~17,900 | LN / attention / softmax / residual — control-heavy |

Double-pumping moves an arithmetic phase to a 2× clock domain (400 MHz), so it finishes
in **half the wall time** while the rest stays at 200 MHz. The ladder:

| build | RUN | RB | total eff. cyc | tok/s @200 fabric |
|---|---|---|---|---|
| today | 25,088 | 10,360 | 53,364 | 59,965 |
| **+ 2× MAC** | 12,544 | 10,360 | 40,820 | **~78,400** |
| **+ 2× readback** | 12,544 | 5,180 | 35,640 | **~89,800** |
| **+ floor stack** (ATT2=1 −1,745, idle/SETTLE trims ~−3,600) | 12,544 | 5,180 | ~30,300 | **~105,000** |

So: **2× MAC alone is ~78k** (already shatters the granny ceiling). **100k needs the MAC
*and* the readback double-pumped, plus the cycle-floor cuts we already have gated.** That
is the honest target — 100k is the full stack, not one trick. (Upside: the DSP can take
2.5× / 500 MHz on −2LV; at 2.5× the MAC the ladder lifts a further ~5–6k tok/s. We design
for 2× and probe 2.5× on the board, exactly as we sweep `--fclk` today.)

## The key feasibility argument

The 200 MHz wall is **routing congestion on the general fabric**, not logic speed. The
double-pumped MAC sidesteps it because:

- **The DSP cascade is dedicated silicon.** A systolic GEMV chained through PCIN/PCOUT
  uses the DSP column's hard interconnect, which is not subject to the LUT/routing
  congestion that adds 2.9 ns to the general datapath. A 400 MHz MAC cascade can close
  in a 95%-full device *because it does not use the congested routing.*
- **The 2× clock is phase-aligned, not async.** `clk2x` comes from the same MMCM/PLL as
  `clk`, locked at exactly 2× and 0° phase. The crossing is a *known-phase* crossing
  (the easy kind) — feed 2 operands per `clk` cycle, the MAC consumes one per `clk2x`
  edge. No metastability, no async FIFO.
- **URAM and the activation memories already run fast.** URAM closes 400 MHz+; the
  weight image read (`weight_bank_tdp`) and the per-stream activation read just need a
  `clk2x` port delivering 2 words/`clk` — the same wide-word staging we already do, one
  level deeper.
- **The arithmetic does not change.** Double-pumping is a *timing* transform: the MAC
  does the same accumulations in the same order, just at 2×. So **every existing
  bit-exact gate still holds** — `seq_ref` is unchanged, the sim proves correctness
  (iverilog drives `clk` + `clk2x`, the math is identical), and the 2× is verified on
  silicon by the board sweep. Bit-honest-before-fast survives intact.

## The risks, named honestly

- **R1 — 400 MHz MAC close in a full device (the big one).** Mitigation: floorplan the
  MAC island into a tight Pblock hugging the DSP/URAM columns so its routes are short
  and it rides the cascade. **Stage 0 exists to kill this risk cheaply before any
  integration.**
- **R2 — 2× operand feed bandwidth.** URAM/xmem must deliver 2 weight + 2 activation
  words per `clk`. Plausible (URAM is fast) but must be measured; the wide-word banking
  already stages P/word, this stages 2P.
- **R3 — CDC at the island boundary.** Mitigated by the phase-aligned 2× clock (known
  phase). The run FSM (200 MHz) hands a start/length to the MAC island (400 MHz) and
  collects a done — a clean, slow handshake; only the operand *stream* is at 2×.
- **R4 — dual-clock simulation.** iverilog is an event simulator and models two clocks
  fine; the TB drives `clk2x` at 2× `clk`. Since the arithmetic is unchanged, the
  per-phase `seq_ref` comparison is the same gate it is today.
- **R5 — power/thermal.** 2× DSP toggling on ~600 DSPs. Minor at this scale; instrument
  board temp during the sweep (the demo already samples it).

## The staged plan (each stage gated bit-honest, like everything here)

### Stage 0 — De-risk the clock domain (the go/no-go, ~1–2 days)
Build a **standalone double-pumped `mac_bank`** (one bank, LANES=128, `clk2x` operand
feed) — no sequencer, no integration. Gate its arithmetic bit-exact vs the current
`mac_bank` in iverilog (clk2x in the TB). Then **OOC + impl + board it inside a tight
Pblock** and *measure the real MAC clock* via an `--fclk2x` sweep. **Decision: if a
floorplanned double-pumped MAC closes ≥400 MHz on silicon, the plan is GO. If it caps
near 200, the plan is dead and we found out for the price of one toy build.** This is the
single most important step; nothing else starts until it is green.

### Stage 1 — Double-pump the MAC (the RUN phase) → ~78k
Rewrite `mac_bank` / `mac_bank_dsp` to consume 2 operands/`clk` on `clk2x`. Add a
`clk2x` read port to `weight_bank_tdp` (2 weight words/`clk`) and to the per-stream
activation memory. The run FSM advances `kc` by 2 per `clk`. Gate: every existing
`run_banked` / `run_gemm_sb` / `run_sb_seq` bit-exact (the math is unchanged; the TB
adds `clk2x`). Build with the MAC island Pblocked at `clk2x` from the MMCM; board sweep
`clk`=200, `clk2x`=400. Target: RUN 25,088 → 12,544, **~78,400 tok/s MEASURED**.

### Stage 2 — Double-pump the readback dequant (the RB phase) → ~90k
The dequant drain (`vec_dequant`, the u_dq path — already pipelined, arithmetic-heavy)
moves to `clk2x`, draining 2 rows/`clk`. Gate bit-exact. Board. Target: RB 10,360 →
~5,180, **~89,800 tok/s MEASURED**.

### Stage 3 — Stack the gated floor cuts → ~100k
Fold in the already-gated cuts the device can now afford (the 2× phases freed timing and
some area): **ATT2=1** per-cohort attention (−1,745 cyc, already gated, fits once the
MAC island Pblock frees general fabric), plus the idle/SETTLE trims. Target: **~100,000
tok/s, 16/16 bit-exact, 3/3 — the headline.**

### Stage 4 — The headline run
MMCM at 200/400, full `--fclk`/`--fclk2x` sweep, 3/3 bit-exact, board temp logged.
Update `fabric/progress.png`, doc 6, and the README with the MEASURED 100k. Write the
double-pump up as its own log section.

## What would make it fail, stated up front

If Stage 0 shows the double-pumped MAC cannot close 400 MHz even Pblocked against the DSP
column, the lever is dead and 59,965.5 @200 stands as the honest ceiling — at which point
100k genuinely requires LANES=256 (a bigger device) or a smaller model (the thesis). The
whole plan is staged precisely so that the one make-or-break question is answered first
and cheapest. Bit-honest before fast, all the way up.
