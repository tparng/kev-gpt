# Double-pump to 100k: running the MAC island at 2× the fabric clock

The honest plan to make **100,000 tok/s the headline** on the KV260 — without a bigger
chip and without a dumber Kevin. This is the one lever left that breaks the MAC floor.

## ⏯ RESUME HERE (state as of 2026-06-08 ~17:00)

**Banked & pushed:** record **59,965.5 tok/s @200 MHz MEASURED** (N=16 split-brain,
bitstream `C:/kevbuild/stage3_seqsb16_60b`, board `gemv_seqsb_53k.bit.bin`); single-clock
**250 MHz proven dead**; **double-pump Stage 0 = GO** (`mac_bank_dp` ~393 MHz OOC, `0e0f7e1`).

**✅ Stage 1 (the FULL MAC double-pump + AQ feed, SIM) — DONE & pushed** (`b4229c7`,
`5d1abb8`, `9a9b469`, `ca022c8`, `7ea16ed`). All bit-exact, behind a `DP` parameter
(default 0 = byte-identical single-pump). The clk2x MAC + 2-wide AQ feed in SIM project:

| config (run_sb_seq --nd 0 --tmax 16) | cyc_total | tok/s @200 | note |
|---|---|---|---|
| `--att2 0 --dp 0` | 53,633 | 59,665 | baseline (also the MEASURED config) |
| `--att2 0 --dp 1` | 44,161 | 72,462 | clk2x MAC + 2-wide AQ |
| **`--att2 1 --dp 1`** | **39,577** | **80,855** | **+ per-cohort attention — the stack** |

All 16/16 bit-exact (tok + x4 + lnf + head, every stream). **80,855 tok/s @200 MHz
projected (sim)** — past the ~78k Stage 1 target. Pieces, each gated:
- **MAC double-pump (LUT + DSP).** `mac_bank_dp` (LUT) and `mac_bank_dsp_dp` (the WP487
  packed-pair leaf, 2 K-steps/clk into the clk2x DSP P-register, sum_act 2/clk). Gates:
  `run_gemm16 --lut-only`/`--dp-full`, `run_gemm_sb --lut-only`/`--dp-dsp`, `run_sb_seq
  --nd 6 --dp 1` — all ALL_BITEXACT.
- **2-wide AQ feed.** The double-pumped MAC consumes 2 K-steps/clk; the AQ producer fed
  1 stream-row/clk → group-0 AQ-bound (the −7,220-not-−12,544 gap). Fixed: nl_engine 2nd
  AQ-feed read port (`ge_ra2`→`lnout1_r2`…), gemm_cohort_vec 2nd xm write port
  (`x_we2`…), cohort_engine GE_AQ processes a stream PAIR/clk (commit in N/2=4 beats).
  Drop grew −7,220 → −9,472.

**THE NEXT STEPS, in order:**
1. **Readback/dequant double-pump (RB, ~10,360 cyc) → ~92k.** GE_RB reads `gv_yout` 1
   row/clk, dequants (`vec_dequant`, already 1/clk), writes NL banks (`dwr`) 1/clk. Mirror
   the AQ widen: 2nd readback port on `gemm_cohort_vec` ymem + 2 `vec_dequant` lanes + 2nd
   write port on the nl_engine qkv/attn/mlp/head banks. Gate `run_sb_seq --dp 1`, re-measure
   (39,577 → ~34,500 target). Then idle/SETTLE trims toward 32,000 cyc = 100k @200.
2. **Stage 1b (Vivado) — the silicon GO/NO-GO.** BD MMCM emits `clk2x` (400 MHz, phase-
   locked 2× of 200); the `weight_bank_tdp` SYNTHESIS branch needs the real clk2x dual-beat
   read (today stubbed 0 — its `g_w` SYNTHESIS comment); the nl_engine/xm 2nd ports map to
   TDP BRAM; Pblock the MAC island off the congested fabric; impl; board `--fclk 200
   --fclk2x 400` sweep → MEASURED. **This is the campaign's real GO/NO-GO** (does clk2x=400
   close + run bit-exact on silicon). Spend-limited — one Vivado run at a time.

**Key files:** `rtl/mac_bank_dp.sv` + `rtl/mac_bank_dsp_dp.sv` + `tb/tb_macdp.sv` (clk/clk2x
convention: `always #5 clk`, `#2.5` quarter-shifted `clk2x`). `DP`+`clk2x` thread through
`gemm_banked_resident_vec`, `gemm_cohort_vec`, `weight_bank_tdp`, `gemm_split_brain`,
`cohort_engine`, `sequencer_sb`, `nl_engine` (2nd AQ read port) + all TBs.

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

### Stage 0 — De-risk the clock domain (the go/no-go) — ✅ DONE, **GO**
- **Phase A (arithmetic): GREEN.** `mac_bank_dp` (the double-pumped bank, 2 K-steps/clk
  on clk2x, accumulator in the clk2x domain) is **bit-identical** to the real `mac_bank`
  over 4 seeds + neg/pos/altsign corners + a LANES{16,64,128}×K{512,1024,2048} sweep,
  0 mismatches (`run_macdp`, commit d299da5). The all−8/−128 corner lands exactly on the
  proven 2²⁰ accumulator range. The double-pump is a pure timing transform → every
  existing `seq_ref` gate still holds. **The arithmetic risk is dead.**
- **Phase B (timing, the go/no-go): GO.** OOC Fmax (`ooc_macdp.tcl`): the critical path
  is **2.545 ns → Fmax 392.9 MHz**, missing the 400 MHz target by only −0.045 ns. And
  this is the *conservative* case — the **LUT variant** (8,201 LUT, 0 DSP), **OOC synth**
  (pessimistic vs silicon), **no Pblock**. The DSP-cascade variant (the real MAC, the
  DSP48E2 P-register at clk2x) is faster, silicon runs faster than OOC, and a tight
  Pblock shortens routes — all three push it comfortably over 400. **A double-pumped MAC
  closing ~393 MHz in its slowest form with zero floorplanning is GO.** The plan proceeds.

  *Caveat carried forward:* 392.9 is the standalone LUT bank. Stage 1 must confirm the
  DSP-cascade variant in the real datapath (with the operand feed and the Pblock) holds
  ≥400 on the board — that is Stage 1's board sweep, not a new unknown.

### Stage 1 — Double-pump the MAC (the RUN phase) → ~78k

**The exact mechanism (decided — Stage 0 settled it).** The RUN kc-loop datapath runs
in the `clk2x` domain; the FSM stays at `clk`. Concretely, in `gemm_banked_resident_vec`
(and its split-brain twin `gemm_cohort_vec`):

- **Keep the FSM single-domain at `clk`.** The IDLE/RUN/DRAIN/SETTLE/FIN state machine,
  the group loop `g`, the drain counter `db`, `start`/`done`, and the `ymem` write/readback
  all stay on `posedge clk`. No FSM split, no async CDC — this is the key simplification.
- **Feed 2 K-steps per `clk` into `mac_bank_dp`.** In the RUN state, `kc += 2` per `clk`
  (was +1). Read TWO weight words (`grp_base+kc`, `grp_base+kc+1`) and TWO activation
  lanes per `clk`, and present them as `(w0,w1)`/`(x0,x1)` to `mac_bank_dp` (the proven
  Stage-0 bank), which does both accumulates across its two `clk2x` edges. The RLAT
  pipeline (`word_p`/`xl_p`/`v_p`) doubles to carry both lanes. `kmac += 2`; RUN ends at
  `kmac >= k_count`. RUN halves in `clk` cycles → 25,088 → 12,544.
- **Odd-K tail:** when `k_count` is odd, the last `clk` issues one valid + one masked
  step (`v1=0`) — `mac_bank_dp` already no-ops a disabled phase.
- **The weight 2-read is the only non-trivial part.** In sim (Stage 1a) model it as two
  reads of the URAM array. On silicon (Stage 1b) it is the URAM port clocked at `clk2x`
  (one word per `clk2x` edge = 2/`clk`) — URAM closes 400 MHz; `weight_bank_tdp`/`g_w`
  gets a `clk2x` read. The per-stream `xm` likewise reads at `clk2x`.
- **DSP-packed streams** (`mac_bank_dsp`, the `ND` streams): need the matching
  double-pump (the DSP P-register accumulator IS the `clk2x` accumulator, PCIN/PCOUT
  cascade at 2×, `sum_act` advancing 2/`clk`). Do the **LUT path first** (`--nd 0` gate,
  the beachhead), then the DSP-cascade variant — same scoping as Stage 0.

**Stage 1a (sim, iverilog, NO Vivado) — the bit-exact cycle drop:**
1. Beachhead: double-pump `gemv_banked.sv` / `gemm_banked_resident_vec.sv` LUT path,
   gate `run_banked` bit-exact (TB drives `clk2x` at 2× `clk`, quarter-shifted per the
   `tb_macdp` convention that killed the coincident-edge race).
2. Split-brain core `gemm_cohort_vec.sv`, gate `run_gemm_sb` ALL_BITEXACT.
3. Full sequencer: `run_sb_seq --nd 0 --tmax 16 --att2 0` → 16/16, and report `cyc_total`
   — it must DROP from 53,637 toward ~41k (the −12,544 RUN halving). **That drop, with
   bit-exactness, is the Stage 1a win.** Then the DSP path (`--nd 6`).

**Stage 1b (Vivado):** BD MMCM emits `clk2x` (400 MHz, phase-locked 2× of the 200 MHz
`clk`); Pblock the MAC island against the DSP/URAM columns; impl; board sweep `clk`=200,
`clk2x`=400. Target: **~78,400 tok/s MEASURED**.

*Status: Stage 0 GO banked. Stage 1a is the next execution step — a focused multi-hour
RTL task across the GEMV cores, best run as one dedicated session with the spec above
(the design is now fully decided; execution is mechanical).*

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
