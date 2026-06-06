# Wide-word P-wide datapath — the engineering log (752 → ~2.3k tok/s push)

Everything tried on the road from the **measured 752 tok/s** (`sequencer_fast`, scalar
datapath) toward the **~2k band** with the P-wide vector datapath (`sequencer_vec`). Written
honest-first: the dead ends and the on-silicon bugs *are* the result. Every number is tagged
**MEASURED** (on silicon / printed by a committed gate), **SYNTH** (Vivado OOC or impl report),
or **PROJECTED** (arithmetic from a measured cycle count).

Anchors going in (already MEASURED on the KV260): **44.32 tok/s** (first correct sequencer,
clock-bug fixed) → **752 tok/s** (`sequencer_fast` PE=128, resident weights, GELU+LN, silicon
overclock to 125 MHz). The A53-optimised wall is ~11 tok/s, so the fabric was already ~68× past
the CPU. This log is the next step: a **P-wide datapath** that processes P scratch elements per
cycle instead of one.

---

## 0. The idea and why it should win

`sequencer_fast` runs the non-linear/serial work (LayerNorm feed, act-quant, dequant, GELU,
attention load, residual) **1 element/cycle**. Cycle profile at PE=128 ≈ **166k cyc/token**, of
which ~100k is that serial work. The P-wide datapath (`sequencer_vec`) widens those paths to
**P elements/cycle** by bank-interleaving every scratch buffer (element `e` → bank `e%P`, row
`e/P`). Two independent knobs:

- **P** = serial-lane width (how many scratch elements/cycle the non-linears chew).
- **LANES** = GEMV MAC width (the resident `gemv_banked_resident`, unchanged core).

New P-wide bricks, each gated **bit-exact in iverilog** before integration: `layernorm_vec`,
`vec_dequant`, `vec_gelu`, `vec_attn`, plus the wide GEMV boundary in `sequencer_vec` itself.

**Measured cycle win (SIM, run_vec_seq):** P=1 (`sequencer_fast`, 166k) → P=8 (`sequencer_vec`,
**67k cyc/token**) → P=16 (65.7k). The big jump is P=1→8 (~100k serial cycles collapse); P=8→16
is marginal because a **~40k 1-cycle floor** (GEMV act-feed + readback + attn/gamma loads, all
inherently 1/cycle) then dominates. **Conclusion: P=8 is the sweet spot; the lever past it is
widening the GEMV boundary, not bigger P.**

---

## 1. What got built and gated (all bit-exact in iverilog)

| Brick | File | Gate | Result |
|---|---|---|---|
| P-wide LayerNorm | `rtl/layernorm_vec.sv` | `run_vec_layernorm` | bit-exact P=4/8/16 |
| P-wide dequant | `rtl/vec_dequant.sv` | `run_vec_dequant` | bit-exact (after unsigned-mant fix) |
| P-wide GELU | `rtl/vec_gelu.sv` | `run_vec_gelu` | bit-exact |
| Single-head attention | `rtl/vec_attn.sv` | `run_vec_attn` | bit-exact T∈{1,8,16,32} |
| **Full single-token forward** | `rtl/sequencer_vec.sv` | `run_vec_seq` | **bit-exact** tok/x4/lnf/head, P=8/16, L=16/128 |
| AXI shell | `rtl/gemv_axi_seq_vec.v` | — | IDCODE `SQRV` 0x53515256 |
| Board driver | `board/pl_seq_vec.py` | bit-honest gate | withholds tok/s unless token == seq_ref |

The full forward (4 blocks + LN_f + head + argmax) is built from **callable** GEMV and LN
micro-sequences (parameter registers + a return state), reused for all 4 GEMVs / 9 LNs. Each
gateable intermediate has its own banked buffer + a readback mux so every phase is inspectable.

Integration bugs found and fixed **in sim** along the way (the gate caught these):
- `vec_dequant`: mantissa is **unsigned** [0, 2²⁴) but was treated as signed → values ≥2²³
  flipped negative. Fixed with `$signed({1'b0, mant})` zero-extend. (The standalone gate had
  *missed* it by testing the signed range; fixed the gate too.)
- `sequencer_vec` attention head counter `hh` uninitialized in the callable path → infinite
  spin. Fixed `hh<=0` in the QKV-return state.
- proj act pre-shift used truncating `>>>`; seq_ref uses round-half-away-from-zero. Matched.
- `g_dqrow` declared `[9:0]` (max 1023) overflowed for block-3 mlp (1120) and head (1152).
  Widened to `[11:0]`.
- iverilog-2012: can't `$readmemh` a 2D-array row → flat ROM + ranged `$readmemh`. Variable
  part-select of an unpacked-array **element** reads X → only ever part-select plain regs.

---

## 2. First build attempt — the LUT/mux blow-up (SYNTH, FAILED)

Banked scratch declared the obvious way: `reg [W] buf [0:P-1][0:ROWS-1]`. Full P=8/L=128 build:

```
ERROR [DRC UTLZ-1]: LUT as Logic 216,309 / 117,120 (185%)
                    MUXF7        88,368 / 58,560  (151%)
```

**Cause:** a `[P][rows]` array read/written at a **variable row** synthesises to a giant
per-lane row-multiplexer. With ~10 variable-index access sites across the datapath, the MUXF7
count exploded. **DRC failed before placement.** Bit-exact, real 2.5× cycle win — but physically
~2× too big for the chip.

---

## 3. Wide-word banking restructure (the fix for §2)

Rewrote **every** scratch buffer and every P-wide-read ROM as **one row-addressed wide word**:

```
reg [P*W-1:0] buf [0:ROWS-1]        // lane l lives in bits [l*W +: W]
```

Now the variable row is a memory **address** (free), leaving only a small P:1 lane-select. P-wide
accesses touch the whole word (constant offsets, no mux). The two inherently-1/cycle producers
(GEMV readback, attention ctx) stage P elements into a plain-reg word and write one wide word per
P. ROMs went wide too (`write_mems_wideword`: `tok_emb_w`/`pos_emb_w`/`gamma_w` P×32, `dqm_w`
P×24, `dqe_w` P×8) — a bonus cycle win (embed dropped 256→32 cycles; the gamma-load phase
vanished). Re-gated **bit-exact**, **64,604 cyc/token** at P=8/L=128.

**The bogus-OOC trap (a lesson):** the first OOC fit check reported a triumphant **46,406 LUT
(40%)** — but it had run with the `.mem` files **missing** (a `Push-Location` had silently
failed), so the ROMs synthesised to X and Vivado **constant-pruned most of the datapath**. The
real numbers (below) were 2.7× higher. *Always verify an OOC ran against real mems — look for the
"RAM too shallow → BRAM/URAM" notes and a non-trivial LUT count.*

---

## 4. The real fit problem — P, not LANES (SYNTH)

Proper synthesis (with real mems) of the wide-word design:

| Config | CLB LUTs | of which Distributed RAM | Verdict |
|---|---|---|---|
| **P=8 / L=128** | **127,681 (109%)** | 31,001 | over budget |
| P=8 / L=64 | 120,829 (103%) | — | over |
| P=8 / L=96 | 124,505 (106%) | — | over |
| **P=4 / L=128** | **106,009 (90.5%)** | ~20,366 | **FITS** |

**Key finding: LANES is not the lever.** L=64 only saved ~6k LUT vs L=128 → the resident GEMV is
cheap (~13k LUT, ~100 LUT/lane). The **P-wide datapath is the ~114k bulk**. Halving to **P=4**
halves it and fits — and the floor-dominated cycle profile means P=4 barely costs speed
(**67,436 cyc/token**, +3k over P=8). Two side-facts: the shallow wide-word buffers infer as
**distributed RAM** (LUTRAM, async read) which eats LUTs but matches sim; and `tok_emb_w`/
`pos_emb_w` → LUT, all scratch buffers → RAM64M8, but `gemvy_bank`/`dq_mant`/`dq_exp` → **Block
RAM** (this mattered later).

---

## 5. Placement (SYNTH, two FAILS then a FIX)

P=8 at 100 MHz and at 40 MHz both **passed synth, failed placement**:

```
ERROR [Place 30-487]: unplaced instances require 9,899 CLBs / 8,474 available
                      Control sets: 197
```

Clock was a red herring (40 MHz failed identically at 9,863 CLBs) — it's **packing density**:
the distributed-RAM buffers are SLICEM-restricted, so CLBs pack at ~58%. The cure was **P=4**
(fits the LUT budget) **plus a logic-spreading impl strategy**:

```tcl
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE AltSpreadLogic_high [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AlternateCLBRouting [get_runs impl_1]
```

**P=4 / L=128 / 40 MHz: BUILT.** Setup slack **+7.504 ns** (closes with margin; the WNS=0.010
the script printed was the *hold* path). Bitstream produced. `IDCODE SQRV` confirmed.

---

## 6. On silicon — control correct, data wrong (MEASURED)

First board run (`pl_seq_vec`, fclk forced + verified):

```
CYCLES = 67,452   (cycle-EXACT to sim)      tok_out = 34   gold = 1   match = False
```

The classic **"control right, data wrong"** signature: the FSM executed the exact right number
of cycles, but the arithmetic was wrong. Identical wrong token at **20, 40, 60 MHz** → deterministic
**functional** divergence, not timing (and setup slack was +7.5 ns anyway). The wide-word datapath
passed every iverilog gate, but these P-wide bricks had **never run on real fabric**, and
async-read iverilog can't model what BRAM/silicon does.

### The bug hunt (each a ~40-min build, hence the discipline shift)

1. **Suspect: G_DQ BRAM read latency.** `gemvy_bank`/`dq_mant`/`dq_exp` map to **Block RAM**
   (synchronous, 1-cycle read) while everything else is distributed (async). G_DQ read them
   combinationally and used them the same cycle → on silicon, the *previous* row's data. Made the
   reads **registered** (stage-1 latch address, stage-2 consume — matches BRAM *and* gives iverilog
   the same latency, so the gate now catches the class). **Re-gated, rebuilt, ran:** byte-identical
   wrong values (`x4 = 0x3e7c05cf`) but CYCLES 67435→67452. So the fixed bitstream was loaded, but
   **this wasn't the bug** — the dequant inputs were already corrupt upstream. (Kept the fix; it's
   correct and necessary.)

2. **Ruled out: weight-stream / model mismatch.** `gemv_banked_resident` assembles low-chunk-first;
   the driver streams low-chunk-first — match. `sequencer_fast` (the working 752) uses the **same**
   resident GEMV core → GEMV proven on silicon. Board npz and build npz are **byte-identical**
   (same sha256) → baked dequant scales match streamed weights. All ruled out.

3. **Readback localizer.** Added `--readback` (read x4/lnf/head via the `rd_sel` port vs seq_ref):
   all wrong, magnitude **blown up** (x4 ≈ +31 vs −0.36 Q6.25). Added **`dbg_stop`** (CTRL bit) to
   halt after block 0, leaving block-0 phases in the banks, then read them in forward order:

   ```
   ln1_out  OK | qkv  OK | ctx  OK | attn_out  OK | ln2_out  >>> FIRST DIVERGENCE <<<
   ```

   Everything up to and including the proj output is **correct on silicon**. First wrong: **LN2**.

4. **The scale-invariance insight.** LN1 uses the *same* `layernorm_vec` and is correct — so the
   bug is either LN2's input or LN2-on-reuse. Crucially, **LayerNorm is scale-invariant**: a correct
   LN1 does *not* prove `xres` (the embedding) has the right *magnitude*, only the right direction.
   Everything downstream of LN1 is scale-immune too — but **res1 mixes raw `xres` with attn_out**,
   so a wrong `xres` scale first bites exactly at LN2. Upgraded `dbg_stop` to a **2-bit, 3-point
   probe** (after-embed / after-LN2 / after-block0):

   ```
   stop-mode 1 (after embed):  x_in   OK
   stop-mode 2 (after LN2):    x_res1  255/256 MISMATCH   ln2  MISMATCH
   ```

   **embed correct, x_res1 wrong → the residual add is the culprit.**

---

## 7. THE bug: un-reset `ci` → residual wrap-around (FIXED, commit 6f9348a)

`G_RB` (the GEMV readback) leaves the shared index `ci` at `g_m` (=256 for the proj GEMV).
`S_RES1`/`S_RES2` stream `ci = 0..ROWS-1` but **never reset it** — so `ci` *enters* the residual
as 256. Then:

- **iverilog:** `xres_bank[256]` is out of range → returns X, the write is **silently dropped**.
  Only after `ci` counts up and **11-bit-wraps** (256→2047→0→…→63) does the real `0..63` pass run —
  **once** — so the gate passed *by luck*.
- **silicon:** the distributed-RAM address is `clog2(ROWS)` = 6 bits, so `xres_bank[256]` **wraps
  to address 0**. As `ci` counts 256→2047→0→…→63, every row gets `attn` added **~29 times** →
  `x_res1 = embed + 29·attn` → the **+31 Q6.25 blow-up** that corrupted everything downstream.

The first-divergence-at-LN2 was a red herring created by LayerNorm's scale-invariance hiding the
bad `xres` magnitude until res1 mixed it with attn.

**Fix:** reset `ci<=0` at the `G_DQ → g_ret` transition (one line). Re-gated **bit-exact** *and*
**14,336 cycles faster** — the wasted wrap cycles are gone:

| | cyc/token |
|---|---|
| before fix | 67,453 |
| **after fix** | **53,117** |

**Lessons banked:** (1) iverilog **silently ignores** out-of-range array reads/writes; silicon
**wraps** the `clog2`-width address — bound-check or reset streaming indices, the gate won't catch
it. (2) LayerNorm scale-invariance can mask an upstream *scale* bug until a residual add.
(3) Localise on silicon with a readback port + per-phase `seq_ref`, don't guess-and-rebuild —
each build is ~40 min.

---

## 8. Where it stands + the numbers — IT WORKS (MEASURED)

**Status (2026-06-03): the wide-word datapath is correct AND fast on silicon.** Loaded the
ci-fixed P=4/L=128 bitstream, ran `pl_seq_vec --readback`: **`tok_out=1 == gold`**, and x4/lnf/head
all read back correct on real fabric. Then swept the PL clock (bit-honest — the driver withholds
tok/s unless `tok_out == seq_ref`):

| PL clock (achievable PLL rate) | MEASURED tok/s | token match |
|---|---|---|
| 40 MHz (build closure) | 753.1 | ✓ |
| 76.9 MHz | 1,448.2 | ✓ |
| **100 MHz** | **1,882.7** | ✓ **3/3 deterministic** |
| 111.1 MHz | (2,091.9) | ✗ MARGINAL — passed 2×, failed 4× → **not claimed** |
| 125 MHz | — | ✗ fails (past the silicon ceiling) |

**Headline: 1,882.7 tok/s, bit-exact, CPU out of the loop — 2.5× over the prior 752.** Reported
honestly at the highest *reliable* clock (100 MHz, 3/3 identical), not the lucky 111 MHz pass.
STA closed at 40 MHz with +7.5 ns; silicon ran clean to 100 MHz (a ~2.5× STA-pessimism factor,
consistent with the `sequencer_fast` 1.76×). The PLL snaps to 1000/N MHz (…100, 111.1, 125…), so
100 is the top reliable rung; nothing exists between 100 and the marginal 111.1.

| Quantity | Value | Tag |
|---|---|---|
| **Wide-datapath best** | **1,882.7 tok/s @ 100 MHz** | **MEASURED (bit-exact, 3/3)** |
| Prior best | 752 tok/s (`sequencer_fast`) | MEASURED |
| `sequencer_vec` P=4/L=128 cyc/token | 53,116 | MEASURED (sim+silicon cycle-exact) |
| Fit (P=4/L=128) | 106,009 LUT (90.5%), 56 URAM, ~210 DSP | SYNTH |
| Timing | +7.504 ns setup @ 40 MHz | SYNTH |

**Next levers (not yet done):**
- **BRAM-buffer (synchronous-read) rewrite** → frees the ~20–31k distributed-RAM LUTs → **P=8
  fits** → ~64k→ lower cyc/token at higher P AND closes the sim/hardware async-read gap by design.
- **Widen the GEMV boundary** (P-wide act-feed + readback) → attacks the ~40k 1-cycle floor — the
  real path to **10k**.
- **LANES=256** once the LUT budget is freed (more GEMV throughput; URAM has headroom).

**The honest ceiling for one KV260, this model:** single-stream ~10–15k (latency-bound); batched
aggregate ~100–150k (B≈8–10 short-Kevin-context streams + DSP-packed INT4 MACs); hard roofline
~200–270k (URAM weight bandwidth). 100k is a *batched* number and needs the short Kevin context —
the joke is still the thesis.

---

## 9. The BRAM sync-read rewrite — P=8 unlocked (SIM, gated bit-exact)

The first "next lever" from §8, executed. Every scratch buffer and every wide ROM in
`sequencer_vec` switched from asynchronous (distributed-RAM) reads to **registered 1-cycle
reads**: one read register + small address mux per memory, FSM loops carry an address counter,
a delayed address and a valid bit (the same stage-1/stage-2 pattern G_DQ already proved on
silicon). Embeddings, gamma, dequant ROMs and all 10 banks now have BRAM semantics; `inv_sact`
(17 deep) stays LUT. Attention load became a 1-deep prefetch (`qkv_r` and `at_ldv` both run one
register behind the address counter, so they land in the same cycle at `vec_attn`).

The gate caught the rewrite's only integration bug: an extra valid register in the attention
prefetch (`at_ldv <= wiv`) lagged data by one cycle — LN1/QKV gated bit-exact while ctx and
everything downstream diverged. The dbg_stop probe localised in ~3 sim runs, no silicon needed —
the async-read sim/hardware divergence class is gone by design.

**Cycles (sim, bit-exact at L=128):** P=4 53,157 (+41 vs LUTRAM 53,116 — the pipeline tax) ·
**P=8 50,325** · P=16 48,909. The sweet spot stays P=8.

Expected fit: P=8/L=128 was 127.7k LUT (109%); −31k distributed-RAM LUTs lands ~83% (PROJECTED,
needs OOC). On silicon: 50,325 cyc/token @ 100 MHz → **~1,987 tok/s** (PROJECTED).

**OOC, P=8/L=128 (SYNTH):** the LUT blow-up is *gone* — **74,165 LUT (63.3%)**, vs 127,681
(109%) async. The squeeze moved to BRAM: both embed ROMs in block RAM hit **174/144 tiles
(121%)**. URAM can't take them (no init support — URAM is why weights are streamed). Fix: the
pos table doesn't need 256 positions for Kevin's short context → **TMAX=64**, ~15 tiles for
pos_emb instead of 57. Re-gated bit-exact (the .mem truncation is benign). TMAX is now a
generic on shell + OOC/BD scripts; gate's `--tmax 64`. WNS at 8.0 ns came out −4.285 → STA
Fmax ≈ 81 MHz, same band as the LUTRAM build; 40 MHz build clock closes comfortably, the
silicon ceiling comes from the board `--fclk` sweep.

**OOC, P=8/L=128/TMAX=64 (SYNTH): FITS.** **72,017 LUT (61.5%) · BRAM 142.5/144 (99.0%) ·
URAM 56/64 · DSP 421.** LUT headroom is back (~38%); the binding resource is BRAM with 1.5
tiles spare — anything new on-chip must displace the embed ROMs. The P=4 LUTRAM design used
106k LUT (90.5%) — sync-read P=8 is both smaller AND ~3k cyc/token faster.

**Full build @40 MHz (SYNTH):** placed + routed without strategies drama — setup slack
**+4.092 ns** (crit ~20.9 ns vs P=4's 17.5 — the BRAM read mux is the slower path), 72,223 LUT
(61.7%). Bitstream `gemv_seqvec.bit/.bin` (IDCODE `SQRV`).

**On silicon (MEASURED, driver bit-honest):** bit-exact on first run, CYCLES = **50,324** —
cycle-exact to sim. Clock sweep:

| fclk | tok/s | match |
|---|---|---|
| 40 MHz | 794.8 | ✓ |
| 76.9 MHz | 1,528.6 | ✓ 3/3 |
| **83.3 MHz** | **1,655.9** | ✓ **3/3** |
| 90.9 MHz | (1,806.5) | ✗ 1/3 — not claimed |
| 100 MHz | (1,987.1) | ✗ 1/3 — first pass cold, then fails |

**P=8 reliable = 1,655.9 tok/s; the P=4 record (1,882.7 @ 100 MHz) still stands.** The lesson:
the BRAM datapath cost ~17 % silicon clock ceiling (~83 vs 100 MHz, the same ~2.0–2.5× STA
factor over its 20.9 ns crit path) and bought only 5 % cycles back. The first 100 MHz run
passed cold then failed warm — only multi-run sweeps count. Next: rebuild constrained at
80 MHz so placement actually tightens the BRAM read paths — silicon @100 MHz × 50,324 cycles
would be 1,987 tok/s.

### The 80 MHz rebuild — NEW RECORD (MEASURED)

Same RTL, constraint 40 → 80 MHz: placement tightened the BRAM paths from 20.9 ns to ~12.6 ns
(STA WNS −0.071, 9 endpoints — accepted; the gate is the board). Sweep, all 3/3 deterministic:

| fclk | tok/s | match |
|---|---|---|
| 100 MHz | 1,987.1 | ✓ 3/3 |
| 111.1 MHz | 2,207.9 | ✓ 3/3 |
| **125 MHz** | **2,483.9** | ✓ **3/3** |
| 142.9 MHz | (2,838.7) | ✗ 1/3 — not claimed |

**Headline: 2,483.9 tok/s, bit-exact, CYCLES = 50,324 == sim, 3.3× over 752, 1.32× over the
P=4 record.** Silicon/STA factor 1.57. The same constraint sweep on P=4 was never tried — the
40 MHz constraint, not the LUTRAM datapath, was the previous ceiling.

**Laptop reference (same model, MEASURED on XPS15):** PyTorch CPU 356, RTX 3050 Ti 719 (full;
KV 654 — launch-bound), ONNX Runtime KV ctx-256 748, ctx-64 best 1,273. The KV260 at ~6 W beats
the laptop's best single-stream by 2.0× at ctx-64 (3.3× at ctx-256) — a 3M-param model is
DDR/launch-bound on both CPU and GPU; on-chip residency wins on every machine, the laptop just
has no fabric to be resident in.

---

## 10. The P-wide GEMV boundary — 35,597 cyc/token (SIM, gated bit-exact)

The §8 lever, executed: `gemv_banked_resident_vec.sv` keeps the proven 128-lane MAC core but
widens the boundary — the act port takes P INT8 lanes/write (acts banked P/row like the scratch),
readback returns P INT32 per address (P-block of the LANES-wide group word). G_AQ quantizes P
lanes/cycle straight from the wide bank read; G_RB writes one full gemvy row/cycle. The two
1/cycle boundary loops (~7.4k + ~9.4k cycles) collapse by P.

Gate green first run on tok 10/48/100: **35,597 cyc/token** (was 50,325; −29%). Cycle floor
now ≈ GEMV MAC (~12k) + attention load (~4.3k) + LN/argmax (~7k) + dequant (~2.2k).
@125 MHz → **3,512 tok/s PROJECTED**; OOC + bitstream pending. Bonus: 9 SLICE-heavy carry-chain
adders fold into 8 DSP-friendly multipliers (act-quant is 64×34 → DSPs).

**OOC (SYNTH):** 74,181 LUT (63.3%) · BRAM 143.5/144 (99.7%) · URAM 56/64 · **DSP 505** (+84,
the eight act-quant multipliers). Fits with half a BRAM tile to spare.

**On silicon (MEASURED): 3,511.6 tok/s @ 125 MHz, 3/3 bit-exact, CYCLES = 35,596 == sim.**
The projection (3,512) landed within 0.4 tok/s. Sweep: 100 MHz 2,809.3 · 111.1 3,121.4 ·
**125 3,511.6** · 142.9 marginal 1/3 (4,013.3 — not claimed). 80 MHz build WNS −0.371,
silicon factor 1.57. 1.87× over yesterday's 1,882.7. Ladder: 752 → 1,883 → 2,484 → 3,512.
Levers left: LANES=256 (~29k cyc), softmax overlap, then 200 MHz pipelining → 10k.

---

## 11. LANES=256 — 22,942 cyc/token (SIM, gated bit-exact)

Pure parameter push: L=128→256 (WWORDS scales inversely to 12,800; the image is fixed
~12.6 Mbit). The GEMV halves twice — half the columns *and* half the group passes per matrix.
Gate green tok 10/48/100: **22,942 cyc/token** (35,597 → −36%). @125 MHz → **5,449 tok/s
PROJECTED**. The 1-cycle floor is now attention load + LN + softmax — the next cycle lever is
qkv/attn restructuring, not LANES.

**URAM geometry is the constraint, not capacity.** First L=256 OOC: 400k LUT (342 %), URAM 0 —
the single 1024-bit × 12,800 wmem pads to 16 URAM wide × 4 cascade = 64 = the whole device, so
Vivado marks the `ultra` attribute INFEASIBLE and silently falls back to LUTRAM. Two 512-bit
banks also fail: 512 b pads to 8 wide (512/72 = 7.1) → 64 total. Fix: bank the weight memory at
URAM-native width — 15 banks × 72 b × 4 cascades = **60 URAM**. **OOC L=256: 87,761 LUT (74.9 %),
BRAM 143.5/144, URAM 60/64, DSP 505 — FITS.** Re-gated bit-exact at L=128 (geometry unchanged)
and 256. The 12.6 Mbit image needs ≥45.5 URAM, so 60 is within 1.32× of the floor — the next
lever after this is the GEMV core, not weight banking.

**On silicon (MEASURED): 5,448.8 tok/s @ 125 MHz, 3/3 bit-exact, CYCLES = 22,941 == sim.**
Sweep: 100 MHz 4,359.0 · 111.1 4,843.3 · **125 5,448.8** · 142.9 fails 0/3 (clean edge).
Projection 5,449 → measured 5,448.8. One day, four bit-honest rungs: **1,883 → 2,484 → 3,512
→ 5,449** (2.9×). 10k now needs ~1.85×: pipeline the GEMV/non-linears to ~200 MHz, or cut the
~23k cyc floor (softmax overlap, attention restructure). Both are pure clock-or-cycles work
within the proven gate ladder.

---

## 12. The cycle-floor cut — 17,839 cyc/token (SIM, gated bit-exact)

Profiled the 22,941 with a per-FSM-state counter in the testbench. The split: GEMV run 12,706 ·
attention 5,216 (load 3,088 + compute 2,128) · readback+dequant 2,456 · act-quant 945 ·
LN 720 · GELU 532 · rest <400. So GEMV is the floor; the fat is attention + boundary.

Three cuts, gated bit-exact after each:
1. **Fused readback→dequant** (−1.2k): readback (2-cyc ymem pipeline) streams into vec_dequant
   in flight; `gemvy_bank` and the entire G_DQ pass deleted.
2. **GELU folded into G_RB** (−0.5k): mlp dequant words feed vec_gelu in flight; S_GELU is now
   just a setup state.
3. **P-wide attention load + ctx writeback** (−3.6k): vec_attn ld/ctx ports → P×32 wide.
   16 head-loads/token at 24 wide rows each (was 192 scalar). Ctx emits one P-row per cycle.

22,942 → **17,839 cyc/token** (−22%). @125 MHz = **7,082 tok/s PROJECTED**, target 10k = 178 MHz.

Fmax prep: all heavy multiplies registered into 2-stage pipelines (one extra cycle each loop):
G_AQ 64×40 mult, vec_dequant 32×25 + 96-bit barrel shift, vec_attn q·k tree and prob·v lanes,
LN load squares (DSP retime). Total latency cost +187 cyc — buys silicon clock headroom.

**On silicon (MEASURED): 9,295.4 tok/s @ 166.7 MHz, 3/3 bit-exact, CYCLES = 17,930 == sim.**
Sweep: 125 MHz 6,971.6 · 142.9 7,967.5 · **166.7 9,295.4** · 200 fails (PLL has no 187.5 step:
fclk snaps 166.7 -> 200). Constraint 8.0 ns; impl WNS −0.266 → silicon factor 1.60 at the edge.
1.71x over the morning's 5,448.8; 12.4x over last week's 752. 10k = 179 MHz: one PLL step away.
Next spin: 3-stage act-quant (mux | mult | round) targets WNS ≥ 0 at 8 ns → 200 MHz silicon
(11,154 tok/s). BRAM gotcha: the AXI shim pushed 291/288 RAMB18 — pos_emb rows pad to a
1k-row BRAM tile, so TMAX 64 -> 48 saved nothing; TMAX=32 (= Kevin attention window) freed
4 tiles -> 283/288 fits.

---

## 13. 10k broken — 11,143.9 tok/s MEASURED @ 200 MHz, 3/3 bit-exact

The last sub-8 ns path was act-quant's BRAM-read -> source-mux -> 64x40 DSP chain. Split into
3 stages (mux | mult | round/sat): impl closes WNS +0.011 @ 125 MHz - first non-negative build.
Silicon: **200 MHz, 3/3 bit-exact, CYCLES 17,947 == sim.** PLL steps 1000/N — no fclk between
166.7 and 200, so the gate is exactly 200 or 9.3k.

**11,143.9 tok/s.** vs targets: 10k cleared with 11% margin. The day: 5,449 -> 6,972 -> 7,968 ->
9,295 -> 11,144. 14.8x over last week's 752; 8.8x over the laptop ORT best; 1,013x over the A53.

Ceiling: GEMV reads 12.8k URAM words; cycle floor ~12.8k -> 200 MHz = 15.6k single-stream.
Next: GEMM batch N=2/4/8 (keystroke speculative streams) -> 22k-43k aggregate. ROADMAP-100K.

---

## 14. Batch GEMM N=4 — four streams bit-exact, 11,786 cyc/token (SIM)

The architecture jump after 11k: keystroke speculative decoding wants N concurrent streams.
`gemm_banked_resident_vec` reads ONE resident weight word/cycle and feeds N=4 MAC banks —
the URAM weight bandwidth (the single-stream floor at 12.8k words/token) is shared across 4
tokens. Sequencer banks are stream-flattened (row = stream*ROWS + r); the non-linear phases
round-robin over streams; one GEMM run serves all four.

Gate: **4/4 streams bit-exact vs seq_ref full forward** (toks 48/10/100/77), 47,145 cyc /
4 tokens = **11,786 cyc/token → ~17.0k tok/s aggregate @ 200 MHz (PROJECTED)**, 1.5x single
stream. Next: overlap boundary phases with GEMM run → ~31k.

The whole debug was iverilog NBA traps, every failing path "last stream lives, earlier
streams X": packed-vector writes (`tok_outs[bs*9 +: 9] <=`, `xrow_p[0][b*W +: W] <=`) commit
at update-time bs/b → all land in the LAST stream's slot; runtime-indexed generate-wire reads
(`acc_flat[db]`) X; per-lane part-select on an unpacked element (`accb[b][L*32 +: 32]`) X.
Fixes: per-stream generate MAC banks + packed concat with constant-base case mux.
The killer: per-call act-quant streamed across stream switch — the pipeline tail wrote
xptr 0 of the NEXT stream, shifting all rows by 1 (s3 OK = no successor). G_AQN drain fixed.

**N=4 on silicon (MEASURED): 16,969.3 tok/s aggregate @ 200 MHz, 3/3, all four
streams bit-exact, CYCLES = 47,144 == sim.** Sweep: 125 → 10,605.8 · 142.9 →
12,120.9 · 166.7 → 14,141.1 · 200 → 16,969.3. impl WNS +0.012 (74 min).

The fit campaign (192 → 128 BRAM): wsel always@*-copy made Vivado trim the URAM
read register to 4 bits (weights → 400k LUTRAM); per-lane genvar MAC banks fixed
the unroll; banks pad to 512 rows; tok embeds → 8 SDP URAM, streamed at boot; pos
in BRAM; GELU LUT split into even/odd banks (2 lanes per BRAM pair); dq tables in
LUTRAM. Final: 82k LUT (70%), BRAM 128/144, URAM 64/64 — full house.

Silicon bugs found by the gate ladder: per-stream readback X (drain the dequant
tail before stream switch), gelu_lut_e/o.mem missing from BD (zeroed GELU table).

---

## 15. Ping-pong N=8 — 17,740.6 tok/s MEASURED @166.7 MHz, 8/8 bit-exact

Two engines, two groups of four streams: while the GEMM engine streams a weight
pass for group A, the NL engine runs embed/LN/attention/residual/argmax for B.
Descriptor handshake per call. Banks rate-halved (LN Q.22 -> 32-bit lanes, GELU
sat16), 88k LUT, BRAM 139.5/144, URAM 64/64.

Silicon: 125 -> 13,305.5 · 142.9 -> 15,206.3 · **166.7 -> 17,740.6 (3/3, 8/8 streams)**
· 200 fails (tighter mux path). CYCLES = 75,157 == sim. WNS +0.017, impl 47m45.

Profile: GEMM busy 51.2k cycles (2 weight passes), NL 24k overlapped. The lever:
N=8 MAC banks - one URAM pass per 8 tokens - 75k -> ~40k cycles -> ~40k tok/s
@200 MHz. LUT cost ~115k - knife-edge fit. Or LANES=64: 32k LUT MACs, ~50k tok/s
@250 MHz. (~33k cycles per 8 tokens, NL-bound: overlap covers GEMM 14k cyc.)

---

## 16. Single-pass N=8 and the 1,024-multiplier URAM cliff (SIM gated; fit 95.6%)

The merge: GE_IDLE serves both groups in ONE weight pass when their descriptors
agree on d_wbase (gwait patience 2,048 cyc before solo fallback). Core widened
N=4 -> N=8. Gate: **8/8 bit-exact, 69,469 cyc / 8 tok = 8,683.6 cyc/token —
23,033 tok/s @200 PROJECTED.** Then every full-design synth dumped the weight
URAMs into ~29,600 RAM64M8 (~450k LUT, "Infeasible ram_style=ultra").

Eight OOC A/Bs to corner it. Falsified in order: cross-boundary flattening
(keep_hierarchy on the instance — no change), the rd/word_p 512->4b trim
warnings (present in GOOD builds too; dont_touch removed them, URAM still died),
URAM cascade sizing (cascade_height=7 pinned — no change), y_lat's dead regs,
ternary-vs-case select form, the SETTLE state, the merge sequencer itself
(old pre-merge sequencer + new core also died), and the 32,768-bit acc concat
(two 16,384-bit halves died identically). The decisive matrix: standalone N=8
clean / in-context N=4 clean / in-context N=8 broken under ANY sequencer.

**Root cause: the multiplier population.** At N=8 the core holds 1,024 4x8
MACs; Vivado 2025.2's bulk multiplier optimization (runs right after DSP
absorption, before RAM mapping) restructures them and detaches the URAM read
register's loads — the mapper then sees a dead read port and refuses ultra.
512 mults (N=4) never trips it. **Fix: each stream's 128 MACs are one
keep_hierarchy leaf (`mac_bank`)** — an opaque boundary the bulk pass cannot
cross. URAM 64/64 restored on the first try and in every variant since.
(use_dsp on the leaves: 1,239 DSPs consumed, zero LUT saved — the accumulate
adds stayed in fabric. Reverted; DSPs reserved for the packed N=16 core. A
bottom-up DCP-link flow was prepared as Plan A — synth_gemm_core_dcp.tcl /
ooc_seq_pp_linkdcp.tcl, kept as fallback — but DCP stubs are parameterless,
so the instance's #(...) overrides must go if it is ever needed.)

The fit campaign that followed (126.6k LUT = 108% after the fix — single-pass
honestly costs +512 MAC lanes over the ping-pong's N=4 core):
- **ymem -> distributed** (both cores): 64-deep x 4,096-wide in BRAM burned a
  RAMB36 per 72b of width = 57 tiles for 256 kb; as LUTRAM ~4.7k LUT.
  BRAM 149.5 -> 100/144.
- **xres_bank -> one write site**: three FSM states wrote the bank; Vivado saw
  a multi-writer RAM, replicated LUTRAM ~3.5x (1,776 RAM64M8 ~ 14k LUT,
  "Infeasible ram_style=block" — in EVERY build since the bank existed, good
  ones included). Blocking-muxed xr_we/xr_wa/xr_wd into a single write after
  the case — same cycle, same values. xres now BRAM (+7.5 tiles), -14.3k LUT.
- SETTLE is sim-only now (y_lat latch); synth goes RUN->DRAIN directly, so
  silicon runs ~4 cyc/call (~0.8%) faster than sim — sim cyc/tok is an upper
  bound, and the tok/s claim is board wall-clock anyway.

**OOC fit: 111.9k LUT (95.6%), BRAM 107.5/144, URAM 64/64, DSP 403.** In
reserve if impl congests: 32->24-bit accumulators (|acc| <= 2^20 at k<=1024,
range-proven exact) ~ -10k LUT.

**Silicon (MEASURED): 19,275.6 tok/s aggregate @166.7 MHz — 8/8 streams
bit-exact, 3/3 runs. CYCLES = 69,172** (sim 69,469; the 297-cycle gap = 4 cyc x
~74 GEMM calls, exactly the sim-only SETTLE — silicon faster than sim, as
predicted in the §16 fit notes). Sweep: 125 -> 14,456.7 · 142.9 -> 16,522.0 ·
**166.7 -> 19,275.6** · 200 -> match=False x3 (no PLL step exists between).
Build: BD 1m21 + impl 109m12 at 95.6% LUT — the router closed WNS +0.026 from
-1.05 over ~14 passes. 200 MHz needs slack back: the 24-bit accumulator trim
(range-proven, ~-10k LUT) is the queued lever — at 69,172 cyc, 200 MHz = 23,131.

---

## 17. N=16 on silicon — 24,134.0 tok/s MEASURED @166.7 MHz, 16/16 bit-exact

The fit campaign's payoff (see §16 + commits 66646a7 and the three before it):
12 DSP-packed banks + 4 LUT banks, raw-48b drain with readback recovery, one
shared LayerNorm + one shared attention arbitrated between the two NL engines.
Fit: 106,493 LUT (90.9%), 1,171 DSP, 132 BRAM, 64/64 URAM. Impl WNS -0.203
(102 min); silicon overclock factor 1.37x to 166.7 holds, 1.64x to 200 fails
(garbage tokens, withheld) — consistent with every prior build's ~1.6 ceiling.

Sweep: 125 -> 18,100.5 · 142.9 -> 20,686.3 · **166.7 -> 24,134.0 (3/3, 16/16
streams)** · 200 -> match=False. CYCLES = 110,494 vs sim 110,791: the 297-cycle
SETTLE signature AGAIN (the merged design makes the same ~74 GEMM calls as the
N=8 single-pass — both groups share every pass, so the same sim-only settle
count). Prediction methodology: two for two.

Ladder context: 19,275.6 -> 24,134.0 = +25.2%. The arbitration tax (+20.7k cyc
vs the unbuildable dual-private design) is the known next target: attention
overhead (344 cyc/head at T=1, mostly softmax+drain latency) shrinks both the
NL time and the collision windows. Then P=16 act/RB boundary, then more
streams on 151-LUT/64-DSP banks, then 250 MHz.

---

## 18. The fit campaign + the 2.0/DSP proof + the softmax cut

**The fit campaign (overnight 06-05→06, three gated steps, all committed):**
N=16 with dual NL engines was 16/16 bit-exact in sim at 90,115 cyc — and
needed 325k LUT on a 117k device. The receipt:
1. **DSP eviction cascade** (366→189k): the 2-DSPs-per-packed-MACC culprit was
   NOT the multiply (27s×9s = 1 DSP, toy-proven) — Vivado absorbed the
   recovery SUBTRACTS into DSPs (C-A2:B2 mode), overflowing 1,248 and evicting
   LN/attention's 32×32-class mults to fabric at ~1.1k LUT each.
   `use_dsp="no"` on two wires.
2. **Raw-48b drain + readback recovery** (189→152.3k): 64 pairs × 48b ==
   128 lanes × 24b == 3,072b, so ymem/drain are encoding-transparent; recovery
   moved to the P-wide readback path (4 pairs/cycle) with a sumact side-mem.
   mac_bank_dsp: 5.4k → **151 LUT** each. Zero cycle change.
3. **One shared LN + one shared attention** arbitrated between the engines
   (fixed priority, hold-until-done; LN/attn never nest → no deadlock), freed
   283 DSPs → 4 more streams flipped to DSP banks (ND=12): **106.5k LUT
   (90.9%), 1,171 DSP**. Cost: arbitration tax, 90,115 → 110,791 cyc.
Result on silicon: §17's 24,134.0.

**The 2.0/DSP impossibility proof** (`research/dsp3_pack_proof.py`): 3 INT4×INT8
MACs/DSP with a shared activation fails on two independent walls — three
no-bleed nibbles need 28 bits against the 27-bit port, and three K=1024 neurons
are 66 bits of state against a 48-bit accumulator (side terms are O(log K)).
Drain windows collapse (D=0); 5-per-2-DSP dies the same way. 1.2M randomized
K=1024 lane-products, exhaustive K=1, adversarial corners: 0 mismatches on the
proven 2.0 scheme. **Consequence: N=16 is the stream ceiling at LANES=128**
(L=256 needs 2,048 DSPs — also dead). The 100k identity: 16 × 250 MHz / 40k cyc.

**Softmax latency cut**: dead wait-states between exp/sum/reciprocal phases
removed (arithmetic untouched). N=16: 110,791 → **103,879 cyc** (16/16);
N=8: 68,183 → 64,727 (8/8). Fit 107.4k LUT. 25,671 @166.7 SIM — bitstream
building from commit 210d385 at this state.

Next per the 100k chain: GE-engine profile-led cuts (AQ/RB overlap, group-gap
fills, GE_IDLE pre-AQ) toward the ~40k floor, then the 7ns-target timing
campaign for 200/250 MHz silicon.

**§18 silicon addendum: 25,744.5 tok/s MEASURED @166.7 (16/16, 3/3).** The
softmax-cut build closed at WNS −0.044 (best yet — the cut shortened a real
path) and ran 103,582 cyc on silicon vs 103,879 sim: the 297-cycle SETTLE
signature, third consecutive build. 200 MHz still fails (needs 1.61×; the
timing campaign remains the gate to 30k+ in one step).

## 19. Split-brain — 68,799 cyc / 16 tok, 38,768 tok/s @166.7 (SIM, 16/16 bit-exact)

The §7607751 research play, built and gated: two fully independent N=8 cohorts
(`cohort_engine` = sequencer_pp with one stream-group; merge/GWAIT/aq_eng
DELETED — a cohort never shares a weight pass, so the whole desync problem
dissolves) sharing only `weight_bank_tdp` (the §2f3ba17-proven xpm TDP-ultra
URAM image, loader on port A at boot, cohort-1 reads port A / cohort-0 port B)
plus ONE arbitrated LN + attention + embed (fixed priority, hold-until-done;
LN/attn never nest → no deadlock — the §18 sharing, unchanged).

**Gates (both green, iverilog):**
- `run_gemm_sb` — GEMM-level split-brain, cohorts on DIFFERENT shapes/weight
  bases started SKEW cycles apart (the read-port-independence proof):
  **ALL_BITEXACT** across 6 configs (same-shape skewed, qkv-vs-proj, K=1024
  both ways, the |acc|=2^20 corner at skew 0, all-DSP ND=8).
- `run_sb_seq` — full 4-layer forward, every stream vs
  `seq_ref.full_forward_signals` (x4/lnf/head/tok): **16/16 bit-exact at ND=0
  AND ND=6/cohort** (=12 DSP-banked streams, the §18 build config; identical
  cycles — DSP recovery rides the readback path, zero cycle change, as §18).
- **68,799 cyc / 16 tok = 4,299 cyc/token** vs 103,879 single-engine = **1.51×**;
  38,768 tok/s @166.7, 46,512 @200 (SIM). The ~63k research target missed by
  ~9% — the residual is the LN/attn arbitration tax now hit by two engines
  with no GWAIT slack to hide it (profile-led cuts remain on the table).

Two gate-harness bugs found en route, RTL innocent both times: (1) tb_gemm_sb
waited on `c0_done && c1_done` but each cohort's done is a ONE-CYCLE pulse and
the deliberate skew guarantees the pulses never overlap → latch sticky (the
3×30-min TB_TIMEOUTs were vvp grinding to the 5M-cycle limit; real run is
seconds); (2) three harness configs based cohort-1's weights INSIDE cohort-0's
G0·K0 footprint, so c1's load overwrote c0's upper groups — mismatch counts
(1024 = 8 streams × 1 group, 4095 = 4 groups with one lucky match) fingered
the overlap exactly. Overlap now asserted in the harness.

Next: OOC fit of `sequencer_sb` (the LUT question: 2× GE FSM + dequant + gelu
vs ONE of each shared at §18's 107.4k; ND=12 total DSP banks), then BD+impl →
silicon. PROJECTED if the §17/18 sim→silicon pattern holds: ~38.7k tok/s
@166.7 — past the 30k line in one step, before the timing campaign.

## 20. The split-brain fit campaign — NC=7 fits at 97.0%, 36,805 tok/s SIM

First OOC of `sequencer_sb` (N=16, ND=6/cohort): **143,552 LUT = 122.6%** of
the device. The hierarchical autopsy vs §18's 107.4k single-engine:
duplicated dequant engines + the per-cohort distributed dqm/dqe ROM copies
(~12k), a fatter shared attention (14.1k @ 32 DSP vs 8.2k @ 50 — LUT-evicted
because the DUPLICATED AQ multiplier eats 88 DSP/cohort vs pp16's 96 total),
duplicated gelu, and a coh0/coh1 asymmetry that is const-prop through the
fixed-priority embed arbiter (coh0 gets `gnt0 ≡ req0` and shrinks; coh1's
10.5k nl_engine is the honest size).

**Cut 1 — the shared dq+gelu channel (committed 800bf48):** one vec_dequant +
ONE dqm/dqe ROM copy + one vec_gelu serve both cohorts behind a third
hold-until-done arbiter (the LN/attn pattern; new GE_DQW park state; a cohort
holds the channel for one whole GEMM-call readback, so beats never
interleave; the ROM fetch free-runs 2-stage on the shared side, cycle-aligned
with the old local mwr/dq_mant registers). **143.6k → 127,177 LUT (108.6%),
BRAM 143 → 133. Cycle cost: +17 cycles TOTAL** (68,799 → 68,816 for 16 tok) —
the cohorts' readbacks essentially never collide. 16/16 bit-exact.

**Cut 2 — NC=7 (N=14):** one fewer stream per cohort drops one 6,720-LUT
mac_bank per cohort: **113,659 LUT = 97.04%, BRAM 131/144 — FITS.**
14/14 bit-exact, **63,410 cyc / 14 tok = 4,529 cyc/token → 36,805 tok/s
@166.7 / 44,157 @200 (SIM)**. En route NC=7 flushed a real bug: the
host-readback cohort select used a high-bit slice (`rd_stream[CSH]`,
`[CSH-1:0]`) that only works when NC is a power of two — cohort 1's streams
read back shifted by one. Now arithmetic (`>= NC`, `- NC`); N=16 regated
bit-exact at identical cycles. The AXI shell (`gemv_axi_seq_sb`, IDCODE
"SQSB") is N-agnostic: 16-slot register map padded, upper TOK_OUT reads 0.

Bitstream building (BD+impl @166.7, C:/kevbuild/stage3_seqsb14_bit).
PROJECTED on the §17/18 sim→silicon pattern: **~36.8k MEASURED** (+43% on the
25,744.5 record). The NC=8 path back to ~38.7k: range-prove + narrow the AQ
multiplier (frees ~80 DSP) → un-evict attention (−5.8k) → ~121k, then one
more cut (TMAX=16 or the asymmetry) closes it.

**§20 silicon addendum: 36,970.7 tok/s MEASURED @166.7 (14/14, 3/3) — NEW
RECORD, +43.6%.** Impl: BD 55s + synth/impl 1h32m, WNS **−1.876** (the LN
sum-of-squares path again, degraded by 97% density — 75k failing endpoints).
STA says ~127 MHz; silicon ran 166.7 bit-exact anyway (1.31× of the ~1.76×
observed margin) — 63,113 cyc on silicon vs 63,410 sim, the −297 SETTLE
signature for the fourth consecutive build, CYCLES identical across all
reps. Sweep: 125 → 27,728.0 (already a record) / 142.9 → 31,689.2 / 166.7 →
**36,970.7** / 200 → match=False (WITHHELD, as every build). Driver:
`board/pl_seq_sb.py` (--n 14, IDCODE SQSB). Next levers, in expected-value
order: the 7ns timing campaign (200 MHz at NC=7 = 44.3k — the WNS −1.876
says LN needs another pipeline stage first), AQ-mult narrowing → NC=8
(~38.7k @166.7), and KV-DDR for context.

## 21. 46,604.4 tok/s MEASURED @200 MHz — N=16, the first 200-clean build

Three timing levers, each found by retiring the previous worst path, all
bit-exact and composed (gates: N=16 69,120 cyc 16/16; N=14 63,676 14/14;
total cycle cost of all three: +0.46%):
1. **LN un-retimed** (two splits): the qsh barrel-shift registered (Y0_r)
   so it can't fuse with the Newton squarer — the WNS −1.876 impl path —
   and the output prod*gamma DSP registered before the >>>49 shift/pack.
   LN now has ZERO top-30 entries.
2. **AQ-mult 32×48** (range proof, `research/aq_range_proof.py`): lnt is
   31b by construction (|n|<√D × |gamma|<8; the LN sources are literally
   the 32-bit Q.22 regs), GELU hard 27b, ctx observed 24b; inv_sact =
   round(2^40/s_act) ≤ ~2^47 → [47:0] slice. Per-cohort glue 88→48 DSP;
   **attention un-evicted** (14,062 LUT/32 DSP → 8,249/50, the pp16 shape).
   NC=8 dropped 127,177 → 113,246 LUT (96.7%) — N=16 FITS.
3. **Attn sc_prod operand register**: qmem/kmem read → multiply split;
   one-product-group/cycle throughput unchanged. sc_prod gone from top-30.

Composed OOC NC=8: **111,295 LUT (95.03%)**, WNS −0.541 @6ns (2 endpoints,
both softmax `div_bit→er0` — the next lever, agent running) / −1.541 @5ns.
Impl (BD 50s + 1h46m): **WNS −0.616** (vs −1.876 the night before).

**Silicon: 125→ok, 166.7 → 38,837.0 (record for ~an hour), 200 →
46,604.4 — match=True 3/3, 68,663 cyc (sim 68,960 − 297, the SETTLE
signature's fifth consecutive build). 250 → match=False (softmax).**
The 100k identity is now 16 × 250 MHz / 40k cyc: the softmax lever buys
the clock (58.2k at this cycle count); the cycle-floor cuts buy the rest.

