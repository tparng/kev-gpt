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
