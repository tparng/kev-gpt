# Genesys2 port notes

Working notes for the kev-gpt -> Genesys2 (X-HEEP + cv32e40px) port. See
`/home/tparng/.claude/plans/steady-mixing-storm.md` for the full phased plan
and its rationale; this file tracks what's actually been done and any
refinements discovered along the way.

## Sizing decision

`sizing.py` sweeps three options; **Option A (d=128, n_layer=2, n_head=4,
ctx=128, ~280KB)** is the first bring-up target. Option B (d=192, n_layer=3,
ctx=192, ~909KB) is the "real chat" target once Option A is verified end to
end on hardware. Run `python -m fabric.genesys2.sizing` to regenerate the
table.

## Phase 1 — done

- `fabric/genesys2/sizing.py` written and self-checked (KV260 config -> weight
  bytes match the project's own "~1.5MB" figure exactly; `kv_bank.sv`'s own
  512KB/48KB worked example reproduced exactly).
- SoC/board repo forked locally: `~/RVchatbot/kevgpt-genesys2-soc` (`git clone
  --local` from `~/RVchatbot/Genesys2AiChatbot`, so it stays independently
  bisectable). Its `.venv` (Python 3.8, fusesoc/edalize/etc.) was physically
  copied across rather than rebuilt from PyPI (no network cost, exact version
  parity) — the copy's embedded absolute paths (`activate`, `pyvenv.cfg`,
  every `bin/*` shebang) were rewritten from the original repo's path to the
  fork's path with `sed`; confirmed working (`fusesoc --version`,
  `import edalize`).
- **Gotcha hit and worked around**: invoking `make mcu-gen` via the repo-root
  wrapper Makefile hit the documented `$(PWD)`-staleness bug even harder than
  CLAUDE.md's own note implied — it resolved the venv python to a truncated,
  wrong path (`<repo>/./.venv/bin/python`, missing
  `hw/vendor/esl_epfl_x_heep/`) and `unset PWD` made it worse, not better, in
  this shell. Fix: invoke `make mcu-gen PYTHON_X_HEEP_CFG=... X_HEEP_CFG=...`
  **directly inside `hw/vendor/esl_epfl_x_heep/`** with the venv activated
  from that same directory — matches CLAUDE.md's documented workaround for
  `vivado-fpga`, evidently also required for `mcu-gen` in this environment.
  `configs/cv32e40px_ai_accel.py` referenced by the wrapper's Makefile as
  `configs/...` actually resolves inside `hw/vendor/esl_epfl_x_heep/configs/`,
  not a repo-root `configs/` (there isn't one) — another reason direct
  invocation is simpler than the wrapper here.
- Stock `mcu-gen` (unmodified `cv32e40px_ai_accel.py` config, i.e. still
  ai_accel's own config, not kevgpt's yet) verified working from the fork.
  `verilator-build` was still running as of this note (Verilator compiles the
  whole X-HEEP SoC, meaningfully slower than mcu-gen) — see the plan file /
  task tracker for its outcome.

## Phase 2 — RTL primitive-swap: done (edits), gate run: in progress

Added a `MEM_PRIMITIVE` parameter (default `"ultra"`, so the KV260 build's
own gates and bitstream are byte-for-byte unaffected) threaded through:

- `weight_bank_tdp.sv` (all 3 `xpm_memory_tdpram` sites: the DP!=0 even/odd
  column-parity banks `u_e`/`u_o`, and the DP=0 `u_tdp`).
- `kv_bank.sv` (the code bank `u_code`; the header bank `u_hdr` already used
  `"block"`, untouched).
- Threaded up through `gemv_banked_resident_vec.sv` (its `weight_bank_tdp`
  instantiation) and `sequencer_vec.sv` (both its `gemv_banked_resident_vec`
  and `kv_bank` instantiations), each gaining the same parameter with the
  same `"ultra"`-preserving default.

**Refinement over the plan's stated verification approach**: the plan said to
run `fabric/stage3/run_vec_kv.py` twice (once per `MEM_PRIMITIVE` value) to
gate this change. Having now read the actual RTL: `MEMORY_PRIMITIVE` only
appears inside the `` `ifdef SYNTHESIS `` branch of each XPM instantiation:
iverilog (which `run_vec_kv.py` uses) always takes the `` `else `` behavioral
branch, which doesn't reference `MEM_PRIMITIVE` at all. **The two parameter
values are therefore bit-identical under iverilog by construction** — running
the gate twice would exercise the exact same simulated code path twice, not
two different ones. One run (at the default) is the correct regression check
for "did this parameter plumbing break anything"; the real `"ultra"` vs
`"block"` distinction is synthesis-only and can only be checked by Vivado
(Phase 6/7's build step) and, ultimately, real hardware. Updated here rather
than silently deviating from the written plan.

Also note (not a change, just confirmed while reading `weight_bank_tdp.sv`'s
own header comment): the "true-dual-port" requirement this port routes around
is about the **XPM `MEMORY_PRIMITIVE="ultra"` inference guarantee**, not
independent dual-port access per se — Xilinx 7-series `BRAM36`/`BRAM18` is
*natively* true-dual-port (two fully independent read/write ports), same as
UltraRAM. `MEMORY_PRIMITIVE="block"` gives up UltraRAM's larger native
geometry and per-port bandwidth, not dual-port capability. `BANKW` (the
72-bit-wide banking `weight_bank_tdp.sv` uses above 512 bits) is a geometry
choice tuned for URAM's native 72-bit port width; left unchanged for this
port since XPM composes whatever primitive is chosen internally regardless
of `BANKW`, and re-tuning it for BRAM's native width is an area-efficiency
optimization, not a correctness requirement — deferred, not forgotten.

Still pending: training Option A's config (`model/train.py`) and exporting
(`model/export_fabric.py`) to get a real weight image, so
`fabric/stage3/run_vec_kv.py` can actually run (it needs
`fabric/export/goformer.npz`, which is gitignored/regenerable and did not
exist in this checkout). `torch` is not in `requirements.txt` (only needed
for `model/`, installed separately) and was not present in the venv; install
was in progress as of this note.

## Phase 3 — done

`fabric/genesys2/rtl/xheep_kevgpt_peripheral.sv`: a `reg_req_t`/`reg_rsp_t`
adapter wrapping the unmodified `sequencer_vec.sv`, porting
`gemv_axi_seq_vec.v`'s register file and semantics (identical register map,
identical IDCODE "SQRV") onto X-HEEP's flat single-cycle valid/ready
protocol — collapses AXI's separate AW/W/B/AR/R channel state machine into
one always-ready decode (`reg_rsp_o.ready = reg_req_i.valid`), since X-HEEP's
register_interface has no separate address/data phase to manage and this
peripheral has no internal wait-state condition.

`fabric/genesys2/tb/tb_kevgpt_peripheral.sv`: directed protocol-conformance
test (IDCODE readback, STATUS before/after `go`, a full `go`..`done` round
trip through the register interface into `sequencer_vec` and back, CYCLES
readback) — no weights loaded, so this checks the bus handshake only, not
model correctness (that stays `run_vec_kv.py`'s job, unmodified). **Verified,
first try**: `KEVGPT_PERIPH_VERDICT pass=True errors=0` (iverilog -g2012;
compiled clean against `sequencer_vec.sv`'s real instantiation closure per
`VENDOR_MANIFEST.txt`; `$readmemh` warnings for LUT-init `.mem` files are
expected/harmless here — those are only populated by `run_vec_kv.py`'s own
setup step for a real model-correctness run, irrelevant to this protocol
test). 7,196 register polls / 14,391 simulated cycles for one full
`go`..`done` pass at the deployed engine's default parameters
(P=16, LANES=128, NLAYER=4, WWORDS=262144, TMAX=64).

## Phase 2 — model-correctness gate: BLOCKED, bigger than the plan anticipated

Trained an Option A checkpoint (`model/train.py --n-layer 2 --n-head 4 --n-embd
128 --block-size 128`, then `--qat --init-from`), exported both
`fabric/export/weights.npz` (195.6KB packed INT4 -- in the right ballpark of
`sizing.py`'s 204.1KB estimate; the gap is `sizing.py` using the KV260's
vocab_size=193 default for all three options rather than the real trained
vocab (57 chars for this corpus) -- a minor sizing.py correction, not
load-bearing) and `fabric/export/goformer.npz` (the golden-reference params),
then ran `fabric.stage3.run_vec_kv`.

**First failure, fixed**: `run_layernorm.py`'s `_ln_int_quantized` (called by
`seq_ref.py`'s `_ln`, which `model.goformer_kvq.IntKVQSequencer` uses every
token) hardcoded `D=256` -- `mean = S >> 8` and `for i in range(D)` never
looked at the actual input length. **And** `layernorm_vec.sv` (the RTL) had
the identical hardcoding one level deeper: `D` was a `localparam`, not a
`parameter`, so `sequencer_vec.sv`'s own `D` module parameter was silently
never reaching the LayerNorm block at all -- meaning `sequencer_vec.sv` has
in practice only ever been exercised at D=256, despite exposing `D` as if it
were already general. Fixed both (Python: derive `d_shift` from `len(X)`;
RTL: `D` promoted to a real parameter, `D_SH = $clog2(D)` replaces the
hardcoded `>>>8`, threaded through from `sequencer_vec.sv`). Both changes are
regression-verified as bit-exact-preserving at D=256
(`python -m fabric.stage3.run_layernorm` still gives
`LN_VERDICT bitexact=True mismatches=0/16384 ... pass=True`).

**Second failure, NOT yet fixed -- this is the one that needs a decision**:
`model/goformer_kvq.py` and `fabric/stage3/seq_ref.py` hardcode `NHEAD=4` /
`HEAD_DIM=64` as **module-level globals**, not per-instance state derived
from the actual model config, and use them as default arguments across
several functions in the KV-cache write/read path (`hadamard_rotate_q16`,
`position_ddr_row`, the `for h in range(NHEAD)` loops in `_store_kv`, the
`nh, hd = NHEAD, HEAD_DIM` in `_attn_step`). Option A's actual per-position
K/V vector is 128 long (d=128), but `position_ddr_row` still assumes
`nh=4, hd=64` (256 total) by default -- silently slicing past the end of a
128-long vector, raising `ValueError: min() iterable argument is empty`
several heads in.

Worse, this isn't just a "pass the right nh/hd" fix: the attention score
scale `1/sqrt(head_dim)` is baked in as an **exact numeric shift amount**,
correct only for `head_dim=64` specifically:
- Python: `seq_ref.py:63` `ISQRT = 3  # /sqrt(head_dim=64) = >>3`, used at
  `seq_ref.py:233`.
- RTL: `vec_attn_w.sv:53` `localparam integer SCORE_SH = 27;  // 2*VFRAC +
  ISQRT - SCORE_FRAC` -- a hand-computed literal, not derived from a
  parameter at elaboration time.

`1/sqrt(head_dim)` is only expressible as an *exact* integer right-shift
(what this whole fixed-point design relies on for bit-exactness) when
`head_dim` is a power of 4 (so its square root is itself a power of 2) --
64, 16, 4, 1, 256, .... Also, separately, `goformer_kvq.py`'s Hadamard
rotation path (`_HAD_SHIFT`, gated on `rotate=True`) asserts the same
"perfect square power of two" constraint on `HEAD_DIM`, though `run_vec_kv.py`
actually calls `IntKVQSequencer(..., rotate=False)`, so that specific path
isn't in the way for *this* gate.

**Why this changes the sizing picture from the approved plan**: `NHEAD=4,
HEAD_DIM=64` multiply out to `d=256` -- the KV260's exact deployed width.
Keeping `d=256` fixed sidesteps every one of these hardcoded constants with
zero further changes, but weight bytes scale as `12*d^2*n_layer`, so even
`n_layer=1` at `d=256` costs ~384KB in weights alone, already over Option
A's ~280KB target with nothing left for KV cache or margin -- not a workable
"safe bring-up" tier. Getting `sizing.py`'s Option A/B (`d=128`/`d=192`)
actually running requires real, scoped work: threading `nh`/`hd` as genuine
per-instance state through `seq_ref.py`/`goformer_kvq.py` instead of module
globals, and deriving `ISQRT`/`SCORE_SH` from the *actual* `head_dim` at both
Python-reference and RTL-elaboration time (restricted to `head_dim` values
that are powers of 4, so the shift stays exact) -- not a blind auto-fix,
since it touches the same numerically-sensitive fixed-point scaling this
whole project's "bit-honest" discipline is built to protect. See the
AskUserQuestion asked at this point in the session for how to proceed.

## Phase 2 continued — three real fixes made, gate still not green

After the "vary n_head, keep head_dim=64 fixed" decision above, retrained
Option A at `n_head=2, n_embd=128` (head_dim=64) and made three further
fixes, each regression-verified not to disturb the KV260 (D=256/NLAYER=4/
NHEAD=4) path:

1. `goformer_kvq.py`'s `_store_kv`/`_attn_step`: `NHEAD`/`HEAD_DIM` globals
   and a hardcoded `C = 256` replaced with `self.p["n_head"]` and
   `len(qkv_q16)//3` (both derived from the actual checkpoint). Added an
   assertion enforcing `head_dim == HEAD_DIM` so a future n_embd/n_head
   choice that breaks the fixed-head_dim=64 assumption fails loudly.
2. `fabric/stage3/tb/tb_seq_vec_kv.sv`: added `DVAL`/`NLAYERVAL`/`NHEADVAL`/
   `VOCABVAL` `` `define ``s (defaults 256/4/4/193, so nothing changes unless
   overridden) and threaded them into `sequencer_vec`'s instantiation
   (`.D`/`.D3`/`.D_MLP`/`.NLAYER`/`.NHEAD`/`.VOCAB`) -- previously only
   `P`/`LANES`/`TMAX` were wired, so this gate had *never* actually exercised
   `sequencer_vec.sv` at a non-deployed D/NLAYER/NHEAD/VOCAB despite those
   being real module parameters.
3. `run_vec_kv.py`: derived `d_model`/`nlayer`/`nhead`/`vocab` from the
   loaded checkpoint (`p["tok_emb"].shape`, `len(p["blocks"])`,
   `p["n_head"]`) instead of a hardcoded `4` at the `write_mems_wideword`
   call and nothing at all for the new testbench defines.

`sizing.py`'s Option B was also revised: `layernorm_vec.sv`'s mean/var divide
is *also* an exact shift, valid only for a power-of-2 D -- independent of
the head_dim=64-fixed constraint. With head_dim pinned at 64, D=n_head*64 is
a power of 2 only for n_head in {1,2,4,...}; there's no valid D between 128
and 256 under both constraints at once. Revised Option B to grow via
n_layer/context instead of D: `d=128 (n_head=2), n_layer=4, ctx=256` (was
`d=192, n_head=3, ctx=192`, which is no longer reachable at all).

**Ran the gate with all of the above -- `VVP_FAIL`, not a mismatch but a
simulation timeout.** The FSM hangs forever in state `S_ACL` (state 13,
`sequencer_vec.sv:754`, `if (adone_s && bdone_s) begin ...`), stuck at
`blk=0` -- it never even leaves the first transformer block.

**Root cause, traced (not fixed)**: `sequencer_vec.sv`'s twin-engine
attention scheduler (`u_attnA`/`u_attnB`, two `vec_attn_w` instances that
split the heads between them for throughput) assigns head indices via two
**hardcoded FSM states**, not a loop over `NHEAD`:
```
hh<=2'd0; hB<=2'd1;   // "pair 0": engine A does head 0, engine B does head 1
...
hh<=2'd2; hB<=2'd3;   // "pair 1": engine A does head 2, engine B does head 3
```
(`sequencer_vec.sv:562,586,770`). This is baked for exactly NHEAD=4 (two
fixed pairs), independent of the `NHEAD` module parameter -- unlike the
three fixes above (which were real parameters just never threaded), **there
is no parameter to thread here**; the pair count and head indices are
literal constants in the state-transition logic. There are also
head-index-derived magic offsets nearby (`vneedA = hh + 6`, `vneedB = hB +
4`, `sequencer_vec.sv:309-310`) whose derivation I have not traced and would
need to before trusting a fix.

This is a materially different and larger class of change than the three
fixes above: a real (if bounded) FSM redesign -- generalizing two hardcoded
pair-states into a genuine loop over `NHEAD/2` pairs -- touching
numerically/timing-sensitive control logic I have not fully traced (the
`vneedA`/`vneedB` offsets, and whether the KV-cache *write*-side states
S_KVW_S/S_KVW_F/S_KVW_W have the same hardcoded-pair pattern on the write
side, not just here on the read/attention side). Given the user's
"vary n_head" decision was specifically chosen *to avoid* the bigger,
riskier "generalize head_dim" RTL work, and this turns out to need real FSM
surgery anyway (just in a different place than expected), this changed the
risk/effort picture behind that decision -- flagged back to the user rather
than pushed through at this point in the session.

## Phase 2 continued — traced the scheduler in full; stopping short of rewriting it

User direction: generalize the pair scheduler rather than revert to n_head=4
or try a different sequencer. Traced it fully before touching anything
further (no RTL edits made past this point) -- it is **two interacting,
independently hand-tuned FSMs**, not the one simple pair-loop the earlier
framing assumed:

1. **The "main" KV writer** (`S_KVW_S`/`S_KVW_F`/`S_KVW_W`,
   `sequencer_vec.sv:568-588`): a blocking, serial writer that writes K then
   V for head 0, then K then V for head 1 (`kvw_h` 0->1, `kvw_kv` toggling),
   entered before the qkv GEMV even starts. On completion (`sequencer_vec.sv
   :584-587`) it hands off to (2), hardcoding `kvw_h<=2` (head *2*, literally
   assuming exactly 4 heads exist and heads 0-1 are already done).
2. **The "feeder" FSM** (`kvf_st`/`kvf_active`, driven combinationally at
   `sequencer_vec.sv:899-941`): writes the *remaining* heads (2, 3) in a
   deliberately shuffled order -- K2, K3, then V1, V0, V3, V2 (not V2, V3) --
   chosen so each engine's V-write lands in a specific gap in the *other*
   engine's read schedule (`sequencer_vec.sv:288-301`'s comment spells out
   the intent). This runs concurrently with pair-0's attention compute
   (`vec_attn_w` engines A/B), i.e. its writes are deliberately hidden under
   already-in-flight work -- a real latency-hiding optimization, not
   incidental structure.
3. The read side's `vneedA = hh+6` / `vneedB = hB+4`
   (`sequencer_vec.sv:309-310`) are the exact indices into this specific
   8-step interleaved order that gate when each engine's V-read may start
   without a read-before-write race against the feeder -- i.e. they encode
   the interleaving's timing contract, not an independent piece.

**Why I stopped here rather than rewriting it**: a safe generalization needs
either (a) faithfully extending this exact two-tier hide-the-latency design
to `NHEAD/2` pairs generically (real, delicate FSM design work -- the
overlap/gap structure that makes the shuffled V-order correct doesn't have
an obvious general form I could derive with confidence by inspection), or
(b) a deliberate *simplification* -- drop the overlap optimization entirely,
serialize all `2*NHEAD` K/V writes up front (simple K0..K(NHEAD-1),
V0..V(NHEAD-1) order), and gate `S_AST` on `kvp_done >= 2*NHEAD` for every
pair (no more per-pair partial gating, no more vneed/vpA/vpB interlocks at
all -- attention only ever starts once every write for the position is
already committed). (b) is very likely correct-by-construction and a small,
mechanical diff, at the cost of losing the write-under-compute latency
hiding (a real cycles-per-token regression on top of what this port is
already trading away elsewhere). I traced far enough to be confident (b) is
the right shape of fix, but did not make the edit: doing it correctly still
means resizing `kvw_h`/`kvp_done`/`pair` for a general (not just NHEAD=2)
bound, re-deriving the K/V write-order state transitions from scratch
(`sequencer_vec.sv:932-941`), and removing/proving-dead the vneed/vpA/vpB
interlocks (`sequencer_vec.sv:899-910`) -- real edits across several
interacting `always` blocks I want a waveform (VCD dump, not just the
pass/fail cycle counter this gate currently reports) to check against
before trusting, not just careful reading. That's a dedicated follow-up, not
a few more minutes.

## Phase 2 — RESOLVED: scheduler generalized, gate green at both shapes

Implemented the surgical version of fix (b) above -- not "wait for
everything" (too big a latency-hiding regression), but the smaller, more
precise fix the full trace made possible: keep the K-writes-first structure
and per-pair partial gating (which generalize with no changes at all —
`S_AST`'s existing `kvp_done >= (pair ? 4'd4 : 4'd2)` was already correct for
NPAIRS=1 without modification, since it only ever evaluates the pair=0
branch), and only replace the **V-write order** — plain ascending
(K0..K(NHEAD-1), V0..V(NHEAD-1)) instead of the hand-shuffled
(K0,K1,K2,K3,V1,V0,V3,V2) — since that shuffle, not the pairing/gating
structure itself, was the part with no general form. Four edits, all in
`sequencer_vec.sv`:

1. Added `localparam integer NPAIRS = NHEAD/2` (documented as staying valid
   only while `pair` (1 bit) can represent it, i.e. NPAIRS in {1,2} — this
   port's sizing options only ever need that).
2. `vneedA`/`vneedB` (the "V for this engine's head has committed" kvp_done
   thresholds): re-derived for the new ascending order (`NHEAD+h+1`), was
   `hh+6`/`hB+4` (exact only for the old shuffled order at NHEAD=4).
3. `S_CDR`'s next-pair transition: `if (pair)` -> `if (pair == NPAIRS-1)`;
   `hh<=2'd2; hB<=2'd3;` -> `hh<=(pair+1)*2; hB<=(pair+1)*2+1` (derived, not
   literal) — produces identical values to the original at NHEAD=4 (the only
   case checkable against the pre-port design), and is simply never taken at
   NHEAD=2 (NPAIRS=1, the first and only pair is always the last one).
4. The feeder's `KF_W` write-order case statement: replaced the 4-branch
   hardcoded shuffle with a 2-branch ascending-index increment
   (`kvw_h != NHEAD-1 -> kvw_h+1`, else advance phase/finish) — general for
   any NHEAD, not just 4.

Left untouched, deliberately: `S_KVW_S`/`S_KVW_F`/`S_KVW_W`
(`sequencer_vec.sv:568-588`) turned out to be **dead code** — grepped for
every place anything sets `st<=S_KVW_S` and found none; the real (and only)
write mechanism is the `kvf_st`-driven feeder, armed at `S_QKVRET` and
free-running (gated by `kvf_grant`) for the whole position, not a two-tier
"main FSM does heads 0-1, feeder does heads 2-3" split as an earlier,
incomplete read of this file suggested. Left in place rather than deleted —
removing it cleanly would also mean touching `kvf_grant`'s condition and
`kvw_src`'s mux (both reference `S_KVW_F`), a larger diff for a
purely-cosmetic win.

**Verified, both bit-exact via `fabric.stage3.run_vec_kv` (the same golden
reference, `model.goformer_kvq.IntKVQSequencer`, unmodified)**:
- Option A shape (d=128, n_layer=2, n_head=2, head_dim=64): `VEC_KV_VERDICT
  match=True` -- `gen=[1,29,30,34,1,40]` decodes to `'once' -> ' him s'`,
  exact token-for-token match with the golden model.
- **Regression check at the original deployed KV260 shape** (d=256,
  n_layer=4, n_head=4, head_dim=64 — a fresh checkpoint trained specifically
  for this check, `data/ckpt_kv260regress.qat.pt`): also `VEC_KV_VERDICT
  match=True`, `avg_cyc=14948`/token — in the range the project's own stated
  estimate for this exact engine implies (`fabric/stage3/README.md`: "~16,100
  cyc/token" for the PE=256 design; this run is P=8/LANES=128, a different
  point on the same curve, but the same order of magnitude, not a red flag).
  This is the load-bearing check: it proves the write-order generalization
  produces byte-identical *decode output* to the original hand-tuned
  schedule at the one shape it's possible to check it against, not just "no
  hang."

Sizing options remain as revised (`fabric/genesys2/sizing.py`): Option A
(d=128, n_layer=2, n_head=2, ctx=128) is now gate-verified end to end, not
just sized on paper.

## Phase 4 — SoC integration: scaffolding was already done, one build gotcha found + fixed

Resuming a later session found `fabric/genesys2/rtl/xheep_kevgpt_peripheral.sv` already
vendored into the fork (`hw/vendor/esl_epfl_x_heep/hw/ip/kevgpt_seq/`) along with
`kevgpt_seq.core`, `configs/cv32e40px_kevgpt.py`, a stripped
`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`, a lint waiver, and
`sw/applications/kevgpt_bringup/` (probe IDCODE / go / poll done / read tok_out) — all
dated earlier the same day, `make mcu-gen` + `make verilator-build` had already been run
(build tree present) but with no confirmed-successful completion log.

**Found and fixed: two orphaned `make app PROJECT=kevgpt_bringup` invocations stuck at
99.9% CPU for 14h and 5h respectively**, going nowhere. Root-caused via strace + a
timed/polled direct re-run of the `cmake` configure step (confirmed compiler-ID/ABI
detection succeeds fine; the hang is silent, right after "Detecting CXX compile features
- done", i.e. inside `sw/CMakeLists.txt`'s own logic, not toolchain detection). Cause:
both invocations manually passed `SOURCE=../../../sw/` on the command line — that value
is correct **only** for `make vivado-fpga` invoked directly inside
`hw/vendor/esl_epfl_x_heep/` (this fork's `CLAUDE.md` documents it there specifically).
For `make app`, `external.mk`'s own `SOURCE ?= $(SW_TO_SW_REL_PATH)` already
auto-computes the right (4-levels-up) path when left unset — passing the 3-up value
overrides that with one level too shallow. `sw/Makefile`'s
`source_path := $(realpath $(mkfile_path)/$(SOURCE))` then resolves to a **nonexistent**
directory (`hw/sw/` instead of the real `sw/` at repo root); GNU Make's `$(realpath)`
silently returns empty string for a nonexistent path (no error), so
`SOURCE_PATH = $(source_path)/` becomes literally `/`. `CMakeLists.txt:69`'s
`FILE(GLOB_RECURSE new_list FOLLOW_SYMLINKS ${SOURCE_PATH}*.h)` (and the matching
`*.c`/`*.s`/`*.S`/`*.cpp` globs right after) then walks the **entire filesystem root**,
following symlinks — explaining both the CPU-pegged hang and why it produced no error,
ever. This fork's own `CLAUDE.md` already documents the correct bare-wrapper form for
`ai_accel_bringup` (`make app PROJECT=ai_accel_bringup`, no `SOURCE=`) — the bug was
copying the *other* gotcha's `SOURCE=../../../sw/` value into this different invocation
context, not a defect in the wrapper itself.

**Fix applied (this session): killed both stuck processes, `rm -rf
hw/vendor/esl_epfl_x_heep/sw/build`, re-ran `make app PROJECT=kevgpt_bringup` with no
`SOURCE` override** — configures and links in seconds, produces `main.elf`/`main.hex`/
`main.bin` (`sw/build/`, `main.elf` 178,584 bytes). The run does still hit the
already-documented, benign trailing `$(PWD)`-staleness failure in the `mem_usage.py`
report step (`make[1]: .../.venv/bin/python: No such file or directory`, Error 127) —
expected per this fork's own `CLAUDE.md`, fires only after the binary is already fully
built, safe to ignore for simulation purposes.

**Lesson for future invocations in this fork: never pass an explicit `SOURCE=` to
`make app`** (any PROJECT, not just `kevgpt_bringup`) — let `external.mk` compute it.
Only `vivado-fpga`, invoked directly inside `hw/vendor/esl_epfl_x_heep/`, needs the
explicit `SOURCE=../../../sw/` override.

## Phase 4 — DONE: first CPU+peripheral+UART Verilator co-sim, green

`make verilator-run` (invoked directly inside `hw/vendor/esl_epfl_x_heep/` with
`SOURCE=../../../sw/`, the *other* documented `$(PWD)`-staleness workaround — the
top-level wrapper resolves the venv's `fusesoc` to a nonexistent path for this target
too, same class of bug as the one just fixed above but in the opposite direction: this
target needs the explicit override, `make app` needs it omitted): cv32e40px boots,
`kevgpt_bringup`'s control-plane smoke test runs end to end through the real X-HEEP bus/
crossbar into `xheep_kevgpt_peripheral` into unmodified `sequencer_vec`, and back out
over UART:

```
KEVGPT_PHASE,control_plane
KEVGPT_ID,0x53515256          -- "SQRV" (0x53='S',0x51='Q',0x52='R',0x56='V'), matches IDCODE
KEVGPT_STATUS_PRE,0x00000000  -- idle before go, as expected
KEVGPT_TOK_OUT,0
KEVGPT_CYCLES,14391
KEVGPT_PASS,control_plane
```

**Cross-check**: 14,391 cycles is an *exact* match to Phase 3's standalone iverilog
protocol testbench (`tb_kevgpt_peripheral.sv`) figure quoted above ("14,391 simulated
cycles for one full go..done pass at the deployed engine's default parameters"). Same
register-level transaction, same cycle count, now reached through the full SoC integration
(bus/crossbar/firmware) instead of a directed testbench driving the peripheral in
isolation — strong evidence the integration layer didn't perturb the timing-sensitive
handshake at all, not just "didn't crash."

**Scope note, deliberate**: this run has no real weight image loaded (`$readmem file not
found` warnings for `gamma_w.mem`/`gelu_lut_*.mem`/etc. are expected/harmless, same as
Phase 3's note) — `kevgpt_bringup/main.c`'s own comment says so explicitly: this is the
control-plane/handshake smoke test, and streaming a real trained weight image + asserting
`tok_out` against a precomputed value is deferred to Phase 5's `kevgpt_chat` (the
model-correctness side stays `fabric.stage3.run_vec_kv`'s job, per the project's
bit-honest rule — this phase only had to prove the bus plumbing, and did).

## Phase 5 — DONE: firmware streams real weights, bit-exact match, PASS

Re-exported Option A (`data/ckpt_optionA.qat.pt`) to `fabric/export_optionA/`
(`model/export_fabric.py` + a direct `goformer_full.save_params` call for the
golden-reference npz `run_vec_kv.py` needs) since `fabric/export/` had since been
overwritten by the KV260-regression checkpoint. 195.6KB packed weights, matching
`sizing.py` exactly.

**Three real bugs found and fixed before the firmware work could even be gated:**

1. **`xheep_kevgpt_peripheral.sv` was never actually parameterized to Option A.**
   Its `sequencer_vec` instantiation only threaded `P`/`LANES`/`NLAYER`/`WWORDS`/`TMAX`
   -- `D`/`D3`/`D_MLP`/`NHEAD`/`VOCAB` silently fell back to `sequencer_vec`'s own
   KV260-deployed defaults (256/768/1024/4/193). Added those five parameters (defaults
   now Option A's shape: 128/384/512/2/57) and threaded them into the instantiation,
   in both the fork's vendored copy and kev-gpt's canonical
   `fabric/genesys2/rtl/xheep_kevgpt_peripheral.sv` (kept in sync per
   `VENDOR_MANIFEST.txt`). Also explicitly overridden at the wrapper instantiation
   site (`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`) for visibility.
2. **P=16 (this instance's other never-actually-verified default) produces X-state
   `tok_out`** at the full-sequencer level -- found by running
   `fabric.stage3.run_vec_kv --p 16 --lanes 128` against the Option A export:
   `VEC_KV_VERDICT match=False`, `hw gen=[None]*6`. Every previously-green
   `run_vec_kv` result in this project (including every check earlier in this file)
   used P=8; P=16 is not exercised by any gate in the repo (`run_banked.sv` sweeps
   `lanes`, not `P`) and is apparently untested/broken at the `sequencer_vec` level.
   Not investigated further -- flagged in both peripheral copies' parameter comments,
   defaulted to P=8 (the only value this project has ever gated green). Re-ran the
   gate at P=8/LANES=128 against the Option A export: `VEC_KV_VERDICT match=True`,
   `hw=' him s' gold=' him s'`, `gen=[1, 29, 30, 34, 1, 40]` -- the exact config now
   deployed.
3. **Linker region overflow.** `configs/cv32e40px_kevgpt.py`'s memory_ss (inherited
   unmodified from `cv32e40px_ai_accel.py`) bumped from `[32]*5` (288KB total) to
   `[64]*5` (448KB total) for the ~293KB embedded weight blob (see below), but the
   *first* attempt left the code/data linker-section split at ai_accel's own
   `0x10000`/`0x10000` (64KB code, rest data) -- `ld: section .rodata will not fit in
   region ram0`, overflowed by 244,272 bytes, because ai_accel's own footprint is
   mostly buffers/state (so its split gives "data" the bulk of the room), while
   kevgpt_chat's ~293KB weight array is a `static const` -> `.rodata` -> the "code"
   region. Fixed by moving the split to `0x60000`/`0x10000` (384KB code, 64KB data).

**Architectural finding (the load-bearing one): the `wl_we`/`W_DATA` register path is
NOT a complete weight-loading mechanism.** `sequencer_vec.sv`, `layernorm_vec.sv`,
`softmax_f.sv`, `kv_bank.sv`, `gelu_lut.sv`/`gelu_lut2.sv` each `$readmemh` several
ROMs directly from hardcoded relative filenames -- `gamma_w.mem`/`inv_sact.mem`/
`dqm_w.mem`/`dqe_w.mem` (model-specific: LayerNorm gains + dequant scales, depend on
the trained checkpoint) and `gelu_lut.mem`/`gelu_lut_e.mem`/`gelu_lut_o.mem`/
`inv_lut_lo.mem`/`inv_lut_hi.mem`/`exp_lut.mem`/`gumbel_lut.mem`/`seed.mem` (fixed
math tables, not model-specific) -- completely independent of `wl_we`, which only
feeds `gemv_banked_resident_vec`'s big weight matrix (+ the embed tables appended
into the same `wrom.mem` stream, per the wide-word doc comment above). This matches
`webchat/demo/a53_daemon.py`'s own "baked-ROM bitstream" phrasing for the KV260's real
deployment: the *whole* model (not just the big matrix) is `$readmemh`-baked into the
bitstream at Vivado synthesis time; `wl_we`/`W_DATA` streaming exists for the gate
harness's convenience (swap weight sets across iverilog runs without re-synthesizing),
not as the production loading path.

First `kevgpt_chat` Verilator run (firmware streaming only `wrom.mem`'s content via
`wl_we`, nothing else present) completed cleanly -- no hang, `KEVGPT_CYCLES` matching
the gate's `avg_cyc` almost exactly -- but every generated token was `0`
(`KEVGPT_FAIL,generate_mismatch`). Root cause confirmed directly from the sim log's
own `%Warning: gamma_w.mem:0: $readmem file not found` lines (and the matching ones for
every other ROM above) -- none of those files existed in the Verilator build's working
directory, so LayerNorm gains/dequant scales/GELU/softmax/gumbel LUTs were all
garbage/zero regardless of the correctly-streamed big weight matrix. **Not a firmware
or protocol bug** -- confirmed by checking `weight_bank_tdp.sv`/
`gemv_banked_resident_vec.sv` have no `$readmemh` at all (the big matrix is genuinely
`wl_we`-only), while grepping the other RTL files for `readmemh` turned up all eight-ish
hardcoded filenames directly.

**Fix (simulation-only, this phase's actual scope): generated the missing ROM `.mem`
files for the Option A checkpoint (via `seq_ref.build` + `write_mems_wideword`, the
identical code path `run_vec_kv.py`'s gate uses, plus the same gelu-split/inv-lut/
gumbel-lut post-processing steps) directly into the Verilator build's working
directory** (`hw/vendor/esl_epfl_x_heep/build/.../sim-verilator/`) before re-running.
Re-ran `make verilator-run` -- the `$readmem file not found` warnings are gone, and:

```
KEVGPT_GEN,pos=3,tok=1     KEVGPT_GEN,pos=6,tok=34
KEVGPT_GEN,pos=4,tok=29    KEVGPT_GEN,pos=7,tok=1
KEVGPT_GEN,pos=5,tok=30    KEVGPT_GEN,pos=8,tok=40
KEVGPT_CYCLES,22464
KEVGPT_CMP,i=0,got=1,want=1   ... (all 6 match)
KEVGPT_PASS,generate
```

Bit-exact match against `model.goformer_kvq.IntKVQSequencer.generate_greedy` (the same
golden reference `run_vec_kv.py` gates against): `got == want == [1, 29, 30, 34, 1, 40]`
(`"once"` -> `" him s"`), `Program Finished with value 0`. This is the first time real
compiled RISC-V firmware, through the real X-HEEP bus, has driven `sequencer_vec`
through a full multi-token generation and produced output matching the Python
reference -- not just a protocol-conformance smoke test (Phase 4's scope) but genuine
model correctness end to end.

**Scope note for Phase 6/7**: this fix (dropping generated `.mem` files into the sim
build directory) only works because Verilator's `$readmemh` reads from its CWD at
simulation start. **Real Vivado synthesis needs the equivalent files present at
*synthesis* time**, in whatever relative location `weight_bank_tdp.sv`/etc.'s
`$readmemh` calls resolve against inside the Vivado build -- untested, flagged here
rather than assumed. `fabric/genesys2/gen_chat_fw.py` (the weight-image/expected-token
header generator, still useful for the firmware side) and this session's inline
ROM-file generation should be consolidated into one script before Phase 6 so both the
firmware weight stream and the synthesis-time ROM files come from one invocation
against one checkpoint, not two separately-run snippets.

## Phase 6 — DONE: real Genesys2 bitstream, synth+place+route+bitgen clean, positive timing

Six real bugs found and fixed, in order, each verified before moving to the next
(bit-honest before fast, applied to the synthesis flow itself, not just the RTL):

1. **`$readmemh` resolution for Vivado**: same architectural fact as Phase 5 (the
   `wl_we` register path only loads the big GEMV weight matrix; LayerNorm gains,
   dequant scales, and the fixed LUTs are baked in via hardcoded `$readmemh` calls),
   but for *synthesis* the resolution directory is different from Verilator's:
   confirmed empirically (a synth-only `make synth` checkpoint, deliberately run
   *before* committing to the full multi-hour build) that Vivado's `launch_runs
   synth_1` spawns synthesis as its own `vivado -mode batch` subprocess with CWD =
   `<project>.runs/synth_1/` -- generated the same 21 ROM `.mem` files there (and,
   redundantly but harmlessly, at the project root) before every synth/build run
   this phase.
2. **Wrong FPGA target entirely.** First "successful" build (0 errors, positive
   timing, 71.46% LUTs) turned out to have synthesized `xilinx_core_v_mini_mcu_
   wrapper` (the **ai_accel-combined** base design) instead of `xilinx_core_v_
   mini_mcu_wrapper_kevgpt` -- `core-v-mini-mcu.core`'s `genesys2` target's
   `toplevel` is a `no_ai_accel`-flag ternary between exactly those two wrappers;
   kevgpt's own dedicated `genesys2_kevgpt` target (already defined in the .core
   file, `toplevel: [xilinx_core_v_mini_mcu_wrapper_kevgpt]`) exists specifically
   to avoid this, and I simply used the wrong `FPGA_BOARD=` value. Caught by
   grepping the generated `.tcl` for `set_property top` before trusting the run --
   worth doing on every future FPGA_BOARD invocation in this fork.
3. **`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv` was never added to the `.core`
   file's fileset**, only referenced as a `toplevel` value -- `ERROR: [Synth 8-439]
   module 'xilinx_core_v_mini_mcu_wrapper_kevgpt' not found` at RTL elaboration
   once the *correct* target was actually used for the first time. Added it to
   `rtl-fpga`'s `files:` list alongside the other two wrappers.
4. **`gpio`'s `no-clock-gate` fileset never fires for `genesys2_kevgpt`.**
   `pulp_platform_gpio.core`'s `target_genesys2? (no-clock-gate)` is FuseSoC's
   implicit `target_<name>` flag, exact-match per target name -- `genesys2_kevgpt`
   being a *separate* target name (not a flag on `genesys2`, a deliberate choice
   documented in `core-v-mini-mcu.core`'s own comment, since FuseSoC's
   list-conditional syntax wasn't confirmed to support the compound `&&` a
   three-way ternary would need) means it silently misses every
   `target_genesys2?`-gated dependency elsewhere in the vendor tree. Grepped the
   whole tree for `target_genesys2?`; found exactly one real hit (this one, gpio's
   `gpio_input_stage` -- `ERROR: [Synth 8-439] module 'gpio_input_stage' not
   found`) and added a matching `target_genesys2_kevgpt?` line.
5. **`WWORDS` oversized** (fixed earlier in Phase 5's session but re-confirmed
   costly here): the untightened 262144-word default alone put LUTs at 123.75%
   before place&route could even be attempted. Already fixed to 8192 in Phase 5;
   confirmed again here as a real, necessary fix (not premature optimization).
6. **The big one: twin attention engines (`u_attnA`+`u_attnB`) blew both LUTs and
   DSPs, independent of `LANES`.** A synth-only checkpoint at the *correct* target
   (P=8, LANES=64, WWORDS=8192, everything else right) still showed **252,194 LUTs
   (123.75% of the xc7k325t's 203,800)** and **804/840 DSP48E1 (95.71%)** --
   `LANES` tuning (128->64) barely moved either number (LUTs 252,194->243,617,
   DSPs unchanged at 804), because `LANES` only sizes `gemv_banked_resident_vec`
   (0 DSPs, uses BRAM), not attention. A Vivado hierarchical utilization report on
   the completed checkpoint (`report_utilization -hierarchical`, run against the
   already-synthesized `.dcp` -- seconds, not another 15-minute resynth) showed
   `u_attnA`+`u_attnB` alone at 494/804 DSPs (61%) and 88,206/243,617 LUTs (36%),
   entirely inside `vec_attn_w.sv`'s own logic (wide fixed-point Q*K/prob*V
   multiplies, e.g. a ~48x~128-bit multiply, replicated across all P=8 lanes,
   times two concurrent engines) -- not further decomposable, i.e. genuinely
   structural, not a parameter.

   **Fix: collapsed `sequencer_vec.sv`'s twin-engine (head-PAIR-concurrent)
   attention scheduler to a single engine processing all NHEAD heads serially.**
   Traced the full structure before touching anything (matching this file's own
   Phase-2 precedent for changes of this class): removed `hB`/`pair`/`NPAIRS`/
   `qsel`/`bdone_s`/`vpB`/`ctxbufB`/`vneedB`/`u_attnB` and `kv_bank`'s second read
   port entirely (tied `rd2_start=1'b0` permanently at the instantiation --
   `kv_bank.sv` itself is UNCHANGED, still shared with the KV260 gate); `hh`
   becomes the single active head index, looping 0..NHEAD-1; `S_AST`'s pair gate
   (`kvp_done >= (pair?4:2)`) becomes a per-head gate (`kvp_done >= hh+1`); `S_ALD`
   drops its A-then-B two-phase Q-stream for a single-phase stream; `S_ACL` waits
   on `adone_s` alone; `S_CDR` drains `HR` cycles/head (was `2*HR`/pair) into
   `ctxv_bank[hh*HR+...]`, looping back to `S_AST` for the next head until
   `hh==NHEAD-1`. `vneedA`'s formula (`NHEAD+hh+1`) needed no change -- it already
   only depended on `hh`, never on the pair structure.

   **Verified bit-exact both shapes** (`fabric.stage3.run_vec_kv`, same golden
   reference): Option A (D=128/NLAYER=2/NHEAD=2/LANES=64) `VEC_KV_VERDICT
   match=True`, `gen=[1,29,30,34,1,40]` unchanged, avg_cyc **4038->4174 (+3.4%)** --
   far cheaper than the ~2x expected, since Q-streaming to the twin engines was
   already partially serial (stream to A, *then* to B) even in the concurrent
   design; only the K/V-read-and-compute phase actually overlapped. KV260
   regression (D=256/NLAYER=4/NHEAD=4/LANES=128) also `match=True`,
   `gen=[1,41,29,26,34,1]` unchanged, avg_cyc 14948->15492 (+3.6%). Re-verified
   Phase 5's full firmware `kevgpt_chat` end-to-end too: `KEVGPT_PASS,generate`,
   36324->37548 cycles (+3.4%), same match.

   **Result, re-synthesized: LUTs 252,194->93,651 (123.75%->45.95%)** -- fixed
   completely, as expected. **DSPs 804->803 (95.71%->95.60%) -- essentially
   unchanged**, which was NOT expected: a fresh hierarchical report on the new
   checkpoint showed `u_attnA` alone (the one remaining instance) now uses 384
   DSPs, up from 247 in the twin-engine version, for bit-identical RTL just
   invoked without a concurrent twin -- apparently a Vivado optimization-context
   effect (different surrounding logic changes DSP-inference/sharing decisions for
   the same module), not something controllable from the RTL as written. Flagged,
   not chased further -- 803/840 is still *under* 100%, and the real test is
   whether it places & routes, not whether the DSP count matches my mental model.

   **It does.** Full `make vivado-fpga FPGA_BOARD=genesys2_kevgpt SOURCE=../../../sw/`
   (synth->place->route->bitstream, ~35 min from a fresh synth) completed with
   **0 Errors, 0 Critical Warnings**: `place_design completed successfully`,
   `route_design completed successfully`, `write_bitstream completed successfully`,
   `Bitgen Completed Successfully`. Final placed utilization: **LUTs 96,424/203,800
   (47.31%)**, **DSPs 803/840 (95.60%)**, **Block RAM 371.5/445 tiles (83.48%)**.
   **Timing: WNS=2.640ns, TNS=0.000, "All user specified timing constraints are
   met."** -- comparable to or better than this fork's own ai_accel reference build
   (WNS=2.360ns per `CLAUDE.md`). Produces
   `.../genesys2_kevgpt-vivado/openhwgroup.org_systems_core-v-mini-mcu_1.0.5.bit`
   (11.4MB).

## Phase 6/7 real-hardware bring-up — DONE, bit-exact AND cycle-exact on real silicon

The board was physically connected (FT2232H JTAG on channel 0, FT232R UART, both
onboard) and powered on this session, so the "mandatory hands-on" step from the
original plan turned out to be doable directly: JTAG-programmed, loaded firmware via
GDB, and read real UART -- all from this environment, no separate human-operated
terminal needed. Corrected one stale-doc gotcha before trusting anything: this
fork's actual `sw/device/target/genesys2/x-heep.h` has `UART_BAUDRATE 115200` (not
9600 -- `QuadxHeep/CLAUDE.md`'s "9600 8N1" note is accurate for *its own* unmodified
X-HEEP copy but stale for this fork, which diverged; trusted the real file over
either doc).

**Programming**: `vivado -mode batch -source
openhwgroup.org_systems_core-v-mini-mcu_1.0.5_pgm.tcl -tclargs xc7k325tffg900-2 ...bit`
(FuseSoC's own `--run` is a no-op in this lineage, documented in the sibling
QuadxHeep repo) -- `INFO: SUCCESS! FPGA xc7k325tffg900-2 successfully programmed`.
`openocd -f quad-x-heep-openocd-genesys2.cfg` then found the RISC-V debug TAP
cleanly: `JTAG tap: riscv.cpu ... found`, `Examined RISC-V core; found 1 harts`.

**Process-hygiene gotcha, hit repeatedly**: orphaned `openocd`/`cat
/dev/ttyUSB0` processes from earlier attempts (in this session and, based on one
very stale PID, an earlier one) silently held the JTAG USB interface and/or the
UART tty exclusively, causing `LIBUSB_ERROR_BUSY` on reprogram attempts and
multiple simultaneous readers splitting/dropping UART bytes unpredictably (a run
that had genuinely completed successfully on the target produced an *empty*
capture file because 5 stale `cat` processes were racing for the same bytes).
Fix each time: `pgrep -af "cat /dev/ttyUSB0|riscv32-corev-elf-gdb|openocd"`,
kill everything found, **reprogram the bitstream fresh** (guaranteed full fabric
reset, sidesteps needing to reason about JTAG-reset-vs-peripheral-reset scope --
`reset_config none` in this fork's OpenOCD config means `monitor reset halt`
does NOT assert a hardware reset line, so stale peripheral state like
`xheep_kevgpt_peripheral`'s `done_latched` can survive a JTAG-only CPU reset
between two firmware loads), then restart OpenOCD, confirm exactly one UART
reader, and only then load.

**`kevgpt_bringup` (`TARGET=genesys2`), real hardware, first try after the above**:
```
KEVGPT_PHASE,control_plane
KEVGPT_ID,0x53515256          -- "SQRV", correct IDCODE readback on real silicon
KEVGPT_STATUS_PRE,0x00000000
KEVGPT_TOK_OUT,0
KEVGPT_CYCLES,4124
KEVGPT_PASS,control_plane
```

**`kevgpt_chat` (`TARGET=genesys2`), real hardware, after the process-hygiene fix**:
```
KEVGPT_WEIGHT_WORDS,73856
KEVGPT_STATUS_PRE,0x00000000
KEVGPT_GEN,pos=3,tok=1  ... pos=8,tok=40
KEVGPT_CYCLES,37548
KEVGPT_CMP,i=0..5: all match
KEVGPT_PASS,generate
```
`got=[1,29,30,34,1,40]` -- bit-exact with the Python golden reference (same as
every gate this session) **and cycle-exact with the Verilator simulation
(37,548 cycles both places)**, not just functionally matching. `"once" -> " him
s"` now confirmed on real Genesys2 silicon, weights streamed via the real
`wl_we` register path (not a testbench shortcut), model-correctness end to end.

Debugging note for future sessions: a `nm`-based "nearest preceding symbol"
lookup for a halted PC can point at the wrong function if the real containing
function has no exported symbol (a PC sitting in the CPU's ordinary post-`main()`
exit sequence -- `soc_ctrl_set_exit_value`/`soc_ctrl_set_valid`/`wfi` -- briefly
looked like an infinite loop in the unrelated `_fstat` stub this way; disassembly
around the actual PC resolved it in seconds once checked directly).

## Real-hardware token rate: ~11,900-12,000 tok/s (Option A, 50MHz)

Measured directly, not estimated: added `KEVGPT_PASS_CYCLES,pi=<n>,cyc=<count>`
per-pass logging to `kevgpt_chat/main.c` (each pass's own hardware cycle counter,
read right after that `go`->`done` round trip, on top of the existing summed
`KEVGPT_CYCLES` total) so per-token latency could be read directly off real
silicon instead of only inferred from the aggregate. Rebuilt (`TARGET=genesys2`),
reflashed, reran on the physical board:

```
KEVGPT_PASS_CYCLES,pi=0,cyc=4124   (prompt)      KEVGPT_PASS_CYCLES,pi=5,cyc=4184   (gen)
KEVGPT_PASS_CYCLES,pi=1,cyc=4136   (prompt)      KEVGPT_PASS_CYCLES,pi=6,cyc=4196   (gen)
KEVGPT_PASS_CYCLES,pi=2,cyc=4148   (prompt)      KEVGPT_PASS_CYCLES,pi=7,cyc=4208   (gen)
KEVGPT_PASS_CYCLES,pi=3,cyc=4160   (prompt/gen)  KEVGPT_PASS_CYCLES,pi=8,cyc=4220   (gen)
KEVGPT_PASS_CYCLES,pi=4,cyc=4172   (gen)
KEVGPT_CYCLES,37548   -- sum of the above, matches the earlier (unbroken-down) run exactly
KEVGPT_PASS,generate  -- same bit-exact [1,29,30,34,1,40] result, unaffected by the added logging
```

Per-pass cost grows linearly with context length (+12 cycles/pass, T=1..9 --
attention scans more committed positions each step), a clean, predictable curve,
not noise.

**Clock frequency confirmed from the actual Vivado config, not assumed**:
`hw/vendor/esl_epfl_x_heep/hw/fpga/scripts/genesys2/xilinx_generate_clk_wizard.tcl`
sets `CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {50}` -- 50MHz -- matching firmware's own
`REFERENCE_CLOCK_Hz` in `sw/device/target/genesys2/x-heep.h` exactly (if these
disagreed, UART would have transmitted at the wrong baud and produced garbled
text, not the clean output actually received -- an independent cross-check that
50MHz is the real running frequency, not just a firmware-side belief about it).

**Result**:
- Steady-state generation (the 6 passes that produce an output token, avg 4190
  cycles): **83.8 us/token -> ~11,930 tokens/sec**.
- Including prompt-priming passes (all 9, avg 4172 cycles): **83.4 us/token ->
  ~11,985 tokens/sec**.

This is a per-token *decode* latency for one sequence (no batching in this
design), and it's fast for the reason this whole architecture exists: Option A's
weights (195.6KB) and KV cache live entirely in on-chip BRAM -- no DRAM
round-trip per token, no bandwidth wall. ~12,000 tok/s at 50MHz for a d=128/
2-layer model is the expected payoff of that design, not an outlier; it's the
project's central bet paying off, measured on real silicon rather than claimed
from simulation alone.

## Open items / next up

- Interactive `kevgpt_chat` UART loop (arbitrary user prompts, not just the one
  baked-in `"once"` self-check) is future work -- current firmware is a
  self-contained correctness check, not yet a live chat interface.
- Consolidate `gen_chat_fw.py` + this session's inline ROM-file-generation snippet
  into one script (used identically ~6 times this session; still two separately
  invoked pieces).
- DSP margin is tight (37 DSP48E1 slots free, 95.6% used) -- fine for this exact
  build, but leaves little room for future growth (e.g. Option B's larger config)
  without either reducing HEAD_DIM/precision inside `vec_attn_w.sv` (numerically
  sensitive, would need its own bit-exactness re-verification) or accepting
  another architectural change of this session's magnitude.
- The pre-existing, unrelated whitespace/formatting diffs in `ai_accel/` RTL and
  vendor `.core`/wrapper files in the fork (present before this port's work started)
  are still untouched -- not this port's concern, flagged in an earlier session.
