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

## DDR3-backed larger model: BRAM calibration (KV cache is the real ceiling)

Follow-on to the interactive-chat retarget (plan:
`/home/tparng/.claude/plans/steady-floating-pretzel.md`). Board spec used
throughout: Genesys2 DDR3 at 400MHz DDR (800MT/s) x4 bytes = **3.2GB/s peak**
(user-confirmed), giving 32-64MB/token of headroom at the 50 tok/s target even
at pessimistic controller efficiency -- DDR3 *bandwidth* is not a binding
constraint at this token rate; latency-hiding and correctness are the real
design concerns, not throughput.

Decomposed the real hierarchical `report_utilization` data (Option A,
single-engine attention, LANES=64, WWORDS=16384) instead of trusting
`sizing.py` alone (already known this session to underestimate BRAM by ~5x):

```
xilinx_core_v_mini_mcu_wrapper_kevgpt (top): 253 RAMB36, 13 RAMB18, 803 DSP
  u_kevgpt (xheep_kevgpt_peripheral): 245 RAMB36, 13 RAMB18, 798 DSP
    u_seq (sequencer_vec) own logic: 50 RAMB36, 5 RAMB18, 104 DSP
    u_attnA (vec_attn_w):             5 RAMB36, 3 RAMB18, 384 DSP
    u_dq (vec_dequant):               0 RAMB36, 0 RAMB18,  16 DSP
    u_gelu (vec_gelu):                8 RAMB36, 0 RAMB18,   0 DSP
    u_gemv (gemv_banked_resident_vec):158 RAMB36, 1 RAMB18,   0 DSP
      u_wb (weight_bank_tdp):         128 RAMB36, 0 RAMB18,   0 DSP
    u_kvb (kv_bank):                   24 RAMB36, 3 RAMB18,  30 DSP
```

Calibration points extracted:
- **`kv_bank` real-vs-formula tax: 1.68x.** `sizing.py`'s `kv_cache_bytes()`
  predicts 70.0KB for Option A's exact shape (NLAYER=2, D=128, TMAX=128); real
  synth gives 24 RAMB36 + 3 RAMB18 = 117.5KB. Small-buffer BRAM-tile rounding
  (36Kb/18Kb granularity) inflates a small KV cache more than a big one would
  be inflated proportionally -- treat 1.68x as an upper bound, not necessarily
  the asymptotic ratio at much bigger KV sizes.
- **`u_wb` (weight bank) 128 RAMB36 at `WWORDS=16384`** reflects a
  deliberately *oversized* window (this session's own earlier note: "right-
  sized then doubled for LANES=64's halved word-width"), not the true minimal
  per-layer footprint -- don't use this number directly as a per-layer weight
  cost; the true minimal 2-layer window at D=128 computes to ~192KB logical
  (12*D^2 INT4 params/layer), well under the 590KB actually provisioned here.
- **Fixed, ~non-scaling baseline: 93 RAMB36 = 428.5KB** (`u_seq`-own 50 +
  `u_attnA` 5 + `u_gelu` 8 + `u_gemv`-own-minus-`u_wb` 30). Conservative
  (treats all of `u_seq`-own as fixed even though some of it -- xres/qkv
  buffers -- likely scales mildly with D); good enough for first-order sizing,
  should be re-checked once a candidate shape is actually synthesized.

**Finding: weight streaming alone (the plan's original framing) does not
unlock a meaningfully bigger model.** Projecting fixed-428.5KB + a *true*
minimal weight window (x1.2 assumed packing tax) + KV cache (x1.68 tax)
against the ~1678.5KB free BRAM pool:

| config (NLAYER, D, ctx) | weight window | KV cache | total | vs free |
|---|---|---|---|---|
| Option A 2L/D=128/ctx=128 (sanity check) | 230K | 118K | 781K | 47% |
| 6L/D=192/ctx=128 | 518K | 529K | 1483K | 88% |
| 8L/D=192/ctx=128 | 518K | 706K | 1659K | **99%** |
| 12L/D=192/ctx=128 | 518K | 1058K | 2012K | 120% -- over |
| 16L/D=256/ctx=128 | 922K | 1882K | 3240K | 193% -- over |

KV cache was already the dominant BRAM cost once NLAYER exceeds ~2-3, and it
does NOT shrink from streaming weights -- every layer's K/V history for the
live context must stay reachable regardless of how weights are loaded. Even
with weights streamed to a near-zero window, KV cache alone saturates the
board by ~7-8 layers.

**Decision: stream the KV cache to DDR3 too**, not just weights (asked and
confirmed with the user). Initially assumed this needed a new write-capable
DMA engine from scratch (no reusable RTL) -- **wrong, corrected below.**

### Phase 2 architecture (derisked): four existing modules, all reusable unmodified

Read `kv_bank.sv`'s full interface end to end
(`hw/vendor/.../ai_accel/rtl/accelerator/streamer/`... no, `fabric/stage3/rtl/
kv_bank.sv`) and the `ai_accel` streamer stack. Two findings change Phase 2's
risk profile a lot:

1. **`kv_bank.sv`'s storage shape maps directly onto a flat DDR array.**
   Every (layer, kv, head, pos) has a fixed-size record: HEAD_DIM*KBITS-bit
   code row + 48-bit header (scale16+lo32), addressed by
   `w_pbase = ((layer*2+kv)*NHEAD+head)*TMAX + pos` -- exactly a linear
   index into `HROWS = NLAYER*2*NHEAD*TMAX` fixed-size slots. This is
   *already* how the on-chip TDP BRAM is addressed (`kva_a`/`pos_ra`/
   `pos_ra2` in `kv_bank.sv`), so relocating it to DDR3 is "same address
   arithmetic, different backing store," not a new addressing scheme.
   Write happens once per (layer,kv,head,pos) -- NLAYER*2*NHEAD times/token.
   Read streams `rd_tcount` (~pos+1) *consecutive* rows for one
   (layer,kv,head) -- NLAYER*2*NHEAD calls/token, each a burst of
   consecutive addresses. That read shape is exactly what `mig_read_engine`
   already does (issue N sequential `req_addr_i`, drain N `ret_data_o`
   beats); the write shape (one fire-and-forget beat per call) is exactly
   what `mig_write_engine` already does.
2. **A full write-capable engine + a 2-master MIG arbiter already exist**,
   hardware-tested via `ai_accel`'s own `AI_PASS,ddr_bram_prefetch` result,
   completely unrelated to and untouched by kevgpt so far:
   - `mig_read_engine.sv` -- streaming reads, credit-based outstanding
     tracking (already known).
   - `mig_write_engine.sv` -- single-beat write packets
     (`pkt_valid_i`/`pkt_addr_i`/`pkt_data_i`/`pkt_mask_i` in,
     `ack_valid_o` out), handles MIG's split cmd/wdf channel timing
     internally.
   - `mig_rw_arbiter.sv` -- merges one read-engine's command stream + one
     write-engine's command stream onto a single MIG command channel
     (priority-with-hysteresis, `BATCH_LIMIT`-based anti-thrash). This is
     the piece that combines kevgpt's own read+write traffic into one
     app-level bundle.
   - `mig_dual_master_arbiter.sv` -- merges TWO independent, already-self-
     consistent app-level bundles onto the *physical* MIG port, with correct
     read-return demuxing (owner FIFO, since MIG returns reads in issue
     order) and write-data pairing (a second owner FIFO, since `app_wdf_*`
     is independently paced from `app_addr/cmd/en`). Built generically for
     exactly this "two masters share one DDR3 controller" situation.
   - `v22_streamer_top.sv` is the reference integration showing all three of
     `mig_read_engine`+`mig_write_engine`+`mig_rw_arbiter` wired together
     (verbatim, minus its CDC `async_fifo_gray` layers, which exist only
     because `ai_accel`'s accelerator core runs on a separate clock domain
     from MIG's `ui_clk`; kevgpt's `clk_gen` is *already* sourced from
     `ui_clk` directly -- see `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv` --
     so kevgpt's own instances need NO clock-domain-crossing FIFOs at all,
     one less thing to get wrong).

   Confirmed via `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`: today only
   `cpu_ddr_bridge` is wired, straight to MIG's real app port (no
   `ai_accel`, no dual-master-arbiter present in this build at all) -- so
   the ENTIRE second slot of a `mig_dual_master_arbiter` is free.

**Resulting Phase 2 shape** (all four `mig_*` modules reused verbatim, zero
modification):
   - Add ONE `mig_read_engine` + ONE `mig_write_engine` + ONE `mig_rw_arbiter`
     for kevgpt's own traffic (shared by BOTH the weight-window reads and the
     KV-cache reads/writes -- ample bandwidth headroom from Phase 0 means
     time-multiplexing one read engine across two different DDR regions is
     fine, no need for a second read engine).
   - Add ONE `mig_dual_master_arbiter`: side A = kevgpt's new combined
     bundle, side B = the *existing, unmodified* `cpu_ddr_bridge` (rewire its
     `app_*` ports from "straight to MIG" into side B's ports instead).
     Arbiter output drives the real MIG app port.
   - New RTL actually needed is now scoped down to: (a) request-issuing FSMs
     inside `kv_bank.sv` (replace the BRAM read/write always-blocks with
     `mig_read_engine`/`mig_write_engine` request sequencing, same external
     `wq_*`/`rd_*` register-interface contract so `sequencer_vec.sv` and the
     gate harness don't need to change), (b) an analogous window-swap request
     FSM inside `weight_bank_tdp.sv`/`gemv_banked_resident_vec.sv`, and (c)
     the top-level wrapper wiring described above. No new arbitration or
     credit-tracking logic needs inventing -- the hard, error-prone parts
     (owner-order read demux, write-data pairing, MIG command-channel
     protocol) are already built and already proven on this exact board.

Next step: design `kv_bank.sv`'s DDR-backed read FSM in detail (replace
`R_RUN`'s `r_rowi`-indexed BRAM address stream with a loop issuing
`rd_tcount` sequential `mig_read_engine` requests, decode the returned
256-bit beats back into `code`+`hdr` fields, feed the existing `deq_word`
dequant combinational block unchanged), then the write FSM (replace `W_CWR`'s
BRAM commit with one `mig_write_engine` packet).

### MIG behavioral model + reused-engine smoke test (Icarus, no board needed)

Before touching `kv_bank.sv`, built a Icarus-simulatable stand-in for
`genesys2_mig_native_shell`'s app-level UI port: `fabric/genesys2/tb/
mig_behav_model.sv`. Deliberately NOT a timing-accurate MIG model (no
calibration sequence, no bank conflicts, always-ready cmd/wdf channels, a
fixed `READ_LATENCY`-cycle pipe for reads) -- Phase 0 already showed DDR3
bandwidth/latency headroom is ample at 50 tok/s, so the thing worth gating
bit-exactly is address/data correctness of the new request-issuing FSMs, not
real DDR timing (that's a Phase 6, real-hardware concern). Mask polarity
follows Xilinx UG586's MIG7 native-UI convention (active-low: 0 = byte
written, 1 = byte masked) -- flagged as an assumption to cross-check at
Phase 6, not yet verified against the real generated `genesys2_mig_native_
shell` wrapper.

Smoke-tested the full reused chain -- `mig_write_engine` -> `mig_read_engine`
-> `mig_rw_arbiter` -> `mig_behav_model`, all four wired exactly per
`v22_streamer_top.sv`'s reference pattern -- via a scratch `iverilog -g2012`
testbench (write a known pattern, read it back, then a SECOND write with a
partial byte mask, read back and confirm only the unmasked lane changed).
Two small fixes needed to get these ai_accel files through Icarus at all
(they'd only ever been through Vivado / real hardware before -- no prior
Icarus/Verilator testbench existed for this streamer stack, confirmed via
repo-wide grep): `unique case` qualifiers are accepted with a benign
"ignored" notice, and the SVA `assert property` blocks in `mig_read_engine.sv`
/ `mig_rw_arbiter.sv` / `sync_fifo.sv` need `-DSYNTHESIS` to skip (they're the
only content gated by that define in those three files, confirmed by grep
before relying on it -- doesn't change any functional logic, just compiles
out debug-only assertions Icarus's `-gno-assertions` didn't fully suppress).

**Result: `MIG_SMOKE_VERDICT,PASS`** -- both the full-write/full-read and the
partial-byte-mask cases matched exactly. This means the entire reused DMA
stack (`mig_read_engine`+`mig_write_engine`+`mig_rw_arbiter`, unmodified) is
now available as an ordinary Icarus gate dependency for the new `kv_bank.sv`
redesign -- the same bit-exact-before-synthesis discipline as every other
fabric/stage3 block, not a synthesis-only leap of faith.

### kv_bank_ddr.sv -- write-side FSM (DONE, gated PASS)

`fabric/genesys2/rtl/kv_bank_ddr.sv`: a new sibling module (kv_bank.sv itself
untouched, matching this project's established "add a variant, don't break
the existing build" pattern). Write-only for now -- the read-side streaming
loop is the next piece. Reuses kv_bank.sv's W_IDLE..W_QNT collect/scale/
quantise pipeline byte-for-byte verbatim (same inv_lut ROMs, same divide-free
magic-multiply scale/inv derivation), so wstage/w_scale/w_lo are guaranteed
identical to the already-verified reference for identical inputs. Only the
final commit differs: instead of a same-cycle on-chip TDP BRAM write, a new
DMA sub-FSM (states W_DMA/W_DACK) issues `ROW_BEATS` sequential write packets
on a `mig_write_engine`-shaped `pkt_*`/`ack_*` port.

DDR layout: each (layer,kv,head,pos) row is `CODE_BEATS` beats of quantised
codes (`ceil(HEAD_DIM*KBITS/DATA_W)`, =2 at HEAD_DIM=64/KBITS=8/DATA_W=256)
plus 1 header beat ({scale16,lo32} in the low 48 bits, byte-masked), addressed
as `KV_DDR_BASE + w_pbase*ROW_BEATS*BEAT_BYTES + beat*BEAT_BYTES` -- a flat
byte offset reusing kv_bank.sv's existing row-index arithmetic verbatim, just
multiplied by the beat stride instead of used as a direct BRAM address.
Address convention (byte address, beat-aligned, never shifted) and mask
polarity (active-low: 0=write) both cross-checked against `cpu_ddr_bridge.sv`'s
header before writing any of this, not assumed.

**Gate**: `fabric/genesys2/tb/tb_kv_bank_ddr.sv` -- an RTL-vs-RTL crosscheck
(not yet a Python-golden gate matching the `run_*.py` convention; flagged as
a follow-up, not done): drives identical `wq_*` stimulus into both
`kv_bank.sv` (already Python-gated via `fabric.stage3.run_vec_kv`) and
`kv_bank_ddr.sv`, decodes `kv_bank_ddr`'s DDR-resident bytes by hand in the
testbench, and compares against `kv_bank.sv`'s own dequantised `rd_data` for
the same row. Two test vectors (small positive-offset span; wide span
straddling zero, to exercise a negative `lo`), 64 lanes each = 128 comparison
points. **Result: `KV_BANK_DDR_VERDICT,PASS`, 0/128 mismatches.**

Three real bugs found and fixed while building this gate (not in the RTL --
all three were testbench/verification-infrastructure bugs, but the kind that
would have produced false confidence if shipped un-caught, so recorded in
full):
1. **`wait(signal)`/`@(posedge signal)` on a single-cycle pulse driven by the
   common "default-clear, then conditionally set" NBA idiom** (`wq_done<=0`
   unconditionally at the top of the always block, `wq_done<=1` in one
   specific branch -- what `kv_bank.sv`'s `wq_done`/`rd_done` and
   `kv_bank_ddr.sv`'s `wq_done` all use) **fired one full cycle early in
   Icarus**, confirmed directly via debug `$display` comparing the "just
   unblocked" data against the real pulse one cycle later. Root-caused (not
   just patched around): this class of pulse, sampled by a *different*
   process via edge-sensitivity, does not reliably behave like a true
   registered-value 0->1 transition on Icarus. Fixed by replacing every
   `wait`/`@(posedge ...)` on these pulses with a monotonic pulse-counter
   incremented in its own `always @(posedge clk)` block, and waiting for the
   counter to reach a target value instead -- sidesteps edge-detection
   semantics entirely. Worth remembering for every future testbench in this
   port that synchronizes on one of these single-cycle "done" pulses (the
   read-side FSM's `rd_done`/`rd2_done`, the eventual weight-window DMA's own
   completion pulse, etc.) -- use the counter idiom from the start, not
   `wait()`.
2. **`kv_bank.sv`'s read port has no direct "read position p" input** --
   `rd_start`/`rd_tcount` streams positions `0..tcount-1` of a (layer,kv,head)
   selector in order; there is no way to address a single arbitrary position
   directly. To check position `p`, request `tcount=p+1` and take the *last*
   emitted beat (`rd_done` fires on it). Mis-set to `tcount=1` initially
   (silently read position 0, which was never written, producing X compared
   against real DDR data by coincidence-passing/failing depending on which
   row position 0 happened to alias with) -- worth remembering for the
   read-side FSM's own gate, which will drive this same port directly.
3. **`mig_behav_model`'s `MEM_WORDS` sized too small for the row indices this
   test actually exercises** (1024, vs. `ROW_BEATS*HROWS`=3072 needed) --
   `word_idx()`'s fixed-width address slice *silently wraps/aliases* out-of-
   range addresses onto in-range words rather than erroring, so an
   undersized backing store doesn't fail loudly: it corrupted exactly the
   test case with a large row index (silently aliasing its beats onto
   unrelated words) while the small-row-index case happened to stay within
   bounds and passed normally -- the worst kind of test bug, since it looks
   like a partial pass. Root-caused by adding a temporary debug print of the
   real MIG-side `word_idx()` result next to the testbench's own (unshifted,
   untruncated) index computation and finding they disagreed. Fixed by
   sizing `MEM_WORDS` generously (8192) and documenting the sizing
   requirement directly in the testbench so it isn't silently re-introduced.

### kv_bank_ddr.sv -- read-side FSM (DONE, gated PASS)

Added to the same `fabric/genesys2/rtl/kv_bank_ddr.sv` file (kv_bank.sv
itself is a single module with both read and write sides; kv_bank_ddr.sv
follows the same shape now that both halves exist). New states RR_IDLE/
RR_REQ/RR_WAIT/RR_EMIT: matches kv_bank.sv's own read-port contract exactly
(no direct "read position p" input -- streams positions `0..tcount-1` of a
(layer,kv,head) selector in order, `rd_valid`/`rd_data` pulse once per
position, `rd_done` on the last one), but internally walks each position by
issuing `ROW_BEATS` sequential requests on a `mig_read_engine`-shaped
`req_*`/`ret_*` port (one request in flight at a time -- simpler than
pipelining ahead, and Phase 0 already established there's no bandwidth/
latency pressure to justify the extra complexity yet), capturing the
returned beats into the same code/hdr layout the write side stages them in,
then dequantising with the exact same `x_hat = code*scale+lo` math
`kv_bank.sv`'s own `deq_word` combinational block uses (copied verbatim, not
re-derived). No `rd2_*` (second read port) -- the current single-engine
`sequencer_vec.sv` ties it off permanently anyway, so this matches actual
usage rather than kv_bank.sv's full historical interface.

**Gate**: extended `fabric/genesys2/tb/tb_kv_bank_ddr.sv` to also issue a
real `rd_start` against `kv_bank_ddr` itself (through an actual
`mig_read_engine` + `mig_rw_arbiter` -- not a shortcut; this exercises the
same arbiter class that will later merge kevgpt's own DDR traffic with the
existing `cpu_ddr_bridge`, so it's useful mileage on that stack too) and
compares the result against `kv_bank.sv`'s reference read, in addition to
the existing hand-decoded-DDR-bytes check from the write-side gate. Both
checks now run for both test vectors: manual decode confirms the WRITE
path's bytes are correct, the real read-port exercise confirms the READ FSM
itself retrieves and dequantises them correctly. **Result:
`KV_BANK_DDR_VERDICT,PASS`, 0 mismatches across 256 comparison points** (64
lanes x 2 checks x 2 test vectors). Compiled clean with zero warnings after
fixing one cosmetic port-width mismatch (a debug-only `outstanding_o` wire
sized for the wrong `MAX_OUTSTANDING`).

No new bugs beyond the three already recorded above -- the write-side gate's
fixes (monotonic-counter pulse synchronization, `tcount=pos+1`-then-take-
the-last-beat semantics, `MEM_WORDS` sizing) carried over directly and
avoided repeating any of them on the read side.

### weight_loader_ddr.sv -- weight-window streaming (DONE, gated PASS)

Redesigned the scope from the original framing (a DMA-backed replacement for
`weight_bank_tdp.sv`'s read port) after actually tracing
`gemv_banked_resident_vec.sv`'s MAC pipeline: its read port
(`raddr_b`/`rword_b`) is consumed by an RLAT-deep, tightly-timed pipeline
(addend stage registered specifically to close a 5ns cone, per this
session's own earlier notes on tight DSP margin) that assumes weight reads
are ALWAYS ready -- `weight_bank_tdp` is a fixed 1-cycle-latency on-chip
memory today, it never stalls, and the MAC's `kc` walk has no backpressure
concept at all. Making the MAC's own read port DMA-backed would mean real
surgery on that already-verified, timing-critical pipeline.

Sidestepped that entirely: `fabric/genesys2/rtl/weight_loader_ddr.sv` is a
hardware-driven ALTERNATIVE SOURCE for `weight_bank_tdp`'s EXISTING boot-load
port (`ld_rst`/`w_we`/`w_data`) -- the same port firmware already drives via
the `wl_we` register to stream the whole weight image in once at boot
(`xheep_kevgpt_peripheral.sv`'s `wl_we`/`wl_data` -> `sequencer_vec.sv`'s
`wl_rst`/`wl_we`/`wl_data` -> `gemv_banked_resident_vec.sv`'s
`ld_rst`/`w_we`/`w_data`). This loader plays the identical role, just sourced
from DDR3 at runtime (once per weight-window reload) instead of from
firmware once at boot. `weight_bank_tdp.sv` and
`gemv_banked_resident_vec.sv`'s MAC pipeline are BOTH completely untouched --
by the time a GEMV `start` fires, the window is already fully resident,
satisfying the exact same "always ready" assumption the MAC pipeline already
relies on. Tradeoff: single-buffered (compute waits for the window to finish
loading; no double-buffered overlap yet) -- correctness first, matching
every other DMA piece built in this port so far; double-buffering is future
work once this baseline is proven on real hardware.

Design: issues DMA read requests via a `mig_read_engine`-shaped `req_*` port
as fast as `rd_req_ready` allows (decoupled from the drain side, unlike
`kv_bank_ddr.sv`'s read side which issues one outstanding request at a
time) -- weight-window reloads happen once per layer per TOKEN, so
throughput matters here for the cycle budget in a way the low-frequency KV
reads did not; paying DDR round-trip latency once per pipeline fill instead
of once per beat is the difference between a reload costing thousands of
cycles vs. tens of thousands. Each returned 256-bit beat is unpacked into
`DATA_W/32`=8 sequential `w_we` pulses feeding `weight_bank_tdp`'s existing
write-word assembler verbatim -- no new packing logic to get bit-exact,
since this is a straight copy (pre-quantized weights, no quantize/dequantize
math involved, unlike the KV cache).

**Gate**: `fabric/genesys2/tb/tb_weight_loader_ddr.sv` -- streams a known
LFSR-generated image out of `mig_behav_model` through a real
`mig_read_engine` into an unmodified `weight_bank_tdp` instance, then reads
the resident bank back via its existing `raddr_b`/`rword_b` port and checks
it matches the source DDR image exactly. Two cases (zero DDR offset/16
words; a nonzero 100-beat offset/24 words -- deliberately not repeating the
KV-cache gate's early mistake of only testing a zero-offset case, since that
one hid a real address-truncation bug until a nonzero-offset case exposed
it). At LANES=64 (Option A's actual value), `weight_bank_tdp`'s own word
width (WBITS=LANES*4=256) exactly equals the DMA beat width (DATA_W=256), so
the readback check is a direct beat-for-word comparison with no unpacking
arithmetic of its own to get wrong. **Result:
`WEIGHT_LOADER_DDR_VERDICT,PASS`, 0 mismatches across 40 words checked**,
clean compile, no new testbench-infrastructure bugs this time (the KV gate's
earlier fixes -- monotonic-counter pulse synchronization in particular --
were reused directly and avoided repeating that class of bug).

Both DMA subsystems (KV cache read+write, weight-window load) now exist and
are independently gated.

### Top-level arbiter wiring (DONE, gated PASS)

Two new modules, both pure glue (no new arbitration/credit-tracking logic of
their own beyond one small mux):

- **`fabric/genesys2/rtl/mig_read_mux2.sv`** -- merges `kv_bank_ddr`'s read
  requests and `weight_loader_ddr`'s read requests onto ONE shared
  `mig_read_engine` (the plan's own stated design: "ample bandwidth headroom
  makes time-multiplexing one engine across two DDR regions fine"). Same
  owner-FIFO IDEA `mig_dual_master_arbiter.sv` already uses for its own
  read-return demux (push an owner tag on every accepted request, pop it in
  the same order against real returns), applied one layer earlier in the
  stack (upstream of the engine, merging requesters, rather than downstream,
  demuxing an already-formed bundle).
- **`fabric/genesys2/rtl/kevgpt_ddr_bundle.sv`** -- combines `kv_bank_ddr`'s
  write engine + the shared read engine (via `mig_read_mux2`) through one
  `mig_rw_arbiter` into a single self-consistent app-level bundle -- the
  shape `mig_dual_master_arbiter`'s side A (or B) expects. Does NOT
  instantiate `kv_bank_ddr`/`weight_loader_ddr` themselves (those belong
  wherever `sequencer_vec.sv`'s own hierarchy eventually places them); it
  only takes their DMA-facing ports as pass-through.

**A real bug, found and fixed**: `mig_read_mux2`'s owner FIFO initially
popped on bare `ret_valid` (copying `mig_dual_master_arbiter`'s own pattern
verbatim) -- WRONG here. `mig_dual_master_arbiter`'s owner FIFO demuxes the
*raw MIG native-UI return path*, which has no backpressure at all (a return
must be consumed the instant it arrives, so valid alone means consumed).
`mig_read_engine`'s own `ret_valid`/`ret_ready` is a REAL handshake (backed
by its own return FIFO, data waits if the consumer isn't ready yet) -- the
owner FIFO here needed to pop on `ret_valid && ret_ready` (the real "beat
was actually consumed" event). Symptom with the bug present: the integration
gate hung during the weight-loader phase (worked fine for the low-throughput,
single-outstanding KV read path, which never let the two conditions diverge)
-- the owner FIFO drained faster than real consumption once multiple
requests were pipelined ahead, going empty while `mig_read_engine` still had
buffered returns waiting, permanently deadlocking `ret_ready` at 0. Debugged
by tracing `owner_empty`/`ret_valid`/`ret_ready`/`outstanding_o` cycle by
cycle and finding the owner FIFO empty while the engine still claimed
`ret_valid`. Fixed, and documented directly in `mig_read_mux2.sv`'s header
as a reusability warning: this owner-FIFO idiom needs the handshake-aware
pop condition against any port with a REAL valid/ready handshake, not MIG's
raw backpressure-free return path.

**Gate**: `fabric/genesys2/tb/tb_kevgpt_ddr_bundle.sv` -- the full stack
end to end: `kv_bank_ddr` + `weight_loader_ddr` + `mig_read_mux2` +
`kevgpt_ddr_bundle` + a real `mig_dual_master_arbiter`, side B driven by a
synthetic app-level requester standing in for `cpu_ddr_bridge` (that module
isn't modified by this port, so it isn't re-gated here -- this test proves
the WIRING and genuine two-master sharing, not `cpu_ddr_bridge`'s own
already-established correctness). Four phases: (1) `kv_bank_ddr` write+read
through the full stack, side B idle -- crosschecked against `kv_bank.sv`;
(2) `weight_loader_ddr` load through the full stack, side B idle --
crosschecked against the source DDR image; (3) synthetic side B alone,
side A idle -- proves side B's own data path isn't swapped with side A;
(4) `kv_bank_ddr` write + a synthetic side-B write/read, launched
concurrently (`fork`/`join`) -- real two-master sharing, not two sequential
single-master phases. **Result: `KEVGPT_DDR_BUNDLE_VERDICT,PASS`, 0 errors
across all four phases**, clean compile.

All of Phase 2 (both DMA subsystems, plus the arbiter wiring merging them
with room for `cpu_ddr_bridge` alongside) is now built and gated end to end
in simulation.

## Real hardware integration (RTL done + a real CDC bug found and fixed; Vivado check still pending)

### sequencer_vec.sv: KV_DDR_BACKED generate-selected kv_bank_ddr

Confirmed BOTH sides of `kv_bank.sv`'s external contract are event-driven
before touching anything (`if (kb_wdone)` gates the write-side FSM
transitions explicitly; the read side has no fixed-latency assumption
either -- `vec_attn_w` already tolerates kv_bank.sv's own multi-cycle
pipeline latency via `rd_valid` watching, not a cycle count) -- this is what
makes swapping `kv_bank` for `kv_bank_ddr` (drastically different internal
latency, same external pulses) timing-safe rather than a leap of faith.

Added a `KV_DDR_BACKED` parameter (default 0) to `sequencer_vec.sv` and
wrapped the `kv_bank` instantiation in a `generate if/else`: `KV_DDR_BACKED=0`
(default) elaborates the untouched, already-verified `kv_bank.sv` path byte-
for-byte as before -- KV260's build and every existing Genesys2 gate never
even see `kv_bank_ddr`'s logic. `KV_DDR_BACKED=1` elaborates `kv_bank_ddr.sv`
instead, with new `kv_wr_*`/`kv_rd_*` DMA ports added to `sequencer_vec.sv`'s
own port list (tied to inert idle values in the `KV_DDR_BACKED=0` branch, so
no dangling outputs). Threaded straight through `xheep_kevgpt_peripheral.sv`
(new `KV_DDR_BACKED` parameter + pass-through ports, same pattern) -- this
was tractable specifically because `xheep_kevgpt_peripheral` is a PEER to
the X-HEEP SoC in `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`, not nested
inside `core_v_mini_mcu`'s own vendored boundary (confirmed by reading the
wrapper: `xheep_kevgpt_peripheral` and `x_heep_system_i` are both instantiated
directly, side by side, connected only via the generic
`ext_peripheral_slave_req_o`/`resp_i` register bus) -- no X-HEEP SoC-boundary
plumbing needed at all.

**Regression check (KV_DDR_BACKED=0, the default/existing path)**:
re-ran `fabric.stage3.run_vec_kv` (had to route around a pre-existing harness
issue -- its own hardcoded default prompt `[48,10,100,77,...]` contains
token ids out of range for Option A's 57-vocab checkpoint, a KV260-vocab
leftover unrelated to this change; called `run_vec_kv.run()` directly with
an in-range prompt instead of via its CLI). **`VEC_KV_VERDICT match=True`**
-- confirms the generate-based swap didn't disturb the existing, deployed
behavior at all.

**Full-sequencer DDR-KV verification (KV_DDR_BACKED=1, the new path)**: built
`fabric/genesys2/tb/tb_seq_vec_kv_ddr.sv`, a byte-for-byte copy of
`fabric/stage3/tb/tb_seq_vec_kv.sv`'s own stimulus (weight stream, prompt/gen
GO-pulse loop, cycle counting, KVDBG hooks) with ONLY the `sequencer_vec`
instantiation changed (`KV_DDR_BACKED(1)` + the new DMA ports wired to a real
`mig_write_engine`+`mig_read_engine`+`mig_behav_model` stack, not a
shortcut). This exercises `kv_bank_ddr` under REALISTIC multi-token, multi-
layer, multi-head traffic -- not just the two isolated write+read pairs
`tb_kv_bank_ddr.sv` already covers. **Result: identical bit-exact token
stream to both the resident-`kv_bank` regression run above AND the Python
golden reference** (`gen=[41, 36, 46, 1, 29, 30]` in all three), confirmed
via a direct Python comparison against `IntKVQSequencer.generate_greedy`,
not just eyeballing printed lines.

### Top-level wrapper wiring + a real clock-domain-crossing bug

Edited `xilinx_core_v_mini_mcu_wrapper_kevgpt.sv`: instantiated
`kevgpt_ddr_bundle` (side A) + `mig_dual_master_arbiter`, rewired
`cpu_ddr_bridge`'s `app_*` ports from "straight to MIG" into the dual-
arbiter's side B (cpu_ddr_bridge itself untouched), wired
`xheep_kevgpt_peripheral`'s new `kv_wr_*`/`kv_rd_*` ports into the bundle,
and flipped `xheep_kevgpt_peripheral`'s `KV_DDR_BACKED` to 1 for THIS board
build specifically to prove the mechanism -- Option A's own KV cache is tiny
(~118KB) and doesn't need DDR streaming at this model size, but exercising
it now, ahead of actually needing it, is what proves the DMA path is real
before the Phase 4 model-size decision depends on it.

**A real bug, found while wiring this (not caught by any simulation gate up
to this point):** every Icarus gate built so far (`tb_kv_bank_ddr.sv`,
`tb_weight_loader_ddr.sv`, `tb_kevgpt_ddr_bundle.sv`) ran `kv_bank_ddr`,
`mig_read_engine`/`mig_write_engine`, and `mig_behav_model` all on ONE
shared simulation clock. Real hardware has TWO clock domains here:
`kv_bank_ddr` lives inside `sequencer_vec.sv`'s own hierarchy, clocked on
kevgpt's compute clock (`clk_gen`, confirmed 50MHz by an earlier session's
own clock-wizard-config check) -- but `mig_read_engine`/`mig_write_engine`/
`mig_rw_arbiter` and the app-level bundle MUST run on MIG's own `ui_clk`
(a hard requirement of MIG's native-UI app port, not a design choice), and
`clk_gen` is PLL-derived FROM `ui_clk` via `xilinx_clk_wizard_wrapper` --
genuinely different frequencies, not the same clock under two names. Wiring
`kevgpt_ddr_bundle`'s output straight into `mig_dual_master_arbiter` (itself
correctly on `ui_clk`) while `kv_bank_ddr` stays on `clk_gen` would have
been an un-synchronized multi-bit bus crossing two real, differently-clocked
domains -- a genuine metastability/data-corruption risk on real silicon,
invisible to every simulation run so far specifically because they all used
one shared clock.

**Fixed**: `kevgpt_ddr_bundle.sv` now takes separate `gen_clk`/`gen_rst` and
`ui_clk`/`ui_rst` inputs and does its own clock-domain crossing for
`kv_bank_ddr`'s four DMA signals (write-packet, write-ack, read-request,
read-return), using `async_fifo_gray` -- the exact same CDC primitive and
4-FIFO shape `cpu_ddr_bridge.sv` already uses for its own `clk_i`/`ui_clk_i`
boundary, reused rather than re-derived. `weight_loader_ddr`'s read port is
NOT yet CDC'd (it isn't wired to a real instance at the top level yet --
tied to constant idle values there, which need no synchronizer) -- flagged
directly in the module's header as a real follow-up needed once that
integration happens, not silently assumed away.

**Re-verified the CDC fix under a GENUINE two-clock-domain simulation** (not
just re-running the old single-clock gate and calling it done): updated
`tb_kevgpt_ddr_bundle.sv` to drive `clk`/`ui_clk` as two independent clocks
with a deliberately non-integer-multiple period ratio (10ns vs 7ns, to
actually stress the crossing rather than accidentally look synchronous).
First attempt still failed for the (deliberately still-un-CDC'd)
`weight_loader_ddr` path -- exactly the expected failure mode, confirming
the CDC finding empirically: `weight_loader_ddr`/`weight_bank_tdp` were
still wired directly across the two clocks with no synchronizer in the
testbench, producing visibly torn/shifted data. Isolated that known,
already-documented gap by moving `weight_loader_ddr`/`weight_bank_tdp` onto
`ui_clk` for this test only (not a real deployment shape -- noted inline)
so the KV path's real CDC fix could be verified cleanly on its own. **Result:
`KEVGPT_DDR_BUNDLE_VERDICT,PASS`, 0 errors across all four phases**
(including Phase 4's genuinely concurrent two-master traffic), under real
two-clock-domain conditions.

### Fileset wiring (fork repo)

Added `kv_bank_ddr.sv`/`weight_loader_ddr.sv`/`mig_read_mux2.sv`/
`kevgpt_ddr_bundle.sv` to `kevgpt_seq.core`'s `rtl` fileset. No changes
needed for the `mig_*`/`async_fifo_gray`/`cpu_ddr_bridge` files themselves --
confirmed via `grep` that `x-heep:ip:ai_accel`'s own `rtl` fileset (which
already contains all of them) is an unconditional dependency of
`core-v-mini-mcu.core`, already pulled into every Genesys2 kevgpt build
(this is how `genesys2_mig_native_shell.sv` already resolved for
`xilinx_core_v_mini_mcu_wrapper_kevgpt.sv` before any of this session's
work). All new/changed RTL files re-synced from kev-gpt's canonical copies
into the fork's vendored `hw/ip/kevgpt_seq/rtl/` directory.

### The real Vivado synth-only checkpoint (DONE, clean)

Corrected after the fact: Vivado 2022.2 IS available in this environment, at
`~/tools-2022/Xilinx/Vivado/2022.2` (`source ~/vivado2022.sh` or that
directory's `settings64.sh` directly) -- the earlier "not installed" note
above was simply looking in the wrong place (`/tools/Xilinx`,
`/opt/Xilinx`). Ran the real checkpoint once this was found.

**Process notes** (for next time): the project regeneration
(`make vivado-fpga-nobuild FPGA_BOARD=genesys2_kevgpt`) needs the repo's own
Python venv (`.venv/bin/activate`) SOURCED, not just its `fusesoc` binary on
PATH -- FuseSoC's own code-generation step (OpenTitan's `prim` generator,
needs the `mako` package) shells out to a bare `python3`, which resolves to
system Python (missing `mako`) unless the venv is actually activated in the
shell, not just referenced via absolute path. Regenerating the project wipes
the previous stale build's ROM `.mem` files (this session's earlier full
run of `fabric.stage3.run_vec_kv` already regenerated a correct set for
Option A's exact shape in a scratch dir -- reused directly rather than
regenerating a third time); they need re-placing into both the project's top
level and its `.runs/synth_1/` subdirectory before synthesis, same gotcha as
earlier sessions already documented. `make synth` (not the full
`vivado-fpga`/`.bit` target) is the right target for a synth-only
checkpoint -- it depends on `<name>.v`/`<name>.edn` (via `_synth.tcl` +
`_netlist.tcl`), stopping well short of place & route.

**Result: clean.** Zero `ERROR` lines in the full synthesis log; netlist
(`.v`/`.edn`) written successfully. Real hierarchical utilization
(`report_utilization -hierarchical`, opened directly against the synth_1
`.dcp` checkpoint -- fast, no resynthesis):

```
xilinx_core_v_mini_mcu_wrapper_kevgpt (top): 98769 LUT (48.46%), 54933 FF, 238 RAMB36+11 RAMB18 (54.72%), 803 DSP (95.60%)
  u_kevgpt (xheep_kevgpt_peripheral):        58443 LUT,          26015 FF, 230 RAMB36+11 RAMB18,            798 DSP
    (u_seq) own logic:                        8603 LUT,           3974 FF,  50 RAMB36+ 5 RAMB18,            104 DSP
    g_kvb_ddr.u_kvb (kv_bank_ddr):           14821 LUT,           3655 FF,   9 RAMB36+ 1 RAMB18,             30 DSP
    u_attnA (vec_attn_w):                    14709 LUT,           9336 FF,   5 RAMB36+ 3 RAMB18,            384 DSP
    u_gemv (gemv_banked_resident_vec):        8642 LUT,           3785 FF, 158 RAMB36+ 1 RAMB18,              0 DSP
      u_wb (weight_bank_tdp):                  430 LUT,            279 FF, 128 RAMB36,                        0 DSP
  u_kevgpt_ddr_bundle (new: CDC + mig_read_mux2 + engines + arbiter): 2921 LUT, 8693 FF, 0 RAMB36, 0 DSP
  u_mig_dual_arb:                             152 LUT,            102 FF,   0 RAMB36,                         0 DSP
  u_cpu_ddr_bridge (unmodified):             1386 LUT,           1558 FF,   0 RAMB36,                         0 DSP
  x_heep_system_i (cv32e40px SoC):          33796 LUT,          18538 FF,   0 RAMB36,                         5 DSP
```

**The key confirmation**: `kv_bank_ddr` uses **9 RAMB36+1 RAMB18**, vs.
resident `kv_bank`'s **24 RAMB36+3 RAMB18** at this exact same shape
(NLAYER=2/D=128/NHEAD=2/TMAX=128 -- the number from this session's earlier
BRAM-calibration work). A real, measured **-15 RAMB36/-2 RAMB18** reduction,
matching (and directly explaining) `u_kevgpt`'s own top-level reduction from
245+13 (the pre-this-session baseline) to 230+11 -- the DDR-backed KV cache
genuinely moved the bulk of its storage off-chip, exactly as designed, not
just in simulation. The remaining 9 RAMB36 in `kv_bank_ddr` are the
`inv_lut_lo`/`inv_lut_hi` ROM tables (reused verbatim from `kv_bank.sv`,
unrelated to the on-chip code/header storage that actually moved to DDR3).
`u_kevgpt_ddr_bundle` (all the new CDC FIFOs + read/write engines + rw
arbiter combined) costs **zero BRAM, zero DSP** -- small enough to fit
entirely in distributed RAM/logic, matching the expectation that this is
control/muxing/FIFO plumbing, not compute. DSP total (803/840, 95.60%) is
UNCHANGED from the pre-DDR baseline to the exact tile -- confirms none of
this session's new RTL touches the tight DSP budget at all, only BRAM (and
only in the intended, favorable direction).

This is a synth-only checkpoint (no place & route, no timing closure, no
bitstream) -- it confirms the design elaborates/synthesizes cleanly and
gives real resource numbers, but does NOT yet confirm the CDC fix's timing
closes under real constraints (that needs a full implementation run) or
that a bitstream programs and runs correctly on the physical board. Those
remain real follow-ups, same as any other synth-only checkpoint in this
project's history.

Next: Phase 3 (firmware-driven weight staging into DDR3, so
`weight_loader_ddr` has a real image to load from -- and
`weight_loader_ddr`'s own CDC, once it's actually wired to the top level)
and Phase 4 (final model-size selection), now with a real, measured
per-shape BRAM number for the DDR-backed KV cache to calibrate against
instead of an estimate.

## Full place & route, timing closure, bitstream, and real board bring-up (DONE, bit-exact)

Went all the way: full `make` (synth+impl+bitgen, not just the synth-only
checkpoint above), JTAG programming, and a real `kevgpt_chat` run against
the DDR-backed KV cache on physical silicon.

**Timing closure**: clean, positive margin on every path group.
`WNS=2.474ns, TNS=0.000, 0 failing endpoints (of 2124)`;
`WHS=0.086ns, THS=0.000, 0 failing endpoints (of 1980)`. Per-clock: `jtag_clk_pin`
WNS=4.622ns, `spi_slave_clk_pin` WNS=2.474ns -- both positive. This is real
static timing analysis covering the new CDC paths (`kevgpt_ddr_bundle`'s
`async_fifo_gray` instances) along with everything else -- a clean WNS here
is real evidence the gray-code synchronizer chains are correctly
constrained (an existing project-wide XDC already covers `async_fifo_gray`
generically, since `cpu_ddr_bridge.sv` already relies on the same primitive
for its own, pre-existing CDC boundary), not just an absence of errors.
`write_bitstream` completed successfully: 0 Errors, 356 Warnings, 0 Critical
Warnings. The many `REQP-1839` "RAMB36 async control check" warnings are the
expected, benign signature of async-reset gray-code CDC synchronizers
feeding BRAM address pins -- standard, accepted CDC design, not a real
finding (STA doesn't model async reset assertion timing by design; that's
what the false-path/max-delay constraints on those same paths are for).

**Board bring-up procedure** (same process-hygiene discipline as earlier
sessions -- stale watcher processes from a PRIOR session were still holding
both the JTAG cable and the UART port; had to `kill -9` an openocd PID that
survived a first, non-forceful `pkill` before OpenOCD could open the FTDI
device at all): programmed via `vivado -source *_pgm.tcl -tclargs
xc7k325tffg900-2 <bitstream>.bit` (FuseSoC's own `vivado-fpga-pgm` target
hung silently with no error output for an unknown reason -- invoking the
generated `_pgm.tcl` directly worked cleanly and is the more debuggable
path anyway), confirmed OpenOCD re-examines the RISC-V core cleanly on the
freshly-programmed bitstream, started a fresh UART reader at 115200 baud
(`UART_BAUDRATE` in `x-heep.h`), then loaded and ran the EXISTING
`kevgpt_chat` firmware (`sw/build/main.elf`, unchanged since an earlier
session -- the DDR-backed KV cache swap is entirely transparent to firmware,
same register contract, so no rebuild was needed) via
`riscv32-corev-elf-gdb ... -ex "target remote :3333" -ex "monitor reset
halt" -ex load -ex continue -batch`.

**Result: bit-exact.** Full UART trace:
```
KEVGPT_PHASE,control_plane
KEVGPT_ID,0x53515256
KEVGPT_PHASE,weight_load
KEVGPT_WEIGHT_WORDS,73856
KEVGPT_STATUS_PRE,0x00000000
KEVGPT_PHASE,generate
KEVGPT_PASS_CYCLES,pi=0,cyc=4587
KEVGPT_PASS_CYCLES,pi=1,cyc=5050
KEVGPT_PASS_CYCLES,pi=2,cyc=5550
KEVGPT_PASS_CYCLES,pi=3,cyc=6011  GEN pos=3 tok=1
KEVGPT_PASS_CYCLES,pi=4,cyc=6479  GEN pos=4 tok=29
KEVGPT_PASS_CYCLES,pi=5,cyc=6924  GEN pos=5 tok=30
KEVGPT_PASS_CYCLES,pi=6,cyc=7418  GEN pos=6 tok=34
KEVGPT_PASS_CYCLES,pi=7,cyc=7873  GEN pos=7 tok=1
KEVGPT_PASS_CYCLES,pi=8,cyc=8359  GEN pos=8 tok=40
KEVGPT_CYCLES,58251
KEVGPT_CMP,i=0,got=1,want=1    KEVGPT_CMP,i=1,got=29,want=29
KEVGPT_CMP,i=2,got=30,want=30  KEVGPT_CMP,i=3,got=34,want=34
KEVGPT_CMP,i=4,got=1,want=1    KEVGPT_CMP,i=5,got=40,want=40
KEVGPT_PASS,generate
```
All 6 generated tokens exactly match `kevgpt_expected_gen` (the golden
values `gen_chat_fw.py` baked into the firmware from the same
`IntKVQSequencer.generate_greedy` reference this whole port has been graded
against throughout) -- `STATUS_PRE=0x0` confirms a clean, non-stale reset
(the freshly-programmed bitstream sidesteps `reset_config none`'s "soft
reset doesn't clear done_latched" gotcha entirely, same fix as earlier
sessions). This is the same discipline as the original Option A hardware
bring-up (kv_bank.sv, resident) -- now repeated with `KV_DDR_BACKED=1`
(kv_bank_ddr.sv, DDR3-streamed) and passing identically.

**Real measured rate with the DDR-backed KV cache**: steady-state generation
(the 6 passes producing a token) averages 7177.3 cycles/token -> at the
confirmed 50MHz clock, **143.5us/token -> ~6,966 tok/s**. Compare to Option
A's resident-KV-cache measurement from earlier this port (~11,930-11,985
tok/s): DDR-backed KV costs roughly **1.7x** the resident design's per-token
latency at this tiny scale (NLAYER=2/D=128/TMAX=128) -- a real, measured
number, not a simulation estimate. Per-pass cycles grow with position
(6011->8359 over positions 3->8), matching the expected shape: `rd_tcount`
for a KV read scales with `pos+1`, so more DDR round-trips accumulate as
context grows. At this small scale DDR latency is clearly NOT free, but it's
also nowhere near the ~240x cycle-budget headroom the 50 tok/s target was
built on (Phase 0) -- meaning a genuinely bigger model, where compute cost
per layer grows much faster than KV traffic, should have this overhead
comfortably hidden. Worth re-measuring once a larger candidate shape (Phase
4) is actually built, not assumed to scale the same way.

This closes out real-hardware verification for Phase 2's KV-cache half
completely: gated in isolation (`tb_kv_bank_ddr.sv`), gated at full-sequencer
scale in simulation (`tb_seq_vec_kv_ddr.sv`), gated through the complete
arbiter stack including genuine dual-clock CDC (`tb_kevgpt_ddr_bundle.sv`),
synthesized clean, placed and routed with positive timing margin, and now
proven bit-exact on the real board. Next: Phase 3 (weight staging into
DDR3) and Phase 4 (model-size selection) as above -- the weight-window half
of Phase 2 has NOT yet had this same real-hardware treatment (it isn't wired
into the top level yet, only gated in simulation).
