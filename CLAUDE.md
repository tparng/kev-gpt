# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

"Kevin on Kria" — running a tiny telegraphic INT4 language model entirely inside the FPGA
fabric of a Xilinx Kria KV260, with weights baked into on-chip BRAM/URAM so they never touch
DDR. It is **both** a design-argument (numbered `.md` docs `0`–`6`) **and** a real implementation
(Python + SystemVerilog/Verilog). When a task says "work on the code," it usually means one of:
editing the RTL/Python under `fabric/` or `model/`, keeping the cross-document argument
consistent, or running the bit-honest gate ladder. Do not invent commands/files that the docs
only describe as *future* work — but most of the pipeline now exists and runs.

## Repository layout

Three code pillars feed the hardware, plus the design docs:

- **`keviniser/`** — the data tool. A spaCy part-of-speech preprocessor that strips function
  words / flattens inflection to compress English into telegraphic "Kevin-speak." `keviniser.py`
  is the engine, `harness.py` the CLI + stats, `fetch_tinystories.py` the corpus fetcher.
- **`model/`** — a 2–4M-param nanoGPT-lineage transformer ("goformer"): `train.py` / `sample.py`
  / `data.py` (FP proof-of-life), `qgpt.py` (Brevitas INT4 QAT), and the **goformer reference
  ladder** (see below) that is the bit-true spec the RTL must reproduce. `export_fabric.py`
  emits `.mem` weight files into `fabric/export/`; `validate_goformer.py` is the export gate.
- **`fabric/`** — the hardware build, in stages 0→4 (see "Build order"). Stage 3 (`fabric/stage3/`)
  is where current work lives: RTL cores (`rtl/*.sv,*.v`), Python **gate harnesses** (`run_*.py`),
  the per-phase reference (`seq_ref.py`), Vivado scripts (`tcl/`), testbenches (`tb/`), and
  on-board drivers (`board/pl_*.py`).
- **Design docs** `0-master.md`…`7-kevin-remembers.md` (read in numeric order; `0-master.md`
  is the through-line). `5-demo-prd.md` is the live-demo PRD; `6-past-the-stream-ceiling.md` is the
  speed campaign past the N=16 stream ceiling (split-brain + worst-path retirement toward 100k);
  `7-kevin-remembers.md` is the faithful-stream campaign (N=1, T=256 full trained context, 20k tok/s average — real
  messages instead of T=1 degenerate text).
  Plus `README.md`, `ROADMAP-10K.md`, `BUILD-LOG.md`, `DEPLOYMENT.md`, `GLOSSARY.md`.

## The core thesis (keep all edits consistent with it)

Two load-bearing facts. Nothing may contradict them:

1. **The bandwidth wall.** Single-stream autoregressive decode is memory-bandwidth bound, not
   compute bound. On the KV260 the A53 cores and the PL fabric share one ~20 GB/s DDR controller,
   so a DDR-resident model gets *no* uplift from the fabric. The only source of speed is keeping
   the whole model (≈1–2 MB INT4, a few M params) on-chip in BRAM/URAM (hundreds of GB/s–TB/s).
   The ~3 MB on-chip budget is a hard ceiling.
2. **The fusion (the joke is the thesis).** Telegraphic "few word do trick" output = fewer tokens
   = less weight streamed + smaller KV cache = more fits on-chip = faster. The model's dumbness
   and its speed are the same property. The comedy and the optimisation are one thing, never a
   bolt-on gag. Corollary: the Keviniser strips the **training corpus, not the inference input** —
   the model learns the compressed distribution and generates telegraphic text itself.

## The goformer reference ladder + the bit-honest gate workflow

This is the architectural spine and the single most important thing to understand. The hardware
is developed against a chain of Python references, each a refinement of the last, and **every**
block is proven bit-exact (or cosine > 0.9999 for transcendentals) against its reference *before*
any speed number is trusted ("bit-honest before fast"). The references, in order:

- `model/goformer_full.py` — full integer forward.
- `model/goformer_kv.py` — incremental KV-cached decode (process only the new position). The
  prerequisite for 10k+; bit-identical to full-recompute by causality.
- `model/goformer_q.py` — pins the fixed-point Q-format spec the RTL uses.
- `model/goformer_fixed.py` — fabric-precision non-linears (softmax/LayerNorm/GELU/dequant the
  way the LUTs/shifts actually compute), the bit-true reference the RTL copies.
- `model/goformer_seq.py` — integrated sequencer reference (KV decode + fabric non-linears).
- `fabric/stage3/seq_ref.py` — the **per-phase** sequencer reference. `block0_phase_signals()` /
  `full_forward_signals()` expose every intermediate (ln1/qkv/ctx/attn/x_res/ln2/gelu/mlp/x_out,
  then x4/lnf/head/tok) so an RTL rewrite can be gated phase-by-phase, not just on the final token.

**The gate-harness pattern** (`fabric/stage3/run_*.py`): each writes the needed `.mem` files,
compiles the RTL with `iverilog -g2012`, runs `vvp`, dumps per-phase `.out` files, and compares
to the Python reference, printing a one-line `*_VERDICT` / `SEQ_VEC_*` result. This is THE inner
loop for RTL work — make it green in sim before spending 30–60 min on a synth. Examples:
`run_banked.py` (GEMV), `run_layernorm.py`, `run_softmax.py`, `run_gelu.py`, the `run_vec_*.py`
P-wide bricks, and `run_vec_seq.py` (full single-token forward).

## Common commands

```bash
# --- Keviniser (data tool) ---
python -m keviniser.harness samples/canonical.txt -o out.kevin.txt --stats stats.json
cat corpus.txt | python -m keviniser.harness > corpus.kevin.txt   # stdin/stdout filter
#   needs: pip install -r requirements.txt ; python -m spacy download en_core_web_sm

# --- Model (train / sample / validate) ---
python -m model.train --smoke 50                 # measure throughput + time estimate, then stop
python -m model.train --max-iters 4000 --eval-interval 500    # defaults: 3.16M params, L4 d256 ctx256
python -m model.sample data/ckpt.pt --prompt "once upon time" -n 300
python -m model.evolution data/states.jsonl -o data/evolution.md
python -m model.validate_goformer                # bit-honest export gate (cosine vs trained model)

# --- Tests ---
pytest                                           # test_keviniser.py + test_fabric_pack.py
pytest tests/test_keviniser.py -k some_name      # single test

# --- RTL gate harnesses (sim, iverilog) — the inner loop ---
python -m fabric.stage3.run_vec_seq --tok 48 --p 8 --lanes 128 --dir C:/kevbuild/stage3_seq_vec_ww128
python -m fabric.stage3.run_banked               # GEMV bit-exact across PE widths
python -m fabric.stage3.run_layernorm            # (also run_softmax / run_gelu) non-linear gates

# --- Vivado (OOC fit/Fmax, then full BD+impl bitstream) — run FROM the .mem dir ---
vivado -mode batch -source fabric/stage3/tcl/ooc_seq_vec.tcl      -tclargs 8 128 8.0 25600
vivado -mode batch -source fabric/stage3/tcl/build_bd_seq_vec.tcl -tclargs 8 128 25600 100
vivado -mode batch -source fabric/stage3/tcl/impl_seq_vec.tcl     # synth+impl+bitstream

# --- On-board (Kria over SSH; bit-honest measured tok/s) ---
# scp the .bit.bin to ~/kevbit/, fpgautil -b to load, then a board driver forces the PL clock:
python -m fabric.stage3.board.pl_seq_vec --lanes 128 --tok 48 --fclk 40e6
```

## Hardware / build environment + hard-won gotchas

- **Toolchain:** Vivado **2025.2**, target part `xck26-sfvc784-2LV-c` (KV260). There is **no
  `vitis_hls`** on this box → everything is hand-written RTL, not HLS. No local C compiler for
  the A53 baseline either — verify via file-based compare in sim.
- **BUILD OUTSIDE OneDrive.** Vivado build dirs go in `C:/kevbuild/...`, never inside the repo
  (which is under OneDrive) — the `cldflt` cloud-sync filter locks files mid-build and corrupts
  runs. The `.mem` sim dirs also live under `C:/kevbuild/`.
- **Forward slashes for `--dir` paths.** When invoking `run_*.py` through the Bash tool, pass
  `--dir C:/kevbuild/...` with forward slashes — bash eats backslashes and you get a junk
  relative dir in the repo instead of the intended absolute path.
- **iverilog-2012 traps** (the rewrites repeatedly hit these): can't `$readmemh` a 2D-array row
  (use a flat/packed ROM + ranged `$readmemh(file,mem,start,end)`); a **variable** `+:`
  part-select on an *element of an unpacked array* reads X — only ever variable-part-select a
  **plain reg**; over-wide part-selects read X.
- **Wide-word banking, not `[P][rows]`.** A banked `reg [W] buf [0:P-1][0:ROWS-1]` accessed at a
  variable row synthesises to giant per-lane row-muxes (a MUXF7/LUT blow-up that overflows the
  KV260). The correct layout is one row-addressed wide word per buffer: `reg [P*W-1:0] buf
  [0:ROWS-1]` (lane `l` in bits `[l*W +: W]`) so the variable row is a memory *address*, not a
  mux. Inherently-1/cycle producers stage P elements into a plain-reg word and write one wide
  word per P.
- **Silicon overclock vs STA.** OOC/impl timing on `-2LV` is pessimistic — a design that closes
  ~70–85 MHz in STA has run bit-exact at 125 MHz on silicon (~1.76×). So builds target a clock
  that closes, and the real ceiling is found by the board `--fclk` sweep. A flat `fpgautil -b`
  load does **not** apply the BD's `PL0_REF FREQMHZ`; the driver must force `fclk0` via
  `/sys/devices/platform/fclk0/set_rate` (and verify it) or the PL runs at the wrong clock.

## Document reading order

`0-master.md` is the entry point (thesis, four-piece map, staged build, the honesty section).
`1-keviniser.md` = the data tool (canonical design rationale). `2-llm-on-kria.md` = the platform
(HLS/RTL systolic GEMV, on-chip weights, fabric-native softmax/RMSNorm). `3-kevin-on-kria.md` =
the fusion (train doc-2's model on a doc-1 corpus). `4-live-chatbot.md` = the public stress test
(web front end behind a Cloudflare Tunnel). `webchat/demo/server.py` serves the live chat
(`webchat/demo/client.html`) at chat.mikeayles.com — the single canonical front end.
`5-demo-prd.md` = the speculative-typing chat + load-dashboard PRD. `6-past-the-stream-ceiling.md`
= the speed campaign (split-brain on the dual-ported URAM, the cycle/clock lever campaign, the
KV-DDR context-restore path, the 100k identity) — read it before touching `fabric/stage3` to know
which levers are live vs proven-dead. `7-kevin-remembers.md` = the faithful-stream campaign
(N=1, T=256 on-chip KV, 20k tok/s average — the rung ladder R0–R5, the INT8-KV reference change,
and the dual-port GEMV / wide-attention / wide-NL levers).

## The Keviniser implementation (doc 1 → `keviniser/`)

Uses spaCy with `en_core_web_sm`. Design decisions to preserve if you touch it:
- Strips on **part-of-speech tags, not a flat stopword list** — so it drops auxiliary `do`
  (`AUX`) but keeps main-verb `do` (`VERB`). This is the whole reason for spaCy.
- Three knobs: `lemmatise` (flatten inflection), `objectify` (subject→object pronouns, full
  Kevin), `keep_punct`. Defaults `lemmatise=True, objectify=True, keep_punct=False`.
- `KEEP_LEMMAS` rescues quantifiers/negation (`few`, `lot`, `not`…) the tagger would strip;
  `DROP_LEMMAS` removes filler (`that`) even when tagged as content.
- Prints the **compression ratio to stderr** — that number is the headline metric feeding the
  speed story in docs 3/4. Don't silence it. TinyStories is the lead corpus.

## Writing conventions (project rules, not style preferences)

- **Mischief in the title, dry in the body.** Playful headlines; the body is measurement,
  tradeoffs, the roofline crossover. Never pretend the prose output is good — the deliberate
  dumbness is the point; the speed numbers carry the post.
- **Honest-first.** Every doc states where the approach *loses* (DDR-resident models, long
  context once KV spills, high dev effort). The "crossover" (the model size where the fabric
  stops winning) is a *result*, not a caveat. Don't drift toward overclaiming.
- **Bit-honest before fast.** Fabric output is validated against the goformer golden reference
  to cosine > 0.9999 (or bit-exact) before any speed number is trusted.
- Numbers are honest ranges ("55–70% of original tokens," "tens of ×, not 100×") and tagged
  **MEASURED** (on silicon / by a committed tool) / **DERIVED** (arithmetic) / **PROJECTED**
  (needs a synth or board run). Preserve the hedging; don't sharpen into false precision.

## The build order (stages 0–4, each ships something demonstrable)

(0) A53 baseline proving it's bandwidth bound · (1) raw on-chip matmul throughput · (2) the
measured heterogeneous ping-pong tax · (3) full zero-DRAM uplift + crossover plot · (4) live
chatbot under load with two ceilings (inference vs. serving). The data track runs alongside:
Keviniser on TinyStories → train (Brevitas INT4 QAT) → validate vs goformer → wire into stage 3.

**Current work (the post-stage-3 speed era, doc 6):** the model has long been running fully in
fabric; the optimisation is now *cycles and clock*, not getting it on-chip. The MEASURED record is
**59,965.5 tok/s @ 200 MHz** (split-brain N=16 TMAX=16 wave, 53,364 cyc, 16/16 bit-exact, 3/3 —
log §27). The live design is
`sequencer_sb` (two N=8 cohorts on the true-dual-port URAM, parameterized: `ATT2` = per-cohort vs
shared attention, `TMAX` = on-chip KV window, `DBG` = board-debug readback on/off; bitstream builds
set `DBG=0`/`ATT2=0` for fit). The target is the **100k identity: 16 streams × 250 MHz / 40k cyc** —
streams are maxed (N=16, `dsp3_pack_proof.py`), cycles are ~53k gated heading to ~40k, and 250 MHz
needs a post-route-MET 5ns build so silicon's ~1.3× margin covers it. KV-to-DDR (`kv_dma` +
`kv_prefetch`, sim-complete, bit-exact) is the context-restore path once the on-chip window is too
small. Before proposing a stage-3 cycle/timing lever, check doc 6 / the log §19–26 for whether it's
already proven dead (the LN→AQ and attention→PROJ schedule overlaps are; the shared-attention serial
cost is not fixable by arbitration).
