# Kevin-on-Kria — PL / Fabric Architecture Reference

_Generated 2026-07-20 by a deep-read of the RTL / Python / TCL. A standing reference so the fabric need not be re-scouted from scratch. Numbers tagged MEASURED / DERIVED / PROJECTED; treat file:line citations as of this date and re-verify before relying on them for edits._

---

## 0. Orientation

## Orientation: the Kevin-on-Kria PL fabric

**What it is.** A tiny INT4 char-level GPT ("goformer": 4 layers, d=256, 4 heads, head_dim=64, MLP hidden 1024, vocab 193) running *entirely* inside the KV260's programmable-logic fabric. The whole ~12.6 Mbit INT4 weight image is resident in on-chip UltraRAM/BRAM and never touches DDR. This is the load-bearing thesis: single-stream autoregressive decode is **memory-bandwidth bound**, and the A53 PS + PL fabric share one ~20 GB/s DDR controller, so a DDR-resident model gets zero uplift from the fabric. The only source of speed is keeping the model on-chip (hundreds of GB/s–TB/s of BRAM/URAM bandwidth). The ~3 MB on-chip budget is the hard ceiling, and the fusion joke — telegraphic "Kevin-speak" output means fewer tokens, less weight streamed, more fits on-chip — is the optimisation, not a bolt-on.

**There is no HLS.** The box has no `vitis_hls` and no local C compiler, so everything is hand-written SystemVerilog/Verilog, validated in `iverilog` sim against Python "goformer" references before any silicon number is trusted ("bit-honest before fast").

### Tracing ONE decoded token through the fabric (the live split-brain record engine, `sequencer_sb`)

The A53 driver (`board/pl_seq_sb.py`) has already streamed the INT4 weight image into resident URAM once at boot and forced the PL clock. To decode, it writes N token ids + positions, streams the token/position embeddings through the `EDATA` register, and pulses `GO`. From there the fabric runs with zero host I/O until `DONE`:

1. **Embed + control.** `sequencer_sb` (the split-brain top) splits the N=16 streams into two independent cohorts of NC=8 (cohort 0 = streams 0–7 on URAM port B, cohort 1 = streams 8–15 on URAM port A). Each cohort has its own `cohort_engine`, which wraps one `nl_engine` (the phase FSM) + one `gemm_cohort_vec` datapath driven by a local GE (GEMM-executor) FSM. There is **no** monolithic phase-decode FSM at the top; the transformer phase sequence lives in `nl_engine`'s 23-state `nl` machine.
2. **Per layer × 4 blocks**, the `nl_engine` walks the phases and issues five GEMM descriptors to its GE FSM over a small req/descriptor/served handshake. For each phase the residual stream (signed **Q6.25**, 32-bit) is:
   - **LayerNorm 1** (`layernorm_vec`, gamma-only, exact-integer variance, rsqrt via seed-LUT + 2 Newton steps) → Q.22, then **INT8 activation-quant** (`inv_sact = 2^40/s_act`).
   - **QKV GEMM** (768×256): the INT8 activations are multiplied against wide URAM weight words (LANES=128 INT4 nibbles/cycle) in the banked-resident datapath → INT32 accumulators → **dequant** (`vec_dequant`, per-channel Q.16).
   - **Attention** (`vec_attn`, P-wide single-head, shared between the two cohorts behind a hold-until-done arbiter because `ATT2=0` in fit builds): score = q·k>>27 (Q8.8) → **softmax** (`softmax.sv`, scalar, exp-LUT + restoring-division reciprocal 2^40/sum, Q1.20 probs) → ctx = prob·v>>11. In the record build the on-chip KV window is **TMAX=16** with fresh full-precision Q.16 K/V re-streamed each call (this is *not* the persistent INT8 KV cache — that lives in the other engine). **Critically, even with the TMAX=16 window the record engine HARDWIRES the attention to T=1** (`at_tcount_o <= 9'd1`, `nl_engine.sv:459`): it attends a single, degenerate position, so only the first decoded token per stream is model-faithful — this is the most conflation-prone fact in the project (record = aggregate throughput on textually-degenerate output, not the faithful chat).
   - **PROJ GEMM** (256×256) → dequant → residual add 1.
   - **LayerNorm 2 → FC GEMM** (1024×256) → **GELU** (`vec_gelu`, 8192-entry Q4.12 LUT + interp) → **MP GEMM** (256×1024) → dequant → residual add 2.
3. **After 4 blocks:** final LayerNorm, **HEAD GEMM** (193×256) produces the vocab logits, and each cohort's `nl_engine` does a plain P-wide greedy **argmax** to pick the token id. (There is *no* on-chip sampling in `sequencer_sb`.)
4. **Readback.** The driver polls `DONE`, reads the 16 argmax token ids + the fabric cycle counter, computes tok/s = N·fclk/cyc, and reports it **only if all 16 tokens match the `seq_ref` golden reference** (the bit-honest gate).

The stream-granular overlap (per-stream `s_done`/`rb_rdy`) lets each stream's post-loops (attn/residual/argmax) start while the GE FSM still drains later streams — this is why the ~53k-cycle serial queue runs as two parallel cohort halves.

### Map of the 8 subsystems

1. **Sequencers & top-level control (the spine).** `sequencer_sb` → 2× `cohort_engine` → (`nl_engine` phase FSM + GE FSM). Engine lineage: sequencer → _fast → _gemm → _pp → _vec → split-brain `_sb`. Everything older than `sequencer_sb`/`cohort_engine`/`nl_engine` is historical.
2. **GEMM/GEMV datapath, DSP packing & weight residency.** `gemm_banked_resident_vec` lineage; wide-word banking, 72-bit URAM banks, boot-streamed transposed INT4 weights, DSP packing proven exactly 2.0 MAC/DSP.
3. **Attention & the KV window (TMAX).** Two *distinct* tracks: Track A (`vec_attn`, record engine, transient Q.16 K/V) vs Track B (`kv_bank` + `vec_attn_w`, the faithful persistent INT8 K8/V8 cache). Plus the sim-only `kv_dma`/`kv_prefetch` DDR spill path.
4. **Non-linears & numerics.** LayerNorm, softmax, GELU, dequant, activation-quant, and (in the faithful engine only) on-chip Gumbel-max sampling — each pins its own Q-format.
5. **Board drivers, /dev/mem register map & A53 runtime.** `board/pl_*.py` mmap `/dev/mem` at `0xA0000000`; three distinct register maps; the `a53_daemon` fronting the public chat.
6. **Build flow, bitstreams, timing & utilization.** Three-TCL flow (ooc → build_bd → impl), silicon-overclock-vs-STA, PLL clock quantisation.
7. **Reference ladder, gate harnesses & performance model.** The goformer Python chain terminating in `seq_ref.IntSequencer`; `run_*.py` gate harnesses; `cycle_model.py`/`batched_model.py`.
8. **Speed-campaign narrative.** The two campaigns/two ceilings, the 100k identity, dead levers.

**The two-campaign split is the key mental model:** doc 6 = N=16 aggregate throughput (record 59,965.5 tok/s, but every N≥4 design hardwires attention T=1 → degenerate text); doc 7 = N=1 faithful stream (real messages with full context, ~16k tok/s deployed). Both build from the same tree; the daemon just loads a different bitstream. **The public chat runs the doc-7 faithful engine, not the record engine.**

## Parameter reference

All values below are the **live/record** settings reconciled across subsystems. Where the same name means different things in different engines, both are listed.

| Parameter | Meaning | Live value | Range | Where |
|---|---|---|---|---|
| `N` | Total streams per pass (split-brain = 2·NC) | 16 (record) / 1 (faithful chat) | even ≥2 for SB; 1 for faithful | `sequencer_sb.sv:25` |
| `NC` | Streams per cohort (arithmetic cohort-select, need not be pow2) | 8 | ≥1; fit variant 7 (N=14) | `sequencer_sb.sv:26` |
| `P` | Vector/serial-lane width (scratch elements/cycle; LN/GELU/dequant/argmax lanes) | 8 (measured sweet spot) | pow2 divisor of D: 2/4/8/16 | `sequencer_sb.sv:23`, `gemm_banked_resident_vec.sv:26` |
| `LANES` | GEMM PE width = INT4 nibbles per wide URAM word; WBITS=LANES·4 | 128 (SB record) / 256 (faithful gum bitstream) | 16/64/128/256 | `sequencer_sb.sv:24`, `gemv_axi_seq_vec.v:20` |
| `TMAX` (Track A, `vec_attn`) | On-chip KV window of the record-path attention; effective ceiling 32 (probmem[ji[4:0]] wrap) | 16 (record build; RTL default 32) | 1..32 usable | `sequencer_sb.sv:29`, `vec_attn.sv:38` |
| `TMAX` (Track B, `kv_bank`/`vec_attn_w`) | Max faithful persistent KV window; sizes kv_bank banks | 128 (shipping/deployed) | 1..256 designed; 256 busts BRAM with K8 | `vec_attn_w.sv:32`, `kv_bank.sv:31` |
| `ATT2` | Attention topology: 1 = `vec_attn` per cohort; 0 = one shared unit + arbiter | 0 in SB bitstreams (fit); RTL/sim default 1 | {0,1}; SB-only generic | `sequencer_sb.sv:43` |
| `DBG` | Board-debug per-phase readback mux | 0 in SB bitstreams (fit); default/sim 1 | {0,1}; SB-only generic (KV engine readback is hard-wired) | `sequencer_sb.sv:42`, `gemv_axi_seq_sb.v:25` |
| `DP` | DOUBLE-PUMP: MAC at 2 K-steps/clk via clk2x MMCM | 0 (all measured records; the _2/_b paths idle) | {0,1}; SB-only; OOC-validated not silicon | `sequencer_sb.sv:44`, `gemm_banked_resident_vec.sv:39` |
| `ND` | Of N/NC streams, how many use the DSP-packed leaf | 6/cohort in the SB record (DERIVED from BD default + committed gate config + ~95% DSP occupancy); RTL default is 0 — ND=0 as a *built* value was a misread of the RTL default | 0..NC | `sequencer_sb.sv:27` (RTL default 0), `build_bd_seq_sb.tcl:24` (BD default 6), `gemm_banked_resident_vec.sv:25` |
| `WWORDS` | Resident weight URAM depth in wide words (~12.6 Mbit fixed, scales inversely with LANES) | 25600 (L=128) / 16384 (L=256 gum) | — | `sequencer_sb.sv:33`, `build_bd_seq_kv.tcl:24` |
| `MMAX / KMAX` | Max output rows / max reduction length of any layer | 1024 / 1024 | — | `gemm_banked_resident_vec.sv:27-28` |
| `ABITS` | Per-lane accumulator width; \|acc\|≤8·128·1024=2^20, 3 bits margin | 24 | — | `gemm_banked_resident_vec.sv:43` |
| `RLAT` | Read→MAC pipeline depth (stage 0 = registered URAM read) | 2 | ≥2 | `gemm_banked_resident_vec.sv:30` |
| `BANKW` | URAM-native bank width (72 if WBITS>512 else WBITS) | 512 (L=128, single bank) / 72 (L=256) | 72 or WBITS | `gemm_banked_resident_vec.sv:79` |
| `GAP2` | DSP-pack gap: two INT4 weights 22 bits apart in 27-bit operand | 22 | 22 (24 cascades 2 DSPs) | `dsp3_pack_proof.py:96` |
| `D / D3 / D_MLP` | Model width / qkv width (=3D) / MLP hidden | 256 / 768 / 1024 | model-fixed | `sequencer_sb.sv:20-22` |
| `VOCAB / NLAYER / NHEAD / HEAD_DIM` | Kevin vocab / blocks / heads / head dim | 193 / 4 / 4 / 64 | model-fixed | `sequencer_sb.sv:28,34,35,36` |
| `RESID_FRAC` | Residual stream fraction (Q6.25); also dequant target for proj/mp/head | 25 | constant | `seq_ref.py:56`, `nl_engine.sv:37` |
| `LN_OUT_FRAC / OUT_FRAC` | LayerNorm output fraction (Q.22) | 22 | constant | `layernorm.sv:40`, `run_layernorm.py:35` |
| `OUT_SH` | LN final shift = QX+Y_FRAC+G_FRAC−OUT_FRAC | 49 | derived | `layernorm_vec.sv:52` |
| `G_FRAC` | LN gamma fraction (Q4.20) | 20 | constant | `run_layernorm.py:32` |
| `EPS_A` | LayerNorm epsilon in Q.26 = round(1e-5·2^26) | 671 | constant | `layernorm_vec.sv:53` |
| `SEED_IDX_BITS` | rsqrt seed-LUT address bits (64-entry mantissa table) | 6 | constant | `layernorm.sv:42` |
| `VFRAC` | q/k/v stored fraction (Q.16); also dequant frac for qkv/kv | 16 | constant | `seq_ref.py:62`, `vec_attn.sv:56` |
| `ISQRT` | 1/sqrt(head_dim=64) as right-shift by 3 | 3 | constant | `seq_ref.py:63` |
| `SCORE_FRAC` | Attention score fraction (Q8.8) into softmax | 8 | constant | `softmax.sv:42` |
| `SCORE_SH / CTX_SH` | score = q·k>>27; ctx = prob·v>>11 | 27 / 11 | constant | `vec_attn.sv:61-62` |
| `PROB_FRAC / EXP_FRAC` | Softmax prob + exp-LUT fraction (Q1.20) | 20 | constant | `softmax.sv:46`, `run_softmax.py:31` |
| `RECIP_R / ISH / INV_SACT_SH` | Reciprocal dividend exponent 2^40; act-quant inv_sact = round(2^40/s_act) | 40 | constant | `softmax.sv:12`, `nl_engine.sv:41`, `seq_ref.py:158` |
| `DIV_STEPS` | Restoring-division steps/cycle (radix-2^n); bit-identical quotient any value | 2 (radix-4, 21 cyc) | 1..3 | `softmax.sv:36`, `softmax_f.sv:18` |
| `ZMAX / NZ` | exp-LUT input clip [−16,0] Q8.8 → 4097 entries | 16 / 4096 | constant | `run_softmax.py:32-35` |
| `GELU_FRAC / FRAC` | GELU I/O fraction (Q4.12) | 12 | constant | `gelu_lut.sv:3`, `run_gelu.py:21` |
| `N_LUT` (GELU) | GELU LUT entries + 3-bit interp | 8192 | constant | `run_gelu.py:23` |
| `SCALE_MANT_BITS` | Dequant per-channel folded-scale mantissa width | 24 | constant | `seq_ref.py:58` |
| `KBITS` (kv_bank, Track B) | Bits per K/V code in the on-chip persistent cache (INT8) | 8 (K8/V8, +0.09% NLL) | 8 | `kv_bank.sv:32` |
| `KBITS` (kv_dma DDR path) | Bits per code in DDR spill layout | 4 default (doc budget uses 8 worst-case) | 2/4/8 | `kv_dma.sv:53` |
| `INV_SH / QMAX` (kv_bank) | Divfree-quant shift (2^24) / code clamp | 24 / 255 | constant | `kv_bank.sv:33,69` |
| `HROWS` (kv_bank) | Code/hdr bank depth = NLAYER·2·NHEAD·TMAX = 32·TMAX | 4096 at T=128; 8192 at T=256 | — | `kv_bank.sv:71` |
| `TEMP` (Gumbel LUT) | Baked on-chip sampling temperature. **UNRESOLVED whether the live path is on-chip (baked) or host-side** — see the "live sampling" note | 0.85 (baked); memory note claims host-side temp 0.4, but committed code with those flags samples on-chip | constant | `gumbel.py:23` |
| `GLBITS` | Gumbel noise-LUT address bits (1024 entries = 1 BRAM) | 10 | constant | `gumbel.py:24` |
| `BASE` | AXI-Lite slave base mmap'd from /dev/mem (one 4KiB page) | 0xA0000000 | fixed | `pl_seq_chat.py:61` |
| `FCLK_SET` | sysfs node forced to set PL clock; readback = quantised actual rate | /sys/devices/platform/fclk0/set_rate | path | `pl_seq_chat.py:71` |
| `tol_hz` | fclk verify fatal tolerance | 6e6 | Hz | `pl_seq_chat.py:75` |
| `poll_timeout` | seconds to busy-poll DONE before TIMEOUT | 30.0 | s | `pl_seq_vec.py:104` |
| `gen_chars` | completion chars/request (construction-time only, no per-request override) | 104 (live kv256 daemon) / 48 default kv256 / 6 otherwise | int | `pl_kv256.py:109`, live daemon line |
| `temp` (deployed sampling) | Sampling temp. **UNRESOLVED path**: memory note says host-side 0.4; committed code with `--temp 0.4` samples ON-CHIP at the baked 0.85 and ignores the runtime value | 0.4 requested (memory note); on-chip baked 0.85 per committed code | float | live daemon `--temp 0.4` |
| `top_k` (deployed sampling) | Top-k truncation — **applies only on the host_sample path; the committed on-chip path IGNORES top-k** (UNRESOLVED which is live) | 10 requested (memory note) | int | live daemon `--top-k 10` |
| `capacity` | daemon batch width advertised to server | 16 | int | `a53_daemon.py:76` |

Perf-model constants: `RLAT=2`, `N_LAYER=4 N_HEAD=4 D=256 D_MLP=1024 VOCAB=193`, `ONCHIP_LEFTOVER_KB=1216` (KV budget after ~1.5 MB weights) — `cycle_model.py:23-24`, `batched_model.py:25`.

## Bitstream / engine map

There are **two live engine families**, each with its own AXI-Lite shell (distinct IDCODE), build/impl TCL, and board driver. Everything else is historical (kept for the log's cycle-march story, not built).

| Engine / bitstream | Params (TMAX / N / LANES / P / fclk / DBG / ATT2) | Status | Source of truth |
|---|---|---|---|
| **`sequencer_sb`** (split-brain N=16 aggregate; shell `gemv_axi_seq_sb.v`, IDCODE **SQSB** 0x53515342) — the **MEASURED record** | TMAX=16, N=16 (NC=8), LANES=128, P=8, built @6.0 ns (impl WNS −0.702), runs 200 MHz silicon; DBG=0, ATT2=0 (shared attn, fits). ND=6/cohort in the SB record (DERIVED from BD default + committed gate config + ~95% DSP occupancy); DP=0. | **Live (record engine)** — 59,965.5 tok/s @200 MHz, 53,364 cyc/16 tok, 16/16 bit-exact, 3/3 | `WIDE-WORD-DATAPATH-LOG.md:983-992`, `build_bd_seq_sb.tcl:78-79`, `board/pl_seq_sb.py:1-9,152-157` |
| **`gemv_seqkv_gum.bit.bin`** (doc-7 KV-faithful single stream; `sequencer_vec` / shell `gemv_axi_seq_vec.v`, IDCODE **SQRV** 0x53515256, + on-chip Gumbel-max LUT) — **the engine the public chat actually runs** | **P=8, LANES=256, TMAX=128, WWORDS=16384, NLAYER=4**; build target 125 MHz (8 ns), **runs 166.7 MHz on silicon**. DBG/ATT2 **do not apply** (SQRV has no such generics; qkv/ctx readback is hard-wired). Gumbel-LUT temp baked 0.85. | **Live (deployed public chat)** — greedy 9,292 cyc/tok = ~17,936.6 tok/s @166.7 (fabric); build WNS +0.013 MET, LUT 93.4% / BRAM 94.4% / URAM 100% / DSP 97.4% | `WIDE-WORD-DATAPATH-LOG.md:1426-1433`, `build_bd_seq_kv.tcl:33` (mems `stage3_vec_kvk8t128_smp`), live daemon line |
| `sequencer_sb` DP=1 (double-pump 100k campaign) | clk/clk2x 200/400 MHz OOC MET (WNS +0.070/+0.074) | **DEAD.** Full DP=1 build: OOC-timing-MET only, never board-run. A MAC-only DP branch (`dp-hw-maconly`) WAS run on silicon bit-exact but the fabric→clk2x data feed walled at 50 MHz — that measurement is what killed the lever. | `100K-REVIEW.md:44`, `DOUBLE-PUMP-100K.md:8,34-35` |
| `sequencer_vec` (base P-wide single-stream, pre-Gumbel) | LANES 128/256, P 8/16, TMAX 64 default | Historical (superseded by the gum build) | `gemv_axi_seq_vec.v` |
| `sequencer_pp` / `_gemm` / `_fast` / base `sequencer` | N=8 ping-pong / N-batch / P=1 KV / P=1 | Historical lineage — not built as bitstreams | `WIDE-WORD-DATAPATH-LOG.md`, engine lineage |
| `kv_dma` + `kv_prefetch` (KV-to-DDR spill/restore) | K4/V4 default DDR layout | Historical / **SIM-ONLY**, bit-exact-gated; instantiated only by testbench, in no sequencer or bitstream | grep of `rtl/*.sv`; CLAUDE.md |

**Confidence note on the live `gemv_seqkv_gum` row.** The build config (P=8, LANES=256, TMAX=128, WWORDS=16384, NLAYER=4, 125 MHz target → 166.7 MHz silicon) is **corroborated by three independent sources** and is HIGH confidence: (1) the daemon invocation captured in project memory (`a53_daemon --engine kv256 --lanes 256 --tmax 128 --fclk 166.7e6`), (2) the build script's hardcoded mems dir `stage3_vec_kvk8t128_smp` ("t128"/"smp" = TMAX 128 + sampling), and (3) shipping-config log entries §37/§39/§45. **One item UNRESOLVED:** the out-of-repo memory note says host-side softmax @temp 0.4/top-k 10, but the committed code with those flags (`--temp 0.4 --top-k 10`, no `--host-sample`) samples ON-CHIP at a baked temperature and ignores top-k (`a53_daemon.py:289-304` does not pass `host_sample`; `pl_kv256.py:126` sets `_onchip = temp>0 and not host_sample`). Resolve only by reading the Kria's systemd ExecStart + the on-board `a53_daemon.py` (which may be hand-edited vs repo). So whether the live sampling is on-chip (baked ~0.85) or host-side (0.4) is UNRESOLVED (`live-demo-topology.md:13`). Also note `--gen-chars 104` in the live unit (vs the 48 kv256 default). What could NOT be independently re-derived from the repo alone: the exact runtime daemon flags — those come from project memory, which explicitly flags itself "not derivable from the repo."

## MEASURED throughput ladder

The authoritative MEASURED tok/s ladder is `fabric/progress.py:30-81`. All rungs are the same 4-layer d=256 goformer, B=1 greedy unless noted. Everything up to the split-brain record is **doc-6 aggregate throughput** (N≥4 rungs decode with attention hardwired T=1 → blistering aggregate but degenerate text). The final block is the **doc-7 faithful stream** (N=1, full on-chip KV window, real messages) — a different, honestly-separated metric.

| tok/s | Config | Cycles / clock | Tag | Note |
|---|---|---|---|---|
| 0.07–10.35 | Early on-fabric baselines (Python MMIO → C MMIO driver) | — | MEASURED | matmul off the critical path once compiled |
| 44.32 | HW sequencer @40 MHz (CPU out of loop) | — | MEASURED | A53 doing the non-matmul forward was the ~11 wall |
| 75.8 | + resident-read GEMV (no per-matmul reload) | — | **SIM** | ~42% of cycles were re-streaming weights |
| 231.0 | + PE=256 wide lanes (256 MACs/cycle) | — | MEASURED | 16× fewer group passes |
| 751.78 | + GELU stream + LN pipeline (PE=128 @125 MHz) | — | MEASURED | Fmax 50→125 MHz (STA-safe 71/430) |
| 1,882.7 | + wide P-lane datapath (P=4 @100 MHz) | 53,116 cyc/tok | MEASURED | 1-elem/cyc serial loops → P-wide (2.5×) |
| 2,483.9 | + BRAM sync-read scratch (P=8 @125 MHz) | 50,324 cyc/tok | MEASURED | LUTRAM → BRAM; STA 79.5, clean to 125, breaks 142.9 |
| 3,511.6 | + P-wide GEMV boundary (@125 MHz) | 35,596 cyc/tok | MEASURED | act-feed + readback P/cycle |
| 5,448.8 | + LANES=256 (72-bit URAM banks, 60/64) | 22,941 cyc/tok | MEASURED | half the GEMV passes; breaks 142.9 |
| 9,295.4 | + cycle-floor cut + deep pipeline (@166.7 MHz) | 17,930 cyc/tok | MEASURED | fused RB+DQ+GELU, P-wide attn; Fmax 85→131 STA |
| 11,143.9 | + 3-stage act-quant → 200 MHz | 17,947 cyc/tok | MEASURED | last sub-8ns path; silicon clean at 200 |
| 16,969.3 | + batch GEMM N=4 (4 streams share one weight pass) | 47,144 cyc / 4 tok @200 | MEASURED | 3/3 bit-exact ×4 |
| 17,740.6 | + ping-pong N=8 (NL overlaps GEMM @166.7) | 75,157 cyc / 8 tok | MEASURED | 8/8 bit-exact; 200 fails |
| 19,275.6 | + single-pass merge N=8 (both groups share one weight pass @166.7) | 69,172 cyc / 8 tok | MEASURED | 8/8 bit-exact; 200 fails |
| 24,134.0 | + N=16: 12 DSP-packed banks, shared LN/attn (fits: 106.5k LUT @166.7) | 110,494 cyc / 16 tok | MEASURED | 16/16 bit-exact; 200 fails |
| 25,744.5 | + softmax latency cut (@166.7) | 103,582 cyc / 16 tok | MEASURED | dead wait-states between exp/sum/recip removed |
| 36,970.7 | + **SPLIT-BRAIN N=14**: two cohorts on dual-ported URAM @166.7 | 63,113 cyc / 14 tok | MEASURED | TDP image + shared LN/attn/dq; 14/14; 200 fails |
| 46,604.4 | + N=16 @200 MHz (LN un-retimed + AQ 32×48 range-proof) | 68,663 cyc / 16 tok | MEASURED | 16/16; 250 fails |
| 56,262.7 | + schedule pipelining (AQ/RUN overlap, stream-granular NL, attn call cuts @200) | 56,876 cyc / 16 tok | MEASURED | 16/16; 250 fails |
| **59,965.5** | **+ TMAX=16 + per-cohort attn + CTX stream + LN prod·gamma split @200** | **53,364 cyc / 16 tok** | **MEASURED (RECORD)** | 16/16 bit-exact, 3/3; **250 hangs** |
| 11,343.2 | **FAITHFUL N=1, T=128 window** (doc-7 R1–R4f, on-chip KV @142.9) | 12,594 cyc/tok avg | MEASURED | 119-tok real message, 3/3 bit-exact |
| 13,162.3 | + R5 cone ladder → 166.7 MHz (7 paths pipelined, OOC MET @5ns) | 12,662 cyc/tok | MEASURED | 3/3; 200 fails on MAC floor |
| 16,087.5 | + schedule trims + MAC stage (KVW/RB/LN overlap, 8ns route @166.7) | 10,360 cyc/tok | MEASURED | 3/3; **the earlier trims build — NOT the deployed one** (deployed = gum build, 16,227 fabric / 17,936.6 greedy, §45); 200 fails |
| 20,000.0 | × faster clock OR fewer cycles (MMCM ~187 MHz, or K4/smaller model) | — | **PROJECTED** | PS PLL caps at 166.7; 20k needs ~207 MHz on this design |

Reference guide-lines (not fabric rungs, same model B=1 greedy): A53 chat 11.0, A53 INT4 GEMV microbench 177.8, XPS15 CPU (PyTorch KV) 356.0, XPS15 RTX 3050 Ti 719.0, XPS15 ONNX Runtime KV 1,273.0 — all MEASURED (`progress.py:85-89`).

## Hard-won gotchas

Hard-won traps. Every one of these cost a failed build or a silent-wrong-answer run.

**iverilog-2012 sim traps** (the RTL rewrites hit these repeatedly):
- Cannot `$readmemh` a 2D-array row — use a flat/packed ROM + ranged `$readmemh(file, mem, start, end)`.
- A **variable** `+:` part-select on an *element of an unpacked array* reads X. Only ever variable-part-select a **plain reg**. Over-wide part-selects also read X.
- The synth FSM goes RUN→DRAIN directly, but **SETTLE is sim-only** (waits ~4 cyc for the MAC tail / latches y_lat). Silicon finishes each GEMM call ~4 cyc earlier than sim, so sim cyc/tok is an **upper bound** (~0.8%). The −273/−297-cyc silicon-vs-sim SETTLE signature has held 5–6 consecutive builds (`gemm_banked_resident_vec.sv:327-348`, `docs/6:90-102`).

**Wide-word banking, NOT `[P][rows]`** (the single most load-bearing structural gotcha): a banked `reg [W] buf [0:P-1][0:ROWS-1]` read at a *variable row* synthesises to giant per-lane row-muxes (MUXF7/LUT blow-up) — measured at **LUT 185% / MUXF7 151%, DRC-failed before placement**. The correct layout is one row-addressed wide word: `reg [P*W-1:0] buf [0:ROWS-1]` with lane l in bits `[l*W +: W]`, so the variable row is a memory *address* (free) and only a small constant lane-select remains. Applies to **every** scratch buffer, not just weights (`WIDE-WORD-DATAPATH-LOG.md:71-91`).

**URAM residency geometry:** a single 1024-bit-wide memory pads to 16 wide × 4 cascade = 64 URAM = the whole device → Vivado marks `ultra` infeasible and silently falls back to ~400k LUTRAM. The fix is **72-bit-wide banks** (URAM-native; 15×4 = 60 URAM at L=256) (`gemm_banked_resident_vec.sv:79-88`).

**Each stream's MACs must be one `keep_hierarchy` leaf:** at N=8 (1,024 mults) Vivado's bulk-multiplier optimization detaches the URAM read register and refuses `ultra`, dumping weights to ~450k LUTRAM. Keep each stream's 128 MACs hierarchically isolated (`gemm_banked_resident_vec.sv:465-474`).

**TDP UltraRAM cannot be HDL-inferred** — dead in Vivado 2025.2, falls back to BRAM. True-dual-port (split-brain's shared weight banks; kv_bank's twin read streams) requires an **`xpm_memory_tdpram` macro, `MEMORY_PRIMITIVE=ultra`, `WRITE_MODE=no_change` on both ports**. RTL uses a dual-dialect pattern: xpm under `SYNTHESIS`, behavioral 2-port array under iverilog (`weight_bank_tdp.sv:14-25`, `SPLIT-BRAIN.md:57-68`).

**Silent all-zero ROM from a missing `$readmemh`:** a missing ROM init file synthesises to a silent all-zero ROM. On first silicon a missing `inv_lut` made every KV code = 0 (ctx wrong, qkv perfect — a maddening partial failure). Build TCL now **hard-errors on any missing .mem** (`build_bd_seq_kv.tcl:64`) and impl greps every synth `runme.log` for `[Synth 8-4445]` and fails (`impl_seq_kv.tcl:11-17`) (`WIDE-WORD-DATAPATH-LOG.md:1247-1276`).

**BUILD OUTSIDE OneDrive.** Vivado build dirs and .mem sim dirs go in `C:/kevbuild/...`, never inside the repo (which is under OneDrive) — the `cldflt` cloud-sync filter locks files mid-build and corrupts runs.

**Forward slashes for `--dir` paths.** When invoking `run_*.py` through bash, pass `--dir C:/kevbuild/...` with forward slashes — bash eats backslashes and you get a junk relative dir in the repo.

**Silicon overclock vs STA pessimism** (`-2LV` speed grade): STA is untrustworthy here — designs that close only ~70–114 MHz in STA run bit-exact at 125–200 MHz on silicon (~1.3×–1.76× margin band, e.g. STA 103.6→ran 142.9; STA 114.3→ran 166.7). So **no tok/s is ever claimed from a timing report**; builds target a clock that closes and the real ceiling is found by the board `--fclk` sweep (`CLAUDE.md`, `docs/6:90-102`).

**The PL clock must be forced AND verified.** A flat `fpgautil -b` bitstream load does **NOT** apply the block design's `PL0_REF_CTRL FREQMHZ` — `pl0_ref` stays at the ~100 MHz system default. A design timing-closed at 40 MHz then violates setup on the wide arithmetic and produces **non-deterministic garbage while the short MMIO/cycle-counter paths still look sane** (so a broken run shows a plausible CYCLES value with random compute). Every driver forces + verifies via `/sys/devices/platform/fclk0/set_rate` and computes tok/s from the **PLL's quantised readback rate**, not the requested rate (`pl_seq_chat.py:65-100`).

**PLL clock quantisation:** the PS PLL only offers 1000/N divider steps, so usable board clocks are quantised to **125 / 142.9 / 166.7 / 200 MHz** — there is NO step between 166.7 and 200 (175 snaps down, 183.3 snaps up), which is why several records sit exactly at 166.7 (`WIDE-WORD-DATAPATH-LOG.md:1379-1381`).

**Register-map confusion (silent mis-read):** three distinct maps exist. Baseline SEQR/SQRF puts CYCLES@0x30, IDCODE@0x34; SQRV and SQ16/SQSB put CYCLES@0x28, IDCODE@0x2C. Only whole 32-bit stores are legal (byte slices corrupt the W_DATA stream). Every driver **verifies IDCODE first and aborts on mismatch** (`pl_seq_chat.py:103-107`, `pl_seq_vec.py:40`).

**Embed upload TMAX must equal the bitstream TMAX generic**, or positions >0 corrupt. SQ16/SQSB stream embeds at runtime through `EDATA` (0x4C) as Q6.25 INT32; SQRV/vec/kv bake them into the bitstream (`pl_seq_sb.py:76-84`).

**`vec_attn`'s `probmem[ji[4:0]]` silently WRAPS at T>32** — a latent trap in the record-path attention; `vec_attn_w` fixes it with full-width `probmem[jc]` (`vec_attn.sv:310`, `vec_attn_w.sv:11-13`).

**DSP packing is exactly 2.0 MAC/DSP — 3.0 is impossible.** Two +8-biased INT4 weights 22 bits apart share one INT8 act in one DSP48E2; 3/DSP needs a 28-bit operand > the 27-bit port AND a 66-bit acc > the 48-bit accumulator, falsified over 1.2M randomized products. This caps DSP-batch streams at **N=16** at LANES=128 (`dsp3_pack_proof.py:26-32`).

**Live sampling path is UNRESOLVED (on-chip baked vs host-side).** The bitstream bakes a Gumbel-max LUT at temp 0.85. The out-of-repo memory note says the deployed demo samples host-side @`--temp 0.4 --top-k 10` (0.85 "was gibberish"), but the committed code with exactly those flags samples ON-CHIP at the baked temp and ignores top-k (`a53_daemon.py:289-304` passes no `host_sample`; `pl_kv256.py:126`). Resolve only by reading the Kria's systemd ExecStart + the on-board `a53_daemon.py` (which may be hand-edited vs repo). **Per committed code, editing `--temp` on the unit does NOT change output temperature (on-chip temp is baked)** (`live-demo-topology.md:13`).

## Glossary

| Term | Meaning |
|---|---|
| **goformer** | The nanoGPT-lineage transformer this project runs in fabric: 4 layers, d=256, 4 heads, head_dim=64, MLP hidden 1024, vocab 193, INT4 weights. |
| **Kevin / Keviniser** | The spaCy POS-based preprocessor that strips function words / flattens inflection to compress English into telegraphic "Kevin-speak." Trains the compressed distribution; the model itself generates telegraphic text. |
| **the bandwidth wall** | Single-stream decode is memory-bandwidth bound; A53 + PL share one ~20 GB/s DDR controller, so DDR-resident weights get zero fabric uplift. The whole thesis. |
| **zero-DRAM / resident** | The entire INT4 weight image lives in on-chip URAM/BRAM, never DDR. The ~3 MB on-chip budget is the hard ceiling. |
| **the fusion** | Telegraphic output = fewer tokens = less weight streamed + smaller KV = more fits on-chip = faster. Dumbness and speed are the same property. |
| **split-brain** | The record engine's architecture: 16 streams cut into two independent cohorts of NC=8 on the two ports of a true-dual-port URAM weight image (cohort 0 port B, cohort 1 port A). Shares only weights + arbitrated LN/dequant/(shared) attention. |
| **`sequencer_sb`** | The live split-brain top module (IDCODE SQSB). Holds the record 59,965.5 tok/s. Instantiates two `cohort_engine`s. |
| **`cohort_engine`** | One split-brain cohort: one `nl_engine` + one `gemm_cohort_vec` + a local GE FSM. Mechanically = `sequencer_pp` collapsed to one stream-group with merge/GWAIT deleted. |
| **`nl_engine`** | The non-linear / phase FSM (23 states NL_IDLE..NL_DONE) that walks the transformer phases and issues GEMM descriptors. Where the phase sequence lives (there is no monolithic top FSM). |
| **GE FSM** | The GEMM-executor state machine inside `cohort_engine` (GE_IDLE/AQ/WAIT/RB/RBN/DQW) that runs the actual matmul calls. |
| **`sequencer_vec`** | The doc-7 KV-faithful single-stream engine (IDCODE SQRV). Adds the persistent INT8 KV cache + on-chip Gumbel sampler. **The engine the public chat runs** (as `gemv_seqkv_gum.bit.bin`). |
| **Track A vs Track B (attention)** | Track A = `vec_attn` (record engine): transient full-precision Q.16 K/V re-streamed each call, no persistent cache. Track B = `kv_bank` + `vec_attn_w` (faithful engine): persistent INT8 K8/V8 cache across tokens. Do not conflate. |
| **the two campaigns** | Doc 6 = N=16 aggregate throughput (record 59,965.5, but N≥4 hardwires attention T=1 → degenerate text). Doc 7 = N=1 faithful stream (real messages with full context, ~16k deployed). |
| **the two ceilings** | FABRIC ceiling = pure PL cycles (the 59,965.5 / 16,087.5 numbers). ROUND-TRIP ceiling = what a chat user feels, bound by serving/RTT not silicon. |
| **the 100k identity** | "100k tok/s = 16 streams × 250 MHz / 40,000 cyc." Does NOT close on the KV260 at this model size: streams maxed at 16, clock 200 MEASURED (250 route-dead), cycle floor ~51,100 not 40,000. CLOSED (model-shrink was user-vetoed). |
| **wide-word banking** | The load-bearing memory layout: `reg [P*W-1:0] buf [0:ROWS-1]` (row = address), not `reg[W] buf[0:P-1][0:ROWS-1]` (row = giant mux). |
| **DSP packing** | Two INT4 weights sharing one INT8 activation in one DSP48E2 (WP487-style, 22-bit gap). Proven exactly 2.0 MAC/DSP; 3.0 impossible. |
| **double-pump (DP)** | The 100k-campaign lever: MAC accumulator runs in a 2× clock domain (clk2x MMCM), 2 K-steps/clk. Full DP=1 build: OOC-timing-MET only (200/400 MHz), never board-run. A MAC-only DP branch (`dp-hw-maconly`) WAS run on silicon bit-exact but the fabric→clk2x data feed walled at 50 MHz — that measurement killed the lever (DEAD). |
| **TMAX** | On-chip KV window depth. Record build = 16 (Track A); shipping faithful = 128 (Track B); 256 = full trained context (busts BRAM with K8). |
| **KV cache / K8-V8** | The persistent quantise-at-write/dequantise-at-read INT8 K/V codes + per-(layer,kv,head,pos) asymmetric headers in `kv_bank`. K8/V8 no-Hadamard is +0.09% NLL, the shipping config. |
| **Q-formats** | Fixed-point layouts pinned per block: residual **Q6.25**, LN-out **Q.22**, gamma **Q4.20**, q/k/v **Q.16**, scores **Q8.8**, probs **Q1.20**, GELU **Q4.12**. |
| **dequant / activation-quant** | Dequant: INT32 accumulator → residual Q-format via per-channel folded scale (24-bit mantissa). Act-quant: Q.22 → INT8 via `inv_sact = round(2^40/s_act)` (ISH=40). |
| **rsqrt** | LayerNorm reciprocal-sqrt: 64-entry Q1.16 seed LUT + √2 correction for odd exponent + 2 Newton steps. |
| **Gumbel-max** | Sampling trick: argmax(logit + T·g) with g~Gumbel(0,1) ≡ softmax(logit/T) sample. Lets the fabric sample with no softmax-over-logits and no host logit readback. On-chip path baked at temp 0.85. **UNRESOLVED whether on-chip (baked) or host-side (temp 0.4) is the live path** — committed code with the memory-note flags samples on-chip; see the "live sampling" note. |
| **the gate / bit-honest** | No speed number is trusted until the fabric token stream matches `seq_ref` bit-exactly (or transcendentals cosine > 0.9999; the binding gate is always token-stream identity). |
| **`seq_ref.IntSequencer`** | The per-phase integer reference (`fabric/stage3/seq_ref.py`) — the single bit-true contract the RTL FSMs are gated against. Exposes every phase intermediate. |
| **goformer reference ladder** | The Python refinement chain: `goformer_full` (float) → `_kv` (incremental) → `_q` (Q-format) → `_fixed` (fabric non-linears) → `_seq` (integrated) → `seq_ref` (per-phase integer). |
| **gate harness** | A `run_*.py` script: writes .mem ROMs → `iverilog -g2012` → `vvp` → per-phase `.out` compare → one sentinel verdict line. THE inner loop for RTL work. |
| **IDCODE** | Per-bitstream 32-bit identifier a driver reads first to confirm it's talking to the right register map: SEQR/SQRF (baseline), SQRV (0x53515256, KV-faithful), SQSB (0x53515342, split-brain), SQ16. |
| **fclk / fclk0** | The forced PL reference clock. Set + verified via `/sys/devices/platform/fclk0/set_rate`; PLL-quantised to 125/142.9/166.7/200 MHz. |
| **-2LV / STA overclock** | The KV260 part `xck26-sfvc784-2LV-c` speed grade; static timing is pessimistic ~1.3×–1.76× vs measured silicon. |
| **a53_daemon** | The asyncio TCP daemon on the Kria that owns the /dev/mem device and speaks length-prefixed-JSON RPC; selects engine t1/kv/kv256 at startup; fronts the public chat. |
| **round-trip lift** | Moving sampling on-chip via Gumbel-max removed 193 host logit-reads/token, lifting localhost round-trip ~5.6× (1k→5,600 tok/s) while keeping the fabric bit-exact. |

---

# Subsystem detail

## Sequencers & Top-Level Control — the Split-Brain Spine (sequencer_sb / cohort_engine / nl_engine)

The "spine" of the Kevin-on-Kria fabric is the control hierarchy that turns a `go` pulse into N argmax tokens with zero host I/O between them. The LIVE engine is a three-level hierarchy: `sequencer_sb` (the split-brain top) instantiates two identical `cohort_engine`s; each `cohort_engine` wraps one `nl_engine` (the non-linear FSM that decodes the transformer phases and requests GEMM calls) plus one GEMM datapath (`gemm_cohort_vec`) driven by a local GE (GEMM-executor) FSM. There is NO single monolithic phase-decode FSM inside `sequencer_sb`; the phase sequence lives in `nl_engine` (NL_* states), and the GEMM execution lives in `cohort_engine` (GE_* states). The two talk over a small request/descriptor/ack handshake (`req`/`d_*`/`served`).

Split-brain means 16 streams/pass are cut into two fully independent cohorts of NC=8 (cohort 0 = streams 0..7, cohort 1 = streams 8..15). The cohorts share ONLY the resident INT4 weight image (`weight_bank_tdp`, read through both true-dual-port URAM ports — cohort 0 on port B, cohort 1 on port A) plus a 2-way-arbitrated LayerNorm unit, a 2-way-arbitrated dequant+GELU readback channel, and a per-cohort-ported embed bank. Attention is either per-cohort (ATT2=1, sim default) or one shared+arbitrated unit (ATT2=0, the config bitstreams build for fit). There is no merge and no cross-cohort weight pass: the ~53k-cycle serial queue that used to bottleneck a single N=16 engine on one URAM read port now runs in two parallel halves.

The engine lineage runs sequencer → sequencer_fast (P=1 single-stream KV decode) → sequencer_gemm (N-stream batched, one weight pass serves N) → sequencer_pp (N=8 two-group ping-pong, birthed nl_engine) → sequencer_vec (P-wide vector datapath, the wide-word/BRAM sync-read rewrite) → and finally the split-brain pair sequencer_sb + cohort_engine (cohort_engine is sequencer_pp mechanically collapsed to one stream-group with merge/GWAIT deleted). Everything older than sequencer_sb/cohort_engine/nl_engine is historical — kept for reference and the log's cycle-march story, not built.

The MEASURED record is 59,965.5 tok/s @200 MHz: the §26/§27 "TMAX=16 architectural wave" running 53,364 silicon cycles per 16-token pass, 16/16 bit-exact, 3/3 runs (fabric/stage3/WIDE-WORD-DATAPATH-LOG.md:983). The 100k target ("16 streams × 250 MHz / 40k cyc") needs a post-route-MET 5ns build plus silicon's ~1.3× overclock margin; streams are maxed (N=16), cycles are ~53k gated toward ~40k, and 250 MHz silicon is the open gate.

## 1. Engine lineage — which sequencer is LIVE

All seven sequencer modules live in `fabric/stage3/rtl/`. Only the last (split-brain) is built to silicon today; the rest are historical rungs on the cycle-reduction ladder documented in `WIDE-WORD-DATAPATH-LOG.md`.

| Module | File | Role in lineage | Status |
|---|---|---|---|
| `sequencer` | sequencer.sv (795 L) | Tier-3 single-stream KV-cached decode FSM, CPU out of the loop; the original 4-block autoregressive loop wiring the 4 gated datapath blocks (gemv_banked/layernorm/softmax/gelu_lut) (sequencer.sv:1-30) | historical |
| `sequencer_fast` | sequencer_fast.sv (825 L) | Same single-stream decode, P=1 serial lanes, ~166k cyc/token (WIDE-WORD-DATAPATH-LOG.md:31) — the P=1 baseline | historical |
| `sequencer_gemm` | sequencer_gemm.sv (667 L) | N-stream batched: GEMV→GEMM, ONE weight pass serves N tokens; scratch banks stream-flattened row=stream*ROWS+r (sequencer_gemm.sv:2-14) | historical |
| `sequencer_pp` | sequencer_pp.sv (696 L) | N=8 ping-pong, two stream-groups of G=N/2; birthed `nl_engine` (two parallel per-group NL FSMs) with shared GEMM/dequant/gelu (sequencer_pp.sv:1-19) | historical (direct parent of cohort_engine) |
| `sequencer_vec` | sequencer_vec.sv (992 L) | P-wide vector datapath, wide-word banking + BRAM synchronous-read rewrite (sequencer_vec.sv:1-24); the P=8 sweet-spot core | historical |
| **`sequencer_sb`** | **sequencer_sb.sv (394 L)** | **LIVE split-brain top: two independent NC=8 cohorts sharing the TDP weight image + arbitrated LN/attn/embed/dequant-gelu** (sequencer_sb.sv:1-16) | **LIVE** |
| **`cohort_engine`** | **cohort_engine.sv (658 L)** | **LIVE: one split-brain cohort = 1 nl_engine + 1 GE FSM + 1 gemm_cohort_vec; sequencer_pp collapsed to one group, merge/GWAIT/aq_eng deleted** (cohort_engine.sv:1-16) | **LIVE** |
| `nl_engine` | nl_engine.sv (628 L) | The non-linear phase-decode FSM (NL_* states); shared by sequencer_pp and cohort_engine (nl_engine.sv:1-21) | **LIVE (inside cohort_engine)** |

Lineage note: `cohort_engine` was "derived MECHANICALLY from sequencer_pp by collapsing its two stream-groups to ONE engine and DELETING the merge / GWAIT / aq_eng machinery (a cohort never shares a weight pass — that whole desync problem dissolves)" (cohort_engine.sv:9-12). The GE FSM in a cohort "always serves its single engine's 8 streams: gmerge≡1, glim≡N-1, no GE_AQW" (cohort_engine.sv:11-12).

## 2. The three-level LIVE hierarchy

```
sequencer_sb  (N=16 top; owns shared resources + 2-way arbiters)
 ├─ weight_bank_tdp  u_wbank   (INT4 weight image; port B→coh0, port A→coh1)
 ├─ embed_bank_tdp   u_embank  (tok_emb URAM; port B→coh0, port A→coh1; gnt==req)
 ├─ layernorm_vec    u_ln      (SHARED, 2-way arbiter ln_owner/ln_busy)
 ├─ vec_dequant×2 + vec_gelu×2 + dqm_w/dqe_w ROMs (SHARED dq channel, arbiter dq_owner/dq_busy)
 ├─ vec_attn         (ATT2=1: u_attn0 + u_attn1 per cohort; ATT2=0: 1 shared u_attn + arbiter)
 ├─ cohort_engine    coh0      (streams 0..7)
 │   ├─ nl_engine    eng       (NL_* phase FSM; owns all N=8 streams; G=NC)
 │   └─ gemm_cohort_vec u_gemm  (GEMM datapath) + GE_* FSM (in cohort_engine body)
 └─ cohort_engine    coh1      (streams 8..15)  — identical instance
```

- `sequencer_sb` instantiates the two cohorts at sequencer_sb.sv:316-372, wiring each cohort's shared-resource ports (`waddr/wword`, `emb_*`, `ln_*`, `at_*`, `dq_*/gl_*`) to the arbitrated shared units. `tok_outs = {tok_outs1, tok_outs0}` (sequencer_sb.sv:374).
- `cohort_engine` instantiates one `nl_engine` (as `eng`, cohort_engine.sv:148-181) and one `gemm_cohort_vec` (as `u_gemm`, cohort_engine.sv:201-212). The GE FSM lives directly in cohort_engine's always-block (cohort_engine.sv:393-611).
- `nl_engine` holds the transformer phase FSM `nl` (NL_IDLE..NL_DONE) at nl_engine.sv:372-627.

## 3. The phase-decode FSM (nl_engine `nl` state machine)

The per-token forward is: `embed → NLAYER×[LN1 → qkv GEMM → attn(softmax+KV) → proj GEMM → +res → LN2 → mlp_fc GEMM → GELU → mlp_proj GEMM → +res] → LN_f → head GEMM → argmax → tok_out` (sequencer.sv:8-11 documents the canonical dataflow).

State encoding (nl_engine.sv:230-236):
```
NL_IDLE=0, NL_EMB=1, NL_LGAM=2, NL_LFEED=3, NL_LCOLL=4,
NL_QKV=5, NL_WQKV=6, NL_AST=7, NL_ALD=8, NL_ACL=9,
NL_PROJ=10, NL_WPROJ=11, NL_RES1=12,
NL_FC=13, NL_WFC=14, NL_MP=15, NL_WMP=16, NL_RES2=17,
NL_HEAD=18, NL_WHEAD=19, NL_ARG=20, NL_ARG2=21, NL_DONE=22
```

Two control regs sequence the block/phase structure: `blkg` (block index 0..NLAYER-1) and `lnphase` (0=LN1→QKV, 1=LN2→FC, 2=LNF→HEAD) (nl_engine.sv:237-238).

Phase-by-phase (all citations nl_engine.sv unless noted):

1. **EMBED (NL_EMB, :390-409)** — for each stream bs, for each of ROWS=D/P rows, read tok_emb (via shared emb port, `emb_addr = tok_id_b*EROWS + fr`, :309) and add pos_emb (`pemb_r`), writing the Q6.25 residual into `xres_bank`. The residual add is registered one cycle into `emb_wd_p` (the "250 GATE" cone-split, :263-273, :398-401). Loops bs=0..G-1 then → NL_LGAM.

2. **LAYERNORM (NL_LGAM/NL_LFEED/NL_LCOLL, :410-441)** — LN1 (or LN2/LNF depending on lnphase). NL_LGAM asserts `ln_req` and idle-waits for `ln_gnt` from the shared arbiter (:410-414). NL_LFEED streams ROWS rows of (xres_r, gamma) into the shared `layernorm_vec` (:416-423). NL_LCOLL collects the Q.22 result rows into `lnout1_bank` (lnphase 0/2) or `lnout2_bank` (lnphase 1) (:424-441). Loops over streams; drops `ln_req` on the last stream's last row.

3. **QKV GEMM (NL_QKV/NL_WQKV, :443-454)** — issues descriptor: `d_m=D3=768`, `d_k=D=256`, `d_asrc=0` (LN1 source), `d_dst=0` (qkv bank), `d_frac=16`; `req<=1; rb_clr=1` (:444-447). NL_WQKV waits for `rb_rdy[0]` (stream-0 readback committed) then enters the attention loop (:454) — the stream-granular NL/GEMM overlap.

4. **ATTENTION (NL_AST/NL_ALD/NL_ACL, :455-479)** — per stream, per head (hh=0..NHEAD-1). NL_AST requests the attention unit gated on this stream's readiness (`at_req <= sready`, :456). NL_ALD loads 3*HR q/k/v rows into vec_attn (:462-469). NL_ACL collects ctx rows into `ctxv_bank` and advances head/stream (:470-479) → NL_PROJ when done.

5. **PROJ GEMM (NL_PROJ/NL_WPROJ, :480-488)** — `d_m=D=256`, `d_k=D=256`, `d_asrc=1` (ctx source, note the >>>(RESID_FRAC-LN_OUT_FRAC) shift, cohort_engine.sv:452-457), `d_dst=1` (attn bank), `d_frac=25`. NL_WPROJ waits `rb_rdy[0]` → NL_RES1.

6. **RESIDUAL-1 (NL_RES1, :489-504)** — per stream (gated on `sready`), `xres = xres + attn` over ROWS rows (:492-496), writing back to `xres_bank`. Last stream → NL_LGAM with lnphase=1 (LN2).

7. **LN2** — reuses NL_LGAM/LFEED/LCOLL with lnphase=1 → NL_FC.

8. **FC GEMM (NL_FC/NL_WFC, :505-511)** — `d_m=D_MLP=1024`, `d_k=D=256`, `d_asrc=2` (LN2 source), `d_dst=2` (mlpbuf via dwm, GELU'd on readback), `d_frac=12`. NL_WFC waits `!req` (whole-call, since FC dst=2 has no per-stream nl post-processing) → NL_MP.

9. **MLP-PROJ GEMM (NL_MP/NL_WMP, :512-520)** — `d_m=D=256`, `d_k=D_MLP=1024`, `d_asrc=3` (GELU'd mlpbuf source, <<<(LN_OUT_FRAC-GELU_FRAC), cohort_engine.sv:462-465), `d_dst=3` (mlp bank), `d_frac=25`. NL_WMP waits `rb_rdy[0]` → NL_RES2.

10. **RESIDUAL-2 (NL_RES2, :521-542)** — per stream, `xres = xres + mlp`. On last stream: if `blkg==NLAYER-1` go to LNF (lnphase=2), else advance `blkg`, reset lnphase=0, loop to NL_LGAM (next block).

11. **LN_f** — NL_LGAM/LFEED/LCOLL with lnphase=2 (gamma index `lgam=NLAYER*2`, :534) → NL_HEAD.

12. **HEAD GEMM (NL_HEAD/NL_WHEAD, :543-558)** — `d_m=VOCAB=193`, `d_k=D=256`, `d_asrc=0` (LNF via lnout1), `d_dst=4` (head bank), `d_frac=25`, `d_wbase=WB_HEAD`. NL_WHEAD waits `rb_rdy[0]`, re-arms argmax scratch → NL_ARG.

13. **ARGMAX (NL_ARG/NL_ARG2, :559-601)** — per stream, a pipelined 8-way-then-tree argmax over ARROWS=(VOCAB+P-1)/P head rows. Reads head_r, per-row computes 4 pairwise maxes (pv0..pv3/pi0..pi3, :564-576), reduces to a row-winner (:579-586), tracks running `best_val/best_idx`, masks indices ≥ VOCAB to -2^31 (:568-569). On last row writes `tok_out_b[bs]`. NL_ARG2 advances stream or, on last stream, sets `done_o<=1` → NL_DONE (:593-601).

14. **NL_DONE (:602)** — holds done_o high until next `go`.

### Phase / GEMM-descriptor summary table
| NL phase | GEMM? | d_m | d_k | d_asrc | d_dst | d_frac | source→dest |
|---|---|---|---|---|---|---|---|
| QKV | yes | 768 | 256 | 0 | 0 | 16 | LN1→qkv |
| PROJ | yes | 256 | 256 | 1 | 1 | 25 | ctx→attn |
| FC | yes | 1024 | 256 | 2 | 2 | 12 | LN2→mlpbuf(GELU) |
| MP | yes | 256 | 1024 | 3 | 3 | 25 | GELU→mlp |
| HEAD | yes | 193 | 256 | 0 | 4 | 25 | LNF→head |

(from NL_QKV/PROJ/FC/MP/HEAD descriptor writes, nl_engine.sv:444-546)

## 4. The GEMM-executor (GE) FSM — cohort_engine

The GE FSM services the descriptor requests. State encoding (cohort_engine.sv:288-290):
```
GE_IDLE=0, GE_AQ=1, GE_AQN=2, GE_RUN=3, GE_WAIT=4, GE_RB=5, GE_RBN=6, GE_DQW=7
```
(GE_AQN/GE_RUN are retired — "AQ now starts the run", cohort_engine.sv:294.)

- **GE_IDLE (:403-414)** — on `g_req` (nl_engine's `req`), latch the descriptor into active-call regs `a_wbase/a_m/a_k/a_asrc/a_asel/a_frac/a_dqrow/a_dst`, reset counters, pulse `gv_xrst`, → GE_AQ.
- **GE_AQ (:423-557)** — ROW-MAJOR activation-quantize, OVERLAPPED with the GEMM run. A 4-deep pipeline (issue→multiply→shift→write) reads AQ sources (lnout1/ctxv/lnout2/mlpbuf per d_asrc, :446-466), multiplies by `inv_sact[a_asel]` (32×48 signed, :497), rounds/shifts (`>>>(LN_OUT_FRAC+ISH)`, :508-510), saturates to INT8 (:523-524), and writes activation rows into the gemm's x-buffer (`gv_xwe`, :531). The run is launched after AQ_START_MARGIN=2 rows commit (:295, :538-542); gemm_cohort_vec's stall-guard holds the N=P=8 rate tie. When the last x-write commits → GE_WAIT (:546-556).
- **GE_WAIT (:558-566)** — on `gv_done`, reset readback counters; `ge <= dq_gnt ? GE_RB : GE_DQW` (zero-cycle grant if the shared dq channel is free).
- **GE_DQW (:567)** — park (holding `dq_req`) until the other cohort releases the dq channel.
- **GE_RB (:568-600)** — readback walk: read a pair of gemm result rows/clk (`gv_yout`+`gv_yout2`), stream through the shared dequant (`dq_vin_o`, :577) and — for dst=2 (FC) — the shared GELU (`gl_vin_o`, :583). Dequant/GELU results are written back into the nl_engine banks via `dwr_*/dwmr_*` (combinational drive at cohort_engine.sv:365-391). Advances stream on completion; `g_done_p` pulses when the last stream's last readback pair commits (:337-339).
- **GE_RBN (:601-607)** — 2-cycle gap to refill the gemm readback pipeline when advancing streams.

The dq-channel request spans the whole readback: `dq_req = (GE_WAIT && gv_done) || GE_DQW || GE_RB || GE_RBN` (cohort_engine.sv:299-300).

### DOUBLE-PUMP (DP) note
The entire GE_AQ / GE_RB path is duplicated into "the pair's 2nd stream/row" logic (aq_*_b registers, dq_gemvy_o2, gl_x_o2, etc., cohort_engine.sv:267-283, :424-534). This is the DOUBLE-PUMP-100K Stage 1 lever: with DP=1 the MAC runs at 2 K-steps/clk and the AQ producer emits 2 stream-rows/clk. **DP defaults to 0** (parameter default, cohort_engine.sv:42; sequencer_sb.sv:44); the run_sb_seq harness `--dp` default is 0 (run_sb_seq.py:138). At DP=0 the `_2`/`_b` paths still exist in RTL but the second read is the same-address behavioral model (weight_bank_tdp.sv:31-40 explains the sim model vs. clk2x silicon plan).

## 5. Module parameters (definition sites, meaning, live values, ranges)

Parameters are declared identically on `sequencer_sb` (sequencer_sb.sv:19-45) and `cohort_engine` (cohort_engine.sv:19-42); `nl_engine` (nl_engine.sv:24-41) takes a subset with `G` in place of `N`/`NC`.

**The load-bearing five the task asks for:**

- **N** — total streams per pass. `sequencer_sb` default 16 (sequencer_sb.sv:25). Legal ≥ 2, even (the DP pair logic and the two-cohort split assume even; N=NC×2). Live/record value 16. In `cohort_engine`/`nl_engine`, `N`/`G` = the cohort's own streams (=NC).
- **NC** — streams per cohort. `sequencer_sb` default 8 (sequencer_sb.sv:26). Passed as `.N(NC)` into each cohort_engine (sequencer_sb.sv:316,345). Fit variant NC=7 → N=14 (WIDE-WORD-DATAPATH-LOG.md:676). Note: the host-debug cohort select is arithmetic (`rd_is_c1 = rd_stream >= NC`, `rd_local1 = rd_stream - NC`, sequencer_sb.sv:311-312) precisely so NC need not be a power of two (comment at :310, :388-390 warns a high-bit slice broke at NC=7).
- **TMAX** — on-chip KV window (positions cached in fabric). Default 32 in RTL (sequencer_sb.sv:29; cohort_engine.sv:28; nl_engine.sv:32). **Built at 16** — the §26 wave dropped 32→16 to free 7 BRAM tiles for the per-cohort attention (WIDE-WORD-DATAPATH-LOG.md:964-969); build_bd_seq_sb.tcl passes `CONFIG.TMAX $tmax` (:78) with the driver default --tmax=16. Legal ≥ 1; larger = more context but more BRAM/URAM. The embed upload MUST match the build TMAX or pos>0 embeddings corrupt (WIDE-WORD-DATAPATH-LOG.md:967-969). Beyond TMAX, the kv_dma/kv_prefetch DDR context-restore path would take over (sim-only brick; not wired into any sequencer or bitstream).
- **ATT2** — attention topology. Default 1 (sequencer_sb.sv:43). **1** = one `vec_attn` per cohort (u_attn0/u_attn1, gnt==req, no arbiter) — the §26 "un-share" that kills the critical cohort's ~5.8k serial attention queue but costs ~+5.7k LUT and needs TMAX=16 BRAM (sequencer_sb.sv:154-179). **0** = the proven single shared `vec_attn` + hold-until-done arbiter (fixed priority cohort 0) (sequencer_sb.sv:180-212). **Bitstream builds set ATT2=0** for fit (build_bd_seq_sb.tcl:79 `CONFIG.ATT2 {0}`; comment sequencer_sb.sv:156-158). Legal {0,1}. The record §27 build ran ATT2=0.
- **DBG** — board-debug readback mux. Default 1 (sequencer_sb.sv:42; cohort_engine.sv:41). **1** (all sim gates) = the full per-phase `rd_data` mux tree the run_sb_* harnesses compare against seq_ref — the bit-honest path (cohort_engine.sv:613-654). **0** (bitstream builds) = mux tied off, ~1.5-2k LUT/device back; the record protocol reads only tok_outs+CYCLES (cohort_engine.sv:614-618, :655-657). Built at 0 (build_bd_seq_sb.tcl:79 `CONFIG.DBG {0}`). Legal {0,1}.

**The datapath-width params:**
- **P** — serial-lane / vector width (scratch elements per cycle). Default 8 (sequencer_sb.sv:23). P=8 is the measured sweet spot; P=8→16 is marginal because a ~40k 1-cycle floor dominates (WIDE-WORD-DATAPATH-LOG.md:31-34). Legal: power-of-two divisor of D (2/4/8/16). EROWS=D/P must be integer.
- **LANES** — GEMM PE width / URAM read-word lanes (INT4 weights/cycle). Default 128 (sequencer_sb.sv:24). §11 showed LANES=256 nearly halves GEMM cycles (WIDE-WORD-DATAPATH-LOG.md:388-393). Legal typ. 128/256; WBITS=LANES*4.

**The other params (all sequencer_sb.sv:20-44):**
- D=256 (model width), D3=768 (=3D, qkv), D_MLP=1024 (MLP hidden), VOCAB=193 (Kevin vocab), GAMMA_N=9 (LN gamma sets: 2 per block ×4 + 1 final), DQ_N=9409 (dequant-scale entries), NSACT=17 (inv-scale-act ROM entries), WWORDS=25600 (resident weight words), NLAYER=4, NHEAD=4, HEAD_DIM=64.
- Q-format fracs: RESID_FRAC=25 (residual Q6.25), LN_OUT_FRAC=22 (LN out Q.22), VFRAC=16 (q/k/v Q.16), GELU_FRAC=12 (GELU Q4.12), ISH=40 (act-quant reciprocal shift). These are the bit-true glue formats pinned in seq_ref.py:44-... and must match the reference.
- ND — DSP-packed GEMM streams per cohort. RTL default 0 (sequencer_sb.sv:27), but the SB record built ND=6/cohort (12 DSP-banked streams) — DERIVED from BD default + committed gate config + ~95% DSP occupancy; §18/§19 gates ran ND=6 with identical cycles to ND=0 (WIDE-WORD-DATAPATH-LOG.md:658-662). ND=0 as a *built* value was a misread of the RTL default.

## 6. How the two cohorts share the true-dual-port URAM

`weight_bank_tdp u_wbank` is the ONLY resource fundamentally split (sequencer_sb.sv:73-77):
- **Port B → cohort 0**: `raddr_b(waddr0)`, `rword_b(wword0)` (sequencer_sb.sv:76).
- **Port A → cohort 1**: `raddr_a(waddr1)`, `rword_a(wword1)` (sequencer_sb.sv:77). Port A also carries the boot loader write (`w_we/w_data`); the loader is idle at runtime so the address mux is clean (weight_bank_tdp.sv:6-9).
- Both ports return a REGISTERED read (1-cycle latency) = the pipeline stage-0 the cohort datapath expects (weight_bank_tdp.sv:11-12).
- On silicon this is `xpm_memory_tdpram` with `MEMORY_PRIMITIVE="ultra"` — HDL inference of two-address-port UltraRAM is dead in 2025.2 (falls to BRAM), so the XPM macro is mandatory (weight_bank_tdp.sv:14-22). In iverilog it's a behavioral 2-port array; the board run is the final bit-exactness check on the XPM side.

The point of split-brain: each cohort streams the SAME resident INT4 weight image through its OWN read port at its own layer/call, so the two halves' GEMM passes run truly in parallel instead of queuing behind one port (sequencer_sb.sv:7-11).

`embed_bank_tdp u_embank` shares the same trick for token embeddings — port B→coh0, port A→coh1, and `gnt==req` for both (no arbitration) so both nl_engines const-prop their embed-wait identically, avoiding a coh1/coh0 asymmetry (sequencer_sb.sv:79-118).

## 7. The three 2-way arbiters (shared LN / dq-gelu / optional attn)

All three are identical hold-until-done, fixed-priority-cohort-0 arbiters (a busy flag + owner flag; grant to req0 first, hold until the owner drops its req):
- **LayerNorm** (`ln_owner`/`ln_busy`, sequencer_sb.sv:128-140) muxes ln_start/vin/x/g by owner into the single `layernorm_vec u_ln` (:149-152). One shared LN serves both cohorts.
- **Dequant+GELU readback channel** (`dq_owner`/`dq_busy`, sequencer_sb.sv:238-250) — one `vec_dequant` (+DP 2nd lane u_dq2), one `vec_gelu` (+u_gelu2), and ONE copy of the dqm_w/dqe_w ROMs (:222-227) serve both cohorts. A cohort holds the channel for one whole GEMM call's readback so beats never interleave; replicating this cost ~12k LUT at N=16 (:216-219). ROM fetch is a free-running 2-stage pipeline on the granted cohort's row address (:252-270).
- **Attention** — only arbitrated when ATT2=0 (`at_owner`/`at_busy`, sequencer_sb.sv:182-194); when ATT2=1 each cohort has its own vec_attn and `at_gnt=at_req` (:168-169).

LN and attention never nest, so no deadlock (WIDE-WORD-DATAPATH-LOG.md:645-646).

## 8. Handshake glue and completion

- **nl_engine → GE**: `req` + descriptor `d_*` (nl_engine.sv:116-124); GE latches on `g_req` in GE_IDLE. GE acknowledges the whole call via `served` (=`g_done_p`, cohort_engine.sv:155), which clears `req` (`if (served) req<=0`, nl_engine.sv:385).
- **Stream-granular overlap**: `s_done`/`s_done_idx` pulse when a single stream's readback commits (cohort_engine.sv:343-345); nl_engine latches into `rb_rdy[s]` (:624-625) so per-stream post-loops (attn/RES1/RES2/argmax) start for stream s while GE still drains s+1.. (nl_engine.sv:51-55, :245-247). `rb_clr` clears rb_rdy at each new descriptor (:447 etc.).
- **Top-level done** (sequencer_sb.sv:376-386): `both_done = done_o0 && done_o1`; `done` pulses on the rising edge of both_done. `tok_outs = {tok_outs1, tok_outs0}` (:374). Host-debug `rd_data` is muxed by a registered arithmetic cohort select `rd_coh_d` (:391-393).

## 9. Cycle budget and the record (all from WIDE-WORD-DATAPATH-LOG.md)

The design has NO clean per-phase closed-form cycle budget in the RTL — the number is measured end-to-end per 16-token pass. The cycle-march:
- N=16 single engine: ~99,828–103,879 cyc/16-tok (the serial-URAM baseline, log:11,655).
- §19 first split-brain: 68,799 cyc/16 tok = 4,299 cyc/token, 1.51× (log:665-666).
- §22 stream-granular readback overlap: 66,285→61,245 (log:753).
- §23 vec_attn per-call cut: 61,245→57,149 (log:816).
- §25 CTX cross-group stream: 57,149→53,565 (log:897).
- §26 composed wave (TMAX 32→16 + per-cohort attn + CTX): **51,892 cyc/16 tok** SIM, 16/16 bit-exact (log:962-981).
- **§27 RECORD (MEASURED silicon)**: **53,364 cyc/16 tok** (sim 53,637 − 273 SETTLE), 16/16 bit-exact, 3/3 runs, ATT2=0 shared-attn build at 6.0ns → **49,971.3 tok/s @166.7 MHz / 59,965.5 tok/s @200 MHz** (log:983-990). 250 MHz still TIMEOUTs (hangs, not corrupt — needs 2.0× margin; 6.0ns gives ~1.6×).

The floor: a ~40k 1-cycle-per-element floor (GEMV act-feed + readback + attention/gamma loads) dominates once P≥8, which is why P=8 is the sweet spot and the remaining levers are cycle-schedule + clock, not width (log:31-34, CLAUDE.md build-order note).

The 100k identity: 16 streams × 250 MHz / 40k cyc. Streams maxed (N=16), cycles ~53k→~40k, and 250 MHz needs a post-route-MET 5ns build (silicon ~1.3× overclock covers it) — the open gate (CLAUDE.md).

**Parameters**

| name | meaning | live | range | where |
|---|---|---|---|---|
| `N` | Total streams per pass (two cohorts of NC) | 16 (the record) | even, >=2, = NC*2 | sequencer_sb.sv:25 |
| `NC` | Streams per cohort; passed as .N into each cohort_engine | 8 | >=1; fit variant 7 (N=14). Need not be power-of-2 (arithmetic cohort select) | sequencer_sb.sv:26 |
| `TMAX` | On-chip KV window (positions cached in fabric) | 16 in bitstream builds (RTL default 32) | >=1; embed upload MUST match build TMAX or pos>0 corrupts; beyond it -> kv_dma DDR restore would take over (sim-only brick, not wired into any bitstream) | sequencer_sb.sv:29 (default); build_bd_seq_sb.tcl:78 |
| `ATT2` | Attention topology: 1=vec_attn per cohort (un-share), 0=one shared+arbiter | 0 in bitstream builds (build_bd_seq_sb.tcl:79); RTL/sim default 1 | {0,1} | sequencer_sb.sv:43 |
| `DBG` | Board-debug per-phase readback mux (1=bit-honest sim path, 0=tied off for fit) | 0 in bitstream builds (build_bd_seq_sb.tcl:79); default 1 | {0,1} | sequencer_sb.sv:42, cohort_engine.sv:41 |
| `P` | Vector/serial-lane width (scratch elements per cycle) | 8 (the measured sweet spot) | power-of-2 divisor of D: 2/4/8/16 | sequencer_sb.sv:23 |
| `LANES` | GEMM PE width / URAM read-word lanes (INT4 weights/cycle); WBITS=LANES*4 | 128 | typ. 128/256 | sequencer_sb.sv:24 |
| `DP` | DOUBLE-PUMP-100K: MAC at 2 K-steps/clk, each read port serves raddr and raddr+1 | 0 (record); the _2/_b paths exist but idle | {0,1}; 1 requires clk2x on silicon | sequencer_sb.sv:44, cohort_engine.sv:42 |
| `ND` | DSP-packed GEMM streams per cohort (of the NC) | 0 (also built 6/cohort in §18/19, identical cycles) | 0..NC | sequencer_sb.sv:27 |
| `D / D3 / D_MLP` | Model width / qkv width (=3D) / MLP hidden | 256 / 768 / 1024 | model-fixed | sequencer_sb.sv:20-22 |
| `VOCAB / NLAYER / NHEAD / HEAD_DIM` | Kevin vocab / transformer blocks / attention heads / head dim | 193 / 4 / 4 / 64 | model-fixed | sequencer_sb.sv:28,34,35,36 |
| `RESID_FRAC / LN_OUT_FRAC / VFRAC / GELU_FRAC / ISH` | Pinned fixed-point fracs: residual Q6.25, LN-out Q.22, qkv Q.16, GELU Q4.12, act-quant reciprocal shift | 25 / 22 / 16 / 12 / 40 | must match seq_ref.py bit-true spec | sequencer_sb.sv:37-41 |
| `WWORDS / DQ_N / NSACT / GAMMA_N` | Resident weight words / dequant-scale entries / inv-scale-act ROM / LN gamma sets | 25600 / 9409 / 17 / 9 | export-derived | sequencer_sb.sv:33,31,32,30 |

**Key facts**

- The LIVE engine is sequencer_sb (split-brain top) instantiating two cohort_engine cohorts; each cohort wraps one nl_engine (phase FSM) + one gemm_cohort_vec + a local GE FSM (sequencer_sb.sv:316-372, cohort_engine.sv:148-212)
- There is NO monolithic phase-decode FSM in sequencer_sb: the transformer phase sequence lives in nl_engine's `nl` state machine (NL_IDLE..NL_DONE, 23 states) at nl_engine.sv:230-236,388-604
- GEMM execution lives in cohort_engine's `ge` FSM (GE_IDLE/GE_AQ/GE_WAIT/GE_RB/GE_RBN/GE_DQW; GE_AQN/GE_RUN retired) at cohort_engine.sv:288-290,393-611
- The five NL GEMM descriptors are QKV(768x256,dst0), PROJ(256x256,dst1), FC(1024x256,dst2/GELU), MP(256x1024,dst3), HEAD(193x256,dst4) (nl_engine.sv:444-546)
- The two cohorts share ONLY the weight image via true-dual-port URAM: cohort 0 on port B, cohort 1 on port A; the loader writes port A at boot then goes idle (sequencer_sb.sv:73-77, weight_bank_tdp.sv:6-9)
- TDP UltraRAM must be an xpm_memory_tdpram macro (MEMORY_PRIMITIVE=ultra) — HDL inference of two-address-port UltraRAM is dead in Vivado 2025.2 and falls back to BRAM (weight_bank_tdp.sv:14-22)
- Shared LN, shared dequant+GELU channel, and (ATT2=0) shared attention each use an identical hold-until-done fixed-priority-cohort-0 2-way arbiter (sequencer_sb.sv:128-140,238-250,182-194)
- Bitstream builds set DBG=0 (drop the readback mux, ~1.5-2k LUT back) and ATT2=0 (shared attention, fits today); sim gates run DBG=1 ATT2=1 (build_bd_seq_sb.tcl:79, sequencer_sb.sv:42-43,156-158)
- MEASURED record: 53,364 silicon cyc / 16-token pass, 16/16 bit-exact, 3/3 runs -> 49,971.3 tok/s @166.7 / 59,965.5 tok/s @200 MHz; 250 MHz still TIMEOUTs (WIDE-WORD-DATAPATH-LOG.md:983-990)
- Cohort host-debug select is ARITHMETIC (rd_stream>=NC, rd_stream-NC) not a bit-slice, specifically so NC need not be a power of two (NC=7/N=14 fit variant) (sequencer_sb.sv:310-312,388-390)
- P=8 is the measured sweet spot: a ~40k 1-cycle-per-element floor (act-feed + readback + attn/gamma loads) dominates above it, so remaining levers are schedule+clock not width (WIDE-WORD-DATAPATH-LOG.md:31-34)
- The stream-granular NL/GEMM overlap uses s_done/rb_rdy[s] so per-stream post-loops (attn/RES1/RES2/argmax) start while GE still drains later streams (nl_engine.sv:51-55,624-625; cohort_engine.sv:343-345)
- cohort_engine was mechanically derived from sequencer_pp by collapsing two stream-groups to one and deleting merge/GWAIT/aq_eng (a cohort never shares a weight pass) (cohort_engine.sv:9-12)
- DOUBLE-PUMP (DP) logic (_2/_b duplicate datapaths) is present throughout GE_AQ/GE_RB but DP defaults to 0 and the record ran DP=0 (cohort_engine.sv:42, run_sb_seq.py:138)

**Files**

- `fabric/stage3/rtl/sequencer_sb.sv` — LIVE split-brain top: two cohort_engine cohorts + shared weight/embed URAM + 2-way arbiters for LN, dequant+GELU, and (ATT2=0) attention
- `fabric/stage3/rtl/cohort_engine.sv` — LIVE single cohort: instantiates one nl_engine + one gemm_cohort_vec; contains the GE (GEMM-executor) FSM and the act-quant / readback / DP datapath
- `fabric/stage3/rtl/nl_engine.sv` — LIVE non-linear phase-decode FSM (NL_* states): embed/LN/attention/residual/argmax + GEMM-call descriptor generation; the actual per-token phase sequencer
- `fabric/stage3/rtl/weight_bank_tdp.sv` — Shared resident INT4 weight image on true-dual-port URAM (xpm macro on silicon); the only resource split-brain fundamentally shares
- `fabric/stage3/rtl/embed_bank_tdp.sv` — Shared token-embedding URAM with a real read port per cohort (gnt==req, no arbitration)
- `fabric/stage3/rtl/sequencer_vec.sv` — HISTORICAL P-wide vector datapath (wide-word banking + BRAM sync-read); the P=8 core the split-brain cohort descends from
- `fabric/stage3/rtl/sequencer_pp.sv` — HISTORICAL N=8 ping-pong two-group sequencer; direct parent of cohort_engine, birthed nl_engine
- `fabric/stage3/rtl/sequencer_gemm.sv` — HISTORICAL N-stream batched (GEMV->GEMM, one weight pass serves N)
- `fabric/stage3/rtl/sequencer_fast.sv` — HISTORICAL P=1 single-stream KV-decode baseline (~166k cyc/token)
- `fabric/stage3/rtl/sequencer.sv` — HISTORICAL original Tier-3 single-stream decode FSM (the canonical dataflow comment)
- `fabric/stage3/seq_ref.py` — Per-phase bit-true reference (block0_phase_signals / full_forward_signals) the RTL is gated against; pins the Q-format glue formats
- `fabric/stage3/WIDE-WORD-DATAPATH-LOG.md` — The cycle-march + MEASURED record log; §19-27 = split-brain campaign, §27 = the 59,965.5 tok/s record
- `fabric/stage3/tcl/build_bd_seq_sb.tcl` — Bitstream BD build for sequencer_sb; sets CONFIG.DBG 0 / CONFIG.ATT2 0 and passes P/LANES/N/NC/ND/TMAX
- `fabric/stage3/tcl/ooc_seq_sb.tcl` — OOC fit/Fmax check for sequencer_sb; documents the tclargs (P LANES PERIOD WWORDS TMAX ND NC ATT2)
- `fabric/stage3/run_sb_seq.py` — The split-brain full-forward gate harness (SEQ_SB_FULL verdict); shows live default args --p 8 --lanes 128 --tmax 32 --att2 1 --dp 0

**Gotchas**

- No closed-form per-phase cycle budget exists in the RTL — cycle counts are measured end-to-end per 16-token pass and only exist as the log's cycle-march numbers. Do NOT quote a per-phase cycle count as if the RTL guarantees it.
- sequencer_sb has NO phase FSM of its own — reviewers looking for the LN1/QKV/attn decode in sequencer_sb.sv will not find it; it is in nl_engine.sv (NL_* states) inside each cohort. sequencer_sb is only shared resources + arbiters + two cohort instances.
- DBG and ATT2 differ between sim and bitstream: sim gates run DBG=1/ATT2=1 (bit-honest per-phase readback, per-cohort attention), but every SILICON number was measured with DBG=0/ATT2=0. The compute datapath is identical; only the readback mux and attention topology change.
- TMAX default is 32 in the RTL but bitstreams build TMAX=16. The board embed upload MUST match the build TMAX (upload 16 pos_emb rows) or pos>0 embeddings corrupt (WIDE-WORD-DATAPATH-LOG.md:967-969).
- The DOUBLE-PUMP (_2 / _b) datapath is pervasive in cohort_engine but DP defaults to 0 and every gated/measured result ran DP=0. Reading the _b registers as active logic will mislead — at DP=0 the 2nd read is a same-address behavioral model.
- The cohort host-debug select is arithmetic (rd_stream>=NC), NOT a high-bit slice — a slice broke at NC=7 because global stream 7 is cohort-1-local 0. Preserve the arithmetic form if changing NC.
- True-dual-port UltraRAM CANNOT be HDL-inferred in Vivado 2025.2 (it falls to BRAM); weight_bank_tdp/embed_bank_tdp rely on the xpm_memory_tdpram macro under `ifdef SYNTHESIS. The iverilog behavioral 2-port array only proves the sim side; the board run is the final TDP bit-exactness check.
- Silicon runs ~1.3-1.76x faster than STA suggests: the 6.0ns record build (~166 MHz STA) runs bit-exact at 200 MHz on silicon; 250 MHz still hangs. Do not conclude 250 MHz is reached from an OOC MET-at-5ns number alone — it needs a post-route-MET 5ns build that also ROUTES at BD density.

**Open questions**

- Exact per-phase cycle attribution within the 53,364-cycle pass is not decomposed in the RTL or the log — I could determine the total and the deltas from each lever (§22-26) but not a clean phase-by-phase breakdown (e.g. how many of the 53k cycles are GEMM run vs. attention vs. argmax).
- **RESOLVED** (§7 / dead-lever ledger): the full DP=1 build was never board-run (OOC-timing-MET only); a MAC-only DP branch (`dp-hw-maconly`) WAS run on silicon bit-exact, but the fabric→clk2x data feed walled at 50 MHz — that measurement killed the lever (100K-REVIEW.md:44).
- The precise LUT/BRAM/DSP fit numbers for the ATT2=1 (per-cohort attention) config at BD level — the RTL comment says it is ~1.9k LUT over the device (sequencer_sb.sv:156-158) but I did not open an OOC/impl report to confirm the current margin.
- How completion (`done_o`) interacts with the stream-granular overlap under skew between cohorts on real silicon beyond the 3/3 record runs — the log notes an earlier tb_gemm_sb latch bug from one-cycle done pulses under deliberate skew (WIDE-WORD-DATAPATH-LOG.md:667-673), but that was a harness bug, not the sequencer_sb both_done edge-detect (sequencer_sb.sv:376-386), which I did not independently stress-check.

---

## GEMM/GEMV Datapath, DSP Packing & Weight Residency (fabric/stage3)

This subsystem is the matrix-multiply heart of the Kevin-on-Kria fabric: it streams the model's INT4 weights out of resident URAM once per token and multiplies each wide weight word against N activation streams, producing INT32 accumulator outputs the sequencer dequantises. Every matmul in the transformer (QKV projection, attention output projection, MLP up/down, the final head) is one call into this datapath at a per-layer weight-base offset. The core is "banked resident": the whole INT4 weight image lives in on-chip UltraRAM (never DDR — that is the bandwidth-wall thesis), packed transposed into wide words of LANES nibbles so one URAM read delivers LANES INT4 weights per cycle.

The datapath has evolved through a clear lineage of RTL files, all sharing the same arithmetic and memory shape: gemv_banked_resident_vec.sv (single stream, the proven base), gemm_banked_resident_vec.sv (N batch streams, self-contained URAM), gemm_cohort_vec.sv (one split-brain cohort with the URAM banks hoisted out to a shared weight_bank_tdp), and gemm_split_brain.sv (two cohorts on one true-dual-port URAM image — the live engine wrapped by sequencer_sb via cohort_engine). Two MAC-leaf styles exist: LUT-based mac_bank (INT4xINT8 per lane, use_dsp=no) and DSP-packed mac_bank_dsp (two INT4 weights sharing one INT8 activation in one DSP48E2, WP487-style). Double-pump twins (mac_bank_dp / mac_bank_dsp_dp) run the accumulator in a 2x clock domain for the 100k campaign.

The single load-bearing structural decision is the wide-word banking layout: `reg [P*W-1:0] buf [0:ROWS-1]` with lane l in bits [l*W +: W], NOT `reg [W] buf [0:P-1][0:ROWS-1]`. The naive [P][rows] shape read at a variable row synthesises to per-lane row muxes that blew the KV260's LUT/MUXF7 budget to 185%/151% (DRC failed before placement). The wide-word shape makes the variable row a memory address (free) and leaves only a small constant lane select. This is documented as a hard-won gotcha and applies to every scratch buffer, not just weights.

Weight residency is 72-bit-wide URAM banks (the URAM-native width), because a single 1024-bit-wide memory pads to 16 URAM wide x 4 cascade = 64 = the whole device and Vivado marks the ultra attribute infeasible, silently falling back to ~400k LUTRAM. Weights are boot-STREAMED (not baked into the bitstream): a 32-bit chunk loader assembles SUBW=WBITS/32 chunks into one wide word and commits all banks in one shot; the on-board driver preloads every layer's words once at boot and records the per-layer w_base. The DSP-packing ceiling is proven exactly 2.0 INT4xINT8 MACs/DSP (dsp3_pack_proof.py: 3/DSP is arithmetically impossible on the 27x18 multiplier), capping DSP-batch streams at N=16.

## 1. Module lineage and what is live

| File | Role | Weights | Streams |
|---|---|---|---|
| `gemv_banked_resident_vec.sv` | single-stream base (P-wide boundary) | `weight_bank_tdp` instance, in-core | 1 |
| `gemm_banked_resident_vec.sv` | N batch streams, **self-contained** URAM banks (inline `g_w` generate) + defines the `mac_bank`/`mac_bank_dsp` leaves | in-core `(* ram_style="ultra" *)` | N |
| `gemm_cohort_vec.sv` | ONE split-brain cohort: same FSM/MAC/readback as `gemm_banked_resident_vec` but URAM banks **hoisted out**; drives `waddr`, consumes `wword_rd` (1-cyc latency) | external (shared bank) | N (per cohort) |
| `gemm_split_brain.sv` | two `gemm_cohort_vec` + one shared `weight_bank_tdp` (cohort0 on port B, cohort1 on port A) | shared TDP URAM | 2xN |
| `gemm_dsp_resident_vec.sv` | all-DSP variant (older 24-bit-gap encoding, in-core recovery) | in-core | N |

**The LIVE engine is the split-brain path**, reached through `sequencer_sb` -> `cohort_engine` -> `gemm_cohort_vec`, all sharing one `weight_bank_tdp`. The `sequencer_sb` RTL *defaults* are `LANES=128, P=8, N=16 (NC=8 per cohort), ND=0, WWORDS=25600, TMAX=32, DP=0` (fabric/stage3/rtl/sequencer_sb.sv:23-44); the record bitstream overrides TMAX=16 and ND=6/cohort (DP=0 for the record; DP=1 is the OOC-only 100k lever). `cohort_engine` instantiates `gemm_cohort_vec #(.LANES,.N,.ND,.P,.MMAX(1024),...)` and passes the shared `.waddr/.wword_rd/.wword1_rd` (fabric/stage3/rtl/cohort_engine.sv:202-212). The MEASURED record 59,965.5 tok/s @200 MHz (16/16 bit-exact) is this split-brain design (WIDE-WORD-DATAPATH-LOG.md:990; CLAUDE.md).

The `mac_bank` and `mac_bank_dsp` module DEFINITIONS live at the bottom of `gemm_banked_resident_vec.sv:474-557` — `gemm_cohort_vec` does NOT redefine them, it reuses them (fabric/stage3/rtl/gemm_cohort_vec.sv:12-14).

---

## 2. The wide-word banking layout (THE load-bearing gotcha)

**Correct layout** — one row-addressed wide word per buffer:
```
reg [P*W-1:0] buf [0:ROWS-1]      // lane l lives in bits [l*W +: W]
```
The variable row is a memory ADDRESS (free); only a small constant-offset lane select remains (WIDE-WORD-DATAPATH-LOG.md:87-91).

**Wrong layout** — the obvious `reg [W] buf [0:P-1][0:ROWS-1]`. Read/written at a variable row it synthesises to a giant per-lane row multiplexer. The full P=8/L=128 build hit:
```
ERROR [DRC UTLZ-1]: LUT as Logic 216,309 / 117,120 (185%)
                    MUXF7        88,368 / 58,560  (151%)
```
with ~10 variable-index sites across the datapath — DRC failed before placement (WIDE-WORD-DATAPATH-LOG.md:71-81). The fix cut it to a fitting design (P=4/L=128: 106,009 LUT / 90.5%, WIDE-WORD-DATAPATH-LOG.md:117). Key secondary finding: **LANES is not the lever, P is** — L=64 saved only ~6k LUT (resident GEMV ~13k LUT, ~100 LUT/lane); the P-wide datapath was the ~114k bulk (WIDE-WORD-DATAPATH-LOG.md:119-121).

In the GEMV core this shows in:
- `xmem [0:XROWS-1]` holding `P*8` bits/row (P INT8 acts, lane l at `[l*8 +: 8]`) — gemv_banked_resident_vec.sv:90.
- `ymem [0:GROUPS-1]` holding `YBITS = LANES*32` (or `LANES*ABITS`) — one group word = LANES INT32 outputs (gemv_banked_resident_vec.sv:69,91).
- Weight ROM `mem [0:WWORDS-1]` at `BANKW` bits/bank (gemm_banked_resident_vec.sv:134).

**iverilog trap that rides on this**: a variable `+:` part-select on an *element of an unpacked array* reads X — you may only variable-part-select a **plain reg/wire**. So the code always copies the array element to a plain wire first (`wire [P*8-1:0] xrow = xr[RLAT-1]; wire signed [7:0] xsel = xrow[xl0*8 +: 8];` — gemm_banked_resident_vec.sv:225-227; gemm_cohort_vec.sv:159-161). Also: `wsel` must be a WIRE, not an `always@*` copy — an always@* copy made Vivado trim the URAM read register to 4 bits and fall back to ~400k LUTRAM (gemm_banked_resident_vec.sv:186-189).

---

## 3. Parameters (the design knobs)

From the module headers (gemm_banked_resident_vec.sv:22-43, gemm_cohort_vec.sv:18-33) and live values in sequencer_sb.sv:20-44:

| Param | Meaning | Live value | Constraints |
|---|---|---|---|
| `LANES` | PE lanes = INT4 nibbles per wide URAM word (pow2) | 128 | pow2; `WBITS=LANES*4` |
| `P` | boundary width: INT8 acts/write in, INT32 outs/read | 8 | P divides LANES, KMAX, MMAX |
| `N` | batch streams (per cohort in split-brain) | 8 (NC) | 16 total across 2 cohorts |
| `ND` | of which DSP-packed (streams N-ND..N-1) | 0 | rest are LUT `mac_bank` |
| `MMAX` | max output rows of any layer | 1024 | — |
| `KMAX` | max reduction length of any layer | 1024 | — |
| `WWORDS` | resident capacity in wide words | 25600 | image is ~12.6 Mbit fixed; scales inversely with LANES (12,800 words at L=256) |
| `RLAT` | read->mac pipeline depth (cycles) | 2 | stage 0 = URAM read reg |
| `DP` | double-pump: 2 K-steps/clk in clk2x domain | 0 | ND=0 only (until DSP leaf DP lands) |
| `ABITS` | internal accumulator width per lane | 24 | `|acc| <= 8*128*1024 = 2^20`, 24b holds it with 3 bits margin (gemm_banked_resident_vec.sv:40-43) |
| `K2` | (gemv only) 2 K-steps/clk via 2nd URAM read port | 0 | free at N=1 |

Derived locals: `WBITS=LANES*4`, `YBITS=LANES*ABITS` (or `LANES*32` in the gemv/dsp cores), `GROUPS=ceil(MMAX/LANES)`, `XROWS=KMAX/P`, `PPG=LANES/P` (P-groups per ymem word), `SUBW=WBITS/32` (32-bit load chunks per wide word). URAM banking locals: `BANKW=(WBITS>512)?72:WBITS`, `NB=ceil(WBITS/BANKW)`, `WPAD=NB*BANKW` (gemm_banked_resident_vec.sv:68-81).

---

## 4. Weight packing (transposed wide-word) — pack_banked_resident.py

The whole model's transposed INT4 weights live in one big URAM image; a per-layer `w_base` WORD offset selects the layer (fabric/stage3/pack_banked_resident.py:1-20).

Word addressing (pack_banked_resident.py:8-13):
```
word(layer, g, k) = w_base[layer] + g*K_layer + k          # word units
word(g,k) bit [L*4 +: 4] = nibble( W[g*LANES + L, k] )      # L = 0..LANES-1
words_per_layer(M,K) = ceil(M/LANES) * K
w_base[0]=0 ; w_base[i] = w_base[i-1] + words_per_layer(M_{i-1},K_{i-1})
```
So one wide word is a single **transposed column** (fixed k) of LANES consecutive output rows within group g. `pack_transposed` zero-pads rows up to a multiple of LANES (padded lanes contribute 0 and are never read since `rd_addr < M`) and emits, per group g, per column k, a word with lane L in bits `[L*4 +: 4]` = `col[L] & 0xF` (pack_banked_resident.py:41-62). Bit-exact contract: `y[m] = sum_k W[m,k]*x[k]`, signed INT4 weights `[-8,7]`, signed INT8 acts `[-128,127]`, exact int32 accumulate (pack_banked_resident.py:15-16, 31-33). The `.mem` file is written `lanes` hex chars per word (4 bits/lane) (pack_banked_resident.py:101-103). The gate `pack_banked_resident.py check` runs >=2 distinct layer shapes through the same resident core at their w_base offsets and prints `BANKED_RESIDENT_VERDICT bitexact=... mismatches=...` (pack_banked_resident.py:118-147).

---

## 5. Weight residency: boot-streamed, in 72-bit URAM banks

**Boot-STREAMED, not baked.** Weights arrive at runtime through a 32-bit chunk loader, not `$readmemh` in the bitstream. The on-board driver preloads all layers' words once, records w_base per layer, then per matmul writes W_BASE+M+K, streams x, runs (pack_banked_resident.py:18-20).

The loader/assembler (gemm_banked_resident_vec.sv:97-118, weight_bank_tdp.sv:65-81): `w_we` presents one 32-bit `w_data`; `wnext = wbuf | (w_data << (wsub*32))` accumulates `SUBW` chunks into a `WBITS`-wide word; on `wcommit = w_we && (wsub==SUBW-1)` all `NB` banks are written at address `wword` in one shot from `wnext_pad`, and `wword` increments. `ld_rst` rewinds the pointers.

**URAM banking is 72b-wide by geometry constraint** (gemm_banked_resident_vec.sv:79-88, WIDE-WORD-DATAPATH-LOG.md:396-403). A URAM is 4096 x 72b; Vivado pads each memory to a multiple of 72b x 4096. A single 1024-bit x 12800 memory pads to 16 URAM wide x 4 cascade = 64 = the whole device -> the `ultra` attribute is marked infeasible -> silent fallback to ~400k LUTRAM (342% at L=256). Even two 512b banks pad to 64. The dense fix: `BANKW=(WBITS>512)?72:WBITS` — 72b-native banks, 15 banks x 4 cascade = 60 URAM at LANES=256. For LANES<=128 the single 512b x 25600 memory (WBITS=512, one bank) is kept (56 URAM). MEASURED fit at L=256: 87,761 LUT (74.9%), BRAM 143.5/144, URAM 60/64, DSP 505 (WIDE-WORD-DATAPATH-LOG.md:400-401). The 12.6 Mbit image needs ≥45.5 URAM so 60 is within 1.32× of the floor (the log's figures, WIDE-WORD-DATAPATH-LOG.md:402; note the raw ratio 12.6 Mbit / 288 Kbit ≈ 43.75 — the log's 45.5 folds in cascade/padding overhead).

Each bank generate (gemm_banked_resident_vec.sv:131-147):
```
(* ram_style = "ultra" *) reg [BANKW-1:0] mem [0:WWORDS-1];
reg [BANKW-1:0] rd;
always @(posedge clk) begin
  if (wcommit) mem[wword] <= wnext_pad[gb*BANKW +: BANKW];
  rd <= mem[waddr];            // registered read = pipeline stage 0
end
```
The registered read (1-cycle latency) IS pipeline stage 0. DP=1 adds a 2nd read `rd1 <= mem[waddr1]` (gemm_banked_resident_vec.sv:141-145).

**The 1,024-multiplier URAM cliff** (the reason for `keep_hierarchy` MAC leaves): at N=8 the core holds 1,024 4x8 MACs, and Vivado 2025.2's bulk multiplier optimization (runs after DSP absorption, before RAM mapping) restructures them and detaches the URAM read register's loads — the mapper sees a dead read port and refuses `ultra`, dumping weights into ~29,600 RAM64M8 (~450k LUT). N=4 (512 mults) never trips it. **Fix: each stream's 128 MACs is one `(* keep_hierarchy="yes" *)` leaf** — an opaque boundary the bulk pass cannot cross; URAM 64/64 restored (WIDE-WORD-DATAPATH-LOG.md:529-535; gemm_banked_resident_vec.sv:465-474).

---

## 6. weight_bank_tdp — the shared dual-port image (the split-brain foundation)

The ONLY resource split-brain shares (fabric/stage3/rtl/weight_bank_tdp.sv:1-11). One true-dual-port URAM image read by both cohorts at once:
- **Port A**: loader WRITES at boot (`we_a`) + cohort-1 READS at runtime (loader idle after boot, so `aaddr = wcommit ? wword : raddr_a` is a clean mux — weight_bank_tdp.sv:84).
- **Port B**: cohort-0 READS (unchanged single-read timing).

Both ports return a REGISTERED read = 1-cycle latency = pipeline stage 0 the cohort expects (weight_bank_tdp.sv:8-12).

**Dual-dialect pattern** (weight_bank_tdp.sv:14-25): `` `ifdef SYNTHESIS `` -> `xpm_memory_tdpram` with `MEMORY_PRIMITIVE="ultra"` (the PROVEN mapping: URAM 56 / 0 LUT / 0 BRAM, both ports independent); `else` (iverilog) -> a behavioral 2-port array. **HDL inference of TDP UltraRAM is DEAD in 2025.2** — two-address-port templates fall to BRAM ("invalid write mode" for every variant); do NOT try to infer it. Gates verify the behavioral side; the board verifies the XPM side. TDP UltraRAM REQUIRES `WRITE_MODE="no_change"` on both ports (read_first is a synth error) (weight_bank_tdp.sv:129, embed_bank_tdp.sv:103-105).

`embed_bank_tdp.sv` is the analogous shared token/pos embedding image (same TDP dual-dialect pattern; URAM in the single-pump record, BRAM under DP=1 to free URAM). It exists to fix a coh0/coh1 const-prop asymmetry — a fixed-priority single-port arbiter let Vivado const-prop coh0's embed-wait away while coh1 paid full freight; giving both a real port makes both nl_engines simplify symmetrically (embed_bank_tdp.sv:4-13). Also note the gemv core steals the idle port-B for embed reads while its FSM is IDLE (emb_sel path, gemv_banked_resident_vec.sv:56-66,128-129).

---

## 7. The DSP packing scheme (2.0 MACs/DSP — the proven ceiling)

**WP487 weight-share, adapted INT4 weight x INT8 activation.** Two +8-biased INT4 weights at a 22-BIT gap share one INT8 activation in ONE DSP48E2 (gemm_banked_resident_vec.sv:500-557, `mac_bank_dsp`):
```
w0 = w[gl*4 +: 4]     ^ 4'h8;      // +8 bias -> unsigned 0..15
w1 = w[(gl+1)*4 +: 4] ^ 4'h8;
wpak = {w1, 18'b0, w0};            // 26-bit, < 2^26: fits one DSP's 27-bit signed port
a <= a + $signed({1'b0, wpak}) * xs;   // 48-bit accumulator, xs = sign-extended INT8
sum_act <= sum_act + xs;           // per-stream +8-bias correction term
```
The gap is 22, NOT 24: `wpak` must stay under 2^26 so the signed-extended operand fits the 27-bit port — at a 24-bit gap (28-bit wpak) Vivado cascaded TWO DSPs per multiply (gemm_banked_resident_vec.sv:504-507). The 22-bit lower field exactly holds K<=1024 of 12-bit products (12+10=22), and `|y0| < 2^21` keeps the mod-2^22 window unambiguous.

**RAW-pair encoding**: the leaf outputs UNRECOVERED 48-bit pair accumulators packed in the acc bus (pair p at `acc[p*48 +: 48]`); `(LANES/2)*48 == LANES*ABITS` so the bus width is unchanged, only the ENCODING (gemm_banked_resident_vec.sv:509-514,552-554). The ~5.4k LUT/bank recovery is hoisted to the parent's P-wide readback path (4 pairs/cycle) using the exported `sum_act` (gemm_banked_resident_vec.sv:437-461). Exact integer recovery:
```
y0    = (acc[21:0] - 8*sum_act) mod 2^22      // sign-extended to 32b
carry = (y0 + 8*sum_act) >>> 22
y1    = acc[47:22] - 8*sum_act - carry
```
(gemm_banked_resident_vec.sv:450-459). The older `gemm_dsp_resident_vec.sv` uses a **24-bit gap** with a 28-bit `wpak = {w1,20'b0,w0}` and in-core recovery latched at SETTLE (gemm_dsp_resident_vec.sv:160-186) — the newer 22-bit-gap RAW-pair form in `mac_bank_dsp` superseded it.

**MACs/DSP is exactly 2.0, proven** (fabric/stage3/research/dsp3_pack_proof.py; DSP-PACKED-GEMM.md:192-286). 3-per-DSP (shared activation) is IMPOSSIBLE on the 27x18 multiplier, by two independent walls:
- **Wall 1 (operand port)**: a single int4xint8 product is 12 bits, so no-bleed needs gap `g>=12`; but three nibbles at gap g need `2g+4` bits and the 27-bit signed operand allows `15*(1+2^g+2^2g) < 2^26 -> g<=11`. `11 < 12` -> no gap works, even at K=1. Concrete demo at g=11: field0 readback=128 vs intended -1920 (bled) (dsp3_pack_proof.py:265-288).
- **Wall 2 (accumulator info)**: three distinct 22-bit neuron sums carry 66 bits; one 48-bit acc holds 48. `66 > 48` -> not separable at full K, and the drain-window escape collapses (`D=0`) because Wall 1 caps the usable field at 11 bits (dsp3_pack_proof.py:146-173).

Proof run: `DSP3_PROOF ... result=IMPOSSIBLE(operand27<28 & info66>48) fallback=2.0-per-DSP(22b-gap,debias) mismatches=0/610756 OK` (DSP-PACKED-GEMM.md:256-263). **Stream budget**: at 2/DSP, 128 lanes = 64 DSP/stream -> 1024/64 = **N=16** hard ceiling on 1,024 DSPs at LANES=128 (dsp3_pack_proof.py:381-385; DSP-PACKED-GEMM.md:276-286). The LUT vs DSP tradeoff is deliberate: a 4x8 mult is ~15 fabric LUTs, so cheap mults stay in fabric and DSPs are reserved for the expensive 32x32-class LN/attention/dequant multipliers — at N=16 the auto-absorbed DSPs overflowed (1,304 > 1,248) and evicted those to fabric (gemm_banked_resident_vec.sv:465-473).

---

## 8. Datapath dataflow (FSM phases)

RUN FSM states: `IDLE, RUN, DRAIN, FIN, SETTLE` (gemm_banked_resident_vec.sv:150; gemv uses `IDLE, RUN, FIN` only, gemv_banked_resident_vec.sv:117).

**Per matmul call** (`start` pulse with m_count/k_count/w_base):
1. **IDLE->RUN**: clears accumulators (`acc_clr`/`accb<=0`), sets `grp_base <= w_base`, `g=kc=kmac=0` (gemm_banked_resident_vec.sv:312-321).
2. **RUN** — the K-walk, once per group g (of `gcount = ceil(m_count/LANES)` groups):
   - `issue = (kc < k_count)`; each issued cycle: read weight word at `waddr = grp_base + kc`, read act row `xmem[kc >> LSHP]`, latch lane index `kc[LSHP-1:0]`; `kc += STEP` (STEP=2 under DP else 1) (gemm_banked_resident_vec.sv:322-326,160-165).
   - Pipeline depth RLAT=2: stage 0 = URAM read reg; the weight word / act row / lane index / valid flow through `word_p`, `xr[]`, `xl_p`, `v_p` (gemm_banked_resident_vec.sv:288-302).
   - MAC stage (`mac_v = v_p[RLAT-1]`): each of N `mac_bank`/`mac_bank_dsp` leaves does LANES INT4xINT8 products into its per-lane ABITS accumulator; `kmac += (mac_v1?2:1)` (gemm_banked_resident_vec.sv:322-326, mac_bank at :488-493).
   - When `kmac == k_count`: group g's reduction is complete -> `DRAIN` (synth) or `SETTLE` (sim) (gemm_banked_resident_vec.sv:327-335).
3. **SETTLE (sim only)**: waits 4 cycles for the MAC pipeline tail, latches per-stream `acc_str[b] -> y_lat[b]` for constant-index unroll. Synth skips it (goes RUN->DRAIN), so silicon finishes each call ~4 cyc earlier than sim -> sim cyc/tok is an upper bound (~0.8%) (gemm_banked_resident_vec.sv:337-348,327-332).
4. **DRAIN**: one stream's group word per cycle. `ymem[db*GROUPS + g] <= acc_sel` (and `sumact_mem[...]` for DSP streams). Under SYNTHESIS the stream select is a chunked `<=4-stream` case + a 4:1 final select (`sel_q0..3` -> `acc_sel`) — a flat N*YBITS concat wraps iverilog's 16-bit part-select index space at stream 4 / bit 16384, so streams are kept in named <=12,288-bit chunks (gemm_banked_resident_vec.sv:190-204,349-393). When `db == N-1`: `acc_clr`; if `g == gcount-1` -> FIN, else advance to next group (`g++`, `kc=kmac=0`, `grp_base += k_count`, back to RUN) (gemm_banked_resident_vec.sv:394-404).
5. **FIN**: `done <= 1`, back to IDLE (gemm_banked_resident_vec.sv:405).

**Boundary I/O** (the P-wide interface that let the sequencer's act-quant feed in ceil(K/P) and drain in ceil(M/P), saving ~16k of the ~50k cyc/token — gemv_banked_resident_vec.sv:6-8):
- Act feed: `x_we` writes P INT8 lanes/cycle into stream `x_stream` at row `x_row` (cohort core takes an explicit row for row-major AQ; base core auto-increments `xptr`) (gemm_cohort_vec.sv:44-54,153-155; gemm_banked_resident_vec.sv:105-118).
- Readback: `rd_addr` is a P-group index; `y_out = rd_word[rd_off*(P*32) +: P*32]` returns P consecutive INT32 outputs, 2-cycle latency. `rd_word = ymem[rd_stream*GROUPS + rd_addr/PPG]`, `rd_off = rd_addr % PPG` (gemm_banked_resident_vec.sv:411-461; gemv:259-267). Cohort core adds `y_out2` (rd_addr+1, same group word, free 2nd P-group) for 2-rows/clk dequant under DP (gemm_cohort_vec.sv:69-72,324-333).

**gemm_cohort_vec extras**: an AQ/RUN overlap stall guard — `rows_committed` bumps on `x_rowcommit`; `row_ready = !ovl_en || (xrd_row < rows_committed)` gates `issue` so an under-run inserts MAC bubbles (bit-identical, just delayed) (gemm_cohort_vec.sv:58-100,116-118). A 2nd act write port (`x_we2/x_stream2/x_row2/x_data2`) lets the AQ write two streams/clk (independent per-stream xm arrays) (gemm_cohort_vec.sv:47-54,154-155). In `gemm_split_brain` these overlap/2nd-port inputs are tied off (`ovl_en=0`, `x_we2=0`) (gemm_split_brain.sv:80-93).

**Per-stream act memories are separate 1W/1R SDPs**, declared inside the `g_mac` generate — ONE xmem with N read ports made Vivado replicate it at N=8 and refuse outright at N=16 ("131072 bits too large") (gemm_banked_resident_vec.sv:83-89,214-221).

---

## 9. Double-pump (DP=1) — the 100k campaign lever

`mac_bank_dp` (LUT) and `mac_bank_dsp_dp` (DSP) run the accumulator in a 2x, 0-degree-aligned `clk2x` domain, consuming TWO K-steps (2j and 2j+1) per `clk` — bit-exact because it's a pure timing transform: same K products, same order, same integer adds (mac_bank_dp.sv:1-60; mac_bank_dsp_dp.sv:1-34). Phase anchoring: `phase = ~clk` sampled at the clk2x edge (clk high -> phase 0 uses w0/x0; clk low -> phase 1 uses w1/x1) — a pure level sample, no coincident-edge race, deterministic from t=0 (mac_bank_dp.sv:30-42,77-95). `weight_bank_tdp` under DP=1 uses a COLUMN-PARITY split: even columns in `mem_e`, odd in `mem_o`, each half-depth (shorter cascade closes 5ns); one read at `raddr>>1` returns both K-steps of a pair (grp_base and k_count are always even). The URAM stays in the slow clk domain (its deep read cascade cannot run at 400 MHz — OOC-proven dead); only the MAC accumulator runs at clk2x (weight_bank_tdp.sv:96-113,195-211). DP=1 is ND=0-only until the DSP leaf's own double-pump lands (gemm_banked_resident_vec.sv:34-38). Live design runs DP=0.

---

## 10. Performance anchors (all cite the log)

- Single stream: **11,143.9 tok/s @200 MHz, 3/3 bit-exact, CYCLES=17,947** (MEASURED, WIDE-WORD-DATAPATH-LOG.md:445-449).
- Cycle floor: GEMV reads **12,800** wide URAM words/token (the irreducible cost of streaming the whole INT4 image once); 12.8k -> 200 MHz = 15.6k single-stream ceiling (WIDE-WORD-DATAPATH-LOG.md:455; DSP-PACKED-GEMM.md:8-19).
- Batch amortises those SAME 12.8k reads across N streams (read once, fan out in the activation dimension; URAM bandwidth is independent of N) (DSP-PACKED-GEMM.md:141-157).
- N=4: 16,969.3 tok/s (MEASURED); N=8: 19,275.6 @166.7 (MEASURED); N=16: 24,134.0 @166.7 then 46,604.4 @200 (first 200-clean) (WIDE-WORD-DATAPATH-LOG.md:480,560-564,580-581,721-753).
- Split-brain N=16 record: **59,965.5 tok/s @200 MHz, 16/16 bit-exact** (MEASURED, WIDE-WORD-DATAPATH-LOG.md:990).

**Parameters**

| name | meaning | live | range | where |
|---|---|---|---|---|
| `LANES` | PE lanes = INT4 nibbles per wide URAM word; WBITS=LANES*4 | 128 | pow2 | gemm_banked_resident_vec.sv:23 / sequencer_sb.sv:24 |
| `P` | boundary width: P INT8 acts written in per cycle, P INT32 outs read per address | 8 | divides LANES,KMAX,MMAX | gemm_banked_resident_vec.sv:26 / sequencer_sb.sv:23 |
| `N / NC` | batch streams total / per cohort (split-brain) | 16 total, 8 per cohort | <=16 (2.0/DSP ceiling) | sequencer_sb.sv:25-26 |
| `ND` | of N streams, how many use the DSP-packed leaf (streams N-ND..N-1); rest are LUT | 0 | 0..N | gemm_banked_resident_vec.sv:25 |
| `MMAX / KMAX` | max output rows / max reduction length of any single layer | 1024 / 1024 | — | gemm_banked_resident_vec.sv:27-28 |
| `WWORDS` | resident URAM capacity in wide words; image ~12.6 Mbit fixed, scales inversely with LANES | 25600 (12,800 at L=256) | — | gemm_banked_resident_vec.sv:29 / sequencer_sb.sv:33 |
| `RLAT` | read->mac pipeline depth; stage 0 = registered URAM read | 2 | >=2 | gemm_banked_resident_vec.sv:30 |
| `ABITS` | per-lane accumulator width; |acc|<=8*128*1024=2^20 so 24b holds it with 3 bits margin | 24 | — | gemm_banked_resident_vec.sv:43 |
| `DP` | double-pump: MAC accumulator runs at 2x clk, 2 K-steps/clk (ND=0 only) | 0 | 0 or 1 | gemm_banked_resident_vec.sv:39 / sequencer_sb.sv:44 |
| `K2` | (gemv only) 2 K-steps/clk via 2nd URAM read port; free at N=1 | 0 (gemv base) | 0 or 1 | gemv_banked_resident_vec.sv:28 |
| `BANKW` | URAM-native bank width: 72 if WBITS>512 else WBITS (geometry-forced) | 512 (L=128, single bank); 72 at L=256 | 72 or WBITS | gemm_banked_resident_vec.sv:79 |
| `GAP2` | DSP pack gap: two INT4 weights 22 bits apart in the 27-bit signed operand port | 22 | 22 (24 cascades 2 DSPs) | dsp3_pack_proof.py:96 / mac_bank_dsp wpak |

**Key facts**

- The live GEMM engine is the split-brain path: sequencer_sb -> two cohort_engine -> gemm_cohort_vec, all sharing one weight_bank_tdp (cohort0 port B, cohort1 port A) (fabric/stage3/rtl/gemm_split_brain.sv:68-96, sequencer_sb.sv:316-350)
- Wide-word banking layout is `reg [P*W-1:0] buf [0:ROWS-1]` (lane l in bits [l*W +: W]); the naive [P][rows] shape read at a variable row blew LUT to 185% / MUXF7 to 151% and failed DRC before placement (fabric/stage3/WIDE-WORD-DATAPATH-LOG.md:71-91)
- LANES is not the fit lever, P is: L=64 saved only ~6k LUT (resident GEMV ~13k LUT); the P-wide datapath was the ~114k bulk (fabric/stage3/WIDE-WORD-DATAPATH-LOG.md:119-121)
- Weights are transposed wide-word packed: one wide word = one column k of LANES output rows in group g; word(layer,g,k)=w_base[layer]+g*K+k, lane L at bits [L*4 +: 4] (fabric/stage3/pack_banked_resident.py:8-13,41-62)
- Weights are boot-STREAMED via a 32-bit chunk loader that assembles SUBW=WBITS/32 chunks into one wide word and commits all NB banks in one shot, NOT baked into the bitstream (fabric/stage3/rtl/gemm_banked_resident_vec.sv:97-118)
- URAM banks are 72b-wide by geometry: a single 1024b memory pads to 16 wide x 4 cascade = 64 URAM = whole device -> ultra infeasible -> ~400k LUTRAM fallback; 72b banks (15 x 4 = 60 URAM at L=256) is the dense fix (fabric/stage3/rtl/gemm_banked_resident_vec.sv:79-88, WIDE-WORD-DATAPATH-LOG.md:396-403)
- DSP packing is exactly 2.0 INT4xINT8 MACs/DSP: two +8-biased weights at a 22-bit gap share one INT8 act in one DSP48E2 (wpak={w1,18'b0,w0} < 2^26); 3/DSP is proven arithmetically impossible on the 27x18 multiplier (fabric/stage3/rtl/gemm_banked_resident_vec.sv:500-557, research/dsp3_pack_proof.py:26-32)
- The DSP-batch stream ceiling is N=16 at LANES=128 on 1,024 DSPs (128/2=64 DSP/stream) — N=24 requires the impossible 3/DSP (fabric/stage3/research/DSP-PACKED-GEMM.md:276-286)
- Each stream's 128 MACs must be one keep_hierarchy leaf: at N=8 (1,024 mults) Vivado's bulk multiplier optimization detaches the URAM read register and refuses ultra, dumping weights to ~450k LUTRAM (fabric/stage3/rtl/gemm_banked_resident_vec.sv:465-474, WIDE-WORD-DATAPATH-LOG.md:529-535)
- Accumulator width ABITS=24 is range-proven exact: |acc| <= 8*128*1024 = 2^20 at the (w=-8,x=-128,k=1024) corner, 3 bits margin (fabric/stage3/rtl/gemm_banked_resident_vec.sv:40-43)
- HDL inference of TDP UltraRAM is dead in Vivado 2025.2; weight_bank_tdp/embed_bank_tdp use the dual-dialect pattern — xpm_memory_tdpram ultra under SYNTHESIS, behavioral 2-port array under iverilog; TDP ultra requires WRITE_MODE=no_change both ports (fabric/stage3/rtl/weight_bank_tdp.sv:14-25,129)
- Single-stream cycle floor is 12,800 wide URAM word reads/token; batch amortises the SAME reads across N streams (fan-out is in the activation dimension, URAM bandwidth independent of N) (fabric/stage3/WIDE-WORD-DATAPATH-LOG.md:455, research/DSP-PACKED-GEMM.md:141-157)
- MEASURED: single-stream 11,143.9 tok/s @200MHz (CYCLES=17,947); split-brain N=16 record 59,965.5 tok/s @200MHz, 16/16 bit-exact (fabric/stage3/WIDE-WORD-DATAPATH-LOG.md:445-449,990)
- The synth FSM goes RUN->DRAIN directly; SETTLE is sim-only (waits 4 cyc for the MAC tail, latches y_lat) so silicon finishes each call ~4 cyc earlier than sim -> sim cyc/tok is an upper bound (~0.8%) (fabric/stage3/rtl/gemm_banked_resident_vec.sv:327-348)

**Files**

- `fabric/stage3/rtl/gemv_banked_resident_vec.sv` — Single-stream base GEMV with P-wide boundary + in-core weight_bank_tdp; the proven datapath (K2 dual-read, addend-stage timing split, embed-port steal)
- `fabric/stage3/rtl/gemm_banked_resident_vec.sv` — N-batch-stream GEMM, self-contained URAM banks; DEFINES mac_bank (LUT) and mac_bank_dsp (DSP-packed) leaves; the reference for the cohort core
- `fabric/stage3/rtl/gemm_cohort_vec.sv` — ONE split-brain cohort: same FSM/MAC/readback but URAM hoisted out (drives waddr, consumes wword_rd); adds AQ/RUN overlap stall, 2nd act write port, y_out2
- `fabric/stage3/rtl/gemm_split_brain.sv` — Two gemm_cohort_vec + one shared weight_bank_tdp (cohort0 port B, cohort1 port A); the GEMM-level split-brain
- `fabric/stage3/rtl/weight_bank_tdp.sv` — Shared resident URAM weight image, dual-port (loader+cohort1 on A, cohort0 on B); dual-dialect xpm/behavioral; DP column-parity split
- `fabric/stage3/rtl/embed_bank_tdp.sv` — Shared token/pos embedding dual-port image (same TDP pattern); fixes coh0/coh1 const-prop asymmetry
- `fabric/stage3/rtl/mac_bank_dp.sv` — Double-pumped LUT MAC leaf: 2 K-steps/clk in clk2x domain, phase=~clk anchoring; bit-exact timing transform
- `fabric/stage3/rtl/mac_bank_dsp_dp.sv` — Double-pumped DSP-packed MAC leaf: raw 48-bit pairs + sum_act in clk2x domain
- `fabric/stage3/rtl/gemm_dsp_resident_vec.sv` — Older all-DSP variant with 24-bit-gap pack and in-core SETTLE-latched recovery (superseded by mac_bank_dsp 22-bit RAW-pair)
- `fabric/stage3/pack_banked_resident.py` — Transposed wide-word INT4 packer: builds the resident .mem image + per-layer w_base; the bit-exact gate (BANKED_RESIDENT_VERDICT)
- `fabric/stage3/research/dsp3_pack_proof.py` — Bit-accurate DSP48E2 model + 610,756-case proof that 3/DSP is impossible and 2.0/DSP is exact
- `fabric/stage3/research/DSP-PACKED-GEMM.md` — Design note: WP487 packing math, 740-free-DSP budget, the 2.0/DSP verdict and N=16 ceiling
- `fabric/stage3/WIDE-WORD-DATAPATH-LOG.md` — The chronological build log: the mux-blowup gotcha (§2-3), URAM geometry (§11), URAM cliff/keep_hierarchy (§16), all MEASURED tok/s records (§13,27)
- `fabric/stage3/rtl/sequencer_sb.sv` — Live top-level. RTL defaults LANES=128/P=8/N=16/NC=8/ND=0/DP=0/TMAX=32; the record bitstream overrides TMAX=16 and ND=6/cohort (DP=0). Instantiates shared weight_bank_tdp + two cohort_engine
- `fabric/stage3/rtl/cohort_engine.sv` — Per-cohort wrapper instantiating gemm_cohort_vec and exposing the shared weight-bank read port

**Gotchas**

- Never use `reg [W] buf [0:P-1][0:ROWS-1]` for a variable-row buffer — it synthesises to per-lane row muxes and overflows the KV260 (185% LUT). Use `reg [P*W-1:0] buf [0:ROWS-1]` so the row is a memory address (WIDE-WORD-DATAPATH-LOG.md:71-91).
- iverilog: a variable +: part-select on an element of an unpacked array reads X. Always copy the element to a plain wire first (e.g. `wire [P*8-1:0] xrow = xr[RLAT-1]`) before the lane select (gemm_banked_resident_vec.sv:225-227).
- wsel must be a WIRE (word_p[RLAT-2]), not an always@* copy — an always@* copy made Vivado trim the URAM read register to 4 bits and fall to ~400k LUTRAM (gemm_banked_resident_vec.sv:186-189).
- Do NOT infer TDP UltraRAM — HDL inference is dead in Vivado 2025.2 (falls to BRAM / 'invalid write mode'). Use the dual-dialect xpm_memory_tdpram(ultra) under SYNTHESIS, behavioral array under iverilog (weight_bank_tdp.sv:14-25).
- A single 1024b-wide weight memory pads to 64 URAM (whole device) and fails as ultra. Must bank at 72b-native width (weight_bank_tdp geometry; WIDE-WORD-DATAPATH-LOG.md:396-403).
- At N>=8 (1,024 mults) each stream's MACs MUST be a keep_hierarchy leaf or Vivado's bulk multiplier pass detaches the URAM read register and dumps weights to LUTRAM (gemm_banked_resident_vec.sv:465-474).
- iverilog's 16-bit part-select index space wraps at bit 16384 (stream 4 at YBITS=24*128*4): a flat N*YBITS concat silently makes streams 4..7 read 0..3. The SYNTHESIS path uses named <=12,288-bit chunks + a 4:1 select; the sim path uses an unpacked wire array (gemm_banked_resident_vec.sv:190-204).
- The DSP pack gap is 22, NOT 24 — a 24-bit gap (28-bit wpak) exceeds the 27-bit signed port and Vivado cascades TWO DSPs per multiply, halving the packing benefit (gemm_banked_resident_vec.sv:504-507).
- Sim cyc/token is an UPPER BOUND: SETTLE (the 4-cycle MAC-tail wait + y_lat latch) is `ifndef SYNTHESIS only; silicon goes RUN->DRAIN directly and runs ~0.8% faster than sim (gemm_banked_resident_vec.sv:327-348).
- Silicon overclock vs STA: designs closing ~70-125 MHz in STA run bit-exact at 166.7-200 MHz on the -2LV silicon (~1.37-1.64x); 200 MHz needs a genuinely 200-clean build, and the PLL only steps 1000/N so there is no fclk between 166.7 and 200 (WIDE-WORD-DATAPATH-LOG.md:436,449,578).

**Open questions**

- **RESOLVED** (§7 / dead-lever ledger): the full DP=1 split-brain build was never board-run; the on-silicon clk2x bit-exact result exists only for the MAC-only DP branch (`dp-hw-maconly`), where the fabric→clk2x data feed walled at 50 MHz and killed the lever (100K-REVIEW.md:44).
- gemm_dsp_resident_vec.sv (the 24-bit-gap all-DSP core) appears superseded by mac_bank_dsp's 22-bit RAW-pair form used in the batch/cohort cores; whether gemm_dsp_resident_vec is still instantiated anywhere live was not confirmed (no top-level instantiation found in sequencer_sb).
- The exact current URAM/DSP/LUT fit of the LIVE sequencer_sb split-brain build (DBG=0/ATT2=0 bitstream config) was not re-derived from a fresh OOC in this read — the cited fit numbers (URAM 60-64/64, DSP 505-1171) come from the log's historical build entries, not a current run.
- ND (DSP-packed streams per cohort) defaults to 0 in the RTL, but the 59,965.5 tok/s record build ran ND=6/cohort (12 of 16 streams on the DSP-packed leaf) — DERIVED from BD default + committed gate config + ~95% DSP occupancy. The all-LUT (ND=0) path is gated bit-exact and synth-proven but was NOT the record build. (Earlier drafts stating the record ran ND=0 / "all-LUT mac_bank" misread the RTL default.)

---

## Attention & the KV Window (TMAX) — Kevin-on-Kria Stage 3

There are two distinct attention/KV subsystems in this repo, and conflating them is the single biggest source of confusion — keep them separate.

**Track A — the split-brain record engine (`sequencer_sb` + `vec_attn`, doc 6).** This is the design that holds the MEASURED record of 59,965.5 tok/s @200 MHz (§27 of the log). Attention here is `vec_attn` (rtl/vec_attn.sv): a P-wide, single-head kernel with SHORT internal K/V/prob buffers sized by a small `TMAX` generic (default 32, record build 16). Its K/V are FULL-PRECISION 32-bit Q.16 activations that the sequencer streams in fresh on every attention call — there is NO persistent, quantised, cross-token cache here, and NO INT8-KV. The record build runs two N=8 cohorts (N=16 total) that SHARE one `vec_attn` behind a hold-until-done arbiter (`ATT2=0`), because per-cohort attention (`ATT2=1`) does not fit the device. The `TMAX=16` in that build is the whole point of §26–27: shrinking the window from 32→16 freed the BRAM/URAM that let the architectural wave close (fabric/stage3/rtl/vec_attn.sv:38, rtl/sequencer_sb.sv:170-212, WIDE-WORD-DATAPATH-LOG.md:983-998).

**Track B — the faithful decode engine (`sequencer_vec` + `kv_bank` + `vec_attn_w`, doc 7).** This is where the real on-chip KV cache lives. `kv_bank` (rtl/kv_bank.sv) is a quantise-at-write / dequantise-at-read INT8 (K8/V8) cache that PERSISTS across tokens, holding K and V codes + per-(layer,kv,head,position) asymmetric headers for a whole conversation. `vec_attn_w` (rtl/vec_attn_w.sv) consumes one dequantised HEAD_DIM-wide position per cycle. This path decodes with full T=window context (a real message, not degenerate T=1 text) and is MEASURED at 11,343 tok/s @142.9 MHz, N=1, full T=128 window, 3/3 bit-exact (WIDE-WORD-DATAPATH-LOG.md:1278-1299). The designed TMAX is 256; the SHIPPING TMAX is 128 (K8 fits; K8/T256 busts BRAM — see §37).

**The DDR spill/restore path (`kv_dma` + `kv_prefetch`).** These are the context-restore bricks for when the on-chip window is too small. They are SIM-ONLY / bit-exact-gated — `kv_dma` is instantiated only by `kv_prefetch`, and `kv_prefetch` is instantiated only by its testbench; neither appears in any sequencer or bitstream build. Their bandwidth story (KV-DDR-100K.md) is the aggregate-100k analysis: ~6.4 GB/s KV read at 100k tok/s, which is essentially the entire ~6–7.5 GB/s sustained KV260 DDR budget, so 80k is the reliable claim and 100k the stretch.

## 1. Two tracks, one table

| | Track A (record) | Track B (faithful) | DDR path |
|---|---|---|---|
| Attention core | `vec_attn` (rtl/vec_attn.sv) | `vec_attn_w` (rtl/vec_attn_w.sv) | (feeds vec_attn) |
| KV storage | internal 32-bit Q.16 buffers, re-streamed per call | `kv_bank` INT8 (K8/V8), persistent | `kv_dma`/`kv_prefetch`, DDR |
| Sequencer | `sequencer_sb` | `sequencer_vec` | none (sim brick) |
| TMAX (default / built) | 32 / **16** | 256 / **128 shipping** | prefetch 32, dma per-window |
| Precision | Q.16 (VFRAC=16, 32b) | INT8 codes + int32 lo / uint16 scale | INT4 default (KBITS=4) |
| Status | MEASURED 59,965.5 tok/s @200 (§27) | MEASURED 11,343 tok/s @142.9 (§39) | SIM-ONLY, bit-exact gated |
| On silicon? | yes | yes | **no** |

---

## 2. Track A — `vec_attn` (the record path's attention)

Single-head, P-wide kernel computing `ctx[d] = sum_j prob[j]*v_j[d]` for one HEAD_DIM=64 head, bit-identical to `seq_ref._attn_step` (rtl/vec_attn.sv:1-32).

### 2.1 Pinned integer datapath (rtl/vec_attn.sv:55-64)
- `VFRAC=16`, `ISQRT=3`, `SCORE_FRAC=8`, `PROB_FRAC=20`, `RESID_FRAC=25`.
- `SCORE_SH = 2*VFRAC + ISQRT - SCORE_FRAC = 27`; `CTX_SH = PROB_FRAC + VFRAC - RESID_FRAC = 11`.
- q,k,v are signed Q.16 32-bit; `score[j] = sat16(rsh_round(sum_d q·k, 27))` → Q8.8; softmax `prob` is unsigned 21-bit Q1.20; `ctx[d] = rsh_round(sum_j prob·v, 11)` → Q6.25 (rtl/vec_attn.sv:9-14).
- `NGRP = HEAD_DIM/P` (=8 at P=8) — adder-tree passes per score / number of dim-groups (rtl/vec_attn.sv:64).

### 2.2 On-chip storage — internal, per-call, NOT persistent (rtl/vec_attn.sv:66-70)
Wide-word banked (CLAUDE.md rule: lane l in bits [l*32 +: 32], row = memory address):
- `qmem[0:HEAD_DIM/P-1]` — q, P*32-wide (8 words at P=8).
- `kmem[0:TMAX*HEAD_DIM/P-1]` — K cache, position-major, P*32-wide.
- `vmem[0:TMAX*HEAD_DIM/P-1]` — V cache, position-major.
- `probmem[0:TMAX-1]` — softmax probs Q1.20 (21-bit unsigned).

These are 32-bit Q.16 activation buffers streamed IN fresh on every `start` (load protocol: q words, then T*NGRP K words, then T*NGRP V words, rtl/vec_attn.sv:25-31). There is no INT8 quantisation and no cross-token persistence in Track A.

**KV-window trap (load-bearing):** `probmem` is written at `probmem[ji[4:0]]` (rtl/vec_attn.sv:310) — a 5-bit index that SILENTLY WRAPS at T>32. So `vec_attn` is only correct for T ≤ 32; this is the "old vec_attn probmem[ji[4:0]] silently wrapped at T>32" latent trap called out in vec_attn_w's header (rtl/vec_attn_w.sv:11-13) and the log (WIDE-WORD-DATAPATH-LOG.md:1062-1063). Track B's `vec_attn_w` fixes it with a full-width index.

### 2.3 FSM (rtl/vec_attn.sv:88-98)
`S_IDLE → S_LD_Q → S_LD_K → {S_SCORE ↔ S_SCORE_EMIT} → S_SM_COLL → S_CTX → S_DONE`.
- `S_LD_V` is declared but effectively unused: "Cut 1" overlaps V-loading with score+softmax via a background V-loader (`vld_active`/`vld_cnt`) armed on the last K word — V words stream into `vmem` concurrently while `S_SCORE`/`S_SM_COLL` run (rtl/vec_attn.sv:104-113, 181-189, 218-239). The `S_SM_COLL→S_CTX` transition is fenced on `!vld_active` so ctx never reads an unlanded V word (rtl/vec_attn.sv:308-324).
- Score: 3-stage pipeline (operand-reg A → product-reg B → accumulate C) over `grp` passes, one product-group/cycle (rtl/vec_attn.sv:248-280).
- Ctx: "Cut 3" cross-group streaming — stage A emits one operand/cycle as a continuous stream over (grp,pj), never draining between groups; accumulator double-buffered by group parity (`ctx_acc[2*P]`, bank=`grp[0]`); first position overwrites its bank, last emits combinationally. Per-dim add order preserved → bit-identical (rtl/vec_attn.sv:121-150, 326-391). `S_CTX_EMIT` is folded into `S_CTX`.

### 2.4 Sizing at the record build (TMAX=16, P=8, HEAD_DIM=64)
- kmem = vmem = 16*64/8 = 128 words × 256 bits = **32 Kbit each**; qmem = 8×256 = 2 Kbit; probmem = 16×21b.
- These are small per-unit buffers. The record build uses **one shared** `vec_attn` (`ATT2=0`), so only one copy exists. `ATT2=1` (per-cohort attention) needs 2 copies + TMAX=16 BRAM and ~+5.7k LUT, and is ~1.9k LUT over the device, so bitstreams disable it (rtl/sequencer_sb.sv:154-159, WIDE-WORD-DATAPATH-LOG.md:962-1005).

### 2.5 Where Track A's K/V comes from
The cohorts drive `at_ldv*`/`at_lddat*` from `cohort_engine` (rtl/sequencer_sb.sv:334,363), whose attention datapath regenerates/streams the window each call. There is no separate persistent KV bank in `sequencer_sb`; the short window (16) is re-streamed. (I did not fully trace cohort_engine's internal K/V scratch layout — it is out of this subsystem's core scope, and the KV-window story lives in vec_attn's TMAX and kv_bank.)

---

## 3. Track B — `kv_bank` (the real on-chip INT8 KV cache)

rtl/kv_bank.sv. Quantise-at-write, dequantise-at-read-stream, K8/V8, per-(head,position) asymmetric — the pinned `IntKVQSequencer(kbits=8, vbits=8, rotate=False, divfree=True)` contract from model/goformer_kvq.py (rtl/kv_bank.sv:1-19, WIDE-WORD-DATAPATH-LOG.md:1007-1025).

### 3.1 Parameters (rtl/kv_bank.sv:26-33)
`P=8, HEAD_DIM=64, NHEAD=4, NLAYER=4, TMAX=256, KBITS=8, INV_SH=24`.
Derived (rtl/kv_bank.sv:68-73):
- `HR = HEAD_DIM/P = 8` (beats per head vector).
- `QMAX = (1<<KBITS)-1 = 255`.
- `NHSEL = NLAYER*2*NHEAD = 32` (the (layer, K|V, head) selector count).
- `HROWS = NHSEL*TMAX = 32*TMAX` — **one code row per (selector, position)**.

### 3.2 Memory layout (rtl/kv_bank.sv:84-95, 293-312)
Two dual-port memories, **dual-dialect** (`ifdef SYNTHESIS` → `xpm_memory_tdpram`; else behavioral 2-port array). HDL inference of true-dual-port UltraRAM is dead in Vivado 2025.2, so XPM is mandatory (rtl/kv_bank.sv:84-89; SPLIT-BRAIN.md:57-68):
- `code_bank` — width `HEAD_DIM*KBITS = 64*8 = 512 bits` (= 64 bytes = one head's INT8 codes for one position), depth `HROWS`, `MEMORY_PRIMITIVE="ultra"` (rtl/kv_bank.sv:257-274).
- `hdr_bank` — width 48 bits = `{scale16, lo32}`, depth `HROWS`, `MEMORY_PRIMITIVE="block"` (rtl/kv_bank.sv:275-292).
- Row address: `pbase = ((layer*2 + kv)*NHEAD + head)*TMAX + pos` (rtl/kv_bank.sv:329-330, 410-411). Position is a memory ADDRESS, never a per-lane mux (the CLAUDE.md wide-word banking rule).
- **Banks are never cleared** between tokens or conversations: a conversation restarts at pos 0 and overwrites; attention at pos p reads only rows 0..p, all freshly written (rtl/kv_bank.sv:16-19).

### 3.3 Quantise-write datapath (rtl/kv_bank.sv:100-194)
Per (head, K|V, position) vector of HEAD_DIM Q.16 ints, streamed HR beats of P lanes:
- `lo = min(x)`, `span = max(x) - lo` (running min/max collected over beats, pipelined quad-reductions in W_COLL, drained in W_MMD; rtl/kv_bank.sv:141-155, 335-358).
- `scale = rdiv(span, 255)` computed **divider-free** by the exact magic multiply `((span+127)*0x80808081) >> 39`, proven == floor((span+127)/255) over [0, 2^22) (rtl/kv_bank.sv:103-112, 360-368).
- `inv = rdiv(2^24, scale)` read from a two-ROM constant LUT (`inv_lut_lo[0:4095]` 25-bit + `inv_lut_hi` 13-bit — a BRAM-diet split), initialised from `inv_lut_lo.mem`/`inv_lut_hi.mem` (rtl/kv_bank.sv:112-133). **The empty-ROM silicon bug (§38): a missing $readmemh is a silent all-zero ROM → w_inv=0 → every code=0; build tcl now HARD-ERRORS on missing ROM init** (WIDE-WORD-DATAPATH-LOG.md:1247-1276).
- `code = clip((u*inv + 2^23) >> 24, 255)`, `u = x - lo ≥ 0` (rtl/kv_bank.sv:185-194).

### 3.4 Dequant-read datapath (rtl/kv_bank.sv:196-242)
`x_hat = code*scale + lo` (exact integer, kv_dma-verbatim), lo sign-extended int32, scale uint16. Zero-bubble, **one dequantised HEAD_DIM position per cycle** after a 2-cycle fill (header co-read alongside the code row); total `T + 2` cycles per stream (rtl/kv_bank.sv:196-199, 245-249, 414-428).

### 3.5 Write FSM (rtl/kv_bank.sv:75-79, 326-400)
`W_IDLE → W_COLL → W_MMD → W_SCALE → W_SCALE2 → W_INVL → W_INVR → W_QNT → W_CWR`.
- `W_CWR` commit is **folded into read port A** (`cwr_fire = (wst==W_CWR) && (rst_st==R_IDLE)`): each memory is write-else-read on A plus read on B = exactly two ports (the URAM TDP contract). The quant commit stalls until stream A is idle — it lands in the softmax gaps between K and V streams (rtl/kv_bank.sv:248-255, 392-398).

### 3.6 Read FSMs — TWO independent read ports (rtl/kv_bank.sv:81-83, 402-458)
`R_IDLE → R_RUN` for stream A (port A, muxed with the write) and stream B (port B). This is the doc-7 R4e "twin engine" enabler: `vec_attn_w` A reads port A, B reads port B, so a head PAIR runs concurrently (rtl/kv_bank.sv:58-66, rtl/sequencer_vec.sv:256-343).

---

## 4. Track B — `vec_attn_w` (full-head-width attention)

rtl/vec_attn_w.sv. Consumes a whole position's HEAD_DIM row per beat (vs vec_attn's P-wide 528 cyc/position). `TMAX=256` default, `TW = $clog2(TMAX)` (rtl/vec_attn_w.sv:32,55).
- FSM: `W_IDLE → W_Q → W_K → W_SMC → W_V → W_EMIT` (rtl/vec_attn_w.sv:75-76).
- K phase: 5-stage score pipeline (register row → 64 registered products → 8 partial sums of 8 → two 4-way sums → final sum+round/sat), 1 position/cycle, feeding `softmax_f` inline at PASS1 pace. Integer 64-bit adds with no mid-sum saturation → any regrouping bit-exact (rtl/vec_attn_w.sv:85-205).
- **Full-width `probmem[0:TMAX-1]`, indexed `probmem[jc]` / `probmem[vj[TW-1:0]]`** — this engine OWNS T up to TMAX, no [4:0] wrap (rtl/vec_attn_w.sv:82,222,259).
- V phase: 64 per-dim `ctx_acc` accumulators, position order j=0..T-1 preserved → bit-exact; emit NGRP P-wide ctx strobes (rtl/vec_attn_w.sv:104-111, 217-254).
- `tcount` is driven as `pos+1` by the sequencer (rtl/sequencer_vec.sv:323,326) — attention window grows by one each token.

---

## 5. The hard constraint: prompt + gen − 1 ≤ TMAX

Enforced in the harnesses, not in RTL (the RTL just uses `tcount = pos+1` and would wrap/overflow if exceeded):
- Sim gate: `assert plen + ngen - 1 <= tmax, "prompt+gen must fit the KV window"` (run_vec_kv.py:74).
- Board driver: identical assert, `--tmax` documented as "the bitstream's KV window (TMAX generic); prompt+gen must fit" (board/pl_seq_kv.py:145-146,159).
- `kv_bank.rd_tcount = pos + 1` and `wq_pos = pos`; a fresh decode rewrites KV[p] before any pass reads it, so stale rows are harmless (board/pl_seq_kv.py:113-115, rtl/sequencer_vec.sv:321-326).

The board driver must upload embed rows matching the build's TMAX or pos>0 corrupts (the TMAX=16 record build needed `--tmax 16`; a stale driver corrupted it — WIDE-WORD-DATAPATH-LOG.md:996-998, 969-974).

---

## 6. KV storage size vs TMAX (DERIVED, kv_bank / Track B)

General formula = mandate's `2 * layers * d * tmax * bytes` where d = NHEAD*HEAD_DIM = 4*64 = 256, layers=4, bytes=1 (INT8/KBITS=8):

**Codes** = `2 * NLAYER * NHEAD * HEAD_DIM * TMAX * (KBITS/8)` = `2048 * TMAX` bytes (rtl/kv_bank.sv row math: HROWS=32*TMAX rows × 64 B).
**Headers** = `6 B * NHSEL * TMAX` = `192 * TMAX` bytes (48-bit hdr rows).

| TMAX | codes | headers | total | log cross-check |
|---|---|---|---|---|
| 16  | 32 KB  | 3 KB  | 35 KB  | — |
| 32  | 64 KB  | 6 KB  | 70 KB  | KV-DDR §0: "64 KB/stream" at INT8 T=32 |
| 128 | 256 KB | 24 KB | 280 KB | §37: code "4096×512b = 57 BRAM tiles" |
| 256 | 512 KB | 48 KB | 560 KB | **§28: "Codes 512 KB (URAM), headers 48 KB (BRAM)"** (rtl/kv_bank.sv:14) |

### Fraction of the ~3 MB on-chip budget
On-chip = URAM (64 × 288 Kb = 2.25 MB) + BRAM (144 × 36 Kb = 648 KB) ≈ 2.9 MB. At TMAX=256 the codes alone (512 KB) are ~17% of on-chip, but the weights already fill URAM (60/64, §13/§34), so the code_bank spills to BRAM — and that is the binding constraint (below).

---

## 7. Max TMAX before overflow (from code + logs)

**The binding resource is BRAM, not raw capacity** (the weights own the URAM):
- **TMAX=256, K8: does NOT fit.** OOC take-5 (§36) with XPM banks: LUT 110,392 (94.3%), URAM 64/64, but **BRAM 281.5/144** — the code_bank landed in BRAM (~114 tiles) because URAM was full of weights (WIDE-WORD-DATAPATH-LOG.md:1207-1214).
- **TMAX=128, K8: FITS and SHIPS.** The shipping config is "K8 no-rotate at TMAX=128", code_bank 4096×512b = 57 BRAM tiles, total ~129/144 after moving embeds into URAM's spare depth (WIDE-WORD-DATAPATH-LOG.md:1228-1245). This is what was MEASURED at 11,343 tok/s @142.9 MHz, N=1, full T=128 window, 3/3 bit-exact (§39, :1278-1292). T=128 "still holds a full chat turn + reply."
- **TMAX=256 return path (parked):** needs either K4/V4+Hadamard (halves code to ~57 BRAM, +0.72% NLL, in band) — but the Hadamard butterflies blew LUT to 181% and K4 diverged at long T (~pos 84), so it's parked (§37, :1228-1237); or code_bank INTO URAM via 3-cascade weights freeing ~15 URAM (§35, :1199-1204).

**So: the fabric-supported max faithful window is TMAX=128 (K8) today; TMAX=256 is designed and gated in sim but does not fit the KV260 with K8 weights resident.** For the record engine (Track A/vec_attn) the effective max is TMAX≤32 (the probmem [4:0] wrap), and the record build deliberately uses 16.

---

## 8. Split-brain dual-port URAM sharing (SPLIT-BRAIN.md)

The observation: the N=16 single-engine design has 73k cycles queuing behind the URAM weight banks, which are busy only 27% of the time, and port A of the TDP URAM is idle after boot (loader-only) (SPLIT-BRAIN.md:7-13).

The architecture: two fully independent 8-stream cohorts, phase-free (no merge, no partner-waiting). They share ONLY the URAM weight banks, read through both TDP ports — cohort 0 on port B (today's path), cohort 1 on port A (muxed with the boot loader's write; loader idle at runtime). Independent addresses; the desync problem dissolves (SPLIT-BRAIN.md:15-30).

Key resolved risk: **true-dual-port UltraRAM works only via `xpm_memory_tdpram` with `MEMORY_PRIMITIVE="ultra"`, NOT HDL inference** (which maps to 400 BRAM tiles or refuses with `[Synth 8-12186] ram_style="ultra" ignored: invalid write mode`). This is exactly the dual-dialect pattern kv_bank uses (SPLIT-BRAIN.md:57-68, rtl/kv_bank.sv:84-89, 256-312).

Note the naming subtlety: split-brain's *shared* dual-port structure is the **weight** URAM, not the KV cache. Attention is shared (ATT2=0) with an arbiter in the record build. `kv_bank` (Track B) independently uses the same TDP-URAM trick for its two READ streams (ports A and B feeding the twin vec_attn_w engines).

---

## 9. KV-to-DDR spill/restore (`kv_dma` + `kv_prefetch`) — SIM-ONLY

### 9.1 Live vs sim
**Sim-only, bit-exact gated, never on silicon.** `kv_dma` is instantiated only inside `kv_prefetch` (rtl/kv_prefetch.sv:122); `kv_prefetch` is instantiated only by its testbench `tb_kv_prefetch.sv` (run via run_kv_prefetch.py). Neither appears in any `sequencer_*` or `build_bd_*`/`impl_*` tcl. In sim the DDR is served by `rtl/ddr_latency_model.sv`; in silicon it would be an `S_AXI_HP` master (kv_dma.sv:36-41, KV-DDR-100K.md:10-12). CLAUDE.md confirms: "KV-to-DDR (kv_dma + kv_prefetch, sim-complete, bit-exact) is the context-restore path once the on-chip window is too small."

### 9.2 When it triggers
Conceptually when prompt+gen exceeds the on-chip TMAX window — the on-chip window becomes a live cache and older positions are restored from DDR. Today the shipping path just uses a bigger on-chip TMAX (128) instead.

### 9.3 kv_dma (rtl/kv_dma.sv)
Burst-read + dequant of one DDR position-row. Pinned row layout head-major: `[head0_hdr | head0_codes | ... ]`, per head `hdr = lo(int32 LSB-first) || scale(uint16) = 6 B`, `codes = HEAD_DIM*KBITS/8 B`. K4/V4: 38 B/head, 152 B/position (kv_dma.sv:15-19, 79-84). **Default KBITS=4** (kv_dma.sv:53) — the DDR path is written for INT4; the on-chip kv_bank is INT8. Emits P=8 dequantised channels/cycle (`x_hat = code*scale + lo`), wide-word banked obuf[head] (kv_dma.sv:28-34, 119-134). FSM: `S_IDLE → S_REQ → S_HDR → S_GETB → S_EMIT → S_HEND → S_DONE` (kv_dma.sv:109-117).

### 9.4 kv_prefetch (rtl/kv_prefetch.sv)
Double-buffered (ping-pong) window prefetch that hides the DDR round-trip behind attention compute. Window-granular (not per-head) because the DDR layout is position-row-major: read each (kv,pos) row ONCE and demux all NHEAD heads → 2*T row reads per window regardless of NHEAD (kv_prefetch.sv:11-31). Two `buf[HALFW]` halves, `HALFW=2*NHEAD*TMAX` words; slot(kv,head,pos)=`kv*(NHEAD*TMAX)+head*TMAX+pos` (kv_prefetch.sv:104-146). Fill FSM: `F_IDLE → F_AREQ → F_ISSUE → F_WAIT → F_DEMUX → F_NEXT`. `stall_cnt` counts cycles the consumer waited (≈0 once compute > 2T fill; cold window is the only un-hidden cost) (kv_prefetch.sv:44-48, 154-353). TMAX default 32 (kv_prefetch.sv:60).

### 9.5 Bandwidth story (KV-DDR-100K.md, all DERIVED/PROJECTED)
- Per stream at full window: **2 KB/token written, up to 64 KB/token read** at INT8 TMAX=32 (KV-DDR §0, :23-32).
- **100k aggregate = ~6.4 GB/s KV read regardless of N** (= 100,000 × 64 KB); write ~0.2 GB/s negligible (§1, :48-51).
- KV260 sustained PL→DDR ≈ **6–7.5 GB/s total, shared with the PS** (32-bit DDR4 ~9.6 GB/s raw × 65–80%) (§1, :52-60).
- **Verdict: KV read at 100k ≈ the whole DDR budget** → co-limited with compute (GEMM batch B=8 lifts the compute ceiling to ~125k). Honest reachable: **80k reliable, 100k stretch**; PS contention is the single biggest reason to claim 80k (§4-5, :136-181).
- The thesis escape hatch: TMAX=16 Kevin window HALVES KV traffic to ~3.2 GB/s (§1, :68-71). K4/V4+Hadamard is +0.72% NLL at 1.78× compression, MEASURED in model/exp_kvarn.py (§5 headroom table, :185-211).

---

## 10. ASCII data-flow (Track B faithful decode, one token)

```
 new token ─► GEMV(qkv) ─► K,V vectors (Q.16) ─► kv_bank quantise-write (K8/V8)
                                                    │  code = clip((x-lo)*inv>>24)
                                                    ▼
   [kv_bank persistent cache, rows 0..pos]   port A ──► vec_attn_w A (heads 0,2)
        code_bank 512b × 32*TMAX (URAM/BRAM) port B ──► vec_attn_w B (heads 1,3)
        hdr_bank  48b  × 32*TMAX (BRAM)         │  x_hat = code*scale+lo (dequant)
                                                ▼
                              score=q·k>>27 ─► softmax_f ─► ctx=Σ prob·v>>11 ─► residual
```

Compare Track A: `vec_attn` holds q/K/V/prob in its own 32-bit buffers, re-streamed each call, T≤32.

**Parameters**

| name | meaning | live | range | where |
|---|---|---|---|---|
| `TMAX (vec_attn, Track A)` | Max KV window depth of the record-path attention kernel; sizes internal kmem/vmem/probmem. Effective ceiling 32 due to probmem[ji[4:0]] wrap. | 16 (record build, §26-27); default 32 | 1..32 usable | rtl/vec_attn.sv:38 |
| `TMAX (vec_attn_w / kv_bank, Track B)` | Max faithful KV window; sizes kv_bank code/hdr banks and vec_attn_w full-width probmem. | 128 shipping (fits, MEASURED §39); 256 designed but busts BRAM with K8 (§36) | 1..256 designed | rtl/vec_attn_w.sv:32, rtl/kv_bank.sv:31 |
| `KBITS (kv_bank)` | Bits per K/V code in the on-chip cache (INT8). | 8 (K8/V8, +0.09% NLL, shipping) | 8 | rtl/kv_bank.sv:32 |
| `KBITS (kv_dma DDR path)` | Bits per code in the DDR spill layout. | 4 default (K4/V4); doc budget uses 8 as worst case | 2/4/8 | rtl/kv_dma.sv:53 |
| `HEAD_DIM` | Attention head width; code row = HEAD_DIM*KBITS bits. | 64 | 64 | rtl/vec_attn.sv:37, kv_bank.sv:28 |
| `NHEAD` | Heads per layer. | 4 | 4 | rtl/kv_bank.sv:29 |
| `NLAYER` | Transformer blocks; multiplies KV storage. | 4 | 4 | rtl/kv_bank.sv:30 |
| `P` | Reduction/emit lanes; HR=HEAD_DIM/P beats per head. | 8 | HEAD_DIM%P==0 | rtl/vec_attn.sv:36, kv_bank.sv:27 |
| `INV_SH / QMAX` | Divfree-quant fixed-point shift (2^24) and code clamp (255). | 24 / 255 | 24 / 255 | rtl/kv_bank.sv:33,69 |
| `SCORE_SH / CTX_SH` | Right-shifts: score = q·k>>27 (Q8.8), ctx = prob·v>>11 (Q6.25). | 27 / 11 | 27 / 11 | rtl/vec_attn.sv:61-62, vec_attn_w.sv:53-54 |
| `ATT2` | 1 = per-cohort vec_attn (needs TMAX=16 BRAM, ~1.9k LUT over device); 0 = shared unit + arbiter. | 0 in bitstreams (fit) | 0/1 | rtl/sequencer_sb.sv:167 |
| `HROWS (kv_bank)` | Code/hdr bank depth = NLAYER*2*NHEAD*TMAX = 32*TMAX rows. | 4096 at T=128; 8192 at T=256 | — | rtl/kv_bank.sv:71 |

**Key facts**

- Two separate attention/KV subsystems: Track A record path uses vec_attn with 32-bit Q.16 internal buffers re-streamed per call (no persistent/INT8 cache); Track B faithful path uses kv_bank as the real persistent INT8 K8/V8 cache (rtl/vec_attn.sv:66-70 vs rtl/kv_bank.sv:1-19).
- kv_bank stores codes as 512-bit rows (HEAD_DIM*KBITS = 64*8) one per (layer,kv,head,position), depth HROWS = NLAYER*2*NHEAD*TMAX = 32*TMAX; headers are 48-bit {scale16,lo32} rows (rtl/kv_bank.sv:71,84-95,257-292).
- KV storage = 2*layers*d*tmax*bytes = 2048*TMAX bytes of codes + 192*TMAX bytes of headers; at TMAX=256 that is 512 KB codes + 48 KB headers = 560 KB (rtl/kv_bank.sv:14, WIDE-WORD-DATAPATH-LOG.md:1023).
- Max fabric-supported faithful window is TMAX=128 (K8) which FITS and was MEASURED at 11,343 tok/s @142.9 MHz N=1 3/3 bit-exact; TMAX=256 with K8 busts BRAM at 281.5/144 (WIDE-WORD-DATAPATH-LOG.md:1207-1214, 1228-1245, 1278-1292).
- The hard constraint prompt+gen-1 <= tmax is enforced in Python harnesses only (run_vec_kv.py:74, board/pl_seq_kv.py:159), not in RTL; the RTL drives kv_bank.rd_tcount = pos+1 (rtl/sequencer_vec.sv:323).
- INT8 (K8/V8) is the on-chip cache precision; quant is divider-free: scale=rdiv(span,255) via magic multiply ((span+127)*0x80808081)>>39, inv from a 2-ROM constant LUT, code=clip((u*inv+2^23)>>24,255) (rtl/kv_bank.sv:103-133,185-194).
- vec_attn's probmem is indexed probmem[ji[4:0]] and SILENTLY WRAPS at T>32 - a latent trap; vec_attn_w fixes it with full-width probmem[jc] (rtl/vec_attn.sv:310, vec_attn_w.sv:11-13,259).
- kv_dma + kv_prefetch (the KV-to-DDR spill/restore) are SIM-ONLY: kv_dma instantiated only by kv_prefetch, kv_prefetch only by its testbench, neither in any sequencer or bitstream build (grep of rtl/*.sv; CLAUDE.md).
- 100k aggregate tok/s needs ~6.4 GB/s KV read (INT8, TMAX=32) which equals the entire ~6-7.5 GB/s sustained KV260 DDR budget shared with the PS; 80k is the reliable claim, 100k the stretch, TMAX=16 halves traffic (KV-DDR-100K.md:48-71,152-181).
- True-dual-port UltraRAM works only via xpm_memory_tdpram MEMORY_PRIMITIVE=ultra, never HDL inference (dead in Vivado 2025.2); both kv_bank's twin read streams and split-brain's shared weight banks rely on this dual-dialect pattern (SPLIT-BRAIN.md:57-68, rtl/kv_bank.sv:84-89,256-312).
- The record 59,965.5 tok/s @200 MHz build is sequencer_sb with TMAX=16, ATT2=0 (shared vec_attn + arbiter), 53,364 cyc, 16/16 bit-exact (WIDE-WORD-DATAPATH-LOG.md:983-992).
- A missing $readmemh ROM (inv_lut) synthesises to a silent all-zero ROM: on first silicon w_inv=0 made every KV code=0 (ctx wrong, qkv perfect); build tcl now hard-errors on missing ROM init (WIDE-WORD-DATAPATH-LOG.md:1247-1276).

**Files**

- `fabric/stage3/rtl/vec_attn.sv` — Track A record-path attention: P-wide single-head kernel, internal 32-bit Q.16 K/V/prob buffers (TMAX default 32, probmem[4:0] wrap trap), used by sequencer_sb
- `fabric/stage3/rtl/vec_attn_w.sv` — Track B faithful attention: full-head-width, 1 position/cycle, full-width probmem, consumes kv_bank dequant rows; twin instances in sequencer_vec
- `fabric/stage3/rtl/kv_bank.sv` — The real on-chip persistent INT8 (K8/V8) KV cache: quantise-write / dequant-read, dual-port (XPM ultra), code+hdr banks, TMAX 256/shipping 128
- `fabric/stage3/rtl/kv_dma.sv` — SIM-ONLY DDR burst-read + dequant of one position-row (KBITS=4 default); instantiated only by kv_prefetch
- `fabric/stage3/rtl/kv_prefetch.sv` — SIM-ONLY ping-pong window prefetch hiding DDR latency; instantiated only by its testbench
- `fabric/stage3/rtl/sequencer_sb.sv` — Split-brain record engine: two N=8 cohorts, shared vec_attn (ATT2=0) + arbiter, TMAX=16 record build
- `fabric/stage3/rtl/sequencer_vec.sv` — Faithful decode sequencer: instantiates kv_bank + twin vec_attn_w (ports A/B), KV-write feeder, tcount=pos+1
- `fabric/stage3/research/SPLIT-BRAIN.md` — Design note: dual-port URAM cohort sharing; the XPM-vs-inference URAM resolution
- `fabric/stage3/research/KV-DDR-100K.md` — KV-to-DDR bandwidth/scheduling/cap analysis; the 80k/100k co-limit and low-bit KV headroom table
- `fabric/stage3/WIDE-WORD-DATAPATH-LOG.md` — Build log: §27 record, §28 kv_bank R1, §34-39 the faithful-stream KV fit fight and first silicon (empty-ROM bug, TMAX=128 shipping, 11,343 tok/s MEASURED)
- `fabric/stage3/run_vec_kv.py` — Sim gate for the faithful KV decode; enforces plen+ngen-1<=tmax
- `fabric/stage3/board/pl_seq_kv.py` — On-board driver for faithful decode; --tmax must match build, same prompt+gen assert
- `fabric/stage3/rtl/ddr_latency_model.sv` — Sim DDR model backing kv_dma in the KV-DDR gate

**Gotchas**

- Do not conflate the two KV subsystems: 'the record engine' (sequencer_sb/vec_attn) has NO INT8 KV cache and NO persistent cross-token KV - it re-streams 32-bit Q.16 K/V per attention call. The INT8-KV / bytes-per-token story is exclusively kv_bank (sequencer_vec).
- vec_attn is only correct for T<=32: probmem[ji[4:0]] silently wraps (rtl/vec_attn.sv:310). Never raise vec_attn's TMAX past 32 without fixing the index. vec_attn_w is the T-up-to-256 engine.
- The KV-to-DDR path (kv_dma/kv_prefetch) is NOT on silicon and NOT in any bitstream - it is a sim-complete, bit-exact-gated brick only. Any claim that the demo spills KV to DDR is wrong.
- kv_dma defaults to KBITS=4 (INT4) while the on-chip kv_bank is KBITS=8 (INT8). The KV-DDR-100K bandwidth budget uses INT8 (2KB/token, 64KB@T32) as the worst case; the two are separate contracts.
- TMAX=256 faithful decode does NOT fit the KV260 today with K8 weights resident (BRAM 281.5/144, log §36). Shipping is K8@TMAX=128. TMAX=256 needs K4/V4+Hadamard (parked: LUT explosion + long-T divergence) or code-bank-into-URAM.
- A missing $readmemh init file is a SILENT all-zero ROM in the bitstream (not an error) - it made every KV code 0 on first silicon. build_bd_seq_kv.tcl/impl_seq_kv.tcl now hard-error on this; do not remove those guards.
- True-dual-port UltraRAM must use xpm_memory_tdpram MEMORY_PRIMITIVE=ultra; HDL inference maps to ~400 BRAM tiles or refuses synth. The behavioral iverilog branch only proves the sim side - the board run is the final bit-exactness check on the XPM side.
- The on-chip budget wall is BRAM (144 tiles), not URAM capacity: weights fill URAM (60/64), so the code_bank spills to BRAM. TMAX scaling is a BRAM-tile problem.

**Open questions**

- I did not fully trace where sequencer_sb / cohort_engine sources and stores the K/V it streams into the shared vec_attn between tokens (cohort_engine.sv:150 references a datapath submodule with NHEAD/HEAD_DIM). The record path's cross-token KV scratch layout was outside this subsystem's core (kv_bank/vec_attn/DDR) and I did not confirm it.
- Exact URAM/BRAM tile counts I report for code_bank at each TMAX (57 @128, ~114 @256) are quoted from the log's OOC runs, not independently recomputed from primitive geometry; they depend on how XPM tiles a 512-bit-wide memory.
- Whether the K4/V4+Hadamard TMAX=256 path (parked with a long-T divergence at ~pos 84) has been revisited/fixed since log §37 - the log leaves it un-debugged.
- The DDR path's KBITS=4 default vs the doc's INT8 budget: I did not find a single reconciled spec pinning the intended production DDR precision (the modules are parameterised, the doc analyses both).

---

## Kevin-on-Kria Subsystem 4: Non-linears & Numerics (LN / softmax / GELU / dequant / Gumbel sampling)

This subsystem is the fixed-point "glue arithmetic" of the goformer fabric datapath: everything that is NOT a matrix multiply. The residual stream is carried as signed Q6.25 32-bit integers; between GEMV calls the sequencer converts through five committed fixed-point blocks — LayerNorm (gamma-only, produces Q.22), INT8 activation-quant, per-channel INT32 dequant back to Q-format, attention softmax (Q8.8 scores -> Q1.20 probs), and GELU (Q4.12 LUT) — and finishes with an argmax/Gumbel-max sampler over the VOCAB head logits. Each block pins its own Q-format and is proven bit-exact (transcendentals: cosine > 0.9999, but the binding gate is token-stream identity) against a Python reference before any speed number is trusted (seq_ref.py:3-7, seq_ref.py:19-33).

There are two live engine families and they wire these blocks differently. The split-brain N=16 record engine (`sequencer_sb`, MEASURED 59,965.5 tok/s @200MHz) instantiates `layernorm_vec`, `vec_attn` (which contains `softmax.sv`, the 3-cycle/element shared brick), `vec_gelu` (paired-LUT `gelu_lut2`), and `vec_dequant`; its per-cohort NL FSM (`cohort_engine`/`nl_engine`) does a plain P-wide greedy argmax — there is NO on-chip sampling in the split-brain path. The doc-7 faithful-stream engine (`sequencer_vec`, N=1 T=256) instantiates `layernorm_vec`, `vec_attn_w` (which contains `softmax_f.sv`, a fork pipelined to 1 element/cycle), `vec_gelu`, `vec_dequant`, AND the on-chip Gumbel-max sampler (xorshift32 + 1024-entry noise LUT). So "sampling on silicon" is a `sequencer_vec` feature, not a `sequencer_sb` one.

Which blocks are P-wide (vectorized) vs scalar matters for cycle counts: LayerNorm I/O (`layernorm_vec`), GELU (`vec_gelu`), dequant (`vec_dequant`) and the argmax compare tree are all P lanes/cycle. Softmax is inherently scalar — one score in / one prob out per cycle — because the running-max, the exp accumulate, and the 2^40/sum reciprocal are sequential reductions over the T-length causal row; `softmax_f` only pipelines the read/compute/store sub-phases from 3 cyc/elem down to 1 cyc/elem, it does not go P-wide.

The Gumbel-max trick is the elegant piece: sampling from softmax(logit/T) is identically argmax_i(logit_i + T*g_i) with g_i a Gumbel(0,1) draw, so the fabric needs NO softmax over logits and NO host logit readback — it adds precomputed noise to each head logit during the head GEMV and the existing argmax winner IS the sample. The host writes ONE 32-bit seed per request; seed==0 is greedy (bit-exact to old argmax).

## 1. Q-format spec (the pinned numeric contract)

All formats are pinned in `seq_ref.py` and copied verbatim into RTL. `to_q` is round-half-away-from-zero (`seq_ref.py:86-88`).

| Quantity | Format | Bits | Where pinned |
|---|---|---|---|
| residual stream x | signed Q6.25 | 32 | `seq_ref.py:56` (RESID_FRAC=25, RESID_INT=6) |
| tok_emb + pos_emb | pre-quantized Q6.25 | 32 | `seq_ref.py:20`, `seq_ref.py:125-126` |
| LN gamma | signed Q4.20 | ~24 | `seq_ref.py:21`; `layernorm.sv:37` G_FRAC=20 |
| LN output y | signed Q.22 | fits 32 | `layernorm.sv:40` OUT_FRAC=22 |
| INT8 activations | INT8 [-128,127] | 8 | `seq_ref.py:164-172` |
| dequant mantissa | signed 24-bit | 24 | `vec_dequant.sv:52`; `seq_ref.py:180` |
| dequant exponent | signed 8-bit | 8 | `vec_dequant.sv:53` |
| q/k/v | signed Q.16 | 32 | `seq_ref.py:62` VFRAC=16; `vec_attn.sv:56` |
| attn score | signed Q8.8 | 16 | `softmax.sv:42`; `vec_attn.sv:58` SCORE_FRAC=8 |
| exp LUT value | Q1.20 unsigned | 21 | `softmax.sv:49-51` |
| softmax prob | Q1.20 unsigned | 21 | `softmax.sv:46`; `seq_ref.py:64` PROB_FRAC_A=20 |
| GELU I/O | signed Q4.12 | 16 | `gelu_lut.sv:2-3`; `run_gelu.py:21` FRAC=12 |
| head logit (argmax) | signed Q6.25 (INT32) | 32 | `seq_ref.py:295-298` |
| Gumbel noise | signed Q.25 | 32 | `gumbel.py:26` FRAC=25 |

Key derived shifts (all baked as localparams):
- `RESID_FRAC=25`, `LN_OUT_FRAC=22`, `VFRAC=16`, `GELU_FRAC=12`, `ISH=40` in `nl_engine.sv:37-41`.
- LN output shift `OUT_SH = (QX + Y_FRAC + G_FRAC) - OUT_FRAC = (25+26+20)-22 = 49` (`layernorm_vec.sv:52`, `layernorm.sv:44`).
- Attention score shift `SCORE_SH = 2*VFRAC + ISQRT - SCORE_FRAC = 27` (`vec_attn.sv:61`); ctx shift `CTX_SH = PROB_FRAC + VFRAC - RESID_FRAC = 11` (`vec_attn.sv:62`).
- Activation quant: `INV_SACT_SH=40`, `inv_sact=round(2^40/s_act)`, INT8 = `sat(rsh_round(y_q22 * inv_sact, LN_OUT_FRAC+40), -128,127)` (`seq_ref.py:158-172`). This `2^40` must match RTL `ISH=40`.

## 2. LayerNorm (gamma-only, Welford-free algebraic variance)

Three RTL variants exist, all bit-exact to `run_layernorm._ln_int_quantized`:
- `layernorm.sv` — scalar serial reference core (1 elem/cycle in, 1 out; ~264 non-streaming cycles between the two 256-cyc streams). Documented but not in the live datapath.
- `layernorm_par.sv` — throughput variant that builds sum(x) and sum(x*x) during the load (`layernorm_par.sv:8-20`).
- **`layernorm_vec.sv`** — the P-WIDE-I/O 10k-datapath block, instantiated by BOTH live engines: `sequencer_sb.sv:149` and `sequencer_vec.sv:209` as `layernorm_vec #(.P(P)) u_ln`.

Math (`layernorm_vec.sv:6-20`):
```
mean = sum >>> 8                                    (Q6.25, floored /256)
ssq  = sum(x*x) - 2*mean*sum + D*mean^2             (Q12.50, EXACT integer identity)
var  = ssq >>> 8                                    (Q12.50)
A    = (var >>> 24) + EPS                           (Q.26), EPS_A = 671 = round(1e-5*2^26)
rsqrt(A)  -> Q.26   via 64-entry seed LUT + 2 Newton steps
y    = ((d*rsqrt)*gamma) >>> 49                     (Q.22)
```
The exactness rests on the identity `sum_i(x_i-mean)^2 == sum(x*x) - 2*mean*sum + D*mean^2` for the SAME floored `mean = sum>>>8`, and integer-add associativity so an adder tree equals the serial sum (`layernorm_vec.sv:12-20`, `layernorm_par.sv:16-20`).

**rsqrt** (`layernorm_vec.sv:74-82`, `run_layernorm.py:45-72`): priority-encode MSB of A (`layernorm_vec.sv:124-128`); seed index = 6 mantissa bits below the MSB (`layernorm_vec.sv:242-245`); seed_rom is 64 entries of `1/sqrt(mant)` in Q1.16 loaded from `seed.mem` (`layernorm_vec.sv:63-64`, generated by `run_layernorm.seed_table` at `run_layernorm.py:45-49`); exponent handled by shifts with a `*sqrt2` (SQRT2Q15=46341) when E is odd (`layernorm_vec.sv:271-274`); then 2 Newton iterations `y <- y*(1.5 - 0.5*A*y*y)` (`layernorm_vec.sv:277-295`).

**Fmax pipelining** (bit-identical, just more latency): variance is 2-stage (`S_VAR`/`S_VAR2`, `layernorm_vec.sv:221-232`); each Newton multiply is split product-reg | shift (`S_NEWTA..S_NEWTC3`); the qsh barrel-shift is isolated into `Y0_r` across `S_NEWT0`/`S_NEWT0B` so retiming cannot fuse it into the squarer (`layernorm_vec.sv:255-275`); the output is a 5-stage pipeline s0..s3 (`layernorm_vec.sv:130-159`, `297-352`) where the 96×32 `prod*gamma` is split into hi/lo 48-bit positional partials (`prod == signed(prod[95:48])*2^48 + prod[47:0]`) so neither half is a 4-DSP cascade (`layernorm_vec.sv:319-341`). Load reduction is a 3-stage skew A/B/C (`layernorm_vec.sv:192-219`).

Protocol: pulse `start`, drive `x_in`/`gamma_in` (each P*32 packed, lane k = [32k+:32]) with `valid_in` for D/P=32 rows; `y_out` (P*64 packed) streams with `y_valid`; `done` pulses at end (`layernorm_vec.sv:22-25`). In `sequencer_vec` the LN feed is FUSED into the producing state (S_EMB/S_RES1/S_RES2) so L_GAM/L_FEED are retired (`sequencer_vec.sv:443-447`). In `nl_engine` LN is a SHARED arbitrated unit hoisted into `sequencer_pp` — the engine drives `ln_req/ln_start_o/ln_vin_o/ln_x_o/ln_g_o` and idle-waits on `ln_gnt` in NL_LGAM/NL_LFEED/NL_LCOLL (`nl_engine.sv:93-102`, `410-441`).

Gate: `run_vec_layernorm.py` prints `LN_VEC_VERDICT bitexact=True mismatches=0/N` (`run_vec_layernorm.py:6,84`); scalar gate `run_layernorm.py:260` also emits worst_step_cosine (informational; LN is exact so cosine=1.0).

## 3. Softmax (scalar, 3-pass, restoring-division reciprocal)

Two RTL variants, both bit-true to `run_softmax.int_softmax_q`:
- **`softmax.sv`** — the shared brick, 3 cycles/element per pass (read->compute->store sub-phases). Instantiated inside `vec_attn.sv:81` (`softmax #(.TMAX(TMAX)) u_sm`), which is the attention used by `sequencer_sb` (per-cohort or shared `vec_attn`, `sequencer_sb.sv:170,175,202`).
- **`softmax_f.sv`** — doc-7 R4c fork with PASS2/PASS3 pipelined to 1 element/cycle (3-stage pipelines, `softmax_f.sv:1-13,49,64`). Instantiated inside `vec_attn_w.sv:68` (`softmax_f #(.TMAX(TMAX)) u_sm`), the attention used by `sequencer_vec` (twin engines u_attnA/u_attnB, `sequencer_vec.sv:329-337`). Emitted probs are bit-identical; only cycle count changes.

Datapath (`softmax.sv:2-24`):
```
PASS1 (S_LOAD): stream T Q8.8 scores in; track running max (runmax init -32768); store in scoremem.
PASS2 (S_EXP) : zq = clip(score[j]-max, -4096, 0); e = explut[-zq]; store expmem; sum += e.
RECIP (S_RECIP): r = floor(2^40 / sum) by restoring long division.
PASS3 (S_NORM): prob[j] = (expmem[j]*r) >> 20  -> Q1.20.
```
- **exp LUT**: 4097 entries, `explut[i] = round(exp(-i/256)*2^20)` Q1.20, `explut[0]=2^20`, from `exp_lut.mem` ($readmemh -> BRAM ROM) (`softmax.sv:49-51`, `run_softmax.py:40-43`). Input clip range is [-16,0] in Q8.8 -> index 0..4096 (`run_softmax.py:33-35` ZMAX=16, NZ=4096).
- **z-index** (`softmax.sv:98-102`): `zq_full = sc_v - runmax` (<=0 for the max); `zidx = clip` then `-zq` in 0..4096.
- **sum**: 29-bit accumulator (<= 256*2^20) (`softmax.sv:74`).
- **reciprocal** (`softmax.sv:174-235`): restoring long division of dividend 2^40 by `sum`, UNROLLED `DIV_STEPS` restoring steps/cycle (radix-2^DIV_STEPS). DIV_STEPS=3 = radix-8 (14 cyc, deepest CARRY8, the 5ns path); DIV_STEPS=2 = radix-4 (21 cyc, ~2/3 depth); DIV_STEPS=1 = 41 cyc. Emitted quotient bits b40..b0 are IDENTICAL for any DIV_STEPS (`softmax.sv:31-36`). Default is **2** in both `softmax.sv:36` and `softmax_f.sv:18` (WARNING: the in-file comment at `softmax.sv:35` still reads "Default 3 preserves the shipped radix-8" — that comment is STALE; the actual parameter default one line down at `:36` is 2). A `dont_touch` `recip_q` FF->FF copy stops Vivado retiming the divider's combinational last step into the S_NORM DSP A-input (`softmax.sv:80-90,248`).
- **prob out**: `prob = er[40:20]` (>>20, keep 21 bits) (`softmax.sv:256`).

Softmax is scalar by nature: `in_valid`/`score` one word in, `out_valid`/`prob` one word out; latency ~3T+41+few (`softmax.sv:22-24`).

Gate: `run_softmax.py` — MODULE gate `SOFTMAX_VERDICT bitexact mismatches=0` over real+synthetic score rows (`run_softmax.py:118-217`), plus the BINDING token-stream gate (plug int_softmax into goformer, greedy decode identical to float over 60 tokens, `run_softmax.py:220-257`).

## 4. GELU (Q4.12 LUT + 3-bit linear interpolation)

- **`gelu_lut.sv`** — single lane. 8192-entry signed Q4.12 LUT from `gelu_lut.mem`; index = `(x + 0x8000)` top 13 bits, low 3 bits are interp fraction; `y = lut[i] + ((lut[i+1]-lut[i])*f) >>> 3`; 3-cycle latency (`gelu_lut.sv:1-40`).
- **`gelu_lut2.sv`** — two lanes sharing one even/odd-banked LUT pair (`lut_e[i]=lut[2i]`, `lut_o[i]=lut[2i+1]`), 2 lanes per BRAM pair instead of 1, bit-true, 3-cycle latency (`gelu_lut2.sv:1-9,20-25`). Reads adjacent (i,i+1) = one even + one odd, mux on `idx[0]` (`gelu_lut2.sv:35-67`). Loaded from `gelu_lut_e.mem`/`gelu_lut_o.mem`.
- **`vec_gelu.sv`** — P-wide wrapper: P/2 `gelu_lut2` instances (`vec_gelu.sv:29-39`); accept P-wide `x` (lane k = [16k+:16]) every cycle, emit P-wide `y` 3 cycles later via a 3-deep valid shift register (`vec_gelu.sv:41-46`). Instantiated by `sequencer_sb.sv:296,302` (u_gelu, u_gelu2) and `sequencer_vec.sv:252` (u_gelu).

Reference `gelu_q` (`run_gelu.py:33-41`): `u = x+32768; i = u>>3; f = u&7; y = lut[i] + ((lut[i1]-lut[i])*f >> 3)`. Gate `run_gelu.py`: `GELU_VERDICT bitexact mismatches=0` (`run_gelu.py:114`). NOTE: the end-to-end raw-logit cosine (~0.9994) is a BRITTLE proxy for this char model (mlp_proj INT8 requant half-boundaries); the binding gate is the token stream, which is identical (`run_gelu.py:108-113`).

In `sequencer_vec` the GELU feed converts the residual to a Q4.12 sat16 word (`mword`), pushes through vec_gelu, and collects with a `<<< (LN_OUT_FRAC-GELU_FRAC)` = <<10 rescale to Q.22 (`sequencer_vec.sv:697,709-712`).

## 5. Dequant (`vec_dequant.sv`, P-wide per-channel)

Bit-exact to the scalar dequant `seq_ref._dequant_to_q` (`vec_dequant.sv:1-12`, `seq_ref.py:179-187`). Per element:
```
dq_prod = signed(gemvy_int32) * signed({1'b0, mant_24})   // mant zero-extended, up to 96b
dq_shv  = signed(exp_8) + frac                            // frac RUNTIME
dq_val  = (dq_shv>=0) ? dq_prod <<< dq_shv                // left
                      : rsh_round(dq_prod, -dq_shv)       // round-half-away-from-zero
out     = dq_val[31:0]
```
(`vec_dequant.sv:88-151`). `frac` is a runtime signed-7-bit input: 16 (qkv/kv), 25 (resid/proj/mp), 12 (gelu) — see the descriptor `d_frac` values `7'd16/7'd25/7'd12` in `nl_engine.sv:446,483,507,514,545`. Round-half-away-from-zero: for the right-shift case, `half = 1<<<(rs-1)`, `op = (v>=0)? v+half : (-v)+half`, negate after if v<0 (`vec_dequant.sv:112-131`). 3-cycle latency (product / round-bias-add / shift+publish), throughput 1 P-wide row/cycle (`vec_dequant.sv:20-37,155-159`). Inputs are PACKED buses (lane k in [k*W+:W]), each lane copies its slice to a plain reg before arithmetic (iverilog-safe, `vec_dequant.sv:64-92`). Instantiated in `sequencer_gemm.sv:190`, `sequencer_pp.sv:366`, and (SHARED, hoisted) referenced by `cohort_engine.sv:122` / `sequencer_sb.sv:216,255`.

## 6. Argmax + on-chip Gumbel-max sampling

### 6a. `nl_engine` / `cohort_engine` (split-brain `sequencer_sb`) — GREEDY ONLY
The split-brain NL FSM does a plain P-wide argmax over VOCAB=193 head logits with NO sampling. States NL_HEAD/NL_WHEAD/NL_ARG/NL_ARG2 (`nl_engine.sv:543-601`): read head_bank P logits/row; stage a 4-pair max (P=8 -> 4 pairs, `nl_engine.sv:564-576`); reduce to a word-max then running best (`nl_engine.sv:579-591`); indices >= VOCAB forced to `32'sh80000000` (`nl_engine.sv:568-569`); first index wins ties (strict `>`). There is NO seed/xorshift/gumbel port in `sequencer_sb.sv` — grep confirms none. So the MEASURED 59,965.5 tok/s record engine is greedy argmax.

### 6b. `sequencer_vec` (doc-7 faithful stream) — ON-CHIP GUMBEL-MAX (landed on silicon)
The single source of truth is `gumbel.py`, shared by RTL (`gumbel_lut.mem`) and the bit-exact Python golden `GumbelRng` (`gumbel.py:9-16`). The trick: sampling softmax(logit/T) == argmax_i(logit_i + T*g_i), g_i = -log(-log(u_i)) (`gumbel.py:3-8`). So NO on-chip softmax over logits and NO host logit readback: add precomputed noise to each head logit; the argmax winner IS the sample.

- **Noise LUT**: `gumbel_lut[idx] = round(temp*(-log(-log((idx+0.5)/1024)))*2^25)`, 1024 signed Q.25 entries (one BRAM tile), GLBITS=10, FRAC=25, temp baked at **0.85** (`gumbel.py:23,26,30-40`). RTL: `gumbel_lut [0:1023]` from `gumbel_lut.mem` (`sequencer_vec.sv:190-192`).
- **RNG**: 32-bit xorshift32 (`x^=x<<13; x^=x>>17; x^=x<<5`), persistent `rng_state`, one advance per logit. `xorshift32` in `gumbel.py:43-48` == RTL combinational `xs1/xs2/xs_ns` (`sequencer_vec.sv:404-406`). LUT index = `next_state >> 22` (top 10 bits) (`gumbel.py:69`, `sequencer_vec.sv:407` `gpre_idx_w = xs_ns[31:22]`).
- **Seed load**: `input [31:0] seed` + `seed_we`; `if(seed_we) rng_state<=seed; smp_en<=(seed!=0)` — lives OUTSIDE the FSM case (host writes once between GOs) (`sequencer_vec.sv:65-70,480-483`). seed==0 => greedy, bit-exact to old behaviour.
- **Precompute FSM** (rides the head GEMV, armed in S_HEADSET only if smp_en, `sequencer_vec.sv:833-844`): ADVANCE stage steps the xorshift and presents the LUT address; PLACE stage (1 cyc later, LUT read registered) drops `gumbel_lut_r` into lane `gj_d%P` of `gpre_word`, flushing to `gumbel_bank[gj_d/P]` at lane P-1 or the final logit (`sequencer_vec.sv:392-407,942-965`). State advances BEFORE each logit's noise -> bit-exact to `GumbelRng.sample_token` (`gumbel.py:70-78`).
- **Argmax fold-in** (S_ARGMAX, `sequencer_vec.sv:846-888`): compare path WIDENED to signed 34-bit (logit int32 + noise int32 can overflow int32); each logit gets `+ (smp_en ? gumbel_bank noise : 0)` BEFORE the pair-max (`sequencer_vec.sv:859-862`); `NEG_INF34 = {1'b1,33'b0}` for idx>=VOCAB; first index wins ties. 3-stage pipeline A/B/C (`sequencer_vec.sv:416-425,853-887`). Winner -> `tok_out`.

### 6c. Where temp / top-k / greedy live per engine (board driver `pl_kv256.py`)
- `R_SEED = 0x30`: nonzero => on-chip Gumbel-max (`pl_kv256.py:21,54`).
- **temp==0 => greedy**: fast bit-exact, TOK_OUT with no readback (`pl_kv256.py:111-112,157`).
- **ON-CHIP path (default, temp>0, host_sample=False)**: write ONE random nonzero seed to R_SEED; fabric samples via Gumbel-max; on-chip temp is FIXED at the LUT's baked 0.85 — the runtime `temp` knob does NOT change on-chip temperature, it only gates on/off (`pl_kv256.py:113-119,158-160`, `_arm_onchip_seed` at `pl_kv256.py:216-224`).
- **HOST path (host_sample=True, fallback)**: read VOCAB head logits back through the debug readback port and do temperature + top-k softmax sampling on the A53 (`_sample` at `pl_kv256.py:206-215`); honours runtime temp/top_k. Used when a bitstream lacks R_SEED or to A/B the two samplers (`pl_kv256.py:119-122,161-162,228-229`).

So: **top-k is HOST-ONLY** (no on-chip top-k). **Temperature is on-chip only as a baked LUT constant (0.85)**; runtime temperature is a host-path feature. Greedy is native to both engines.

Gate: `run_vec_kv.py` — `_sample_stream` drives one `GumbelRng(seed)` across the sampled passes and checks the fabric token stream matches; seed==0 gives the greedy gold (`run_vec_kv.py:44-65,71-81`); writes `gumbel_lut.mem` VERBATIM from `gumbel.make_gumbel_lut()` (`run_vec_kv.py:103-108`).

## 7. Diagram — per-token non-linear datapath (sequencer_vec)

```
 tok_emb+pos_emb (Q6.25) --> xres_bank
        |
   [layernorm_vec] gamma Q4.20 --> y Q.22
        |
   [act-quant] inv_sact 2^40 --> INT8
        |
   (GEMV qkv, dequant frac=16 -> Q.16 q/k/v)
        |
   [vec_attn_w]: score Q8.8 = rsh_round(sum q*k, 27) sat16
        |           -> [softmax_f] exp LUT Q1.20, r=2^40/sum -> prob Q1.20
        |           -> ctx = rsh_round(sum prob*v, 11) Q6.25
   (residual add) --> [layernorm_vec] --> [GEMV fc, dequant frac=12]
        |                                   -> [vec_gelu] Q4.12 -> <<10 -> Q.22
   (GEMV mlp_proj, dequant frac=25) --> residual add
        | (x4 blocks)
   [layernorm_vec LN_f] --> [GEMV head] --> head logits Q6.25
        |
   [argmax + Gumbel noise (gumbel_bank, +T*g)] --> tok_out
```

## 8. Acceptance bar
"Bit-honest before fast" (`seq_ref.py:12`): LayerNorm/softmax/dequant/argmax are BIT-EXACT (mismatches=0); transcendental blocks target cosine > 0.9999, but the actual binding gate is TOKEN-STREAM IDENTITY (same greedy decode as the float reference), because the raw-logit cosine is brittle at INT8 requant half-boundaries (`run_gelu.py:108-113`, `run_softmax.py:16-21`, `seq_ref.py:9-13`).

**Parameters**

| name | meaning | live | range | where |
|---|---|---|---|---|
| `RESID_FRAC` | Residual stream fixed-point fraction (Q6.25). Also the dequant target frac for proj/mp/head. | 25 | constant 25 | seq_ref.py:56, nl_engine.sv:37 |
| `LN_OUT_FRAC / OUT_FRAC` | LayerNorm output fraction (Q.22). | 22 | constant 22 | layernorm.sv:40, nl_engine.sv:38 |
| `OUT_SH` | LN final shift = QX+Y_FRAC+G_FRAC-OUT_FRAC. | 49 | derived | layernorm_vec.sv:52 |
| `VFRAC` | q/k/v stored fraction (Q.16). Also dequant frac for qkv/kv. | 16 | constant 16 | seq_ref.py:62, vec_attn.sv:56 |
| `SCORE_FRAC` | Attention score fraction (Q8.8) feeding softmax. | 8 | constant 8 | softmax.sv:42, vec_attn.sv:58 |
| `PROB_FRAC / EXP_FRAC` | Softmax prob + exp LUT fraction (Q1.20). | 20 | constant 20 | softmax.sv:46, run_softmax.py:31-32 |
| `RECIP_R` | Reciprocal dividend exponent: r = floor(2^40/sum). | 40 | constant 40 | run_softmax.py:34, softmax.sv:12 |
| `GELU_FRAC / FRAC` | GELU I/O fraction (Q4.12). | 12 | constant 12 | run_gelu.py:21, gelu_lut.sv:3 |
| `ISH / INV_SACT_SH` | Activation-quant reciprocal shift: inv_sact = round(2^40/s_act). | 40 | constant 40 | nl_engine.sv:41, seq_ref.py:158 |
| `DIV_STEPS` | Restoring-division steps/cycle (radix-2^DIV_STEPS); 3=radix-8 14cyc, 2=radix-4 21cyc. Bit-identical quotient for any value. | 2 (default in both) | 1..3 | softmax.sv:36, softmax_f.sv:18 |
| `P / LANES` | Vector lanes for LN/GELU/dequant/argmax (P per cycle). D must divide by P. | 8 | 8 typical (128 lanes) | nl_engine.sv:28-29, layernorm_vec.sv:29 |
| `TEMP (Gumbel LUT)` | Baked on-chip sampling temperature; the ONLY on-chip temp (runtime temp knob does not change it). | 0.85 | constant 0.85 | gumbel.py:23 |
| `GLBITS` | Gumbel noise LUT address bits (1024 entries = one BRAM); index = xorshift>>22. | 10 | constant 10 | gumbel.py:24 |
| `Gumbel FRAC` | Gumbel noise fraction (Q.25), same scale as head logits. | 25 | constant 25 | gumbel.py:26 |
| `VOCAB` | Vocabulary size = number of head logits argmax'd / noise draws per token. | 193 | constant 193 | nl_engine.sv:31, gumbel.py:27 |
| `EPS_A` | LayerNorm epsilon in Q.26 = round(1e-5 * 2^26). | 671 | constant 671 | layernorm_vec.sv:53 |
| `SEED_IDX_BITS` | rsqrt seed LUT address bits (64-entry mantissa table). | 6 | constant 6 | layernorm.sv:42, run_layernorm.py:37 |
| `NZ / ZMAX` | exp LUT input clip range [-16,0] Q8.8 -> 4097 entries. | 4096 | constant 4096 | run_softmax.py:32-35 |

**Key facts**

- The residual stream is signed Q6.25; the whole non-linear subsystem converts between this and per-block formats (LN gamma Q4.20 -> y Q.22; scores Q8.8; probs Q1.20; GELU Q4.12) (seq_ref.py:19-33).
- LayerNorm is gamma-only with algebraic variance ssq = sum(x*x) - 2*mean*sum + D*mean^2 for floored mean = sum>>>8 — an EXACT integer identity, not an approximation (layernorm_vec.sv:12-20).
- rsqrt uses a 64-entry Q1.16 seed LUT (seed.mem) keyed on 6 mantissa bits below A's MSB, exponent by shifts + a *sqrt2 (46341 Q15) for odd E, then 2 Newton steps y<-y*(1.5-0.5*A*y*y) (layernorm_vec.sv:74-82, run_layernorm.py:45-72).
- layernorm_vec is P-wide I/O and is the LayerNorm used by BOTH live engines (sequencer_sb.sv:149, sequencer_vec.sv:209). Softmax is scalar (1 score in / 1 prob out per cycle) — NOT vectorized.
- Softmax: exp LUT is 4097 entries exp(-i/256) Q1.20 (exp_lut.mem), z clipped to [-16,0]; reciprocal r=floor(2^40/sum) by restoring long division, DIV_STEPS=2 default (radix-4, 21 cyc), quotient bit-identical for any DIV_STEPS (softmax.sv:49-51,174-235).
- softmax.sv (3 cyc/elem) is used via vec_attn by sequencer_sb; softmax_f.sv (1 elem/cycle pipelined, bit-identical probs) is used via vec_attn_w by sequencer_vec (vec_attn.sv:81, vec_attn_w.sv:68).
- GELU is an 8192-entry Q4.12 LUT with 3-bit linear interpolation; vec_gelu is P-wide using P/2 gelu_lut2 paired-LUT cores (2 lanes per BRAM pair), 3-cycle latency (gelu_lut.sv:1-40, gelu_lut2.sv:1-9, vec_gelu.sv:29-46).
- vec_dequant is P-wide per-channel: dq_prod = gemvy*mant(24b), shift = exp+frac (frac runtime 16/25/12), round-half-away-from-zero on right shift; 3-cycle latency (vec_dequant.sv:1-12,88-151).
- On-chip Gumbel-max sampling lives ONLY in sequencer_vec (doc-7 faithful stream, N=1), NOT in the split-brain record engine sequencer_sb which is greedy-argmax only (sequencer_vec.sv:185-192, no gumbel/seed in sequencer_sb.sv).
- Gumbel-max: argmax(logit + T*g) == softmax(logit/T) sample; fabric adds precomputed noise (1024-entry Q.25 LUT, temp baked at 0.85) during the head GEMV, so no on-chip softmax over logits and no host logit readback; host writes ONE 32-bit seed, seed==0 => greedy (gumbel.py:3-16,23, sequencer_vec.sv:480-483).
- The RNG is xorshift32, persistent across a request's tokens, advancing once per logit BEFORE its noise — bit-exact between gumbel.GumbelRng and the RTL precompute FSM (gumbel.py:43-48,70-78, sequencer_vec.sv:404-407,942-965).
- Temperature is on-chip only as a baked LUT constant (0.85); the runtime temp knob and top-k are HOST-side only (read VOCAB logits back, softmax/top-k on the A53) (pl_kv256.py:104-122,206-215).
- Argmax with Gumbel widens the compare path to signed-34 bits (int32 logit + int32 noise can overflow), NEG_INF34 for idx>=VOCAB, first index wins ties (sequencer_vec.sv:409-425,853-887).
- Acceptance bar: LN/softmax/dequant/argmax are bit-exact (mismatches=0); transcendentals target cosine>0.9999 but the BINDING gate is token-stream identity because raw-logit cosine (~0.9994 for GELU) is brittle at INT8 requant half-boundaries (run_gelu.py:108-113, run_softmax.py:16-21).
- Activation quant to INT8: inv_sact=round(2^40/s_act), INT8=sat(rsh_round(y_q22*inv_sact, 22+40), -128,127); the 2^40 must match RTL ISH=40 (seq_ref.py:158-172, nl_engine.sv:41).

**Files**

- `fabric/stage3/rtl/layernorm_vec.sv` — P-wide-I/O gamma-only LayerNorm (Q6.25 x, Q4.20 gamma -> Q.22 y); the live LN in both engines. Algebraic variance + pipelined rsqrt/output.
- `fabric/stage3/rtl/layernorm.sv` — Scalar serial LayerNorm reference core (bit-true spec; documented, not in live datapath).
- `fabric/stage3/rtl/layernorm_par.sv` — Throughput LayerNorm variant (sum + sum(x*x) accumulated during load).
- `fabric/stage3/rtl/softmax.sv` — Shared softmax brick, 3 cyc/elem; Q8.8 scores -> Q1.20 probs; 4097-entry exp LUT + 2^40/sum restoring-division reciprocal. Used by vec_attn (sequencer_sb).
- `fabric/stage3/rtl/softmax_f.sv` — softmax fork pipelined to 1 elem/cycle (doc-7 R4c), bit-identical probs. Used by vec_attn_w (sequencer_vec).
- `fabric/stage3/rtl/gelu_lut.sv` — Single-lane GELU: 8192-entry Q4.12 LUT + 3-bit linear interp, 3-cyc latency.
- `fabric/stage3/rtl/gelu_lut2.sv` — Two-lane GELU sharing an even/odd-banked LUT pair (BRAM-saving), bit-true to gelu_lut.
- `fabric/stage3/rtl/vec_gelu.sv` — P-wide GELU wrapper: P/2 gelu_lut2 cores, 3-deep valid SR. Instantiated in both live engines.
- `fabric/stage3/rtl/vec_dequant.sv` — P-wide per-channel INT32->Q dequant (mant 24b, exp 8b, runtime frac), round-half-away-from-zero, 3-cyc latency.
- `fabric/stage3/rtl/nl_engine.sv` — Split-brain per-cohort NL FSM: LN feed/collect, attention feed, residuals, and GREEDY P-wide argmax (no sampling). Drives shared LN/attn ports.
- `fabric/stage3/rtl/sequencer_vec.sv` — Doc-7 faithful-stream engine: instantiates layernorm_vec, vec_attn_w, vec_gelu, vec_dequant AND the on-chip Gumbel-max sampler (xorshift + noise LUT + argmax fold-in).
- `fabric/stage3/rtl/sequencer_sb.sv` — Split-brain N=16 record engine: instantiates layernorm_vec, vec_attn (softmax.sv), vec_gelu, vec_dequant; greedy argmax via cohort_engine/nl_engine, no Gumbel.
- `fabric/stage3/rtl/vec_attn.sv` — Single-head P-wide attention containing softmax.sv; score Q8.8 = rsh_round(sum q*k,27) sat16, ctx = rsh_round(sum prob*v,11) Q6.25.
- `fabric/stage3/rtl/vec_attn_w.sv` — Attention variant containing softmax_f (1 elem/cycle); twin instances in sequencer_vec.
- `fabric/stage3/gumbel.py` — Single source of truth for on-chip Gumbel-max: make_gumbel_lut (temp 0.85, Q.25, 1024 entries), xorshift32, and the bit-exact GumbelRng golden reference.
- `fabric/stage3/seq_ref.py` — Per-phase sequencer reference pinning all Q-formats (RESID_FRAC=25, VFRAC=16, ISH=40, etc.) and the exact integer dequant/act-quant/argmax the RTL reproduces.
- `fabric/stage3/run_softmax.py` — Softmax gate: exp_table, int_softmax_q reference, SOFTMAX_VERDICT bit-true + token-stream identity gate.
- `fabric/stage3/run_gelu.py` — GELU gate: gelu_table/gelu_q reference, GELU_VERDICT bit-true (+ brittle end-to-end cosine proxy note).
- `fabric/stage3/run_layernorm.py` — LayerNorm gate + seed_table (64-entry rsqrt seed LUT), LN_VERDICT.
- `fabric/stage3/run_vec_layernorm.py` — P-wide LayerNorm gate: LN_VEC_VERDICT bitexact mismatches=0/N.
- `fabric/stage3/run_vec_kv.py` — Faithful-stream KV gate including the Gumbel sampling path: writes gumbel_lut.mem, checks fabric token stream vs GumbelRng golden.
- `fabric/stage3/board/pl_kv256.py` — On-board driver: R_SEED sampling wiring; greedy/on-chip-Gumbel(temp fixed 0.85)/host-softmax+top-k path selection.

**Gotchas**

- The MEASURED 59,965.5 tok/s record engine (sequencer_sb, split-brain N=16) is GREEDY-ARGMAX ONLY — it has no seed/xorshift/Gumbel port. On-chip sampling is a sequencer_vec (doc-7 faithful stream, N=1) feature. Do not assume the record engine samples.
- On-chip temperature is FIXED at 0.85 (baked into gumbel_lut.mem). The runtime `temp` knob in pl_kv256 only gates sampling on/off for the on-chip path; changing on-chip temperature requires rebaking the LUT. Runtime temperature + top-k are HOST-side only (require reading VOCAB logits back through the debug port).
- Softmax is inherently scalar (sequential running-max, exp-accumulate, and 2^40/sum reciprocal reductions). softmax_f only pipelines the sub-phases to 1 elem/cycle; there is no P-wide softmax. Cycle budgets must treat it as ~T cycles/pass, not T/P.
- The exp LUT has 4097 entries (index 0..4096 = clip range [-16,0] Q8.8), NOT 4096. The gumbel LUT has 1024 (10-bit index = xorshift>>22). Off-by-one on either breaks $readmemh alignment.
- DIV_STEPS default is 2 (radix-4, 21 cycles) in both softmax.sv and softmax_f.sv, not the historical radix-8 (3). The emitted quotient is bit-identical across DIV_STEPS, so gates pass regardless, but timing/cycle-count differ.
- The recip_q FF->FF copy in softmax is dont_touch and load-bearing for timing: without it Vivado retimes the divider's combinational last radix step into the S_NORM DSP A-input (the reported WNS path). Do not merge it away.
- Gumbel bit-exactness depends on the xorshift advancing BEFORE each logit's noise and index = next_state>>22. The RTL precompute is a 2-stage advance/place pipeline (LUT read registered); the gate GumbelRng must mirror this exact ordering (state advances once per logit, VOCAB=193 draws/token, persisting across tokens).
- The binding correctness gate for transcendentals is TOKEN-STREAM IDENTITY, not cosine. The GELU end-to-end raw-logit cosine is only ~0.9994 (below the stated 0.9999) because a few elements sit on mlp_proj INT8 requant half-boundaries; this is expected and the token stream is still identical. Do not 'fix' the GELU LUT to chase the cosine.
- layernorm_vec's y_out is P*64 packed (each Q.22 sign-extended into 64b) even though the value fits 32b — collectors take [pp*64 +: 32] (nl_engine.sv:426-427). vec_dequant/vec_gelu buses are P*32 / P*16. Bus widths differ per block.
- The activation-quant 2^40 (INV_SACT_SH=40) and the softmax reciprocal 2^40 (RECIP_R=40) are unrelated constants that happen to share the value; and ISH=40 in nl_engine is the act-quant shift. Keep them conceptually separate.

**Open questions**

- I did not find any on-chip top-k or on-chip runtime-temperature capability — both appear host-side only (pl_kv256._sample). If a bitstream variant with on-chip top-k exists it is not in the files reviewed.
- I did not measure/verify the exact cycle cost of the Gumbel precompute overlap with the head GEMV on silicon (the code claims it 'rides' the head GEMV with no extra read ports, sequencer_vec.sv:188-189); whether it ever stalls S_ARGMAX (gated on gpre_done) for small VOCAB vs GEMV length was not timing-analyzed.
- The task named layernorm_par.sv as a primary file but neither live engine instantiates it (only layernorm_vec is wired into sequencer_sb/sequencer_vec). Its current role appears to be a documented throughput reference; whether any tb/build still uses it was not exhaustively checked.
- I did not confirm on-silicon that the Gumbel sampler is 'recently landed' beyond the RTL + gate + board-driver support being present and bit-exact in sim; no MEASURED board tok/s or acceptance log for the sampled path was located in the files I read.
- Whether sequencer_sb could be given the Gumbel port (the split-brain record engine currently cannot sample) is not addressed in the reviewed RTL; the two features (max throughput vs on-chip sampling) currently live in different engines.

---

## Board drivers, /dev/mem register map & A53 runtime (Kevin-on-Kria subsystem 5)

This subsystem is everything the ARM A53 (Cortex-A53 PS on the KV260) does to drive the PL fabric that holds the goformer. There is NO Vitis/PetaLinux driver stack here — every "driver" is a small Python module that mmaps `/dev/mem` at the AXI-Lite slave base `0xA0000000` and pokes 32-bit registers directly through a numpy `uint32` view. Each engine's bitstream exposes a control/status register file plus a small readback port; the Python driver streams the INT4 weight image in once, then loops writing token ids and pulsing GO, polling a DONE bit, and reading back the argmax token id and a fabric cycle counter. The cycle counter over the forced PL clock is the bit-honest tok/s number, but only reported when the fabric token(s) match the `seq_ref` golden reference (the gate).

There are four distinct bitstream families with THREE distinct register maps, and this is the biggest trap in the subsystem: (1) the baseline `SEQR/SQRF` map (`pl_seq_chat.py`) puts prompt-write, token-stream drain, NGEN, CYCLES at 0x30, IDCODE at 0x34; (2) the P-wide single-stream `SQRV` map (`pl_seq_vec.py`/`pl_seq_kv.py`/`pl_kv256.py`, driving `sequencer_vec`) with CYCLES at 0x28, IDCODE at 0x2C, plus a 4-bit `rd_sel` readback mux, a `dbg_stop` halt field, and a write-only SEED register for on-chip sampling; and (3) the N=16 batched map `SQ16`/`SQSB` (`pl_seq_pp16.py`/`pl_seq_sb.py`, and the live split-brain record engine) which scatters 16 TOK_ID and 16 TOK_OUT slots across the register file and adds an EDATA register for streaming the token/position embedding tables (which the SQRV builds instead bake into the bitstream). Reading the wrong map against a bitstream reads/writes garbage, so every driver verifies IDCODE first and aborts on mismatch.

The A53 runtime that fronts the public demo is `webchat/demo/a53_daemon.py`: a thin asyncio TCP daemon that owns the device and speaks a length-prefixed-JSON (or msgpack) RPC. The i7 "Precision" box runs the web server with a `TcpPLBackend` (defined in the same file so the wire format has one source of truth) and sends `{prompts:[...]}` batches; the daemon fires one PL submission and returns `{completions, busy_ms, fabric_ms, tokens, passes}`. Three interchangeable engines are selected at daemon startup by `make_device`: `t1` (16-stream single-token PL pass, only the first token is model-faithful), `kv` (host-side KV decoder over a numpy/PL/C GEMV backend), and `kv256` (the doc-7 fabric-resident faithful single-stream KV decode). All generation/window/sampling parameters (gen_chars, tmax, lanes, fclk, temp, top_k) are fixed at device construction — there is NO per-request override; a request carries only the prompt text (plus an optional stream flag).

The other load-bearing runtime fact is the PL clock. A flat `fpgautil -b` bitstream load does NOT apply the block design's `PL0_REF_CTRL FREQMHZ`, so `pl0_ref` stays at the system default (~100 MHz). A design timing-closed at, say, 40 MHz then violates setup on the wide arithmetic and produces non-deterministic garbage — while the short MMIO/cycle-counter paths still work, which is exactly why a broken run can show a plausible CYCLES value with random compute. Every driver therefore forces and verifies the clock via `/sys/devices/platform/fclk0/set_rate` before trusting any output, and computes tok/s from the read-back actual rate (the PLL quantises to discrete divisors).

## 1. The `/dev/mem` device object (common to every driver)

Every driver defines the same tiny AXI4-Lite master. Canonical copy in `fabric/stage3/board/pl_seq_chat.py:110-131` (class `Seqr`), duplicated verbatim as `Dev` in `pl_seq_vec.py:43-66`, `pl_seq_kv.py:54-65`, `pl_seq_sb.py:52-63`, `pl_seq_pp16.py:53-64`, `pl_kv256.py:59-72`:

```
self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
self.mm = mmap.mmap(self.fd, 0x1000, MAP_SHARED, PROT_READ|PROT_WRITE, offset=base)  # base=0xA0000000
self.reg = np.frombuffer(self.mm, dtype=np.uint32)     # 0x1000/4 = 1024 words
def wr(self, off, val): self.reg[off>>2] = np.uint32(val & 0xFFFFFFFF)   # ONE 32-bit store
def rd(self, off):      return int(self.reg[off>>2])
```

Load-bearing rules baked into the comments:
- **Only 32-bit stores.** Byte-slice writes corrupt the W_DATA assembler stream, so never do them (`pl_seq_chat.py:111-112`, `:120-121`).
- **Drop the numpy view before `mm.close()`.** `np.frombuffer` keeps an exported pointer, so `close()` sets `self.reg = None` first or `mm.close()` raises `BufferError` (`pl_seq_chat.py:127-131`).
- Window is exactly one 4 KiB page (`0x1000`), base `0xA0000000` (`pl_seq_chat.py:61`, all drivers).

## 2. The three register maps (verified against RTL wrappers)

### 2a. Baseline `SEQR/SQRF` map — `pl_seq_chat.py` (drives `gemv_axi_seq.v` / `gemv_axi_seq_fast.v`)
Register byte offsets at `pl_seq_chat.py:103-107`, documented `:15-26`:

| off | name | dir | meaning |
|-----|------|-----|---------|
| 0x00 | CTRL | W | b0=go pulse, b1=wl_rst, b2=soft_reset |
| 0x04 | STATUS | R | b0=done_latched, b1=core_busy |
| 0x10 | W_DATA | W | 32-bit weight chunk (pulses wl_we); 64-bit word = low32 then high32 |
| 0x14 | PL_DATA | W | prompt token id (pulses prompt write @ PL_ADDR) |
| 0x18 | PL_ADDR | W | prompt slot address |
| 0x1C / 0x20 | RD_ADDR / RD_DATA | W/R | residual-x readback (Q6.25) |
| 0x24 / 0x28 | TS_ADDR / TS_DATA | W/R | emitted token id (low 9 bits) |
| 0x2C | NGEN | R | number of generated tokens |
| **0x30** | **CYCLES** | R | run cycle count latched at done |
| **0x34** | **IDCODE** | R | 0x53455152 "SEQR" or 0x53515246 "SQRF" |

This is the ONLY map that streams the prompt into a slot array (PL_ADDR/PL_DATA) and drains an output token *array* (TS_ADDR/TS_DATA) — it does a full NGEN=8-token autoregressive run inside one GO. Note CYCLES/IDCODE sit two slots higher than in the other maps.

### 2b. P-wide single-stream `SQRV` map — `pl_seq_vec.py` / `pl_seq_kv.py` / `pl_kv256.py` (drives `sequencer_vec` via `gemv_axi_seq_vec.v`)
Offsets at `pl_seq_vec.py:37-40`; RTL decode confirmed in `gemv_axi_seq_vec.v:75-84` (write) and `:100-108` (read):

| off | idx | name | dir | meaning (RTL line) |
|-----|-----|------|-----|--------|
| 0x00 | 6'h0 | CTRL | W | b0 go, b1 wl_rst, b2 soft_reset, **b4:3 dbg_stop** (`:76`) |
| 0x04 | 6'h1 | STATUS | R | {30'b0, core_busy, done_latched} (`:101`) |
| 0x08 | 6'h2 | TOK_ID | W | [8:0] (`:77`) |
| 0x0C | 6'h3 | POS | W | [8:0] (`:78`) |
| 0x10 | 6'h4 | W_DATA | W | pulses wl_we, wl_data=WDATA (`:79`) |
| 0x14 | 6'h5 | RD_SEL | W | [3:0] readback bank select (`:80`) |
| 0x18 | 6'h6 | RD_ADDR | W | [10:0] readback element (`:81`) |
| 0x1C | 6'h7 | RD_DATA_LO | R | core_rd_data[31:0] (`:102`) |
| 0x20 | 6'h8 | RD_DATA_HI | R | core_rd_data[63:32] (`:103`) |
| 0x24 | 6'h9 | TOK_OUT | R | {23'b0, core_tok_out} (`:104`) |
| 0x28 | 6'hA | CYCLES | R | cycles_latched (`:105`) |
| 0x2C | 6'hB | IDCODE | R | 0x53515256 "SQRV" (`:106`) |
| 0x30 | 6'hC | SEED | W | writes seed + pulses seed_we; nonzero => on-chip Gumbel-max sampling, persists across GOs (`:82`, `:13`) |

The SEED register only exists in this map and only `pl_kv256.py` uses it (`R_SEED=0x30`, `pl_kv256.py:54`).

### 2c. N=16 batched `SQ16`/`SQSB` map — `pl_seq_pp16.py` / `pl_seq_sb.py` (drives `sequencer_sb` via `gemv_axi_seq_sb.v`)
The 16 token slots are *scattered* across the register file. `R_TOKID` / `R_TOKOUT` lists at `pl_seq_sb.py:38-41` (identical in `pl_seq_pp16.py:43-46`):

```
R_TOKID  = [0x08,0x30,0x34,0x38, 0x50,0x54,0x58,0x5C, 0x80,0x84,0x88,0x8C, 0x90,0x94,0x98,0x9C]
R_TOKOUT = [0x24,0x3C,0x40,0x44, 0x60,0x64,0x68,0x6C, 0xA0,0xA4,0xA8,0xAC, 0xB0,0xB4,0xB8,0xBC]
```
Confirmed against `gemv_axi_seq_sb.v` write decode (`:84-102`) and read decode (`:120-142`). Other registers: 0x00 CTRL (b0 go, b1 wl_rst, b2 soft_reset, **b4:3 dbg_stop** — `gemv_axi_seq_sb.v:84`), 0x04 STATUS (b0 done, b1 busy — `:120`), 0x0C POS (`:86`), 0x10 W_DATA/wl_we (`:87`), 0x14 RD_SEL, 0x18 RD_ADDR, 0x1C/0x20 RD_LO/HI, 0x28 CYCLES (6'h0A, `:124`), 0x2C IDCODE (SQSB=0x53515342 `:125`; SQ16=0x53513136), **0x48 RD_STREAM** (SQ16 only, `pl_seq_pp16.py:47`), **0x4C EDATA** (embed loader, pulses el_we, `pl_seq_sb.py:42` / `gemv_axi_seq_sb.v:94` decodes 6'h13 → el_we).

`sequencer_sb` internally runs `N` (8 or 16) real streams but pads its `tok_outs` back to 16 slots so the SQ16 map holds verbatim; slots ≥ N read 0 and TOK_ID writes to them are ignored (`gemv_axi_seq_sb.v:151-159`, `pl_seq_sb.py:6-7`).

**GOTCHA — file mislabel:** `fabric/stage3/board/pl_seq_pp16.py` opens with a docstring titled "pl_seq_gemm — drive the N=4 batch GEMM sequencer" (`pl_seq_pp16.py:1-16`) but the actual code is the N=16 `SQ16` driver (`N=16`, `IDCODE_SQ16=0x53513136`, 16-slot lists). Trust the code, not the header.

## 3. Forcing and verifying the PL clock

`set_and_verify_fclk(target_hz, tol_hz=6e6)` at `pl_seq_chat.py:75-100` is THE single source of truth; every other driver imports it (`pl_seq_vec.py:32`, `pl_seq_kv.py:41`, `pl_seq_sb.py:30`, `pl_kv256.py:45`, `dbg_board_kv.py:27`).

- Writes `int(target_hz)` to `FCLK_SET = "/sys/devices/platform/fclk0/set_rate"` (`pl_seq_chat.py:71`, `:83-84`).
- Reads the value back; the PLL quantises to discrete divisors, so **tok/s must be computed from the readback**, not the requested rate (`:87-91`, `:80-81`).
- Raises `SystemExit` if `|actual - target| > tol_hz` (6 MHz) — a totally failed set (`:93-98`).
- Returns the actual Hz used everywhere for tok/s.

**Why this matters (the deep reason, `pl_seq_chat.py:65-70`):** a flat `fpgautil -b` load does NOT apply the block design's `PL0_REF_CTRL FREQMHZ` (that only happens via the xmutil / device-tree-overlay app flow), so `pl0_ref` stays at the ~100 MHz system default. Running a design timing-closed at 40 MHz at 100 MHz violates setup on the wide arithmetic → non-deterministic garbage. The AXI/MMIO path and the cycle counter survive because their paths are short — which is the trap: CYCLES looks sane while the compute is random. The `--fclk` arg is also the tool used to SWEEP the silicon Fmax upward (highest rate still bit-exact), exploiting the ~1.3–1.76× silicon-over-STA margin.

## 4. Weight boot-stream sequence (identical shape in every driver)

`build_weight_image(intseq, lanes)` (e.g. `pl_seq_vec.py:84-93`, `pl_seq_sb.py:66-73`, `pl_seq_chat.py:143-153`) produces the transposed INT4 image in the EXACT order the sequencer's `w_base` offsets assume:
1. For each of `NLAYER=4` blocks, in order: `qkv`, `proj`, `mlp_fc`, `mlp_proj` (each `pack_banked.pack_transposed(int8_weight, lanes)`).
2. Then `head`.

Boot sequence (e.g. `pl_seq_sb.py:125-132`):
```
wr(CTRL, 0x4); wr(CTRL, 0x0)   # soft_reset assert then release  -> FSM/KV/scratch to defined state
wr(CTRL, 0x2)                  # wl_rst pulse -> reset the load pointer
subw = (lanes*4)//32           # 32-bit W_DATA chunks per wide word (128 lanes -> 16)
for w in words:
    for s in range(subw):
        wr(W_DATA, (w >> (32*s)) & 0xFFFFFFFF)   # LOW chunk first
```
The soft_reset is not optional: without it the first GO after config runs on undefined power-up state → non-deterministic garbage (`pl_seq_chat.py:238-242`).

**Embeddings differ by family:**
- **SQRV vec / kv** (`pl_seq_vec.py`, `pl_seq_kv.py`): tok/pos embed ROMs are `$readmemh`'d at bitstream build time — nothing uploaded (`pl_seq_kv.py:8-9`, `:187-188`).
- **SQRV kv256** (`pl_kv256.py:85-91`): embed tables *ride the weight image's spare depth* via `run_sequencer.wrom_embed_words(intseq, lanes, len(words))` appended to `words` — one shared packer with the sim writer so the two layouts cannot drift.
- **SQ16 / SQSB** (`pl_seq_pp16.py`, `pl_seq_sb.py`): embeddings are streamed at runtime through the **EDATA (0x4C)** register AFTER the weight stream. `build_embed_chunks(intseq, tmax)` (`pl_seq_sb.py:76-84`) emits `tok_emb` then `pos_emb[:tmax]` rounded to **Q6.25** (`* (1<<25)`, `RESID_FRAC=25`) as INT32 chunks, element order matching the P-wide rows. **The `tmax` passed MUST equal the bitstream's TMAX generic** (build arg 5); a mismatch uploads the wrong number of pos words and corrupts every embedding (`pl_seq_sb.py:43-47`, `:79`). `TMAX_DEFAULT=16` for the §24 build (`pl_seq_sb.py:47`); the old SQ16 build used `TMAX=32` (`pl_seq_pp16.py:49`).

## 5. Token generation loops (per engine)

### 5a. Single token — `pl_seq_vec.py` (SQRV)
`pl_seq_vec.py:139-150`: write TOK_ID, write POS, `wr(CTRL, 0x1 | (stop_mode<<3))` (go + dbg_stop), poll `STATUS & 0x1`, read CYCLES + TOK_OUT. tok/s = fclk/cyc, reported only if `TOK_OUT == seq_ref.full_forward_signals(tok)['tok']` (`:148-176`).

### 5b. Faithful multi-token KV — `pl_seq_kv.py` and `pl_kv256.py` (SQRV, KV persists)
The KV banks stay ALIVE between GO pulses (`pl_seq_kv.py:5-8`, `pl_kv256.py:4-10`), so a decode is a train of GOs at increasing POS:
- `_pass`/`_go`: write TOK_ID & POS, `wr(CTRL, 0x1)`, poll DONE, return TOK_OUT + accumulate CYCLES (`pl_seq_kv.py:99-109`, `pl_kv256.py:168-181`).
- Prompt phase: feed prompt ids at pos 0..L-1 (fills the KV window). Generation phase: feed the previous TOK_OUT back at pos L+g (greedy feedback) (`pl_seq_kv.py:112-134`, `pl_kv256.py:244-259`).
- **No reset between requests:** each request restarts at pos 0 and simply OVERWRITES the previous KV; a fresh pass rewrites KV[p] before any pass reads it (tcount = pos+1), so stale rows are never attended (`pl_kv256.py:8-10`, `pl_seq_kv.py:113-117`).
- `pl_seq_kv.py` gates the whole generated stream against `IntKVQSequencer(kbits=8, vbits=8, rotate=False, divfree=True).generate_greedy` (the R0 contract) and runs `--runs 3` requiring all three mutually identical (`:12-14`, `:163-164`, `:191-216`).
- `pl_kv256.py` stop conditions on `_STOPS = (". ","! ","? ","\n")` and `_tidy` trims to one utterance (`:100`, `:279-293`).

### 5c. N=16 parallel — `pl_seq_sb.py` / `pl_seq_pp16.py` (SQSB/SQ16)
Write all N TOK_IDs into the scattered slots, write POS=0, one GO, poll DONE, read N TOK_OUTs + CYCLES (`pl_seq_sb.py:138-149`, `pl_seq_pp16.py:133-144`, `backend_stub._submit_once`). Aggregate tok/s = `N * fclk / cyc`, reported only if all N match `seq_ref.full_forward_signals` per stream (`pl_seq_sb.py:152-157`). `--n` may be 14 (NC=7, the 97%-LUT fit build) or 16 (`:6-7`, `:91`).

## 6. Config wiring — construction-time vs per-request (CONFIRMED: no per-request override)

`PLKV256Device.__init__` (`pl_kv256.py:102-165`) captures ALL knobs once: `gen_chars`, `tmax`, `temp`, `top_k`, `host_sample`, `lanes`, `fclk`, `base`, `poll_timeout`. `_stream_gen` reads `self.gen_chars` / `self.tmax` (`pl_kv256.py:239`, `:252`); `infer(prompts)` and `stream_into(prompt, on_char)` take ONLY prompt text (`:264`, `:296`). The daemon RPC frame carries only `{prompts:[...]}` (+ optional `stream`) — there is **no field for gen length, window, temperature, or top_k** (`a53_daemon.py:238-259`). So a running daemon's generation length/window/sampling are fixed at startup; changing them requires restarting the daemon. `make_device` sets them from argparse once (`a53_daemon.py:289-304`): `gen_chars` default 48 for kv256 / 6 otherwise (`:356-357`), `tmax` default 256 (`:333`), `temp` forced to 0 under `--greedy` (`:302`, `:338-345`), `top_k` default 40 (`:346`).

## 7. Sampling paths in kv256 (temp / top_k)

`_onchip = (temp > 0) and not host_sample` (`pl_kv256.py:126`). Two paths:
- **On-chip Gumbel-max (default when temp>0):** write ONE random nonzero seed to R_SEED once per request, just before the LAST prompt GO, then read TOK_OUT directly — no per-token logit readback (`pl_kv256.py:216-223`, `:247-249`). The noise LUT is baked at temp=0.85 in the bitstream, so on-chip temperature is fixed by the bitstream, not `self.temp` (`:117-120`). Saves the ~58 ms/reply VOCAB-logit `/dev/mem` read.
- **Host sampling (`host_sample=True`, fallback):** `_read_logits` reads the VOCAB head logits back through the debug readback port (rd_sel=8), then `_sample` does temperature + top-k softmax on the A53 honouring `self.temp`/`self.top_k` at runtime (`pl_kv256.py:187-214`, `:228-229`). Used if a bitstream lacks R_SEED or to A/B the samplers.
- temp==0 → greedy argmax straight from TOK_OUT, bit-exact, no readback (`:157-158`).

Note the daemon's own `--temp` default is 0.0 (`a53_daemon.py:341`) i.e. greedy unless raised; `make_device` passes it as `temp=(0.0 if greedy else args.temp)` (`:302`). host_sample defaults False in the constructor, so raising temp uses the on-chip path.

## 8. DBG readback path

The readback port is `rd_sel[3:0]` + `rd_addr[10:0]` → 64-bit `rd_data` (RD_LO 0x1C / RD_HI 0x20), 2-cycle registered (`sequencer_vec.sv:52-56`, `:987-990`). The RTL `DBG` generic gates it: `DBG=1` enables readback; record/fit bitstreams set `DBG=0` to tie it off (`gemv_axi_seq_sb.v:25`, `sequencer_sb` DBG param `gemv_axi_seq_sb.v:175`).

**rd_sel bank map (sequencer_vec, `sequencer_vec.sv:975-984`):**
| sel | bank | format |
|-----|------|--------|
| 0 | ln1 out | Q.22 (64-bit) |
| 1 | qkv | Q.16 |
| 2 | ctx | Q6.25 |
| 3 | attn_out | Q6.25 |
| 4 | ln2 out | Q.22 (64-bit) |
| 5 | gelu | Q.22 (64-bit) |
| 6 | mlp_out | Q6.25 |
| 7 | x4 residual | Q6.25 |
| 8 (default) | head logits | Q6.25 |

**dbg_stop (CTRL b4:3) halt points (`sequencer_vec.sv:525`, `:790`, `:824`):** 1 = halt after embed (xres holds x_in); 2 = halt after LN2 (xres=x_res1, lnout2=ln2); 3 = halt after block 0 (full phase snapshot).

**Consumers:**
- `pl_seq_vec.py --readback` compares x4 (sel 7, 256 elts), lnf (sel 0, 256), head (sel 8, 193) to `seq_ref` after the forward (`:151-155`, `:72-81`). `--stop-mode` does an ordered block-0 phase sweep [ln1,qkv,ctx,attn,ln2,gelu,mlp,x_out] and prints the FIRST divergence (`:156-169`). `read_bank` does a **dummy RD_LO read to flush the 2-cycle registered pipe** before the real lo/hi reads (`:53-63`).
- `dbg_board_kv.py` runs one pass at pos 0 with dbg_stop=3, reads qkv (sel 1, 768) and ctx (sel 2, 256), compares to an `IntKVQSequencer` block-0 sink — splits a fault between the GEMV path and the KV/attention path (`:1-8`, `:91-128`). Its `rd_bank` settles the pipe with **two STATUS reads** (`:36-38`) and sign-extends 64-bit (`:41-44`).
- `pl_kv256._read_logits` sets rd_sel=8 (`RD_SEL_HEAD=8`, `pl_kv256.py:55`), reads `self.vocab` logits via a tight inlined loop over the mmap view, sign-extends to 32-bit and divides by `2^25` (`HEAD_FRAC=25`) to real logits (`:187-204`). It relies on the AXI round-trip being longer than the 2-cycle settle, verified argmax==TOK_OUT on board (`:190-194`).

## 9. The a53_daemon RPC protocol

Wire format (`a53_daemon.py:44-54`): `encode(obj) = struct.pack(">I", len(body)) + body`, body = `msgpack.packb` (if available and not `--json`) else `json.dumps(...).encode()`. `read_frame` reads a 4-byte big-endian length then that many body bytes. msgpack is optional; JSON is the universal fallback (`:36-41`, `:308`).

**Server → daemon requests:**
- Batch: `{prompts: [str, ...]}`.
- Stream: `{prompts: [p], stream: true}` (only used if the device has `stream_into`, `:246`).

**Daemon → server responses (`handle_conn`, `a53_daemon.py:238-263`):**
- Batch: `{completions:[...], busy_ms, tokens, [fabric_ms, passes]}`. `busy_ms` = daemon wall time (`time.monotonic` around `dev.infer`, `:249-251`); `fabric_ms` = `dev.last_fabric_ms` = pure PL cycles / fclk × 1000 (`:254-257`, `pl_kv256.py:183-185`); `passes` = forward-pass count.
- Stream (`_stream_one`, `:198-235`): the device runs on an executor thread (blocking MMIO) and pushes each char through a threadsafe queue; each char is framed as `{chunk: text}` the instant it lands, then a final `{completions:[full], busy_ms, done:true, tokens, [fabric_ms, passes]}`.

**i7-side adapter `TcpPLBackend` (`a53_daemon.py:63-165`)** lives in the same file so encode/read_frame have one source of truth. `capacity=16`. One persistent connection carries a serial stream of batches under an asyncio lock. `infer_batch` sends `{prompts}`, and on a dead/half-open socket (`OSError`/`IncompleteReadError`/`TimeoutError`/`ConnectionError`) it drops the socket and **retries once** on a fresh connection (`:140-147`). `infer_stream` streams chunks to an `on_chunk` callback with **no mid-stream retry** (re-streaming would double chars on the client) and returns `(completion, busy_s, fabric_s, tokens, passes)` on the final frame (`:112-135`). Timeouts: connect 5 s, rpc 30 s (`:79-80`).

## 10. The three engines (`make_device`, `a53_daemon.py:289-304`)

- **`t1`** → `PLDevice` (`a53_daemon.py:167-196`) → `backend_stub.PLSingleTokenBackend`. Drives the N=16 single-token `SQ16`/`SQSB` sequencer over the `pl_seq_pp16` register protocol. Generates a completion by iterative single-token feedback (argmax → append tail → resubmit), `gen_chars` PL passes. HONESTY: only the FIRST token per stream is model-faithful — the fabric has no KV cache and attention is T=1, so fed-back tokens are plumbing exercise, tagged `"[T=1 stub] "` in output (`backend_stub.py` class docstring; `_submit_once` at the shown lines). At construction it forces the clock, streams weights + embed (choosing `pl_seq_pp16.build_embed_chunks` for SQ16/TMAX=32 vs `pl_seq_sb.build_embed_chunks(..., TMAX_DEFAULT=16)` for SQSB, keyed on IDCODE), and runs a startup gate: one 16-stream pass over `pl_seq_sb.TOKS16` must reproduce `seq_ref` tokens exactly or it refuses to serve.
- **`kv`** → `backend_kv.KVChatDevice(gemv_backend=args.kv_backend, ...)` (`a53_daemon.py:294-297`). Model-faithful multi-token Kevin via a host-side KV decoder (`model.goformer_kv.KVDecoder`) over a numpy/PL/C GEMV backend — the matmul goes to fabric (`--kv-backend pl`/`c`) but attention/non-linears/KV live on the A53. See `pl_kv_chat.py` for the standalone form (backends `numpy`/`pl`/`c`, `:60-117`; `c` = compiled MMIO driver `pl_resident_c.PLResidentC`).
- **`kv256`** → `pl_kv256.PLKV256Device` (`a53_daemon.py:298-303`). Model-faithful multi-token Kevin FULLY in fabric (doc-7 R1 SQRV KV bitstream). Single stream, so the daemon serves it strictly serially under `self._lock` (`pl_kv256.py:134`, `:296-311`).

## 11. Diagram — daemon data path

```
 i7 (Precision) server.py            GigE/Tailscale            Kria A53 daemon              PL fabric
 ┌─────────────────────┐   {prompts:[...]}  len-prefix   ┌──────────────────┐   /dev/mem   ┌──────────┐
 │  TcpPLBackend        │ ───────────────────────────►   │ handle_conn      │  mmap pokes  │ sequencer│
 │  .infer_batch/stream │                                 │  dev.infer(...)  │ ───────────► │ (SQRV/   │
 │  (persistent conn,   │ ◄───────────────────────────   │  make_device:    │   TOK_ID,POS │  SQSB/   │
 │   reconnect-once)    │  {completions,busy_ms,          │  t1 | kv | kv256 │   GO,poll    │  SQ16)   │
 └─────────────────────┘   fabric_ms,tokens,passes}       └──────────────────┘   TOK_OUT    └──────────┘
```
`--bench` mode (`a53_daemon.py:266-286`) skips the server and hammers `dev.submit_raw([0]*16)` for `--secs`, reporting launches/s and the 16× coalescing win (the PRD's open question).

**Parameters**

| name | meaning | live | range | where |
|---|---|---|---|---|
| `BASE` | AXI-Lite slave base mapped from /dev/mem; a single 4KiB page | 0xA0000000 | fixed | pl_seq_chat.py:61 (all drivers) |
| `FCLK_SET` | sysfs node written to force the PL clock; readback gives the quantised actual rate used for tok/s | /sys/devices/platform/fclk0/set_rate | path | pl_seq_chat.py:71 |
| `tol_hz` | set_and_verify_fclk fatal tolerance; larger deviation raises SystemExit | 6e6 | Hz | pl_seq_chat.py:75 |
| `NLAYER` | transformer blocks in the weight-image build order | 4 | int | pl_seq_vec.py:34, pl_seq_sb.py:32, pl_kv256.py:47 |
| `lanes` | PE width the LOADED bitstream was built with; sets subw and the pack | 128 (default) | 16/128/256 | argparse / device ctor, e.g. pl_seq_vec.py:99 |
| `subw` | 32-bit W_DATA chunks per wide weight word = (lanes*4)/32 | 16 at lanes=128 | derived | pl_seq_vec.py:120 |
| `RESID_FRAC / HEAD_FRAC` | Q6.25 fixed-point fraction bits for embeddings and head logits | 25 | fixed | pl_seq_sb.py:48, pl_kv256.py:56 |
| `TMAX_DEFAULT` | on-chip KV/pos window; MUST match the bitstream TMAX generic (build arg 5) or embeds corrupt | 16 (SQSB §24 build); 32 for old SQ16 | 16/32 | pl_seq_sb.py:47 |
| `tmax (kv256/daemon)` | prompt+reply must fit; maxp = tmax - gen_chars | 256 (default) | int | pl_kv256.py:104, a53_daemon.py:333 |
| `gen_chars` | completion chars per request; construction-time only, no per-request override | 48 for kv256, 6 otherwise | int | pl_kv256.py:109, a53_daemon.py:356-357 |
| `temp` | sampling temperature; 0=greedy. >0 uses on-chip Gumbel-max (LUT temp fixed 0.85) unless host_sample | 0.0 (daemon default); on-chip LUT baked at 0.85 | float | pl_kv256.py:123, a53_daemon.py:341 |
| `top_k` | host-sampling top-k truncation (only used on the host_sample path) | 40 (default) | int | pl_kv256.py:124, a53_daemon.py:346 |
| `N / --n` | parallel streams in the SQSB/SQ16 batch pass | 14 (NC7 97%-LUT fit) or 16 | 14 or 16 | pl_seq_sb.py:91, pl_seq_pp16.py:36 |
| `poll_timeout` | seconds to busy-poll STATUS.done before declaring TIMEOUT | 30.0 | s | pl_seq_vec.py:104, pl_kv256.py:106 |
| `rpc_timeout / connect_timeout` | TcpPLBackend per-RPC and connect timeouts | 30.0 / 5.0 | s | a53_daemon.py:80,79 |
| `capacity` | TcpPLBackend batch width advertised to the server (streams per launch) | 16 | int | a53_daemon.py:76 |
| `RD_SEL_HEAD` | rd_sel value selecting the final head-logit readback bank | 8 | fixed | pl_kv256.py:55 |
| `DBG (RTL generic)` | 1 enables the rd_sel/rd_addr debug readback; record/fit bitstreams tie it off with 0 | 1 debug, 0 for fit/record | 0/1 | gemv_axi_seq_sb.v:25 |

**Key facts**

- Every board driver mmaps /dev/mem at base 0xA0000000 for one 4KiB page and views it as a numpy uint32 array; only whole 32-bit stores are legal (byte slices corrupt the W_DATA stream) (fabric/stage3/board/pl_seq_chat.py:110-131).
- There are three distinct register maps: baseline SEQR/SQRF has CYCLES at 0x30 and IDCODE at 0x34 (pl_seq_chat.py:103-107), whereas SQRV and SQ16/SQSB put CYCLES at 0x28 and IDCODE at 0x2C (pl_seq_vec.py:40, pl_seq_sb.py:37). Using the wrong map silently mis-reads.
- The PL clock must be forced+verified via /sys/devices/platform/fclk0/set_rate before any output is trusted; a flat fpgautil -b load does NOT apply the BD's PL0_REF FREQMHZ so pl0_ref stays ~100MHz, violating setup and producing non-deterministic compute while short MMIO/cycle paths still look sane (fabric/stage3/board/pl_seq_chat.py:65-100).
- tok/s is always computed from the PLL's quantised readback rate, not the requested rate, and is WITHHELD unless fabric tokens match the seq_ref golden reference (pl_seq_chat.py:80-81, pl_seq_vec.py:171-179).
- Weight image order is fixed: for each of 4 blocks qkv|proj|mlp_fc|mlp_proj, then head, each pack_banked.pack_transposed at LANES; streamed as subw=(lanes*4)/32 32-bit chunks low-first after soft_reset(0x4->0x0) then wl_rst(0x2) (fabric/stage3/board/pl_seq_sb.py:66-73,125-132).
- SQRV vec/kv bake embed ROMs into the bitstream ($readmemh); kv256 appends them to the weight image spare depth via run_sequencer.wrom_embed_words; SQ16/SQSB stream them at runtime through EDATA(0x4C) as Q6.25 INT32, and the tmax used MUST equal the bitstream TMAX generic (fabric/stage3/board/pl_seq_kv.py:187-188, pl_kv256.py:85-91, pl_seq_sb.py:76-84).
- kv256/kv sequencers keep the K8/V8 KV banks alive between GO pulses, so multi-token decode is a train of GOs at rising POS; no reset between requests because a fresh pass overwrites KV[p] before it is read (fabric/stage3/board/pl_kv256.py:8-10, pl_seq_kv.py:113-117).
- All generation knobs (gen_chars, tmax, temp, top_k, lanes, fclk) are fixed at device construction; the RPC frame carries only {prompts:[...]} plus an optional stream flag, so there is NO per-request generation/window/sampling override (fabric/stage3/board/pl_kv256.py:102-165, webchat/demo/a53_daemon.py:238-259).
- The daemon RPC is a 4-byte big-endian length prefix + JSON (or msgpack) body: request {prompts:[...]} or {prompts:[p],stream:true}; batch reply {completions,busy_ms,tokens,fabric_ms,passes}; stream reply is {chunk} frames then a final {completions,done,...} (webchat/demo/a53_daemon.py:44-54,238-263).
- TcpPLBackend (i7 side) uses one persistent locked connection, reconnects+retries once on a dead socket for batch calls, and does NO mid-stream retry for streaming (webchat/demo/a53_daemon.py:137-161,112-135).
- make_device selects three engines: t1=PLDevice/PLSingleTokenBackend (SQ16 single-token, only 1st token faithful, T=1 stub), kv=host-side KVChatDevice over numpy/pl/c GEMV, kv256=PLKV256Device fabric-resident faithful single stream (webchat/demo/a53_daemon.py:289-304).
- Debug readback: rd_sel[3:0]+rd_addr[10:0]->64-bit rd_data, 2-cycle registered; sequencer_vec banks are 0=ln1,1=qkv,2=ctx,3=attn,4=ln2,5=gelu,6=mlp,7=x4,8=head; dbg_stop(CTRL b4:3) halts after embed(1)/LN2(2)/block0(3); DBG generic ties it off in record/fit builds (fabric/stage3/rtl/sequencer_vec.sv:975-990, gemv_axi_seq_sb.v:25).
- kv256 sampling has two paths: on-chip Gumbel-max via a nonzero SEED write at 0x30 (LUT temp fixed at 0.85 in the bitstream, armed once before the last prompt GO, no per-token readback) or host-side softmax/top-k over head logits read back at rd_sel=8 and /2^25 (fabric/stage3/board/pl_kv256.py:126,187-223).
- The MEASURED 59,965.5 tok/s record is the split-brain SQSB N=16 path; pl_seq_sb.py drives that engine (IDCODE SQSB 0x53515342) and reports aggregate tok/s = N*fclk/cyc only on a full 16/16 bit-exact match (fabric/stage3/board/pl_seq_sb.py:1-9,152-157).

**Files**

- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_seq_chat.py` — Canonical /dev/mem AXI master (Seqr) + set_and_verify_fclk (imported by all drivers); drives the baseline SEQR/SQRF sequencer with the distinct prompt-array/token-stream register map (CYCLES@0x30, IDCODE@0x34).
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_seq_vec.py` — P-wide single-token SQRV driver; single GO, TOK_OUT/CYCLES gate, and the --readback/--stop-mode debug phase sweep against seq_ref.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_seq_kv.py` — SQRV multi-token faithful decode (KV persists between GOs); full-stream gate vs IntKVQSequencer(kbits=8,vbits=8) with --runs 3 determinism check.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_kv256.py` — PLKV256Device: the doc-7 fabric-resident faithful KV chat device (SQRV); on-chip Gumbel-max vs host sampling, head-logit readback, daemon-facing infer/stream_into surface.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_seq_sb.py` — Split-brain SQSB N=8/16 batched driver (the 59,965.5 tok/s record engine); scattered 16-slot TOK_ID/TOK_OUT map, EDATA embed upload, TMAX_DEFAULT=16, aggregate tok/s gate.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_seq_pp16.py` — N=16 SQ16 batched driver (docstring stale-labeled 'pl_seq_gemm N=4'); IDCODE_SQ16, TMAX=32, build_embed_chunks used by the t1 backend.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/dbg_board_kv.py` — First-silicon fault localiser: one dbg_stop=3 pass at pos0, reads qkv(sel1)/ctx(sel2) banks vs an IntKVQSequencer block-0 sink to split GEMV vs KV/attention faults.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_kv_chat.py` — Host-side KV-cached chat over numpy/pl/c GEMV backends (NumpyResident/PLResident/PLResidentC); provides load_meta reused by pl_kv256; the --engine kv lineage.
- `/home/mikeayles/Desktop/Projects/kev-gpt/webchat/demo/a53_daemon.py` — The A53 TCP daemon: length-prefixed JSON/msgpack RPC, handle_conn batch+stream, TcpPLBackend (i7 side), PLDevice, make_device (t1/kv/kv256), --bench launch-rate mode.
- `/home/mikeayles/Desktop/Projects/kev-gpt/webchat/demo/backend_stub.py` — PLSingleTokenBackend (the t1 engine): SQ16/SQSB single-token 16-stream feedback loop, one-time weight+embed boot, startup bit-exact gate, launch counting; plus the off-box StubBackend.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/rtl/gemv_axi_seq_vec.v` — SQRV AXI-Lite wrapper: authoritative register decode (CTRL/STATUS/TOK_ID/POS/W_DATA/RD_SEL/RD_ADDR/RD_LO/RD_HI/TOK_OUT/CYCLES/IDCODE/SEED) confirming the Python offsets.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/rtl/gemv_axi_seq_sb.v` — SQSB AXI-Lite wrapper: confirms the scattered 16-slot TOK_ID/TOK_OUT map, EDATA(0x13/0x4C) el_we, dbg_stop, DBG/TMAX generics, IDCODE 0x53515342.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/rtl/sequencer_vec.sv` — Source of the rd_sel bank meanings (0..8), dbg_stop halt points, kv_bank persistence, and the persistent xorshift/Gumbel-max seed handling behind R_SEED.

**Gotchas**

- Three incompatible register maps share base 0xA0000000. The baseline SEQR/SQRF map (pl_seq_chat.py) has CYCLES@0x30, IDCODE@0x34, PL_DATA/PL_ADDR prompt writes and TS_ADDR/TS_DATA token drain; the SQRV and SQ16/SQSB maps have CYCLES@0x28, IDCODE@0x2C. Always verify IDCODE first and match the driver to the loaded bitstream.
- pl_seq_pp16.py is MISLABELED: its docstring says 'pl_seq_gemm — N=4 batch GEMM (SQGM)' but the code is the N=16 SQ16 driver (IDCODE 0x53513136, 16-slot lists). Trust the constants, not the header comment.
- A flat `fpgautil -b` load does NOT set the PL clock to the BD's FREQMHZ; you MUST force fclk0 via /sys/devices/platform/fclk0/set_rate and verify it, or the wide arithmetic runs at the wrong clock and gives non-deterministic garbage while CYCLES/MMIO still look plausible.
- tmax passed to the SQ16/SQSB embed loader MUST equal the bitstream's TMAX generic (build arg 5): 16 for the SQSB §24 build (TMAX_DEFAULT), 32 for the older SQ16. A mismatch streams the wrong pos_emb count and corrupts every embedding. SQRV vec/kv instead bake embeds into the bitstream (nothing uploaded).
- No per-request generation control exists. gen_chars/tmax/temp/top_k/lanes/fclk are all captured at device construction and the RPC only carries prompt text; to change any of them you must restart the daemon with new args.
- The registered readback pipe is 2 cycles: read_bank in pl_seq_vec does a dummy RD_LO read to flush it, dbg_board_kv does two STATUS reads, and pl_kv256._read_logits relies on the AXI round-trip being longer than the settle. Skipping the settle reads stale/X data. Also the DBG RTL generic is 0 in record/fit bitstreams, so readback returns nothing there.
- t1 engine output beyond the first token per stream is NOT model-faithful (no KV cache, attention at T=1); completions are tagged '[T=1 stub] ' and exist only to exercise the software/AXI plane and measure A53 launch rate — never present them as real text.
- Must set self.reg = None before mm.close(); numpy's frombuffer keeps an exported pointer and close() raises BufferError otherwise (every Dev.close).

**Open questions**

- The exact register decode of the baseline gemv_axi_seq.v (SEQR) and gemv_axi_seq_fast.v (SQRF) wrappers was not opened in this pass; the SEQR/SQRF map here is taken from pl_seq_chat.py's documented offsets (pl_seq_chat.py:15-26,103-107), which are internally consistent but not cross-checked against that RTL as the SQRV/SQSB maps were.
- The C MMIO backend (pl_resident_c.PLResidentC / gemv_axi_drv) used by --engine kv --kv-backend c and by PLResident (fabric.pl_gemv) was referenced but not read; its ctypes register protocol and whether it matches the SEQR/SQRF map is not documented here.
- backend_kv.KVChatDevice (the --engine kv daemon device) was not read in full; only its make_device wiring and the standalone pl_kv_chat.py equivalent are documented — its exact gen_chars/greedy plumbing and fabric_ms reporting may differ.
- The precise poll strategy differs between drivers (busy-poll with wall-clock timeout vs a fixed 1,000,000-iteration spin in backend_stub._submit_once); which is used in the live daemon path for each engine, and whether the spin can under-run at low fclk, was not measured.
- How server.py maps its --backend pl/tcp/kv choices onto local PLDevice vs remote TcpPLBackend, and whether the live chat.mikeayles.com deployment uses the split (i7+Kria) or single-box (server-on-board) topology, is only partially visible here (server.py:407-418) and not fully traced.

---

## Build flow, bitstreams, timing & resource utilization (Kevin-on-Kria stage 3)

Stage-3 hardware ships as PL bitstreams for the KV260 (part `xck26-sfvc784-2LV-c`). There are two live engine families, each with its own AXI-Lite shell (distinct IDCODE), its own build/impl TCL, and its own board driver: (1) the split-brain N=16 throughput engine `sequencer_sb` (shell `gemv_axi_seq_sb.v`, IDCODE "SQSB"), which holds the MEASURED record 59,965.5 tok/s @200 MHz; and (2) the doc-7 KV-faithful single-stream engine `sequencer_vec` (shell `gemv_axi_seq_vec.v`, IDCODE "SQRV"), which is what actually runs the public chat. Every bitstream is built by a three-script flow: an out-of-context (OOC) fit/Fmax probe (`ooc_*.tcl`), a PS+PL block-design builder (`build_bd_*.tcl`), and a synth+impl+bitstream runner (`impl_*.tcl`). Builds live under `C:/kevbuild/...`, never in the OneDrive-backed repo.

The single most load-bearing fact for the blocked measurement task: the live Kria daemon `kevkv.service` loads `~/kevbit/gemv_seqkv_gum.bit.bin`, which is the KV-faithful engine (`sequencer_vec`/`gemv_axi_seq_vec`, IDCODE SQRV) with on-chip Gumbel-max sampling added (log §44-45), NOT the split-brain engine. It was BUILT with P=8, LANES=256, TMAX=128, WWORDS=16384, NLAYER=4 at a 125 MHz (8 ns) target, and it RUNS on silicon at 166.7 MHz. DBG and ATT2 do NOT apply to it — those are `sequencer_sb`-only generics; the KV engine's qkv/ctx debug readback is hard-wired, not a build knob. This is corroborated by three independent sources: the daemon invocation captured in project memory (`a53_daemon --engine kv256 --lanes 256 --tmax 128 --fclk 166.7e6`), the build script's hardcoded mems dir `stage3_vec_kvk8t128_smp` (the "t128"/"smp" = TMAX 128 + sampling), and the shipping-config log entries §37/§39/§45.

The core timing story is silicon overclock vs STA pessimism on the -2LV speed grade: designs that close only ~70-114 MHz in static timing analysis run bit-exact at 125-200 MHz on real silicon, a measured margin band of roughly 1.3x-1.76x. Because STA is untrustworthy here, no tok/s number is claimed from timing reports; the real clock ceiling is found by a board `--fclk` sweep that forces the PL clock via `/sys/devices/platform/fclk0/set_rate` and records the highest frequency that still matches the golden reference 3/3. A hard constraint on that sweep: the PS PLL only offers 1000/N divider steps, so the usable clocks are quantized to 125 / 142.9 / 166.7 / 200 MHz — there is no step between 166.7 and 200 (175 snaps down, 183.3 snaps up), which is why several records sit exactly at 166.7.

The forward path to 100k is the double-pump (DP=1) variant of `sequencer_sb`: a Clocking Wizard MMCM makes a phase-aligned clk2x, and the DSP MAC island runs at 2x the fabric clock. It fits and CLOSES timing at 200/400 MHz in OOC (clk WNS +0.070, clk2x +0.074) and projects ~88,449 tok/s @200/400 in sim. But: the full DP=1 build is OOC-timing-MET only, never board-run; a MAC-only DP branch (`dp-hw-maconly`) WAS run on silicon bit-exact but the fabric→clk2x data feed walled at 50 MHz — that measurement killed the lever (DEAD, `100K-REVIEW.md:44`). The record remains the single-pump 59,965.5.

## 1. The bitstream / engine / IDCODE map

Each AXI-Lite shell hard-codes a 32-bit ASCII IDCODE the driver reads back at register `0x2C` (KV/SB) or `0x34` (old SEQR) to confirm which bitstream is loaded.

| Engine (top RTL) | AXI shell | IDCODE | Bitstream file(s) | Build TCL family |
|---|---|---|---|---|
| `sequencer_vec` (KV-faithful, doc-7, N=1) | `rtl/gemv_axi_seq_vec.v` | "SQRV" `0x53515256` (`gemv_axi_seq_vec.v:3`) | `gemv_seqkv.bit.bin`, `gemv_seqkv_trims.bit.bin`, **`gemv_seqkv_gum.bit.bin`** | `build_bd_seq_kv.tcl` + `impl_seq_kv.tcl` (OOC: `ooc_seq_vec.tcl`) |
| `sequencer_sb` (split-brain, N=16, two N=8 cohorts) | `rtl/gemv_axi_seq_sb.v` | "SQSB" `0x53515342` (`gemv_axi_seq_sb.v:3`) | `gemv_seqsb.bit.bin` | `build_bd_seq_sb.tcl` + `impl_seq_sb.tcl` (OOC: `ooc_seq_sb.tcl`) |
| `sequencer_sb` DP=1 (double-pump, clk2x MAC) | `rtl/gemv_axi_seq_sb.v` | "SQSB" `0x53515342` | `gemv_seqsb_dp.bit.bin` (built, not board-measured) | `build_bd_seq_sb_dp.tcl` + `impl_seq_sb_dp.tcl` (OOC: `ooc_seq_sb_dp.tcl`) |
| `sequencer_pp16` (ping-pong N=16, pre-split-brain) | `rtl/gemv_axi_seq_pp16.v` | "SQ16" `0x53513136` (`gemv_axi_seq_pp16.v:3`) | historical | `build_bd_seq_pp16.tcl` + `impl_seq_pp16.tcl` |
| `sequencer_fast` (256-MAC single stream) | `rtl/gemv_axi_seq_fast.v` | "SQRF" `0x53515246` (`gemv_axi_seq_fast.v:6`) | `gemv_seqfast.bit.bin`, `gemv_seqfast_p128.bit.bin` | `build_bd_seq_fast.tcl` + `impl_seq_fast.tcl` |
| `sequencer` (first HW sequencer) | `rtl/gemv_axi_seq.v` | "SEQR" `0x53455152` (`gemv_axi_seq.v:144`) | `gemv_seq.bit.bin` | `build_bd_seq.tcl` + `impl_seq.tcl` |

(Older bitstream lineage — file names, sizes, IDCODEs — is logged in `fabric/stage3/BOARD-TEST.md:74,109,129`.)

## 2. THE LIVE BITSTREAM: `gemv_seqkv_gum.bit.bin` — resolved build parameters

**This is the fact that blocked a prior measurement task. It is now settled.** The `kevkv.service` unit on the Kria loads `~/kevbit/gemv_seqkv_gum.bit.bin` and runs `a53_daemon --port 9099 --json --engine kv256 --lanes 256 --tmax 128 --fclk 166.7e6 --gen-chars 104` (captured in project memory `live-demo-topology.md`; this exact daemon line is NOT in the repo — it lives in the systemd unit on the board).

Build identity, cross-checked three ways:

- **Engine:** `sequencer_vec` via shell `gemv_axi_seq_vec.v`, IDCODE **SQRV** (`WIDE-WORD-DATAPATH-LOG.md:1428` "Bitstream gemv_seqkv_gum.bit.bin, idcode SQRV"; §45).
- **Built by** `build_bd_seq_kv.tcl` (BD) then `impl_seq_kv.tcl` (synth/impl/bitstream). `impl_seq_kv.tcl` emits `seq_kv.bit`/`.bin`; the deployed copy was renamed `gemv_seqkv_gum` on deploy.
- **P = 8** — the wide-word datapath width; `build_bd_seq_kv.tcl:22` default `P=8`, and the doc-7 shipping config is "K8 ... at TMAX=128" (`WIDE-WORD-DATAPATH-LOG.md:1121` §37). (Note: the RTL module default is `P=16` at `gemv_axi_seq_vec.v:19`, but the BD script overrides to 8.)
- **LANES = 256** — definitive because the host weight-image packing (`pl_kv256.build_weight_image`, `pl_kv256.py:75`) is LANES-dependent and MUST match the bitstream; the daemon passes `--lanes 256`. The doc-7 shipping GEMV is LANES=256 (`docs/7-kevin-remembers.md:51`; log §11 landed LANES=256). (RTL/BD default is different: `gemv_axi_seq_vec.v:20` default LANES=128, `build_bd_seq_kv.tcl:23` default LANES=256.)
- **TMAX = 128** — daemon `--tmax 128`; build script mems dir hardcoded to `stage3_vec_kvk8t128_smp` (`build_bd_seq_kv.tcl:33`, the "t128" dir); shipping config §37 ("K8 no-rotate ... at TMAX=128"); board run §39 used `--tmax 128`. (The BD script's own default is TMAX=256 at `build_bd_seq_kv.tcl:27`, so the gum build was invoked with an explicit `tmax=128` tclarg.)
- **WWORDS = 16384** — resident weight URAM depth; `build_bd_seq_kv.tcl:24` default, and §35 arithmetic uses "16,384 − 12,496 = 3,888 spare addresses" confirming 16384.
- **NLAYER = 4** (fixed, `build_bd_seq_kv.tcl:79`).
- **Clock target: 125 MHz / 8 ns.** §45: "Built BD+impl at a 125MHz target (the 200-target trims build was unroutable at 182k overlaps; the looser target packs tighter and routes)." (The BD script's `freq` default is **40** at `build_bd_seq_kv.tcl:25` — the 166.666666 default belongs to `build_bd_seq_sb.tcl:16`, a different script — so an explicit freq was passed for the gum build; the 125 MHz target was not a default.)
- **Runs on silicon at 166.7 MHz** — the daemon forces `--fclk 166.7e6`; §45 gates are "all green on board @166.7MHz."
- **DBG / ATT2: DO NOT APPLY.** `sequencer_vec` and `gemv_axi_seq_vec.v` have NEITHER generic (`grep DBG|ATT2` returns nothing in those files). The KV engine's qkv/ctx debug readback (used by `dbg_board_kv.py`: "qkv 0/768, ctx 0/256") is hard-wired, always present — it is not a synthesis knob. DBG and ATT2 are `sequencer_sb`-only (see §5 below). So the correct answer to "what DBG/ATT2 was this bitstream built with" is: **the question does not apply to the KV engine; there are no such generics to set.**
- **On-chip sampling ("gum"):** the "_gum" suffix = the Gumbel-max on-chip sampler (log §44-45): `gumbel_lut.mem` baked at temp=0.85, xorshift32 RNG persisting across GOs, SEED register at `0x30`; seed=0 ⇒ greedy/argmax. **UNRESOLVED which sampler is live:** project memory says temp/top-k are done host-side over head-logit readback (`--temp 0.4 --top-k 10`, on-chip 0.85 "was gibberish"), but the committed code with those exact flags samples ON-CHIP at the baked temp and ignores top-k (`a53_daemon.py:289-304` passes no `host_sample`; `pl_kv256.py:126`). Resolve only by reading the Kria's systemd ExecStart + the on-board `a53_daemon.py` (which may be hand-edited vs repo).

**Reported build results for the gum bitstream (§45):** ROUTED clean, **WNS +0.013 ns MET**, util **LUT 93.4% / BRAM 94.4% / URAM 100% / DSP 97.4%**. Greedy: 9,292 cyc/tok = 17,936.6 tok/s @166.7 MHz MEASURED, 3/3 bit-identical to golden; fabric 16,227 tok/s in the round-trip test (16k record preserved).

**What would settle any residual doubt beyond the repo:** the Vivado invocation/log on the build box (`C:/kevbuild/stage3_seqkv_bit/...` — the `runme.log` tclargs line, `util_impl.rpt`, `timing_impl.rpt`), or reading the `kevkv.service` `ExecStart=` on the Kria directly. Both are outside this repo; the repo evidence + memory file agree on P=8/LANES=256/TMAX=128/125-target/166.7-run.

## 3. The three-script build flow (per engine)

Every engine follows OOC → build_bd → impl:

1. **`ooc_*.tcl`** — `synth_design -mode out_of_context` on just the sequencer RTL (no PS, no BD). Answers "does it fit (LUT/BRAM/URAM/DSP) and at what Fmax?" via `report_utilization` + `report_timing_summary`. Fast inner loop before a 1-4h full build. Runs from a `C:/kevbuild/...` dir holding the `.mem` ROM-init files.
2. **`build_bd_*.tcl`** — creates the Vivado project, instantiates the Zynq MPSoC PS (`zynq_ultra_ps_e`) with board preset, sets `PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ` = the requested PL clock, drops the engine as a `-type module` BD cell with its `CONFIG.*` generics, auto-wires `M_AXI_HPM0_FPD → seq/S_AXI` through an AXI SmartConnect, assigns address `0xA0000000`, validates, wraps, and generates targets. Build-only.
3. **`impl_*.tcl`** — opens the project, `launch_runs synth_1`, then sets placer/router directives, `launch_runs impl_1 -to_step write_bitstream`, opens the impl run, writes `util_impl.rpt` + `timing_impl.rpt`, extracts WNS, copies the `.bit` out and `write_cfgmem -format BIN` to produce the `.bit.bin` for `fpgautil`.

### tclargs signatures (positional; defaults in parentheses)

**KV engine:**
- `ooc_seq_vec.tcl` — `<P> <LANES> <PERIOD_NS> <WWORDS> <TMAX>` (16, 128, 8.0, `3276800/LANES`, 64). Note WWORDS scales INVERSELY with LANES because the ~12.6 Mbit image is fixed and word width = LANES×4 bits: the formula `3276800/LANES` gives L=16→204800, L=128→**25600**, L=256→**12800** (`ooc_seq_vec.tcl:16-17`; the tcl's own comment "32768/16384" is stale — the formula on the next line is authoritative). These are *image-words* actually used (~12,800 at L=256 OOC; 12,496 at L=256 in the gum build). Distinct from that, the gum bitstream reserves a deeper **16,384**-word bank (`build_bd_seq_kv.tcl:24`), the spare 3,888 addresses holding the token/pos embeds (log ~1221) — so 12,800/12,496 (used) vs 16,384 (reserved depth) are not a contradiction.
- `build_bd_seq_kv.tcl` — `<P> <LANES> <WWORDS> <FREQMHZ> <TMAX>` (defaults 8, 256, 16384, **40**, 256; the gum build passed explicit tclargs, incl. a 125 MHz target). Sets only `CONFIG.P/LANES/NLAYER/WWORDS/TMAX/C_S_AXI_ADDR_WIDTH` (`build_bd_seq_kv.tcl:79-80`) — no DBG/ATT2. HARD-ERRORS on any missing `.mem` (`build_bd_seq_kv.tcl:64`) — see gotcha §38.
- `impl_seq_kv.tcl` — no tclargs; `bdir` hardcoded `C:/kevbuild/stage3_seqkv_bit`. Directives: `PLACE_DESIGN=AltSpreadLogic_high`, `ROUTE_DESIGN=AlternateCLBRouting` (`impl_seq_kv.tcl:22-23`). Also greps every synth `runme.log` for `[Synth 8-4445]` (empty-ROM guard) and fails loudly (`impl_seq_kv.tcl:11-17`).

**Split-brain engine:**
- `ooc_seq_sb.tcl` — `<P> <LANES> <PERIOD_NS> <WWORDS> <TMAX> <ND> <NC> <ATT2>` (8, 128, 6.0, 25600, 16, 6, 8, 0). ND = DSP-packed GEMM streams **per cohort** (6→12 of 16 total, the "SS18" config — the build-log/tcl shorthand for the split-brain configuration with ND=6/cohort = 12 of 16 streams on the DSP-packed leaf); NC = streams per cohort (8→N=16; 7→N=14 fit variant); N = 2·NC; ATT2 = per-cohort vs shared attention (`ooc_seq_sb.tcl:5-24`).
- `build_bd_seq_sb.tcl` — `<P> <LANES> <WWORDS> <FREQMHZ> <TMAX> <ND> <BDIR> <NC>` (8, 128, 25600, 166.67, 16, 6, `C:/kevbuild/stage3_seqsb_bit`, 8). **Forces `CONFIG.DBG {0}` and `CONFIG.ATT2 {0}` for the fit build** (`build_bd_seq_sb.tcl:79`).
- `impl_seq_sb.tcl` — `[<bdir>]`. Same two placer/router directives as the KV impl (`impl_seq_sb.tcl:15-16`).

**Split-brain double-pump (DP=1):**
- `ooc_seq_sb_dp.tcl` — `<P> <LANES> <CLK_NS> <CLK2X_NS> <WWORDS> <TMAX> <ND> <NC> <ATT2>` (8, 128, 5.0, 2.5, 25600, 16, 0, 8, 0). Adds `mac_bank_dp.sv` + `mac_bank_dsp_dp.sv` to the source list, synths with `-generic DP=1 -verilog_define SYNTHESIS`, creates two clocks (`clk` @5ns, `clk2x` @2.5ns) and marks them an async clock group (`ooc_seq_sb_dp.tcl:47-56`).
- `build_bd_seq_sb_dp.tcl` — `<P> <LANES> <WWORDS> <FIN_MHZ> <TMAX> <ND> <BDIR> <NC> [<PH2X>]` (8, 128, 25600, 200.0, 16, 0, ..., 8, 180.0). Adds a Clocking Wizard MMCM: `pl_clk0 = clk` (AXI/fabric) fans into `clk_out1 = clk2x = 2×pl_clk0`, phase 180° so clk2x rising edges land mid clk half-period (matches the tb_macdp quarter-shift convention, avoids a coincident-edge race). Sets `CONFIG.DBG {0} CONFIG.ATT2 {0} CONFIG.DP {1}` (`build_bd_seq_sb_dp.tcl:76-102`). Board sweeps `fclk0`: clk=2×fclk0, clk2x=4×fclk0 (fclk0=100 → 200/400).
- `impl_seq_sb_dp.tcl` — `[<bdir>]`. Adds `PHYS_OPT_DESIGN` enabled with `AggressiveExplore` on top of the spread-logic + alt-CLB-routing directives, for the near-full 97% FF / 100% URAM density (`impl_seq_sb_dp.tcl:15-18`).

## 4. Block-design structure (common to all build_bd scripts)

`create_bd_design design_1` → PS cell `zynq_ultra_ps_e` with `apply_board_preset 1` → set PL clock (`PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ`), enable `M_AXI_HPM0_FPD` (GP0), `FPGA_PL0_ENABLE 1` → engine as `create_bd_cell -type module -reference <shell>` with `CONFIG.*` generics → `apply_bd_automation axi4` wires GP0 master to `seq/S_AXI` via a new AXI SmartConnect (auto clocks) → `assign_bd_address` + force offset `0xA0000000` → validate/wrap/generate. The `.mem` ROM-init files are added as `Memory Initialization Files` from the sim dir so `$readmemh` sees them at synth.

Register map (KV/SB shells, from `gemv_axi_seq_vec.v:12` / `gemv_axi_seq_sb.v:9`): `0x1C RD_DATA_LO, 0x20 RD_DATA_HI, 0x24 TOK_OUT(0), 0x28 CYCLES, 0x2C IDCODE`; KV adds `0x30 SEED` for the gumbel sampler.

## 5. `sequencer_sb` generics (the split-brain engine) — DBG/ATT2/DP live HERE

From `rtl/gemv_axi_seq_sb.v:17-31`:

| Generic | Default | Meaning |
|---|---|---|
| `P` | 8 | wide-word datapath width (elements/word) |
| `LANES` | 128 | GEMV MAC width |
| `N` | 16 | total streams (= 2·NC) |
| `NC` | 8 | streams per cohort (8→N=16; 7→N=14 fit variant) |
| `ND` | 6 | DSP-packed streams per cohort |
| `NLAYER` | 4 | transformer layers |
| `WWORDS` | 25600 | resident weight URAM depth (wide words) |
| `TMAX` | 32 | on-chip KV window (record build uses 16; §26) |
| `DBG` | 1 | **0 = tie off debug readback (record/bitstream builds)** |
| `ATT2` | 1 | **1 = per-cohort attention (un-share, §26); 0 = shared+arbiter (fits today)** |
| `DP` | 0 | **1 = double-pump MAC at 2 K-steps/clk (100k campaign)** |

The record and shipping SB bitstreams set DBG=0 / ATT2=0 for fit (`build_bd_seq_sb.tcl:79`). This is the source of the CLAUDE.md line "bitstream builds set DBG=0/ATT2=0 for fit" — and it is why DBG/ATT2 are meaningful for SB but nonexistent for the KV engine.

## 6. MEASURED record build: 59,965.5 tok/s @200 MHz (log §27)

The record is the **split-brain** engine, the §26 "architectural wave": TMAX 32→16, shared attention (ATT2=0) + arbiter, CTX cross-group stream, plus the LN prod×gamma split. Built at a **6.0 ns target** (a 5.5 ns route diverged — overlaps climbed 295k→339k at 98% BD density, killed at 2h; the density tax). 6.0 ns routed clean: **impl WNS −0.702 ns, impl 1h43m** (`WIDE-WORD-DATAPATH-LOG.md:988-990`).

Silicon (board `--fclk` sweep): **53,364 cyc** (sim 53,637 − 273 SETTLE), **16/16 bit-exact, 3/3 runs**, → **49,971.3 tok/s @166.7 MHz / 59,965.5 tok/s @200 MHz**; 250 MHz → TIMEOUT (hangs, not corrupt; the 6.0 ns target gives ~1.6x silicon margin, 250 needs ~2.0x). New record, +6.6% over the prior 56,262.7 (`WIDE-WORD-DATAPATH-LOG.md:989-992`).

Related OOC corner for the same topology family (§25, `ooc_seq_sb.tcl 8 128 5.0 25600 32 6 8`): WNS **+0.074 ns MET @5ns**, worst path in `vec_dequant` (`u_dq` shift→out cone), **CLB LUTs 110,723**, BRAM **141/144 (97.9%)**, URAM **64/64 (100%)**, DSP **1171/1248** (`WIDE-WORD-DATAPATH-LOG.md:944-951`). The ATT2=1 per-cohort un-share OOC (§26) was 115,284 LUT (98.43%), BRAM 142, DSP 1215 — ~1.9k LUT over what routes at full density, hence ATT2=0 ships.

## 7. Double-pump (DP=1) — the 100k plan: full build OOC-only, MAC-only branch ran on silicon then DIED

`DOUBLE-PUMP-100K.md` + log: the full DP=1 sequencer **fits and CLOSES timing at 200/400 MHz OOC** — clk 200 MHz WNS **+0.070**, clk2x 400 MHz WNS **+0.074**, 0 failing endpoints. SIM projects (run_sb_seq --nd 0 --tmax 16): `--dp 1` = 38,256 cyc = **83,647 tok/s @200**, and with the further pieces **88,449 tok/s @200/400 (SIM PROJECTED)** — 4,179 cyc from 100k. Key architectural choice: only the DSP MAC accumulator runs at clk2x; the URAM weight banks stay at the slow clock (the 25600-deep cascade = 6 CAS hops is timing-dead at 400 MHz), so the board only has to validate the MAC's clk2x phase. This is built (`gemv_seqsb_dp`) but the FULL DP=1 build is OOC-timing-MET only and was never board-run. A MAC-only DP branch (`dp-hw-maconly`) WAS run on silicon bit-exact — but the fabric→clk2x data feed walled at 50 MHz, and that measurement is what killed the lever (DEAD, `100K-REVIEW.md:44`). The record stays single-pump 59,965.5. The failure mode the plan itself named — if clk2x can't hold 400 on silicon it caps the fabric to ~half and the gain is negated (~60k) — is essentially what the 50 MHz data-feed wall confirmed.

## 8. The silicon-overclock story (STA pessimism on -2LV)

STA/impl timing on the `-2LV` (low-voltage) speed grade is systematically pessimistic; designs run bit-exact well above their STA Fmax. Measured margin examples:

| Log § | STA-max (impl) | Silicon clock (bit-exact) | Margin |
|---|---|---|---|
| §39 | 103.6 MHz (WNS −1.654 @8ns) | 142.9 MHz | ~1.38x |
| §20 | ~127 MHz | 166.7 MHz | 1.31x |
| §41 | 114.3 MHz | 166.7 MHz | 1.46x |
| CLAUDE.md/§ (early vec) | ~70-85 MHz | 125 MHz | ~1.76x |

So the working rule (CLAUDE.md "Silicon overclock vs STA"): a design that closes ~70-85 MHz STA has run at 125 MHz silicon (~1.76x); the proven margin band is ~1.3x-1.76x. **Consequence for the build strategy:** target a clock that merely *closes/routes* (often the looser 6-8 ns target routes where an aggressive 5 ns target congests, §42→§43), ship it, then find the true ceiling on the board.

### The board `--fclk` sweep (finds the real ceiling)

A flat `fpgautil -b` load does NOT apply the BD's `PL0_REF FREQMHZ`; the driver must force the clock. `set_and_verify_fclk()` (imported from `board/pl_seq_chat.py`, used by `pl_kv256.py:137`, `pl_seq_sb.py:106`) writes `/sys/devices/platform/fclk0/set_rate` and verifies the readback. The sweep runs the same generation at each candidate fclk, checks the token stream is bit-exact to the golden reference, records tok/s = cyc/fclk, and takes the highest fclk that still matches 3/3 as the ceiling.

**PLL granularity trap:** the PS PLL only offers 1000/N divider steps, so usable clocks are quantized to **125 / 142.9 / 166.7 / 200 MHz**. There is NO step between 166.7 and 200 (175→snaps to 166.7, 183.3→snaps to 200, both verified — §43). This is why many records land exactly at 166.7 (200 fails, nothing in between). Escaping this needs an MMCM `clk_wiz` in the BD (the double-pump path does exactly this).

## 9. Board load / deploy path

`sudo xmutil unloadapp ; sudo fpgautil -b ~/kevbit/<name>.bit.bin` loads the PL, then a Python board driver (`pl_kv256.py` / `pl_seq_sb.py` / `pl_seq_kv.py`) forces `fclk0`, uploads the weight image + embeds over /dev/mem (packing MUST match the build's LANES; embed rows MUST match the build's TMAX), and drives GOs. The live daemon `a53_daemon.py` wraps `PLKV256Device` (`a53_daemon.py:299`) and serves JSON to the chat server.

## 10. Utilization figures collected

| Build | LUT | BRAM | URAM | DSP | WNS / target |
|---|---|---|---|---|---|
| KV gum (§45, live) | 93.4% | 94.4% | 100% | 97.4% | +0.013 ns MET @125-target |
| KV early P8/L128/TMAX64 (§8/§13) | 72,017 (61.5%) | 142.5/144 (99.0%) | — | — | −4.285 @8ns (STA), ran on silicon |
| KV R5 200-target (§41) | ~91% | — | — | — | impl WNS −3.752 (ran 166.7) |
| KV trims 125-target (§43) | ~92% | — | — | — | +0.006 ns MET (route worked) |
| SB record OOC (§25) | 110,723 | 141/144 (97.9%) | 64/64 | 1171/1248 | +0.074 @5ns |
| SB record impl (§27) | — | — | — | — | −0.702 @6ns (ran 200) |
| SB ATT2=1 un-share OOC (§26) | 115,284 (98.43%) | 142 | 64/64 | 1215 | +0.074 @5ns |
| SB DP=1 OOC (double-pump) | fits | — | 100% | — | clk +0.070 @200 / clk2x +0.074 @400 |

The device is consistently URAM-limited (64/64 = 100%) and LUT/BRAM/DSP all 90%+ — the ~3 MB on-chip budget is the hard ceiling and the design lives at the edge of it, which is why routing congestion (not timing per se) is the recurring wall (§42 "182340 node overlaps" unroutable at 5ns; §43 the looser 8ns target routed).

**Parameters**

| name | meaning | live | range | where |
|---|---|---|---|---|
| `P` | Wide-word datapath width (elements packed per URAM/BRAM word). The live KV bitstream uses 8. | 8 (gum bitstream) | 8 or 16 | gemv_axi_seq_vec.v:19 (default 16); build_bd_seq_kv.tcl:22 (default 8) |
| `LANES` | GEMV MAC width. Weight-image packing is LANES-dependent so the driver must match the build. | 256 (gum bitstream, daemon --lanes 256) | 16 / 64 / 128 / 256 (512 aspirational) | gemv_axi_seq_vec.v:20 (default 128); build_bd_seq_kv.tcl:23 (default 256) |
| `TMAX` | On-chip KV / pos-table window depth. Driver embed upload rows must equal build TMAX or pos>0 corrupts. | 128 (gum bitstream, daemon --tmax 128) | 16 (SB record) / 128 (KV live) / 256 (full doc-7 window) | build_bd_seq_kv.tcl:27 (default 256); gemv_axi_seq_vec.v:23 (default 64); sequencer_sb TMAX default 32 (gemv_axi_seq_sb.v:24) |
| `WWORDS` | Resident weight URAM depth in wide words; scales inversely with LANES (fixed ~12.6 Mbit image). | 16384 (gum bitstream) | 16384 (L=256) up to 262144 (L=16) | build_bd_seq_kv.tcl:24 (16384); ooc_seq_vec.tcl:17 (3276800/LANES) |
| `NLAYER` | Transformer layers (goformer is 4-layer). | 4 | 4 | build_bd_seq_kv.tcl:79 |
| `FREQMHZ / fclk target` | PL0 reference clock the BD requests (PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ). Build target, not the silicon run clock. | 125 MHz build target; runs 166.7 MHz on silicon | 125 (route-safe) to 200 (aggressive) | build_bd_seq_kv.tcl:25 (default 40); overridden to 125 for gum (log §45) |
| `DBG` | 1 = qkv/ctx debug readback present; 0 = tied off for fit. Applies ONLY to split-brain; the KV engine has no DBG generic (readback hard-wired). | N/A for the KV gum bitstream; SB record/bitstream builds use 0 | 0 or 1 | gemv_axi_seq_sb.v:25 (sequencer_sb ONLY); build_bd_seq_sb.tcl:79 sets 0 |
| `ATT2` | 1 = per-cohort attention (faster, +~1.9k LUT, needs TMAX=16 BRAM); 0 = shared attention + arbiter (fits). Split-brain only. | N/A for the KV gum bitstream; SB fit builds use 0 | 0 or 1 | gemv_axi_seq_sb.v:26 (sequencer_sb ONLY); build_bd_seq_sb.tcl:79 sets 0 |
| `DP` | 1 = double-pump MAC at 2 K-steps/clk via clk2x MMCM (100k campaign). Full build OOC-MET only, never board-run; MAC-only branch `dp-hw-maconly` ran on silicon bit-exact but the clk2x data feed walled at 50 MHz (DEAD). | 0 for all measured records | 0 or 1 | gemv_axi_seq_sb.v:27 (sequencer_sb ONLY) |
| `ND` | DSP-packed GEMM streams per cohort (6 → 12 of 16 total, the SS18 DSP budget). | 6 (SB record); N/A for KV | 0 (all-LUT) to 8 | ooc_seq_sb.tcl:20, build_bd_seq_sb.tcl (default 6) |
| `NC / N` | Streams per cohort; total N = 2*NC. 8→N=16 (record), 7→N=14 (-2-LUT-bank fit variant). | NC=8, N=16 (SB record); N/A for KV (N=1) | NC 7 or 8; N 14 or 16 | ooc_seq_sb.tcl:22-25 |

**Key facts**

- The live public-chat bitstream ~/kevbit/gemv_seqkv_gum.bit.bin is the KV-faithful single-stream engine sequencer_vec (shell gemv_axi_seq_vec.v, IDCODE SQRV 0x53515256), NOT the split-brain engine (WIDE-WORD-DATAPATH-LOG.md:1428).
- gemv_seqkv_gum was BUILT with P=8, LANES=256, TMAX=128, WWORDS=16384, NLAYER=4 at a 125 MHz (8 ns) target, and RUNS on silicon at 166.7 MHz (live-demo-topology.md daemon line 'a53_daemon --engine kv256 --lanes 256 --tmax 128 --fclk 166.7e6'; corroborated by build_bd_seq_kv.tcl:33 mems dir stage3_vec_kvk8t128_smp and log §37/§45).
- DBG and ATT2 do NOT apply to the KV gum bitstream: sequencer_vec/gemv_axi_seq_vec.v have neither generic; the qkv/ctx debug readback is hard-wired always-on. DBG/ATT2 are sequencer_sb-only generics (gemv_axi_seq_sb.v:25-26).
- The MEASURED record 59,965.5 tok/s @200 MHz is the split-brain engine sequencer_sb, TMAX=16 wave, ATT2=0 (shared attention), 53,364 cyc, 16/16 bit-exact, 3/3, built at a 6.0 ns target with impl WNS -0.702 ns (WIDE-WORD-DATAPATH-LOG.md:989-990).
- The gum bitstream build results: WNS +0.013 ns MET, LUT 93.4% / BRAM 94.4% / URAM 100% / DSP 97.4%; greedy 9,292 cyc/tok = 17,936.6 tok/s @166.7 MHz MEASURED (WIDE-WORD-DATAPATH-LOG.md:1426-1433).
- Every engine builds via three TCL scripts: ooc_*.tcl (OOC fit/Fmax probe), build_bd_*.tcl (PS+PL block design), impl_*.tcl (synth+impl+bitstream, extracts WNS, writes .bit.bin) (ooc_seq_sb.tcl, build_bd_seq_sb.tcl, impl_seq_sb.tcl headers).
- Silicon overclock margin on -2LV is ~1.3x-1.76x over STA: STA 103.6 MHz ran 142.9 (§39), STA 114.3 ran 166.7 (§41), ~70-85 STA ran 125 (CLAUDE.md). No tok/s is claimed from STA; the board --fclk sweep finds the real ceiling.
- The PS PLL only offers 1000/N divider steps, so board clocks are quantized to 125 / 142.9 / 166.7 / 200 MHz with NO step between 166.7 and 200 (175 snaps down, 183.3 snaps up) (WIDE-WORD-DATAPATH-LOG.md:1379-1381).
- The board driver must force the PL clock via set_and_verify_fclk() writing /sys/devices/platform/fclk0/set_rate; a flat fpgautil -b load does not apply the BD's PL0_REF FREQMHZ (pl_kv256.py:137, pl_seq_sb.py:106).
- The double-pump DP=1 split-brain fits and CLOSES timing at 200/400 MHz OOC (clk WNS +0.070, clk2x +0.074) and projects ~88,449 tok/s @200/400 in SIM. The full DP=1 build was never board-run; a MAC-only DP branch (`dp-hw-maconly`) ran on silicon bit-exact but the fabric→clk2x data feed walled at 50 MHz, killing the lever (DEAD, 100K-REVIEW.md:44). The record stays single-pump 59,965.5 (DOUBLE-PUMP-100K.md:8,34-35).
- impl_seq_kv.tcl and build_bd_seq_kv.tcl both guard against the silent-empty-ROM bug: build_bd HARD-ERRORS on any missing .mem (build_bd_seq_kv.tcl:64), impl greps every synth runme.log for [Synth 8-4445] and fails (impl_seq_kv.tcl:11-17) — first silicon shipped all-zero KV codes from missing inv_lut ROMs (log §38).
- Impl directives for these dense (URAM 100%, LUT/BRAM/DSP 90%+) designs: PLACE_DESIGN=AltSpreadLogic_high + ROUTE_DESIGN=AlternateCLBRouting (all impl scripts); the DP build adds PHYS_OPT_DESIGN=AggressiveExplore (impl_seq_sb_dp.tcl:15-18).

**Files**

- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/tcl/build_bd_seq_kv.tcl` — BD builder for the live KV engine (gemv_seqkv_gum lineage); tclargs P/LANES/WWORDS/FREQMHZ/TMAX; hardcodes mems dir stage3_vec_kvk8t128_smp; HARD-ERRORS on missing .mem
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/tcl/impl_seq_kv.tcl` — synth+impl+bitstream for the KV engine; emits seq_kv.bit/.bin (deployed as gemv_seqkv_gum); greps for empty-ROM [Synth 8-4445]
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/tcl/ooc_seq_vec.tcl` — OOC fit/Fmax probe for sequencer_vec; WWORDS = 3276800/LANES inverse-scaling rule
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/tcl/ooc_seq_sb.tcl` — OOC probe for split-brain sequencer_sb; tclargs P/LANES/PERIOD/WWORDS/TMAX/ND/NC/ATT2
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/tcl/build_bd_seq_sb.tcl` — BD builder for the record split-brain bitstream; forces CONFIG.DBG 0 / ATT2 0 for fit
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/tcl/impl_seq_sb.tcl` — synth+impl+bitstream for split-brain; emits gemv_seqsb.bit/.bin
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/tcl/ooc_seq_sb_dp.tcl` — OOC probe for double-pump DP=1 (clk + clk2x async group, mac_bank_dp/dsp_dp)
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/tcl/build_bd_seq_sb_dp.tcl` — BD builder for double-pump; adds Clocking Wizard MMCM (clk2x=2x pl_clk0, phase 180); sets DP=1
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/tcl/impl_seq_sb_dp.tcl` — impl for double-pump; adds AggressiveExplore phys-opt for 97% FF / 100% URAM density
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/WIDE-WORD-DATAPATH-LOG.md` — THE build log: §27 the 59,965.5 record, §37 K8/TMAX=128 shipping config, §38 empty-ROM bug, §39-45 KV silicon runs + gum bitstream (idcode SQRV, util figures)
- `/home/mikeayles/Desktop/Projects/kev-gpt/docs/DOUBLE-PUMP-100K.md` — the 100k plan; DP=1 OOC timing-validated at 200/400, ~88k SIM projected, silicon risks
- `/home/mikeayles/Desktop/Projects/kev-gpt/docs/BUILD-LOG.md` — narrative build history stages 0-8: A53 baseline, first-silicon empty-ROM prune bug, resident URAM weights, the honest thesis
- `/home/mikeayles/Desktop/Projects/kev-gpt/docs/7-kevin-remembers.md` — doc-7 faithful-stream campaign; LANES=256 GEMV cycle table, the KV engine spec
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/rtl/gemv_axi_seq_sb.v` — split-brain AXI shell; DBG/ATT2/DP generics live here (lines 17-31); IDCODE SQSB
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/rtl/gemv_axi_seq_vec.v` — KV engine AXI shell; P/LANES/NLAYER/WWORDS/TMAX generics only (no DBG/ATT2); IDCODE SQRV; SEED reg 0x30
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_kv256.py` — board driver for the KV gum bitstream; build_weight_image is LANES-dependent; set_and_verify_fclk
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/board/pl_seq_sb.py` — board driver for split-brain (SQSB); --n/--tmax/--fclk sweep; embed upload must match build TMAX
- `/home/mikeayles/.claude/projects/-home-mikeayles-Desktop-Projects-kev-gpt/memory/live-demo-topology.md` — OUT-OF-REPO source of truth for the live daemon invocation (kevkv.service loads gemv_seqkv_gum, --lanes 256 --tmax 128 --fclk 166.7e6, host-side temp 0.4/top-k 10)

**Gotchas**

- The gum bitstream is the KV engine, NOT split-brain. A prior measurement task was blocked by assuming the live bitstream was sequencer_sb and looking for its DBG/ATT2 — those generics don't exist on the KV engine. The correct params are P=8/LANES=256/TMAX=128, built 125 MHz target, runs 166.7 MHz.
- Build script DEFAULTS != the shipped build. build_bd_seq_kv.tcl defaults are TMAX=256 and freq=**40** (build_bd_seq_kv.tcl:25; the 166.666 default is build_bd_seq_sb.tcl's), but the gum bitstream was built with explicit tclargs tmax=128 and a 125 MHz target (per log §37/§45 and the daemon --tmax 128). Do not read the script defaults as the as-built params — the freq=40 default strengthens the inference that an explicit freq was passed.
- Missing .mem = silent all-zero ROM in the bitstream. A $readmemh the synthesiser can't open emits only CRITICAL WARNING [Synth 8-4445] and ships a grounded ROM (first silicon quantised every KV code to 0, log §38). Both build_bd_seq_kv.tcl (hard-error) and impl_seq_kv.tcl (log grep) now guard this; other engines' scripts only print a soft WARNING.
- Driver LANES and TMAX MUST match the bitstream. The weight-image packing (pl_kv256.build_weight_image / pl_seq_sb) is LANES-dependent and the embed upload streams exactly TMAX pos rows; a mismatch corrupts weights or pos>0 embeddings. This is why the daemon args (lanes 256, tmax 128) are load-bearing evidence for the build params.
- fpgautil -b does NOT set the PL clock. The BD's PL0_REF FREQMHZ is ignored on a flat load; the driver must force fclk0 via /sys/devices/platform/fclk0/set_rate and verify it, or the PL runs at the default (often 100 MHz) and all tok/s numbers are wrong.
- PL clock is quantized to 125/142.9/166.7/200 MHz (1000/N PLL steps). There is no clock between 166.7 and 200, so a design that fails at 200 is stuck at 166.7 for tok/s purposes unless an MMCM clk_wiz is added (the double-pump path). 175/183.3 requests silently snap to a neighbor.
- STA is not the ceiling on -2LV. A negative WNS (e.g. -3.752 @200-target, §41) can still run bit-exact at 166.7 on silicon (1.46x margin). Never claim tok/s from timing reports; only from a board --fclk sweep that checks bit-exactness 3/3.
- Aggressive timing target can make a full device UNROUTABLE. The 5 ns (200 MHz) target spread logic and hit 182,340 node overlaps (§42); the looser 8 ns (125 MHz) target packed tighter and routed MET (§43). On a URAM-100%/LUT-93% device, relax the target to route, then overclock on the board.
- Build outside OneDrive. All build dirs are C:/kevbuild/... because the OneDrive cldflt cloud-sync filter locks files mid-build and corrupts runs (BUILD-LOG.md §4). The .mem sim dirs also live under C:/kevbuild/.

**Open questions**

- The exact tclargs string used to build gemv_seqkv_gum is not in the repo (build dirs are on the Windows build box under C:/kevbuild/stage3_seqkv_bit). The params are firmly inferred (P=8/LANES=256/TMAX=128/125-target/166.7-run) from three agreeing sources but the literal invocation line / util_impl.rpt / timing_impl.rpt on the build box would confirm WWORDS and P beyond doubt.
- The gum bitstream contains on-chip Gumbel sampling (SEED reg 0x30). **UNRESOLVED** whether the deployed daemon samples on-chip or host-side: the memory note says host-side temp 0.4/top-k 10 (on-chip 0.85 "was gibberish"), but the committed code with those flags samples ON-CHIP at the baked temp and ignores top-k. Knowable only from the live kevkv.service ExecStart + the on-board a53_daemon.py on the Kria (which may be hand-edited vs repo), not the repo.
- Whether a gemv_seqsb (split-brain record) bitstream was ever deployed to the live chat, or only bench-measured, is not stated — the live daemon loads the KV gum bitstream, so the 59,965.5 record engine appears to be a bench/record artifact rather than the served engine.
- **RESOLVED**: the FULL double-pump DP=1 build (gemv_seqsb_dp) is OOC-timing-validated at 200/400 MHz and was never board-run. A MAC-only DP branch (`dp-hw-maconly`) DID run on silicon bit-exact — but the fabric→clk2x data feed walled at 50 MHz, so clk2x did NOT hold 400, and the lever is DEAD (100K-REVIEW.md:44).

---

## Reference Ladder, Gate Harnesses & Performance Model (Kevin-on-Kria, subsystem 7)

Kevin-on-Kria is validated against a chain of Python "goformer" references, each a strict refinement of the one before, so that the SystemVerilog RTL running in the KV260 fabric can be proven correct without ever trusting a speed number that hasn't first been proven bit-honest. The ladder runs float -> integer -> fixed-format -> fabric-precision non-linears -> integrated KV sequencer -> per-phase sequencer, and terminates in `fabric/stage3/seq_ref.py::IntSequencer`, which is the single bit-true contract the RTL FSMs are gated against. The chain is bound by two gates end to end: the float `goformer_seq` must emit the SAME token stream as the exact-float full-recompute model (cosine > 0.9999 on raw logits is treated as a brittle proxy; the *token stream* is the binding check), and the integer `seq_ref` must be bit-exact (mismatches=0) to the RTL.

The gate-harness pattern (`fabric/stage3/run_*.py`) is the inner loop for all RTL work: each harness writes the `.mem` ROMs the RTL `$readmemh`-loads, compiles with `iverilog -g2012`, runs `vvp`, reads back per-phase `.out` dumps, compares element-by-element against the Python reference, and prints one sentinel verdict line (`*_VERDICT` / `SEQ_VEC_*` / `SEQ_SB_*`). Block harnesses (`run_layernorm`, `run_softmax`, `run_gelu`, `run_banked`) each pin the fixed-point format of ONE op and prove it two ways: a module bit-exact gate against an integer reference, and a token-stream identity gate that plugs the integer op back into goformer. `seq_ref.py` imports the block references directly so there is exactly one source of truth per op.

`seq_ref.IntSequencer` exposes not just the final token but every phase intermediate through `block0_phase_signals()` and `full_forward_signals()` (ln1/qkv/ctx/attn/x_res/ln2/gelu/mlp/x_out, then x4/lnf/head/tok), so a P-wide RTL rewrite can be localised phase-by-phase rather than only on the emitted token. The current live gates are `run_vec_seq.py` (single-token P-wide full forward), `run_sb_seq.py` (the split-brain N=16 two-cohort engine), and `run_vec_kv.py` (the doc-7 faithful multi-token KV-window stream, gated against the K8/V8 `goformer_kvq` reference).

Two performance models sit alongside the gates. `cycle_model.py` predicts single-stream cyc/token by SUMMING stage latencies (decode is autoregressive, no cross-stage overlap) and shows the serial non-linears cap the sequencer ~10-15k regardless of GEMV width. `batched_model.py` predicts aggregate serving throughput by taking Fmax / busiest-shared-unit cycles (batching PIPELINES streams through shared units), and prices the three levers to 100k. `fabric/progress.py` is the authoritative MEASURED tok/s milestone ladder, from 0.07 tok/s (first on-fabric generation) to the 59,965.5 tok/s @200MHz split-brain record, plus the doc-7 faithful-stream rungs.

## 1. The goformer reference ladder

Each module is runnable standalone (`python -m model.goformer_*`) and prints a `GOFORMER_*_PASS/FAIL` verdict. They form a strict refinement chain; each level changes exactly one thing and re-proves the gate.

### 1.1 `model/goformer_full.py` — full integer forward (the base)
- The whole Kevin forward with a **pluggable matmul**: every quantised Linear does INT8 activation quant → `matmul(int_w, int_x)` → per-channel dequant, everything else (embeddings, LayerNorm, causal attention, GELU, sampling) in float exactly as `model/qgpt.py` (`model/goformer_full.py:1-13`).
- Float ops matched to torch defaults: `layernorm` uses biased/population variance, eps 1e-5, gamma-only (`goformer_full.py:30-33`); `gelu` is exact erf GELU (`goformer_full.py:36-37`); `softmax` max-subtract (`goformer_full.py:40-43`); `numpy_matmul` is exact int64 GEMV (`goformer_full.py:46-48`).
- `qlin(x, layer)`: `layer = (int_w, w_scale, s_act)`; `ix = clip(round(x/s_act), -128, 127)`, `out = matmul(int_w, ix[t]) * w_scale * s_act` (`goformer_full.py:68-77`).
- `forward`: `x = tok_emb[idx] + pos_emb[arange(T)]`, then per block `x += attn(ln1(x)); x += mlp(ln2(x))`, then `ln_f`, then head (`goformer_full.py:99-108`).
- `params_from_ckpt` builds the params dict from a Brevitas QGPT checkpoint; `save_params`/`load_params` flatten to a single `.npz` the board loads with numpy only (`goformer_full.py:125-188`). `fabric/export/goformer.npz` is the canonical exported model everything loads.
- **Gate**: `_validate` compares to the torch QGPT, `GOFORMER_FULL_PASS` when cosine > 0.9999 (`goformer_full.py:191-214`). Because the PL GEMV is bit-exact to `int_w @ int_x`, validating the numpy version vs QGPT is enough to trust the fabric one — the bit-honest bridge (`goformer_full.py:9-13`).

### 1.2 `model/goformer_kv.py` — incremental KV-cached decode
- Processes only the NEW token each step (O(T) not O(T²)), caching each layer's K/V; the prerequisite for 10k+ (`goformer_kv.py:1-15`).
- `KVDecoder` wraps a `GoformerFull` (or any engine) and reuses its exact `qlin/_ln/_smax/_mlp` so arithmetic matches (`goformer_kv.py:24-33`).
- `_attn_step`: computes qkv of the new token, vstacks k/v into `k_cache[bi]`/`v_cache[bi]`, `scores = (qh @ kh.T)/sqrt(hd)` with **no mask** (all cached positions are causal by construction) (`goformer_kv.py:41-59`).
- **Gate**: for every prefix, KV last-token logits == full-recompute last-token logits with `maxabsdiff == 0`, AND greedy stream identical → `GOFORMER_KV_PASS` (`goformer_kv.py:115-143`). Bit-IDENTICAL by causality, not approximate.
- Also holds `build_random_params(seed, n_layer=4, n_head=4, d=256, d_mlp=1024, vocab=193)` — a checkpoint-free params dict used across the ladder's equivalence gates when the `.npz` is absent (`goformer_kv.py:85-112`).

### 1.3 `model/goformer_q.py` — pin the fixed-point Q-format
- `goformer_fixed` proved *op* precision on a float residual stream; the hardware can't carry floats, so this quantizes the whole datapath to signed Q(int.frac), profiles ranges, finds the smallest (int,frac) still holding cosine > 0.9999 (`goformer_q.py:1-11`).
- `FixedPointGoformer` quantizes ONLY the residual stream (the value carried/added between blocks) via `_q`; LN outputs feed the GEMV INT8 requant so they need no residual-format rounding (`goformer_q.py:31-53`).
- `_validate` sweeps `frac in (16,18,20,22,25)`, prints the passing floor, and pins the recommended datapath: **residual = signed Q6.25 (32-bit)** matching the GEMV accumulator width; activations→GEMV INT8; GEMV accum INT32; dequant scale 24-bit; GELU LUT 8192@[lo,hi]; exp Q1.20 z∈[-16,0] (`goformer_q.py:78-98`). This is the format table the RTL is built to.

### 1.4 `model/goformer_fixed.py` — fabric-precision non-linears
- The transcendentals can't be bit-exact to float, so the gate softens to cosine > 0.9999; this module implements each the way the fabric will (LUTs + fixed-point + Newton) and MEASURES the precision needed before any RTL is written (`goformer_fixed.py:1-18`).
- `FixedConfig` (frozen dataclass) holds the MEASURED knobs: `gelu_n=8192`, `gelu_lo/hi=±8` (resized per model), `exp_zmax=16.0`, `exp_frac=20` (Q1.20), `rsqrt_seed_bits=8`, `rsqrt_iters=2`, `scale_mant_bits=24` (`goformer_fixed.py:36-48`). Comment: these are LARGER than per-op error predicts because char-level logits amplify coherent error — the measured precision crossover (`goformer_fixed.py:30-35`).
- `gelu_lut` (linear interp over LUT), `rsqrt_fixed` (8-bit seed + Newton `y←y(1.5−0.5·a·y²)`), `fixed_layernorm` (population var), `fixed_softmax` (running-max subtract, exp LUT clamp [-zmax,0], causal -inf → exactly 0, Q1.f round) (`goformer_fixed.py:64-93`).
- **Gate**: `GOFORMER_FIXED_PASS` when cosine > 0.9999 AND same argmax at every position (`goformer_fixed.py:174-206`). `_profile_gelu_range` sizes the LUT domain to the max |GELU input| the real model produces (`goformer_fixed.py:160-171`) — used everywhere via `dom = ceil(profile+1)`.

### 1.5 `model/goformer_seq.py` — integrated sequencer reference
- Composes the two verified prerequisites (KV decode + fabric non-linears) into the exact per-token dataflow the FSM runs: `embed → 4×[LN,qkv,attn(+KV),proj,+res,LN,mlp(GELU),+res] → LN_f → head → argmax → append → loop` (`goformer_seq.py:1-16`).
- `build_engines` returns `(p, src, GoformerFull, FixedGoformer)` with the LUT domain profiled from the model (`goformer_seq.py:29-37`).
- **Gate**: `KVDecoder(engine=fixed_engine).generate_greedy` must equal the float full-recompute greedy stream → `GOFORMER_SEQ_PASS` (`goformer_seq.py:40-71`). This is the golden token stream the RTL sequencer is validated against.

### 1.6 `fabric/stage3/seq_ref.py` — the per-phase integer sequencer (THE contract)
This is the load-bearing module. It does PURE INTEGER arithmetic step-for-step the way the FSM does, and is itself gated (identical token stream) against float `goformer_seq`. Two gates chain to bind hardware to the float model (`seq_ref.py:11-16`):
```
float goformer_seq  --(cosine>0.9999, same token)-->  seq_ref (integer)
seq_ref             --(bit-exact, mismatches=0)    -->  rtl/sequencer*.sv
```

**Pinned glue formats** (the arithmetic NOT covered by any single block's gate — residual→LN→INT8 act-quant→INT32 GEMV→per-channel dequant→Q6.25; INT32 scores→Q8.8 softmax; probs·V→ctx; argmax over INT32 logits) (`seq_ref.py:5-34`):
- residual x: signed **Q6.25** (`RESID_FRAC=25`, `RESID_INT=6`, `seq_ref.py:56-57`)
- LN gamma Q4.20, LN out Q.22; GEMV act-quant INT8; weights INT4, accum INT32; dequant = 24-bit mantissa·2^exp (`SCALE_MANT_BITS=24`, `seq_ref.py:58`)
- attention: q/k/v stored **Q.16** (`VFRAC=16`), /sqrt(64) = `>>3` (`ISQRT=3`), softmax prob Q1.20 (`PROB_FRAC_A=20`), `NHEAD=4`, `HEAD_DIM=64` (`seq_ref.py:61-64`)

**Exact-integer helpers** (bit-identical to RTL names): `q_round_div` round-half-away-from-zero divide (`seq_ref.py:67-71`); `rsh_round` arithmetic right shift with round-half-away, `s<=0`→left shift, "bit-identical to rtl/sequencer.rsh_round" (`seq_ref.py:74-79`); `sat`, `to_q`, `quantize_scale_24` (folds per-channel scale to 24-bit mant·2^exp, matches `goformer_fixed._quantize_mantissa(scale,24)`) (`seq_ref.py:82-109`).

**`IntSequencer`** (`seq_ref.py:112-372`):
- `__init__` pre-quantizes embeddings to Q6.25, ln_f to Q4.20, and pre-pins every GEMV layer (`_pin_layer` folds `w_scale*s_act` → 24-bit (mant,exp)) (`seq_ref.py:118-143`).
- `INV_SACT_SH=40`: `inv_sact = round(2^40/s_act)`; `_act_quant` does `sat(rsh_round(y_q22*inv, LN_OUTFRAC+40), -128,127)` (`seq_ref.py:158-173`) — must match RTL `ISH`.
- `_ln` delegates to `run_layernorm._ln_int_quantized` (ONE source of truth) → Q.22 (`seq_ref.py:151-155`).
- `_gemv_int` exact INT32; `_dequant_to_q` per-channel `rsh_round(y_int*mant, -(exp+out_frac))` bit-identical to RTL dequant (`seq_ref.py:175-201`).
- `_attn_step` (`seq_ref.py:204-244`): LN1 → act-quant → qkv GEMV → dequant to Q.16 → split q/k/v (C=256), append to k/v cache. Per head h (base=h·64): scores `acc = Σ_d q·k` (Q.32) → `s_q88 = rsh_round(acc, 2·16+3−8)`, sat to int16 → `int_softmax_q` (Q1.20) → `ctx[d] = rsh_round(Σ_j prob·v, 20+16−25)` Q6.25. Then `_proj_after_attn`.
- `_proj_after_attn`: ctx Q6.25 → Q.22 (`>>3`) → act-quant → proj GEMV → Q6.25 (`seq_ref.py:246-255`).
- `_mlp_step` (`seq_ref.py:258-276`): LN2 → act-quant → mlp_fc GEMV → dequant to Q4.12 (`GELU_FRAC`), sat int16 → `gelu_q` LUT → carry as Q.22 (`<<(22−12)`) → act-quant → mlp_proj GEMV → Q6.25.
- `step` (`seq_ref.py:279-292`): full decode step → `(logits_real, argmax)`.
- `step_head_q25` (`seq_ref.py:294-312`): returns the **Q6.25 head-logit integers the fabric argmax and the on-chip Gumbel-max sampler operate on** (`head_q25 = _dequant_to_q(head, gemv_int, 25)`), argmax-equivalent to `argmax(logits_real)`; used by the sampling gate.

**Per-phase signal exposure** (the reason this module exists for RTL localisation):
- `block0_signals(idx_t)` → `{x_in_q25, x_out_q25}` (`seq_ref.py:314-325`).
- `block0_phase_signals(idx_t)` → every phase intermediate via `sink`: `x_in_q25`, then from `_attn_step`'s sink `ln1_out_q22, qkv_ix, qkv_yint, qkv_q16, ctx_q25, attn_out_q25`, then `x_res1_q25`, from `_mlp_step` `ln2_out_q22, gelu_q22, mlp_out_q25`, then `x_out_q25` (`seq_ref.py:212-215, 242-243, 253-254, 272-275, 327-342`).
- `full_forward_signals(idx_t)` → the keys the P-wide sequencer_vec/sb gate on: **`x4_q25`** (residual after last block), **`lnf_q22`** (LN_f out), **`head_q25`** (Q6.25 argmax-equivalent head logits), **`tok`** (emitted argmax). Resets KV/t first (single token at pos 0) (`seq_ref.py:344-361`).
- **Gate** `_validate`: integer stream must be identical to float full-recompute → `SEQREF_PASS`. Note (`seq_ref.py:416-419`): the BINDING gate is the token stream — raw-logit cosine is a brittle proxy because INT8 requant half-boundaries flip sub-1e-7 detail; even float `goformer_seq` only holds ~0.99995 worst-step.

### 1.7 `model/goformer_kvq.py` — the doc-7 KV-quantised reference
Used by `run_vec_kv.py` as `IntKVQSequencer(p, cfg, kbits=8, vbits=8, rotate=False, divfree=True)` (`run_vec_kv.py:77`). Docstring in the harness describes it as the "KVarN K4/V4+Hadamard contract" but the live call uses **K8/V8, rotate=False, divfree=True** — the INT8-KV reference change of doc-7 (the faithful-stream campaign). Full internals not read here (see the file directly, 23KB) — **open question flagged below**.

---

## 2. The gate-harness pattern (`fabric/stage3/run_*.py`)

Every harness follows the same skeleton: (1) build refs from `fabric/export/goformer.npz` via `seq_ref.build(npz)` which profiles the GELU domain and returns `(p, cfg)` (`seq_ref.py:375-380`); (2) write the `.mem` ROMs the RTL loads; (3) `iverilog -g2012 -o sim.vvp <-D defines> <TB> <RTL files>`; (4) `vvp sim.vvp` in the sim dir, require `TB_DONE` in stdout or print `IVERILOG_COMPILE_FAIL`/`VVP_FAIL`; (5) read per-phase `.out` files, compare to the Python reference element-by-element; (6) print one sentinel verdict line.

**Block harnesses — each pins one op's format, gates two ways:**

### 2.1 `run_layernorm.py` (imported by seq_ref as the LN source of truth)
- Pinned formats (`run_layernorm.py:30-41`): input **Q6.25** (`QX=25`), gamma **Q4.20** (`G_FRAC=20`), rsqrt input/result **Q.26** (`A_FRAC=Y_FRAC=26`), output **Q.22** (`OUT_FRAC=22`), d·d Q12.50 (`VAR_FRAC=50`), 64-entry rsqrt seed LUT (`SEED_IDX_BITS=6`, `SEED_OUT_FRAC=16`).
- `rsqrt_int` = 64-entry seed LUT (priority-encode normalize) + 2 Newton steps, exact integer (`run_layernorm.py:60-92`).
- `_ln_int_quantized(X,G)`: `mean = ΣX >> 8`, `var = Σd² >> 8`, `A = (var>>(50−26))+eps`, `Yr=rsqrt_int(A)`, `y = (d·Yr·G) >> 49` (`sh = 25+26+20−22`) → Q.22 (`run_layernorm.py:95-117`).
- **Gates** (`run_layernorm.py:199-263`): module bit-exact RTL vs `_ln_int_quantized` over 64 cases incl. edges (all-zero, max-magnitude, wide-spread), `mismatches=0`; plus token-stream identity via `IntLNGoformer` → `LN_VERDICT bitexact=… token_stream_identical=…`.

### 2.2 `run_softmax.py` (imported by seq_ref: `exp_table`, `int_softmax_q`, `quantize_scores`, `SCORE_FRAC`, `PROB_FRAC`)
- Pinned (`run_softmax.py:30-37`): score **Q8.8** (`SCORE_FRAC=8`), exp out **Q1.20** (`EXP_FRAC=20`), z-range [-16,0] (`ZMAX=16`), prob **Q1.20**, reciprocal `floor(2^40/sum)` (`RECIP_R=40`), LUT `NZ+1 = 4097` entries.
- `exp_table`: entry i = `round(exp(-i/256)·2^20)` (`run_softmax.py:40-43`). `int_softmax_q`: max-subtract, gather exp, `r = 2^40 // sum`, `prob = (e·r) >> 20` (`run_softmax.py:56-70`).
- **Gates** (`run_softmax.py:118-217`): module bit-exact vs `int_softmax_q` on real score rows profiled from the model + synthetic edges; plus token-stream identity via `IntSmaxGoformer` → `SOFTMAX_VERDICT`.

### 2.3 `run_gelu.py` (imported by seq_ref: `gelu_table`, `gelu_q`, `FRAC`)
- Format I/O **Q4.12** (`FRAC=12`), 8192-entry LUT, `index=(x+0x8000)>>3`, low 3 bits = interp fraction (`run_gelu.py:1-8,21-24`).
- `gelu_table`: entry i ≈ `gelu(i/512 − 8)`; `gelu_q`: `i=u>>3, f=u&7, lut[i] + ((lut[i+1]−lut[i])·f >> 3)` arithmetic shift (`run_gelu.py:27-41`).
- **Gate** (`run_gelu.py:64-116`): module bit-exact vs `gelu_q` (resolves pipeline latency by best-shift), plus end-to-end cosine (~0.9994, explicitly labelled a BRITTLE proxy — binding gate is the token stream) → `GELU_VERDICT`.

### 2.4 `run_banked.py` — the GEMV core gate
- Generates (M,K) vectors via `pack_banked.write_case`, compiles `gemv_banked.sv`, checks bit-exact via `pack_banked.check` → `BANKED_VERDICT` (`run_banked.py:1-51`). Tests non-divisible M (e.g. M=40, lanes=16).

**Integrated harnesses — gate the assembled sequencer against `seq_ref`:**

### 2.5 `run_sequencer.py` — the tiered integration gate
Three modes (`run_sequencer.py:1-29`):
- `--nlayer 1`: BLOCK-0 gate, one full block, residual `x_out` bit-exact vs `block0_signals` → `SEQ_VERDICT block0_bitexact`.
- `--nlayer 4`: FULL-FORWARD gate (single token pos=0), `tok_out` vs `step` argmax → `SEQ_VERDICT tokens_identical`.
- `--multitoken`: autoregressive gate, FSM primes a resident prompt then greedily decodes NGEN with per-block persistent KV, stream vs `generate_greedy` → `SEQ_VERDICT multitoken`.
- Writes every ROM the RTL loads: tok_emb/pos_emb (Q6.25), gamma (Q4.20), wrom (transposed banked INT4), dq_mant/dq_exp (24-bit), inv_sact (`round(2^40/s_act)`), prompt, plus seed/exp_lut/gelu_lut (`run_sequencer.py:25-28`).
- **`write_mems_wideword`** (`run_sequencer.py:243-282+`): for `sequencer_vec`, emits the standard 1-wide ROMs PLUS P-wide-read ROMs packed wide-word (`tok_emb_w`/`pos_emb_w`/`gamma_w` = P×32b/word, `dqm_w`/`dqe_w`). Per log §36 fit-plan 2, it also APPENDS the embed tables to `wrom.mem` so the RTL reads them from the resident weight URAM's spare depth through the GEMV bank's idle port-B instead of dedicated BRAM (requires `EPW≥1`, i.e. `LANES≥8·P`). This is the "wide-word banking not [P][rows]" gotcha realised in the ROM layout.

### 2.6 `run_vec_seq.py` — single-token P-wide full-forward gate
- `run(sim_dir, tok, P, lanes=16, tmax=256)`: builds `full_forward_signals(tok)`, writes wide-word mems, splits gelu LUT into even/odd (`gelu_lut2` paired-lane core), compiles `sequencer_vec.sv` + kv_bank/layernorm_vec/vec_dequant/vec_attn/vec_gelu/gelu_lut/gelu_lut2/softmax/gemv_banked_resident_vec (`run_vec_seq.py:35-67`).
- PHASES gated: `x4.out↔x4_q25` (256), `lnf.out↔lnf_q22` (256), `head.out↔head_q25` (193); binding check is `tok` (`run_vec_seq.py:22, 74-93`). Reports `fwd_cyc` from `cyc.out`. Prints `SEQ_VEC_FULL … ALL=… fwd_cyc=…` (`run_vec_seq.py:95-103`). Defines passed: `-DTOK -DPVAL -DLVAL -DWROMN -DTMAXVAL`.

### 2.7 `run_sb_seq.py` — the split-brain N=16 gate (the live engine)
- `run(sim_dir, toks, P, lanes=128, tmax=32, nd=0, att2=1, dp=0)`: builds `full_forward_signals` per token (16 tokens, `TOKS16` default), each stream compared independently to seq_ref (tok + x4 + lnf + head) (`run_sb_seq.py:37-46, 91-111`).
- Compiles the split-brain RTL set: `mac_bank_dp`, `mac_bank_dsp_dp`, **`sequencer_sb`**, `cohort_engine`, `nl_engine`, layernorm_vec, vec_dequant, vec_attn, vec_gelu, gelu_lut/2, softmax, `weight_bank_tdp`, `embed_bank_tdp`, `gemm_cohort_vec`, `gemm_banked_resident_vec` (`run_sb_seq.py:64-79`).
- Parameters: `NDVAL` (DSP-packed GEMM streams/cohort), `ATT2VAL` (1 = per-cohort `vec_attn`; 0 = shared+arbiter, the fitting BD config), `DPUMP` (double-pump MAC at 2 K-steps/clk, use with nd=0). (`run_sb_seq.py:57-62, 136-141`).
- Prints `SEQ_SB_FULL N=… … cyc_total=… cyc/token=…` and `SEQ_SB_TOKS/S @166.7=… @200=…` (`run_sb_seq.py:119-123`). Target ~63k cyc / 16 tok vs 99,828 single-engine (`run_sb_seq.py:1-5`).

### 2.8 `run_vec_kv.py` — the doc-7 faithful multi-token KV gate
- Drives PLEN prompt positions + NGEN greedy-feedback positions through repeated GO pulses (`kv_bank` persists across passes); the generated stream must equal `IntKVQSequencer(kbits=8,vbits=8,rotate=False,divfree=True).generate_greedy` (`run_vec_kv.py:1-10, 70-81`).
- Also gates on-chip Gumbel-max SAMPLING: with `--seed≠0`, `_sample_stream` mirrors the TB pass schedule exactly (prompt passes greedy, gen passes sampled via `GumbelRng(seed)` perturbing the SAME Q6.25 `step_head_q25` logits) (`run_vec_kv.py:44-67`). Writes `gumbel_lut.mem` from `gumbel.make_gumbel_lut()` (single source of truth) and split `inv_lut_lo/hi.mem` for the kv_bank K8 inverse-scale ROMs (`run_vec_kv.py:92-109`).
- Compiles `sequencer_vec` + `vec_attn_w` + `softmax_f` (the faithful variants) (`run_vec_kv.py:27-29`). Prints per-pass cyc min/max/avg and `VEC_KV_VERDICT match=… mode=greedy|sample`.

---

## 3. The performance models

### 3.1 `cycle_model.py` — single-stream cyc/token (SUM of stages)
- Honest premise: decode is data-dependent and autoregressive (layer L needs L−1's residual; token t+1 needs token t's id), so per-token latency is the SUM of stage latencies on the critical path — a wide GEMV does NOT overlap the non-linears unless they are pipelined element-by-element (`cycle_model.py:1-19`).
- Constants: `RLAT=2`, `N_LAYER=4, N_HEAD=4, D=256, D_MLP=1024, VOCAB=193` (`cycle_model.py:23-24`). Per-token GEMVs: qkv(768,256), proj(256,256), mlp_fc(1024,256), mlp_proj(256,1024) (`cycle_model.py:27`).
- `gemv_cycles(PE) = Σ ceil(M/PE)·(K+2)` over layers ·4 + head (`cycle_model.py:34-38`); `attn_cycles(PE,T) = ceil(2·(D/nh)·T·nh·nl / PE)` (`cycle_model.py:41-44`).
- `nonlinear_cycles(T, mode)`: **serial** (built today) `ln_each=770, sm_each=3T+45, gelu_each=D_MLP+3`, with `n_ln=8, n_sm_rows=16, n_gelu=4`; **parallel** (PROJECTED) `ln_p=23, sm_p=T+12, gelu_p=64` (`cycle_model.py:47-60`).
- Fmax by OOC synth on xck26-2LV (0 DSP, ~1.5MB): PE=256→293MHz, 512→292, 1024→239 (the LUT wall — pure-LUT INT4 MACs) (`cycle_model.py:76-78`).
- **Finding**: serial non-linears (8×770 LN + 16 softmax) dominate, capping the assembled single-stream sequencer ~10-15k regardless of GEMV width. 10k reachable (PE≥1024, short ctx); 100k needs re-architected parallel non-linears + wide GEMV + moderate context. Tags: GEMV/attn DERIVED, non-linear-serial MEASURED, non-linear-parallel PROJECTED (`cycle_model.py:84-91`).

### 3.2 `batched_model.py` — aggregate serving throughput (Fmax / busiest unit)
- Batching B streams PIPELINES them through SHARED units (while A is in softmax, B is in GEMV, C in GELU…), so aggregate throughput = Fmax / busiest-single-unit cycles, NOT the sum (`batched_model.py:1-13`).
- `ONCHIP_LEFTOVER_KB=1216` after ~1.5MB weights (16 URAM + 144 BRAM) (`batched_model.py:25`). `kv_per_stream_kb(T, bits=8) = nl·2·T·D·bits/8/1024` (`batched_model.py:28-29`).
- `unit_cycles` returns per-unit busy cyc for {GEMV(÷n_gemv), LN, softmax, GELU}; `batched` computes `bottleneck=max(u)`, `b_fill=ceil(single_sum/bottleneck)`, `agg=fmax/bottleneck`; KV caps B to `b_kv = leftover/kv_per_stream`, real_agg scales down if KV-capped (`batched_model.py:32-61`).
- **Finding**: single-stream ~10k → ~30k just by pipelining B~4; then GEMV is the bottleneck; splitting it across 3-4 engines (into the 1248 idle DSPs) reaches ~100k AGGREGATE at B~4, T=128, INT8 KV (which fits on-chip). 100k is a SERVING number (B concurrent users), not single-stream. Tags: GEMV DERIVED, non-linear-serial MEASURED, parallel + multi-engine PROJECTED (`batched_model.py:64-81, 16-18`).

Note: these two models predate the split-brain measured record; the actual campaign path (split-brain on TDP URAM) is what delivered the 59,965.5 measured number, distinct from the batched model's multi-GEMV-engine projection.

---

## 4. The MEASURED tok/s milestone ladder (`fabric/progress.py`)

`LADDER` is an ordered list of `(label, tok/s, tag, removed-by-this-step)` (`progress.py:23-82`). Tags: **MEASURED** (real silicon, token-stream bit-exact), **SIM** (RTL bit-exact in iverilog vs seq_ref, bitstream pending), **PROJECTED** (modelled). Rungs 1-4 are the CPU-in-loop path (PE=1, asymptotes to the A53 ~11); rung 5 is the architectural jump (sequencer takes CPU out of loop); 6+ widen it.

| # | Rung | tok/s | tag | config / removed |
|---|------|-------|-----|------------------|
| 1 | PL re-stream weights/forward (Python AXI, O(T²)) | 0.07 | MEASURED | first on-fabric generation |
| 2 | + resident weights in URAM | 0.22 | MEASURED | per-token weight movement (~83%) |
| 3 | + KV cache (incremental decode) | 2.71 | MEASURED | the O(T²) full-context recompute |
| 4 | + C MMIO driver | 10.35 | MEASURED | Python per-poke overhead |
| 5 | HW sequencer @40MHz (CPU out of loop) | 44.32 | MEASURED | the A53 non-matmul-forward ~11 wall |
| 6 | + resident-read GEMV | 75.8 | SIM | re-streaming each matmul's weights (~42% cyc) |
| 7 | + PE=256 wide lanes | 231.0 | MEASURED | GEMV run phase (16× fewer group passes), @40MHz |
| 8 | + GELU stream + LN pipeline (PE=128 @125MHz) | 751.78 | MEASURED | GELU stall + LN cascade; Fmax 50→125 (STA-safe 71/430) |
| 9 | + wide P-lane datapath (P=4 LN/dq/attn/GELU @100MHz) | 1882.7 | MEASURED | 1-elem/cyc serial loops; 53,116 cyc/tok |
| 10 | + BRAM sync-read scratch (P=8 @125MHz) | 2483.9 | MEASURED | LUTRAM→BRAM; 50,324 cyc/tok; STA 79.5, breaks 142.9 |
| 11 | + P-wide GEMV boundary (act-feed + readback P/cyc) | 3511.6 | MEASURED | 1-elem/cyc GEMV loops; 35,596 cyc/tok; 125MHz 3/3 |
| 12 | + LANES=256 (72-bit URAM banks, 60/64) | 5448.8 | MEASURED | half the GEMV passes; 22,941 cyc/tok; breaks 142.9 |
| 13 | + cycle-floor cut + deep pipeline (fused RB+DQ+GELU, P-wide attn @166.7MHz) | 9295.4 | MEASURED | attn scalar load + dequant round-trip + Fmax 85→131; 17,930 cyc/tok |
| 14 | + 3-stage act-quant → 200MHz | 11143.9 | MEASURED | last sub-8ns path (BRAM→mux→DSP); 17,947 cyc/tok 3/3 |
| 15 | + batch GEMM N=4 | 16969.3 | MEASURED | per-stream weight reads; 47,144 cyc/4 tok @200; 3/3 ×4 |
| 16 | + ping-pong N=8 (NL overlaps GEMM @166.7MHz) | 17740.6 | MEASURED | non-linear bubbles; 75,157 cyc/8 tok; 200 fails |
| 17 | + single-pass merge N=8 | 19275.6 | MEASURED | second weight pass; 69,172 cyc/8 tok @166.7 |
| 18 | + N=16: 12 DSP-packed banks, shared LN/attn (106.5k LUT) | 24134.0 | MEASURED | per-stream MAC fabric; 110,494 cyc/16 tok @166.7 |
| 19 | + softmax latency cut | 25744.5 | MEASURED | dead wait-states exp/sum/recip; 103,582 cyc on silicon |
| 20 | + SPLIT-BRAIN N=14: two cohorts on dual-ported URAM @166.7 | 36970.7 | MEASURED | single weight read port; 63,113 cyc/14 tok; 14/14 |
| 21 | + N=16 @200MHz (LN un-retimed + AQ 32×48 range-proof) | 46604.4 | MEASURED | LN qsh/output + DSP famine; 68,663 cyc/16 tok; 250 fails |
| 22 | + schedule pipelining (AQ/RUN overlap, stream-granular NL, attn call cuts) | 56262.7 | MEASURED | GE/nl ping-pong + per-call attn; 56,876 cyc/16 tok @200 |
| 23 | **+ TMAX=16 + per-cohort attn + CTX stream + LN prod·gamma split** | **59965.5** | **MEASURED** | shared-attn wall + LN critical path; **53,364 cyc/16 tok @200; 16/16 bit-exact 3/3; 250 hangs** |
| — | FAITHFUL N=1, T=128 window (doc-7 R1-R4f, on-chip KV @142.9MHz) | 11343.2 | MEASURED | T=1 degenerate attn; 119-tok message, 12,594 cyc/tok avg, 3/3 |
| — | + R5 cone ladder → 166.7MHz | 13162.3 | MEASURED | timing cones; 12,662 cyc/tok, 3/3; 200 fails MAC floor |
| — | + schedule trims + MAC stage | 16087.5 | MEASURED | ~2.2k cyc; 10,360 cyc/tok, 3/3 @166.7 |
| — | × faster clock OR fewer cycles (MMCM ~187MHz, or K4/smaller) | 20000.0 | PROJECTED | PS PLL caps 166.7; 20k needs ~207MHz → MMCM ~187 gets ~18k |

Reference baselines (guide lines, all MEASURED, same model B=1 greedy): A53 chat 11.0, A53 INT4 GEMV bench 177.8, XPS15 torch CPU 356.0, XPS15 RTX 3050 Ti 719.0, XPS15 ONNX Runtime CPU best 1273.0 (`progress.py:85-89`).

**The MEASURED record is rung 23: 59,965.5 tok/s @200MHz, split-brain N=16 TMAX=16 wave, 53,364 cyc, 16/16 bit-exact, 3/3.** The two doc-7 axes are honestly separated: rungs 1-23 decode with attention T=1 (blistering aggregate, degenerate text); the FAITHFUL rungs are ONE stream with the full on-chip KV window (real messages), a different metric labelled as such (`progress.py:70-82`).

`print_table` renders the table + baselines + a long note; `plot` renders `fabric/progress.png` (log-scale bars, colour by tag, ×multiplier annotations, baseline lines) (`progress.py:92-179`). `python -m fabric.progress [--plot PATH]`.

**Parameters**

| name | meaning | live | range | where |
|---|---|---|---|---|
| `RESID_FRAC` | Residual stream fixed-point fraction bits (signed Q6.25, 32-bit) — the value carried/added between blocks | 25 | fixed | fabric/stage3/seq_ref.py:56 |
| `SCALE_MANT_BITS` | Dequant per-channel folded scale mantissa width (mant*2^exp) | 24 | fixed | fabric/stage3/seq_ref.py:58 |
| `VFRAC` | q/k/v stored fraction bits (Q.16) in the attention datapath | 16 | fixed | fabric/stage3/seq_ref.py:62 |
| `ISQRT` | 1/sqrt(head_dim=64) implemented as right-shift by 3 | 3 | fixed | fabric/stage3/seq_ref.py:63 |
| `INV_SACT_SH` | inv_sact = round(2^40/s_act); must match RTL ISH for INT8 act-quant | 40 | fixed | fabric/stage3/seq_ref.py:158 |
| `NHEAD / HEAD_DIM` | attention heads / per-head dim (C=256=4*64) | 4 / 64 | fixed | fabric/stage3/seq_ref.py:59-60 |
| `OUT_FRAC (LN)` | LayerNorm output Q.22 (feeds GEMV INT8 requant) | 22 | fixed | fabric/stage3/run_layernorm.py:35 |
| `G_FRAC (LN gamma)` | gamma Q4.20 | 20 | fixed | fabric/stage3/run_layernorm.py:32 |
| `SCORE_FRAC / PROB_FRAC` | softmax score Q8.8 in, prob Q1.20 out | 8 / 20 | fixed | fabric/stage3/run_softmax.py:30,33 |
| `EXP_FRAC / ZMAX / RECIP_R` | exp LUT Q1.20, z-range [-16,0], reciprocal floor(2^40/sum) | 20 / 16 / 40 | fixed | fabric/stage3/run_softmax.py:31-34 |
| `FRAC / N_LUT (GELU)` | GELU I/O Q4.12, 8192-entry LUT + 3-bit interp | 12 / 8192 | fixed | fabric/stage3/run_gelu.py:21,23 |
| `FixedConfig knobs` | MEASURED precision floor: gelu_n=8192, exp_frac=20, rsqrt_seed_bits=8, rsqrt_iters=2, scale_mant_bits=24 | 8192/20/8/2/24 | measured floor | model/goformer_fixed.py:36-48 |
| `RLAT` | per-group K-stream latency added to each GEMV pass | 2 | fixed | fabric/stage3/cycle_model.py:23 |
| `N_LAYER/N_HEAD/D/D_MLP/VOCAB` | model shape used by both perf models | 4/4/256/1024/193 | fixed | fabric/stage3/cycle_model.py:24 |
| `ONCHIP_LEFTOVER_KB` | on-chip KB left for KV after ~1.5MB weights (16 URAM + 144 BRAM) | 1216 | fixed | fabric/stage3/batched_model.py:25 |
| `P (run_vec_seq/sb --p)` | P-wide datapath lanes for LN/dequant/attn/GELU | 8 (default) | harness arg | fabric/stage3/run_vec_seq.py:109 / run_sb_seq.py:133 |
| `lanes (--lanes)` | GEMV bank width (LANES); wide-word banking requires LANES>=8*P for embed-in-wrom | 128 (sb default) | harness arg | fabric/stage3/run_sb_seq.py:134 |
| `tmax (--tmax TMAXVAL)` | on-chip KV window depth baked into RTL | 32 (sb default), 16 (record) | harness arg | fabric/stage3/run_sb_seq.py:135 |
| `att2 (ATT2VAL)` | 1=per-cohort vec_attn; 0=shared+arbiter (the fitting BD config) | 1 (sim default), 0 (bitstream fit) | 0|1 | fabric/stage3/run_sb_seq.py:136-137 |
| `nd (NDVAL)` | DSP-packed GEMM streams per cohort | 0 (default) | harness arg | fabric/stage3/run_sb_seq.py:132 |
| `seed (SEEDVAL)` | on-chip Gumbel-max sampling seed (0 => greedy argmax) | 0 (greedy default) | uint32 | fabric/stage3/run_vec_kv.py:171 |

**Key facts**

- The reference ladder is a strict refinement chain, each level runnable and self-gating: full (float) -> kv (O(T) incremental, maxabsdiff=0) -> q (pins signed Q6.25 residual/INT8/INT32/24-bit dequant) -> fixed (fabric-precision non-linears, cosine>0.9999) -> seq (integrated) -> seq_ref (per-phase integer) (model/goformer_full.py:1, goformer_kv.py:1, goformer_q.py:1, goformer_fixed.py:1, goformer_seq.py:1, fabric/stage3/seq_ref.py:1).
- Two chained gates bind hardware to the float model: float goformer_seq -(identical token stream)-> seq_ref integer, and seq_ref -(bit-exact mismatches=0)-> the RTL FSM (fabric/stage3/seq_ref.py:11-16).
- The BINDING gate is always the token stream, NOT raw-logit cosine: INT8 requant half-boundaries flip sub-1e-7 detail so cosine is a brittle proxy; even float goformer_seq holds only ~0.99995 worst-step (fabric/stage3/seq_ref.py:416-419).
- seq_ref.IntSequencer pins ALL the glue arithmetic no single block gate covers, and imports each block's integer reference directly (run_layernorm._ln_int_quantized, run_softmax.int_softmax_q, run_gelu.gelu_q) so there is one source of truth per op (fabric/stage3/seq_ref.py:50-53).
- full_forward_signals() exposes x4_q25/lnf_q22/head_q25/tok and block0_phase_signals() exposes ln1/qkv/ctx/attn/x_res/ln2/gelu/mlp/x_out so an RTL rewrite is localised phase-by-phase, not only on the emitted token (fabric/stage3/seq_ref.py:327-361).
- The gate-harness pattern: write .mem ROMs -> iverilog -g2012 -> vvp (require TB_DONE) -> per-phase .out compare -> one sentinel verdict line (*_VERDICT / SEQ_VEC_* / SEQ_SB_*) (fabric/stage3/run_vec_seq.py:52-103, run_sb_seq.py:60-123).
- The live engine gate run_sb_seq.py compiles sequencer_sb with two N=8 cohorts on the TDP URAM (weight_bank_tdp/embed_bank_tdp/cohort_engine/nl_engine), parameterized by ATT2 (per-cohort vs shared attn), TMAX (KV window), NDVAL (DSP-pack), DPUMP (double-pump) (fabric/stage3/run_sb_seq.py:37-79).
- cycle_model.py predicts single-stream cyc/token as the SUM of stage latencies (autoregressive, no overlap); serial non-linears (8x770 LN + 16 softmax) dominate and cap the sequencer ~10-15k regardless of GEMV width (fabric/stage3/cycle_model.py:47-91).
- batched_model.py predicts aggregate SERVING throughput = Fmax / busiest-shared-unit cycles; batching B~4 lifts single-stream ~10k to ~30k, and splitting the GEMV across 3-4 DSP engines reaches ~100k aggregate at T=128 INT8 KV (fabric/stage3/batched_model.py:64-81).
- The MEASURED record is 59,965.5 tok/s @200MHz: split-brain N=16 TMAX=16 wave, 53,364 cyc/16 tok, 16/16 bit-exact, 3/3; 250 MHz hangs (fabric/progress.py:68-69).
- The perf-model constants are fixed at N_LAYER=4, N_HEAD=4, D=256, D_MLP=1024, VOCAB=193 (fabric/stage3/cycle_model.py:24); the per-token GEMVs are qkv(768,256), proj(256,256), mlp_fc(1024,256), mlp_proj(256,1024) (cycle_model.py:27).
- OOC-synth Fmax on xck26-2LV (0 DSP, ~1.5MB): PE=256->293MHz, 512->292, 1024->239 (the LUT wall for pure-LUT INT4 MACs); silicon overclocks past STA (e.g. STA 71->clean 125MHz) (fabric/stage3/cycle_model.py:76-78, fabric/progress.py:38).
- The doc-7 faithful rungs are honestly separated from the T=1 aggregate rungs: N=1 T=128 on-chip KV window = 11,343.2 tok/s (real 119-tok message, 12,594 cyc/tok avg, 3/3), rising to 16,087.5 @166.7MHz; 20k is PROJECTED needing MMCM ~187MHz (fabric/progress.py:70-81).
- run_vec_kv.py gates the doc-7 stream against IntKVQSequencer(kbits=8, vbits=8, rotate=False, divfree=True) — the INT8-KV reference — and also gates on-chip Gumbel-max sampling via a shared gumbel_lut.mem source of truth (fabric/stage3/run_vec_kv.py:77, 44-67, 107-109).

**Files**

- `fabric/stage3/seq_ref.py` — THE per-phase integer sequencer reference — the bit-true contract the RTL FSMs are gated against; pins all glue arithmetic and exposes every phase signal
- `model/goformer_full.py` — Ladder base: full integer forward with pluggable matmul; float ops (LN/GELU/softmax); load_params/.npz
- `model/goformer_kv.py` — Ladder: incremental KV-cached decode (O(T), maxabsdiff=0 vs full); build_random_params
- `model/goformer_q.py` — Ladder: pins the signed Q6.25 32-bit residual / INT8 / INT32 / 24-bit-dequant datapath format
- `model/goformer_fixed.py` — Ladder: fabric-precision non-linears (FixedConfig knobs measured to cosine>0.9999); _profile_gelu_range
- `model/goformer_seq.py` — Ladder: integrated KV+fixed sequencer, the golden token stream (GOFORMER_SEQ_PASS)
- `model/goformer_kvq.py` — doc-7 KV-quantised reference (K8/V8) used by run_vec_kv; internals not fully read
- `fabric/stage3/run_layernorm.py` — LN block gate + _ln_int_quantized/rsqrt_int (seq_ref imports it as the LN source of truth); pins Q6.25 in / Q.22 out
- `fabric/stage3/run_softmax.py` — softmax block gate + exp_table/int_softmax_q (seq_ref imports); pins Q8.8 score / Q1.20 prob
- `fabric/stage3/run_gelu.py` — GELU block gate + gelu_table/gelu_q (seq_ref imports); pins Q4.12 8192-LUT
- `fabric/stage3/run_banked.py` — GEMV core bit-exact gate (BANKED_VERDICT)
- `fabric/stage3/run_sequencer.py` — Tiered integration gate (block0 / full-forward / multitoken) + write_mems / write_mems_wideword ROM packers
- `fabric/stage3/run_vec_seq.py` — Single-token P-wide full-forward gate for sequencer_vec (SEQ_VEC_FULL)
- `fabric/stage3/run_sb_seq.py` — Split-brain N=16 two-cohort gate for the live sequencer_sb engine (SEQ_SB_FULL / SEQ_SB_TOKS/S)
- `fabric/stage3/run_vec_kv.py` — doc-7 faithful multi-token KV-window + Gumbel-sampling gate (VEC_KV_VERDICT)
- `fabric/stage3/cycle_model.py` — Single-stream cyc/token perf model (SUM of stages); shows serial non-linears cap ~10-15k
- `fabric/stage3/batched_model.py` — Aggregate serving perf model (Fmax/busiest unit); prices the three levers to 100k
- `fabric/progress.py` — The authoritative MEASURED tok/s milestone ladder (0.07 -> 59,965.5) + doc-7 faithful rungs + plot

**Gotchas**

- The token stream is the ONLY binding gate; raw-logit cosine (~0.9994 for GELU, ~0.99995 for the full seq) is a documented BRITTLE proxy on this char model because INT8 requant half-boundaries flip on sub-1e-7 detail. Do not treat a cosine dip as a failure if the stream is identical (seq_ref.py:416-419, run_gelu.py:108-113).
- seq_ref does NOT re-implement LN/softmax/GELU — it imports run_layernorm._ln_int_quantized, run_softmax.int_softmax_q/exp_table, run_gelu.gelu_q/gelu_table so there is exactly one integer source of truth per op. Changing a block's format means editing the run_*.py, not seq_ref (seq_ref.py:50-53).
- cycle_model and batched_model are pre-split-brain projections built from the OLD serial-non-linear RTL and PE-lane GEMV; they do NOT model the split-brain TDP-cohort architecture that actually delivered the 59,965.5 measured record. Use fabric/progress.py for real numbers, the models only for lever intuition (cycle_model.py:76-78, batched_model.py:16-18).
- The rungs 1-23 in progress.py decode with attention T=1 (aggregate throughput, degenerate text); the FAITHFUL rungs (N=1, T=128 window) are a DIFFERENT metric (real messages). Never compare a T=1 aggregate number to a faithful number as if the same axis (progress.py:70-73).
- STA/OOC Fmax on -2LV is pessimistic — silicon overclocks ~1.3-1.76x (STA 71 -> clean 125; STA 79.5 -> breaks at 142.9). Every rung's 'breaks/fails at X MHz' is the measured board ceiling, not the STA number (progress.py:38-51).
- write_mems_wideword requires LANES>=8*P (EPW>=1) to append the embed tables into wrom.mem; smaller LANES are silently skipped and the embed-in-URAM fit-plan is not exercised (run_sequencer.py:268-269, per the code comment).
- run_vec_kv's docstring says 'K4/V4+Hadamard' but the LIVE call is kbits=8, vbits=8, rotate=False, divfree=True — the INT8-KV reference. Trust the code call, not the stale docstring (run_vec_kv.py:1-10 vs 77).
- seq_ref pos handling: full_forward_signals() resets KV/t and decodes a single token at pos 0; step()/generate_greedy advance self.t. Mixing them without reset() gives wrong pos_emb (seq_ref.py:344-350, 145-148).

**Open questions**

- model/goformer_kvq.py (23KB) internals were not read in this pass — the exact K8/V8 quantisation, rotate/divfree semantics, and how IntKVQSequencer differs from seq_ref.IntSequencer's attention path are undocumented here. Read it directly before modifying the doc-7 KV gate.
- The RTL modules themselves (sequencer_sb.sv, cohort_engine.sv, nl_engine.sv, vec_attn.sv, layernorm_vec.sv, etc.) were not read — this section documents the Python reference/gate/model side only. The claim 'bit-identical to rtl/sequencer.rsh_round' etc. is asserted by the reference's comments, not verified against the RTL here.
- fabric/stage3/gumbel.py (make_gumbel_lut / GumbelRng) was not read; the exact on-chip Gumbel-max sampling arithmetic is only known through run_vec_kv's usage.
- The precise cyc/token and Fmax attached to each progress.py rung are transcribed from the LADDER string literals (self-reported MEASURED values); they were not independently re-derived from the harness cyc.out outputs in this pass.
- pack_banked.write_case/check (the GEMV stimulus + bit-exact checker behind run_banked) were not read in detail.
- The doc-6/doc-7 design docs (6-past-the-stream-ceiling.md, 7-kevin-remembers.md) and WIDE-WORD-DATAPATH-LOG.md §19-36 referenced throughout the code comments were not read; they hold the authoritative lever-alive/dead status and the schedule-overlap history.

---

## The speed campaign: two ceilings, the 100k identity, the rung ladders, and the dead-lever ledger

This subsystem is the strategic layer of Kevin-on-Kria: not RTL, but the argument that decides which RTL is worth building. The model has long run entirely in the KV260 PL fabric (CPU out of the loop, weights resident in URAM/BRAM, zero-DRAM inference). Once that was true, "faster" stopped meaning "get it on-chip" and split into two distinct campaigns with two distinct headline numbers. Campaign one (doc 6, `6-past-the-stream-ceiling.md`) is the N=16 aggregate-throughput engine — sixteen concurrent streams sharing one weight-read pass — whose MEASURED record is 59,965.5 tok/s @200 MHz (`docs/6-past-the-stream-ceiling.md:160`, `fabric/progress.py:68`). Campaign two (doc 7, `7-kevin-remembers.md`) is the N=1 faithful-stream engine — one stream that actually remembers its context (full trained T up to 256, real messages instead of "he he he he") — whose deployed MEASURED number is ~16,087.5–16,227 tok/s @166.7 MHz (`docs/7-kevin-remembers.md`, `README.md:81`, log §43/§45).

The tension between the two is the load-bearing tradeoff: every N>=4 design in campaign one hardwires attention T=1 (`at_tcount_o <= 9'd1`), so it produces blistering aggregate tok/s but degenerate text (`docs/7-kevin-remembers.md:19-25`). Campaign two spends 15/16 of the freed MAC fabric on width and a real KV window, sells "messages" not "tok/s," and lands at ~1/3 the aggregate record. Both build from the same tree; the daemon just loads a different bitstream.

Two ceilings, distinct from the two campaigns, govern what a user actually feels: the FABRIC ceiling (pure PL cycles — the 59,965.5 / 16,227 numbers) versus the ROUND-TRIP ceiling (what a chat user sees, bound by serving not silicon). The biggest round-trip tax was host-side sampling (193 head-logit reads/token over /dev/mem, ~58% of a reply); moving it on-chip via Gumbel-max lifted the localhost round-trip ~5.6x (1k -> 5,600 tok/s) while keeping the fabric record bit-exact (`README.md:75-93`, log §44-45). (Whether the deployed daemon currently invokes this on-chip path or the host-side one is UNRESOLVED — see the "live sampling" note.)

The strategic conclusion is honest and settled: the "100k tok/s = 16 streams x 250 MHz / 40,000 cyc" identity does NOT close on this chip at this model size. Streams are maxed at 16 (2.0 MAC/DSP packing wall), clock is 200 MHz MEASURED with 250 unlocked only in STA but route-blocked at 98-99% density, and the cycle floor is MEASURED at ~51,100 — not 40,000. So the architecture's real ceiling on the KV260 is ~62-78k, and 100k needs either a bigger part (LANES=256, 2,048 DSPs) or a smaller/dumber model. Model-shrink to hit 100k was explicitly VETOED by the user (`fabric/stage3/research/100K-REVIEW.md:115-128`), so 100k is CLOSED on this board.

## 1. The two campaigns and their headline numbers

| Campaign | Engine | What it sells | Headline | Attention | Tag |
|---|---|---|---|---|---|
| Doc 6 — aggregate throughput | `sequencer_sb` (split-brain, two N=8 cohorts) | tok/s | **59,965.5 tok/s @200 MHz** | T=1 (degenerate text) | MEASURED, 16/16, 3/3 (`fabric/progress.py:68`, `docs/6-past-the-stream-ceiling.md:160`) |
| Doc 7 — faithful stream | `sequencer_vec` + `kv_bank` (N=1, on-chip KV) | messages | **~16,087.5–16,227 tok/s @166.7 MHz** deployed | T up to 256 (real text) | MEASURED, 3/3 bit-exact (`README.md:81`, log §43/§45) |

These are **different metrics on different axes**, both honestly labelled. The N>=4 aggregate designs are hardwired to attention T=1: `nl_engine.sv` drives `at_tcount_o <= 9'd1`, `sequencer_vec.sv` likewise; the silicon confirmed it (`test_sb_kvwin.py`, MEASURED 2026-06-10, drove the record bitstream position-by-position, got `match=False` and a constant 53,364 cyc/pass at every position — no KV survives a GO) (`docs/7-kevin-remembers.md:19-25`). The only faithful decoders before doc 7 were the early single-stream sequencers (`sequencer.sv`/`sequencer_fast.sv`, `sm_tcount <= tpos`) topping out at **751.8 tok/s MEASURED** (`docs/7-kevin-remembers.md:27-31`). Doc 7 delivers BOTH speed and faithfulness at N=1, a 15.1x jump on the axis that actually spells text (log §39, line 1291-1292).

## 2. The two CEILINGS (distinct from the two campaigns)

`README.md:75-93` names them explicitly:
- **Fabric ceiling** — pure PL cycles. 59,965.5 (N=16 aggregate) or 16,227 fabric tok/s (N=1 faithful deployed build).
- **Round-trip ceiling** — what a live chat user feels, bound by serving not silicon.

The dominant round-trip tax was **host-side sampling**: reading 193 head logits per token back over `/dev/mem` for A53 temperature sampling — ~58% of a ~100 ms reply (`README.md:85`, log §44 lines 1398-1400). The fix (log §44-45): the **Gumbel-max trick** — sampling from `softmax(logit/T)` is exactly `argmax_i(logit_i + T*g_i)`, `g_i ~ Gumbel(0,1)` — reuses the existing S_ARGMAX hardware. The host writes ONE seed register (0x30 in `gemv_axi_seq_vec.v`) per request instead of 193 reads/token; `seed=0` forces gumbel=0 => pure argmax, bit-exact to greedy. `gumbel_lut[idx]=round(0.85*(-log(-log((idx+0.5)/2^GLBITS)))*2^25)`, T=0.85 baked; RNG is a seedable xorshift32 persisting across a request's GOs (log §44 lines 1407-1416).

SHIPPED result (`gemv_seqkv_gum.bit.bin`, MEASURED 2026-06-12, log §45): built at a 125 MHz target (the 200-target trims build was unroutable — see §42 below), routed clean WNS +0.013 ns, util LUT 93.4% / BRAM 94.4% / URAM 100% / DSP 97.4%. Greedy no-regression 3/3 bit-identical, 9292 cyc/tok = 17,936.6 tok/s; on-chip sampling 6/6 unique coherent Kevin-speak. Round-trip: localhost 17.8 ms median -> ~5,600 trip tok/s (fabric 7.21 ms / 117 passes = 16,227 fabric tok/s, the 16k RECORD PRESERVED); full public path (dev box -> Cloudflare -> Precision -> Kria -> back) 60.9 ms median -> ~1,658 public tok/s (the +43 ms is WAN/tunnel RTT, not fabric) (log §45 lines 1438-1445).

## 3. The stream ceiling — why N stops at 16

Batching wins because a single resident-weight GEMM pass reads the whole ~12.6 Mbit INT4 weight image out of URAM exactly once; N streams amortise that one read across N tokens (`docs/6-past-the-stream-ceiling.md:16-21`). The ladder: N=4 (16,969.3 @166.7) -> N=8 ping-pong (17,740.6) -> N=8 single-pass (19,275.6) -> N=16 (24,134.0), all bit-exact on silicon (`docs/6:20-21`, log §14-17).

**N=16 is a proven hard ceiling at LANES=128.** The packing is exactly **2.0 INT4xINT8 MACs per DSP**; 3.0/DSP is impossible on two independent walls, proven exhaustively in `research/dsp3_pack_proof.py`:
- **Operand-port wall:** three no-bleed 4-bit nibbles at 12-bit gaps need `2*12+4 = 28 bits` against the DSP48E2's **27-bit** signed operand port (`dsp3_pack_proof.py:28-30`).
- **Accumulator wall:** three distinct K=1024 neurons hold `3*22 = 66 bits` of accumulator state against the **48-bit** accumulator — "INFORMATION-THEORETICALLY IMPOSSIBLE" (`dsp3_pack_proof.py:26-27`).
- The proof: 0 mismatches on the proven 2.0 scheme over 1.2M randomized K=1024 lane-products + exhaustive K=1 + adversarial corners; the 3.0 and 5-per-2 variants die the same way. Maximum exact rate is **2.0/DSP** (`dsp3_pack_proof.py:29-30`, `docs/6:24-29`).
- **LANES=256 is also dead** — it needs 2,048 DSPs; the part has 1,248 (`docs/6:29`, `100K-REVIEW.md:43`).

So past N=16 (pushed to 25,744.5 @166.7 by a softmax-latency cut, §18), **more streams was over**. The only remaining levers are CYCLES and CLOCK (`docs/6:29-31`).

## 4. Split-brain — the one big cycle play (the live engine)

The single weight-read port serialised everything. But the KV260's URAM is genuinely dual-port; an `xpm_memory_tdpram` in "ultra" mode maps both ports independently — **56 URAM, 0 LUT, 0 BRAM, proven** (commit `2f3ba17`, `docs/6:36-37`). The structural move (`fabric/stage3/research/SPLIT-BRAIN.md`): run **two independent N=8 cohorts**, each reading the same resident weight image through its own port, sharing only the weight image + arbitrated LayerNorm/attention/embed/dequant. A cohort never shares a weight pass, so the stream-desync problem that haunted the single-pass merge **dissolves instead of being managed** (`docs/6:38-41`, `SPLIT-BRAIN.md:15-31`).

Critical enabling finding (`SPLIT-BRAIN.md:57-68`): **HDL inference of TDP UltraRAM is DEAD in Vivado 2025.2** — two-address ports map to BRAM (400 tiles!); symmetric UG901 and NO_CHANGE templates both refuse with `[Synth 8-12186] ram_style="ultra" ignored: invalid write mode`. The supported path is `xpm_memory_tdpram, MEMORY_PRIMITIVE="ultra"` at bank geometry (72b x 25,600 x 8 banks) -> URAM 56 / 0 LUT / 0 BRAM, both ports independent. Loader keeps port A (writes at boot, cohort-1 reads at runtime via an address mux); cohort-0 stays on port B.

Payoff: N=14 split-brain (NC=7, the one that fit after the LUT campaign) = **36,970.7 tok/s @166.7** (+43.6% record, log §20). N=16 (NC=8) fit later once the AQ-multiplier was range-narrowed and attention un-evicted (§21).

## 5. `sequencer_sb` parameters (the live design)

From `fabric/stage3/rtl/sequencer_sb.sv:20-44`:

| Param | Default | Meaning |
|---|---|---|
| `D` | 256 | model dim |
| `D3` | 768 | qkv width (3*256) |
| `D_MLP` | 1024 | MLP hidden (ff) |
| `P` | 8 | P-wide non-linear/readback lanes |
| `LANES` | 128 | GEMV MAC lanes |
| `N` | 16 | total streams (two cohorts of NC) |
| `NC` | 8 | streams per cohort |
| `ND` | 0 | DSP-packed GEMM streams per cohort |
| `TMAX` | 32 | on-chip KV attention window (the record bitstream builds with **TMAX=16**) |
| `NLAYER` | 4 | transformer layers |
| `NHEAD` | 4, `HEAD_DIM`=64 | attention shape |
| `DBG` | 1 | 0 = tie off board-debug readback for **bitstream builds** |
| `ATT2` | 1 | **1 = vec_attn per cohort; 0 = shared+arbiter (fits today)** |
| `DP` | 0 | DOUBLE-PUMP Stage 1: MAC at 2 K-steps/clk (the dead lever, see §9) |

`ATT2` generate block: `sequencer_sb.sv:167-202` — `ATT2=1` instantiates `u_attn0`/`u_attn1` (one `vec_attn` per cohort, the §24/§26 un-share that kills the critical cohort's ~5.8k serial queue, needs TMAX=16 BRAM); `ATT2=0` uses the proven single shared `u_attn` + arbiter. The **shipping N=16 record bitstream sets `ATT2=0`** because per-cohort attention is ~1.9k LUT over budget on the 98%-dense device (`sequencer_sb.sv:154-166`, log §27 lines 1003-1004).

## 6. The doc-6 lever campaign — MEASURED silicon records

From `docs/6-past-the-stream-ceiling.md:54-60` plus log §27:

| tok/s | clock | cyc (silicon) | config | what cleared it | source |
|---|---|---|---|---|---|
| 24,134.0 | 166.7 | 110,494 / 16 tok | N=16 merged | N=16 fit campaign | §17 |
| 25,744.5 | 166.7 | 103,582 / 16 tok | N=16 | softmax-latency cut | §18 |
| 36,970.7 | 166.7 | 63,113 / 14 tok | N=14 split-brain | dual-port URAM cohorts | §20 |
| 46,604.4 | 200 | 68,663 / 16 tok | N=16 split-brain | first 200-clean build | §21 |
| 56,262.7 | 200 | 56,876 / 16 tok | N=16 split-brain | schedule-pipelining wave | §24 |
| **59,965.5** | **200** | **53,364 / 16 tok** | **N=16 split-brain, TMAX=16** | **architectural wave (§26)** | **§27, `progress.py:68`** |

The 46,604.4 build (§21) was the first to close 200 MHz clean via three composed timing levers: un-retiming the LayerNorm sum-of-squares path so its barrel-shift couldn't fuse with the Newton squarer (had been failing WNS -1.876); a 32x48 range proof on the activation-quant multiplier (`research/aq_range_proof.py`) that freed enough DSP to **un-evict attention** from fabric back into DSPs; and an attention operand-register split (`docs/6:62-67`). The 56,262.7 build (§24) was the schedule-pipelining wave: AQ/RUN overlap + stream-granular NL/GEMM readback overlap + per-call `vec_attn` cuts, composed.

The **59,965.5** record (§27, log lines 983-992): the §26 wave (TMAX 32->16, per-cohort attention OFF for fit / shared+arbiter ON via ATT2=0, CTX cross-group stream) + LN prod*gamma split, built at 6.0 ns after a 5.5 ns route diverged (overlaps 295k->339k at 98% BD density, killed at 2h). 6.0 ns routed clean WNS -0.702, impl 1h43m. Silicon: 53,364 cyc (sim 53,637 - 273 SETTLE, the sixth build to hit the signature), 16/16 bit-exact, 3/3 — 49,971.3 @166.7 / 59,965.5 @200. **250 -> TIMEOUT (hangs, not corrupt; the 6.0 ns target gives ~1.6x silicon margin, 250 needs 2.0x).**

The sim cycle campaign ran ahead of silicon (`docs/6:71-76`): 71,441 -> 66,285 (AQ/RUN overlap) -> 61,245 (stream-granular NL/GEMM readback overlap, §22) -> 57,149 (`vec_attn` per-call cuts, §23) -> 53,565 (CTX cross-group streaming, §25) -> 51,892 (TMAX 32->16 + per-cohort attention un-share + CTX cross-group composed, §26).

## 7. The 100k identity and the floor that caps it

`docs/6:124-155` states the target factors cleanly: **100,000 tok/s = 16 streams x 250 MHz / 40,000 cyc.** Three knobs, all now walled:

- **Streams: maxed at 16, proven** — the 2.0/DSP packing wall (§3 above).
- **Clock: 200 MHz MEASURED; 250 MHz unlocked in STA but route-blocked.** The campaign retired worst paths one at a time — LayerNorm (§21 + a second `prod*gamma` split), attention score/context multiplies, softmax recip/divider, and the embed->residual URAM cone — until a 5 ns OOC corner closes MET (+0.074, zero failing). With silicon's ~1.3x margin that is a genuine 250 MHz / ~75-78k shot, BUT **the route fails at 98-99% LUT density** (~2.9 ns pure routing on the worst path, 4 builds tried) (`docs/6:131-135`, `100K-REVIEW.md:44`).
- **Cycles: the floor is ~51,100, NOT 40,000 (the load-bearing MEASURED finding).** The GE engine is a strictly serial FSM (RUN -> readback -> idle, one state per cycle). The per-state profile reconciles exactly to the gated 53,637: **RUN = 25,088 cyc** (the irreducible MAC at LANES=128, DERIVED from GEMM shapes), **readback = 10,360** (irreducible at P=8; widening needs a P=16 dequant that busts the LUT budget — proven dead), and the rest is the LN->attention->residual serial chain the lockstep GEMM consume won't let fully hide. Drive GE_IDLE to its physical minimum and the floor is **~51,100 cyc silicon** (`docs/6:136-142`).

**The honest arithmetic** (`docs/6:144-147`): at the ~51,100-cyc floor, 200 MHz gives ~62.6k and 250 MHz gives ~78.3k. 40,000 cyc (and therefore 100k) is **not reachable at LANES=128/P=8** — off by ~11,000 cycles even if every idle were hidden, because the serial RUN+readback floor alone is ~37,000 and the un-hideable non-linear chain sits on top.

**What 100k actually needs is the model, not the schedule** (`docs/6:149-155`). The cycle floor is MAC-bound on the GEMM dimensions, so it scales with model size. LANES=256 would halve RUN but needs 2,048 DSPs. The other halving is a smaller model — the thesis closing on itself ("the last lever to 100k on this chip is not a cleverer datapath, it is a dumber Kevin").

## 8. The 100K-REVIEW verdict — model shrink VETOED, 100k CLOSED

`fabric/stage3/research/100K-REVIEW.md` (2026-06-09) verified the cycle law **bit-exactly** against silicon (§1): RUN decomposes as a sum over GEMV calls of `rows x ceil(K/LANES)` at LANES=128 -> `4*6,144 + head 256r*2b = 25,088` ✓; readback `RB = rows*35/32 = 9,472*35/32 = 10,360` ✓; remaining 17,916 is serial NL + FSM idle. So the full model reproduces the silicon record to the cycle (`100K-REVIEW.md:16-31`).

Key constraint the review adds (`100K-REVIEW.md:32-37`): **the K dimension is quantized to 128-lane beats** — `ceil(192/128)=ceil(256/128)=2`, so a d=192 model pays the same MAC beats/row as d=256 (25% lanes idle). **d and d_ff must stay multiples of 128** or the shrink is partially wasted. This killed the intuitive "L4 d192 -> 118.8k" projection (corrected to ~81-87k).

The candidate table for a smaller model (`100K-REVIEW.md:65-73`, all N=16/LANES=128/P=8, PROJECTED): current L4 d256 ff1024 (3.28M) = 59,966 MEASURED; **C: L3 d256 ff512 (1.70M) = ~98,231**; **D: L2 d256 ff1024 (1.70M) = ~113,665** — both keeping d=256 so every d-width datapath is bit-untouched (the RTL delta is only loop bounds + ROM sizes).

**Disposition (`100K-REVIEW.md:115-128`): model shrink VETOED same day** — user decision: "I don't want to make the chat/model any worse than it already is." With every datapath/clock/parallelism lever measured to a wall, the honest ceiling is: **59,965.5 @200 MHz MEASURED (the record/deliverable)**; ~62-64k @200 if residual gated serial cuts are ever routed (marginal, not pursued); ~78k @250 only if route-congestion ever yields (4 builds say it won't). **"100,000 tok/s is not reachable on the KV260 at this model size. Do not reopen without a bigger part or an explicit user reversal on model size."**

## 9. THE DEAD-LEVER LEDGER — do not re-propose these

Consolidated from `100K-REVIEW.md:39-53`, `docs/6:78-87`, `docs/7:152-157`:

| Lever | Verdict | Evidence |
|---|---|---|
| **Streams > 16** | DEAD — 2.0 MAC/DSP packing wall, 1,187/1,248 DSPs used | doc 6 §18, `dsp3_pack_proof.py` |
| **LANES=256** | DEAD — needs 2,048 DSPs, part has 1,248 | doc 6 §18 |
| **3 INT4xINT8 MACs/DSP (3.0 packing)** | DEAD — 28b operand > 27b port AND 66b acc > 48b, 0-mismatch proof of the 2.0 scheme, 3.0 dies | `dsp3_pack_proof.py:26-30` |
| **250 MHz (current design)** | DEAD — STA closes (5 ns MET +0.074) but route fails at 98-99% LUT density (~2.9 ns pure routing), 4 builds | doc 6 §14-31, `100K-REVIEW.md:44` |
| **Double-pump MAC @clk2x** | DEAD — the MAC-only branch (`dp-hw-maconly`) was bit-exact on silicon but the fabric->clk2x data feed walls at 50 MHz (the full DP=1 build was OOC-MET only, never board-run); full N=14 integration violates -3.3 ns at 99% CLB density; `tok/s=14*clk/41,753`, 80k needs 238.6 MHz > 200 wall; routed WNS -4.658 on clk2x | `100K-REVIEW.md:45`, `DOUBLE-PUMP-100K.md`, branch `dp-hw-maconly`. **NOTE the tension**: `DOUBLE-PUMP-100K.md` (2026-06-08) is an aspirational "~88k GO" plan; the next-day `100K-REVIEW.md` (2026-06-09) explicitly invalidated its stack ("assumed DP closes 400 MHz, which silicon (50 MHz data-feed) and routed timing (WNS -4.658) both refute"). Doc 7 §5 lists it dead. The authoritative verdict is DEAD. |
| **P=16 readback/dequant** | DEAD — LUT budget bust (127.6k LUT at P=8/L=128 already) | doc 6 §139, WIDE-WORD log §4 |
| **LN->AQ schedule overlap** | DEAD — lockstep GEMM consume gates RUN-start on the slowest stream's LayerNorm; no per-stream wavefront to exploit; two agent HONEST-STOPs | doc 6 §78-87 |
| **attention->PROJ schedule overlap** | DEAD — same shape as LN->AQ | doc 6 §84 |
| **Shared-attention arbiter-fairness fixes** | DEAD — measured **exactly Delta 0**; the shared-`vec_attn` cost is pure serial unit throughput, not a priority artifact; re-prioritising buys nothing | doc 6 §23-24 lines 82-87 |

**Schedule overlap is exhausted at this topology** — "the road on from there is architectural, not a scheduler tweak" (`docs/6:86-87`). The shared-attention serial wall was only broken structurally (per-cohort `vec_attn`, ATT2=1), not by arbitration.

## 10. The silicon margin and the SETTLE signature

STA on `-2LV` is pessimistic; the whole campaign leans on that gap honestly. Designs that close ~70-85 MHz in STA have run bit-exact at 125-200 MHz on silicon — observed factor **~1.3-1.76x** across builds (`docs/6:90-93`). Policy: build at a clock that closes in STA, find the real ceiling with the board `--fclk` sweep, withhold the tok/s claim unless the token matches `seq_ref` 3/3. The PS PLL snaps to 1000/N MHz (...125, 142.9, 166.7, 200, 250...), so a build clears a rung or it doesn't — no fractional headroom (`docs/6:94-96`, log §43 lines 1379-1381: 175->166.7, 183.3->200, both verified).

**The SETTLE signature** (`docs/6:98-102`): silicon runs a few hundred cycles fewer than sim (-297 or -273, depending on call count) because a sim-only settle state (y_lat latch) is skipped on real fabric; synth goes RUN->DRAIN directly (log §16 lines 552-562). The gap held for five/six consecutive builds, predicted ahead of each — "the kind of small thing that, when it keeps coming out exactly right, means the model of the silicon is honest."

## 11. The context-for-cycles trade (TMAX 32->16)

The 53,565 -> 51,892 cut (§26) was NOT free (`docs/6:104-111`): TMAX dropped 32->16 (on-chip attention window halved to 16 positions), which freed the BRAM tiles that funded per-cohort attention (deleting the shared-`vec_attn` arbiter, the §23/§24-named serial wall). Documented, not silent: `pl_seq_sb` gained a `--tmax` flag; **the embed upload MUST match the build TMAX or positions past 0 corrupt** (`sequencer_sb.sv` `.TMAX(TMAX)` now parameterised, log §26 lines 965-970, §27 lines 996-998).

The restore path exists and is gated: the KV-DDR stack (`kv_dma` + `kv_prefetch`, sim-complete, bit-exact, P=8 wide-emit) double-buffers a prefetch that hides DDR latency behind the per-head compute window. Bandwidth math (DERIVED, `research/KV-DDR-100K.md`): full-window 100k-aggregate KV read ~6.4 GB/s; TMAX=16 Kevin window halves to ~3.2 GB/s; K4/V4 + Hadamard KV quant (MEASURED +0.72% NLL for 1.78x compression in `model/exp_kvarn.py`) shrinks it further — all under the ~6-7.5 GB/s sustained HP-port ceiling, so DDR is NOT the binding wall at 100k for the short Kevin context (`docs/6:113-122`).

## 12. Doc 7 — the faithful-stream campaign and the R0-R5 rung ladder

**Target:** 20,000 tok/s AVERAGE on N=1 at T=256 (the model's entire trained context), model-faithful decode — a real message that remembers the previous exchange (`docs/7:3-15`). The identity: `20,000 tok/s avg = <=10,000 cyc/tok avg @200 MHz = <=12,500 @250` (`docs/7:35-38`), "average" over a full-window generation (window 1->256, mean T-bar ~128).

The cost model (`docs/7:48-58`, DERIVED): per token 3,195,136 MACs; GEMV LANES=256 = 12,481 cyc (lever: LANES=512 via second port -> ~6,241); attention P=8 = 32,896 avg/65,536 worst (lever: P=128 -> 2,056/4,096); NL/control ~5,466 -> ~2,500. KV cache at T=256/N=1 INT8 = 512 KB.

**The rungs as designed (`docs/7:67-121`):**
- **R0 — reference. DONE.** `goformer_kvq.IntKVQSequencer(kbits=8, vbits=8, rotate=False)`. Pinned contract: **K8/V8, NO Hadamard** — NLL delta +0.03% (24x128=3,072 held-out tok). FP-identity gate (kbits=vbits=16) bit-identical to IntSequencer. Dropping rotation deletes the 6-stage butterfly for 0.03 NLL points.
- **R1 — make it remember. GREEN IN SIM (commit fb9940e).** `kv_bank.sv` (K8/V8 quant-at-write, zero-bubble dequant-read) + `sequencer_vec` S_KVW states + `tcount=pos+1`. Gate `run_vec_kv.py`: 10 positions, KV persisting across GO pulses, token stream bit-identical to R0 golden.
- **R2 — wide attention.** `vec_attn` P=8->128.
- **R3 — dual-port GEMV.** At N=1 the second URAM port (split-brain's cohort-2 port) is free -> LANES=512, GEMV floor halves. Explicitly **NOT the dead clock-domain double-pump** — no second clock (`docs/7:104-106`).
- **R4 — P-wide non-linears + widened softmax walk.**
- **R5 — the clock.** 250 MHz on a design a fraction of the SB's density (N=1 removes the routing-density blocker). Contingency if 250 won't close: 214.3 MHz (next PL divisor) or the realistic-traffic number.

**How the rungs actually landed (sim-MEASURED, `docs/7:122-137`, log §29-35), every rung bit-exact vs the pinned golden:**

| rung | cycle law | avg tok/s @200 (T-bar=80) | @250 | log |
|---|---|---|---|---|
| R1 KV banks | 19,842 + 528(T-1) | 3,235 | - | §28 |
| R2 wide attention | 19,666 + 128(T-1) | 6,706 | - | §29 |
| R3 dual-port GEMV | 13,394 + 128(T-1) | 8,485 | 10,606 | §30 |
| R4a divide-free quantiser | 11,730 + 128(T-1) | 9,131 | 11,414 | §31 |
| R4c softmax_f + V-overlap | 11,714 + 48(T-1) | 12,870 | 16,088 | §32 |
| R4e twin engines | 11,442 + 24(T-1) | 14,981 | 18,727 | §33 |
| R4f feeder + trims | 11,138 + 22.7(T-1) | 15,454 | 19,318 | §34-35 |

The slope ladder (attention cyc/position) is the story: **528 (R1) -> 128 (R2/R3) -> 48 (R4c) -> 24 (R4e) -> 22.7 (R4f)**. A full 127-token generation averages 12,499 cyc = **16,001 tok/s @200 / 20,001 @250 (DERIVED)**; **full-window correctness 248/248 tokens bit-exact to T=255** (`docs/7:136-137`, log §34-35).

**Silicon reality for doc 7 (log §39-45):** the R5 build hit 13,162.3 tok/s @166.7 (200 fails on the GEMV MAC accumulate floor, accb carry chain ~8.75 ns, §41); the schedule-trims build initially failed route at 200 (182k node overlaps, §42) but the looser 125 MHz target routed and hit **16,087.5 tok/s @166.7** (§43), then the on-chip-sampling gumbel build shipped at 17,936.6 greedy tok/s / 16,227 fabric tok/s (§45). **20k @250 is SIM-proven but NOT reached on silicon** — the PS PLL offers only 166.7/200 and 200 fails timing on this RTL, so 20k needs either an MMCM ~187 MHz PL clock (-> ~18.1k) or the K4/smaller-model cycle cut (log §43 lines 1386-1394).

**The shipping pivot (log §37, `docs/7:139-150`):** the K4 + Hadamard cache diet FIT the BRAM budget (129/144) but cost ~66k LUT of butterfly adders (181%) AND showed an un-debugged long-T divergence, so the SHIPPING build is **K8 no-rotate (+0.09% contract, clean to T=255) at TMAX=128** — OOC LUT 85.4% / BRAM 92.4% / URAM 64/64 / DSP 98.6%, WNS -1.99 @5ns. T=128 still holds a turn + reply. The T=256 rung (a shared constant-geometry Hadamard unit ~3k LUT + the K4 long-T debug) is queued.

## 13. Where it loses (said up front, `docs/7:171-185`)

- Aggregate throughput drops: N=1 ~20k avg is ~1/3 the 59,965.5 sixteen-stream record. Different product (messages vs tok/s), both buildable from the same tree.
- The bit-exact contract changes: INT8 KV means a new golden reference (R0); pre/post numbers not token-for-token comparable. The NLL gate keeps it honest.
- The 20k headline leans on R5 (250 MHz); at 200 MHz the average lands ~18.5k.
- T > 256 does not exist — that is the model's trained context, not a window choice. KV-DDR (doc 6) is the road for a future bigger-context model, not this one.

## Diagram: the strategic map

```
                 FABRIC CEILING (pure PL cycles)
                          |
   +----------------------+----------------------+
   |                                             |
 DOC 6: aggregate N=16                    DOC 7: faithful N=1
 sequencer_sb (split-brain)              sequencer_vec + kv_bank
 attention T=1 (degenerate)              attention T<=256 (real text)
 59,965.5 tok/s @200 MHz                 ~16,087.5-16,227 tok/s @166.7 (deployed)
 MEASURED, 16/16, 3/3                    20,001 @250 SIM-only
   |                                             |
 100k identity = 16 x 250MHz / 40k cyc     R0..R5 rung ladder
   streams maxed (2.0 MAC/DSP)              slope 528->22.7 cyc/pos
   clock 200 MEAS / 250 route-dead          K8/no-Hadamard shipping (TMAX=128)
   cycles floor ~51,100 (NOT 40k)
   => ceiling ~62-78k, 100k CLOSED
   => 100k needs bigger part OR dumber model (VETOED)
   |
 ROUND-TRIP CEILING (serving, not silicon)
   Gumbel-max on-chip sampling: 1k -> 5,600 localhost tok/s (5.6x)
   public path ~1,658 tok/s = WAN/tunnel RTT
```

**Parameters**

| name | meaning | live | range | where |
|---|---|---|---|---|
| `TMAX` | On-chip KV attention window (positions). Default 32; the 59,965.5 record bitstream builds TMAX=16 (context-for-cycles trade). Board --tmax must match the build or pos>0 corrupts. | 16 (record N=16 bitstream); 128 (deployed faithful chat) | 16 / 32 (doc-6); 128 shipping faithful (doc-7 §37); 256 = full trained context (queued) | fabric/stage3/rtl/sequencer_sb.sv:29 |
| `ATT2` | 1 = vec_attn per cohort (kills the shared-attention serial wall, needs TMAX=16 BRAM, ~1.9k LUT over budget); 0 = single shared unit + arbiter (fits today). | 0 in the shipping bitstream (fits); 1 in sim (the §26 51,892-cyc cut) | 0 or 1 | fabric/stage3/rtl/sequencer_sb.sv:43 |
| `N / NC` | Total streams / streams per cohort. Split-brain runs two cohorts of NC on the two URAM ports. | N=16, NC=8 | N=16 NC=8 (record); N=14 NC=7 (first split-brain fit) | fabric/stage3/rtl/sequencer_sb.sv:25-26 |
| `LANES` | GEMV MAC lanes. LANES=128 is the aggregate design; LANES=256 is DEAD (needs 2,048 DSPs). Doc-7 faithful uses LANES=256, R3 dual-port -> effective LANES=512. | 128 | 128 (aggregate) / 256 (faithful) | fabric/stage3/rtl/sequencer_sb.sv:24 |
| `P` | P-wide non-linear/readback lanes. P=16 readback/dequant is DEAD (LUT budget bust). | 8 | 8 | fabric/stage3/rtl/sequencer_sb.sv:23 |
| `DP` | DOUBLE-PUMP Stage 1: MAC at 2 K-steps/clk (clk2x). The dead lever — the MAC-only branch (`dp-hw-maconly`) ran bit-exact on silicon but the fabric→clk2x data feed walls at 50 MHz; the full DP=1 build was OOC-MET only, never board-run. Declared DEAD by 100K-REVIEW. | 0 (never shipped) | 0 or 1 | fabric/stage3/rtl/sequencer_sb.sv:44 |
| `DBG` | 0 = tie off board-debug readback for bitstream builds (frees LUT/routing). | 0 in bitstream builds, 1 in sim | 0 or 1 | fabric/stage3/rtl/sequencer_sb.sv:42 |
| `kbits/vbits/rotate (R0 contract)` | The pinned faithful-stream KV-quant reference: K8/V8, NO Hadamard rotation. NLL delta +0.03% (pinned) / +0.09% with the divfree writer amendment. | K8/V8 no-rotate (shipping); K4/V4+Hadamard parked (butterfly LUT bomb + long-T bug) | kbits=8, vbits=8, rotate=False | model/goformer_kvq.py via docs/7:82-87 |

**Key facts**

- The N=16 aggregate record is 59,965.5 tok/s @200 MHz MEASURED (16/16 bit-exact, 3/3), 53,364 cyc/16 tok on silicon (docs/6-past-the-stream-ceiling.md:160, fabric/progress.py:68, WIDE-WORD-DATAPATH-LOG.md:989)
- The deployed faithful single-stream (doc-7) runs ~16,087.5 tok/s @166.7 (log §43) / 16,227 fabric tok/s with on-chip sampling (README.md:81, log §45); the previous faithful record was 751.8 (docs/7-kevin-remembers.md:29-31)
- Every N>=4 aggregate design hardwires attention T=1 (at_tcount_o <= 9'd1) => blistering aggregate but degenerate text; test_sb_kvwin.py MEASURED match=False at every position, 53,364 cyc/pass constant (docs/7-kevin-remembers.md:19-25)
- The 100k identity = 16 streams x 250 MHz / 40,000 cyc does NOT close: streams maxed at 16, clock 200 MEASURED / 250 route-dead, cycle floor MEASURED at ~51,100 NOT 40,000 (docs/6:126-147)
- At the ~51,100-cyc floor: 200 MHz -> ~62.6k, 250 MHz -> ~78.3k; 100k is off by ~11,000 cycles even with every idle hidden (RUN 25,088 + readback 10,360 = ~37k serial floor alone) (docs/6:139-147)
- The cycle law was verified bit-exactly against silicon: RUN = 4*6,144 + head 256r*2b = 25,088; RB = 9,472*35/32 = 10,360; remainder 17,916 = serial NL + FSM idle (100K-REVIEW.md:16-31)
- d and d_ff must stay multiples of 128 or a model shrink is partially wasted (K quantized to 128-lane beats: ceil(192/128)=ceil(256/128)=2) (100K-REVIEW.md:32-37)
- 2.0 INT4xINT8 MACs/DSP is the proven max: 3.0 needs 28b operand > 27b port AND 66b acc > 48b, falsified over 1.2M randomized products (dsp3_pack_proof.py:26-30)
- Split-brain works via xpm_memory_tdpram MEMORY_PRIMITIVE=ultra (56 URAM/0 LUT/0 BRAM); HDL inference of TDP UltraRAM is DEAD in Vivado 2025.2 (SPLIT-BRAIN.md:57-68)
- Model shrink to reach 100k was VETOED by the user (2026-06-09) => 100k is CLOSED on the KV260 at this model size; do not reopen without a bigger part or explicit reversal (100K-REVIEW.md:115-128)
- Silicon overclock margin is ~1.3-1.76x over STA; the SETTLE signature (silicon runs -273/-297 cyc vs sim, a skipped sim-only settle state) has held for 5-6 consecutive builds (docs/6:90-102)
- The round-trip ceiling was lifted ~5.6x (1k -> 5,600 localhost tok/s) by moving sampling on-chip via Gumbel-max (argmax(logit + T*g)), eliminating 193 host logit-reads/token; public path ~1,658 tok/s is WAN/tunnel RTT (README.md:75-93, log §44-45)
- Doc-7 faithful ladder slope fell 528 -> 128 -> 48 -> 24 -> 22.7 cyc/position; a 127-token gen averages 12,499 cyc = 16,001 @200 / 20,001 @250 SIM, full-window 248/248 bit-exact to T=255 (docs/7:122-137, log §29-35)
- 20k @250 is SIM-proven only; on silicon 200 fails timing (GEMV MAC accb carry chain ~8.75ns) and the PS PLL offers only 166.7/200, so the deployed faithful is ~16k (log §41-43)
- The shipping faithful build is K8 no-Hadamard at TMAX=128 (+0.09% NLL, clean to T=255); K4/V4+Hadamard was parked (66k-LUT butterfly bomb at 181% + un-debugged long-T divergence) (log §37, docs/7:139-150)

**Files**

- `/home/mikeayles/Desktop/Projects/kev-gpt/docs/6-past-the-stream-ceiling.md` — Doc 6 — the aggregate-throughput speed campaign: stream ceiling, split-brain, lever campaign, dead ends, the 100k identity and cycle floor. Primary narrative source.
- `/home/mikeayles/Desktop/Projects/kev-gpt/docs/7-kevin-remembers.md` — Doc 7 — the N=1 faithful-stream campaign: 20k target, R0-R5 rung ladder, the T=1 confession, KV-quant contract, where-it-loses.
- `/home/mikeayles/Desktop/Projects/kev-gpt/docs/ROADMAP-10K.md` — The original 10k roadmap + board-ceiling roofline analysis (tiers, ~200k single-stream ceiling, 500k impossibility, full-TinyStories fork). Predates the doc-6/7 era; the 24k STATUS banner at top is the bridge.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/research/100K-REVIEW.md` — The authoritative 2026-06-09 audit: bit-exact cycle-law verification, the dead-lever ledger, the model-shrink candidate table, and the VETO that CLOSED 100k on the KV260.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/research/SPLIT-BRAIN.md` — The split-brain design note: two N=8 cohorts on dual-ported URAM, the xpm_memory_tdpram-not-inference finding, budget and risks.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/progress.py` — The committed speed-progress ladder tool (tok/s per rung, MEASURED/SIM/PROJECTED tags). The single source for every rung's number; lines 23-82 are the LADDER list.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/WIDE-WORD-DATAPATH-LOG.md` — The engineering log; §12-27 are the doc-6 aggregate campaign, §28-45 the doc-7 faithful campaign + on-chip sampling. The section numbers cited throughout doc 6/7 point here.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/rtl/sequencer_sb.sv` — The live aggregate engine RTL. Lines 20-44 are the parameter block (N/NC/TMAX/ATT2/DBG/DP); 154-202 the ATT2 per-cohort-vs-shared attention generate.
- `/home/mikeayles/Desktop/Projects/kev-gpt/fabric/stage3/research/dsp3_pack_proof.py` — The exhaustive proof that 2.0 MAC/DSP is the max (3.0 dies on the 27b operand and 48b accumulator walls). The load-bearing streams-maxed evidence.
- `/home/mikeayles/Desktop/Projects/kev-gpt/docs/DOUBLE-PUMP-100K.md` — Aspirational (2026-06-08) clk2x MAC-island plan projecting ~88k. SUPERSEDED/invalidated the next day by 100K-REVIEW; read it knowing double-pump is DEAD, not live.
- `/home/mikeayles/Desktop/Projects/kev-gpt/README.md` — The two-ceilings section (fabric vs round-trip, lines 75-93) — the canonical framing of why the fabric number and the user-felt number differ.

**Gotchas**

- Doc 6's lever-campaign table (docs/6:54-60) tops out at 56,262.7 and line 76 says 'the 56.3k record is the last build that reached silicon' — this is STALE relative to its own closing (docs/6:160) and §27 of the log, which report 59,965.5 @200 as the sixth silicon build (53,364 cyc). Trust §27 / progress.py:68 for the record, not the mid-doc table.
- The 100k identity's '40,000 cyc' factor is aspirational and NOT achievable — the MEASURED cycle floor is ~51,100 at LANES=128/P=8. Anyone quoting '100k = 16 x 250 / 40k' as a live plan is reading the target, not the finding (docs/6:136-147).
- DOUBLE-PUMP-100K.md reads as an optimistic GO plan (~88k) but was invalidated the next day by 100K-REVIEW.md and is listed dead in doc-7 §5. Do not treat the double-pump as a live lever; the fabric->clk2x feed walls at 50 MHz and the routed clk2x path is WNS -4.658.
- 'Two campaigns' (N=16 aggregate vs N=1 faithful) and 'two ceilings' (fabric vs round-trip) are ORTHOGONAL concepts that are easy to conflate — the README two-ceilings section is about fabric-vs-serving; the doc-6/doc-7 split is about aggregate-vs-faithful. Each faithful/aggregate engine has its own fabric number AND a round-trip number.
- The faithful 20,001 tok/s @250 number is SIM-only (log §35); on silicon the best faithful deployed is ~16k @166.7 because 200 MHz fails timing (GEMV MAC accb carry chain) and the PS PLL has no rung between 166.7 and 200. 20k on this RTL is not reached on silicon.
- The N=16 record bitstream builds with TMAX=16 and ATT2=0 (shared attention + arbiter), NOT the sequencer_sb.sv defaults (TMAX=32, ATT2=1). The defaults reflect the sim/development config; the shipping build parameters are set in the tcl/board flow.
- The board driver pl_seq_sb --tmax MUST match the build TMAX or attention positions past 0 corrupt (the embed upload row count is TMAX-dependent) (log §27 lines 996-998).
- §45's greedy 17,936.6 tok/s (9292 cyc/tok on the gumbel build) is higher than progress.py's top faithful MEASURED rung of 16,087.5 (10,360 cyc/tok full-window). They are not directly comparable — the 9292 is a shorter-T gate run and 10,360 is a full-window average. README pins the deployed fabric number as 16,227.

**Open questions**

- Exact reconciliation of the faithful fabric numbers: §45 reports greedy 9292 cyc/tok = 17,936.6 tok/s and 16,227 fabric tok/s 'RECORD PRESERVED' in the same section, while progress.py's top faithful rung is 16,087.5 @166.7 (10,360 cyc/tok). Which cycle law / T applies to which number is not spelled out in one place; a future engineer should confirm against the actual board run logs before quoting a single faithful fabric figure.
- Whether the 250 MHz aggregate route ever became achievable after the §27 'next lever' (the embed->residual URAM cone + an ATT2=0 density cut) — the log at §27 says '250 silicon remains the gate between 60k and 75k+' but I found no later section proving a 250-clean aggregate build. Status appears to be: still route-blocked.
- The doc-7 T=256 rung status: the shipping build is TMAX=128 (K8 no-Hadamard); the T=256 rung (shared constant-geometry Hadamard unit + K4 long-T divergence debug) is described as 'queued' in log §37 but I did not find a section confirming it shipped. Assume T=256 faithful is NOT yet on silicon.
- ROADMAP-10K.md's ~200k single-stream and full-TinyStories fork analysis predates the doc-6/7 measured era and uses a different (roofline) framing than the later MEASURED cycle-floor finding (~51,100 -> ~62-78k). The two are not reconciled in one document; the MEASURED 100K-REVIEW numbers should be treated as authoritative over the earlier roofline projections.
- The DOUBLE-PUMP-100K.md vs 100K-REVIEW.md conflict (GO vs DEAD, one day apart) is resolved in favor of DEAD by dates + doc-7 §5, but I did not find an explicit note IN the DOUBLE-PUMP doc marking it superseded — a future reader opening that file first could be misled into re-exploring it.

---

# Verification log

Adversarial re-check of load-bearing facts. Non-confirmed items:

- **[UNRESOLVED]** Which sampler is live (on-chip baked 0.85 vs host-side 0.4). _The out-of-repo memory note (live-demo-topology.md:13) says host-side softmax @temp 0.4/top-k 10, but the committed code with those exact flags samples ON-CHIP at a baked temperature and ignores top-k (`a53_daemon.py:289-304` passes no host_sample; `pl_kv256.py:126` sets `_onchip = temp>0 and not host_sample`). The baked-0.85 LUT constant is correct; the live sampling path can be resolved only by reading the Kria's systemd ExecStart + the on-board a53_daemon.py (which may be hand-edited vs repo)._

## Gaps flagged (facts no section could establish)

- The gum bitstream's TMAX=128 is NOT a build_bd_seq_kv.tcl default (default is tmax=256, build_bd_seq_kv.tcl:27); it rests on the invocation tclargs, evidenced only indirectly by the mems dir name (stage3_vec_kvk8t128_smp) and the live daemon '--tmax 128'. The doc should state TMAX=128 was passed at build invocation, not inferred from the TCL default.
- **UNRESOLVED** which sampler is live: the memory note (live-demo-topology.md) records the kevkv daemon flags '--temp 0.4 --top-k 10' as host-side, but the committed code with those flags samples ON-CHIP at the baked temp and ignores top-k. So neither "on-chip is live" nor "host-side is live" can be asserted from the repo; resolve by reading the Kria's systemd ExecStart + on-board a53_daemon.py (may be hand-edited vs repo).
- **RESOLVED (ND=6/cohort):** the §27 log text does not literally state ND, but it is DERIVED as ND=6/cohort from build_bd_seq_sb.tcl's default nd=6, the committed §18/§22-26 gate config (`run_sb_seq --nd 6`), and the ~95% DSP occupancy of the record fit (100K-REVIEW.md:53 "DSP 95.1%") — all-LUT ND=0 would leave the DSPs idle. The RTL/sim default ND=0 (pure LUT MACs) is a *default*, not the built value; earlier "ND=0 built" statements misread it.
- gemv_axi_seq_vec.v wrapper parameter defaults (P=16, LANES=128, WWORDS=262144, TMAX=64 at L19-24) differ from the values the gum bitstream actually shipped with; those are overridden by build_bd_seq_kv.tcl's set_property CONFIG.* at build time. A future engineer reading the RTL defaults alone would get the wrong build parameters.
- The '~37k serial floor' (RUN 25,088 + readback 10,360) is arithmetically 35,448, not 37,000 -- the source docs/6:147 and 100K-REVIEW round it up by ~1,550 cyc. The downstream 'off by ~11,000' and floor ~51,100 figures are self-consistent, but the ~37k intermediate is loose. Not a mis-tag (all MEASURED/DERIVED tags are correct), just an imprecise recap in the prose.
- The faithful faithful-stream numbers use two slightly different tok/s: progress.py:78-79 records 16,087.5 @166.7 (greedy, 10,360 cyc trims build) while README.md:81 headlines 16,227 (the on-chip-sampling gum build, 117 passes/7.21ms). Both are MEASURED and traceable, but the doc should make clear they are two different bitstreams (trims vs gum) so a future reader does not treat 16,087.5 and 16,227 as the same build.
- fabric/progress.py has one SIM rung (75.8, 'resident-read GEMV') interleaved among MEASURED rungs; it is correctly tagged SIM in the tuple but the ladder is not monotonic in provenance -- worth a footnote so the SIM value is not read as a silicon measurement.
- Could NOT independently reconfirm the historical mid-ladder MEASURED rungs (231.0, 751.78, 1882.7, 2483.9, 3511.6, 5448.8) against their originating board logs within this pass -- they are internally consistent with the cyc/tok and MHz annotations in progress.py but the underlying silicon runs were not re-opened; treat as CONFIRMED-by-annotation, not re-derived.
- CLAUDE.md is STALE/overclaiming relative to the research docs: it states 'The target is the 100k identity' and 'cycles are ~53k gated heading to ~40k', but docs/6:126-147 and 100K-REVIEW.md:115-128 conclude the cycle floor is ~51,100 (NOT 40,000) and 100k is CLOSED/VETOED (2026-06-09). A future engineer reading CLAUDE.md alone would think 100k is still an open target. The permanent doc should cite the research-doc verdict, not CLAUDE.md's optimistic framing.
- Two sampling paths coexist and the doc must NOT overclaim which is LIVE — it is **UNRESOLVED**. The memory note says the deployed daemon does host-side softmax @temp 0.4/top-k 10 over head logits (rd_sel=8), but the committed code with those exact flags samples ON-CHIP at the baked 0.85 LUT and ignores top-k (`a53_daemon.py:289-304`, `pl_kv256.py:126`). The on-chip Gumbel path shipped and passed gates (log §45). Resolve only by reading the Kria's systemd ExecStart + on-board a53_daemon.py (may be hand-edited vs repo).
- The record engine (sequencer_sb, split-brain, N=16, @200MHz) has NEVER been deployed to the public demo and produces degenerate text (attention T=1 hardwired). The 59,965.5 tok/s headline is an aggregate-throughput microbenchmark on bit-exact-but-textually-degenerate output, not the user-facing chat. The deployed engine is the faithful single-stream sequencer_vec at ~16-17.9k tok/s. This separation is honest in docs/6 and docs/7 but is the single most important thing a future engineer must not conflate.

## Open questions (union across subsystems)

- Exact per-phase cycle attribution within the 53,364-cycle pass is not decomposed in the RTL or the log — I could determine the total and the deltas from each lever (§22-26) but not a clean phase-by-phase breakdown (e.g. how many of the 53k cycles are GEMM run vs. attention vs. argmax).
- **RESOLVED** (§7 / dead-lever ledger): the full DP=1 build was never board-run (OOC-timing-MET only); a MAC-only DP branch (`dp-hw-maconly`) WAS run on silicon bit-exact, but the fabric→clk2x data feed walled at 50 MHz — that measurement killed the lever (100K-REVIEW.md:44).
- The precise LUT/BRAM/DSP fit numbers for the ATT2=1 (per-cohort attention) config at BD level — the RTL comment says it is ~1.9k LUT over the device (sequencer_sb.sv:156-158) but I did not open an OOC/impl report to confirm the current margin.
- How completion (`done_o`) interacts with the stream-granular overlap under skew between cohorts on real silicon beyond the 3/3 record runs — the log notes an earlier tb_gemm_sb latch bug from one-cycle done pulses under deliberate skew (WIDE-WORD-DATAPATH-LOG.md:667-673), but that was a harness bug, not the sequencer_sb both_done edge-detect (sequencer_sb.sv:376-386), which I did not independently stress-check.
- **RESOLVED** (§7 / dead-lever ledger): the full DP=1 split-brain build was never board-run; the on-silicon clk2x bit-exact result exists only for the MAC-only DP branch (`dp-hw-maconly`), where the fabric→clk2x data feed walled at 50 MHz and killed the lever (100K-REVIEW.md:44).
- gemm_dsp_resident_vec.sv (the 24-bit-gap all-DSP core) appears superseded by mac_bank_dsp's 22-bit RAW-pair form used in the batch/cohort cores; whether gemm_dsp_resident_vec is still instantiated anywhere live was not confirmed (no top-level instantiation found in sequencer_sb).
- The exact current URAM/DSP/LUT fit of the LIVE sequencer_sb split-brain build (DBG=0/ATT2=0 bitstream config) was not re-derived from a fresh OOC in this read — the cited fit numbers (URAM 60-64/64, DSP 505-1171) come from the log's historical build entries, not a current run.
- ND (DSP-packed streams per cohort) defaults to 0 in the RTL, but the 59,965.5 tok/s record build ran ND=6/cohort (12 of 16 streams on the DSP-packed leaf) — DERIVED from BD default + committed gate config + ~95% DSP occupancy. The all-LUT (ND=0) path is gated bit-exact and synth-proven but was NOT the record build. (Earlier drafts stating the record ran ND=0 / "all-LUT mac_bank" misread the RTL default.)
- I did not fully trace where sequencer_sb / cohort_engine sources and stores the K/V it streams into the shared vec_attn between tokens (cohort_engine.sv:150 references a datapath submodule with NHEAD/HEAD_DIM). The record path's cross-token KV scratch layout was outside this subsystem's core (kv_bank/vec_attn/DDR) and I did not confirm it.
- Exact URAM/BRAM tile counts I report for code_bank at each TMAX (57 @128, ~114 @256) are quoted from the log's OOC runs, not independently recomputed from primitive geometry; they depend on how XPM tiles a 512-bit-wide memory.
- Whether the K4/V4+Hadamard TMAX=256 path (parked with a long-T divergence at ~pos 84) has been revisited/fixed since log §37 - the log leaves it un-debugged.
- The DDR path's KBITS=4 default vs the doc's INT8 budget: I did not find a single reconciled spec pinning the intended production DDR precision (the modules are parameterised, the doc analyses both).
- I did not find any on-chip top-k or on-chip runtime-temperature capability — both appear host-side only (pl_kv256._sample). If a bitstream variant with on-chip top-k exists it is not in the files reviewed.
- I did not measure/verify the exact cycle cost of the Gumbel precompute overlap with the head GEMV on silicon (the code claims it 'rides' the head GEMV with no extra read ports, sequencer_vec.sv:188-189); whether it ever stalls S_ARGMAX (gated on gpre_done) for small VOCAB vs GEMV length was not timing-analyzed.
- The task named layernorm_par.sv as a primary file but neither live engine instantiates it (only layernorm_vec is wired into sequencer_sb/sequencer_vec). Its current role appears to be a documented throughput reference; whether any tb/build still uses it was not exhaustively checked.
- I did not confirm on-silicon that the Gumbel sampler is 'recently landed' beyond the RTL + gate + board-driver support being present and bit-exact in sim; no MEASURED board tok/s or acceptance log for the sampled path was located in the files I read.
- Whether sequencer_sb could be given the Gumbel port (the split-brain record engine currently cannot sample) is not addressed in the reviewed RTL; the two features (max throughput vs on-chip sampling) currently live in different engines.
- The exact register decode of the baseline gemv_axi_seq.v (SEQR) and gemv_axi_seq_fast.v (SQRF) wrappers was not opened in this pass; the SEQR/SQRF map here is taken from pl_seq_chat.py's documented offsets (pl_seq_chat.py:15-26,103-107), which are internally consistent but not cross-checked against that RTL as the SQRV/SQSB maps were.
- The C MMIO backend (pl_resident_c.PLResidentC / gemv_axi_drv) used by --engine kv --kv-backend c and by PLResident (fabric.pl_gemv) was referenced but not read; its ctypes register protocol and whether it matches the SEQR/SQRF map is not documented here.
- backend_kv.KVChatDevice (the --engine kv daemon device) was not read in full; only its make_device wiring and the standalone pl_kv_chat.py equivalent are documented — its exact gen_chars/greedy plumbing and fabric_ms reporting may differ.
- The precise poll strategy differs between drivers (busy-poll with wall-clock timeout vs a fixed 1,000,000-iteration spin in backend_stub._submit_once); which is used in the live daemon path for each engine, and whether the spin can under-run at low fclk, was not measured.
- How server.py maps its --backend pl/tcp/kv choices onto local PLDevice vs remote TcpPLBackend, and whether the live chat.mikeayles.com deployment uses the split (i7+Kria) or single-box (server-on-board) topology, is only partially visible here (server.py:407-418) and not fully traced.
- The exact tclargs string used to build gemv_seqkv_gum is not in the repo (build dirs are on the Windows build box under C:/kevbuild/stage3_seqkv_bit). The params are firmly inferred (P=8/LANES=256/TMAX=128/125-target/166.7-run) from three agreeing sources but the literal invocation line / util_impl.rpt / timing_impl.rpt on the build box would confirm WWORDS and P beyond doubt.
- The gum bitstream contains on-chip Gumbel sampling (SEED reg 0x30). **UNRESOLVED** whether the deployed daemon samples on-chip or host-side: the memory note says host-side temp 0.4/top-k 10 (on-chip 0.85 "was gibberish"), but the committed code with those flags samples ON-CHIP at the baked temp and ignores top-k. Knowable only from the live kevkv.service ExecStart + the on-board a53_daemon.py on the Kria (which may be hand-edited vs repo), not the repo.
- Whether a gemv_seqsb (split-brain record) bitstream was ever deployed to the live chat, or only bench-measured, is not stated — the live daemon loads the KV gum bitstream, so the 59,965.5 record engine appears to be a bench/record artifact rather than the served engine.
- **RESOLVED**: the FULL double-pump DP=1 build (gemv_seqsb_dp) is OOC-timing-validated at 200/400 MHz and was never board-run. A MAC-only DP branch (`dp-hw-maconly`) DID run on silicon bit-exact — but the fabric→clk2x data feed walled at 50 MHz, so clk2x did NOT hold 400, and the lever is DEAD (100K-REVIEW.md:44).
- model/goformer_kvq.py (23KB) internals were not read in this pass — the exact K8/V8 quantisation, rotate/divfree semantics, and how IntKVQSequencer differs from seq_ref.IntSequencer's attention path are undocumented here. Read it directly before modifying the doc-7 KV gate.
- The RTL modules themselves (sequencer_sb.sv, cohort_engine.sv, nl_engine.sv, vec_attn.sv, layernorm_vec.sv, etc.) were not read — this section documents the Python reference/gate/model side only. The claim 'bit-identical to rtl/sequencer.rsh_round' etc. is asserted by the reference's comments, not verified against the RTL here.
- fabric/stage3/gumbel.py (make_gumbel_lut / GumbelRng) was not read; the exact on-chip Gumbel-max sampling arithmetic is only known through run_vec_kv's usage.
- The precise cyc/token and Fmax attached to each progress.py rung are transcribed from the LADDER string literals (self-reported MEASURED values); they were not independently re-derived from the harness cyc.out outputs in this pass.
- pack_banked.write_case/check (the GEMV stimulus + bit-exact checker behind run_banked) were not read in detail.
- The doc-6/doc-7 design docs (6-past-the-stream-ceiling.md, 7-kevin-remembers.md) and WIDE-WORD-DATAPATH-LOG.md §19-36 referenced throughout the code comments were not read; they hold the authoritative lever-alive/dead status and the schedule-overlap history.
- Exact reconciliation of the faithful fabric numbers: §45 reports greedy 9292 cyc/tok = 17,936.6 tok/s and 16,227 fabric tok/s 'RECORD PRESERVED' in the same section, while progress.py's top faithful rung is 16,087.5 @166.7 (10,360 cyc/tok). Which cycle law / T applies to which number is not spelled out in one place; a future engineer should confirm against the actual board run logs before quoting a single faithful fabric figure.
- Whether the 250 MHz aggregate route ever became achievable after the §27 'next lever' (the embed->residual URAM cone + an ATT2=0 density cut) — the log at §27 says '250 silicon remains the gate between 60k and 75k+' but I found no later section proving a 250-clean aggregate build. Status appears to be: still route-blocked.
- The doc-7 T=256 rung status: the shipping build is TMAX=128 (K8 no-Hadamard); the T=256 rung (shared constant-geometry Hadamard unit + K4 long-T divergence debug) is described as 'queued' in log §37 but I did not find a section confirming it shipped. Assume T=256 faithful is NOT yet on silicon.
- ROADMAP-10K.md's ~200k single-stream and full-TinyStories fork analysis predates the doc-6/7 measured era and uses a different (roofline) framing than the later MEASURED cycle-floor finding (~51,100 -> ~62-78k). The two are not reconciled in one document; the MEASURED 100K-REVIEW numbers should be treated as authoritative over the earlier roofline projections.
- The DOUBLE-PUMP-100K.md vs 100K-REVIEW.md conflict (GO vs DEAD, one day apart) is resolved in favor of DEAD by dates + doc-7 §5, but I did not find an explicit note IN the DOUBLE-PUMP doc marking it superseded — a future reader opening that file first could be misled into re-exploring it.
