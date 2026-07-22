# Board test runbook — what to run when Tailscale re-auths

Honest state going in: today's work was all laptop sim/synth. The board still has the
**existing PE=1 resident bitstream** loaded and runs at **0.22 tok/s** (PL, Python) /
~11 tok/s (A53 chat). Nothing faster is *running* yet. This runbook brings the new,
gated pieces onto the board and measures them.

Re-auth first (the SSH check tonight hit an expired Tailscale session):
```
# on the board console or after `tailscale up` re-login:
ssh ubuntu@<kria-ip>              # alias: kria-kev ; Tailscale address, key auth set up
```

---

## Test A — the certain win: KV cache + C driver on the EXISTING bitstream
No new bitstream. Replaces the slow Python; kills the O(T²) recompute. **Laptop gate
already green: `PL_KV_VERDICT identical=True stream=32/32`.**

```
# 1. sync the repo + board package to the board (from laptop or git pull on board)
cd ~/kevpl && git pull            # or: scp -r fabric/stage3/board ubuntu@kria-kev:~/kevpl/fabric/stage3/

# 2. build the C MMIO driver on the board (gcc 11.4 is there)
cd ~/kevpl/fabric/stage3/board
gcc -O2 -fPIC -shared -o libgemv_axi_drv.so gemv_axi_drv.c
gcc -O2 -o gemv_axi_bench gemv_axi_bench.c gemv_axi_drv.c

# 3. confirm the bitstream is loaded (IDCODE check) — if not, load the resident one:
#    sudo xmutil unloadapp && sudo fpgautil -b ~/kevpl/gemv.bit.bin

# 4. re-run the equivalence gate ON the board python, then generate:
cd ~/kevpl && /home/ubuntu/kevweb/venv/bin/python -m fabric.stage3.board.gate_pl_kv
sudo /home/ubuntu/kevweb/venv/bin/python -m fabric.stage3.board.pl_kv_chat --backend pl --prompt 'once upon time' --n 40 --greedy
sudo /home/ubuntu/kevweb/venv/bin/python -m fabric.stage3.board.pl_kv_chat --backend c  --prompt 'once upon time' --n 40 --greedy

# 5. raw matmul microbench (cycles for one GEMV on silicon):
sudo ./gemv_axi_bench 768 256 0 1000
```
**Expect:** `--backend pl` ≈ **5 tok/s** (Python + KV, ~20× over today's 0.22),
`--backend c` ≈ **40–126 tok/s** (compiled AXI loop — the Tier-2 ceiling). These are
DERIVED from the AXI transaction count (~26k reg txns/token); the board run is the
first MEASURED number. The wall here is the A53 doing ~26k AXI transactions/token —
*not* the matmul. That wall is why the dataflow sequencer (below) is the real path.

---

## Test B — the new wide-GEMV bitstream (BUILT ✅, timing-closed)
**Built on the laptop, placed+routed+timing-closed on `xck26-2LV`:**
`C:/kevbuild/stage3_bit/gemv_banked.bit.bin` (7.8 MB), 60/64 URAM (94%), 100 MHz,
*all timing constraints met*. AXI slave @ 0xA000_0000, IDCODE `0x47454D42` ("GEMB").
```
# copy the bitstream to the board (from laptop, once re-authed):
scp C:/kevbuild/stage3_bit/gemv_banked.bit.bin ubuntu@kria-kev:~/kevpl/
# load it:
sudo xmutil unloadapp && sudo fpgautil -b ~/kevpl/gemv_banked.bit.bin
# confirm IDCODE = 0x47454D42 (GEMB), then point the driver at it (PLBankedResident
# in fabric/pl_gemv.py preloads transposed weights + per-layer W_BASE) and re-run
# pl_kv_chat / the bench.
```
**Honest expectation:** via the AXI driver this is **still ~100 tok/s** — the PE=256
matmul is *hidden behind the AXI transaction overhead*. Its value is (a) silicon-
validating the resident-banked design (URAM fit @ ~60 blocks, 100 MHz timing) for the
sequencer, and (b) the microbench measuring the wide GEMV's raw cycles. **Do not expect
a token/s jump from this bitstream over Test A** — the speedup needs the CPU out of the
per-token loop (the sequencer), not a wider matmul behind the same driver.

---

## Test C — the SEQUENCER bitstream (on silicon, MEASURED ✅)
The capstone: the **whole single-stream autoregressive transformer in fabric, CPU out of
the loop**. Host streams the INT4 weight image into URAM once, writes the prompt, pulses
GO; the fabric does embed → 4 blocks → ln_f → head → argmax for NGEN tokens on-chip.
Bitstream `~/kevbit/gemv_seq.bit.bin` (7.8 MB), IDCODE `0x53455152` ("SEQR"),
40 MHz timing-closed (WNS +0.686 ns), 54/64 URAM.

```
# load the flat bitstream (no dtbo/app overlay needed):
sudo xmutil unloadapp ; sudo fpgautil -b ~/kevbit/gemv_seq.bit.bin
# run the gated driver (it FORCES + VERIFIES the PL clock itself — see the gotcha below):
cd ~/kevpl/plchat
sudo ~/kevweb/venv/bin/python -m fabric.stage3.board.pl_seq_chat \
     --npz goformer.npz --meta goformer_meta.json --prompt "once upo"
```
**MEASURED:** `tokens_match=True (8/8)`, hw stream `[47,1,53,42,46,38,1,53]` = `"n time t"`,
CYCLES=7,220,220 → **44.32 tok/s @ 40 MHz**, deterministic across 5/5 runs. This is the
first VALID on-silicon single-stream number — token-stream bit-exact to `seq_ref`, ~200×
the 0.22 cold baseline and ~4× the A53-optimised ~11 tok/s wall, with the CPU idle.

### ⚠️ THE CLOCK GOTCHA (cost a whole debug session — do not forget)
A **flat `fpgautil` bitstream load does NOT apply the block design's `PL0_REF_CTRL
FREQMHZ` setting** — that only happens through the xmutil/device-tree *app* flow. So
`pl0_ref` stays at the system default (**~100 MHz**). Running this 40 MHz-closed design
at 100 MHz violates setup on the wide arithmetic (the 64×64 / 96-bit dequant clouds) →
the data path emits **non-deterministic garbage**, while the cycle counter + AXI/MMIO
keep working (short paths). That asymmetry — CYCLES bit-identical to sim but tokens
random run-to-run — is the fingerprint. Fix, no resynthesis:
```
echo 40000000 | sudo tee /sys/devices/platform/fclk0/set_rate   # force pl0_ref to 40 MHz
grep pl0_ref /sys/kernel/debug/clk/clk_summary                  # verify == 40000000
```
`pl_seq_chat.py` / `pl_seq_diag.py` now do this automatically (`set_and_verify_fclk`) and
ABORT if the clock is wrong — a tok/s number measured at the wrong clock is a lie.
`pl_seq_diag.py` is the localiser that nailed it: MMIO-stability + load-vs-compute
(GO ×N with no reload) + reload-determinism probes.

## Test D — the LEAP bitstream: resident-read + PE=256 (on silicon, MEASURED ✅)
`sequencer_fast` (whole INT4 image resident in the GEMV core's URAM, no per-matmul reload)
widened to 256 MACs/cycle. Bitstream `~/kevbit/gemv_seqfast.bit.bin`, IDCODE `SQRF`
(0x53515246), 40 MHz, **URAM 60/64, LUT 87.4%, WNS +0.010 ns** (closed razor-thin — the
one thing to watch; if a future rebuild congests worse, drop the forced clock a few MHz).
```
sudo xmutil unloadapp ; sudo fpgautil -b ~/kevbit/gemv_seqfast.bit.bin
cd ~/kevpl/plchat
sudo ~/kevweb/venv/bin/python -m fabric.stage3.board.pl_seq_chat --lanes 256 \
     --npz goformer.npz --meta goformer_meta.json --prompt "once upo"
```
The driver auto-detects via `--lanes 256`: streams SUBW=32 32-bit W_DATA chunks per
1024-bit word, accepts the SQRF IDCODE, forces fclk0=40 MHz. **MEASURED:** stream
`[47,1,53,42,46,38,1,53]` = `"n time t"`, CYCLES=1,385,205 → **231.01 tok/s @ 40 MHz**,
`tokens_match=True`, **deterministic 5/5 runs** (the +10 ps margin held). 5.2× the 44.32
baseline, ~21× the A53 ~11 wall, CPU fully out of the loop. The weight load is ~400k MMIO
pokes (~3 s); a C/DMA loader would shrink that but it is one-time, not per-token.

## Test E — GELU+LN optimisations + on-silicon clock sweep (MEASURED ✅ 751.78 tok/s)
`sequencer_fast` + streaming GELU (kills the per-element gelu_lut stall) + pipelined
LayerNorm (Newton + S_OUT split into single-multiply stages, breaking the DSP cascade that
capped Fmax). Built at **PE=128** (vs 256) to relieve LUT congestion (77.6% LUT, URAM 56/64).
Bitstream `~/kevbit/gemv_seqfast_p128.bit.bin`, IDCODE `SQRF`, 166,273 cyc/token.
```
sudo fpgautil -b ~/kevbit/gemv_seqfast_p128.bit.bin
cd ~/kevpl/plchat
# sweep the PL clock to find the silicon Fmax (--fclk; driver uses the actual readback rate):
sudo ~/kevweb/venv/bin/python -m fabric.stage3.board.pl_seq_chat --lanes 128 \
     --fclk 125000000 --npz goformer.npz --meta goformer_meta.json --prompt "once upo"
```
**Clock sweep (bit-exact gated, --lanes 128):** 71→430, 91→547, 100→601, 111→668, all
deterministic; **125 MHz → 751.78 tok/s, tokens_match=True, 3/3 deterministic** ("n time t").
142 MHz → tokens_match=False (non-deterministic — past the silicon edge). So the measured
ceiling is **125 MHz / 751.78 tok/s** (3.26× the PE=256 baseline, 17× the A53 ~11 wall).
HONEST: STA closes the build at only ~71 MHz (430 tok/s, guaranteed-across-corners); the
silicon overclocks clean to 125 (this chip, this temp) and breaks at 142 — the 751 is
bit-exact-gated but chip/temperature-specific, not an STA spec. Lesson: STA on this -2LV
part is very pessimistic; the bit-exact token-stream gate is what bounds the real Fmax.

## What this session will and won't show
- **Will:** the fabric path goes from 0.22 → ~5–126 tok/s (finally beating the A53),
  measured, bit-identical Kevin output. The resident-banked design validated on silicon.
- **Won't:** 10k (needs the sequencer FSM bitstream — CPU out of loop) or 100k (needs
  the batched + multi-GEMV-engine build). Those are the next bitstreams, not this one.

## tok/s ladder (where each rung comes from)
| Rung | tok/s | What it needs | Status |
|---|---|---|---|
| today (PE=1 Python) | 0.22 | — | MEASURED |
| Python + KV cache | ~5 | board pull + run | staged, gated |
| C + KV cache | ~10 | gcc build on board | MEASURED (10.35; A53-orchestration-bound) |
| **single-stream dataflow (sequencer)** | **44.32** | sequencer bitstream @ 40 MHz, CPU out of loop | **MEASURED, bit-exact** |
| + resident-read (sequencer_fast, PE=16) | ~76 | drop per-matmul weight reload | SIM bit-exact (527,616 cyc/tok) |
| **+ PE=256 (sequencer_fast, LANES=256)** | **231.01** | 16-wide lanes on top of resident | **MEASURED, bit-exact, 5/5 deterministic** (1,385,205 cyc; SQRF) |
| **+ streaming GELU + pipelined LN (PE=128)** | **751.78** | GELU stall + LN DSP cascade → Fmax 50→125 MHz | **MEASURED @125 MHz, bit-exact, 3/3** (STA-safe 71/430; breaks at 142) |
| + P-wide serial datapath | ~2k–10k (target) | the remaining 1-elem/cyc loops (dequant/attn/act-feed) + ~200 MHz | `PLAN-10K-DATAPATH.md` |
| batched serving | ~30–124k | + parallel non-linears + 3–4 GEMV engines | model proven; RTL TBD |
