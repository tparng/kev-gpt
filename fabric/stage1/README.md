# Stage 1 — on-chip INT4 systolic GEMV

Doc 2's stage 1: the hand-rolled INT4×INT8 matrix-vector engine with weights
resident on-chip, microbenchmarked in isolation. This is the "satisfying
checkpoint" — raw on-chip matmul throughput, before the fiddly non-linearities.

**Status: bit-exact in simulation against the numpy golden, all 17 model layers
(`maxabserr=0`).** Synthesis script is ready; running it needs Vivado (installed)
but no board.

## What it computes

```
y[m] = sum_k  W[m,k] * x[k]      m in 0..M-1, k in 0..K-1
```

- `W` INT4 signed [-8,7], resident in on-chip ROM (URAM-targeted) — never DDR.
- `x` INT8 signed activation vector (one decode step).
- `y` INT32 exact integer accumulation; dequant scales (`*.wscale.mem`, per-channel)
  are applied downstream. The integer core is what stage 1 proves.

`PE_LANES` output rows accumulate in parallel; `M` is processed in
`ceil(M/PE_LANES)` row-groups with one activation broadcast per cycle, so a run
is `~ceil(M/PE_LANES)*K` cycles. The weight memory layout is exactly the
`fabric/pack.py` contract (packed bytes, row-major, low nibble = even k), so the
RTL `$readmemh`s the same image the exporter wrote.

## Files

```
rtl/gemv_int4.sv      the engine (conservative style: Icarus + Vivado both take it)
tb/tb_gemv_int4.sv    self-checking TB; dumps y to .mem, Python does the compare
sim/run_iverilog.sh   generate config -> iverilog -> vvp -> bit-exact Python verdict
tcl/synth.tcl         Vivado OOC synth for utilisation + timing (Fmax)
```

## Simulate (Icarus, no board)

Prereqs — produce the weights and golden vectors first:

```
python -m model.export_fabric data/ckpt.qat.pt   # INT4 .mem + scales + manifest
python -m fabric.golden --all                     # per-layer x.mem / y.mem golden
```

Then run one layer (keys are the exporter's: `blocks_0_qkv`, `blocks_0_proj`,
`blocks_0_mlp_fc`, `blocks_0_mlp_proj`, …, `head`):

```
bash fabric/stage1/sim/run_iverilog.sh blocks_0_proj 16
# -> PYVERDICT bitexact=True maxabserr=0 n=256
```

The verdict is computed in Python from the RTL's `$writememh` dump vs the numpy
golden — bit-exact or it fails. "bit-honest before fast": this gate must pass
before any throughput number is trusted.

## Synthesise (Vivado, no board)

```
vivado -mode batch -source fabric/stage1/tcl/synth.tcl -tclargs 256 256 16
# reports -> fabric/stage1/synth/{util,timing}_*.rpt
```

Out-of-context, weights left uninitialised (this measures the datapath + memory
resources and Fmax, not a specific weight image). Part is the KV260 SOM
`xck26-sfvc784-2LV-c`.

## Honest scope / next

- This is the **integer GEMV**. Per-channel dequant (`y*wscale*xscale`),
  activation quant, and the non-linearities (softmax, RMSNorm, GELU) are stages
  2–3 — moved into fabric next to the matmul to kill the A53 round trips.
- The weight memory here is a single packed-byte array; the real throughput
  build banks it per lane so `PE_LANES` weights land per cycle. The current core
  proves correctness and gives a representative resource/timing point; the
  banked, fully-pipelined version is the stage-1 throughput follow-up.
- Numbers to harvest from `synth.tcl`: DSP/LUT/FF/URAM and Fmax, which feed the
  roofline crossover (`fabric/roofline.py`) with measured rather than assumed
  on-chip bandwidth.
