#!/usr/bin/env bash
# Simulate the Stage 1 GEMV against a layer's golden vectors with Icarus Verilog,
# then bit-exact compare the RTL dump to the numpy golden in Python.
#
#   fabric/stage1/sim/run_iverilog.sh [layer_key] [PE]
#
# Defaults to blocks_0_proj (256x256), PE=16. Layer keys are the exporter's
# (dots -> underscores), e.g. blocks_0_qkv, blocks_0_mlp_fc, head.
# Prereqs:
#   python -m model.export_fabric data/ckpt.qat.pt
#   python -m fabric.golden --all
set -euo pipefail

LAYER="${1:-blocks_0_proj}"
PE="${2:-16}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

EXP="fabric/export"
TB="$EXP/tb"
SIM="fabric/stage1/sim"
DIMS="$TB/${LAYER}.dims.json"

if [[ ! -f "$DIMS" ]]; then
  echo "missing $DIMS — run: python -m fabric.golden --layer $LAYER" >&2
  exit 1
fi

read -r M K < <(python -c "import json;d=json.load(open(r'$DIMS'));print(d['M'],d['K'])")

cat > "$SIM/config.svh" <<EOF
\`define CFG_M $M
\`define CFG_K $K
\`define CFG_PE $PE
\`define CFG_WFILE "$EXP/${LAYER}.w.mem"
\`define CFG_XFILE "$TB/${LAYER}.x.mem"
\`define CFG_YDUT  "$SIM/${LAYER}.ydut.mem"
EOF

echo "layer=$LAYER M=$M K=$K PE=$PE"
iverilog -g2012 -I "$SIM" -o "$SIM/sim.vvp" \
    fabric/stage1/rtl/gemv_int4.sv \
    fabric/stage1/tb/tb_gemv_int4.sv
vvp "$SIM/sim.vvp"

# Authoritative compare in Python: RTL dump vs numpy golden, bit-exact.
python - "$SIM/${LAYER}.ydut.mem" "$TB/${LAYER}.y.mem" <<'PY'
import sys
def load(p):
    out = []
    for ln in open(p):
        ln = ln.strip()
        if not ln or ln.startswith("//") or ln.startswith("@"):
            continue  # skip iverilog $writememh comment/address lines
        v = int(ln.split()[0], 16)
        out.append(v - (1 << 32) if v >= (1 << 31) else v)
    return out
dut = load(sys.argv[1]); gold = load(sys.argv[2])
n = min(len(dut), len(gold))
maxerr = max((abs(dut[i] - gold[i]) for i in range(n)), default=-1)
ok = dut[:n] == gold[:n] and len(dut) == len(gold)
print(f"PYVERDICT bitexact={ok} maxabserr={maxerr} n={n}")
PY
