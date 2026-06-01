#!/usr/bin/env bash
# Simulate gemv_core (Stage 2 keystone) against a layer's golden vectors with
# Icarus, then bit-exact compare the RTL dump to the numpy golden in Python.
#   fabric/stage2/sim/run_core.sh [layer_key]
# Defaults to blocks_0_proj. Prereqs: model.export_fabric + fabric.golden --all.
set -euo pipefail

LAYER="${1:-blocks_0_proj}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

EXP="fabric/export"; TB="$EXP/tb"; SIM="fabric/stage2/sim"
DIMS="$TB/${LAYER}.dims.json"
[[ -f "$DIMS" ]] || { echo "missing $DIMS — run: python -m fabric.golden --layer $LAYER" >&2; exit 1; }
read -r M K < <(python -c "import json;d=json.load(open(r'$DIMS'));print(d['M'],d['K'])")

cat > "$SIM/config.svh" <<EOF
\`define CFG_M $M
\`define CFG_K $K
\`define CFG_WFILE "$EXP/${LAYER}.w.mem"
\`define CFG_XFILE "$TB/${LAYER}.x.mem"
\`define CFG_YDUT  "$SIM/${LAYER}.ydut.mem"
EOF

echo "layer=$LAYER M=$M K=$K (gemv_core, registered-read BRAM)"
iverilog -g2012 -I "$SIM" -o "$SIM/sim.vvp" \
    fabric/stage2/rtl/gemv_core.sv \
    fabric/stage2/tb/tb_gemv_core.sv
vvp "$SIM/sim.vvp"

python - "$SIM/${LAYER}.ydut.mem" "$TB/${LAYER}.y.mem" <<'PY'
import sys
def load(p):
    out = []
    for ln in open(p):
        ln = ln.strip()
        if not ln or ln.startswith("//") or ln.startswith("@"):
            continue
        v = int(ln.split()[0], 16)
        out.append(v - (1 << 32) if v >= (1 << 31) else v)
    return out
dut = load(sys.argv[1]); gold = load(sys.argv[2])
n = min(len(dut), len(gold))
maxerr = max((abs(dut[i] - gold[i]) for i in range(n)), default=-1)
ok = dut[:n] == gold[:n] and len(dut) == len(gold)
print(f"PYVERDICT bitexact={ok} maxabserr={maxerr} n={n}")
PY
