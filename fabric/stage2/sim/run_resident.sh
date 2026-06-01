#!/usr/bin/env bash
set -euo pipefail
LAYER="${1:-blocks_0_proj}"; OFF="${2:-0}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$ROOT"
EXP="fabric/export"; TB="$EXP/tb"; SIM="fabric/stage2/sim"
read -r M K < <(python -c "import json;d=json.load(open(r'$TB/${LAYER}.dims.json'));print(d['M'],d['K'])")
cat > "$SIM/config.svh" <<CFG
\`define CFG_M $M
\`define CFG_K $K
\`define CFG_OFF $OFF
\`define CFG_WFILE "$EXP/${LAYER}.w.mem"
\`define CFG_XFILE "$TB/${LAYER}.x.mem"
\`define CFG_YDUT  "$SIM/${LAYER}.ydut.mem"
CFG
echo "layer=$LAYER M=$M K=$K OFF=$OFF (gemv_resident, wide URAM)"
iverilog -g2012 -I "$SIM" -o "$SIM/sim.vvp" \
    fabric/stage2/rtl/gemv_resident.sv fabric/stage2/tb/tb_gemv_resident.sv
vvp "$SIM/sim.vvp"
python - "$SIM/${LAYER}.ydut.mem" "$TB/${LAYER}.y.mem" <<'PY'
import sys
def load(p):
    out=[]
    for ln in open(p):
        ln=ln.strip()
        if not ln or ln.startswith("//") or ln.startswith("@"): continue
        v=int(ln.split()[0],16); out.append(v-(1<<32) if v>=(1<<31) else v)
    return out
d=load(sys.argv[1]); g=load(sys.argv[2]); n=min(len(d),len(g))
print(f"PYVERDICT bitexact={d[:n]==g[:n] and len(d)==len(g)} maxabserr={max((abs(d[i]-g[i]) for i in range(n)),default=-1)} n={n}")
PY
