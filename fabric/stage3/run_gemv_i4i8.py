"""Gate for gemv_i4i8: bit-exact INT32 accumulators vs numpy (doc 9 §7).

    python -m fabric.stage3.run_gemv_i4i8 --steps 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

from fabric.stage3._simdir import kevbuild

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROWS, D_IN = 1160, 256


def tool(name):
    t = shutil.which(name) or os.path.expanduser(f"~/.local/bin/{name}")
    if not os.path.exists(t):
        sys.exit(f"missing tool: {name}")
    return t


def pack_w4(w):
    """(ROWS, D_IN) int in [-8,7] -> 32-bit words, 8 nibbles LSB-first."""
    n = (w & 0xF).astype(np.uint32).reshape(ROWS, D_IN // 8, 8)
    words = np.zeros((ROWS, D_IN // 8), dtype=np.uint32)
    for k in range(8):
        words |= n[:, :, k] << (4 * k)
    return words


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_gemv_i4i8")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)

    sim = args.dir or kevbuild("stage3_gemv48")
    os.makedirs(sim, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    T = args.steps

    w = rng.integers(-8, 8, (ROWS, D_IN)).astype(np.int64)
    xs = rng.integers(-128, 128, (T, D_IN)).astype(np.int64)
    ref = np.stack([w @ xs[t] for t in range(T)])

    with open(f"{sim}/gv_w.mem", "w") as f:
        for v in pack_w4(w).ravel():
            f.write(f"{int(v):08x}\n")
    with open(f"{sim}/gv_x.mem", "w") as f:
        for v in xs.ravel():
            f.write(f"{int(v) & 0xFF:02x}\n")
    with open(f"{sim}/gv_cfg.mem", "w") as f:
        f.write(f"{T:08x}\n")

    src = [f"{REPO}/fabric/stage3/rtl/gemv_i4i8.sv",
           f"{REPO}/fabric/stage3/tb/tb_gemv_i4i8.sv"]
    subprocess.run([tool("iverilog"), "-g2012", "-o", "gv.vvp"] + src,
                   cwd=sim, check=True)
    out = subprocess.run([tool("vvp"), "gv.vvp"], cwd=sim, check=True,
                         capture_output=True, text=True)
    if "TB_GV_DONE" not in out.stdout:
        sys.exit(f"sim incomplete:\n{out.stdout[-400:]}")

    got = np.array([int(l, 16) for l in open(f"{sim}/gv_acc.out")], dtype=np.int64)
    got = np.where(got >= 1 << 31, got - (1 << 32), got).reshape(T, ROWS)
    mism = int((got != ref).sum())
    print(f"GEMV_I4I8_VERDICT: {'BIT-EXACT' if mism == 0 else 'FAIL'} "
          f"({T} calls x {ROWS} rows, {mism} mismatches)  [{sim}]")
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
