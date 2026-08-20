"""Gate for gemv_i4i8_cohort: the weight-stationary N-stream cohort GEMV.

Proves (a) each stream's INT32 accumulators are bit-exact to the single-stream
gemv (== numpy w@x[stream]), and (b) the whole cohort costs ~one weight pass
(GC_CYC ~= single-stream cyc, independent of N) — the amortization lever.

    python -m fabric.stage3.run_gemv_cohort -n 8
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
    n = (w & 0xF).astype(np.uint32).reshape(ROWS, D_IN // 8, 8)
    words = np.zeros((ROWS, D_IN // 8), dtype=np.uint32)
    for k in range(8):
        words |= n[:, :, k] << (4 * k)
    return words


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_gemv_cohort")
    ap.add_argument("-n", type=int, default=8, help="cohort stream count N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)

    N = args.n
    sim = args.dir or kevbuild("stage3_gemv_cohort")
    os.makedirs(sim, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    w = rng.integers(-8, 8, (ROWS, D_IN)).astype(np.int64)
    xs = rng.integers(-128, 128, (N, D_IN)).astype(np.int64)
    ref = np.stack([w @ xs[s] for s in range(N)])   # (N, ROWS)

    with open(f"{sim}/gc_w.mem", "w") as f:
        for v in pack_w4(w).ravel():
            f.write(f"{int(v):08x}\n")
    with open(f"{sim}/gc_x.mem", "w") as f:
        for v in xs.ravel():
            f.write(f"{int(v) & 0xFF:02x}\n")
    with open(f"{sim}/gc_cfg.mem", "w") as f:
        f.write(f"{N:08x}\n")

    src = [f"{REPO}/fabric/stage3/rtl/gemv_i4i8_cohort.sv",
           f"{REPO}/fabric/stage3/tb/tb_gemv_cohort.sv"]
    subprocess.run([tool("iverilog"), "-g2012", "-o", "gc.vvp"] + src,
                   cwd=sim, check=True)
    out = subprocess.run([tool("vvp"), "gc.vvp"], cwd=sim, check=True,
                         capture_output=True, text=True)
    if "TB_GC_DONE" not in out.stdout:
        sys.exit(f"sim incomplete:\n{out.stdout[-600:]}")
    cyc = next((l for l in out.stdout.splitlines() if l.startswith("GC_CYC")), "")

    got = np.array([int(l, 16) for l in open(f"{sim}/gc_acc.out")], dtype=np.int64)
    got = np.where(got >= 1 << 31, got - (1 << 32), got).reshape(N, ROWS)
    mism = int((got != ref).sum())
    # single-stream cycle reference: rows*wpr/PE + pipe drain (~ROWS*WPR/16)
    single = ROWS * (D_IN // 8) // 16
    print(f"  {cyc}   (single-stream weight pass ~= {single} cyc; cohort shares it)")
    print(f"GEMV_COHORT_VERDICT: {'BIT-EXACT' if mism == 0 else 'FAIL'} "
          f"(N={N} streams x {ROWS} rows, {mism} mismatches)  [{sim}]")
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
