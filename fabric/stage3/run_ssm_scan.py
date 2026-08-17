"""Gate harness for the Mamba-2 SSM scan core (doc 9 §7 rung 2).

Writes T steps of stimulus, runs iverilog on ssm_scan.sv + tb_ssm_scan.sv,
and compares y bit-exactly against model/mamba2_fixed_scan.py. This is the
first mamba rung of the stage-3 pattern: green here before any synth.

    python -m fabric.stage3.run_ssm_scan               # T=256 random soak
    python -m fabric.stage3.run_ssm_scan --steps 2048  # longer soak
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

from fabric.stage3._simdir import kevbuild
from model.mamba2_fixed_scan import (Q_A, Q_ACT, Q_STATE, a_from_dtA, quant,
                                     scan_step_fixed)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = N = 64


def tool(name):
    t = shutil.which(name) or os.path.expanduser(f"~/.local/bin/{name}")
    if not os.path.exists(t):
        sys.exit(f"missing tool: {name}")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_ssm_scan")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)

    sim = args.dir or kevbuild("stage3_ssm_scan")
    os.makedirs(sim, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    T = args.steps

    # stimulus in the measured envelope (see mamba2_fixed_scan.soak)
    negA = rng.uniform(1.0, 16.0)
    dts = np.exp(rng.uniform(np.log(1e-3), np.log(0.1), T))
    xs = np.clip(np.round(rng.normal(0, 38, (T, P))), -127, 127).astype(np.int64)
    Bs = np.clip(np.round(rng.normal(0, 38, (T, N))), -127, 127).astype(np.int64)
    Cs = np.clip(np.round(rng.normal(0, 38, (T, N))), -127, 127).astype(np.int64)

    a_qs = np.array([a_from_dtA(dt * negA) for dt in dts], dtype=np.int64)
    dtxs = np.stack([quant(dts[t] * xs[t] / (1 << Q_ACT), Q_STATE)
                     for t in range(T)])

    def hexdump(path, vals, width):
        mask = (1 << width) - 1
        with open(path, "w") as f:
            for v in np.asarray(vals).ravel():
                f.write(f"{int(v) & mask:0{width // 4}x}\n")

    hexdump(f"{sim}/ssm_a.mem", a_qs, 16)
    hexdump(f"{sim}/ssm_dtx.mem", dtxs, 16)
    hexdump(f"{sim}/ssm_b.mem", Bs, 8)
    hexdump(f"{sim}/ssm_c.mem", Cs, 8)
    with open(f"{sim}/ssm_cfg.mem", "w") as f:
        f.write(f"{T:08x}\n")

    # reference (bit-true python)
    h = np.zeros((P, N), dtype=np.int64)
    ref = np.zeros((T, P), dtype=np.int64)
    for t in range(T):
        ref[t] = scan_step_fixed(h, xs[t], Bs[t], Cs[t], float(dts[t]), negA)

    # compile + run
    src = [f"{REPO}/fabric/stage3/rtl/ssm_scan.sv",
           f"{REPO}/fabric/stage3/tb/tb_ssm_scan.sv"]
    subprocess.run([tool("iverilog"), "-g2012", "-o", "ssm_scan.vvp"] + src,
                   cwd=sim, check=True)
    out = subprocess.run([tool("vvp"), "ssm_scan.vvp"], cwd=sim, check=True,
                         capture_output=True, text=True)
    if "TB_SSM_DONE" not in out.stdout:
        sys.exit(f"sim did not complete:\n{out.stdout[-500:]}")

    got = np.array([int(l, 16) for l in open(f"{sim}/ssm_y.out")], dtype=np.int64)
    got = np.where(got >= 1 << 15, got - (1 << 16), got).reshape(T, P)

    mism = int((got != ref).sum())
    first = np.argwhere(got != ref)
    where = f" first@step{first[0][0]},p{first[0][1]}" if len(first) else ""
    print(f"SSM_SCAN_VERDICT: {'BIT-EXACT' if mism == 0 else 'FAIL'} "
          f"({T} steps x {P} ch, {mism} mismatches{where})  [{sim}]")
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
