"""Sim driver for the banked GEMV: gen vectors -> iverilog -> vvp -> bit-exact check.

    python -m fabric.stage3.run_banked --M 64 --K 128 --lanes 16
    python -m fabric.stage3.run_banked --M 40 --K 96  --lanes 16   # non-divisible M

Mirrors the project's verification discipline: the testbench dumps y.out and a
Python file-based compare prints one sentinel line (BANKED_VERDICT ...).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import pack_banked

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "gemv_banked.sv")
TB = os.path.join(HERE, "tb", "tb_gemv_banked.sv")


def run(M: int, K: int, lanes: int, mmax: int, kmax: int, seed: int, sim_dir: str) -> bool:
    os.makedirs(sim_dir, exist_ok=True)
    pack_banked.write_case(sim_dir, M, K, lanes, seed)

    vvp = os.path.join(sim_dir, "sim.vvp")
    compile_cmd = [
        "iverilog", "-g2012", "-o", vvp,
        f"-DLANES={lanes}", f"-DMMAX={mmax}", f"-DKMAX={kmax}",
        f"-DMVAL={M}", f"-DKVAL={K}",
        TB, RTL,
    ]
    cp = subprocess.run(compile_cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL")
        print(cp.stdout); print(cp.stderr)
        return False

    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    sys.stdout.write(rp.stdout)
    if rp.returncode != 0 or "TB_DONE" not in rp.stdout:
        print("VVP_RUN_FAIL"); print(rp.stderr)
        return False

    return pack_banked.check(sim_dir)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.stage3.run_banked")
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--lanes", type=int, default=16)
    p.add_argument("--mmax", type=int, default=1024)
    p.add_argument("--kmax", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dir", default=os.path.join("C:\\kevbuild", "stage3_sim"))
    a = p.parse_args(argv)
    ok = run(a.M, a.K, a.lanes, a.mmax, a.kmax, a.seed, a.dir)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
