"""Standalone gate for the N-stream batch GEMM core with mixed LUT + DSP leaves.

Every stream's outputs must be bit-exact vs exact-integer numpy across shapes,
configs (N=8/ND=0 — the shipping single-pass build — and N=16/ND=8), and the
accumulator range-proof corner: ABITS=24 holds |acc| <= 8*128*1024 = 2^20, so
the harness forces W rows of -8 against x = -128 at K=1024 to sit ON the corner.

    python -m fabric.stage3.run_gemm16 --dir C:/kevbuild/stage3_gemm16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

from .pack_banked import pack_transposed

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "gemm_banked_resident_vec.sv")
TB = os.path.join(HERE, "tb", "tb_gemm16.sv")
P = 8


def write_case(d, M, K, lanes, n, seed, corner=False):
    rng = np.random.default_rng(seed)
    w = rng.integers(-8, 8, size=(M, K)).astype(np.int8)
    xs = rng.integers(-128, 128, size=(n, K)).astype(np.int8)
    if corner:
        w[0, :] = -8                       # |acc| = 2^20 exactly on stream 0
        xs[0, :] = -128
        w[1, :] = 7                        # max positive weight row
        xs[1, :] = 127
    ys = w.astype(np.int64) @ xs.astype(np.int64).T          # (M, n) exact

    words = pack_transposed(w, lanes)
    with open(os.path.join(d, "w.mem"), "w") as f:
        f.write("\n".join(format(v, f"0{lanes}x") for v in words) + "\n")
    rows = []
    for s in range(n):
        for r in range(K // P):
            v = 0
            for l in range(P):
                v |= (int(xs[s, r * P + l]) & 0xFF) << (l * 8)
            rows.append(v)
    with open(os.path.join(d, "xrows.mem"), "w") as f:
        f.write("\n".join(format(v, f"0{P*2}x") for v in rows) + "\n")
    return ys


def run_cfg(d, M, K, lanes, n, nd, seed, corner):
    os.makedirs(d, exist_ok=True)
    ys = write_case(d, M, K, lanes, n, seed, corner)
    vvp = os.path.join(d, "sim.vvp")
    cc = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                         f"-DLANES={lanes}", f"-DNSTR={n}", f"-DNDSP={nd}",
                         f"-DMVAL={M}", f"-DKVAL={K}", TB, RTL],
                        capture_output=True, text=True)
    if cc.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cc.stdout); print(cc.stderr)
        return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=d, capture_output=True, text=True)
    if rp.returncode != 0 or "TB_DONE" not in rp.stdout:
        print("VVP_RUN_FAIL"); print(rp.stdout); print(rp.stderr)
        return False

    bad = 0
    for s in range(n):
        with open(os.path.join(d, f"y_s{s}.out")) as f:
            got = np.array([np.int32(np.uint32(int(l, 16))) for l in f if l.strip()],
                           dtype=np.int64)
        bad += int(np.sum(got[:M] != ys[:, s]))
    tag = "corner" if corner else f"seed{seed}"
    print(f"GEMM16 N={n} ND={nd} M={M} K={K} {tag}: mismatches={bad}/{M*n} "
          f"{'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.stage3.run_gemm16")
    p.add_argument("--lanes", type=int, default=128)
    p.add_argument("--dir", default=os.path.join("C:\\kevbuild", "stage3_gemm16"))
    a = p.parse_args(argv)

    cases = [
        # (M, K, N, ND, seed, corner)
        (256, 256, 8, 0, 1, False),       # shipping N=8 all-LUT regression
        (256, 256, 16, 8, 2, False),      # qkv-ish shape, mixed leaves
        (1024, 256, 16, 8, 3, False),     # mlp1 shape
        (256, 1024, 16, 8, 4, False),     # mlp2 shape, K=1024 reduction
        (256, 1024, 16, 8, 5, True),      # |acc| = 2^20 range-proof corner
        (256, 256, 16, 16, 6, False),     # all-DSP sanity
    ]
    ok = True
    for M, K, n, nd, seed, corner in cases:
        d = os.path.join(a.dir, f"n{n}_nd{nd}_m{M}_k{K}_{'c' if corner else seed}")
        ok &= run_cfg(d, M, K, a.lanes, n, nd, seed, corner)
    print("GEMM16_VERDICT " + ("ALL_BITEXACT" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
