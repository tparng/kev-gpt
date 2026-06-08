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
MAC_DP = os.path.join(HERE, "rtl", "mac_bank_dp.sv")       # double-pumped LUT leaf (DP=1)
MAC_DSP_DP = os.path.join(HERE, "rtl", "mac_bank_dsp_dp.sv")  # double-pumped DSP leaf (DP=1, ND>0)
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


def run_cfg(d, M, K, lanes, n, nd, seed, corner, dpump=False):
    os.makedirs(d, exist_ok=True)
    ys = write_case(d, M, K, lanes, n, seed, corner)
    vvp = os.path.join(d, "sim.vvp")
    defs = [f"-DLANES={lanes}", f"-DNSTR={n}", f"-DNDSP={nd}",
            f"-DMVAL={M}", f"-DKVAL={K}"]
    if dpump:
        defs.append("-DDPUMP")
    cc = subprocess.run(["iverilog", "-g2012", "-o", vvp] + defs +
                        [TB, RTL, MAC_DP, MAC_DSP_DP],
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
    dp = "DP " if dpump else ""
    print(f"GEMM16 {dp}N={n} ND={nd} M={M} K={K} {tag}: mismatches={bad}/{M*n} "
          f"{'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.stage3.run_gemm16")
    p.add_argument("--lanes", type=int, default=128)
    p.add_argument("--dir", default=os.path.join("C:\\kevbuild", "stage3_gemm16"))
    p.add_argument("--lut-only", action="store_true",
                   help="DOUBLE-PUMP Stage 1a: gate the double-pumped LUT path "
                        "(DP=1) on ND=0 cases only.")
    p.add_argument("--dp-full", action="store_true",
                   help="DOUBLE-PUMP: gate DP=1 over the FULL case list incl the "
                        "DSP-packed leaves (mac_bank_dsp_dp), ND=8/16 + 2^20 corner.")
    a = p.parse_args(argv)

    if a.lut_only:
        # double-pump (DP=1) LUT-only beachhead. ND=0 => the g_dsp branch makes
        # zero instances, so these are complete. K is always a multiple of P=8
        # (even) here, so the odd-K phase-1 mask is exercised only defensively.
        cases = [
            # (M, K, N, ND, seed, corner)
            (256, 256, 8, 0, 1, False),       # shipping N=8 all-LUT, double-pumped
            (1024, 256, 16, 0, 3, False),     # N=16 all-LUT, mlp1 shape
            (256, 1024, 16, 0, 4, False),     # K=1024 reduction, all-LUT
            (256, 1024, 8, 0, 5, True),       # |acc| = 2^20 range-proof corner
        ]
        ok = True
        for M, K, n, nd, seed, corner in cases:
            d = os.path.join(a.dir, f"dp_n{n}_nd{nd}_m{M}_k{K}_{'c' if corner else seed}")
            ok &= run_cfg(d, M, K, a.lanes, n, nd, seed, corner, dpump=True)
        print("GEMM16_DP_VERDICT " + ("ALL_BITEXACT" if ok else "FAIL"))
        return 0 if ok else 1

    cases = [
        # (M, K, N, ND, seed, corner)
        (256, 256, 8, 0, 1, False),       # shipping N=8 all-LUT regression
        (256, 256, 16, 8, 2, False),      # qkv-ish shape, mixed leaves
        (1024, 256, 16, 8, 3, False),     # mlp1 shape
        (256, 1024, 16, 8, 4, False),     # mlp2 shape, K=1024 reduction
        (256, 1024, 16, 8, 5, True),      # |acc| = 2^20 range-proof corner
        (256, 256, 16, 16, 6, False),     # all-DSP sanity
    ]
    if a.dp_full:
        # DP=1 over the mixed/all-DSP cases — gates mac_bank_dsp_dp (the packed-pair
        # leaf at 2 K-steps/clk) incl the 2^20 corner and the all-DSP ND=16 case.
        ok = True
        for M, K, n, nd, seed, corner in cases:
            d = os.path.join(a.dir, f"dpf_n{n}_nd{nd}_m{M}_k{K}_{'c' if corner else seed}")
            ok &= run_cfg(d, M, K, a.lanes, n, nd, seed, corner, dpump=True)
        print("GEMM16_DPFULL_VERDICT " + ("ALL_BITEXACT" if ok else "FAIL"))
        return 0 if ok else 1

    ok = True
    for M, K, n, nd, seed, corner in cases:
        d = os.path.join(a.dir, f"n{n}_nd{nd}_m{M}_k{K}_{'c' if corner else seed}")
        ok &= run_cfg(d, M, K, a.lanes, n, nd, seed, corner)
    print("GEMM16_VERDICT " + ("ALL_BITEXACT" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
