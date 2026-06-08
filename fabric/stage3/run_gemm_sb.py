"""Split-brain GEMM gate: two independent cohorts (gemm_split_brain) sharing one
TDP weight image. Each cohort's every stream must be bit-exact vs exact-integer
numpy, with the two cohorts on DIFFERENT shapes / weight bases and started a few
cycles apart (the desync that proves the two URAM read ports are independent).

    python -m fabric.stage3.run_gemm_sb --dir C:/kevbuild/stage3_gemm_sb
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

from .pack_banked import pack_transposed

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
TB = os.path.join(HERE, "tb", "tb_gemm_sb.sv")
P = 8


def write_xrows(path, xs, n, K):
    rows = []
    for s in range(n):
        for r in range(K // P):
            v = 0
            for l in range(P):
                v |= (int(xs[s, r * P + l]) & 0xFF) << (l * 8)
            rows.append(v)
    with open(path, "w") as f:
        f.write("\n".join(format(v, f"0{P*2}x") for v in rows) + "\n")


def write_case(d, lanes, n, M0, K0, WB0, M1, K1, WB1, seed, corner=False):
    rng = np.random.default_rng(seed)
    G0 = (M0 + lanes - 1) // lanes
    G1 = (M1 + lanes - 1) // lanes
    NW = max(WB1 + G1 * K1, WB0 + G0 * K0)
    # the two regions share ONE image — they must not overlap (cohort-1's load
    # would overwrite cohort-0's groups and fail the gate for the wrong reason)
    assert WB1 >= WB0 + G0 * K0 or WB0 >= WB1 + G1 * K1, (
        f"weight regions overlap: c0 [{WB0},{WB0 + G0 * K0}) vs c1 [{WB1},{WB1 + G1 * K1})")

    w0 = rng.integers(-8, 8, size=(M0, K0)).astype(np.int8)
    w1 = rng.integers(-8, 8, size=(M1, K1)).astype(np.int8)
    x0 = rng.integers(-128, 128, size=(n, K0)).astype(np.int8)
    x1 = rng.integers(-128, 128, size=(n, K1)).astype(np.int8)
    if corner:
        w0[0, :] = -8; x0[0, :] = -128; w0[1, :] = 7; x0[1, :] = 127
        w1[0, :] = -8; x1[0, :] = -128; w1[1, :] = 7; x1[1, :] = 127

    y0 = w0.astype(np.int64) @ x0.astype(np.int64).T          # (M0, n)
    y1 = w1.astype(np.int64) @ x1.astype(np.int64).T          # (M1, n)

    words = [0] * NW
    for off, w, K in ((WB0, w0, K0), (WB1, w1, K1)):
        for i, v in enumerate(pack_transposed(w, lanes)):
            words[off + i] = v
    with open(os.path.join(d, "w.mem"), "w") as f:
        f.write("\n".join(format(v, f"0{lanes}x") for v in words) + "\n")
    write_xrows(os.path.join(d, "x0.mem"), x0, n, K0)
    write_xrows(os.path.join(d, "x1.mem"), x1, n, K1)
    return y0, y1


def run_cfg(d, lanes, n, nd, M0, K0, WB0, M1, K1, WB1, skew, seed, corner, dpump=False):
    os.makedirs(d, exist_ok=True)
    y0, y1 = write_case(d, lanes, n, M0, K0, WB0, M1, K1, WB1, seed, corner)
    vvp = os.path.join(d, "sim.vvp")
    defs = [f"-DLANES={lanes}", f"-DNSTR={n}", f"-DNDSP={nd}",
            f"-DM0={M0}", f"-DK0={K0}", f"-DWB0={WB0}",
            f"-DM1={M1}", f"-DK1={K1}", f"-DWB1={WB1}", f"-DSKEW={skew}"]
    if dpump:
        defs.append("-DDPUMP")
    cc = subprocess.run(["iverilog", "-g2012", "-o", vvp, *defs, TB,
                         os.path.join(RTL, "gemm_split_brain.sv"),
                         os.path.join(RTL, "gemm_cohort_vec.sv"),
                         os.path.join(RTL, "weight_bank_tdp.sv"),
                         os.path.join(RTL, "gemm_banked_resident_vec.sv"),
                         os.path.join(RTL, "mac_bank_dp.sv")],
                        capture_output=True, text=True)
    if cc.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cc.stdout); print(cc.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=d, capture_output=True, text=True)
    if rp.returncode != 0 or "TB_DONE" not in rp.stdout:
        print("VVP_RUN_FAIL"); print(rp.stdout[-2000:]); print(rp.stderr[-1000:]); return False

    bad = 0
    for coh, (ys, M) in enumerate(((y0, M0), (y1, M1))):
        for s in range(n):
            with open(os.path.join(d, f"y{coh}_s{s}.out")) as f:
                got = np.array([np.int32(np.uint32(int(l, 16))) for l in f if l.strip()],
                               dtype=np.int64)
            bad += int(np.sum(got[:M] != ys[:, s]))
    tag = "corner" if corner else f"seed{seed}"
    dp = "DP " if dpump else ""
    print(f"GEMM_SB {dp}N={n} ND={nd} c0=({M0},{K0})@{WB0} c1=({M1},{K1})@{WB1} "
          f"skew={skew} {tag}: mismatches={bad}/{(M0+M1)*n} "
          f"{'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.stage3.run_gemm_sb")
    p.add_argument("--lanes", type=int, default=128)
    p.add_argument("--dir", default=os.path.join("C:\\kevbuild", "stage3_gemm_sb"))
    p.add_argument("--lut-only", action="store_true",
                   help="DOUBLE-PUMP Stage 1a: gate the double-pumped cohort path "
                        "(DP=1) on the ND=0 cases only (DSP not yet double-pumped).")
    a = p.parse_args(argv)

    # (M0,K0,WB0, M1,K1,WB1, skew, seed, corner)
    cases = [
        (256, 256, 0,   256, 256, 512,  7,  1, False),  # same shape, diff base, skewed
        (768, 256, 0,   256, 256, 1536, 3,  2, False),  # qkv vs proj, big skew gap
        (256, 1024, 0,  1024, 256, 2048, 19, 3, False), # mlp2 vs mlp1, K=1024 (c0 G0*K0=2048)
        (256, 1024, 0,  256, 1024, 2048, 0, 4, True),   # |acc|=2^20 corner, no skew
        (1024, 256, 0,  768, 256, 2048, 11, 5, False),  # mlp1 vs qkv (c0 G0*K0=2048)
    ]
    if a.lut_only:
        # DP=1 cohort beachhead — the same ND=0 cases (shapes, bases, skew, the
        # 2^20 corner), double-pumped. The desync (skew) still proves the two
        # read ports are independent while each runs 2 K-steps/clk.
        ok = True
        for (M0, K0, WB0, M1, K1, WB1, skew, seed, corner) in cases:
            d = os.path.join(a.dir, f"dp_c{M0}x{K0}_{M1}x{K1}_s{seed}")
            ok &= run_cfg(d, a.lanes, 8, 0, M0, K0, WB0, M1, K1, WB1, skew, seed,
                          corner, dpump=True)
        print("GEMM_SB_DP_VERDICT " + ("ALL_BITEXACT" if ok else "FAIL"))
        return 0 if ok else 1

    ok = True
    for (M0, K0, WB0, M1, K1, WB1, skew, seed, corner) in cases:
        d = os.path.join(a.dir, f"c{M0}x{K0}_{M1}x{K1}_s{seed}")
        ok &= run_cfg(d, a.lanes, 8, 0, M0, K0, WB0, M1, K1, WB1, skew, seed, corner)
    # one ND=8-style all-DSP-ish sanity at N=8 (cohort streams all DSP)
    d = os.path.join(a.dir, "dsp_n8")
    ok &= run_cfg(d, a.lanes, 8, 8, 256, 256, 0, 256, 256, 512, 5, 6, False)
    print("GEMM_SB_VERDICT " + ("ALL_BITEXACT" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
