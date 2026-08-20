"""Gate harness for rmsnorm_gated (Mamba-2 gated RMSNorm, cosine gate).

Reference = model.mamba2_fixed.rmsnorm_fixed at fabric resolution: the float
path g = y * silu_lut(z), rms = 1/sqrt(mean(g^2)+1e-5), out = g * rms * gamma,
per-token cosine > 0.9999 (house rule for LUT/transcendental phases). The RTL
is ALSO checked bit-exact against an integer replica of its own datapath (the
conv_silu-style floor-indexed SiLU + the layernorm.sv rsqrt minus the mean).

Two contract adaptations to the float reference (documented, both are the
fixed-point contract, not RTL slack):
  * silu_lut is evaluated at the floor-quantized LUT grid point the RTL
    actually indexes (gated z is Q6.9 now: (z+2048)>>4; silu_lut floors to the
    same grid) — a half-bin contract mismatch that otherwise caps per-token
    cosine at ~0.9998 for off-grid z.
  * the gated product and the output are clipped to the INT16 Q3.12
    saturating range (+-8), which the RTL contract pins.

Both modes are gated: gated=1 (g = y*silu(z)) and gated=0 (g = y; the
engine's pre-norm / final norm have no z-gate).

    python -m fabric.stage3.run_rmsnorm_gated --tokens-per-mode 8
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

from fabric.stage3._simdir import kevbuild
from fabric.stage3.run_layernorm import rsqrt_int, seed_table
from model.mamba2_fixed import silu_lut

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = 512
LOG2D = 9
QF = 12
EPS_A = 671                      # round(1e-5 * 2^26)
OUT_SH = 26 + 14                 # Yr Q.26 * gamma Q1.14 back to Q3.12
SAT_HI = 32767 / 4096.0
SAT_LO = -8.0


def tool(name):
    t = shutil.which(name) or os.path.expanduser(f"~/.local/bin/{name}")
    if not os.path.exists(t):
        sys.exit(f"missing tool: {name}")
    return t


def silu_table():
    grid = (np.arange(256) - 128) / 32.0
    return np.round(grid / (1.0 + np.exp(-grid)) * 4096).astype(np.int64)


def ref_int_short(yq, gq):
    """Bit-true replica of the SHORT mode: 256 x Q6.9 in, ungated, Q3.12 out.
    A = sum(g^2) + eps (sum of 256 Q6.9 squares IS mean*2^26); out >>> 37."""
    g = yq[: D // 2].astype(np.int64)
    A = int((g * g).sum()) + EPS_A
    Yr = rsqrt_int(A)
    out = np.empty(D // 2, dtype=np.int64)
    rnd = 1 << (OUT_SH - 3 - 1)
    for i in range(D // 2):
        t = int(g[i]) * Yr * int(gq[i])
        out[i] = min(32767, max(-32768, (t + rnd) >> (OUT_SH - 3)))
    return out


def ref_float_short(yq, gq):
    yf, gf = yq[: D // 2] / 512.0, gq[: D // 2] / 16384.0
    rms = 1.0 / np.sqrt((yf * yf).mean() + 1e-5)
    return np.clip(yf * rms * gf, SAT_LO, SAT_HI)


GSH = 6                                                 # gated-product headroom
A_SH = LOG2D + 2 * QF - 26                               # 7


def ref_int(yq, zq, gq, lut, mode):
    """Bit-true integer replica of the RTL datapath (one token). Gated mode
    right-shifts the product by GSH (RMS scale-invariant) so |g| up to ~512
    real does not saturate Q3.12; A left-shifts and OUT_SH-=GSH to compensate.
    Gated inputs are y Q4.11 (scan out ~10) and z Q6.9 (gate ~28), silsel Q6.9
    -> gate_sh 14; ungated y Q3.12, silsel 1.0 -> gate_sh 12."""
    if mode:
        idx = np.clip((zq + 2048) >> 4, 0, 255)                # Q6.9, 4.0 = 2048
        sil = np.where(zq >= 2048, zq,
                       np.where(zq < -2048, 0, (lut[idx] + 4) >> 3))  # Q3.12->Q6.9
        gsh, out_sh = 14, OUT_SH - GSH
    else:
        sil = np.full(D, 4096, dtype=np.int64)         # 1.0 Q3.12 -> g = y exact
        gsh, out_sh = QF, OUT_SH
    g = np.clip((yq * sil + (1 << (gsh - 1))) >> gsh, -32768, 32767)
    ss = int((g * g).sum())
    A = (ss << (2 * GSH - A_SH) if mode else ss >> A_SH) + EPS_A
    Yr = rsqrt_int(A)                                   # layernorm.sv rsqrt, Q.26
    out = np.empty(D, dtype=np.int64)
    rnd = 1 << (out_sh - 1)
    for i in range(D):                                  # >int64: python ints
        t = int(g[i]) * Yr * int(gq[i])
        out[i] = min(32767, max(-32768, (t + rnd) >> out_sh))
    return out


def ref_float(yq, zq, gq, mode):
    """Float-path reference (rmsnorm_fixed recipe at contract resolution)."""
    gf = gq / 16384.0
    if mode:
        yf = yq / 2048.0                                # gated y Q4.11
        zf = zq / 512.0                                 # gated z Q6.9; silu_lut
        g = yf * silu_lut(zf)                           # floors to the same grid
    else:
        yf = yq / 4096.0                                # ungated y Q3.12
        g = yf.copy()
    # gated g now has GSH headroom (real range +-8*2^GSH); only the OUTPUT is
    # Q3.12-saturated. Clipping g at +-8 here was the scale-blind bug's twin.
    g = np.clip(g, -8.0 * (1 << GSH), 8.0 * (1 << GSH)) if mode \
        else np.clip(g, SAT_LO, SAT_HI)
    rms = 1.0 / np.sqrt((g * g).mean() + 1e-5)
    return np.clip(g * rms * gf, SAT_LO, SAT_HI)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_rmsnorm_gated")
    ap.add_argument("--tokens-per-mode", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)

    sim = args.dir or kevbuild("stage3_rmsn")
    os.makedirs(sim, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    TPM = args.tokens_per_mode
    T = 3 * TPM
    modes = np.array([1] * TPM + [0] * TPM + [2] * TPM)   # 2 = short (Q6.9)

    ys = np.zeros((T, D), dtype=np.int64)
    zs = np.zeros((T, D), dtype=np.int64)
    # gated block [0,TPM): y Q4.11 (scan out ~10), z Q6.9 (gate ~28)
    ys[:TPM] = np.clip(np.round(rng.normal(0, 1.5, (TPM, D)) * 2048), -32768, 32767)
    zs[:TPM] = np.clip(np.round(rng.normal(0, 2.0, (TPM, D)) * 512), -32768, 32767)
    # ungated block [TPM,2TPM): y Q3.12 (g = y exact)
    ys[TPM:2 * TPM] = np.clip(np.round(rng.normal(0, 1.2, (TPM, D)) * 4096), -32768, 32767)
    # SATURATING-GATE rows (gated mode): a few channels with a large product
    # y*silu(z) (real |g| ~ 60-280, z>=4 -> silu(z)=z up to ~28) — the engine's
    # actual regime, which Q3.12 clipped and no cosine gate could see. These
    # exercise the GSH headroom and the Q6.9 z / Q4.11 y widening.
    for t in range(min(3, TPM)):
        big = rng.choice(D, 5, replace=False)
        ys[t, big] = rng.choice([-1, 1], 5) * rng.integers(6, 10, 5) * 2048  # |y|~6-10
        zs[t, big] = rng.integers(8, 28, 5) * 512                           # z>=4 -> silu=z
    # short-mode rows: Q6.9 residual-view values. Cover the eps-sensitive
    # small-magnitude case (layer-0 emb ~ +-0.02 -> ~10 counts) and the wide
    # late-layer case (up to ~+-30).
    ys[2 * TPM] = np.clip(np.round(rng.normal(0, 0.02, D) * 512), -32768, 32767)
    for t in range(2 * TPM + 1, T):
        s = rng.uniform(0.05, 12.0)
        ys[t] = np.clip(np.round(rng.normal(0, s, D) * 512), -32768, 32767)
    gq = np.clip(np.round(rng.normal(1, 0.2, D) * 16384), -32768, 32767).astype(np.int64)
    lut = silu_table()

    def hexdump(path, vals, nib=4):
        mask = (1 << (4 * nib)) - 1
        with open(path, "w") as f:
            for v in np.asarray(vals).ravel():
                f.write(f"{int(v) & mask:0{nib}x}\n")

    hexdump(f"{sim}/rg_lut.mem", lut)
    hexdump(f"{sim}/rg_gamma.mem", gq)
    hexdump(f"{sim}/rg_y.mem", ys)
    hexdump(f"{sim}/rg_z.mem", zs)
    hexdump(f"{sim}/rg_mode.mem", modes, nib=1)
    hexdump(f"{sim}/rg_cfg.mem", [T], nib=8)
    hexdump(f"{sim}/seed.mem", seed_table(), nib=5)     # layernorm.sv ROM format

    src = [f"{REPO}/fabric/stage3/rtl/rmsnorm_gated.sv",
           f"{REPO}/fabric/stage3/tb/tb_rmsnorm_gated.sv"]
    subprocess.run([tool("iverilog"), "-g2012", "-o", "rg.vvp"] + src,
                   cwd=sim, check=True)
    out = subprocess.run([tool("vvp"), "rg.vvp"], cwd=sim, check=True,
                         capture_output=True, text=True)
    if "TB_RG_DONE" not in out.stdout:
        sys.exit(f"sim incomplete:\n{out.stdout[-400:]}")

    got = np.array([int(l, 16) for l in open(f"{sim}/rg_o.out")], dtype=np.int64)
    got = np.where(got >= 1 << 15, got - (1 << 16), got).reshape(T, D)

    mism = 0
    min_cos = {1: 1.0, 0: 1.0, 2: 1.0}
    max_abs = {1: 0.0, 0: 0.0, 2: 0.0}
    max_rel = 0.0
    for t in range(T):
        m = int(modes[t])
        if m == 2:
            mism += int((got[t][: D // 2] != ref_int_short(ys[t], gq)).sum())
            ref = ref_float_short(ys[t], gq)
            gf = got[t][: D // 2] / 4096.0
        else:
            mism += int((got[t] != ref_int(ys[t], zs[t], gq, lut, m)).sum())
            ref = ref_float(ys[t], zs[t], gq, m)
            gf = got[t] / 4096.0
        cos = float(gf @ ref / ((np.linalg.norm(gf) * np.linalg.norm(ref)) or 1.0))
        min_cos[m] = min(min_cos[m], cos)
        err = np.abs(gf - ref)
        max_abs[m] = max(max_abs[m], float(err.max()))
        max_rel = max(max_rel, float((err / np.maximum(np.abs(ref), 1e-2)).max()))

    # ABSOLUTE-scale gate (not just cosine): a scale/saturation error is
    # invisible to cosine but shows as large abs error. The Q3.12 output is in
    # [-8,8]; a real scale bug produces abs errors ~O(1). Bar 0.05 (~205 LSB)
    # passes LUT/rsqrt jitter, catches the gated-product saturation class.
    ABS_BAR = 0.05
    ok = (min(min_cos.values()) > 0.9999 and mism == 0
          and all(v < ABS_BAR for v in max_abs.values()))
    print(f"RMSNORM_GATED_INFO: max_abs_err gated={max_abs[1]:.6f} "
          f"ungated={max_abs[0]:.6f} short={max_abs[2]:.6f}  "
          f"max_rel_err {max_rel:.6f}  int_replica_mismatches {mism}")
    print(f"RMSNORM_GATED_VERDICT: {'PASS' if ok else 'FAIL'} "
          f"gated_cos={min_cos[1]:.8f} ungated_cos={min_cos[0]:.8f} "
          f"short_cos={min_cos[2]:.8f} max_abs={max(max_abs.values()):.5f} "
          f"({TPM}x3 tokens; bar cos>0.9999 + abs<{ABS_BAR} all modes + "
          f"bit-exact vs int replica)  [{sim}]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
