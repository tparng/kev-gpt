"""vec_dequant RTL gate: P-lane parallel dequant, bit-true vs the scalar reference.

    python -m fabric.stage3.run_vec_dequant

The P-wide module rtl/vec_dequant.sv must reproduce, lane-for-lane, EXACTLY the
per-element dequant the sequencer does (S_QKV_DEQ / S_PROJ_DEQ / S_FC_DEQ / S_MP_DEQ /
S_HEAD_DEQ in rtl/sequencer_fast.sv) and the golden seq_ref._dequant_to_q:

    dq_prod = gemvy(int32) * mant(int24, signed)      # signed product
    dq_shv  = exp(int8, signed) + FRAC
    dq_val  = dq_prod << dq_shv            if dq_shv >= 0
              rsh_round(dq_prod, -dq_shv)  otherwise
    out     = dq_val & 0xFFFFFFFF          # dq_val[31:0]

rsh_round is round-half-away-from-zero (== seq_ref.rsh_round, == sequencer_fast.rsh_round).
The caller passes FRAC = 16 (qkv->kv Q.16), 25 (residual Q6.25), or 12 (gelu Q4.12).
The gate generates realistic random (gemvy, mant, exp) vectors, runs them P-wide through
the module, and requires mismatches=0 for FRAC in {16, 25, 12} at P=8.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE_MANT_BITS = 24


def rsh_round(v, s):
    """Arithmetic right shift by s with round-half-away-from-zero; s<=0 -> left shift.
    Bit-identical to seq_ref.rsh_round / rtl/sequencer.rsh_round / vec_dequant.rsh_round."""
    v = int(v)
    if s <= 0:
        return v << (-s)
    half = 1 << (s - 1)
    if v >= 0:
        return (v + half) >> s
    return -(((-v) + half) >> s)


def dequant_scalar(gemvy, mant, exp, frac):
    """The EXACT scalar dequant contract (matches seq_ref._dequant_to_q low-32-bits).
    Returns dq_val[31:0] as an unsigned 32-bit pattern (two's complement)."""
    prod = int(gemvy) * int(mant)
    shv = int(exp) + int(frac)
    if shv >= 0:
        dq_val = prod << shv
    else:
        dq_val = rsh_round(prod, -shv)
    return dq_val & 0xFFFFFFFF


def _gen_vectors(n, seed=0):
    """Realistic random (gemvy INT32, mant ~+-2^23 signed, exp small signed) vectors.

    Ranges mirror the model's pinned dequant:
      - mant : 24-bit folded scale mantissa, magnitude ~2^23 (per quantize_scale_24).
      - exp  : signed, small in magnitude (log2 of a tiny per-channel scale -> negative,
               roughly -40..0 in the real model; widen a touch to exercise both branches).
      - gemvy: full INT32 GEMV accumulator; realistic accumulators are far smaller, so we
               mix a plausible bulk (|y| < few*10^5) with full-INT32 edge stressors.
    """
    rng = np.random.default_rng(seed)
    gemvy = np.empty(n, dtype=object)
    mant = np.empty(n, dtype=object)
    exp = np.empty(n, dtype=object)

    # plausible 24-bit mantissas: in [2^23, 2^24) magnitude (as quantize_scale_24 yields),
    # both signs; clamp into signed 24-bit range.
    m_mag = rng.integers(1 << 23, 1 << 24, size=n)
    m_sgn = rng.choice([-1, 1], size=n)
    m = (m_mag * m_sgn).astype(np.int64)
    m = np.clip(m, -(1 << 23), (1 << 23) - 1)

    # exp: mostly negative (real model), span both shift branches around all three FRACs.
    e = rng.integers(-45, 16, size=n).astype(np.int64)

    # gemvy: realistic accumulators + full-range edge stressors.
    g = rng.integers(-200000, 200001, size=n).astype(np.int64)

    for i in range(n):
        mant[i] = int(m[i]); exp[i] = int(e[i]); gemvy[i] = int(g[i])

    # deterministic edge cases in the first slots (exercise saturation/overflow of <<<,
    # both rounding branches, sign combinations, and full INT32 gemvy).
    edges = [
        (0,            0,                   0),
        (1,            1,                   0),
        (-1,           1,                   0),
        (2**31 - 1,    (1 << 23) - 1,       10),
        (-(2**31),     -(1 << 23),          10),
        (2**31 - 1,    -(1 << 23),         -30),
        (-(2**31),     (1 << 23) - 1,      -30),
        (123456,       1 << 23,             0),
        (-123456,      1 << 23,             0),
        (3,            5,                  -1),      # round-half cases
        (-3,           5,                  -1),
        (1,            3,                  -2),
        (-1,           3,                  -2),
        (7,            -7,                 -3),
        (255,          (1 << 23) - 1,      15),      # large left shift
        (1,            1,                  -40),     # shift all the way out
    ]
    for j, (gg, mm, ee) in enumerate(edges):
        if j < n:
            gemvy[j] = int(gg); mant[j] = int(mm); exp[j] = int(ee)
    return gemvy, mant, exp


def _w(path, vals, nib, mask):
    with open(path, "w") as fh:
        fh.write("\n".join(f"{int(v) & mask:0{nib}x}" for v in vals) + "\n")


def run_one(sim_dir, frac, P=8, N=2048, seed=0):
    os.makedirs(sim_dir, exist_ok=True)
    gemvy, mant, exp = _gen_vectors(N, seed=seed + frac)

    gold = [dequant_scalar(gemvy[i], mant[i], exp[i], frac) for i in range(N)]

    _w(os.path.join(sim_dir, "gemvy.mem"), gemvy, 8, 0xFFFFFFFF)
    _w(os.path.join(sim_dir, "mant.mem"),  mant,  6, 0xFFFFFF)
    _w(os.path.join(sim_dir, "exp.mem"),   exp,   2, 0xFF)

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(
        ["iverilog", "-g2012", "-o", vvp,
         f"-DNELEM={N}", f"-DP={P}", f"-DFRAC={frac}",
         os.path.join(HERE, "tb", "tb_vec_dequant.sv"),
         os.path.join(HERE, "rtl", "vec_dequant.sv")],
        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL")
        print(cp.stdout); print(cp.stderr)
        return False, 0, N
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL")
        print(rp.stdout); print(rp.stderr)
        return False, 0, N

    def _hx(s):
        if "x" in s or "z" in s:
            return None
        return int(s, 16) & 0xFFFFFFFF

    with open(os.path.join(sim_dir, "y.out")) as fh:
        got = [_hx(s.strip()) for s in fh if s.strip()]

    # y.out has NCYC*P padded outputs; the first N are the real elements (TB streams in order).
    mism = 0
    n = min(len(got), N)
    for i in range(n):
        if got[i] != gold[i]:
            mism += 1
    if len(got) < N:
        mism += N - len(got)
    return (mism == 0), mism, N


def main():
    base = os.path.join("C:\\kevbuild", "stage3_vec_dequant")
    all_ok = True
    for frac in (16, 25, 12):
        sim_dir = os.path.join(base, f"frac{frac}")
        ok, mism, N = run_one(sim_dir, frac, P=8)
        all_ok = all_ok and ok
        print(f"VEC_DEQUANT_VERDICT bitexact={ok} mismatches={mism}/{N} frac={frac} P=8")
    print("VEC_DEQUANT_" + ("PASS" if all_ok else "FAIL"))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
