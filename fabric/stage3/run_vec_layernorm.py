"""Standalone bit-exact gate for layernorm_vec (the P-wide-I/O LayerNorm, 10k datapath).

Reuses run_layernorm's reference (_ln_int_quantized) + test cases (_rand_cases) + seed LUT,
but packs each 256-vector into ROWS=256/P rows of P lanes (lane k in the low 32 bits of bank
position k) and compares the module's P-wide y stream to the scalar ln_int reference,
element-for-element. Pass criterion: LN_VEC_VERDICT bitexact=True mismatches=0/N.

    python -m fabric.stage3.run_vec_layernorm            # P=8, 64 cases
    python -m fabric.stage3.run_vec_layernorm --p 8 --n 128 --seed 3
"""

from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np

from fabric.stage3.run_layernorm import _ln_int_quantized, _rand_cases, seed_table

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
TB = os.path.join(HERE, "tb", "tb_layernorm_vec.sv")
D = 256


def _w(path, vals, nib):
    mask = (1 << (4 * nib)) - 1
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v) & mask:0{nib}x}" for v in vals) + "\n")


def _pack_rows(vec, P):
    """256-vector -> ROWS packed words; element r*P+k -> row r, lane k (low 32 bits = lane 0)."""
    rows = []
    for r in range(len(vec) // P):
        val = 0
        for k in range(P):
            val |= (int(vec[r * P + k]) & 0xFFFFFFFF) << (32 * k)
        rows.append(val)
    return rows


def run(sim_dir, n_cases, seed, P):
    os.makedirs(sim_dir, exist_ok=True)
    assert D % P == 0
    cases = _rand_cases(n_cases, seed)

    xrows, grows, gold = [], [], []
    for X, G in cases:
        xrows += _pack_rows(X, P)
        grows += _pack_rows(G, P)
        out, *_ = _ln_int_quantized(list(X), list(G))
        gold += [int(o) & 0xFFFFFFFFFFFFFFFF for o in out]

    nib_row = (P * 32) // 4
    _w(os.path.join(sim_dir, "x.mem"), xrows, nib_row)
    _w(os.path.join(sim_dir, "g.mem"), grows, nib_row)
    _w(os.path.join(sim_dir, "seed.mem"), seed_table(), 5)

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                         f"-DNCASE={n_cases}", f"-DPVAL={P}",
                         TB, os.path.join(RTL, "layernorm_vec.sv")],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cp.stdout); print(cp.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL"); print(rp.stdout[-2000:]); print(rp.stderr[-1000:]); return False

    with open(os.path.join(sim_dir, "y.out")) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    def _hx(s):
        return None if ("x" in s or "z" in s) else int(s, 16) & 0xFFFFFFFFFFFFFFFF

    got = [_hx(s) for s in lines]
    n = min(len(got), len(gold))
    mism = sum(1 for i in range(n) if got[i] != gold[i]) + abs(len(got) - len(gold))
    nx = sum(1 for g in got if g is None)
    ok = (mism == 0) and (len(got) == len(gold))
    print(f"LN_VEC_VERDICT bitexact={ok} mismatches={mism}/{len(gold)} "
          f"x_lines={nx} P={P} ncase={n_cases} got={len(got)}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_vec_layernorm")
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default=os.path.join("C:\\kevbuild", "stage3_ln_vec"))
    a = ap.parse_args(argv)
    ok = run(a.dir, a.n, a.seed, a.p)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
