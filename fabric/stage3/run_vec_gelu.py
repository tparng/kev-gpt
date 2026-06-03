"""vec_gelu RTL gate: bit-true vs the fixed-point reference, P lanes wide.

    python -m fabric.stage3.run_vec_gelu

vec_gelu (rtl/vec_gelu.sv) instantiates P copies of the single-lane gelu_lut, so
each lane must reproduce gelu_q (fabric/stage3/run_gelu.gelu_q) EXACTLY. This gate
feeds random Q4.12 vectors (full signed 16-bit range incl. the +-8 saturating ends)
P-wide through the module, captures the out_valid-gated stream, and compares every
lane of every row against gelu_q. Format: I/O signed Q4.12 (16-bit, +-8).
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

from fabric.stage3.run_gelu import gelu_q, gelu_table

HERE = os.path.dirname(os.path.abspath(__file__))


def run(sim_dir, N=1024, P=8, seed=0):
    os.makedirs(sim_dir, exist_ok=True)
    lut = gelu_table()
    rng = np.random.default_rng(seed)

    # full Q4.12 signed range, P-wide x N rows
    x = rng.integers(-32768, 32768, size=(N, P)).astype(np.int64)
    # force the saturating / boundary edges into the first row(s)
    edges = np.array([-32768, -22528, -4096, 0, 1, 4096, 22528, 32767], dtype=np.int64)
    x[0, :min(P, edges.size)] = edges[:P]

    y = gelu_q(x, lut)  # (N, P) golden, elementwise

    # gelu_lut.mem must live in the sim run dir (each gelu_lut instance $readmemh's it)
    with open(os.path.join(sim_dir, "gelu_lut.mem"), "w") as fh:
        fh.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in lut) + "\n")
    # one packed 16*P-bit word per row: lane k in bits [16*k +: 16] (lane 0 = LSBs)
    with open(os.path.join(sim_dir, "xin.mem"), "w") as fh:
        for row in x:
            word = 0
            for k in range(P):
                word |= (int(row[k]) & 0xFFFF) << (16 * k)
            fh.write(f"{word:0{4 * P}x}\n")

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(
        ["iverilog", "-g2012", "-o", vvp, f"-DN={N}", f"-DP={P}",
         os.path.join(HERE, "tb", "tb_vec_gelu.sv"),
         os.path.join(HERE, "rtl", "vec_gelu.sv"),
         os.path.join(HERE, "rtl", "gelu_lut.sv")],
        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cp.stdout, cp.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL"); print(rp.stdout, rp.stderr); return False

    # out_valid gates the stream -> exactly N rows, in order, no pipeline-fill garbage
    with open(os.path.join(sim_dir, "y.out")) as fh:
        rows = [line.strip() for line in fh if line.strip()]

    total = N * P
    mism = 0
    if len(rows) != N:
        # row-count mismatch is itself a failure; count everything missing as wrong
        mism = total
    else:
        for r in range(N):
            word = int(rows[r], 16)
            for k in range(P):
                got = (word >> (16 * k)) & 0xFFFF
                gold = int(y[r, k]) & 0xFFFF
                if got != gold:
                    mism += 1

    ok = (mism == 0) and (len(rows) == N)
    print(f"VEC_GELU_VERDICT bitexact={ok} mismatches={mism}/{total} P={P}")
    return ok


def main(argv=None):
    sim_dir = os.path.join("C:\\kevbuild", "stage3_vec_gelu")
    raise SystemExit(0 if run(sim_dir) else 1)


if __name__ == "__main__":
    main()
