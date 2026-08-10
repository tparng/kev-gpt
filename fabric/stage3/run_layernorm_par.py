"""Parallel LayerNorm RTL gate: bit-true vs the SAME integer reference the serial
core uses, PLUS measured start->done latency vs the serial ~770.

    python -m fabric.stage3.run_layernorm_par

rtl/layernorm_par.sv is a throughput-optimized rewrite of rtl/layernorm.sv. It
eliminates the serial core's 256-cycle d*d pass by accumulating sum(x) and sum(x*x)
during the unavoidable 256-cycle load (pipelined lane reduction), then forms the
exact centered sum-of-squares algebraically:

    sum_i (x_i - mean)^2  ==  sum(x*x) - 2*mean*sum(x) + D*mean^2   (mean = sum>>8)

This is an exact integer identity for the SAME floored integer mean, so the module
is bit-IDENTICAL to fabric.stage3.run_layernorm._ln_int_quantized — the same gold
the serial core is checked against. We reuse that reference verbatim.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

from fabric.stage3._simdir import kevbuild

# reuse the EXACT integer reference + stimulus the serial gate uses
from fabric.stage3.run_layernorm import (
    _ln_int_quantized,
    _rand_cases,
    _token_stream,
    _SEED,
    D,
)

HERE = os.path.dirname(os.path.abspath(__file__))
# MEASURED serial-core latency (tb instrumented, same stimulus): total 777 cyc,
# of which 264 are non-streaming compute (256-cyc d*d pass + mean/var/rsqrt). The
# docs' "~770" is the serial TOTAL; the lever this module pulls is the 264-cyc
# non-streaming portion. See run() for the head-to-head measured comparison.
SERIAL_TOTAL_CYC = 777
SERIAL_NONSTREAM_CYC = 264


def run(sim_dir, n_cases=64, seed=0):
    os.makedirs(sim_dir, exist_ok=True)
    cases = _rand_cases(n_cases, seed)

    # gold: Q.OUT_FRAC ints, one row of 256 per case (identical reference to serial)
    gold = []
    for X, G in cases:
        out, _, _, _, _ = _ln_int_quantized(list(X), list(G))
        gold.append([int(o) & 0xFFFFFFFFFFFFFFFF for o in out])

    def w(name, vals, width_nib):
        mask = (1 << (4 * width_nib)) - 1
        with open(os.path.join(sim_dir, name), "w") as fh:
            fh.write("\n".join(f"{int(v) & mask:0{width_nib}x}" for v in vals) + "\n")

    xs = np.concatenate([c[0] for c in cases])
    gs = np.concatenate([c[1] for c in cases])
    w("x.mem", xs, 8)
    w("gamma.mem", gs, 8)
    w("seed.mem", _SEED, 5)

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(
        ["iverilog", "-g2012", "-o", vvp, f"-DNCASE={n_cases}",
         os.path.join(HERE, "tb", "tb_layernorm_par.sv"),
         os.path.join(HERE, "rtl", "layernorm_par.sv")],
        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL")
        print(cp.stdout)
        print(cp.stderr)
        return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL")
        print(rp.stdout)
        print(rp.stderr)
        return False

    def _hx(s):
        if "x" in s or "z" in s:
            return None
        return int(s, 16) & 0xFFFFFFFFFFFFFFFF

    with open(os.path.join(sim_dir, "y.out")) as fh:
        got = [_hx(s.strip()) for s in fh if s.strip()]
    flat_gold = [v for row in gold for v in row]
    mism = 0
    n = min(len(got), len(flat_gold))
    for i in range(n):
        if got[i] != flat_gold[i]:
            mism += 1
    mism += abs(len(got) - len(flat_gold))
    total = len(flat_gold)
    ok = (mism == 0)

    # measured latency (start->done cycles, includes 256 load + reduction + rsqrt + 256 out)
    with open(os.path.join(sim_dir, "lat.out")) as fh:
        lats = [int(s.strip()) for s in fh if s.strip()]
    lat_total = max(lats) if lats else -1
    # non-streaming latency = total - the two unavoidable 256-cycle streams
    lat_nonstream = lat_total - 2 * D

    # binding token-stream gate (reuse serial's: serial LN is exact => same stream)
    same, n_match, n_tot, worst = _token_stream()

    speedup = (SERIAL_NONSTREAM_CYC / lat_nonstream) if lat_nonstream > 0 else float("inf")
    print(f"LN_PAR_VERDICT bitexact={ok} mismatches={mism}/{total} "
          f"lat_total_cyc={lat_total} (serial_total={SERIAL_TOTAL_CYC}) "
          f"lat_nonstream_cyc={lat_nonstream} serial_nonstream_cyc={SERIAL_NONSTREAM_CYC} "
          f"nonstream_speedup={speedup:.1f}x "
          f"token_stream_identical={same} stream_match={n_match}/{n_tot} "
          f"worst_step_cosine={worst:.7f} pass={ok and same}")
    return ok and same


def main():
    sim_dir = kevbuild("stage3_lnpar")
    raise SystemExit(0 if run(sim_dir) else 1)


if __name__ == "__main__":
    main()
