"""GE-state-only profiler: like run_sb_attnprof but points at tb_seq_sb_geprof.sv
which forces ATT2=0 (shared attention) and drops the per-cohort attn instrumentation
that only binds under ATT2=1. Gives the per-cohort GE-state cycle breakdown
(GE_IDLE/AQ/RUN/WAIT/RB/RBN/DQW) for the att2=0 build.

    python -m fabric.stage3.run_sb_geprof --nd 6 --tmax 16 --dir C:/kevbuild/agent_rb/geprof
"""
from __future__ import annotations
import argparse, os, subprocess
from fabric.stage3 import run_sb_seq

HERE = os.path.dirname(os.path.abspath(__file__))
run_sb_seq.TB = os.path.join(HERE, "tb", "tb_seq_sb_geprof.sv")

_orig_run = subprocess.run
def _run_echo(*a, **k):
    cp = _orig_run(*a, **k)
    out = getattr(cp, "stdout", None)
    if out and ("PROFILE" in out or "vvp" in str(a[0])):
        if "==== PROFILE" in out:
            print(out[out.index("==== PROFILE"):])
    return cp
subprocess.run = _run_echo
run_sb_seq.subprocess.run = _run_echo


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_sb_geprof")
    ap.add_argument("--toks", default=run_sb_seq.TOKS16)
    ap.add_argument("--nd", type=int, default=6)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--lanes", type=int, default=128)
    ap.add_argument("--tmax", type=int, default=16)
    ap.add_argument("--dir", default="C:/kevbuild/agent_rb/geprof")
    a = ap.parse_args(argv)
    toks = [int(t) for t in a.toks.split(",")]
    # att2 fixed to 0 in the TB; pass att2=0 so run_sb_seq's -DATT2VAL matches.
    ok = run_sb_seq.run(a.dir, toks, a.p, a.lanes, a.tmax, a.nd, 0)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
