"""Observer-only attn-state profiler runner: identical to run_sb_seq but points
the iverilog compile at tb/tb_seq_sb_attnprof.sv (adds vec_attn + softmax per-state
residency counters, DUT untouched -> cyc_total must stay 66,285 at ND=6).

    python -m fabric.stage3.run_sb_attnprof --nd 6 --dir C:/kevbuild/agent_idle/sb16ap
"""
from __future__ import annotations
import argparse, os, subprocess
from fabric.stage3 import run_sb_seq

HERE = os.path.dirname(os.path.abspath(__file__))
# monkeypatch the TB path used by run_sb_seq.run
run_sb_seq.TB = os.path.join(HERE, "tb", "tb_seq_sb_attnprof.sv")

# run_sb_seq.run captures vvp stdout and only prints it on failure. Wrap
# subprocess.run so the PROFILE block (between markers) is echoed on success too.
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
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_sb_attnprof")
    ap.add_argument("--toks", default=run_sb_seq.TOKS16)
    ap.add_argument("--nd", type=int, default=6)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--lanes", type=int, default=128)
    ap.add_argument("--tmax", type=int, default=32)
    ap.add_argument("--dir", default="C:/kevbuild/agent_idle/sb16ap")
    a = ap.parse_args(argv)
    toks = [int(t) for t in a.toks.split(",")]
    ok = run_sb_seq.run(a.dir, toks, a.p, a.lanes, a.tmax, a.nd)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
