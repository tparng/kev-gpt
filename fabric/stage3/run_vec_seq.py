"""Phase-gated bring-up of sequencer_vec (the P-wide datapath sequencer), milestone by
milestone against seq_ref.block0_phase_signals.

MILESTONE 1 (this gate): embed -> LN1. Runs sequencer_vec for token TOK at pos 0, reads back
the 256 lnout (Q.22) values, compares to ln1_out_q22 from the reference. Pass:
SEQ_VEC_VERDICT ln1 bitexact=True mismatches=0/256.

    python -m fabric.stage3.run_vec_seq --tok 48 --p 8
"""

from __future__ import annotations

import argparse
import os
import subprocess

from fabric.stage3 import seq_ref
from fabric.stage3.run_sequencer import write_mems

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
TB = os.path.join(HERE, "tb", "tb_seq_vec.sv")


def run(sim_dir, tok, P, npz="fabric/export/goformer.npz"):
    os.makedirs(sim_dir, exist_ok=True)
    p, cfg = seq_ref.build(npz)
    iseq = seq_ref.IntSequencer(p, cfg)
    iseq.reset()
    sig = iseq.block0_phase_signals(int(tok))
    gold = [int(v) & 0xFFFFFFFFFFFFFFFF for v in sig["ln1_out_q22"]]

    # write_mems emits tok_emb/pos_emb/gamma/seed (+ others we don't use here); lanes only
    # affects wrom which milestone 1 doesn't touch. gamma row 0 = block0 ln1 (GBASE=0).
    iseq.reset()
    write_mems(sim_dir, iseq, 8, 4)

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                         f"-DTOK={int(tok)}", f"-DPVAL={P}",
                         TB, os.path.join(RTL, "sequencer_vec.sv"),
                         os.path.join(RTL, "layernorm_vec.sv")],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cp.stdout); print(cp.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL"); print(rp.stdout[-2000:]); print(rp.stderr[-1000:]); return False

    with open(os.path.join(sim_dir, "lnout.out")) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    def _hx(s):
        return None if ("x" in s or "z" in s) else int(s, 16) & 0xFFFFFFFFFFFFFFFF

    got = [_hx(s) for s in lines]
    n = min(len(got), len(gold))
    mism = sum(1 for i in range(n) if got[i] != gold[i]) + abs(len(got) - len(gold))
    nx = sum(1 for g in got if g is None)
    ok = (mism == 0) and (len(got) == len(gold))
    print(f"SEQ_VEC_VERDICT ln1 bitexact={ok} mismatches={mism}/{len(gold)} "
          f"x_lines={nx} tok={tok} P={P} got={len(got)}")
    if not ok and n:
        for i in range(min(n, 256)):
            if got[i] != gold[i]:
                print(f"  first mismatch @ {i}: got={got[i]:016x} gold={gold[i]:016x}")
                break
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_vec_seq")
    ap.add_argument("--tok", type=int, default=48)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--dir", default=os.path.join("C:\\kevbuild", "stage3_seq_vec"))
    a = ap.parse_args(argv)
    ok = run(a.dir, a.tok, a.p)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
