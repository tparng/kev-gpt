"""Full-forward gate for sequencer_vec (M5): single token through 4 blocks + LN_f + head +
argmax, P-wide. Compares tok_out to seq_ref.full_forward_signals' argmax (the binding check),
plus x4 (residual after the last block), lnf (LN_f out), and head logits for localisation.

    python -m fabric.stage3.run_vec_seq --tok 48 --p 8
"""

from __future__ import annotations

import argparse
import os
import subprocess

from fabric.stage3 import seq_ref
from fabric.stage3.run_sequencer import write_mems_wideword

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
TB = os.path.join(HERE, "tb", "tb_seq_vec.sv")
M64 = (1 << 64) - 1

PHASES = [("x4.out", "x4_q25", 256), ("lnf.out", "lnf_q22", 256), ("head.out", "head_q25", 193)]


def _read_hex(path):
    out = []
    with open(path) as fh:
        for ln in fh:
            s = ln.strip()
            if s:
                out.append(None if ("x" in s or "z" in s) else int(s, 16) & M64)
    return out


def run(sim_dir, tok, P, lanes=16, tmax=256, npz="fabric/export/goformer.npz"):
    os.makedirs(sim_dir, exist_ok=True)
    p, cfg = seq_ref.build(npz)
    iseq = seq_ref.IntSequencer(p, cfg)
    sig = iseq.full_forward_signals(int(tok))
    iseq.reset()
    write_mems_wideword(sim_dir, iseq, lanes, 4, P)
    with open(os.path.join(sim_dir, "wrom.mem")) as fh:
        wrom_n = sum(1 for ln in fh if ln.strip())

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                         f"-DTOK={int(tok)}", f"-DPVAL={P}",
                         f"-DLVAL={lanes}", f"-DWROMN={wrom_n}", f"-DTMAXVAL={tmax}",
                         TB,
                         os.path.join(RTL, "sequencer_vec.sv"),
                         os.path.join(RTL, "layernorm_vec.sv"),
                         os.path.join(RTL, "vec_dequant.sv"),
                         os.path.join(RTL, "vec_attn.sv"),
                         os.path.join(RTL, "vec_gelu.sv"),
                         os.path.join(RTL, "gelu_lut.sv"),
                         os.path.join(RTL, "softmax.sv"),
                         os.path.join(RTL, "gemv_banked_resident.sv")],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cp.stdout); print(cp.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL"); print(rp.stdout[-2000:]); print(rp.stderr[-1500:]); return False

    with open(os.path.join(sim_dir, "tok.out")) as fh:
        got_tok = int(fh.read().strip())
    tok_ok = (got_tok == sig["tok"])

    parts = [f"tok={tok_ok}(got={got_tok},gold={sig['tok']})"]
    all_ok = tok_ok
    for fname, key, n in PHASES:
        got = _read_hex(os.path.join(sim_dir, fname))
        gold = [int(v) & M64 for v in sig[key]]
        m = min(len(got), len(gold))
        mism = sum(1 for i in range(m) if got[i] != gold[i]) + abs(len(got) - len(gold))
        ok = (mism == 0) and (len(got) == len(gold))
        all_ok = all_ok and ok
        parts.append(f"{key.split('_')[0]}={ok}({mism}/{len(gold)})")
        if not ok:
            for i in range(m):
                if got[i] != gold[i]:
                    g = got[i]
                    print(f"  {key} first mismatch @ {i}: "
                          f"got={'x' if g is None else format(g, '016x')} gold={gold[i]:016x}")
                    break
    cyc = None
    try:
        with open(os.path.join(sim_dir, "cyc.out")) as fh:
            cyc = int(fh.read().strip())
    except Exception:
        pass
    print(f"SEQ_VEC_FULL tok={tok} P={P} L={lanes} TMAX={tmax} " + " ".join(parts) +
          f" ALL={all_ok} fwd_cyc={cyc}")
    return all_ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_vec_seq")
    ap.add_argument("--tok", type=int, default=48)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--tmax", type=int, default=256,
                    help="pos table depth baked into the RTL (64 = BRAM-budget build)")
    ap.add_argument("--dir", default=os.path.join("C:\\kevbuild", "stage3_seq_vec"))
    a = ap.parse_args(argv)
    ok = run(a.dir, a.tok, a.p, a.lanes, a.tmax)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
