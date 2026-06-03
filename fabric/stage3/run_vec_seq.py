"""Phase-gated bring-up of sequencer_vec (the P-wide datapath sequencer).

M4 = the FULL block-0 forward, P-wide, every phase bit-exact vs seq_ref.block0_phase_signals:
  ln1 -> qkv -> ctx -> attn_out -> ln2 -> gelu -> mlp_out -> x_out.

All banks are read back as 64-bit (32-bit signed banks sign-extended) and compared to the
reference masked to 64-bit. Pass: SEQ_VEC_VERDICT ... all phases True, x_out the binding one.

    python -m fabric.stage3.run_vec_seq --tok 48 --p 8
"""

from __future__ import annotations

import argparse
import os
import subprocess

from fabric.stage3 import seq_ref
from fabric.stage3.run_sequencer import write_mems_banked

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
TB = os.path.join(HERE, "tb", "tb_seq_vec.sv")
M64 = (1 << 64) - 1

# (file, seq_ref key, length)
PHASES = [
    ("ln1.out",  "ln1_out_q22", 256),
    ("qkv.out",  "qkv_q16",     768),
    ("ctx.out",  "ctx_q25",     256),
    ("attn.out", "attn_out_q25",256),
    ("ln2.out",  "ln2_out_q22", 256),
    ("gelu.out", "gelu_q22",    1024),
    ("mlp.out",  "mlp_out_q25", 256),
    ("xout.out", "x_out_q25",   256),
]


def _read_hex(path):
    out = []
    with open(path) as fh:
        for ln in fh:
            s = ln.strip()
            if not s:
                continue
            out.append(None if ("x" in s or "z" in s) else int(s, 16) & M64)
    return out


def run(sim_dir, tok, P, npz="fabric/export/goformer.npz"):
    os.makedirs(sim_dir, exist_ok=True)
    p, cfg = seq_ref.build(npz)
    iseq = seq_ref.IntSequencer(p, cfg)
    iseq.reset()
    sig = iseq.block0_phase_signals(int(tok))

    iseq.reset()
    write_mems_banked(sim_dir, iseq, 16, 4, P)       # wrom@LANES16 + banked dq (P)

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                         f"-DTOK={int(tok)}", f"-DPVAL={P}",
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

    parts = []
    all_ok = True
    for fname, key, n in PHASES:
        got = _read_hex(os.path.join(sim_dir, fname))
        gold = [int(v) & M64 for v in sig[key]]
        m = min(len(got), len(gold))
        mism = sum(1 for i in range(m) if got[i] != gold[i]) + abs(len(got) - len(gold))
        ok = (mism == 0) and (len(got) == len(gold))
        all_ok = all_ok and ok
        tag = fname.split(".")[0]
        parts.append(f"{tag}={ok}({mism}/{len(gold)})")
        if not ok:
            for i in range(m):
                if got[i] != gold[i]:
                    g = got[i]
                    print(f"  {tag} first mismatch @ {i}: got="
                          f"{'x' if g is None else format(g, '016x')} gold={gold[i]:016x}")
                    break
    print(f"SEQ_VEC_VERDICT tok={tok} P={P} " + " ".join(parts) + f" ALL={all_ok}")
    return all_ok


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
