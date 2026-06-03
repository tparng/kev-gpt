"""Phase-gated bring-up of sequencer_vec (the P-wide datapath sequencer), milestone by
milestone vs seq_ref.block0_phase_signals.

  M1: embed -> LN1            -> lnout (Q.22)  vs ln1_out_q22
  M2: + act-quant -> qkv GEMV -> P-wide dequant -> qkv (Q.16) vs qkv_q16

Pass: SEQ_VEC_VERDICT ... ln1=True qkv=True.

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


def _read_hex(path, mask):
    with open(path) as fh:
        out = []
        for ln in fh:
            s = ln.strip()
            if not s:
                continue
            out.append(None if ("x" in s or "z" in s) else int(s, 16) & mask)
    return out


def run(sim_dir, tok, P, npz="fabric/export/goformer.npz"):
    os.makedirs(sim_dir, exist_ok=True)
    p, cfg = seq_ref.build(npz)
    iseq = seq_ref.IntSequencer(p, cfg)
    iseq.reset()
    sig = iseq.block0_phase_signals(int(tok))
    gold_ln = [int(v) & 0xFFFFFFFFFFFFFFFF for v in sig["ln1_out_q22"]]
    gold_qkv = [int(v) & 0xFFFFFFFF for v in sig["qkv_q16"]]
    gold_gy = [int(v) & 0xFFFFFFFF for v in sig["qkv_yint"]]
    gold_ctx = [int(v) & 0xFFFFFFFF for v in sig["ctx_q25"]]

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
                         os.path.join(RTL, "softmax.sv"),
                         os.path.join(RTL, "gemv_banked_resident.sv")],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cp.stdout); print(cp.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL"); print(rp.stdout[-2000:]); print(rp.stderr[-1000:]); return False

    got_ln = _read_hex(os.path.join(sim_dir, "lnout.out"), 0xFFFFFFFFFFFFFFFF)
    got_qkv = _read_hex(os.path.join(sim_dir, "qkv.out"), 0xFFFFFFFF)
    got_gy = _read_hex(os.path.join(sim_dir, "gemvy.out"), 0xFFFFFFFF)
    got_ctx = _read_hex(os.path.join(sim_dir, "ctx.out"), 0xFFFFFFFF)

    def _cmp(got, gold):
        n = min(len(got), len(gold))
        mism = sum(1 for i in range(n) if got[i] != gold[i]) + abs(len(got) - len(gold))
        return mism, len(gold), (mism == 0 and len(got) == len(gold))

    m_ln, t_ln, ok_ln = _cmp(got_ln, gold_ln)
    m_gy, t_gy, ok_gy = _cmp(got_gy, gold_gy)
    m_qkv, t_qkv, ok_qkv = _cmp(got_qkv, gold_qkv)
    m_ctx, t_ctx, ok_ctx = _cmp(got_ctx, gold_ctx)
    print(f"SEQ_VEC_VERDICT tok={tok} P={P} "
          f"ln1={ok_ln}({m_ln}/{t_ln}) gemvy={ok_gy}({m_gy}/{t_gy}) "
          f"qkv={ok_qkv}({m_qkv}/{t_qkv}) ctx={ok_ctx}({m_ctx}/{t_ctx})")
    for nm, got, gold in (("gemvy", got_gy, gold_gy), ("qkv", got_qkv, gold_qkv),
                          ("ctx", got_ctx, gold_ctx)):
        for i in range(min(len(got), len(gold))):
            if got[i] != gold[i]:
                g = got[i]
                print(f"  {nm} first mismatch @ {i}: got={g if g is None else format(g,'08x')} "
                      f"gold={gold[i]:08x}")
                break
    return ok_ln and ok_qkv and ok_ctx


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
