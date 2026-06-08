"""Split-brain full-forward gate for sequencer_sb: N=16 = two independent N=8
cohorts sharing the TDP weight image + shared LN/attn/embed. Each stream is
compared independently to seq_ref.full_forward_signals (tok + x4 + lnf + head),
and the total cycles / 16 tok is reported (target ~63k, vs 99,828 single-engine).

    python -m fabric.stage3.run_sb_seq --nd 0 --dir C:/kevbuild/stage3_seq_sb
"""

from __future__ import annotations

import argparse
import os
import subprocess

from fabric.stage3 import seq_ref
from fabric.stage3.run_sequencer import write_mems_wideword

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
TB = os.path.join(HERE, "tb", "tb_seq_sb.sv")
M64 = (1 << 64) - 1

PHASES = [("x4_{b}.out", "x4_q25", 256), ("lnf_{b}.out", "lnf_q22", 256),
          ("head_{b}.out", "head_q25", 193)]


def _read_hex(path):
    out = []
    with open(path) as fh:
        for ln in fh:
            s = ln.strip()
            if s:
                out.append(None if ("x" in s or "z" in s) else int(s, 16) & M64)
    return out


def run(sim_dir, toks, P, lanes=128, tmax=32, nd=0, att2=1, dp=0,
        npz="fabric/export/goformer.npz"):
    os.makedirs(sim_dir, exist_ok=True)
    p, cfg = seq_ref.build(npz)
    iseq = seq_ref.IntSequencer(p, cfg)
    sigs = []
    for t in toks:
        sigs.append(iseq.full_forward_signals(int(t)))
        iseq.reset()
    write_mems_wideword(sim_dir, iseq, lanes, 4, P)
    with open(os.path.join(sim_dir, "gelu_lut.mem")) as fh:
        lut = [ln.strip() for ln in fh if ln.strip()]
    with open(os.path.join(sim_dir, "gelu_lut_e.mem"), "w") as fh:
        fh.write("\n".join(lut[0::2]) + "\n")
    with open(os.path.join(sim_dir, "gelu_lut_o.mem"), "w") as fh:
        fh.write("\n".join(lut[1::2]) + "\n")
    with open(os.path.join(sim_dir, "wrom.mem")) as fh:
        wrom_n = sum(1 for ln in fh if ln.strip())

    vvp = os.path.join(sim_dir, "sim.vvp")
    defs = [f"-DTOK{i}={int(t)}" for i, t in enumerate(toks)]
    if dp:
        defs.append("-DDPUMP")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                         *defs, f"-DPVAL={P}", f"-DNVAL={len(toks)}", f"-DNDVAL={nd}", f"-DATT2VAL={att2}",
                         f"-DLVAL={lanes}", f"-DWROMN={wrom_n}", f"-DTMAXVAL={tmax}",
                         TB,
                         os.path.join(RTL, "mac_bank_dp.sv"),
                         os.path.join(RTL, "sequencer_sb.sv"),
                         os.path.join(RTL, "cohort_engine.sv"),
                         os.path.join(RTL, "nl_engine.sv"),
                         os.path.join(RTL, "layernorm_vec.sv"),
                         os.path.join(RTL, "vec_dequant.sv"),
                         os.path.join(RTL, "vec_attn.sv"),
                         os.path.join(RTL, "vec_gelu.sv"),
                         os.path.join(RTL, "gelu_lut.sv"),
                         os.path.join(RTL, "gelu_lut2.sv"),
                         os.path.join(RTL, "softmax.sv"),
                         os.path.join(RTL, "weight_bank_tdp.sv"),
                         os.path.join(RTL, "embed_bank_tdp.sv"),
                         os.path.join(RTL, "gemm_cohort_vec.sv"),
                         os.path.join(RTL, "gemm_banked_resident_vec.sv")],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL"); print(cp.stdout); print(cp.stderr); return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL"); print(rp.stdout[-3000:]); print(rp.stderr[-1500:]); return False

    with open(os.path.join(sim_dir, "tok.out")) as fh:
        got_toks = [int(ln.strip()) if "x" not in ln.lower() else -1
                    for ln in fh if ln.strip()]

    all_ok = True
    parts = []
    for b, (tok, sig) in enumerate(zip(toks, sigs)):
        tok_ok = (got_toks[b] == sig["tok"])
        ok_b = tok_ok
        for fname, key, n in PHASES:
            got = _read_hex(os.path.join(sim_dir, fname.format(b=b)))
            gold = [int(v) & M64 for v in sig[key]]
            m = min(len(got), len(gold))
            mism = sum(1 for i in range(m) if got[i] != gold[i]) + abs(len(got) - len(gold))
            ok = (mism == 0) and (len(got) == len(gold))
            ok_b = ok_b and ok
            if not ok:
                for i in range(m):
                    if got[i] != gold[i]:
                        g = got[i]
                        print(f"  s{b} {key} first mismatch @ {i}: "
                              f"got={'x' if g is None else format(g, '016x')} gold={gold[i]:016x}")
                        break
        all_ok = all_ok and ok_b
        parts.append(f"s{b}(tok{tok})={'OK' if ok_b else 'FAIL'}")
    cyc = None
    try:
        with open(os.path.join(sim_dir, "cyc.out")) as fh:
            cyc = int(fh.read().strip())
    except Exception:
        pass
    n = len(toks)
    print(f"SEQ_SB_FULL N={n} ND={nd} P={P} L={lanes} TMAX={tmax} ATT2={att2} DP={dp} " + " ".join(parts) +
          f" ALL={all_ok} cyc_total={cyc} cyc/token={(cyc or 0)//n}")
    if cyc:
        print(f"SEQ_SB_TOKS/S @166.7={166.7e6*n/cyc:,.0f}  @200={200e6*n/cyc:,.0f}")
    return all_ok


TOKS16 = "48,10,100,77,5,60,120,180,33,7,150,90,14,66,111,173"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_sb_seq")
    ap.add_argument("--toks", default=TOKS16)
    ap.add_argument("--nd", type=int, default=0, help="DSP-packed GEMM streams per cohort")
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--lanes", type=int, default=128)
    ap.add_argument("--tmax", type=int, default=32)
    ap.add_argument("--att2", type=int, default=1,
                    help="1 = per-cohort vec_attn; 0 = shared+arbiter (the fitting BD config)")
    ap.add_argument("--dp", type=int, default=0,
                    help="DOUBLE-PUMP-100K Stage 1: 1 = run the MAC at 2 K-steps/clk "
                         "(clk2x). Use with --nd 0 (DSP path not yet double-pumped).")
    ap.add_argument("--dir", default=os.path.join("C:\\kevbuild", "stage3_seq_sb"))
    a = ap.parse_args(argv)
    toks = [int(t) for t in a.toks.split(",")]
    ok = run(a.dir, toks, a.p, a.lanes, a.tmax, a.nd, a.att2, a.dp)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
