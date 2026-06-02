"""Tier-3 sequencer integration gate: preload real weights -> run the FSM -> compare
to seq_ref.IntSequencer. Two gate modes:

  --nlayer 1   BLOCK-0 gate: one full transformer block (LN1->qkv->attn(+KV)->proj->
               +res->LN2->mlp(GELU)->+res), residual x_out compared BIT-EXACT to
               seq_ref.block0_signals -> SEQ_VERDICT block0_bitexact.
  --nlayer 4   FULL-FORWARD gate: embed -> 4 blocks -> ln_f -> head -> argmax. The
               emitted token id tok_out is compared to seq_ref.step's argmax ->
               SEQ_VERDICT tokens_identical.

    python -m fabric.stage3.run_sequencer --nlayer 1            # block-0 bit-exact
    python -m fabric.stage3.run_sequencer --nlayer 4 --tok 12   # full forward -> token

Gate chain binding the hardware to the float model:
  float goformer_seq  --(identical token stream)-->  seq_ref (integer)   [SEQREF_PASS]
  seq_ref             --(bit-exact / same argmax)->  rtl/sequencer.sv     [SEQ_VERDICT]

The driver writes every .mem ROM the RTL ($readmemh-)loads: tok_emb/pos_emb (Q6.25),
gamma ((ln1|ln2)*L | ln_f, Q4.20), wrom (transposed banked INT4: 4 GEMV/block * L +
head), dq_mant/dq_exp (24-bit per-channel), inv_sact (round(2^40/s_act)), plus the
submodule LUTs seed.mem / exp_lut.mem / gelu_lut.mem.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

from fabric.stage3 import pack_banked, seq_ref
from fabric.stage3.run_layernorm import seed_table, G_FRAC as LN_GFRAC
from fabric.stage3.run_softmax import exp_table
from fabric.stage3.run_gelu import gelu_table
from fabric.stage3.seq_ref import RESID_FRAC, VFRAC

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
TB = os.path.join(HERE, "tb", "tb_sequencer.sv")
D, D3, D_MLP, VOCAB = 256, 768, 1024, 193


def _w(path, vals, nib):
    mask = (1 << (4 * nib)) - 1
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v) & mask:0{nib}x}" for v in vals) + "\n")


def write_mems(sim_dir, intseq: seq_ref.IntSequencer, lanes: int, nlayer: int):
    os.makedirs(sim_dir, exist_ok=True)
    p = intseq.p

    # embeddings (Q6.25)
    _w(os.path.join(sim_dir, "tok_emb.mem"),
       np.round(p["tok_emb"] * (1 << RESID_FRAC)).astype(np.int64).reshape(-1), 8)
    _w(os.path.join(sim_dir, "pos_emb.mem"),
       np.round(p["pos_emb"] * (1 << RESID_FRAC)).astype(np.int64).reshape(-1), 8)

    # gamma: (ln1|ln2) per block, then ln_f
    gam = []
    for bi in range(nlayer):
        gam += list(np.round(p["blocks"][bi]["ln1"] * (1 << LN_GFRAC)).astype(np.int64))
        gam += list(np.round(p["blocks"][bi]["ln2"] * (1 << LN_GFRAC)).astype(np.int64))
    gam += list(np.round(p["ln_f"] * (1 << LN_GFRAC)).astype(np.int64))
    _w(os.path.join(sim_dir, "gamma.mem"), gam, 8)

    # transposed banked weight ROM: per block (qkv|proj|mlp_fc|mlp_proj), then head
    words = []
    for bi in range(nlayer):
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            words += pack_banked.pack_transposed(np.asarray(p["blocks"][bi][nm][0], np.int8), lanes)
    words += pack_banked.pack_transposed(np.asarray(p["head"][0], np.int8), lanes)
    _w(os.path.join(sim_dir, "wrom.mem"), words, lanes)

    # dequant per-channel mant/exp: per block (qkv|proj|mlp_fc|mlp_proj), then head
    mant_all, exp_all, inv = [], [], []
    for bi in range(nlayer):
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            L = intseq.layers[(bi, nm)]
            mant_all += [int(m) for m in L["mant"]]
            exp_all += [int(e) for e in L["exp"]]
            inv.append(intseq._inv_sact(L["s_act"]))
    mant_all += [int(m) for m in intseq.head["mant"]]
    exp_all += [int(e) for e in intseq.head["exp"]]
    inv.append(intseq._inv_sact(intseq.head["s_act"]))
    _w(os.path.join(sim_dir, "dq_mant.mem"), mant_all, 8)
    _w(os.path.join(sim_dir, "dq_exp.mem"), exp_all, 2)
    _w(os.path.join(sim_dir, "inv_sact.mem"), inv, 16)

    # submodule LUTs
    _w(os.path.join(sim_dir, "seed.mem"), seed_table(), 5)
    _w(os.path.join(sim_dir, "exp_lut.mem"), exp_table(), 6)
    _w(os.path.join(sim_dir, "gelu_lut.mem"), gelu_table(), 4)


def _compile_run(sim_dir, tok, pos, lanes, nlayer):
    srcs = [TB,
            os.path.join(RTL, "sequencer.sv"),
            os.path.join(RTL, "gemv_banked.sv"),
            os.path.join(RTL, "layernorm.sv"),
            os.path.join(RTL, "softmax.sv"),
            os.path.join(RTL, "gelu_lut.sv")]
    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                         f"-DTOKID={tok}", f"-DPOS={pos}", f"-DLANES={lanes}",
                         f"-DPNLAYER={nlayer}", *srcs],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL")
        print(cp.stdout); print(cp.stderr)
        return None
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    sys.stdout.write(rp.stdout[-1500:])
    if "TB_DONE" not in rp.stdout:
        print("VVP_RUN_FAIL"); print(rp.stderr[-2000:])
        return None
    return rp.stdout


def run(sim_dir, tok, pos, lanes, nlayer, npz="fabric/export/goformer.npz"):
    p, cfg = seq_ref.build(npz)
    intseq = seq_ref.IntSequencer(p, cfg)
    intseq.reset()
    intseq.t = pos
    write_mems(sim_dir, intseq, lanes, nlayer)

    if not _compile_run(sim_dir, tok, pos, lanes, nlayer):
        return False

    if nlayer == 1:
        # block-0 residual bit-exact gate
        sig = intseq.block0_signals(int(tok))
        gold = [int(v) & 0xFFFFFFFF for v in sig["x_out_q25"]]

        def _hx(s):
            s = s.strip()
            return None if (not s or "x" in s or "z" in s) else int(s, 16) & 0xFFFFFFFF
        with open(os.path.join(sim_dir, "xout.out")) as f:
            got = [_hx(l) for l in f if l.strip()]
        n = min(len(got), len(gold))
        mism = sum(1 for i in range(n) if got[i] != gold[i]) + abs(len(got) - len(gold))
        ok = (mism == 0) and (len(got) == len(gold) == D)
        print(f"SEQ_VERDICT block0_bitexact={ok} mismatches={mism}/{len(gold)} "
              f"tok={tok} pos={pos} LANES={lanes} stream=1/1")
        return ok

    # full forward: emitted token vs seq_ref argmax
    intseq.reset()
    intseq.t = pos
    _, ref_tok = intseq.step(int(tok))      # seq_ref's integer-logit argmax
    with open(os.path.join(sim_dir, "tokout.out")) as f:
        got_tok = int(f.read().strip())
    ok = (got_tok == ref_tok)
    print(f"SEQ_VERDICT tokens_identical={ok} stream=1/1 "
          f"rtl_tok={got_tok} ref_tok={ref_tok} tok_in={tok} pos={pos} "
          f"NLAYER={nlayer} LANES={lanes}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_sequencer")
    ap.add_argument("--tok", type=int, default=0)
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--nlayer", type=int, default=1, choices=[1, 4])
    ap.add_argument("--dir", default=os.path.join("C:\\kevbuild", "stage3_seq"))
    a = ap.parse_args(argv)
    raise SystemExit(0 if run(a.dir, a.tok, a.pos, a.lanes, a.nlayer) else 1)


if __name__ == "__main__":
    main()
