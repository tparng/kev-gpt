"""Tier-3 sequencer integration gate: preload real weights -> run the FSM -> compare
to seq_ref.IntSequencer. Gate modes:

  --nlayer 1   BLOCK-0 gate: one full transformer block (LN1->qkv->attn(+KV)->proj->
               +res->LN2->mlp(GELU)->+res), residual x_out compared BIT-EXACT to
               seq_ref.block0_signals -> SEQ_VERDICT block0_bitexact.
  --nlayer 4   FULL-FORWARD gate (single token at pos=0): embed -> 4 blocks -> ln_f ->
               head -> argmax. The emitted token tok_out is compared to seq_ref.step's
               argmax -> SEQ_VERDICT tokens_identical.
  --multitoken MULTI-TOKEN autoregressive gate: the FSM primes a resident prompt then
               greedily decodes NGEN tokens with PER-BLOCK persistent KV caches (CPU out
               of the loop after `go`). The emitted NGEN-token stream is compared to
               seq_ref.IntSequencer.generate_greedy -> SEQ_VERDICT multitoken. Per the
               project rule the BINDING gate is the token stream (raw-logit cosine is a
               brittle proxy on this char model).

    python -m fabric.stage3.run_sequencer --nlayer 1                         # block-0 bit-exact
    python -m fabric.stage3.run_sequencer --nlayer 4 --tok 12                 # 1 token  (pos=0)
    python -m fabric.stage3.run_sequencer --multitoken --prompt-len 8 --ngen 16 --seed 3

Gate chain binding the hardware to the float model:
  float goformer_seq  --(identical token stream)-->  seq_ref (integer)   [SEQREF_PASS]
  seq_ref             --(bit-exact / same stream)->  rtl/sequencer.sv     [SEQ_VERDICT]

The driver writes every .mem ROM the RTL ($readmemh-)loads: tok_emb/pos_emb (Q6.25),
gamma ((ln1|ln2)*L | ln_f, Q4.20), wrom (transposed banked INT4: 4 GEMV/block * L +
head), dq_mant/dq_exp (24-bit per-channel), inv_sact (round(2^40/s_act)), prompt (the
resident prompt token ids), plus the submodule LUTs seed/exp_lut/gelu_lut.mem.
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
TB_AXI = os.path.join(HERE, "tb", "tb_seq_axi.sv")
D, D3, D_MLP, VOCAB = 256, 768, 1024, 193


def _w(path, vals, nib):
    mask = (1 << (4 * nib)) - 1
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v) & mask:0{nib}x}" for v in vals) + "\n")


def write_mems(sim_dir, intseq: seq_ref.IntSequencer, lanes: int, nlayer: int,
               prompt=None):
    os.makedirs(sim_dir, exist_ok=True)
    p = intseq.p

    # resident prompt tokens (primes the KV cache); single-token gates use a 1-entry rom
    if prompt is None:
        prompt = [0]
    _w(os.path.join(sim_dir, "prompt.mem"), [int(t) for t in prompt], 3)

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


def _compile_run(sim_dir, tok, pos, lanes, nlayer, prompt_len=1, ngen=1, kvmax=32):
    srcs = [TB,
            os.path.join(RTL, "sequencer.sv"),
            os.path.join(RTL, "gemv_banked.sv"),
            os.path.join(RTL, "layernorm.sv"),
            os.path.join(RTL, "softmax.sv"),
            os.path.join(RTL, "gelu_lut.sv")]
    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                         f"-DTOKID={tok}", f"-DPOS={pos}", f"-DLANES={lanes}",
                         f"-DPNLAYER={nlayer}", f"-DPPROMPT={prompt_len}",
                         f"-DPNGEN={ngen}", f"-DPKVMAX={kvmax}", *srcs],
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


def _compile_run_axi(sim_dir, lanes, nlayer, prompt_len, ngen, kvmax):
    """Compile + run the AXI-LEVEL tb (tb_seq_axi.sv) driving gemv_axi_seq over AXI4-Lite:
    weights streamed in via W_DATA (the URAM LOAD PORT, not $readmemh), prompt over PL_*,
    GO, poll DONE, drain TS_*. Returns vvp stdout (tokstream.out is dumped in sim_dir)."""
    srcs = [TB_AXI,
            os.path.join(RTL, "gemv_axi_seq.v"),
            os.path.join(RTL, "sequencer.sv"),
            os.path.join(RTL, "gemv_banked.sv"),
            os.path.join(RTL, "layernorm.sv"),
            os.path.join(RTL, "softmax.sv"),
            os.path.join(RTL, "gelu_lut.sv")]
    vvp = os.path.join(sim_dir, "sim_axi.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp,
                         f"-DLANES={lanes}", f"-DPNLAYER={nlayer}",
                         f"-DPPROMPT={prompt_len}", f"-DPNGEN={ngen}",
                         f"-DPKVMAX={kvmax}", *srcs],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL")
        print(cp.stdout); print(cp.stderr)
        return None
    rp = subprocess.run(["vvp", "sim_axi.vvp"], cwd=sim_dir, capture_output=True, text=True)
    sys.stdout.write(rp.stdout[-1500:])
    if "TB_DONE" not in rp.stdout:
        print("VVP_RUN_FAIL"); print(rp.stderr[-2000:])
        return None
    return rp.stdout


def run(sim_dir, tok, pos, lanes, nlayer, npz="fabric/export/goformer.npz"):
    """Single-token gates: block-0 bit-exact (nlayer=1) or full-forward token (nlayer=4).
    Runs the FSM in legacy single-token mode (PROMPT_LEN=1, NGEN=1: one forward at pos)."""
    p, cfg = seq_ref.build(npz)
    intseq = seq_ref.IntSequencer(p, cfg)
    intseq.reset()
    intseq.t = pos
    write_mems(sim_dir, intseq, lanes, nlayer, prompt=[tok])

    if not _compile_run(sim_dir, tok, pos, lanes, nlayer, prompt_len=1, ngen=1):
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


def run_multitoken(sim_dir, lanes, nlayer, prompt_len, ngen, seed=0,
                   npz="fabric/export/goformer.npz"):
    """MULTI-TOKEN gate: the FSM autoregressively decodes `ngen` tokens from a resident
    `prompt_len`-token prompt, with PER-BLOCK persistent KV caches. The RTL's emitted
    greedy stream must be IDENTICAL to seq_ref.IntSequencer.generate_greedy. The binding
    gate is the token stream (project rule: logit cosine is a brittle proxy here)."""
    p, cfg = seq_ref.build(npz)
    intseq = seq_ref.IntSequencer(p, cfg)

    # the prompt: random ids (same construction as seq_ref._validate)
    rng = np.random.default_rng(seed)
    prompt = [int(t) for t in rng.integers(0, p["tok_emb"].shape[0], size=prompt_len)]

    # KV cache depth must hold the whole run (prompt + generated positions)
    kvmax = 1
    while kvmax < prompt_len + ngen:
        kvmax <<= 1                                     # power-of-two depth for the addr slice

    # golden integer stream = prompt + ngen greedy-predicted tokens; we compare the
    # generated tail (the ngen tokens the hardware emits).
    ref_full = intseq.generate_greedy(prompt, ngen)     # len = prompt_len + ngen
    ref_gen = [int(t) for t in ref_full[prompt_len:]]

    write_mems(sim_dir, intseq, lanes, nlayer, prompt=prompt)
    if not _compile_run(sim_dir, prompt[0], 0, lanes, nlayer,
                        prompt_len=prompt_len, ngen=ngen, kvmax=kvmax):
        return False

    with open(os.path.join(sim_dir, "tokstream.out")) as f:
        got = [int(l.strip()) for l in f if l.strip()]

    n = min(len(got), len(ref_gen))
    n_match = sum(1 for i in range(n) if got[i] == ref_gen[i])
    identical = (got == ref_gen)
    ok = identical and (len(got) == ngen)
    print(f"prompt={prompt}")
    print(f"ref_gen ={ref_gen}")
    print(f"rtl_gen ={got}")
    print(f"SEQ_VERDICT multitoken tokens_identical={identical} stream={n_match}/{ngen} "
          f"NLAYER={nlayer} LANES={lanes} PROMPT_LEN={prompt_len} NGEN={ngen} KVMAX={kvmax} "
          f"pass={ok}")
    return ok


def run_axi(sim_dir, lanes, nlayer, prompt_len, ngen, seed=0,
            npz="fabric/export/goformer.npz"):
    """AXI GATE (the capstone, binding): drive gemv_axi_seq over the AXI4-Lite bus —
    stream the weight image in through W_DATA (the URAM LOAD PORT, not $readmemh), write
    the prompt over PL_*, pulse GO, poll DONE, drain the token stream over TS_*. The
    emitted NGEN-token stream must be IDENTICAL to seq_ref.generate_greedy. This proves
    the synth-safe weight path AND the full register interface end to end."""
    p, cfg = seq_ref.build(npz)
    intseq = seq_ref.IntSequencer(p, cfg)

    rng = np.random.default_rng(seed)
    prompt = [int(t) for t in rng.integers(0, p["tok_emb"].shape[0], size=prompt_len)]
    kvmax = 1
    while kvmax < prompt_len + ngen:
        kvmax <<= 1

    ref_full = intseq.generate_greedy(prompt, ngen)
    ref_gen = [int(t) for t in ref_full[prompt_len:]]

    write_mems(sim_dir, intseq, lanes, nlayer, prompt=prompt)
    if not _compile_run_axi(sim_dir, lanes, nlayer, prompt_len, ngen, kvmax):
        return False

    with open(os.path.join(sim_dir, "tokstream.out")) as f:
        got = [int(l.strip()) for l in f if l.strip()]

    n = min(len(got), len(ref_gen))
    n_match = sum(1 for i in range(n) if got[i] == ref_gen[i])
    identical = (got == ref_gen)
    ok = identical and (len(got) == ngen)
    print(f"prompt={prompt}")
    print(f"ref_gen ={ref_gen}")
    print(f"axi_gen ={got}")
    print(f"SEQ_AXI_VERDICT tokens_identical={identical} stream={n_match}/{ngen} "
          f"NLAYER={nlayer} LANES={lanes} PROMPT_LEN={prompt_len} NGEN={ngen} KVMAX={kvmax} "
          f"pass={ok}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_sequencer")
    ap.add_argument("--tok", type=int, default=0)
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--nlayer", type=int, default=None, choices=[1, 4])
    ap.add_argument("--multitoken", action="store_true",
                    help="run the autoregressive multi-token stream gate")
    ap.add_argument("--axi", action="store_true",
                    help="run the AXI-wrapped capstone gate (gemv_axi_seq over AXI4-Lite)")
    ap.add_argument("--prompt-len", type=int, default=4)
    ap.add_argument("--ngen", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default=os.path.join("C:\\kevbuild", "stage3_seqmt"))
    a = ap.parse_args(argv)
    # default layer count: the multi-token / AXI gates are for the REAL 4-layer model
    # (nlayer=1 multitoken is a non-physical single-layer config and is not supported);
    # the single-token default is the block-0 bit-exact gate (nlayer=1).
    nlayer = a.nlayer if a.nlayer is not None else (4 if (a.multitoken or a.axi) else 1)
    if a.axi:
        if nlayer != 4:
            raise SystemExit("AXI gate requires --nlayer 4 (the real model)")
        ok = run_axi(a.dir if a.dir != os.path.join("C:\\kevbuild", "stage3_seqmt")
                     else os.path.join("C:\\kevbuild", "stage3_seqaxi"),
                     a.lanes, nlayer, a.prompt_len, a.ngen, a.seed)
    elif a.multitoken:
        if nlayer != 4:
            raise SystemExit("multi-token gate requires --nlayer 4 (the real model)")
        ok = run_multitoken(a.dir, a.lanes, nlayer, a.prompt_len, a.ngen, a.seed)
    else:
        ok = run(a.dir, a.tok, a.pos, a.lanes, nlayer)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
