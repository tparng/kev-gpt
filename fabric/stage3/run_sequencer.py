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


def write_banked_mem(prefix, vals, P, nib, pad=True):
    """Bank-interleave a length-L vector into P ROM files for the P-wide datapath.

    Element e of `vals` goes to bank (e % P) at position (e // P) within that bank.
    Equivalently bank b holds vals[b::P]. Writes files prefix.b0.mem .. prefix.b{P-1}.mem,
    each value `nib` hex digits, masked (reuses the same per-value formatting as `_w`).

    L must be divisible by P. When pad=True (default) a length that is not a multiple of
    P is zero-padded up to the next multiple before interleaving, so each bank file has an
    equal number of words (what the P-wide datapath expects: it reads P channels per beat
    and the last partial beat is zero-filled). The padding tail is purely trailing zeros,
    so de-interleaving the first L words reconstructs `vals` exactly. With pad=False the
    divisibility is asserted instead.
    """
    vals = list(vals)
    L = len(vals)
    if L % P != 0:
        assert pad, f"banked rom length {L} not divisible by P={P}"
        vals = vals + [0] * (P - L % P)
    for b in range(P):
        _w(f"{prefix}.b{b}.mem", vals[b::P], nib)


def write_mems_banked(sim_dir, intseq: seq_ref.IntSequencer, lanes: int, nlayer: int,
                      P: int, prompt=None):
    """Same outputs as write_mems, but the per-channel/per-element ROMs the P-wide datapath
    reads P-at-a-time (dq_mant, dq_exp, inv_sact, gamma) are ALSO emitted bank-interleaved
    into P files each via write_banked_mem. The values + nibble widths are IDENTICAL to
    write_mems; only the file layout differs. The remaining 1-wide ROMs (tok_emb/pos_emb/
    prompt/wrom/seed/exp_lut/gelu_lut) are produced by calling write_mems unchanged.
    """
    # produce all the standard 1-wide ROMs (incl. the flat dq_mant/dq_exp/inv_sact/gamma)
    write_mems(sim_dir, intseq, lanes, nlayer, prompt=prompt)
    p = intseq.p

    # gamma: (ln1|ln2) per block, then ln_f  (same construction + Q4.20 / 8-nib as write_mems)
    gam = []
    for bi in range(nlayer):
        gam += list(np.round(p["blocks"][bi]["ln1"] * (1 << LN_GFRAC)).astype(np.int64))
        gam += list(np.round(p["blocks"][bi]["ln2"] * (1 << LN_GFRAC)).astype(np.int64))
    gam += list(np.round(p["ln_f"] * (1 << LN_GFRAC)).astype(np.int64))

    # dq_mant / dq_exp / inv_sact: per block (qkv|proj|mlp_fc|mlp_proj), then head
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

    # bank-interleaved copies (same nibble widths as write_mems: mant 8, exp 2, inv 16, gamma 8)
    write_banked_mem(os.path.join(sim_dir, "dq_mant"), mant_all, P, 8)
    write_banked_mem(os.path.join(sim_dir, "dq_exp"), exp_all, P, 2)
    write_banked_mem(os.path.join(sim_dir, "inv_sact"), inv, P, 16)
    write_banked_mem(os.path.join(sim_dir, "gamma"), gam, P, 8)


def write_wideword_mem(path, vals, P, bits):
    """Pack a length-L vector into WIDE words of P elements each (lane 0 in the low bits),
    one hex line per word — the layout the WIDE-WORD sequencer_vec reads. word w holds
    vals[w*P + 0 .. w*P + P-1]; lane `l` occupies bits [l*bits +: bits]. A non-multiple tail
    is zero-padded. Each line is ceil(P*bits/4) hex nibbles (e.g. P=8,bits=32 -> 64 nibbles).
    This is the de-banked counterpart of write_banked_mem: the SAME element->(word,lane)
    mapping, but emitted as one packed word/line instead of P separate bank files (so each
    buffer maps to ONE row-addressed BRAM instead of P arrays that synth-dissolve to muxes)."""
    vals = list(vals)
    mask = (1 << bits) - 1
    nwords = (len(vals) + P - 1) // P
    nib = (P * bits + 3) // 4
    lines = []
    for w in range(nwords):
        acc = 0
        for lane in range(P):
            idx = w * P + lane
            v = int(vals[idx]) & mask if idx < len(vals) else 0
            acc |= v << (lane * bits)
        lines.append(f"{acc:0{nib}x}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_mems_wideword(sim_dir, intseq: seq_ref.IntSequencer, lanes: int, nlayer: int,
                        P: int, prompt=None):
    """Outputs for the WIDE-WORD sequencer_vec: the standard 1-wide ROMs (wrom/seed/exp_lut/
    gelu_lut/inv_sact/prompt via write_mems) PLUS the P-wide-read ROMs packed wide-word —
    tok_emb_w / pos_emb_w / gamma_w (P x 32b/word) and dqm_w / dqe_w (P x 24b / P x 8b). The
    values + element order are IDENTICAL to write_mems_banked; only the file layout differs
    (one packed word/line vs P bank files)."""
    write_mems(sim_dir, intseq, lanes, nlayer, prompt=prompt)
    p = intseq.p

    tok = np.round(p["tok_emb"] * (1 << RESID_FRAC)).astype(np.int64).reshape(-1)
    pos = np.round(p["pos_emb"] * (1 << RESID_FRAC)).astype(np.int64).reshape(-1)
    write_wideword_mem(os.path.join(sim_dir, "tok_emb_w.mem"), tok, P, 32)
    write_wideword_mem(os.path.join(sim_dir, "pos_emb_w.mem"), pos, P, 32)

    # log §36 fit-plan 2: APPEND the embed tables to wrom.mem so sequencer_vec reads
    # them from the resident weight URAM's spare depth (through the GEMV bank's idle
    # port-B) instead of dedicated BRAM ROMs. Word format matches write_mems' wrom
    # lines exactly (LANES hex nibbles = LANES*4 bits, streamed low-32-bits-first by
    # the loader). EPW embed rows (each P x 32b, lane 0 in the low bits — the same
    # element->(row,lane) map as tok_emb_w.mem) pack into one word:
    # word = {row_{EPW-1}, ..., row1, row0}. Both bases are padded EVEN so the DP=1
    # column-parity pair-read returns 2*EPW consecutive rows; the RTL derives the
    # same bases (EMB_TOK_BASE/EMB_POS_BASE) from its weight-image localparams.
    # The old tok_emb_w.mem/pos_emb_w.mem files above are still emitted for other
    # harnesses. Requires EPW >= 1 (LANES >= 8*P); smaller LANES are skipped.
    epw = (lanes * 4) // (P * 32)
    if epw >= 1:
        nib = lanes                                   # hex chars/word, as in write_mems

        def _emb_words(flat):
            nrows = (len(flat) + P - 1) // P
            rows = []
            for w in range(nrows):                    # P*32b rows, lane l at [l*32 +: 32]
                acc = 0
                for l in range(P):
                    idx = w * P + l
                    v = int(flat[idx]) & 0xFFFFFFFF if idx < len(flat) else 0
                    acc |= v << (l * 32)
                rows.append(acc)
            words = []
            for b in range(0, len(rows), epw):        # EPW rows per wrom word, row0 low
                acc = 0
                for r, rv in enumerate(rows[b:b + epw]):
                    acc |= rv << (r * P * 32)
                words.append(acc)
            return words

        wrom_path = os.path.join(sim_dir, "wrom.mem")
        with open(wrom_path) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if len(lines) % 2:                            # even-align EMB_TOK_BASE
            lines.append("0" * nib)
        lines += [f"{w:0{nib}x}" for w in _emb_words(tok)]
        if len(lines) % 2:                            # even-align EMB_POS_BASE
            lines.append("0" * nib)
        lines += [f"{w:0{nib}x}" for w in _emb_words(pos)]
        with open(wrom_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    gam = []
    for bi in range(nlayer):
        gam += list(np.round(p["blocks"][bi]["ln1"] * (1 << LN_GFRAC)).astype(np.int64))
        gam += list(np.round(p["blocks"][bi]["ln2"] * (1 << LN_GFRAC)).astype(np.int64))
    gam += list(np.round(p["ln_f"] * (1 << LN_GFRAC)).astype(np.int64))
    write_wideword_mem(os.path.join(sim_dir, "gamma_w.mem"), gam, P, 32)

    mant_all, exp_all = [], []
    for bi in range(nlayer):
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            L = intseq.layers[(bi, nm)]
            mant_all += [int(m) for m in L["mant"]]
            exp_all += [int(e) for e in L["exp"]]
    mant_all += [int(m) for m in intseq.head["mant"]]
    exp_all += [int(e) for e in intseq.head["exp"]]
    write_wideword_mem(os.path.join(sim_dir, "dqm_w.mem"), mant_all, P, 24)
    write_wideword_mem(os.path.join(sim_dir, "dqe_w.mem"), exp_all, P, 8)


def _compile_run(sim_dir, tok, pos, lanes, nlayer, prompt_len=1, ngen=1, kvmax=32,
                 fast=False, dbg_phase=False):
    # fast=True gates sequencer_fast (resident-read GEMV: weights preloaded once into the
    # core's URAM, no per-matmul reload). Both DUTs share the tb (DUTMOD macro) and the
    # same wl_* weight stream + token-stream contract, so the gate is apples-to-apples.
    srcs = [TB,
            os.path.join(RTL, "sequencer_fast.sv" if fast else "sequencer.sv"),
            os.path.join(RTL, "gemv_banked_resident.sv" if fast else "gemv_banked.sv"),
            os.path.join(RTL, "layernorm.sv"),
            os.path.join(RTL, "softmax.sv"),
            os.path.join(RTL, "gelu_lut.sv")]
    defs = [f"-DTOKID={tok}", f"-DPOS={pos}", f"-DLANES={lanes}",
            f"-DPNLAYER={nlayer}", f"-DPPROMPT={prompt_len}",
            f"-DPNGEN={ngen}", f"-DPKVMAX={kvmax}"]
    if fast:
        defs.append("-DSEQ_FAST")
    if dbg_phase:
        defs.append("-DDBG_PHASE")
    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(["iverilog", "-g2012", "-o", vvp, *defs, *srcs],
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
                   npz="fabric/export/goformer.npz", fast=False, dbg_phase=False):
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
    out = _compile_run(sim_dir, prompt[0], 0, lanes, nlayer,
                       prompt_len=prompt_len, ngen=ngen, kvmax=kvmax, fast=fast,
                       dbg_phase=dbg_phase)
    if not out:
        return False
    # pull the tb cycle count (TB_DONE ... cycles=N) for the A/B speed compare
    cyc = None
    for line in out.splitlines():
        if "TB_DONE" in line and "cycles=" in line:
            cyc = int(line.split("cycles=")[1].split()[0])
    # per-phase cycle breakdown (PHASE_BREAKDOWN ...), if -DDBG_PHASE was set
    for line in out.splitlines():
        if "PHASE_BREAKDOWN" in line:
            kv = dict(tok.split("=") for tok in line.split()[1:])
            tot = sum(int(v) for v in kv.values())
            print(f"--- per-phase cycles (total {tot}, /{ngen}tok = {tot//ngen}/tok) ---")
            for k, v in sorted(kv.items(), key=lambda x: -int(x[1])):
                v = int(v)
                print(f"  {k:10s} {v:>10d}  {v//ngen:>8d}/tok  {100*v/max(tot,1):5.1f}%")

    with open(os.path.join(sim_dir, "tokstream.out")) as f:
        got = [int(l.strip()) for l in f if l.strip()]

    n = min(len(got), len(ref_gen))
    n_match = sum(1 for i in range(n) if got[i] == ref_gen[i])
    identical = (got == ref_gen)
    ok = identical and (len(got) == ngen)
    print(f"prompt={prompt}")
    print(f"ref_gen ={ref_gen}")
    print(f"rtl_gen ={got}")
    cyc_s = f" cycles={cyc} ({cyc/ngen:.0f}/tok)" if cyc else ""
    print(f"SEQ_VERDICT{' FAST' if fast else ''} multitoken tokens_identical={identical} "
          f"stream={n_match}/{ngen} NLAYER={nlayer} LANES={lanes} PROMPT_LEN={prompt_len} "
          f"NGEN={ngen} KVMAX={kvmax} pass={ok}{cyc_s}")
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
    ap.add_argument("--fast", action="store_true",
                    help="gate sequencer_fast (resident-read GEMV) instead of sequencer")
    ap.add_argument("--dbg-phase", action="store_true",
                    help="dump per-phase cycle breakdown (LN/GEMV/dequant/attn/GELU/...)")
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
        ok = run_multitoken(a.dir, a.lanes, nlayer, a.prompt_len, a.ngen, a.seed,
                            fast=a.fast, dbg_phase=a.dbg_phase)
    else:
        ok = run(a.dir, a.tok, a.pos, a.lanes, nlayer)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
