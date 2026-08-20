"""run_mamba_seq — THE mamba full-engine gate (doc 9 §7 rung 3).

Builds every table image from the export dir + shifts.json, drives the full
sequencer through iverilog for T tokens, and compares logits + final residual
per token against model/mamba2_fixed.FixedMamba2 (calibrated identically).
Gate: logits cosine > 0.999 per token AND argmax agreement (the assembled
engine crosses two LUT-resolution norms per layer, so bit-exactness is not
the bar here — the four cores underneath each have their own bit-exact gates).

    python -m fabric.stage3.run_mamba_seq --tokens 4
    python -m fabric.stage3.run_mamba_seq --tokens 4 --export data/fabric_mamba
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from fabric.stage3._simdir import kevbuild
from model.mamba2_fixed import FixedMamba2, scale_me

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D, DIN, NST, H, L, V = 256, 512, 64, 8, 7, 1024
CONVD, INROWS = DIN + 2 * NST, 2 * DIN + 2 * NST + H


def tool(name):
    t = shutil.which(name) or os.path.expanduser(f"~/.local/bin/{name}")
    if not os.path.exists(t):
        sys.exit(f"missing tool: {name}")
    return t


def rd16(path):
    return np.array([int(l, 16) for l in open(path) if l.strip()], dtype=np.int64)


def w32(path, vals):
    with open(path, "w") as f:
        for v in np.asarray(vals).ravel():
            f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")


def me_word(s):
    m, e = scale_me(s)
    return (e << 16) | m


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.run_mamba_seq")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--export", default="data/fabric_mamba")
    ap.add_argument("--ckpt", default="data/ckpt.mamba2.chat6q.baked.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)

    sim = args.dir or kevbuild("stage3_mamba_seq")
    os.makedirs(sim, exist_ok=True)
    ex = args.export
    sj = json.load(open(f"{ex}/shifts.json"))

    # ---- t0: gemv weight bank: in l0..6 | out l0..6 | head ----------------
    inw = [rd16(f"{ex}/l{l}_in_proj_w4.mem") for l in range(L)]
    outw = [rd16(f"{ex}/l{l}_out_proj_w4.mem") for l in range(L)]
    hdw = rd16(f"{ex}/head_w4.mem")
    gw = np.concatenate(inw + outw + [hdw])
    out_base = sum(len(a) for a in inw)
    head_base = out_base + sum(len(a) for a in outw)
    w32(f"{sim}/ms_t0.mem", gw)

    # ---- other tables -----------------------------------------------------
    def cat(fmt):
        return np.concatenate([rd16(f"{ex}/l{l}_{fmt}.mem") for l in range(L)])

    tables = {
        1: rd16(f"{ex}/tok_emb_w4.mem"),
        2: rd16(f"{ex}/tok_emb_scale.mem"),
        3: cat("conv_w"), 4: cat("conv_b"),
        5: rd16(f"{ex}/silu_lut.mem"),
        6: cat("alut"), 7: cat("dtlut"),
        8: cat("ln_g"), 9: cat("norm_g"),
        10: rd16(f"{ex}/normf_g.mem"),
        11: cat("in_proj_scale"), 12: cat("out_proj_scale"),
        13: rd16(f"{ex}/head_scale.mem"),
    }
    for s, v in tables.items():
        w32(f"{sim}/ms_t{s}.mem", v)

    # ---- consts ------------------------------------------------------------
    consts = np.zeros(128, dtype=np.uint64)
    for l in range(L):
        lj = sj["layers"][l] if "layers" in sj else sj[str(l)]
        consts[l*16 + 0] = me_word(1.0 / lj["in_scale"])
        consts[l*16 + 1] = me_word(lj["in_scale"])
        consts[l*16 + 2] = me_word(1.0 / lj["out_scale"])
        consts[l*16 + 3] = me_word(lj["out_scale"])
        consts[l*16 + 4] = (lj["bX"] << 8) | (lj["bC"] << 4) | lj["bB"]
        Dq = [int(np.clip(round(d * 8192), -32768, 32767)) & 0xFFFF
              for d in lj["D"]]
        for k in range(4):
            consts[l*16 + 5 + k] = (Dq[2*k+1] << 16) | Dq[2*k]
    hs = sj.get("head_scale") or sj["layers"][0]["head_scale"]
    consts[112] = me_word(1.0 / hs)
    consts[113] = me_word(hs)
    consts[120] = out_base
    consts[121] = head_base
    w32(f"{sim}/ms_t14.mem", consts)

    # ---- tokens + cfg -------------------------------------------------------
    val = np.fromfile("data/bpe1024_chat6/val.bin", dtype=np.uint16)
    toks = val[: args.tokens].astype(np.int64)
    with open(f"{sim}/ms_tok.mem", "w") as f:
        for t in toks:
            f.write(f"{int(t):04x}\n")
    cfg = np.zeros(16, dtype=np.int64)
    cfg[0] = args.tokens
    for s, v in tables.items():
        cfg[s] = len(v)
    cfg[14] = 128
    cfg[15] = len(gw)
    w32(f"{sim}/ms_cfg.mem", cfg)

    # ---- reference -----------------------------------------------------------
    fx = FixedMamba2(args.ckpt)
    fx.calibrate(val[:64])
    st = fx.alloc_state()
    ref_logits, ref_x = [], []
    for t in toks:
        c = {}
        lg = fx.step(int(t), st, collect=c)
        ref_logits.append(np.clip(np.round(lg * 4096), -32768, 32767))
        ref_x.append(np.clip(np.round(c[f"l{L-1}.x_out"] * (1 << 19)),
                             -(1 << 31), (1 << 31) - 1))

    # ---- simulate -------------------------------------------------------------
    src = [f"{REPO}/fabric/stage3/rtl/{m}.sv" for m in
           ("mamba_seq", "gemv_i4i8", "conv_silu", "ssm_scan_row", "rmsnorm_gated")]
    src.append(f"{REPO}/fabric/stage3/tb/tb_mamba_seq.sv")
    # rmsnorm needs seed.mem in cwd
    from fabric.stage3.run_layernorm import seed_table
    with open(f"{sim}/seed.mem", "w") as f:
        for v in seed_table():
            f.write(f"{int(v):05x}\n")
    subprocess.run([tool("iverilog"), "-g2012", "-o", "ms.vvp"] + src,
                   cwd=sim, check=True)
    out = subprocess.run([tool("vvp"), "ms.vvp"], cwd=sim, check=True,
                         capture_output=True, text=True, timeout=7200)
    if "TB_MS_DONE" not in out.stdout:
        sys.exit(f"sim incomplete:\n{out.stdout[-600:]}")
    print(out.stdout[-300:].strip())

    got_l = rd16(f"{sim}/ms_logit.out")
    got_l = np.where(got_l >= 1 << 15, got_l - (1 << 16), got_l).reshape(-1, V)
    got_x = rd16(f"{sim}/ms_x.out")
    got_x = np.where(got_x >= 1 << 31, got_x - (1 << 32), got_x).reshape(-1, D)

    ok = True
    for t in range(args.tokens):
        rl, gl = np.asarray(ref_logits[t], float), got_l[t].astype(float)
        cosl = float(rl @ gl / ((np.linalg.norm(rl) * np.linalg.norm(gl)) or 1))
        rx, gx = np.asarray(ref_x[t], float), got_x[t].astype(float)
        cosx = float(rx @ gx / ((np.linalg.norm(rx) * np.linalg.norm(gx)) or 1))
        am = int(np.argmax(rl)) == int(np.argmax(gl))
        ok &= cosl > 0.999 and am
        print(f"tok#{t} ({int(toks[t])}): logits cos {cosl:.5f}  x cos {cosx:.5f}  "
              f"argmax {'MATCH' if am else 'DIFF'} "
              f"(ref {int(np.argmax(rl))} rtl {int(np.argmax(gl))})")
    print(f"MAMBA_SEQ_VERDICT: {'PASS' if ok else 'FAIL'} "
          f"({args.tokens} tokens; bar logits cos>0.999 + argmax match)  [{sim}]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
