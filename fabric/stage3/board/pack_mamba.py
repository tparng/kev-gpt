"""pack_mamba — laptop-side prep for pl_mamba: computes the reference logits
for the gate tokens (same calibration/recipe as run_mamba_seq) and writes
ms_ref.npz next to the ms_t*.mem images in the sim dir. Run after a green
run_mamba_seq, then scp the sim dir's ms_* files to the Kria.

  python -m fabric.stage3.board.pack_mamba --tokens 2
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from fabric.stage3._simdir import kevbuild
from model.mamba2_fixed import FixedMamba2


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pack_mamba")
    ap.add_argument("--tokens", type=int, default=2)
    ap.add_argument("--ckpt", default="data/ckpt.mamba2.chat6q.baked.pt")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)

    sim = args.dir or kevbuild("stage3_mamba_seq")
    val = np.fromfile("data/bpe1024_chat6/val.bin", dtype=np.uint16)
    toks = val[: args.tokens].astype(np.int64)

    fx = FixedMamba2(args.ckpt)
    fx.calibrate(val[:64])
    st = fx.alloc_state()
    ref_logits = []
    for t in toks:
        lg = fx.step(int(t), st)
        ref_logits.append(np.clip(np.round(lg * 4096), -32768, 32767))

    out = os.path.join(sim, "ms_ref.npz")
    np.savez(out, toks=toks, ref_logits=np.array(ref_logits, dtype=np.int64))
    print(f"wrote {out} ({args.tokens} tokens)")


if __name__ == "__main__":
    main()
