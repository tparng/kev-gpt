"""State-precision study: can the Mamba2 recurrent state survive INT8? (doc 8 §3.1)

The fabric pitch says "237 KB/stream at INT8, constant in T" — this probe makes
that MEASURED instead of hoped. It is the SSM twin of doc 7's R0 K8/V8 study:
teacher-forced NLL on held-out text with the state quantised at every step,
against the fp32-state baseline.

Two quantisation modes per width:
- dynamic: per-(layer, head) absmax scale recomputed every step — the quality
  upper bound, but not what cheap fabric does.
- fixed: one Q-format per layer chosen from a measured range pass (p99.9 absmax
  with 2x headroom) — shifts, not multipliers; the fabric-realistic contract.

Also prints the per-layer state dynamic-range scan (the M3 Q-format input).

    python -m model.probe_state_quant data/ckpt.mamba2.m1.pt
    python -m model.probe_state_quant data/ckpt.mamba2.m1.pt --bits 8,12,16 --chars 1024 --seqs 8
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_split
from .mamba2 import Mamba2, Mamba2Config


def quantize(t: torch.Tensor, bits: int, scale: torch.Tensor) -> torch.Tensor:
    """Symmetric mid-tread quantise t to `bits` with the given absmax scale."""
    qmax = 2 ** (bits - 1) - 1
    step = scale / qmax
    return torch.clamp(torch.round(t / step), -qmax, qmax) * step


@torch.no_grad()
def run(model, seqs, mode, bits, fixed_scales=None, collect_range=False):
    """Teacher-forced NLL over seqs; quantise every layer's ssm state each step.

    Returns (nll, range_stats). range_stats: per-layer list of absmax samples.
    """
    nll, count = 0.0, 0
    ranges = [[] for _ in model.layers] if collect_range else None
    for seq in seqs:
        states = model.alloc_state(1, seq.device)
        logits = None
        for i, tok in enumerate(seq.tolist()):
            if logits is not None:
                nll -= F.log_softmax(logits[0].float(), dim=-1)[tok].item()
                count += 1
            logits = model.step(torch.tensor([tok]), states)
            for li, st in enumerate(states):
                h = st["ssm"]
                if collect_range and i % 16 == 0:
                    # per-head absmax sample: (H,)
                    ranges[li].append(h.abs().amax(dim=(0, 2, 3)).numpy())
                if mode == "dynamic":
                    scale = h.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
                    st["ssm"].copy_(quantize(h, bits, scale))
                elif mode == "fixed-layer":
                    st["ssm"].copy_(quantize(h, bits, fixed_scales[li].max()))
                elif mode == "fixed-head":
                    st["ssm"].copy_(quantize(h, bits, fixed_scales[li]))
    return nll / count, ranges


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.probe_state_quant")
    p.add_argument("ckpt")
    p.add_argument("--bits", default="8,12,16")
    p.add_argument("--chars", type=int, default=1024, help="chars per sequence")
    p.add_argument("--seqs", type=int, default=8)
    p.add_argument("--data-dir", default="data/char")
    args = p.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Mamba2(Mamba2Config(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()

    val = load_split(args.data_dir, "val")
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(val) - args.chars - 1, args.seqs)
    seqs = [torch.from_numpy(val[s:s + args.chars].astype(np.int64)) for s in starts]
    n_pred = args.seqs * (args.chars - 1)
    print(f"{args.ckpt}: {args.seqs} x {args.chars} held-out chars "
          f"({n_pred:,} predictions)")

    base, ranges = run(model, seqs, mode="none", bits=0, collect_range=True)
    print(f"\nfp32 state baseline NLL: {base:.4f}\n")

    print("state dynamic range per layer (per-head p99-of-maxima, 2x headroom):")
    fixed_scales = []
    for li, r in enumerate(ranges):
        r = np.stack(r)                      # (samples, H)
        p99 = np.percentile(r, 99, axis=0)   # per head
        scale = torch.tensor((p99 * 2.0).astype(np.float32)).view(1, -1, 1, 1)
        fixed_scales.append(scale)
        print(f"  L{li}: layer absmax {r.max():6.2f}   head scales "
              + " ".join(f"{s:.2f}" for s in p99 * 2.0))

    print(f"\n{'width':8s} {'mode':12s} {'NLL':>8s} {'delta':>9s}")
    for bits in [int(b) for b in args.bits.split(",")]:
        for mode in ("dynamic", "fixed-head", "fixed-layer"):
            nll, _ = run(model, seqs, mode=mode, bits=bits,
                         fixed_scales=fixed_scales)
            d = (nll - base) / base * 100
            print(f"INT{bits:<5d} {mode:12s} {nll:8.4f} {d:+8.2f}%")
    print("\n(the pinnable contract wants the fixed-scale column; doc 7's R0 "
          "bar was +0.03% NLL for K8/V8)")


if __name__ == "__main__":
    main()
