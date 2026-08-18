"""PTQ gate: does the trained mamba2 survive the fabric quantization contract?

Fake-quantizes a checkpoint to the doc-9 contract — per-output-channel
symmetric INT4 weights on every matrix, INT8 saturating activations at the
GEMV boundaries — and measures what it costs: plain val CE and the held-out
recall probe, FP vs PTQ. This decides whether the fabric model is a free
export (like the transformer's goformer path hoped) or needs a QAT retrain
(like the transformer actually did).

    python -m model.probe_ptq data/ckpt.mamba2.chat6.pt
    python -m model.probe_ptq data/ckpt.mamba2.chat6.pt --wbits 4 --abits 8
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from .chat_demo import load
from .train import pick_device


def quant_weight_(w: torch.Tensor, bits: int):
    """Per-output-channel symmetric fake-quant, in place. w: (out, in)."""
    qmax = (1 << (bits - 1)) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    w.copy_((w / scale).round().clamp(-qmax - 1, qmax) * scale)


class ActQuant(torch.nn.Module):
    """INT8 saturating activation fake-quant with a calibrated absmax."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.register_buffer("absmax", torch.tensor(0.0))
        self.calibrating = True

    def forward(self, x):
        if self.calibrating:
            self.absmax.fill_(max(float(self.absmax), float(x.abs().max())))
        else:
            s = self.absmax.clamp(min=1e-8) / 127.0
            x = (x / s).round().clamp(-128, 127) * s
        return self.inner(x)


def apply_ptq(model, wbits: int, abits: int):
    n_w = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            quant_weight_(mod.weight.data, wbits)
            n_w += 1
    if abits:
        for layer in model.layers:
            layer.mixer.in_proj = ActQuant(layer.mixer.in_proj)
            layer.mixer.out_proj = ActQuant(layer.mixer.out_proj)
        model.head = ActQuant(model.head)
    return n_w


def set_calibrating(model, flag: bool):
    for m in model.modules():
        if isinstance(m, ActQuant):
            m.calibrating = flag


def val_ce(model, val, device, iters=40, batch=8, block=512, seed=7):
    g = torch.Generator().manual_seed(seed)
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            ix = torch.randint(len(val) - block - 1, (batch,), generator=g)
            x = torch.stack([torch.from_numpy(val[i:i + block].astype(np.int64))
                             for i in ix]).to(device)
            y = torch.stack([torch.from_numpy(val[i + 1:i + block + 1].astype(np.int64))
                             for i in ix]).to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.probe_ptq")
    p.add_argument("ckpt")
    p.add_argument("--wbits", type=int, default=4)
    p.add_argument("--abits", type=int, default=8)
    p.add_argument("--val-bin", default="data/bpe1024_chat6/val.bin")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    val = np.fromfile(args.val_bin, dtype=np.uint16)

    model, meta, it, best = load(args.ckpt, device)
    model.cfg.z_loss = 0.0
    fp_ce = val_ce(model, val, device)
    print(f"# {args.ckpt} (iter {it})  FP val CE {fp_ce:.4f}")

    qmodel = copy.deepcopy(model)
    n_w = apply_ptq(qmodel, args.wbits, args.abits)
    if args.abits:
        set_calibrating(qmodel, True)
        val_ce(qmodel, val, device, iters=8)      # calibration pass
        set_calibrating(qmodel, False)
    q_ce = val_ce(qmodel, val, device)
    dpct = (q_ce - fp_ce) / fp_ce * 100
    print(f"PTQ W{args.wbits}A{args.abits} ({n_w} matrices): val CE {q_ce:.4f}  "
          f"(+{q_ce - fp_ce:.4f}, {dpct:+.2f}%)")
    verdict = ("FREE" if dpct < 1.0 else
               "ACCEPTABLE" if dpct < 3.0 else "NEEDS QAT")
    print(f"PTQ_VERDICT: {verdict} (+1% free / +3% acceptable thresholds)")


if __name__ == "__main__":
    main()
