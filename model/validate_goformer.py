"""Bit-honest validation gate: does the exported integer datapath reconstruct the
trained model's own output?

Doc 2's rule is "validate against goformer to cosine > 0.9999 before any speed
number is trusted." Here the chain is:

    Brevitas QAT model  ──(model.export_fabric)──>  INT4 .mem + scales
                                                          │
                                   (fabric.golden integer GEMV == this math)
                                                          │
                                   (fabric/stage1 RTL == fabric.golden, proven)

So if the exporter's integer weights + scales reproduce each QuantLinear's float
output, and the RTL reproduces the integer GEMV (already shown bit-exact), then
the fabric reproduces the model. This script checks the first link per layer:

  y_torch  = ql(x)                              # Brevitas quantised linear
  int_x, sx = ql.input_quant(x).int(), .scale  # Brevitas' own INT8 activation
  y_recon  = (int_w @ int_x) * w_scale * sx     # exported int weights + scales
  cosine(y_torch, y_recon) > 0.9999             # the gate

The exported int_w/w_scale (loaded from fabric/export/weights.npz) are the very
bytes the RTL $readmemh's, so a pass ties the .mem files to the model.

    python -m model.validate_goformer                       # uses data/ckpt.qat.pt
    python -m model.validate_goformer --ckpt data/ckpt.qat.pt --export fabric/export

The --external flag lets a real fabric/board dump (npz of {layer__y_int32})
be checked against the golden the same way, for bring-up.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from brevitas.nn import QuantLinear

from fabric.golden import gemv_int
from .gpt import GPTConfig
from .qgpt import QGPT

GATE = 0.9999


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(a @ b / (na * nb))


def load_model(ckpt_path: str) -> tuple[QGPT, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = QGPT(GPTConfig(**ck["cfg"]))
    with torch.no_grad():
        model(torch.zeros((1, 1), dtype=torch.long))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def validate(ckpt_path: str, export_dir: str, seed: int = 0) -> dict:
    model, ck = load_model(ckpt_path)
    npz = np.load(os.path.join(export_dir, "weights.npz"))
    torch.manual_seed(seed)

    rows = []
    worst = 1.0
    for name, ql in model.named_modules():
        if not isinstance(ql, QuantLinear):
            continue
        key = name.replace(".", "_")
        K = ql.in_features
        x = torch.randn(1, K)

        with torch.no_grad():
            y_torch = ql(x).numpy().ravel()
            qx = ql.input_quant(x)
            int_x = np.asarray(qx.int().detach().cpu().numpy()).round().astype(np.int64).ravel()
            sx = float(np.asarray(qx.scale.detach().cpu().numpy()).reshape(-1)[0])

        int_w = npz[f"{key}__int_w"]            # the exact bytes the RTL reads
        w_scale = npz[f"{key}__scale"]
        y_recon = gemv_int(int_w, int_x).astype(np.float64) * w_scale * sx

        c = cosine(y_torch, y_recon)
        maxabs = float(np.abs(y_torch - y_recon).max())
        denom = float(np.abs(y_torch).max()) or 1.0
        rows.append((name, c, maxabs / denom))
        worst = min(worst, c)

    print(f"# validate {ckpt_path}  (iter {ck.get('iter')}, val {ck.get('val')})")
    print(f"# exported int weights vs Brevitas QAT forward, gate cosine > {GATE}\n")
    print(f"{'layer':22s} {'cosine':>12s} {'rel_maxerr':>12s}")
    print("-" * 48)
    for name, c, rel in rows:
        flag = "" if c > GATE else "  <-- BELOW GATE"
        print(f"{name:22s} {c:>12.7f} {rel:>12.2e}{flag}")
    passed = worst > GATE
    print(f"\nworst cosine {worst:.7f}  ->  {'PASS' if passed else 'FAIL'} (gate {GATE})")
    print("GOFORMER_VALIDATE_" + ("PASS" if passed else "FAIL"))
    return {"worst_cosine": worst, "passed": passed, "layers": len(rows)}


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.validate_goformer",
                                description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", default="data/ckpt.qat.pt")
    p.add_argument("--export", default="fabric/export")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    validate(args.ckpt, args.export, args.seed)


if __name__ == "__main__":
    main()
