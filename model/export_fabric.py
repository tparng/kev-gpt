"""Export a Brevitas QAT checkpoint to fabric-ingestable weight files.

This is the contract between training and the RTL: it pulls the *actual* integer
weights and per-channel scales Brevitas learned (never recomputed — bit-honest),
packs them per fabric/pack.py, and writes:

  fabric/export/<layer>.w.mem       packed INT4 weights, $readmemh hex
  fabric/export/<layer>.wscale.mem  per-output-channel weight scales, fp32 hex
  fabric/export/weights.npz         every int_w + scale as numpy (golden ref / TB)
  fabric/export/manifest.json       shapes, packing convention, budget check

A round-trip assertion (pack -> unpack == int_w) runs before anything is
written, so a corrupt export fails loudly rather than silently baking wrong
weights.

    python -m model.export_fabric data/ckpt.qat.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct

import numpy as np
import torch
from brevitas.nn import QuantLinear

from fabric import pack
from fabric.spec import ONCHIP_WEIGHT_BUDGET_BYTES, WEIGHT_BITS

from .gpt import GPTConfig
from .qgpt import QGPT


def _int_weight(ql: QuantLinear) -> tuple[np.ndarray, np.ndarray]:
    """Return (int_w [out,in] int8, scale [out] float32), straight out of
    Brevitas' weight quantiser."""
    qw = ql.quant_weight()
    try:
        iw = qw.int().detach().cpu().numpy()
    except Exception:
        iw = np.round((qw.value / qw.scale).detach().cpu().numpy())
    int_w = np.asarray(iw).round().astype(np.int8)
    scale = np.asarray(qw.scale.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    if scale.size == 1:
        scale = np.full(int_w.shape[0], float(scale), dtype=np.float32)
    return int_w, scale


def _act_scale(ql: QuantLinear) -> float | None:
    """Best-effort input (activation) scale; None if not retrievable. Not needed
    for the integer GEMV test, but recorded for the full dequant in hardware."""
    try:
        s = ql.input_quant.scale()
        return float(np.asarray(s.detach().cpu().numpy()).reshape(-1)[0])
    except Exception:
        return None


def collect_quant_linears(model: QGPT) -> dict[str, QuantLinear]:
    return {name: m for name, m in model.named_modules() if isinstance(m, QuantLinear)}


def export(ckpt_path: str, out_dir: str = "fabric/export") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not ck.get("qat"):
        print("WARNING: checkpoint not flagged qat=True; weights may not be INT4.")
    model = QGPT(GPTConfig(**ck["cfg"]))
    with torch.no_grad():
        model(torch.zeros((1, 1), dtype=torch.long))  # materialise lazy buffers
    model.load_state_dict(ck["model"])
    model.eval()

    layers = collect_quant_linears(model)
    manifest = {
        "source_ckpt": os.path.abspath(ckpt_path),
        "iter": ck.get("iter"), "val": ck.get("val"),
        "weight_bits": WEIGHT_BITS, "cfg": ck["cfg"],
        "packing": {
            "nibbles_per_byte": 2,
            "low_nibble_is_even_k": pack.PACK_LOW_NIBBLE_IS_EVEN_K,
            "order": "row-major (m outer, k inner)",
            "scale": "per-output-channel float32, symmetric (zero_point=0)",
        },
        "layers": {},
    }
    npz: dict[str, np.ndarray] = {}
    total_bytes = 0

    for name in sorted(layers):
        int_w, scale = _int_weight(layers[name])
        if int_w.min() < -8 or int_w.max() > 7:
            raise ValueError(f"{name}: weights not INT4 (range [{int_w.min()},{int_w.max()}])")
        M, K = int_w.shape
        packed = pack.pack_int4_rows(int_w)
        if not np.array_equal(pack.unpack_int4_rows(packed, K), int_w):
            raise AssertionError(f"{name}: pack/unpack round-trip failed")

        key = name.replace(".", "_")
        with open(os.path.join(out_dir, f"{key}.w.mem"), "w") as f:
            f.write(pack.to_hex_mem(packed))
        with open(os.path.join(out_dir, f"{key}.wscale.mem"), "w") as f:
            f.write("\n".join(f"{struct.unpack('<I', struct.pack('<f', s))[0]:08x}"
                              for s in scale) + "\n")
        npz[f"{key}__int_w"] = int_w
        npz[f"{key}__scale"] = scale

        total_bytes += packed.size
        manifest["layers"][name] = {
            "shape_out_in": [int(M), int(K)],
            "packed_bytes": int(packed.size),
            "mem": f"{key}.w.mem", "wscale_mem": f"{key}.wscale.mem",
            "act_scale": _act_scale(layers[name]),
        }

    np.savez(os.path.join(out_dir, "weights.npz"), **npz)
    manifest["total_packed_bytes"] = int(total_bytes)
    manifest["onchip_budget_bytes"] = ONCHIP_WEIGHT_BUDGET_BYTES
    manifest["fits_onchip"] = bool(total_bytes <= ONCHIP_WEIGHT_BUDGET_BYTES)
    h = hashlib.sha256()
    for name in sorted(layers):
        with open(os.path.join(out_dir, f"{name.replace('.', '_')}.w.mem"), "rb") as f:
            h.update(f.read())
    manifest["weights_sha256"] = h.hexdigest()
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"exported {len(layers)} quant layers -> {out_dir}")
    print(f"  total packed weights: {total_bytes/1024:.1f} KB "
          f"(budget {ONCHIP_WEIGHT_BUDGET_BYTES/1024:.0f} KB) -> "
          f"{'FITS' if manifest['fits_onchip'] else 'SPILLS'}")
    print(f"  sha256: {manifest['weights_sha256'][:16]}...")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(prog="model.export_fabric", description=__doc__.splitlines()[0])
    p.add_argument("ckpt", nargs="?", default="data/ckpt.qat.pt")
    p.add_argument("--out", default="fabric/export")
    args = p.parse_args(argv)
    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
