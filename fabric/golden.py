"""The numpy golden reference — doc 2's "goformer" role for the integer datapath.

Bit-honest before fast: the systolic GEMV in RTL must reproduce these integer
accumulations exactly before any speed number is trusted. Pure numpy, so it is
the unambiguous source of truth the testbench checks against.

Scope is the integer GEMV that stage 1 builds:

    y_int32[m] = sum_k  W_int4[m, k] * x_int8[k]            (exact, no rounding)
    y_fp[m]    = y_int32[m] * w_scale[m] * x_scale          (dequant)

It loads the exporter's output, generates deterministic INT8 activation vectors,
and dumps x.mem + y.mem next to the weights so the Verilog testbench reads
identical bytes. The full multi-layer transformer golden (attention/softmax/
RMSNorm) is stage-3 work and layers on top of this.

    python -m fabric.golden --all
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import pack
from .spec import ACT_QMAX, ACT_QMIN


def load_export(out_dir: str = "fabric/export"):
    npz = np.load(os.path.join(out_dir, "weights.npz"))
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    return npz, manifest


def gemv_int(int_w: np.ndarray, x_int8: np.ndarray) -> np.ndarray:
    """Exact integer matrix-vector product, the RTL's datapath, int32 accum.
    int_w (M,K) INT4 in [-8,7]; x_int8 (K,) INT8. K<=1024,|W|<=8,|x|<=128 ->
    |acc| <= ~1.05e6, comfortably inside int32."""
    int_w = np.asarray(int_w, dtype=np.int32)
    x = np.asarray(x_int8, dtype=np.int32)
    if x.shape[0] != int_w.shape[1]:
        raise ValueError(f"K mismatch: W K={int_w.shape[1]}, x {x.shape[0]}")
    return int_w @ x


def gemv_from_mem(out_dir: str, layer_key: str, x_int8: np.ndarray) -> np.ndarray:
    """Unpack the packed $readmemh bytes and GEMV — proves the RTL's input bytes
    are the same weights the golden uses."""
    _, manifest = load_export(out_dir)
    name = next(n for n in manifest["layers"] if n.replace(".", "_") == layer_key)
    M, K = manifest["layers"][name]["shape_out_in"]
    mem_path = os.path.join(out_dir, manifest["layers"][name]["mem"])
    with open(mem_path) as f:
        bytes_hex = [int(line, 16) for line in f if line.strip()]
    packed = np.array(bytes_hex, dtype=np.uint8).reshape(M, K // 2)
    return gemv_int(pack.unpack_int4_rows(packed, K), x_int8)


def write_gemv_vectors(out_dir: str, layer_key: str, seed: int = 0):
    """Deterministic INT8 activation -> golden int32 output; write x.mem + y.mem
    + dims.json for the testbench. Returns (x_int8, y_int32)."""
    npz, manifest = load_export(out_dir)
    name = next(n for n in manifest["layers"] if n.replace(".", "_") == layer_key)
    M, K = manifest["layers"][name]["shape_out_in"]
    int_w = npz[f"{layer_key}__int_w"]

    rng = np.random.default_rng(seed)
    x_int8 = rng.integers(ACT_QMIN, ACT_QMAX + 1, size=K).astype(np.int8)
    y_int32 = gemv_int(int_w, x_int8)

    tb_dir = os.path.join(out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    with open(os.path.join(tb_dir, f"{layer_key}.x.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFF:02x}" for v in x_int8) + "\n")
    with open(os.path.join(tb_dir, f"{layer_key}.y.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in y_int32) + "\n")
    with open(os.path.join(tb_dir, f"{layer_key}.dims.json"), "w") as f:
        json.dump({"M": int(M), "K": int(K), "layer": name, "seed": seed}, f)
    return x_int8, y_int32


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.golden", description=__doc__.splitlines()[0])
    p.add_argument("--out", default="fabric/export")
    p.add_argument("--layer", default="blocks_0_qkv")
    p.add_argument("--all", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    _, manifest = load_export(args.out)
    keys = ([n.replace(".", "_") for n in manifest["layers"]] if args.all else [args.layer])
    n_ok = 0
    for k in keys:
        x, y = write_gemv_vectors(args.out, k, args.seed)
        ok = np.array_equal(y, gemv_from_mem(args.out, k, x))
        n_ok += ok
        print(f"GOLDEN {k:22s} K={x.shape[0]:4d} M={y.shape[0]:4d} "
              f"y[min,max]=[{int(y.min())},{int(y.max())}] mem_match={ok}")
    print(f"SUMMARY mem_match {n_ok}/{len(keys)}")


if __name__ == "__main__":
    main()
