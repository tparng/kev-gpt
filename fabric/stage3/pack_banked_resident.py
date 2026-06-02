"""Transposed (wider-word) RESIDENT INT4 packing for gemv_banked_resident.

Same transposed wide-word layout as fabric/stage3/pack_banked, but for the
RESIDENT core: the WHOLE model's transposed weights live in one big URAM and a
per-layer w_base WORD offset selects the layer. Word units are wide words — one
transposed column of LANES consecutive output rows:

    word(layer, g, k) = w_base[layer] + g*K_layer + k          (word units)
    word(g,k) bit [L*4 +: 4] = nibble( W[g*LANES + L, k] )      L = 0..LANES-1

w_base[layer] is the cumulative wide-word count of every earlier layer:
    words_per_layer(M,K) = ceil(M/LANES) * K
    w_base[0] = 0;  w_base[i] = w_base[i-1] + words_per_layer(M_{i-1}, K_{i-1})

Bit-exact contract identical to fabric/golden: y[m] = sum_k W[m,k]*x[k], signed
INT4 weights, signed INT8 activations, exact int32 accumulation. Pure numpy.

The on-board host driver (PLBankedResident, follow-up) preloads all layers' words
once, records w_base per layer, then per matmul writes W_BASE+M+K, streams x, runs.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def gemv_int(int_w: np.ndarray, x_int8: np.ndarray) -> np.ndarray:
    """Exact integer GEMV — the same datapath the RTL reproduces, int32 accum."""
    return np.asarray(int_w, dtype=np.int32) @ np.asarray(x_int8, dtype=np.int32)


def words_per_layer(M: int, K: int, lanes: int) -> int:
    groups = (M + lanes - 1) // lanes
    return groups * K


def pack_transposed(int_w: np.ndarray, lanes: int) -> list[int]:
    """(M,K) signed INT4 -> list of (g*K+k) wide words, each LANES*4 bits.

    Rows are zero-padded up to a multiple of LANES; padded lanes contribute a 0
    weight and their outputs are never read (rd_addr < M). Identical to
    pack_banked.pack_transposed — kept here so this module stands alone."""
    int_w = np.asarray(int_w, dtype=np.int8)
    M, K = int_w.shape
    groups = (M + lanes - 1) // lanes
    pad = groups * lanes - M
    if pad:
        int_w = np.vstack([int_w, np.zeros((pad, K), dtype=np.int8)])
    words: list[int] = []
    for g in range(groups):
        rows = int_w[g * lanes:(g + 1) * lanes]          # (LANES, K)
        for k in range(K):
            w = 0
            col = rows[:, k]
            for L in range(lanes):
                w |= (int(col[L]) & 0xF) << (L * 4)
            words.append(w)
    return words


def build_resident(layers: list[np.ndarray], lanes: int):
    """layers: list of (M,K) signed INT4 arrays in MODEL ORDER.
    Returns (all_words, layer_meta) where layer_meta[i] = dict(w_base, M, K, n_words)
    and all_words is the concatenated wide-word stream (w_base = running index)."""
    all_words: list[int] = []
    meta = []
    wbase = 0
    for iw in layers:
        iw = np.asarray(iw, dtype=np.int8)
        M, K = int(iw.shape[0]), int(iw.shape[1])
        words = pack_transposed(iw, lanes)
        meta.append({"w_base": wbase, "M": M, "K": K, "n_words": len(words)})
        all_words.extend(words)
        wbase += len(words)
        assert len(words) == words_per_layer(M, K, lanes)
    return all_words, meta


def write_case(out_dir: str, shapes: list[tuple[int, int]], lanes: int, seed: int = 0):
    """Generate random INT4 W + INT8 x PER LAYER, write the concatenated resident
    weight stream (w.mem), per-layer x/y mem files, and a manifest. Returns meta.

    Each layer gets its own random W and its own random x; the gate runs every
    layer through the SAME resident core at its w_base offset and checks bit-exact
    output — proving the offset addressing is correct across >=2 distinct layers."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    layers_w, layers_x, layers_y = [], [], []
    for (M, K) in shapes:
        iw = rng.integers(-8, 8, size=(M, K)).astype(np.int8)     # INT4 [-8,7]
        x = rng.integers(-128, 128, size=K).astype(np.int8)       # INT8 [-128,127]
        layers_w.append(iw)
        layers_x.append(x)
        layers_y.append(gemv_int(iw, x))

    all_words, meta = build_resident(layers_w, lanes)
    hexw = lanes  # 4 bits/lane -> lanes/4 bytes -> lanes hex chars per word
    with open(os.path.join(out_dir, "w.mem"), "w") as f:
        f.write("\n".join(format(w, f"0{hexw}x") for w in all_words) + "\n")

    for i, (x, y) in enumerate(zip(layers_x, layers_y)):
        with open(os.path.join(out_dir, f"x{i}.mem"), "w") as f:
            f.write("\n".join(format(int(v) & 0xFF, "02x") for v in x) + "\n")
        with open(os.path.join(out_dir, f"y{i}.mem"), "w") as f:
            f.write("\n".join(format(int(v) & 0xFFFFFFFF, "08x") for v in y) + "\n")

    manifest = {"LANES": lanes, "seed": seed, "n_layers": len(shapes),
                "n_words_total": len(all_words), "layers": meta}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def check(out_dir: str) -> bool:
    """Compare each layer's testbench dump y{i}.out vs golden y{i}.mem at its
    w_base. Prints ONE sentinel line aggregating all layers/offsets."""
    with open(os.path.join(out_dir, "manifest.json")) as f:
        man = json.load(f)

    def load(name):
        with open(os.path.join(out_dir, name)) as f:
            return [int(line, 16) & 0xFFFFFFFF for line in f if line.strip()]

    total_mis = 0
    total_m = 0
    all_ok = True
    per_layer = []
    for i, lm in enumerate(man["layers"]):
        M = lm["M"]
        gold = load(f"y{i}.mem")
        got = load(f"y{i}.out")
        ok = len(got) >= M and gold[:M] == got[:M]
        nmis = sum(1 for a, b in zip(gold[:M], got[:M]) if a != b) + abs(len(got) - M)
        per_layer.append((i, lm["w_base"], M, lm["K"], ok, nmis))
        total_mis += nmis
        total_m += M
        all_ok = all_ok and ok

    for (i, wb, M, K, ok, nmis) in per_layer:
        print(f"  layer{i}: w_base={wb:6d} M={M:5d} K={K:5d} ok={ok} mismatches={nmis}")
    print(f"BANKED_RESIDENT_VERDICT bitexact={all_ok} mismatches={total_mis} "
          f"n_layers={man['n_layers']} LANES={man['LANES']} total_elems={total_m}")
    return all_ok


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.stage3.pack_banked_resident")
    p.add_argument("cmd", choices=["gen", "check"])
    p.add_argument("--dir", required=True)
    p.add_argument("--lanes", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    # shapes default mirrors the smallest real-model proportions (qkv-ish + proj-ish)
    p.add_argument("--shapes", default="768x256,256x256",
                   help="comma list of MxK layer shapes")
    a = p.parse_args(argv)
    shapes = [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")]
    if a.cmd == "gen":
        man = write_case(a.dir, shapes, a.lanes, a.seed)
        print(f"GEN dir={a.dir} layers={man['n_layers']} "
              f"n_words_total={man['n_words_total']} LANES={a.lanes}")
    else:
        ok = check(a.dir)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
