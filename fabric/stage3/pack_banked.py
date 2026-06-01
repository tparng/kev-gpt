"""Transposed (wider-word) INT4 packing for the Stage-3 banked GEMV core.

The resident core stores weights row-major (two nibbles/byte along K). The banked
throughput core needs the TRANSPOSED layout: one wide memory word holds the same
column k of LANES consecutive output rows, so one read feeds LANES lanes sharing
the activation x[k].

    word(g, k) bit [L*4 +: 4] = nibble( W[g*LANES + L, k] )    L = 0..LANES-1
    word index in the load stream = g*K + k   (g = 0..ceil(M/LANES)-1, k = 0..K-1)

Bit-exact contract identical to fabric/golden: y[m] = sum_k W[m,k]*x[k], signed
INT4 weights, signed INT8 activations, exact int32 accumulation. Pure numpy.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def gemv_int(int_w: np.ndarray, x_int8: np.ndarray) -> np.ndarray:
    """Exact integer GEMV — the same datapath the RTL reproduces, int32 accum."""
    return np.asarray(int_w, dtype=np.int32) @ np.asarray(x_int8, dtype=np.int32)


def pack_transposed(int_w: np.ndarray, lanes: int) -> list[int]:
    """(M,K) signed INT4 -> list of (g*K+k) wide words, each LANES*4 bits.

    Rows are zero-padded up to a multiple of LANES; padded lanes contribute a 0
    weight and their outputs are never read (rd_addr < M)."""
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


def write_case(out_dir: str, M: int, K: int, lanes: int, seed: int = 0) -> np.ndarray:
    """Generate a random INT4 W and INT8 x, write w/x/y mem files, return golden y."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    int_w = rng.integers(-8, 8, size=(M, K)).astype(np.int8)      # INT4 [-8,7]
    x = rng.integers(-128, 128, size=K).astype(np.int8)           # INT8 [-128,127]
    y = gemv_int(int_w, x)                                         # int32 golden

    hexw = lanes                                                  # hex chars per word
    words = pack_transposed(int_w, lanes)
    with open(os.path.join(out_dir, "w.mem"), "w") as f:
        f.write("\n".join(format(w, f"0{hexw}x") for w in words) + "\n")
    with open(os.path.join(out_dir, "x.mem"), "w") as f:
        f.write("\n".join(format(int(v) & 0xFF, "02x") for v in x) + "\n")
    with open(os.path.join(out_dir, "y.mem"), "w") as f:
        f.write("\n".join(format(int(v) & 0xFFFFFFFF, "08x") for v in y) + "\n")
    with open(os.path.join(out_dir, "dims.json"), "w") as f:
        json.dump({"M": M, "K": K, "LANES": lanes, "seed": seed,
                   "n_words": len(words)}, f)
    return y


def check(out_dir: str) -> bool:
    """Compare the testbench dump y.out against the golden y.mem. Prints a sentinel."""
    with open(os.path.join(out_dir, "dims.json")) as f:
        dims = json.load(f)
    def load(name):
        with open(os.path.join(out_dir, name)) as f:
            return [int(line, 16) & 0xFFFFFFFF for line in f if line.strip()]
    gold = load("y.mem")
    got = load("y.out")
    M = dims["M"]
    ok = len(got) >= M and gold[:M] == got[:M]
    nmis = sum(1 for a, b in zip(gold[:M], got[:M]) if a != b) + abs(len(got) - M)
    print(f"BANKED_VERDICT bitexact={ok} mismatches={nmis}/{M} "
          f"M={M} K={dims['K']} LANES={dims['LANES']}")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.stage3.pack_banked")
    p.add_argument("cmd", choices=["gen", "check"])
    p.add_argument("--dir", required=True)
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--lanes", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "gen":
        write_case(a.dir, a.M, a.K, a.lanes, a.seed)
        print(f"GEN dir={a.dir} M={a.M} K={a.K} LANES={a.lanes}")
    else:
        ok = check(a.dir)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
