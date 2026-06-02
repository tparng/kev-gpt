"""Incremental KV-cached chat on the EXISTING resident bitstream (no new bitstream).

Why this is the guaranteed near-term win
-----------------------------------------
The old board driver (fabric/stage2/driver/pl_resident.py) calls
`GoformerFull.generate` -> `forward(idx[-block_size:])` every token. `forward`
recomputes the WHOLE context each step (O(T^2)): at T tokens deep it issues ~T x
more PL GEMV calls than necessary, and the PL GEMV call (an AXI poke-stream over
/dev/mem) is the entire cost. That is the 0.22 tok/s floor.

This module swaps that for `model.goformer_kv.KVDecoder`, which processes only the
NEW token each step and caches each layer's K/V (O(T)). The arithmetic is
bit-IDENTICAL to the full recompute (goformer_kv proves maxabsdiff=0), so the PL
GEMV stays bit-exact -- we just stop throwing away ~T x of the work. Per generated
token the matmul-call count drops from O(T) Linears-per-position x T positions to a
fixed 4*n_layer+1 Linears for the single new position.

Backends (selected by --backend):
  pl     : the real fabric (fabric.pl_gemv.PLResident over /dev/mem; needs root +
           the resident bitstream loaded). Whole model preloaded into URAM once.
  c      : same PLResident AXI map, but the activation-stream + readback inner loop
           is the compiled C MMIO driver (gemv_axi_drv via ctypes). ~10-40x faster
           per GEMV than per-register Python pokes. Math/orchestration stays Python.
  numpy  : no board; exact int32 GEMV in numpy. Used by the laptop GATE to prove the
           KV wiring emits the same token stream as the reference full-recompute.

Run on the board (see run_on_board.sh):
  sudo ~/kevweb/venv/bin/python -m fabric.stage3.board.pl_kv_chat --backend c \
       --prompt "once upon time" --n 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


# --------------------------------------------------------------------------- #
# repo import path: this file lives at fabric/stage3/board/, repo root is 3 up.
# On the board the package is laid out the same way (the board mirrors the repo).
# --------------------------------------------------------------------------- #
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from model.goformer_full import GoformerFull, load_params, numpy_matmul  # noqa: E402
from model.goformer_kv import KVDecoder  # noqa: E402


# --------------------------------------------------------------------------- #
# Backends. Each exposes preload(params) + matmul(int_w, int_x) -> int32 y, and
# is a drop-in for GoformerFull's `matmul`. They key weights by id(int_w) so the
# resident URAM image is streamed once and per-call only the activation moves.
# --------------------------------------------------------------------------- #
class NumpyResident:
    """Reference backend: behaves like PLResident (preload-by-id, then matmul) but
    computes the GEMV in numpy. The gate uses it so the KV wiring/offset logic is
    exercised WITHOUT a board, while the GEMV stays the exact int32 the PL reproduces."""

    def __init__(self):
        self._map = {}          # id(int_w) -> (w_base, M, K)
        self._w = {}            # id(int_w) -> int_w (kept so we can compute)
        self.total_bytes = 0
        self.last_cycles = 0
        self.n_matmul = 0

    def preload(self, params):
        from fabric import pack
        off = 0
        for iw in _layer_weights(params):
            iw8 = np.asarray(iw, dtype=np.int8)
            M, K = iw8.shape
            packed = pack.pack_int4_rows(iw8).ravel()
            self._map[id(iw)] = (off, int(M), int(K))
            self._w[id(iw)] = iw8
            off += int(packed.size)
        self.total_bytes = off
        return off

    def matmul(self, int_w, int_x):
        wb, M, K = self._map[id(int_w)]                 # resident: id must be known
        iw = self._w[id(int_w)]
        self.n_matmul += 1
        return iw.astype(np.int64) @ np.asarray(int_x, dtype=np.int64)


def _layer_weights(params):
    """The fixed layer order that defines URAM byte offsets (matches PLResident)."""
    for blk in params["blocks"]:
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            yield blk[nm][0]
    yield params["head"][0]


def make_backend(name):
    if name == "numpy":
        return NumpyResident()
    if name == "pl":
        from fabric.pl_gemv import PLResident
        return PLResident()
    if name == "c":
        from fabric.stage3.board.pl_resident_c import PLResidentC
        return PLResidentC()
    raise ValueError(f"unknown backend {name!r} (numpy|pl|c)")


# --------------------------------------------------------------------------- #
# meta / token coding
# --------------------------------------------------------------------------- #
def load_meta(meta_path):
    m = json.load(open(meta_path))
    stoi = m["stoi"]
    itos = {int(k): v for k, v in m["itos"].items()}
    enc = lambda s: [stoi.get(c, 0) for c in s]
    dec = lambda ids: "".join(itos[int(i)] for i in ids)
    return enc, dec, m["vocab_size"]


def kv_generate(dec_engine, prompt_ids, n_tokens, block_size,
                temperature=0.85, top_k=40, rng=None, greedy=False):
    """KV-cached generation that matches GoformerFull.generate's sampling exactly
    when greedy=False, or argmax when greedy=True. Returns the full id list."""
    from model.goformer_full import softmax
    rng = rng or np.random.default_rng(0)
    out = list(prompt_ids)
    dec_engine.reset()
    logits = None
    for tok in out:                                     # prime the K/V caches
        logits = dec_engine.step(int(tok))
    for _ in range(n_tokens):
        if logits is None:                              # empty prompt guard
            logits = dec_engine.step(0)
            out.append(0)
        if greedy:
            nxt = int(np.argmax(logits))
        else:
            lg = logits / max(temperature, 1e-6)
            if top_k:
                kth = np.sort(lg)[-min(top_k, len(lg))]
                lg = np.where(lg < kth, -np.inf, lg)
            probs = softmax(lg)
            nxt = int(rng.choice(len(probs), p=probs))
        out.append(nxt)
        logits = dec_engine.step(nxt)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="KV-cached PL chat (resident bitstream)")
    ap.add_argument("--backend", default="numpy", choices=("numpy", "pl", "c"))
    ap.add_argument("--npz", default=os.path.join(_REPO, "fabric", "export", "goformer.npz"))
    ap.add_argument("--meta", default=os.path.join(_REPO, "fabric", "export", "goformer_meta.json"))
    ap.add_argument("--prompt", default="once upon time")
    ap.add_argument("--n", type=int, default=40, help="tokens to generate")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    p = load_params(args.npz)
    enc, dec, vocab = load_meta(args.meta)
    backend = make_backend(args.backend)

    t0 = time.time()
    nb = backend.preload(p)
    t_pre = time.time() - t0
    print(f"[preload] {nb} bytes ({nb/1024:.0f} KB) into URAM in {t_pre:.2f}s (one-time)")

    engine = KVDecoder(p, matmul=backend.matmul)        # KV cache over the PL GEMV
    prompt_ids = enc(args.prompt)
    print(f"[prompt] {args.prompt!r} -> {len(prompt_ids)} tokens, generating {args.n}")

    t0 = time.time()
    out = kv_generate(engine, prompt_ids, args.n, p["block_size"],
                      temperature=args.temperature, top_k=args.top_k,
                      rng=np.random.default_rng(args.seed), greedy=args.greedy)
    dt = time.time() - t0
    n_gen = len(out) - len(prompt_ids)
    tok_s = n_gen / dt if dt > 0 else float("inf")

    print("KEVIN:", dec(out))
    print(f"[speed] {dt:.2f}s / {n_gen} tok = {dt/max(n_gen,1)*1000:.1f} ms/tok "
          f"= {tok_s:.2f} tok/s  (backend={args.backend})")
    if hasattr(backend, "last_cycles") and backend.last_cycles:
        print(f"[pl] last GEMV {backend.last_cycles} cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
