"""backend_kv.py — model-faithful MULTI-TOKEN chat (real Kevin, not T=1 stub).

Where PLSingleTokenBackend is first-token-faithful only, this produces genuine
telegraphic Kevin: model.goformer_kv.KVDecoder keeps the per-stream K/V on the
host and the chosen GEMV backend does the per-position matmuls --

  numpy : no board, exact int32 GEMV (the off-box correctness/demo path)
  c     : the Kria's compiled MMIO driver over /dev/mem (fast per-GEMV)
  pl    : the Kria's per-register PL driver (slow but pure)

HONEST SCOPE (5-demo-prd.md gap 1): this is the HOST-ORCHESTRATED KV path. It is
NOT the 60k single-token fabric pass -- that bitstream (sequencer_sb, pl_seq_sb)
does one forward per GO and persists no K/V between calls. A fully on-chip
multi-token KV decode at fabric speed is the unbuilt P1 RTL. So this path serves
real text with the on-chip INT4 weights doing the matmuls, at the heterogeneous
(host-in-the-loop) rate, not at 60k.

Two shapes of the same engine:
  KVChatDevice  -- .infer(prompts) -> completions; what the A53 daemon drives
                   (a53_daemon --engine kv).
  KVChatBackend -- an InferenceBackend; what the server drives directly
                   (server --backend kv) for a single-box deployment.
"""

from __future__ import annotations

import asyncio
import os
import time

from .backend import BatchOutcome, InferenceBackend, InferResult


class KVChatDevice:
    """Host-KV multi-token generator over a swappable GEMV backend."""

    def __init__(self, *, gemv_backend: str = "numpy", npz: str | None = None,
                 meta: str | None = None, gen_chars: int = 24,
                 greedy: bool = True, temperature: float = 0.85,
                 top_k: int = 40, seed: int = 0):
        # heavy imports kept local so importing this module is cheap off-box
        import numpy as np
        from model.goformer_full import load_params
        from model.goformer_kv import KVDecoder
        from fabric.stage3.board.pl_kv_chat import (kv_generate, load_meta,
                                                    make_backend)

        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        npz = npz or os.path.join(repo, "fabric", "export", "goformer.npz")
        meta = meta or os.path.join(repo, "fabric", "export", "goformer_meta.json")

        self._kv_generate = kv_generate
        self.gen_chars = int(gen_chars)
        self.greedy = bool(greedy)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self._rng = np.random.default_rng(seed)

        p = load_params(npz)
        self._block = int(p["pos_emb"].shape[0])
        self._enc, self._dec, self.vocab = load_meta(meta)
        self._be = make_backend(gemv_backend)
        self._be.preload(p)                          # stream weights once
        self._engine = KVDecoder(p, matmul=self._be.matmul)

    def _gen_one(self, prompt: str) -> str:
        ids = self._enc((prompt or "").strip().lower())
        if not ids:
            ids = self._enc("\n") or [0]
        full = self._kv_generate(self._engine, ids, self.gen_chars, self._block,
                                 temperature=self.temperature, top_k=self.top_k,
                                 rng=self._rng, greedy=self.greedy)
        text = self._dec(full[len(ids):])
        return text.split("\n", 1)[0]                # one utterance per reply

    def _gen_all(self, prompts):
        # one KVDecoder reused per prompt — kv_generate resets it each call, so
        # streams don't leak K/V into each other.
        return [self._gen_one(p) for p in prompts]

    async def infer(self, prompts):
        # the GEMV loop blocks (numpy / MMIO); run it off the event loop
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._gen_all, list(prompts))

    # --- parity with PLDevice so a53_daemon can treat them interchangeably ---
    @property
    def launches(self) -> int:
        return int(getattr(self._be, "n_matmul", 0))

    def submit_raw(self, tok_ids):
        raise NotImplementedError("--bench measures the T=1 PL pass; use --engine t1")

    def close(self):
        pass


class KVChatBackend(InferenceBackend):
    """InferenceBackend wrapper so the server can serve real Kevin directly."""

    def __init__(self, *, gemv_backend: str = "numpy", capacity: int = 16, **kw):
        self.capacity = capacity
        self._dev = KVChatDevice(gemv_backend=gemv_backend, **kw)

    async def infer_batch(self, reqs):
        t0 = time.perf_counter()
        comps = await self._dev.infer([r.prompt for r in reqs])
        busy_s = time.perf_counter() - t0           # real host-KV generation cost
        results = [InferResult(client_id=r.client_id, prompt=r.prompt, seq=r.seq,
                               completion=c) for r, c in zip(reqs, comps)]
        return BatchOutcome(results=results, busy_s=busy_s,
                            filled=len(reqs), capacity=self.capacity)

    async def close(self) -> None:
        self._dev.close()
