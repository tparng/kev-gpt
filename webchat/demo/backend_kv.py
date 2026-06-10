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
import threading
import time

# The chat GEMVs are tiny (<=1024x256): a multi-thread BLAS pool spends ~100us
# per call on thread sync where a single thread needs ~10us (MEASURED 14-thread
# vs 1-thread shootout). Must land before numpy first loads its BLAS.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from .backend import BatchOutcome, InferenceBackend, InferResult


def _fast_decoder_cls():
    """FastKVDecoder, built lazily so importing this module stays cheap off-box.

    Bit-identical arithmetic to model.goformer_kv.KVDecoder (same qlin/ln/smax
    through the same GoformerFull) with two mechanical changes the chat path
    needs — the reference stays untouched for its other consumers:
      * preallocated (block, C) K/V buffers instead of per-step vstack (drops
        the O(T) cache copy every step), and
      * truncate(L): roll the cache back to position L, so a session can keep
        the typed-prompt prefix across keystrokes instead of re-priming.
    """
    import numpy as np
    from model.goformer_kv import KVDecoder

    class FastKVDecoder(KVDecoder):
        def reset(self):
            if not hasattr(self, "_kbuf"):
                block, C = self.p["pos_emb"].shape
                nb = len(self.p["blocks"])
                self._kbuf = [np.empty((block, C)) for _ in range(nb)]
                self._vbuf = [np.empty((block, C)) for _ in range(nb)]
            self.t = 0

        def truncate(self, length):
            assert 0 <= length <= self.t
            self.t = int(length)

        def _attn_step(self, h_t, blk, bi):
            C = h_t.shape[0]
            nh, hd = self.n_head, C // self.n_head
            qkv = self.gf.qlin(h_t[None, :], blk["qkv"])[0]
            q, k, v = qkv[:C], qkv[C:2 * C], qkv[2 * C:]
            self._kbuf[bi][self.t] = k
            self._vbuf[bi][self.t] = v
            T = self.t + 1
            kc, vc = self._kbuf[bi][:T], self._vbuf[bi][:T]
            qh = q.reshape(1, nh, hd).transpose(1, 0, 2)
            kh = kc.reshape(T, nh, hd).transpose(1, 0, 2)
            vh = vc.reshape(T, nh, hd).transpose(1, 0, 2)
            scores = (qh @ kh.transpose(0, 2, 1)) / np.sqrt(hd)
            attn = self.gf._smax(scores) @ vh
            y = attn.transpose(1, 0, 2).reshape(C)
            return self.gf.qlin(y[None, :], blk["proj"])[0]

    return FastKVDecoder


class KVChatDevice:
    """Host-KV multi-token generator over a swappable GEMV backend."""

    def __init__(self, *, gemv_backend: str = "numpy", npz: str | None = None,
                 meta: str | None = None, gen_chars: int = 24,
                 greedy: bool = True, temperature: float = 0.85,
                 top_k: int = 40, seed: int = 0):
        # heavy imports kept local so importing this module is cheap off-box
        import numpy as np
        from model.goformer_full import load_params
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
        self._np = np

        p = load_params(npz)
        self._params = p
        self._block = int(p["pos_emb"].shape[0])
        self._enc, self._dec, self.vocab = load_meta(meta)
        self._be = make_backend(gemv_backend)
        self._be.preload(p)                          # stream weights once
        self._fast_cls = _fast_decoder_cls()
        self._engine = self._fast_cls(p, matmul=self._be.matmul)
        # per-client prefix-KV sessions (greedy path): client_id -> dict with
        # its own decoder + the ids/logits already primed, LRU-capped.
        self._sessions: dict = {}
        self._max_sessions = 32
        self._lock = threading.Lock()

    def _gen_one(self, prompt: str) -> str:
        ids = self._enc((prompt or "").strip().lower())
        if not ids:
            ids = self._enc("\n") or [0]
        full = self._kv_generate(self._engine, ids, self.gen_chars, self._block,
                                 temperature=self.temperature, top_k=self.top_k,
                                 rng=self._rng, greedy=self.greedy)
        text = self._dec(full[len(ids):])
        return self._tidy(text)

    @staticmethod
    def _tidy(text: str) -> str:
        """One presentable utterance from a raw char-budget continuation: first
        line only, stop at the first sentence end if one landed, otherwise drop
        the trailing partial word the hard char cut leaves behind ("…come in h")."""
        text = text.split("\n", 1)[0]
        for stop in (". ", "! ", "? "):
            i = text.find(stop)
            if i != -1:
                return text[: i + 1].rstrip()
        if text.rstrip().endswith((".", "!", "?")):
            return text.rstrip()
        cut = text.rstrip()
        i = cut.rfind(" ")
        return cut[:i].rstrip() if i > 0 else cut

    # --- per-client prefix-KV path (greedy only) --------------------------- #
    _STOPS = (". ", "! ", "? ", "\n")

    def _gen_session(self, cid, prompt: str) -> str:
        """Like _gen_one but reuses the client's primed K/V across keystrokes:
        roll back to the common prefix and step only the new chars, instead of
        re-priming the whole prompt every speculation. Display-identical to the
        stateless path (gated): generation early-stops once a _tidy boundary
        has landed, since _tidy discards everything after it anyway."""
        np = self._np
        ids = self._enc((prompt or "").strip().lower())
        if not ids:
            ids = self._enc("\n") or [0]
        # keep prompt + generation inside the position budget (pos_emb rows)
        maxp = self._block - self.gen_chars - 1
        if len(ids) > maxp:
            ids = ids[-maxp:]

        s = self._sessions.pop(cid, None)            # pop+reinsert = LRU bump
        if s is None:
            if len(self._sessions) >= self._max_sessions:
                self._sessions.pop(next(iter(self._sessions)))
            s = {"dec": self._fast_cls(self._params, matmul=self._be.matmul),
                 "ids": [], "lh": []}
        self._sessions[cid] = s
        dec, sids, lh = s["dec"], s["ids"], s["lh"]

        n_common = 0                                 # longest shared prefix
        for a, b in zip(sids, ids):
            if a != b:
                break
            n_common += 1
        if n_common == 0:
            dec.reset()
            sids.clear()
            lh.clear()
        else:
            dec.truncate(n_common)
            del sids[n_common:], lh[n_common:]
        logits = lh[-1] if lh else None
        for tok in ids[n_common:]:                   # prime only the new chars
            logits = dec.step(int(tok))
            sids.append(int(tok))
            lh.append(logits)

        text = ""
        for i in range(self.gen_chars):
            nxt = int(np.argmax(logits))
            text += self._dec([nxt])
            if any(stp in text for stp in self._STOPS):
                break                                # _tidy drops the rest
            if i + 1 < self.gen_chars:
                logits = dec.step(nxt)
        dec.truncate(len(sids))                      # drop gen K/V, keep prompt
        return self._tidy(text)

    def _gen_all(self, items):
        # items: prompts, or (client_id, prompt) pairs from the server. The
        # session path needs greedy (sampling keeps kv_generate's exact rng
        # stream through the stateless path).
        out = []
        for it in items:
            if isinstance(it, tuple):
                cid, prompt = it
                out.append(self._gen_session(cid, prompt) if self.greedy
                           else self._gen_one(prompt))
            else:
                out.append(self._gen_one(it))
        return out

    async def infer(self, prompts):
        # the GEMV loop blocks (numpy / MMIO); run it off the event loop. The
        # lock keeps overlapping batches from racing the session decoders.
        loop = asyncio.get_running_loop()
        items = list(prompts)

        def work():
            with self._lock:
                return self._gen_all(items)

        return await loop.run_in_executor(None, work)

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
        comps = await self._dev.infer([(r.client_id, r.prompt) for r in reqs])
        busy_s = time.perf_counter() - t0           # real host-KV generation cost
        results = [InferResult(client_id=r.client_id, prompt=r.prompt, seq=r.seq,
                               completion=c) for r, c in zip(reqs, comps)]
        return BatchOutcome(results=results, busy_s=busy_s,
                            filled=len(reqs), capacity=self.capacity)

    async def close(self) -> None:
        self._dev.close()
