"""Incremental KV-cached decode for goformer — the prerequisite for 10k+ tok/s.

`goformer_full` recomputes the WHOLE context every token (O(T^2)): at T=256 that
is ~256x the necessary work, and it makes the hardware sequencer impossible. But
generation only needs the LAST position's logits, and by causality a position's
key/value never change once computed. So we cache each layer's K and V and process
only the new token each step (O(T)).

This is bit-IDENTICAL to the full-recompute forward — same float ops, same integer
GEMV, just no wasted recomputation. The gate below proves it: for every prefix of a
random prompt, the KV-cached last-token logits equal `goformer_full.forward(prefix)[-1]`
to maxabsdiff = 0, and greedy generation emits the same token stream.

    python -m model.goformer_kv          # KV == full-recompute, maxabsdiff=0
"""

from __future__ import annotations

import numpy as np

from .goformer_full import GoformerFull, gelu, layernorm, numpy_matmul, softmax


class KVDecoder:
    """Streaming decoder over a GoformerFull's params. step(id) -> logits for the
    new position, maintaining per-layer K/V caches. Reuses GoformerFull's exact
    qlin/layernorm/gelu/softmax so the arithmetic matches the full forward."""

    def __init__(self, params=None, matmul=numpy_matmul, engine=None):
        self.gf = engine if engine is not None else GoformerFull(params, matmul)
        self.p = self.gf.p
        self.n_head = self.p["n_head"]
        self.reset()

    def reset(self):
        nb = len(self.p["blocks"])
        self.k_cache = [None] * nb     # each: (t, C) appended row per step
        self.v_cache = [None] * nb
        self.t = 0

    def _attn_step(self, h_t, blk, bi):
        """h_t: (C,) = ln1 of the new token. Returns proj output (C,)."""
        C = h_t.shape[0]
        nh, hd = self.n_head, C // self.n_head
        qkv = self.gf.qlin(h_t[None, :], blk["qkv"])[0]          # (3C,)
        q, k, v = qkv[:C], qkv[C:2 * C], qkv[2 * C:]

        kc = k[None, :] if self.k_cache[bi] is None else np.vstack([self.k_cache[bi], k[None, :]])
        vc = v[None, :] if self.v_cache[bi] is None else np.vstack([self.v_cache[bi], v[None, :]])
        self.k_cache[bi], self.v_cache[bi] = kc, vc              # (T, C)
        T = kc.shape[0]

        qh = q.reshape(1, nh, hd).transpose(1, 0, 2)             # (nh, 1, hd)
        kh = kc.reshape(T, nh, hd).transpose(1, 0, 2)            # (nh, T, hd)
        vh = vc.reshape(T, nh, hd).transpose(1, 0, 2)            # (nh, T, hd)
        scores = (qh @ kh.transpose(0, 2, 1)) / np.sqrt(hd)      # (nh, 1, T) — no mask: all causal
        attn = self.gf._smax(scores) @ vh                        # (nh, 1, hd)
        y = attn.transpose(1, 0, 2).reshape(C)                   # (C,)
        return self.gf.qlin(y[None, :], blk["proj"])[0]

    def step(self, idx_t: int) -> np.ndarray:
        """Feed one token id, return its logits (vocab,). Advances the cache."""
        p = self.p
        x = p["tok_emb"][idx_t] + p["pos_emb"][self.t]           # (C,)
        for bi, blk in enumerate(p["blocks"]):
            x = x + self._attn_step(self.gf._ln(x, blk["ln1"]), blk, bi)
            x = x + self.gf._mlp(self.gf._ln(x, blk["ln2"])[None, :], blk)[0]
        x = self.gf._ln(x, p["ln_f"])
        self.t += 1
        return self.gf.qlin(x[None, :], p["head"])[0]

    def generate_greedy(self, idx, n_tokens, block_size):
        out = list(idx)
        self.reset()
        for tok in out:                                          # prime the cache
            logits = self.step(tok)
        for _ in range(n_tokens):
            nxt = int(np.argmax(logits))
            out.append(nxt)
            logits = self.step(nxt)
        return out


# --- a random, checkpoint-free params dict for the equivalence gate -----------
def build_random_params(seed=0, n_layer=4, n_head=4, d=256, d_mlp=1024, vocab=193):
    rng = np.random.default_rng(seed)

    def qlayer(M, K):
        int_w = rng.integers(-8, 8, size=(M, K)).astype(np.int64)
        w_scale = (rng.random(M) * 0.05 + 0.01).astype(np.float64)
        act_scale = float(rng.random() * 0.05 + 0.02)
        return (int_w, w_scale, act_scale)

    p = {
        "tok_emb": rng.standard_normal((vocab, d)) * 0.1,
        "pos_emb": rng.standard_normal((256, d)) * 0.1,
        "ln_f": (rng.random(d) * 0.4 + 0.8),
        "n_head": n_head,
        "block_size": 256,
        "head": qlayer(vocab, d),
        "blocks": [],
    }
    for _ in range(n_layer):
        p["blocks"].append({
            "ln1": (rng.random(d) * 0.4 + 0.8),
            "ln2": (rng.random(d) * 0.4 + 0.8),
            "qkv": qlayer(3 * d, d),
            "proj": qlayer(d, d),
            "mlp_fc": qlayer(d_mlp, d),
            "mlp_proj": qlayer(d, d_mlp),
        })
    return p


def _validate(seed=0, T0=24, n_gen=16):
    p = build_random_params(seed)
    gf = GoformerFull(p)
    rng = np.random.default_rng(seed + 1)
    prompt = rng.integers(0, p["tok_emb"].shape[0], size=T0)

    # 1) every prefix: KV last-token logits == full-recompute last-token logits
    dec = KVDecoder(p)
    worst = 0.0
    for t in range(T0):
        kv_logits = dec.step(int(prompt[t]))
        full_logits = gf.forward(prompt[:t + 1])[-1]
        worst = max(worst, float(np.abs(kv_logits - full_logits).max()))

    # 2) greedy stream equality vs a full-recompute greedy loop
    def full_greedy(idx, n):
        out = list(idx)
        for _ in range(n):
            out.append(int(np.argmax(gf.forward(out[-256:])[-1])))
        return out
    kv_stream = KVDecoder(p).generate_greedy(prompt, n_gen, 256)
    full_stream = full_greedy(prompt, n_gen)
    same_stream = kv_stream == full_stream

    print(f"goformer_kv vs full-recompute: prefix_maxabsdiff={worst:.3e} "
          f"greedy_stream_identical={same_stream} (T0={T0}, gen={n_gen})")
    ok = (worst == 0.0) and same_stream
    print("GOFORMER_KV_" + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    import sys
    raise SystemExit(0 if _validate() else 1)
