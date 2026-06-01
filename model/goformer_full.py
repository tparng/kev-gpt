"""Full integer "goformer" forward — Stage 3.

The whole Kevin forward with a PLUGGABLE matmul: every quantised Linear becomes
  int_x = clamp(round(x / s_act), -128, 127)        # INT8 activation quant
  y_int = matmul(int_w, int_x)                       # <-- numpy OR the PL fabric
  y     = y_int * w_scale * s_act                    # per-channel dequant
and everything else (embeddings, LayerNorm, causal attention, GELU, sampling)
runs in float, exactly as model/qgpt.py does it.

Because the PL GEMV is bit-exact to numpy's `int_w @ int_x` (proven on silicon),
swapping `matmul` from numpy to the fabric driver leaves the forward bit-identical
-- so validating the numpy version against the Brevitas QGPT (cosine > 0.9999) is
enough to trust the fabric one. That is the bit-honest bridge to a real PL chat.

    python -m model.goformer_full            # validate vs QGPT on data/ckpt.qat.pt
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.special import erf as _erf
except Exception:  # board may lack scipy; vectorised math.erf fallback
    import math
    _erf = np.vectorize(math.erf)


# --- float ops, matched to torch defaults ---------------------------------
def layernorm(x, gamma, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)          # biased/population variance (torch)
    return (x - mu) / np.sqrt(var + eps) * gamma


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))   # exact (erf) GELU, torch default


def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def numpy_matmul(int_w, int_x):
    """Reference matmul: exact int32 GEMV (what the PL reproduces bit-exactly)."""
    return int_w.astype(np.int64) @ int_x.astype(np.int64)


class GoformerFull:
    """params: dict with tok_emb, pos_emb, ln_f, n_head, and blocks[i] holding
    ln1, ln2 and the four (int_w, w_scale, act_scale) layer tuples, plus head."""

    def __init__(self, params, matmul=numpy_matmul):
        self.p = params
        self.matmul = matmul
        self.n_head = params["n_head"]

    def qlin(self, x, layer):
        """x (T, K) float -> y (T, M) float, GEMV on the pluggable backend."""
        int_w, w_scale, s_act = layer
        ix = np.clip(np.round(x / s_act), -128, 127).astype(np.int64)   # (T, K)
        T = x.shape[0]
        out = np.empty((T, int_w.shape[0]), dtype=np.float64)
        for t in range(T):
            y_int = np.asarray(self.matmul(int_w, ix[t]))
            out[t] = y_int.astype(np.float64) * w_scale * s_act
        return out

    def _attn(self, h, blk):
        T, C = h.shape
        nh = self.n_head
        hd = C // nh
        qkv = self.qlin(h, blk["qkv"])                 # (T, 3C)
        q, k, v = qkv[:, :C], qkv[:, C:2 * C], qkv[:, 2 * C:]
        # (T, nh, hd) -> (nh, T, hd)
        q = q.reshape(T, nh, hd).transpose(1, 0, 2)
        k = k.reshape(T, nh, hd).transpose(1, 0, 2)
        v = v.reshape(T, nh, hd).transpose(1, 0, 2)
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)        # (nh, T, T)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)         # causal
        scores = np.where(mask[None], -np.inf, scores)
        attn = softmax(scores, axis=-1) @ v                      # (nh, T, hd)
        y = attn.transpose(1, 0, 2).reshape(T, C)
        return self.qlin(y, blk["proj"])

    def _mlp(self, h, blk):
        return self.qlin(gelu(self.qlin(h, blk["mlp_fc"])), blk["mlp_proj"])

    def forward(self, idx):
        """idx: 1-D array of token ids (length T). Returns logits (T, vocab)."""
        idx = np.asarray(idx)
        T = len(idx)
        x = self.p["tok_emb"][idx] + self.p["pos_emb"][np.arange(T)]
        for blk in self.p["blocks"]:
            x = x + self._attn(layernorm(x, blk["ln1"]), blk)
            x = x + self._mlp(layernorm(x, blk["ln2"]), blk)
        x = layernorm(x, self.p["ln_f"])
        return self.qlin(x, self.p["head"])

    def generate(self, idx, n_tokens, block_size, temperature=0.85, top_k=40, rng=None):
        rng = rng or np.random.default_rng(0)
        idx = list(idx)
        for _ in range(n_tokens):
            cond = idx[-block_size:]
            logits = self.forward(cond)[-1] / max(temperature, 1e-6)
            if top_k:
                kth = np.sort(logits)[-min(top_k, len(logits))]
                logits = np.where(logits < kth, -np.inf, logits)
            probs = softmax(logits)
            idx.append(int(rng.choice(len(probs), p=probs)))
        return idx


# --- build params from a Brevitas QGPT checkpoint (laptop side) ------------
def params_from_ckpt(ckpt_path="data/ckpt.qat.pt"):
    import torch
    from .gpt import GPTConfig
    from .qgpt import QGPT
    from .export_fabric import _int_weight, _act_scale

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = QGPT(GPTConfig(**ck["cfg"]))
    with torch.no_grad():
        model(torch.zeros((1, 1), dtype=torch.long))
    model.load_state_dict(ck["model"])
    model.eval()

    def lyr(ql):
        iw, ws = _int_weight(ql)
        return (iw.astype(np.int64), ws.astype(np.float64), float(_act_scale(ql)))

    p = {
        "tok_emb": model.tok_emb.weight.detach().numpy().astype(np.float64),
        "pos_emb": model.pos_emb.weight.detach().numpy().astype(np.float64),
        "ln_f": model.ln_f.weight.detach().numpy().astype(np.float64),
        "n_head": model.cfg.n_head,
        "block_size": model.cfg.block_size,
        "head": lyr(model.head),
        "blocks": [],
    }
    for b in model.blocks:
        p["blocks"].append({
            "ln1": b.ln1.weight.detach().numpy().astype(np.float64),
            "ln2": b.ln2.weight.detach().numpy().astype(np.float64),
            "qkv": lyr(b.qkv), "proj": lyr(b.proj),
            "mlp_fc": lyr(b.mlp_fc), "mlp_proj": lyr(b.mlp_proj),
        })
    return p, model, ck


def save_params(p, path):
    """Flatten the params dict to a single .npz the board can load with just numpy."""
    flat = {"tok_emb": p["tok_emb"], "pos_emb": p["pos_emb"], "ln_f": p["ln_f"],
            "n_head": np.array(p["n_head"]), "block_size": np.array(p["block_size"]),
            "n_blocks": np.array(len(p["blocks"]))}
    iw, ws, sa = p["head"]
    flat["head_iw"], flat["head_ws"], flat["head_sa"] = iw, ws, np.array(sa)
    for i, b in enumerate(p["blocks"]):
        flat[f"b{i}_ln1"], flat[f"b{i}_ln2"] = b["ln1"], b["ln2"]
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            iw, ws, sa = b[nm]
            flat[f"b{i}_{nm}_iw"], flat[f"b{i}_{nm}_ws"], flat[f"b{i}_{nm}_sa"] = iw, ws, np.array(sa)
    np.savez(path, **flat)
    return path


def load_params(path):
    z = np.load(path)
    nb = int(z["n_blocks"])
    def lyr(pfx): return (z[pfx + "_iw"], z[pfx + "_ws"], float(z[pfx + "_sa"]))
    p = {"tok_emb": z["tok_emb"], "pos_emb": z["pos_emb"], "ln_f": z["ln_f"],
         "n_head": int(z["n_head"]), "block_size": int(z["block_size"]),
         "head": lyr("head"), "blocks": []}
    for i in range(nb):
        p["blocks"].append({"ln1": z[f"b{i}_ln1"], "ln2": z[f"b{i}_ln2"],
                            "qkv": lyr(f"b{i}_qkv"), "proj": lyr(f"b{i}_proj"),
                            "mlp_fc": lyr(f"b{i}_mlp_fc"), "mlp_proj": lyr(f"b{i}_mlp_proj")})
    return p


def _validate(ckpt_path="data/ckpt.qat.pt"):
    import torch
    p, model, ck = params_from_ckpt(ckpt_path)
    gf = GoformerFull(p)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, p["tok_emb"].shape[0], size=24)

    with torch.no_grad():
        t_logits = model(torch.tensor(idx)[None])[0][0].numpy()
    g_logits = gf.forward(idx)

    a, b = t_logits.ravel().astype(np.float64), g_logits.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel = float(np.abs(a - b).max() / (np.abs(a).max() or 1.0))
    # do they pick the same next token (argmax)?
    same_argmax = int((t_logits.argmax(-1) == g_logits.argmax(-1)).all())
    print(f"goformer_full vs QGPT: cosine={cos:.7f} rel_maxerr={rel:.2e} "
          f"same_argmax_all={bool(same_argmax)}")
    print("GOFORMER_FULL_" + ("PASS" if cos > 0.9999 else "FAIL"))


if __name__ == "__main__":
    import sys
    _validate(sys.argv[1] if len(sys.argv) > 1 else "data/ckpt.qat.pt")
