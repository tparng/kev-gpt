"""Fabric-precision non-linears for goformer — the bit-true reference the RTL copies.

The matmuls are exact integer (bit-exact on silicon already). The non-linears
(softmax, LayerNorm, GELU) cannot be bit-exact to a float reference — exp/rsqrt/erf
are transcendental — so the project gate softens to cosine > 0.9999. This module
implements each non-linear the way the fabric will (LUTs + fixed-point + a couple of
Newton steps) and measures the end-to-end cosine vs the float `goformer_full`, so the
required precision (LUT sizes, fractional bits, Newton iterations) is a *measured*
result before any RTL is written.

    python -m model.goformer_fixed          # cosine, same-argmax vs float goformer

Strategy (matches ROADMAP-10K "the non-linears"):
- GELU  : N-entry LUT over [-8, 8], linear interpolation.
- softmax: running-max subtract, exp LUT clamped to [-zmax, 0] with Q1.f output;
           causal -inf entries forced to exactly 0 (mask never enters the LUT range).
- LayerNorm (gamma-only, eps 1e-5, population var): 8-bit-seed rsqrt + Newton steps.
- dequant: fold w_scale*act_scale to one per-channel scale, optionally 16-bit mantissa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .goformer_full import GoformerFull, gelu, numpy_matmul


# Precision MEASURED against the real exported model (model.goformer_fixed): these
# are the smallest knobs that hold end-to-end logits cosine = 1.0000000 (same argmax
# at every position). They are larger than the per-op abs-error budget predicts,
# because char-level logits amplify coherent error — this is the precision crossover,
# measured not assumed. All hardware-realistic (8K-entry GELU LUT ~= 4 BRAM, 24-bit
# per-channel scale mult, Q1.20 exp LUT, 8-bit-seed rsqrt + 2 Newton steps).
@dataclass(frozen=True)
class FixedConfig:
    gelu_n: int = 8192         # GELU LUT entries over [gelu_lo, gelu_hi]
    gelu_lo: float = -8.0
    gelu_hi: float = 8.0
    exp_zmax: float = 16.0     # exp LUT input range [-zmax, 0]
    exp_frac: int = 20         # exp output fractional bits (Q1.20)
    rsqrt_seed_bits: int = 8   # rsqrt seed mantissa bits
    rsqrt_iters: int = 2       # Newton-Raphson iterations
    scale_mant_bits: int = 24  # dequant per-channel scale mantissa bits (0 = exact)
    use_gelu: bool = True      # toggles for per-op error attribution
    use_softmax: bool = True
    use_ln: bool = True


def _quantize_mantissa(v, bits):
    """Keep `bits` mantissa bits of |v| (emulates a fixed-width fractional store)."""
    if bits <= 0:
        return np.asarray(v, float)
    v = np.asarray(v, float)
    out = np.array(v, dtype=float)
    nz = v != 0
    e = np.floor(np.log2(np.abs(v[nz])))
    scale = 2.0 ** (e - bits + 1)
    out[nz] = np.round(v[nz] / scale) * scale
    return out


def gelu_lut(x, cfg: FixedConfig):
    lx = np.linspace(cfg.gelu_lo, cfg.gelu_hi, cfg.gelu_n)
    ly = gelu(lx)
    return np.interp(np.clip(x, cfg.gelu_lo, cfg.gelu_hi), lx, ly)


def rsqrt_fixed(a, cfg: FixedConfig):
    """1/sqrt(a) via an 8-bit-accurate seed + Newton: y <- y*(1.5 - 0.5*a*y^2)."""
    seed = _quantize_mantissa(1.0 / np.sqrt(a), cfg.rsqrt_seed_bits)
    y = seed
    for _ in range(cfg.rsqrt_iters):
        y = y * (1.5 - 0.5 * a * y * y)
    return y


def fixed_layernorm(x, gamma, cfg: FixedConfig, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)                 # biased/population (torch)
    return (x - mu) * rsqrt_fixed(var + eps, cfg) * gamma


def fixed_softmax(x, cfg: FixedConfig, axis=-1):
    finite = np.isfinite(x)
    m = np.max(np.where(finite, x, -np.inf), axis=axis, keepdims=True)
    z = np.clip(x - m, -cfg.exp_zmax, 0.0)
    e = np.exp(z)
    e = np.where(finite, e, 0.0)                   # causal -inf -> exactly 0
    q = 2.0 ** cfg.exp_frac
    e = np.round(e * q) / q                        # Q1.f exp output
    return e / e.sum(axis, keepdims=True)


class FixedGoformer(GoformerFull):
    """GoformerFull with the three non-linears replaced by their fabric-precision
    versions. The integer GEMV path is untouched (already bit-exact)."""

    def __init__(self, params, cfg: FixedConfig = FixedConfig(), matmul=numpy_matmul):
        super().__init__(params, matmul)
        self.cfg = cfg

    def qlin(self, x, layer):
        int_w, w_scale, s_act = layer
        if self.cfg.scale_mant_bits:
            w_scale = _quantize_mantissa(w_scale * s_act, self.cfg.scale_mant_bits)
            s_act_eff, scale = 1.0, w_scale          # folded per-channel scale
        else:
            s_act_eff, scale = s_act, w_scale * s_act
        ix = np.clip(np.round(x / s_act), -128, 127).astype(np.int64)
        T = x.shape[0]
        out = np.empty((T, int_w.shape[0]), dtype=np.float64)
        for t in range(T):
            y_int = np.asarray(self.matmul(int_w, ix[t]))
            out[t] = y_int.astype(np.float64) * scale
        return out

    def _attn(self, h, blk):
        T, C = h.shape
        nh = self.n_head
        hd = C // nh
        qkv = self.qlin(h, blk["qkv"])
        q, k, v = qkv[:, :C], qkv[:, C:2 * C], qkv[:, 2 * C:]
        q = q.reshape(T, nh, hd).transpose(1, 0, 2)
        k = k.reshape(T, nh, hd).transpose(1, 0, 2)
        v = v.reshape(T, nh, hd).transpose(1, 0, 2)
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask[None], -np.inf, scores)
        attn = self._smax(scores) @ v
        y = attn.transpose(1, 0, 2).reshape(T, C)
        return self.qlin(y, blk["proj"])

    def _smax(self, s):
        from .goformer_full import softmax
        return fixed_softmax(s, self.cfg) if self.cfg.use_softmax else softmax(s, axis=-1)

    def _ln(self, x, gamma):
        from .goformer_full import layernorm
        return fixed_layernorm(x, gamma, self.cfg) if self.cfg.use_ln else layernorm(x, gamma)

    def _gelu(self, x):
        return gelu_lut(x, self.cfg) if self.cfg.use_gelu else gelu(x)

    def _mlp(self, h, blk):
        return self.qlin(self._gelu(self.qlin(h, blk["mlp_fc"])), blk["mlp_proj"])

    def forward(self, idx):
        idx = np.asarray(idx)
        T = len(idx)
        x = self.p["tok_emb"][idx] + self.p["pos_emb"][np.arange(T)]
        for blk in self.p["blocks"]:
            x = x + self._attn(self._ln(x, blk["ln1"]), blk)
            x = x + self._mlp(self._ln(x, blk["ln2"]), blk)
        x = self._ln(x, self.p["ln_f"])
        return self.qlin(x, self.p["head"])


def _profile_gelu_range(p, idx):
    """Max |GELU input| seen on this model — sizes the LUT domain honestly."""
    gf = GoformerFull(p)
    mx = 0.0
    x = p["tok_emb"][idx] + p["pos_emb"][np.arange(len(idx))]
    for blk in p["blocks"]:
        from .goformer_full import layernorm
        x = x + gf._attn(layernorm(x, blk["ln1"]), blk)
        h = gf.qlin(layernorm(x, blk["ln2"]), blk["mlp_fc"])    # the GELU input
        mx = max(mx, float(np.abs(h).max()))
        x = x + gf._mlp(layernorm(x, blk["ln2"]), blk)
    return mx


def _validate(seed=0, T0=48, cfg: FixedConfig | None = None,
              npz="fabric/export/goformer.npz"):
    import os
    if os.path.exists(npz):
        from .goformer_full import load_params
        p = load_params(npz)
        src = npz
    else:
        from .goformer_kv import build_random_params
        p = build_random_params(seed)
        src = "random-params"
    rng = np.random.default_rng(seed + 7)
    idx = rng.integers(0, p["tok_emb"].shape[0], size=T0)

    gmax = _profile_gelu_range(p, idx)
    if cfg is None:                                   # size the LUT to the profiled range
        dom = float(np.ceil(gmax + 1.0))
        cfg = FixedConfig(gelu_lo=-dom, gelu_hi=dom)

    ref = GoformerFull(p).forward(idx)
    fix = FixedGoformer(p, cfg).forward(idx)

    a, b = ref.ravel(), fix.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel = float(np.abs(a - b).max() / (np.abs(a).max() or 1.0))
    same_argmax = bool((ref.argmax(-1) == fix.argmax(-1)).all())
    print(f"src={src}  max|gelu_in|={gmax:.2f} -> LUT domain [{cfg.gelu_lo},{cfg.gelu_hi}]")
    print(f"goformer_fixed vs float: cosine={cos:.7f} rel_maxerr={rel:.2e} "
          f"same_argmax_all={same_argmax}  [gelu_n={cfg.gelu_n} exp=Q1.{cfg.exp_frac} "
          f"rsqrt_iters={cfg.rsqrt_iters} scale_bits={cfg.scale_mant_bits}]")
    ok = cos > 0.9999 and same_argmax
    print("GOFORMER_FIXED_" + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _validate() else 1)
