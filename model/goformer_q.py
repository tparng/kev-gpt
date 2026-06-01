"""Pin the fixed-point datapath Q-format — the spec the non-linear/sequencer RTL uses.

`goformer_fixed` proved the *op* precision (LUTs, scale bits) with a float residual
stream. The hardware can't carry floats: the residual stream and the non-linear I/O
must be fixed-point. This module quantizes the whole datapath to a signed Q(int.frac)
format, profiles the ranges, and finds the smallest (int, frac) that still holds
end-to-end logits cosine > 0.9999 on the real model. The chosen widths are the
contract the RTL datapath is built to.

    python -m model.goformer_q          # profile + sweep -> the pinned format + PASS
"""

from __future__ import annotations

import os

import numpy as np

from .goformer_fixed import FixedConfig, FixedGoformer, _profile_gelu_range
from .goformer_full import GoformerFull, layernorm, load_params
from .goformer_kv import build_random_params


def q_fixed(x, int_bits, frac_bits):
    """Signed Q(int_bits.frac_bits): round to a 2^-frac grid, clip to +-2^int_bits."""
    s = 2.0 ** frac_bits
    hi = 2.0 ** int_bits - 1.0 / s
    return np.clip(np.round(np.asarray(x) * s) / s, -2.0 ** int_bits, hi)


class FixedPointGoformer(FixedGoformer):
    """FixedGoformer with the residual stream + non-linear I/O quantized to a pinned
    Q-format, so the whole datapath is integer (no float carried between blocks)."""

    def __init__(self, params, cfg: FixedConfig, resid_int=6, resid_frac=25):
        super().__init__(params, cfg)
        self.ri, self.rf = resid_int, resid_frac

    def _q(self, x):
        return q_fixed(x, self.ri, self.rf)

    def forward(self, idx):
        # Quantize only the RESIDUAL STREAM (the value carried + added between blocks).
        # LayerNorm outputs feed the GEMV's INT8 requant, so they need no residual-format
        # rounding; quantizing them just adds error.
        idx = np.asarray(idx)
        T = len(idx)
        x = self._q(self.p["tok_emb"][idx] + self.p["pos_emb"][np.arange(T)])
        for blk in self.p["blocks"]:
            x = self._q(x + self._attn(self._ln(x, blk["ln1"]), blk))
            x = self._q(x + self._mlp(self._ln(x, blk["ln2"]), blk))
        x = self._ln(x, self.p["ln_f"])           # ln_f feeds head's INT8 requant directly
        return self.qlin(x, self.p["head"])


def _profile_resid(p, idx, cfg):
    """Max |residual stream| over the forward — sets the integer bits."""
    gf = FixedGoformer(p, cfg)
    x = gf.p["tok_emb"][idx] + gf.p["pos_emb"][np.arange(len(idx))]
    mx = float(np.abs(x).max())
    for blk in p["blocks"]:
        x = x + gf._attn(gf._ln(x, blk["ln1"]), blk); mx = max(mx, float(np.abs(x).max()))
        x = x + gf._mlp(gf._ln(x, blk["ln2"]), blk);  mx = max(mx, float(np.abs(x).max()))
    return mx


def _validate(npz="fabric/export/goformer.npz", T0=48):
    p, src = (load_params(npz), npz) if os.path.exists(npz) else (build_random_params(), "random")
    idx = np.random.default_rng(7).integers(0, p["tok_emb"].shape[0], size=T0)
    dom = float(np.ceil(_profile_gelu_range(p, idx) + 1))
    cfg = FixedConfig(gelu_lo=-dom, gelu_hi=dom)

    rmax = _profile_resid(p, idx, cfg)
    int_bits = int(np.ceil(np.log2(rmax))) + 1          # +1 guard
    ref = GoformerFull(p).forward(idx).ravel()
    nr = np.linalg.norm(ref)

    print(f"src={src}  max|residual|={rmax:.2f} -> int_bits={int_bits} (range +-{2**int_bits})")
    floor = None
    for frac in (16, 18, 20, 22, 25):
        fx = FixedPointGoformer(p, cfg, resid_int=int_bits, resid_frac=frac).forward(idx).ravel()
        cos = float(ref @ fx / (nr * np.linalg.norm(fx)))
        w = 1 + int_bits + frac
        if cos > 0.9999 and floor is None:
            floor = (frac, w, cos)
        print(f"  Q{int_bits}.{frac:<2d} ({w:2d}-bit)  cosine={cos:.7f}"
              + ("  <- floor" if floor and floor[0] == frac else ""))
    # recommended datapath: 32-bit signed residual (matches the GEMV accumulator width)
    rec_frac = 31 - int_bits
    rec = FixedPointGoformer(p, cfg, resid_int=int_bits, resid_frac=rec_frac).forward(idx).ravel()
    rec_cos = float(ref @ rec / (nr * np.linalg.norm(rec)))
    print(f"PINNED DATAPATH: residual = signed Q{int_bits}.{rec_frac} (32-bit), cosine={rec_cos:.7f}")
    print(f"  (floor that passes: Q{int_bits}.{floor[0]} = {floor[1]}-bit @ cosine {floor[2]:.7f})")
    print(f"  activations->GEMV: INT8 | GEMV accum: INT32 | dequant scale: 24-bit | "
          f"GELU LUT: {cfg.gelu_n}@[{cfg.gelu_lo:.0f},{cfg.gelu_hi:.0f}] | "
          f"exp: Q1.{cfg.exp_frac} z in[-{cfg.exp_zmax:.0f},0]")
    ok = rec_cos > 0.9999 and floor is not None
    print("GOFORMER_Q_" + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _validate() else 1)
