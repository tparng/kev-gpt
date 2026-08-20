"""Fabric-aware QAT forward for Mamba-2 (differentiable STE mirror of the
fabric datapath defined by model/mamba2_fixed.py::FixedMamba2).

The +19% NLL fabric gap (measured via fabric/stage3/mamba_seq_ref.py over 159
chat tokens) is the fabric's LUT / rsqrt / requant / INT8-act PRECISION floor
accumulated over 7 layers -- NOT the INT4 weights (already QAT) nor a bug. This
module builds a torch forward that applies those same lossy operations through
straight-through estimators (STE) so the model can be fine-tuned to match the
float ideal *through the fabric's quantised datapath*.

What is modelled (matching FixedMamba2.step contracts):
  - residual stream requant  : Q6.19 round/sat between layers + at the embedding
  - RMSNorm input view        : Q6.9 round (norm is scale-invariant; rsqrt itself
                                converges to true rsqrt in the fabric -> torch.rsqrt)
  - INT8 activation quant     : per-tensor absmax scale at in_proj / out_proj /
                                head GEMV boundaries (fixed, calibrated)
  - SiLU                       : 256-entry floor-to-grid LUT over [-4,4) (conv + gate)
  - dt / decay                 : dt quantised to Q1.14, a = exp(-Q4.12(dt*|A|))
  - scan x/B/C                  : INT8 with per-layer power-of-2 scales (bX/bB/bC)
  - INT4 weights                : left to train.py's --qat-w4 W4STE parametrization

NOT modelled: the INT16 Q3.13 recurrent scan *state* trajectory (kept float in the
quadratic training form). FixedMamba2 measures that at +0.00% NLL, so the quadratic
float scan is a faithful stand-in for the logits (validated in validate()).

Calibration reuses FixedMamba2.calibrate() so the STE scales are exactly the
fabric export's scales.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .mamba2 import Mamba2, Mamba2Config

Q_RES = 19          # residual stream Q6.19 (matches mamba2_fixed.Q_RES)
Q_NVIEW = 9         # RMSNorm input view Q6.9 (round x*512)
Q_DT = 14           # dt Q1.14
Q_DTA = 12          # dt*|A| quantised to Q4.12 before exp (a_from_dtA contract)


def _ste_round(x):
    return x + (torch.round(x) - x).detach()


def _q_fixed(x, frac, bits=32):
    """Q-format round + saturate, STE."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    q = torch.clamp(_ste_round(x * (1 << frac)), lo, hi)
    return q / (1 << frac)


def _q_int8(x, scale):
    """Per-tensor INT8 fake-quant with a fixed float scale, STE."""
    q = torch.clamp(_ste_round(x / scale), -128, 127)
    return q * scale


def _q_int8_pow2(x, bshift):
    """INT8 fake-quant with a power-of-two scale 2^-bshift (scan x/B/C), STE.
    bshift may be a python int (per-layer, calibrated)."""
    s = 2.0 ** (-bshift)
    q = torch.clamp(_ste_round(x / s), -128, 127)
    return q * s


def _silu_lut_ste(x):
    """SiLU via the 256-entry floor-to-grid LUT over [-4,4) step 1/32, tails
    linear (x>=4) / zero (x<=-4). Matches mamba2_fixed.silu_lut. STE floor."""
    xs = x * 32.0
    xg = (xs + (torch.floor(xs) - xs).detach()) / 32.0   # STE floor-to-grid
    val = xg * torch.sigmoid(xg)
    inside = x.abs() < 4.0
    return torch.where(x >= 4.0, x, torch.where(inside, val, torch.zeros_like(x)))


def _rmsnorm_fabric(x, gamma, gate, eps=1e-5):
    """Fabric RMSNorm: Q6.9-view input, optional SiLU-LUT gate, true rsqrt
    (the fabric's seed+2-Newton converges to truth), * gamma."""
    if gate is not None:
        x = x * _silu_lut_ste(gate)
    xv = _q_fixed(x, Q_NVIEW, bits=16)                    # Q6.9 view (norm is scale-free)
    ms = xv.float().pow(2).mean(-1, keepdim=True) + eps
    return (xv.float() * torch.rsqrt(ms)) * gamma


class FabricMamba2(Mamba2):
    """Mamba2 whose (quadratic, training) forward runs the fabric-aware STE
    datapath. step()/generate() are inherited (float) -- used only for eval
    sampling. Calibrated scales live in per-layer python-float buffers set by
    calibrate_from_fixed()."""

    def __init__(self, cfg: Mamba2Config):
        super().__init__(cfg)
        L = cfg.n_layer
        # scales (defaults; overwritten by calibrate_from_fixed)
        self.fab_in_scale = [0.06] * L
        self.fab_out_scale = [0.06] * L
        self.fab_head_scale = 0.06
        self.fab_bX = [7] * L
        self.fab_bB = [7] * L
        self.fab_bC = [7] * L

    # ---- calibration: reuse FixedMamba2 so STE scales == fabric export -----
    def calibrate_from_fixed(self, ckpt_path, val_toks):
        from .mamba2_fixed import FixedMamba2
        fx = FixedMamba2(ckpt_path)
        fx.calibrate(val_toks)
        L = self.cfg.n_layer
        self.fab_in_scale = [float(fx.xs[("in", l)]) for l in range(L)]
        self.fab_out_scale = [float(fx.xs[("out", l)]) for l in range(L)]
        self.fab_head_scale = float(fx.xs["head"])
        self.fab_bX = [int(fx.layers[l]["bX"]) for l in range(L)]
        self.fab_bB = [int(fx.layers[l]["bB"]) for l in range(L)]
        self.fab_bC = [int(fx.layers[l]["bC"]) for l in range(L)]
        return self

    # ---- one fabric-aware mixer block over a full sequence (quadratic) -----
    def _fab_block(self, blk, u, li):
        cfg = blk.cfg
        Bb, T, _ = u.shape
        nh, P, N = cfg.n_heads, cfg.headdim, cfg.d_state

        # INT8-quant the (already RMSNorm'd) in_proj input, then GEMV (weight is
        # INT4-fake-quant via the --qat-w4 parametrization on blk.in_proj).
        u_q = _q_int8(u, self.fab_in_scale[li])
        z, xBC, dt = blk._split(blk.in_proj(u_q))

        # conv + SiLU-LUT
        xBC = _silu_lut_ste(blk.conv(xBC.transpose(1, 2))[..., :T].transpose(1, 2))
        x, Bmat, Cmat = torch.split(xBC, [cfg.d_inner, N, N], dim=-1)

        # scan inputs INT8 with per-layer pow2 scales
        xq = _q_int8_pow2(x, self.fab_bX[li])
        Bq = _q_int8_pow2(Bmat, self.fab_bB[li])
        Cq = _q_int8_pow2(Cmat, self.fab_bC[li])

        x32 = xq.float().view(Bb, T, nh, P)
        A = -torch.exp(blk.A_log.float())                          # (H,)
        dt32 = F.softplus(dt.float() + blk.dt_bias.float())        # (B,T,H)
        dt_q = _q_fixed(dt32, Q_DT, bits=16)                       # dt Q1.14
        # a = exp(-Q4.12(dt*|A|))  -> decay in log space for the cumsum path
        dtA_q = _q_fixed(dt32 * (-A), Q_DTA, bits=32)              # dt*|A| >=0, Q4.12
        a = torch.exp(-dtA_q)                                      # (B,T,H) in (0,1]
        loga = torch.log(a.clamp(min=1e-12))
        La = torch.cumsum(loga, dim=1)                            # (B,T,H)
        diff = La.permute(0, 2, 1).unsqueeze(-1) - La.permute(0, 2, 1).unsqueeze(-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=u.device))
        Ldec = torch.exp(diff.masked_fill(~mask, float("-inf")))  # (B,H,T,S)
        CB = torch.einsum("btn,bsn->bts", Cq.float(), Bq.float())
        M = CB.unsqueeze(1) * Ldec * dt_q.permute(0, 2, 1).unsqueeze(-2)
        y = torch.einsum("bhts,bshp->bthp", M, x32)
        y = y + blk.D.float().view(1, 1, nh, 1) * x.float().view(Bb, T, nh, P)
        y = y.reshape(Bb, T, cfg.d_inner).to(u.dtype)

        # gated RMSNorm (gate via SiLU-LUT) + INT8-quant + out_proj GEMV
        yn = _rmsnorm_fabric(y, blk.norm.weight, gate=z)
        yn_q = _q_int8(yn, self.fab_out_scale[li])
        return blk.out_proj(yn_q)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = _q_fixed(x, Q_RES)                                     # emb -> Q6.19 residual
        for li, layer in enumerate(self.layers):
            xn = _rmsnorm_fabric(x, layer.norm.weight, gate=None)  # pre-norm (Q6.9 view)
            dx = self._fab_block(layer.mixer, xn, li)
            x = _q_fixed(x + dx, Q_RES)                            # residual requant Q6.19
        xn = _rmsnorm_fabric(x, self.norm_f.weight, gate=None)
        xn_q = _q_int8(xn, self.fab_head_scale)
        logits = self.head(xn_q)
        loss = None
        if targets is not None:
            per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      targets.view(-1), reduction="none")
            if self.cfg.loss_clamp > 0:
                with torch.no_grad():
                    w = (self.cfg.loss_clamp / per_tok.clamp(min=1e-6)).clamp(max=1.0)
                loss = (w * per_tok).sum() / w.sum().clamp(min=1.0)
            else:
                loss = per_tok.mean()
            if self.cfg.z_loss > 0:
                z = torch.logsumexp(logits.float(), dim=-1)
                loss = loss + self.cfg.z_loss * (z * z).mean()
        return logits, loss


# ------------------------------------------------------------- validation ----

def validate(ckpt_path, val_bin="data/bpe1024_chat6/val.bin", n_tok=96,
             calib=64, device="cpu"):
    """Gate: does the torch STE fabric-forward match FixedMamba2.step()?
    Runs both over the first n_tok val tokens and reports per-token logit
    cosine + argmax agreement. Print MAMBA2_FABQAT_VALIDATE."""
    import numpy as np
    from .mamba2_fixed import FixedMamba2

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Mamba2Config(**ck["cfg"])
    m = FabricMamba2(cfg)
    m.load_state_dict(ck["model"])
    m.eval().to(device)

    val = np.fromfile(val_bin, dtype=np.uint16)
    m.calibrate_from_fixed(ckpt_path, val[:calib])

    # torch STE forward over the sequence (quadratic gives every position)
    seq = torch.tensor(val[:n_tok].astype(np.int64), device=device).unsqueeze(0)
    with torch.no_grad():
        logits_ste, _ = m(seq)
    logits_ste = logits_ste[0].float().cpu().numpy()

    # FixedMamba2 step reference over the same tokens
    fx = FixedMamba2(ckpt_path)
    fx.calibrate(val[:calib])
    st = fx.alloc_state()
    coss, agree = [], 0
    for t in range(n_tok):
        lf = fx.step(int(val[t]), st)
        ls = logits_ste[t]
        num = float(ls @ lf)
        den = (np.linalg.norm(ls) * np.linalg.norm(lf)) or 1.0
        coss.append(num / den)
        agree += int(np.argmax(ls) == np.argmax(lf))
    coss = np.array(coss)
    print(f"MAMBA2_FABQAT_VALIDATE: logits cosine mean {coss.mean():.5f} "
          f"min {coss.min():.5f} tail {coss[len(coss)//2:].mean():.5f}  "
          f"argmax-agree {agree}/{n_tok} ({100*agree/n_tok:.1f}%)  "
          f"[STE fabric-forward vs FixedMamba2.step, {n_tok} val tokens]")
    ok = coss[len(coss) // 2:].mean() > 0.99 and agree / n_tok > 0.75
    print(f"MAMBA2_FABQAT_VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(bar: tail cosine > 0.99 AND argmax-agree > 75%)")
    return ok


def bake(in_ckpt, out_ckpt):
    """Fold the --qat-w4 W4STE fake-quant into the weights (the same
    remove_parametrizations step qscratch.baked.pt was made with) so
    mamba2_fixed / export_mamba can consume the checkpoint."""
    import torch.nn.utils.parametrize as P
    ck = torch.load(in_ckpt, map_location="cpu", weights_only=False)
    cfg = Mamba2Config(**ck["cfg"])
    m = FabricMamba2(cfg)

    class W4STE(torch.nn.Module):
        def forward(self, w):
            s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7
            q = (w / s).round().clamp(-8, 7) * s
            return w + (q - w).detach()

    parametrized = any(k.endswith("parametrizations.weight.original")
                       for k in ck["model"])
    if parametrized:
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                P.register_parametrization(mod, "weight", W4STE())
    m.load_state_dict(ck["model"])
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear) and P.is_parametrized(mod, "weight"):
            P.remove_parametrizations(mod, "weight", leave_parametrized=True)
    torch.save({"model": m.state_dict(), "cfg": ck["cfg"], "meta": ck.get("meta"),
                "iter": ck.get("iter"), "val": ck.get("val"), "arch": "mamba2"},
               out_ckpt)
    print(f"baked {in_ckpt} -> {out_ckpt} (parametrizations {'folded' if parametrized else 'none'})")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="model.mamba2_fabqat")
    ap.add_argument("ckpt", nargs="?", default="data/ckpt.mamba2.qscratch.baked.pt")
    ap.add_argument("--val-bin", default="data/bpe1024_chat6/val.bin")
    ap.add_argument("--n-tok", type=int, default=96)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bake", metavar="OUT", default=None,
                    help="bake ckpt (fold W4STE) to OUT instead of validating")
    args = ap.parse_args(argv)
    if args.bake:
        bake(args.ckpt, args.bake)
    else:
        validate(args.ckpt, args.val_bin, args.n_tok, args.calib, args.device)


if __name__ == "__main__":
    main()
