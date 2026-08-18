"""mamba2_fixed — full-forward fixed-point reference (doc 9 §7 rung 1).

The seq_ref equivalent for the mamba engine: one decode step, every phase in
the integer arithmetic the fabric will use, with every intermediate exposed
so the sequencer RTL can be gated phase-by-phase. Numpy only (runs on the A53).

Contracts (per doc 9 + the silicon-proven scan core):
  weights           INT4 per-out-channel symmetric, FP16-ish scale -> the
                    GEMV computes INT4xINT8 -> INT32, dequant = *scale
  activations x     INT8 saturating, per-tensor absmax scale (calibrated)
  residual stream   INT16 Q3.12 (range +-8 covers post-conv activations)
  scan state h      INT16 Q3.13, fused single-rounding update (ssm_scan.sv,
                    bit-exact on silicon 100-300 MHz)
  decay a           UINT16 Q0.16 via per-head fused softplus*exp LUT on the
                    raw dt projection (256 entries)
  SiLU / RMSNorm    256-entry LUT / Newton-Raphson rsqrt, the goformer_fixed
                    recipe adapted

Gate: cosine vs the float model per phase (> 0.999 per block output,
> 0.9999 end-to-end logits is the ship bar; scan is bit-exact by contract).

    python -m model.mamba2_fixed data/ckpt.mamba2.fabricq2.baked.pt
"""

from __future__ import annotations

import argparse

import numpy as np

from .mamba2_fixed_scan import (Q_A, Q_ACT, Q_STATE, a_from_dtA, quant, rsh,
                                sat16, scan_step_fixed, scan_step_fixed2)

QW = 4                     # weight bits
Q_RES = 12                 # residual stream Q3.12


# ---------------------------------------------------------------- weights ----

def quant_w(w: np.ndarray):
    """Per-out-channel symmetric INT4. Returns (int8 array holding [-8,7], scale)."""
    qmax = (1 << (QW - 1)) - 1
    s = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-8) / qmax
    q = np.clip(np.round(w / s), -qmax - 1, qmax).astype(np.int8)
    return q, s.astype(np.float32)


def gemv_i4i8(wq: np.ndarray, ws: np.ndarray, x_i8: np.ndarray, xs: float):
    """INT4 x INT8 -> INT32 accumulate, dequant to float (the fabric's
    dequant multiplies by ws*xs in fixed point; float here is the contract's
    mathematical value — the RTL gate compares the INT32 accumulators)."""
    acc = wq.astype(np.int64) @ x_i8.astype(np.int64)          # exact INT32
    return acc, acc * (ws[:, 0] * xs)


def quant_act(x: np.ndarray, s: float):
    return np.clip(np.round(x / s), -128, 127).astype(np.int64)


# ------------------------------------------------------------- non-linears ---

_SILU_LUT = None

def silu_lut(x: np.ndarray):
    """SiLU via 256-entry LUT over [-8, 8) in Q3.12-ish input, INT16 out.
    x float in, float out quantized to LUT resolution (the RTL indexes the
    same table with the top 8 bits of the Q3.12 value)."""
    global _SILU_LUT
    if _SILU_LUT is None:
        grid = (np.arange(256) - 128) / 32.0                   # [-4, 4) step 1/32
        _SILU_LUT = grid / (1.0 + np.exp(-grid))
    x = np.asarray(x, dtype=np.float64)
    out = np.where(x >= 4.0, x, 0.0)                           # asymptotic tails:
    inside = np.abs(x) < 4.0                                   # silu(x)~x / ~0
    idx = np.clip(np.round(x[inside] * 32.0) + 128, 0, 255).astype(np.int64)
    out[inside] = _SILU_LUT[idx]
    return out


def rmsnorm_fixed(x: np.ndarray, gamma: np.ndarray, gate: np.ndarray | None,
                  eps=1e-5, nr_iters=2):
    """RMSNorm with Newton-Raphson rsqrt from a coarse seed (the LN datapath
    minus the mean). Gate (mamba: silu(z)) multiplies pre-gamma."""
    if gate is not None:
        x = x * silu_lut(gate)      # gate BEFORE the rms (matches the float
                                    # model: rms is taken over the gated value)
    ms = float((x * x).mean()) + eps
    # NR rsqrt: seed from exponent halving, 2 iterations (matches the fabric's
    # 4-stage pipeline at fp-precision level; the RTL gate compares exactly)
    y = 2.0 ** (-np.floor(np.log2(ms)) / 2.0)
    for _ in range(nr_iters):
        y = y * (1.5 - 0.5 * ms * y * y)
    return x * y * gamma


# ------------------------------------------------------------------ model ----

class FixedMamba2:
    """Integer decode-step reference built from a baked checkpoint."""

    scan_float = False   # ablation: float scan inputs+state, isolates the
                         # scan-quantization contract from the GEMV/LUT one

    def __init__(self, ckpt_path: str):
        import torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = dict(ck["cfg"])
        sd = {k: v.numpy().astype(np.float64) for k, v in ck["model"].items()}
        c = self.cfg
        self.L = c["n_layer"]; self.D = c["d_model"]
        self.d_in = c["expand"] * self.D; self.N = c["d_state"]
        self.P = c["headdim"]; self.H = self.d_in // self.P
        self.conv_k = c["d_conv"]; self.V = c["vocab_size"]

        self.emb = sd["tok_emb.weight"]
        self.layers = []
        for l in range(self.L):
            p = f"layers.{l}."
            lay = {
                "in_q":  quant_w(sd[p + "mixer.in_proj.weight"]),
                "out_q": quant_w(sd[p + "mixer.out_proj.weight"]),
                "conv_w": sd[p + "mixer.conv.weight"][:, 0, :],   # (conv_dim, k)
                "conv_b": sd[p + "mixer.conv.bias"],
                "dt_bias": sd[p + "mixer.dt_bias"],
                "negA": np.exp(sd[p + "mixer.A_log"]),            # |A| per head
                "Dh": sd[p + "mixer.D"],
                "norm_g": sd[p + "mixer.norm.weight"],
                "ln_g": sd[p + "norm.weight"] if p + "norm.weight" in sd else None,
            }
            self.layers.append(lay)
        self.normf_g = sd["norm_f.weight"]
        self.head_q = quant_w(sd["head.weight"])

        # activation scales: calibrated per boundary (absmax/127); start with
        # a generous fixed guess, refine via calibrate()
        self.xs = {("in", l): 0.06 for l in range(self.L)}
        self.xs.update({("out", l): 0.06 for l in range(self.L)})
        self.xs["head"] = 0.06

    def alloc_state(self):
        dt_ = float if self.scan_float else np.int64
        return {
            "h": [np.zeros((self.H, self.P, self.N), dtype=dt_)
                  for _ in range(self.L)],
            "conv": [np.zeros((self.d_in + 2 * self.N, self.conv_k - 1))
                     for _ in range(self.L)],
        }

    def step(self, tok: int, state, collect=None):
        """One decode step. collect: optional dict capturing per-phase signals."""
        x = self.emb[tok].copy()                                # float residual
        rng_ = getattr(self, "_ranges", None)
        for l, lay in enumerate(self.layers):
            # pre-norm residual block: x + mixer(RMSNorm(x)) — the first
            # reference draft fed x straight into in_proj (phases all gated
            # clean individually; the walk-through found the missing norm)
            xn_in = rmsnorm_fixed(x, lay["ln_g"], gate=None)
            wq, ws = lay["in_q"]
            xs = self.xs[("in", l)]
            if rng_ is not None:
                rng_[("in", l)] = max(rng_[("in", l)], float(np.abs(xn_in).max()))
            x_i8 = quant_act(xn_in, xs)
            acc, zxbcdt = gemv_i4i8(wq, ws, x_i8, xs)
            if collect is not None:
                collect[f"l{l}.in_acc"] = acc

            d_in, N, H = self.d_in, self.N, self.H
            z = zxbcdt[:d_in]
            xBC = zxbcdt[d_in:d_in + d_in + 2 * N]
            dt_raw = zxbcdt[-H:]

            # conv1d (k-tap) + SiLU, per channel
            cbuf = np.concatenate([lay["conv"] if False else state["conv"][l],
                                   xBC[:, None]], axis=1)
            state["conv"][l] = cbuf[:, 1:]
            xBC = silu_lut((cbuf * lay["conv_w"]).sum(1) + lay["conv_b"])
            xv, Bv, Cv = xBC[:d_in], xBC[d_in:d_in + N], xBC[d_in + N:]
            if collect is not None:
                collect[f"l{l}.conv"] = xBC

            # per-head fused softplus*exp "LUT": dt = softplus(dt_raw + bias),
            # a = exp(-dt*|A|) quantized exactly as the scan contract expects
            v = dt_raw + lay["dt_bias"]
            dt = np.where(v > 20, v, np.log1p(np.exp(np.minimum(v, 20.0))))

            # scan per head — the silicon-proven integer core
            if self.scan_float:
                a = np.exp(-dt * lay["negA"])
                hs = state["h"][l]                              # float here
                y = np.zeros(d_in)
                for hh in range(H):
                    xh = xv[hh * self.P:(hh + 1) * self.P]
                    hs[hh] = a[hh] * hs[hh] + np.outer(dt[hh] * xh, Bv)
                    y[hh * self.P:(hh + 1) * self.P] = hs[hh] @ Cv
            else:
                # generalized shift-algebra contract: per-layer power-of-2
                # scales for x/B/C (real-text absmax 2.5-6.7 was CLIPPING the
                # old Q0.7 inputs) + per-layer state scale qh (old fixed Q3.13
                # used ~600/32767 counts on real states)
                if rng_ is not None:
                    lay["maxX"] = max(lay.get("maxX", 1e-6), float(np.abs(xv).max()))
                    lay["maxB"] = max(lay.get("maxB", 1e-6), float(np.abs(Bv).max()))
                    lay["maxC"] = max(lay.get("maxC", 1e-6), float(np.abs(Cv).max()))
                    lay["bX"] = int(np.floor(-np.log2(lay["maxX"] / 127)))
                    lay["bB"] = int(np.floor(-np.log2(lay["maxB"] / 127)))
                    lay["bC"] = int(np.floor(-np.log2(lay["maxC"] / 127)))
                bX = lay.get("bX", Q_ACT); bB = lay.get("bB", Q_ACT)
                bC = lay.get("bC", Q_ACT); qh = lay.get("qh", Q_STATE)
                x8 = quant_act(xv, 2.0 ** -bX)
                B8 = quant_act(Bv, 2.0 ** -bB)
                C8 = quant_act(Cv, 2.0 ** -bC)
                y = np.zeros(d_in)
                for hh in range(H):
                    dtv = float(dt[hh])
                    a_q = int(a_from_dtA(dtv * float(lay["negA"][hh])))
                    x8h = x8[hh * self.P:(hh + 1) * self.P]
                    eh = int(np.clip(7 - np.ceil(np.log2(max(dtv, 1e-9) * 127)), 0, 20))
                    dtx_q = quant(dtv * x8h * 2.0 ** eh, 0)
                    if rng_ is not None:
                        lay.setdefault("maxH", np.zeros(H))
                        # track real h range via the float shadow update
                    yq = scan_step_fixed2(state["h"][l][hh], dtx_q, B8, C8, a_q,
                                          16 + qh - bX - eh - bB, qh + bC - 13)
                    y[hh * self.P:(hh + 1) * self.P] = yq / (1 << Q_STATE)
            y = y + np.repeat(lay["Dh"], self.P) * xv            # D skip
            if collect is not None:
                collect[f"l{l}.scan"] = y

            # gated RMSNorm + out_proj
            yn = rmsnorm_fixed(y, lay["norm_g"], gate=z)
            os_ = self.xs[("out", l)]
            if rng_ is not None:
                rng_[("out", l)] = max(rng_[("out", l)], float(np.abs(yn).max()))
            y_i8 = quant_act(yn, os_)
            wq2, ws2 = lay["out_q"]
            _, dx = gemv_i4i8(wq2, ws2, y_i8, os_)
            x = x + dx
            if collect is not None:
                collect[f"l{l}.x_out"] = x

        xn = rmsnorm_fixed(x, self.normf_g, gate=None)
        if rng_ is not None:
            rng_["head"] = max(rng_["head"], float(np.abs(xn).max()))
        h_i8 = quant_act(xn, self.xs["head"])
        wqh, wsh = self.head_q
        _, logits = gemv_i4i8(wqh, wsh, h_i8, self.xs["head"])
        if collect is not None:
            collect["logits"] = logits
        return logits

    def calibrate(self, toks, iters=2):
        """Instrumented fixed passes: record the true absmax at every quant
        boundary, set scale = absmax/127, repeat (scales converge fast)."""
        for _ in range(iters):
            self._ranges = {k: 1e-6 for k in self.xs}
            st = self.alloc_state()
            for t in toks:
                self.step(int(t), st)
            for k, v in self._ranges.items():
                self.xs[k] = v / 127.0
        self._ranges = None

    def _old_calibrate(self, toks):
        """One pass tracking absmax at each quant boundary; sets scales."""
        maxes = {}
        orig_quant = globals()["quant_act"]

        def spy(x, s):
            return orig_quant(x, s)
        st = self.alloc_state()
        for l in range(self.L):
            self.xs[("in", l)] = 1.0
            self.xs[("out", l)] = 1.0
        self.xs["head"] = 1.0
        # crude two-pass: run once with loose scales tracking float ranges
        for t in toks:
            c = {}
            self.step(int(t), st, collect=c)
        # second pass proper calibration by direct range probing
        st = self.alloc_state()
        ranges = {k: 1e-6 for k in self.xs}
        x_probe = {}
        for t in toks:
            x = self.emb[int(t)].copy()
            for l, lay in enumerate(self.layers):
                ranges[("in", l)] = max(ranges[("in", l)], np.abs(x).max())
                wq, ws = lay["in_q"]
                zx = (wq * ws) @ x
                d_in, N, H = self.d_in, self.N, self.H
                z, xBC, dt_raw = zx[:d_in], zx[d_in:2 * d_in + 2 * N], zx[-H:]
                cbuf = np.concatenate([st["conv"][l], xBC[:, None]], axis=1)
                st["conv"][l] = cbuf[:, 1:]
                xBC = silu_lut((cbuf * lay["conv_w"]).sum(1) + lay["conv_b"])
                xv = xBC[:d_in]
                y = xv * np.repeat(lay["Dh"], self.P)            # rough proxy
                yn = rmsnorm_fixed(y, lay["norm_g"], gate=z)
                ranges[("out", l)] = max(ranges[("out", l)], np.abs(yn).max())
                wq2, ws2 = lay["out_q"]
                x = x + (wq2 * ws2) @ np.clip(yn, -4, 4)
                x_probe[l] = x
            xn = rmsnorm_fixed(x, self.normf_g, gate=None)
            ranges["head"] = max(ranges["head"], np.abs(xn).max())
        for k, v in ranges.items():
            self.xs[k] = float(v) / 127.0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="model.mamba2_fixed")
    ap.add_argument("ckpt")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--nll-tokens", type=int, default=512)
    ap.add_argument("--val-bin", default="data/bpe1024_chat6/val.bin")
    ap.add_argument("--scan-float", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from .mamba2 import Mamba2, Mamba2Config
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    fmodel = Mamba2(Mamba2Config(**ck["cfg"]))
    fmodel.load_state_dict(ck["model"])
    fmodel.eval()

    FixedMamba2.scan_float = args.scan_float
    fx = FixedMamba2(args.ckpt)
    rng = np.random.default_rng(args.seed)
    toks = rng.integers(0, fx.V, args.tokens)
    fx.calibrate(toks[:16])

    st = fx.alloc_state()
    fstate = fmodel.alloc_state(1, "cpu")
    coss = []
    for t in toks:
        c = {}
        logits_q = fx.step(int(t), st, collect=c)
        with torch.no_grad():
            logits_f = fmodel.step(torch.tensor([int(t)]), fstate)[0].numpy()
        num = float(logits_q @ logits_f)
        den = (np.linalg.norm(logits_q) * np.linalg.norm(logits_f)) or 1.0
        coss.append(num / den)
    coss = np.array(coss)
    tail = coss[len(coss) // 2:].mean()
    print(f"MAMBA2_FIXED: logits cosine tail {tail:.6f}  min {coss.min():.6f}  "
          f"({args.tokens} random tokens; diagnostic only)")

    # the honest gate: NLL on real val text, fixed vs float (the INT16 state
    # walks its own trajectory by design — doc 8 measured that at +0.00% NLL,
    # so per-logit cosine over long horizons is the wrong arbiter)
    val = np.fromfile(args.val_bin, dtype=np.uint16)[:args.nll_tokens + 1]
    st = fx.alloc_state(); fstate = fmodel.alloc_state(1, "cpu")
    nll_q = nll_f = 0.0
    for i in range(args.nll_tokens):
        t, nxt = int(val[i]), int(val[i + 1])
        lq = fx.step(t, st)
        with torch.no_grad():
            lf = fmodel.step(torch.tensor([t]), fstate)[0].numpy()
        for L, acc in ((lq, "q"), (lf, "f")):
            m = L.max(); p = np.exp(L - m); ce = -(L[nxt] - m - np.log(p.sum()))
            if acc == "q": nll_q += ce
            else: nll_f += ce
    nll_q /= args.nll_tokens; nll_f /= args.nll_tokens
    d = (nll_q - nll_f) / nll_f * 100
    print(f"MAMBA2_FIXED_NLL: float {nll_f:.4f}  fixed {nll_q:.4f}  ({d:+.2f}%)"
          f"  [{args.nll_tokens} val tokens]")
    print(f"MAMBA2_FIXED_VERDICT: {'PASS' if d < 2.0 else 'FAIL'} "
          f"(bar +2% NLL vs float; per-phase RTL gates compare exactly)")


if __name__ == "__main__":
    main()
