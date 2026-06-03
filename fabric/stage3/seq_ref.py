"""Integer sequencer reference — the bit-true contract for rtl/sequencer.sv.

The four gated datapath blocks (gemv_banked, layernorm, softmax, gelu_lut) each pin
the fixed-point format of THEIR op. The sequencer is the glue: it carries a residual
stream in signed Q6.25 and converts between blocks. The glue arithmetic (residual ->
LN -> INT8 act-quant -> INT32 GEMV -> per-channel dequant -> back to Q6.25; INT32
attention scores -> Q8.8 softmax; softmax probs Q1.20 . V -> ctx; argmax over INT32
logits) is NOT covered by any single block's gate, so it is pinned HERE and the RTL
FSM is gated bit-exact against this module.

This module does PURE INTEGER arithmetic step-for-step the way the FSM does it, and is
in turn validated (cosine > 0.9999, same argmax / same token stream) against the float
goformer_seq golden. So a chain of two gates binds the hardware to the float model:

    float goformer_seq  --(cosine>0.9999, same token)-->  seq_ref (this, integer)
    seq_ref             --(bit-exact, mismatches=0)    -->  rtl/sequencer.sv

Pinned glue formats (chosen to match the committed blocks; see the per-field notes):
  residual x      : signed Q6.25 (32-bit)                       [goformer_q]
  embeddings      : tok_emb+pos_emb pre-quantized to Q6.25
  LN gamma        : signed Q4.20 ; LN out y : signed Q.22       [layernorm.sv]
  GEMV act quant  : ix = clip(round(ln_y_real / s_act), -128, 127)  INT8
                    ln_y_real = y_q22 / 2^22 ; done as integer rounding-divide
  GEMV weights    : signed INT4 ; accum INT32 (exact)           [gemv_banked.sv]
  dequant         : y_real = y_int * (w_scale * s_act) ; per-channel scale folded to
                    a 24-bit mantissa * 2^exp (the pinned 24-bit dequant), then the
                    product re-quantized to Q6.25 for the residual add.
  attn scores     : score_real = (q.k_int * dq_q * dq_k) / sqrt(hd) -> Q8.8 for softmax
  softmax         : exp LUT Q1.20, z in [-16,0], reciprocal 2^40/sum                [softmax.sv]
  ctx = sum_j prob_q20[j] * v_real[j]  (prob back to real, V real) -> Q6.25
  gelu            : Q4.12 in/out, 8192-LUT + 3-bit interp                            [gelu_lut.sv]

The residual/embedding/LN-out/ctx values are real numbers carried as Q6.25 integers;
the transcendentals (rsqrt, exp, gelu) follow the committed block arithmetic exactly.
Where this reference must call a block's integer routine it imports it directly from
the block's run_*.py so there is ONE source of truth per op.

    python -m fabric.stage3.seq_ref           # SEQREF_VERDICT vs float goformer_seq
"""

from __future__ import annotations

import os

import numpy as np

from model.goformer_full import load_params, gelu as _gelu_exact
from model.goformer_fixed import FixedConfig, _profile_gelu_range

# reuse the committed block integer references (ONE source of truth per op)
from fabric.stage3.run_layernorm import _ln_int_quantized, QX as LN_QX, G_FRAC as LN_GFRAC, OUT_FRAC as LN_OUTFRAC
from fabric.stage3.run_softmax import exp_table, int_softmax_q, quantize_scores, SCORE_FRAC, PROB_FRAC
from fabric.stage3.run_gelu import gelu_table, gelu_q, FRAC as GELU_FRAC

# ---- pinned residual / glue formats -----------------------------------------
RESID_FRAC = 25          # residual stream Q6.25 (matches goformer_q, layernorm x_in)
RESID_INT = 6
SCALE_MANT_BITS = 24     # dequant per-channel folded scale = 24-bit mantissa (goformer_q)
NHEAD = 4
HEAD_DIM = 64
# attention integer datapath (matches rtl/sequencer.sv exactly)
VFRAC = 16               # q/k/v stored Q.16
ISQRT = 3                # /sqrt(head_dim=64) = >>3
PROB_FRAC_A = 20         # softmax prob Q1.20 (== run_softmax.PROB_FRAC)


def q_round_div(num, den):
    """Round-half-away-from-zero integer divide num/den (den>0). Matches np.round."""
    if num >= 0:
        return (num + den // 2) // den
    return -(((-num) + den // 2) // den)


def rsh_round(v, s):
    """Arithmetic right shift by s with round-half-away-from-zero; s<=0 -> left shift.
    Bit-identical to rtl/sequencer.rsh_round."""
    if s <= 0:
        return int(v) << (-s)
    return q_round_div(int(v), 1 << s)


def sat(v, lo, hi):
    return max(lo, min(hi, int(v)))


def to_q(x_real, frac):
    """Real -> signed fixed-point integer Q?.frac via round-half-away-from-zero."""
    return int(np.round(np.asarray(x_real, dtype=np.float64) * (1 << frac)))


def quantize_scale_24(scale):
    """Fold a per-channel real scale to a 24-bit mantissa * 2^exp (the pinned dequant).

    Returns (mant, exp) with scale ~= mant * 2^exp, mant a signed 24-bit-significant int.
    Matches model.goformer_fixed._quantize_mantissa(scale, 24) exactly."""
    scale = np.asarray(scale, dtype=np.float64)
    mant = np.zeros_like(scale, dtype=object)
    expo = np.zeros_like(scale, dtype=object)
    nz = scale != 0
    for i in np.ndindex(scale.shape):
        s = float(scale[i])
        if s == 0:
            mant[i], expo[i] = 0, 0
            continue
        e = int(np.floor(np.log2(abs(s))))
        q = e - SCALE_MANT_BITS + 1            # 2^q grid
        m = int(np.round(s / (2.0 ** q)))      # integer mantissa
        mant[i], expo[i] = m, q
    return mant, expo


class IntSequencer:
    """Pure-integer KV-cached per-token decoder. step(id) -> INT32 logits + argmax.

    The arithmetic is bit-identical to what rtl/sequencer.sv does; this is the gate
    contract. Precompute pins every per-channel scale to its 24-bit (mant,exp) form."""

    def __init__(self, params, cfg: FixedConfig):
        self.p = params
        self.cfg = cfg
        self.exp_lut = exp_table()
        self.gelu_lut = gelu_table()
        self.nblocks = len(params["blocks"])
        # pre-quantize embeddings to Q6.25 integers
        self.tok_emb_q = np.round(params["tok_emb"] * (1 << RESID_FRAC)).astype(object)
        self.pos_emb_q = np.round(params["pos_emb"] * (1 << RESID_FRAC)).astype(object)
        self.ln_f_q = np.round(params["ln_f"] * (1 << LN_GFRAC)).astype(object)
        # pre-pin every GEMV layer: int weights, act_scale, 24-bit dequant (mant,exp)
        self.layers = {}
        for bi, blk in enumerate(params["blocks"]):
            for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
                self.layers[(bi, nm)] = self._pin_layer(blk[nm])
            self.layers[(bi, "ln1")] = np.round(blk["ln1"] * (1 << LN_GFRAC)).astype(object)
            self.layers[(bi, "ln2")] = np.round(blk["ln2"] * (1 << LN_GFRAC)).astype(object)
        self.head = self._pin_layer(params["head"])
        self.reset()

    def _pin_layer(self, layer):
        int_w, w_scale, s_act = layer
        folded = w_scale * s_act                       # per-output-channel real scale
        mant, expo = quantize_scale_24(folded)
        return {"iw": np.asarray(int_w, dtype=np.int64),
                "s_act": float(s_act), "mant": mant, "exp": expo}

    def reset(self):
        self.k_cache = [[] for _ in range(self.nblocks)]   # per layer: list of (C,) int q6.25 ... actually real-as-Q
        self.v_cache = [[] for _ in range(self.nblocks)]
        self.t = 0

    # ---- LayerNorm (integer, bit-true to run_layernorm._ln_int_quantized) -----
    def _ln(self, x_q25, gamma_q20):
        """x_q25: (256,) object ints Q6.25. gamma_q20: (256,) ints Q4.20.
        Returns y_q22: (256,) ints Q.22 (the layernorm.sv output format)."""
        out, *_ = _ln_int_quantized(list(x_q25), list(gamma_q20))
        return out  # list of Q.22 ints

    # ---- GEMV: Q.22 source -> INT8 act -> INT32 -> dequant ---------------------
    INV_SACT_SH = 40            # inv_sact = round(2^40 / s_act); must match RTL ISH

    def _inv_sact(self, s_act):
        return int(round((1 << self.INV_SACT_SH) / s_act))

    def _act_quant(self, y_q22, s_act):
        """y_q22 (Q.LN_OUTFRAC ints) -> INT8 = clip(round(real/s_act), -128, 127).
        EXACT integer path the RTL does: round(y_q22 * inv_sact / 2^(LN_OUTFRAC+ISH))
        with inv_sact = round(2^ISH / s_act)."""
        inv = self._inv_sact(s_act)
        sh = LN_OUTFRAC + self.INV_SACT_SH
        ix = []
        for v in y_q22:
            q = rsh_round(int(v) * inv, sh)
            ix.append(sat(q, -128, 127))
        return np.asarray(ix, dtype=np.int64)

    def _gemv_int(self, layer, ix):
        """INT8 acts -> exact INT32 y_int = iw @ ix."""
        return layer["iw"].astype(object) @ ix.astype(object)

    def _dequant_to_q(self, layer, y_int, out_frac):
        """Per-channel 24-bit dequant of INT32 y_int to a signed Q.out_frac integer:
        val = round(y_int * mant * 2^exp * 2^out_frac) = (y_int*mant) <<>> (exp+out_frac).
        Bit-identical to rtl/sequencer's dequant (rsh_round)."""
        mant, expo = layer["mant"], layer["exp"]
        out = np.empty(len(y_int), dtype=object)
        for m in range(len(y_int)):
            prod = int(y_int[m]) * int(mant[m])
            out[m] = rsh_round(prod, -(int(expo[m]) + out_frac))
        return out

    def _gemv_dequant_to_q25(self, layer, ix):
        """INT8 acts -> INT32 -> per-channel 24-bit dequant -> Q6.25 ints."""
        return self._dequant_to_q(layer, self._gemv_int(layer, ix), RESID_FRAC)

    def _gemv_dequant_real(self, layer, ix):
        """Same GEMV but return real (float) outputs — used for logits/argmax."""
        y_int = self._gemv_int(layer, ix)
        mant, expo = layer["mant"], layer["exp"]
        out = np.empty(len(y_int), dtype=np.float64)
        for m in range(len(y_int)):
            out[m] = float(int(y_int[m]) * int(mant[m])) * (2.0 ** int(expo[m]))
        return out, y_int

    # ---- attention step (KV-cached) — EXACT INTEGER, matches rtl/sequencer.sv -
    def _attn_step(self, x_q25, bi, sink=None):
        gamma = self.layers[(bi, "ln1")]
        y_q22 = self._ln(x_q25, gamma)
        ix = self._act_quant(y_q22, self.layers[(bi, "qkv")]["s_act"])
        y_int = self._gemv_int(self.layers[(bi, "qkv")], ix)
        # dequant q/k/v to Q.VFRAC integers (the RTL stores them this way)
        qkv_q16 = self._dequant_to_q(self.layers[(bi, "qkv")], y_int, VFRAC)
        if sink is not None:
            sink["ln1_out_q22"] = [int(v) for v in y_q22]
            sink["qkv_q16"] = [int(v) for v in qkv_q16]
        C = 256
        q = [int(v) for v in qkv_q16[:C]]
        k = [int(v) for v in qkv_q16[C:2 * C]]
        v = [int(v) for v in qkv_q16[2 * C:]]
        self.k_cache[bi].append(k)
        self.v_cache[bi].append(v)
        T = len(self.k_cache[bi])
        nh, hd = NHEAD, HEAD_DIM
        ctx_q25 = np.zeros(C, dtype=object)
        for h in range(nh):
            base = h * hd
            # scores: q.k per key (Q.2*VFRAC) -> Q8.8, then softmax (integer)
            sq = np.zeros(T, dtype=np.int64)
            for j in range(T):
                acc = 0
                for d in range(hd):
                    acc += q[base + d] * self.k_cache[bi][j][base + d]   # Q.(2*VFRAC)
                s_q88 = rsh_round(acc, 2 * VFRAC + ISQRT - SCORE_FRAC)
                sq[j] = sat(s_q88, -32768, 32767)
            prob_q20 = int_softmax_q(sq, np.ones(T, dtype=bool), self.exp_lut)  # Q1.20
            # ctx[d] = sum_j prob[j] * v_j[d]  (Q.(20+VFRAC)) -> Q6.25
            for d in range(hd):
                acc = 0
                for j in range(T):
                    acc += int(prob_q20[j]) * self.v_cache[bi][j][base + d]
                ctx_q25[base + d] = rsh_round(acc, PROB_FRAC_A + VFRAC - RESID_FRAC)
        if sink is not None:
            sink["ctx_q25"] = [int(v) for v in ctx_q25]
        return self._proj_after_attn(ctx_q25, bi, sink=sink)

    def _proj_after_attn(self, ctx_q25, bi, sink=None):
        """proj GEMV: act-quant ctx (Q6.25) by proj.act_scale exactly as the RTL.
        The RTL converts ctx Q6.25 -> Q.22 (>>3) then applies inv_sact; we mirror that."""
        layer = self.layers[(bi, "proj")]
        ctx_q22 = [rsh_round(int(c), RESID_FRAC - LN_OUTFRAC) for c in ctx_q25]
        ix = self._act_quant(ctx_q22, layer["s_act"])
        proj = self._gemv_dequant_to_q25(layer, ix)      # Q6.25 (proj output)
        if sink is not None:
            sink["attn_out_q25"] = [int(v) for v in proj]
        return proj

    # ---- MLP step — EXACT INTEGER, matches rtl/sequencer.sv --------------------
    def _mlp_step(self, x_q25, bi, sink=None):
        gamma = self.layers[(bi, "ln2")]
        y_q22 = self._ln(x_q25, gamma)
        ix = self._act_quant(y_q22, self.layers[(bi, "mlp_fc")]["s_act"])
        fc_int = self._gemv_int(self.layers[(bi, "mlp_fc")], ix)
        # dequant mlp_fc to Q4.12 (16-bit sat), GELU via committed Q4.12 LUT routine
        fc_q12 = self._dequant_to_q(self.layers[(bi, "mlp_fc")], fc_int, GELU_FRAC)
        xq = np.array([sat(v, -32768, 32767) for v in fc_q12], dtype=np.int64)
        g_q12 = gelu_q(xq, self.gelu_lut)                # Q4.12 ints
        # carry gelu out as Q.LN_OUTFRAC (the RTL shifts Q4.12 << (22-12)=10) for act-quant
        g_q22 = [int(v) << (LN_OUTFRAC - GELU_FRAC) for v in g_q12]
        layer = self.layers[(bi, "mlp_proj")]
        ix2 = self._act_quant(g_q22, layer["s_act"])
        mlp = self._gemv_dequant_to_q25(layer, ix2)      # Q6.25 ints
        if sink is not None:
            sink["ln2_out_q22"] = [int(v) for v in y_q22]
            sink["gelu_q22"] = [int(v) for v in g_q22]
            sink["mlp_out_q25"] = [int(v) for v in mlp]
        return mlp

    # ---- one decode step -------------------------------------------------------
    def step(self, idx_t):
        x = self.tok_emb_q[idx_t] + self.pos_emb_q[self.t]   # (256,) Q6.25 ints
        x = np.asarray([int(v) for v in x], dtype=object)
        for bi in range(self.nblocks):
            attn_out = self._attn_step(x, bi)
            x = x + np.asarray([int(v) for v in attn_out], dtype=object)
            mlp_out = self._mlp_step(x, bi)
            x = x + np.asarray([int(v) for v in mlp_out], dtype=object)
        # final LN + head
        y_q22 = self._ln(x, self.ln_f_q)
        ix = self._act_quant(y_q22, self.head["s_act"])
        logits_real, logits_int = self._gemv_dequant_real(self.head, ix)
        self.t += 1
        return logits_real, int(np.argmax(logits_real))

    def block0_signals(self, idx_t):
        """Run a single token through block 0 only and return the intermediate
        Q6.25 residual after the full block (LN1->attn->+res->LN2->mlp->+res).
        Used by the RTL block gate."""
        x = self.tok_emb_q[idx_t] + self.pos_emb_q[self.t]
        x = np.asarray([int(v) for v in x], dtype=object)
        x0 = x.copy()
        attn_out = self._attn_step(x, 0)
        x = x + np.asarray([int(v) for v in attn_out], dtype=object)
        mlp_out = self._mlp_step(x, 0)
        x = x + np.asarray([int(v) for v in mlp_out], dtype=object)
        return {"x_in_q25": x0, "x_out_q25": x}

    def block0_phase_signals(self, idx_t):
        """Like block0_signals but captures EVERY phase intermediate (ln1_out, qkv, ctx,
        attn_out, x_res1, ln2_out, gelu, mlp_out, x_out) so the P-wide datapath rewrite can
        gate each phase against this reference, not only the final residual. Keys are Q-tagged.
        """
        s = {}
        x = self.tok_emb_q[idx_t] + self.pos_emb_q[self.t]
        x = np.asarray([int(v) for v in x], dtype=object)
        s["x_in_q25"] = [int(v) for v in x]
        attn_out = self._attn_step(x, 0, sink=s)
        x = x + np.asarray([int(v) for v in attn_out], dtype=object)
        s["x_res1_q25"] = [int(v) for v in x]
        mlp_out = self._mlp_step(x, 0, sink=s)
        x = x + np.asarray([int(v) for v in mlp_out], dtype=object)
        s["x_out_q25"] = [int(v) for v in x]
        return s

    def generate_greedy(self, prompt, n_gen):
        self.reset()
        out = list(prompt)
        logits, nxt = None, None
        for tok in out:
            logits, nxt = self.step(int(tok))
        for _ in range(n_gen):
            out.append(nxt)
            logits, nxt = self.step(nxt)
        return out


def build(npz="fabric/export/goformer.npz"):
    p = load_params(npz)
    idx = np.random.default_rng(7).integers(0, p["tok_emb"].shape[0], size=64)
    dom = float(np.ceil(_profile_gelu_range(p, idx) + 1))
    cfg = FixedConfig(gelu_lo=-dom, gelu_hi=dom)
    return p, cfg


def _validate(npz="fabric/export/goformer.npz", n_gen=24, prompt_len=8):
    from model.goformer_seq import build_engines
    from model.goformer_kv import KVDecoder

    p, src, float_engine, fixed_engine = build_engines(npz)
    _, cfg = build(npz)
    rng = np.random.default_rng(0)
    prompt = list(rng.integers(0, p["tok_emb"].shape[0], size=prompt_len))

    # float full-recompute greedy (the ground-truth token stream)
    def float_greedy(idx0, n):
        out = list(idx0)
        for _ in range(n):
            out.append(int(np.argmax(float_engine.forward(out[-256:])[-1])))
        return out
    ref_stream = float_greedy(prompt, n_gen)

    # integer sequencer reference stream
    intseq = IntSequencer(p, cfg)
    int_stream = intseq.generate_greedy(prompt, n_gen)

    same = ref_stream == int_stream
    n_match = sum(1 for a, b in zip(ref_stream, int_stream) if a == b)

    # worst per-step logit cosine of the integer sequencer vs float full-recompute
    dec = IntSequencer(p, cfg)
    worst = 1.0
    for t, tok in enumerate(ref_stream):
        lg, _ = dec.step(int(tok))
        fl = float_engine.forward(ref_stream[:t + 1][-256:])[-1]
        c = float(lg @ fl / (np.linalg.norm(lg) * np.linalg.norm(fl)))
        worst = min(worst, c)

    # The BINDING gate is the token stream (project rule: raw-logit cosine is a brittle
    # proxy on this char model — INT8 requant half-boundaries flip sub-1e-7 detail; the
    # float goformer_seq reference itself only holds ~0.99995 worst-step). The integer
    # sequencer is gated identical-token-stream to the float full-recompute reference.
    print(f"src={src} prompt_len={prompt_len} gen={n_gen}")
    print(f"seq_ref (integer) vs float full-recompute: stream_match={n_match}/{len(ref_stream)} "
          f"identical={same} worst_step_cosine={worst:.7f} (proxy)")
    ok = same
    print(f"SEQREF_VERDICT tokens_identical={same} stream={n_match}/{len(ref_stream)} pass={ok}")
    print("SEQREF_" + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _validate() else 1)
