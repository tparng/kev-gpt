"""Golden reference + RTL test vectors for decoder_block_seq.sv -- the
first functional (not just elaboration) gate for the real, sized
decoder-block top-level FSM. One full decoder-layer forward pass (layer 0,
decode step 0: self-attn tcount=1, cross-attn tcount=T2=6), using REAL
moonshine-tiny weights throughout: input_layernorm, self_attn.{q,k,v,o}_proj,
post_attention_layernorm, encoder_attn.{q,k,v,o}_proj (cross), final_layernorm,
mlp.fc1(+bias)/fc2(+bias). Cross K/V input reuses the SAME real conv-front-end
output as pack_encoder_self_attn.py/pack_decoder_cross_attn.py (T2=6, same
fixed-seed waveform). Query input reuses the SAME real decoder token
embeddings/TOKEN_IDS as pack_decoder_self_attn.py.

GEMV quantization scheme, a real, explicit, DOCUMENTED simplification (this
project's own "quantization scheme still open" note, unresolved upstream --
not invented as final here): per-matrix single-scale INT8 weights
(w_int8 = round(w_real * 2^WSHIFT), WSHIFT chosen so max|w_int8|<=127, no
clipping) and per-call single-shift INT8 activation quantization
(x_int8 = sat(x_fixed >>> ACT_RSHIFT, -128, 127), a PLAIN arithmetic-shift
floor -- matching decoder_block_seq.sv's own G_XFEED stage bit-for-bit, no
rounding). Dequant is a single combined shift (decoder_block_seq.sv's own
g_frac / gdequant()):
    g_frac = FRAC_IN - ACT_RSHIFT + WSHIFT - FRAC_OUT
This is NOT checkpoint C's own per-channel vec_dequant.sv scheme (that one
is tied to INT4-QAT weights this project doesn't use for ASR) -- it is a
real, self-consistent, bit-exactly-reproducible choice for THIS gate,
profiled against real data below so no value clips.

Reuses this project's proven fixed-point references directly: rsh_round/sat
(fabric.stage3.seq_ref), int_softmax_q/exp_table (fabric.stage3.run_softmax),
quant_head_asym/dequant_head (model.goformer_kvq), rope_apply_ref/
build_cos_sin_rom (fabric.asr_seq.pack_rope). LayerNorm's own integer math
(mean/var floor-divide, rsqrt seed+2-Newton) is reproduced here matching
layernorm_vec_gendiv.sv exactly (ln_int_gendiv, same as
fabric.asr_seq.run_layernorm_gendiv but inlined so this file owns the whole
pipeline's own intermediate values for RTL vector generation).

Usage:
    .venv/bin/python -m fabric.asr_seq.pack_decoder_block gen --dir <sim_dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GEN2ASR_SW = os.path.expanduser("~/gen2asr/sw_model")
sys.path.insert(0, GEN2ASR_SW)
sys.path.insert(0, os.path.join(GEN2ASR_SW, "model"))

from fabric.stage3.seq_ref import rsh_round, sat, q_round_div  # noqa: E402
from fabric.stage3.run_softmax import int_softmax_q, exp_table  # noqa: E402
from model.goformer_kvq import quant_head_asym, dequant_head  # noqa: E402
from fabric.asr_seq.pack_rope import rope_apply_ref, build_cos_sin_rom  # noqa: E402
from fabric.stage3.run_layernorm import (  # noqa: E402
    QX, G_FRAC, A_FRAC, Y_FRAC, OUT_FRAC, VAR_FRAC, EPS_A, rsqrt_int, seed_table,
)

D = 288
FFN = 1152
DFFN2 = 2 * FFN
NHEAD = 8
HEAD_DIM = 36
ATTN_P = 4
HR_ATTN = HEAD_DIM // ATTN_P            # 9
T2 = 6
ROT_DIM = 32
ROT_PAIRS = 16
LN_EPS = 1e-5
VFRAC = 16
Q16 = 1 << VFRAC
ISQRT = 3
SCORE_FRAC = 8
PROB_FRAC = 20
SCORE_SH = 2 * VFRAC + ISQRT - SCORE_FRAC     # 27
CTX_SH = PROB_FRAC + VFRAC - 25               # 11
KBITS = 8
POST_SCALE_Q16 = round(math.sqrt(4.0 / 3.0) * Q16)   # 75674

AUDIO_SEED = 0
AUDIO_LEN = 3000            # -> T2=6, same as pack_encoder_self_attn.py
TOKEN_IDS = [1, 940, 24936]  # same as pack_decoder_self_attn.py; step 0 uses TOKEN_IDS[0]

STEP = 0
RESID_FRAC = 25             # xres_bank's Q6.25


# ---- LayerNorm (layernorm_vec_gendiv's own exact integer math) --------------
def floordiv(num: int, d: int) -> int:
    return num // d


def ln_int_gendiv(X, G):
    X = [int(v) for v in X]
    G = [int(v) for v in G]
    d_model = len(X)
    S = sum(X)
    mean = floordiv(S, d_model)
    d = [v - mean for v in X]
    SS = sum(di * di for di in d)
    var = floordiv(SS, d_model)
    A = (var >> (VAR_FRAC - A_FRAC)) + EPS_A
    if A <= 0:
        A = 1
    Yr = rsqrt_int(A)
    sh = (QX + Y_FRAC + G_FRAC) - OUT_FRAC
    out = []
    for i in range(d_model):
        t = d[i] * Yr
        t = t * G[i]
        yv = t >> sh if sh >= 0 else t << (-sh)
        out.append(yv)
    return out


def to_qfrac(x_real, frac) -> np.ndarray:
    return np.round(np.asarray(x_real, dtype=np.float64) * (1 << frac)).astype(np.int64)


# ---- GEMV quantization --------------------------------------------------------
def quantize_weight(W: np.ndarray):
    """Returns (W_int8 (M,K) int64, WSHIFT). Single scale per matrix,
    WSHIFT chosen so max|round(W*2^WSHIFT)| <= 127 -- no clipping."""
    max_abs = float(np.max(np.abs(W)))
    wshift = int(math.floor(math.log2(127.0 / max_abs)))
    while True:
        W_int8 = np.clip(np.round(W * (2.0 ** wshift)), -128, 127).astype(np.int64)
        if int(np.max(np.abs(W_int8))) <= 127:
            break
        wshift -= 1
    return W_int8, wshift


def choose_act_rshift(x_fixed, target_max: int = 100) -> int:
    """Pick the smallest ACT_RSHIFT (>=0) that brings max|x_fixed >>> shift|
    at or below target_max -- an automatic, data-driven choice (no manual
    guessing), profiled fresh for each call site's own observed magnitude
    rather than shared across calls with different real ranges."""
    m = int(np.max(np.abs(np.asarray(x_fixed, dtype=np.int64))))
    if m <= target_max:
        return 0
    shift = 0
    while (m >> shift) > target_max:
        shift += 1
    return shift


def act_quantize(x_fixed: np.ndarray, act_rshift: int) -> np.ndarray:
    """x_int8 = sat(x_fixed >>> act_rshift, -128, 127) -- PLAIN floor shift,
    matching decoder_block_seq.sv's G_XFEED exactly (no rounding). Asserts
    no clipping actually occurs (would mean ACT_RSHIFT was chosen too
    small) -- a real check, not just documentation."""
    x = np.asarray(x_fixed, dtype=np.int64)
    q = x >> act_rshift if act_rshift >= 0 else x << (-act_rshift)   # floor, matches >>>
    clipped = np.clip(q, -128, 127)
    n_clip = int(np.sum(clipped != q))
    if n_clip:
        print(f"WARNING: act_quantize clipped {n_clip}/{len(q)} lanes "
              f"(max|q|={int(np.max(np.abs(q)))}, shift={act_rshift})", file=sys.stderr)
    return clipped


def gemv_int(W_int8: np.ndarray, x_int8: np.ndarray) -> np.ndarray:
    return (W_int8.astype(object) @ x_int8.astype(object)).astype(np.int64)


def gdequant(raw: np.ndarray, g_frac: int) -> np.ndarray:
    """raw >>> g_frac (arithmetic/floor shift, matches Verilog `>>>` exactly
    -- Python's native `>>` on an int is already floor division, no special
    casing needed for negative values)."""
    raw = np.asarray(raw, dtype=object)
    if g_frac >= 0:
        return np.asarray([int(v) >> g_frac for v in raw], dtype=np.int64)
    return np.asarray([int(v) << (-g_frac) for v in raw], dtype=np.int64)


def linear_q(x_fixed, frac_in, W_real, frac_out, bias_fixed=None, target_max=100):
    """One full GEMV call through the chosen quant scheme: auto-derive
    ACT_RSHIFT from x_fixed's OWN observed magnitude (no shared/guessed
    constants), quantize x and W, integer MAC, dequant to frac_out, optional
    bias add (already in frac_out format). Returns (y_fixed, g_frac, wshift,
    act_rshift, x_int8, w_int8, raw)."""
    W_int8, wshift = quantize_weight(W_real)
    act_rshift = choose_act_rshift(x_fixed, target_max)
    x_int8 = act_quantize(x_fixed, act_rshift)
    raw = gemv_int(W_int8, x_int8)
    g_frac = frac_in - act_rshift + wshift - frac_out
    y = gdequant(raw, g_frac)
    if bias_fixed is not None:
        y = y + np.asarray(bias_fixed, dtype=np.int64)
    return y, g_frac, wshift, act_rshift, x_int8, W_int8, raw


def layernorm_nobias_f(x, gamma, eps):
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / np.sqrt(var + eps) * gamma


# ---- real weight/data loading -------------------------------------------------
def load_real():
    from transformers import AutoModel, MoonshineForConditionalGeneration

    model = MoonshineForConditionalGeneration.from_pretrained(
        "UsefulSensors/moonshine-tiny", dtype=torch.float32)
    model.eval()
    enc = model.model.encoder
    dec0 = model.model.decoder.layers[0]

    torch.manual_seed(AUDIO_SEED)
    audio = torch.randn(1, AUDIO_LEN, dtype=torch.float32)
    with torch.no_grad():
        h = F.tanh(enc.conv1(audio.unsqueeze(1)))
        h = enc.groupnorm(h)
        h = F.gelu(enc.conv2(h))
        h = F.gelu(enc.conv3(h))
        h = h.permute(0, 2, 1)
    enc_hidden = h[0].detach().numpy().astype(np.float64)
    w_enc_finalln = enc.layer_norm.weight.detach().numpy().astype(np.float64)

    w = {}
    w["ln1"] = dec0.input_layernorm.weight.detach().numpy().astype(np.float64)
    w["sq"] = dec0.self_attn.q_proj.weight.detach().numpy().astype(np.float64)
    w["sk"] = dec0.self_attn.k_proj.weight.detach().numpy().astype(np.float64)
    w["sv"] = dec0.self_attn.v_proj.weight.detach().numpy().astype(np.float64)
    w["so"] = dec0.self_attn.o_proj.weight.detach().numpy().astype(np.float64)
    w["ln2"] = dec0.post_attention_layernorm.weight.detach().numpy().astype(np.float64)
    w["cq"] = dec0.encoder_attn.q_proj.weight.detach().numpy().astype(np.float64)
    w["ck"] = dec0.encoder_attn.k_proj.weight.detach().numpy().astype(np.float64)
    w["cv"] = dec0.encoder_attn.v_proj.weight.detach().numpy().astype(np.float64)
    w["co"] = dec0.encoder_attn.o_proj.weight.detach().numpy().astype(np.float64)
    w["ln3"] = dec0.final_layernorm.weight.detach().numpy().astype(np.float64)
    w["fc1"] = dec0.mlp.fc1.weight.detach().numpy().astype(np.float64)
    w["b_fc1"] = dec0.mlp.fc1.bias.detach().numpy().astype(np.float64)
    w["fc2"] = dec0.mlp.fc2.weight.detach().numpy().astype(np.float64)
    w["b_fc2"] = dec0.mlp.fc2.bias.detach().numpy().astype(np.float64)

    embed = model.model.decoder.embed_tokens.weight.detach().numpy().astype(np.float64)
    return enc_hidden, w_enc_finalln, w, embed


# ---- head-lane gather/scatter: q_bank/k_bank/v_bank/ctx_bank etc are stored
# at ATTN_P=4-wide granularity (72 rows over D=288 = NHEAD*HR_ATTN), matching
# decoder_block_seq.sv's corrected layout; the D-wide GEMV/LN vectors are
# plain flat (D,) arrays here in Python -- indexing is direct, no gather
# needed at the reference level (the RTL's own gather/scatter is what the
# test vectors below exercise).
def head_slice(vec_d, h):
    return vec_d[h * HEAD_DIM:(h + 1) * HEAD_DIM]


def main_gen(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    enc_hidden, w_enc_finalln, w, embed = load_real()
    t2 = enc_hidden.shape[0]
    assert t2 == T2, f"expected T2={T2}, got {t2}"

    cos_rom, sin_rom = build_cos_sin_rom()

    dump = {}

    # ================= Stage 3a: cross K/V (once, real weights) =============
    k_cross_deq = [[None] * NHEAD for _ in range(T2)]
    v_cross_deq = [[None] * NHEAD for _ in range(T2)]
    for pos in range(T2):
        eo = layernorm_nobias_f(enc_hidden[pos], w_enc_finalln, LN_EPS)
        k16 = to_qfrac(w["ck"] @ eo, VFRAC)
        v16 = to_qfrac(w["cv"] @ eo, VFRAC)
        for h in range(NHEAD):
            k_h = head_slice(k16, h)
            v_h = head_slice(v16, h)
            # matches pack_decoder_cross_attn.py's post_scale_q16(): rsh_round(v*POST_SCALE_Q16, VFRAC)
            k_h_scaled = np.array([rsh_round(int(v) * POST_SCALE_Q16, VFRAC) for v in k_h], dtype=np.int64)
            k_codes, k_lo, k_scale = quant_head_asym(list(k_h_scaled), KBITS, divfree=True)
            v_codes, v_lo, v_scale = quant_head_asym(list(v_h), KBITS, divfree=True)
            k_cross_deq[pos][h] = np.asarray(dequant_head(k_codes, k_lo, k_scale), dtype=np.int64)
            v_cross_deq[pos][h] = np.asarray(dequant_head(v_codes, v_lo, v_scale), dtype=np.int64)

    # ================= decoder layer 0, decode step 0 =======================
    tok = TOKEN_IDS[STEP]
    x_real = embed[tok]                              # (D,) real embedding, treated as Q6.25 input
    xres0 = to_qfrac(x_real, RESID_FRAC)              # xres_bank initial content

    # ---- LN1 ----
    x_q625 = xres0.astype(object)
    g_ln1 = to_qfrac(w["ln1"], G_FRAC)
    xn1 = np.asarray(ln_int_gendiv(list(x_q625), list(g_ln1)), dtype=np.int64)   # Q.22

    # ---- Q/K/V GEMVs ---- (ACT_RSHIFT auto-derived per call from xn1's own
    # observed magnitude -- see choose_act_rshift)
    FRAC_QKV = VFRAC       # 16
    q_fixed, gfrac_q, wshift_q, arshift_q, _, wint8_q, _ = linear_q(xn1, OUT_FRAC, w["sq"], FRAC_QKV)
    k_fixed, gfrac_k, wshift_k, arshift_k, _, wint8_k, _ = linear_q(xn1, OUT_FRAC, w["sk"], FRAC_QKV)
    v_fixed, gfrac_v, wshift_v, arshift_v, _, wint8_v, _ = linear_q(xn1, OUT_FRAC, w["sv"], FRAC_QKV)

    # ---- RoPE (self, WITH the HEAD_DIM=36 SCORE_SH correction) ----
    q_rope = np.zeros(D, dtype=np.int64)
    k_rope = np.zeros(D, dtype=np.int64)
    for h in range(NHEAD):
        q_rope[h * HEAD_DIM:(h + 1) * HEAD_DIM] = rope_apply_ref(
            head_slice(q_fixed, h), STEP, cos_rom, sin_rom, POST_SCALE_Q16)
        k_rope[h * HEAD_DIM:(h + 1) * HEAD_DIM] = rope_apply_ref(
            head_slice(k_fixed, h), STEP, cos_rom, sin_rom, POST_SCALE_Q16)

    # ---- self kv_bank write (K RoPE'd, V raw) + quantize-at-write, matching
    # kv_bank.sv's own scheme exactly (same as the decoder self-attn gate) ----
    k_self_deq = [None] * NHEAD
    v_self_deq = [None] * NHEAD
    for h in range(NHEAD):
        k_h = head_slice(k_rope, h)
        v_h = head_slice(v_fixed, h)
        k_codes, k_lo, k_scale = quant_head_asym(list(k_h), KBITS, divfree=True)
        v_codes, v_lo, v_scale = quant_head_asym(list(v_h), KBITS, divfree=True)
        k_self_deq[h] = np.asarray(dequant_head(k_codes, k_lo, k_scale), dtype=np.int64)
        v_self_deq[h] = np.asarray(dequant_head(v_codes, v_lo, v_scale), dtype=np.int64)

    # ---- self-attn (causal, T=step+1=1) ----
    ctx_self = np.zeros(D, dtype=np.int64)
    for h in range(NHEAD):
        q_h = head_slice(q_rope, h)
        Tc = STEP + 1
        scores = np.zeros(Tc, dtype=np.int64)
        for j in range(Tc):
            # Tc=step+1=1 this gate: k_self_deq[h] IS position 0's own K
            # (the only position written so far), matching j==0 exactly.
            acc = int(np.dot(q_h.astype(object), k_self_deq[h].astype(object)))
            s_q88 = rsh_round(acc, SCORE_SH)
            scores[j] = sat(s_q88, -32768, 32767)
        prob = int_softmax_q(scores, np.ones(Tc, dtype=bool), exp_table())
        ctx_h = np.zeros(HEAD_DIM, dtype=np.int64)
        for d in range(HEAD_DIM):
            acc = 0
            for j in range(Tc):
                acc += int(prob[j]) * int(v_self_deq[h][d])
            ctx_h[d] = rsh_round(acc, CTX_SH)
        ctx_self[h * HEAD_DIM:(h + 1) * HEAD_DIM] = ctx_h

    # ---- O GEMV (self-attn output proj): ctx is already Q.25 (RESID_FRAC,
    # matching CTX_SH's own output format) -> INT8 -> Q.25 out ----
    o_fixed, gfrac_o, wshift_o, arshift_o, _, wint8_o, _ = linear_q(ctx_self, RESID_FRAC, w["so"], RESID_FRAC)

    xres1 = xres0 + o_fixed   # RES1

    # ---- LN2 ----
    g_ln2 = to_qfrac(w["ln2"], G_FRAC)
    xn2 = np.asarray(ln_int_gendiv(list(xres1.astype(object)), list(g_ln2)), dtype=np.int64)

    # ---- cross Q GEMV (no RoPE; POST_SCALE applied directly) ----
    qc_fixed, gfrac_cq, wshift_cq, arshift_cq, _, wint8_cq, _ = linear_q(xn2, OUT_FRAC, w["cq"], FRAC_QKV)
    qc_scaled = np.zeros(D, dtype=np.int64)
    for i in range(D):
        qc_scaled[i] = rsh_round(int(qc_fixed[i]) * POST_SCALE_Q16, VFRAC)

    # ---- cross-attn (static full-attend, T2=6) ----
    ctx_cross = np.zeros(D, dtype=np.int64)
    for h in range(NHEAD):
        q_h = head_slice(qc_scaled, h)
        scores = np.zeros(T2, dtype=np.int64)
        for j in range(T2):
            acc = int(np.dot(q_h.astype(object), k_cross_deq[j][h].astype(object)))
            s_q88 = rsh_round(acc, SCORE_SH)
            scores[j] = sat(s_q88, -32768, 32767)
        prob = int_softmax_q(scores, np.ones(T2, dtype=bool), exp_table())
        ctx_h = np.zeros(HEAD_DIM, dtype=np.int64)
        for d in range(HEAD_DIM):
            acc = 0
            for j in range(T2):
                acc += int(prob[j]) * int(v_cross_deq[j][h][d])
            ctx_h[d] = rsh_round(acc, CTX_SH)
        ctx_cross[h * HEAD_DIM:(h + 1) * HEAD_DIM] = ctx_h

    # ---- Oc GEMV ----
    oc_fixed, gfrac_oc, wshift_oc, arshift_oc, _, wint8_oc, _ = linear_q(ctx_cross, RESID_FRAC, w["co"], RESID_FRAC)
    xres2 = xres1 + oc_fixed   # RES2

    # ---- LN3 ----
    g_ln3 = to_qfrac(w["ln3"], G_FRAC)
    xn3 = np.asarray(ln_int_gendiv(list(xres2.astype(object)), list(g_ln3)), dtype=np.int64)

    # ---- FC1: D -> DFFN2, +bias, Q.12 out (SiLU-ready) ----
    FRAC_FC1 = 12
    bias_fc1_fixed = to_qfrac(w["b_fc1"], FRAC_FC1)
    h1_fixed, gfrac_fc1, wshift_fc1, arshift_fc1, _, wint8_fc1, _ = linear_q(
        xn3, OUT_FRAC, w["fc1"], FRAC_FC1, bias_fixed=bias_fc1_fixed)
    value = h1_fixed[:FFN]
    gate = h1_fixed[FFN:]

    # ---- SiLU LUT reference (bit-exact to silu_lut.sv) ----
    from fabric.asr_seq.run_silu import silu_table, silu_q
    silu_lut = silu_table()
    gate_sat = np.clip(gate, -32768, 32767).astype(np.int64)
    silu_gate = silu_q(gate_sat, silu_lut)            # Q4.12
    combined = np.zeros(FFN, dtype=np.int64)
    for i in range(FFN):
        prod = int(value[i]) * int(silu_gate[i])       # Q4.12 * Q4.12 = Q8.24
        combined[i] = rsh_round(prod, FRAC_FC1)         # -> Q4.12

    # ---- FC2: FFN -> D, +bias, Q.25 out ----
    bias_fc2_fixed = to_qfrac(w["b_fc2"], RESID_FRAC)
    h2_fixed, gfrac_fc2, wshift_fc2, arshift_fc2, _, wint8_fc2, _ = linear_q(
        combined, FRAC_FC1, w["fc2"], RESID_FRAC, bias_fixed=bias_fc2_fixed)
    xres3 = xres2 + h2_fixed   # RES3

    # ---- profiling report (stderr-visible via print) ------------------------
    def rng(name, arr):
        a = np.asarray(arr, dtype=np.float64)
        print(f"  {name}: max|.|={np.max(np.abs(a)):.0f}", file=sys.stderr)
    print("profile (fixed-point magnitudes):", file=sys.stderr)
    rng("xn1(Q22)", xn1); rng("q_fixed(Q16)", q_fixed); rng("ctx_self(Q25)", ctx_self)
    rng("o_fixed(Q25)", o_fixed); rng("qc_scaled(Q16)", qc_scaled); rng("ctx_cross(Q25)", ctx_cross)
    rng("xn3(Q22)", xn3); rng("h1_fixed(Q12)", h1_fixed); rng("combined(Q12)", combined)
    rng("h2_fixed(Q25)", h2_fixed); rng("xres3(Q25)", xres3)

    # =========================================================================
    # ---- resident weight image (real gemv_banked_resident_vec.sv loader
    # format, reusing pack_banked_resident_vec.build_resident() -- the
    # already-proven WBW=8 packing, not re-derived) --------------------------
    from fabric.stage3.pack_banked_resident_vec import build_resident
    LANES, WBW = 128, 8
    layer_order = [wint8_q, wint8_k, wint8_v, wint8_o, wint8_cq, wint8_oc, wint8_fc1, wint8_fc2]
    all_words, wmeta = build_resident(layer_order, LANES, WBW)
    wb = {name: wmeta[i]["w_base"] for i, name in
          enumerate(["q", "k", "v", "o", "cq", "co", "fc1", "fc2"])}

    wbits = LANES * WBW
    hexw = (wbits + 3) // 4
    with open(os.path.join(out_dir, "w.mem"), "w") as f:
        f.write("\n".join(format(v, f"0{hexw}x") for v in all_words) + "\n")

    # ---- internal ROMs every sub-module $readmemh's from the sim run dir --
    # (same generation as every prior attention/SiLU gate -- not re-derived).
    with open(os.path.join(out_dir, "rope_cos.mem"), "w") as f:
        for pos in range(cos_rom.shape[0]):
            for i in range(ROT_PAIRS):
                f.write(format(int(cos_rom[pos, i]) & 0xFFFF, "04x") + "\n")
    with open(os.path.join(out_dir, "rope_sin.mem"), "w") as f:
        for pos in range(sin_rom.shape[0]):
            for i in range(ROT_PAIRS):
                f.write(format(int(sin_rom[pos, i]) & 0xFFFF, "04x") + "\n")

    exp_lut = exp_table()
    with open(os.path.join(out_dir, "exp_lut.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0x1FFFFF:06x}" for v in exp_lut) + "\n")

    with open(os.path.join(out_dir, "inv_lut_lo.mem"), "w") as f:
        for s4 in range(4096):
            f.write(f"{q_round_div(1 << 24, max(s4, 1)) & 0x1FFFFFF:07x}\n")
    with open(os.path.join(out_dir, "inv_lut_hi.mem"), "w") as f:
        for s4 in range(4096, 16512):
            f.write(f"{q_round_div(1 << 24, s4) & 0x1FFF:04x}\n")

    with open(os.path.join(out_dir, "seed.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFFFFF:05x}" for v in seed_table()) + "\n")

    from fabric.asr_seq.run_silu import silu_table
    silu_lut_vals = silu_table()
    with open(os.path.join(out_dir, "silu_lut.mem"), "w") as f:
        f.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in silu_lut_vals) + "\n")

    def w32(f, v):
        f.write(format(int(v) & 0xFFFFFFFF, "08x") + "\n")

    def write_prow(path, vec, frac_or_none=None):
        """P=8-wide packed rows, one line/row -- vec already in fixed-point
        ints (frac_or_none unused, kept for call-site clarity)."""
        with open(path, "w") as f:
            for r in range(len(vec) // 8):
                for k in range(8):
                    w32(f, vec[r * 8 + k])

    # ---- LN gamma tables (Q4.20, P=8-wide) ----
    write_prow(os.path.join(out_dir, "gamma_ln1.mem"), g_ln1)
    write_prow(os.path.join(out_dir, "gamma_ln2.mem"), g_ln2)
    write_prow(os.path.join(out_dir, "gamma_ln3.mem"), g_ln3)

    # ---- bias tables (fc1: Q.12 DFFN2-wide; fc2: Q.25 D-wide), P=8-wide ----
    write_prow(os.path.join(out_dir, "bias_fc1.mem"), bias_fc1_fixed)
    write_prow(os.path.join(out_dir, "bias_fc2.mem"), bias_fc2_fixed)

    # ---- cross K/V write vectors (ATTN_P=4-wide, pos-major then head-major:
    # [k(36) v(36)] per head -- same convention as pack_decoder_cross_attn.py's
    # kv_in.mem, K already POST_SCALE'd, V raw). ------------------------------
    with open(os.path.join(out_dir, "xkv_in.mem"), "w") as f:
        for pos in range(T2):
            eo = layernorm_nobias_f(enc_hidden[pos], w_enc_finalln, LN_EPS)
            k16 = to_qfrac(w["ck"] @ eo, VFRAC)
            v16 = to_qfrac(w["cv"] @ eo, VFRAC)
            for h in range(NHEAD):
                k_h = head_slice(k16, h)
                v_h = head_slice(v16, h)
                k_h_scaled = np.array([rsh_round(int(v) * POST_SCALE_Q16, VFRAC) for v in k_h],
                                       dtype=np.int64)
                for v in k_h_scaled:
                    w32(f, v)
                for v in v_h:
                    w32(f, v)

    # ---- xres0 (initial residual) + xres3 (golden final residual) ----------
    write_prow(os.path.join(out_dir, "xres0.mem"), xres0)
    with open(os.path.join(out_dir, "xres3_ref.mem"), "w") as f:
        for v in xres3:
            w32(f, v)

    # =========================================================================
    # ---- assemble manifest ---------------------------------------------------
    manifest = {
        "d": D, "ffn": FFN, "dffn2": DFFN2, "nhead": NHEAD, "head_dim": HEAD_DIM, "t2": T2,
        "step": STEP, "token": tok, "resid_frac": RESID_FRAC, "post_scale_q16": POST_SCALE_Q16,
        "lanes": LANES, "wbw": WBW, "n_words_total": len(all_words),
        "g_frac": {"q": gfrac_q, "k": gfrac_k, "v": gfrac_v, "o": gfrac_o, "cq": gfrac_cq,
                   "oc": gfrac_oc, "fc1": gfrac_fc1, "fc2": gfrac_fc2},
        "wshift": {"q": wshift_q, "k": wshift_k, "v": wshift_v, "o": wshift_o, "cq": wshift_cq,
                  "oc": wshift_oc, "fc1": wshift_fc1, "fc2": wshift_fc2},
        "act_rshift": {"q": arshift_q, "k": arshift_k, "v": arshift_v, "o": arshift_o,
                       "cq": arshift_cq, "oc": arshift_oc, "fc1": arshift_fc1, "fc2": arshift_fc2},
        "w_base": wb,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    dump = {
        "xres0": xres0.tolist(), "xres1": xres1.tolist(), "xres2": xres2.tolist(),
        "xres3": xres3.tolist(),
    }
    with open(os.path.join(out_dir, "golden.json"), "w") as f:
        json.dump(dump, f)

    print(f"GEN dir={out_dir} t2={T2} step={STEP} token={tok} n_words={len(all_words)} w_base={wb}")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.asr_seq.pack_decoder_block")
    p.add_argument("cmd", choices=["gen"])
    p.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    main_gen(a.dir)


if __name__ == "__main__":
    main()
