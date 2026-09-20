"""Shared integer reference + packing helpers for conv1d_seq.sv's own gates
(pack_conv1.py/pack_conv2.py/pack_conv3.py) -- Stage 1 conv front-end's conv
engine. See conv1d_seq.sv's own header for the design (K-major/channel-minor
window-as-row-copies, CIN padded up to a multiple of P with zero weight/
input lanes for shapes where the real channel count doesn't divide P).
"""
from __future__ import annotations

import numpy as np

from fabric.stage3.seq_ref import rsh_round

QX = 25   # Q6.25 -- conv1d_seq.sv's own output format (matches layernorm_vec_gendiv.sv/groupnorm1_vec.sv's x_in)


def pad_cin_weight(w, p):
    """(COUT, CIN_REAL, KW) real weight -> (COUT, CIN_PAD, KW), CIN_PAD the
    next multiple of P, new channels zero (their MACs then always contribute 0)."""
    cout, cin_r, kw = w.shape
    cin_pad = ((cin_r + p - 1) // p) * p
    if cin_pad == cin_r:
        return w
    out = np.zeros((cout, cin_pad, kw), dtype=w.dtype)
    out[:, :cin_r, :] = w
    return out


def pad_cin_input(x, p):
    """(CIN_REAL, TIN) real -> (CIN_PAD, TIN), new rows zero."""
    cin_r, tin = x.shape
    cin_pad = ((cin_r + p - 1) // p) * p
    if cin_pad == cin_r:
        return x
    out = np.zeros((cin_pad, tin), dtype=x.dtype)
    out[:cin_r, :] = x
    return out


def transpose_weight(w_padded):
    """(COUT,CIN,KW) -> (COUT, KW*CIN), e = k*CIN+ci (k-major/ci-minor --
    conv1d_seq.sv's own window order, NOT PyTorch's native [out,in,k] flatten)."""
    cout, cin, kw = w_padded.shape
    return w_padded.transpose(0, 2, 1).reshape(cout, kw * cin)


def quantize_weight_per_matrix(w_flat):
    """One shift for the whole matrix -- decoder_block_seq.sv/
    encoder_block_seq.sv's own simpler convention (no giant per-row-sensitive
    argmax downstream like output_head_seq.sv has, so no per-row scheme needed).
    Already uses INT8's full +-127 ceiling directly (no target_max headroom
    -- there never was one here; an earlier `target_max=100.0` default
    parameter existed but was dead code, never read by this function's own
    body, which always searched against the true 127.0 ceiling)."""
    m = float(np.max(np.abs(w_flat)))
    if m <= 0:
        return np.zeros_like(w_flat, dtype=np.int64), 0
    wshift = 0
    while m * (2.0 ** (wshift + 1)) <= 127.0:
        wshift += 1
    while m * (2.0 ** wshift) > 127.0:
        wshift -= 1
    w_int8 = np.clip(np.round(w_flat * (2.0 ** wshift)), -128, 127).astype(np.int64)
    return w_int8, wshift


def quantize_weight_per_row(w_flat):
    """PER-ROW (per-output-channel) INT8 quantization -- one wshift per
    output row instead of quantize_weight_per_matrix's own single shared
    shift. Needed wherever a single shared weight scale under-serves
    output channels whose own dynamic range is much smaller than the
    matrix's own max -- found gating gelu1->conv3 specifically (see
    conv1d_seq.sv's own header). Shape-generic reproduction of
    pack_output_head.py's own quantize_weight_per_row (that file's own
    version is VOCAB-row-specific for the lm_head weight; this one takes
    any (COUT, K) matrix) -- same floor-shift-search against INT8's real
    127 ceiling, same per-row re-check-and-shrink fixup for the rare
    floor-then-round boundary clip."""
    w_flat = np.asarray(w_flat, dtype=np.float64)
    max_abs = np.max(np.abs(w_flat), axis=1)                  # (COUT,)
    max_abs = np.where(max_abs == 0, 1.0, max_abs)
    wshift = np.floor(np.log2(127.0 / max_abs)).astype(np.int64)
    w_int8 = np.round(w_flat * (2.0 ** wshift[:, None]))
    w_int8 = np.clip(w_int8, -128, 127).astype(np.int64)
    bad = np.max(np.abs(w_int8), axis=1) > 127
    while np.any(bad):
        wshift[bad] -= 1
        w_int8[bad] = np.clip(np.round(w_flat[bad] * (2.0 ** wshift[bad, None])), -128, 127).astype(np.int64)
        bad = np.max(np.abs(w_int8), axis=1) > 127
    return w_int8, wshift


def choose_ashift(x_real, target_max=127.0):
    """Smallest right-shift-equivalent scale keeping |x_real*2^ashift| within
    target_max. Was defaulted to 100.0 (leaving ~21% of INT8's own +-127
    range unused, a real, avoidable precision loss compounding across this
    pipeline's several cascaded INT8 boundaries -- found investigating
    gelu1->conv3's own quantization noise, see conv_front_end_seq.sv's own
    header). 127.0 is safe here (not just "close to the edge"): the search
    below is a floor-style shift count, so `m*2^ashift <= target_max` holds
    by construction, and round()-ing any individual element (<=m) to the
    nearest int can't push it past target_max=127 either -- rounding a
    value already <=127.0 never produces >127."""
    m = float(np.max(np.abs(x_real)))
    if m <= 0:
        return 0
    ashift = 0
    while m * (2.0 ** (ashift + 1)) <= target_max:
        ashift += 1
    return ashift


def quantize_act(x_real, ashift):
    q = np.round(np.asarray(x_real, dtype=np.float64) * (2.0 ** ashift))
    return np.clip(q, -128, 127).astype(np.int64)


def conv1d_int_ref(x_int8, w_int8_flat, kw, stride, cout, cin, mant, exp, bias_q=None):
    """Exact-integer conv1d matching conv1d_seq.sv's own per-output-position
    GEMV + per-row vec_dequant.sv dequant scheme bit-for-bit (frac=0, see
    conv1d_seq.sv's own header for why: mant/exp are chosen to already
    target the output format directly, Q4.12 for every real call site in
    this pipeline). x_int8 (CIN,TIN) int8 (padded), w_int8_flat
    (COUT, KW*CIN) int8 (see transpose_weight), mant/exp (COUT,) --
    vec_dequant.sv's own per-row table (fabric.stage3.seq_ref.
    quantize_scale_24's own output, mant unsigned/positive per that
    module's own convention). Returns (TOUT,COUT) int64, truncated to
    32 bits per element (matching vec_dequant.sv's own `dq_out[31:0]`
    write and y_data's own 32-bit storage, bias added AFTER that
    truncation in the SAME 32-bit domain, exactly like the RTL)."""
    cin_ax, tin = x_int8.shape
    assert cin_ax == cin
    tout = (tin - kw) // stride + 1
    w = w_int8_flat.astype(np.int64)                     # (COUT, KW*CIN)
    mant = np.asarray(mant, dtype=np.int64)
    exp = np.asarray(exp, dtype=np.int64)
    left = exp >= 0
    out = np.zeros((tout, cout), dtype=np.int64)
    for t in range(tout):
        win = x_int8[:, t * stride: t * stride + kw].T.reshape(-1).astype(np.int64)  # e=k*CIN+ci
        raw = w @ win                                     # (COUT,) exact int64 -- gemvy
        dq_prod = raw.astype(object) * mant.astype(object)
        dq_val = np.empty(cout, dtype=object)
        dq_val[left] = [int(p) << int(s) for p, s in zip(dq_prod[left], exp[left])]
        dq_val[~left] = [rsh_round(int(p), int(-s)) for p, s in zip(dq_prod[~left], exp[~left])]
        deq = np.array([int(v) & 0xFFFFFFFF for v in dq_val], dtype=np.int64)
        deq = np.where(deq >= 0x80000000, deq - 0x100000000, deq)
        if bias_q is not None:
            deq = (deq + bias_q) & 0xFFFFFFFF
            deq = np.where(deq >= 0x80000000, deq - 0x100000000, deq)
        out[t] = deq
    return out


def pack_rows_p8(vec, p):
    """Flat int8-range values -> list of P*8-bit packed rows (lane k = bits [8k+:8])."""
    rows = []
    n = len(vec)
    for r in range((n + p - 1) // p):
        val = 0
        for k in range(p):
            i = r * p + k
            v = int(vec[i]) if i < n else 0
            val |= (v & 0xFF) << (8 * k)
        rows.append(val)
    return rows


def pack_rows_p32(vec, p):
    """Flat int32-range values -> list of P*32-bit packed rows (lane k = bits [32k+:32])."""
    rows = []
    n = len(vec)
    for r in range((n + p - 1) // p):
        val = 0
        for k in range(p):
            i = r * p + k
            v = int(vec[i]) if i < n else 0
            val |= (v & 0xFFFFFFFF) << (32 * k)
        rows.append(val)
    return rows


def pack_rows_pw(vec, p, w):
    """Flat values -> list of P*w-bit packed rows (lane k = bits [w*k +: w]),
    masked to w bits each -- generic form of pack_rows_p8/p32, used for
    vec_dequant.sv's own per-row mant (w=24) / exp (w=8) tables."""
    mask = (1 << w) - 1
    rows = []
    n = len(vec)
    for r in range((n + p - 1) // p):
        val = 0
        for k in range(p):
            i = r * p + k
            v = int(vec[i]) if i < n else 0
            val |= (v & mask) << (w * k)
        rows.append(val)
    return rows
