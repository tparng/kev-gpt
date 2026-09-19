"""Shared integer reference + packing helpers for conv1d_seq.sv's own gates
(pack_conv1.py/pack_conv2.py/pack_conv3.py) -- Stage 1 conv front-end's conv
engine. See conv1d_seq.sv's own header for the design (K-major/channel-minor
window-as-row-copies, CIN padded up to a multiple of P with zero weight/
input lanes for shapes where the real channel count doesn't divide P).
"""
from __future__ import annotations

import numpy as np

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


def conv1d_int_ref(x_int8, w_int8_flat, kw, stride, cout, cin, bias_q=None, dq_shift=0):
    """Exact-integer conv1d matching conv1d_seq.sv's own per-output-position
    GEMV scheme bit-for-bit. x_int8 (CIN,TIN) int8 (padded), w_int8_flat
    (COUT, KW*CIN) int8 (see transpose_weight). Returns (TOUT,COUT) int64
    Q6.25 (gdequant()'d, + optional per-channel bias in the same domain)."""
    cin_ax, tin = x_int8.shape
    assert cin_ax == cin
    tout = (tin - kw) // stride + 1
    w = w_int8_flat.astype(np.int64)                     # (COUT, KW*CIN)
    out = np.zeros((tout, cout), dtype=np.int64)
    for t in range(tout):
        win = x_int8[:, t * stride: t * stride + kw].T.reshape(-1).astype(np.int64)  # e=k*CIN+ci
        raw = w @ win                                     # (COUT,) exact int64
        deq = raw >> dq_shift if dq_shift >= 0 else raw << (-dq_shift)
        if bias_q is not None:
            deq = deq + bias_q
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
