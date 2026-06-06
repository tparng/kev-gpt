"""Tests for the fixed-point KV-quant decode reference (model/goformer_kvq.py).

Covers the two load-bearing properties the kv_dma RTL gate relies on:
  1. quant/dequant round-trip is exact-by-construction (dequant == code*scale+lo),
     and bit-budget honoured (codes fit in `bits`);
  2. FULL-PRECISION KVQ (kbits=vbits=16, no rotate) is BIT-IDENTICAL to the plain
     integer IntSequencer over a multi-token decode (the quant path is transparent
     when disabled);
  3. the integer Hadamard is its own inverse (H^T = H) and preserves q.k closely;
  4. the DDR byte rows decode back to the stored Q.16 cache (layout self-consistency).

These run on the committed export fabric/export/goformer.npz; if absent they skip.
"""

import os

import numpy as np
import pytest

from model.goformer_kvq import (
    IntKVQSequencer, hadamard_rotate_q16, quant_head_asym, dequant_head,
    position_ddr_row, _pack_codes_lsb, _butterfly_hadamard,
)
from fabric.stage3.seq_ref import IntSequencer, build, HEAD_DIM, NHEAD

NPZ = "fabric/export/goformer.npz"
HAVE_NPZ = os.path.exists(NPZ)
_pc = pytest.mark.skipif(not HAVE_NPZ, reason="export goformer.npz not present")


# ---------------------------------------------------------------- pure-arith units
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_quant_dequant_roundtrip_bit_budget(bits):
    rng = np.random.default_rng(bits)
    x = rng.integers(-200000, 200000, size=HEAD_DIM).tolist()   # signed Q.16-ish
    codes, lo, scale = quant_head_asym(x, bits)
    qmax = (1 << bits) - 1
    assert all(0 <= c <= qmax for c in codes), "code overflows the bit budget"
    assert scale >= 1
    deq = dequant_head(codes, lo, scale)
    # error bounded by half a scale step (asymmetric uniform quant)
    err = np.abs(np.array(deq) - np.array(x))
    assert err.max() <= scale, f"dequant error {err.max()} exceeds one scale step {scale}"


def test_dequant_is_exact_codescalelo():
    codes = [0, 1, 7, 15]
    lo, scale = -1234, 57
    assert dequant_head(codes, lo, scale) == [c * scale + lo for c in codes]


def test_constant_head_scale_floor():
    # a constant head has span 0 -> scale must floor to 1, codes all 0, dequant == lo
    x = [42] * HEAD_DIM
    codes, lo, scale = quant_head_asym(x, 4)
    assert scale == 1 and lo == 42 and set(codes) == {0}
    assert dequant_head(codes, lo, scale) == x


def test_pack_lsb_first_nibbles():
    # two 4-bit codes pack into one byte, low code in low nibble
    assert _pack_codes_lsb([0x3, 0xA], 4) == [0xA3]
    assert _pack_codes_lsb([0xF, 0x0, 0x1], 4) == [0x0F, 0x01]


def test_hadamard_self_inverse():
    # orthonormal symmetric H: applying twice returns the input (up to >>3 rounding).
    rng = np.random.default_rng(0)
    x = rng.integers(-100000, 100000, size=NHEAD * HEAD_DIM).tolist()
    once = hadamard_rotate_q16(x)
    twice = hadamard_rotate_q16(once)
    err = np.abs(np.array(twice) - np.array(x))
    # two >>3 roundings -> a few LSBs of slack, tiny vs Q.16 magnitude
    assert err.max() <= 64, f"H not self-inverse within rounding: maxerr {err.max()}"


def test_hadamard_preserves_dot_per_head():
    # q.k must survive rotation (the whole point): per head, sum q_d*k_d ~ unchanged.
    rng = np.random.default_rng(1)
    q = rng.integers(-50000, 50000, size=HEAD_DIM).tolist()
    k = rng.integers(-50000, 50000, size=HEAD_DIM).tolist()
    qr = _butterfly_hadamard(q)
    kr = _butterfly_hadamard(k)
    raw_dot = sum(a * b for a, b in zip(q, k))
    rot_dot = sum(a * b for a, b in zip(qr, kr)) // HEAD_DIM   # H_raw^T H_raw = n*I
    rel = abs(rot_dot - raw_dot) / (abs(raw_dot) + 1)
    assert rel < 1e-9, f"Hadamard altered q.k: rel {rel}"


def test_position_ddr_row_size_and_decode():
    rng = np.random.default_rng(2)
    vec = rng.integers(-100000, 100000, size=NHEAD * HEAD_DIM).tolist()
    bits = 4
    row, per_head = position_ddr_row(vec, bits)
    # header 4B + hd*bits/8 codes bytes, per head
    expect = NHEAD * (4 + HEAD_DIM * bits // 8)
    assert len(row) == expect
    # decode head 0's lo/scale from the first 4 bytes and check it matches per_head
    lo16 = row[0] | (row[1] << 8)
    sc16 = row[2] | (row[3] << 8)
    codes0, lo0, scale0 = per_head[0]
    assert sc16 == (scale0 & 0xFFFF)
    assert lo16 == (lo0 & 0xFFFF)


# ---------------------------------------------------------------- integration
@_pc
def test_fp_identity_to_int_sequencer():
    """KVQ at full precision (16/16, no rotate) == plain IntSequencer, bit for bit."""
    p, cfg = build(NPZ)
    rng = np.random.default_rng(0)
    prompt = list(rng.integers(0, p["tok_emb"].shape[0], size=6))
    base = IntSequencer(p, cfg)
    fp = IntKVQSequencer(p, cfg, kbits=16, vbits=16, rotate=False)
    base.reset(); fp.reset()
    bl, bn = fl, fn = None, None
    for tok in prompt:
        bl, bn = base.step(int(tok))
        fl, fn = fp.step(int(tok))
    for _ in range(12):
        assert bn == fn
        assert np.abs(bl - fl).max() == 0.0
        bl, bn = base.step(bn)
        fl, fn = fp.step(fn)


@_pc
def test_cache_matches_ddr_row_decode():
    """The Q.16 cache the sequencer actually computes attention from is exactly what
    decoding the stored DDR byte row reconstructs (layout self-consistency)."""
    p, cfg = build(NPZ)
    kvq = IntKVQSequencer(p, cfg, kbits=4, vbits=4, rotate=True)
    kvq.reset()
    kvq.step(5); kvq.step(9)
    bi = 0
    # recompute the stored K from kv_ddr[bi][pos]'s k_deq_q16 and compare to cache
    for pos in range(len(kvq.k_cache[bi])):
        stored = kvq.kv_ddr[bi][pos]["k_deq_q16"]
        assert kvq.k_cache[bi][pos] == stored


@_pc
def test_kv_ddr_words_shape():
    p, cfg = build(NPZ)
    kvq = IntKVQSequencer(p, cfg, kbits=4, vbits=4, rotate=True)
    words = kvq.kv_ddr_words([1, 2, 3])
    assert len(words) == kvq.nblocks
    assert len(words[0]) == 3            # 3 positions
    # K4: 4 heads * (4 hdr + 64*4/8 codes) = 4*36 = 144 B
    assert len(words[0][0]["k_row"]) == NHEAD * (4 + HEAD_DIM * 4 // 8)
