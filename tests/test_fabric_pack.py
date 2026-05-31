"""Bit-exactness tests for the INT4 packing contract (fabric/pack.py).

These let us trust the URAM-init files without a board: if pack -> unpack is
identity and the hex layout is what we documented, the RTL reading the same
bytes sees the same weights.
"""

import numpy as np
import pytest

from fabric import pack
from fabric.spec import WEIGHT_QMAX, WEIGHT_QMIN


def test_nibble_roundtrip_full_range():
    vals = np.arange(WEIGHT_QMIN, WEIGHT_QMAX + 1, dtype=np.int8).reshape(1, -1)
    assert np.array_equal(pack.from_nibble(pack.to_nibble(vals)), vals)


def test_nibble_twos_complement_encoding():
    vals = np.array([[-1, -8, 7, 0]], dtype=np.int8)
    nib = pack.to_nibble(vals)
    assert list(nib.reshape(-1)) == [0xF, 0x8, 0x7, 0x0]


def test_to_nibble_rejects_out_of_range():
    with pytest.raises(ValueError):
        pack.to_nibble(np.array([[8]], dtype=np.int8))
    with pytest.raises(ValueError):
        pack.to_nibble(np.array([[-9]], dtype=np.int8))


@pytest.mark.parametrize("M,K", [(1, 2), (4, 8), (16, 256), (768, 256), (256, 1024)])
def test_pack_unpack_roundtrip(M, K):
    rng = np.random.default_rng(0)
    w = rng.integers(WEIGHT_QMIN, WEIGHT_QMAX + 1, size=(M, K)).astype(np.int8)
    packed = pack.pack_int4_rows(w)
    assert packed.shape == (M, K // 2)
    assert packed.dtype == np.uint8
    back = pack.unpack_int4_rows(packed, K)
    assert np.array_equal(back, w)


def test_pack_low_nibble_is_even_k():
    # row [k0=1, k1=2]: low nibble holds k0=1, high holds k1=2 -> byte 0x21
    w = np.array([[1, 2]], dtype=np.int8)
    packed = pack.pack_int4_rows(w)
    assert packed[0, 0] == 0x21


def test_odd_K_rejected():
    with pytest.raises(ValueError):
        pack.pack_int4_rows(np.zeros((2, 3), dtype=np.int8))


def test_hex_mem_format():
    w = np.array([[1, 2, -1, -8]], dtype=np.int8)  # bytes: 0x21, 0x8f
    text = pack.to_hex_mem(pack.pack_int4_rows(w))
    assert text == "21\n8f\n"


def test_reference_quantiser_roundtrips_through_pack():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    q, scale = pack.quantise_per_channel_symmetric(w, bits=4)
    assert q.min() >= WEIGHT_QMIN and q.max() <= WEIGHT_QMAX
    assert scale.shape == (32,)
    back = pack.unpack_int4_rows(pack.pack_int4_rows(q), 64)
    assert np.array_equal(back, q)
