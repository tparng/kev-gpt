"""INT4 nibble packing — the bit-exact contract between the exporter, the golden
reference, and the RTL. Pure numpy, no torch, so tests and the golden model
import it without the ML stack.

Layout (see fabric/spec.py): weights are signed INT4 in [-8, 7], stored
row-major over (M_out, K_in). Two nibbles per byte along the contiguous K axis;
the low nibble is the even-k element, the high nibble the odd-k element. K is
even for every matrix in this model. The RTL unpacker assumes exactly this.
"""

from __future__ import annotations

import numpy as np

from .spec import PACK_LOW_NIBBLE_IS_EVEN_K, WEIGHT_QMAX, WEIGHT_QMIN


def to_nibble(int4: np.ndarray) -> np.ndarray:
    """Signed INT4 values in [-8,7] -> unsigned 4-bit two's-complement [0,15]."""
    if int4.min() < WEIGHT_QMIN or int4.max() > WEIGHT_QMAX:
        raise ValueError(f"values out of INT4 range [{WEIGHT_QMIN},{WEIGHT_QMAX}]: "
                         f"[{int4.min()},{int4.max()}]")
    return (int4.astype(np.int16) & 0x0F).astype(np.uint8)


def from_nibble(nib: np.ndarray) -> np.ndarray:
    """Unsigned 4-bit two's-complement [0,15] -> signed INT4 [-8,7]."""
    n = nib.astype(np.int16) & 0x0F
    return np.where(n >= 8, n - 16, n).astype(np.int8)


def pack_int4_rows(int_w: np.ndarray) -> np.ndarray:
    """(M, K) signed INT4 -> (M, K//2) packed bytes, low nibble = even k.

    K must be even. Output is row-major: byte (m, j) holds k=2j in its low
    nibble and k=2j+1 in its high nibble (the default layout)."""
    int_w = np.asarray(int_w, dtype=np.int8)
    if int_w.ndim != 2:
        raise ValueError(f"expected 2-D (M,K), got shape {int_w.shape}")
    M, K = int_w.shape
    if K % 2 != 0:
        raise ValueError(f"K={K} must be even to pack two nibbles per byte")
    nib = to_nibble(int_w)
    even = nib[:, 0::2]   # k = 0,2,4,...
    odd = nib[:, 1::2]    # k = 1,3,5,...
    if PACK_LOW_NIBBLE_IS_EVEN_K:
        packed = (even | (odd << 4)).astype(np.uint8)
    else:
        packed = (odd | (even << 4)).astype(np.uint8)
    return packed


def unpack_int4_rows(packed: np.ndarray, K: int) -> np.ndarray:
    """(M, K//2) packed bytes -> (M, K) signed INT4. Inverse of pack_int4_rows."""
    packed = np.asarray(packed, dtype=np.uint8)
    M, Kb = packed.shape
    if Kb != K // 2:
        raise ValueError(f"packed width {Kb} != K//2 = {K // 2}")
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = np.empty((M, K), dtype=np.uint8)
    if PACK_LOW_NIBBLE_IS_EVEN_K:
        out[:, 0::2] = low
        out[:, 1::2] = high
    else:
        out[:, 0::2] = high
        out[:, 1::2] = low
    return from_nibble(out)


def to_hex_mem(packed: np.ndarray) -> str:
    """Row-major bytes -> $readmemh text: one 2-char hex byte per line, m outer,
    k/2 inner. This is exactly what the RTL's $readmemh expects."""
    flat = packed.reshape(-1)
    return "\n".join(f"{b:02x}" for b in flat) + "\n"


def quantise_per_channel_symmetric(w: np.ndarray, bits: int = 4):
    """Reference symmetric per-output-channel quantiser (for synthetic tests and
    sanity checks). Brevitas is the source of truth at export time; this mirrors
    its arithmetic so round-trip tests can run without torch.

    Returns (int_w [M,K] int8, scale [M] float32). Symmetric: zero_point = 0,
    scale = max(|w_row|) / qmax.
    """
    qmax = (1 << (bits - 1)) - 1            # 7 for INT4
    amax = np.abs(w).max(axis=1, keepdims=True)
    scale = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)
    q = np.round(w / scale).astype(np.int32)
    q = np.clip(q, -(qmax + 1), qmax).astype(np.int8)
    return q, scale.reshape(-1)
