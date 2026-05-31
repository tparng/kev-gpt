"""The fabric contract: one source of truth shared by the exporter, the golden
reference, the roofline model, and the RTL.

Every number the software side and the hardware side must agree on lives here so
they cannot drift. If you change the PE width, the INT4 packing order, or the
target part, change it once, here.

Nothing in this module imports torch or numpy — it is plain data so the RTL
generators and the pure-Python tests can import it without pulling in the ML
stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Target device — Kria KV260 SOM (Zynq UltraScale+ MPSoC)
# --------------------------------------------------------------------------- #
# The SOM's PL device. Used verbatim by the Vivado build scripts.
PART = "xck26-sfvc784-2LV-c"
BOARD = "xilinx.com:kv260_som:part0:1.4"  # board part, if board files installed

# PL resources, from the device datasheet / doc 2's table. Approximate but
# close enough to plan and to sanity-check utilisation reports against.
DSP48E2 = 1248
BRAM_KB = 5 * 1024 // 8          # ~5 Mb  -> ~640 KB
URAM_KB = 18 * 1024 // 8         # ~18 Mb -> ~2304 KB
ONCHIP_WEIGHT_BUDGET_BYTES = 3 * 1024 * 1024  # ~3 MB usable ceiling (hard wall)

# Bandwidth, doc 2. DDR is shared by PS+PL — the wall. On-chip is the escape.
DDR_BW_GBPS = 20.0               # shared DDR4 controller, PS+PL
# On-chip aggregate read BW is a range; we carry low/high so the roofline can
# show a band rather than false precision. Derived in roofline.py from URAM/BRAM
# port widths × clock, kept here as the headline estimates doc 2 cites.
ONCHIP_BW_GBPS_LOW = 200.0
ONCHIP_BW_GBPS_HIGH = 1000.0

# Fabric clock the systolic GEMV is designed to close timing at.
TARGET_FCLK_MHZ = 300.0          # conservative; 400-500 is the stretch goal

# A53 INT8 baseline ceiling (ARMv8.0-A, pre-SDOT). Peak, not sustained.
A53_INT8_GOPS = 75.0


# --------------------------------------------------------------------------- #
# Model geometry — the deployable Kevin GPT (matches model/gpt.py defaults)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelDims:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256            # d_model
    block_size: int = 256        # context length (KV stays on-chip)
    vocab_size: int = 46         # char-level; overwritten from meta.json at export
    mlp_ratio: int = 4

    @property
    def d_mlp(self) -> int:
        return self.mlp_ratio * self.n_embd

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def linear_shapes(self) -> dict[str, tuple[int, int]]:
        """Every quantised weight matrix as (out_features, in_features), the
        PyTorch nn.Linear convention. These are the matrices the GEMV streams
        and the exporter packs."""
        d, dm = self.n_embd, self.d_mlp
        return {
            "qkv":      (3 * d, d),
            "attn_proj": (d, d),
            "mlp_fc":   (dm, d),
            "mlp_proj": (d, dm),
        }

    def weight_params(self) -> int:
        """Quantised (Linear) parameter count across all layers — the bytes the
        GEMV streams per token. Excludes embeddings/pos/norm (kept FP)."""
        per_layer = sum(o * i for o, i in self.linear_shapes().values())
        return per_layer * self.n_layer


# --------------------------------------------------------------------------- #
# Quantisation contract — must match model/qgpt.py (Brevitas)
# --------------------------------------------------------------------------- #
WEIGHT_BITS = 4                  # INT4 weights
ACT_BITS = 8                     # INT8 activations
WEIGHT_SIGNED = True
# Symmetric signed INT4 range. Brevitas' default (narrow_range=False) uses the
# full two's-complement range [-8, 7]. The exporter reads the *actual* learned
# scales out of the Brevitas module rather than recomputing, so this constant is
# only for documentation and the synthetic round-trip tests.
WEIGHT_QMIN = -8
WEIGHT_QMAX = 7
ACT_QMIN = -128
ACT_QMAX = 127


# --------------------------------------------------------------------------- #
# Systolic GEMV / URAM packing layout — the HW<->SW data contract
# --------------------------------------------------------------------------- #
# Decode is batch-1, so each Linear is a matrix-vector product
#   y[m] = sum_k  W[m, k] * x[k]      for m in 0..M-1, k in 0..K-1
# The array has PE_LANES processing elements; each lane owns a set of output
# rows and streams that row's K weights from its own URAM bank, aligned with the
# shared activation stream x[0..K-1]. M is padded to a multiple of PE_LANES.
PE_LANES = 16

# INT4 packing: two signed nibbles per byte along the contiguous K (reduction)
# axis. Low nibble = even k, high nibble = odd k. K is even for every matrix
# here (256, 1024). Weights are stored row-major over (M, K) so a lane reads its
# row's bytes sequentially. This order is the contract the RTL unpacker assumes.
PACK_NIBBLES_PER_BYTE = 2
PACK_LOW_NIBBLE_IS_EVEN_K = True

# Memory-init file format for $readmemh: one packed byte per line, two hex
# chars, row-major (m outer, k/2 inner). Scales go in a separate file as IEEE
# fixed-point (see export_fabric for the chosen scale representation).
MEM_BYTES_PER_LINE = 1


def onchip_fits(dims: ModelDims, weight_bits: int = WEIGHT_BITS) -> tuple[int, bool]:
    """Return (weight_bytes_on_chip, fits_under_budget) for the given precision.
    The load-bearing check: the whole model must sit under the ~3 MB ceiling or
    the fabric is back behind the DDR wall."""
    nbytes = dims.weight_params() * weight_bits // 8
    return nbytes, nbytes <= ONCHIP_WEIGHT_BUDGET_BYTES


if __name__ == "__main__":
    d = ModelDims()
    wb, fits = onchip_fits(d)
    print(f"part            {PART}")
    print(f"model           L={d.n_layer} d={d.n_embd} heads={d.n_head} "
          f"ctx={d.block_size} vocab={d.vocab_size}")
    print(f"linear params   {d.weight_params():,}")
    print(f"INT4 weights    {wb/1024:.1f} KB  (budget {ONCHIP_WEIGHT_BUDGET_BYTES/1024:.0f} KB)"
          f"  -> {'FITS' if fits else 'SPILLS TO DDR'}")
    print(f"PE lanes        {PE_LANES}  @ {TARGET_FCLK_MHZ:.0f} MHz")
    for name, (o, i) in d.linear_shapes().items():
        print(f"  {name:9s} [{o:4d} x {i:4d}]  K={i} {'(even)' if i % 2 == 0 else '(ODD!)'}")
