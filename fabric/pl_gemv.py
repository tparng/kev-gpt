"""Python backend that runs the INT4 GEMV on the KV260 PL fabric.

Drives the loadable engine (gemv_axi_load @ 0xA000_0000) over /dev/mem: streams a
layer's packed weights + INT8 activation in, runs, reads the INT32 result out.
Drop-in for goformer_full's pluggable `matmul`, so the whole model forward runs
with the matmuls on silicon:

    from model.goformer_full import GoformerFull, load_params
    from fabric.pl_gemv import PLGemv
    pl = PLGemv()                       # needs root (/dev/mem)
    gf = GoformerFull(load_params("fabric/export/goformer.npz"), matmul=pl.matmul)

Reload-per-call (simple, slow). A weight-resident fast path needs the x-only
pointer reset (CTRL bit3 / x_rst) — a follow-up bitstream.
"""

from __future__ import annotations

import mmap
import os
import struct

import numpy as np

from . import pack

IDCODE = 0x47454D4C  # "GEML"
# register byte offsets (gemv_axi_load)
CTRL, STAT, MCNT, KCNT, WDAT, XDAT, RADD, RDAT, CYC, IDC = \
    0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24


class PLGemv:
    def __init__(self, base=0xA0000000):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 0x1000, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        # 32-bit register view: numpy element assignment is ONE aligned 32-bit
        # access (a byte-wise mmap slice write decomposes into 4 AXI writes, which
        # for the streaming W_DATA/X_DATA port corrupts the load).
        self.reg = np.frombuffer(self.mm, dtype=np.uint32)
        got = self._rd(IDC)
        if got != IDCODE:
            raise RuntimeError(f"bad IDCODE 0x{got:08x} (bitstream loaded?)")
        self.last_cycles = 0
        self._wkey = None        # id() of the weight matrix currently resident

    def _wr(self, off, val):
        self.reg[off >> 2] = np.uint32(val & 0xFFFFFFFF)      # single 32-bit store

    def _rd(self, off):
        return int(self.reg[off >> 2])

    def matmul(self, int_w, int_x):
        """int_w (M,K) INT4 in [-8,7]; int_x (K,) INT8 -> y (M,) int32, on fabric.

        Weights are cached by object identity: ld_rst rewinds the load pointers but
        leaves BRAM intact, so re-running the same layer with a new activation only
        re-streams x (the T positions of one layer share one weight load)."""
        key = id(int_w)
        if key != self._wkey:
            iw = np.asarray(int_w, dtype=np.int8)
            self._M, self._K = iw.shape
            self._packed = pack.pack_int4_rows(iw).ravel().tolist()
        M, K = self._M, self._K
        self._wr(CTRL, 0x4)                                    # soft reset: clear stale done/busy/FSM
        self._wr(CTRL, 0x0)                                    #   (wmem/xmem persist -> cached weights survive)
        self._wr(MCNT, int(M))
        self._wr(KCNT, int(K))
        self._wr(CTRL, 0x2)                                    # ld_rst (ptrs->0, BRAM kept)
        if key != self._wkey:
            for b in self._packed:
                self._wr(WDAT, int(b))                         # stream weights (first time only)
            self._wkey = key
        for v in np.asarray(int_x, dtype=np.int64).tolist():
            self._wr(XDAT, int(v) & 0xFF)                      # always stream new activation
        self._wr(CTRL, 0x1)                                    # start
        spins = 0
        while not (self._rd(STAT) & 0x1):
            spins += 1
            if spins > 50_000_000:
                raise RuntimeError("PL GEMV timeout")
        self.last_cycles = self._rd(CYC)
        y = np.empty(M, dtype=np.int64)
        for m in range(M):
            self._wr(RADD, m)
            v = self._rd(RDAT)                                 # raw uint32
            y[m] = v - (1 << 32) if v >= (1 << 31) else v      # -> signed int32
        return y

    def close(self):
        self.mm.close()
        os.close(self.fd)


IDCODE_RESIDENT = 0x47454D52  # "GEMR"
WBASE = 0x28


class PLResident:
    """Resident-weights backend (gemv_axi_resident): the WHOLE model is streamed
    into URAM ONCE via preload(); per matmul we only set W_BASE + stream the
    activation. Eliminates per-token weight re-streaming (~83% of the old time)."""

    def __init__(self, base=0xA0000000):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 0x1000, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        self.reg = np.frombuffer(self.mm, dtype=np.uint32)
        got = self._rd(IDC)
        if got != IDCODE_RESIDENT:
            raise RuntimeError(f"bad IDCODE 0x{got:08x} (resident bitstream loaded?)")
        self._map = {}          # id(int_w) -> (w_base, M, K)
        self.total_bytes = 0
        self.last_cycles = 0

    def _wr(self, off, val):
        self.reg[off >> 2] = np.uint32(val & 0xFFFFFFFF)

    def _rd(self, off):
        return int(self.reg[off >> 2])

    def preload(self, params):
        """Stream every layer's weights into URAM once, recording byte offsets.
        Layers are gathered in a fixed order; offsets are cumulative packed bytes."""
        layers = []
        for blk in params["blocks"]:
            for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
                layers.append(blk[nm][0])
        layers.append(params["head"][0])

        self._wr(CTRL, 0x4); self._wr(CTRL, 0x0)        # soft reset
        self._wr(CTRL, 0x2)                              # ld_rst (load ptrs -> 0)
        off = 0
        for iw in layers:
            iw8 = np.asarray(iw, dtype=np.int8)
            M, K = iw8.shape
            packed = pack.pack_int4_rows(iw8).ravel()
            for b in packed.tolist():
                self._wr(WDAT, int(b))
            self._map[id(iw)] = (off, int(M), int(K))
            off += int(packed.size)
        self.total_bytes = off
        return off

    def matmul(self, int_w, int_x):
        wb, M, K = self._map[id(int_w)]                  # resident -> no weight streaming
        self._wr(CTRL, 0x4); self._wr(CTRL, 0x0)         # soft reset (clear done)
        self._wr(MCNT, M); self._wr(KCNT, K); self._wr(WBASE, wb)
        self._wr(CTRL, 0x2)                              # ld_rst (xptr->0; URAM kept)
        for v in np.asarray(int_x, dtype=np.int64).tolist():
            self._wr(XDAT, int(v) & 0xFF)
        self._wr(CTRL, 0x1)                              # start
        spins = 0
        while not (self._rd(STAT) & 0x1):
            spins += 1
            if spins > 50_000_000:
                raise RuntimeError("PL resident GEMV timeout")
        self.last_cycles = self._rd(CYC)
        y = np.empty(M, dtype=np.int64)
        for m in range(M):
            self._wr(RADD, m)
            v = self._rd(RDAT)
            y[m] = v - (1 << 32) if v >= (1 << 31) else v
        return y

    def close(self):
        self.mm.close()
        os.close(self.fd)


IDCODE_BANKED = 0x47454D42  # "GEMB"


class PLBankedResident:
    """Resident WIDE-GEMV backend (gemv_axi_banked / gemv_banked_resident): the
    WHOLE model's TRANSPOSED weights stream into URAM ONCE via preload(); per matmul
    we set W_BASE (wide-word offset) + stream the INT8 activation, and the core does
    LANES MACs/cycle. Drop-in for goformer_full's pluggable matmul, same as PLResident.

    Two differences from PLResident:
      * W_DATA carries a FULL 32-bit chunk of the LANES*4-bit wide word (low chunk
        first); the core assembles SUBW = LANES*4/32 chunks per word.
      * W_BASE / the offset map is in WIDE WORDS, not bytes.
    """

    def __init__(self, lanes=256, base=0xA0000000):
        from .stage3.pack_banked_resident import pack_transposed, words_per_layer
        self._pack_transposed = pack_transposed
        self._words_per_layer = words_per_layer
        self.lanes = lanes
        self.subw = (lanes * 4) // 32                    # 32-bit chunks per wide word
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 0x1000, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        self.reg = np.frombuffer(self.mm, dtype=np.uint32)
        got = self._rd(IDC)
        if got != IDCODE_BANKED:
            raise RuntimeError(f"bad IDCODE 0x{got:08x} (banked bitstream loaded?)")
        self._map = {}          # id(int_w) -> (w_base_words, M, K)
        self.total_words = 0
        self.last_cycles = 0

    def _wr(self, off, val):
        self.reg[off >> 2] = np.uint32(val & 0xFFFFFFFF)

    def _rd(self, off):
        return int(self.reg[off >> 2])

    def _word_chunks(self, word):
        """Split one LANES*4-bit wide word into self.subw 32-bit chunks, low first."""
        for s in range(self.subw):
            yield (word >> (s * 32)) & 0xFFFFFFFF

    def preload(self, params):
        """Stream every layer's TRANSPOSED wide words into URAM once (as 32-bit
        chunks), recording per-layer wide-word offsets. Fixed model order matches
        PLResident.preload so w_base bookkeeping is identical in spirit."""
        layers = []
        for blk in params["blocks"]:
            for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
                layers.append(blk[nm][0])
        layers.append(params["head"][0])

        self._wr(CTRL, 0x4); self._wr(CTRL, 0x0)        # soft reset
        self._wr(CTRL, 0x2)                              # ld_rst (load ptrs -> 0)
        wbase = 0
        for iw in layers:
            iw8 = np.asarray(iw, dtype=np.int8)
            M, K = int(iw8.shape[0]), int(iw8.shape[1])
            words = self._pack_transposed(iw8, self.lanes)
            for w in words:
                for chunk in self._word_chunks(w):
                    self._wr(WDAT, int(chunk))
            self._map[id(iw)] = (wbase, M, K)
            wbase += len(words)
        self.total_words = wbase
        return wbase

    def matmul(self, int_w, int_x):
        wb, M, K = self._map[id(int_w)]                  # resident -> no weight streaming
        self._wr(CTRL, 0x4); self._wr(CTRL, 0x0)         # soft reset (clear done)
        self._wr(MCNT, M); self._wr(KCNT, K); self._wr(WBASE, wb)
        self._wr(CTRL, 0x2)                              # ld_rst (xptr->0; URAM kept)
        for v in np.asarray(int_x, dtype=np.int64).tolist():
            self._wr(XDAT, int(v) & 0xFF)
        self._wr(CTRL, 0x1)                              # start
        spins = 0
        while not (self._rd(STAT) & 0x1):
            spins += 1
            if spins > 50_000_000:
                raise RuntimeError("PL banked GEMV timeout")
        self.last_cycles = self._rd(CYC)
        y = np.empty(M, dtype=np.int64)
        for m in range(M):
            self._wr(RADD, m)
            v = self._rd(RDAT)
            y[m] = v - (1 << 32) if v >= (1 << 31) else v
        return y

    def close(self):
        self.mm.close()
        os.close(self.fd)
