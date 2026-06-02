"""PLResidentC — resident PL GEMV backend whose hot loop is the compiled C driver.

Same contract as fabric.pl_gemv.PLResident (preload the whole model into URAM once,
then matmul(int_w,int_x) keyed by id(int_w)), but the per-call activation stream +
result readback go through libgemv_axi_drv.so (gemv_axi_drv.c) instead of per-register
Python /dev/mem pokes. Weight STREAMING at preload still uses the Python W_DATA path
(it happens once; not worth a second C entry point). The win is on the per-token hot
path: ~7.75us/register Python -> a single ctypes call doing all the volatile stores.

Drop-in for GoformerFull/KVDecoder `matmul`. Needs root (/dev/mem) and the resident
bitstream loaded (IDCODE "GEMR"). Build the .so first — see run_on_board.sh.
"""

from __future__ import annotations

import ctypes
import mmap
import os

import numpy as np

from fabric import pack

IDCODE_RESIDENT = 0x47454D52  # "GEMR"
# register byte offsets (gemv_axi_resident) — used only for the one-time W_DATA load
CTRL, STAT, MCNT, KCNT, WDAT, XDAT, RADD, RDAT, CYC, IDC, WBASE = \
    0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24, 0x28

_SO_NAME = "libgemv_axi_drv.so"


def _find_so():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, _SO_NAME), os.path.join(os.getcwd(), _SO_NAME), _SO_NAME):
        if os.path.exists(cand):
            return cand
    return os.path.join(here, _SO_NAME)        # let ctypes raise a clear error


class PLResidentC:
    def __init__(self, base=0xA0000000, so_path=None):
        self.base = base
        self.lib = ctypes.CDLL(so_path or _find_so())
        self.lib.gemv_open.argtypes = [ctypes.c_ulong]
        self.lib.gemv_open.restype = ctypes.c_int
        self.lib.gemv_close.argtypes = []
        self.lib.gemv_idcode.restype = ctypes.c_uint
        self.lib.gemv_run.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_uint,
            ctypes.POINTER(ctypes.c_byte), ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.gemv_run.restype = ctypes.c_long

        rc = self.lib.gemv_open(ctypes.c_ulong(base))
        if rc != 0:
            raise RuntimeError(
                f"gemv_open failed rc={rc} "
                "(-1 /dev/mem open, -2 mmap, -3 bad IDCODE/no bitstream). Run as root.")

        # one-time weight load still uses a Python mmap on the same window. The C
        # driver mmaps its own copy; both point at the same physical AXI slave, so
        # writes from either are visible — we only use this for the W_DATA stream.
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 0x1000, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        self.reg = np.frombuffer(self.mm, dtype=np.uint32)

        self._map = {}          # id(int_w) -> (w_base, M, K)
        self.total_bytes = 0
        self.last_cycles = 0

    def _wr(self, off, val):
        self.reg[off >> 2] = np.uint32(val & 0xFFFFFFFF)   # single 32-bit store

    def preload(self, params):
        """Stream every layer's weights into URAM once (Python W_DATA path),
        recording cumulative packed-byte offsets. Identical layer order + packing
        to PLResident, so URAM contents and w_base values match exactly."""
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
        x = np.asarray(int_x, dtype=np.int8)             # INT8 activation, K elements
        if x.shape[0] != K:
            raise ValueError(f"activation length {x.shape[0]} != K {K}")
        y = np.empty(M, dtype=np.int32)
        xp = x.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
        yp = y.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        cyc = self.lib.gemv_run(int(M), int(K), int(wb), xp, yp)
        if cyc < 0:
            raise RuntimeError("PL resident GEMV timeout (C driver)")
        self.last_cycles = int(cyc)
        return y.astype(np.int64)

    def close(self):
        try:
            self.mm.close(); os.close(self.fd)
        finally:
            self.lib.gemv_close()
