"""KV256CDriver — ctypes wrapper over libkv256_drv.so (kv256_drv.c).

The compiled hot loop for the resident kv256 SQRV engine: one C call runs the
whole prompt-prime + feedback GO train, replacing PLKV256Device._go's per-token
Python /dev/mem pokes (~5-6 x 7.75us each). Same register writes, same order ->
BIT-EXACT token stream (verify on the board with kv256_ab_test.py).

Usage (on the Kria, resident kv256 bitstream loaded, as root):
    from fabric.stage3.board.pl_kv256_c import KV256CDriver
    drv = KV256CDriver()                       # mmap + IDCODE "SQRV" check
    toks, cyc = drv.run(ids, gen=48, seed=0)   # seed 0 => greedy; !=0 => on-chip sample
    drv.close()

This is a PREPARED optimization: it needs the .so built on-board and an A/B pass
before it's wired into the daemon. It does NOT replace pl_kv256.py; it is an
optional fast path whose reference/fallback is the existing Python _go.
Streaming note: run() is one blocking call over the whole reply (max fabric
win, batch path). Per-char websocket streaming needs the token-ring extension
(a follow-up C entry point) — see kv256_drv.c header.
"""

from __future__ import annotations

import ctypes
import os

IDCODE_SQRV = 0x53515256  # "SQRV"
_SO_NAME = "libkv256_drv.so"


def _find_so():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, _SO_NAME), os.path.join(os.getcwd(), _SO_NAME), _SO_NAME):
        if os.path.exists(cand):
            return cand
    return os.path.join(here, _SO_NAME)   # let ctypes raise a clear error


class KV256CDriver:
    def __init__(self, base=0xA0000000, so_path=None):
        self.base = base
        self.lib = ctypes.CDLL(so_path or _find_so())
        self.lib.kv256_open.argtypes = [ctypes.c_ulong]
        self.lib.kv256_open.restype = ctypes.c_int
        self.lib.kv256_close.argtypes = []
        self.lib.kv256_idcode.restype = ctypes.c_uint
        self.lib.kv256_run.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
            ctypes.c_uint, ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.kv256_run.restype = ctypes.c_long

        rc = self.lib.kv256_open(ctypes.c_ulong(base))
        if rc != 0:
            raise RuntimeError(
                f"kv256_open failed rc={rc} "
                "(-1 /dev/mem open, -2 mmap, -3 bad IDCODE/no bitstream). Run as root "
                "with the resident kv256 (SQRV) bitstream loaded.")
        self.last_cycles = 0

    def idcode(self) -> int:
        return int(self.lib.kv256_idcode())

    def run(self, ids, gen: int, seed: int = 0):
        """Prime `ids` from pos 0, generate `gen` tokens. Returns (token_ids, cycles).
        token_ids[0] is the last-prompt TOK_OUT (matches _stream_gen's 1st emit)."""
        ids = [int(t) & 0x1FF for t in ids]
        if not ids or gen < 1:
            raise ValueError("need >=1 prompt id and gen >= 1")
        L = len(ids)
        c_ids = (ctypes.c_int * L)(*ids)
        out = (ctypes.c_int * gen)()
        cyc = self.lib.kv256_run(c_ids, L, int(gen), ctypes.c_uint(seed & 0xFFFFFFFF), out)
        if cyc < 0:
            raise RuntimeError("kv256_run hang/timeout (STATUS.done never asserted)")
        self.last_cycles = int(cyc)
        return list(out), int(cyc)

    def close(self):
        try:
            self.lib.kv256_close()
        except Exception:
            pass
