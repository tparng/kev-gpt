"""pl_ssm — first-silicon driver for ssm_scan_axi (doc 9 §7 rung 4).

Loads T steps of the same stimulus run_ssm_scan uses (same seed => same
vectors), drives the core step-by-step over AXI-Lite, and compares y
bit-exactly against model/mamba2_fixed_scan.py. Then the bench: reload,
CTRL.go_loop with STEPS=N for the silicon cycles/step number.

On the Kria (run as root for /dev/mem):
  sudo python3 -m fabric.stage3.board.pl_ssm --steps 64 --bench 4096 --fclk 100e6

The fclk MUST be forced via /sys (a flat fpgautil load does not apply the
BD's PL0_REF FREQMHZ — the known gotcha) and is verified after setting.
"""

from __future__ import annotations

import argparse
import mmap
import os
import struct
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from model.mamba2_fixed_scan import (Q_ACT, Q_STATE, a_from_dtA, quant,  # noqa: E402
                                     scan_step_fixed)

BASE = 0xA000_0000
R_CTRL, R_STATUS, R_AQ = 0x00, 0x04, 0x08
R_DTX_ADDR, R_DTX_DATA = 0x0C, 0x10
R_B_ADDR, R_B_DATA = 0x14, 0x18
R_C_ADDR, R_C_DATA = 0x1C, 0x20
R_STEPS, R_Y_ADDR, R_Y_DATA = 0x24, 0x28, 0x2C
R_CYCLES, R_IDCODE = 0x30, 0x34
IDCODE = 0x53534D53
P = N = 64


class Ssm:
    def __init__(self, base=BASE):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.m = mmap.mmap(self.fd, 0x1000, offset=base)

    def wr(self, off, val):
        self.m[off:off + 4] = struct.pack("<I", val & 0xFFFFFFFF)

    def rd(self, off):
        return struct.unpack("<I", self.m[off:off + 4])[0]

    def wait_status(self, bit, timeout=5.0):
        t0 = time.time()
        while not (self.rd(R_STATUS) >> bit) & 1:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"status bit {bit} timeout")

    def load_vec(self, addr_reg, data_reg, vals, mask):
        for i, v in enumerate(vals):
            self.wr(addr_reg, i)
            self.wr(data_reg, int(v) & mask)

    def read_y(self):
        out = np.zeros(P, dtype=np.int64)
        for i in range(P):
            self.wr(R_Y_ADDR, i)
            v = self.rd(R_Y_DATA)
            out[i] = v - (1 << 32) if v >= 1 << 31 else v
        return out


def set_and_verify_fclk(hz):
    path = "/sys/devices/platform/fclk0/set_rate"
    if not os.path.exists(path):
        for cand in ("/sys/class/clk/fclk0/set_rate",):
            if os.path.exists(cand):
                path = cand
                break
        else:
            print("WARN: no fclk0 set_rate node; PL clock NOT forced", file=sys.stderr)
            return
    with open(path, "w") as f:
        f.write(str(int(hz)))
    print(f"fclk0 forced to {hz/1e6:.1f} MHz")


def stimulus(T, seed):
    rng = np.random.default_rng(seed)
    negA = rng.uniform(1.0, 16.0)
    dts = np.exp(rng.uniform(np.log(1e-3), np.log(0.1), T))
    xs = np.clip(np.round(rng.normal(0, 38, (T, P))), -127, 127).astype(np.int64)
    Bs = np.clip(np.round(rng.normal(0, 38, (T, N))), -127, 127).astype(np.int64)
    Cs = np.clip(np.round(rng.normal(0, 38, (T, N))), -127, 127).astype(np.int64)
    return negA, dts, xs, Bs, Cs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pl_ssm")
    ap.add_argument("--steps", type=int, default=64, help="bit-exact check steps")
    ap.add_argument("--bench", type=int, default=4096, help="loop-mode steps for cyc/step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fclk", type=float, default=100e6)
    args = ap.parse_args(argv)

    set_and_verify_fclk(args.fclk)
    d = Ssm()
    ident = d.rd(R_IDCODE)
    if ident != IDCODE:
        sys.exit(f"IDCODE mismatch: {ident:08x} != {IDCODE:08x} — wrong bitstream?")
    print(f"IDCODE ok (SSMS)")

    # --- bit-exact ladder: step-by-step vs the Python reference ---
    d.wr(R_CTRL, 0b010)               # soft_reset -> clear sweep
    d.wait_status(2)                   # ready
    negA, dts, xs, Bs, Cs = stimulus(args.steps, args.seed)
    h = np.zeros((P, N), dtype=np.int64)
    mism = 0
    for t in range(args.steps):
        a_q = int(a_from_dtA(dts[t] * negA))
        dtx = quant(dts[t] * xs[t] / (1 << Q_ACT), Q_STATE)
        d.load_vec(R_DTX_ADDR, R_DTX_DATA, dtx, 0xFFFF)
        d.load_vec(R_B_ADDR, R_B_DATA, Bs[t], 0xFF)
        d.load_vec(R_C_ADDR, R_C_DATA, Cs[t], 0xFF)
        d.wr(R_AQ, a_q)
        d.wr(R_CTRL, 0b001)           # go
        d.wait_status(0)               # done
        got = d.read_y()
        got = np.where(got >= 1 << 15, got - (1 << 16), got)
        ref = scan_step_fixed(h, xs[t], Bs[t], Cs[t], float(dts[t]), negA)
        step_mism = int((got != ref).sum())
        mism += step_mism
        if step_mism and mism <= 3:
            i = int(np.argwhere(got != ref)[0])
            print(f"  step {t} p{i}: got {got[i]} ref {ref[i]}")
    verdict = "BIT-EXACT" if mism == 0 else "FAIL"
    print(f"PL_SSM_VERDICT: {verdict} ({args.steps} steps on silicon, {mism} mismatches)")

    # --- bench: back-to-back steps, cycles from the fabric counter ---
    if args.bench and mism == 0:
        d.wr(R_STEPS, args.bench)
        t0 = time.time()
        d.wr(R_CTRL, 0b100)           # go_loop
        d.wait_status(0, timeout=60)
        wall = time.time() - t0
        cyc = d.rd(R_CYCLES)
        per = cyc / args.bench
        print(f"PL_SSM_BENCH: {args.bench} steps, {cyc} cycles "
              f"({per:.1f} cyc/step @ {args.fclk/1e6:.0f} MHz "
              f"= {args.fclk/per:.0f} head-steps/s; wall {wall:.2f}s)")


if __name__ == "__main__":
    main()
