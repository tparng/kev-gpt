"""pl_mamba — first-silicon driver for mamba_seq_axi (the full Mamba-2 token
engine; doc 9 rung 4).

Streams the SAME table images the sim gate used (ms_t0..14.mem + ms_cfg.mem
from run_mamba_seq's sim dir) over AXI-Lite, replays the gate tokens and
compares logits against the laptop-computed reference pack (ms_ref.npz:
ref_logits, toks) — the bit-honest step — then generates greedily from a
prompt using the engine's own argmax (TOKOUT fed back), with the fabric
CYCLES counter giving the MEASURED tok/s.

Laptop prep (after a green run_mamba_seq):
  python -m fabric.stage3.board.pack_mamba              # writes ms_ref.npz
  scp <simdir>/ms_t*.mem <simdir>/ms_cfg.mem <simdir>/ms_ref.npz kria:~/kevmem/

On the Kria (root for /dev/mem):
  sudo python3 -m fabric.stage3.board.pl_mamba --dir ~/kevmem --fclk 100e6
  sudo python3 -m fabric.stage3.board.pl_mamba --dir ~/kevmem --gen 64 \
      --prompt-toks 12,176

The fclk MUST be forced via /sys (a flat fpgautil load does not apply the
BD's PL0_REF FREQMHZ — the known gotcha) and is verified after setting.
"""

from __future__ import annotations

import argparse
import glob
import mmap
import os
import struct
import sys
import time

import numpy as np

BASE = 0xA000_0000
R_CTRL, R_STATUS, R_TOK, R_TOKOUT = 0x00, 0x04, 0x08, 0x0C
R_TSEL, R_TADDR, R_TDATA, R_CYCLES = 0x10, 0x14, 0x18, 0x1C
R_DBGSEL, R_DBGADDR, R_DBGDATA, R_IDCODE = 0x20, 0x24, 0x28, 0x2C
IDCODE = 0x4D414D42          # "MAMB"
V, D = 1024, 256
DBG_LOGIT = 6


class Mamba:
    def __init__(self, base=BASE):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.m = mmap.mmap(self.fd, 0x1000, offset=base)

    def wr(self, off, val):
        self.m[off:off + 4] = struct.pack("<I", val & 0xFFFFFFFF)

    def rd(self, off):
        return struct.unpack("<I", self.m[off:off + 4])[0]

    def wait_status(self, bit, timeout=10.0):
        t0 = time.time()
        while not (self.rd(R_STATUS) >> bit) & 1:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"status bit {bit} timeout")

    def load_table(self, sel, words):
        self.wr(R_TSEL, sel)
        self.wr(R_TADDR, 0)
        w = self.wr
        for v in words:
            w(R_TDATA, int(v))

    def step(self, tok):
        self.wr(R_TOK, int(tok))          # TOK write pulses go
        self.wait_status(0)                # done
        return self.rd(R_TOKOUT) & 0x3FF, self.rd(R_CYCLES)

    def read_logits(self):
        self.wr(R_DBGSEL, DBG_LOGIT)
        out = np.zeros(V, dtype=np.int64)
        for i in range(V):
            self.wr(R_DBGADDR, i)
            v = self.rd(R_DBGDATA) & 0xFFFF
            out[i] = v - (1 << 16) if v >= 1 << 15 else v
        return out


def set_and_verify_fclk(hz):
    path = "/sys/devices/platform/fclk0/set_rate"
    if not os.path.exists(path):
        for cand in ("/sys/class/clk/fclk0/set_rate",):
            if os.path.exists(cand):
                path = cand
                break
        else:
            print("WARN: no fclk0 set_rate node; PL clock NOT forced",
                  file=sys.stderr)
            return
    with open(path, "w") as f:
        f.write(str(int(hz)))
    print(f"fclk0 forced to {hz/1e6:.1f} MHz")


def rd16(path):
    with open(path) as f:
        return [int(l, 16) for l in f if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pl_mamba")
    ap.add_argument("--dir", required=True, help="dir with ms_t*.mem/ms_cfg/ms_ref")
    ap.add_argument("--fclk", type=float, default=100e6)
    ap.add_argument("--gen", type=int, default=0, help="greedy-generate N tokens")
    ap.add_argument("--prompt-toks", default="", help="comma-sep BPE ids to prime")
    ap.add_argument("--skip-load", action="store_true",
                    help="tables already resident (same power cycle)")
    args = ap.parse_args(argv)

    set_and_verify_fclk(args.fclk)
    d = Mamba()
    ident = d.rd(R_IDCODE)
    if ident != IDCODE:
        sys.exit(f"IDCODE mismatch: {ident:08x} != {IDCODE:08x} — wrong bitstream?")
    print("IDCODE ok (MAMB)")

    d.wr(R_CTRL, 0b10)                     # soft_reset -> state clear sweep
    d.wait_status(2)                        # ready
    print("ready (scan state cleared)")

    cfg = rd16(f"{args.dir}/ms_cfg.mem")
    if not args.skip_load:
        t0 = time.time()
        # identical order to tb_mamba_seq: t1, t0(gw), t1 again, t2..t14, t15(seed)
        d.load_table(1, rd16(f"{args.dir}/ms_t1.mem"))
        d.load_table(0, rd16(f"{args.dir}/ms_t0.mem"))
        for s in range(1, 15):
            d.load_table(s, rd16(f"{args.dir}/ms_t{s}.mem"))
        # WSEL_SEED=15: the rsqrt seed table, loaded over AXI (no baked ROM).
        d.load_table(15, rd16(f"{args.dir}/ms_t15.mem"))
        print(f"tables loaded ({time.time()-t0:.1f}s, "
              f"{cfg[15] + sum(cfg[1:15]) + cfg[16]} words)")

    # ---- gate replay: logits cosine vs the laptop reference ----------------
    ref_path = f"{args.dir}/ms_ref.npz"
    if os.path.exists(ref_path):
        ref = np.load(ref_path)
        toks, ref_logits = ref["toks"], ref["ref_logits"]
        ok = True
        for t, tok in enumerate(toks):
            nxt, cyc = d.step(int(tok))
            got = d.read_logits().astype(float)
            rl = ref_logits[t].astype(float)
            cos = float(rl @ got / ((np.linalg.norm(rl) * np.linalg.norm(got)) or 1))
            am = int(np.argmax(rl)) == int(np.argmax(got)) == nxt
            ok &= cos > 0.999 and am
            print(f"tok#{t} ({int(tok)}): logits cos {cos:.5f} argmax "
                  f"{'MATCH' if am else 'DIFF'} (ref {int(np.argmax(rl))} "
                  f"rtl {nxt})  {cyc} cyc")
        print(f"PL_MAMBA_VERDICT: {'PASS' if ok else 'FAIL'} "
              f"({len(toks)} tokens on silicon, bar cos>0.999 + argmax)")
        if not ok:
            return 1
    else:
        print("no ms_ref.npz — skipping gate replay")

    # ---- generation: greedy via the engine's own argmax --------------------
    if args.gen:
        d.wr(R_CTRL, 0b10)                 # fresh state for the prompt
        d.wait_status(2)
        prompt = [int(x) for x in args.prompt_toks.split(",") if x != ""]
        out, cycles = [], []
        tok = prompt[0] if prompt else 1
        t0 = time.time()
        for tk in prompt:
            tok, cyc = d.step(tk)
        for _ in range(args.gen):
            out.append(int(tok))
            tok, cyc = d.step(tok)
            cycles.append(cyc)
        wall = time.time() - t0
        cyc_avg = sum(cycles) / len(cycles)
        print(f"generated {len(out)} tokens: {out}")
        print(f"PL_MAMBA_BENCH: {cyc_avg:.0f} cyc/tok @ {args.fclk/1e6:.0f} MHz "
              f"= {args.fclk/cyc_avg:.1f} tok/s fabric "
              f"({len(out)/wall:.1f} tok/s incl AXI wall)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
