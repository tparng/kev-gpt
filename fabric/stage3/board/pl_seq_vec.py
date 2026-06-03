"""pl_seq_vec — drive the P-WIDE datapath sequencer bitstream (gemv_axi_seq_vec) on the Kria
and measure single-token cyc/token + tok/s, bit-honest.

The vector engine runs the WHOLE forward (4 blocks + LN_f + head + argmax) on-chip. Host
streams the INT4 weight image once (packed at LANES), sets TOK_ID/POS, pulses GO, polls DONE,
reads TOK_OUT + CYCLES. tok/s = FCLK / CYCLES (single token). The number is reported ONLY if
TOK_OUT == seq_ref.full_forward_signals(tok)['tok'] (the bit-exact gate).

Register map (gemv_axi_seq_vec.v):
  0x00 CTRL(b0 go,b1 wl_rst,b2 soft_reset)  0x04 STATUS(b0 done,b1 busy)
  0x08 TOK_ID  0x0C POS  0x10 W_DATA  0x14 RD_SEL  0x18 RD_ADDR
  0x1C RD_DATA_LO  0x20 RD_DATA_HI  0x24 TOK_OUT  0x28 CYCLES  0x2C IDCODE(0x53515256 "SQRV")

  sudo ~/kevweb/venv/bin/python -m fabric.stage3.board.pl_seq_vec --lanes 128 --tok 48 --fclk 40e6
"""

from __future__ import annotations

import argparse
import mmap
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from fabric.stage3 import pack_banked, seq_ref  # noqa: E402
from fabric.stage3.board.pl_seq_chat import set_and_verify_fclk  # noqa: E402

NLAYER = 4
IDCODE_SQRV = 0x53515256
BASE = 0xA0000000
R_CTRL, R_STATUS = 0x00, 0x04
R_TOKID, R_POS, R_WDATA = 0x08, 0x0C, 0x10
R_RDSEL, R_RDADDR, R_RDLO, R_RDHI = 0x14, 0x18, 0x1C, 0x20
R_TOKOUT, R_CYCLES, R_IDCODE = 0x24, 0x28, 0x2C


class Dev:
    def __init__(self, base=BASE):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 0x1000, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        self.reg = np.frombuffer(self.mm, dtype=np.uint32)

    def wr(self, off, val): self.reg[off >> 2] = np.uint32(val & 0xFFFFFFFF)
    def rd(self, off): return int(self.reg[off >> 2])

    def close(self):
        self.reg = None; self.mm.close(); os.close(self.fd)


def build_weight_image(intseq, lanes):
    """Full transposed INT4 image at `lanes`: per block (qkv|proj|mlp_fc|mlp_proj) for NLAYER
    blocks, then the head — the exact order sequencer_vec's w_base offsets assume."""
    p = intseq.p
    words = []
    for bi in range(NLAYER):
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            words += pack_banked.pack_transposed(np.asarray(p["blocks"][bi][nm][0], np.int8), lanes)
    words += pack_banked.pack_transposed(np.asarray(p["head"][0], np.int8), lanes)
    return words


def main(argv=None):
    ap = argparse.ArgumentParser(description="P-wide vector sequencer on silicon")
    ap.add_argument("--npz", default=os.path.join(_REPO, "fabric", "export", "goformer.npz"))
    ap.add_argument("--lanes", type=int, default=128)
    ap.add_argument("--tok", type=int, default=48)
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--fclk", type=float, default=40e6)
    ap.add_argument("--base", type=lambda s: int(s, 0), default=BASE)
    ap.add_argument("--poll-timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    fclk_hz = set_and_verify_fclk(args.fclk)
    p, cfg = seq_ref.build(args.npz)
    intseq = seq_ref.IntSequencer(p, cfg)
    gold = intseq.full_forward_signals(int(args.tok))
    gold_tok = int(gold["tok"])
    lanes = args.lanes
    subw = (lanes * 4) // 32
    words = build_weight_image(intseq, lanes)
    print(f"[ref ] tok={args.tok} -> gold token {gold_tok}")
    print(f"[wimg] {len(words)} x {lanes*4}-bit words, SUBW={subw}")

    dev = Dev(args.base)
    try:
        idc = dev.rd(R_IDCODE)
        print(f"[idc ] 0x{idc:08X} {'OK' if idc == IDCODE_SQRV else 'MISMATCH (expect SQRV)'}")
        if idc != IDCODE_SQRV:
            return 3
        dev.wr(R_CTRL, 0x4); dev.wr(R_CTRL, 0x0)              # soft_reset
        dev.wr(R_CTRL, 0x2)                                  # wl_rst
        t0 = time.time()
        for w in words:
            w = int(w)
            for s in range(subw):
                dev.wr(R_WDATA, (w >> (32 * s)) & 0xFFFFFFFF)
        print(f"[load] {len(words)*subw} chunks in {time.time()-t0:.2f}s")
        dev.wr(R_TOKID, int(args.tok) & 0x1FF)
        dev.wr(R_POS, int(args.pos) & 0x1FF)
        dev.wr(R_CTRL, 0x1)                                  # go
        t0 = time.time(); done = False
        while time.time() - t0 < args.poll_timeout:
            if dev.rd(R_STATUS) & 0x1:
                done = True; break
        if not done:
            print(f"[run ] TIMEOUT STATUS=0x{dev.rd(R_STATUS):08X}"); return 4
        cyc = dev.rd(R_CYCLES); got_tok = dev.rd(R_TOKOUT) & 0x1FF
        match = (got_tok == gold_tok)
        print(f"[hw  ] tok_out={got_tok} gold={gold_tok} match={match}  CYCLES={cyc}")
        if match and cyc > 0:
            tok_s = fclk_hz / cyc
            print(f"[meas] cyc/token = {cyc}   MEASURED tok/s = {tok_s:.1f}  "
                  f"(= {fclk_hz:.0f}/{cyc} @ {fclk_hz/1e6:.1f} MHz)")
            print(f"PL_SEQVEC_VERDICT match=True cyc={cyc} tok_s={tok_s:.1f} "
                  f"fclk={fclk_hz:.0f} idcode=0x{idc:08X}")
        else:
            print(f"PL_SEQVEC_VERDICT match={match} cyc={cyc} (tok/s WITHHELD unless match) "
                  f"idcode=0x{idc:08X}")
        return 0 if match else 1
    finally:
        dev.close()


if __name__ == "__main__":
    raise SystemExit(main())
