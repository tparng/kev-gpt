"""pl_seq_sb — drive the SPLIT-BRAIN sequencer bitstream (gemv_axi_seq_sb, IDCODE
"SQSB") on the Kria and measure cyc/N-tokens + aggregate tok/s, bit-honest.

Two independent cohorts of NC streams share only the TDP URAM weight image + the
arbitrated LN/attn/embed/dq units. The register map is the SQ16 map verbatim
(same loader, TOK_ID/TOK_OUT slots); N may be 14 (NC=7, the 97%-LUT fit build)
or 16 — slots >= N read 0 and their TOK_ID writes are ignored by the core.
Aggregate tok/s = N * FCLK / CYCLES, reported ONLY if ALL N tok_outs match
seq_ref.full_forward_signals per stream (the bit-exact gate).

  sudo ~/kevweb/venv/bin/python -m fabric.stage3.board.pl_seq_sb \
      --n 14 --lanes 128 --fclk 166.7e6
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
IDCODE_SQSB = 0x53515342
BASE = 0xA0000000
R_CTRL, R_STATUS = 0x00, 0x04
R_POS, R_WDATA = 0x0C, 0x10
R_CYCLES, R_IDCODE = 0x28, 0x2C
R_TOKID = [0x08, 0x30, 0x34, 0x38, 0x50, 0x54, 0x58, 0x5C,
           0x80, 0x84, 0x88, 0x8C, 0x90, 0x94, 0x98, 0x9C]
R_TOKOUT = [0x24, 0x3C, 0x40, 0x44, 0x60, 0x64, 0x68, 0x6C,
            0xA0, 0xA4, 0xA8, 0xAC, 0xB0, 0xB4, 0xB8, 0xBC]
R_EDATA = 0x4C
TMAX = 32
RESID_FRAC = 25
TOKS16 = "48,10,100,77,5,60,120,180,33,7,150,90,14,66,111,173"


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
    p = intseq.p
    words = []
    for bi in range(NLAYER):
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            words += pack_banked.pack_transposed(np.asarray(p["blocks"][bi][nm][0], np.int8), lanes)
    words += pack_banked.pack_transposed(np.asarray(p["head"][0], np.int8), lanes)
    return words


def build_embed_chunks(intseq):
    """tok_emb then pos_emb[:TMAX] as Q6.25 INT32 chunks, element order matching the
    P-wide rows (the el loader assembles P*32-bit words from consecutive chunks)."""
    p = intseq.p
    tok = np.round(np.asarray(p["tok_emb"], np.float64) * (1 << RESID_FRAC)).astype(np.int64)
    pos = np.round(np.asarray(p["pos_emb"], np.float64)[:TMAX] * (1 << RESID_FRAC)).astype(np.int64)
    vals = np.concatenate([tok.reshape(-1), pos.reshape(-1)])
    return [int(v) & 0xFFFFFFFF for v in vals]


def main(argv=None):
    ap = argparse.ArgumentParser(description="split-brain sequencer (SQSB) on silicon")
    ap.add_argument("--npz", default=os.path.join(_REPO, "fabric", "export", "goformer.npz"))
    ap.add_argument("--lanes", type=int, default=128)
    ap.add_argument("--n", type=int, default=14, help="total streams (14 = NC7 build)")
    ap.add_argument("--toks", default=None, help="comma list; default = first N of the SQ16 set")
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--fclk", type=float, default=125e6)
    ap.add_argument("--base", type=lambda s: int(s, 0), default=BASE)
    ap.add_argument("--poll-timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    n = int(args.n)
    toks = [int(t) for t in (args.toks or ",".join(TOKS16.split(",")[:n])).split(",")]
    assert len(toks) == n, f"need {n} tokens"

    fclk_hz = set_and_verify_fclk(args.fclk)
    p, cfg = seq_ref.build(args.npz)
    intseq = seq_ref.IntSequencer(p, cfg)
    golds = []
    for t in toks:
        golds.append(int(intseq.full_forward_signals(t)["tok"]))
        intseq.reset()
    lanes = args.lanes
    subw = (lanes * 4) // 32
    words = build_weight_image(intseq, lanes)
    print(f"[ref ] toks={toks} -> gold tokens {golds}")
    print(f"[wimg] {len(words)} x {lanes*4}-bit words, SUBW={subw}")

    dev = Dev(args.base)
    try:
        idc = dev.rd(R_IDCODE)
        print(f"[idc ] 0x{idc:08X} {'OK' if idc == IDCODE_SQSB else 'MISMATCH (expect SQSB)'}")
        if idc != IDCODE_SQSB:
            return 3
        dev.wr(R_CTRL, 0x4); dev.wr(R_CTRL, 0x0)              # soft_reset
        dev.wr(R_CTRL, 0x2)                                  # wl_rst
        t0 = time.time()
        for w in words:
            w = int(w)
            for s in range(subw):
                dev.wr(R_WDATA, (w >> (32 * s)) & 0xFFFFFFFF)
        print(f"[load] {len(words)*subw} chunks in {time.time()-t0:.2f}s")
        emb = build_embed_chunks(intseq)
        t0 = time.time()
        for v in emb:
            dev.wr(R_EDATA, v)
        print(f"[load] embed {len(emb)} chunks in {time.time()-t0:.2f}s")
        for b, t in enumerate(toks):
            dev.wr(R_TOKID[b], t & 0x1FF)
        dev.wr(R_POS, int(args.pos) & 0x1FF)
        dev.wr(R_CTRL, 0x1)                                   # go
        t0 = time.time(); done = False
        while time.time() - t0 < args.poll_timeout:
            if dev.rd(R_STATUS) & 0x1:
                done = True; break
        if not done:
            print(f"[run ] TIMEOUT STATUS=0x{dev.rd(R_STATUS):08X}"); return 4
        cyc = dev.rd(R_CYCLES)
        outs = [dev.rd(R_TOKOUT[b]) & 0x1FF for b in range(n)]
        match = (outs == golds)
        print(f"[hw  ] tok_outs={outs} gold={golds} match={match}  CYCLES={cyc}")
        if match and cyc > 0:
            agg = n * fclk_hz / cyc
            print(f"[meas] cyc/{n}tok = {cyc}  cyc/token = {cyc/n:.0f}  "
                  f"AGGREGATE tok/s = {agg:.1f}  (= {n}*{fclk_hz:.0f}/{cyc} @ {fclk_hz/1e6:.1f} MHz)")
            print(f"PL_SEQSB_VERDICT match=True n={n} cyc={cyc} agg_tok_s={agg:.1f} "
                  f"fclk={fclk_hz:.0f} idcode=0x{idc:08X}")
        else:
            print(f"PL_SEQSB_VERDICT match={match} n={n} cyc={cyc} (tok/s WITHHELD unless match) "
                  f"idcode=0x{idc:08X}")
        return 0 if match else 1
    finally:
        dev.close()


if __name__ == "__main__":
    raise SystemExit(main())
