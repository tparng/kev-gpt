"""pl_seq_kv — drive the KV-FAITHFUL single-stream sequencer bitstream (gemv_axi_seq_vec,
IDCODE "SQRV") on the Kria through a full multi-token decode and measure avg cyc/token
+ tok/s, bit-honest.

The doc-7 R1 sequencer_vec keeps the K8/V8 kv_bank ALIVE between GO pulses, so one
bitstream decodes a whole prompt + greedy generation position by position: stream the
INT4 weight image once (tok/pos embed ROMs are $readmemh'd at build time — nothing to
upload), then per position p write TOK_ID/POS, pulse GO, poll DONE, read TOK_OUT +
CYCLES. tok_out of the pass at pos p is the prediction for p+1; the prompt phase feeds
prompt chars at pos 0..plen-1, generation feeds TOK_OUT back. tok/s = FCLK / avg-CYCLES,
reported ONLY if the FULL generated token stream equals the PINNED golden reference
IntKVQSequencer(kbits=8, vbits=8, rotate=False, divfree=True).generate_greedy — the R0
contract (the bit-exact gate). --runs N (default 3, the 3/3 convention) repeats the
whole decode; all runs must also be mutually identical (the deterministic gate).

Register map (gemv_axi_seq_vec.v):
  0x00 CTRL(b0 go,b1 wl_rst,b2 soft_reset,b4:3 dbg_stop)  0x04 STATUS(b0 done,b1 busy)
  0x08 TOK_ID  0x0C POS  0x10 W_DATA  0x14 RD_SEL  0x18 RD_ADDR
  0x1C RD_DATA_LO  0x20 RD_DATA_HI  0x24 TOK_OUT  0x28 CYCLES  0x2C IDCODE(0x53515256 "SQRV")

  sudo ~/kevweb/venv/bin/python -m fabric.stage3.board.pl_seq_kv \
      --lanes 128 --prompt "once upon time" --ngen 24 --fclk 200e6
"""

from __future__ import annotations

import argparse
import json
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
from model.goformer_kvq import IntKVQSequencer  # noqa: E402

NLAYER = 4
IDCODE_SQRV = 0x53515256
BASE = 0xA0000000
R_CTRL, R_STATUS = 0x00, 0x04
R_TOKID, R_POS, R_WDATA = 0x08, 0x0C, 0x10
R_RDSEL, R_RDADDR, R_RDLO, R_RDHI = 0x14, 0x18, 0x1C, 0x20
R_TOKOUT, R_CYCLES, R_IDCODE = 0x24, 0x28, 0x2C
PROMPT8 = [48, 10, 100, 77, 5, 60, 120, 180]   # the run_vec_kv default prompt ids


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


def _encode(meta_path, text):
    """Prompt text -> token ids via the exported stoi (mirrors run_vec_kv)."""
    m = json.load(open(meta_path))
    stoi = m["stoi"]
    return [stoi.get(c, 0) for c in text]


def _decode(meta_path, ids):
    m = json.load(open(meta_path))
    itos = {int(k): v for k, v in m["itos"].items()}
    return "".join(itos.get(int(i), "?") for i in ids)


def build_weight_image(intseq, lanes):
    """Full transposed INT4 image at `lanes`: per block (qkv|proj|mlp_fc|mlp_proj) for NLAYER
    blocks, then the head — the exact order sequencer_vec's w_base offsets assume."""
    p = intseq.p
    words = []
    for bi in range(NLAYER):
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            words += pack_banked.pack_transposed(np.asarray(p["blocks"][bi][nm][0], np.int8), lanes)
    words += pack_banked.pack_transposed(np.asarray(p["head"][0], np.int8), lanes)
    # log SS36 fit-plan 2: the embed tables ride the weight image's spare depth
    # (shared packer with the sim writer so the two layouts cannot drift).
    from fabric.stage3.run_sequencer import wrom_embed_words
    emb = wrom_embed_words(intseq, lanes, len(words))
    if emb is not None:
        words += emb
    return words


def _pass(dev, tok, pos, timeout):
    """One GO pulse at (tok, pos): returns (tok_out, cycles) or None on timeout.
    go_pulse clears done_latched in the wrapper, so the poll is race-free."""
    dev.wr(R_TOKID, int(tok) & 0x1FF)
    dev.wr(R_POS, int(pos) & 0x1FF)
    dev.wr(R_CTRL, 0x1)                                   # go (dbg_stop=0)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if dev.rd(R_STATUS) & 0x1:
            return dev.rd(R_TOKOUT) & 0x1FF, dev.rd(R_CYCLES)
    return None


def run_decode(dev, prompt_ids, ngen, timeout):
    """Full multi-token decode: plen prompt passes + (ngen-1) feedback passes, positions
    0..plen+ngen-2, KV persisting in fabric throughout. No reset between runs is needed:
    a fresh decode rewrites KV[p] before any pass reads it (tcount = pos+1), so stale
    rows from a previous run are never attended to. Returns (gen_ids, per_pass_cycles)
    or (None, None) on timeout."""
    cycs = []
    nxt = None
    for p, tok in enumerate(prompt_ids):                  # prompt phase
        r = _pass(dev, tok, p, timeout)
        if r is None:
            return None, None
        nxt, cyc = r
        cycs.append(cyc)
    gen = [nxt]                                           # prediction for pos plen
    for i in range(ngen - 1):                             # greedy feedback phase
        r = _pass(dev, nxt, len(prompt_ids) + i, timeout)
        if r is None:
            return None, None
        nxt, cyc = r
        cycs.append(cyc)
        gen.append(nxt)
    return gen, cycs


def main(argv=None):
    ap = argparse.ArgumentParser(description="KV-faithful single-stream sequencer (SQRV) on silicon")
    ap.add_argument("--npz", default=os.path.join(_REPO, "fabric", "export", "goformer.npz"))
    ap.add_argument("--meta", default=os.path.join(_REPO, "fabric", "export", "goformer_meta.json"))
    ap.add_argument("--lanes", type=int, default=128)
    ap.add_argument("--prompt", default=None, help="prompt text (encoded via the meta stoi)")
    ap.add_argument("--plen", type=int, default=8, help="default-prompt length if no --prompt")
    ap.add_argument("--ngen", type=int, default=24)
    ap.add_argument("--tmax", type=int, default=256,
                    help="the bitstream's KV window (TMAX generic); prompt+gen must fit")
    ap.add_argument("--runs", type=int, default=3, help="full decodes (the 3/3 convention)")
    ap.add_argument("--fclk", type=float, default=200e6)
    ap.add_argument("--base", type=lambda s: int(s, 0), default=BASE)
    ap.add_argument("--poll-timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    if args.prompt:
        prompt_ids = _encode(args.meta, args.prompt)
    else:
        prompt_ids = PROMPT8[:args.plen]
    plen, ngen = len(prompt_ids), int(args.ngen)
    assert ngen >= 1 and plen >= 1
    assert plen + ngen - 1 <= args.tmax, "prompt+gen must fit the KV window"

    fclk_hz = set_and_verify_fclk(args.fclk)
    p, cfg = seq_ref.build(args.npz)
    gold_seq = IntKVQSequencer(p, cfg, kbits=8, vbits=8, rotate=False, divfree=True)
    gold = gold_seq.generate_greedy(list(prompt_ids), ngen)[plen:]
    lanes = args.lanes
    subw = (lanes * 4) // 32
    words = build_weight_image(gold_seq, lanes)
    ptxt = _decode(args.meta, prompt_ids)
    gtxt = _decode(args.meta, gold)
    print(f"[ref ] prompt={ptxt!r} ({plen} ids) -> gold gen={gtxt!r}"
          .encode("ascii", "replace").decode())           # cp1252-console safe
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
        print(f"[load] {len(words)*subw} chunks in {time.time()-t0:.2f}s "
              f"(embed ROMs are baked into the bitstream)")

        streams, all_match = [], True
        for r in range(int(args.runs)):
            gen, cycs = run_decode(dev, prompt_ids, ngen, args.poll_timeout)
            if gen is None:
                print(f"[run{r+1}] TIMEOUT STATUS=0x{dev.rd(R_STATUS):08X}"); return 4
            streams.append(gen)
            match = (gen == gold)
            all_match = all_match and match
            total = sum(cycs); avg = total / len(cycs)
            print(f"[run{r+1}] hw  gen={gen}")
            print(f"[run{r+1}] txt hw={_decode(args.meta, gen)!r} gold={gtxt!r}"
                  .encode("ascii", "replace").decode())
            print(f"[run{r+1}] cyc per-pass min..max = {min(cycs)}..{max(cycs)}, "
                  f"avg = {avg:.0f} over {len(cycs)} passes (T grows 1..{plen+ngen-1})")
            if match and avg > 0:
                tok_s = fclk_hz / avg
                print(f"PL_SEQKV_VERDICT match=True plen={plen} ngen={ngen} "
                      f"total_cyc={total} avg_cyc_per_tok={avg:.0f} tok_s={tok_s:.1f} "
                      f"fclk={fclk_hz:.0f} idcode=0x{idc:08X}")
            else:
                print(f"PL_SEQKV_VERDICT match={match} plen={plen} ngen={ngen} "
                      f"total_cyc={total} avg_cyc_per_tok={avg:.0f} "
                      f"(tok/s WITHHELD unless match) idcode=0x{idc:08X}")
        identical = all(s == streams[0] for s in streams)
        print(f"[det ] {len(streams)} runs mutually identical = {identical} "
              f"({sum(s == gold for s in streams)}/{len(streams)} match golden)")
        return 0 if (all_match and identical) else 1
    finally:
        dev.close()


if __name__ == "__main__":
    raise SystemExit(main())
