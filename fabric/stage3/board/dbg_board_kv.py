"""First-silicon localiser for the doc-7 sequencer: run ONE pass (tok 48, pos 0),
then read back qkv_bank (rd_sel 1, the GEMV output — pre-KV) and ctxv_bank
(rd_sel 2, the attention output — post-KV round-trip) and compare each against
the golden reference computed on this box. Splits the fault between the GEMV
path (weights/embeds/AQ) and the KV/attention path (the XPM-side suspects).

  sudo ~/kevweb/venv/bin/python -m fabric.stage3.board.dbg_board_kv --fclk 125e6
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from fabric.stage3 import seq_ref                                    # noqa: E402
from fabric.stage3.board.pl_seq_kv import (Dev, build_weight_image,  # noqa: E402
                                           R_CTRL, R_STATUS, R_TOKID, R_POS,
                                           R_WDATA, R_IDCODE)
from fabric.stage3.board.pl_seq_chat import set_and_verify_fclk      # noqa: E402
from model.goformer_kvq import IntKVQSequencer                       # noqa: E402

R_RDSEL, R_RDADDR, R_RDLO, R_RDHI = 0x14, 0x18, 0x1C, 0x20


def rd_bank(dev, sel, n):
    out = []
    for k in range(n):
        dev.wr(R_RDSEL, sel)
        dev.wr(R_RDADDR, k)
        dev.rd(R_STATUS); dev.rd(R_STATUS)        # 2-cycle registered readback settle
        lo = dev.rd(R_RDLO)
        hi = dev.rd(R_RDHI)
        v = (hi << 32) | lo
        if v >= (1 << 63):
            v -= (1 << 64)
        out.append(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(_REPO, "fabric", "export", "goformer.npz"))
    ap.add_argument("--lanes", type=int, default=256)
    ap.add_argument("--fclk", type=float, default=125e6)
    ap.add_argument("--tok", type=int, default=48)
    a = ap.parse_args(argv)

    set_and_verify_fclk(a.fclk)
    p, cfg = seq_ref.build(a.npz)
    iseq = seq_ref.IntSequencer(p, cfg)
    words = build_weight_image(iseq, a.lanes)
    subw = (a.lanes * 4) // 32

    # golden pos-0 internals from the KVQ reference (capture the attn sink)
    gold = IntKVQSequencer(p, cfg, kbits=8, vbits=8, rotate=False, divfree=True)
    sink = {}
    orig = gold._attn_step

    def patched(x, bi, s=None):
        s2 = {}
        r = orig(x, bi, sink=s2)
        if bi == 0 and "blk0" not in sink:
            sink["blk0"] = s2
        return r

    gold._attn_step = patched
    gold.step(a.tok)

    dev = Dev()
    try:
        idc = dev.rd(R_IDCODE)
        print(f"[idc ] 0x{idc:08X}")
        dev.wr(R_CTRL, 0x4); dev.wr(R_CTRL, 0x0)
        dev.wr(R_CTRL, 0x2)
        t0 = time.time()
        for w in words:
            w = int(w)
            for s in range(subw):
                dev.wr(R_WDATA, (w >> (32 * s)) & 0xFFFFFFFF)
        print(f"[load] {len(words)*subw} chunks in {time.time()-t0:.2f}s")
        # ONE pass at pos 0 with dbg_stop=3 (halt after block 0 so the banks hold
        # the block-0 snapshot): CTRL bits [4:3] = dbg_stop
        dev.wr(R_TOKID, a.tok & 0x1FF)
        dev.wr(R_POS, 0)
        dev.wr(R_CTRL, (3 << 3) | 0x1)
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if dev.rd(R_STATUS) & 0x1:
                break
        else:
            print("[run ] TIMEOUT"); return 4

        s0 = sink["blk0"]
        qkv_hw = rd_bank(dev, 1, 768)
        qkv_ref = [int(v) for v in s0["qkv_q16"]]
        qm = sum(1 for i in range(768) if qkv_hw[i] != qkv_ref[i])
        print(f"[qkv ] mismatches {qm}/768 "
              f"(first hw={qkv_hw[0]} ref={qkv_ref[0]})")

        # ctx reference at pos 0: recompute exactly as _attn_step does, from the sink
        from fabric.stage3.run_softmax import int_softmax_q
        from fabric.stage3.seq_ref import (rsh_round, sat, ISQRT, SCORE_FRAC,
                                           PROB_FRAC_A, RESID_FRAC, VFRAC, NHEAD,
                                           HEAD_DIM)
        q = s0["q_rot_q16"]; ks = s0["k_stored_q16"]; vs = s0["v_stored_q16"]
        ctx_ref = [0] * 256
        for h in range(NHEAD):
            b = h * HEAD_DIM
            acc = sum(q[b + d] * ks[b + d] for d in range(HEAD_DIM))
            sq = np.array([sat(rsh_round(acc, 2 * VFRAC + ISQRT - SCORE_FRAC),
                               -32768, 32767)], dtype=np.int64)
            prob = int_softmax_q(sq, np.ones(1, dtype=bool), gold.exp_lut)
            for d in range(HEAD_DIM):
                ctx_ref[b + d] = rsh_round(int(prob[0]) * vs[b + d],
                                           PROB_FRAC_A + VFRAC - RESID_FRAC)
        ctx_hw = rd_bank(dev, 2, 256)
        cm = sum(1 for i in range(256) if ctx_hw[i] != ctx_ref[i])
        print(f"[ctx ] mismatches {cm}/256 "
              f"(first hw={ctx_hw[0]} ref={ctx_ref[0]})")
        print(f"DBG_VERDICT qkv_ok={qm == 0} ctx_ok={cm == 0}")
        return 0
    finally:
        dev.close()


if __name__ == "__main__":
    raise SystemExit(main())
