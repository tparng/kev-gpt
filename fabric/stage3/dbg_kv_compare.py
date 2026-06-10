"""Doc-7 R1 debug: compare the RTL's KVW/KVR traces (vvp_stdout.log, KVDBG build)
against IntKVQSequencer's per-call capture for a 2-pass run (pos 0, pos 1).

    python -m fabric.stage3.dbg_kv_compare --dir C:/kevbuild/stage3_vec_kv2
"""

from __future__ import annotations

import argparse
import os
import re

from fabric.stage3 import seq_ref
from model.goformer_kvq import IntKVQSequencer, INV_SH
from fabric.stage3.seq_ref import q_round_div, NHEAD, HEAD_DIM


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="C:/kevbuild/stage3_vec_kv2")
    ap.add_argument("--toks", default="48,10")
    a = ap.parse_args(argv)
    toks = [int(t) for t in a.toks.split(",")]

    p, cfg = seq_ref.build("fabric/export/goformer.npz")
    seq = IntKVQSequencer(p, cfg, kbits=8, vbits=8, rotate=False, divfree=True)

    captures = []          # (pos, blk, sink)
    orig = seq._attn_step
    posbox = {"pos": 0, "blk": 0}

    def patched(x, bi, sink=None):
        s = {}
        r = orig(x, bi, sink=s)
        captures.append((posbox["pos"], bi, s))
        return r

    seq._attn_step = patched
    for i, t in enumerate(toks):
        posbox["pos"] = i
        seq.step(int(t))

    # reference hdrs per (pos, blk, kv, head) from the recorded DDR rows
    def ref_hdr(pos, blk, kv, h):
        row = seq.kv_ddr[blk][pos]["k_row" if kv == 0 else "v_row"]
        hb = 38 if False else (6 + HEAD_DIM)        # K8: 6 hdr + 64 codes = 70 B/head
        off = h * hb
        lo = int.from_bytes(bytes(row[off:off+4]), "little", signed=True)
        sc = int.from_bytes(bytes(row[off+4:off+6]), "little")
        return lo, sc

    # parse the RTL trace
    log = os.path.join(a.dir, "vvp_stdout.log")
    kvw = []   # (blk, kv, h, pos, lo, scale, inv)
    kvr = []   # (blk, kv, h, l0) in stream order
    for ln in open(log, encoding="utf-8", errors="replace"):
        m = re.match(r"KVW blk=(\d+) kv=(\d+) h=(\d+) pos=(\d+) lo=(-?\d+) scale=(\d+) inv=(\d+)", ln)
        if m:
            kvw.append(tuple(int(g) for g in m.groups()))
        m = re.match(r"KVR blk=(\d+) kv=(\d+) h=(\d+) l0=(-?\d+)", ln)
        if m:
            kvr.append(tuple(int(g) for g in m.groups()))

    print("=== KVW hdr compare (all writes) ===")
    bad = 0
    for (blk, kv, h, pos, lo, sc, inv) in kvw:
        rlo, rsc = ref_hdr(pos, blk, kv, h)
        rinv = q_round_div(1 << INV_SH, rsc)
        ok = (lo == rlo and sc == rsc and inv == rinv)
        if not ok:
            bad += 1
            print(f"  MISMATCH blk={blk} kv={kv} h={h} pos={pos}: "
                  f"rtl(lo={lo},sc={sc},inv={inv}) ref(lo={rlo},sc={rsc},inv={rinv})")
    print(f"  {len(kvw)} writes, {bad} mismatches")

    # KVR check: for pass at pos1, blk0, K, h0: stream = pos0 rows then pos1 rows.
    # lane0 of row (j, g) = k_deq_q16[h*64 + g*8] at position j.
    print("=== KVR stream compare (pass pos=1, blk0, K, h0) ===")
    want = []
    for j in range(2):
        kd = seq.kv_ddr[0][j]["k_deq_q16" if True else None]
        for g in range(HEAD_DIM // 8):
            want.append(kd[0 * 64 + g * 8])
    # find the SECOND pass's first K/h0/blk0 stream: skip pass-1 reads (8 rows)
    got_all = [l0 for (blk, kv, h, l0) in kvr if blk == 0 and kv == 0 and h == 0]
    got = got_all[8:8+16]          # pass2: 2 positions x 8 rows
    print(f"  rtl : {got}")
    print(f"  ref : {want}")
    print(f"  match = {got == want}")


if __name__ == "__main__":
    main()
