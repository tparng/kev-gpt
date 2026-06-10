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
    KBITS = 4
    seq = IntKVQSequencer(p, cfg, kbits=KBITS, vbits=KBITS, rotate=True, divfree=True)

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
        hb = 6 + HEAD_DIM * KBITS // 8              # K4: 6 hdr + 32 code bytes = 38 B/head
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

    # KVR check: for pass at pos1, blk0, K, h0: the wide reader emits ONE row per
    # position; lane0 of position j = k_deq_q16[h*64 + 0] (the dequantised value
    # the cache reconstructs from the stored K4 codes).
    print("=== KVR stream compare (pass pos=1, blk0, K, h0) ===")
    want = [seq.kv_ddr[0][j]["k_deq_q16"][0] for j in range(2)]
    # the SECOND pass's K/h0/blk0 stream: skip pass-1's read (1 position)
    got_all = [l0 for (blk, kv, h, l0) in kvr if blk == 0 and kv == 0 and h == 0]
    got = got_all[1:1+2]           # pass2: 2 positions x 1 row
    print(f"  rtl : {got}")
    print(f"  ref : {want}")
    print(f"  match = {got == want}")


if __name__ == "__main__":
    main()
