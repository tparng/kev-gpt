"""test_sb_kvwin — prove the SQSB sequencer's on-chip KV window on silicon.

The bench (pl_seq_sb) only ever runs POS=0, so the chat backend inherited a
T=1 assumption. But TMAX is documented as the on-chip KV window: a GO at POS=p
should attend over the KV rows written by the GOs at 0..p-1, which means true
incremental multi-token decode with the CPU feeding one token id per position.

This test drives the record bitstream position-by-position: an 8-token prompt
(one GO per position, KV accumulating), then n_gen feedback steps. Streams 0-7
get prompt A and streams 8-15 prompt B, so it also proves per-stream KV
independence across the two split-brain cohorts. The gate is exact equality of
each stream's emitted token stream with seq_ref.IntSequencer.generate_greedy.

  sudo ~/kevweb/venv/bin/python -m fabric.stage3.board.test_sb_kvwin \
      --fclk 200e6 --ngen 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from fabric.stage3.board import pl_seq_sb as sb  # noqa: E402


def enc(meta, s):
    stoi = meta["stoi"]
    return [stoi.get(c, 0) for c in s]


def dec(meta, ids):
    itos = {int(k): v for k, v in meta["itos"].items()}
    return "".join(itos.get(int(i), "?") for i in ids)


def main(argv=None):
    ap = argparse.ArgumentParser(description="SQSB on-chip KV window gate")
    ap.add_argument("--npz", default=os.path.join(_REPO, "fabric", "export", "goformer.npz"))
    ap.add_argument("--meta", default=os.path.join(_REPO, "fabric", "export", "goformer_meta.json"))
    ap.add_argument("--lanes", type=int, default=128)
    ap.add_argument("--tmax", type=int, default=sb.TMAX_DEFAULT)
    ap.add_argument("--fclk", type=float, default=200e6)
    ap.add_argument("--prompt-a", default="once upo")
    ap.add_argument("--prompt-b", default="big bear")
    ap.add_argument("--ngen", type=int, default=8)
    args = ap.parse_args(argv)

    meta = json.load(open(args.meta))
    pa = enc(meta, args.prompt_a)
    pb = enc(meta, args.prompt_b)
    L = len(pa)
    assert len(pb) == L, "prompts must be equal length (one shared POS register)"
    assert L + args.ngen - 1 <= args.tmax, "prompt+gen must fit the KV window"

    fclk_hz = sb.set_and_verify_fclk(args.fclk)
    p, cfg = sb.seq_ref.build(args.npz)
    intseq = sb.seq_ref.IntSequencer(p, cfg)

    ref_a = intseq.generate_greedy(pa, args.ngen)[L:]
    ref_b = intseq.generate_greedy(pb, args.ngen)[L:]
    print(f"[ref ] A {args.prompt_a!r} -> {dec(meta, ref_a)!r}")
    print(f"[ref ] B {args.prompt_b!r} -> {dec(meta, ref_b)!r}")

    intseq.reset()
    words = sb.build_weight_image(intseq, args.lanes)
    emb = sb.build_embed_chunks(intseq, args.tmax)
    subw = (args.lanes * 4) // 32

    dev = sb.Dev()
    try:
        idc = dev.rd(sb.R_IDCODE)
        print(f"[idc ] 0x{idc:08X} {'OK' if idc == sb.IDCODE_SQSB else 'MISMATCH'}")
        if idc != sb.IDCODE_SQSB:
            return 3
        dev.wr(sb.R_CTRL, 0x4); dev.wr(sb.R_CTRL, 0x0)
        dev.wr(sb.R_CTRL, 0x2)
        for w in words:
            w = int(w)
            for s in range(subw):
                dev.wr(sb.R_WDATA, (w >> (32 * s)) & 0xFFFFFFFF)
        for v in emb:
            dev.wr(sb.R_EDATA, v)

        def go(tok_ids, pos):
            for b in range(16):
                dev.wr(sb.R_TOKID[b], tok_ids[b] & 0x1FF)
            dev.wr(sb.R_POS, pos & 0x1FF)
            dev.wr(sb.R_CTRL, 0x1)
            for _ in range(2_000_000):
                if dev.rd(sb.R_STATUS) & 0x1:
                    break
            else:
                raise RuntimeError(f"TIMEOUT at pos={pos}")
            return [dev.rd(sb.R_TOKOUT[b]) & 0x1FF for b in range(16)], dev.rd(sb.R_CYCLES)

        gen = [[] for _ in range(16)]
        cycles = []
        t0 = time.perf_counter()
        outs = None
        for pos in range(L):                       # prompt phase: KV fills 0..L-1
            toks = [pa[pos]] * 8 + [pb[pos]] * 8
            outs, cyc = go(toks, pos)
            cycles.append(cyc)
        for s in range(16):                        # out at pos L-1 = first gen token
            gen[s].append(outs[s])
        for g in range(1, args.ngen):              # feedback phase
            outs, cyc = go(outs, L + g - 1)
            cycles.append(cyc)
            for s in range(16):
                gen[s].append(outs[s])
        wall = time.perf_counter() - t0

        ok_a = all(gen[s] == ref_a for s in range(8))
        ok_b = all(gen[s] == ref_b for s in range(8, 16))
        print(f"[hw  ] A streams 0-7  : {dec(meta, gen[0])!r}  match={ok_a}")
        print(f"[hw  ] B streams 8-15 : {dec(meta, gen[8])!r}  match={ok_b}")
        print(f"[meas] {len(cycles)} passes, cyc/pass min..max = {min(cycles)}..{max(cycles)}, "
              f"wall {wall*1000:.1f} ms for 16x{args.ngen} faithful tokens")
        if ok_a and ok_b:
            agg = 16 * args.ngen / wall
            print(f"SB_KVWIN_VERDICT match=True tmax={args.tmax} L={L} ngen={args.ngen} "
                  f"faithful_agg_tok_s={agg:.1f} fclk={fclk_hz:.0f}")
        else:
            print("SB_KVWIN_VERDICT match=False (KV window NOT confirmed at this pos protocol)")
        return 0 if (ok_a and ok_b) else 1
    finally:
        dev.close()


if __name__ == "__main__":
    raise SystemExit(main())
