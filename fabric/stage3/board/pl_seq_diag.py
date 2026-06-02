"""pl_seq_diag — localize the on-silicon sequencer data-path non-determinism.

The capstone bitstream runs (FSM cycle count is bit-identical to sim) but emits a
WRONG and NON-DETERMINISTIC token stream: the final residual x differs run-to-run with
frozen weights + frozen prompt. The AXI-level iverilog gate (same W_DATA URAM-load path,
same FSM order) passes bit-exact, so the bug is HARDWARE-PHYSICAL, not RTL-logical.

This probe separates the candidate causes WITHOUT a resynthesis:

  A) MMIO stability   : read IDCODE/CYCLES many times -> is the AXI read path itself
                        stable? (rules out /dev/mem caching / numpy access flakiness)
  B) load vs compute  : load weights ONCE, then fire GO N times re-writing only the
                        prompt (NO weight re-stream, NO soft_reset of the URAM). If the
                        residual is IDENTICAL across all N -> the URAM holds its content
                        and the compute is deterministic; the run-to-run difference comes
                        from the (re)LOAD path. If it DIFFERS across N -> there is a
                        per-run non-deterministic source in the compute (independent of
                        the load).
  C) reload determinism: load+GO, then reload+GO -> does a fresh reload reproduce the
                        same residual? (is the load itself repeatable?)

Read-only on the model side; only pokes the PL we already own. Run as root for /dev/mem:
  sudo ~/kevweb/venv/bin/python -m fabric.stage3.board.pl_seq_diag --npz goformer.npz \
       --meta goformer_meta.json --prompt "once upo" --gos 4
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

from fabric.stage3 import pack_banked, seq_ref  # noqa: E402
from fabric.stage3.board.pl_seq_chat import (  # noqa: E402
    Seqr, load_meta, build_weight_image, set_and_verify_fclk,
    NLAYER, LANES, PROMPT_LEN, NGEN, FCLK, IDCODE_EXPECTED, BASE,
    R_CTRL, R_STATUS, R_WDATA, R_PLDATA, R_PLADDR,
    R_RDADDR, R_RDDATA, R_TSADDR, R_TSDATA, R_NGEN, R_CYCLES, R_IDCODE,
)


def stream_weights(dev, words):
    dev.wr(R_CTRL, 0x2)                            # wl_rst
    for w in words:
        w = int(w)
        dev.wr(R_WDATA, w & 0xFFFFFFFF)
        dev.wr(R_WDATA, (w >> 32) & 0xFFFFFFFF)


def write_prompt(dev, prompt_ids):
    for i, tok in enumerate(prompt_ids):
        dev.wr(R_PLADDR, i)
        dev.wr(R_PLDATA, int(tok) & 0x1FF)


def go_and_read(dev, ndump=8, timeout=30.0):
    dev.wr(R_CTRL, 0x1)                            # go
    t0 = time.time()
    while time.time() - t0 < timeout:
        if dev.rd(R_STATUS) & 0x1:
            break
    ng = dev.rd(R_NGEN) & 0x1FF
    cyc = dev.rd(R_CYCLES)
    toks = []
    for i in range(ng):
        dev.wr(R_TSADDR, i)
        toks.append(dev.rd(R_TSDATA) & 0x1FF)
    resid = []
    for i in range(ndump):
        dev.wr(R_RDADDR, i)
        raw = dev.rd(R_RDDATA)
        sv = raw - (1 << 32) if raw & 0x80000000 else raw
        resid.append(sv)                          # raw Q6.25 int (exact for compare)
    return ng, cyc, toks, resid


def main(argv=None):
    ap = argparse.ArgumentParser(description="sequencer on-silicon non-determinism probe")
    ap.add_argument("--npz", default=os.path.join(_REPO, "fabric", "export", "goformer.npz"))
    ap.add_argument("--meta", default=os.path.join(_REPO, "fabric", "export", "goformer_meta.json"))
    ap.add_argument("--prompt", default="once upo")
    ap.add_argument("--base", type=lambda s: int(s, 0), default=BASE)
    ap.add_argument("--gos", type=int, default=4, help="GO repeats per phase")
    ap.add_argument("--ndump", type=int, default=8)
    args = ap.parse_args(argv)

    set_and_verify_fclk(FCLK)             # force PL clock to closure freq before probing
    enc, dec, vocab = load_meta(args.meta)
    p, cfg = seq_ref.build(args.npz)
    intseq = seq_ref.IntSequencer(p, cfg)
    prompt_ids = enc(args.prompt)[:PROMPT_LEN]
    if len(prompt_ids) < PROMPT_LEN:
        prompt_ids += [0] * (PROMPT_LEN - len(prompt_ids))
    ref_full = intseq.generate_greedy(list(prompt_ids), NGEN)
    ref_gen = [int(t) for t in ref_full[PROMPT_LEN:]]
    words = build_weight_image(intseq, LANES, NLAYER)

    print(f"[ref ] prompt={prompt_ids} ref_gen={ref_gen} {dec(ref_gen)!r}")
    print(f"[wimg] {len(words)} words")

    dev = Seqr(args.base)
    try:
        idc = dev.rd(R_IDCODE)
        print(f"[idc ] 0x{idc:08X} {'OK' if idc == IDCODE_EXPECTED else 'MISMATCH'}")
        if idc != IDCODE_EXPECTED:
            return 3

        # ---- A) MMIO stability: hammer IDCODE -----------------------------------
        idcs = {dev.rd(R_IDCODE) for _ in range(2000)}
        print(f"[A.mmio] IDCODE over 2000 reads: distinct={sorted(hex(x) for x in idcs)} "
              f"-> {'STABLE' if idcs == {IDCODE_EXPECTED} else 'FLAKY'}")

        # ---- B) load ONCE, GO N times (prompt rewritten, NO reload, NO soft_reset) ---
        dev.wr(R_CTRL, 0x4); dev.wr(R_CTRL, 0x0)      # one soft_reset before the first load
        stream_weights(dev, words)
        print(f"[B] weights streamed ONCE; firing {args.gos} GOs with NO reload:")
        residsB = []
        for k in range(args.gos):
            write_prompt(dev, prompt_ids)
            ng, cyc, toks, resid = go_and_read(dev, args.ndump)
            residsB.append(tuple(resid))
            print(f"  [B.go{k}] cyc={cyc} toks={toks} resid0={resid[:4]}")
        uniqB = len(set(residsB))
        print(f"[B] distinct residuals across {args.gos} no-reload GOs = {uniqB} "
              f"-> {'DETERMINISTIC (URAM stable; bug is in LOAD path)' if uniqB == 1 else 'NON-DETERMINISTIC (per-run compute source, NOT the load)'}")

        # ---- C) reload determinism: fresh reload+GO twice -----------------------
        print(f"[C] reload+GO twice (fresh stream each):")
        residsC = []
        for k in range(2):
            dev.wr(R_CTRL, 0x4); dev.wr(R_CTRL, 0x0)
            stream_weights(dev, words)
            write_prompt(dev, prompt_ids)
            ng, cyc, toks, resid = go_and_read(dev, args.ndump)
            residsC.append(tuple(resid))
            print(f"  [C.reload{k}] cyc={cyc} toks={toks} resid0={resid[:4]}")
        print(f"[C] reload reproducible = {residsC[0] == residsC[1]}")

        print("PL_SEQ_DIAG_DONE "
              f"mmio={'stable' if idcs == {IDCODE_EXPECTED} else 'flaky'} "
              f"noreload_distinct={uniqB}/{args.gos} "
              f"reload_repro={residsC[0] == residsC[1]}")
        return 0
    finally:
        dev.close()


if __name__ == "__main__":
    raise SystemExit(main())
