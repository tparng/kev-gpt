"""kv_prefetch RTL gate (KV-DDR-100K.md §2-3, rung 3): the double-buffered
(ping-pong) per-head KV window between kv_dma and the attention unit.

    python -m fabric.stage3.run_kv_prefetch

Op: arm a TMAX-position window; for each of NHEAD heads, the brick prefetches head
h+1's K then V rows through kv_dma into the OTHER ping-pong half while head h is
being consumed (a modelled per-head compute window). The consumer reads each head's
K window then V window as HEAD_DIM-wide Q.16 words (vec_attn load-phase shaped).

Gate: every presented K/V channel word MUST equal model.goformer_kvq's
k_deq_q16 / v_deq_q16 — head by head, position by position, channel by channel,
mismatches=0 — at EVERY swept first-word latency. AND the prefetch must hide DDR
latency: report measured stall cycles per head (the whole point: ~0 once the
compute window exceeds the fill time). Prints KV_PREFETCH_VERDICT.

Compiles ONLY the new bricks (rtl/kv_prefetch.sv + rtl/kv_dma.sv +
rtl/ddr_latency_model.sv + tb/tb_kv_prefetch.sv) — touches no sequencer RTL.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

NHEAD = 4
HEAD_DIM = 64
KBITS = 4
VBITS = 4
WORDW = 24
P = 8                                   # channels/cycle the wide emit reconstructs
WMASK = (1 << WORDW) - 1
HERE = os.path.dirname(os.path.abspath(__file__))


def _build_ddr_image(npz, prompt_len, seed, tpos, nlayers_cap):
    """Replay a decode on the real export; lay every (layer,K|V,position) position-row
    out contiguously in a flat DDR byte image. Returns:
      ddr_bytes          - the flat DDR image
      row_addr[layer][kv][pos] = base byte address of that position-row
      gold[layer][head][kv][pos] = list of HEAD_DIM full-int Q.16 words (masked WORDW)
      row_bytes, nlayers, tpos
    A window uses the FIRST `tpos` positions of each layer (the live window the
    prefetch streams); the harness reads head-major K-then-V per the tb dump order."""
    from fabric.stage3.seq_ref import build
    from model.goformer_kvq import IntKVQSequencer

    p, cfg = build(npz)
    rng = np.random.default_rng(seed)
    # generate enough positions that every layer has >= tpos captured
    prompt = list(rng.integers(0, p["tok_emb"].shape[0], size=max(prompt_len, tpos)))

    kvq = IntKVQSequencer(p, cfg, kbits=KBITS, vbits=VBITS, rotate=True)
    kvq.kv_ddr_words(prompt)                 # populates kvq.kv_ddr[layer][pos]

    nlayers = min(kvq.nblocks, nlayers_cap)
    ddr = bytearray()
    row_addr = [[[0] * tpos for _ in range(2)] for _ in range(nlayers)]   # [layer][kv][pos]
    gold = [[[[None] * tpos for _ in range(2)] for _ in range(NHEAD)] for _ in range(nlayers)]

    for bi in range(nlayers):
        entries = kvq.kv_ddr[bi]
        assert len(entries) >= tpos, f"layer {bi} has {len(entries)} < tpos={tpos}"
        for kvi, kv in enumerate(("k", "v")):
            for pos in range(tpos):
                e = entries[pos]
                row = e[kv + "_row"]
                deq = e[kv + "_deq_q16"]
                assert row is not None and deq is not None
                assert len(deq) == NHEAD * HEAD_DIM
                base = len(ddr)
                ddr += bytes(int(b) & 0xFF for b in row)
                row_addr[bi][kvi][pos] = base
                for h in range(NHEAD):
                    seg = deq[h * HEAD_DIM:(h + 1) * HEAD_DIM]
                    gold[bi][h][kvi][pos] = [int(w) & WMASK for w in seg]

    row_bytes = NHEAD * (6 + HEAD_DIM * KBITS // 8)     # 6-B hdr (lo32||scale16)
    return bytes(ddr), row_addr, gold, row_bytes, nlayers, tpos


def _flatten_gold(gold, nlayers, tpos):
    """Flatten gold to the exact order tb_kv_prefetch dumps: per layer, per head,
    K window (pos 0..T-1, each HEAD_DIM channels) then V window, channel-major."""
    flat_rows = []   # one list per (layer,head,kv-window-block) is implicit; we just
                     # build the full expected channel stream split by HEAD/LAYER markers
    # We return a structured list mirroring the tb's marker layout for compare().
    per_head_blocks = []   # list over (layer,head) of expected channel list
    for bi in range(nlayers):
        for h in range(NHEAD):
            block = []
            for kvi in range(2):                 # K then V
                for pos in range(tpos):
                    block.extend(gold[bi][h][kvi][pos])
            per_head_blocks.append(block)
    return per_head_blocks


def _write_mems(sim_dir, ddr_bytes, row_addr, ddr_depth, nlayers, tpos):
    img = bytearray(ddr_bytes)
    if len(img) < ddr_depth:
        img += bytes(ddr_depth - len(img))
    with open(os.path.join(sim_dir, "ddr.mem"), "w") as fh:
        fh.write("\n".join(f"{b:02x}" for b in img) + "\n")
    # addr.mem order matches tb rom index = (layer*2 + kv)*tpos + pos
    lines = []
    for bi in range(nlayers):
        for kvi in range(2):
            for pos in range(tpos):
                lines.append(f"{row_addr[bi][kvi][pos]:06x}")
    with open(os.path.join(sim_dir, "addr.mem"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _compile(sim_dir, tpos, nlayers, ddr_depth, first_word_lat, beat_gap, per_head_cyc):
    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(
        ["iverilog", "-g2012", "-o", vvp,
         f"-DTPOS={tpos}", f"-DNLAYERS={nlayers}", f"-DDDR_DEPTH={ddr_depth}",
         f"-DFIRST_WORD_LAT={first_word_lat}", f"-DBEAT_GAP={beat_gap}",
         f"-DKBITS={KBITS}", f"-DPER_HEAD_CYC={per_head_cyc}", f"-DP={P}",
         os.path.join(HERE, "tb", "tb_kv_prefetch.sv"),
         os.path.join(HERE, "rtl", "kv_prefetch.sv"),
         os.path.join(HERE, "rtl", "kv_dma.sv"),
         os.path.join(HERE, "rtl", "ddr_latency_model.sv")],
        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL")
        print(cp.stdout, cp.stderr)
        return None
    return vvp


def _run_vvp(sim_dir):
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL")
        print(rp.stdout[-2000:], rp.stderr[-2000:])
        return None, None, None
    fill_cyc = None
    for line in rp.stdout.splitlines():
        if line.startswith("TB_FILLCYC"):
            fill_cyc = int(line.split()[1])
    # parse kv.out into per-(layer,head) channel blocks, split on HEAD markers
    blocks, cur = [], []
    with open(os.path.join(sim_dir, "kv.out")) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line == "HEAD":
                blocks.append(cur); cur = []
            elif line == "LAYER":
                continue
            else:
                try:
                    cur.append(int(line, 16) & WMASK)
                except ValueError:
                    cur.append(-1)
    if cur:
        blocks.append(cur)
    stalls = []
    with open(os.path.join(sim_dir, "stall.out")) as fh:
        for line in fh:
            line = line.strip()
            if line:
                stalls.append(int(line))
    return blocks, stalls, fill_cyc


def _compare(expected_blocks, rtl_blocks):
    total, mism, bad = 0, 0, 0
    if len(rtl_blocks) != len(expected_blocks):
        # block-count mismatch is itself a failure
        return False, -1, 0, abs(len(rtl_blocks) - len(expected_blocks))
    for exp, got in zip(expected_blocks, rtl_blocks):
        total += len(exp)
        if len(got) != len(exp):
            mism += abs(len(got) - len(exp)) + min(len(got), len(exp))
            bad += 1
            continue
        rm = sum(1 for a, b in zip(exp, got) if a != b)
        mism += rm
        if rm:
            bad += 1
    return (mism == 0), mism, total, bad


def run(sim_dir, npz="fabric/export/goformer.npz", prompt_len=12, seed=0,
        tpos=8, nlayers=2, latencies=(1, 8, 30), beat_gap=0, per_head_cyc=2200):
    os.makedirs(sim_dir, exist_ok=True)
    ddr_bytes, row_addr, gold, row_bytes, nlayers, tpos = _build_ddr_image(
        npz, prompt_len, seed, tpos, nlayers)
    expected_blocks = _flatten_gold(gold, nlayers, tpos)
    ddr_depth = max(65536, ((len(ddr_bytes) + 4095) // 4096) * 4096)
    _write_mems(sim_dir, ddr_bytes, row_addr, ddr_depth, nlayers, tpos)

    # per-window compute window = NHEAD heads * per_head_cyc (the budget the prefetch
    # must hide the next window's 2*T-row fill behind). The doc cites ~1,300 cyc/head.
    win_compute = NHEAD * per_head_cyc
    print(f"src={npz} prompt_len={prompt_len} layers={nlayers} T(window)={tpos} "
          f"row_bytes={row_bytes} (NHEAD={NHEAD} head-major, K{KBITS}/V{VBITS}) "
          f"ddr_image={len(ddr_bytes)}B per_head_cyc={per_head_cyc} "
          f"win_compute={win_compute} blocks(layer*head)={len(expected_blocks)}")

    all_ok = True
    for L in latencies:
        vvp = _compile(sim_dir, tpos, nlayers, ddr_depth, L, beat_gap, per_head_cyc)
        if vvp is None:
            return False
        rtl_blocks, stalls, fill_cyc = _run_vvp(sim_dir)
        if rtl_blocks is None:
            return False
        ok, mism, total, bad = _compare(expected_blocks, rtl_blocks)
        all_ok = all_ok and ok
        # stalls is per-WINDOW (NLAYERS entries): stalls[0] = cold window 0 (no prior
        # compute to hide behind, but the tb fills it before the first read so ~0);
        # the rest are warm windows whose 2*T-row fill should be hidden by the prior
        # window's compute -> 0 once win_compute >= fill_cyc.
        cold = stalls[0] if stalls else -1
        warm = stalls[1:] if len(stalls) > 1 else []
        warm_max = max(warm) if warm else 0
        # fill_cyc is the whole-window fill = 2*T row reads; derive per-row cost.
        nrows_win = 2 * tpos
        per_row = (fill_cyc / nrows_win) if (fill_cyc and nrows_win) else -1
        hidden = (warm_max == 0)
        print(f"  L={L:>2} first-word cyc: bitexact={ok} mismatches={mism}/{total} "
              f"bad_blocks={bad} blocks={len(rtl_blocks)}/{len(expected_blocks)} | "
              f"fill_cyc(2*T rows)={fill_cyc} per_row~{per_row:.1f} "
              f"win_compute={win_compute} stall/window={stalls} "
              f"(window0={cold}, warm_max={warm_max}, hidden={hidden})")

    print(f"KV_PREFETCH_VERDICT bitexact_all={all_ok} latencies={list(latencies)} "
          f"T={tpos} layers={nlayers} per_head_cyc={per_head_cyc} emit_P={P}")
    print("KV_PREFETCH_" + ("OK" if all_ok else "FAIL"))
    return all_ok


def main():
    base = os.path.join("C:\\kevbuild", "agent_kv4", "kv_prefetch")
    # default warm window (T=8) sanity, then the full T=32 Kevin window: prove the
    # 2*T-row fill hides behind an ~18k-cyc token compute (per_head_cyc=4500 -> NHEAD*
    # 4500 = 18,000-cyc compute window, the single-token compute floor).
    ok8 = run(os.path.join(base, "t8"), tpos=8, nlayers=2)
    print()
    ok32 = run(os.path.join(base, "t32"), tpos=32, nlayers=2, per_head_cyc=4500)
    raise SystemExit(0 if (ok8 and ok32) else 1)


if __name__ == "__main__":
    main()
