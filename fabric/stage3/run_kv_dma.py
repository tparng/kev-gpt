"""kv_dma RTL gate: bit-exact burst read + dequant vs goformer_kvq, under a sim
DDR latency model swept across first-word latencies (Demo P1, KV-DDR-100K.md §2).

    python -m fabric.stage3.run_kv_dma

Op: read one per-position DDR row (the pinned goformer_kvq byte layout —
[head_hdr | head_codes]*NHEAD, lo(int32)||scale(uint16) 6-B header then LSB-first
nibbles, 38 B/head at K4 -> 152 B/position) through a simple byte-wide burst-read
port, parse hdr, unpack nibbles, dequant code*scale+lo to signed Q.16, and present
the reconstructed channels per head.

Gate: the reconstructed channel words MUST equal model.goformer_kvq's k_deq_q16 /
v_deq_q16 (the exact integer cache the sequencer computes attention from), word
for word, mismatches=0 — at EVERY latency setting (the DDR latency model exists to
flush ready/valid handshake bugs before real AXI). Prints KV_DMA_VERDICT and the
per-row cycle counts at each latency.

Compiles ONLY the new bricks (rtl/kv_dma.sv + rtl/ddr_latency_model.sv + the tb) —
it does NOT touch the gemm core or any sequencer RTL.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

from fabric.stage3._simdir import kevbuild

NHEAD = 4
HEAD_DIM = 64
KBITS = 4
VBITS = 4
WORDW = 24                              # reconstructed Q.16 word width (full int)
P = 8                                   # channels/cycle the wide emit reconstructs
WMASK = (1 << WORDW) - 1
HERE = os.path.dirname(os.path.abspath(__file__))


def _build_ddr_image(npz, prompt_len, seed):
    """Replay a decode on the real export; lay every (layer,K|V,position) row out
    contiguously in a flat DDR byte image. Return (ddr_bytes, schedule) where
    schedule is a list of (base_addr, gold_channels) — gold is the NHEAD*HEAD_DIM
    full-int k_deq_q16/v_deq_q16 words masked to WORDW two's complement."""
    from fabric.stage3.seq_ref import build
    from model.goformer_kvq import IntKVQSequencer

    p, cfg = build(npz)
    rng = np.random.default_rng(seed)
    prompt = list(rng.integers(0, p["tok_emb"].shape[0], size=prompt_len))

    kvq = IntKVQSequencer(p, cfg, kbits=KBITS, vbits=VBITS, rotate=True)
    kvq.kv_ddr_words(prompt)             # populates kvq.kv_ddr[layer][pos]

    ddr = bytearray()
    schedule = []
    nb = kvq.nblocks
    for bi in range(nb):
        for kv in ("k", "v"):
            for pos, e in enumerate(kvq.kv_ddr[bi]):
                row = e[kv + "_row"]
                deq = e[kv + "_deq_q16"]
                assert row is not None and deq is not None
                base = len(ddr)
                ddr += bytes(int(b) & 0xFF for b in row)
                gold = [int(w) & WMASK for w in deq]
                assert len(gold) == NHEAD * HEAD_DIM
                schedule.append((base, gold))
    row_bytes = NHEAD * (6 + HEAD_DIM * KBITS // 8)     # 6-B hdr (lo32||scale16)
    return bytes(ddr), schedule, row_bytes, nb


def _write_mems(sim_dir, ddr_bytes, schedule, ddr_depth):
    img = bytearray(ddr_bytes)
    if len(img) < ddr_depth:
        img += bytes(ddr_depth - len(img))     # pad so $readmemh fills the model
    with open(os.path.join(sim_dir, "ddr.mem"), "w") as fh:
        fh.write("\n".join(f"{b:02x}" for b in img) + "\n")
    with open(os.path.join(sim_dir, "addr.mem"), "w") as fh:
        fh.write("\n".join(f"{base:06x}" for base, _ in schedule) + "\n")


def _compile(sim_dir, nrows, ddr_depth, first_word_lat, beat_gap):
    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(
        ["iverilog", "-g2012", "-o", vvp,
         f"-DNROWS={nrows}", f"-DDDR_DEPTH={ddr_depth}",
         f"-DFIRST_WORD_LAT={first_word_lat}", f"-DBEAT_GAP={beat_gap}",
         f"-DKBITS={KBITS}", f"-DP={P}",
         os.path.join(HERE, "tb", "tb_kv_dma.sv"),
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
        return None, None
    # parse kv.out into per-row channel lists
    rtl_rows, cur = [], []
    with open(os.path.join(sim_dir, "kv.out")) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line == "ROW":
                rtl_rows.append(cur); cur = []
            else:
                try:
                    cur.append(int(line, 16) & WMASK)
                except ValueError:
                    cur.append(-1)             # X -> never matches
    if cur:
        rtl_rows.append(cur)
    cyc = []
    with open(os.path.join(sim_dir, "cyc.out")) as fh:
        for line in fh:
            line = line.strip()
            if line:
                cyc.append(int(line))
    return rtl_rows, cyc


def _compare(schedule, rtl_rows):
    total, mism, bad_rows = 0, 0, 0
    for (_, gold), got in zip(schedule, rtl_rows):
        total += len(gold)
        if len(got) != len(gold):
            mism += abs(len(got) - len(gold)) + min(len(got), len(gold))
            bad_rows += 1
            continue
        rm = sum(1 for a, b in zip(gold, got) if a != b)
        mism += rm
        if rm:
            bad_rows += 1
    row_ok = (len(rtl_rows) == len(schedule))
    return (mism == 0 and row_ok), mism, total, bad_rows, len(rtl_rows)


def run(sim_dir, npz="fabric/export/goformer.npz", prompt_len=8, seed=0,
        latencies=(1, 8, 30), beat_gap=0):
    os.makedirs(sim_dir, exist_ok=True)
    ddr_bytes, schedule, row_bytes, nb = _build_ddr_image(npz, prompt_len, seed)
    nrows = len(schedule)
    ddr_depth = max(65536, ((len(ddr_bytes) + 4095) // 4096) * 4096)
    _write_mems(sim_dir, ddr_bytes, schedule, ddr_depth)

    print(f"src={npz} prompt_len={prompt_len} layers={nb} rows={nrows} "
          f"row_bytes={row_bytes} (NHEAD={NHEAD} head-major, K{KBITS}/V{VBITS}) "
          f"ddr_image={len(ddr_bytes)}B emit_P={P} (channels/cycle)")

    all_ok = True
    summary = []
    for L in latencies:
        vvp = _compile(sim_dir, nrows, ddr_depth, L, beat_gap)
        if vvp is None:
            return False
        rtl_rows, cyc = _run_vvp(sim_dir)
        if rtl_rows is None:
            return False
        ok, mism, total, bad_rows, n_rtl = _compare(schedule, rtl_rows)
        all_ok = all_ok and ok
        # cyc[r] = cycles from start pulse to row_valid for row r (req->valid->readout)
        cyc_med = int(np.median(cyc)) if cyc else -1
        cyc_min = min(cyc) if cyc else -1
        cyc_max = max(cyc) if cyc else -1
        summary.append((L, ok, mism, total, cyc_min, cyc_med, cyc_max))
        print(f"  L={L:>2} first-word cyc: bitexact={ok} mismatches={mism}/{total} "
              f"bad_rows={bad_rows} rows={n_rtl}/{nrows} "
              f"cyc/row(req->valid) min/med/max={cyc_min}/{cyc_med}/{cyc_max}")

    print(f"KV_DMA_VERDICT bitexact_all={all_ok} latencies={list(latencies)} "
          f"beat_gap={beat_gap} rows={nrows}")
    print("KV_DMA_" + ("OK" if all_ok else "FAIL"))
    return all_ok


def main():
    sim_dir = kevbuild("agent_kv4", "kv_dma")
    raise SystemExit(0 if run(sim_dir) else 1)


if __name__ == "__main__":
    main()
