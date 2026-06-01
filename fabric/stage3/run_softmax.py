"""Softmax RTL gate: bit-true vs an integer reference, + token-stream identity.

    python -m fabric.stage3.run_softmax

Op: softmax over a causal row of T scores (T up to 256).
    probs[j] = exp(s[j] - max) / sum_k exp(s[k] - max)

Pinned fixed-point datapath (bit-true to rtl/softmax.sv):
  score in   : signed Q8.8 (16-bit); real model |score| ~< 14, headroom +-128.
  z          : clip(s - max, -16, 0) in Q8.8 -> z-index = (-zq) in 0..4096.
  exp LUT    : 4097 entries, exp(-i/256) as Q1.20 (LUT[0] = 2^20).
  sum        : 29-bit accumulator of the T exp values (Q1.20).
  reciprocal : r = floor(2^40 / sum), 21-bit, by restoring long division.
  probs out  : (e[j] * r) >> 20  -> Q1.20 (21-bit).

`int_softmax_q` is the exact integer arithmetic the RTL reproduces. Two gates:
  1) MODULE bit-true : RTL prob == int_softmax_q.probs_q over random in-range rows,
     mismatches = 0. Prints SOFTMAX_VERDICT.
  2) TOKEN-STREAM    : plug int_softmax_q (as the model's _smax) into goformer and
     show greedy decode is IDENTICAL to the float reference over ~60 tokens.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

SCORE_FRAC = 8                      # score Q8.8
EXP_FRAC = 20                       # exp output Q1.20
ZMAX = 16                           # exp LUT input range [-16, 0]
PROB_FRAC = 20                      # prob output Q1.20
RECIP_R = 40                        # reciprocal = floor(2^40 / sum)
NZ = ZMAX * (1 << SCORE_FRAC)       # 4096 -> LUT has NZ+1 = 4097 entries
SCALE_S = 1 << SCORE_FRAC           # 256
HERE = os.path.dirname(os.path.abspath(__file__))


def exp_table():
    """4097 Q1.20 exp LUT entries; entry i = round(exp(-i/256) * 2^20)."""
    i = np.arange(NZ + 1)
    return np.round(np.exp(-i.astype(np.float64) / SCALE_S) * (1 << EXP_FRAC)).astype(np.int64)


def quantize_scores(s_row):
    """Float score row (with -inf for causal-masked entries) -> (sq_int, finite_mask).
    sq_int is Q8.8 (clamped to signed 16-bit); masked entries flagged, not quantized."""
    s_row = np.asarray(s_row, dtype=np.float64)
    finite = np.isfinite(s_row)
    sq = np.zeros_like(s_row, dtype=np.int64)
    sq[finite] = np.clip(np.round(s_row[finite] * SCALE_S), -32768, 32767).astype(np.int64)
    return sq, finite


def int_softmax_q(sq_int, finite, exp_lut):
    """The exact integer softmax the RTL does, on already-quantized Q8.8 scores.
    Only the `finite` (valid/causal) entries participate; returns Q1.20 probs (int)."""
    sq_int = np.asarray(sq_int, dtype=np.int64)
    finite = np.asarray(finite, dtype=bool)
    if not finite.any():
        return np.zeros_like(sq_int)
    m = sq_int[finite].max()
    zq = np.clip(sq_int - m, -NZ, 0)                 # <= 0 for valid
    e = exp_lut[(-zq)]                               # gather Q1.20
    e = np.where(finite, e, 0)
    s = int(e.sum())                                 # 29-bit
    r = (1 << RECIP_R) // s                          # floor reciprocal, 21-bit
    prob = (e.astype(np.int64) * r) >> (RECIP_R - PROB_FRAC)
    return prob.astype(np.int64)


def int_softmax_float(s_row, exp_lut):
    """Convenience: float score row -> float probs (Q1.20 scaled back). Used as the
    model's _smax hook for the token-stream gate. Bit-true to RTL on each valid row."""
    s_row = np.asarray(s_row, dtype=np.float64)
    sq, finite = quantize_scores(s_row)
    prob = int_softmax_q(sq, finite, exp_lut)
    return prob.astype(np.float64) / (1 << PROB_FRAC)


# --- profile real score rows for the module gate ------------------------------
def _profile_score_rows(npz, n_rows, seed):
    """Collect real causal attention score rows from the exported model (the exact
    distribution the fabric softmax sees), plus some random rows for edge coverage."""
    from model.goformer_full import GoformerFull, load_params, layernorm
    p = load_params(npz)
    gf = GoformerFull(p)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, p["tok_emb"].shape[0], size=64)
    T = len(idx)
    nh = p["n_head"]
    x = p["tok_emb"][idx] + p["pos_emb"][np.arange(T)]
    rows = []
    for blk in p["blocks"]:
        h = layernorm(x, blk["ln1"])
        C = h.shape[1]
        hd = C // nh
        qkv = gf.qlin(h, blk["qkv"])
        q, k = qkv[:, :C], qkv[:, C:2 * C]
        q = q.reshape(T, nh, hd).transpose(1, 0, 2)
        k = k.reshape(T, nh, hd).transpose(1, 0, 2)
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)   # (nh, T, T)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        for hi in range(nh):
            for ti in range(T):                              # causal: row ti uses 0..ti
                row = np.where(mask[ti], -np.inf, scores[hi, ti])
                rows.append(row[: ti + 1].copy())            # length ti+1 valid entries
        x = x + gf._attn(h, blk)
        x = x + gf._mlp(layernorm(x, blk["ln2"]), blk)
    # subsample to n_rows, but always keep a length-1 and a length-256 row
    rng2 = np.random.default_rng(seed + 1)
    pick = list(rng2.choice(len(rows), size=min(n_rows, len(rows)), replace=False))
    rows = [rows[i] for i in pick]
    return rows


def run(sim_dir, n_rows=200, seed=0, npz="fabric/export/goformer.npz"):
    os.makedirs(sim_dir, exist_ok=True)
    exp_lut = exp_table()

    # build the row set: real model rows (varied lengths) + a few synthetic edge rows
    rows = _profile_score_rows(npz, n_rows, seed)
    rng = np.random.default_rng(seed + 99)
    # synthetic edges: length 1, length 256 flat, extreme spread, all-equal
    rows.append(np.array([0.0]))                                   # T=1
    rows.append(np.zeros(256))                                     # all equal, full width
    rows.append(np.linspace(-13.0, 13.0, 256))                    # full spread
    rows.append(rng.uniform(-14.0, 14.0, size=128))               # random
    rows.append(np.array([13.8, -10.4, 0.0, 5.5]))               # mixed

    # quantize every row; build the bit-true gold and the concatenated mem files
    sq_rows, fin_rows, gold_rows, tlens = [], [], [], []
    score_words, gold_words = [], []
    for row in rows:
        sq, fin = quantize_scores(row)
        # the RTL only ever sees the VALID (causal) prefix; drop masked entries here
        valid = fin
        sq_v = sq[valid]
        gold = int_softmax_q(sq_v, np.ones_like(sq_v, dtype=bool), exp_lut)
        sq_rows.append(sq_v); gold_rows.append(gold); tlens.append(len(sq_v))
        for v in sq_v:
            score_words.append(int(v) & 0xFFFF)
        for g in gold:
            gold_words.append(int(g) & 0x1FFFFF)

    nrows = len(tlens)
    nscores = len(score_words)

    with open(os.path.join(sim_dir, "exp_lut.mem"), "w") as fh:
        fh.write("\n".join(f"{int(v) & 0x1FFFFF:06x}" for v in exp_lut) + "\n")
    with open(os.path.join(sim_dir, "score.mem"), "w") as fh:
        fh.write("\n".join(f"{w:04x}" for w in score_words) + "\n")
    with open(os.path.join(sim_dir, "tlen.mem"), "w") as fh:
        fh.write("\n".join(f"{t:03x}" for t in tlens) + "\n")

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(
        ["iverilog", "-g2012", "-o", vvp,
         f"-DNROWS={nrows}", f"-DNSCORES={nscores}",
         os.path.join(HERE, "tb", "tb_softmax.sv"),
         os.path.join(HERE, "rtl", "softmax.sv")],
        capture_output=True, text=True)
    if cp.returncode != 0:
        print("IVERILOG_COMPILE_FAIL")
        print(cp.stdout, cp.stderr)
        return False
    rp = subprocess.run(["vvp", "sim.vvp"], cwd=sim_dir, capture_output=True, text=True)
    if "TB_DONE" not in rp.stdout:
        print("VVP_FAIL")
        print(rp.stdout[-2000:], rp.stderr[-2000:])
        return False

    # parse prob.out, split on ROW markers
    rtl_rows = []
    cur = []
    with open(os.path.join(sim_dir, "prob.out")) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line == "ROW":
                rtl_rows.append(cur)
                cur = []
            else:
                try:
                    cur.append(int(line, 16) & 0x1FFFFF)
                except ValueError:
                    cur.append(-1)              # X -> never matches
    if cur:
        rtl_rows.append(cur)

    # compare row by row
    total = 0
    mism = 0
    bad_rows = 0
    for gi, (gold, got) in enumerate(zip(gold_rows, rtl_rows)):
        total += len(gold)
        if len(got) != len(gold):
            mism += abs(len(got) - len(gold)) + min(len(got), len(gold))
            bad_rows += 1
            continue
        rm = sum(1 for a, b in zip(gold, got) if a != b)
        mism += rm
        if rm:
            bad_rows += 1
    if len(rtl_rows) != len(gold_rows):
        print(f"ROW_COUNT_MISMATCH rtl={len(rtl_rows)} gold={len(gold_rows)}")

    ok = (mism == 0) and (len(rtl_rows) == len(gold_rows))

    # token-stream gate (the binding one)
    stream_ok, stream_detail = _token_stream_gate(npz, exp_lut)

    print(f"SOFTMAX_VERDICT bitexact={ok} mismatches={mism}/{total} bad_rows={bad_rows} "
          f"nrows={nrows} token_stream_identical={stream_ok} {stream_detail}")
    return ok and stream_ok


def _token_stream_gate(npz, exp_lut, n_gen=60, prompt_len=8, seed=0):
    """Plug int_softmax into goformer; greedy decode must equal the float reference."""
    from model.goformer_fixed import FixedConfig, FixedGoformer, _profile_gelu_range
    from model.goformer_full import GoformerFull, load_params

    p = load_params(npz)

    class IntSmaxGoformer(FixedGoformer):
        def _smax(self, s):
            # s: (..., T, T) with -inf on causal-masked entries; apply per last axis row
            out = np.empty_like(s, dtype=np.float64)
            flat = s.reshape(-1, s.shape[-1])
            ofl = out.reshape(-1, s.shape[-1])
            for r in range(flat.shape[0]):
                ofl[r] = int_softmax_float(flat[r], exp_lut)
            return out

    idx = np.random.default_rng(7).integers(0, p["tok_emb"].shape[0], size=64)
    dom = float(np.ceil(_profile_gelu_range(p, idx) + 1))
    cfg = FixedConfig(gelu_lo=-dom, gelu_hi=dom)

    float_engine = GoformerFull(p)
    int_engine = IntSmaxGoformer(p, cfg)
    rng = np.random.default_rng(seed)
    prompt = list(rng.integers(0, p["tok_emb"].shape[0], size=prompt_len))

    def greedy(engine, idx0, n):
        out = list(idx0)
        for _ in range(n):
            out.append(int(np.argmax(engine.forward(out[-256:])[-1])))
        return out

    ref = greedy(float_engine, prompt, n_gen)
    got = greedy(int_engine, prompt, n_gen)
    nm = sum(1 for a, b in zip(ref, got) if a == b)
    same = ref == got
    detail = f"stream_match={nm}/{len(ref)}"
    return same, detail


def main():
    sim_dir = os.path.join("C:\\kevbuild", "stage3_softmax")
    raise SystemExit(0 if run(sim_dir) else 1)


if __name__ == "__main__":
    main()
