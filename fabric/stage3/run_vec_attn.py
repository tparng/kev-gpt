"""vec_attn RTL gate: single-head P-wide attention, bit-true vs the integer reference.

    python -m fabric.stage3.run_vec_attn

Op (one attention head, HEAD_DIM=64), mirroring seq_ref._attn_step exactly:
  score[j] = sat( rsh_round( sum_d q[d]*k_j[d], 27 ), -32768, 32767 )   Q8.8
  probs    = int_softmax_q(scores, all-valid, exp_lut)                  Q1.20
  ctx[d]   = rsh_round( sum_j prob[j]*v_j[d], 11 )                      Q6.25

Random q/k/v in plausible Q.16 ranges for T in {1, 8, 16, 32}. Drives rtl/vec_attn.sv
(which instantiates rtl/softmax.sv) and compares the captured ctx[0..63] per case.
Prints VEC_ATTN_VERDICT. Scratch under C:/kevbuild (not the repo / OneDrive).
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

from fabric.stage3._simdir import kevbuild

from fabric.stage3.run_softmax import exp_table, int_softmax_q

VFRAC = 16
ISQRT = 3
SCORE_FRAC = 8
PROB_FRAC = 20
RESID_FRAC = 25
HEAD_DIM = 64
TMAX = 32
P = 8
SCORE_SH = 2 * VFRAC + ISQRT - SCORE_FRAC   # 27
CTX_SH = PROB_FRAC + VFRAC - RESID_FRAC     # 11

HERE = os.path.dirname(os.path.abspath(__file__))


def q_round_div(num, den):
    if num >= 0:
        return (num + den // 2) // den
    return -(((-num) + den // 2) // den)


def rsh_round(v, s):
    if s <= 0:
        return int(v) << (-s)
    return q_round_div(int(v), 1 << s)


def sat(v, lo, hi):
    return max(lo, min(hi, int(v)))


def head_ref(q, k_cache, v_cache, exp_lut):
    """q: (64,) ; k_cache,v_cache: (T,64) int Q.16. Returns ctx (64,) int Q6.25."""
    T = len(k_cache)
    sq = np.zeros(T, dtype=np.int64)
    for j in range(T):
        acc = 0
        for d in range(HEAD_DIM):
            acc += int(q[d]) * int(k_cache[j][d])
        s_q88 = rsh_round(acc, SCORE_SH)
        sq[j] = sat(s_q88, -32768, 32767)
    prob = int_softmax_q(sq, np.ones(T, dtype=bool), exp_lut)
    ctx = [0] * HEAD_DIM
    for d in range(HEAD_DIM):
        acc = 0
        for j in range(T):
            acc += int(prob[j]) * int(v_cache[j][d])
        ctx[d] = rsh_round(acc, CTX_SH)
    return ctx


def _u32(v):
    return int(v) & 0xFFFFFFFF


def run(sim_dir, seed=0):
    os.makedirs(sim_dir, exist_ok=True)
    exp_lut = exp_table()
    rng = np.random.default_rng(seed)

    Ts = [1, 8, 16, 32]
    cases = []          # (q, k_cache, v_cache, ctx_ref)
    for T in Ts:
        # a few cases per T with varied magnitudes (Q.16: +-2^15..2^17)
        for mag in (1 << 15, 1 << 16, 1 << 17):
            q = rng.integers(-mag, mag + 1, size=HEAD_DIM).astype(np.int64)
            k_cache = rng.integers(-mag, mag + 1, size=(T, HEAD_DIM)).astype(np.int64)
            v_cache = rng.integers(-mag, mag + 1, size=(T, HEAD_DIM)).astype(np.int64)
            ctx = head_ref(q, k_cache, v_cache, exp_lut)
            cases.append((T, q, k_cache, v_cache, ctx))

    ncase = len(cases)

    # write mem files (signed -> 32-bit / 21-bit hex two's complement)
    q_words, k_words, v_words, tlens = [], [], [], []
    for (T, q, k_cache, v_cache, _ctx) in cases:
        tlens.append(T)
        for d in range(HEAD_DIM):
            q_words.append(_u32(q[d]))
        for j in range(T):
            for d in range(HEAD_DIM):
                k_words.append(_u32(k_cache[j][d]))
        for j in range(T):
            for d in range(HEAD_DIM):
                v_words.append(_u32(v_cache[j][d]))

    qwords = len(q_words)
    kvwords = len(k_words)

    with open(os.path.join(sim_dir, "exp_lut.mem"), "w") as fh:
        fh.write("\n".join(f"{int(v) & 0x1FFFFF:06x}" for v in exp_lut) + "\n")
    with open(os.path.join(sim_dir, "q.mem"), "w") as fh:
        fh.write("\n".join(f"{w:08x}" for w in q_words) + "\n")
    with open(os.path.join(sim_dir, "k.mem"), "w") as fh:
        fh.write("\n".join(f"{w:08x}" for w in k_words) + "\n")
    with open(os.path.join(sim_dir, "v.mem"), "w") as fh:
        fh.write("\n".join(f"{w:08x}" for w in v_words) + "\n")
    with open(os.path.join(sim_dir, "tlen.mem"), "w") as fh:
        fh.write("\n".join(f"{t:03x}" for t in tlens) + "\n")

    vvp = os.path.join(sim_dir, "sim.vvp")
    cp = subprocess.run(
        ["iverilog", "-g2012", "-o", vvp,
         f"-DNCASE={ncase}", f"-DQWORDS={qwords}", f"-DKVWORDS={kvwords}",
         f"-DPVAL={P}",
         os.path.join(HERE, "tb", "tb_vec_attn.sv"),
         os.path.join(HERE, "rtl", "vec_attn.sv"),
         os.path.join(HERE, "rtl", "softmax_f.sv")],
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

    # parse ctx.out split on ROW
    rtl_rows = []
    cur = []
    with open(os.path.join(sim_dir, "ctx.out")) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line == "ROW":
                rtl_rows.append(cur)
                cur = []
            else:
                try:
                    val = int(line, 16)
                    if val >= (1 << 31):
                        val -= (1 << 32)
                    cur.append(val)
                except ValueError:
                    cur.append(None)
    if cur:
        rtl_rows.append(cur)

    total = 0
    mism = 0
    bad_cases = 0
    if len(rtl_rows) != ncase:
        print(f"ROW_COUNT_MISMATCH rtl={len(rtl_rows)} cases={ncase}")
    for ci, (T, _q, _k, _v, ctx_ref) in enumerate(cases):
        total += HEAD_DIM
        if ci >= len(rtl_rows) or len(rtl_rows[ci]) != HEAD_DIM:
            mism += HEAD_DIM
            bad_cases += 1
            continue
        got = rtl_rows[ci]
        rm = sum(1 for a, b in zip(ctx_ref, got) if int(a) != int(b))
        if rm:
            bad_cases += 1
            # show first few diffs for the first bad case
            if bad_cases == 1:
                diffs = [(d, int(ctx_ref[d]), got[d])
                         for d in range(HEAD_DIM) if int(ctx_ref[d]) != got[d]][:6]
                print(f"  case {ci} T={T} first diffs (d, ref, rtl): {diffs}")
        mism += rm

    ok = (mism == 0) and (len(rtl_rows) == ncase)
    print(f"VEC_ATTN_VERDICT bitexact={ok} mismatches={mism}/{total} "
          f"bad_cases={bad_cases} ncase={ncase} T={sorted(set(Ts))} P={P}")
    return ok


def main():
    sim_dir = kevbuild("stage3_vec_attn")
    raise SystemExit(0 if run(sim_dir) else 1)


if __name__ == "__main__":
    main()
