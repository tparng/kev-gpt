"""AQ multiplier range proof — narrow the GE_AQ per-lane multiply in cohort_engine.sv.

The GE_AQ phase computes, per lane gp:

    aq_prod_r[gp] <= $signed(lnt_r[gp]) * $signed(inv_sact[a_asel]);   // 64x64 -> 96
    aq_sh = round( aq_prod >>> (LN_OUT_FRAC + ISH) )                   // = >>> 62
    aq_int = clip(aq_sh, -128, 127)                                    // INT8 act

`lnt_r` is the per-lane activation in Q.LN_OUT_FRAC (=Q.22) coming from FOUR sources
(d_asrc in cohort_engine.sv):

  d_asrc=0  lnout1_r       Q.22 LN1 output (32b sign-extended to 64)        [also lnf + head]
  d_asrc=1  ctxv_g >>>3    residual ctx Q6.25 -> Q.22 (rounded)             [proj act]
  d_asrc=2  lnout2_r       Q.22 LN2 output                                  [mlp_fc act]
  d_asrc=3  mlpbuf_r <<<10 16b GELU Q4.12 -> Q.22                           [mlp_proj act]

`inv_sact[a_asel]` = round(2^ISH / s_act), ISH=40, declared signed [63:0], loaded from
inv_sact.mem.

GOAL: prove the true signed bit-widths W1 (lnt) and W2 (inv_sact) so the synthesised
multiplier shrinks from 64x64 (~88 DSP/cohort) to something far smaller, WITHOUT changing
the rounding/shift/clip semantics (bit-identical aqw bytes), so cycle counts are unchanged.

PROVEN (by construction) bounds are separated from OBSERVED (empirical) bounds. The RTL
narrowing only relies on PROVEN bounds (with margin); the empirical scan is a sanity
cross-check that the proven bounds are not absurdly loose AND a guard that nothing observed
ever exceeds them.

    python -m fabric.stage3.research.aq_range_proof
"""

from __future__ import annotations

import os

import numpy as np

from fabric.stage3 import seq_ref
from fabric.stage3.seq_ref import IntSequencer, RESID_FRAC
from fabric.stage3.run_layernorm import OUT_FRAC as LN_OUT_FRAC  # 22

ISH = 40
NSACT = 17  # 4 GEMV/block * 4 blocks + head

# npz lives outside the worktree (gitignored); default to the main checkout.
DEFAULT_NPZ = os.environ.get(
    "GOFORMER_NPZ",
    r"C:\Users\mikea\OneDrive\Desktop\Projects\kev-gpt\fabric\export\goformer.npz")


def signed_bits(mag: int) -> int:
    """Min signed two's-complement width that holds values in [-mag, +mag]."""
    if mag <= 0:
        return 1
    n = 1
    while (1 << (n - 1)) - 1 < mag:
        n += 1
    return n


# ---------------------------------------------------------------------------
#  PROVEN bounds (by construction)
# ---------------------------------------------------------------------------
def proven_inv_sact_bound(inv_vals):
    """inv_sact = round(2^40 / s_act). The values are a fixed, committed ROM image — the
    'by construction' bound here is the ACTUAL ROM content (these are constants baked into
    the .mem at build time, not runtime-variable). So the proven bound for inv_sact IS the
    exact ROM max. Returns (max_abs, signed_width)."""
    mx = max(abs(int(v)) for v in inv_vals)
    return mx, signed_bits(mx)


def proven_lnt_bounds():
    """Proven-by-construction magnitude bounds for each lnt source, in Q.22 integer units.

    Source-by-source construction bounds:

    d_asrc=3 (GELU): mlpbuf_r is a signed 16-bit value (Q4.12, saturated to [-32768,32767]
        in seq_ref._mlp_step via sat(.,-32768,32767)). Shifted <<<10 to Q.22. So
        |lnt| <= 32768 * 2^10 = 2^25. PROVEN width = 26 bits.  (HARD: 16b sat is explicit.)

    d_asrc=0,2 (LN1/LN2/lnf outputs, Q.22): y = ((d*rsqrt)*gamma) >>> sh. The value is
        (normalized deviation) * gamma. The normalized deviation n_i = d_i / sqrt(var+eps)
        is mathematically bounded by |n_i| < sqrt(D) = 16 (a single coordinate of a unit-RMS
        vector cannot exceed sqrt(D); the +EPS regularizer makes it strictly smaller). gamma
        is a learned Q4.20 scale whose by-construction container bound is |gamma| < 8 (4
        integer bits). So |y_real| < 16 * 8 = 128, i.e. |lnt| < 128 * 2^22 = 2^29.
        PROVEN width = 30 bits.

    d_asrc=1 (ctx >>>3): ctx_real = sum_j prob_j * v_j is a CONVEX combination of the v_j
        (softmax weights sum to 1), hence |ctx_real| <= max_j |v_j_real|. v_j is the dequant
        of the qkv GEMV stored Q.16; there is no clean *container* magnitude bound on v
        without bounding the dequant, so the by-construction bound for source 1 is reported
        as 'convex-combo of v' and we take the OBSERVED bound x4 for the width, flagged NOT
        fully by-construction (RTL must assert/clamp — see report).

    Returns dict source -> (proven_q22_mag or None, width or None, note)."""
    g3 = 1 << 25                       # GELU: 16b sat <<<10
    w3 = signed_bits(g3)
    g02 = 128 * (1 << LN_OUT_FRAC)     # LN: |n|<sqrt(256)=16, |gamma|<8 -> |y|<128
    w02 = signed_bits(g02)
    return {
        0: (g02, w02, "LN1/lnf: |n|<sqrt(D)=16 * |gamma|<8 -> |lnt|<2^29"),
        2: (g02, w02, "LN2: same as source 0"),
        3: (g3, w3, "GELU: 16b sat <<<10 -> |lnt|<=2^25 (HARD)"),
        1: (None, None, "ctx>>>3: convex combo of v; no container bound -> OBSERVED x4"),
    }


# ---------------------------------------------------------------------------
#  OBSERVED bounds (empirical scan over a spread of tokens / KV positions)
# ---------------------------------------------------------------------------
def main(npz=None):
    npz = npz or DEFAULT_NPZ
    p, cfg = seq_ref.build(npz)
    iseq = IntSequencer(p, cfg)

    # inv_sact ROM image (the actual committed constants)
    inv_vals = []
    for bi in range(iseq.nblocks):
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            inv_vals.append(iseq._inv_sact(iseq.layers[(bi, nm)]["s_act"]))
    inv_vals.append(iseq._inv_sact(iseq.head["s_act"]))
    inv_mx, inv_w = proven_inv_sact_bound(inv_vals)

    print("=" * 78)
    print("inv_sact ROM (PROVEN: committed constants in inv_sact.mem)")
    print("=" * 78)
    for i, v in enumerate(inv_vals):
        print(f"  inv_sact[{i:2d}] = {int(v):>22d}  ({signed_bits(abs(int(v)))}b signed)")
    print(f"  MAX |inv_sact| = {inv_mx}  -> PROVEN signed width W2 = {inv_w} bits")
    print()

    # empirical scan: a spread of token ids generated greedily so KV depth (pos) varies.
    rng = np.random.default_rng(123)
    seed_toks = [48, 10, 100, 77, 5, 60, 120, 180, 33, 7, 150, 90, 14, 66, 111, 173, 1, 192, 95]
    extra = list(rng.integers(0, p["tok_emb"].shape[0], size=40))
    scan_toks = seed_toks + [int(t) for t in extra]

    obs = {0: 0, 1: 0, 2: 0, 3: 0}
    iseq.reset()
    cur = scan_toks[0]
    for _ in scan_toks:
        # capture lnt at this exact KV state (pos = iseq.t); save caches so the capturing
        # forward (which appends to k/v) does not corrupt the canonical decode.
        save_kv = ([list(c) for c in iseq.k_cache], [list(v) for v in iseq.v_cache], iseq.t)
        x = iseq.tok_emb_q[cur] + iseq.pos_emb_q[iseq.t]
        x = np.asarray([int(v) for v in x], dtype=object)
        for bi in range(iseq.nblocks):
            s = {}
            attn_out = iseq._attn_step(x, bi, sink=s)
            obs[0] = max(obs[0], max(abs(int(v)) for v in s["ln1_out_q22"]))
            ctx_q22 = [seq_ref.rsh_round(int(c), RESID_FRAC - LN_OUT_FRAC) for c in s["ctx_q25"]]
            obs[1] = max(obs[1], max(abs(int(v)) for v in ctx_q22))
            x = x + np.asarray([int(v) for v in attn_out], dtype=object)
            s2 = {}
            mlp_out = iseq._mlp_step(x, bi, sink=s2)
            obs[2] = max(obs[2], max(abs(int(v)) for v in s2["ln2_out_q22"]))
            obs[3] = max(obs[3], max(abs(int(v)) for v in s2["gelu_q22"]))
            x = x + np.asarray([int(v) for v in mlp_out], dtype=object)
        y_q22 = iseq._ln(x, iseq.ln_f_q)
        obs[0] = max(obs[0], max(abs(int(v)) for v in y_q22))
        # restore caches, then do a clean decode step to advance KV by exactly one position.
        iseq.k_cache, iseq.v_cache, iseq.t = save_kv
        _, nxt = iseq.step(int(cur))
        cur = nxt

    proven = proven_lnt_bounds()
    print("=" * 78)
    print("lnt (Q.22) per source: PROVEN (by construction) vs OBSERVED (empirical)")
    print("=" * 78)
    print(f"{'src':>3} {'name':<10} {'OBS |lnt|':>14} {'OBS w':>6} "
          f"{'PROVEN |lnt|':>14} {'PRV w':>6}  note")
    names = {0: "LN1/lnf", 1: "ctx>>>3", 2: "LN2", 3: "GELU"}
    src_widths = {}
    for src in (0, 1, 2, 3):
        ow = signed_bits(obs[src])
        pmag, pw, note = proven[src]
        pstr = f"{pmag:>14d}" if pmag is not None else f"{'(none)':>14}"
        pwstr = f"{pw:>6d}" if pw is not None else f"{'--':>6}"
        print(f"{src:>3} {names[src]:<10} {obs[src]:>14d} {ow:>6d} {pstr} {pwstr}  {note}")
        if pw is not None:
            src_widths[src] = pw
        else:
            margin_mag = obs[src] * 4
            src_widths[src] = signed_bits(margin_mag)
            print(f"      ctx fallback: OBSERVED x4 = {margin_mag} -> {src_widths[src]}b "
                  f"(NOT by-construction; RTL must clamp/assert)")
    print()

    # The RTL uses ONE lnt register width for all four sources (a single multiplier).
    w1 = max(src_widths.values())
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  Per-source chosen widths: {src_widths}")
    print(f"  W1 (lnt signed width, max over sources) = {w1} bits")
    print(f"  W2 (inv_sact signed width)              = {inv_w} bits")
    print(f"  Multiply shrinks: 64 x 64  ->  {w1} x {inv_w}")
    for src in (0, 1, 2, 3):
        cap = (1 << (src_widths[src] - 1)) - 1
        assert obs[src] <= cap, f"src{src} OBSERVED {obs[src]} exceeds width cap {cap}!"
    for src in (0, 2, 3):
        pmag, _, _ = proven[src]
        assert obs[src] <= pmag, f"src{src} OBSERVED {obs[src]} > PROVEN {pmag}!"
    print("  ASSERT_OK: observed within chosen widths; by-construction bounds hold.")
    print(f"AQ_RANGE_PROOF W1={w1} W2={inv_w} INVMAX={inv_mx}")
    return w1, inv_w


if __name__ == "__main__":
    main()
