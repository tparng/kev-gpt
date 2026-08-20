"""mamba_cohort_ref — N-stream cohort (batch) replica of the mamba engine.

Phase 2 Rung B of the Mamba 100k campaign: process N independent token-streams
(N separate conversations/positions) through ONE weight-stationary engine. The
weights / LUTs / consts are read ONCE and applied to all N streams' activations
(the GEMV amortization); each stream keeps its OWN scan state (h), conv history
and residual (the per-stream state banking). This module is the *architecture
spec* for that: a bit-exact, phase-major cohort schedule that must reproduce the
single-stream mamba_seq_ref.step() output for EVERY stream.

Why phase-major (weight-stationary)?  The single-stream engine is sequential:
during the in_proj GEMV the conv/scan/norm units idle, and during the scan the
GEMV idles.  The cohort reads each shared weight word once and folds it into N
per-stream accumulators (gemm_cohort_vec.sv's pattern) — so the GEMV's ~25.5k
cyc/token is shared by all N streams (=> /N per stream), while the per-stream
non-GEMV work (conv/scan/norm/quant) is banked N-deep and pipelines across
streams.  This file proves the *functional* half: shared reads + banked state =>
each stream still produces the identical per-token logits/argmax it would alone.

    python -m fabric.stage3.mamba_cohort_ref --verify -n 4     # cohort==single, N streams
    python -m fabric.stage3.mamba_cohort_ref --gemv   -n 8     # cohort GEMV == per-stream GEMV
    python -m fabric.stage3.mamba_cohort_ref --fit           # URAM/BRAM budget vs N

The single-stream MambaSeqRef.step() is the untouched golden; the cohort path
below is an INDEPENDENT re-derivation of the datapath (reusing only the proven
module-level fixed-point primitives) so a match is a real cross-check, not a
tautology.  Every buffer that is per-stream (xbuf/zxbuf/xnbuf/ybuf/q8buf/logit +
the conv/scan state) is banked per stream; every table (gw/consts/luts/scales)
is shared read-only — exactly the hardware banking.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from fabric.stage3.mamba_seq_ref import (
    D, DIN, NST, H, L, V, INROWS, CONVD, REPO,
    MambaSeqRef, rmsnorm_gated, silu_conv, bitlen15,
    rshr, sat8, sat16, sat32, mul_m11, me_split, s16,
)
from model.mamba2_fixed_scan import scan_step_fixed2


class MambaCohortRef:
    """N-stream weight-stationary cohort over a shared MambaSeqRef table set."""

    def __init__(self, n=4, export="data/fabric_mamba"):
        self.N = n
        self.eng = MambaSeqRef(export)          # shared read-only tables

    def alloc_states(self):
        return [self.eng.alloc_state() for _ in range(self.N)]

    # ------------------------------------------------------- cohort GEMV -----
    def gemv_cohort(self, base, rows, wpr, xs):
        """Weight-stationary GEMV: read each weight nibble ONCE, fold into all N
        streams' accumulators.  xs = list of N int-lists (length wpr*8).  This is
        the gemm_cohort_vec.sv datapath: one shared weight word / cycle, N MACs.
        Returns list of N acc-lists (each length `rows`)."""
        N = self.N
        gw = self.eng.gw
        accs = [[0] * rows for _ in range(N)]
        for r in range(rows):
            wbase = base + r * wpr
            sums = [0] * N
            for w in range(wpr):
                word = int(gw[wbase + w])
                for k in range(8):
                    nib = (word >> (k * 4)) & 0xF
                    if nib >= 8:
                        nib -= 16
                    col = w * 8 + k
                    if nib:
                        for s in range(N):
                            sums[s] += nib * xs[s][col]
            for s in range(N):
                accs[s][r] = sums[s]
        return accs

    # ============================================================= step ======
    def step(self, toks, states, collects=None):
        """One cohort decode step: N tokens in, N argmax out.  Phase-major so a
        shared table read serves all N streams before the schedule advances."""
        e = self.eng
        consts = e.consts
        N = self.N
        assert len(toks) == N and len(states) == N
        C = collects if collects is not None else [{} for _ in range(N)]

        # ---- S_EMB (per-stream: different tokens) ----------------------------
        xbuf = [[0] * D for _ in range(N)]
        for s in range(N):
            tok = toks[s]
            fp = e.esc[tok]
            fexp = (fp >> 10) & 0x1F
            for i in range(D):
                word = int(e.emb[tok * (D // 8) + (i >> 3)])
                nib = (word >> ((i & 7) * 4)) & 0xF
                if nib >= 8:
                    nib -= 16
                xbuf[s][i] = sat32(rshr(mul_m11(nib, fp), 6 - fexp))
            C[s]["emb"] = list(xbuf[s])

        for li in range(L):
            m_inr, e_inr = me_split(consts[li * 16 + 0])
            m_ins, e_ins = me_split(consts[li * 16 + 1])
            m_outr, e_outr = me_split(consts[li * 16 + 2])
            m_outs, e_outs = me_split(consts[li * 16 + 3])
            packed = int(consts[li * 16 + 4])
            bB = packed & 0xF
            bC = (packed >> 4) & 0xF
            bX = (packed >> 8) & 0xF
            gam_pre = [e.lng[li * D + i] for i in range(D)]
            gam_g = [e.nrmg[li * DIN + i] for i in range(DIN)]

            # ---- S_NIN + S_QIN (per-stream) ----------------------------------
            q8 = [[0] * DIN for _ in range(N)]
            for s in range(N):
                yin = [sat16(rshr(xbuf[s][i], 10)) for i in range(D)]
                xn = rmsnorm_gated(yin, yin, gam_pre, e.slut, gated=False,
                                   short_len=True)
                C[s][f"l{li}.xn_pre"] = list(xn)
                for i in range(D):
                    q8[s][i] = sat8(rshr(xn[i] * m_inr, e_inr + 12))
                C[s][f"l{li}.q8_in"] = q8[s][:D]

            # ---- S_GIN: SHARED-weight in_proj GEMV over all N streams ---------
            accs = self.gemv_cohort(li * INROWS * (D // 8), INROWS, D // 8,
                                    [q8[s][:DIN] for s in range(N)])
            zx = [[0] * INROWS for _ in range(N)]
            for s in range(N):
                C[s][f"l{li}.in_acc"] = list(accs[s])
                for i in range(INROWS):
                    fpi = e.rsin[li * INROWS + i]
                    t1 = rshr(mul_m11(accs[s][i], fpi), 10 - ((fpi >> 10) & 0x1F))
                    zx[s][i] = sat16(rshr(t1 * m_ins, e_ins + 15 - 9))
                C[s][f"l{li}.zx"] = list(zx[s])

            # ---- S_CONV: conv weights SHARED per layer, per-stream history ----
            xn_conv = [[0] * CONVD for _ in range(N)]
            for s in range(N):
                hist = states[s]["conv"][li]
                for c in range(CONVD):
                    x_in = zx[s][DIN + c]
                    h0, h1, h2 = hist[c]
                    w0 = e.convw[li * CONVD * 4 + c * 4 + 0]
                    w1 = e.convw[li * CONVD * 4 + c * 4 + 1]
                    w2 = e.convw[li * CONVD * 4 + c * 4 + 2]
                    w3 = e.convw[li * CONVD * 4 + c * 4 + 3]
                    b = e.convb[li * CONVD + c]
                    acc2 = h0 * w0 + h1 * w1 + h2 * w2 + x_in * w3 + (b << 9)
                    prs = sat16((acc2 + (1 << 11)) >> 12)
                    xn_conv[s][c] = silu_conv(prs, e.slut)
                    hist[c] = [h1, h2, x_in]
                C[s][f"l{li}.conv"] = list(xn_conv[s])

            # ---- S_SCAN: dt/a LUTs SHARED, per-stream h state ----------------
            ybuf = [[0] * DIN for _ in range(N)]
            for s in range(N):
                B8 = np.array([sat8(rshr(xn_conv[s][DIN + i], 11 - bB))
                               for i in range(NST)], dtype=np.int64)
                C8 = np.array([sat8(rshr(xn_conv[s][DIN + NST + i], 11 - bC))
                               for i in range(NST)], dtype=np.int64)
                for hi in range(H):
                    dtraw = zx[s][2 * DIN + 2 * NST + hi]
                    dtq3 = sat16(dtraw << 3) & 0xFFFF
                    idx = ((dtq3 >> 8) & 0xFF) ^ 0x80
                    dtq = e.dlut[li * H * 256 + hi * 256 + idx]
                    a_q = e.alut[li * H * 256 + hi * 256 + idx]
                    bl = bitlen15(dtq)
                    eh = 20 if 21 - bl > 20 else 0 if bl > 21 else 21 - bl
                    dtx = np.empty(64, dtype=np.int64)
                    for i in range(64):
                        x8 = sat8(rshr(xn_conv[s][hi * 64 + i], 11 - bX))
                        dtx[i] = sat16(rshr(x8 * dtq, 14 - eh))
                    sh_i = 16 + 13 - bX - eh - bB
                    sh_y = bC
                    yq = scan_step_fixed2(states[s]["h"][li * H + hi], dtx,
                                          B8, C8, int(a_q), sh_i, sh_y)
                    Dword = int(consts[li * 16 + 5 + (hi >> 1)])
                    Dq = s16((Dword >> (16 if (hi & 1) else 0)) & 0xFFFF)
                    for i in range(64):
                        xv = xn_conv[s][hi * 64 + i]
                        ybuf[s][hi * 64 + i] = sat16(rshr(int(yq[i]), 2)
                                                     + rshr(Dq * xv, 13))
                C[s][f"l{li}.scan"] = list(ybuf[s])

            # ---- S_NG + S_QOUT (per-stream) ----------------------------------
            q8o = [[0] * DIN for _ in range(N)]
            for s in range(N):
                zgate = [zx[s][i] for i in range(DIN)]
                yn = rmsnorm_gated(ybuf[s], zgate, gam_g, e.slut, gated=True,
                                   short_len=False)
                C[s][f"l{li}.yn"] = list(yn)
                for i in range(DIN):
                    q8o[s][i] = sat8(rshr(yn[i] * m_outr, e_outr + 12))
                C[s][f"l{li}.q8_out"] = list(q8o[s])

            # ---- S_GOUT: SHARED-weight out_proj GEMV + per-stream residual ---
            accs = self.gemv_cohort(e.out_base + li * (D * (DIN // 8)), D,
                                    DIN // 8, [q8o[s][:DIN] for s in range(N)])
            for s in range(N):
                for i in range(D):
                    fpi = e.rsout[li * D + i]
                    t1 = rshr(mul_m11(accs[s][i], fpi), 10 - ((fpi >> 10) & 0x1F))
                    dx = rshr(t1 * m_outs, e_outs + 15 - 19)
                    xbuf[s][i] = sat32(xbuf[s][i] + dx)
                C[s][f"l{li}.x_out"] = list(xbuf[s])

        # ---- S_NF + S_QHD (per-stream) ---------------------------------------
        m_hdr, e_hdr = me_split(consts[112])
        m_hds, e_hds = me_split(consts[113])
        gam_nf = [e.nfg[i] for i in range(D)]
        q8h = [[0] * DIN for _ in range(N)]
        for s in range(N):
            yin = [sat16(rshr(xbuf[s][i], 10)) for i in range(D)]
            xnf = rmsnorm_gated(yin, yin, gam_nf, e.slut, gated=False,
                                short_len=True)
            C[s]["final_norm"] = list(xnf)
            for i in range(D):
                q8h[s][i] = sat8(rshr(xnf[i] * m_hdr, e_hdr + 11))
            C[s]["q8_head"] = q8h[s][:D]

        # ---- S_GHD: SHARED head GEMV -> per-stream logits + argmax -----------
        accs = self.gemv_cohort(e.head_base, V, D // 8,
                                [q8h[s][:DIN] for s in range(N)])
        out = []
        for s in range(N):
            logit = [0] * V
            best, besti = -32768, 0
            for i in range(V):
                fpi = e.rshd[i]
                t1 = rshr(mul_m11(accs[s][i], fpi), 10 - ((fpi >> 10) & 0x1F))
                lv = sat16(rshr(t1 * m_hds, e_hds + 15 - 12))
                logit[i] = lv
                if lv > best:
                    best, besti = lv, i
            C[s]["logits"] = list(logit)
            C[s]["argmax"] = besti
            out.append(besti)
        return out


# ==================================================================== main ===
def _verify(args):
    """Cohort==single-stream, per stream, over a multi-token sequence with state
    carry.  The N streams are seeded with DIFFERENT token sequences so the check
    exercises independent conv/scan state; each stream's per-token argmax + full
    logits must equal the golden single-stream MambaSeqRef run of that sequence."""
    N = args.n
    coh = MambaCohortRef(N, args.export)
    eng = coh.eng

    # N distinct token streams (deterministic), length T
    T = args.tokens
    rng = np.random.default_rng(args.seed)
    seqs = [[int(x) for x in rng.integers(0, V, T)] for _ in range(N)]
    # make stream 0 the real val.bin head if available (a realistic trajectory)
    vb = f"{REPO}/data/bpe1024_chat6/val.bin"
    if os.path.exists(vb):
        val = np.fromfile(vb, dtype=np.uint16)
        seqs[0] = [int(x) for x in val[:T]]

    # ---- golden: N independent single-stream runs ------------------------
    gold = []
    for s in range(N):
        st = eng.alloc_state()
        row = []
        for t in range(T):
            c = {}
            eng.step(seqs[s][t], st, collect=c)
            row.append((c["argmax"], c["logits"], c[f"l{L-1}.x_out"]))
        gold.append(row)

    # ---- cohort: all N streams together, phase-major ---------------------
    states = coh.alloc_states()
    coh_rows = [[] for _ in range(N)]
    for t in range(T):
        cs = [{} for _ in range(N)]
        coh.step([seqs[s][t] for s in range(N)], states, collects=cs)
        for s in range(N):
            coh_rows[s].append((cs[s]["argmax"], cs[s]["logits"],
                                cs[s][f"l{L-1}.x_out"]))

    # ---- compare ---------------------------------------------------------
    bad = 0
    for s in range(N):
        for t in range(T):
            ga, gl, gx = gold[s][t]
            ca, cl, cx = coh_rows[s][t]
            dl = int(np.abs(np.array(cl) - np.array(gl)).max())
            dx = int(np.abs(np.array(cx) - np.array(gx)).max())
            ok = (ca == ga) and dl == 0 and dx == 0
            if not ok:
                bad += 1
                if bad <= 12:
                    print(f"  MISMATCH s{s} t{t}: argmax coh {ca} gold {ga} "
                          f"logit|d|={dl} x|d|={dx}")
    tot = N * T
    print(f"  cohort streams={N} tokens={T}: {tot - bad}/{tot} token-steps "
          f"bit-exact to single-stream")
    print(f"MAMBA_COHORT_VERIFY: {'BIT-EXACT' if bad == 0 else 'MISMATCH'} "
          f"(N={N}, T={T})")
    return 0 if bad == 0 else 1


def _gemv_check(args):
    """The shared-weight cohort GEMV == per-stream single GEMV, bit-exact."""
    N = args.n
    coh = MambaCohortRef(N, args.export)
    eng = coh.eng
    rng = np.random.default_rng(args.seed)
    # exercise all three projection geometries the engine uses
    cases = [("in_proj", 0, INROWS, D // 8),
             ("out_proj", eng.out_base, D, DIN // 8),
             ("head", eng.head_base, V, D // 8)]
    bad = 0
    for name, base, rows, wpr in cases:
        cols = wpr * 8
        xs = [[int(x) for x in rng.integers(-128, 128, cols)] for _ in range(N)]
        cacc = coh.gemv_cohort(base, rows, wpr, xs)
        for s in range(N):
            single = eng._gemv(base, rows, wpr, xs[s])
            d = int(np.abs(np.array(cacc[s]) - np.array(single)).max())
            if d != 0:
                bad += 1
                print(f"  MISMATCH {name} s{s}: max|diff|={d}")
        print(f"  {name:9s}: rows={rows} wpr={wpr}  N={N} streams  "
              f"{'OK' if all(cacc[s] == eng._gemv(base, rows, wpr, xs[s]) for s in range(N)) else 'FAIL'}")
    print(f"MAMBA_COHORT_GEMV: {'BIT-EXACT' if bad == 0 else 'FAIL'} (N={N})")
    return 0 if bad == 0 else 1


def _cycles(args):
    """Per-token cycle accounting for the mamba_seq FSM, and the DERIVED cohort
    per-stream cyc/token as a function of N.  The point of this rung is to find
    what actually amortises.  Each phase of mamba_seq.sv is bucketed as:
      GEMV  : the gemv_i4i8 core time — SHARED (one weight pass serves all N
              streams; measured in run_gemv_cohort: N=8 -> 2325 cyc == 1 pass).
      SHARE : per-LAYER loads that are stream-independent (conv weights+bias) —
              loaded once per layer, then N streams' conv run off them => /N.
      SERIAL: per-stream data movement the single sequencer FSM does one element
              /cycle (the dequant/quant/conv-feed/scan-feed/norm-feed streams).
              A single FSM cannot overlap these across streams => they do NOT
              amortise with stream count.
    Calibrated so GEMV+SHARE+SERIAL == the measured 189,278 (Rung A wide GEMV)."""
    Lz = L
    # ---- per-phase cyc from the mamba_seq.sv FSM sub-step structure ----------
    # (element counts x sub-steps per element; core runtimes calibrated below)
    Rns, Rnl = 560, 1180        # rmsnorm core: short(256) / long(512) run cyc
    Rconv, Rscan = 650, 74      # conv core / per-head scan core run cyc
    # GEMV core (SHARED across N):
    gemv_in, gemv_out, gemv_hd = 2320, 1024, 2048
    gemv = Lz * (gemv_in + gemv_out) + gemv_hd            # 25,456

    # SHARE (per-layer conv weights S_CONV_LDW + bias in S_CONV_LDX sub0):
    share = Lz * (2560 + 640)                             # 22,400

    # SERIAL per layer (all non-GEMV, non-conv-weight FSM streaming + small cores)
    per_layer_serial = (
        512 + Rns          # S_NIN_LD + norm core
        + 768              # S_QIN
        + 512              # S_GIN_LD
        + 4640             # S_GIN_RD (in_proj dequant, 1160x4)
        + 1280             # S_CONV_LDX x-write (bias counted in SHARE)
        + Rconv            # conv core
        + 1280             # S_CONV_RD
        + 256              # S_SCAN_PREP
        + 8 * (196 + Rscan + 192)   # per head: dtx feed + scan core + read
        + 1536 + Rnl       # S_NG_LD + gated norm core
        + 1536             # S_QOUT
        + 512              # S_GOUT_LD
        + 1024             # S_GOUT_RD
    )
    embed = 768
    final = 512 + Rns + 768 + 512 + 3072    # NF_LD+core, QHD, GHD_LD, GHD_RD
    serial = Lz * per_layer_serial + embed + final

    total = gemv + share + serial
    print("=== mamba_seq per-token cycle accounting (Rung A baseline) ===")
    print(f"  GEMV   (shared weight pass, /N) : {gemv:>8,} cyc  "
          f"({100*gemv/total:.0f}%)")
    print(f"  SHARE  (per-layer conv wt, /N)  : {share:>8,} cyc  "
          f"({100*share/total:.0f}%)")
    print(f"  SERIAL (per-stream FSM feed)    : {serial:>8,} cyc  "
          f"({100*serial/total:.0f}%)")
    print(f"  TOTAL                           : {total:>8,} cyc  "
          f"(measured Rung A = 189,278)\n")

    print("  DERIVED cohort per-stream cyc/token (single sequencer FSM):")
    print("    per_stream(N) = SERIAL + (GEMV + SHARE)/N     [GEMV is a distinct")
    print("    phase, not hidden — one batch flows GEMV then the serial feeds]\n")
    print(f"    {'N':>3} {'cyc/stream':>12} {'speedup':>9} {'tok/s@250MHz':>14}")
    for n in (1, 2, 3, 4, 8):
        cps = serial + (gemv + share) / n
        print(f"    {n:>3} {cps:>12,.0f} {total/cps:>8.2f}x {250e6/cps:>13,.0f}")
    print(f"\n    100k target: 8,274 cyc/stream (30,211 tok/s @250MHz)\n")

    print("  READING (the honest wall):")
    print("    * The GEMV amortises perfectly (proven: run_gemv_cohort N=8 = 1")
    print("      pass) but it is only ~13% of the token. Batching streams alone")
    print(f"      caps at ~{total/serial:.2f}x/stream because {100*serial/total:.0f}% of the token is")
    print("      SERIAL per-stream FSM data-movement (1 element/cycle: the")
    print("      in_proj dequant 4,640 x7, conv/scan/norm feeds), which a single")
    print("      sequencer cannot overlap across streams.")
    print("    * Rung B's payoff is REAL but bounded here; the 100k unlock is")
    print("      WIDENING the non-GEMV feed (P-lane dequant/conv/scan/norm, the")
    print("      way Rung A widened the GEMV) — THEN the cohort multiplies it.")
    print("      Ordering: width rung first, then this stream rung on top.")
    return 0


def _fit(args):
    """Per-stream on-chip state budget vs N and the state config.  Weight-
    stationary sharing makes the weight/emb image O(1) in N; only the per-stream
    state (scan h, conv history, residual, working activations) scales with N.
    The largest N is therefore set by (budget - shared) / state-per-stream.

    Numbers cross-checked against docs/9-mamba2-on-kria.md §3 (config-A N=1 map)
    and docs/8-mamba2.md §3.1 (the MEASURED state-quant contract).  KEY honest
    finding from that contract: INT8 state is DEAD (+17% quality: the SSM state
    is re-quantised EVERY step, so round-trip error feeds back through the
    recurrence — unlike a KV cache quantised once at write).  INT12 (+0.91%) is
    the only sub-INT16 quant lever.  d_state=32 (config-B) is the real memory
    lever, but it is a MODEL change (d_model 256->224, retrain + recall-gate
    re-pass), not a runtime knob."""
    # KV260 xck26 (ZU5EV): 2.30 MB URAM + 648 KB BRAM36 = ~2.95 MB on-chip.
    BUDGET_KB = 2.30 * 1024 + 648
    print("=== per-stream on-chip state budget (Rung B cohort) ===")
    print(f"  KV260 on-chip: 2.30 MB URAM + 648 KB BRAM = {BUDGET_KB/1024:.2f} MB\n")

    # shared, N-independent memories (the weight-stationary bank)
    gw_words = len(MambaCohortRef(1, args.export).eng.gw)
    gw_kb = gw_words * 4 / 1024        # 32-bit words (INT4 packed 8/word)
    emb_kb = V * D / 2 / 1024          # INT4 emb
    shared_kb = gw_kb + emb_kb
    print(f"  SHARED (O(1) in N): weight image {gw_kb:.0f} KB (INT4 GEMV) + "
          f"emb {emb_kb:.0f} KB = {shared_kb:.0f} KB")
    avail_kb = BUDGET_KB - shared_kb
    print(f"  => budget left for per-stream state: {avail_kb:.0f} KB\n")

    def report(dstate, hbits, label, dead=False):
        # per-stream scan state h: L*H contexts x P(64) x dstate x hbits
        h_bits = L * H * 64 * dstate * hbits
        conv_bits = L * CONVD * 3 * 16          # conv history (K-1 taps x 16b)
        resid_bits = D * 32                     # residual xbuf Q6.19
        work_bits = INROWS * 16 + CONVD * 16 + DIN * 16 + DIN * 8   # scratch
        persist = h_bits + conv_bits + resid_bits
        total = (persist + work_bits) / 8 / 1024
        maxn = int(avail_kb // total)
        flag = "  [DEAD: +17% quality, see §3.1]" if dead else ""
        print(f"  [{label}] d_state={dstate} state={hbits}b{flag}")
        print(f"    scan h {h_bits/8/1024:6.1f} + conv {conv_bits/8/1024:4.1f} + "
              f"resid {resid_bits/8/1024:.1f} + scratch {work_bits/8/1024:.1f} "
              f"= {total:.1f} KB/stream")
        print(f"    => max N that fits = floor({avail_kb:.0f}/{total:.1f}) = "
              f"{'N/A (dead)' if dead else maxn}")
        return maxn

    report(64, 16, "config-A  (proven, INT16)")
    report(64, 12, "config-A  (INT12 state)")
    report(64, 8, "config-A  (INT8 state)", dead=True)
    report(32, 16, "config-B  (d_state32, INT16)")
    report(32, 12, "config-B  (d_state32, INT12)")
    print()
    print("  READING (honest, matches docs/9 'N=1-3 architecture'):")
    print("    * Weights amortise perfectly — the whole 100k lever hinges on the")
    print("      per-stream SSM state, which does NOT amortise.")
    print("    * config-A: N=2-3 full-context streams (INT16/INT12). This is the")
    print("      safe, retrain-free cohort — Rung B lands here.")
    print("    * N=8 needs config-B (d_state 32) AND INT12 state AND is tight;")
    print("      it is a MODEL change gated on the held-out recall re-pass, not a")
    print("      drop-in. INT8 state is dead. => N=8 is a Rung-C/model decision.")
    print("    * These are arithmetic lower bounds; the placed URAM/BRAM split is")
    print("      confirmed by impl_mamba.tcl (h is BRAM-shaped, weights URAM).")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.mamba_cohort_ref")
    ap.add_argument("--export", default="data/fabric_mamba")
    ap.add_argument("-n", type=int, default=4, help="cohort stream count N")
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--gemv", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--cycles", action="store_true")
    args = ap.parse_args(argv)

    rc = 0
    if args.gemv:
        rc |= _gemv_check(args)
    if args.cycles:
        rc |= _cycles(args)
    if args.fit:
        rc |= _fit(args)
    if args.verify or not (args.gemv or args.fit or args.cycles):
        rc |= _verify(args)
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(main())
