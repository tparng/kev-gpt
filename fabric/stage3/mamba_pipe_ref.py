"""mamba_pipe_ref — Phase 2 Rung C: the PIPELINED (wave) multi-stream schedule.

Rungs A/B/B' attacked the token from two sides: Rung A widened the GEMV (16
weight-words/cycle), Rung B proved a bit-exact N-stream weight-stationary cohort,
Rung B' widened the five biggest one-element/cycle FSM feeds (P=4 dequant/quant).
The committed single-stream token is 144,246 cyc.  But the cohort of Rung B is
PHASE-MAJOR — every stream does phase 1, then every stream does phase 2 — so the
sub-cores idle between phases and the batch only buys ~1.2x/stream (the
`mamba_cohort_ref --cycles` wall: 75% of the token is serial per-stream feed a
single sequencer FSM cannot overlap across streams).

This module models the lever that breaks that plateau: a WAVE schedule where
independent streams occupy DIFFERENT sub-cores in the SAME cycle-group — stream A
in ssm_scan while stream B is in conv_silu while stream C is in gemv while stream
D is in rmsnorm.  Streams are independent (each owns its h / conv-history /
residual — proven in mamba_cohort_ref, 16/16 bit-exact), so this is legal; token
throughput then approaches 1 token per (busiest-CORE time) instead of the SUM of
all phases.  This is exactly how the transformer's cohort_engine.sv beat its
serial queue (overlap the GEMM run with the act-quant feed and the readback).

WHAT THIS FILE IS: a cycle-accurate, resource-constrained discrete-event
schedule simulator.  It is the TIMING half of the rung; the VALUE half is already
proven — mamba_cohort_ref.step() reproduces the single-stream logits/argmax for
every stream bit-exactly, and the wave schedule changes only WHEN each stream's
op runs, never the arithmetic.  So a measured schedule here + a bit-exact replica
there = the full Rung-C claim, without a 25-minute iverilog spin per data point.

    python -m fabric.stage3.mamba_pipe_ref            # NC sweep, wave vs phase-major
    python -m fabric.stage3.mamba_pipe_ref --gemv-cohort   # + weight-stationary GEMV
    python -m fabric.stage3.mamba_pipe_ref --loads   # per-resource load table + wall

The op durations are the COMMITTED post-B' per-phase cycle counts (calibrated so
the NC=1 schedule reproduces the MEASURED 144,246-cyc token); the schedule result
is DERIVED (an idealized non-delay list scheduler — a good on-chip round-robin
arbiter realizes within its scheduling efficiency of this bound).
"""

from __future__ import annotations

import argparse

# committed single-stream token (MEASURED, run_mamba_seq --tokens 2, HEAD 2091222)
TOKEN_MEASURED = 144_246

# ---- physical sub-cores / datapaths (one instance each; a wave stream occupies
#      at most one per cycle-group).  Mirrors the mamba_seq.sv sub-cores plus the
#      B' widened feed datapaths, which a pipelined engine instantiates as
#      independently-sequenced units with per-stream banked I/O (cohort_engine
#      pattern), NOT one monolithic FSM. --------------------------------------
GEMV    = "gemv"      # gemv_i4i8 MAC array (in_proj / out_proj / head)
NORM    = "norm"      # rmsnorm_gated core + its stream feed (pre / gated / final)
CONV    = "conv"      # conv_silu core + LDX/RD feed
SCAN    = "scan"      # ssm_scan_row core + dtx feed + y read (per head)
DEQUANT = "dequant"   # gemv-accumulator two-stage dequant (B' P=4 lanes)
QUANT   = "quant"     # activation INT8 quant + the gemv act-quant x-write
EMB     = "emb"       # token-embed lookup + scale
RESOURCES = [GEMV, NORM, CONV, SCAN, DEQUANT, QUANT, EMB]

# ---- per-phase durations (post-B', from the commit + mamba_cohort_ref._cycles).
#      (name, resource, cycles, shared)  shared=True => weight-stationary per-LAYER
#      load reused by all NC streams (amortised /NC in steady state).
#      Feeds are folded into the CORE that consumes them — the realizable
#      partition where each sub-core owns its feed FSM (cohort_engine overlaps the
#      act-quant x-write with the MAC, so GIN_LD/GOUT_LD sit on QUANT, not GEMV).
PER_LAYER = [
    ("NIN_LD",   NORM,    512, False),   # pre-norm stream-in
    ("NIN",      NORM,    560, False),   # rmsnorm core (short 256)
    ("QIN",      QUANT,   192, False),   # quant xn -> int8 (B': 768->192)
    ("GIN_LD",   QUANT,   512, False),   # act-quant x-write into gemv bank
    ("GIN",      GEMV,   2320, False),   # in_proj GEMV (1160 rows)
    ("GIN_RD",   DEQUANT,1160, False),   # in_proj dequant (B': 4640->1160)
    ("CONV_LDW", CONV,   2560, True),    # conv weights (per-layer, shared)
    ("CONV_BIAS",CONV,    640, True),    # conv bias    (per-layer, shared)
    ("CONV_LDX", CONV,   1280, False),   # conv x stream-in
    ("CONV",     CONV,    650, False),   # conv_silu core
    ("CONV_RD",  CONV,   1280, False),   # conv result read
    ("SCAN_PREP",SCAN,    256, False),   # B8/C8 shared prep
    ("SCAN_H",   SCAN,   2160, False),   # 8 heads x (196 dtx feed + 74 scan core)
    ("SCAN_RD",  SCAN,   1536, False),   # 8 heads x 192 y read + D-skip
    ("NG_LD",    NORM,   1536, False),   # gated-norm stream-in (512-wide)
    ("NG",       NORM,   1180, False),   # rmsnorm core (gated, 512)
    ("QOUT",     QUANT,   384, False),   # quant yn -> int8 (B': 1536->384)
    ("GOUT_LD",  QUANT,   512, False),   # act-quant x-write
    ("GOUT",     GEMV,   1024, False),   # out_proj GEMV (256 rows)
    ("GOUT_RD",  DEQUANT, 256, False),   # out_proj dequant (B': 1024->256) + resid
]
EMBED = [("EMB", EMB, 768, False)]
FINAL = [
    ("NF_LD",   NORM,    512, False),
    ("NF",      NORM,    560, False),
    ("QHD",     QUANT,   768, False),
    ("GHD_LD",  QUANT,   512, False),
    ("GHD",     GEMV,   2048, False),    # head GEMV (1024 rows)
    ("GHD_RD",  DEQUANT,1024, False),    # head dequant + argmax (B': 4096->1024)
]
L = 7


def token_ops(nc, gemv_cohort=False):
    """The op-chain for ONE token of ONE stream (strictly ordered = data deps).
    shared per-layer loads are amortised /nc (weight-stationary); optionally the
    GEMV core too (--gemv-cohort: one weight pass folds nc streams' MACs)."""
    ops = []

    def emit(table):
        for name, res, cyc, shared in table:
            d = cyc
            if shared:
                d = max(1, round(cyc / nc))          # per-layer load reused by nc
            elif res == GEMV and gemv_cohort:
                d = max(1, round(cyc / nc))          # weight-stationary MAC share
            ops.append((res, d))

    emit(EMBED)
    for _ in range(L):
        emit(PER_LAYER)
    emit(FINAL)
    return ops


# ---------------------------------------------------------------- scheduler ---
def schedule(nc, tokens, gemv_cohort=False):
    """Non-delay resource-constrained list schedule of `nc` INDEPENDENT streams,
    each decoding `tokens` tokens back-to-back.  Ops within a stream are strictly
    ordered (data deps) and tokens are serial within a stream (decode t+1 needs
    t's argmax) — overlap is ONLY across the nc streams.  Each resource is a
    single non-preemptive server.  Returns makespan (cycles)."""
    per_tok = token_ops(nc, gemv_cohort)
    per = len(per_tok) * tokens
    chain = per_tok * tokens                            # one stream's whole chain
    idx = [0] * nc
    savail = [0] * nc                                   # stream next-op ready time
    rfree = {r: 0 for r in RESOURCES}                   # resource free time
    total = nc * per
    done = 0
    makespan = 0
    while done < total:
        best = None                                     # (start, res, dur, s)
        for s in range(nc):
            if idx[s] < per:
                r, dur = chain[idx[s]]
                start = savail[s] if savail[s] > rfree[r] else rfree[r]
                if best is None or start < best[0]:
                    best = (start, r, dur, s)
        start, r, dur, s = best
        fin = start + dur
        rfree[r] = fin
        savail[s] = fin
        idx[s] += 1
        done += 1
        if fin > makespan:
            makespan = fin
    return makespan


def _phase_major_cost(nc):
    """Rung-B phase-major cohort: nc streams clear a phase (serially on its one
    core) before the batch advances; shared per-layer loads amortise /nc."""
    c = 0
    for name, res, cyc, shared in (EMBED + PER_LAYER * L + FINAL):
        c += max(1, round(cyc / nc)) if shared else cyc * nc
    return c


def resource_loads(nc, gemv_cohort=False):
    """Per-stream per-token cycles on each physical resource (the wall)."""
    load = {r: 0 for r in RESOURCES}
    for name, res, cyc, shared in (EMBED + PER_LAYER * L + FINAL):
        d = max(1, round(cyc / nc)) if shared else (
            max(1, round(cyc / nc)) if (res == GEMV and gemv_cohort) else cyc)
        load[res] += d
    return load


# ------------------------------------------------------------------- report ---
def _tok_s(cyc_per_tok, mhz):
    return mhz * 1e6 / cyc_per_tok


def run(args):
    calib = TOKEN_MEASURED / sum(d for _, d in token_ops(1))
    print("=== Rung C: pipelined (wave) multi-stream schedule ===")
    print(f"  committed single-stream token: {TOKEN_MEASURED:,} cyc (MEASURED)")
    print(f"  op-model NC=1 sum: {sum(d for _,d in token_ops(1)):,} cyc "
          f"-> calib x{calib:.5f} to match\n")

    if args.loads:
        # asymptotic wall: shared per-layer loads amortise to ~0 (large NC)
        load = resource_loads(1024, args.gemv_cohort)
        wall_r = max(load, key=load.get)
        print("  per-stream per-token load on each physical sub-core "
              f"(cyc, calibrated, NC->inf){'  [GEMV weight-stationary]' if args.gemv_cohort else ''}:")
        for r in sorted(RESOURCES, key=lambda r: -load[r]):
            print(f"    {r:8s}: {load[r]*calib:>10,.0f}"
                  f"{'   <-- WALL' if r == wall_r else ''}")
        wall = load[wall_r] * calib
        print(f"\n  wave floor (NC->inf) = busiest core = {wall:,.0f} cyc/token")
        print(f"  => max theoretical speedup vs 144,246 = {TOKEN_MEASURED/wall:.2f}x")
        print(f"     @200MHz {_tok_s(wall,200):,.0f} tok/s   "
              f"@250MHz {_tok_s(wall,250):,.0f} tok/s")
        print("\n  the wall is the ssm_scan resource — but only ~4.1k of its "
              f"{load[SCAN]*calib:,.0f} cyc\n  is the scan CORE; the rest is the per-head "
              "dtx-feed + y-read still at ~1\n  element/cycle. That feed is the next "
              "P-lane widen target (Rung D).")
        return 0

    # fit caps (mamba_cohort_ref --fit): max streams that fit on the device.
    # split-brain adds a 2nd GEMV via the URAM TDP ports but NOT more state room,
    # so the TOTAL live state N across both cohorts is bounded by the SAME budget.
    fit = {1: "config-A INT16", 2: "config-A INT16  <= single-cohort cap (retrain-free)",
           3: "config-A INT12  <= single-cohort cap (retrain-free)",
           4: "config-B INT16  (model change: retrain + recall re-gate)",
           5: "config-B INT16  <= device state cap",
           6: "config-B INT12  <= device state cap",
           8: "OVER BUDGET (needs a bigger part or sub-INT12 state)",
           16: "OVER BUDGET (aspirational; NC->inf floor reference)"}
    T = args.tokens
    print(f"  wave schedule, {T} tokens/stream, "
          f"{'weight-stationary GEMV' if args.gemv_cohort else 'per-stream GEMV'}:\n")
    print(f"  {'NC':>3} {'eff cyc/tok':>12} {'vs 144k':>8} {'tok/s@200':>11} "
          f"{'tok/s@250':>10} {'phase-maj':>10}   fit")
    for nc in (1, 2, 3, 4, 5, 6, 8, 16):
        mk = schedule(nc, T, args.gemv_cohort) * calib
        eff = mk / (nc * T)
        pm = _phase_major_cost(nc) * calib / nc          # phase-major per-stream-tok
        note = fit.get(nc, "")
        star200 = "*" if _tok_s(eff, 200) > 6344 else " "
        print(f"  {nc:>3} {eff:>12,.0f} {TOKEN_MEASURED/eff:>7.2f}x "
              f"{_tok_s(eff,200):>10,.0f}{star200}{_tok_s(eff,250):>9,.0f} "
              f"{pm:>10,.0f}   {note}")

    print(f"\n  bars:  6,344 tok/s = 21k chars/s (old usable record, * = beats @200)")
    print(f"        30,211 tok/s = 100k chars/s (the 100k identity)\n")

    # ---- split-brain: 2 cohorts, each of size NC, sharing the weight URAM (TDP)
    #      and the SAME device state budget => total N = 2*NC <= device cap.
    #      Each cohort runs the wave schedule on its OWN sub-cores; aggregate
    #      throughput = 2 * per-cohort. The fit-valid split is 2 x 3 (config-B
    #      INT12, total N=6) — the best retrain-gated point.
    print("  split-brain (2 wave cohorts on the URAM TDP ports; total state = 2xNC):")
    print(f"  {'2xNC':>5} {'agg cyc/tok':>12} {'tok/s@200':>11} {'tok/s@250':>10}   fit")
    for nc in (1, 2, 3):
        mk = schedule(nc, T, args.gemv_cohort) * calib
        eff = mk / (nc * T)
        agg = eff / 2.0                                  # 2 cohorts in parallel
        star = "*" if _tok_s(agg, 200) > 6344 else " "
        tot = 2 * nc
        cap = ("config-B INT12 (2x3, total N=6; retrain + recall re-gate)" if nc == 3
               else "config-B INT16 (2x2, total N=4; retrain + recall re-gate)" if nc == 2
               else "config-A INT16 (2x1, total N=2)")
        print(f"  {tot:>5} {agg:>12,.0f} {_tok_s(agg,200):>10,.0f}{star}"
              f"{_tok_s(agg,250):>9,.0f}   {cap}")

    load = resource_loads(1024, args.gemv_cohort)
    wall_r = max(load, key=load.get)
    print(f"\n  READING: the wave floor is the busiest CORE ('{wall_r}'), not the")
    print(f"  serial sum — that is the plateau-breaker (phase-major stays ~123k,")
    print(f"  the ~1.2x Rung-B wall).  RETRAIN-FREE (config-A) tops out at N=3")
    print(f"  single-cohort = 2.58x = 4,478 tok/s@250 (under the 6,344 bar).")
    print(f"  Beating 6,344 needs config-B (retrain + recall re-gate): N=6 -> 3.81x")
    print(f"  -> 6,609@250, or split-brain 2x3 -> 8,957@250 (2.9x over the bar,")
    print(f"  ~30% of 100k).  The remaining wall is the SCAN/NORM feed (dtx-feed +")
    print(f"  y-read + norm stream-in still ~1 element/cycle); widening THOSE")
    print(f"  (P-lane, the Rung-B' move on the dequant) drops the floor toward the")
    print(f"  ~4.1k scan-CORE bound, and this schedule multiplies it — the credible")
    print(f"  path to the 30,211 (100k) identity.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.stage3.mamba_pipe_ref")
    ap.add_argument("--tokens", type=int, default=12, help="tokens/stream (steady state)")
    ap.add_argument("--gemv-cohort", action="store_true",
                    help="weight-stationary GEMV (share the weight pass across NC)")
    ap.add_argument("--loads", action="store_true", help="per-resource load table")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
