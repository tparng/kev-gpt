"""Batched-sequencer throughput model — the quantitative path to 100k.

Single-stream decode SUMS its stage latencies (autoregressive: each stage needs the
previous; each token needs the previous token's id). That is why a wide GEMV alone
only gets ~10-13k: the GEMV, LayerNorm, softmax and GELU run one-after-another.

Batching B concurrent streams PIPELINES them through the SHARED units: while stream A
is in softmax, stream B is in the GEMV array, stream C in GELU, ... So aggregate
throughput is set by the BUSIEST single unit's per-token cycles, NOT the sum. Batching
overlaps the units. The levers to 100k then are exactly three, and this model prices them:
  1. parallel non-linears (so a non-linear isn't the bottleneck unit),
  2. enough streams B to fill the pipeline (KV-cache-on-chip limited),
  3. multiple GEMV engines (the 1248 idle DSPs) so the GEMV unit isn't the bottleneck.

    python -m fabric.stage3.batched_model

Tags: GEMV cycles DERIVED; non-linear-serial latencies MEASURED (the gated RTL);
non-linear-parallel + multi-GEMV-engine PROJECTED (need the parallel RTL + a synth).
"""

from __future__ import annotations

from .cycle_model import ceil_div, gemv_cycles, D, D_MLP, N_HEAD, N_LAYER

ONCHIP_LEFTOVER_KB = 1216          # after ~1.5 MB weights: 16 URAM + 144 BRAM


def kv_per_stream_kb(T, bits=8):
    return N_LAYER * 2 * T * D * bits // 8 // 1024


def unit_cycles(PE, T, mode, n_gemv):
    """Per-token busy cycles of each SHARED hardware unit."""
    gemv = (gemv_cycles(PE) + ceil_div(2 * (D // N_HEAD) * T * N_HEAD * N_LAYER, PE)) / n_gemv
    if mode == "serial":           # the non-linear RTL as built today
        ln, sm, gelu = 2 * N_LAYER * 770, N_HEAD * N_LAYER * (3 * T + 45), N_LAYER * (D_MLP + 3)
    else:                          # throughput-optimized parallel non-linears (PROJECTED)
        ln, sm, gelu = 2 * N_LAYER * 23, N_LAYER * (T + 12), N_LAYER * 64
    return {"GEMVx%d" % n_gemv: gemv, "LN": ln, "softmax": sm, "GELU": gelu}


def batched(PE, T, fmax, mode, n_gemv):
    u = unit_cycles(PE, T, mode, n_gemv)
    bottleneck = max(u.values())
    single_sum = sum(u.values())                       # ~ single-stream cyc/token
    b_fill = ceil_div(int(single_sum), int(bottleneck))   # streams to fill the pipeline
    agg = fmax * 1e6 / bottleneck                       # aggregate tok/s (pipeline full)
    return u, bottleneck, single_sum, b_fill, agg


def line(PE, T, fmax, mode, n_gemv):
    u, bn, ss, bf, agg = batched(PE, T, fmax, mode, n_gemv)
    kv = kv_per_stream_kb(T)
    b_kv = ONCHIP_LEFTOVER_KB // kv                     # max streams that keep KV on-chip
    B = min(bf, b_kv)
    real_agg = fmax * 1e6 / bn * (B / bf) if bf else 0  # if KV caps B below fill, throughput scales down
    busiest = max(u, key=u.get)
    fits = "fits" if b_kv >= bf else f"KV-capped B={b_kv}"
    print(f"{mode[:4]:>4} PE={PE:<4} {fmax}M T={T:<3} gemvx{n_gemv}  "
          f"bottleneck={busiest}({bn:.0f})  Bfill={bf} KVmax={b_kv}  "
          f"single={fmax*1e6/ss:>7,.0f}  AGG={real_agg:>8,.0f} tok/s  [{fits}]")


def main():
    print("BATCHED SEQUENCER — aggregate tok/s = Fmax / busiest-unit cycles (pipeline full)\n")
    print("Single GEMV engine, current SERIAL non-linears (what exists today):")
    for T in (64, 128, 256):
        line(1024, T, 239, "serial", 1)
    print("\nSingle GEMV engine, PROJECTED parallel non-linears:")
    for T in (64, 128, 256):
        line(1024, T, 239, "parallel", 1)
    print("\nThe 100k push — parallel non-linears + MULTIPLE GEMV engines (DSP-MACs):")
    for ng in (2, 3, 4):
        line(1024, 128, 239, "parallel", ng)
    print("\nReading it: batching overlaps the units, so aggregate = Fmax/busiest (not the sum)")
    print("-> single-stream ~10k becomes ~30k just by pipelining B~4 streams. The bottleneck")
    print("then is the GEMV unit; splitting it across 3-4 engines (into the 1248 idle DSPs)")
    print("reaches ~100k AGGREGATE at B~4, T=128, INT8 KV (which fits on-chip). 100k is a")
    print("SERVING number (B concurrent users), not single-stream. KV-on-chip caps B; T=128")
    print("INT8 KV = %d KB/stream, leftover %d KB -> B<=%d." %
          (kv_per_stream_kb(128), ONCHIP_LEFTOVER_KB, ONCHIP_LEFTOVER_KB // kv_per_stream_kb(128)))


if __name__ == "__main__":
    main()
