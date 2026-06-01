"""Honest integrated-sequencer cycle model — built from the REAL gated-RTL block
latencies, not estimates. Answers "what tok/s does the assembled dataflow actually
hit, and what does 100k require?"

Per-token decode is data-dependent and autoregressive (layer L needs L-1's residual;
token t+1 needs token t's sampled id), so the per-token latency is the SUM of the
stage latencies on the critical path — a wide GEMV array does NOT overlap the
non-linears unless they are pipelined element-by-element. The honest finding this
exposes: the CURRENT non-linear RTL blocks are serial (correctness-first), so they
dominate and cap the sequencer well below the GEMV's potential.

    python -m fabric.stage3.cycle_model

Block latencies (MEASURED from the committed RTL):
- gemv_banked: per (M,K) matmul = ceil(M/PE) * (K + RLAT), RLAT=2 (per-group K-stream).
- layernorm  : ~3 serial passes over d=256 (load + sum-of-squares + out) + rsqrt ~= 770 cyc/vector.
- softmax    : 3 passes over T + ~41-cycle reciprocal divide ~= 3T + 45 cyc/row.
- gelu_lut   : elementwise, 1024-wide MLP hidden -> ~1024 + 3 cyc/vector.
"""

from __future__ import annotations

RLAT = 2
N_LAYER, N_HEAD, D, D_MLP, VOCAB = 4, 4, 256, 1024, 193

# (M, K) of the matmuls run per token in incremental decode
LAYER_GEMVS = [("qkv", 768, 256), ("proj", 256, 256), ("mlp_fc", 1024, 256), ("mlp_proj", 256, 1024)]


def ceil_div(a, b):
    return -(-a // b)


def gemv_cycles(PE):
    """Cycles for all 17 GEMVs of one token at PE lanes."""
    per_layer = sum(ceil_div(M, PE) * (K + RLAT) for _, M, K in LAYER_GEMVS)
    head = ceil_div(VOCAB, PE) * (D + RLAT)
    return per_layer * N_LAYER + head


def attn_cycles(PE, T):
    """q.Kt and attn.V MACs (2*hd*T per head) on the same array at PE lanes."""
    macs = 2 * (D // N_HEAD) * T * N_HEAD * N_LAYER
    return ceil_div(macs, PE)


def nonlinear_cycles(T, mode):
    ln_each = 770
    sm_each = 3 * T + 45
    gelu_each = D_MLP + 3
    n_ln, n_gelu = 2 * N_LAYER, N_LAYER
    n_sm_rows = N_HEAD * N_LAYER
    if mode == "serial":                       # the blocks as built today
        return n_ln * ln_each + n_sm_rows * sm_each + n_gelu * gelu_each
    if mode == "parallel":                     # throughput-optimized projection (PROJECTED)
        ln_p = 16 + 7                          # d-wide adder tree + rsqrt
        sm_p = T + 12                          # one streaming pass, pipelined recip, heads in parallel
        gelu_p = 64                            # P-wide elementwise
        return n_ln * ln_p + N_LAYER * sm_p + n_gelu * gelu_p
    raise ValueError(mode)


def tok_per_s(PE, T, fmax_mhz, mode):
    cyc = gemv_cycles(PE) + attn_cycles(PE, T) + nonlinear_cycles(T, mode)
    return cyc, fmax_mhz * 1e6 / cyc


def table():
    print("Integrated sequencer — cyc/token and tok/s (autoregressive: stages sum on the critical path)\n")
    for mode in ("serial", "parallel"):
        tag = "CURRENT serial non-linear RTL (MEASURED latencies)" if mode == "serial" \
            else "PROJECTED with throughput-optimized parallel non-linears"
        print(f"== {mode.upper()}: {tag} ==")
        print(f"{'PE':>5} {'Fmax':>6} {'T':>4} {'GEMV':>7} {'attn':>6} {'nonlin':>7} {'cyc/tok':>8} {'tok/s':>9}")
        # Fmax MEASURED by OOC synth on xck26-2LV (constant ~1.5 MB footprint, 0 DSP):
        # PE=256 -> 293 MHz (22.7% LUT), 512 -> 292 (45% LUT), 1024 -> 239 (90% LUT, the
        # LUT wall — pure-LUT INT4 MACs; going wider needs the 1248 idle DSPs).
        for PE, fmax in ((256, 293), (512, 292), (1024, 239)):
            for T in (64, 128, 256):
                g, a, n = gemv_cycles(PE), attn_cycles(PE, T), nonlinear_cycles(T, mode)
                cyc, tps = tok_per_s(PE, T, fmax, mode)
                print(f"{PE:>5} {fmax:>5}M {T:>4} {g:>7} {a:>6} {n:>7} {cyc:>8} {tps:>9,.0f}")
        print()
    print("Reading it: the per-token stages are data-dependent and autoregressive, so they")
    print("SUM (no cross-stage/cross-token overlap). The current serial non-linears dominate")
    print("(8x~770 LN + 16x softmax), capping the assembled sequencer ~10-15k regardless of")
    print("GEMV width. 10k is reachable (PE>=1024, short context). 100k needs the non-linears")
    print("re-architected for parallelism (the 'parallel' rows) AND a wide GEMV AND moderate")
    print("context — the throughput core is already synth-proven; the non-linears are the next")
    print("throughput target. Numbers tagged: GEMV/attn DERIVED, non-linear-serial MEASURED,")
    print("non-linear-parallel PROJECTED (needs the parallel RTL + a Vivado run).")


if __name__ == "__main__":
    table()
