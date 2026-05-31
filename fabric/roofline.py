"""The roofline / crossover model — doc 2's headline result, computed not measured.

Single-stream autoregressive decode is memory-bandwidth bound: each token drags
the whole weight set through the multipliers once, so to first order

    tokens/sec  ~=  effective_read_bandwidth / bytes_streamed_per_token

The whole project hangs on one consequence of that equation: the PS A53 cores
and the PL fabric share the same ~20 GB/s DDR controller, so a DDR-resident
model gets the *same* bandwidth either way — the fabric only wins by leaving DDR
entirely and keeping the model in BRAM/URAM at hundreds of GB/s to TB/s. That
advantage exists only while the model fits under the ~3 MB on-chip ceiling. Past
that the fabric, too, must stream from DDR and the advantage collapses. The model
size where that happens is the *crossover*, and doc 2 treats the crossover plot
as as much the result as the peak number.

This script is runnable today with no hardware and no trained model — it is the
analytical "before" picture the measured stage-0/stage-3 numbers will be checked
against. Run it to print the table and (with matplotlib) write the crossover plot.

    python -m fabric.roofline                      # table to stdout
    python -m fabric.roofline --plot fabric/roofline.png
"""

from __future__ import annotations

import argparse

from .spec import (
    A53_INT8_GOPS,
    DDR_BW_GBPS,
    ModelDims,
    ONCHIP_BW_GBPS_HIGH,
    ONCHIP_BW_GBPS_LOW,
    ONCHIP_WEIGHT_BUDGET_BYTES,
    WEIGHT_BITS,
)


def bytes_per_token(n_params: int, weight_bits: int = WEIGHT_BITS) -> int:
    """Weight bytes streamed per decode step. To first order this dominates the
    KV-cache traffic at short context, so it is the headline term."""
    return n_params * weight_bits // 8


def decode_toks_per_sec(n_params: int, bw_gbps: float, weight_bits: int = WEIGHT_BITS) -> float:
    """Bandwidth-bound decode throughput: bandwidth / bytes-per-token."""
    bpt = bytes_per_token(n_params, weight_bits)
    return (bw_gbps * 1e9) / bpt if bpt else float("inf")


def on_chip_toks_per_sec(n_params: int, bw_gbps: float, weight_bits: int = WEIGHT_BITS,
                         budget_bytes: int = ONCHIP_WEIGHT_BUDGET_BYTES) -> float | None:
    """On-chip decode throughput, or None if the model spills the budget — past
    the cliff the fabric is back on the DDR line, so there is no on-chip number."""
    if bytes_per_token(n_params, weight_bits) > budget_bytes:
        return None
    return decode_toks_per_sec(n_params, bw_gbps, weight_bits)


def crossover_params(weight_bits: int = WEIGHT_BITS,
                     budget_bytes: int = ONCHIP_WEIGHT_BUDGET_BYTES) -> int:
    """The model size (params) at which the weights exactly fill the on-chip
    budget. Bigger than this and the fabric advantage is gone."""
    return budget_bytes * 8 // weight_bits


def sweep(weight_bits: int = WEIGHT_BITS):
    """Yield rows over a param sweep spanning the on-chip regime and well past
    the cliff, each row carrying the DDR line and the on-chip band."""
    xover = crossover_params(weight_bits)
    # geometric-ish sweep from 0.25M to 64M params
    points = [0.25e6, 0.5e6, 1e6, 2e6, 3.16e6, 5e6, xover, 8e6, 16e6, 32e6, 64e6]
    for p in sorted(set(int(p) for p in points)):
        ddr = decode_toks_per_sec(p, DDR_BW_GBPS, weight_bits)
        oc_lo = on_chip_toks_per_sec(p, ONCHIP_BW_GBPS_LOW, weight_bits)
        oc_hi = on_chip_toks_per_sec(p, ONCHIP_BW_GBPS_HIGH, weight_bits)
        yield {
            "params": p,
            "mb_int4": bytes_per_token(p, weight_bits) / 1024 / 1024,
            "ddr_toks": ddr,
            "onchip_lo": oc_lo,
            "onchip_hi": oc_hi,
            "speedup_lo": (oc_lo / ddr) if oc_lo else None,
            "speedup_hi": (oc_hi / ddr) if oc_hi else None,
        }


def print_table(weight_bits: int = WEIGHT_BITS) -> None:
    d = ModelDims()
    xover = crossover_params(weight_bits)
    print(f"# Roofline — INT{weight_bits} decode, KV260 (DDR {DDR_BW_GBPS:.0f} GB/s shared, "
          f"on-chip {ONCHIP_BW_GBPS_LOW:.0f}-{ONCHIP_BW_GBPS_HIGH:.0f} GB/s)")
    print(f"# on-chip budget {ONCHIP_WEIGHT_BUDGET_BYTES/1024/1024:.1f} MB  ->  "
          f"crossover at ~{xover/1e6:.1f}M params")
    print(f"# the deployable Kevin GPT is {d.weight_params()/1e6:.2f}M linear params "
          f"({bytes_per_token(d.weight_params(), weight_bits)/1024/1024:.2f} MB INT{weight_bits})\n")
    hdr = f"{'params':>9} {'MB':>6} {'DDR tok/s':>11} {'on-chip tok/s (lo-hi)':>24} {'speedup':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in sweep(weight_bits):
        if r["onchip_lo"] is None:
            oc = "  spills -> DDR-bound  "
            su = "    1x (no win)"
        else:
            oc = f"{r['onchip_lo']:>10,.0f}-{r['onchip_hi']:<11,.0f}"
            su = f"{r['speedup_lo']:>5.0f}-{r['speedup_hi']:<5.0f}x"
        mark = "  <- deployable" if abs(r["params"] - ModelDims().weight_params()) < 1e5 else ""
        mark = "  <- CROSSOVER" if r["params"] == xover else mark
        print(f"{r['params']:>9,} {r['mb_int4']:>6.2f} {r['ddr_toks']:>11,.0f} {oc:>24} {su:>14}{mark}")
    print()
    # A53 honesty line: the baseline is compute-weak too, not just bandwidth-bound.
    a53_compute_ceiling = A53_INT8_GOPS * 1e9 / (2 * d.weight_params())  # 2 ops/MAC
    print(f"note: A53 also has a compute ceiling ~{a53_compute_ceiling:,.0f} tok/s "
          f"(INT8 {A53_INT8_GOPS:.0f} GOPS / 2·params), so the real A53 number is "
          f"min(bandwidth, compute) — a genuinely weak baseline, stated up front.")


def plot(path: str, weight_bits: int = WEIGHT_BITS) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = ModelDims()
    xover = crossover_params(weight_bits)
    params = np.logspace(np.log10(0.25e6), np.log10(64e6), 200)
    ddr = np.array([decode_toks_per_sec(p, DDR_BW_GBPS, weight_bits) for p in params])
    # on-chip curve: only defined under budget, NaN past the cliff so the line stops
    oc_lo = np.array([on_chip_toks_per_sec(p, ONCHIP_BW_GBPS_LOW, weight_bits) or np.nan for p in params])
    oc_hi = np.array([on_chip_toks_per_sec(p, ONCHIP_BW_GBPS_HIGH, weight_bits) or np.nan for p in params])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(params / 1e6, ddr, color="#c0392b", lw=2, label=f"DDR-resident ({DDR_BW_GBPS:.0f} GB/s, PS & PL share)")
    ax.fill_between(params / 1e6, oc_lo, oc_hi, color="#27ae60", alpha=0.25,
                    label=f"on-chip BRAM/URAM ({ONCHIP_BW_GBPS_LOW:.0f}-{ONCHIP_BW_GBPS_HIGH:.0f} GB/s)")
    ax.loglog(params / 1e6, oc_hi, color="#27ae60", lw=1.5)
    ax.loglog(params / 1e6, oc_lo, color="#27ae60", lw=1.5)
    ax.axvline(xover / 1e6, color="#2c3e50", ls="--", lw=1.5)
    ax.text(xover / 1e6, ax.get_ylim()[1] * 0.5, f"  crossover\n  ~{xover/1e6:.1f}M params\n  ({ONCHIP_WEIGHT_BUDGET_BYTES/1024/1024:.0f} MB cliff)",
            va="top", fontsize=8, color="#2c3e50")
    ax.axvline(d.weight_params() / 1e6, color="#2980b9", ls=":", lw=1.5)
    ax.text(d.weight_params() / 1e6, ax.get_ylim()[0] * 2, f"Kevin GPT\n{d.weight_params()/1e6:.1f}M",
            fontsize=8, color="#2980b9")
    ax.set_xlabel("model size (M params)")
    ax.set_ylabel(f"decode throughput (tok/s, INT{weight_bits})")
    ax.set_title("Kevin-on-Kria roofline: on-chip wins until the model spills the ~3 MB budget")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.roofline", description=__doc__.splitlines()[0])
    p.add_argument("--bits", type=int, default=WEIGHT_BITS, help="weight bit-width (default 4)")
    p.add_argument("--plot", metavar="PATH", default=None, help="also write the crossover plot here")
    args = p.parse_args(argv)
    print_table(args.bits)
    if args.plot:
        plot(args.plot, args.bits)


if __name__ == "__main__":
    main()
