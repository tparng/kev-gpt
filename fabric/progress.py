"""The speed-progress ladder — each optimisation's contribution to tok/s, for the
writeup. Every rung names what it *removed* and the multiplier it bought; measured
rungs are on real silicon, projected rungs are the modelled path beyond them.

    python -m fabric.progress                       # table to stdout
    python -m fabric.progress --plot fabric/progress.png

The honest spine: the CPU-in-the-loop fabric path *asymptotes to the A53's own
speed* — once the matmul is offloaded, the A53 doing the non-matmul forward (attention,
softmax, norm, GELU, sampling) in Python is the wall. The fabric only *beats* the CPU
by removing it from the loop entirely (the sequencer), which is the ~1000x rung.
"""

from __future__ import annotations

import argparse

# (label, tok/s, tag, removed-by-this-step) — ordered; multiplier is vs the prior rung.
# Tiers: MEASURED (real silicon, token-stream bit-exact), SIM (RTL bit-exact in iverilog
# vs seq_ref, bitstream pending), PROJECTED (modelled — the scoped path beyond).
# Rungs 1-4 are the CPU-IN-LOOP path (PE=1): it asymptotes to the A53 (~11). Rung 5 is the
# architectural jump — the sequencer takes the CPU OUT of the loop. 6-7 widen the sequencer.
LADDER = [
    ("PL: re-stream weights / forward\n(Python AXI, O(T^2))", 0.07, "MEASURED",
     "first on-fabric generation"),
    ("+ resident weights in URAM\n(load once at boot)", 0.22, "MEASURED",
     "per-token weight movement (~83%)"),
    ("+ KV cache\n(incremental decode)", 2.71, "MEASURED",
     "the O(T^2) full-context recompute"),
    ("+ C MMIO driver\n(compiled AXI inner loop)", 10.35, "MEASURED",
     "Python per-poke overhead -> matmul off the critical path"),
    ("HW sequencer @40MHz\n(CPU out of the loop)", 44.32, "MEASURED",
     "the A53 doing the non-matmul forward (the ~11 wall)"),
    ("+ resident-read GEMV\n(no per-matmul reload)", 75.8, "SIM",
     "re-streaming each matmul's weights (~42% of cycles)"),
    ("+ PE=256 wide lanes\n(256 MACs/cycle)", 231.0, "MEASURED",
     "the GEMV run phase (16x fewer group passes)"),
    ("+ GELU stream + LN pipeline\n(PE=128 @125 MHz silicon)", 751.78, "MEASURED",
     "GELU per-elem stall + LN DSP cascade -> Fmax 50->125 MHz (STA-safe 71/430)"),
    ("+ wide P-lane datapath\n(P=4 LN/dequant/attn/GELU @100MHz)", 1882.7, "MEASURED",
     "the 1-elem/cyc serial loops -> P-wide (53,116 cyc/tok, 2.5x over 752)"),
    ("+ BRAM sync-read scratch\n(P=8 @125 MHz silicon)", 2483.9, "MEASURED",
     "LUTRAM datapath -> BRAM (50,324 cyc/tok; STA 79.5 MHz, clean to 125, breaks 142.9)"),
    ("+ P-wide GEMV boundary\n(act-feed + readback P/cycle)", 3511.6, "MEASURED",
     "the 1-elem/cyc GEMV act/readback loops (35,596 cyc/tok; 125 MHz 3/3)"),
    ("+ LANES=256\n(72-bit URAM banks, 60/64)", 5448.8, "MEASURED",
     "half the GEMV passes (22,941 cyc/tok; 125 MHz 3/3, breaks 142.9)"),
    ("+ cycle-floor cut + deep pipeline\n(fused RB+DQ+GELU, P-wide attn @166.7MHz)", 9295.4, "MEASURED",
     "attention scalar load + dequant round-trip + Fmax 85->131 STA (17,930 cyc/tok; 166.7 MHz 3/3)"),
    ("+ batched serving\n(concurrent streams)", 100000.0, "PROJECTED",
     "per-stage idle — overlap units across streams"),
]

# reference baselines (drawn as guide lines, not rungs) — all the same model, B=1 greedy
A53_CHAT = 11.0          # MEASURED: the pure-A53 char chat
A53_BENCH = 177.8        # MEASURED: the A53 raw INT4 GEMV microbench
XPS_CPU = 356.0          # MEASURED: XPS15 PyTorch KV decode (CPU, ctx 256)
XPS_GPU = 719.0          # MEASURED: XPS15 RTX 3050 Ti, fastest torch mode (full recompute)
XPS_CPU_OPT = 1273.0     # MEASURED: XPS15 ONNX Runtime KV, 4 threads, Kevin ctx 64


def print_table():
    print(f"{'rung':52s} {'tok/s':>10s} {'x prev':>8s}  tag")
    print("-" * 86)
    prev = None
    for label, tps, tag, removed in LADDER:
        one = label.replace("\n", " ")
        mult = f"{tps/prev:.0f}x" if prev else "—"
        print(f"{one:52s} {tps:>10,.2f} {mult:>8s}  {tag}   (removes: {removed})")
        prev = tps
    print(f"\nreference: A53 char chat = {A53_CHAT:.0f} tok/s, A53 INT4 GEMV bench = "
          f"{A53_BENCH:.0f} tok/s, XPS15 laptop: torch CPU {XPS_CPU:.0f} / RTX 3050 Ti "
          f"{XPS_GPU:.0f} / ONNX Runtime CPU best {XPS_CPU_OPT:.0f} (all MEASURED).")
    print("note: the C-driver rung (10.35) ~= the A53 chat (11) — CPU-in-loop asymptotes "
          "to the A53; the HW sequencer (CPU OUT of the loop) is the leap that beats it. "
          "MEASURED on silicon, token-stream bit-exact, deterministic: 44.32 (baseline PE16) "
          "-> 231 (resident-read + PE=256, @40MHz) -> 751.78 (+ streaming GELU + pipelined "
          "LayerNorm, PE=128 @125 MHz, 3/3 runs). The 125 MHz is the silicon Fmax (STA closes "
          "at 71 MHz=430 tok/s; it overclocks clean to 125, breaks at 142). The P-wide serial "
          "datapath (sequencer_vec, P=4/L=128, wide-word banking) is now MEASURED too: 1,882.7 "
          "tok/s @100 MHz, 3/3 bit-exact (53,116 cyc/token; STA closes 40 MHz, overclocks clean "
          "to 100, marginal at 111). BRAM sync-read scratch (P=8/L=128, TMAX=64): MEASURED "
          "2,483.9 tok/s @125 MHz 3/3 (50,324 cyc/token; STA 79.5 MHz, breaks at 142.9) — beats "
          "an XPS15 laptop on the same model (ORT CPU 1,273 best, RTX 3050 Ti 719: launch-bound). "
          "Next: widen the GEMV boundary -> ~10k (see WIDE-WORD-DATAPATH-LOG.md).")


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [l for l, _, _, _ in LADDER]
    vals = [v for _, v, _, _ in LADDER]
    tags = [t for _, _, t, _ in LADDER]
    x = np.arange(len(LADDER))
    meas = "#27ae60"        # MEASURED on silicon
    sim = "#5dade2"         # SIM: RTL bit-exact vs seq_ref, bitstream pending
    proj = "#b0b7bd"        # PROJECTED: modelled
    cmap = {"MEASURED": meas, "SIM": sim, "PROJECTED": proj}
    hmap = {"MEASURED": "", "SIM": "..", "PROJECTED": "//"}
    colors = [cmap[t] for t in tags]

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    bars = ax.bar(x, vals, color=colors, edgecolor="#2c3e50", width=0.62,
                  hatch=[hmap[t] for t in tags])
    ax.set_yscale("log")
    ax.set_ylim(0.04, 300000)

    # value + multiplier annotations
    prev = None
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.25,
                f"{v:,.0f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        if prev:
            ax.annotate(f"×{v/prev:.0f}", xy=(i - 0.5, (v * prev) ** 0.5),
                        ha="center", va="center", fontsize=10, color="#c0392b",
                        fontweight="bold")
        prev = v

    # baselines — the references to beat (same model, B=1 greedy decode)
    refs = [
        (A53_CHAT, f"A53 chat ({A53_CHAT:.0f})", "#2980b9", "bottom"),
        (XPS_CPU, f"XPS15 torch CPU ({XPS_CPU:.0f})", "#8e44ad", "bottom"),
        (XPS_GPU, f"XPS15 RTX 3050 Ti ({XPS_GPU:.0f})", "#d35400", "top"),
        (XPS_CPU_OPT, f"XPS15 ORT CPU ({XPS_CPU_OPT:,.0f})", "#7f1d3c", "bottom"),
    ]
    for yv, lbl, col, va in refs:
        ax.axhline(yv, color=col, ls=":", lw=1.4)
        ax.text(-0.55, yv * (1.1 if va == "bottom" else 0.9), lbl,
                color=col, fontsize=8, ha="left", va=va)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.2, rotation=30, ha="right")
    ax.set_ylabel("tokens / second  (log scale)")
    ax.set_title("Kevin on Kria — the speed ladder: each step's contribution to the gains")
    ax.grid(True, axis="y", which="both", alpha=0.18)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=meas, edgecolor="#2c3e50", label="MEASURED on silicon"),
                       Patch(facecolor=sim, edgecolor="#2c3e50", hatch="..",
                             label="SIM (RTL bit-exact vs seq_ref; bitstream pending)"),
                       Patch(facecolor=proj, edgecolor="#2c3e50", hatch="//", label="PROJECTED")],
              loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fabric.progress", description=__doc__.splitlines()[0])
    p.add_argument("--plot", metavar="PATH", default=None)
    a = p.parse_args(argv)
    print_table()
    if a.plot:
        plot(a.plot)


if __name__ == "__main__":
    main()
