"""metrics_report.py — read the over-time rollup log (server.py --rollup-jsonl)
and show how latency and request volume moved over time.

    python -m webchat.demo.metrics_report live.rollup.jsonl
    python -m webchat.demo.metrics_report live.rollup.jsonl --plot metrics.png
    python -m webchat.demo.metrics_report live.rollup.jsonl --since 2h --tail 40

Each rollup line is one window (default 60 s): requests in the window, EXACT
windowed latency percentiles, tok/s, peak concurrent users, unique visitors
(30 m). Percentiles across windows can't be exactly recombined, so the summary
reports the request-weighted mean p50 and the worst-window p95/p99 (honest).
"""

from __future__ import annotations

import argparse
import json
import time


def _parse_since(s: str | None) -> float:
    if not s:
        return 0.0
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s[-1] in mult:
        return time.time() - float(s[:-1]) * mult[s[-1]]
    return time.time() - float(s)


def load(path: str, since_ts: float) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "rollup" and r.get("ts", 0) >= since_ts:
                out.append(r)
    return out


def hhmm(ts: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


def table(rows: list[dict], tail: int) -> None:
    shown = rows[-tail:] if tail else rows
    hdr = f"{'time':12s} {'req':>5s} {'/min':>6s} {'p50':>6s} {'p95':>6s} {'p99':>6s} {'tok/s':>6s} {'peak':>4s} {'vis30':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for r in shown:
        def f(x, w=6):
            return (f"{x:>{w}.0f}" if isinstance(x, (int, float)) else f"{'—':>{w}}")
        print(f"{hhmm(r['ts']):12s} {f(r.get('requests'),5)} {f(r.get('req_per_min')):>6} "
              f"{f(r.get('lat_p50'))} {f(r.get('lat_p95'))} {f(r.get('lat_p99'))} "
              f"{f(r.get('tok_s'))} {f(r.get('peak_concurrent'),4)} {f(r.get('visitors_30m'),5)}")


def summary(rows: list[dict]) -> None:
    if not rows:
        print("no rollup records in range.")
        return
    span = rows[-1]["ts"] - rows[0]["ts"] + rows[0].get("window_s", 60)
    req = sum(r.get("requests", 0) for r in rows)
    toks = sum(r.get("tokens", 0) for r in rows)
    wreq = [(r.get("lat_p50"), r.get("requests", 0)) for r in rows if r.get("lat_p50") is not None]
    wmean_p50 = (sum(p * n for p, n in wreq) / sum(n for _, n in wreq)
                 if any(n for _, n in wreq) else None)
    worst_p95 = max((r.get("lat_p95") or 0) for r in rows)
    worst_p99 = max((r.get("lat_p99") or 0) for r in rows)
    peak_rpm = max((r.get("req_per_min") or 0) for r in rows)
    peak_conc = max((r.get("peak_concurrent") or 0) for r in rows)
    max_vis = max((r.get("visitors_30m") or 0) for r in rows)
    print(f"\nspan {span/3600:.1f}h · {len(rows)} windows · {req} requests "
          f"({req/max(1,span)*60:.1f}/min avg, {peak_rpm:.0f}/min peak)")
    print(f"latency: req-weighted p50 {wmean_p50:.0f}ms · worst-window p95 {worst_p95:.0f}ms "
          f"· worst-window p99 {worst_p99:.0f}ms" if wmean_p50 is not None
          else "latency: no samples")
    print(f"tokens {toks:,} · peak concurrent {peak_conc:.0f} · max visitors(30m) {max_vis:.0f}")


def plot(rows: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    import datetime as dt

    t = [dt.datetime.fromtimestamp(r["ts"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(t, [r.get("req_per_min") or 0 for r in rows], color="#3fb950", lw=1.6)
    ax1.fill_between(t, [r.get("req_per_min") or 0 for r in rows], color="#3fb950", alpha=0.15)
    ax1.set_ylabel("requests / min"); ax1.grid(alpha=0.25)
    ax1.set_title("Kevin on Kria — request volume & latency over time")
    for q, c in (("lat_p50", "#58a6ff"), ("lat_p95", "#d29922"), ("lat_p99", "#f85149")):
        ax2.plot(t, [r.get(q) for r in rows], color=c, lw=1.4, label=q.replace("lat_", ""))
    ax2.set_ylabel("latency (ms)"); ax2.grid(alpha=0.25); ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="the rollup JSONL (server --rollup-jsonl)")
    ap.add_argument("--since", default=None, help="e.g. 30m, 2h, 1d (default: all)")
    ap.add_argument("--tail", type=int, default=30, help="rows to print (0 = all)")
    ap.add_argument("--plot", default=None, metavar="PATH", help="write a PNG time-series")
    a = ap.parse_args(argv)
    rows = load(a.path, _parse_since(a.since))
    table(rows, a.tail)
    summary(rows)
    if a.plot and rows:
        plot(rows, a.plot)


if __name__ == "__main__":
    main()
