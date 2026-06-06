"""telemetry.py — the 1 Hz aggregate the dashboard renders.

This is the telemetry PLANE: every second we collapse the last window of events
into one small JSON blob, write it to a local JSONL (the post-mortem record) and
POST it to the Cloudflare Worker (fan-out + survives-death rendering). The
metrics and their honest definitions are baked in here so the dashboard and the
server agree on what each number means; the dashboard page restates them.

Metric definitions (honest-first; these strings ship to the page)
-----------------------------------------------------------------
* active_users        : distinct clients with a keystroke in the last
                        active_window_s seconds. NOT total connections.
* agg_tok_s           : tokens (chars) emitted across all completions in the
                        last second / 1 s. Aggregate, not per-stream.
* latency p50/95/99   : end-to-end ms from a job's submit-eligible instant to
                        its completion delivery, over a reservoir of recent
                        jobs. Percentiles, never the mean.
* queue_depth         : prompts past their debounce window not yet submitted at
                        tick time (work waiting on the fabric).
* fabric_occupancy    : backend busy-seconds in the window / window seconds,
                        weighted by slot-fill (filled/capacity). A software
                        proxy until a hardware busy-cycle counter exists; LABELLED
                        as such on the page.
* amplification        : keystrokes_in : inferences_run : blits_shown over the
                        run. The fusion made visible: many keystrokes collapse
                        into few inferences.
* ready_at_enter       : fraction of Enters that found a fresh buffer and blitted
                        with no round trip (the magic working).
* power_w / temp_c     : Kria sysfs readings; None off-box (graceful).
"""

from __future__ import annotations

import bisect
import collections
import dataclasses
import json
import random
import time
from typing import Optional


# ---- percentile reservoir -------------------------------------------------
class Reservoir:
    """Fixed-size reservoir sample of recent latency values (ms). Reservoir
    sampling keeps memory bounded under a flood while staying representative, so
    p99 does not degrade to 'last N only' or blow up under load."""

    def __init__(self, size: int = 2048, seed: int = 1234):
        self.size = size
        self.buf: list[float] = []
        self.n = 0
        self._rng = random.Random(seed)

    def add(self, v: float) -> None:
        self.n += 1
        if len(self.buf) < self.size:
            self.buf.append(v)
        else:
            j = self._rng.randint(0, self.n - 1)
            if j < self.size:
                self.buf[j] = v

    def percentile(self, q: float) -> Optional[float]:
        if not self.buf:
            return None
        s = sorted(self.buf)
        k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        return s[k]

    def snapshot(self) -> dict:
        return {
            "p50": self.percentile(0.50),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
            "n": self.n,
        }


# ---- power / temp readers (graceful None off-box) -------------------------
# Kria exposes power via the INA226 hwmon and temp via the AMS thermal zone.
# Paths vary by image; we probe a couple and return None if absent.
_POWER_PATHS = [
    "/sys/class/hwmon/hwmon0/power1_input",     # microwatts (INA226 total)
    "/sys/class/hwmon/hwmon1/power1_input",
]
_TEMP_PATHS = [
    "/sys/class/thermal/thermal_zone0/temp",    # millidegrees C
    "/sys/class/thermal/thermal_zone1/temp",
]


def _read_first(paths, scale):
    for p in paths:
        try:
            with open(p) as f:
                return int(f.read().strip()) * scale
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            continue
    return None


def read_power_w() -> Optional[float]:
    """Wall-ish power in watts from the board INA226, or None off-box."""
    return _read_first(_POWER_PATHS, 1e-6)   # uW -> W


def read_temp_c() -> Optional[float]:
    """SoC temperature in C from the thermal zone, or None off-box."""
    return _read_first(_TEMP_PATHS, 1e-3)    # m°C -> °C


# ---- the aggregator -------------------------------------------------------
@dataclasses.dataclass
class _Window:
    """Rolling counters reset each tick (per-second rates) vs cumulative."""
    tokens: int = 0          # chars emitted this second
    busy_s: float = 0.0      # backend busy seconds this second
    filled_sum: int = 0      # sum of filled slots over batches this second
    cap_sum: int = 0         # sum of capacity over batches this second


class Telemetry:
    """Accumulates events; `tick(now)` collapses the window to one record.

    Cumulative counters (keystrokes/inferences/blits/enters) feed amplification
    and ready-at-Enter; per-second windows feed rates and occupancy. The server
    calls the on_* hooks from its hot path (cheap increments only) and calls
    tick() once a second from a scheduler task.
    """

    def __init__(self, active_window_s: float = 5.0, reservoir_size: int = 2048):
        self.active_window_s = active_window_s
        self.latency = Reservoir(reservoir_size)
        self.window = _Window()
        # cumulative
        self.keystrokes = 0
        self.inferences = 0
        self.blits = 0
        self.enters = 0
        self.fresh_hits = 0      # Enters that blitted (fresh buffer)
        # active-user tracking: client_id -> last keystroke monotonic
        self._last_seen: dict[str, float] = {}
        self._last_tick = time.monotonic()
        self.peak_active = 0

    # --- hot-path hooks (O(1)) ---
    def on_keystroke(self, client_id: str, now: float) -> None:
        self.keystrokes += 1
        self._last_seen[client_id] = now

    def on_inference(self, n_jobs: int) -> None:
        self.inferences += n_jobs

    def on_enter(self, blitted: bool) -> None:
        self.enters += 1
        if blitted:
            self.blits += 1
            self.fresh_hits += 1

    def on_batch(self, busy_s: float, filled: int, capacity: int,
                 tokens: int) -> None:
        self.window.busy_s += busy_s
        self.window.filled_sum += filled
        self.window.cap_sum += capacity
        self.window.tokens += tokens

    def add_latency(self, ms: float) -> None:
        self.latency.add(ms)

    # --- per-tick collapse ---
    def active_users(self, now: float) -> int:
        cutoff = now - self.active_window_s
        # prune while counting
        stale = [c for c, t in self._last_seen.items() if t < cutoff]
        for c in stale:
            del self._last_seen[c]
        n = len(self._last_seen)
        self.peak_active = max(self.peak_active, n)
        return n

    def tick(self, now: Optional[float] = None, queue_depth: int = 0) -> dict:
        if now is None:
            now = time.monotonic()
        dt = max(1e-6, now - self._last_tick)
        self._last_tick = now
        w = self.window
        occ_busy = min(1.0, w.busy_s / dt)
        fill = (w.filled_sum / w.cap_sum) if w.cap_sum else 0.0
        record = {
            "type": "telemetry",
            "ts": time.time(),
            "active_users": self.active_users(now),
            "peak_active": self.peak_active,
            "agg_tok_s": round(w.tokens / dt, 1),
            "latency_ms": self.latency.snapshot(),
            "queue_depth": queue_depth,
            "fabric_occupancy": round(occ_busy * (fill if fill else 1.0), 4),
            "occupancy_busy_frac": round(occ_busy, 4),
            "slot_fill_frac": round(fill, 4),
            "amplification": {
                "keystrokes": self.keystrokes,
                "inferences": self.inferences,
                "blits": self.blits,
                "ratio": (round(self.keystrokes / self.inferences, 2)
                          if self.inferences else None),
            },
            "ready_at_enter": (round(self.fresh_hits / self.enters, 4)
                               if self.enters else None),
            "enters": self.enters,
            "power_w": read_power_w(),
            "temp_c": read_temp_c(),
        }
        # reset per-second window
        self.window = _Window()
        return record


# ---- sinks ----------------------------------------------------------------
class JsonlSink:
    """Append-only local JSONL — the post-mortem record that survives the box."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


# definitions shipped to the dashboard so the page restates the method per metric
METRIC_DEFINITIONS = {
    "active_users": "Distinct clients with a keystroke in the last N seconds. Not total connections.",
    "agg_tok_s": "Chars emitted across all completions in the last second. Aggregate, not per-stream.",
    "latency_ms": "End-to-end ms, submit-eligible to delivery. Percentiles (p50/95/99), never the mean.",
    "queue_depth": "Prompts past debounce not yet submitted at tick time. Work waiting on the fabric.",
    "fabric_occupancy": "Backend busy-seconds / window, weighted by slot-fill. SOFTWARE PROXY until a hardware busy-cycle counter exists.",
    "amplification": "keystrokes : inferences : blits. The fusion made visible.",
    "ready_at_enter": "Fraction of Enters that blitted a fresh buffer with no round trip.",
    "power_w": "Board INA226 total rail power (W). None off-box.",
    "temp_c": "SoC thermal-zone temperature (C). None off-box.",
}
