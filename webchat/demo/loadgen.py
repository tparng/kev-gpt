"""loadgen.py — synthetic typists to rehearse the death experiment.

M concurrent simulated users, each a WebSocket client typing with human-ish
cadence: 80-200 ms between keystrokes, occasional bursts (fast typist) and
pauses before Enter (most users hesitate). Each user types a short telegraphic
phrase char-by-char, sending a `keystroke` per char, then (after a pre-Enter
pause) sends `enter`. We measure CLIENT-SIDE latency — the time from a keystroke
to its matching speculation arriving — and the ready-at-Enter rate (did a fresh
speculation buffer exist when Enter fired?). That is the user-perceived magic,
and the number HN will feel.

This is how we rehearse before HN does it: run M=50..5000 and watch where p95
latency and ready-at-Enter fall off, and whether the server's telemetry shows
the death as fabric-bound (high occupancy) or plumbing-bound.

Run (against a local stub server):
    python -m webchat.demo.loadgen --url ws://127.0.0.1:8090 --users 50 --secs 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import websockets

_PHRASES = [
    "once upon time", "me want go fast", "tiny model big claim", "few word do trick",
    "kevin run hot", "no ddr just chip", "type thing me finish", "fast fast fast",
    "what you do today", "story about cat", "me like cheese", "go to shop now",
]


class TypistStats:
    def __init__(self):
        self.spec_latencies: list[float] = []   # ms keystroke->speculation
        self.enters = 0
        self.ready_at_enter = 0                  # fresh buffer existed on Enter
        self.keystrokes = 0


async def typist(url: str, stats: TypistStats, deadline: float, rng: random.Random):
    try:
        async with websockets.connect(url, max_size=64 * 1024,
                                      ping_interval=20) as ws:
            # drain the hello
            await asyncio.wait_for(ws.recv(), timeout=5)

            # state shared between send loop and recv loop for this typist
            pending: dict[str, float] = {}   # prompt -> send monotonic
            buffer = {"prompt": None, "completion": None}

            async def receiver():
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        t = msg.get("type")
                        if t in ("speculation", "authoritative"):
                            p = msg.get("prompt", "")
                            sent = pending.pop(p, None)
                            if sent is not None:
                                stats.spec_latencies.append(
                                    (time.monotonic() - sent) * 1000.0)
                            buffer["prompt"] = p
                            buffer["completion"] = msg.get("completion")
                except websockets.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(receiver())
            seq = 0
            while time.monotonic() < deadline:
                phrase = rng.choice(_PHRASES)
                typed = ""
                burst = rng.random() < 0.25   # a quarter of phrases typed fast
                for ch in phrase:
                    typed += ch
                    seq += 1
                    stats.keystrokes += 1
                    pending[typed] = time.monotonic()
                    try:
                        await ws.send(json.dumps({"type": "keystroke",
                                                  "prompt": typed, "seq": seq}))
                    except websockets.ConnectionClosed:
                        return
                    gap = rng.uniform(0.04, 0.09) if burst else rng.uniform(0.08, 0.20)
                    await asyncio.sleep(gap)
                    if time.monotonic() >= deadline:
                        break
                # pre-Enter pause: most users hesitate
                await asyncio.sleep(rng.uniform(0.15, 0.6))
                # freshness: did a speculation for the EXACT final prompt arrive?
                stats.enters += 1
                if buffer["prompt"] == typed:
                    stats.ready_at_enter += 1
                seq += 1
                try:
                    await ws.send(json.dumps({"type": "enter",
                                              "prompt": typed, "seq": seq}))
                except websockets.ConnectionClosed:
                    return
                # think before the next message
                await asyncio.sleep(rng.uniform(0.3, 1.2))
            recv_task.cancel()
    except (OSError, websockets.WebSocketException, asyncio.TimeoutError):
        pass


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return round(s[k], 1)


async def run(args):
    deadline = time.monotonic() + args.secs
    stats = [TypistStats() for _ in range(args.users)]
    rng_master = random.Random(args.seed)
    tasks = []
    for i in range(args.users):
        r = random.Random(rng_master.randint(0, 2**31))
        tasks.append(asyncio.create_task(
            typist(args.url, stats[i], deadline, r)))
        # stagger connects so we don't SYN-flood the box at t=0
        await asyncio.sleep(args.ramp / max(1, args.users))
    await asyncio.gather(*tasks)

    all_lat = [x for s in stats for x in s.spec_latencies]
    enters = sum(s.enters for s in stats)
    ready = sum(s.ready_at_enter for s in stats)
    keys = sum(s.keystrokes for s in stats)
    print("=" * 60)
    print(f"loadgen summary: users={args.users} secs={args.secs}")
    print(f"  keystrokes sent       : {keys}")
    print(f"  speculations received : {len(all_lat)}")
    print(f"  spec latency p50/95/99 ms: "
          f"{pct(all_lat,0.5)} / {pct(all_lat,0.95)} / {pct(all_lat,0.99)}")
    print(f"  enters                : {enters}")
    rae = (ready / enters) if enters else None
    print(f"  ready-at-Enter        : {ready}/{enters} = "
          f"{round(rae,3) if rae is not None else None}")
    print("=" * 60)
    return {
        "users": args.users, "secs": args.secs, "keystrokes": keys,
        "speculations": len(all_lat),
        "spec_p50": pct(all_lat, 0.5), "spec_p95": pct(all_lat, 0.95),
        "spec_p99": pct(all_lat, 0.99),
        "enters": enters, "ready_at_enter": rae,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="synthetic typists / death rehearsal")
    ap.add_argument("--url", default="ws://127.0.0.1:8090")
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--ramp", type=float, default=2.0,
                    help="seconds to stagger all connects over")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
