"""End-to-end load bench through the FULL public path (Cloudflare -> tunnel ->
Precision -> Kria fabric -> back). Each virtual user mirrors the real client:
types a phrase with human cadence (keystrokes fire speculations), then Enter
with the blitted flag, and AWAITS the user-visible reply — handling blit / stream
/ stream_end / authoritative exactly like the browser. We measure what a real
person feels: Enter -> reply-complete, Enter -> first char, and the system's
serving throughput (completions/sec) as a transient spike saturates the serial
fabric.

  python bench_e2e.py --url wss://chat.mikeayles.com/ --users 40 --secs 60 --ramp 12
"""
import argparse
import asyncio
import json
import random
import time

import websockets

PHRASES = ["once upon time", "little girl find dog", "one day big bear",
           "me want go fast", "story about cat", "tiny boy big dream",
           "kevin run hot today", "what you do now", "me like cheese lot",
           "go to shop buy milk", "she play in park", "he find magic stone"]


class Stat:
    def __init__(self):
        self.e2e = []          # Enter -> reply complete (ms), MISS path only
        self.ttfc = []         # Enter -> first streamed char (ms)
        self.blit_ms = []      # Enter -> local blit (instant path)
        self.enters = 0
        self.blits = 0
        self.misses = 0
        self.completed = 0     # replies fully received (any path)
        self.errors = 0
        self.complete_ts = []  # wall-clock of each completion (for the timeline)


async def user(url, st, deadline, rng, t0):
    try:
        async with websockets.connect(url, max_size=128 * 1024, ping_interval=20,
                                      open_timeout=20) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)   # hello
            buf = {"p": None, "c": None}
            waits = []          # pending miss replies: {prompt, t_enter, first, acc}
            seq = 0

            async def rx():
                try:
                    async for raw in ws:
                        m = json.loads(raw)
                        t = m.get("type")
                        now = time.perf_counter()
                        if t == "speculation":
                            buf["p"] = m.get("prompt"); buf["c"] = m.get("completion")
                        elif t == "stream":
                            w = next((w for w in waits if w["p"] == m.get("prompt")), None)
                            if w and w["first"] is None:
                                w["first"] = now
                                st.ttfc.append((now - w["t"]) * 1000)
                        elif t in ("stream_end", "authoritative"):
                            w = next((w for w in waits if w["p"] == m.get("prompt")), None)
                            if w:
                                st.e2e.append((now - w["t"]) * 1000)
                                st.completed += 1; st.complete_ts.append(now - t0)
                                waits.remove(w)
                            if t == "authoritative":
                                buf["p"] = m.get("prompt"); buf["c"] = m.get("completion")
                except Exception:
                    pass

            rxt = asyncio.create_task(rx())
            while time.perf_counter() < deadline:
                phrase = rng.choice(PHRASES)
                typed = ""
                fast = rng.random() < 0.3
                for ch in phrase:
                    typed += ch
                    seq += 1
                    try:
                        await ws.send(json.dumps({"type": "keystroke", "prompt": typed, "seq": seq}))
                    except Exception:
                        st.errors += 1; return
                    await asyncio.sleep(rng.uniform(0.04, 0.08) if fast else rng.uniform(0.08, 0.18))
                    if time.perf_counter() >= deadline:
                        break
                await asyncio.sleep(rng.uniform(0.15, 0.5))   # pre-Enter hesitation
                fresh = (buf["p"] == typed and buf["c"] is not None)
                seq += 1
                st.enters += 1
                tnow = time.perf_counter()
                try:
                    await ws.send(json.dumps({"type": "enter", "prompt": typed, "seq": seq, "blitted": fresh}))
                except Exception:
                    st.errors += 1; return
                if fresh:                       # instant local blit, no server reply awaited
                    st.blits += 1; st.completed += 1
                    st.blit_ms.append(0.0); st.complete_ts.append(tnow - t0)
                    buf["p"] = None
                else:
                    st.misses += 1
                    waits.append({"p": typed, "t": tnow, "first": None})
                await asyncio.sleep(rng.uniform(0.4, 1.3))    # think before next message
            await asyncio.sleep(3.0)            # let in-flight replies land
            rxt.cancel()
    except Exception:
        st.errors += 1


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))], 0)


async def run(a):
    t0 = time.perf_counter()
    deadline = t0 + a.secs
    stats = [Stat() for _ in range(a.users)]
    master = random.Random(a.seed)
    tasks = []
    for i in range(a.users):
        r = random.Random(master.randint(0, 2**31))
        tasks.append(asyncio.create_task(user(a.url, stats[i], deadline, r, t0)))
        await asyncio.sleep(a.ramp / max(1, a.users))   # transient ramp-up spike
    await asyncio.gather(*tasks)

    e2e = [x for s in stats for x in s.e2e]
    ttfc = [x for s in stats for x in s.ttfc]
    enters = sum(s.enters for s in stats)
    blits = sum(s.blits for s in stats)
    misses = sum(s.misses for s in stats)
    completed = sum(s.completed for s in stats)
    errors = sum(s.errors for s in stats)
    all_ct = sorted(x for s in stats for x in s.complete_ts)

    print("=" * 66)
    print(f"E2E BENCH  url={a.url}  users={a.users}  secs={a.secs}  ramp={a.ramp}")
    print("=" * 66)
    print(f"enters {enters} | completed {completed} | blits {blits} "
          f"({100*blits/max(1,enters):.0f}%) | misses {misses} | errors {errors}")
    print(f"throughput: {completed/a.secs:.1f} replies/s avg")
    print(f"MISS (streamed) reply latency ms  p50/p95/p99: "
          f"{pct(e2e,.5)} / {pct(e2e,.95)} / {pct(e2e,.99)}  (n={len(e2e)})")
    print(f"MISS time-to-first-char ms        p50/p95/p99: "
          f"{pct(ttfc,.5)} / {pct(ttfc,.95)} / {pct(ttfc,.99)}  (n={len(ttfc)})")
    # 5-second throughput timeline (transient response)
    print("timeline (replies per 5s bucket):")
    if all_ct:
        nb = int(a.secs // 5) + 1
        buckets = [0] * nb
        for ct in all_ct:
            b = int(ct // 5)
            if 0 <= b < nb:
                buckets[b] += 1
        for b, c in enumerate(buckets):
            bar = "#" * c
            print(f"  {b*5:>3}-{b*5+5:<3}s | {c:>4} {bar}")
    print("=" * 66)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="wss://chat.mikeayles.com/")
    ap.add_argument("--users", type=int, default=40)
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--ramp", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
