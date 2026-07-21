#!/usr/bin/env python3
"""Slam chat.mikeayles.com with ramping concurrency: each virtual user types
char-by-char (keystroke: o,on,onc,one...) then Enter. Logs every event to SQLite:
per-message round-trip + TTFT + completion, speculation receipts, and server
telemetry (fabric util / queue / tok-s). Model is correlated post-hoc by timestamp
against the Kria rotation event log.
  python3 loadtest.py <dur_s> <max_clients> <db_path>
"""
import asyncio, json, time, sqlite3, random, sys
import websockets

URL = "wss://chat.mikeayles.com/"
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 900
MAXC = int(sys.argv[2]) if len(sys.argv) > 2 else 60
DB = sys.argv[3] if len(sys.argv) > 3 else "loadtest.db"
PROMPTS = ["once upon a time", "one day a little", "the big dog ran", "she went to the",
           "a small boy named", "there was a cat", "tom and his friend", "lily loved to play",
           "the sun was bright", "in the forest there", "he wanted to find", "they played all day"]

def now(): return time.time()
def iso(t=None):
    t = t or time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ("%.3f" % (t % 1))[1:] + "Z"

Q = asyncio.Queue()
async def logE(client, kind, seq=0, prompt="", rtt=None, ttft=None, infer=None, chars=None, comp=None):
    await Q.put(("e", (now(), iso(), client, kind, seq, prompt, rtt, ttft, infer, chars, (comp or "")[:280])))

async def writer(conn):
    batch = []
    last = now()
    while True:
        try:
            item = await asyncio.wait_for(Q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            item = "FLUSH"
        if item is None:
            break
        if item != "FLUSH":
            batch.append(item)
        if batch and (len(batch) >= 200 or now() - last > 0.5 or item in (None, "FLUSH")):
            conn.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                             [r for tb, r in batch if tb == "e"])
            conn.executemany("INSERT INTO telemetry VALUES(?,?,?)",
                             [r for tb, r in batch if tb == "t"])
            conn.commit(); batch = []; last = now()

async def client(cid, stop_at):
    seq = 0
    try:
        async with websockets.connect(URL, open_timeout=15, max_size=2**20, ping_interval=20, ping_timeout=20) as ws:
            await asyncio.wait_for(ws.recv(), timeout=15)  # hello
            await logE(cid, "connect")
            pend = {}   # prompt -> {t0, ttft}  (enter only)
            async def recv_loop():
                async for raw in ws:
                    t = now()
                    try: m = json.loads(raw)
                    except Exception: continue
                    typ = m.get("type"); p = m.get("prompt", "")
                    if typ == "stats":
                        await Q.put(("t", (t, iso(t), raw[:2500]))); continue
                    if typ == "speculation":
                        await logE(cid, "speculation", prompt=p, infer=m.get("infer_ms"),
                                   chars=len(m.get("completion", "")), comp=m.get("completion"))
                    elif typ == "stream":
                        pd = pend.get(p)
                        if pd and pd.get("ttft") is None: pd["ttft"] = (t - pd["t0"]) * 1000
                    elif typ in ("authoritative", "stream_end"):
                        pd = pend.pop(p, None)
                        await logE(cid, typ, prompt=p,
                                   rtt=(t - pd["t0"]) * 1000 if pd else None,
                                   ttft=pd.get("ttft") if pd else None, infer=m.get("infer_ms"),
                                   chars=len(m.get("completion", "")), comp=m.get("completion"))
            rt = asyncio.create_task(recv_loop())
            while now() < stop_at:
                prompt = random.choice(PROMPTS); seq += 1
                for i in range(1, len(prompt) + 1):          # char-by-char: o, on, onc...
                    await ws.send(json.dumps({"type": "keystroke", "prompt": prompt[:i], "seq": seq}))
                    await logE(cid, "keystroke", seq=seq, prompt=prompt[:i])
                    await asyncio.sleep(random.uniform(0.03, 0.09))
                pend[prompt] = {"t0": now(), "ttft": None}    # Enter -> measure RTT/TTFT
                await ws.send(json.dumps({"type": "enter", "prompt": prompt, "seq": seq, "blitted": False}))
                await logE(cid, "enter_sent", seq=seq, prompt=prompt)
                await asyncio.sleep(random.uniform(0.4, 1.4))
            rt.cancel()
    except Exception as e:
        await logE(cid, "error", comp=repr(e)[:200])

async def main():
    conn = sqlite3.connect(DB)
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS events(ts REAL, iso TEXT, client INT, kind TEXT, seq INT,
        prompt TEXT, rtt_ms REAL, ttft_ms REAL, infer_ms REAL, chars INT, completion TEXT);
      CREATE TABLE IF NOT EXISTS telemetry(ts REAL, iso TEXT, raw TEXT);
      CREATE TABLE IF NOT EXISTS meta(k TEXT, v TEXT);""")
    conn.execute("INSERT INTO meta VALUES('start', ?)", (iso(),)); conn.commit()
    wt = asyncio.create_task(writer(conn))
    stop_at = now() + DUR
    ramp = max(0.5, (DUR * 0.5) / MAXC)   # ramp all clients in over the first half
    tasks = []
    for i in range(MAXC):
        if now() >= stop_at: break
        tasks.append(asyncio.create_task(client(i, stop_at)))
        print("clients: %d  (t+%.0fs)" % (i + 1, now() - (stop_at - DUR)), flush=True)
        await asyncio.sleep(ramp)
    await asyncio.gather(*tasks, return_exceptions=True)
    await Q.put(None); await wt
    conn.execute("INSERT INTO meta VALUES('end', ?)", (iso(),)); conn.commit(); conn.close()
    print("LOADTEST_DONE db=%s" % DB, flush=True)

asyncio.run(main())
