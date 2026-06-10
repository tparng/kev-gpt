import asyncio, json, time, statistics
import websockets

async def bench(url, label, n=6):
    lats, infers = [], []
    async with websockets.connect(url) as ws:
        json.loads(await ws.recv())  # hello
        for i in range(n):
            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "keystroke", "prompt": "once upon time " + "x" * i, "seq": i + 1}))
            m = json.loads(await asyncio.wait_for(ws.recv(), 30))
            lats.append((time.perf_counter() - t0) * 1000)
            infers.append(m.get("infer_ms", 0))
    print(f"{label}: min {min(lats):.0f}  median {statistics.median(lats):.0f}  max {max(lats):.0f} ms   server infer_ms median {statistics.median(infers):.0f}")

asyncio.run(bench("ws://127.0.0.1:8090", "LOCAL  ws://127.0.0.1:8090   "))
asyncio.run(bench("wss://chat.mikeayles.com", "EDGE   wss://chat.mikeayles.com"))
