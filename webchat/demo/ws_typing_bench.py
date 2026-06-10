"""Realistic typing bench: send growing-prefix keystrokes from one client (so
the server's prefix-KV session engages) and measure keystroke->speculation RTT."""
import asyncio, json, time, statistics
import websockets

async def bench(url, label):
    text = 'once upon time there be little girl who love play'
    lats = []
    async with websockets.connect(url) as ws:
        json.loads(await ws.recv())  # hello
        for i in range(5, len(text) + 1):
            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "keystroke", "prompt": text[:i], "seq": i}))
            m = json.loads(await asyncio.wait_for(ws.recv(), 30))
            lats.append((time.perf_counter() - t0) * 1000)
    lats = lats[2:]  # skip session-cold first hits
    print(f'{label}: median {statistics.median(lats):5.0f} ms  p90 {sorted(lats)[int(len(lats)*.9)]:5.0f}  '
          f'min {min(lats):5.0f}  max {max(lats):5.0f}   ({len(lats)} keystrokes)')
    print(f'   last spec: {m["completion"]!r}')

asyncio.run(bench('ws://127.0.0.1:8090', 'LOCAL'))
asyncio.run(bench('wss://chat.mikeayles.com', 'EDGE '))
