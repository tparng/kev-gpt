# Live chat demo — the software inference plane (PRD 5, phase P0)

This is the **software plane** of the live load demo from `5-demo-prd.md`: the i7
async server (debounce / keystroke-coalesce / 16-slot batch assembly / completion
streaming / freshness-on-Enter), the thin Kria-side A53 daemon, the 1 Hz
telemetry tick, the Cloudflare Worker dashboard, a synthetic-typist load
generator, and a minimal chat client. It has **zero RTL dependency**: inference
is abstracted behind `InferenceBackend`, so the whole plumbing is developed and
load-tested off-box against a deterministic stub, then pointed at the real PL.

Honest scope: the current fabric does **single-token** forwards with no KV cache
(attention T=1). So the PL-backed completions past the first character are **not
model-faithful** — they exist to exercise the real AXI/GigE path and measure the
A53 launch rate, not to produce real text. The stub backend produces deterministic
telegraphic gibberish for load tests. Real multi-token chat is the P1 RTL campaign
(see the PRD gap analysis).

## Files

| file | role |
|---|---|
| `protocol.py` | pure-python core: `Coalescer` (debounce + latest-wins coalescing), `BatchAssembler` (16-slot fill + queue depth), `freshness_blit`. Socket-free, unit-tested. |
| `backend.py` | `InferenceBackend` interface (`infer_batch` -> `BatchOutcome`). |
| `backend_stub.py` | `StubBackend` (deterministic fake completions + modelled latency) and `PLSingleTokenBackend` (drives the real N=16 single-token PL via `pl_seq_pp16`, **T=1 stub past first token**). |
| `server.py` | the i7 asyncio WebSocket server. One task/client + a submission loop + a telemetry loop. |
| `a53_daemon.py` | the Kria daemon: length-prefixed JSON / msgpack batch in -> one PL submission -> results out. `--bench` measures launches/s. |
| `telemetry.py` | the 1 Hz aggregator: active users, agg tok/s, latency reservoir p50/95/99, queue depth, fabric occupancy (software proxy), amplification, ready-at-Enter, power/temp (Kria sysfs, None off-box). JSONL sink. |
| `client.html` | minimal chat UI: debounced keystroke send, speculation buffer, blit-on-Enter, freshness re-fire, inline latency readout. |
| `loadgen.py` | synthetic typists (human-ish cadence, bursts, pre-Enter pauses); reports client-side spec latency p50/95/99 + ready-at-Enter. The death rehearsal. |
| `dashboard/` | Cloudflare Worker (`worker.js`, `wrangler.toml`) + `dashboard.html` (single-file, hand-rolled canvas charts, honest metric definitions, last-known + RIP state). |

## Run the whole loop locally (off-box, stub backend)

Three terminals (or background the first two):

```bash
# 1. the i7 server with the deterministic stub backend
python -m webchat.demo.server --backend stub --port 8090 --jsonl demo_telemetry.jsonl
#    add --push-url https://<worker>.workers.dev/ingest --push-token <secret> to feed the dashboard

# 2. the chat client: open webchat/demo/client.html in a browser.
#    It connects to ws://<host>:<port> by default (same host the page is served from).
#    For a file:// open, set window.WS_URL in the console:  WS_URL='ws://127.0.0.1:8090'
#    (or serve client.html from any static server on the same origin as the WS).

# 3. synthetic typists (the load / death rehearsal)
python -m webchat.demo.loadgen --url ws://127.0.0.1:8090 --users 50 --secs 60
```

`demo_telemetry.jsonl` is the local post-mortem record — one JSON line per second.

### Dashboard dev mode

```bash
cd webchat/demo/dashboard
npx wrangler dev          # serves the Worker + dashboard locally
# POST a fake record to see it render:
curl -X POST localhost:8787/ingest -H 'content-type: application/json' \
     -d '{"active_users":7,"agg_tok_s":1234,"latency_ms":{"p50":8,"p95":20,"p99":45},"queue_depth":0,"fabric_occupancy":0.4,"amplification":{"keystrokes":900,"inferences":120,"blits":40,"ratio":7.5},"ready_at_enter":0.9}'
# open localhost:8787/ in a browser
```

## On-box (the real topology)

```bash
# On the Kria (needs root for /dev/mem; chat bitstream loaded; weights boot-streamed):
sudo python -m webchat.demo.a53_daemon --port 9099 --lanes 128 --fclk 125e6

# On the i7 (if it has the device locally — it does NOT in the real split, so this
# is the in-process form for a single-box bring-up):
sudo python -m webchat.demo.server --backend pl --lanes 128 --fclk 125e6 --port 8090
```

In the real split the i7 has no `/dev/mem`; the `pl` backend talks JSON to the
A53 daemon over GigE. Wiring the i7 server to the daemon (a `TcpPLBackend`) is a
small adapter left for integration — the daemon protocol is already defined in
`a53_daemon.py`.

## The A53 launch-rate bench (the PRD's open question)

```bash
sudo python -m webchat.demo.a53_daemon --bench --secs 5 --lanes 128 --fclk 125e6
# -> BENCH_VERDICT launches_per_s=... batched_streams_per_s=...
```

Reports launches/s (one launch == one 16-stream PL pass) and the aggregate
stream-completions/s with batching vs without, so the coalescing win is explicit.

## Tests

```bash
python -m pytest tests/test_demo_protocol.py
```

Pure-python units (no sockets): debounce/coalesce, freshness, batch assembly,
telemetry aggregation.

## Open items a human must do

- **`wrangler deploy`** the dashboard Worker and `wrangler secret put INGEST_TOKEN`;
  then pass `--push-url`/`--push-token` to the server. (`dashboard/wrangler.toml`
  is deploy-ready.)
- **cloudflared tunnel** for the WebSocket chat route — and verify WebSockets over
  Tunnel at the target concurrency; relax bot/DDoS protection on the route if it
  throttles the spike.
- **GigE addressing**: put i7 + Kria + router on the wired switch; set the daemon
  `--host`/`--port` and the i7→daemon adapter to the wired addresses; confirm both
  the RPC link and the tunnel uplink are wired, not WiFi.
- **Laptop hygiene during the event** (mains, sleep/USB-suspend off, watch thermals).
- Wire the in-process `pl` backend to the A53 daemon over TCP for the real split
  (small `TcpPLBackend` adapter), once the chat bitstream + daemon are co-running.
```
