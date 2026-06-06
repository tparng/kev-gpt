# PRD: live on-chip chat demo

> Authored by Mike, 2026-06-06. Engineering gap analysis appended at the bottom.

## One liner

A 3M-param INT4 transformer running entirely on one Zynq UltraScale+ (fabric does inference, the A53 just feeds it, an i7 laptop serves), answers appearing before you finish typing, with a live load dashboard as the centrepiece. It lives embedded in a mikeayles.com blog post that tells the story, and the post is the HN submission. Goal: either it soaks the load on one low-power chip, or it falls over fully instrumented and the post becomes the post-mortem. Both are wins.

## Goals

- Zero perceived latency on send.
- Live telemetry as the headline, not a footnote.
- Push a single chip until something gives, and have the charts to explain what gave.
- One honest efficiency number: active users per measured watt.

## Non-goals

- Model quality or being a useful assistant. Tiny model, coherent-ish, that is the charm.
- Accounts, auth, history, persistence.
- Horizontal scale or HA. Deliberate single-chip vertical-limit experiment, one box, one chip.
- Heavy moderation. Input length cap and basic rate sanity, nothing more.

## Architecture

```
Client (chat UI + dashboard)
   |  WebSocket
   v
Cloudflare edge  <----  dashboard served from edge (Worker / Durable Object)
   |  cloudflared tunnel                  ^
   v                                      |  ~1 Hz aggregate push
i7 laptop: cloudflared + async server     |
           + debounce + batch assembly ---+
   |  wired GigE (switch)
   v
Kria A53: thin daemon, unpack batch, one PL submission, return
   |  AXI
   v
PL fabric: batched inference core
```

Two planes, kept separate:

- **Inference plane.** i7 does connections, TLS, debounce and batch assembly. A53 is demoted to a daemon that only unpacks a batch, fires one PL submission over AXI, and returns results.
- **Telemetry plane.** i7 pushes a ~1 Hz aggregate to a Cloudflare Worker or Durable Object, dashboard served from the edge. Fan-out is Cloudflare's job, and the dashboard stays up when the box falls over.

i7, Kria and router all sit on a GigE switch so both the RPC link and the tunnel uplink are wired, not WiFi.

Keystroke flow: debounce, send current prompt, i7 coalesces into a batch, fabric completes, result streamed back to that client and buffered, Enter blits the buffer locally after a freshness check.

## Functional requirements

Chat:

- **Speculative inference.** On each debounced keystroke (40 to 60 ms idle), run a completion on the current prompt, stream it to the originating client, buffer it client-side.
- **Blit on Enter.** Render the buffered result locally, no round trip.
- **Freshness check.** If the buffer does not match the exact current input (fast typist outran the last speculation), fire one authoritative inference rather than blit a stale answer.
- **Coalesce and drop** keystroke bursts nobody will blit.

Dashboard (the centrepiece):

- **Live metrics.** Concurrent active users, aggregate tok/s, latency p50 / p95 / p99, queue depth, fabric utilisation, load over time, measured power and temp, amplification factor (keystrokes in vs inferences run vs blits shown), ready-at-Enter hit rate.
- **Honest definitions on the page.** "Active user = keystroke within last N seconds." Percentiles not averages. One line of method per metric.
- **Survives death.** Stats computed once per tick, pushed to the edge, keeps rendering last-known values with an explicit RIP state when the box is gone.

## The load experiment (the actual product)

- **The death I want is fabric-bound.** Queue climbs, p99 creeps, ready-at-Enter hit rate slides. A graceful, watchable slide.
- **The death to avoid is plumbing-bound** (connections, the box, the link). The i7 plus wired GigE exists to move the wall off the plumbing and onto either feeding the fabric or the fabric itself, both of which are legitimate inference-speed stories.
- **Proof of the flex is fabric utilisation at the moment of death.** Tip over near 100 percent fabric and the lede holds. Tip over at 40 percent while the shim or the box chokes and it is buried. So fabric occupancy is a first-class metric, and "death near 100 percent fabric" is the success criterion.
- **To keep the wall on the fabric:** async server, coalesce many keystrokes into one PL submission, pin cloudflared and the server to separate cores, tiny payloads.

## Latency

- Perceived latency is zero only while one round trip fits inside the typing gap (~100 ms worst case, more if the user pauses before Enter, which most do).
- The internal hops (server, GigE, shim, AXI) are noise. Two terms matter: the user's distance to the box, which I cannot control and which is transatlantic for US users, so the magic is strongest near home and degrades to a normal round trip far away. And inference latency under batching, which I do control.
- Levers: batch size, batch window, speculative completion length. A freshness miss costs a full extra round trip on Enter, so misses convert hidden latency into felt latency.

## Constraints

- **Throughput is a moving target.** Design and report against measured current values, not a fixed number.
- **A53 launch-rate** is the likely internal ceiling. Measure launches per second with and without coalescing.
- **Cloudflare.** Verify WebSockets over Tunnel at this connection count, and watch its own bot or DDoS mitigation throttling the flood. Relax protection on the route if needed.
- **Power.** Hero stat is measured wall power. The board is likely well under any round number I have quoted, which only helps the story.
- **Laptop hygiene during the event.** Mains not battery, sleep and USB selective suspend off, watch thermals so the 5550 does not throttle and cap serving.

## Risks and open questions

- Plumbing-bound death (i7, link, or A53 shim) rather than fabric-bound. Mitigate as above, or accept it and write that post honestly.
- Cloudflare throttles my own spike.
- WebSocket-over-tunnel limits at high concurrency.
- Thermal throttle, on the fabric under sustained occupancy or on the laptop. Instrument both temps.
- How hard to harden the i7 path versus just accepting whatever death comes.

## Stretch

- Persist a peak high-water-mark ("peak concurrent: N") so the trophy survives the death.

---

# Engineering gap analysis (Claude, 2026-06-06 — fabric reality vs this PRD)

Two hard gaps sit between today's fabric and "run a completion on the current
prompt." Everything else in the PRD is conventional software.

## Gap 1: the fabric computes single tokens, not completions

Today's record design (25,744.5 tok/s aggregate) runs **single-position
forwards**: one token in, next-token logits out, no K/V persists between calls.
A completion needs multi-token KV decode in fabric: per-stream K/V caches
across layers, attention over T>1, a position loop, and per-stream completion
state (streams finish at different lengths). The bit-true reference exists
(`model/goformer_kv.py`); the RTL does not. This is the demo's critical-path
RTL campaign.

## Gap 2: the window math is brutal at char-level

The model is char-level (vocab 193): **32 fabric positions = 32 characters.**
A usable chat exchange (prompt + telegraphic answer) realistically needs
T = 128–256 chars; the model was trained at block_size 256. KV cost per stream
at T=256, 4 layers, d=256:

| precision | KB/stream | 8 streams | 16 streams |
|---|---|---|---|
| INT8 K/V | 512 | 4 MB | 8 MB |
| K4/V4+Hadamard (measured +0.72% NLL) | ~288 | 2.3 MB | 4.6 MB |

On-chip BRAM is 144 tiles ≈ 633 KB total, with 132 used by the benchmark
design. **Chat KV at a usable window does not fit on-chip at any stream count
worth demoing → KV-in-DDR (`KV-DDR-100K.md`) is not the optional scale-up — it
is the demo's prerequisite.** The note's bandwidth math survives: at K4/V4 the
DDR read budget for even very high aggregate rates stays well under the
~6–7.5 GB/s ceiling. The pieces conveniently already exist: the KVarN result
(quantized KV gated in Python), the burst-layout analysis, and an HP-port DMA
design sketch.

## Phasing (proposed)

- **P0 — software plane, starts now, zero RTL dependency.** i7 async server
  (debounce/coalesce/batch), A53 daemon protocol, Cloudflare Worker dashboard,
  telemetry tick. Develop and load-test against the EXISTING single-token PL
  (or the A53 model) as a stand-in inference backend — the protocol, batching,
  and dashboard don't care that completions are stubbed.
- **P1 — fabric KV decode, the RTL campaign.** kv_dma burst engine + sim DDR
  latency model in the gate harness (the log's async-bug class returns
  otherwise), K4/V4+Hadamard quant-at-write RTL gated against an extended
  goformer_kv reference, position loop + per-stream sequencing in the
  sequencer. Weeks, gated stepwise like everything else.
- **P2 — integration + the load experiment.** Chat-variant bitstream, fabric
  occupancy counters (a hardware busy-cycle register — trivial, should ride
  along in any P1 build), powerapp/temp sampling on the A53, dashboard live.

## PRD points the fabric numbers inform

- **Batch geometry:** the fabric's natural batch is N=16 streams/pass. The i7's
  batch assembler should fill 16-slots; "fabric utilisation" ≈ filled-slot
  fraction × busy-cycle fraction.
- **Completion length lever:** at ~25–30 chars of telegraphic answer per
  inference and (say) 60 completions/s/stream-slot equivalent, the current
  aggregate token rate supports roughly **dozens of simultaneously typing
  users** before queueing — the dashboard's amplification metric will make the
  real number visible fast.
- **A53 launch rate:** the current driver reloads nothing per call (weights
  boot-streamed); a submission is ~20 register pokes + result reads. Thousands
  of launches/s should hold, but measure early (P0 stub can measure it).
