"""server.py — the i7 asyncio inference-plane server.

This is the box that does connections, debounce, keystroke coalescing, 16-slot
batch assembly, completion streaming, and the freshness-on-Enter decision. It
owns NO model: inference is abstracted behind InferenceBackend (stub for off-box
load tests, PLSingleTokenBackend for the real fabric). The A53 is demoted to a
daemon the PL backend talks to; here we just call infer_batch.

Two cooperating asyncio loops:

* one task per WebSocket client, parsing keystroke/enter messages and pushing
  speculation/authoritative/telemetry frames back;
* one submission-loop task that wakes every `tick_ms`, asks the BatchAssembler
  for up to BATCH due jobs, fires ONE backend call per batch, then fans the
  results back to the originating clients.

Plus a 1 Hz telemetry task that collapses the window and POSTs it to the worker
+ appends JSONL. All the fiddly logic (debounce/coalesce/freshness/batch) lives
in protocol.py and is unit-tested; this file is the I/O shell.

Run (off-box, stub backend):
    python -m webchat.demo.server --backend stub --port 8090
Run (on the Kria, real single-token PL):
    sudo python -m webchat.demo.server --backend pl --lanes 128 --fclk 125e6
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import time
from typing import Optional

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response


def _rotation_state(path=os.environ.get("KEV_ROTATION_STATE", "/home/mike/rotation_state.json"),
                    ttl=90.0):
    """Active-model + countdown for the rotating demo, from a state file the rotation
    orchestrator pushes each switch. Returns {} (=> single model, not rotating) if the
    file is absent or stale — so the demo shows plain Kevin when nothing is rotating."""
    try:
        if time.time() - os.path.getmtime(path) > ttl:
            return {}
        d = json.load(open(path))
        ns = d.get("next_switch")
        return {"model": d.get("model"), "label": d.get("label"),
                "switch_in": max(0, int(ns - time.time())) if ns else None}
    except Exception:
        return {}

from .backend import InferRequest
from .backend_stub import StubBackend
from .protocol import (MSG_AUTHORITATIVE, MSG_ENTER, MSG_KEYSTROKE,
                       MSG_SPECULATION, MSG_STREAM, MSG_STREAM_END,
                       BatchAssembler, Coalescer, Reason, freshness_blit)
from .telemetry import JsonlSink, Telemetry


class Hub:
    """Shared server state: connections, coalescer, assembler, telemetry."""

    def __init__(self, *, backend, debounce_ms: float, batch: int,
                 tick_ms: float, jsonl: str, push_url: Optional[str],
                 push_token: Optional[str], fabric_depth: int = 1,
                 rollup_s: float = 60.0, rollup_jsonl: Optional[str] = None):
        self.backend = backend
        self.co = Coalescer(debounce_ms=debounce_ms)
        self.asm = BatchAssembler(self.co, batch=batch)
        self.tel = Telemetry()
        self.tick_s = tick_ms / 1000.0
        self.sink = JsonlSink(jsonl)
        # over-time rollup: a compact per-rollup_s line (latency percentiles +
        # request volume in the window) to its own JSONL, also printed for the log.
        self.rollup_s = rollup_s
        self.rollup_sink = JsonlSink(rollup_jsonl) if rollup_jsonl else None
        self._rollup_at = time.monotonic() + rollup_s
        self.push_url = push_url
        self.push_token = push_token
        # The fabric is ONE resource: it runs `fabric_depth` batches at once
        # (1 = strictly serial, the honest single-server model; >1 models an A53
        # that assembles the next batch while the PL drains the current one). When
        # all slots are busy, due jobs accumulate -> queue_depth climbs and the
        # death is fabric-bound, which is the PRD's success criterion. It is also
        # required for the real PL backend: concurrent infer_batch calls would race
        # the single PL over /dev/mem.
        self.fabric_depth = max(1, fabric_depth)
        self._inflight_n = 0
        # lifetime aggregates feeding the chat-page stat panel (the "flexes"):
        # total chars emitted, peak 1s aggregate rate, and a wall-clock start so
        # the panel can show an average tok/s since boot.
        self._start_mono = time.monotonic()
        self.total_tokens = 0
        self.peak_tok_s = 0.0
        # the two ceilings, smoothed over recent completions: FABRIC tok/s (pure
        # PL decode, ~16k) vs ROUND-TRIP tok/s (host loop + readback + sampling,
        # ~1k). EMA so the chat-page tiles don't jitter per request.
        self.fabric_tok_s = 0.0
        self.trip_tok_s = 0.0
        # rolling unique-visitor log: (open_ts, visitor_key) per connection, keyed
        # by the real client IP (CF-Connecting-IP behind the tunnel) so a refresh /
        # reconnect from the same person counts once. Pruned to the 30-min window.
        self._visits: "collections.deque[tuple[float, str]]" = collections.deque()
        self.visit_window_s = 1800.0
        # client_id -> websocket (for fan-out of completions)
        self.conns: dict[str, object] = {}
        # client_id -> the submit-eligible (debounce-expiry) perf_counter instant.
        # Latency is measured from here, so it includes queue wait under load.
        self._eligible_perf: dict[str, float] = {}
        # job tracking: client_id -> eligibility instant snapshotted at assembly
        self._submit_t: dict[str, float] = {}
        self._next_id = 0
        self._http = None   # lazy aiohttp session for telemetry push

    def new_client_id(self) -> str:
        self._next_id += 1
        return f"c{self._next_id}"

    def _note_rates(self, tokens: int, passes: int, busy_s: float, fabric_s) -> None:
        """EMA-update the two ceilings from a finished request: round-trip =
        delivered chars / wall-time; fabric = forward PASSES / pure-PL time
        (passes, not chars, so it matches the per-token decode record)."""
        a = 0.3
        if tokens and busy_s and busy_s > 0:
            r = tokens / busy_s
            self.trip_tok_s = r if self.trip_tok_s == 0 else (1 - a) * self.trip_tok_s + a * r
        if passes and fabric_s and fabric_s > 0:
            r = passes / fabric_s
            self.fabric_tok_s = r if self.fabric_tok_s == 0 else (1 - a) * self.fabric_tok_s + a * r

    def note_visit(self, key: str, now: float) -> None:
        self._visits.append((now, key))

    def unique_visitors(self, now: float) -> int:
        """Distinct visitors (by IP) seen in the last `visit_window_s` — a real
        reach number, not the instantaneous 0/1. Prunes the window first."""
        cutoff = now - self.visit_window_s
        while self._visits and self._visits[0][0] < cutoff:
            self._visits.popleft()
        return len({k for _, k in self._visits})

    async def send(self, client_id: str, obj: dict) -> None:
        ws = self.conns.get(client_id)
        if ws is None:
            return
        try:
            await ws.send(json.dumps(obj))
        except Exception:
            pass

    # ---- submission loop -------------------------------------------------
    async def submission_loop(self):
        while True:
            now = time.monotonic()
            # Only submit if a fabric slot is free; otherwise due jobs wait in the
            # coalescer (and show up as queue_depth) until the fabric drains.
            if self._inflight_n < self.fabric_depth:
                jobs = self.asm.assemble(now)
                if jobs:
                    self._inflight_n += 1
                    self.tel.on_inference(len(jobs))
                    submit_perf = time.perf_counter()
                    for j in jobs:
                        # Snapshot the submit-eligible instant at assembly so a
                        # later keystroke for this client can't move it. Falls back
                        # to now if we never saw the keystroke (e.g. refire path).
                        self._submit_t[j.client_id] = self._eligible_perf.get(
                            j.client_id, submit_perf)
                    reqs = [InferRequest(client_id=j.client_id, prompt=j.prompt,
                                         seq=j.seq) for j in jobs]
                    # one backend call == one PL submission for the whole batch
                    asyncio.create_task(self._run_batch(reqs, jobs, submit_perf))
            await asyncio.sleep(self.tick_s)

    async def _run_batch(self, reqs, jobs, submitted_at):
        # STREAMING PATH: a lone AUTHORITATIVE job is the Enter freshness-miss —
        # stream it char-by-char so the first char shows after one token instead
        # of the full ~100ms reply. Speculations and multi-job batches keep the
        # one-shot path (a precomputed spec is blitted whole; streaming it buys
        # nothing). _run_stream owns the in-flight decrement for this branch.
        if (len(jobs) == 1 and jobs[0].reason is Reason.AUTHORITATIVE
                and hasattr(self.backend, "infer_stream")):
            await self._run_stream(reqs[0], jobs[0], submitted_at)
            return
        try:
            outcome = await self.backend.infer_batch(reqs)
        except Exception:
            # A backend failure must not strand these clients as in-flight forever
            # (they'd never speculate again). Release them so the next keystroke
            # re-queues; the fabric slot is freed in the finally below.
            for j in jobs:
                self.asm.on_result(j.client_id, None)
            return
        finally:
            self._inflight_n -= 1
        deliver_t = time.perf_counter()
        tokens = 0
        reason_by_client = {j.client_id: j.reason for j in jobs}
        for res in outcome.results:
            self.asm.on_result(res.client_id, res.prompt)
            tokens += len(res.completion)
            elig = self._submit_t.get(res.client_id, submitted_at)
            self.tel.add_latency((deliver_t - elig) * 1000.0)
            kind = (MSG_AUTHORITATIVE
                    if reason_by_client.get(res.client_id) is Reason.AUTHORITATIVE
                    else MSG_SPECULATION)
            await self.send(res.client_id, {
                "type": kind,
                "prompt": res.prompt,
                "completion": res.completion,
                "infer_ms": round(outcome.busy_s * 1000.0, 2),
            })
        self.total_tokens += tokens
        self._note_rates(tokens, outcome.passes, outcome.busy_s, outcome.fabric_s)
        self.tel.on_batch(busy_s=outcome.busy_s, filled=outcome.filled,
                          capacity=outcome.capacity, tokens=tokens)

    async def _run_stream(self, req, job, submitted_at):
        """Stream one Enter-miss completion: forward each fabric char to the
        client as a `stream` frame, then a `stream_end` with the tidied final.
        Owns the in-flight slot release (the streaming branch of _run_batch
        skips that method's finally)."""
        cid = job.client_id

        async def on_chunk(text):
            await self.send(cid, {"type": MSG_STREAM, "prompt": job.prompt,
                                  "text": text})
        try:
            completion, busy_s, fabric_s, ntok, npass = await self.backend.infer_stream(
                req, on_chunk)
        except Exception:
            # release the client so a later Enter re-queues, and close the bubble
            self.asm.on_result(cid, None)
            await self.send(cid, {"type": MSG_STREAM_END, "prompt": job.prompt,
                                  "completion": ""})
            return
        finally:
            self._inflight_n -= 1
        deliver_t = time.perf_counter()
        self.asm.on_result(cid, job.prompt)
        elig = self._submit_t.get(cid, submitted_at)
        self.tel.add_latency((deliver_t - elig) * 1000.0)
        tok = ntok or len(completion)
        self.total_tokens += tok
        self._note_rates(tok, npass, busy_s, fabric_s)
        self.tel.on_batch(busy_s=busy_s, filled=1,
                          capacity=getattr(self.backend, "capacity", 1),
                          tokens=tok)
        await self.send(cid, {"type": MSG_STREAM_END, "prompt": job.prompt,
                              "completion": completion})

    # ---- telemetry loop --------------------------------------------------
    async def telemetry_loop(self):
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            qd = self.asm.queue_depth(now)
            rec = self.tel.tick(now=now, queue_depth=qd)
            rec["connections"] = len(self.conns)
            rec["dropped_keystrokes"] = self.co.dropped
            self.sink.write(rec)
            await self._broadcast_stats(rec, now)
            if self.push_url:
                await self._push(rec)
            # ---- over-time rollup (latency + request volume) every rollup_s ----
            if now >= self._rollup_at:
                self._rollup_at = now + self.rollup_s
                r = self.tel.rollup(now)
                r["visitors_30m"] = self.unique_visitors(now)
                r["connections"] = len(self.conns)
                if self.rollup_sink:
                    self.rollup_sink.write(r)
                p50 = r["lat_p50"]; p95 = r["lat_p95"]
                print(f"[metrics] {time.strftime('%H:%M:%S')} "
                      f"req={r['requests']} ({r['req_per_min']}/min) "
                      f"lat p50={p50}ms p95={p95}ms (n={r['lat_n']}) "
                      f"tok/s={r['tok_s']} peak={r['peak_concurrent']} "
                      f"visitors30m={r['visitors_30m']}", flush=True)

    # (defined below at module scope: _rotation_state)
    async def _broadcast_stats(self, rec: dict, now: float):
        """Push a compact, chat-page-shaped stat frame to every connected client.

        The dashboard Worker gets the full telemetry record; the chat page only
        needs the handful of headline 'flex' numbers, pre-shaped so the client
        renders without re-deriving anything. Same honest definitions as the
        dashboard: percentiles not means, duty cycle a labelled software proxy."""
        if not self.conns:
            return
        agg = rec.get("agg_tok_s") or 0.0
        if agg > self.peak_tok_s:
            self.peak_tok_s = agg
        uptime = max(1e-6, now - self._start_mono)
        amp = rec.get("amplification") or {}
        rot = _rotation_state()
        stats = {
            "type": "stats",
            "model": rot.get("model", "kevin"),
            "model_label": rot.get("label", "Kevin — telegraphic, full speed"),
            "rotating": bool(rot),
            "switch_in": rot.get("switch_in"),
            "live_users": rec.get("active_users"),
            "peak_users": rec.get("peak_active"),
            "users_30m": self.unique_visitors(now),
            "fabric_tok_s": round(self.fabric_tok_s, 0),
            "trip_tok_s": round(self.trip_tok_s, 0),
            "agg_tok_s": agg,
            "peak_tok_s": round(self.peak_tok_s, 1),
            "avg_tok_s": round(self.total_tokens / uptime, 1),
            "duty_cycle": rec.get("occupancy_busy_frac"),
            "requests": amp.get("inferences"),
            "amp_ratio": amp.get("ratio"),
            "total_tokens": self.total_tokens,
            "ready_at_enter": rec.get("ready_at_enter"),
            "uptime_s": round(uptime, 0),
        }
        payload = json.dumps(stats)
        for ws in list(self.conns.values()):
            try:
                await ws.send(payload)
            except Exception:
                pass

    async def _push(self, rec: dict):
        try:
            import aiohttp
        except ImportError:
            return
        if self._http is None:
            self._http = aiohttp.ClientSession()
        headers = {"Content-Type": "application/json"}
        if self.push_token:
            headers["Authorization"] = f"Bearer {self.push_token}"
        try:
            async with self._http.post(self.push_url, data=json.dumps(rec),
                                       headers=headers,
                                       timeout=aiohttp.ClientTimeout(total=2)) as r:
                await r.read()
        except Exception:
            pass   # telemetry push is best-effort; never block the demo on it


async def client_handler(ws, hub: Hub):
    """One task per WebSocket connection. Parses the client protocol and pushes
    completions/telemetry. Keystroke -> coalescer; Enter -> freshness check
    (blit locally if fresh else queue an authoritative inference)."""
    cid = hub.new_client_id()
    hub.conns[cid] = ws
    # the real visitor key: CF-Connecting-IP (set by Cloudflare, forwarded by the
    # tunnel) so a refresh from the same person is one visitor; fall back to the
    # forwarded-for first hop, then the per-connection id for local/no-proxy runs.
    vkey = cid
    try:
        hdrs = ws.request.headers
        vkey = (hdrs.get("CF-Connecting-IP")
                or (hdrs.get("X-Forwarded-For") or "").split(",")[0].strip()
                or cid)
    except Exception:
        pass
    hub.note_visit(vkey, time.monotonic())
    try:
        await ws.send(json.dumps({"type": "hello", "client_id": cid}))
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            now = time.monotonic()
            mtype = msg.get("type")
            prompt = str(msg.get("prompt", ""))[:512]
            seq = int(msg.get("seq", 0))
            if mtype == MSG_KEYSTROKE:
                hub.tel.on_keystroke(cid, now)
                hub.co.on_keystroke(cid, prompt, seq, now)
                # the job becomes submit-eligible one debounce window from now;
                # latency is measured from there (PRD: from the submit-eligible
                # instant), so it excludes the debounce but includes queue wait.
                hub._eligible_perf[cid] = time.perf_counter() + hub.co.debounce_s
            elif mtype == MSG_ENTER:
                # THE CLIENT IS THE AUTHORITY on blit-vs-reply. It alone knows
                # whether its buffer held the EXACT prompt; the server's own
                # last_speculated can disagree (e.g. a stream updates it but not
                # the client's buffer), which used to leave a "…" with no reply
                # ever coming. So trust msg.blitted: if the client blitted, just
                # count it; otherwise ALWAYS queue a reply. (A legacy client
                # without the flag falls back to the server freshness check.)
                if "blitted" in msg:
                    blitted = bool(msg["blitted"])
                else:
                    blitted = freshness_blit(hub.co.client(cid), prompt)
                if blitted:
                    hub.tel.on_enter(blitted=True)
                else:
                    hub.tel.on_enter(blitted=False)
                    hub.co.on_enter_refire(cid, prompt, seq, now)
                    hub._eligible_perf[cid] = time.perf_counter()  # eligible now
    except websockets.ConnectionClosed:
        pass
    finally:
        hub.conns.pop(cid, None)
        hub.co.clients.pop(cid, None)
        hub._submit_t.pop(cid, None)
        hub._eligible_perf.pop(cid, None)


async def serve(args):
    # Lift the open-file soft limit toward the hard ceiling so the server can hold
    # many concurrent WebSocket connections (one fd each) during a traffic spike
    # without hitting EMFILE. The 16-slot batch time-slices the fabric across the
    # active typists among them (the "pseudo-parallelism"); idle lurkers cost only
    # an fd, so headroom here is what lets us push the serving ceiling.
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(65536, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f"[server] raised NOFILE soft limit {soft} -> {target}")
    except Exception as e:
        print(f"[server] could not raise NOFILE limit: {e}")

    if args.backend == "stub":
        backend = StubBackend(latency_ms=args.latency_ms, capacity=args.batch)
    elif args.backend == "pl":
        # local /dev/mem PL — only valid when the server runs ON the Kria.
        from .backend_stub import PLSingleTokenBackend
        backend = PLSingleTokenBackend(lanes=args.lanes, fclk=args.fclk,
                                       gen_chars=args.gen_chars)
    elif args.backend == "tcp":
        # the real split: server on the Precision, fabric on the Kria, reached
        # over the network (Tailscale/GigE) via the A53 daemon.
        if not args.daemon_host:
            raise SystemExit("--backend tcp needs --daemon-host (the Kria address)")
        from .a53_daemon import TcpPLBackend
        backend = TcpPLBackend(args.daemon_host, args.daemon_port,
                               use_msgpack=(args.daemon_transport == "msgpack"),
                               capacity=args.batch)
    elif args.backend == "kv":
        # real model-faithful MULTI-TOKEN Kevin, in-process (host-KV path). numpy
        # off-box; pl/c on the Kria (server-on-board single-box). Not the 60k
        # single-token fabric pass — see backend_kv.py / PRD gap 1.
        from .backend_kv import KVChatBackend
        backend = KVChatBackend(gemv_backend=args.kv_backend, capacity=args.batch,
                                gen_chars=args.gen_chars, greedy=args.greedy)
    else:
        raise SystemExit(f"unknown backend {args.backend!r}")

    # default the rollup log next to the per-second jsonl: live.jsonl -> live.rollup.jsonl
    rollup_jsonl = args.rollup_jsonl
    if rollup_jsonl is None and args.jsonl:
        base, ext = os.path.splitext(args.jsonl)
        rollup_jsonl = f"{base}.rollup{ext or '.jsonl'}"
    hub = Hub(backend=backend, debounce_ms=args.debounce_ms, batch=args.batch,
              tick_ms=args.tick_ms, jsonl=args.jsonl, push_url=args.push_url,
              push_token=args.push_token, fabric_depth=args.fabric_depth,
              rollup_s=args.rollup_s, rollup_jsonl=rollup_jsonl)

    sub_task = asyncio.create_task(hub.submission_loop())
    tel_task = asyncio.create_task(hub.telemetry_loop())

    async def handler(ws):
        await client_handler(ws, hub)

    # Serve the chat page over plain HTTP on the SAME origin as the WebSocket, so
    # a single tunnel hostname (e.g. chat.mikeayles.com via cloudflared) hands out
    # both the UI and the wss:// upgrade. A browser GET gets client.html; a WS
    # upgrade request is passed straight through to the handshake.
    client_path = args.client_html or os.path.join(os.path.dirname(__file__),
                                                    "client.html")
    try:
        with open(client_path, "rb") as f:
            client_html = f.read()
    except OSError:
        client_html = None

    def process_request(connection, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            # Survival backstop: past the cap, refuse new WS upgrades with a 503 so
            # the server process stays alive (idle lurkers still cost one fd each)
            # instead of exhausting fds and crashing the whole demo. The client
            # auto-reconnects, so a shed visitor keeps retrying until a slot frees.
            if args.max_clients and len(hub.conns) >= args.max_clients:
                body = b"at capacity\n"
                return Response(503, "Service Unavailable", Headers([
                    ("Content-Type", "text/plain; charset=utf-8"),
                    ("Content-Length", str(len(body))),
                    ("Retry-After", "3"),
                ]), body)
            return None   # not an HTTP page request -> let the WS handshake run
        if client_html is not None and request.path.split("?")[0] in ("/", "/client.html"):
            return Response(200, "OK", Headers([
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(client_html))),
                ("Cache-Control", "no-cache"),
            ]), client_html)
        body = b"not found\n"
        return Response(404, "Not Found", Headers([
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ]), body)

    print(f"[server] backend={args.backend} debounce={args.debounce_ms}ms "
          f"batch={args.batch} tick={args.tick_ms}ms ws://{args.host}:{args.port} "
          f"(serving client.html: {client_html is not None})")
    async with websockets.serve(handler, args.host, args.port,
                                process_request=process_request,
                                max_size=64 * 1024, ping_interval=20):
        # run until a stop signal (or forever)
        stop = asyncio.Future()
        if args.run_for > 0:
            asyncio.get_running_loop().call_later(args.run_for, stop.set_result, None)
        try:
            await stop
        finally:
            sub_task.cancel(); tel_task.cancel()
            await backend.close()
            hub.sink.close()


def build_parser():
    ap = argparse.ArgumentParser(description="i7 inference-plane WebSocket server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-clients", type=int, default=0,
                    help="survival backstop: cap concurrent WS connections "
                         "(0 = unlimited). Over the cap, new WS upgrades get a 503 "
                         "(client auto-reconnects) so the process survives a spike "
                         "instead of exhausting file descriptors and crashing.")
    ap.add_argument("--backend", choices=["stub", "pl", "tcp", "kv"],
                    default="stub",
                    help="stub=fake; pl=local single-token /dev/mem; tcp=remote "
                         "Kria daemon (Precision-serves split); kv=in-process "
                         "model-faithful multi-token Kevin (--kv-backend)")
    ap.add_argument("--kv-backend", choices=["numpy", "pl", "c"], default="numpy",
                    help="GEMV backend for --backend kv: numpy off-box, pl/c on "
                         "the Kria")
    ap.add_argument("--greedy", action="store_true",
                    help="--backend kv: argmax decode (reproducible) vs sampled")
    ap.add_argument("--daemon-host", default=None,
                    help="Kria A53 daemon address for --backend tcp "
                         "(e.g. the Tailscale IP <kria-ip>)")
    ap.add_argument("--daemon-port", type=int, default=9099,
                    help="Kria A53 daemon port (a53_daemon --port)")
    ap.add_argument("--daemon-transport", choices=["json", "msgpack"],
                    default="json",
                    help="frame codec; MUST match the daemon (run it with --json "
                         "for json, default msgpack-if-available otherwise)")
    ap.add_argument("--debounce-ms", type=float, default=20.0,
                    help="idle debounce before a speculation fires. The client "
                         "already coalesces keystrokes (~20ms), so this is just a "
                         "small server-side settle; with faithful infer ~20ms and "
                         "the gigabit hop ~1ms it's the dominant controllable "
                         "latency, so keep it tight (was 50 per the old PRD).")
    ap.add_argument("--batch", type=int, default=16,
                    help="max streams per PL submission (fabric natural N=16)")
    ap.add_argument("--tick-ms", type=float, default=10.0,
                    help="submission-loop period; how often batches are assembled")
    ap.add_argument("--fabric-depth", type=int, default=1,
                    help="batches the fabric runs at once (1 = serial single-server; "
                         ">1 models A53 pipelining). The single resource that makes "
                         "the death fabric-bound.")
    ap.add_argument("--latency-ms", type=float, default=8.0,
                    help="stub backend per-batch modelled latency")
    ap.add_argument("--lanes", type=int, default=128, help="PL backend PE width")
    ap.add_argument("--fclk", type=float, default=125e6, help="PL backend clock Hz")
    ap.add_argument("--gen-chars", type=int, default=6,
                    help="PL backend completion length (chars; T=1 stub)")
    ap.add_argument("--client-html", default=None,
                    help="path to client.html to serve over HTTP on the same "
                         "origin as the WS (default: the sibling client.html)")
    ap.add_argument("--jsonl", default="demo_telemetry.jsonl",
                    help="local JSONL telemetry record (post-mortem trail)")
    ap.add_argument("--rollup-s", type=float, default=60.0,
                    help="seconds per over-time rollup line (latency percentiles "
                         "+ request volume in that window); also printed to the log")
    ap.add_argument("--rollup-jsonl", default=None,
                    help="rollup log path (default: <jsonl>.rollup.<ext>)")
    ap.add_argument("--push-url", default=None,
                    help="Cloudflare Worker ingest URL for 1 Hz telemetry POST")
    ap.add_argument("--push-token", default=None,
                    help="bearer token for the worker ingest route")
    ap.add_argument("--run-for", type=float, default=0.0,
                    help="seconds to run then exit (0 = forever); used by tests")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
