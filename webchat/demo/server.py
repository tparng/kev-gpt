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
import json
import time
from typing import Optional

import websockets

from .backend import InferRequest
from .backend_stub import StubBackend
from .protocol import (MSG_AUTHORITATIVE, MSG_ENTER, MSG_KEYSTROKE,
                       MSG_SPECULATION, BatchAssembler, Coalescer, Reason,
                       freshness_blit)
from .telemetry import JsonlSink, Telemetry


class Hub:
    """Shared server state: connections, coalescer, assembler, telemetry."""

    def __init__(self, *, backend, debounce_ms: float, batch: int,
                 tick_ms: float, jsonl: str, push_url: Optional[str],
                 push_token: Optional[str], fabric_depth: int = 1):
        self.backend = backend
        self.co = Coalescer(debounce_ms=debounce_ms)
        self.asm = BatchAssembler(self.co, batch=batch)
        self.tel = Telemetry()
        self.tick_s = tick_ms / 1000.0
        self.sink = JsonlSink(jsonl)
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
        self.tel.on_batch(busy_s=outcome.busy_s, filled=outcome.filled,
                          capacity=outcome.capacity, tokens=tokens)

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
            if self.push_url:
                await self._push(rec)

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
                cs = hub.co.client(cid)
                if freshness_blit(cs, prompt):
                    # fresh buffer -> the client blits locally; record the hit
                    hub.tel.on_enter(blitted=True)
                    await ws.send(json.dumps({"type": "blit_ok", "prompt": prompt}))
                else:
                    # stale/absent -> one authoritative inference, no blit
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
    if args.backend == "stub":
        backend = StubBackend(latency_ms=args.latency_ms, capacity=args.batch)
    elif args.backend == "pl":
        from .backend_stub import PLSingleTokenBackend
        backend = PLSingleTokenBackend(lanes=args.lanes, fclk=args.fclk,
                                       gen_chars=args.gen_chars)
    else:
        raise SystemExit(f"unknown backend {args.backend!r}")

    hub = Hub(backend=backend, debounce_ms=args.debounce_ms, batch=args.batch,
              tick_ms=args.tick_ms, jsonl=args.jsonl, push_url=args.push_url,
              push_token=args.push_token, fabric_depth=args.fabric_depth)

    sub_task = asyncio.create_task(hub.submission_loop())
    tel_task = asyncio.create_task(hub.telemetry_loop())

    async def handler(ws):
        await client_handler(ws, hub)

    print(f"[server] backend={args.backend} debounce={args.debounce_ms}ms "
          f"batch={args.batch} tick={args.tick_ms}ms ws://{args.host}:{args.port}")
    async with websockets.serve(handler, args.host, args.port,
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
    ap.add_argument("--backend", choices=["stub", "pl"], default="stub")
    ap.add_argument("--debounce-ms", type=float, default=50.0,
                    help="idle debounce before a speculation fires (PRD: 40-60)")
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
    ap.add_argument("--jsonl", default="demo_telemetry.jsonl",
                    help="local JSONL telemetry record (post-mortem trail)")
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
