"""a53_daemon.py — the thin Kria-side batch daemon.

The PRD demotes the A53 to a daemon that does nothing but: accept a batch of
prompts over the wired GigE link from the i7, unpack it, fire ONE PL submission
over AXI, and return the results. No model state, no per-call weight reload
(weights are boot-streamed once at startup). The transport is length-prefixed
JSON by default (msgpack if available + selected) — tiny payloads, the PRD wants
the wall on the fabric not the link.

This daemon and the i7's PLSingleTokenBackend are two deployment shapes of the
same thing: you can run the PL backend IN the i7 process (if the i7 had /dev/mem,
which it doesn't), or split it so the i7 talks JSON to this daemon which owns the
device. The split is the real topology. The i7 server's "pl" backend can target
this daemon via the TcpPLBackend below.

--bench mode answers the PRD's open question directly: how many PL submissions
("launches") per second can the A53 sustain, with and without batching? It loops
_submit_once at full tilt and reports launches/s and the implied aggregate
chars/s, with batching (16 streams/launch) vs without (1 stream/launch,
modelled as 16x the launches for the same user-visible work).

Run on the Kria:
    sudo python -m webchat.demo.a53_daemon --port 9099 --lanes 128 --fclk 125e6
Bench only (no clients):
    sudo python -m webchat.demo.a53_daemon --bench --secs 5 --lanes 128 --fclk 125e6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time

# msgpack optional; fall back to length-prefixed JSON everywhere.
try:
    import msgpack            # type: ignore
    _HAVE_MSGPACK = True
except ImportError:           # pragma: no cover
    _HAVE_MSGPACK = False


def encode(obj: dict, use_msgpack: bool) -> bytes:
    body = (msgpack.packb(obj, use_bin_type=True) if use_msgpack
            else json.dumps(obj).encode("utf-8"))
    return struct.pack(">I", len(body)) + body


async def read_frame(reader: asyncio.StreamReader, use_msgpack: bool):
    hdr = await reader.readexactly(4)
    (n,) = struct.unpack(">I", hdr)
    body = await reader.readexactly(n)
    return msgpack.unpackb(body, raw=False) if use_msgpack else json.loads(body)


# --------------------------------------------------------------------------- #
# The i7-SIDE adapter: the server's `pl` split-deployment backend. Lives here so
# the wire protocol (encode/read_frame above) has one source of truth shared by
# both ends. The Precision runs the server (TcpPLBackend); the Kria runs the
# daemon (handle_conn). Over Tailscale/GigE they speak the frames above.
# --------------------------------------------------------------------------- #
class TcpPLBackend:
    """InferenceBackend that runs the PL submission on a REMOTE Kria daemon.

    One infer_batch == one framed RPC: send {prompts:[...]}, get back
    {completions:[...], busy_ms}. busy_ms is the daemon's MEASURED PL submission
    time, so fabric occupancy stays honest end-to-end (not a software proxy).

    A single persistent connection carries a serial stream of batches (the server
    serializes the fabric, so calls don't overlap; a lock guards it anyway so the
    backend is correct standalone). On a dropped socket it reconnects once and
    retries; if that fails the call raises and the server releases those clients.
    """

    capacity: int = 16

    def __init__(self, host: str, port: int = 9099, *, use_msgpack: bool = False,
                 capacity: int = 16, connect_timeout: float = 5.0,
                 rpc_timeout: float = 30.0):
        self.host = host
        self.port = port
        self.use_msgpack = use_msgpack and _HAVE_MSGPACK
        self.capacity = capacity
        self.connect_timeout = connect_timeout
        self.rpc_timeout = rpc_timeout
        self._reader: "asyncio.StreamReader | None" = None
        self._writer: "asyncio.StreamWriter | None" = None
        self._lock = asyncio.Lock()

    async def _ensure_conn(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            return
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port), self.connect_timeout)

    async def _reset(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        self._reader = self._writer = None

    async def _rpc(self, obj: dict) -> dict:
        await self._ensure_conn()
        self._writer.write(encode(obj, self.use_msgpack))
        await self._writer.drain()
        return await asyncio.wait_for(read_frame(self._reader, self.use_msgpack),
                                      self.rpc_timeout)

    async def infer_batch(self, reqs):
        from .backend import BatchOutcome, InferResult
        prompts = [r.prompt for r in reqs]
        async with self._lock:
            try:
                resp = await self._rpc({"prompts": prompts})
            except (OSError, asyncio.IncompleteReadError, asyncio.TimeoutError,
                    ConnectionError):
                # dead/half-open socket -> drop it and retry once on a fresh conn
                await self._reset()
                resp = await self._rpc({"prompts": prompts})
        comps = list(resp.get("completions", []))
        results = [
            InferResult(client_id=r.client_id, prompt=r.prompt, seq=r.seq,
                        completion=(comps[i] if i < len(comps) else ""))
            for i, r in enumerate(reqs)
        ]
        busy_s = float(resp.get("busy_ms", 0.0)) / 1000.0
        return BatchOutcome(results=results, busy_s=busy_s,
                            filled=len(reqs), capacity=self.capacity)

    async def close(self) -> None:
        await self._reset()


class PLDevice:
    """Owns the real PL via pl_seq_pp16 (boot-streamed weights). Lazy heavy
    import so the daemon module loads on a dev box; construction raises if no
    /dev/mem. Reuses PLSingleTokenBackend's setup to avoid duplicating the
    register dance — same protocol, single source of truth."""

    def __init__(self, lanes: int, fclk: float, gen_chars: int):
        from .backend_stub import PLSingleTokenBackend
        self._be = PLSingleTokenBackend(lanes=lanes, fclk=fclk, gen_chars=gen_chars)

    async def infer(self, prompts: list[str]) -> list[str]:
        from .backend import InferRequest
        reqs = [InferRequest(client_id=str(i), prompt=p, seq=0)
                for i, p in enumerate(prompts)]
        outcome = await self._be.infer_batch(reqs)
        return [r.completion for r in outcome.results]

    @property
    def launches(self) -> int:
        return self._be.launches

    def submit_raw(self, tok_ids):
        return self._be._submit_once(tok_ids)

    def close(self):
        try:
            asyncio.get_event_loop()
        except Exception:
            pass


async def handle_conn(reader, writer, dev, use_msgpack):
    """One i7 connection: stream of {prompts:[...]} batches -> {completions:[...]}."""
    peer = writer.get_extra_info("peername")
    try:
        while True:
            req = await read_frame(reader, use_msgpack)
            prompts = list(req.get("prompts", []))
            t0 = time.monotonic()
            comps = await dev.infer(prompts)
            dt_ms = (time.monotonic() - t0) * 1000.0
            writer.write(encode({"completions": comps, "busy_ms": round(dt_ms, 2)},
                                use_msgpack))
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionResetError):
        pass
    finally:
        writer.close()


def bench(dev, secs: float, batch: int):
    """Measure A53 launch rate. One launch == one _submit_once == one PL pass of
    16 streams. We hammer it for `secs` and report launches/s. 'With batching'
    serves 16 streams/launch; 'without' would need 1 launch/stream, so the same
    user work costs 16x the launches — we report both so the coalescing win is
    explicit (the PRD's exact question)."""
    tok_ids = [0] * 16
    t0 = time.monotonic()
    n = 0
    while time.monotonic() - t0 < secs:
        dev.submit_raw(tok_ids)
        n += 1
    dt = time.monotonic() - t0
    lps = n / dt
    print(f"[bench] {n} launches in {dt:.2f}s -> {lps:.0f} launches/s")
    print(f"[bench] WITH batching (16 streams/launch): "
          f"{lps*16:.0f} stream-completions/s aggregate")
    print(f"[bench] WITHOUT batching (1 stream/launch): "
          f"{lps:.0f} stream-completions/s (16x fewer) "
          f"-> coalescing buys ~16x here")
    print(f"BENCH_VERDICT launches_per_s={lps:.0f} batched_streams_per_s={lps*16:.0f}")


async def serve(args):
    use_msgpack = _HAVE_MSGPACK and not args.json
    dev = PLDevice(args.lanes, args.fclk, args.gen_chars)
    print(f"[a53d] PL ready (lanes={args.lanes} fclk={args.fclk/1e6:.0f}MHz "
          f"transport={'msgpack' if use_msgpack else 'json'})")
    server = await asyncio.start_server(
        lambda r, w: handle_conn(r, w, dev, use_msgpack),
        args.host, args.port)
    print(f"[a53d] listening on {args.host}:{args.port}")
    async with server:
        await server.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kria A53 thin batch daemon")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9099)
    ap.add_argument("--lanes", type=int, default=128)
    ap.add_argument("--fclk", type=float, default=125e6)
    ap.add_argument("--gen-chars", type=int, default=6)
    ap.add_argument("--json", action="store_true",
                    help="force length-prefixed JSON even if msgpack is present")
    ap.add_argument("--bench", action="store_true",
                    help="measure A53 launch rate then exit (no server)")
    ap.add_argument("--secs", type=float, default=5.0, help="bench duration")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args(argv)

    if args.bench:
        dev = PLDevice(args.lanes, args.fclk, args.gen_chars)
        bench(dev, args.secs, args.batch)
        return 0
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
