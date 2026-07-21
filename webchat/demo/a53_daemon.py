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

    async def infer_stream(self, req, on_chunk):
        """Stream ONE prompt: send {prompts:[p], stream:true}, deliver each
        {chunk} to on_chunk(text) as it arrives, return (completion, busy_s) on
        the final frame. The server uses this for the Enter freshness-miss so the
        first char shows after one token. No mid-stream retry (re-streaming would
        double the chars on the client); a clean socket error before any chunk
        falls through to the caller's release path."""
        async with self._lock:
            await self._ensure_conn()
            self._writer.write(encode({"prompts": [req.prompt], "stream": True},
                                      self.use_msgpack))
            await self._writer.drain()
            while True:
                frame = await asyncio.wait_for(
                    read_frame(self._reader, self.use_msgpack), self.rpc_timeout)
                if "chunk" in frame:
                    await on_chunk(frame["chunk"])
                else:
                    comp = (frame.get("completions") or [""])[0]
                    fab = frame.get("fabric_ms")
                    return (comp, float(frame.get("busy_ms", 0.0)) / 1000.0,
                            (float(fab) / 1000.0 if fab else None),
                            int(frame.get("tokens", len(comp))),
                            int(frame.get("passes", 0)))

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
        fab = resp.get("fabric_ms")
        return BatchOutcome(results=results, busy_s=busy_s,
                            filled=len(reqs), capacity=self.capacity,
                            fabric_s=(float(fab) / 1000.0 if fab else None),
                            tokens=int(resp.get("tokens",
                                                sum(len(c) for c in comps))),
                            passes=int(resp.get("passes", 0)))

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


async def _stream_one(dev, prompt, writer, use_msgpack):
    """Run ONE streaming generation: the device produces chars on an executor
    thread (it's blocking MMIO), bridged to this async writer through a queue so
    each char is framed as {chunk} the instant it lands, then a final
    {completions, busy_ms, done}."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    t0 = time.monotonic()

    def work():
        try:
            full = dev.stream_into(
                prompt, lambda ch: loop.call_soon_threadsafe(q.put_nowait, ("c", ch)))
            loop.call_soon_threadsafe(q.put_nowait, ("done", full))
        except Exception as exc:                       # noqa: BLE001
            loop.call_soon_threadsafe(q.put_nowait, ("err", str(exc)))

    fut = loop.run_in_executor(None, work)
    try:
        while True:
            kind, val = await q.get()
            if kind == "c":
                writer.write(encode({"chunk": val}, use_msgpack))
                await writer.drain()
            else:
                dt = round((time.monotonic() - t0) * 1000.0, 2)
                comp = val if kind == "done" else ""
                frame = {"completions": [comp], "busy_ms": dt, "done": True,
                         "tokens": len(comp)}
                fab_ms = getattr(dev, "last_fabric_ms", None)
                if fab_ms:
                    frame["fabric_ms"] = round(fab_ms, 2)
                    frame["passes"] = getattr(dev, "last_fab_passes", 0)
                writer.write(encode(frame, use_msgpack))
                await writer.drain()
                break
    finally:
        await fut


async def handle_conn(reader, writer, dev, use_msgpack):
    """One i7 connection: batches {prompts:[...]} -> {completions:[...]}, or a
    streaming request {prompts:[p], stream:true} -> {chunk}* {completions,done}."""
    peer = writer.get_extra_info("peername")
    try:
        while True:
            req = await read_frame(reader, use_msgpack)
            prompts = list(req.get("prompts", []))
            if req.get("stream") and prompts and hasattr(dev, "stream_into"):
                await _stream_one(dev, prompts[0], writer, use_msgpack)
                continue
            t0 = time.monotonic()
            comps = await dev.infer(prompts)
            dt_ms = (time.monotonic() - t0) * 1000.0
            resp = {"completions": comps, "busy_ms": round(dt_ms, 2),
                    "tokens": sum(len(c) for c in comps)}
            fab_ms = getattr(dev, "last_fabric_ms", None)
            if fab_ms:                                  # pure-fabric time (PL CYCLES)
                resp["fabric_ms"] = round(fab_ms, 2)
                resp["passes"] = getattr(dev, "last_fab_passes", 0)
            writer.write(encode(resp, use_msgpack))
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


def make_device(args):
    """t1 = single-token PL pass (the 60k path, first-token-faithful); kv =
    model-faithful multi-token Kevin over the host-KV decoder (--kv-backend);
    kv256 = model-faithful multi-token Kevin fully in fabric (doc-7 R1 KV
    bitstream, single stream — the daemon serves it serially)."""
    if args.engine == "kv":
        from .backend_kv import KVChatDevice
        return KVChatDevice(gemv_backend=args.kv_backend, gen_chars=args.gen_chars,
                            greedy=args.greedy)
    if args.engine == "kv256":
        from fabric.stage3.board.pl_kv256 import PLKV256Device
        return PLKV256Device(lanes=args.lanes, fclk=args.fclk,
                             gen_chars=args.gen_chars, tmax=args.tmax,
                             temp=(0.0 if args.greedy else args.temp),
                             top_k=args.top_k, npz=args.npz, meta=args.meta)
    return PLDevice(args.lanes, args.fclk, args.gen_chars)


async def serve(args):
    use_msgpack = _HAVE_MSGPACK and not args.json
    dev = make_device(args)
    print(f"[a53d] {args.engine} ready (lanes={args.lanes} fclk={args.fclk/1e6:.0f}MHz "
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
    ap.add_argument("--gen-chars", type=int, default=None,
                    help="completion chars (default: 48 for kv256, 6 otherwise)")
    ap.add_argument("--engine", choices=["t1", "kv", "kv256"], default="t1",
                    help="t1=single-token PL pass (first-token-faithful); "
                         "kv=model-faithful multi-token Kevin (host-KV decoder); "
                         "kv256=model-faithful multi-token Kevin in fabric "
                         "(doc-7 KV bitstream, single stream)")
    ap.add_argument("--tmax", type=int, default=256,
                    help="--engine kv256: on-chip KV window; MUST match the "
                         "bitstream's TMAX generic")
    ap.add_argument("--kv-backend", choices=["numpy", "pl", "c"], default="c",
                    help="--engine kv GEMV backend (c=compiled MMIO on the Kria)")
    ap.add_argument("--greedy", action="store_true",
                    help="argmax decode (reproducible); for kv256 this forces "
                         "temp=0 and overrides --temp")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="--engine kv256: host-side sampling temperature over the "
                         "fabric head logits (0 = greedy/argmax, deterministic; "
                         "~0.7-0.9 = varied stories). Same prompt -> different "
                         "completions when >0.")
    ap.add_argument("--top-k", type=int, default=40,
                    help="--engine kv256: keep only the top-k logits when sampling "
                         "(0 = full distribution)")
    ap.add_argument("--npz", default=None,
                    help="--engine kv256: model weights .npz to boot-stream "
                         "(default: fabric/export/goformer.npz). Point at a "
                         "different model to A/B — but its baked-ROM bitstream "
                         "(gamma/inv_sact/dqm/dqe/pos_emb) MUST match this model.")
    ap.add_argument("--meta", default=None,
                    help="--engine kv256: tokenizer meta .json paired with --npz "
                         "(default: fabric/export/goformer_meta.json)")
    ap.add_argument("--json", action="store_true",
                    help="force length-prefixed JSON even if msgpack is present")
    ap.add_argument("--bench", action="store_true",
                    help="measure A53 launch rate then exit (no server)")
    ap.add_argument("--secs", type=float, default=5.0, help="bench duration")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args(argv)
    if args.gen_chars is None:
        args.gen_chars = 48 if args.engine == "kv256" else 6

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
