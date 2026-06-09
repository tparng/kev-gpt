"""Wire-protocol tests for the split-deployment TcpPLBackend.

No hardware: a fake A53 daemon (the real encode/read_frame from a53_daemon, so
this exercises the actual frame format) runs on a loopback ephemeral port, and
TcpPLBackend drives it. Proves the Precision->Kria leg end-to-end in software:
order-preserving prompt->completion mapping, busy_ms->busy_s, persistent-connection
reuse, and reconnect-on-drop.

Windows note: these run on an explicit SelectorEventLoop. The default Windows
ProactorEventLoop can stall on loop.close() with leftover socket transports when
driven inside pytest (it only "works" standalone because the process exits and
nukes the loop); the SelectorEventLoop tears sockets down cleanly. Each test also
closes its server + backend before the loop ends so teardown has nothing pending.
"""

import asyncio

import pytest

from webchat.demo.a53_daemon import TcpPLBackend, encode, read_frame
from webchat.demo.backend import InferRequest


def run_async(coro):
    loop = asyncio.SelectorEventLoop()
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


def _reqs(prompts):
    return [InferRequest(client_id=f"c{i}", prompt=p, seq=i)
            for i, p in enumerate(prompts)]


def _make_handler(accepts, *, busy_ms=7.5, one_then_close=False, drop_last=False):
    async def handle(reader, writer):
        accepts["n"] += 1
        try:
            while True:
                req = await read_frame(reader, False)
                prompts = req["prompts"]
                if drop_last:                 # return one fewer completion
                    prompts = prompts[:-1]
                writer.write(encode({"completions": [p.upper() for p in prompts],
                                     "busy_ms": busy_ms}, False))
                await writer.drain()
                if one_then_close:
                    break
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()
    return handle


async def _serve(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def test_tcp_roundtrip_persistent():
    async def run():
        accepts = {"n": 0}
        server, port = await _serve(_make_handler(accepts))
        be = TcpPLBackend("127.0.0.1", port)
        out1 = await be.infer_batch(_reqs(["once", "upon", "time"]))
        out2 = await be.infer_batch(_reqs(["me", "go"]))
        await be.close()
        server.close()
        await server.wait_closed()
        return out1, out2, accepts["n"]

    out1, out2, accepts = run_async(run())
    assert [r.completion for r in out1.results] == ["ONCE", "UPON", "TIME"]
    assert [r.client_id for r in out1.results] == ["c0", "c1", "c2"]
    assert [r.seq for r in out1.results] == [0, 1, 2]
    assert out1.busy_s == pytest.approx(0.0075)   # daemon busy_ms -> busy_s
    assert out1.filled == 3 and out1.capacity == 16
    assert [r.completion for r in out2.results] == ["ME", "GO"]
    assert accepts == 1                           # one persistent connection, reused


def test_tcp_reconnect_after_drop():
    async def run():
        accepts = {"n": 0}
        server, port = await _serve(_make_handler(accepts, busy_ms=1.0,
                                                  one_then_close=True))
        be = TcpPLBackend("127.0.0.1", port)
        o1 = await be.infer_batch(_reqs(["a"]))
        await asyncio.sleep(0.05)                 # let the FIN land
        o2 = await be.infer_batch(_reqs(["b"]))   # forces a reconnect
        await be.close()
        server.close()
        await server.wait_closed()
        return o1, o2, accepts["n"]

    o1, o2, accepts = run_async(run())
    assert o1.results[0].completion == "A"
    assert o2.results[0].completion == "B"
    assert accepts == 2


def test_tcp_short_completion_list_pads():
    """If the daemon returns fewer completions than prompts (truncated/garbled),
    pad rather than IndexError-crash the server."""
    async def run():
        accepts = {"n": 0}
        server, port = await _serve(_make_handler(accepts, busy_ms=1.0,
                                                  drop_last=True))
        be = TcpPLBackend("127.0.0.1", port)
        out = await be.infer_batch(_reqs(["x", "y", "z"]))
        await be.close()
        server.close()
        await server.wait_closed()
        return out

    out = run_async(run())
    assert [r.completion for r in out.results] == ["X", "Y", ""]
