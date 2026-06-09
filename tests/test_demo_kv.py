"""Off-board test for the model-faithful KV chat backend.

Uses the numpy GEMV backend (exact int32, no hardware) to prove KVChatBackend
emits real, deterministic, multi-token Kevin. Needs the trained export
(fabric/export/goformer.npz), which is gitignored/regenerable, so the whole
module skips when it's absent. SelectorEventLoop to match the tcp tests' Windows
teardown workaround.
"""

import os

import pytest

from webchat.demo.backend import InferRequest

_NPZ = os.path.join(os.path.dirname(__file__), "..", "fabric", "export",
                    "goformer.npz")
pytestmark = pytest.mark.skipif(
    not os.path.exists(_NPZ),
    reason="needs fabric/export/goformer.npz (gitignored; regenerate via "
           "model.export_fabric)")


def _run(coro):
    import asyncio
    loop = asyncio.SelectorEventLoop()
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


def test_kv_backend_real_multitoken_greedy():
    from webchat.demo.backend_kv import KVChatBackend

    async def run(prompts):
        be = KVChatBackend(gemv_backend="numpy", gen_chars=10, greedy=True)
        reqs = [InferRequest(client_id=f"c{i}", prompt=p, seq=i)
                for i, p in enumerate(prompts)]
        oc = await be.infer_batch(reqs)
        await be.close()
        return oc

    oc = _run(run(["once upon time", "me want go fast"]))
    assert oc.filled == 2 and oc.capacity == 16
    assert oc.busy_s > 0                                   # real generation cost
    comps = [r.completion for r in oc.results]
    assert all(isinstance(c, str) and len(c) > 0 for c in comps)
    assert [r.client_id for r in oc.results] == ["c0", "c1"]

    # greedy is argmax -> deterministic: a fresh backend reproduces the same
    # completion for the same prompt, regardless of what follows it in the batch.
    oc2 = _run(run(["once upon time"]))
    assert oc2.results[0].completion == comps[0]
