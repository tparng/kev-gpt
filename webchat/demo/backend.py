"""backend.py — the InferenceBackend interface the server batches against.

The whole point of this seam is that the server (debounce/coalesce/batch/stream)
does not care whether completions come from a deterministic stub, the real
single-token PL over GigE, or — eventually — a KV-decode chat bitstream. It
hands the backend a list of (client_id, prompt) jobs and gets back a list of
completions plus the wall time the backend was busy, which telemetry turns into
fabric occupancy.

`infer_batch` is async so a network/AXI backend can await its RPC without
blocking the event loop. The stub implements it with asyncio.sleep so off-box
load tests have a realistic, configurable latency knob.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Optional


@dataclasses.dataclass
class InferRequest:
    client_id: str
    prompt: str
    seq: int


@dataclasses.dataclass
class InferResult:
    client_id: str
    prompt: str
    seq: int
    completion: str


@dataclasses.dataclass
class BatchOutcome:
    """Results plus the timing telemetry needs to compute fabric occupancy.

    busy_s is the time the BACKEND itself was working on this batch (the PL
    submission, or the stub's modelled latency) — not the server's wall time
    around it. occupancy = busy_s summed over a telemetry window / window.
    filled / capacity gives the slot-fill fraction the PRD wants folded in.
    """
    results: list[InferResult]
    busy_s: float
    filled: int
    capacity: int
    # PURE FABRIC time for this batch (sum of the PL CYCLES register / fclk),
    # vs busy_s which is the ROUND-TRIP (host loop + readback + sampling). The two
    # give the chat its honest "two ceilings": fabric tok/s vs round-trip tok/s.
    # None when the backend can't measure it (stub, host-KV).
    fabric_s: Optional[float] = None
    tokens: int = 0
    passes: int = 0          # forward passes (GOs) — fabric tok/s = passes / fabric_s


class InferenceBackend(abc.ABC):
    """One PL submission == one infer_batch call."""

    capacity: int = 16

    @abc.abstractmethod
    async def infer_batch(self, reqs: list[InferRequest]) -> BatchOutcome:
        ...

    async def close(self) -> None:   # pragma: no cover - trivial default
        return None
