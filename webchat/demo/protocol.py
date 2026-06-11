"""protocol.py — the pure-Python core of the demo inference plane.

Everything here is socket-free on purpose. The PRD's hard parts (debounce,
keystroke coalescing, freshness-on-Enter, the 16-slot batch assembler) are the
bits most likely to have subtle bugs, so they are factored OUT of the asyncio
server and into plain classes/functions that pytest can drive deterministically
with a fake clock. server.py is then a thin I/O shell around these.

Design decisions / honest notes
--------------------------------
* "Latest prompt wins." A client typing fast produces a burst of keystrokes;
  only the most recent prompt is worth speculating on, so the coalescer keeps
  exactly one pending prompt per client and silently drops the stale ones. The
  dropped count is surfaced as the amplification metric (keystrokes in vs
  inferences run) — we WANT that number visible, not hidden.
* Debounce is idle-based, not rate-based: we fire a speculation only once a
  client has been quiet for `debounce_ms`. Default 50 ms (PRD says 40-60).
* Freshness is an exact-string check. The speculation buffered on the client is
  only blittable if it was computed for the EXACT current input; a fast typist
  who outran the last speculation gets one authoritative inference instead of a
  stale blit. We model that decision here so it is unit-testable.
* The batch assembler fills up to BATCH (16, the fabric's natural stream count)
  slots per submission tick. It is FIFO by ready-time with no per-client
  fairness beyond "one in-flight speculation per client" — a client cannot hog
  multiple slots because a newer keystroke replaces its pending prompt rather
  than enqueuing a second one.

Time is always passed in as an explicit monotonic seconds float (`now`) so the
tests can supply a fake clock; nothing here calls time.monotonic() itself.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
from typing import Optional


# ---- wire message types ---------------------------------------------------
# Client -> server
MSG_KEYSTROKE = "keystroke"   # {type, prompt, seq}  speculative: latest wins
MSG_ENTER = "enter"           # {type, prompt, seq}  user committed; blit or re-fire

# Server -> client
MSG_SPECULATION = "speculation"   # {type, prompt, completion, infer_ms}
MSG_AUTHORITATIVE = "authoritative"  # {type, prompt, completion, infer_ms}
MSG_STREAM = "stream"             # {type, prompt, text}  one char-chunk of an Enter-miss
MSG_STREAM_END = "stream_end"     # {type, prompt, completion}  the tidied final
MSG_TELEMETRY = "telemetry"       # {type, ...aggregate...} (optional per-client echo)


class Reason(enum.Enum):
    """Why a job entered the batch — drives the amplification accounting."""
    SPECULATION = "speculation"
    AUTHORITATIVE = "authoritative"


@dataclasses.dataclass
class PendingPrompt:
    """The single pending prompt a client is waiting to have speculated on."""
    prompt: str
    seq: int
    arrived_at: float       # monotonic seconds of the LAST keystroke
    ready_at: float         # arrived_at + debounce window; eligible after this
    reason: Reason


@dataclasses.dataclass
class ClientState:
    """Per-client speculation bookkeeping. One pending prompt at most.

    `last_speculated` holds the prompt string of the most recent completion we
    sent this client, so the freshness check on Enter can decide blit vs re-fire
    without a round trip-shaped guess.
    """
    client_id: str
    pending: Optional[PendingPrompt] = None
    inflight: bool = False           # a job for this client is in a batch
    last_keystroke_at: float = 0.0   # for "active user" accounting
    last_speculated: Optional[str] = None   # prompt of last completion we sent
    # committed Enters QUEUE here (FIFO) so every one gets a reply even under a
    # burst — unlike `pending` (a single latest-wins speculation slot).
    auth_q: "collections.deque[PendingPrompt]" = dataclasses.field(
        default_factory=collections.deque)

    # amplification counters (monotonic, per client)
    keystrokes: int = 0
    inferences: int = 0


class Coalescer:
    """Per-client keystroke debounce + coalescing.

    Call `on_keystroke` for every keystroke message; it overwrites the client's
    single pending prompt (latest wins) and counts the keystroke. Call
    `due(now)` each scheduler tick to pull out the prompts whose debounce window
    has elapsed and that have no speculation already in flight — those become
    batch jobs.
    """

    AUTH_CAP = 8   # max queued Enters per client (a spammer can't grow it forever)

    def __init__(self, debounce_ms: float = 50.0):
        self.debounce_s = debounce_ms / 1000.0
        self.clients: dict[str, ClientState] = {}
        self.dropped = 0   # keystrokes superseded before they were ever inferred

    def client(self, client_id: str) -> ClientState:
        cs = self.clients.get(client_id)
        if cs is None:
            cs = ClientState(client_id=client_id)
            self.clients[client_id] = cs
        return cs

    def on_keystroke(self, client_id: str, prompt: str, seq: int, now: float) -> None:
        cs = self.client(client_id)
        cs.keystrokes += 1
        cs.last_keystroke_at = now
        if cs.pending is not None:
            # the previously-pending prompt never made it to inference -> dropped
            self.dropped += 1
        cs.pending = PendingPrompt(
            prompt=prompt, seq=seq, arrived_at=now,
            ready_at=now + self.debounce_s, reason=Reason.SPECULATION,
        )

    def on_enter_refire(self, client_id: str, prompt: str, seq: int, now: float) -> None:
        """QUEUE an AUTHORITATIVE inference (freshness miss on Enter). An Enter is
        a committed message, so unlike a keystroke it does NOT supersede — it is
        appended to the client's FIFO queue so every Enter gets a reply, in order,
        even when the user spams them faster than the fabric drains. Bypasses the
        debounce window (the user is already waiting). A not-yet-run speculation
        for this client is now stale intent and is dropped. The queue is capped so
        a spammer can't grow it without bound (oldest un-started Enter falls off)."""
        cs = self.client(client_id)
        if cs.pending is not None and cs.pending.reason is Reason.SPECULATION:
            self.dropped += 1
            cs.pending = None
        cs.auth_q.append(PendingPrompt(
            prompt=prompt, seq=seq, arrived_at=now,
            ready_at=now, reason=Reason.AUTHORITATIVE,
        ))
        while len(cs.auth_q) > self.AUTH_CAP:
            cs.auth_q.popleft()
            self.dropped += 1

    def due(self, now: float) -> list[tuple[str, PendingPrompt]]:
        """Return (client_id, PendingPrompt) for every client with a job ready and
        nothing in flight: a queued Enter (committed, FIFO, takes priority) else a
        speculation past its debounce window. Sorted by ready_at so the assembler
        is fair-ish across clients."""
        out: list[tuple[str, PendingPrompt]] = []
        for cid, cs in self.clients.items():
            if cs.inflight:
                continue
            if cs.auth_q:                       # committed Enters first (FIFO head)
                out.append((cid, cs.auth_q[0]))
            elif cs.pending is not None and now >= cs.pending.ready_at:
                out.append((cid, cs.pending))
        out.sort(key=lambda kp: kp[1].ready_at)
        return out


@dataclasses.dataclass
class BatchJob:
    client_id: str
    prompt: str
    seq: int
    reason: Reason


class BatchAssembler:
    """Fills up to `batch` slots per submission tick from the due list.

    The fabric's natural batch is N=16 streams/pass, so we never submit more
    than that in one go; overflow waits for the next tick (and shows up as queue
    depth in telemetry). Pulling a job marks the client in-flight so a second
    speculation cannot be queued for the same client until the result returns.
    """

    def __init__(self, coalescer: Coalescer, batch: int = 16):
        self.co = coalescer
        self.batch = batch

    def assemble(self, now: float) -> list[BatchJob]:
        due = self.co.due(now)
        jobs: list[BatchJob] = []
        for cid, p in due:
            if len(jobs) >= self.batch:
                break
            cs = self.co.client(cid)
            cs.inflight = True
            cs.inferences += 1
            if cs.auth_q and cs.auth_q[0] is p:   # popped the Enter we scheduled
                cs.auth_q.popleft()
            else:
                cs.pending = None
            jobs.append(BatchJob(client_id=cid, prompt=p.prompt,
                                 seq=p.seq, reason=p.reason))
        return jobs

    def queue_depth(self, now: float) -> int:
        """Prompts that are due but did not fit this tick's batch.

        This is the honest 'work waiting on the fabric' number the dashboard
        plots. It counts everything past its debounce window and not in flight,
        minus what a single batch can absorb."""
        ready = len(self.co.due(now))
        # plus the queued Enters behind each client's head (committed, waiting)
        backlog = sum(max(0, len(cs.auth_q) - 1)
                      for cs in self.co.clients.values())
        return max(0, ready - self.batch) + backlog

    def on_result(self, client_id: str, prompt: str) -> None:
        """Mark a client's speculation no longer in flight + record the prompt
        the completion was computed for (feeds the freshness check)."""
        cs = self.co.client(client_id)
        cs.inflight = False
        cs.last_speculated = prompt


def freshness_blit(client_state: ClientState, current_prompt: str) -> bool:
    """True if the buffered speculation can be blitted locally on Enter.

    Bit-honest exact match: the speculation is only valid for the EXACT prompt
    it was computed against. Anything else (fast typist outran it, or nothing
    speculated yet) returns False -> the caller fires one authoritative
    inference. This is the single function the freshness requirement reduces to;
    the client mirrors the same check locally to avoid a round trip when it CAN
    blit.
    """
    return (client_state.last_speculated is not None
            and client_state.last_speculated == current_prompt)
