"""Unit tests for the demo inference-plane logic.

Pure-python, no sockets: the fiddly PRD logic (debounce, keystroke coalescing,
freshness-on-Enter, 16-slot batch assembly, telemetry aggregation) was factored
into webchat/demo/protocol.py and webchat/demo/telemetry.py exactly so it could
be driven here with a fake clock. If these pass, the asyncio shell in server.py
is just I/O.
"""

import pytest

from webchat.demo.protocol import (BatchAssembler, ClientState, Coalescer,
                                   Reason, freshness_blit)
from webchat.demo.telemetry import Reservoir, Telemetry


# ---- debounce + coalescing -------------------------------------------------
def test_keystroke_not_due_before_debounce():
    co = Coalescer(debounce_ms=50)
    co.on_keystroke("a", "he", 1, now=0.0)
    assert co.due(now=0.0) == []            # 0 ms elapsed
    assert co.due(now=0.049) == []          # 49 ms < 50 ms
    due = co.due(now=0.050)                 # 50 ms -> eligible
    assert [c for c, _ in due] == ["a"]


def test_latest_prompt_wins_and_counts_drop():
    co = Coalescer(debounce_ms=50)
    co.on_keystroke("a", "h", 1, now=0.0)
    co.on_keystroke("a", "he", 2, now=0.01)
    co.on_keystroke("a", "hel", 3, now=0.02)
    # two earlier prompts were superseded before ever being inferred
    assert co.dropped == 2
    due = co.due(now=1.0)
    assert len(due) == 1
    assert due[0][1].prompt == "hel"        # only the latest survives


def test_due_sorted_by_ready_time():
    co = Coalescer(debounce_ms=50)
    co.on_keystroke("a", "a1", 1, now=0.00)
    co.on_keystroke("b", "b1", 1, now=0.02)
    co.on_keystroke("c", "c1", 1, now=0.01)
    order = [c for c, _ in co.due(now=1.0)]
    assert order == ["a", "c", "b"]         # FIFO by ready_at


def test_inflight_blocks_second_speculation():
    co = Coalescer(debounce_ms=50)
    asm = BatchAssembler(co, batch=16)
    co.on_keystroke("a", "he", 1, now=0.0)
    jobs = asm.assemble(now=0.1)
    assert len(jobs) == 1 and co.client("a").inflight
    # a new keystroke arrives while in flight -> pending set but NOT due
    co.on_keystroke("a", "hel", 2, now=0.2)
    assert asm.assemble(now=0.3) == []      # blocked until result returns
    asm.on_result("a", "he")
    assert co.client("a").inflight is False
    # now the queued "hel" becomes available
    jobs2 = asm.assemble(now=0.4)
    assert [j.prompt for j in jobs2] == ["hel"]


# ---- freshness on Enter ----------------------------------------------------
def test_freshness_blit_exact_match():
    cs = ClientState(client_id="a", last_speculated="once upon")
    assert freshness_blit(cs, "once upon") is True
    assert freshness_blit(cs, "once upo") is False     # typist outran it
    assert freshness_blit(ClientState(client_id="b"), "x") is False  # nothing buffered


def test_enter_refire_supersedes_and_is_authoritative():
    co = Coalescer(debounce_ms=50)
    co.on_keystroke("a", "hell", 1, now=0.0)
    co.on_enter_refire("a", "hello", 2, now=0.1)
    assert co.dropped == 1                  # the stale speculation was dropped
    due = co.due(now=0.1)                   # authoritative bypasses debounce
    assert len(due) == 1
    assert due[0][1].reason is Reason.AUTHORITATIVE
    assert due[0][1].prompt == "hello"


def test_enter_queue_every_committed_message_is_answered_in_order():
    # spamming Enter must not LOSE messages (the "… never pops in" bug): each
    # committed Enter is queued and answered FIFO, one at a time (serial fabric).
    co = Coalescer(debounce_ms=50)
    asm = BatchAssembler(co, batch=16)
    for i, p in enumerate(["p1", "p2", "p3"]):
        co.on_enter_refire("a", p, i + 1, now=0.0)
    got = []
    for _ in range(3):
        jobs = asm.assemble(now=0.1)
        assert len(jobs) == 1                # one in flight at a time
        assert asm.assemble(now=0.1) == []   # blocked until the result returns
        got.append(jobs[0].prompt)
        asm.on_result("a", jobs[0].prompt)
    assert got == ["p1", "p2", "p3"]         # all answered, in order


def test_enter_queue_capped_drops_oldest():
    co = Coalescer(debounce_ms=50)
    for i in range(Coalescer.AUTH_CAP + 3):
        co.on_enter_refire("a", f"p{i}", i, now=0.0)
    q = list(co.client("a").auth_q)
    assert len(q) == Coalescer.AUTH_CAP      # capped
    assert q[0].prompt == "p3"               # the 3 oldest fell off
    assert co.dropped == 3


# ---- batch assembly --------------------------------------------------------
def test_batch_caps_at_capacity_and_reports_queue_depth():
    co = Coalescer(debounce_ms=10)
    asm = BatchAssembler(co, batch=16)
    for i in range(20):
        co.on_keystroke(f"c{i}", f"p{i}", 1, now=0.0)
    assert asm.queue_depth(now=1.0) == 4    # 20 due - 16 batch
    jobs = asm.assemble(now=1.0)
    assert len(jobs) == 16
    # the 4 that didn't fit are still pending and due next tick
    assert len(asm.assemble(now=1.0)) == 4
    assert asm.queue_depth(now=1.0) == 0


def test_assemble_marks_inferences_and_clears_pending():
    co = Coalescer(debounce_ms=10)
    asm = BatchAssembler(co, batch=16)
    co.on_keystroke("a", "hi", 1, now=0.0)
    asm.assemble(now=0.5)
    cs = co.client("a")
    assert cs.inferences == 1 and cs.pending is None and cs.keystrokes == 1


# ---- telemetry aggregation -------------------------------------------------
def test_reservoir_percentiles():
    r = Reservoir(size=1000)
    for v in range(1, 101):                 # 1..100
        r.add(float(v))
    assert r.percentile(0.50) == pytest.approx(50, abs=1)
    assert r.percentile(0.99) == pytest.approx(99, abs=1)
    assert r.n == 100


def test_active_users_window_prunes():
    tel = Telemetry(active_window_s=5.0)
    tel.on_keystroke("a", now=0.0)
    tel.on_keystroke("b", now=4.0)
    assert tel.active_users(now=4.0) == 2   # both within 5 s
    assert tel.active_users(now=6.0) == 1   # "a" aged out (0 < 6-5)
    assert tel.active_users(now=10.0) == 0  # both aged out


def test_telemetry_tick_rates_and_amplification():
    tel = Telemetry()
    tel._last_tick = 0.0
    for _ in range(50):
        tel.on_keystroke("a", now=0.5)
    tel.on_inference(3)                      # 50 keystrokes coalesced into 3
    tel.on_batch(busy_s=0.4, filled=8, capacity=16, tokens=120)
    tel.on_enter(blitted=True)
    tel.on_enter(blitted=False)
    rec = tel.tick(now=1.0, queue_depth=2)   # 1 s window
    assert rec["agg_tok_s"] == pytest.approx(120.0, abs=0.1)
    assert rec["queue_depth"] == 2
    assert rec["amplification"]["keystrokes"] == 50
    assert rec["amplification"]["inferences"] == 3
    assert rec["amplification"]["ratio"] == pytest.approx(50 / 3, abs=0.01)
    # occupancy = busy 0.4/1.0 * fill 8/16 = 0.2
    assert rec["fabric_occupancy"] == pytest.approx(0.2, abs=1e-6)
    assert rec["ready_at_enter"] == pytest.approx(0.5, abs=1e-6)  # 1 of 2 enters blitted
    assert rec["power_w"] is None and rec["temp_c"] is None        # off-box


def test_telemetry_window_resets_each_tick():
    tel = Telemetry()
    tel._last_tick = 0.0
    tel.on_batch(busy_s=0.5, filled=16, capacity=16, tokens=100)
    tel.tick(now=1.0)
    rec2 = tel.tick(now=2.0)                 # nothing happened in 2nd window
    assert rec2["agg_tok_s"] == 0.0
    assert rec2["fabric_occupancy"] == 0.0
