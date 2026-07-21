"""kv256_ab_test.py — on-board bit-exact + speed A/B: C hot loop vs Python _go.

Runs each prompt through BOTH drivers on the SAME resident kv256 fabric and
asserts the emitted token stream is identical. Bit-exact by construction (the C
driver issues the same register writes in the same order; the on-chip sampler is
re-seeded per request, so a fixed seed is deterministic regardless of fabric
history). Also reports the wall-clock speedup — the whole point.

Run on the Kria as root, kv256 (SQRV) bitstream loaded, .so built:
    cc -O2 -fPIC -shared -o libkv256_drv.so kv256_drv.c
    sudo python3 -m fabric.stage3.board.kv256_ab_test --lanes 128 --tmax 128 --gen 48

Exit 0 + AB_PASS iff every prompt matches. This is the gate before wiring the C
path into a53_daemon — do not deploy the C driver without a green AB_PASS.
"""

from __future__ import annotations

import argparse
import time

from fabric.stage3.board.pl_kv256 import PLKV256Device
from fabric.stage3.board.pl_kv256_c import KV256CDriver

PROMPTS = [
    "once upon time",
    "the dog run fast",
    "kevin like eat",
    "she go park play friend",
    "\n",
]


def python_token_stream(dev: PLKV256Device, prompt: str, gen: int, seed: int):
    """Reproduce _stream_gen's raw pre-tidy token GO train via dev._go, with a
    fixed seed so it is deterministic and comparable to the C run. Returns the
    list of `gen` emitted token ids + total fabric cycles."""
    ids = dev._enc((prompt or "").strip().lower()) or (dev._enc("\n") or [0])
    maxp = dev.tmax - dev.gen_chars
    if len(ids) > maxp:
        ids = ids[-maxp:]
    L = len(ids)
    dev._fab_cyc = 0
    dev._fab_passes = 0
    am = 0
    for pos, tok in enumerate(ids):
        if seed and pos == L - 1:
            dev._dev.wr(0x30, seed)          # R_SEED: arm on-chip sampler once
        am = dev._go(tok, pos)
    out = [am]
    prev = am
    for g in range(1, gen):
        prev = dev._go(prev, L + g - 1)
        out.append(prev)
    return out, int(dev._fab_cyc), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=128)
    ap.add_argument("--tmax", type=int, default=128)
    ap.add_argument("--fclk", type=float, default=166.667e6)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0,
                    help="0 = greedy (fully deterministic); nonzero = on-chip Gumbel "
                         "(still deterministic given the seed).")
    ap.add_argument("--reps", type=int, default=5, help="timing reps per driver")
    a = ap.parse_args()

    # Python driver (also owns encode/decode + the reference _go).
    dev = PLKV256Device(lanes=a.lanes, tmax=a.tmax, fclk=a.fclk, temp=0.0)
    dev.gen_chars = a.gen
    drv = KV256CDriver(base=0xA0000000)
    print(f"IDCODE python=0x{dev._dev.rd(0x2C):08X} c=0x{drv.idcode():08X}")

    all_ok = True
    for p in PROMPTS:
        ref, ref_cyc, ids = python_token_stream(dev, p, a.gen, a.seed)
        got, got_cyc = drv.run(ids, gen=a.gen, seed=a.seed)
        ok = (ref == got)
        all_ok = all_ok and ok
        print(f"[{'OK ' if ok else 'MISMATCH'}] prompt={p!r:30} "
              f"cyc py={ref_cyc} c={got_cyc} "
              + ("" if ok else f"\n  ref={ref}\n  got={got}"))

    # speed A/B on the first prompt
    ids = python_token_stream(dev, PROMPTS[0], a.gen, a.seed)[2]
    t = time.time()
    for _ in range(a.reps):
        python_token_stream(dev, PROMPTS[0], a.gen, a.seed)
    py_ms = (time.time() - t) / a.reps * 1000
    t = time.time()
    for _ in range(a.reps):
        drv.run(ids, gen=a.gen, seed=a.seed)
    c_ms = (time.time() - t) / a.reps * 1000
    print(f"\nWALL-CLOCK per {a.gen}-tok reply: python={py_ms:.2f}ms  c={c_ms:.2f}ms  "
          f"speedup={py_ms / c_ms:.2f}x  "
          f"(python {a.gen / py_ms * 1000:.0f} tok/s -> c {a.gen / c_ms * 1000:.0f} tok/s)")

    drv.close()
    print("AB_PASS" if all_ok else "AB_FAIL")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
