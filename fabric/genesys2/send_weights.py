"""Sends a trained checkpoint's weight image to kevgpt_interactive's UART
loader (fabric/genesys2/PORT-NOTES.md "interactive UART chat" pass) --
replaces the GDB `restore ... binary` step every earlier real-hardware
bring-up this port used, so a chat session only needs a serial cable, not
a live JTAG/OpenOCD/GDB session driving every run.

Protocol (matches sw/applications/kevgpt_interactive/main.c's receive
loop exactly, both sides hand-written since there's no existing UART
bulk-transfer protocol anywhere in this project to reuse):
  1. Host waits for firmware's own "KEVGPT_UART_READY" marker line (same
     KEVGPT_* marker convention every other firmware app already prints)
     before sending anything -- avoids a race where the host starts
     transmitting before the board has booted far enough to read it.
  2. Host sends a 4-byte little-endian word count, then that many 4-byte
     little-endian words -- the exact same wrom_to_words() packing
     (low-chunk-first, one 32-bit chunk per WBITS/32 split) every other
     weight-loading path in this project already uses (kevgpt_ddr_stage,
     kevgpt_wld_bringup, kevgpt_optionB_bringup, kevgpt_chat's own
     firmware-embedded array) -- reused, not re-derived.
  3. Host waits for "KEVGPT_UART_LOAD_DONE" to confirm the DMA reload
     (weight_loader_ddr, triggered by the firmware after the transfer)
     completed before declaring success.

    python -m fabric.genesys2.send_weights \\
        --npz fabric/export_optionB/goformer.npz --lanes 64 --p 8 \\
        --port /dev/ttyUSB0
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
import time

from fabric.stage3 import seq_ref
from fabric.stage3.run_sequencer import write_mems_wideword
from fabric.genesys2.gen_chat_fw import wrom_to_words

BAUDRATE = 115200  # matches x-heep.h's UART_BAUDRATE -- every other real-hardware
                    # bring-up this session used the same rate, not a new choice.
READY_MARKER = b"KEVGPT_UART_READY"
DONE_MARKER = b"KEVGPT_UART_LOAD_DONE"


def weight_words_from_checkpoint(npz, lanes, p):
    goformer_p, cfg = seq_ref.build(npz)
    nlayer = len(goformer_p["blocks"])
    iseq = seq_ref.IntSequencer(goformer_p, cfg)
    with tempfile.TemporaryDirectory() as td:
        write_mems_wideword(td, iseq, lanes, nlayer, p)
        return wrom_to_words(os.path.join(td, "wrom.mem"), lanes)


def _wait_for_marker(ser, marker, timeout_s, echo=True):
    deadline = time.time() + timeout_s
    buf = b""
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        if echo:
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
        buf += chunk
        if marker in buf:
            return True
        # keep the tail bounded -- markers are short, no need to accumulate forever
        if len(buf) > 4096:
            buf = buf[-len(marker):]
    return False


def send(port, weight_words, ready_timeout_s=30.0, done_timeout_s=30.0):
    import serial  # deferred: only needed for this real-hardware tool

    with serial.Serial(port, BAUDRATE, timeout=0.2) as ser:
        print(f"waiting for {READY_MARKER.decode()} on {port} ...")
        if not _wait_for_marker(ser, READY_MARKER, ready_timeout_s):
            print(f"SEND_WEIGHTS_FAIL,timeout_waiting_for_ready")
            return False

        print(f"\nsending {len(weight_words)} words ({len(weight_words) * 4} bytes) ...")
        ser.write(struct.pack("<I", len(weight_words)))
        payload = b"".join(struct.pack("<I", w) for w in weight_words)
        ser.write(payload)
        ser.flush()

        print(f"waiting for {DONE_MARKER.decode()} ...")
        if not _wait_for_marker(ser, DONE_MARKER, done_timeout_s):
            print(f"SEND_WEIGHTS_FAIL,timeout_waiting_for_done")
            return False
        print("\nSEND_WEIGHTS_PASS")
        return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fabric.genesys2.send_weights")
    ap.add_argument("--npz", default="fabric/export_optionB/goformer.npz")
    ap.add_argument("--lanes", type=int, default=64)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--port", required=True)
    ap.add_argument("--ready-timeout", type=float, default=30.0)
    ap.add_argument("--done-timeout", type=float, default=30.0)
    a = ap.parse_args(argv)

    weight_words = weight_words_from_checkpoint(a.npz, a.lanes, a.p)
    ok = send(a.port, weight_words, a.ready_timeout, a.done_timeout)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
