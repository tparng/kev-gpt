"""pl_seq_chat — drive the Tier-3 SEQUENCER bitstream (gemv_axi_seq) on the Kria over
/dev/mem and read the FIRST on-silicon cycle count + emitted token stream.

This is the capstone single-stream autoregressive run with the CPU OUT of the loop:
the host streams the resident INT4 weight image once, writes an 8-token prompt, pulses
GO, and the fabric does embed -> NLAYER blocks -> ln_f -> head -> argmax for NGEN tokens
entirely on-chip. We then read NGEN, the latched CYCLES counter, and drain the token
stream over TS_*.

Bit-honest gate: the hardware token stream MUST equal
seq_ref.IntSequencer.generate_greedy(prompt, NGEN). A tok/s number is ONLY reported as
valid when tokens_match=True (the binding gate; raw-logit cosine is a brittle proxy on
this char model). tok/s = NGEN / (CYCLES / FCLK).

Register map (gemv_axi_seq.v; addr = idx*4 from base 0xA0000000):
  0x00 CTRL    write: b0=go pulse, b1=wl_rst, b2=soft_reset
  0x04 STATUS  read:  b0=done_latched, b1=core_busy
  0x10 W_DATA  write: 32-bit weight chunk (pulses wl_we); 64-bit word = low32 then high32
  0x14 PL_DATA write: prompt token id (pulses prompt write @ PL_ADDR)
  0x18 PL_ADDR write: prompt slot address
  0x1C RD_ADDR / 0x20 RD_DATA : residual-x readback (Q6.25)
  0x24 TS_ADDR / 0x28 TS_DATA : emitted token id (low 9 bits)
  0x2C NGEN    read:  number of generated tokens (low 9 bits)
  0x30 CYCLES  read:  run cycle count latched at done -- THE measured per-run cycles
  0x34 IDCODE  read:  0x53455152 "SEQR"

Run on the board (needs root for /dev/mem; bitstream already loaded):
  sudo ~/kevweb/venv/bin/python -m fabric.stage3.board.pl_seq_chat --prompt "once upo"
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import sys
import time

import numpy as np

# repo import path: this file lives at fabric/stage3/board/, repo root is 3 up.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from fabric.stage3 import pack_banked, seq_ref  # noqa: E402

# ---- baked-in bitstream config (must match the synthesized gemv_seq.bit.bin) -------
NLAYER = 4
LANES = 16
PROMPT_LEN = 8
NGEN = 8
FCLK = 40e6                      # timing-closed @ 40 MHz
IDCODE_EXPECTED = 0x53455152     # "SEQR"
BASE = 0xA0000000

# register byte offsets
R_CTRL, R_STATUS = 0x00, 0x04
R_WDATA, R_PLDATA, R_PLADDR = 0x10, 0x14, 0x18
R_RDADDR, R_RDDATA = 0x1C, 0x20
R_TSADDR, R_TSDATA = 0x24, 0x28
R_NGEN, R_CYCLES, R_IDCODE = 0x2C, 0x30, 0x34


class Seqr:
    """Thin /dev/mem AXI4-Lite master. Single 32-bit stores only (numpy uint32 view);
    byte-slice writes corrupt the W_DATA assembler stream, so never do them."""

    def __init__(self, base=BASE):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 0x1000, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        self.reg = np.frombuffer(self.mm, dtype=np.uint32)   # 0x1000/4 = 1024 words

    def wr(self, off, val):
        self.reg[off >> 2] = np.uint32(val & 0xFFFFFFFF)     # one volatile 32-bit store

    def rd(self, off):
        return int(self.reg[off >> 2])

    def close(self):
        # drop the numpy view before closing the mmap (np.frombuffer keeps an
        # exported pointer -> mm.close() raises BufferError otherwise)
        self.reg = None
        self.mm.close()
        os.close(self.fd)


def load_meta(meta_path):
    m = json.load(open(meta_path))
    stoi = m["stoi"]
    itos = {int(k): v for k, v in m["itos"].items()}
    enc = lambda s: [stoi.get(c, 0) for c in s]
    dec = lambda ids: "".join(itos.get(int(i), "?") for i in ids)
    return enc, dec, m["vocab_size"]


def build_weight_image(intseq, lanes, nlayer):
    """The transposed INT4 load stream, EXACTLY matching run_sequencer.write_mems' order:
    per block (qkv|proj|mlp_fc|mlp_proj) for nlayer blocks, then the head."""
    p = intseq.p
    words = []
    for bi in range(nlayer):
        for nm in ("qkv", "proj", "mlp_fc", "mlp_proj"):
            words += pack_banked.pack_transposed(
                np.asarray(p["blocks"][bi][nm][0], np.int8), lanes)
    words += pack_banked.pack_transposed(np.asarray(p["head"][0], np.int8), lanes)
    return words


def verify_against_wrom(words, wrom_path, lanes):
    """Self-check: the rebuilt image must equal the committed wrom.mem (if present)."""
    if not os.path.exists(wrom_path):
        return None
    with open(wrom_path) as f:
        ref = [int(line.strip(), 16) for line in f if line.strip()]
    if len(ref) != len(words):
        return (False, f"length mismatch rebuilt={len(words)} wrom.mem={len(ref)}")
    mism = sum(1 for a, b in zip(words, ref) if a != b)
    return (mism == 0, f"mismatches={mism}/{len(ref)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="PL sequencer chat (capstone bitstream)")
    ap.add_argument("--npz", default=os.path.join(_REPO, "fabric", "export", "goformer.npz"))
    ap.add_argument("--meta", default=os.path.join(_REPO, "fabric", "export", "goformer_meta.json"))
    ap.add_argument("--wrom", default=os.path.join(_REPO, "fabric", "stage3", "mems", "wrom.mem"),
                    help="committed wrom.mem to self-check the rebuilt image against")
    ap.add_argument("--prompt", default="once upo", help="prompt text (encoded then padded to PROMPT_LEN)")
    ap.add_argument("--base", type=lambda s: int(s, 0), default=BASE)
    ap.add_argument("--poll-timeout", type=float, default=30.0)
    ap.add_argument("--dump-resid", type=int, default=0,
                    help="if tokens mismatch, dump this many residual-x words for diagnosis")
    args = ap.parse_args(argv)

    enc, dec, vocab = load_meta(args.meta)

    # ---- reference (the gate): integer sequencer greedy stream ----------------------
    p, cfg = seq_ref.build(args.npz)
    intseq = seq_ref.IntSequencer(p, cfg)

    prompt_ids = enc(args.prompt)[:PROMPT_LEN]
    if len(prompt_ids) < PROMPT_LEN:
        prompt_ids = prompt_ids + [0] * (PROMPT_LEN - len(prompt_ids))
    print(f"[prompt] {args.prompt!r} -> ids {prompt_ids} (PROMPT_LEN={PROMPT_LEN})")

    ref_full = intseq.generate_greedy(list(prompt_ids), NGEN)   # len = PROMPT_LEN+NGEN
    ref_gen = [int(t) for t in ref_full[PROMPT_LEN:]]
    print(f"[ref ] seq_ref.generate_greedy gen tokens = {ref_gen}")
    print(f"[ref ] seq_ref decoded            = {dec(ref_gen)!r}")

    # ---- build + self-check the weight image ----------------------------------------
    words = build_weight_image(intseq, LANES, NLAYER)
    print(f"[wimg] {len(words)} x {LANES*4}-bit words ({len(words)*8} bytes -> "
          f"{len(words)*2} 32-bit W_DATA chunks)")
    chk = verify_against_wrom(words, args.wrom, LANES)
    if chk is None:
        print(f"[wimg] WARNING: {args.wrom} not present -> cannot self-check (using rebuilt)")
    else:
        ok, msg = chk
        print(f"[wimg] vs wrom.mem: {'MATCH' if ok else 'MISMATCH'} ({msg})")
        if not ok:
            print("[wimg] FATAL: rebuilt image != committed wrom.mem; aborting (correctness risk)")
            return 2

    # ---- open the AXI slave + verify IDCODE -----------------------------------------
    dev = Seqr(args.base)
    try:
        idcode = dev.rd(R_IDCODE)
        print(f"[idc ] IDCODE = 0x{idcode:08X} (expected 0x{IDCODE_EXPECTED:08X}) "
              f"-> {'OK' if idcode == IDCODE_EXPECTED else 'MISMATCH'}")
        if idcode != IDCODE_EXPECTED:
            print("[idc ] FATAL: wrong IDCODE -> bitstream not loaded / wrong base. Aborting.")
            return 3

        # 0) soft_reset the core so FSM/KV/scratch start from a defined state (the
        #    proven resident driver does this every run; without it the first GO after
        #    config runs on undefined power-up state -> non-deterministic garbage).
        dev.wr(R_CTRL, 0x4)                          # soft_reset asserted
        dev.wr(R_CTRL, 0x0)                          # release

        # 1) reset the load pointer
        dev.wr(R_CTRL, 0x2)                          # wl_rst pulse

        # 2) stream the whole transposed INT4 image: low32 then high32 per 64-bit word
        t0 = time.time()
        for w in words:
            w = int(w)
            dev.wr(R_WDATA, w & 0xFFFFFFFF)          # LOW chunk first
            dev.wr(R_WDATA, (w >> 32) & 0xFFFFFFFF)  # then HIGH chunk
        t_load = time.time() - t0
        print(f"[load] streamed {len(words)*2} chunks in {t_load:.2f}s "
              f"({len(words)*2/max(t_load,1e-9):.0f} chunks/s)")

        # 3) write the prompt: PL_ADDR=i then PL_DATA=token_id, in order
        for i, tok in enumerate(prompt_ids):
            dev.wr(R_PLADDR, i)
            dev.wr(R_PLDATA, int(tok) & 0x1FF)

        # 4) GO, poll STATUS.done
        dev.wr(R_CTRL, 0x1)                          # go pulse
        t0 = time.time()
        done = False
        while time.time() - t0 < args.poll_timeout:
            st = dev.rd(R_STATUS)
            if st & 0x1:
                done = True
                break
        if not done:
            st = dev.rd(R_STATUS)
            print(f"[run ] TIMEOUT after {args.poll_timeout}s STATUS=0x{st:08X} "
                  f"(busy={(st>>1)&1} done={st&1})")
            return 4
        t_run_wall = time.time() - t0

        # 5) read NGEN, CYCLES, drain the token stream
        ng = dev.rd(R_NGEN) & 0x1FF
        cycles = dev.rd(R_CYCLES)
        got = []
        for i in range(ng):
            dev.wr(R_TSADDR, i)
            got.append(dev.rd(R_TSDATA) & 0x1FF)

        # ---- verdict --------------------------------------------------------------
        tokens_match = (got == ref_gen) and (ng == NGEN)
        print(f"[hw  ] NGEN={ng} CYCLES={cycles} (wall {t_run_wall*1000:.1f} ms)")
        print(f"[hw  ] hw gen tokens = {got}")
        print(f"[hw  ] hw decoded    = {dec(got)!r}")
        print(f"[gate] tokens_match  = {tokens_match} "
              f"(stream {sum(1 for a,b in zip(got,ref_gen) if a==b)}/{min(len(got),len(ref_gen))})")

        if cycles > 0 and tokens_match:
            per_tok = cycles / ng
            tok_s = ng * FCLK / cycles
            print(f"[meas] per-token cycles = {per_tok:.0f}  (NGEN={ng})")
            print(f"[meas] MEASURED tok/s   = {tok_s:.2f}  "
                  f"(= {ng}*{FCLK:.0f}/{cycles} @ {FCLK/1e6:.0f} MHz)")
            print(f"PL_SEQ_VERDICT tokens_match=True ngen={ng} cycles={cycles} "
                  f"tok_s={tok_s:.2f} idcode=0x{idcode:08X}")
        elif not tokens_match:
            print(f"[meas] tok/s WITHHELD: tokens_match=False -> the cycle count would be "
                  f"for WRONG output (would be {ng*FCLK/cycles:.2f} tok/s if it were valid).")
            print(f"PL_SEQ_VERDICT tokens_match=False ngen={ng} cycles={cycles} "
                  f"idcode=0x{idcode:08X}")
            # diagnostics: residual readback
            if args.dump_resid > 0:
                print(f"[diag] first {args.dump_resid} residual-x words (Q6.25, signed):")
                vals = []
                for i in range(args.dump_resid):
                    dev.wr(R_RDADDR, i)
                    raw = dev.rd(R_RDDATA)
                    sv = raw - (1 << 32) if raw & 0x80000000 else raw
                    vals.append(sv / (1 << seq_ref.RESID_FRAC))
                print("[diag]", [f"{v:.5f}" for v in vals])
        else:
            print(f"PL_SEQ_VERDICT cycles=0 (counter not latched) ngen={ng} idcode=0x{idcode:08X}")
        return 0 if tokens_match else 1
    finally:
        dev.close()


if __name__ == "__main__":
    raise SystemExit(main())
