"""mamba_diag — localize the mamba_seq silicon hang. Loads tables, pulses start
on one token, then polls CYCLES + STATUS to see whether the engine is busy-but-
stuck (cycles incrementing, done never asserts) or dead (cycles frozen).
  sudo python3 mamba_diag.py --dir ~/kevmem --fclk 100e6
"""
import argparse, mmap, os, struct, sys, time

BASE = 0xA000_0000
R_CTRL, R_STATUS, R_TOK, R_TOKOUT = 0x00, 0x04, 0x08, 0x0C
R_TSEL, R_TADDR, R_TDATA, R_CYCLES = 0x10, 0x14, 0x18, 0x1C
R_DBGSEL, R_DBGADDR, R_DBGDATA, R_IDCODE = 0x20, 0x24, 0x28, 0x2C
IDCODE = 0x4D414D42


def rd16(p):
    return [int(l, 16) for l in open(p) if l.strip()]


def set_fclk(hz):
    for path in ("/sys/devices/platform/fclk0/set_rate", "/sys/class/clk/fclk0/set_rate"):
        if os.path.exists(path):
            open(path, "w").write(str(int(hz)))
            rb = None
            gr = "/sys/devices/platform/fclk0/clk_rate"
            if os.path.exists(gr):
                rb = open(gr).read().strip()
            print(f"  fclk0 set {hz/1e6:.0f}MHz (readback {rb})")
            return
    print("  WARN no fclk0 node")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fclk", type=float, default=100e6)
    a = ap.parse_args()
    set_fclk(a.fclk)
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    m = mmap.mmap(fd, 0x1000, offset=BASE)

    def wr(o, v): m[o:o+4] = struct.pack("<I", v & 0xFFFFFFFF)
    def rd(o): return struct.unpack("<I", m[o:o+4])[0]

    ident = rd(R_IDCODE)
    print(f"  IDCODE {ident:08x} ({'MAMB ok' if ident == IDCODE else 'MISMATCH'})")
    if ident != IDCODE:
        print("  -> AXI read wrong; engine/clock bad before we even start. STOP.")
        return

    wr(R_CTRL, 0b10)                                  # soft_reset -> clear sweep
    t0 = time.time()
    while not (rd(R_STATUS) >> 2) & 1:                # ready
        if time.time() - t0 > 5:
            print("  READY never asserts (clear-sweep stuck). scan clear FSM hang.")
            return
    print(f"  ready OK ({time.time()-t0:.2f}s clear-sweep)")

    cfg = rd16(f"{a.dir}/ms_cfg.mem")
    print("  loading tables...")
    wr(R_TSEL, 1); wr(R_TADDR, 0)
    for v in rd16(f"{a.dir}/ms_t1.mem"): wr(R_TDATA, v)
    wr(R_TSEL, 0); wr(R_TADDR, 0)
    for v in rd16(f"{a.dir}/ms_t0.mem"): wr(R_TDATA, v)
    for s in range(1, 15):
        wr(R_TSEL, s); wr(R_TADDR, 0)
        for v in rd16(f"{a.dir}/ms_t{s}.mem"): wr(R_TDATA, v)
    print("  tables loaded")

    toks = rd16(f"{a.dir}/ms_tok.mem") if os.path.exists(f"{a.dir}/ms_tok.mem") else [12]
    print(f"  pulsing start on tok {toks[0]} ...")
    wr(R_TOK, int(toks[0]))                           # TOK write pulses go
    # poll STATUS + CYCLES for ~3s
    for i in range(30):
        st = rd(R_STATUS)
        cyc = rd(R_CYCLES)
        done, busy, ready = st & 1, (st >> 1) & 1, (st >> 2) & 1
        print(f"    t={i*0.1:.1f}s STATUS={st:03b}(done={done} busy={busy} rdy={ready}) CYCLES={cyc}")
        if done:
            print(f"  *** DONE at ~{cyc} cyc -> tok_out={rd(R_TOKOUT)&0x3ff} (NO HANG!)")
            return
        time.sleep(0.1)
    print("  HANG: never asserted done in 3s.")
    print(f"    final busy={busy} cycles={cyc}")
    if busy and cyc > 1000:
        print("    -> engine IS running (busy, cycles advancing) but a sub-core never")
        print("       finishes. Functional FSM/sub-core hang (gemv/conv/scan/norm done).")
    elif not busy:
        print("    -> busy is LOW: start didn't latch or FSM fell through to idle.")
    else:
        print(f"    -> cycles low ({cyc}): stuck almost immediately (first phase).")


if __name__ == "__main__":
    main()
