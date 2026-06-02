# Board test runbook — what to run when Tailscale re-auths

Honest state going in: today's work was all laptop sim/synth. The board still has the
**existing PE=1 resident bitstream** loaded and runs at **0.22 tok/s** (PL, Python) /
~11 tok/s (A53 chat). Nothing faster is *running* yet. This runbook brings the new,
gated pieces onto the board and measures them.

Re-auth first (the SSH check tonight hit an expired Tailscale session):
```
# on the board console or after `tailscale up` re-login:
ssh ubuntu@<kria-ip>          # alias: kria-kev ; pw <redacted> ; key auth set up
```

---

## Test A — the certain win: KV cache + C driver on the EXISTING bitstream
No new bitstream. Replaces the slow Python; kills the O(T²) recompute. **Laptop gate
already green: `PL_KV_VERDICT identical=True stream=32/32`.**

```
# 1. sync the repo + board package to the board (from laptop or git pull on board)
cd ~/kevpl && git pull            # or: scp -r fabric/stage3/board ubuntu@kria-kev:~/kevpl/fabric/stage3/

# 2. build the C MMIO driver on the board (gcc 11.4 is there)
cd ~/kevpl/fabric/stage3/board
gcc -O2 -fPIC -shared -o libgemv_axi_drv.so gemv_axi_drv.c
gcc -O2 -o gemv_axi_bench gemv_axi_bench.c gemv_axi_drv.c

# 3. confirm the bitstream is loaded (IDCODE check) — if not, load the resident one:
#    sudo xmutil unloadapp && sudo fpgautil -b ~/kevpl/gemv.bit.bin

# 4. re-run the equivalence gate ON the board python, then generate:
cd ~/kevpl && /home/ubuntu/kevweb/venv/bin/python -m fabric.stage3.board.gate_pl_kv
sudo /home/ubuntu/kevweb/venv/bin/python -m fabric.stage3.board.pl_kv_chat --backend pl --prompt 'once upon time' --n 40 --greedy
sudo /home/ubuntu/kevweb/venv/bin/python -m fabric.stage3.board.pl_kv_chat --backend c  --prompt 'once upon time' --n 40 --greedy

# 5. raw matmul microbench (cycles for one GEMV on silicon):
sudo ./gemv_axi_bench 768 256 0 1000
```
**Expect:** `--backend pl` ≈ **5 tok/s** (Python + KV, ~20× over today's 0.22),
`--backend c` ≈ **40–126 tok/s** (compiled AXI loop — the Tier-2 ceiling). These are
DERIVED from the AXI transaction count (~26k reg txns/token); the board run is the
first MEASURED number. The wall here is the A53 doing ~26k AXI transactions/token —
*not* the matmul. That wall is why the dataflow sequencer (below) is the real path.

---

## Test B — the new wide-GEMV bitstream (if the build completed)
Output (if the laptop Vivado build finished): `C:/kevbuild/stage3_bit/*.bit.bin`,
AXI slave @ 0xA000_0000, IDCODE `0x47454D42` ("GEMB").
```
# copy the bitstream to the board, then:
sudo xmutil unloadapp && sudo fpgautil -b gemv_banked.bit.bin
# point the driver at the GEMB engine and re-run pl_kv_chat / the bench.
```
**Honest expectation:** via the AXI driver this is **still ~100 tok/s** — the PE=256
matmul is *hidden behind the AXI transaction overhead*. Its value is (a) silicon-
validating the resident-banked design (URAM fit @ ~60 blocks, 100 MHz timing) for the
sequencer, and (b) the microbench measuring the wide GEMV's raw cycles. **Do not expect
a token/s jump from this bitstream over Test A** — the speedup needs the CPU out of the
per-token loop (the sequencer), not a wider matmul behind the same driver.

---

## What this session will and won't show
- **Will:** the fabric path goes from 0.22 → ~5–126 tok/s (finally beating the A53),
  measured, bit-identical Kevin output. The resident-banked design validated on silicon.
- **Won't:** 10k (needs the sequencer FSM bitstream — CPU out of loop) or 100k (needs
  the batched + multi-GEMV-engine build). Those are the next bitstreams, not this one.

## tok/s ladder (where each rung comes from)
| Rung | tok/s | What it needs | Status |
|---|---|---|---|
| today | 0.22 | — | MEASURED |
| Python + KV cache | ~5 | board pull + run | staged, gated |
| C + KV cache | ~40–126 | gcc build on board | staged, gated |
| single-stream dataflow | ~7–13k | sequencer FSM → bitstream → silicon | FSM in progress |
| batched serving | ~30–124k | + parallel non-linears + 3–4 GEMV engines | model proven; RTL TBD |
