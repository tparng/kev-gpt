#!/usr/bin/env bash
# run_on_board.sh — KV-cached PL chat on the Kria, on the EXISTING resident
# bitstream (gemv_axi_resident @ 0xA000_0000, IDCODE "GEMR"). No new bitstream.
#
# This is the guaranteed near-term win: the old driver recomputed the whole
# context every token (O(T^2)); this uses model.goformer_kv.KVDecoder so only the
# new token is processed each step (O(T)), with the AXI inner loop in compiled C.
#
# Prereqs on the board (already true per project notes): gcc 11.4, Python 3.10,
# the resident bitstream loaded, the repo synced, root for /dev/mem.
#
#   ssh ubuntu@<kria-ip>              # (re-auth Tailscale first if the session expired)
#   bash run_on_board.sh              # build + gate; then it runs the real chat
#
# Edit REPO/VENV/BASE if your layout differs.
set -euo pipefail

REPO="${REPO:-/home/ubuntu/kevpl}"          # repo root on the board (mirrors this tree)
VENV="${VENV:-/home/ubuntu/kevweb/venv}"    # the venv with numpy (and torch, unused here)
BASE="${BASE:-0xA0000000}"
BOARD="$REPO/fabric/stage3/board"
PY="$VENV/bin/python"

echo "== 0. sanity =="
test -f "$REPO/fabric/export/goformer.npz"   || { echo "missing goformer.npz under $REPO/fabric/export"; exit 1; }
test -f "$REPO/fabric/export/goformer_meta.json" || { echo "missing goformer_meta.json"; exit 1; }
command -v gcc >/dev/null || { echo "gcc not found on PATH"; exit 1; }
"$PY" -c "import numpy" || { echo "venv has no numpy"; exit 1; }

echo "== 1. build the C MMIO driver (.so) + standalone bench =="
cd "$BOARD"
gcc -O2 -fPIC -shared -o libgemv_axi_drv.so gemv_axi_drv.c
gcc -O2 -o gemv_axi_bench gemv_axi_bench.c gemv_axi_drv.c
echo "built: $BOARD/libgemv_axi_drv.so  $BOARD/gemv_axi_bench"

echo "== 2. laptop-equivalent GATE on the board (no fabric needed) =="
# Proves the KV wiring emits the SAME token stream as the full-recompute reference
# on the real exported weights. Must print: PL_KV_VERDICT identical=True
cd "$REPO"
"$PY" -m fabric.stage3.board.gate_pl_kv

echo "== 3. real PL chat — KV-cached, C inner loop (needs root + bitstream) =="
# If this fails with 'bad IDCODE', load the resident bitstream first, then retry.
sudo "$PY" -m fabric.stage3.board.pl_kv_chat \
     --backend c --prompt "once upon time" --n 40 --greedy

echo "== 4. (optional) compare backends: pure-Python pokes vs C inner loop =="
echo "   sudo $PY -m fabric.stage3.board.pl_kv_chat --backend pl --prompt 'once upon time' --n 40 --greedy"
echo "   sudo $BOARD/gemv_axi_bench 768 256 0 1000   # raw per-GEMV AXI latency (us)"

echo "DONE — KV-cached PL chat staged."
