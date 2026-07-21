#!/usr/bin/env bash
set +e
PLCHAT=/home/ubuntu/kevpl/plchat; EXP=$PLCHAT/fabric/export; KEVBIT=/home/ubuntu/kevbit
sudo systemctl stop kevkv 2>/dev/null
for r in $(seq 1 10); do pids=$(pgrep -f "webchat.demo.a53_daemon"); [ -z "$pids" ] && break; sudo kill -9 $pids 2>/dev/null; sleep 1; done
for i in $(seq 1 20); do ss -ltn 2>/dev/null|grep -q 9099 || break; sleep 1; done
sudo ln -sfn "$PLCHAT/goformer.npz" "$EXP/goformer.npz"; sudo ln -sfn "$PLCHAT/goformer_meta.json" "$EXP/goformer_meta.json"
sudo fpgautil -b "$KEVBIT/gemv_seqkv_gum.bit.bin" >/dev/null 2>&1
sudo systemctl start kevkv
for i in $(seq 1 30); do ss -ltn 2>/dev/null|grep -q 9099 && break; sleep 1; done
echo "RESTORE_DONE kevkv=$(systemctl is-active kevkv) 9099=$(ss -ltn 2>/dev/null|grep -q 9099 && echo UP || echo DOWN)"
