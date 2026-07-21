#!/usr/bin/env bash
# activate_model.sh <kevin|legit> : one clean model activation on the fabric.
set +e
M=$1
PLCHAT=/home/ubuntu/kevpl/plchat; EXP=$PLCHAT/fabric/export
KEVBIT=/home/ubuntu/kevbit; LEGIT=/home/ubuntu/legit_export
PY=/home/ubuntu/kevweb/venv/bin/python
if [ "$M" = kevin ]; then NPZ=$PLCHAT/goformer.npz; META=$PLCHAT/goformer_meta.json; BIT=$KEVBIT/gemv_seqkv_gum.bit.bin; FCLK=166.7e6
elif [ "$M" = legit ]; then NPZ=$LEGIT/goformer.npz; META=$LEGIT/goformer_meta.json; BIT=$KEVBIT/gemv_seqkv_legit.bit.bin; FCLK=142.857e6
else echo "ACTIVATE_ERR bad model $M"; exit 9; fi
sudo systemctl stop kevkv 2>/dev/null
for r in $(seq 1 10); do pids=$(pgrep -f "webchat.demo.a53_daemon"); [ -z "$pids" ] && break; sudo kill -9 $pids 2>/dev/null; sleep 1; done
for i in $(seq 1 20); do ss -ltn 2>/dev/null|grep -q 9099 || break; sleep 1; done
sudo ln -sfn "$NPZ" "$EXP/goformer.npz"; sudo ln -sfn "$META" "$EXP/goformer_meta.json"
sudo fpgautil -b "$BIT" >/dev/null 2>&1
cd "$PLCHAT"
nohup sudo -n setsid $PY -m webchat.demo.a53_daemon --port 9099 --json --engine kv256 \
    --lanes 256 --tmax 128 --gen-chars 104 --temp 0.4 --top-k 10 --fclk "$FCLK" \
    </dev/null >> /home/ubuntu/daemon_$M.log 2>&1 &
disown
for i in $(seq 1 30); do ss -ltn 2>/dev/null|grep -q 9099 && break; sleep 1; done
P=$(sudo $PY /home/ubuntu/health_probe.py 2>&1); HR=$?
echo "ACTIVATE model=$M fclk=$FCLK ready_s=$i health_rc=$HR probe=$P"
exit $HR
