#!/usr/bin/env bash
# Drives Kevin<->legit rotation from HERE (ssh to kria + precision). Per switch:
# activate model on the fabric + reconnect the chat server. Guaranteed Kevin restore.
#   orchestrate.sh <dur_s> <cycles>
DUR=${1:-300}; CYCLES=${2:-3}
S=/tmp/claude-1000/-home-mikeayles-Desktop-Projects-kev-gpt/348ee0b6-977c-4936-b584-6179d4de5e92/scratchpad
LOG=$S/rotation_events.jsonl; : > "$LOG"
SSHK="ssh -o ConnectTimeout=10 -o ServerAliveInterval=10 kria"
SSHP="ssh -o ConnectTimeout=10 precision"
ts(){ date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }; ep(){ date +%s; }
logev(){ echo "{\"ts\":\"$(ts)\",\"epoch\":$(ep),$1}" >> "$LOG"; }
jstr(){ python3 -c 'import json,sys;print(json.dumps(sys.stdin.read().strip()[:200]))'; }
label(){ [ "$1" = legit ] && echo "Legit — readable English" || echo "Kevin — telegraphic, full speed"; }
push_state(){ # model  (server reads /home/mike/rotation_state.json for the UI countdown)
  printf '{"model":"%s","label":"%s","next_switch":%s}\n' "$1" "$(label "$1")" "$(( $(ep)+DUR ))" \
    | $SSHP 'cat > /home/mike/rotation_state.json' 2>/dev/null
}
restore(){
  logev "\"event\":\"restore_start\""
  $SSHP 'rm -f /home/mike/rotation_state.json' 2>/dev/null   # UI reverts to plain Kevin
  R=$($SSHK 'bash /home/ubuntu/restore_kevin.sh' 2>&1 | grep RESTORE_DONE)
  $SSHP 'systemctl --user restart kevin-chat-server' >/dev/null 2>&1
  logev "\"event\":\"restore_done\",\"detail\":$(printf '%s' "$R" | jstr)"
  echo "RESTORED: $R"
}
trap restore EXIT INT TERM
logev "\"event\":\"test_start\",\"dur\":$DUR,\"cycles\":$CYCLES"
for ((c=0; c<CYCLES; c++)); do
  M=kevin; (( c%2==1 )) && M=legit
  logev "\"event\":\"switch_start\",\"model\":\"$M\",\"cycle\":$c"
  R=$($SSHK "bash /home/ubuntu/activate_model.sh $M" 2>&1 | grep ACTIVATE)
  $SSHP 'systemctl --user restart kevin-chat-server' >/dev/null 2>&1; sleep 3
  push_state "$M"
  logev "\"event\":\"switch_done\",\"model\":\"$M\",\"activate\":$(printf '%s' "$R" | jstr)"
  echo "[$(ts)] $M up :: $R"
  sleep "$DUR"
done
logev "\"event\":\"test_end\""
