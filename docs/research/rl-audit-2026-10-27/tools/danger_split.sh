#!/bin/bash
# Why is the training world ~4.5x more hit-prone than the evaluation? The entry's PPO base, actor FROZEN (lr 0), driven with its
# training noise in variants of the training env, 256 envs x 20 updates each; hard contacts per km from progress.jsonl.
# usage: G=1 danger_split.sh
D=$(cd $(dirname $0) && pwd); G=${G:-1}
SLOTJ='{"kind_mix":["forzaeth","forzaeth_pred","lane_switch","interactive"],"speed_scale":[0.7,1.05],"present":0.5,"spawn":"random","reactive":{"defend":0.3,"line":0.2,"oblivious":0.1},"events":["brake","stop","shift","weave"],"event_rate":0.5}'
run() { local name=$1; shift
  [ -f ~/f1sim_runs/$name/progress.jsonl ] || NAME=$name GPU=$G ENVS=256 TOTAL=$((20*2048)) bash $D/run_train.sh --lr 0 --lr-end 0 --critic-warmup 1 --lap-time-bonus 0 "$@" > $D/../out/$name.log 2>&1
  echo "$name done $(date +%H:%M)"; }
run danger_v0_asis
run danger_v1_noreplay --level-replay 0
run danger_v2_props10 --procedural-max-props 10
run danger_v3_noleague --opp-slots "[$SLOTJ,$SLOTJ,$SLOTJ]" --league-slots ""
run danger_v4_evallike --level-replay 0 --procedural-max-props 10 --race-size 2 --opp-slots "[{\"kind_mix\":[\"forzaeth\",\"forzaeth_pred\",\"lane_switch\",\"interactive\"],\"speed_scale\":[0.7,1.0]}]" --league-slots ""
python3 - <<'PY'
import json, os
print("variant | hard wall+prop /km | hard car /km | total | samples")
for n in ["danger_v0_asis", "danger_v1_noreplay", "danger_v2_props10", "danger_v3_noleague", "danger_v4_evallike"]:
    p = os.path.expanduser(f"~/f1sim_runs/{n}/progress.jsonl")
    if not os.path.exists(p):
        print(n, "missing"); continue
    rows = [json.loads(l) for l in open(p)][4:]
    w = [r["contacts/wall_hard_per_km"] for r in rows if "contacts/wall_hard_per_km" in r]
    c = [r["contacts/car_onset_hard_per_km"] for r in rows if "contacts/car_onset_hard_per_km" in r]
    mw, mc = sum(w) / max(len(w), 1), sum(c) / max(len(c), 1)
    print(f"{n} | {mw:.2f} | {mc:.2f} | {mw + mc:.2f} | {len(rows)} upd")
PY
