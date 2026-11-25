#!/bin/bash
# (11-25) 정해 둔 update까지 돌고 멈추는 학습(fork 시험 run): RUN의 plan_u<U>.pt가 저장되면 두 rank의 plan_rl을 PID로 끄고
# (pkill 안 씀) 콘솔 목록 이름을 "멈춤 · ..."으로 바꿈. 학습이 먼저 죽으면(one_<RUN>.done) 그대로 적고 끝.
# 인자: RUN U [멈춤 사유 한 줄]
RUN=$1; U=$2; WHY=${3:-"u$2에서 멈춤 (계획대로)"}
R=/home/shchon11/f1sim_runs; D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
C=$R/$RUN/plan_u$U.pt
until [ -e $C ] || [ -e $D/one_$RUN.done ]; do sleep 20; done
if [ ! -e $C ]; then echo "$(date +%T) $RUN: u$U 전에 학습이 끝남 (one_$RUN.done)"; exit 1; fi
# 저장이 끝날 때까지: 크기가 10 s 동안 그대로
s0=-1; s1=$(stat -c %s $C); while [ "$s0" != "$s1" ]; do sleep 10; s0=$s1; s1=$(stat -c %s $C); done
PIDS=$(ps -eo pid=,args= | awk -v n="--name $RUN " '$2 ~ /^python/ && index($0, "f1sim.learn.plan_rl") && index($0, n) {print $1}')
echo "$(date +%T) $RUN: plan_u$U.pt 저장됨 ($s1 바이트) -> PID $PIDS 끔"
[ -n "$PIDS" ] && kill $PIDS
sleep 20
ps -eo pid=,args= | grep -E "plan_rl --name $RUN " | grep -v grep && echo "아직 살아 있음" || echo "$(date +%T) 두 rank 꺼짐"
python3 - "$RUN" "$WHY" <<'PY'
import json, shutil, sys, time
run, why = sys.argv[1], sys.argv[2]
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + time.strftime(".bak_%m%d_%H%M"))
d = json.load(open(p))
for e in d["picks"]:
    if e["run"] == f"watch_{run}" and e["label"].startswith("학습 중"):
        e["label"] = "멈춤" + e["label"][len("학습 중"):] + f" · {why}"
        print("콘솔 이름:", e["label"])
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
