#!/bin/bash
# 2026-10-29 rl-audit: 새 행동 구조(주행선 기준 옆 위치 + 속도 배율, 0.25초마다)의 출발점과 상한을 정책 평가와 똑같은 1대1
# 조건(eval_ckpt.sh의 경주: 상자 주행선 위, 상대 4종 × 시드 2)에서 잰다. 학습자 차를 선생이 몬다(evaluate --teacher):
#   raceline + TSPEED  -- 주행선을 옆 0으로 따라감, 속도 배율 TSPEED (구조의 출발점)
#   rollout            -- 같은 (옆 위치, 속도 배율) 후보를 미래 시뮬로 골라 몬다 (구조의 상한, 특권)
# usage: eval_teacher.sh GPU NAME KIND [TSPEED]
D=$(cd $(dirname $0) && pwd); G=$1; N=$2; KIND=$3; TS=${4:-1.0}; MAPJ=$(cat $D/entry_map.json)
O=${OUT_DIR:-$D/../out}/$N; mkdir -p $O
CK=/home/shchon11/f1sim_runs/rlc6_s3010/ppo_u256.pt      # 관측 규격만 읽음, 운전은 선생
cd /home/shchon11/F1tenth_E2E/f1sim && export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
COMMON="--action-mode plan --tracks real:korea_2026_competition --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 --device cuda:0 --collision-mode soft --speed-cap ${CAP:-10.0} --dial-value 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92 --teacher --teacher-kind $KIND --teacher-speed $TS"
F=$O/solo_eval.json
[ -f $F ] || CUDA_VISIBLE_DEVICES=$G WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 $PY -u -m f1sim.learn.evaluate $CK $COMMON \
  --plan-follower map --plan-follower-map "$MAPJ" --output $F --envs 16 --steps 2400 --seed 81 --race-size 1 --graphs \
  --procedural-obstacles 0.0 > $F.log 2>&1 || { echo "$N solo FAILED"; tail -3 $F.log; exit 1; }
echo "[$(date +%H:%M)] $N solo done"
for S in 81 82; do for K in forzaeth forzaeth_pred lane_switch interactive; do
  F=$O/race_s${S}_$K.json; [ -f $F ] && continue
  CUDA_VISIBLE_DEVICES=$G WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 $PY -u -m f1sim.learn.evaluate $CK $COMMON \
    --plan-follower map --plan-follower-map "$MAPJ" --output $F --envs 32 --steps 2400 --seed $S --race-size 2 --graphs \
    --movable-obstacles --spawn-runway 3.0 --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 \
    --procedural-raceline-corridor off --opp-slots "[{\"kind\": \"$K\", \"speed_scale\": [0.7, 1.0]}]" > $F.log 2>&1 \
    || { echo "$N s$S $K FAILED"; tail -3 $F.log; exit 1; }
done; done
echo "[$(date +%H:%M)] $N races done"
