#!/bin/bash
# 학습 설정 전수조사 1단계 (11-04 19:40, 사용자: "좆같은 학습 설정들 싹 뒤져봐", "전수조사좀 하자"). 빠른 solo policy에서 시작한 네트워크 하나
# (srp25ows)가 빈 트랙 6.50 -> 6.71 -> 6.90 s(u50, u100). 원인이 (1) world가 섞여서(obstacle·차 episode의 gradient가 빈 트랙을 덮어씀 = forgetting)인지,
# (2) 빈 트랙에서도 reward·학습 설정이 solo 학습 때와 달라서인지를 가름. solo 학습(srp25h2_solo2)과 다른 설정: severity_min 1.5 -> 1.0, hit_shaping /
# hit_head 0 -> 1, soft_bound / bound_loss 0 -> 1, level_replay 0 -> 0.5, lr 1e-4 -> 5e-5, envs 2048 -> 1024.
#   A: obstacle·차 없는 빈 트랙만 + srp25ows 설정 그대로. 느려지면 (2).
#   B: 네 경우 world 그대로 + 위 설정을 solo 학습 값으로(envs는 1024 그대로). 느려지면 (1).
# 둘 다 srp25h2_solo2 u300(6.50 s)에서 100 update, u50 / u100 solo lap. 기각(미리 정함): 'A 느려짐' = u100 solo lap > 6.65 s.
# 인자: 이름 GPU 방식(A|B) 시드 업데이트
NAME=$1; GPU=$2; MODE=$3; S=$4; U=$5
M=real:korea_2025_iccas; MC=real/iccas25
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/audit_${NAME}.log
echo "== ${NAME}: 방식 $MODE, 시작 srp25h2_solo2/state_u300.pt (6.50 s)" > $L
( for UU in $(seq 50 50 $U); do
    until [ -e $R/${NAME}/state_u${UU}.pt ] || [ -e $D/audit_${NAME}.done ]; do sleep 20; done
    [ -e $R/${NAME}/state_u${UU}.pt ] || break
    echo "  u${UU} solo: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${NAME}/state_u${UU}.pt $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $L
  done ) &
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
OBS="--corridor-obs 1 --scan-obs 1 --vis-actor 1 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0"
BASECOMMON="--envs 1024 --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --noise-corr 0 --removal-s 10 30 --prop-count uniform --prop-bank $R/prop_banks --price 80 --price-start 80 --price-hold 0 --price-ramp 1 \
  --pos-bonus 80 --prop-touch-price 40"
if [ "$MODE" = A ]; then
  # 빈 트랙만, 설정은 srp25ows 그대로
  X="--prop-fraction 0 --removal-p 0 --opp-present 0 --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 --severity-min 1.0 --lr 5e-5 --lr-end 5e-5"
else
  # 네 경우 world, 설정은 solo 학습 값
  X="--prop-fraction 0.5 --removal-p 0.5 --opp-present 0.5 --soft-bound 0 --bound-loss 0 --hit-head 0 --hit-shaping 0 --level-replay 0 --severity-min 1.5 --lr 1e-4 --lr-end 5e-5"
fi
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage race $PL $OBS $BASECOMMON $X --updates $U --seed $S --opp-slot '{"speed_scale": [0.4, 1.2]}' \
  --init $R/srp25h2_solo2/state_u300.pt > $D/${NAME}.log 2>&1 || { echo fail > $D/audit_${NAME}.done; exit 1; }
wait
echo done > $D/audit_${NAME}.done
