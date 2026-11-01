#!/bin/bash
# --plan-relabel 반증 시험 (11-01 밤): srp25 혼자 250회에서 이어 60회, 계획을 실제로 달린 길에 맞추는 손실 1.0 (4070)
# 통과: 계획 [0,L] 어긋남 중앙 < 0.15 m (시작 0.62), 콘솔 워커 랩 <= 6.66 s(250회 6.50~6.53의 +2 %), 벽 충돌 0
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.state_rl --name srp25r_test --stage solo --tracks real:korea_2025_iccas \
  --action plan --plan-lookahead 0.235 --plan-horizon 0.3 --repeat 2 --envs 2048 --warps 16 --beams 36 --episode-s 40 \
  --updates 60 --horizon 64 --epochs 4 --minibatches 8 --lr 1e-4 --lr-end 5e-5 --gamma 0.995 --lam 0.95 --clip 0.2 \
  --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --save-every 20 --seed 7101 --plan-relabel 1.0 \
  --init /home/shchon11/f1sim_runs/srp25_solo/state_u250.pt > $D/srp25r_test.log 2>&1
for U in 20 60; do
  echo "u$U 콘솔: $(CUDA_VISIBLE_DEVICES=0 timeout 900 python3 $T/console_laps.py ~/f1sim_runs/srp25r_test/state_u$U.pt real/iccas25 60 2>&1 | grep '주행\|Error')"
  CUDA_VISIBLE_DEVICES=0 timeout 900 python3 $T/plan_consistency.py ~/f1sim_runs/srp25r_test/state_u$U.pt real/iccas25 30 2>&1 | grep -v Warning | tail -4
done
CUDA_VISIBLE_DEVICES=0 timeout 900 python3 $T/drive_pic.py ~/f1sim_runs/srp25r_test/state_u60.pt real/iccas25 20 $D/../drive_srp25r_u60.png 2>&1 | grep "그림\|Error"
echo done > $D/srp25r_test.done
