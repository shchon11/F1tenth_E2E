#!/bin/bash
# 학습 중 상자 접촉은 탐색 노이즈 탓인가 (11-02 06:50): 상자 정책을 학습과 같은 세계에서 평균 행동 / 학습 노이즈 행동으로 몰아
# 상자 닿은 시간(s/km)을 비교. 인자: 체크포인트 맵 GPU
C=$1; M=$2; GPU=$3
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
for N in 0 1; do
  CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name eval_tmp --eval $C --eval-noise $N --stage props --tracks $M \
    --envs 512 --warps 16 --beams 108 --episode-s 40 --eval-steps 2400 --noise-corr 0.9 --std-floor 0.05 0.2 --corridor-obs 1 --scan-obs 1 \
    --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30 --seed 4242 2>&1 | grep "평가\|Error"
done
