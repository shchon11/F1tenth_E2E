#!/bin/bash
# 라이다 격자 학생 (11-02 03:50): 선생 = 상자 단계 200회 + 혼자 선생 혼합(빈 길은 혼자 단계 정책), 정직한 궤적(0.6 s, 재라벨).
# 상대 단계는 정책을 망가뜨려(느려짐, 출발 지점 엉킴, 느린 차 추월 0) 멈췄고, 혼자 + 상자 주행을 실차용 학생으로 먼저 만든다.
# 인자: 이름 머리  GPU  시드
TAG=$1; GPU=$2; S=$3
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_student --name sst_${TAG}_props --teacher /home/shchon11/f1sim_runs/${TAG}_propsmix/state_u200.pt \
  --stage props --envs 768 --iters 80 --steps 256 --buffer 300000 --buffer-cpu 1 --epochs 3 --frames 4 --stride 4 --danger-weight 4 \
  --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30 --unwedge 0 --grid 1 --seed $S > $D/sst_${TAG}_props.log 2>&1
echo done > $D/sst_${TAG}_props.done
