#!/bin/bash
# 경주 라이다 학생 (11-02): 선생 = 상대 단계 racec(앞차를 상자 통로에, 상대 칸 비움, 혼자 선생 혼합). 차가 있는 세계(상대 0.3~1.05배속,
# 반응·돌발 행동 기본값, 상자 0.8, 치우기 0.5)에서 DAgger. 학생은 라이다 4장 + 자기 상태만(실차와 같은 입력) -- 앞차도 라이다로만 본다.
# 인자: 이름머리 체크포인트 GPU 시드
TAG=$1; CK=$2; GPU=$3; S=$4
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_student --name sst_${TAG}_race --teacher $CK \
  --stage race --opp-present 0.7 --opp-slot '{"speed_scale": [0.3, 1.05]}' --envs 768 --iters 80 --steps 256 --buffer 300000 \
  --buffer-cpu 1 --epochs 3 --frames 4 --stride 4 --danger-weight 4 \
  --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30 --unwedge 0 --grid 1 --seed $S > $D/sst_${TAG}_race.log 2>&1
echo done > $D/sst_${TAG}_race.done
