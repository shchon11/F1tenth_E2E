#!/bin/bash
# 라이다 격자 학생 재증류 (11-02): 선생 = 상자 접촉 대가로 이어 학습한 상자 정책(*_propst u100) + 혼자 선생 혼합(*_propstmix).
# 설정은 student_propsmix.sh와 같음(선생만 바뀜). 인자: 이름 머리  GPU  시드
TAG=$1; GPU=$2; S=$3
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_student --name sst_${TAG}_propst --teacher /home/shchon11/f1sim_runs/${TAG}_propstmix/state_u100.pt \
  --stage props --envs 768 --iters 80 --steps 256 --buffer 300000 --buffer-cpu 1 --epochs 3 --frames 4 --stride 4 --danger-weight 4 \
  --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30 --unwedge 0 --grid 1 --seed $S > $D/sst_${TAG}_propst.log 2>&1
echo done > $D/sst_${TAG}_propst.done
