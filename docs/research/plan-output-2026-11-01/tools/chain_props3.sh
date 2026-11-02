#!/bin/bash
# 상자 단계 다시 학습 (11-03 03:30, 사용자: "선생 문제는 무조건 해결해 / 편법·학생과 갭 큰 특권 입력 쓰면 안 됨 / 모델 용량 부족일 수도").
# 진단(README "선생이 상자 때문에 느린 원인"): 상자 정책은 상자가 하나도 없는 맵에서도 7.18 s(혼자 정책 6.50 s) -- 트랙 폭을 다 쓰는
# 넓은 선 대신 중심선 근처 좁은 선. 원인 확정: 상자 단계에서 탐색 잡음을 상관 0.9(약 0.5 s씩 이어짐) + 하한으로 바꿨는데, 이 잡음 아래서는
# 혼자 정책의 넓은 선이 벽에 40배 박힘(0.11 -> 4.61/km, J -40 %) -> 학습이 벽에서 떨어진 좁은 선을 배움. 용량은 아님(256 x 2가 합성
# 행동을 속도 명령 오차 중앙 0.04-0.05 m/s로 담음). 학습 세계 편향: 상자 있는 판은 언제나 패턴 7개(13-18개), 빈 길 34 %.
# 바꿈(원인마다 하나): (1) 탐색 잡음을 혼자 단계와 같게(상관 0, 하한 없음), (2) 판마다 패턴 수 1..7 고르게(--prop-count uniform),
# (3) 배우는 학생이 볼 수 있는 상자만(학생 격자 안, 무잡음 라이다 시선; --vis-actor 1) -- 전체 정답 상자 칸·통로·남은 시간은 비평가만.
# 기각(미리 정함): F1 상자 정책 혼자 깨끗한 맵 60 s 랩 > 6.70 s(전 7.18) / F2 선생(혼합) 30배치(시드 777) 랩 중앙이 전 선생보다
# 안 빠르거나 접촉 난 배치가 전 선생 + 1 넘음 / F3 학생 접촉 난 배치 > 선생 + 2.
# 인자: 이름머리 맵 콘솔맵 혼자체크포인트 GPU 환경수 시드머리 지킴이(1/0)
TAG=$1; M=$2; MC=$3; SOLO=$4; GPU=$5; ENVS=$6; S=$7; GUARD=$8
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --noise-corr 0 --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1"
if [ "$GUARD" = 1 ]; then
  bash $T/gpu_guard3.sh "state_[r]l --name ${TAG}_|state_[s]tudent --name sst_${TAG}" $D/chain_${TAG}.done > $D/gpu_guard_${TAG}.log 2>&1 &
fi
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_props --stage props $PL $COMMON --updates 200 \
  --lr 1e-4 --lr-end 5e-5 --price 80 --price-start 20 --price-hold 0 --price-ramp 100 --seed ${S}2 \
  --init $SOLO > $D/${TAG}_props.log 2>&1 || exit 1
echo props > $D/chain_${TAG}.props_done
# F1 right away: the props policy ALONE on the empty track (console worker)
echo "상자 정책 혼자(깨끗한 맵): $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${TAG}_props/state_u200.pt $MC 60 2>&1 | grep '주행\|Error' | cut -c1-220)" > $D/check_${TAG}.log
for U in 50 100 150; do
  echo "  u$U 혼자(깨끗한 맵): $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${TAG}_props/state_u$U.pt $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $D/check_${TAG}.log
done
python3 - <<PY
import torch, os, shutil
ck = torch.load("$R/${TAG}_props/state_u200.pt", map_location="cpu", weights_only=False)
a = dict(ck["args"]); a.update(base="$SOLO", base_reach=10.0, base_hold=10); ck["args"] = a
os.makedirs("$R/${TAG}_propsmix", exist_ok=True)
torch.save(ck, "$R/${TAG}_propsmix/state_u200.pt")
shutil.copy("$R/${TAG}_props/args.json", "$R/${TAG}_propsmix/args.json")
PY
echo "선생(혼합) 혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${TAG}_propsmix/state_u200.pt $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $D/check_${TAG}.log
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $R/${TAG}_propsmix/state_u200.pt $MC 40 30 777 $D/layouts/t_${TAG} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_t_${TAG}.log
echo "선생(혼합) 30배치: $(tail -n 1 $D/layouts_t_${TAG}.log)" >> $D/check_${TAG}.log
# the student (LiDAR grid), its world as the teacher's (layouts 1..7 patterns)
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_student --name sst_${TAG}_props --teacher $R/${TAG}_propsmix/state_u200.pt \
  --stage props --envs 768 --iters 80 --steps 256 --buffer 300000 --buffer-cpu 1 --epochs 3 --frames 4 --stride 4 --danger-weight 4 \
  --prop-fraction 0.8 --prop-count uniform --removal-p 0.5 --removal-s 10 30 --unwedge 0 --grid 1 --seed ${S}9 > $D/sst_${TAG}_props.log 2>&1 || exit 1
C=$(ls -t $R/sst_${TAG}_props/student_i*.pt | head -1)
echo "학생 혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $C $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $D/check_${TAG}.log
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $C $MC 40 30 777 $D/layouts/s_${TAG} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_s_${TAG}.log
echo "학생 30배치: $(tail -n 1 $D/layouts_s_${TAG}.log)" >> $D/check_${TAG}.log
echo done > $D/chain_${TAG}.done
