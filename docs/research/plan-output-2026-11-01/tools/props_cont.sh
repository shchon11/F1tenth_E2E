#!/bin/bash
# 상자 단계 이어서 (11-03 04:45): srp25h3(잡음 상관 0, 상자 수 1..7, 배우는 보이는 상자만)은 30배치 랩 중앙 7.22 s(전 선생 9.00)로 빨라졌지만
# 접촉 난 배치 10/30(세게 36; 전 3/30, 13). 장면: 모퉁이 뒤에서 늦게 보인 라인 위 상자를 1.2~1.9 m/s로 받고 밀며 멈춤(배치 5: 14.9 s) --
# 1.0~1.5 m/s 접촉은 콘솔에선 세게인데 학습에선 공짜(contact_severity_min 1.5). 학습 끝까지 충돌이 줄던 중(9.3 -> 6.1/km).
# 바꿈: 200회 더(가격 80 그대로) + --severity-min 1.0(판정 기준과 같게). 기각(미리 정함, 같은 30배치): 랩 중앙 > 7.6 s, 또는 접촉 난 배치 > 5/30,
# 또는 세게 > 15. 인자: 이름머리 맵 콘솔맵 GPU 환경수 시드
TAG=$1; M=$2; MC=$3; GPU=$4; ENVS=$5; S=$6
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
SOLO=$(python3 -c "import torch; print(torch.load('$R/${TAG}_propsmix/state_u200.pt', map_location='cpu', weights_only=False)['args']['base'])")
bash $T/gpu_guard3.sh "state_[r]l --name ${TAG}b_" $D/${TAG}b.done > $D/gpu_guard_${TAG}b.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}b_props --stage props $PL $COMMON --updates 200 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --severity-min 1.0 --seed ${S}3 \
  --init $R/${TAG}_props/state_u200.pt > $D/${TAG}b_props.log 2>&1 || { echo fail > $D/${TAG}b.done; exit 1; }
python3 - <<PY
import torch, os, shutil
ck = torch.load("$R/${TAG}b_props/state_u200.pt", map_location="cpu", weights_only=False)
a = dict(ck["args"]); a.update(base="$SOLO", base_reach=10.0, base_hold=10); ck["args"] = a
os.makedirs("$R/${TAG}b_propsmix", exist_ok=True)
torch.save(ck, "$R/${TAG}b_propsmix/state_u200.pt")
shutil.copy("$R/${TAG}b_props/args.json", "$R/${TAG}b_propsmix/args.json")
PY
echo "상자 정책 혼자(깨끗한 맵): $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${TAG}b_props/state_u200.pt $MC 60 2>&1 | grep '주행' | cut -c1-200)" > $D/check_${TAG}b.log
echo "선생(혼합) 혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${TAG}b_propsmix/state_u200.pt $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $D/check_${TAG}b.log
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $R/${TAG}b_propsmix/state_u200.pt $MC 40 30 777 $D/layouts/t_${TAG}b 2>&1 | grep "배치\|합계\|Error" > $D/layouts_t_${TAG}b.log
echo "선생(혼합) 30배치: $(tail -n 1 $D/layouts_t_${TAG}b.log)" >> $D/check_${TAG}b.log
echo done > $D/${TAG}b.done
