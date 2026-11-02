#!/bin/bash
# 상자 정책을 배치될 자리에서만 학습 (11-03): 두 단계(props_dodge.sh) 기각 -- 상관 잡음(d)은 상자 하나 접촉을 대조 5곳 -> 3곳으로 줄였지만
# 독립 잡음 선 되살리기(e)가 그걸 지움(접촉 7 -> 14), 30배치 9/30·세게 25, 멈추는 배치 5·11·20 그대로. 원인 사슬: 상자 단계는 --base
# 없이 학습 -- 상자 정책이 빈 길까지 모두 몰고(경주 정책 몫 1.00) 그 결정이 기울기 대부분, 상관 잡음은 빈 길의 선까지 좁힘(상자 정책 혼자
# 깨끗한 맵 7.05~7.2 s, 혼자 정책 6.5 s). 그런데 배치되는 선생은 혼합(SoloMix: 보이는 상자가 10 m 안일 때만 상자 정책) -- 학습과 배치가
# 다름. 상대 단계(chain_race3.sh)는 이미 --base로 학습. 고침: 상자 단계도 --base(같은 혼자 정책·10 m·10결정) -- 기울기는 상자 정책이
# 실제로 몬 결정에서만, 잡음도 거기서만, 빈 길은 혼자 정책 그대로. 나머지는 d와 같음(같은 시작 srp25h3 u200, 상관 0.9 + 하한, 가격 80,
# 학습률 5e-5, 세게 문턱 1.0, 100회) -> d u100이 대조(잡음 같고 --base만 다름).
# 기각(미리 정함): 상자 하나 12곳 중 접촉 > 2곳, 30배치 랩 중앙 > 7.3 s, 접촉 난 배치 > 5/30, 세게 > 15, 멈춤 1 s 넘는 배치가 하나라도.
# 인자: 이름머리(srp25h3) 맵 콘솔맵 GPU 환경수 시드
TAG=$1; M=$2; MC=$3; GPU=$4; ENVS=$5; S=$6
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --severity-min 1.0"
SOLO=$(python3 -c "import torch; print(torch.load('$R/${TAG}_propsmix/state_u200.pt', map_location='cpu', weights_only=False)['args']['base'])")
L=$D/gate_${TAG}.log
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}g_props --stage props $PL $COMMON --updates 100 \
  --noise-corr 0.9 --std-floor 0.05 0.2 --base $SOLO --base-reach 10 --base-hold 10 --seed ${S}6 \
  --init $R/${TAG}_props/state_u200.pt > $D/${TAG}g_props.log 2>&1 || { echo fail > $D/gate_${TAG}.done; exit 1; }
echo "== 상자 정책을 배치 자리(--base)에서만 학습: ${TAG}g u100 (대조 ${TAG}d u100: 같은 시작·잡음, --base 없음)" > $L
python3 - >> $L <<PY
import torch
for p in "$R/${TAG}g_props/state_u100.pt $R/${TAG}d_props/state_u100.pt".split():
    ck = torch.load(p, map_location="cpu", weights_only=False)
    W = ck["model"]["pi.0.weight"]; cn = W.norm(dim=0)
    print(f"  {'/'.join(p.split('/')[-2:])}: 첫 층 열 가중치 -- 혼자 입력 {cn[:107].mean():.3f}, 라이다 {cn[107:215].mean():.3f}, 보이는 상자 칸 {cn[306:318].mean():.3f}, 보이는 통로 {cn[318:409].mean():.3f}")
PY
CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 -u $T/one_crate.py $MC 30x34 lineref=$R/srp25h2_propsmix/state_u200.pt \
  ${TAG}g_u100=$R/${TAG}g_props/state_u100.pt 2>&1 | grep -A 12 "==" >> $L
echo "  선생(혼합) 혼자 ${TAG}g u100: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${TAG}g_props/state_u100.pt $MC 60 2>&1 | grep '주행' | cut -c1-160)" >> $L
for x in g_props d_propsmix; do
  CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $R/${TAG}${x}/state_u100.pt $MC 40 30 777 $D/layouts/t_${TAG}${x%%_*} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_t_${TAG}${x%%_*}.log
  echo "  선생(혼합) 30배치 ${TAG}${x%%_*} u100: $(tail -n 1 $D/layouts_t_${TAG}${x%%_*}.log)" >> $L
done
echo done > $D/gate_${TAG}.done
