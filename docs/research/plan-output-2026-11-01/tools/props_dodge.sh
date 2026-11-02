#!/bin/bash
# 상자 비켜 가기를 배우는 탐색 (11-03): 보이는 상자만 보는 상자 정책(srp25h3)은 주행선 위 상자 하나에 12곳 중 5곳에서 선을 그대로 탄 채
# 박음(가로 위치가 상자 없을 때와 같음, one_crate_view.py). 첫 층의 상자 입력 가중치는 200회에 0.14~0.16(혼자 입력 1.0; 포화 아님 1~10 %,
# 주행선 위 상자도 드물지 않음: 배치당 1개, 절반의 배치). 측정(explore_reach.py, 같은 출발): 결정마다 독립 잡음은 1 s 안에 차를
# 0.3 m 넘게 옆으로 보내는 일이 2 %(상관 0.9는 21 %) -- 정책이 바로 되돌려서. 그래서 비켜 가면 이득이라는 표본이 거의 없고
# 늦추기만 배움(srp25h3b: 200회 더 -> 배치마다 0.33 s 느려짐). 상관 0.9는 비켜 가기를 탐색하지만 넓은 선을 벽에 박게 해 좁은 선을 가르침.
# 그래서 두 단계: (1) 비켜 가기 -- srp25h3 u200에서 상관 0.9 + 하한(예전 상자 단계와 같은 잡음) 100회, 나머지는 srp25h3b와 같음
# (같은 시작, 같은 가격·학습률·세게 문턱) -> 같은 회차의 srp25h3b u100이 대조. (2) 선 되살리기 -- (1)의 u100에서 상관 0 100회.
# 기각(미리 정함): (1) 상자 하나 12곳 중 접촉 > 2곳(대조 srp25h3b u100과 함께 잼) 또는 상자 입력 가중치가 대조보다 안 큼 -> 탐색 가설 기각.
# (2) 상자 정책 혼자 깨끗한 맵 랩 > 7.05 s, 30배치 랩 중앙 > 7.3 s, 접촉 난 배치 > 5/30, 세게 > 15, 상자 하나 접촉 > 2곳 중 하나라도.
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
L=$D/dodge_${TAG}.log
mix() {   # $1 = run dir, $2 = update: the deployed teacher (solo policy on an empty road) next to it
python3 - <<PY
import torch, os, shutil
ck = torch.load("$R/$1/state_u$2.pt", map_location="cpu", weights_only=False)
a = dict(ck["args"]); a.update(base="$SOLO", base_reach=10.0, base_hold=10); ck["args"] = a
os.makedirs("$R/${1}mix", exist_ok=True)
torch.save(ck, "$R/${1}mix/state_u$2.pt")
shutil.copy("$R/$1/args.json", "$R/${1}mix/args.json")
PY
}
weights() {
python3 - <<PY
import torch
for p in "$1".split():
    ck = torch.load(p, map_location="cpu", weights_only=False)
    W = ck["model"]["pi.0.weight"]; cn = W.norm(dim=0)
    print(f"  {'/'.join(p.split('/')[-2:])}: 첫 층 열 가중치 -- 혼자 입력 {cn[:107].mean():.3f}, 라이다 {cn[107:215].mean():.3f}, 보이는 상자 칸 {cn[306:318].mean():.3f}, 보이는 통로 {cn[318:409].mean():.3f}")
PY
}
# (1) 비켜 가기
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}d_props --stage props $PL $COMMON --updates 100 \
  --noise-corr 0.9 --std-floor 0.05 0.2 --seed ${S}4 --init $R/${TAG}_props/state_u200.pt > $D/${TAG}d_props.log 2>&1 || { echo fail > $D/dodge_${TAG}.done; exit 1; }
mix ${TAG}d_props 100; mix ${TAG}b_props 100
echo "== (1) 비켜 가기 ${TAG}d u100 대 대조 ${TAG}b u100 (같은 시작·가격, 잡음만 다름)" > $L
weights "$R/${TAG}d_props/state_u100.pt $R/${TAG}b_props/state_u100.pt $R/${TAG}_props/state_u200.pt" >> $L
CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 -u $T/one_crate.py $MC 30x34 lineref=$R/srp25h2_propsmix/state_u200.pt \
  ${TAG}d_u100=$R/${TAG}d_propsmix/state_u100.pt ${TAG}b_u100=$R/${TAG}b_propsmix/state_u100.pt 2>&1 | grep "==" >> $L
echo "  상자 정책 혼자(깨끗한 맵) ${TAG}d u100: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${TAG}d_props/state_u100.pt $MC 60 2>&1 | grep '주행' | cut -c1-160)" >> $L
# (2) 선 되살리기
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}e_props --stage props $PL $COMMON --updates 100 \
  --noise-corr 0 --seed ${S}5 --init $R/${TAG}d_props/state_u100.pt > $D/${TAG}e_props.log 2>&1 || { echo fail > $D/dodge_${TAG}.done; exit 1; }
mix ${TAG}e_props 100
echo "== (2) 선 되살리기 ${TAG}e u100" >> $L
weights "$R/${TAG}e_props/state_u100.pt" >> $L
CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 -u $T/one_crate.py $MC 30x34 lineref=$R/srp25h2_propsmix/state_u200.pt \
  ${TAG}e_u100=$R/${TAG}e_propsmix/state_u100.pt 2>&1 | grep "==" >> $L
echo "  상자 정책 혼자(깨끗한 맵) ${TAG}e u100: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${TAG}e_props/state_u100.pt $MC 60 2>&1 | grep '주행' | cut -c1-160)" >> $L
echo "  선생(혼합) 혼자 ${TAG}e u100: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${TAG}e_propsmix/state_u100.pt $MC 60 2>&1 | grep '주행' | cut -c1-160)" >> $L
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $R/${TAG}e_propsmix/state_u100.pt $MC 40 30 777 $D/layouts/t_${TAG}e 2>&1 | grep "배치\|합계\|Error" > $D/layouts_t_${TAG}e.log
echo "  선생(혼합) 30배치 ${TAG}e u100: $(tail -n 1 $D/layouts_t_${TAG}e.log)" >> $L
echo done > $D/dodge_${TAG}.done
