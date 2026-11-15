#!/bin/bash
# 다른 지도로 같은 실험을 돌릴 준비 (11-14, 사용자: "이런 실험들을 맵만 싹 바꿔서 할 수 있게 ... 나중에 다른맵도 한번 돌려볼 수 있게").
# 쓰는 법: bash map_setup.sh 지도이름 [GPU번호, 기본 0 = 4070]      예: bash map_setup.sh korea_2026_competition
#   1) 학습 세계 json: Korea 2025 것을 복사해 tracks / prop_bank / exam_bank 를 바꾸고, 2)에서 contact_scale 을 씀 -> f1sim/f1sim/learn/configs/plan_world_<지도>.json
#   2) raceline 캐시(상대 차 4종이 이 선 위를 달림) 확인: 원본 + 뒤튼 16개 모두 있어야 함. 없으면 멈춤(f1sim/scripts/prebuild_racelines.py로 먼저 만듦)
#   3) 학습 상자 bank(원본 + 뒤튼 16개, 트랙당 400배치; 한 프로세스로 약 70분 -> 11-15부터 layout_bank_par.py 로 비어 있는 코어 수만큼 나눠서;
#      그동안 4070 학습은 gpu_guard로 멈춤)와
#      시험 상자 bank(실제 지도만, 400배치, 다른 시드; 몇 분)를 만듦
#   4) 세계를 한 번 띄워 bank가 모든 트랙에 있는지 확인("drawing uniform instead"가 나오면 실패)
# 그 뒤: launch에 --world <1의 json>. 시험·평가·영상 도구는 checkpoint가 학습한 세계를 따라감(world_of.py). 시험 bank는 prop_banks_exam1_<지도>.
# 접촉 값: plan_reward의 T_REF 7.5 s는 Korea 2025 기준이라 한 바퀴가 긴 지도에서는 접촉이 상대적으로 비싸짐.
#   같은 무게를 원하면 --contact-scale = 6.59 / 그 지도의 raceline 한 바퀴 시간 (Korea 2026 = 0.82). 아래 2)에서 그 값을 출력함.
set -e
MAP=$1; GPU=${2:-0}
[ -n "$MAP" ] || { echo "지도 이름이 필요함"; exit 1; }
F=/home/shchon11/F1tenth_E2E/f1sim; T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
W=$F/f1sim/learn/configs/plan_world_${MAP}.json
TB=/home/shchon11/f1sim_runs/prop_banks_box2_${MAP}; EB=/home/shchon11/f1sim_runs/prop_banks_exam1_${MAP}
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=$F:/home/shchon11/F1tenth_E2E/f1sim_ros F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 - "$MAP" "$W" "$TB" "$EB" <<'PY'
import json, sys
m, w, tb, eb = sys.argv[1:5]
a = json.load(open("/home/shchon11/F1tenth_E2E/f1sim/f1sim/learn/configs/plan_world_korea25.json"))
a["tracks"], a["prop_bank"], a["exam_bank"], a["name"] = f"real:{m}", tb, eb, f"plan_world_{m}"
a.pop("contact_scale", None)                             # Korea 2025's; this map's is written below from its own lap
json.dump(dict(sorted(a.items())), open(w, "w"), indent=1)
print("세계", w)
PY
python3 - "$W" <<'PY'
import argparse, json, sys, torch
from f1sim.learn import state_rl as SR
a = argparse.Namespace(**json.load(open(sys.argv[1]))); a.envs, a.seed, a.device, a.prop_bank = 8, 1, "cpu", ""
env = SR.build_env(a, torch.device("cpu"))               # RACELINE_CACHE_ONLY: a missing raceline stops here
names = getattr(env, "_track_names", None)
_, rls = SR.common.load_tracks(names, racelines=True, **dict(SR.common.teacher_limits(7.0, 6.5, 4.0), objective="min_time"))
lap = float(rls[0].lap_time) if hasattr(rls[0], "lap_time") else float("nan")
print(f"트랙 {len(names)}개 raceline 있음 | 원본 raceline 한 바퀴 {lap:.2f} s -> Korea 2025와 같은 접촉 무게: --contact-scale {6.59 / lap:.2f}")
w = json.load(open(sys.argv[1])); w["contact_scale"] = round(6.59 / lap, 2)       # 11-15: launch reads it from the world
json.dump(dict(sorted(w.items())), open(sys.argv[1], "w"), indent=1)
PY
mkdir -p $TB $EB
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
[ -n "$(ls $TB/*.pt 2>/dev/null)" ] || BANK_SHAPES=boxes CUDA_VISIBLE_DEVICES=$GPU python3 -u tools/layout_bank_par.py $W $TB/${MAP}.pt 400 32 300 > out/bank_${MAP}.log 2>&1
[ -n "$(ls $EB/*.pt 2>/dev/null)" ] || BANK_SHAPES=boxes BANK_TRACKS=real:$MAP WORKERS=1 CUDA_VISIBLE_DEVICES=$GPU python3 -u tools/layout_bank_par.py $W $EB/${MAP}_exam1.pt 400 32 300 > out/bank_${MAP}_exam1.log 2>&1
python3 - "$W" <<'PY' 2>&1 | tee /dev/stderr | grep -q "uniform instead" && { echo "bank가 일부 트랙에 없음"; exit 1; } || true
import argparse, json, sys, torch
from f1sim.learn import state_rl as SR
a = argparse.Namespace(**json.load(open(sys.argv[1]))); a.envs, a.seed, a.device = 8, 1, "cpu"
SR.build_env(a, torch.device("cpu")); print("세계 확인 끝")
PY
echo "준비 끝: --world $W"
