#!/bin/bash
# 실차 bag -> 그 지도의 학습 환경까지 한 번에 (11-15, 사용자: "백 데이터(ROS bag)를 넣으면 매핑까지 해 주면서 그 맵에 대한
# 학습 환경 준비까지 효율화해 주는 파이프라인").
#
# 쓰는 법:  setsid nohup bash new_map.sh <bag 폴더> <지도이름> [GPU번호, 기본 0 = 4070] > /dev/null 2>&1 &
#           진행 기록: ~/f1sim_runs/new_map_<지도이름>/log.txt   (다시 실행하면 끝난 단계는 건너뛰고 이어서 함)
#   <bag 폴더>: bag 하나(metadata.yaml 이 있는 폴더) 또는 한 장소의 bag 들을 담은 폴더 (가장 긴 것을 씀)
#   <지도이름>: 영문·숫자·_ (학습 이름 real:<지도이름>, 세계 plan_world_<지도이름>.json)
#   CHECK=1   싼 단계(1 지도, 2 등록·그림, 4 세계)만 하고, 오래 걸리는 3 raceline · 5 bank 는 할 일과 명령만 보여 줌
#   SOURCE=map|slam   지도 출처를 고정 (기본 auto),  COMPARE=1  /map 이 있으면 스캔 지도도 만들어 맞대 본 그림을 남김
#   WORKERS=N 동시 프로세스 수 (기본: 비어 있는 코어 - 1),  BANK_N=400 트랙당 상자 배치 수
#   RUNS / WORLD_DIR / F1SIM_MAPS   출력 위치 (기본 ~/f1sim_runs, f1sim/f1sim/learn/configs, ~/.f1sim/maps)
#
# 단계 (시간은 이 컴퓨터 기준):
#  1) bag -> 지도 (python -m f1sim.bag_map)                                                     수 초 ~ 30 초
#     /map 이 있고 크기 검사(파티클 필터 위치에서 스캔을 /map 벽에 맞춘 거리 배율이 1 ± 3 %)를 통과하면 그 /map,
#     아니면 /scan + /odom 으로 직접 만듦 (깨끗하게 3바퀴 돈 구간; 충돌 전에서 끊음). 주행 방향과 차선 위 한 점도 bag 에서.
#  2) 등록 + 미리보기 그림 (python -m f1sim.slam_map --install): 지도 등록부(F1SIM_MAPS)에 들어가 코드 수정 없이
#     real:<지도이름> 으로 바로 쓰임 (콘솔 목록에도 뜸). 그림 = 벽(덕트/높은 벽), 차선 가운데, 진행 방향 화살표, 실제 주행 경로   수 초
#  3) raceline (상대 차 4종이 달리는 선, 원본 + 뒤튼 16개): 원본 먼저 (싱글스레드 20~70 분, 나눌 수 없음),
#     그다음 뒤튼 16개를 WORKERS 개씩 동시에 (원본 선에서 출발, 각 0.2~23 분)
#  4) 세계 json: Korea 2025 세계에서 tracks / prop_bank / exam_bank / contact_scale(= 6.59 / 원본 raceline 한 바퀴) 만 바꿈  수 초
#  5) 상자 bank: 학습용(17개 트랙, 트랙당 BANK_N개; layout_bank_par.py 로 WORKERS 개에 나눔; 한 프로세스로는 약 70 분)
#     + 시험용(원본만, 400개, 다른 시드; 몇 분). GPU 를 씀 (4070 이면 gpu_guard 가 그동안 학습을 멈춤)
#  6) 확인: 세계를 한 번 띄워 bank 가 모든 트랙에 있는지 보고, 요약
set -o pipefail
BAG=$1; MAP=$2; GPU=${3:-0}
if [ -z "$BAG" ] || [ -z "$MAP" ]; then echo "쓰는 법: bash new_map.sh <bag 폴더> <지도이름> [GPU번호]"; exit 1; fi
case "$MAP" in *[!A-Za-z0-9_]*) echo "지도 이름은 영문·숫자·_ 만: $MAP"; exit 1;; esac
BAG=$(readlink -f "$BAG")
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$TOOLS/../../../.." && pwd)
F=$REPO/f1sim
RUNS=${RUNS:-$HOME/f1sim_runs}
OUT=$RUNS/new_map_$MAP
WORLD_DIR=${WORLD_DIR:-$F/f1sim/learn/configs}
W=$WORLD_DIR/plan_world_${MAP}.json
TB=$RUNS/prop_banks_box2_${MAP}
EB=$RUNS/prop_banks_exam1_${MAP}
BANK_N=${BANK_N:-400}
mkdir -p "$OUT/map"
LOG=$OUT/log.txt
exec > >(tee -a "$LOG") 2>&1
# ROS 2 (rosbag2_py, 1단계) + 가상환경. activate.sh 는 홈에서 (레포 안에서 ROS setup 을 부르면 설치 스크립트가 돎)
cd ~ && source ~/f1tenth_ws/activate.sh > /dev/null
export PYTHONPATH=$F:$PYTHONPATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset F1SIM_RACELINE_CACHE_ONLY
cd "$OUT"
workers() { [ -n "$WORKERS" ] && { echo "$WORKERS"; return; }
  python3 -c "import sys; sys.path.insert(0, '$TOOLS'); from layout_bank_par import free_workers; print(free_workers())"; }
step() { echo; echo "=== $(date '+%m-%d %H:%M:%S') $*"; }
took() { echo "    ($(( ($SECONDS - $1) / 60 )) 분 $(( ($SECONDS - $1) % 60 )) 초)"; }
NAMES="real:$MAP"; for k in $(seq 0 15); do NAMES="$NAMES,real:$MAP+warp$k"; done
echo "### new_map $MAP  bag $BAG  레포 $REPO  $(date)  CHECK=${CHECK:-0}"

# ---------------------------------------------------------------- 1) bag -> 지도
step "1) bag -> 지도"
t=$SECONDS
if [ -f "$OUT/map/$MAP.yaml" ] && [ -f "$OUT/map/${MAP}_bag.json" ]; then
  echo "    있음: $OUT/map/$MAP.yaml (다시 만들려면 그 폴더를 비우세요)"
else
  nice -n 5 python3 -u -m f1sim.bag_map "$BAG" --name "$MAP" --out "$OUT/map" --source "${SOURCE:-auto}" \
    $( [ "${COMPARE:-0}" = 1 ] && echo --compare ) 2>&1 | grep -v "^\[INFO\]" || { echo "1단계 실패"; exit 1; }
fi
took $t

# ---------------------------------------------------------------- 2) 등록 + 그림
step "2) 등록 (real:$MAP) + 미리보기 그림"
t=$SECONDS
REG=$(python3 -c "from f1sim import map_registry as R; e=R.get('$MAP'); print(e['yaml'] if e else '')")
if [ -n "$REG" ] && [ -f "$OUT/${MAP}_preview.png" ]; then
  echo "    이미 등록됨: $REG"
else
  python3 -u -m f1sim.slam_map "$OUT/map/$MAP.yaml" --install "$MAP" --overwrite --from-bag "$OUT/map/${MAP}_bag.json" \
    --no-raceline --preview "$OUT/${MAP}_preview.png" 2>&1 | grep -v "UserWarning\|warnings.warn"
  python3 -c "from f1sim import map_registry as R; assert R.get('$MAP')" || { echo "2단계 실패: 등록이 안 됨 (위 보고의 problems 참고)"; exit 1; }
fi
echo "    그림: $OUT/${MAP}_preview.png"
took $t

# ---------------------------------------------------------------- 3) raceline
step "3) raceline: 원본 + 뒤튼 16개 (캐시 ~/.cache/f1sim/racelines)"
t=$SECONDS
missing() { python3 - "$NAMES" <<'PY'
import sys
from f1sim import maps
from f1sim.raceline import Raceline, RacelineCacheMiss
miss = []
for n in sys.argv[1].split(","):
    try:
        Raceline.build_cached(maps.load(n), cache_only=True, objective="min_time", a_lat=7.0, a_acc=6.5, a_brake=4.0)
    except RacelineCacheMiss:
        miss.append(n)
print(",".join(miss))
PY
}
MISS=$(missing | tail -1)
if [ -z "$MISS" ]; then
  echo "    17개 모두 캐시에 있음"
elif [ "${CHECK:-0}" = 1 ]; then
  echo "    없음 $(echo "$MISS" | tr ',' '\n' | wc -l)개. 할 일:"
  echo "      OMP_NUM_THREADS=1 python3 $F/scripts/prebuild_racelines.py real:$MAP 1                  # 원본, 20~70 분"
  echo "      OMP_NUM_THREADS=1 python3 $F/scripts/prebuild_racelines.py ${NAMES#real:$MAP,} $(workers)   # 뒤튼 16개, $(workers)개씩"
else
  case ",$MISS," in *",real:$MAP,"*)
    echo "    원본 (싱글스레드, 20~70 분)"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 5 python3 -u "$F/scripts/prebuild_racelines.py" "real:$MAP" 1 || { echo "3단계 실패 (원본)"; exit 1; } ;;
  esac
  N=$(workers)
  echo "    뒤튼 16개를 $N개씩 동시에"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 5 python3 -u "$F/scripts/prebuild_racelines.py" "${NAMES#real:$MAP,}" "$N" || { echo "3단계 실패 (뒤튼 것)"; exit 1; }
  MISS=$(missing | tail -1)
  [ -z "$MISS" ] || { echo "3단계 뒤에도 없는 raceline: $MISS"; exit 1; }
fi
took $t

# ---------------------------------------------------------------- 4) 세계 json
step "4) 세계 json: $W"
t=$SECONDS
python3 - "$MAP" "$W" "$TB" "$EB" "$WORLD_DIR" "$F" <<'PY' || { echo "4단계 실패"; exit 1; }
import json, os, sys
m, w, tb, eb, wd, f = sys.argv[1:7]
tmpl = os.path.join(f, "f1sim", "learn", "configs", "plan_world_korea25.json")
a = json.load(open(w)) if os.path.exists(w) else json.load(open(tmpl))
a.update(tracks=f"real:{m}", prop_bank=tb, exam_bank=eb, name=f"plan_world_{m}")
lap = None
try:
    from f1sim import maps
    from f1sim.raceline import Raceline
    lap = float(Raceline.build_cached(maps.load(f"real:{m}"), cache_only=True, objective="min_time",
                                      a_lat=7.0, a_acc=6.5, a_brake=4.0).lap_time)
except Exception:
    pass
if lap:
    a["contact_scale"] = round(6.59 / lap, 2)          # Korea 2025 의 접촉 무게와 같게 (plan_reward T_REF 7.5 s 가 Korea 2025 기준)
    print(f"    원본 raceline 한 바퀴 {lap:.2f} s -> contact_scale {a['contact_scale']}")
else:
    a.pop("contact_scale", None)
    print("    원본 raceline 이 아직 없어 contact_scale 은 비워 둠 (3단계 뒤에 다시 실행하면 채움)")
os.makedirs(wd, exist_ok=True)
json.dump(dict(sorted(a.items())), open(w, "w"), indent=1)
print(f"    tracks real:{m} (+ 뒤튼 {a.get('warps')}개), 학습 bank {tb}, 시험 bank {eb}")
PY
took $t

# ---------------------------------------------------------------- 5) 상자 bank
step "5) 상자 bank (학습: 17개 트랙 x $BANK_N, 시험: 원본 x 400)"
t=$SECONDS
have() { ls "$1"/*.pt > /dev/null 2>&1; }
TB_CMD="CUDA_VISIBLE_DEVICES=$GPU BANK_SHAPES=boxes python3 -u $TOOLS/layout_bank_par.py $W $TB/${MAP}.pt $BANK_N 32 300"
EB_CMD="CUDA_VISIBLE_DEVICES=$GPU BANK_SHAPES=boxes BANK_TRACKS=real:$MAP WORKERS=1 python3 -u $TOOLS/layout_bank_par.py $W $EB/${MAP}_exam1.pt 400 32 300"
if [ "${CHECK:-0}" = 1 ]; then
  have "$TB" && echo "    학습 bank 있음: $TB" || echo "    할 일 (raceline 17개가 있어야 함): $TB_CMD"
  have "$EB" && echo "    시험 bank 있음: $EB" || echo "    할 일: $EB_CMD"
else
  [ -n "$(missing | tail -1)" ] && { echo "5단계: raceline 이 다 있어야 함"; exit 1; }
  mkdir -p "$TB" "$EB"
  if have "$TB"; then echo "    학습 bank 있음: $TB"; else
    eval "nice -n 5 env $TB_CMD" || { echo "5단계 실패 (학습 bank)"; exit 1; }; fi
  if have "$EB"; then echo "    시험 bank 있음: $EB"; else
    eval "nice -n 5 env $EB_CMD" || { echo "5단계 실패 (시험 bank)"; exit 1; }; fi
fi
took $t

# ---------------------------------------------------------------- 6) 확인 + 요약
step "6) 확인 + 요약"
if [ "${CHECK:-0}" != 1 ]; then
  F1SIM_RACELINE_CACHE_ONLY=1 python3 - "$W" <<'PY' 2>&1 | tee "$OUT/check.txt" | grep -v "Warning\|warn("
import argparse, json, sys, torch
from f1sim.learn import state_rl as SR
a = argparse.Namespace(**json.load(open(sys.argv[1]))); a.envs, a.seed, a.device = 8, 1, "cpu"
SR.build_env(a, torch.device("cpu")); print("세계를 띄움")
PY
  grep -q "uniform instead" "$OUT/check.txt" && { echo "확인 실패: bank 가 일부 트랙에 없음"; exit 1; }
fi
python3 - "$MAP" "$OUT" "$W" <<'PY'
import json, os, sys
m, out, w = sys.argv[1:4]
b = json.load(open(os.path.join(out, "map", f"{m}_bag.json")))
wj = json.load(open(w)) if os.path.exists(w) else {}
src = {"map": "bag 안의 /map", "slam": "스캔으로 직접 만든 지도"}.get(b.get("source"), b.get("source"))
sc = (b.get("bagmap_scale") or {}).get("scale")
print(f"지도 real:{m}: {src} (bag {os.path.basename(b['bag'])}), 진행 방향 {'반시계' if b.get('direction') == 'ccw' else '시계'}"
      + (f", /map 크기 배율 {sc:.3f}" if isinstance(sc, float) and sc == sc else ""))
if b.get("compare"):
    c = b["compare"]
    print(f"스캔 지도 vs /map: 자유공간 IoU {c['free_iou']:.3f}, 차선 IoU {c['lane_iou']:.3f}, 벽 10 cm 일치 {c['wall_f1_10cm']:.3f}, "
          f"크기 {c['scale_similarity']:.3f}  그림 {c.get('png')}")
print(f"미리보기 {os.path.join(out, m + '_preview.png')}")
print(f"세계 {w}: contact_scale {wj.get('contact_scale', '아직 없음')}, 학습 bank {wj.get('prop_bank')}, 시험 bank {wj.get('exam_bank')}")
print(f"학습 시작: python3 -m f1sim.learn.plan_rl --world {w} --name <런이름> ...  (setsid nohup 으로)")
PY
echo "### 끝 $(date)  (전체 $(( SECONDS / 60 )) 분)"
