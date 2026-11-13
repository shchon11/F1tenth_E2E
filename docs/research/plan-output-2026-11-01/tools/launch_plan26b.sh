#!/bin/bash
# plan26b (11-13 13:30). 처음부터. 사용자: "저번에 분명히 서 있는거에 보상 매겨서 후진하도록 유도했는데 뭐냐" -> 제안 (가)에 "가 ㄱㄱ".
# 질문: 세게 닿는 값 2배(plan26a)에 11-06의 서 있는 벌점을 다시 넣으면, (1) 상자 세션에서 늦게 보이는 코너 출구를 조심해 세게 박는 것이
#       줄고, (2) 박은 뒤 서 버리지 않고 스스로 빠져나와 다시 달리며, (3) 빈 트랙 속도는 그대로인가.
# 근거:
#   - 다가가는 속도가 지렛대(11-13 복사 실험, plan25y u150): 박기 1-3 s 전부터 1 s만 0.9배 -> 세게 박음 44 % -> 22 %, 0.8배 -> 48 % -> 14 %;
#     대가 0.13 s / 0.25 s. 지금 값(세게 닿으면 약 5 s)에서는 조심할 까닭이 약함 -> 닿는 값 2배.
#   - 서 있는 벌점이 11-07 새 보상에서 빠진 채였음(plan_reward: "No stall price"). 박은 뒤 5 s 안에 다시 못 달린 비율이 40-45 %
#     (u50 시험, plan25y·plan26a 모두), plan26a u100 영상: 상자를 치워 길이 비어도 40 s 끝까지 섬. 서 있는 것은 바퀴 보상을 못 받을 뿐.
#   - plan26a(닿는 값 2배만)는 u100까지 상자 세게 닿음이 plan25y와 비슷(u75 1.52/1.39 vs 1.86/1.41)하고 상자 트랙에서 서 있는 시간이 1.5배.
# 바꾼 것: --contact-scale 2.0 + --stall-s 1.0 --stall-vmin 1.0 --stall-window 2.0 (최근 2 s 동안 앞으로 간 거리가 초당 1 m보다 모자라면
#   모자란 몫만큼 초당 1 s 벌점; 서 있기·앞뒤로 흔들기·거꾸로 가기 모두; 빠르게 뒤로 빠져나오는 동안만 냄. 확인: 선 차 5 s에 2.9 s어치,
#   달리는 차 0). 그 밖은 plan25y 그대로.
# 판정(u125·u150, plan25y 같은 update와 견줌):
#   반증 1: 학습 조건 평가의 상자 세게 닿음 /분이 plan25y보다 40 % 넘게 줄지 않음(plan25y u125 0.73 / 1.25, u150 0.63 / 0.83).
#   반증 2: 시험(exam1 uniform)에서 세게 닿은 뒤 5 s 안에 다시 못 달린 비율이 plan25y(u50 43-44 %)의 절반 밑으로 안 내려감.
#   반증 3(부작용): 빈 트랙 한 바퀴가 plan25y보다 0.1 s 넘게 느림(u125 8.30 s, u150 8.10 s).
#   반증 4(부작용): 상자 트랙에서 서 있는 시간은 줄었는데 벽·상자 닿음(약한 것 포함)이 plan25y보다 늘어남(벌점을 피하려 마구 움직임).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26b
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95
export F1SIM_LIDAR_MOUNT_X=0.12555
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25 --contact-scale 2.0 --stall-s 1.0 --stall-vmin 1.0 --stall-window 2.0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1113b.done > out/guard4070_1113b.log 2>&1 &
setsid nohup bash tools/eval_loop_plan5.sh $N 1 100 100 1000 256 > out/eval_loop5_$N.out 2>&1 &
setsid nohup bash tools/analysis_loop10.sh $N 25 25 1000 > out/analysis_loop10_$N.out 2>&1 &
setsid nohup bash tools/video_loop7.sh $N 25 25 1000 > out/video_loop7_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + ".bak_" + time.strftime("%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26a",):
        e["run"] = "watch_plan26b"
        e["label"] = "학습 중 · 처음부터: 세게 닿는 값 2배 + 서 있는 벌점 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-13 오후 처음부터 · 모든 닿음의 시간 벌점 2배(세게 3 m/s면 7 s) + 최근 2 s 동안 앞으로 간 거리가 모자라면 서 있는 벌점 · 그 밖은 plan25y와 같음(경로 하나, 기억 가치 출력, "
                     "트랙 33개, 상대 차 다시 출발, LiDAR 자리 = 실제 차) · 콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · "
                     "25 update마다 갱신 · 초반 checkpoint는 아직 못 달림")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작" >> $D/switch_$N.log
