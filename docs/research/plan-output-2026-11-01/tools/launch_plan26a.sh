#!/bin/bash
# plan26a (11-13 09:20). 처음부터. 사용자 지시(11-12): "근거 있게 학습을 돌리라고. 내 허락이 필요하다는 말이 아니었음" -> 4070이 비어 있어
# 제 추천 값(②)으로 시작; 사용자가 다른 값(① 그대로 / ③ 대회 기준)을 고르면 그 값으로 다시 시작.
# 질문: 세게 닿는 값(보상의 시간 벌점)을 두 배로 하면, policy가 상자가 있는 세션에서만 늦게 보이는 코너 출구를 조심해서(느리게) 지나가
#       상자에 세게 닿는 것이 줄고, 빈 트랙 속도는 그대로인가.
# 근거(11-13 오늘의 실험, out/replay/nudge_* · slow_*, memory nudge-replay-cause-2026-11-13):
#   - 세게 닿는 것은 왼쪽·오른쪽을 못 정해서가 아님: 박기 1-3 s 전에 복사해 0.3 s 틀어 줘도 48 % -> 36-39 %, 곧 원래 줄로 돌아옴.
#     결과는 마지막 1 s에 갈림(그대로 복사본: 0.5 s 전 97 %, 1-3 s 전 36-53 %). 박는 상자는 대부분 코너 출구, 줄 위, 늦게 보임.
#   - 다가가는 속도가 지렛대: 1-3 s 전부터 1 s만 속도 0.8배 -> 48 % -> 14 %, 지난 바퀴에 박았던 상자 33 % -> 12 %; 대가 약 1.5 m(0.25 s).
#   - 지금 값(세게 닿으면 약 5 s 손해)에서는 늘 빠르게 가다 가끔 박는 것이 조심하는 것과 비슷한 값 -> RL이 조심을 배울 까닭이 약함.
#     실제 대회에서 박으면 손으로 2 m 뒤로 옮기고 다시 위치를 잡음 -> 실제 손해는 더 큼.
#   - 기억은 '이 세션에 상자가 있다'를 이미 앎(plan25y: 기억에서 읽은 구분 0.85) -> 상자 세션에서만 조심하는 것이 가능.
# 바꾼 것: --contact-scale 2.0 (모든 닿음의 시간 벌점 x2: 세게 3 m/s면 3.5 s -> 7 s, plan이 벽·상자를 뚫는 값도 같은 비율).
#   그 밖은 plan25y 그대로(경로 하나, 기억 가치 출력 0.15, 문 초기값 30 s, 트랙 33개, 시드 1, 1000 update). 경로 후보 둘·장면 다시 달리기는 뺌(plan25z에서 실패).
# 판정(u125·u150, plan25y 같은 update와 견줌):
#   반증 1: 학습 조건 평가의 상자 세게 닿음 /분이 plan25y보다 40 % 넘게 줄지 않음(plan25y u125 0.73 / 1.25, u150 ?).
#   반증 2(부작용): 빈 트랙 한 바퀴가 plan25y보다 0.1 s 넘게 느림(u125 8.30 s, u150 8.10 s) -> 그냥 느려지기만 함(예전 '값만 올리면 느려지기만' 재현).
#   반증 3: 시험(exam1, 상자 uniform 256세션)의 상자 세션 '세게 안 닿은 세션' 비율이 plan25y u150(54 % / 46 %)보다 15 %p 넘게 오르지 않음.
#   반증 4: 바퀴마다 기억을 지워도 상자 세션의 세게 닿음이 안 늘어남(= 조심이 기억에서 오지 않음; 늘 조심하거나 아예 안 함).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26a
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95
export F1SIM_LIDAR_MOUNT_X=0.12555
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25 --contact-scale 2.0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1113a.done > out/guard4070_1113a.log 2>&1 &
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
    if e["run"] in ("watch_plan25z",):
        e["run"] = "watch_plan26a"
        e["label"] = "학습 중 · 처음부터: 세게 닿는 값 2배 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-13 아침 처음부터 · 모든 닿음의 시간 벌점 2배(세게 3 m/s면 7 s) · 그 밖은 plan25y와 같음(경로 하나, 기억 가치 출력, "
                     "트랙 33개, 상대 차 다시 출발, LiDAR 자리 = 실제 차) · 콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · "
                     "25 update마다 갱신 · 초반 checkpoint는 아직 못 달림")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작" >> $D/switch_$N.log
