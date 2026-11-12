#!/bin/bash
# plan25z (11-13 새벽, 사용자: "2는 괜찮네." / "1도 같이 해야하는거 아님?" / "아이디어 다시 쭉 정리해보고 구현해"). 처음부터.
# 질문: 세게 닿기 1-3 s 전 순간에서 자주 다시 달리게 하고(1), plan을 둘(상자 왼쪽용·오른쪽용) 내고 하나를 고르게 하면(2),
#       policy가 상자의 어느 쪽으로 갈지 일찍 정해서 상자에 세게 닿는 것이 줄어드는가.
# 근거(plan25y u125, 같은 시험 256세션; policy 안을 들여다본 결과, memory policy-internals-2026-11-13):
#   - 상자는 봄(상자 빔에 gradient가 빔 비율의 2.5-7배), 그런데 쪽을 늦게 정함: 마지막 쪽이 정해진 때 상자까지 세게 닿음 1.89 m / 안 닿고 지나감 3.08 m,
#     1.5 m 밖에서 plan이 상자 중심 0.3 m 안을 지나가는 결정 44 % / 14 %. 세게 닿는 접근에서 plan이 상자를 뚫고 지나가는 결정 62-95 %(1.5-4 m).
#     = 정규분포 하나로 왼쪽·오른쪽을 평균하는 꼴 -> (2) 두 plan + 고르기(어느 쪽인지 정하는 출력).
#   - 결정적인 1-2 s는 학습 결정의 약 0.3 %, 같은 순간을 복사해 다시 달리면 72 %가 또 세게 닿음 = 그 순간의 상태가 정함
#     -> (1) 그 순간을 자주 연습: 경주의 1/8이 접촉 직전 장면에서 다시 시작(5 s), 장면은 보통 경주에서만 모음(사용자의 고른 상자 설계는 그대로).
# 바꾼 것: --branches 2 (plan 두 개 + 고르는 출력, 고른 plan 번호가 다음 입력; 0번이 늘 더 왼쪽, 처음 간격 0.5; 학습 때 고르기는 최소 10 %씩,
#            고른 것을 결정마다 90 % 유지) + --scene-replay 0.125 (세게 닿기 10-30결정 전 상태: 두 차·상자·물리값·policy 입력·기억 그대로 저장,
#            5 s episode, 장면 1024개 보관). 그 밖은 plan25y 그대로(기억 가치 출력 0.15, 문 초기값 30 s, 트랙 33개, 시드 1, 1000 update).
# 판정(u150, plan25y u125/u150의 같은 시험과 견줌; side_flip.py, plan_table.py):
#   반증 1: 세게 닿는 접근에서 마지막 쪽이 정해진 때 상자까지 거리가 2.5 m 미만(plan25y 1.9 m).
#   반증 2: 1.5 m 밖에서 plan이 상자 중심 0.3 m 안을 지나가는 결정이 25 % 넘음(plan25y 44 %).
#   반증 3: 정면 상자에서 두 plan이 상자의 다른 쪽을 지나가는 결정이 50 % 미만(= 둘이 갈라지지 않음, 고르기가 하는 일이 없음).
#   반증 4: 상자 바퀴의 세게 닿음이 plan25y 같은 update보다 안 줄어듦.
#   반증 5(부작용): 빈 트랙 바퀴가 plan25y 같은 update보다 0.2 s 넘게 느림(u125 8.30 s, u150 8.10 s).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan25z
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25 --branches 2 --branch-floor 0.1 --branch-keep 0.9 --scene-replay 0.125 --scene-lag 20 --scene-size 1024 --scene-episode-s 5 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1113z.done > out/guard4070_1113z.log 2>&1 &
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
    if e["run"] in ("watch_plan25y",):
        e["run"] = "watch_plan25z"
        e["label"] = "학습 중 · 처음부터: plan 두 개(왼쪽·오른쪽) + 고르기, 세게 닿기 직전 장면에서 다시 달리기 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-13 새벽 처음부터 · plan 두 개 중 하나를 고름(고른 번호가 다음 입력) · 경주의 1/8은 세게 닿기 1-3 s 전 장면에서 5 s씩 다시 달림 · "
                     "그 밖은 plan25y와 같음(기억 가치 출력, 트랙 33개, 상대 차 다시 출발, LiDAR 자리 = 실제 차) · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신 · 초반 checkpoint는 아직 못 달림")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작" >> $D/switch_$N.log
