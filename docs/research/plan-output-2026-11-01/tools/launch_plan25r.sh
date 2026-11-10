#!/bin/bash
# plan25r (11-11, 사용자: "학습은 멈춰. 지금은 뭔가 전환점이 필요한 단계여" -> 논의 -> "ㅇㅋㅇㅋ 앞쪽으로 가자"): 처음부터 다시.
# plan25n8까지 바뀐 것: (1) 기억을 특징 처리 앞쪽에(--mem-front: GRU가 LiDAR 인코딩 + 상태를 읽고, 처리 2층이 그것들 + 기억 상태를 읽음)
# (2) 기억을 운전과 처음부터 같이(기억 없는 policy 1,900 update 위에 붙였던 것 대신; 조향 탐색 잡음 0.03에서 시작하지 않음)
# (3) 기억 보조 문제 없음(--mem-aux 0) (4) 보상의 남은 거리 고친 것(18a63d3). 그 밖은 plan25n8 그대로(2바퀴 구간 기억 학습, plan
# 뒷부분 예측, plan 충돌 감점 끔, 타이어 한계 95 %, 첫 걸음 속도 탐색 바닥 0.4), 학습률은 처음부터 했던 plan25b처럼 3e-4 -> 1e-4.
# 롤아웃이 plan25b의 두 배(128결정)라 1000 update = plan25b 2000 update 분량(약 34 h).
# 판정: 표의 다섯째 칸(기억 그대로 - 바퀴마다 기억 지움, 혼자·상자 계속)이 -쪽(기억을 두면 바퀴가 빠르거나 접촉이 적음)으로 가는가.
# plan25b가 같은 데이터 양에서 간 곳(빈 트랙 바퀴 시간)보다 느리지 않은가.
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan25r
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95" > $D/$N.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1111r.done > out/guard4070_1111r.log 2>&1 &
setsid nohup bash tools/eval_loop_plan5.sh $N 1 100 100 1000 256 > out/eval_loop5_$N.out 2>&1 &
setsid nohup bash tools/analysis_loop8.sh $N 25 25 1000 > out/analysis_loop8_$N.out 2>&1 &
setsid nohup bash tools/video_loop6.sh $N 25 25 1000 > out/video_loop6_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan25n8",):
        e["run"] = "watch_plan25r"
        e["label"] = "학습 중 · 처음부터 다시: planner v2 + 기억(GRU)을 특징 처리 앞쪽에 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-11 처음부터 · 기억(GRU)이 LiDAR 인코딩 + 상태를 읽고 plan을 만드는 층들이 기억 상태를 같이 읽음 · 기억 보조 문제 없음 · "
                     "보상의 남은 거리 고침(벽 틈으로 질러가던 것) · plan 타이어 한계 95 % · 기억은 12.8 s(약 2바퀴) 구간으로 학습 · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신 · 초반 checkpoint는 아직 못 달림")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작" >> $D/switch_$N.log
