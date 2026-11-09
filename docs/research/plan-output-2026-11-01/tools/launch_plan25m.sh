#!/bin/bash
# plan25m (11-09 17:10, 사용자: "A1. 기억으로 적응 ㅇㅇ", "가속 한계도 ㄱㄱㄱ", 앞서 "ㅇㅋ용용" 붙잡기): (17:25 사용자 "plan25b2는 끝내": u225 = 원래 u1925에서 바로) 그
# checkpoint에서 이어 넷을 한 번에: (1) 기억 -- actor에 GRU 128(더하는 층 0에서 시작: 처음엔 그대로), 32결정 묶음으로 갱신 (2) 학습 때만 새
# plan의 20 %를 0.2-0.4 s 붙잡기 (3) 첫 단계 속도 출력의 탐색 표준편차 하한 0.4(직선 끝에서 최대 가속을 시도하도록) (4) 학습 세계에 대회처럼
# 상자 치우기(세계 설정 removal_p 0.5, 10-30 s; plan_rl이 지금까지 안 했음). 평가는 0.1 s마다 새 plan. 판단: 학습 조건 빈 트랙 바퀴 시간
# (plan25b2 끝 기준, 6.6 s 근처까지), plan과 실제 0.3/0.5 s 차이(12/44 cm), 상자 치운 뒤 빈 트랙 라인
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
echo done > $D/guard4070_1109b2.done
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25m --format v2 --updates 500 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25b2/plan_u225.pt --actor-freeze 10 --lr 1e-4 --lr-end 1e-4 --hold-p 0.2 --hold-len 2,4 --memory 128 --mem-chunk 32 --speed-std-floor 0.4" > $D/plan25m.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25m " out/guard4070_1109m.done > out/guard4070_1109m.log 2>&1 &
setsid nohup bash tools/eval_loop_plan3.sh plan25m 1 100 100 500 256 > out/eval_loop_plan25m.log 2>&1 &
setsid nohup bash tools/ideal_videos.sh plan25m 1 100 100 500 > out/ideal_videos_plan25m.log 2>&1 &
mkdir -p $R/watch_plan25m
setsid nohup bash tools/watch_latest_plan.sh watch_plan25m plan25m $D/one_plan25m.done > out/watch_plan25m.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] == "watch_plan25b2":
        e["run"] = "watch_plan25m"
        e["label"] = "학습 중 · planner v2 + 기억(GRU) + 학습 때 plan 붙잡기 + 상자 치우기(plan25b 2000에서 이어) · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("plan25b2 u225(= plan25b 1925 update)에서 이어 500 update · 학습 때만 새 plan의 20 %를 0.2-0.4 s 붙잡고 실행 · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) plan25m 시작" >> $D/switch_plan25m.log
