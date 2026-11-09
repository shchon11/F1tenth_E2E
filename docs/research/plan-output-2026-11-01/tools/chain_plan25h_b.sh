#!/bin/bash
# (11-09 16:00 다시 띄움: 콘솔 기본 제어기가 새 제어기로 바뀌어 안내 문구만 고침) plan25h (11-09 14:40, 사용자 "ㅇㅋ용용"): plan25b2가 끝나면(u300 = 원래 u2000) 그 checkpoint에서 이어, 학습 때만 새 plan의 20 %를
# 0.2-0.4 s 동안 새 plan 없이 실행(--hold-p 0.2 --hold-len 2,4: policy는 어느 plan이 붙잡힐지 모르니 plan 뒷부분도 실제 갈 길로 그려야 함).
# 평가·콘솔·실차는 지금처럼 0.1 s마다 새 plan. critic은 새로(붙잡기 남은 결정 수 입력 추가), 처음 10 update는 critic만, lr 1e-4.
# 기각선: 붙잡지 않는 평가에서 학습 조건 빈 트랙이 plan25b2 끝보다 0.2 s 넘게 느려지면 실패. plan과 실제 0.3/0.5 s 차이(u1700 12/44 cm)가
# 크게 줄면 성공.
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
until [ -e $R/plan25b2/plan_u300.pt ] && ! pgrep -f "plan_[r]l --name plan25b2 " > /dev/null; do sleep 60; done
sleep 30
echo done > $D/guard4070_1109b2.done
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25h --format v2 --updates 500 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25b2/plan_u300.pt --actor-freeze 10 --lr 1e-4 --lr-end 1e-4 --hold-p 0.2 --hold-len 2,4" > $D/plan25h.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25h " out/guard4070_1109h.done > out/guard4070_1109h.log 2>&1 &
setsid nohup bash tools/eval_loop_plan3.sh plan25h 1 100 100 500 256 > out/eval_loop_plan25h.log 2>&1 &
setsid nohup bash tools/ideal_videos.sh plan25h 1 100 100 500 > out/ideal_videos_plan25h.log 2>&1 &
mkdir -p $R/watch_plan25h
setsid nohup bash tools/watch_latest_plan.sh watch_plan25h plan25h $D/one_plan25h.done > out/watch_plan25h.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] == "watch_plan25b2":
        e["run"] = "watch_plan25h"
        e["label"] = "학습 중 · planner v2 + 학습 때 plan 붙잡기(plan25b 2000에서 이어) · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("plan25b2 u300(= plan25b 2000 update)에서 이어 500 update · 학습 때만 새 plan의 20 %를 0.2-0.4 s 붙잡고 실행 · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) plan25h 시작" >> $D/switch_plan25h.log
