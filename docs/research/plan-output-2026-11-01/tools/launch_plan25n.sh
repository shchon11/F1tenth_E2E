#!/bin/bash
# plan25n (11-09 19:55, 사용자: "(가)가 낫겠네"): plan25m u150에서 이어. plan의 첫 0.1 s(조향·속도 출력 2개)만 강화학습의 행동(탐색 잡음,
# 로그 확률), 나머지 18개는 0.1 s 뒤 새 plan의 같은 시각(0-0.9 s) 자리·속도를 정답으로 배움 -- 직전 plan의 뒷부분만 끌어당기고 새 plan은
# 정답으로만(반응을 늦추는 압력 없음, 이벤트 조건 불필요), 출력층만 배움(운전하는 부분은 안 건드림). 붙잡기 끔. 기억 128, 속도 탐색 하한 0.4,
# 상자 치우기 그대로. critic은 입력이 달라(붙잡기 칸 없음) 새로, 처음 10 update는 critic만. 판단: plan과 실제 0.3/0.5 s 차이(12/43 cm,
# -0.9/-1.7 m/s), 학습 조건 빈 트랙 바퀴(plan25m u100 6.91 s), MAP 조건 바퀴(뒷부분이 정직해지면 MAP이 좋아져야 함)
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
echo done > $D/guard4070_1109b2.done
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25n --format v2 --updates 500 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25m/plan_u150.pt --actor-freeze 10 --lr 1e-4 --lr-end 1e-4 --memory 128 --mem-chunk 32 --speed-std-floor 0.4 --tail-consistency 1.0" > $D/plan25n.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25n " out/guard4070_1109n.done > out/guard4070_1109n.log 2>&1 &
setsid nohup bash tools/eval_loop_plan3.sh plan25n 1 100 100 500 256 > out/eval_loop_plan25n.log 2>&1 &
setsid nohup bash tools/ideal_videos.sh plan25n 1 100 100 500 > out/ideal_videos_plan25n.log 2>&1 &
mkdir -p $R/watch_plan25n
setsid nohup bash tools/watch_latest_plan.sh watch_plan25n plan25n $D/one_plan25n.done > out/watch_plan25n.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] == "watch_plan25b2":
        e["run"] = "watch_plan25n"
        e["label"] = "학습 중 · planner v2 + 기억(GRU) + plan 뒷부분 일관성 + 상자 치우기(plan25m u150에서 이어) · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("plan25m u150에서 이어 500 update · 첫 0.1 s만 강화학습, 뒷부분은 다음 plan을 정답으로 (붙잡기 없음) · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) plan25n 시작" >> $D/switch_plan25n.log
