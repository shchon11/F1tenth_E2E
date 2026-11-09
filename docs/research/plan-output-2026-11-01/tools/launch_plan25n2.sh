#!/bin/bash
# plan25n2 (11-09 21:35, 사용자 "ㅇㅋㅇㅋㅇㅋ"): plan25n u100에서 이어, plan 충돌 감점(1 s plan 발자국이 벽·상자에 닿으면 접촉 비용의
# 0.1배)만 끔. 근거: 뒷부분이 예측이 되자(가) 1 s 예측 오차(약 1 m)로 닿는 plan이 1.6 -> 16 %, 보상 0.0145 -> 0.004(실제 접촉 비용은 그대로)
# -- 실행하는 첫 0.1 s가 예측 오차로 벌을 받음, 그 뒤 속도 5.14 -> 4.95 m/s. 실제 벽·상자·차 접촉 비용은 그대로. critic은 이어 받고(입력 같음)
# 처음 3 update는 critic만(보상이 바뀜). 확인: 실제 벽·상자 접촉이 늘지 않아야 함(늘면 앞부분만 감점으로), 속도·보상 회복
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
until [ -e $R/plan25n/plan_u100.pt ]; do sleep 10; done
sleep 20
for p in $(pgrep -f "plan_[r]l --name plan25n "); do kill $p; done
echo done > $D/guard4070_1109n.done
echo done > $D/one_plan25n.done                 # plan25n's loops finish u100 and stop
sleep 10
echo done > $D/guard4070_1109b2.done
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25n2 --format v2 --updates 500 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25n/plan_u100.pt --init-critic --actor-freeze 3 --lr 1e-4 --lr-end 1e-4 --memory 128 --mem-chunk 32 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0" > $D/plan25n2.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25n2 " out/guard4070_1109n2.done > out/guard4070_1109n2.log 2>&1 &
setsid nohup bash tools/eval_loop_plan3.sh plan25n2 1 100 100 500 256 > out/eval_loop_plan25n2.log 2>&1 &
setsid nohup bash tools/ideal_videos.sh plan25n2 1 100 100 500 > out/ideal_videos_plan25n2.log 2>&1 &
mkdir -p $R/watch_plan25n2
setsid nohup bash tools/watch_latest_plan.sh watch_plan25n2 plan25n2 $D/one_plan25n2.done > out/watch_plan25n2.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] == "watch_plan25b2":
        e["run"] = "watch_plan25n2"
        e["label"] = "학습 중 · planner v2 + 기억(GRU) + plan 뒷부분 일관성 + 상자 치우기(plan25m u150에서 이어) · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("plan25n u100에서 이어 · plan 충돌 감점 끔(실제 접촉 비용은 그대로) · 첫 0.1 s만 강화학습, 뒷부분은 다음 plan을 정답으로 · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) plan25n2 시작" >> $D/switch_plan25n2.log
