#!/bin/bash
# plan25b2 (11-09 13:20, 사용자: "1700"): plan25b(v2) u1700에서 이어 원래 계획 2000 update까지(300 update, u100 = 원래 u1800). policy는 u1700 그대로,
# critic은 새로(센터라인 입력 뺌), 처음 10 update는 critic만, lr 1.3e-4 -> 1e-4. 평가는 학습 조건 / 새 제어기 / MAP.
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25b2 --format v2 --updates 300 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25b/plan_u1700.pt --actor-freeze 10 --lr 1.3e-4 --lr-end 1e-4" > $D/plan25b2.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25b2 " out/guard4070_1109b2.done > out/guard4070_1109b2.log 2>&1 &
setsid nohup bash tools/eval_loop_plan3.sh plan25b2 1 100 100 300 256 > out/eval_loop_plan25b2.log 2>&1 &
setsid nohup bash tools/ideal_videos.sh plan25b2 1 100 100 300 > out/ideal_videos_plan25b2.log 2>&1 &
mkdir -p $R/watch_plan25b2
setsid nohup bash tools/watch_latest_plan.sh watch_plan25b2 plan25b2 $D/one_plan25b2.done > out/watch_plan25b2.log 2>&1 &
echo "$(date +%T) plan25b2 시작" >> $D/switch_plan25b2.log
