#!/bin/bash
# plan25f(v3) 시작 (11-09 09:10): 제어 지연을 아는 plan, 명령 = 직전 명령 +- v2와 같은 폭, plan25b u1700(v2)에서 이어 시작(critic은 새로, 처음
# 10 update는 critic만), lr 1e-4 -> 5e-5. v3를 처음부터 배우면(plan25c/d/e, 시험 셋) 모두 서 있기로 굳었음.
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25f --format v3 --updates 1500 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25b/plan_u1700.pt --actor-freeze 10 --lr 1e-4 --lr-end 5e-5" > $D/plan25f.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25f " out/guard4070_1109f.done > out/guard4070_1109f.log 2>&1 &
setsid nohup bash tools/eval_loop_plan3.sh plan25f 1 100 100 2000 256 > out/eval_loop_plan25f.log 2>&1 &
setsid nohup bash tools/ideal_videos.sh plan25f 1 100 100 2000 > out/ideal_videos_plan25f.log 2>&1 &
mkdir -p $R/watch_plan25f
setsid nohup bash tools/watch_latest_plan.sh watch_plan25f plan25f $D/one_plan25f.done > out/watch_plan25f.log 2>&1 &
echo "$(date +%T) plan25f 시작" >> $D/switch_plan25f.log
