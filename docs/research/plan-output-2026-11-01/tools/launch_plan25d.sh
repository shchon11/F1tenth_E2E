#!/bin/bash
# plan25d(v3) 시작 (11-09 06:20): 제어 지연을 아는 plan, 속도 명령 = 구간 시작 속도 + 출력 x 1.5 m/s(plan25c는 [-2, 10] 절대값이라 처음부터 벽에
# 붙어 23 update 동안 박는 plan 95 %). 4070에서 처음부터: plan_rl --format v3, 지킴이, 평가 반복(세 조건), 학습 조건 영상, 콘솔 watch_plan25d.
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25d --format v3 --updates 2000 --seed 1 --save-every 25" > $D/plan25d.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25d " out/guard4070_1109d.done > out/guard4070_1109d.log 2>&1 &
setsid nohup bash tools/eval_loop_plan3.sh plan25d 1 100 100 2000 256 > out/eval_loop_plan25d.log 2>&1 &
setsid nohup bash tools/ideal_videos.sh plan25d 1 100 100 2000 > out/ideal_videos_plan25d.log 2>&1 &
mkdir -p $R/watch_plan25d
setsid nohup bash tools/watch_latest_plan.sh watch_plan25d plan25d $D/one_plan25d.done > out/watch_plan25d.log 2>&1 &
echo "$(date +%T) plan25d 시작" >> $D/switch_plan25d.log
