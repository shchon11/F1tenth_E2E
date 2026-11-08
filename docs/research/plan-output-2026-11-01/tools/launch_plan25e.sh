#!/bin/bash
# plan25e(v3) 시작 (11-09 08:00): 제어 지연을 아는 plan, 조향 명령 = 직전 명령 +- 서보 0.1 s 회전(타이어 한계 안), 속도 명령 = 구간 시작 속도
# +- 1.5 m/s. plan25d(조향 = 0 기준 절대값)는 초반 벽 접촉이 v2의 40배라 서 있기로 굳음(u94 멈춤 70 %).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25e --format v3 --updates 2000 --seed 1 --save-every 25" > $D/plan25e.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25e " out/guard4070_1109e.done > out/guard4070_1109e.log 2>&1 &
setsid nohup bash tools/eval_loop_plan3.sh plan25e 1 100 100 2000 256 > out/eval_loop_plan25e.log 2>&1 &
setsid nohup bash tools/ideal_videos.sh plan25e 1 100 100 2000 > out/ideal_videos_plan25e.log 2>&1 &
mkdir -p $R/watch_plan25e
setsid nohup bash tools/watch_latest_plan.sh watch_plan25e plan25e $D/one_plan25e.done > out/watch_plan25e.log 2>&1 &
echo "$(date +%T) plan25e 시작" >> $D/switch_plan25e.log
