#!/bin/bash
# plan25b(v2) -> plan25c(v3, 제어 지연을 아는 plan) 전환 (11-09 05:45). plan25b의 u1700이 저장되면 학습만 멈추고(평가 반복·학습 조건 영상은
# u1700까지 마치고 one_plan25b.done을 보고 스스로 끝남), 4070에서 plan25c를 처음부터: plan_rl --format v3, 지킴이, 평가 반복(세 조건),
# 학습 조건 영상, 콘솔 watch_plan25c.
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
until [ -e $R/plan25b/plan_u1700.pt ]; do sleep 15; done
sleep 30
kill -CONT 2001608 2>/dev/null; kill 2001608 2>/dev/null
echo done > $D/guard4070_1108a.done
echo done > $D/one_plan25b.done
echo "$(date +%T) plan25b 학습 멈춤 (u1700)" >> $D/switch_plan25c.log
sleep 20
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25c --format v3 --updates 2000 --seed 1 --save-every 25" > $D/plan25c.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25c " out/guard4070_1109c.done > out/guard4070_1109c.log 2>&1 &
setsid nohup bash tools/eval_loop_plan3.sh plan25c 1 100 100 2000 256 > out/eval_loop_plan25c.log 2>&1 &
setsid nohup bash tools/ideal_videos.sh plan25c 1 100 100 2000 > out/ideal_videos_plan25c.log 2>&1 &
mkdir -p $R/watch_plan25c
setsid nohup bash tools/watch_latest_plan.sh watch_plan25c plan25c $D/one_plan25c.done > out/watch_plan25c.log 2>&1 &
echo "$(date +%T) plan25c 시작" >> $D/switch_plan25c.log
