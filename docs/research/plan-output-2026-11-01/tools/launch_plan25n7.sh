#!/bin/bash
# plan25n7 (11-10 17:20): plan25n6(지그재그 비용 0.005) 25-34 update 동안 비용 1.23 -> 1.27 그대로, u25 흔들림 학습 조건 지그재그 14 -> 12 %,
# 콘솔 경로 조향 떨림 4.30 -> 3.82 deg로 조금만 -> 말씀드린 대로 u25에서 이어 무게 4배(0.02). 나머지 그대로. 판정 같음(학습 조건 지그재그 5 %
# 아래, 콘솔 경로 2 deg 아래, 빈 트랙 바퀴 그대로).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
for p in $(pgrep -f "plan_[r]l --name plan25n6 "); do kill $p; done
echo done > $D/guard4070_1110n6.done
echo done > $D/one_plan25n6.done                 # plan25n6's loops finish what they have and stop
sleep 10
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25n7 --format v2 --updates 500 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25n6/plan_u25.pt --init-critic --lr 1e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --mem-aux 0.3 --box-range 12 --smooth-reg 0.02" > $D/plan25n7.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25n7 " out/guard4070_1110n7.done > out/guard4070_1110n7.log 2>&1 &
setsid nohup bash tools/eval_loop_plan5.sh plan25n7 1 100 100 500 256 > out/eval_loop5_plan25n7.out 2>&1 &
setsid nohup bash tools/analysis_loop7.sh plan25n7 25 25 500 > out/analysis_loop7_plan25n7.out 2>&1 &
setsid nohup bash tools/video_loop6.sh plan25n7 25 25 500 > out/video_loop6_plan25n7.out 2>&1 &
mkdir -p $R/watch_plan25n7
setsid nohup bash tools/watch_latest_plan.sh watch_plan25n7 plan25n7 $D/one_plan25n7.done > out/watch_plan25n7.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan25n6",):
        e["run"] = "watch_plan25n7"
        e["label"] = "학습 중 · planner v2 + 기억(GRU, 2바퀴 구간 + 앞 상자 자리 맞히기) + plan 뒷부분 예측 + 상자 치우기 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("plan25n6 u25에서 이어 · plan 지그재그 비용(무게 0.02) · 기억이 트랙을 따라 앞 12 m 상자 자리(안 보이는 것 포함)를 맞히도록 같이 학습 · 기억은 12.8 s(약 2바퀴) 구간으로 학습 · "
                     "plan 충돌 감점 끔(실제 접촉 비용은 그대로) · 콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) plan25n7 시작" >> $D/switch_plan25n7.log
