#!/bin/bash
# plan25n5 (11-10 03:35): plan25n4의 보조 문제 정답이 직선거리 12 m(벽 건너편 반대 방향 직선의 상자까지)라 사용자 "이건 아니지" -> plan25n4는
# 시작 직후 멈추고, 앞 상자를 트랙을 따라(보상과 같은 벽 기준 남은 거리 장의 차이, 중심선 아님) 12 m 안으로 -- 기억 보조 문제의 정답과
# critic의 상자 칸 둘 다(plan_rl --box-range 12). 트랙 기준 12 m = 달리는 라인으로 약 15 m = 최고 속도로 약 2초(가장 짧은 길로 잰 한 바퀴
# 30.4 m, 달리는 라인 38.4 m). 첫 update KL이 작은 시험에서 0.1이라 보조 무게 0.3, critic이 바뀐 상자 칸에 맞추게 처음 3 update는 critic만.
# 나머지는 plan25n3 u125에서 그대로(롤아웃 128, 기억 구간 128, 묶음 16). 판정은 plan25n4와 같음.
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25n5 --format v2 --updates 500 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25n3/plan_u125.pt --init-critic --lr 1e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --mem-aux 0.3 --box-range 12 --actor-freeze 3" > $D/plan25n5.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25n5 " out/guard4070_1110n5.done > out/guard4070_1110n5.log 2>&1 &
setsid nohup bash tools/eval_loop_plan5.sh plan25n5 1 100 100 500 256 > out/eval_loop5_plan25n5.out 2>&1 &
setsid nohup bash tools/analysis_loop7.sh plan25n5 25 25 500 > out/analysis_loop7_plan25n5.out 2>&1 &
setsid nohup bash tools/video_loop6.sh plan25n5 25 25 500 > out/video_loop6_plan25n5.out 2>&1 &
mkdir -p $R/watch_plan25n5
setsid nohup bash tools/watch_latest_plan.sh watch_plan25n5 plan25n5 $D/one_plan25n5.done > out/watch_plan25n5.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan25n3", "watch_plan25n4"):
        e["run"] = "watch_plan25n5"
        e["label"] = "학습 중 · planner v2 + 기억(GRU, 2바퀴 구간 + 앞 상자 자리 맞히기) + plan 뒷부분 예측 + 상자 치우기 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("plan25n3 u125에서 이어 · 기억이 트랙을 따라 앞 12 m 상자 자리(안 보이는 것 포함)를 맞히도록 같이 학습 · 기억은 12.8 s(약 2바퀴) 구간으로 학습 · "
                     "plan 충돌 감점 끔(실제 접촉 비용은 그대로) · 콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) plan25n5 시작" >> $D/switch_plan25n5.log
