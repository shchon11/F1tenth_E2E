#!/bin/bash
# plan25n6 (11-10 16:05, 사용자 "ㅇㅇ": plan끼리 지그재그를 덜 하게 작은 비용): plan25n5 u350에서 이어 --smooth-reg 0.005 --
# 연속한 결정 사이 첫 단계 조향 행동의 변화(조향이 꺾이는 방향이 뒤집히는 것)를 횡저크 단위 v^2 ds/L / 4 m/s^2의 Huber로, policy 평균에.
# 근거: 콘솔 경로(물리 차 + 새 제어기) 조향 떨림 3.24 -> 4.30 deg/0.1 s(지그재그 27 -> 34 %), 학습 조건 지그재그 10 -> 14 %, 원인 =
# plan의 첫 단계 조향이 결정마다 뒤집히는 것 x 한계 근처 조향 표의 3-4배 키움(표 입력을 평균 내면 벽 6배, 표를 끄면 1.6 s 느림).
# 판정(50 update 안에): 학습 조건 지그재그 14 -> 5 % 아래, 콘솔 경로 조향 떨림 4.3 -> 2 deg 아래, 빈 트랙 바퀴 그대로.
# 나머지는 plan25n5 그대로(기억 보조 0.3, 트랙 기준 상자 12 m, 롤아웃 128, 기억 구간 128, 묶음 16)
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
for p in $(pgrep -f "plan_[r]l --name plan25n5 "); do kill $p; done
echo done > $D/guard4070_1110n5.done
echo done > $D/one_plan25n5.done                 # plan25n5's loops finish what they have and stop
sleep 10
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25n6 --format v2 --updates 500 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25n5/plan_u350.pt --init-critic --lr 1e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --mem-aux 0.3 --box-range 12 --smooth-reg 0.005" > $D/plan25n6.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25n6 " out/guard4070_1110n6.done > out/guard4070_1110n6.log 2>&1 &
setsid nohup bash tools/eval_loop_plan5.sh plan25n6 1 100 100 500 256 > out/eval_loop5_plan25n6.out 2>&1 &
setsid nohup bash tools/analysis_loop7.sh plan25n6 25 25 500 > out/analysis_loop7_plan25n6.out 2>&1 &
setsid nohup bash tools/video_loop6.sh plan25n6 25 25 500 > out/video_loop6_plan25n6.out 2>&1 &
mkdir -p $R/watch_plan25n6
setsid nohup bash tools/watch_latest_plan.sh watch_plan25n6 plan25n6 $D/one_plan25n6.done > out/watch_plan25n6.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan25n5",):
        e["run"] = "watch_plan25n6"
        e["label"] = "학습 중 · planner v2 + 기억(GRU, 2바퀴 구간 + 앞 상자 자리 맞히기) + plan 뒷부분 예측 + 상자 치우기 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("plan25n5 u350에서 이어 · plan 지그재그 비용 · 기억이 트랙을 따라 앞 12 m 상자 자리(안 보이는 것 포함)를 맞히도록 같이 학습 · 기억은 12.8 s(약 2바퀴) 구간으로 학습 · "
                     "plan 충돌 감점 끔(실제 접촉 비용은 그대로) · 콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) plan25n6 시작" >> $D/switch_plan25n6.log
