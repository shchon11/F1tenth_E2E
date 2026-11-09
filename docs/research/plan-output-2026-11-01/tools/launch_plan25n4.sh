#!/bin/bash
# plan25n4 (11-10 03:25, 사용자 "기억은 필요해 ... 사각지대에 장애물이 있어서 사고나는거 ... Tta는 필수" -> 기억 고치기 B "ㄱㄱ"):
# plan25n3 u125에서 이어, 기억에 보조 문제: 0.1 s마다 기억 상태만으로 '앞 12 m 안 가까운 상자 3개의 자리(벽 뒤라 안 보이는 것 포함)'를
# 맞힘(critic이 받는 것과 같은 정보, plan_rl --mem-aux 1.0). LiDAR 특징 부분은 떼어 내 기억(GRU)과 맞히기 출력만 배우고, 운전은 맞히기
# 출력을 읽지 않음. 근거: 기억 구간 2바퀴(plan25n3) 100 update 동안 기억이 더 아는 것 +0 %p, 운전 영향 0.009 -- 강화학습 신호만으로는
# '지난 바퀴에 본 상자를 들고 있기'를 못 배움. 나머지 설정은 plan25n3 그대로(롤아웃 128, 기억 구간 128, 묶음 16).
# 판정(50 update 안에): 지금 안 보이는 앞 상자를 기억이 맞히는 비율이 확실히 오름, 배치 0 U턴 입구 상자에 두 번째 바퀴부터 덜 닿음,
# 학습 조건 표의 세게 닿음(상자) 감소 + 기억이 운전을 바꾸는 크기 증가
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
until [ -e $R/plan25n3/plan_u125.pt ]; do sleep 10; done
sleep 20
for p in $(pgrep -f "plan_[r]l --name plan25n3 "); do kill $p; done
echo done > $D/guard4070_1109n3.done
echo done > $D/one_plan25n3.done                 # plan25n3's loops finish u125 and stop
sleep 10
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25n4 --format v2 --updates 500 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25n3/plan_u125.pt --init-critic --lr 1e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --mem-aux 1.0" > $D/plan25n4.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25n4 " out/guard4070_1110n4.done > out/guard4070_1110n4.log 2>&1 &
setsid nohup bash tools/eval_loop_plan5.sh plan25n4 1 100 100 500 256 > out/eval_loop5_plan25n4.out 2>&1 &
setsid nohup bash tools/analysis_loop7.sh plan25n4 25 25 500 > out/analysis_loop7_plan25n4.out 2>&1 &
setsid nohup bash tools/video_loop6.sh plan25n4 25 25 500 > out/video_loop6_plan25n4.out 2>&1 &
mkdir -p $R/watch_plan25n4
setsid nohup bash tools/watch_latest_plan.sh watch_plan25n4 plan25n4 $D/one_plan25n4.done > out/watch_plan25n4.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] == "watch_plan25n3":
        e["run"] = "watch_plan25n4"
        e["label"] = "학습 중 · planner v2 + 기억(GRU, 2바퀴 구간 + 앞 상자 자리 맞히기) + plan 뒷부분 예측 + 상자 치우기 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("plan25n3 u125에서 이어 · 기억이 앞 12 m 상자 자리(안 보이는 것 포함)를 맞히도록 같이 학습 · 기억은 12.8 s(약 2바퀴) 구간으로 학습 · "
                     "plan 충돌 감점 끔(실제 접촉 비용은 그대로) · 콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) plan25n4 시작" >> $D/switch_plan25n4.log
