#!/bin/bash
# plan25s (11-11, 사용자: "알아서좀 해라 ... 난 학습 잘됐는지만 궁금해" -> 합의한 순서의 4단계 "학습 한 번"): plan25r u300에서 이어, 바뀐 것은
# 하나 -- 2 s 동안 0.5 m도 못 나아간 상대 차를 다시 출발시킴(--opp-restart-s 2, f48aa1c: 상자 있는 경주에서 상대 차가 걸려 있는 시간 24 -> 0.8 %).
# 내가 정한 것: (1) 나란히 출발은 그대로(세게 닿는 차 접촉의 7 %쯤, 옆에 붙어 달리는 것도 배울 장면) (2) plan 뒤쪽 점 비용은 끔 그대로
# (11-09 plan25n2: 1 s 예측 오차를 벌해 속도가 떨어져서 뺐음; u275 평가에서 그 비용은 실제 접촉 비용의 12배 크기) (3) 기억은 더 손대지 않고
# network 그대로(이어 받으려면 같아야 함). plan25r이 계획한 1000 update까지: 700 update, 학습률은 plan25r의 u300 값 2.4e-4 -> 1e-4 (약 23 h).
# plan25r의 학습 중 바퀴 수는 멈출 때도 오르고 있었음(u150 7.07 -> u307 7.78 /분).
# 판정: 표(학습 조건, 학습 차 512대)의 빈 트랙 / 상자+상대 한 바퀴와 세게 닿음이 시작 행(u300을 같은 세계에서 잰 것)보다 나아지는가.
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan25s
P=$R/plan25r/plan_u300.pt
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
[ -e $P ] || { echo "$P 없음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 700 --seed 1 --save-every 25 --init $P --init-critic --lr 2.4e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2" > $D/$N.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1111s.done > out/guard4070_1111s.log 2>&1 &
# the start row: plan25r u300 in this run's world (opponents restarted), the table's condition
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && F1SIM_OPP_RESTART_S=2 CUDA_VISIBLE_DEVICES=1 bash ../docs/research/plan-output-2026-11-01/tools/slot5060.sh timeout 2400 python3 ../docs/research/plan-output-2026-11-01/tools/plan_validate.py $P ideal 1024 40 '  u0 ' 2>&1 | grep '^  u0' >> $D/one_${N}_i.log; echo done > $D/start_$N.done" > out/start_$N.out 2>&1 &
setsid nohup bash tools/eval_loop_plan5.sh $N 1 100 100 700 256 > out/eval_loop5_$N.out 2>&1 &
setsid nohup bash tools/analysis_loop9.sh $N 25 25 700 > out/analysis_loop9_$N.out 2>&1 &
setsid nohup bash tools/video_loop7.sh $N 25 25 700 > out/video_loop7_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan25r",):
        e["run"] = "watch_plan25s"
        e["label"] = "학습 중 · planner v2 + 기억(GRU) · plan25r u300에서 이어 · 못 나아가는 상대 차 다시 출발 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-11 plan25r u300에서 이어 · 학습 세계에서 2 s 동안 0.5 m도 못 나아간 상대 차를 다시 출발시킴(상자에 걸려 서 있던 것) · "
                     "그 밖은 plan25r 그대로(plan 타이어 한계 95 %, plan 충돌 감점 끔) · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작" >> $D/switch_$N.log
