#!/bin/bash
# plan26r (11-15 03:25). plan26l u200에서 이어(--init --init-critic --update-offset 200), plan26p(같은 출발점, 같은 설정)와 다른 것 하나:
#   상자가 1.5-4 m 앞(트랙을 따라, 옆 1 m 안)에 있을 때만 조향 출력의 탐색 잡음이 결정마다 0.9씩 이어짐(--noise-corr 0.9 --noise-corr-steer
#   --noise-corr-box 1.5,4.0; 상자 위치는 critic이 받는 칸, policy 입력 아님). 그 밖의 결정은 지금처럼 결정마다 새로(결정의 2-5 %만 해당).
# 질문: 상자 앞에서 '일관되게 0.3 s 비키기'를 탐색이 실제로 해 보게 하면, 학습이 그 방향을 고르나.
# 근거(11-15 02:59-03:15, out/diag1115/, research_log.md):
#   - box_react.py(plan26l u200): policy는 상자를 보고 plan 1.0 s 점을 16-20 cm 옮기지만 바로 실행되는 0.5 s 점은 6 cm; 세게 닿는 결정의
#     plan은 62-74 %가 상자를 관통.
#   - box_replay.py 복제(plan26l u200): 상자 1.5-2.5 m 앞에서 2도 x 0.3 s 비키면 세게 4.9 -> 1.2 %, 2.5-4 m 3.8 -> 1.7 %, 진행 손실 없음
#     (보상 합 +0.21 +- 0.16 / +0.05 +- 0.08 s). 4 m 넘으면 효과 없음.
#   - 결정마다 새 잡음은 통과 틈을 0.4 cm만 바꿈; 모든 곳에서 이은 잡음(plan26o)은 1-2 cm 바꾸지만 벽 +0.14·느려짐 -> 효과가 있는 구간에서만 이음.
#   - 장면 반복(plan26q)·학습 세계 바꾸기로는 '빠르고 늦게 보인 상자' 칸(22-26 %)이 안 움직였음.
# 판정(frozen exam1 ideal/uniform/256, plan26p u250과 같은 세션끼리):
#   반증 1: u250에서 '빠르고 늦게 보인 상자' 칸 세게 닿는 비율이 plan26p(25.7 %)보다 뚜렷이 낮지 않고 전체 상자 세게 닿음도 안 줆(u300에서 다시 봄).
#   반증 2: 빈 트랙 혼자 거리가 10 m 넘게 짧거나 벽 세게 닿음이 뚜렷이 늚.
VEST=1.0
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26r
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=1 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --align-episodes --init /home/shchon11/f1sim_runs/plan26l/plan_u200.pt --init-critic --update-offset 200 --noise-corr 0.9 --noise-corr-steer --noise-corr-box 1.5,4.0 --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 416 --dist-world 2 --dist-init tcp://127.0.0.1:29572 --dist-rank 1 > $D/${N}_rank1.log 2>&1" > /dev/null 2>&1 &
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --align-episodes --init /home/shchon11/f1sim_runs/plan26l/plan_u200.pt --init-critic --update-offset 200 --noise-corr 0.9 --noise-corr-steer --noise-corr-box 1.5,4.0 --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 608 --dist-world 2 --dist-init tcp://127.0.0.1:29572 --dist-rank 0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1115r.done > out/guard4070_1115r.log 2>&1 &
CORE_VAL_ENVS=512 setsid nohup bash tools/core_loop.sh $N 225 25 1000 > out/core_loop_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + time.strftime(".bak_%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26k", "watch_plan26l", "watch_plan26m", "watch_plan26n", "watch_plan26o", "watch_plan26p", "watch_plan26q"):
        e["run"] = "watch_plan26r"
        e["label"] = "학습 중 · plan26l u200에서 이어 + 상자 1.5-4 m 앞에서만 조향 탐색 잡음을 이어 줌"
        e["note"] = "11-15 plan26l u200에서 이어 · 상자가 1.5-4 m 앞(옆 1 m 안)일 때만 조향 잡음이 결정마다 0.9씩 이어짐 · 그 밖은 plan26l 그대로"
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
setsid nohup bash tools/edge_loop.sh $N > out/edge_loop_$N.out 2>&1 &
echo "$(date +%T) $N 시작 (plan26l u200에서 이어 + 상자 앞에서만 조향 잡음 잇기)" >> $D/switch_$N.log
