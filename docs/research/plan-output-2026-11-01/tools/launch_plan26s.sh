#!/bin/bash
# plan26s (11-15 10:00). plan26r u600에서 이어(--update-offset 600), plan26r과 다른 것 하나: 상대 차가 2-4 m 앞(차 좌표, 옆 1.5 m 안)일 때도
#   조향 잡음을 0.9씩 이어 줌(--noise-corr-car 2.0,4.0; critic이 받는 상대 차 칸, policy 입력 아님). 상자 쪽(--noise-corr-box 1.5,4.0)은 그대로.
# 질문: 상자에서 효과가 있던 방식을 상대 차로 넓히면 남은 차 충돌(세션당 약 0.5)이 줄어드나.
# 근거(11-15 09:20-09:50, out/diag1115/car_nudge_value_*):
#   - plan26r u500/u550: 상자 충돌은 크게 줄었는데(통과당 5.1 -> 1.2-1.6 %) 차는 덜 줄어듦(세션당 -0.05~-0.16).
#   - 남은 차 충돌 51건(u550): forzaeth 23 · forzaeth_pred 12 · lane_switch 11 · interactive 5; 코너에서 옆으로 지나가다 닿는 것이 대부분.
#   - 복제 실험(plan26r u550, 상대 0.5-0.75배, 상자 없음): lane_switch 2-4 m 앞에서 그대로 22 % -> 반대쪽 2도 x 0.3 s 14 % / 1 s 느리게 12 %;
#     forzaeth는 짧은 비키기로 거의 안 줄어듦(10-13 %); 선 따라가는 차는 원래 1-2 %.
#   - 근거가 lane_switch 쪽에만 있음(차 충돌의 약 1/4) -> 기대 효과는 작음.
# 판정(frozen exam1, plan26r u600과 같은 세션끼리; 여러 checkpoint 묶어서):
#   반증 1: u650·u700에서 차 세게 닿음/세션이 plan26r u600보다 뚜렷이 적지 않음.
#   반증 2: 상자 통과당 세게 닿음이 plan26r u500-u600 수준(1.2-1.6 %)보다 뚜렷이 늘거나 빈 트랙이 10 m 넘게 줆.
VEST=1.0
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26s
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=1 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --align-episodes --init /home/shchon11/f1sim_runs/plan26r/plan_u600.pt --init-critic --update-offset 600 --noise-corr 0.9 --noise-corr-steer --noise-corr-box 1.5,4.0 --noise-corr-car 2.0,4.0 --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 416 --dist-world 2 --dist-init tcp://127.0.0.1:29573 --dist-rank 1 > $D/${N}_rank1.log 2>&1" > /dev/null 2>&1 &
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --align-episodes --init /home/shchon11/f1sim_runs/plan26r/plan_u600.pt --init-critic --update-offset 600 --noise-corr 0.9 --noise-corr-steer --noise-corr-box 1.5,4.0 --noise-corr-car 2.0,4.0 --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 608 --dist-world 2 --dist-init tcp://127.0.0.1:29573 --dist-rank 0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1115s.done > out/guard4070_1115s.log 2>&1 &
CORE_VAL_ENVS=512 setsid nohup bash tools/core_loop.sh $N 625 25 1000 > out/core_loop_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + time.strftime(".bak_%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26k", "watch_plan26l", "watch_plan26m", "watch_plan26n", "watch_plan26o", "watch_plan26p", "watch_plan26q", "watch_plan26r"):
        e["run"] = "watch_plan26s"
        e["label"] = "학습 중 · plan26r u600에서 이어 + 상대 차 2-4 m 앞에서도 조향 잡음 잇기"
        e["note"] = "11-15 plan26r u600에서 이어 · 상자 1.5-4 m 앞 또는 상대 차 2-4 m 앞(옆 1.5 m 안)일 때 조향 잡음이 0.9씩 이어짐"
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
setsid nohup bash tools/edge_loop.sh $N > out/edge_loop_$N.out 2>&1 &
echo "$(date +%T) $N 시작 (plan26r u600에서 이어 + 상자·상대 차 앞에서 조향 잡음 잇기)" >> $D/switch_$N.log
