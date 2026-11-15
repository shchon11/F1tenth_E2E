#!/bin/bash
# plan26t (11-15 16:10). plan26s가 정한 끝(u1000)에 이르러 같은 설정으로 이어 학습(--update-offset 1000, --updates 1500). 바꾼 것: 학습률을
#   plan26s가 끝난 값 1e-4로 고정(--lr 1e-4 --lr-end 1e-4; 이어 시작하며 1500 기준 일정으로 다시 올라가지 않게).
# 질문: 가장 좋은 줄기(plan26s)가 u1000 뒤에도 빨라지며 충돌 수준을 지키나. 배포는 최근 checkpoint 셋의 가중치 평균.
# 근거: plan26s u900·u950이 지금까지 가장 빠름(빈 트랙 +33-35 m), u850·900·950 평균이 세게 -0.36/세션(plan26p u250 대비).
# 판정(frozen exam1, plan26s u850·900·950 평균과 같은 세션끼리): u1100 뒤 평균(u1050·1100·1150)이 빠르기를 지키며 세게 닿음이 늘지 않는가.
VEST=1.0
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26t
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=1 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1500 --seed 1 --save-every 25 --lr 1e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --align-episodes --init /home/shchon11/f1sim_runs/plan26s/plan_u1000.pt --init-critic --update-offset 1000 --noise-corr 0.9 --noise-corr-steer --noise-corr-box 1.5,4.0 --noise-corr-car 2.0,4.0 --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 416 --dist-world 2 --dist-init tcp://127.0.0.1:29574 --dist-rank 1 > $D/${N}_rank1.log 2>&1" > /dev/null 2>&1 &
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1500 --seed 1 --save-every 25 --lr 1e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --align-episodes --init /home/shchon11/f1sim_runs/plan26s/plan_u1000.pt --init-critic --update-offset 1000 --noise-corr 0.9 --noise-corr-steer --noise-corr-box 1.5,4.0 --noise-corr-car 2.0,4.0 --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 608 --dist-world 2 --dist-init tcp://127.0.0.1:29574 --dist-rank 0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1115t.done > out/guard4070_1115t.log 2>&1 &
CORE_VAL_ENVS=512 setsid nohup bash tools/core_loop.sh $N 1025 25 1500 > out/core_loop_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + time.strftime(".bak_%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26k", "watch_plan26l", "watch_plan26m", "watch_plan26n", "watch_plan26o", "watch_plan26p", "watch_plan26q", "watch_plan26r", "watch_plan26s"):
        e["run"] = "watch_plan26t"
        e["label"] = "학습 중 · plan26s u1000에서 같은 설정으로 이어(학습률 1e-4 고정)"
        e["note"] = "11-15 plan26s u1000에서 이어 · 학습률 1e-4 고정 · 상자 1.5-4 m 앞 또는 상대 차 2-4 m 앞(옆 1.5 m 안)일 때 조향 잡음이 0.9씩 이어짐"
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
setsid nohup bash tools/edge_loop.sh $N > out/edge_loop_$N.out 2>&1 &
echo "$(date +%T) $N 시작 (plan26s u1000에서 같은 설정으로 이어)" >> $D/switch_$N.log
