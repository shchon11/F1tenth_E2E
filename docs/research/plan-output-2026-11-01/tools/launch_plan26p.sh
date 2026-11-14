#!/bin/bash
# plan26p (11-15 01:20). plan26l u200의 policy·critic에서 같은 설정 그대로 이어 학습(--init --init-critic --update-offset 200; Adam만 새로)
#   (+ --align-episodes: 학습 속도만, 결과 같음).
# 질문: 지금까지 가장 빠른 plan26l이 u200 뒤에도 좋아지나, plan26i처럼 u200-300에 세게 닿음이 커지며 무너지나(plan26i u150 -> u300
#   세게 닿음 0.55 -> 2.23/세션; plan26l은 minibatch 합치는 수가 내려가지 않게 해서 그 되먹임은 막혀 있음).
# 근거: plan26o(u100-150에 조향 잡음 잇기)는 u200에서 느려지고(-13.6 m) 세게 닿음 +0.27로 실패; 그 batch에서도 이어진 잡음이 통과 틈을
#   1-2 cm만 바꾸고 비키기 신호는 잡음 속(+0.01-0.03 +- 0.02 s/도) -> 틈 쪽 방향을 접고, 가장 좋은 줄기를 계속 키우며 세게 닿은 장면을
#   종류별로 다시 봄. plan26o u200은 안 닿은 상자 틈이 넓어졌는데(19 -> 24 cm) 세게 닿음은 늘어 '좁은 틈이 원인' 가설도 약해짐.
# 판정(frozen exam1 ideal/uniform/256, plan26l u200 대비, 같은 세션끼리):
#   반증(무너짐): u250 또는 u300에서 세게 닿음/세션이 plan26l u200보다 +0.3 넘게 늘거나 빈 트랙 거리가 10 m 넘게 줆.
VEST=1.0
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26p
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=1 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --align-episodes --init /home/shchon11/f1sim_runs/plan26l/plan_u200.pt --init-critic --update-offset 200 --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 416 --dist-world 2 --dist-init tcp://127.0.0.1:29570 --dist-rank 1 > $D/${N}_rank1.log 2>&1" > /dev/null 2>&1 &
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --align-episodes --init /home/shchon11/f1sim_runs/plan26l/plan_u200.pt --init-critic --update-offset 200 --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 608 --dist-world 2 --dist-init tcp://127.0.0.1:29570 --dist-rank 0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1114p.done > out/guard4070_1114p.log 2>&1 &
CORE_VAL_ENVS=512 setsid nohup bash tools/core_loop.sh $N 225 25 1000 > out/core_loop_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + time.strftime(".bak_%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26k", "watch_plan26l", "watch_plan26m", "watch_plan26n", "watch_plan26o"):
        e["run"] = "watch_plan26p"
        e["label"] = "학습 중 · plan26l u200에서 같은 설정으로 이어 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = "11-15 01:20 plan26l(지금까지 가장 빠름) u200에서 이어 학습 · 설정은 plan26l 그대로 · 25 update마다 갱신"
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
setsid nohup bash tools/edge_loop.sh $N > out/edge_loop_$N.out 2>&1 &
echo "$(date +%T) $N 시작 (plan26l u200에서 같은 설정으로 이어)" >> $D/switch_$N.log
