#!/bin/bash
# plan26q (11-15 02:00). plan26l u200에서 이어(--init --init-critic --update-offset 200), plan26p(같은 출발점, 같은 설정)와 다른 것 하나:
#   4판 중 1판(--scene-replay 0.25)은 저장된 장면에서 5 s씩 다시 출발 -- 장면 = 6 m/s 넘게 달리던 차가 상자에 세게 닿기 1-3 s 전의 상태
#   (--scene-kinds box --scene-min-speed 6.0; 상자 배치·차 상태·policy 입력·기억까지; 나머지 3판이 장면을 만듦, 통계도 그 3판만).
# 질문: '빠르게 들어가다 늦게 보인 상자'를 더 자주 겪게 하면 그 상황에서 비키거나 줄이는 것을 배우나.
# 근거(11-15 01:17, out/edge1114/box_hits_plan26l_u200.png, research_log.md):
#   - plan26l u200 상자 세게 닿음의 44 %가 '1 s 전 6.5 m/s 넘게 · 상자가 1 s 안쪽에 보임' 칸(통과의 6 %, 그 칸 세게 닿음 22 %, 다른 칸 1-4 %).
#   - 장면: 안쪽 벽 뒤 상자를 평소 속도·평소 길로 지나며 1-3 cm만 비킴(안 닿은 통과는 평균 37 cm). 같은 칸 78 %는 비켜 지나감 = 피할 수 있음.
#   - 학습에서는 update마다 그런 통과가 약 50번(세게 약 10번) -> 드묾. 장면 다시 출발 시험(두 GPU, 4 update): 장면에서 다시 출발한 판의
#     43-50 %가 또 세게 닿음 = 반은 피하고 반은 닿는 자리(배울 거리가 있음).
#   - 레퍼런스: 실패 직전에서 다시 시작(ε-retrain 2024: 50-100 %, 효과 작음; Mila: 해로움) -> 여기서는 1-3 s 전(피할 시간이 있는 때), 25 %, 5 s.
# 판정(frozen exam1 ideal/uniform/256, plan26p 같은 update와 세션끼리):
#   반증 1: u250에서 '6.5 m/s 넘게 · 1 s 안쪽' 칸 세게 닿는 비율이 plan26p u250보다 뚜렷이 낮지 않음(u300에서 다시 봄).
#   반증 2: 빈 트랙 혼자 거리가 plan26p보다 10 m 넘게 짧음, 또는 전체 세게 닿음/세션이 뚜렷이 늚.
VEST=1.0
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26q
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=1 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --align-episodes --init /home/shchon11/f1sim_runs/plan26l/plan_u200.pt --init-critic --update-offset 200 --scene-replay 0.25 --scene-kinds box --scene-min-speed 6.0 --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 416 --dist-world 2 --dist-init tcp://127.0.0.1:29571 --dist-rank 1 > $D/${N}_rank1.log 2>&1" > /dev/null 2>&1 &
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --align-episodes --init /home/shchon11/f1sim_runs/plan26l/plan_u200.pt --init-critic --update-offset 200 --scene-replay 0.25 --scene-kinds box --scene-min-speed 6.0 --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 608 --dist-world 2 --dist-init tcp://127.0.0.1:29571 --dist-rank 0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1115q.done > out/guard4070_1115q.log 2>&1 &
CORE_VAL_ENVS=512 setsid nohup bash tools/core_loop.sh $N 225 25 1000 > out/core_loop_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + time.strftime(".bak_%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26k", "watch_plan26l", "watch_plan26m", "watch_plan26n", "watch_plan26o", "watch_plan26p"):
        e["run"] = "watch_plan26q"
        e["label"] = "학습 중 · plan26l u200에서 이어 + 빠르게 들어가 상자에 닿기 전 장면에서 다시 출발(4판 중 1판) · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = "11-15 plan26l u200에서 이어 · 4판 중 1판은 6 m/s 넘게 들어가다 상자에 세게 닿기 1-3 s 전 장면에서 5 s씩 · 그 밖은 plan26l 그대로 · 25 update마다 갱신"
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
setsid nohup bash tools/edge_loop.sh $N > out/edge_loop_$N.out 2>&1 &
echo "$(date +%T) $N 시작 (plan26l u200에서 이어 + 상자 장면 다시 출발 0.25)" >> $D/switch_$N.log
