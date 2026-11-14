#!/bin/bash
# plan26m (11-14 밤, 처음부터). plan26l과 같고 두 가지만 다름: --noise-corr 0.9(탐색 잡음이 결정마다 새로 뽑히지 않고 AR(1) 0.9로 이어짐,
#   한 결정의 잡음 크기는 그대로) + --align-episodes(학습 속도, 결과 같음).
# 질문: 상자·느린 차 옆을 지나는 틈을 학습이 넓힐 수 있게 되나 -- 틈을 바꾸는 탐색이 batch에 들어가면 학습이 그것을 고르나.
# 근거(11-14 21:10-22:05 분석, out/edge1114/, out/rwexam/, out/batchdump/plan26l_dump150*):
#   - plan26l은 상자·차 옆을 더 좁게 지나서 빨라짐: 안 닿은 통과 틈 중앙 상자 21 cm(plan26i 34), 차 21 cm(31); 10 cm 안 통과는 다음 바퀴
#     8-12 % 닿음(10 cm 넘으면 1 %); 느린 차 만남 16 %에서 세게 닿음(+6.1 s vs 안 닿으면 +0.7 s).
#   - 학습 보상으로 다시 채점하면 plan26l u150 = plan26i u150(-1.3 +- 1.6 s): 빨라진 만큼 접촉 값으로 내줌.
#   - critic은 좁은 상자 통과를 지나기 0.3 s 전까지 넓은 통과와 똑같이 매김(AUC 0.40-0.54) -> 그 정보가 학습 신호에 없음.
#   - "닿을 뻔한 통과에도 예상 접촉 값"(후보 가)을 batch에 넣어 advantage를 다시 계산해도 1.5-6 m 앞 결정의 비키기 기울기 0 +- 0.01 s/도
#     -> 보상 쪽은 원인이 아님(가 버림).
#   - 원인: 결정 하나의 조향 잡음(약 2도)이 그 통과의 틈을 0.0-0.4 cm만 바꿈(다음 결정들이 되돌림). 그래서 틈에 대한 학습 신호가 원리상 0.
#   - 11-13 측정(plan25y): 잡음을 0.9로 이으면 1도가 0.5 s에 15 cm를 옮기고 상자 앞 조향 방향 신호가 batch에 나타남(0.122 s; 0.7과 흰
#     잡음에서는 0). 대가: rollout 안에서 세게 닿는 결정 0.98 -> 1.42 %. 레퍼런스: gSDE(Raffin 2021), PPO의 coloured noise(Hollenstein 2024).
# 판정(frozen exam1 ideal/uniform/256, plan26l 같은 update와 세션끼리 짝지어; 시험은 평균 행동이라 잡음과 무관):
#   반증 1: u150에서 안 닿은 상자 통과 틈 중앙이 plan26l u150(21 cm)보다 5 cm 넘게 넓지 않고, 세게 닿음/세션도 뚜렷이 적지 않음.
#   반증 2: u100-u150 빈 트랙 혼자 거리가 plan26l 같은 update보다 10 m 넘게 짧음(이어지는 잡음 때문에 조심스러운 plan만 배움).
#   반증 3: u50까지 빈 트랙 벽 세게 닿음이 plan26l보다 뚜렷이 많음(11-07: 이어지는 잡음이 처음 net을 벽에 박게 함), 또는 조향 std가
#     u100 전에 0.15 아래로 줄어 잡음이 사라짐(progress의 std[0]).
VEST=1.0
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26m
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=1 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --noise-corr 0.9 --align-episodes --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 416 --dist-world 2 --dist-init tcp://127.0.0.1:29567 --dist-rank 1 > $D/${N}_rank1.log 2>&1" > /dev/null 2>&1 &
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --noise-corr 0.9 --align-episodes --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 608 --dist-world 2 --dist-init tcp://127.0.0.1:29567 --dist-rank 0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1114m.done > out/guard4070_1114m.log 2>&1 &
CORE_VAL_ENVS=512 setsid nohup bash tools/core_loop.sh $N 25 25 1000 > out/core_loop_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + time.strftime(".bak_%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26l",):
        e["run"] = "watch_plan26m"
        e["label"] = "학습 중 · 처음부터: plan26l + 여러 결정 이어지는 탐색 잡음(0.9) · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = "11-14 처음부터 · plan26l과 같고 탐색 잡음만 결정마다 새로 뽑지 않고 0.9씩 이어짐(약 1 s) · 콘솔은 평균 행동 · 25 update마다 갱신"
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
setsid nohup bash tools/edge_loop.sh $N > out/edge_loop_$N.out 2>&1 &
echo "$(date +%T) $N 시작 (plan26l + --noise-corr 0.9 --align-episodes)" >> $D/switch_$N.log
