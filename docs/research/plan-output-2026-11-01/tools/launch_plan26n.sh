#!/bin/bash
# plan26n (11-14 23:25, 처음부터). plan26l과 같고: u100까지 흰 잡음(plan26l과 같음), u100-150에 조향 출력의 탐색 잡음만 AR(1) 0 -> 0.9로
#   이어지게 올림(--noise-corr 0.9 --noise-corr-steer --noise-corr-ramp 100,150; 속도 잡음은 결정마다 새로) + --align-episodes(속도만).
# 질문: 잘 달리게 된 뒤 조향 탐색이 옆 틈을 실제로 바꾸게 하면, 학습이 상자·느린 차 옆을 더 넓게 지나 세게 닿음을 줄이나.
# 근거(11-14 21:10-23:20, out/edge1114/, out/rwexam/, out/batchdump/):
#   - plan26l은 상자·차 옆을 좁게 지나 빨라짐(안 닿은 통과 틈 중앙 상자 21 cm vs plan26i 34, 차 21 vs 31); 학습 보상은 둘을 같게 매김.
#   - critic은 좁은 상자 통과를 0.3 s 전까지 넓은 통과와 똑같이 매김. "예상 접촉 값"을 batch에 넣어도 비키기 기울기 0 +- 0.01 s/도(버림).
#   - 원인: 결정 하나의 조향 잡음(약 2도)이 통과 틈을 0.0-0.4 cm만 바꿈 -> 틈에 대한 학습 신호가 원리상 없음.
#   - 잡음을 이으면 1도가 0.5 s에 15 cm 옮기고 상자 앞 조향 신호가 batch에 생김(11-13, plan25y u150, 0.9에서만).
#   - plan26m(처음부터 모든 출력 0.9)은 무너짐: u25 빈 트랙 90 s에 1 m(plan26l 151 m). 같은 rollout 시험(stop_noise_test2/3.sh):
#     plan26l u25에 이으면 서 있는 비율 35-38 %(흰 잡음 7 %) -- 조향만 이어도 같음; plan26l u150에서는 1 %(셋 다), 학습 중 세게 닿음
#     0.79 -> 1.09/분, 벽 0.04 -> 0.20(조향만) / 0.34(모두) -> 달릴 줄 안 뒤에 조향만 이음.
# 판정(frozen exam1 ideal/uniform/256, plan26l 같은 update와 세션끼리 짝지어; 시험은 평균 행동):
#   확인: u100까지는 plan26l과 같은 수준(같은 설정; 다르면 시드 차이의 크기로 읽음).
#   반증 1: u200에서 안 닿은 상자 통과 틈 중앙이 plan26l u200보다 5 cm 넘게 넓지 않고, 세게 닿음/세션도 뚜렷이 적지 않음(u250에서 다시 봄).
#   반증 2: u200 빈 트랙 혼자 거리가 plan26l u200보다 10 m 넘게 짧음.
#   반증 3: u100-175 학습 중 서 있는 비율이 10 %를 넘거나 시험에서 선 세션이 뚜렷이 늚.
VEST=1.0
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26n
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=1 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --noise-corr 0.9 --noise-corr-steer --noise-corr-ramp 100,150 --align-episodes --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 416 --dist-world 2 --dist-init tcp://127.0.0.1:29568 --dist-rank 1 > $D/${N}_rank1.log 2>&1" > /dev/null 2>&1 &
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --noise-corr 0.9 --noise-corr-steer --noise-corr-ramp 100,150 --align-episodes --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 608 --dist-world 2 --dist-init tcp://127.0.0.1:29568 --dist-rank 0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1114n.done > out/guard4070_1114n.log 2>&1 &
CORE_VAL_ENVS=512 setsid nohup bash tools/core_loop.sh $N 25 25 1000 > out/core_loop_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + time.strftime(".bak_%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26l", "watch_plan26m"):
        e["run"] = "watch_plan26n"
        e["label"] = "학습 중 · 처음부터: plan26l + u100부터 조향 탐색 잡음을 이어 줌(0.9) · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = "11-14 처음부터 · u100까지 plan26l과 같음 · u100-150에 조향 잡음만 0 -> 0.9로 이어짐(속도 잡음은 그대로) · 콘솔은 평균 행동 · 25 update마다 갱신"
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
setsid nohup bash tools/edge_loop.sh $N > out/edge_loop_$N.out 2>&1 &
echo "$(date +%T) $N 시작 (plan26l + 조향 잡음 잇기 u100-150에 0 -> 0.9)" >> $D/switch_$N.log
