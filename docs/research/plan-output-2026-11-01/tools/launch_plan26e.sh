#!/bin/bash
# plan26e (11-14 02:00, 사용자 11-13: "네트워크 구조자체에서 분석좀 하라고 / 계속 설정좀 바꾸고 깔짝대다가 안되네만 도대체 언제까지할거야").
# 처음부터. 보상·세계·네트워크는 plan25y 그대로, update 안에서 actor가 한 걸음에 쓰는 결정 수만 바꿈: 그 update의 policy gradient 잡음 크기에 맞춤.
# 질문: actor의 한 걸음이 gradient의 신호가 잡음만큼은 되게 결정을 모으면(초반엔 minibatch 하나, 신호가 약해지면 rollout 전체까지), plan25y보다
#       같은 update에서 더 빨리 달리고 덜 닿는가.
# 근거(plan25y; out/netaudit/grad_step_*, track_*; 메모리 net-structure-audit):
#   - u150: 학습 없이 rollout 하나(= update 한 번의 데이터)의 gradient 방향으로 0.1 옮기면 frozen exam 빈 트랙 +7.3 m·세게 닿음 -0.36/세션, 7개 평균
#     방향은 +20.0 m; 무작위 방향은 변화 없음. 그런데 학습은 그 방향으로 25 update에 ~0.02만 감.
#   - u150 update 안: minibatch(4,096 결정)의 policy gradient는 그 방향 위에서 부호가 무작위 -> actor가 update마다 ~1.0 흔들리고 rollout gradient의
#     부호가 update마다 뒤집힘. 16 minibatch를 합쳐 epoch마다 한 걸음: 흔들림 0.16, 8 update 뒤 빈 트랙 +6.3 m(minibatch마다는 -3.3 m).
#   - 그러나 처음부터 16개 합치기(plan26d, u13에서 멈춤): actor 걸음이 update당 4번이라 초반 KL이 plan25y의 1/100~1/300, u12 바퀴 0.06 vs 0.26/분.
#   - policy gradient 잡음 크기(McCandlish et al. 2018 "An Empirical Model of Large-Batch Training": 신호와 잡음이 같아지는 batch 크기 =
#     tr(Sigma)/|G|^2, 두 batch 크기의 gradient 크기로 잼; PPO 포함 RL에서도 학습이 진행되며 커진다고 보고): plan25y u25 ~5k, u50 ~38k,
#     u150 25k~1.4M 결정. minibatch는 4,096 -> 초반엔 minibatch마다 걷는 것이 맞고, u50 무렵부터는 여러 개를 합쳐야 함.
# 시험하는 변화: --accum-actor-auto (update마다 epoch 전에 rollout에서 policy gradient 잡음 크기를 재고(걸음 없이, minibatch별 gradient와 그 평균),
#   update 사이로 고르게(0.9) 한 값을 덮는 가장 작은 2의 거듭제곱 개의 minibatch를 합쳐 actor가 한 걸음; 1..16. critic은 따로 minibatch마다).
# 뺀 것: 속도 짐작 손실 무게 낮추기(8 update 측정에서 효과 없음), 탐색 잡음 잇기(흔들림은 그대로; 다음 후보).
# 그 밖은 plan25y 그대로(트랙 33개(크기 0.58-1.19), 상대 차 다시 출발 2 s, LiDAR 자리 CoG + 0.126 m, 시드 1, 학습률 3e-4 -> 1e-4 / 1000 update,
#   기억 128 앞쪽 + 기억 value 0.15 + 붙드는 문 30 s, 롤아웃 128, 타이어 한계 95 %, plan 충돌 감점 끔, plan 뒷부분은 다음 plan 예측으로 학습).
# 판정(frozen exam1 ideal/uniform/256, plan25y 같은 update와 세션끼리 짝지어; exam_judge.sh가 u25/u50/u100/u150에 기록):
#   반증 1: u100·u150의 빈 트랙 혼자 거리가 plan25y 같은 update보다 크지 않음(95 % 구간이 0을 넘지 못함).
#   반증 2: 세게 닿음/세션(벽+상자+차)이 plan25y 같은 update보다 많음.
#   반증 3(초반): u25에 plan25y u25(빈 트랙 한 바퀴 33.4 s, 90 s에 102 m)보다 뚜렷이 못 달림.
#   읽기: 학습 로그의 '잡음 크기 -> 합친 minibatch 수'(초반 1~2, u50 무렵 8~16이어야 측정과 맞음), KL·잘림.
VEST=1.0
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26e
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --accum-actor-auto --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1114e.done > out/guard4070_1114e.log 2>&1 &
setsid nohup bash tools/eval_loop_plan5.sh $N 1 100 100 1000 256 > out/eval_loop5_$N.out 2>&1 &
setsid nohup bash tools/analysis_loop10.sh $N 25 25 1000 > out/analysis_loop10_$N.out 2>&1 &
setsid nohup bash tools/video_loop7.sh $N 25 25 1000 > out/video_loop7_$N.out 2>&1 &
setsid nohup bash -c "for u in 25 50 100 150; do bash tools/exam_judge.sh $N \$u; done" > out/exam_loop_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + time.strftime(".bak_%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26d",):
        e["run"] = "watch_plan26e"
        e["label"] = "학습 중 · 처음부터: actor 한 걸음에 쓰는 결정 수를 gradient 잡음 크기에 맞춤 (보상·세계는 plan25y 그대로) · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-14 02:00 처음부터 · update마다 policy gradient 잡음 크기를 재고 actor는 그만큼 minibatch를 합쳐 한 걸음(1~16개), critic은 그대로 · "
                     "그 밖은 plan25y 그대로(기억 value 0.15, 붙드는 문 30 s, 트랙 33개, LiDAR 자리 = 실제 차) · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신 · 초반 checkpoint는 아직 못 달림")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작 (VEST $VEST)" >> $D/switch_$N.log
