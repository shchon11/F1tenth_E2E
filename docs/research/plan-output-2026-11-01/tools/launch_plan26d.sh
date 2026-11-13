#!/bin/bash
# plan26d (11-14 새벽, 사용자 11-13: "네트워크 구조자체에서 분석좀 하라고 / 계속 설정좀 바꾸고 깔짝대다가 안되네만 도대체 언제까지할거야").
# 처음부터. 보상·세계·네트워크는 plan25y 그대로, update 안에서 actor가 걷는 방식만 바꿈.
# 질문: update마다 batch에 있는 학습 신호가 actor 가중치에 쌓이게 하면(actor는 rollout 전체를 합쳐 epoch마다 한 걸음), plan25y보다 같은 update에서
#       더 빨리 달리고 덜 닿는가.
# 근거(plan25y u150, 11-13 밤~11-14; out/netaudit/batch_grad_compare_*, grad_step_*, track_*; 메모리 net-structure-audit):
#   - 학습 없이 u150 actor를 rollout 7개의 평균 gradient 방향으로 0.1 옮기면 frozen exam(128 세션 x 90 s, 같은 세계)에서 빈 트랙 +20.0 +- 0.8 m,
#     rollout 하나(= update 한 번이 받는 데이터)의 방향으로도 +7.3 m·세게 닿음 -0.36/세션; 무작위 방향 0.1은 변화 없음(-0.3 m). batch에 신호는 있음.
#   - 그런데 학습은 그 방향으로 25 update에 약 0.02만 감(checkpoint u25..u150의 자리), 가중치 전체는 25 update에 5.9 움직임.
#   - update 안(--track-steps): minibatch(차 32대 x 12.8 s) 하나의 policy gradient는 그 방향 위에 거의 전부 있으나(cosine 최대 +-0.9) 부호가
#     minibatch마다 무작위 -> Adam 걸음 64번이 앞뒤로 오감, actor가 update마다 ~1.0 흔들림, 다음 rollout의 gradient는 그 흔들림을 되돌리는 쪽을
#     가리킴(rollout 전체 gradient의 부호가 update마다 뒤집힘: 8번 중 3번만 +).
#   - 같은 출발점·같은 첫 rollout에서 16 minibatch를 합쳐 epoch마다 한 걸음(--accum 16): 흔들림 0.13/update, rollout gradient 8번 모두 +,
#     좋은 방향으로 2배(0.025 vs 0.012), 8 update 뒤 빈 트랙 +6.3 +- 0.4 m (minibatch마다 걸은 쪽 -3.3 +- 0.3 m), 세게 닿음 변화 없음.
# 시험하는 변화: --accum-actor 16 (actor만 rollout 전체를 합쳐 epoch마다 한 걸음, 4 epoch = 4걸음; critic은 따로 Adam으로 지금처럼 minibatch마다
#   걸음 -- 처음부터 배울 때 critic이 늦지 않게; 측정은 둘 다 합친 --accum 16으로 했음).
#   속도 짐작 손실 무게는 그대로 1.0: 몸통 가중치에서 그 gradient가 policy의 ~18배라 낮춰 봤으나(--accum 16 + --vest-coef 0.1, 8 update)
#   좋은 방향으로 0.003만 가고(1.0일 때 0.025) 빈 트랙 -0.6 +- 0.2 m, 속도 짐작 오차 0.05 -> 0.12-0.14 m/s -> 걸림돌 아님(버림).
#   잡음 이음 0.7(minibatch마다 걸음, 8 update)은 빈 트랙 +2.3 m·세게 닿음 -0.29 +- 0.11 이나 흔들림은 그대로(0.97/update), 처음부터 배울 때 벽에
#   박던 11-07 기록이 있어 이번에는 안 넣음(이 run의 초반이 나오면 다음 후보).
# 그 밖은 plan25y 그대로(트랙 33개(크기 0.58-1.19), 상대 차 다시 출발 2 s, LiDAR 자리 CoG + 0.126 m, 시드 1, 학습률 3e-4 -> 1e-4 / 1000 update,
#   기억 128 앞쪽 + 기억 value 0.15 + 붙드는 문 30 s, 롤아웃 128, 타이어 한계 95 %, plan 충돌 감점 끔, plan 뒷부분은 다음 plan 예측으로 학습).
# 판정(frozen exam1 ideal/uniform/256, plan25y 같은 update와 세션끼리 짝지어; exam_judge.sh가 u25/u50/u100/u150에 기록):
#   반증 1: u100·u150의 빈 트랙 혼자 거리가 plan25y 같은 update보다 크지 않음(95 % 구간이 0을 넘지 못함).
#   반증 2: 세게 닿음/세션(벽+상자+차)이 plan25y 같은 update보다 많음.
#   반증 3(초반): u25/u50에 아직 못 달림(빈 트랙 한 바퀴 못 돎) -- 걸음 수가 16분의 1이라 처음 배우는 것이 늦을 수 있음 -> 그러면 원인 보고 후 판단.
#   읽기: 학습 표의 KL·잘림, 그리고 actor가 update마다 움직인 거리(학습 로그) -- 흔들림이 다시 커지면 이 변화가 처음부터의 학습에서는 안 먹힘.
VEST=${VEST:-1.0}
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26d
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --accum-actor 16 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1114d.done > out/guard4070_1114d.log 2>&1 &
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
    if e["run"] in ("watch_plan26c",):
        e["run"] = "watch_plan26d"
        e["label"] = "학습 중 · 처음부터: actor가 rollout 전체를 합쳐 epoch마다 한 걸음 (보상·세계는 plan25y 그대로) · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-14 새벽 처음부터 · update 안에서 actor 걸음을 minibatch마다(64번) -> rollout 전체를 합쳐 epoch마다(4번), critic은 그대로 · "
                     "그 밖은 plan25y 그대로(기억 value 0.15, 붙드는 문 30 s, 트랙 33개, LiDAR 자리 = 실제 차) · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신 · 초반 checkpoint는 아직 못 달림")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작 (VEST $VEST)" >> $D/switch_$N.log
