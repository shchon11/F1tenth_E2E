#!/bin/bash
# plan26h (11-14 09:22, 사용자: "난 더 빠른 속도를 원해서.. 오히려 update 시간은 늘었길래" -> "그냥 해 ㄱㄱ"). 처음부터.
# plan26g(u4에서 멈춤)와 같고 두 GPU를 '같은 데이터, 짧은 update'로 씀: 차 1024대를 4070 640 + 5060 384대로 나눔(update당 결정 65,536 =
#   plan26f와 같음). 어림: rollout 4070 27 + 0.085 s/대, 5060 26 + 0.148 s/대 -> 둘 다 약 82 s, update 약 95 s (한 GPU 125 s).
# rollout 시간 측정(09:15, 차 1024대, 동기화해서 함수별): 주행 한 걸음 시간의 68 %가 상대 차들의 행동 계산(0.025 s마다 planner),
#   시뮬레이터 물리 18 % -> 더 줄일 곳은 상대 차 계산(따로 다룸; 이 run의 세계는 그대로).
# (아래는 plan26g 머리말)
# 질문: plan이 1초 앞을 실제로 내다보게 배우고(뒷부분 = 차가 실제로 간 위치·속도, 그 손실이 몸통까지; 몸통을 당기는 힘은 policy의 절반으로
#       맞춤) update마다 데이터를 늘리면(두 GPU, 결정 약 1.6배), 같은 빠르기에서 세게 닿음이 plan26f보다 줄어드는가.
# 근거:
#   - plan26f(u161에서 멈춤, 반증 2): 빠르기는 plan25y보다 u100 +62 m / u150 +38 m(90 s 빈 트랙)이나 세게 닿음 1.10 / 1.25 vs 0.56 / 0.62 /세션.
#     plan이 타이어 한계 95 %를 넘는 결정 37-44 %, 상자를 늦게 피하다 벽·상자에 닿음. 상자 세게 닿음은 마지막 1초 안의 반응으로 정해짐(11-13).
#   - plan 뒷부분은 RL 행동이 아니고(첫 0.1 s만), 다음 plan을 따라 출력 층만 배워 1.0초 점이 실제로 간 곳에서 72-95 cm(plan26f).
#     같은 입력의 새 네트워크는 5 cm까지 맞힘(11-13) -> 정보는 있는데 안 배움.
#   - ② 확인(plan26f u75에서 8 update, frozen exam): 1.0초 점 95 -> 50 cm, 0.7초 점 38 -> 19 cm, 빈 트랙 +32 m, 세게 닿음 그대로.
#   - 두 GPU 시험(11-14): 매 update 두 GPU 가중치가 정확히 같음. 실제 크기(이 설정, 08:48-09:03): 한 GPU 1024대 65,536 결정 125 s/update,
#     4070 1024 + 5060 640대 106,496 결정 133 s/update(결정/초 1.53배); 5060 rollout이 7 s 늦어 608대로 맞춤.
#   - 후반 policy gradient 잡음 크기 2.5만-5만 결정(plan26f u115-157) -> 한 걸음에 더 많은 결정이 필요; 두 GPU는 걸음 수를 줄이지 않고 늘림.
# 시험하는 변화: --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 + 두 GPU(--dist-world 2: 4070 차 1024대 rank 0, 5060 차 608대 rank 1).
# 그 밖은 plan26f 그대로(critic 따로 걷기, 잡음 크기에 맞춰 합치기 frac 0.5, 세게 닿음 값 그대로(사용자 결정 전), 나머지는 plan25y).
# 판정(frozen exam1 ideal/uniform/256, plan26f 같은 update와 세션끼리 짝지어; u25/50/100/150):
#   반증 1: u100·u150 세게 닿음/세션이 plan26f(1.10 / 1.25)보다 적지 않음.
#   반증 2: 빈 트랙 혼자 거리가 plan26f 같은 update보다 10 m 넘게 짧음.
#   반증 3: plan 1.0초 점이 실제로 간 곳에서 u100에 40 cm 아래로 안 내려감(plan26f u100 76 cm; tools/plan_vs_driven.py).
# 분석은 핵심만(core_loop.sh); 5060이 학습도 하므로 학습 조건 평가는 차 512대(학습 차 256대).
VEST=1.0
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26h
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=1 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25 --envs 384 --dist-world 2 --dist-init tcp://127.0.0.1:29562 --dist-rank 1 > $D/${N}_rank1.log 2>&1" > /dev/null 2>&1 &
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25 --envs 640 --dist-world 2 --dist-init tcp://127.0.0.1:29562 --dist-rank 0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1114h.done > out/guard4070_1114h.log 2>&1 &
CORE_VAL_ENVS=512 setsid nohup bash tools/core_loop.sh $N 25 25 1000 > out/core_loop_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + time.strftime(".bak_%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26g",):
        e["run"] = "watch_plan26h"
        e["label"] = "학습 중 · 처음부터: plan 뒷부분 = 실제로 간 길(몸통까지) + 두 GPU · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-14 09:22 처음부터 · plan 0.2-1.0 s를 차가 실제로 간 위치·속도로 학습(몸통까지, policy의 절반 힘) · 4070+5060 두 GPU · "
                     "그 밖은 plan26f 그대로 · 콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작 (두 GPU 640+384대, 뒷부분 = 실제로 간 길)" >> $D/switch_$N.log
