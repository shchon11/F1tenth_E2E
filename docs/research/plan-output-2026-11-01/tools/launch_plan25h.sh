#!/bin/bash
# plan25h (11-11 저녁, 사용자: 제어기가 추종을 못 하는 게 출력 구조 탓인지 걱정 / "실험같은건 ... scratch로 돌려" / 제안에 "ㅇㅇ 그려").
# 측정(제어기 없이 plan을 그대로 달림, plan25s u150 빈 트랙): 0.1 / 0.2 / 0.3 / 0.5 s -> 한 바퀴 6.80 / 7.10 / 8.40 / 15.9 s, 벽에 세게
# 0 / 0.67 / 2.10 / 3.46 /분 = 새 제어기(7.30 s, 0.68)·MAP(7.80 s, 1.89)과 같은 크기. 그 policy는 plan의 첫 0.1 s만 실행되고 뒷부분은
# 다음 plan을 예측하게만 배움(--tail-consistency).
# 질문: plan의 뒷부분까지 실제로 달리는 조건으로 처음부터 학습하면, 지금 출력 구조(0.1 s 걸음 10개) 그대로 0.3-0.5 s를 그대로 달려도
#   버티는 policy가 나오는가.
# 바꾼 것 하나(plan25r의 처음부터 학습 대비): 뒷부분 예측(--tail-consistency 1.0)을 끄고 모든 plan을 0.1-0.5 s(고르게) 그대로 달림
#   (--hold-p 1.0 --hold-len 1,5: 출력 20개가 모두 강화학습의 행동, 어느 plan이 얼마나 붙잡힐지 policy는 모름). 차·상자 근처에서도 붙잡음
#   (--hold-clear 0): 11-09의 우려(반응이 늦어져 소심해짐)는 가정하지 않고 상대 차 경주의 속도로 잼. 그 밖은 plan25r 그대로
#   (시드 1, 학습률 3e-4 -> 1e-4 / 1000 update 일정, 기억 128 앞쪽, 롤아웃 128, 타이어 한계 95 %, plan 충돌 감점 끔).
# 비교 기준: plan25r(같은 코드로 처음부터, u25-u300 checkpoint 있음)의 같은 update. 판정 시점 u300(약 10 h).
# 판정 숫자(tools/hold_loop.sh, 50 update마다): 빈 트랙에서 plan을 0.3 / 0.5 s 그대로 달렸을 때 한 바퀴와 벽에 세게 닿음.
# 반증: (1) u300에서 0.3 s 붙잡기가 plan25r u300만큼 무너짐 (2) 0.1 s마다 다시 계획하는 조건의 빈 트랙 한 바퀴가 plan25r보다 크게 느림
#   (3) plan 뒷부분 속도가 실제 주행 속도보다 크게 낮음(plan25m: 0.5 s 지점 -1.59 m/s) = 궤적이 아니라 감속으로 피함.
#   -> 이 중 하나면 출력 구조를 바꾸는 논의로.
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan25h
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --horizon 128 --minibatches 16 --speed-std-floor 0.4 --hold-p 1.0 --hold-len 1,5 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95" > $D/$N.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1111h.done > out/guard4070_1111h.log 2>&1 &
# the control's missing rows first (plan25r u100/u200/u300 held plans, u300 with the controllers), then this run's
setsid nohup bash -c "bash tools/hold_loop.sh plan25r 100 100 300; bash tools/hold_loop.sh $N 50 50 1000" > out/hold_loop_$N.out 2>&1 &
setsid nohup bash -c "bash tools/eval_loop_plan5.sh plan25r 1 300 100 300 256; bash tools/eval_loop_plan5.sh $N 1 100 100 1000 256" > out/eval_loop5_$N.out 2>&1 &
setsid nohup bash tools/analysis_loop9.sh $N 25 25 1000 > out/analysis_loop9_$N.out 2>&1 &
setsid nohup bash tools/video_loop7.sh $N 25 25 1000 > out/video_loop7_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan25s",):
        e["run"] = "watch_plan25h"
        e["label"] = "학습 중 · 처음부터: 모든 plan을 0.1~0.5초 그대로 달리며 학습 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-11 저녁 처음부터 · 학습 때 plan 하나를 0.1~0.5초 그대로 달림(plan 뒷부분까지 실제로 실행) · 뒷부분 예측 학습은 끔 · "
                     "그 밖은 plan25r 그대로(기억 128, plan 타이어 한계 95 %, plan 충돌 감점 끔) · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신 · 초반 checkpoint는 아직 못 달림")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작" >> $D/switch_$N.log
