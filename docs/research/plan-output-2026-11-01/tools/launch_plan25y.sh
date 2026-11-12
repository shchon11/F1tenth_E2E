#!/bin/bash
# plan25y (11-12 저녁, 사용자: "분석 및 개선, 연구, 구현, 세팅, 학습 등등 쭉 하고 학습을 근거있개 돌리라고" / "지금 본질은 잊지마. Edge case 관리.
# 기억 활용 등등" / 기억 설계 질문에 "ㅇㅋ"). 처음부터.
# 질문: 기억(GRU)이 '이 세션에 상자가 있다'를 붙들게 만들면(가치 출력 + 오래 붙드는 문 초기값), 같은 상자에 되풀이해 세게 닿는 것이 줄어드는가.
# 근거(plan25x·plan25r u300, 같은 시험 128세션 x 90 s; out/replay/t_*.txt, 11-12 저녁 표):
#   - 세게 닿음은 몇 개 상자에 몰려 있음: 한 세션에서 2번 이상 세게 닿은 상자가 세게 닿음의 53-70 %(1~18개), 그런 상자는 통과당 57-63 %가 세게.
#     첫 통과에 세게 닿은 상자는 그 뒤 통과에서도 33-50 %가 또 세게(plan25x 10/21, plan25r 10/31; 그 밖의 상자는 1-2 %). 통과 순서가 늘어도 안 내려감.
#   - 그 상자들: 닿기 약 1 s 전(5.6 m 앞)에야 보이고 6.5 m/s 넘게 다가가는 자리(윗직선 끝 -> 왼쪽 헤어핀 들머리 등). 이 칸의 세게 닿음 16-17.5 %
#     (plan25x 10/57, plan25r 23/143), 같은 때 보여도 5-6.5 m/s면 2.7 %.
#   - 같은 순간을 복사해 다시 달린 실험(box_replay.py): 그 상자에서는 센서 잡음만 다른 복사본도 72 %(21/29) 세게 닿음 = 그 순간의 상태(다가가는
#     속도·라인)가 정함. 기억을 지난 바퀴 같은 자리 것으로 바꿔 넣어도 그대로(5/11 vs 11/32) = 기억에 상자에 대한 것이 없음. 기억을 0으로 한 복사본은
#     0.2-0.3 m/s 느리게·안쪽으로 들어가 안 닿음(0/7) = 필요한 바뀜은 작음.
#   - 왜 기억에 없나(memory-design-question, gru_probe.py): GRU에 닿는 학습 신호는 1-2 s 전까지만. 같은 크기 GRU에 학습의 value와 같은 꼴을 가르치면
#     문 초기값이 기본일 때 '이 세션에 상자가 있다' 구분 0.56-0.66, 붙드는 시간을 1~30 s에 퍼지게 두면 0.86-0.89(policy 학습과 같은 조각 조건).
#     지금 policy의 기억은 0.61.
#   - 학습 세계에는 되풀이가 있음: 한 경주 40 s(약 5바퀴) 동안 같은 상자 배치, 닿아도 이어 달림, 반은 10-30 s에 상자를 치움.
# 시험하는 변화: --mem-value 0.15 (기억 상태만 읽는 value 출력; 주행은 안 읽음, 특징은 끊고 넣음, advantage는 그대로 privileged critic)
#               + --mem-keep-s 30 (GRU update 문이 처음부터 0.2~30 s를 붙들게; 기록 실험에서 통과한 설정).
# 뺀 것: 세게 닿았던 경주 다시 내기(plan25x에서 상자·되풀이 닿음에 효과 없었음: u300 같은 시험 세게 47 -> 36건이지만 통과당·되풀이 비율은 같음).
#   곁가지 읽기: 빈 바퀴가 u300에 6.9-7.0 s로 돌아오면 plan25x의 넓은 오른쪽 헤어핀(+1.9 m, +0.34 s)은 다시 내기 탓, 7.2-7.3 s면 크기 변형·LiDAR 자리 탓.
# 그 밖은 plan25x 그대로(트랙 33개(크기 0.58-1.19), 상대 차 다시 출발 2 s, LiDAR 자리 CoG + 0.126 m, 시드 1, 학습률 3e-4 -> 1e-4 / 1000 update,
#   기억 128 앞쪽, 롤아웃 128, 타이어 한계 95 %, plan 충돌 감점 끔, plan 뒷부분은 다음 plan 예측으로 학습).
# 판정(u300, plan25x·plan25r의 같은 시험과 견줌):
#   반증 1: 기억에서 '이 세션에 상자가 있다'를 직선으로 읽은 구분이 0.8 미만(지금 0.61) -> 가치 출력이 학습 조건에서 기억을 못 만듦.
#   반증 2: 첫 통과에 세게 닿은 상자의 그 뒤 통과가 여전히 33-50 % 세게 / 세게 닿은 다음 통과가 여전히 35-40 % 세게.
#   반증 3: 기억을 바퀴마다 지워도 상자 트랙 바퀴 시간·접촉이 안 나빠짐(= 기억이 있어도 주행이 안 읽음).
#   반증 4: 늦게 보이는(0.8-1.2 s 전) 상자로 6.5 m/s 넘게 다가가는 통과에서, 둘째 통과부터의 다가가는 속도·세게 닿음이 첫 통과와 같음.
#   반증 5(부작용): 상자 없는 세션의 빈 바퀴가 plan25x 같은 update보다 0.1 s 넘게 느림(기억이 '상자 없음'을 못 가려 늘 조심).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan25y
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1112y.done > out/guard4070_1112y.log 2>&1 &
setsid nohup bash tools/eval_loop_plan5.sh $N 1 100 100 1000 256 > out/eval_loop5_$N.out 2>&1 &
setsid nohup bash tools/analysis_loop10.sh $N 25 25 1000 > out/analysis_loop10_$N.out 2>&1 &
setsid nohup bash tools/video_loop7.sh $N 25 25 1000 > out/video_loop7_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan25x",):
        e["run"] = "watch_plan25y"
        e["label"] = "학습 중 · 처음부터: 기억이 '이 세션에 상자가 있다'를 붙들게(가치 출력 + 오래 붙드는 초기값) · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-12 저녁 처음부터 · 기억(GRU) 상태만 읽는 value 출력 0.15 + update 문이 0.2~30 s를 붙드는 초기값 · "
                     "세게 닿았던 경주 다시 내기는 뺌 · 트랙 33개(크기 x0.58~1.19) · 서 버린 상대 차 다시 출발 · LiDAR 자리 = 실제 차 · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신 · 초반 checkpoint는 아직 못 달림")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작" >> $D/switch_$N.log
