#!/bin/bash
# plan25x (11-12 새벽, 사용자: "분석 및 개선, 연구, 구현, 세팅, 학습 등등 쭉 하고 학습을 근거있개 돌리라고" / "Edge case 관리. 기억 활용
# 등등 ... 하나하나 잊지마" / "트랙 변형도 스케일 자체가 변하는것도좀 해줘"). 처음부터.
# 질문: 세게 닿았던 경주를 더 자주 다시 내면(1) 세게 닿음이 plan25r처럼 평평하게 머물지 않고 내려가는가, (2) 같은 상자를 다시 지날 때
#   다르게 달리게 되는가(= 기억을 쓰게 되는가: 기억을 지우면 나빠져야 함).
# 근거(plan25r u300, out/level_repeat_plan25r_u300.txt): 같은 경주를 탐색 잡음만 바꿔 다시 달리면 세게 닿은 경주의 62.7 %가 또 세게
#   닿음(안 닿았던 경주는 12.0 %); 점수 위 10 % 경주가 둘째 주행의 세게 닿음 34-37 %, 위 20 %가 55-59 %를 가짐. 세게 닿음은 update당 결정의
#   0.26 %뿐이라(상자 34건) 학습 신호가 작고, u50-u275 동안 0.5-0.9/분에서 안 내려갔음. 지난 바퀴에 세게 닿은 상자를 다음 바퀴에 또
#   세게 닿는 비율 40.8 % vs 깨끗이 지난 뒤 2.8 %(기억이 벌 수 있는 몫인데 그런 통과는 update당 약 20건).
# 시험하는 변화 하나: --level-replay 0.5 --level-replay-from 50 (경주 자리마다 지난 24경주를 점수 = 학습 차의 세게 닿음 + 서 있던 시간/2 s로
#   기억, 통째로 새로 시작하는 경주의 반을 점수 순위대로 다시 냄; 처음 50 update는 서 있는 차가 대부분이라 끔: plan25r의 선 비율 u1 67 % -> u50 3.5 %).
#   문헌: Prioritized Level Replay(2010.03934)·Robust PLR(2110.02439), GT Sophy의 상황별 표.
# 같이 들어간 환경 고침/요청(실험 대상이 아님; plan25r와의 같은-update 비교는 이것들 때문에 섞임 -> 판정은 아래 run 안의 숫자로):
#   - 트랙 크기 변형 16개(--track-scales korea25: 적용 크기 0.58-1.19, 원래 + 변형 16개는 그대로 = 트랙 33개; 사용자 요청)
#   - 서 버린 상대 차 다시 출발(--opp-restart-s 2: 상자 경주에서 상대가 박혀 있던 시간 24 % -> 0.8 %)
#   - LiDAR 자리를 실제 차와 같게(--lidar-mount-x 0.12555: 차 앞 끝이 아니라 CoG + 0.126 m)
#   - 변형 트랙의 스캔에 안 보이던 벽 고침(track.with_warp)
#   그 밖은 plan25r 그대로(시드 1, 학습률 3e-4 -> 1e-4 / 1000 update, 기억 128 앞쪽, 롤아웃 128, 타이어 한계 95 %, plan 충돌 감점 끔,
#   plan 뒷부분은 다음 plan 예측으로 학습).
# 판정(다시 내기 없는 고정 시드 평가로; 학습 로그의 접촉률은 다시 낸 경주 때문에 부풀어 있음):
#   반증 1: u300까지 고정 시드 평가의 세게 닿음/분이 plan25r의 0.5-0.9 띠 안에 그대로.
#   반증 2: 지난 바퀴에 세게 닿은 상자를 다음 바퀴에 또 세게 닿는 비율이 plan25r(40.8 %)에서 안 내려감 / 기억을 바퀴마다 지워도
#           상자 트랙 바퀴 시간·접촉이 안 나빠짐(= 여전히 기억을 안 씀).
#   반증 3: 라인을 0.8 m 넘게 바꿔야 하는 상자에서 세게 닿는 비율이 plan25r u175의 14.6 %에서 안 내려감.
#   반증 4(부작용): 원래 트랙 빈 바퀴가 plan25r 같은 update보다 0.1 s 넘게 느림(다시 낸 경주가 소심하게 만듦).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan25x
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --level-replay 0.5 --level-replay-from 50 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1112x.done > out/guard4070_1112x.log 2>&1 &
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
    if e["run"] in ("watch_plan25h",):
        e["run"] = "watch_plan25x"
        e["label"] = "학습 중 · 처음부터: 세게 닿았던 경주 다시 내기 + 트랙 크기 변형 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-12 새벽 처음부터 · 세게 닿았던 경주를 반의 확률로 다시 냄(u50부터) · 트랙 크기 변형 16개(x0.58~1.19) 추가 · "
                     "서 버린 상대 차 다시 출발 · LiDAR 자리 = 실제 차(CoG + 0.126 m) · 그 밖은 plan25r 그대로 · "
                     "콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신 · 초반 checkpoint는 아직 못 달림")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작" >> $D/switch_$N.log
