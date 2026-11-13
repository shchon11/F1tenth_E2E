#!/bin/bash
# plan26c (11-13 15:20). 처음부터. plan26b(닿는 값 2배 + 넓은 서 있는 벌점)의 부작용을 고친 판.
# plan26b u50(15:03): 빈 트랙 9.4 s로 빨라졌지만 빈 트랙에서도 벽에 세게 0.92 /분(같은 바퀴 시간의 plan25y u75 0.04, plan26a u50 0.00),
#   plan이 타이어 한계 80 %를 넘는 결정 35 %(plan26a u50 8 %, plan25y u50 4 %), 세게 닿음 합 2.73 /분(plan25y u75 1.33) -> 판정의 부작용 기준.
#   plan26a와 다른 것은 서 있는 벌점뿐 -> 벌점이 원인. 그 벌점은 최근 2 s 동안 초당 1 m보다 느리면 붙어(출발·상자 사이를 천천히 비켜 가기·
#   박은 뒤 다시 출발까지) 학습 초반 '느린 상태'의 값을 크게 깎았고, policy가 타이어 한계까지 몰아붙이게 된 것으로 봄.
# 질문: 서 있는 벌점을 '정말 서 있는 것'(최근 3 s 동안 0.6 m도 못 감)에만 붙이면, 박은 뒤 서 버리는 것은 줄고(벌점의 목적) 정상 주행은
#       plan26a처럼 그대로인가(빈 트랙 벽 닿음 0, 타이어 한계 결정 plan26a 수준).
# 바꾼 것(plan26b 대비): --stall-vmin 0.2 --stall-window 3.0 (그 밖은 같음: 닿는 값 2배, --stall-s 1.0, plan25y 설정).
# 판정:
#   반증 1(부작용 확인): u50에서 빈 트랙 벽에 세게 > 0.1 /분, 또는 타이어 한계 80 % 넘는 결정 > 15 % -> 좁혀도 같은 부작용.
#   반증 2(목적): 시험(exam1 uniform)에서 세게 닿은 뒤 5 s 안에 다시 못 달린 비율이 plan25y(u50 43-44 %)의 절반 밑으로 안 내려감.
#   반증 3: u125·u150에서 학습 조건 평가의 상자 세게 닿음이 plan25y보다 40 % 넘게 줄지 않음(plan25y u125 0.73 / 1.25, u150 0.63 / 0.83).
#   반증 4: 빈 트랙 한 바퀴가 plan25y보다 0.1 s 넘게 느림(u125 8.30 s, u150 8.10 s).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
N=plan26c
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95
export F1SIM_LIDAR_MOUNT_X=0.12555
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25 --contact-scale 2.0 --stall-s 1.0 --stall-vmin 0.2 --stall-window 3.0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1113c.done > out/guard4070_1113c.log 2>&1 &
setsid nohup bash tools/eval_loop_plan5.sh $N 1 100 100 1000 256 > out/eval_loop5_$N.out 2>&1 &
setsid nohup bash tools/analysis_loop10.sh $N 25 25 1000 > out/analysis_loop10_$N.out 2>&1 &
setsid nohup bash tools/video_loop7.sh $N 25 25 1000 > out/video_loop7_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + ".bak_" + time.strftime("%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26b",):
        e["run"] = "watch_plan26c"
        e["label"] = "학습 중 · 처음부터: 세게 닿는 값 2배 + 정말 서 있을 때만 벌점 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-13 오후 처음부터 · 모든 닿음의 시간 벌점 2배(세게 3 m/s면 7 s) + 최근 3 s 동안 0.6 m도 못 가면 서 있는 벌점 · 그 밖은 plan25y와 같음(경로 하나, 기억 가치 출력, "
                     "트랙 33개, 상대 차 다시 출발, LiDAR 자리 = 실제 차) · 콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · "
                     "25 update마다 갱신 · 초반 checkpoint는 아직 못 달림")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작" >> $D/switch_$N.log
