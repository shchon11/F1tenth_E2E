#!/bin/bash
# plan25n3 (11-09 23:00, 사용자 "ㅇㅋㅇㅋ" -- 기억 고치기 A): plan25n2 u100에서 이어, 기억을 가르치는 구간을 3.2 s -> 12.8 s(약 2바퀴).
# 근거: 기억이 신경망 계산에 끼치는 크기가 175 update째 4.5-4.9 %에서 멈췄고, 지금 보이는 것보다 더 아는 것은 처음부터 +0-2 %p.
# 학습이 기억을 3.2 s 묶음으로만 거슬러 배워서, 한 바퀴(약 7 s) 넘게 들고 있어야 쓸모 있는 정보(상자가 있었나 / 치워졌나 / 코너 뒤 상자)를
# 배울 길이 없음. 바꾸는 것: 한 번에 모으는 주행 --horizon 64 -> 128, --mem-chunk 32 -> 128, --minibatches 8 -> 16(묶음 하나 크기 그대로
# 4096결정). 보상·세계는 그대로라 critic은 이어 받고 actor를 멈추지 않음. 판정(50 update 안에): 기억이 운전(첫 단계 속도 출력)을 바꾸는 크기가
# 0.013에서 확실히 커짐, 시야보다 더 맞힘 +10 %p 이상, 배치 0 U턴 입구 상자에 두 번째 바퀴부터 덜 닿음, 치운 뒤 빈 트랙 바퀴가 빨라짐 -- 안 되면 되돌림
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
until [ -e $R/plan25n2/plan_u100.pt ]; do sleep 10; done
sleep 20
for p in $(pgrep -f "plan_[r]l --name plan25n2 "); do kill $p; done
echo done > $D/guard4070_1109n2.done
echo done > $D/one_plan25n2.done                 # plan25n2's loops finish u100 and stop
sleep 10
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25n3 --format v2 --updates 500 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25n2/plan_u100.pt --init-critic --lr 1e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0" > $D/plan25n3.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25n3 " out/guard4070_1109n3.done > out/guard4070_1109n3.log 2>&1 &
setsid nohup bash tools/eval_loop_plan4.sh plan25n3 1 100 100 500 256 > out/eval_loop4_plan25n3.out 2>&1 &
setsid nohup bash tools/ideal_videos2.sh plan25n3 1 100 100 500 > out/ideal_videos2_plan25n3.out 2>&1 &
setsid nohup bash tools/analysis_loop3.sh plan25n3 25 25 500 plan25m > out/analysis_loop3_plan25n3.out 2>&1 &
setsid nohup bash tools/video_loop4.sh plan25n3 25 25 500 > out/video_loop4_plan25n3.out 2>&1 &
mkdir -p $R/watch_plan25n3
setsid nohup bash tools/watch_latest_plan.sh watch_plan25n3 plan25n3 $D/one_plan25n3.done > out/watch_plan25n3.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] == "watch_plan25n2":
        e["run"] = "watch_plan25n3"
        e["label"] = "학습 중 · planner v2 + 기억(GRU, 2바퀴 구간으로 학습) + plan 뒷부분 예측 + 상자 치우기 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("plan25n2 u100에서 이어 · 기억을 12.8 s(약 2바퀴) 구간으로 학습 · plan 충돌 감점 끔(실제 접촉 비용은 그대로) · 첫 0.1 s만 강화학습, "
                     "뒷부분은 다음 plan을 정답으로 · 콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) plan25n3 시작" >> $D/switch_plan25n3.log
