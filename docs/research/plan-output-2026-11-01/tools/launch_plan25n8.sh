#!/bin/bash
# plan25n8 (11-10 21:10, the user: "ㅇㅋ"): plan25n7 u100에서 이어, 지그재그 비용 끔(학습 조건에서 plan은 이미 매끄러워 줄일 게 없었음),
# plan 타이어 한계 95 % (--plan-lat-margin 0.95). 근거: u300 학습 조건에서 plan이 한계(조향 표 최댓값 = 조향 끝)의 95 % 넘는 결정 29 %,
# 정상 상태로 22 deg(끝 24) 넘는 조향이 필요한 결정 17 % -- 표 꼭대기가 평평해서(5 m/s에서 마지막 5 %에 18 -> 24 deg) 물리 차에는 고칠
# 조향이 안 남아 콘솔에서 흔들림. 학습 없이 한계만 95 %: 22 deg 넘는 결정 1 %, 콘솔 지금 제어기 plan 떨림 1.84 -> 1.22 deg, 차 128대 세게
# 0.60 -> 0.54. 판정: 학습 조건 바퀴 시간이 한계 100 % 때(빈 6.70) 가까이 돌아오는지, 콘솔 흔들림(지금 제어기)이 줄어드는지.
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
for p in $(pgrep -f "plan_[r]l --name plan25n7 "); do kill $p; done
echo done > $D/guard4070_1110n7.done
echo done > $D/one_plan25n7.done                 # plan25n7's loops finish what they have and stop
sleep 10
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name plan25n8 --format v2 --updates 500 --seed 1 --save-every 25 --init /home/shchon11/f1sim_runs/plan25n7/plan_u100.pt --init-critic --lr 1e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --horizon 128 --minibatches 16 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --mem-aux 0.3 --box-range 12 --plan-lat-margin 0.95" > $D/plan25n8.log 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name plan25n8 " out/guard4070_1110n8.done > out/guard4070_1110n8.log 2>&1 &
# the start under this run's limit: the init checkpoint, its own eval as u0 (plan_table takes it as the start row)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=1 bash ../docs/research/plan-output-2026-11-01/tools/slot5060.sh timeout 2400 python3 ../docs/research/plan-output-2026-11-01/tools/plan_validate.py $R/plan25n7/plan_u100.pt ideal 1024 40 '  u0 ' 2>&1 | grep '^  u0' >> $D/one_plan25n8_i.log" > out/start_eval_plan25n8.out 2>&1 &
setsid nohup bash tools/eval_loop_plan5.sh plan25n8 1 100 100 500 256 > out/eval_loop5_plan25n8.out 2>&1 &
setsid nohup bash tools/analysis_loop7.sh plan25n8 25 25 500 > out/analysis_loop7_plan25n8.out 2>&1 &
setsid nohup bash tools/video_loop6.sh plan25n8 25 25 500 > out/video_loop6_plan25n8.out 2>&1 &
setsid nohup bash tools/blind_loop.sh plan25n8 25 25 500 > out/blind_loop_plan25n8.out 2>&1 &
setsid nohup bash tools/wobble_loop.sh plan25n8 25 25 500 > out/wobble_loop_plan25n8.out 2>&1 &
mkdir -p $R/watch_plan25n8
setsid nohup bash tools/watch_latest_plan.sh watch_plan25n8 plan25n8 $D/one_plan25n8.done > out/watch_plan25n8.log 2>&1 &
python3 - <<'PY'
import json
p = "/home/shchon11/f1sim_runs/console_picks.json"
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan25n7",):
        e["run"] = "watch_plan25n8"
        e["label"] = "학습 중 · planner v2 + 기억(GRU, 2바퀴 구간 + 앞 상자 자리 맞히기) + plan 뒷부분 예측 + 상자 치우기 · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("plan25n7 u100에서 이어 · plan 타이어 한계 95 %(조향 끝까지 쓰던 한계에서 여유: 물리 차 제어기가 고칠 조향을 남김) · 지그재그 비용 끔 · "
                     "기억이 트랙을 따라 앞 12 m 상자 자리(안 보이는 것 포함)를 맞히도록 같이 학습 · 기억은 12.8 s(약 2바퀴) 구간으로 학습 · "
                     "plan 충돌 감점 끔(실제 접촉 비용은 그대로) · 콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) plan25n8 시작" >> $D/switch_plan25n8.log
