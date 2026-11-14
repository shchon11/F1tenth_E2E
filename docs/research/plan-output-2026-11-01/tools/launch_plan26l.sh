#!/bin/bash
# plan26l (11-14 18:40): plan26k와 같은 질문·판정(아래 plan26k 머리말), 학습 속도만 다름 -- plan26k(u12)를 멈추고 처음부터.
#   --compile-leaves: 물리 핵심 계산(타이어·차체, IMU)을 합쳐서(콘솔과 같은 방식) control step당 GPU 작업 약 24,000개 -> 훨씬 적게.
#   graph_runtime: LiDAR 상자 합치기 graph가 첫 재시작의 다른 크기 입력 한 번에 영원히 꺼지던 것 고침(그 호출만 graph 없이).
#   4070 차 608대, 3 update: update 45.0 -> 39.0 s(rollout 37.9 -> 32.0 s); 합친 물리 테스트 3개 통과, 고침 전후 첫 update 결과 같음.
# (아래는 plan26k 머리말)
# plan26k (11-14 18:05, 사용자: "그려 크기변형은 빼자"). 처음부터. plan26j와 같고 트랙 크기 변형(60-120 %, 16개)만 뺌 -> 원래 지도 + 벽 변형 16개(모두 실제 크기).
# 질문: 실제 크기 트랙만으로 배우면 plan26i가 u150 -> u225에 학습 세계에서 얻은 빠르기를, 실제 지도의 세게 닿음 증가 없이 얻는가.
# 근거(11-14 17:40-18:00 분석, out/drift/, out/rwexam/, tools/drift_scenes.py, seg_time.py, rw_sum.py):
#   - plan26i u150 -> u300 frozen exam(실제 지도) 세게 닿음 0.55 -> 2.23/세션. 학습 보상으로 계산하면 u300이 u150보다 -5.1 +- 1.4 s/90 s
#     (진행 +2.4, 접촉 값 -7.6); 학습 잡음으로 돌려도 -1.8 +- 1.3 s, 세게 닿음은 잡음과 상관없음(2.23 vs 2.39) -> 잡음 가설 반증.
#   - 학습 rollout(트랙 33개)에서는 u150 -> u225 진행 +8.5 s/90 s, 접촉 값 그대로(-5.3 -> -5.6 s) -> 학습이 보는 세계에서는 접촉이 늘지 않고 빨라짐;
#     접촉 값을 2배로 해도 이 기간의 학습 신호는 같음 -> ① 유지.
#   - 실제 지도 장면: 벽 세게 닿음은 거의 모두 한 오른쪽 코너, u300이 8-9 m/s로 바깥에서 들어와 바깥 벽(u150 6-7 m/s 안쪽); 닿은 그 통과는
#     같은 구간 u150보다 +0.54 +- 0.06 s 느림(72/79). 상자는 같은 길을 1 m/s 더 빠르게 지나다 모서리를 스침. 대부분 이미 지나 본 자리.
#   - 그래서 33개 평균이 좋아지는 동안 실제 크기 지도에서는 너무 빠르게 들어가게 됨(크기 변형이 학습의 절반) -> 크기 변형을 뺌.
# 판정(frozen exam1 ideal/uniform/256, plan26i 같은 update와 세션끼리 짝지어):
#   반증 1: u200 세게 닿음/세션이 plan26i u200(0.94)보다 뚜렷이 적지 않음.
#   반증 2: u150-u250 빈 트랙 혼자 거리가 plan26i 같은 update보다 10 m 넘게 짧음(실제 지도 빠르기를 잃음).
#   반증 3: u200 뒤 학습 보상이 떨어지는 구간(KL 0.006 넘음)이 생김(합친 수 고정이 안 먹힘).
# (아래는 plan26j 머리말)
# plan26j (11-14 17:15, 사용자: "1, 2 ㄱㄱ 3 ㄴㄴ 분석을 좀 심도깊게 하고 정해" -- 세게 닿음 값 ① 그대로 / ② 2배 중 분석해서 정하라). 처음부터.
# 질문: plan26i가 u150 뒤로 나빠진 것이 세게 닿음 값 때문인가, 학습 걸음 규칙의 되먹임 때문인가 -> 되먹임을 막으면 u150의 좋은 상태가 이어지는가.
# 분석(plan26i progress, out/netaudit/plan26i_drift_cause.png):
#   - u180-235 합친 수 16, KL 0.003, 학습 보상 최고(0.0145-0.0151), 학습 중 접촉은 오히려 줄거나 그대로(상자 1.17 -> 0.9/분).
#   - u236에 잡음 크기가 190k -> 56k로 재지고(접촉이 늘기 시작한 때: 고칠 신호가 커지면 |G|가 커져 잡음 크기가 작게 재짐) 합친 수 16 -> 4,
#     u270-280에 15k로 -> 1. update당 actor 걸음 4~16배 -> KL 0.003 -> 0.010-0.028, 잘림 0.03 -> 0.15-0.30, critic 오차 2배,
#     학습 중 접촉 3배, 학습 보상 0.0151 -> 0.0129 (policy가 '닿는 게 이득'이라 고른 것이 아니라 학습이 흔들림).
#   - 그래서 세게 닿음 값은 그대로(①). ②는 이 되먹임을 막지 못하고 빠르기만 깎을 위험.
#   - 남는 것 하나: u150-225 안정 구간에도 frozen exam(평균 행동) 세게 닿음 0.48 -> 0.94인데 학습 rollout(탐색 잡음 포함) 접촉은 그대로 ->
#     평균 행동과 잡음 섞인 행동의 차이. 되먹임을 막은 뒤에도 이것이 크면 그때 ②를 다시 봄.
# 바꾼 것: --accum-ratchet (합친 minibatch 수는 내려가지 않음). 코드는 속도 개선 2차 포함(결과 같음, update 약 45 s).
# 판정(frozen exam1 ideal/uniform/256, plan26i 같은 update와 세션끼리 짝지어):
#   반증 1: u250·u300 세게 닿음/세션이 plan26i(1.52 / 2.29)보다 뚜렷이 적지 않음.
#   반증 2: u200 이후 KL이 0.006을 넘는 구간이 생김(되먹임이 다른 길로 다시 생김).
#   반증 3: u150까지 plan26i보다 뚜렷이 나쁨(합칠 수 규칙은 내려갈 때만 다르므로 u150까지는 같은 성질이어야 함; 빈 트랙 -10 m 또는 세게 닿음 +0.3).
# (아래는 plan26i 머리말)
# plan26i (11-14 10:10, 사용자: "병목 계속 찾아봐서 개선해. 아직도 개선할게 많을거야"). 처음부터. plan26h(u7에서 멈춤)와 같은 학습에 속도 개선만:
#   - 상대 차 planner를 0.1 s마다(opp-every 4; 전에는 0.025 s마다). frozen exam(plan26f u150)에서 상대 차 속도·선 시간·다시 출발과
#     우리 결과가 간격 1과 같음(빈 트랙 -0.06 +- 0.44 m, 세게 닿음 -0.09 +- 0.11).
#   - 결과가 그대로인 고침 셋(차 1024대 1 update의 rollout 기록이 이전과 똑같음): ideal 실행 차의 MAP 계산 건너뛰기, 상자 접촉 계산을
#     CUDA graph로, 재시작 걸음에서 state/odom 텐서 이름이 갈라져 상대 차 planner 5개가 매번 느린 경로로 돌던 것.
#   - 잡음 크기 측정을 4 update마다(noise-probe-every 4).
#   - 잰 시간: 한 GPU 1024대 125 s -> 75 s/update; 두 GPU 640+384대 93 s -> 약 56 s/update. 5060이 조금 빨라 608 + 416대로 나눔.
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
N=plan26l
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95                # every tool the loops run builds its plan limits with it
export F1SIM_LIDAR_MOUNT_X=0.12555               # ... and its scanner where this run's is (plan_policy.load_actor2 checks)
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=1 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 416 --dist-world 2 --dist-init tcp://127.0.0.1:29566 --dist-rank 1 > $D/${N}_rank1.log 2>&1" > /dev/null 2>&1 &
setsid nohup bash -c "cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 1000 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef $VEST --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --envs 608 --dist-world 2 --dist-init tcp://127.0.0.1:29566 --dist-rank 0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
sleep 5
setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_1114l.done > out/guard4070_1114l.log 2>&1 &
CORE_VAL_ENVS=512 setsid nohup bash tools/core_loop.sh $N 25 25 1000 > out/core_loop_$N.out 2>&1 &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 &
python3 - <<'PY'
import json, shutil, time
p = "/home/shchon11/f1sim_runs/console_picks.json"
shutil.copy(p, p + time.strftime(".bak_%m%d_%H%M"))
d = json.load(open(p)); L = d["picks"]
for e in L:
    if e["run"] in ("watch_plan26k",):
        e["run"] = "watch_plan26l"
        e["label"] = "학습 중 · 처음부터: plan 뒷부분 = 실제로 간 길(몸통까지) + 두 GPU · Korea 2025 상자 obstacle · 최신 checkpoint"
        e["note"] = ("11-14 18:40 처음부터 · 크기 변형 없이(실제 크기 트랙 17개) · 합친 수가 내려가지 않게 · plan 0.2-1.0 s를 차가 실제로 간 위치·속도로 학습(몸통까지, policy의 절반 힘) · 4070+5060 두 GPU · "
                     "그 밖은 plan26f 그대로 · 콘솔은 0.1 s마다 새 plan(물리 + 새 제어기, F1SIM_PLAN_MODE=map이면 MAP) · 25 update마다 갱신")
json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date +%T) $N 시작 (두 GPU 608+416대, 상대 차 0.1 s마다, 뒷부분 = 실제로 간 길)" >> $D/switch_$N.log
