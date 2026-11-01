#!/bin/bash
# 4070: ×1.2 학생이 끝나면(완료 파일이 학습 시작보다 새것) Korea 2025 -> ×1.2 상자 접촉 대가 이어 학습, 지킴이 하나로
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
until [ -f $D/sst_srp26xh2_props.done ] && [ $D/sst_srp26xh2_props.done -nt $R/sst_srp26xh2_props/args.json ]; do sleep 30; done
bash $T/gpu_guard3.sh "state_[r]l --name srp2[56]x?h2_propst" $D/chain_touch_4070.done > $D/gpu_guard_touch.log 2>&1 &
bash $T/props_touch.sh real:korea_2025_iccas srp25h2 0 91
bash $T/props_touch.sh real:korea_2026_x1p2 srp26xh2 0 92
echo done > $D/chain_touch_4070.done
