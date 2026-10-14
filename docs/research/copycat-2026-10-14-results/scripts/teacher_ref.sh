#!/bin/bash
# The PREREG's teacher reference: the WP4 teacher on ICCAS solo, the M1 protocol, through the CLI
# (which now takes --speed-mode wp4).
source /tmp/claude-1000/-home-shchon11-F1tenth-E2E/d98b5699-f98e-4674-9a12-ab5a57bf35f1/scratchpad/v2cc/common.sh
export CUDA_VISIBLE_DEVICES=0
timeout 1800 python3 -m f1sim.learn.evaluate --teacher --action-mode plan --speed-mode wp4 \
  --tracks real:korea_2025_iccas --envs 128 --steps 2400 --protocol trials --budget-laps 3 --speed-cap 8 \
  --seed 123 $TEACH --output $D/teacher_iccas.json > $D/teacher_iccas.log 2>&1
echo "exit $?" >> $D/teacher_iccas.log
