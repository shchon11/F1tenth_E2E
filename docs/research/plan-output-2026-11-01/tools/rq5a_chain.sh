#!/bin/bash
# (11-27) RQ5 arm on T-maze v3: asymmetric recurrent SAC (branch rq5-sac, worktree F1tenth_E2E_sac), forked from scan31b2 u600,
# the parent's reward (shaping off), EV-only gate (the slope gate measures the lateral direction, but the nudge test found the
# gain at L in SPEED -- u600: a 2.5 m/s cap 1 s before sight, L - none = -0.075 +- 0.032 s per pass), then the tmaze exam of the
# final checkpoint + tmaze_measure (compare: tm39a u150 / tm39a2 = PPO on the same world).
W=/home/shchon11/F1tenth_E2E_sac/f1sim/f1sim/learn/configs/plan_world_korea26scan_tmaze.json
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python; OUT=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/rq
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
while true; do av=$(free -g | awk '/Mem:/{print $7}'); n2=$(ps -eo args | grep -c "[e]xam_record.py"); [ "$av" -ge 6 ] && [ "$n2" -eq 0 ] && break; sleep 60; done
echo "$(date +%T) rq5a start" >> $OUT/chain.log
cd /home/shchon11/F1tenth_E2E_sac/f1sim && CUDA_VISIBLE_DEVICES=0 $PY -u -m f1sim.learn.plan_rl --name rq5a --world "$W" --envs 416 --updates 1000 --init /home/shchon11/f1sim_runs/scan31b2/plan_u600.pt --init-critic --scene-replay 0 --shaping-fade 0,0 --horizon 64 --gamma 0.995 --lam 0.95 --epochs 4 --minibatches 8 --accum 1 --accum-actor 1 --accum-actor-auto --auto-hold 0 --accum-ratchet --horizon-auto-mult 2.0 --horizon-auto 4 --horizon-auto-min 2 --horizon-auto-down --mem-chunk-rollout --mem-prev 80 --noise-probe-every 4 --opp-every 4 --accum-auto-frac 0.5 --clip 0.2 --lr 0.0003 --lr-end 0.0001 --std0 0.6 --vf-coef 0.5 --max-grad 0.5 --format v2 --vest-coef 1.0 --noise-corr 0.9 --noise-corr-steer --noise-corr-box 1.5,4.0 --online-late 4.0 --online-boxes 0.5 --stuck-noise 0.0 --noise-corr-car 2.0,4.0 --noise-corr-free 0.0 --lr-warmup 0 --actor-freeze 0 --plan-hit-share 0.0 --stall-s 0.0 --stall-vmin 1.0 --stall-window 2.0 --contact-scale 0.82 --plan-hit-s 1.0 --hold-p 0.0 --hold-len 2,4 --hold-clear 0.0 --tail-consistency 1.0 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --memory 128 --mem-chunk 128 --mem-front --plan-lat-margin 0.95 --box-range 12.0 --mem-value 0.15 --speed-std-floor 0.4 --steer-std-floor 0.3 --removal-p 0.0 --opp-restart-s 2.0 --lidar-mount-x 0.12555 --lidar-tilt 0.0 --align-episodes --compile-leaves --branches 1 --live-tap 0 --live-taps 7 --seed 1 --device cuda --algo sac --sac-gate ev > $OUT/rq5a.log 2>&1
echo "$(date +%T) rq5a exit $?" >> $OUT/chain.log
CK=$(ls -t /home/shchon11/f1sim_runs/rq5a/*.pt 2>/dev/null | grep -v sac_gate | head -1)
[ -n "$CK" ] || { echo "$(date +%T) rq5a: no checkpoint" >> $OUT/chain.log; exit 1; }
cd $T && CUDA_VISIBLE_DEVICES=0 F1SIM_EXAM_WORLD=/home/shchon11/F1tenth_E2E/f1sim/f1sim/learn/configs/plan_world_korea26scan_tmaze.json F1SIM_SRC=/home/shchon11/F1tenth_E2E_sac/f1sim $PY exam_record.py $CK $OUT/exam_tmaze_rq5a_n384.npz ideal 384 tmaze 32 999 > $OUT/exam_rq5a.log 2>&1
$PY $T/tmaze_measure.py $OUT/exam_tmaze_rq5a_n384.npz > $OUT/measure_rq5a.txt 2>&1
echo "$(date +%T) rq5a exam done ($CK)" >> $OUT/chain.log
