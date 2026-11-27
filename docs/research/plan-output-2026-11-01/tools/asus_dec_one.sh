#!/bin/bash
# (11-27, user on /live for an ACE run: "여기선 못봐?") one decoder on THIS PC for a checkpoint that has none (ACE runs are examined
# here without the policy state): an exam3 record with F1SIM_EXAM_SAVE_MEM=1 on CUDA $GPU (0 = the 4070) into out/decoder/asus_rec/,
# then state_decoder_train.py on it -> out/decoder/<record>_dec.npz (+ decin, decode.txt) and ~/f1sim_runs/RUN/plan_uN_dec.npz
# (the console overlay; live_dec.py uses it for RUN and for forks of RUN until they have their own).  usage: asus_dec_one.sh RUN U [GPU]
RUN=$1; U=$2; GPU=${3:-0}
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools; D=$T/../out; PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
W=/home/shchon11/F1tenth_E2E/f1sim/f1sim/learn/configs/plan_world_korea26scan_qual.json
B=exam3_${RUN}_u${U}_ideal_line3_n256; R=$D/decoder/asus_rec/$B
cd /home/shchon11/F1tenth_E2E/f1sim
[ -e $R.npz ] || PYTHONPATH=$PWD F1SIM_EXAM_SAVE_MEM=1 F1SIM_EXAM_WORLD=$W WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 \
  F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555 CUDA_VISIBLE_DEVICES=$GPU timeout 5400 nice $PY -u -W ignore \
  $T/exam_record.py /home/shchon11/f1sim_runs/$RUN/plan_u$U.pt $R.npz ideal 256 line3 > $R.log 2>&1
[ -e ${R}_mem.npz ] || { echo "no mem record"; exit 1; }
CC=/home/shchon11/f1sim_runs/$RUN/plan_u${U}_dec.npz
CUDA_VISIBLE_DEVICES=$GPU nice $PY -u -W ignore $T/state_decoder_train.py $R.npz --out $D/decoder/$B --ckpt-copy=$CC > $D/decoder/$B.declog 2>&1
echo "rc $?" >> $D/decoder/$B.declog
