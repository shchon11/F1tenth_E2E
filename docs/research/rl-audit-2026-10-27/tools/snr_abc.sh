#!/bin/bash
# fresh SNR measurements: (a) the current recipe's update (256 envs, T32, 2048 samples), (b) big batch (1024 envs, T64, 16384),
# (c) big batch + AR(1) speed noise rho 0.95. Actor frozen (lr 0). GPU from $G.
SP=$(cd $(dirname $0) && pwd) G=${G:-1}
NAME=snr_a_s7 GPU=$G ENVS=256 TOTAL=$((46*2048)) bash $SP/run_snr.sh --lr 0 --lr-end 0 --critic-warmup 1 > $SP/snr_a.log 2>&1
echo "a done $(date +%H:%M)"
NAME=snr_b_s7 GPU=$G ENVS=1024 TOTAL=$((34*16384)) bash $SP/run_snr.sh --lr 0 --lr-end 0 --critic-warmup 1 --horizon 64 --minibatch 4096 > $SP/snr_b.log 2>&1
echo "b done $(date +%H:%M)"
NAME=snr_c_s7 GPU=$G ENVS=1024 TOTAL=$((34*16384)) bash $SP/run_snr.sh --lr 0 --lr-end 0 --critic-warmup 1 --horizon 64 --minibatch 4096 --speed-noise-rho 0.95 > $SP/snr_c.log 2>&1
echo "c done $(date +%H:%M)"
