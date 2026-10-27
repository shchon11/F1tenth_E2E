#!/bin/bash
# 16 updates of rl_clean.sh at each actor lr: approx_kl / clipfrac / actor move per update (the step size, not a leash)
D=$(cd $(dirname $0) && pwd)
for LR in ${LRS:-5e-5 1e-4}; do
  GPU=${GPU:-0} LR=$LR LR_END=$LR TOTAL=$((16*16384)) CW=2 SAVE_EVERY=1000 bash $D/rl_clean.sh calib_lr$LR 7 > /dev/null 2>&1
  echo "lr $LR done $(date +%H:%M)"
done
