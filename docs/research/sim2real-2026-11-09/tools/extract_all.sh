#!/bin/bash
# 실측 bag 22개를 차례로 npz로 (bag_extract.py). 끝나면 /data/sim2real_1109/extract.done
cd ~ && source ~/f1tenth_ws/activate.sh > /dev/null 2>&1
T=/home/shchon11/F1tenth_E2E/docs/research/sim2real-2026-11-09/tools
for b in /home/shchon11/F1tenth_E2E/real_data/0*/*/; do
  n=$(basename $b)
  [ -e /data/sim2real_1109/$n.npz ] && continue
  timeout 900 python3 $T/bag_extract.py $b /data/sim2real_1109 2>&1 | grep -v "INFO\|WARN" | cut -c1-600
done
echo done > /data/sim2real_1109/extract.done
