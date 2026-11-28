#!/bin/bash
cd ~ && source ~/f1tenth_ws/activate.sh > /dev/null 2>&1
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:$PYTHONPATH
OUT=/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/D/bagmap
R=/home/shchon11/F1tenth_E2E/real_data
for b in 01_competition_0826-0827_map08x23/20260827-115713_map08x23_v9.2_161s_fastest-longest 01_competition_0826-0827_map08x23/20260826-192819_map08x23_v8.8_204s_longest+10s-standstill 02_pre-competition/20260725-015428_map12x16_v4.5_79s_GYRO-BAD_96pct-moving; do
  n=$(basename $b | cut -c1-15)
  echo "=== $n"; /usr/bin/time -f "%e s %M KB" timeout 1200 python3 -m f1sim.bag_map $R/$b --name m$n --out $OUT --compare 2>&1 | grep -v "^\[INFO\]" | tail -40
done
echo ALLDONE
