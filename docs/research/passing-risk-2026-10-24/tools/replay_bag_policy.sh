#!/bin/bash
# Play a real bag through the policy node (CPU) and record its nearest-car readings. usage: replay_bag_policy.sh BAG OUT.npz [extra policy -p args]
BAG=$1; OUT=$2; shift 2
cd ~ && source ~/f1tenth_ws/activate.sh >/dev/null 2>&1
CK=/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar.pt
DUR=$(ros2 bag info $BAG 2>/dev/null | awk '/Duration/ {print int($2) + 1}')
timeout $((DUR + 40)) ros2 run f1sim_ros policy --ros-args -p checkpoint:=$CK -p device:=cpu -p speed_cap:=9.0 "$@" > ${OUT%.npz}.policy.log 2>&1 &
PP=$!
sleep 12
timeout $((DUR + 30)) python3 $(dirname $0)/record_perception.py $OUT $((DUR + 20)) > ${OUT%.npz}.rec.log 2>&1 &
RP=$!
sleep 2
timeout $((DUR + 20)) ros2 bag play $BAG --topics /scan /odom /sensors/imu/raw /sensors/imu /sensors/core > ${OUT%.npz}.play.log 2>&1
wait $RP; kill $PP 2>/dev/null; wait $PP 2>/dev/null
cat ${OUT%.npz}.rec.log
