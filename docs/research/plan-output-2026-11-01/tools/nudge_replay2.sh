#!/bin/bash
# nudge_replay.sh의 이어서 (11-13 05:15): 64묶음 x 7대 복제 주행은 5060 GPU 메모리를 5.2 GB 써서 자리 하나로 돌리면 다른 작업이 메모리 부족으로
# 죽음(plan25z u75 학습 영상) -> 남은 시드는 5060을 혼자 쓰는 판(slot5060_all.sh). 먼저 죽은 u75 학습 영상을 다시 만듦.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
cd /home/shchon11/F1tenth_E2E/f1sim || exit 1
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
until grep -q "^rc " $R/out/replay/nudge_plan25y_u150_s20261301.log; do sleep 20; done
V=$R/out/videos/plan25z_u75_train_4cases
if [ ! -e $V.done ]; then
  CUDA_VISIBLE_DEVICES=1 bash $R/tools/slot5060.sh timeout 2400 python3 -u $R/tools/train_video.py /home/shchon11/f1sim_runs/plan25z/plan_u75.pt $V.mp4 40 256 > $V.log 2>&1
  echo done > $V.done
fi
unset F1SIM_PROP_BANK F1SIM_PROP_COUNT F1SIM_PLAN_LAT_MARGIN F1SIM_LIDAR_MOUNT_X
run=plan25y; u=150
for seed in 20261302 20261303; do
  out=$R/out/replay/nudge_${run}_u${u}_s${seed}
  [ -e $out.npz ] && continue
  F1SIM_EXAM_SEED=$seed BOX_REPLAY_ARMS="2,0,0,2,2" BOX_REPLAY_NUDGE="2.0,3" CUDA_VISIBLE_DEVICES=1 bash $R/tools/slot5060_all.sh \
    timeout 3600 python3 -u -W ignore $R/tools/box_replay.py /home/shchon11/f1sim_runs/$run/plan_u$u.pt $out.npz 64 90 3.0 uniform > $out.log 2>&1
  echo "rc $?" >> $out.log
done
python3 -W ignore $R/tools/box_replay_table.py $R/out/replay/nudge_${run}_u${u} $R/out/replay/nudge_${run}_u${u}_s*.npz > $R/out/replay/nudge_${run}_u${u}.txt 2>&1
echo done > $R/out/replay/nudge_${run}_u${u}.done
