#!/bin/bash
# (11-14 23:00) plan26m 학습 rollout에서 차가 서 있는 비율 68 %(plan26l 같은 때 6 %)의 원인 시험: 이어지는 속도 잡음이 차를 세우나.
# plan26m u25를 같은 세계에서 rollout만(update 없음): 잡음 모두 0.9 / 조향만 0.9 / 흰 잡음. 읽기: 서 있는 비율과 벽 세게 닿음.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
for job in "all --noise-corr 0.9" "steer --noise-corr 0.9 --noise-corr-steer" "white"; do
  set -- $job; tag=$1; shift
  N=plan26m_stop_$tag; OUT=$R/out/batchdump/$N
  [ -e /home/shchon11/f1sim_runs/$N ] && continue
  mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=0 timeout 1200 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 4 --seed 5 --save-every 100000 \
    --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 \
    --compile-leaves --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef 1.0 \
    --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 \
    --lidar-mount-x 0.12555 --envs 384 "$@" \
    --init /home/shchon11/f1sim_runs/plan26m/plan_u25.pt --init-critic --dump-rollout $OUT > $OUT/run.log 2>&1
  echo "rc $?" >> $OUT/run.log
done
python3 - <<'PY'
import glob, numpy as np
for tag in ("all", "steer", "white"):
    fs = sorted(glob.glob(f"/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan26m_stop_{tag}/u*.npz"))[1:]
    if not fs:
        print(tag, "없음"); continue
    st, sp, wall, hit = [], [], [], []
    for f in fs:
        z = np.load(f); v = np.abs(z["pose"][..., 3]); st.append((v < 0.3).mean()); sp.append(v.mean())
        h = z["hard"]; wall.append(h[..., 0].sum() / (h.shape[0] * h.shape[1] * 0.1 / 60)); hit.append(h.sum() / (h.shape[0] * h.shape[1] * 0.1 / 60))
    print(f"{tag:6s}: 서 있는 비율 {100 * np.mean(st):.0f} % · 평균 속도 {np.mean(sp):.2f} m/s · 벽 세게 {np.mean(wall):.2f}/분 · 전체 세게 {np.mean(hit):.2f}/분 ({len(fs)} rollout)")
PY
