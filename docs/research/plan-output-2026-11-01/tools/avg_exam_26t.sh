#!/bin/bash
# (11-15 18:37) plan26t u1050·u1100·u1150 가중치 평균(avg26t/plan_u1100.pt) 시험, plan26p u250 기준으로 지금 추천 평균(u850·900·950)과 비교
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam; X=$R/out/edge1114
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
until [ -e $R/out/analysis/plan26t_u1150_exam.done ]; do sleep 30; done
cd /home/shchon11/F1tenth_E2E/f1sim
python3 - <<'PY'
import torch, copy
R = "/home/shchon11/f1sim_runs/plan26t"
cks = [torch.load(f"{R}/plan_u{u}.pt", map_location="cpu", weights_only=False) for u in (1050, 1100, 1150)]
out = copy.deepcopy(cks[1])
for part in ("actor", "critic"):
    for k, v in out[part].items():
        if torch.is_floating_point(v):
            out[part][k] = sum(c[part][k].float() for c in cks).div(len(cks)).to(v.dtype)
out["averaged_from"] = ["plan26t/plan_u1050.pt", "plan26t/plan_u1100.pt", "plan26t/plan_u1150.pt"]
torch.save(out, "/home/shchon11/f1sim_runs/avg26t/plan_u1100.pt"); print("평균 저장")
PY
N=$E/exam1_avg26t_u1100_ideal_uniform_n256.npz
[ -e $N ] || CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 2400 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/avg26t/plan_u1100.pt $N ideal 256 uniform > ${N%.npz}.log 2>&1
cd $T
python3 -W ignore grad_step_table.py $E/exam1_plan26p_u250_ideal_uniform_n256.npz "평균 u850·900·950=$E/exam1_avg26s_u900_ideal_uniform_n256.npz" "plan26t u1150=$E/exam1_plan26t_u1150_ideal_uniform_n256.npz" "평균(u1050·1100·1150)=$N" 2>&1 | tail -6
for r in plan26t_u1150 avg26t_u1100; do
  EXAM_ROWS=$X/boxrows_$r.npy python3 -W ignore exam_boxes.py $E/exam1_${r}_ideal_uniform_n256.npz > $X/boxes_$r.txt 2>&1
done
python3 - <<'PY'
import numpy as np
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
for nm in ("plan26t_u1050", "plan26t_u1100", "plan26t_u1150", "avg26t_u1100"):
    r = np.load(f"/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/edge1114/boxrows_{nm}.npy"); on = r[:, IGAP] < 0.40
    fast = on & (r[:, IVIN] >= 6.5) & (r[:, ISEE] < 1.0)
    k, n = int((r[fast, IHARD] > 0).sum()), int(fast.sum()); k2, n2 = int((r[on, IHARD] > 0).sum()), int(on.sum())
    print(f"{nm}: 빠르고 늦게 보인 칸 {100 * k / n:.1f} % ({k}/{n}) · 상자 통과당 세게 {100 * k2 / n2:.1f} % ({k2}/{n2})")
PY
