#!/bin/bash
# (11-15 15:37) plan26s u850·u900·u950 가중치 평균(avg26s/plan_u900.pt) 만들고 시험, plan26p u250 기준으로 앞 평균·각 checkpoint와 비교
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam; X=$R/out/edge1114
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
until [ -e $R/out/analysis/plan26s_u950_exam.done ]; do sleep 30; done
cd /home/shchon11/F1tenth_E2E/f1sim
python3 - <<'PY'
import torch, copy
R = "/home/shchon11/f1sim_runs/plan26s"
cks = [torch.load(f"{R}/plan_u{u}.pt", map_location="cpu", weights_only=False) for u in (850, 900, 950)]
out = copy.deepcopy(cks[1])
for part in ("actor", "critic"):
    for k, v in out[part].items():
        if torch.is_floating_point(v):
            out[part][k] = sum(c[part][k].float() for c in cks).div(len(cks)).to(v.dtype)
out["averaged_from"] = ["plan26s/plan_u850.pt", "plan26s/plan_u900.pt", "plan26s/plan_u950.pt"]
torch.save(out, "/home/shchon11/f1sim_runs/avg26s/plan_u900.pt"); print("평균 저장")
PY
N=$E/exam1_avg26s_u900_ideal_uniform_n256.npz
[ -e $N ] || CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 2400 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/avg26s/plan_u900.pt $N ideal 256 uniform > ${N%.npz}.log 2>&1
cd $T
python3 -W ignore grad_step_table.py $E/exam1_plan26p_u250_ideal_uniform_n256.npz "평균 u650·700·750=$E/exam1_avg26s_u700_ideal_uniform_n256.npz" "plan26s u850=$E/exam1_plan26s_u850_ideal_uniform_n256.npz" "plan26s u900=$E/exam1_plan26s_u900_ideal_uniform_n256.npz" "plan26s u950=$E/exam1_plan26s_u950_ideal_uniform_n256.npz" "평균(u850·900·950)=$N" 2>&1 | tail -6
for r in plan26s_u950 avg26s_u900; do
  EXAM_ROWS=$X/boxrows_$r.npy python3 -W ignore exam_boxes.py $E/exam1_${r}_ideal_uniform_n256.npz > $X/boxes_$r.txt 2>&1
done
python3 - <<'PY'
import numpy as np
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
for nm in ("avg26s_u700", "plan26s_u850", "plan26s_u900", "plan26s_u950", "avg26s_u900"):
    r = np.load(f"/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/edge1114/boxrows_{nm}.npy"); on = r[:, IGAP] < 0.40
    fast = on & (r[:, IVIN] >= 6.5) & (r[:, ISEE] < 1.0)
    k, n = int((r[fast, IHARD] > 0).sum()), int(fast.sum()); k2, n2 = int((r[on, IHARD] > 0).sum()), int(on.sum())
    print(f"{nm}: 빠르고 늦게 보인 칸 {100 * k / n:.1f} % ({k}/{n}) · 상자 통과당 세게 {100 * k2 / n2:.1f} % ({k2}/{n2})")
PY
