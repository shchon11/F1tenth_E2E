"""Step a checkpoint's actor along a measured policy gradient, no training (11-13): does the direction the rollouts point to
make the car drive better?

`grad_dump.sh` saves per rollout the gradient of the PPO policy loss (steering mean / speed mean apart). This writes a copy of
the checkpoint with the actor moved against that gradient (descent) by a set distance in weight space; exam_record.py then
drives the copy in the frozen exam (same seed -> same worlds as the unmoved checkpoint).

    python3 grad_step.py CHECKPOINT DUMP_DIR OUT.pt DISTANCE [ROLLOUTS, default 4-10: their mean | one number: that rollout] [steer+speed | steer | speed]

DISTANCE < 0 moves the other way (the control: if the direction is real, that must drive worse).
"""
import sys, os, glob
import numpy as np, torch

ck_path, dump, out, dist = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
rr = sys.argv[5] if len(sys.argv) > 5 else "4-10"
parts = (sys.argv[6] if len(sys.argv) > 6 else "steer+speed").split("+")
lo, hi = (int(x) for x in rr.split("-")) if "-" in rr else (int(rr), int(rr))
fs = [os.path.join(dump, f"g{u}.npz") for u in range(lo, hi + 1)]
Z = [np.load(f) for f in fs]
names = [str(n) for n in Z[0]["names"]]; sizes = Z[0]["sizes"]
g = np.mean([sum(z[p].astype(np.float64) for p in parts) for z in Z], 0)
step = -dist * g / np.linalg.norm(g)
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
sd = ck["actor"]; o = 0
for n, k in zip(names, sizes):
    sd[n] = sd[n] + torch.as_tensor(step[o:o + k], dtype=sd[n].dtype).reshape(sd[n].shape); o += int(k)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
torch.save(ck, out)
print(f"{out}: {os.path.basename(dump)} rollout {rr} ({'+'.join(parts)}) 방향으로 {dist:+g} (평균 gradient 크기 {np.linalg.norm(g):.3f}, 가중치 {o:,}개)")
