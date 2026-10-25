"""Research (2026-10-25): a dedicated 'car beside us' reading on the policy's frozen features, for the lateral guard.

The lateral guard (lateral_guard_race.py) helped from the true state (hard car contacts -25 %, passes +29 %) but HURT from the
policy's nearest-car output (total 1.00 -> 1.24 /km): that output is trained on cars the LiDAR sees (>= 3 beams), and a car
right beside us is seen 87 % of the time, a little behind 67 %. Trained here on the TRUE state instead (any visibility):
[logit of 'a car overlaps us lengthwise within 0.7 m and sits within 0.8 m sideways', its sideways offset dy] from the
policy's features z (the GRU state remembers a car that left the scan). Probe on seed 81: recall 0.87 / precision 0.83 vs
the nearest-car output's 0.77 / 0.86; side right 99.7 %, |dy| error 0.07 m.
usage: beside_head_train.py TRAIN.pt[,...] VAL.pt OUT.pt [EPOCHS=6]"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

TR, VA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 6
torch.manual_seed(0)


def load(paths):
    vs = [v for p in paths.split(",") for v in torch.load(p).values()]
    z, g = torch.cat([v["z"] for v in vs]).float(), torch.cat([v["gt"] for v in vs])
    dist, lx, ly, cl, dvy, vme = g.T
    dx = lx + 0.297
    return z, ((dx.abs() < 0.7) & (ly.abs() < 0.8)).float(), ly


zt, yt, dyt = load(TR)
zv, yv, dyv = load(VA)
net = nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Linear(128, 2))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for ep in range(EPOCHS):
    perm = torch.randperm(len(yt))
    for i in range(0, len(yt), 4096):
        idx = perm[i:i + 4096]
        o = net(zt[idx])
        loss = F.binary_cross_entropy_with_logits(o[:, 0], yt[idx]) + (yt[idx] * (o[:, 1] - dyt[idx]) ** 2).sum() / yt[idx].sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    o = net(zv); pred = o[:, 0] > 0
    tp = (pred & (yv > 0)).sum().item(); fp = (pred & (yv == 0)).sum().item(); fn = (~pred & (yv > 0)).sum().item()
print(f"val: recall {tp / max(tp + fn, 1):.3f} precision {tp / max(tp + fp, 1):.3f}")
torch.save({"state_dict": net.state_dict(), "in_dim": 384, "hidden": 128, "near": 0.8, "overlap": 0.7}, OUT)
print("saved", OUT)
