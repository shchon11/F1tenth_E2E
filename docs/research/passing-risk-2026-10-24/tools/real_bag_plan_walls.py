"""Control for real_bag_plan_vs_objects.py: on the replayed real bags (open loop), do our policy's plans stay off the WALLS?
If plans also run into walls often, the open-loop replay (plan history and memory that follow ForzaETH's path, not ours)
degrades the plans in general and the obstacle result says little about obstacles. Every kept scan's plan, decoded as the
controller does, walked every 0.1 m up to min(L, 4 m), placed on the map with the localisation pose: the smallest distance
to an occupied cell minus 0.15 m (our half width) over the first 1.5 m and over the whole walk.
usage: real_bag_plan_walls.py MAP.npz (real_bag_unexplained.py DUMP: occ, res, origin) OBJS.npz PLAN.npz"""
import math
import sys

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from f1sim import mpc
from f1sim.learn.obs import spec_from_dict

MAP, OBJS, PLANS = sys.argv[1:4]
CK = "/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar.pt"
zm = np.load(MAP)
occ, res, (ox, oy, _) = zm["occ"], float(zm["res"]), zm["origin"]
H, W = occ.shape
dist = distance_transform_edt(~occ) * res
ck = torch.load(CK, map_location="cpu", weights_only=False)
pspec = mpc.contract_spec(mpc.contract_of(ck.get("extra")))
v_max = float(spec_from_dict(dict((ck.get("extra") or {}).get("spec") or {})).v_max)
zo, zp = np.load(OBJS), np.load(PLANS)
scan, od = zo["scan"], zo["odom"]
pst, plan = zp["plan_stamp"], zp["plan"]
j = np.searchsorted(pst, scan[:, 0]).clip(1, len(pst) - 1)
j = np.where(np.abs(pst[j - 1] - scan[:, 0]) < np.abs(pst[j] - scan[:, 0]), j - 1, j)
ok = (np.abs(pst[j] - scan[:, 0]) < 0.02)
v = np.interp(scan[:, 0], od[:, 0], od[:, 1])
ok &= v > 1.0
idx = np.where(ok)[0]
A = torch.as_tensor(plan[j[idx]], dtype=torch.float32)
k, L, _ = mpc.decode_wp4(A, torch.as_tensor(v[idx], dtype=torch.float32), v_max, torch.full((len(idx),), 9.0), pspec)
DS, N = 0.1, 40
s = torch.arange(1, N + 1, dtype=torch.float32) * DS
seg = (s[None, :] / (L[:, None] / k.shape[1]).clamp_min(1e-3)).floor().long().clamp(max=k.shape[1] - 1)
th = torch.cumsum(k.gather(1, seg) * DS, 1)
px, py = torch.cumsum(torch.cos(th) * DS, 1).numpy(), torch.cumsum(torch.sin(th) * DS, 1).numpy()
on = (s[None, :] <= torch.minimum(L, torch.full_like(L, 4.0))[:, None]).numpy()
x, y, yaw = scan[idx, 1:2], scan[idx, 2:3], scan[idx, 3:4]
wx, wy = x + np.cos(yaw) * px - np.sin(yaw) * py, y + np.sin(yaw) * px + np.cos(yaw) * py
gi, gj = np.floor((wy - oy) / res).astype(int), np.floor((wx - ox) / res).astype(int)
inside = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W)
d = np.where(inside, dist[gi.clip(0, H - 1), gj.clip(0, W - 1)], 0.0) - 0.15
first = on & (s.numpy()[None, :] <= 1.5)
c15 = np.where(first, d, np.inf).min(1); call = np.where(on, d, np.inf).min(1)
print(f"{len(idx)} plans on kept scans (moving > 1 m/s): into a wall within 1.5 m {100 * np.mean(c15 < 0):.1f} %, "
      f"within the whole plan (<= 4 m) {100 * np.mean(call < 0):.1f} %; wall clearance p10/p50 first 1.5 m "
      f"{np.percentile(c15, 10):+.2f}/{np.percentile(c15, 50):+.2f} m, whole {np.percentile(call, 10):+.2f}/{np.percentile(call, 50):+.2f} m")
