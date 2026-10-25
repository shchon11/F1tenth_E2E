"""Compare the policy's plans on a bag and on its copy with the static objects erased (real_bag_erase_objects.py), at the
scans where an object sits 0.5-3.5 m ahead within 0.6 m of our heading: the plan's clearance from the object (computed in
the ORIGINAL geometry for both) and the plan's sideways offset at the object's distance. If erasing changes nothing the
policy never reacted to the object.
usage: real_bag_plan_ablation.py PLANOBJ.npz (real_bag_plan_vs_objects.py on the original) OBJS.npz PLAN_ERASED.npz"""
import math
import sys

import numpy as np
import torch

from f1sim import mpc
from f1sim.learn.obs import spec_from_dict

PO, OBJS, PE = sys.argv[1:4]
CK = "/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False)
pspec = mpc.contract_spec(mpc.contract_of(ck.get("extra")))
v_max = float(spec_from_dict(dict((ck.get("extra") or {}).get("spec") or {})).v_max)
z = np.load(PO); rows, paths = z["rows"], z["paths"]
od = np.load(OBJS)["odom"]
ze = np.load(PE); pst, plan = ze["plan_stamp"], ze["plan"]
DS, N = 0.1, 40


def path_of(a, v):
    k, L, _ = mpc.decode_wp4(torch.as_tensor(a, dtype=torch.float32)[None], torch.tensor([v], dtype=torch.float32), v_max,
                             torch.tensor([9.0]), pspec)
    k, L = k[0].double().numpy(), float(L[0])
    n = int(max(1, min(L, 4.0) / DS)); s = np.arange(1, n + 1) * DS
    seg = np.minimum((s / max(L / len(k), 1e-3)).astype(int), len(k) - 1)
    th = np.cumsum(k[seg] * DS)
    return np.stack([np.cumsum(np.cos(th) * DS), np.cumsum(np.sin(th) * DS)], 1)


def clearance(p, ox, oy):
    rr = np.hypot(p[:, 0], p[:, 1])
    if rr.max() < math.hypot(ox, oy) - 0.15:
        return np.nan
    q = p[rr <= math.hypot(ox, oy) + 0.5]
    return float(np.min(np.hypot(q[:, 0] - ox, q[:, 1] - oy))) - 0.30


def y_at(p, x):
    i = np.searchsorted(p[:, 0], x)
    return float(p[min(i, len(p) - 1), 1]) if p[:, 0].max() >= x else np.nan


out = []
for i, r in enumerate(rows):
    t0, ox, oy, v_me = r[0], r[1], r[2], r[3]
    j = int(np.argmin(np.abs(pst - t0)))
    if abs(pst[j] - t0) > 0.02:
        continue
    pe = path_of(plan[j], v_me)
    po = paths[i][np.isfinite(paths[i][:, 0])]
    out.append((ox, oy, r[5], clearance(pe, ox, oy), y_at(po, ox), y_at(pe, ox)))
o = np.array(out)
print(f"{len(o)} scans with an object ahead (and a plan in both replays)")
for lo, hi in ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5)):
    m = (o[:, 0] >= lo) & (o[:, 0] < hi)
    a, b = o[m, 2], o[m, 3]; d = np.isfinite(a) & np.isfinite(b)
    if d.sum() == 0:
        continue
    away = np.sign(o[m, 1][d]) * (o[m, 5][d] - o[m, 4][d])       # > 0: erased plan sits closer to the object's side
    print(f"  object {lo}-{hi} m (n {int(d.sum())}): plan into it {100 * np.mean(a[d] < 0):.0f} % with the object, {100 * np.mean(b[d] < 0):.0f} % with it "
          f"erased; clearance p50 {np.median(a[d]):+.2f} vs {np.median(b[d]):+.2f} m; |plan shift at the object| p50 "
          f"{np.nanmedian(np.abs(o[m, 5][d] - o[m, 4][d])):.3f} m, toward the object's side when erased p50 {np.nanmedian(away):+.3f} m")
