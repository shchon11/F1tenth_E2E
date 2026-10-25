"""Research (2026-10-25): on the real venue bags, does our policy's PLAN go around the fixed objects on the track?

The venue bags carry fixed unmapped objects (~0.3 m wide, every lap at the same spot, some in the racing lane;
real_bag_head_check.py) -- the competition's obstacles, or parked cars. The ForzaETH stack drove those runs; our policy
only planned alongside (replay_bag_policy.sh, open loop: its plan history and memory follow a path it did not choose).
For every well-localised scan with a STATIC object 0.5-3.5 m ahead near our heading (|y| < 0.6 m, car frame), the plan
is decoded exactly as the controller does (mpc.contract_of / contract_spec / decode_wp4) and walked every 0.1 m up to
its length (<= 4 m): the clearance between the path and the object's centre minus 0.15 m (object half width) minus 0.15 m
(our half width). Compared with the path the car really drove (localisation, the next 1.5 s, same frame).
usage: real_bag_plan_vs_objects.py OBJS.npz PLAN.npz OUT.npz [CKPT]"""
import math
import sys

import numpy as np
import torch

from f1sim import mpc
from f1sim.learn.obs import spec_from_dict

OBJS, PLANS, OUT = sys.argv[1:4]
CK = sys.argv[4] if len(sys.argv) > 4 else "/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar.pt"
MOUNT_X = 0.297
DS = 0.1
ck = torch.load(CK, map_location="cpu", weights_only=False)
contract = mpc.contract_of(ck.get("extra"))
pspec = mpc.contract_spec(contract)
v_max = float(spec_from_dict(dict((ck.get("extra") or {}).get("spec") or {})).v_max)
zo, zp = np.load(OBJS), np.load(PLANS)
obj, scan, od = zo["obj"], zo["scan"], zo["odom"]
cls = np.load(OBJS.replace(".npz", "_cls.npy"))
ts = scan[:, 0]
pst, plan = zp["plan_stamp"], zp["plan"]
print(f"contract {contract.get('speed_mode')} / follower {contract.get('plan_follower')}; v_max {v_max}; {len(pst)} plans, {len(scan)} kept scans")


def path_of(a, v_meas):
    k, L, v = mpc.decode_wp4(torch.as_tensor(a, dtype=torch.float32)[None], torch.tensor([v_meas], dtype=torch.float32), v_max,
                             torch.tensor([9.0]), pspec)
    k, L = k[0].double(), float(L[0])
    n = int(max(1, min(L, 4.0) / DS))
    s = np.arange(1, n + 1) * DS
    seg = np.minimum((s / max(L / k.shape[0], 1e-3)).astype(int), k.shape[0] - 1)
    kap = k.numpy()[seg]
    th = np.cumsum(kap * DS)
    return np.stack([np.cumsum(np.cos(th) * DS), np.cumsum(np.sin(th) * DS)], 1), L, v[0].numpy()


def driven(k0, t0, horizon=1.5):
    """the path the car really drove over the next `horizon` s, in the car frame at scan k0"""
    sel = (ts >= t0) & (ts <= t0 + horizon)
    x0, y0, yaw0 = scan[k0, 1], scan[k0, 2], scan[k0, 3]
    c, s = math.cos(yaw0), math.sin(yaw0)
    dx, dy = scan[sel, 1] - x0, scan[sel, 2] - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy], 1)


rows = []
PATHS, DRV = [], []
for n in np.where(cls == 2)[0]:
    k0, t0 = int(obj[n, 0]), obj[n, 1]
    ox, oy = obj[n, 2] + MOUNT_X, obj[n, 3]                      # car frame
    if not (0.5 < ox < 3.5 and abs(oy) < 0.6):
        continue
    j = int(np.argmin(np.abs(pst - t0)))
    if abs(pst[j] - t0) > 0.02:
        continue
    v_me = float(np.interp(t0, od[:, 0], od[:, 1]))
    if v_me < 1.0:
        continue
    pts, L, vprof = path_of(plan[j], v_me)
    reach = pts[np.hypot(pts[:, 0], pts[:, 1]) <= ox + 0.5]
    # only a path that gets as far as the object says anything about it (a short plan ends before it: undecided)
    long_enough = len(reach) and float(np.max(np.hypot(pts[:, 0], pts[:, 1]))) >= math.hypot(ox, oy) - 0.15
    clear_plan = float(np.min(np.hypot(reach[:, 0] - ox, reach[:, 1] - oy))) - 0.30 if long_enough else np.nan
    dp = driven(k0, t0)
    dp = dp[np.hypot(dp[:, 0], dp[:, 1]) <= ox + 0.5]
    clear_drv = float(np.min(np.hypot(dp[:, 0] - ox, dp[:, 1] - oy))) - 0.30 if len(dp) > 3 else np.nan
    rows.append((t0, ox, oy, v_me, L, clear_plan, clear_drv, float(vprof[0]), float(vprof[-1]), j, k0))
    pad = np.full((40, 2), np.nan); pad[:min(len(pts), 40)] = pts[:40]; PATHS.append(pad)
    dpad = np.full((60, 2), np.nan); dd = driven(k0, t0); dpad[:min(len(dd), 60)] = dd[:60]; DRV.append(dpad)
r = np.array(rows, dtype=np.float64).reshape(-1, 11)
print(f"scans with a static object 0.5-3.5 m ahead within 0.6 m of our heading, moving > 1 m/s: {len(r)}")
if len(r):
    for lo, hi in ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5)):
        m = (r[:, 1] >= lo) & (r[:, 1] < hi)
        if not m.any():
            continue
        cp, cd = r[m, 5], r[m, 6]
        print(f"  object {lo}-{hi} m ahead (n {int(m.sum())}, plan reaches it {int(np.isfinite(cp).sum())}): plan clears it (> 0.05 m) {100 * np.nanmean(cp > 0.05):.0f} %, plan runs into it "
              f"(< 0) {100 * np.nanmean(cp < 0):.0f} %, plan clearance p50 {np.nanmedian(cp):+.2f} m | the car really drove: clears "
              f"{100 * np.nanmean(cd > 0.05):.0f} %, into {100 * np.nanmean(cd < 0):.0f} %, p50 {np.nanmedian(cd):+.2f} m | "
              f"plan first speed p50 {np.median(r[m, 7]):.2f} vs car {np.median(r[m, 3]):.2f} m/s")
np.savez(OUT, rows=r, paths=np.array(PATHS).reshape(-1, 40, 2), driven=np.array(DRV).reshape(-1, 60, 2))
