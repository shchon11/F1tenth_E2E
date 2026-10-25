"""Research (2026-10-26): the map-free obstacle limit (scan_limit_race.py CLUSTER_FILTER) on the real venue runs, open loop: on every
scan, the small isolated clusters of the REAL scan in the policy's recorded plan (as the controller decodes it), minus returns within
0.45 m of the policy's own nearest-car reading; how often would it hold the car below its own speed, by how much, and what was
there (a fixed box / a moving object / something else, from real_bag_head_check.py's classes on well-localised scans)?
Step 1 (ROS sourced): dump the scans -> SCANS.npz; step 2 (venv): the analysis.
usage: real_bag_cluster_limit.py dump BAG SCANS.npz  |  real_bag_cluster_limit.py run SCANS.npz PLAN.npz OBJS.npz"""
import math
import sys

import numpy as np

if sys.argv[1] == "dump":
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    BAG, OUT = sys.argv[2:4]
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=["/scan"]))
    st, rr = [], []
    a0 = inc = None
    while r.has_next():
        _, data, _ = r.read_next()
        m = deserialize_message(data, get_message(types["/scan"]))
        st.append(m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec); rr.append(np.asarray(m.ranges, np.float32))
        a0, inc = m.angle_min, m.angle_increment
    np.savez(OUT, stamp=np.array(st), ranges=np.stack(rr), angle_min=a0, angle_inc=inc)
    print("saved", OUT, len(st))
    sys.exit(0)

import torch
from f1sim import mpc
from f1sim.learn.obs import spec_from_dict

SC, PL, OBJS = sys.argv[2:5]
CK = "/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar_slow.pt"
# the shipped rule (f1sim.obstacle_speed_limit.cluster_keep); env LOOK: its stand_look (10-26: 1 = the rule before, 3 = after)
import os
from dataclasses import replace
from f1sim.obstacle_speed_limit import DEFAULT as _P, cluster_keep as _ck
_PL = replace(_P, stand_look=int(os.environ.get("LOOK", str(_P.stand_look))),
              stand_valid_only=os.environ.get("VALID", "1" if _P.stand_valid_only else "0") == "1")
cluster_keep = lambda r, ok, px, py: _ck(r, ok, px, py, _PL)
ck = torch.load(CK, map_location="cpu", weights_only=False)
pspec = mpc.contract_spec(mpc.contract_of(ck.get("extra")))
v_max = float(spec_from_dict(dict((ck.get("extra") or {}).get("spec") or {})).v_max)
zs, zp, zo = np.load(SC), np.load(PL), np.load(OBJS)
od = zo["odom"]; obj, scan_k = zo["obj"], zo["scan"]
cls = np.load(OBJS.replace(".npz", "_cls.npy"))
MX, THR, FRONT, GMIN, TAU, A_B, DS, N_S = 0.297, 0.19, 0.45, 0.1, 0.2, 3.0, 0.1, 60
pst, plan, rd = zp["plan_stamp"], zp["plan"], zp["reading"]
st_s, R_ = zs["stamp"], zs["ranges"]
ang = float(zs["angle_min"]) + float(zs["angle_inc"]) * np.arange(R_.shape[1])
j = np.searchsorted(st_s, pst).clip(1, len(st_s) - 1)
j = np.where(np.abs(st_s[j - 1] - pst) < np.abs(st_s[j] - pst), j - 1, j)
m = np.abs(st_s[j] - pst) < 0.002
v = np.interp(pst, od[:, 0], od[:, 1])
idx = np.where(m & (v > 1.0))[0]
r = torch.as_tensor(R_[j[idx]], dtype=torch.float32)
ok = torch.isfinite(r) & (r > 0.05) & (r < 12.0)
rr = torch.where(ok, r, torch.zeros_like(r))
a = torch.as_tensor(ang, dtype=torch.float32)[None]
px, py = MX + rr * torch.cos(a), rr * torch.sin(a)
keep = cluster_keep(r, ok, px, py)
rdx = torch.as_tensor(rd[idx], dtype=torch.float32)
ex, cx, cy = rdx[:, 0] > 0, rdx[:, 1] * 5 + MX, rdx[:, 2] * 2
keep = keep & ~(ex[:, None] & (torch.hypot(px - cx[:, None], py - cy[:, None]) < 0.45))
A = torch.as_tensor(plan[idx], dtype=torch.float32); V = torch.as_tensor(v[idx], dtype=torch.float32)
k, L, _ = mpc.decode_wp4(A, V, v_max, torch.full((len(idx),), 9.0), pspec)
s = torch.arange(1, N_S + 1, dtype=torch.float32) * DS
seg = (s[None, :] / (L[:, None] / k.shape[1]).clamp_min(1e-3)).floor().long().clamp(max=k.shape[1] - 1)
th = torch.cumsum(k.gather(1, seg) * DS, 1)
qx, qy = torch.cumsum(torch.cos(th) * DS, 1), torch.cumsum(torch.sin(th) * DS, 1)
lim = np.full(len(idx), np.inf); hit_xy = np.full((len(idx), 2), np.nan)
for c0 in range(0, len(idx), 256):
    sl = slice(c0, c0 + 256)
    d2 = (qx[sl, :, None] - px[sl, None, :]) ** 2 + (qy[sl, :, None] - py[sl, None, :]) ** 2
    d2 = torch.where(keep[sl, None, :], d2, torch.full_like(d2, 1e9))
    dmin, bi = d2.min(2)
    near = (dmin.sqrt() < THR) & (s[None, :] <= L[sl, None]) & (s[None, :] > FRONT)
    sh = torch.where(near, s[None, :].expand_as(near), torch.full_like(near, float("inf"), dtype=s.dtype))
    shit, si = sh.min(1)
    room = (shit - FRONT - GMIN - V[sl].clamp_min(0) * TAU).clamp_min(0.0)
    lim[sl] = torch.where(torch.isfinite(shit), torch.sqrt(2 * A_B * room), torch.full_like(shit, float("inf"))).numpy()
    b = bi.gather(1, si[:, None]).squeeze(1)
    rows = torch.arange(b.shape[0])
    hit_xy[sl, 0], hit_xy[sl, 1] = px[sl][rows, b].numpy(), py[sl][rows, b].numpy()
vv = v[idx]
slow = np.isfinite(lim) & (lim < vv)
# what triggered it (well-localised scans): a fixed box / moving object within 0.6 m of the triggering return
ts = scan_k[:, 0]
kind = np.full(len(idx), -1)
for q in np.where(slow)[0]:
    t = pst[idx[q]]; kk = int(np.argmin(np.abs(ts - t)))
    if abs(ts[kk] - t) > 0.02:
        continue
    cands = np.where(obj[:, 0] == kk)[0]
    best = None
    for n in cands:
        d = math.hypot(obj[n, 2] + MX - hit_xy[q, 0], obj[n, 3] - hit_xy[q, 1])
        if d < 0.6 and (best is None or d < best[0]):
            best = (d, n)
    kind[q] = 0 if best is None else (1 if cls[best[1]] == 1 else 2 if cls[best[1]] == 2 else 3)
kk = kind[slow]; known = kk >= 0
print(f"{len(idx)} scans (moving > 1 m/s): the obstacle limit would hold the car below its speed on {100 * slow.mean():.1f} % "
      f"(mean cut {np.mean((vv - lim)[slow]) if slow.any() else 0:.2f} m/s); on well-localised scans ({int(known.sum())}): fixed box "
      f"{100 * np.mean(kk[known] == 2):.0f} %, moving object {100 * np.mean(kk[known] == 1):.0f} %, other object {100 * np.mean(kk[known] == 3):.0f} %, "
      f"nothing found there (a wall feature?) {100 * np.mean(kk[known] == 0):.0f} %")
