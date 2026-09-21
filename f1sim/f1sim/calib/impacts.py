"""Impacts in the 22 recordings: what the real car's sensors do when it hits something."""
import sys
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
import numpy as np
from f1sim.calib import bagread

G = 9.80665
import glob, os
bags = sorted(d for d in glob.glob("/home/shchon11/F1tenth_E2E/real_data/*/*") if os.path.isdir(d) and any(f.endswith(".db3") for f in os.listdir(d)))
print(f"{len(bags)} bags\n")
rows = []
for p in bags:
    try:
        b = bagread.read(p, topics=["/odom", "/sensors/imu/raw"])
    except Exception as e:
        print(f"  {p.split('/')[-1][:40]:42s} unreadable: {type(e).__name__}"); continue
    if not b.has("/odom", "/sensors/imu/raw"):
        print(f"  {b.name[:42]:42s} missing a topic"); continue
    ti, imu = b.t["/sensors/imu/raw"], b.v["/sensors/imu/raw"]
    ax, ay = imu[:, 3], imu[:, 4]   # [wx,wy,wz,ax,ay,az]
    amag = np.hypot(ax, ay)
    to, od = b.t["/odom"], b.v["/odom"]
    v = od[:, 3]                      # [x,y,yaw,vx,vy,wz]
    # an impact: |a| far above anything braking or cornering can do (mu*g ~ 1.05 g), sustained
    # for only a few samples, with the speed dropping across it
    thr = 2.5 * G
    spikes = np.flatnonzero(amag > thr)
    ev = []
    for i in spikes:
        if ev and ti[i] - ev[-1][-1] < 0.25:      # same event
            ev[-1][-1] = ti[i]; continue
        ev.append([ti[i], ti[i]])
    for t0, t1 in ev:
        pre = (to > t0 - 0.30) & (to <= t0)
        post = (to >= t1) & (to < t1 + 0.30)
        if pre.sum() < 2 or post.sum() < 2:
            continue
        vb, va = float(np.median(v[pre])), float(np.median(v[post]))
        w = (ti >= t0 - 0.05) & (ti <= t1 + 0.05)
        rows.append((b.name, t0, float(amag[w].max()) / G, vb, va, vb - va, t1 - t0))
print(f"{'bag':44s}{'t [s]':>8}{'peak |a| [g]':>14}{'v before':>10}{'v after':>9}{'drop':>7}{'dur [s]':>9}")
for n, t0, pk, vb, va, dv, dur in sorted(rows, key=lambda r: -r[2])[:25]:
    print(f"{n[:44]:44s}{t0:8.2f}{pk:14.2f}{vb:10.2f}{va:9.2f}{dv:7.2f}{dur:9.3f}")
print(f"\n{len(rows)} spike events over {len(bags)} bags")
