import sys, glob, os
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
import numpy as np
from f1sim.calib import bagread
G = 9.80665
bags = sorted(d for d in glob.glob("/home/shchon11/F1tenth_E2E/real_data/01_competition*/*")
              if os.path.isdir(d) and any(f.endswith(".db3") for f in os.listdir(d)))
rat, dur, peak, dv = [], [], [], []
for p in bags:
    try: b = bagread.read(p, topics=["/odom", "/sensors/imu/raw"])
    except Exception: continue
    if not b.has("/odom", "/sensors/imu/raw"): continue
    ti, imu = b.t["/sensors/imu/raw"], b.v["/sensors/imu/raw"]
    amag = np.hypot(imu[:, 3], imu[:, 4])
    to, v = b.t["/odom"], b.v["/odom"][:, 3]
    ev = []
    for i in np.flatnonzero(amag > 2.5 * G):
        if ev and ti[i] - ev[-1][1] < 0.25: ev[-1][1] = ti[i]; continue
        ev.append([ti[i], ti[i]])
    for t0, t1 in ev:
        pre, post = (to > t0 - 0.30) & (to <= t0), (to >= t1) & (to < t1 + 0.30)
        if pre.sum() < 2 or post.sum() < 2: continue
        vb, va = float(np.max(v[pre])), float(np.min(v[post]))
        if vb < 1.0: continue                       # not an impact worth calling one
        w = (ti >= t0 - 0.05) & (ti <= t1 + 0.05)
        rat.append(va / vb); dur.append(t1 - t0 + 0.02); peak.append(float(amag[w].max()) / G)
        dv.append(vb - va)
r = np.array(rat); d = np.array(dur); pk = np.array(peak); dvv = np.array(dv)
print(f"n = {len(r)} impacts above 1 m/s, competition track (duct boundary)")
q = lambda a, x: float(np.percentile(a, x))
print(f"  speed kept  v_after/v_before : p10 {q(r,10):.2f}  median {q(r,50):.2f}  p90 {q(r,90):.2f}")
print(f"  speed lost  [m/s]            : p10 {q(dvv,10):.2f}  median {q(dvv,50):.2f}  p90 {q(dvv,90):.2f}")
print(f"  peak |a|    [g]              : p10 {q(pk,10):.2f}  median {q(pk,50):.2f}  max {pk.max():.2f}")
print(f"  duration    [s]              : p10 {q(d,10):.3f}  median {q(d,50):.3f}  p90 {q(d,90):.3f}")
print(f"  came to a full stop          : {int((r < 0.05).sum())} of {len(r)}")
