"""Research (2026-10-25): the policy's nearest-car output on a real bag, against the objects the map cannot explain.

Inputs: OBJS.npz from real_bag_unexplained.py (OBJS=..., GOOD=0.95: object candidates of well-localised scans, LiDAR and
map frame) and the policy's readings (record_perception.py). An object is MOVING when its nearest candidate 0.1 s later
(map frame) lies 0.25-1.5 m away (2.5-15 m/s), STATIC when within 0.1 m. Reports
  * what the objects are: moving / static, where (map), how big;
  * precision: of the policy's 'car seen' readings, how many have an object candidate within 0.6 m of where it says --
    against the same test with the readings shuffled in time (the chance level when objects are everywhere);
  * recall: of the scans with a moving object 0.3-4 m ahead within 1.5 m sideways (>= 8 beams), how many the policy
    calls 'car seen', and its position error there (head vs the object's visible-surface centre).
usage: real_bag_head_check.py OBJS.npz PERCEPTION.npz"""
import math
import sys

import numpy as np

zo = np.load(sys.argv[1]); zp = np.load(sys.argv[2])
obj, scan = zo["obj"], zo["scan"]            # obj: k, t, ax, ay, beams, clearance, wx, wy
ts = scan[:, 0]
by_scan = {}
for r in obj:
    by_scan.setdefault(int(r[0]), []).append(r)
xs, ys = scan[:, 1], scan[:, 2]; yaw_s = np.unwrap(scan[:, 3])


def track(n, half=0.2):
    """the object followed scan to scan over +-half s (map frame): (times, x, y) or None"""
    r = obj[n]; k0 = int(r[0]); t0 = r[1]
    pts = [(t0, r[6], r[7])]
    for step in (1, -1):
        k, (tl, xl, yl) = k0, pts[0]
        while True:
            k += step
            if k < 0 or k >= len(ts) or abs(ts[k] - t0) > half:
                break
            if k not in by_scan:
                continue
            gate = 0.35 + 6.0 * abs(ts[k] - tl)
            o = min(by_scan[k], key=lambda o: math.hypot(o[6] - xl, o[7] - yl))
            if math.hypot(o[6] - xl, o[7] - yl) > gate:
                break
            tl, xl, yl = ts[k], o[6], o[7]; pts.append((tl, xl, yl))
    p = np.array(sorted(pts))
    return p if len(p) >= 6 and p[-1, 0] - p[0, 0] >= 0.3 else None


# motion over the 0.4 s track: MOVING = fitted speed 2.5-15 m/s and >= 0.8 m net displacement (a static object's visible
# centre slides by at most its size as we pass it, which a single 0.1 s match read as 2.5+ m/s); STATIC = < 0.8 m/s
TR = [track(n) for n in range(len(obj))]
spd = np.full(len(obj), np.nan); disp = np.full(len(obj), np.nan)
for n, p in enumerate(TR):
    if p is None:
        continue
    v = np.polyfit(p[:, 0] - p[0, 0], p[:, 1:], 1)[0]
    spd[n] = float(np.hypot(*v)); disp[n] = float(np.hypot(p[-1, 1] - p[0, 1], p[-1, 2] - p[0, 2]))
ok = np.isfinite(spd)
moving = ok & (spd > 2.5) & (spd < 15) & (disp > 0.8)
static = ok & (spd < 0.8)
np.save(sys.argv[1].replace(".npz", "_cls.npy"), np.where(moving, 1, np.where(static, 2, 0)))   # for real_bag_frames.py
print(f"object candidates {len(obj)} in {len(scan)} well-localised scans; followed for >= 0.3 s {int(ok.sum())}: "
      f"moving (2.5-15 m/s, >= 0.8 m) {100 * moving[ok].mean():.1f} %, static (< 0.8 m/s) {100 * static[ok].mean():.1f} %, other {100 * (1 - moving[ok].mean() - static[ok].mean()):.1f} %")
if moving.any():
    print(f"  moving: speed p10/50/90 {np.percentile(spd[moving], [10, 50, 90]).round(2)} m/s, beams p50 {np.median(obj[moving, 4]):.0f}, "
          f"range p50 {np.median(np.hypot(obj[moving, 2], obj[moving, 3])):.2f} m")
if static.any():
    print(f"  static: beams p50 {np.median(obj[static, 4]):.0f}, wall clearance p50 {np.median(obj[static, 5]):.2f} m")
    # where the static ones sit (map frame, 0.5 m cells, top 8)
    cells = {}
    for r in obj[static]:
        c = (round(r[6] * 2) / 2, round(r[7] * 2) / 2); cells[c] = cells.get(c, 0) + 1
    print("  static hot spots (map x, y: count):", sorted(cells.items(), key=lambda kv: -kv[1])[:8])
# the policy's readings on the kept scans
st, rd = zp["stamp"], zp["reading"]
idx = np.searchsorted(ts, st).clip(1, len(ts) - 1)
idx = np.where(np.abs(ts[idx - 1] - st) < np.abs(ts[idx] - st), idx - 1, idx)
m = np.abs(ts[idx] - st) < 0.02
seen = (rd[:, 0] > 0) & m
hx, hy = rd[:, 1] * 5.0, rd[:, 2] * 2.0


def support(k, x, y, cand_mask=None):
    for o in by_scan.get(int(k), []):
        if math.hypot(o[2] - x, o[3] - y) < 0.6:
            return True
    return False


sup = np.array([support(idx[i], hx[i], hy[i]) for i in range(len(st))])
rng = np.random.default_rng(0)
sh = rng.permutation(np.where(seen)[0])
sup_null = np.array([support(idx[i], hx[j], hy[j]) for i, j in zip(np.where(seen)[0], sh)])
print(f"policy 'car seen' on kept scans {int(seen.sum())} of {int(m.sum())} ({100 * seen.sum() / max(m.sum(), 1):.1f} %): "
      f"an object within 0.6 m {100 * sup[seen].mean():.1f} % (readings shuffled in time: {100 * sup_null.mean():.1f} %)")
# recall on scans with a moving object ahead
mv_scans = {}
for n in np.where(moving)[0]:
    r = obj[n]
    if 0.3 < r[2] < 4.0 and abs(r[3]) < 1.5 and r[4] >= 8:
        k = int(r[0])
        if k not in mv_scans or r[2] < mv_scans[k][2]:
            mv_scans[k] = r
inv = {int(k): i for i, k in enumerate(idx) if m[i]}
hit = []; ex = []; ey = []
for k, r in mv_scans.items():
    if k not in inv:
        continue
    i = inv[k]
    hit.append(rd[i, 0] > 0)
    if rd[i, 0] > 0:
        ex.append(hx[i] - r[2]); ey.append(hy[i] - r[3])
hit = np.array(hit)
print(f"scans with a MOVING object 0.3-4 m ahead (>= 8 beams): {len(hit)}; policy says 'car seen' {100 * hit.mean():.1f} %")
if ex:
    ex, ey = np.array(ex), np.array(ey)
    print(f"  head minus object surface centre: dx median {np.median(ex):+.2f} (|.| p50 {np.median(np.abs(ex)):.2f}, p90 {np.percentile(np.abs(ex), 90):.2f}) m, "
          f"dy median {np.median(ey):+.2f} (|.| p50 {np.median(np.abs(ey)):.2f}, p90 {np.percentile(np.abs(ey), 90):.2f}) m")
# the speeds the speed limit reads: closing (along the line of sight) and v_lat (sideways, relative, our frame). Reference:
# the moving object's track (above), a straight-line fit of its position minus the same fit of ours (localisation), rotated
# into our frame at the reading's time


def rel_speeds(n):
    p = TR[n]
    if p is None:
        return None
    r = obj[n]; k0 = int(r[0]); t0 = r[1]
    vo = np.polyfit(p[:, 0] - t0, p[:, 1:], 1)[0]                    # object velocity (map)
    sel = np.abs(ts - t0) <= 0.2
    vm = np.polyfit(ts[sel] - t0, np.stack([xs[sel], ys[sel]], 1), 1)[0]   # our velocity (map)
    c, s_ = math.cos(yaw_s[k0]), math.sin(yaw_s[k0])
    dvx, dvy = c * (vo[0] - vm[0]) + s_ * (vo[1] - vm[1]), -s_ * (vo[0] - vm[0]) + c * (vo[1] - vm[1])
    rr = math.hypot(r[2], r[3])
    return -(dvx * r[2] + dvy * r[3]) / rr, dvy


ref = {}
pairs = []
static_hits = moving_hits = 0
sup_kind = np.zeros(len(st), dtype=int)                          # 0 none / no kept scan, 1 moving, 2 static, 3 other
for i in np.where(seen)[0]:
    k = int(idx[i]); best = None
    for o in by_scan.get(k, []):
        d = math.hypot(o[2] - hx[i], o[3] - hy[i])
        if d < 0.6 and (best is None or d < best[0]):
            best = (d, o)
    if best is None:
        continue
    o = best[1]
    n = int(np.where((obj[:, 0] == o[0]) & (obj[:, 2] == o[2]) & (obj[:, 3] == o[3]))[0][0])
    sup_kind[i] = 1 if moving[n] else 2 if static[n] else 3
    static_hits += bool(static[n]); moving_hits += bool(moving[n])
    if moving[n]:
        if n not in ref:
            ref[n] = rel_speeds(n)
        if ref[n] is not None:
            pairs.append((rd[i, 3] * 5.0, ref[n][0], rd[i, 4] * 2.0, ref[n][1]))
print(f"'car seen' readings on an object: on a MOVING one {moving_hits}, on a STATIC one {static_hits}")
if pairs:
    pr = np.array(pairs)
    e_cl, e_vl = pr[:, 0] - pr[:, 1], pr[:, 2] - pr[:, 3]
    np.save(sys.argv[1].replace(".npz", "_speedpairs.npy"), pr)   # (head closing, measured, head v_lat, measured)
    print(f"  closing head minus measured: median {np.median(e_cl):+.2f} m/s (> 0: the head reads the other car slower than it is)")
    print(f"  on moving objects (n {len(pr)}): closing head p50 {np.median(pr[:, 0]):+.2f} vs measured {np.median(pr[:, 1]):+.2f} m/s, "
          f"|error| p50 {np.median(np.abs(e_cl)):.2f} p90 {np.percentile(np.abs(e_cl), 90):.2f}, corr {np.corrcoef(pr[:, 0], pr[:, 1])[0, 1]:.2f}; "
          f"v_lat head |.| p50 {np.median(np.abs(pr[:, 2])):.2f} vs measured {np.median(np.abs(pr[:, 3])):.2f}, |error| p50 {np.median(np.abs(e_vl)):.2f} "
          f"p90 {np.percentile(np.abs(e_vl), 90):.2f}, corr {np.corrcoef(pr[:, 2], pr[:, 3])[0, 1]:.2f}")
# what the shipped speed limit would have done on this run: limit vs the car's own speed at each scan
import torch
from f1sim.opponent_speed_limit import speed_limit
od = zo["odom"]
v_me = np.interp(st, od[:, 0], od[:, 1])
lim = speed_limit(torch.as_tensor(rd, dtype=torch.float32), torch.as_tensor(v_me, dtype=torch.float32)).numpy()
fin = np.isfinite(lim); slow = fin & (lim < v_me)
print(f"speed limit (shipped rule) on all {len(st)} scans: applies {100 * fin.mean():.1f} %, below the car's own speed {100 * slow.mean():.1f} % "
      f"(mean cut {np.mean(v_me[slow] - lim[slow]) if slow.any() else 0:.2f} m/s, car speed there p50 {np.median(v_me[slow]) if slow.any() else 0:.2f})")
kept = slow & m
if kept.any():
    kinds = np.bincount(sup_kind[kept], minlength=4)
    print(f"  of those on well-localised scans ({int(kept.sum())}): behind a MOVING object {100 * kinds[1] / kept.sum():.1f} %, a STATIC one "
          f"{100 * kinds[2] / kept.sum():.1f} %, another object {100 * kinds[3] / kept.sum():.1f} %, NOTHING within 0.6 m {100 * kinds[0] / kept.sum():.1f} %")
