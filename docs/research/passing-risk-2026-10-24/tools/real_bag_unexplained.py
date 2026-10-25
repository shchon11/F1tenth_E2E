"""Research (2026-10-25): is there another car in a real bag where the policy's nearest-car output says so?

The bag carries the venue's map (/map), the particle filter's pose (/pf/pose/odom) and the LiDAR mount (/tf_static). Every
scan point is placed on the map; a point farther than TOL from any occupied cell is UNEXPLAINED (another car, a person, a
prop -- or a localisation error). Runs of >= MIN_BEAMS consecutive unexplained beams that hang together (range jumps <
0.3 m) are object candidates, as the simulator's own visibility rule counts a car the LiDAR sees (>= 3 beams).
With PERCEPTION.npz (record_perception.py): for each of the policy's 'car seen' readings, is an object candidate within
0.6 m of where it says the car is?
usage (ROS sourced): real_bag_unexplained.py BAG [PERCEPTION.npz] [TOL=0.25] [MIN_BEAMS=3]"""
import math
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.ndimage import distance_transform_edt

BAG = sys.argv[1]
PERC = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".npz") else None
TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
MIN_BEAMS = int(sys.argv[4]) if len(sys.argv) > 4 else 3


def read(topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    out = {t: [] for t in topics}
    while r.has_next():
        t, data, _ts = r.read_next()
        out[t].append(deserialize_message(data, get_message(types[t])))
    return out


def stamp(h):
    return h.stamp.sec + 1e-9 * h.stamp.nanosec


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


m = read(["/scan", "/pf/pose/odom", "/map", "/tf_static", "/odom"])
grid = m["/map"][0]
res, W, H = grid.info.resolution, grid.info.width, grid.info.height
ox, oy, oyaw = grid.info.origin.position.x, grid.info.origin.position.y, yaw_of(grid.info.origin.orientation)
occ = (np.asarray(grid.data, dtype=np.int16).reshape(H, W) >= 50)
dist = distance_transform_edt(~occ) * res                       # [m] to the nearest occupied cell
laser = None
for tfm in m["/tf_static"]:
    for tr in tfm.transforms:
        if "laser" in tr.child_frame_id:
            laser = (tr.header.frame_id, tr.child_frame_id, tr.transform.translation.x, tr.transform.translation.y,
                     yaw_of(tr.transform.rotation))
poses = m["/pf/pose/odom"]
pf_frame = (poses[0].header.frame_id, poses[0].child_frame_id)
print(f"map {W}x{H} @ {res} m, origin ({ox:.2f}, {oy:.2f}, {oyaw:.3f}); laser tf {laser}; pf pose frame {pf_frame}; "
      f"{len(m['/scan'])} scans, {len(poses)} poses")
pt = np.array([stamp(p.header) for p in poses])
px = np.array([p.pose.pose.position.x for p in poses]); py = np.array([p.pose.pose.position.y for p in poses])
pyaw = np.unwrap(np.array([yaw_of(p.pose.pose.orientation) for p in poses]))
lx, ly, lyaw = (laser[2], laser[3], laser[4]) if laser else (0.0, 0.0, 0.0)


def map_dist(wx, wy):
    c, s = math.cos(-oyaw), math.sin(-oyaw)
    gx, gy = c * (wx - ox) - s * (wy - oy), s * (wx - ox) + c * (wy - oy)
    i, j = np.floor(gy / res).astype(int), np.floor(gx / res).astype(int)
    inside = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    d = np.full(wx.shape, np.inf)
    d[inside] = dist[i[inside], j[inside]]
    return d


scans = []                                                      # per scan: stamp, explained fraction, object candidates
n_pts = n_expl = 0
import os
DUMP = os.environ.get("DUMP")                                   # .npz of the unexplained points on the map (every 5th scan)
# keep only scans whose pose fits the map: of the points within 1 m of a wall, the fraction within TOL is >= GOOD. (Not
# the explained fraction of all points: a car close ahead covers many beams and would drop exactly the scans that matter.)
GOOD = float(os.environ.get("GOOD", "0"))
n_bad = 0
dump_pts, dump_traj = [], []
for k_scan, sc in enumerate(m["/scan"]):
    t = stamp(sc.header)
    if t < pt[0] or t > pt[-1]:
        continue
    x, y, yaw = np.interp(t, pt, px), np.interp(t, pt, py), np.interp(t, pt, pyaw)
    r = np.asarray(sc.ranges, dtype=np.float64)
    th = sc.angle_min + sc.angle_increment * np.arange(len(r))
    ok = np.isfinite(r) & (r > max(sc.range_min, 0.05)) & (r < min(sc.range_max, 12.0))
    qx, qy = r * np.cos(th + lyaw) + lx, r * np.sin(th + lyaw) + ly          # base frame
    wx, wy = x + math.cos(yaw) * qx - math.sin(yaw) * qy, y + math.sin(yaw) * qx + math.cos(yaw) * qy
    d = map_dist(wx, wy)
    un = ok & (d > TOL)
    frac = float((ok & ~un).sum()) / max(int((ok & (d < 1.0)).sum()), 1)
    if frac < GOOD:
        n_bad += 1; continue
    n_pts += int(ok.sum()); n_expl += int((ok & ~un).sum())
    if DUMP and k_scan % 5 == 0:
        dump_pts.append(np.stack([wx[un], wy[un], np.full(int(un.sum()), t)], 1)); dump_traj.append((t, x, y, yaw))
    objs = []
    i = 0
    while i < len(r):
        if not un[i]:
            i += 1; continue
        j = i
        while j + 1 < len(r) and un[j + 1] and abs(r[j + 1] - r[j]) < 0.3:
            j += 1
        if j - i + 1 >= MIN_BEAMS:
            sl = slice(i, j + 1)
            ax, ay = r[sl] * np.cos(th[sl]), r[sl] * np.sin(th[sl])            # LiDAR frame
            objs.append((float(ax.mean()), float(ay.mean()), j - i + 1, float(d[sl].min()), float(wx[sl].mean()),
                         float(wy[sl].mean())))
        i = j + 1
    scans.append((t, objs, x, y, yaw))
if DUMP:
    np.savez(DUMP, pts=np.concatenate(dump_pts), traj=np.array(dump_traj), occ=occ, res=res, origin=np.array([ox, oy, oyaw]))
print(f"scan points on the map within {TOL} m of a wall: {100 * n_expl / max(n_pts, 1):.1f} % of {n_pts}"
      + (f" (scans kept whose wall points fit the map >= {GOOD:.2f}: {len(scans)}, dropped {n_bad})" if GOOD else ""))
ts = np.array([s[0] for s in scans])
ahead = [[o for o in s[1] if 0.0 < o[0] < 6.0] for s in scans]
close = [[o for o in s[1] if 0.0 < o[0] < 3.0 and abs(o[1]) < 0.6] for s in scans]
print(f"scans with an unexplained object (>= {MIN_BEAMS} beams): anywhere {100 * np.mean([len(s[1]) > 0 for s in scans]):.1f} %, "
      f"ahead within 6 m {100 * np.mean([len(a) > 0 for a in ahead]):.1f} %, within 3 m ahead in our lane {100 * np.mean([len(c) > 0 for c in close]):.1f} %")
sizes = [o[2] for s in scans for o in s[1]]
if sizes:
    print(f"object candidates: {len(sizes)}, beams p50 {np.median(sizes):.0f} p90 {np.percentile(sizes, 90):.0f}, "
          f"wall clearance p50 {np.median([o[3] for s in scans for o in s[1]]):.2f} m")
if os.environ.get("OBJS"):                                      # every kept scan's object candidates, for the motion analysis
    rows = [(k, s[0], *o) for k, s in enumerate(scans) for o in s[1]]
    np.savez(os.environ["OBJS"], obj=np.array(rows, dtype=np.float64).reshape(-1, 8),
             scan=np.array([(s[0], s[2], s[3], s[4]) for s in scans]),
             odom=np.array([(stamp(o.header), o.twist.twist.linear.x) for o in m["/odom"]]))
if PERC:
    z = np.load(PERC)
    st, rd = z["stamp"], z["reading"]
    seen = rd[:, 0] > 0
    hx, hy = rd[:, 1] * 5.0, rd[:, 2] * 2.0                                   # LiDAR frame
    idx = np.searchsorted(ts, st).clip(1, len(ts) - 1)
    idx = np.where(np.abs(ts[idx - 1] - st) < np.abs(ts[idx] - st), idx - 1, idx)
    matched = np.abs(ts[idx] - st) < 0.02
    print(f"  all policy readings: 'car seen' {100 * np.mean(rd[:, 0] > 0):.1f} % of {len(st)}")
    sup = np.array([any(math.hypot(o[0] - hx[k], o[1] - hy[k]) < 0.6 for o in scans[idx[k]][1]) for k in range(len(st))])
    anyobj = np.array([len(scans[idx[k]][1]) > 0 for k in range(len(st))])
    sel = seen & matched
    near = sel & (hx > 0) & (hx < 3.0) & (np.abs(hy) < 0.6)
    print(f"policy readings {len(st)}, matched to a scan {int(matched.sum())}; 'car seen' {int(sel.sum())} "
          f"({100 * sel.sum() / max(matched.sum(), 1):.1f} %): an object candidate within 0.6 m of it {100 * sup[sel].mean():.1f} %, "
          f"any object in that scan {100 * anyobj[sel].mean():.1f} %")
    print(f"  of them 'within 3 m ahead in our lane' {int(near.sum())}: supported {100 * sup[near].mean() if near.any() else float('nan'):.1f} %")
    print(f"  where it says the car is (LiDAR frame, 'car seen'): dx p10/50/90 {np.percentile(hx[sel], [10, 50, 90]).round(2)}, "
          f"dy p10/50/90 {np.percentile(hy[sel], [10, 50, 90]).round(2)}; logit p50 {np.median(rd[sel, 0]):.2f}")
    np.savez(PERC.replace(".npz", "_match.npz"), stamp=st, reading=rd, supported=sup, matched=matched, any_object=anyobj,
             pose_x=np.interp(st, pt, px), pose_y=np.interp(st, pt, py), pose_yaw=np.interp(st, pt, pyaw))
