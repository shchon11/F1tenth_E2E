"""Research (2026-10-26): which part of the real scan's context makes our policy treat a box like a car? IDEALISE the walls.

The transplant (real_bag_box_transplant.py) showed the box's own look is not it: an ideal sim-like box put into the real scans
is still read as a car as often (24-57 %) and crossed 34 % at 1.5-2.5 m (sim 4-6 %). Here every well-localised scan keeps the
real trajectory, timing and ego inputs and its real non-wall returns (other cars, people), but
  * every return the map explains (a wall), and every beam with no return, is replaced by the map's own ray-cast range plus
    the simulator's 1 cm noise -- ideal walls, no dropouts;
  * each fixed box becomes the ideal 0.30 x 0.30 m square (as in the transplant).
Replayed through the policy node: if the plans now clear the boxes like the sim, the real WALL TEXTURE (noise, dropouts,
reflections, shape) is the context; if they still cross them, it is the ego inputs / trajectory / the other cars.
usage (ROS sourced): real_bag_idealize.py BAG OBJS.npz OUT_BAG_DIR"""
import math
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.ndimage import distance_transform_edt

BAG, OBJS, OUT = sys.argv[1:4]
TOL, HALF, BACK, LX = 0.25, 0.15, 0.12, 0.297
import os
FULL = bool(os.environ.get("FULL"))
zo = np.load(OBJS); obj, scan = zo["obj"], zo["scan"]
cls = np.load(OBJS.replace(".npz", "_cls.npy"))
pose = {round(float(s[0]), 4): (s[1], s[2], s[3]) for s in scan}
boxes = {}
for n in np.where(cls == 2)[0]:
    boxes.setdefault(round(float(obj[n, 1]), 4), []).append((float(obj[n, 2]), float(obj[n, 3])))
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
topics = r.get_all_topics_and_types()
types = {t.name: t.type for t in topics}
r.set_filter(rosbag2_py.StorageFilter(topics=["/map"]))
grid = None
while r.has_next():
    _, data, _ = r.read_next()
    grid = deserialize_message(data, get_message(types["/map"]))
res, W, H = grid.info.resolution, grid.info.width, grid.info.height
ox, oy = grid.info.origin.position.x, grid.info.origin.position.y
occ = np.asarray(grid.data, dtype=np.int16).reshape(H, W) >= 50
dist = distance_transform_edt(~occ) * res


def field(px, py):
    i, j = np.floor((py - oy) / res).astype(int), np.floor((px - ox) / res).astype(int)
    inside = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    return np.where(inside, dist[i.clip(0, H - 1), j.clip(0, W - 1)], np.inf), inside


def ideal(bx, by, ang, rmax):
    c, s = np.cos(ang), np.sin(ang)
    t = np.zeros(len(ang)); done = np.zeros(len(ang), bool); hit = np.zeros(len(ang), bool)
    for _ in range(500):
        d, inside = field(bx + t * c, by + t * s)
        new = ~done & (d < res * 0.6)
        hit |= new; done |= new | (t > rmax) | ~inside
        if done.all():
            break
        t = np.where(done, t, t + np.maximum(d - res * 0.5, res * 0.5))
    return np.where(hit, t, np.inf)


rng = np.random.default_rng(0)
w = rosbag2_py.SequentialWriter()
w.open(rosbag2_py.StorageOptions(uri=OUT, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
for t in topics:
    w.create_topic(rosbag2_py.TopicMetadata(name=t.name, type=t.type, serialization_format="cdr"))
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
n_scans = n_ideal = n_kept_real = n_box = 0
while r.has_next():
    t, data, ts = r.read_next()
    if t == "/scan":
        n_scans += 1
        m = deserialize_message(data, get_message(types[t]))
        st = round(m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, 4)
        if st in pose:
            x, y, yaw = pose[st]
            rr = np.asarray(m.ranges, dtype=np.float64)
            th = m.angle_min + m.angle_increment * np.arange(len(rr))
            bx, by = x + math.cos(yaw) * LX, y + math.sin(yaw) * LX
            rmax = min(float(m.range_max), 12.0)
            ri = ideal(bx, by, yaw + th, rmax)
            ok = np.isfinite(rr) & (rr > 0.05) & (rr < rmax)
            hx, hy = bx + rr * np.cos(yaw + th), by + rr * np.sin(yaw + th)
            dmap, _ = field(np.where(ok, hx, bx), np.where(ok, hy, by))
            wall = ~ok | (dmap <= TOL)
            if FULL:
                wall = np.ones_like(wall)          # FULL=1: every beam ray-cast -- no real return of any kind is left
            # the fixed boxes' own beams go to the background as well (they get the ideal square below)
            px, py = rr * np.cos(th), rr * np.sin(th)
            for (cx, cy) in boxes.get(st, []):
                wall |= ok & (np.hypot(px - cx, py - cy) < 0.45)
            out = np.where(wall, ri + rng.normal(0.0, 0.01, len(rr)), rr)
            n_kept_real += int((~wall).sum())
            for (cx, cy) in boxes.get(st, []):
                d = math.hypot(cx, cy)
                bcx, bcy = cx + BACK * cx / d, cy + BACK * cy / d
                lo_x, hi_x, lo_y, hi_y = bcx - HALF, bcx + HALF, bcy - HALF, bcy + HALF
                cdx, cdy = np.cos(th), np.sin(th)
                with np.errstate(divide="ignore", invalid="ignore"):
                    tx1, tx2 = lo_x / cdx, hi_x / cdx
                    ty1, ty2 = lo_y / cdy, hi_y / cdy
                t0 = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
                t1 = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
                hitb = (t1 >= t0) & (t0 > 0.05) & np.isfinite(t0) & (t0 < out)
                out = np.where(hitb, t0 + rng.normal(0.0, 0.01, len(rr)), out)
                n_box += int(hitb.sum())
            m.ranges = [float(v) if np.isfinite(v) else float("inf") for v in out]
            data = serialize_message(m)
            n_ideal += 1
    w.write(t, data, ts)
del w
print(f"{n_scans} scans, {n_ideal} idealised (walls and dropouts ray-cast, {n_kept_real / max(n_ideal, 1):.1f} real non-wall beams kept "
      f"per scan, {n_box} beams on ideal boxes) -> {OUT}")
