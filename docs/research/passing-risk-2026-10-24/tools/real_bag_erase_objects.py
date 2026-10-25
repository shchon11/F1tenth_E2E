"""Research (2026-10-25): does our policy see the real fixed objects at all? Write a copy of a venue bag whose /scan has
them ERASED (each beam of a static object candidate gets the range the map says it would have without the object, by
ray-marching the occupancy grid from the LiDAR), everything else byte for byte. Replaying both bags through the policy
node (replay_bag_policy.sh) and comparing the plans at the same scans answers it: unchanged plans = the policy never
reacted to the object.

Static candidates come from real_bag_head_check.py (OBJS.npz + its _cls.npy: 2 = static), so only well-localised scans
are edited (the pose places the ray); other scans pass unchanged.
usage (ROS sourced): real_bag_erase_objects.py BAG OBJS.npz OUT_BAG_DIR [CLASSES=2]"""
import math
import os
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.ndimage import distance_transform_edt

BAG, OBJS, OUT = sys.argv[1:4]
CLASSES = {int(c) for c in (sys.argv[4] if len(sys.argv) > 4 else "2").split(",")}
TOL = 0.25
zo = np.load(OBJS)
obj, scan = zo["obj"], zo["scan"]
cls = np.load(OBJS.replace(".npz", "_cls.npy"))
keep = np.isin(cls, list(CLASSES))
want = {}                                                       # scan stamp (rounded) -> list of LiDAR-frame candidate centres
for n in np.where(keep)[0]:
    want.setdefault(round(float(obj[n, 1]), 4), []).append((obj[n, 2], obj[n, 3]))
pose = {round(float(s[0]), 4): (s[1], s[2], s[3]) for s in scan}

r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
topics = r.get_all_topics_and_types()
types = {t.name: t.type for t in topics}
grid = None
r.set_filter(rosbag2_py.StorageFilter(topics=["/map", "/tf_static"]))
lx = 0.297
while r.has_next():
    t, data, _ = r.read_next()
    m = deserialize_message(data, get_message(types[t]))
    if t == "/map":
        grid = m
res, W, H = grid.info.resolution, grid.info.width, grid.info.height
ox, oy = grid.info.origin.position.x, grid.info.origin.position.y
occ = np.asarray(grid.data, dtype=np.int16).reshape(H, W) >= 50
dist = distance_transform_edt(~occ) * res


def map_d(wx, wy):
    i, j = int(math.floor((wy - oy) / res)), int(math.floor((wx - ox) / res))
    return dist[i, j] if 0 <= i < H and 0 <= j < W else 0.0


def march(x0, y0, ang, rmax):
    """distance along the ray to the first occupied cell (sphere tracing on the distance field)"""
    s = 0.0
    while s < rmax:
        d = map_d(x0 + s * math.cos(ang), y0 + s * math.sin(ang))
        if d < res * 0.6:
            return s
        s += max(d - res * 0.5, res * 0.5)
    return float("inf")


w = rosbag2_py.SequentialWriter()
w.open(rosbag2_py.StorageOptions(uri=OUT, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
for t in topics:
    w.create_topic(rosbag2_py.TopicMetadata(name=t.name, type=t.type, serialization_format="cdr"))
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
n_scans = n_edit = n_beams = 0
while r.has_next():
    t, data, ts = r.read_next()
    if t == "/scan":
        n_scans += 1
        m = deserialize_message(data, get_message(types[t]))
        st = round(m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, 4)
        if st in want and st in pose:
            x, y, yaw = pose[st]
            rr = np.asarray(m.ranges, dtype=np.float64)
            th = m.angle_min + m.angle_increment * np.arange(len(rr))
            ok = np.isfinite(rr) & (rr > 0.05) & (rr < 12.0)
            bx, by = x + math.cos(yaw) * lx, y + math.sin(yaw) * lx      # the LiDAR on the map
            changed = 0
            for (cx, cy) in want[st]:
                # the candidate's beams: unexplained, within 0.45 m of its centre
                px, py = rr * np.cos(th), rr * np.sin(th)
                sel = ok & (np.hypot(px - cx, py - cy) < 0.45)
                for b in np.where(sel)[0]:
                    wx, wy = bx + rr[b] * math.cos(yaw + th[b]), by + rr[b] * math.sin(yaw + th[b])
                    if map_d(wx, wy) <= TOL:
                        continue                                 # a wall point next to it: leave it
                    behind = march(bx, by, yaw + th[b], min(float(m.range_max), 12.0))
                    rr[b] = behind if math.isfinite(behind) else float("inf")
                    changed += 1
            if changed:
                m.ranges = [float(v) for v in rr]
                data = serialize_message(m)
                n_edit += 1; n_beams += changed
    w.write(t, data, ts)
del w
print(f"{n_scans} scans, {n_edit} edited ({n_beams} beams, {n_beams / max(n_edit, 1):.1f} per edited scan) -> {OUT}")
