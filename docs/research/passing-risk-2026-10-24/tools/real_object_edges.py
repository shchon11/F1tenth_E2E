"""Research (2026-10-25): the LiDAR signature of the real fixed objects and the other cars on the venue bags, against what the
simulator's scanner can produce. The sim (`lidar.Lidar._post`) adds range noise, spikes and dropouts but no MIXED PIXELS: a
beam that grazes an object's edge returns a range between the object and what is behind it. For every fixed object / moving
car 0.8-3.5 m ahead within 0.8 m of our heading (real_bag_head_check.py classes), on its LiDAR beams: the angular width,
physical width, beams, depth (range spread over its beams), and at each edge, among the 3 beams beyond it, how many sit
BETWEEN the object's edge range and the background behind (+0.15 / -0.15 m): the veil a mixed pixel leaves.
usage (ROS sourced): real_object_edges.py BAG OBJS.npz"""
import math
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BAG, OBJS = sys.argv[1:3]
zo = np.load(OBJS); obj = zo["obj"]
cls = np.load(OBJS.replace(".npz", "_cls.npy"))
want = {}
for n in range(len(obj)):
    ax, ay = obj[n, 2], obj[n, 3]
    if cls[n] in (1, 2) and 0.8 < ax < 3.5 and abs(ay) < 0.8:
        want.setdefault(round(float(obj[n, 1]), 4), []).append(n)
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
r.set_filter(rosbag2_py.StorageFilter(topics=["/scan"]))
rows = {1: [], 2: []}
while r.has_next():
    _, data, _ = r.read_next()
    m = deserialize_message(data, get_message(types["/scan"]))
    t = round(m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, 4)
    if t not in want:
        continue
    rr = np.asarray(m.ranges, dtype=np.float64)
    th = m.angle_min + m.angle_increment * np.arange(len(rr))
    px, py = rr * np.cos(th), rr * np.sin(th)
    for n in want[t]:
        ax, ay = obj[n, 2], obj[n, 3]
        sel = np.where(np.isfinite(rr) & (np.hypot(px - ax, py - ay) < 0.45))[0]
        if len(sel) < 3:
            continue
        i0, i1 = sel.min(), sel.max()
        ro = rr[i0:i1 + 1]; ro = ro[np.isfinite(ro)]
        width = float(np.hypot(px[i1] - px[i0], py[i1] - py[i0]))
        depth = float(np.nanmax(ro) - np.nanmin(ro))
        veil = 0; edges = 0
        for edge, step in ((i0, -1), (i1, 1)):
            far = [rr[edge + step * j] for j in range(5, 10) if 0 <= edge + step * j < len(rr) and np.isfinite(rr[edge + step * j])]
            if not far:
                continue
            r_bg, r_e = float(np.median(far)), float(rr[edge])
            if r_bg < r_e + 0.5:
                continue                              # nothing far behind this edge: no veil to see
            edges += 1
            for j in range(1, 4):
                b = edge + step * j
                if 0 <= b < len(rr) and np.isfinite(rr[b]) and r_e + 0.15 < rr[b] < r_bg - 0.15:
                    veil += 1
        rows[int(cls[n])].append((len(sel), width, depth, veil, edges, float(np.hypot(ax, ay))))
for c, tag in ((2, "fixed objects"), (1, "moving cars")):
    a = np.array(rows[c]).reshape(-1, 6)
    if not len(a):
        continue
    e = a[:, 4] > 0
    print(f"{tag}: n {len(a)} | beams p50 {np.median(a[:, 0]):.0f} | width p50 {np.median(a[:, 1]):.2f} m | depth p50 {np.median(a[:, 2]):.2f} m | "
          f"edges with a background behind {int(a[:, 4].sum())}: veil beams per such edge {a[e, 3].sum() / max(a[e, 4].sum(), 1):.2f}, "
          f"objects with >= 1 veil beam {100 * np.mean(a[e, 3] > 0):.0f} % | range p50 {np.median(a[:, 5]):.2f} m")
