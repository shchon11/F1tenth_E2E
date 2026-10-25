"""Draw real scans (car frame) with our policy's plan path (blue), the path the car really drove (green), the static object
(red circle, 0.15 m) -- picks from real_bag_plan_vs_objects.py's OUT.npz: plans that run into the object and plans that clear it.
usage (ROS sourced): real_bag_plan_frames.py BAG PLANOBJ.npz OUT.png"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BAG, PO, OUT = sys.argv[1:4]
z = np.load(PO); rows, paths, drv = z["rows"], z["paths"], z["driven"]
into = np.where(rows[:, 5] < 0)[0]; clear = np.where(rows[:, 5] > 0.05)[0]
pick = [into[int(q * (len(into) - 1))] for q in (0.15, 0.5, 0.85)] if len(into) else []
pick += [clear[int(q * (len(clear) - 1))] for q in (0.3, 0.7)] if len(clear) else []
want = {round(float(rows[i, 0]), 3): i for i in pick}
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
r.set_filter(rosbag2_py.StorageFilter(topics=["/scan"]))
got = {}
while r.has_next():
    _, data, _ = r.read_next()
    m = deserialize_message(data, get_message(types["/scan"]))
    t = round(m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, 3)
    for tw, i in want.items():
        if abs(t - tw) < 0.002 and i not in got:
            got[i] = m
fig, axs = plt.subplots(1, len(pick), figsize=(4.6 * len(pick), 5.6))
for a, i in zip(np.atleast_1d(axs), pick):
    m = got.get(i)
    if m is not None:
        rr = np.asarray(m.ranges); th = m.angle_min + m.angle_increment * np.arange(len(rr))
        ok = np.isfinite(rr) & (rr > 0.05) & (rr < 8)
        a.scatter(-(rr * np.sin(th))[ok], (rr * np.cos(th))[ok] + 0.297, s=1, c="0.55")
    p = paths[i]; d = drv[i]
    a.plot(-p[:, 1], p[:, 0], "-", c="tab:blue", lw=2.5, label="our plan")
    a.plot(-d[:, 1], d[:, 0], "-", c="tab:green", lw=1.5, label="driven (ForzaETH)")
    a.add_patch(plt.Circle((-rows[i, 2], rows[i, 1]), 0.15, color="tab:red", alpha=0.6))
    a.plot(0, 0, "^", c="k", ms=9)
    a.set_title(f"t={rows[i, 0] - rows[0, 0]:.1f} s, car {rows[i, 3]:.1f} m/s, plan v1 {rows[i, 7]:.1f}\n"
                f"clearance plan {rows[i, 5]:+.2f} m, driven {rows[i, 6]:+.2f} m", fontsize=9)
    a.set_xlim(-2.5, 2.5); a.set_ylim(-0.5, 4.5); a.set_aspect("equal"); a.set_xlabel("left <- y [m] -> right")
np.atleast_1d(axs)[0].legend(loc="lower left", fontsize=8)
plt.tight_layout(); plt.savefig(OUT, dpi=100)
print("saved", OUT, len(pick))
