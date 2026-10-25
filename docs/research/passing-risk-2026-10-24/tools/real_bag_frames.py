"""Draw a few real scans in the car's frame: points the map explains (grey), the rest (red), the policy's nearest-car
reading (blue cross + its 0.6 m circle). Picks scans where the reading sits on a MOVING object and on a STATIC one; UNSUPPORTED=1 picks 'car seen' readings with no
object within 0.6 m instead (the false alarms).
usage (ROS sourced): real_bag_frames.py BAG OBJS.npz PERCEPTION.npz OUT.png"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.ndimage import distance_transform_edt

BAG, OBJS, PERC, OUT = sys.argv[1:5]
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
r.set_filter(rosbag2_py.StorageFilter(topics=["/scan", "/map"]))
scans = {}; grid = None
while r.has_next():
    t, data, _ = r.read_next()
    msg = deserialize_message(data, get_message(types[t]))
    if t == "/map":
        grid = msg
    else:
        scans[round(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec, 4)] = msg
res, W, H = grid.info.resolution, grid.info.width, grid.info.height
ox, oy = grid.info.origin.position.x, grid.info.origin.position.y
dist = distance_transform_edt(~(np.asarray(grid.data, dtype=np.int16).reshape(H, W) >= 50)) * res
zo, zp = np.load(OBJS), np.load(PERC)
obj, sc = zo["obj"], zo["scan"]
cls = np.load(OBJS.replace(".npz", "_cls.npy"))                 # real_bag_head_check.py: 1 moving, 2 static, 0 other
st, rd = zp["stamp"], zp["reading"]
keys = np.array(sorted(scans))


def kind_at(k_scan, x, y):
    best = None
    for n in np.where(obj[:, 0] == k_scan)[0]:
        d = math.hypot(obj[n, 2] - x, obj[n, 3] - y)
        if d < 0.6 and (best is None or d < best[0]):
            best = (d, n)
    return best


import os
picks = {"moving": [], "static": [], "none": []}
for i in range(0, len(st), 7):
    if rd[i, 0] <= 0:
        continue
    k = int(np.argmin(np.abs(sc[:, 0] - st[i])))
    if abs(sc[k, 0] - st[i]) > 0.02:
        continue
    b = kind_at(k, rd[i, 1] * 5, rd[i, 2] * 2)
    if b is None:
        picks["none"].append((i, k))
        continue
    if cls[b[1]] == 2:
        picks["static"].append((i, k))
    elif cls[b[1]] == 1 and 0.5 < rd[i, 1] * 5 < 3.5:
        picks["moving"].append((i, k))
sel = [picks["none"][int(q * (len(picks["none"]) - 1))] for q in (0.1, 0.4, 0.7, 0.95)] if os.environ.get("UNSUPPORTED") and picks["none"] else picks["moving"][len(picks["moving"]) // 3: len(picks["moving"]) // 3 + 1] + picks["moving"][2 * len(picks["moving"]) // 3:][:1] \
    + picks["static"][len(picks["static"]) // 4:][:1] + picks["static"][3 * len(picks["static"]) // 4:][:1]
fig, axs = plt.subplots(1, len(sel), figsize=(5 * len(sel), 5.5))
for a, (i, k) in zip(np.atleast_1d(axs), sel):
    m = scans[keys[np.argmin(np.abs(keys - st[i]))]]
    x, y, yaw = sc[k, 1], sc[k, 2], sc[k, 3]
    rr = np.asarray(m.ranges); th = m.angle_min + m.angle_increment * np.arange(len(rr))
    ok = np.isfinite(rr) & (rr > 0.05) & (rr < 12)
    px, py = rr * np.cos(th), rr * np.sin(th)
    wx = x + math.cos(yaw) * (px + 0.297) - math.sin(yaw) * py; wy = y + math.sin(yaw) * (px + 0.297) + math.cos(yaw) * py
    gi, gj = np.floor((wy - oy) / res).astype(int).clip(0, H - 1), np.floor((wx - ox) / res).astype(int).clip(0, W - 1)
    un = ok & (dist[gi, gj] > 0.25)
    a.scatter(-py[ok & ~un], px[ok & ~un], s=1, c="0.6"); a.scatter(-py[un], px[un], s=3, c="tab:red")
    hx, hy = rd[i, 1] * 5, rd[i, 2] * 2
    a.plot(-hy, hx, "x", c="tab:blue", ms=14, mew=3); a.add_patch(plt.Circle((-hy, hx), 0.6, fill=False, color="tab:blue"))
    a.plot(0, 0, "^", c="k", ms=10)
    tag = "moving car" if (i, k) in picks["moving"] else "static object" if (i, k) in picks["static"] else "NOTHING within 0.6 m"
    a.set_title(f"t={st[i] - st[0]:.1f} s, {tag}\nhead: dx {hx:.2f} dy {hy:+.2f} closing {rd[i, 3] * 5:+.2f} v_lat {rd[i, 4] * 2:+.2f}", fontsize=9)
    a.set_xlim(-3, 3); a.set_ylim(-1, 6); a.set_aspect("equal"); a.set_xlabel("left <-  y [m]  -> right (mirrored)"); a.set_ylabel("ahead x [m]")
plt.tight_layout(); plt.savefig(OUT, dpi=100)
print("saved", OUT, {k: len(v) for k, v in picks.items()})
