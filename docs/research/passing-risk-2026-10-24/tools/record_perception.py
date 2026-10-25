"""Record the policy node's /f1sim/policy/perception (scan stamp + nearest-car reading) to an .npz while a bag plays.
usage (ROS sourced): record_perception.py OUT.npz [SECONDS=180]"""
import sys
import time

import numpy as np
import rclpy
from f1sim_interfaces.msg import Perception

OUT = sys.argv[1]
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
rclpy.init()
node = rclpy.create_node("record_perception")
rows = []
node.create_subscription(Perception, "/f1sim/policy/perception",
                         lambda m: rows.append([m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, *(list(m.nearest_car) or [np.nan] * 5)]), 100)
t0 = time.time()
while time.time() - t0 < DUR:
    rclpy.spin_once(node, timeout_sec=0.1)
a = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
np.savez(OUT, stamp=a[:, 0], reading=a[:, 1:])
print(f"recorded {len(a)} readings -> {OUT}", flush=True)
