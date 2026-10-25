"""Dump a venue bag's localisation trajectory (/pf/pose/odom: t, x, y, yaw) and /odom (t, v, yaw rate) to an .npz.
usage (ROS sourced): real_bag_traj.py BAG OUT.npz"""
import math
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BAG, OUT = sys.argv[1:3]
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
r.set_filter(rosbag2_py.StorageFilter(topics=["/pf/pose/odom", "/odom"]))
pf, od = [], []
while r.has_next():
    t, data, _ = r.read_next()
    m = deserialize_message(data, get_message(types[t]))
    s = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
    if t == "/odom":
        od.append((s, m.twist.twist.linear.x, m.twist.twist.angular.z))
    else:
        q = m.pose.pose.orientation
        pf.append((s, m.pose.pose.position.x, m.pose.pose.position.y,
                   math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))))
pf = np.array(pf); pf[:, 3] = np.unwrap(pf[:, 3])
np.savez(OUT, pf=pf, odom=np.array(od))
print("saved", OUT, len(pf), len(od))
