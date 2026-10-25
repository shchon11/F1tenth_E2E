"""Dump a venue bag's IMU gyro z and /odom twist to an .npz (t, gyro_z) / (t, v, w), for joining with the plan rows.
usage (ROS sourced): real_bag_gyro.py BAG OUT.npz"""
import sys
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
BAG, OUT = sys.argv[1:3]
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
r.set_filter(rosbag2_py.StorageFilter(topics=["/odom", "/sensors/imu/raw"]))
imu, od = [], []
while r.has_next():
    t, data, _ = r.read_next()
    m = deserialize_message(data, get_message(types[t]))
    s = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
    if t == "/odom":
        od.append((s, m.twist.twist.linear.x, m.twist.twist.angular.z))
    else:
        imu.append((s, m.angular_velocity.z))
np.savez(OUT, imu=np.array(imu), odom=np.array(od))
print("saved", OUT, len(imu), len(od))
