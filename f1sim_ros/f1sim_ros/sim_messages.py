"""Simulator state -> the exact messages the graph's sensor topics carry.

There are three places a simulator publishes car 0's sensors: `bridge_node` (the standalone
bridge), `eval_node` (a suite cell driven over the graph) and the console's own link
(`f1sim/viewer/ros_link.py`, which has its own copy for reasons of process boundaries). The first
two share this module.

They have to agree down to the field, because the whole claim of the split is that a node cannot
tell which of them it is attached to -- and because a scan stamped by one convention and one
stamped by another produce different plan-to-scan pairings in `controller_node`, which is the
thing `test_graph_parity.py` measures.

Pure functions of arrays and a stamp: no node, no simulator, no ROS graph. `test_sim_messages.py`
checks them, and `test_ros_link_builders.py` does the same job for the console's copy.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion(); q.z = math.sin(yaw / 2); q.w = math.cos(yaw / 2); return q


def rpy_to_quat(r: float, p: float, y: float) -> Quaternion:
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2),
                              math.sin(p / 2), math.cos(y / 2), math.sin(y / 2))
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy; q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy; q.z = cr * cp * sy - sr * sp * cy
    return q


def quat_to_yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def scan_sweep_seconds(meta: dict, n_ranges: int) -> float:
    """How far before the end of the control step the FIRST ray was traced.

    `LaserScan.header.stamp` is the acquisition time of the first ray, per the message definition,
    not of the scan's completion. The simulator traces the last beam at the end of the step
    (`Lidar.scan` anchors `time_frac` at 0 for the final beam), so the header goes back by one full
    sweep.

    The convention a particular real driver used when the raw bags were recorded is a separate,
    unvalidated question -- this only makes what we publish self-consistent with the metadata we
    publish beside it.
    """
    return float(meta["time_increment"]) * max(0, n_ranges - 1)


def scan_message(meta: dict, ranges: np.ndarray, stamp, frame_id: str = "laser") -> LaserScan:
    """`meta` is `Simulator.scan_meta()`; `stamp` is already the first-ray time."""
    m = LaserScan()
    m.header.stamp = stamp; m.header.frame_id = frame_id
    m.angle_min, m.angle_max = meta["angle_min"], meta["angle_max"]
    m.angle_increment = meta["angle_increment"]
    m.time_increment, m.scan_time = meta["time_increment"], meta["scan_time"]
    m.range_min, m.range_max = meta["range_min"], meta["range_max"]
    m.ranges = np.asarray(ranges, dtype=np.float32).tolist()
    return m


def odom_message(od: Sequence[float], stamp, odom_frame: str = "odom",
                 base_frame: str = "base_link") -> Odometry:
    """The VESC dead-reckoning: `od` is (x, y, yaw, wheel speed, yaw rate).

    `twist.twist.linear.x` is the ERPM WHEEL speed, not ground speed -- it is wrong about the
    vehicle under lock or spin, which is what makes it the traction guard's slip sensor -- and the
    pose drifts. Both are properties of the real car this reproduces.
    """
    o = Odometry()
    o.header.stamp = stamp; o.header.frame_id = odom_frame; o.child_frame_id = base_frame
    o.pose.pose.position.x, o.pose.pose.position.y = float(od[0]), float(od[1])
    o.pose.pose.orientation = yaw_to_quat(float(od[2]))
    o.twist.twist.linear.x, o.twist.twist.angular.z = float(od[3]), float(od[4])
    return o


def gt_odom_message(st: Sequence[float], stamp, map_frame: str = "map",
                    base_frame: str = "base_link") -> Odometry:
    """Ground truth in the map frame: `st` is (x, y, yaw, vx, vy, yaw rate).

    Simulation only. A policy or planner that consumes this will not transfer, which is why it is a
    separate topic with a separate name rather than a better `/odom`.
    """
    g = Odometry()
    g.header.stamp = stamp; g.header.frame_id = map_frame; g.child_frame_id = base_frame
    g.pose.pose.position.x, g.pose.pose.position.y = float(st[0]), float(st[1])
    g.pose.pose.orientation = yaw_to_quat(float(st[2]))
    g.twist.twist.linear.x, g.twist.twist.linear.y = float(st[3]), float(st[4])
    g.twist.twist.angular.z = float(st[5])
    return g


def _imu(samples_row, stamp, frame_id) -> Imu:
    m = Imu(); m.header.frame_id = frame_id; m.header.stamp = stamp
    g, a = samples_row[:3], samples_row[3:]
    m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (float(g[0]), float(g[1]),
                                                                        float(g[2]))
    m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (
        float(a[0]), float(a[1]), float(a[2]))
    return m


def imu_messages(samples: np.ndarray, att: np.ndarray, offsets: np.ndarray, stamp_at,
                 frame_id: str = "imu") -> Tuple[List[Imu], Imu]:
    """(the raw samples of this control step, the summary message).

    `samples` is (K, 6) gyro xyz then linear acceleration xyz, in SI. `offsets` is (K,) seconds
    before the END of the step each sample was taken -- the simulator reports the real offsets
    because the IMU runs on its own clock, so a 50 Hz sensor on a 40 Hz loop emits 1, 1, 1, 2 per
    step and an assumed spacing puts every sample but the occasional last one at a time it was not
    taken. `stamp_at(seconds_before_now)` turns one into a message stamp.

    `att` is one attitude estimate for the whole step, measured at its end, so only the LAST sample
    carries an orientation. Publishing it on the earlier samples would copy a future attitude
    backwards; publishing it on none would leave a consumer of this topic alone with no quaternion
    at all. The others are marked `orientation_covariance[0] = -1`, which is how a ROS publisher
    states it has no orientation estimate -- and which `deploy.attitude_from_orientation` honours.
    """
    samples = np.asarray(samples)
    K = samples.shape[0]
    raw = []
    for k in range(K):
        m = _imu(samples[k], stamp_at(float(offsets[k])), frame_id)
        if k == K - 1:
            m.orientation = rpy_to_quat(float(att[0]), float(att[1]), float(att[2]))
        else:
            cov = list(m.orientation_covariance); cov[0] = -1.0
            m.orientation_covariance = cov
        raw.append(m)
    summary = _imu(samples[-1], stamp_at(float(offsets[-1])), frame_id)
    summary.orientation = rpy_to_quat(float(att[0]), float(att[1]), float(att[2]))
    return raw, summary
