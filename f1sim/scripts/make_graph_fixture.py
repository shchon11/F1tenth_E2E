#!/usr/bin/env python3
"""Turn a rosbag2 into the frame sequence the graph parity tests replay.

    python3 f1sim/scripts/make_graph_fixture.py <bag> --out f1sim/tests/data/graph_seq_sim.npz \
        --frames 120

The fixture is the messages as they were on the wire -- `LaserScan.ranges` in metres,
`Imu.linear_acceleration` in whatever unit the driver published (g on the car, m/s^2 in the
simulator), the quaternion and `orientation_covariance[0]` exactly as sent, `Odometry`'s wheel
speed -- grouped into the frames a node would see: for each scan, the IMU and odometry messages
that arrived since the previous one.

Deliberately NOT `calib/bagread.py`: that converts the accelerometer to SI on the way in, and the
thing being tested is the node's own unit detection. `learn/bagdata.py` is the one that goes
through bagread.

The arrays are small on purpose (120 frames of 1081 beams at float32 is ~500 KB) because they are
committed: the parity claim has to be reproducible without the 22 GB of recordings.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

SCAN = "/scan"
IMU = "/sensors/imu/raw"
ODOM = "/odom"


def read_records(path, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in types]
    if missing:
        raise SystemExit(f"{path} has no {missing}; it carries {sorted(types)}")
    cls = {t: get_message(types[t]) for t in topics}
    out = []
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if topic in cls:
            out.append((int(stamp), topic, deserialize_message(raw, cls[topic])))
    out.sort(key=lambda r: r[0])
    return out


def build(records, frames: int, skip: int, min_speed: float):
    scans, imus, odoms = [], [], []
    imu_ptr, odom_ptr = [0], [0]
    t_scan = []
    pending_imu, pending_odom = [], []
    n_skipped = 0
    for _stamp, topic, m in records:
        if topic == IMU:
            w, a = m.angular_velocity, m.linear_acceleration
            q = m.orientation
            cov = m.orientation_covariance
            pending_imu.append(([w.x, w.y, w.z, a.x, a.y, a.z],
                                [q.w, q.x, q.y, q.z],
                                float(cov[0]) if len(cov) else 0.0))
        elif topic == ODOM:
            pending_odom.append(float(m.twist.twist.linear.x))
        elif topic == SCAN:
            if not pending_imu or not pending_odom:
                pending_imu, pending_odom = [], []          # an incomplete frame is not a frame
                continue
            if n_skipped < skip or (min_speed > 0 and abs(pending_odom[-1]) < min_speed):
                n_skipped += 1
                pending_imu, pending_odom = [], []
                continue
            scans.append((np.asarray(m.ranges, dtype=np.float32), float(m.range_max),
                          float(m.angle_min), float(m.angle_max)))
            t_scan.append(_stamp)
            imus.extend(pending_imu); imu_ptr.append(len(imus))
            odoms.extend(pending_odom); odom_ptr.append(len(odoms))
            pending_imu, pending_odom = [], []
            if len(scans) >= frames:
                break
    if len(scans) < frames:
        raise SystemExit(f"only {len(scans)} usable frames (wanted {frames}): the bag is short, or "
                         f"--min-speed excluded most of it")
    widths = {len(s[0]) for s in scans}
    if len(widths) != 1:
        raise SystemExit(f"the scan width changes inside this bag: {sorted(widths)}")
    t = np.asarray(t_scan, dtype=np.int64)
    return {
        "ranges": np.stack([s[0] for s in scans]),
        "range_max": np.float32(scans[0][1]),
        "angle_min": np.float32(scans[0][2]),
        "angle_max": np.float32(scans[0][3]),
        "imu": np.asarray([r[0] for r in imus], dtype=np.float32),
        "imu_quat": np.asarray([r[1] for r in imus], dtype=np.float32),
        "imu_cov0": np.asarray([r[2] for r in imus], dtype=np.float32),
        "imu_ptr": np.asarray(imu_ptr, dtype=np.int32),
        "odom_v": np.asarray(odoms, dtype=np.float32),
        "odom_ptr": np.asarray(odom_ptr, dtype=np.int32),
        "t_scan": (t - t[0]).astype(np.float64) * 1e-9,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--skip", type=int, default=0, help="usable frames to drop from the start")
    ap.add_argument("--min-speed", type=float, default=0.0,
                    help="only take frames where the wheel speed is at least this [m/s]")
    a = ap.parse_args()
    d = build(read_records(a.bag, (SCAN, IMU, ODOM)), a.frames, a.skip, a.min_speed)
    d["source"] = np.asarray(os.path.basename(a.bag.rstrip("/")))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez_compressed(a.out, **d)
    n = d["ranges"].shape
    print(f"{a.out}: {n[0]} frames x {n[1]} beams, {d['imu'].shape[0]} IMU samples, "
          f"{d['odom_v'].shape[0]} odom, {d['t_scan'][-1]:.1f} s, "
          f"|a| first = {np.linalg.norm(d['imu'][0, 3:]):.2f}, "
          f"{os.path.getsize(a.out) / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
