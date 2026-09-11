"""Read an F1TENTH rosbag2 (sqlite3) into plain numpy arrays.

Conventions the recordings impose, taken from the dataset README and re-checked here:
  * `/sensors/imu/raw` linear_acceleration is in **g**, not m/s^2 (az ~ 1.00 at rest).
  * `/odom` twist is VESC ERPM wheel speed, not ground speed: it lies under wheel spin or lock-up,
    and differentiating it produces physically impossible spikes.
  * Some recordings carry a raw gyro whose |gz| p99 is 49-255 rad/s against a normal 3-4. Do NOT
    read that as established hardware corruption: a unit mismatch would look the same, and two of
    seven bags cross-checked independently are consistent with one. **Validate the gyro's units per
    bag** before using it; a large magnitude on its own proves nothing either way.
  * Whatever the gyro turns out to be, `/odom` twist.angular.z is **not an independent fallback**.
    Command reconstruction was confirmed on seven pre-competition bags -- there the channel restates
    the commanded steering angle and the ERPM wheel speed rather than measuring what the car did.
    The producer of the competition bags has not been verified, so the mechanism is not established
    for those, which is a reason for caution rather than permission. Substituting this channel for a
    questionable gyro turns a measurement into an assumption in either case.

`t` is relative to the smallest record timestamp in the whole bag -- see `read` and
`BagData.origin_record_ns`. It is NOT relative to the first record of the topics requested, which is
what earlier versions returned; values from a subset read therefore differ from those.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

G = 9.80665


@dataclass
class BagData:
    name: str
    t: Dict[str, np.ndarray] = field(default_factory=dict)      # topic -> timestamps [s], bag clock
    v: Dict[str, np.ndarray] = field(default_factory=dict)      # topic -> value array
    #: The record timestamp [ns] that `t` is measured from: the smallest one in the WHOLE bag,
    #: independent of which topics were asked for. Provenance, so two reads can be shown to share a
    #: clock. None for a bag with no records.
    origin_record_ns: Optional[int] = None

    def has(self, *topics) -> bool:
        return all(k in self.t and len(self.t[k]) for k in topics)

    def at(self, topic: str, times: np.ndarray) -> np.ndarray:
        """Sample a topic onto `times` by linear interpolation (nearest at the edges)."""
        tt, vv = self.t[topic], self.v[topic]
        if vv.ndim == 1:
            return np.interp(times, tt, vv)
        return np.stack([np.interp(times, tt, vv[:, i]) for i in range(vv.shape[1])], 1)


WANTED = {
    "/drive": "ackermann", "/ackermann_cmd": "ackermann", "/teleop": "ackermann",
    "/odom": "odom", "/pf/pose/odom": "odom", "/car_state/odom": "odom",
    "/sensors/imu/raw": "imu",
    "/sensors/core": "vesc",
    "/sensors/servo_position_command": "float",
    "/imu/filtered_angular_velocity": "float",
    "/commands/motor/speed": "float",
    "/scan": "scan",
    "/state": "string",                       # the stack's mode: manual vs autonomous
    "/joy": "joy",
}


def read(path: str, topics: Optional[List[str]] = None, want_scan: bool = False) -> BagData:
    """Decode the topics we identify vehicle parameters from. `want_scan` also keeps LiDAR ranges."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    keep = set(topics) if topics else {k for k in WANTED if k in types}
    if not want_scan:
        keep.discard("/scan")
    acc: Dict[str, list] = {k: [] for k in keep}
    stamps: Dict[str, list] = {k: [] for k in keep}
    cls: Dict[str, object] = {}
    origin_ns: Optional[int] = None
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        # The origin is a property of the BAG, not of the subset being decoded, and it is collected
        # before the topic filter for exactly that reason: taking it from the selected topics made
        # `read(bag, ["/odom"])` and `read(bag, ["/odom", "/drive"])` return different `t` for
        # /odom -- by up to seconds, since /drive starts 1.96 s in -- so results from separate reads
        # could not be combined. `min` over every record rather than the first one also survives a
        # bag whose records are not written in timestamp order.
        stamp = int(stamp)
        origin_ns = stamp if origin_ns is None else min(origin_ns, stamp)
        if topic not in keep:
            continue
        if topic not in cls:
            cls[topic] = get_message(types[topic])
        m = deserialize_message(raw, cls[topic])
        kind = WANTED.get(topic, "float")
        if kind == "ackermann":
            d = m.drive
            val = [d.steering_angle, d.speed, d.acceleration, d.steering_angle_velocity]
        elif kind == "odom":
            p, q = m.pose.pose.position, m.pose.pose.orientation
            tw = m.twist.twist
            yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
            val = [p.x, p.y, yaw, tw.linear.x, tw.linear.y, tw.angular.z]
        elif kind == "imu":
            a, w = m.linear_acceleration, m.angular_velocity
            # stored in g; converted here so every consumer sees m/s^2
            val = [w.x, w.y, w.z, a.x * G, a.y * G, a.z * G]
        elif kind == "vesc":
            s = m.state
            val = [s.current_motor, s.current_input, s.duty_cycle, s.speed, s.voltage_input]
        elif kind == "scan":
            val = np.asarray(m.ranges, dtype=np.float32)
        elif kind == "string":
            acc[topic].append(m.data); stamps[topic].append(stamp); continue
        elif kind == "joy":
            val = list(m.axes) + [float(b) for b in m.buttons]
        else:
            val = [float(m.data)]
        acc[topic].append(val)
        stamps[topic].append(stamp)
    out = BagData(name=os.path.basename(path.rstrip("/")), origin_record_ns=origin_ns)
    # ONE origin for the whole bag. Zeroing each topic against its own first message destroys the
    # very alignment every lag estimate depends on: in these recordings /sensors/imu/raw starts
    # 194 ms after /odom and /drive starts 1.96 s after it, so per-topic zeroing injects an offset
    # the same size as the command delay being measured.
    for k in keep:
        if not stamps[k]:
            continue
        # Subtract in integer nanoseconds, then convert. Going to float seconds first loses the low
        # bits to the epoch: float64 spacing at a 2025 timestamp in ns is 256 ns, so differences
        # under a microsecond -- which is the resolution these lag estimates are quoted at -- came
        # out quantised.
        ns = np.asarray(stamps[k], dtype=np.int64)
        # `at()` interpolates with np.interp, which requires increasing x and returns nonsense
        # rather than an error if it is not. Records are normally written in order, but nothing in
        # the format guarantees it, so sort each topic (stable, so equal stamps keep writer order)
        # and carry the values with it.
        if np.any(np.diff(ns) < 0):
            order = np.argsort(ns, kind="stable")
            ns = ns[order]
            acc[k] = [acc[k][i] for i in order]
        t = (ns - np.int64(origin_ns)).astype(np.float64) * 1e-9
        if WANTED.get(k) == "string":
            out.t[k] = t; out.v[k] = np.asarray(acc[k], dtype=object); continue
        try:
            arr = np.asarray(acc[k], dtype=np.float32)
        except ValueError:                                  # ragged (joy layouts differ per driver)
            n = min(len(x) for x in acc[k])
            arr = np.asarray([x[:n] for x in acc[k]], dtype=np.float32)
        out.t[k] = t
        out.v[k] = arr[:, 0] if arr.ndim == 2 and arr.shape[1] == 1 else arr
    return out


def bags_under(root: str) -> List[str]:
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and
                  any(f.endswith(".db3") for f in os.listdir(os.path.join(root, d))))
