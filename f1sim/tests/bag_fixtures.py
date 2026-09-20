"""Write a small rosbag2 with exactly the topics and timings a test wants.

The real recordings are 300 MB each and live outside the repository, and the graph's own sim bag is
a minute of simulator time. Neither belongs in a unit test, and neither lets a test say "now drop
`/odom` for half a second and see what the checker reports". So the tests that are about the
*rules* -- `learn/bagdata.py`'s reconstruction, `system_check`'s verdicts -- write their own bag
here, and the tests that are about real data (`test_graph_parity.py`) use fixtures made from real
bags by `scripts/make_graph_fixture.py`.
"""
from __future__ import annotations

import math
import os

import numpy as np

TYPES = {
    "/scan": "sensor_msgs/msg/LaserScan",
    "/sensors/imu/raw": "sensor_msgs/msg/Imu",
    "/odom": "nav_msgs/msg/Odometry",
    "/drive": "ackermann_msgs/msg/AckermannDriveStamped",
    "/f1sim/plan": "f1sim_interfaces/msg/Plan",
    "/f1sim/policy_state": "f1sim_interfaces/msg/PolicyState",
    "/f1sim/controller/diag": "diagnostic_msgs/msg/DiagnosticArray",
    "/f1sim/reset": "std_msgs/msg/Empty",
    "/ego_racecar/odom": "nav_msgs/msg/Odometry",
}


def _stamp(t):
    from builtin_interfaces.msg import Time
    return Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)) % 1_000_000_000)


def scan(t, n_beams=64, r=3.0, range_max=10.0):
    from sensor_msgs.msg import LaserScan
    m = LaserScan()
    m.header.stamp = _stamp(t); m.header.frame_id = "laser"
    m.angle_min, m.angle_max = -2.35619449, 2.35619449
    m.angle_increment = (m.angle_max - m.angle_min) / (n_beams - 1)
    m.range_min, m.range_max = 0.0, range_max
    m.ranges = [float(r)] * n_beams if np.isscalar(r) else [float(x) for x in r]
    return m


def imu(t, gz=0.0, ax_si=0.0, roll=0.0, pitch=0.0, cov0=0.0):
    from sensor_msgs.msg import Imu
    m = Imu()
    m.header.stamp = _stamp(t); m.header.frame_id = "imu"
    m.angular_velocity.z = float(gz)
    m.linear_acceleration.x = float(ax_si)
    m.linear_acceleration.z = 9.80665
    cr, sr, cp, sp = (math.cos(roll / 2), math.sin(roll / 2),
                      math.cos(pitch / 2), math.sin(pitch / 2))
    m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z = (cr * cp, sr * cp,
                                                                          cr * sp, -sr * sp)
    cov = list(m.orientation_covariance); cov[0] = float(cov0)
    m.orientation_covariance = cov
    return m


def odom(t, v=0.0, x=0.0, y=0.0, yaw=0.0):
    from nav_msgs.msg import Odometry
    m = Odometry()
    m.header.stamp = _stamp(t); m.header.frame_id = "odom"; m.child_frame_id = "base_link"
    m.pose.pose.position.x, m.pose.pose.position.y = float(x), float(y)
    m.pose.pose.orientation.w, m.pose.pose.orientation.z = math.cos(yaw / 2), math.sin(yaw / 2)
    m.twist.twist.linear.x = float(v)
    return m


def drive(t, steer=0.0, speed=0.0):
    from ackermann_msgs.msg import AckermannDriveStamped
    m = AckermannDriveStamped()
    m.header.stamp = _stamp(t)
    m.drive.steering_angle = float(steer); m.drive.speed = float(speed)
    return m


def plan(t, values=None, seq=0, checkpoint="run/ppo.pt@0123456789ab"):
    from f1sim_interfaces.msg import Plan
    m = Plan()
    m.header.stamp = _stamp(t)
    m.plan = [float(x) for x in (values if values is not None else [0.0] * 8)]
    m.seq = int(seq); m.checkpoint = checkpoint
    return m


def empty(_t):
    from std_msgs.msg import Empty
    return Empty()


def policy_state(t, checkpoint="run/ppo.pt@0123456789ab", memory="", clears=0):
    from f1sim_interfaces.msg import PolicyState
    m = PolicyState()
    m.header.stamp = _stamp(t)
    m.checkpoint = checkpoint
    m.memory_kind = memory
    m.memory_present = bool(memory)
    m.memory_clears = int(clears)
    return m


def controller_diag(t, arm="fixed_low", mu="0.73423", traction="off", state="off"):
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    a = DiagnosticArray()
    a.header.stamp = _stamp(t)
    st = DiagnosticStatus(level=DiagnosticStatus.OK, name="f1sim: controller")
    st.values = [KeyValue(key="arm", value=arm), KeyValue(key="mu", value=mu),
                 KeyValue(key="traction", value=traction),
                 KeyValue(key="traction_state", value=state)]
    a.status = [st]
    return a


def write(path: str, records) -> str:
    """`records` is an iterable of `(time [s], topic, message)`, written in the order given."""
    import rosbag2_py
    from rclpy.serialization import serialize_message
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("", ""))
    created = set()
    for t, topic, msg in records:
        if topic not in created:
            w.create_topic(rosbag2_py.TopicMetadata(name=topic, type=TYPES[topic],
                                                    serialization_format="cdr"))
            created.add(topic)
        w.write(topic, serialize_message(msg), int(round(t * 1e9)))
    w.close()
    return path


def graph_run(path: str, *, frames=40, dt=0.025, imu_per_frame=1, t0=100.0, speeds=None,
              with_plan=True, with_drive=True, with_state=False, reset_at=None,
              drop_odom_from=None):
    """A bag shaped like a run of the graph: sensors, then the plan, then the command.

    `speeds` is the `/odom` wheel speed per frame (default: a ramp). `reset_at` is a frame index
    to publish `/f1sim/reset` before. `drop_odom_from` stops publishing `/odom` at that frame,
    which is how a test makes a topic go stale without deleting it from the bag.
    """
    speeds = list(speeds) if speeds is not None else [0.1 * i for i in range(frames)]
    rec = []
    for i in range(frames):
        t = t0 + i * dt
        if reset_at is not None and i == reset_at:
            rec.append((t - dt / 3, "/f1sim/reset", empty(t)))
        if drop_odom_from is None or i < drop_odom_from:
            rec.append((t - dt / 4, "/odom", odom(t - dt / 4, v=speeds[i], x=0.05 * i, yaw=0.01 * i)))
        for k in range(imu_per_frame):
            ti = t - dt / 2 + k * dt / (2 * imu_per_frame)
            rec.append((ti, "/sensors/imu/raw",
                        imu(ti, gz=0.02 * i, ax_si=0.5, roll=0.01, pitch=-0.02)))
        rec.append((t, "/scan", scan(t, r=3.0 + 0.01 * i)))
        if with_plan:
            rec.append((t + dt / 8, "/f1sim/plan", plan(t, [0.01 * i] * 8, seq=i)))
        if with_drive:
            rec.append((t + dt / 6, "/drive", drive(t, steer=0.001 * i, speed=0.5 + 0.05 * i)))
        if with_state:
            rec.append((t + dt / 7, "/f1sim/policy_state", policy_state(t, memory="gru(16)")))
            rec.append((t + dt / 5, "/f1sim/controller/diag", controller_diag(t)))
    return write(path, rec)
