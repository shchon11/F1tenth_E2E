"""The two ROS bridges must stamp what they publish at the time it was actually measured.

Both `publish_scan` and `publish_imu` are exercised directly with a fake clock and fake message
types -- no rclpy node, no discovery, no network. They are unbound methods called on a stand-in, so
this tests the code the nodes run without importing them (the node modules pull in ROS packages that
are not importable in this environment).

Three contracts:
  * LaserScan.header.stamp is the FIRST ray, per the ROS 2 message definition, which is one sweep
    before the end of the control step the scan was traced in.
  * every IMU sample is stamped at its own offset before the step end, and the summary message is
    stamped at the latest sample -- not at the step end.
  * orientation appears only on the sample the attitude estimate belongs to (the last one); the
    others declare it unavailable with covariance[0] = -1.
"""
import math

import numpy as np
import pytest

from f1sim.params import Config


# --------------------------------------------------------------------------- fakes
class FakeDuration:
    def __init__(self, seconds=0.0):
        self.seconds = float(seconds)


class FakeTime:
    """Just enough of rclpy.time.Time: subtraction by a Duration and .to_msg()."""

    def __init__(self, seconds):
        self.seconds = float(seconds)

    def __sub__(self, other):
        return FakeTime(self.seconds - other.seconds)

    def to_msg(self):
        return round(self.seconds, 12)


class FakeHeader:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class FakeMsg:
    def __init__(self):
        self.header = FakeHeader()
        self.orientation = None
        self.orientation_covariance = [0.0] * 9
        self.angular_velocity = type("V", (), {})()
        self.linear_acceleration = type("V", (), {})()
        self.ranges = []
        self.angle_min = self.angle_max = self.angle_increment = 0.0
        self.time_increment = self.scan_time = 0.0
        self.range_min = self.range_max = 0.0


class FakePub:
    def __init__(self):
        self.sent = []

    def publish(self, m):
        self.sent.append(m)


class FakeNode:
    """Stands in for the bridge node: only what the two publish_* methods touch."""

    def __init__(self, meta):
        self._meta = meta
        self.laser_frame = "laser"
        self.pub_scan = FakePub()
        self.pub_imu_raw = FakePub()
        self.pub_imu = FakePub()
        self.sim = type("S", (), {"scan_meta": lambda _s: meta})()


def rpy_to_quat(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2),
                             math.sin(p / 2), math.cos(y / 2), math.sin(y / 2))
    q = type("Q", (), {})()
    q.w = cr * cp * cy + sr * sp * sy; q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy; q.z = cr * cp * sy - sr * sp * cy
    return q


# --------------------------------------------------------------------------- the code under test
# Transcribed from f1sim_ros/bridge_node.py and vesc_sim_node.py: the node modules import ROS
# packages that are not available here, so the timing rules are exercised as the bridges implement
# them. test_ros_bridge_sources_match_these_rules below pins them to the real files.
def publish_scan(node, ranges, now):
    m = node.sim.scan_meta()
    sweep = m["time_increment"] * (len(ranges) - 1)
    stamp = (now - FakeDuration(seconds=sweep)).to_msg()
    msg = FakeMsg()
    msg.header.stamp = stamp
    msg.header.frame_id = node.laser_frame
    msg.time_increment, msg.scan_time = m["time_increment"], m["scan_time"]
    msg.ranges = list(ranges)
    node.pub_scan.publish(msg)


def publish_imu(node, samples, att, now, offsets):
    K = samples.shape[0]
    for k in range(K):
        m = FakeMsg()
        m.header.frame_id = "imu"
        m.header.stamp = (now - FakeDuration(seconds=float(offsets[k]))).to_msg()
        if k == K - 1:
            m.orientation = rpy_to_quat(float(att[0]), float(att[1]), float(att[2]))
        else:
            m.orientation_covariance[0] = -1.0
        node.pub_imu_raw.publish(m)
    s = FakeMsg()
    s.header.frame_id = "imu"
    s.header.stamp = (now - FakeDuration(seconds=float(offsets[-1]))).to_msg()
    s.orientation = rpy_to_quat(float(att[0]), float(att[1]), float(att[2]))
    node.pub_imu.publish(s)


# --------------------------------------------------------------------------- tests
@pytest.fixture
def meta():
    c = Config()
    control_dt = 1.0 / c.sim.control_rate
    sweep = (c.lidar.fov / (2 * math.pi)) * control_dt
    return {"time_increment": sweep / (c.lidar.n_beams - 1), "scan_time": control_dt,
            "n_beams": c.lidar.n_beams, "sweep": sweep}


def test_scan_header_is_the_first_ray_not_the_step_end(meta):
    node = FakeNode(meta)
    n = meta["n_beams"]
    step_end = 10.0
    publish_scan(node, np.ones(n, dtype=np.float32), FakeTime(step_end))
    msg = node.pub_scan.sent[0]
    expected = step_end - meta["time_increment"] * (n - 1)
    assert msg.header.stamp == pytest.approx(expected, abs=1e-12)
    # and that offset really is the 18.75 ms sweep of a 270 deg scan on a 40 Hz loop
    assert step_end - msg.header.stamp == pytest.approx(meta["sweep"], abs=1e-12)
    assert step_end - msg.header.stamp == pytest.approx(0.01875, abs=1e-9)
    # the header must not be the completion time
    assert msg.header.stamp != step_end


def test_every_imu_sample_is_stamped_at_its_own_offset(meta):
    node = FakeNode(meta)
    step_end = 10.0
    offsets = np.array([0.02, 0.0])                 # the 2-sample phase of the 50 Hz schedule
    samples = np.arange(12, dtype=np.float64).reshape(2, 6)
    publish_imu(node, samples, np.array([0.1, 0.2, 0.3]), FakeTime(step_end), offsets)

    raw = node.pub_imu_raw.sent
    assert len(raw) == 2
    for m, off in zip(raw, offsets):
        assert m.header.stamp == pytest.approx(step_end - off, abs=1e-12)
    assert raw[0].header.stamp < raw[1].header.stamp, "samples must be in time order"
    # the summary reports the latest sample, so it carries that sample's time, not the step end
    assert node.pub_imu.sent[0].header.stamp == pytest.approx(step_end - offsets[-1], abs=1e-12)


def test_orientation_only_on_the_sample_it_was_measured_at(meta):
    node = FakeNode(meta)
    offsets = np.array([0.02, 0.0])
    publish_imu(node, np.zeros((2, 6)), np.array([0.1, 0.2, 0.3]), FakeTime(5.0), offsets)
    first, last = node.pub_imu_raw.sent
    assert first.orientation is None and first.orientation_covariance[0] == -1.0, (
        "an earlier sample must not carry the attitude measured at the end of the step")
    assert last.orientation is not None and last.orientation_covariance[0] == 0.0
    q = last.orientation
    assert q.w ** 2 + q.x ** 2 + q.y ** 2 + q.z ** 2 == pytest.approx(1.0, abs=1e-12)


def test_single_sample_step_still_carries_an_orientation(meta):
    node = FakeNode(meta)
    publish_imu(node, np.zeros((1, 6)), np.array([0.0, 0.0, 0.0]), FakeTime(5.0), np.array([0.005]))
    only = node.pub_imu_raw.sent[0]
    assert only.orientation is not None, "the single sample of a 1-sample step is the last one"
    assert only.header.stamp == pytest.approx(4.995, abs=1e-12)


def test_ros_bridge_sources_match_these_rules():
    """Pin the transcription above to what the bridges actually contain."""
    import os
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "f1sim_ros", "f1sim_ros")
    for name in ("bridge_node.py", "vesc_sim_node.py"):
        src = open(os.path.join(root, name)).read()
        assert 'sweep = m["time_increment"] * (len(ranges) - 1)' in src, f"{name}: scan sweep offset"
        assert "stamp = (now - Duration(seconds=sweep)).to_msg()" in src, f"{name}: first-ray stamp"
        assert "Duration(seconds=float(offsets[k]))" in src, f"{name}: per-sample imu stamp"
        assert "Duration(seconds=float(offsets[-1]))" in src, f"{name}: summary at latest sample"
        assert "if k == K - 1:" in src, f"{name}: orientation only on the last sample"


# --------------------------------------------------------------------------- VESC ypr convention
def _vesc_ypr_from_att(roll, pitch, yaw):
    """The rule vesc_sim_node publishes, transcribed; pinned to the source below."""
    return (-math.degrees(roll), math.degrees(pitch), -math.degrees(yaw))


@pytest.mark.parametrize("roll,pitch,yaw", [
    (0.0, 0.0, 0.0),
    (0.10, -0.20, 1.50),
    (-0.35, 0.05, -2.90),
    (0.02, 0.40, 0.00),
])
def test_vesc_ypr_matches_the_measured_hardware_convention(roll, pitch, yaw):
    """ypr.x = -roll, ypr.y = +pitch, ypr.z = -yaw in degrees, cross-checked against the quaternion.

    Measured on the same VescImuStamped message in four recordings (competition, pre-competition,
    SPIN). The node used to publish (yaw, pitch, roll), which matched the dataset in neither order
    nor sign, so a consumer calibrated on a bag disagreed with the same consumer on the simulator.
    The quaternion stays canonical and is what this checks against.
    """
    ypr = _vesc_ypr_from_att(roll, pitch, yaw)
    q = rpy_to_quat(roll, pitch, yaw)

    # recover roll/pitch/yaw from the canonical quaternion and compare against the published ypr
    q_roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x ** 2 + q.y ** 2))
    q_pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    q_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))

    assert ypr[0] == pytest.approx(-math.degrees(q_roll), abs=1e-6)
    assert ypr[1] == pytest.approx(math.degrees(q_pitch), abs=1e-6)
    def wrapped(a, b):
        return math.degrees(math.atan2(math.sin(math.radians(a - b)), math.cos(math.radians(a - b))))
    assert wrapped(ypr[2], -math.degrees(q_yaw)) == pytest.approx(0.0, abs=1e-6)
    # and the old convention really was different, so this is not a no-op rename
    if abs(roll - yaw) > 1e-9:
        assert ypr[0] != pytest.approx(math.degrees(yaw), abs=1e-9)


def test_vesc_source_uses_the_measured_ypr_convention():
    import os
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "f1sim_ros", "f1sim_ros")
    src = open(os.path.join(root, "vesc_sim_node.py")).read()
    assert "(-math.degrees(roll), math.degrees(pitch)," in src and "-math.degrees(yaw))" in src, \
        "vesc_sim_node must publish ypr as (-roll, +pitch, -yaw) degrees"
