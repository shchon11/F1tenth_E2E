"""The deployment node's sensor contract, exercised through its real callbacks, offline.

`policy_node` feeds `ObsBuilder` an `imu_att` pair the policy was trained to read as
**(roll, pitch)** -- the training side builds it from `r.imu_att[:, :2]` in `gym_env._obs`. Putting
the wrong quantity there crashes nothing; it drives the car on an observation it never saw. Same for
running the actor on sensor values that stopped arriving.

These tests drive the actual `PolicyNode.on_imu` / `on_vesc_imu` / `on_odom` / `on_scan` with stub
ROS objects (a recording publisher, a recording model, a fake clock) and assert what the model was
handed and what the publisher received. Nothing here starts a ROS graph, opens a device, or
commands a vehicle: the zero-speed path below is checked as code, and is not an authorisation to
actuate anything.

Where the constants come from
-----------------------------
`work/claude-sim2real-review/numbers/{vesc_attitude,quat_check}.json`, measured over 9 recordings:

* `/sensors/imu/raw.orientation` is a finite unit quaternion in every sample checked (norm
  1.000000 at p01/p50/p99, never all-zero, `orientation_covariance[0] = 0`), and the canonical
  extraction reproduces the gravity roll and pitch to within 0.003 deg at rest.
* `VescImu.ypr` is (roll, pitch, yaw) in degrees -- `ypr.z` spans 357-360 deg and tracks the
  integrated gyro at |r| = 0.994-1.000 -- while an earlier version of this node read `ypr.z` as
  roll. `ypr` is no longer used at all; the quaternion is the single source.
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "f1sim_ros"))

pytest.importorskip("rclpy")
torch = pytest.importorskip("torch")

from builtin_interfaces.msg import Time                   # noqa: E402

import f1sim_ros.policy_node as pn                       # noqa: E402
from f1sim.learn.obs import ObsBuilder, ObsSpec          # noqa: E402


def ros_time(sec=1.0):
    t = Time()
    t.sec = int(sec)
    t.nanosec = int((sec - int(sec)) * 1e9)
    return t

G = 9.80665

#: (bag, quaternion w x y z, accel in g) with the car standing still.
REST_SAMPLES = [
    ("20260827-115713", (0.681194007, -0.01486029, -0.011151765, 0.731867194),
     (-0.007105, -0.036621, 1.005148)),
    ("20260827-115713", (0.696202278, -0.011271207, 0.020475559, -0.717465043),
     (-0.007924, -0.046387, 1.01006)),
    ("20260826-192819", (0.650787294, -0.009041953, 0.01127251, -0.75912261),
     (-0.001028, -0.029297, 1.011791)),
    ("20260727-204907", (0.826096654, 0.005349036, -0.003283029, -0.563493669),
     (-0.000556, 0.012695, 1.011287)),
]


def gravity_rp(accel_g):
    """Roll and pitch from the gravity vector, in radians -- independent of any attitude filter."""
    ax, ay, az = (c * G for c in accel_g)
    return math.atan2(ay, az), math.atan2(-ax, math.hypot(ay, az))


# ==================================================================== stubs
class RecordingPub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


class RecordingModel:
    """Stands in for the actor: records what it was asked and returns a fixed action."""

    def __init__(self, act_dim=2):
        self.calls = []
        self.act_dim = act_dim

    def act(self, scan, pro, deterministic=True):
        self.calls.append((scan.clone(), pro.clone()))
        return torch.zeros(1, self.act_dim), None


class Logger:
    def __init__(self):
        self.lines = []

    def _add(self, kind):
        return lambda s: self.lines.append((kind, s))

    def __getattr__(self, name):
        return self._add(name)


def make_node(spec=None, enabled=True, timeout=0.25):
    """A PolicyNode with every ROS dependency replaced, and its real methods intact."""
    spec = spec or ObsSpec(n_beams=64, scan_stack=2, scan_stride=1, action_history=2,
                           act_dim=2, hist_len=0, range_max=10.0, v_max=8.0)
    n = pn.PolicyNode.__new__(pn.PolicyNode)
    n.device = torch.device("cpu")
    n.spec = spec
    n.obs = ObsBuilder(spec, "cpu")
    n.model = RecordingModel(spec.act_dim)
    n.pub = RecordingPub()
    n.speed_cap = 4.0
    n.steer_max = 0.4189
    n.v = 0.0
    n.imu_buf = []
    n.imu_stamps = []
    n.att = (0.0, 0.0)
    n.yaw_rate = 0.0
    n.accel_scale = G                      # bags publish g; fix it so the test is about attitude
    n._unit_warned = False
    n.tracker = None
    n.cal = (0.0, 1.0, 1.0)
    n.timeout = timeout
    n.t_att = n.t_imu = n.t_odom = None
    n._inhibited = False
    n._last_inhibit_log = -1e9
    n.last_t = None
    n._log = Logger()
    n._enabled = enabled
    n._now = 100.0
    n.imu_mean = None
    n.t_imu_mean = None
    n.t_scan = None
    n.att_stamp = None
    n.get_logger = lambda: n._log
    n.get_parameter = lambda name: SimpleNamespace(value=n._enabled)
    n.clock = lambda: n._now
    n.stamp_now = lambda: ros_time(n._now)
    return n


def imu_msg(q=(1.0, 0.0, 0.0, 0.0), accel_g=(0.0, 0.0, 1.0), gyro=(0.0, 0.0, 0.0), cov0=0.0,
            stamp=None):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp),
        orientation=SimpleNamespace(w=q[0], x=q[1], y=q[2], z=q[3]),
        orientation_covariance=[cov0] + [0.0] * 8,
        angular_velocity=SimpleNamespace(x=gyro[0], y=gyro[1], z=gyro[2]),
        linear_acceleration=SimpleNamespace(x=accel_g[0], y=accel_g[1], z=accel_g[2]))


def vesc_msg(q=(1.0, 0.0, 0.0, 0.0), cov0=0.0, ypr=(0.0, 0.0, 0.0), stamp=None):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp),
        imu=SimpleNamespace(
            orientation=SimpleNamespace(w=q[0], x=q[1], y=q[2], z=q[3]),
            orientation_covariance=[cov0] + [0.0] * 8,
            ypr=SimpleNamespace(x=ypr[0], y=ypr[1], z=ypr[2])))


def odom_msg(v):
    return SimpleNamespace(twist=SimpleNamespace(twist=SimpleNamespace(
        linear=SimpleNamespace(x=v), angular=SimpleNamespace(z=0.0))))


def scan_msg(n_beams=64, r=3.0):
    return SimpleNamespace(ranges=[r] * n_beams, range_max=10.0,
                           header=SimpleNamespace(stamp=ros_time(1.0)))


def feed_ready(n, q=(1.0, 0.0, 0.0, 0.0), accel_g=(0.0, 0.0, 1.0)):
    n.on_odom(odom_msg(1.5))
    n.on_imu(imu_msg(q=q, accel_g=accel_g))


# ==================================================================== attitude source
def test_quat_to_rp_known_answers():
    for roll_deg, pitch_deg, yaw_deg in [(0, 0, 0), (10, 0, 0), (-10, 0, 0), (0, 10, 0),
                                         (0, -10, 0), (0, 0, 90), (5, -7, 140), (-12, 3, -60)]:
        r, p, y = (math.radians(v) for v in (roll_deg, pitch_deg, yaw_deg))
        cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2),
                                  math.sin(p / 2), math.cos(y / 2), math.sin(y / 2))
        q = SimpleNamespace(w=cr * cp * cy + sr * sp * sy, x=sr * cp * cy - cr * sp * sy,
                            y=cr * sp * cy + sr * cp * sy, z=cr * cp * sy - sr * sp * cy)
        got_r, got_p = pn.quat_to_rp(q)
        assert got_r == pytest.approx(r, abs=1e-9)
        assert got_p == pytest.approx(p, abs=1e-9)


@pytest.mark.parametrize("bag,q,accel", REST_SAMPLES)
def test_valid_quaternion_reaches_the_proprio_attitude_channel(bag, q, accel):
    """The regression this file exists for: the last two proprio entries must be the real
    roll and pitch, scaled by `att_scale`."""
    n = make_node()
    roll_g, pitch_g = gravity_rp(accel)
    n.on_odom(odom_msg(2.0))
    n.on_imu(imu_msg(q=q, accel_g=accel))
    n.on_scan(scan_msg())

    assert len(n.model.calls) == 1, f"{bag}: the actor was not called"
    pro = n.model.calls[0][1][0].numpy()
    assert pro[-2] * n.spec.att_scale == pytest.approx(roll_g, abs=0.01), f"{bag}: roll channel"
    assert pro[-1] * n.spec.att_scale == pytest.approx(pitch_g, abs=0.01), f"{bag}: pitch channel"
    assert len(n.pub.msgs) == 1


def test_yaw_never_reaches_the_roll_channel():
    """Level car, arbitrary heading. Reading `ypr.z` would put the heading in the roll channel:
    3 rad / 0.35 = 8.6 in a channel whose training range is about +-0.43."""
    n = make_node()
    for yaw_deg in (-179.0, -90.0, 0.0, 90.0, 179.0):
        h = math.radians(yaw_deg) / 2
        q = (math.cos(h), 0.0, 0.0, math.sin(h))          # level, yaw only
        n.on_odom(odom_msg(1.0))
        n.on_imu(imu_msg(q=q))
        n.on_scan(scan_msg())
        pro = n.model.calls[-1][1][0].numpy()
        assert abs(pro[-2]) < 0.06, f"yaw {yaw_deg} deg leaked into roll as {pro[-2]}"
        assert abs(pro[-1]) < 0.06


def test_invalid_quaternion_is_ignored_and_the_last_good_one_held():
    n = make_node()
    good = (0.9961947, 0.0871557, 0.0, 0.0)               # roll +10 deg
    n.on_odom(odom_msg(1.0)); n.on_imu(imu_msg(q=good)); n.on_scan(scan_msg())
    roll_first = n.model.calls[-1][1][0].numpy()[-2] * n.spec.att_scale
    assert roll_first == pytest.approx(math.radians(10.0), abs=0.01)

    for bad in (imu_msg(q=(0.0, 0.0, 0.0, 0.0)),
                imu_msg(q=(1.0, 0.0, 0.0, 0.0), cov0=-1.0),
                imu_msg(q=(0.5, 0.0, 0.0, 0.0)),
                imu_msg(q=(float("nan"), 0.0, 0.0, 1.0))):
        n.on_odom(odom_msg(1.0)); n.on_imu(bad); n.on_scan(scan_msg())
        held = n.model.calls[-1][1][0].numpy()[-2] * n.spec.att_scale
        assert held == pytest.approx(math.radians(10.0), abs=0.01), "an unusable orientation was used"


def test_vesc_orientation_works_when_the_raw_topic_carries_none():
    """The simulator bridge marks every raw sample `covariance = -1` except the one where the
    attitude is measured. A VESC message with a valid quaternion must still drive the channel."""
    n = make_node()
    q = (0.9961947, 0.0871557, 0.0, 0.0)                  # roll +10 deg
    n.on_odom(odom_msg(1.0))
    n.on_imu(imu_msg(q=(1.0, 0.0, 0.0, 0.0), cov0=-1.0))  # raw has no orientation estimate
    n.on_vesc_imu(vesc_msg(q=q, ypr=(999.0, 999.0, 999.0)))   # ypr is deliberately absurd
    n.on_scan(scan_msg())
    roll = n.model.calls[-1][1][0].numpy()[-2] * n.spec.att_scale
    assert roll == pytest.approx(math.radians(10.0), abs=0.01)


def test_a_valid_raw_sample_does_not_permanently_disable_the_vesc_source():
    n = make_node()
    n.on_odom(odom_msg(1.0))
    n.on_imu(imu_msg(q=(1.0, 0.0, 0.0, 0.0)))                              # valid, level
    n.on_vesc_imu(vesc_msg(q=(0.9961947, 0.0871557, 0.0, 0.0)))            # newer, roll +10
    n.on_scan(scan_msg())
    roll = n.model.calls[-1][1][0].numpy()[-2] * n.spec.att_scale
    assert roll == pytest.approx(math.radians(10.0), abs=0.01), "newest valid orientation must win"


# ==================================================================== freshness
def test_stale_sensors_inhibit_the_actor_and_publish_zero_speed():
    n = make_node(timeout=0.25)
    feed_ready(n)
    n.on_scan(scan_msg())
    assert len(n.model.calls) == 1

    n._now += 1.0                                    # nothing new arrives for a second
    n.on_scan(scan_msg())
    assert len(n.model.calls) == 1, "the actor ran on stale sensors"
    assert len(n.pub.msgs) == 2
    last = n.pub.msgs[-1]
    assert last.drive.speed == 0.0 and last.drive.steering_angle == 0.0
    assert any("no actor command" in s for _k, s in n._log.lines)


def test_no_attitude_at_startup_means_no_command():
    """`VescImuStamped` being importable is not evidence the topic publishes: one of the 22
    recordings has no `/sensors/imu` at all."""
    n = make_node()
    n.on_odom(odom_msg(1.0))
    n.on_imu(imu_msg(q=(0.0, 0.0, 0.0, 0.0)))        # samples arrive, orientation never valid
    n.on_scan(scan_msg())
    assert n.model.calls == [], "drove without ever having a valid attitude"
    assert n.pub.msgs and n.pub.msgs[-1].drive.speed == 0.0


def test_missing_odom_alone_inhibits():
    n = make_node()
    n.on_imu(imu_msg())
    n.on_scan(scan_msg())
    assert n.model.calls == []
    assert n.pub.msgs[-1].drive.speed == 0.0


def test_disabled_publishes_nothing_in_either_path():
    n = make_node(enabled=False)
    feed_ready(n)
    n.on_scan(scan_msg())
    assert len(n.model.calls) == 1 and n.pub.msgs == [], "published while disabled"
    n._now += 1.0
    n.on_scan(scan_msg())
    assert n.pub.msgs == [], "published a zero-speed command while disabled"


def test_recovery_clears_the_observation_history():
    """After a gap the action history describes a segment that is over; stitching the new one onto
    it feeds the policy a history that never happened."""
    n = make_node()
    feed_ready(n)
    n.on_scan(scan_msg(r=3.0))
    n.obs.push_action(torch.tensor([0.9, 0.9]))
    assert float(n.obs.act_hist.abs().sum()) > 0

    n._now += 1.0
    n.on_scan(scan_msg())                            # inhibited
    assert n._inhibited

    n._now += 0.01
    feed_ready(n)
    n.on_scan(scan_msg(r=5.0))                       # recovered
    assert not n._inhibited
    assert any("recovered" in s for _k, s in n._log.lines)
    pro = n.model.calls[-1][1][0].numpy()
    n_hist = n.spec.act_dim * n.spec.action_history
    assert np.allclose(pro[1:1 + n_hist], 0.0), "stale action history survived the gap"


def test_imu_buffer_is_bounded_and_excludes_stale_samples():
    n = make_node(timeout=0.25)
    n.on_odom(odom_msg(1.0))
    for _ in range(pn.IMU_BUF_MAX + 20):
        n.on_imu(imu_msg(gyro=(0.0, 0.0, 1.0)))
    assert len(n.imu_buf) == pn.IMU_BUF_MAX, "unbounded IMU buffer"

    n._now += 0.30                                   # those samples are now stale
    n.on_imu(imu_msg(gyro=(0.0, 0.0, 0.5)))          # one fresh sample
    n.on_odom(odom_msg(1.0))
    n.on_scan(scan_msg())
    assert len(n.model.calls) == 1
    pro = n.model.calls[0][1][0].numpy()
    gz = pro[1 + n.spec.act_dim * n.spec.action_history + 1 + 2] * n.spec.gyro_scale
    assert gz == pytest.approx(0.5, abs=1e-4), "stale IMU samples were averaged into a new scan"


def test_a_brief_imu_gap_holds_the_last_mean_instead_of_reporting_freefall():
    """A dropped sample is tolerated; `zeros(6)` is not.

    An empty buffer used to become `zeros(6)`, which is not "no rotation" but zero specific force
    -- freefall -- in the channel the policy reads as gravity. Within the timeout the last real
    mean is reused instead.
    """
    n = make_node(timeout=0.25)
    feed_ready(n)
    n.on_scan(scan_msg())
    az_idx = 1 + n.spec.act_dim * n.spec.action_history + 1 + 5
    first = n.model.calls[0][1][0].numpy()[az_idx] * n.spec.accel_scale
    assert first == pytest.approx(G, abs=0.1)

    n._now += 0.02                                   # one dropped sample, well inside the timeout
    n.on_odom(odom_msg(1.0))
    n.on_scan(scan_msg())
    assert len(n.model.calls) == 2, "a single dropped IMU sample stopped the actor"
    held = n.model.calls[1][1][0].numpy()[az_idx] * n.spec.accel_scale
    assert held == pytest.approx(G, abs=0.1), "an IMU gap was reported as freefall"


def test_reusing_the_imu_mean_does_not_renew_its_age():
    """The hold is bounded by when the samples were taken, not by when they were last reused --
    otherwise a stalled IMU is refreshed forever by the scans that consume it."""
    n = make_node(timeout=0.25)
    feed_ready(n)
    n.on_scan(scan_msg())
    assert len(n.model.calls) == 1
    for _ in range(6):                               # scans keep coming, IMU does not
        n._now += 0.05
        n.on_odom(odom_msg(1.0))
        n.on_scan(scan_msg())
    assert len(n.model.calls) < 7, "a stale IMU mean was renewed by every scan"
    assert n.pub.msgs[-1].drive.speed == 0.0
    assert any("imu" in s for _k, s in n._log.lines)


# ==================================================================== watchdog: scans stop
def test_watchdog_stops_the_car_when_scans_cease():
    """Every other guard lives in `on_scan`, which is the callback that stops running when the
    LiDAR goes away -- leaving the last command standing with nothing to countermand it."""
    n = make_node(timeout=0.25)
    feed_ready(n)
    n.on_scan(scan_msg())
    assert len(n.pub.msgs) == 1

    n.on_watchdog()                                  # everything still fresh
    assert len(n.pub.msgs) == 1, "the watchdog fired while the sensors were fine"

    n._now += 1.0                                    # scans (and everything else) stop
    n.on_watchdog()
    assert len(n.model.calls) == 1, "the watchdog called the actor"
    assert len(n.pub.msgs) == 2
    assert n.pub.msgs[-1].drive.speed == 0.0 and n.pub.msgs[-1].drive.steering_angle == 0.0
    assert any("scan" in s for _k, s in n._log.lines)


def test_watchdog_stamps_with_the_current_time_not_an_old_scan_header():
    n = make_node(timeout=0.25)
    feed_ready(n)
    n.on_scan(scan_msg())
    n._now += 5.0
    n.on_watchdog()
    stamp = n.pub.msgs[-1].header.stamp
    assert stamp.sec == int(n._now), "the stop command carried a stale scan header stamp"


def test_watchdog_publishes_nothing_while_disabled():
    n = make_node(enabled=False, timeout=0.25)
    feed_ready(n)
    n._now += 1.0
    n.on_watchdog()
    assert n.pub.msgs == []


def test_recovery_after_total_scan_loss_clears_history():
    n = make_node(timeout=0.25)
    feed_ready(n)
    n.on_scan(scan_msg())
    n.obs.push_action(torch.tensor([0.9, 0.9]))
    n._now += 1.0
    n.on_watchdog()
    assert n._inhibited

    n._now += 0.01
    feed_ready(n)
    n.on_scan(scan_msg())
    assert not n._inhibited
    pro = n.model.calls[-1][1][0].numpy()
    n_hist = n.spec.act_dim * n.spec.action_history
    assert np.allclose(pro[1:1 + n_hist], 0.0), "history survived a total scan loss"


# ==================================================================== non-finite inputs
def test_non_finite_inputs_are_refused_and_do_not_count_as_fresh():
    n = make_node(timeout=0.25)
    feed_ready(n)
    n.on_scan(scan_msg())
    good_v = n.v

    n.on_odom(odom_msg(float("nan")))
    assert n.v == good_v, "a NaN speed was accepted"
    n.on_imu(imu_msg(gyro=(float("nan"), 0.0, 0.0)))
    assert all(all(math.isfinite(c) for c in row) for row in n.imu_buf), "a NaN IMU row was buffered"

    n._now += 1.0                                    # only the NaN messages arrived since
    n.on_odom(odom_msg(float("inf")))
    n.on_imu(imu_msg(accel_g=(0.0, float("nan"), 1.0)))
    n.on_scan(scan_msg())
    assert len(n.model.calls) == 1, "non-finite messages were treated as fresh data"
    assert n.pub.msgs[-1].drive.speed == 0.0


def test_a_message_older_than_the_accepted_one_is_ignored_when_stamps_exist():
    """`_note_attitude` prefers header stamps when both messages carry them; with no stamps it
    falls back to arrival order, which is documented as a limit rather than hidden."""
    n = make_node()
    newer = (0.9961947, 0.0871557, 0.0, 0.0)         # roll +10 deg
    n.on_odom(odom_msg(1.0))
    n.on_imu(imu_msg(q=newer, stamp=ros_time(10.0)))
    n.on_vesc_imu(vesc_msg(q=(1.0, 0.0, 0.0, 0.0), stamp=ros_time(9.0)))   # older: level
    n.on_scan(scan_msg())
    roll = n.model.calls[-1][1][0].numpy()[-2] * n.spec.att_scale
    assert roll == pytest.approx(math.radians(10.0), abs=0.01), "an older message overwrote a newer one"


# ==================================================================== real message classes
# Everything above builds messages from SimpleNamespace, which is fast but shaped by what the test
# author expected. These use the generated classes, offline (no Node, no ROS graph), because two
# real defects hid behind the stubs: `orientation_covariance` is a numpy array of nine doubles on
# Humble, so `if cov` raises ValueError on every genuine message, and a NaN sample reaching
# `_accel_to_si` latches the g-vs-SI scale before the finite check rejects it.
def _real_imu(q=(1.0, 0.0, 0.0, 0.0), accel_g=(0.0, 0.0, 1.0), gyro=(0.0, 0.0, 0.0), cov0=None):
    from sensor_msgs.msg import Imu
    m = Imu()
    m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z = q
    m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = gyro
    m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = accel_g
    if cov0 is not None:
        m.orientation_covariance[0] = cov0
    return m


def _real_odom(v):
    from nav_msgs.msg import Odometry
    m = Odometry()
    m.twist.twist.linear.x = float(v)
    return m


def _real_scan(n_beams=64, r=3.0):
    from sensor_msgs.msg import LaserScan
    m = LaserScan()
    m.ranges = [float(r)] * n_beams
    m.range_max = 10.0
    m.header.stamp = ros_time(1.0)
    return m


def test_end_to_end_with_generated_message_classes():
    """One pass through the real callbacks with the real message types."""
    n = make_node()
    q = (0.9961947, 0.0871557, 0.0, 0.0)                 # roll +10 deg
    n.on_odom(_real_odom(2.0))
    n.on_imu(_real_imu(q=q, accel_g=(0.0, 0.17, 0.985)))
    n.on_scan(_real_scan())
    assert len(n.model.calls) == 1, "the actor was not called with real messages"
    pro = n.model.calls[0][1][0].numpy()
    assert pro[-2] * n.spec.att_scale == pytest.approx(math.radians(10.0), abs=0.02)
    assert len(n.pub.msgs) == 1


def test_real_message_covariance_minus_one_is_honoured():
    """`orientation_covariance` is a numpy array here; truth-testing it would raise."""
    n = make_node()
    n.on_odom(_real_odom(1.0))
    n.on_imu(_real_imu(q=(0.9961947, 0.0871557, 0.0, 0.0)))          # valid: roll +10
    n.on_scan(_real_scan())
    roll = n.model.calls[-1][1][0].numpy()[-2] * n.spec.att_scale
    assert roll == pytest.approx(math.radians(10.0), abs=0.02)

    n.on_odom(_real_odom(1.0))
    n.on_imu(_real_imu(q=(1.0, 0.0, 0.0, 0.0), cov0=-1.0))           # level, but flagged unusable
    n.on_scan(_real_scan())
    held = n.model.calls[-1][1][0].numpy()[-2] * n.spec.att_scale
    assert held == pytest.approx(math.radians(10.0), abs=0.02), "covariance -1 was not honoured"


def test_real_vesc_message_is_accepted():
    vesc_msgs = pytest.importorskip("vesc_msgs.msg")
    n = make_node()
    m = vesc_msgs.VescImuStamped()
    m.imu.orientation.w, m.imu.orientation.x = 0.9961947, 0.0871557  # roll +10 deg
    n.on_odom(_real_odom(1.0))
    n.on_imu(_real_imu(q=(1.0, 0.0, 0.0, 0.0), cov0=-1.0))           # raw offers no orientation
    n.on_vesc_imu(m)
    n.on_scan(_real_scan())
    roll = n.model.calls[-1][1][0].numpy()[-2] * n.spec.att_scale
    assert roll == pytest.approx(math.radians(10.0), abs=0.02)


def test_a_nan_first_sample_does_not_poison_the_accel_unit_detection():
    """`_accel_to_si` picks g vs SI from the first magnitude it sees, and `nan < 3.0` is False, so
    a NaN reaching it latches scale = 1.0 and divides every later reading by 9.81 forever."""
    n = make_node()
    n.accel_scale = None                                  # undetected, as at startup
    n.on_imu(_real_imu(accel_g=(float("nan"), 0.0, 1.0)))
    assert n.accel_scale is None, "a NaN sample latched the accelerometer scale"
    assert n.imu_buf == []

    n.on_imu(_real_imu(accel_g=(0.0, 0.0, 1.0)))          # normal gravity in g
    assert n.accel_scale == pytest.approx(G), "g was not detected after a NaN sample"
    n.on_odom(_real_odom(1.0))
    n.on_scan(_real_scan())
    az_idx = 1 + n.spec.act_dim * n.spec.action_history + 1 + 5
    az = n.model.calls[-1][1][0].numpy()[az_idx] * n.spec.accel_scale
    assert az == pytest.approx(G, abs=0.1), "accelerometer channel is off by the g/SI factor"
