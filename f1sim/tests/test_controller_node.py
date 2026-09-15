"""The controller node's own behaviour: what it tracks, and when it refuses to.

`test_graph_parity.py` proves the controller reproduces the old node command for command. This file
is about the things the old node could not have, because they only exist once the two halves are
separate processes:

* a plan is tracked against **the scan it names**, not against whatever the sensors happen to say
  when it arrives;
* a plan that overtakes its own scan is held for one scan rather than mispaired;
* the plan stream stopping is its own failure, distinct from the sensors stopping, and has its own
  timeout;
* the stale-sensor rule is unchanged, at the same threshold, because this node subscribes to the
  same topics the old one did.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "f1sim_ros"))

pytest.importorskip("rclpy")
pytest.importorskip("f1sim_interfaces.msg",
                    reason="f1sim_interfaces is not built; see f1sim_interfaces/README.md")
torch = pytest.importorskip("torch")

from builtin_interfaces.msg import Time                       # noqa: E402

import f1sim_ros.controller_node as cn                        # noqa: E402
import f1sim_ros.deploy as deploy                             # noqa: E402
from f1sim.learn.obs import ObsSpec                           # noqa: E402
from f1sim_interfaces.msg import Plan, PolicyState            # noqa: E402

G = deploy.G
DT = 0.025


class RecordingPub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


class Logger:
    def __init__(self):
        self.lines = []

    def _add(self, kind):
        return lambda s: self.lines.append((kind, s))

    def __getattr__(self, name):
        return self._add(name)

    def text(self, kind=None):
        return " | ".join(s for k, s in self.lines if kind is None or k == kind)


def ros_time(sec):
    t = Time(); t.sec = int(sec); t.nanosec = int(round((sec - int(sec)) * 1e9)) % 1_000_000_000
    return t


class SpyTracker:
    """Records the (plan, v, cap, yaw) it was asked for and returns a fixed command.

    `wb` and `s_max` because the controller publishes the tracker's own model of the car on its
    diagnostics -- which is the point of that message, and which a stub has to carry too.
    """

    last_ref = None
    wb = 0.3302
    s_max = 0.4189
    v_max = 8.0

    def __init__(self, speed=2.0, steer=0.1):
        self.calls = []
        self.resets = 0
        self.speed, self.steer = speed, steer

    def __call__(self, a, v, cap, yaw_rate, delay=None):
        self.calls.append((a.clone(), float(v[0]), float(cap[0]),
                           None if yaw_rate is None else float(yaw_rate[0])))
        return torch.tensor([[self.steer, self.speed]])

    def reset(self, ids):
        self.resets += 1


def make_node(plan_timeout=0.25, timeout=0.25, speed=2.0):
    spec = ObsSpec(n_beams=64, scan_stack=2, act_dim=8, hist_len=0, range_max=10.0, v_max=8.0)
    n = cn.ControllerNode.__new__(cn.ControllerNode)
    n.device = torch.device("cpu")
    n.spec = spec
    n.tracker = SpyTracker(speed)
    n.delay = torch.tensor([0.035])
    n.pub = RecordingPub(); n.pub_diag = RecordingPub()
    n.pub_viz = None; n.pub_viz_clear = None
    n.speed_cap = 4.0; n.steer_max = 0.4189; n.cal = (0.0, 1.0, 1.0)
    n.timeout = timeout; n.plan_timeout = plan_timeout
    n.grip = None; n.clearance = None; n._scan_geometry_checked = True
    n.controller_arm = "legacy"
    n._snapshots = []; n._pending = None
    n.plan = None; n.t_plan = None; n.plan_seq = None; n.plan_checkpoint = ""
    n._inhibited = False; n._last_inhibit_log = -1e9; n._braking = False
    n._plan_unmatched = 0; n._last_unmatched_log = -1e9; n._commands = 0
    n.last_cmd = (0.0, 0.0); n.last_t = None
    n.traction_arm = "off"; n.traction = None; n.traction_state = "off"
    n._log = Logger(); n._now = 100.0
    n.get_logger = lambda: n._log
    n.get_parameter = lambda name: SimpleNamespace(value=True)
    n.clock = lambda: n._now
    n.stamp_now = lambda: ros_time(n._now)
    n.sensors = deploy.SensorIntake(n.clock, n._log, timeout)
    return n


def imu_msg():
    return SimpleNamespace(header=SimpleNamespace(stamp=None),
                           orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                           orientation_covariance=[0.0] * 9,
                           angular_velocity=SimpleNamespace(x=0.0, y=0.0, z=0.3),
                           linear_acceleration=SimpleNamespace(x=0.0, y=0.0, z=9.81))


def odom_msg(v):
    return SimpleNamespace(twist=SimpleNamespace(twist=SimpleNamespace(
        linear=SimpleNamespace(x=v), angular=SimpleNamespace(z=0.0))))


def scan_msg(stamp, n_beams=64, r=3.0):
    return SimpleNamespace(ranges=[r] * n_beams, range_max=10.0, angle_min=-2.356,
                           angle_max=2.356, header=SimpleNamespace(stamp=stamp))


def plan_msg(stamp, seq=0, values=None):
    m = Plan()
    m.header.stamp = stamp
    m.plan = [float(x) for x in (values if values is not None else [0.0] * 8)]
    m.seq = int(seq)
    m.checkpoint = "run/ppo_latest.pt@abcdef012345"
    return m


def sensors_at(n, v, stamp):
    """One control cycle of sensors and a scan, without a plan."""
    n.on_odom(odom_msg(v))
    n.on_imu(imu_msg())
    n.on_scan(scan_msg(stamp))


# ==================================================================== the message on the wire
def test_the_plan_message_is_eight_floats_a_checkpoint_and_a_sequence():
    m = plan_msg(ros_time(1.0), seq=7, values=range(8))
    assert list(m.plan) == [float(i) for i in range(8)]
    assert m.seq == 7 and m.checkpoint.endswith("abcdef012345")
    with pytest.raises(AssertionError):
        m.plan = [0.0] * 7                 # the action space is 8 wide; a variable length can be wrong


def test_the_policy_state_message_carries_the_memory_and_the_staleness():
    s = PolicyState()
    s.memory_present = True
    s.memory_kind = "gru(64)"
    s.stale = ["imu", "odom"]
    s.inhibited = True
    s.scan_age_s = 0.4
    assert list(s.stale) == ["imu", "odom"] and s.memory_present and s.inhibited


# ==================================================================== pairing
def test_a_plan_is_tracked_against_the_scan_it_names():
    """The measured speed in the command is the one from the plan's own scan, not the newest."""
    n = make_node()
    sensors_at(n, 1.0, ros_time(1.0))
    sensors_at(n, 5.0, ros_time(2.0))                # a newer scan, a very different speed
    n.on_plan(plan_msg(ros_time(1.0)))               # the plan of the OLD one
    assert len(n.tracker.calls) == 1
    assert n.tracker.calls[0][1] == pytest.approx(1.0)
    assert n._plan_unmatched == 0


def test_a_plan_that_arrives_before_its_scan_is_held_and_then_paired_exactly():
    n = make_node()
    sensors_at(n, 1.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(2.0)))               # its scan has not reached this node yet
    assert n.tracker.calls == [] and n._pending is not None
    assert n.pub.msgs == [], "a plan was tracked against a scan it was not made from"
    sensors_at(n, 5.0, ros_time(2.0))
    assert len(n.tracker.calls) == 1
    assert n.tracker.calls[0][1] == pytest.approx(5.0)
    assert n._plan_unmatched == 0 and n._pending is None


def test_a_held_plan_whose_scan_never_comes_is_tracked_one_scan_late_and_counted():
    """Bounded: waiting forever would skip a command, so the next scan takes it -- and says so."""
    n = make_node()
    sensors_at(n, 1.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(9.0)))               # a stamp no scan will ever carry
    sensors_at(n, 4.0, ros_time(2.0))
    assert len(n.tracker.calls) == 1
    assert n.tracker.calls[0][1] == pytest.approx(4.0)
    assert n._plan_unmatched == 1
    assert "never matched a held scan" in n._log.text("warning")


def test_an_unstamped_plan_takes_the_newest_scan_immediately_and_is_counted():
    """A zero header is not a time. Holding it would wait for a match that cannot happen."""
    n = make_node()
    sensors_at(n, 2.0, ros_time(1.0))
    n.on_plan(plan_msg(Time()))
    assert len(n.tracker.calls) == 1 and n.tracker.calls[0][1] == pytest.approx(2.0)
    assert n._plan_unmatched == 1


def test_the_command_is_stamped_with_the_scan_it_answers():
    n = make_node()
    sensors_at(n, 2.0, ros_time(3.25))
    n.on_plan(plan_msg(ros_time(3.25)))
    assert n.pub.msgs[-1].header.stamp == ros_time(3.25)


def test_the_yaw_rate_handed_to_the_tracker_is_this_scans_imu_mean():
    n = make_node()
    n.on_odom(odom_msg(2.0))
    n.on_imu(imu_msg())                              # gyro z = 0.3
    n.on_imu(imu_msg())
    n.on_scan(scan_msg(ros_time(1.0)))
    n.on_plan(plan_msg(ros_time(1.0)))
    assert n.tracker.calls[0][3] == pytest.approx(0.3)


# ==================================================================== when it stops
def test_no_plan_at_all_brakes_to_zero():
    n = make_node()
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_watchdog()
    assert n.pub.msgs and n.pub.msgs[-1].drive.speed == 0.0
    assert n.pub.msgs[-1].drive.steering_angle == 0.0
    assert "no plan on the plan topic" in n._log.text("warning")


def test_a_plan_younger_than_the_timeout_leaves_the_last_command_standing():
    n = make_node(plan_timeout=0.25)
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0)))
    assert len(n.pub.msgs) == 1 and n.pub.msgs[-1].drive.speed == pytest.approx(2.0)
    n._now += 0.2                                     # inside the timeout: the plan is still held
    n.on_watchdog()
    assert len(n.pub.msgs) == 1, "the watchdog published while the held plan was still fresh"


def test_the_plan_timeout_brakes_even_though_the_sensors_are_fine():
    n = make_node(plan_timeout=0.25)
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0)))
    for k in range(1, 12):                            # sensors keep arriving; plans do not
        n._now += DT
        n.on_odom(odom_msg(3.0)); n.on_imu(imu_msg())
        n.on_scan(scan_msg(ros_time(1.0 + k * DT)))
        n.on_watchdog()
    assert n.sensors.stale(n._now, check_scan=True) == []
    assert n.pub.msgs[-1].drive.speed == 0.0
    assert len(n.tracker.calls) == 1, "the controller kept tracking a plan past its timeout"


def test_stale_sensors_publish_zero_at_the_old_nodes_threshold():
    n = make_node(timeout=0.25)
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0)))
    n._now += 1.0                                     # everything stops
    n.on_scan(scan_msg(ros_time(2.0)))
    assert n._inhibited
    assert n.pub.msgs[-1].drive.speed == 0.0
    assert "no command" in n._log.text("warning")
    assert len(n._snapshots) == 1, "a stale scan was snapshotted and could be tracked against"


def test_a_plan_arriving_while_inhibited_is_not_tracked():
    n = make_node()
    sensors_at(n, 3.0, ros_time(1.0))
    n._now += 1.0
    n.on_scan(scan_msg(ros_time(2.0)))                # stale -> inhibit
    before = len(n.pub.msgs)
    n.on_plan(plan_msg(ros_time(2.0)))
    assert n.tracker.calls == [] and len(n.pub.msgs) == before


def test_recovery_clears_the_tracker_history_and_the_plan_from_before_the_gap():
    n = make_node()
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0)))
    n._now += 1.0
    n.on_scan(scan_msg(ros_time(2.0)))                # inhibited
    n._now += DT
    sensors_at(n, 3.0, ros_time(3.0))                 # sensors are back
    assert not n._inhibited
    assert n.tracker.resets == 1 and n.plan is None and n._pending is None
    assert "tracker history and held plan cleared" in n._log.text("info")


def test_reset_clears_the_same_things():
    n = make_node()
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0)))
    n.on_reset(object())
    assert n.tracker.resets == 1 and n.plan is None and n._snapshots == []


# ==================================================================== diagnostics
def test_the_diag_says_which_arm_which_friction_and_what_the_guard_is_doing():
    n = make_node()
    n.controller_arm = "fixed_low+clearance"
    n.grip = SimpleNamespace(mu=[0.73423])
    n.clearance = SimpleNamespace(cspec=SimpleNamespace(margin=0.2), last=None,
                                  update_scan=lambda _s: None)
    n.traction_arm = "on"; n.traction_state = "LOCK"
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0), seq=11))
    kv = {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}
    assert kv["arm"] == "fixed_low+clearance"
    assert kv["mu"] == "0.73423"
    assert kv["clearance_margin_m"] == "0.200"
    assert kv["traction"] == "on" and kv["traction_state"] == "LOCK"
    assert kv["plan_seq"] == "11" and kv["plan_checkpoint"].endswith("abcdef012345")
    assert kv["speed_cmd_mps"] == "2.0000"


# ==================================================================== the observation spec
def test_the_observation_spec_is_read_from_the_checkpoint(tmp_path):
    from f1sim.learn.model import ActorCritic, save_checkpoint
    spec = ObsSpec(n_beams=541, act_dim=8, range_max=12.0, v_max=7.0, scan_stack=2, hist_len=0)
    meta = dict(n_stack=spec.scan_stack, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                priv_dim=9, act_dim=spec.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(5)
        model = ActorCritic(**meta)
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, model, {"spec": spec.__dict__.copy()})
    n = cn.ControllerNode.__new__(cn.ControllerNode)
    n._log = Logger(); n.get_logger = lambda: n._log
    got = n._observation_spec(path, 0, 0.0, 0.0)
    assert got.n_beams == 541 and got.range_max == pytest.approx(12.0)
    assert got.v_max == pytest.approx(7.0)
    # and a parameter that contradicts it is refused rather than resolved by precedence
    with pytest.raises(ValueError, match="disagrees with the checkpoint"):
        n._observation_spec(path, 1081, 0.0, 0.0)
    # with no checkpoint the explicit parameters are the source, and the absence is said out loud
    bare = n._observation_spec("", 0, 0.0, 0.0)
    assert bare.n_beams == ObsSpec().n_beams
    assert "right only by accident" in n._log.text("warning")
