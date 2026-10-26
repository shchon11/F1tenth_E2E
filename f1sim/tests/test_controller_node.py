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
from f1sim_interfaces.msg import Perception, Plan, PolicyState            # noqa: E402

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
    n._perception = {}; n._floor_from_frontend = 0; n._warned = set()
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


def perception_msg(stamp, floor=None, source="frontend"):
    m = Perception()
    m.header.stamp = stamp
    m.floor = [] if floor is None else [float(x) for x in floor]
    m.attitude = [0.0, 0.0]
    m.attitude_ok = True
    m.source = source
    return m


class FloorSpyArm:
    """A clearance arm that records what the controller handed it, and for which scan."""

    def __init__(self, gate=True, mode="brake"):
        self.cspec = SimpleNamespace(margin=0.2, floor_gate=gate, floor_gate_mode=mode)
        self.last = None
        self.scans = []
        self.floors = []
        self.attitudes = []

    def update_scan(self, s):
        self.scans.append(s)

    def update_attitude(self, g, a, v):
        self.attitudes.append((g, a, v))

    def set_floor(self, p):
        self.floors.append(p)


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
    # `floor_gate` off, which is `ClearanceSpec`'s default and the arm that ships: the diag has to
    # say so, and no attitude or front-end path may run.
    n.clearance = SimpleNamespace(cspec=SimpleNamespace(margin=0.2, floor_gate=False,
                                                        floor_gate_mode="brake"),
                                  last=None, update_scan=lambda _s: None)
    n.traction_arm = "on"; n.traction_state = "LOCK"
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0), seq=11))
    kv = {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}
    assert kv["arm"] == "fixed_low+clearance"
    assert kv["mu"] == "0.73423"
    assert kv["clearance_margin_m"] == "0.200"
    assert kv["clearance_floor_gate"] == "off" and kv["clearance_floor_source"] == ""
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


# ============================================ the policy's perception, across the node split
#
# The monolithic node had the sensor front-end and the clearance arm in one process, so the gate
# could read `frontend.last_probs` directly (`feat/floor-mask`). The split put the network in
# `policy_node` and the arm here, so it arrives on `/f1sim/policy/perception`, stamped with the
# scan it belongs to and paired the way the plan is. These tests are about that pairing: nothing
# below is on unless `floor_gate` is, which is off by default.
def test_the_floor_gate_reads_the_policys_front_end_for_the_scan_it_is_tracking():
    n = make_node()
    n.controller_arm = "fixed_low+clearance"
    n.clearance = FloorSpyArm()
    want = [0.25] * 64
    n.on_perception(perception_msg(ros_time(1.0), floor=want))
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0)))
    assert len(n.clearance.floors) == 1
    assert n.clearance.floors[0].shape == (1, 64)
    assert float(n.clearance.floors[0][0, 0]) == pytest.approx(0.25)
    # the geometric estimator is advanced anyway, so it never integrates across a gap in the topic
    assert len(n.clearance.attitudes) == 1
    kv = {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}
    assert kv["clearance_floor_gate"] == "brake" and kv["clearance_floor_source"] == "frontend"


def test_a_perception_for_another_scan_never_gates_this_one():
    """The whole reason it is stamped. A message for a scan this command is not tracking is not
    'the newest answer', it is an answer about a different picture."""
    n = make_node()
    n.controller_arm = "fixed_low+clearance"
    n.clearance = FloorSpyArm()
    n.on_perception(perception_msg(ros_time(9.0), floor=[0.9] * 64))
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0)))
    assert n.clearance.floors == [], "a perception from another scan reached this command"
    kv = {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}
    assert kv["clearance_floor_source"] == "geometry"


def test_no_perception_at_all_is_the_geometric_gate():
    """A policy node that publishes nothing -- every published checkpoint, none of which carries a
    front-end -- leaves the gate exactly as `feat/floor-mask` shipped it."""
    n = make_node()
    n.controller_arm = "fixed_low+clearance"
    n.clearance = FloorSpyArm()
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0)))
    assert n.clearance.floors == [] and len(n.clearance.attitudes) == 1
    assert {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}[
        "clearance_floor_source"] == "geometry"


def test_the_gate_off_touches_neither_the_attitude_nor_the_front_end():
    n = make_node()
    n.controller_arm = "fixed_low+clearance"
    n.clearance = FloorSpyArm(gate=False)
    n.on_perception(perception_msg(ros_time(1.0), floor=[0.9] * 64))
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0)))
    assert n.clearance.floors == [] and n.clearance.attitudes == []
    assert len(n.clearance.scans) == 1, "the scan still reaches the arm"


def test_a_perception_of_the_wrong_width_is_named_and_ignored():
    """Same beam count or nothing: the two are built from one `ObsSpec`, so a disagreement means
    the policy and the controller are not reading the same observation, and gating on it would be
    a silently rotated floor."""
    n = make_node()
    n.controller_arm = "fixed_low+clearance"
    n.clearance = FloorSpyArm()
    n.on_perception(perception_msg(ros_time(1.0), floor=[0.4] * 32))
    sensors_at(n, 3.0, ros_time(1.0))
    n.on_plan(plan_msg(ros_time(1.0)))
    assert n.clearance.floors == []
    assert "32 floor values" in n._log.text("warning")
    n.on_perception(perception_msg(ros_time(2.0), floor=[0.4] * 32))
    sensors_at(n, 3.0, ros_time(2.0))
    n.on_plan(plan_msg(ros_time(2.0), seq=1))
    assert n._log.text("warning").count("32 floor values") == 1, "warned once, not per scan"


def test_the_perception_store_is_bounded():
    n = make_node()
    for i in range(cn.SNAPSHOT_MAX * 3):
        n.on_perception(perception_msg(ros_time(1.0 + i), floor=[0.1] * 64))
    assert len(n._perception) <= cn.SNAPSHOT_MAX


# ==================================================================== the plan follower
#
# 2026-10-19, the user: "우리도 그냥 map 쓰자". A checkpoint trained with --plan-follower map records it
# in its contract, and the car has to follow its plans the way the simulator did: through
# `map_controller.MAPPlanTracker`, with the same measurements handed over.
def _small_lut():
    from f1sim.map_controller import GRIP_LEVELS, SteerLUT
    steers = torch.linspace(0.0, 0.4189, 12)
    vels = torch.linspace(0.5, 10.0, 8)
    # (grip, steer, v) -> lateral acceleration, rising in steer: a kinematic car, more grip, more turn
    g = torch.tensor(GRIP_LEVELS)[:, None, None]
    table = torch.tan(steers)[None, :, None] * (vels[None, None, :] ** 2) / 0.3302 * g.clamp(max=1.0)
    return SteerLUT(steers, vels, table), {"mu": 1.0489, "mu_f_scale": 0.92}


def _follower_node(monkeypatch, contract, arm="legacy"):
    import f1sim.map_controller as mc
    from f1sim.mpc import PlanTracker, contract_spec
    lut, nominal = _small_lut()
    monkeypatch.setattr(mc, "config_lut", lambda cfg=None, device="cpu": (lut, nominal))
    n = make_node()
    n.contract = dict(contract)
    n.controller_arm = arm
    n.tracker = PlanTracker(1, torch.device("cpu"), 0.3302, 0.4189, 8.0, spec=contract_spec(n.contract))
    return n, lut, nominal


def test_the_plan_follower_is_the_checkpoints_unless_overridden(monkeypatch):
    from f1sim.map_controller import POLICY_MAP, MAPPlanTracker
    lin = {"speed_mode": "linear", "speed_command": "tracker"}
    on_map = dict(lin, plan_follower="map", plan_follower_map={"q_l1": 0.4})
    n, _, _ = _follower_node(monkeypatch, on_map)
    n._install_follower("auto", 1.0)
    assert n.follower == "map" and isinstance(n.tracker, MAPPlanTracker)
    assert n.tracker.map.p.q_l1 == pytest.approx(0.4)                    # the contract's numbers...
    assert n.tracker.map.p.l1_point == POLICY_MAP["l1_point"]            # ... over POLICY_MAP
    n, _, _ = _follower_node(monkeypatch, lin)
    n._install_follower("auto", 1.0)
    assert n.follower == "mpc" and not isinstance(n.tracker, MAPPlanTracker)
    n, _, _ = _follower_node(monkeypatch, on_map)
    n._install_follower("mpc", 1.0)
    assert n.follower == "mpc" and not isinstance(n.tracker, MAPPlanTracker)
    n, _, _ = _follower_node(monkeypatch, lin)
    n._install_follower("map", 1.0)                                      # an iLQR policy on MAP: POLICY_MAP
    assert isinstance(n.tracker, MAPPlanTracker) and n.tracker.map.p.q_l1 == pytest.approx(POLICY_MAP["q_l1"])
    with pytest.raises(ValueError, match="one of auto, mpc, map"):
        _follower_node(monkeypatch, lin)[0]._install_follower("pp", 1.0)


def test_map_refuses_an_arm_that_shapes_the_ilqr(monkeypatch):
    on_map = {"speed_mode": "linear", "speed_command": "tracker", "plan_follower": "map"}
    n, _, _ = _follower_node(monkeypatch, on_map, arm="fixed_low")
    with pytest.raises(ValueError, match="controller:=legacy"):
        n._install_follower("auto", 1.0)


def test_the_car_follows_a_map_plan_as_the_simulator_does(monkeypatch):
    """The node's command for a plan is the simulator's `MAPPlanTracker` command for the same plan,
    speed, cap, yaw rate (this scan's gyro mean) and delay (the node's `cmd_delay`), at the nominal
    grip -- the convention `F1VecEnv.step` calls it with."""
    from f1sim.map_controller import POLICY_MAP, MAPParams, MAPPlanTracker
    from f1sim.mpc import PlanTracker, contract_spec
    on_map = {"speed_mode": "linear", "speed_command": "tracker", "plan_follower": "map"}
    n, lut, nominal = _follower_node(monkeypatch, on_map)
    n._install_follower("auto", 1.0)
    ref = MAPPlanTracker(PlanTracker(1, torch.device("cpu"), 0.3302, 0.4189, 8.0, spec=contract_spec(on_map)),
                         lut, MAPParams.from_dict(dict(POLICY_MAP)), nominal)
    ref.map_rows = torch.ones(1, dtype=torch.bool)
    ref.P = {"mu": torch.tensor([nominal["mu"]]), "mu_f_scale": torch.tensor([nominal["mu_f_scale"]])}
    plans = ([0.5, -0.2, 0.1, 0.3, 0.2, 0.4, 0.0, 0.0], [-0.3, -0.4, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0])
    for i, plan in enumerate(plans):
        stamp = ros_time(1.0 + 0.025 * i)
        sensors_at(n, 3.0, stamp)
        n.on_plan(plan_msg(stamp, seq=i, values=plan))
        got = n.last_cmd
        ref.push_imu(torch.tensor([[0.0]]))
        want = ref(torch.tensor([plan]), torch.tensor([3.0]), torch.tensor([n.speed_cap]),
                   torch.tensor([0.3]), delay=n.delay)[0]
        assert got[0] == pytest.approx(float(want[0]), abs=1e-6)
        assert got[1] == pytest.approx(float(want[1]), abs=1e-6)
        assert n.tracker.last_map is not None and ref.last_map is not None      # MAP drove, not the iLQR
    kv = {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}
    assert kv["plan_follower"] == "map"


def test_the_speed_knob_slows_a_wp4_plan_as_the_simulator_does(monkeypatch):
    """The venue's knob on a v2 (wp4) policy under MAP: the node's command is the simulator's `MAPPlanTracker`
    command for the plan the simulator's own knob makes (`mpc.scale_plan_speeds`), every waypoint speed x the
    factor. Until 2026-10-23 the node scaled the linear layout's columns, v_3 and v_4 of a wp4 plan."""
    from f1sim import mpc
    from f1sim.map_controller import POLICY_MAP, MAPParams, MAPPlanTracker
    from f1sim.mpc import PlanTracker, contract_spec
    on_map = {"speed_mode": "wp4", "speed_command": "tracker", "plan_follower": "map"}
    n, lut, nominal = _follower_node(monkeypatch, on_map)
    n._install_follower("auto", 1.0)
    n.plan_speed_scale = 0.8
    ref = MAPPlanTracker(PlanTracker(1, torch.device("cpu"), 0.3302, 0.4189, 8.0, spec=contract_spec(on_map)),
                         lut, MAPParams.from_dict(dict(POLICY_MAP)), nominal)
    ref.map_rows = torch.ones(1, dtype=torch.bool)
    ref.P = {"mu": torch.tensor([nominal["mu"]]), "mu_f_scale": torch.tensor([nominal["mu_f_scale"]])}
    plan = [0.1, 0.05, -0.05, 0.0, -0.1, 0.0, 0.2, 0.25]                  # 3.6, 4.0, 4.8, 5.0 m/s
    stamp = ros_time(1.0)
    sensors_at(n, 4.0, stamp)
    n.on_plan(plan_msg(stamp, seq=0, values=plan))
    got = n.last_cmd
    ref.push_imu(torch.tensor([[0.0]]))
    scaled = mpc.scale_plan_speeds(torch.tensor([plan]), "wp4", 0.8)
    want = ref(scaled, torch.tensor([4.0]), torch.tensor([n.speed_cap]), torch.tensor([0.3]), delay=n.delay)[0]
    assert got[0] == pytest.approx(float(want[0]), abs=1e-6)
    assert got[1] == pytest.approx(float(want[1]), abs=1e-6)
    assert got[1] == pytest.approx(0.8 * 3.6, abs=0.15)                   # MAP drives v_1, now x 0.8


# ==================================================================== the speed limit for a car ahead
# `opponent_speed_limit` (2026-10-25, `f1sim.opponent_speed_limit`): the policy node sends its own nearest-car
# reading with each scan's perception message; the controller holds the speed command to the limit it sets for the
# scan it is tracking. Off by default; with it on and no reading for a scan, that command is not limited.
def car_ahead_perception(stamp, dx_lidar=0.6, dy=0.0, closing=1.5, v_lat=0.0):
    m = perception_msg(stamp)
    m.nearest_car = [4.0, dx_lidar / 5.0, dy / 2.0, closing / 5.0, v_lat / 2.0]
    return m


def test_the_speed_limit_holds_the_command_for_a_car_ahead_in_this_scan():
    from f1sim.opponent_speed_limit import speed_limit
    n = make_node(speed=3.0)
    n.opponent_speed_limit = True; n._opp_limited = 0; n._opp_no_reading = 0
    stamp = ros_time(1.0)
    per = car_ahead_perception(stamp)
    n.on_perception(per)
    sensors_at(n, 2.5, stamp)
    n.on_plan(plan_msg(stamp))
    want = float(speed_limit(torch.tensor([list(per.nearest_car)]), torch.tensor([2.5]), deploy.LIDAR_MOUNT_X, 0.0)[0])
    assert want < 3.0
    assert n.last_cmd[1] == pytest.approx(want, abs=1e-5)
    kv = {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}
    assert kv["opponent_speed_limit"] == "on" and kv["opponent_limited_commands"] == "1"


def test_no_reading_for_this_scan_leaves_the_command_and_is_counted():
    n = make_node(speed=3.0)
    n.opponent_speed_limit = True; n._opp_limited = 0; n._opp_no_reading = 0
    n.on_perception(car_ahead_perception(ros_time(0.5)))          # another scan's reading
    stamp = ros_time(1.0)
    sensors_at(n, 2.5, stamp)
    n.on_plan(plan_msg(stamp))
    assert n.last_cmd[1] == pytest.approx(3.0)
    kv = {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}
    assert kv["opponent_no_reading_commands"] == "1" and kv["opponent_limited_commands"] == "0"


def test_the_speed_limit_off_ignores_the_reading():
    n = make_node(speed=3.0)
    n.opponent_speed_limit = False; n._opp_limited = 0; n._opp_no_reading = 0
    stamp = ros_time(1.0)
    n.on_perception(car_ahead_perception(stamp))
    sensors_at(n, 2.5, stamp)
    n.on_plan(plan_msg(stamp))
    assert n.last_cmd[1] == pytest.approx(3.0)
    kv = {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}
    assert kv["opponent_speed_limit"] == "off"


def test_a_reading_from_the_scan_just_before_stands_in_when_this_scans_is_not_in_yet():
    """The plan and the perception are two topics with no order between them: a plan can be tracked before its own
    scan's perception arrives. A reading from the last 0.1 s stands in, and is counted."""
    n = make_node(speed=3.0)
    n.opponent_speed_limit = True; n._opp_limited = 0; n._opp_no_reading = 0
    n.on_perception(car_ahead_perception(ros_time(0.975)))      # the scan before (25 ms)
    stamp = ros_time(1.0)
    sensors_at(n, 2.5, stamp)
    n.on_plan(plan_msg(stamp))
    assert n.last_cmd[1] < 3.0
    kv = {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}
    assert kv["opponent_limited_commands"] == "1" and kv["opponent_previous_scan_readings"] == "1"


def test_a_car_the_limit_holds_stays_held_out_to_lane_exit_and_no_reading_lets_go():
    """2026-10-25: once the limit applies to a car, later scans keep it while its offset is under `lane_exit` (0.5 m);
    the same offset is out of our lane for a car the limit was not holding, and a scan without a reading lets go."""
    n = make_node(speed=3.0)
    n.opponent_speed_limit = True; n._opp_limited = 0; n._opp_no_reading = 0; n._opp_held = False
    for t, dy, limited in ((1.0, 0.45, False), (1.1, 0.0, True), (1.2, 0.45, True)):
        s = ros_time(t)
        n.on_perception(car_ahead_perception(s, dy=dy))
        sensors_at(n, 2.5, s)
        n.on_plan(plan_msg(s))
        assert (n.last_cmd[1] < 3.0) == limited and n._opp_held == limited, (t, dy)
    s = ros_time(1.5)                                               # nothing from the last 0.1 s
    sensors_at(n, 2.5, s)
    n.on_plan(plan_msg(s))
    assert n.last_cmd[1] == pytest.approx(3.0) and not n._opp_held


# ==================================================================== the speed limit for an obstacle on the plan's path
# `obstacle_speed_limit` (2026-10-26, `f1sim.obstacle_speed_limit`): small isolated clusters of THIS scan on the path the
# plan draws hold the speed command to the speed the car can still stop from. Off by default.
def box_scan_msg(stamp, n_beams=64, face=1.2, half=0.25, far=5.0):
    """a box `face` m straight ahead of the LiDAR (half-width `half`), a wall `far` m away everywhere else"""
    import math
    r = []
    for i in range(n_beams):
        a_ = -2.356 + i * (4.712 / (n_beams - 1))
        c, s_ = math.cos(a_), math.sin(a_)
        r.append(face / c if c > 0 and abs(face / c * s_) <= half else far)
    return SimpleNamespace(ranges=r, range_max=10.0, angle_min=-2.356, angle_max=2.356,
                           header=SimpleNamespace(stamp=stamp))


@pytest.mark.parametrize("on, box", [(True, True), (True, False), (False, True)])
def test_the_obstacle_limit_holds_the_command_for_a_box_on_the_plan(on, box):
    from f1sim.mpc import PlanSpec
    n = make_node(speed=3.0)
    n.opponent_speed_limit = False; n._opp_limited = 0; n._opp_no_reading = 0
    n.obstacle_speed_limit = on; n._obs_limited = 0; n._scan_angles = None
    n.tracker.spec = PlanSpec()                                 # a zero plan: straight, 3.75 m long
    stamp = ros_time(1.0)
    n.on_odom(odom_msg(2.5)); n.on_imu(imu_msg())
    n.on_scan(box_scan_msg(stamp) if box else scan_msg(stamp, r=5.0))
    n.on_plan(plan_msg(stamp))
    import math
    limited = on and box
    # box face 1.2 m from the LiDAR, 1.497 m from the reference point: the first path point within 0.19 m is s = 1.4
    want = math.sqrt(2 * 3.0 * (1.4 - 0.45 - 0.1 - 2.5 * 0.2)) if limited else 3.0
    assert n.last_cmd[1] == pytest.approx(want, abs=1e-4)
    kv = {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}
    assert kv["obstacle_speed_limit"] == ("on" if on else "off")
    assert kv["obstacle_limited_commands"] == ("1" if limited else "0")


def test_the_obstacle_limit_leaves_the_policys_car_to_the_car_limit_also_when_that_is_off():
    """2026-10-26: returns at the policy's own nearest-car reading are the car limit's, with or without the car limit on,
    and a reading from the scan just before stands in for this scan's as it does for the car limit"""
    from f1sim.mpc import PlanSpec
    for car in (False, True):
        n = make_node(speed=3.0)
        n.opponent_speed_limit = False; n._opp_limited = 0; n._opp_no_reading = 0
        n.obstacle_speed_limit = True; n._obs_limited = 0; n._scan_angles = None
        n.tracker.spec = PlanSpec()
        if car:
            n.on_perception(car_ahead_perception(ros_time(0.975), dx_lidar=1.4))   # the scan before: a car where the box is
        stamp = ros_time(1.0)
        n.on_odom(odom_msg(2.5)); n.on_imu(imu_msg())
        n.on_scan(box_scan_msg(stamp))
        n.on_plan(plan_msg(stamp))
        assert (n.last_cmd[1] == pytest.approx(3.0)) == car, car
        assert n._opp_limited == 0


def test_the_limits_lag_is_a_parameter_and_a_longer_one_holds_harder():
    """`speed_limit_lag` (2026-10-26): the command-to-brakes lag both limits assume; the competition car's speed loop had
    ~0.1 s more than the simulator, so the node defaults to 0.3 s. A longer lag holds a car closing in harder."""
    from dataclasses import replace
    from f1sim.opponent_speed_limit import DEFAULT, speed_limit
    got = {}
    for lag in (0.2, 0.3):
        n = make_node(speed=3.0)
        n.opponent_speed_limit = True; n._opp_limited = 0; n._opp_no_reading = 0; n.speed_limit_lag = lag
        stamp = ros_time(1.0)
        per = car_ahead_perception(stamp, dx_lidar=1.2, closing=1.5)
        n.on_perception(per)
        sensors_at(n, 2.5, stamp)
        n.on_plan(plan_msg(stamp))
        want = float(speed_limit(torch.tensor([list(per.nearest_car)]), torch.tensor([2.5]), deploy.LIDAR_MOUNT_X, 0.0,
                                 p=replace(DEFAULT, lag=lag))[0])
        assert n.last_cmd[1] == pytest.approx(min(3.0, want), abs=1e-5)
        kv = {k.key: k.value for k in n.pub_diag.msgs[-1].status[0].values}
        assert kv["speed_limit_lag_s"] == f"{lag:.2f}"
        got[lag] = n.last_cmd[1]
    assert got[0.3] < got[0.2]
