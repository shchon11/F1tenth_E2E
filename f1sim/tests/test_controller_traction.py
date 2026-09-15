"""The controller node's traction guard: the `traction` parameter, and what it does to `/drive`.

The guard moved here from `policy_node` with the split, unchanged; this file moved with it. Same
shape as `test_controller_arms.py` and `test_policy_sensor_contract.py`: no ROS graph, no device,
no vehicle. The node's real `on_odom` / `on_imu` / `on_core` / `on_scan` / `on_plan` / `_resume`
run against stub messages and a fake clock, and the assertions are about what the publisher
received.

The iLQR tracker is stubbed to a fixed command, because what is under test is the guard's wiring
and its authority over the published speed, not the tracker -- which has `test_mpc.py`, and whose
output through this node is pinned bit-for-bit in `test_graph_parity.py`.

What is worth pinning here, as opposed to in `test_traction_guard.py`:

* `traction:=off` is the default and must be *inert* -- not merely quiet. A guard that is off holds
  no state, reads no topic and cannot change a published speed.
* the guard is fed at the `/odom` rate, not the scan rate, because that is the rate
  `scripts/replay_traction.py` validated it at and because skipping wheel-speed samples is exactly
  what the windowed derivative cannot afford.
* body acceleration reaches it in **m/s^2**. This car publishes `linear_acceleration` in g, the
  node already detects and scales that, and handing the guard the unscaled number would divide
  every threshold by 9.81 silently.
* shaping happens before `speed_gain`, so a calibration gain does not rescale the guard's metres
  per second.
* a sensor gap resets the guard along with the observation history.
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

from builtin_interfaces.msg import Time                       # noqa: E402

pytest.importorskip("f1sim_interfaces.msg",
                    reason="f1sim_interfaces is not built; see f1sim_interfaces/README.md")

import f1sim_ros.controller_node as cn                        # noqa: E402
import f1sim_ros.deploy as deploy                             # noqa: E402
from f1sim.learn.obs import ObsSpec                           # noqa: E402
from f1sim_interfaces.msg import Plan                         # noqa: E402
from f1sim_ros.traction import LOCK, OK, SPIN, TractionGuard, TractionParams  # noqa: E402

G = deploy.G
DT = 0.02                                   # /odom period on this car
SCAN_DT = 0.025                             # scan period


# ==================================================================== stubs
class RecordingPub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


class FixedModel:
    """Returns an action that asks for a definite speed, so the guard's effect is visible."""

    def __init__(self, speed_frac=0.0, act_dim=2):
        self.speed_frac = speed_frac
        self.act_dim = act_dim

    def act(self, scan, pro, deterministic=True, c=None, h=None):
        # Same shape as `ActorCritic.act`: action, log probability, next hidden state. A stub
        # without memory returns None for the state, which is what a legacy checkpoint does.
        a = torch.zeros(1, self.act_dim)
        a[0, 1] = self.speed_frac           # maps to (frac + 1) / 2 * v_max
        return a, None, None


class RecordingGuard:
    """Records what the node fed the guard, so the wiring can be asserted directly."""

    def __init__(self):
        self.updates = []
        self.shapes = []
        self.resets = 0
        self.p = TractionParams()

    def update(self, t, wheel_speed, imu_ax, motor_current=None):
        self.updates.append((t, wheel_speed, imu_ax, motor_current))
        from f1sim_ros.traction import TractionState
        return TractionState(state=OK, t=t, wheel_speed=wheel_speed)

    def shape(self, cmd_speed, cmd_accel_hint=None):
        self.shapes.append(cmd_speed)
        return cmd_speed

    def reset(self):
        self.resets += 1


class Logger:
    def __init__(self):
        self.lines = []

    def _add(self, kind):
        return lambda s: self.lines.append((kind, s))

    def __getattr__(self, name):
        return self._add(name)

    def text(self, kind=None):
        return " | ".join(s for k, s in self.lines if kind is None or k == kind)


def ros_time(sec=1.0):
    t = Time(); t.sec = int(sec); t.nanosec = int((sec - int(sec)) * 1e9)
    return t


def _fixed_tracker(speed, steer=0.0):
    """The tracker, stubbed to a fixed (steer, speed). What the guard is given to shape."""
    class T:
        last_ref = None

        def __call__(self, a, v, cap, yaw_rate, delay=None):
            return torch.tensor([[steer, speed]])

        def reset(self, idx):
            pass
    return T()


def make_node(traction="on", overrides="", speed=0.0, cal=(0.0, 1.0, 1.0), v_max=8.0):
    """`speed` is what the stubbed tracker returns: 0 is the policy already braking, which is the
    case the lock release has to act on -- the command is cut and the wheel locked anyway."""
    spec = ObsSpec(n_beams=64, scan_stack=2, scan_stride=1, action_history=2, act_dim=8,
                   hist_len=0, range_max=10.0, v_max=v_max)
    n = cn.ControllerNode.__new__(cn.ControllerNode)
    n.device = torch.device("cpu")
    n.spec = spec
    n.tracker = _fixed_tracker(speed)
    n.delay = torch.tensor([0.035])
    n.pub = RecordingPub()
    n.pub_diag = RecordingPub()
    n.pub_viz = None; n.pub_viz_clear = None
    n.speed_cap = 8.0
    n.steer_max = 0.4189
    n.cal = cal
    n.timeout = 0.25
    n.plan_timeout = 0.25
    n.grip = None
    # The guard is what this file is about; the plan-geometry layer stays off so the only thing
    # shaping the published speed is the one under test.
    n.clearance = None; n._scan_geometry_checked = True
    n.controller_arm = "legacy"
    n._snapshots = []; n._pending = None
    n.plan = None; n.t_plan = None; n.plan_seq = None; n.plan_checkpoint = ""
    n._inhibited = False; n._last_inhibit_log = -1e9; n._braking = False
    n._plan_unmatched = 0; n._last_unmatched_log = -1e9; n._commands = 0
    n.last_cmd = (0.0, 0.0); n.last_t = None
    n._log = Logger()
    n._now = 100.0
    n.traction_arm = traction
    n.traction = deploy.build_traction_guard(traction, overrides)
    n.traction_state = "off" if n.traction is None else OK
    n.get_logger = lambda: n._log
    n.get_parameter = lambda name: SimpleNamespace(value=True)
    n.clock = lambda: n._now
    n.stamp_now = lambda: ros_time(n._now)
    n.sensors = deploy.SensorIntake(n.clock, n._log, n.timeout)
    return n


def imu_msg(ax_si=0.0, q=(1.0, 0.0, 0.0, 0.0)):
    """An IMU sample as this car publishes it: linear_acceleration in **g**, gravity on z."""
    return SimpleNamespace(
        header=SimpleNamespace(stamp=None),
        orientation=SimpleNamespace(w=q[0], x=q[1], y=q[2], z=q[3]),
        orientation_covariance=[0.0] * 9,
        angular_velocity=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        linear_acceleration=SimpleNamespace(x=ax_si / G, y=0.0, z=1.0))


def core_msg(current):
    return SimpleNamespace(header=SimpleNamespace(stamp=None),
                           state=SimpleNamespace(current_motor=current))


def odom_msg(v):
    return SimpleNamespace(twist=SimpleNamespace(twist=SimpleNamespace(
        linear=SimpleNamespace(x=v), angular=SimpleNamespace(z=0.0))))


STAMP = ros_time(1.0)


def scan_msg(n_beams=64, r=3.0):
    return SimpleNamespace(ranges=[r] * n_beams, range_max=10.0, angle_min=-2.356, angle_max=2.356,
                           header=SimpleNamespace(stamp=STAMP))


def plan_msg(seq=0):
    """The plan the policy would have published for that scan. Its stamp is the scan's, which is
    how the controller pairs the two; a constant stamp here means every plan matches the newest
    snapshot, which is what a steady stream does anyway."""
    m = Plan()
    m.header.stamp = STAMP
    m.plan = [0.0] * 8
    m.seq = int(seq)
    return m


def step(n, v, ax_si, current=None, scan=True):
    """One /odom + /sensors/imu/raw (+ /sensors/core) cycle, optionally a scan and its plan."""
    n._now += DT
    n.on_imu(imu_msg(ax_si))
    if current is not None:
        n.on_core(core_msg(current))
    n.on_odom(odom_msg(v))
    if scan:
        n.on_scan(scan_msg())
        n.on_plan(plan_msg(n._commands))
    return n.pub.msgs[-1].drive.speed if n.pub.msgs else None


def drive_cruise(n, v=6.0, samples=30, current=0.0):
    for _ in range(samples):
        step(n, v, 0.0, current)


def drive_lock(n, v0=6.0, a_wheel=-60.0, a_body=-6.0, samples=12, current=-25.0):
    """Wheel collapsing far faster than the body: the real car's brake lock."""
    out = []
    v = v0
    for _ in range(samples):
        v = max(0.0, v + a_wheel * DT)
        out.append(step(n, v, a_body, current))
    return out


# ==================================================================== the parameter
def test_off_is_the_default_and_installs_nothing():
    assert deploy.build_traction_guard("off") is None
    assert deploy.build_traction_guard("") is None
    n = make_node(traction="off")
    assert n.traction is None


def test_on_installs_the_validated_guard():
    g = deploy.build_traction_guard("on")
    assert isinstance(g, TractionGuard)
    assert g.p == TractionParams()


def test_an_unknown_arm_is_refused_rather_than_defaulted():
    for arm in ("ON", "true", "1", "yes", "enabled"):
        with pytest.raises(ValueError):
            deploy.build_traction_guard(arm)


def test_thresholds_are_reachable_from_the_launch_line():
    g = deploy.build_traction_guard("on", "lock_rate=22.5, spin_rate=9 lock_persist=2")
    assert g.p.lock_rate == pytest.approx(22.5)
    assert g.p.spin_rate == pytest.approx(9.0)
    assert g.p.lock_persist == 2 and isinstance(g.p.lock_persist, int)
    assert g.p.lock_accel == TractionParams().lock_accel      # untouched fields keep their default


@pytest.mark.parametrize("bad", ["lock_rate", "lock_rate=", "nonsense=1", "lock_rate=high",
                                 "lock_rate=18 spin_rate=-1"])
def test_a_bad_traction_params_string_raises(bad):
    with pytest.raises(ValueError):
        deploy.build_traction_guard("on", bad)


# ==================================================================== off is inert
def test_with_the_guard_off_the_published_speed_is_untouched_through_a_lock():
    off, on = make_node(traction="off"), make_node(traction="on")
    for n in (off, on):
        drive_cruise(n)
    a = drive_lock(off)
    b = drive_lock(on)
    assert all(x == pytest.approx(0.0) for x in a), a          # the policy asked for zero speed
    assert max(b) > 0.5, b                                     # the guard releases the brake
    assert "traction" not in off._log.text()


# ==================================================================== detection through the node
def test_a_lock_is_detected_from_odom_and_the_scaled_imu_and_logged_with_the_numbers():
    n = make_node()
    drive_cruise(n)
    drive_lock(n)
    st = n.traction.state
    assert st.locks == 1
    # the body acceleration the guard saw is SI, not the g the message carried
    assert st.body_accel == pytest.approx(-6.0, abs=1.5)
    text = n._log.text("info")
    assert "traction lock" in text and "m/s^2" in text


def test_the_guard_sees_metres_per_second_squared_not_g():
    """The failure this guards against is silent: at 1/9.81 scale no threshold is ever reached and
    the guard simply never fires."""
    n = make_node()
    drive_cruise(n)
    n.on_imu(imu_msg(-9.81))
    assert n.sensors.ax_body == pytest.approx(-9.81, abs=1e-3)
    assert n.sensors.accel_scale == pytest.approx(G)


def test_the_guard_is_fed_every_odom_sample_and_shaped_once_per_scan():
    """/odom is 50 Hz and scans are 40 Hz. Feeding the guard from the scan (or the plan) would
    drop one wheel speed in five, which is the sample a 40 ms lock lives in; shaping per /odom
    would rate-limit a command that has not been published yet."""
    n = make_node()
    n.traction = rec = RecordingGuard()
    for i in range(21):                     # three odom samples per scan, an extreme version
        n._now += DT
        n.on_imu(imu_msg(-1.5))
        n.on_odom(odom_msg(6.0 + 0.01 * i))
        if i % 3 == 0:
            n.on_scan(scan_msg())
            n.on_plan(plan_msg(i))
    assert len(rec.updates) == 21
    assert len(rec.shapes) == 7
    assert [u[1] for u in rec.updates] == [pytest.approx(6.0 + 0.01 * i) for i in range(21)]
    assert all(u[2] == pytest.approx(-1.5, abs=1e-3) for u in rec.updates)


def test_a_spin_needs_the_core_current_when_the_topic_is_there():
    """`on_core` is what makes the node's spin gate match the replayed configuration."""
    n = make_node()
    drive_cruise(n, v=1.0, current=60.0)
    for i in range(8):
        step(n, 1.0 + 25.0 * (i + 1) * DT, 4.0, 60.0)
    assert n.traction.state.spins == 1
    quiet = make_node()
    drive_cruise(quiet, v=1.0, current=-30.0)
    for i in range(8):
        step(quiet, 1.0 + 25.0 * (i + 1) * DT, 4.0, -30.0)
    assert quiet.traction.state.spins == 0


def test_a_stale_imu_or_current_reaches_the_guard_as_unknown_not_as_its_last_value():
    """Handing the guard a body acceleration from half a second ago is worse than handing it
    nothing: it holds its filters over a None, but it would compare a live wheel against a dead
    body."""
    n = make_node()
    n.timeout = n.sensors.timeout = 0.10
    n.traction = rec = RecordingGuard()
    n.on_imu(imu_msg(-3.0)); n.on_core(core_msg(40.0))
    n._now += DT; n.on_odom(odom_msg(5.0))
    assert rec.updates[-1][2] == pytest.approx(-3.0, abs=1e-3)
    assert rec.updates[-1][3] == pytest.approx(40.0)
    for _ in range(8):                      # 160 ms of /odom with no IMU and no core
        n._now += DT
        n.on_odom(odom_msg(5.0))
    assert rec.updates[-1][2] is None and rec.updates[-1][3] is None


def test_a_spin_is_still_detectable_with_no_core_topic_at_all():
    """One recording has no `/sensors/core` (20260725-152213), and `vesc_msgs` may not be
    importable. The current is then unknown, which the guard treats as no information rather than
    as no torque -- more willing to call a spin, not less."""
    n = make_node()
    drive_cruise(n, v=1.0, current=None)
    for i in range(8):
        step(n, 1.0 + 25.0 * (i + 1) * DT, 4.0, None)
    assert n.traction.state.spins == 1
    assert n.sensors.motor_current is None


# ==================================================================== shaping the command
def test_the_lock_release_raises_the_published_speed_but_not_past_its_authority():
    n = make_node()
    drive_cruise(n)
    out = drive_lock(n)
    assert max(out) > 0.5
    assert max(out) <= TractionParams().release_max + 1e-6


def test_the_published_speed_never_exceeds_the_speed_cap():
    n = make_node(speed=8.0)                # the tracker already asks for v_max
    n.speed_cap = 3.0
    drive_cruise(n)
    out = drive_lock(n)
    assert max(out) <= 3.0 + 1e-9


def test_shaping_happens_before_the_speed_gain_calibration():
    """`speed_gain` converts the car's metres per second into the number the stack wants. The guard
    reasons in the former; shaping the latter would scale its authority by the gain."""
    outs = {}
    for gain in (1.0, 2.0):
        n = make_node(cal=(0.0, 1.0, gain))
        drive_cruise(n)
        outs[gain] = max(drive_lock(n))
    assert outs[1.0] > 0.5
    assert outs[2.0] == pytest.approx(outs[1.0] / 2.0, rel=1e-6)


def test_shaping_is_neutral_while_nothing_is_slipping():
    n = make_node(speed=4.0, v_max=8.0)
    drive_cruise(n, samples=60)
    speeds = [m.drive.speed for m in n.pub.msgs]
    assert speeds and all(s == pytest.approx(4.0) for s in speeds)
    assert n.traction.state.state == OK


# ==================================================================== lifecycle
def test_a_sensor_gap_resets_the_guard_with_the_observation_history():
    n = make_node()
    drive_cruise(n)
    drive_lock(n)
    assert n.traction.state.locks == 1
    n._inhibited = True
    n._resume()
    st = n.traction.state
    assert st.locks == 0 and st.spins == 0 and st.state == OK
    assert st == n.traction.state and st.t == 0.0
    assert "tracker history and held plan cleared" in n._log.text()


def test_the_guard_is_reset_not_recreated_so_its_parameters_survive_a_gap():
    n = make_node(overrides="lock_rate=25")
    n.traction = rec = RecordingGuard()
    n._inhibited = True
    n._resume()
    assert rec.resets == 1
    n2 = make_node(overrides="lock_rate=25")
    n2._inhibited = True
    n2._resume()
    assert n2.traction.p.lock_rate == pytest.approx(25.0)


def test_a_core_message_with_a_nonfinite_current_is_ignored():
    n = make_node()
    n.on_core(core_msg(12.0))
    assert n.sensors.motor_current == pytest.approx(12.0)
    n.on_core(core_msg(float("nan")))
    assert n.sensors.motor_current == pytest.approx(12.0)
    n.on_core(SimpleNamespace(state=None))
    assert n.sensors.motor_current == pytest.approx(12.0)
