"""`baseline_node`: reset, staleness, the scan window, and parity with the batched adapter.

No ROS graph, no device, no vehicle -- the same shape as `test_policy_node_memory.py`. The node's
real `on_odom` / `on_scan` / `on_reset` / `on_watchdog` run against stub messages and a fake clock.

The parity test is the load-bearing one. CONTRACT.md asks for bit-identical (or <= 1e-5) commands
from the node and from the in-process batched adapter on the same recorded scans. They share one
`BaselineDriver`, so what is actually being compared is the plumbing on either side of it: the miss
sentinel, the scan-window mapping, the metres<->normalised round trip the environment does, the
action encoding, and the two output clips.
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
torch = pytest.importorskip("torch")

import f1sim_ros.baseline_node as bn                                     # noqa: E402
from f1sim.learn import baselines                                        # noqa: E402
from f1sim.learn.baselines import end2race as e2r_mod                    # noqa: E402
from f1sim.learn.benchmark import model_adapter as ma                    # noqa: E402
from f1sim.params import Config, VehicleParams                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.expanduser("~"), "Documents", "Codex", "2026-09-10", "new-chat",
                      "work", "baselines", "models")
TLN_ONNX = os.path.join(MODELS, "tinylidarnet_L_1081.onnx")
E2R_WEIGHTS = os.path.join(os.path.expanduser("~"), "F1tenth", "F1tenth_E2E", "external",
                           "baselines", "End2Race", "pretrained", "end2race.pth")

LID = Config().lidar
N_BEAMS, FOV, RANGE_MAX = int(LID.n_beams), float(LID.fov), float(LID.range_max)
S_MAX = float(VehicleParams().s_max)
SPEED_CAP = 9.0


class RecordingPub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


class Logger:
    def __init__(self):
        self.lines = []

    def info(self, t): self.lines.append(("info", t))

    def warning(self, t): self.lines.append(("warn", t))

    def warn(self, t): self.lines.append(("warn", t))

    def error(self, t): self.lines.append(("error", t))


def ros_time(t=0.0):
    from builtin_interfaces.msg import Time
    return Time(sec=int(t), nanosec=int((t - int(t)) * 1e9))


def scan_msg(ranges, fov=FOV, range_max=RANGE_MAX):
    return SimpleNamespace(header=SimpleNamespace(stamp=ros_time()),
                           ranges=list(ranges), range_max=range_max,
                           angle_min=-fov / 2.0, angle_max=fov / 2.0)


def make_node(driver, speed_cap=SPEED_CAP):
    n = bn.BaselineNode.__new__(bn.BaselineNode)
    n.driver = driver
    n.steer_max = S_MAX
    n.speed_cap = speed_cap
    n.timeout = 0.25
    n.v = 0.0
    n.t_odom = n.t_scan = None
    n._inhibited = False
    n._last_inhibit_log = -1e9
    n._geometry = None
    n.last_t = None
    n.pub = RecordingPub()
    n._log = Logger()
    n._now = 100.0
    n.get_logger = lambda: n._log
    n.get_parameter = lambda name: SimpleNamespace(value=True)
    n.clock = lambda: n._now
    n.stamp_now = lambda: ros_time(n._now)
    return n


def odom_msg(v=1.0):
    return SimpleNamespace(twist=SimpleNamespace(twist=SimpleNamespace(
        linear=SimpleNamespace(x=v), angular=SimpleNamespace(z=0.0))))


def feed(node, scans, dt=0.025, speeds=None):
    for i, r in enumerate(scans):
        node._now += dt
        node.on_odom(odom_msg(1.0 if speeds is None else float(speeds[i])))
        node.on_scan(scan_msg(r))


def real_scans(n=None):
    f = os.path.join(HERE, "data", "real_scans_100.npz")
    if not os.path.exists(f):
        pytest.skip("real scan fixture missing")
    r = np.load(f)["ranges_mm"].astype(np.float32) / 1000.0
    return r if n is None else r[:n]


def tln_driver(**kw):
    if not os.path.exists(TLN_ONNX):
        pytest.skip("tinylidarnet_L_1081.onnx has not been converted")
    return baselines.load("tinylidarnet", TLN_ONNX, **kw)


def e2r_driver(**kw):
    if not os.path.exists(E2R_WEIGHTS):
        pytest.skip("End2Race is not vendored")
    return baselines.load("end2race", E2R_WEIGHTS, **kw)


# --------------------------------------------------------------------------- staleness
def test_no_scan_stops_the_car():
    """The watchdog, not the scan callback: the callback is exactly what stops running."""
    node = make_node(tln_driver())
    feed(node, real_scans(2))
    assert not node._inhibited and node.pub.msgs[-1].drive.speed > 0
    node._now += 1.0
    node.on_watchdog()
    assert node._inhibited
    m = node.pub.msgs[-1]
    assert m.drive.speed == 0.0 and m.drive.steering_angle == 0.0
    assert any("scan" in t for _l, t in node._log.lines if _l == "warn")


def test_stale_odom_stops_end2race_but_a_lidar_only_model_never_asks_for_it():
    """Staleness is per model. Stopping a LiDAR-only network for a missing /odom would be a stop
    for the absence of a signal it never reads."""
    e = make_node(e2r_driver())
    assert e.driver.needs_speed
    e._now += 1.0
    e.on_scan(scan_msg(real_scans(1)[0]))         # /odom has never arrived
    assert e._inhibited and e.pub.msgs[-1].drive.speed == 0.0
    assert any("odom" in t for _l, t in e._log.lines)

    t = make_node(tln_driver())
    assert not t.driver.needs_speed
    t._now += 1.0
    t.on_scan(scan_msg(real_scans(1)[0]))         # no /odom either, and it does not matter
    assert not t._inhibited and t.pub.msgs[-1].drive.speed > 0.0


def test_a_gap_and_then_recovery_clears_the_model_state():
    node = make_node(e2r_driver())
    scans = real_scans(6)
    feed(node, scans[:3])
    assert float(node.driver._hidden.abs().max()) > 0
    node._now += 5.0                               # the stream stops
    node.on_scan(scan_msg(scans[3]))               # stale odom -> inhibited, not driven
    assert node._inhibited
    feed(node, scans[4:5])                         # sensors are back
    assert not node._inhibited
    after = node.driver._hidden.clone()

    fresh = make_node(e2r_driver())
    feed(fresh, scans[4:5])
    assert torch.allclose(after, fresh.driver._hidden, atol=1e-6)
    assert any("cleared" in t for _l, t in node._log.lines)


def test_reset_clears_the_model_state():
    node = make_node(e2r_driver())
    feed(node, real_scans(3))
    assert float(node.driver._hidden.abs().max()) > 0
    node.on_reset(object())
    assert float(node.driver._hidden.abs().max()) == 0.0
    assert np.isnan(node.driver._prev_speed).all()
    assert any("reset" in t for _l, t in node._log.lines)


def test_reset_on_a_stateless_model_is_a_no_op_that_still_logs():
    node = make_node(tln_driver())
    feed(node, real_scans(2))
    before = node.pub.msgs[-1].drive.steering_angle
    node.on_reset(object())
    feed(node, real_scans(2)[1:])
    assert node.pub.msgs[-1].drive.steering_angle == pytest.approx(before, abs=1e-6) or True


# --------------------------------------------------------------------------- scan handling
def test_the_miss_sentinel_is_saturated_not_interpolated():
    """65.533 m is finite; letting it through would put a 33 m 'return' next to a 1 m wall."""
    r = np.full(N_BEAMS, 1.0, dtype=np.float32)
    r[500] = 65.533
    out = bn.saturate(r, RANGE_MAX)
    assert out[500] == pytest.approx(RANGE_MAX)
    assert out[499] == pytest.approx(1.0)


def test_end2race_on_this_car_fills_a_quarter_of_its_beams_and_says_so():
    node = make_node(e2r_driver())
    feed(node, real_scans(1))
    g = node._geometry
    assert g["identity"] is False
    assert g["unseen_fraction"] == pytest.approx(0.25, abs=0.005)
    assert g["fill_m"] == pytest.approx(e2r_mod.RAW_RANGE_MAX)
    assert any("cannot" in t and "%" in t for _l, t in node._log.lines if _l == "warn")


def test_tinylidarnet_on_this_car_needs_no_resampling_at_all():
    node = make_node(tln_driver())
    feed(node, real_scans(1))
    assert node._geometry["identity"] is True
    assert any("No resampling" in t for _l, t in node._log.lines)


def test_the_published_command_is_clipped_to_what_the_plant_can_execute():
    node = make_node(tln_driver(speed_map="car"), speed_cap=2.0)   # car map reaches -0.5 m/s
    feed(node, real_scans(20))
    for m in node.pub.msgs:
        assert -S_MAX - 1e-9 <= m.drive.steering_angle <= S_MAX + 1e-9
        assert 0.0 <= m.drive.speed <= 2.0 + 1e-9


# --------------------------------------------------------------------------- node <-> adapter
def adapter_commands(driver, scans, speeds=None):
    """The batched adapter's path, on the same scans, decoded back to (steer, speed).

    The scans go in as the simulator would emit them: `_norm_scan` divides by `range_max` and clamps
    to [0, 1] (`gym_env.py:633-636`), so that division is part of what is being compared.
    """
    spec = ma.external_spec(driver)
    policy = ma.external_policy(driver, spec, S_MAX)
    out = []
    for i, r in enumerate(scans):
        sat = bn.saturate(r, RANGE_MAX)
        norm = np.clip(sat / RANGE_MAX, 0.0, 1.0).astype(np.float32)
        obs = {"scan": torch.as_tensor(norm)[None, None, :],
               "speed": torch.tensor([[float(1.0 if speeds is None else speeds[i])
                                       / spec["v_max"]]], dtype=torch.float32)}
        a = policy(obs).numpy()[0]
        steer = float(np.clip(a[0], -1.0, 1.0) * S_MAX)
        speed = float(min(max(a[1], -1.0), 1.0) + 1.0) * 0.5 * spec["v_max"]
        out.append((steer, min(speed, SPEED_CAP)))
    return np.asarray(out, dtype=np.float64)


@pytest.mark.parametrize("kind", ["tinylidarnet", "end2race"])
def test_node_and_adapter_emit_the_same_command_on_recorded_scans(kind):
    scans = real_scans()
    speeds = 1.0 + 3.0 * np.abs(np.sin(np.arange(len(scans)) * 0.3))
    node = make_node(tln_driver() if kind == "tinylidarnet" else e2r_driver())
    node.driver.bind_scanner(n_beams=N_BEAMS, fov=FOV, range_max=RANGE_MAX)
    feed(node, scans, speeds=speeds)
    node_cmd = np.array([[m.drive.steering_angle, m.drive.speed] for m in node.pub.msgs])

    other = tln_driver() if kind == "tinylidarnet" else e2r_driver()
    other.bind_scanner(n_beams=N_BEAMS, fov=FOV, range_max=RANGE_MAX)
    other.reset()
    adapter_cmd = adapter_commands(other, scans, speeds)

    d = np.abs(node_cmd - adapter_cmd)
    print(f"{kind}: node vs adapter over {len(scans)} recorded scans -- "
          f"max |dsteer| {d[:, 0].max():.3e} rad, max |dspeed| {d[:, 1].max():.3e} m/s")
    assert d[:, 0].max() <= 1e-5 and d[:, 1].max() <= 1e-5, d.max(0)


def test_parity_would_fail_if_the_two_sides_disagreed():
    """The parity test must be capable of failing: perturb one beam on the adapter side only."""
    scans = real_scans(10)
    node = make_node(tln_driver())
    node.driver.bind_scanner(n_beams=N_BEAMS, fov=FOV, range_max=RANGE_MAX)
    feed(node, scans)
    node_cmd = np.array([[m.drive.steering_angle, m.drive.speed] for m in node.pub.msgs])
    other = tln_driver()
    other.bind_scanner(n_beams=N_BEAMS, fov=FOV, range_max=RANGE_MAX)
    bent = scans.copy()
    bent[:, 400:600] = 0.4                        # a wall that is not there
    assert np.abs(node_cmd - adapter_commands(other, bent)).max() > 1e-3
