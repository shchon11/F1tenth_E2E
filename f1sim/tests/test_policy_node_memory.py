"""The ROS policy node carries a recurrent policy's hidden state across scan callbacks.

Same shape as `test_controller_traction.py`: no ROS graph, no device, no vehicle. The node's real
`on_odom` / `on_imu` / `on_scan` / `_resume` / `on_reset` run against stub messages and a fake
clock, and the assertions are about the state the node keeps between them.

What is worth pinning:

* the hidden state ADVANCES between callbacks -- the node is not running a recurrent policy from
  zeros every scan, which would look exactly like a working policy and drive like a different one;
* it is cleared when the scan stream comes back after a gap (`_resume`), because the observation
  history it belongs to is cleared there too;
* it is cleared on `/f1sim/reset`, the topic the simulator nodes announce after they reset a car;
* a legacy checkpoint reaches none of this: the runtime is inert and the node behaves as before.
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

import f1sim_ros.policy_node as pn                            # noqa: E402
from f1sim.learn.memory import memory_spec, runtime_for       # noqa: E402
from f1sim.learn.model import ActorCritic, load_for_memory, save_checkpoint  # noqa: E402
from f1sim.learn.obs import ObsBuilder, ObsSpec               # noqa: E402

G = pn.G
SPEC = ObsSpec(n_beams=64, scan_stack=2, scan_stride=1, action_history=2, act_dim=2, hist_len=0,
               range_max=10.0, v_max=8.0)


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


def memory_model(tmp_path, channels=("memory", "edges")):
    """A warm-started memory model, deterministic across calls.

    The whole build is inside one forked, seeded RNG -- not just the feedforward half. The GRU's
    weights are the tensors `load_for_memory` leaves fresh, so they come from the global generator
    and two unseeded calls would return two different policies; a test that compares a recovered
    node against a fresh one would then fail for the wrong reason.
    """
    meta = dict(n_stack=SPEC.scan_stack, n_beams=SPEC.n_beams, proprio_dim=SPEC.proprio_dim,
                priv_dim=9, act_dim=SPEC.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(31)
        ff = ActorCritic(**meta)
        base = str(tmp_path / "ff.pt")
        save_checkpoint(base, ff, {"spec": {}})
        m, _e, _f = load_for_memory(base, "cpu", memory_spec(hidden_size=16),
                                    scan_channels=({"channels": list(channels)} if channels else None))
        with torch.no_grad():                   # so that carrying the state visibly matters
            m.actor.memory.out.weight.normal_(0, 0.3)
    return m.eval()


def make_node(model, spec=SPEC):
    """The node with every ROS dependency replaced and its real methods intact.

    `SPEC` is a direct-action checkpoint (`act_dim=2`), which publishes `/drive` from this node and
    bypasses the controller, so the counts below are commands. `plan_node` below is the same stub
    for a plan checkpoint, where the published thing is a `Plan` -- the memory rules are the same
    either way, and that is the point of checking both.
    """
    from f1sim_ros.deploy import SensorIntake
    n = pn.PolicyNode.__new__(pn.PolicyNode)
    n.device = torch.device("cpu")
    n.spec = spec
    n.direct = int(spec.act_dim) == 2
    n.obs = ObsBuilder(spec, "cpu")
    n.model = model
    n.policy_state = runtime_for(model, batch=1, device="cpu")
    n.pub = RecordingPub()
    n.pub_drive = n.pub
    n.pub_plan = RecordingPub()
    n.pub_state = RecordingPub()
    n.speed_cap = 8.0; n.steer_max = 0.4189
    n.timeout = 0.25
    n._inhibited = False; n._last_inhibit_log = -1e9; n.last_t = None
    n.seq = 0; n.checkpoint_id = "test/stub.pt@000000000000"
    n.memory_clears = 0; n.memory_cleared_at = -1.0; n.memory_cleared_reason = ""
    n._log = Logger(); n._now = 100.0
    n.get_logger = lambda: n._log
    n.get_parameter = lambda name: SimpleNamespace(value=True)
    n.clock = lambda: n._now
    n.stamp_now = lambda: ros_time(n._now)
    n.sensors = SensorIntake(n.clock, n._log, n.timeout)
    return n


def published(node):
    """What this node put on the wire this run: commands for a direct checkpoint, plans for a plan
    one. The memory assertions are about how many steps were taken, not about which topic."""
    return node.pub_plan.msgs if not node.direct else node.pub.msgs


def imu_msg():
    return SimpleNamespace(header=SimpleNamespace(stamp=None),
                           orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                           orientation_covariance=[0.0] * 9,
                           angular_velocity=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                           linear_acceleration=SimpleNamespace(x=0.0, y=0.0, z=1.0))


def ros_time(t=0.0):
    """A real `builtin_interfaces/Time`: the node assigns it into an `AckermannDriveStamped`, whose
    setter checks the type."""
    from builtin_interfaces.msg import Time
    return Time(sec=int(t), nanosec=int((t - int(t)) * 1e9))


def scan_msg(r=3.0):
    return SimpleNamespace(header=SimpleNamespace(stamp=ros_time()),
                           ranges=[r] * SPEC.n_beams, range_max=10.0)


def feed(node, steps=1, dt=0.025, r=3.0):
    for _ in range(steps):
        node._now += dt
        node.on_odom(SimpleNamespace(twist=SimpleNamespace(twist=SimpleNamespace(
            linear=SimpleNamespace(x=1.0), angular=SimpleNamespace(z=0.0)))))
        node.on_imu(imu_msg())
        node.on_scan(scan_msg(r))


def hidden(node):
    h = node.policy_state.hidden
    return None if h is None or h.actor is None else h.actor.clone()


def test_hidden_state_is_carried_between_scan_callbacks(tmp_path):
    node = make_node(memory_model(tmp_path))
    feed(node, 1)
    h1 = hidden(node)
    assert h1 is not None and float(h1.abs().max()) > 0, "the first scan must leave a state"
    feed(node, 1)
    h2 = hidden(node)
    assert not torch.allclose(h1, h2), "the second scan must act on the carried state"
    assert len(published(node)) == 2


def test_a_sensor_gap_clears_the_memory_with_the_observation_history(tmp_path):
    node = make_node(memory_model(tmp_path))
    feed(node, 3)
    assert float(hidden(node).abs().max()) > 0
    mem_before = node.policy_state.scan.mem.clone()
    assert float(mem_before.min()) < 1.0, "the occupancy channel saw something"
    node._now += 5.0                                  # the stream stops for five seconds
    node.on_scan(scan_msg())                          # stale odom/imu: inhibited, not driven
    assert node._inhibited
    feed(node, 1)                                     # sensors are back
    assert not node._inhibited
    h = hidden(node)
    # Exactly one scan since the clear, so the state is whatever one step from zero produces --
    # and the occupancy channel is the fresh scan, not a decayed trace of the old segment.
    fresh = make_node(memory_model(tmp_path))
    feed(fresh, 1)
    assert torch.allclose(h, hidden(fresh), atol=1e-6)
    assert any("cleared" in t for _l, t in node._log.lines)


def test_reset_clears_the_memory(tmp_path):
    node = make_node(memory_model(tmp_path))
    feed(node, 3)
    assert float(hidden(node).abs().max()) > 0
    node.on_reset(object())                           # what `/f1sim/reset` delivers
    assert hidden(node) is None
    assert float(node.policy_state.scan.mem.min()) == 1.0
    assert any("reset" in t for _l, t in node._log.lines)


def test_a_legacy_checkpoint_keeps_the_node_exactly_as_it_was(tmp_path):
    meta = dict(n_stack=SPEC.scan_stack, n_beams=SPEC.n_beams, proprio_dim=SPEC.proprio_dim,
                priv_dim=9, act_dim=SPEC.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(7)
        ff = ActorCritic(**meta).eval()
    node = make_node(ff)
    assert not node.policy_state.stateful
    feed(node, 3)
    assert hidden(node) is None and node.policy_state.scan is None
    assert len(published(node)) == 3
    node.on_reset(object())                           # still a legal, inert call
    assert hidden(node) is None


def test_the_simulator_nodes_announce_their_resets():
    """The node listens on a topic; something has to speak on it. Both simulator nodes publish a
    std_msgs/Empty on `/f1sim/reset` after they reset, next to the service of the same name."""
    import inspect
    from f1sim_ros import bridge_node, vesc_sim_node
    for mod in (bridge_node, vesc_sim_node):
        src = inspect.getsource(mod)
        assert 'create_publisher(EmptyMsg, "/f1sim/reset"' in src, mod.__name__
        assert "self.pub_reset.publish(EmptyMsg())" in src, mod.__name__
