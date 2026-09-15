"""Replay a recorded sensor sequence through the nodes, deterministically, with no ROS graph.

The parity claim -- "the split graph publishes the `/drive` the monolithic node published" -- is
only worth as much as the way it is checked, so this harness is deliberately not a reimplementation
of either node:

* the nodes are **really constructed**, through their own `__init__`, with a real `rclpy` context
  and a real parameter file. Every parameter default, every subscription, every warm-up step is the
  one the car runs. (The earlier shape of this test built nodes with `__new__` and filled the
  fields in by hand, which cannot catch a constructor that stopped installing an arm.)
* the messages are **real** `LaserScan` / `Imu` / `Odometry` / `Plan` objects, so the numpy
  covariance array, the `builtin_interfaces/Time` typing and the fixed-size `float32[8]` are all
  exercised;
* only the clock and the publishers are replaced: a monotonic clock the test drives, and a
  collector in place of the middleware. `policy.pub_plan.publish` is wired straight into
  `controller.on_plan`, so the graph is the real graph minus the transport.

The sequences come from real bags (`scripts/make_graph_fixture.py`): `graph_seq_sim.npz` from a
simulator run of this graph, `graph_seq_real.npz` from one of the 22 car recordings. Their
accelerometers are in different units (m/s^2 and g), so the node's unit detection is on both paths.

One frame is: every `/odom` and `/sensors/imu/raw` message that arrived since the previous scan,
then the scan. Every callback of a frame sees the same clock value, on both sides, so the two runs
differ in nothing but the code under test.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SIM_SEQ = os.path.join(DATA, "graph_seq_sim.npz")
REAL_SEQ = os.path.join(DATA, "graph_seq_real.npz")


@dataclass
class Frame:
    ranges: np.ndarray
    range_max: float
    angle_min: float
    angle_max: float
    imu: np.ndarray            # (K, 6) as published: gyro xyz, linear acceleration xyz
    quat: np.ndarray           # (K, 4) w x y z
    cov0: np.ndarray           # (K,)
    odom_v: np.ndarray         # (J,)
    t: float                   # seconds from the first scan of the sequence


class Sequence:
    """A recorded sensor stream, grouped into the frames a node sees."""

    def __init__(self, path: str):
        self.path = path
        d = np.load(path, allow_pickle=False)
        self.d = {k: d[k] for k in d.files}
        self.source = str(self.d.get("source", ""))

    def __len__(self):
        return int(self.d["ranges"].shape[0])

    @property
    def n_beams(self) -> int:
        return int(self.d["ranges"].shape[1])

    def frames(self, n: Optional[int] = None) -> List[Frame]:
        d = self.d
        out = []
        for i in range(len(self) if n is None else min(n, len(self))):
            a, b = int(d["imu_ptr"][i]), int(d["imu_ptr"][i + 1])
            c, e = int(d["odom_ptr"][i]), int(d["odom_ptr"][i + 1])
            out.append(Frame(ranges=d["ranges"][i], range_max=float(d["range_max"]),
                             angle_min=float(d["angle_min"]), angle_max=float(d["angle_max"]),
                             imu=d["imu"][a:b], quat=d["imu_quat"][a:b], cov0=d["imu_cov0"][a:b],
                             odom_v=d["odom_v"][c:e], t=float(d["t_scan"][i])))
        return out


# ------------------------------------------------------------------ messages
def ros_time(t: float):
    from builtin_interfaces.msg import Time
    return Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)) % 1_000_000_000)


def scan_msg(f: Frame, stamp):
    from sensor_msgs.msg import LaserScan
    m = LaserScan()
    m.header.stamp = stamp
    m.header.frame_id = "laser"
    m.angle_min, m.angle_max = f.angle_min, f.angle_max
    m.angle_increment = (f.angle_max - f.angle_min) / max(1, len(f.ranges) - 1)
    m.range_min, m.range_max = 0.0, f.range_max
    m.ranges = [float(x) for x in f.ranges]
    return m


def imu_msg(row, quat, cov0, stamp):
    from sensor_msgs.msg import Imu
    m = Imu()
    m.header.stamp = stamp
    m.header.frame_id = "imu"
    m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (float(row[0]), float(row[1]), float(row[2]))
    m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (float(row[3]), float(row[4]), float(row[5]))
    m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    cov = list(m.orientation_covariance)
    cov[0] = float(cov0)
    m.orientation_covariance = cov
    return m


def odom_msg(v: float, stamp):
    from nav_msgs.msg import Odometry
    m = Odometry()
    m.header.stamp = stamp
    m.header.frame_id = "odom"
    m.child_frame_id = "base_link"
    m.twist.twist.linear.x = float(v)
    return m


class Collector:
    """A publisher's `publish`, replaced. Keeps the messages and, optionally, forwards them."""

    def __init__(self, forward=None):
        self.msgs = []
        self.forward = forward

    def __call__(self, msg):
        self.msgs.append(msg)
        if self.forward is not None:
            self.forward(msg)

    @property
    def commands(self) -> List[Tuple[float, float]]:
        return [(float(m.drive.steering_angle), float(m.drive.speed)) for m in self.msgs]


# ------------------------------------------------------------------ parameter files
POLICY_PARAMS = ("checkpoint", "device", "speed_cap", "sensor_timeout", "imu_accel_scale",
                 "enabled", "steer_max", "drive_topic")
#: The monolithic node's parameters, in ONE node section: it is a single node named `f1sim_policy`
#: and it declares the policy's and the controller's parameters together, which is the thing the
#: split undid.
MONOLITH_PARAMS = POLICY_PARAMS + ("controller", "grip_mu", "clearance_margin", "traction",
                                   "traction_params", "wheelbase", "cmd_delay", "steer_bias",
                                   "steer_gain", "speed_gain")
CONTROLLER_PARAMS = ("checkpoint", "device", "speed_cap", "steer_max", "sensor_timeout",
                     "plan_timeout", "controller", "grip_mu", "clearance_margin", "traction",
                     "traction_params", "wheelbase", "cmd_delay", "steer_bias", "steer_gain",
                     "speed_gain", "enabled", "viz")


def _yaml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return repr(v)


def write_params(path, sections: dict) -> str:
    """`{node_name: {param: value}}` as the params file `rclpy` reads from `--params-file`."""
    lines = []
    for node, params in sections.items():
        lines.append(f"/**/{node}:")
        lines.append("  ros__parameters:")
        for k, v in params.items():
            lines.append(f"    {k}: {_yaml_value(v)}")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return path


class Context:
    """`rclpy.init` / `shutdown` around one run, with a parameter file as the only configuration.

    One run at a time on the default context, because the monolithic node and the split policy node
    are both called `f1sim_policy` and both read `/**/f1sim_policy` -- running them in one context
    would hand each the other's parameters.
    """

    def __init__(self, params_file: str):
        self.params_file = params_file

    def __enter__(self):
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
        rclpy.init(args=["--ros-args", "--params-file", self.params_file])
        return self

    def __exit__(self, *exc):
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
        return False


# ------------------------------------------------------------------ the two runs
class Clock:
    """The monotonic clock both sides are driven by. One value per frame."""

    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self):
        return self.t


def _drive_clock(node, clock):
    """Point every clock the node reads at ours.

    `SensorIntake` captured the node's bound `clock` at construction, so replacing the node's
    attribute alone would leave the intake on `time.monotonic` and the freshness logic on a
    different clock from the rest of the node -- which is exactly the kind of half-applied stub
    that makes a parity test pass for the wrong reason.
    """
    node.clock = clock
    if getattr(node, "sensors", None) is not None:
        node.sensors.clock = clock
    node.stamp_now = lambda: ros_time(clock())


def run_monolith(seq: Sequence, frames: int, params: dict, tmpdir) -> List[Tuple[float, float]]:
    """The node as it was at `0b78111`, driven over the sequence. The reference."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference",
                        "monolithic_policy_node.py")
    spec = importlib.util.spec_from_file_location("monolithic_policy_node", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pf = write_params(os.path.join(str(tmpdir), "monolith.yaml"),
                      {"f1sim_policy": {k: v for k, v in params.items() if k in MONOLITH_PARAMS}})
    clock = Clock()
    with Context(pf):
        node = mod.PolicyNode()
        _drive_clock(node, clock)
        out = Collector()
        node.pub.publish = out
        _feed(seq, frames, clock, [node], scan_order=[node])
        node.destroy_node()
    return out.commands


@dataclass
class GraphRun:
    """What one replay of the graph produced.

    `tracked` is which of `commands` came from tracking a plan, as opposed to the zero-speed
    commands the controller publishes when an input is stale. The pairing is recorded rather than
    inferred from the counts: a zero command is a legitimate output of the tracker too, so "the
    leading zeros are the inhibits" is an assumption and this is a fact.
    """
    commands: List[Tuple[float, float]]
    plans: List[List[float]]
    tracked: List[int]

    def tracked_commands(self) -> List[Tuple[float, float]]:
        return [self.commands[i] for i in self.tracked]


def run_graph(seq: Sequence, frames: int, params: dict, tmpdir, *, keep=None) -> "GraphRun":
    """`policy_node -> /f1sim/plan -> controller_node -> /drive` over the same sequence."""
    from f1sim_ros.controller_node import ControllerNode
    from f1sim_ros.policy_node import PolicyNode
    pf = write_params(os.path.join(str(tmpdir), "graph.yaml"),
                      {"f1sim_policy": {k: v for k, v in params.items() if k in POLICY_PARAMS},
                       "f1sim_controller": {k: v for k, v in params.items()
                                            if k in CONTROLLER_PARAMS}})
    clock = Clock()
    with Context(pf):
        policy = PolicyNode()
        controller = ControllerNode()
        _drive_clock(policy, clock)
        _drive_clock(controller, clock)
        drive = Collector()
        controller.pub.publish = drive
        tracked = []
        _track = controller._track_and_publish

        def spy(plan, snap, _t=_track):
            _t(plan, snap)
            tracked.append(len(drive.msgs) - 1)

        controller._track_and_publish = spy
        # The wire. `controller.on_plan` is what the middleware would have called.
        plans = Collector(forward=controller.on_plan)
        policy.pub_plan.publish = plans
        policy.pub_state.publish = Collector()
        controller.pub_diag.publish = Collector()
        # The controller's scan callback runs before the policy's, because in the graph the
        # controller's snapshot of a scan is what the plan made from that scan is tracked against.
        # (When it does not -- a busy machine, a slow subscriber -- the controller holds the plan
        # for one scan rather than pairing it with the wrong measurement; see `_snapshot_for`.)
        _feed(seq, frames, clock, [controller, policy], scan_order=[controller, policy])
        if keep is not None:
            keep.update(policy=policy, controller=controller, plans=plans, drive=drive)
        result = GraphRun(drive.commands, [list(m.plan) for m in plans.msgs], tracked)
        if keep is None:
            policy.destroy_node(); controller.destroy_node()
    return result


def _feed(seq: Sequence, frames: int, clock: Clock, nodes, scan_order):
    """One frame at a time: the odometry, then the IMU samples, then the scan."""
    for f in seq.frames(frames):
        clock.t = 1000.0 + f.t
        stamp = ros_time(f.t)
        for v in f.odom_v:
            m = odom_msg(v, stamp)
            for n in nodes:
                n.on_odom(m)
        for row, q, c0 in zip(f.imu, f.quat, f.cov0):
            m = imu_msg(row, q, c0, stamp)
            for n in nodes:
                n.on_imu(m)
        m = scan_msg(f, stamp)
        for n in scan_order:
            n.on_scan(m)
