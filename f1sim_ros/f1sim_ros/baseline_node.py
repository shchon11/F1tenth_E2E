"""Published F1TENTH baselines as a ROS 2 node: `/scan` (+ `/odom`) -> `/drive` at the scan rate.

    ros2 run f1sim_ros baseline --ros-args -p model:=tinylidarnet -p weights:=<path>.onnx
    ros2 run f1sim_ros baseline --ros-args -p model:=end2race     -p weights:=<path>.pth

It plugs into the same link everything else here uses -- the console's ROS 2 mode
("센서 발행 + /drive 로 외부 제어"), `f1tenth_stack_sim.launch.py`, the standalone bridge, and the
real car -- because it speaks only the topics the car already publishes.

**It publishes `/drive` directly.** These models emit a steering angle and a speed; there is no plan
to track, so there is no `PlanTracker`, no controller arm, no clearance layer and no traction guard
in this path. With worker 19's split graph (`policy_node` -> `/f1sim/plan` -> `controller_node` ->
`/drive`) that makes `baseline_node` a *bypass*: it takes the place of both nodes at once, and
`docs/ros2.md` says so. Topic names, the `/f1sim/reset` contract and the staleness rule are kept
identical to `policy_node` so the two are interchangeable at the graph's edge.

What this node owns, and what it deliberately does not:

* it owns the **message plumbing** -- saturating the 0xFFFF miss sentinel, mapping the driver's
  scan window onto this scanner's, holding the measured speed, freshness, reset;
* it owns **none of the preprocessing**. Beam selection, clipping, the pressure token, the speed
  scaling and the output ranges all live in `f1sim.learn.baselines`, which the batched benchmark
  adapter calls too. That is why the parity test between them can be exact rather than close: they
  are one function, and the test proves the plumbing around it agrees.

Staleness, and why it is per model rather than copied from `policy_node`. That node inhibits when
the IMU, the attitude or the odometry go stale, because its policy reads all three. These do not:
TinyLidarNet is LiDAR-only and End2Race reads LiDAR and speed. Requiring a fresh IMU here would
stop a car for the absence of a signal the network never looks at, which is not safety, it is
superstition. `/scan` always counts (a watchdog, because the scan callback cannot notice its own
absence), and `/odom` counts exactly when `driver.needs_speed`.
"""
import math
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty as EmptyMsg

from f1sim.learn import baselines


#: Anything at or beyond this reads as a miss. Same rule and same reason as `policy_node.on_scan`:
#: the driver reports a miss as 65.533 m (the 0xFFFF mm sentinel), which is finite, so interpolating
#: across it invents ranges -- one missed beam next to a 1 m wall becomes a 33 m "return" halfway
#: between them. Saturate first, then resample.
MISS_SENTINEL_M = 65.0


def saturate(ranges, range_max: float) -> np.ndarray:
    """Raw `LaserScan.ranges` -> finite metres, misses at `range_max`."""
    r = np.asarray(ranges, dtype=np.float32)
    return np.where(np.isfinite(r) & (r > 0) & (r < MISS_SENTINEL_M), r, np.float32(range_max))


def scan_window(msg: LaserScan, fallback_fov: float):
    """`(fov, ok)` from the message's own header, or the nominal window when it is unusable."""
    lo, hi = float(msg.angle_min), float(msg.angle_max)
    if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
        return hi - lo, True
    return float(fallback_fov), False


class BaselineNode(Node):
    def __init__(self):
        super().__init__("f1sim_baseline")
        self.declare_parameter("model", "")
        self.declare_parameter("weights", "")
        self.declare_parameter("drive_topic", "drive")
        self.declare_parameter("enabled", True)
        self.declare_parameter("sensor_timeout", 0.25)
        self.declare_parameter("reset_topic", "/f1sim/reset")
        # The plant's limits, not the model's. A command outside them cannot be executed by the car
        # or by the simulator, so clipping here is what makes the node and the batched adapter
        # publish the same number: `gym_env.step` clamps the normalised action to [-1, 1], which is
        # exactly `|steer| <= s_max` and `0 <= speed <= speed_cap` (gym_env.py:1052-1055).
        self.declare_parameter("steer_max", 0.4189)
        self.declare_parameter("speed_cap", 9.0)
        # TinyLidarNet's two upstream output mappings, `sim` (1..8 m/s) and `car` (-0.5..7.0).
        self.declare_parameter("speed_map", "sim")
        self.declare_parameter("skip_n", 0)                  # 0 = infer from the model's width
        # End2Race
        self.declare_parameter("hidden_scale", 4)
        self.declare_parameter("repo", "")
        self.declare_parameter("n_features", 0)              # 0 = their 360
        # What a bearing this scanner cannot see is filled with. End2Race wants 360 deg and this
        # car's Hokuyo spans 270, so 90 of its 360 features have no measurement behind them. NaN =
        # the model's own no-return value (30 m, what their ray tracer returns for a beam that hits
        # nothing); 0.0 is the other convention their code uses, for beams it masks out
        # (`eval_singleagent.py:114`). Both are scored; neither is a measurement.
        self.declare_parameter("scan_fill", float("nan"))
        # Their evaluation ticks the GRU at 100 Hz; this loop runs at the scan rate. 0 = one tick
        # per scan. 100.0 = their recurrence rate, on the held scan, against `caller_rate`.
        self.declare_parameter("tick_hz", 0.0)
        self.declare_parameter("caller_rate", 40.0)

        p = lambda n: self.get_parameter(n).value
        kind = str(p("model") or "").strip().lower()
        if kind not in baselines.KINDS:
            raise ValueError(f"model must be one of {', '.join(baselines.KINDS)}, got {p('model')!r}")
        weights = str(p("weights") or "")
        if not weights:
            raise ValueError("weights is required: these are published networks, and there is no "
                             "default set of them to fall back on")
        opts = {}
        if kind == "tinylidarnet":
            opts["speed_map"] = str(p("speed_map"))
            if int(p("skip_n")):
                opts["skip_n"] = int(p("skip_n"))
        else:
            opts["hidden_scale"] = int(p("hidden_scale"))
            if str(p("repo")):
                opts["repo"] = str(p("repo"))
            if int(p("n_features")):
                opts["n_features"] = int(p("n_features"))
            fill = float(p("scan_fill"))
            if math.isfinite(fill):
                opts["scan_fill"] = fill
            if float(p("tick_hz")) > 0:
                opts["tick_hz"] = float(p("tick_hz"))
                opts["caller_rate_hz"] = float(p("caller_rate"))
        self.driver = baselines.load(kind, weights, **opts)

        self.steer_max = float(p("steer_max"))
        self.speed_cap = float(p("speed_cap"))
        self.timeout = float(p("sensor_timeout"))
        if not math.isfinite(self.timeout) or self.timeout <= 0.0:
            raise ValueError(f"sensor_timeout must be a positive finite number, got {self.timeout}")

        self.v = 0.0
        self.t_odom = None
        self.t_scan = None
        self._inhibited = False
        self._last_inhibit_log = 0.0
        self._geometry = None                 # what the first real scan turned out to be
        self.last_t = None

        if self.driver.needs_speed:
            self.create_subscription(Odometry, "odom", self.on_odom, 1)
        self.create_subscription(LaserScan, "scan", self.on_scan, 1)
        # A std_msgs/Empty TOPIC, the one `bridge_node` / `vesc_sim_node` announce after a reset --
        # deliberately not the std_srvs/Empty SERVICE of the same name. Same contract as
        # `policy_node.on_reset`; on the real car nothing publishes it and the node is unaffected.
        self.create_subscription(EmptyMsg, str(p("reset_topic")), self.on_reset, 1)
        self.create_timer(max(0.02, self.timeout / 4.0), self.on_watchdog)
        self.pub = self.create_publisher(AckermannDriveStamped, str(p("drive_topic")), 1)

        # Warm up on a scan of the driver's own shape, then clear: a first inference that takes
        # 40 ms because a graph is being built is a missed deadline on the first real scan, and for
        # End2Race it would also leave a hidden state from a scan that never happened.
        warm = np.full((1, self.driver.scan.n_beams), self.driver.scan.range_max, dtype=np.float32)
        self.driver.command(warm, np.zeros(1, dtype=np.float32) if self.driver.needs_speed else None)
        self.driver.reset()

        d = self.driver.describe()
        self.get_logger().info(
            f"baseline {kind} from {weights}: {d['n_beams']} beams over "
            f"{math.degrees(d['fov']):.0f} deg, range_max {d['range_max']:.1f} m, its own "
            f"{d['control_rate']:.0f} Hz. {d['note']}. Backend {d['backend']['backend']} "
            f"{d['backend']['version']}. Publishing /drive DIRECTLY: no plan tracker, no controller "
            f"arm, no clearance layer, no traction guard -- this model's output is the command. "
            f"Clipped to |steer| <= {self.steer_max:.4f} rad and 0 <= speed <= {self.speed_cap:.2f} m/s."
            + ("" if self.driver.needs_speed else " Reads /scan only; /odom is not subscribed."))

    # -- inputs -----------------------------------------------------------------------------------
    def clock(self):
        """Monotonic seconds. A method so tests can drive it without a ROS clock."""
        return time.monotonic()

    def stamp_now(self):
        return self.get_clock().now().to_msg()

    def on_odom(self, m: Odometry):
        v = m.twist.twist.linear.x
        if not math.isfinite(v):          # a NaN speed must not reach the model, nor count as fresh
            return
        self.v = float(v)
        self.t_odom = self.clock()

    def on_reset(self, _msg: EmptyMsg):
        """The car was reset. A GRU state and a held previous speed describe a run that is over."""
        self.driver.reset()
        self.get_logger().info("reset: baseline model state cleared")

    # -- freshness --------------------------------------------------------------------------------
    def _stale_inputs(self, now, check_scan=False):
        out = []
        if self.driver.needs_speed and (self.t_odom is None or now - self.t_odom > self.timeout):
            out.append("odom")
        if check_scan and (self.t_scan is None or now - self.t_scan > self.timeout):
            out.append("scan")
        return out

    def on_watchdog(self):
        """Stop driving when the scans themselves stop -- `on_scan` cannot notice its own absence."""
        now = self.clock()
        stale = self._stale_inputs(now, check_scan=True)
        if stale:
            self._inhibit(stale, self.stamp_now())

    def _inhibit(self, stale, stamp):
        """Zero speed, zero steering, for a car already under this node's control. Not a request to
        move: it keeps the mux fed so the last non-zero command does not stand."""
        self._inhibited = True
        msg = AckermannDriveStamped()
        msg.header.stamp = stamp
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        if self.get_parameter("enabled").value:
            self.pub.publish(msg)
        now = time.monotonic()
        if now - self._last_inhibit_log > 1.0:
            self._last_inhibit_log = now
            self.get_logger().warning(
                f"no command: {', '.join(stale)} older than {self.timeout:.2f} s. Publishing zero "
                f"speed until the sensors come back.")

    def _resume(self):
        """A gap ended. Whatever the model remembers belongs to a segment that is over."""
        self._inhibited = False
        self.driver.reset()
        self.get_logger().info("sensors recovered: baseline model state cleared")

    # -- geometry ---------------------------------------------------------------------------------
    def _note_geometry(self, msg: LaserScan):
        """What this scanner is, against what the driver's repo assumed. Decided once, logged once.

        The window comes from the message's own header rather than a constant: a driver that is
        actually publishing 240 deg, or publishing it backwards, would otherwise have every one of
        its returns placed at a bearing it was not measured at, and nothing downstream would look
        wrong.

        The decision itself is `driver.bind_scanner`, which the batched adapter also calls -- so the
        node and the benchmark cannot disagree about which beams the model reads or what fills the
        ones this car cannot see.
        """
        if self._geometry is not None:
            return self._geometry
        sc = self.driver.scan
        fov, from_header = scan_window(msg, sc.fov)
        n = len(msg.ranges)
        rmax = (float(msg.range_max) if math.isfinite(msg.range_max) and msg.range_max > 0
                else sc.range_max)
        g = dict(self.driver.bind_scanner(n_beams=n, fov=fov, range_max=rmax))
        g["fov_from_header"] = from_header
        self._geometry = g
        if not from_header:
            self.get_logger().warning(
                f"scan angle_min/angle_max = {msg.angle_min}/{msg.angle_max} is not a usable "
                f"window; assuming the model's own {math.degrees(sc.fov):.0f} deg")
        if g["identity"]:
            self.get_logger().info(
                f"scan matches the model's own: {n} beams over {math.degrees(fov):.1f} deg, "
                f"range_max {rmax:.1f} m. No resampling.")
        else:
            self.get_logger().warning(
                f"scan is {n} beams over {math.degrees(fov):.1f} deg (range_max {rmax:.1f} m); the "
                f"model was trained on {sc.n_beams} over {math.degrees(sc.fov):.1f} deg "
                f"(range_max {sc.range_max:.1f} m). Mapping by BEARING, and filling "
                f"{g['unseen_fraction'] * 100:.0f} % of its beams -- the ones this scanner cannot "
                f"see -- with {g['fill_m']:.1f} m. That substitution is not a measurement; it is "
                f"declared here and in every result this configuration produces.")
        return self._geometry

    # -- one scan ---------------------------------------------------------------------------------
    def on_scan(self, m: LaserScan):
        t0 = time.perf_counter()
        now = self.clock()
        self.t_scan = now
        stale = self._stale_inputs(now)
        if stale:
            self._inhibit(stale, self.stamp_now())
            return
        if self._inhibited:
            self._resume()
        g = self._note_geometry(m)
        r = self.driver.adapt(saturate(m.ranges, g["range_max"])[None, :])
        cmd = self.driver.command(r,
                                  np.array([self.v], dtype=np.float32) if self.driver.needs_speed
                                  else None)[0]
        msg = AckermannDriveStamped()
        msg.header.stamp = m.header.stamp
        msg.drive.steering_angle = float(np.clip(cmd[0], -self.steer_max, self.steer_max))
        # Floored at 0 rather than allowed negative: the batched action space this is checked
        # against cannot express a reverse command either (`gym_env.py:1054` maps a clamped
        # normalised action to `(a+1)/2 * v_max`, which is never below zero), and a parity claim
        # between the two has to be about the same command.
        msg.drive.speed = float(np.clip(cmd[1], 0.0, self.speed_cap))
        if self.get_parameter("enabled").value:
            self.pub.publish(msg)
        dt = (time.perf_counter() - t0) * 1000
        if self.last_t is None or time.perf_counter() - self.last_t > 5:
            self.get_logger().info(
                f"inference {dt:.1f} ms  v={self.v:.2f} cmd=({msg.drive.steering_angle:+.2f} rad, "
                f"{msg.drive.speed:.2f} m/s)")
            self.last_t = time.perf_counter()


def main():
    rclpy.init()
    n = BaselineNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
