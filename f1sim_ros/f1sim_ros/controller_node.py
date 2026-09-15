"""The controller half of the graph: /f1sim/plan + /scan + /odom + /sensors/imu -> /drive.

Everything that used to sit between the policy's plan and the VESC lives here: the iLQR
`PlanTracker`, the runtime arms (`fixed_low`, `clearance` and their combination) and the traction
guard. The node knows nothing about the network that produced the plan -- it reads eight floats off
a topic -- which is what makes the arms comparable across policies and what lets worker 18's
`baseline_node` publish `/drive` directly without going through any of this.

**Cadence.** One command per plan, and the policy emits one plan per scan, so `/drive` comes out at
the scan rate exactly as the monolithic node's did. The command for a plan is computed against the
sensor snapshot of *the scan that plan was made from*, found by the plan's `header.stamp` -- the
scan's own stamp, not the publish time. That pairing is the reason the split is bit-exact: it
cannot drift if the plan arrives late, and it cannot silently pair a plan with the next scan's
measured speed.

**When it stops.** Three ways, all of which publish zero speed rather than letting the last command
stand:

* the same stale-sensor rule the monolithic node had, at the same `sensor_timeout`, evaluated on
  this node's own subscriptions -- so a dead LiDAR stops the car at the instant it used to;
* `plan_timeout` since the last plan: the policy died, or is inhibiting;
* a command asked for while no scan has been seen at all.
"""
import math
import time

import numpy as np
import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Empty as EmptyMsg
from visualization_msgs.msg import Marker

from f1sim.learn.obs import ObsSpec, norm_scan
from f1sim_interfaces.msg import Plan

from f1sim_ros.deploy import (LIDAR_FOV, LIDAR_MOUNT_X, SensorIntake, build_traction_guard,
                              install_clearance_arm, install_grip_arm, resample_ranges,
                              split_deployable)

try:
    from vesc_msgs.msg import VescImuStamped
except ImportError:
    VescImuStamped = None

try:
    from vesc_msgs.msg import VescStateStamped
except ImportError:
    VescStateStamped = None


#: Scan snapshots kept while their plan is in flight. One is enough when the policy answers inside
#: a scan period; a handful covers a policy that is a step or two behind without ever letting an
#: old snapshot be found by a new plan.
SNAPSHOT_MAX = 8


def stamp_key(stamp):
    """`(sec, nanosec)` for a message stamp, or None when it is absent or the zero default.

    A zero stamp is not a time: `rclpy`'s default-constructed header has one, and matching plans
    against it would pair every plan with the first scan forever. Treated as "this publisher does
    not stamp", which falls back to the newest snapshot and is counted in the diagnostics.
    """
    if stamp is None:
        return None
    try:
        sec, nsec = int(stamp.sec), int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None
    return None if (sec == 0 and nsec == 0) else (sec, nsec)


class ScanSnapshot:
    """What the car measured at one scan: everything the tracker needs, frozen at that instant."""

    __slots__ = ("key", "t", "v", "imu_mean", "scan_norm", "stale", "stamp")

    def __init__(self, key, t, v, imu_mean, scan_norm, stale, stamp):
        self.key, self.t, self.v = key, t, v
        self.imu_mean, self.scan_norm, self.stale, self.stamp = imu_mean, scan_norm, stale, stamp


class ControllerNode(Node):
    def __init__(self):
        super().__init__("f1sim_controller")
        d = self.declare_parameter
        d("device", "cpu"); d("enabled", True)
        d("plan_topic", "/f1sim/plan"); d("drive_topic", "drive")
        d("diag_topic", "/f1sim/controller/diag"); d("reset_topic", "/f1sim/reset")
        d("speed_cap", 4.0); d("steer_max", 0.4189)
        # The plan action space: the same iLQR tracker as in training turns the local trajectory
        # into (steer, speed); cmd_delay = the measured command latency of this car.
        d("wheelbase", 0.3302); d("cmd_delay", 0.035)
        # Residual servo calibration the stack's steering_angle_to_servo_offset/gain do not absorb
        # (rad, ratio); the published angle is (cmd - steer_bias) / steer_gain. See
        # docs/ros2.md -- the polarity is opposite between this car's two servo configurations, so
        # the default is 1.0 and the direction is confirmed at walking pace before anything drives.
        d("steer_bias", 0.0); d("steer_gain", 1.0); d("speed_gain", 1.0)
        # The observation geometry the clearance grid is built on. It is the POLICY's, not this
        # node's: the grid is fed the scan in the observation's own units, resampled to the
        # observation's beam count, because that is the frame the monolithic node bent the plan in.
        # `checkpoint` is the authoritative source; the three overrides exist for a controller run
        # against something that is not a checkpoint on this disk (a baseline node, a replay).
        d("checkpoint", ""); d("n_beams", 0); d("range_max", 0.0); d("v_max", 0.0)
        d("controller", "fixed_low"); d("grip_mu", 0.0); d("clearance_margin", 0.0)
        d("traction", "off"); d("traction_params", "")
        d("sensor_timeout", 0.25); d("plan_timeout", 0.25); d("watchdog_period", 0.01)
        d("viz", False)
        p = lambda n: self.get_parameter(n).value

        self.device = torch.device(p("device"))
        self.spec = self._observation_spec(str(p("checkpoint")), int(p("n_beams")),
                                           float(p("range_max")), float(p("v_max")))
        self.speed_cap = float(p("speed_cap")); self.steer_max = float(p("steer_max"))
        self.cal = (float(p("steer_bias")), float(p("steer_gain")), float(p("speed_gain")))
        self.timeout = float(p("sensor_timeout"))
        self.plan_timeout = float(p("plan_timeout"))
        if not math.isfinite(self.plan_timeout) or self.plan_timeout <= 0.0:
            raise ValueError(f"plan_timeout must be a positive finite number, got {self.plan_timeout}")
        self.sensors = SensorIntake(self.clock, self.get_logger(), self.timeout)

        from f1sim.mpc import PlanTracker
        self.tracker = PlanTracker(1, self.device, float(p("wheelbase")), self.steer_max,
                                   float(self.spec.v_max))
        self.delay = torch.tensor([float(p("cmd_delay"))], device=self.device)
        # Plan controller arm. "fixed_low" (default) limits corner speed and accel/brake budgets for
        # a conservative constant friction; "legacy" is the untouched tracker. A "+clearance" suffix
        # adds the geometry layer: a local occupancy built from this scan alone (no map), and the
        # plan bent and slowed until it keeps `clearance_margin` from anything the scanner saw.
        self.controller_arm = str(p("controller"))
        self.grip = install_grip_arm(self.tracker, self.controller_arm, self.device,
                                     mu=(float(p("grip_mu")) or None))
        self.clearance = install_clearance_arm(self.tracker, self.controller_arm, self.device,
                                               self.spec,
                                               margin=(float(p("clearance_margin")) or None))
        #: Whether the beam bearings the clearance grid is built from have been checked against a
        #: real `LaserScan` header yet. Until then they are the nominal 270 deg window.
        self._scan_geometry_checked = False
        # Traction guard: wheel lock / launch spin from `/odom` wheel speed against the IMU, with
        # the speed command shaped when either fires. OFF by default -- it has been validated only
        # by replaying the recordings (`scripts/replay_traction.py`), never on the moving car, and
        # it is the one thing in this node that can raise a commanded speed the policy lowered.
        self.traction_arm = str(p("traction"))
        self.traction = build_traction_guard(self.traction_arm, str(p("traction_params")))
        self.traction_state = "off" if self.traction is None else "OK"

        self._snapshots = []                      # newest last
        #: A plan whose own scan this node has not processed yet. Held for one scan rather than
        #: tracked against the wrong measurement: in a live graph the two nodes receive `/scan`
        #: independently, and the policy's plan overtakes the controller's own scan callback a few
        #: per cent of the time. Waiting one scan costs nothing (the command comes out at the same
        #: cadence) and keeps the pairing exact; the fallback below bounds the wait.
        self._pending = None                      # (plan, stamp, key)
        self.plan = None; self.t_plan = None; self.plan_seq = None; self.plan_checkpoint = ""
        self._inhibited = False; self._last_inhibit_log = 0.0
        self._braking = False                     # currently zeroing for a missing plan
        self._plan_unmatched = 0                  # plans that did not find their own scan
        self._last_unmatched_log = 0.0
        self._commands = 0
        self.last_cmd = (0.0, 0.0)
        self.last_t = None

        if self.traction is not None and VescStateStamped is not None:
            self.create_subscription(VescStateStamped, "sensors/core", self.on_core, 1)
        self.create_subscription(Odometry, "odom", self.on_odom, 1)
        self.create_subscription(Imu, "sensors/imu/raw", self.on_imu, 10)
        if VescImuStamped is not None:
            self.create_subscription(VescImuStamped, "sensors/imu", self.on_vesc_imu, 1)
        self.create_subscription(LaserScan, "scan", self.on_scan, 1)
        self.create_subscription(Plan, str(p("plan_topic")), self.on_plan, 1)
        self.create_subscription(EmptyMsg, str(p("reset_topic")), self.on_reset, 1)
        self.pub = self.create_publisher(AckermannDriveStamped, p("drive_topic"), 1)
        self.pub_diag = self.create_publisher(DiagnosticArray, str(p("diag_topic")), 1)
        self.pub_viz = (self.create_publisher(Marker, "/f1sim/viz/plan", 1)
                        if bool(p("viz")) else None)
        self.pub_viz_clear = (self.create_publisher(Marker, "/f1sim/viz/clearance", 1)
                              if bool(p("viz")) and self.clearance is not None else None)
        #: The diagnostics as something an rviz window can show: which arm, at what friction, with
        #: the guard in what state, and the command going out. The `DiagnosticArray` has all of it
        #: and more, but reading it means a second tool -- and the question "what is this car
        #: running" is the one you ask while watching it drive.
        self.pub_viz_diag = (self.create_publisher(Marker, "/f1sim/viz/diag", 1)
                             if bool(p("viz")) else None)
        # The plan callback cannot notice its own absence, and neither can the scan callback.
        self.create_timer(max(0.002, float(p("watchdog_period"))), self.on_watchdog)

        # Warm the tracker's solver (and, on CUDA, its compiled graph) before anything drives.
        self.tracker(torch.zeros(1, 8, device=self.device), torch.zeros(1, device=self.device),
                     torch.tensor([self.speed_cap], device=self.device), None, delay=self.delay)
        self.tracker.reset(torch.zeros(1, dtype=torch.long, device=self.device))
        if self.clearance is not None:
            self.clearance.scan.fill_(1.0)
        self.get_logger().info(
            f"controller on {self.device}: {p('plan_topic')} -> {p('drive_topic')}, speed cap "
            f"{self.speed_cap} m/s, arm {self.controller_arm}"
            + (f" (mu {float(self.grip.mu[0]):.3f})" if self.grip is not None else "")
            + ("" if self.clearance is None else
               f", clearance margin {self.clearance.cspec.margin:.2f} m body edge "
               f"({self.clearance.cspec.margin + self.clearance.cspec.body_radius:.2f} m centre) "
               f"on a {self.clearance.cspec.cell * 100:.0f} cm grid from /scan alone")
            + f", traction {self.traction_arm}"
            + ("" if self.traction is None else
               f" (lock past {self.traction.p.lock_accel:.1f} m/s^2 wheel decel and "
               f"{self.traction.p.lock_rate:.1f} m/s^2 residual, spin past "
               f"{self.traction.p.spin_accel:.1f} / {self.traction.p.spin_rate:.1f}, release "
               f"authority {self.traction.p.release_max:.1f} m/s)")
            + f", plan timeout {self.plan_timeout:.2f} s")
        if self.traction is not None and VescStateStamped is None:
            self.get_logger().warning(
                "vesc_msgs is not importable, so /sensors/core motor current is unavailable: the "
                "traction guard will not require drive torque before calling a launch spin.")

    # ---------------------------------------------------------------- setup
    def _observation_spec(self, checkpoint: str, n_beams: int, range_max: float, v_max: float):
        """The observation geometry the clearance grid and the tracker are built on.

        Read from the checkpoint when there is one, because a clearance grid built on a different
        beam count or range scale than the policy's is a silently rotated and rescaled obstacle
        rather than an error. The explicit parameters are for a controller with no checkpoint on
        this disk; giving both and disagreeing is refused rather than resolved by precedence.
        """
        from f1sim.learn.obs import ObsSpec as _Spec
        override = {k: v for k, v in (("n_beams", int(n_beams)), ("range_max", float(range_max)),
                                      ("v_max", float(v_max))) if v}
        if not checkpoint:
            if not override:
                self.get_logger().warning(
                    "no checkpoint and no n_beams/range_max/v_max: the clearance grid and the "
                    "tracker fall back to the ObsSpec defaults, which is right only by accident.")
            return _Spec(**override)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        spec_d = dict((ck.get("extra") or {}).get("spec") or {})
        spec = _Spec(**spec_d) if spec_d else _Spec()
        for k, v in override.items():
            if abs(float(getattr(spec, k)) - float(v)) > 1e-9:
                raise ValueError(
                    f"controller parameter {k}={v} disagrees with the checkpoint's observation "
                    f"spec ({getattr(spec, k)}). The policy and the clearance grid have to read "
                    f"the same scan; pick one source.")
        return spec

    # ---------------------------------------------------------------- clocks
    def clock(self):
        """Monotonic seconds. A method so tests can drive it without a ROS clock."""
        return time.monotonic()

    def stamp_now(self):
        """Current ROS time as a message stamp. Overridden in tests."""
        return self.get_clock().now().to_msg()

    # ---------------------------------------------------------------- callbacks
    def on_odom(self, m: Odometry):
        if not self.sensors.on_odom(m):
            return
        if self.traction is not None:
            # Fed at the `/odom` rate -- 50 Hz on this car, and the rate the replay validated it at
            # -- not at the scan rate, so no wheel-speed sample is skipped. Stale inputs are passed
            # as None rather than as their last value: the guard holds its filters over a missing
            # sample, and `update` re-seeds rather than detecting across a gap.
            now = self.sensors.t_odom
            st = self.traction.update(
                now, self.sensors.v,
                self.sensors.ax_body if self.sensors.fresh(self.sensors.t_ax, now) else None,
                self.sensors.motor_current if self.sensors.fresh(self.sensors.t_current, now) else None)
            self.traction_state = st.state
            if st.changed:
                self.get_logger().info(
                    f"traction {st.state}: wheel {st.wheel_speed:+.2f} m/s at "
                    f"{st.wheel_accel:+.1f} m/s^2, body {st.body_speed:+.2f} m/s at "
                    f"{st.body_accel:+.1f} m/s^2 (residual {st.residual:+.1f}, slip {st.slip:+.2f}), "
                    f"{st.locks} locks / {st.spins} spins so far")

    def on_core(self, m):
        self.sensors.on_core(m)

    def on_imu(self, m: Imu):
        self.sensors.on_imu(m)

    def on_vesc_imu(self, m):
        self.sensors.on_vesc_imu(m)

    def on_scan(self, m: LaserScan):
        """Freeze what the car measured at this scan. The command comes with the plan."""
        now = self.clock()
        self.sensors.t_scan = now
        imu_mean = self.sensors.take_imu_mean(now)
        stale = self.sensors.stale(now)
        if stale:
            self._inhibit(stale, self.stamp_now())
            return
        if self._inhibited:                       # coming back from a gap
            self._resume()
        r = resample_ranges(m.ranges, m.range_max, self.spec.n_beams)
        scan_norm = norm_scan(torch.as_tensor(r, dtype=torch.float32, device=self.device)[None],
                              float(self.spec.range_max))
        if self.clearance is not None:
            self._check_scan_geometry(m)
        snap = ScanSnapshot(stamp_key(m.header.stamp), now, self.sensors.v, imu_mean, scan_norm,
                            stale, m.header.stamp)
        self._snapshots.append(snap)
        while len(self._snapshots) > SNAPSHOT_MAX:
            self._snapshots.pop(0)
        if self._pending is not None:
            # A plan that arrived before its own scan did. If this is that scan, the pairing is
            # exact after all; if it is not, the plan is a scan old and waiting further would just
            # skip a command, so it is tracked here against this measurement and counted.
            plan, _stamp, key = self._pending
            self._pending = None
            if key is not None and key != snap.key:
                self._note_unmatched(key)
            self._track_and_publish(plan, snap)

    def on_plan(self, m: Plan):
        """The plan the car will drive, tracked against the scan it was computed from."""
        t0 = time.perf_counter()
        self.plan = np.asarray(m.plan, dtype=np.float32)
        self.t_plan = self.clock()
        self.plan_seq = int(m.seq); self.plan_checkpoint = str(m.checkpoint)
        if self._inhibited:
            return                     # stale sensors: `_inhibit` already published zero
        snap = self._snapshot_for(m.header.stamp)
        if snap is None:
            # Either no scan at all yet, or this plan's scan has not reached this node. Held, not
            # dropped and not paired with the wrong scan; `on_scan` picks it up.
            self._pending = (self.plan, m.header.stamp, stamp_key(m.header.stamp))
            return
        self._track_and_publish(self.plan, snap)
        dt = (time.perf_counter() - t0) * 1000
        if self.last_t is None or time.perf_counter() - self.last_t > 5:
            self.get_logger().info(
                f"track {dt:.1f} ms  v={snap.v:.2f} plan#{self.plan_seq} "
                f"cmd=({self.last_cmd[0]:+.2f} rad, {self.last_cmd[1]:.2f} m/s)")
            self.last_t = time.perf_counter()

    def _snapshot_for(self, stamp):
        """The scan snapshot a plan belongs to: the one whose stamp it carries, or None.

        None means "not here yet", and the caller holds the plan for one scan rather than pairing
        it with a measurement it was not made from. An UNSTAMPED plan (a zero header, which is not
        a time) can never match, so it takes the newest snapshot immediately and is counted -- that
        is what a node without this pairing would always have done.
        """
        key = stamp_key(stamp)
        if key is None:
            if not self._snapshots:
                return None
            self._note_unmatched(key)
            return self._snapshots[-1]
        for s in reversed(self._snapshots):
            if s.key == key:
                return s
        return None

    def _note_unmatched(self, key):
        """A plan tracked against a scan that is not its own. Counted, and in the diagnostics: it
        is the difference between "bit-exact with the monolithic node" and "close"."""
        self._plan_unmatched += 1
        now = time.monotonic()
        if now - self._last_unmatched_log > 5.0:
            self._last_unmatched_log = now
            self.get_logger().warning(
                f"plan stamp {key} never matched a held scan ({self._plan_unmatched} so far): "
                f"tracking it against the newest one instead. The command is one scan's "
                f"measurement away from what the plan was made for.")

    def _track_and_publish(self, plan, snap: ScanSnapshot):
        a = torch.as_tensor(plan, dtype=torch.float32, device=self.device).reshape(1, -1)
        if self.clearance is not None:
            # This scan, in the observation's own units, before the tracker is asked for anything.
            # The arm holds one frame and nothing else: no map, no pose, no memory across scans.
            self.clearance.update_scan(snap.scan_norm)
        cmd = self.tracker(a, torch.tensor([snap.v], device=self.device),
                           torch.tensor([self.speed_cap], device=self.device),
                           torch.tensor([float(snap.imu_mean[2])], device=self.device),
                           delay=self.delay)[0]
        steer = float(max(-self.steer_max,
                          min(self.steer_max, (float(cmd[0]) - self.cal[0]) / self.cal[1])))
        speed = float(cmd[1])
        if self.traction is not None:
            # Last thing before the command leaves, and *before* `speed_gain`: the guard reasons in
            # the car's own m/s -- it compares the command against a body speed estimated from this
            # car's sensors -- while the published field is that speed divided by the calibration
            # gain, so shaping the published number would mix the two scales whenever the gain is
            # not 1.
            speed = min(self.traction.shape(speed), self.speed_cap)
        msg = AckermannDriveStamped()
        msg.header.stamp = snap.stamp                 # the scan the command answers, as the monolith did
        msg.drive.steering_angle = steer
        msg.drive.speed = float(speed) / self.cal[2]
        self.last_cmd = (msg.drive.steering_angle, msg.drive.speed)
        self._commands += 1
        self._braking = False
        if self.get_parameter("enabled").value:
            self.pub.publish(msg)
        self._publish_diag(snap.stamp, "OK", "")
        self._publish_viz(snap)

    # ---------------------------------------------------------------- watchdog
    def on_watchdog(self):
        """Stop the car when the scans stop, or when the plans do.

        The stale-sensor half is the monolithic node's watchdog, unchanged and at the same
        threshold: this node subscribes to the same topics, so a dead LiDAR reaches it at the same
        instant it used to. The plan half is new, and is the only thing standing between a dead
        policy process and a car still driving on the last plan it sent.
        """
        now = self.clock()
        stale = self.sensors.stale(now, check_scan=True)
        if stale:
            self._inhibit(stale, self.stamp_now())
            return
        if self.t_plan is None or now - self.t_plan > self.plan_timeout:
            self._brake(now)

    def _brake(self, now):
        """The plan stream dried up while the sensors are fine. Zero speed, steering held."""
        msg = AckermannDriveStamped(); msg.header.stamp = self.stamp_now()
        msg.drive.speed = 0.0; msg.drive.steering_angle = 0.0
        if self.get_parameter("enabled").value:
            self.pub.publish(msg)
        self.last_cmd = (0.0, 0.0)
        age = None if self.t_plan is None else now - self.t_plan
        shown = "ever" if age is None else f"{age:.2f} s"
        self._publish_diag(msg.header.stamp, "ERROR",
                           f"no plan ({shown}, timeout {self.plan_timeout:.2f} s)")
        if not self._braking:
            self._braking = True
            self.get_logger().warning(
                f"no plan on the plan topic ({'none since start' if age is None else shown}): "
                f"holding zero speed. The sensors are fine, so this is the policy node, not the "
                f"car.")

    def _inhibit(self, stale, stamp):
        """Stop driving on data we do not have.

        Publishing a zero-speed command is the safe output *for a car already under this node's
        control*: it keeps the mux fed so the last non-zero command does not stand. It is not an
        instruction to move, and steering is held at zero.
        """
        self._inhibited = True
        msg = AckermannDriveStamped(); msg.header.stamp = stamp
        msg.drive.speed = 0.0; msg.drive.steering_angle = 0.0
        if self.get_parameter("enabled").value:
            self.pub.publish(msg)
        self.last_cmd = (0.0, 0.0)
        self._publish_diag(stamp, "ERROR", f"stale: {', '.join(stale)}")
        now = time.monotonic()
        if now - self._last_inhibit_log > 1.0:
            self._last_inhibit_log = now
            self.get_logger().warning(
                f"no command: {', '.join(stale)} older than {self.timeout:.2f} s. "
                f"Publishing zero speed until the sensors come back.")

    def _resume(self):
        """A gap ended. The tracker's warm start and the guard's filters describe a segment that is
        over, and so does any plan still held from before it."""
        self._inhibited = False
        if self.traction is not None:
            # The wheel-speed derivative, the body-speed estimate and any latched release all
            # describe a segment that is over.
            self.traction.reset()
        self.tracker.reset(torch.zeros(1, dtype=torch.long, device=self.device))
        self._snapshots.clear()
        self.plan = None; self.t_plan = None; self._pending = None
        self.get_logger().info("sensors recovered: tracker history and held plan cleared")

    def on_reset(self, _msg: EmptyMsg):
        """The car was reset. Everything that describes the run it was in is now wrong."""
        if self.traction is not None:
            self.traction.reset()
        if self.clearance is not None:
            # Belt and braces: `on_scan` writes this buffer before the tracker is ever asked for a
            # command, so a stale frame cannot reach a plan. Clearing it anyway means that if that
            # ever stops being true, a car that has been picked up and put down is shaped by
            # nothing rather than by the room it used to be in.
            self.clearance.scan.fill_(1.0)
        self.tracker.reset(torch.zeros(1, dtype=torch.long, device=self.device))
        self._snapshots.clear()
        self.plan = None; self.t_plan = None; self._pending = None
        self.get_logger().info("reset: tracker history and held plan cleared")

    def _check_scan_geometry(self, m: LaserScan):
        """Re-declare the clearance grid's bearings from the driver's own window, once.

        The nominal 270 deg is what this car's `urg_node` publishes, but the grid is built by
        turning each return into a point at its bearing: a window that is actually 240 deg, or one
        published backwards, would place every obstacle somewhere it is not, and nothing downstream
        would look wrong. The ranges are resampled linearly over the message's own index range, so
        the resampled beam bearings are `linspace(angle_min, angle_max, n_beams)`.
        """
        if self._scan_geometry_checked:
            return
        self._scan_geometry_checked = True
        lo, hi = float(m.angle_min), float(m.angle_max)
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            self.get_logger().warning(
                f"scan angle_min/angle_max = {lo}/{hi} is not a usable window; the clearance grid "
                f"keeps the nominal {math.degrees(LIDAR_FOV):.0f} deg")
            return
        nominal = self.clearance.angles
        if abs(lo - float(nominal[0])) < 1e-3 and abs(hi - float(nominal[-1])) < 1e-3:
            return
        self.clearance.set_angles(torch.linspace(lo, hi, int(self.spec.n_beams),
                                                 device=self.device))
        self.get_logger().info(
            f"clearance grid rebuilt for this scanner's window: {math.degrees(lo):.1f} to "
            f"{math.degrees(hi):.1f} deg over {self.spec.n_beams} beams "
            f"(nominal was {math.degrees(float(nominal[0])):.1f} to "
            f"{math.degrees(float(nominal[-1])):.1f})")

    # ---------------------------------------------------------------- diagnostics
    def diag_values(self):
        """The controller's state as `(name, value)` strings. Also what `system_check` reports."""
        v = [("arm", self.controller_arm),
             ("mu", f"{float(self.grip.mu[0]):.5f}" if self.grip is not None else ""),
             ("clearance_margin_m",
              f"{self.clearance.cspec.margin:.3f}" if self.clearance is not None else ""),
             ("traction", self.traction_arm), ("traction_state", self.traction_state),
             ("plan_seq", "" if self.plan_seq is None else str(self.plan_seq)),
             ("plan_checkpoint", self.plan_checkpoint),
             ("plan_age_s", "-1.0" if self.t_plan is None else f"{self.clock() - self.t_plan:.3f}"),
             ("plan_unmatched", str(self._plan_unmatched)),
             ("commands", str(self._commands)),
             ("steer_cmd_rad", f"{self.last_cmd[0]:+.4f}"),
             ("speed_cmd_mps", f"{self.last_cmd[1]:.4f}"),
             ("speed_cap_mps", f"{self.speed_cap:.3f}"),
             ("n_beams", str(int(self.spec.n_beams))),
             ("range_max_m", f"{float(self.spec.range_max):.3f}"),
             ("v_max_mps", f"{float(self.spec.v_max):.3f}"),
             # The tracker's model of this car. Published because it is the difference between the
             # command this node sends and the one an in-process tracker with the simulator's own
             # per-car draw would send -- see docs/research/ros-graph-2026-09-15.md.
             ("sensor_timeout_s", f"{self.timeout:.3f}"),
             ("plan_timeout_s", f"{self.plan_timeout:.3f}"),
             ("cmd_delay_s", f"{float(self.delay[0]):.5f}"),
             ("wheelbase_m", f"{float(self.tracker.wb):.5f}"),
             ("steer_max_rad", f"{self.steer_max:.5f}"),
             ("steer_bias_rad", f"{self.cal[0]:.5f}"),
             ("steer_gain", f"{self.cal[1]:.5f}"),
             ("speed_gain", f"{self.cal[2]:.5f}")]
        if self.clearance is not None and self.clearance.last is not None:
            v.append(("clearance_after_m", f"{float(self.clearance.last.clear_after[0]):.3f}"))
        return v

    def _publish_diag(self, stamp, level: str, message: str):
        arr = DiagnosticArray(); arr.header.stamp = stamp
        st = DiagnosticStatus()
        st.level = {"OK": DiagnosticStatus.OK, "WARN": DiagnosticStatus.WARN,
                    "ERROR": DiagnosticStatus.ERROR}[level]
        st.name = "f1sim: controller"
        st.hardware_id = self.plan_checkpoint
        st.message = message or f"{self.controller_arm} / traction {self.traction_arm}"
        st.values = [KeyValue(key=k, value=str(x)) for k, x in self.diag_values()]
        arr.status = [st]
        self.pub_diag.publish(arr)

    def _publish_viz(self, snap: ScanSnapshot):
        """`/f1sim/viz/plan`: the reference the iLQR is following, in `base_link`.

        The tracker's `last_ref` is what the arms actually changed, so this is the plan after the
        clearance bend and the grip speed limit -- the thing the car is driving, not the thing the
        policy asked for.
        """
        if self.pub_viz is None or self.tracker.last_ref is None:
            return
        ref = self.tracker.last_ref[0].cpu().numpy()
        mk = Marker()
        mk.header.stamp = snap.stamp; mk.header.frame_id = "base_link"
        mk.ns = "f1sim_plan"; mk.id = 0
        mk.type = Marker.LINE_STRIP; mk.action = Marker.ADD
        mk.scale.x = 0.04
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.1, 0.9, 0.3, 0.9
        mk.pose.orientation.w = 1.0
        from geometry_msgs.msg import Point
        mk.points = [Point(x=float(p[0]), y=float(p[1]), z=0.05) for p in ref]
        self.pub_viz.publish(mk)
        if self.pub_viz_clear is not None:
            self._publish_clearance_viz(snap)
        if self.pub_viz_diag is not None:
            self._publish_diag_viz(snap)

    def _publish_diag_viz(self, snap: ScanSnapshot):
        """`/f1sim/viz/diag`: the arm, the friction, the guard and the command, as text."""
        mk = Marker()
        mk.header.stamp = snap.stamp; mk.header.frame_id = "base_link"
        mk.ns = "f1sim_diag"; mk.id = 0
        mk.type = Marker.TEXT_VIEW_FACING; mk.action = Marker.ADD
        mk.scale.z = 0.22
        mk.color.r = mk.color.g = mk.color.b = 0.95; mk.color.a = 0.95
        mk.pose.orientation.w = 1.0
        mk.pose.position.x, mk.pose.position.z = -0.6, 0.5
        mu = f" mu {float(self.grip.mu[0]):.3f}" if self.grip is not None else ""
        clear = ("" if self.clearance is None else
                 f" clearance {self.clearance.cspec.margin:.2f} m")
        mk.text = (f"{self.controller_arm}{mu}{clear}\n"
                   f"traction {self.traction_arm}"
                   + ("" if self.traction is None else f" ({self.traction_state})")
                   + f"\n{self.last_cmd[0]:+.3f} rad  {self.last_cmd[1]:.2f} m/s"
                   + f"  cap {self.speed_cap:.1f}\nplan #{self.plan_seq}")
        self.pub_viz_diag.publish(mk)

    def _publish_clearance_viz(self, snap: ScanSnapshot):
        """`/f1sim/viz/clearance`: the cells of this scan's occupancy grid, in `base_link`."""
        from f1sim.learn import clearance as cl
        from geometry_msgs.msg import Point
        c = self.clearance.cspec
        occ = cl.occupancy(self.clearance.scan, self.clearance.angles, c,
                           self.clearance.range_max, self.clearance.mount_x,
                           self.clearance.mount_y)[0]
        ys, xs = torch.nonzero(occ > 0.5, as_tuple=True)
        mk = Marker()
        mk.header.stamp = snap.stamp; mk.header.frame_id = "base_link"
        mk.ns = "f1sim_clearance"; mk.id = 0
        mk.type = Marker.CUBE_LIST; mk.action = Marker.ADD
        mk.scale.x = mk.scale.y = float(c.cell); mk.scale.z = 0.02
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.9, 0.4, 0.1, 0.5
        mk.pose.orientation.w = 1.0
        # Cell (ix, iy) spans [x_min + ix*cell, ...] x [-y_half + iy*cell, ...]; the marker wants
        # the centre. Same indexing as `clearance.occupancy`, read off the same spec.
        mk.points = [Point(x=float(c.x_min + (float(i) + 0.5) * c.cell),
                           y=float(-c.y_half + (float(j) + 0.5) * c.cell), z=0.0)
                     for i, j in zip(xs.tolist(), ys.tolist())]
        self.pub_viz_clear.publish(mk)


def main():
    rclpy.init(); n = ControllerNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    main()
