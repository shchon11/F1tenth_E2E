"""LiDAR-only e2e policy node: /scan + /odom + /sensors/imu(/raw) -> /drive at the scan rate.
Uses f1sim.learn.obs.ObsBuilder, the same encoding the policy was trained with. Works against the
simulator (f1tenth_stack_sim.launch.py) and the real f1tenth_stack unchanged."""
import math
import time

import numpy as np
import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan

from f1sim.learn.model import load_checkpoint
from f1sim.learn.obs import ObsBuilder, ObsSpec

try:
    from vesc_msgs.msg import VescImuStamped
except ImportError:
    VescImuStamped = None


G = 9.80665

#: IMU samples kept between scans. At 50 Hz against a 40 Hz scan that is one or two per scan; a
#: buffer allowed to grow without bound averages in samples from before the consumer stalled.
IMU_BUF_MAX = 16


def quat_to_rp(q):
    sinr = 2 * (q.w * q.x + q.y * q.z); cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
    sinp = max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))
    return math.atan2(sinr, cosr), math.asin(sinp)


def _covariance0(msg):
    """`orientation_covariance[0]`, or 0.0 when the field is absent or empty.

    Written out rather than `cov[0] if cov else 0.0` because on Humble this field is a
    `numpy.ndarray` of nine doubles (`sensor_msgs/msg/_imu.py`), and truth-testing an array with
    more than one element raises `ValueError: The truth value of an array ... is ambiguous`. That
    would have fired on every real message while passing every test built from a Python list.
    """
    cov = getattr(msg, "orientation_covariance", None)
    if cov is None or len(cov) == 0:
        return 0.0
    return float(cov[0])


def attitude_from_orientation(q, cov0=0.0):
    """(roll, pitch) from a quaternion, or None if it is not a usable orientation.

    The policy reads this channel as body roll and pitch: the training side builds it from
    `r.imu_att[:, :2]` in `gym_env._obs`. Only a canonical quaternion is accepted, because the
    alternative on this hardware -- `VescImu.ypr` -- has neither a standard component order nor a
    standard unit. The recordings show the driver writing `ypr.x = roll, ypr.y = pitch,
    ypr.z = yaw` in degrees (`ypr.z` spans 357-360 deg and tracks the integrated gyro at
    |r| = 0.994-1.000 over 9 bags), an earlier version of this node read `ypr.z` as roll, and the
    simulator bridges do not agree with either. A quaternion means the same thing everywhere, so it
    is the only source used.

    Refused: the all-zero default of an unset field, a non-finite component, anything that is not a
    unit quaternion, and `orientation_covariance[0] == -1`, which is how a ROS publisher states it
    has no orientation estimate. The simulator bridge marks every IMU sample that way except the
    one where the attitude is actually measured, so honouring it is what lets this node work there
    without any topic or message-type discovery.
    """
    if cov0 == -1.0:
        return None
    w, x, y, z = q.w, q.x, q.y, q.z
    if not all(math.isfinite(c) for c in (w, x, y, z)):
        return None
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if abs(n - 1.0) > 1e-3:                     # also covers the all-zero default, where n = 0
        return None
    return quat_to_rp(q)


class PolicyNode(Node):
    def __init__(self):
        super().__init__("f1sim_policy")
        self.declare_parameter("checkpoint", ""); self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("speed_cap", 4.0); self.declare_parameter("steer_max", 0.4189)
        self.declare_parameter("drive_topic", "drive"); self.declare_parameter("enabled", True)
        # This car's /sensors/imu/raw publishes linear_acceleration in **g**, not the m/s^2 the Imu
        # message specifies (az ~ 1.00 at rest in all 22 recordings). The policy was trained on SI,
        # so feeding g straight through scales the accelerometer channel by 1/9.81 and the mistake
        # is silent -- the car just drives with a dead input. 0.0 = detect from the gravity vector,
        # which is unambiguous because the two candidates are an order of magnitude apart.
        self.declare_parameter("imu_accel_scale", 0.0)
        p = lambda n: self.get_parameter(n).value
        self.device = torch.device(p("device"))
        self.model, extra = load_checkpoint(p("checkpoint"), self.device); self.model.eval()
        self.spec = ObsSpec(**extra["spec"]) if extra.get("spec") else ObsSpec()
        self.obs = ObsBuilder(self.spec, self.device)
        self.speed_cap = float(p("speed_cap")); self.steer_max = float(p("steer_max"))
        self.v = 0.0; self.imu_buf = []; self.att = (0.0, 0.0); self.yaw_rate = 0.0
        # Freshness. Every input the actor reads carries the time it was last valid, and inference
        # is inhibited when any of them goes stale. Holding the last value through a dropped sample
        # is fine; driving on a value from a second ago is not, and the previous version had no way
        # to tell the two apart -- `VescImuStamped` being importable was taken as evidence that the
        # topic was live, which it is not (one recording has no `/sensors/imu` at all).
        self.declare_parameter("sensor_timeout", 0.25)
        self.timeout = float(p("sensor_timeout"))
        if not math.isfinite(self.timeout) or self.timeout <= 0.0:
            raise ValueError(f"sensor_timeout must be a positive finite number, got {self.timeout}")
        self.t_att = None; self.t_imu = None; self.t_odom = None; self.t_scan = None
        self.imu_stamps = []
        #: The last IMU mean actually built from fresh samples, and when those samples were taken.
        #: Reusing it across a dropped sample is the "brief gap" allowance; its age is *not*
        #: refreshed by the reuse, so a gap still expires at `timeout` rather than being renewed
        #: every scan.
        self.imu_mean = None; self.t_imu_mean = None
        self.att_stamp = None            # header stamp of the accepted orientation, when available
        self._inhibited = False; self._last_inhibit_log = 0.0
        self.accel_scale = float(p("imu_accel_scale")) or None      # None until detected
        self._unit_warned = False
        # plan action space: the same iLQR tracker as in training turns the local trajectory into
        # (steer, speed); cmd_delay = the measured command latency of this car (calibrate once)
        self.declare_parameter("wheelbase", 0.3302); self.declare_parameter("cmd_delay", 0.035)
        # Residual servo calibration the stack's steering_angle_to_servo_offset/gain do not absorb
        # (rad, ratio); the published angle is (cmd - steer_bias) / steer_gain.
        #
        # Measured by replaying 233 windows of recorded commands through the simulator and matching
        # the yaw rate it produces against the car's gyro: the simulator turns about 1.2x too much
        # for a given command, so steer_gain ~ 0.85. (Lowering the tyre friction instead also matches
        # the amplitude but fits worse -- correlation 0.63 against 0.72 -- which is weak evidence
        # for a steering gain rather than a slippery floor.) Left at 1.0 by default because that
        # number belongs to ONE servo configuration:
        #
        #   competition (0826-0827):     servo = 0.40 + 0.52 * steering_angle
        #   pre-competition (0714-0727): servo = 0.61 - 0.35 * steering_angle
        #
        # The polarity is opposite between the two. Check which configuration the car is running
        # and confirm the steering direction at walking pace before trusting any policy with it.
        self.declare_parameter("steer_bias", 0.0); self.declare_parameter("steer_gain", 1.0); self.declare_parameter("speed_gain", 1.0)
        self.cal = (float(p("steer_bias")), float(p("steer_gain")), float(p("speed_gain")))
        self.tracker = None
        if self.spec.act_dim >= 5:
            from f1sim.mpc import PlanTracker
            self.tracker = PlanTracker(1, self.device, float(p("wheelbase")), self.steer_max, self.spec.v_max)
            self.delay = torch.tensor([float(p("cmd_delay"))], device=self.device)
        self.create_subscription(Odometry, "odom", self.on_odom, 1)
        self.create_subscription(Imu, "sensors/imu/raw", self.on_imu, 10)
        if VescImuStamped is not None:
            self.create_subscription(VescImuStamped, "sensors/imu", self.on_vesc_imu, 1)
        self.create_subscription(LaserScan, "scan", self.on_scan, 1)
        # The scan callback cannot notice its own absence; this can.
        self.create_timer(max(0.02, self.timeout / 4.0), self.on_watchdog)
        self.pub = self.create_publisher(AckermannDriveStamped, p("drive_topic"), 1)
        self.last_t = None
        # warm up
        s, pr = self.obs.build(np.full(self.spec.n_beams, 5.0), 0.0, np.zeros(6), np.zeros(2), self.speed_cap)
        with torch.no_grad():
            a0, _ = self.model.act(s, pr, deterministic=True)
        if self.tracker is not None:                                 # warm up the tracker's compiled solver too
            self.tracker(a0, torch.zeros(1, device=self.device), torch.tensor([self.speed_cap], device=self.device), None, delay=self.delay)
            self.tracker.reset(torch.zeros(1, dtype=torch.long, device=self.device))
        self.obs.reset()
        self.get_logger().info(f"policy {p('checkpoint')} on {self.device}, speed cap {self.speed_cap} m/s")

    def on_odom(self, m: Odometry):
        v = m.twist.twist.linear.x
        if not math.isfinite(v):            # a NaN speed must not reach the actor, nor count as fresh
            return
        self.v = v
        self.t_odom = self.clock()

    def clock(self):
        """Monotonic seconds. A method so tests can drive it without a ROS clock."""
        return time.monotonic()

    def stamp_now(self):
        """Current ROS time as a message stamp. Overridden in tests."""
        return self.get_clock().now().to_msg()

    @staticmethod
    def _header_seconds(m):
        h = getattr(m, "header", None)
        s = getattr(h, "stamp", None)
        if s is None:
            return None
        try:
            return float(s.sec) + float(s.nanosec) * 1e-9
        except (AttributeError, TypeError, ValueError):
            return None

    def _note_attitude(self, att, stamp=None):
        """Adopt an attitude if it is usable.

        Both orientation sources are accepted and the most recent one wins, so a valid raw sample
        never permanently disables the VESC one or the reverse. "Most recent" uses the message
        header stamp when both messages carry one; when they do not, it falls back to **callback
        arrival order**, which is not the same thing -- a message delayed in transport can arrive
        after a newer one. That limit is accepted here rather than hidden: the two sources are the
        same sensor a few milliseconds apart, so the ordering only matters for which of two nearly
        identical attitudes is used.
        """
        if att is None:
            return False
        if stamp is not None and self.att_stamp is not None and stamp < self.att_stamp:
            return False                     # demonstrably older than what we already accepted
        self.att = att
        self.att_stamp = stamp
        self.t_att = self.clock()
        return True

    def _accel_to_si(self, ax, ay, az):
        """Scale linear_acceleration to m/s^2, detecting g vs SI from the gravity vector once."""
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if self.accel_scale is None:
            if mag < 0.2:                                            # no gravity yet: cannot tell
                return ax, ay, az
            self.accel_scale = G if mag < 3.0 else 1.0
            self.get_logger().warning(
                f"IMU linear_acceleration |a|={mag:.2f} on the first sample -> treating it as "
                f"{'g' if self.accel_scale != 1.0 else 'm/s^2'} (scale {self.accel_scale:.5f}). "
                f"Override with the imu_accel_scale parameter if this is wrong.")
        elif not self._unit_warned and 0.2 < mag * self.accel_scale < 3.0:
            self._unit_warned = True                                 # scaled gravity should be ~9.8
            self.get_logger().error(f"IMU |a|={mag * self.accel_scale:.2f} m/s^2 after scaling by "
                                    f"{self.accel_scale:.3f}: the unit assumption looks wrong.")
        k = self.accel_scale
        return ax * k, ay * k, az * k

    def on_imu(self, m: Imu):
        now = self.clock()
        w, a = m.angular_velocity, m.linear_acceleration
        raw = (w.x, w.y, w.z, a.x, a.y, a.z)
        # Checked *before* `_accel_to_si`, not after: that function decides g vs SI from the first
        # sample's magnitude, and `nan < 3.0` is False, so one NaN sample would latch scale = 1.0
        # and every later valid reading would be divided by 9.81 for the rest of the run.
        if not all(math.isfinite(c) for c in raw):
            return                          # a NaN channel must not reach the actor nor look fresh
        row = [raw[0], raw[1], raw[2], *self._accel_to_si(raw[3], raw[4], raw[5])]
        self.imu_buf.append(row)
        self.imu_stamps.append(now)
        # Bounded: one scan period at 50 Hz is a couple of samples, so anything beyond a handful
        # means the consumer stalled and the oldest of them do not belong to the next scan.
        while len(self.imu_buf) > IMU_BUF_MAX:
            self.imu_buf.pop(0); self.imu_stamps.pop(0)
        self.yaw_rate = m.angular_velocity.z
        self.t_imu = now
        self._note_attitude(
            attitude_from_orientation(m.orientation, _covariance0(m)), self._header_seconds(m))

    def on_vesc_imu(self, m):
        """The VESC summary carries the same quaternion the driver puts on `/sensors/imu/raw`
        (`vesc_driver.cpp:243-246`, both filled from `q_w/q_x/q_y/q_z`). Its `ypr` field is not
        used: see `attitude_from_orientation`."""
        q = getattr(getattr(m, "imu", None), "orientation", None)
        if q is None:
            return
        self._note_attitude(
            attitude_from_orientation(q, _covariance0(m.imu)), self._header_seconds(m))

    def _stale_inputs(self, now, check_scan=False):
        """Which actor inputs are not fresh enough to drive on. Empty means go."""
        out = []
        if self.t_imu_mean is None or now - self.t_imu_mean > self.timeout:
            out.append("imu")
        if self.t_att is None or now - self.t_att > self.timeout:
            out.append("attitude")
        if self.t_odom is None or now - self.t_odom > self.timeout:
            out.append("odom")
        if check_scan and (self.t_scan is None or now - self.t_scan > self.timeout):
            out.append("scan")
        return out

    def on_watchdog(self):
        """Stop driving when the scans themselves stop.

        Every other guard lives in `on_scan`, which is exactly the callback that stops running when
        the LiDAR goes away -- leaving the last command standing with nothing to countermand it.
        This runs on a timer instead. It never calls the actor; it only inhibits, and it stamps with
        the current time rather than the header of a scan that may be seconds old.
        """
        now = self.clock()
        stale = self._stale_inputs(now, check_scan=True)
        if stale:
            self._inhibit(stale, self.stamp_now())

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
        now = time.monotonic()
        if now - self._last_inhibit_log > 1.0:
            self._last_inhibit_log = now
            self.get_logger().warning(
                f"no actor command: {', '.join(stale)} older than {self.timeout:.2f} s. "
                f"Publishing zero speed until the sensors come back.")

    def _resume(self):
        """A gap ended. The observation and tracker histories describe a segment that is over, and
        stitching the new one onto them feeds the policy an action history that never happened."""
        self._inhibited = False
        self.obs.reset()
        if self.tracker is not None:
            self.tracker.reset(torch.zeros(1, dtype=torch.long, device=self.device))
        self.get_logger().info("sensors recovered: observation and tracker history cleared")

    def on_scan(self, m: LaserScan):
        t0 = time.perf_counter()
        r = np.asarray(m.ranges, dtype=np.float32)
        # A miss reads 65.533 m (the 0xFFFF mm sentinel), which is finite, so interpolating across
        # it invents ranges: one missed beam next to a 1 m wall becomes a 33 m "return" halfway
        # between them. Saturate first, then resample.
        r = np.where(np.isfinite(r) & (r > 0) & (r < 65.0), r, np.float32(m.range_max))
        if len(r) != self.spec.n_beams:                              # e.g. 1081 beams from urg_node: resample
            r = np.interp(np.linspace(0, len(r) - 1, self.spec.n_beams), np.arange(len(r)), r)
        now = self.clock()
        self.t_scan = now
        # Samples no older than `timeout`. That window is wider than one scan period on purpose:
        # it is the freshness bound, not a per-scan boundary, so a sample that arrived slightly
        # before this scan still counts while one from a stall does not.
        fresh = [v for v, t in zip(self.imu_buf, self.imu_stamps) if now - t <= self.timeout]
        oldest = min((t for t in self.imu_stamps if now - t <= self.timeout), default=None)
        self.imu_buf = []; self.imu_stamps = []
        if fresh:
            # A new mean, dated by the samples it came from -- not by the scan that consumed them.
            self.imu_mean = np.mean(fresh, 0)
            self.t_imu_mean = oldest
        # else: keep the previous mean *and its age*, so a dropped sample is tolerated but a real
        # gap still expires at `timeout` instead of being renewed by every scan.
        stale = self._stale_inputs(now)
        if stale:
            self._inhibit(stale, self.stamp_now())
            return
        if self._inhibited:                     # coming back from a gap
            self._resume()
        imu_mean = self.imu_mean
        scan, pro = self.obs.build(r, self.v, imu_mean, self.att, self.speed_cap)
        with torch.no_grad():
            a, _ = self.model.act(scan, pro, deterministic=True)
        self.obs.push_action(a[0])
        msg = AckermannDriveStamped(); msg.header.stamp = m.header.stamp
        if self.tracker is not None:                                 # local plan -> tracker -> command
            cmd = self.tracker(a, torch.tensor([self.v], device=self.device), torch.tensor([self.speed_cap], device=self.device),
                               torch.tensor([float(imu_mean[2])], device=self.device), delay=self.delay)[0]
            msg.drive.steering_angle = float(max(-self.steer_max, min(self.steer_max, (float(cmd[0]) - self.cal[0]) / self.cal[1])))
            msg.drive.speed = float(cmd[1]) / self.cal[2]
        else:
            a = a[0].cpu().numpy()
            msg.drive.steering_angle = float(a[0] * self.steer_max)
            msg.drive.speed = float(min((a[1] + 1) * 0.5 * self.spec.v_max, self.speed_cap))
        if self.get_parameter("enabled").value:
            self.pub.publish(msg)
        dt = (time.perf_counter() - t0) * 1000
        if self.last_t is None or time.perf_counter() - self.last_t > 5:
            self.get_logger().info(f"inference {dt:.1f} ms  v={self.v:.2f} cmd=({msg.drive.steering_angle:+.2f} rad, {msg.drive.speed:.2f} m/s)")
            self.last_t = time.perf_counter()


def main():
    rclpy.init(); n = PolicyNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    main()
