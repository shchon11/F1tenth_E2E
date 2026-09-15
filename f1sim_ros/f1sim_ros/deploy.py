"""What both halves of the graph share: the runtime arms, and the sensor intake underneath them.

`policy_node` and `controller_node` are two processes reading the same topics. Everything that
turns a message into a number they both use lives here, in one implementation, because the
load-bearing claim of the split -- that the graph publishes the same `/drive` the old monolithic
node did -- is only as strong as the amount of code the two sides cannot disagree about.

Three groups:

* the arms (`split_deployable`, `install_grip_arm`, `install_clearance_arm`,
  `build_traction_guard`), moved here verbatim from `policy_node` because they now belong to the
  controller and the policy must not be able to install one;
* the geometry constants the arms are built from (`LIDAR_FOV`, `LIDAR_MOUNT_X`);
* `SensorIntake`, the `/odom` + `/sensors/imu(/raw)` + `/scan` reader both nodes run. The policy
  needs the IMU mean for the observation and the controller needs its gyro-z for the tracker and
  its accelerometer for the traction guard, and those have to be the same number computed the same
  way from the same samples or the parity claim is a coincidence.
"""
import math

import numpy as np
import torch

from f1sim_ros.traction import TractionGuard, TractionParams


#: The scanner's nominal window and where it sits. 270 deg for the Hokuyo UST-10LX, and
#: base_link -> laser = (0.297, 0, 0.110) from `/tf_static` in all 22 recordings. The offset matters
#: for the clearance grid: the plan is in base_link and the returns are in the sensor's frame, and
#: 0.297 m is one and a half of the margin being defended.
LIDAR_FOV = 4.71238898
LIDAR_MOUNT_X = 0.297

G = 9.80665

#: IMU samples kept between scans. At 50 Hz against a 40 Hz scan that is one or two per scan; a
#: buffer allowed to grow without bound averages in samples from before the consumer stalled.
IMU_BUF_MAX = 16


#: The arms the controller will install. Both layers of each run on the car's own sensors:
#: `fixed_low` reads nothing at all, `clearance` reads `/scan` and nothing else. `oracle` and
#: `estimated` are simulator research arms -- the first is privileged, the second needs a frozen
#: estimator checkpoint -- and are refused rather than silently downgraded.
DEPLOYABLE_ARMS = ("legacy", "fixed_low", "clearance", "fixed_low+clearance")


def split_deployable(arm: str):
    """`"fixed_low+clearance"` -> `("fixed_low", True)`. Refuses anything this graph cannot run."""
    if arm not in DEPLOYABLE_ARMS:
        raise ValueError(f"controller arm {arm!r} is not deployable here: only "
                         f"{', '.join(DEPLOYABLE_ARMS)} run without a simulator-side estimator or "
                         f"privileged friction")
    from f1sim.learn.grip_runtime import split_arm
    parts = split_arm(arm)
    return parts.base, parts.clearance


def install_grip_arm(tracker, arm: str, device, mu: float = None):
    """Wrap the plan tracker's solver with the grip-aware speed/acceleration limits.

    `fixed_low` is the deployment default: a constant conservative friction (0.73423, the low end of
    the training range) limits corner speed and the acceleration/brake budgets of every plan the
    policy emits. On suite v1 (2026-09-12) it was the safest arm on every stability column against
    the same policy, at a 2 % lap-time cost; it needs no estimator and reads no sensor. `legacy`
    installs nothing. Returns the installed `GripMPC` (call `.release()` to undo) or None.

    A `+clearance` suffix is not this function's business: that layer binds `tracker._plan_hook`
    while this binds `tracker._solver`, so `install_clearance_arm` puts it on separately and the
    order the two are installed in cannot change the command.
    """
    if tracker is None or arm == "legacy":
        return None
    base, _clear = split_deployable(arm)
    if base == "legacy":
        return None
    from f1sim.learn import grip_control as gc
    spec = gc.GripSpec(mode="fixed", mu_fixed=float(gc.MU_FIXED_LOW if mu is None else mu)).validate()
    grip = gc.GripMPC(tracker, spec, 1, torch.device(device), tracker.wb, tracker.s_max, tracker.v_max)
    return grip.install(graph=False)


def install_clearance_arm(tracker, arm: str, device, spec, margin: float = None):
    """Install the `clearance` layer: the plan bent and slowed off what `/scan` can see.

    Built from the observation's own beam geometry (`ObsSpec.n_beams`, `range_max`) and the nominal
    270 deg window; `ControllerNode` re-declares the bearings from the first `LaserScan`'s own
    `angle_min` / `angle_max` if the driver publishes a different window, because a grid built from
    bearings the returns do not have is a silently rotated obstacle rather than an error.

    Returns the installed `ClearanceArm` (call `.release()` to undo) or None.
    """
    if tracker is None or arm == "legacy":
        return None
    _base, clear = split_deployable(arm)
    if not clear:
        return None
    from f1sim.learn import clearance as cl
    cspec = cl.ClearanceSpec() if margin is None else cl.ClearanceSpec(margin=float(margin))
    angles = cl.beam_angles(int(spec.n_beams), LIDAR_FOV, device=torch.device(device))
    arm_obj = cl.ClearanceArm(tracker, cspec.validate(), 1, torch.device(device),
                              float(spec.v_max), angles, float(spec.range_max),
                              mount_x=LIDAR_MOUNT_X)
    return arm_obj.install()


def build_traction_guard(arm: str, overrides: str = ""):
    """The `traction` parameter, as a `TractionGuard` or None.

    `off` (the default) installs nothing at all: no subscription is used for it, no command is
    touched, and the node behaves exactly as it did before this existed. `on` installs the guard
    validated in `scripts/replay_traction.py` against the 22 real recordings. `overrides` is the
    `traction_params` parameter: `name=value` pairs, comma or whitespace separated, applied to
    `TractionParams` -- every threshold is reachable from the launch line without editing code.

    Raises rather than falling back: an unreadable traction configuration on a car that is about to
    drive is not something to paper over with a default.
    """
    if arm in ("off", ""):
        return None
    if arm != "on":
        raise ValueError(f"traction must be 'off' or 'on', got {arm!r}")
    kw = {}
    fields = TractionParams.__dataclass_fields__
    for item in str(overrides).replace(",", " ").split():
        name, sep, value = item.partition("=")
        if not sep or name not in fields:
            raise ValueError(f"traction_params entry {item!r} is not NAME=VALUE for one of: "
                             f"{', '.join(sorted(fields))}")
        kw[name] = int(value) if fields[name].type == "int" else float(value)
    return TractionGuard(TractionParams(**kw).validate())


# ------------------------------------------------------------------ message arithmetic
#: A `LaserScan.ranges` field as the metres the observation is built from. Both nodes call it --
#: the policy hands the result to `ObsBuilder`, the controller normalises it for the clearance grid
#: -- and it lives in `learn/obs.py` with the rest of the observation contract rather than here,
#: because a scan resampled differently is a different observation and the training side has to be
#: able to reproduce it.
from f1sim.learn.obs import resample_ranges  # noqa: E402,F401


def quat_to_rp(q):
    sinr = 2 * (q.w * q.x + q.y * q.z); cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
    sinp = max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))
    return math.atan2(sinr, cosr), math.asin(sinp)


def covariance0(msg):
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


def header_seconds(m):
    h = getattr(m, "header", None)
    s = getattr(h, "stamp", None)
    if s is None:
        return None
    try:
        return float(s.sec) + float(s.nanosec) * 1e-9
    except (AttributeError, TypeError, ValueError):
        return None


class SensorIntake:
    """`/odom` + `/sensors/imu(/raw)` + `/scan` as the numbers a node drives on.

    One object, two nodes. The policy builds its observation from `imu_mean`, `v` and `att`; the
    controller feeds `imu_mean[2]` to the tracker as the measured yaw rate and `ax_body` to the
    traction guard. Those are the same quantities off the same topics, and the whole point of the
    split is that they stay the same quantities -- so there is one reader, called twice, rather
    than two readers that happen to agree today.

    `clock` is a callable returning monotonic seconds (a method on the node, so tests can drive it)
    and `log` is anything with `info` / `warning` / `error`.
    """

    def __init__(self, clock, log, timeout: float, accel_scale: float = 0.0):
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError(f"sensor_timeout must be a positive finite number, got {timeout}")
        self.clock, self.log, self.timeout = clock, log, float(timeout)
        self.v = 0.0
        self.imu_buf = []; self.imu_stamps = []
        self.att = (0.0, 0.0); self.yaw_rate = 0.0
        self.t_att = None; self.t_imu = None; self.t_odom = None; self.t_scan = None
        #: The last IMU mean actually built from fresh samples, and when those samples were taken.
        #: Reusing it across a dropped sample is the "brief gap" allowance; its age is *not*
        #: refreshed by the reuse, so a gap still expires at `timeout` rather than being renewed
        #: every scan.
        self.imu_mean = None; self.t_imu_mean = None
        self.att_stamp = None          # header stamp of the accepted orientation, when available
        self.accel_scale = float(accel_scale) or None            # None until detected
        self._unit_warned = False
        #: Latest body longitudinal acceleration [m/s^2] and motor current [A] for the traction
        #: guard, with the times they were taken.
        self.ax_body = None; self.t_ax = None
        self.motor_current = None; self.t_current = None

    # -- callbacks -------------------------------------------------------------
    def on_odom(self, m) -> bool:
        """`/odom` twist.linear.x. False for a sample that must not count as fresh."""
        v = m.twist.twist.linear.x
        if not math.isfinite(v):        # a NaN speed must not reach the actor, nor count as fresh
            return False
        self.v = v
        self.t_odom = self.clock()
        return True

    def on_core(self, m) -> bool:
        """`/sensors/core` motor current, the drive-torque corroboration for a spin. Optional: the
        guard treats None as "no information" rather than as "no torque", and one recording has no
        such topic at all."""
        i = getattr(getattr(m, "state", None), "current_motor", None)
        if i is None or not math.isfinite(float(i)):
            return False
        self.motor_current = float(i)
        self.t_current = self.clock()
        return True

    def on_imu(self, m) -> bool:
        now = self.clock()
        w, a = m.angular_velocity, m.linear_acceleration
        raw = (w.x, w.y, w.z, a.x, a.y, a.z)
        # Checked *before* `_accel_to_si`, not after: that function decides g vs SI from the first
        # sample's magnitude, and `nan < 3.0` is False, so one NaN sample would latch scale = 1.0
        # and every later valid reading would be divided by 9.81 for the rest of the run.
        if not all(math.isfinite(c) for c in raw):
            return False                # a NaN channel must not reach the actor nor look fresh
        row = [raw[0], raw[1], raw[2], *self._accel_to_si(raw[3], raw[4], raw[5])]
        self.imu_buf.append(row)
        self.imu_stamps.append(now)
        self.ax_body = row[3]; self.t_ax = now   # SI body x acceleration, for the traction guard
        # Bounded: one scan period at 50 Hz is a couple of samples, so anything beyond a handful
        # means the consumer stalled and the oldest of them do not belong to the next scan.
        while len(self.imu_buf) > IMU_BUF_MAX:
            self.imu_buf.pop(0); self.imu_stamps.pop(0)
        self.yaw_rate = m.angular_velocity.z
        self.t_imu = now
        self.note_attitude(attitude_from_orientation(m.orientation, covariance0(m)),
                           header_seconds(m))
        return True

    def on_vesc_imu(self, m) -> bool:
        """The VESC summary carries the same quaternion the driver puts on `/sensors/imu/raw`
        (`vesc_driver.cpp:243-246`, both filled from `q_w/q_x/q_y/q_z`). Its `ypr` field is not
        used: see `attitude_from_orientation`."""
        q = getattr(getattr(m, "imu", None), "orientation", None)
        if q is None:
            return False
        return self.note_attitude(attitude_from_orientation(q, covariance0(m.imu)),
                                  header_seconds(m))

    def note_attitude(self, att, stamp=None) -> bool:
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
            return False                # demonstrably older than what we already accepted
        self.att = att
        self.att_stamp = stamp
        self.t_att = self.clock()
        return True

    def _accel_to_si(self, ax, ay, az):
        """Scale linear_acceleration to m/s^2, detecting g vs SI from the gravity vector once.

        The rule is `calib.bagread.accel_scale_for`, shared with the bag reader: a recording read
        on one convention and driven on the other is a silent factor of 9.81, and the place that
        would be found is a car that does not brake.
        """
        from f1sim.calib.bagread import accel_scale_for
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if self.accel_scale is None:
            detected = accel_scale_for(mag)
            if detected is None:                                 # no gravity yet: cannot tell
                return ax, ay, az
            self.accel_scale = detected
            self.log.warning(
                f"IMU linear_acceleration |a|={mag:.2f} on the first sample -> treating it as "
                f"{'g' if self.accel_scale != 1.0 else 'm/s^2'} (scale {self.accel_scale:.5f}). "
                f"Override with the imu_accel_scale parameter if this is wrong.")
        elif not self._unit_warned and 0.2 < mag * self.accel_scale < 3.0:
            self._unit_warned = True                             # scaled gravity should be ~9.8
            self.log.error(f"IMU |a|={mag * self.accel_scale:.2f} m/s^2 after scaling by "
                           f"{self.accel_scale:.3f}: the unit assumption looks wrong.")
        k = self.accel_scale
        return ax * k, ay * k, az * k

    # -- per scan --------------------------------------------------------------
    def take_imu_mean(self, now):
        """Consume the samples this scan is entitled to and return the mean, or the previous one.

        Samples no older than `timeout`. That window is wider than one scan period on purpose: it
        is the freshness bound, not a per-scan boundary, so a sample that arrived slightly before
        this scan still counts while one from a stall does not.
        """
        fresh = [v for v, t in zip(self.imu_buf, self.imu_stamps) if now - t <= self.timeout]
        oldest = min((t for t in self.imu_stamps if now - t <= self.timeout), default=None)
        self.imu_buf = []; self.imu_stamps = []
        if fresh:
            # A new mean, dated by the samples it came from -- not by the scan that consumed them.
            self.imu_mean = np.mean(fresh, 0)
            self.t_imu_mean = oldest
        # else: keep the previous mean *and its age*, so a dropped sample is tolerated but a real
        # gap still expires at `timeout` instead of being renewed by every scan.
        return self.imu_mean

    def stale(self, now, check_scan=False):
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

    def ages(self, now):
        """Age of each input [s], -1.0 for one never seen. Diagnostics only."""
        age = lambda t: -1.0 if t is None else float(now - t)
        return {"scan": age(self.t_scan), "imu": age(self.t_imu_mean), "odom": age(self.t_odom),
                "attitude": age(self.t_att)}

    def fresh(self, t, now=None):
        """Is a timestamp inside the freshness window? Used for the traction guard's inputs, which
        are passed as None rather than as their last value: the guard holds its filters over a
        missing sample, and `update` re-seeds rather than detecting across a gap."""
        return t is not None and (self.clock() if now is None else now) - t <= self.timeout
