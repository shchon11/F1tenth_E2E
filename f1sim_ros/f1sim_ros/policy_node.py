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
from std_msgs.msg import Empty as EmptyMsg

from f1sim.learn.memory import describe as describe_memory, runtime_for
from f1sim.learn.model import load_checkpoint


#: The scanner's nominal window and where it sits. 270 deg for the Hokuyo UST-10LX, and
#: base_link -> laser = (0.297, 0, 0.110) from `/tf_static` in all 22 recordings. The offset matters
#: for the clearance grid: the plan is in base_link and the returns are in the sensor's frame, and
#: 0.297 m is one and a half of the margin being defended.
LIDAR_FOV = 4.71238898
LIDAR_MOUNT_X = 0.297
#: The scan plane's height above the floor at rest, the same 0.110 m of that `/tf_static` line. It
#: is what makes a two-degree tilt a wall at 3.2 m, and it is the floor gate's whole geometry.
LIDAR_MOUNT_Z = 0.110
#: [Hz] the policy / LiDAR rate (`f1sim.params.SimParams.control_rate`), for the attitude tracker's
#: integration step.
CONTROL_RATE = 40.0


#: The arms this node will install. Both layers of each run on the car's own sensors: `fixed_low`
#: reads nothing at all, `clearance` reads `/scan` and nothing else. `oracle` and `estimated` are
#: simulator research arms -- the first is privileged, the second needs a frozen estimator
#: checkpoint -- and are refused rather than silently downgraded.
DEPLOYABLE_ARMS = ("legacy", "fixed_low", "clearance", "fixed_low+clearance")


def split_deployable(arm: str):
    """`"fixed_low+clearance"` -> `("fixed_low", True)`. Refuses anything this node cannot run."""
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


def install_clearance_arm(tracker, arm: str, device, spec, margin: float = None,
                          floor_gate: bool = False):
    """Install the `clearance` layer: the plan bent and slowed off what `/scan` can see.

    Built from the observation's own beam geometry (`ObsSpec.n_beams`, `range_max`) and the nominal
    270 deg window; `PolicyNode` re-declares the bearings from the first `LaserScan`'s own
    `angle_min` / `angle_max` if the driver publishes a different window, because a grid built from
    bearings the returns do not have is a silently rotated obstacle rather than an error.

    `floor_gate` turns on the floor-aware occupancy (`learn/floor.py`): a return the geometry says
    is the FLOOR, given the scan plane's tilt, is left out of the grid instead of bending the plan
    around it. The tilt comes from `ClearanceArm`'s own `AttitudeTracker`, fed from `/sensors/imu/raw`
    by `on_scan` -- **not** from the orientation quaternion, which five of the thirteen competition
    recordings swing past 40 deg and which in simulation is wrong by 8 deg rms while driving. Until
    the car has stood still once the tracker has no zero reference, the likelihood reads
    `floor.UNKNOWN` and the gate removes nothing.

    Returns the installed `ClearanceArm` (call `.release()` to undo) or None.
    """
    if tracker is None or arm == "legacy":
        return None
    _base, clear = split_deployable(arm)
    if not clear:
        return None
    from f1sim.learn import clearance as cl
    kw = {"floor_gate": bool(floor_gate)}
    if margin is not None:
        kw["margin"] = float(margin)
    cspec = cl.ClearanceSpec(**kw)
    angles = cl.beam_angles(int(spec.n_beams), LIDAR_FOV, device=torch.device(device))
    from f1sim.learn import floor as fl
    arm_obj = cl.ClearanceArm(tracker, cspec.validate(), 1, torch.device(device),
                              float(spec.v_max), angles, float(spec.range_max),
                              mount_x=LIDAR_MOUNT_X,
                              fspec=fl.FloorSpec(mount_x=LIDAR_MOUNT_X, mount_z=LIDAR_MOUNT_Z),
                              dt=1.0 / float(CONTROL_RATE))
    return arm_obj.install()
from f1sim.learn.obs import ObsBuilder, ObsSpec

from f1sim_ros.traction import TractionGuard, TractionParams

try:
    from vesc_msgs.msg import VescImuStamped
except ImportError:
    VescImuStamped = None

try:
    from vesc_msgs.msg import VescStateStamped
except ImportError:
    VescStateStamped = None


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
    #: Class defaults, so a node the tests build with `__new__` -- several fixtures hand-assemble a
    #: minimal node to exercise one callback -- starts from the shipped behaviour instead of an
    #: AttributeError. `__init__` overwrites both from the parameters.
    att_source = "vesc"
    ego_att = None

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
        #: The policy's episode state: the recurrent hidden state and the decayed scan-occupancy
        #: channel, for this one car. Empty (and `stateful` False) for a feedforward checkpoint, so
        #: nothing below changes for one. It is cleared in exactly two places, both of which mean
        #: "the run you remember is over": `_resume`, when the scan stream comes back after a gap,
        #: and `on_reset`, when something resets the car. Carrying it across either would drive a
        #: car that has been picked up and put down with the memory of where it used to be.
        self.policy_state = runtime_for(self.model, batch=1, device=self.device)
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
        # Plan controller arm. "fixed_low" (default) limits corner speed and accel/brake budgets for a
        # conservative constant friction; "legacy" is the untouched tracker. Set `grip_mu` to override
        # the constant once the floor has been characterised. A "+clearance" suffix adds the geometry
        # layer: a local occupancy built from this scan alone (no map), and the plan bent and slowed
        # until it keeps `clearance_margin` from anything the scanner saw.
        self.declare_parameter("controller", "fixed_low"); self.declare_parameter("grip_mu", 0.0)
        self.declare_parameter("clearance_margin", 0.0)
        # The floor-aware occupancy. Off by default, and off it is byte-identical to the arm that
        # shipped: `clearance.occupancy` does not look at the likelihood at all. See
        # `install_clearance_arm` and `docs/ros2.md`.
        self.declare_parameter("clearance_floor_gate", False)
        # Where the policy's roll/pitch observation columns come from. `vesc` is the orientation
        # quaternion this node has always read; `ego` is `f1sim.learn.floor.EgoStateAttitude`, the
        # suspension's calibrated response to the accelerations the car itself produces, computed
        # from the wheel speed, the gyro's yaw and the accelerometer. Two reasons to have the
        # choice, both measured and both in `docs/research/floor-mask-2026-09-15.md`:
        #
        #   * the quaternion is wrong by 0.14 rad rms in roll and 0.09 in pitch while driving,
        #     against 0.022 / 0.022 for the ego-state path -- and those columns are a POLICY INPUT,
        #     not only the floor channel's;
        #   * five of the thirteen competition recordings carry a quaternion that swings the
        #     extracted roll past 40 deg, and on those this node refuses to drive at all today.
        #     The ego-state path does not use the quaternion, so it survives them.
        #
        # `vesc` stays the default: every trained checkpoint saw `imu_att` in training, and
        # swapping the meaning of two observation columns under a policy is a change to make
        # deliberately with a finetune behind it, not a default.
        self.declare_parameter("attitude_source", "vesc")
        self.controller_arm = str(p("controller"))
        self.grip = install_grip_arm(self.tracker, self.controller_arm, self.device,
                                     mu=(float(p("grip_mu")) or None))
        self.clearance = install_clearance_arm(self.tracker, self.controller_arm, self.device,
                                               self.spec,
                                               margin=(float(p("clearance_margin")) or None),
                                               floor_gate=bool(p("clearance_floor_gate")))
        #: Whether the beam bearings the clearance grid is built from have been checked against a
        #: real `LaserScan` header yet. Until then they are the nominal 270 deg window.
        self._scan_geometry_checked = False
        #: The ego-state attitude estimator. Built whenever anything could read it -- the policy's
        #: own observation (`attitude_source:=ego`) or a front-end channel the checkpoint declares
        #: -- and advanced every scan so those two cannot see different attitudes.
        self.att_source = str(p("attitude_source"))
        if self.att_source not in ("vesc", "ego", "frontend"):
            raise ValueError(f"attitude_source must be 'vesc', 'ego' or 'frontend', got "
                             f"{self.att_source!r}")
        if self.att_source == "frontend" and getattr(self.policy_state, "scan", None) is None:
            raise ValueError(
                "attitude_source:=frontend needs a checkpoint that declares a front-end channel "
                "(meta['scan_channels'] with fe_floor / fe_range): the estimate is one of that "
                "network's outputs and there is nothing to read without it.")
        from f1sim.learn import floor as _floor
        self.ego_att = _floor.EgoStateAttitude(1, device=self.device, dt=1.0 / float(CONTROL_RATE),
                                               source="wheel")
        if self.att_source == "ego":
            self.get_logger().warning(
                "attitude_source:=ego -- the policy's roll/pitch observation columns now come from "
                "f1sim.learn.floor.EgoStateAttitude, not from the orientation quaternion. The "
                "checkpoint was trained against the quaternion unless it says otherwise; see "
                "docs/research/floor-mask-2026-09-15.md for what each is worth.")
        # Traction guard: wheel lock / launch spin from `/odom` wheel speed against the IMU, with
        # the speed command shaped when either fires. OFF by default -- it has been validated only
        # by replaying the recordings (`scripts/replay_traction.py`), never on the moving car, and
        # it is the one thing in this node that can raise a commanded speed the policy lowered.
        # See docs/ros2.md, "Traction guard".
        self.declare_parameter("traction", "off"); self.declare_parameter("traction_params", "")
        self.traction_arm = str(p("traction"))
        self.traction = build_traction_guard(self.traction_arm, str(p("traction_params")))
        #: Latest body longitudinal acceleration [m/s^2] and motor current [A] for the guard, with
        #: the times they were taken. The guard is fed at the `/odom` rate -- 50 Hz on this car, and
        #: the rate the replay validated it at -- not at the scan rate, so no wheel-speed sample is
        #: skipped; `shape` then runs once per published command.
        self.ax_body = None; self.t_ax = None
        self.motor_current = None; self.t_current = None
        if self.traction is not None and VescStateStamped is not None:
            self.create_subscription(VescStateStamped, "sensors/core", self.on_core, 1)
        self.create_subscription(Odometry, "odom", self.on_odom, 1)
        self.create_subscription(Imu, "sensors/imu/raw", self.on_imu, 10)
        if VescImuStamped is not None:
            self.create_subscription(VescImuStamped, "sensors/imu", self.on_vesc_imu, 1)
        self.create_subscription(LaserScan, "scan", self.on_scan, 1)
        # `/f1sim/reset` as a std_msgs/Empty TOPIC, announced by the simulator nodes after they
        # reset (`bridge_node`, `vesc_sim_node`). It is deliberately not the std_srvs/Empty SERVICE
        # of the same name: topic and service names are separate in the ROS graph, and offering a
        # second server for that service would make which node answers a reset ambiguous. On the
        # real car nothing publishes it and the node behaves exactly as it did.
        self.declare_parameter("reset_topic", "/f1sim/reset")
        self.create_subscription(EmptyMsg, str(p("reset_topic")), self.on_reset, 1)
        # The scan callback cannot notice its own absence; this can.
        self.create_timer(max(0.02, self.timeout / 4.0), self.on_watchdog)
        self.pub = self.create_publisher(AckermannDriveStamped, p("drive_topic"), 1)
        self.last_t = None
        # warm up
        s, pr = self.obs.build(np.full(self.spec.n_beams, 5.0), 0.0, np.zeros(6), np.zeros(2), self.speed_cap)
        with torch.no_grad():
            a0, _, _ = self.model.act(self.policy_state.observe(s, pr), pr, deterministic=True,
                                      h=self.policy_state.hidden)
        if self.tracker is not None:                                 # warm up the tracker's compiled solver too
            self.tracker(a0, torch.zeros(1, device=self.device), torch.tensor([self.speed_cap], device=self.device), None, delay=self.delay)
            self.tracker.reset(torch.zeros(1, dtype=torch.long, device=self.device))
        self.obs.reset()
        self.policy_state.reset()                # the warm-up is not part of any episode
        if self.policy_state.stateful:
            self.get_logger().info(f"policy memory: {describe_memory(self.model.meta)}; hidden state "
                                   f"carried across scan callbacks, cleared on "
                                   f"{p('reset_topic')} and when the scan stream restarts")
        self.get_logger().info(f"policy {p('checkpoint')} on {self.device}, speed cap {self.speed_cap} m/s, "
                               f"controller {self.controller_arm}"
                               + (f" (mu {float(self.grip.mu[0]):.3f})" if self.grip is not None else "")
                               + ("" if self.clearance is None else
                                  f", clearance margin {self.clearance.cspec.margin:.2f} m body edge "
                                  f"({self.clearance.cspec.margin + self.clearance.cspec.body_radius:.2f} m "
                                  f"centre) on a {self.clearance.cspec.cell * 100:.0f} cm grid from /scan alone")
                               + f", traction {self.traction_arm}"
                               + ("" if self.traction is None else
                                  f" (lock past {self.traction.p.lock_accel:.1f} m/s^2 wheel decel "
                                  f"and {self.traction.p.lock_rate:.1f} m/s^2 residual, spin past "
                                  f"{self.traction.p.spin_accel:.1f} / {self.traction.p.spin_rate:.1f}, "
                                  f"release authority {self.traction.p.release_max:.1f} m/s)"))
        if self.traction is not None and VescStateStamped is None:
            self.get_logger().warning(
                "vesc_msgs is not importable, so /sensors/core motor current is unavailable: the "
                "traction guard will not require drive torque before calling a launch spin.")

    def on_odom(self, m: Odometry):
        v = m.twist.twist.linear.x
        if not math.isfinite(v):            # a NaN speed must not reach the actor, nor count as fresh
            return
        self.v = v
        self.t_odom = now = self.clock()
        if self.traction is not None:
            # Stale inputs are passed as None rather than as their last value: the guard holds its
            # filters over a missing sample, and `update` re-seeds rather than detecting across a
            # gap longer than `max_step_dt`.
            fresh = lambda t: t is not None and now - t <= self.timeout
            st = self.traction.update(now, v, self.ax_body if fresh(self.t_ax) else None,
                                      self.motor_current if fresh(self.t_current) else None)
            if st.changed:
                self.get_logger().info(
                    f"traction {st.state}: wheel {st.wheel_speed:+.2f} m/s at "
                    f"{st.wheel_accel:+.1f} m/s^2, body {st.body_speed:+.2f} m/s at "
                    f"{st.body_accel:+.1f} m/s^2 (residual {st.residual:+.1f}, slip {st.slip:+.2f}), "
                    f"{st.locks} locks / {st.spins} spins so far")

    def on_core(self, m):
        """`/sensors/core` motor current, the drive-torque corroboration for a spin. Optional: the
        guard treats None as "no information" rather than as "no torque", and one recording has no
        such topic at all."""
        i = getattr(getattr(m, "state", None), "current_motor", None)
        if i is None or not math.isfinite(float(i)):
            return
        self.motor_current = float(i)
        self.t_current = self.clock()

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
        self.ax_body = row[3]; self.t_ax = now      # SI body x acceleration, for the traction guard
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
        if self.att_source == "vesc" and (self.t_att is None or now - self.t_att > self.timeout):
            # With `attitude_source:=ego` or `:=frontend` there is no quaternion in the loop at
            # all, so its absence is not a reason to stop driving. The signals those paths need --
            # the wheel speed, the IMU and the scan -- are already checked above and by the
            # watchdog.
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
        # Same argument as the observation history, one step further in: a recurrent policy's
        # hidden state and the decayed scan-occupancy channel both describe a segment that is over.
        self.policy_state.reset()
        if self.traction is not None:
            # Same argument as the observation history: the wheel-speed derivative, the body-speed
            # estimate and any latched release describe a segment that is over.
            self.traction.reset()
        if self.tracker is not None:
            self.tracker.reset(torch.zeros(1, dtype=torch.long, device=self.device))
        self.get_logger().info("sensors recovered: observation, policy memory and tracker history cleared")

    def on_reset(self, _msg: EmptyMsg):
        """The car was reset. Everything that describes the run it was in is now wrong."""
        self.obs.reset()
        self.policy_state.reset()
        if self.traction is not None:
            self.traction.reset()
        if self.clearance is not None:
            # Belt and braces: `on_scan` writes this buffer before the tracker is ever asked for a
            # command, so a stale frame cannot reach a plan. Clearing it anyway means that if that
            # ever stops being true, a car that has been picked up and put down is shaped by
            # nothing rather than by the room it used to be in.
            self.clearance.scan.fill_(1.0)
        if self.tracker is not None:
            self.tracker.reset(torch.zeros(1, dtype=torch.long, device=self.device))
        self.get_logger().info("reset: observation, policy memory and tracker history cleared")

    def _check_scan_geometry(self, m: LaserScan):
        """Re-declare the clearance grid's bearings from the driver's own window, once.

        The nominal 270 deg is what this car's `urg_node` publishes, but the grid is built by
        turning each return into a point at its bearing: a window that is actually 240 deg, or one
        published backwards, would place every obstacle somewhere it is not, and nothing downstream
        would look wrong. `ObsBuilder` resamples the ranges linearly over the message's own index
        range, so the resampled beam bearings are `linspace(angle_min, angle_max, n_beams)`.
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
        att = self.att
        if self.ego_att is not None:
            # The ego-state estimator runs every scan whatever the source is, because the clearance
            # gate and any front-end read it; only whether the POLICY sees it depends on the
            # parameter.
            rp = self.ego_att.update(
                torch.tensor([float(self.v)], device=self.device),
                torch.tensor([float(imu_mean[2])], device=self.device),
                torch.as_tensor(imu_mean[3:], dtype=torch.float32, device=self.device)[None])
            if self.att_source == "ego":
                att = (float(rp[0, 0]), float(rp[0, 1]))
            elif self.att_source == "frontend":
                # The front-end reads the scan stack, and the stack is what `obs.build` below
                # produces -- so its estimate for THIS scan does not exist yet. What is available is
                # the one it made for the previous scan, 25 ms ago, and that is what is used: one
                # control step of lag on a quantity whose own process has a 0.4 s time constant.
                # Until the first scan has been through the network there is none, and the
                # ego-state estimate stands in.
                fe = getattr(getattr(self.policy_state, "scan", None), "fe", None)
                last = None if fe is None else fe.last_att
                att = ((float(last[0, 0]), float(last[0, 1])) if last is not None
                       else (float(rp[0, 0]), float(rp[0, 1])))
        scan, pro = self.obs.build(r, self.v, imu_mean, att, self.speed_cap)
        if self.clearance is not None:
            self._check_scan_geometry(m)
            # This scan, in the observation's own units, before the tracker is asked for anything.
            # The arm holds one frame and nothing else: no map, no pose, no memory across scans.
            self.clearance.update_scan(scan)
            if self.clearance.cspec.floor_gate:
                # The gate's own attitude, integrated from the IMU mean this node already computed
                # for the observation -- the same three signals, no new topic and no new message
                # type. Gyro first three, accelerometer last three, both SI by here.
                g = torch.as_tensor(imu_mean[:3], dtype=torch.float32,
                                    device=self.device)[None]
                acc = torch.as_tensor(imu_mean[3:], dtype=torch.float32,
                                      device=self.device)[None]
                self.clearance.update_attitude(
                    g, acc, torch.tensor([float(self.v)], device=self.device))
        with torch.no_grad():
            # The hidden state goes in and the next one comes out: the policy's memory of this run
            # lives here, between callbacks, and nowhere else.
            a, _, self.policy_state.hidden = self.model.act(
                self.policy_state.observe(scan, pro), pro, deterministic=True,
                h=self.policy_state.hidden)
        if self.clearance is not None and self.clearance.cspec.floor_gate:
            # The gate reads the front-end's floor class when the checkpoint carries one, and the
            # geometry otherwise. This sits between the actor and the tracker on purpose: the
            # front-end ran inside `policy_state.observe` just above, and the plan hook that reads
            # this runs inside `self.tracker` just below, so the gate uses THIS scan's answer.
            fe = getattr(getattr(self.policy_state, "scan", None), "fe", None)
            if fe is not None and fe.last_probs is not None:
                from f1sim.learn.frontend import FLOOR
                self.clearance.set_floor(fe.last_probs[:, FLOOR])
        self.obs.push_action(a[0])
        msg = AckermannDriveStamped(); msg.header.stamp = m.header.stamp
        if self.tracker is not None:                                 # local plan -> tracker -> command
            cmd = self.tracker(a, torch.tensor([self.v], device=self.device), torch.tensor([self.speed_cap], device=self.device),
                               torch.tensor([float(imu_mean[2])], device=self.device), delay=self.delay)[0]
            msg.drive.steering_angle = float(max(-self.steer_max, min(self.steer_max, (float(cmd[0]) - self.cal[0]) / self.cal[1])))
            speed = float(cmd[1])
        else:
            a = a[0].cpu().numpy()
            msg.drive.steering_angle = float(a[0] * self.steer_max)
            speed = float(min((a[1] + 1) * 0.5 * self.spec.v_max, self.speed_cap))
        if self.traction is not None:
            # Last thing before the command leaves, and *before* `speed_gain`: the guard reasons in
            # the car's own m/s -- it compares the command against a body speed estimated from this
            # car's sensors -- while the published field is that speed divided by the calibration
            # gain, so shaping the published number would mix the two scales whenever the gain is
            # not 1. A released brake or a capped launch is about the wheel, not about the plan, so
            # nothing upstream needs to know it happened.
            speed = min(self.traction.shape(speed), self.speed_cap)
        msg.drive.speed = float(speed) / self.cal[2] if self.tracker is not None else float(speed)
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
