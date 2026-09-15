"""The policy half of the graph: /scan + /odom + /sensors/imu(/raw) -> /f1sim/plan at the scan rate.

This node is the network and nothing else. It builds the observation with
`f1sim.learn.obs.ObsBuilder` -- the same encoding the policy was trained with -- runs the actor,
and publishes the raw normalized plan. The iLQR tracker, the grip and clearance arms and the
traction guard used to live here too; they are `controller_node` now, and the plan on the wire
between them is arm-agnostic, so the same plan stream can be replayed through any arm.

Two things still end here rather than on the wire:

* a **direct-action** checkpoint (`act_dim == 2`, the legacy `--action-mode direct`) has no plan to
  publish. It emits `/drive` from this node and bypasses the controller entirely -- see
  `docs/ros2.md`, "Direct-action checkpoints".
* the **episode memory** (a recurrent hidden state, a decayed scan-occupancy channel) is the
  policy's own state and is cleared here, in the same two places it always was.

Works against the simulator (`graph_sim.launch.py`), the console link (`graph_console.launch.py`)
and the real car (`graph_car.launch.py`) unchanged.
"""
import hashlib
import math
import os
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
from f1sim.learn.obs import ObsBuilder, ObsSpec
from f1sim_interfaces.msg import Perception, Plan, PolicyState

from f1sim_ros.deploy import (CONTROL_RATE, G, IMU_BUF_MAX, LIDAR_FOV, LIDAR_MOUNT_X, SensorIntake,
                              attitude_from_orientation, covariance0, header_seconds,
                              quat_to_rp, resample_ranges)

try:
    from vesc_msgs.msg import VescImuStamped
except ImportError:
    VescImuStamped = None

#: The arms are the controller's now. Re-exported so a caller that used to reach them through this
#: module keeps working and gets pointed at where they live.
from f1sim_ros.deploy import (DEPLOYABLE_ARMS, build_traction_guard,  # noqa: E402,F401
                              install_clearance_arm, install_grip_arm, split_deployable)


def checkpoint_id(path: str) -> str:
    """`<run>@<sha256[:12]>` for the file the actor came from, or `""` for no checkpoint.

    On every `Plan` rather than latched once: a controller that started after the policy, or that
    reconnected to a second one, still knows what it is tracking, and a bag says which weights
    drove the car. Hashing 27 MB costs ~30 ms, once, at startup.
    """
    if not path or not os.path.exists(path):
        return str(path or "")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    run = os.path.basename(os.path.dirname(os.path.abspath(path))) or "?"
    return f"{run}/{os.path.basename(path)}@{h.hexdigest()[:12]}"


class PolicyNode(Node):
    #: Class defaults, so a node the tests build with `__new__` -- several fixtures hand-assemble a
    #: minimal node to exercise one callback -- starts from the shipped behaviour instead of an
    #: AttributeError. `__init__` overwrites both from the parameters.
    att_source = "vesc"
    ego_att = None
    pub_perception = None

    def __init__(self):
        super().__init__("f1sim_policy")
        self.declare_parameter("checkpoint", ""); self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("speed_cap", 4.0)
        self.declare_parameter("plan_topic", "/f1sim/plan")
        self.declare_parameter("state_topic", "/f1sim/policy_state")
        self.declare_parameter("enabled", True)
        # Only a direct-action checkpoint publishes these: it has no plan, so it drives /drive
        # itself and the controller is not in its graph at all.
        self.declare_parameter("drive_topic", "drive"); self.declare_parameter("steer_max", 0.4189)
        # This car's /sensors/imu/raw publishes linear_acceleration in **g**, not the m/s^2 the Imu
        # message specifies (az ~ 1.00 at rest in all 22 recordings). The policy was trained on SI,
        # so feeding g straight through scales the accelerometer channel by 1/9.81 and the mistake
        # is silent -- the car just drives with a dead input. 0.0 = detect from the gravity vector,
        # which is unambiguous because the two candidates are an order of magnitude apart.
        self.declare_parameter("imu_accel_scale", 0.0)
        # Freshness. Every input the actor reads carries the time it was last valid, and inference
        # is inhibited when any of them goes stale. Holding the last value through a dropped sample
        # is fine; driving on a value from a second ago is not.
        self.declare_parameter("sensor_timeout", 0.25)
        # Where the policy's roll/pitch observation columns come from. `vesc` is the orientation
        # quaternion this node has always read; `ego` is `f1sim.learn.floor.EgoStateAttitude`, the
        # suspension's calibrated response to the accelerations the car itself produces, computed
        # from the wheel speed, the gyro's yaw and the accelerometer; `frontend` is the estimate a
        # checkpoint's own sensor front-end makes. Two reasons to have the choice, both measured
        # and both in `docs/research/floor-mask-2026-09-15.md`:
        #
        #   * the quaternion is wrong by 0.14 rad rms in roll and 0.09 in pitch while driving,
        #     against 0.022 / 0.022 for the ego-state path -- and those columns are a POLICY INPUT,
        #     not only a gate's;
        #   * five of the thirteen competition recordings carry a quaternion that swings the
        #     extracted roll past 40 deg, and on those this node refuses to drive at all today.
        #     The ego-state path does not use the quaternion, so it survives them.
        #
        # `vesc` stays the default: every trained checkpoint saw `imu_att` in training, and
        # swapping the meaning of two observation columns under a policy is a change to make
        # deliberately with a finetune behind it, not a default.
        self.declare_parameter("attitude_source", "vesc")
        # What this node tells the controller about the scan it just planned from -- the front-end's
        # per-beam floor class and the attitude the observation was built with. Diagnostic unless
        # the controller's clearance arm has its floor gate on, and OPTIONAL either way: a
        # controller that never receives it falls back to its own geometry.
        self.declare_parameter("perception_topic", "/f1sim/policy/perception")
        p = lambda n: self.get_parameter(n).value
        self.device = torch.device(p("device"))
        ckpt = str(p("checkpoint"))
        self.model, extra = load_checkpoint(ckpt, self.device); self.model.eval()
        self.spec = ObsSpec(**extra["spec"]) if extra.get("spec") else ObsSpec()
        if self.spec.opp_token:
            # The privileged opponent block (`gym_env.OPP_TOKEN_MODES`) is the simulator's ground
            # truth about the other cars -- where they are, how fast, and where they will be. No
            # sensor on this car produces it. A checkpoint trained on it is an oracle arm and an
            # experimental result; driving a real car with zeros in those columns would run a
            # policy on an input that permanently says "no car is anywhere near".
            raise ValueError(
                f"checkpoint declares the privileged opponent block opp_token="
                f"{self.spec.opp_token!r} (future model "
                f"{self.spec.opp_future_model!r}). It is a simulator oracle and is not deployable.")
        self.obs = ObsBuilder(self.spec, self.device)
        self.checkpoint_id = checkpoint_id(ckpt)
        #: The policy's episode state: the recurrent hidden state and the decayed scan-occupancy
        #: channel, for this one car. Empty (and `stateful` False) for a feedforward checkpoint, so
        #: nothing below changes for one. It is cleared in exactly two places, both of which mean
        #: "the run you remember is over": `_resume`, when the scan stream comes back after a gap,
        #: and `on_reset`, when something resets the car. Carrying it across either would drive a
        #: car that has been picked up and put down with the memory of where it used to be.
        self.policy_state = runtime_for(self.model, batch=1, device=self.device)
        self.speed_cap = float(p("speed_cap")); self.steer_max = float(p("steer_max"))
        self.timeout = float(p("sensor_timeout"))
        self.sensors = SensorIntake(self.clock, self.get_logger(), self.timeout,
                                    accel_scale=float(p("imu_accel_scale")))
        self._inhibited = False; self._last_inhibit_log = 0.0
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
        self.ego_att = _floor.EgoStateAttitude(1, device=self.device,
                                               dt=1.0 / float(CONTROL_RATE), source="wheel")
        if self.att_source == "ego":
            self.get_logger().warning(
                "attitude_source:=ego -- the policy's roll/pitch observation columns now come from "
                "f1sim.learn.floor.EgoStateAttitude, not from the orientation quaternion. The "
                "checkpoint was trained against the quaternion unless it says otherwise; see "
                "docs/research/floor-mask-2026-09-15.md for what each is worth.")
        #: Plan sequence number, and what the state message reports about the memory.
        self.seq = 0
        self.memory_clears = 0; self.memory_cleared_at = -1.0; self.memory_cleared_reason = ""
        #: A direct-action checkpoint keeps the old /drive path; a plan checkpoint has none.
        self.direct = int(self.spec.act_dim) == 2
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
        self.pub_plan = self.create_publisher(Plan, str(p("plan_topic")), 1)
        self.pub_state = self.create_publisher(PolicyState, str(p("state_topic")), 1)
        self.pub_perception = self.create_publisher(Perception, str(p("perception_topic")), 1)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, p("drive_topic"), 1) if self.direct else None
        self.last_t = None
        # warm up
        s, pr = self.obs.build(np.full(self.spec.n_beams, 5.0), 0.0, np.zeros(6), np.zeros(2), self.speed_cap)
        with torch.no_grad():
            self.model.act(self.policy_state.observe(s, pr), pr, deterministic=True,
                           h=self.policy_state.hidden)
        self.obs.reset()
        self.policy_state.reset()                # the warm-up is not part of any episode
        if self.policy_state.stateful:
            self.get_logger().info(f"policy memory: {describe_memory(self.model.meta)}; hidden state "
                                   f"carried across scan callbacks, cleared on "
                                   f"{p('reset_topic')} and when the scan stream restarts")
        self.get_logger().info(
            f"policy {self.checkpoint_id} on {self.device}, speed cap {self.speed_cap} m/s, "
            + (f"direct action: publishing {p('drive_topic')} and bypassing the controller"
               if self.direct else
               f"publishing {p('plan_topic')} ({self.spec.act_dim}-D plan) at the scan rate; "
               f"the tracker, the arms and the traction guard are controller_node's"))

    # ---------------------------------------------------------------- clocks
    def clock(self):
        """Monotonic seconds. A method so tests can drive it without a ROS clock."""
        return time.monotonic()

    def stamp_now(self):
        """Current ROS time as a message stamp. Overridden in tests."""
        return self.get_clock().now().to_msg()

    # ---------------------------------------------------------------- callbacks
    def on_odom(self, m: Odometry):
        self.sensors.on_odom(m)

    def on_imu(self, m: Imu):
        self.sensors.on_imu(m)

    def on_vesc_imu(self, m):
        self.sensors.on_vesc_imu(m)

    def on_watchdog(self):
        """Stop planning when the scans themselves stop.

        Every other guard lives in `on_scan`, which is exactly the callback that stops running when
        the LiDAR goes away -- leaving the controller tracking the last plan with nothing to
        countermand it. This runs on a timer instead. It never calls the actor; it only inhibits,
        which the controller sees as a plan that stopped arriving.
        """
        now = self.clock()
        stale = self.sensors.stale(now, check_scan=True)
        if stale:
            self._inhibit(stale, self.stamp_now())

    def _inhibit(self, stale, stamp):
        """Stop planning on data we do not have.

        The policy simply stops publishing. What that means for the car is the controller's
        business: it inhibits on the same stale inputs at the same threshold (it reads the same
        topics), and independently brakes to zero when the plan stream itself dries up. Nothing
        here publishes a "safe plan" -- a plan is a trajectory, and there is no trajectory that
        means "stop" without a controller to interpret it.
        """
        self._inhibited = True
        self._publish_state(stale, stamp)
        if self.direct:
            msg = AckermannDriveStamped(); msg.header.stamp = stamp
            msg.drive.speed = 0.0; msg.drive.steering_angle = 0.0
            if self.get_parameter("enabled").value:
                self.pub_drive.publish(msg)
        now = time.monotonic()
        if now - self._last_inhibit_log > 1.0:
            self._last_inhibit_log = now
            self.get_logger().warning(
                f"no actor command: {', '.join(stale)} older than {self.timeout:.2f} s. "
                + ("Publishing zero speed until the sensors come back."
                   if self.direct else
                   "Publishing no plan until the sensors come back; the controller brakes."))

    def _clear_memory(self, reason: str):
        """The observation history and the policy's episode memory, cleared and accounted for."""
        self.obs.reset()
        self.policy_state.reset()
        self.memory_clears += 1
        self.memory_cleared_at = self.clock()
        self.memory_cleared_reason = reason

    def _resume(self):
        """A gap ended. The observation history describes a segment that is over, and stitching the
        new one onto it feeds the policy an action history that never happened. Same argument one
        step further in for a recurrent policy's hidden state and the decayed scan channel."""
        self._inhibited = False
        self._clear_memory("resume")
        self.get_logger().info("sensors recovered: observation and policy memory cleared")

    def on_reset(self, _msg: EmptyMsg):
        """The car was reset. Everything that describes the run it was in is now wrong."""
        self._clear_memory("reset")
        self.get_logger().info("reset: observation and policy memory cleared")

    def on_scan(self, m: LaserScan):
        t0 = time.perf_counter()
        r = resample_ranges(m.ranges, m.range_max, self.spec.n_beams)
        now = self.clock()
        self.sensors.t_scan = now
        imu_mean = self.sensors.take_imu_mean(now)
        stale = self.sensors.stale(now)
        if stale:
            self._inhibit(stale, self.stamp_now())
            return
        if self._inhibited:                     # coming back from a gap
            self._resume()
        att = self.sensors.att
        # The ego-state estimator runs every scan whatever the source is, because the perception
        # message carries it for the controller's gate; only whether the POLICY sees it depends on
        # the parameter. `None` is a hand-assembled node (the class default above) -- then the
        # quaternion is the only source there is, which is what this node always did.
        rp = None
        if self.ego_att is not None:
            rp = self.ego_att.update(
                torch.tensor([float(self.sensors.v)], device=self.device),
                torch.tensor([float(imu_mean[2])], device=self.device),
                torch.as_tensor(imu_mean[3:], dtype=torch.float32, device=self.device)[None])
        if self.att_source == "ego" and rp is not None:
            att = (float(rp[0, 0]), float(rp[0, 1]))
        elif self.att_source == "frontend":
            # The front-end reads the scan stack, and the stack is what `obs.build` below produces
            # -- so its estimate for THIS scan does not exist yet. What is available is the one it
            # made for the previous scan, 25 ms ago, and that is what is used: one control step of
            # lag on a quantity whose own process has a 0.4 s time constant. Until the first scan
            # has been through the network there is none, and the ego-state estimate stands in.
            fe = getattr(getattr(self.policy_state, "scan", None), "fe", None)
            last = None if fe is None else fe.last_att
            if last is not None:
                att = (float(last[0, 0]), float(last[0, 1]))
            elif rp is not None:
                att = (float(rp[0, 0]), float(rp[0, 1]))
        scan, pro = self.obs.build(r, self.sensors.v, imu_mean, att, self.speed_cap)
        with torch.no_grad():
            # The hidden state goes in and the next one comes out: the policy's memory of this run
            # lives here, between callbacks, and nowhere else.
            # The `aligned` channel warps this scan against the one from k steps ago with the
            # speed, yaw rate and roll/pitch that are already in `pro` -- the same three sensors
            # this node already reads, and the reason the channel is deployable at all.
            a, _, self.policy_state.hidden = self.model.act(
                self.policy_state.observe(scan, pro), pro, deterministic=True,
                h=self.policy_state.hidden)
        self.obs.push_action(a[0])
        enabled = bool(self.get_parameter("enabled").value)
        if self.direct:
            act = a[0].cpu().numpy()
            msg = AckermannDriveStamped(); msg.header.stamp = m.header.stamp
            msg.drive.steering_angle = float(act[0] * self.steer_max)
            msg.drive.speed = float(min((act[1] + 1) * 0.5 * self.spec.v_max, self.speed_cap))
            if enabled:
                self.pub_drive.publish(msg)
            shown = f"cmd=({msg.drive.steering_angle:+.2f} rad, {msg.drive.speed:.2f} m/s)"
        else:
            plan = Plan()
            # The stamp of the SCAN this plan was computed from, not the time it was published: the
            # controller pairs a plan with the sensor snapshot of that scan, and measures the plan's
            # age against the sensor it came from. A publish-time stamp would hide the inference
            # latency inside it and pair the plan with the wrong scan.
            plan.header.stamp = m.header.stamp
            plan.header.frame_id = "base_link"
            plan.plan = [float(x) for x in a[0].cpu().numpy()]
            plan.checkpoint = self.checkpoint_id
            plan.seq = self.seq
            if enabled:
                self.pub_plan.publish(plan)
            shown = "plan=[" + " ".join(f"{x:+.2f}" for x in plan.plan) + "]"
        self._publish_perception(m.header.stamp, att)
        self._publish_state([], m.header.stamp)
        self.seq = (self.seq + 1) % (1 << 32)
        dt = (time.perf_counter() - t0) * 1000
        if self.last_t is None or time.perf_counter() - self.last_t > 5:
            self.get_logger().info(f"inference {dt:.1f} ms  v={self.sensors.v:.2f} {shown}")
            self.last_t = time.perf_counter()

    def _publish_perception(self, stamp, att):
        """`/f1sim/policy/perception`: what this node worked out about the scan it just planned from.

        The front-end's per-beam floor class, when the checkpoint carries one, and the attitude the
        observation was actually built with. The monolithic node handed the first of these straight
        to the clearance arm in the same process; the arm is `controller_node`'s now, so it goes on
        a topic stamped with the same scan the plan is stamped with and the controller pairs the
        two the same way. A controller that never receives it uses its own geometry, which is what
        `clearance_floor_gate` did before a front-end existed.
        """
        msg = Perception()
        msg.header.stamp = stamp
        msg.header.frame_id = "laser"
        msg.source = self.att_source
        msg.attitude = [float(att[0]), float(att[1])]
        msg.attitude_ok = True
        fe = getattr(getattr(self.policy_state, "scan", None), "fe", None)
        probs = None if fe is None else fe.last_probs
        if probs is not None:
            from f1sim.learn.frontend import FLOOR
            msg.floor = [float(x) for x in probs[0, FLOOR].detach().cpu().numpy()]
        if self.pub_perception is not None and bool(self.get_parameter("enabled").value):
            self.pub_perception.publish(msg)

    def _publish_state(self, stale, stamp):
        """`/f1sim/policy_state`: what this node knows about itself. Diagnostic only."""
        ages = self.sensors.ages(self.clock())
        s = PolicyState()
        s.header.stamp = stamp
        s.header.frame_id = "base_link"
        s.memory_present = bool(self.policy_state.stateful)
        s.memory_kind = describe_memory(self.model.meta) if self.policy_state.stateful else ""
        s.memory_clears = int(self.memory_clears)
        s.memory_cleared_at = float(self.memory_cleared_at)
        s.memory_cleared_reason = str(self.memory_cleared_reason)
        s.inhibited = bool(stale)
        s.stale = list(stale)
        s.scan_age_s = ages["scan"]; s.imu_age_s = ages["imu"]
        s.odom_age_s = ages["odom"]; s.attitude_age_s = ages["attitude"]
        s.seq = int(self.seq)
        s.checkpoint = self.checkpoint_id
        self.pub_state.publish(s)


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
