"""Hardware emulation for the f1tenth_system stack: this node stands in for vesc_driver_node and
urg_node. Everything else of f1tenth_stack (ackermann_mux, ackermann_to_vesc, vesc_to_odom,
joy_teleop, ...) runs unchanged on top of it, with the stack's own config files.

Subscribes (exactly what vesc_driver consumes):
    commands/motor/speed        std_msgs/Float64   ERPM  (speed_to_erpm_gain/offset params, like vesc.yaml)
    commands/motor/brake        std_msgs/Float64   brake current -> speed 0
    commands/motor/duty_cycle   std_msgs/Float64   mapped to speed (duty * v_max)
    commands/servo/position     std_msgs/Float64   0..1 (steering_angle_to_servo_gain/offset params)
Publishes (what vesc_driver + urg_node publish):
    sensors/core                    vesc_msgs/VescStateStamped   speed [ERPM], tachometer, voltage, current
    sensors/servo_position_command  std_msgs/Float64              echo of the servo command
    sensors/imu                     vesc_msgs/VescImuStamped      ypr [deg] + quaternion + rates/accels
    sensors/imu/raw                 sensor_msgs/Imu
    scan                            sensor_msgs/LaserScan          frame_id laser
Sim-only extras: /map (latched), /ego_racecar/odom (ground truth, map frame), /f1sim/collision,
    tf map->odom so that map->base_link is ground truth (publish_gt_tf, off when running a localizer),
    /initialpose teleport, /f1sim/reset service. odom->base_link tf and /odom come from vesc_to_odom.
"""
import math
import time

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float64
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster

from ackermann_msgs.msg import AckermannDriveStamped

from f1sim import Config, Simulator, Track, maps
from f1sim.teleop import DriveModel, KeyRamp, TeleopConfig

try:
    from vesc_msgs.msg import VescImuStamped, VescStateStamped
    HAVE_VESC_MSGS = True
except ImportError:  # pragma: no cover
    HAVE_VESC_MSGS = False


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion(); q.z = math.sin(yaw / 2); q.w = math.cos(yaw / 2); return q


def rpy_to_quat(r, p, y) -> Quaternion:
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy; q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy; q.z = cr * cp * sy - sr * sp * cy
    return q


def quat_to_yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class VescSimNode(Node):
    def __init__(self):
        super().__init__("vesc_sim")
        # --- the same parameters vesc_driver / vesc_ackermann read from vesc.yaml
        self.declare_parameter("speed_to_erpm_gain", 4614.0)
        self.declare_parameter("speed_to_erpm_offset", 0.0)
        self.declare_parameter("steering_angle_to_servo_gain", -1.2135)
        self.declare_parameter("steering_angle_to_servo_offset", 0.5304)
        self.declare_parameter("servo_min", 0.15)
        self.declare_parameter("servo_max", 0.85)
        self.declare_parameter("speed_min", -23250.0)
        self.declare_parameter("speed_max", 23250.0)
        # --- sim
        self.declare_parameter("map", "gen:competition:0")     # catalog name or map yaml
        self.declare_parameter("config_yaml", "")
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("randomize", True)
        self.declare_parameter("laser_frame", "laser")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("publish_gt_tf", True)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("auto_reset_on_collision", False)
        self.declare_parameter("start_s", -1.0)
        self.declare_parameter("viewer", False)
        self.declare_parameter("teleop", True)        # drive with the keyboard in the viewer window (publishes /teleop)
        self.declare_parameter("teleop_v_max", 5.0)
        self.declare_parameter("teleop_v_boost", 8.0)
        self.declare_parameter("teleop_a_throttle", 2.0)     # gentle by default; raise for a punchier car
        self.declare_parameter("wheelbase_note", "vesc_to_odom's wheelbase param should be 0.3302 for this car")

        p = lambda n: self.get_parameter(n).value
        self.erpm_gain, self.erpm_off = float(p("speed_to_erpm_gain")), float(p("speed_to_erpm_offset"))
        self.servo_gain, self.servo_off = float(p("steering_angle_to_servo_gain")), float(p("steering_angle_to_servo_offset"))
        cfg = Config.from_yaml(p("config_yaml")) if p("config_yaml") else Config()
        cfg.rand.enabled = bool(p("randomize")); cfg.sim.device = p("device")
        cfg.sim.compile = (cfg.sim.device == "cuda"); cfg.sim.terminate_on_collision = False
        self.cfg = cfg
        self.track = maps.load(p("map"))
        self.sim = Simulator(self.track, cfg, num_envs=1, device=cfg.sim.device)
        self.get_logger().info("compiling simulator kernels ...")
        self.sim.warmup()
        self.rate = cfg.sim.control_rate
        self.base_frame, self.laser_frame = p("base_frame"), p("laser_frame")
        self.odom_frame, self.map_frame = p("odom_frame"), p("map_frame")
        self.publish_gt_tf = bool(p("publish_gt_tf")); self.cmd_timeout = float(p("cmd_timeout"))
        self.auto_reset = bool(p("auto_reset_on_collision")); self.start_s = float(p("start_s"))
        self.v_max = cfg.vehicle.v_max

        # --- command state (what the VESC would hold)
        self.erpm_cmd = 0.0; self.servo_cmd = self.servo_off; self.last_cmd_time = None
        self.tacho = 0.0
        self.create_subscription(Float64, "commands/motor/speed", self.on_speed, 1)
        self.create_subscription(Float64, "commands/motor/brake", self.on_brake, 1)
        self.create_subscription(Float64, "commands/motor/duty_cycle", self.on_duty, 1)
        self.create_subscription(Float64, "commands/servo/position", self.on_servo, 1)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initialpose, 1)
        self.odom_pose = None                                             # latest odom->base_link from vesc_to_odom
        self.create_subscription(Odometry, "odom", self.on_odom, 1)
        # --- publishers
        self.pub_scan = self.create_publisher(LaserScan, "scan", 1)
        self.pub_core = self.create_publisher(VescStateStamped, "sensors/core", 10) if HAVE_VESC_MSGS else None
        self.pub_servo = self.create_publisher(Float64, "sensors/servo_position_command", 10)
        self.pub_imu = self.create_publisher(VescImuStamped, "sensors/imu", 10) if HAVE_VESC_MSGS else None
        self.pub_imu_raw = self.create_publisher(Imu, "sensors/imu/raw", 10)
        self.pub_gt = self.create_publisher(Odometry, "/ego_racecar/odom", 1)
        self.pub_coll = self.create_publisher(Bool, "/f1sim/collision", 1)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_map = self.create_publisher(OccupancyGrid, "/map", latched)
        self.tf = TransformBroadcaster(self)
        self.create_service(Empty, "/f1sim/reset", self.on_reset_srv)
        if not HAVE_VESC_MSGS:
            self.get_logger().warn("vesc_msgs not found: sensors/core and sensors/imu are not published (build the workspace)")

        self.viewer = None; self.teleop = None
        if bool(p("viewer")):
            from f1sim.viewer.native import NativeViewer
            self.viewer = NativeViewer(self.sim, max_fps=0)
            if bool(p("teleop")):
                tcfg = TeleopConfig(v_max=float(p("teleop_v_max")), v_boost=float(p("teleop_v_boost")),
                                    a_throttle=float(p("teleop_a_throttle")))
                self.teleop = (DriveModel(tcfg), KeyRamp(tcfg))
                self.pub_teleop = self.create_publisher(AckermannDriveStamped, "teleop", 1)
                self.teleop_active = False; self.v_meas_odom = 0.0
                self.get_logger().info("window teleop: W/S throttle-brake, A/D or arrows steer, Shift boost, Space handbrake, R reset")
        self.publish_map()
        self.do_reset()
        self.timer = self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(f"vesc_sim: map={self.track.name} {self.track.shape} device={self.sim.device} rate={self.rate} Hz "
                               f"erpm_gain={self.erpm_gain} servo gain/offset={self.servo_gain}/{self.servo_off}")

    # ------------------------------------------------------------------ commands (vesc_driver semantics)
    def on_speed(self, msg: Float64):
        lo, hi = float(self.get_parameter("speed_min").value), float(self.get_parameter("speed_max").value)
        self.erpm_cmd = min(max(msg.data, lo), hi); self.last_cmd_time = self.get_clock().now()

    def on_brake(self, msg: Float64):
        self.erpm_cmd = 0.0; self.last_cmd_time = self.get_clock().now()

    def on_duty(self, msg: Float64):
        self.erpm_cmd = float(msg.data) * self.v_max * self.erpm_gain; self.last_cmd_time = self.get_clock().now()

    def on_servo(self, msg: Float64):
        lo, hi = float(self.get_parameter("servo_min").value), float(self.get_parameter("servo_max").value)
        self.servo_cmd = min(max(msg.data, lo), hi)
        self.pub_servo.publish(Float64(data=self.servo_cmd))          # vesc_driver echoes the servo command

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose
        self.odom_pose = (p.position.x, p.position.y, quat_to_yaw(p.orientation))
        self.v_meas_odom = msg.twist.twist.linear.x

    def window_teleop(self):
        """Poll the viewer keys; publish /teleop only while the driver touches a key (so /drive keeps
        working otherwise, exactly like the mux on the real car)."""
        model, ramp = self.teleop
        k = self.viewer.keys_held()
        any_key = any(k.values())
        if k.get("reset"):
            self.do_reset(); model.reset()
        if any_key:
            self.teleop_active = True
        inp = ramp.update(k.get("left", False), k.get("right", False), k.get("up", False), k.get("down", False),
                          self.sim.control_dt, boost=k.get("boost", False), handbrake=k.get("handbrake", False))
        if not self.teleop_active:
            return
        steer, v = model.update(inp, self.v_meas_odom, self.sim.control_dt)
        m = AckermannDriveStamped(); m.header.stamp = self.get_clock().now().to_msg()
        m.drive.steering_angle = float(steer); m.drive.speed = float(v)
        self.pub_teleop.publish(m)
        self.viewer.extra_hud = model.hud(inp, self.v_meas_odom)
        if not any_key and abs(v) < 0.05 and abs(self.v_meas_odom) < 0.1:
            self.teleop_active = False                    # stopped and hands off: release the mux
            self.viewer.extra_hud = ["MANUAL idle (press a key to drive, /drive active)"]

    def on_initialpose(self, msg):
        pose = msg.pose.pose
        self.do_reset(torch.tensor([[pose.position.x, pose.position.y, quat_to_yaw(pose.orientation)]]))

    def on_reset_srv(self, req, resp):
        self.do_reset(); return resp

    def do_reset(self, pose=None):
        if pose is None and self.track.centerline is not None:
            s = None if self.start_s < 0 else torch.tensor([self.start_s])
            pose = self.sim.sample_spawn(1, lateral_std=0.0, yaw_std=0.0, s=s)
        self.sim.reset(poses=pose)
        self.erpm_cmd = 0.0; self.servo_cmd = self.servo_off; self.tacho = 0.0
        self.get_logger().info("reset at %s" % self.sim.state[0, :3].cpu().numpy().round(2))

    # ------------------------------------------------------------------ main loop
    def tick(self):
        now = self.get_clock().now()
        erpm = self.erpm_cmd
        if self.last_cmd_time is None or (now - self.last_cmd_time).nanoseconds * 1e-9 > self.cmd_timeout:
            erpm = 0.0                                                  # VESC timeout: motor off
        speed = (erpm - self.erpm_off) / self.erpm_gain
        steer = (self.servo_cmd - self.servo_off) / self.servo_gain
        r = self.sim.step(torch.tensor([[steer, speed]], dtype=torch.float32))
        stamp = now.to_msg()
        st = r.state[0].cpu().numpy(); od = r.odom[0].cpu().numpy()
        # --- VESC telemetry: measured speed in ERPM (with the odom model's speed noise / calibration error)
        # vesc_to_odom (f1tenth_system) reads speed as (-state.speed - offset) / gain: the real VESC
        # reports the motor turning "backwards" for forward driving with the stack's wiring/gain sign.
        v_meas = float(od[3]); erpm_meas = -(v_meas * self.erpm_gain + self.erpm_off)
        self.tacho += v_meas * self.sim.control_dt * self.erpm_gain / 60.0 * 6.0     # ~ tachometer counts
        if self.pub_core is not None:
            m = VescStateStamped(); m.header.stamp = stamp
            m.state.speed = erpm_meas; m.state.voltage_input = 12.4; m.state.duty_cycle = float(np.clip(v_meas / self.v_max, -1, 1))
            m.state.current_motor = float(abs(self.sim.ax[0].item()) * 3.0); m.state.current_input = m.state.current_motor * 0.5
            m.state.displacement = int(self.tacho); m.state.distance_traveled = int(abs(self.tacho))
            m.state.temp_fet = 35.0; m.state.temp_motor = 40.0; m.state.fault_code = 0
            self.pub_core.publish(m)
        self.publish_scan(r.scan[0].cpu().numpy(), now)
        if r.imu is not None and r.imu.shape[1] > 0:
            self.publish_imu(r.imu[0].cpu().numpy(), r.imu_att[0].cpu().numpy(), now,
                             r.imu_offsets.cpu().numpy())
        self.publish_ground_truth(st, od, stamp)
        self.pub_coll.publish(Bool(data=bool(r.collision[0])))
        if bool(r.collision[0]) and self.auto_reset:
            self.do_reset()
        if self.viewer is not None:
            self.viewer.update(r); self.viewer.render()
            if self.teleop is not None:
                self.window_teleop()
            if not self.viewer.alive:
                rclpy.shutdown()

    def publish_scan(self, ranges, now):
        """LaserScan.header.stamp is the acquisition time of the FIRST ray, per the ROS 2
        message definition, not of the scan's completion. `now` is the end of the control step,
        which is when the LAST ray was traced (Lidar.scan anchors time_frac at 0 for the final
        beam), so the header goes back by one full sweep.

        The convention a particular driver actually used when the raw bags were recorded is a
        separate, unvalidated question -- this only makes what we publish self-consistent with the
        metadata we publish beside it.
        """
        m = self.sim.scan_meta()
        sweep = m["time_increment"] * (len(ranges) - 1)
        stamp = (now - Duration(seconds=sweep)).to_msg()
        msg = LaserScan(); msg.header.stamp = stamp; msg.header.frame_id = self.laser_frame
        msg.angle_min, msg.angle_max, msg.angle_increment = m["angle_min"], m["angle_max"], m["angle_increment"]
        msg.time_increment, msg.scan_time = m["time_increment"], m["scan_time"]
        msg.range_min, msg.range_max = m["range_min"], m["range_max"]
        msg.ranges = ranges.astype(np.float32).tolist()
        self.pub_scan.publish(msg)

    def publish_imu(self, samples, att, now, offsets):
        # The sensor runs on its own clock: the last sample of a control step is generally not at
        # the step boundary and the count per step varies (1, 1, 1, 2 at 50 Hz on a 40 Hz loop), so
        # the simulator reports each sample's offset before `now` and they are used as given.
        K = samples.shape[0]
        for k in range(K):
            m = Imu(); m.header.frame_id = "imu"; m.header.stamp = (now - Duration(seconds=float(offsets[k]))).to_msg()
            g, a = samples[k, :3], samples[k, 3:]
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, g)
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, a)
            # `att` is one attitude estimate for the whole control step, measured at its end, so
            # only the LAST sample has an orientation that belongs to it. Copying it onto the
            # earlier samples would attribute a future attitude to them; omitting it everywhere
            # would leave a consumer of this topic alone without any quaternion.
            if k == K - 1:
                m.orientation = rpy_to_quat(float(att[0]), float(att[1]), float(att[2]))
            else:
                m.orientation_covariance[0] = -1.0              # not measured at this sample
            self.pub_imu_raw.publish(m)
        if self.pub_imu is not None:
            # the summary message reports the LATEST sample, so it is stamped at that sample's time
            v = VescImuStamped(); v.header.frame_id = "imu"
            v.header.stamp = (now - Duration(seconds=float(offsets[-1]))).to_msg()
            g, a = samples[-1, :3], samples[-1, 3:]
            # Measured against the quaternion in the SAME VescImuStamped message across four
            # recordings (competition / pre-competition / SPIN): ypr.x = -roll, ypr.y = +pitch,
            # ypr.z = -yaw, in degrees, with yaw wrapped. p95 of the summed deviation rounds to 0.
            # The previous (yaw, pitch, roll) tuple did not match the dataset in name or in sign, so
            # anything reading these fields off a bag and off this node disagreed. The quaternion
            # below stays canonical; ypr is the hardware's own convention.
            roll, pitch, yaw = float(att[0]), float(att[1]), float(att[2])
            v.imu.ypr.x, v.imu.ypr.y, v.imu.ypr.z = (-math.degrees(roll), math.degrees(pitch),
                                                     -math.degrees(yaw))
            v.imu.linear_acceleration.x, v.imu.linear_acceleration.y, v.imu.linear_acceleration.z = (float(a[0]) / 9.81, float(a[1]) / 9.81, float(a[2]) / 9.81)   # VESC reports g
            v.imu.angular_velocity.x, v.imu.angular_velocity.y, v.imu.angular_velocity.z = (math.degrees(float(g[0])), math.degrees(float(g[1])), math.degrees(float(g[2])))   # deg/s
            v.imu.orientation = rpy_to_quat(float(att[0]), float(att[1]), float(att[2]))
            self.pub_imu.publish(v)

    def publish_ground_truth(self, st, od, stamp):
        g = Odometry(); g.header.stamp = stamp; g.header.frame_id = self.map_frame; g.child_frame_id = self.base_frame
        g.pose.pose.position.x, g.pose.pose.position.y = float(st[0]), float(st[1])
        g.pose.pose.orientation = yaw_to_quat(float(st[2]))
        g.twist.twist.linear.x, g.twist.twist.linear.y, g.twist.twist.angular.z = float(st[3]), float(st[4]), float(st[5])
        self.pub_gt.publish(g)
        if self.publish_gt_tf:
            # map -> odom such that (map -> odom) * (odom -> base_link) == ground truth, i.e. a perfect
            # localizer. odom -> base_link is whatever vesc_to_odom integrated from our telemetry
            # (its own wheelbase/gain params), so use its latest /odom pose rather than our model.
            ox, oy, oyaw = self.odom_pose if self.odom_pose is not None else (float(od[0]), float(od[1]), float(od[2]))
            dyaw = float(st[2]) - oyaw; c, s = math.cos(dyaw), math.sin(dyaw)
            mx = float(st[0]) - (c * ox - s * oy); my = float(st[1]) - (s * ox + c * oy)
            t = TransformStamped(); t.header.stamp = stamp; t.header.frame_id = self.map_frame; t.child_frame_id = self.odom_frame
            t.transform.translation.x, t.transform.translation.y = mx, my; t.transform.rotation = yaw_to_quat(dyaw)
            self.tf.sendTransform([t])

    def publish_map(self):
        tr = self.track
        m = OccupancyGrid(); m.header.frame_id = self.map_frame; m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = tr.resolution; m.info.height, m.info.width = tr.occupancy.shape
        m.info.origin.position.x, m.info.origin.position.y = tr.origin[0], tr.origin[1]; m.info.origin.orientation.w = 1.0
        m.data = np.where(tr.occupancy, 100, 0).astype(np.int8).reshape(-1).tolist()
        self.pub_map.publish(m)


def main():
    rclpy.init()
    node = VescSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
