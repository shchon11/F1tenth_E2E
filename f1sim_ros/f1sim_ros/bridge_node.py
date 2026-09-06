"""ROS 2 bridge: runs one f1sim environment in real time with the real car's interfaces.

Subscribes  /drive            ackermann_msgs/AckermannDriveStamped (steering_angle [rad], speed [m/s])
            /initialpose      geometry_msgs/PoseWithCovarianceStamped  -> teleport / reset
Publishes   /scan             sensor_msgs/LaserScan   (frame: laser)   noisy Hokuyo model
            /odom             nav_msgs/Odometry       (odom -> base_link) VESC dead-reckoning, drifts
            /ego_racecar/odom nav_msgs/Odometry       (map frame) ground truth
            /map              nav_msgs/OccupancyGrid  (latched)
            /f1sim/collision  std_msgs/Bool
            /sensors/imu      sensor_msgs/Imu   VESC IMU (vesc_driver topic): orientation = VESC attitude
                              estimate, rates/accels = latest sample; every raw sample also goes to
                              /sensors/imu/raw with its own stamp (imu.imu_rate, default 100 Hz)
TF          map -> odom (so that map -> base_link is ground truth; disable with publish_gt_tf:=false),
            odom -> base_link (from /odom), base_link -> laser (static)
Service     /f1sim/reset      std_srvs/Empty
"""
import math
import os
import time

import numpy as np
import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from f1sim import Config, Simulator, Track


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion(); q.z = math.sin(yaw / 2); q.w = math.cos(yaw / 2); return q


def quat_to_yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class BridgeNode(Node):
    def __init__(self):
        super().__init__("f1sim_bridge")
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("random_track_seed", 0)
        self.declare_parameter("config_yaml", "")
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("randomize", True)
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("gt_odom_topic", "/ego_racecar/odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("laser_frame", "laser")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("publish_gt_tf", True)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("auto_reset_on_collision", False)
        self.declare_parameter("start_s", -1.0)

        p = lambda n: self.get_parameter(n).value
        cfg = Config.from_yaml(p("config_yaml")) if p("config_yaml") else Config()
        cfg.rand.enabled = bool(p("randomize"))
        cfg.sim.device = p("device")
        cfg.sim.compile = (cfg.sim.device == "cuda")   # ~10 s warm-up once; needed to hold 40 Hz (CPU eager: ~30 ms/step)
        map_yaml = p("map_yaml")
        track = Track.from_ros_map(map_yaml) if map_yaml else Track.generate_random(int(p("random_track_seed")))
        self.sim = Simulator(track, cfg, num_envs=1, device=cfg.sim.device)
        self.cfg = cfg
        self.track = track
        self.rate = cfg.sim.control_rate
        self.base_frame, self.laser_frame = p("base_frame"), p("laser_frame")
        self.odom_frame, self.map_frame = p("odom_frame"), p("map_frame")
        self.publish_gt_tf = bool(p("publish_gt_tf"))
        self.cmd_timeout = float(p("cmd_timeout"))
        self.auto_reset = bool(p("auto_reset_on_collision"))
        self.start_s = float(p("start_s"))

        self.cmd = torch.zeros(1, 2)
        self.last_cmd_time = None
        self.create_subscription(AckermannDriveStamped, p("drive_topic"), self.on_drive, 1)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initialpose, 1)
        self.pub_scan = self.create_publisher(LaserScan, p("scan_topic"), 1)
        self.pub_odom = self.create_publisher(Odometry, p("odom_topic"), 1)
        self.pub_gt = self.create_publisher(Odometry, p("gt_odom_topic"), 1)
        self.pub_coll = self.create_publisher(Bool, "/f1sim/collision", 1)
        self.pub_imu = self.create_publisher(Imu, "/sensors/imu", 1) if cfg.imu.enabled else None
        self.pub_imu_raw = self.create_publisher(Imu, "/sensors/imu/raw", 10) if cfg.imu.enabled else None
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_map = self.create_publisher(OccupancyGrid, "/map", latched)
        self.tf = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)
        self.create_service(Empty, "/f1sim/reset", self.on_reset_srv)

        self.publish_map()
        self.publish_static_tf()
        self.do_reset()
        self.timer = self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(f"f1sim bridge: track={track.name} {track.shape} device={self.sim.device} "
                               f"rate={self.rate} Hz randomize={cfg.rand.enabled}")

    # ---------------------------------------------------------------- callbacks
    def on_drive(self, msg: AckermannDriveStamped):
        self.cmd = torch.tensor([[msg.drive.steering_angle, msg.drive.speed]], dtype=torch.float32)
        self.last_cmd_time = self.get_clock().now()

    def on_initialpose(self, msg: PoseWithCovarianceStamped):
        pose = msg.pose.pose
        self.do_reset(torch.tensor([[pose.position.x, pose.position.y, quat_to_yaw(pose.orientation)]]))

    def on_reset_srv(self, req, resp):
        self.do_reset(); return resp

    def do_reset(self, pose=None):
        if pose is None and self.track.centerline is not None:
            s = None if self.start_s < 0 else torch.tensor([self.start_s])
            pose = self.sim.sample_spawn(1, lateral_std=0.0, yaw_std=0.0, s=s)
        self.sim.reset(poses=pose)
        self.cmd = torch.zeros(1, 2)
        self.get_logger().info("reset at %s" % self.sim.state[0, :3].cpu().numpy().round(2))

    # ---------------------------------------------------------------- main loop
    def tick(self):
        now = self.get_clock().now()
        cmd = self.cmd
        if self.last_cmd_time is None or (now - self.last_cmd_time).nanoseconds * 1e-9 > self.cmd_timeout:
            cmd = torch.tensor([[cmd[0, 0].item(), 0.0]])      # command timeout: coast to stop
        r = self.sim.step(cmd)
        stamp = now.to_msg()
        st = r.state[0].cpu().numpy()
        od = r.odom[0].cpu().numpy()
        self.publish_scan(r.scan[0].cpu().numpy(), stamp)
        self.publish_odom(od, st, stamp)
        self.pub_coll.publish(Bool(data=bool(r.collision[0])))
        if self.pub_imu is not None and r.imu is not None and r.imu.shape[1] > 0:
            self.publish_imu(r.imu[0].cpu().numpy(), r.imu_att[0].cpu().numpy(), now)
        if bool(r.collision[0]) and self.auto_reset:
            self.do_reset()

    def publish_imu(self, samples: np.ndarray, att: np.ndarray, now):
        """samples (K, 6): gyro xyz, accel xyz at 1/imu_rate spacing ending at `now`."""
        from rclpy.duration import Duration
        K = samples.shape[0]; ts = 1.0 / self.cfg.imu.imu_rate
        for k in range(K):
            m = Imu(); m.header.frame_id = "imu"
            m.header.stamp = (now - Duration(seconds=(K - 1 - k) * ts)).to_msg()
            g, a = samples[k, :3], samples[k, 3:]
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = float(g[0]), float(g[1]), float(g[2])
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = float(a[0]), float(a[1]), float(a[2])
            m.orientation_covariance[0] = -1.0                      # raw: no orientation
            self.pub_imu_raw.publish(m)
        m = Imu(); m.header.frame_id = "imu"; m.header.stamp = now.to_msg()
        g, a = samples[-1, :3], samples[-1, 3:]
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = float(g[0]), float(g[1]), float(g[2])
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = float(a[0]), float(a[1]), float(a[2])
        r, p, y = float(att[0]), float(att[1]), float(att[2])
        cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
        m.orientation.w = cr * cp * cy + sr * sp * sy; m.orientation.x = sr * cp * cy - cr * sp * sy
        m.orientation.y = cr * sp * cy + sr * cp * sy; m.orientation.z = cr * cp * sy - sr * sp * cy
        self.pub_imu.publish(m)

    def publish_scan(self, ranges: np.ndarray, stamp):
        m = self.sim.scan_meta()
        msg = LaserScan()
        msg.header.stamp = stamp; msg.header.frame_id = self.laser_frame
        msg.angle_min, msg.angle_max, msg.angle_increment = m["angle_min"], m["angle_max"], m["angle_increment"]
        msg.time_increment, msg.scan_time = m["time_increment"], m["scan_time"]
        msg.range_min, msg.range_max = m["range_min"], m["range_max"]
        msg.ranges = ranges.astype(np.float32).tolist()
        self.pub_scan.publish(msg)

    def publish_odom(self, od, st, stamp):
        # VESC odom (drifting), odom -> base_link
        o = Odometry(); o.header.stamp = stamp; o.header.frame_id = self.odom_frame; o.child_frame_id = self.base_frame
        o.pose.pose.position.x, o.pose.pose.position.y = float(od[0]), float(od[1])
        o.pose.pose.orientation = yaw_to_quat(float(od[2]))
        o.twist.twist.linear.x, o.twist.twist.angular.z = float(od[3]), float(od[4])
        self.pub_odom.publish(o)
        t = TransformStamped(); t.header.stamp = stamp; t.header.frame_id = self.odom_frame; t.child_frame_id = self.base_frame
        t.transform.translation.x, t.transform.translation.y = float(od[0]), float(od[1])
        t.transform.rotation = yaw_to_quat(float(od[2]))
        tfs = [t]
        # ground truth in map frame
        g = Odometry(); g.header.stamp = stamp; g.header.frame_id = self.map_frame; g.child_frame_id = self.base_frame
        g.pose.pose.position.x, g.pose.pose.position.y = float(st[0]), float(st[1])
        g.pose.pose.orientation = yaw_to_quat(float(st[2]))
        g.twist.twist.linear.x, g.twist.twist.linear.y, g.twist.twist.angular.z = float(st[3]), float(st[4]), float(st[5])
        self.pub_gt.publish(g)
        if self.publish_gt_tf:
            # map -> odom such that (map -> odom) * (odom -> base) == ground truth
            dyaw = float(st[2] - od[2])
            c, s = math.cos(dyaw), math.sin(dyaw)
            mx = float(st[0]) - (c * od[0] - s * od[1]); my = float(st[1]) - (s * od[0] + c * od[1])
            mo = TransformStamped(); mo.header.stamp = stamp; mo.header.frame_id = self.map_frame; mo.child_frame_id = self.odom_frame
            mo.transform.translation.x, mo.transform.translation.y = mx, my
            mo.transform.rotation = yaw_to_quat(dyaw)
            tfs.append(mo)
        self.tf.sendTransform(tfs)

    def publish_static_tf(self):
        t = TransformStamped(); t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_frame; t.child_frame_id = self.laser_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = self.cfg.lidar.mount_x, self.cfg.lidar.mount_y, self.cfg.lidar.mount_z
        t.transform.rotation = yaw_to_quat(self.cfg.lidar.mount_yaw)
        ti = TransformStamped(); ti.header.stamp = t.header.stamp
        ti.header.frame_id = self.base_frame; ti.child_frame_id = "imu"
        ti.transform.translation.x, ti.transform.translation.y, ti.transform.translation.z = self.cfg.imu.imu_x, self.cfg.imu.imu_y, self.cfg.imu.imu_z
        ti.transform.rotation.w = 1.0
        self.tf_static.sendTransform([t, ti])

    def publish_map(self):
        tr = self.track
        m = OccupancyGrid(); m.header.frame_id = self.map_frame; m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = tr.resolution; m.info.height, m.info.width = tr.occupancy.shape
        m.info.origin.position.x, m.info.origin.position.y = tr.origin[0], tr.origin[1]
        m.info.origin.orientation.w = 1.0
        m.data = np.where(tr.occupancy, 100, 0).astype(np.int8).reshape(-1).tolist()
        self.pub_map.publish(m)


def main():
    rclpy.init()
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
