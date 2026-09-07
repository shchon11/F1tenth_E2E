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


def quat_to_rp(q):
    sinr = 2 * (q.w * q.x + q.y * q.z); cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
    sinp = max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))
    return math.atan2(sinr, cosr), math.asin(sinp)


class PolicyNode(Node):
    def __init__(self):
        super().__init__("f1sim_policy")
        self.declare_parameter("checkpoint", ""); self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("speed_cap", 4.0); self.declare_parameter("steer_max", 0.4189)
        self.declare_parameter("drive_topic", "drive"); self.declare_parameter("enabled", True)
        p = lambda n: self.get_parameter(n).value
        self.device = torch.device(p("device"))
        self.model, extra = load_checkpoint(p("checkpoint"), self.device); self.model.eval()
        self.spec = ObsSpec(**extra["spec"]) if extra.get("spec") else ObsSpec()
        self.obs = ObsBuilder(self.spec, self.device)
        self.speed_cap = float(p("speed_cap")); self.steer_max = float(p("steer_max"))
        self.v = 0.0; self.imu_buf = []; self.att = (0.0, 0.0); self.yaw_rate = 0.0
        # plan action space: the same iLQR tracker as in training turns the local trajectory into
        # (steer, speed); cmd_delay = the measured command latency of this car (calibrate once)
        self.declare_parameter("wheelbase", 0.3302); self.declare_parameter("cmd_delay", 0.06)
        # residual servo calibration the stack's steering_angle_to_servo_offset/gain do not absorb (rad, ratio)
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
        self.v = m.twist.twist.linear.x

    def on_imu(self, m: Imu):
        self.imu_buf.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
                             m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        self.yaw_rate = m.angular_velocity.z
        if VescImuStamped is None:                                  # no VESC attitude: use the Imu orientation
            self.att = quat_to_rp(m.orientation)

    def on_vesc_imu(self, m):
        self.att = (math.radians(m.imu.ypr.z), math.radians(m.imu.ypr.y))   # VescImu ypr is (yaw, pitch, roll) in degrees

    def on_scan(self, m: LaserScan):
        t0 = time.perf_counter()
        r = np.asarray(m.ranges, dtype=np.float32)
        if len(r) != self.spec.n_beams:                              # e.g. 1081 beams from urg_node: resample
            r = np.interp(np.linspace(0, len(r) - 1, self.spec.n_beams), np.arange(len(r)), r)
        imu_mean = np.mean(self.imu_buf, 0) if self.imu_buf else np.zeros(6); self.imu_buf = []
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
