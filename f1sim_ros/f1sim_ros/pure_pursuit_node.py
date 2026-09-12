"""Pure pursuit on the published racing line -- the smallest possible *external* controller.

This node knows nothing about the simulator. It reads three topics any ROS 2 stack could provide
and publishes the one the car listens to:

    in   /f1sim/raceline        nav_msgs/Path               the line to follow (latched)
         /f1sim/raceline_speed  std_msgs/Float32MultiArray  target speed per path pose (optional)
         /ego_racecar/odom      nav_msgs/Odometry           where the car is (param `odom_topic`)
    out  /drive                 ackermann_msgs/AckermannDriveStamped
         /pure_pursuit/lookahead visualization_msgs/Marker   the point being chased, for rviz

The default pose source is the simulator's ground truth. `/odom` is the drifting dead-reckoning
the real car has, and pure pursuit on it walks off the line within a lap -- which is the point of
publishing both. Switch with `odom_topic:=/odom` to see it happen, or run a localiser that
publishes a corrected pose and point this at that.

    ros2 run f1sim_ros pure_pursuit --ros-args -p max_speed:=4.0 -p lookahead_gain:=0.4

The geometry (`pure_pursuit_step`) is a plain function so it can be tested without ROS.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


# ================================================================ pure geometry
def nearest_index(path_xy: np.ndarray, x: float, y: float) -> int:
    d = path_xy - np.array([x, y], dtype=np.float64)
    return int(np.argmin(np.einsum("ij,ij->i", d, d)))


def lookahead_index(path_xy: np.ndarray, start: int, x: float, y: float, lookahead: float) -> int:
    """First path index at or after `start` (cyclically) that is at least `lookahead` away from
    (x, y). Walks forward, so the target never jumps backwards across a hairpin."""
    n = int(path_xy.shape[0])
    L2 = float(lookahead) ** 2
    for k in range(n):
        j = (start + k) % n
        dx, dy = float(path_xy[j, 0] - x), float(path_xy[j, 1] - y)
        if dx * dx + dy * dy >= L2:
            return j
    return start


def pure_pursuit_step(x: float, y: float, yaw: float, path_xy: np.ndarray, lookahead: float,
                      wheelbase: float, max_steer: float = math.pi / 2) -> Tuple[float, int, Tuple[float, float]]:
    """One pure-pursuit update. Returns (steer [rad], target index, target xy).

    Standard geometry: the target in the body frame is (xl, yl); the arc through the rear axle and
    that point has curvature 2 * yl / d^2 where d is the actual distance to the target (not the
    nominal lookahead, which the discrete path only approximates); a kinematic bicycle turns that
    arc with steer = atan(wheelbase * curvature)."""
    path_xy = np.asarray(path_xy, dtype=np.float64)
    i0 = nearest_index(path_xy, x, y)
    it = lookahead_index(path_xy, i0, x, y, lookahead)
    tx, ty = float(path_xy[it, 0]), float(path_xy[it, 1])
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = tx - x, ty - y
    xl, yl = c * dx + s * dy, -s * dx + c * dy
    d2 = xl * xl + yl * yl
    if d2 < 1e-6:
        return 0.0, it, (tx, ty)
    curvature = 2.0 * yl / d2
    steer = math.atan(wheelbase * curvature)
    steer = max(-max_steer, min(max_steer, steer))
    return steer, it, (tx, ty)


def lookahead_for_speed(v: float, gain: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, gain * abs(float(v))))


# ================================================================ the node
def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry, Path
    from std_msgs.msg import Float32MultiArray
    from visualization_msgs.msg import Marker

    class PurePursuit(Node):
        def __init__(self):
            super().__init__("pure_pursuit")
            self.declare_parameter("path_topic", "/f1sim/raceline")
            self.declare_parameter("speed_topic", "/f1sim/raceline_speed")
            self.declare_parameter("odom_topic", "/ego_racecar/odom")
            self.declare_parameter("drive_topic", "/drive")
            self.declare_parameter("wheelbase", 0.3302)
            self.declare_parameter("max_steer", 0.4189)
            self.declare_parameter("lookahead_min", 0.8)
            self.declare_parameter("lookahead_max", 2.5)
            self.declare_parameter("lookahead_gain", 0.35)     # [s]: lookahead = gain * speed
            self.declare_parameter("max_speed", 4.0)
            self.declare_parameter("speed_scale", 1.0)          # x the line's own target speed
            self.declare_parameter("fallback_speed", 2.0)       # when no speed profile arrived
            self.declare_parameter("rate", 40.0)
            self.declare_parameter("pose_timeout", 0.5)
            p = lambda n: self.get_parameter(n).value
            self.wheelbase, self.max_steer = float(p("wheelbase")), float(p("max_steer"))
            self.l_min, self.l_max, self.l_gain = float(p("lookahead_min")), float(p("lookahead_max")), float(p("lookahead_gain"))
            self.v_max, self.v_scale, self.v_fallback = float(p("max_speed")), float(p("speed_scale")), float(p("fallback_speed"))
            self.pose_timeout = float(p("pose_timeout"))
            self.path: Optional[np.ndarray] = None
            self.speeds: Optional[np.ndarray] = None
            self.pose = None                     # (x, y, yaw, v)
            self.pose_time = None
            latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(Path, p("path_topic"), self.on_path, latched)
            self.create_subscription(Float32MultiArray, p("speed_topic"), self.on_speed, latched)
            self.create_subscription(Odometry, p("odom_topic"), self.on_odom, 1)
            self.pub_drive = self.create_publisher(AckermannDriveStamped, p("drive_topic"), 1)
            self.pub_mark = self.create_publisher(Marker, "/pure_pursuit/lookahead", 1)
            self.timer = self.create_timer(1.0 / float(p("rate")), self.tick)
            self._said_waiting = False
            self.get_logger().info(f"pure pursuit: path={p('path_topic')} pose={p('odom_topic')} -> {p('drive_topic')} "
                                   f"L=[{self.l_min}, {self.l_max}] gain={self.l_gain}s v_max={self.v_max}")

        def on_path(self, msg):
            if len(msg.poses) < 2:
                return
            self.path = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses], dtype=np.float64)
            self.get_logger().info(f"path: {len(self.path)} poses")

        def on_speed(self, msg):
            self.speeds = np.asarray(msg.data, dtype=np.float64) if len(msg.data) else None

        def on_odom(self, msg):
            q = msg.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw, msg.twist.twist.linear.x)
            self.pose_time = self.get_clock().now()

        def tick(self):
            if self.path is None or self.pose is None:
                if not self._said_waiting:
                    self.get_logger().info("waiting for the path and a pose…")
                    self._said_waiting = True
                return
            if (self.get_clock().now() - self.pose_time).nanoseconds * 1e-9 > self.pose_timeout:
                return                          # stale pose: publish nothing, the car times out and coasts
            x, y, yaw, v = self.pose
            L = lookahead_for_speed(v, self.l_gain, self.l_min, self.l_max)
            steer, it, (tx, ty) = pure_pursuit_step(x, y, yaw, self.path, L, self.wheelbase, self.max_steer)
            if self.speeds is not None and len(self.speeds) == len(self.path):
                v_cmd = self.v_scale * float(self.speeds[it])
            else:
                v_cmd = self.v_fallback
            v_cmd = max(0.0, min(self.v_max, v_cmd))
            d = AckermannDriveStamped()
            d.header.stamp = self.get_clock().now().to_msg(); d.header.frame_id = "base_link"
            d.drive.steering_angle, d.drive.speed = float(steer), float(v_cmd)
            self.pub_drive.publish(d)
            m = Marker()
            m.header.stamp = d.header.stamp; m.header.frame_id = "map"
            m.ns, m.id, m.type, m.action = "pure_pursuit", 0, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = tx, ty, 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.18
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.9, 0.9
            self.pub_mark.publish(m)

    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
