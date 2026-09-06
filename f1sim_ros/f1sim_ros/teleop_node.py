"""Manual driving node with racing-game feel. Publishes ackermann_msgs/AckermannDriveStamped on
`teleop`, the topic the f1tenth_stack ackermann_mux gives top priority, so it works on the
simulator and on the real car alike.

Inputs:  /joy (sensor_msgs/Joy) with a preset mapping (`preset`: xbox | f1tenth), and/or the
         terminal keyboard (`keyboard:=true`: WASD / arrows, space handbrake, b boost toggle,
         r reset the sim, q quit). /odom gives the measured speed for the speed-sensitive lock.
"""
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_srvs.srv import Empty

from f1sim.teleop import PRESETS, DriveModel, Inputs, JoyDecoder, KeyRamp, TeleopConfig


class TeleopNode(Node):
    def __init__(self):
        super().__init__("f1sim_teleop")
        self.declare_parameter("preset", "xbox")
        self.declare_parameter("keyboard", False)
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("topic", "teleop")
        for k, v in TeleopConfig().__dict__.items():
            self.declare_parameter(k, v)
        p = lambda n: self.get_parameter(n).value
        self.cfg = TeleopConfig(**{k: float(p(k)) for k in TeleopConfig().__dict__})
        self.model = DriveModel(self.cfg)
        self.joy = JoyDecoder(self.cfg, PRESETS[p("preset")])
        self.keys = KeyRamp(self.cfg)
        self.pub = self.create_publisher(AckermannDriveStamped, p("topic"), 1)
        self.create_subscription(Joy, "joy", self.on_joy, 1)
        self.create_subscription(Odometry, "odom", self.on_odom, 1)
        self.reset_cli = self.create_client(Empty, "/f1sim/reset")
        self.v_meas = 0.0
        self.joy_inp = None; self.joy_time = 0.0
        self.held = {}          # key -> expiry time (terminal autorepeat keeps refreshing)
        self.boost = False
        self.rate = float(p("rate"))
        self.last = time.perf_counter()
        if bool(p("keyboard")):
            threading.Thread(target=self._key_reader, daemon=True).start()
            self.get_logger().info("keyboard: W/S throttle-brake, A/D steer, space handbrake, b boost, r reset, q quit")
        self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(f"teleop -> /{p('topic')}  preset={p('preset')}  v_max={self.cfg.v_max} boost={self.cfg.v_boost}")

    def on_joy(self, msg: Joy):
        self.joy_inp = self.joy.decode(list(msg.axes), list(msg.buttons)); self.joy_time = time.perf_counter()

    def on_odom(self, msg: Odometry):
        self.v_meas = msg.twist.twist.linear.x

    # --- terminal keyboard (raw mode); a key counts as held for 0.3 s after the last event
    def _key_reader(self):
        fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch == "\x1b":                              # arrow keys: ESC [ A/B/C/D
                        seq = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0.01)[0] else ""
                        ch = {"[A": "w", "[B": "s", "[C": "d", "[D": "a"}.get(seq, "")
                    ch = ch.lower()
                    if ch == "q":
                        rclpy.shutdown(); return
                    if ch == "b":
                        self.boost = not self.boost; continue
                    if ch == "r":
                        self.model.reset()
                        if self.reset_cli.service_is_ready():
                            self.reset_cli.call_async(Empty.Request())
                        continue
                    if ch in "wasd ":
                        self.held[ch] = time.perf_counter() + 0.3
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def tick(self):
        now = time.perf_counter(); dt = min(0.1, now - self.last); self.last = now
        if self.joy_inp is not None and now - self.joy_time < 0.5:
            inp = self.joy_inp
        else:
            h = lambda k: self.held.get(k, 0.0) > now
            inp = self.keys.update(h("a"), h("d"), h("w"), h("s"), dt, boost=self.boost, handbrake=h(" "))
        if inp.autonomy:
            return                                               # hand over: mux times out to /drive
        steer, v = self.model.update(inp, self.v_meas, dt)
        m = AckermannDriveStamped(); m.header.stamp = self.get_clock().now().to_msg()
        m.drive.steering_angle = float(steer); m.drive.speed = float(v)
        self.pub.publish(m)


def main():
    rclpy.init()
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
