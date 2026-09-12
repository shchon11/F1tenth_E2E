"""`RosLink` against a real rclpy context: a CPU simulator publishes, a second node hears the scan
and answers on `/drive`, and the command reaches the link. Skipped where rclpy is not importable
(open a shell that sourced the ROS workspace to run it)."""
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
torch = pytest.importorskip("torch")
pytest.importorskip("ackermann_msgs.msg")

from f1sim import Config, Track                            # noqa: E402
from f1sim.gym_env import EnvConfig, F1VecEnv              # noqa: E402
from f1sim.viewer.ros_link import RosLink                  # noqa: E402


@pytest.fixture(scope="module")
def track():
    return Track.generate_random(5)


@pytest.fixture(scope="module")
def env(track):
    cfg = Config()
    cfg.sim.compile = False
    cfg.rand.enabled = False
    e = EnvConfig(action_mode="direct", speed_cap=4.0, max_steps=10_000, compile_tracker=False)
    env = F1VecEnv([track], cfg, e, num_envs=2, device="cpu")
    env.reset()
    return env


class Listener:
    """A plain node on the same context: counts scans, sends one /drive back."""

    def __init__(self):
        from rclpy.executors import SingleThreadedExecutor
        from ackermann_msgs.msg import AckermannDriveStamped
        from sensor_msgs.msg import LaserScan
        from visualization_msgs.msg import MarkerArray
        self.node = rclpy.create_node("ros_link_test_listener")
        self.scans = 0; self.markers = 0; self.n_beams = None
        self.node.create_subscription(LaserScan, "/scan", self._scan, 5)
        self.node.create_subscription(MarkerArray, "/f1sim/viz/cars", self._cars, 5)
        self.pub = self.node.create_publisher(AckermannDriveStamped, "/drive", 1)
        self.exe = SingleThreadedExecutor(); self.exe.add_node(self.node)
        self.alive = True
        self.thread = threading.Thread(target=self._spin, daemon=True); self.thread.start()

    def _scan(self, msg):
        self.scans += 1; self.n_beams = len(msg.ranges)

    def _cars(self, msg):
        self.markers = len(msg.markers)

    def _spin(self):
        while self.alive:
            self.exe.spin_once(timeout_sec=0.05)

    def drive(self, steer, speed):
        from ackermann_msgs.msg import AckermannDriveStamped
        m = AckermannDriveStamped(); m.drive.steering_angle = steer; m.drive.speed = speed
        self.pub.publish(m)

    def close(self):
        self.alive = False; self.thread.join(1.0)
        self.exe.remove_node(self.node); self.node.destroy_node()


def _wait(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def test_link_publishes_and_takes_drive_commands(env, track):
    link = RosLink(env.sim, track, None, mode="drive", car=0, node_name="f1sim_console_test", viz_hz=0.0)
    lis = Listener()
    try:
        # discovery takes a moment; step and publish until the listener sees a scan
        def pump():
            env.step(torch.zeros(env.B, env.act_dim))
            link.publish(env.last_result, env)
        assert _wait(lambda: (pump(), lis.scans > 0)[1], timeout=8.0), "no /scan arrived"
        assert lis.n_beams == int(env.last_result.scan.shape[1])
        assert lis.markers == env.B                                 # every car in the marker array
        steer, speed, fresh = link.command()
        assert not fresh and speed == 0.0                           # nothing on /drive yet: coast
        lis.drive(0.15, 2.0)
        assert _wait(lambda: link.drive_count() > 0, timeout=5.0), "/drive never reached the link"
        steer, speed, fresh = link.command()
        assert fresh and steer == pytest.approx(0.15) and speed == pytest.approx(2.0)
        # and through the env: the command is what car 0 drives, car 1 stays the policy's
        env.set_external_command(link.car, steer, speed)
        act = torch.zeros(env.B, env.act_dim); act[:, 1] = -1.0
        env.step(act)
        assert float(env.last_cmd[0, 1]) == pytest.approx(2.0) and float(env.last_cmd[1, 1]) == pytest.approx(0.0)
        assert link.published > 0
        facts = link.facts()
        assert facts["mode"] == "drive" and facts["car"] == 0 and "/scan" in facts["topics"]
    finally:
        lis.close()
        link.close()
