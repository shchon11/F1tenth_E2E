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


class ValueListener:
    """Keeps the last message of every sensor topic, so what went out can be compared with the
    tensor it was built from rather than merely counted."""

    def __init__(self):
        from rclpy.executors import SingleThreadedExecutor
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, LaserScan
        self.node = rclpy.create_node("ros_link_value_listener")
        self.last = {}
        for topic, typ in (("/scan", LaserScan), ("/odom", Odometry),
                           ("/ego_racecar/odom", Odometry), ("/sensors/imu/raw", Imu)):
            self.node.create_subscription(typ, topic, lambda m, t=topic: self.last.__setitem__(t, m), 10)
        self.exe = SingleThreadedExecutor(); self.exe.add_node(self.node)
        self.alive = True
        self.thread = threading.Thread(target=self._spin, daemon=True); self.thread.start()

    def _spin(self):
        while self.alive:
            self.exe.spin_once(timeout_sec=0.05)

    def have_all(self):
        return len(self.last) == 4

    def close(self):
        self.alive = False; self.thread.join(1.0)
        self.exe.remove_node(self.node); self.node.destroy_node()


def test_what_is_published_is_the_simulators_own_numbers(env, track):
    """Values, not counts. `RosLink.publish` flattens a StepResult into one host transfer and slices
    it back apart, and it used to slice with written-in widths -- state 7, odom 5, attitude 2, IMU
    attitude 3. The wheel model made the state 8 wide and every field after it came out one float
    late: `/odom` led with the wheel speed, each IMU row read [previous a_z, g_x, g_y, g_z, a_x, a_y]
    -- gravity in `angular_velocity.x` one message in five, none in `linear_acceleration.z` -- and
    every LiDAR beam was one index late. The test above only counted scans and beams, so all of it
    passed. At a standstill most of it reads zero, so it survived until the user looked at a parked
    car's IMU in the console and saw it.

    The car is driven first so position and speed are not zero, which is the case a shifted slice
    cannot hide in.
    """
    link = RosLink(env.sim, track, None, mode="publish", car=0, node_name="f1sim_console_values",
                   viz_hz=0.0)
    lis = ValueListener()
    try:
        act = torch.zeros(env.B, env.act_dim); act[:, 1] = 0.3
        for _ in range(40):
            env.step(act)
        r = env.last_result
        assert float(r.odom[0, 3].abs()) > 0.2, "the car is not moving, so a shift could hide"
        assert _wait(lambda: (link.publish(r, env), lis.have_all())[1], timeout=8.0), \
            f"not every topic arrived: {sorted(lis.last)}"
        c = link.car
        o = lis.last["/odom"]
        assert o.pose.pose.position.x == pytest.approx(float(r.odom[c, 0]), abs=1e-5)
        assert o.pose.pose.position.y == pytest.approx(float(r.odom[c, 1]), abs=1e-5)
        assert o.twist.twist.linear.x == pytest.approx(float(r.odom[c, 3]), abs=1e-5), \
            "/odom speed is not the odometry's speed"
        g = lis.last["/ego_racecar/odom"]
        assert g.pose.pose.position.x == pytest.approx(float(r.state[c, 0]), abs=1e-5)
        scan = lis.last["/scan"]
        src = r.scan[c].cpu().numpy()
        got = __import__("numpy").asarray(scan.ranges, dtype="float32")
        finite = __import__("numpy").isfinite(src)
        assert got.shape == src.shape
        assert __import__("numpy").allclose(got[finite], src[finite], atol=1e-4), \
            "the published scan is not the simulated one (a shifted slice puts every beam one late)"
        m = lis.last["/sensors/imu/raw"]
        row = r.imu[c, -1].cpu().numpy()                       # the last sample of the step
        pub = [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
               m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]
        assert __import__("numpy").allclose(pub, row, atol=1e-4), f"IMU {pub} != source {row.tolist()}"
        # and the physics of it: a car on the ground reads about 1 g, on z, and no gravity on a gyro
        assert abs(m.linear_acceleration.z) > 7.0, "no gravity on the accelerometer's z axis"
        assert abs(m.angular_velocity.x) < 3.0, "a gyro axis is reading something like 9.81"
    finally:
        lis.close()
        link.close()
