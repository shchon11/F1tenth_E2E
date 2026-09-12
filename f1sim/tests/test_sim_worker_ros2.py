"""The real worker process with the ROS 2 link on: a session built with `ros2="drive"` publishes
car 0's sensors, takes `/drive` from an outside node, answers `/f1sim/reset`, and reports the link
in its facts. Skipped without rclpy (open a shell that sourced the ROS workspace) or without a
display-free checkpoint the fixture generates.

CPU, one car, the generated legacy checkpoint: what is checked is the seam between the worker,
the env override and the ROS graph, not physics or policy quality.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("ackermann_msgs.msg")
pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f1sim.viewer.console import protocol as P            # noqa: E402
from test_sim_worker_smoke import WorkerProc, SMOKE_MAP, READY_TIMEOUT   # noqa: E402


class Outside:
    """A node that is not the worker: hears the scan, drives the car, calls the reset service."""

    def __init__(self):
        from rclpy.executors import SingleThreadedExecutor
        from ackermann_msgs.msg import AckermannDriveStamped
        from sensor_msgs.msg import LaserScan
        from std_srvs.srv import Empty
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("ros2_worker_test_outside")
        self.scans = 0
        self.node.create_subscription(LaserScan, "/scan", self._scan, 5)
        self.pub = self.node.create_publisher(AckermannDriveStamped, "/drive", 1)
        self.reset = self.node.create_client(Empty, "/f1sim/reset")
        self.exe = SingleThreadedExecutor(); self.exe.add_node(self.node)
        self.alive = True
        self.thread = threading.Thread(target=self._spin, daemon=True); self.thread.start()

    def _scan(self, _msg):
        self.scans += 1

    def _spin(self):
        while self.alive:
            self.exe.spin_once(timeout_sec=0.05)

    def drive_for(self, seconds, steer, speed, hz=40.0):
        from ackermann_msgs.msg import AckermannDriveStamped
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            m = AckermannDriveStamped(); m.drive.steering_angle = steer; m.drive.speed = speed
            self.pub.publish(m)
            time.sleep(1.0 / hz)

    def call_reset(self, timeout=5.0):
        from std_srvs.srv import Empty
        assert self.reset.wait_for_service(timeout_sec=timeout), "/f1sim/reset not offered"
        fut = self.reset.call_async(Empty.Request())
        end = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < end:
            time.sleep(0.02)
        assert fut.done(), "/f1sim/reset did not answer"

    def close(self):
        self.alive = False; self.thread.join(1.0)
        self.exe.remove_node(self.node); self.node.destroy_node()


@pytest.fixture(scope="module")
def worker(tmp_path_factory, tmp_legacy_run):
    w = WorkerProc(tmp_path_factory.mktemp("worker_ros2"))
    yield w
    w.close()


def test_ros2_drive_session(worker, tmp_legacy_run):
    cfg = P.SessionConfig(run=tmp_legacy_run, map_name=SMOKE_MAP, races=1, cars_per_race=1,
                          device="cpu", compile=False, ros2="drive", controller="legacy")
    worker.send(P.CMD_START, gen=300, config=cfg.to_dict())
    ready = worker.wait_for(P.MSG_READY, timeout=READY_TIMEOUT, gen=300)
    facts = ready["facts"]
    assert facts["ros2"] and facts["ros2"]["mode"] == "drive" and facts["ros2"]["car"] == 0
    assert "/scan" in facts["ros2"]["topics"]
    out = Outside()
    try:
        # sensors on the wire
        end = time.monotonic() + 15.0
        while out.scans == 0 and time.monotonic() < end:
            time.sleep(0.05)
        assert out.scans > 0, "the worker published no /scan"
        # nothing on /drive yet -> the commanded speed the console shows is 0
        idle = worker.collect_frames(1.0)
        assert idle and all(abs(f["dash"][1]) < 1e-6 for f in idle[-5:]), [f["dash"][1] for f in idle[-5:]]
        # an outside node drives: the console's dash shows that command, and the car moves
        t = threading.Thread(target=out.drive_for, args=(4.0, 0.05, 1.5), daemon=True); t.start()
        time.sleep(1.5)
        driven = worker.collect_frames(2.0)
        t.join()
        assert driven
        v_cmd = np.array([f["dash"][1] for f in driven])
        assert np.allclose(v_cmd[-10:], 1.5, atol=1e-5), v_cmd[-10:]
        assert max(f["vx"][0] for f in driven) > 0.5
        # silence -> coast: the command times out to speed 0 within a second
        time.sleep(1.2)
        quiet = worker.collect_frames(0.5)
        assert quiet and abs(quiet[-1]["dash"][1]) < 1e-6
        # the reset service is answered and the worker says so
        n_logs = sum(1 for m in worker.messages if m.get("kind") == P.MSG_LOG)
        out.call_reset()
        end = time.monotonic() + 10.0
        while time.monotonic() < end:
            worker.collect_frames(0.2)
            if worker.ctl.poll(0.0):
                worker.messages.append(worker.ctl.recv())
            logs = [m["text"] for m in worker.messages if m.get("kind") == P.MSG_LOG]
            if len(logs) > n_logs and any("/f1sim/reset" in t for t in logs[n_logs:]):
                break
        else:
            pytest.fail("no reset log from the worker after /f1sim/reset")
    finally:
        out.close()
    worker.send(P.CMD_STOP)
    worker.wait_for(P.MSG_STOPPED, timeout=60)


def test_ros2_mode_is_validated_before_anything_expensive(worker, tmp_legacy_run):
    cfg = P.SessionConfig(run=tmp_legacy_run, map_name=SMOKE_MAP, device="cpu", ros2="sideways")
    worker.send(P.CMD_START, gen=301, config=cfg.to_dict())
    err = worker.wait_for(P.MSG_ERROR, timeout=30, gen=301)
    assert "ROS2" in err["message"] and "sideways" in err["message"]
