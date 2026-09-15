"""Record the graph's own topics from inside a node, so a sim run and a car run match.

`ros2 bag record $(python3 -m f1sim_ros.recorder --args)` is the other way to get the same bag, and
is what the car uses. This class is what lets `bridge_node record:=<dir>` produce one without a
second process and without the launch file having to know the topic list: a simulator run then
writes exactly the profile a car run writes, and `learn/bagdata.py` cannot tell them apart.

The recorder subscribes to every topic in the profile -- including the ones the host node publishes
itself, which reach it through the middleware like anyone else's. A type that is not installed
(`vesc_msgs` on a desk machine) is skipped with a log line rather than failing the run.
"""
from __future__ import annotations

import atexit
import os
import time

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from f1sim_ros.record_profile import RecordProfile, load as load_profile


#: Latched publishers (`/tf_static`, `/map`) need a transient-local subscription or the recorder
#: joins after the one message was sent and records nothing.
LATCHED = ("/tf_static", "/map", "/f1sim/raceline", "/f1sim/raceline_speed", "/f1sim/centerline")


class BagRecorder:
    """A rosbag2 writer fed by subscriptions to a `RecordProfile`."""

    def __init__(self, node, out_dir: str, profile: RecordProfile = None, *, sim: bool = True):
        import rosbag2_py
        from rclpy.serialization import serialize_message
        from rosidl_runtime_py.utilities import get_message

        self.node = node
        self.profile = profile or load_profile()
        if not sim:
            self.profile = self.profile.on_car()
        self._serialize = serialize_message
        self.path = out_dir
        os.makedirs(os.path.dirname(os.path.abspath(out_dir)) or ".", exist_ok=True)
        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(rosbag2_py.StorageOptions(uri=out_dir, storage_id="sqlite3"),
                         rosbag2_py.ConverterOptions("", ""))
        self.counts = {}
        self.skipped = []
        self._subs = []
        for t in self.profile.topics:
            try:
                cls = get_message(t.type)
            except Exception as exc:                       # a message package that is not installed
                self.skipped.append((t.name, str(exc)))
                continue
            self.writer.create_topic(rosbag2_py.TopicMetadata(
                name=t.name, type=t.type, serialization_format="cdr"))
            self.counts[t.name] = 0
            qos = QoSProfile(depth=10, history=QoSHistoryPolicy.KEEP_LAST,
                             reliability=QoSReliabilityPolicy.RELIABLE)
            if t.name in LATCHED:
                qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
            self._subs.append(node.create_subscription(
                cls, t.name, self._writer_for(t.name), qos))
        # A bag whose `metadata.yaml` was never written cannot be opened by anything -- not
        # `ros2 bag info`, not `learn/bagdata.py`, not `system_check`. `main()` closes the recorder
        # on a clean shutdown; this covers the run that was killed, which is most of them.
        atexit.register(self.close)
        node.get_logger().info(
            f"recording {len(self.counts)} topics of {self.profile.path} to {out_dir}"
            + (f" (skipped: {', '.join(n for n, _ in self.skipped)})" if self.skipped else ""))

    def _writer_for(self, name):
        def cb(msg, _name=name):
            self.writer.write(_name, self._serialize(msg), self.node.get_clock().now().nanoseconds)
            self.counts[_name] += 1
        return cb

    def close(self):
        """Stop recording and finalise the bag.

        `writer.close()` explicitly, not just dropping the reference: it is what writes
        `metadata.yaml`, and a bag directory with a `.db3` and no metadata cannot be opened by
        anything -- `rosbag2_py`, `ros2 bag info`, `learn/bagdata.py` or `system_check` -- so a run
        that was killed without reaching here has recorded nothing usable.
        """
        for s in self._subs:
            try:
                self.node.destroy_subscription(s)
            except Exception:            # at exit the context may already be gone
                pass
        self._subs = []
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.node.get_logger().info(
            "recorded " + ", ".join(f"{k} x{v}" for k, v in sorted(self.counts.items())))


def timestamped_dir(root: str, prefix: str = "f1sim") -> str:
    return os.path.join(root, f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}")


def main():
    """`--args` prints the topic list for `ros2 bag record`; no argument prints the profile."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="")
    ap.add_argument("--args", action="store_true", help="print the topics as ros2 bag record args")
    ap.add_argument("--car", action="store_true", help="drop the simulator-only topics")
    a = ap.parse_args()
    p = load_profile(a.profile)
    if a.car:
        p = p.on_car()
    if a.args:
        print(" ".join(p.names))
        return 0
    print(f"{p.path}  (version {p.version})")
    for t in p.topics:
        flag = "required" if t.required else ("sim-only" if t.sim_only else "optional")
        print(f"  {t.name:28s} {t.type:44s} {t.rate_hz:5.1f} Hz  {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
