"""`system_check`: the deployment "is everything there?" tool, live on the graph.

    ros2 run f1sim_ros system_check
    ros2 run f1sim_ros system_check --ros-args -p period:=2.0 -p once:=true
    python3 -m f1sim_ros.system_check <bag>           # the same rules over a finished recording

It subscribes to every topic in `config/record.yaml`, measures the rate and the staleness of each
against what the profile says it should be, and reports what is in force -- the checkpoint from
`/f1sim/policy_state`, the arm, the friction and the traction state from
`/f1sim/controller/diag`. The report goes to the log and to `/f1sim/system_check` as a
`DiagnosticArray`, so it is in the bag with everything else.

It never publishes a command and never subscribes to anything the graph does not already carry.
"""
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from f1sim_ros.record_profile import load as load_profile
from f1sim_ros.recorder import LATCHED
from f1sim_ros.system_check import MISSING, OK, WARN, check_timeline

#: Arrival times kept per topic. 400 at 40 Hz is ten seconds, which is more window than any rate
#: question needs and bounds the memory of a node that is meant to be left running.
WINDOW = 400


class SystemCheckNode(Node):
    def __init__(self):
        super().__init__("f1sim_system_check")
        self.declare_parameter("profile", "")
        self.declare_parameter("period", 1.0)
        self.declare_parameter("window_s", 5.0)
        self.declare_parameter("rate_tolerance", 0.5)
        self.declare_parameter("stale_after", 0.5)
        self.declare_parameter("once", False)
        self.declare_parameter("car", False)
        p = lambda n: self.get_parameter(n).value
        self.profile = load_profile(str(p("profile")))
        if bool(p("car")):
            self.profile = self.profile.on_car()
        self.window_s = float(p("window_s"))
        self.rate_tolerance = float(p("rate_tolerance"))
        self.stale_after = float(p("stale_after"))
        self.once = bool(p("once"))
        self.stamps = {t.name: [] for t in self.profile.topics}
        self.versions = {}
        self.unavailable = []
        from rosidl_runtime_py.utilities import get_message
        for t in self.profile.topics:
            try:
                cls = get_message(t.type)
            except Exception as exc:
                # A message package that is not installed. Reported rather than skipped silently:
                # `vesc_msgs` missing on the car is the difference between "no motor current" and
                # "the traction guard cannot corroborate a spin".
                self.unavailable.append(t.name)
                self.get_logger().warning(f"{t.name}: {t.type} is not importable here ({exc})")
                continue
            qos = QoSProfile(depth=10, history=QoSHistoryPolicy.KEEP_LAST,
                             reliability=QoSReliabilityPolicy.RELIABLE)
            if t.name in LATCHED:
                qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
            self.create_subscription(cls, t.name, self._cb_for(t.name), qos)
        self.pub = self.create_publisher(DiagnosticArray, "/f1sim/system_check", 1)
        self.t0 = self.clock()
        self.create_timer(float(p("period")), self.report)
        self.get_logger().info(
            f"system check over {len(self.stamps)} topics of {self.profile.path}, every "
            f"{float(p('period')):.1f} s")

    def clock(self):
        return time.monotonic()

    def _cb_for(self, name):
        def cb(msg, _n=name):
            self.stamps[_n].append(self.clock())
            if len(self.stamps[_n]) > WINDOW:
                del self.stamps[_n][:-WINDOW]
            if _n == "/f1sim/policy_state":
                self.versions["checkpoint"] = str(msg.checkpoint)
                self.versions["memory"] = str(msg.memory_kind) or "feedforward"
                self.versions["memory_clears"] = str(int(msg.memory_clears))
                self.versions["policy_inhibited"] = str(bool(msg.inhibited))
            elif _n == "/f1sim/controller/diag":
                for st in msg.status:
                    for kv in st.values:
                        if kv.key in ("arm", "mu", "traction", "traction_state",
                                      "clearance_margin_m", "speed_cap_mps", "plan_unmatched"):
                            self.versions[kv.key] = str(kv.value)
        return cb

    def report(self):
        now = self.clock()
        lo = now - self.window_s
        window = {k: [t for t in v if t >= lo] for k, v in self.stamps.items()}
        rep = check_timeline(window, self.profile, duration_s=min(self.window_s, now - self.t0),
                             now=now, rate_tolerance=self.rate_tolerance,
                             stale_after=self.stale_after, versions=self.versions, source="live")
        (self.get_logger().info if rep.ok else self.get_logger().warning)("\n" + rep.text())
        arr = DiagnosticArray(); arr.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus()
        st.level = DiagnosticStatus.OK if rep.ok else DiagnosticStatus.ERROR
        st.name = "f1sim: system check"
        st.hardware_id = self.versions.get("checkpoint", "")
        st.message = ("every required topic present and on rate" if rep.ok else
                      "missing or slow: " + ", ".join(t.name for t in rep.topics
                                                      if t.status != OK and t.required))
        st.values = ([KeyValue(key=t.name, value=f"{t.status} {t.rate_hz:.1f} Hz "
                                                 f"age {t.age_s:.2f} s {t.note}".strip())
                      for t in rep.topics]
                     + [KeyValue(key=k, value=v) for k, v in sorted(self.versions.items())])
        arr.status = [st]
        self.pub.publish(arr)
        if self.once:
            raise SystemExit(0 if rep.ok else 1)


def main():
    rclpy.init(); n = SystemCheckNode()
    code = 0
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit) as exc:
        code = int(getattr(exc, "code", 0) or 0)
    n.destroy_node()
    if rclpy.ok(): rclpy.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
