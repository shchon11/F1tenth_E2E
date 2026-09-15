""""Is everything there?" -- the deployment check, as a function of timestamps.

Two callers, one rule set: `system_check_node` watching a live graph, and `check_bag` reading a
finished recording. Both reduce to `check_timeline`, which takes the arrival times of each topic
and the profile they were supposed to arrive at and says what is missing, slow or stale. Keeping
the rules here rather than in the node is what lets "a bag with a missing topic" be a test that
runs in a second with no ROS graph at all.

Nothing in this module imports rclpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

OK, WARN, MISSING = "ok", "warn", "missing"


@dataclass
class TopicCheck:
    name: str
    status: str
    required: bool
    count: int = 0
    rate_hz: float = 0.0
    expect_hz: float = 0.0
    age_s: float = -1.0
    max_gap_s: float = -1.0
    note: str = ""

    def line(self) -> str:
        mark = {OK: "ok  ", WARN: "WARN", MISSING: "MISS"}[self.status]
        rate = f"{self.rate_hz:6.1f} Hz" if self.count > 1 else f"{self.count:6d} msg"
        want = f" (want {self.expect_hz:.0f})" if self.expect_hz else ""
        age = f"  age {self.age_s:.2f} s" if self.age_s >= 0 else ""
        return f"  [{mark}] {self.name:26s} {rate}{want}{age}  {self.note}".rstrip()


@dataclass
class SystemReport:
    topics: List[TopicCheck] = field(default_factory=list)
    duration_s: float = 0.0
    #: What was driving, as the graph itself reported it: the checkpoint from
    #: `/f1sim/policy_state`, the arm / friction / traction state from `/f1sim/controller/diag`.
    versions: Dict[str, str] = field(default_factory=dict)
    source: str = ""

    @property
    def ok(self) -> bool:
        """True when every REQUIRED topic is present and on rate. An optional topic that is missing
        is information, not a failure: `/f1sim/plan` is absent for every baseline node by design."""
        return all(t.status == OK for t in self.topics if t.required)

    @property
    def missing_required(self) -> List[str]:
        return [t.name for t in self.topics if t.required and t.status == MISSING]

    def text(self) -> str:
        head = (f"system check: {self.source or 'live'}  {self.duration_s:.1f} s  "
                f"{'OK' if self.ok else 'NOT READY'}")
        lines = [head] + [t.line() for t in self.topics]
        if self.versions:
            lines.append("  in force: " + "  ".join(f"{k}={v}" for k, v in sorted(self.versions.items())
                                                    if v not in ("", None)))
        if self.missing_required:
            lines.append("  MISSING REQUIRED: " + ", ".join(self.missing_required))
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "source": self.source, "duration_s": self.duration_s,
                "versions": dict(self.versions),
                "topics": [{"name": t.name, "status": t.status, "required": t.required,
                            "count": t.count, "rate_hz": t.rate_hz, "expect_hz": t.expect_hz,
                            "age_s": t.age_s, "max_gap_s": t.max_gap_s, "note": t.note}
                           for t in self.topics]}


def check_timeline(stamps: Dict[str, "np.ndarray | list"], profile, *, duration_s: float = 0.0,
                   now: Optional[float] = None, rate_tolerance: float = 0.5,
                   stale_after: float = 0.5, versions: Optional[dict] = None,
                   source: str = "") -> SystemReport:
    """`{topic: arrival times [s]}` against a `RecordProfile`.

    `duration_s` is the window the counts were taken over; the rate is counted over the span the
    messages actually cover when that is longer than nothing, because a topic that started late has
    a rate over its own life and not over the whole window -- the alternative reports every latched
    or late-joining topic as slow.

    `now` is the end of the window (the last stamp when it is not given), so `age_s` is the
    staleness a live checker would see.
    """
    rep = SystemReport(duration_s=float(duration_s), versions=dict(versions or {}), source=source)
    ends = [float(np.asarray(v)[-1]) for v in stamps.values() if len(v)]
    if now is None:
        now = max(ends) if ends else 0.0
    for spec in profile.topics:
        t = np.asarray(stamps.get(spec.name, ()), dtype=float)
        c = TopicCheck(name=spec.name, status=OK, required=bool(spec.required), count=int(t.size),
                       expect_hz=float(spec.rate_hz))
        if t.size == 0:
            c.status = MISSING
            c.note = "no messages" + ("" if spec.required else " (optional)")
            if spec.sim_only:
                c.note = "no messages (simulator-only topic)"
            rep.topics.append(c)
            continue
        span = float(t[-1] - t[0])
        c.age_s = float(now - t[-1])
        if t.size > 1 and span > 0:
            c.rate_hz = (t.size - 1) / span
            c.max_gap_s = float(np.diff(t).max())
        if spec.rate_hz <= 0:
            # A latched or event topic (`/tf_static`, `/f1sim/reset`): it is present, and "when did
            # it last fire" is not a health question about it. Silence is its normal state, and a
            # "rate" over two latched messages a microsecond apart is a number with no meaning.
            c.rate_hz = 0.0
            c.note = "present"
        elif c.age_s > stale_after:
            c.status = WARN
            c.note = f"stale: last message {c.age_s:.2f} s ago"
        elif t.size > 1 and c.rate_hz < spec.rate_hz * rate_tolerance:
            c.status = WARN
            c.note = f"slow: {c.rate_hz:.1f} Hz against {spec.rate_hz:.0f} Hz"
        elif t.size == 1:
            c.status = WARN
            c.note = "one message only"
        rep.topics.append(c)
    return rep


def check_bag(path: str, profile=None, **kw) -> SystemReport:
    """The same check over a finished rosbag2. Reads timestamps only -- no message is deserialised
    except the two that say what was driving."""
    import rosbag2_py
    from f1sim_ros.record_profile import load as load_profile

    profile = profile or load_profile()
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = {t.name for t in profile.topics}
    stamps = {k: [] for k in wanted if k in types}
    raw_last = {}
    origin = None
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        stamp = int(stamp)
        origin = stamp if origin is None else min(origin, stamp)
        if topic in stamps:
            stamps[topic].append(stamp)
            raw_last[topic] = raw
    out = {k: (np.asarray(v, dtype=np.int64) - np.int64(origin)).astype(float) * 1e-9
           for k, v in stamps.items() if v}
    duration = max((float(v[-1]) for v in out.values()), default=0.0)
    return check_timeline(out, profile, duration_s=duration, source=path,
                          versions=_versions_from_bag(raw_last, types), **kw)


def _versions_from_bag(raw_last: dict, types: dict) -> dict:
    """The checkpoint / arm / controller versions a bag says were in force.

    From the last `/f1sim/policy_state` and `/f1sim/controller/diag` in the bag. A bag without
    them -- a baseline run, a raw car recording -- simply reports nothing, which is the honest
    answer rather than a guess from the topic list.
    """
    out = {}
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception:
        return out
    raw = raw_last.get("/f1sim/policy_state")
    if raw is not None:
        try:
            m = deserialize_message(raw, get_message(types["/f1sim/policy_state"]))
            out["checkpoint"] = str(m.checkpoint)
            out["memory"] = str(m.memory_kind) or "feedforward"
            out["memory_clears"] = str(int(m.memory_clears))
        except Exception:
            pass
    raw = raw_last.get("/f1sim/plan")
    if raw is not None and "checkpoint" not in out:
        # The plan carries the checkpoint on every message, which is the whole reason it is on
        # there: a bag with no `/f1sim/policy_state` -- one recorded with a narrower profile, or by
        # a `ros2 bag record` line somebody typed -- still says which weights drove the car.
        try:
            m = deserialize_message(raw, get_message(types["/f1sim/plan"]))
            out["checkpoint"] = str(m.checkpoint)
        except Exception:
            pass
    raw = raw_last.get("/f1sim/controller/diag")
    if raw is not None:
        try:
            m = deserialize_message(raw, get_message(types["/f1sim/controller/diag"]))
            for st in m.status:
                for kv in st.values:
                    if kv.key in ("arm", "mu", "traction", "traction_state",
                                  "clearance_margin_m", "speed_cap_mps"):
                        out[kv.key] = str(kv.value)
        except Exception:
            pass
    return out


def main():
    import argparse
    import json
    from f1sim_ros.record_profile import load as load_profile
    ap = argparse.ArgumentParser(description="system check over a finished rosbag2")
    ap.add_argument("bag")
    ap.add_argument("--profile", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--stale-after", type=float, default=0.5)
    a = ap.parse_args()
    rep = check_bag(a.bag, load_profile(a.profile), stale_after=a.stale_after)
    print(json.dumps(rep.as_dict(), indent=2) if a.json else rep.text())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
