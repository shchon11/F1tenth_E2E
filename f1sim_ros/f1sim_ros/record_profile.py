"""`config/record.yaml` as an object: the topics the graph needs, in one place.

ROS-free apart from the YAML read, so `system_check`'s offline half and the tests can use it
without a graph. The default location is the installed share directory when there is one and this
checkout's `config/` when there is not, so a worktree that has never been `colcon build`-ed still
reads its own profile rather than another one's.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class TopicSpec:
    name: str
    type: str
    rate_hz: float = 0.0
    required: bool = False
    sim_only: bool = False
    why: str = ""


@dataclass(frozen=True)
class RecordProfile:
    version: int
    topics: tuple
    path: str = ""

    @property
    def names(self) -> List[str]:
        return [t.name for t in self.topics]

    @property
    def required(self) -> List[str]:
        return [t.name for t in self.topics if t.required]

    def get(self, name: str) -> Optional[TopicSpec]:
        for t in self.topics:
            if t.name == name:
                return t
        return None

    def on_car(self) -> "RecordProfile":
        """The same profile without the simulator-only topics. What `graph_car` records."""
        return RecordProfile(self.version, tuple(t for t in self.topics if not t.sim_only),
                             self.path)


def default_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("f1sim_ros"), "config", "record.yaml")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                        "record.yaml")


def load(path: str = "") -> RecordProfile:
    path = path or default_path()
    with open(path) as fh:
        d = yaml.safe_load(fh) or {}
    topics = []
    for raw in d.get("topics", ()):
        known = {k: raw[k] for k in ("name", "type", "rate_hz", "required", "sim_only", "why")
                 if k in raw}
        unknown = sorted(set(raw) - set(TopicSpec.__dataclass_fields__))
        if unknown:
            raise ValueError(f"{path}: topic {raw.get('name')!r} has unknown key(s) {unknown}")
        if "name" not in known or "type" not in known:
            raise ValueError(f"{path}: every topic needs a name and a type, got {raw!r}")
        topics.append(TopicSpec(**known))
    if not topics:
        raise ValueError(f"{path}: a record profile with no topics records nothing")
    dup = sorted({t.name for t in topics if [x.name for x in topics].count(t.name) > 1})
    if dup:
        raise ValueError(f"{path}: {dup} listed more than once")
    return RecordProfile(int(d.get("version", 1)), tuple(topics), path)
