"""Test segments marked during recording: /sysid/segment (std_msgs/String), "T3 start" ... "T3 end".

`parse` turns the messages into intervals per test name; `mask` turns a test's intervals into a boolean mask on the
100 Hz grid. A measurement that belongs to a test is computed only inside that test's intervals when the bag has
them; otherwise the tool finds the stretches itself (the rules in docs/bag_recording_guide.md section 4) -- and the
summary says which of the two was used for every measurement.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import thresholds as TH

_LABEL = re.compile(r"^\s*([A-Za-z]\w*)\s+(start|begin|end|stop)\s*$", re.IGNORECASE)


def parse(events: Sequence[Tuple[float, str]], t_end: float) -> Tuple[Dict[str, List[Tuple[float, float]]], List[str]]:
    """events: (time [s], label) -> ({test: [(t0, t1), ...]}, warnings). A start with no end runs to `t_end`; an end
    with no start and a label that is not '<name> start|end' are reported and ignored."""
    ints: Dict[str, List[Tuple[float, float]]] = {}
    open_at: Dict[str, float] = {}
    warn: List[str] = []
    last: Dict[Tuple[str, str], float] = {}
    for t, lab in sorted(events):
        m = _LABEL.match(lab or "")
        if not m:
            warn.append(f"{t:.1f} s: 알 수 없는 표시 '{lab}' (형식: 'T3 start' / 'T3 end')")
            continue
        name, kind = m.group(1).upper(), m.group(2).lower()
        kind = "start" if kind in ("start", "begin") else "end"
        if (name, kind) in last and t - last[(name, kind)] < TH.SEGMENT_DEDUP_S:
            last[(name, kind)] = t
            continue                      # the same mark published again within SEGMENT_DEDUP_S: one mark
        last[(name, kind)] = t
        if kind == "start":
            if name in open_at:
                warn.append(f"{t:.1f} s: {name} start 가 두 번 (앞의 것을 끝으로 닫음)")
                ints.setdefault(name, []).append((open_at[name], t))
            open_at[name] = t
        else:
            if name not in open_at:
                warn.append(f"{t:.1f} s: {name} end 에 짝이 되는 start 없음")
                continue
            ints.setdefault(name, []).append((open_at.pop(name), t))
    for name, t0 in open_at.items():
        warn.append(f"{name} start({t0:.1f} s) 에 end 가 없어 bag 끝까지로 침")
        ints.setdefault(name, []).append((t0, t_end))
    return ints, warn


def mask(t: np.ndarray, ints: Dict[str, List[Tuple[float, float]]], tests: Sequence[str]) -> Optional[np.ndarray]:
    """True on the grid inside any interval of any of `tests`; None when none of them was marked."""
    spans = [iv for name in tests for iv in ints.get(name.upper(), [])]
    if not spans:
        return None
    m = np.zeros(len(t), bool)
    for a, b in spans:
        m |= (t >= a) & (t <= b)
    return m


def describe(ints: Dict[str, List[Tuple[float, float]]], tests: Sequence[str]) -> str:
    spans = [iv for name in tests for iv in ints.get(name.upper(), [])]
    return f"{len(spans)}개 구간, {sum(b - a for a, b in spans):.0f} s"
