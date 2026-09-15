"""Published F1TENTH end-to-end baselines, run on this project's sensors.

`load(kind, weights, **opts)` is the only entry point. Everything downstream -- the ROS
`baseline_node`, the benchmark's `external` roster kind, the budget line, the parity tests -- goes
through it, so there is exactly one place where a baseline's identity is decided.

The vendored upstream checkouts live under `external/baselines/` and are read-only:

  TinyLidarNet   CSL-KU/TinyLidarNet @ 516800231b935673e034156e6c8bfee49045a52c (2025-05-15)
                 NO LICENCE FILE -- evaluation and citation only.
  End2Race       michigan-traffic-lab/End2Race @ c563dd07a8cefc978f88a09def82153469b6879c
                 (2025-11-12), MIT.
"""
from __future__ import annotations

from .common import (BaselineDriver, BaselineError, OUR_FOV, OUR_N_BEAMS, ScanContract,  # noqa: F401
                     map_scan, scan_is_identity, unseen_fraction)

KINDS = ("tinylidarnet", "end2race")


def load(kind: str, weights: str, **opts) -> BaselineDriver:
    """One baseline, ready to drive. `opts` are the driver's own, documented in its module."""
    k = str(kind or "").strip().lower()
    if k == "tinylidarnet":
        from . import tinylidarnet
        return tinylidarnet.load(weights, **opts)
    if k == "end2race":
        from . import end2race
        return end2race.load(weights, **opts)
    raise BaselineError(f"unknown baseline kind {kind!r}; known: {', '.join(KINDS)}")
