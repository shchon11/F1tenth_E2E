"""Published F1TENTH end-to-end baselines, run on this project's sensors.

`load(kind, weights, **opts)` is the only entry point. Everything downstream -- the ROS
`baseline_node`, the benchmark's `external` roster kind, the budget line, the parity tests -- goes
through it, so there is exactly one place where a baseline's identity is decided.

The vendored upstream checkouts live under `external/baselines/` and are read-only:

  TinyLidarNet   CSL-KU/TinyLidarNet @ 516800231b935673e034156e6c8bfee49045a52c (2025-05-15)
                 NO LICENCE FILE -- evaluation and citation only.
  End2Race       michigan-traffic-lab/End2Race @ c563dd07a8cefc978f88a09def82153469b6879c
                 (2025-11-12), MIT.

The reactive classics (`reactive.py`, 2026-10-17) need no weights: `ftg` is Follow-the-Gap (the
RoboRacer course's reactive method; ForzaETH's implementation, race_stack @ e7d07f7, MIT, vendored
under `external/baselines/forzaeth_ftg/`), `disparity` the Disparity Extender built from its authors'
description. Their `weights` argument names nothing.

Every kind also takes `speed_scale` (default 1): a multiplier on the speed it commands. It is the pace
knob a team turns on a network that outputs speed, and the one a fair comparison tunes for the
networks here (2026-10-18, the user: a baseline needing a speed limit must be tuned, not run as is).
"""
from __future__ import annotations

from .common import (BaselineDriver, BaselineError, OUR_FOV, OUR_N_BEAMS, ScanContract,  # noqa: F401
                     map_scan, scan_is_identity, unseen_fraction)

KINDS = ("tinylidarnet", "end2race", "ftg", "disparity")


def load(kind: str, weights: str, **opts) -> BaselineDriver:
    """One baseline, ready to drive. `opts` are the driver's own, documented in its module, plus
    `speed_scale` for every kind."""
    scale = float(opts.pop("speed_scale", 1.0))
    drv = _load(kind, weights, **opts)
    drv.speed_scale = scale
    return drv


def _load(kind: str, weights: str, **opts) -> BaselineDriver:
    k = str(kind or "").strip().lower()
    if k == "tinylidarnet":
        from . import tinylidarnet
        return tinylidarnet.load(weights, **opts)
    if k == "end2race":
        from . import end2race
        return end2race.load(weights, **opts)
    if k == "ftg":
        from . import reactive
        return reactive.load_ftg(weights, **opts)
    if k == "disparity":
        from . import reactive
        return reactive.load_disparity(weights, **opts)
    raise BaselineError(f"unknown baseline kind {kind!r}; known: {', '.join(KINDS)}")
