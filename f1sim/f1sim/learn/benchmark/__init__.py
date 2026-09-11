"""Checkpoint benchmark v1.

Portable by construction: imports only the standard library, torch/numpy and `f1sim.*`. The
contracts it depends on -- seeded reset order, pre-autoreset boundary capture, the static-mu guard,
strict `allow_controller` loading -- are implemented here rather than imported from an experiment's
working directory, so the package moves as a unit.

Nothing here mutates `f1sim`: no `__init__` edit, no `__path__` manipulation.
"""
from __future__ import annotations

SUITE_VERSION = "v1"

__all__ = ["SUITE_VERSION", "geom", "overtake", "routed_tracker", "tally", "roster"]
