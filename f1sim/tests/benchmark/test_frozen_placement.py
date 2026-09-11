"""The frozen placement must be enforced in full, not just in the inputs it was rebuilt from.

Native's repro: mutate the recorded `size_m` to [100, 100] in memory. The reconstruction still
returned an ordinary small box and was accepted, because nothing compared the regenerated placement
to the record beyond `s_obs` and `side`.
"""
from __future__ import annotations
import copy
import importlib
import json
import os
import sys

import pytest

pytest.importorskip("f1sim.track")

# The example suite ships with the package, so the test finds it through the import rather than a
# path relative to the test tree.
import f1sim.learn.benchmark as _bench
SUITE = os.path.join(os.path.dirname(os.path.abspath(_bench.__file__)), "suite-v1.example.json")


@pytest.fixture
def mm(bench):
    return importlib.import_module("f1sim.learn.benchmark.__main__")


@pytest.fixture
def su(bench):
    return importlib.import_module("f1sim.learn.benchmark.suite")


@pytest.fixture(scope="module")
def frozen_suite():
    if not os.path.exists(SUITE):
        pytest.skip("no frozen candidate suite on this machine")
    return json.load(open(SUITE))


def _suite_with(su, frozen_suite, map_id, **mutate):
    """The frozen suite, with one placement field mutated in memory only."""
    raw = copy.deepcopy(frozen_suite)
    raw["suite"]["placements"][map_id]["placement"].update(mutate)
    s = su.Suite(**{k: (tuple(v) if isinstance(v, list) else v)
                    for k, v in raw["suite"].items()})
    return s


def _cell(su, s, map_id):
    return su.Cell("A", map_id, s.solo_mus[0], s.seeds[0], 2)


def test_the_unmutated_frozen_placement_reproduces(mm, su, frozen_suite):
    """Baseline: without mutation the regeneration must match and be accepted."""
    s = _suite_with(su, frozen_suite, list(frozen_suite["suite"]["placements"])[0])
    map_id = list(s.placements)[0]
    tracks, _rls, s_obs = mm._obstacle_track_for(_cell(su, s, map_id), s)
    assert tracks and s_obs == s.placements[map_id]["placement"]["s_obs_m"]


@pytest.mark.parametrize("field,value", [
    ("size_m", [100.0, 100.0]),                 # native's exact mutation
    ("n_cells", 999999),
    ("centre_xy", [0.0, 0.0]),
    ("half_lane_m", 42.0),
    ("corridor_centre_offset_m", -42.0),
])
def test_a_mutated_frozen_field_is_refused(mm, su, frozen_suite, field, value):
    """Any disagreement between the record and the regeneration must refuse, not be ignored."""
    map_id = list(frozen_suite["suite"]["placements"])[0]
    s = _suite_with(su, frozen_suite, map_id, **{field: value})
    with pytest.raises(SystemExit, match="differs from the freeze|covers .* cells"):
        mm._obstacle_track_for(_cell(su, s, map_id), s)


def test_a_missing_placement_is_refused(mm, su, frozen_suite):
    map_id = list(frozen_suite["suite"]["placements"])[0]
    s = _suite_with(su, frozen_suite, map_id)
    s.placements = {}
    with pytest.raises(SystemExit, match="no frozen placement"):
        mm._obstacle_track_for(_cell(su, s, map_id), s)
