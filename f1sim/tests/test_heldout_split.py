"""The held-out split has to stay held out.

Benchmark suite v1 scored `gen:control:1400` (a training map) and `real:korea_2026_competition`
(trained through twenty obstacle variants), so its numbers were in-distribution and no
generalisation figure existed. v2 scores `HELDOUT_TRACKS`, and that is only worth anything while
nothing in `TRAIN_TRACKS` touches one of those floors.

"Touches" is the part that needs a test rather than a convention. `real:map16x07+obs5~mir` is a
different string from `real:map16x07`, so a `in` check on names passes while the policy trains on
the exact geometry the benchmark then calls unseen. Adding one line to a training list is the
cheapest edit in the repository and the one that silently destroys the only number here that
answers "does it generalise".
"""
import pytest

from f1sim import maps
from f1sim.learn import common


def test_the_current_split_has_no_leakage():
    offenders = common.heldout_leakage(common.TRAIN_TRACKS,
                                       common.HELDOUT_TRACKS + common.HELDOUT_OBSTACLE_TRACKS)
    assert offenders == [], (
        "training entries stand on a held-out floor: " + ", ".join(offenders))


@pytest.mark.parametrize("entry", [
    "real:map16x07",
    "real:map16x07~rev",
    "real:map16x07~mir",
    "real:map16x07~mir~rev",
    "real:map16x07+obs5",
    "real:map16x07+obs5~mir",
    "real:map16x07+rlobs5~rev",
    "real:map16x07+pinch5",
    "real:map16x07+props5~rev",
    "real:map12x16+rlobs9~mir~rev",
    "real:blackbox2022_3+obs1",
    "gen:control:9100+pinch2~rev",
])
def test_a_variant_of_a_held_out_map_in_training_is_caught(entry):
    """Every suffix the catalog understands, not only the ones a training list happens to use."""
    train = list(common.TRAIN_TRACKS) + [entry]
    assert common.heldout_leakage(train, common.HELDOUT_TRACKS
                                  + common.HELDOUT_OBSTACLE_TRACKS) == [entry]


def test_the_guard_covers_every_variant_suffix_the_catalog_supports():
    """A suffix family added to `maps.py` must not walk past the guard unnoticed.

    `heldout_leakage` strips suffixes through `maps.MODIFIERS` and `maps._split_obstacle_suffix`
    rather than a private list, so this asserts the coupling holds rather than restating it.
    """
    for mod in maps.MODIFIERS:
        assert common.base_map(f"real:map16x07{mod}") == "real:map16x07", mod
    for tag in ("+obs", "+rlobs", "+pinch", "+props"):
        assert common.base_map(f"real:map16x07{tag}7") == "real:map16x07", tag
        assert common.base_map(f"real:map16x07{tag}7~rev") == "real:map16x07", tag


def test_korea_2026_is_training_only():
    """The decision that makes v2 mean anything: it is a training venue and nothing else."""
    assert any(n.startswith(common.KOREA26) for n in common.TRAIN_TRACKS)
    held = common.HELDOUT_TRACKS + common.HELDOUT_OBSTACLE_TRACKS
    assert [n for n in held if common.base_map(n) == common.KOREA26] == []


def test_the_two_real_floors_are_held_out_in_both_directions():
    import os
    for base in common.HELDOUT_BASE_MAPS:
        assert base in common.HELDOUT_TRACKS
        assert f"{base}~rev" in common.HELDOUT_TRACKS
        name = base.split(":", 1)[1]
        assert name in maps.REAL, f"{base} is held out but not in the catalog"
        yaml_path, boundary = maps.REAL[name][:2]
        assert os.path.exists(yaml_path), yaml_path
        assert boundary == "duct"      # the recordings' evidence; see docs/benchmark.md


def test_eval_tracks_is_the_held_out_list_itself():
    """An alias, not a copy: two lists drift, and the drift is invisible until a number is wrong."""
    assert common.EVAL_TRACKS is common.HELDOUT_TRACKS
    assert common.EVAL_OBSTACLE_TRACKS is common.HELDOUT_OBSTACLE_TRACKS
    assert common.track_names("eval") == common.track_names("heldout") == list(common.HELDOUT_TRACKS)


def test_base_map_leaves_a_plain_catalog_name_alone():
    for n in ("real:map16x07", "rt:Monza", "gen:competition:0", "gym:levine"):
        assert common.base_map(n) == n
