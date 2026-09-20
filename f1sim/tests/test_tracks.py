"""The track registry and the scenario grammar.

Three things are being asserted, and they matter for different reasons.

1. **The splits did not move.** `learn/common.py` generates `TRAIN_TRACKS` and the held-out lists
   from `SplitRule`s now. `tracks_oracle.py` holds the literals from before the change and the
   comparison is string for string, in order. A refactor that changes one name is not a refactor.
2. **Both grammars round-trip.** Every catalogue entry, in every direction, with every obstacle
   family it supports, has to survive `legacy -> parse -> short -> parse -> legacy` unchanged.
   That is what lets a frozen benchmark file, a checkpoint manifest and the new picker name the
   same thing.
3. **The random seed is a draw, not a surprise.** `#line:*` has to give the same maps for the same
   run seed and different ones for different seeds, and it has to stay unable to produce a loader
   name until a seed is actually drawn.
"""
import random

import pytest

from f1sim import maps, tracks
from f1sim.learn import common

from tracks_oracle import (FROZEN_HELDOUT_OBSTACLE_TRACKS, FROZEN_HELDOUT_TRACKS,
                           FROZEN_TRAIN_TRACKS)


# ================================================================ the frozen split oracles
def test_train_tracks_is_exactly_the_frozen_list():
    assert common.TRAIN_TRACKS == FROZEN_TRAIN_TRACKS


def test_heldout_tracks_is_exactly_the_frozen_list():
    assert common.HELDOUT_TRACKS == FROZEN_HELDOUT_TRACKS


def test_heldout_obstacle_tracks_is_exactly_the_frozen_list():
    assert common.HELDOUT_OBSTACLE_TRACKS == FROZEN_HELDOUT_OBSTACLE_TRACKS


def test_track_names_still_answers_every_split_alias_with_the_old_strings():
    assert common.track_names("train") == FROZEN_TRAIN_TRACKS
    assert common.track_names("eval") == common.track_names("heldout") == FROZEN_HELDOUT_TRACKS
    assert (common.track_names("eval_obstacles") == common.track_names("heldout_obstacles")
            == FROZEN_HELDOUT_OBSTACLE_TRACKS)
    assert (common.track_names("eval_all")
            == FROZEN_HELDOUT_TRACKS + FROZEN_HELDOUT_OBSTACLE_TRACKS)


def test_the_derived_constants_other_modules_import_are_unchanged():
    assert common.KOREA26 == "real:korea_2026_competition"
    assert len(common.KOREA26_TRAIN) == 25
    assert common.HELDOUT_BASE_MAPS == ("real:map16x07", "real:map12x16")
    assert common.REAL_TRAIN == ["icra2022", "blackbox2021_1", "blackbox2021_2", "blackbox2021_3",
                                 "blackbox2022_1", "blackbox2022_2"]
    assert common.RT_TRAIN == ["Spielberg", "Oschersleben"]
    assert common.TRAIN_DIRECTIONS == ("", "~rev", "~mir", "~mir~rev")


def test_every_split_entry_names_a_registry_track():
    """No split may contain a name the registry cannot describe: the picker lists base tracks, and
    a variant whose base is unknown is a variant the user cannot reach."""
    for name in FROZEN_TRAIN_TRACKS + FROZEN_HELDOUT_TRACKS + FROZEN_HELDOUT_OBSTACLE_TRACKS:
        sc = tracks.parse(name)
        assert not sc.raw, f"{name} did not resolve to a registry track"
        tracks.get(sc.track)


# ================================================================ the registry
def test_ids_are_short_lowercase_and_unique():
    seen = set()
    for e in tracks.catalog(scenes=False):
        assert e.id not in seen, e.id
        seen.add(e.id)
        family, _, slug = e.id.partition("/")
        assert family in tracks.FAMILIES, e.id
        assert slug and slug == slug.lower(), e.id
        assert len(slug) <= 12, f"{e.id}: slug is {len(slug)} characters"
        assert e.display and e.legacy


def test_real_notes_match_the_loader_entries():
    """The one-line note is `maps.REAL`'s own. Copied here because `tracks` must stay torch-free;
    asserted here so the copy cannot drift."""
    for e in tracks.catalog(scenes=False):
        if e.family != "real":
            continue
        key = e.legacy.split(":", 1)[1]
        assert key in maps.REAL, e.id
        assert e.note == maps.REAL[key][3], e.id


def test_the_registry_covers_every_real_map_in_the_loader():
    ids = {e.legacy for e in tracks.catalog(scenes=False)}
    for key in maps.REAL:
        assert f"real:{key}" in ids, key


def test_an_unknown_track_id_says_what_a_good_one_looks_like():
    with pytest.raises(tracks.TrackError) as exc:
        tracks.get("real/not_a_map")
    assert "real/not_a_map" in str(exc.value)
    with pytest.raises(tracks.TrackError):
        tracks.get("nosuchfamily/x")


# ================================================================ the grammar
def _every_scenario():
    """Every catalogue entry x every direction x every 장애물 *choice* it can carry.

    The choice is what the control holds and what the grammar spells -- `""`, `bare`, a family, or
    `bare+<family>` -- so it is what has to round trip. A choice that places nothing takes no seed.
    """
    for e in tracks.catalog(scenes=False):
        for direction in tracks.DIRECTIONS:
            for choice in e.obstacle_options():
                for key in ({choice} if choice in ("", tracks.BARE)
                            else {choice, f"{tracks.BARE}+{choice}"}):
                    bare, family = tracks.split_choice(key)
                    yield tracks.Scenario(track=e.id, mirror="mir" in direction,
                                          reverse="rev" in direction, obstacle=family,
                                          seed=7 if family else None, bare=bare)


def test_every_catalogue_scenario_round_trips_through_both_grammars():
    for sc in _every_scenario():
        assert tracks.parse(sc.short()) == sc, sc.short()
        assert tracks.parse(sc.legacy()) == sc, sc.legacy()
        assert tracks.resolve(sc.short()) == sc.legacy()
        assert tracks.short(sc.legacy()) == sc.short()


def test_resolve_is_the_identity_on_a_legacy_name():
    for name in FROZEN_TRAIN_TRACKS + FROZEN_HELDOUT_TRACKS + FROZEN_HELDOUT_OBSTACLE_TRACKS:
        assert tracks.resolve(name) == name


@pytest.mark.parametrize("spec,legacy", [
    ("real/bb22-1", "real:blackbox2022_1"),
    ("real/bb22-1@rev", "real:blackbox2022_1~rev"),
    ("real/bb22-1@mir", "real:blackbox2022_1~mir"),
    ("real/bb22-1@mir+rev", "real:blackbox2022_1~mir~rev"),
    ("real/bb22-1@rev#line:44", "real:blackbox2022_1+rlobs44~rev"),
    ("real/korea26#edge:3", "real:korea_2026_competition+obs3"),
    ("real/korea26#pinch:7", "real:korea_2026_competition+pinch7"),
    ("rt/monza#props:5", "rt:Monza+props5"),
    ("rt/spielberg@mir+rev", "rt:Spielberg~mir~rev"),
    ("gen/control-1400@rev", "gen:control:1400~rev"),
    ("gen/comp-1006", "gen:competition:1006"),
    ("gen/hall-1100", "gen:hallway:1100"),
    ("gen/serp-9000", "gen:serpentine:9000"),
    ("real/lab16x07", "real:map16x07"),
    ("real/lab12x16@rev", "real:map12x16~rev"),
    ("real/iccas25", "real:korea_2025_iccas"),
    ("real/icra22", "real:icra2022"),
    ("scene/my_hall@rev", "scene:my_hall~rev"),
])
def test_the_documented_examples_resolve(spec, legacy):
    """The table in `docs/tracks.md`. Every legacy suffix maps onto exactly one new option."""
    assert tracks.resolve(spec) == legacy
    assert tracks.short(legacy) == spec


def test_a_name_outside_the_catalogue_passes_through_untouched():
    for name in ("/abs/path/map.yaml", "some_weird_name", "scene:/abs/dir"):
        sc = tracks.parse(name)
        assert sc.raw == name and sc.legacy() == name and sc.short() == name
        assert sc.track == ""


def test_an_obstacle_family_this_version_does_not_know_still_names_its_map():
    """Another branch is training with `+hard<seed>`. The string is not ours to rebuild -- it comes
    back verbatim -- but the map under it is `real:icra2022`, and a list of such entries has to
    group and count as that map rather than as one new map per entry."""
    sc = tracks.parse("real:icra2022+weird1~rev")
    assert sc.track == "real/icra22"
    assert sc.raw == sc.legacy() == sc.short() == "real:icra2022+weird1~rev"
    assert sc.display() == "ICRA 2022 · weird1~rev"
    assert tracks.resolve("real:icra2022+weird1~rev") == "real:icra2022+weird1~rev"
    # and it must not have been mistaken for one of the families we do know
    assert sc.obstacle == "" and not sc.reverse


def test_a_track_is_not_offered_an_obstacle_it_cannot_carry():
    """`maps._load_base` raises for `rt:x+obs3`; the grammar refuses it before the loader does."""
    with pytest.raises(tracks.TrackError) as exc:
        tracks.parse("rt/monza#edge:3")
    assert "Monza" in str(exc.value)
    # `기본` and `없음` are on every map -- neither adds anything, so neither can be unsupported.
    assert tracks.get("rt/monza").obstacle_options() == ("", "bare", "props", "hard")
    assert tracks.get("real/korea26").obstacle_options() == ("", "bare", "edge", "line", "pinch",
                                                             "props", "hard")


@pytest.mark.parametrize("bad", ["real/bb22-1@sideways", "real/bb22-1#nope:3", "real/bb22-1@rev#line",
                                 "real/bb22-1#line:x"])
def test_a_malformed_spec_is_rejected_with_the_format_in_the_message(bad):
    with pytest.raises(tracks.TrackError) as exc:
        tracks.parse(bad)
    assert bad.split("@")[0].split("#")[0] in str(exc.value) or "형식" in str(exc.value) \
        or "방향" in str(exc.value) or "장애물" in str(exc.value)


def test_maps_load_accepts_the_new_grammar():
    """One map, loaded twice under both spellings: the same object out of the loader's cache."""
    assert maps.load("real/lab16x07") is maps.load("real:map16x07")


# ================================================================ random seeds
def test_a_random_seed_refuses_to_become_a_loader_name():
    sc = tracks.parse("real/bb22-1#line:*")
    assert sc.random_seed and sc.short() == "real/bb22-1#line:*"
    with pytest.raises(tracks.TrackError) as exc:
        sc.legacy()
    assert "무작위" in str(exc.value)
    assert sc.with_seed(44).legacy() == "real:blackbox2022_1+rlobs44"


def test_expansion_is_deterministic_in_the_run_seed():
    a = tracks.expand("real/bb22-1#line:*", 8, random.Random(701))
    b = tracks.expand("real/bb22-1#line:*", 8, random.Random(701))
    c = tracks.expand("real/bb22-1#line:*", 8, random.Random(702))
    assert a == b
    assert a != c
    assert len(set(a)) == 8
    assert all(n.startswith("real:blackbox2022_1+rlobs") for n in a)


def test_a_fixed_seed_is_one_track_however_many_draws_are_asked_for():
    assert tracks.expand("real/bb22-1#line:44", 8) == ["real:blackbox2022_1+rlobs44"]
    assert tracks.expand("real/bb22-1@rev", 8) == ["real:blackbox2022_1~rev"]


def test_track_names_expands_a_random_seed_and_leaves_everything_else_alone():
    got = common.track_names("real/bb22-1#line:*,rt/monza@rev,real:icra2022+obs3", draws=3, seed=5)
    assert got[3:] == ["rt:Monza~rev", "real:icra2022+obs3"]
    assert len(got[:3]) == len(set(got[:3])) == 3
    assert common.track_names("real/bb22-1#line:*", draws=3, seed=5)[:3] == got[:3], \
        "the draw must depend on the run seed and the list order, nothing else"


# ================================================================ the three groups
def test_the_ui_has_exactly_three_groups_and_each_says_what_it_means():
    assert tracks.GROUP_ORDER == ("학습", "검증", "내 환경")
    for g in tracks.GROUP_ORDER:
        assert len(tracks.GROUP_HINT[g]) > 20


def test_the_groups_are_base_tracks_not_variants():
    g = tracks.groups(scene_ids=["scene/mine"])
    assert list(g) == ["학습", "검증", "내 환경"]
    for name, ids in g.items():
        assert ids == list(dict.fromkeys(ids)), name
        for tid in ids:
            assert "@" not in tid and "#" not in tid, tid


def test_the_two_splits_do_not_share_a_base_track():
    """The whole point of the split, expressed over base tracks rather than strings."""
    train = set(tracks.split_tracks("train"))
    held = set(tracks.split_tracks("heldout")) | set(tracks.split_tracks("heldout_obstacles"))
    assert train & held == set()
    assert tracks.group_of("real/korea26") == "학습"
    assert tracks.group_of("real/lab16x07") == "검증"
    assert tracks.group_of("scene/anything") == "내 환경"


def test_split_summary_reports_the_policy_the_rules_actually_use():
    s = tracks.split_summary("train")
    assert s["n_variants"] == len(FROZEN_TRAIN_TRACKS)
    assert s["n_tracks"] == len(set(tracks.split_tracks("train")))
    assert s["directions"] == ["", "rev", "mir", "mir+rev"]
    assert set(s["obstacles"]) == {"", "pinch", "edge", "line"}
    assert "Blackbox" in " ".join(s["displays"])
    h = tracks.split_summary("heldout")
    assert h["n_variants"] == len(FROZEN_HELDOUT_TRACKS)
    assert h["group"] == "검증" and "일반화" in h["hint"]
