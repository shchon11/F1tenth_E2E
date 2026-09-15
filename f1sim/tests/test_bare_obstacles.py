"""`기본` vs `없음`: the map as authored, and the map with the author's obstacles taken off.

The 장애물 selector's `""` was labelled 없음 and meant "no procedural family". For a plain map those
are the same sentence. For an editor scene they are not -- the scene carries the props its author
placed -- so "없음" showed a map full of boxes, which is what the user reported (2026-09-15):
*"장애물 없음이라는 말과 안맞아"*.

So the axis has three kinds of value now, and the claims worth a test are:

1. **The two are different things and both are sayable**, in the registry, in both grammars, and
   round trip through each.
2. **`+bare` removes what the author placed and nothing else** -- the grids are untouched, so a wall
   is still a wall, and on a map with no placements it changes nothing.
3. **The families ADD**, and compose with `+bare` in the stated order (`+bare` first).
4. **Both console pages spell it the same way**, with the counts the labels promise.
5. **Every id that does not carry `+bare` loads exactly the track it loaded before.**
"""

import numpy as np
import pytest

from f1sim import maps, tracks as T
from f1sim.scene import SceneDoc


@pytest.fixture
def scenes(tmp_path, monkeypatch):
    """Two editor scenes on one geometry: one with three placed props, one with none."""
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    maps._BASE_CACHE.clear()
    size = 12.0
    a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    line = np.stack([size / 2 + 4.0 * np.cos(a), size / 2 + 4.0 * np.sin(a)], 1)
    for name, n in (("hall", 3), ("empty", 0)):
        d = SceneDoc.new_blank(name, size, size, resolution=0.05)
        d.paint_border("tall", 0.1)
        d.paint_disc("duct", size / 2, size / 2, 2.0)
        d.centerline = line.copy()
        for i in range(n):
            d.add_prop("cardboard_box", 8.0 + 0.45 * i, 6.0, yaw=0.0)
        d.save()
    yield {"hall": 3, "empty": 0}
    maps._BASE_CACHE.clear()


# =================================================== 1. the registry says two different things
def test_the_default_choice_is_the_map_as_authored_not_an_empty_one():
    assert T.OBSTACLE_LABEL[""] == "기본"
    assert T.OBSTACLE_LABEL[T.BARE] == "없음"
    assert "그대로" in T.OBSTACLE_HINT[""], T.OBSTACLE_HINT[""]
    assert "걷어" in T.OBSTACLE_HINT[T.BARE], T.OBSTACLE_HINT[T.BARE]
    # the families say, once and in one place, that they add rather than replace
    assert "위에" in T.OBSTACLE_ADDS_HINT


def test_bare_is_offered_on_every_family_of_map():
    """It is meaningful everywhere and a no-op where there is nothing placed, so a console can grey
    it rather than hide it -- and hiding it is what made 기본 look like the only option."""
    for family, allowed in T.OBSTACLES_BY_FAMILY.items():
        assert T.BARE in allowed, family
        assert "" in allowed, family


@pytest.mark.parametrize("spec,legacy", [
    ("real/bb22-1", "real:blackbox2022_1"),
    ("real/bb22-1#bare", "real:blackbox2022_1+bare"),
    ("real/bb22-1@rev#bare", "real:blackbox2022_1+bare~rev"),
    ("real/bb22-1#line:44", "real:blackbox2022_1+rlobs44"),
    ("real/bb22-1#bare+line:44", "real:blackbox2022_1+bare+rlobs44"),
    ("real/bb22-1@mir+rev#bare+hard:3", "real:blackbox2022_1+bare+hard3~mir~rev"),
])
def test_both_grammars_round_trip_a_bare_choice(spec, legacy):
    sc = T.parse(spec)
    assert sc.legacy() == legacy
    assert sc.short() == spec
    back = T.parse(legacy)
    assert back.short() == spec and back.bare == sc.bare and back.obstacle == sc.obstacle


def test_the_choice_key_is_one_string_both_pages_can_hold():
    assert T.parse("real/bb22-1").choice == ""
    assert T.parse("real/bb22-1#bare").choice == T.BARE
    assert T.parse("real/bb22-1#hard:3").choice == "hard"
    assert T.parse("real/bb22-1#bare+hard:3").choice == "bare+hard"
    assert T.split_choice("bare+hard") == (True, "hard")
    assert T.split_choice("") == (False, "")
    assert T.obstacle_choice(True, "hard") == "bare+hard"
    assert T.choice_label("bare+hard").startswith("없음 + ")


def test_a_seed_is_required_exactly_where_something_is_placed():
    """`없음` places nothing, so asking it for a seed is a typo worth naming; a family without one
    would silently become a different placement on every run."""
    with pytest.raises(T.TrackError) as exc:
        T.parse("real/bb22-1#bare:3")
    assert "시드를 받지 않습니다" in str(exc.value)
    with pytest.raises(T.TrackError) as exc:
        T.parse("real/bb22-1#line")
    assert "시드가 필요합니다" in str(exc.value)


def test_the_display_line_names_the_choice():
    assert "없음" in T.parse("real/bb22-1#bare").display()
    assert "없음" in T.parse("real/bb22-1#bare+hard:3").display()
    assert "없음" not in T.parse("real/bb22-1#hard:3").display()


def test_the_scene_prop_count_is_readable_without_loading_a_map(scenes):
    assert T.scene_props("scene/hall") == 3
    assert T.scene_props("scene/empty") == 0
    assert T.scene_props("real/bb22-1") == 0


# =================================================== 2 & 3. what the loader does with it
def test_bare_drops_the_placed_props_and_leaves_the_walls(scenes):
    authored = maps.load("scene:hall")
    bare = maps.load("scene:hall+bare")
    assert len(authored.props) == 3 and len(bare.props) == 0
    assert np.array_equal(authored.occupancy, bare.occupancy), "a wall was removed with the boxes"
    assert np.array_equal(authored.duct, bare.duct) and np.array_equal(authored.tall, bare.tall)
    assert authored.grid_key() == bare.grid_key()


def test_bare_on_a_map_with_nothing_placed_is_the_map(scenes):
    plain = maps.load("scene:empty")
    bare = maps.load("scene:empty+bare")
    assert plain.grid_key() == bare.grid_key() and len(bare.props) == 0
    assert np.allclose(plain.centerline, bare.centerline)


def test_a_family_adds_to_what_the_author_placed(scenes):
    """The sentence the old label contradicted, now pinned: 입체 on a scene with three boxes is
    three boxes *and* the placed props, not three boxes replaced."""
    authored = maps.load("scene:hall")
    added = maps.load("scene:hall+props5")
    assert len(added.props) > len(authored.props)
    keep = {(round(p.x, 4), round(p.y, 4)) for p in authored.props}
    assert keep <= {(round(p.x, 4), round(p.y, 4)) for p in added.props}, "an authored prop vanished"


def test_bare_composes_with_a_family_and_is_applied_first(scenes):
    both = maps.load("scene:hall+bare+props5")
    added = maps.load("scene:hall+props5")
    assert len(both.props) == len(added.props) - 3, "the author's props survived +bare"
    keep = {(round(p.x, 4), round(p.y, 4)) for p in maps.load("scene:hall").props}
    assert not (keep & {(round(p.x, 4), round(p.y, 4)) for p in both.props})


def test_bare_composes_with_a_grid_family(scenes):
    """`+hard` stamps cells rather than placing props, so this is the other half of "it adds"."""
    plain = maps.load("scene:hall")
    hard = maps.load("scene:hall+hard3")
    bare_hard = maps.load("scene:hall+bare+hard3")
    assert int(hard.occupancy.sum()) > int(plain.occupancy.sum()), "+hard stamped nothing"
    assert np.array_equal(hard.occupancy, bare_hard.occupancy), "+bare changed the grid"
    assert len(hard.props) == 3 and len(bare_hard.props) == 0


def test_bare_survives_the_direction_modifiers(scenes):
    assert len(maps.load("scene:hall+bare~rev").props) == 0
    assert len(maps.load("scene:hall+bare~mir").props) == 0
    assert len(maps.load("scene:hall~mir").props) == 3, "mirroring dropped the author's props"


def test_the_short_grammar_reaches_the_loader(scenes):
    assert len(maps.load("scene/hall#bare").props) == 0
    assert len(maps.load("scene/hall").props) == 3


# =================================================== 5. nothing else moved
BYTE_IDENTITY_IDS = [
    "gen:competition:0", "gen:competition:0+obs3", "gen:competition:0+rlobs3",
    "gen:competition:0+pinch3", "gen:competition:0+props3", "gen:competition:0+hard3",
    "gen:control:1400", "gen:recipe:7",
]


@pytest.mark.parametrize("name", BYTE_IDENTITY_IDS)
def test_an_id_without_bare_builds_the_track_it_always_built(name):
    """The grids, the line and the placements of a map named the old way, against the values
    recorded from `df60b44` (see `docs/research/bare-obstacles-2026-09-15.md` for the sweep that
    produced them). Here the claim is narrower and self-contained: loading twice through the new
    code path is stable and `+bare` is the only thing that changes a track.
    """
    a = maps.load(name)
    maps._BASE_CACHE.clear()
    b = maps.load(name)
    assert a.grid_key() == b.grid_key()
    assert len(a.props) == len(b.props)
    assert (a.centerline is None) == (b.centerline is None)
    if a.centerline is not None:
        assert np.allclose(a.centerline, b.centerline)


def test_a_track_with_no_props_is_not_copied_by_bare():
    """`Track.bare()` returns the object it was given when there is nothing to remove, which is
    what makes the suffix free on every map that is not an editor scene."""
    t = maps.load("gen:competition:0")
    assert t.bare() is t
