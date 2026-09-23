"""The driving page's map card: a list of maps, and three controls that build the scenario.

The complaint this answers, in the user's words: "맵 목록에서도 너무 찾기 불편해. 맵들 이름 너무
난잡하고 추잡스러워 … 장애물같은경우도 그 맵 고르면 그냥 랜덤시드로 바로 실행되게끔 하면 되잖아."

So the assertions are about exactly that: the rows are maps (not maps × directions × obstacle
seeds), the names are readable, choosing a map is enough to drive it, and the obstacle seed defaults
to 무작위 and ends up visible as a number.
"""
import os

import pytest
from PyQt5 import QtCore, QtWidgets

from f1sim import tracks
from f1sim.viewer.console.catalog import MapCatalog
from f1sim.viewer.console.protocol import STATE_RUNNING


@pytest.fixture(scope="module")
def qapp():
    _prev = os.environ.get("QT_QPA_PLATFORM")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from f1sim.viewer.console import app as A
    yield QtWidgets.QApplication.instance() or A.create_app(["test"])
    if _prev is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = _prev


def _catalog(scene_ids=()):
    groups = tracks.groups(scene_ids=list(scene_ids))
    entries = {e.id: {"id": e.id, "family": e.family, "family_label": e.family_label,
                      "display": e.display, "legacy": e.legacy, "note": e.note,
                      "obstacles": list(e.obstacle_options())}
               for e in tracks.catalog(scenes=False,
                                       extra=[t for ids in groups.values() for t in ids])}
    return MapCatalog(groups=groups, entries=entries, ready=True)


@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    from f1sim.viewer.console import window as W
    w = W.ConsoleWindow()
    w.set_maps(_catalog())
    try:
        yield w
    finally:
        w.editor.shutdown()
        w.deleteLater()


def _list_keys(win):
    out, it = [], QtWidgets.QTreeWidgetItemIterator(win.map_list.tree)
    while it.value():
        key = it.value().data(0, QtCore.Qt.UserRole)
        if key is not None:
            out.append(key)
        it += 1
    return out


# ================================================================ the list
def test_the_group_box_offers_exactly_the_three_splits(win, monkeypatch):
    assert [win.map_group.itemData(i) for i in range(win.map_group.count())] == ["학습", "검증"]
    # 내 환경 appears when there is something in it, and is read from the scenes folder rather than
    # from the worker: the editor can save one while the worker is already up.
    monkeypatch.setattr("f1sim.viewer.console.catalog.scene_ids", lambda: ["scene/mine"])
    win.set_maps(_catalog())
    assert [win.map_group.itemData(i) for i in range(win.map_group.count())] == \
        ["학습", "검증", "내 환경"]


def test_the_rows_are_base_maps_with_a_readable_name_and_the_id_beside_it(win):
    keys = _list_keys(win)
    assert keys and all("@" not in k and "#" not in k for k in keys)
    assert len(keys) == len(set(keys)), "a map appears once, not once per variant"
    # `real:korea_2026_competition+rlobs213~mir~rev` and its nineteen siblings are one row now.
    assert keys.count("real/korea26") == 1
    it = win.map_list._find("real/korea26")
    assert it.text(0) == "Korea 2026"
    assert it.text(1) == "real/korea26 · 실측"


def test_the_training_group_is_forty_odd_maps_not_a_hundred_and_forty_nine_strings(win):
    from f1sim.learn import common
    assert len(_list_keys(win)) == len(tracks.split_tracks("train"))
    assert len(common.TRAIN_TRACKS) == 149, "the training set itself is unchanged"
    assert len(_list_keys(win)) < 60


def test_typing_searches_the_whole_catalogue_not_only_the_selected_group(win):
    """Three groups is the right number of groups and the wrong number of places to look."""
    assert "rt/silverstone" not in _list_keys(win)
    win.map_list.search.setText("silverstone")
    keys = _list_keys(win)
    assert keys == ["rt/silverstone"]
    win.map_list.search.setText("")
    assert "rt/silverstone" not in _list_keys(win)


def test_searching_by_the_readable_name_works_too(win):
    win.map_list.search.setText("Blackbox 2022 #3")
    assert _list_keys(win) == ["real/bb22-3"]


# ================================================================ the scenario row
def test_picking_a_map_is_enough_to_start(win):
    win._on_run_selected("some_run")
    win.map_list.select("real/korea26")
    assert win._selected_map == "real/korea26"
    assert win._can_start()
    cfg = win.current_config()
    assert cfg.map_name == "real/korea26", "정방향 / 없음 by default: no suffixes to choose"


def test_the_defaults_are_forward_no_obstacles_random_seed(win):
    win.map_list.select("real/korea26")
    assert win.seg_direction.current() == ""
    assert win.combo_obstacle.currentData() == ""
    assert win.combo_seed.currentData() == "random"
    # the seed controls only mean something once there are obstacles
    assert not win.spin_seed.isEnabled() and not win.btn_reroll.isEnabled()


@pytest.mark.parametrize("direction,obstacle,seed,spec,legacy", [
    ("", "", None, "real/korea26", "real:korea_2026_competition"),
    ("rev", "", None, "real/korea26@rev", "real:korea_2026_competition~rev"),
    ("mir+rev", "", None, "real/korea26@mir+rev", "real:korea_2026_competition~mir~rev"),
    # A family now carries the asset choice too (`tracks.asset_scenario`): the control places
    # modelled props of a random style and size, so the spec says which catalogue and at what
    # scale. `!assets=mixed:1` is what "고른 종류 없음, 크기 그대로" reads as.
    ("rev", "line", 44, "real/korea26@rev#line:44!assets=mixed:1",
     "real:korea_2026_competition+rlobs44~rev!assets=mixed:1"),
    ("", "edge", 3, "real/korea26#edge:3!assets=mixed:1",
     "real:korea_2026_competition+obs3!assets=mixed:1"),
])
def test_the_three_controls_build_the_scenario(win, direction, obstacle, seed, spec, legacy):
    win.map_list.select("real/korea26")
    win.seg_direction.set_current(direction)
    win.combo_obstacle.setCurrentIndex(win.combo_obstacle.findData(obstacle))
    if seed is not None:
        win.combo_seed.setCurrentIndex(win.combo_seed.findData("fixed"))
        win.spin_seed.setValue(seed)
    win._on_scenario_changed()
    assert win._scenario() == spec
    assert win.scenario_line.text() == spec
    assert win.current_config().map_name == spec
    assert tracks.resolve(spec) == legacy


def test_a_random_seed_is_drawn_by_the_worker_and_is_stable_for_one_config(win):
    """The GUI must not invent the number: a seed drawn on the paint thread is a different map
    every repaint, and the facts strip could not honestly report it."""
    win.map_list.select("real/korea26")
    win.combo_obstacle.setCurrentIndex(win.combo_obstacle.findData("line"))
    win._on_scenario_changed()
    cfg = win.current_config()
    assert cfg.map_name == "real/korea26#line:*!assets=mixed:1"
    assert win.btn_reroll.isEnabled()
    drawn = cfg.scenario()
    assert drawn.seed is not None and not drawn.random_seed
    assert cfg.scenario().legacy() == drawn.legacy(), "same config, same map"
    assert drawn.legacy().startswith("real:korea_2026_competition+rlobs")


def test_reroll_changes_the_map_and_therefore_the_generation(win):
    win.map_list.select("real/korea26")
    win.combo_obstacle.setCurrentIndex(win.combo_obstacle.findData("line"))
    win._on_scenario_changed()
    before = win.current_config()
    win._on_reroll()
    after = win.current_config()
    assert after.seed != before.seed
    assert after.scenario().legacy() != before.scenario().legacy()
    assert before.affects_simulation(after), "a new placement is a new session, not a live command"


def test_a_track_is_only_offered_the_obstacles_its_loader_can_carry(win):
    """The *families* are the loader's limit. 기본 and 없음 are not families and are on every map:
    neither adds anything, so neither can be unsupported.

    Since the families became placements of modelled props (`tracks.ASSET_OBSTACLES`, with the
    catalogue named by `!assets=`), a racetrack carries all three: what it could never carry was
    the old grid-rasterised `+obs`/`+pinch`, and there is no such thing on this control any more.
    Checked against the loader rather than assumed -- `maps.load(tracks.resolve(...))` builds
    `rt:Monza+rlobs44!assets=mixed:1`, `+hard3` and `+obs7` -- so this list is the loader's answer
    and not a preference. 학습과 같음 sits after 기본 and is not a family at all (it is the env's
    generator), so it appears on every map that has the control.
    """
    from f1sim.viewer.console.window import TRAIN_OBSTACLES
    win.map_group.setCurrentIndex(win.map_group.findData("검증"))
    assert win.map_list.select("rt/monza")
    offered = [win.combo_obstacle.itemData(i) for i in range(win.combo_obstacle.count())]
    assert offered == ["bare", "", TRAIN_OBSTACLES, "edge", "line", "hard"]
    win.map_group.setCurrentIndex(win.map_group.findData("학습"))
    assert win.map_list.select("real/korea26")
    offered = [win.combo_obstacle.itemData(i) for i in range(win.combo_obstacle.count())]
    assert offered == ["bare", "", TRAIN_OBSTACLES, "edge", "line", "hard"]


def test_the_selection_card_names_the_map_in_korean(win):
    win.map_list.select("real/bb22-1")
    win.seg_direction.set_current("rev")
    win.combo_obstacle.setCurrentIndex(win.combo_obstacle.findData("line"))
    win.combo_seed.setCurrentIndex(win.combo_seed.findData("fixed"))
    win.spin_seed.setValue(44)
    win._on_scenario_changed()
    assert win.sel_map.text() == "맵: Blackbox 2022 #1 · 역방향 · 랜덤 · 중간 (시드 44) · 에셋·크기 자동 무작위"
    assert win.sel_map.toolTip() == "real/bb22-1@rev#line:44!assets=mixed:1"


# ================================================================ the facts strip
def test_the_facts_strip_shows_the_scenario_that_was_actually_built(win):
    """`#line:*` is what was asked for; the strip has to show the number that came out."""
    win.apply_state(STATE_RUNNING)
    win.set_session_facts({"gen": 3, "run": "ppo_v1", "map": "real/bb22-1@rev#line:*",
                           "scenario": "real/bb22-1@rev#line:4417",
                           "scenario_display": "Blackbox 2022 #1 · 역방향 · 주행선 위 (시드 4417)",
                           "map_legacy": "real:blackbox2022_1+rlobs4417~rev",
                           "races": 1, "cars_per_race": 1, "total_cars": 1, "car_ids": [0],
                           "device": "cpu"})
    assert "real/bb22-1@rev#line:4417" in win.header_summary.text()
    assert "4417" in win.header_summary.toolTip()
    assert "real:blackbox2022_1+rlobs4417~rev" in win.header_summary.toolTip()
