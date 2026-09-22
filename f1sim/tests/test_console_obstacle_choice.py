"""The 장애물 control, on both console pages, after 기본 and 없음 stopped being one entry.

What a label claims is the whole subject here, so the tests are about text as much as about state:
a scene with three placed obstacles must say so on 기본, must offer 없음 as something that removes
them, and must not offer 없음 at all -- except greyed, with the reason -- on a map that has none.

Offscreen Qt, like the other console tests; nothing here touches GL or torch's map loader.
"""
import os

import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

from f1sim import tracks as T
from f1sim.scene import SceneDoc
from f1sim.viewer.console import training as TR
from f1sim.viewer.console.catalog import MapCatalog


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


@pytest.fixture
def scenes(tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
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
    return {"scene/hall": 3, "scene/empty": 0}


def _catalog(scenes):
    entries = {tid: {"id": tid, "family": "scene", "family_label": "에디터",
                     "display": tid.split("/", 1)[1], "legacy": f"scene:{tid.split('/', 1)[1]}",
                     "note": "", "obstacles": list(T.OBSTACLES_BY_FAMILY["scene"]), "props": n}
               for tid, n in scenes.items()}
    return MapCatalog(groups={"내 환경": list(scenes)}, entries=entries, ready=True)


@pytest.fixture
def window(qapp, scenes):
    from f1sim.viewer.console.window import ConsoleWindow
    w = ConsoleWindow()
    w.set_maps(_catalog(scenes))
    try:
        yield w
    finally:
        w.deleteLater()


def _select(window, tid):
    window._selected_map = tid
    window._sync_obstacle_options()
    window._on_scenario_changed()


def _items(window):
    c = window.combo_obstacle
    return [(c.itemText(i), c.itemData(i),
             bool(c.model().item(i).flags() & QtCore.Qt.ItemIsEnabled),
             c.itemData(i, QtCore.Qt.ToolTipRole) or "")
            for i in range(c.count())]


# ================================================================ the catalogue knows the count
def test_the_catalogue_reports_how_many_obstacles_an_author_placed(scenes):
    cat = _catalog(scenes)
    assert cat.authored_props("scene/hall") == 3
    assert cat.authored_props("scene/empty") == 0
    assert cat.authored_props("real/bb22-1") == 0


def test_a_scene_saved_after_the_catalogue_was_built_is_counted_anyway(scenes, tmp_path):
    """The editor is in this process. A count that lagged a save would put the wrong number on the
    labels this number exists to make honest."""
    cat = _catalog({"scene/hall": 0})                 # a stale entry, as if it were saved empty
    assert cat.authored_props("scene/hall") == 3


# ================================================================ 주행 page
def test_the_default_entry_says_what_the_map_already_carries(window):
    _select(window, "scene/hall")
    text = dict((d, t) for t, d, _e, _tip in _items(window))
    assert text[""] == "기본 (배치된 장애물 3개)"
    assert text[T.BARE] == "없음 (배치 장애물 제거)"


def test_a_map_with_nothing_placed_greys_the_none_entry_and_says_why(window):
    _select(window, "scene/empty")
    rows = {d: (t, en, tip) for t, d, en, tip in _items(window)}
    assert rows[""][0] == "기본", "a plain map must not claim a count"
    assert rows[T.BARE][0] == "없음"
    assert rows[T.BARE][1] is False, "없음 is meaningless here and must not be selectable"
    assert "배치 장애물이 없음" in rows[T.BARE][2]


def test_choosing_none_emits_a_seedless_spec(window):
    _select(window, "scene/hall")
    window.combo_obstacle.setCurrentIndex(window.combo_obstacle.findData(T.BARE))
    assert window._scenario() == "scene/hall#bare"
    assert T.parse(window._scenario()).legacy() == "scene:hall+bare"
    # and the seed controls have nothing to seed
    assert not window.combo_seed.isEnabled() and not window.btn_reroll.isEnabled()


def test_the_default_choice_emits_the_map_itself(window):
    _select(window, "scene/hall")
    window.combo_obstacle.setCurrentIndex(window.combo_obstacle.findData(""))
    assert window._scenario() == "scene/hall"


def test_training_obstacles_emit_the_bare_map_and_turn_the_generator_on(window):
    """학습과 같음: the map as it is -- no family, no `!assets` -- and `procedural` in the config.
    The seed controls stay live: the generator draws from the session seed, so 다시 뽑기 means
    something, and 고정 hands the typed number to the generator since the spec has none."""
    from f1sim.viewer.console.window import TRAIN_OBSTACLES
    _select(window, "scene/hall")
    window.combo_obstacle.setCurrentIndex(window.combo_obstacle.findData(TRAIN_OBSTACLES))
    assert window._scenario() == "scene/hall"
    cfg = window.current_config()
    assert cfg.procedural is True and cfg.map_name == "scene/hall"
    assert window.combo_seed.isEnabled() and not window.chk_bare_first.isEnabled()
    window.combo_seed.setCurrentIndex(window.combo_seed.findData("fixed"))
    window.spin_seed.setValue(4321)
    assert window.current_config().seed == 4321
    # and every other choice leaves the generator off
    window.combo_obstacle.setCurrentIndex(window.combo_obstacle.findData(""))
    assert window.current_config().procedural is False


def test_training_obstacles_survive_a_map_change_and_the_prefs(window):
    from f1sim.viewer.console.window import TRAIN_OBSTACLES
    _select(window, "scene/hall")
    window.combo_obstacle.setCurrentIndex(window.combo_obstacle.findData(TRAIN_OBSTACLES))
    _select(window, "scene/empty")
    assert window.combo_obstacle.currentData() == TRAIN_OBSTACLES
    assert window.collect_prefs()["obstacle"] == TRAIN_OBSTACLES


def test_a_family_can_be_asked_to_clear_the_map_first(window):
    """The composition the id grammar spells `scene:hall+bare+hard3`, offered where it means
    something: a map with placed obstacles, and a family going on top of them."""
    _select(window, "scene/hall")
    window.combo_obstacle.setCurrentIndex(window.combo_obstacle.findData("hard"))
    assert window.chk_bare_first.isEnabled()
    assert window._scenario() == "scene/hall#hard:*!assets=mixed:1"
    window.chk_bare_first.setChecked(True)
    assert window._scenario() == "scene/hall#bare+hard:*!assets=mixed:1"
    assert T.parse("scene/hall#bare+hard:3").legacy() == "scene:hall+bare+hard3"


def test_the_composition_control_is_off_where_there_is_nothing_to_clear(window):
    _select(window, "scene/empty")
    window.combo_obstacle.setCurrentIndex(window.combo_obstacle.findData("hard"))
    assert not window.chk_bare_first.isEnabled()
    assert window._scenario() == "scene/empty#hard:*!assets=mixed:1"


def test_the_hint_says_a_family_adds(window):
    _select(window, "scene/hall")
    window.combo_obstacle.setCurrentIndex(window.combo_obstacle.findData("hard"))
    assert "위에" in window.row_obstacle.hint.text()
    window.combo_obstacle.setCurrentIndex(window.combo_obstacle.findData(""))
    assert "그대로" in window.row_obstacle.hint.text()


def test_switching_to_a_map_with_nothing_placed_leaves_a_meaningless_none_behind(window):
    _select(window, "scene/hall")
    window.combo_obstacle.setCurrentIndex(window.combo_obstacle.findData(T.BARE))
    _select(window, "scene/empty")
    assert str(window.combo_obstacle.currentData() or "") == "", \
        "없음 stayed selected on a map that has nothing to remove"


def test_the_header_names_the_choice_that_was_built(window):
    window.set_session_facts({"gen": 3, "run": "r", "checkpoint": "c.pt", "map": "scene/hall",
                              "scenario": "scene/hall#bare", "scenario_display": "hall · 없음",
                              "total_cars": 1, "races": 1, "cars_per_race": 1, "car_ids": [0],
                              "device": "cpu", "obstacle_text": "없음", "authored_props": 3})
    assert "장애물 없음" in window.header_summary.text()


# ================================================================ 학습 page
@pytest.fixture
def picker(qapp, scenes):
    p = TR.TrackPicker()
    try:
        yield p
    finally:
        p.deleteLater()


def test_the_training_picker_offers_the_same_two_words(picker):
    assert picker.obs_boxes[""].text() == "기본"
    assert picker.obs_boxes[T.BARE].text() == "없음"
    assert "제거" in picker.obs_boxes[T.BARE].toolTip()


def test_the_training_picker_emits_a_seedless_bare(picker):
    picker.clear_selection()
    for lw in picker._lists.values():
        for i in range(lw.count()):
            if lw.item(i).data(QtCore.Qt.UserRole) == "scene/hall":
                lw.item(i).setCheckState(QtCore.Qt.Checked)
    for d, cb in picker.dir_boxes.items():
        cb.setChecked(d == "")
    picker.obs_boxes[""].setChecked(False)
    picker.obs_boxes[T.BARE].setChecked(True)
    spec = picker.spec()
    assert spec == "scene/hall#bare", spec
    assert T.parse(spec).legacy() == "scene:hall+bare"


def test_a_job_started_with_bare_is_described_as_such():
    text = TR.describe_tracks("scene/hall#bare,scene/hall")
    assert "없음" in text and "기본" not in text.split("장애물")[0]


def test_only_five_asset_obstacle_choices_are_visible(window, picker):
    """Five asset choices, plus 학습과 같음 on the driving page -- which is not a family and is not
    offered to the training picker, whose own 환경 editors already set the generator."""
    from f1sim.viewer.console.window import TRAIN_OBSTACLES
    _select(window, "scene/hall")
    expected = ["없음", "기본", "랜덤 · 낮음", "랜덤 · 중간", "랜덤 · 높음"]
    want = list(T.ASSET_OBSTACLES)
    want.insert(want.index("") + 1, TRAIN_OBSTACLES)
    assert [window.combo_obstacle.itemData(i) for i in range(window.combo_obstacle.count())] == want
    assert [picker.obs_boxes[k].text() for k in T.ASSET_OBSTACLES] == expected
    assert "props" not in picker.obs_boxes and "pinch" not in picker.obs_boxes
    assert not hasattr(window, "obstacle_assets"), "asset type and size are automatic, not new controls"


def test_restoring_an_old_split_does_not_convert_its_recorded_obstacles(picker):
    picker.set_spec("train")
    assert picker.spec() == "train"
    picker.btn_all_train.click()
    scenarios = [T.parse(n) for n in picker.spec().split(",")]
    assert all(s.asset == "mixed" for s in scenarios if s.obstacle)
