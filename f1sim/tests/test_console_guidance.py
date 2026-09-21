"""The parts of the console that tell a person what to do, and the settings it promises to keep.

Three things are asserted here, each of which broke silently once:

* **The panel reads as the order you work in.** The two cards a session cannot start without are
  numbered, and everything optional either sits below them or is folded. A card added in the wrong
  place, or a fold opened by default, costs nothing at import time and makes the first screen a
  list of settings again.
* **Something always says what the next action is.** `시작` being greyed out is not an explanation.
  `_start_blocker` is that explanation, and a wrong or empty string is indistinguishable from the
  button being broken.
* **Every setting the console collects, it restores.** `collect_prefs` and `apply_prefs` are two
  lists that have to agree, and they disagree silently: a key saved with no matching restore, or a
  restore that looks the value up in the wrong type (`combo.findData("30")` against int data),
  leaves the control on its default and looks exactly like nothing was ever saved -- which is the
  bug the user reported in the first place.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtWidgets                                             # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    _prev_qpa = os.environ.get("QT_QPA_PLATFORM")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from f1sim.viewer.console import app as A
    existing = QtWidgets.QApplication.instance()
    yield existing or A.create_app(["test"])
    if _prev_qpa is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = _prev_qpa


def _new_window():
    from f1sim.viewer.console.window import ConsoleWindow
    return ConsoleWindow()


def _drop(w):
    # Same dance as `test_console_layout`: `close()` is refused until a controller allows it.
    w.allow_close()
    w.deleteLater()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.fixture
def window(qapp):
    w = _new_window()
    # `isVisible()` is false for every child of a window that was never shown, whatever the child's
    # own flag says -- so without this the fold assertions below would pass on a panel with all
    # three sections wide open.
    w.show()
    yield w
    _drop(w)


def panel_sections(w):
    """The left panel's cards and folds, top to bottom, by their visible titles."""
    from f1sim.viewer.console.widgets import Card, Collapsible
    out = []
    for c in w.left_panel.findChildren((Card, Collapsible)):
        title = getattr(c, "title_label", None)
        if title is not None:
            out.append(title.text())
        elif isinstance(c, Collapsible):
            out.append(c.toggle.text().strip())
    return out


# ------------------------------------------------------------------ the order you work in
def test_the_two_required_cards_are_numbered_and_come_first(window):
    sections = panel_sections(window)
    assert "① 정책 런" in sections and "② 맵" in sections, sections
    assert sections.index("① 정책 런") < sections.index("② 맵"), sections
    # Nothing optional may push either of them down the panel.
    for later in ("구성", "녹화", "고급 설정"):
        hits = [i for i, s in enumerate(sections) if later in s]
        assert hits, f"{later} is gone from the panel: {sections}"
        assert min(hits) > sections.index("② 맵"), (later, sections)


@pytest.mark.parametrize("fold", ["record_fold", "adv_fold", "ckpt_fold"])
def test_the_optional_forms_start_folded(window, fold):
    assert not getattr(window, fold).content.isVisible()


def test_recording_stays_one_click_while_folded(window):
    """The seven recording settings fold away; the button that starts one does not."""
    assert not window.record_fold.content.isVisible()
    btn = window.btn_record
    # On the fold's header row, so it is on screen with the section shut.
    assert btn.parent() is window.record_fold or btn.parentWidget() is window.record_fold
    assert btn not in window.record_fold.content.findChildren(type(btn))


# ------------------------------------------------------------------ what to do next
def test_a_fresh_console_says_both_things_are_missing(window):
    blocker = window._start_blocker()
    assert "①" in blocker and "②" in blocker, blocker
    assert window.start_hint.text() == blocker
    assert not window._can_start()


def test_the_blocker_names_only_what_is_actually_missing(window):
    window._selected_run = "/tmp/nonexistent/ppo_latest.pt"
    window._update_start_enabled()
    blocker = window._start_blocker()
    assert "②" in blocker and "①" not in blocker, blocker

    window._selected_map = "real:map12x16"
    window._update_start_enabled()
    assert window._start_blocker() == ""
    assert "시작" in window.start_hint.text()


# ------------------------------------------------------------------ the viewport toast
def test_notify_puts_the_message_over_the_scene(window):
    vp = window.viewport
    assert not vp._toast.isVisible()
    vp.notify("그립 다이얼 = 0.734", "good")
    assert vp._toast.isVisible()
    assert vp._toast.text() == "그립 다이얼 = 0.734"
    # Bottom-centre of the viewport, clear of the top-left/top-right chips.
    vp.resize(900, 600)
    vp._place_overlays()
    assert vp._toast.y() > vp.height() // 2


def test_notify_ignores_an_empty_message(window):
    window.viewport.notify("")
    assert not window.viewport._toast.isVisible()


# ------------------------------------------------------------------ settings survive a restart
#: Set each remembered control to something that is not its default. A key left out of this map is
#: reported by `test_every_collected_preference_is_exercised`, so the round-trip below cannot be
#: passed by a control that simply never moved.
def _move_every_control(w):
    w.spin_races.setValue(4)
    w.spin_grid.setValue(3)
    w.spin_cap.setValue(7.5)
    w.chk_dr.setChecked(False)
    w.chk_soft.setChecked(False)
    w.chk_stoch.setChecked(True)
    w.chk_saliency.setChecked(not w.chk_saliency.isChecked())
    w.chk_internals.setChecked(not w.chk_internals.isChecked())
    w.spin_mu.setValue(0.812)
    w.spin_dial.setValue(0.655)
    w.spin_seed.setValue(4242)
    w.seg_direction.set_current("rev")
    w.combo_seed.setCurrentIndex(w.combo_seed.findData("fixed"))
    w.combo_mu.setCurrentIndex(w.combo_mu.findData("fixed"))
    w.combo_ros.setCurrentIndex(w.combo_ros.findData("publish"))
    w.combo_obstacle.setCurrentIndex(w.combo_obstacle.count() - 1)   # a family, not 없음/기본
    w.chk_bare_first.setChecked(True)                                # reset unless a family is set
    if w.combo_device.count() > 1:
        w.combo_device.setCurrentIndex(w.combo_device.count() - 1)
    w.edit_record.setText("/tmp/clips")
    w.combo_res.setCurrentIndex(w.combo_res.count() - 1)
    w.combo_fps.setCurrentIndex(0 if w.combo_fps.currentIndex() else 1)
    w.combo_rec_cam.setCurrentIndex(w.combo_rec_cam.count() - 1)
    w.spin_rec_secs.setValue(45.0)
    w.chk_rec_overlay.setChecked(False)
    w.combo_rec_enc.setCurrentIndex(w.combo_rec_enc.count() - 1)
    w.record_fold.toggle.setChecked(True)
    w.adv_fold.toggle.setChecked(True)
    w.ckpt_fold.toggle.setChecked(True)


#: `run` needs a checkpoint on disk and `map` needs the worker's catalogue (`set_maps` applies it),
#: so neither can be round-tripped by a bare window. Both are covered by `test_console_startup`.
_NEEDS_A_WORKER = {"run", "map"}


def test_every_setting_comes_back_after_a_restart(window, qapp):
    _move_every_control(window)
    saved = window.collect_prefs()

    fresh = _new_window()
    try:
        fresh.apply_prefs(saved)
        got = fresh.collect_prefs()
    finally:
        _drop(fresh)

    differs = {k: (saved[k], got[k]) for k in saved
               if k not in _NEEDS_A_WORKER and saved[k] != got[k]}
    assert not differs, f"saved but not restored: {differs}"


def test_every_collected_preference_is_exercised(window):
    """A new key in `collect_prefs` that `_move_every_control` does not touch would ride through
    the round-trip above on its default value, proving nothing. This is what says so."""
    before = window.collect_prefs()
    _move_every_control(window)
    after = window.collect_prefs()
    untouched = [k for k in before
                 if k not in _NEEDS_A_WORKER and before[k] == after[k]]
    assert not untouched, (
        f"{untouched} never changed: add them to _move_every_control so the round-trip test "
        f"actually covers them")
