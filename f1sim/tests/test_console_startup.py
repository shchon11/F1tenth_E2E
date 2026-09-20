"""Viewer startup: the ordinary driving page exposes automatic runtime choices clearly."""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtWidgets                                             # noqa: E402

from f1sim.viewer.console.frames import Freshness
from f1sim.viewer.console.protocol import SessionConfig, STATE_IDLE, STATE_PREPARING, STATE_RUNNING


@pytest.fixture(scope="module")
def qapp():
    # Scoped: `os.environ` is process-global and a module fixture that sets it without restoring
    # leaks the platform into every later test AND into any child process they spawn. That is not
    # hypothetical -- `test_console_launch_smoke` builds its child env from `os.environ`, so an
    # offscreen platform left here made the console open with no X window and its `xdotool` search
    # find nothing. Restored on teardown.
    _prev_qpa = os.environ.get("QT_QPA_PLATFORM")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from f1sim.viewer.console import app as A
    yield QtWidgets.QApplication.instance() or A.create_app(["test"])
    if _prev_qpa is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = _prev_qpa


@pytest.fixture
def window(qapp):
    from f1sim.viewer.console.window import ConsoleWindow
    w = ConsoleWindow()
    w.resize(1200, 760)
    w.show()
    qapp.processEvents()
    yield w
    # `close()` is refused until `allow_close()` sets the flag (`window.py:471-489`), so this used
    # to leave the window alive for the rest of the session. `allow_close()` already tears the
    # viewport down, and the deferred-delete drain frees the widget while the QApplication is alive.
    w.allow_close()
    w.deleteLater()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


# ----------------------------------------------------------------- ordinary driving needs nothing
def test_a_default_session_needs_nothing_but_the_checkpoint():
    """The default arm must be one that runs with no file the user does not have.

    This asserted `auto`, and `auto` **requires** a frozen grip estimator: not in the repository,
    not produced by any default run, not on the machine this was written on. So the test passed
    while pressing 시작 with the defaults failed with "자동 노면 추정 모델을 찾지 못했습니다" --
    which is the whole reason it is now `legacy`, the plain tracker, which needs only the
    checkpoint that was picked. The estimator arms remain for callers that have the file.
    """
    config = SessionConfig()
    assert config.controller == "legacy"
    assert config.estimator == ""
    assert config.compile is False


def test_ordinary_driving_hides_experiment_settings(window):
    for old_widget in ("combo_controller", "edit_estimator", "chk_compile"):
        assert not hasattr(window, old_widget), old_widget
    assert any("노면 한계 자동 추정" in w.text()
               for w in window.findChildren(QtWidgets.QLabel))

    window._selected_run, window._selected_map = "r", "real:korea_2026_competition"
    config = window.current_config()
    # The window must not name an arm of its own: it asks for whatever a default session is, so
    # the two cannot drift apart again.
    assert config.controller == SessionConfig.controller == "legacy"
    assert config.estimator == ""
    assert config.compile is False


def test_device_selection_stays_independent_of_automatic_grip(window):
    """The GPU choice remains visible even though runtime experiment controls are gone."""
    window._selected_run, window._selected_map = "r", "m"
    # The device combo lists the machine's cards by name (`cuda:0`, `cuda:1`, ...), so the test
    # picks whichever CUDA entry it offers rather than a spelling that may not be in the list.
    cuda = next(window.combo_device.itemText(i) for i in range(window.combo_device.count())
                if window.combo_device.itemText(i).startswith("cuda"))
    window.combo_device.setCurrentText(cuda)
    config = window.current_config()
    assert config.device == cuda
    # Picking a card says nothing about the plan controller; that stays the default either way.
    assert config.controller == SessionConfig.controller and config.estimator == ""
    assert config.compile is False


def test_runtime_grip_estimate_is_separate_from_simulator_truth(window):
    estimate = window.drive_info.value_label("추정 μ")
    truth = window.drive_info.value_label("시뮬 참값 μ")
    assert estimate is not None and truth is not None
    assert estimate.text() == "추정 대기"
    assert "q50" in estimate.toolTip() and "used_mu" in estimate.toolTip()

    frame = {
        "focus": 0, "n": 1, "vx": [1.0], "lap": [1], "wall": [0.5], "steer": [0.0],
        "roll": [0.0], "pitch": [0.0], "s": [0.0], "mu": 0.72, "coll": None,
        "grip_estimate": {"used_mu": 0.76, "q10": 0.88, "q50": 0.90, "q90": 0.93,
                           "warm": 1.0, "finite": 1.0, "has_evidence": 1.0, "informative": 1.0},
    }
    fresh = Freshness(0.0, 0.0, 0.0, 0, False)
    window.update_telemetry(frame, fresh, {}, None, None, None)
    assert estimate.text() == "추정 μ 0.90 · 적용 0.76"
    assert truth.text() == "0.720"

    frame["grip_estimate"].update(q50=0.42, used_mu=0.40, informative=0.0)
    window.update_telemetry(frame, fresh, {}, None, None, None)
    assert estimate.text() == "추정 μ 0.90 · 적용 0.40"

    frame["grip_estimate"].update(warm=0.0, informative=1.0)
    window.update_telemetry(frame, fresh, {}, None, None, None)
    assert estimate.text() == "추정 대기"

    frame["grip_estimate"].update(warm=1.0, fault=1.0, used_mu=0.66)
    window.update_telemetry(frame, fresh, {}, None, None, None)
    assert estimate.text() == "센서 확인 · 적용 0.66"


# ----------------------------------------------------------------- preparing says preparing
def test_preparing_reports_the_stage_not_an_unresponsive_worker(window):
    """The reported symptom: a stage arrives, then 0.7 s later the line claims no response."""
    window.apply_state(STATE_PREPARING, "checkpoint")
    window._on_start()                                  # stamps the window's start ledger
    window.set_stage("map", "real:korea_2026_competition")
    window._pending["start"] = time.monotonic() - 45.0  # long past the ack threshold
    window._stage_since = time.monotonic() - 40.0
    window.tick_pending()
    text = window.status_text.text()
    assert "응답 지연" not in text, f"preparation reported as an unresponsive worker: {text!r}"
    assert "맵 로드 중" in text, f"the stage is missing from the status line: {text!r}"
    assert "준비 중" in text and "초" in text, "elapsed time should be visible"


def test_a_long_compile_stage_does_not_point_to_a_removed_setting(window):
    window.apply_state(STATE_PREPARING, "checkpoint")
    window._on_start()
    window.set_stage("compile", "1/8")
    window._pending["start"] = time.monotonic() - 90.0
    window._stage_since = time.monotonic() - 80.0
    window.tick_pending()
    text = window.status_text.text()
    # Explicit benchmark/debug callers may still report this stage, but ordinary driving has no
    # checkbox to point at and its config always leaves compile off.
    assert "torch.compile" in text
    assert "고급 설정" not in text


def test_runtime_control_ack_warnings_are_untouched(window):
    """A genuinely unanswered pause/reset/stop must still be reported."""
    window.apply_state(STATE_RUNNING)
    window._pending.clear()
    window._pending["pause"] = time.monotonic() - 3.0
    window.tick_pending()
    text = window.status_text.text()
    assert "응답 지연" in text and "일시정지" in text, text


def test_leaving_preparing_clears_the_stage(window):
    window.apply_state(STATE_PREPARING, "checkpoint")
    window.set_stage("env")
    assert window._stage == "env"
    window.apply_state(STATE_RUNNING)
    assert window._stage is None and window._stage_since is None


def test_preparing_never_claims_ready(window):
    window.apply_state(STATE_PREPARING, "checkpoint")
    window._on_start()
    window.set_stage("warmup")
    window._pending["start"] = time.monotonic() - 10.0
    window.tick_pending()
    assert window.state == STATE_PREPARING
    assert "주행" not in window.status_text.text()


# ----------------------------------------------------------------- the three groups
def test_there_are_exactly_three_groups_and_each_says_what_it_means():
    """The five groups the console used to show ("기본 평가셋", "장애물", "기본 학습셋",
    "이전 실험 재현", "전체 카탈로그") named where a list came from, not what it means, and two of
    them were the same maps with different obstacles stamped in. The obstacle families are an
    option on a map now, so what is left is the one question a group can answer."""
    from f1sim.viewer.console.catalog import GROUP_HINT, GROUP_ORDER, SCENES_GROUP
    assert GROUP_ORDER == ["학습", "검증", "내 환경"]
    assert SCENES_GROUP == "내 환경"
    assert "학습한" in GROUP_HINT["학습"]
    assert "학습에 쓰인 적 없" in GROUP_HINT["검증"] and "일반화" in GROUP_HINT["검증"]
    assert "에디터" in GROUP_HINT["내 환경"] or "환경 페이지" in GROUP_HINT["내 환경"]


def test_the_deleted_groups_are_gone_from_the_console_and_the_worker():
    from f1sim.viewer.console import catalog as C
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "f1sim", "viewer", "sim_worker.py")).read()
    for dead in ("기본 평가셋", "기본 학습셋", "이전 실험 재현", "전체 카탈로그",
                 "장애물 (상자·궤짝·드럼)"):
        assert dead not in C.GROUP_ORDER and dead not in C.GROUP_HINT, dead
        assert dead not in src, f"{dead} still built by the worker"


def test_worker_and_console_agree_on_the_group_names():
    """One definition, in `f1sim.tracks`, read by both sides. The worker used to build the groups
    from its own literal dict and the console from another one, which is two lists to keep in sync
    and a test to notice when they are not."""
    from f1sim.viewer.console.catalog import GROUP_ORDER
    from f1sim import tracks
    assert GROUP_ORDER == list(tracks.GROUP_ORDER)
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "f1sim", "viewer", "sim_worker.py")).read()
    assert "T.groups(scene_ids=scene_ids)" in src, "the worker must use the registry's groups"


def test_the_groups_hold_base_tracks_not_variants():
    """The list is a list of maps. A direction or an obstacle seed is not a row in it."""
    from f1sim import tracks
    for names in tracks.groups(scene_ids=["scene/x"]).values():
        for tid in names:
            assert "@" not in tid and "#" not in tid and "~" not in tid and "+" not in tid, tid


def test_props_names_use_the_loader_s_suffix_order():
    """`<base>+props<seed>~rev`, the order `+obs101~rev` already uses."""
    from f1sim.learn import common
    names = [n for n in common.EVAL_OBSTACLE_TRACKS if "~" in n]
    assert names and all(n.index("+") < n.index("~") for n in names), (
        "the obstacle suffix comes before the direction suffix")


def test_the_training_obstacle_split_is_untouched():
    """The viewer relabels a group; it must not redefine what training evaluates against."""
    from f1sim.learn import common
    assert len(common.EVAL_OBSTACLE_TRACKS) == 12
    assert "real:korea_2025_iccas+obs101" in common.EVAL_OBSTACLE_TRACKS
    assert not any("+props" in n for n in common.EVAL_OBSTACLE_TRACKS), (
        "props must not have been quietly added to the training obstacle split")


# ---------------------------------------------------------------- remembered settings (2026-09-20)
def test_console_remembers_its_settings_between_launches(qapp, tmp_path, monkeypatch):
    """Every control was reset to its default on each launch; the same run set up twice cost the
    whole sidebar twice."""
    from f1sim.viewer.console.window import ConsoleWindow
    monkeypatch.setenv("F1SIM_CONSOLE_PREFS", str(tmp_path / "console.json"))

    w = ConsoleWindow()
    w.spin_cap.setValue(7.5)
    w.spin_grid.setValue(3)
    w.chk_dr.setChecked(False)
    w.combo_mu.setCurrentIndex(w.combo_mu.findData("fixed"))
    w.spin_mu.setValue(0.82)
    w.spin_dial.setValue(0.90)
    ckpt = tmp_path / "run.pt"; ckpt.write_text("x")
    w._on_run_selected(str(ckpt))
    w._selected_map = "real/map12x16"
    w.save_prefs()

    back = ConsoleWindow()
    assert back.spin_cap.value() == 7.5 and back.spin_grid.value() == 3
    assert back.chk_dr.isChecked() is False
    assert back.combo_mu.currentData() == "fixed" and abs(back.spin_mu.value() - 0.82) < 1e-6
    assert abs(back.spin_dial.value() - 0.90) < 1e-6
    assert back._selected_run == str(ckpt)
    assert back._pref_map == "real/map12x16"       # applied when the worker's catalogue arrives


def test_a_setting_that_can_no_longer_be_restored_costs_only_itself(qapp, tmp_path, monkeypatch):
    from f1sim.viewer.console.window import ConsoleWindow
    monkeypatch.setenv("F1SIM_CONSOLE_PREFS", str(tmp_path / "console.json"))
    ckpt = tmp_path / "run.pt"; ckpt.write_text("x")
    w = ConsoleWindow(); w.spin_cap.setValue(7.5); w._on_run_selected(str(ckpt)); w.save_prefs()

    ckpt.unlink()                                   # the checkpoint is gone; the rest is still valid
    back = ConsoleWindow()
    assert back._selected_run is None and back.spin_cap.value() == 7.5

    (tmp_path / "console.json").write_text("{ not json")
    assert ConsoleWindow().spin_cap.value() == 6.0  # unreadable: defaults, silently
