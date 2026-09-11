"""Viewer startup: what the user waited ten minutes for, and what the status line told them.

Three separate faults produced one experience -- open the viewer, wait five to ten minutes on every
launch, and read "worker 응답 지연" the whole time:

* `SessionConfig.compile` defaulted to True and the checkbox was ticked and labelled "CUDA 가속",
  so the expensive `torch.compile` looked like the switch for using the GPU at all. It is not:
  `device` decides that, and a session runs on CUDA either way.
* `start` is answered by `ready`, not by an ack, but `tick_pending` measured it against the 0.7 s
  ack threshold and overwrote the stage line on every tick -- so the console reported the worker
  unresponsive while the worker was compiling.
* The modelled obstacles sat in a demo group of three maps while the obvious-looking obstacle group
  was the legacy raster set.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtWidgets                                             # noqa: E402

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


# ----------------------------------------------------------------- compile is opt-in
def test_a_default_session_does_not_compile():
    """Opening the viewer must not buy minutes of torch.compile nobody asked for."""
    assert SessionConfig().compile is False


def test_the_checkbox_is_off_and_does_not_call_itself_cuda(window):
    """The label said 'CUDA 가속', so turning it off looked like turning the GPU off."""
    assert window.chk_compile.isChecked() is False
    label = window.chk_compile.text()
    assert "CUDA 가속" not in label, f"the label still reads as a GPU switch: {label!r}"
    assert "컴파일" in label
    tip = window.chk_compile.toolTip()
    assert "GPU 사용 여부와는 무관" in tip, "the tooltip must say this is not the GPU switch"


def test_the_config_the_start_button_sends_has_compile_off(window):
    window._selected_run, window._selected_map = "r", "real:korea_2026_competition"
    assert window.current_config().compile is False
    window.chk_compile.setChecked(True)
    assert window.current_config().compile is True, "the option must still be available"


def test_device_and_compile_are_independent(window):
    """Compiling off must not change which device the session asks for."""
    window._selected_run, window._selected_map = "r", "m"
    window.combo_device.setCurrentText("cuda")
    off = window.current_config()
    window.chk_compile.setChecked(True)
    on = window.current_config()
    assert off.device == on.device == "cuda"
    assert off.compile is False and on.compile is True


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


def test_a_long_compile_stage_says_how_to_avoid_it(window):
    window.apply_state(STATE_PREPARING, "checkpoint")
    window._on_start()
    window.set_stage("compile", "1/8")
    window._pending["start"] = time.monotonic() - 90.0
    window._stage_since = time.monotonic() - 80.0
    window.tick_pending()
    text = window.status_text.text()
    # `compile` is inductor and costs minutes; the explicit CUDA-graph capture is a separate,
    # ~1 s stage named `graph`. Calling the slow one "그래프 캡처" is what made the checkbox read
    # as if switching it off gave up the GPU.
    assert "torch.compile" in text
    assert "고급 설정" in text, "a multi-minute stage should say how to switch it off"


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


# ----------------------------------------------------------------- the obstacle groups
def test_the_modelled_obstacles_are_the_normal_obstacle_group():
    from f1sim.viewer.console.catalog import GROUP_HINT, GROUP_ORDER
    props, legacy = "장애물 (상자·궤짝·드럼)", "이전 실험 재현 (격자 장애물)"
    assert props in GROUP_ORDER and legacy in GROUP_ORDER
    assert GROUP_ORDER.index(props) < GROUP_ORDER.index(legacy), (
        "the legacy raster set is listed above the modelled obstacles")
    assert "부딪" in GROUP_HINT[props] and "LiDAR" in GROUP_HINT[props]
    # the legacy group must say what it is, not merely be present
    assert "격자" in GROUP_HINT[legacy] and "이전" in GROUP_HINT[legacy]
    assert "윗면이 없" in GROUP_HINT[legacy], "the legacy set's actual limitation should be stated"


def test_worker_and_console_agree_on_the_group_names():
    import re
    from f1sim.viewer.console.catalog import GROUP_ORDER
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "f1sim", "viewer", "sim_worker.py")).read()
    block = src[src.index('        return {\n            "기본 평가셋"'):]
    worker = re.findall(r'"([^"]+)": uniq\(', block[:1200])
    assert worker == GROUP_ORDER, f"worker {worker} vs console {GROUP_ORDER}"


def test_props_names_use_the_loader_s_suffix_order():
    """`<base>+props<seed>~rev`, the order `+obs101~rev` already uses."""
    from f1sim.learn import common
    names = [n for n in common.EVAL_OBSTACLE_TRACKS if "~" in n]
    assert names and all(n.index("+") < n.index("~") for n in names), (
        "the obstacle suffix comes before the direction suffix")


def test_the_training_obstacle_split_is_untouched():
    """The viewer relabels a group; it must not redefine what training evaluates against."""
    from f1sim.learn import common
    assert len(common.EVAL_OBSTACLE_TRACKS) == 24
    assert "real:korea_2025_iccas+obs101" in common.EVAL_OBSTACLE_TRACKS
    assert not any("+props" in n for n in common.EVAL_OBSTACLE_TRACKS), (
        "props must not have been quietly added to the training obstacle split")
