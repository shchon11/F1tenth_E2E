"""The diagnostic panel must not reserve space it is not using.

A user reported dead space under the LiDAR/BEV panel and the gauges. The cause was that the panel's
height came from `sizeHint()`, which is not width-aware: the two panels sit in a flow layout whose
hint is the stacked one-column height regardless of the width the row actually has. At 1600x950 it
asked for 414 px to draw 228 px of content, so 186 px sat empty and the 3D view -- which takes the
remaining stretch -- got 358 px instead of 538.

The other half of the requirement is that fixing it must not clip anything: at a narrow window the
two panels stack, the content is genuinely taller than the cap allows, and the area has to scroll.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtWidgets                                             # noqa: E402


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
    existing = QtWidgets.QApplication.instance()
    yield existing or A.create_app(["test"])
    if _prev_qpa is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = _prev_qpa


@pytest.fixture
def window(qapp):
    from f1sim.viewer.console.window import ConsoleWindow
    w = ConsoleWindow()
    w.show()
    yield w
    # `close()` alone does NOT close this window: `closeEvent` refuses the first request and emits
    # `close_requested` so the controller can wind a worker down first (`window.py:471-484`).
    # Without a controller nothing ever calls `allow_close()`, so the window stayed alive and
    # visible for the rest of the session, holding its GL viewport. `allow_close()` tears the
    # viewport down while the context still exists and then closes for real; `deleteLater` plus a
    # drained deferred-delete queue frees the QWidget while the QApplication is still up to do it.
    w.allow_close()
    w.deleteLater()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def settle(qapp, times=10):
    for _ in range(times):
        qapp.processEvents()


#: (width, height) -- the two sizes the delivery screenshots are taken at, plus the size a
#: maximising window manager actually produced on the development machine.
SIZES = [(1600, 950), (1120, 700), (1920, 1043)]


@pytest.mark.parametrize("size", SIZES)
def test_panel_is_capped_to_its_content_and_never_clips(window, qapp, size):
    w, h = size
    window.resize(w, h)
    settle(qapp)
    window.policy_fold.toggle.setChecked(True)          # as if the user opened it
    settle(qapp)

    ps = window.policy_scroll
    need = window.policy_fold.heightForWidth(ps.viewport().width())
    assert need > 0, "the fold is expanded, so it should report a real height for this width"
    if ps.height() < need:
        assert ps.verticalScrollBar().isVisible(), (
            "the panel is shorter than its content, so it has to scroll rather than clip")
    else:
        # capped to the content: a little slack for margins, never a whole extra row
        assert ps.height() - need <= 12, (
            f"panel reserves {ps.height()}px for {need}px of content at {w}x{h}")


@pytest.mark.parametrize("size", SIZES)
def test_the_viewport_takes_the_slack(window, qapp, size):
    w, h = size
    window.resize(w, h)
    settle(qapp)
    window.policy_fold.toggle.setChecked(True)
    settle(qapp)
    assert window.viewport.height() >= 240, "the 3D view never goes below its minimum"
    # the picture, not the readout beneath it, should own most of the column
    assert window.viewport.height() > window.policy_scroll.height(), (
        f"3D view {window.viewport.height()}px vs panel {window.policy_scroll.height()}px at {w}x{h}")


def test_collapsing_the_fold_returns_the_space(window, qapp):
    window.resize(1600, 950)
    settle(qapp)
    window.policy_fold.toggle.setChecked(True)
    settle(qapp)
    open_h = window.viewport.height()
    window.policy_fold.toggle.setChecked(False)
    settle(qapp)
    assert window.viewport.height() > open_h
    assert window.policy_scroll.height() <= 40
