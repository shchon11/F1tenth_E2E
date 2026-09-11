"""The real worker against the real console controller.

`test_console_session.py` drives the controller with a deterministic stand-in worker, and
`test_sim_worker_*.py` drive the worker with a stub simulator. Both halves passing does not prove
they fit together: the seams that break are the ones neither side owns -- which end of
`Pipe(duplex=False)` the worker gets, whether `dash` unpacks in the order the panel expects,
whether the geometry dict has the keys `TrackGeometry` reads, whether a frame survives
`FrameBuffer.push`'s validation.

So this file wires `SessionController` straight to `f1sim.viewer.sim_worker.main` with a real
checkpoint, on the CPU, and checks that a picture actually arrives. It is skipped without a
display or without a checkpoint.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets                                      # noqa: E402

from f1sim.viewer.console import app as console_app                      # noqa: E402
from f1sim.viewer.console.protocol import (STATE_PAUSED, STATE_RUNNING,  # noqa: E402
                                           SessionConfig)
from f1sim.viewer.console.session import SessionController               # noqa: E402
from f1sim.viewer.console.window import ConsoleWindow                    # noqa: E402
import f1sim.viewer.sim_worker as sim_worker                             # noqa: E402

SMOKE_MAP = "gen:competition:2"


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        pytest.skip("no display")
    app = QtWidgets.QApplication.instance() or console_app.create_app(["test"])
    yield app


@pytest.fixture(scope="module")
def live(qapp, tmp_legacy_run):
    """One real session, shared by the checks below: it costs a checkpoint load and a map build."""
    window = ConsoleWindow()
    window.resize(1200, 760)
    controller = SessionController(window, worker_target=sim_worker.main)
    window.show()

    def pump(seconds=0.05):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            qapp.processEvents(QtCore.QEventLoop.AllEvents, 5)
            time.sleep(0.002)

    def wait_for(predicate, timeout=300.0, what="condition"):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            pump(0.02)
        pytest.fail(f"timed out waiting for {what} (state={window.state})")

    harness = type("Live", (), {})()
    harness.window, harness.controller, harness.pump, harness.wait_for = window, controller, pump, wait_for
    # `start_session` and the wait live INSIDE the try. They used to sit above it, so a setup
    # failure -- the wait timing out -- raised before the try was entered and `shutdown()` never
    # ran. The reader QThreads started by `start_session` then outlived their objects, and Qt
    # aborted the interpreter with "QThread: Destroyed while thread is still running", rc 134,
    # taking the whole pytest process and its unflushed summary with it. `SessionController.
    # shutdown()` is correct and joins its readers; it was simply never called.
    try:
        pump(0.2)
        controller.start_session(SessionConfig(run=tmp_legacy_run, map_name=SMOKE_MAP,
                                               races=1, cars_per_race=1, device="cpu",
                                               compile=False))
        wait_for(lambda: window.state == STATE_RUNNING, what="RUNNING with the real worker")
        yield harness
    finally:
        controller.shutdown()
        window.viewport.teardown()
        window.close()
        window.deleteLater()
        pump(0.1)


def test_the_console_reaches_running_with_the_real_worker(live):
    assert live.window.state == STATE_RUNNING
    assert live.controller._worker_hello["pid"] == live.controller._proc.pid


def test_real_frames_pass_the_console_validator(live):
    """`FrameBuffer.push` rejects a frame missing any of the keys the renderer needs, silently.
    Counting them is the only way a shape mistake in the worker shows up as a test failure rather
    than as an empty viewport."""
    live.wait_for(lambda: live.controller.buffer.received > 5, what="frames through the buffer")
    stats = live.controller.buffer.stats()
    assert stats["malformed"] == 0, "the worker sent frames the console refused at the door"
    assert live.controller.buffer.latest is not None
    frame = live.controller.buffer.frame_to_draw()
    assert frame is not None and frame["n"] >= 1


def test_the_facts_reached_the_window(live):
    assert SMOKE_MAP in live.window.header_summary.text()
    assert live.window.combo_focus.count() >= 1
    assert live.window.viewport.lidar_cfg.get("n_beams", 0) > 0
    assert live.window.viewport.color_v_max > 0


def test_geometry_uploaded_and_the_viewport_has_a_map(live):
    live.wait_for(lambda: live.window.viewport.geometry_data is not None, what="geometry upload")
    geom = live.window.viewport.geometry_data
    assert geom.nbytes() > 0 and geom.build_ms > 0
    assert geom.bounds[2] > geom.bounds[0] and geom.bounds[3] > geom.bounds[1]


def test_the_dash_panel_unpacks_the_frame_in_the_right_order(live):
    """`dash` is seven numbers and `gg` is a separate key; the panel's argument list interleaves
    them. Getting it wrong swaps the speedometer with the steering angle and neither reads as
    obviously wrong on screen."""
    live.wait_for(lambda: live.window.dash_panel._have, what="dash data")
    panel = live.window.dash_panel
    assert 0.0 <= panel.v < 30.0 and 0.0 <= panel.v_cmd < 30.0
    assert abs(panel.steer) < 1.0 and abs(panel.steer_cmd) < 1.0, "steering is radians, not m/s"
    assert panel.v_cap > 0 and panel.v_max >= panel.v_cap
    assert panel.mu_g is None or 0.0 < panel.mu_g < 30.0


def test_pause_settles_only_on_the_workers_ack(live):
    live.controller.set_paused(True)
    live.wait_for(lambda: live.window.state == STATE_PAUSED, timeout=30, what="PAUSED after ack")
    t_paused = live.controller.buffer.latest["t"]
    live.pump(0.6)
    assert live.controller.buffer.latest["t"] <= t_paused + 1e-9

    live.controller.set_paused(False)
    live.wait_for(lambda: live.window.state == STATE_RUNNING, timeout=30, what="RUNNING after ack")
    live.wait_for(lambda: live.controller.buffer.latest["t"] > t_paused, timeout=30,
                  what="time advancing again")


def test_the_event_loop_kept_answering_throughout(live):
    """The whole reason the worker is a process. If this is slow, nothing else in the rebuild
    matters."""
    worst = 0.0
    for _ in range(60):
        t0 = time.monotonic()
        QtWidgets.QApplication.instance().processEvents(QtCore.QEventLoop.AllEvents, 5)
        worst = max(worst, time.monotonic() - t0)
        time.sleep(0.005)
    assert worst < 0.25, f"the Qt event loop was blocked for {worst*1e3:.0f} ms"
