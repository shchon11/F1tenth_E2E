"""Controller wiring, against a deterministic stand-in worker.

What is under test is the console's half of the contract: generations, acks, the pause semantics,
error and cancel paths, staged shutdown, and -- the point of the whole rebuild -- that none of it
can stop the Qt event loop from running. The real simulation worker is owned elsewhere and needs a
GPU and half a minute of CUDA graph capture; none of that is what this file is about.

`console_fake_worker` speaks the same protocol with synthetic data. Scenarios are selected by the
map name (`ok:` / `fail:` / `slow:` / `hang:` / `mute:`).
"""
import os
import sys
import time

import pytest

pytest.importorskip("PyQt5")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))    # so a spawned child can import it

from PyQt5 import QtCore, QtWidgets                                # noqa: E402

import console_fake_worker                                        # noqa: E402
from f1sim.viewer.console import app as console_app                # noqa: E402
from f1sim.viewer.console.protocol import (STATE_FAILED, STATE_IDLE, STATE_PAUSED,  # noqa: E402
                                           STATE_PREPARING, STATE_RUNNING, SessionConfig)
from f1sim.viewer.console.session import SessionController        # noqa: E402
from f1sim.viewer.console.window import ACK_WARN_S, ConsoleWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY"):
        pytest.skip("no display")
    app = QtWidgets.QApplication.instance() or console_app.create_app(["test"])
    yield app


class Harness:
    """A window + controller driven by a fake worker, with a pump that never blocks."""

    def __init__(self, qapp):
        self.app = qapp
        self.window = ConsoleWindow()
        self.window.resize(1200, 760)
        self.controller = SessionController(self.window, worker_target=console_fake_worker.main)
        self.window.show()
        self.pump(0.2)

    def pump(self, seconds=0.05):
        """Run the event loop for a while, the way a user's machine would."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.app.processEvents(QtCore.QEventLoop.AllEvents, 5)
            time.sleep(0.002)

    def wait_for(self, predicate, timeout=15.0, what="condition"):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            self.pump(0.02)
        pytest.fail(f"timed out waiting for {what} (state={self.window.state})")

    def start(self, map_name="ok:alpha", **kw):
        cfg = SessionConfig(run="fake_run", map_name=map_name, device="cpu", compile=False, **kw)
        self.controller.start_session(cfg)
        return cfg

    def close(self):
        self.controller.shutdown()
        self.window.viewport.teardown()      # free GL while the context is still alive
        self.window.close()
        self.window.deleteLater()
        self.pump(0.05)


@pytest.fixture
def h(qapp):
    harness = Harness(qapp)
    try:
        yield harness
    finally:
        harness.close()


# ================================================================ handshake
def test_worker_handshake_and_map_catalog(h):
    h.controller.start_worker()
    h.wait_for(lambda: bool(h.controller._worker_hello), what="hello")
    hello = h.controller._worker_hello
    assert hello["pid"] > 0
    assert hello["protocol"] >= 1
    h.wait_for(lambda: h.controller._maps.ready, what="map catalog")
    assert set(h.controller._maps.groups) == set(console_fake_worker.MAP_GROUPS)
    assert h.window.map_group.count() == len(console_fake_worker.MAP_GROUPS)


def test_clock_link_measures_a_local_round_trip(h):
    h.controller.start_worker()
    h.wait_for(lambda: h.controller.buffer.clock.ready, what="clock link")
    c = h.controller.buffer.clock
    assert abs(c.offset) < 0.5, c.offset            # same machine, shared CLOCK_MONOTONIC
    assert c.uncertainty < 0.5


def test_slow_worker_startup_does_not_inflate_frame_age(qapp, monkeypatch):
    """End to end version of the clock bug: a worker that answers hello two seconds late.

    A midpoint offset estimate would decide its clock runs ~1 s ahead, and every fresh frame would
    then be reported as a second old -- a freshness display that is wrong in the direction that
    makes people distrust a healthy viewer.
    """
    monkeypatch.setenv("F1SIM_FAKE_WORKER_SLOW_HELLO", "2.0")
    harness = Harness(qapp)
    try:
        harness.start("ok:alpha")
        harness.wait_for(lambda: harness.window.state == STATE_RUNNING, timeout=25, what="RUNNING")
        harness.wait_for(lambda: harness.controller.buffer.stats()["received"] > 5, what="frames")
        harness.wait_for(lambda: harness.controller.buffer.clock.ready, timeout=25, what="hello")
        c = harness.controller.buffer.clock
        assert c.rtt >= 1.5, "the fixture should have taken at least ~2 s to answer"
        assert c.offset == 0.0, "a local worker's clock must not be 'corrected'"
        fresh = harness.controller.buffer.freshness()
        assert fresh.creation_age < 0.5, f"fresh frame reported as {fresh.creation_age:.2f}s old"
        assert not fresh.is_stale()
    finally:
        harness.close()


def test_describe_failure_is_reported_and_the_worker_survives(h):
    h.controller.start_worker()
    h.wait_for(lambda: bool(h.controller._worker_hello), what="hello")
    h.controller.describe("missing_run")
    h.wait_for(lambda: "없습니다" in h.window.run_note.text(), what="describe error")
    assert h.controller._proc.is_alive()
    h.controller.describe("fake_run")
    h.wait_for(lambda: h.window.ckpt_info.value_label("체크포인트").text() == "fake.pt",
               what="describe success")


# ================================================================ happy path
def test_start_runs_and_delivers_frames(h):
    h.start("ok:alpha", races=2, cars_per_race=2)
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    h.wait_for(lambda: h.controller.buffer.stats()["received"] > 5, what="frames")
    facts_cars = h.window._session_cars
    assert facts_cars == 4
    assert h.window.combo_focus.count() == 4
    assert h.window.viewport.geometry_data is not None
    fresh = h.controller.buffer.freshness()
    assert fresh.have_data and not fresh.is_stale()


def test_stage_progression_is_reported_during_preparation(h):
    seen = []
    orig = h.window.set_stage
    h.window.set_stage = lambda stage, note="": (seen.append(stage), orig(stage, note))
    h.start("slow:prepare")
    h.wait_for(lambda: len(seen) >= 3, what="stages")
    assert h.window.state == STATE_PREPARING
    assert seen[0] == "checkpoint"


def test_ui_stays_responsive_while_preparing(h):
    """The whole point of the process split, measured rather than asserted by design."""
    h.start("slow:prepare")
    h.wait_for(lambda: h.window.state == STATE_PREPARING, what="PREPARING")
    worst = 0.0
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline and h.window.state == STATE_PREPARING:
        t0 = time.monotonic()
        h.app.processEvents(QtCore.QEventLoop.AllEvents, 5)
        worst = max(worst, time.monotonic() - t0)
        # and a control the user might touch really does respond
        h.window.camera_buttons._pick("top")
        time.sleep(0.005)
    assert h.window.viewport.camera == "top"
    assert worst < 0.25, f"event loop blocked for {worst * 1e3:.0f} ms while preparing"


# ================================================================ pause semantics
def test_pause_is_pending_until_acked_then_paused(h):
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    h.wait_for(lambda: h.controller.buffer.stats()["received"] > 3, what="frames")

    h.window.btn_pause.click()
    # the click alone must not claim the sim stopped
    assert h.window.btn_pause.property("pending") == "true"
    assert h.window.btn_pause.is_confirmed() is False

    h.wait_for(lambda: h.window.state == STATE_PAUSED, what="PAUSED after ack")
    assert h.window.btn_pause.is_confirmed() is True
    assert h.window.btn_pause.property("pending") == "false"
    assert h.window.btn_pause.text() == "재개"

    h.pump(0.3)
    t_a = h.controller.buffer.latest["t"]
    h.pump(0.3)
    assert h.controller.buffer.latest["t"] == t_a, "sim time advanced while paused"


def test_paused_is_not_reported_as_stale_data(h):
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    h.window.btn_pause.click()
    h.wait_for(lambda: h.window.state == STATE_PAUSED, what="PAUSED")
    h.pump(1.0)                       # well past the staleness limit
    h.controller._on_tick()
    assert "일시정지" in h.window.health_line.text()
    assert "데이터 정지" not in h.window.viewport._badge.text()


def test_resume_returns_to_running(h):
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    h.window.btn_pause.click()
    h.wait_for(lambda: h.window.state == STATE_PAUSED, what="PAUSED")
    t_paused = h.controller.buffer.latest["t"]
    h.window.btn_pause.click()
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING again")
    h.wait_for(lambda: h.controller.buffer.latest["t"] > t_paused, what="time advancing again")
    assert h.window.btn_pause.text() == "일시정지"


def test_stale_data_is_flagged_only_while_running(h):
    h.start("mute:silent")            # becomes ready, then never sends anything
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    h.pump(1.2)
    h.controller._on_tick()
    fresh = h.controller.buffer.freshness()
    if fresh.have_data:
        assert fresh.is_stale()
    assert h.window.state == STATE_RUNNING       # still running, just not fresh


# ================================================================ generations
def test_a_newer_start_wins_over_an_older_one(h):
    """Two quick starts: the second must be what ends up on screen."""
    h.start("slow:prepare")
    h.wait_for(lambda: h.window.state == STATE_PREPARING, what="first PREPARING")
    first_gen = h.controller._preparing_gen
    h.pump(0.1)
    h.start("ok:beta")
    second_gen = h.controller._preparing_gen
    assert second_gen > first_gen
    h.wait_for(lambda: h.window.state == STATE_RUNNING, timeout=20, what="second RUNNING")
    assert h.controller._active_gen == second_gen
    h.wait_for(lambda: h.controller.buffer.stats()["received"] > 3, what="frames")
    assert h.controller.buffer.latest["gen"] == second_gen
    assert h.window.header_summary.text().count("ok:beta") == 1


def test_frames_from_a_superseded_generation_are_refused(h):
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    gen = h.controller._active_gen
    # RUNNING means the session is up, not that a frame has crossed the pipe yet -- on a loaded
    # machine the first one can be a moment behind, and reading `latest` too early gets None.
    h.wait_for(lambda: h.controller.buffer.latest is not None, what="a first frame")
    stale = dict(h.controller.buffer.latest)
    h.controller.buffer.set_generation(gen + 1)
    stale["gen"] = gen
    assert h.controller.buffer.push(stale) is False


def test_generation_numbers_are_never_reused(h):
    seen = set()
    for name in ("ok:alpha", "ok:beta", "ok:train1"):
        h.start(name)
        h.wait_for(lambda: h.window.state == STATE_RUNNING, what=f"RUNNING {name}")
        assert h.controller._active_gen not in seen
        seen.add(h.controller._active_gen)
    assert len(seen) == 3


# ================================================================ failures
def test_start_failure_shows_an_error_and_allows_retry(h):
    h.start("fail:boom")
    h.wait_for(lambda: h.window.state == STATE_FAILED, what="FAILED")
    assert "이 맵을 열 수 없습니다" in h.window.status_text.text()
    assert h.window.btn_retry.isVisible()
    assert h.window.btn_detail.isVisible()
    assert h.controller._proc.is_alive(), "a bad map must not take the worker down"
    # and a good session still starts afterwards
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING after failure")


def test_cancel_during_preparation_returns_to_idle(h):
    h.start("slow:prepare")
    h.wait_for(lambda: h.window.state == STATE_PREPARING, what="PREPARING")
    assert h.window.btn_cancel.isEnabled()
    h.window.btn_cancel.click()
    h.wait_for(lambda: h.window.state == STATE_IDLE, timeout=15, what="IDLE after cancel")
    assert h.controller._proc.is_alive()
    assert h.window.viewport.geometry_data is None


def test_settings_stay_editable_while_preparing(h):
    h.start("slow:prepare")
    h.wait_for(lambda: h.window.state == STATE_PREPARING, what="PREPARING")
    assert h.window.map_list.isEnabled()
    assert h.window.spin_races.isEnabled()
    assert h.window.run_list.isEnabled()
    assert h.window.btn_cancel.isEnabled()


def test_worker_death_is_noticed(h):
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    h.controller._proc.kill()
    h.wait_for(lambda: h.window.state == STATE_FAILED, timeout=10, what="FAILED after kill")
    # whichever notices first is a correct report: the pipe reader seeing EOF, or the liveness poll
    text = h.window.status_text.text()
    assert ("종료" in text) or ("연결이 끊" in text), text


# ================================================================ live commands
def test_focus_change_is_acked_and_applied(h):
    h.start("ok:alpha", races=1, cars_per_race=3)
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    h.wait_for(lambda: h.controller.buffer.stats()["received"] > 3, what="frames")
    h.window.combo_focus.setCurrentIndex(2)
    h.window._on_focus_combo(2)
    h.wait_for(lambda: h.controller.buffer.latest.get("focus_env") == 2, what="focus applied")


def test_reset_restarts_the_episode(h):
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    # RUNNING says the session is up, not that a frame has arrived; on a loaded machine the first
    # one can be a moment behind, and indexing `latest` before then raises rather than waits.
    h.wait_for(lambda: (h.controller.buffer.latest or {}).get("t", 0.0) > 0.4,
               what="some sim time")
    h.window.btn_reset.click()
    h.wait_for(lambda: h.controller.buffer.latest is not None
               and h.controller.buffer.latest["t"] < 0.4, what="time reset")


def test_expensive_overlays_are_off_until_asked_for(h):
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    h.wait_for(lambda: h.controller.buffer.stats()["received"] > 3, what="frames")
    assert "saliency" not in h.controller.buffer.latest
    h.window.chk_saliency.setChecked(True)
    h.wait_for(lambda: "saliency" in (h.controller.buffer.latest or {}), what="saliency on")
    assert h.window.viewport.point_colors is not None


def test_camera_is_local_truth_and_needs_no_worker(h):
    """A camera angle is decided here; it must not wait for a round trip that never comes."""
    h.window.camera_buttons._pick("chase")
    assert h.window.viewport.camera == "chase"
    assert h.window.camera_buttons.current() == "chase"


# ================================================================ shutdown
def test_window_close_without_worker_needs_no_launcher_wiring(h):
    h.window.close()
    h.wait_for(lambda: not h.window.isVisible(), timeout=1.0, what="window closed")
    assert h.controller._shutdown_stage == "done"


def test_window_close_stops_its_worker_without_launcher_wiring(h):
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    proc = h.controller._proc
    h.window.close()
    h.wait_for(lambda: not h.window.isVisible(), timeout=6.0, what="window closed")
    assert not proc.is_alive()


def test_staged_shutdown_does_not_block_the_event_loop(h):
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    done = []
    h.controller.shutdown_finished.connect(lambda: done.append(True))
    h.controller.begin_shutdown()
    worst = 0.0
    end = time.monotonic() + 6.0
    while time.monotonic() < end and not done:
        t0 = time.monotonic()
        h.app.processEvents(QtCore.QEventLoop.AllEvents, 5)
        worst = max(worst, time.monotonic() - t0)
        time.sleep(0.002)
    assert done, "shutdown never finished"
    assert worst < 0.25, f"event loop blocked for {worst * 1e3:.0f} ms during shutdown"
    assert not h.controller._proc.is_alive()


def test_unresponsive_worker_is_escalated_to_a_kill(h):
    """A worker that ignores `shutdown` must still be gone, and the UI must stay alive throughout."""
    h.start("hang:forever")
    h.pump(0.4)
    pid = h.controller._proc.pid
    done = []
    h.controller.shutdown_finished.connect(lambda: done.append(True))
    h.controller.begin_shutdown()
    worst = 0.0
    end = time.monotonic() + 20.0
    while time.monotonic() < end and not done:
        t0 = time.monotonic()
        h.app.processEvents(QtCore.QEventLoop.AllEvents, 5)
        worst = max(worst, time.monotonic() - t0)
        time.sleep(0.002)
    assert done, "escalation never completed"
    assert worst < 0.3, f"event loop blocked for {worst * 1e3:.0f} ms during escalation"
    assert not h.controller._proc.is_alive()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_shutdown_removes_the_instance_runtime_dir(h):
    h.controller.start_worker()
    h.wait_for(lambda: bool(h.controller._worker_hello), what="hello")
    runtime = h.controller.runtime_dir
    assert os.path.isdir(runtime)
    h.controller.shutdown()
    assert not os.path.exists(runtime)


def test_two_controllers_get_separate_runtime_dirs(qapp):
    a = Harness(qapp)
    b = Harness(qapp)
    try:
        assert a.controller.runtime_dir != b.controller.runtime_dir
        assert a.controller.instance_id != b.controller.instance_id
    finally:
        a.close()
        b.close()


# ================================================================ late and out-of-order messages
#
# The worker's replies can arrive after the console has moved on: a `ready` for a session the user
# replaced, a `stopped` for one they cancelled, an ack the worker deferred and then superseded. Each
# of these was able to corrupt the state of a *newer* request. The messages below are synthetic
# replicas of what the worker sends, injected straight into the controller's dispatch, so the
# ordering is exact and the test cannot be flaky about it.

def _ready(gen, run="fake_run", mp="ok:alpha", cars=1):
    from f1sim.viewer.console import protocol as P
    return P.message(P.MSG_READY, gen=gen, facts={
        "gen": gen, "run": run, "checkpoint": "fake.pt", "map": mp, "races": 1,
        "cars_per_race": cars, "total_cars": cars, "max_render_cars": 64,
        "car_ids": list(range(cars)), "device": "cpu", "compile": False, "speed_cap": 6.0,
        "v_max_policy": 8.0, "action_mode": "direct", "info_line": "fake", "randomize": True,
        "lidar": {"fov": 4.712, "range_max": 10.0, "n_beams": 64},
        "vehicle": {"lr": 0.15, "cog_z": 0.06, "wheel_r": 0.056}})


def _stopped(gen, cancelled=False):
    from f1sim.viewer.console import protocol as P
    return P.message(P.MSG_STOPPED, gen=gen, cancelled=cancelled)


def _ack(seq, command, gen, **state):
    from f1sim.viewer.console import protocol as P
    return P.message(P.MSG_ACK, seq=seq, command=command, gen=gen, state=state)


def test_late_stopped_does_not_erase_a_newer_pending_start(h):
    """gen2 stops late while gen3 is preparing: gen3 must still be the one we are waiting for."""
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="gen1 RUNNING")
    h.start("slow:prepare")
    h.wait_for(lambda: h.window.state == STATE_PREPARING, what="gen2 PREPARING")
    gen2 = h.controller._preparing_gen
    h.controller.cancel()
    h.wait_for(lambda: h.window.state == STATE_IDLE, timeout=15, what="IDLE after cancel")

    h.start("ok:beta")                       # gen3
    h.wait_for(lambda: h.window.state == STATE_PREPARING, what="gen3 PREPARING")
    gen3 = h.controller._preparing_gen
    assert gen3 > gen2

    h.controller._on_control(_stopped(gen2, cancelled=True))     # gen2's stop, arriving late
    assert h.controller._preparing_gen == gen3, "a late stop erased the request we are waiting for"
    assert h.window.state == STATE_PREPARING

    h.wait_for(lambda: h.window.state == STATE_RUNNING, timeout=20, what="gen3 RUNNING")
    assert h.controller._active_gen == gen3


def test_late_ready_does_not_clear_a_newer_requests_pending(h):
    """A ready for an abandoned generation must not resolve the start we are still waiting on."""
    h.start("slow:prepare")
    h.wait_for(lambda: h.window.state == STATE_PREPARING, what="PREPARING")
    gen = h.controller._preparing_gen
    pending_before = {s: dict(r) for s, r in h.controller._pending.items()}
    assert any(r["command"] == "start" and r["gen"] == gen for r in pending_before.values())

    h.controller._on_control(_ready(gen - 1))            # a generation we already moved past
    assert h.window.state == STATE_PREPARING, "a stale ready started a session"
    assert h.controller._preparing_gen == gen
    assert any(r["command"] == "start" and r["gen"] == gen
               for r in h.controller._pending.values()), "stale ready cleared our pending start"


def test_superseded_ack_does_not_touch_the_current_session(h):
    """The worker acks deferred commands with superseded=True; they must change nothing."""
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    h.wait_for(lambda: h.controller.buffer.stats()["received"] > 3, what="frames")
    gen = h.controller._active_gen
    received_before = h.controller.buffer.stats()["received"]

    h.controller._on_control(_ack(9999, "pause", gen, paused=True, superseded=True))
    assert h.window.state == STATE_RUNNING, "a superseded pause paused the live session"
    assert h.window.btn_pause.is_confirmed() is False

    h.controller._on_control(_ack(9998, "reset", gen, superseded=True))
    assert h.controller.buffer.stats()["received"] == received_before, \
        "a superseded reset cleared the live frame buffer"


def test_ack_for_an_older_generation_is_ignored(h):
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="gen1 RUNNING")
    old_gen = h.controller._active_gen
    h.start("ok:beta")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, timeout=20, what="gen2 RUNNING")
    h.wait_for(lambda: h.controller.buffer.stats()["received"] > 3, what="frames")
    before = h.controller.buffer.stats()["received"]

    h.controller._on_control(_ack(9997, "pause", old_gen, paused=True))
    assert h.window.state == STATE_RUNNING
    h.controller._on_control(_ack(9996, "reset", old_gen))
    assert h.controller.buffer.stats()["received"] == before


def test_pause_acked_after_its_session_ended_does_not_pause_the_next_one(h):
    """The exact hazard: a pause issued for gen1, answered after gen2 is on screen."""
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="gen1 RUNNING")
    gen1 = h.controller._active_gen
    seq = h.controller.send("pause", for_gen=gen1, paused=True)   # issued under gen1
    h.start("ok:beta")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, timeout=20, what="gen2 RUNNING")

    h.controller._on_control(_ack(seq, "pause", gen1, paused=True))
    assert h.window.state == STATE_RUNNING, "gen1's pause paused gen2"
    assert h.window.btn_pause.is_confirmed() is False


def test_describe_reply_for_a_superseded_selection_is_ignored(h):
    """Clicking two runs quickly must not leave the first one's metadata on screen."""
    h.controller.start_worker()
    h.wait_for(lambda: bool(h.controller._worker_hello), what="hello")
    from f1sim.viewer.console import protocol as P

    first = h.controller.send(P.CMD_DESCRIBE, run="run_a")
    h.controller._describe_seq = first
    second = h.controller.send(P.CMD_DESCRIBE, run="run_b")
    h.controller._describe_seq = second

    h.controller._on_control(P.message(P.MSG_DESCRIBED, seq=second, ok=True, info={
        "run": "run_b", "file": "b.pt", "age_s": 1.0, "progress": "B", "metrics": "-", "speed_cap": 6.0}))
    assert h.window.ckpt_info.value_label("체크포인트").text() == "b.pt"

    # run_a's reply, arriving after the user moved on
    h.controller._on_control(P.message(P.MSG_DESCRIBED, seq=first, ok=True, info={
        "run": "run_a", "file": "a.pt", "age_s": 1.0, "progress": "A", "metrics": "-", "speed_cap": 6.0}))
    assert h.window.ckpt_info.value_label("체크포인트").text() == "b.pt", \
        "a late describe overwrote the current selection"


# ================================================================ replacement has no rollback
def test_cancelling_a_replacement_leaves_nothing_running(h):
    """gen1 running -> gen2 preparing -> cancel: idle, with no simulation behind it.

    This version has no rollback: the worker stops the old session when a replacement is requested
    and does not resume it. The console has to end up genuinely idle -- empty view, no frames -- and
    say so, rather than looking idle while something still runs.
    """
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="gen1 RUNNING")
    h.wait_for(lambda: h.controller.buffer.stats()["received"] > 3, what="gen1 frames")

    h.start("slow:prepare")                                   # gen2 replaces it
    h.wait_for(lambda: h.window.state == STATE_PREPARING, what="gen2 PREPARING")
    assert h.controller._replacing is True
    h.window.btn_cancel.click()
    h.wait_for(lambda: h.window.state == STATE_IDLE, timeout=15, what="IDLE after cancel")

    assert h.window.viewport.geometry_data is None, "the old map is still on screen"
    assert h.controller._active_gen == -1 and h.controller._preparing_gen == -1
    assert "이전 세션도" in h.window.status_text.text(), h.window.status_text.text()

    settled = h.controller.buffer.stats()["received"]
    h.pump(0.6)
    assert h.controller.buffer.stats()["received"] == settled, "frames still arriving after cancel"
    assert h.controller.buffer.latest is None


def test_failed_replacement_leaves_nothing_running(h):
    h.start("ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="gen1 RUNNING")
    h.start("fail:boom")
    h.wait_for(lambda: h.window.state == STATE_FAILED, timeout=15, what="FAILED")
    assert h.window.viewport.geometry_data is None
    assert h.controller._active_gen == -1
    assert "이전 세션도" in h.window.status_text.text(), h.window.status_text.text()
    settled = h.controller.buffer.stats()["received"]
    h.pump(0.6)
    assert h.controller.buffer.stats()["received"] == settled


def test_spawn_bootstrap_failure_is_explained_in_korean():
    """`spawn` re-imports __main__; an unguarded caller gets a paragraph of English otherwise.

    The shipped entry points are guarded, so this only reaches someone embedding the console -- but
    when it does, the message should name the one-line fix rather than paste multiprocessing's
    lecture into a Korean status bar.
    """
    exc = RuntimeError(
        "An attempt has been made to start a new process before the current process has finished "
        "its bootstrapping phase. ... you have forgotten to use the proper idiom ... freeze_support()")
    text = SessionController._spawn_error_text(exc)
    assert "__main__" in text
    assert "spawn" in text
    assert "freeze_support" not in text, "the raw lecture should not be shown"
    # anything else is passed through unchanged
    assert "boom" in SessionController._spawn_error_text(RuntimeError("boom"))


# --------------------------------------------------------------------- the window's own ledger
#
# The window stamps `_pending['start']` when the 시작 button is pressed, and `tick_pending` turns
# that stamp into "worker 응답 지연 N초 — start 명령을 아직 확인하지 못했습니다". The controller
# resolved the *controller's* ledger and then acked the button, which is a different thing: the
# window's entry stayed, and a session that had been running for a minute still reported its start
# as unanswered. Every test above drives `controller.start_session` directly, so none of them ever
# created a window entry -- which is exactly why the bug survived them and showed up in a
# screenshot instead.

def _press_start(h, map_name="ok:alpha"):
    """Start the way the UI does: select in the lists, then press the button."""
    h.window.run_list.select("fake_run")
    h.window.map_list.select(map_name)
    h.pump(0.1)
    if not h.window.btn_start.isEnabled():
        # the fake catalogue may not carry this run; set the selection directly, still going
        # through _on_start so the window ledger is stamped
        h.window._selected_run, h.window._selected_map = "fake_run", map_name
        h.window._update_start_enabled()
        h.pump(0.05)
    h.window._on_start()
    return h.controller._preparing_gen


def test_button_started_session_does_not_report_a_late_worker(h):
    """The reported symptom: RUNNING, yet the footer says the start was never confirmed."""
    _press_start(h)
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    assert "start" not in h.window._pending, (
        "the window still lists 'start' as unanswered while the session is running")

    # advance past the warning threshold and pump: nothing should complain
    h.window._pending = {k: v - (ACK_WARN_S + 5.0) for k, v in h.window._pending.items()}
    h.window.tick_pending()
    h.pump(0.1)
    assert "응답 지연" not in h.window.status_text.text(), h.window.status_text.text()


def test_button_started_stop_clears_the_windows_stop_entry(h):
    """정지 has its own entry, and an accepted STOPPED is what resolves it."""
    _press_start(h)
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="RUNNING")
    h.window._on_stop()
    assert "stop" in h.window._pending
    h.wait_for(lambda: h.window.state == STATE_IDLE, what="IDLE after stop")
    assert "stop" not in h.window._pending
    h.window._pending = {k: v - (ACK_WARN_S + 5.0) for k, v in h.window._pending.items()}
    h.window.tick_pending()
    assert "응답 지연" not in h.window.status_text.text(), h.window.status_text.text()


def test_resolving_an_older_start_leaves_a_newer_one_pending(h):
    """Generation-correct: gen1 finishing must not silence the warning about gen2."""
    _press_start(h, "ok:alpha")
    h.wait_for(lambda: h.window.state == STATE_RUNNING, what="gen1 RUNNING")
    gen1 = h.controller._active_gen

    _press_start(h, "slow:prepare")                      # gen2, still preparing
    h.wait_for(lambda: h.window.state == STATE_PREPARING, what="gen2 PREPARING")
    assert "start" in h.window._pending, "the button press should stamp the window ledger"

    h.controller._clear_start_pending(gen1)              # gen1's late resolution
    assert "start" in h.window._pending, (
        "resolving the older generation cleared the warning about the one still preparing")

    h.wait_for(lambda: h.window.state == STATE_RUNNING, timeout=20, what="gen2 RUNNING")
    assert "start" not in h.window._pending
