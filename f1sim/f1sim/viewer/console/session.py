"""The controller: owns the worker process and translates between it and the window.

Everything that could make the UI lie or stall lives here, so it is all in one place:

* **Generations are allocated here, monotonically, and never reused.** A `start` gets the next
  number; anything the worker sends tagged with an older one is dropped. That is what stops the
  second of two quick map switches from being overwritten by the first one's late frames.
* **Acks, not optimism.** A command goes out with a sequence number and the control it came from
  stays "적용 중" until the worker answers with that number. Pause in particular: the session is
  RUNNING with a pending pause until the ack lands, because until then the simulation really is
  still stepping and the picture really should keep following it.
* **Nothing here blocks the Qt event loop.** Both pipes are read by threads; frames land in a
  bounded drop-oldest slot and the main thread drains it. If the worker hangs, the reader threads
  hang and the UI carries on saying so.
* **This process kills only the process it started.** No pattern matching over the process table,
  no shared control file; the runtime directory is per instance.
"""
from __future__ import annotations

import multiprocessing
import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt5 import QtCore

from . import catalog, protocol as P
from .catalog import MapCatalog
from .frames import FrameBuffer
from .protocol import (LatestSlot, SessionConfig, STATE_FAILED, STATE_IDLE, STATE_PAUSED,
                       STATE_PREPARING, STATE_RUNNING, STATE_STOPPING)
from .viewport import TrackGeometry

#: How long to wait for the worker to leave on its own before escalating.
SHUTDOWN_GRACE_S = 8.0
TERMINATE_GRACE_S = 3.0

#: The policy/dash panels are diagnostics, redrawn on their own clock rather than at the frame
#: rate. At 40 frames a second they were repainting 25 times a second and taking about a third of
#: the GUI thread with them -- more than the 3D scene they annotate. Twelve a second is faster than
#: anyone reads a gauge, and the 3D view keeps following every frame.
OVERLAY_HZ = 12.0

#: Inbound frame slot in the GUI process. Same reasoning as the worker's: bounded and drop-oldest,
#: so a busy main thread costs freshness rather than growing latency for the rest of the session.
INBOUND_CAPACITY = 4


def _plan_in_body_frame(fr: dict):
    """The plan, rotated into the focus car's frame for the 2D panel.

    The frame carries it in world coordinates because that is what the 3D scene draws. The panel
    draws the car at the origin looking up, so the same polyline has to come back to body frame --
    and doing it here means the worker sends one representation, not two.
    """
    plan = fr.get("plan")
    if plan is None or len(plan) < 2:
        return None
    f, n = int(fr.get("focus", 0)), int(fr.get("n", 0))
    if not (0 <= f < n):
        return None
    try:
        x, y, yaw = float(fr["x"][f]), float(fr["y"][f]), float(fr["yaw"][f])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    p = np.asarray(plan, np.float32)
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    dx, dy = p[:, 0] - x, p[:, 1] - y
    bx = dx * c - dy * s_
    by = dx * s_ + dy * c
    speed = p[:, 2] if p.shape[1] > 2 else np.zeros(len(p), np.float32)
    # (forward, left, _, speed): the layout PolicyInputPanel reads
    return np.stack([bx, by, np.zeros_like(bx), speed], 1).astype(np.float32)


def _saliency_colors(sal) -> np.ndarray:
    """Per-beam attention as brightness on one hue (see `overlays.PolicyInputPanel`)."""
    t = np.clip(np.asarray(sal, np.float32), 0.0, 1.0)[:, None]
    lo = np.array([0.26, 0.34, 0.46], np.float32)
    hi = np.array([1.00, 1.00, 0.98], np.float32)
    rgb = lo + (hi - lo) * (t ** 0.7)
    return np.concatenate([rgb, 0.45 + 0.55 * t], 1).astype(np.float32)


class _Reader(QtCore.QThread):
    """Blocking reads off one pipe, handed to the main thread as Qt signals.

    Control messages are forwarded individually -- they are rare and each one matters. Frames go
    through a bounded slot and only a wake-up is signalled, so a main thread that falls behind
    drains what is current instead of working through a backlog of stale pictures.
    """
    control_message = QtCore.pyqtSignal(dict)
    frames_available = QtCore.pyqtSignal()
    closed = QtCore.pyqtSignal(str)

    def __init__(self, conn, slot: Optional[LatestSlot] = None, name: str = "reader", parent=None):
        super().__init__(parent)
        self.setObjectName(name)
        self._conn = conn
        self._slot = slot
        self._stop = threading.Event()
        self._pending_signal = threading.Event()

    def run(self):
        reason = "닫힘"
        try:
            while not self._stop.is_set():
                if not self._conn.poll(0.2):
                    continue
                msg = self._conn.recv()
                if self._slot is not None:
                    self._slot.put(msg)
                    if not self._pending_signal.is_set():
                        self._pending_signal.set()
                        self.frames_available.emit()
                else:
                    self.control_message.emit(msg)
        except (EOFError, BrokenPipeError, ConnectionResetError):
            reason = "worker 연결이 끊겼습니다."
        except OSError as exc:
            reason = f"파이프 오류: {exc}"
        except Exception as exc:                       # never take the process down from a thread
            reason = f"{type(exc).__name__}: {exc}"
        if not self._stop.is_set():
            self.closed.emit(reason)

    def rearm(self):
        self._pending_signal.clear()

    def stop(self):
        self._stop.set()


#: Vertex arrays every prop batch must carry. A batch missing one of them cannot be uploaded, and
#: dropping it is better than handing GL a half-built mesh -- see `_prop_batches`.
_PROP_ARRAYS = ("pos", "nrm", "col", "idx")


class PropPayloadError(ValueError):
    """A prop batch that cannot be drawn. Fatal to the session, deliberately -- see `_prop_batches`."""


def _prop_batches(raw):
    """Validate the worker's merged prop batches into what the viewport can upload.

    The worker sends these already transformed into world space and merged by material, so there is
    nothing to build here. What there is to do is check them, because the payload crosses a process
    boundary and a wrongly-shaped array becomes a GL error inside `paintGL`.

    **A batch that fails validation raises.** Dropping it would leave the simulator colliding with
    obstacles the LiDAR still returns and the screen does not show -- the invisible collider that
    `StaticProp` exists to prevent. The caller turns this into a refused session, which is the only
    outcome that does not put a car on a track with objects the driver cannot see.

    No props at all is a different thing and is perfectly valid: `None` in, `None` out.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise PropPayloadError(f"프롭 배치 목록이 아닙니다: {type(raw).__name__}")
    if len(raw) == 0:
        return None                       # a track that places nothing
    out = []
    for i, b in enumerate(raw):
        def bad(why):
            return PropPayloadError(f"프롭 배치 {i} ({(b or {}).get('material', '?')!r}) 를 그릴 수 "
                                    f"없습니다: {why}")
        if not isinstance(b, dict):
            raise PropPayloadError(f"프롭 배치 {i} 가 dict 가 아닙니다: {type(b).__name__}")
        missing = [k for k in _PROP_ARRAYS if b.get(k) is None]
        if missing:
            raise bad(f"배열 누락 {missing}")
        pos, nrm, col, idx = (np.asarray(b[k]) for k in _PROP_ARRAYS)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise bad(f"pos 모양 {pos.shape}, (V,3) 이어야 합니다")
        if nrm.shape != pos.shape:
            raise bad(f"nrm 모양 {nrm.shape} 이 pos {pos.shape} 와 다릅니다")
        if col.ndim != 2 or col.shape[0] != pos.shape[0] or col.shape[1] != 4:
            raise bad(f"col 모양 {col.shape}, ({pos.shape[0]},4) 이어야 합니다")
        if idx.ndim != 1 or len(idx) % 3:
            raise bad(f"idx 길이 {len(idx)} 가 삼각형 단위가 아닙니다")
        if len(idx):
            # Both ends. A negative index is not "out of range" to numpy -- it wraps -- but GL
            # takes it as an enormous unsigned one, so checking only the maximum lets a batch
            # through that draws nothing or draws garbage. Either way the obstacle is in the
            # simulation and not on the screen, which is the case this guard exists for.
            if int(idx.min()) < 0:
                raise bad(f"idx 최소값 {int(idx.min())} 이 음수입니다")
            if int(idx.max()) >= len(pos):
                raise bad(f"idx 최대값 {int(idx.max())} 이 정점 수 {len(pos)} 를 넘습니다")
        # A NaN position collapses its triangle; an infinite one throws it off the map. Either way
        # the prop stops being visible while the simulator keeps colliding with it.
        for name, arr in (("pos", pos), ("nrm", nrm), ("col", col)):
            if not np.isfinite(np.asarray(arr, np.float64)).all():
                raise bad(f"{name} 에 NaN/Inf 가 있습니다")
        out.append({"material": str(b.get("material", "plastic")),
                    "pos": pos, "nrm": nrm, "col": col, "idx": idx,
                    "n_props": int(b.get("n_props", 0)),
                    "n_tris": int(b.get("n_tris", len(idx) // 3))})
    return tuple(out)


class SessionController(QtCore.QObject):
    """Wires one `ConsoleWindow` to one worker process."""

    shutdown_finished = QtCore.pyqtSignal()

    def __init__(self, window, parent=None, worker_target=None):
        """`worker_target`: the process entry point, for tests that need a deterministic worker.

        It must be importable by name in a spawned child (`multiprocessing` pickles it by module
        path), and must have the signature `main(control, frames, log_path=None)`. Default is the
        real simulation worker.
        """
        super().__init__(parent)
        self.window = window
        self._worker_target = worker_target
        self.buffer = FrameBuffer(clock=None)
        self.window.viewport.attach(self.buffer)

        self.instance_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.runtime_dir = os.path.join(
            os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir(),
            "f1sim-console", self.instance_id)
        os.makedirs(self.runtime_dir, exist_ok=True)

        self._proc = None
        self._ctl = None
        self._frames = None
        self._ctl_reader: Optional[_Reader] = None
        self._frame_reader: Optional[_Reader] = None
        self._frame_slot = LatestSlot(INBOUND_CAPACITY)

        self._seq = 0
        self._generation = 0            # allocated here, monotonic, never reused
        self._active_gen = -1           # the generation currently on screen
        self._preparing_gen = -1
        self._pending: Dict[int, str] = {}          # seq -> command
        self._last_config: Optional[SessionConfig] = None
        self._describe_seq = -1
        self._describe_run = ""
        self._maps = MapCatalog()
        self._worker_hello: Dict[str, Any] = {}
        self.last_facts: Dict[str, Any] = {}
        #: Generations refused because their map could not be drawn faithfully. Kept only until the
        #: worker's `stopped` for each arrives, so a late `ready` can be refused and the pending
        #: ledgers resolved -- never long enough to affect a newer start.
        self._rejected_gens: set = set()
        #: True while a start is replacing a session that was already up. The backend stops the old
        #: session as soon as a replacement is requested and does not resume it if the replacement
        #: is cancelled or fails, so the console says so plainly instead of leaving someone to
        #: wonder whether the previous run is still going behind an IDLE window.
        self._replacing = False
        self._geometry_bytes = 0
        self._last_geometry_ms = 0.0
        self._last_overlay = 0.0
        self._shutdown_stage: Optional[str] = None
        self._shutdown_t0 = 0.0
        self._shutdown_timer = QtCore.QTimer(self)
        self._shutdown_timer.timeout.connect(self._shutdown_step)

        self._connect_window()

        # One timer for everything periodic. Telemetry at 10 Hz is plenty for numbers a human
        # reads, and keeping it off the render path means a slow label never costs a frame.
        self._tick = QtCore.QTimer(self)
        self._tick.setInterval(100)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()

    # ================================================================ wiring
    def _connect_window(self):
        w = self.window
        # Every window/controller pair must close, including embedded and capture launchers.
        w.close_requested.connect(self.begin_shutdown)
        # A missing/dead worker finishes synchronously: wait until closeEvent has returned.
        self.shutdown_finished.connect(w.allow_close, type=QtCore.Qt.QueuedConnection)
        w.start_requested.connect(self.start_session)
        w.cancel_requested.connect(self.cancel)
        w.stop_requested.connect(self.stop_session)
        w.pause_requested.connect(self.set_paused)
        w.reset_requested.connect(self.reset)
        w.focus_requested.connect(self.set_focus)
        w.overlay_requested.connect(self.set_overlay)
        w.describe_requested.connect(self.describe)
        w.retry_requested.connect(self.retry)
        w.viewport.frame_timed.connect(w.push_frame_time)
        w.viewport.gl_ready.connect(w.set_gl_info)
        w.viewport.gl_failed.connect(self._on_gl_failed)
        w.viewport.upload_timed.connect(self._on_upload_timed)

    # ================================================================ worker lifecycle
    def start_worker(self) -> bool:
        """Spawn the worker and say hello. Returns False if it could not be started."""
        if self._proc is not None and self._proc.is_alive():
            return True
        try:
            ctx = multiprocessing.get_context("spawn")
            self._ctl, ctl_theirs = ctx.Pipe(duplex=True)
            # `Pipe(duplex=False)` returns (receive_end, send_end) -- verified, the first end
            # raises "connection is read-only" on send. The console reads frames, the worker
            # writes them, so the console keeps the first and hands over the second.
            self._frames, frames_theirs = ctx.Pipe(duplex=False)
            target = self._worker_target
            if target is None:
                from .. import sim_worker
                target = sim_worker.main
            self._proc = ctx.Process(target=target, args=(ctl_theirs, frames_theirs),
                                     kwargs={"log_path": os.path.join(self.runtime_dir, "worker.log")},
                                     name="f1sim-sim-worker", daemon=False)
            self._proc.start()
            ctl_theirs.close()
            frames_theirs.close()
        except Exception as exc:
            self.window.set_error("worker 시작", self._spawn_error_text(exc),
                                  traceback.format_exc(), retryable=True)
            return False

        self._ctl_reader = _Reader(self._ctl, None, "ctl-reader", self)
        self._ctl_reader.control_message.connect(self._on_control)
        self._ctl_reader.closed.connect(self._on_pipe_closed)
        self._ctl_reader.start()

        self._frame_reader = _Reader(self._frames, self._frame_slot, "frame-reader", self)
        self._frame_reader.frames_available.connect(self._drain_frames)
        self._frame_reader.start()

        self.window.status_text.setText(f"시뮬레이터 프로세스를 시작했습니다 (pid {self._proc.pid}).")
        self.send(P.CMD_HELLO)
        self.send(P.CMD_LIST_MAPS)
        return True

    # ---------------------------------------------------------------- shutdown
    def begin_shutdown(self):
        """Start closing down, without blocking the event loop for a moment of it.

        A worker mid-way through a CUDA graph capture can take seconds to notice it has been asked
        to leave, and `join(timeout=8)` on the GUI thread would freeze the window for those
        seconds -- the exact failure this rebuild exists to remove. So the escalation runs as a
        timer-driven state machine: ask, wait, terminate, wait, kill. The window keeps repainting
        and the status bar keeps saying what stage it is at.

        `shutdown_finished` fires when the worker is gone (or was never there).
        """
        if self._shutdown_stage is not None:
            return
        self._tick.stop()
        self._shutdown_stage = "asked"
        self._shutdown_t0 = time.monotonic()
        self.window.apply_state(STATE_STOPPING)
        if self._proc is None or not self._proc.is_alive():
            self._finish_shutdown("worker 없음")
            return
        try:
            self.send(P.CMD_SHUTDOWN)
        except Exception:
            pass
        self._shutdown_timer.start(100)

    def _shutdown_step(self):
        proc = self._proc
        if proc is None or not proc.is_alive():
            self._finish_shutdown("정상 종료")
            return
        dt = time.monotonic() - self._shutdown_t0
        if self._shutdown_stage == "asked":
            self.window.status_text.setText(f"시뮬레이터 종료를 기다리는 중… {dt:.0f}초")
            if dt > SHUTDOWN_GRACE_S:
                self._shutdown_stage = "terminated"
                self.window.status_text.setText(
                    f"시뮬레이터가 {SHUTDOWN_GRACE_S:.0f}초 안에 끝나지 않아 종료 신호를 보냅니다.")
                proc.terminate()               # our own handle -- never a name or pattern match
        elif self._shutdown_stage == "terminated":
            if dt > SHUTDOWN_GRACE_S + TERMINATE_GRACE_S:
                self._shutdown_stage = "killed"
                self.window.status_text.setText("응답이 없어 시뮬레이터 프로세스를 강제 종료합니다.")
                proc.kill()
        elif self._shutdown_stage == "killed" and dt > SHUTDOWN_GRACE_S + TERMINATE_GRACE_S + 3.0:
            self._finish_shutdown("강제 종료 후에도 응답 없음")

    def _finish_shutdown(self, how: str):
        self._shutdown_timer.stop()
        self._shutdown_stage = "done"
        for reader in (self._ctl_reader, self._frame_reader):
            if reader is not None:
                reader.stop()
        proc = self._proc
        if proc is not None:
            try:
                proc.join(timeout=0.1)        # already dead, or we are past caring
            except Exception:
                pass
        for reader in (self._ctl_reader, self._frame_reader):
            if reader is not None:
                reader.wait(1500)
        for conn in (self._ctl, self._frames):
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
        shutil.rmtree(self.runtime_dir, ignore_errors=True)
        self.window.status_text.setText(f"종료했습니다 ({how}).")
        self.shutdown_finished.emit()

    @staticmethod
    def _spawn_error_text(exc: Exception) -> str:
        """Turn a spawn failure into something the person reading it can act on.

        The one that actually happens: `spawn` re-imports the parent's `__main__` in the child, so a
        script that calls `launch()` at module level tries to start the worker again from inside the
        worker, and multiprocessing stops it with a paragraph of English about `freeze_support()`.
        The shipped entry points are guarded, so this only reaches someone embedding the console --
        and when it does, the fix is one line and worth naming instead of pasting the lecture into a
        Korean status bar.
        """
        text = str(exc)
        if "bootstrapping phase" in text or "freeze_support" in text:
            return ("시뮬레이터 프로세스를 띄우지 못했습니다: 콘솔을 여는 스크립트가 "
                    "`if __name__ == \"__main__\":` 안에서 실행되지 않았습니다. "
                    "worker는 spawn으로 뜨고, spawn은 자식에서 부모 스크립트를 다시 import하므로 "
                    "모듈 최상위에서 launch()를 부르면 무한히 프로세스를 만들게 됩니다. "
                    "`python -m f1sim.learn.watch` / `python -m f1sim.viewer.console` 로 실행하거나, "
                    "직접 부를 때는 그 가드 안에 넣으세요.")
        return f"시뮬레이터 프로세스를 띄우지 못했습니다: {exc}"

    def shutdown(self):
        """Synchronous last resort, for `aboutToQuit` where there is no event loop left to drive
        the staged path. Kept short on purpose: by the time Qt is quitting, `begin_shutdown` has
        usually already finished the job."""
        if self._shutdown_stage == "done":
            return
        self._tick.stop()
        self._shutdown_timer.stop()
        for reader in (self._ctl_reader, self._frame_reader):
            if reader is not None:
                reader.stop()
        proc = self._proc
        if proc is not None and proc.is_alive():
            try:
                self.send(P.CMD_SHUTDOWN)
            except Exception:
                pass
            proc.join(timeout=1.5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        for reader in (self._ctl_reader, self._frame_reader):
            if reader is not None:
                reader.wait(1000)
        for conn in (self._ctl, self._frames):
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
        shutil.rmtree(self.runtime_dir, ignore_errors=True)
        self._shutdown_stage = "done"

    def _on_pipe_closed(self, reason: str):
        if self.window.state in (STATE_RUNNING, STATE_PAUSED, STATE_PREPARING):
            self.window.set_error("worker", reason,
                                  "worker 로그: " + os.path.join(self.runtime_dir, "worker.log"),
                                  retryable=True)
        self._active_gen = -1
        self._preparing_gen = -1

    # ================================================================ sending
    def send(self, kind: str, for_gen: Optional[int] = None, **fields) -> int:
        """Send a command, remembering which generation it belongs to.

        The generation travels with the pending entry because a reply has to be judged against the
        session the command was *issued* for, not against whatever is current when it lands. A
        pause acked after the user has already started a different session must not pause that one.
        """
        self._seq += 1
        seq = self._seq
        msg = P.message(kind, seq=seq, **fields)
        self._pending[seq] = {"command": kind, "gen": for_gen, "sent": time.monotonic()}
        try:
            self._ctl.send(msg)
        except (BrokenPipeError, OSError) as exc:
            self._pending.pop(seq, None)
            self.window.set_error("worker", f"명령을 보내지 못했습니다: {exc}", "", retryable=True)
        return seq

    def _resolve(self, seq) -> Optional[dict]:
        """Take the pending record for a reply that has been accepted."""
        return self._pending.pop(seq, None) if seq is not None else None

    def _session_gen(self) -> int:
        """The generation any session-scoped reply must match to be worth applying."""
        return self._preparing_gen if self._preparing_gen >= 0 else self._active_gen

    # ================================================================ commands from the window
    def start_session(self, cfg: SessionConfig):
        if not self.start_worker():
            return
        # Starting replaces whatever was up. The worker stops the old session immediately and does
        # not bring it back if this one is cancelled or fails -- there is no rollback in this
        # version, and pretending otherwise would leave someone believing a run is still going.
        self._replacing = self._active_gen >= 0 or self._preparing_gen >= 0
        self._generation += 1                  # monotonic; a generation number is never reused
        gen = self._generation
        self._preparing_gen = gen
        self._last_config = cfg
        self.window.clear_error()
        self.window.apply_state(STATE_PREPARING, "checkpoint")
        self.buffer.set_generation(gen)
        self._teardown_view()
        self.send(P.CMD_START, for_gen=gen, gen=gen, config=cfg.to_dict())
        if self._replacing:
            self.window.status_text.setText(
                "이전 세션을 종료하고 새 세션을 준비합니다. 취소하거나 실패하면 이전 세션으로 "
                "돌아가지 않고 대기 상태가 됩니다.")

    def cancel(self):
        if self._preparing_gen < 0:
            return
        self.window.status_text.setText("취소를 요청했습니다. worker가 현재 단계를 끝내고 멈춥니다.")
        self.send(P.CMD_CANCEL, for_gen=self._preparing_gen, gen=self._preparing_gen)

    def stop_session(self):
        if self._proc is None or not self._proc.is_alive():
            self.window.apply_state(STATE_IDLE)
            return
        self.window.apply_state(STATE_STOPPING)
        self.send(P.CMD_STOP, for_gen=self._session_gen())

    def set_paused(self, paused: bool):
        # Deliberately does NOT touch the state here. The simulation is still stepping until the
        # worker says otherwise, so the picture keeps following it and the button stays pending.
        self.send(P.CMD_PAUSE, for_gen=self._active_gen, paused=bool(paused))

    def reset(self):
        self.send(P.CMD_RESET, for_gen=self._active_gen)

    def set_focus(self, env: int):
        self.send(P.CMD_FOCUS, for_gen=self._active_gen, env=int(env))

    def set_overlay(self, overlay: dict):
        if self._proc is None or not self._proc.is_alive():
            return
        self.send(P.CMD_OVERLAY, for_gen=self._active_gen, overlay=overlay)

    def describe(self, run: str):
        if not self.start_worker():
            return
        self._describe_run = run
        self._describe_seq = self.send(P.CMD_DESCRIBE, run=run)

    def retry(self):
        if self._last_config is not None:
            self.start_session(self._last_config)
        else:
            self.window.apply_state(STATE_IDLE)

    # ================================================================ messages from the worker
    def _on_control(self, msg: dict):
        kind = msg.get("kind")
        seq = msg.get("seq")
        # Only replies that are unconditionally terminal for their request are resolved here.
        # `ready` / `error` / `stopped` / `ack` decide for themselves, because a stale one must not
        # be allowed to clear a newer request's pending entry.
        if kind in (P.MSG_HELLO, P.MSG_MAPS):
            record = self._resolve(seq)
            if record is not None and kind == P.MSG_HELLO:
                # the round trip also tells us how long the worker took to answer
                self.buffer.clock.observe(record["sent"], float(msg.get("monotonic", 0.0)),
                                          time.monotonic())

        if kind == P.MSG_HELLO:
            self._worker_hello = dict(msg)
            dev = msg.get("device_name", "?")
            self.window.status_text.setText(
                f"시뮬레이터 준비됨 — pid {msg.get('pid')}, torch {msg.get('torch')}, "
                f"{'CUDA: ' + str(dev) if msg.get('cuda') else 'CPU'}")
            if msg.get("runs_dir") and msg["runs_dir"] != catalog.RUNS_DIR:
                self.window.status_text.setText(
                    f"경고: worker의 런 경로({msg['runs_dir']})가 콘솔이 읽은 경로"
                    f"({catalog.RUNS_DIR})와 다릅니다.")
        elif kind == P.MSG_MAPS:
            self._maps = MapCatalog(groups=msg.get("groups") or {}, ready=not msg.get("error"),
                                    error=msg.get("error"))
            self.window.set_maps(self._maps)
        elif kind == P.MSG_DESCRIBED:
            self._on_described(msg, seq)
        elif kind == P.MSG_STAGE:
            if msg.get("gen") == self._preparing_gen:
                self.window.set_stage(msg.get("stage", ""), msg.get("note", ""))
        elif kind == P.MSG_GEOMETRY:
            if msg.get("gen") == self._preparing_gen:
                self._apply_geometry(msg["geometry"])
        elif kind == P.MSG_READY:
            self._on_ready(msg)
        elif kind == P.MSG_ACK:
            self._on_ack(msg)
        elif kind == P.MSG_ERROR:
            self._on_error(msg)
        elif kind == P.MSG_STOPPED:
            self._on_stopped(msg)
        elif kind == P.MSG_LOG:
            self.window.status_text.setText(str(msg.get("text", "")))
        elif kind == P.MSG_BYE:
            self._active_gen = -1

    def _apply_geometry(self, g: dict):
        """Turn one `build_geometry` payload into what the viewport draws.

        Everything the worker added in geometry v2 is read with `.get`, and the version is taken
        from the payload rather than assumed: a message with no `geometry_version` is v1, which is
        what an older worker sends, and it must still produce a drawable map.
        """
        pres = g.get("presentation")
        cb = g.get("content_bounds")
        try:
            props = _prop_batches(g.get("props"))
        except PropPayloadError as exc:
            # Not a drawing problem to work around: the obstacles are in the simulation whether or
            # not we can draw them. Refuse the session rather than run one whose track contains
            # things the screen does not show.
            self._reject_geometry(g, str(exc))
            return
        geom = TrackGeometry(
            name=g.get("name", ""), bounds=tuple(g["bounds"]),
            duct_height=float(g.get("duct_height", 0.2)),
            floor=g.get("floor"), ducts=g.get("ducts"), walls=g.get("walls"),
            centerline=g.get("centerline"), raceline_xy=g.get("raceline_xy"),
            raceline_v=g.get("raceline_v"), build_ms=float(g.get("build_ms", 0.0)),
            geometry_version=int(g.get("geometry_version", 1)),
            backdrop=g.get("backdrop"),
            content_bounds=tuple(cb) if cb is not None else None,
            presentation=dict(pres) if isinstance(pres, dict) else None,
            props=props)
        self._geometry_bytes = geom.nbytes()
        self._last_geometry_ms = geom.build_ms
        self.window.viewport.set_geometry(geom)

    def _reject_geometry(self, g: dict, why: str):
        """Refuse a map we cannot draw faithfully, and stop the session that carries it.

        Routed through the paths that already exist -- the worker is told to stop, the view is torn
        down, and the error is shown where every other preparation failure is shown -- so there is
        no second way for a session to end.

        The generation has to be *resolved*, not merely forgotten. Clearing `_preparing_gen` and
        `_active_gen` first would leave the `stopped` that this very method asks for failing the
        guard in `_on_stopped` -- so neither the start nor the stop pending record would ever be
        cleared, and the console would sit in FAILED reporting a command the worker had already
        answered. So: resolve the ledgers now, and remember the generation so its termination is
        consumed when it arrives and its `ready` is refused if it races us.
        """
        gen = self._preparing_gen if self._preparing_gen >= 0 else self._active_gen
        self.window.set_error(
            "맵 지오메트리",
            f"맵 '{g.get('name', '?')}' 의 정적 장애물을 그릴 수 없어 세션을 중단했습니다.",
            why + "\n\n이 장애물들은 시뮬레이터와 LiDAR 에는 존재합니다. 그리지 못한 채로 "
                  "주행하면 화면에 보이지 않는 충돌체가 되므로, 그리는 대신 멈춥니다.",
            True)
        if gen >= 0:
            self._clear_start_pending(gen)          # this generation will never become ready
            self.send(P.CMD_STOP, for_gen=gen, gen=gen)
            self._rejected_gens.add(gen)
        self._preparing_gen = -1
        self._active_gen = -1
        self._replacing = False
        self._teardown_view()
        self.window.apply_state(STATE_FAILED)

    def _consume_rejected(self, kind: str, gen: int) -> bool:
        """Did this message belong to a generation we refused? Then it is answered, not ignored.

        Only ever true for a generation this console rejected, so a newer start in flight is
        untouched: its own generation is not in the set.
        """
        if gen < 0 or gen not in self._rejected_gens:
            return False
        if kind == "stopped":
            self._rejected_gens.discard(gen)
            # The start ledger for this generation was already resolved when we rejected it; the
            # stop ledger is this message's to clear -- but only if nothing newer is in flight.
            # Clearing it unconditionally wiped the pending record of a start the user had just
            # made, which is the same generation mistake this console has made before.
            if self._preparing_gen < 0 and self._active_gen < 0:
                self.window.note_ack("stop")
            # the error and the FAILED state stay: the user asked why, and this is the answer
        return True

    def _clear_start_pending(self, gen: int):
        """Resolve the `start` request for one generation.

        `start` is answered by ready / error / stopped, not by an ack, so its pending entry has to
        be cleared by hand -- but only its own. Clearing every start ever sent let a late reply for
        an abandoned session silence the "waiting for the worker" warning about the one the user is
        actually waiting for.
        """
        for seq, rec in list(self._pending.items()):
            if rec["command"] == P.CMD_START and rec.get("gen") == gen:
                self._pending.pop(seq, None)
        # The window keeps a ledger of its own, stamped when the 시작 button was pressed, and that
        # is what produces the "worker 응답 지연" line. Acking the button alone left the entry in
        # place, so a session that had been running for a minute still reported the start command
        # as unanswered. Tests that called `start_session` directly never saw it: they never went
        # through `Window._on_start`, so there was no window entry to leave behind.
        #
        # Only when no newer start is in flight: resolving generation 4 must not silence the
        # warning about generation 5, which is still genuinely waiting.
        newer = any(r["command"] == P.CMD_START and (r.get("gen") or -1) > gen
                    for r in self._pending.values())
        if newer:
            self.window.btn_start.ack()
        else:
            self.window.note_ack("start")

    def _ignore(self, what: str, gen, why: str):
        """A late message that must not touch the current session. Say so rather than act."""
        self.window.status_text.setText(f"이전 세션(#{gen})의 {what}는 무시했습니다 — {why}")

    def _on_described(self, msg: dict, seq):
        """Checkpoint metadata, but only for the run currently selected.

        Clicking through runs quickly issues several of these. Without this guard the last reply to
        arrive wins, which is not the same as the last run picked, and the panel ends up describing
        a checkpoint the user is no longer looking at.
        """
        self._resolve(seq)
        if seq is not None and self._describe_seq >= 0 and seq != self._describe_seq:
            return                              # a run the user has already moved on from
        if msg.get("ok"):
            self.window.set_checkpoint_info(msg.get("info"))
        else:
            self.window.set_checkpoint_info(
                None, error=str(msg.get("error", "체크포인트를 읽지 못했습니다.")))

    def _on_ready(self, msg: dict):
        gen = int(msg.get("gen", -1))
        if gen in self._rejected_gens:
            # It raced our stop. Its map is one we refused to draw, so it must not come up.
            self._ignore("준비 완료", gen, "그릴 수 없는 장애물이 있어 거부한 세션")
            return
        if gen != self._preparing_gen:
            # Validate before resolving anything. A ready for a generation we have moved past must
            # not clear the pending entry -- or the state -- of the one we are waiting for.
            self._ignore("준비 완료", gen, "더 새로운 세션을 기다리는 중")
            return
        self._clear_start_pending(gen)
        self._active_gen = gen
        self._preparing_gen = -1
        self._replacing = False
        facts = msg.get("facts") or {}
        #: kept verbatim so QA can read the worker's own view of the session (device, memory at
        #: the session boundary) without a second message kind
        self.last_facts = dict(facts)
        self.buffer.set_generation(gen)
        self.window.set_session_facts(facts)
        self.window.apply_state(STATE_RUNNING)
        self.window.settle_pause(False)
        self.window.viewport.reset_timing()
        self.window.spark.clear()
        self.window.status_text.setText(
            f"주행 중 — {facts.get('map', '?')} · {facts.get('total_cars', 0)}대 · "
            f"{facts.get('device', '?')}"
            + ("" if facts.get("compile") else " · CUDA 그래프 미사용"))

    def _on_ack(self, msg: dict):
        """Apply a confirmed command -- to the session it was issued for, and no other.

        Two ways an ack can be stale. It can arrive after the user has started a different session,
        in which case the generation it was sent under no longer matches. Or the worker can report
        `superseded`, meaning it deferred the command and a newer session overtook it before it
        could run. Either way the command did not happen to the session on screen, and applying it
        would move a control -- or clear a frame buffer -- that belongs to something else.
        """
        command = msg.get("command", "")
        state = msg.get("state") or {}
        seq = msg.get("seq")
        record = self._resolve(seq)
        issued_gen = record.get("gen") if record else None
        ack_gen = int(msg.get("gen", -1))

        self.window.note_ack(command)           # the button stops being pending either way
        if state.get("superseded"):
            self._ignore(f"'{command}' 확인", issued_gen if issued_gen is not None else ack_gen,
                         "worker가 더 새로운 세션으로 넘어감")
            return
        if command in ("pause", "reset", "focus", "overlay"):
            # session-scoped: only meaningful for the session that is actually on screen
            if self._active_gen < 0 or (issued_gen is not None and issued_gen != self._active_gen):
                self._ignore(f"'{command}' 확인", issued_gen, "그 세션은 이미 끝났습니다")
                return
            if ack_gen >= 0 and ack_gen != self._active_gen:
                self._ignore(f"'{command}' 확인", ack_gen, "다른 세션의 응답")
                return

        if command == "pause":
            paused = bool(state.get("paused"))
            self.window.settle_pause(paused)
            # Only now is the session paused. Frames already in flight are states the worker really
            # simulated before it stopped, so play forward to the newest of them and hold there.
            if paused:
                self.buffer.freeze()
                self.window.apply_state(STATE_PAUSED)
            else:
                self.window.apply_state(STATE_RUNNING)
        elif command == "reset":
            self.buffer.clear()
            self.window.spark.clear()
        elif command == "focus":
            self.window._pending.pop("focus", None)

    def _on_error(self, msg: dict):
        gen = int(msg.get("gen", -1))
        if gen not in (self._preparing_gen, self._active_gen):
            self._ignore("오류", gen, str(msg.get("message", "")))
            return
        replacing = self._replacing and gen == self._preparing_gen
        self._preparing_gen = -1
        self._active_gen = -1
        self._clear_start_pending(gen)
        self._replacing = False
        self._teardown_view()
        self.window.set_error(str(msg.get("where", "worker")), str(msg.get("message", "")),
                              str(msg.get("detail", "")), bool(msg.get("retryable", True)))
        if replacing:
            self.window.status_text.setText(
                self.window.status_text.text() + "  (이전 세션도 함께 종료되었습니다)")

    def _on_stopped(self, msg: dict):
        """The worker has torn a session down.

        Guarded on the generation for the same reason as `ready`: a `stopped` for the session the
        user just replaced would otherwise wipe the state of the replacement, and the replacement's
        `ready` would then be refused as belonging to a generation nobody is waiting for -- leaving
        a console stuck at IDLE with a session running behind it.
        """
        gen = int(msg.get("gen", -1))
        if self._consume_rejected("stopped", gen):
            return
        if gen >= 0 and gen not in (self._preparing_gen, self._active_gen):
            self._ignore("종료 통지", gen, "이미 다른 세션을 기다리는 중")
            return
        replacing = self._replacing and gen == self._preparing_gen
        self._clear_start_pending(gen)
        self._preparing_gen = -1
        self._active_gen = -1
        self._replacing = False
        self._teardown_view()
        self.window.apply_state(STATE_IDLE)
        # `_clear_start_pending(gen)` above already resolved the start ledger for this generation.
        # The stop ledger is this message's to clear -- and only this message's: a STOPPED for an
        # older generation must not clear a stop the user has just asked for on a newer one.
        newer_stop = any(r["command"] == P.CMD_STOP and (r.get("gen") or -1) > gen
                         for r in self._pending.values())
        if newer_stop:
            self.window.btn_stop.ack()
        else:
            self.window.note_ack("stop")
        if msg.get("cancelled"):
            self.window.status_text.setText(
                "준비를 취소했습니다." + ("  이전 세션도 함께 종료되었습니다." if replacing else ""))
        elif replacing:
            self.window.status_text.setText("새 세션이 시작되지 못했고, 이전 세션도 종료되었습니다.")

    def _teardown_view(self):
        """Put the picture back to empty. Nothing is running behind it."""
        self.buffer.clear()
        self.window.viewport.set_geometry(None)
        self.window.viewport.plan = None
        self.window.viewport.plan_pred = None
        self.window.viewport.point_colors = None
        self.window.policy_panel.clear()
        self.window.dash_panel.clear()
        self.window.activation_panel.clear()

    # ================================================================ frames
    def _drain_frames(self):
        """Take everything waiting, in order, and keep only what the buffer will hold."""
        reader = self._frame_reader
        if reader is not None:
            reader.rearm()
        n = 0
        while True:
            frame = self._frame_slot.get(timeout=0.0)
            if frame is None:
                break
            self.buffer.push(frame)
            n += 1
            if n > 64:                 # a runaway producer must not starve the event loop
                break
        self.buffer.note_receiver_drops(self._frame_slot.dropped)
        if n:
            self._apply_overlays()

    def _apply_overlays(self):
        """Hand the newest frame's overlay payload to the widgets that draw it.

        The 3D view is updated from every frame; these panels are not. See OVERLAY_HZ.
        """
        fr = self.buffer.latest
        if fr is None:
            return
        w = self.window
        vp = w.viewport
        if w.chk_plan.isChecked():
            vp.plan = fr.get("plan")
            vp.plan_pred = fr.get("plan_pred")
        else:
            vp.plan = vp.plan_pred = None
        sal = fr.get("saliency")
        # The 3D point cloud and the 2D policy panel use the same attention colours, so a bright
        # beam means the same thing in both. Brightness, not hue: hue is already the plan's speed.
        vp.point_colors = _saliency_colors(sal) if sal is not None else None

        now = time.monotonic()
        if now - self._last_overlay < 1.0 / OVERLAY_HZ:
            return                       # the 3D view above is already current; the panels can wait
        self._last_overlay = now
        w.set_policy_overlay(fr.get("scan"), sal, _plan_in_body_frame(fr),
                             w.viewport.color_v_max, fr.get("saliency_age"))
        dash = fr.get("dash")
        if dash is not None and len(dash) >= 7:
            w.set_dash(dash[0], dash[1], dash[2], dash[3], dash[4], dash[5],
                       fr.get("gg") or [], dash[6])
        if w.chk_internals.isChecked():
            w.set_activations(fr.get("hidden"), fr.get("stem"))

    # ================================================================ periodic
    def _on_tick(self):
        w = self.window
        fresh = self.buffer.freshness()
        stats = self.buffer.stats()
        # The round trip is the worker's response time, not a clock difference (see ClockLink).
        if self.buffer.clock.ready:
            stats["worker_rtt_ms"] = self.buffer.clock.rtt * 1e3
        frame = self.buffer.latest
        rate = frame.get("sim_rate") if frame else None
        w.update_telemetry(frame, fresh, stats, rate, w.viewport.rate.fps(),
                           w.viewport.rate.percentiles())
        w.tick_pending()
        self._warn_slow_acks()
        if self._proc is not None and not self._proc.is_alive() and w.state in (
                STATE_RUNNING, STATE_PAUSED, STATE_PREPARING):
            code = self._proc.exitcode
            w.set_error("worker", f"시뮬레이터 프로세스가 종료됐습니다 (exit {code}).",
                        "로그: " + os.path.join(self.runtime_dir, "worker.log"), retryable=True)
            self._active_gen = self._preparing_gen = -1

    def _warn_slow_acks(self):
        if not self._pending:
            return
        now = time.monotonic()
        seq, rec = min(self._pending.items(), key=lambda kv: kv[1]["sent"])
        dt = now - rec["sent"]
        if dt > 2.0 and self.window.state == STATE_PREPARING:
            return               # preparing is expected to be slow; the stage line already says so
        if dt > 2.0:
            self.window.status_text.setText(
                f"worker가 {dt:.1f}초째 '{rec['command']}' 명령에 응답하지 않습니다. "
                f"화면 조작은 계속 가능합니다.")

    # ================================================================ misc
    def _on_gl_failed(self, message: str):
        self.window.set_error("OpenGL", f"3D 화면을 그릴 수 없습니다: {message}",
                              "GL 3.3 core 컨텍스트가 필요합니다.", retryable=False)

    def _on_upload_timed(self, what: str, ms: float):
        if what.startswith("map:"):
            self.window.status_text.setText(
                f"맵 지오메트리 업로드 {ms:.0f} ms (worker 생성 {self._last_geometry_ms:.0f} ms, "
                f"{self._geometry_bytes / 1e6:.1f} MB)")

    def evidence(self) -> dict:
        """A snapshot of the measurements this session made, for the evidence file."""
        p = self.window.viewport.rate.percentiles()
        d = self.window.viewport.draw_percentiles()
        return {
            "instance": self.instance_id,
            "worker": {k: self._worker_hello.get(k) for k in
                       ("pid", "torch", "cuda", "device_name", "protocol")},
            "generation": self._generation,
            "frames": self.buffer.stats(),
            "frame_interval_ms": {"p50": p[0], "p95": p[1], "max": p[2]} if p else None,
            "draw_ms": {"p50": d[0], "p95": d[1], "max": d[2]} if d else None,
            "render_fps_median": self.window.viewport.rate.fps(),
            "geometry": {"bytes": self._geometry_bytes, "worker_build_ms": self._last_geometry_ms},
            "clock": {"offset_s": self.buffer.clock.offset,
                      "same_host": self.buffer.clock.same_host,
                      "worker_rtt_s": self.buffer.clock.rtt,
                      "apparent_skew_s": self.buffer.clock.skew,
                      "samples": self.buffer.clock.samples},
        }
