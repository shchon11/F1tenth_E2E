"""A stand-in simulation worker that speaks the console protocol without torch.

The real worker is owned by another session and needs a GPU, a checkpoint and half a minute to
capture CUDA graphs. None of that is what the controller's wiring is about: generations, acks,
pause semantics, error paths and shutdown are protocol behaviour, and they are worth testing
deterministically, in a second, with the failure modes reproducible on demand.

Scenario is chosen by the map name in the session config:

  ``ok:*``      normal session; frames at ``FRAME_HZ``
  ``fail:*``    raises during preparation -> MSG_ERROR(retryable=True), process stays alive
  ``slow:*``    preparation takes ``SLOW_PREPARE_S``; polls for cancel throughout
  ``hang:*``    accepts start, never becomes ready, ignores shutdown (tests the kill escalation)
  ``mute:*``    becomes ready, then stops sending frames (tests the stale-data path)
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback

import numpy as np

from f1sim.viewer.console import protocol as P
from f1sim.viewer.console.protocol import LatestSlot

#: Seconds to sleep before answering the first hello. Reproduces a real worker's `import torch`
#: startup, which is what made a midpoint clock estimate conclude the worker's clock was ahead.
SLOW_HELLO_S = float(os.environ.get("F1SIM_FAKE_WORKER_SLOW_HELLO", "0"))

FRAME_HZ = 50.0
SLOW_PREPARE_S = 3.0
N_BEAMS = 64

MAP_GROUPS = {
    "기본 평가셋": ["ok:alpha", "ok:beta", "fail:boom", "slow:prepare", "hang:forever", "mute:silent"],
    "기본 평가셋 + 장애물": ["ok:alpha+obs1"],
    "기본 학습셋": ["ok:train1", "ok:train2"],
    "전체 카탈로그": ["ok:alpha", "ok:beta", "ok:train1"],
}


def _geometry(name):
    from f1sim.viewer.gl_scene import duct_mesh_arrays, floor_mesh_arrays
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    outer = np.stack([6 + 5.0 * np.cos(th), 4.5 + 3.4 * np.sin(th)], 1)
    inner = np.stack([6 + 3.2 * np.cos(th), 4.5 + 1.7 * np.sin(th)], 1)

    def occupied(xy):
        dx, dy = xy[:, 0] - 6.0, xy[:, 1] - 4.5
        return (((dx / 5.0) ** 2 + (dy / 3.4) ** 2) > 1.0) | (((dx / 3.2) ** 2 + (dy / 1.7) ** 2) < 1.0)

    bounds = (0.2, 0.4, 11.8, 8.6)
    return {"name": name, "bounds": bounds, "duct_height": 0.2,
            "floor": floor_mesh_arrays(bounds),
            "ducts": duct_mesh_arrays([outer, inner], 0.2, occupied=occupied),
            "walls": None, "centerline": None, "raceline_xy": None, "raceline_v": None,
            "build_ms": 1.0}


class FakeWorker:
    def __init__(self, control, frames):
        self.control = control
        self.frames = frames
        self.alive = True
        self.gen = -1
        self.cancel_gen = None
        self.seq = 0
        self.paused = False
        self.running = False
        self.sim_t = 0.0
        self.focus = 0
        self.n_cars = 1
        self.scenario = "ok"
        self.overlay = {"saliency": False, "internals": False, "plan": True}
        self._slot = LatestSlot(2)
        self._sender = threading.Thread(target=self._send_loop, daemon=True)
        self._sender.start()
        self._sim = None

    # ---------------------------------------------------------------- plumbing
    def say(self, kind, **f):
        try:
            self.control.send(P.message(kind, **f))
        except (BrokenPipeError, OSError):
            self.alive = False

    def _send_loop(self):
        while True:
            item = self._slot.get(timeout=0.2)
            if item is None:
                if self._slot.closed:
                    return
                continue
            try:
                self.frames.send(item)
            except (BrokenPipeError, OSError):
                return

    # ---------------------------------------------------------------- session
    def _frame(self):
        n = self.n_cars
        ang = self.sim_t * 0.5 + np.arange(n) * 0.4
        f = {
            "gen": self.gen, "seq": self.seq, "t": self.sim_t, "n": n,
            "x": (6 + 4.1 * np.cos(ang)).astype(np.float32),
            "y": (4.5 + 2.55 * np.sin(ang)).astype(np.float32),
            "yaw": np.arctan2(2.55 * np.cos(ang), -4.1 * np.sin(ang)).astype(np.float32),
            "vx": (3.0 + 0.5 * np.sin(ang)).astype(np.float32),
            "steer": (0.1 * np.sin(ang)).astype(np.float32),
            "roll": np.zeros(n, np.float32), "pitch": np.zeros(n, np.float32),
            "lap": np.zeros(n, np.float32), "coll": np.zeros(n, np.float32),
            "s": np.zeros(n, np.float32), "wall": np.full(n, 0.4, np.float32),
            "rear": np.tile([0.10, 0.28, 0.0, 0.20], (n, 1)).astype(np.float32),
            "len": np.full(n, 0.55, np.float32),
            "focus": min(self.focus, n - 1), "focus_env": self.focus,
            "ids": np.arange(n), "control_dt": 0.025,
            "scan": np.full(N_BEAMS, 1.5, np.float32),
            "scan_type": np.ones(N_BEAMS, np.int32),
            "P": {"mount_x": 0.28, "mount_y": 0.0, "mount_z": 0.16,
                  "mount_yaw": 0.0, "mount_roll": 0.0, "mount_pitch": 0.0},
            "created_monotonic": time.monotonic(),
            "sim_rate": 1.0, "worker_dropped": self._slot.dropped,
            "mu": 1.04,
            "dash": [3.0, 3.4, 6.0, 0.1, 0.11, 8.0, 10.3],
            "gg": [(0.5, 0.3)],
            "info_line": "FAKE worker", "ckpt_age_s": 1.0,
        }
        if self.overlay.get("saliency"):
            f["saliency"] = np.linspace(0, 1, N_BEAMS).astype(np.float32)
            f["saliency_age"] = 0.1
        if self.overlay.get("internals"):
            f["hidden"] = np.zeros(256, np.float32)
            f["stem"] = np.zeros(256, np.float32)
        self.seq += 1
        return f

    def _sim_loop(self, gen):
        while self.alive and self.running and self.gen == gen:
            if not self.paused:
                self.sim_t += 0.025
                self._slot.put(self._frame())
            time.sleep(1.0 / FRAME_HZ)

    def start(self, msg):
        gen = int(msg["gen"])
        cfg = P.SessionConfig.from_dict(msg["config"])
        # A replacement stops whatever was running, and a cancelled or failed replacement does NOT
        # bring it back -- matching the backend contract. Anything else leaves a simulation running
        # behind an idle window.
        self.stop_session()
        self.gen = gen
        self.cancel_gen = None
        self.seq = 0
        self.sim_t = 0.0
        self.paused = False
        self.focus = 0
        self.n_cars = max(1, cfg.races * cfg.cars_per_race)
        self.scenario = (cfg.map_name.split(":", 1)[0] if ":" in cfg.map_name else "ok")
        self.overlay.update({"saliency": cfg.saliency, "internals": cfg.internals})
        try:
            for stage, _text in P.STAGES[:-1]:
                self.say(P.MSG_STAGE, gen=gen, stage=stage)
                if self.scenario == "slow":
                    deadline = time.monotonic() + SLOW_PREPARE_S / len(P.STAGES)
                    while time.monotonic() < deadline:
                        self._poll_during_prepare()
                        if self.cancel_gen == gen:
                            raise _Cancelled()
                        time.sleep(0.02)
                else:
                    self._poll_during_prepare()
                    if self.cancel_gen == gen:
                        raise _Cancelled()
                if self.scenario == "fail" and stage == "map":
                    raise ValueError(f"{cfg.map_name}: 이 맵을 열 수 없습니다 (테스트용 실패).")
            if self.scenario == "hang":
                return                       # accepted, never ready
        except _Cancelled:
            self.say(P.MSG_STOPPED, gen=gen, cancelled=True)
            self.gen = -1
            return
        except Exception as exc:
            self.say(P.MSG_ERROR, gen=gen, where="맵", message=str(exc),
                     detail=traceback.format_exc(), retryable=True)
            self.gen = -1
            return
        self.say(P.MSG_GEOMETRY, gen=gen, geometry=_geometry(cfg.map_name))
        self.say(P.MSG_READY, gen=gen, facts={
            "gen": gen, "run": cfg.run, "checkpoint": "fake.pt", "map": cfg.map_name,
            "races": cfg.races, "cars_per_race": cfg.cars_per_race,
            "total_cars": self.n_cars, "max_render_cars": cfg.max_render_cars,
            "car_ids": list(range(self.n_cars)), "device": "cpu", "compile": False,
            "speed_cap": 6.0, "v_max_policy": 8.0, "action_mode": "direct",
            "info_line": "FAKE worker", "randomize": cfg.randomize,
            "lidar": {"fov": 4.712, "range_max": 10.0, "n_beams": N_BEAMS},
            "vehicle": {"lr": 0.15, "cog_z": 0.06, "wheel_r": 0.056},
        })
        if self.scenario == "mute":
            self.running = False             # ready, but nothing will ever arrive
            return
        self.running = True
        self._sim = threading.Thread(target=self._sim_loop, args=(gen,), daemon=True)
        self._sim.start()

    def stop_session(self):
        self.running = False
        t, self._sim = self._sim, None
        if t is not None and t.is_alive():
            t.join(timeout=2.0)

    def _poll_during_prepare(self):
        while self.control.poll():
            self.handle(self.control.recv(), during_prepare=True)

    # ---------------------------------------------------------------- commands
    def handle(self, msg, during_prepare=False):
        kind = msg.get("kind")
        seq = msg.get("seq")
        if kind == P.CMD_HELLO:
            if SLOW_HELLO_S:
                time.sleep(SLOW_HELLO_S)        # as if we were still importing torch
            self.say(P.MSG_HELLO, seq=seq, pid=os.getpid(), monotonic=time.monotonic(),
                     torch="fake", cuda=False, device_name="cpu",
                     runs_dir=os.path.expanduser("~/f1sim_runs"), protocol=P.PROTOCOL_VERSION)
        elif kind == P.CMD_LIST_MAPS:
            self.say(P.MSG_MAPS, seq=seq, groups=MAP_GROUPS)
        elif kind == P.CMD_DESCRIBE:
            run = msg.get("run", "")
            if run.startswith("missing"):
                self.say(P.MSG_DESCRIBED, seq=seq, ok=False, run=run,
                         error=f"{run}: 체크포인트가 없습니다.")
            else:
                self.say(P.MSG_DESCRIBED, seq=seq, ok=True, info={
                    "run": run, "file": "fake.pt", "age_s": 42.0,
                    "progress": f"FAKE {run}", "metrics": "저장 시점: 충돌률 0.05",
                    "speed_cap": 6.0})
        elif kind == P.CMD_CANCEL:
            self.cancel_gen = int(msg.get("gen", -1))
            self.say(P.MSG_ACK, seq=seq, command="cancel", gen=self.cancel_gen)
        elif kind == P.CMD_PAUSE:
            self.paused = bool(msg.get("paused"))
            self.say(P.MSG_ACK, seq=seq, command="pause", gen=self.gen,
                     state={"paused": self.paused, "last_seq": self.seq - 1})
        elif kind == P.CMD_RESET:
            self.sim_t = 0.0
            self.say(P.MSG_ACK, seq=seq, command="reset", gen=self.gen)
        elif kind == P.CMD_FOCUS:
            self.focus = max(0, min(self.n_cars - 1, int(msg.get("env", 0))))
            self.say(P.MSG_ACK, seq=seq, command="focus", gen=self.gen, state={"env": self.focus})
        elif kind == P.CMD_OVERLAY:
            self.overlay.update({k: bool(v) for k, v in msg.get("overlay", {}).items()})
            self.say(P.MSG_ACK, seq=seq, command="overlay", gen=self.gen, state=dict(self.overlay))
        elif kind == P.CMD_STOP:
            self.stop_session()
            self.say(P.MSG_ACK, seq=seq, command="stop", gen=self.gen)
            self.say(P.MSG_STOPPED, gen=self.gen)
            self.gen = -1
        elif kind == P.CMD_SHUTDOWN:
            if self.scenario == "hang":
                return                        # deliberately unresponsive: the console must escalate
            self.stop_session()
            self.alive = False
        elif kind == P.CMD_START:
            if during_prepare:
                self._queued = msg
            else:
                self.start(msg)

    def serve(self):
        self._queued = None
        while self.alive:
            if self._queued is not None:
                msg, self._queued = self._queued, None
                self.start(msg)
                continue
            if not self.control.poll(0.05):
                continue
            try:
                msg = self.control.recv()
            except (EOFError, OSError):
                break
            self.handle(msg)
        self.stop_session()
        self._slot.close()
        self.say(P.MSG_BYE)


class _Cancelled(Exception):
    pass


def main(control, frames, log_path=None):
    if log_path:
        try:
            sys.stderr = open(log_path, "a", buffering=1)
        except OSError:
            pass
    try:
        FakeWorker(control, frames).serve()
    finally:
        try:
            control.close()
            frames.close()
        except Exception:
            pass
