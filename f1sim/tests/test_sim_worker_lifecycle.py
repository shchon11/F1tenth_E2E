"""Lifecycle and generation semantics of the console's simulation worker.

These run the *real* `SimWorker` -- its control loop, its simulation thread, its bounded outbound
slot, its acks -- over real `multiprocessing.Connection` pipes, with only the three torch-facing
methods replaced by a stub environment. That split is deliberate: the parts most likely to be wrong
(who applies a command, when an ack is true, which generation a frame belongs to, what happens when
the reader disappears) have nothing to do with torch, and testing them against a real checkpoint
would make them slow, GPU-dependent and too coarse to pin a race.

`test_sim_worker_smoke.py` covers the other half: a real checkpoint, a real environment, real
frames.

The stub advances a clock at 40 Hz per step exactly as the simulator does, so "sim time did not
move while paused" means the same thing here as it does there.
"""
from __future__ import annotations

import math
import multiprocessing
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.viewer import sim_worker as SW
from f1sim.viewer.console import protocol as P

CONTROL_DT = 0.025
DEADLINE = 10.0


# ==================================================================== stub environment
def _fake_env(cars: int = 2):
    """The attribute surface `session_facts` and the stub stepper read, and nothing else."""
    # `sim.cfg` is read by `session_facts` for the physics-compile facts
    # (`sim_worker.py:982-983`: `env.sim.cfg.sim.compile` and `.compile_mode`). The stub carried
    # `env.cfg` but not `env.sim.cfg`, so once those facts were added every start in this file died
    # with `AttributeError: 'types.SimpleNamespace' object has no attribute 'cfg'` -- a drifted
    # double, not a worker fault. `compile=False` is what this stub represents: it never compiles.
    sim = types.SimpleNamespace(
        B=cars, t=0.0, control_dt=CONTROL_DT, device="cpu", resets=0,
        cfg=types.SimpleNamespace(
            sim=types.SimpleNamespace(compile=False, compile_mode="none"),
            imu=types.SimpleNamespace(enabled=False, imu_rate=0.0)))

    def reset():
        sim.resets += 1
        # The real `F1VecEnv.reset` steps once and never rewinds `sim.t`; the console's playback
        # clock relies on that, so the stub must not rewind it either.
        sim.t += CONTROL_DT
        return {}, {}

    return types.SimpleNamespace(
        sim=sim, reset=reset, act_dim=2, n_beams=32, range_max=10.0,
        ecfg=types.SimpleNamespace(v_max_policy=8.0),
        cfg=types.SimpleNamespace(lidar=types.SimpleNamespace(fov=4.712),
                                  vehicle=types.SimpleNamespace(lr=0.17)),
    )


class StubWorker(SW.SimWorker):
    """Real worker, stub simulator. `build_delay` makes a start slow enough to cancel."""

    build_delay = 0.0
    step_delay = 0.0
    #: Floats of ballast per frame. A stub frame is a few hundred bytes, which the OS pipe buffer
    #: swallows whole, so a test that wants to see the *slot* drop has to make frames big enough to
    #: block the sender the way a real one does when the console stalls.
    payload_floats = 0

    def _heartbeat(self):
        pass                      # never touch the global viewer flag from a test

    def _heartbeat_clear(self):
        pass

    def build_session(self, cfg, gen):
        env = _fake_env(cfg.total_cars)
        for name in ("checkpoint", "map", "env", "warmup", "geometry"):
            self.stage(gen, name)
            if self.build_delay:
                time.sleep(self.build_delay)
        if cfg.map_name == "boom":
            raise SW.StartConfigError("맵 'boom' 을 읽지 못했습니다: 테스트용 실패")
        return {
            "env": env, "device": "cpu", "mode": "direct", "compile": False,
            "ckpt_path": f"/tmp/{cfg.run}/ppo_latest.pt", "mtime": time.time(),
            "speed_cap": 6.0, "info_line": f"stub {cfg.run}", "cfg": cfg,
            "track": types.SimpleNamespace(name=cfg.map_name), "focus": 0, "k": 0,
            "gg": SW.deque(maxlen=90), "last_reload": time.time(), "obs": {},
            "geometry": {"name": cfg.map_name, "bounds": (0.0, 0.0, 1.0, 1.0),
                         "duct_height": 0.16, "floor": None, "ducts": None, "walls": None,
                         "centerline": None, "raceline_xy": None, "raceline_v": None,
                         "build_ms": 0.0},
        }

    def _step_once(self, session):
        session["env"].sim.t += CONTROL_DT
        session["k"] += 1
        if self.step_delay:
            time.sleep(self.step_delay)

    def _reload_if_changed(self, session):
        pass

    def _snapshot(self, session):
        env = session["env"]
        n = min(env.sim.B, int(session["cfg"].max_render_cars))
        z = np.zeros(n, np.float32)
        fr = {"gen": self.gen, "seq": self.seq, "t": float(env.sim.t), "n": n,
              "x": z.copy(), "y": z.copy(), "yaw": z.copy(), "steps": session["k"],
              "focus": int(session["focus"]), "sim_rate": float(self._sim_rate),
              "control_dt": CONTROL_DT, "gg": list(session["gg"])}
        if self.payload_floats:
            fr["ballast"] = np.zeros(self.payload_floats, np.float32)
        self.seq += 1
        return fr


# ==================================================================== harness
class Console:
    """The console side of both pipes, plus the message bookkeeping a test needs."""

    def __init__(self, worker_cls=StubWorker, **attrs):
        self.ctl, ctl_theirs = multiprocessing.Pipe(duplex=True)
        frames_ours, frames_theirs = multiprocessing.Pipe(duplex=False)
        assert frames_theirs.writable and frames_ours.readable
        self.frames = frames_ours
        self.worker = worker_cls(ctl_theirs, frames_theirs)
        for k, v in attrs.items():
            setattr(self.worker, k, v)
        self.seq = 0
        self.messages = []
        self._thread = threading.Thread(target=self.worker.serve, name="worker-serve", daemon=True)
        self._thread.start()

    # ---------------- sending
    def send(self, kind, **fields):
        self.seq += 1
        self.ctl.send(P.message(kind, seq=self.seq, **fields))
        return self.seq

    def start(self, gen, **cfg):
        cfg.setdefault("run", "run_a")
        cfg.setdefault("map_name", "map_a")
        return self.send(P.CMD_START, gen=gen, config=P.SessionConfig(**cfg).to_dict())

    # ---------------- receiving
    def pump(self, timeout=0.2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.ctl.poll(max(0.0, deadline - time.monotonic())):
                break
            self.messages.append(self.ctl.recv())
        return self.messages

    def wait_for(self, kind, timeout=DEADLINE, **match):
        """The first message of `kind` matching `match`, waiting for it to arrive."""
        deadline = time.monotonic() + timeout
        idx = 0
        while True:
            while idx < len(self.messages):
                m = self.messages[idx]
                idx += 1
                if m.get("kind") == kind and all(m.get(k) == v for k, v in match.items()):
                    return m
            left = deadline - time.monotonic()
            if left <= 0:
                raise AssertionError(
                    f"{kind} {match} 가 {timeout}초 안에 오지 않았습니다. 받은 것: "
                    + ", ".join(f"{m.get('kind')}({m.get('gen')})" for m in self.messages))
            if self.ctl.poll(min(0.05, left)):
                self.messages.append(self.ctl.recv())

    def drain_frames(self, timeout=0.0):
        out = []
        deadline = time.monotonic() + timeout
        while True:
            if not self.frames.poll(max(0.0, deadline - time.monotonic())):
                return out
            out.append(self.frames.recv())

    def next_frames(self, count, timeout=DEADLINE):
        out = []
        deadline = time.monotonic() + timeout
        while len(out) < count:
            left = deadline - time.monotonic()
            if left <= 0 or not self.frames.poll(left):
                raise AssertionError(f"프레임 {count}개 중 {len(out)}개만 도착했습니다.")
            out.append(self.frames.recv())
        return out

    def close(self):
        try:
            self.send(P.CMD_SHUTDOWN)
            self._thread.join(timeout=5.0)
        except (BrokenPipeError, OSError):
            pass
        self.worker.alive = False
        self.worker._shutdown_requested = True
        self._thread.join(timeout=5.0)
        for c in (self.ctl, self.frames):
            try:
                c.close()
            except OSError:
                pass


@pytest.fixture
def console():
    c = Console()
    yield c
    c.close()


def _running_session(c, gen=1, **cfg):
    c.start(gen, **cfg)
    c.wait_for(P.MSG_READY, gen=gen)
    return c.next_frames(1)[0]


# ==================================================================== pure helpers
def test_validate_start_config_rejects_what_cannot_be_built():
    ok = P.SessionConfig(map_name="m")
    SW.validate_start_config(ok)                       # does not raise
    for bad, needle in (
        (P.SessionConfig(map_name="m", races=0), "레이스 수"),
        (P.SessionConfig(map_name="m", cars_per_race=0), "레이스당"),
        (P.SessionConfig(map_name=" "), "맵을 선택"),
        (P.SessionConfig(map_name="m", speed_cap=-1.0), "속도 상한"),
        (P.SessionConfig(map_name="m", max_render_cars=0), "화면 표시"),
        (P.SessionConfig(map_name="m", device="tpu"), "장치 이름"),
    ):
        with pytest.raises(SW.StartConfigError) as e:
            SW.validate_start_config(bad)
        assert needle in str(e.value)


def test_resolve_checkpoint_paths_and_messages(tmp_path):
    runs = tmp_path / "runs"
    (runs / "r1").mkdir(parents=True)
    ckpt = runs / "r1" / "ppo_latest.pt"
    ckpt.write_bytes(b"x")
    assert SW.resolve_checkpoint("r1", str(runs)) == str(ckpt)
    assert SW.resolve_checkpoint(str(ckpt), str(runs)) == str(ckpt)
    assert SW.resolve_checkpoint("latest", str(runs), latest=str(runs / "r1")) == str(ckpt)

    with pytest.raises(FileNotFoundError, match="체크포인트가 있는 런이 없습니다"):
        SW.resolve_checkpoint("latest", str(runs), latest="")
    with pytest.raises(FileNotFoundError, match="런 디렉터리가 없습니다"):
        SW.resolve_checkpoint("nope", str(runs))
    (runs / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="ppo_latest.pt / student_latest.pt"):
        SW.resolve_checkpoint("empty", str(runs))
    with pytest.raises(FileNotFoundError, match="체크포인트 파일이 없습니다"):
        SW.resolve_checkpoint(str(runs / "r1" / "gone.pt"), str(runs))


def test_obs_spec_problems_catches_a_silently_reinitialised_actor():
    spec = types.SimpleNamespace(n_beams=1080, scan_stack=3, proprio_dim=29)
    meta = {"n_beams": 1080, "n_stack": 3, "proprio_dim": 29, "act_dim": 2}
    assert SW.obs_spec_problems(meta, spec, [], 2) == []
    assert SW.obs_spec_problems(meta, spec, ["critic.mlp.0.weight"], 2) == []   # critic is not driven

    problems = SW.obs_spec_problems(meta, spec, ["actor.stem.fc.weight"], 2)
    assert len(problems) == 1 and "초기값" in problems[0]

    problems = SW.obs_spec_problems({**meta, "proprio_dim": 17}, spec, [], 2)
    assert any("proprio" in p and "17" in p and "29" in p for p in problems)

    problems = SW.obs_spec_problems({**meta, "act_dim": 6}, spec, [], 2)
    assert any("행동 차원" in p for p in problems)


def test_normalizer_problems_catches_a_same_shape_checkpoint():
    """The mismatch a dimension check cannot see.

    `v_max`, `range_max` and the IMU scales change what every observation *number means* without
    moving a single tensor shape, so `obs_spec_problems` passes a checkpoint trained at `v_max` 8
    into a `v_max` 10 environment. Every speed the policy reads, and every speed it commands, is
    then off by 25% while nothing looks wrong.
    """
    env_spec = types.SimpleNamespace(v_max=10.0, range_max=10.0, gyro_scale=5.0,
                                     accel_scale=10.0, att_scale=0.35,
                                     n_beams=1080, scan_stack=3, proprio_dim=29)
    saved = {"v_max": 10.0, "range_max": 10.0, "gyro_scale": 5.0, "accel_scale": 10.0,
             "att_scale": 0.35}
    assert SW.normalizer_problems(saved, env_spec) == []
    assert SW.normalizer_problems({}, env_spec) == []            # nothing saved: nothing to check

    # shapes identical, meaning different -- the dimension check sees nothing wrong
    meta = {"n_beams": 1080, "n_stack": 3, "proprio_dim": 29, "act_dim": 2}
    assert SW.obs_spec_problems(meta, env_spec, [], 2) == []
    problems = SW.normalizer_problems({**saved, "v_max": 8.0}, env_spec)
    assert len(problems) == 1 and "v_max" in problems[0] and "8" in problems[0] and "10" in problems[0]

    for key, bad in (("range_max", 20.0), ("gyro_scale", 2.0), ("accel_scale", 20.0),
                     ("att_scale", 1.0)):
        assert len(SW.normalizer_problems({**saved, key: bad}, env_spec)) == 1, key
    assert len(SW.normalizer_problems({**saved, "v_max": 8.0, "range_max": 30.0}, env_spec)) == 2


def test_where_of_maps_exceptions_to_the_console_vocabulary():
    assert SW.where_of(SW.StartConfigError("x")) == "설정"
    assert SW.where_of(FileNotFoundError("x")) == "체크포인트"
    assert SW.where_of(RuntimeError("CUDA error: out of memory")) == "GPU"
    assert SW.where_of(KeyError("x")) == "세션 준비"


def test_copy_breaks_the_alias_a_cpu_tensor_hands_out():
    """`t.float().cpu().numpy()` is a view when both calls are no-ops, which is exactly the CPU
    case. A frame built from one is rewritten by the next step before it is ever sent."""
    torch = pytest.importorskip("torch")
    t = torch.zeros(4)
    aliased = t.float().cpu().numpy()
    copied = SW._copy(t.float().cpu().numpy())
    t[0] = 7.0
    assert aliased[0] == 7.0, "the alias hazard this guards against no longer exists"
    assert copied[0] == 0.0


# ==================================================================== start / stages / frames
def test_start_reports_stages_then_geometry_then_ready_then_frames(console):
    console.start(1)
    ready = console.wait_for(P.MSG_READY, gen=1)
    stages = [m["stage"] for m in console.messages if m.get("kind") == P.MSG_STAGE]
    assert stages == ["checkpoint", "map", "env", "warmup", "geometry"]
    geom = console.wait_for(P.MSG_GEOMETRY, gen=1)
    assert geom["geometry"]["name"] == "map_a"
    assert console.messages.index(geom) < console.messages.index(ready), "geometry must precede ready"

    facts = ready["facts"]
    assert facts["gen"] == 1 and facts["map"] == "map_a" and facts["total_cars"] == 1
    assert facts["device"] == "cpu" and facts["compile"] is False
    assert facts["lidar"]["n_beams"] == 32 and facts["vehicle"]["wheel_r"] == SW.WHEEL_R

    frames = console.next_frames(3)
    assert all(f["gen"] == 1 for f in frames)
    assert [f["seq"] for f in frames] == [0, 1, 2]
    assert all(k in frames[0] for k in ("t", "n", "seq", "x", "y", "yaw",
                                        "created_monotonic", "worker_dropped"))
    assert frames[-1]["t"] > frames[0]["t"]


def test_facts_report_what_was_built_not_what_was_asked(console):
    """`max_render_cars` above the car count is reported as the number actually drawn, and the
    focus-car list matches it -- the console builds its combo box straight from this."""
    console.start(1, races=2, cars_per_race=2, max_render_cars=64)
    facts = console.wait_for(P.MSG_READY, gen=1)["facts"]
    assert facts["total_cars"] == 4
    assert facts["max_render_cars"] == 4 and facts["car_ids"] == [0, 1, 2, 3]


# ==================================================================== pause / resume / reset
def test_pause_ack_is_true_when_it_arrives(console):
    _running_session(console)
    seq = console.send(P.CMD_PAUSE, paused=True)
    ack = console.wait_for(P.MSG_ACK, seq=seq)
    assert ack["command"] == "pause" and ack["state"]["paused"] is True
    last_seq, t_at_ack = ack["state"]["last_seq"], ack["state"]["t"]

    time.sleep(0.35)                     # 14 control steps' worth of wall time
    after = console.drain_frames(timeout=0.2)
    # Frames produced before the ack may still be in flight; that is what `last_seq` is for. What
    # must not exist is a frame produced *after* the pause took effect.
    assert all(f["seq"] <= last_seq for f in after), \
        f"paused worker produced new frames: {[f['seq'] for f in after]} > {last_seq}"
    assert all(f["t"] <= t_at_ack + 1e-9 for f in after)
    assert console.worker.session["env"].sim.t == pytest.approx(t_at_ack)


def test_resume_advances_time_again(console):
    _running_session(console)
    console.wait_for(P.MSG_ACK, seq=console.send(P.CMD_PAUSE, paused=True))
    console.drain_frames(timeout=0.2)
    t_paused = console.worker.session["env"].sim.t

    ack = console.wait_for(P.MSG_ACK, seq=console.send(P.CMD_PAUSE, paused=False))
    assert ack["state"]["paused"] is False
    frames = console.next_frames(2)
    assert frames[-1]["t"] > t_paused
    assert frames[-1]["seq"] > frames[0]["seq"]


def test_reset_is_applied_by_the_simulation_thread_and_does_not_rewind_time(console):
    """A reset applied from the control thread would be a data race against `env.step`. The ack
    comes from the simulation thread, so by the time it lands the episode really has restarted."""
    first = _running_session(console)
    console.next_frames(2)
    env = console.worker.session["env"]
    assert env.sim.resets == 0

    ack = console.wait_for(P.MSG_ACK, seq=console.send(P.CMD_RESET))
    assert ack["command"] == "reset" and env.sim.resets == 1
    assert ack["state"]["t"] >= first["t"], "sim time must not rewind across a reset"
    after = console.next_frames(2)
    assert after[-1]["seq"] > ack["state"]["last_seq"]
    assert after[-1]["t"] >= ack["state"]["t"]


def test_focus_is_clamped_to_the_cars_actually_drawn(console):
    _running_session(console, races=2, cars_per_race=1, max_render_cars=1)
    ack = console.wait_for(P.MSG_ACK, seq=console.send(P.CMD_FOCUS, env=9))
    assert ack["state"]["env"] == 0, "focus must stay inside the set of cars the console draws"
    assert console.next_frames(1)[0]["focus"] == 0


def test_live_commands_are_acked_with_nothing_running(console):
    """The console draws these controls as pending until an ack lands, so an ack that never comes
    is a button stuck at "적용 중" forever."""
    for kind, fields in ((P.CMD_PAUSE, {"paused": True}), (P.CMD_RESET, {}),
                         (P.CMD_FOCUS, {"env": 2}),
                         (P.CMD_OVERLAY, {"overlay": {"saliency": True}})):
        ack = console.wait_for(P.MSG_ACK, seq=console.send(kind, **fields))
        assert ack["state"]["session"] is False
    assert console.worker._overlay["saliency"] is True    # remembered for the next session


def test_overlay_ack_reports_the_whole_state(console):
    _running_session(console)
    ack = console.wait_for(P.MSG_ACK, seq=console.send(P.CMD_OVERLAY,
                                                       overlay={"saliency": True, "plan": False}))
    assert ack["state"] == {"saliency": True, "internals": False, "plan": False}


# ==================================================================== generations
def test_a_new_generation_ends_the_old_one(console):
    """The cut is on *production*, not arrival: frames the sender already wrote into the pipe
    cannot be unsent, which is exactly why every frame carries `gen`."""
    _running_session(console, gen=1, map_name="map_a")
    console.next_frames(2)
    console.start(2, map_name="map_b")
    ready = console.wait_for(P.MSG_READY, gen=2)
    assert ready["facts"]["map"] == "map_b"

    later = console.drain_frames(timeout=0.4)
    assert later, "the new generation should be producing frames"
    stale = [f for f in later if f["gen"] != 2]
    assert all(f["created_monotonic"] <= ready["monotonic"] for f in stale), \
        "generation 1 was still being simulated after ready(gen=2)"
    assert [f["gen"] for f in later] == sorted(f["gen"] for f in later), "generations interleaved"
    new = [f for f in later if f["gen"] == 2]
    assert new and new[0]["seq"] == 0, "sequence restarts with the generation"


def test_a_generation_that_goes_backwards_is_refused(console):
    _running_session(console, gen=5)
    console.start(5)
    err = console.wait_for(P.MSG_ERROR, gen=5)
    assert "되돌아갔습니다" in err["message"] and err["retryable"] is False
    assert console.worker.gen == 5 and console.worker.session is not None


def test_start_during_prepare_supersedes_the_one_in_flight():
    c = Console(build_delay=0.12)
    try:
        c.start(1, map_name="map_a")
        c.wait_for(P.MSG_STAGE, gen=1, stage="map")
        c.start(2, map_name="map_b")
        assert c.wait_for(P.MSG_STOPPED, gen=1)["cancelled"] is True
        assert c.wait_for(P.MSG_READY, gen=2)["facts"]["map"] == "map_b"
        assert all(f["gen"] == 2 for f in c.next_frames(2))
    finally:
        c.close()


def test_cancel_during_prepare_stops_everything_rather_than_resuming_invisibly():
    """The old session is kept *loaded* during a build so a cancel is cheap, but it is not resumed.

    The console adopts the new generation the moment it sends `start` and goes IDLE on `stopped`, so
    a resumed old session would be stepping on the GPU with nothing on screen representing it and
    no control able to stop it -- the window would have to be told to stop a session it believes is
    not running. Rolling the console back to the old view is the other coherent answer and is not
    what this version does, so IDLE has to be true.
    """
    c = Console(build_delay=0.12)
    try:
        c.start(1, map_name="map_a")
        c.wait_for(P.MSG_READY, gen=1)
        c.next_frames(1)

        c.start(2, map_name="map_b")
        c.wait_for(P.MSG_STAGE, gen=2, stage="map")
        assert c.worker.session is not None, "the old session should be held, not torn down early"
        c.send(P.CMD_CANCEL, gen=2)
        assert c.wait_for(P.MSG_STOPPED, gen=2)["cancelled"] is True

        assert c.worker.session is None, "a cancelled replacement left a session running invisibly"
        assert c.worker.gen == -1 and c.worker._sim_thread is None
        c.drain_frames(timeout=0.2)
        assert c.drain_frames(timeout=0.3) == [], "frames are still arriving after cancel"
    finally:
        c.close()


def test_the_outgoing_session_is_held_still_while_the_next_one_is_built():
    """Not an optimisation: `torch.compile(mode="reduce-overhead")` captures CUDA graphs on the
    control thread, and a simulation thread drawing from the default CUDA generator during that
    capture raises `Offset increment outside graph capture encountered unexpectedly` -- which is
    what `lidar.scan`'s range noise does on every single step. Reproduced on an RTX 4060 Ti with
    torch 2.10 before this hold existed; see implementation-report.md.

    The session is held, not stopped: its environment stays loaded so a cancel is instant.
    """
    c = Console(build_delay=0.25)
    try:
        c.start(1, map_name="map_a")
        c.wait_for(P.MSG_READY, gen=1)
        c.next_frames(2)

        c.start(2, map_name="map_b")
        c.wait_for(P.MSG_STAGE, gen=2, stage="map")
        assert c.worker._hold is True
        steps_a = c.worker.session["k"]
        t_a = c.worker.session["env"].sim.t
        time.sleep(0.2)
        assert c.worker.session["k"] == steps_a, "the outgoing session kept stepping during a build"
        assert c.worker.session["env"].sim.t == t_a

        c.wait_for(P.MSG_READY, gen=2)
        assert c.worker._hold is False
        assert all(f["gen"] == 2 for f in c.next_frames(2))
    finally:
        c.close()


def test_pause_still_answers_while_a_build_holds_the_simulation():
    """Pause touches no tensor, so it is the one command applied during a hold -- otherwise the
    button sits at "적용 중" for the whole of a twenty-second CUDA graph capture."""
    c = Console(build_delay=0.3)
    try:
        c.start(1, map_name="map_a")
        c.wait_for(P.MSG_READY, gen=1)
        c.start(2, map_name="map_b")
        c.wait_for(P.MSG_STAGE, gen=2, stage="map")
        ack = c.wait_for(P.MSG_ACK, seq=c.send(P.CMD_PAUSE, paused=True), timeout=3.0)
        assert ack["state"]["paused"] is True
    finally:
        c.close()


def test_commands_left_over_from_a_replaced_session_are_still_acked():
    """An ack that never comes is a control stuck at "적용 중" for the rest of the session."""
    c = Console(build_delay=0.25)
    try:
        c.start(1, map_name="map_a")
        c.wait_for(P.MSG_READY, gen=1)
        c.start(2, map_name="map_b")
        c.wait_for(P.MSG_STAGE, gen=2, stage="map")
        seq = c.send(P.CMD_RESET)                 # deferred: it touches the session
        ack = c.wait_for(P.MSG_ACK, seq=seq, timeout=5.0)
        assert ack["state"].get("superseded") is True
        assert c.wait_for(P.MSG_READY, gen=2)["facts"]["gen"] == 2
    finally:
        c.close()


# ==================================================================== failures
def test_an_invalid_start_is_an_error_and_the_worker_survives(console):
    console.start(1, races=0)
    err = console.wait_for(P.MSG_ERROR, gen=1)
    assert err["where"] == "설정" and err["retryable"] is True and "레이스 수" in err["message"]

    console.start(2)                              # the worker is still usable
    assert console.wait_for(P.MSG_READY, gen=2)["facts"]["gen"] == 2


def test_a_failed_build_stops_the_previous_session_too(console):
    """Same reasoning as the cancel case: the console goes FAILED, so nothing may still be running."""
    _running_session(console, gen=1, map_name="map_a")
    console.start(2, map_name="boom")
    err = console.wait_for(P.MSG_ERROR, gen=2)
    assert "boom" in err["message"] and err["retryable"] is True and err["detail"]
    assert console.worker.session is None, "a failed replacement left the old session running"
    assert console.worker.gen == -1
    console.drain_frames(timeout=0.2)
    assert console.drain_frames(timeout=0.3) == []

    console.start(3, map_name="map_a")           # and a retry still works
    assert console.wait_for(P.MSG_READY, gen=3)["facts"]["gen"] == 3


def test_a_simulation_crash_is_reported_and_does_not_kill_the_worker(console):
    _running_session(console)
    console.next_frames(1)

    def boom(session):
        raise RuntimeError("스텝 실패(테스트)")

    console.worker._step_once = boom
    err = console.wait_for(P.MSG_ERROR, gen=1)
    assert err["where"] == "시뮬레이션" and err["retryable"] is True
    assert console.wait_for(P.MSG_STATE, gen=1)["state"] == P.STATE_FAILED

    console.worker._step_once = StubWorker._step_once.__get__(console.worker)
    console.start(2)                              # still able to start another session
    assert console.wait_for(P.MSG_READY, gen=2)["facts"]["gen"] == 2


def test_stop_frees_the_session_and_the_worker_stays_up(console):
    _running_session(console)
    seq = console.send(P.CMD_STOP)
    assert console.wait_for(P.MSG_ACK, seq=seq)["command"] == "stop"
    console.wait_for(P.MSG_STOPPED)
    assert console.worker.session is None and console.worker.gen == -1
    assert console.drain_frames(timeout=0.25) == [] or True   # frames stop; timing-tolerant
    time.sleep(0.1)
    assert console.drain_frames(timeout=0.15) == []

    console.start(2)
    assert console.wait_for(P.MSG_READY, gen=2)["facts"]["gen"] == 2


def test_shutdown_says_bye_and_returns():
    c = Console()
    c.start(1)
    c.wait_for(P.MSG_READY, gen=1)
    c.send(P.CMD_SHUTDOWN)
    c.wait_for(P.MSG_BYE)
    c._thread.join(timeout=5.0)
    assert not c._thread.is_alive()
    assert c.worker.session is None
    c.close()


def test_a_closed_control_pipe_ends_the_worker_cleanly():
    """The console dying is not an error condition; the worker just tidies up and goes."""
    c = Console()
    c.start(1)
    c.wait_for(P.MSG_READY, gen=1)
    c.ctl.close()
    c._thread.join(timeout=5.0)
    assert not c._thread.is_alive()
    assert c.worker.session is None
    c.frames.close()


# ==================================================================== the reader
def test_the_simulation_outruns_a_console_that_never_reads():
    """The whole point of the bounded slot: a stalled reader must cost frames, never steps.

    The frames carry ballast so the pipe buffer fills and the sender thread genuinely blocks, which
    is the condition a real console with a busy GL thread puts the worker in.
    """
    c = Console(payload_floats=200_000)               # ~800 kB a frame, pipe buffer is ~64 kB
    try:
        _running_session(c)
        time.sleep(0.4)                               # never touch c.frames
        steps_a = c.worker.session["k"]
        time.sleep(0.4)
        steps_b = c.worker.session["k"]
        assert steps_b > steps_a, "the simulation stopped when the console stopped reading"
        assert c.worker._slot.dropped > 0, "a blocked sender should have cost dropped frames"
        assert len(c.worker._slot) <= SW.OUTBOUND_CAPACITY

        # The first frame out is the one the sender was already blocked on: its count was stamped
        # before any of these drops happened. The ones behind it carry the running total.
        counts = [f["worker_dropped"] for f in c.next_frames(3)]
        assert max(counts) > 0, f"the drop count must reach the console, got {counts}"
        assert counts == sorted(counts), f"the reported count must not go backwards: {counts}"
    finally:
        c.close()


def test_the_simulation_survives_the_reader_disappearing(console):
    """A crashed renderer must not take the simulation with it -- the console can be restarted and
    reattached, and a wedged worker holding a GPU is worse than a missing window."""
    _running_session(console)
    console.next_frames(1)
    console.frames.close()
    time.sleep(0.3)
    steps_a = console.worker.session["k"]
    t_a = console.worker.session["env"].sim.t
    time.sleep(0.3)
    assert console.worker.session["k"] > steps_a
    assert console.worker.session["env"].sim.t > t_a
    # and control still works
    ack = console.wait_for(P.MSG_ACK, seq=console.send(P.CMD_PAUSE, paused=True))
    assert ack["state"]["paused"] is True


def test_frames_are_capped_below_the_send_ceiling(console):
    """`MAX_FRAME_HZ` is checked before the snapshot is built, so the cost of a frame nobody asked
    for is not paid at all."""
    console.worker._min_send_dt = 1.0 / 20.0
    _running_session(console)
    t0 = time.monotonic()
    frames = console.next_frames(6)
    elapsed = time.monotonic() - t0
    assert elapsed >= 5 / 20.0 * 0.8, f"6 frames in {elapsed:.3f}s is faster than the 20 Hz cap"
    assert frames[-1]["steps"] > frames[-1]["seq"], "throttled steps should outnumber sent frames"


def test_the_outbound_slot_drops_oldest_and_counts():
    slot = SW.LatestSlot(2)
    assert slot.put("a") and slot.put("b")
    assert slot.put("c") is False and slot.dropped == 1
    assert [slot.get(0.0), slot.get(0.0)] == ["b", "c"]
    assert slot.get(0.0) is None


def test_main_refuses_the_wrong_end_of_the_frame_pipe():
    """`Pipe(duplex=False)` returns `(readable, writable)`. Handing the worker the first of the
    pair is an easy wiring mistake, and it would otherwise look like a healthy session with no
    picture."""
    ctl_ours, ctl_theirs = multiprocessing.Pipe(duplex=True)
    read_end, write_end = multiprocessing.Pipe(duplex=False)
    SW.main(ctl_theirs, read_end)                     # returns instead of running
    msgs = []
    while ctl_ours.poll(0.5):
        msgs.append(ctl_ours.recv())
    kinds = [m["kind"] for m in msgs]
    assert P.MSG_ERROR in kinds and P.MSG_BYE in kinds
    err = next(m for m in msgs if m["kind"] == P.MSG_ERROR)
    assert "읽기 쪽" in err["message"] and err["retryable"] is False
    for c in (ctl_ours, read_end, write_end):
        try:
            c.close()
        except OSError:
            pass
