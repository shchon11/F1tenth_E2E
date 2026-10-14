"""Guards and failure semantics of the viewer's CUDA-graph fast path.

Most of this is CPU-only: a *failed* capture poisons the process, so the fatal-path tests inject
`CaptureFailed` rather than provoking one -- what the worker does when capture fails is the point,
not that it can fail. Those checks must keep passing on a machine with no GPU.

The rest are the equivalence checks, which do capture, and they are skipped without CUDA. They are
expensive -- each builds whole environments and holds a graph's private memory pool -- so they tear
down explicitly, and `_free_cuda_between_tests` sweeps after each. That fixture also restores the
current CUDA device, which is what actually made this file's results depend on the order it was run
in; the note on it has the measurement.

Run them on a named card (`CUDA_VISIBLE_DEVICES`), not on whichever one happens to be there: one of
these tests asks for the *last* device, and on a laptop with a second GPU that is the card the
user's console is drawing on.
"""
from __future__ import annotations

import gc
import sys
import types

import pytest
import torch

from f1sim.mpc import PlanSpec, PlanTracker, solve
from f1sim.viewer import sim_worker
from f1sim.viewer.graph_fastpath import (CaptureFailed, GuardViolation, NotCapturable,
                                         SimGraphFastPath, _dict_sig, _freeze, mpc_eligible,
                                         roll_eligible)


@pytest.fixture(autouse=True)
def _free_cuda_between_tests():
    """Put back the device this test was given, and the memory it used.

    **The device is the one that mattered.** `test_a_captured_soft_roll_matches_eager` picks
    `cuda:(device_count() - 1)` and calls `torch.cuda.set_device`, which is process-wide and
    outlives the test. Every later test that says `device="cuda"` with no index -- the actor
    capture, the recorder, the training runtime -- then ran on whichever card that one chose. On
    this two-GPU machine that is the small one, which is also the card the user's console sits on,
    so the captures near the end of the file died for want of room while passing on their own.

    Measured 2026-09-23, with only the big card visible (`CUDA_VISIBLE_DEVICES` naming it): the
    whole file passes, 34 of 34 -- and it passes with the pre-fix version of this file too, because
    with one card there is nothing to leak the device to. So the failures were never this file
    exhausting one GPU; they were this file quietly moving to the other one.

    The memory sweep stays, because it is true and cheap: the CUDA caching allocator does not
    return a freed block to the driver, a captured graph owns a private pool that only goes back
    when the last reference to it is dropped, and an env reaches its own tensors through reference
    cycles, so refcounts alone do not free it. It runs before as well as after, because a test that
    fails leaves its objects alive in the traceback pytest keeps. None of that was the trigger, and
    the console's own teardown (`SimWorker._release_session`) already does both.
    """
    def sweep():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    dev0 = torch.cuda.current_device() if torch.cuda.is_available() else None
    sweep()
    try:
        yield
    finally:
        if dev0 is not None and torch.cuda.current_device() != dev0:
            torch.cuda.set_device(dev0)
        sweep()


# --------------------------------------------------------------------- snapshots
def test_freeze_sees_a_mutated_planspec():
    """The reason `_other` stores snapshots and not the object.

    Keeping the caller's `PlanSpec` and comparing it against itself is unconditionally equal: a
    changed `N` mutates the very object the check reads.
    """
    spec = PlanSpec()
    before = _freeze(spec)
    same_object = spec
    spec.N = spec.N + 1
    assert same_object == spec, "the reference compare that a snapshot has to replace"
    assert _freeze(spec) != before


def test_freeze_sees_a_changed_tuple_field():
    spec = PlanSpec()
    before = _freeze(spec)
    spec.q = (2.0, 6.0, 1.0, 0.4)
    assert _freeze(spec) != before


def test_freeze_rejects_a_value_it_cannot_snapshot():
    with pytest.raises(NotCapturable):
        _freeze(object())


def test_dict_sig_sees_a_rebound_key():
    P = {"a": torch.zeros(3)}
    before = _dict_sig(P)
    P["a"] = torch.zeros(3)                 # same values, new storage: the graph reads the address
    assert _dict_sig(P) != before


def test_dict_sig_sees_an_added_scalar():
    """A non-tensor entry is frozen too, not skipped: it would be baked in by value."""
    P = {"a": torch.zeros(3)}
    before = _dict_sig(P)
    P["susp_wn"] = 12.0
    assert _dict_sig(P) != before


def test_dict_sig_sees_a_changed_scalar():
    P = {"a": torch.zeros(3), "susp_wn": 12.0}
    before = _dict_sig(P)
    P["susp_wn"] = 13.0
    assert _dict_sig(P) != before


def test_dict_sig_rejects_an_unknown_value():
    with pytest.raises(NotCapturable):
        _dict_sig({"a": torch.zeros(3), "weird": object()})


# ------------------------------------------------------------------ eligibility
class _FakeSim:
    def __init__(self, **kw):
        self.device = torch.device("cpu")
        self.cfg = types.SimpleNamespace(
            sim=types.SimpleNamespace(compile=False, terminate_on_collision=True),
            imu=types.SimpleNamespace(enabled=True, imu_rate=50.0))
        self.imu_period = 4
        self.imu_schedule = [[19], [14], [9], [4, 24]]
        self.state = torch.zeros(1, 8)
        self._imu_phase = 0
        self._roll = self._roll_physics
        self.__dict__.update(kw)

    def _roll_physics(self, *a):
        return a


def test_the_wheel_model_switch_is_guarded():
    """`vehicle.wheel_model` is read at Python level inside `_roll_physics` -- it picks the
    longitudinal model, the VESC's feedback speed and whether the IMU shock term runs -- so it is
    baked into the graph at capture and a later change would be replayed away silently."""
    from f1sim.viewer.graph_fastpath import SimGraphFastPath
    sim = _FakeSim(wheel_model=True, tid=torch.zeros(1, dtype=torch.long), B=1,
                   control_dt=0.025, dt=0.001, hist_len=3, substeps=25, imu_ts=0.02,
                   track=types.SimpleNamespace())
    g = SimGraphFastPath(sim)._guards_for(sim)
    assert "sim.wheel_model" in g and g["sim.wheel_model"]() is True
    sim.wheel_model = False
    assert g["sim.wheel_model"]() is False


def test_cpu_session_is_not_eligible_and_says_so():
    ok, why = roll_eligible(_FakeSim())
    assert ok is False and why


def test_soft_wall_is_no_longer_refused():
    """Soft walls used to be refused because `_resolve_wall_contact` rebound `prop_touched` inside
    the substep loop, which a graph replay drops. It is an in-place `logical_or_` now; whether the
    captured roll really keeps those contacts is `test_a_captured_soft_roll_matches_eager`, on a GPU.
    What this pins is only that the refusal is gone -- the day soft became a default it silently
    cost the viewer and training their graphs (0.1x realtime; 185 steps/s against 320)."""
    sim = _FakeSim()
    sim.device = torch.device("cuda")
    sim.cfg.sim.terminate_on_collision = False
    _ok, why = roll_eligible(sim)
    assert "soft wall" not in why


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to capture a graph")
def test_a_captured_soft_roll_matches_eager():
    """The thing the old refusal protected: substep prop contacts under soft walls, replayed from a
    graph, have to be the contacts eager mode sees -- and the car has to end up in the same place.

    Two envs, same seed, same actions, obstacles on the racing line so contacts actually happen;
    one steps eager and one through `prepare_graph_runtime`. Randomisation, road tilt and the IMU
    are off, so the only difference between the two is the capture.
    """
    import os
    from f1sim import Config, maps
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    from f1sim.learn.graph_runtime import prepare_graph_runtime
    from f1sim.raceline import Raceline
    dev = "cuda:%d" % (torch.cuda.device_count() - 1)
    torch.cuda.set_device(torch.device(dev))
    tr = maps.load("gen:competition:0")
    rl = Raceline.build_cached(tr)

    def build():
        cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"
        cfg.lidar.n_beams = 36; cfg.rand.enabled = False; cfg.vehicle.road_tilt = 0.0
        cfg.imu.enabled = False
        ecfg = EnvConfig(race_size=2, opponent="slots", max_steps=4000, hist_len=0,
                         action_mode="direct", collision_mode="soft",
                         procedural_obstacles=1.0, procedural_density=3.0,
                         procedural_raceline_corridor="off", spawn_runway=3.0,
                         opponent_slots=[{"kind": "forzaeth"}])
        env = common.make_env([tr], 16, dev, ecfg, cfg=cfg, seed=11, rls=[rl])
        env.reset(seed=11)
        return env

    eager, graphed = build(), build()
    fp = None
    try:
        fp = prepare_graph_runtime(graphed, log=lambda _t: None)
        assert fp is not None, "the soft-wall roll was not captured"
        # prepare_graph_runtime steps the env to record arguments; bring the eager twin level with it
        act = torch.zeros(eager.B, eager.act_dim, device=dev)
        while int(eager.ep_step.max()) < int(graphed.ep_step.max()):
            eager.step(act)
        g = torch.Generator(device="cpu").manual_seed(3)
        touched_e = touched_g = 0
        for _ in range(80):
            a = (torch.rand(eager.B, eager.act_dim, generator=g) * 2 - 1).to(dev)
            _o, _r, _t, _tr, ie = eager.step(a)
            _o, _r, _t, _tr, ig = graphed.step(a)
            touched_e += int((ie["wall_dist"] <= 0).sum()); touched_g += int((ig["wall_dist"] <= 0).sum())
        assert touched_e > 0, "nothing was ever touched, so the contact path was not exercised"
        assert touched_g == touched_e, f"contacts differ: eager {touched_e}, graph {touched_g}"
        diff = float((eager.sim.state[:, :4] - graphed.sim.state[:, :4]).abs().max())
        assert diff < 1e-3, f"the graphed roll drifted from eager by {diff}"
    finally:
        # Two 16-env environments and a graph pool, on the device the later captures need. The
        # `release()` alone is not enough: it restores the hooks, and it is dropping the last
        # reference that hands the pool back (`_free_cuda_between_tests` does the collecting).
        if fp is not None:
            fp.release()
        del fp, eager, graphed


def test_schedule_disagreeing_with_period_is_not_eligible():
    sim = _FakeSim()
    sim.imu_period = 4
    sim.imu_schedule = [[19], [14]]
    ok, why = roll_eligible(sim)
    assert ok is False


def test_mpc_eligibility_refuses_a_compiled_tracker():
    sim = _FakeSim()
    trk = PlanTracker(1, "cpu", 0.32, 0.4, 9.0, compile_solver=True)
    ok, why = mpc_eligible(sim, trk)
    assert ok is False and why


def test_mpc_eligibility_refuses_a_non_float32_tracker():
    """`build_ilqr_consts` matches the inline branch only at torch's default dtype."""
    sim = _FakeSim()
    sim.device = torch.device("cuda")
    sim.cfg.sim.compile = False
    trk = PlanTracker(1, "cpu", 0.32, 0.4, 9.0, compile_solver=False)
    trk.u_prev = trk.u_prev.double()
    ok, why = mpc_eligible(sim, trk)
    assert ok is False
    assert "float32" in why


# ------------------------------------------------------------------- dispatch
def test_roll_refuses_an_uncaptured_phase_instead_of_wrapping():
    """A modulo over "how many graphs exist" would map phase 3 onto graph 1, whose IMU sample lands
    at a different substep. Wrong data, no error. A direct index raises instead."""
    sim = _FakeSim()
    fp = SimGraphFastPath(sim)
    fp._by_phase = {0: lambda *a: "phase0", 1: lambda *a: "phase1"}
    sim._imu_phase = 0
    assert fp.roll() == "phase0"
    sim._imu_phase = 3
    with pytest.raises(KeyError):
        fp.roll()


# --------------------------------------------------------------------- hooks
class _FakeTracker:
    def __init__(self):
        self._solver = None


def test_install_then_release_restores_both_hooks():
    sim = _FakeSim()
    original_roll = sim._roll
    trk = _FakeTracker()
    fp = SimGraphFastPath(sim)
    fp.mpc = lambda *a: None                 # stands in for a captured solver
    fp.install(trk)
    assert sim._roll == fp.roll
    assert trk._solver is fp.mpc
    fp.release()
    assert sim._roll is original_roll
    assert trk._solver is None


def test_release_drops_the_simulator_reference():
    """`_release_session` holds this object in a local while it runs `gc.collect()`; keeping `sim`
    would keep the whole env reachable through it for that pass."""
    sim = _FakeSim()
    fp = SimGraphFastPath(sim)
    fp.install(None)
    fp.release()
    assert fp.sim is None


def test_install_twice_is_refused():
    fp = SimGraphFastPath(_FakeSim())
    fp.install(None)
    with pytest.raises(GuardViolation):
        fp.install(None)


def test_release_is_idempotent():
    fp = SimGraphFastPath(_FakeSim())
    fp.install(None)
    fp.release()
    fp.release()                             # must not raise on the second teardown path


# ------------------------------------------------------- fatal capture failure
def _bare_worker():
    """A `SimWorker` without its constructor: these tests exercise two methods, and the real
    constructor opens pipes and starts a sender thread."""
    w = sim_worker.SimWorker.__new__(sim_worker.SimWorker)
    w.session = None
    w.gen = 1
    w.prepare_gen = None
    w.cancel_gen = None
    w.alive = True
    w.running = True
    w.said = []
    w.say = lambda kind, **kw: w.said.append((kind, kw))
    w.teardown = lambda: None
    return w


def test_capture_failure_is_not_retried_even_when_it_says_out_of_memory():
    """The OOM retry exists for a build that lost a race with the previous session's memory. A
    capture that ran out of memory still poisoned the context, so retrying on it is worse than
    failing -- and the message contains the words the retry looks for."""
    w = _bare_worker()
    w.session = {"env": object()}            # a resident previous session: the retry's precondition
    calls = []

    def build(cfg, gen):
        calls.append(gen)
        raise CaptureFailed("mpc.solve: OutOfMemoryError: CUDA out of memory")

    w.build_session = build
    with pytest.raises(CaptureFailed):
        w._build_with_oom_retry(object(), 2)
    assert calls == [2], "capture failure must not be retried on the same context"


def test_a_plain_oom_is_still_retried_once():
    """The pre-existing behaviour the test above must not have broken."""
    w = _bare_worker()
    w.session = {"env": object()}
    calls = []

    def build(cfg, gen):
        calls.append(gen)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return {"ok": True}

    w.build_session = build
    assert w._build_with_oom_retry(object(), 2) == {"ok": True}
    assert calls == [2, 2]


def test_a_capture_failure_stops_the_worker_before_it_lifts_the_build_hold():
    """Reporting the error and staying alive would hand the next retry a poisoned context, and the
    console retries on whatever worker is up. Order matters too: the hold must not lift while the
    loop is still alive, or the outgoing session takes one more step and draws sensor noise from
    the generator the capture just poisoned.
    """
    w = _bare_worker()
    seen = {}

    def abandon():
        seen["alive_when_abandoned"] = w.alive
        seen["running_when_abandoned"] = w.running

    w._abandon_build = abandon
    w._build_hold = lambda on: True
    w._swap_in = lambda *a: None
    w._build_with_oom_retry = lambda cfg, gen: (_ for _ in ()).throw(
        CaptureFailed("_roll_physics[phase 0]: RuntimeError: CUDA out of memory"))
    # A config that passes `validate_start_config`, so the build is actually reached: an empty one
    # fails validation and returns from the "설정" branch without ever getting near a capture.
    w.start({"gen": 2, "config": {"map_name": "real:korea_2026_competition"}})

    assert w.alive is False, "the process must stop; the console spawns a fresh worker"
    assert seen["alive_when_abandoned"] is False
    assert seen["running_when_abandoned"] is False
    errors = [kw for kind, kw in w.said if kind.endswith("error")]
    assert errors, "the console still has to be told why"
    assert "CUDA 그래프" in errors[-1]["where"]
    assert errors[-1]["retryable"] is True


def test_plan_tracker_solver_hook_defaults_to_the_module_level_choice():
    """Nothing global is replaced: a tracker with no hook picks exactly what it picked before."""
    trk = PlanTracker(1, "cpu", 0.32, 0.4, 9.0, compile_solver=False)
    assert trk._solver is None
    seen = []
    trk._solver = lambda *a: seen.append(a) or solve(*a)
    assert trk._solver is not None


def test_mpc_solve_still_defaults_to_building_its_constants_inline():
    """Training and every existing caller must be untouched by the opt-in `consts` argument.

    Checked by NAME rather than against the whole `__defaults__` tuple. The tuple version asserted
    `== (None,)` and so failed the moment `v_limit`, `bounds` and `a_walk` were added -- all three
    of which default to `None` and change nothing for an existing caller. It was reporting the
    arrival of new optional arguments, not the property the docstring describes.

    The property is: every optional argument of `solve` defaults to `None`, so a caller passing only
    the required ones gets the original behaviour -- constants built inline, no precomputed table,
    no extra limits. A non-`None` default is what would silently change training, and that is what
    this still catches.
    """
    import inspect

    from f1sim import mpc
    optional = {n: p.default for n, p in inspect.signature(mpc.solve).parameters.items()
                if p.default is not inspect.Parameter.empty}
    assert "consts" in optional, "the opt-in `consts` argument is gone; this test is out of date"
    assert all(v is None for v in optional.values()), optional
    assert mpc.ilqr.__defaults__[-1] is None


# --------------------------------------------------------------------- the actor's own step
def _memory_actor(device="cpu", beams=64, batch=1):
    """A small recurrent actor plus one real example of each of its three arguments."""
    from f1sim.learn.memory import memory_spec
    from f1sim.learn.model import ActorCritic
    with torch.random.fork_rng():
        torch.manual_seed(77)
        m = ActorCritic(n_stack=2, n_beams=beams, proprio_dim=8, priv_dim=4, act_dim=2,
                        scan_stem="plain", memory=memory_spec(hidden_size=16)).to(device).eval()
    scan = torch.rand(batch, 2, beams, device=device)
    pro = torch.rand(batch, 8, device=device)
    h = m.actor.initial_hidden(batch, device=scan.device, dtype=scan.dtype)
    return m.actor, (scan, pro, h)


def test_actor_eligibility_is_decided_before_capture():
    """A CPU session, a train-mode actor or a non-float32 argument select eager and say so, rather
    than reaching a capture that would be fatal."""
    from f1sim.viewer.graph_fastpath import actor_eligible, graph_actor_step
    actor, args = _memory_actor()
    ok, why = actor_eligible(actor, args)
    assert not ok and "CPU" in why
    with pytest.raises(NotCapturable):
        graph_actor_step(actor, args)


def test_prepare_actor_graph_runs_eager_rather_than_capturing_on_the_cpu():
    """`learn.graph_runtime.prepare_actor_graph` is the entry point a session build calls; on an
    ineligible configuration it returns None and logs why, so the caller simply runs eager."""
    from f1sim.learn.graph_runtime import prepare_actor_graph
    actor, (scan, pro, h) = _memory_actor()
    said = []
    assert prepare_actor_graph(actor, scan, pro, h, log=said.append) is None
    assert any("not eligible" in t for t in said), said
    # A feedforward actor has nothing to carry and is not captured here either.
    said.clear()
    from f1sim.learn.model import ActorCritic
    ff = ActorCritic(n_stack=2, n_beams=64, proprio_dim=8, priv_dim=4, act_dim=2,
                     scan_stem="plain").eval()
    assert prepare_actor_graph(ff.actor, scan, pro, None, log=said.append) is None
    assert any("feedforward" in t for t in said), said


@pytest.mark.skipif(not torch.cuda.is_available(), reason="capturing a graph needs CUDA")
def test_captured_actor_step_matches_eager_and_keeps_the_hidden_state_static():
    """The replayed graph agrees with eager, and the hidden state lives in a static buffer: the
    caller passes its one tensor in and the graph copies it there, so nothing a previous replay
    produced is still being read."""
    from f1sim.learn.graph_runtime import prepare_actor_graph
    actor, (scan, pro, h) = _memory_actor(device="cuda", batch=2)
    g = prepare_actor_graph(actor, scan, pro, h, log=lambda _t: None)
    assert g is not None
    with torch.no_grad():
        want_mu, want_h = actor.step(scan, pro, None, h)
        got_mu, got_h = g(scan, pro, h)
    torch.testing.assert_close(got_mu, want_mu, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got_h, want_h, atol=1e-5, rtol=1e-5)
    # Two steps of carrying: the graph is replayed against whatever the caller hands it.
    with torch.no_grad():
        mu2, h2 = g(scan, pro, got_h)
        want2, wh2 = actor.step(scan, pro, None, want_h)
    torch.testing.assert_close(mu2, want2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(h2, wh2, atol=1e-5, rtol=1e-5)
    # And the guard on the actor's own weights fires when they are replaced under it.
    with torch.no_grad():
        actor.mu.weight = torch.nn.Parameter(actor.mu.weight.clone())
    with pytest.raises(GuardViolation):
        g(scan, pro, got_h)
    # `GraphedCallable` has no release: the pool comes back when the last reference goes.
    del g, actor, scan, pro, h, got_h, want_h, got_mu, want_mu, mu2, h2, want2, wh2


# ------------------------------------------------------------------ the opponent teacher
def _fake_race_env(solver, **tracker_hooks):
    tracker = types.SimpleNamespace(_solver=solver, _input_hook=None, _plan_hook=None,
                                    _command_hook=None)
    tracker.__dict__.update(tracker_hooks)
    return types.SimpleNamespace(sim=_FakeSim(device=torch.device("cuda")), M=3, teacher=object(),
                                 teacher_any=True, pool=None, alt_teachers=[], tracker=tracker)


def test_a_graphed_plan_solver_no_longer_keeps_the_teacher_eager():
    """This used to be refused: "with the legacy controller the interactive teacher previews the
    installed MPC, which is the solver's own graph". The preview calls `mpc.solve` itself, at its
    own K x B batch, with constants cached on the tracker (`PlanTracker.ilqr_consts`) so the capture
    makes no host copy -- nothing replays the solver graph from inside a teacher call, and the
    refusal kept every legacy-controller race's opponents eager (61 % of s911's step). The two
    parity tests below are what vouch for it now; they failed at "should capture its teacher"
    while the refusal stood."""
    from f1sim.viewer.graph_fastpath import GraphedCallable, teacher_eligible
    graphed = GraphedCallable.__new__(GraphedCallable)
    ok, why = teacher_eligible(_fake_race_env(graphed))
    assert ok, why
    ok, _ = teacher_eligible(_fake_race_env(graphed, _plan_hook=lambda *a: None))
    assert ok
    ok, why = teacher_eligible(types.SimpleNamespace(sim=_FakeSim(), M=3, teacher=object()))
    assert not ok and why == "CPU 세션"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the recorder follows CUDA tensors")
def test_input_recorder_keeps_views_it_made_out_of_the_inputs():
    """`P["lf"][:, None, None]` shares the parameter buffer's storage but is the call's own view:
    it must not be reported as an input nobody holds (which would refuse every capture), while the
    entry it came from must be."""
    from f1sim.viewer.graph_fastpath import _InputRecorder, _tensor_paths, _view_key
    buf = torch.arange(6.0, device="cuda")
    root = types.SimpleNamespace(P={"lf": buf[0:3], "lr": buf[3:6]})
    with _InputRecorder() as rec:
        x = root.P["lf"][:, None, None] * 2.0
        x + 1.0
    paths = _tensor_paths(_Root(root.P))
    assert set(rec.inputs) == {_view_key(root.P["lf"])}
    assert all(k in paths for k in rec.inputs)
    assert not rec.written


class _Root:
    """`_tensor_paths` walks f1sim objects only; this stands in for the environment."""
    __module__ = "f1sim.tests"

    def __init__(self, P):
        self.P = P


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the recorder follows CUDA tensors")
def test_input_recorder_notices_a_write_into_an_input():
    from f1sim.viewer.graph_fastpath import _InputRecorder
    held = torch.zeros(4, device="cuda")
    with _InputRecorder() as rec:
        held[1:3].add_(1.0)                # through a view: still the input's memory
    assert rec.written


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="capturing a graph needs CUDA")
@pytest.mark.parametrize("slots", [None, [{"kind": "interactive"}, {"kind": "interactive"}]],
                         ids=["raceline-teacher", "interactive-slots"])
def test_teacher_graph_matches_eager_bit_for_bit_and_restores_the_env(slots):
    """Three cars on a procedural map: every replayed teacher call equals the eager call on the
    same state, over steps that include a race reset, and releasing the graph puts the env's own
    method back. The slot table is how the console builds any race with opponents, and naming the
    interactive kind makes the env call a second teacher object every step."""
    from f1sim.viewer.console import protocol as P

    class _Null:
        def send(self, *a):
            pass

        def poll(self, *a):
            return False

    w = sim_worker.SimWorker(_Null(), _Null())
    w.say = lambda *a, **k: None
    w.stage = lambda *a, **k: None
    w._hold_parked_ok = True
    extra = {"opponent": "slots", "opponent_slots": slots} if slots else {}
    s = w.build_session(P.SessionConfig(map_name="gen:competition:2", cars_per_race=3,
                                        stochastic=True, **extra), 0)
    try:
        tg = s.get("teacher_graph")
        assert tg is not None, "the auto-controller race should capture its teacher"
        for gc_ in (s["fastpath"], s.get("controller"), tg):
            if gc_ is not None:
                gc_.adopt()
        env = s["env"]
        # Only the teachers the env actually calls: an all-interactive table never asks the
        # raceline teacher, whose rows would all be overwritten.
        teachers = ([env.teacher] if env._raceline_teacher_needed() else []) + list(env.alt_teachers)
        assert teachers and len(tg._captures) == len(teachers)
        # The arguments the env passes: a teacher that plans a subset is asked for its rows.
        rows_of = {id(t): (env._alt_rows[k],) for k, t in zip(env.alt_teacher_kinds, env.alt_teachers)
                   if hasattr(t, "plan_rows")}
        for _ in range(120):
            follow, v_cap = env.follow_cap(env.sim.state)
            for t in teachers:
                extra = rows_of.get(id(t), ())
                with torch.no_grad():
                    want = tg._eager(t, None, None, follow, v_cap, *extra)
                    got = tg(t, None, None, follow, v_cap, *extra)
                assert torch.equal(want, got)
            w._step_once(s)
        assert not tg.fell_back and tg.replays > 100 * len(teachers)
    finally:
        w._release_session(s)
    assert "_teacher_normalized" not in env.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to capture a graph")
def test_the_console_session_captures_soft_walls_with_props():
    """The console's own session path, on the user's own scenario, because that is the only path the
    user drives through and it is not the training path.

    `SimWorker._prepare_fastpath` records the arguments of `sim._prop_contact` through a spy and
    then captures that call as a leaf graph -- both positional-only, one arity. The day soft walls
    became capturable, `_resolve_wall_contact` called the same attribute from inside the roll with
    `fn=` and `want_slot=`, and session start died with "contact_spy() got an unexpected keyword
    argument 'fn'". The fix was verified through `learn.graph_runtime` and never through this, which
    is how it shipped.

    `real/iccas25#line:*!assets=mixed:1` is exactly what the console hands the worker for
    "iccas25, line obstacles, random seed" (`tracks.asset_scenario`); a bare `#line:44` rasterises
    its obstacles into the grid, has no props, and never reaches the crashing call.
    """
    from multiprocessing import Pipe
    from f1sim.viewer import sim_worker
    from f1sim.viewer.console import protocol as P
    from f1sim.learn.watch import latest_run
    run = latest_run()
    if run is None:
        pytest.skip("no training run on this machine to load a policy from")
    dev = "cuda:%d" % (torch.cuda.device_count() - 1)
    ctl_w, ctl_c = Pipe(duplex=True)
    fr_r, fr_w = Pipe(duplex=False)
    w = sim_worker.SimWorker(ctl_w, fr_w)
    w._hold_parked_ok = True                     # what the hold sets when nothing is stepping
    cfg = P.SessionConfig(run=str(run), map_name="real/iccas25#line:*!assets=mixed:1",
                          races=1, cars_per_race=1, device=dev, compile=False,
                          collision_mode="soft")
    session = w.build_session(cfg, gen=1)
    try:
        env = session["env"]
        assert env.sim.track.has_props, "the scenario did not produce props, so nothing was tested"
        assert not env.sim.cfg.sim.terminate_on_collision
        assert session.get("fastpath") is not None, "the soft-wall session fell back to eager"
        del env
    finally:
        # The worker's own teardown, not just the fast path's: a built session also holds the
        # policy, the controller and -- on a scenario with opponents -- a teacher graph, and this
        # used to release one of the four. What was left standing is what a later capture in this
        # file could not find room beside.
        w._release_session(session)
        ctl_c.close(); fr_r.close()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="capturing a graph needs CUDA")
def test_training_captures_the_opponent_planners_and_they_carry_their_state():
    """The training runtime graphs the opponents too, and a planner with memory keeps it.

    s911's recipe draws each race's opponent from four planners; eager, they were 61 % of an env
    step at 256 envs. Two of them remember things between steps -- the spliner which side it
    committed to, the lane planner its lane and its offset -- so every step each planner is asked
    twice from the same state: eagerly, and through its graph after the state is put back. The
    command and the state it leaves behind must match to the bit, over steps that include races
    resetting with obstacles standing on the racing line."""
    from f1sim import Config
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common, graph_runtime
    tracks, rls = common.load_tracks(["gen:competition:2"], racelines=True)
    ecfg = EnvConfig(race_size=2, opponent="slots", max_steps=90,
                     opponent_slots=[{"kind_mix": ["forzaeth", "forzaeth_pred", "lane_switch", "interactive"],
                                      "speed_scale": [0.7, 1.0]}],
                     action_mode="plan", procedural_obstacles=1.0, procedural_density=2.0,
                     procedural_max_props=8, procedural_raceline_corridor="off", spawn_runway=3.0,
                     collision_mode="soft", movable_obstacles=True, stagger_first_episode=True,
                     compile_tracker=False)
    cfg = Config(); cfg.sim.compile = False; cfg.lidar.n_beams = 181
    env = common.make_env(tracks, 16, "cuda", ecfg, cfg=cfg, seed=3, rls=rls)
    env.reset(seed=3)
    logs = []
    rt = graph_runtime.prepare_graph_runtime(env, log=logs.append)
    try:
        tg = rt.teacher_graph
        assert tg is not None, f"the planners were not captured: {logs}"
        teachers = list(env.alt_teachers)
        assert len(tg._captures) == len(teachers) == 4
        stateful = [t for t in teachers if getattr(t, "GRAPH_STATE", ())]
        assert len(stateful) == 3, [type(t).__name__ for t in stateful]
        a = torch.zeros(env.B, env.act_dim, device=env.device)
        rows_of = {id(t): (env._alt_rows[k],) for k, t in zip(env.alt_teacher_kinds, env.alt_teachers)
                   if hasattr(t, "plan_rows")}
        assert rows_of, "the interactive opponent should be planned for its own rows"
        for _ in range(120):
            follow, v_cap = env.follow_cap(env.sim.state)
            for t in teachers:
                names = getattr(t, "GRAPH_STATE", ())
                extra = rows_of.get(id(t), ())
                snap = [getattr(t, n).clone() for n in names]
                with torch.no_grad():
                    want = tg._eager(t, None, None, follow, v_cap, *extra)
                    want_state = [getattr(t, n).clone() for n in names]
                    for n, v in zip(names, snap):
                        getattr(t, n).copy_(v)
                    got = tg(t, None, None, follow, v_cap, *extra)
                assert torch.equal(want, got), type(t).__name__
                for n, v in zip(names, want_state):
                    assert torch.equal(getattr(t, n), v), f"{type(t).__name__}.{n} was not carried"
            env.step(a)
        assert not tg.fell_back, tg.fell_back
        assert tg.replays > 100 * len(teachers)
    finally:
        graph_runtime.release_graph_runtime(rt)
    assert "_teacher_normalized" not in env.__dict__
