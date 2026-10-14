"""Per-row plan formats: one tracker, each car's plan read in its own format (output-format design 6,
migration step 6).

A wp4 learner races scripted cars and league checkpoints that keep the linear plan they were built
and trained in. The tracker builds both references for every row, each row takes its own at the
reference stage (`mpc.solve`'s `row_format`), and the iLQR runs once. What has to hold, and is
checked here: every row of a mixed solve is the single-format solve of that row, bit for bit --
the solve, the tracker's state across steps, an env's first command per car -- a tracker without
codes is the one it always was, the opponents' side of an env plans, scales, certifies and resets
in the opponents' format, a league checkpoint is held to the format its rows are read in, and the
runtime arms (which decode linear) refuse a mixed tracker. The closed-loop consequences are
docs/research/wp4-label-gates-2026-10-14/ (RESULTS 7).
"""
from __future__ import annotations

import types

import pytest
import torch

from f1sim import Config, mpc
from f1sim.gym_env import EnvConfig
from f1sim.learn import common

LIN, WP4 = mpc.PlanSpec(), mpc.PlanSpec(speed_mode="wp4")
WB, SMAX, VMAX = 0.33, 0.42, 10.0
TRACK = "gen:competition:0"


def _inputs(B: int, seed: int, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    a = r(B, 8) * 2.4 - 1.2                        # past the box too: the decoders clamp
    v = r(B) * 10 - 0.5                            # standing, reversing a little, at pace
    return [t.to(device) for t in (a, v, r(B) * 9 + 0.5, torch.randn(B, generator=g), r(B) * 0.08,
                                   torch.randn(B, 2, generator=g) * 0.3, torch.randn(B, 12, 2, generator=g) * 0.2)]


@pytest.mark.parametrize("kappa_mode", ["absolute", "feasible"])
@pytest.mark.parametrize("pfp", [False, True])
@pytest.mark.parametrize("own", ["wp4", "linear"])
def test_each_row_of_a_mixed_solve_is_its_own_formats_solve_bit_for_bit(kappa_mode, pfp, own):
    """u, z and the reference of every row, against the same batch solved in one format."""
    from dataclasses import replace
    base = mpc.PlanSpec(speed_mode=own, kappa_mode=kappa_mode, profile_from_prediction=pfp)
    B = 257
    args = _inputs(B, 1)
    codes = (torch.rand(B, generator=torch.Generator().manual_seed(2)) < 0.5).to(torch.uint8)
    mixed = mpc.solve(*args, base, WB, SMAX, VMAX, row_format=codes)
    alone = {c: mpc.solve(*args, replace(base, speed_mode=m), WB, SMAX, VMAX) for c, m in enumerate(mpc.ROW_FORMATS)}
    for c in (0, 1):
        rows = codes == c
        assert rows.any()
        for got, want in zip(mixed, alone[c]):
            assert torch.equal(got[rows], want[rows]), (mpc.ROW_FORMATS[c], kappa_mode, pfp, own)
    # every row in one format is exactly that format's tracker
    for c, m in enumerate(mpc.ROW_FORMATS):
        same = mpc.solve(*args, base, WB, SMAX, VMAX, row_format=torch.full((B,), c, dtype=torch.uint8))
        assert all(torch.equal(x, y) for x, y in zip(same, alone[c]))


def test_a_tracker_with_codes_keeps_every_row_on_its_own_formats_trajectory():
    """Five steps through `PlanTracker`, warm starts carried: the command, the warm start, the last
    applied control, the reference and the prediction of every row are its own format's tracker's."""
    B = 64
    trk = {"linear": mpc.PlanTracker(B, "cpu", WB, SMAX, VMAX, spec=LIN, compile_solver=False),
           "wp4": mpc.PlanTracker(B, "cpu", WB, SMAX, VMAX, spec=WP4, compile_solver=False)}
    mixed = mpc.PlanTracker(B, "cpu", WB, SMAX, VMAX, spec=WP4, compile_solver=False)
    codes = (torch.arange(B) % 3 == 0).to(torch.uint8)                 # a third of the rows wp4
    mixed.set_row_formats(codes)
    for step in range(5):
        a, v, cap, yaw, delay, _, _ = _inputs(B, 10 + step)
        out = {m: t(a, v, cap, yaw, delay=delay) for m, t in trk.items()}
        got = mixed(a, v, cap, yaw, delay=delay)
        for c, m in enumerate(mpc.ROW_FORMATS):
            rows = codes == c
            t = trk[m]
            assert torch.equal(got[rows], out[m][rows]), (step, m)
            for name in ("u_seq", "u_prev", "last_ref", "last_pred"):
                assert torch.equal(getattr(mixed, name)[rows], getattr(t, name)[rows]), (step, m, name)


def test_a_tracker_without_codes_is_the_one_it_always_was():
    trk = mpc.PlanTracker(4, "cpu", WB, SMAX, VMAX, compile_solver=False)
    assert trk.row_format is None
    seen = []
    trk._solver = lambda *a: seen.append(len(a)) or mpc.solve(*a)
    a, v, cap, yaw, delay, _, _ = _inputs(4, 3)
    trk(a, v, cap, yaw, delay=delay)
    assert seen == [11], "a tracker with one format hands the solver the eleven arguments it always did"
    trk.set_row_formats(torch.ones(4, dtype=torch.uint8))
    held = trk.row_format
    trk.set_row_formats(torch.zeros(4, dtype=torch.uint8))
    assert trk.row_format is held, "the codes are rewritten in place: a captured graph reads them by address"
    trk(a, v, cap, yaw, delay=delay)
    assert seen[-1] == 16, "the codes ride positionally, after the four optional slots"
    trk.set_row_formats(None)
    assert trk.row_format is None
    with pytest.raises(ValueError, match="per-row plan formats"):
        mpc.PlanTracker(4, "cpu", WB, SMAX, VMAX, spec=mpc.PlanSpec(speed_mode="envelope"),
                        compile_solver=False).set_row_formats(torch.zeros(4, dtype=torch.uint8))


def test_the_runtime_arms_refuse_a_mixed_tracker():
    """The grip and clearance arms decode every row as a linear plan; on a mixed tracker that would
    misread the wp4 rows, so the tracker refuses to run them rather than steer by a misreading."""
    trk = mpc.PlanTracker(4, "cpu", WB, SMAX, VMAX, spec=WP4, compile_solver=False)
    trk.set_row_formats(torch.tensor([1, 0, 1, 0], dtype=torch.uint8))
    trk._plan_hook = lambda action, v, cap: action
    a, v, cap, yaw, delay, _, _ = _inputs(4, 3)
    with pytest.raises(ValueError, match="plain tracker"):
        trk(a, v, cap, yaw, delay=delay)


def test_the_mixed_solve_traces_as_one_graph():
    """`replace` and the selection are trace-time Python: dynamo takes the per-row solve whole."""
    B = 16
    args = (*_inputs(B, 4), WP4, WB, SMAX, VMAX)
    codes = (torch.arange(B) % 2).to(torch.uint8)
    f = torch.compile(mpc.solve, fullgraph=True, backend="eager")
    got = f(*args, None, None, None, None, codes)
    want = mpc.solve(*args, row_format=codes)
    assert all(torch.equal(x, y) for x, y in zip(got, want))


# ----------------------------------------------------------------------------------------- the env
def _race(fmt: str, opp, B: int = 8, **kw):
    from f1sim import maps
    from f1sim.raceline import Raceline
    tr = maps.load(TRACK)
    rl = Raceline.build_cached(tr)
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(action_mode="plan", speed_mode=fmt, opponent_speed_mode=opp, race_size=2, opponent="teacher",
                     opp_speed_range=(0.7, 1.0), collision_mode="soft", compile_tracker=False, max_steps=400,
                     hist_len=0, **kw)
    env = common.make_env([tr], B, "cpu", ecfg, cfg=cfg, seed=4, rls=[rl])
    teacher = common.make_teacher([rl], env)
    env.reset(seed=4)
    return env, teacher


def test_a_mixed_race_commands_each_car_as_its_own_format_would():
    """A wp4 learner beside linear raceline-teacher opponents: on the first step -- before any car's
    plan has been seen by another's teacher -- the learners' commands are a wp4 race's and the
    opponents' a linear race's, bit for bit, and they are not the other format's."""
    races = {k: _race(*k) for k in (("wp4", None), ("linear", None), ("wp4", "linear"))}
    starts = [e.sim.state for e, _ in races.values()]
    assert all(torch.equal(starts[0], s) for s in starts)
    for env, teacher in races.values():
        env.step(env.teacher_label(teacher))
    mix = races[("wp4", "linear")][0]
    ego = mix.learner
    assert (mix.tracker.row_format == ego.to(torch.uint8)).all()        # wp4 = 1 on the learners' rows
    assert mix.opp_plan_spec.speed_mode == "linear" and mix.tracker.spec.speed_mode == "wp4"
    wp4, lin = races[("wp4", None)][0], races[("linear", None)][0]
    assert torch.equal(mix.last_cmd[ego], wp4.last_cmd[ego])
    assert torch.equal(mix.last_cmd[~ego], lin.last_cmd[~ego])
    assert not torch.equal(mix.last_cmd[~ego], wp4.last_cmd[~ego])
    assert torch.equal(mix.tracker.last_ref[~ego], lin.tracker.last_ref[~ego])


def test_one_format_envs_build_no_codes_and_share_the_trackers_spec():
    for fmt, opp in (("linear", None), ("wp4", None), ("wp4", "wp4"), ("linear", "linear")):
        env, _ = _race(fmt, opp, B=4)
        assert env.tracker.row_format is None and env._row_codes is None
        assert env.opp_plan_spec is env.tracker.spec


def test_a_respawned_opponent_holds_its_speed_in_its_own_format():
    """`prev_action` after a reset is "hold this speed" -- in the linear layout for an opponent row
    (a league checkpoint reads it back as its own previous action), in wp4's for the learner's."""
    env, _ = _race("wp4", "linear", B=4)
    v = env.last_cmd[:, 1]                         # the spawn speed (the reset's settling step moves the car)
    n = (v / env.ecfg.v_max_policy * 2 - 1)
    ego = env.learner
    lin_sp, wp_sp = mpc.plan_layout("linear").speed, mpc.plan_layout("wp4").speed
    assert torch.allclose(env.prev_action[~ego][:, lin_sp], n[~ego][:, None].expand(-1, 2))
    assert (env.prev_action[~ego][:, :lin_sp.start] == 0).all()
    assert torch.allclose(env.prev_action[ego][:, wp_sp], n[ego][:, None].expand(-1, 4))
    assert (env.prev_action[ego][:, :wp_sp.start] == 0).all()


def test_an_opponent_format_needs_a_plan_tracker_that_can_mix_it():
    from f1sim import maps
    tr = maps.load(TRACK)
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    for kw in ({"action_mode": "direct", "opponent_speed_mode": "wp4"},
               {"action_mode": "plan", "speed_mode": "envelope", "opponent_speed_mode": "linear"},
               {"action_mode": "plan", "speed_mode": "wp4", "opponent_speed_mode": "knots"}):
        with pytest.raises(ValueError):
            common.make_env([tr], 4, "cpu", EnvConfig(race_size=2, opponent="policy", **kw), cfg=cfg, seed=0)


def test_a_league_checkpoint_is_held_to_the_format_its_rows_are_read_in():
    from f1sim.learn.opponent_pool import _check_contract
    env = types.SimpleNamespace(ecfg=EnvConfig(action_mode="plan", speed_mode="wp4", opponent_speed_mode="linear"))
    _check_contract("league.pt", {"speed_mode": "linear"}, env)                 # the league as it is
    with pytest.raises(ValueError, match="plan-output"):
        _check_contract("ego.pt", {"speed_mode": "wp4"}, env)
    one = types.SimpleNamespace(ecfg=EnvConfig(action_mode="plan", speed_mode="wp4"))
    _check_contract("ego.pt", {"speed_mode": "wp4"}, one)
    with pytest.raises(ValueError, match="plan-output"):
        _check_contract("league.pt", {"speed_mode": "linear"}, one)


# ----------------------------------------------------------------------------------------- the GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_on_the_gpu_a_mixed_tracker_runs_eager_and_from_its_captured_graph_as_each_format():
    """Eager on CUDA, each row is its own format's solve bit for bit, as on the CPU. And the solver
    graph the training runtime captures (`SimGraphFastPath.capture_mpc`, recorded from the tracker's
    own call) carries the codes as a tensor input: rewritten in place between replays, the graphed
    tracker follows them and commands what an eager twin does."""
    from dataclasses import replace
    from f1sim.viewer.graph_fastpath import SimGraphFastPath
    dev = "cuda"
    B = 256
    args = _inputs(B, 7, dev)
    codes = (torch.arange(B, device=dev) % 2).to(torch.uint8)
    mixed = mpc.solve(*args, WP4, WB, SMAX, VMAX, row_format=codes)
    for c, m in enumerate(mpc.ROW_FORMATS):
        rows = codes == c
        alone = mpc.solve(*args, replace(WP4, speed_mode=m), WB, SMAX, VMAX)
        assert all(torch.equal(x[rows], y[rows]) for x, y in zip(mixed, alone)), m
    eager = mpc.PlanTracker(B, dev, WB, SMAX, VMAX, spec=WP4, compile_solver=False)
    graphed = mpc.PlanTracker(B, dev, WB, SMAX, VMAX, spec=WP4, compile_solver=False)
    for t in (eager, graphed):
        t.set_row_formats(codes)
    rec = {}

    def spy(*a):
        rec["args"] = a
        return mpc.solve(*a)

    graphed._solver = spy
    a, v, cap, yaw, delay, _, _ = args
    assert torch.equal(graphed(a, v, cap, yaw, delay=delay), eager(a, v, cap, yaw, delay=delay))
    sim = types.SimpleNamespace(device=torch.device(dev), cfg=types.SimpleNamespace(sim=types.SimpleNamespace(compile=False)))
    fp = SimGraphFastPath(sim)
    try:
        fp.capture_mpc(graphed, rec["args"])
        graphed._solver = fp.mpc
        for step in range(4):
            flip = ((torch.arange(B, device=dev) + step) % 3 == 0).to(torch.uint8)
            for t in (eager, graphed):
                t.set_row_formats(flip)
            a, v, cap, yaw, delay, _, _ = _inputs(B, 20 + step, dev)
            got, want = graphed(a, v, cap, yaw, delay=delay), eager(a, v, cap, yaw, delay=delay)
            assert torch.equal(got, want), step
            assert torch.equal(graphed.last_ref, eager.last_ref) and torch.equal(graphed.u_seq, eager.u_seq), step
    finally:
        graphed._solver = None
        del fp
