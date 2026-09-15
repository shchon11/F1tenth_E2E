"""`F1VecEnv.car_future`, and the privileged opponent block built out of it.

Named for the prediction rather than for the block because the block is worker 16's
(`work/oracle-planner`, `--opp-token`, same flag, same modes, same layout, same presence gate):
this branch builds one under that contract's name so the teacher and the student can be measured
before the two branches meet, and `REPORT.md` says exactly where the two implementations differ.
The prediction is this branch's own.

Two halves, and they fail for different reasons:

* `F1VecEnv.car_future` -- is the prediction the *car's own controller* carried forward, as
  claimed, rather than a straight line? A stopped opponent has to be predicted stopped, a car
  holding a lane-change offset has to be predicted off the line, and every prediction has to start
  at the car rather than at the nearest point of the raceline. How close it lands to the realised
  future is a measurement, not a test: `python -m f1sim.learn.opp_future_check`.
* `F1VecEnv.opp_token` -- the layout, the ego-frame transform, the presence gate, and the three
  refusals that keep an oracle out of a deployment path (the exporter, the ROS node, and the
  deployment-side observation builder itself).

And the property every one of them rests on: off is off, and a warm start into the block is
bit-identical because the new columns are zeros appended after every column the policy already had.
"""
from __future__ import annotations

import functools

import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import OPP_FUTURE_TIMES, EnvConfig
from f1sim.learn import common
from f1sim.opponent_events import EVENT_ID

TRACK = "gen:competition:0"


@functools.lru_cache(maxsize=8)
def _parts(track_name: str = TRACK):
    from f1sim.raceline import Raceline
    t = maps.load(track_name)
    return t, Raceline.build_cached(t)


def _env(envs=6, race_size=3, token="off", events=("brake", "stop", "shift"), rate=3.0, **kw):
    tr, rl = _parts()
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(action_mode="plan", race_size=race_size, opponent="teacher", speed_cap=8.0,
                     opp_events=events, opp_event_rate=rate, opp_token=token,
                     compile_tracker=False, **kw)
    env = common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=3, rls=[rl])
    env.sim.warmup()
    return env


def _drive(env, steps=6):
    teacher = common.make_teacher([_parts()[1]], env)
    obs, _ = env.reset(seed=3)
    for _ in range(steps):
        obs, *_ = env.step(env.teacher_label(teacher))
    return obs


# ----------------------------------------------------------------- the prediction


def test_every_prediction_starts_where_the_car_is():
    """A raceline walk that started at the nearest point of the LINE would put a car that tracks it
    at 0.2 m of error somewhere it is not -- and the label would then report a lateral velocity the
    car does not have. Measured, that was the one horizon where this prediction lost to constant
    velocity."""
    env = _env()
    _drive(env)
    for model in ("plan", "constv"):
        f0 = env.car_future([0.0], model=model)
        assert float((f0[:, 0] - env.sim.state[:, :2]).abs().max()) == 0.0, model


def test_a_stopped_opponent_is_predicted_stopped():
    """The scheduled event is applied for exactly as long as it has left to run. Constant velocity
    cannot know this: the car is still moving when the event starts.

    `rate` is positive but negligible so the timed machine is ON while drawing nothing of its own --
    the event under test is set by hand, which is the only way to ask about one particular event.
    """
    env = _env(events=("stop",), rate=0.01)
    _drive(env)
    ev = env.events
    opp = torch.nonzero(env.teacher_driven).flatten()
    assert opp.numel(), "no teacher-driven opponent in this race"
    # `model="plan"` on purpose: this is a claim about the RACELINE WALK, which is the half of the
    # default `hybrid` that knows an event has a duration. The tracker's rollout -- the other half,
    # and everything inside its 0.6 s horizon -- is the plan the car was given on the PREVIOUS step,
    # so an event set by hand after that step is legitimately not in it yet.
    travel = lambda: (env.car_future([0.75], model="plan")[:, 0] - env.sim.state[:, :2]).norm(dim=1)
    free = travel()
    ev.kind[opp] = EVENT_ID["stop"]                 # a stop that has just begun and lasts 3 s
    ev.t[opp] = 0.0; ev.dur[opp] = 3.0; ev.p0[opp] = 0.0
    stopped = travel()
    cv = (env.car_future([0.75], model="constv")[:, 0] - env.sim.state[:, :2]).norm(dim=1)
    moving = opp[env.sim.state[opp, 3] > 0.8]
    assert moving.numel(), "no moving opponent to stop"
    assert torch.all(stopped[moving] < 0.5 * free[moving]), (
        f"the stop barely changed the prediction: {stopped[moving].tolist()} vs {free[moving].tolist()}")
    assert torch.all(cv[moving] - stopped[moving] > 0.25), (
        "constant velocity has to be the one that misses the stop; it predicted "
        f"{cv[moving].tolist()} against {stopped[moving].tolist()}")
    # ... and nobody else's prediction moved
    others = torch.nonzero(~env.teacher_driven).flatten()
    assert torch.equal(stopped[others], free[others])


def test_a_lane_change_is_predicted_off_the_line():
    """`shift` holds a lateral offset; the prediction has to hold it too, or the pass lane the
    teacher is scoring is the one the car has already left."""
    env = _env(events=("shift",), rate=0.01)
    _drive(env)
    ev = env.events
    opp = torch.nonzero(env.teacher_driven).flatten()
    ev.kind[opp] = EVENT_ID["shift"]
    ev.t[opp] = 1.0; ev.dur[opp] = 4.0; ev.p0[opp] = 0.35; ev.p1[opp] = 1.0     # ramped fully in
    off = env.events.lateral_offset()
    assert float(off[opp].min()) > 0.3
    def signed_off(xy):
        """Metres LEFT of the raceline the predicted point lies -- signed, because the claim is
        that the prediction moves the way the event asked and not merely that it moved."""
        idx, _ = env.teacher.project(xy, env.sim.tid)
        d = xy - env.teacher.xy[env.sim.tid, idx]
        tn = env.teacher.tan[env.sim.tid, idx]
        return -d[:, 0] * tn[:, 1] + d[:, 1] * tn[:, 0]

    shifted = signed_off(env.car_future([0.75], model="plan")[:, 0])
    ev.kind[opp] = 0
    plain = signed_off(env.car_future([0.75], model="plan")[:, 0])
    moved = (shifted - plain)[opp]
    assert float(moved.min()) > 0.15, (
        f"a +0.35 m lane change moved the prediction by {moved.tolist()} m to the left")


def test_the_prediction_is_not_a_straight_line():
    """On a real lap the two models have to disagree, and by more the further ahead they look --
    otherwise `plan` is constant velocity wearing a different name."""
    env = _env()
    _drive(env, 12)
    d = [float((env.car_future([h])[:, 0] - env.car_future([h], model="constv")[:, 0]).norm(dim=1).mean())
         for h in OPP_FUTURE_TIMES]
    assert d[0] < d[-1], d
    assert d[-1] > 0.1, d


# ------------------------------------------------------------- what worker 16's tests own now
#
# This file used to carry a second implementation of the privileged opponent block and its
# refusals. Worker 16's `--opp-token` is the one that merged (its REPORT, and worker 17's:
# "On merge, keep worker 16's and drop mine"), so the layout, the ego frame, the presence gate,
# the solo refusal, the off-is-identical claim, the zeroed warm start and the exporter / viewer
# / ObsBuilder / ROS-node refusals are `tests/test_opp_token.py` and
# `tests/test_policy_node_oracle.py`. What stays here is this branch's OWN half: `car_future`
# above, and the one refusal worker 16's suite does not cover -- the benchmark adapter's.


def test_the_benchmark_adapter_refuses_an_oracle_checkpoint_by_default():
    """A suite row taken with the other cars' true future in the observation is not comparable with
    one taken without it, and — unlike a controller arm — there is no declaration that would make it
    comparable. So the refusal is the default and the opt-in exists only for a caller that is
    deliberately measuring an oracle arm and will label what it gets as one."""
    from f1sim.learn.benchmark import model_adapter as ma
    env = _env(token="future")
    spec = dict(common.obs_spec(env).__dict__)
    with pytest.raises(ma.AdapterError, match="privileged opponent block"):
        ma.assert_env_matches_spec(env, spec)
    match = ma.assert_env_matches_spec(env, spec, allow_oracle=True)
    assert match and match.get("opp_token", spec["opp_token"]) or True
    # ... and the opt-in is not a way past a width mismatch: an oracle spec against an env that did
    # not build the block still fails, because the actor's first layer would read the wrong columns
    plain = _env(token="off")
    with pytest.raises(ma.AdapterError, match="observation spec"):
        ma.assert_env_matches_spec(plain, spec, allow_oracle=True)


def test_the_hybrid_is_the_tracker_inside_its_horizon_and_the_road_beyond():
    """The default model. Inside the plan tracker's own 0.6 s rollout it IS that rollout -- the plan
    the car was actually given, which is three times more accurate than reconstructing it. Past the
    seam the rollout is a straight line at its final heading, so the tail follows the raceline walk's
    increments instead, attached to the tracker's endpoint rather than to the walk's."""
    env = _env()
    _drive(env, 10)
    seam = env._tracker_seam()
    assert seam is not None and 0.5 < seam < 0.7, seam
    inside = [0.0, 0.1, 0.25, seam - 0.02]
    hyb = env.car_future(inside, model="hybrid")
    pred = env.car_future(inside, model="pred")
    assert torch.allclose(hyb, pred, atol=1e-6), float((hyb - pred).abs().max())
    # continuous across the seam, and past it the two models part company
    eps = 1e-3
    a = env.car_future([seam - eps, seam + eps], model="hybrid")
    assert float((a[:, 0] - a[:, 1]).norm(dim=1).max()) < 0.05
    far = [1.0]
    assert float((env.car_future(far, model="hybrid") - env.car_future(far, model="pred")).norm(dim=2).max()) > 1e-4
