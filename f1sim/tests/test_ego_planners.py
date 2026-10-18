"""A published planner can drive the ego car in `evaluate` (2026-10-17), with the follow cap the env
gives the same planner as an opponent."""
import importlib
import types

import torch

from f1sim import opponent_slots
from f1sim.learn import evaluate as ev


def test_every_ego_planner_kind_resolves_to_an_importable_planner():
    for k in ev.EGO_PLANNER_KINDS:
        mod, cls = opponent_slots.kind_of(k).teacher_factory.split(":")
        assert callable(getattr(importlib.import_module(mod), cls)), k


def _fake_env(follow, v_cap, v_max=10.0):
    env = types.SimpleNamespace(act_dim=2, M=2, ecfg=types.SimpleNamespace(v_max_policy=v_max),
                                sim=types.SimpleNamespace(state=torch.zeros(len(follow), 7)))
    env.follow_cap = lambda state: (torch.tensor(follow), torch.tensor(v_cap))
    return env


def test_the_follow_cap_slows_only_the_rows_that_are_following():
    env = _fake_env([True, False], [2.0, 2.0])
    an = torch.tensor([[0.1, 0.6], [-0.2, 0.6]])          # 0.6 -> 8 m/s on a 10 m/s scale
    out = ev._ego_follow(env, None, an)
    v = (out[:, 1] + 1) / 2 * 10.0
    assert abs(float(v[0]) - 2.0) < 1e-5                   # held behind the car ahead
    assert abs(float(v[1]) - 8.0) < 1e-5                   # nobody ahead: untouched
    assert torch.equal(out[:, 0], an[:, 0])                # steering is never touched


def test_the_follow_cap_never_raises_a_slower_command():
    env = _fake_env([True], [5.0])
    out = ev._ego_follow(env, None, torch.tensor([[0.0, -0.6]]))   # 2 m/s, below the cap
    assert abs(float((out[0, 1] + 1) / 2 * 10.0) - 2.0) < 1e-5


def test_alone_on_the_track_there_is_no_follow_cap():
    """A solo session has no `other_idx`; the cap must not be asked for (2026-10-18: the console crashed)."""
    env = _fake_env([True], [1.0]); env.M = 1
    an = torch.tensor([[0.0, 0.6]])
    assert torch.equal(ev._ego_follow(env, None, an), an)
