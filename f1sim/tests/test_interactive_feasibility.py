"""Oriented body feasibility gates ranking independently of soft cost weights."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from f1sim.interactive_teacher import InteractiveTeacher, TeacherCost
from f1sim.mpc import ACT_DIM, PlanSpec
from f1sim.raceline import Raceline
from f1sim.teacher import RacelineTeacher


def scoring_scene(monkeypatch, lateral, *, present=True):
    angle = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    line = Raceline.from_xy(np.stack([8 * np.cos(angle), 8 * np.sin(angle)], 1), np.full(80, 5.0))
    teacher = InteractiveTeacher(RacelineTeacher(line), offsets=(0.0, 0.6), speeds=(1.0,),
                                  cost=TeacherCost(progress=1, wall=0, opp=0, clear=0, smooth=0))
    spec = PlanSpec()
    h = len(teacher.horizon_times(spec))
    lateral = torch.as_tensor(lateral, dtype=torch.float32)
    if lateral.ndim == 1:
        lateral = lateral[:, None]
    k, batch = lateral.shape
    world = torch.zeros(k, batch, h, 2)
    world[..., 1] = lateral[:, :, None]
    cand = torch.arange(k, dtype=torch.float32)[:, None, None].expand(k, batch, ACT_DIM) * 0.2
    state = torch.zeros(batch, 8)
    state[:, 3] = 5.0
    future = torch.zeros(batch, 1, h, 2)
    teacher.env = SimpleNamespace(M=2, opponent_future=lambda *a, **kw:
                                  (future, torch.full((batch, 1), present), torch.zeros(batch, 1, h)))
    monkeypatch.setattr(teacher, "rollout", lambda *a, **kw:
                        (world, torch.zeros(k, batch, h), torch.full((k, batch, h), 5.0)))
    monkeypatch.setattr(teacher, "_candidates", lambda *a, **kw: cand)
    progress = torch.tensor([-1e6, 0.0])[:, None].expand(k, batch)
    monkeypatch.setattr(teacher, "_progress_cost", lambda *a: progress)
    return teacher, cand, state, spec, progress


def test_hard_violation_cannot_buy_progress_even_with_zero_opponent_weight(monkeypatch):
    teacher, cand, state, spec, _ = scoring_scene(monkeypatch, [0.0, 2.0])
    total, parts = teacher.score(cand, state, None, 9.0, spec, parts=True)
    assert total[1, 0] < total[0, 0], "progress outweighed a hard-envelope violation"
    assert torch.isfinite(total).all()
    assert torch.equal(sum(parts.values()), total)
    assert parts["opp"].count_nonzero() == 0
    assert parts["hard_feasibility"][0, 0] > 0
    assert torch.equal(teacher.plan_action(state, v_max=9.0, spec=spec), cand[1])
    assert teacher.last_hard_clear[:, 0].tolist() == [False, True]
    assert teacher.last_blocked.tolist() == [False]


@pytest.mark.parametrize("case", ["no_env", "absent", "all_clear"])
def test_no_opponent_and_all_clear_scores_are_unchanged(monkeypatch, case):
    lateral = [2.0, 3.0] if case == "all_clear" else [0.0, 0.0]
    teacher, cand, state, spec, raw = scoring_scene(monkeypatch, lateral, present=case != "absent")
    if case == "no_env":
        teacher.env = None
    total, parts = teacher.score(cand, state, None, 9.0, spec, parts=True)
    assert torch.equal(total, raw)
    assert torch.equal(total, sum(parts.values()))
    assert parts["hard_feasibility"].count_nonzero() == 0
    assert teacher.last_hard_clear.all()
    assert not teacher.last_blocked.any()


def test_all_blocked_ties_use_original_cost_and_mark_label_invalid(monkeypatch):
    teacher, cand, state, spec, raw = scoring_scene(monkeypatch, [0.0, 0.1])
    first = teacher.plan_action(state, v_max=9.0, spec=spec)
    second = teacher.plan_action(state, v_max=9.0, spec=spec)
    assert torch.equal(first, second)
    assert torch.equal(first, cand[0])
    assert teacher.last_blocked.tolist() == [True]
    assert teacher.last_label_valid.tolist() == [False]
    assert not teacher.last_hard_clear.any()
    assert torch.isfinite(teacher.last_cost).all()
    assert torch.equal(teacher.last_cost, raw), "all-blocked fallback changed the original objective"


def test_feasibility_and_blocked_fallback_are_independent_per_car(monkeypatch):
    teacher, cand, state, spec, _ = scoring_scene(monkeypatch, [[0.0, 0.0], [2.0, 0.1]])
    action = teacher.plan_action(state, v_max=9.0, spec=spec)
    assert torch.equal(action[0], cand[1, 0])
    assert torch.equal(action[1], cand[0, 1])
    assert teacher.last_blocked.tolist() == [False, True]


def test_wall_collision_cannot_buy_progress_with_zero_wall_weight(monkeypatch):
    teacher, cand, state, spec, _ = scoring_scene(monkeypatch, [0.0, 2.0], present=False)
    teacher.track = SimpleNamespace(sample_edt=lambda xy, tid: xy[..., 1].clamp_min(0),
                                    t_res=torch.tensor([.05]), has_props=False)
    total = teacher.score(cand, state, None, 9., spec)
    assert total[1, 0] < total[0, 0]
    assert teacher.last_hard_clear[:, 0].tolist() == [False, True]
    assert teacher.last_label_valid.tolist() == [True]


def test_blocked_fallback_delays_contact_before_duration_then_cost():
    # Candidate 0 has attractive progress but contact first; candidate 1 delays
    # contact longest; candidate 2 ties its start but has longer contact duration.
    hits = torch.tensor([[[False, True, True, True]],
                         [[False, False, True, False]],
                         [[False, False, True, True]]])
    cost = torch.tensor([[-1e6], [5.], [-1e6]])
    assert InteractiveTeacher._contact_fallback(hits, cost).tolist() == [1]


@pytest.mark.parametrize('regular_clear', [True, False])
def test_immediate_evasion_only_eligible_when_ordinary_family_blocked(monkeypatch, regular_clear):
    teacher, _, state, spec, _ = scoring_scene(monkeypatch, [0., 2.])
    # Two regular offset plans plus their automatic brake, followed by four escapes.
    k, h = 7, len(teacher.horizon_times(spec))
    cand = torch.zeros(k, 1, ACT_DIM)
    world = torch.zeros(k, 1, h, 2)
    world[..., 1] = 2.  # all four escapes are physically clear
    world[:3, ..., 1] = 2. if regular_clear else 0.
    monkeypatch.setattr(teacher, 'rollout', lambda *a, **kw: (world, torch.zeros(k, 1, h), torch.zeros(k, 1, h)))
    monkeypatch.setattr(teacher, '_progress_cost', lambda *a: torch.tensor([0., 1., 2., -1e6, -2e6, -3e6, -4e6])[:, None])
    total, parts = teacher.score(cand, state, None, 9., spec, parts=True)
    assert teacher.last_hard_clear[3:].all()  # diagnostics retain true geometry
    assert torch.isfinite(total).all()
    assert torch.equal(total, sum(parts.values()))
    if regular_clear:
        assert total.argmin(0).item() < 3
        assert not teacher.last_candidate_eligible[3:].any()
    else:
        assert total.argmin(0).item() >= 3
        assert teacher.last_candidate_eligible.all()
    assert teacher.last_label_valid.item()
