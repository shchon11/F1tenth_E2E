"""(11-18) The plan-hit cost can look at walls less far into the plan than at boxes (`PlanReward.plan_hit_wall_n`): a plan's far
end that grazes a wall is mostly the plan's own prediction error, a box on it is what the car is about to hit."""
import types

import torch

from f1sim.learn.plan_reward import PlanReward

CORNERS = torch.tensor([[0.29, 0.155], [0.29, -0.155], [-0.29, 0.155], [-0.29, -0.155]])


class _Track:
    def __init__(self, wall_x):
        self.wall_x = wall_x

    def sample_edt(self, pts, tt):
        return torch.where(pts[..., 0] >= self.wall_x, torch.zeros_like(pts[..., 0]), torch.ones_like(pts[..., 0]))


class _Boxes:
    def __init__(self, box_x):
        self.box_x = box_x

    def clearance(self, pts, eid):
        return torch.where((pts[:, 0] - self.box_x).abs() < 0.1, torch.zeros_like(pts[:, 0]), torch.ones_like(pts[:, 0]))


def _pr(wall_x=1e9, box_x=None, wall_n=None):
    pr = object.__new__(PlanReward)
    sim = types.SimpleNamespace(tid=torch.zeros(1, dtype=torch.long), eid=torch.zeros(1, dtype=torch.long), track=_Track(wall_x))
    pr.env = types.SimpleNamespace(sim=sim, procedural=None if box_x is None else _Boxes(box_x))
    pr.lid = torch.zeros(1, dtype=torch.long)
    pr.corners = CORNERS
    pr.plan_hit_share, pr.plan_hit_n, pr.plan_hit_wall_n, pr.contact_scale = 0.1, None, wall_n, 1.0
    return pr


def _plan(v=5.0, n=41):
    """straight along +x at v m/s, a pose every 0.025 s for 1 s (index 0 = the car now)"""
    t = torch.arange(n, dtype=torch.float32) * 0.025
    return (v * t)[None], torch.zeros(1, n), torch.zeros(1, n), torch.full((1, n), v)


def test_far_wall_priced_by_default_not_with_a_short_wall_horizon():
    # the front corners reach x = 4.0 at (4.0 - 0.29) / 5 = 0.74 s: past 0.5 s
    assert bool(_pr(wall_x=4.0).plan_cost(*_plan())["hit"][0])
    assert not bool(_pr(wall_x=4.0, wall_n=20).plan_cost(*_plan())["hit"][0])


def test_near_wall_still_priced_with_a_short_wall_horizon():
    out = _pr(wall_x=2.0, wall_n=20).plan_cost(*_plan())       # reached at 0.34 s
    assert bool(out["hit"][0]) and int(out["first"][0]) <= 20 and float(out["price"][0]) > 0


def test_boxes_look_at_the_whole_plan():
    out = _pr(box_x=4.0, wall_n=20).plan_cost(*_plan())         # a box 0.7 s ahead: walls cut at 0.5 s, boxes not
    assert bool(out["hit"][0]) and int(out["first"][0]) > 20
