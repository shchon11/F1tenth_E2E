"""The wrong-way meter (`learn.wrong_way_meter`): facing backwards along the lane while driving forwards."""
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.learn.wrong_way_meter import WrongWayMeter   # noqa: E402

DT = 0.025


class FakeSim:
    """Cars on a lane along +x (`pose_at_s` yaw 0), each with a fixed yaw and body-frame forward speed."""

    def __init__(self, yaw, vx):
        B = len(yaw)
        self.state = torch.zeros(B, 8)
        self.state[:, 2] = torch.tensor(yaw)
        self.state[:, 3] = torch.tensor(vx)
        self.steps = torch.zeros(B, dtype=torch.long)
        self.tid = torch.zeros(B, dtype=torch.long)
        self.control_dt = DT
        self.track = SimpleNamespace(pose_at_s=lambda s, tid: (torch.zeros(s.shape[0], 2), torch.zeros(s.shape[0])))

    def step(self):
        self.steps += 1
        return SimpleNamespace(state=self.state.clone(), s=torch.zeros(self.state.shape[0]))


def run(yaw, vx, steps=40, learner=None, between=None):
    sim = FakeSim(yaw, vx)
    env = SimpleNamespace(sim=sim, learner=torch.ones(len(yaw), dtype=torch.bool) if learner is None else learner)
    with WrongWayMeter(env) as m:
        for k in range(steps):
            if between is not None:
                between(k, sim)
            sim.step()
    return m.report()["wrong_way"]


def test_facing_backwards_and_driving_forwards_is_counted_once_per_onset():
    r = run([math.pi, 0.0], [3.0, 3.0])
    assert r["seconds"] == pytest.approx(40 * DT)
    assert r["share"] == pytest.approx(0.5)
    assert r["events"] == 1
    assert r["km"] == pytest.approx(2 * 3.0 * 40 * DT / 1000)


def test_backing_out_and_turning_at_a_standstill_are_not_wrong_way():
    r = run([0.0, math.pi], [-0.8, 0.3])       # backing out of a wedge; turned round, not yet driving
    assert r["seconds"] == 0.0 and r["events"] == 0


def test_only_the_evaluated_cars_count():
    r = run([math.pi, math.pi], [3.0, 3.0], learner=torch.tensor([False, True]))
    assert r["events"] == 1 and r["share"] == pytest.approx(1.0)


def test_a_new_episode_is_a_new_onset():
    def respawn(k, sim):
        if k == 20:
            sim.steps.zero_()
    r = run([math.pi], [3.0], between=respawn)
    assert r["events"] == 2
