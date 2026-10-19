"""The trajectory tracking meter (`learn.tracking_meter`): the car against the time-indexed reference it
was handed, h control steps after each issue."""
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim import dynamics as dyn                                   # noqa: E402
from f1sim.learn.tracking_meter import TrackingMeter, resample, walk_path   # noqa: E402

DT = 0.025


class FakeSim:
    """One car per row driving at `v` along +x, drifting sideways at `vy` (world frame)."""

    def __init__(self, B, v=2.0, vy=0.0):
        self.state = torch.zeros(B, 8)
        self.state[:, dyn.IVX] = v                       # already at speed: the walk starts from the measured speed
        self.steps = torch.zeros(B, dtype=torch.long)
        self.control_dt = DT
        self.v, self.vy = v, vy

    def step(self):
        self.state[:, 0] += self.v * DT
        self.state[:, 1] += self.vy * DT
        self.state[:, dyn.IVX] = self.v
        self.steps += 1
        return None


def plan_env(B, v=2.0, vy=0.0):
    sim = FakeSim(B, v, vy)
    spec = SimpleNamespace(dt=0.05, N=12, a_max=6.0, a_brake=5.0)
    tr = SimpleNamespace(spec=spec, last_ref=None)
    return SimpleNamespace(sim=sim, tracker=tr, act_dim=8, learner=torch.ones(B, dtype=torch.bool),
                           external_driven=torch.zeros(B, dtype=torch.bool), _all_external=False,
                           _plan_pose=None)


def straight_plan(B, v):
    """The tracker reference of a straight plan at constant speed, body frame, on 0, 0.05, ..., 0.6 s."""
    t = torch.arange(13, dtype=torch.float32) * 0.05
    ref = torch.zeros(B, 13, 4)
    ref[..., 0] = v * t
    ref[..., 3] = v
    return ref


def run_plan(env, steps):
    m = TrackingMeter(env)
    with m:
        for _ in range(steps):
            env._plan_pose = env.sim.state[:, :3].clone()
            env.tracker.last_ref = straight_plan(env.sim.state.shape[0], env.sim.v)
            env.sim.step()
    return m.report()["tracking"]


def test_a_car_on_its_plan_has_no_tracking_error():
    r = run_plan(plan_env(3), 60)
    assert r["source"].startswith("plan")
    for h in ("0.1", "0.2", "0.3", "0.5"):
        assert r["horizons"][h]["pos_rmse_m"] == pytest.approx(0.0, abs=1e-5)
        assert r["horizons"][h]["speed_rmse_mps"] == pytest.approx(0.0, abs=1e-5)
    assert r["horizons"]["0.5"]["n"] == 3 * (60 - 20 + 1)


def test_a_sideways_drift_is_lateral_error_growing_with_the_horizon():
    """0.2 m/s across a straight plan re-anchored at the car every step: h seconds later the car is
    0.2 h to the side of where the plan put it, and nothing along it."""
    r = run_plan(plan_env(2, v=2.0, vy=0.2), 60)
    for hs in (0.1, 0.2, 0.3, 0.5):
        h = r["horizons"][f"{hs:.1f}"]
        assert h["lat_rmse_m"] == pytest.approx(0.2 * hs, rel=1e-3)
        assert h["lon_rmse_m"] == pytest.approx(0.0, abs=1e-5)
    assert r["objective_pos_rmse_m"] == pytest.approx(0.1, rel=1e-3)


def test_a_reset_row_drops_the_references_of_its_old_episode():
    env = plan_env(2)
    m = TrackingMeter(env)
    with m:
        for k in range(40):
            if k == 30:
                env.sim.steps[1] = 0                      # row 1 starts a new episode at step 30 ...
                env.sim.state[1, 1] = 5.0                 # ... somewhere else entirely
            env._plan_pose = env.sim.state[:, :3].clone()
            env.tracker.last_ref = straight_plan(2, env.sim.v)
            env.sim.step()
    r = m.report()["tracking"]
    assert r["horizons"]["0.5"]["pos_rmse_m"] == pytest.approx(0.0, abs=1e-5)   # the jump is never compared


def test_the_planner_waypoints_are_walked_from_the_cars_projection():
    """A planner ego 0.1 m beside a straight waypoint path at the path's speed: the cross-track error at
    issue is 0.1 m, and so is the lateral error at every horizon."""
    B = 2
    sim = FakeSim(B, v=2.0)
    sim.state[:, 1] = 0.1
    env = SimpleNamespace(sim=sim, tracker=None, act_dim=2, learner=torch.ones(B, dtype=torch.bool))
    xs = torch.arange(-10, 200, dtype=torch.float32) * 0.1
    m = TrackingMeter(env)
    with m:
        for _ in range(50):
            xy = torch.stack([xs, torch.zeros_like(xs)], -1)[None].expand(B, -1, -1).clone()
            m._planner_ref = (torch.arange(B), xy, torch.full((B, xs.numel()), 2.0))
            sim.step()
    r = m.report()["tracking"]
    assert r["source"].startswith("planner")
    assert r["xtrack_rmse_m"] == pytest.approx(0.1, rel=1e-4)
    for h in ("0.1", "0.5"):
        assert r["horizons"][h]["lat_rmse_m"] == pytest.approx(0.1, rel=1e-3)
        assert r["horizons"][h]["lon_rmse_m"] == pytest.approx(0.0, abs=2e-3)


def test_walk_path_accelerates_within_the_bound_and_resample_keeps_a_line():
    xy = torch.stack([torch.arange(0, 100, dtype=torch.float32) * 0.1, torch.zeros(100)], -1)[None]
    ref, xt = walk_path(xy, torch.full((1, 100), 4.0), torch.tensor([[0.0, 0.0]]), torch.tensor([2.0]),
                        0.05, 10, a_max=6.0, a_brake=5.0)
    v_walk = torch.diff(ref[0, :, 0]) / 0.05
    assert float(v_walk[0]) == pytest.approx(2.3, abs=1e-4)          # 2 + 6 * 0.05 after the first step
    assert float(xt[0]) == pytest.approx(0.0, abs=1e-6)
    lin = torch.zeros(1, 13, 4)
    lin[..., 0] = torch.arange(13) * 0.05 * 3.0
    out = resample(lin, 0.05, 0.025, 21)
    assert torch.allclose(out[0, :, 0], torch.arange(21) * 0.025 * 3.0, atol=1e-6)
