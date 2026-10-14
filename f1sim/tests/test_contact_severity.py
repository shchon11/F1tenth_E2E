"""Contact severity (2026-10-15): how hard a contact was, measured, and an option to charge only the hard ones.

The goal the user set: run fast; a graze is not fatal, a hard bang is. Every contact used to be one onset
whatever it was, so the simulator now reports the speed of the two surfaces into each other along the
contact normal at the contact point (`StepResult.wall_impact`, `car_impact`), the evaluation files onsets
by it (`learn.severity_meter`), and `EnvConfig.contact_severity_min` charges only onsets at or above it.

Scenes are scripted on an open floor (duct walls, the 5 cm grid of the real map) and on a floor with one
marker post, so each contact's true normal speed is known.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import ndimage

from f1sim.contact_severity import after_onset, bucket_of, carry_wall_impact
from f1sim.gym_env import EnvConfig, F1VecEnv, REWARD_COMPONENT_KEYS
from f1sim.params import Config
from f1sim.sim import Simulator
from f1sim.track import StaticProp, Track

POST = (6.0, 3.0)


def _floor(res: float = 0.05) -> Track:
    """30 x 30 m, duct-hose walls 0.2 m thick at the border, a circular centerline in the middle."""
    n, k = int(round(30 / res)), int(round(0.2 / res))
    occ = np.zeros((n, n), bool)
    occ[:k] = occ[-k:] = True
    occ[:, :k] = occ[:, -k:] = True
    edt = ndimage.distance_transform_edt(~occ).astype(np.float32) * res
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    return Track(occupancy=occ, resolution=res, origin=(0.0, 0.0), edt=edt,
                 centerline=np.stack([15 + 10 * np.cos(th), 15 + 10 * np.sin(th)], 1), name="open",
                 duct=occ.copy(), tall=np.zeros_like(occ))


def _post_floor() -> Track:
    """12 x 12 m with one marker post (5.5 cm radius) on the centerline, far from the walls."""
    res, N = 0.05, 240
    occ = np.zeros((N, N), bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    cl = np.stack([6.0 + 3.0 * np.cos(a - np.pi / 2), 6.0 + 3.0 * np.sin(a - np.pi / 2)], 1)
    t = Track.from_occupancy(occ, res, (0.0, 0.0), cl, "post", duct=np.zeros_like(occ), tall=occ.copy(),
                             duct_height=0.33)
    t.props = (StaticProp("marker_post", POST[0], POST[1], 0.0, seed=0),)
    return t


def _cfg(terminate: bool = False) -> Config:
    cfg = Config()
    cfg.sim.device = "cpu"
    cfg.sim.compile = False
    cfg.rand.enabled = False
    cfg.sim.terminate_on_collision = terminate
    cfg.lidar.n_beams = 24
    return cfg


def _drive(sim: Simulator, poses, speeds, steps: int, steer=None):
    """Place the cars and hold their speed command; every step's result."""
    B = len(poses)
    sim.reset(torch.arange(B), torch.tensor(poses, dtype=torch.float32), torch.tensor(speeds, dtype=torch.float32))
    act = torch.zeros(B, 2)
    act[:, 1] = torch.tensor(speeds, dtype=torch.float32)
    if steer is not None:
        act[:, 0] = torch.tensor(steer, dtype=torch.float32)
    return [sim.step(act) for _ in range(steps)]


# ------------------------------------------------------------------ the simulator's measurement
def test_a_head_on_wall_hit_is_hard_and_a_shallow_scrape_is_soft():
    sim = Simulator(_floor(), _cfg(), num_envs=2, device="cpu")
    # car 0 square into the bottom wall at 4 m/s; car 1 along it, 0.05 rad in: 4 sin(0.05) = 0.2 m/s
    rs = _drive(sim, [[10.0, 0.75, -math.pi / 2], [20.0, 0.42, -0.05]], [4.0, 4.0], 40)
    imp = torch.stack([r.wall_impact for r in rs])                       # (T, 2)
    first = [int(torch.nonzero(imp[:, b] > 0)[0]) for b in range(2)]
    assert 3.8 < float(imp[first[0], 0]) < 4.05, imp[:, 0]               # the whole approach speed
    assert 0.15 < float(imp[first[1], 1]) < 0.25, imp[:, 1]              # the scrape's own
    # and the record is the step's alone: nothing on the steps before the first touch
    assert float(imp[:first[0], 0].abs().sum()) == 0.0


def test_a_terminate_crash_is_measured_only_when_asked():
    sim = Simulator(_floor(), _cfg(terminate=True), num_envs=1, device="cpu")
    rs = _drive(sim, [[10.0, 0.75, -math.pi / 2]], [3.0], 12)
    assert all(r.wall_impact is None for r in rs)                        # nobody asked: nothing computed
    sim.measure_impacts = True
    rs = _drive(sim, [[10.0, 0.75, -math.pi / 2]], [3.0], 12)
    hit = [r for r in rs if bool(r.wall_collision[0])]
    assert hit, "never reached the wall"
    assert 2.8 < float(hit[0].wall_impact[0]) < 3.05


def test_a_prop_hit_is_measured_on_the_props_normal():
    sim = Simulator(_post_floor(), _cfg(), num_envs=2, device="cpu")
    rs = _drive(sim, [[POST[0] - 1.0, POST[1], 0.0], [POST[0] - 0.45, POST[1], 0.0]], [2.0, 0.3], 80)
    for b, v in ((0, 2.0), (1, 0.3)):
        on = [r for r in rs if bool(r.wall_collision[b])]
        assert on, f"car {b} never touched the post"
        assert 0.9 * v < float(on[0].wall_impact[b]) < 1.02 * v, (b, float(on[0].wall_impact[b]))


def _race(B: int = 2) -> Simulator:
    sim = Simulator(_floor(), _cfg(), num_envs=B, device="cpu", race_size=2)
    sim.measure_impacts = True
    return sim


@pytest.mark.parametrize("poses, speeds, want", [
    ([[10.0, 15.0, 0.0], [11.0, 15.0, 0.0]], [5.0, 2.0], 3.0),                # rear-end, 3 m/s closing
    ([[10.0, 15.0, 0.0], [11.5, 15.0, math.pi]], [2.0, 2.0], 4.0),            # head-on, 2 + 2
    ([[10.0, 15.0, math.pi / 2], [10.0, 16.0, 0.0]], [3.0, 0.0], 3.0),        # T-bone
    # a corner-to-corner rear-end 29 cm across: least overlap would call it a sideways 0 m/s graze
    ([[10.0, 15.0, 0.0], [11.0, 15.29, 0.0]], [5.0, 2.0], 3.0),
    ([[10.0, 15.0, 0.08], [10.0, 15.40, 0.0]], [6.0, 6.0], 6.0 * math.sin(0.08)),  # side merge, 0.48 m/s
])
def test_a_car_contact_reads_the_closing_speed_at_its_first_step(poses, speeds, want):
    sim = _race()
    rs = _drive(sim, poses, speeds, 40)
    on = [r for r in rs if bool(r.car_collision.any())]
    assert on, "the cars never touched"
    got = on[0].car_impact
    assert bool(on[0].car_collision.all()) and float(got[0]) == float(got[1])  # one contact, one number
    assert abs(float(got[0]) - want) < 0.08 * want + 0.02, (float(got[0]), want)


def test_car_impact_is_none_unless_measured():
    sim = _race()
    sim.measure_impacts = False
    rs = _drive(sim, [[10.0, 15.0, 0.0], [11.0, 15.0, 0.0]], [5.0, 2.0], 10)
    assert all(r.car_impact is None for r in rs)
    assert any(bool(r.car_collision.any()) for r in rs)


# ------------------------------------------------------------------ onset association
def test_the_carry_gives_an_onset_the_impact_its_contact_began_with():
    """The wall flag is a step-end test on a nearest-cell field, and the soft contact pushes the car back
    out within milliseconds: a flag rising steps after the hit must still be charged the hit."""
    imp = torch.tensor([[0.0], [3.9], [0.4], [0.3], [0.0], [0.0], [0.2], [0.25], [0.0]])
    flag = torch.tensor([[0], [0], [0], [1], [0], [0], [0], [1], [0]], dtype=torch.bool)
    pend = torch.zeros(1)
    prev = torch.zeros(1, dtype=torch.bool)
    sev = []
    for t in range(imp.shape[0]):
        onset = flag[t] & ~prev
        prev = flag[t]
        carry = carry_wall_impact(pend, imp[t], flag[t])
        pend = after_onset(carry, onset)
        if bool(onset):
            sev.append(round(float(carry), 3))
    # the first flag rose two steps after the 3.9 m/s hit; the second belongs to a new touch at scrape speed
    assert sev == [3.9, 0.25]
    assert bucket_of(torch.tensor([0.0, 0.49, 0.5, 0.99, 1.0, 1.99, 2.0, 9.0])).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


# ------------------------------------------------------------------ the evaluation's meter
def _script(env: F1VecEnv, poses, speeds, steps: int):
    """Put the cars where the scene wants them and drive it; per step (components, info)."""
    B = env.B
    env.sim.reset(torch.arange(B), torch.tensor(poses, dtype=torch.float32), torch.tensor(speeds, dtype=torch.float32))
    out = []
    for _ in range(steps):
        a = torch.zeros(B, env.act_dim)
        a[:, 1] = torch.tensor(speeds, dtype=torch.float32) / env.ecfg.v_max_policy * 2.0 - 1.0
        _o, _r, _t, _tr, info = env.step(a)
        out.append((torch.stack([info["reward_components"][k] for k in REWARD_COMPONENT_KEYS], 1), info))
    return out


def test_the_meter_files_each_onset_by_its_severity():
    from f1sim.learn.severity_meter import ContactSeverityMeter
    env = F1VecEnv(_post_floor(), _cfg(), EnvConfig(collision_mode="soft", speed_cap=6.0), num_envs=3, device="cpu")
    env.reset(seed=3)
    with ContactSeverityMeter(env) as meter:
        assert env.sim.measure_impacts is True
        _script(env, [[POST[0] - 1.0, POST[1], 0.0], [POST[0] - 0.45, POST[1], 0.0], [POST[0] - 0.6, POST[1], 0.0]],
                [2.0, 0.3, 0.8], 80)
    assert env.sim.measure_impacts is False                               # put back
    rep = meter.report()["contact_severity"]
    w = rep["wall_prop"]
    assert w["onsets"] >= 3
    # the 2 m/s hit, the 0.8 m/s bump and the 0.3 m/s nudge (and the cars' re-touches, all soft)
    assert w["counts"][3] == 0 and w["counts"][2] == 1 and w["counts"][1] >= 1 and w["counts"][0] >= 1
    assert w["hard"] == 1 and abs(w["max_mps"] - 2.0) < 0.1               # >= 1 m/s: the hit alone
    assert rep["wall_prop_physical"]["hard"] == 1
    assert rep["prop"]["counts"] == w["counts"]                           # every one of them was the post
    assert rep["prop_physical"]["counts"] == rep["wall_prop_physical"]["counts"]
    assert "car" not in rep                                               # a solo run
