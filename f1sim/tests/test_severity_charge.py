"""`EnvConfig.contact_severity_min` / `--contact-severity-min` (2026-10-15): charge only the hard contacts.

The user's goal: run fast; a graze is not fatal, a hard bang is. With the option at V > 0 an onset whose impact
severity (`f1sim.contact_severity`) is below V is free -- for the fixed collision and car-contact penalties and
for `--lagrange`'s onset counts, which read the same reward components -- and one at or above V is charged
exactly as before. V = 0 is every run before it (the v1 bit-identity digest is the step-for-step proof).
Scenes from `test_contact_severity`: a marker post, an open floor with duct walls, two-car races.
"""
from __future__ import annotations

import math
import sys

import pytest
import torch

from f1sim.gym_env import EnvConfig, F1VecEnv, REWARD_COMPONENT_KEYS
from f1sim.learn.ppo import lagrange_step, severity_onsets, severity_rates
from test_contact_severity import POST, _cfg, _floor, _post_floor, _script

COLL = REWARD_COMPONENT_KEYS.index("collision")
CAR = REWARD_COMPONENT_KEYS.index("car_contact")


def _env(track, B: int, v_min: float, race_size: int = 1) -> F1VecEnv:
    cfg = _cfg()
    env = F1VecEnv(track, cfg, EnvConfig(race_size=race_size, opponent="policy", collision_mode="soft",
                                         reward_collision=-1.0, reward_car_contact=1.0, speed_cap=6.0,
                                         contact_severity_min=v_min),
                   num_envs=B, device="cpu")
    env.reset(seed=3)
    return env


def test_severity_zero_is_todays_env():
    """V = 0 measures nothing extra and charges every onset, as before (the v1 bit-identity digest is the
    step-for-step proof; this is the switch itself)."""
    env = _env(_post_floor(), 2, 0.0)
    assert env.sim.measure_impacts is False
    steps = _script(env, [[POST[0] - 1.0, POST[1], 0.0], [POST[0] - 0.45, POST[1], 0.0]], [2.0, 0.3], 80)
    paid = torch.stack([c[:, COLL] for c, _ in steps])                    # (T, 2)
    assert ((paid < 0).sum(0) >= 1).all(), "both cars hit the post and are charged at V = 0"
    assert not any("wall_onset_all" in info for _, info in steps)


def test_a_soft_prop_contact_is_free_and_a_hard_one_is_charged_once():
    env = _env(_post_floor(), 2, 1.0)
    assert env.sim.measure_impacts is True
    steps = _script(env, [[POST[0] - 1.0, POST[1], 0.0], [POST[0] - 0.45, POST[1], 0.0]], [2.0, 0.3], 80)
    paid = torch.stack([c[:, COLL] for c, _ in steps])
    all_on = torch.stack([info["wall_onset_all"] for _, info in steps])
    charged = torch.stack([info["wall_onset"] for _, info in steps])
    assert int(all_on[:, 1].sum()) >= 1, "the 0.3 m/s nudge was an onset"
    assert int((paid[:, 1] < 0).sum()) == 0 and int(charged[:, 1].sum()) == 0      # ... and free
    assert int((paid[:, 0] < 0).sum()) == 1 and int(charged[:, 0].sum()) == 1      # the 2 m/s hit, once
    # the wall constraint counts what the component charged: one contact, from car 0
    prog = torch.full(paid.shape, 0.05)
    comps = torch.stack([c for c, _ in steps])
    spec = {"wall": {"index": COLL, "weight": 1.0, "target": 0.0, "lam": 0.0, "rate": float("nan")}}
    lagrange_step(torch.zeros_like(paid), comps, prog, torch.ones_like(paid), spec, 0.0, 10.0)
    km = float(prog.sum()) / 1000.0
    assert abs(spec["wall"]["rate"] * km - 1.0) < 1e-6


def test_a_wall_hit_is_charged_and_its_onset_counted_both_ways():
    env = _env(_floor(), 1, 1.0)
    steps = _script(env, [[10.0, 0.75, -math.pi / 2]], [4.0], 12)
    paid = torch.stack([c[:, COLL] for c, _ in steps])
    all_on = torch.stack([info["wall_onset_all"] for _, info in steps])
    assert int(all_on.sum()) >= 1
    first = int(torch.nonzero(all_on[:, 0])[0])
    assert float(paid[first, 0]) < 0 and int((paid < 0).sum()) == 1     # the 4 m/s hit, once


def test_a_car_graze_is_free_and_a_car_hit_is_charged_for_its_whole_contact():
    env = _env(_floor(), 4, 1.0, race_size=2)
    # race 0: a 3 m/s rear-end; race 1: a 0.48 m/s side merge
    steps = _script(env, [[10.0, 15.0, 0.0], [11.0, 15.0, 0.0], [10.0, 25.0, 0.08], [10.0, 25.40, 0.0]],
                    [5.0, 2.0, 6.0, 6.0], 30)
    hit = torch.stack([info["car_collision"] for _, info in steps])       # (T, 4) every step in contact
    car = torch.stack([c[:, CAR] for c, _ in steps])
    assert bool(hit[:, 2].any()), "the side merge never touched"
    assert int((car[:, 2:] < 0).sum()) == 0                               # the graze: free all along
    assert bool((car[:, 0] < 0).any())                                    # the hit: the car behind pays ...
    on = torch.stack([info["car_onset_all"] for _, info in steps])
    hard = torch.stack([info["car_onset_hard"] for _, info in steps])
    assert on[:, 2].sum() >= 1 and hard[:, 2].sum() == 0 and hard[:, 0].sum() == 1
    # ... and `--lagrange car=` counts it once, however many steps the bodies overlap
    prog = torch.full(car.shape, 0.05)
    spec = {"car": {"index": CAR, "weight": 1.0, "target": 0.0, "lam": 0.0, "rate": float("nan"), "onset": True}}
    comps = torch.stack([c for c, _ in steps])
    lagrange_step(torch.zeros_like(car), comps[:, :1], prog[:, :1], torch.ones_like(car[:, :1]), spec, 0.0, 10.0)
    assert abs(spec["car"]["rate"] * float(prog[:, :1].sum()) / 1000.0 - 1.0) < 1e-6


def test_a_car_graze_is_charged_at_severity_zero():
    env = _env(_floor(), 2, 0.0, race_size=2)
    steps = _script(env, [[10.0, 25.0, 0.08], [10.0, 25.40, 0.0]], [6.0, 6.0], 30)
    car = torch.stack([c[:, CAR] for c, _ in steps])
    assert bool((car < 0).any())


def test_the_option_refuses_terminate_and_the_legacy_charge():
    with pytest.raises(ValueError, match="soft"):
        F1VecEnv(_floor(), _cfg(), EnvConfig(contact_severity_min=1.0), num_envs=1, device="cpu")
    with pytest.raises(ValueError, match="collision_counts_cars"):
        F1VecEnv(_floor(), _cfg(), EnvConfig(contact_severity_min=1.0, collision_mode="soft",
                                             collision_counts_cars=True), num_envs=1, device="cpu")
    with pytest.raises(ValueError, match=">= 0"):
        F1VecEnv(_floor(), _cfg(), EnvConfig(contact_severity_min=-1.0, collision_mode="soft"), num_envs=1, device="cpu")


def test_the_trainer_logs_the_onsets_both_ways():
    """`--contact-severity-min`'s log: every onset and the charged ones, per km of progress (the constraint's unit)."""
    env = _env(_post_floor(), 2, 1.0)
    steps = _script(env, [[POST[0] - 1.0, POST[1], 0.0], [POST[0] - 0.45, POST[1], 0.0]], [2.0, 0.3], 80)
    rows = torch.arange(2)
    on = torch.stack([severity_onsets(info, rows) for _, info in steps])            # (T, 2, 4)
    prog = torch.full(on.shape[:2], 0.05)                                  # the scene itself barely moves
    rates = severity_rates(on, prog, torch.ones_like(prog))
    km = float(prog.clamp_min(0).sum()) / 1000.0
    assert abs(rates["contacts/wall_hard_per_km"] * km - 1.0) < 1e-6                   # the one 2 m/s hit
    assert rates["contacts/wall_all_per_km"] * km >= 2.0                               # ... and the nudges
    assert rates["contacts/car_onset_all_per_km"] == 0.0                               # a solo run: no cars


@pytest.mark.parametrize("argv, why", [
    (["--contact-severity-min", "1.0"], "--collision-mode soft"),
    (["--contact-severity-min", "1.0", "--collision-mode", "soft", "--legacy-car-wall-charge"], "legacy"),
    (["--contact-severity-min", "-0.5", "--collision-mode", "soft"], ">= 0"),
])
def test_ppo_refuses_the_option_where_it_cannot_mean_anything(monkeypatch, argv, why):
    from f1sim.learn import ppo
    monkeypatch.setattr(sys, "argv", ["f1sim.learn.ppo", *argv])
    with pytest.raises(SystemExit) as e:
        ppo.main()
    assert why in str(e.value)
