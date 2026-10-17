"""A published baseline can drive an OPPONENT from its own sensors (2026-10-17, for the console).

The command a baseline car executes must be exactly what that driver says on that car's own scan --
not the caller's action for the row (the ego network would then silently drive it), and not the
tracker's plan for it.
"""
import math

import numpy as np
import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import baselines, common
from f1sim.opponent_slots import OpponentSlot


def _env(slot: dict, action_mode: str = "direct", envs: int = 4):
    from f1sim.raceline import Raceline
    track = maps.load("real:korea_2025_iccas")
    rl = Raceline.build_cached(track)
    cfg = Config(); cfg.sim.compile_mode = "none"
    ecfg = EnvConfig(max_steps=4000, hist_len=0, race_size=2, opponent="slots", opponent_slots=[slot],
                     action_mode=action_mode)
    return common.make_env([track], envs * 2, "cpu", ecfg, cfg=cfg, seed=3, rls=[rl])


@pytest.mark.parametrize("mode", ["direct", "plan"])
def test_an_ftg_opponent_executes_exactly_what_ftg_says_on_its_own_scan(mode):
    env = _env({"kind": "ftg", "options": {"max_speed": 8.0}}, mode)
    env.reset()
    rows = env.external_slot_rows[1]
    assert rows.numel() == 4 and not bool(env.learner[rows].any())
    ref = baselines.load("ftg", "none", max_speed=8.0)
    lid = env.cfg.lidar
    ref.bind_scanner(n_beams=int(lid.n_beams), fov=float(lid.fov), range_max=float(lid.range_max))
    act = torch.zeros(env.B, env.act_dim)
    for _ in range(5):
        lr = env.last_result
        rng = torch.nan_to_num(lr.scan[rows].float(), nan=lid.range_max, posinf=lid.range_max).clamp(0, lid.range_max)
        want = ref.command(ref.adapt(rng.numpy()), lr.odom[rows, 3].float().numpy())
        env.step(act)
        got = env.last_cmd[rows].numpy()
        steer = np.clip(want[:, 0], -float(env.s_max), float(env.s_max))
        speed = np.minimum(want[:, 1], env.speed_cap[rows].numpy())
        assert np.allclose(got[:, 0], steer, atol=1e-6) and np.allclose(got[:, 1], speed, atol=1e-5)


def test_the_learner_rows_are_not_overridden():
    env = _env({"kind": "disparity", "options": {"top_speed": 5.0}})
    env.reset()
    learner = torch.nonzero(env.learner).flatten()
    act = torch.zeros(env.B, env.act_dim); act[:, 0] = 0.25
    env.step(act)
    assert torch.allclose(env.last_cmd[learner, 0], torch.full((learner.numel(),), 0.25 * float(env.s_max)))


def test_an_external_car_is_predicted_at_constant_velocity():
    env = _env({"kind": "ftg"}, "plan")
    env.reset()
    env.step(torch.zeros(env.B, env.act_dim))
    rows = env.external_slot_rows[1]
    st = env.sim.state
    t = torch.tensor([0.5, 1.0])
    fut = env.car_future(t)
    c, s = torch.cos(st[rows, 2]), torch.sin(st[rows, 2])
    vw = torch.stack([st[rows, 3] * c - st[rows, 4] * s, st[rows, 3] * s + st[rows, 4] * c], 1)
    cv = st[rows, None, :2] + vw[:, None, :] * t[None, :, None]
    assert torch.allclose(fut[rows], cv, atol=1e-5)


def test_a_network_kind_without_weights_is_refused():
    from f1sim.opponent_slots import validate_slots
    with pytest.raises(ValueError, match="needs a weights file"):
        validate_slots([OpponentSlot.from_dict({"kind": "tinylidarnet"})], 2)


def test_an_external_slot_round_trips_through_json():
    s = OpponentSlot.from_dict({"kind": "ftg", "options": {"max_speed": 8.0}, "spawn": "random"})
    assert OpponentSlot.from_dict(s.to_dict()) == s
    w = OpponentSlot.from_dict({"kind": "end2race", "weights": "/w.pth"})
    assert OpponentSlot.from_dict(w.to_dict()) == w
