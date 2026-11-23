"""The race phase's grid (11-24): `EnvConfig.spawn_half_lap` lays a share of the 1v1 races out half a lap apart (the user's
events), the rest `spawn_gap` behind the other car; `F1VecEnv.spawn_gap_race` fixes the gap per race (exam4's conditions);
off, nothing is drawn (the grid of every earlier run). Builds small CPU envs."""
import functools

import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common

TRACK = "gen:competition:0"


@functools.lru_cache(maxsize=2)
def _track():
    return maps.load(TRACK)


def _env(envs=48, seed=7, **kw):
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    seat = {"kind": "ftg", "present": 1.0, "spawn": "ahead"}
    ecfg = EnvConfig(**{"race_size": 2, "opponent": "slots", "opponent_slots": [seat], "max_steps": 4000, "hist_len": 0,
                        "spawn_gap": (5.0, 15.0), **kw})
    return common.make_env([_track()], envs, "cpu", ecfg, cfg=cfg, seed=seed)


def _gap(env):
    """(G,) arc from the learner forward to the other car, in [0, L)"""
    s, L = env.sim.s.view(-1, 2), float(env.sim.track.length[0])
    return torch.remainder(s[:, 1] - s[:, 0], L), L


def test_half_of_the_races_start_half_a_lap_apart():
    env = _env(spawn_half_lap=0.5)
    env.reset(seed=3)
    g, L = _gap(env)
    half = env.half_lap_race
    assert 0.2 < float(half.float().mean()) < 0.8
    assert bool(((g[half] - 0.5 * L).abs() < 1.0).all())                   # opposite sides (a spawn may slide a little)
    assert bool(((g[~half] > 4.0) & (g[~half] < 16.5)).all())               # the rest 5-15 m behind it


def test_a_fixed_gap_per_race_and_off_means_no_draw():
    env = _env()
    G = env.B // env.M
    L = float(env.sim.track.length[0])
    env.spawn_gap_race = torch.where(torch.arange(G) % 2 == 0, torch.full((G,), 0.5 * L), torch.full((G,), 5.0))
    env.reset(seed=3)
    g, _ = _gap(env)
    assert bool(((g[0::2] - 0.5 * L).abs() < 1.0).all()) and bool(((g[1::2] - 5.0).abs() < 1.0).all())
    a, b = _env(seed=11), _env(seed=11, spawn_half_lap=0.0)
    a.reset(seed=5); b.reset(seed=5)
    assert torch.equal(a.sim.state, b.sim.state) and not bool(a.half_lap_race.any())
