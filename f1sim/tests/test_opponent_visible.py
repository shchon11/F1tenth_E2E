"""`F1VecEnv.opponent_visible`: an auxiliary target about the other car only where the scan holds it."""
import torch

from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.params import Config
from test_profile_executor import _open_floor


def _visible_with_other_at(dx: float) -> bool:
    cfg = Config(); cfg.sim.compile = False; cfg.rand.enabled = False
    env = common.make_env([_open_floor()], 2, "cpu", EnvConfig(action_mode="plan", race_size=2, collision_mode="soft",
                                                               compile_tracker=False), cfg=cfg, seed=1)
    env.reset(seed=1)
    env.sim.reset(torch.arange(2), torch.tensor([[10.0, 20.0, 0.0], [10.0 + dx, 20.0, 0.0]]), torch.zeros(2))
    env.step(torch.zeros(2, env.act_dim))
    return bool(env.opponent_visible()[0])


def test_a_car_ahead_is_visible_and_one_behind_is_not():
    assert _visible_with_other_at(3.0)
    assert not _visible_with_other_at(-3.0)                       # the 90 degrees behind the LiDAR are blind
