"""The grip head and the sideslip price: the two ways the policy is pointed at the friction limit."""
import math
import torch

from f1sim import Config, Track
from f1sim.gym_env import EnvConfig, F1VecEnv, REWARD_COMPONENT_KEYS
from f1sim.learn.model import ActorCritic


def test_grip_head_shares_the_trunk_and_trains() -> None:
    m = ActorCritic(3, 36, 12, 5, act_dim=2)
    scan, pro, priv, act = torch.rand(4, 3, 36), torch.rand(4, 12), torch.rand(4, 5), torch.rand(4, 2)
    logp, ent, val, d, grip, opp = m.evaluate_aux(scan, pro, priv, act)
    assert grip.shape == (4,) and opp.shape == (4, 3)
    mu_only = m.actor(scan, pro)
    torch.testing.assert_close(d.mean, mu_only)                    # same action mean as the plain path
    (grip ** 2).mean().backward()
    assert m.actor.mlp[0].weight.grad is not None and m.actor.pro[0].weight.grad is not None
    assert m.actor.mu.weight.grad is None                          # the action head does not


def test_sideslip_is_priced_beyond_the_free_angle() -> None:
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 24
    env = F1VecEnv(Track.generate_random(3), cfg, EnvConfig(reward_sideslip=1.0, reward_progress=0.0),
                   num_envs=2, device="cpu")
    env.reset(seed=1)
    env.sim.state[0, 3], env.sim.state[0, 4] = 4.0, 0.0                       # straight
    env.sim.state[1, 3], env.sim.state[1, 4] = 4.0, 4.0 * math.tan(math.radians(20))   # 20 deg drift
    _, _, _, _, info = env.step(torch.zeros(2, env.act_dim))
    ss = info["reward_components"]["sideslip"]
    assert float(ss[0]) == 0.0 and float(ss[1]) < -0.01, ss.tolist()
    assert "sideslip" in REWARD_COMPONENT_KEYS


def test_priv_mu_index_points_at_friction() -> None:
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 24
    for m in (1, 2):
        env = F1VecEnv(Track.generate_random(3), cfg, EnvConfig(race_size=m, opponent="policy"), num_envs=2 * m, device="cpu")
        env.reset(seed=1)
        priv = env.privileged(env.last_result)
        torch.testing.assert_close(priv[:, env.priv_mu_index], env.sim.P["mu"])
