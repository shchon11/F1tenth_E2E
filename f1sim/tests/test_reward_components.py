import torch

from f1sim import Config, Track
from f1sim.gym_env import EnvConfig, F1VecEnv, REWARD_COMPONENT_KEYS


def test_reward_components_sum_to_reward_and_proximity_is_per_meter_driven() -> None:
    cfg = Config()
    cfg.sim.compile_mode = "none"
    cfg.lidar.n_beams = 36
    env = F1VecEnv(Track.generate_random(4), cfg,
                   EnvConfig(reward_proximity=0.5, proximity_speed_ref=4.0),
                   num_envs=2, device="cpu")
    env.reset(seed=5)
    _, reward, _, _, info = env.step(torch.zeros(2, 2))
    components = info["reward_components"]
    assert tuple(components) == REWARD_COMPONENT_KEYS
    torch.testing.assert_close(sum(components.values()), reward)
    # per metre *driven*: scaled by |v| dt, not by centerline progress, so a car that stops next to
    # a wall still pays and one that reverses away from it is not charged for progress it undid.
    travelled = env.sim.state[:, 3].abs() * env.sim.control_dt
    assert bool((components["proximity"].abs() <= travelled * 1.5 + 1e-6).all())


def test_plan_clearance_penalizes_a_reference_outside_the_track(monkeypatch) -> None:
    cfg = Config()
    cfg.sim.compile_mode = "none"
    cfg.lidar.n_beams = 36
    env = F1VecEnv(Track.generate_random(4), cfg,
                   EnvConfig(action_mode="plan", reward_plan_clearance=2.0),
                   num_envs=2, device="cpu")
    env.reset(seed=5)
    original = type(env.tracker).__call__

    def outside_reference(tracker, *args, **kwargs):
        command = original(tracker, *args, **kwargs)
        tracker.last_ref[:, :, 0] = 1000
        return command

    monkeypatch.setattr(type(env.tracker), "__call__", outside_reference)
    _, reward, _, _, info = env.step(torch.zeros(2, env.act_dim))
    assert bool((info["reward_components"]["plan_clearance"] < 0).all())
    torch.testing.assert_close(sum(info["reward_components"].values()), reward)
