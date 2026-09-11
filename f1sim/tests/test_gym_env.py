import torch
from f1sim import Track, Config
from f1sim.gym_env import F1VecEnv, EnvConfig
from f1sim.raceline import Raceline
from f1sim.teacher import RacelineTeacher


def test_vec_env_shapes_and_autoreset():
    tr = Track.generate_random(0)
    env = F1VecEnv(tr, Config(), EnvConfig(scan_stack=2, scan_subsample=4, max_steps=60), num_envs=32, device="cpu")
    obs, info = env.reset(seed=1)
    # derived, not hardcoded: subsampling a 1081-beam scan by 4 leaves 271, not 1081 // 4 = 270.
    # The literal 270 here was written when the LiDAR had 1080 beams and went stale when the
    # measured count became 1081.
    n_sub = len(range(0, Config().lidar.n_beams, 4))
    assert obs["scan"].shape == (32, 2, n_sub) and obs["speed"].shape == (32, 1) and obs["prev_action"].shape == (32, 2 * env.ecfg.action_history)
    assert info["priv"].shape == (32, 8)
    assert torch.isfinite(obs["scan"]).all() and (obs["scan"] >= 0).all() and (obs["scan"] <= 1).all()
    n_final = 0
    for i in range(80):
        a = torch.rand(32, 2) * 2 - 1
        obs, rew, term, trunc, info = env.step(a)
        assert rew.shape == (32,) and torch.isfinite(rew).all()
        if "final" in info:
            n_final += info["final"]["ids"].numel()
            assert info["final_obs"]["scan"].shape[0] == info["final"]["ids"].numel()
    assert n_final > 0                                   # random driving must end some episodes (collision/timeout)
    assert (env.ep_step < 61).all()


def test_teacher_in_env_gets_positive_return():
    tr = Track.generate_random(1)
    rl = Raceline.build(tr)
    cfg = Config()
    env = F1VecEnv(tr, cfg, EnvConfig(max_steps=400, laps=1, spawn_lateral_std=0.15, spawn_yaw_std=0.1), num_envs=16, device="cpu")
    teacher = RacelineTeacher(rl, wheelbase=cfg.vehicle.lf + cfg.vehicle.lr)
    obs, info = env.reset(seed=0)
    total = torch.zeros(16); collided = 0; finished = 0
    for i in range(400):
        a = env.teacher_action_to_normalized(teacher(env.state, env.sim.P))
        obs, rew, term, trunc, info = env.step(a)
        total += rew
        if "final" in info:
            collided += int(info["final"]["collided"].sum()); finished += int((~info["final"]["collided"]).sum())
    assert collided <= 3, collided
    assert finished >= 10, finished
    assert total.mean() > 20, total.mean()
