"""Training/deployment parity and model plumbing (CPU)."""
import torch
import numpy as np

from f1sim import Track, Config
from f1sim.gym_env import F1VecEnv, EnvConfig
from f1sim.learn.obs import ObsBuilder, ObsSpec, flatten_obs
from f1sim.learn.model import ActorCritic, save_checkpoint, load_checkpoint


def test_obs_builder_matches_env_encoding():
    """Feed the ROS-side builder the raw values the env used; the tensors must match bit for bit."""
    tr = Track.generate_random(0, style="competition")
    env = F1VecEnv(tr, Config(), EnvConfig(scan_stack=3, action_history=2, speed_cap=6.0), num_envs=2, device="cpu")
    obs, info = env.reset(seed=1)
    spec = ObsSpec(n_beams=env.n_beams, scan_stack=3, action_history=2, range_max=env.range_max, v_max=env.ecfg.v_max_policy)
    b = ObsBuilder(spec)
    r = env.last_result
    b.push_action(env.act_hist[0, 1]); b.push_action(env.act_hist[0, 0])     # oldest first
    # the env's history after reset holds the reset scan in every slot; replay the same scans
    for k in range(3):
        scan_k = env.scan_hist[0, 2 - k] * env.range_max                     # oldest first
        s, p = b.build(scan_k.numpy(), float(r.odom[0, 3]), r.imu[0].mean(0).numpy(), r.imu_att[0, :2].numpy(), 6.0)
    scan_env, pro_env = flatten_obs(obs)
    assert torch.allclose(s[0], scan_env[0], atol=1e-6)
    assert torch.allclose(p[0], pro_env[0], atol=1e-5), (p[0], pro_env[0])
    # one more step: both sides advance identically
    a = torch.tensor([[0.3, -0.2], [0.0, 0.0]])
    obs, *_ = env.step(a); r = env.last_result
    b.push_action(a[0])
    s, p = b.build((obs["scan"][0, 0] * env.range_max).numpy(), float(r.odom[0, 3]), r.imu[0].mean(0).numpy(), r.imu_att[0, :2].numpy(), 6.0)
    scan_env, pro_env = flatten_obs(obs)
    assert torch.allclose(s[0], scan_env[0], atol=1e-6) and torch.allclose(p[0], pro_env[0], atol=1e-5)


def test_model_shapes_and_checkpoint_roundtrip(tmp_path):
    m = ActorCritic(3, 1080, 14, 17)
    scan, pro, priv = torch.rand(4, 3, 1080), torch.rand(4, 14), torch.rand(4, 17)
    a, logp = m.act(scan, pro)
    assert a.shape == (4, 2) and a.abs().max() <= 1.0 and logp.shape == (4,)
    lp, ent, v, d = m.evaluate(scan, pro, priv, a)
    assert v.shape == (4,) and lp.shape == (4,)
    n_actor = sum(p.numel() for p in m.actor.parameters())
    assert 0.3e6 < n_actor < 3e6, n_actor
    path = str(tmp_path / "m.pt"); save_checkpoint(path, m, {"spec": {"n_beams": 1080}})
    m2, extra = load_checkpoint(path)
    assert torch.allclose(m2.actor(scan, pro), m.actor(scan, pro)) and extra["spec"]["n_beams"] == 1080
