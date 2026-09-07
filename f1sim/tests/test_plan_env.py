import torch
from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.learn.obs import flatten_obs


def test_plan_mode_teacher_drives_through_tracker():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks, rls = common.load_tracks(["gen:competition:0", "real:icra2022"], racelines=True)
    cfg = Config(); cfg.rand.enabled = False                     # nominal car: the tracker's model matches
    env = common.make_env(tracks, 32, dev, EnvConfig(action_mode="plan", speed_cap=5.0), cfg=cfg, seed=2)
    teacher = common.make_teacher(rls, env)
    obs, info = env.reset(seed=2)
    assert env.act_dim == 5 and obs["prev_action"].shape == (32, 10)
    scan, pro = flatten_obs(obs); assert pro.shape[1] == common.obs_spec(env).proprio_dim
    prog = torch.zeros(32, device=dev); coll = 0
    for _ in range(400):
        a = env.teacher_label(teacher)
        assert a.shape == (32, 5) and a.abs().max() <= 1.0
        obs, rew, term, trunc, info = env.step(a)
        prog += info["progress"]; coll += int(term.sum())
        assert "plan" in info and info["plan"].shape[0] == 32
    assert prog.mean() > 15.0, prog.mean()                  # 10 s of driving at a few m/s along the raceline
    assert coll <= 4, coll                                  # the plan + tracker keeps the teacher on the track
    p = env.plan_world(0); assert p is not None and p.shape[1] == 3          # x, y, speed
