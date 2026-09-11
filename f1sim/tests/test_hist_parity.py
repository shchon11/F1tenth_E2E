"""The proprio history (speed, imu, roll/pitch, action rows) must be encoded identically by the env and
by the deployment-side ObsBuilder, step for step."""
import torch
from f1sim import Config
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.learn.obs import ObsBuilder, flatten_obs


def test_history_rows_match():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks, rls = common.load_tracks(["gen:competition:0"], racelines=True)
    cfg = Config(); cfg.rand.enabled = False
    env = common.make_env(tracks, 2, dev, EnvConfig(action_mode="plan", speed_cap=4.0, hist_len=4, hist_stride=2), cfg=cfg, seed=1)
    teacher = common.make_teacher(rls, env)
    spec = common.obs_spec(env)                                   # derive from the spec: the plan
    assert spec.hist_len == 4                                     # action dimension is not a constant
    assert spec.proprio_dim == 1 + spec.act_dim * spec.action_history + 1 + 8 + 4 * spec.row_dim
    obs, _ = env.reset(seed=1)
    b = ObsBuilder(spec, "cpu")
    b.push_action(env.act_hist[0, 1].cpu()); b.push_action(env.act_hist[0, 0].cpu())
    b._pending = None                                              # the two seeding pushes are not history rows
    r = env.last_result
    s, p = b.build((obs["scan"][0, 0] * env.range_max).cpu().numpy(), float(r.odom[0, 3]), r.imu[0].mean(0).cpu().numpy(), r.imu_att[0, :2].cpu().numpy(), 4.0)
    for t in range(12):
        a = env.teacher_label(teacher)
        b.push_action(a[0].cpu())
        obs, rew, term, trunc, info = env.step(a)
        r = env.last_result
        s, p = b.build((obs["scan"][0, 0] * env.range_max).cpu().numpy(), float(r.odom[0, 3]), r.imu[0].mean(0).cpu().numpy(), r.imu_att[0, :2].cpu().numpy(), 4.0)
        _, p_env = flatten_obs({k: v[0:1] for k, v in obs.items()})
        d = (p - p_env.cpu()).abs().max().item()
        assert d < 1e-4, (t, d)
