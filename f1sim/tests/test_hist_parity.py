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
    spec = common.obs_spec(env); assert spec.hist_len == 4 and spec.proprio_dim == 1 + 12 + 1 + 8 + 4 * spec.row_dim
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


def test_stacked_scans_match_with_a_stride():
    """ppo_v15 on stacks its three scans 4 control steps apart (200 ms) so another car's closing speed is
    visible. The deployment ObsBuilder keeps its own scan buffer, so its stride has to line up with the
    env's frame for frame -- otherwise the real car feeds the policy an observation it never trained on."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks, rls = common.load_tracks(["gen:competition:0"], racelines=True)
    cfg = Config(); cfg.rand.enabled = False
    env = common.make_env(tracks, 2, dev, EnvConfig(action_mode="plan", speed_cap=4.0, scan_stack=3, scan_stride=4), cfg=cfg, seed=1)
    teacher = common.make_teacher(rls, env)
    spec = common.obs_spec(env); assert spec.scan_stride == 4 and spec.scan_stack == 3
    obs, _ = env.reset(seed=1)
    b = ObsBuilder(spec, "cpu")
    r = env.last_result
    b.build((obs["scan"][0, 0] * env.range_max).cpu().numpy(), float(r.odom[0, 3]), r.imu[0].mean(0).cpu().numpy(), r.imu_att[0, :2].cpu().numpy(), 4.0)
    warm = (spec.scan_stack - 1) * spec.scan_stride            # frames until both buffers hold real scans
    seen_difference = False
    for t in range(warm + 8):
        a = env.teacher_label(teacher)
        b.push_action(a[0].cpu())
        obs, rew, term, trunc, info = env.step(a)
        r = env.last_result
        s, _ = b.build((obs["scan"][0, 0] * env.range_max).cpu().numpy(), float(r.odom[0, 3]), r.imu[0].mean(0).cpu().numpy(), r.imu_att[0, :2].cpu().numpy(), 4.0)
        s_env, _ = flatten_obs({k: v[0:1] for k, v in obs.items()})
        assert s.shape == s_env.shape == (1, 3, spec.n_beams), (s.shape, s_env.shape)
        if t < warm:
            continue                                           # env.reset takes one extra zero-action step that the
        d = (s - s_env.cpu()).abs().max().item()               # car's builder cannot know about; both fill with copies
        assert d < 1e-4, (t, d)                                # of the first scan, and agree once the buffer is real
        if (s[0, 0] - s[0, 2]).abs().max().item() > 0.01:
            seen_difference = True
    assert seen_difference        # the stacked frames are genuinely apart in time, not three copies
