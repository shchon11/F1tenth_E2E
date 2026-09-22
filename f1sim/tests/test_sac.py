"""`learn/sac.py`: the replay buffer rebuilds exactly what the env observed, and the Q functions start
as the value function they were grown from."""
import pytest

torch = pytest.importorskip("torch")

from f1sim.gym_env import EnvConfig                          # noqa: E402
from f1sim.learn import common                               # noqa: E402
from f1sim.learn.model import ActorCritic                    # noqa: E402
from f1sim.learn.obs import flatten_obs                      # noqa: E402
from f1sim.learn.sac import QNet, Replay                     # noqa: E402
from f1sim import Config                                     # noqa: E402


def _env(envs=8, max_steps=37, **kw):
    tracks, rls = common.load_tracks(["gen:control:9100"], racelines=True)
    cfg = Config(); cfg.sim.compile = False; cfg.lidar.n_beams = 61
    ecfg = EnvConfig(action_mode="plan", max_steps=max_steps, scan_stack=6, hist_len=4,
                     stagger_first_episode=True, **kw)
    return common.make_env(tracks, envs, "cpu", ecfg, cfg=cfg, seed=5, rls=rls)


def test_the_buffer_rebuilds_every_observation_it_was_given():
    """Stacks are not stored, they are rebuilt from each step's newest frame and the episode's age;
    an episode's last observation is kept aside. Over steps with resets (a staggered first episode
    and a 37-step limit), every slot's s and s' are what the env handed out -- to the 16-bit scan
    quantum, fp16 proprio, and exact privileged vector and action."""
    env = _env()
    obs, _ = env.reset(seed=5)
    n = env.B
    scan, pro = flatten_obs(obs)
    rb = Replay(200, n, scan.shape[1], scan.shape[2], pro.shape[1], env.privileged(env.last_result).shape[1],
                0, env.act_dim, "cpu")
    seen = []                       # (s_scan, s_pro, s_priv, act, next scan, next pro, next priv)
    for _ in range(90):
        scan, pro = flatten_obs(obs)
        priv = env.privileged(env.last_result)
        act = torch.rand(n, env.act_dim) * 2 - 1
        obs_n, rew, term, trunc, info = env.step(act)
        done = term | trunc
        n_scan, n_pro = flatten_obs(obs_n)
        n_priv = env.privileged(env.last_result)
        final = None
        if "final" in info:
            f = info["final"]; ids = f["ids"]
            f_scan, f_pro = flatten_obs(info["final_obs"])
            final = (ids, f_scan, f_pro, info["final_priv"], None)
            n_scan, n_pro, n_priv = n_scan.clone(), n_pro.clone(), n_priv.clone()
            n_scan[ids], n_pro[ids], n_priv[ids] = f_scan, f_pro, info["final_priv"]
        rb.add(scan[:, 0], pro, priv, None, act, rew, term, done, final)
        seen.append((scan.clone(), pro.clone(), priv.clone(), act.clone(), n_scan, n_pro, n_priv))
        obs = obs_n
    assert any(bool((rb.last[t]).any()) for t in range(90)), "the test has to cross an episode end"
    q = 1.0 / 65535.0
    for t in range(89):
        ti = torch.full((n,), t, dtype=torch.long)
        j = torch.arange(n)
        s_scan, s_pro, s_priv, act, n_scan, n_pro, n_priv = seen[t]
        if t >= scan.shape[1]:          # the buffer's first steps repeat their first frame (documented)
            torch.testing.assert_close(rb._stack(ti, j), s_scan, rtol=0, atol=q)
        torch.testing.assert_close(rb.pro[t].float(), s_pro, rtol=0, atol=2e-3 * s_pro.abs().max().item() + 1e-3)
        assert torch.equal(rb.priv[t], s_priv) and torch.equal(rb.act[t], act)
        # s' as `sample` builds it
        nt = torch.full((n,), t + 1, dtype=torch.long)
        slot = rb.fin_slot[t].to(torch.long)
        rebuilt = torch.where((slot >= 0)[:, None, None], rb._dq(rb.fin_scan[slot.clamp(min=0)]), rb._stack(nt, j))
        if t + 1 >= scan.shape[1] or bool((slot >= 0).all()):
            torch.testing.assert_close(rebuilt, n_scan, rtol=0, atol=q)


def test_the_q_functions_start_as_the_value_function():
    """Q(s, a) = V(s) for any a, and dQ/da = 0: the action enters through columns that start at zero,
    so the actor is not pulled anywhere before the critic has learned what an action does."""
    model = ActorCritic(n_stack=6, n_beams=61, proprio_dim=40, priv_dim=12, act_dim=8, scan_stem="resnet")
    q = QNet(model.critic, 8)
    scan = torch.rand(5, 6, 61); pro = torch.randn(5, 40); priv = torch.randn(5, 12)
    act = torch.rand(5, 8, requires_grad=True)
    v = model.critic(scan, pro, priv)
    qa = q(scan, pro, priv, act)
    torch.testing.assert_close(qa, v, rtol=0, atol=1e-6)
    qa.sum().backward()
    assert torch.count_nonzero(act.grad) == 0
    # and it can learn: the action columns receive a gradient
    assert q.c.pro[0].weight.grad is not None
    assert torch.count_nonzero(q.c.pro[0].weight.grad[:, -8:]) > 0
