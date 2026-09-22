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


def test_the_buffer_keeps_the_draw_and_the_critic_sees_what_the_env_executed(monkeypatch, tmp_path):
    """The replay holds the Gaussian draw `u`, not `clamp(u)`: the AWAC actor fits the policy's
    likelihood of what it drew. Fitted to clamped draws instead, a mean near the edge is pulled
    inward every update -- sac2 slowed from 8.0 to 8.4 s a lap that way. The Q functions and the
    env are still given the clamped action. A few steps of `train` with a wide policy, so draws
    fall outside [-1, 1], through both the critic and the actor updates."""
    import types
    from f1sim.learn import sac as sac_mod
    env = _env(envs=4)
    obs, _ = env.reset(seed=5)
    scan, pro = flatten_obs(obs)
    priv = env.privileged(env.last_result)
    model = ActorCritic(n_stack=scan.shape[1], n_beams=scan.shape[2], proprio_dim=pro.shape[1],
                        priv_dim=priv.shape[1], act_dim=env.act_dim, scan_stem="resnet")
    with torch.no_grad():
        model.actor.log_std.fill_(0.4)                  # std 1.5: most draws land outside [-1, 1]
    stored, executed_by_env, q_inputs = [], [], []
    add, step, q_fwd = Replay.add, env.step, QNet.forward
    monkeypatch.setattr(Replay, "add", lambda self, *x, **k: (stored.append(x[4].clone()), add(self, *x, **k))[1])
    monkeypatch.setattr(env, "step", lambda act: (executed_by_env.append(act.clone()), step(act))[1])
    monkeypatch.setattr(QNet, "forward", lambda self, s, p, v, act: (q_inputs.append(act.detach()), q_fwd(self, s, p, v, act))[1])
    a = types.SimpleNamespace(total=4 * 24, cap0=9.0, cap1=9.0, cap_steps=1, grip_budget_penalty=0.0, seed=0)
    hyper = sac_mod.SACHyper(buffer=256, batch=8, updates_per_step=1, start=4 * 12, critic_warmup=0,
                             actor_every=1, n_step=3, contact_frac=0.0)
    lid = env.learner_ids
    sac_mod.train(a, env=env, model=model, obs=obs, lid=lid, device=torch.device("cpu"), cond_dim=0,
                  cond_mode="none", cond_spec=None, dial=None, dial_new=torch.zeros(env.B, dtype=torch.bool),
                  out=str(tmp_path), progress_log=types.SimpleNamespace(write=lambda r: None),
                  run=types.SimpleNamespace(log=lambda *x, **k: None), spec=None, steps_base=0,
                  t_start=0.0, save=lambda *x: None, hyper=hyper, gamma=0.99, log_every=1000,
                  save_every=1000, amp=False)
    assert len(stored) == len(executed_by_env) == 24
    for u, act in zip(stored, executed_by_env):
        torch.testing.assert_close(u.clamp(-1, 1), act[lid], rtol=0, atol=0)
    draws = torch.cat(stored)
    assert (draws.abs() > 1).float().mean() > 0.2, "the draws kept must include the ones the env clamped"
    assert q_inputs and max(float(x.abs().max()) for x in q_inputs) <= 1.0


def test_n_step_targets_stop_at_endings_and_bootstrap_truncations_from_their_final():
    """`sample(n_step=...)`: the return sums rewards up to the window or the episode's end, whichever
    comes first; a real ending (`term`) is not bootstrapped; a truncation is, from the final
    observation kept for it, discounted by gamma^(steps summed); otherwise from the slot n on."""
    g = torch.Generator().manual_seed(3)
    T, n, k, N, n_step, gamma = 60, 5, 3, 7, 6, 0.9
    rb = Replay(T, n, k, N, 4, 2, 0, 2, "cpu")
    for t in range(45):
        last = torch.rand(n, generator=g) < 0.12
        term = last & (torch.rand(n, generator=g) < 0.5)
        ids = torch.nonzero(last & ~term).flatten()
        final = (ids, torch.rand(ids.numel(), k, N, generator=g), torch.randn(ids.numel(), 4, generator=g),
                 torch.randn(ids.numel(), 2, generator=g), None) if ids.numel() else None
        rb.add(torch.rand(n, N, generator=g), torch.randn(n, 4, generator=g), torch.randn(n, 2, generator=g),
               None, torch.rand(n, 2, generator=g), torch.randn(n, generator=g), term, last, final)
    bt = rb.sample(400, 0.0, g, n_step=n_step, gamma=gamma)
    for b in range(400):
        t0, j = int(bt["ti"][b]), int(bt["j"][b])
        ret, disc, K = 0.0, None, None
        for m in range(n_step):
            ret += gamma ** m * float(rb.rew[t0 + m, j])
            if bool(rb.last[t0 + m, j]):
                K = m
                break
        if K is None:
            disc = gamma ** n_step
            want = rb._stack(torch.tensor([t0 + n_step]), torch.tensor([j]))[0]
        elif bool(rb.term[t0 + K, j]):
            disc, want = 0.0, None
        else:
            disc = gamma ** (K + 1)
            want = rb._dq(rb.fin_scan[int(rb.fin_slot[t0 + K, j])])
        assert abs(float(bt["ret"][b]) - ret) < 1e-5 and abs(float(bt["disc"][b]) - disc) < 1e-6, (b, t0, j)
        if want is not None:
            assert torch.equal(bt["n_scan"][b], want), (b, t0, j)
