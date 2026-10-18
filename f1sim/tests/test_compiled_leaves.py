"""The substep's two hot leaves under torch.compile (`SimParams.compile_leaves`, 2026-10-19).

The console's CUDA graph records a substep op by op, and for one car that was 15 ms of every 25 ms
control step -- the user saw a light baseline run slower than real time. Compiling the tyre/body
dynamics and the IMU turns ~450 small ops a substep into a few fused kernels. They must be the same
computation -- on identical inputs the compiled leaves return what the eager ones do -- and the IMU's
noise, drawn by inductor's generator code, must be fresh noise of the same spread on every replay of
the console's graph, not one draw frozen into it.
"""
import pytest
import torch

from f1sim import Config, dynamics, imu, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _clone(args):
    return [a.clone() if torch.is_tensor(a) else (dict((k, v.clone() if torch.is_tensor(v) else v) for k, v in a.items())
                                                  if isinstance(a, dict) else a) for a in args]


def _recorded_leaf_calls():
    """One real call of each leaf, recorded from an env driving and turning (every term awake)."""
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"
    ecfg = EnvConfig(max_steps=100000, hist_len=0, action_mode="direct")
    env = common.make_env([maps.load("real:korea_2025_iccas")], 4, "cuda", ecfg, cfg=cfg, seed=3)
    env.reset()
    sim, rec = env.sim, {}
    assert sim._step_dynamics is dynamics.step_dynamics and sim._imu_substep is imu.substep   # off unless asked for
    for name in ("_step_dynamics", "_imu_substep"):
        eager = getattr(sim, name)

        def spy(*a, _n=name, _f=eager, **k):
            rec[_n] = (_clone(a), dict(k))
            return _f(*a, **k)
        setattr(sim, name, spy)
    act = torch.tensor([[0.1, 0.6]], device="cuda").expand(env.B, 2)
    for _ in range(40):
        env.step(act)
    return env, rec


def test_the_compiled_leaves_compute_what_the_eager_leaves_compute(monkeypatch):
    _env, rec = _recorded_leaf_calls()
    # The deterministic computation, with the IMU's draws pinned to zero on both sides (a fresh compile
    # under the patch, so the compiled code is traced with it).
    monkeypatch.setattr(torch, "randn", lambda *a, **k: torch.zeros(*a, **k))
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.zeros(*a, **k))
    for name, fn in (("_step_dynamics", dynamics.step_dynamics), ("_imu_substep", imu.substep)):
        args, kw = rec[name]
        out_e = fn(*_clone(args), **kw)
        out_c = torch.compile(fn, dynamic=False)(*_clone(args), **kw)
        out_e = out_e if isinstance(out_e, tuple) else (out_e,)
        out_c = out_c if isinstance(out_c, tuple) else (out_c,)
        for e, c in zip(out_e, out_c):
            assert torch.allclose(e, c, rtol=1e-5, atol=1e-5), (name, float((e - c).abs().max()))


def test_the_compiled_imu_noise_is_fresh_on_every_graph_replay():
    from f1sim.sim import compiled_leaves
    from f1sim.viewer.graph_fastpath import GraphedCallable
    _env, rec = _recorded_leaf_calls()
    args, kw = rec["_imu_substep"]
    shock = bool(kw.get("shock", False))

    def leaf(*a):
        return compiled_leaves()[1](*a, shock=shock)

    def eager(*a):
        return imu.substep(*a, shock=shock)
    fixed = _clone(args)                     # the graph reads its dict argument at the captured addresses
    g = GraphedCallable(leaf, fixed, name="imu leaf")
    g.adopt()
    replays = torch.stack([g(*fixed)[:, :12] for _ in range(400)])
    eagers = torch.stack([eager(*_clone(args))[:, :12] for _ in range(400)])
    assert not torch.equal(replays[0], replays[1]), "the graph replays one frozen draw"

    def spread(x):                           # median absolute deviation: the rare Pareto shocks
        return (x - x.median(0).values).abs().median(0).values     # would own a standard deviation
    sd_g, sd_e = spread(replays), spread(eagers)
    live = sd_e > 1e-6
    ratio = (sd_g[live] / sd_e[live])
    assert bool(live.any()) and float(ratio.min()) > 0.7 and float(ratio.max()) < 1.4, ratio


def test_the_console_sim_turns_them_on_and_only_on_cuda_without_the_whole_roll_compile():
    from f1sim.learn.watch import viewer_config
    from f1sim.sim import compiled_leaves
    assert viewer_config(False).sim.compile_leaves is True
    assert viewer_config(True).sim.compile_leaves is False
    for dev, want in (("cuda", True), ("cpu", False)):
        env = common.make_env([maps.load("real:korea_2025_iccas")], 1, dev,
                              EnvConfig(max_steps=1000, hist_len=0, action_mode="direct"),
                              cfg=viewer_config(False, randomize=False), seed=1)
        assert (env.sim._step_dynamics is compiled_leaves()[0]) == want, dev
