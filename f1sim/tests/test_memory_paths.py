"""Every inference path carries the hidden state, and clears it where the episode ends.

A recurrent policy is only the policy it was trained to be if each caller threads its state
through. Each test here drives a real memory checkpoint down one real path and asserts two things:
the state advances between steps (it is carried), and it is back to zero after a boundary (it is
cleared). A path that silently dropped the state would pass neither.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from f1sim.learn import common
from f1sim.learn.memory import (Hidden, PolicyRuntime, add_boundary_listener, broadcast_boundary,
                                memory_spec, policy_fn, runtime_for)
from f1sim.learn.model import ActorCritic, load_for_memory, save_checkpoint
from f1sim.learn.obs import ObsSpec
from f1sim.params import Config


def memory_checkpoint(tmp_path, name="mem", channels=("memory", "edges"), seed=99, priv_dim=17):
    """A warm-started memory checkpoint on the current six-scan/history ObsSpec, on disk."""
    spec = ObsSpec(n_beams=Config().lidar.n_beams, scan_stack=6, scan_stride=1,
                   hist_len=20, hist_stride=2, action_history=2, act_dim=8)
    meta = dict(n_stack=spec.scan_stack, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                priv_dim=priv_dim, act_dim=spec.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        ff = ActorCritic(**meta)
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    base = str(d / "ff.pt")
    save_checkpoint(base, ff, {"spec": dataclasses.asdict(spec)})
    model, _extra, _fresh = load_for_memory(
        base, "cpu", memory_spec(hidden_size=32),
        scan_channels=({"channels": list(channels)} if channels else None))
    with torch.random.fork_rng(), torch.no_grad():   # a trained-looking memory, so carrying it matters
        torch.manual_seed(seed + 1)
        model.actor.memory.out.weight.normal_(0, 0.2)
        model.critic.memory.out.weight.normal_(0, 0.2)
    path = str(d / "ppo_final.pt")
    save_checkpoint(path, model, {"spec": dataclasses.asdict(spec), "phase": "ppo", "run": name,
                                  "update": 1, "updates": 1, "steps": 1, "total_steps": 1,
                                  "cap": 9.0})
    return path, model, dataclasses.asdict(spec)


def tiny_env(n=4, steps=25):
    from f1sim.gym_env import EnvConfig
    from f1sim.track import Track
    tr = Track.generate_random(7)
    return common.make_env([tr], n, "cpu",
                           EnvConfig(scan_stack=6, scan_stride=1, hist_len=20, hist_stride=2,
                                     action_mode="plan", max_steps=steps, speed_cap=4.0),
                           cfg=Config(), seed=3)


# ================================================================= the runtime itself
def test_policy_fn_carries_and_clears(tmp_path):
    _p, model, _s = memory_checkpoint(tmp_path)
    model.eval()
    env = tiny_env(n=4)
    obs, _info = env.reset(seed=1)
    policy = policy_fn(model, env.B, device="cpu", deterministic=True)
    a0 = policy(obs)
    h1 = policy.runtime.hidden.actor.clone()
    assert a0.shape == (4, 8)
    assert float(h1.abs().max()) > 0, "the hidden state must have moved off zero"
    obs, _r, _t, _tr, _i = env.step(a0)
    policy(obs)
    h2 = policy.runtime.hidden.actor
    assert not torch.allclose(h1, h2), "a second step must advance the state, not restart it"
    policy.reset(torch.tensor([False, True, False, True]))
    assert float(policy.runtime.hidden.actor[:, 1].abs().max()) == 0.0
    assert float(policy.runtime.hidden.actor[:, 3].abs().max()) == 0.0
    assert float(policy.runtime.hidden.actor[:, 0].abs().max()) > 0.0
    # the occupancy channel is cleared with it: they are the same claim
    assert float(policy.runtime.scan.mem[1].min()) == 1.0
    assert float(policy.runtime.scan.mem[0].min()) < 1.0


def test_a_legacy_checkpoint_gets_an_inert_runtime(tmp_path):
    spec = ObsSpec(n_beams=Config().lidar.n_beams, scan_stack=6, hist_len=20, act_dim=8)
    with torch.random.fork_rng():
        torch.manual_seed(5)
        ff = ActorCritic(n_stack=6, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                         priv_dim=17, act_dim=8, scan_deltas=True)
    rt = runtime_for(ff, 4)
    assert not rt.stateful and rt.hidden is None and rt.scan is None
    scan = torch.rand(4, 6, spec.n_beams)
    assert rt.observe(scan) is scan            # nothing is added, not even a copy
    rt.reset(torch.tensor([True] * 4))         # and a boundary is a no-op


# ================================================================= common.rollout_metrics
def test_rollout_metrics_drives_and_resets_a_memory_policy(tmp_path):
    _p, model, _s = memory_checkpoint(tmp_path)
    model.eval()
    env = tiny_env(n=4, steps=8)               # short episodes: boundaries inside the rollout
    policy = policy_fn(model, env.B, device="cpu", deterministic=True)
    seen = {"resets": 0, "rows": 0}
    raw_reset = policy.reset

    def counting_reset(done=None):
        seen["resets"] += 1
        seen["rows"] += 0 if done is None else int(done.sum())
        raw_reset(done)

    policy.reset = counting_reset
    m = common.rollout_metrics(env, policy, 20)
    assert seen["resets"] == 21, "one clear at the seeded reset, then one per step"
    assert seen["rows"] > 0, "8-step episodes in a 20-step rollout must produce boundaries"
    assert m["episodes_started"] >= 4


# ================================================================= the benchmark adapter
def test_benchmark_policy_is_stateful_and_run_cell_clears_it(tmp_path):
    from f1sim.learn.benchmark import model_adapter as ma
    _p, model, _s = memory_checkpoint(tmp_path)
    model.eval()
    policy = ma.policy_for(model)
    assert hasattr(policy, "reset"), "a benchmark policy must be clearable per trial"
    env = tiny_env(n=2)
    obs, _info = env.reset(seed=2)
    policy(obs)
    assert float(policy.runtime.hidden.actor.abs().max()) > 0
    policy.reset()
    assert policy.runtime.hidden.actor is None      # a cleared Hidden builds zeros on next use
    # A cell of a different width rebuilds rather than carrying rows that are not the same cars.
    env2 = tiny_env(n=3)
    obs2, _i = env2.reset(seed=3)
    policy(obs2)
    assert policy.runtime.hidden.actor.shape[1] == 3


def test_run_cell_resets_the_policy_per_trial(tmp_path):
    """`runner.run_cell` is the benchmark's driving loop.

    A cell drives one trial per row and retires a row when its trial ends, so "per trial" is in
    practice the clear at the cell's single seeded reset -- which is the one that must happen, and
    the one a stateful policy would otherwise start the cell without. The per-step call is wired
    anyway, because a suite family that keeps driving past a boundary would need it.
    """
    from f1sim.learn.benchmark import runner
    calls = []

    class Recording:
        def __init__(self, env):
            self.env = env

        def __call__(self, obs):
            return torch.zeros(self.env.B, self.env.act_dim)

        def reset(self, done=None):
            calls.append(None if done is None else int(done.sum()))

    env = tiny_env(n=2, steps=6)
    pol = Recording(env)
    runner.run_cell(env, pol, suite="A", n_steps=12, frozen=False, seed=5)
    assert calls[0] is None, "the seeded reset clears everything"
    assert len(calls) >= 2, "and then the loop reports every step's boundaries"
    assert all(isinstance(c, int) for c in calls[1:]), calls


# ================================================================= the viewer worker's runner
def test_actor_runner_keeps_a_static_hidden_tensor_and_listens_for_boundaries(tmp_path):
    from f1sim.learn.watch import MemoryActorRunner, actor_runner
    _p, model, spec = memory_checkpoint(tmp_path)
    model.eval()
    run = actor_runner(model, torch.device("cpu"), compile_enabled=False)
    assert isinstance(run, MemoryActorRunner) and run.mode == "eager"
    scan = torch.rand(2, 6, spec["n_beams"])
    pro = torch.rand(2, ObsSpec(**spec).proprio_dim)
    mu1 = run(scan, pro)
    assert mu1.shape == (2, 8)
    buf = run.h
    assert buf is not None and float(buf.abs().max()) > 0
    mu2 = run(scan, pro)
    assert run.h is buf, "the hidden state is ONE tensor, updated in place"
    assert not torch.allclose(mu1, mu2), "the same observation must act on a carried state"
    # The env side announces a boundary; the runner is a listener and clears the rows that ended.
    broadcast_boundary(torch.tensor([True, False]), batch=2)
    assert float(run.h[:, 0].abs().max()) == 0.0
    assert float(run.h[:, 1].abs().max()) > 0.0
    broadcast_boundary(None, batch=2)
    assert float(run.h.abs().max()) == 0.0


def test_make_env_announces_boundaries_to_a_listener():
    """`common.make_env` wraps step/reset so a listener that cannot reach the env still hears it."""
    env = tiny_env(n=3, steps=4)
    rt = PolicyRuntime(memory=True)
    rt.hidden = Hidden(torch.ones(1, 3, 8), None)
    rt.batch = 3
    unlisten = add_boundary_listener(rt)
    try:
        env.reset(seed=1)
        assert rt.hidden.actor is None, "a full reset clears everything"
        rt.hidden = Hidden(torch.ones(1, 3, 8), None)
        for _ in range(6):                      # max_steps 4: at least one truncation
            env.step(torch.zeros(env.B, env.act_dim))
        assert float(rt.hidden.actor.abs().max()) == 0.0 or rt.hidden.actor is None
    finally:
        unlisten()
    # Unregistered: the env stops reaching it.
    rt.hidden = Hidden(torch.ones(1, 3, 8), None)
    env.reset(seed=1)
    assert float(rt.hidden.actor.abs().max()) == 1.0


def test_introspector_saliency_works_on_a_memory_actor(tmp_path):
    from f1sim.learn.watch import Introspector
    _p, model, spec = memory_checkpoint(tmp_path)
    model.eval()
    intro = Introspector(model)
    scan = torch.rand(2, 6, spec["n_beams"])
    pro = torch.rand(2, ObsSpec(**spec).proprio_dim)
    sal, mu = intro.saliency(scan, pro, 0)
    assert sal.shape == (spec["n_beams"],) and mu.shape == (8,)
    assert float(intro._rt.hidden.actor.abs().max()) > 0
    intro.saliency(scan, pro, 1)               # a different car is not this car's memory
    assert intro._focus == 1


# ================================================================= export
def test_export_presents_a_memory_actor_as_a_pure_function(tmp_path):
    """The exported graph takes the hidden state in and hands the next one back.

    Module state cannot cross into ONNX / TensorRT / a CUDA graph; an input and an output can. The
    sidecar JSON says what the deployment side has to build (a scan wider than `n_stack`, and which
    channels fill the extra rows) and what it has to keep between steps.
    """
    from f1sim.learn.export import ActorOnly, RecurrentActorOnly
    _p, model, spec = memory_checkpoint(tmp_path, channels=("memory", "edges"))
    model.eval()
    m = RecurrentActorOnly(model.actor).eval()
    n_beams, n_stack = spec["n_beams"], spec["scan_stack"]
    scan = torch.rand(1, n_stack + 2, n_beams)          # frames + the two extra channels
    pro = torch.rand(1, ObsSpec(**spec).proprio_dim)
    h = model.actor.initial_hidden(1)
    with torch.no_grad():
        action, h_next = m(scan, pro, h)
    assert action.shape == (1, 8) and h_next.shape == h.shape
    assert not torch.allclose(h_next, h), "the exported step must advance the state it was given"
    # And a feedforward checkpoint keeps the two-input graph it always had.
    ff = ActorCritic(n_stack=n_stack, n_beams=n_beams, proprio_dim=ObsSpec(**spec).proprio_dim,
                     priv_dim=17, act_dim=8, scan_deltas=True).eval()
    with torch.no_grad():
        assert ActorOnly(ff.actor)(torch.rand(1, n_stack, n_beams), pro).shape == (1, 8)


def test_a_scan_channel_only_runner_still_declares_its_width(tmp_path):
    """A checkpoint with channels but no memory still holds per-row state, so it has to declare a
    width to the boundary registry: a listener with none is sent every env's mask, including masks
    of the wrong shape, and the occupancy channel refuses those (rightly) with a ValueError."""
    from f1sim.learn.watch import MemoryActorRunner, actor_runner
    spec = ObsSpec(n_beams=Config().lidar.n_beams, scan_stack=6, scan_stride=1, hist_len=20,
                   hist_stride=2, action_history=2, act_dim=8)
    meta = dict(n_stack=6, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim, priv_dim=17,
                act_dim=8, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(12)
        ff = ActorCritic(**meta)
    base = str(tmp_path / "ff2.pt"); save_checkpoint(base, ff, {"spec": {}})
    model, _e, _f = load_for_memory(base, "cpu", None, scan_channels={"channels": ["memory"]})
    model.eval()
    run = actor_runner(model, torch.device("cpu"), compile_enabled=False)
    assert isinstance(run, MemoryActorRunner) and run.h is None    # channels, no hidden state
    run(torch.rand(2, 6, spec.n_beams), torch.rand(2, spec.proprio_dim))
    assert run.batch == 2
    broadcast_boundary(torch.tensor([True, True, True]), batch=3)   # a different env: not ours
    assert float(run.rt.scan.mem.min()) < 1.0
    broadcast_boundary(torch.tensor([True, False]), batch=2)
    assert float(run.rt.scan.mem[0].min()) == 1.0 and float(run.rt.scan.mem[1].min()) < 1.0


def test_the_viewer_worker_would_accept_a_memory_checkpoint(tmp_path):
    """The worker refuses a session whose checkpoint does not match the env it built. A memory
    checkpoint's `meta` carries two keys the check has never seen, and `n_stack` still means the
    frames the env emits (the extra channels are added on the policy side), so it must pass."""
    from f1sim.learn.watch import describe_checkpoint_line
    from f1sim.viewer.sim_worker import obs_spec_problems
    path, model, spec = memory_checkpoint(tmp_path)
    env_spec = ObsSpec(**spec)
    assert obs_spec_problems(model.meta, env_spec, [], env_spec.act_dim) == []
    line = describe_checkpoint_line(
        {"phase": "ppo", "run": "mem", "update": 3, "updates": 3, "total_steps": 1e6,
         "experiment": {"memory": model.meta["memory"],
                        "scan_channels": model.meta["scan_channels"]}}, path)
    assert "memory: gru h=32" in line and "scan memory,edges" in line, line
