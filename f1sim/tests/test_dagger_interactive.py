"""DAgger with the interactive teacher, and a recurrent student.

Three claims:

1. **The loop runs end to end** with everything this branch adds at once -- the interactive
   teacher, the privileged opponent block in the observation, a GRU student trained by truncated
   BPTT, and the extra scan channels -- on CPU, from the real CLI.
2. **A chunk is a chunk.** `StepBuffer.sample_chunks` hands back contiguous steps of one env and
   marks exactly the steps that begin a new episode, because that is where the hidden state must
   not be carried in. A recurrent student trained one isolated step at a time is not the student
   that drives.
3. **The teacher fits its budget.** The contract's number is 3x `RacelineTeacher` per step, since
   the label is produced for every env on every collection step. The reported measurement is
   `work/bench/bench_teacher.py`; this is the regression guard.
"""
from __future__ import annotations

import functools
import sys
import time

import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.interactive_teacher import InteractiveTeacher
from f1sim.learn import common, dagger
from f1sim.learn.dagger import StepBuffer, actor_sequence
from f1sim.learn.memory import memory_spec
from f1sim.learn.model import ActorCritic

TRACK = "gen:control:1400"


@functools.lru_cache(maxsize=2)
def _parts(name: str = TRACK):
    from f1sim.raceline import Raceline
    t = maps.load(name)
    return t, Raceline.build_cached(t)


def test_dagger_runs_with_the_interactive_teacher_and_a_recurrent_student(tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_RUNS", str(tmp_path))
    monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
    argv = ["dagger", "--name", "it_smoke", "--tracks", TRACK, "--envs", "6",
            "--race-size", "3", "--opponent", "teacher",
            "--opp-events", "brake,stop,shift,defend,yield,line,oblivious", "--opp-event-rate", "1.0",
            "--opp-defend-prob", "0.3", "--opp-yield-prob", "0.2", "--opp-line-prob", "0.3",
            "--opp-oblivious-prob", "0.1", "--opp-speed", "0.6", "1.15",
            "--action-mode", "plan", "--teacher", "interactive", "--opp-token", "future",
            "--memory", "gru", "--memory-hidden", "32", "--scan-channels", "memory,edges",
            "--chunk-length", "4", "--scan-stack", "3", "--hist-len", "4",
            "--iters", "1", "--steps", "10", "--epochs", "1", "--batch", "16",
            "--eval-steps", "10", "--device", "cpu", "--eager", "--wandb", "disabled"]
    monkeypatch.setattr(sys, "argv", argv)
    dagger.main()
    ck = tmp_path / "it_smoke" / "student_latest.pt"
    assert ck.is_file()
    saved = torch.load(str(ck), map_location="cpu")
    assert saved["extra"]["teacher_kind"] == "interactive"
    assert saved["extra"]["opp_token"] == "future"
    assert saved["extra"]["spec"]["opp_token"] == "future"      # what the exporter and the node refuse on
    assert saved["meta"]["memory"] and saved["meta"]["scan_channels"]["channels"] == ["memory", "edges"]


def test_the_direct_action_space_refuses_the_interactive_teacher(monkeypatch):
    """Its candidates are plans. In `--action-mode direct` there is nothing for the search to
    produce, and falling back to the raceline teacher silently is how a run ends up being the
    unflagged one under another name."""
    monkeypatch.setattr(sys, "argv", ["dagger", "--action-mode", "direct", "--teacher", "interactive",
                                      "--race-size", "2", "--opponent", "teacher", "--device", "cpu"])
    with pytest.raises(SystemExit, match="action-mode plan"):
        dagger.main()


def _buffer(T=20, B=4, beams=16, act=8, breaks=((7, 1), (13, 2))):
    buf = StepBuffer(k=2, stride=1)
    g = torch.Generator().manual_seed(0)
    for t in range(T):
        new = torch.zeros(B, dtype=torch.bool)
        for tb, b in breaks:
            if t == tb:
                new[b] = True
        buf.add(torch.rand(B, beams, generator=g), torch.rand(B, 5, generator=g),
                torch.rand(B, act, generator=g), new)
    return buf.finalize(), breaks


def test_chunks_are_contiguous_and_flag_every_episode_boundary():
    buf, breaks = _buffer()
    torch.manual_seed(2)
    L = 6
    scan, pro, lab, keep = buf.sample_chunks(32, L, "cpu")
    assert scan.shape[:2] == (L, 32) and lab.shape[:2] == (L, 32)
    assert keep.shape == (L, 32) and keep.dtype == torch.bool
    # the flattening `actor_sequence` undoes is row-major (step, env): the label of chunk row (t, m)
    # has to be the buffer's own label at that step of that env
    for _ in range(3):
        torch.manual_seed(7)
        s2, p2, l2, k2 = buf.sample_chunks(8, L, "cpu")
        torch.manual_seed(7)
        s3, p3, l3, k3 = buf.sample_chunks(8, L, "cpu")
        assert torch.equal(l2, l3) and torch.equal(k2, k3)
    # every flagged step is a step this env actually started an episode on, and every one is flagged
    torch.manual_seed(11)
    scan, pro, lab, keep = buf.sample_chunks(200, L, "cpu")
    flagged = (~keep).nonzero()
    assert flagged.numel(), "200 chunks of 6 steps over a buffer with two boundaries hit neither"
    starts = {t for t, _b in breaks}
    for t, m in flagged.tolist():
        # find which (step, env) this row is: the label identifies it uniquely
        hit = (buf.L == lab[t, m]).all(-1).nonzero()
        assert hit.numel() >= 2
        assert int(hit[0, 0]) in starts


def test_the_recurrent_update_actually_uses_the_state():
    """The chunk path has to differ from running every step from a zero hidden state -- otherwise
    the GRU is being trained as a bias term and the policy that drives is a different one."""
    torch.manual_seed(3)
    small = dict(n_stack=2, n_beams=64, proprio_dim=5, priv_dim=9, act_dim=8,
                 scan_deltas=False, temporal_encoder="cnn", scan_stem="plain",
                 memory=memory_spec(hidden_size=16))
    m = ActorCritic(**small).eval()
    torch.nn.init.normal_(m.actor.memory.out.weight, std=0.3)      # a trained projection, not the zero init
    L, n = 5, 3
    g = torch.Generator().manual_seed(1)
    scan = torch.rand(L, n, 2, 64, generator=g)
    pro = torch.rand(L, n, 5, generator=g)
    keep = torch.ones(L, n, dtype=torch.bool)
    with torch.no_grad():
        walked = actor_sequence(m.actor, scan, pro, keep)
        # `Actor.forward` refuses a recurrent actor on purpose; `step` with h=None is the
        # zero-state forward this is meant to differ from.
        per_step = torch.cat([m.actor.step(scan[t], pro[t])[0] for t in range(L)], 0)
    assert not torch.allclose(walked, per_step, atol=1e-5), "the chunk ignored its own hidden state"
    # ... and with every step marked as an episode start the two agree again, because then the
    # state is cleared before each one
    with torch.no_grad():
        cleared = actor_sequence(m.actor, scan, pro, torch.zeros(L, n, dtype=torch.bool))
    assert torch.allclose(cleared, per_step, atol=1e-6)


@pytest.mark.parametrize("envs", [24])
def test_the_interactive_teacher_fits_its_three_times_budget(envs):
    """The regression guard on the contract's budget. The reported number is measured properly by
    `work/bench/bench_teacher.py` (2.14x on CPU at 96 envs); this only has to catch a change that
    makes the label an order of magnitude more expensive."""
    tr, rl = _parts()
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 108
    ecfg = EnvConfig(action_mode="plan", race_size=3, opponent="teacher", speed_cap=8.0,
                     opp_events=("brake", "stop", "shift"), opp_event_rate=1.0, compile_tracker=False)
    env = common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=1, rls=[rl])
    env.sim.warmup()
    rt = common.make_teacher([rl], env)
    it = InteractiveTeacher(rt, env)
    obs, _ = env.reset(seed=1)
    for _ in range(4):
        obs, *_ = env.step(env.teacher_label(rt))

    def block(fn, n=8):
        for _ in range(2):
            fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    ratios = sorted(block(lambda: env.teacher_label(it)) / block(lambda: env.teacher_label(rt))
                    for _ in range(3))
    assert ratios[1] < 3.0, f"interactive / raceline = {ratios[1]:.2f}x (median of {ratios})"


def test_a_resumed_memory_projection_is_only_fatal_when_something_is_leashed_to_it():
    """PPO's KL reference is evaluated with the recurrence off, so it is the frozen original only
    while the memory projection is still zero. A DAgger student distilled into a recurrent actor
    cannot supply one -- and that matters exactly when `--kl-coef` is positive."""
    from f1sim.learn.ppo import kl_reference_is_baseline
    fresh = ActorCritic(n_stack=2, n_beams=64, proprio_dim=5, priv_dim=9, act_dim=8,
                        scan_stem="plain", memory=memory_spec(hidden_size=8)).actor
    assert float(fresh.memory.out.weight.abs().max()) == 0.0
    assert kl_reference_is_baseline(fresh, True, 0.3) is True
    torch.nn.init.normal_(fresh.memory.out.weight, std=0.1)
    assert kl_reference_is_baseline(fresh, True, 0.0) is False        # logged, not leashed
    with pytest.raises(RuntimeError, match="frozen FEEDFORWARD baseline"):
        kl_reference_is_baseline(fresh, True, 0.05)
    # and a feedforward run is untouched either way
    ff = ActorCritic(n_stack=2, n_beams=64, proprio_dim=5, priv_dim=9, act_dim=8,
                     scan_stem="plain").actor
    assert kl_reference_is_baseline(ff, False, 0.3) is True
