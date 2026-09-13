"""The motion branch: where its gradient is allowed to go, where its state lives, what is exported.

The design claim is a sentence from the contract's addendum -- *the auxiliary losses attach to
`h_dyn` and the motion encoder only, never to the main hidden* -- and the reason is mechanical: ego
dynamics are the cheap way to drive any of these losses down, so a main representation that is
allowed to absorb them will, and the experiment will have measured nothing. That claim is a fact
about the autograd graph, so the test for it is a fact about the autograd graph.

The rest is the discipline the contract asks for: the flags off are byte-identical, the flag on is a
warm start that changes no action, `h_dyn` travels inside the one hidden tensor every path already
carries, and no train-time head reaches the exported graph.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from f1sim.learn.memory import Hidden, memory_spec, reset_hidden
from f1sim.learn.model import ActorCritic, load_for_memory, save_checkpoint
from f1sim.learn.motion import (MotionEncoder, OppMaskHead, dv_loss, mask_loss, motion_spec)
from f1sim.learn.obs import ObsSpec, motion_index_spec

SMALL = dict(n_stack=6, n_beams=256, proprio_dim=64, priv_dim=21, act_dim=8,
             scan_deltas=True, temporal_encoder="cnn", scan_stem="resnet")
ROWS = ["aligned", "aligned_prev", "aligned_valid"]
SPEC = ObsSpec(n_beams=SMALL["n_beams"], scan_stack=6, act_dim=8, action_history=2, hist_len=0)
CHAN = {"channels": ["memory", "edges"] + ROWS, "aligned": {"proprio": motion_index_spec(SPEC)}}


def build(heads=("mask", "dv"), seed: int = 0, **kw):
    torch.manual_seed(seed)
    return ActorCritic(**SMALL, memory=memory_spec(hidden_size=32), scan_channels=CHAN,
                       motion=motion_spec(hidden_size=16, channels=8, **kw),
                       motion_heads=list(heads)).eval()


def inputs(batch=3, seed=1, model=None):
    g = torch.Generator().manual_seed(seed)
    extra = len(CHAN["channels"])
    scan = torch.rand(batch, SMALL["n_stack"] + extra, SMALL["n_beams"], generator=g)
    pro = torch.rand(batch, SMALL["proprio_dim"], generator=g) * 2 - 1
    priv = torch.rand(batch, SMALL["priv_dim"], generator=g) * 2 - 1
    return scan, pro, priv


# ------------------------------------------------------------------ the gradient boundary
def test_the_auxiliaries_reach_the_motion_branch_and_nothing_else():
    """The whole design, as a statement about the autograd graph.

    Backward from the two auxiliary losses and ask which parameters received a gradient. The motion
    encoder, the motion GRU and the two heads must; the scan stem, the main GRU, the MLP, the action
    head and the other auxiliary heads must not. If the main state could be reached, the ego-motion
    shortcut would be available to these losses and the split would be decoration.
    """
    m = build()
    # Both heads' output layers start at exactly zero, which is deliberate (`learn.motion`) and
    # means the gradient reaching their INPUT is zero for exactly one update. Nudge them off zero
    # first, so what this measures is where the gradient goes once the path is alive rather than the
    # initialisation.
    with torch.no_grad():
        m.actor.opp_mask.net[2].weight.normal_(0.0, 0.1)
        m.actor.dv.net[2].weight.normal_(0.0, 0.1)
    scan, pro, priv = inputs()
    mu, grip, opp, fut, (mask, dv), h = m.actor.step_all(scan, pro, None, None)
    label = torch.zeros_like(mask); label[:, 40:60] = 1.0
    w = torch.ones(scan.shape[0])
    loss = mask_loss(mask, label, w)[0] + dv_loss(dv, torch.randn(scan.shape[0], 2),
                                                  torch.ones(scan.shape[0]), w)[0]
    loss.backward()
    got = {n for n, p in m.named_parameters() if p.grad is not None and float(p.grad.abs().max()) > 0}
    motion = {n for n in got if ".memory.motion." in n or n.startswith("actor.opp_mask.")
              or n.startswith("actor.dv.")}
    assert got == motion, sorted(got - motion)
    assert any(".memory.motion.encoder." in n for n in got), "the encoder must train"
    assert any(".memory.motion.gru." in n for n in got), "and so must the motion GRU"
    # and specifically, by name, the tensors the addendum says must NOT move
    for n in ("actor.stem.trunk.0.weight", "actor.memory.gru.weight_ih_l0", "actor.mlp.0.weight",
              "actor.mu.weight", "actor.opp.0.weight", "critic.memory.gru.weight_ih_l0"):
        p = dict(m.named_parameters())[n]
        assert p.grad is None or float(p.grad.abs().max()) == 0.0, n


def test_at_init_the_heads_starve_their_input_for_exactly_one_update():
    """The zero output layer's documented consequence, checked rather than asserted in a comment."""
    m = build()
    scan, pro, _priv = inputs()
    _mu, _g, _o, _f, (mask, dv), _h = m.actor.step_all(scan, pro, None, None)
    (mask.sum() + dv.sum()).backward()
    enc = dict(m.named_parameters())["actor.memory.motion.encoder.net.0.weight"]
    assert enc.grad is None or float(enc.grad.abs().max()) == 0.0
    for name in ("actor.opp_mask.net.2.weight", "actor.dv.net.2.weight"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and float(g.abs().max()) > 0, f"{name} must train from the first update"


def test_the_motion_projection_does_not_leak_into_the_heads():
    """`memory.motion.out` is the only path from `h_dyn` to the plan, and the heads do not use it.

    A gradient from the auxiliaries reaching `out` would mean the losses were shaping how the motion
    state is *used by the planner* rather than what it contains -- a different experiment.
    """
    m = build()
    scan, pro, _priv = inputs()
    _mu, _g, _o, _f, (mask, dv), _h = m.actor.step_all(scan, pro, None, None)
    (mask.sum() + dv.sum()).backward()
    out = m.actor.memory.motion.out.weight
    assert out.grad is None or float(out.grad.abs().max()) == 0.0


# ------------------------------------------------------------------ the state
def test_h_dyn_rides_inside_the_one_hidden_tensor():
    m = build()
    mem = m.actor.memory
    h = m.actor.initial_hidden(4)
    assert h.shape == (1, 4, mem.hidden_size + mem.motion.hidden_size)
    main, dyn = mem.split(h)
    assert main.shape[-1] == mem.hidden_size and dyn.shape[-1] == mem.motion.hidden_size
    # every path that carries a hidden state carries this one unchanged
    done = torch.tensor([True, False, False, True])
    z = reset_hidden(torch.ones_like(h), done)
    assert float(z[:, done].abs().max()) == 0.0 and float(z[:, ~done].min()) == 1.0
    scan, pro, _priv = inputs(batch=4)
    with torch.no_grad():
        _mu, h1 = m.actor.step(scan, pro, None, h)
    assert h1.shape == h.shape


def test_the_probe_reads_the_whole_state_and_the_heads_read_h_dyn():
    """One function apart, and the difference is the addendum's requirement, not an accident."""
    m = build()
    scan, pro, _priv = inputs()
    with torch.no_grad():
        _act, state, h_next = m.actor.probe_state(scan, pro, None, None)
    mem = m.actor.memory
    assert state.shape[-1] == mem.hidden_size + mem.motion.hidden_size
    head_in = m.actor.future_input(None, h_next)
    assert head_in.shape[-1] == mem.motion.hidden_size
    assert torch.equal(head_in, state[:, mem.hidden_size:])


def test_without_a_motion_branch_the_two_are_the_same_tensor():
    torch.manual_seed(0)
    m = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32)).eval()
    g = torch.Generator().manual_seed(2)
    scan = torch.rand(2, SMALL["n_stack"], SMALL["n_beams"], generator=g)
    pro = torch.rand(2, SMALL["proprio_dim"], generator=g)
    with torch.no_grad():
        _a, state, h_next = m.actor.probe_state(scan, pro, None, None)
    assert torch.equal(state, m.actor.future_input(None, h_next))


def test_the_encoder_reads_the_recorded_rows_and_not_the_trailing_ones():
    """A checkpoint records WHICH aligned rows its encoder was built over; enabling another channel
    beside them must not shift what it reads."""
    m = build()
    assert m.meta["motion"]["rows"] == ROWS
    assert m.actor.motion_rows == (2, 3, 4)         # within the extra block: memory, edges, then these
    scan, pro, _priv = inputs()
    rows = m.actor.motion_input(scan)
    assert rows.shape[1] == 3
    assert torch.equal(rows, scan[:, SMALL["n_stack"] + 2:SMALL["n_stack"] + 5])
    one = build(heads=(), rows=["aligned"])
    assert one.meta["motion"]["rows"] == ["aligned"] and one.actor.motion_rows == (2,)
    assert one.actor.motion_input(scan).shape[1] == 1


def test_the_future_head_moves_to_h_dyn_when_there_is_a_motion_branch():
    """The addendum puts every auxiliary on `h_dyn`, the future head included.

    Its input width follows, and the checkpoint records which tensor it was built over, so a head
    trained on one cannot be silently rebuilt on the other.
    """
    torch.manual_seed(0)
    m = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32), scan_channels=CHAN,
                    motion=motion_spec(hidden_size=16, channels=8),
                    future_head={"k": 20, "width": 32}).eval()
    assert m.meta["future_head"]["source"] == "motion"
    assert m.actor.future.net[0].in_features == 16
    scan, pro, _priv = inputs()
    _mu, _g, _o, fut, _mot, h = m.actor.step_all(scan, pro, None, None)
    assert fut.shape == (scan.shape[0], 7) and float(fut.abs().max()) == 0.0
    assert torch.equal(m.actor.future_input(None, h), h[-1][:, 32:])
    # and a head that recorded a different source is refused rather than reshaped
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="does not match this actor"):
        ActorCritic(**SMALL, memory=memory_spec(hidden_size=32), scan_channels=CHAN,
                    motion=motion_spec(hidden_size=16, channels=8),
                    future_head={"k": 20, "width": 32, "source": "memory"})


def test_the_future_head_gradient_also_stops_at_the_motion_branch():
    torch.manual_seed(0)
    m = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32), scan_channels=CHAN,
                    motion=motion_spec(hidden_size=16, channels=8),
                    future_head={"k": 20, "width": 32}).eval()
    with torch.no_grad():
        m.actor.future.net[2].weight.normal_(0.0, 0.1)
    scan, pro, _priv = inputs()
    _mu, _g, _o, fut, _mot, _h = m.actor.step_all(scan, pro, None, None)
    fut.pow(2).mean().backward()
    got = {n for n, p in m.named_parameters() if p.grad is not None and float(p.grad.abs().max()) > 0}
    stray = {n for n in got if not (".memory.motion." in n or n.startswith("actor.future."))}
    assert not stray, sorted(stray)


# ------------------------------------------------------------------ off is off
def test_the_flags_off_build_nothing():
    torch.manual_seed(0)
    plain = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32), scan_channels=CHAN)
    assert "motion" not in plain.meta and "motion_heads" not in plain.meta
    assert plain.actor.memory.motion is None and not plain.actor.has_motion
    assert plain.actor.opp_mask is None and plain.actor.dv is None
    assert not any("motion" in k or "opp_mask" in k or ".dv." in k for k in plain.state_dict())
    # and the RNG draw is untouched: building the same thing again gives the same weights
    torch.manual_seed(0)
    again = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32), scan_channels=CHAN)
    for (n, a), (_n2, b) in zip(plain.named_parameters(), again.named_parameters()):
        assert torch.equal(a, b), n


@pytest.mark.parametrize("heads", [(), ("mask",), ("mask", "dv")])
def test_warm_starting_the_motion_branch_is_bit_identical(tmp_path, heads):
    torch.manual_seed(5)
    base = ActorCritic(**SMALL).eval()
    path = str(tmp_path / "base.pt")
    save_checkpoint(path, base, {"spec": {}})
    m, _extra, fresh = load_for_memory(path, "cpu", memory_spec(hidden_size=32), scan_channels=CHAN,
                                       motion=motion_spec(hidden_size=16, channels=8),
                                       motion_heads=list(heads))
    m.eval()
    allowed = lambda f: (".memory." in f or f.startswith("actor.opp_mask.")
                         or f.startswith("actor.dv."))
    assert fresh and all(allowed(f) for f in fresh), fresh
    assert ("mask" in heads) == any(f.startswith("actor.opp_mask.") for f in fresh)
    assert ("dv" in heads) == any(f.startswith("actor.dv.") for f in fresh)
    scan, pro, priv = inputs(batch=4, model=m)
    plain = scan[:, :SMALL["n_stack"]]
    with torch.no_grad():
        a0, lp0, _ = base.act(plain, pro, deterministic=True)
        a1, lp1, _ = m.act(scan, pro, deterministic=True, h=None)
        v0 = base.critic(plain, pro, priv)
        v1, _hc = m.critic.step(scan, pro, priv, None)
        g0, o0 = base.actor.forward_all(plain, pro)[1:]
        g1, o1 = m.actor.step_all(scan, pro, None, None)[1:3]
    assert torch.equal(a0, a1) and torch.equal(lp0, lp1)
    assert torch.equal(v0, v1)
    assert torch.equal(g0, g1) and torch.equal(o0, o1)


def test_a_second_warm_start_onto_a_trained_branch_is_refused(tmp_path):
    torch.manual_seed(5)
    base = ActorCritic(**SMALL).eval()
    p0 = str(tmp_path / "base.pt")
    save_checkpoint(p0, base, {"spec": {}})
    m, _e, _f = load_for_memory(p0, "cpu", memory_spec(hidden_size=32), scan_channels=CHAN,
                                motion=motion_spec(hidden_size=16, channels=8))
    p1 = str(tmp_path / "trained.pt")
    save_checkpoint(p1, m, {"spec": {}})
    with pytest.raises(ValueError, match="already carries a motion branch"):
        load_for_memory(p1, "cpu", None, scan_channels=CHAN,
                        motion=motion_spec(hidden_size=16, channels=8))


def test_motion_memory_refuses_to_exist_without_the_main_memory():
    with pytest.raises(ValueError, match="needs the main recurrent memory"):
        ActorCritic(**SMALL, scan_channels=CHAN, motion=motion_spec(hidden_size=16, channels=8))
    with pytest.raises(ValueError, match="no scan channels"):
        ActorCritic(**SMALL, memory=memory_spec(hidden_size=32),
                    motion=motion_spec(hidden_size=16, channels=8))
    with pytest.raises(ValueError, match="must be in"):
        motion_spec(hidden_size=128)


# ------------------------------------------------------------------ the heads
def test_the_mask_head_is_per_beam_and_partitions_the_axis():
    enc = MotionEncoder(3, channels=8)
    rows = torch.zeros(2, 3, 1081)
    feat, pooled = enc(rows)
    head = OppMaskHead(8, 1081, enc.stride, width=16)
    logit = head(feat)
    assert logit.shape == (2, 1081)
    assert enc.stride == 8 and feat.shape[2] * enc.stride >= 1081
    assert float(logit.abs().max()) == 0.0, "zero-initialised output layer"
    assert pooled.shape == (2, 2 * 8)


def test_the_mask_loss_weights_the_rare_class_and_reports_what_it_did():
    logit = torch.zeros(4, 100, requires_grad=True)
    label = torch.zeros(4, 100); label[:, 10:13] = 1.0            # 3 % positive
    w = torch.ones(4)
    loss, parts = mask_loss(logit, label, w)
    assert float(parts["mask_pos_rate"]) == pytest.approx(0.03)
    assert float(parts["mask_pos_weight"]) == pytest.approx(1 / 0.03, rel=1e-4)
    # an empty road cannot send the weight to infinity
    _l2, p2 = mask_loss(logit, torch.zeros(4, 100), w)
    assert float(p2["mask_pos_weight"]) == pytest.approx(50.0)
    # a head that answers "no car" everywhere is visible as zero recall, whatever the BCE says
    confident_no = torch.full((4, 100), -5.0)
    assert float(mask_loss(confident_no, label, w)[1]["mask_recall"]) == 0.0
    # and the weighted loss punishes it more than the neutral one
    assert float(mask_loss(confident_no, label, w)[0]) > float(loss)


def test_the_dv_loss_is_presence_masked():
    pred = torch.zeros(4, 2)
    target = torch.ones(4, 2)
    w = torch.ones(4)
    present = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss, parts = dv_loss(pred, target, present, w)
    assert float(loss) == pytest.approx(1.0), "the absent rows carry no error"
    assert float(parts["dv_labelled_frac"]) == pytest.approx(0.5)
    assert float(dv_loss(pred, target, torch.zeros(4), w)[0]) == 0.0


# ------------------------------------------------------------------ deployment
def test_the_export_carries_the_motion_branch_and_no_head(tmp_path):
    """The branch is part of the policy and must be exported; the two heads never are.

    They are never called by `forward` or `step`, so the trace cannot contain them -- this is the
    check that the "never called" stays true as the forward is edited.
    """
    onnx = pytest.importorskip("onnx")
    # 288 beams, not 256: the resnet stem's sector profile is an `adaptive_max_pool1d` to 36, and
    # the ONNX exporter refuses an output size that is not a factor of the input. That is a property
    # of the stem this branch did not touch (the real policy has 1081 beams and exports through the
    # same path in `test_future_head.py`); using a divisible count here keeps this test about the
    # heads.
    big = dict(SMALL, n_beams=288)
    spec = ObsSpec(n_beams=288, scan_stack=6, act_dim=8, action_history=2, hist_len=0)
    chan = {"channels": ["memory", "edges"] + ROWS, "aligned": {"proprio": motion_index_spec(spec)}}
    torch.manual_seed(0)
    m = ActorCritic(**big, memory=memory_spec(hidden_size=32), scan_channels=chan,
                    motion=motion_spec(hidden_size=16, channels=8),
                    motion_heads=["mask", "dv"]).eval()
    from f1sim.learn.export import RecurrentActorOnly
    mod = RecurrentActorOnly(m.actor).eval()
    g = torch.Generator().manual_seed(3)
    scan = torch.rand(1, big["n_stack"] + len(chan["channels"]), big["n_beams"], generator=g)
    pro = torch.rand(1, big["proprio_dim"], generator=g)
    h = m.actor.initial_hidden(1)
    out = str(tmp_path / "actor.onnx")
    torch.onnx.export(mod, (scan, pro, h), out, input_names=["scan", "proprio", "hidden"],
                      output_names=["action", "hidden_next"], opset_version=17, dynamo=False)
    names = {i.name for i in onnx.load(out).graph.initializer}
    assert names, "the export produced no weights at all"
    assert not any("opp_mask" in n or n.endswith(".dv") or ".dv." in n for n in names), \
        sorted(n for n in names if "opp_mask" in n or ".dv." in n)
    assert any("motion" in n for n in names), "the motion branch IS part of the policy"
    with torch.no_grad():
        mu, h1 = mod(scan, pro, h)
    assert mu.shape == (1, big["act_dim"]) and h1.shape == h.shape


# ------------------------------------------------------------------ the label
def test_the_beam_mask_label_is_the_lidar_s_own_car_hits():
    """The label is not a new quantity: it is the beam classification the LiDAR already casts.

    Driven in a real race rather than posed by hand, because what has to hold is that the mask the
    trainer reads at the instant it reads it is the car-hit mask of the observation's newest scan --
    which is a claim about `last_result`, not about `ray_box_hits`.
    """
    from f1sim import maps
    from f1sim.gym_env import EnvConfig
    from f1sim.lidar import HIT_CAR
    from f1sim.learn import common
    tracks = [maps.load("gen:competition:0")]
    env = common.make_env(tracks, 8, "cpu",
                          EnvConfig(race_size=2, opponent="teacher", scan_stack=3, scan_stride=1,
                                    speed_cap=4.0), seed=3)
    env.reset(seed=3)
    seen = 0
    for _ in range(60):
        a = torch.zeros(8, env.act_dim); a[:, 1] = -0.2       # learners crawl, the opponent is ahead
        env.step(a)
        mask = env.opponent_beam_mask()
        assert mask.shape == env.last_result.scan.shape
        assert torch.equal(mask > 0.5, env.last_result.scan_type == HIT_CAR)
        assert float(mask.max()) <= 1.0 and float(mask.min()) >= 0.0
        seen += int(mask[env.learner].sum() > 0)
    assert seen > 0, "a learner never saw an opponent, so the label was never exercised"
    # and it is the rare class the weighting exists for
    assert float(env.opponent_beam_mask().mean()) < 0.2


def test_the_beam_mask_is_all_zero_when_there_is_no_other_car():
    from f1sim import maps
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    env = common.make_env([maps.load("gen:competition:0")], 4, "cpu",
                          EnvConfig(race_size=1, scan_stack=3, speed_cap=4.0), seed=3)
    env.reset(seed=3)
    env.step(torch.zeros(4, env.act_dim))
    assert float(env.opponent_beam_mask().abs().max()) == 0.0


def test_the_probe_loads_a_motion_checkpoint_and_reads_the_whole_state(tmp_path):
    """The E1 column for an E3 arm goes through the loader, not through a hand-built model.

    `probe_hidden.load_probed` takes a checkpoint that already carries memory down the
    `load_checkpoint` path, which rebuilds it from `meta` -- so `motion` and `motion_heads` have to
    survive the round trip, and what `probe_state` then returns has to be the composite state and
    not one half of it.
    """
    from f1sim.learn.probe_hidden import load_probed, peek
    m = build()
    path = str(tmp_path / "motion.pt")
    save_checkpoint(path, m, {"spec": SPEC.__dict__})
    meta, _spec = peek(path)
    assert meta["motion"]["hidden_size"] == 16 and meta["motion_heads"] == ["mask", "dv"]
    mod, _extra, note = load_probed(path, "cpu", 128, [])
    assert note == "as trained" and mod.actor.has_motion
    scan, pro, _priv = inputs(batch=2)
    with torch.no_grad():
        _act, state, _h = mod.actor.probe_state(scan, pro, None, None)
    assert state.shape[-1] == mod.actor.memory.hidden_size + mod.actor.memory.motion.hidden_size


# ------------------------------------------------------------------ end to end
CLI_REFUSALS = [
    (["--motion-memory"], "needs --memory gru"),
    (["--memory", "gru", "--motion-memory"], "reads the aligned rows"),
    (["--memory", "gru", "--scan-channels", "aligned", "--aux-opp-mask", "1.0"],
     "no branch without --motion-memory"),
    (["--memory", "gru", "--scan-channels", "memory,edges", "--motion-memory"],
     "reads the aligned rows"),
]


@pytest.mark.parametrize("argv,message", CLI_REFUSALS)
def test_the_cli_refuses_a_motion_branch_with_nothing_to_read(monkeypatch, argv, message):
    """Each of these would produce a run that looks like the one it is named after and is not."""
    import sys
    from f1sim.learn import ppo
    monkeypatch.setattr(sys, "argv", ["ppo", "--name", "t", "--device", "cpu"] + argv)
    with pytest.raises(SystemExit) as e:
        ppo.main()
    assert message in str(e.value), e.value


@pytest.mark.parametrize("stage,extra,want", [
    ("a", ["--aux-opp-mask", "1.0"], ["loss/aux_opp_mask"]),
    ("b", ["--aux-opp-mask", "1.0", "--aux-motion", "1.0"],
     ["loss/aux_opp_mask", "loss/aux_motion_mse"]),
    ("c", ["--aux-opp-mask", "1.0", "--aux-motion", "1.0", "--aux-future", "1.0",
           # k = 2 rather than the default 20: the label for step t is the state at t + k, and a
           # four-step chunk cannot reach twenty. The trainer refuses the combination, which is its
           # own test above; here the point is to run the term, not to argue with it.
           "--aux-future-k", "2"],
     ["loss/aux_opp_mask", "loss/aux_motion_mse", "loss/aux_future_mse"]),
])
def test_a_whole_update_runs_at_every_stage(monkeypatch, tmp_path, stage, extra, want):
    """Two PPO updates through `main()` with the channel and both auxiliaries on.

    Not a result -- two updates on six cars is nothing. What it catches is the class of mistake the
    unit tests above cannot see, because it lives in the trainer's own plumbing rather than in a
    module: a buffer that is not filled, a label that is not aligned, a minibatch that is sliced
    wrong, or a local name that shadows another. One did exactly that -- the beam-mask label's flat
    view was called `f_mask`, which is the on-policy sample weight twenty lines above it, so every
    arm WITHOUT the mask lost its weights to a None. Nothing short of running the loop would have
    found it.
    """
    import sys
    from f1sim.learn import ppo
    monkeypatch.setenv("F1SIM_RUNS", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [
        "ppo", "--name", f"motion_smoke_{stage}", "--device", "cpu", "--sim-backend", "eager",
        "--tracks", "gen:competition:0", "--envs", "6", "--race-size", "3", "--opponent", "policy",
        "--action-mode", "plan", "--horizon", "4", "--minibatch", "12", "--epochs", "1",
        "--total", "48", "--critic-warmup", "0", "--episode-s", "5.0",
        "--scan-stack", "6", "--hist-len", "20", "--wandb", "disabled", "--save-every", "1000",
        "--memory", "gru", "--memory-hidden", "16",
        "--scan-channels", "memory,edges,aligned,aligned_prev,aligned_valid",
        "--motion-memory", "--motion-hidden", "8", "--motion-channels", "8",
        "--aux-grip", "1.0", "--aux-opp", "1.0",
        "--metrics-jsonl", str(tmp_path / "m.jsonl"),
    ] + extra)
    ppo.main()
    import json
    rows = [json.loads(l) for l in open(tmp_path / "m.jsonl")]
    assert rows, "the run logged nothing"
    last = rows[-1]
    for key in want:
        assert key in last and np.isfinite(last[key]), (key, last.get(key))
    # a stage that is off leaves no trace: a coefficient of zero is not a term multiplied by zero
    absent = {"loss/aux_opp_mask", "loss/aux_motion_mse", "loss/aux_future_mse"} - set(want)
    assert not (absent & set(last)), sorted(absent & set(last))
    # the mask head is scored on a real label, so its own diagnostics have to be there and sane
    assert 0.0 <= last["loss/aux_opp_mask/mask_pos_rate"] <= 1.0
    assert 0.0 <= last["loss/aux_opp_mask/mask_recall"] <= 1.0
