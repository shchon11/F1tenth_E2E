"""The v2 actor-critic (`learn.model_v2`) and its train-time heads (`learn.aux_v2`).

What is pinned, block by block (docs/research/architecture-redesign-2026-10-13.md, section 4):
the parameter count of every block against the redesign's table; B5's LayerNorm + chrono gate and
the concatenated memory output; the dial as a required input column; B6's per-dimension log std
and small last layer; the critic's own one-frame stem and no state; the train-time heads built
last and never on the forward path; the B8 set loss (critique A6: 3 slots, 6 permutations, the
nearest three when there are more cars) and its knowability mask; the stem seeded from a v1
checkpoint; the round trip through the one loader every consumer uses.

And the property the whole memory redesign rests on: the redesign's bisection found a GRU learns
to carry a cue only under a dense target that cannot be met without memory (all three GRU variants
learned an 8-step recall with one, none without -- proposal-temporal/aux_*.json). The last test
trains the v2 actor on that task, from scratch, on the CPU.
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.learn import aux_v2  # noqa: E402
from f1sim.learn.memory import Hidden, policy_fn, runtime_for  # noqa: E402
from f1sim.learn.model_v2 import (ARCH, CHRONO_T_MAX, CHRONO_T_MIN, ActorCriticV2,  # noqa: E402
                                  block_parameter_counts, init_stems_from_v1)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAFEST = os.path.join(ROOT, "checkpoints", "iccas_bootcamp_safest.pt")
SMALL = 271          # beams for the tests that train or step a lot; the network is beam-count generic


def _model(n_beams=SMALL, n_frames=1, aux=None, seed=0):
    torch.manual_seed(seed)
    return ActorCriticV2(n_beams=n_beams, n_frames=n_frames, aux=aux).eval()


def _inputs(B=3, n_frames=1, n_beams=SMALL, seed=1):
    g = torch.Generator().manual_seed(seed)
    scan = torch.rand(B, n_frames, n_beams, generator=g)
    ego = 0.3 * torch.randn(B, 210, generator=g)
    c = torch.linspace(-1.0, 0.4, B)[:, None]
    return scan, ego, c


# ---------------------------------------------------------------- blocks and parameter counts
def test_parameter_counts_per_block_match_the_redesign_table():
    """Section 4's table and the critique's recount (A5): stem 794 k ("79.4만"), deployed actor
    1,159,978 at the redesign's 147-wide ego and 5 outputs. Contract v2's ego is 210 wide (no
    duplicate columns, 8 WP4 actions for a_{t-2} and per history row) and the head has 8 outputs,
    which is the whole difference."""
    c = block_parameter_counts(_model(1081))
    assert c["B1+B2 stem"] == 793_632
    assert c["B3 ego"] == 210 * 128 + 128 == 27_008           # the table's 147 x 128 + 128 = 18,944
    assert c["B4 fusion"] == (256 + 128 + 1) * 256 + 256 == 98_816
    assert c["B5 LayerNorm"] == 2 * 256
    assert c["B5 GRU"] == 3 * (256 * 128 + 128 * 128 + 2 * 128) == 148_224
    assert c["B6 head"] == (384 * 256 + 256) + (256 * 8 + 8) == 100_616
    assert c["B6 log_std"] == 8
    redesign_total = 1_159_978
    assert c["actor (deployed)"] == redesign_total + (210 - 147) * 128 + (8 - 5) * 257 + (8 - 5) == 1_168_816
    # the critic reads 33 privileged columns (`gym_env.PRIV_V2_DIM`; 34 with the speed cap until 2026-10-14)
    assert c["B9 critic"] == 793_632 + (210 + 1 + 33) * 128 + 128 + 384 * 256 + 256 + 256 * 256 + 256 + 2 * 257
    assert c["B8 slots"] == c["beam car head"] == c["nearest car head"] == 0, "heads only when flagged"
    full = block_parameter_counts(_model(1081, aux={"b8": True, "beam_car": True, "near_car": True}))
    assert full["B8 slots"] == 384 * 256 + 256 + 256 * 12 + 12
    assert full["nearest car head"] == 384 * 256 + 256 + 256 * 5 + 5
    assert full["beam car head"] == 48 * 64 * 3 + 64 + 64 * 32 + 32
    assert full["actor (deployed)"] == c["actor (deployed)"]
    # one more frame is one more input row of the first convolution: 48 x 7 weights
    assert block_parameter_counts(_model(1081, 3))["B1+B2 stem"] == 793_632 + 2 * 48 * 7


@pytest.mark.parametrize("n_frames", [1, 2, 3])
def test_shapes_and_ranges(n_frames):
    m = _model(n_frames=n_frames)
    scan, ego, c = _inputs(n_frames=n_frames)
    with torch.no_grad():
        mu, h = m.actor.step(scan, ego, c)
        v, logit = m.critic(scan, ego, c, torch.randn(3, 33))
    assert mu.shape == (3, 8) and h.shape == (1, 3, 128) and float(mu.abs().max()) <= 1.0
    assert v.shape == logit.shape == (3,)
    with pytest.raises(ValueError):
        m.actor.step(torch.rand(3, n_frames + 1, SMALL), ego, c)


def test_the_dial_is_a_required_input_column_that_moves_the_action():
    a = _model().actor
    scan, ego, c = _inputs()
    with pytest.raises(ValueError):
        a.step(scan, ego, None)
    with pytest.raises(ValueError):
        a.step(scan, ego, torch.zeros(3, 2))
    with torch.no_grad():
        a0, _ = a.step(scan, ego, torch.zeros(3, 1))
        a1, _ = a.step(scan, ego, torch.ones(3, 1))
    assert float((a0 - a1).abs().max()) > 0, "an ordinary input column, live from the first forward"


def test_b5_is_a_layernormed_chrono_gru_whose_output_is_concatenated():
    a = _model().actor
    H = a.hidden_size
    z_bias = (a.gru.bias_ih_l0[H:2 * H] + a.gru.bias_hh_l0[H:2 * H]).detach()
    assert float(a.gru.bias_hh_l0[H:2 * H].detach().abs().max()) == 0.0
    T = 1.0 + z_bias.exp()                                   # keep 1 - 1/T  <=>  bias log(T - 1)
    assert float(T.min()) >= CHRONO_T_MIN - 1e-3 and float(T.max()) <= CHRONO_T_MAX + 1e-2
    mid = 0.5 * (math.log(CHRONO_T_MIN) + math.log(CHRONO_T_MAX))
    assert abs(float(T.log().median()) - mid) < 0.5, "timescales should be log-uniform"
    assert isinstance(a.ln, torch.nn.LayerNorm) and a.head[0].in_features == 256 + 128
    # concatenated, not a zero-initialised additive term: the state reaches the action at once
    scan, ego, c = _inputs()
    with torch.no_grad():
        m0, _ = a.step(scan, ego, c, torch.zeros(1, 3, H))
        m1, _ = a.step(scan, ego, c, torch.randn(1, 3, H))
    assert float((m0 - m1).abs().max()) > 1e-4


def test_b6_log_std_per_dimension_and_a_small_last_layer():
    a = _model().actor
    assert a.log_std.detach().tolist() == pytest.approx([-2.07, -2.07] + [-1.8] * 6)
    assert float(a.mu.bias.abs().max()) == 0.0
    assert float(a.mu.weight.abs().max()) <= 0.1 / math.sqrt(256) + 1e-7     # default init x 0.1


def test_the_critic_has_its_own_one_frame_stem_and_no_state():
    m = _model(n_frames=3)
    cr = m.critic
    assert cr.stem is not m.actor.stem and cr.stem.n_frames == 1
    assert not any(isinstance(x, torch.nn.RNNBase) for x in cr.modules())
    scan, ego, c = _inputs(n_frames=3)
    priv = torch.randn(3, 33)
    with torch.no_grad():
        v0, l0 = cr(scan, ego, c, priv)
        older = scan.clone(); older[:, 1:] = torch.rand_like(older[:, 1:])
        v1, l1 = cr(older, ego, c, priv)
        v2, _ = cr(scan, ego, c, priv + 1.0)
    assert torch.equal(v0, v1) and torch.equal(l0, l1), "the critic reads frame 0 only"
    assert float((v0 - v2).abs().max()) > 0
    with pytest.raises(ValueError):
        cr(scan, ego, c, torch.randn(3, 22))                  # a v1 privileged vector is not it


# ---------------------------------------------------------------- the sequence hook
def test_sequence_equals_stepping_through_resets_and_burn_in():
    a = _model(n_frames=2).actor
    T, B = 7, 3
    g = torch.Generator().manual_seed(3)
    scan = torch.rand(T, B, 2, SMALL, generator=g)
    ego = torch.randn(T, B, 210, generator=g)
    c = torch.randn(T, B, 1, generator=g)
    keep = torch.ones(T, B); keep[4, 1] = 0.0                  # row 1 starts a new episode at t = 4
    with torch.no_grad():
        h, mus, hs = None, [], []
        for t in range(T):
            if h is not None:
                h = h * keep[t][None, :, None]
            mu, h = a.step(scan[t], ego[t], c[t], h)
            mus.append(mu); hs.append(h)
        seq = a.sequence(scan, ego, c, keep=keep)
        burn = a.sequence(scan, ego, c, keep=keep, burn_in=3)
        plain = a.sequence(scan, ego, c)
    assert torch.allclose(seq["mu"], torch.stack(mus), atol=1e-5)
    assert torch.allclose(seq["h"], hs[-1], atol=1e-5)
    assert burn["mu"].shape[0] == T - 3 and torch.allclose(burn["mu"], seq["mu"][3:], atol=1e-5)
    assert not torch.allclose(plain["mu"][4:, 1], seq["mu"][4:, 1], atol=1e-4)


# ---------------------------------------------------------------- train-time heads
def test_heads_are_built_last_only_when_flagged_and_never_touch_the_forward():
    plain = _model(seed=5)
    full = _model(seed=5, aux={"b8": True, "beam_car": True, "near_car": True})
    sp, sf = plain.state_dict(), full.state_dict()
    heads = [k for k in sf if k.startswith(("actor.b8.", "actor.beam_car.", "actor.near_car."))]
    assert heads and set(sf) - set(sp) == set(heads)
    assert all(torch.equal(sp[k], sf[k]) for k in sp), "a head moved a weight it does not own"
    scan, ego, c = _inputs()
    with torch.no_grad():
        assert torch.equal(plain.actor.step(scan, ego, c)[0], full.actor.step(scan, ego, c)[0])
        out = full.actor.step_all(scan, ego, c, beams=True)
    assert out["b8"].shape == (3, 3, 4) and out["near_car"].shape == (3, 5)
    assert out["beam_car"].shape == (3, SMALL) and out["z"].shape == (3, 384)
    assert full.meta["aux"] == {"b8": True, "beam_car": True, "near_car": True}


def test_b8_loss_is_a_set_loss_over_the_six_slot_permutations():
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(5, 3, 4, generator=g)
    lab = 0.3 * torch.randn(5, 3, 3, generator=g)
    cand = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1], [1, 0, 1]], dtype=torch.bool)
    know = cand.clone(); know[4, 2] = False                    # row 4: one car there, unknowable
    base = aux_v2.b8_loss(pred, lab, cand, know)
    for perm in ([1, 2, 0], [2, 0, 1], [0, 2, 1]):
        assert torch.allclose(aux_v2.b8_loss(pred[:, perm], lab, cand, know), base, atol=1e-6)
        assert torch.allclose(aux_v2.b8_loss(pred, lab[:, perm], cand[:, perm], know[:, perm]), base, atol=1e-6)
    # a perfect, confident answer costs (almost) nothing ...
    perfect = torch.zeros(5, 3, 4)
    perfect[..., 0] = torch.where(cand, 20.0, -20.0)
    perfect[..., 1:3] = lab[..., :2]; perfect[..., 3] = lab[..., 2]
    assert float(aux_v2.b8_loss(perfect, lab, cand, cand).max()) < 1e-6
    # ... an unknowable car is free whatever its slot says, an absent one is "presence 0"
    p = perfect.clone(); p[4, 2] = torch.tensor([-20.0, 9.0, 9.0, 9.0])
    assert float(aux_v2.b8_loss(p, lab, cand, know)[4]) < 1e-6
    assert float(aux_v2.b8_loss(p, lab, cand, cand)[4]) > 1.0
    q = perfect.clone(); q[2, 0, 0] = 20.0
    assert float(aux_v2.b8_loss(q, lab, cand, cand)[2]) == pytest.approx(20.0 / 3, rel=1e-3)


def test_b8_select_keeps_the_three_nearest_candidates_and_pads_small_races():
    lab = torch.arange(5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    cand = torch.tensor([[True, True, False, True, True]])
    dist = torch.tensor([[4.0, 1.0, 0.5, 5.5, 2.0]])           # seat 2 is closest but not a candidate
    l3, c3, k3 = aux_v2.b8_select(lab, cand, cand, dist)
    assert l3.shape == (1, 3, 3) and c3.all() and k3.all()
    assert l3[0, :, 0].tolist() == [3.0, 12.0, 0.0]          # seats 1, 4, 0: nearest first, 3 kept
    l1, c1, k1 = aux_v2.b8_select(lab[:, :1], cand[:, :1], cand[:, :1], dist[:, :1])
    assert l1.shape == (1, 3, 3) and c1.tolist() == [[True, False, False]]


def test_b8_know_mask_reads_age_and_the_window():
    cand = torch.tensor([[True, True, True, False]])
    age = torch.tensor([[0, 30, 121, 0]])
    assert aux_v2.b8_know_mask(cand, age, torch.tensor([40])).tolist() == [[True, True, False, False]]
    # 30 steps since the sighting but only 10 into the window: the state never saw it
    assert aux_v2.b8_know_mask(cand, age, torch.tensor([10])).tolist() == [[True, False, False, False]]


def test_the_beam_and_nearest_car_losses():
    logits = torch.tensor([[5.0, -5.0, 0.0, 9.0]])
    labels = torch.tensor([[1, 0, -1, -1]], dtype=torch.int8)
    assert float(aux_v2.beam_car_loss(logits, labels)) < 0.01  # the -1 beams are not scored
    pred = torch.tensor([[3.0, 0.2, 0.1, 0.0, 0.0], [3.0, 9.0, 9.0, 9.0, 9.0]])
    none = torch.zeros(2, 5)
    l = aux_v2.nearest_car_loss(pred, none)
    assert float(l[0]) == pytest.approx(float(l[1])), "no car: only the presence term counts"


# ---------------------------------------------------------------- seeded from v1, saved, loaded
@pytest.mark.skipif(not os.path.isfile(SAFEST), reason="the archived checkpoint is not present")
@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_v1_stem_seeds_the_v2_stem_and_a_static_scene_reads_the_same_first_layer(k):
    import torch.nn.functional as F
    from f1sim.learn.model import load_checkpoint
    m = _model(1081, k)
    rep = init_stems_from_v1(m, SAFEST)
    assert len(rep["actor"]) == len(rep["critic"]) == len(m.actor.stem.state_dict())
    v1, _ = load_checkpoint(SAFEST, "cpu")
    v1s = v1.actor.stem
    frame = torch.rand(2, 1, 1081)
    near = -F.max_pool1d(-frame, 9, stride=1, padding=4)
    rows_v1 = torch.cat([frame.expand(-1, 6, -1), torch.zeros(2, 1, 1081), near], 1)   # ramp switched off
    rows_v2 = torch.cat([frame.expand(-1, k, -1), near], 1)
    with torch.no_grad():
        assert torch.allclose(v1s.trunk[0](rows_v1), m.actor.stem.trunk[0](rows_v2), atol=1e-5)
    sd1, sd2 = v1s.state_dict(), m.actor.stem.state_dict()
    for key in sd2:
        if key == "fc.0.weight":
            assert torch.equal(sd2[key], sd1[key][:, :48 * 34])
        elif key != "trunk.0.weight":
            assert torch.equal(sd2[key], sd1[key]), key
    assert torch.equal(m.critic.stem.trunk[3].conv1.weight, v1.critic.stem.trunk[3].conv1.weight)


def test_save_and_load_round_trip_through_the_one_loader(tmp_path):
    from f1sim.learn.model import load_checkpoint, load_for_conditioning, load_for_memory, save_checkpoint
    m = _model(aux={"b8": True})
    p = str(tmp_path / "v2.pt")
    save_checkpoint(p, m, {"spec": vars(m.obs_spec)})
    m2, extra = load_checkpoint(p)
    assert isinstance(m2, ActorCriticV2) and m2.meta == m.meta and extra["skipped"] == []
    assert m2.meta["arch"] == ARCH and m2.meta["obs"]["version"] == 2
    scan, ego, c = _inputs()
    with torch.no_grad():
        assert torch.equal(m.actor.step(scan, ego, c)[0], m2.eval().actor.step(scan, ego, c)[0])
    with pytest.raises(ValueError):
        load_for_memory(p, "cpu", {"kind": "gru"})
    with pytest.raises(ValueError):
        load_for_conditioning(p, "cpu", 1, {"dim": 1, "kind": "current_mu", "source": "dial"})


def test_every_policy_runner_carries_a_v2_actors_state():
    m = _model()
    rt = runtime_for(m, batch=3)
    assert rt.memory and rt.hidden == Hidden()
    scan, ego, c = _inputs()
    run = policy_fn(m, batch=3, cond=c)
    a1 = run({"scan": scan, "ego": ego})
    h1 = run.runtime.hidden.actor.clone()
    a2 = run({"scan": scan, "ego": ego})
    assert a1.shape == (3, 8) and not torch.equal(h1, run.runtime.hidden.actor)
    assert not torch.equal(a1, a2), "the second step sees the first step's state"
    _a, logp, hid = m.act(scan, ego, deterministic=True, c=c, h=None)
    assert isinstance(hid, Hidden) and hid.critic is None and logp.shape == (3,)


# ---------------------------------------------------------------- the memory property (slow)
def _corridor(T, B, n, g):
    """(T, B, 1, n) scans of a corridor whose walls and end wander a little every step."""
    ang = torch.linspace(-0.75 * math.pi, 0.75 * math.pi, n)
    s, co = torch.sin(ang)[None], torch.cos(ang)[None]
    wl, wr = 0.8 + 0.7 * torch.rand(B, generator=g), 0.8 + 0.7 * torch.rand(B, generator=g)
    d = 4.0 + 5.0 * torch.rand(B, generator=g)
    far = torch.full((B, n), 1e9)
    out = []
    for _t in range(T):
        wl = (wl + 0.05 * torch.randn(B, generator=g)).clamp(0.6, 2.0)
        wr = (wr + 0.05 * torch.randn(B, generator=g)).clamp(0.6, 2.0)
        d = (d + 0.3 * torch.randn(B, generator=g)).clamp(2.0, 9.5)
        r = torch.minimum(torch.where(s > 1e-3, wl[:, None] / s.clamp_min(1e-3), far),
                          torch.where(s < -1e-3, wr[:, None] / (-s).clamp_min(1e-3), far))
        r = torch.minimum(r, torch.where(co > 1e-3, d[:, None] / co.clamp_min(1e-3), far)).clamp(max=10.0)
        out.append(((r + 0.02 * torch.randn(B, n, generator=g)) / 10.0).clamp(0, 1))
    return torch.stack(out)[:, :, None, :]


@pytest.mark.slow
def test_the_v2_memory_learns_an_8_step_recall_with_a_dense_auxiliary():
    """proposal-temporal's toy on the v2 actor, from scratch: a cue (beams 0..40-equivalent at
    0.5 m) only in the frame of step 0, the action's first dimension must say the cue's sign at step
    8, and a dense auxiliary -- a head on `[e | y]` asked for the cue at EVERY step -- is the only
    other signal. Without memory the best possible loss is 1.0 (the cue is +-1 at random and the
    frames after step 0 carry nothing of it). The run stops as soon as it has learned.

    Whether it learns depends on the draw (initial weights and data). Measured on this CPU, 2
    threads, iterations to a 20-step mean < 0.1 within 700, the actor built alone and seeded as here:
    the 210-wide contract ego 410 / 364 / 258 for seeds 0 / 1 / 2; the 202-wide ego of the commit
    before a_{t-2} had columns of its own: never / 217 / never (the earlier version of this test,
    seed 0 with the critic built in between, learned at 205 there and not within 450 at 210 wide).
    The property is "it CAN learn with the dense target, and is flat at chance without it" (the
    module docstring of `aux_v2`); this pins seed 2 and the first-learned iteration is not a
    statistic. G1 is the gate that measures it on real data."""
    n, K, B, max_iters = SMALL, 8, 32, 450
    seed = 2
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        from f1sim.learn.model_v2 import ActorV2
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        actor = ActorV2(1, n)
        aux_head = torch.nn.Linear(actor.feature_dim, 1)
        params = list(actor.parameters()) + list(aux_head.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        W = round(n * 40 / 1081)

        def batch(b):
            s = _corridor(K + 1, b, n, g)
            cue = torch.randint(0, 2, (b,), generator=g).float() * 2 - 1
            s[0, :, 0, :W] = torch.where(cue[:, None] > 0, torch.full((b, W), 0.05), s[0, :, 0, :W])
            return s, 0.05 * torch.randn(K + 1, b, 210, generator=g), torch.zeros(K + 1, b, 1), cue

        hist, learned = [], None
        for it in range(max_iters):
            s, ego, c, cue = batch(B)
            o = actor.sequence(s, ego, c)
            act = ((o["mu"][K, :, 0] - cue) ** 2).mean()
            aux = ((torch.tanh(aux_head(o["z"]))[..., 0] - cue[None]) ** 2).mean()
            opt.zero_grad(); (act + aux).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            hist.append(float(act))
            if len(hist) >= 20 and sum(hist[-20:]) / 20 < 0.1:
                learned = it + 1
                break
        assert learned is not None, f"no recall after {max_iters} iterations (last 20: {sum(hist[-20:]) / 20:.3f})"
        with torch.no_grad():
            s, ego, c, cue = batch(256)
            with_mem = float(((actor.sequence(s, ego, c)["mu"][K, :, 0] - cue) ** 2).mean())
            cut = float(((actor.sequence(s, ego, c, keep=torch.zeros(K + 1, 256))["mu"][K, :, 0] - cue) ** 2).mean())
        assert with_mem < 0.1 and cut > 0.5, (with_mem, cut)
    finally:
        torch.set_num_threads(threads)


def test_the_console_takes_a_v2_checkpoint_in_a_v2_session_and_names_a_contract_switch():
    """The console drives v2 since UNIT 5a (`tests/test_v2_console.py` drives it): its compatibility
    check accepts a v2 checkpoint in a session built on contract v2, and a checkpoint of the other
    contract -- a hot reload that switched networks -- is one sentence that says so."""
    from f1sim.learn.obs import ObsSpec, ObsSpecV2
    from f1sim.viewer.sim_worker import obs_spec_problems
    m = _model()
    assert obs_spec_problems(m.meta, ObsSpecV2(n_beams=SMALL), [], 8) == []
    assert obs_spec_problems(m.meta, ObsSpec(n_beams=SMALL), [], 8) == ["관측 계약: 체크포인트 v2 vs 이 세션 v1"]
    v1_meta = {"n_beams": SMALL, "n_stack": 3, "act_dim": 8}
    assert obs_spec_problems(v1_meta, ObsSpec(n_beams=SMALL, scan_stack=3, act_dim=8), [], 8) == []
