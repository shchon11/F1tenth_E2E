"""The auxiliary future head: the label, the alignment, the masks, the parity and the probe.

Six claims, one test each, because each of them is a way the head could be wrong while training
perfectly happily:

* the label is the geometry it says it is (a placed opponent, checked by hand);
* the label for step t is the state at t + k, exactly, for a constant-velocity opponent;
* it is dropped across an episode boundary and at the end of a chunk, and never approximated;
* the head changes nothing at init and nothing at all when the term is off;
* the probe reports R^2 ~ 1 for a target the state determines and ~ 0 for one it does not;
* the deployment export does not carry the head.
"""
from __future__ import annotations

import copy
import json
import os
import types

import numpy as np
import pytest
import torch

from f1sim import Config
from f1sim.gym_env import (EnvConfig, F1VecEnv, FUTURE_LABEL_DIM, FUTURE_LABEL_KEYS,
                           FUTURE_PRESENT_INDEX)
from f1sim.learn import probe_hidden
from f1sim.learn.future import (FUTURE_K, FUTURE_OPPONENT_KEYS, align_future_targets,
                                future_labelled_fraction, future_loss, future_spec)
from f1sim.learn.memory import memory_spec
from f1sim.learn.model import ActorCritic, load_for_memory, save_checkpoint
from f1sim.learn.ppo import PPOHyper, minibatch_losses
from f1sim.track import Track

CTRL_DT = 1.0 / Config().sim.control_rate           # 0.025 s -- what k counts in


def race_env(race_size: int = 2, envs: int = 2) -> F1VecEnv:
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 24; cfg.rand.enabled = False
    return F1VecEnv(Track.generate_random(3), cfg,
                    EnvConfig(race_size=race_size, opponent="policy", scan_stack=2, hist_len=0),
                    num_envs=envs, device="cpu")


def placed(env: F1VecEnv, rows) -> torch.Tensor:
    """A ground-truth state tensor with (x, y, yaw, vx, vy, yaw_rate) written per car."""
    st = torch.zeros(env.B, env.sim.state.shape[1])
    for i, row in enumerate(rows):
        st[i, :len(row)] = torch.tensor(row, dtype=st.dtype)
    return st


# ------------------------------------------------------------------ the label
def test_future_labels_are_the_geometry_they_claim():
    """Ego facing +90 deg with a car ahead-left of it: every column checked by hand."""
    env = race_env(race_size=2, envs=2)
    e = env.ecfg
    # car 0: at the origin, heading +90 deg, driving at 4 m/s, yawing at 0.5 rad/s
    # car 1: 3 m north and 1 m east of it, heading +90 deg, driving at 6 m/s
    st = placed(env, [(0.0, 0.0, np.pi / 2, 4.0, 0.0, 0.5),
                      (1.0, 3.0, np.pi / 2, 6.0, 0.0, 0.0)])
    lab = env.future_labels(types.SimpleNamespace(state=st))
    assert lab.shape == (2, FUTURE_LABEL_DIM)
    s_ = 5.0                                                    # PRIV_OPP_DIST_SCALE
    # In car 0's frame (+x is its heading, i.e. world north) the other car is 3 m ahead, 1 m to the
    # RIGHT: rotating (1, 3) by -90 deg gives (3, -1).
    assert float(lab[0, 0]) == pytest.approx(3.0 / s_)
    assert float(lab[0, 1]) == pytest.approx(-1.0 / s_)
    # Both cars head the same way, so the relative velocity is 2 m/s straight ahead and 0 across.
    assert float(lab[0, 2]) == pytest.approx(2.0 / s_)
    assert float(lab[0, 3]) == pytest.approx(0.0, abs=1e-6)
    assert float(lab[0, FUTURE_PRESENT_INDEX]) == 1.0
    assert float(lab[0, 4]) == pytest.approx(4.0 / e.v_max_policy)
    assert float(lab[0, 5]) == pytest.approx(0.5 / e.imu_gyro_scale)
    # ... and the mirror image from car 1's seat: the other car is 3 m BEHIND and 1 m to its left.
    assert float(lab[1, 0]) == pytest.approx(-3.0 / s_)
    assert float(lab[1, 1]) == pytest.approx(1.0 / s_)
    assert float(lab[1, 2]) == pytest.approx(-2.0 / s_)


def test_a_car_beyond_overtake_range_is_absent_and_zeroed():
    """Presence is the column that says whether the other four mean anything."""
    env = race_env(race_size=2, envs=2)
    far = env.ecfg.overtake_range + 5.0
    st = placed(env, [(0.0, 0.0, 0.0, 3.0, 0.0, 0.0), (far, 0.0, 0.0, 3.0, 0.0, 0.0)])
    lab = env.future_labels(types.SimpleNamespace(state=st))
    assert float(lab[0, FUTURE_PRESENT_INDEX]) == 0.0
    assert float(lab[0, :4].abs().max()) == 0.0, "an absent car leaves no relative position behind"
    assert float(lab[0, 4]) == pytest.approx(3.0 / env.ecfg.v_max_policy)   # the ego columns stay


def test_solo_has_no_opponent_columns_but_keeps_the_ego_ones():
    env = race_env(race_size=1, envs=2)
    st = placed(env, [(0.0, 0.0, 0.0, 5.0, 0.0, -1.0), (9.0, 0.0, 0.0, 1.0, 0.0, 0.0)])
    lab = env.future_labels(types.SimpleNamespace(state=st))
    assert float(lab[:, :4].abs().max()) == 0.0
    assert float(lab[:, FUTURE_PRESENT_INDEX].abs().max()) == 0.0
    assert float(lab[0, 4]) == pytest.approx(5.0 / env.ecfg.v_max_policy)
    assert float(lab[0, 5]) == pytest.approx(-1.0 / env.ecfg.imu_gyro_scale)


def test_race_boundary_is_shared_by_every_car_of_a_race():
    env = race_env(race_size=2, envs=4)
    done = torch.tensor([False, True, False, False])
    assert env.race_boundary(done).tolist() == [True, True, False, False]
    assert env.race_boundary(torch.zeros(4, dtype=torch.bool)).tolist() == [False] * 4


# ------------------------------------------------------------------ alignment and masks
def constant_velocity_labels(T: int, k: int, B: int = 1, closing: float = 2.0, lon0: float = 6.0):
    """Labels for a car approaching head-on at `closing` m/s, sampled every control step."""
    t = torch.arange(T + 1, dtype=torch.float32)[:, None].expand(T + 1, B)
    lab = torch.zeros(T + 1, B, FUTURE_LABEL_DIM)
    lab[:, :, 0] = (lon0 - closing * t * CTRL_DT) / 5.0
    lab[:, :, 2] = -closing / 5.0
    lab[:, :, FUTURE_PRESENT_INDEX] = 1.0
    return lab


def test_alignment_gives_the_exact_t_plus_k_state():
    """A constant-velocity opponent: the label at step t is where it will be k steps later, and the
    closed form says exactly where that is."""
    T, k, closing, lon0 = 40, FUTURE_K, 2.0, 6.0
    lab = constant_velocity_labels(T, k, closing=closing, lon0=lon0)
    target, valid = align_future_targets(lab, torch.zeros(T, 1), k)
    for t in range(T - k + 1):
        assert float(valid[t, 0]) == 1.0
        assert torch.equal(target[t, 0], lab[t + k, 0])
        want = (lon0 - closing * (t + k) * CTRL_DT) / 5.0
        assert float(target[t, 0, 0]) == pytest.approx(want, abs=1e-6)
    # and the prediction the head is being asked for really is half a second of motion away
    assert float(target[0, 0, 0] - lab[0, 0, 0]) == pytest.approx(-closing * k * CTRL_DT / 5.0, abs=1e-6)


def test_k_zero_is_the_present():
    T = 8
    lab = constant_velocity_labels(T, 0, B=2)
    target, valid = align_future_targets(lab, torch.zeros(T, 2), 0)
    assert float(valid.min()) == 1.0
    assert torch.equal(target, lab[:T])


def test_the_last_k_steps_of_a_chunk_are_unlabelled():
    """No label is invented for a step whose t + k is in the next rollout: it is dropped."""
    T, k = 32, FUTURE_K
    lab = constant_velocity_labels(T, k, B=3)
    _target, valid = align_future_targets(lab, torch.zeros(T, 3), k)
    # t = T - k is the LAST labelled step, not the first unlabelled one: its label is the row for
    # the state the chunk ends in, which the rollout records alongside the value bootstrap.
    assert float(valid[:T - k + 1].min()) == 1.0
    assert float(valid[T - k + 1:].abs().max()) == 0.0
    assert int(valid.sum()) == 3 * (T - k + 1)
    assert future_labelled_fraction(T, k) == pytest.approx((T - k + 1) / T)
    assert future_labelled_fraction(16, 20) == 0.0                # k past the horizon: nothing


def test_a_boundary_inside_the_window_drops_the_label():
    T, k, B = 12, 4, 1
    lab = constant_velocity_labels(T, k, B=B)
    boundary = torch.zeros(T, B); boundary[6, 0] = 1.0            # the episode ended on step 6
    _target, valid = align_future_targets(lab, boundary, k)
    # steps 3..6 read a label from step 7..10, across the reset; 0..2 and 7.. do not
    assert [int(v) for v in valid[:T - k + 1, 0]] == [1, 1, 1, 0, 0, 0, 0, 1, 1]
    assert float(valid[T - k + 1:].abs().max()) == 0.0            # and the chunk end is still dropped


def test_alignment_refuses_shapes_that_do_not_line_up():
    with pytest.raises(ValueError):
        align_future_targets(torch.zeros(8, 2, FUTURE_LABEL_DIM), torch.zeros(8, 2), 2)   # T+1 != 8
    with pytest.raises(ValueError):
        align_future_targets(torch.zeros(9, 2, FUTURE_LABEL_DIM), torch.zeros(8, 3), 2)


# ------------------------------------------------------------------ the loss
def test_the_loss_masks_the_opponent_columns_and_never_the_presence_one():
    torch.manual_seed(0)
    n = 64
    pred = torch.zeros(n, FUTURE_LABEL_DIM)
    target = torch.randn(n, FUTURE_LABEL_DIM)
    target[:, FUTURE_PRESENT_INDEX] = (torch.arange(n) % 2).float()   # half the rows have a car
    valid = torch.ones(n); w = torch.ones(n)
    _total, parts = future_loss(pred, target, valid, w)
    for key in FUTURE_OPPONENT_KEYS:
        i = FUTURE_LABEL_KEYS.index(key)
        want = float((target[1::2, i] ** 2).mean())               # present rows only
        assert float(parts[key]) == pytest.approx(want, rel=1e-5)
    i = FUTURE_LABEL_KEYS.index("ego_speed")
    assert float(parts["ego_speed"]) == pytest.approx(float((target[:, i] ** 2).mean()), rel=1e-5)
    assert float(parts["opp_present_bce"]) == pytest.approx(np.log(2.0), rel=1e-5)
    assert float(parts["present_frac"]) == pytest.approx(0.5)
    # a 50/50 base rate is the hardest presence label there is, so the floor IS log 2 here; in a
    # tight field it is near zero and a falling BCE means much less than it looks
    assert float(parts["opp_present_bce_base"]) == pytest.approx(np.log(2.0), rel=1e-5)
    # the variance is taken under the same weights as the error, so `1 - mse / var` is meaningful
    for key in FUTURE_OPPONENT_KEYS:
        i = FUTURE_LABEL_KEYS.index(key)
        want = float(target[1::2, i].var(unbiased=False))
        assert float(parts[key + "_var"]) == pytest.approx(want, rel=1e-4)
    # a row the alignment dropped, or a teacher-driven car's row, contributes nothing at all
    total_all, _ = future_loss(pred, target, valid, w)
    total_half, parts_half = future_loss(pred, target, valid * 0.0, w)
    assert float(total_half) == 0.0 and float(parts_half["labelled_frac"]) == 0.0
    assert float(total_all) > 0.0


def test_a_perfect_prediction_costs_only_the_presence_entropy():
    n = 32
    target = torch.randn(n, FUTURE_LABEL_DIM)
    target[:, FUTURE_PRESENT_INDEX] = 1.0
    pred = target.clone()
    pred[:, FUTURE_PRESENT_INDEX] = 20.0                          # a confident "yes"
    total, parts = future_loss(pred, target, torch.ones(n), torch.ones(n))
    assert float(total) < 1e-6
    assert all(float(parts[k]) < 1e-9 for k in FUTURE_LABEL_KEYS if k != "opp_present")


# ------------------------------------------------------------------ parity
SMALL = dict(n_stack=3, n_beams=64, proprio_dim=16, priv_dim=21, act_dim=4, scan_stem="plain")


def test_the_head_is_absent_unless_asked_for_and_costs_no_rng():
    torch.manual_seed(3); plain = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32))
    torch.manual_seed(3); withf = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32),
                                              future_head=future_spec())
    assert "future_head" not in plain.meta and not plain.has_future
    assert withf.meta["future_head"]["k"] == FUTURE_K and withf.has_future
    assert not any(k.startswith("actor.future") for k in plain.state_dict())
    for key, value in plain.state_dict().items():
        assert torch.equal(value, withf.state_dict()[key]), f"{key} moved when the head was added"


def test_the_head_reads_the_recurrent_state_and_starts_at_zero():
    torch.manual_seed(4)
    m = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32), future_head=future_spec(width=16))
    assert m.actor.future.net[0].in_features == 32, "with memory the head reads the GRU state"
    scan, pro = torch.rand(5, 3, 64), torch.rand(5, 16)
    with torch.no_grad():
        fut = m.actor.step_all(scan, pro, None, None)[3]
    assert fut.shape == (5, FUTURE_LABEL_DIM) and float(fut.abs().max()) == 0.0
    # ... and a trained head does move with the state, which is what makes the zero a choice
    with torch.no_grad():
        m.actor.future.net[2].weight.normal_(0, 0.5)
        h = torch.randn(1, 5, 32)
        a = m.actor.step_all(scan, pro, None, None)[3]
        b = m.actor.step_all(scan, pro, None, h)[3]
    assert not torch.allclose(a, b)


def test_without_memory_the_head_reads_the_trunk():
    torch.manual_seed(5)
    m = ActorCritic(**SMALL, future_head=future_spec(source="trunk"))
    assert m.meta["future_head"]["source"] == "trunk"
    assert m.actor.future.net[0].in_features == 256
    with torch.no_grad():
        fut = m.actor.step_all(torch.rand(2, 3, 64), torch.rand(2, 16), None, None)[3]
    assert fut.shape == (2, FUTURE_LABEL_DIM)


def _batch(model, n=12, seed=7):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    b = dict(scan=r(n, 3, 64), pro=r(n, 16), priv=r(n, 21) * 2 - 1, act=r(n, 4) * 2 - 1,
             adv=r(n) * 2 - 1, ret=r(n) * 2 - 1, val=r(n) * 2 - 1, mask=(r(n) > 0.25).float())
    with torch.no_grad():
        d = model.actor.step_dist(b["scan"], b["pro"], None)[0]
    b["logp"] = (d.log_prob(b["act"]).sum(1) + (r(n) * 0.6 - 0.3)).float()
    return b


def test_the_term_off_is_byte_identical_with_and_without_the_head():
    """`--aux-future 0` is the loss it was even on a checkpoint that carries a trained head."""
    torch.set_num_threads(1)
    torch.manual_seed(9); plain = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32))
    torch.manual_seed(9); withf = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32),
                                              future_head=future_spec(width=16))
    with torch.no_grad():                       # a head that is anything but zero
        withf.actor.future.net[2].weight.normal_(0, 0.5)
    ref = copy.deepcopy(plain.actor).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    b = _batch(plain)
    hyper = PPOHyper(clip=0.2, vf=0.5, ent=0.01, kl_coef=0.3, aux_grip=0.5, aux_opp=0.25,
                     aux_future=0.0, m_gt_1=True)
    keys = ("pg", "vf", "ent", "kl_ref", "aux_grip", "aux_opp", "loss", "approx_kl", "clipfrac")

    def call(model, sequence=None):
        seq = lambda t: t if sequence is None else t.reshape(3, 4, *t.shape[1:])
        return minibatch_losses(
            model, ref, scan=seq(b["scan"]), pro=seq(b["pro"]), priv=seq(b["priv"]),
            act=seq(b["act"]), logp_old=b["logp"], adv=b["adv"], ret=b["ret"], val_old=b["val"],
            w=b["mask"], hyper=hyper, sequence=sequence)

    a, c = call(plain), call(withf)
    for key in keys:
        assert float(a[key]) == float(c[key]), f"{key} moved: {float(a[key])} vs {float(c[key])}"
    assert float(a["aux_future"]) == 0.0 and float(c["aux_future"]) == 0.0
    # and the same through the recurrent path, which is the one the arms actually train on
    seq_a = call(plain, (plain.initial_hidden(4), torch.ones(3, 4)))
    seq_c = call(withf, (withf.initial_hidden(4), torch.ones(3, 4)))
    for key in keys:
        assert float(seq_a[key]) == float(seq_c[key]), key


def test_the_term_on_without_labels_is_an_error_not_a_silent_zero():
    torch.manual_seed(9)
    m = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32), future_head=future_spec(width=16))
    ref = copy.deepcopy(m.actor).eval()
    b = _batch(m)
    with pytest.raises(ValueError, match="aux-future"):
        minibatch_losses(m, ref, scan=b["scan"], pro=b["pro"], priv=b["priv"], act=b["act"],
                         logp_old=b["logp"], adv=b["adv"], ret=b["ret"], val_old=b["val"],
                         w=b["mask"], hyper=PPOHyper(clip=0.2, vf=0.5, ent=0.0, kl_coef=0.0,
                                                     aux_future=1.0))


def test_the_term_on_trains_the_head_and_the_gru_but_not_the_action_head():
    """The gradient has to reach the recurrence: that is the whole mechanism.

    At the zero init only the output layer has a gradient -- `dL/dW_hidden = W_out^T delta . a^T` is
    zero while `W_out` is -- which is the same one-update starvation the memory and conditioning
    projections were built with, and the price of a bit-identical warm start. One update later the
    output layer is nonzero and the gradient reaches the GRU, which is what the head is FOR.
    """
    torch.manual_seed(11)
    m = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32), future_head=future_spec(width=16))
    ref = copy.deepcopy(m.actor).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    n = 12
    b = _batch(m, n=n)
    target = torch.randn(n, FUTURE_LABEL_DIM); target[:, FUTURE_PRESENT_INDEX] = 1.0

    def grads():
        out = minibatch_losses(m, ref, scan=b["scan"], pro=b["pro"], priv=b["priv"], act=b["act"],
                               logp_old=b["logp"], adv=b["adv"], ret=b["ret"], val_old=b["val"],
                               w=b["mask"], hyper=PPOHyper(clip=0.2, vf=0.0, ent=0.0, kl_coef=0.0,
                                                           aux_future=1.0),
                               future=target, future_valid=torch.ones(n))
        m.zero_grad()
        out["aux_future"].backward()
        g = lambda t: 0.0 if t.grad is None else float(t.grad.abs().max())
        return (g(m.actor.future.net[2].weight), g(m.actor.future.net[0].weight),
                g(m.actor.memory.gru.weight_ih_l0), g(m.actor.mu.weight))

    out_w, hid_w, gru, mu = grads()
    assert out_w > 0, "the output layer must train from update one"
    assert hid_w == 0.0 and gru == 0.0, "and nothing below it while that layer is still zero"
    with torch.no_grad():                                   # one update's worth of training
        m.actor.future.net[2].weight.normal_(0, 0.5)
    out_w, hid_w, gru, mu = grads()
    assert hid_w > 0 and gru > 0, "the gradient must reach the recurrence the head reads"
    assert mu == 0.0, "and never the action head: this term does not steer the car directly"


def test_warm_start_allows_the_head_and_only_the_head(tmp_path):
    torch.manual_seed(13)
    base = ActorCritic(**SMALL)
    path = str(tmp_path / "ff.pt"); save_checkpoint(path, base, {"spec": {}})
    m, _extra, fresh = load_for_memory(path, "cpu", memory_spec(hidden_size=32),
                                       future_head={"k": FUTURE_K, "width": 16})
    assert all(".memory." in f or f.startswith("actor.future.") for f in fresh), fresh
    assert any(f.startswith("actor.future.") for f in fresh)
    base.eval(); m.eval()
    scan, pro = torch.rand(4, 3, 64), torch.rand(4, 16)
    with torch.no_grad():
        assert torch.equal(base.act(scan, pro, deterministic=True)[0],
                           m.act(scan, pro, deterministic=True, h=None)[0])
    # a checkpoint that already carries one is resumed, never re-initialised -- and adding memory
    # underneath a trained head would change the width of its input, so that is refused by name
    # rather than surfacing as a shape mismatch three screens down.
    torch.manual_seed(14)
    trunk_head = ActorCritic(**SMALL, future_head=future_spec())
    assert trunk_head.meta["future_head"]["source"] == "trunk"
    again = str(tmp_path / "with_head.pt"); save_checkpoint(again, trunk_head, {"spec": {}})
    with pytest.raises(ValueError, match="already carries a future head"):
        load_for_memory(again, "cpu", memory_spec(hidden_size=32))


def test_a_head_from_a_different_build_is_refused():
    with pytest.raises(ValueError, match="targets"):
        future_spec(targets=["only", "these"])
    with pytest.raises(ValueError, match="source"):
        ActorCritic(**SMALL, future_head=future_spec(source="memory"))       # no memory to read
    with pytest.raises(ValueError, match="source"):
        ActorCritic(**SMALL, memory=memory_spec(hidden_size=8),
                    future_head=future_spec(source="trunk"))                 # memory it must read
    # an unresolved spec takes whichever the actor has, and the checkpoint records what was built
    assert ActorCritic(**SMALL).meta.get("future_head") is None
    assert ActorCritic(**SMALL, future_head={"k": 4}).meta["future_head"]["source"] == "trunk"


def test_a_resume_keeps_what_the_checkpoint_has_instead_of_warm_starting_it_again():
    """Leg two of a run passes leg one's command line with `--init` moved. The flag that BUILDS a
    piece is also the flag that keeps training it, so the routing has to ask the checkpoint."""
    from f1sim.learn.ppo import warm_start_additions
    mem, chan, fut = memory_spec(hidden_size=32), {"channels": ["edges"]}, future_spec()
    # nothing there yet: all three are a warm start
    assert warm_start_additions({}, mem, chan, fut) == (mem, chan, fut)
    # everything there: none of them is, and `load_checkpoint` takes it back whole
    have = {"memory": mem, "scan_channels": chan, "future_head": fut}
    assert warm_start_additions(have, mem, chan, fut) == (None, None, None)
    # a run that adds the head to an existing recurrent checkpoint adds only the head
    assert warm_start_additions({"memory": mem, "scan_channels": chan}, mem, chan, fut) == (None, None, fut)
    # and a flag not passed stays not passed
    assert warm_start_additions({}, None, None, fut) == (None, None, fut)


# ------------------------------------------------------------------ the probe
def test_the_probe_finds_what_the_state_determines_and_not_what_it_does_not():
    """Synthetic states, a target that is a known linear function of the state k steps earlier, and
    a target that is noise. The probe has to separate them."""
    torch.manual_seed(17)
    T, L, H, k = 90, 24, 8, 5
    states = torch.randn(T, L, H)
    w = torch.randn(H)
    labels = torch.zeros(T + 1, L, FUTURE_LABEL_DIM)
    labels[:, :, FUTURE_PRESENT_INDEX] = 1.0
    # label row s is built from the state at s - k, so the target at t + k is linear in state t
    labels[k:, :, FUTURE_LABEL_KEYS.index("ego_speed")] = (states @ w)[:T + 1 - k]
    labels[k:, :, FUTURE_LABEL_KEYS.index("opp_lon")] = (states[:, :, 0] * 2.0 - states[:, :, 3])[:T + 1 - k]
    labels[:, :, FUTURE_LABEL_KEYS.index("ego_yaw_rate")] = torch.randn(T + 1, L)   # pure noise
    out = probe_hidden.probe(states, labels, torch.zeros(T, L), k, test_frac=0.25, seed=0)
    assert out["ego_speed"]["r2"] > 0.99, out["ego_speed"]
    assert out["opp_lon"]["r2"] > 0.99, out["opp_lon"]
    assert out["ego_yaw_rate"]["r2"] < 0.2, out["ego_yaw_rate"]
    assert out["ego_speed"]["n_test"] > 0 and out["ego_speed"]["n_train"] > 0


def test_the_probe_s_mae_bins_and_presence_mask_on_a_known_answer():
    """The addendum's three additions, on data whose answer is arithmetic.

    A state that determines `opp_vlon` exactly must give R^2 ~ 1 and MAE ~ 0 **in m/s** -- the label
    is carried divided by `PRIV_OPP_DIST_SCALE`, so a MAE reported in label units would be five
    times too small and would look like a better read-out than it is. The distance bins must count
    the rows the geometry puts in them. And presence has to remove rows rather than score them as
    zeros: half the rows here have no car, and a read-out scored on those would be measuring how
    well it predicts an empty road.
    """
    from f1sim.gym_env import PRIV_OPP_DIST_SCALE as S
    torch.manual_seed(23)
    T, L, H, k = 120, 24, 6, 0
    states = torch.randn(T, L, H)
    labels = torch.zeros(T + 1, L, FUTURE_LABEL_DIM)
    present = torch.zeros(T + 1, L)
    present[:, ::2] = 1.0                                   # every other car is there at all
    labels[:, :, FUTURE_PRESENT_INDEX] = present
    w = torch.randn(H)
    labels[:T, :, FUTURE_LABEL_KEYS.index("opp_vlon")] = states @ w
    labels[:, :, FUTURE_LABEL_KEYS.index("opp_vlat")] = torch.randn(T + 1, L)   # pure noise
    # put each env column at a known distance: 1 m, 3.5 m or 8 m ahead
    dist = torch.tensor([1.0, 3.5, 8.0])[torch.arange(L) % 3]
    labels[:, :, FUTURE_LABEL_KEYS.index("opp_lon")] = (dist / S)[None, :].expand(T + 1, L)
    # One draw, so `n_test` and the bin counts describe the same split and the partition below is
    # exact rather than an average of four different ones.
    out = probe_hidden.probe(states, labels, torch.zeros(T, L), k, test_frac=0.25, seed=0, splits=1)

    v = out["opp_vlon"]
    assert v["r2"] > 0.99, v["r2"]
    assert v["mae"] < 0.05 * S, "a near-exact read-out cannot have a MAE of metres per second"
    assert v["unit"] == "m/s" and v["presence_conditioned"]
    assert v["excluded_frac"] == pytest.approx(0.5, abs=0.02), v["excluded_frac"]
    # the ego columns are not presence-conditioned: they mean something on an empty road
    assert not out["ego_speed"]["presence_conditioned"]
    assert out["ego_speed"]["excluded_frac"] == 0.0

    # the bins hold the rows the geometry put in them, and only the present cars are scored
    n = {b: v["bins"][b]["n"] for b in ("near", "mid", "far")}
    assert sum(n.values()) == pytest.approx(v["n_test"], rel=1e-6), (n, v["n_test"])
    assert all(x > 0 for x in n.values()), n
    # the bins are a partition of the scored rows, not a filter with a hole in it
    # MAE in a bin is the same quantity in the same unit, so an exact read-out is exact everywhere
    for b in ("near", "mid", "far"):
        assert v["bins"][b]["mae"] < 0.05 * S, (b, v["bins"][b])
    # and a column the state does not determine is not rescued by the binning
    assert out["opp_vlat"]["r2"] < 0.2


def test_the_probe_reads_the_tensor_the_head_is_trained_on():
    """One function returns it, so the two cannot drift apart: whatever `probe_state` hands the
    probe is what `future_from` would have fed the head on the same forward."""
    torch.manual_seed(21)
    for memory in (memory_spec(hidden_size=32), None):
        m = ActorCritic(**SMALL, memory=memory, future_head={"k": 4, "width": 16})
        with torch.no_grad():
            m.actor.future.net[2].weight.normal_(0, 0.4)
            m.actor.future.net[2].bias.normal_(0, 0.4)
        scan, pro = torch.rand(3, 3, 64), torch.rand(3, 16)
        h = None if memory is None else torch.randn(1, 3, 32)
        with torch.no_grad():
            act, state, h_next = m.actor.probe_state(scan, pro, None, h)
            from_state = m.actor.future(state)
            from_head = m.actor.step_all(scan, pro, None, h)[3]
        assert torch.equal(from_state, from_head)
        assert state.shape == (3, 32 if memory else 256)
        assert torch.equal(act, m.actor.step(scan, pro, None, h)[0])


def test_repeated_splits_report_the_spread_the_single_split_hides():
    """Eight draws of which cars are held out, not one: on this task the spread between draws is
    larger than the difference between checkpoints, and a single number hides that."""
    torch.manual_seed(29)
    T, L, H, k = 80, 16, 6, 4
    states = torch.randn(T, L, H)
    labels = torch.zeros(T + 1, L, FUTURE_LABEL_DIM)
    labels[:, :, FUTURE_PRESENT_INDEX] = 1.0
    labels[k:, :, FUTURE_LABEL_KEYS.index("ego_speed")] = (states[:, :, 0] * 3.0)[:T + 1 - k]
    labels[:, :, FUTURE_LABEL_KEYS.index("ego_yaw_rate")] = torch.randn(T + 1, L)
    one = probe_hidden.probe(states, labels, torch.zeros(T, L), k, splits=1)
    many = probe_hidden.probe(states, labels, torch.zeros(T, L), k, splits=8)
    assert one["ego_speed"]["splits"] == 1 and many["ego_speed"]["splits"] == 8
    assert np.isnan(one["ego_speed"]["r2_std"]) or one["ego_speed"]["r2_std"] == 0.0
    assert many["ego_speed"]["r2"] > 0.99
    # the noise target is where the draws disagree, and the spread has to be reported, not averaged away
    assert many["ego_yaw_rate"]["r2_max"] > many["ego_yaw_rate"]["r2_min"]
    assert many["ego_yaw_rate"]["r2_std"] > 0.0


def test_the_probe_holds_out_whole_cars():
    """A row split would let the probe interpolate inside a trajectory it has already seen. The
    columns it trains on and the columns it scores on must not overlap."""
    train, test = probe_hidden.split_columns(16, 0.25, seed=1)
    assert int(test.sum()) == 4 and not (train & test).any() and (train | test).all()


def test_the_probe_drops_the_rows_the_loss_drops():
    """Opponent targets are scored where a car is present; the alignment mask applies to all."""
    torch.manual_seed(19)
    T, L, H, k = 40, 12, 6, 3
    states = torch.randn(T, L, H)
    labels = torch.zeros(T + 1, L, FUTURE_LABEL_DIM)
    present = (torch.arange(T + 1) % 2 == 0).float()[:, None].expand(T + 1, L)
    labels[:, :, FUTURE_PRESENT_INDEX] = present
    labels[:, :, FUTURE_LABEL_KEYS.index("opp_lon")] = 1.0
    out = probe_hidden.probe(states, labels, torch.zeros(T, L), k)
    n_all = out["ego_speed"]["n_train"] + out["ego_speed"]["n_test"]
    n_opp = out["opp_lon"]["n_train"] + out["opp_lon"]["n_test"]
    assert n_all == (T - k + 1) * L
    assert n_opp == pytest.approx(n_all / 2, abs=L)
    assert np.isnan(out["opp_lon"]["r2"]), "a constant target has no variance to explain"


# ------------------------------------------------------------------ deployment
def test_export_does_not_carry_the_future_head(tmp_path, monkeypatch):
    """Deployment runs (scan, proprio, hidden) -> (action, hidden). The head is training-only, and
    an exported graph that carried it would ship privileged-label machinery to the car."""
    onnx = pytest.importorskip("onnx")
    from f1sim.learn import export
    torch.manual_seed(23)
    m = ActorCritic(**SMALL, memory=memory_spec(hidden_size=32), future_head=future_spec(width=16))
    with torch.no_grad():
        m.actor.future.net[2].weight.normal_(0, 0.5)              # a trained head, not a zero one
    ck = str(tmp_path / "m.pt"); save_checkpoint(ck, m, {"spec": {}})
    out = str(tmp_path / "m.onnx")
    monkeypatch.setattr("sys.argv", ["export", ck, "--out", out])
    export.main()
    graph = onnx.load(out).graph
    names = ([i.name for i in graph.initializer] + [o.name for o in graph.output]
             + [n.name for n in graph.node] + [i.name for i in graph.input])
    assert not any("future" in n.lower() for n in names), [n for n in names if "future" in n.lower()]
    assert [o.name for o in graph.output] == ["action", "hidden_next"]
    with open(os.path.splitext(out)[0] + ".json") as f:
        side = json.load(f)
    assert side["outputs"] == ["action", "hidden_next"]
    assert "future_head" in side["meta"], "the checkpoint still records what it was trained with"
