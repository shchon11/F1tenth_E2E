"""The pieces both v2 trainers share (`learn.train_v2`), each held on its own.

* the value normaliser: identity until it has seen returns, exactly the batch's statistics after one
  update, an EMA with Adam's debiasing after that, weights respected, a state that round-trips;
* the contact-within-1 s labels: onsets in the same episode only, a terminated episode complete, a
  truncated one and the chunk's end masked whatever the label -- against a brute-force reference;
* the imitation loss in the decoded WP4 space: zero at the label, the units it claims (metres of
  lateral position, m/s of speed), flat where the decode makes two actions one plan;
* the mirror: an involution on every key, applied per sequence, and -- where the simulator can check
  it -- the mirrored teacher label IS the label of the mirrored state, and the mirrored env's scan,
  B8 / nearest-car / per-beam labels and critic geometry are the sign tables applied to the original's;
* the B8 window: knowable only what the state saw since it started;
* per-loss gradient norms and the parameter-movement measure.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.gym_env import OPP_NEVER_SEEN, PRIV_V2_MIRROR  # noqa: E402
from f1sim.learn import train_v2 as tv  # noqa: E402
from f1sim.learn.aux_v2 import B8_LABEL_MIRROR, NEAR_MIRROR  # noqa: E402
from f1sim.learn.obs import ObsSpecV2, action_mirror_signs, ego_columns, mirror_scan  # noqa: E402
from f1sim.mpc import PlanSpec, encode_wp4, plan_length  # noqa: E402

SPEC = PlanSpec(speed_mode="wp4")


# ---------------------------------------------------------------- the value normaliser
def test_value_norm_is_the_identity_until_it_has_seen_returns_then_the_batch_statistics():
    vn = tv.ValueNorm(beta=0.99)
    x = torch.tensor([1.0, -2.0, 3.0])
    assert vn.mean == 0.0 and vn.std == 1.0 and torch.equal(vn.normalize(x), x)
    g = torch.Generator().manual_seed(0)
    r = 5.0 + 3.0 * torch.randn(4000, generator=g)
    vn.update(r)
    # the statistics are kept in float64; the float32 reference sums differ in the 7th digit
    assert vn.mean == pytest.approx(float(r.double().mean()), rel=1e-12)
    assert vn.std == pytest.approx(float(r.double().std(unbiased=False)), rel=1e-9)
    assert torch.allclose(vn.denormalize(vn.normalize(r)), r, atol=1e-5)
    assert abs(float(vn.normalize(r).mean())) < 1e-6 and float(vn.normalize(r).std()) == pytest.approx(1.0, abs=1e-3)


def test_value_norm_is_a_debiased_ema_and_respects_weights():
    b = 0.99
    vn = tv.ValueNorm(beta=b)
    vn.update(torch.zeros(10))
    vn.update(torch.full((10,), 10.0))
    want = ((1 - b) * 10.0) / (b * (1 - b) + (1 - b))                 # the second batch weighs 1/(1+b)
    assert vn.mean == pytest.approx(want, rel=1e-12)
    w = tv.ValueNorm(beta=b)
    w.update(torch.tensor([1.0, 1000.0]), torch.tensor([1.0, 0.0]))  # an off-policy row moves nothing
    assert w.mean == pytest.approx(1.0) and w.std == pytest.approx(math.sqrt(w.eps))
    c = tv.ValueNorm()
    c.load_state_dict(vn.state_dict())
    assert (c.mean, c.std, c.beta) == (vn.mean, vn.std, vn.beta)
    with pytest.raises(ValueError):
        tv.ValueNorm(beta=1.0)


# ---------------------------------------------------------------- contact within 1 s
def _brute(onset, term, trunc, H):
    T, B = onset.shape
    lab, val = torch.zeros(T, B), torch.zeros(T, B, dtype=torch.bool)
    for b in range(B):
        for t in range(T):
            hit, ok = False, None
            for u in range(t, t + H):
                if u >= T:
                    ok = False
                    break
                hit = hit or bool(onset[u, b])
                if term[u, b]:
                    ok = True
                    break
                if trunc[u, b]:
                    ok = u == t + H - 1          # cut on the window's last step: all H transitions happened
                    break
            lab[t, b] = float(hit)
            val[t, b] = True if ok is None else ok
    return lab, val


def test_contact_labels_one_onset_in_a_quiet_chunk():
    T, H = 128, 40
    onset = torch.zeros(T, 1, dtype=torch.bool); onset[50, 0] = True
    z = torch.zeros(T, 1, dtype=torch.bool)
    lab, val = tv.contact_labels(onset, z, z, H)
    assert lab[:, 0].nonzero().flatten().tolist() == list(range(11, 51))     # t + 39 >= 50 and t <= 50
    assert val[:, 0].nonzero().flatten().tolist() == list(range(0, T - H + 1))   # 89+ cross the chunk end


def test_contact_labels_at_episode_ends():
    T, H = 100, 40
    z = torch.zeros(T, 3, dtype=torch.bool)
    onset, term, trunc = z.clone(), z.clone(), z.clone()
    # env 0: a terminate-mode crash with the car at step 60: the episode is over, the label complete
    onset[60, 0] = True; term[60, 0] = True
    # env 1: a truncation at 60 and a contact at 65 that belongs to the NEXT episode
    trunc[60, 1] = True; onset[65, 1] = True
    # env 2: a crash into a wall ends the episode at 60, a car contact at 65 in the next one
    term[60, 2] = True; onset[65, 2] = True
    lab, val = tv.contact_labels(onset, term, trunc, H)
    assert lab[21:61, 0].eq(1).all() and lab[:21, 0].eq(0).all() and val[:61, 0].all()
    # t = 21's 40 transitions end exactly on the truncated step: complete. From t = 22 they are cut.
    assert val[:22, 1].all() and not val[22:61, 1].any(), "a window cut by a truncation is masked"
    assert lab[:61, 1].eq(0).all(), "the next episode's contact is not this one's"
    assert val[:61, 2].all() and lab[:61, 2].eq(0).all() and lab[61:66, 2].eq(1).all()


def test_contact_labels_match_a_brute_force_reference():
    g = torch.Generator().manual_seed(4)
    T, B, H = 90, 24, 40
    onset = torch.rand(T, B, generator=g) < 0.03
    term = torch.rand(T, B, generator=g) < 0.01
    trunc = torch.rand(T, B, generator=g) < 0.01
    lab, val = tv.contact_labels(onset, term, trunc, H)
    rl, rv = _brute(onset, term, trunc, H)
    assert torch.equal(val, rv)
    assert torch.equal(lab[val], rl[val])


def test_contact_onsets_are_the_first_charged_step_and_restart_with_an_episode():
    charge = [[False], [True], [True], [False], [True], [True], [True]]
    done = [[False], [False], [False], [False], [False], [True], [False]]
    prev = torch.zeros(1, dtype=torch.bool)
    ons = []
    for c_, d_ in zip(charge, done):
        o, prev = tv.contact_onsets(torch.tensor(c_), prev, torch.tensor(d_))
        ons.append(bool(o))
    # 1: a contact begins; 4: another one; 6: the episode ended at 5, a contact on 6 is a new onset
    assert ons == [False, True, False, False, True, False, True]


# ---------------------------------------------------------------- the decoded imitation loss
def _wp4(kappa, v, v_meas):
    return encode_wp4(torch.as_tensor(kappa, dtype=torch.float32), torch.as_tensor(v, dtype=torch.float32), 10.0,
                      SPEC, v_meas=v_meas)


def test_bc_loss_is_zero_at_the_label_and_its_lateral_part_is_in_metres():
    v_meas = torch.tensor([3.0, 8.0])
    lab = _wp4([[0.0] * 4] * 2, [[3.0] * 4] * 2, v_meas)
    loss, parts = tv.bc_loss_wp4(lab, lab, v_meas, SPEC, 10.0)
    assert float(loss) == 0.0
    # a constant curvature k is an arc: y(s) = (1 - cos(k s)) / k at each waypoint
    k = 0.4
    mu = _wp4([[k] * 4] * 2, [[3.0] * 4] * 2, v_meas)
    ys, vs = tv.wp4_decoded(mu, v_meas, SPEC, 10.0)
    L = plan_length(v_meas, SPEC)
    s = L[:, None] * torch.arange(1, 5) / 4
    assert torch.allclose(ys, (1 - torch.cos(k * s)) / k, atol=2e-3)
    assert torch.allclose(vs, torch.full_like(vs, 3.0), atol=1e-5)
    # same normalised error, longer plan at 8 m/s: the decoded loss charges what the car would feel
    l3, _ = tv.bc_loss_wp4(mu[:1], lab[:1], v_meas[:1], SPEC, 10.0)
    l8, _ = tv.bc_loss_wp4(mu[1:], lab[1:], v_meas[1:], SPEC, 10.0)
    assert float(l8) > 3 * float(l3) > 0


def test_bc_loss_speed_part_is_in_metres_per_second_and_flat_where_the_decode_is():
    v_meas = torch.tensor([6.0])
    lab = _wp4([[0.0] * 4], [[6.0, 5.0, 4.0, 3.0]], v_meas)
    mu = _wp4([[0.0] * 4], [[6.0, 5.0, 4.0, 2.0]], v_meas)          # v4 one m/s low
    loss, parts = tv.bc_loss_wp4(mu, lab, v_meas, SPEC, 10.0)
    # Huber(1.0 m/s, beta 0.1) = 0.95 on v4; v3 is pulled down by the backward braking pass too
    _ys, v_lab = tv.wp4_decoded(lab, v_meas, SPEC, 10.0)
    _ys, v_mu = tv.wp4_decoded(mu, v_meas, SPEC, 10.0)
    err = (v_mu - v_lab).abs()
    want = torch.where(err < 0.1, 0.5 * err ** 2 / 0.1, err - 0.05).mean()
    assert float(parts["speed"]) == pytest.approx(float(want), rel=1e-5) and float(parts["lat"]) == 0.0
    # with v4 = 0 the braking pass caps v1..v3 well under what is asked: asking for even more
    # there decodes to the same profile, so it costs nothing more (the backward pass is the decode's)
    stop = _wp4([[0.0] * 4], [[6.0, 6.0, 6.0, 0.0]], v_meas)
    more = _wp4([[0.0] * 4], [[9.0, 9.0, 9.0, 0.0]], v_meas)
    l0, _ = tv.bc_loss_wp4(more, stop, v_meas, SPEC, 10.0)
    assert float(l0) == pytest.approx(0.0, abs=1e-6)


def test_bc_loss_gradient_reaches_the_student_and_respects_the_weight():
    v_meas = torch.tensor([4.0, 4.0])
    lab = _wp4([[0.1] * 4, [0.1] * 4], [[4.0] * 4] * 2, v_meas)
    mu = torch.zeros(2, 8, requires_grad=True)
    loss, _ = tv.bc_loss_wp4(mu, lab, v_meas, SPEC, 10.0, weight=torch.tensor([1.0, 0.0]))
    loss.backward()
    assert float(mu.grad[0].abs().sum()) > 0 and float(mu.grad[1].abs().sum()) == 0.0


def test_a_linear_plan_becomes_the_wp4_plan_that_drives_its_first_window():
    lin = tv.linear_spec_of(SPEC)
    v = torch.tensor([2.0, 5.0])
    from f1sim.mpc import encode
    # a constant-curvature 1.5 s linear plan, speed line 5 -> 3 m/s over its length
    a_lin = encode(torch.full((2, 6), 0.3), torch.full((2,), 5.0), torch.full((2,), 3.0), 10.0, lin, v_meas=v)
    a4 = tv.linear_to_wp4(a_lin, v, lin, SPEC, 10.0)
    k = a4[:, :4] * SPEC.kappa_max
    assert torch.allclose(k, torch.full_like(k, 0.3), atol=1e-3), "an arc refits to the same arc"
    vk = (a4[:, 4:] + 1) * 5.0
    Lp, L = plan_length(v, lin), plan_length(v, SPEC)
    want = 5.0 - 2.0 * (L[:, None] * torch.arange(1, 5) / 4) / Lp[:, None]
    assert torch.allclose(vk, want, atol=1e-4)


# ---------------------------------------------------------------- the mirror
def _batch(T=5, m=4, n=31, seed=0):
    g = torch.Generator().manual_seed(seed)
    spec = ObsSpecV2(n_beams=n)
    return spec, {"scan": torch.rand(T, m, 1, n, generator=g), "ego": torch.randn(T, m, spec.ego_dim, generator=g),
                  "label": torch.rand(T, m, 8, generator=g) * 2 - 1, "priv": torch.randn(T, m, 34, generator=g),
                  "b8_lab": torch.randn(T, m, 3, 3, generator=g), "near": torch.randn(T, m, 5, generator=g),
                  "beam": torch.randint(-1, 2, (T, m, n), generator=g).to(torch.int8),
                  "c": torch.randn(T, m, 1, generator=g), "keep": torch.ones(T, m),
                  "b8_age": torch.randint(0, 200, (T, m, 3), generator=g)}


def test_mirroring_twice_is_the_identity_on_every_key_and_only_flipped_sequences_move():
    spec, b = _batch()
    mir = tv.Mirror(spec)
    flip = torch.tensor([True, False, True, False])
    once = mir.apply(b, flip)
    twice = mir.apply(once, flip)
    for k in b:
        assert torch.equal(twice[k], b[k]), k
        assert torch.equal(once[k][:, 1], b[k][:, 1]) and torch.equal(once[k][:, 3], b[k][:, 3]), k
    assert torch.equal(once["scan"][:, 0], mirror_scan(b["scan"][:, 0]))
    assert torch.equal(once["beam"][:, 2], b["beam"][:, 2].flip(-1))
    assert torch.equal(once["c"], b["c"]) and torch.equal(once["b8_age"], b["b8_age"])
    col = {c.name: c.index for c in ego_columns(spec)}
    e, e1 = b["ego"][:, 0], once["ego"][:, 0]
    assert torch.equal(e1[:, col["speed@t"]], e[:, col["speed@t"]])
    assert torch.equal(e1[:, col["gyro_z@t"]], -e[:, col["gyro_z@t"]])
    assert torch.equal(e1[:, col["action[0]@t-2"]], -e[:, col["action[0]@t-2"]])
    assert torch.equal(e1[:, col["action[5]@t-3"]], e[:, col["action[5]@t-3"]])
    assert torch.equal(once["label"][:, 0], b["label"][:, 0] * action_mirror_signs("wp4"))
    assert torch.equal(once["b8_lab"][:, 0], b["b8_lab"][:, 0] * torch.tensor(B8_LABEL_MIRROR))
    assert torch.equal(once["near"][:, 0], b["near"][:, 0] * torch.tensor(NEAR_MIRROR))
    assert torch.equal(once["priv"][:, 0], b["priv"][:, 0] * torch.tensor(PRIV_V2_MIRROR))


def test_the_mirror_of_a_state_is_an_involution():
    s = torch.tensor([[1.0, 2.0, 0.3, 4.0, 0.2, 0.5, 0.1, 13.0]])
    m = tv.mirror_state(s, -1.0, 7.0)
    assert m[0, :2].tolist() == pytest.approx([5.0, 2.0]) and float(m[0, 2]) == pytest.approx(math.pi - 0.3)
    mm = tv.mirror_state(m, -1.0, 7.0)
    assert torch.allclose(torch.atan2(torch.sin(mm[:, 2]), torch.cos(mm[:, 2])), s[:, 2])
    assert torch.allclose(mm[:, [0, 1, 3, 4, 5, 6, 7]], s[:, [0, 1, 3, 4, 5, 6, 7]])


def _mirrored_raceline(track, rl):
    from f1sim.raceline import Raceline
    W = track.occupancy.shape[1]
    x_lo, x_hi = float(track.origin[0]), float(track.origin[0] + (W - 1) * track.resolution)
    xy = rl.xy.copy(); xy[:, 0] = x_lo + x_hi - xy[:, 0]
    return Raceline(xy=xy, v=rl.v.copy(), kappa=-rl.kappa.copy(), s=rl.s.copy(), length=rl.length,
                    lap_time=rl.lap_time), x_lo, x_hi


def _states_on_line(rl, n=16, seed=0):
    rng = np.random.default_rng(seed)
    N = len(rl.xy)
    idx = np.linspace(0, N, n, endpoint=False).astype(int)
    d = rl.xy[(idx + 1) % N] - rl.xy[(idx - 1) % N]
    yaw = np.arctan2(d[:, 1], d[:, 0])
    off = rng.uniform(-0.15, 0.15, n)
    x = rl.xy[idx, 0] - np.sin(yaw) * off
    y = rl.xy[idx, 1] + np.cos(yaw) * off
    vx = rng.uniform(1.5, 6.0, n)
    st = np.stack([x, y, yaw + rng.uniform(-0.1, 0.1, n), vx, rng.uniform(-0.1, 0.1, n),
                   rng.uniform(-0.3, 0.3, n), np.zeros(n), vx / 0.05], 1)
    return torch.tensor(st, dtype=torch.float32)


@pytest.mark.slow
def test_a_mirrored_teacher_label_is_the_label_of_the_mirrored_state():
    """The raceline teacher on a map and on its mirror image (`Track.mirrored`, the line mirrored
    with it): at mirrored states its linear plan and the wp4 label refitted from it are the
    original's with the curvature columns' signs flipped -- `obs.action_mirror_signs`, the table the
    DAgger mirror multiplies the labels by.

    Not to the last bit, and measured why (16 states on gen:competition:0, normalised units): the
    teacher's Gauss-Newton fits use a ONE-SIDED finite-difference Jacobian (eps 0.02 1/m), whose error
    has no mirror, so the mirrored fit converges to a fixed point up to 0.0062 away (0.010 1/m; more
    iterations do not move it: 6 / 12 / 24 / 48 all 0.0062), and the speed certification differs by
    up to 0.0096 (its profile of the mirrored line by 0.02 m/s). With the certification off the speeds
    agree to 1e-7. A wrong sign table would be off by twice the label, up to 0.75 here."""
    from f1sim.learn import common
    from f1sim.teacher import RacelineTeacher
    tracks, rls = common.load_tracks(["gen:competition:0"], racelines=True)
    rl_m, x_lo, x_hi = _mirrored_raceline(tracks[0], rls[0])
    ta, tb = RacelineTeacher([rls[0]]), RacelineTeacher([rl_m])
    sa = _states_on_line(rls[0])
    sb = tv.mirror_state(sa, x_lo, x_hi)
    lin = tv.linear_spec_of(SPEC)
    v = sa[:, 3]
    tid = torch.zeros(len(sa), dtype=torch.long)
    la = ta.plan_action(sa, None, tid, 10.0, lin, plan_speed=v)
    lb = tb.plan_action(sb, None, tid, 10.0, lin, plan_speed=v)
    d = (lb - la * action_mirror_signs("linear")).abs()
    assert float(d[:, :6].max()) < 0.01 and float(d[:, 6:].max()) < 0.02, d.max(0).values
    wa = tv.linear_to_wp4(la, v, lin, SPEC, 10.0)
    wb = tv.linear_to_wp4(lb, v, lin, SPEC, 10.0)
    dw = (wb - wa * action_mirror_signs("wp4")).abs()
    assert float(dw[:, :4].max()) < 0.015 and float(dw[:, 4:].max()) < 0.02, dw.max(0).values
    assert float(wa[:, :4].abs().max()) > 0.2, "the states should not all be on straights"
    assert float((wb - wa).abs()[:, :4].max()) > 0.4, "without the flip the labels disagree"
    ta.certify_speeds = tb.certify_speeds = False
    la = ta.plan_action(sa, None, tid, 10.0, lin, plan_speed=v)
    lb = tb.plan_action(sb, None, tid, 10.0, lin, plan_speed=v)
    assert float((lb - la)[:, 6:].abs().max()) < 1e-5


@pytest.mark.slow
def test_the_mirrored_world_is_the_sign_tables_applied_to_the_original():
    """Two simulators, a map and its mirror image, the same two cars placed mirror-wise: the newest
    scan reversed beam for beam (`obs.mirror_scan`), the B8 track labels times `B8_LABEL_MIRROR`, the
    nearest-visible-car target times `NEAR_MIRROR`, the per-beam car labels reversed, and the critic's
    opponent block and geometry times `PRIV_V2_MIRROR` -- every table the DAgger mirror applies,
    checked against what the simulator computes for the mirrored world. Noise-free scans (the
    noise is drawn per beam and has no mirror); the car drawn as boxes, which are symmetric."""
    from f1sim import Config
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    tracks, _rls = common.load_tracks(["gen:competition:0"], racelines=True)
    t = tracks[0]
    W = t.occupancy.shape[1]
    x_lo, x_hi = float(t.origin[0]), float(t.origin[0] + (W - 1) * t.resolution)

    def env_on(track):
        cfg = Config(); cfg.rand.enabled = False; cfg.sim.compile = False
        cfg.lidar.car_model = "parts"
        ec = EnvConfig(action_mode="plan", speed_mode="wp4", race_size=2, opponent="policy", hist_len=20,
                       hist_stride=2, scan_stack=1, compile_tracker=False, obs_version=2, collision_mode="soft",
                       opp_visibility=True)
        e = common.make_env([track], 2, "cpu", ec, cfg=cfg, seed=3)
        e.reset(seed=3)
        return e

    ea, eb = env_on(t), env_on(t.mirrored())
    sa = ea.sim.state.clone()
    # row 0 where reset put it; row 1 (its opponent) 2.0 m ahead and 0.4 m to its left, turned a little
    x, y, yaw = (float(q) for q in sa[0, :3])
    sa[1, 0] = x + 2.0 * math.cos(yaw) - 0.4 * math.sin(yaw)
    sa[1, 1] = y + 2.0 * math.sin(yaw) + 0.4 * math.cos(yaw)
    sa[1, 2] = yaw + 0.2; sa[1, 3] = 2.0; sa[0, 3] = 3.0
    ea.sim.state[:] = sa
    eb.sim.state[:] = tv.mirror_state(sa, x_lo, x_hi)

    def rescan(env):
        sim = env.sim
        scan, true, typ = sim.lidar.scan(sim.state[:, :3], None, sim.P, motion_distortion=False, noisy=False,
                                         tid=sim.tid, cars=sim._car_boxes(sim.state), compiled=False, eid=sim.eid)
        res = dataclasses.replace(env.last_result, scan=scan, scan_true=true, scan_type=typ, state=sim.state.clone())
        env._update_opp_visibility(res, None)
        return res

    ra, rb = rescan(ea), rescan(eb)
    na, nb = ea._norm_scan(ra.scan), eb._norm_scan(rb.scan)
    d_scan = (nb - mirror_scan(na)).abs()
    assert float(d_scan.median()) < 1e-4 and float((d_scan > 0.01).float().mean()) < 0.01, float(d_scan.max())
    la, lb = ea.beam_car_labels_v2(ra), eb.beam_car_labels_v2(rb)
    assert int((la[0] == 1).sum()) >= 10, "the car 2 m ahead should be in the scan"
    assert float((lb != la.flip(-1)).float().mean()) < 0.01
    ta, tb = ea.opponent_tracks_v2(ra), eb.opponent_tracks_v2(rb)
    assert torch.equal(ta["cand"], tb["cand"]) and int(ta["age"][0, 0]) == int(tb["age"][0, 0]) == 0
    assert torch.allclose(tb["lab"], ta["lab"] * torch.tensor(B8_LABEL_MIRROR), atol=1e-4)
    assert torch.allclose(eb.nearest_visible_car_v2(rb), ea.nearest_visible_car_v2(ra) * torch.tensor(NEAR_MIRROR),
                          atol=1e-4)
    pa, pb = ea.privileged_v2(ra), eb.privileged_v2(rb)
    sign = torch.tensor(PRIV_V2_MIRROR)
    assert torch.allclose(pb[:, 8:], pa[:, 8:] * sign[8:], atol=1e-4), "opponent block / geometry"
    assert float(pa[0, 9]) > 0.2, "the opponent block should hold the car ahead"


# ---------------------------------------------------------------- the B8 window
def test_steps_since_window_start_counts_from_the_window_and_every_episode_start():
    keep = torch.ones(7, 2); keep[3, 1] = 0.0
    s = tv.steps_since_window_start(keep)
    assert s[:, 0].tolist() == [0, 1, 2, 3, 4, 5, 6] and s[:, 1].tolist() == [0, 1, 2, 0, 1, 2, 3]


def test_b8_sequence_loss_asks_only_for_what_the_window_has_seen():
    T, m = 2, 1
    lab = torch.zeros(T, m, 3, 3); lab[..., 0, :] = torch.tensor([0.4, 0.2, 0.0])
    cand = torch.zeros(T, m, 3, dtype=torch.bool); cand[..., 0] = True
    age = torch.full((T, m, 3), OPP_NEVER_SEEN, dtype=torch.long); age[..., 0] = 30
    wrong = torch.zeros(T, m, 3, 4); wrong[..., 0] = -5.0; wrong[..., 1:] = 3.0   # confident "nobody", far off
    wrong[..., 1:, 0] = -20.0
    seen_before = tv.b8_sequence_loss(wrong, lab, cand, age, torch.full((T, m), 10))
    seen_inside = tv.b8_sequence_loss(wrong, lab, cand, age, torch.full((T, m), 40))
    assert float(seen_before[0]) < 1e-6 and float(seen_before[1]["knowable"]) == 0.0
    assert float(seen_inside[0]) > 1.0 and float(seen_inside[1]["knowable"]) == 1.0
    # older than the 3 s memory limit: not knowable however long the window
    age[..., 0] = 121
    assert float(tv.b8_sequence_loss(wrong, lab, cand, age, torch.full((T, m), 500))[0]) < 1e-6


def test_select_tracks_keeps_the_three_nearest_candidates_with_their_age_and_pads_small_races():
    lab = torch.arange(5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    cand = torch.tensor([[True, True, False, True, True]])
    age = torch.tensor([[7, 8, 9, 10, 11]])
    dist = torch.tensor([[4.0, 1.0, 0.5, 5.5, 2.0]])
    l3, c3, a3 = tv.select_tracks(lab, cand, age, dist)
    assert l3[0, :, 0].tolist() == [3.0, 12.0, 0.0] and a3.tolist() == [[8, 11, 7]] and c3.all()
    l1, c1, a1 = tv.select_tracks(lab[:, :1], cand[:, :1], age[:, :1], dist[:, :1])
    assert l1.shape == (1, 3, 3) and c1.tolist() == [[True, False, False]]
    assert a1.tolist() == [[7, OPP_NEVER_SEEN, OPP_NEVER_SEEN]]
    l0, c0, a0 = tv.select_tracks(torch.zeros(2, 0, 3), torch.zeros(2, 0, dtype=torch.bool),
                                  torch.zeros(2, 0, dtype=torch.long), torch.zeros(2, 0))
    assert l0.shape == (2, 3, 3) and not c0.any()


# ---------------------------------------------------------------- gradient norms, movement
def test_loss_grad_norms_measure_each_term_and_leave_grad_alone():
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    u = torch.nn.Parameter(torch.tensor([3.0]))
    x = torch.tensor([3.0, 4.0])
    a = (w * x).sum()                      # d/dw = x, |x| = 5
    b = 2.0 * (w ** 2).sum() + 0 * u.sum()  # d/dw = 4 w, |4 w| = 4 sqrt(5)
    gn = tv.loss_grad_norms({"a": a, "b": b, "none": None, "const": torch.tensor(1.0)}, [w, u])
    assert gn["a"] == pytest.approx(5.0) and gn["b"] == pytest.approx(4 * math.sqrt(5.0))
    assert gn["none"] == 0.0 and gn["const"] == 0.0
    assert w.grad is None
    (a + b).backward()                     # the graph is still there
    assert w.grad is not None


def test_relative_movement():
    p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    before = tv.flat_parameters([p])
    with torch.no_grad():
        p.add_(torch.tensor([0.0, 0.05]))
    assert tv.relative_movement(before, [p]) == pytest.approx(0.01, rel=1e-5)


def test_actor_trunk_parameters_are_the_shared_blocks_only():
    from f1sim.learn.model_v2 import ActorCriticV2
    torch.manual_seed(0)
    m = ActorCriticV2(n_beams=271, aux={"b8": True, "near_car": True})
    trunk = {id(p) for p in tv.actor_trunk_parameters(m.actor)}
    heads = [p for h in (m.actor.b8, m.actor.near_car, m.actor.head, m.actor.mu) for p in h.parameters()]
    assert not any(id(p) in trunk for p in heads) and id(m.actor.gru.weight_hh_l0) in trunk
    deployed = {id(p) for p in tv.deployed_actor_parameters(m.actor)}
    assert not any(id(p) in deployed for h in (m.actor.b8, m.actor.near_car) for p in h.parameters())
    assert id(m.actor.log_std) in deployed and id(m.actor.mu.weight) in deployed
