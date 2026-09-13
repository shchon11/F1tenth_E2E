"""The ego-motion-aligned scan channel: the warp's geometry, its gate, and what it costs elsewhere.

The channel's whole claim is one sentence -- *static geometry cancels, what moved by itself does
not* -- and it is a geometric claim, so it is checked against scenes whose answer is known by
construction rather than against a recorded output. A circular wall with an optional disc in it is
enough: every range is analytic, so "the residual is zero" is an assertion about the warp and not
about a fixture.

The rest of the file is the discipline the contract asks for around it: the flag off is
byte-identical, the flag on is a warm start that changes no action, and the proprio columns the warp
reads are the ones the simulator and the car actually put there.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from f1sim.learn.aligned import (ALIGNED_ROWS, AlignedScan, aligned_spec, beam_angles,
                                 compose_increments, compose_stack, gate, soft_threshold,
                                 step_increment,
                                 tilt_matrix, warp_scan)
from f1sim.learn.obs import (ALIGNED_CHANNELS, SCAN_CHANNELS, ObsSpec, ScanAugment,
                             motion_from_proprio, motion_index_spec)
from f1sim.learn.model import ActorCritic, load_for_memory, save_checkpoint, scan_channel_spec

N_BEAMS, FOV, RANGE_MAX, DT = 361, 4.71238898, 10.0, 0.025
ANG = beam_angles(N_BEAMS, FOV)


# ------------------------------------------------------------------ synthetic worlds
def ring_scan(pose, att=(0.0, 0.0), discs=(), radius: float = 8.0, height: float = 0.11):
    """Ranges from inside a circular wall, with the simulator's own 3D beam geometry.

    `pose` is (x, y, yaw) of the sensor, `att` its (roll, pitch). Beams are built the way
    `lidar.Lidar.rays` builds them -- `Ry(pitch) Rx(roll)` applied to the level bearing -- so a
    tilted plane hits the floor ahead exactly as it does in the simulator, and a test of the warp's
    tilt handling is a test against the same convention the plant uses.

    `discs` are (x, y, radius) vertical cylinders: the moving car stand-in.
    """
    x, y, yaw = pose
    d = torch.stack([torch.cos(ANG), torch.sin(ANG), torch.zeros_like(ANG)], -1)     # (N,3)
    d = (tilt_matrix(torch.tensor([att[0]]), torch.tensor([att[1]]))[0] @ d.T).T
    c, s = math.cos(yaw), math.sin(yaw)
    wx = d[:, 0] * c - d[:, 1] * s
    wy = d[:, 0] * s + d[:, 1] * c
    hn = torch.sqrt(wx * wx + wy * wy).clamp_min(1e-9)
    dh = torch.stack([wx / hn, wy / hn], -1)
    k = d[:, 2] / hn                                     # dz per horizontal metre
    o = torch.tensor([x, y])
    b = (o[None] * dh).sum(-1)
    t_wall = -b + torch.sqrt((b * b - ((o * o).sum() - radius * radius)).clamp_min(0))
    t = t_wall
    desc = k < -1e-9
    t = torch.where(desc, torch.minimum(t, height / (-k).clamp_min(1e-9)), t)        # the floor
    for (cx, cy, rad) in discs:
        oc = torch.tensor([x - cx, y - cy])
        bb = (oc[None] * dh).sum(-1)
        disc = bb * bb - ((oc * oc).sum() - rad * rad)
        th = -bb - torch.sqrt(disc.clamp_min(0))
        t = torch.where((disc > 0) & (th > 0), torch.minimum(t, th), t)
    return (t * torch.sqrt(1 + k * k)).clamp(0.0, RANGE_MAX)


def drive(steps: int, v: float = 6.0, w: float = 0.0, att=lambda i: (0.0, 0.0), discs=lambda i: (),
          spec=None, use_tilt: bool = True, radius: float = 8.0, att_bias=(0.0, 0.0)):
    """Run the channel along a trajectory and return the rows of every step.

    `att_bias` is added to the attitude the CHANNEL is told about and not to the one the scan is
    generated from: that is a bias in the estimate, which is the thing the warp has to be immune to.
    """
    arm = AlignedScan(N_BEAMS, 1, spec or {}, dt=DT, range_max=RANGE_MAX)
    x = y = th = 0.0
    out = []
    for i in range(steps):
        a = att(i)
        r = ring_scan((x, y, th), a, discs(i), radius)
        told = ((a[0] + att_bias[0], a[1] + att_bias[1]) if use_tilt else (0.0, 0.0))
        m = torch.tensor([[v, w, told[0], told[1]]])
        out.append({k: t[0] * RANGE_MAX for k, t in arm((r / RANGE_MAX)[None], m).items()})
        th += w * DT
        x += v * DT * math.cos(th)
        y += v * DT * math.sin(th)
    return out


# ------------------------------------------------------------------ the warp's geometry
@pytest.mark.parametrize("w", [0.0, 1.2, -2.0])
def test_a_static_world_cancels_exactly(w):
    """The claim, on a scene whose every range is analytic: nothing moved, so nothing is reported.

    Exactly zero, not nearly: the warp is fed the same speed and yaw rate the trajectory was built
    from, so any residual at all is the geometry being wrong rather than a measurement being noisy.
    """
    rows = drive(12, v=6.0, w=w)
    for step in rows[6:]:
        assert float(step["aligned"].abs().max()) == 0.0
        assert float(step["aligned_valid"].mean()) > 0.5     # and it did have a prediction to make


def test_a_moving_object_survives_and_is_signed_and_local():
    """A disc crossing in front: a residual only where it is, negative where it arrived."""
    v = 6.0
    # Both discs start in front of the car; the still one is fixed IN THE WORLD, which is what the
    # warp is supposed to cancel. (A disc that kept station with the ego would be moving in the
    # world, and the channel would be right to report it.)
    moving = drive(12, v=v, discs=lambda i: ((4.0, -1.5 + 0.12 * i, 0.25),))
    still = drive(12, v=v, discs=lambda i: ((4.0, -1.5, 0.25),))
    for i in range(6, 12):
        assert float(still[i]["aligned"].abs().max()) == 0.0, "a world-fixed disc is static"
        r = moving[i]["aligned"]
        nz = r != 0
        assert int(nz.sum()) > 0, "a disc moving across the field is not nothing"
        assert float(r.min()) < 0 and float(r.max()) > 0, "it left one bearing and arrived at another"
        deg = torch.rad2deg(ANG[nz])
        assert float(deg.max() - deg.min()) < 60.0, "and it is local, not a whole-scan wash"


def test_a_no_return_warps_as_a_bound_not_as_a_point():
    """Driving up a straight whose far beams read the clamp reports nothing.

    A no-return says "free out to range_max, that way": it is a lower bound. Warped as a point at
    range_max it would move with the car and report 0.9 m of "motion" on every far beam of every
    straight -- a systematic false positive on exactly the part of the scan a policy reads to plan.
    """
    rows = drive(12, v=9.0, radius=40.0)                 # a wall further away than the scanner sees
    far = ring_scan((0.0, 0.0, 0.0), radius=40.0) >= RANGE_MAX - 1e-4
    assert bool(far.any()), "this fixture is meant to produce no-return beams"
    for step in rows[6:]:
        assert float(step["aligned"][far].abs().max()) == 0.0


def test_tilt_compensation_beats_ignoring_it():
    """Rolling and pitching a scan plane changes what it sees; the measured attitude cancels most.

    The plant's own numbers: 1 deg rms of floor wobble with a 0.4 s time constant, on top of
    1.7 deg/g of roll (`docs/real_data_calibration.md` 6.1a). Here the attitude is a deterministic
    wobble of that size and nothing in the world moves, so every residual is tilt.
    """
    att = lambda i: (math.radians(1.5) * math.sin(i * 0.9), math.radians(1.5) * math.cos(i * 0.7))
    on = drive(14, v=6.0, att=att, use_tilt=True)
    off = drive(14, v=6.0, att=att, use_tilt=False)
    err = lambda rows: float(torch.stack([r["aligned"].abs() for r in rows[6:]]).mean())
    assert err(on) < err(off), (err(on), err(off))


def test_a_constant_attitude_bias_changes_nothing():
    """The warp uses how the tilt CHANGED, so a constant offset on the estimate is invisible to it.

    Not a nicety: the VESC attitude estimate carries a bias and drifts (11 deg rms in simulation,
    past 40 deg in five of thirteen competition recordings). A warp that used the absolute value
    would apply the ego's planar motion in a frame tilted by that bias and throw away, as
    out-of-plane, points the current scan can see.
    """
    wobble = lambda i: (math.radians(1.0) * math.sin(i * 0.8), math.radians(1.0) * math.cos(i * 0.6))
    # Same world, same scans, same trajectory; only what the channel is TOLD about the attitude
    # differs, by a constant. Every row must come out identical.
    a = drive(14, v=6.0, w=0.8, att=wobble)
    b = drive(14, v=6.0, w=0.8, att=wobble, att_bias=(math.radians(9.0), math.radians(-6.0)))
    for i in range(6, 14):
        for row in ("aligned", "aligned_prev", "aligned_valid"):
            assert torch.allclose(a[i][row], b[i][row], atol=1e-5), (i, row)


def test_the_warp_uses_the_measured_motion_and_not_a_guess():
    """Told the wrong speed, the channel reports the whole static world as moving."""
    arm_right = AlignedScan(N_BEAMS, 1, {}, dt=DT, range_max=RANGE_MAX)
    arm_wrong = AlignedScan(N_BEAMS, 1, {}, dt=DT, range_max=RANGE_MAX)
    x, v = 0.0, 6.0
    for i in range(10):
        r = (ring_scan((x, 0.0, 0.0)) / RANGE_MAX)[None]
        right = arm_right(r, torch.tensor([[v, 0.0, 0.0, 0.0]]))
        wrong = arm_wrong(r, torch.tensor([[0.0, 0.0, 0.0, 0.0]]))
        x += v * DT
    # The rows are in the scan's own normalised units, like every other scan channel, so a metre is
    # 1 / range_max here.
    assert float(right["aligned"].abs().max()) == 0.0
    assert float(wrong["aligned"].abs().max()) * RANGE_MAX > 0.4, "0.6 m of ego motion, minus tau"


def test_the_closed_form_composition_is_the_recursion():
    """`compose_stack` is what the channel runs; `compose_increments` is what it reads like."""
    g = torch.Generator().manual_seed(7)
    for k in (1, 2, 4, 8):
        p = torch.randn(k, 3, 2, generator=g) * 0.2
        dy = torch.randn(k, 3, generator=g) * 0.05
        a0, b0 = compose_increments([(p[i], dy[i]) for i in range(k)])
        a1, b1 = compose_stack(p, dy)
        assert torch.allclose(a0, a1, atol=1e-6), (k, a0, a1)
        assert torch.allclose(b0, b1, atol=1e-6), (k, b0, b1)


def test_step_increment_and_composition_are_the_arc_they_claim():
    """One composed transform equals integrating the same motion step by step."""
    v, w, k = torch.tensor([5.0]), torch.tensor([1.5]), 4
    incs = [step_increment(v, w, v, w, DT) for _ in range(k)]
    a, b = compose_increments(incs)
    # A world point, carried by hand through k frames the long way round.
    p = torch.tensor([[3.0, 1.0]])
    for dp, dyaw in incs:
        c, s = torch.cos(dyaw), torch.sin(dyaw)
        q = p - dp
        p = torch.stack([q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c], 1)
    got = (a @ torch.tensor([[3.0, 1.0]]).unsqueeze(-1)).squeeze(-1) + b
    assert torch.allclose(got, p, atol=1e-5), (got, p)
    # and a straight line is a straight line
    p0, dyaw0 = step_increment(v, torch.zeros(1), v, torch.zeros(1), DT)
    assert torch.allclose(p0, torch.tensor([[float(v) * DT, 0.0]]), atol=1e-6)
    assert float(dyaw0.abs().max()) == 0.0


# ------------------------------------------------------------------ the gate
def test_soft_threshold_shrinks_rather_than_cuts():
    r = torch.tensor([-0.5, -0.05, 0.0, 0.05, 0.5])
    got = soft_threshold(r, torch.tensor(0.1))
    assert torch.allclose(got, torch.tensor([-0.4, 0.0, 0.0, 0.0, 0.4]), atol=1e-6)
    # continuity at the threshold: a residual just past tau is small, not a step
    assert float(soft_threshold(torch.tensor([0.1001]), torch.tensor(0.1))) < 1e-3


def test_the_consistency_test_drops_a_one_frame_spike_and_keeps_a_two_frame_one():
    raw = torch.zeros(1, 40)
    raw[0, 20] = 1.0
    now = torch.full((1, 40), 5.0)
    none_yet = gate(raw, now, None, None, 0.1, 0.0, 8)[0]
    assert float(none_yet.abs().max()) == 0.0, "nothing has a predecessor on the first step"
    pos = torch.zeros(1, 40, dtype=torch.bool); pos[0, 24] = True      # 4 beams away, same sign
    kept = gate(raw, now, pos, torch.zeros_like(pos), 0.1, 0.0, 8)[0]
    assert float(kept[0, 20]) == pytest.approx(0.9)
    wrong_sign = gate(raw, now, torch.zeros_like(pos), pos, 0.1, 0.0, 8)[0]
    assert float(wrong_sign.abs().max()) == 0.0
    too_far = torch.zeros(1, 40, dtype=torch.bool); too_far[0, 33] = True
    assert float(gate(raw, now, too_far, torch.zeros_like(pos), 0.1, 0.0, 8)[0].abs().max()) == 0.0
    # and with the test switched off the shrinkage stands alone, predecessor or not
    assert float(gate(raw, now, None, None, 0.1, 0.0, 0)[0][0, 20]) == pytest.approx(0.9)


def test_tau_is_what_a_static_floor_of_that_size_would_leave():
    """A synthetic floor of known size is removed by a tau above it and kept by one below."""
    rng = torch.Generator().manual_seed(3)
    raw = torch.randn(1, 2000, generator=rng) * 0.03
    now = torch.full((1, 2000), 4.0)
    prev = torch.ones(1, 2000, dtype=torch.bool)
    for tau, want in ((0.09, 0.005), (0.01, 0.5)):
        out = gate(raw, now, prev, prev, tau, 0.0, 0)[0]
        assert (out != 0).float().mean() < want if tau > 0.05 else (out != 0).float().mean() > want


# ------------------------------------------------------------------ the channel's state
def test_the_channel_is_silent_until_it_has_k_scans_and_after_a_reset():
    arm = AlignedScan(N_BEAMS, 2, {"k": 4}, dt=DT, range_max=RANGE_MAX)
    scan = torch.rand(2, N_BEAMS) * 0.5 + 0.4
    m = torch.tensor([[6.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0]])
    for i in range(4):
        assert float(arm(scan, m)["aligned"].abs().max()) == 0.0, f"step {i} has no scan to warp"
    for _ in range(4):
        arm(torch.rand(2, N_BEAMS) * 0.5 + 0.4, m)
    arm.reset(torch.tensor([True, False]))
    assert int(arm.seen[0]) == 0 and int(arm.seen[1]) > 0
    out = arm(scan, m)["aligned"]
    assert float(out[0].abs().max()) == 0.0, "the reset row starts again"


def test_preview_does_not_advance_the_channel():
    arm = AlignedScan(N_BEAMS, 2, {}, dt=DT, range_max=RANGE_MAX)
    m = torch.tensor([[6.0, 0.1, 0.0, 0.0]] * 2)
    for _ in range(8):
        arm(torch.rand(2, N_BEAMS) * 0.5 + 0.4, m)
    before = (arm.scans.clone(), arm.seen.clone(), arm.prev_pos.clone())
    scan = torch.rand(2, N_BEAMS) * 0.5 + 0.4
    a = arm.preview(scan, m)
    assert torch.equal(arm.scans, before[0]) and torch.equal(arm.seen, before[1])
    assert torch.equal(arm.prev_pos, before[2])
    b = arm.preview(scan, m)
    assert torch.equal(a["aligned"], b["aligned"]), "and it is repeatable"


def test_valid_marks_where_the_warp_actually_predicted():
    """`aligned_valid` is 0 where no warped point reached the bin, and the residual there is 0."""
    rows = drive(10, v=6.0, w=2.5)                        # a hard turn rotates the window
    step = rows[-1]
    invalid = step["aligned_valid"] == 0
    assert bool(invalid.any()), "a hard turn brings new bearings into the window"
    assert float(step["aligned"][invalid].abs().max()) == 0.0
    assert float(step["aligned_prev"][invalid].min()) == pytest.approx(RANGE_MAX)


# ------------------------------------------------------------------ the observation contract
def test_motion_columns_are_the_ones_the_observation_actually_carries():
    spec = ObsSpec(n_beams=64, scan_stack=6, act_dim=8, hist_len=20)
    idx = motion_index_spec(spec)
    pro = torch.zeros(3, spec.proprio_dim)
    pro[:, idx["speed"]] = torch.tensor([0.1, 0.5, 0.9])
    pro[:, idx["yaw_rate"]] = torch.tensor([0.2, -0.4, 0.0])
    pro[:, idx["roll"]] = torch.tensor([0.1, 0.0, -0.2])
    pro[:, idx["pitch"]] = torch.tensor([0.0, 0.3, 0.1])
    m = motion_from_proprio(pro, idx)
    assert torch.allclose(m[:, 0], torch.tensor([1.0, 5.0, 9.0]))
    assert torch.allclose(m[:, 1], torch.tensor([1.0, -2.0, 0.0]))
    assert torch.allclose(m[:, 2], torch.tensor([0.035, 0.0, -0.07]), atol=1e-6)
    with pytest.raises(ValueError, match="wide proprio"):
        motion_from_proprio(torch.zeros(3, spec.proprio_dim + 1), idx)


def test_the_channel_refuses_to_run_on_an_implied_zero_motion():
    spec = ObsSpec(n_beams=32, scan_stack=6, act_dim=8, hist_len=20)
    chan = scan_channel_spec({"channels": ["aligned"], "aligned": {"proprio": motion_index_spec(spec)}})
    aug = ScanAugment(chan["channels"], 32, 2, aligned=chan["aligned"])
    with pytest.raises(ValueError, match="no proprio"):
        aug(torch.rand(2, 6, 32))


def test_the_rows_are_computed_once_however_many_are_asked_for():
    """Three channels, one warp: asking for the mask beside the residual must not advance the ring
    buffers twice, which would compare this scan against the wrong one."""
    spec = ObsSpec(n_beams=48, scan_stack=6, act_dim=8, hist_len=20)
    cfg = {"proprio": motion_index_spec(spec)}
    one = ScanAugment(["aligned"], 48, 1, aligned=cfg)
    three = ScanAugment(list(ALIGNED_CHANNELS), 48, 1, aligned=cfg)
    pro = torch.zeros(1, spec.proprio_dim); pro[:, 0] = 0.6
    for _ in range(8):
        scan = torch.rand(1, 6, 48) * 0.5 + 0.4
        a, b = one(scan, pro), three(scan, pro)
    assert one.aligned.seen.tolist() == three.aligned.seen.tolist()
    assert torch.equal(a[:, -1], b[:, -3]), "the residual row is the same row in both"


def test_the_channel_order_is_the_column_order():
    assert SCAN_CHANNELS[:2] == ("memory", "edges"), "existing checkpoints keep their columns"
    assert tuple(ALIGNED_CHANNELS) == SCAN_CHANNELS[2:]
    assert tuple(ALIGNED_ROWS) == tuple(ALIGNED_CHANNELS)


# ------------------------------------------------------------------ what it costs the rest
SMALL = dict(n_stack=6, n_beams=128, proprio_dim=64, priv_dim=21, act_dim=8,
             scan_deltas=True, temporal_encoder="cnn", scan_stem="resnet")


def test_the_spec_is_idempotent_and_refuses_a_half_specified_channel():
    s = aligned_spec()
    assert aligned_spec(**s) == s
    with pytest.raises(ValueError, match="proprio"):
        scan_channel_spec({"channels": ["aligned"]})
    with pytest.raises(ValueError, match="not the 'aligned' channel"):
        scan_channel_spec({"channels": ["edges"], "aligned": {"proprio": {}}})
    with pytest.raises(ValueError):
        aligned_spec(k=0)
    with pytest.raises(ValueError):
        aligned_spec(tau=-1.0)


@pytest.mark.parametrize("channels", [["aligned"], ["memory", "edges", "aligned"],
                                      ["memory", "edges", "aligned", "aligned_prev",
                                       "aligned_valid"]])
def test_warm_starting_the_aligned_channel_is_bit_identical(tmp_path, channels):
    """The new input columns are zero, so at step 0 the actor is the checkpoint it came from."""
    torch.manual_seed(5)
    base = ActorCritic(**SMALL).eval()
    path = str(tmp_path / "base.pt")
    save_checkpoint(path, base, {"spec": {}})
    spec = ObsSpec(n_beams=SMALL["n_beams"], scan_stack=6, act_dim=8, action_history=2, hist_len=0)
    cfg = {"channels": channels, "aligned": {"proprio": motion_index_spec(spec)}}
    mem, _extra, fresh = load_for_memory(path, "cpu", None, scan_channels=cfg)
    mem.eval()
    assert fresh == [], "extra channels alone add no parameter at all"
    g = torch.Generator().manual_seed(1)
    scan = torch.rand(4, 6, SMALL["n_beams"], generator=g)
    pro = torch.rand(4, SMALL["proprio_dim"], generator=g) * 2 - 1
    priv = torch.rand(4, SMALL["priv_dim"], generator=g) * 2 - 1
    aug = ScanAugment(list(mem.meta["scan_channels"]["channels"]), SMALL["n_beams"], 4,
                      aligned=mem.meta["scan_channels"].get("aligned"))
    pro_for_motion = torch.zeros(4, spec.proprio_dim)
    pro_for_motion[:, 0] = 0.5
    wide = aug(scan, pro_for_motion)
    with torch.no_grad():
        a0, lp0, _ = base.act(scan, pro, deterministic=True)
        a1, lp1, _ = mem.act(wide, pro, deterministic=True)
        v0 = base.critic(scan, pro, priv)
        v1 = mem.critic(wide, pro, priv)
    assert torch.equal(a0, a1) and torch.equal(lp0, lp1)
    assert torch.equal(v0, v1)


def test_the_stem_reads_the_extra_rows_as_channels_and_not_as_frames():
    """`split_channels` peels exactly the rows the meta declares, so the frames keep meaning frames."""
    m = ActorCritic(**SMALL, scan_channels={"channels": ["aligned", "aligned_prev",
                                                         "aligned_valid"],
                                            "aligned": {"proprio": motion_index_spec(
                                                ObsSpec(n_beams=SMALL["n_beams"], scan_stack=6,
                                                        act_dim=8, hist_len=0))}})
    assert m.actor.stem.extra_channels == 3
    frames, extra = m.actor.stem.split_channels(torch.zeros(2, 9, SMALL["n_beams"]))
    assert frames.shape[1] == 6 and extra.shape[1] == 3
    assert m.meta["scan_channels"]["channels"] == ["aligned", "aligned_prev", "aligned_valid"]
