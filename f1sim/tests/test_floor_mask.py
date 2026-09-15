"""The floor channel: its geometry against the simulator, its tolerance band, and its fallbacks.

`learn/floor.py` claims a closed form for where a tilted scan plane meets the floor. The simulator
computes the same intersection by a different route (`Lidar._trace_torch`'s `s_ground`, then
`s * sqrt(1 + k^2)`), so the first test here is the two agreeing on a scan the simulator actually
produced -- not on a re-derivation of the same algebra.
"""
import math

import numpy as np
import pytest
import torch

from f1sim import Track, Config, Simulator
from f1sim.lidar import HIT_GROUND, HIT_NONE, HIT_TALL
from f1sim.learn import clearance as cl
from f1sim.learn import floor as fl

#: Index of the per-beam floor logits in `ActorCritic.evaluate_aux`'s tuple:
#: (logp, entropy, value, dist, grip, opp, future, motion, FLOOR, hidden). It moved from 7
#: to 8 when `feat/motion-memory` merged its own auxiliary in beside this one, which is why
#: it is a name.
FLOOR_LOGITS = 8

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def flat_track(res=0.05, size=24.0, wall_x=8.0, wall_half_y=None):
    """Floor everywhere, one tall wall across the beam at x = wall_x.

    `wall_half_y` makes it a finite block instead of a full-width wall. A full-width wall and the
    floor line are both straight lines across the beam, so whichever is nearer wins at EVERY
    bearing and a scan can never contain both; a block leaves the wider bearings to the floor.
    """
    n = int(size / res)
    xs = (np.arange(n) - n / 2) * res
    gx, gy = np.meshgrid(xs, xs)
    tall = np.abs(gx - wall_x) <= 0.1
    if wall_half_y is not None:
        tall = tall & (np.abs(gy) <= wall_half_y)
    return Track.from_occupancy(tall, res, (xs[0], xs[0]), duct=np.zeros_like(tall), tall=tall)


def make(track, device, **lidar):
    cfg = Config()
    cfg.rand.enabled = False
    cfg.lidar.motion_distortion = False
    cfg.lidar.noise_std = 0.0
    cfg.lidar.noise_std_rel = 0.0
    cfg.lidar.dropout_prob = 0.0
    cfg.lidar.spike_prob = 0.0
    cfg.lidar.floor_dropout = 0.0
    for k, v in lidar.items():
        setattr(cfg.lidar, k, v)
    sim = Simulator(track, cfg, num_envs=1, device=device)
    sim.reset(poses=torch.zeros(1, 3))
    return sim


def scan_at(sim, roll, pitch):
    att = torch.tensor([[roll, pitch]], device=sim.device, dtype=torch.float32)
    r, r_true, typ = sim.lidar.scan(torch.zeros(1, 3, device=sim.device), None, sim.P, False,
                                    noisy=False, att=att)
    return r_true, typ


#: The band the MECHANISM tests are written against. Explicit, not `FloorSpec()`'s default: these
#: tests check that the likelihood does what the geometry says for a given tolerance, and they must
#: not move when the measured attitude error does. `test_the_shipped_band_is_the_measured_one`
#: below is the test that is about the default.
TIGHT = dict(sigma_roll=0.008, sigma_pitch=0.008)


def spec_for(sim, **kw):
    lidar = sim.cfg.lidar
    kw = {**TIGHT, **kw}
    return fl.FloorSpec(mount_x=float(lidar.mount_x), mount_z=float(lidar.mount_z),
                        mount_y=float(lidar.mount_y), **kw).validate()


def angles_of(sim):
    return fl.beam_angles(sim.cfg.lidar.n_beams, sim.cfg.lidar.fov, device=sim.device)


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("roll_deg,pitch_deg", [(0.0, 2.0), (0.0, 3.0), (0.0, 5.0),
                                                (1.5, 2.5), (-2.0, 1.0), (3.0, 4.0)])
def test_floor_range_matches_the_simulator(device, roll_deg, pitch_deg):
    """Every beam the simulator says hit the floor is within a millimetre of the closed form."""
    sim = make(flat_track(wall_x=100.0), device)
    roll, pitch = math.radians(roll_deg), math.radians(pitch_deg)
    r_true, typ = scan_at(sim, roll, pitch)
    spec = spec_for(sim)
    pred = fl.floor_range(angles_of(sim), roll, pitch, spec, rows=1)
    hit = (typ == HIT_GROUND) & (r_true < float(sim.cfg.lidar.range_max) - 1e-3)
    assert int(hit.sum()) > 50, "the tilt should put a wide arc of beams on the floor"
    err = (pred[hit] - r_true[hit]).abs()
    assert float(err.max()) < 1e-3, f"worst floor-range error {float(err.max()):.5f} m"
    # and the beams the simulator did NOT put on the floor are beams the closed form says cannot
    # reach it inside the sensor's range
    miss = (typ == HIT_NONE)
    assert bool((pred[miss] > float(sim.cfg.lidar.range_max)).all())


@pytest.mark.parametrize("device", DEVICES)
def test_return_height_is_zero_on_floor_returns(device):
    """Equation (1) on the simulator's own floor returns reads 0 to a millimetre."""
    sim = make(flat_track(wall_x=100.0), device)
    roll, pitch = math.radians(1.0), math.radians(3.0)
    r_true, typ = scan_at(sim, roll, pitch)
    z = fl.return_height(r_true, angles_of(sim), roll, pitch, spec_for(sim))
    hit = (typ == HIT_GROUND) & (r_true < float(sim.cfg.lidar.range_max) - 1e-3)
    assert float(z[hit].abs().max()) < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_level_scan_has_no_floor_intersection(device):
    sim = make(flat_track(wall_x=100.0), device)
    pred = fl.floor_range(angles_of(sim), 0.0, 0.0, spec_for(sim))
    assert bool(torch.isinf(pred).all()), "a level plane never meets the floor"


def test_quoted_phantom_wall_distances():
    """The numbers the contract quotes, from the closed form with the lever arm removed."""
    spec = fl.FloorSpec(mount_x=0.0, mount_z=0.110).validate()
    ang = torch.zeros(1)
    for deg, want in ((2.0, 3.15), (3.0, 2.10), (5.0, 1.26)):
        r = float(fl.floor_range(ang, 0.0, math.radians(deg), spec)[0, 0])
        assert abs(r - want) < 0.02, f"{deg} deg -> {r:.2f} m, expected {want}"


# ------------------------------------------------------------------ the likelihood
@pytest.mark.parametrize("device", DEVICES)
def test_floor_returns_flagged_and_wall_is_not(device):
    """A tilted scan over a floor with a wall in it: the floor arc is flagged, the wall is not."""
    sim = make(flat_track(wall_x=2.0, wall_half_y=1.0), device, range_max=10.0)
    roll, pitch = 0.0, math.radians(2.0)          # floor line at ~3.1 m, the block at 2 m in front
    r_true, typ = scan_at(sim, roll, pitch)
    p = fl.floor_likelihood(r_true, roll, pitch, spec_for(sim), angles=angles_of(sim),
                            valid=(typ != HIT_NONE))
    spec = spec_for(sim)
    # Scored on the returns the clearance grid can actually hold (x_max 4.44 m). Past ~6 m the
    # sharpness term (4a) deliberately stops claiming anything -- see `floor.py`.
    near = r_true < cl.ClearanceSpec().x_max
    floor = (typ == HIT_GROUND) & near
    solid = (typ == HIT_TALL) & near
    assert int(floor.sum()) > 50 and int(solid.sum()) > 20
    # Every floor return the grid can hold is gated and every solid one survives: the two
    # populations are separated by the threshold, not merely different on average. This is the
    # HARD case for the separation and it is the reason the threshold is what it is: a wall whose
    # return sits 4 cm above the floor at 1.7 m is 1.5 deg of tilt away from the floor line, which
    # is inside the attitude band, and only the `sharp` term and the threshold keep them apart.
    # A quantile, not the minimum: the neighbourhood term mixes the block's beams into the floor
    # arc's outermost ones, so the two or three beams at the boundary sit between the populations.
    # That is the term working, not failing -- it is what stops a narrow object being called floor.
    assert float(torch.quantile(p[floor], 0.10)) > spec.gate_threshold
    assert float((p[floor] >= spec.gate_threshold).float().mean()) > 0.90
    assert float(p[floor].mean()) > 0.80
    assert float(p[solid].max()) < spec.gate_threshold
    assert float(p[floor].mean()) - float(p[solid].mean()) > 0.25


@pytest.mark.parametrize("device", DEVICES)
def test_wall_in_front_of_a_level_sensor_is_not_flagged(device):
    """The contract's explicit case: level scan, wall ahead, nothing may be called floor."""
    sim = make(flat_track(wall_x=3.0), device)
    r_true, typ = scan_at(sim, 0.0, 0.0)
    spec = spec_for(sim)
    p = fl.floor_likelihood(r_true, 0.0, 0.0, spec, angles=angles_of(sim),
                            valid=(typ != HIT_NONE))
    # No beam is gated, anywhere in the 270 deg window, including the ones that run out to 10 m
    # along the wall where a level plane is only 0.11 m above the floor.
    assert float(p.max()) < spec.gate_threshold
    assert float(p[typ == HIT_TALL].mean()) < 0.25


def test_tolerance_band_follows_the_attitude_error():
    """A floor arc stays flagged while the attitude the channel is given is wrong by ~sigma_att,
    and stops being flagged several sigma out -- and the band is the one `sigma_att` names."""
    spec = fl.FloorSpec(**TIGHT).validate()
    ang = fl.beam_angles(1081, 1.5 * math.pi)
    truth = math.radians(3.0)
    r = fl.floor_range(ang, 0.0, truth, spec)
    valid = torch.isfinite(r)
    r = torch.where(valid, r, torch.full_like(r, 10.0))
    mid = slice(500, 582)                                    # the forward sector
    got = []
    for k in (0.0, 1.0, 2.0, 4.0, 8.0):
        p = fl.floor_likelihood(r, 0.0, truth + k * spec.sigma_pitch, spec, angles=ang, valid=valid)
        got.append(float(p[0, mid].mean()))
    assert got[0] > 0.93                                     # exact attitude
    assert got[1] > 0.55                                     # one sigma out: still floor
    assert got[2] > 0.10                                     # two sigma: fading
    assert got[4] < 0.02                                     # eight sigma: gone
    assert got == sorted(got, reverse=True)


def test_unknown_attitude_reads_unknown_not_solid():
    spec = fl.FloorSpec().validate()
    ang = fl.beam_angles(1081, 1.5 * math.pi)
    r = torch.full((2, 1081), 3.0)
    p = fl.floor_likelihood(r, 0.0, math.radians(3.0), spec, angles=ang,
                            att_ok=torch.tensor([True, False]))
    assert float(p[1].min()) == fl.UNKNOWN and float(p[1].max()) == fl.UNKNOWN
    assert float(p[0].max()) > 0.5
    assert fl.UNKNOWN < spec.gate_threshold
    assert float(fl.gate_weight(p[1:], spec).min()) == 1.0, "an unknown attitude gates nothing"


def test_no_return_beams_read_zero():
    spec = fl.FloorSpec().validate()
    ang = fl.beam_angles(64, 1.5 * math.pi)
    scan = torch.full((1, 64), 1.0)                          # every beam a no-return
    p = fl.floor_likelihood_norm(scan, 0.0, math.radians(3.0), 10.0, spec, angles=ang)
    assert float(p.max()) == 0.0


def test_neighbourhood_term_discounts_a_lone_coincidence():
    """One beam that happens to sit on the floor plane scores below a whole arc that does."""
    spec = fl.FloorSpec(**TIGHT).validate()
    ang = fl.beam_angles(1081, 1.5 * math.pi)
    pitch = math.radians(3.0)
    rf = fl.floor_range(ang, 0.0, pitch, spec)
    valid = torch.isfinite(rf)
    arc = torch.where(valid, rf, torch.full_like(rf, 9.0))   # the whole floor arc
    lone = torch.full_like(arc, 9.0)                          # a far wall everywhere ...
    lone[0, 540] = arc[0, 540]                                # ... with one beam on the plane
    p_arc = fl.floor_likelihood(arc, 0.0, pitch, spec, angles=ang, valid=torch.ones_like(valid))
    p_lone = fl.floor_likelihood(lone, 0.0, pitch, spec, angles=ang, valid=torch.ones_like(valid))
    assert float(p_arc[0, 540]) > 0.9
    assert float(p_lone[0, 540]) < 0.5
    # and with the neighbourhood term switched off the two are the same number
    flat = fl.FloorSpec(smooth_weight=0.0, **TIGHT).validate()
    a0 = fl.floor_likelihood(arc, 0.0, pitch, flat, angles=ang, valid=torch.ones_like(valid))
    l0 = fl.floor_likelihood(lone, 0.0, pitch, flat, angles=ang, valid=torch.ones_like(valid))
    assert abs(float(a0[0, 540]) - float(l0[0, 540])) < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_precision_recall_on_a_simulated_scan(device):
    """Against `scan_type == HIT_GROUND`, with the sensor noise model on, over the returns the
    clearance grid can hold (inside `ClearanceSpec.x_max`)."""
    sim = make(flat_track(wall_x=5.0), device)
    sim.cfg.lidar.noise_std = 0.0074
    sim.cfg.lidar.noise_std_rel = 0.0010
    tot = {"tp": 0, "fp": 0, "fn": 0}
    spec = spec_for(sim)
    ang = angles_of(sim)
    for pitch_deg in (1.0, 2.0, 3.0, 4.0, 5.0):
        for roll_deg in (-2.0, 0.0, 2.0):
            roll, pitch = math.radians(roll_deg), math.radians(pitch_deg)
            att = torch.tensor([[roll, pitch]], device=sim.device)
            r, _r_true, typ = sim.lidar.scan(torch.zeros(1, 3, device=sim.device), None, sim.P,
                                             False, noisy=True, att=att)
            valid = (typ != HIT_NONE) & (r < float(sim.cfg.lidar.range_max) - 1e-3)
            p = fl.floor_likelihood(r, roll, pitch, spec, angles=ang, valid=valid)
            scored = valid & (r < cl.ClearanceSpec().x_max)      # what the grid can hold
            pred = (p >= spec.gate_threshold) & scored
            truth = (typ == HIT_GROUND) & scored
            tot["tp"] += int((pred & truth).sum())
            tot["fp"] += int((pred & ~truth).sum())
            tot["fn"] += int((~pred & truth).sum())
    prec = tot["tp"] / max(1, tot["tp"] + tot["fp"])
    rec = tot["tp"] / max(1, tot["tp"] + tot["fn"])
    assert prec > 0.95, f"precision {prec:.3f} ({tot})"
    assert rec > 0.90, f"recall {rec:.3f} ({tot})"


# ------------------------------------------------------------------ the clearance gate
def _grid_inputs(device="cpu"):
    """A scan with a floor arc under 3 deg of nose-down and a wall at 1.2 m, and its geometry."""
    sim = make(flat_track(wall_x=1.2, wall_half_y=0.6), device)
    roll, pitch = 0.0, math.radians(3.0)
    r_true, typ = scan_at(sim, roll, pitch)
    rmax = float(sim.cfg.lidar.range_max)
    scan = torch.where(typ == HIT_NONE, torch.ones_like(r_true), (r_true / rmax).clamp(0, 1))
    return sim, scan, typ, roll, pitch, rmax


@pytest.mark.parametrize("device", DEVICES)
def test_gate_off_is_bit_identical_occupancy(device):
    """`clearance_floor_gate` off: the grid is exactly the grid, and the likelihood is not even
    consulted -- passing a likelihood that would gate every beam changes nothing."""
    sim, scan, _typ, roll, pitch, rmax = _grid_inputs(device)
    ang = angles_of(sim)
    off = cl.ClearanceSpec().validate()
    base = cl.occupancy(scan, ang, off, rmax, float(sim.cfg.lidar.mount_x))
    everything = torch.ones_like(scan)
    assert torch.equal(base, cl.occupancy(scan, ang, off, rmax, float(sim.cfg.lidar.mount_x),
                                          0.0, p_floor=everything, fspec=spec_for(sim)))
    assert torch.equal(cl.distance_field(base, off),
                       cl.distance_field(cl.occupancy(scan, ang, off, rmax,
                                                      float(sim.cfg.lidar.mount_x)), off))


@pytest.mark.parametrize("device", DEVICES)
def test_gate_on_removes_the_floor_and_keeps_the_wall(device):
    """On: the floor arc leaves the occupancy and the wall at 1.2 m stays in it."""
    sim, scan, typ, roll, pitch, rmax = _grid_inputs(device)
    ang = angles_of(sim)
    on = cl.ClearanceSpec(floor_gate=True).validate()
    fspec = spec_for(sim)
    p = fl.floor_likelihood_norm(scan, roll, pitch, rmax, fspec, angles=ang,
                                 range_eps=on.range_eps)
    occ_all = cl.occupancy(scan, ang, cl.ClearanceSpec(), rmax, float(sim.cfg.lidar.mount_x))
    occ_gate = cl.occupancy(scan, ang, on, rmax, float(sim.cfg.lidar.mount_x), 0.0,
                            p_floor=p, fspec=fspec)
    # the privileged ideal: drop exactly the beams the simulator says are floor
    scan_solid = torch.where(typ == HIT_GROUND, torch.ones_like(scan), scan)
    occ_solid = cl.occupancy(scan_solid, ang, cl.ClearanceSpec(), rmax,
                             float(sim.cfg.lidar.mount_x))
    assert float(occ_all.sum()) > float(occ_gate.sum()), "the gate has to remove something"
    # the gated grid is much closer to the solid-only grid than the ungated one is
    d_gate = float((occ_gate - occ_solid).abs().sum())
    d_all = float((occ_all - occ_solid).abs().sum())
    assert d_gate < 0.35 * d_all, f"gated {d_gate} vs ungated {d_all} cells away from the truth"
    # and the wall's own cells survive: every cell the solid-only grid occupies is still occupied
    assert float((occ_solid * (1 - occ_gate)).sum()) == 0.0


def test_gate_refuses_to_run_without_a_likelihood():
    on = cl.ClearanceSpec(floor_gate=True).validate()
    with pytest.raises(ValueError, match="no floor likelihood"):
        cl.occupancy(torch.rand(1, 16) * 0.5, fl.beam_angles(16, 1.5 * math.pi), on, 10.0)


def test_unknown_attitude_gates_nothing():
    """The deployment case the contract names: the estimate is not usable, so the arm is exactly
    the arm it is today rather than one that has forgotten the walls."""
    on = cl.ClearanceSpec(floor_gate=True).validate()
    ang = fl.beam_angles(256, 1.5 * math.pi)
    scan = torch.full((2, 256), 0.2)
    p = fl.floor_likelihood_norm(scan, 0.0, math.radians(4.0), 10.0, angles=ang,
                                 att_ok=torch.tensor([True, False]))
    occ_on = cl.occupancy(scan, ang, on, 10.0, 0.297, 0.0, p_floor=p, fspec=fl.FloorSpec())
    occ_off = cl.occupancy(scan, ang, cl.ClearanceSpec(), 10.0, 0.297, 0.0)
    assert torch.equal(occ_on[1], occ_off[1]), "the untrusted row must be untouched"


# ------------------------------------------------------------------ the auxiliary head
def _ac(**kw):
    from f1sim.learn.model import ActorCritic
    torch.manual_seed(7)
    return ActorCritic(6, 128, 16, 8, act_dim=8, scan_deltas=True, scan_stem="resnet", **kw)


def test_aux_head_off_is_byte_identical():
    """The head adds tensors and changes nothing else: every shared weight is the same number, and
    the forward with the head attached is bit-identical to the forward without it."""
    a, b = _ac(), _ac(floor_head={"width": 16})
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert set(sb) - set(sa) == {k for k in sb if k.startswith("actor.floor.")}
    assert not a.has_floor_head and b.has_floor_head
    scan, pro, priv, act = (torch.rand(3, 6, 128), torch.rand(3, 16), torch.rand(3, 8),
                            torch.rand(3, 8))
    a.eval(); b.eval()
    with torch.no_grad():
        o1, o2 = a.evaluate_aux(scan, pro, priv, act), b.evaluate_aux(scan, pro, priv, act,
                                                                     floor=True)
    for i, name in enumerate(("logp", "entropy", "value")):
        assert torch.equal(o1[i], o2[i]), name
    fl = FLOOR_LOGITS
    assert o1[fl] is None and o2[fl].shape == (3, 128)
    assert float(o2[fl].abs().max()) == 0.0, "a zero output layer predicts 0.5 for every beam"
    assert b.evaluate_aux(scan, pro, priv, act)[fl] is None, "not asked for, not computed"


def test_aux_head_refuses_the_plain_stem():
    from f1sim.learn.model import ActorCritic
    with pytest.raises(ValueError, match="resnet"):
        ActorCritic(6, 128, 16, 8, act_dim=8, scan_stem="plain", floor_head={"width": 16})


def test_aux_loss_reports_what_a_collapsed_head_looks_like():
    """The motion-memory lesson: a head that predicts "never" has a falling loss and zero recall,
    and the loss has to say so."""
    from f1sim.learn.floor_head import floor_loss
    label = (torch.rand(8, 200) < 0.03).float()
    valid = (torch.rand(8, 200) < 0.95).float()
    never = torch.full((8, 200), -20.0)
    _l, parts = floor_loss(never, label, valid, torch.ones(8))
    assert float(parts["recall"]) == 0.0 and float(parts["precision"]) == 0.0
    perfect = torch.where(label > 0, 20.0, -20.0)
    _l2, p2 = floor_loss(perfect, label, valid, torch.ones(8))
    assert float(p2["recall"]) > 0.99 and float(p2["precision"]) > 0.99
    # the positive weight is the batch's own reciprocal rate, and a binding clamp is reported
    assert float(parts["pos_weight"]) > 20.0
    assert float(parts["pos_weight_clamped"]) == 0.0
    _l3, p3 = floor_loss(never, label, valid, torch.ones(8), pos_weight_max=2.0)
    assert float(p3["pos_weight_clamped"]) == 1.0
    # a beam with no return is not scored at all
    lab_all = torch.ones(2, 10)
    _l4, p4 = floor_loss(torch.full((2, 10), 20.0), lab_all, torch.zeros(2, 10), torch.ones(2))
    assert float(p4["recall"]) == 0.0 and float(p4["rate"]) == 0.0


def test_aux_head_can_learn_the_label_it_is_given():
    """A head whose loss cannot go down is untestable in a smoke; this one is fit on a tiny fixed
    batch, so "the plumbing works" is a measurement rather than an absence of exceptions."""
    from f1sim.learn.floor_head import FloorHead, floor_loss
    torch.manual_seed(3)
    ctx, rows = torch.randn(6, 32, 12), torch.randn(6, 5, 384)
    label = (rows[:, 0] > 1.2).float()                       # a function of the input rows
    valid = torch.ones_like(label)
    head = FloorHead(32, 5, width=24)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss, parts = floor_loss(head(ctx, rows), label, valid)
        first = first if first is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.5 * first
    assert float(parts["recall"]) > 0.8 and float(parts["precision"]) > 0.8


@pytest.mark.parametrize("device", DEVICES)
def test_the_shipped_band_is_the_measured_one_and_it_is_too_wide(device):
    """The claim the research note rests on, as executable code.

    `SIGMA_ROLL` / `SIGMA_PITCH` are what `floor.AttitudeTracker` measured on the proxy tracks, and
    at that band the same scene the tight-band test separates cleanly is **not** separated: the
    wall's returns reach the gate threshold. That is why `ClearanceSpec.floor_gate` is off by
    default and why `docs/research/floor-mask-2026-09-15.md` says the geometry is waiting on a
    better attitude rather than on a better decision rule.

    If a future attitude source makes this test fail, that is the good failure: re-measure
    `SIGMA_*`, and this test becomes the one that says the gate is ready.
    """
    assert (fl.SIGMA_ROLL, fl.SIGMA_PITCH) == (0.022, 0.022), \
        "the shipped band is a measurement; changing it means re-running work/measure/tune_attitude.py"
    # And it is at the floor two unobservable terms set: the road-tilt process and the LiDAR's own
    # mounting offset. If this stops holding, the band was not re-derived from the measurement.
    floor_rms = math.hypot(fl.ROAD_TILT_RMS, 0.02 / math.sqrt(3.0))
    assert abs(fl.SIGMA_ROLL - floor_rms) < 0.004, (fl.SIGMA_ROLL, floor_rms)
    sim = make(flat_track(wall_x=2.0, wall_half_y=1.0), device, range_max=10.0)
    roll, pitch = 0.0, math.radians(2.0)
    r_true, typ = scan_at(sim, roll, pitch)
    lidar = sim.cfg.lidar
    shipped = fl.FloorSpec(mount_x=float(lidar.mount_x), mount_z=float(lidar.mount_z)).validate()
    p = fl.floor_likelihood(r_true, roll, pitch, shipped, angles=angles_of(sim),
                            valid=(typ != HIT_NONE))
    near = r_true < cl.ClearanceSpec().x_max
    solid = (typ == HIT_TALL) & near
    assert float(p[solid].max()) >= shipped.gate_threshold, \
        "the measured band no longer confuses this wall with the floor -- re-read the note"


# ------------------------------------------------------------------ roll is not a special case
@pytest.mark.parametrize("device", DEVICES)
def test_pure_roll_puts_beams_on_the_floor_and_the_pitch_only_ring_does_not_predict_it(device):
    """A finding from the deck worker (2026-09-15), pinned here: over a 96-car sweep, floor hits
    track the TOTAL body tilt and correlate more with **roll** (0.31) than with pitch (0.18), the
    deepest nose-up frame had none while a small nose-down one had 93, and the naive
    `mount_z / tan(pitch)` ring does not predict the observed ranges.

    All three follow from the plane geometry and none of them from a pitch-only formula, because
    the beam's vertical direction cosine is

        bz = -cos(a) sin(pitch) + sin(a) sin(roll) cos(pitch)

    and over a 270 deg window `|sin a|` exceeds `|cos a|` for more than half the beams. This test
    is the guard that the implementation stays joint: with **zero pitch**, roll alone must put a
    wide arc on the floor at ranges the closed form predicts and the pitch-only ring cannot.
    """
    sim = make(flat_track(wall_x=100.0), device)
    roll, pitch = math.radians(3.0), 0.0
    r_true, typ = scan_at(sim, roll, pitch)
    hit = (typ == HIT_GROUND) & (r_true < float(sim.cfg.lidar.range_max) - 1e-3)
    assert int(hit.sum()) > 50, "pure roll has to produce floor returns"
    spec = spec_for(sim)
    pred = fl.floor_range(angles_of(sim), roll, pitch, spec, rows=1)
    assert float((pred[hit] - r_true[hit]).abs().max()) < 1e-3
    # the pitch-only ring is undefined here (pitch = 0 -> infinite range) and so cannot be right
    assert not math.isfinite(spec.mount_z / math.tan(pitch)) if pitch else True
    # and the beams it puts on the floor are the WIDE ones, not the forward ones: with no pitch,
    # `bz = sin(a) sin(roll)`, so a beam reaches the floor inside `range_max` only where
    # `|sin a| >= oz / (range_max sin roll)`. That bound is the geometry's, not a guess.
    ang = angles_of(sim)
    oz = spec.mount_z * math.cos(roll)
    a_min = math.asin(min(1.0, oz / (float(sim.cfg.lidar.range_max) * math.sin(roll))))
    assert float(ang[hit[0]].abs().min()) > 0.95 * a_min
    assert a_min > math.radians(10.0)
    assert float(ang[hit[0]].abs().median()) > math.radians(45.0), \
        "roll tilts the plane about the forward axis, so it is the wide beams that descend"

    # Joint case: a nose-UP pitch with roll still produces floor hits, which a pitch-only model
    # reads as "impossible". This is the deck worker's -1.87 deg frame.
    r2, typ2 = scan_at(sim, math.radians(3.0), math.radians(-1.87))
    hit2 = (typ2 == HIT_GROUND) & (r2 < float(sim.cfg.lidar.range_max) - 1e-3)
    pred2 = fl.floor_range(ang, math.radians(3.0), math.radians(-1.87), spec, rows=1)
    assert int(hit2.sum()) > 10
    assert float((pred2[hit2] - r2[hit2]).abs().max()) < 1e-3
    # ... and with no roll, that same nose-up attitude still produces floor returns -- but they
    # are all BEHIND the car. The scan plane is a plane: tilt it any way and half of it descends.
    # Nose-up raises the forward beams and lowers the ones past +-90 deg, which the 270 deg window
    # has. A pitch-only ring model is therefore wrong in both directions, which is the deck
    # worker's point restated.
    r3, typ3 = scan_at(sim, 0.0, math.radians(-1.87))
    hit3 = (typ3 == HIT_GROUND) & (r3 < float(sim.cfg.lidar.range_max) - 1e-3)
    assert int(hit3.sum()) > 10
    assert float(ang[hit3[0]].abs().min()) > math.radians(90.0), \
        "with the nose up and no roll, only the rearward beams can reach the floor"
    pred3 = fl.floor_range(ang, 0.0, math.radians(-1.87), spec, rows=1)
    assert float((pred3[hit3] - r3[hit3]).abs().max()) < 1e-3
    # and none of them is inside the clearance grid, so this case moves no plan
    x = (r3 * torch.cos(ang)[None])[hit3]
    assert float(x.max()) < cl.ClearanceSpec().x_max


# ------------------------------------------------------------------ attitude from the ego state
def test_ego_state_attitude_reproduces_the_calibrated_gains():
    """Steady state: the estimator must return exactly what §6.1a's gains say, and the pitch map
    must be the asymmetric one -- squat 1.7 deg/g under throttle, dive 0.46 under braking."""
    e = fl.EgoStateAttitude(3, dt=0.025)
    v = torch.tensor([8.0, 8.0, 8.0])
    for _ in range(80):
        v = v + torch.tensor([-0.1, +0.1, 0.0])              # -4, +4, 0 m/s^2
        att = e.update(v, torch.tensor([0.0, 0.0, 1.0]))
    assert abs(float(att[0, 1]) - fl.DIVE_PER_G * 4.0 / fl.G_ACC) < 1e-4     # braking: dive
    assert abs(float(att[1, 1]) + fl.SQUAT_PER_G * 4.0 / fl.G_ACC) < 1e-4    # throttle: squat
    assert abs(float(att[0, 1])) < abs(float(att[1, 1])), "the map is asymmetric"
    # cornering: a_y = v * omega_z, and the roll is to the OUTSIDE (the simulator's sign)
    assert abs(float(att[2, 0]) - fl.ROLL_PER_G * float(v[2]) * 1.0 / fl.G_ACC) < 1e-3
    assert float(att[2, 1]) == pytest.approx(0.0, abs=1e-4)


def test_ego_state_attitude_holds_through_a_wheel_lock():
    """This car locks its wheels: §2.12 measures -40 to -143 m/s^2 of WHEEL deceleration against
    -3 to -16 of body. Differentiating that raw is 25 degrees of phantom dive."""
    e = fl.EgoStateAttitude(1, dt=0.025)
    v = torch.tensor([8.0])
    for _ in range(60):
        v = v - torch.tensor([0.1])
        e.update(v, torch.zeros(1))
    steady = float(e.ax)
    assert abs(steady + 4.0) < 0.05
    seen = []
    for _ in range(6):                                        # 100 m/s^2 of wheel deceleration
        v = (v - torch.tensor([2.5])).clamp_min(0.0)
        att = e.update(v, torch.zeros(1))
        seen.append(float(e.ax))
    # The guard holds through the spike and resumes when the derivative is plausible again -- which
    # it should, because by then the wheel speed really has settled. What must never happen is the
    # estimator adopting a wheel deceleration the body cannot produce.
    assert max(abs(x) for x in seen) <= e.A_BODY_MAX
    assert abs(math.degrees(float(att[0, 1]))) < 1.0, "no phantom dive from a lock"
    # and without the guard it would: the same spike, believed, is tens of g
    raw = 2.5 / 0.025
    assert raw > 5 * e.A_BODY_MAX


def test_ego_state_accel_source_inverts_the_gravity_leak():
    """The specific force already contains `g sin(tilt)` and the tilt is what is being estimated,
    so `f = a (1 + k)` closes -- the same closure `real_data_calibration.md` §6.1a uses."""
    for src in ("accel", "blend"):
        e = fl.EgoStateAttitude(1, dt=0.025, source=src)
        a_true = 6.0
        f_y = a_true * (1.0 + fl.ROLL_PER_G)                  # what the sensor would read
        for _ in range(120):
            att = e.update(torch.tensor([6.0]), torch.tensor([1.0]),
                           torch.tensor([[0.0, f_y, fl.G_ACC]]))
        assert abs(float(att[0, 0]) - fl.ROLL_PER_G * a_true / fl.G_ACC) < 2e-4, src
    with pytest.raises(ValueError, match="needs accel"):
        fl.EgoStateAttitude(1, source="accel").update(torch.zeros(1), torch.zeros(1))


def test_ego_state_attitude_clears_per_episode():
    e = fl.EgoStateAttitude(2, dt=0.025)
    for _ in range(40):
        e.update(torch.tensor([6.0, 6.0]), torch.tensor([1.5, 1.5]))
    assert float(e.att[0, 0]) > 0.01 and float(e.att[1, 0]) > 0.01
    e.reset(torch.tensor([True, False]))
    assert float(e.att[0, 0]) == 0.0 and float(e.att[1, 0]) > 0.01


# ------------------------------------------------------------------ the learned front-end
def _frontend(tmp_path, width=20):
    from f1sim.learn.frontend import (FrontEnd, frontend_spec, imu_index_spec, save_frontend)
    from f1sim.learn.obs import ObsSpec
    sp = ObsSpec(n_beams=256, scan_stack=6, act_dim=8, hist_len=20)
    idx = imu_index_spec(sp)
    spec = frontend_spec(width=width, n_beams=256, imu_dim=idx["dim"])
    m = FrontEnd(6, spec)
    path = str(tmp_path / "fe.pt")
    save_frontend(path, m, spec, idx, 6)
    return path, m, spec, idx, sp


def test_frontend_at_init_is_the_uniform_prior_and_the_identity(tmp_path):
    """Zero output layers, the same discipline every other addition here keeps: an untrained
    front-end classifies at 1/3 and denoises to exactly the input."""
    from f1sim.learn.frontend import imu_vector, n_params
    _p, m, spec, idx, _sp = _frontend(tmp_path)
    m.eval()
    scan = torch.rand(3, 6, 256)
    pro = torch.zeros(3, int(idx["proprio_dim"]))
    ego = torch.zeros(3, int(idx["ego_dim"]))
    with torch.no_grad():
        logits, rng, att = m(scan, imu_vector(pro, idx, ego))
    from f1sim.learn.frontend import CLASSES
    assert float(logits.abs().max()) == 0.0
    assert logits.shape[1] == len(CLASSES)
    assert torch.allclose(torch.softmax(logits, 1),
                          torch.full_like(logits, 1.0 / len(CLASSES)))
    assert torch.equal(rng, scan[:, 0])
    assert float(att.abs().max()) == 0.0
    assert n_params(m) <= 150_000, "the contract's parameter budget"


def test_frontend_cannot_move_a_return_further_than_its_span(tmp_path):
    """The denoiser is a bounded residual on the newest frame: a wall cannot be denoised into open
    space however confident the network is."""
    from f1sim.learn.frontend import imu_vector
    _p, m, _spec, idx, _sp = _frontend(tmp_path)
    with torch.no_grad():
        m.head.bias.fill_(50.0)                              # maximally confident, in both signs
        scan = torch.full((2, 6, 256), 0.2)
        rng = m(scan, imu_vector(torch.zeros(2, int(idx["proprio_dim"])), idx,
                                 torch.zeros(2, int(idx["ego_dim"]))))[1]
    assert float((rng - scan[:, 0]).abs().max()) <= m.RANGE_SPAN + 1e-6


def test_frontend_round_trips_with_its_column_map(tmp_path):
    from f1sim.learn.frontend import load_frontend, imu_vector
    path, m, spec, idx, _sp = _frontend(tmp_path)
    m2, spec2, idx2, k2 = load_frontend(path)
    assert spec2 == spec and idx2 == idx and k2 == 6
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    assert not any(p.requires_grad for p in m2.parameters())
    # a proprio vector of the wrong width is an error at the first forward, not a plausible number
    with pytest.raises(ValueError, match="was built for"):
        imu_vector(torch.zeros(1, 7), idx, torch.zeros(1, int(idx["ego_dim"])))
    with pytest.raises(ValueError, match="ego-state columns"):
        imu_vector(torch.zeros(1, int(idx["proprio_dim"])), idx)


def test_frontend_channels_are_zero_init_parity_and_refuse_without_one(tmp_path):
    """`fe_floor` / `fe_range` as scan channels: at init the class row is exactly 1/3 and the range
    row is bit-identical to the raw frame, so the columns a warm start zeroes carry nothing."""
    from f1sim.learn.obs import ScanAugment, att_index_spec
    path, _m, _spec, _idx, sp = _frontend(tmp_path)
    cfg = {"proprio": att_index_spec(sp), "spec": {}, "fov": 1.5 * math.pi,
           "att_source": "ego", "frontend": {"path": path}}
    aug = ScanAugment(("fe_floor", "fe_range"), 256, 2, floor=cfg)
    scan = torch.rand(2, 6, 256) * 0.5
    pro = torch.zeros(2, sp.proprio_dim)
    pro[:, att_index_spec(sp)["accel"] + 2] = 9.81 / 10.0
    out = aug(scan, pro)
    assert out.shape == (2, 8, 256)
    # The contract's rule, asserted: the front-end's rows are appended BESIDE the raw stack and
    # never in place of it, so a hallucinated clean range cannot hide a real wall. The six frames
    # the actor reads are bit-identical to the six it was given.
    assert torch.equal(out[:, :6], scan)
    from f1sim.learn.frontend import CLASSES
    assert torch.allclose(out[:, -2], torch.full_like(out[:, -2], 1.0 / len(CLASSES)), atol=1e-6)
    assert torch.allclose(out[:, -1], scan[:, 0], atol=1e-6)
    assert torch.allclose(out, aug.preview(scan, None, pro), atol=1e-6)
    with pytest.raises(ValueError, match="trained front-end"):
        ScanAugment(("fe_floor",), 256, 2, floor={**cfg, "frontend": None})


# ------------------------------------------------------------------ what reaches the car
def test_the_aux_head_cannot_reach_the_exported_graph():
    """`--aux-floor` is train-time only. The export traces `Actor.step`, which takes the
    `floor=False` branch, so the head is not merely absent from the output -- it is never called.

    Asserted by poisoning it: give the head weights that would dominate anything they touched and
    check the exported path's action is **bit-identical**. That is a stronger statement than
    "the ONNX file has no such node", and it needs no exporter.
    """
    from f1sim.learn.export import ActorOnly
    m = _ac(floor_head={"width": 16})
    m.eval()
    scan, pro = torch.rand(3, 6, 128), torch.rand(3, 16)
    with torch.no_grad():
        before = ActorOnly(m.actor)(scan, pro)
        for p in m.actor.floor.parameters():
            p.mul_(0.0).add_(7.0)
        after = ActorOnly(m.actor)(scan, pro)
        assert torch.equal(before, after)
        # ... and the head still produces something, so the test is not vacuous
        assert float(m.evaluate_aux(scan, pro, torch.rand(3, 8), torch.rand(3, 8),
                                    floor=True)[FLOOR_LOGITS].abs().max()) > 0.0


def test_the_floor_channel_warm_start_is_bit_identical(tmp_path):
    """The channel is one more input column, appended after every column the original had and
    zeroed, so the first action of a run that turns it on is the action the original would have
    produced -- for any value in the new column."""
    from f1sim.learn.model import load_for_memory, save_checkpoint
    from f1sim.learn.obs import ObsSpec, att_index_spec
    torch.manual_seed(5)
    base = _ac()
    base.eval()
    path = str(tmp_path / "base.pt")
    save_checkpoint(path, base, {"spec": {}})
    sp = ObsSpec(n_beams=128, scan_stack=6, act_dim=8, hist_len=0)
    chan = {"channels": ["floor"], "floor": {"proprio": att_index_spec(sp)}}
    m, _extra, fresh = load_for_memory(path, "cpu", None, scan_channels=chan)
    m.eval()
    assert fresh == [], "a channel adds no tensors, only columns"
    scan, pro = torch.rand(4, 6, 128), torch.rand(4, 16)
    with torch.no_grad():
        a0 = base.act(scan, pro, deterministic=True)[0]
        for fill in (0.0, 0.5, 1.0, 7.3):
            aug = torch.cat([scan, torch.full((4, 1, 128), fill)], 1)
            assert torch.equal(a0, m.act(aug, pro, deterministic=True)[0]), fill


def test_beam_labels_separate_the_two_kinds_of_no_return():
    """Root's point, as a label: a floor hit that dropped out is not the same thing as a beam that
    saw nothing, and only the first says the bearing is clear out to the floor."""
    from f1sim.learn.frontend import (CLASSES, FLOOR, NONE_FLOOR, NONE_OTHER, SOLID, beam_labels)
    typ = torch.tensor([[3, 3, 2, 2, 0, 4]])            # ground, ground, tall, tall, none, car
    ret = torch.tensor([[True, False, True, False, False, True]])
    lab = beam_labels(typ, ret)
    assert lab.tolist() == [[FLOOR, NONE_FLOOR, SOLID, NONE_OTHER, NONE_OTHER, SOLID]]
    assert len(CLASSES) == 4 and CLASSES[NONE_FLOOR] == "none_floor"
    # and the two no-return classes are the ones the range loss must skip
    from f1sim.learn.frontend import frontend_losses
    logits = torch.zeros(1, 4, 6)
    rng = torch.full((1, 6), 0.9)
    clean = torch.zeros(1, 6)                            # a huge range error, on every beam
    loss, parts = frontend_losses(logits, rng, torch.zeros(1, 2), lab, clean, torch.zeros(1, 2))
    # only the three beams that returned are scored, so the L1 is 0.9 and not something smaller
    assert abs(float(parts["range_l1"]) - 0.9) < 1e-5


# ------------------------------------------------------------------ the brake-only gate
def _adjust_inputs(device="cpu", B=3):
    """A plan and two occupancies: one with an obstacle, one without."""
    from f1sim import mpc as _mpc
    cspec = cl.ClearanceSpec().validate()
    ang = fl.beam_angles(361, 1.5 * math.pi, device=device)
    scan_all = torch.full((B, 361), 0.9, device=device)
    scan_all[:, 150:210] = 0.20                     # something 2 m ahead, across the path
    scan_none = torch.full((B, 361), 0.9, device=device)
    f = lambda sc: cl.distance_field(
        cl.occupancy(sc, ang, cspec, 10.0, 0.297, 0.0), cspec)
    action = torch.zeros(B, 8, device=device)
    action[:, 6] = action[:, 7] = 0.2               # ask for ~6 m/s at both knots
    v = torch.full((B,), 5.0, device=device)
    cap = torch.full((B,), 9.0, device=device)
    return action, v, cap, f(scan_all), f(scan_none), _mpc.PlanSpec(), cspec


@pytest.mark.parametrize("device", DEVICES)
def test_brake_only_gate_splits_the_two_decisions(device):
    """`dist_speed` lets the speed cap read a gated occupancy while the bend still reads every
    return. The test is that each decision follows its own field and neither follows the other's."""
    action, v, cap, d_all, d_none, spec, cspec = _adjust_inputs(device)
    both_on = cl.adjust(action, v, cap, d_all, spec, cspec, 10.0)          # one field, obstacle
    both_off = cl.adjust(action, v, cap, d_none, spec, cspec, 10.0)        # one field, clear
    assert float(both_on.dv.min()) < -cl.V_EPS, "the obstacle has to slow the plan"
    assert float(both_off.dv.min()) >= -cl.V_EPS, "an empty world must not"

    # bend on the full occupancy, speed on the gated (empty) one: the bend is the obstacle's, the
    # speed is the empty world's.
    split = cl.adjust(action, v, cap, d_all, spec, cspec, 10.0, dist_speed=d_none)
    assert torch.equal(split.dk, both_on.dk), "the bend must still see the obstacle"
    assert float(split.dv.min()) >= -cl.V_EPS, "the speed must not"
    # and the mirror image, to show neither decision is silently taking the other's field
    split2 = cl.adjust(action, v, cap, d_none, spec, cspec, 10.0, dist_speed=d_all)
    assert torch.equal(split2.dk, both_off.dk)
    assert float(split2.dv.min()) < -cl.V_EPS


def test_brake_only_is_the_default_mode_and_the_gate_is_still_off():
    """Brake-only ships as the gate's shape (it measured better than gating both decisions on every
    axis, §3.7), but the gate itself is still OFF -- so nothing about the default arm changed."""
    assert cl.ClearanceSpec().floor_gate_mode == "brake"
    assert cl.ClearanceSpec().floor_gate is False
    cl.ClearanceSpec(floor_gate_mode="both").validate()
    with pytest.raises(ValueError, match="floor_gate_mode"):
        cl.ClearanceSpec(floor_gate_mode="bend").validate()


# ------------------------------------------------------------ where the likelihood comes from


def test_gate_threshold_bound_follows_the_likelihood_source():
    """0.5 is `UNKNOWN`, the geometric channel's "no attitude" value, so a geometric gate must
    stay strictly above it. The front-end's 0.5 is a split vote and carries no such sentinel, so
    the same threshold is legal there -- and the report's sweep needs it."""
    import dataclasses
    assert fl.FloorSpec().gate_source == "geometric"
    with pytest.raises(ValueError, match="UNKNOWN"):
        dataclasses.replace(fl.FloorSpec(), gate_threshold=fl.UNKNOWN).validate()
    ext = dataclasses.replace(fl.FloorSpec(), gate_source="external",
                              gate_threshold=fl.UNKNOWN).validate()
    assert ext.gate_threshold == fl.UNKNOWN
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        dataclasses.replace(fl.FloorSpec(), gate_source="external", gate_threshold=0.0).validate()
    with pytest.raises(ValueError, match="gate_source"):
        dataclasses.replace(fl.FloorSpec(), gate_source="frontend").validate()


def test_an_external_gate_with_nothing_supplied_gates_nothing():
    """The failure this guards: an external gate may sit at 0.5, and the geometric likelihood's
    unknown value IS 0.5, so falling back to it on a step the front-end did not run would gate
    every beam -- the arm would forget the walls exactly when it had least reason to. Checked in
    both gate modes, because the shipped one gates only the speed field."""
    import dataclasses
    from f1sim import mpc as _mpc
    ang = fl.beam_angles(361, 1.5 * math.pi)
    scan = torch.full((1, 361), 0.9)
    scan[:, 150:210] = 0.20                          # a wall the arm must not forget
    ext = dataclasses.replace(fl.FloorSpec(), gate_source="external",
                              gate_threshold=fl.UNKNOWN).validate()

    def build(cspec, fspec):
        tr = _mpc.PlanTracker(1, "cpu", 0.32, 0.4, 10.0, compile_solver=False)
        arm = cl.ClearanceArm(tr, cspec, 1, "cpu", 10.0, ang, 10.0, fspec=fspec)
        arm.update_scan(scan)
        return arm

    off = build(cl.ClearanceSpec().validate(), None)                  # the arm as it ships
    bend_off, speed_off = off.field()
    assert speed_off is None, "gate off must not build a second field"

    for mode in ("both", "brake"):
        on = build(cl.ClearanceSpec(floor_gate=True, floor_gate_mode=mode).validate(), ext)
        bend_on, speed_on = on.field()
        assert torch.equal(bend_on, bend_off), f"{mode}: the bend must be untouched"
        if mode == "brake":
            assert torch.equal(speed_on, bend_off), "brake: the speed field must be untouched too"
        else:
            assert speed_on is None

    # and when one IS supplied at that threshold it still acts, so the no-op above is the
    # fallback and not the gate being dead -- in each mode, on the field that mode gates.
    for mode, idx in (("both", 0), ("brake", 1)):
        on2 = build(cl.ClearanceSpec(floor_gate=True, floor_gate_mode=mode).validate(), ext)
        on2.set_floor(torch.full((1, 361), 0.9))
        gated = on2.field()[idx]
        assert not torch.equal(gated, bend_off), f"{mode}: a supplied likelihood must act"
        if mode == "brake":
            # a brake-only arm's BEND field is never gated, whatever is supplied
            assert torch.equal(on2.field()[0], bend_off)
