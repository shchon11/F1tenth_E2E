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


def spec_for(sim, **kw):
    lidar = sim.cfg.lidar
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
    assert float(p[floor].min()) > spec.gate_threshold
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
    spec = fl.FloorSpec().validate()
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
    spec = fl.FloorSpec().validate()
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
    flat = fl.FloorSpec(smooth_weight=0.0).validate()
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
    assert o1[7] is None and o2[7].shape == (3, 128)
    assert float(o2[7].abs().max()) == 0.0, "a zero output layer predicts 0.5 for every beam"
    assert b.evaluate_aux(scan, pro, priv, act)[7] is None, "not asked for, not computed"


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
