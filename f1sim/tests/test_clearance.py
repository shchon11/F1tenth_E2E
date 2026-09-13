"""The `clearance` arm: the local occupancy, its distance field, what the arm does to a plan, and
the three things that must stay true whatever it does.

The three:

* **`legacy` is untouched.** `PlanTracker._plan_hook` defaults to None; with nothing installed the
  tracker's outputs are bit-identical to what they were.
* **A plan that already keeps the margin leaves unchanged**, bit for bit -- so "the arm did nothing"
  is a fact about the action and not an approximation of one.
* **Order-freedom with `fixed_low`.** The two layers bind different attributes, so installing them
  in either order has to produce the same command.
"""
from __future__ import annotations

import math

import pytest
import torch

from f1sim import mpc
from f1sim.learn import clearance as cl
from f1sim.learn import grip_control as gc
from f1sim.learn import grip_runtime as gr

N_BEAMS, FOV, RANGE_MAX, V_MAX = 1081, 4.71238898, 10.0, 10.0
WB, SMAX = 0.3302, 0.4189
SPEC = mpc.PlanSpec()
ANGLES = cl.beam_angles(N_BEAMS, FOV)


# ---------------------------------------------------------------- synthetic scans
MOUNT_X = 0.297


def ray_scan(half_width=None, circles=(), batch=1, mount_x=MOUNT_X, walls=None):
    """A synthetic LiDAR frame, normalized the way `gym_env._norm_scan` normalizes a real one.

    Geometry is given in **base_link**, the frame the plan lives in -- `half_width` puts two
    infinite walls at y = +-half_width, `walls` places them at signed offsets so a car can be off
    centre between them, and each circle is (x, y, r) -- and the rays are cast from where the
    scanner actually is, `mount_x` ahead of it. Beams that reach nothing read 1.0, which is "no
    return": the value the arm has to treat as unknown rather than as a wall at 10 m.
    """
    sa, ca = torch.sin(ANGLES), torch.cos(ANGLES)
    r = torch.full((N_BEAMS,), float(RANGE_MAX))
    for wy in (walls if walls is not None else
               (() if half_width is None else (half_width, -half_width))):
        sgn = 1.0 if wy > 0 else -1.0
        t = torch.where(sa * sgn > 1e-6, abs(wy) / (sa * sgn).clamp_min(1e-6),
                        torch.full_like(sa, 1e9))
        r = torch.minimum(r, t)
    for bx, by, br in circles:
        cx, cy = bx - mount_x, by                       # the circle seen from the scanner
        b = ca * (-cx) + sa * (-cy)
        disc = b * b - (cx * cx + cy * cy - br * br)
        t = torch.where(disc > 0, -b - torch.sqrt(disc.clamp_min(0.0)), torch.full_like(b, 1e9))
        r = torch.minimum(r, torch.where(t > 0, t, torch.full_like(t, 1e9)))
    return (r / RANGE_MAX).clamp(0.0, 1.0)[None].expand(batch, N_BEAMS).contiguous()


def plan(curvature=0.0, v=5.0, batch=1):
    a = torch.zeros(batch, mpc.ACT_DIM)
    a[:, :mpc.N_KNOTS] = curvature / SPEC.kappa_max
    a[:, mpc.N_KNOTS] = a[:, mpc.N_KNOTS + 1] = 2.0 * (v / V_MAX) - 1.0
    return a


def run(scan, action, v=3.0, cap=4.5, spec=None):
    cs = (spec or cl.ClearanceSpec()).validate()
    dist = cl.distance_field(cl.occupancy(scan, ANGLES, cs, RANGE_MAX), cs)
    n = action.shape[0]
    return cl.adjust(action, torch.full((n,), float(v)), torch.full((n,), float(cap)),
                     dist, SPEC, cs, V_MAX)


# ================================================================ the local occupancy
def test_a_wall_lands_in_the_cells_the_returns_are_in_and_nowhere_else():
    cs = cl.ClearanceSpec().validate()
    occ = cl.occupancy(ray_scan(half_width=0.8), ANGLES, cs, RANGE_MAX)
    assert occ.shape == (1, cs.ny, cs.nx)
    assert set(occ.unique().tolist()) <= {0.0, 1.0}
    rows = occ[0].sum(1).nonzero().flatten().tolist()
    ys = [-cs.y_half + (i + 0.5) * cs.cell for i in rows]
    # exactly two walls, each within half a cell of +-0.8, and nothing in between
    assert len(ys) == 2, ys
    assert all(abs(abs(y) - 0.8) <= cs.cell for y in ys), ys


def test_a_beam_with_no_return_marks_nothing():
    """`range_max` is "I saw nothing that far", not "there is a wall at 10 m"."""
    cs = cl.ClearanceSpec().validate()
    empty = torch.ones(1, N_BEAMS)
    assert float(cl.occupancy(empty, ANGLES, cs, RANGE_MAX).sum()) == 0.0
    # and a field built from it is free everywhere, so nothing the arm does can be triggered by it
    d = cl.distance_field(cl.occupancy(empty, ANGLES, cs, RANGE_MAX), cs)
    assert float(d.min()) == pytest.approx(cs.d_clip)


def test_the_sensor_offset_is_taken_out_so_the_grid_is_in_the_plan_s_frame():
    """The plan starts at base_link; the returns start 0.297 m ahead of it. A grid that forgot the
    offset would put every obstacle 0.297 m too close -- one and a half margins."""
    cs = cl.ClearanceSpec().validate()
    scan = ray_scan(circles=[(2.0, 0.0, 0.15)])          # 2 m ahead of base_link
    front = lambda o: cs.x_min + (float(o[0].sum(0).nonzero().flatten().min()) + 0.5) * cs.cell
    right = cl.occupancy(scan, ANGLES, cs, RANGE_MAX, mount_x=MOUNT_X)
    assert front(right) == pytest.approx(2.0 - 0.15, abs=cs.cell)
    forgotten = cl.occupancy(scan, ANGLES, cs, RANGE_MAX, mount_x=0.0)
    assert front(right) - front(forgotten) == pytest.approx(MOUNT_X, abs=cs.cell)


def test_a_return_outside_the_grid_is_dropped_not_clamped_to_the_edge():
    cs = cl.ClearanceSpec().validate()
    far = cl.occupancy(ray_scan(circles=[(8.0, 0.0, 0.3)]), ANGLES, cs, RANGE_MAX)
    assert float(far.sum()) == 0.0, "an 8 m obstacle must not be folded onto the 4.4 m edge"


# ================================================================ the distance field
def test_the_distance_field_is_the_euclidean_one_not_a_chamfer_approximation():
    cs = cl.ClearanceSpec(x_min=-0.24, x_max=1.20, y_half=0.72).validate()
    occ = torch.zeros(1, cs.ny, cs.nx)
    i, j = cs.ny // 2, cs.nx // 2
    occ[0, i, j] = 1.0
    d = cl.distance_field(occ, cs)
    ii = torch.arange(cs.ny)[:, None].float()
    jj = torch.arange(cs.nx)[None, :].float()
    truth = (((ii - i) ** 2 + (jj - j) ** 2).sqrt() * cs.cell).clamp(max=cs.d_clip)
    assert torch.allclose(d[0], truth, atol=1e-5), float((d[0] - truth).abs().max())


def test_the_field_saturates_at_d_clip_and_is_zero_on_a_return():
    cs = cl.ClearanceSpec().validate()
    d = cl.distance_field(cl.occupancy(ray_scan(half_width=0.8), ANGLES, cs, RANGE_MAX), cs)
    assert float(d.max()) == pytest.approx(cs.d_clip)
    assert float(d.min()) == 0.0


def test_sampling_outside_the_grid_reads_as_free_and_says_so():
    cs = cl.ClearanceSpec().validate()
    d = cl.distance_field(cl.occupancy(ray_scan(half_width=0.8), ANGLES, cs, RANGE_MAX), cs)
    inside = cl.sample_field(d, torch.tensor([[1.0]]), torch.tensor([[0.74]]), cs)
    outside = cl.sample_field(d, torch.tensor([[9.0]]), torch.tensor([[0.0]]), cs)
    assert float(inside) < 0.2 and float(outside) == pytest.approx(cs.d_clip)


# ================================================================ what the arm does to a plan
def test_a_plan_with_room_is_returned_bit_identical():
    a = plan(curvature=0.0, v=4.5)
    out = run(ray_scan(half_width=1.4), a)
    assert torch.equal(out.action, a)
    assert float(out.dk) == 0.0 and float(out.dv) == 0.0
    assert float(out.clear_after) >= cl.MARGIN


def test_a_plan_through_a_return_is_shifted_or_slowed():
    """The contract's claim, in one assertion: a plan that goes through something the scan saw does
    not leave this arm unchanged, and what comes out has more room or less speed."""
    a = plan(curvature=0.0, v=4.5)
    out = run(ray_scan(half_width=1.2, circles=[(1.8, 0.22, 0.20)]), a)
    assert float(out.clear_before) < 0, float(out.clear_before)
    assert not torch.equal(out.action, a)
    assert float(out.dk) != 0.0 or float(out.dv) < 0.0
    assert float(out.clear_after) > float(out.clear_before)


def test_the_bend_goes_away_from_the_nearer_side():
    a = plan(curvature=0.0, v=4.5)
    left = run(ray_scan(half_width=1.2, circles=[(1.8, 0.22, 0.20)]), a)
    right = run(ray_scan(half_width=1.2, circles=[(1.8, -0.22, 0.20)]), a)
    assert float(left.dk) < 0 < float(right.dk), (float(left.dk), float(right.dk))
    assert float(left.shift) < 0 < float(right.shift)
    for out in (left, right):
        assert float(out.clear_after) > float(out.clear_before)
    # mirror images: the same magnitude either way, so the arm has no side it prefers
    assert float(left.dk) == pytest.approx(-float(right.dk), abs=1e-6)
    assert float(left.clear_after) == pytest.approx(float(right.clear_after), abs=2e-2)


def test_a_corridor_with_nowhere_to_go_is_slowed_instead_of_bent():
    """A plan driving at a wall down a corridor too tight to dodge in: the speed comes off."""
    a = plan(curvature=0.0, v=4.5)
    out = run(ray_scan(circles=[(1.6, 0.0, 0.9)]), a, v=3.0, cap=4.5)
    assert float(out.clear_after) < cl.MARGIN, "the test case must actually be blocked"
    assert float(out.v1) < 4.5 - 1e-3, float(out.v1)
    assert float(out.dv) < 0.0


def test_the_arm_never_raises_a_speed_and_never_straightens_a_plan_the_policy_bent():
    torch.manual_seed(4401)
    a = torch.rand(32, mpc.ACT_DIM) * 2 - 1
    scan = ray_scan(half_width=0.9, circles=[(2.0, 0.3, 0.25)], batch=32)
    out = run(scan, a, v=3.5, cap=4.0)
    k0, _Lp, v0, v1 = mpc.decode(a, torch.full((32,), 3.5), V_MAX, torch.full((32,), 4.0), SPEC)
    k1, _Lp1, v0n, v1n = mpc.decode(out.action, torch.full((32,), 3.5), V_MAX,
                                    torch.full((32,), 4.0), SPEC)
    assert bool((v0n <= v0 + 1e-6).all()) and bool((v1n <= v1 + 1e-6).all())
    # the bend is one offset on the knots inside the window, so no knot may move by more than it
    assert float((k1 - k0).abs().max()) <= out.dk.abs().max() + 1e-5


def test_the_tail_of_the_plan_is_left_alone_so_the_friction_envelope_is_not_moved_with_it():
    """The taper's whole job. `fixed_low` reads the curvature of every sample of the plan, so a bend
    that ran to the end would quietly slow the car for a corner the policy never planned."""
    a = plan(curvature=0.0, v=6.0)
    out = run(ray_scan(half_width=1.2, circles=[(1.8, 0.22, 0.20)]), a, v=6.0, cap=6.0)
    assert float(out.dk) != 0.0
    k0, Lp, _v0, _v1 = mpc.decode(a, torch.tensor([6.0]), V_MAX, torch.tensor([6.0]), SPEC)
    k1, _, _, _ = mpc.decode(out.action, torch.tensor([6.0]), V_MAX, torch.tensor([6.0]), SPEC)
    s_eval, Lp = float(out.s_eval), float(Lp)
    for j in range(mpc.N_KNOTS):
        s_j = j / (mpc.N_KNOTS - 1) * Lp
        if s_j > s_eval + Lp / (mpc.N_KNOTS - 1) + 1e-6:
            assert float(k1[0, j]) == pytest.approx(float(k0[0, j]), abs=1e-6), j


def _first_contact(action, dist, cs, v=3.0, cap=4.5):
    """Arc length at which this plan's body edge first enters something the scan saw, or its whole
    length if it never does."""
    k, Lp, _v0, _v1 = mpc.decode(action, torch.tensor([v]), V_MAX, torch.tensor([cap]), SPEC)
    x, y, _psi, s = mpc.path_points(k, Lp, n=cs.n_path)
    c = cl.sample_field(dist, x, y, cs) - cs.body_radius
    inside = (c < 0) & (s >= cs.s_min)
    return float(s[0, int(inside[0].float().argmax())]) if bool(inside.any()) else float(Lp[0])


def test_a_plan_that_drives_through_a_wall_does_not_score_as_free_space_on_the_far_side():
    """The distance field is unsigned, so a point a metre past a wall reads as a metre of free
    space. Everything from the first contact is therefore void, and the bend the arm picks has to be
    the one that stays clear *longer* -- not the one whose far side happens to read well."""
    cs = cl.ClearanceSpec().validate()
    d = cl.distance_field(cl.occupancy(ray_scan(half_width=0.7), ANGLES, cs, RANGE_MAX), cs)
    a = plan(curvature=0.6, v=4.5)                       # curving hard into the left wall
    out = cl.adjust(a, torch.tensor([3.0]), torch.tensor([4.5]), d, SPEC, cs, V_MAX)
    assert float(out.clear_before) < 0.0, "the test case must actually go through the wall"
    assert float(out.dk) < 0.0, "the bend has to be away from the wall it is driving into"
    assert float(out.dv) < 0.0, "and a plan still in the wall has to lose speed"
    assert _first_contact(out.action, d, cs) > _first_contact(a, d, cs)


def test_the_worst_point_of_a_plan_the_bend_cannot_move_does_not_freeze_the_choice():
    """A car already alongside a wall: the tightest point of the window is its first sample, which
    no candidate can move. Scoring on that minimum alone ties every candidate together and the arm
    does nothing on exactly the narrow floors it exists for -- so the score is the mean over the
    window, and the arm still edges away."""
    cs = cl.ClearanceSpec().validate()
    # a 1.6 m corridor with the car 0.51 m off centre: 0.29 m to the left wall, 1.31 m to the right
    scan = ray_scan(walls=(0.29, -1.31))
    d = cl.distance_field(cl.occupancy(scan, ANGLES, cs, RANGE_MAX), cs)
    out = cl.adjust(plan(curvature=0.0, v=4.5), torch.tensor([3.0]), torch.tensor([4.5]),
                    d, SPEC, cs, V_MAX)
    assert 0.0 < float(out.clear_before) < cs.margin, float(out.clear_before)
    assert float(out.dk) < 0.0, "away from the wall it is hugging"
    assert float(out.clear_after) > float(out.clear_before)


def test_the_margin_is_the_knob_and_a_bigger_one_acts_where_a_smaller_one_does_not():
    a = plan(curvature=0.0, v=4.0)
    scan = ray_scan(half_width=0.55)                        # ~0.38 m of body-edge clearance
    tight = run(scan, a, v=3.0, cap=4.0, spec=cl.ClearanceSpec(margin=0.20))
    wide = run(scan, a, v=3.0, cap=4.0, spec=cl.ClearanceSpec(margin=0.45))
    assert torch.equal(tight.action, a), "0.20 m is met here and must be a no-op"
    assert float(wide.dv) < 0.0, "0.45 m is not met here and must cost speed"


def test_spec_validation_refuses_a_field_that_cannot_measure_its_own_margin():
    with pytest.raises(ValueError):
        cl.ClearanceSpec(margin=0.0).validate()
    with pytest.raises(ValueError):
        cl.ClearanceSpec(d_clip=0.10).validate()            # saturates below margin + body radius
    with pytest.raises(ValueError):
        cl.ClearanceSpec(cell=0.0).validate()


def test_the_arm_refuses_a_scan_whose_beam_count_is_not_its_own():
    cs = cl.ClearanceSpec().validate()
    with pytest.raises(ValueError):
        cl.occupancy(torch.ones(1, 540), ANGLES, cs, RANGE_MAX)


# ================================================================ legacy, and composition
def test_the_plan_hook_defaults_to_none_and_the_legacy_tracker_is_bit_identical():
    a = plan(curvature=0.5, v=5.0, batch=3)
    v = torch.full((3,), 3.0)
    cap = torch.full((3,), 5.0)
    tr = mpc.PlanTracker(3, "cpu", WB, SMAX, V_MAX, compile_solver=False)
    assert tr._plan_hook is None
    cmd0 = tr(a, v, cap, None, delay=torch.full((3,), SPEC.delay))
    ref0 = tr.last_ref.clone()
    tr2 = mpc.PlanTracker(3, "cpu", WB, SMAX, V_MAX, compile_solver=False)
    tr2._plan_hook = lambda action, v_meas, speed_cap: action        # an identity hook
    cmd1 = tr2(a, v, cap, None, delay=torch.full((3,), SPEC.delay))
    assert torch.equal(cmd0, cmd1) and torch.equal(ref0, tr2.last_ref)


def _tracker_with(layers, scan, batch=1):
    """A tracker wearing `layers`, installed in the order given, with the clearance arm fed a frame.

    The order is a test parameter, not part of an arm's name: `grip_runtime` spells the composite
    one way only (`fixed_low+clearance`), and what has to be proved is that the *installation*
    order cannot change the command.
    """
    tr = mpc.PlanTracker(batch, "cpu", WB, SMAX, V_MAX, compile_solver=False)
    installed = []
    for name in layers:
        if name == "fixed_low":
            g = gc.GripMPC(tr, gc.GripSpec(mode="fixed").validate(), batch, "cpu", WB, SMAX, V_MAX)
            installed.append(g.install(graph=False))
        elif name == "clearance":
            c = cl.ClearanceArm(tr, cl.ClearanceSpec(), batch, "cpu", V_MAX, ANGLES, RANGE_MAX)
            c.update_scan(scan)
            installed.append(c.install())
        else:
            raise AssertionError(name)
    return tr, installed


def _drive(tr, a, v=3.0, cap=4.5):
    n = a.shape[0]
    tr.reset(torch.zeros(n, dtype=torch.long))
    cmd = tr(a, torch.full((n,), float(v)), torch.full((n,), float(cap)), None,
             delay=torch.full((n,), SPEC.delay))
    return cmd.clone(), tr.last_ref.clone()


def test_fixed_low_and_clearance_compose_in_either_order():
    """They bind different attributes -- `_solver` and `_plan_hook` -- so neither can overwrite the
    other, and the friction envelope is computed on the adjusted geometry whichever went on first."""
    scan = ray_scan(half_width=1.0, circles=[(2.0, 0.25, 0.22)])
    a = plan(curvature=0.0, v=5.0)
    tr_a, inst_a = _tracker_with(("fixed_low", "clearance"), scan)
    cmd_a, ref_a = _drive(tr_a, a)
    tr_b, inst_b = _tracker_with(("clearance", "fixed_low"), scan)     # the other install order
    cmd_b, ref_b = _drive(tr_b, a)
    assert torch.equal(cmd_a, cmd_b), (cmd_a, cmd_b)
    assert torch.equal(ref_a, ref_b)
    for x in inst_a + inst_b:
        x.release()
    assert tr_a._plan_hook is None and tr_a._solver is None


def test_each_layer_takes_speed_off_a_blocked_plan_and_the_composite_takes_both():
    """Both layers lower the command, and the composite lowers it below either base alone.

    Not below the `clearance`-only command, though, and the reason is worth stating: `fixed_low`
    also lowers the *brake* bound the iLQR may command, so a car asked to slow down under it
    arrives at the 0.15 s lead point marginally faster than one that may brake at the full 5 m/s^2.
    Less braking authority is exactly what the friction clamp is for.
    """
    scan = ray_scan(half_width=0.8, circles=[(2.0, 0.0, 0.30)])
    a = plan(curvature=0.0, v=6.0)
    out = {}
    for name, layers in (("legacy", ()), ("fixed_low", ("fixed_low",)),
                         ("clearance", ("clearance",)),
                         ("fixed_low+clearance", ("fixed_low", "clearance"))):
        tr, inst = _tracker_with(layers, scan)
        out[name] = _drive(tr, a, v=4.0, cap=6.0)[0][0, 1].item()
        for x in inst:
            x.release()
    assert out["clearance"] < out["legacy"], out
    assert out["fixed_low"] < out["legacy"], out
    assert out["fixed_low+clearance"] < out["fixed_low"], out


def test_release_puts_the_tracker_back_exactly_as_it_was():
    scan = ray_scan(half_width=1.0, circles=[(2.0, 0.25, 0.22)])
    a = plan(curvature=0.3, v=5.0)
    tr = mpc.PlanTracker(1, "cpu", WB, SMAX, V_MAX, compile_solver=False)
    before = _drive(tr, a)
    arm = cl.ClearanceArm(tr, cl.ClearanceSpec(), 1, "cpu", V_MAX, ANGLES, RANGE_MAX)
    arm.update_scan(scan)
    arm.install()
    assert not torch.equal(_drive(tr, a)[0], before[0])
    arm.release()
    after = _drive(tr, a)
    assert torch.equal(after[0], before[0]) and torch.equal(after[1], before[1])


def test_two_plan_shapers_on_one_tracker_is_refused():
    tr = mpc.PlanTracker(1, "cpu", WB, SMAX, V_MAX, compile_solver=False)
    a1 = cl.ClearanceArm(tr, cl.ClearanceSpec(), 1, "cpu", V_MAX, ANGLES, RANGE_MAX).install()
    a2 = cl.ClearanceArm(tr, cl.ClearanceSpec(), 1, "cpu", V_MAX, ANGLES, RANGE_MAX)
    with pytest.raises(RuntimeError):
        a2.install()
    a1.release()


# ================================================================ the arm names
def test_every_composite_name_exists_and_splits_back_to_its_layers():
    assert "clearance" in gr.ARMS and "fixed_low+clearance" in gr.ARMS
    assert gr.split_arm("clearance") == ("legacy", False, True)
    assert gr.split_arm("fixed_low+clearance") == ("fixed_low", False, True)
    assert gr.split_arm("fixed_low+clearance+tcs") == ("fixed_low", True, True)
    for bad in ("clearance+fixed_low", "legacy+clearance", "clearance+clearance", "Clearance"):
        with pytest.raises(ValueError):
            gr.split_arm(bad)
    from f1sim.learn.benchmark.roster import ARMS as ROSTER_ARMS
    assert "fixed_low+clearance" in ROSTER_ARMS
    from f1sim.viewer.sim_worker import viewer_arms
    assert "fixed_low+clearance" in viewer_arms()


def test_a_clearance_spec_is_only_accepted_by_an_arm_that_has_the_layer():
    with pytest.raises(ValueError):
        gr.ControllerRuntime(object(), "fixed_low", clearance_spec=cl.ClearanceSpec())


def test_the_recorded_meta_carries_the_margin_and_the_geometry():
    tr = mpc.PlanTracker(1, "cpu", WB, SMAX, V_MAX, compile_solver=False)
    arm = cl.ClearanceArm(tr, cl.ClearanceSpec(), 1, "cpu", V_MAX, ANGLES, RANGE_MAX)
    meta = arm.meta()
    assert meta["spec"]["margin"] == cl.MARGIN
    assert meta["spec"]["margin_centre"] == pytest.approx(cl.MARGIN + cl.BODY_RADIUS)
    assert meta["n_beams"] == N_BEAMS and meta["fov_deg"] == pytest.approx(270.0, abs=0.1)
    assert meta["mount_x"] == pytest.approx(0.297)


def test_the_arm_reports_what_it_did_and_drains_it():
    tr = mpc.PlanTracker(2, "cpu", WB, SMAX, V_MAX, compile_solver=False)
    arm = cl.ClearanceArm(tr, cl.ClearanceSpec(), 2, "cpu", V_MAX, ANGLES, RANGE_MAX)
    arm.update_scan(ray_scan(half_width=1.2, circles=[(1.8, 0.22, 0.20)], batch=2))
    arm.install()
    _drive(tr, plan(curvature=0.0, v=6.0, batch=2), v=4.0, cap=6.0)
    m = arm.metrics()
    assert m["controller/clearance_plan_margin_after"] > m["controller/clearance_plan_margin_before"]
    assert m["controller/clearance_bent_frac"] == 1.0
    assert m["controller/clearance_slowed_frac"] > 0.0
    assert m["controller/clearance_shift_mean"] > 0.0
    assert arm.metrics() == {}, "a drained accumulator reports nothing, not zeros"
    arm.release()


# ================================================================ batched, and in a real env
def test_each_env_is_judged_on_its_own_scan():
    """One arm, one batch, two different worlds: the blocked row is adjusted and the clear row is
    handed back untouched. A grid accidentally shared across the batch would move both."""
    a = plan(curvature=0.0, v=4.5, batch=2)
    scan = torch.cat([ray_scan(half_width=1.4),                         # room to spare
                      ray_scan(half_width=1.0, circles=[(1.8, 0.20, 0.22)])], 0)
    out = run(scan, a, v=3.0, cap=4.5)
    assert torch.equal(out.action[0], a[0])
    assert not torch.equal(out.action[1], a[1])
    assert float(out.dk[0]) == 0.0 and float(out.dk[1]) != 0.0
    assert float(out.clear_before[0]) > float(out.clear_before[1])


def _plan_env(n=4):
    from f1sim import Config, Track
    from f1sim.gym_env import EnvConfig, F1VecEnv
    tr = Track.generate_random(0, style="competition")
    cfg = Config()
    cfg.rand.enabled = False
    return F1VecEnv(tr, cfg, EnvConfig(action_mode="plan", scan_stack=2, action_history=1,
                                       speed_cap=4.0, compile_tracker=False),
                    num_envs=n, device="cpu")


def _roll(arm, steps=12, seed=11, env=None):
    """One short rollout under `arm`, from a **fresh** environment.

    Fresh because `Simulator.step` advances `self.t` and `self._imu_phase` and `env.reset()` does
    not rewind them, so a second rollout in the same object starts on a different IMU phase and
    draws different sensor noise. Two rollouts that are meant to be compared bit for bit have to
    start from the same simulator, not merely from the same seed.
    """
    env = env if env is not None else _plan_env()
    rt = gr.ControllerRuntime(env, arm, device="cpu")
    rt.install()
    obs, _ = env.reset(seed=seed)
    rt.begin(obs)
    torch.manual_seed(seed)
    cmds = []
    for _ in range(steps):
        rt.pre_action(obs)
        a = torch.rand(env.B, mpc.ACT_DIM) * 2 - 1
        obs, _r, term, trunc, _i = env.step(a)
        rt.post_step(term, trunc)
        cmds.append(env.last_cmd.clone())
    m, _ = rt.collect_metrics()
    rt.release()
    return torch.stack(cmds), m, env


def test_the_legacy_arm_through_a_real_environment_is_bit_identical_to_no_arm_at_all():
    """The guarantee that matters to everything already measured: with `legacy` installed, and with
    the clearance module imported and its hook attribute present on the tracker, every command the
    simulator receives is the one it received before any of this existed."""
    bare, _m, env = _roll("legacy")
    again, _m2, _e = _roll("legacy")
    assert torch.equal(bare, again)
    assert env.tracker._plan_hook is None and env.tracker._solver is None


def test_the_arm_installs_through_the_runtime_and_leaves_the_env_as_it_found_it():
    legacy, _m, _e = _roll("legacy")
    env = _plan_env()
    clear, metrics, _e2 = _roll("clearance", env=env)
    assert not torch.equal(legacy, clear), "the arm must be doing something on a real track"
    assert metrics["controller/clearance_plan_margin_after"] >= \
           metrics["controller/clearance_plan_margin_before"]
    assert env.tracker._plan_hook is None and env.tracker._solver is None
    back, _m3, _e3 = _roll("legacy")
    assert torch.equal(back, legacy), "release must leave nothing behind"


def test_the_runtime_builds_the_arm_from_the_environment_s_own_beam_geometry():
    env = _plan_env()
    rt = gr.ControllerRuntime(env, "fixed_low+clearance", device="cpu")
    rt.install()
    try:
        assert rt.clearance is not None and rt.grip is not None
        assert rt.clearance.angles.numel() == env.n_beams
        assert float(rt.clearance.range_max) == float(env.range_max)
        assert rt.clearance.mount_x == pytest.approx(env.cfg.lidar.mount_x)
        assert "clearance" in rt.checkpoint_meta()
    finally:
        rt.release()
    assert env.tracker._plan_hook is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_arm_is_the_same_arm_on_the_gpu():
    """Batched on the device the training and evaluation loops run on, and not a different answer.

    Everything here is elementwise or a gather, so CPU and CUDA should agree to float precision --
    but the two places that could disagree materially are the `scatter_` that bins the returns and
    the `argmax` that picks the candidate, and a disagreement in either would make a GPU result a
    different system from the one the rest of this file pins.
    """
    cs = cl.ClearanceSpec().validate()
    torch.manual_seed(4401)
    n = 48
    scan = torch.rand(n, N_BEAMS).clamp(0.05, 1.0)
    a = torch.rand(n, mpc.ACT_DIM) * 2 - 1
    v = torch.rand(n) * 5 + 1
    cap = torch.full((n,), 9.0)

    def run_on(dev):
        ang = ANGLES.to(dev)
        d = cl.distance_field(cl.occupancy(scan.to(dev), ang, cs, RANGE_MAX), cs)
        return cl.adjust(a.to(dev), v.to(dev), cap.to(dev), d, SPEC, cs, V_MAX)

    host, gpu = run_on("cpu"), run_on("cuda")
    assert torch.equal(gpu.dk.cpu(), host.dk), "a different candidate was chosen on the GPU"
    assert float((gpu.action.cpu() - host.action).abs().max()) < 1e-5
    assert float((gpu.v0.cpu() - host.v0).abs().max()) < 1e-4
    assert float((gpu.v1.cpu() - host.v1).abs().max()) < 1e-4
