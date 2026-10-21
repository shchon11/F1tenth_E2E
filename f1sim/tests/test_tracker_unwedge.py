"""A car wedged against something backs out whatever its plan asks (`PlanSpec.wedge_s`, 2026-10-21).

In the console the ICCAS pick, nose-first against a hose, asked for 0.9-1.9 m/s for the whole 6 s it was
watched and never for the stop that `reverse_cmd_gate` needs, so the car pushed the hose until the episode
ended."""
import dataclasses

import pytest
import torch

from f1sim.mpc import N_KNOTS, PlanSpec, PlanTracker

WB, SMAX, VMAX, DT = 0.33, 0.4189, 9.0, 0.025


def tracker(B=2, **spec):
    tr = PlanTracker(B, "cpu", WB, SMAX, VMAX, spec=dataclasses.replace(PlanSpec(), **spec), compile_solver=False)
    tr.control_dt = DT
    return tr


def go_left(B):
    """A plan that bends left and asks for speed: a policy that wants to drive on."""
    a = torch.zeros(B, N_KNOTS + 2)
    a[:, :N_KNOTS] = 0.3
    return a


def drive(tr, steps, contact, v_meas):
    B = tr.B
    tr.contact = torch.as_tensor(contact)
    out = []
    for k in range(steps):
        v = v_meas(k) if callable(v_meas) else v_meas
        out.append(tr(go_left(B), torch.as_tensor(v, dtype=torch.float32).expand(B).clone(),
                      torch.full((B,), VMAX)))
    return torch.stack(out)                                     # (steps, B, 2)


def test_a_wedged_car_backs_out_for_backout_s_then_drives_on():
    sp = PlanSpec()
    n_wedge, n_back = round(sp.wedge_s / DT), round(sp.backout_s / DT)
    cmd = drive(tracker(), n_wedge + n_back + 5, [True, False], 0.0)
    wedged, free = cmd[:, 0], cmd[:, 1]
    assert (wedged[:n_wedge - 1, 1] > sp.reverse_cmd_gate).all()       # still asking to go: no stop request
    back = wedged[n_wedge - 1:n_wedge - 1 + n_back]
    assert torch.equal(back[:, 1], torch.full((n_back,), -sp.v_reverse))
    assert (back[:, 0] < 0).all() and (wedged[:n_wedge - 1, 0] > 0).all()   # mirrored: the nose turns left
    assert (wedged[n_wedge - 1 + n_back:, 1] > 0).all()                  # then its own plan again
    assert (free[:, 1] > 0).all()                                        # not touching: never


def test_a_car_moving_in_contact_does_not_back_out():
    cmd = drive(tracker(1), 200, [True], 1.5)                           # scraping along a hose at 1.5 m/s
    assert (cmd[:, 0, 1] > 0).all()


def test_odometry_spikes_above_the_gate_do_not_restart_the_clock():
    """The measured case: 0.28 m/s median, above the 0.4 gate on 6 % of the steps."""
    sp = PlanSpec()
    cmd = drive(tracker(1), 120, [True], lambda k: 0.45 if k % 16 == 0 else 0.28)
    assert (cmd[:, 0, 1] == -sp.v_reverse).any()


def test_reset_clears_the_clocks_and_zero_turns_it_off():
    tr = tracker()
    drive(tr, 30, [True, True], 0.0)
    tr.reset(torch.tensor([0]))
    assert float(tr.wedged_t[0]) == 0.0 and float(tr.wedged_t[1]) == pytest.approx(30 * DT)
    cmd = drive(tracker(1, wedge_s=0.0), 120, [True], 0.0)
    assert (cmd[:, 0, 1] > 0).all()
