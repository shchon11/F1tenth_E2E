"""`TractionGuard`: the real car's wheel-lock / launch-spin detector and speed-command shaper.

ROS-free by construction (the module imports `math` and nothing else), so these run anywhere.

The profiles are synthetic, and deliberately so: this file pins the *contract* -- what the state
machine does with a given (wheel speed, body acceleration) stream, that a reset leaves no trace,
that the parameters are checked, that the shaper's authority is bounded. Whether the thresholds
catch the events the car actually had is a different question with a different answer, and it is
answered by replaying the 22 recordings: `scripts/replay_traction.py`, table in REPORT.md.

Numbers that come from those recordings and are asserted here as behaviour rather than measured:

* `/odom` runs at 50 Hz with 7.2 % of its steps under 15 ms, down to 0.3 ms, where one ERPM quantum
  of speed change reads as 110 m/s^2 of wheel acceleration
  (20260827-111616 t=5.5412/5.5415, dv = -0.032 m/s over 0.29 ms).
* the raw IMU x acceleration reaches -101 m/s^2 on impact (20260826-173704 t=46.465) and +20 m/s^2
  through an ordinary wheel-speed dip, both far outside mu*g = 10.3 on this floor.
"""
import math
import os
import sys
import time
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "f1sim_ros"))

from f1sim_ros.traction import (A_BODY_MAX, LOCK, OK, SPIN, TractionGuard,  # noqa: E402
                                TractionParams, TractionState)

DT = 0.02                                  # the car's /odom period


# ==================================================================== profile helpers
def run(guard, samples, cmd=None):
    """Feed (t, wheel_speed, imu_ax[, motor_current]) tuples; return the state list.

    When `cmd` is given it is also shaped once per sample, and the shaped values come back too --
    the node calls `shape` immediately after `update`, so the tests do the same.
    """
    states, shaped = [], []
    for s in samples:
        t, v, ax = s[0], s[1], s[2]
        cur = s[3] if len(s) > 3 else None
        states.append(guard.update(t, v, ax, cur))
        if cmd is not None:
            shaped.append(guard.shape(cmd(t, v) if callable(cmd) else cmd))
    return (states, shaped) if cmd is not None else states


def cruise(v=5.0, n=60, t0=0.0, ax=0.0, cur=0.0):
    """Steady speed: wheel and body agree, which is what grip looks like."""
    return [(t0 + i * DT, v, ax, cur) for i in range(n)]


def brake(v0=6.0, a=-8.0, n=30, t0=0.0, cur=-20.0):
    """A hard but honest stop: the wheel decelerates exactly as fast as the body says."""
    out = []
    for i in range(n):
        t = t0 + i * DT
        out.append((t, max(0.0, v0 + a * i * DT), a, cur))
    return out


def lock(v0=6.0, a_wheel=-60.0, a_body=-6.0, n=40, t0=0.0, cur=-25.0):
    """Brake lock: the wheel collapses far faster than the body can possibly slow."""
    out = []
    v, vb = v0, v0
    for i in range(n):
        out.append((t0 + i * DT, max(0.0, v), a_body, cur))
        v += a_wheel * DT
        vb += a_body * DT
    return out


def spin(v0=1.0, a_wheel=25.0, a_body=4.0, n=10, t0=0.0, cur=60.0):
    """Launch spin: the driven wheel runs away from the body, here for `n` samples."""
    return [(t0 + i * DT, v0 + a_wheel * i * DT, a_body, cur) for i in range(n)]


def spin_then_recover(v0=1.0, a_wheel=25.0, a_body=4.0, n=10, tail=80, t0=0.0, cur=60.0):
    """A spin that ends: traction comes back and the wheel drops onto the body speed it had.

    A profile that just keeps accelerating the wheel forever is not a launch spin, it is a broken
    sensor -- and the guard treats it as one, re-entering after `max_hold` for as long as it lasts.
    """
    out = spin(v0, a_wheel, a_body, n, t0, cur)
    vb = v0 + a_body * n * DT
    return out + cruise(vb, tail, t0=t0 + n * DT, ax=0.0, cur=cur)


def states(sts):
    return [s.state for s in sts]


# ==================================================================== detection
def test_a_clean_cruise_never_leaves_ok():
    g = TractionGuard()
    assert set(states(run(g, cruise(5.0, 200)))) == {OK}
    assert g.state.locks == 0 and g.state.spins == 0


def test_an_honest_hard_brake_never_leaves_ok():
    """The distinction the whole detector rests on: braking hard is not locking. -8 m/s^2 is inside
    mu*g, and the wheel and the body agree about it."""
    g = TractionGuard()
    assert set(states(run(g, cruise(6.0, 25) + brake(6.0, -8.0, 40, t0=0.5)))) == {OK}


def test_braking_beyond_the_friction_bound_is_still_not_a_lock_if_the_body_agrees():
    """A body that really is losing 20 m/s^2 is not physical on this floor, so the guard clamps it
    -- but the residual is what decides, and here there is none: no lock."""
    g = TractionGuard()
    assert set(states(run(g, cruise(8.0, 25) + brake(8.0, -20.0, 20, t0=0.5)))) == {OK}


def test_a_brake_lock_is_detected_and_held_through_the_slip():
    g = TractionGuard()
    sts = run(g, cruise(6.0, 25) + lock(6.0, t0=0.5))
    assert LOCK in states(sts)
    first = next(i for i, s in enumerate(sts) if s.state == LOCK)
    # within two samples of the collapse starting, and flagged as a transition exactly once
    assert first <= 26 + 2, first
    assert sum(1 for s in sts if s.changed and s.state == LOCK) == 1
    assert sts[-1].locks == 1 and sts[-1].spins == 0
    # held while the wheel is stopped and the body is believed to still be moving, then released
    held = sum(1 for s in sts if s.state == LOCK) * DT
    assert 0.06 <= held <= TractionParams().max_hold + DT, held
    peak = sts[first]
    assert peak.wheel_accel < -TractionParams().lock_accel
    assert peak.residual >= TractionParams().lock_rate
    assert peak.body_speed > 4.0            # the plausible body speed the release aims at


def test_a_launch_spin_is_detected():
    g = TractionGuard()
    sts = run(g, cruise(1.0, 25) + spin_then_recover(1.0, t0=0.5))
    assert SPIN in states(sts)
    assert sts[-1].spins == 1 and sts[-1].locks == 0 and sts[-1].state == OK
    peak = next(s for s in sts if s.state == SPIN)
    assert peak.wheel_accel >= TractionParams().spin_accel
    assert -peak.residual >= TractionParams().spin_rate


def test_a_spin_needs_drive_torque_when_the_current_is_known():
    """Regen braking cannot be a launch spin. With no `/sensors/core` the check is skipped, not
    failed -- one recording has no such topic (20260725-152213)."""
    braking = [(t, v, ax, -30.0) for (t, v, ax, _) in spin_then_recover(1.0, t0=0.5)]
    assert SPIN not in states(run(TractionGuard(), cruise(1.0, 25) + braking))
    unknown = [(t, v, ax, None) for (t, v, ax, _) in spin_then_recover(1.0, t0=0.5)]
    assert SPIN in states(run(TractionGuard(), cruise(1.0, 25) + unknown))


def test_a_lock_below_the_speed_floor_is_ignored():
    """At a crawl the release has nothing to release towards, and the ERPM channel is quantisation.
    The floor is on the recent *peak* body speed, so a lock from real speed still counts."""
    g = TractionGuard()
    assert LOCK not in states(run(g, cruise(0.4, 25) + lock(0.4, -60.0, -6.0, 20, t0=0.5)))
    g2 = TractionGuard()
    assert LOCK in states(run(g2, cruise(2.0, 25) + lock(2.0, -60.0, -6.0, 20, t0=0.5)))


def test_the_body_acceleration_is_clamped_before_it_is_compared():
    """An IMU shock spike must not be able to explain a collapsing wheel away. Unclamped, a body
    'deceleration' of -101 m/s^2 makes the residual negative and the hardest real lock in the
    recordings invisible."""
    shock = [(t, v, -101.0, cur) for (t, v, _, cur) in lock(6.0, -60.0, n=20, t0=0.5)]
    g = TractionGuard()
    sts = run(g, cruise(6.0, 25) + shock)
    assert LOCK in states(sts)
    assert all(abs(s.body_accel) <= A_BODY_MAX + 1e-9 for s in sts)
    wide = TractionGuard(TractionParams(a_body_max=1000.0).validate())
    assert LOCK not in states(run(wide, cruise(6.0, 25) + shock))


def test_a_missing_imu_sample_holds_the_last_body_acceleration():
    g = TractionGuard()
    run(g, cruise(6.0, 25, ax=-3.0))
    held = g.state.body_accel
    sts = run(g, [(0.5 + i * DT, 6.0, None, 0.0) for i in range(5)])
    assert all(s.body_accel == pytest.approx(held, abs=1e-9) for s in sts)


def test_a_single_sample_timestamp_glitch_cannot_fire_the_detector():
    """7.2 % of /odom steps in the recordings are under 15 ms, some as short as 0.3 ms, and one
    ERPM quantum across such a step is 39-110 m/s^2 of apparent wheel acceleration. Differentiating
    over a window instead of a sample is what rejects them -- at the cost of the two labelled runs
    that ARE such glitches (REPORT.md names both)."""
    g = TractionGuard()
    sts = run(g, cruise(2.72, 25))
    # the 20260827-111616 t=5.5412/5.5415 pair: dv = -0.032 m/s in 0.29 ms
    sts += run(g, [(0.5 + 0.000291, 2.688, -1.0, 0.0), (0.52, 2.690, -1.0, 0.0)])
    assert LOCK not in states(sts)
    # the 20260714-223707 t=2.8782/2.8811 pair: dv = +0.112 m/s in 2.90 ms, from standstill
    g2 = TractionGuard()
    sts2 = run(g2, cruise(0.0, 25, cur=24.0))
    sts2 += run(g2, [(0.5 + 0.002902, 0.1123, 1.4, 24.0), (0.525, 0.3097, 4.5, 60.0)])
    assert SPIN not in states(sts2)


def test_a_gap_reseeds_instead_of_detecting_across_it():
    g = TractionGuard()
    run(g, cruise(6.0, 25))
    st = g.update(0.5 + 10.0, 0.0, -3.0)    # ten seconds later, wheel at zero
    assert st.state == OK and st.dt == 0.0 and not st.detecting
    assert st.body_speed == pytest.approx(0.0)


def test_non_finite_inputs_are_ignored_rather_than_believed():
    g = TractionGuard()
    run(g, cruise(6.0, 25))
    before = g.state
    assert g.update(0.5, float("nan"), -3.0) is before
    assert g.update(float("inf"), 6.0, -3.0) is before
    assert g.update(0.5, 6.0, float("nan")).body_accel == pytest.approx(before.body_accel)


# ==================================================================== shaping
def test_a_lock_releases_the_command_towards_the_plausible_body_speed():
    g = TractionGuard()
    sts, out = run(g, cruise(6.0, 25) + lock(6.0, t0=0.5), cmd=0.0)
    locked = [o for s, o in zip(sts, out) if s.state == LOCK]
    assert locked and max(locked) > 0.5, locked
    # never more than the guard's authority above what the policy asked for
    assert max(out) <= TractionParams().release_max + 1e-9
    assert min(out) >= 0.0                  # and it never brakes harder than it was asked to


def test_the_release_authority_is_a_hard_ceiling():
    """The two sensors here cannot tell a wheel sliding at 7 m/s of body speed from a car that has
    stopped while the estimate catches up. One bag costs 7.6 m/s of commanded speed without this
    cap (20260827-115713 t=40.96); with it the worst case is `release_max`."""
    for rm in (0.5, 2.0, 5.0):
        p = TractionParams(release_max=rm).validate()
        g = TractionGuard(p)
        _, out = run(g, cruise(8.0, 25) + lock(8.0, t0=0.5), cmd=0.0)
        assert max(out) <= rm + 1e-9, (rm, max(out))


def test_a_lock_limits_how_fast_the_command_may_be_cut():
    """The other half of the lock action: a command already above the body speed is not allowed to
    slam to zero while the wheel is locked."""
    g = TractionGuard()
    sts, out = run(g, cruise(6.0, 25) + lock(6.0, t0=0.5),
                   cmd=lambda t, v: 6.0 if t < 0.5 else 0.0)
    i = next(i for i, s in enumerate(sts) if s.state == LOCK)
    drop = (out[i] - out[i + 1]) / DT
    assert drop <= TractionParams().brake_rate + 1e-6, drop


def test_a_spin_caps_the_command_at_body_speed_plus_a_margin_and_ramps_back():
    p = TractionParams()
    g = TractionGuard()
    sts, out = run(g, cruise(1.0, 25) + spin_then_recover(1.0, t0=0.5), cmd=8.0)
    spinning = [(s, o) for s, o in zip(sts, out) if s.state == SPIN]
    assert spinning
    for s, o in spinning:
        assert o <= max(0.0, s.body_speed) + p.spin_margin + 1e-6, (s.body_speed, o)
    # the cap outlives the state by `spin_ramp`, growing as it goes, and is then gone entirely
    cleared = next(i for i, s in enumerate(sts) if s.state == SPIN and not s.changed
                   and sts[i + 1].state == OK) + 1
    ramp = out[cleared:cleared + int(p.spin_ramp / DT) - 1]
    assert ramp and max(ramp) < 8.0 and ramp == sorted(ramp), ramp
    assert out[-1] == pytest.approx(8.0)


def test_a_spin_cap_never_commands_reverse():
    """Body speed can be negative while the command is forward (the ERPM channel reads negative
    while the car is pushed back after a stop, 20260826-173704 t=59.5). Capping at `body + margin`
    without a floor would command reverse."""
    g = TractionGuard()
    rolling_back = [(t, v, ax, 60.0) for (t, v, ax, _) in spin(-1.0, 25.0, 0.0, 20, t0=0.5)]
    _, out = run(g, cruise(-1.0, 25, cur=60.0) + rolling_back, cmd=1.0)
    assert min(out) >= 0.0, min(out)


def test_shaping_is_neutral_while_ok():
    g = TractionGuard()
    _, out = run(g, cruise(5.0, 100), cmd=3.5)
    assert out == [pytest.approx(3.5)] * len(out)


def test_shape_without_an_update_returns_the_command_untouched():
    g = TractionGuard()
    assert g.shape(4.0) == 4.0
    assert math.isnan(g.shape(float("nan")))


def test_shape_uses_the_command_interval_not_the_update_step():
    """`/odom` is 50 Hz and the command goes out at the 40 Hz scan rate; using the 20 ms update step
    for a 25 ms command interval would run every slew limit 20 % slow. Shaping once per two updates
    must move the command twice as far per call."""
    def release(every):
        g = TractionGuard()
        out = []
        samples = cruise(6.0, 25) + lock(6.0, t0=0.5)
        for i, (t, v, ax, cur) in enumerate(samples):
            g.update(t, v, ax, cur)
            if i % every == 0:
                out.append(g.shape(0.0))
        return out
    one, two = release(1), release(2)
    assert max(two) >= max(one) - 1e-9
    # the first shaped step after entering the lock is a rate limit times the interval
    step_one = max(b - a for a, b in zip(one, one[1:]))
    step_two = max(b - a for a, b in zip(two, two[1:]))
    assert step_two == pytest.approx(2 * step_one, rel=0.35), (step_one, step_two)


# ==================================================================== reset / determinism
def test_reset_leaves_no_trace_of_what_came_before():
    dirty = TractionGuard()
    run(dirty, cruise(6.0, 25) + lock(6.0, t0=0.5), cmd=0.0)
    assert dirty.state.locks == 1
    dirty.reset()
    clean = TractionGuard()
    profile = cruise(3.0, 40) + spin_then_recover(3.0, t0=0.8)
    a, oa = run(dirty, profile, cmd=5.0)
    b, ob = run(clean, profile, cmd=5.0)
    assert states(a) == states(b)
    assert oa == ob
    assert [s.body_speed for s in a] == [s.body_speed for s in b]
    assert a[-1].locks == b[-1].locks == 0


def test_the_guard_is_deterministic():
    profile = (cruise(6.0, 25) + lock(6.0, t0=0.5) + cruise(1.0, 20, t0=1.3)
               + spin_then_recover(1.0, t0=1.7))
    runs = []
    for _ in range(3):
        g = TractionGuard()
        sts, out = run(g, profile, cmd=4.0)
        runs.append((states(sts), [round(s.residual, 12) for s in sts], out))
    assert runs[0] == runs[1] == runs[2]


def test_a_fresh_guard_reports_a_neutral_state():
    st = TractionGuard().state
    assert st == TractionState() and not st.active and st.state == OK


# ==================================================================== parameters
def test_default_parameters_validate_and_are_physically_ordered():
    p = TractionParams().validate()
    assert p.a_body_max == pytest.approx(1.05 * 9.80665)
    # both lock gates have to be above what a gripping wheel can do, or the guard fires on braking
    assert p.lock_accel > p.a_body_max and p.spin_accel > p.a_body_max
    assert p.min_diff_dt < p.max_diff_dt <= p.max_step_dt
    assert p.min_hold <= p.max_hold
    assert 0.0 < p.clear_frac <= 1.0 and 0.0 <= p.release_frac <= 1.0


@pytest.mark.parametrize("bad", [
    dict(lock_rate=0.0), dict(lock_rate=-1.0), dict(spin_rate=float("nan")),
    dict(lock_accel=0.0), dict(spin_accel=-3.0), dict(a_body_max=0.0),
    dict(clear_frac=0.0), dict(clear_frac=1.5), dict(release_frac=1.2), dict(release_frac=-0.1),
    dict(lock_decay_frac=0.0), dict(lock_decay_frac=1.1),
    dict(min_diff_dt=0.05, max_diff_dt=0.05), dict(max_diff_dt=0.3, max_step_dt=0.25),
    dict(min_hold=0.9, max_hold=0.4), dict(min_hold=-0.1),
    dict(lock_persist=0), dict(spin_persist=-1), dict(v_ref_hold=0.0),
    dict(release_rate=0.0), dict(brake_rate=-2.0), dict(spin_margin=0.0), dict(spin_ramp=0.0),
])
def test_nonsense_parameters_are_refused(bad):
    with pytest.raises(ValueError):
        TractionParams(**bad).validate()
    with pytest.raises(ValueError):
        TractionGuard(TractionParams(**bad))


def test_thresholds_are_reachable_and_actually_change_the_answer():
    profile = cruise(6.0, 25) + lock(6.0, -25.0, -6.0, 20, t0=0.5)
    assert LOCK in states(run(TractionGuard(), profile))
    deaf = TractionParams(lock_accel=40.0, lock_rate=40.0).validate()
    assert LOCK not in states(run(TractionGuard(deaf), profile))
    assert replace(TractionParams(), lock_rate=9.0).lock_rate == 9.0


def test_update_is_cheap_enough_for_the_scan_rate():
    """It runs at 40-50 Hz on the car's CPU next to the policy. A microsecond-scale budget is not
    the point; the point is that nothing here allocates per-sample or grows without bound."""
    g = TractionGuard()
    samples = cruise(6.0, 2000)
    t0 = time.perf_counter()
    for (t, v, ax, cur) in samples:
        g.update(t, v, ax, cur)
        g.shape(4.0)
    per_call = (time.perf_counter() - t0) / len(samples)
    assert per_call < 1e-3, per_call       # 1 ms against a 20-25 ms period, a 40x margin
    assert len(g._hist) <= 8 and len(g._v_ref) <= 32
