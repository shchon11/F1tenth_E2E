"""The rear axle as a rotating body: the slip-ratio tyre law, the ERPM channel, and the switch.

The physics claims here are the ones `docs/research/wheel-model-2026-09-13.md` is built on, and the
sensor claims are the ones `f1sim_ros/traction.py` reads. Every threshold in this file is either a
measured number from the 22 recordings (quoted with its source) or a bound wide enough that only a
real regression can cross it.
"""
import math

import numpy as np
import pytest
import torch

from f1sim import Config, Simulator, Track
from f1sim import dynamics as dyn


def open_field(size=80.0, res=0.2):
    n = int(size / res)
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return Track.from_occupancy(occ, res, (-size / 2, -size / 2))


@pytest.fixture(scope="module")
def track():
    return open_field()


def make_sim(track, n=1, wheel=True, **over):
    cfg = Config()
    cfg.rand.enabled = False
    cfg.actuator.cmd_delay = 0.0
    cfg.vehicle.wheel_model = wheel
    for k, v in over.items():
        grp, name = k.split(".")
        setattr(getattr(cfg, grp), name, v)
    sim = Simulator(track, cfg, num_envs=n, device="cpu")
    sim.reset(poses=torch.zeros(n, 3))
    return sim


def drive(sim, steer, speed, secs, v0=None):
    """Run a command for `secs`, returning (wheel speed, body speed, odom speed) per control step."""
    if v0 is not None:
        sim.reset(poses=torch.zeros(sim.B, 3), speed=torch.full((sim.B,), float(v0)))
    vw, vb, vo = [], [], []
    for _ in range(int(secs / sim.control_dt)):
        r = sim.step(torch.tensor([[steer, speed]] * sim.B))
        vw.append((r.state[:, dyn.IOMEGA] * sim.P["r_w"]).clone())
        vb.append(r.state[:, dyn.IVX].clone())
        vo.append(r.odom[:, 3].clone())
    return torch.stack(vw), torch.stack(vb), torch.stack(vo)


# ------------------------------------------------------------------ layout and the switch

def test_the_wheel_column_is_appended_not_inserted():
    """Everything that reads `state[:, :3]` or `state[:, 3]` keeps meaning what it meant."""
    assert dyn.STATE_DIM == 8
    assert (dyn.IX, dyn.IY, dyn.IYAW, dyn.IVX, dyn.IVY, dyn.IR, dyn.ISTEER) == (0, 1, 2, 3, 4, 5, 6)
    assert dyn.IOMEGA == 7


def test_switch_off_leaves_the_wheel_riding_the_body_and_the_sensors_clean(track):
    """With the switch off the wheel speed IS the body speed, so the guard's residual is zero and
    the ERPM artefacts are absent -- the state before 2026-09-13, exactly."""
    sim = make_sim(track, wheel=False)
    vw, vb, vo = drive(sim, 0.0, 4.0, 3.0)
    assert torch.allclose(vw[-1], vb[-1], atol=1e-4), (vw[-1], vb[-1])
    r = sim.step(torch.tensor([[0.0, 4.0]]))
    assert r.odom_t is None and r.motor_current is None
    # no quantisation: the reported speed is off the ERPM lattice with probability 1
    q = float(sim.P["erpm_quantum"][0])
    off = abs(float(r.odom[0, 3]) / q - round(float(r.odom[0, 3]) / q))
    assert off > 1e-3, "the switch-off path must not quantise"


def test_switch_off_keeps_the_old_straight_line_speed_tracking(track):
    """`test_dynamics.test_straight_line_speed_tracking`'s bound, under the switch."""
    sim = make_sim(track, wheel=False)
    vw, vb, vo = drive(sim, 0.0, 3.0, 4.0)
    assert abs(float(vb[-1, 0]) - 3.0) < 0.15


# ------------------------------------------------------------------ the tyre law

def test_no_slip_at_cruise(track):
    """Steady cruise on good grip: the wheel and the body agree to well inside one ERPM quantum of
    meaningful slip. If this drifts, every simulated lap is being driven on a slipping tyre."""
    sim = make_sim(track)
    vw, vb, _ = drive(sim, 0.0, 4.0, 4.0)
    slip = (vw[-40:] - vb[-40:]).abs().max().item()
    assert slip < 0.05, slip
    assert abs(float(vb[-1, 0]) - 4.0) < 0.20, float(vb[-1, 0])


def test_lock_under_regen_on_low_grip_in_a_corner(track):
    """A full brake command while cornering on mu 0.73: the rear axle is asked for half the
    drivetrain's force on 37 % of the weight and lets go. The wheel must fall well below the body
    and its deceleration must clear the guard's absolute lock gate (`lock_accel` = 18 m/s^2)."""
    sim = make_sim(track, **{"vehicle.mu": 0.73})
    drive(sim, 0.15, 6.0, 1.5, v0=6.0)
    vw, vb, _ = drive(sim, 0.15, 0.0, 1.0)
    slip = (vb - vw).max().item()
    aw = np.gradient(vw[:, 0].numpy(), sim.control_dt)
    assert slip > 0.8, f"the wheel never let go (max slip {slip:.2f} m/s)"
    assert aw.min() < -18.0, f"peak wheel decel {aw.min():.1f} does not reach the lock gate"


def test_no_lock_on_high_grip_in_a_straight_line(track):
    """The same brake command on mu 1.15, straight: the drivetrain has 60 % margin and the wheel
    keeps rolling. A model that locks here would fire the guard on every stop."""
    sim = make_sim(track, **{"vehicle.mu": 1.15})
    drive(sim, 0.0, 6.0, 1.5, v0=6.0)
    vw, vb, _ = drive(sim, 0.0, 0.0, 1.0)
    assert (vb - vw).max().item() < 0.35, (vb - vw).max().item()


def test_spin_under_torque_on_low_grip(track):
    """Launch from a standstill on a slippery floor: the wheel must run ahead of the body."""
    sim = make_sim(track, **{"vehicle.mu": 0.45})
    vw, vb, _ = drive(sim, 0.0, 9.0, 1.2)
    assert (vw - vb).max().item() > 0.5, (vw - vb).max().item()
    aw = np.gradient(vw[:, 0].numpy(), sim.control_dt)
    assert aw.max() > 14.0, f"peak wheel accel {aw.max():.1f} below the guard's spin gate"


def test_the_vesc_loop_closes_on_the_wheel_not_the_body():
    """The one line that makes both failures possible (`actuators.vesc_accel`)."""
    from f1sim.actuators import vesc_accel
    P = {"speed_gain": torch.ones(1), "v_min": torch.full((1,), -3.0),
         "v_max": torch.full((1,), 12.0), "motor_tau": torch.full((1,), 0.2)}
    cmd = torch.tensor([4.0])
    # a wheel spinning at 6 m/s against a body at 1: the loop must see 6 and back off
    assert float(vesc_accel(cmd, torch.tensor([6.0]), P)) == pytest.approx((4.0 - 6.0) / 0.2)
    # a locked wheel reads 0, so the loop keeps driving -- positive current through a brake lock,
    # which is what the recordings show (20260826-173704 t=46.5, -143 m/s^2 at +53 A)
    assert float(vesc_accel(torch.tensor([2.0]), torch.tensor([0.0]), P)) > 0.0


def test_the_wheel_starts_rolling_after_a_reset_with_speed(track):
    """A car spawned at 4 m/s with a stopped wheel is a car spawned mid-lock."""
    sim = make_sim(track)
    sim.reset(poses=torch.zeros(1, 3), speed=torch.tensor([4.0]))
    v = float(sim.state[0, dyn.IOMEGA] * sim.P["r_w"][0])
    assert abs(v - 4.0) < 1e-4, v


def test_the_wheel_law_is_stable_over_the_randomisation_range(track):
    """The wheel equation is stiff -- its time constant is a quarter of a substep at a standstill --
    and `step_dynamics` integrates it semi-implicitly for that reason. Draw the whole DR range and
    drive it hard; nothing may go non-finite or run away."""
    cfg = Config()
    cfg.vehicle.wheel_model = True
    sim = Simulator(track, cfg, num_envs=64, device="cpu")
    sim.reset()
    g = torch.Generator().manual_seed(7)
    for _ in range(200):
        a = torch.stack([torch.rand(64, generator=g) * 0.8 - 0.4,
                         torch.rand(64, generator=g) * 12.0 - 2.0], 1)
        r = sim.step(a)
    assert torch.isfinite(r.state).all()
    v_wheel = r.state[:, dyn.IOMEGA] * sim.P["r_w"]
    assert v_wheel.abs().max() < cfg.vehicle.v_max + dyn.WHEEL_OVERSPEED + 1e-3


# ------------------------------------------------------------------ the ERPM channel

def test_the_reported_speed_is_on_the_erpm_lattice(track):
    """Measured: the smallest non-zero |dv| in each competition bag is 2.3794e-4 m/s to within one
    float32 ulp, i.e. 1/4202.7 -- the VESC `speed_to_erpm_gain`."""
    sim = make_sim(track)
    _, _, vo = drive(sim, 0.0, 4.0, 2.0)
    q = float(sim.P["erpm_quantum"][0])
    assert q == pytest.approx(2.3794e-4)
    frac = (vo / q - torch.round(vo / q)).abs()
    assert frac.max().item() < 1e-3, frac.max().item()


def test_the_odom_stamp_jitters_the_way_the_recordings_do(track):
    """Measured over 88 975 /odom steps: median 19.998 ms, 7.22 % under 15 ms, 0.12 % under 5 ms,
    down to 0.057 ms. The sub-5 ms steps are the ones that make a quantum read as tens of m/s^2,
    which is the artefact `TractionParams.min_diff_dt` exists for."""
    sim = make_sim(track, n=256)
    ts = []
    for _ in range(200):
        r = sim.step(torch.tensor([[0.0, 4.0]] * 256))
        ts.append(r.odom_t.clone())
    t = torch.stack(ts)                               # (steps, envs)
    d = (t[1:] - t[:-1]).reshape(-1).numpy()
    assert abs(float(np.median(d)) - sim.control_dt) < 1e-3
    # The publish offset is a property of the publisher, not of the period, so it is the measured
    # 2.4 ms here too -- and the *step* is a difference of two offsets, sd 2.4*sqrt(2) = 3.39 ms.
    # What that puts under an ABSOLUTE 15 ms threshold depends on the period, and the simulator's
    # odometry runs at the 40 Hz control rate where the car's runs at 50: 15 ms is 5 ms short of
    # the car's period but 10 ms short of this one, so 0.2 % of steps land there against the car's
    # 7.2 %. The shape is the measured one; the rate of the artefact is not, and
    # docs/research/wheel-model-2026-09-13.md says so.
    # The Gaussian core: the step is a difference of two offsets, so its sd is 2.4*sqrt(2) ms.
    core = d[d > 0.010]
    assert abs(float(np.std(core)) - 0.0024 * math.sqrt(2)) < 0.0012, float(np.std(core))
    # The catch-up branch, which is the one that matters: 0.12 % of the recordings' steps are
    # shorter than 5 ms, and those are the steps that read as tens of m/s^2 of wheel acceleration.
    frac5 = float((d < 0.005).mean())
    assert 0.0004 < frac5 < 0.004, f"{100 * frac5:.3f} % of steps under 5 ms (measured 0.12 %)"
    assert d.min() < 0.003, d.min()
    assert (d > 0).all(), "stamps must stay strictly increasing"


def test_the_stamp_is_exact_when_the_jitter_is_turned_off(track):
    sim = make_sim(track, **{"odom.stamp_jitter_std": 1e-12, "odom.stamp_jitter_burst": 0.0})
    for _ in range(5):
        r = sim.step(torch.tensor([[0.0, 2.0]]))
    assert abs(float(r.odom_t[0]) - r.t) < 1e-6


def test_the_emulated_motor_current_tracks_the_commanded_torque(track):
    """`/sensors/core` `current_motor`, the guard's optional third input. Calibrated so the regen
    limit is the -30 A the competition configuration ran at."""
    sim = make_sim(track)
    drive(sim, 0.0, 6.0, 1.5, v0=6.0)
    cur = []
    for _ in range(30):
        r = sim.step(torch.tensor([[0.0, 0.0]]))
        cur.append(float(r.motor_current[0]))
    assert min(cur) < -20.0, f"hardest braking only reached {min(cur):.1f} A"
    assert min(cur) > -40.0, min(cur)


# ------------------------------------------------------------------ the IMU shock

def test_the_imu_shock_reaches_past_the_friction_bound(track):
    """`traction.A_BODY_MAX` clamps the body acceleration at mu*g because the recordings' raw
    accelerometer does not respect it: measured 0.291 samples per second of motion above 10.3 m/s^2
    and 0.021 above 50, peaking at 120.4. Without a shock term there is nothing to clamp."""
    sim = make_sim(track, n=64)
    ax, n = [], 0
    for _ in range(400):
        r = sim.step(torch.tensor([[0.0, 5.0]] * 64))
        ax.append(r.imu[:, :, 3].reshape(-1).clone())
        n += r.imu.shape[1] * 64
    a = torch.cat(ax).abs().numpy()
    secs = n / sim.cfg.imu.imu_rate / 64       # `n` counts samples over all 64 envs
    rate = float((a > 10.3).sum()) / secs / 64
    assert 0.10 < rate < 0.70, f"{rate:.3f} shocks/s above mu*g (measured 0.291)"
    assert a.max() > 30.0, a.max()


def test_no_shock_with_the_switch_off(track):
    """The bar is 20 m/s^2, not mu*g: the vibration model alone reaches 13 m/s^2 through a hard
    launch, and it did so before the wheel model existed. What must be absent is the impact tail."""
    sim = make_sim(track, n=32, wheel=False)
    sim.reset(poses=torch.zeros(32, 3), speed=torch.full((32,), 5.0))
    hi = 0.0
    for _ in range(300):
        r = sim.step(torch.tensor([[0.0, 5.0]] * 32))
        hi = max(hi, float(r.imu[:, :, 3].abs().max()))
    assert hi < 20.0, hi
