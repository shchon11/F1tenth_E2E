"""VESC IMU model: gravity leakage through roll/pitch, lever arm, vibration, sampling, attitude filter."""
import math
import numpy as np
import torch

from f1sim import Track, Config, Simulator
from f1sim.imu import specific_force, G


def open_field(size=60.0, res=0.1):
    n = int(size / res); occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return Track.from_occupancy(occ, res, (-size / 2, -size / 2), duct=np.zeros_like(occ), tall=occ)


def clean_sim(n=1, **imu):
    """No randomization, no IMU noise/vibration/bias unless overridden."""
    cfg = Config(); cfg.rand.enabled = False; cfg.actuator.cmd_delay = 0.0
    cfg.imu.gyro_noise = 0.0; cfg.imu.accel_noise = 0.0; cfg.imu.vib_accel = 0.0; cfg.imu.vib_gyro = 0.0
    cfg.imu.vib_accel_floor = 0.0; cfg.imu.vib_gyro_floor = 0.0   # the floor is the larger of the two
    cfg.imu.gyro_bias_walk = 0.0; cfg.imu.quant_gyro = 1e-9; cfg.imu.quant_accel = 1e-9
    for k, v in imu.items():
        setattr(cfg.imu, k, v)
    sim = Simulator(open_field(), cfg, num_envs=n, device="cpu")
    sim.reset(poses=torch.zeros(n, 3))
    return sim


def test_specific_force_formula():
    z = torch.zeros(1)
    fx, fy, fz = specific_force(z, z, z, z, z, z, z, z, z, z, torch.zeros(1, 3))
    assert abs(fz.item() - G) < 1e-6 and abs(fx.item()) < 1e-6
    # nose-down pitch: gravity appears as negative x; right-side-down roll: positive y
    th = torch.tensor([math.radians(5.0)]); ph = torch.tensor([math.radians(4.0)])
    fx, fy, fz = specific_force(z, z, ph, th, z, z, z, z, z, z, torch.zeros(1, 3))
    assert abs(fx.item() + G * math.sin(math.radians(5))) < 1e-6
    assert abs(fy.item() - G * math.cos(math.radians(5)) * math.sin(math.radians(4))) < 1e-6
    # lever arm: sensor 0.1 m left of the CoG spinning at 2 rad/s feels -0.4 m/s^2 in y (centripetal)
    fx, fy, fz = specific_force(z, z, z, z, z, z, torch.tensor([2.0]), z, z, z, torch.tensor([[0.0, 0.1, 0.0]]))
    assert abs(fy.item() + 0.4) < 1e-6


def test_shapes_and_rest():
    sim = clean_sim()
    r = None
    for _ in range(40):
        r = sim.step(torch.zeros(1, 2))
    # the sensor keeps its own clock, so the count per control step cycles through the schedule
    # (50 Hz on a 40 Hz loop: 1, 1, 1, 2) rather than being a fixed floor(rate * control_dt)
    counts = {len(idx) for idx in sim.imu_schedule}
    assert r.imu.shape[1] in counts and (r.imu.shape[0], r.imu.shape[2]) == (1, 6)
    assert r.imu_att.shape == (1, 3) and r.imu_offsets.shape[0] == r.imu.shape[1]
    m = r.imu[0].mean(0)
    assert m[:3].abs().max() < 1e-3 and abs(m[5] - G) < 0.02 and m[3:5].abs().max() < 0.02
    assert r.imu_att[0, :2].abs().max() < 0.01


def test_gravity_leaks_into_accelerometer_under_pitch():
    sim = clean_sim()
    r = None
    for _ in range(int(1.5 / sim.control_dt)):          # settle at constant speed
        r = sim.step(torch.tensor([[0.0, 4.0]]))
    for _ in range(int(0.6 / sim.control_dt)):          # then brake: nose dives, accelerometer over-reads
        r = sim.step(torch.tensor([[0.0, 0.0]]))
    ax_true = sim.ax[0].item(); pitch = r.attitude[0, 1].item()
    imu_ax = r.imu[0, -1, 3].item()
    assert ax_true < -2.0 and pitch > 0.01                          # braking, nose down
    assert abs(imu_ax - (ax_true - G * math.sin(pitch))) < 0.35     # low-pass lag + lever arm terms
    assert imu_ax < ax_true                                         # over-reads the deceleration


def test_cornering_gyro_and_lateral():
    sim = clean_sim()
    r = None
    for _ in range(int(3.0 / sim.control_dt)):
        r = sim.step(torch.tensor([[0.25, 3.0]]))
    m = r.imu[0].mean(0); roll, pitch = r.attitude[0, 0].item(), r.attitude[0, 1].item()
    assert abs(m[2].item() - r.state[0, 5].item()) < 0.03
    P = {k: sim.P[k][0].item() for k in ("imu_x", "imu_y", "imu_z", "lr", "h")}
    ry = P["imu_y"]; wz = r.state[0, 5].item()
    expected = sim.ay[0].item() + G * math.cos(pitch) * math.sin(roll) - ry * wz * wz
    assert abs(m[4].item() - expected) < 0.15, (m[4].item(), expected)


def test_vibration_scales_with_speed_and_noise_is_present():
    cfg_sims = []
    for v in (1.0, 6.0):
        sim = clean_sim(vib_accel=0.25, vib_gyro=0.02)
        acc = []
        r = None
        for i in range(int(3.0 / sim.control_dt)):
            r = sim.step(torch.tensor([[0.0, v]]))
            if i > 60: acc.append(r.imu[0, :, 5] - G)
        cfg_sims.append(torch.cat(acc).std().item())
    assert cfg_sims[1] > 3 * cfg_sims[0] and cfg_sims[1] > 0.3


def test_attitude_filter_converges_at_rest_and_bends_under_acceleration():
    sim = clean_sim()
    r = None
    for _ in range(int(4.0 / sim.control_dt)):
        r = sim.step(torch.zeros(1, 2))
    assert r.imu_att[0, :2].abs().max() < 0.005
    for _ in range(int(1.5 / sim.control_dt)):
        r = sim.step(torch.tensor([[0.0, 7.0]]))
        if sim.ax[0] < 3.0: break
    err = r.imu_att[0, 1].item() - r.attitude[0, 1].item()
    assert err < -0.03, err                                          # estimate pitched nose-up beyond the truth


def test_randomized_imu_is_finite_and_biased():
    cfg = Config()
    sim = Simulator(open_field(), cfg, num_envs=32, device="cpu")
    for _ in range(40):
        r = sim.step(torch.zeros(32, 2))
    assert torch.isfinite(r.imu).all()
    gz = r.imu[:, :, 2].mean(1)
    # Derive the expected spread from the randomization range instead of hard-coding it: the range
    # was +-0.05 rad/s (a BMI160 spec bound) until it was measured on this car, where the zero-rate
    # offset never exceeds 0.0018 on any axis, and a fixed 0.01 threshold silently encoded the old
    # range. std of a uniform over [lo, hi] is (hi - lo) / sqrt(12).
    lo, hi = cfg.rand.ranges["imu.gyro_bias_z"]
    expected = (hi - lo) / math.sqrt(12)
    assert gz.std() > 0.5 * expected, (gz.std().item(), expected)     # per-env gyro bias differs
    assert (sim.P["gyro_bias_z"] - gz).abs().mean() < max(0.01, 3 * cfg.imu.gyro_noise)


def test_imu_rate_matches_the_car():
    """The recordings all sample the VESC IMU at 50 Hz, not the 100 Hz vesc_tool default.

    At 100 Hz the sim handed the policy two samples per control step to average where the car
    delivers one, so the accelerometer channel it sees was quieter in sim than on the car by
    sqrt(2) for free -- a gap that no amount of noise-amplitude calibration would show up in.
    """
    from f1sim.params import Config
    assert Config().imu.imu_rate == 50.0


def test_longitudinal_channel_is_specific_force_not_state_derivative():
    """In the rotating body frame the accelerometer reads f_x = vx_dot - vy*r.

    Steady cornering has vy < 0 and r > 0 (or both flipped), so -vy*r is a real offset the IMU
    sees and vx_dot does not. Drive a constant-radius corner and check the returned ax differs
    from the speed derivative by exactly that product.
    """
    import f1sim.dynamics as dyn
    sim = clean_sim()
    for _ in range(int(3.0 / sim.control_dt)):
        sim.step(torch.tensor([[0.30, 5.0]]))
    vx0 = sim.state[0, dyn.IVX].item()
    r = sim.step(torch.tensor([[0.30, 5.0]]))
    vx1, vy, yr = (sim.state[0, i].item() for i in (dyn.IVX, dyn.IVY, dyn.IR))
    assert abs(vy) > 0.02 and abs(yr) > 1.0                     # actually cornering with sideslip
    vxdot = (vx1 - vx0) / sim.control_dt
    ax = sim.ax[0].item()
    assert abs(ax - (vxdot - vy * yr)) < 0.5                    # substep averaging, not exact
    assert abs(ax - vxdot) > 0.5 * abs(vy * yr)                 # and the term is not being dropped


def test_vibration_floor_switches_on_with_motion_and_is_silent_at_rest():
    """Measured on the car: the >5 Hz residual is 0.018 m/s^2 truly still and 1.42 at 0.5 m/s.

    A floor that acts at a standstill leaves a parked car buzzing at 80x its real noise; a pure
    `coefficient * speed` is silent at the low speeds where the policy reads its proprio history
    to infer grip. The floor has to ramp in with wheel speed, and it has to be broadband -- at
    0.5 m/s the wheel turns at 1.6 Hz, so a tonal floor would sit below the band it was fitted in.
    """
    from f1sim.params import Config
    c = Config().imu
    assert c.vib_accel_floor > 0 and c.vib_gyro_floor > 0 and c.vib_onset_v > 0

    def resid_rms(speed, secs=4.0):
        sim = clean_sim(vib_accel_floor=c.vib_accel_floor, vib_gyro_floor=c.vib_gyro_floor,
                        vib_accel=c.vib_accel, vib_gyro=c.vib_gyro, vib_onset_v=c.vib_onset_v)
        acc = []
        for i in range(int(secs / sim.control_dt)):
            r = sim.step(torch.tensor([[0.0, speed]]))
            if i > int(1.5 / sim.control_dt):
                acc.append(r.imu[0, :, 3:6])
        x = torch.cat(acc)
        return (x - x.mean(0)).pow(2).mean().sqrt().item()

    still, slow, fast = resid_rms(0.0), resid_rms(0.6), resid_rms(6.0)
    assert still < 0.1, still                       # a parked car is quiet
    assert slow > 20 * still, (slow, still)         # and switches on as soon as it rolls
    assert fast > slow                              # then grows with speed


def test_delivered_imu_rate_is_the_configured_one():
    """The sensor delivers the rate it is configured for, including on a non-integer ratio.

    This test used to assert the opposite: `floor(imu_rate * control_dt)` gave one sample per
    control step and a 50 Hz IMU came out at 40 Hz. That was the implementation's rounding, not the
    sensor's behaviour, and it also handed the noise model a 20 ms step while emitting every 25 ms.
    The emitter now follows the sensor's own clock and cycles 1, 1, 1, 2 samples per step.
    """
    from f1sim.params import Config
    from f1sim.imu import sample_schedule
    c = Config()
    control_dt = 1.0 / c.sim.control_rate
    substeps = int(round(control_dt / c.sim.physics_dt))
    schedule, offsets, period = sample_schedule(c.imu.imu_rate, control_dt, c.sim.physics_dt, substeps)
    assert [len(s) for s in schedule] == [1, 1, 1, 2]
    assert sum(len(s) for s in schedule) / (period * control_dt) == c.imu.imu_rate

    sim = clean_sim()
    steps = int(2.0 / sim.control_dt)
    n = sum(sim.step(torch.zeros(1, 2)).imu.shape[1] for _ in range(steps))
    assert n / 2.0 == c.imu.imu_rate, (n, c.imu.imu_rate)
    # `sample_indices` is still exact whenever the ratio is an integer, which is all it claims now
    from f1sim.imu import sample_indices
    assert len(sample_indices(40.0, control_dt, c.sim.physics_dt, substeps)) == 1
