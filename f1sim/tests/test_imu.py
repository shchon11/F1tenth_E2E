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
    assert r.imu.shape == (1, 2, 6) and r.imu_att.shape == (1, 3)        # 100 Hz / 40 Hz -> 2 samples per step
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
    assert gz.std() > 0.01                                           # per-env gyro bias differs
    assert (sim.P["gyro_bias_z"] - gz).abs().mean() < 0.01
