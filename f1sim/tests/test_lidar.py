import math
import numpy as np
import torch
import pytest

from f1sim import Track, Config, Simulator


def circular_room(radius=5.0, res=0.05):
    """Tall (wall) boundary only, so the 2D analytic solution applies."""
    n = int(2 * (radius + 1) / res)
    xs = (np.arange(n) - n / 2) * res
    gx, gy = np.meshgrid(xs, xs)
    occ = (gx ** 2 + gy ** 2) >= radius ** 2
    return Track.from_occupancy(occ, res, (xs[0], xs[0]), duct=np.zeros_like(occ), tall=occ)


def level_sim(track, device, n_envs=1):
    cfg = Config(); cfg.rand.enabled = False; cfg.lidar.motion_distortion = False
    cfg.lidar.noise_std = 0.0; cfg.lidar.dropout_prob = 0.0; cfg.lidar.spike_prob = 0.0
    cfg.lidar.mount_x = 0.0; cfg.lidar.range_max = 30.0
    sim = Simulator(track, cfg, num_envs=n_envs, device=device)
    sim.reset(poses=torch.zeros(n_envs, 3))
    return sim


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_circular_room_ranges_match_analytic(device):
    R = 5.0
    tr = circular_room(R)
    sim = level_sim(tr, device); lid = sim.lidar
    # sensor at (1, 0.5), yaw 0.3, no mount offset
    pose = torch.tensor([[1.0, 0.5, 0.3]], device=sim.device)
    _, r, _ = lid.scan(pose, None, sim.P, motion_distortion=False, noisy=False)
    ang = (pose[0, 2] + lid.angles).cpu().numpy()
    ox, oy = 1.0, 0.5
    dx, dy = np.cos(ang), np.sin(ang)
    b = ox * dx + oy * dy
    c = ox ** 2 + oy ** 2 - R ** 2
    t_exact = -b + np.sqrt(b ** 2 - c)
    err = np.abs(r[0].cpu().numpy() - t_exact)
    assert err.max() < 1.5 * tr.resolution, err.max()
    assert err.mean() < 0.6 * tr.resolution


def test_noise_and_dropout_statistics():
    tr = circular_room(5.0)
    cfg = Config()
    cfg.rand.enabled = False
    cfg.lidar.noise_std = 0.03; cfg.lidar.noise_std_rel = 0.0
    cfg.lidar.dropout_prob = 0.05; cfg.lidar.spike_prob = 0.0
    cfg.lidar.motion_distortion = False
    sim = Simulator(tr, cfg, num_envs=8, device="cpu")
    sim.reset(poses=torch.zeros(8, 3))
    r = sim.step(torch.zeros(8, 2))
    finite = torch.isfinite(r.scan)
    drop = 1 - finite.float().mean().item()
    assert 0.03 < drop < 0.07, drop
    resid = (r.scan - r.scan_true)[finite]
    assert abs(resid.std().item() - 0.03) < 0.005


def test_motion_distortion_shifts_beams():
    tr = circular_room(5.0)
    sim = level_sim(tr, "cpu"); lid = sim.lidar; P = sim.P
    # sensor off-center, rotated 0.5 rad during the scan
    pose = torch.tensor([[2.0, 0.0, 0.0]]); prev = torch.tensor([[2.0, 0.0, -0.5]])
    _, r_static, _ = lid.scan(pose, None, P, motion_distortion=False, noisy=False)
    _, r_dist, _ = lid.scan(pose, prev, P, motion_distortion=True, noisy=False)
    # last beam is at scan time -> identical; first beams differ
    assert abs(r_static[0, -1] - r_dist[0, -1]) < 1e-4
    assert (r_static[0, :100] - r_dist[0, :100]).abs().max() > 0.1
