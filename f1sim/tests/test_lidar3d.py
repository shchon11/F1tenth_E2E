"""3D beam geometry: floor returns under pitch, beams passing over duct hoses, tall objects outside."""
import math
import numpy as np
import torch
import pytest

from f1sim import Track, Config, Simulator

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def strip_track(res=0.05, size=16.0, duct_x=2.0, wall_x=6.0, duct_h=0.2):
    """Floor everywhere; a duct hose line at x=duct_x and a tall wall at x=wall_x (both spanning y)."""
    n = int(size / res)
    xs = (np.arange(n) - n / 2) * res
    gx, _ = np.meshgrid(xs, xs)
    duct = np.abs(gx - duct_x) <= duct_h / 2
    tall = np.abs(gx - wall_x) <= 0.1
    return Track.from_occupancy(duct | tall, res, (xs[0], xs[0]), duct=duct, tall=tall, duct_height=duct_h)


def make(track, device, **lidar):
    cfg = Config(); cfg.rand.enabled = False; cfg.lidar.motion_distortion = False
    cfg.lidar.noise_std = 0.0; cfg.lidar.dropout_prob = 0.0; cfg.lidar.spike_prob = 0.0
    for k, v in lidar.items():
        setattr(cfg.lidar, k, v)
    sim = Simulator(track, cfg, num_envs=1, device=device)
    sim.reset(poses=torch.zeros(1, 3))
    return sim


@pytest.mark.parametrize("device", DEVICES)
def test_pitch_hits_floor_at_analytic_range(device):
    sim = make(strip_track(duct_x=100, wall_x=100), device, mount_x=0.0, mount_z=0.15)
    pitch = math.radians(3.0)                                   # nose down
    att = torch.tensor([[0.0, pitch]], device=sim.device)
    _, r, typ = sim.lidar.scan(torch.zeros(1, 3, device=sim.device), None, sim.P, False, noisy=False, att=att)
    mid = sim.cfg.lidar.n_beams // 2
    assert typ[0, mid] == 3
    assert abs(r[0, mid].item() - 0.15 / math.tan(pitch)) < 1e-3   # sensor height drops to 0.15*cos(pitch)
    # beams at the sides are horizontal -> no floor, nothing within range. The delivered scan reads
    # the driver's no-return sentinel there, not inf: urg_node on this car emits 0xFFFF mm on every
    # recording checked, and a policy trained against inf would meet a number it had never seen.
    assert typ[0, 0] == 0
    noisy = sim.lidar.scan(torch.zeros(1, 3, device=sim.device), None, sim.P, False, True, att=att)[0]
    assert abs(noisy[0, 0].item() - sim.cfg.lidar.dropout_value) < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_beam_passes_over_duct_when_tilted_up(device):
    tr = strip_track()
    sim = make(tr, device, mount_x=0.0, mount_z=0.15)
    mid = sim.cfg.lidar.n_beams // 2
    z = torch.zeros(1, 3, device=sim.device)
    # level: hits the duct at ~2 m (round hose: at z=0.15 the surface is 1.3 cm behind the footprint edge)
    _, r0, t0 = sim.lidar.scan(z, None, sim.P, False, noisy=False, att=torch.zeros(1, 2, device=sim.device))
    assert t0[0, mid] == 1 and abs(r0[0, mid].item() - (2.0 - 0.1 + 0.0134)) < 1.5 * tr.resolution
    # nose up 5 deg: at x=2 the beam is at 0.15 + 2*tan(5deg) = 0.325 m > duct -> sees the tall wall at 6 m
    att = torch.tensor([[0.0, -math.radians(5.0)]], device=sim.device)
    _, r1, t1 = sim.lidar.scan(z, None, sim.P, False, noisy=False, att=att)
    assert t1[0, mid] == 2 and abs(r1[0, mid].item() - (6.0 - 0.1) / math.cos(math.radians(5))) < 2 * tr.resolution
    # nose down 5 deg: floor before the duct (0.15/tan5 = 1.71 m)
    att = torch.tensor([[0.0, math.radians(5.0)]], device=sim.device)
    _, r2, t2 = sim.lidar.scan(z, None, sim.P, False, noisy=False, att=att)
    assert t2[0, mid] == 3 and abs(r2[0, mid].item() - 0.15 / math.tan(math.radians(5))) < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_roll_tilts_side_beams(device):
    sim = make(strip_track(duct_x=100, wall_x=100), device, mount_x=0.0, mount_z=0.15)
    roll = math.radians(4.0)                                    # right side down
    att = torch.tensor([[roll, 0.0]], device=sim.device)
    _, r, typ = sim.lidar.scan(torch.zeros(1, 3, device=sim.device), None, sim.P, False, noisy=False, att=att)
    angles = sim.lidar.angles
    right = int(torch.argmin((angles + math.pi / 2).abs()))    # beam pointing to -y (right)
    left = int(torch.argmin((angles - math.pi / 2).abs()))
    assert typ[0, right] == 3 and abs(r[0, right].item() - 0.15 / math.tan(roll)) < 2e-3
    assert typ[0, left] == 0                                    # left side beam goes up: nothing


def test_triton_matches_torch_on_random_track():
    if not torch.cuda.is_available():
        pytest.skip("cuda")
    from f1sim import lidar as L
    tr = Track.generate_random(2)
    cfg = Config(); cfg.lidar.motion_distortion = False
    sim = Simulator(tr, cfg, num_envs=64, device="cuda")
    for _ in range(20):
        sim.step(torch.stack([torch.rand(64) * 0.6 - 0.3, torch.rand(64) * 6], 1).cuda())
    att = torch.randn(64, 2, device="cuda") * 0.05
    pose = sim.state[:, :3]
    _, a, ta = sim.lidar.scan(pose, None, sim.P, False, noisy=False, att=att)
    L.HAVE_TRITON = False
    try:
        _, b, tb = sim.lidar.scan(pose, None, sim.P, False, noisy=False, att=att)
    finally:
        L.HAVE_TRITON = True
    same = (ta == tb)
    assert same.float().mean() > 0.995
    assert ((a - b).abs()[same] > 0.05).float().mean() < 1e-3
    assert (ta == 3).any() and (ta == 1).any() and (ta == 2).any()      # floor, ducts and outside objects all present
