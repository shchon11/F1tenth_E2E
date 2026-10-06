"""Integration: a trivial test-only centerline follower must complete a lap without collision."""
import math
import torch

from f1sim import Track, Config, Simulator
from f1sim.mpc import PlanSpec


def centerline_pursuit(sim, state, lookahead=1.2, speed=2.5, k_us=PlanSpec.k_us):
    """Test-only pure pursuit on the centerline (not a reference planner). The steer for a curvature uses the
    effective wheelbase L + k_us v^2: the refitted car understeers like the real one (k_us measured 0.018-0.023),
    so the bare wheelbase under-steers by ~40 % at 2.5 m/s and runs wide into a hallway's walls."""
    s, _, _ = sim.track.project(state[:, :2], sim.tid)
    tgt, _ = sim.track.pose_at_s(s + lookahead, sim.tid)
    dx, dy = tgt[:, 0] - state[:, 0], tgt[:, 1] - state[:, 1]
    yaw = state[:, 2]
    lx = dx * torch.cos(yaw) + dy * torch.sin(yaw)
    ly = -dx * torch.sin(yaw) + dy * torch.cos(yaw)
    L = sim.cfg.vehicle.lf + sim.cfg.vehicle.lr + k_us * speed ** 2
    curv = 2 * ly / (lx ** 2 + ly ** 2 + 1e-6)
    steer = torch.atan(L * curv).clamp(-0.4, 0.4)
    return torch.stack([steer, torch.full_like(steer, speed)], 1)


def test_full_lap_on_random_tracks():
    # A hallway's square corners take the refitted car's full lock: at 2.5 m/s its tightest radius is 1.03 m (0.78 m
    # on the pre-2026-09-28 stiff-tyre car), and the centreline pursuit runs wide into the wall; 2.0 m/s laps.
    for seed, style, speed in ((0, "circuit", 2.5), (1, "competition", 2.5), (2, "hallway", 2.0)):
        tr = Track.generate_random(seed, style=style)
        cfg = Config(); cfg.rand.enabled = False
        sim = Simulator(tr, cfg, num_envs=4, device="cpu")
        s0 = torch.zeros(4)
        sim.reset(poses=sim.sample_spawn(4, lateral_std=0.0, yaw_std=0.0, s=s0))
        total = torch.zeros(4)
        for i in range(int(sim.track.length / speed / sim.control_dt * 1.3)):
            r = sim.step(centerline_pursuit(sim, r.state if i else sim.state, speed=speed))
            total += r.progress
            assert not r.collision.any(), f"seed {seed} collided at step {i}, s={r.s}"
            if (r.lap >= 1).all():
                break
        assert (r.lap >= 1).all(), (seed, r.lap, total)
        assert (total - sim.track.length).abs().max() < 1.0   # progress integrates to one lap


def test_odom_drifts_but_stays_plausible():
    tr = Track.generate_random(1, style="circuit")      # smooth track: drift is dominated by calibration, not hairpins
    cfg = Config()
    sim = Simulator(tr, cfg, num_envs=2, device="cpu")
    sim.reset(poses=sim.sample_spawn(2, 0.0, 0.0, torch.zeros(2)))
    r = None; dist = torch.zeros(2)
    for i in range(400):
        r = sim.step(centerline_pursuit(sim, r.state if r is not None else sim.state))
        dist += r.progress
    err = (r.odom[:, :2] - r.state[:, :2]).norm(dim=1)
    assert (err > 0.02).all(), err                 # it does drift ...
    assert (err < 0.3 * dist).all(), (err, dist)   # ... but not absurdly: the competition car's odom heading is
                                                   # the gyro's (OdomParams.yaw_from_gyro), so it drifts by speed error
