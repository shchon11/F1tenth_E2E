"""The LiDAR sees the same car the viewer draws: outlines sliced from f1tenth_car.glb."""
import math

import pytest
import torch

from f1sim import lidar as lidar_module
from f1sim.lidar import (car_slices, ray_slices_hits, _ray_slices_hits_gathered,
                         _ray_slices_hits_original_streamed, _ray_slices_hits_streamed)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _beam(origin_xyz, heading, k=0.0):
    o = torch.tensor([[origin_xyz]], dtype=torch.float32)          # (1,1,3)
    d = torch.tensor([[[math.cos(heading), math.sin(heading)]]])
    return o, d, torch.tensor([[k]])


def test_slices_match_the_mesh_footprint() -> None:
    z, segs, valid = car_slices("cpu")
    i = int(((z - 0.11).abs()).argmin())                           # the scan plane
    pts = segs[i][valid[i]].reshape(-1, 2)
    ext = pts.max(0).values - pts.min(0).values
    assert 0.38 < float(ext[0]) < 0.45 and 0.30 < float(ext[1]) < 0.36, ext   # wheels + chassis at 11 cm


def test_a_beam_from_behind_hits_the_rear_bumper() -> None:
    o, d, k = _beam((-2.0, 0.0, 0.11), 0.0)                         # 2 m behind a car at the origin, level
    poses = torch.zeros(1, 1, 3); scale = torch.ones(1, 1)
    r, hit = ray_slices_hits(o, d, k, poses, scale)
    # at 11 cm the rearmost surface is the chassis tail at x = -0.207 (the -0.242 bumper lip sits lower)
    assert bool(hit[0, 0]) and abs(float(r[0, 0]) - (2.0 - 0.207)) < 0.02, float(r[0, 0])


def test_a_beam_from_the_side_hits_the_wheel_face() -> None:
    """Aimed at the rear axle line: at 11 cm the widest thing is the wheels (y = +-0.168), and
    between the axles the scan plane clears the chassis plate -- a beam at x = 0 sees only the
    narrow deck, which is what the real car looks like from the side too."""
    o, d, k = _beam((-0.165, 2.0, 0.11), -math.pi / 2)              # 2 m to the left of the rear wheels
    r, hit = ray_slices_hits(o, d, k, torch.zeros(1, 1, 3), torch.ones(1, 1))
    assert bool(hit[0, 0]) and abs(float(r[0, 0]) - (2.0 - 0.168)) < 0.02, float(r[0, 0])


def test_a_beam_over_the_roof_passes() -> None:
    o, d, k = _beam((-2.0, 0.0, 0.30), 0.0)                         # scan plane above the LiDAR tower
    r, hit = ray_slices_hits(o, d, k, torch.zeros(1, 1, 3), torch.ones(1, 1))
    assert not bool(hit[0, 0])


def test_a_car_turned_sideways_is_wider_than_it_is_long() -> None:
    o, d, k = _beam((-2.0, 0.0, 0.11), 0.0)
    r_len, _ = ray_slices_hits(o, d, k, torch.zeros(1, 1, 3), torch.ones(1, 1))
    o2, d2, k2 = _beam((-2.0, 0.165, 0.11), 0.0)                    # along the sideways car's front-wheel line
    poses = torch.tensor([[[0.0, 0.0, math.pi / 2]]])
    r_side, _ = ray_slices_hits(o2, d2, k2, poses, torch.ones(1, 1))
    assert abs(float(r_side[0, 0]) - (2.0 - 0.168)) < 0.02 and float(r_side[0, 0]) > float(r_len[0, 0])


def _mesh_parity_case(device: str, num_cars: int):
    """Build one deterministic mix of valid, out-of-band, and arbitrary rays."""
    B, N = 3, 64
    gen = torch.Generator().manual_seed(20260917 + num_cars)

    poses = torch.empty(B, num_cars, 3)
    if num_cars:
        poses[..., :2] = torch.rand((B, num_cars, 2), generator=gen) * 2.4 - 1.2
        poses[..., 2] = torch.rand((B, num_cars), generator=gen) * (2.0 * math.pi) - math.pi
        scale = torch.rand((B, num_cars), generator=gen) * 0.7 + 0.65
    else:
        scale = torch.empty(B, 0)

    origin = torch.empty(B, N, 3)
    origin[..., :2] = torch.rand((B, N, 2), generator=gen) * 8.0 - 4.0
    origin[..., 2] = torch.rand((B, N), generator=gen) * 0.7 - 0.2
    angle = torch.rand((B, N), generator=gen) * (2.0 * math.pi) - math.pi
    dh = torch.stack((angle.cos(), angle.sin()), dim=-1)
    k = torch.rand((B, N), generator=gen) * 1.8 - 0.9

    if num_cars:
        # Three aimed beams per car use level, positive-slope, and negative-slope rays. The latter
        # two start above/below the mesh but reach the same 11 cm scan slice at the car centre.
        slopes = (0.0, 0.06, -0.07)
        for b in range(B):
            for c in range(num_cars):
                j = 3 * c
                yaw = float(poses[b, c, 2])
                forward = torch.tensor((math.cos(yaw), math.sin(yaw)))
                origin[b, j, :2] = poses[b, c, :2] - 2.0 * forward
                origin[b, j, 2] = 0.11 - 2.0 * slopes[c % len(slopes)]
                dh[b, j] = forward
                k[b, j] = slopes[c % len(slopes)]

        # These rays point away from the first car, so their below/above heights stay out of band.
        j = 3 * num_cars
        for offset, z, slope in ((0, -0.1, 0.0), (1, 0.3, 0.0),
                                 (2, -0.1, 0.05), (3, 0.3, -0.05)):
            origin[:, j + offset, :2] = poses[:, 0, :2] - 2.0 * torch.stack((
                poses[:, 0, 2].cos(), poses[:, 0, 2].sin()), dim=-1)
            origin[:, j + offset, 2] = z
            dh[:, j + offset] = -torch.stack((poses[:, 0, 2].cos(), poses[:, 0, 2].sin()), dim=-1)
            k[:, j + offset] = slope

    return tuple(t.to(device) for t in (origin, dh, k, poses, scale))


@pytest.mark.parametrize("num_cars", [0, 3], ids=["zero_cars", "three_cars"])
@pytest.mark.parametrize("device", DEVICES)
def test_gathered_mesh_hits_match_streamed_mesh_hits(device: str, num_cars: int) -> None:
    origin, dh, k, poses, scale = _mesh_parity_case(device, num_cars)
    gathered_range, gathered_hit = _ray_slices_hits_gathered(origin, dh, k, poses, scale)
    streamed_range, streamed_hit = _ray_slices_hits_streamed(origin, dh, k, poses, scale, chunk=7)

    # torch.equal deliberately checks every finite range bit-for-bit and also preserves inf parity.
    assert torch.equal(gathered_hit, streamed_hit)
    assert torch.equal(gathered_range, streamed_range)
    assert bool(torch.isinf(gathered_range).any())
    if num_cars:
        assert bool(torch.isfinite(gathered_range).any())
    else:
        assert bool(torch.isinf(gathered_range).all())


def _large_mesh_parity_case(device: str):
    """One beam tile boundary case with both finite and infinite mesh results."""
    B, N, C = 1, 4097, 2
    gen = torch.Generator().manual_seed(20260917)
    poses = torch.tensor([[[0.0, 0.0, 0.0], [0.7, -0.35, 0.65]]], dtype=torch.float32)
    scale = torch.tensor([[1.0, 0.85]], dtype=torch.float32)
    origin = torch.rand((B, N, 3), generator=gen) * 8.0 - 4.0
    origin[..., 2] = torch.rand((B, N), generator=gen) * 0.7 - 0.2
    angle = torch.rand((B, N), generator=gen) * (2.0 * math.pi) - math.pi
    dh = torch.stack((angle.cos(), angle.sin()), dim=-1)
    k = torch.rand((B, N), generator=gen) * 1.8 - 0.9
    for c, slope in enumerate((0.0, 0.06)):
        yaw = float(poses[0, c, 2])
        forward = torch.tensor((math.cos(yaw), math.sin(yaw)))
        origin[0, c, :2] = poses[0, c, :2] - 2.0 * forward
        origin[0, c, 2] = 0.11 - 2.0 * slope
        dh[0, c] = forward
        k[0, c] = slope
    # Keep an explicit no-hit ray so the comparison checks inf handling too.
    forward = torch.tensor((1.0, 0.0))
    origin[0, -1, :2] = poses[0, 0, :2] - 2.0 * forward
    origin[0, -1, 2] = 0.30
    dh[0, -1] = -forward
    k[0, -1] = 0.0
    return tuple(t.to(device) for t in (origin, dh, k, poses, scale))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_tiled_mesh_hits_match_original_streamed_mesh(device: str = "cuda") -> None:
    origin, dh, k, poses, scale = _large_mesh_parity_case(device)
    tiled_range, tiled_hit = ray_slices_hits(origin, dh, k, poses, scale)
    oracle_range, oracle_hit = _ray_slices_hits_original_streamed(origin, dh, k, poses, scale)

    assert torch.equal(tiled_hit, oracle_hit)
    assert torch.equal(tiled_range, oracle_range)
    assert bool(torch.isfinite(tiled_range).any())
    assert bool(torch.isinf(tiled_range).any())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("batch,beams,expected", [
    (1, 4096, [(1, 4096)]),
    (1, 4097, [(1, 4096), (1, 1)]),
    (4097, 1, [(4096, 1), (1, 1)]),
])
def test_cuda_mesh_dispatch_boundary(monkeypatch, batch: int, beams: int, expected) -> None:
    calls = []

    def fake_gathered(*args):
        calls.append(tuple(args[2].shape))
        k_arg = args[2]
        return torch.full_like(k_arg, float("inf")), torch.zeros_like(k_arg, dtype=torch.bool)

    monkeypatch.setattr(lidar_module, "_ray_slices_hits_gathered", fake_gathered)
    origin = torch.zeros(batch, beams, 3, device="cuda")
    dh = torch.zeros(batch, beams, 2, device="cuda")
    dh[..., 0] = 1.0
    k = torch.zeros(batch, beams, device="cuda")
    poses = torch.empty(batch, 0, 3, device="cuda")
    scale = torch.empty(batch, 0, device="cuda")

    ray_slices_hits(origin, dh, k, poses, scale)
    assert calls == expected
