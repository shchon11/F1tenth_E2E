"""The LiDAR sees the same car the viewer draws: outlines sliced from f1tenth_car.glb."""
import math
import torch

from f1sim.lidar import ray_slices_hits, car_slices


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
