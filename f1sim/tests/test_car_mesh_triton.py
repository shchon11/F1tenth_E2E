"""The Triton car-mesh kernel against the eager gathered helper it replaces.

`lidar.ray_slices_hits` sends CUDA beams to `lidar_triton.car_mesh_hits_triton` (2026-10-27: the
eager tiles cost 93 ms a step at 256 envs in races of 4, 441 ms at 1024). Two levels of claim:

* **Portable, every GPU.** Wherever the kernel and the reference differ at all, both lie inside the
  float32 rounding band of the exact geometry (`float32_oracle.mesh_band`): a beam can change its
  hit only where it is within rounding of an outline endpoint, an edge, a level boundary or a
  parallel segment, and a range only by what the rounding of that beam's geometry allows.
* **Bit for bit, where measured** (`float32_oracle.BIT_EXACT`: sm_89 and sm_120). The kernel repeats
  the helper's float32 arithmetic operation for operation as explicit IEEE intrinsics, and there the
  comparison is `torch.equal` -- every range to the last bit, every hit and every inf.

The kernel's early-out is compared with the kernel without it, which must be bit-identical on any
GPU: both sides are this kernel, so no toolchain difference can excuse a mismatch.
"""
import math

import numpy as np
import pytest
import torch

from f1sim import lidar as lidar_module
from f1sim.lidar import (_car_mesh_tables, _ray_slices_hits_gathered, _ray_slices_hits_tiled,
                         car_slices, ray_slices_hits)
from f1sim.lidar_triton import HAVE_TRITON, car_mesh_hits_triton
from float32_oracle import bit_exact_here, flatten_beams, mesh_band, violations

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and HAVE_TRITON),
                                reason="needs CUDA and Triton")
DEV = "cuda"


def _reference(origin, dh, k, poses, scale):
    """The eager gathered helper, tiled when large (each beam is independent: tiling changes nothing)."""
    if k.numel() > lidar_module._GATHERED_RAY_LIMIT:
        return _ray_slices_hits_tiled(origin, dh, k, poses, scale)
    return _ray_slices_hits_gathered(origin, dh, k, poses, scale)


def _scan_case(B, C, seed, d_max=8.0, z_lo=0.10, z_hi=0.20, k_amp=0.05, axis_aligned=False, N=1081):
    """Scans as the simulator builds them: a 270 deg fan of N beams per env from an ego pose that
    slides a few cm over the scan, a pitch/roll slope fan, and C cars within d_max of the ego."""
    gen = torch.Generator().manual_seed(seed)
    ego = torch.rand((B, 3), generator=gen) * torch.tensor([20.0, 20.0, 2 * math.pi]) - torch.tensor([10.0, 10.0, math.pi])
    if axis_aligned:
        ego[:, 2] = torch.randint(0, 4, (B,), generator=gen).float() * (math.pi / 2)
    a = torch.linspace(-0.75 * math.pi, 0.75 * math.pi, N)
    yaw = ego[:, 2:3] + a[None]
    dh = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
    slide = torch.rand((B, 2), generator=gen) * 0.2 - 0.1
    f = torch.linspace(1.0, 0.0, N)[None]
    origin = torch.empty(B, N, 3)
    origin[..., 0] = ego[:, 0:1] - f * slide[:, 0:1]
    origin[..., 1] = ego[:, 1:2] - f * slide[:, 1:2]
    origin[..., 2] = (torch.rand((B, 1), generator=gen) * (z_hi - z_lo) + z_lo
                      + 0.002 * (torch.rand((B, N), generator=gen) - 0.5))
    pitch = (torch.rand((B, 1), generator=gen) * 2 - 1) * k_amp
    roll = (torch.rand((B, 1), generator=gen) * 2 - 1) * k_amp
    k = (-pitch * torch.cos(a)[None] + roll * torch.sin(a)[None]
         + 0.1 * k_amp * (torch.rand((B, N), generator=gen) * 2 - 1)).clamp(-k_amp, k_amp)
    d = torch.rand((B, C), generator=gen) * d_max
    brg = torch.rand((B, C), generator=gen) * 2 * math.pi
    poses = torch.empty(B, C, 3)
    poses[..., 0] = ego[:, 0:1] + d * torch.cos(brg)
    poses[..., 1] = ego[:, 1:2] + d * torch.sin(brg)
    poses[..., 2] = torch.rand((B, C), generator=gen) * 2 * math.pi - math.pi
    if axis_aligned:
        poses[..., 2] = torch.randint(0, 4, (B, C), generator=gen).float() * (math.pi / 2)
    scale = torch.rand((B, C), generator=gen) * 0.4 + 0.8
    return [t.to(DEV) for t in (origin, dh, k, poses, scale)]


def _wild_case(B, C, seed, N=1081):
    """Arbitrary origins and directions, steep slopes, heights in and out of the slice band."""
    gen = torch.Generator().manual_seed(seed)
    poses = torch.empty(B, C, 3)
    poses[..., :2] = torch.rand((B, C, 2), generator=gen) * 2.4 - 1.2
    poses[..., 2] = torch.rand((B, C), generator=gen) * 2 * math.pi - math.pi
    scale = torch.rand((B, C), generator=gen) * 0.7 + 0.65
    origin = torch.empty(B, N, 3)
    origin[..., :2] = torch.rand((B, N, 2), generator=gen) * 8.0 - 4.0
    origin[..., 2] = torch.rand((B, N), generator=gen) * 0.7 - 0.2
    ang = torch.rand((B, N), generator=gen) * 2 * math.pi - math.pi
    dh = torch.stack((ang.cos(), ang.sin()), -1)
    k = torch.rand((B, N), generator=gen) * 1.8 - 0.9
    return [t.to(DEV) for t in (origin, dh, k, poses, scale)]


def _graze_case(B, C, seed, N=1081):
    """Every beam aimed at an outline endpoint of the 11 cm slice (u = 0 or 1 to rounding), a third
    of them laid along the segment itself (den clamped to 1e-9 in the reference)."""
    z, segs, valid = car_slices("cpu")
    lv = 9
    nv = int(valid[lv].sum())
    origin, dh, k, poses, scale = [t.cpu() for t in _scan_case(B, C, seed, d_max=3.0, N=N)]
    gen = torch.Generator().manual_seed(seed + 1)
    c = torch.randint(0, C, (B, N), generator=gen)
    s = torch.randint(0, nv, (B, N), generator=gen)
    j = torch.arange(N)[None].expand(B, N)
    end = torch.where((j % 2 == 0)[..., None], segs[lv, s, 0:2], segs[lv, s, 2:4]).double()
    pc = torch.gather(poses, 1, c[..., None].expand(B, N, 3)).double()
    sc = torch.gather(scale, 1, c).double()
    cy, sy = torch.cos(pc[..., 2]), torch.sin(pc[..., 2])
    ex, ey = end[..., 0] * sc, end[..., 1] * sc
    wx, wy = pc[..., 0] + ex * cy - ey * sy, pc[..., 1] + ex * sy + ey * cy
    seg_d = (segs[lv, s, 2:4] - segs[lv, s, 0:2]).double()
    along = torch.atan2(seg_d[..., 1], seg_d[..., 0]) + pc[..., 2]
    ang = torch.where(j % 3 == 2, along, torch.rand((B, N), generator=gen, dtype=torch.float64) * 2 * math.pi)
    dist = 0.5 + 3.0 * torch.rand((B, N), generator=gen, dtype=torch.float64)
    origin[..., 0] = (wx - dist * torch.cos(ang)).float()
    origin[..., 1] = (wy - dist * torch.sin(ang)).float()
    origin[..., 2] = float(z[lv])
    dh = torch.stack([torch.cos(ang), torch.sin(ang)], -1).float()
    k = torch.zeros(B, N)
    return [t.to(DEV) for t in (origin, dh, k, poses, scale)]


def _assert_bit_identical(got, ref):
    (r, h), (r_ref, h_ref) = got, ref
    assert torch.equal(h, h_ref), f"{int((h != h_ref).sum())} beams disagree on hit"
    both = h & h_ref
    assert torch.equal(r, r_ref), (
        f"{int((r[both] != r_ref[both]).sum())} ranges differ, max {float((r[both] - r_ref[both]).abs().max()):.3g} m")


def _assert_matches_reference(got, ref, args):
    """Portable on every GPU: where the two differ at all, both sit in the float32 rounding band of the
    exact geometry. Bit-identical on the GPUs where that was measured."""
    (r, h), (r_ref, h_ref) = got, ref
    differ = (h != h_ref) | (h & h_ref & (r != r_ref))
    if bool(differ.any()):
        origin, dh, k, poses, scale = args
        band = mesh_band(*flatten_beams(differ, origin, dh, k, poses, scale))
        bad_k = violations(r[differ], h[differ], band)
        bad_r = violations(r_ref[differ], h_ref[differ], band)
        assert not bool(bad_k.any()), f"{int(bad_k.sum())} of {int(differ.sum())} differing beams: kernel outside the band"
        assert not bool(bad_r.any()), f"{int(bad_r.sum())} of {int(differ.sum())} differing beams: reference outside the band"
    if bit_exact_here():
        _assert_bit_identical(got, ref)


def _case(kind, B, C, seed):
    if kind == "scan":
        return _scan_case(B, C, seed)
    if kind == "scan_axis":                                      # axis-aligned cars and beams: den == 0 exactly
        return _scan_case(B, C, seed, axis_aligned=True)
    if kind == "scan_low":                                       # 1.5 .. 25 cm: both band edges
        return _scan_case(B, C, seed, z_lo=0.015, z_hi=0.25)
    if kind == "wild":
        return _wild_case(B, C, seed)
    return _graze_case(B, C, seed)


CASES = [("scan", 1, 1), ("scan", 7, 3), ("scan", 64, 3), ("scan_axis", 7, 3), ("scan_low", 64, 1),
         ("wild", 7, 1), ("wild", 64, 3), ("graze", 7, 3), ("graze", 32, 1)]


@pytest.mark.parametrize("kind,B,C", CASES, ids=[f"{k}-B{b}-C{c}" for k, b, c in CASES])
def test_triton_mesh_hits_match_the_gathered_helper(kind, B, C) -> None:
    args = _case(kind, B, C, 20261027 + 31 * B + C)
    ref = _reference(*args)
    got = ray_slices_hits(*args)                                 # the dispatch: Triton on CUDA
    _assert_matches_reference(got, ref, args)
    if B > 1:
        assert bool(ref[1].any()) and not bool(ref[1].all())     # the case has hits and misses


@pytest.mark.parametrize("kind", ["scan", "scan_low", "wild", "graze"])
def test_both_stay_within_float32_rounding_of_the_geometry(kind) -> None:
    """The band itself, on every beam: the reference and the kernel each report what the exact
    geometry says up to float32 rounding -- every robust hit hit, in range, and nothing where no
    hit is possible. The reference passing is what shows the band is not too tight."""
    args = _case(kind, 6, 3, 777)
    origin, dh, k, poses, scale = args
    band = mesh_band(*flatten_beams(torch.ones_like(k, dtype=torch.bool), origin, dh, k, poses, scale))
    must, may = band[0], band[1]
    for name, (r, h) in (("reference", _reference(*args)), ("kernel", ray_slices_hits(*args))):
        bad = violations(r.reshape(-1), h.reshape(-1), band)
        assert not bool(bad.any()), f"{name}: {int(bad.sum())} beams outside the band"
    assert int(must.sum()) > 10 and int((~may).sum()) > 10      # both kinds of beam are in the case
    if kind != "graze":                                          # graze aims every beam at an endpoint
        ambiguous = int((may & ~must).sum())
        assert ambiguous <= 0.02 * must.numel(), f"{ambiguous} ambiguous beams of {must.numel()}"


def test_the_early_out_never_changes_a_beam() -> None:
    """Beams whose line passes the outermost outline endpoint at offsets from -1e-5 m (through the
    outline) to +1e-2 m: the skip test (kernel comment) fires from about +1.2e-4 m, and on both
    sides of it the kernel with the early-out returns the bits of the kernel without it -- on any
    GPU, as both are this kernel -- and both match the reference."""
    z, segs, valid = car_slices("cpu")
    pts = torch.cat([segs[..., 0:2], segs[..., 2:4]], 1).double()
    r = pts.norm(dim=-1).masked_fill(~torch.cat([valid, valid], 1), -1.0)
    lv = int(r.max(1).values.argmax())
    P = pts[lv, int(r[lv].argmax())]
    offsets = torch.tensor([0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1.2e-4, 1.5e-4, 3e-4, 1e-3, 1e-2, -1e-7, -1e-6, -1e-5],
                           dtype=torch.float64)
    B, N = 16, 4 * len(offsets) * 8
    gen = torch.Generator().manual_seed(4242)
    poses = torch.zeros(B, 1, 3, dtype=torch.float64)
    poses[:, 0, :2] = torch.rand((B, 2), generator=gen, dtype=torch.float64) * 4 - 2
    poses[:, 0, 2] = torch.rand(B, generator=gen, dtype=torch.float64) * 2 * math.pi - math.pi
    scale = torch.rand((B, 1), generator=gen, dtype=torch.float64) * 0.4 + 0.8
    i = torch.arange(N)
    off = offsets[i % len(offsets)][None]                                   # (1,N)
    sgn = torch.where((i // len(offsets)) % 2 == 0, 1.0, -1.0).double()[None]
    away = ((i // (2 * len(offsets))) % 2 == 1)[None]                       # start past the tangent point
    Pc = P[None] * scale                                                    # (B,2) car frame
    Rn = Pc.norm(dim=-1, keepdim=True)
    nrm = Pc / Rn
    tang = torch.stack([-nrm[:, 1], nrm[:, 0]], -1)
    Q = nrm[:, None] * (Rn[:, None] + off[..., None])                       # (B,N,2) closest point of the line
    D = tang[:, None] * sgn[..., None]
    back = 0.3 + 2.0 * torch.rand((B, N, 1), generator=gen, dtype=torch.float64)
    O = torch.where(away[..., None], Q + back * D, Q - back * D)
    cy, sy = torch.cos(poses[:, :, 2]), torch.sin(poses[:, :, 2])            # (B,1)
    origin = torch.empty(B, N, 3, dtype=torch.float64)
    origin[..., 0] = poses[:, :, 0] + O[..., 0] * cy - O[..., 1] * sy
    origin[..., 1] = poses[:, :, 1] + O[..., 0] * sy + O[..., 1] * cy
    origin[..., 2] = float(z[lv])
    dh = torch.stack([D[..., 0] * cy - D[..., 1] * sy, D[..., 0] * sy + D[..., 1] * cy], -1)
    args = [t.float().to(DEV) for t in (origin, dh, torch.zeros(B, N), poses, scale)]
    ref = _reference(*args)
    with_eo = car_mesh_hits_triton(*args, _car_mesh_tables(DEV), early_out=True)
    without = car_mesh_hits_triton(*args, _car_mesh_tables(DEV), early_out=False)
    _assert_bit_identical(with_eo, without)
    _assert_matches_reference(with_eo, ref, args)
    hit_by_offset = {float(o): int(ref[1][:, (i % len(offsets) == n).to(DEV)].sum()) for n, o in enumerate(offsets)}
    assert hit_by_offset[-1e-5] > 0 and hit_by_offset[1e-4] == 0 and hit_by_offset[1e-2] == 0, hit_by_offset


def test_the_early_out_never_changes_a_scan() -> None:
    """The same, on ordinary scans and the wild mix: with and without the early-out, the same bits."""
    for kind, B, C in (("scan", 32, 3), ("scan_low", 16, 3), ("wild", 16, 3), ("graze", 8, 3)):
        args = _case(kind, B, C, 99 + B)
        _assert_bit_identical(car_mesh_hits_triton(*args, _car_mesh_tables(DEV), early_out=True),
                              car_mesh_hits_triton(*args, _car_mesh_tables(DEV), early_out=False))


def test_no_cars_and_no_beams() -> None:
    origin, dh, k, poses, scale = _scan_case(3, 1, 7)
    r, h = ray_slices_hits(origin, dh, k, poses[:, :0], scale[:, :0])
    assert bool(torch.isinf(r).all()) and not bool(h.any()) and r.dtype == torch.float32 and h.dtype == torch.bool
    r, h = ray_slices_hits(origin[:0], dh[:0], k[:0], poses[:0], scale[:0])
    assert r.shape == (0, 1081) and h.shape == (0, 1081)


def test_cuda_float32_dispatches_to_triton(monkeypatch) -> None:
    calls = []
    real = lidar_module.car_mesh_hits_triton

    def spy(*args, **kw):
        calls.append(tuple(args[2].shape))
        return real(*args, **kw)

    monkeypatch.setattr(lidar_module, "car_mesh_hits_triton", spy)
    origin, dh, k, poses, scale = _scan_case(2, 3, 11)
    ray_slices_hits(origin, dh, k, poses, scale)
    assert calls == [(2, 1081)]
    monkeypatch.setattr(lidar_module, "MESH_TRITON", False)            # the eager tiles stay reachable
    ray_slices_hits(origin, dh, k, poses, scale)
    assert calls == [(2, 1081)]


def test_a_race_scan_matches_with_and_without_the_kernel(monkeypatch) -> None:
    """Lidar.scan in a race of 4 on an open floor, noise off. The scan merges the car hits into the
    trace and nothing else differs between the two runs, so the scans agree on every beam where the
    two car-mesh answers agree -- and those are checked like any other (`_assert_matches_reference`)
    on the very inputs the scan built. Bit-identical scans where that was measured."""
    from f1sim import Config, Simulator
    from f1sim.track import Track
    occ = np.zeros((400, 400), bool); occ[:2] = occ[-2:] = True; occ[:, :2] = occ[:, -2:] = True
    track = Track.from_occupancy(occ, 0.05, (-10.0, -10.0), name="open")
    cfg = Config(); cfg.sim.compile = False
    sim = Simulator(track, cfg, num_envs=8, device=DEV, race_size=4)
    gen = torch.Generator().manual_seed(5)
    poses = torch.zeros(8, 3)
    poses[:, 0] = torch.rand(8, generator=gen) * 6 - 3
    poses[:, 1] = torch.rand(8, generator=gen) * 6 - 3
    poses[:, 2] = torch.rand(8, generator=gen) * 2 * math.pi
    sim.reset(torch.arange(8, device=DEV), poses=poses.to(DEV), speed=torch.zeros(8, device=DEV))
    sim.step(torch.zeros(8, 2, device=DEV))
    cars = sim._car_boxes(sim.state)
    assert cars[0][0] == "mesh"
    seen = []
    real = lidar_module.ray_slices_hits

    def spy(*a, **kw):
        out_ = real(*a, **kw)
        seen.append((a, out_))
        return out_

    monkeypatch.setattr(lidar_module, "ray_slices_hits", spy)
    out = {}
    for flag in (False, True):
        monkeypatch.setattr(lidar_module, "MESH_TRITON", flag)
        out[flag] = sim.lidar.scan(sim.state[:, :3], sim.pose_prev, sim.P, True, noisy=False,
                                   att=sim.att[:, :3:2], att_prev=sim.att_prev, tid=sim.tid, cars=cars, eid=sim.eid)
    (args_ref, mesh_ref), (args_tri, mesh_tri) = seen
    assert all(torch.equal(x, y) for x, y in zip(args_ref, args_tri))   # the same beams, both runs
    _assert_matches_reference(mesh_tri, mesh_ref, args_tri)
    same = (mesh_tri[1] == mesh_ref[1]) & ((mesh_tri[0] == mesh_ref[0]) | ~mesh_ref[1])
    for i in (1, 2):                                                    # ranges_true, hit types
        assert torch.equal(out[True][i][same], out[False][i][same])
    if bit_exact_here():
        assert torch.equal(out[True][1], out[False][1]) and torch.equal(out[True][2], out[False][2])
    assert int((out[True][2] == lidar_module.HIT_CAR).sum()) > 0
