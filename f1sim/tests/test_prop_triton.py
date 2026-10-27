"""The Triton prism kernel against the eager `prop_math.ray_prisms_hits` it replaces on CUDA.

`Lidar._merge_props` sends CUDA beams to `prop_triton.prism_hits_triton` (2026-10-27: the eager
function was 69.6 ms of a 74 ms scan at 1024 envs, races of 4, procedural props). As for the car
mesh (`test_car_mesh_triton.py`), two levels of claim:

* **Portable, every GPU.** Wherever the kernel and the reference differ at all, both lie inside the
  float32 rounding band of the exact geometry (`float32_oracle.prism_band`): a hit can change only
  where a beam is within rounding of a face, an edge or a top, starts on a prism, or runs parallel to
  a face; a range only by what that beam's rounding allows.
* **Bit for bit, where measured** (`float32_oracle.BIT_EXACT`: sm_89 and sm_120): `torch.equal`.
"""
import math

import numpy as np
import pytest
import torch

from f1sim import lidar as lidar_module
from f1sim.procedural_obstacles import build_catalogue
from f1sim.prop_math import ray_prisms_hits, section_halfplanes
from f1sim.prop_triton import HAVE_TRITON, prism_hits_triton
from float32_oracle import bit_exact_here, flatten_beams, prism_band, violations

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and HAVE_TRITON),
                                reason="needs CUDA and Triton")
DEV = "cuda"
K_PAD = 8
_SHAPES = build_catalogue(K_PAD)[0]


# ---------------------------------------------------------------------------------- cases
def _polygon(gen):
    """A convex CCW polygon: 3..K_PAD vertices on an ellipse at sorted random angles."""
    m = int(torch.randint(3, K_PAD + 1, (1,), generator=gen))
    while True:
        ang = torch.sort(torch.rand(m, generator=gen, dtype=torch.float64) * 2 * math.pi).values
        gaps = torch.diff(torch.cat([ang, ang[:1] + 2 * math.pi]))
        if float(gaps.min()) > 0.25 and float(gaps.max()) < math.pi - 0.05:
            break
    a, b = (0.05 + 0.3 * float(torch.rand((), generator=gen)) for _ in range(2))
    return np.stack([a * np.cos(ang.numpy()), b * np.sin(ang.numpy())], 1)


def _vertices(n, d):
    """Polygon vertices from ordered half-planes (prop frame), float64."""
    real = np.isfinite(d)
    nn, dd = n[real].astype(np.float64), d[real].astype(np.float64)
    out = []
    for i in range(len(nn)):
        a1, a2, d1, d2 = nn[i], nn[(i + 1) % len(nn)], dd[i], dd[(i + 1) % len(nn)]
        det = a1[0] * a2[1] - a1[1] * a2[0]
        out.append(((d1 * a2[1] - a1[1] * d2) / det, (a1[0] * d2 - d1 * a2[0]) / det))
    return np.array(out)


def _slots(B, C, gen, live_frac=0.7):
    """C slots per env: catalogue props, random convex polygons (some floating bands), and dead slots
    of all three kinds -- all padding, `z_hi <= z_lo`, and fewer than three real faces."""
    pn = np.zeros((B, C, K_PAD, 2), np.float32)
    pd = np.full((B, C, K_PAD), np.inf, np.float32)
    zlo, zhi = np.zeros((B, C), np.float32), np.zeros((B, C), np.float32)
    live = []
    kinds = torch.rand((B, C), generator=gen).numpy()
    for b in range(B):
        for c in range(C):
            u = kinds[b, c]
            if u > live_frac:
                v = (u - live_frac) / (1 - live_frac)
                if v >= 0.5:
                    pn[b, c], pd[b, c] = section_halfplanes(_polygon(gen), K_PAD)
                    if v < 0.75:
                        zlo[b, c], zhi[b, c] = 0.3, 0.3
                    else:
                        zhi[b, c] = 0.3
                        pn[b, c, 2:], pd[b, c, 2:] = 0.0, np.inf
                continue
            if u < live_frac / 2:
                sh = _SHAPES[int(torch.randint(0, len(_SHAPES), (1,), generator=gen))]
                pn[b, c], pd[b, c], zlo[b, c], zhi[b, c] = sh.n, sh.d, sh.z0, sh.z1
            else:
                pn[b, c], pd[b, c] = section_halfplanes(_polygon(gen), K_PAD)
                lo = 0.0 if float(torch.rand((), generator=gen)) < 0.6 else 0.05 + 0.15 * float(torch.rand((), generator=gen))
                zlo[b, c], zhi[b, c] = lo, lo + 0.08 + 0.35 * float(torch.rand((), generator=gen))
            live.append((b, c))
    return pn, pd, zlo, zhi, live


def _case(kind, B, C, seed, N=1081):
    """A 270 deg fan of N beams per env at 10-20 cm (1.5-50 cm for "low") with +-0.05 slopes, C slots
    0.6-8 m away ("far": 20 m more). "graze" aims every other beam at a vertex or along a face,
    "inside" starts them inside a prism, "top" drops them onto a prism's top face (k < 0)."""
    gen = torch.Generator().manual_seed(seed)
    ego = torch.rand((B, 3), generator=gen) * torch.tensor([20.0, 20.0, 2 * math.pi]) - torch.tensor([10.0, 10.0, math.pi])
    if kind == "axis":
        ego[:, 2] = torch.randint(0, 4, (B,), generator=gen).float() * (math.pi / 2)
    a = torch.linspace(-0.75 * math.pi, 0.75 * math.pi, N)
    yaw = ego[:, 2:3] + a[None]
    dh = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
    origin = torch.empty(B, N, 3)
    origin[..., 0], origin[..., 1] = ego[:, 0:1].expand(B, N).clone(), ego[:, 1:2].expand(B, N).clone()
    z_lo, z_hi = (0.015, 0.5) if kind == "low" else (0.10, 0.20)
    origin[..., 2] = torch.rand((B, 1), generator=gen) * (z_hi - z_lo) + z_lo
    pitch, roll = ((torch.rand((B, 1), generator=gen) * 2 - 1) * 0.05 for _ in range(2))
    k = (-pitch * torch.cos(a)[None] + roll * torch.sin(a)[None]).clamp(-0.05, 0.05)
    pn, pd, zlo, zhi, live = _slots(B, C, gen)
    d = 0.6 + torch.rand((B, C), generator=gen) * ((3.0 if kind in ("graze", "inside", "top") else 8.0) - 0.6)
    if kind == "far":
        d = d + 20.0
    brg = torch.rand((B, C), generator=gen) * 2 * math.pi
    poses = torch.stack([ego[:, 0:1] + d * torch.cos(brg), ego[:, 1:2] + d * torch.sin(brg),
                         torch.rand((B, C), generator=gen) * 2 * math.pi - math.pi], -1)
    if kind == "axis":
        poses[..., 2] = torch.randint(0, 4, (B, C), generator=gen).float() * (math.pi / 2)
    if kind in ("graze", "inside", "top"):
        for b in range(B):
            mine = [c for (bb, c) in live if bb == b]
            for j in range(0, N, 2) if mine else ():
                c = mine[int(torch.randint(0, len(mine), (1,), generator=gen))]
                verts = _vertices(pn[b, c], pd[b, c])
                px, py, pyaw = (float(v) for v in poses[b, c])
                cy, sy = math.cos(pyaw), math.sin(pyaw)
                w = lambda p: (px + p[0] * cy - p[1] * sy, py + p[0] * sy + p[1] * cy)   # noqa: E731
                r = lambda: float(torch.rand((), generator=gen))                         # noqa: E731
                if kind == "inside":
                    origin[b, j, 0], origin[b, j, 1] = w(verts.mean(0) * r() * 0.9)
                    origin[b, j, 2] = float(zlo[b, c]) + (float(zhi[b, c]) - float(zlo[b, c])) * r()
                    continue
                if kind == "graze":
                    vi = int(torch.randint(0, len(verts), (1,), generator=gen))
                    vx, vy = w(verts[vi])
                    e = verts[(vi + 1) % len(verts)] - verts[vi]
                    ang = math.atan2(e[1], e[0]) + pyaw if (j // 2) % 3 == 2 else r() * 2 * math.pi
                    dist = 0.4 + 3.0 * r()
                    origin[b, j, 2] = float(zlo[b, c]) + 0.5 * (float(zhi[b, c]) - float(zlo[b, c]))
                    kk = 0.0
                else:                                                                      # "top"
                    tgt = verts.mean(0) if (j // 2) % 2 == 0 else verts[int(torch.randint(0, len(verts), (1,), generator=gen))]
                    vx, vy = w(tgt)
                    ang, dist, h = r() * 2 * math.pi, 0.5 + 2.5 * r(), 0.02 + 0.3 * r()
                    origin[b, j, 2] = float(zhi[b, c]) + h
                    kk = -h / dist
                origin[b, j, 0], origin[b, j, 1] = vx - dist * math.cos(ang), vy - dist * math.sin(ang)
                dh[b, j, 0], dh[b, j, 1] = math.cos(ang), math.sin(ang)
                k[b, j] = kk
    t = lambda x: torch.as_tensor(x, dtype=torch.float32).to(DEV)                          # noqa: E731
    return [origin.to(DEV), dh.to(DEV), k.to(DEV), poses.to(DEV), t(pn), t(pd), t(zlo), t(zhi)]


# ---------------------------------------------------------------------------------- checks
def _assert_bit_identical(got, ref):
    (r, h), (r_ref, h_ref) = got, ref
    assert torch.equal(h, h_ref), f"{int((h != h_ref).sum())} beams disagree on hit"
    assert torch.equal(r, r_ref), f"{int(((r != r_ref) & h & h_ref).sum())} ranges differ"


def _assert_matches_reference(got, ref, args):
    """Portable on every GPU: where the two differ at all, both sit in the float32 rounding band of the
    exact geometry. Bit-identical on the GPUs where that was measured."""
    (r, h), (r_ref, h_ref) = got, ref
    differ = (h != h_ref) | (r != r_ref)
    if bool(differ.any()):
        band = prism_band(*flatten_beams(differ, *args[:3], *args[3:]))
        for name, (rr, hh) in (("kernel", (r, h)), ("reference", (r_ref, h_ref))):
            bad = violations(rr[differ], hh[differ], band)
            assert not bool(bad.any()), f"{int(bad.sum())} of {int(differ.sum())} differing beams: {name} outside the band"
    if bit_exact_here():
        _assert_bit_identical(got, ref)


CASES = [(kind, B, C) for kind in ("scan", "axis", "low", "far") for B, C in ((1, 1), (4, 8), (16, 26))] + \
        [(kind, B, C) for kind in ("graze", "inside", "top") for B, C in ((2, 8), (4, 26))]


@pytest.mark.parametrize("kind,B,C", CASES, ids=[f"{k}-B{b}-C{c}" for k, b, c in CASES])
def test_triton_prisms_match_the_eager_reference(kind, B, C) -> None:
    args = _case(kind, B, C, 20261027 + 17 * B + C)
    ref = ray_prisms_hits(*args)
    got = prism_hits_triton(*args)
    _assert_matches_reference(got, ref, args)
    if kind in ("graze", "inside", "top") or B * C >= 32:
        assert bool(ref[1].any())                                   # the case reaches the props at all


@pytest.mark.parametrize("kind", ["scan", "low", "graze", "inside", "top"])
def test_both_stay_within_float32_rounding_of_the_geometry(kind) -> None:
    """The band on every beam: each implementation reports what the exact geometry says up to float32
    rounding. The reference passing is what shows the band is not too tight."""
    args = _case(kind, 4, 26, 555)
    band = prism_band(*flatten_beams(torch.ones_like(args[2], dtype=torch.bool), *args[:3], *args[3:]))
    must, may = band[0], band[1]
    for name, (r, h) in (("reference", ray_prisms_hits(*args)), ("kernel", prism_hits_triton(*args))):
        bad = violations(r.reshape(-1), h.reshape(-1), band)
        assert not bool(bad.any()), f"{name}: {int(bad.sum())} beams outside the band"
    assert int(must.sum()) > 10 and int((~may).sum()) > 10           # both kinds of beam are in the case
    if kind in ("scan", "low", "inside"):                          # graze/top aim beams at vertices on purpose
        ambiguous = int((may & ~must).sum())
        assert ambiguous <= 0.02 * must.numel(), f"{ambiguous} ambiguous beams of {must.numel()}"


def test_dead_slots_and_padding_are_nothing() -> None:
    """A batch whose every slot is dead -- all padding, flat (`z_hi <= z_lo`), or two faces short --
    returns no hit anywhere, as the reference does; and so does a batch with no slots at all."""
    origin, dh, k, poses, pn, pd, zlo, zhi = _case("scan", 4, 8, 3)
    flat = (zhi * 0, zhi * 0)
    for pn_, pd_, (lo, hi) in ((pn * 0, torch.full_like(pd, float("inf")), flat),
                               (pn, pd, (zhi, zhi)),
                               (torch.cat([pn[..., :2, :], pn[..., 2:, :] * 0], -2),
                                torch.cat([pd[..., :2], torch.full_like(pd[..., 2:], float("inf"))], -1), (zlo, zhi))):
        args = (origin, dh, k, poses, pn_, pd_, lo, hi)
        r, h = prism_hits_triton(*args)
        assert not bool(h.any()) and bool(torch.isinf(r).all())
        _assert_bit_identical((r, h), ray_prisms_hits(*args))
    r, h = prism_hits_triton(origin, dh, k, poses[:, :0], pn[:, :0], pd[:, :0], zlo[:, :0], zhi[:, :0])
    assert not bool(h.any()) and bool(torch.isinf(r).all())


def test_only_inside_hit_is_implemented() -> None:
    with pytest.raises(ValueError):
        prism_hits_triton(*_case("scan", 1, 1, 0), inside="miss")


def test_merge_props_dispatches_to_triton_on_cuda(monkeypatch) -> None:
    """`Lidar._merge_props` on CUDA float32 runs the kernel; with `PROP_TRITON` off, `Lidar._prisms`."""
    origin, dh, k, poses, pn, pd, zlo, zhi = _case("scan", 2, 8, 4)
    calls = []

    class Track:
        has_props = True

        def props_for(self, tid, eid=None):
            return poses, pn, pd, zlo, zhi

    class Lid:
        _merge_props = lidar_module.Lidar._merge_props
        track = Track()

        def _prisms(self, *a):
            calls.append("eager")
            return ray_prisms_hits(*a)

    real = lidar_module.prism_hits_triton
    monkeypatch.setattr(lidar_module, "prism_hits_triton", lambda *a, **kw: calls.append("triton") or real(*a, **kw))
    r0 = torch.full(k.shape, 30.0, device=DEV)
    typ0 = torch.zeros(k.shape, dtype=torch.int32, device=DEV)
    rmax = torch.full((k.shape[0],), 30.0, device=DEV)
    tid = torch.zeros(k.shape[0], dtype=torch.long, device=DEV)
    out_t = Lid()._merge_props(origin, dh, k, rmax, tid, r0, typ0)
    monkeypatch.setattr(lidar_module, "PROP_TRITON", False)
    out_e = Lid()._merge_props(origin, dh, k, rmax, tid, r0, typ0)
    assert calls == ["triton", "eager"]
    assert int((out_t[1] == lidar_module.HIT_TALL).sum()) > 0
    if bit_exact_here():
        assert torch.equal(out_t[0], out_e[0]) and torch.equal(out_t[1], out_e[1])


def test_the_kernel_replays_in_a_cuda_graph() -> None:
    """Captured once, replayed on new inputs written into the same buffers: the eager reference's
    answer each time (the scan's `_merge_props` is a captured leaf in the console fast path)."""
    static = _case("scan", 8, 26, 6)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            prism_hits_triton(*static)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_r, out_h = prism_hits_triton(*static)
    for seed in (7, 8, 9):
        fresh = _case("scan", 8, 26, seed)
        for dst, src in zip(static, fresh):
            dst.copy_(src)
        g.replay()
        torch.cuda.synchronize()
        _assert_matches_reference((out_r, out_h), ray_prisms_hits(*fresh), fresh)
        assert bool(out_h.any())


def test_a_scan_with_procedural_props_matches_with_and_without_the_kernel(monkeypatch) -> None:
    """`Lidar.scan` of a race of 4 with a procedural layout (training's obstacle settings), noise off,
    with the kernel and with `Lidar._prisms`. The scans agree on every beam where the two prism answers
    agree, and those answers are checked like any other, on the inputs the scan built."""
    from f1sim.gym_env import EnvConfig, F1VecEnv
    from f1sim.params import Config
    from test_procedural_obstacles import ring_track
    cfg = Config(); cfg.sim.compile = False; cfg.imu.enabled = False
    ec = EnvConfig(compile_tracker=False, max_steps=400, race_size=4, procedural_obstacles=1.0,
                   procedural_density=1.5, procedural_max_props=18, collision_mode="soft",
                   movable_obstacles=True)
    env = F1VecEnv([ring_track()], cfg, ec, num_envs=8, device=DEV)
    env.reset(seed=3)
    sim = env.sim
    seen = []
    real_t, real_e = lidar_module.prism_hits_triton, sim.lidar._prisms
    monkeypatch.setattr(lidar_module, "prism_hits_triton", lambda *a, **kw: seen.append(("t", a)) or real_t(*a, **kw))
    monkeypatch.setattr(sim.lidar, "_prisms", lambda *a: seen.append(("e", a)) or real_e(*a))
    out = {}
    for flag in (False, True):
        monkeypatch.setattr(lidar_module, "PROP_TRITON", flag)
        out[flag] = sim.lidar.scan(sim.state[:, :3], None, sim.P, motion_distortion=False, noisy=False,
                                   tid=sim.tid, eid=sim.eid)
    (ke, ae), (kt, at) = seen
    assert (ke, kt) == ("e", "t") and all(torch.equal(x, y) for x, y in zip(ae, at))
    got, ref = real_t(*at), ray_prisms_hits(*ae)
    assert int(ref[1].sum()) > 100                                  # the layout is in view (walls are HIT_TALL too)
    _assert_matches_reference(got, ref, at)
    same = (got[1] == ref[1]) & (got[0] == ref[0])
    for i in (1, 2):
        assert torch.equal(out[True][i][same], out[False][i][same])
    if bit_exact_here():
        assert torch.equal(out[True][1], out[False][1]) and torch.equal(out[True][2], out[False][2])
    assert int((out[True][2] == lidar_module.HIT_TALL).sum()) > 0
