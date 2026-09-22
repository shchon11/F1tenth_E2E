"""Obstacle layouts redrawn per env at every reset (`f1sim.procedural_obstacles`).

What is worth testing here is not that props exist -- `test_prop_sim.py` already establishes that a
prop collides and that `prop_math`'s primitives are right -- but the four claims this feature makes
that nothing else can check:

1. off is off: with `EnvConfig.procedural_obstacles == 0` nothing is allocated, nothing is drawn
   from the generator and a run is bit-identical to the same run before any of this existed. The
   reference digest is recorded from the merge base and pinned here.
2. the layout really is new at every reset, and really is a function of `sim.gen` alone.
3. the gap is there by construction: measured on the catalogue maps, over 1000 draws, on the
   geometry actually standing on the track rather than on the arithmetic that placed it.
4. the teacher opponents are never routed through one.
"""
from __future__ import annotations

import ast
import inspect
import math

import numpy as np
import pytest
import torch

from f1sim import hard_obstacles as hard
from f1sim import procedural_obstacles as po
from f1sim import props as props_mod
from f1sim.gym_env import EnvConfig, F1VecEnv
from f1sim.params import Config
from f1sim.prop_math import _polygon_vertices, _to_world, section_halfplanes
from f1sim.track import Track


# ==================================================================== fixtures
def ring_track(name="ring", R=8.0, w=1.30, res=0.05) -> Track:
    """A circular lane of half-width `w` between two walls. Cheap, and its lane width is known."""
    N = int((2 * R + 4.0) / res)
    ys, xs = np.meshgrid(np.arange(N) * res, np.arange(N) * res, indexing="ij")
    c = R + 2.0
    r = np.hypot(xs - c, ys - c)
    occ = (r > R + w) | (r < R - w)
    a = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    cl = np.stack([c + R * np.cos(a), c + R * np.sin(a)], 1)
    return Track.from_occupancy(occ, res, (0.0, 0.0), cl, name, duct=np.zeros_like(occ), tall=occ.copy())


def cpu_cfg(n_beams=64) -> Config:
    cfg = Config()
    cfg.sim.device = "cpu"
    cfg.sim.compile = False
    cfg.lidar.n_beams = n_beams
    cfg.imu.enabled = False
    return cfg


def make_env(tracks, B=8, seed=0, **kw) -> F1VecEnv:
    cfg = cpu_cfg(kw.pop("n_beams", 64))
    cfg.sim.seed = seed
    ec = EnvConfig(compile_tracker=False, max_steps=400, **kw)
    return F1VecEnv(tracks if isinstance(tracks, list) else [tracks], cfg, ec, num_envs=B, device="cpu")


@pytest.fixture(scope="module")
def catalogue_tracks():
    """Two real catalogue maps: a long twisty one and a generated one."""
    from f1sim.learn import common
    trs, _ = common.load_tracks(common.track_names("real:blackbox2022_1,gen:control:1400"))
    return trs


# ==================================================================== 1. the rules are the same rules
def test_pattern_rules_are_hard_obstacles_rules():
    """The numbers are the user's scenes, via `hard_obstacles`. Copied, so pinned equal."""
    assert po.PATTERNS == hard.PATTERNS
    assert po.GAP_FRAC == hard.GAP_FRAC
    assert po.GAP_MIN == hard.GAP_MIN
    assert po.ROW_GAP == hard.ROW_GAP
    assert po.SMALL_SHARE == hard.SMALL_SHARE
    assert (po.BOX_W, po.BOX_D) == (hard.BOX_W, hard.BOX_D)


def test_catalogue_pieces_are_the_props_they_name():
    """Every catalogue slot is exactly `props.build(...)`'s declared envelope, not a box drawn here."""
    k_pad = po.catalogue_k_pad()
    shapes, row_ids, small_ids = po.build_catalogue(k_pad)
    assert shapes, "empty catalogue"
    for sh in shapes:
        prop = props_mod.build(sh.style, seed=0, **dict(sh.dims))
        n, d = section_halfplanes(prop.envelope.footprint, k_pad)
        assert np.array_equal(sh.n, n)
        assert np.array_equal(sh.d, d)
        assert sh.z1 == pytest.approx(prop.envelope.height)
        fp = np.asarray(prop.envelope.footprint)
        # `across` is the across-lane extent inflated by the yaw jitter, so it bounds the rotated
        # piece -- which is what the gap arithmetic subtracts.
        nominal = float(fp[:, 0].max() - fp[:, 0].min())
        assert sh.across >= nominal
        assert sh.across <= nominal + float(fp[:, 1].max() - fp[:, 1].min()) * math.sin(po.YAW_JITTER) + 1e-9
    assert sorted(row_ids + small_ids) == list(range(len(shapes)))
    # the small set is small: the user's point is that avoiding only big boxes is not avoiding
    assert max(shapes[i].across for i in small_ids) <= 0.30


def test_row_ladder_is_sorted():
    _, row_ids, _ = po.build_catalogue(po.catalogue_k_pad())
    shapes, _, _ = po.build_catalogue(po.catalogue_k_pad())
    across = [shapes[i].across for i in row_ids]
    assert across == sorted(across)


# ==================================================================== 2. off is off
def _digest(procedural: float) -> str:
    """A fixed rollout reduced to one hash: observations, rewards, flags and the generator state."""
    import hashlib
    kw = {} if procedural <= 0 else {"procedural_obstacles": procedural}
    env = make_env([ring_track("a"), ring_track("b", R=6.5, w=1.15)], B=16, seed=7, **kw)
    obs, _ = env.reset(seed=7)
    g = torch.Generator().manual_seed(11)
    h = hashlib.sha256()
    for _ in range(40):
        act = torch.rand(env.B, env.act_dim, generator=g) * 2 - 1
        obs, rew, term, trunc, info = env.step(act)
        for t in (obs["scan"], obs["speed"], rew, term, trunc):
            h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
    h.update(env.sim.gen.get_state().numpy().tobytes())
    return h.hexdigest()


#: `_digest(0.0)` on the merge base (main `4209ec2`, before this branch), recorded 2026-09-13 by
#: running the same function against a `git archive` of that commit. It is the whole of what
#: "defaults to byte-identical off" means, so it is pinned rather than compared against a second
#: run of the same code, which would pass however far the off path had drifted.
BASE_DIGEST = "7a0f4024a0daec49c60f2e54d437a3c7f1bc4df95b3786b00a7422f9328f162e"


def test_off_is_byte_identical_to_the_merge_base():
    assert _digest(0.0) == BASE_DIGEST


def test_off_allocates_nothing():
    env = make_env(ring_track(), B=4)
    assert env.procedural is None
    assert env.sim.track.env_props is None
    assert not getattr(env.sim.track, "has_props", False)
    tid = env.sim.tid
    poses, pn, pd, zlo, zhi = env.sim.track.props_for(tid)
    assert poses.shape[1] == 0 and pn.shape[1] == 0


def test_on_changes_the_rollout():
    """A guard on the guard: if the digest above matched with the option *on*, it would prove nothing."""
    assert _digest(1.0) != BASE_DIGEST


# ==================================================================== 3. fresh at every reset
def _live_poses(p):
    live = p.p_zhi > p.p_zlo
    return torch.where(live[..., None], p.p_poses, torch.zeros_like(p.p_poses))


def test_layout_differs_between_consecutive_resets():
    env = make_env(ring_track(), B=8, procedural_obstacles=1.0)
    env.reset(seed=3)
    first = _live_poses(env.procedural).clone()
    ids = torch.arange(env.B)
    env._reset_envs(ids)
    second = _live_poses(env.procedural).clone()
    assert not torch.allclose(first, second)
    # and every env moved, not just one of them
    per_env = (first - second).abs().amax(dim=(1, 2))
    assert bool((per_env > 1e-6).all())


def test_layout_is_a_function_of_the_generator_alone():
    env = make_env(ring_track(), B=8, procedural_obstacles=1.0)
    env.reset(seed=3)
    state = env.sim.gen.get_state().clone()
    env._reset_envs(torch.arange(env.B))
    a = _live_poses(env.procedural).clone()
    env.sim.gen.set_state(state)
    env._reset_envs(torch.arange(env.B))
    b = _live_poses(env.procedural).clone()
    assert torch.equal(a, b)


def test_a_partial_reset_leaves_the_other_envs_alone():
    env = make_env(ring_track(), B=8, procedural_obstacles=1.0)
    env.reset(seed=3)
    before = _live_poses(env.procedural).clone()
    env._reset_envs(torch.tensor([2, 5]))
    after = _live_poses(env.procedural)
    kept = [i for i in range(8) if i not in (2, 5)]
    assert torch.equal(before[kept], after[kept])
    assert not torch.equal(before[[2, 5]], after[[2, 5]])


def test_a_race_shares_one_layout():
    """Cars of a race see each other; different crates would have one collide with what another
    cannot see."""
    env = make_env(ring_track(), B=8, procedural_obstacles=1.0, race_size=2, opponent="policy")
    env.reset(seed=1)
    p = env.procedural
    for r in range(env.B // env.M):
        assert torch.equal(p.p_poses[r * env.M], p.p_poses[r * env.M + 1])
        assert torch.equal(p.p_zhi[r * env.M], p.p_zhi[r * env.M + 1])


def test_fraction_leaves_some_resets_empty():
    env = make_env(ring_track(), B=64, procedural_obstacles=0.5)
    env.reset(seed=5)
    empty = (env.procedural.p_zhi <= env.procedural.p_zlo).all(1)
    assert 0.2 < float(empty.float().mean()) < 0.8


# ==================================================================== 4. the gap, measured
def measure_gap(env, n_samples: int = 3, reach: float = 0.15):
    """The widest free lateral run at every arc index a prop stands at. Returns `(B, J)` metres and
    the `(B, J)` mask of indices where anything is blocked at all.

    Measured on the geometry that is on the track: each prop's world polygon is recovered from the
    half-planes the LiDAR and the contact test are given, projected onto the centerline frame, and
    the free space is what is left of `[-lane_right, +lane_left]` after the union of the blocked
    intervals. A vertex projection over-states a convex polygon's slice, so this under-states the
    gap -- the safe direction for a test that asserts a lower bound.
    """
    p, tr = env.procedural, env.sim.track
    B, C = p.p_zhi.shape
    live = p.p_zhi > p.p_zlo
    nw, dw = _to_world(p.p_n.reshape(B * C, -1, 2), p.p_d.reshape(B * C, -1), p.p_poses.reshape(B * C, 3))
    valid = nw.pow(2).sum(-1) > 0.5
    verts = _polygon_vertices(nw, dw, valid).reshape(B, C, -1, 2)             # (B,C,K,2)

    tid = env.sim.tid
    s, _, idx = tr.project(p.p_poses[..., :2].reshape(-1, 2), tid[:, None].expand(B, C).reshape(-1))
    idx = idx.reshape(B, C)
    ds = (tr.length[tid] / p.N)[:, None, None]
    off = (torch.arange(n_samples, device=idx.device).float() - (n_samples - 1) / 2) * reach
    j = (idx[..., None] + torch.round(off / ds).long()) % p.N                 # (B,C,n)
    j = j.reshape(B, -1)                                                      # (B,J)
    tq = tid[:, None].expand_as(j)
    c = tr.cl[tq, j]
    tan = tr.cl_tangent[tq, j]
    nrm = torch.stack([-tan[..., 1], tan[..., 0]], -1)
    rel = verts[:, None] - c[:, :, None, None, :]                             # (B,J,C,K,2)
    u = (rel * tan[:, :, None, None, :]).sum(-1)
    v = (rel * nrm[:, :, None, None, :]).sum(-1)
    blocks = live[:, None, :] & (u.amin(-1) <= 0.0) & (u.amax(-1) >= 0.0)     # (B,J,C)
    wl = p.lane_l[tq, j]
    wr = p.lane_r[tq, j]
    # a slot that blocks nothing becomes a zero-width interval at the far edge, so the sweep below
    # needs no mask: it contributes the trailing gap once and nothing after that
    vlo = torch.where(blocks, v.amin(-1), wl[..., None].expand_as(blocks))
    vhi = torch.where(blocks, v.amax(-1), wl[..., None].expand_as(blocks))
    order = vlo.argsort(-1)
    lo_s, hi_s = vlo.gather(-1, order), vhi.gather(-1, order)
    pref = torch.cummax(hi_s, -1).values
    prev = torch.cat([(-wr)[..., None], pref[..., :-1]], -1)
    prev = torch.maximum(prev, (-wr)[..., None])
    gaps = (lo_s - prev).clamp_min(0.0)
    last = (wl - torch.maximum(pref[..., -1], -wr)).clamp_min(0.0)
    return torch.maximum(gaps.amax(-1), last), blocks.any(-1)


@pytest.mark.parametrize("density", [1.0])
def test_constructed_gap_on_catalogue_maps(catalogue_tracks, density):
    """1000 layouts on the catalogue maps: the realised gap is >= 1.20 m in >= 99 % of them."""
    env = make_env(catalogue_tracks, B=50, seed=17, procedural_obstacles=1.0,
                   procedural_density=density, n_beams=8)
    env.reset(seed=17)
    per_layout = []
    per_index = []
    for _ in range(20):
        env._reset_envs(torch.arange(env.B))
        gap, blocked = measure_gap(env)
        g = torch.where(blocked, gap, torch.full_like(gap, float("inf")))
        per_layout.append(g.amin(1))
        per_index.append(gap[blocked])
    layout = torch.cat(per_layout)
    index = torch.cat(per_index)
    assert layout.numel() == 1000, layout.numel()
    ok = float((layout >= po.GAP_MIN - 1e-3).float().mean())
    worst = float(layout.min())
    print(f"\n1000 layouts: gap >= {po.GAP_MIN} m in {ok * 100:.1f} %, worst {worst:.3f} m, "
          f"median {float(index.median()):.3f} m over {index.numel()} blocked arc indices")
    assert ok >= 0.99, f"only {ok * 100:.1f} % of layouts kept the gap (worst {worst:.3f} m)"


def test_patterns_keep_their_spacing(catalogue_tracks):
    """>= 6 m of arc between patterns, which is what makes each one its own problem.

    Read off the draw (`last_s`) rather than off the placed props: projecting a prop back onto the
    centerline picks the nearest lap position, and on a track that folds back on itself -- which
    `blackbox2022_1` does, 149.6 m of lap inside 55 x 39 m -- that is not the one it was placed at.
    """
    env = make_env(catalogue_tracks, B=24, seed=4, procedural_obstacles=1.0, n_beams=8)
    env.reset(seed=4)
    p = env.procedural
    for b in range(env.B):
        ss = torch.sort(p.last_s[b][p.last_live[b]]).values
        if ss.numel() < 2:
            continue
        L = float(env.sim.track.length[env.sim.tid[b]])
        gaps = torch.cat([torch.diff(ss), (L - ss[-1] + ss[0])[None]])
        assert float(gaps.min()) >= po.MIN_SPACING - 1e-3, f"env {b}: {gaps.tolist()}"


def test_a_dense_lap_still_moves_its_patterns():
    """At a density where the 6 m spacing leaves each sector almost no free arc, a pattern must
    still land anywhere on the lap from one reset to the next. On ICCAS at 1.5 per 10 m each of
    the seven stood at the same point +-0.1 m at every reset until the sector grid was given a
    random phase: the policy was learning seven places."""
    env = make_env(ring_track(), B=16, seed=3, procedural_obstacles=1.0, procedural_density=1.5, n_beams=8)
    p = env.procedural
    L = float(env.sim.track.length[0])
    assert float(L / int(p.n_pat[0]) - po.MIN_SPACING) < 0.5, "the test needs a lap with no free arc"
    seen = []
    for r in range(8):
        env.reset(seed=30 + r)
        seen.append(p.last_s[:, 0][p.last_live[:, 0]])
    s0 = torch.cat(seen)
    counts = torch.histc(s0, bins=8, min=0.0, max=L)
    assert bool((counts > 0).all()), f"pattern 0 never reached part of the lap: {counts.tolist()}"


def test_the_slot_budget_drops_patterns_at_random_not_in_lap_order():
    """More live pieces than slots: which patterns keep theirs is random. Kept in lap order, the
    last patterns of a dense lap were dropped at nearly every reset (on ICCAS the last three
    supplied 51 of 640 kept pieces) and that stretch of the lap never had an obstacle."""
    env = make_env(ring_track(), B=32, seed=5, procedural_obstacles=1.0, procedural_density=1.5,
                   procedural_max_props=8, n_beams=8)
    p = env.procedural
    kept = torch.zeros(p.P)
    for r in range(6):
        env.reset(seed=50 + r)
        live = p.p_zhi > p.p_zlo
        kept += torch.bincount(p.slot_pattern[live], minlength=p.P).float()
    assert p.stats()["dropped_per_layout"] > 1.0, "the test needs a budget that actually binds"
    share = kept / kept.sum()
    assert float(share.min()) > 0.5 / p.P, f"a pattern slot is nearly always the one dropped: {share.tolist()}"


def test_every_pattern_kind_is_drawn(catalogue_tracks):
    """Six kinds, all of them: a lap with fewer than six patterns still cycles a fresh permutation."""
    env = make_env(catalogue_tracks, B=64, seed=6, procedural_obstacles=1.0, n_beams=8)
    env.reset(seed=6)
    p = env.procedural
    seen = torch.bincount(p.last_kind[p.last_live], minlength=len(po.PATTERNS)).float()
    assert bool((seen > 0).all()), seen.tolist()
    share = seen / seen.sum()
    assert float(share.min()) > 0.08, f"a kind is nearly never drawn: {share.tolist()}"


# ==================================================================== 5. the sim actually sees them
def test_lidar_returns_a_placed_prop():
    """A beam aimed at a piece comes back short, and says HIT_TALL, exactly as a `+props` prop does."""
    from f1sim.lidar import HIT_TALL
    env = make_env(ring_track(), B=4, procedural_obstacles=1.0, n_beams=360)
    env.reset(seed=2)
    p = env.procedural
    live = p.p_zhi > p.p_zlo
    b = int(torch.nonzero(live.any(1))[0])
    c = int(torch.nonzero(live[b])[0])
    target = p.p_poses[b, c, :2]
    # stand 2 m short of the piece, looking straight at it, on the floor plane
    tr = env.sim.track
    s, _, idx = tr.project(target[None], env.sim.tid[b:b + 1])
    back, _ = tr.pose_at_s((s - 2.0) % tr.length[env.sim.tid[b:b + 1]], env.sim.tid[b:b + 1])
    d = target - back[0]
    yaw = torch.atan2(d[1], d[0])
    pose = env.sim.state[:, :3].clone()
    pose[b] = torch.tensor([back[0, 0], back[0, 1], yaw])
    env.sim.state[:, :3] = pose
    r, r_true, typ = env.sim.lidar.scan(pose, None, env.sim.P, motion_distortion=False, noisy=False,
                                        tid=env.sim.tid, eid=env.sim.eid)
    mid = r_true.shape[1] // 2
    rng = float(r_true[b, mid])
    assert rng < 2.6, f"beam straight at a piece 2 m away returned {rng:.2f} m"
    assert int(typ[b, mid]) == HIT_TALL
    # with the same geometry and no layout, that beam sees the far wall instead
    env.sim.track.env_props = None
    env.sim.track.has_props = False
    _, r_off, _ = env.sim.lidar.scan(pose, None, env.sim.P, motion_distortion=False, noisy=False,
                                     tid=env.sim.tid)
    assert float(r_off[b, mid]) > rng + 0.2


def test_driving_into_a_piece_is_a_collision():
    env = make_env(ring_track(), B=4, procedural_obstacles=1.0, n_beams=32)
    env.reset(seed=2)
    p = env.procedural
    live = p.p_zhi > p.p_zlo
    b = int(torch.nonzero(live.any(1))[0])
    c = int(torch.nonzero(live[b])[0])
    target = p.p_poses[b, c]
    tr = env.sim.track
    s, _, _ = tr.project(target[None, :2], env.sim.tid[b:b + 1])
    start, _ = tr.pose_at_s((s - 1.5) % tr.length[env.sim.tid[b:b + 1]], env.sim.tid[b:b + 1])
    d = target[:2] - start[0]
    pose = env.sim.state[:, :3].clone()
    pose[b] = torch.tensor([start[0, 0], start[0, 1], torch.atan2(d[1], d[0])])
    env.sim.reset(torch.arange(env.B), pose, torch.full((env.B,), 3.0))
    act = torch.zeros(env.B, env.act_dim)
    hit = False
    for _ in range(30):
        _, _, term, _, _ = env.step(act)
        hit = hit or bool(term[b])
        if hit:
            break
    assert hit, "drove straight through the piece"


def test_the_contact_cull_never_drops_a_prop_that_could_touch():
    """The contact tests look at the eight nearest slots. That is exact only while the ninth is
    further than a prop can be and still reach the car; the drawer counts the ones it was not."""
    env = make_env(ring_track(), B=24, procedural_obstacles=1.0, n_beams=16)
    env.reset(seed=21)
    act = torch.zeros(env.B, env.act_dim)
    for _ in range(120):
        env.step(act)
    assert env.sim.prop_reach > 0.4                       # car circumradius + the widest piece's
    assert env.procedural.stats()["missed_in_reach"] == 0


def test_props_for_refuses_a_partial_batch_without_env_rows():
    env = make_env(ring_track(), B=8, procedural_obstacles=1.0)
    env.reset(seed=1)
    with pytest.raises(ValueError, match="env rows"):
        env.sim.track.props_for(env.sim.tid[:3])


def test_spawn_is_clear_of_the_layout():
    env = make_env(ring_track(), B=32, procedural_obstacles=1.0, n_beams=16)
    env.reset(seed=8)
    inside = env.sim._spawn_in_prop(env.sim.state[:, :3], env.sim.tid, env.sim.eid)
    assert not bool(inside.any()), f"{int(inside.sum())} car(s) spawned inside a piece"


# ==================================================================== 6. the teacher stays out
def test_teacher_opponents_are_never_routed_through_a_piece():
    """The raceline the opponents drive is kept clear of every piece, by construction.

    The teacher is pure pursuit on a line built from the occupancy grid and the props are not in the
    grid: it cannot see one and will not steer round one. So the test is the one that matters --
    drive the teacher cars and count their contacts.
    """
    from f1sim.learn import common
    trs, rls = common.load_tracks(common.track_names("gen:control:1400"), racelines=True)
    cfg = cpu_cfg(16)
    ec = EnvConfig(compile_tracker=False, max_steps=600, race_size=2, opponent="teacher",
                   procedural_obstacles=1.0, procedural_density=1.0)
    env = F1VecEnv(trs, cfg, ec, num_envs=16, device="cpu")
    env.set_teacher(common.make_teacher(rls, env))
    assert env.procedural.has_raceline
    env.reset(seed=13)
    opp = ~env.learner                                 # slots 1.. of each race
    touched = torch.zeros(env.B, dtype=torch.bool)
    for _ in range(200):
        env.step(torch.zeros(env.B, env.act_dim))
        pen, _ = env.sim._prop_contact(env.sim.state)
        touched |= (pen > 0)
    assert not bool(touched[opp].any()), (
        f"{int(touched[opp].sum())} of {int(opp.sum())} teacher cars touched a piece")


def test_no_piece_stands_on_the_raceline():
    """The geometric form of the same claim: every placed piece keeps the car's half-width clear of
    the line, measured as a distance in the plane rather than as an offset in a lane frame."""
    from scipy.spatial import cKDTree
    from f1sim.learn import common
    trs, rls = common.load_tracks(common.track_names("gen:control:1400,scene:scene_0912_2344"),
                                  racelines=True)
    cfg = cpu_cfg(8)
    env = F1VecEnv(trs, cfg, EnvConfig(compile_tracker=False, procedural_obstacles=1.0,
                                       race_size=2, opponent="teacher"), num_envs=24, device="cpu")
    env.set_teacher(common.make_teacher(rls, env))
    env.reset(seed=31)
    trees = [cKDTree(np.asarray(r.xy)) for r in rls]
    half = 0.5 * cfg.vehicle.width
    worst = 1e9
    for _ in range(6):
        env._reset_envs(torch.arange(env.B))
        p = env.procedural
        B, C = p.p_zhi.shape
        live = p.p_zhi > p.p_zlo
        nw, dw = _to_world(p.p_n.reshape(B * C, -1, 2), p.p_d.reshape(B * C, -1),
                           p.p_poses.reshape(B * C, 3))
        valid = nw.pow(2).sum(-1) > 0.5
        verts = _polygon_vertices(nw, dw, valid).reshape(B, C, -1, 2).numpy()
        for b in range(B):
            tree = trees[int(env.sim.tid[b])]
            for c in torch.nonzero(live[b]).flatten().tolist():
                d, _ = tree.query(verts[b, c][: int(valid.reshape(B, C, -1)[b, c].sum())])
                worst = min(worst, float(d.min()))
    print(f"\nclosest a piece ever comes to the raceline: {worst:.3f} m (car half-width {half:.3f})")
    assert worst >= half, f"a piece stands {worst:.3f} m from the line, inside the car's half-width"


def test_pattern_reaches_cover_the_pieces():
    """`REACH` has to bound what each kind actually places, or the gap arithmetic is measuring the
    lane somewhere the pattern does not reach."""
    assert set(po.REACH) == set(po.PATTERNS)
    assert po.WINDOW == max(po.REACH.values())
    # the longest `along` each kind can draw, plus the deepest piece
    deepest = max(sh.along for sh in po.build_catalogue(po.catalogue_k_pad())[0])
    longest = {"gate": 0.0, "diagonal": 0.5 * (5 - 1), "chicane": 3.5, "apex": 0.0,
               "cluster": po.BOX_D + 0.03, "scatter": 2.0 + 0.5}
    for kind, along in longest.items():
        assert po.REACH[kind] >= along + 0.5 * deepest - 1e-9, kind


def test_raceline_corridor_is_off_without_a_teacher():
    env = make_env(ring_track(), B=4, procedural_obstacles=1.0)
    assert not env.procedural.has_raceline
    assert float(env.procedural.rl_hi.max()) < 0        # nothing is kept clear
    assert float(env.procedural.rl_lo.min()) > 0


# ==================================================================== 7. batched, not looped
def test_the_draw_has_no_python_loop_over_envs():
    """`redraw` must be batched: a Python loop over envs per reset is the thing this replaces.

    Checked structurally rather than by timing, which would be a flaky assertion about a shared GPU.
    The only loops allowed in the drawing path are `_row`'s, over a Python int slot count fixed at
    import (`PIECES_PER_PATTERN`), and the catalogue build, which runs once.
    """
    src = inspect.getsource(po.ProceduralObstacles)
    tree = ast.parse("class _X:\n" + "\n".join("    " + ln for ln in src.splitlines()[1:]))
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in ("redraw", "_pieces", "_write", "select"):
            loops = [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While, ast.comprehension))]
            assert not loops, f"{fn.name} contains a loop: {ast.dump(loops[0])[:120]}"


def test_slot_budget_is_reported_and_respected(catalogue_tracks):
    env = make_env(catalogue_tracks, B=16, seed=1, procedural_obstacles=1.0, n_beams=8)
    env.reset(seed=1)
    p = env.procedural
    live = (p.p_zhi > p.p_zlo).sum(1)
    assert int(live.max()) <= p.C
    st = p.stats()
    assert st["draws"] >= 16
    assert st["pieces_per_layout"] > 1.0
    assert "slots" in p.describe()


def test_explicit_slot_budget_drops_the_overflow(catalogue_tracks):
    env = make_env(catalogue_tracks, B=16, seed=1, procedural_obstacles=1.0,
                   procedural_max_props=6, n_beams=8)
    env.reset(seed=1)
    p = env.procedural
    assert p.C == 6
    assert int((p.p_zhi > p.p_zlo).sum(1).max()) <= 6
    assert p.stats()["dropped_per_layout"] > 0        # and it says so rather than hiding it


# ==================================================================== 8. evaluation is untouched
def test_evaluation_defaults_are_off():
    assert EnvConfig().procedural_obstacles == 0.0
    src = inspect.getsource(__import__("f1sim.learn.evaluate", fromlist=["x"]))
    assert "procedural" not in src, "evaluate grew a procedural option; the frozen suites move"
