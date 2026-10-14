"""A race is placed as one: no car starts in contact, or in a contact it cannot avoid (2026-10-14).

The G4 boot-camp audit found 122 of 379 charged car-contact onsets in the first 2 s of an
episode, 59 of them on step 0, every one charged to the learner. They came from placement, not
from driving: `sample_spawn` rejected a pose (a wall, the 3 m runway, a crate on the heading) and
then moved that one car -- pulled to the centerline beside its abreast partner, or walked along
the lap onto another car -- with no idea the rest of the race existed. The grid tests in
`test_opponent_diversity.py` run with `spawn_runway` 0, so the runway and heading rejections never
fire there and could not have caught it. These run the configuration that does:

1. **No overlap, no crowding, at every spawn path.** ICCAS and its warps, race of 4, three random
   seats each present with p 0.5, `spawn_runway` 3.0, density 1.5 movable props, no raceline
   corridor; over 1,000 races. Every present pair separated, every pair that is not an abreast row
   at least `spawn_gap[0]` minus the abreast stagger apart in arc. And a car rejoining a race
   keeps `spawn_gap[0]` from every race mate.
2. **An empty seat has no place on the lap.** `signed_gaps` reads half a lap for it.
3. **The preferred abreast separation is used where it fits** (`spawn_alongside_sep`), the minimum
   (`spawn_alongside_sep_min`) where only that does, a stagger where neither does.

(A rolling start for abreast opponents had its tests here; it was measured not to remove the merge
and was taken out -- docs/research/rolling-start-2026-10-14.md.)
"""
import functools
import math

import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common

ICCAS = ["real:korea_2025_iccas", "real:korea_2025_iccas+warp3", "real:korea_2025_iccas+warp11"]
WIDE = "real:blackbox2022_1"
TRACK = "gen:competition:0"


@functools.lru_cache(maxsize=8)
def _track_and_raceline(name: str):
    from f1sim.raceline import Raceline
    track = maps.load(name)
    return track, Raceline.build_cached(track)


def _make(tracks, envs, seed=7, **cfg_kw):
    trs, rls = zip(*[_track_and_raceline(t) for t in tracks])
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(**{"max_steps": 4000, "hist_len": 0, **cfg_kw})
    return common.make_env(list(trs), envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=list(rls))


def _boot_env(envs=64, seed=7, **kw):
    seat = {"kind": "forzaeth", "present": 0.5, "spawn": "random"}
    return _make(ICCAS, envs, seed, race_size=4, opponent="slots", opponent_slots=[seat, seat, seat],
                 spawn_runway=3.0, procedural_obstacles=1.0, procedural_density=1.5,
                 procedural_max_props=18, procedural_raceline_corridor="off",
                 movable_obstacles=True, collision_mode="soft", **kw)


def _body_gap(sim, i: int, j: int) -> float:
    """Largest separation along the four box axes between cars i and j (> 0: apart, a lower bound
    on the body gap), on the footprint `Simulator._car_contacts` tests."""
    st = sim.state
    pts = []
    for r in (i, j):
        c, s_ = math.cos(float(st[r, 2])), math.sin(float(st[r, 2]))
        R = torch.tensor([[c, -s_], [s_, c]])
        local = sim.corners * (float(sim.car_dims[r, 0]) / 0.50)
        p = local @ R.T + st[r, :2]
        ext = torch.tensor([-float(sim.car_rear[r, 0]) * c, -float(sim.car_rear[r, 0]) * s_])
        pts.append(torch.cat([p, p[2:] + ext], 0))
    best = -math.inf
    for r in (i, j):
        c, s_ = math.cos(float(st[r, 2])), math.sin(float(st[r, 2]))
        for ax in (torch.tensor([c, s_]), torch.tensor([-s_, c])):
            a, b = pts[0] @ ax, pts[1] @ ax
            best = max(best, float(b.min() - a.max()), float(a.min() - b.max()))
    return best


def _arc(env, i: int, j: int) -> float:
    L = float(env.sim.track.length[env.sim.tid[i]])
    return (float(env.sim.s[j] - env.sim.s[i]) + L / 2) % L - L / 2


# --------------------------------------------------------------------------- 1. placement
def test_a_boot_camp_race_never_spawns_in_contact_or_crowded():
    env = _boot_env()
    e = env.ecfg
    floor = e.spawn_gap[0] - e.spawn_alongside_gap[1] - 0.1     # the abreast stagger is the one allowed slack
    M, races = env.M, 0
    overlaps, crowded, abreast_pairs, worst_arc = [], [], 0, math.inf
    for k in range(63):
        env.reset(seed=500 + k)
        for r in range(env.B // M):
            races += 1
            rows = [i for i in range(r * M, (r + 1) * M) if not bool(env.sim.absent[i])]
            for x in range(len(rows)):
                for y in range(x + 1, len(rows)):
                    i, j = rows[x], rows[y]
                    gap = _body_gap(env.sim, i, j)
                    if gap <= 0.0:
                        overlaps.append((k, i, j, gap))
                    if bool(env.spawn_abreast[i]) and bool(env.spawn_abreast[j]):
                        abreast_pairs += 1
                        continue
                    a = abs(_arc(env, i, j))
                    worst_arc = min(worst_arc, a)
                    if a < floor:
                        crowded.append((k, i, j, round(a, 2)))
    st = env.sim.spawn_stats
    assert races >= 1000
    # the test has to reach the paths that used to fail, or it proves nothing
    assert st["rejected"] > 100 and st["group_walks"] > 50, st
    assert abreast_pairs > 50, f"only {abreast_pairs} abreast pairs in {races} races"
    assert not overlaps, f"{len(overlaps)} present pairs overlapping at placement: {overlaps[:5]}"
    assert not crowded, f"{len(crowded)} non-abreast pairs closer than {floor:.2f} m of arc: {crowded[:5]}"
    assert st["crowded"] == 0, st


def test_the_first_step_of_a_boot_camp_race_has_no_car_contact():
    """The contact test itself, rather than a re-derivation of it: one zero-action step after each
    reset, on the same configuration."""
    env = _boot_env(envs=32, seed=11)
    contacts = 0
    for k in range(20):
        env.reset(seed=900 + k)
        _, _, _, _, info = env.step(torch.zeros(env.B, env.act_dim))
        contacts += int(info["car_collision"].sum())
    assert contacts == 0, f"{contacts} car contacts on the first step over {20 * env.B // env.M} races"


def test_a_car_rejoining_a_race_keeps_clear_of_every_race_mate():
    """The single-car respawn (a crash under `collision_mode="terminate"`) used to sit `gap` behind
    whichever car is first in its roster, wherever the others were."""
    env = _make([TRACK], 32, 13, race_size=4, opponent="teacher", spawn_gap=(2.5, 3.0))
    gap0 = env.ecfg.spawn_gap[0]
    M, near, overlaps = env.M, [], 0
    for k in range(12):
        env.reset(seed=40 + k)
        for _ in range(20):
            env.step(torch.zeros(env.B, env.act_dim))
        # respawn one car per race: slot (k % (M-1)) + 1, the others stay where they are
        ids = torch.arange(env.B // M) * M + (k % (M - 1)) + 1
        env._reset_envs(ids)
        for i in ids.tolist():
            for j in env.sim.other_idx[i].tolist():
                a = abs(_arc(env, i, j))
                if a < gap0 - 0.05:
                    near.append((k, i, j, round(a, 2)))
                overlaps += int(_body_gap(env.sim, i, j) <= 0.0)
    assert overlaps == 0, f"{overlaps} rejoining cars overlap a race mate"
    assert not near, f"{len(near)} rejoining cars within {gap0} m of arc of a race mate: {near[:5]}"


# --------------------------------------------------------------------------- 2. empty seats
def test_an_empty_seat_has_no_place_on_the_lap():
    env = _boot_env(envs=32, seed=17)
    seen = 0
    for k in range(4):
        env.reset(seed=70 + k)
        for _ in range(30):
            env.step(torch.zeros(env.B, env.act_dim))
            d = env.signed_gaps(env.sim.s, env.sim.tid)
            gone = env.sim.absent[env.sim.other_idx]
            L = env.sim.track.length[env.sim.tid][:, None].expand_as(d)
            seen += int(gone.sum())
            assert torch.allclose(d[gone], L[gone] / 2), "a parked seat still has an arc gap"
            assert bool((d[~gone].abs() < L[~gone] / 2 + 1e-4).all())
    assert seen > 0, "no empty seat was drawn: the test would pass with the mask off"


# --------------------------------------------------------------------------- 3. separation
def _abreast_spacing(env):
    """(min |lateral| of each learner's race mate in the learner's frame, abreast-row extent)."""
    e = env.ecfg
    psi = 3.0 * min(e.spawn_yaw_std, e.spawn_alongside_yaw)
    extent = env.cfg.vehicle.width * math.cos(psi) + env.cfg.vehicle.length * math.sin(psi)
    return env.learner_view().lat[env.learner, 0].abs(), extent


def test_the_preferred_abreast_separation_is_used_where_the_lane_has_room():
    env = _make([WIDE], 32, 29, race_size=2, opponent="teacher", spawn_order="alongside")
    e = env.ecfg
    assert e.spawn_alongside_sep > e.spawn_alongside_sep_min
    env.reset(seed=31)
    lat, extent = _abreast_spacing(env)
    assert bool(env.spawn_abreast.all())
    # centre spacing along the normal = extent + separation; the pair's own frame differs from the
    # normal by the heading jitter and the stagger, a few centimetres
    assert float(lat.min()) > extent + 0.5 * (e.spawn_alongside_sep + e.spawn_alongside_sep_min), (
        f"abreast centres {float(lat.min()):.3f} m apart: the {e.spawn_alongside_sep} m separation was not used")


@pytest.mark.parametrize("minimum", [0.15, 3.0])
def test_a_lane_without_room_for_the_preferred_gap_takes_the_minimum_or_staggers(minimum):
    """3 m preferred never fits a lane; the minimum then decides: 0.15 m is a side-by-side start at
    that gap, 3 m is no side-by-side start at all."""
    env = _make([WIDE], 32, 29, race_size=2, opponent="teacher", spawn_order="alongside",
                spawn_alongside_sep=3.0, spawn_alongside_sep_min=minimum)
    env.reset(seed=31)
    if minimum > 1.0:
        assert not bool(env.spawn_abreast.any()), "nothing fits: the race has to spawn staggered"
        return
    lat, extent = _abreast_spacing(env)
    assert bool(env.spawn_abreast.all())
    assert float(lat.min()) > extent + minimum - 0.06 and float(lat.max()) < extent + minimum + 0.1, (
        f"abreast centres {float(lat.min()):.3f}-{float(lat.max()):.3f} m apart, extent {extent:.3f}")
