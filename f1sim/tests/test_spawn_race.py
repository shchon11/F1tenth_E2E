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
3. **An abreast teacher holds its lane position, then hands the side back to its planner**
   (`spawn_alongside_hold`, `teacher.LaneHold`): through the hold it keeps the centreline offset it
   was placed at -- whatever its planner kind would choose -- so it does not close on the car beside
   it in the first second; over the ramp its own choice comes back, and at the end it is exactly that
   choice. It needs the opponent-event offset channel: without events the teachers' offset stays
   None. What it was measured to do to car contacts is in `EnvConfig.spawn_alongside_hold`.
4. **The preferred abreast separation is used where it fits** (`spawn_alongside_sep`), the minimum
   (`spawn_alongside_sep_min`) where only that does, a stagger where neither does.
"""
import copy
import functools
import math

import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common

ICCAS = ["real:korea_2025_iccas", "real:korea_2025_iccas+warp3", "real:korea_2025_iccas+warp11"]
#: The opponent-event channel switched on with no event that will ever fire in a test this long:
#: the rolling start exists only where that channel does (`EnvConfig.spawn_alongside_hold`).
QUIET_EVENTS = dict(opp_events=("shift",), opp_event_rate=1e-9)
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


# --------------------------------------------------------------------------- 3. rolling start
def _lateral_from_line(env, rows):
    t = env.teacher
    xy = env.sim.state[rows, :2]; tid = env.sim.tid[rows]
    idx = t.project(xy, tid)[0]
    tan = t.tan[tid, idx]
    return ((xy - t.xy[tid, idx]) * torch.stack([-tan[:, 1], tan[:, 0]], 1)).sum(1)


def _lateral_from_centreline(env, rows):
    return env.sim.track.project(env.sim.state[rows, :2], env.sim.tid[rows], prev_idx=env.sim.cl_idx[rows])[1]


@pytest.mark.parametrize("hold", [1.0, 0.0])
def test_an_abreast_teacher_holds_its_offset_then_eases_onto_its_line(hold):
    """The offset held is the *centreline* one the car was placed at: the learner beside it holds its
    lane, and a car held at a fixed raceline offset would follow the raceline across the lane --
    sideways, into the learner, which is the closing the rolling start is for."""
    # soft contacts: under "terminate" a car that crashes respawns mid-test with a fresh offset.
    # The event channel on with a rate no event fires at: the rolling start rides on it.
    env = _make([WIDE], 32, 19, race_size=2, opponent="teacher", spawn_order="alongside",
                collision_mode="soft", **QUIET_EVENTS,
                spawn_alongside_hold=hold, spawn_alongside_ramp=1.0 if hold else 0.0)
    env.reset(seed=23)
    opp = (~env.learner).nonzero().flatten()
    assert bool(env.spawn_abreast[opp].all()), "the wide track should fit every row abreast"
    if not hold:
        assert env._roll_cfg is False and env.teacher.lane_hold is None
        return
    assert bool(env.roll_on[opp].all()) and not bool(env.roll_on[env.learner].any()), \
        "only the teacher cars roll: the learner has no line to hold"
    assert env.teacher.lane_hold.w is env.roll_w, "the teacher reads the env's weights in place"
    cl0 = env.roll_cl0[opp].clone()
    # `reset` ends with one zero-action physics step, so the car has moved a step (<= 3 m/s x 25 ms
    # along a heading jittered by `spawn_alongside_yaw`) since the offset was read at placement
    assert torch.allclose(cl0, _lateral_from_centreline(env, opp), atol=0.04)
    rl0 = _lateral_from_line(env, opp)
    far = rl0.abs() > 0.25                                        # placed well off its line
    assert int(far.sum()) >= 4
    dt = env.sim.control_dt
    cl, rl = [], []
    for k in range(int(round(2.6 / dt))):
        assert bool((env.ep_step[opp] == k).all()), "a car reset during the test"
        w_now = env._rolling_start()[opp]
        t = k * dt
        w = 1.0 if t <= hold else max(0.0, 1.0 - (t - hold))
        assert torch.allclose(w_now, torch.full_like(w_now, w), atol=1e-4), f"t={t:.2f}"
        env.step(torch.zeros(env.B, env.act_dim))
        cl.append(_lateral_from_centreline(env, opp)); rl.append(_lateral_from_line(env, opp))
    # held: through the hold the car stays on the centreline offset it was placed at; after hold +
    # ramp + 0.6 s it is near its line
    for k in (int(0.5 / dt), int(1.0 / dt)):
        drift = (cl[k] - cl0)[far].abs()
        assert float(drift.median()) < 0.03, f"drifted {float(drift.median()):.2f} m off its placement at {k * dt:.2f} s"
    end = rl[-1][far].abs()
    assert float(end.median()) < 0.5 * float(rl0[far].abs().median()), "never eased onto its line"
    assert float(env.roll_w.abs().max()) == 0.0, "still held after hold + ramp"


KINDS = ("raceline", "forzaeth", "forzaeth_pred", "lane_switch", "interactive")


def _kind_env(kind, envs=16, seed=37, **kw):
    """Race of 2 in the plan action space: the learner and one opponent of `kind` placed beside it,
    with the event channel on and silent."""
    seat = {"kind": kind, "present": 1.0, "spawn": "alongside", "events": ["shift"], "event_rate": 1e-9}
    return _make([WIDE], envs, seed, race_size=2, opponent="slots", opponent_slots=[seat], action_mode="plan",
                 collision_mode="soft", **kw)


def _held_learner(env):
    """A copy of the env's teacher that holds each LEARNER on the lane position it was placed at (a
    `LaneHold` of its own, the env's centreline index shared), so that the learner beside a held
    opponent does not close the gap either: whatever moves toward it in the hold is the opponent."""
    from f1sim.teacher import LaneHold
    h = env.teacher.lane_hold
    tl = copy.copy(env.teacher)
    cl = _lateral_from_centreline(env, torch.arange(env.B))
    tl.lane_hold = LaneHold(env.learner.float(), cl, env.roll_ci, h.cxy, h.ctan, h.seg, h.room, h.kap,
                            h.step, h.n_steps)
    return tl


def _drivable(env, rows, reach=4.0, k_max=0.3):
    """Whether the lane position each held car keeps is one it can drive through the hold without a
    turn-in transient: no bend of it tighter than `k_max` (3.3 m radius) over the `reach` metres it
    covers. Inside a hairpin a fixed offset can be tighter than the car's full lock, and no hold makes
    that drivable; and a car placed in a bend starts at zero steer, so the tracker's turn-in carries
    it out first -- measured on this track, 0.4 1/m at 3.5 m/s: 6.9 cm out in 0.3 s, then 12 cm back
    in, the same for every planner kind (the spawn, not the hold)."""
    t_, h = env.teacher, env.teacher.lane_hold
    env.roll_ci.copy_(env.sim.cl_idx)
    tid = env.sim.tid
    line = t_._hold_line(h, tid, env.sim.state[:, :2])
    d = torch.linspace(0.0, reach, 17)[None].expand(env.B, -1)
    k = t_._hold_kappa(h, tid, t_._hold_sample(line, d)[1]).abs().amax(1)
    return k[rows] <= k_max


@pytest.mark.parametrize("kind", KINDS)
def test_the_hold_holds_for_every_planner_kind(kind):
    """Through the hold a held opponent does not move toward the learner beside it -- whatever its
    planner kind would choose (`LaneHold` scales that by 1 - w) and however the raceline runs across
    the lane -- on every start whose lane position is drivable at all (`_drivable`)."""
    env = _kind_env(kind)
    assert env._roll_cfg and env.teacher.lane_hold is not None
    tl = _held_learner(env)
    dt = env.sim.control_dt
    hold_k = int(round(env.ecfg.spawn_alongside_hold / dt))
    worst, n_ok, tight = [], 0, []
    for r in range(3):
        env.reset(seed=60 + r)
        opp = (env.roll_on & ~env.learner).nonzero().flatten()
        assert opp.numel() >= env.B // 4, f"only {opp.numel()} of {env.B // 2} opponents placed abreast"
        lrn = opp - 1
        tl.lane_hold.cl.copy_(_lateral_from_centreline(env, torch.arange(env.B)))
        cl0 = env.roll_cl0[opp].clone()
        side = torch.sign(_lateral_from_centreline(env, lrn) - cl0)
        ok = _drivable(env, opp)
        toward = torch.full_like(cl0, -math.inf)
        for k in range(hold_k):
            env.step(env.teacher_label(tl, offset=torch.zeros(env.B)))
            toward = torch.maximum(toward, (_lateral_from_centreline(env, opp) - cl0) * side)
        worst += toward[ok].tolist(); n_ok += int(ok.sum())
        tight += toward[~ok].tolist()
    assert n_ok >= 12, f"only {n_ok} drivable starts: the test would pass on nothing"
    assert max(worst) < 0.05, (f"{kind}: moved up to {max(worst):.3f} m toward the learner in the hold "
                               f"({sorted(worst)[-5:]}); hairpin starts excluded: {sorted(tight)[-3:]}")


def _plans_with_choice(env, alt, d, w):
    """The plan `alt` hands every opponent row when its own lateral choice is `d` [m] and the hold's
    weight is `w`, and the same with no hold installed at all."""
    B = env.B
    off = env.events.lateral_offset()
    choice = torch.full((B,), float(d))
    alt.decide = lambda state, tid=None: (choice[:state.shape[0]], torch.zeros(state.shape[0], dtype=torch.long),
                                          torch.ones(state.shape[0]))
    env.roll_w.copy_(torch.where(env.roll_on, torch.full((B,), float(w)), torch.zeros(B)))
    env.roll_ci.copy_(env.sim.cl_idx)
    args = (env.sim.state, env.sim.P, env.sim.tid, env.ecfg.v_max_policy, env.tracker.spec)
    held = alt.plan_action(*args, offset=off, plan_speed=env._tracker_plan_speed())
    keep = env.teacher.lane_hold
    env.teacher.lane_hold = None
    try:
        bare = alt.plan_action(*args, offset=off, plan_speed=env._tracker_plan_speed())
    finally:
        env.teacher.lane_hold = keep
    return held, bare


@pytest.mark.parametrize("kind", ["forzaeth", "lane_switch"])
def test_the_hold_overrides_a_planners_choice_and_the_ramp_hands_it_back(kind):
    """A planner kind's own offset (`FrenetOpponentPlanner.decide`) does not move a held car: at full
    weight the plan is the held line whichever side the planner picks. At weight 0 -- the end of the
    ramp -- the plan is the planner's own, bit for bit the plan with no hold installed at all."""
    env = _kind_env(kind, envs=8, seed=41)
    env.reset(seed=43)
    alt = env.alt_teachers[env.alt_teacher_kinds.index(kind)]
    rows = (env.roll_on & ~env.learner).nonzero().flatten()
    assert rows.numel() >= 2
    left, _ = _plans_with_choice(env, alt, +0.45, 1.0)
    right, _ = _plans_with_choice(env, alt, -0.45, 1.0)
    assert torch.allclose(left[rows], right[rows], atol=1e-4), "a held car's plan moved with its planner's choice"
    for d in (+0.45, -0.45):
        held, bare = _plans_with_choice(env, alt, d, 0.0)
        assert torch.equal(held, bare), "at the end of the ramp the plan is not exactly the planner's own"
    mid, _ = _plans_with_choice(env, alt, +0.45, 0.5)
    _, own = _plans_with_choice(env, alt, +0.45, 0.0)
    assert not torch.allclose(mid[rows], left[rows], atol=1e-3) and not torch.allclose(mid[rows], own[rows], atol=1e-3)
    assert (left[~env.roll_on] == own[~env.roll_on]).all(), "a car that is not held planned differently"


def test_an_idle_hold_changes_nothing():
    """Installed but holding nobody -- every race not laid abreast, and every race once its ramp is
    over -- the hold leaves every car's trajectory bit for bit what it is with the rolling start off."""
    kw = dict(race_size=2, opponent="teacher", spawn_order="random", action_mode="plan", collision_mode="soft",
              **QUIET_EVENTS)
    on = _make([WIDE], 16, 47, **kw)
    off = _make([WIDE], 16, 47, spawn_alongside_hold=0.0, spawn_alongside_ramp=0.0, **kw)
    assert on._roll_cfg and on.teacher.lane_hold is not None
    assert not off._roll_cfg and off.teacher.lane_hold is None
    n = int(round(2.5 / on.sim.control_dt))

    def run(env):
        # one after the other, not interleaved: `reset(seed)` seeds the global generator as well as the
        # simulator's, and the sensor noise draws from it
        env.reset(seed=50)
        states = [env.sim.state.clone()]
        for _ in range(n):
            env.step(torch.zeros(env.B, env.act_dim))
            states.append(env.sim.state.clone())
        return states
    s_on, s_off = run(on), run(off)
    still = ~on.roll_on.view(-1, 2).any(1).repeat_interleave(2)            # races nobody in which is held
    assert bool(still.any()) and bool((~still).any()), "the draw should give both kinds of race"
    # the same draws at the reset: every car not held exactly where it was; a held car starts lined up
    # with its lane (`_reset_envs_core`), so only its heading differs -- and its position by the one
    # zero-action step `reset` ends with
    assert torch.equal(s_on[0][~on.roll_on], s_off[0][~on.roll_on]), "the hold drew from the generator at the reset"
    assert float((s_on[0][on.roll_on, :2] - s_off[0][on.roll_on, :2]).norm(dim=1).max()) < 0.02
    for k in range(n + 1):
        assert torch.equal(s_on[k][still], s_off[k][still]), f"an unheld race diverged at step {k}"
    assert not torch.equal(s_on[-1][~still], s_off[-1][~still]), "the held races did not differ: vacuous"
    # after hold + ramp the weights are all 0 and the planners are handed back exactly
    assert float(on.roll_w.abs().max()) == 0.0 and on._roll_steps_left == 0


# --------------------------------------------------------------------------- 4. separation
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


# --------------------------------------------------------------------------- 5. the offset channel
def _spy_offsets(env):
    """Record (offset handed to the teacher, the events' own offset, rolling steps left) per call."""
    seen = []
    eager = env._teacher_normalized

    def spy(teacher, ev_off, *a, **k):
        ref = None if env.events is None or not env.events.enabled else env.events.lateral_offset()
        seen.append((None if ev_off is None else ev_off.clone(), ref, env._roll_steps_left))
        return eager(teacher, ev_off, *a, **k)

    env._teacher_normalized = spy
    return seen


def test_without_events_the_teachers_offset_stays_none():
    """`OffsetPlanner._merge_offset`: a zeros tensor is not None to a teacher, and an argument that is a
    tensor on some steps and None on others is not one captured CUDA graph (`TeacherGraph`)."""
    env = _make([WIDE], 8, 19, race_size=2, opponent="teacher", spawn_order="alongside",
                collision_mode="soft")
    assert env._roll_cfg is False
    seen = _spy_offsets(env)
    env.reset(seed=23)
    assert bool(env.spawn_abreast[~env.learner].all())
    for _ in range(5):
        env.step(torch.zeros(env.B, env.act_dim))
    assert seen and all(off is None for off, _, _ in seen)


def test_the_offset_channel_is_the_events_own_and_the_weights_run_down():
    """The hold replaces the lateral target inside the teacher (`LaneHold`); it adds nothing to the
    offset the events hand the teachers, which stays a tensor on every step. Its weights are written
    only while some row can still be rolling, and are 0 once the window is over."""
    env = _make([WIDE], 8, 19, race_size=2, opponent="teacher", spawn_order="alongside",
                collision_mode="soft", **QUIET_EVENTS)
    assert env._roll_cfg is True
    seen = _spy_offsets(env)
    weights = []
    env.reset(seed=23)
    window = int(math.ceil((env.ecfg.spawn_alongside_hold + env.ecfg.spawn_alongside_ramp)
                           / env.sim.control_dt)) + 1
    for _ in range(window + 5):
        env.step(torch.zeros(env.B, env.act_dim))
        weights.append(env.roll_w.clone())
    assert all(off is not None and torch.equal(off, ref) for off, ref, _ in seen)
    assert float(weights[0].max()) == 1.0, "nobody was held: the test would pass with it off"
    k0 = next(i for i, (_, _, left) in enumerate(seen) if left == 0)
    assert len(seen) - k0 >= 3 and all(float(w.abs().max()) == 0.0 for w in weights[k0:])


def test_a_layout_driven_car_does_not_roll():
    """It drives its own line, and the table the rolling start converts offsets with is the raceline's."""
    env = _make([WIDE], 8, 19, race_size=2, opponent="teacher", spawn_order="alongside",
                collision_mode="soft", **QUIET_EVENTS)
    env.reset(seed=23)
    opp = ~env.learner
    assert bool(env.roll_on[opp].all())
    env.alt_teacher_mask["layout"] = opp.clone()       # as `_build_slot_tables` keeps a `layout` slot's rows
    try:
        env.reset(seed=23)
    finally:
        env.alt_teacher_mask.pop("layout")
    assert bool(env.spawn_abreast[opp].all()) and not bool(env.roll_on.any())
