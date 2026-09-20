"""`f1sim.interactive_teacher.InteractiveTeacher` -- the teacher that races the other car.

What is worth testing here is not that the code runs but that the *decision* is the one the design
claims, and every claim below is one a present-tense cost would fail:

1. The candidate family spans the four moves a racing teacher has to be able to make -- go left, go
   right, follow, brake -- and each candidate does in the world what its name says.
2. The label is interchangeable with `RacelineTeacher`'s: same shape, same range, same action space.
3. With nothing to race, the argmin is the raceline teacher's own plan.
4. **The cost is time-indexed.** Three synthetic scenes -- an opponent drifting right, an opponent
   braking, an opponent moving into the pass lane while a second car holds the other side -- and
   the choice in each is the racing one. The same scenes scored against the opponent's position
   *now*, held frozen, cannot separate left from right at all: that comparison is the experiment,
   not decoration.
5. Two Gauss-Newton iterations per candidate are close enough to six that the family is the
   raceline teacher's own offset plans and not a cheaper approximation of them.

The scenes are built directly on a raceline and a track, with the opponents' futures supplied as
data. `car_future`, which is what supplies them in a real race, is tested on its own in
`test_opp_token.py` -- the two halves fail for different reasons and a test that mixes them cannot
say which.
"""
import functools
import math

import pytest
import torch

from f1sim import Config, maps
from f1sim.mpc import ACT_DIM, PlanSpec
from f1sim.track import TrackTensors
from f1sim.interactive_teacher import DEFAULT_OFFSETS, DEFAULT_SPEEDS, InteractiveTeacher
from f1sim.opponent_events import raceline_offset_limit
from f1sim.raceline import Raceline
from f1sim.teacher import RacelineTeacher

TRACK = "gen:control:1400"          # a training map with the straightest wide stretch of the set
V_MAX = 9.0
EGO_SPEED = 5.0


@functools.lru_cache(maxsize=2)
def _scene_parts(track_name: str = TRACK):
    """(track, raceline teacher with its lane clamp, the straightest roomy index on it)."""
    track = TrackTensors(maps.load(track_name), "cpu")
    rl = Raceline.build_cached(maps.load(track_name))
    cfg = Config()
    t = RacelineTeacher([rl], wheelbase=cfg.vehicle.lf + cfg.vehicle.lr, device="cpu")
    t.offset_limit = raceline_offset_limit(t, track, 0.5 * cfg.vehicle.width, 0.10)
    return track, t, _open_straight(t, track)


def _open_straight(teacher: RacelineTeacher, track, span_m: float = 7.0,
                   clearance_m: float = 0.55) -> int:
    """The straightest raceline index whose next `span_m` keeps at least `clearance_m` of room.

    Straight so that going left and going right are nearly the same move and a scene can be built
    out of arc and lateral offsets alone; roomy so that the +-0.6 m candidates are decided by the
    opponent rather than by a wall, which is a different term with its own tests. These are narrow
    tracks -- the widest 7 m stretch on the training maps tried holds 0.6 m of distance-field
    clearance either side of the line -- so this is a search for the best there is and not a
    threshold anything passes.
    """
    step = float(teacher.ds[0])
    w = max(2, int(round(span_m / step)))
    roll = lambda v: torch.stack([torch.roll(v, -i) for i in range(w)], 0)
    straight = roll(teacher.kappa[0].abs()).max(0).values
    edt = track.sample_edt(teacher.xy[0], torch.zeros(teacher.N, dtype=torch.long))
    room = roll(edt).min(0).values
    ok = room > clearance_m
    if not bool(ok.any()):
        ok = room >= float(room.max())
    return int(torch.where(ok, straight, torch.full_like(straight, 9e9)).argmin())


def _point(teacher, idx0: int, arc: float, lat: float) -> torch.Tensor:
    """World point `arc` metres along the raceline from `idx0` and `lat` metres to its left."""
    j = (idx0 + int(round(arc / float(teacher.ds[0])))) % teacher.N
    p, t = teacher.xy[0, j], teacher.tan[0, j]
    return p + lat * torch.stack([-t[1], t[0]])


def _ego(teacher, idx0: int, speed: float = EGO_SPEED) -> torch.Tensor:
    """(1, 8) ground-truth state: on the line at `idx0`, pointing along it."""
    p, t = teacher.xy[0, idx0], teacher.tan[0, idx0]
    st = torch.zeros(1, 8)
    st[0, 0], st[0, 1] = p[0], p[1]
    st[0, 2] = math.atan2(float(t[1]), float(t[0]))
    st[0, 3] = speed
    return st


class _StubRace:
    """The two things `InteractiveTeacher` asks a race for: the walls, and the opponents' futures.

    A stub rather than a simulator because these tests are about the cost, and a scene whose
    opponent future is *given* is the only way to ask "would this teacher pick the gap?" without
    also asking "did the predictor see the gap?" -- which is a separate measurement with its own
    script (`f1sim.learn.opp_future_check`).
    """

    M = 2

    def __init__(self, track, batch: int = 1, cap: float = V_MAX):
        self.cfg = Config()
        self.sim = type("S", (), {"track": track})()
        self.speed_cap = torch.full((batch,), float(cap))
        self.fut = None                       # (B, C, K, 2)
        self.present = None                   # (B, C)

    def set(self, fut: torch.Tensor, present=None):
        self.fut = fut
        self.present = torch.ones(fut.shape[:2]) if present is None else present

    def opponent_future(self, times, model=None, state=None, *, return_yaw=False):
        k = len(times) if not torch.is_tensor(times) else int(times.numel())
        if self.fut is None:
            raise AssertionError("the scene did not set an opponent future")
        assert self.fut.shape[2] == k, f"scene future has {self.fut.shape[2]} samples, teacher asked for {k}"
        if return_yaw:
            yaw = state[:, 2, None, None].expand(self.fut.shape[:-1])
            return self.fut, self.present, yaw
        return self.fut, self.present


def _teacher(track, base, batch: int = 1, **kw) -> InteractiveTeacher:
    it = InteractiveTeacher(base, **kw)
    it.attach(_StubRace(track, batch))
    return it


def _track_opponent(teacher, idx0, times, arc0, arc_rate, lat0, lat_rate):
    """(1, 1, K, 2) an opponent that starts at (arc0, lat0) and moves at (arc_rate, lat_rate)."""
    return torch.stack([_point(teacher, idx0, arc0 + arc_rate * float(t), lat0 + lat_rate * float(t))
                        for t in times], 0)[None, None]


def _choice(it, state, spec, cand=None):
    """Offset (or signed immediate-evasion direction), pace, cost and candidates."""
    tid = torch.zeros(1, dtype=torch.long)
    cand = it._candidates(state, None, tid, V_MAX, spec, None,
                          it.base.project(state[:, :2], tid)[0]) if cand is None else cand
    cost = it.score(cand, state, tid, V_MAX, spec)
    j = int(cost.argmin(0)[0])
    n_spd = it.speeds.numel()
    regular = it.offsets.numel() * n_spd
    if j >= regular:
        if j == regular:
            return 0., 0., cost[:, 0], cand
        evasive = j - regular - 1
        return (1. if evasive < 2 else -1.), (1. if evasive % 2 == 0 else 0.), cost[:, 0], cand
    return float(it.offsets[j // n_spd]), float(it.speeds[j % n_spd]), cost[:, 0], cand


# ----------------------------------------------------------------- the family


def test_candidate_family_covers_left_right_follow_brake():
    track, base, idx0 = _scene_parts()
    it = _teacher(track, base)
    assert it.n_candidates == len(DEFAULT_OFFSETS) * len(DEFAULT_SPEEDS) + 5
    assert min(DEFAULT_OFFSETS) < 0 < max(DEFAULT_OFFSETS) and 0.0 in DEFAULT_OFFSETS
    assert max(DEFAULT_SPEEDS) == 1.0 and min(DEFAULT_SPEEDS) < 0.5
    spec = PlanSpec()
    st = _ego(base, idx0)
    tid = torch.zeros(1, dtype=torch.long)
    cand = it._candidates(st, None, tid, V_MAX, spec, None, base.project(st[:, :2], tid)[0])
    assert cand.shape == (it.n_candidates, 1, ACT_DIM)
    assert torch.equal(cand[-1, :, -2:], -torch.ones_like(cand[-1, :, -2:]))
    world, psi, v = it.rollout(cand, st, V_MAX, spec)
    # Compare path geometry at equal arc, not at equal time: feasibility can
    # legitimately slow a tighter offset and change its one-second progress.
    from f1sim.mpc import N_KNOTS, path_points, plan_length
    _, y, _, _ = path_points(cand[:, 0, :N_KNOTS] * spec.kappa_max,
                             plan_length(st[:, 3], spec).expand(cand.shape[0]))
    lat = y[:, -1] - y[it.base_index, -1]
    n_spd = it.speeds.numel()
    lats = [float(lat[i * n_spd]) for i in range(len(DEFAULT_OFFSETS))]
    # Ordered, spanning both sides, and the raceline teacher's own plan exactly in the middle. Not
    # "each candidate reached its nominal offset": the lane clamp cuts a 0.6 m offset to whatever
    # the section has room for, and the horizon ends part-way along a plan that is longer than it,
    # so the displacement AT one second is smaller than the offset by construction.
    assert all(b > a + 0.02 for a, b in zip(lats, lats[1:])), lats
    assert lats[0] < -0.10 and lats[-1] > 0.10, lats
    assert abs(lats[it.base_index // n_spd]) < 1e-5, lats
    # and the speed family is a speed family: less arc, lower final speed, same side of the line
    base_arc = float(v[it.base_index, 0].sum())
    for m, sc in enumerate(DEFAULT_SPEEDS):
        arc = float(v[it.base_index - it.base_index % n_spd + m, 0].sum())
        # Independently certified scales are quantized to 1/(16*2**8).
        assert arc <= base_arc + 2 * V_MAX * len(it.horizon_times(spec)) / 4096
        if sc == min(DEFAULT_SPEEDS):
            assert arc < base_arc, f"speed scale {sc} travelled as far as 1.0"


def test_label_is_interchangeable_with_the_raceline_teacher():
    """Same action space, same shape, same bounds -- a DAgger buffer cannot tell the two apart.

    The bounds are the ones `RacelineTeacher.plan_action` itself produces: curvature knots clamped
    at `0.85 * kappa_max` (`k_lim`) and speeds in [-1, 1]. And the speed columns are never HIGHER
    than the raceline teacher's, because the speed family only ever scales it down.
    """
    from f1sim.mpc import N_KNOTS
    track, base, idx0 = _scene_parts()
    n = 24
    it = _teacher(track, base, batch=n)
    spec = PlanSpec()
    st = torch.cat([_ego(base, (idx0 + 31 * i) % base.N, 2.0 + 0.25 * i) for i in range(n)], 0)
    tid = torch.zeros(n, dtype=torch.long)
    times = it.horizon_times(spec)
    far = torch.stack([_point(base, idx0, 60.0, 0.0)] * len(times))[None, None].expand(n, 1, len(times), 2)
    it.env.set(far.contiguous(), torch.zeros(n, 1))
    a_ref = base.plan_action(st, None, tid, V_MAX, spec)
    a_it = it.plan_action(st, None, tid, V_MAX, spec)
    assert a_it.shape == a_ref.shape == (n, ACT_DIM)
    assert a_it.dtype == a_ref.dtype
    assert float(a_ref[:, :N_KNOTS].abs().max()) <= 0.85 + 1e-6
    assert float(a_it[:, :N_KNOTS].abs().max()) <= 0.85 + 1e-6
    assert float(a_it[:, N_KNOTS:].abs().max()) <= 1.0
    # Each geometry now has its own physical ceiling. An easier offset may
    # exceed the projected centre-line speed, but not its unprojected request.
    import copy
    raw = copy.copy(base)
    raw._defer_profile_projection = True
    upper = raw.plan_action(st, None, tid, V_MAX, spec)
    assert torch.all(a_it[:, N_KNOTS:] <= upper[:, N_KNOTS:] + 1e-5)


def test_with_nothing_to_race_the_argmin_is_the_raceline_plan():
    track, base, idx0 = _scene_parts()
    it = _teacher(track, base)
    spec = PlanSpec()
    st = _ego(base, idx0)
    times = it.horizon_times(spec)
    far = torch.stack([_point(base, idx0, 60.0, 0.0)] * len(times))[None, None]
    it.env.set(far, torch.zeros(1, 1))                            # present = 0: nobody is being raced
    off, sc, cost, cand = _choice(it, st, spec)
    assert (off, sc) == (0.0, 1.0), f"chose offset {off} at {sc} with no opponent"
    assert int(cost.argmin()) == it.base_index


# ----------------------------------------------------------------- the point


def test_the_gap_the_opponent_is_leaving_wins():
    """An opponent drifting right, and the left of it is the place to be -- but only in the future.

    Right now it is straight ahead on the line, so a cost that reads its position now cannot prefer
    either side, and it cannot tell a car that is leaving from one that is sitting there. The second
    half of this test is that comparison, and it is the experiment the whole teacher rests on.

    Compared on the opponent term alone, and deliberately: the wall, clearance, progress and
    smoothness of a candidate do not depend on what the other car does, so they are identical
    between the two scorings and every asymmetry they carry (these lanes are not straight and one
    side of this one is a wall) cancels out of the comparison.
    """
    track, base, idx0 = _scene_parts()
    it = _teacher(track, base)
    spec = PlanSpec()
    st = _ego(base, idx0)
    tid = torch.zeros(1, dtype=torch.long)
    times = it.horizon_times(spec)
    fut = _track_opponent(base, idx0, times, arc0=2.6, arc_rate=2.0, lat0=0.0, lat_rate=-0.55)
    cand = it._candidates(st, None, tid, V_MAX, spec, None, base.project(st[:, :2], tid)[0])
    n_spd = it.speeds.numel()
    at = lambda off: int(list(DEFAULT_OFFSETS).index(off)) * n_spd      # that offset at full speed

    it.env.set(fut)
    off, sc, cost, _ = _choice(it, st, spec, cand)
    assert off > 0, f"opponent leaving to the right, teacher chose offset {off} at {sc}"
    moving = it.score(cand, st, tid, V_MAX, spec, parts=True)[1]["opp"][:, 0]

    it.env.set(fut[:, :, :1].expand_as(fut).contiguous())               # frozen where it is NOW
    frozen = it.score(cand, st, tid, V_MAX, spec, parts=True)[1]["opp"][:, 0]

    assert float(moving[at(0.3)]) < float(moving[at(-0.3)]) - 2.0, (
        f"the motion did not separate the two sides: {float(moving[at(0.3)]):.3f} vs "
        f"{float(moving[at(-0.3)]):.3f}")
    assert abs(float(frozen[at(0.6)] - frozen[at(-0.6)])) < 0.2, (
        "with the opponent held where it is, the two wide passes are the same move; got "
        f"{float(frozen[at(0.6)]):.3f} vs {float(frozen[at(-0.6)]):.3f}")
    assert float(moving[at(-0.6)]) > float(frozen[at(-0.6)]) + 1.0, (
        "the right-hand pass is expensive only because of where the car is GOING; frozen "
        f"{float(frozen[at(-0.6)]):.3f}, moving {float(moving[at(-0.6)]):.3f}")
    assert float(frozen[at(0.0)]) > float(moving[at(0.0)]) + 2.0, (
        "staying on the line is expensive against the car's present position and cheap against its "
        f"future one; frozen {float(frozen[at(0.0)]):.3f}, moving {float(moving[at(0.0)]):.3f}")


def test_a_braking_opponent_is_passed_rather_than_followed():
    track, base, idx0 = _scene_parts()
    it = _teacher(track, base)
    spec = PlanSpec()
    st = _ego(base, idx0)
    times = it.horizon_times(spec)
    # 3 m ahead and stopping: it covers 0.4 m in the second the ego would cover 5
    it.env.set(_track_opponent(base, idx0, times, arc0=3.0, arc_rate=0.4, lat0=0.0, lat_rate=0.0))
    off, sc, cost, cand = _choice(it, st, spec)
    assert off != 0.0, f"drove straight at a stopping car (offset {off}, speed {sc})"
    assert sc >= 0.55, f"passed but crawled: speed scale {sc}"
    # and staying on the line is only survivable by braking
    n_spd = it.speeds.numel()
    line = int(list(DEFAULT_OFFSETS).index(0.0)) * n_spd
    assert int(cost[line:line + n_spd].argmin()) > 0, "on the line, full speed was still the best"


def test_a_car_moving_into_the_gap_is_not_passed_into():
    """The opponent moves into the pass lane while a second car holds the other side."""
    track, base, idx0 = _scene_parts()
    it = _teacher(track, base)
    spec = PlanSpec()
    st = _ego(base, idx0)
    times = it.horizon_times(spec)
    closer = _track_opponent(base, idx0, times, arc0=3.0, arc_rate=0.6, lat0=-0.1, lat_rate=0.7)
    blocker = _track_opponent(base, idx0, times, arc0=2.2, arc_rate=2.2, lat0=-0.55, lat_rate=0.0)
    it.env.set(torch.cat([closer, blocker], 1))
    off, sc, cost, cand = _choice(it, st, spec)
    assert sc < 1.0, f"both lanes closing and the teacher held racing speed (offset {off})"
    assert off <= 0.3, f"drove into the lane the opponent was moving into (offset {off})"


def test_two_iteration_candidates_are_the_raceline_teachers_own_offset_plans():
    """The family is generated with a short Gauss-Newton budget so it fits in DAgger's loop. If
    that budget changed what the candidates ARE, the search would be over a different family than
    the one this teacher claims to search."""
    track, base, idx0 = _scene_parts()
    spec = PlanSpec()
    st = torch.cat([_ego(base, (idx0 + 17 * i) % base.N, 2.0 + 0.7 * i) for i in range(8)], 0)
    tid = torch.zeros(8, dtype=torch.long)
    idx = base.project(st[:, :2], tid)[0]
    worst = 0.0
    for off in DEFAULT_OFFSETS:
        o = torch.full((8,), float(off))
        full = base.plan_action(st, None, tid, V_MAX, spec, iters=6, offset=o, idx=idx)
        cheap = base.plan_action(st, None, tid, V_MAX, spec, iters=2, offset=o, idx=idx)
        # The fit budget is geometric; physical projection may amplify a small
        # curvature difference into a different safe speed ceiling.
        worst = max(worst, float((full[:, :-2] - cheap[:, :-2]).abs().max()))
    assert worst < 0.05, f"2 iterations differ from 6 by {worst:.4f} of the normalized plan range"
