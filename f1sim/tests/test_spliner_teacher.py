"""The ForzaETH `spliner` baseline: does it actually go round the other car?

A comparison opponent is only worth having if it does the thing it is named after. Four claims,
and none of them is visible in a rollout video:

1. **It goes to the side with room.** Put a car on the reference line with a wall close on one
   side, and the offset has to come out on the other -- with the sign convention the teacher's
   `offset` argument actually uses, which is the one bug a video of a working pass would hide.
2. **It comes back.** The post-apex control points exist so the line rejoins; an implementation
   that drops the opponent the instant it is passed snaps the offset to zero at the worst moment,
   and a spline evaluated only ahead of the apex would never show it.
3. **It does not weave.** Committed to a side and alongside the car, a flipped preference must not
   flip the line.
4. **An empty road is the raceline.** Byte for byte the reference teacher's own plan -- otherwise
   every number this baseline produces is measured against a different driver than the raceline
   one, and the solo/traffic comparison stops meaning anything.

Small tracks and a 36-beam LiDAR, the convention of `test_opponent_slots.py`: nothing here reads
a scan.
"""
import functools

import numpy as np
import pytest
import torch

from f1sim import Config, maps
from f1sim import opponent_slots as osl
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.spliner_teacher import (CONTROL_S, FIXED_PRED_TIME, OVERTAKING, RACING, TRAILING,
                                   PredictiveSplinerTeacher, SplinerTeacher, _SHAPE,
                                   _natural_cubic)

CHEAP = "gen:competition:0"           # cached, and roomy enough on one side for an evasion


@functools.lru_cache(maxsize=4)
def _track_and_raceline(name: str):
    from f1sim.raceline import Raceline
    track = maps.load(name)
    return track, Raceline.build_cached(track)


def _env(envs=8, seed=7, track=CHEAP, **cfg_kw):
    tr, rl = _track_and_raceline(track)
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(**{"race_size": 2, "opponent": "teacher", "max_steps": 4000, "hist_len": 0,
                        **cfg_kw})
    return common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl])


def _planner(env) -> SplinerTeacher:
    """The planner over this env's reference teacher, building one if the env has none (a solo
    env never gets a teacher, and this baseline still has a line to drive)."""
    base = env.teacher
    if base is None:
        base = common.make_teacher([_track_and_raceline(CHEAP)[1]], env)
    return SplinerTeacher(base, env=env)


def _place(env, row, s_idx, lateral=0.0, yaw_of_line=True, speed=3.0):
    """Put car `row` on the reference line at index `s_idx`, `lateral` m to its left."""
    t = env.teacher
    tid = env.sim.tid[row].item()
    p = t.xy[tid, s_idx].clone()
    tan = t.tan[tid, s_idx]
    n = torch.stack([-tan[1], tan[0]])
    p = p + lateral * n
    env.sim.state[row, 0], env.sim.state[row, 1] = p[0], p[1]
    if yaw_of_line:
        env.sim.state[row, 2] = torch.atan2(tan[1], tan[0])
    env.sim.state[row, 3] = speed
    return tid


def _scene(env, ego_row, opp_row, opp_idx, gap_m, opp_lateral=0.0, ego_lateral=0.0):
    """Opponent at raceline index `opp_idx`, ego `gap_m` metres behind it along the line.

    `gap_m` may be negative, which puts the ego alongside or past -- the half of the manoeuvre the
    post-apex control points exist for.
    """
    t = env.teacher
    tid = env.sim.tid[ego_row].item()
    ds = float(t.ds[tid])
    e_idx = int(round(opp_idx - gap_m / ds)) % t.N
    _place(env, ego_row, e_idx, ego_lateral)
    _place(env, opp_row, opp_idx, opp_lateral)
    # `signed_gaps` reads `sim.s`, the centreline arc, which only `step` maintains. Set it from the
    # raceline stations directly so the scene is the scene and not one step of physics later.
    env.sim.s[ego_row] = e_idx * ds
    env.sim.s[opp_row] = opp_idx * ds
    return e_idx


def _lopsided_station(env, track_name):
    """(index, +1 if the left side is the roomier one) where a pass fits on exactly one side.

    Measured with `raceline.track_widths`, which marches the occupancy grid along the line's
    normals -- a different implementation from the distance field the planner itself reads. That
    is the point: if the two agree about which side of the car the room is on, the planner's
    normal direction, its offset sign and the teacher's own `offset` convention all line up, and
    if any one of them is backwards this is where it shows.
    """
    from f1sim.raceline import track_widths
    track = _track_and_raceline(track_name)[0]
    pts = env.teacher.xy[0].cpu().numpy()
    wl, wr = track_widths(track, pts)
    need = 0.65 + 0.5 * float(env.cfg.vehicle.width) + 0.2          # evasion + body + boundary
    fits = np.maximum(wl, wr) >= need
    if not fits.any():
        pytest.skip(f"{track_name} has no station a 0.65 m evasion fits on")
    spread = np.where(fits, np.abs(wl - wr), -np.inf)
    i = int(spread.argmax())
    return i, (1 if wl[i] >= wr[i] else -1)


# ============================================================ the spline itself
def test_the_shape_function_is_the_spline_through_the_control_points():
    """One metre at the apex, on the line everywhere else -- and the batched evaluator agrees.

    The shape is precomputed at import because a cubic spline is linear in its control values.
    That claim is only worth the saving if the curve it produces is the curve the seven points
    define, which is what is checked: at every knot, and against the closed-form coefficients at
    points between them.
    """
    want = np.eye(len(CONTROL_S))[3]
    p = SplinerTeacher.__new__(SplinerTeacher)
    p.shape = torch.tensor(_SHAPE, dtype=torch.float32)
    p.knots = torch.tensor(CONTROL_S, dtype=torch.float32)
    got = p._shape_at(torch.tensor(CONTROL_S, dtype=torch.float32))
    assert torch.allclose(got, torch.tensor(want, dtype=torch.float32), atol=1e-5), got.tolist()
    # outside the control arc the manoeuvre is not happening
    out = p._shape_at(torch.tensor([-6.0, -4.001, 4.001, 12.0]))
    assert torch.equal(out, torch.zeros(4)), out.tolist()
    # and it is a real interpolant, not a bump that happens to hit the knots: the apex is the max
    dense = p._shape_at(torch.linspace(-4, 4, 801))
    assert 0.99 <= float(dense.max()) <= 1.05
    assert float(dense.argmax()) == pytest.approx(400, abs=15)


def test_the_spline_solver_is_a_natural_cubic():
    """Continuity and the natural end condition, on a case with a known answer.

    A wrong tridiagonal solve still passes through every control point -- interpolation is the
    easy half -- so what is checked is the part that is not forced: matching first derivatives
    across each interior knot, and zero second derivative at the ends.
    """
    x = np.array([0.0, 1.0, 2.5, 4.0])
    y = np.array([0.0, 2.0, -1.0, 0.5])
    c = _natural_cubic(x, y)
    for i in range(len(x) - 2):                       # slope matches across the interior knots
        h = x[i + 1] - x[i]
        left = c[i, 1] + 2 * c[i, 2] * h + 3 * c[i, 3] * h * h
        assert left == pytest.approx(c[i + 1, 1], abs=1e-9)
    assert c[0, 2] == pytest.approx(0.0, abs=1e-12)                       # natural at the start
    h = x[-1] - x[-2]
    assert 2 * c[-1, 2] + 6 * c[-1, 3] * h == pytest.approx(0.0, abs=1e-9)  # and at the end


# ============================================================ the planner
def test_an_empty_road_is_the_reference_teacher_exactly():
    """No car in the way, no difference. Every number this baseline reports in traffic is read
    against its own solo pace, so a planner that quietly drives a different line when alone would
    make its own comparison meaningless."""
    env = _env(envs=6, race_size=1, opponent="policy")   # race_size 1: nobody to race
    env.reset(seed=3)
    p = _planner(env)
    from f1sim.mpc import PlanSpec
    P, tid, spec = env.sim.P, env.sim.tid, PlanSpec()
    a_ref = p.base.plan_action(env.sim.state, P, tid, env.ecfg.v_max_policy, spec)
    a_spl = p.plan_action(env.sim.state, P, tid, env.ecfg.v_max_policy, spec)
    assert torch.equal(a_ref, a_spl), (a_ref - a_spl).abs().max().item()
    assert torch.equal(p.last_state, torch.full_like(p.last_state, RACING))


def test_it_passes_on_the_side_with_room():
    """A car on the line 3 m ahead, and the offset goes where the track is.

    The side the room is on is measured independently of the planner (`_lopsided_station`), so
    what this pins is the whole sign chain: the reference line's normal, this planner's `side`,
    and the `offset` convention of the teacher it hands the number to -- metres LEFT of the line.
    An implementation with any one of them backwards drives into the wall it just measured, and
    a video of it passing on the other side of an empty track would look fine.
    """
    env = _env(envs=4, race_size=2, opponent="teacher")
    env.reset(seed=9)
    p = _planner(env)
    idx, roomy_side = _lopsided_station(env, CHEAP)
    swept = []
    for gap_m in (4.0, 3.0, 2.0, 1.0, 0.5, 0.0):
        _scene(env, 0, 1, opp_idx=idx, gap_m=gap_m)
        d, mode, _scale = p.decide(env.sim.state, env.sim.tid)
        assert int(mode[0]) == OVERTAKING, f"at {gap_m} m the state was {int(mode[0])}"
        assert float(p._side[0]) == roomy_side, \
            f"at {gap_m} m it committed to the {'left' if float(p._side[0]) > 0 else 'right'}, " \
            f"but the room is on the {'left' if roomy_side > 0 else 'right'}"
        swept.append((gap_m, round(float(d[0]), 4)))
    # The line itself: the apex is the widest point of the manoeuvre and it has to be on that
    # side. Away from the apex a natural cubic dips the other way by about a tenth of the apex
    # displacement, which is the spline's own shape and not a side change -- `_side` above is what
    # says which side the planner chose, and it does not move.
    widest = max(swept, key=lambda gv: abs(gv[1]))
    assert abs(widest[1]) > 0.3, f"the evasion never got wide enough to be one: {swept}"
    assert widest[1] * roomy_side > 0, \
        f"apex {widest[1]:+.3f} m at {widest[0]} m, room on the " \
        f"{'left' if roomy_side > 0 else 'right'}: {swept}"


def test_the_line_comes_back_after_the_pass():
    """The post-apex control points are not decoration.

    An opponent just behind is still inside the spline's arc, so the offset is still non-zero and
    shrinking -- that is the rejoin. A planner that drops a car the moment its gap goes negative
    snaps the line back across the track with the other car alongside, which is the one moment it
    must not; and a spline evaluated only ahead of the apex would never show it.
    """
    env = _env(envs=4, race_size=2, opponent="teacher")
    env.reset(seed=13)
    p = _planner(env)
    idx, _ = _lopsided_station(env, CHEAP)
    seen = []
    for gap_m in (2.0, 0.5, -0.5, -1.5, -3.0, -6.0):
        _scene(env, 0, 1, opp_idx=idx, gap_m=gap_m)
        d, mode, _ = p.decide(env.sim.state, env.sim.tid)
        seen.append((gap_m, round(float(d[0]), 4), int(mode[0])))
    passing = [o for g, o, m in seen if m == OVERTAKING]
    if len(passing) < 2:
        pytest.skip(f"this station never becomes a pass: {seen}")
    behind = [o for g, o, m in seen if g <= -0.5 and m == OVERTAKING]
    assert behind and max(abs(o) for o in behind) > 0.05, f"the rejoin never happened: {seen}"
    assert abs(seen[-1][1]) < 1e-6, f"still evading a car 6 m behind: {seen}"


def test_it_does_not_switch_sides_while_alongside():
    """Committed and off its line, the planner holds the side it chose.

    Two 0.31 m cars in a lane this narrow: a line that crosses to the other side of a car it is
    already beside is exactly the contact the side-switch rule exists to prevent. Forced by
    handing the planner a commitment to the side its free preference is not.
    """
    env = _env(envs=4, race_size=2, opponent="teacher")
    env.reset(seed=21)
    p = _planner(env)
    idx, _ = _lopsided_station(env, CHEAP)
    # alongside, and well off its own line -- the two conditions `committed` reads
    _scene(env, 0, 1, opp_idx=idx, gap_m=0.3, ego_lateral=0.45)
    d_free, mode, _ = p.decide(env.sim.state, env.sim.tid)
    if int(mode[0]) != OVERTAKING:
        pytest.skip("this scene is not a pass on this track; the rule has nothing to hold")
    chosen = float(d_free[0])
    p._side[0] = -np.sign(chosen)                            # committed the OTHER way
    d_held, _mode, _ = p.decide(env.sim.state, env.sim.tid)
    assert float(d_held[0]) * chosen <= 0, \
        f"the side flipped despite the commitment: free {chosen:+.3f}, held {float(d_held[0]):+.3f}"


def test_no_race_means_no_opponent_term():
    """Solo, or handed a state it cannot match to the simulator's rows, it drives the raceline and
    says so. Guessing where the other cars are would be worse than not knowing."""
    env = _env(envs=4, race_size=2, opponent="teacher")
    env.reset(seed=5)
    p = _planner(env)
    half = env.sim.state[: env.B // 2]                       # a batch that is not the env's rows
    d, mode, scale = p.decide(half, env.sim.tid[: env.B // 2])
    # None, not zeros: the reference teacher measures its own off-line error differently once it
    # has been handed an offset, so "no opponent" has to reach it as "no offset was asked for".
    assert d is None
    assert torch.equal(mode, torch.full_like(mode, RACING))
    assert torch.equal(scale, torch.ones_like(scale))
    assert torch.equal(p.last_offset, torch.zeros_like(p.last_offset))


# ============================================================ the registry
def test_the_kind_plugs_in_through_the_registry():
    """One entry, and the env builds the real planner for the rows that named it -- not the
    raceline teacher wearing another name, which is the substitution the registry exists to
    prevent and the reason a baseline can be silently not a baseline."""
    kind = osl.kind_of("forzaeth")
    assert kind.teacher and kind.available
    assert kind.teacher_factory == "f1sim.spliner_teacher:SplinerTeacher"
    env = _env(envs=12, race_size=3, opponent="slots",
               opponent_slots=[{"kind": "forzaeth"}, {"kind": "raceline"}])
    env.reset(seed=11)
    assert env.alt_teacher_kinds == ("forzaeth",)
    assert isinstance(env.alt_teachers[0], SplinerTeacher)
    assert env.alt_teachers[0].base is env.teacher, "it was not built from the raceline teacher"
    assert env.alt_teacher_mask["forzaeth"].view(-1, 3)[0].tolist() == [False, True, False]


def test_a_slot_band_and_grip_label_still_reach_the_teacher():
    """The wrapper keeps `speed_scale` and `label_grip` on the reference teacher, where the per-car
    tensors live. A copy of them on the wrapper would leave a slot's speed band configured and
    not applied, which no assertion about the line would catch."""
    env = _env(envs=4, race_size=2, opponent="teacher")
    env.reset(seed=7)
    p = _planner(env)
    p.speed_scale = 0.5
    assert env.teacher.speed_scale == 0.5 and p.speed_scale == 0.5
    p.label_grip = "conservative"
    assert env.teacher.label_grip == "conservative" and p.label_grip == "conservative"


def test_the_predictive_variant_aims_where_the_other_car_will_be():
    """The published pair differ in one number, so the difference has to be exactly that number.

    Asserted as an identity rather than as "it is different": the predictive planner facing a car
    `g` ahead doing `v` must draw the same line the plain one draws facing a *stopped* car
    `g + FIXED_PRED_TIME * v` ahead. Anything else means the prediction leaked into a second
    decision -- the side, the state machine, the speed -- which is the drift a shared
    implementation exists to prevent.
    """
    env = _env(envs=4, race_size=2, opponent="teacher")
    env.reset(seed=17)
    plain, pred = SplinerTeacher(env.teacher, env=env), PredictiveSplinerTeacher(env.teacher, env=env)
    assert plain.fixed_pred_time == 0.0 and pred.fixed_pred_time == FIXED_PRED_TIME
    idx, _ = _lopsided_station(env, CHEAP)
    v_opp, gap_m = 4.0, 3.0
    _scene(env, 0, 1, opp_idx=idx, gap_m=gap_m)
    env.sim.state[1, 3] = v_opp
    d_pred, m_pred, s_pred = pred.decide(env.sim.state, env.sim.tid)
    # the same scene with the car already where the prediction says it will be, and stopped
    _scene(env, 0, 1, opp_idx=idx, gap_m=gap_m + FIXED_PRED_TIME * v_opp)
    env.sim.state[1, 3] = 0.0
    d_plain, m_plain, s_plain = plain.decide(env.sim.state, env.sim.tid)
    assert int(m_pred[0]) == int(m_plain[0]) == OVERTAKING
    assert float(d_pred[0]) == pytest.approx(float(d_plain[0]), abs=2e-3), \
        f"predicted {float(d_pred[0]):+.4f} m, present-tense equivalent {float(d_plain[0]):+.4f} m"
    assert float(s_pred[0]) == pytest.approx(float(s_plain[0]))


def test_both_published_planners_are_registered_and_share_one_implementation():
    """Two baselines, one class. A second module would let a fix land in one and not the other,
    and two lines that were supposed to differ in a single number would quietly differ in more."""
    plain, pred = osl.kind_of("forzaeth"), osl.kind_of("forzaeth_pred")
    assert plain.available and pred.available
    assert plain.module == pred.module == "f1sim.spliner_teacher"
    assert issubclass(PredictiveSplinerTeacher, SplinerTeacher)
    env = _env(envs=12, race_size=3, opponent="slots",
               opponent_slots=[{"kind": "forzaeth"}, {"kind": "forzaeth_pred"}])
    env.reset(seed=23)
    assert set(env.alt_teacher_kinds) == {"forzaeth", "forzaeth_pred"}
    by_kind = dict(zip(env.alt_teacher_kinds, env.alt_teachers))
    assert type(by_kind["forzaeth"]) is SplinerTeacher
    assert type(by_kind["forzaeth_pred"]) is PredictiveSplinerTeacher
    assert by_kind["forzaeth"].fixed_pred_time == 0.0
    assert by_kind["forzaeth_pred"].fixed_pred_time == FIXED_PRED_TIME


# ============================================================ the lane-switch family
def test_the_lane_planner_holds_its_lane_and_comes_back():
    """Two claims a fixed-lane planner lives or dies on, and neither shows in a rollout.

    **Hysteresis**: an argmin over lanes re-run every step flips between two nearly equal ones at
    the step rate, and the car weaves. **The rate limit**: the lane is a decision, the line to it
    is a manoeuvre, so the commanded offset may not step.
    """
    from f1sim.lane_teacher import CHANGING, LANE_RATE, LaneSwitchTeacher, RACING as L_RACING
    env = _env(envs=4, race_size=2, opponent="teacher")
    env.reset(seed=31)
    p = LaneSwitchTeacher(env.teacher, env=env)
    idx, _ = _lopsided_station(env, CHEAP)
    step_cap = LANE_RATE * p.dt
    prev, offs = 0.0, []
    for _ in range(80):                       # a car sitting in the racing line, held there
        _scene(env, 0, 1, opp_idx=idx, gap_m=4.0)
        d, mode, _ = p.decide(env.sim.state, env.sim.tid)
        o = float(d[0])
        assert abs(o - prev) <= step_cap + 1e-6, \
            f"the offset stepped {abs(o - prev):.3f} m in one tick, cap {step_cap:.3f}"
        prev = o
        offs.append(o)
    assert abs(offs[-1]) > 0.05, f"it never left the blocked lane: {offs[-5:]}"
    lane_held = p._lane.clone()
    for _ in range(40):                       # the same scene again: the lane must not wander
        _scene(env, 0, 1, opp_idx=idx, gap_m=4.0)
        p.decide(env.sim.state, env.sim.tid)
    assert torch.equal(p._lane, lane_held), "the lane moved with nothing about the scene changing"
    # and with the road clear it returns to the racing line, at the same bounded rate
    for _ in range(400):
        _scene(env, 0, 1, opp_idx=idx, gap_m=40.0)
        d, mode, _ = p.decide(env.sim.state, env.sim.tid)
    assert abs(float(d[0])) < 0.05, f"it never came back to the line: {float(d[0]):.3f}"
    assert int(mode[0]) == L_RACING


def test_the_lane_planner_is_registered_and_says_what_it_is_not():
    """The registry entry has to carry the claim, because the claim is the honest part: this is
    the UNICORN *family*, not a reproduction, and a reader choosing an opponent sees only this."""
    from f1sim.lane_teacher import LaneSwitchTeacher
    kind = osl.kind_of("lane_switch")
    assert kind.teacher and kind.available
    assert "재현이 아니라" in kind.note, "the entry does not say it is not a reproduction"
    env = _env(envs=12, race_size=3, opponent="slots",
               opponent_slots=[{"kind": "lane_switch"}, {"kind": "forzaeth"}])
    env.reset(seed=29)
    by_kind = dict(zip(env.alt_teacher_kinds, env.alt_teachers))
    assert isinstance(by_kind["lane_switch"], LaneSwitchTeacher)
    assert by_kind["lane_switch"].base is env.teacher


def test_an_empty_road_is_the_reference_teacher_for_the_lane_planner_too():
    """Same claim as the spline family's, for the same reason: a baseline's solo pace is what its
    traffic pace is read against, so with nothing to race it has to be the reference teacher."""
    from f1sim.lane_teacher import LaneSwitchTeacher
    from f1sim.mpc import PlanSpec
    env = _env(envs=6, race_size=1, opponent="policy")
    env.reset(seed=33)
    base = common.make_teacher([_track_and_raceline(CHEAP)[1]], env)
    p = LaneSwitchTeacher(base, env=env)
    P, tid, spec = env.sim.P, env.sim.tid, PlanSpec()
    assert torch.equal(base.plan_action(env.sim.state, P, tid, env.ecfg.v_max_policy, spec),
                       p.plan_action(env.sim.state, P, tid, env.ecfg.v_max_policy, spec))


def test_a_lane_planner_that_could_not_come_back_is_refused():
    """The bound above, as a constructor check rather than a comment.

    Found the hard way: with the first hysteresis chosen by feel, returning from the innermost
    lane was worth 0.12 and the threshold was 0.25, so every lane was absorbing and the baseline
    drove a permanent offset. That reads as a tuning problem and is an arithmetic one, so the
    arithmetic is checked where the numbers are set.
    """
    from f1sim.lane_teacher import LaneSwitchTeacher
    env = _env(envs=4, race_size=2, opponent="teacher")
    env.reset(seed=37)
    with pytest.raises(ValueError, match="no way back to the racing line"):
        LaneSwitchTeacher(env.teacher, env=env, hysteresis=0.25)
    with pytest.raises(ValueError, match="no way back to the racing line"):
        LaneSwitchTeacher(env.teacher, env=env, switch_cost=1.0, offline_cost=1.0)
    with pytest.raises(ValueError, match="racing line"):
        LaneSwitchTeacher(env.teacher, env=env, lanes=(0.3, 0.6))     # nowhere to come back to
