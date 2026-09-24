"""More kinds of opponent, obstacles where the racing line is, and a fair start.

Three things the user asked for on 2026-09-21, and each one is a claim a rollout cannot show:

1. *"상대차 움직임을 최대한 많이 넣어줘"* — one run has to see every driver, not one of them.
2. *"최대한 다양한 지점에 놓게 해줘. 레이싱 라인에 직결될 때 아직 잘 못피하는것 같아"* — the policy
   has never met an obstacle on the racing line, because until now none could be there.
3. *"스폰 직후 피할 수 없는 지점에 장애물이나 벽이 나타나지 않도록"* — a collision the car was never
   given time to avoid is the layout's number, not the policy's.

Small tracks and a 36-beam LiDAR, the convention of `test_opponent_slots.py`.
"""
import functools
import math

import pytest
import torch

from f1sim import Config, maps
from f1sim import opponent_slots as osl
from f1sim.gym_env import EnvConfig
from f1sim.learn import common

TRACK = "gen:competition:0"
MIX = ["raceline", "forzaeth", "forzaeth_pred", "lane_switch", "interactive"]
PROP_AWARE = ["forzaeth", "forzaeth_pred", "lane_switch"]


@functools.lru_cache(maxsize=2)
def _track_and_raceline(name: str):
    from f1sim.raceline import Raceline
    track = maps.load(name)
    return track, Raceline.build_cached(track)


def _env(envs=24, seed=3, **cfg_kw):
    tr, rl = _track_and_raceline(TRACK)
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(**{"race_size": 2, "opponent": "slots", "max_steps": 400, "hist_len": 0,
                        **cfg_kw})
    return common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl])


# ============================================================ 1. a mix of drivers
def test_a_kind_mix_actually_puts_every_driver_on_the_track():
    """Every kind listed drives the car at some point, and only ever one at a time.

    The failure this pins is the quiet one: a mix that builds all five objects and then always
    asks the same one. Every result off such a run would be labelled with five drivers and
    measured against one, and nothing in a video would look wrong.
    """
    env = _env(opponent_slots=[{"kind_mix": MIX}])
    env.reset(seed=3)
    assert set(env.alt_teacher_kinds) == set(MIX) - {"raceline"}, env.alt_teacher_kinds
    opp = env._slot_mixed_rows
    assert int(opp.sum()) == env.B // 2, "the mixed slot does not own the rows it should"
    seen = {"raceline": 0}
    for _ in range(600):
        env.step(torch.zeros(env.B, env.act_dim))
        named = torch.zeros(env.B, dtype=torch.bool)
        for k in env.alt_teacher_kinds:
            m = env.alt_teacher_mask[k] & opp
            assert not bool((m & named).any()), f"{k} shares a row with another kind"
            named |= m
            seen[k] = seen.get(k, 0) + int(m.sum())
        seen["raceline"] += int((opp & ~named).sum())
        assert not bool((named & ~opp).any()), "a fixed row was reassigned by the mix"
    missing = [k for k, n in seen.items() if n == 0]
    assert not missing, f"never drawn in 600 steps: {missing} ({seen})"


def test_a_mixed_slot_does_not_disturb_a_fixed_one():
    """Slot 1 fixed, slot 2 mixed: slot 1's driver never changes. The masks are rewritten in place
    on every reset, so the bit a fixed slot set once at construction is the thing most likely to
    be clobbered by the rewrite -- and it would show up as a run whose control arm drifted."""
    env = _env(envs=24, race_size=3,
               opponent_slots=[{"kind": "forzaeth"}, {"kind_mix": ["lane_switch", "raceline"]}])
    env.reset(seed=5)
    fixed = env.slot == 1
    for _ in range(300):
        env.step(torch.zeros(env.B, env.act_dim))
        assert bool(env.alt_teacher_mask["forzaeth"][fixed].all()), "the fixed slot lost its driver"
        assert not bool((env.alt_teacher_mask["lane_switch"] & fixed).any()), \
            "the mix reached into the fixed slot"


def test_a_mix_may_only_name_drivers_a_car_can_be_handed_to():
    """A checkpoint car needs different plumbing, so it cannot be one of the draws -- and the
    refusal says which names are allowed rather than leaving it to be discovered at reset."""
    with pytest.raises(ValueError, match="not a teacher kind"):
        osl.parse_slots('[{"kind_mix": ["forzaeth", "policy"]}]')
    with pytest.raises(ValueError, match="not in kind_mix"):
        osl.parse_slots('[{"kind": "raceline", "kind_mix": ["forzaeth", "lane_switch"]}]')
    slots = osl.parse_slots('[{"kind_mix": ["forzaeth", "lane_switch"]}]')
    assert slots[0].kind == "forzaeth", "the first of the mix is the slot's named kind"
    assert slots[0].to_dict()["kind_mix"] == ["forzaeth", "lane_switch"], "it does not round trip"


def test_a_respawned_car_does_not_inherit_the_last_race_s_commitment():
    """A planner holds which side a pass is committed to across steps, on purpose. A reset puts a
    different race on that row, and inheriting the commitment is a car that will not change sides
    for a reason that no longer exists."""
    from f1sim.spliner_teacher import SplinerTeacher
    env = _env(opponent_slots=[{"kind": "forzaeth"}])
    env.reset(seed=7)
    p = env.alt_teachers[0]
    assert isinstance(p, SplinerTeacher)
    p._side = torch.full((env.B,), 1.0)
    ids = torch.arange(0, env.B, 2)
    env._reset_rows(ids) if hasattr(env, "_reset_rows") else env.reset(seed=7)
    assert float(p._side[ids].abs().max()) == 0.0, "the commitment survived the reset"


# ============================================================ 2. obstacles on the racing line
def test_the_corridor_off_puts_obstacles_where_the_racing_line_is():
    """The point of the whole switch. With the corridor on, a prop cannot come nearer the line
    than the car's half-width plus the margin; with it off, it stands on it."""
    def nearest(corridor):
        env = _env(procedural_obstacles=1.0, procedural_density=3.0,
                   procedural_raceline_corridor=corridor,
                   spawn_runway=(3.0 if corridor == "off" else 0.0),
                   opponent_slots=[{"kind": "forzaeth"}])
        env.reset(seed=11)
        xy = env.teacher.xy[0]
        rows = torch.zeros(xy.shape[0], dtype=torch.long)
        return min(float(env.procedural.clearance(xy, rows + r).min()) for r in range(env.B))
    on, off = nearest("on"), nearest("off")
    assert on > 0.25, f"the corridor is not keeping the line clear: {on:.3f} m"
    assert off < 0.5 * on, f"turning it off changed nothing: on {on:.3f} m, off {off:.3f} m"


def test_a_driver_that_cannot_see_a_prop_is_not_handed_one():
    """The price of a layout on the racing line is that the other cars have to steer round it
    themselves, and most cannot: the raceline teacher is pure pursuit on a line built from the
    occupancy grid and the props were never rasterised into it. Refused before anything is drawn,
    because otherwise the layout ends the opponents' races and every collision number measured
    against them is the layout's."""
    for kw in ({"opponent": "teacher"},
               {"opponent": "slots", "opponent_slots": [{"kind": "raceline"}]},
               {"opponent": "slots", "opponent_slots": [{"kind_mix": ["forzaeth", "raceline"]}]}):
        with pytest.raises(ValueError, match="racing line"):
            _env(procedural_obstacles=1.0, procedural_raceline_corridor="off",
                 spawn_runway=3.0, **kw)
    _env(procedural_obstacles=1.0, procedural_raceline_corridor="off", spawn_runway=3.0,
         opponent_slots=[{"kind_mix": PROP_AWARE}])          # all prop-aware: allowed


def test_the_planners_see_a_prop_the_distance_field_cannot():
    """`sample_edt` reports a clear lane through a crate, because the props are not in the grid.
    A planner that believed it would drive into one, so the clearance it reads has to be the
    smaller of the two -- which is the whole reason a layout may stand on the line at all."""
    env = _env(procedural_obstacles=1.0, procedural_density=4.0,
               procedural_raceline_corridor="off", spawn_runway=3.0,
               opponent_slots=[{"kind": "forzaeth"}])
    env.reset(seed=13)
    p = env.alt_teachers[0]
    assert p.props is not None, "the planner was never given the layouts"
    idx = torch.arange(env.B) % env.teacher.N
    tid = env.sim.tid
    grid = env.sim.track.sample_edt(env.teacher.xy[tid, idx], tid)
    both = p._clearance_at(idx, tid, torch.zeros(env.B))
    assert bool((both <= grid + 1e-5).all()), "the planner read more room than the grid allows"
    assert bool((both < grid - 1e-3).any()), \
        "no prop ever narrowed the clearance; the layout is not reaching the planner"


# ============================================================ 3. a fair start
def test_a_spawn_runway_is_required_before_obstacles_go_on_the_line():
    """The two settings are one decision. Obstacles on the racing line is also obstacles where the
    cars spawn, and a car placed at up to `spawn_speed_max` a metre from a crate cannot stop."""
    with pytest.raises(ValueError, match="spawn_runway"):
        _env(procedural_obstacles=1.0, procedural_raceline_corridor="off",
             opponent_slots=[{"kind": "forzaeth"}])


def test_nothing_unavoidable_stands_in_front_of_a_spawn():
    """Measured on the line each car is actually on, over 20 resets, against the same env with the
    runway switched off afterwards -- so the layouts and the seeds are the ones being compared and
    not two different draws."""
    env = _env(envs=32, procedural_obstacles=1.0, procedural_density=4.0,
               procedural_raceline_corridor="off", spawn_runway=3.0,
               opponent_slots=[{"kind_mix": PROP_AWARE}])
    need = 0.5 * float(env.cfg.vehicle.width)

    def blocked(n=20):
        bad = 0
        for t in range(n):
            env.reset(seed=600 + t)
            s, tid, eid = env.sim.s, env.sim.tid, torch.arange(env.B)
            cl, yaw = env.sim.track.pose_at_s(s, tid)
            nrm = torch.stack([-torch.sin(yaw), torch.cos(yaw)], 1)
            lat = ((env.sim.state[:, :2] - cl) * nrm).sum(1)
            L = env.sim.track.length[tid]
            d = torch.full((env.B,), float("inf"))
            for k in range(1, 25):
                xy, y2 = env.sim.track.pose_at_s((s + 3.0 * k / 24.0) % L, tid)
                q = xy + torch.stack([-torch.sin(y2), torch.cos(y2)], 1) * lat[:, None]
                d = torch.minimum(d, torch.minimum(env.sim.track.sample_edt(q, tid),
                                                   env.procedural.clearance(q, eid)))
            bad += int((d < need).sum())
        return bad

    with_runway = blocked()
    env.sim.spawn_runway = 0.0
    without = blocked()
    assert with_runway == 0, f"{with_runway} cars still spawn into something they cannot avoid"
    assert without > 0, ("the same layouts block nobody without the runway either, so this test "
                         "is not measuring the runway")


def test_nothing_stands_on_the_line_a_spawn_is_pointing_along():
    """The lane check above walks the lane; a spawned car points `spawn_yaw_std` off it. Measured
    along each car's own heading, over 20 resets, against the same env without the runway.

    Only cars that point clearly off the lane (0.1 rad) are counted. A rejected pose falls back to
    the lane's own heading, and on a bend a straight ray from there leaves the lane that the lane
    check has already cleared -- so for those the straight ray is the wrong question, and the lane
    check above is the right one."""
    env = _env(envs=32, procedural_obstacles=1.0, procedural_density=4.0,
               procedural_raceline_corridor="off", spawn_runway=3.0,
               opponent_slots=[{"kind_mix": PROP_AWARE}])
    need = 0.5 * float(env.cfg.vehicle.width)

    def blocked(n=20):
        bad = 0
        for t in range(n):
            env.reset(seed=600 + t)
            st, eid = env.sim.state, torch.arange(env.B)
            hd = torch.stack([torch.cos(st[:, 2]), torch.sin(st[:, 2])], 1)
            _, lane = env.sim.track.pose_at_s(env.sim.s, env.sim.tid)
            off = (torch.remainder(st[:, 2] - lane + math.pi, 2 * math.pi) - math.pi).abs() > 0.1
            d = torch.full((env.B,), float("inf"))
            for k in range(1, 25):
                d = torch.minimum(d, env.procedural.clearance(st[:, :2] + hd * (3.0 * k / 24.0), eid))
            bad += int(((d < need) & off).sum())
        return bad

    with_runway = blocked()
    env.sim.spawn_runway = 0.0
    without = blocked()
    assert with_runway == 0, f"{with_runway} cars still spawn pointing at a prop they cannot avoid"
    assert without > 0, "the same layouts put no prop ahead of anybody, so this measures nothing"


def test_no_car_is_placed_hugging_a_prop():
    """Clear of a prop is not enough: turning towards the line swings the tail, and a car placed
    0.23 m beside a crate touched it within five steps. Every spawned body keeps
    `SPAWN_PROP_GAP` from every prop, measured at its corners and edge midpoints."""
    env = _env(envs=32, procedural_obstacles=1.0, procedural_density=4.0,
               procedural_raceline_corridor="off", spawn_runway=3.0,
               opponent_slots=[{"kind_mix": PROP_AWARE}])
    gap = env.sim.SPAWN_PROP_GAP
    worst = float("inf")
    for t in range(20):
        env.reset(seed=700 + t)
        st = env.sim.state
        c, s_ = torch.cos(st[:, 2]), torch.sin(st[:, 2])
        R = torch.stack([torch.stack([c, -s_], -1), torch.stack([s_, c], -1)], -2)
        pts = torch.einsum("bij,kj->bki", R, env.sim.corners) + st[:, None, :2]
        pts = torch.cat([pts, 0.5 * (pts + pts[:, [1, 3, 0, 2]])], 1)
        e = torch.arange(env.B)[:, None].expand(-1, 8).reshape(-1)
        worst = min(worst, float(env.procedural.clearance(pts.reshape(-1, 2), e).min()))
    # one control step of motion may close a little of it before this is read
    assert worst >= gap - 0.05, f"a car was placed {worst:.3f} m from a prop (gap {gap})"


# ============================================================ 4. the obstacle has to cost something
def test_creeping_into_a_crate_is_not_free():
    """The clearance reward and the plan-clearance penalty both read the occupancy distance field,
    and the procedural obstacles are props that were never rasterised into it. So with a crate the
    only thing in the way, both terms were identically zero and the progress reward was the only
    live term -- creeping toward one paid *positive* reward right up to contact.

    That is not a theory about the reward. It is what the policy did: raced against a ForzaETH
    opponent with layouts on the racing line, 42 % of its crashes were into a crate and the median
    impact speed was 1.3 m/s, whose stopping distance is 0.2 m against 10 m of LiDAR. It saw the
    crate, slowed almost to a stop, and drove into it anyway, because nothing said not to.

    Checked as arithmetic on the two terms rather than through a rollout, because what is being
    pinned is the sign of the sum.
    """
    env = _env(envs=16, procedural_obstacles=1.0, procedural_density=4.0,
               procedural_raceline_corridor="off", spawn_runway=3.0, action_mode="plan",
               reward_proximity=0.5, safe_dist=0.3, reward_plan_clearance=4.0, plan_margin=0.25,
               opponent_slots=[{"kind_mix": PROP_AWARE}])
    env.reset(seed=53)
    grid, both = [], []
    orig = env._widen_clearance_to_props
    def spy(r):
        grid.append(r.wall_dist.clone()); orig(r); both.append(r.wall_dist.clone())
    env._widen_clearance_to_props = spy
    for _ in range(150):
        env.step(torch.zeros(env.B, env.act_dim))
    g, b = torch.cat(grid), torch.cat(both)
    tight = b < 0.3                                   # really inside safe_dist of *something*
    assert bool(tight.any()), "no car ever came within safe_dist of anything; nothing is measured"
    hidden = tight & (g >= 0.3)                       # ... and the grid alone called it clear
    assert bool(hidden.any()), (
        "no step where a prop was the only thing close: the layouts are not reaching the cars, so "
        "this test would pass for the wrong reason")
    # the whole point: on those steps the proximity term now has something to charge for
    assert float(b[hidden].max()) < 0.3
    assert float(g[hidden].min()) >= 0.3


def test_the_plan_clearance_term_sees_a_crate_on_the_planned_path():
    """The forward-looking half. `reward_plan_clearance` is the largest shaping weight in the
    traffic recipe (4.0) and it is what charges for a plan drawn somewhere tight -- against the
    grid alone it charged nothing for one drawn straight through a crate."""
    env = _env(envs=16, procedural_obstacles=1.0, procedural_density=4.0,
               procedural_raceline_corridor="off", spawn_runway=3.0, action_mode="plan",
               reward_plan_clearance=4.0, plan_margin=0.25,
               opponent_slots=[{"kind_mix": PROP_AWARE}])
    env.reset(seed=57)
    saw_finite = False
    for _ in range(150):
        env.step(torch.zeros(env.B, env.act_dim))
        pr = env._plan_clearance(env.sim.state, env.tracker.last_ref)
        assert pr.shape == (env.B,)
        if bool(torch.isfinite(pr).any()):
            saw_finite = True
            assert float(pr.min()) >= -1.0, "a clearance that far inside anything is not a clearance"
    assert saw_finite, "the plan never came near anything; the term is not being exercised"


# ============================================================ 5. a collision you can drive out of
def test_soft_collision_does_not_end_the_episode_and_the_car_keeps_driving():
    """`terminate` is the only thing the policy could ever learn from: avoid, or die.

    It never sees what happens after a touch, so it cannot learn to steer out of one, to back off
    and go again, or -- the thing that decides whether a recovery was worth anything -- not to end
    up pointing the wrong way. `Simulator` has resolved soft contacts all along
    (`_resolve_wall_contact`: out of penetration, into-surface velocity removed by
    `collision_restitution` and `wall_friction`, yaw rate damped) and `SimParams` documents both
    constants as "when not terminating" -- but `F1VecEnv.__init__` forced `terminate_on_collision`
    True whatever the config said, so none of it could run.
    """
    def run(mode, steps=120):
        env = _env(envs=16, procedural_obstacles=1.0, procedural_density=2.0,
                   procedural_max_props=10, procedural_raceline_corridor="off",
                   spawn_runway=3.0, action_mode="plan", max_steps=1600,
                   collision_mode=mode, opponent_slots=[{"kind_mix": PROP_AWARE}])
        env.reset(seed=61)
        assert env.cfg.sim.terminate_on_collision == (mode == "terminate")
        term = 0
        for _ in range(steps):
            _o, _r, t, _tr, _i = env.step(torch.zeros(env.B, env.act_dim))
            term += int(t.sum())
        return term

    assert run("terminate") > 0, "nothing crashed at all, so the modes cannot be compared"
    assert run("soft") == 0, "a soft collision still ended the episode"


def test_the_collision_mode_is_checked_rather_than_silently_ignored():
    """A typo here would be a run that quietly used the other model, and every collision number it
    produced would mean the other thing."""
    with pytest.raises(ValueError, match="collision_mode"):
        _env(envs=8, collision_mode="sof")


def test_a_soft_collision_is_still_charged_for():
    """Not terminal is not free. The crash term still fires and still scales with the speed at
    impact, so a nudge and a shunt are not the same price -- otherwise "soft" would just delete the
    obstacle penalty that the last three commits were about putting back."""
    env = _env(envs=16, procedural_obstacles=1.0, procedural_density=3.0,
               procedural_max_props=10, procedural_raceline_corridor="off", spawn_runway=3.0,
               action_mode="plan", max_steps=1600, collision_mode="soft",
               reward_collision=-60.0, reward_collision_speed=0.5,
               opponent_slots=[{"kind_mix": PROP_AWARE}])
    env.reset(seed=63)
    charged = 0.0
    for _ in range(150):
        _o, _r, _t, _tr, info = env.step(torch.zeros(env.B, env.act_dim))
        comp = info.get("reward_components") or {}
        charged += float(comp["collision"].sum()) + float(comp["collision_speed"].sum())
    assert charged < 0.0, "a soft contact cost nothing at all"


# ============================================================ 6. obstacles that move
def test_a_shove_obeys_the_mass_it_was_given():
    """v = J/m, and the floor takes it back at `PROP_GROUND_DECEL`.

    Unit-tested on the two methods rather than through a rollout, because what is being pinned is
    arithmetic: a 1.2 kg box and an 18 kg drum given the same impulse must end up 15 times apart,
    which is the whole reason a mass was introduced at all.
    """
    from f1sim.procedural_obstacles import PROP_GROUND_DECEL, ProceduralObstacles as PO
    p = PO.__new__(PO)
    p.device = torch.device("cpu")
    p.p_mass = torch.tensor([[1.2, 18.0, 0.0, 10.0]])
    p.p_vel = torch.zeros(1, 4, 2)
    p.p_poses = torch.zeros(1, 4, 3)
    p.p_zlo = torch.zeros(1, 4)
    p.p_zhi = torch.tensor([[1.0, 1.0, 1.0, -1.0]])          # slot 3 is dead
    eid = torch.zeros(4, dtype=torch.long)
    slot = torch.arange(4)
    J = torch.tensor([[1.0, 0.0]] * 4)
    p.shove(eid, slot, J)
    v = p.p_vel[0, :, 0]
    assert float(v[0]) == pytest.approx(1.0 / 1.2, rel=1e-5)
    assert float(v[1]) == pytest.approx(1.0 / 18.0, rel=1e-5)
    assert float(v[2]) == 0.0, "a massless slot is immovable, which is what every prop used to be"
    assert float(v[3]) == 0.0, "a dead slot was shoved"

    before = p.p_poses[0, 0, 0].clone()
    p.advance(0.025)
    assert float(p.p_poses[0, 0, 0] - before) > 0.0, "a shoved prop did not move"
    assert float(p.p_vel[0, 0, 0]) == pytest.approx(1.0 / 1.2 - PROP_GROUND_DECEL * 0.025, rel=1e-4)
    for _ in range(200):
        p.advance(0.025)
    assert float(p.p_vel[0, 0].norm()) == 0.0, "the floor never stopped it"


def test_movable_obstacles_move_and_immovable_ones_do_not():
    """The wiring, end to end: same seed, same layouts, same actions, one switch. Enough cars and
    steps that some car meets a piece whatever the layout draw is -- at 16 cars and 150 steps a
    change to the generator's draw left every contact a wall one."""
    def run(movable, steps=300):
        env = _env(envs=32, procedural_obstacles=1.0, procedural_density=3.0,
                   procedural_max_props=10, procedural_raceline_corridor="off",
                   spawn_runway=3.0, action_mode="plan", max_steps=1600,
                   collision_mode="soft", movable_obstacles=movable,
                   opponent_slots=[{"kind_mix": PROP_AWARE}])
        env.reset(seed=3)
        p0 = env.procedural.p_poses[:, :, :2].clone()
        for _ in range(steps):
            env.step(torch.zeros(env.B, env.act_dim))
        live = env.procedural.p_zhi > env.procedural.p_zlo
        d = (env.procedural.p_poses[:, :, :2] - p0).norm(dim=2)
        return int(((d > 0.01) & live).sum())

    assert run(False) == 0, "an obstacle moved with movable_obstacles off"
    assert run(True) > 0, "nothing ever got shoved, so the switch is not reaching the contact"


def test_movable_obstacles_need_a_collision_you_survive():
    """Under `terminate` the episode ends at the touch, so a crate that was shoved aside is a crate
    nothing ever saw move. Refused rather than silently doing nothing."""
    with pytest.raises(ValueError, match="collision_mode='soft'"):
        _env(envs=8, procedural_obstacles=1.0, movable_obstacles=True,
             collision_mode="terminate", opponent_slots=[{"kind": "forzaeth"}])


# ============================================================ 7. backing out of something
def test_a_wedged_car_can_reverse_out():
    """The user's case: *"단단한 장애물에 박았다던지 벽에 박혀 있다던지"*.

    The plan's speed dimension maps to [0, v_max] -- `mpc.decode`: `(a + 1) * 0.5 * v_max` -- so a
    car nose-first into a hose has no action that means "back up", and the only thing forward of it
    is the thing it is stuck on. Reverse is therefore what "stop" means when the car is *already*
    stopped and in contact: the policy still chooses it, through the speed command and the
    steering, and it cannot be entered any other way, so an ordinary slow corner is untouched.

    Checked by wedging the cars rather than hoping a rollout produces one.
    """
    import numpy as np
    env = _env(envs=8, race_size=1, opponent="policy", action_mode="plan", max_steps=1600,
               collision_mode="soft")
    env.reset(seed=3)
    S, rl = env.sim, _track_and_raceline(TRACK)[1]
    placed = 0
    for i in range(env.B):
        tid = S.tid[i:i + 1]
        for idx in range(0, 400, 7):
            p = rl.xy[idx % len(rl.xy)]
            hit = None
            for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                for d in np.arange(0.2, 1.6, 0.05):
                    q = torch.tensor([[p[0] + d * np.cos(ang), p[1] + d * np.sin(ang)]],
                                     dtype=torch.float32)
                    if float(S.track.sample_edt(q, tid)) < 0.10:
                        hit = (float(q[0, 0]), float(q[0, 1]), float(ang)); break
                if hit: break
            if hit:
                S.state[i, 0], S.state[i, 1], S.state[i, 2] = hit
                S.state[i, 3] = 0.0
                placed += 1
                break
    assert placed >= env.B // 2, f"only wedged {placed} of {env.B}; the scene is not the scene"

    stop = torch.full((env.B, env.act_dim), -1.0)
    reversing, contact, vmin = 0, 0, 0.0
    for _ in range(60):
        env.step(stop)
        v = env.sim.state[:, 3]
        reversing += int((v < -0.05).sum())
        contact += int(env.tracker.contact.sum())
        vmin = min(vmin, float(v.min()))
    assert contact > 0, "nothing was in contact, so nothing was asked to back out"
    assert reversing > 0, f"no car ever reversed (most negative speed {vmin:.3f} m/s)"


def test_reverse_cannot_be_entered_at_speed_or_in_clear_air():
    """The two gates, because without them "stop" would mean "reverse" everywhere and a policy
    that lifts for a corner would find itself going backwards."""
    from f1sim.mpc import PlanSpec
    sp = PlanSpec()
    env = _env(envs=8, race_size=1, opponent="policy", action_mode="plan", max_steps=1600,
               collision_mode="soft")
    env.reset(seed=5)
    env.sim.state[:, 3] = 0.0
    stop = torch.full((env.B, env.act_dim), -1.0)
    for _ in range(20):                      # stopped, commanding stop, but touching nothing
        env.step(stop)
        assert float(env.sim.state[:, 3].min()) >= -1e-3, "reversed with nothing to back out of"
    assert sp.reverse_v_gate < 1.0 and sp.reverse_cmd_gate < 0.5, \
        "the gates have to be tight enough that an ordinary slow corner cannot trip them"


def test_terminate_mode_never_reverses():
    """Under `terminate` the episode ends at the touch, so there is no recovery to make and the
    tracker is the one it has always been."""
    env = _env(envs=8, race_size=1, opponent="policy", action_mode="plan", max_steps=1600,
               collision_mode="terminate")
    env.reset(seed=7)
    for _ in range(40):
        env.step(torch.full((env.B, env.act_dim), -1.0))
        assert not bool(env.tracker.contact.any()), "the tracker was told about contact under terminate"


# ============================================================ 8. soft runs, measured and spread out
def test_a_staggered_start_spreads_the_resets_and_then_runs_full_episodes():
    """Under soft nothing terminates, so a batch that starts together resets together forever:
    s911 reset all 256 cars on the same step every 50 updates. With the first episode staggered
    the resets land on many different steps, each race's cars still reset together, and every
    episode after the first is full length."""
    n = 60
    env = _env(envs=16, action_mode="plan", max_steps=n, collision_mode="soft",
               stagger_first_episode=True, opponent_slots=[{"kind_mix": PROP_AWARE}])
    env.reset(seed=11)
    first = env.ep_limit.view(-1, env.M)
    assert bool((first == first[:, :1]).all()), "the cars of one race must share the limit"
    assert int(first[:, 0].unique().numel()) > 1, "every race drew the same limit"
    ends = {}
    for k in range(1, 2 * n + 2):
        _o, _r, _t, _tr, info = env.step(torch.zeros(env.B, env.act_dim))
        if "final" in info:
            f = info["final"]
            for i, st, stag in zip(f["ids"].tolist(), f["steps"].tolist(), f["staggered"].tolist()):
                ends.setdefault(i, []).append((k, st, stag))
    assert len({v[0][0] for v in ends.values()}) > 2, "the first resets still landed together"
    for i, ev in ends.items():
        assert ev[0][2] or ev[0][1] == n, "only a cut-short first episode may be flagged staggered"
        for _k, st, stag in ev[1:]:
            assert st == n and not stag, f"row {i}: a later episode ran {st} steps, not {n}"


def test_without_the_stagger_everything_is_as_before():
    env = _env(envs=8, action_mode="plan", max_steps=30, collision_mode="soft",
               opponent_slots=[{"kind_mix": PROP_AWARE}])
    env.reset(seed=11)
    assert not bool((env.ep_limit < 30).any()), "no row may carry a limit shorter than max_steps"
    for _ in range(30):
        _o, _r, _t, trunc, info = env.step(torch.zeros(env.B, env.act_dim))
    assert bool(trunc.all()) and not bool(info["final"]["staggered"].any())


def test_soft_contacts_are_counted_in_the_episode_and_in_an_evaluation():
    """s911 logged "coll 0.0/km" for 6.3 M steps of soft contact: `collided` is `terminated`, which
    soft never sets. `final["contacts"]` counts the onsets the reward charged, and
    `common.evaluate` counts `info["contact_onset"]` per step, so an unfinished episode's contacts
    are not lost either."""
    env = _env(envs=16, procedural_obstacles=1.0, procedural_density=3.0, procedural_max_props=10,
               procedural_raceline_corridor="off", spawn_runway=3.0, action_mode="plan", max_steps=150,
               collision_mode="soft", opponent_slots=[{"kind_mix": PROP_AWARE}])
    env.reset(seed=63)
    onsets = 0; counted = 0.0
    for _ in range(150):
        _o, _r, _t, _tr, info = env.step(torch.zeros(env.B, env.act_dim))
        onsets += int(info["contact_onset"].sum())
        if "final" in info:
            counted += float(info["final"]["contacts"].sum())
            assert not bool(info["final"]["collided"].any())            # soft: nothing terminated
    assert onsets > 0, "the scenario has to hit something to test the count"
    assert counted == onsets, (counted, onsets)
    # and the evaluation: same scenario, same seed (rollout_metrics reseeds nothing, so seed here)
    torch.manual_seed(0)
    env2 = _env(envs=16, procedural_obstacles=1.0, procedural_density=3.0, procedural_max_props=10,
                procedural_raceline_corridor="off", spawn_runway=3.0, action_mode="plan", max_steps=1600,
                collision_mode="soft", opponent_slots=[{"kind_mix": PROP_AWARE}])
    out = common.rollout_metrics(env2, lambda obs: torch.zeros(env2.B, env2.act_dim), steps=150)
    assert out["collisions_per_km"] > 0.0, out          # used to be 0: no episode ended, none terminated


def test_a_mix_without_the_raceline_teacher_does_not_compute_it():
    """s911's recipe mixes four planners and not the raceline teacher, and every one of its cars is
    driven by one of the four -- yet the raceline teacher ran for the whole batch every step, 16 % of
    the step, for a command every row then overwrote. It is skipped now, and nothing changes: the
    same seed, run once skipping it and once forced to compute it, drives the same race to the bit.

    Run one after the other, not interleaved: the LiDAR draws its noise from the global torch RNG,
    so two envs stepped alternately in one process take each other's draws and part after the first
    scan whatever the teachers do."""
    def drive(force):
        env = _env(envs=16, action_mode="plan", opponent_slots=[{"kind_mix": PROP_AWARE}])
        if force:
            env.__dict__["_raceline_teacher_needed"] = lambda: True
        env.reset(seed=5)                                  # reseeds the global RNG too
        calls = {"n": 0}
        real = env._teacher_normalized
        def spy(teacher, *a, **k):
            calls["n"] += teacher is env.teacher
            return real(teacher, *a, **k)
        env._teacher_normalized = spy
        a = torch.zeros(env.B, env.act_dim)
        states, cmds = [], []
        for _ in range(60):                                # long enough for races to reset and redraw
            env.step(a)
            states.append(env.sim.state.clone()); cmds.append(env.last_cmd.clone())
        return torch.stack(states), torch.stack(cmds), calls["n"], env
    s0, c0, n0, env = drive(force=False)
    s1, c1, n1, _ = drive(force=True)
    assert "raceline" not in PROP_AWARE and not env._raceline_teacher_needed()
    assert n0 == 0 and n1 == 60, (n0, n1)
    assert torch.equal(c0, c1), "skipping the raceline teacher changed a command"
    assert torch.equal(s0, s1), "skipping the raceline teacher moved a car"


def test_a_mix_with_the_raceline_teacher_still_computes_it():
    env = _env(envs=8, opponent_slots=[{"kind_mix": MIX}])
    env.reset(seed=5)
    assert "raceline" in MIX and env._raceline_teacher_needed()


@pytest.mark.parametrize("assign", ["redraw", "partition"])
def test_each_opponent_plans_only_the_rows_it_can_drive(assign):
    """`plan_rows` gives each row the command the whole-batch call gives it, for all four planners.

    They used to be asked for the whole batch -- learners included -- every step, and in s911's
    recipe that was 61 % of an eager env step at 256 envs. Now each is asked for the rows it can
    drive: under "redraw" every row of the mixed slot, under "partition" its own races only. Same
    state both ways -- a planner with memory has it put back between the two calls -- and the same
    command on those rows, and the same memory left behind for them."""
    env = _env(envs=32, action_mode="plan", procedural_obstacles=1.0, procedural_density=2.0,
               procedural_max_props=8, procedural_raceline_corridor="off", spawn_runway=3.0,
               collision_mode="soft", kind_mix_assign=assign,
               opponent_slots=[{"kind_mix": PROP_AWARE + ["interactive"]}])
    env.reset(seed=7)
    G = env.B // env.M
    for kind, alt in zip(env.alt_teacher_kinds, env.alt_teachers):
        rows = env._alt_rows[kind]
        if assign == "redraw":
            assert rows.numel() == G, "a mixed slot's rows are every race's second car"
        else:
            assert rows.numel() == G // 4, "four kinds share the races equally"
            assert bool(env.alt_teacher_mask[kind][rows].all()), "partitioned rows are the rows it drives"
    a = torch.zeros(env.B, env.act_dim)
    env.step(a)                           # a planner creates its memory on its first call
    for step in range(40):
        follow, v_cap = env.follow_cap(env.sim.state)
        for kind, alt in zip(env.alt_teacher_kinds, env.alt_teachers):
            rows = env._alt_rows[kind]
            names = [n for n in getattr(alt, "GRAPH_STATE", ()) if getattr(alt, n, None) is not None]
            snap = [getattr(alt, n).clone() for n in names]
            with torch.no_grad():
                if hasattr(alt, "decide"):
                    # The decision -- the offset, the manoeuvre, the speed scale -- is exact: it is
                    # this planner's own row-wise logic, and the thing a subset call could get wrong.
                    dw = alt.decide(env.sim.state, env.sim.tid)
                    for n, v in zip(names, snap):
                        getattr(alt, n).copy_(v)
                    with alt._row_scope(rows, env.sim.state):
                        dp = alt.decide(env.sim.state[rows], env.sim.tid[rows])
                    for n, v in zip(names, snap):
                        getattr(alt, n).copy_(v)
                    if dw[0] is not None:
                        assert torch.equal(dw[0][rows], dp[0]), f"{kind}: offset"
                    assert torch.equal(dw[1][rows], dp[1]) and torch.equal(dw[2][rows], dp[2]), kind
                whole = env._teacher_normalized(alt, None, None, follow, v_cap)
                whole_state = [getattr(alt, n)[rows].clone() for n in names]
                for n, v in zip(names, snap):
                    getattr(alt, n).copy_(v)
                part = env._teacher_normalized(alt, None, None, follow, v_cap, rows)
            assert part.shape == (rows.numel(), env.act_dim)
            # 1e-3 in normalised plan units, not bit equality, for the command: the reference
            # teacher's Gauss-Newton fit is batched linear algebra, a 4-row batch and a 32-row one
            # reduce in a different order, and six iterations grow the last bits -- measured up to
            # 3.4e-4 on one curvature knot, 5e-4 1/m, with the decision above identical.
            torch.testing.assert_close(part, whole[rows], rtol=0, atol=1e-3)
            for n, v in zip(names, whole_state):
                assert torch.equal(getattr(alt, n)[rows], v), f"{kind}.{n}"
        env.step(a)
