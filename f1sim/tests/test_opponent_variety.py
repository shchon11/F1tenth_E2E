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
