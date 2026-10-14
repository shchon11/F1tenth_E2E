"""What a v2 env exposes for the v2 critic and the v2 actor's auxiliary heads.

* `privileged_v2` -- the critic's vector: versioned beside v1's `privileged()` (which keeps its 21
  columns), every column O(1), opponents nearest first, the align gate's explicit geometry.
* per-seat LiDAR visibility -- which seat each car beam of the newest scan landed on, and how long
  ago each seat was last seen. The labels of the opponent-slot head (B8) and of the nearest-visible-
  car head are built on it, and its age is what the trainer's knowability mask reads.

All of these build a real simulator (CPU, a few seconds each).
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.gym_env import (OPP_NEVER_SEEN, OPP_SEEN_BEAMS, PRIV_V2_DIM, PRIV_V2_KEYS,  # noqa: E402
                           PRIV_V2_MIRROR, PRIV_V2_OPP_KEYS)

pytestmark = pytest.mark.slow


def _env(race_size=2, envs=2, **kw):
    from f1sim import Config
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    tracks, _rls = common.load_tracks(["gen:competition:0"], racelines=True)
    cfg = Config(); cfg.rand.enabled = False; cfg.sim.compile = False
    args = dict(action_mode="plan", speed_mode="wp4", race_size=race_size, opponent="policy",
                hist_len=20, hist_stride=2, scan_stack=1, compile_tracker=False, obs_version=2,
                collision_mode="soft", opp_visibility=True)
    args.update(kw)
    return common.make_env(tracks, envs, "cpu", EnvConfig(**args), cfg=cfg, seed=2)


def _drive(env, steps):
    for t in range(steps):
        a = torch.zeros(env.B, env.act_dim)
        a[:, 0] = 0.1 * math.sin(0.2 * t)
        a[:, 4:] = -0.3
        env.step(a)


def _rescan(env):
    """The StepResult of the env's current state, rescanned (no motion distortion): what a step
    ending in this pose would have handed the labels."""
    sim = env.sim
    scan, true, typ = sim.lidar.scan(sim.state[:, :3], None, sim.P, motion_distortion=False, tid=sim.tid,
                                     cars=sim._car_boxes(sim.state), compiled=False, eid=sim.eid)
    return dataclasses.replace(env.last_result, scan=scan, scan_true=true, scan_type=typ,
                               state=sim.state.clone())


def _place(env, row, ahead, left, dyaw=0.0, speed=None):
    """Put `row`'s car `ahead` m in front of and `left` m beside row 0's (row 0's frame)."""
    st = env.sim.state
    x, y, yaw = (float(v) for v in st[0, :3])
    c, s = math.cos(yaw), math.sin(yaw)
    st[row, 0] = x + ahead * c - left * s
    st[row, 1] = y + ahead * s + left * c
    st[row, 2] = yaw + dyaw
    if speed is not None:
        st[row, 3] = speed


# ---------------------------------------------------------------- the critic's vector
def test_privileged_v2_is_versioned_bounded_and_ordered():
    # ego, three opponents, the geometry -- no speed-cap column since 2026-10-14 (it served the
    # learner's random cap; the env's own cap is one number for every row)
    assert len(PRIV_V2_KEYS) == len(set(PRIV_V2_KEYS)) == PRIV_V2_DIM == 8 + 3 * 7 + 4
    assert "speed_cap" not in PRIV_V2_KEYS
    assert len(PRIV_V2_MIRROR) == PRIV_V2_DIM
    env = _env(race_size=4, envs=8)
    env.reset(seed=2)
    _drive(env, 20)
    pv = env.privileged_v2()
    assert pv.shape == (8, PRIV_V2_DIM)
    assert torch.isfinite(pv).all() and float(pv.abs().max()) <= 6.0 + 1e-5
    assert env.privileged(env.last_result).shape == (8, 8 + 4 + 8 + 1), "v1's vector must not move"
    col = {k: i for i, k in enumerate(PRIV_V2_KEYS)}
    d = pv[:, [col[f"opp{j}_dist"] for j in range(3)]]
    assert bool((d[:, 0] <= d[:, 1]).all() and (d[:, 1] <= d[:, 2]).all()), "not nearest first"
    assert torch.equal(pv[:, col["near_dist"]], pv[:, col["opp0_dist"]])
    assert bool(((pv[:, col["min_ttc"]] <= pv[:, col["near_ttc"]] + 1e-6)).all())


def test_a_solo_car_reads_no_opponent():
    env = _env(race_size=1, envs=2, opp_visibility=False)
    env.reset(seed=2)
    _drive(env, 3)
    pv = env.privileged_v2()
    k = len(PRIV_V2_OPP_KEYS)
    assert pv[:, 8:8 + 3 * k].abs().sum() == 0
    assert pv[:, 8 + 3 * k:8 + 3 * k + 4].tolist() == [[6.0, 0.0, 1.0, 1.0]] * 2


def test_the_geometry_sees_a_car_being_closed_on():
    env = _env(race_size=2, envs=2)
    env.reset(seed=2)
    _place(env, 1, 2.0, 0.0, speed=1.0)
    env.sim.state[0, 3] = 3.0; env.sim.state[0, 4] = 0.0
    pv = env.privileged_v2(_rescan(env))
    col = {k: i for i, k in enumerate(PRIV_V2_KEYS)}
    assert float(pv[0, col["opp0_dx"]]) == pytest.approx(2.0 / 5.0, abs=1e-4)
    assert float(pv[0, col["near_closing"]]) == pytest.approx(2.0 / 5.0, abs=1e-3)   # 3 - 1 m/s
    assert float(pv[0, col["near_ttc"]]) == pytest.approx(1.0 / 10.0, abs=1e-3)      # 2 m / 2 m/s


# ---------------------------------------------------------------- per-seat visibility and labels
def test_a_car_ahead_is_attributed_to_its_seat_and_one_behind_is_not_seen():
    env = _env(race_size=2, envs=2)
    env.reset(seed=2)
    _place(env, 1, 2.0, 0.3, speed=2.0)
    res = _rescan(env)
    beams = env._attribute_car_beams(res)
    car_valid = int(((res.scan_type[0] == 4) & env._beam_valid(res)[0]).sum())
    assert car_valid >= 10, "a car 2 m ahead should fill a good arc of the scan"
    assert int(beams[0, 0]) == car_valid, "every car return belongs to the only other car"
    env._update_opp_visibility(res, None)
    tr = env.opponent_tracks_v2(res)
    assert int(tr["age"][0, 0]) == 0 and bool(tr["cand"][0, 0])
    mx = float(env.sim.P["mount_x"][0]) if "mount_x" in env.sim.P else env.cfg.lidar.mount_x
    assert tr["lab"][0, 0].tolist() == pytest.approx([(2.0 - mx) / 5.0, 0.3 / 2.0, 0.0], abs=2e-3)
    nv = env.nearest_visible_car_v2(res)
    assert nv[0, 0] == 1.0 and float(nv[0, 1]) == pytest.approx((2.0 - mx) / 5.0, abs=2e-3)
    # straight behind is the LiDAR's 90-degree blind sector: present, close, never seen
    _place(env, 1, -1.5, 0.0)
    res = _rescan(env)
    assert int(env._attribute_car_beams(res)[0, 0]) == 0
    env._update_opp_visibility(res, None)
    tr = env.opponent_tracks_v2(res)
    assert bool(tr["cand"][0, 0]) and int(tr["age"][0, 0]) == 1      # seen one observation ago
    assert env.nearest_visible_car_v2(res)[0].abs().sum() == 0


def test_the_age_counts_steps_since_a_sighting_and_restarts_with_either_car(monkeypatch):
    env = _env(race_size=3, envs=6)
    env.reset(seed=2)
    beams = torch.zeros(env.B, 2, dtype=torch.long)
    monkeypatch.setattr(env, "_attribute_car_beams", lambda res: beams.clone())
    # everybody close and in range, so the beam count alone decides "seen"
    for row in range(1, 3):
        _place(env, row, 1.5 * row, 0.0)
    env._opp_age.fill_(OPP_NEVER_SEEN)
    res = _rescan(env)
    beams[0, 0] = OPP_SEEN_BEAMS
    env._update_opp_visibility(res, None)
    assert int(env._opp_age[0, 0]) == 0 and int(env._opp_age[0, 1]) == OPP_NEVER_SEEN
    beams.zero_()
    for k in range(1, 5):
        env._update_opp_visibility(res, None)
        assert int(env._opp_age[0, 0]) == k
    beams[0, 0] = OPP_SEEN_BEAMS - 1                                     # two beams are not a sighting
    env._update_opp_visibility(res, None)
    assert int(env._opp_age[0, 0]) == 5
    # the seat's car respawns: whatever row 0 remembered about it is about a different car now
    seat_car = int(env.sim.other_idx[0, 0])
    gone = torch.zeros(env.B, dtype=torch.bool); gone[seat_car] = True
    env._update_opp_visibility(res, gone)
    assert int(env._opp_age[0, 0]) == OPP_NEVER_SEEN
    # and row 0's own reset clears every track it held
    beams[0, :] = OPP_SEEN_BEAMS
    env._update_opp_visibility(res, None)
    assert env._opp_age[0].tolist() == [0, 0]
    gone = torch.zeros(env.B, dtype=torch.bool); gone[0] = True
    env._update_opp_visibility(res, gone)
    assert env._opp_age[0].tolist() == [0, 0], "seen on the very step it restarted: a fresh sighting"
    beams.zero_()
    env._update_opp_visibility(res, gone)
    assert env._opp_age[0].tolist() == [OPP_NEVER_SEEN] * 2


def test_visibility_is_tracked_through_real_steps_and_off_costs_nothing():
    env = _env(race_size=4, envs=8)
    env.reset(seed=2)
    _drive(env, 25)
    tr = env.opponent_tracks_v2()
    assert tr["age"].shape == (8, 3) and tr["lab"].shape == (8, 3, 3)
    assert bool(((tr["age"] == 0) == (tr["present"] & (tr["beams"] >= OPP_SEEN_BEAMS)
                                      & (tr["dist"] < 6.0))).all())
    # every attributed beam is a valid car return of that row's newest scan
    res = env.last_result
    car_valid = ((res.scan_type == 4) & env._beam_valid(res)).sum(1)
    assert bool((tr["beams"].sum(1) <= car_valid).all())
    off = _env(race_size=2, envs=2, opp_visibility=False)
    off.reset(seed=2)
    assert off._opp_age is None
    with pytest.raises(RuntimeError):
        off.opponent_tracks_v2()


def test_a_v2_step_says_whether_the_cap_clipped_the_plan():
    """Critique B5: the cap is not observed, so the trainer has to be told when it bit."""
    env = _env(race_size=1, envs=2, opp_visibility=False)
    env.set_speed_cap(4.0)
    env.reset(seed=2)
    a = torch.zeros(env.B, env.act_dim)
    a[0, 4:] = 1.0                                   # 10 m/s everywhere: clipped to 4
    a[1, 4:] = -0.5                                  # 2.5 m/s: under the cap
    _o, _r, _te, _tr, info = env.step(a)
    assert info["speed_capped"].tolist() == [True, False]
