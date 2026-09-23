"""Contacts counted per opportunity, with the crates told apart from the walls.

`collisions_per_km` answers "how often does it hit" multiplied by "how long does it scrape", and it
calls a cardboard box a wall. Both were load-bearing on 2026-09-23: four interventions were judged
against a number whose contacted-encounter multiplier swung 4.6x between draws of the same policy,
and `traffic["wall_collisions"]` reported 10 wall collisions in a race where 9 of the 10 were
crates.

Small tracks and a 36-beam LiDAR, the convention of `test_opponent_variety.py`.
"""
import functools

import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.learn.encounter import EncounterMeter

TRACK = "gen:competition:0"


@functools.lru_cache(maxsize=2)
def _track_and_raceline(name: str):
    from f1sim.raceline import Raceline
    track = maps.load(name)
    return track, Raceline.build_cached(track)


def _env(envs=16, seed=5, **cfg_kw):
    tr, rl = _track_and_raceline(TRACK)
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(**{"race_size": 1, "max_steps": 600, "hist_len": 0, "action_mode": "plan",
                        "collision_mode": "soft", "speed_cap": 6.0,
                        "procedural_obstacles": 1.0, "procedural_density": 2.0,
                        "procedural_max_props": 8, "procedural_raceline_corridor": "off",
                        "movable_obstacles": True, "spawn_runway": 2.0, **cfg_kw})
    return common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl]), rl


def test_no_obstacles_means_no_measurement_rather_than_a_zero():
    """A track with nothing on it has no encounters to report, and saying "0 contacts in 0
    approaches" would average into a pooled table as a perfect score."""
    env, _ = _env(procedural_obstacles=0.0)
    with EncounterMeter(env) as meter:
        env.reset(seed=5)
        for _ in range(20):
            env.step(torch.zeros(env.B, env.act_dim))
    assert meter.report()["obstacle_encounters"] is None


def test_every_contact_is_counted_once_and_charged_to_what_it_hit():
    """Props, walls and cars partition the onsets the gym layer publishes.

    Conservation is the test that catches the spy going wrong in either direction: a `prop_touched`
    read after `Simulator.step` zeroes it reports every crate as a wall, and a latch that is never
    cleared reports every wall as a crate. Both look plausible in a summary.
    """
    env, _ = _env()
    env.reset(seed=5)
    g = torch.Generator().manual_seed(11)
    total = 0
    with EncounterMeter(env) as meter:
        for _ in range(400):
            act = (torch.rand(env.B, env.act_dim, generator=g) * 2 - 1) * 0.6
            _, _, _, _, info = env.step(act)
            total += int((info["contact_onset"] & env.learner).sum())
    rep = meter.report()["obstacle_encounters"]
    assert rep["prop_onsets"] + rep["wall_onsets"] + rep["car_onsets"] == total
    # A run that hit only one kind of thing would pass conservation with the split broken.
    assert rep["prop_onsets"] > 0 and rep["wall_onsets"] > 0, rep


def test_an_approach_at_speed_always_has_a_record_to_charge():
    """Every prop contact made while actually driving belongs to an approach.

    This is the property the closing rule exists for. Closing a record when the piece stops being
    the nearest one *ahead* -- which is how this started -- shuts it one or two steps before the
    touch, because the candidate test needs the piece's centre ahead of the CoG and a piece about
    to be scraped is 0.3 m ahead at most. Measured then: 24 of 29 prop contacts belonged to no
    approach. The records now close on the piece being behind the car's *body*.

    Contacts made at a crawl are deliberately not covered: those are a car already wedged against a
    crate touching it again, which is the repetition this metric exists to stop counting.
    """
    env, rl = _env()
    teacher = common.make_teacher([rl], env, a_lat=7.0, a_acc=6.5, a_brake=4.0)
    env.reset(seed=5)
    fast_contacts = 0
    with EncounterMeter(env) as meter:
        for _ in range(600):
            _, _, _, _, info = env.step(env.teacher_label(teacher))
            onset = info["contact_onset"] & env.learner
            # `_prop_step` is this step's crate flag and `state[:, 3]` the speed it came off at.
            fast = onset & meter._prop_step & (env.sim.state[:, 3] > meter.min_speed_mps)
            fast_contacts += int(fast.sum())
    rep = meter.report()["obstacle_encounters"]
    assert rep["prop_onsets"] > 0, "nothing was hit, so the assertion below is vacuous"
    assert rep["prop_onsets"] - rep["prop_onsets_outside_encounter"] >= fast_contacts, rep
    assert rep["contacted"] <= rep["encounters"]


def test_a_long_scrape_is_one_encounter_and_the_repetition_is_still_visible():
    """The point of the unit: onsets can repeat within an approach, the approach counts once.

    The raceline teacher drives into crates and stays against them, so this run has the repetition
    in it. `onsets_per_contacted_encounter` is reported so the multiplier stays readable instead of
    being silently folded into the rate, as `collisions_per_km` folds it.
    """
    env, rl = _env()
    teacher = common.make_teacher([rl], env, a_lat=7.0, a_acc=6.5, a_brake=4.0)
    env.reset(seed=5)
    with EncounterMeter(env) as meter:
        for _ in range(600):
            env.step(env.teacher_label(teacher))
    rep = meter.report()["obstacle_encounters"]
    assert rep["contacted"] >= 1 and rep["encounters"] > rep["contacted"]
    assert rep["onsets_per_contacted_encounter"] >= 1.0
    lo, hi = rep["contact_rate_ci95"]
    assert 0.0 <= lo <= rep["contact_rate"] <= hi <= 1.0


def test_the_meter_puts_the_simulator_back_the_way_it_found_it():
    """It swaps two attributes on a live simulator. Leaving either one behind would hand the next
    measurement in the same process a spy that writes into a dead meter's counters."""
    env, _ = _env(envs=4)
    step, contact = env.sim.step, env.sim._prop_contact
    with EncounterMeter(env):
        env.reset(seed=5)
        env.step(torch.zeros(env.B, env.act_dim))
        assert env.sim.step is not step
    assert env.sim.step == step
    assert "_prop_contact" not in env.sim.__dict__
    assert env.sim._prop_contact == contact
