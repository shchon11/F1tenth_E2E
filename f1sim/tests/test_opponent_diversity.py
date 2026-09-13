"""Opponents that behave like opponents: the population, the reactive behaviours, the grid.

The events work before this one was accepted on "it fires". The bar here is "the learner is in it",
so the claims worth testing are the ones a census number would otherwise have to be taken on trust:

1. **Off is off.** Everything new, off, leaves the env the env it was -- same rollout, and nothing
   drawn from the simulator's generator. Every existing checkpoint and benchmark number was measured
   on that path. (`test_opponent_events.py` holds the same property for the timed events.)
2. **The population is a population, drawn per race and from the seed.** Every configured entry
   turns up; all the opponent slots of one race share it; slot 0 is never anything but the policy; a
   car respawning mid-race rejoins the same opponent; the same seed replays the same draw.
3. **Each reactive behaviour reacts to where the learner is.** Driven by a synthetic `LearnerView`
   rather than by hoping a rollout produces the geometry: a learner behind and to the left makes a
   defending car move left, a learner alongside makes a yielding car move away, a corner makes a
   `line` car sweep across it, and `oblivious` stops the follow cap binding.
4. **The grid is the grid asked for, and nobody spawns in contact.** Two cars put side by side where
   the lane has room for one is the failure that terminates every episode on step 1.
5. **The census is reproducible and reports what happened.** Same seed, same seconds; and a
   configuration built to contain a situation produces seconds in it.
6. **The flags reach the EnvConfig, and a silently-no-op combination is refused** rather than run.

Small tracks and a 36-beam LiDAR, the convention in `test_sim_audit.py` and friends: what is under
test is where the cars are and what they are commanded, which does not read the scan.
"""
import dataclasses
import functools
import math

import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import (EnvConfig, F1VecEnv, OPP_DRIVER_POLICY, OPP_DRIVER_POOL,
                           OPP_DRIVER_TEACHER, POOL_SELF, POOL_TEACHER, SPAWN_ORDER_ID,
                           pool_entries)
from f1sim.learn import common, opponent_config as oc, ppo
from f1sim.opponent_events import (EVENT_ID, LearnerView, REACTIVE_BIT, REACTIVE_NAMES,
                                   raceline_corners, split_events)

TRACK = "gen:competition:0"          # a real lane with a centerline and a varying width
WIDE = "real:blackbox2022_1"         # 1.6 m half-width: two cars fit abreast anywhere on it


@functools.lru_cache(maxsize=4)
def _track_and_raceline(name: str):
    from f1sim.raceline import Raceline
    track = maps.load(name)
    return track, Raceline.build_cached(track)


def _env(envs=8, seed=7, track=TRACK, **cfg_kw):
    """A race per track instance with teacher-driven opponents, small enough for CPU."""
    tr, rl = _track_and_raceline(track)
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(**{"race_size": 2, "opponent": "teacher", "max_steps": 4000, "hist_len": 0,
                        **cfg_kw})
    return common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl])


def _throttle(env, speed: float) -> torch.Tensor:
    a = torch.zeros(env.B, env.act_dim)
    a[:, 1 if env.act_dim == 2 else slice(-2, None)] = speed
    return a


def _actions(steps: int, B: int, seed: int, act_dim: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(steps, B, act_dim, generator=g) * 2 - 1


def _rollout(env, tape, seed: int):
    env.reset(seed=seed)
    frames = []
    for a in tape:
        obs, rew, term, trunc, info = env.step(a)
        frames.append((obs["scan"].clone(), obs["speed"].clone(), rew.clone(), term.clone(),
                       trunc.clone(), info["priv"].clone(), env.sim.state.clone(),
                       env.last_cmd.clone(), env.speed_cap.clone()))
    return frames


# --------------------------------------------------------------------------- 1. off is off
@pytest.mark.parametrize("opponent", ["policy", "teacher", "mixed"])
def test_everything_new_off_leaves_the_rollout_untouched(opponent):
    """The grid order, the reactive layer and the pool all default to off; with them off the
    rollout has to be the one the env produced before any of them existed.

    `env.events = None` is the honest control for the behaviour layer -- `_opponent_actions` then
    runs the instructions it ran before, calling the teacher with no `offset` at all. The default
    `spawn_order` "behind" is checked by construction: it draws nothing and returns zeros, so the
    generator stream is the same one, which the next test pins directly.
    """
    steps, envs = 120, 8
    tape = _actions(steps, envs, seed=11)
    off = _env(envs=envs, opponent=opponent)
    control = _env(envs=envs, opponent=opponent)
    control.events = None
    assert off.ecfg.spawn_order == "behind" and off.ecfg.opp_pool == () and off.pool is None
    for i, (fa, fb) in enumerate(zip(_rollout(off, tape, seed=5), _rollout(control, tape, seed=5))):
        for name, ta, tb in zip(("scan", "speed", "reward", "terminated", "truncated", "priv",
                                 "state", "cmd", "speed_cap"), fa, fb):
            assert torch.equal(ta, tb), (
                f"step {i}: {name} differs with everything off -- max |delta| "
                f"{float((ta.float() - tb.float()).abs().max()):.3e}")


def test_the_default_grid_draws_nothing_from_the_generator():
    """The other half of "off is off": the seeded stream itself must not move.

    A feature that consumes randomness at a point nothing downstream reads still desyncs every
    *future* draw, so an unflagged run would drift from the run it reproduces as soon as anything
    else is added.
    """
    env = _env(envs=8)
    env.reset(seed=5)
    ids = torch.arange(env.B)
    before = env.sim.gen.get_state().clone()
    order = env._spawn_order(ids, env.race, torch.ones(env.B, dtype=torch.bool), env.B // env.M,
                             env.sim.gen)
    assert int(order.abs().sum()) == 0, "the default grid is not 'behind'"
    assert torch.equal(env.sim.gen.get_state(), before), "the default grid drew from sim.gen"


def test_two_cars_are_staggered_by_the_gap_they_are_told_to_be():
    """The spawn arc is now cumulative rather than `rank x gap`. For two cars those are the same
    number, which is what keeps every trained race unchanged; this pins it."""
    env = _env(envs=16, spawn_gap=(3.0, 3.0))
    env.reset(seed=13)
    gap = env.signed_gaps(env.sim.s, env.sim.tid)[env.learner, 0]
    # Not exact: the commanded arc is the centerline arc, and a car spawned off the line at a yaw
    # projects back onto it a few centimetres from where it was asked for.
    assert torch.allclose(gap, torch.full_like(gap, 3.0), atol=0.12), gap.tolist()


def test_three_cars_are_staggered_monotonically():
    """`rank x gap` with a per-car gap scrambles a grid of three: rank 1 drawing 6 m and rank 2
    drawing 2.5 m puts the third car a metre *ahead* of the second, and close draws put them in
    contact. Measured before the fix: 14 of 528 three-car spawns in contact."""
    env = _env(envs=18, race_size=3, spawn_gap=(2.5, 6.0))
    env.reset(seed=17)
    s = env.sim.s.view(-1, 3)
    L = env.sim.track.length[env.sim.tid].view(-1, 3)[:, 0]
    # slot 0 is the learner and starts last, so along the lane: slot 2 ahead of slot 1 ahead of 0
    ahead_1 = (s[:, 1] - s[:, 0] + L / 2) % L - L / 2
    ahead_2 = (s[:, 2] - s[:, 1] + L / 2) % L - L / 2
    assert float(ahead_1.min()) >= 2.4 and float(ahead_2.min()) >= 2.4, (
        f"grid not monotone: {ahead_1.tolist()} {ahead_2.tolist()}")


# --------------------------------------------------------------------------- 2. the population
def _tiny_checkpoint(path, env, seed: int):
    """A checkpoint that fits `env`'s observation, with its own weights. Not a good driver; a
    *different* one, which is all a population entry has to be."""
    from f1sim.learn.model import ActorCritic, save_checkpoint
    torch.manual_seed(seed)
    spec = common.obs_spec(env)
    model = ActorCritic(n_stack=env.ecfg.scan_stack, n_beams=env.n_beams,
                        proprio_dim=spec.proprio_dim, priv_dim=env.privileged(env.last_result).shape[1]
                        if env.last_result is not None else 21, act_dim=env.act_dim)
    save_checkpoint(str(path), model, {"spec": dataclasses.asdict(spec)})
    return str(path)


def _pool_env(tmp_path, entries, envs=8, seed=7, n_ck=2, **cfg_kw):
    """An env in pool mode whose checkpoint entries are freshly built to fit it."""
    probe = _env(envs=envs, seed=seed, **cfg_kw)
    probe.reset(seed=seed)
    paths = [_tiny_checkpoint(tmp_path / f"pool{i}.pt", probe, 100 + i) for i in range(n_ck)]
    names = [p if p == POOL_SELF or p == POOL_TEACHER else paths[int(p)] for p in entries]
    return _env(envs=envs, seed=seed, opponent="pool", opp_pool=tuple(names), **cfg_kw), names


def test_the_pool_draw_is_per_race_and_reproducible(tmp_path):
    env, names = _pool_env(tmp_path, [POOL_TEACHER, POOL_SELF, "0", "1"], envs=16)
    env.reset(seed=23)
    d = env.opp_driver.view(-1, env.M)
    assert torch.equal(d[:, 0], torch.zeros_like(d[:, 0])), "slot 0 is not driven by the policy"
    first = env.opp_driver.clone()
    # every configured entry is reachable, and a race's opponents all share the one drawn for it
    seen = set()
    for k in range(12):
        env.reset(seed=23 + k)
        seen |= set(env.opp_driver.view(-1, env.M)[:, 1:].flatten().tolist())
    assert seen >= {OPP_DRIVER_POLICY, OPP_DRIVER_TEACHER, OPP_DRIVER_POOL, OPP_DRIVER_POOL + 1}, seen
    again, _ = _pool_env(tmp_path, [POOL_TEACHER, POOL_SELF, "0", "1"], envs=16)
    again.reset(seed=23)
    assert torch.equal(again.opp_driver, first), "the same seed drew a different population"
    other, _ = _pool_env(tmp_path, [POOL_TEACHER, POOL_SELF, "0", "1"], envs=16)
    other.reset(seed=24)
    assert not torch.equal(other.opp_driver, first), "a different seed drew the same population"


def test_three_car_races_give_every_opponent_of_a_race_the_same_entry(tmp_path):
    env, _ = _pool_env(tmp_path, [POOL_TEACHER, POOL_SELF, "0"], envs=18, race_size=3)
    env.reset(seed=29)
    d = env.opp_driver.view(-1, 3)
    assert torch.equal(d[:, 1], d[:, 2]), f"one race drew two different opponents: {d.tolist()}"


def test_a_car_respawning_mid_race_keeps_its_race_s_opponent(tmp_path):
    """"Per race" has to mean the race, not the reset: a car that crashes rejoins the race it was
    in, against the car it was racing."""
    env, _ = _pool_env(tmp_path, [POOL_TEACHER, "0", "1"], envs=16)
    env.reset(seed=31)
    before = env.opp_driver.clone()
    env._reset_envs(torch.tensor([0]))                   # slot 0 of race 0 alone: a crashed learner
    assert torch.equal(env.opp_driver, before), "a partial reset re-drew the race's opponent"
    env._reset_envs(torch.tensor([0, 1]))                # the whole race
    assert env.opp_driver[2:].equal(before[2:]), "a full reset of one race touched another"


def test_a_pool_driven_car_is_driven_by_its_checkpoint(tmp_path):
    """The commands of the pool-driven cars have to come from the pool, not from the caller's
    action and not from the teacher."""
    env, _ = _pool_env(tmp_path, ["0"], envs=8, n_ck=1)
    env.reset(seed=37)
    assert env.pool is not None and len(env.pool) == 1
    a = _throttle(env, 0.0)
    an = env._opponent_actions(a)
    pooled = env.pool_driven
    assert bool(pooled.any()), "no car was pool-driven"
    assert not torch.allclose(an[pooled], a[pooled]), "a pool car was driven by the caller's action"
    assert torch.equal(an[~pooled], a[~pooled]), "the pool overwrote a policy-driven car"


def test_a_pool_entry_that_does_not_fit_the_observation_is_refused(tmp_path):
    from f1sim.learn.model import ActorCritic, save_checkpoint
    from f1sim.learn.opponent_pool import OpponentPool
    env = _env(envs=8, scan_stack=3)
    env.reset(seed=41)
    torch.manual_seed(3)
    wrong = ActorCritic(n_stack=9, n_beams=env.n_beams, proprio_dim=99, priv_dim=21, act_dim=2)
    path = str(tmp_path / "wrong.pt")
    save_checkpoint(path, wrong, {})
    with pytest.raises(ValueError) as exc:
        OpponentPool.load([path], env)
    assert "n_stack" in str(exc.value) and "opp-pool" in str(exc.value), str(exc.value)


def test_a_pool_without_self_keeps_the_training_buffers_narrow(tmp_path):
    """A race whose opponent can never be the policy has nothing to learn from in that slot, and a
    buffer that is half masked out of every update is half a buffer."""
    with_self, _ = _pool_env(tmp_path, [POOL_SELF, "0"], envs=8)
    without, _ = _pool_env(tmp_path, [POOL_TEACHER, "0"], envs=8)
    assert int(with_self.learner.sum()) == with_self.B
    assert int(without.learner.sum()) == without.B // without.M


def test_pool_mode_refuses_an_empty_population():
    with pytest.raises(ValueError) as exc:
        _env(envs=8, opponent="pool")
    assert "opp_pool" in str(exc.value)


# --------------------------------------------------------------------------- 3. the reactive layer
def _view(env, gap, lat, lon=None, closing=None, learner=True):
    """A synthetic `LearnerView` for one opponent per row: the geometry under test, stated."""
    B = env.B
    f = lambda v: torch.full((B, 1), float(v))
    return LearnerView(gap=f(gap), lon=f(gap if lon is None else lon), lat=f(lat),
                       closing=f(0.0 if closing is None else closing),
                       learner=torch.full((B, 1), bool(learner)),
                       rl_idx=None, tid=env.sim.tid)


def _reactive_env(name, prob, **kw):
    env = _env(envs=4, opp_events=(name,), **{f"opp_{name}_prob": prob}, **kw)
    env.reset(seed=53)
    env.events.set_gate(torch.ones(env.B, dtype=torch.bool))     # act on every row, learner or not
    return env


def _settle(env, view, steps=80):
    """Run the reactive layer to its slew-limited steady state and return the offset it holds."""
    for _ in range(steps):
        env.events.step_reactive(view)
    return env.events.lateral_offset()


@pytest.mark.parametrize("side,sign", [(0.30, +1), (-0.30, -1)])
def test_defend_moves_toward_the_side_the_learner_is_coming_down(side, sign):
    env = _reactive_env("defend", 1.0, opp_defend_offset_range=(0.30, 0.30))
    off = _settle(env, _view(env, gap=-2.0, lat=side))
    assert sign * float(off.mean()) > 0.25, (
        f"a learner {side:+.2f} m to the side, 2 m behind, produced offset {float(off.mean()):+.3f}")


def test_defend_ignores_a_learner_that_is_not_behind_it():
    env = _reactive_env("defend", 1.0, opp_defend_offset_range=(0.30, 0.30))
    far = _settle(env, _view(env, gap=-40.0, lat=0.30))          # outside overtake_range
    assert abs(float(far.mean())) < 0.01, f"defended against a car 40 m back: {float(far.mean()):+.3f}"
    ahead = _settle(_reactive_env("defend", 1.0), _view(env, gap=+2.0, lat=0.30))
    assert abs(float(ahead.mean())) < 0.01, "defended against a car it is already behind"


def test_defend_is_strongest_when_the_learner_is_closest():
    env = _reactive_env("defend", 1.0, opp_defend_offset_range=(0.30, 0.30))
    close = abs(float(_settle(env, _view(env, gap=-2.0, lat=0.30)).mean()))
    env2 = _reactive_env("defend", 1.0, opp_defend_offset_range=(0.30, 0.30))
    far = abs(float(_settle(env2, _view(env2, gap=-9.0, lat=0.30)).mean()))
    assert close > far + 0.05, f"block did not ramp with distance: {close:.3f} at 2 m, {far:.3f} at 9 m"


@pytest.mark.parametrize("side,sign", [(0.35, -1), (-0.35, +1)])
def test_yield_moves_away_from_a_car_alongside(side, sign):
    env = _reactive_env("yield", 1.0, opp_yield_offset_range=(0.30, 0.30))
    off = _settle(env, _view(env, gap=0.0, lon=0.0, lat=side))
    assert sign * float(off.mean()) > 0.25, (
        f"a learner {side:+.2f} m alongside produced offset {float(off.mean()):+.3f} (should move away)")


def test_yield_ignores_a_car_that_is_not_alongside():
    env = _reactive_env("yield", 1.0, opp_yield_offset_range=(0.30, 0.30))
    off = _settle(env, _view(env, gap=-3.0, lon=-3.0, lat=0.25))
    assert abs(float(off.mean())) < 0.01, f"yielded to a car 3 m behind: {float(off.mean()):+.3f}"


def test_the_reactive_offset_is_rate_limited():
    """A target that jumps sideways asks pure pursuit for a step steer input, so the offset ramps."""
    env = _reactive_env("defend", 1.0, opp_defend_offset_range=(0.35, 0.35), opp_react_slew=0.6)
    view = _view(env, gap=-1.0, lat=0.35)
    env.events.step_reactive(view)
    one = float(env.events.lateral_offset().abs().max())
    assert one <= 0.6 * env.sim.control_dt + 1e-6, f"the offset moved {one:.3f} m in one step"
    assert float(_settle(env, view).abs().mean()) > 0.30, "it never got there"


def test_the_reactive_offset_is_capped():
    env = _reactive_env("defend", 1.0, opp_defend_offset_range=(3.0, 3.0), opp_react_max=0.45)
    off = _settle(env, _view(env, gap=-1.0, lat=0.35), steps=200)
    assert float(off.abs().max()) <= 0.45 + 1e-6, f"offset {float(off.abs().max()):.3f} past the cap"


def test_only_the_gated_cars_react():
    env = _env(envs=8, opp_events=("defend",), opp_defend_prob=1.0)
    env.reset(seed=59)
    ids, offs = [], []
    a = _throttle(env, 0.4)
    for _ in range(120):
        _, _, _, _, info = env.step(a)
        offs.append(info["opp_event"]["offset"].clone())
        ids.append(info["opp_event"]["disposition"].clone())
    offs, ids = torch.stack(offs), torch.stack(ids)
    assert float(offs[:, env.learner].abs().max()) == 0.0, "a learner was given a reactive offset"
    assert int(ids[:, env.learner].abs().sum()) == 0, "a learner was given a disposition"
    assert float(offs[:, ~env.learner].abs().max()) > 0.05, "no opponent ever reacted"


def test_the_corner_tables_find_the_corners_and_the_line_sweeps_across_one():
    tr, rl = _track_and_raceline(TRACK)
    env = _env(envs=4, opp_events=("line",), opp_line_prob=1.0, opp_line_offset_range=(0.30, 0.30))
    c = env.events.corners
    assert c is not None, "the corner tables were not built"
    assert int(c["id"].max()) >= 3, f"only {int(c['id'].max()) + 1} corners found on {TRACK}"
    assert set(torch.unique(c["sign"][c["id"] >= 0]).tolist()) == {-1.0, 1.0}, "corners all turn one way"
    # phase sweeps 0 -> 1 inside each corner, and is 0 outside
    assert float(c["phase"][c["id"] < 0].abs().max()) == 0.0
    for cid in range(int(c["id"].max()) + 1):
        ph = c["phase"][0][c["id"][0] == cid]
        if ph.numel() > 4:
            assert float(ph.min()) < 0.3 and float(ph.max()) > 0.7, f"corner {cid} phase {ph.min()}..{ph.max()}"
    env.reset(seed=61)
    a = _throttle(env, 0.35)
    offs = []
    for _ in range(300):
        _, _, _, _, info = env.step(a)
        offs.append(info["opp_event"]["offset"].clone())
    o = torch.stack(offs)[:, ~env.learner]
    assert float(o.min()) < -0.1 and float(o.max()) > 0.1, (
        f"the corner line never swept both ways: {float(o.min()):.3f}..{float(o.max()):.3f}")


def test_the_line_mode_is_drawn_per_corner():
    """Out-in and in-out are the two draws; over a lap a car has to do both, or the "drawn per
    corner" in the flag's help is a lie and the opponent just has a fixed line."""
    env = _env(envs=8, opp_events=("line",), opp_line_prob=1.0, opp_line_offset_range=(0.30, 0.30))
    env.reset(seed=67)
    a = _throttle(env, 0.45)
    modes, corners = [], []
    for _ in range(400):
        env.step(a)
        modes.append(env.events.corner_mode.clone()); corners.append(env.events.corner.clone())
    modes, corners = torch.stack(modes), torch.stack(corners)
    opp = ~env.learner
    entered = (corners[1:] != corners[:-1]) & (corners[1:] >= 0)
    assert int(entered[:, opp].sum()) > 8, f"only {int(entered[:, opp].sum())} corner entries seen"
    drawn = modes[1:][:, opp][entered[:, opp]]
    assert float(drawn.min()) < 0 < float(drawn.max()), "every corner drew the same line"


def test_oblivious_does_not_brake_for_the_car_in_front():
    """The follow-gap cap is what makes a teacher opponent polite. `oblivious` removes it, and the
    check is against the same rollout with the behaviour off, so the geometry is the same geometry.

    The learner leads and crawls, which is the situation that makes the cap bind at all.
    """
    def over_cap(prob):
        env = _env(envs=32, seed=71, spawn_order="ahead", spawn_gap=(1.0, 1.6),
                   opp_events=("oblivious",) if prob else (), opp_oblivious_prob=prob)
        env.reset(seed=71)
        a = _throttle(env, -0.7)                       # the leading learner crawls
        following, over = 0, 0
        for _ in range(200):
            follow, v_cap = env.follow_cap(env.sim.state)
            an = env._opponent_actions(a)
            v_cmd = (an[:, 1] + 1) * 0.5 * env.ecfg.v_max_policy
            gated = follow & env.teacher_driven
            following += int(gated.sum())
            over += int((gated & (v_cmd > v_cap + 0.05)).sum())
            env.step(a)
        return following, over

    n_off, over_off = over_cap(0.0)
    n_on, over_on = over_cap(1.0)
    assert n_off > 100 and n_on > 100, f"too few following steps to measure: {n_off}, {n_on}"
    assert over_off == 0, f"{over_off} of {n_off} polite opponents were commanded above their cap"
    assert over_on > 0.5 * n_on, (
        f"only {over_on} of {n_on} oblivious opponents ignored the cap: the behaviour is not doing "
        f"anything")


def test_a_reactive_behaviour_at_probability_zero_is_off():
    env = _env(envs=8, opp_events=("defend", "yield"), opp_defend_prob=0.0, opp_yield_prob=0.0)
    assert env.events.react == () and not env.events.react_on
    assert env.events.lateral_offset() is None and env.events.speed_scale() is None


def test_the_reactive_offset_is_clamped_by_the_lane():
    """Same budget the scripted offsets use: free space minus the body minus the margin."""
    env = _env(envs=8, opp_events=("defend",), opp_defend_prob=1.0,
               opp_defend_offset_range=(0.45, 0.45), opp_react_max=0.45, opp_react_slew=10.0)
    env.reset(seed=73)
    assert env.teacher.offset_limit is not None
    a = _throttle(env, 0.3)
    worst = 0.0
    for _ in range(200):
        env.step(a)
        idx, _ = env.teacher.project(env.sim.state[:, :2], env.sim.tid)
        want = env.events.lateral_offset()
        got = env.teacher.clamp_offset(want, env.sim.tid, idx)
        lim = env.teacher.offset_limit[env.sim.tid, idx]
        assert float((got.abs() - lim).max()) <= 1e-5, "a reactive offset was outside the lane's budget"
        worst = max(worst, float((want.abs() - got.abs()).max()))
    assert worst > 0.0, "the clamp never bit: this track has a section narrower than 0.45 m of offset"


# --------------------------------------------------------------------------- 4. the grid
@pytest.mark.parametrize("order,expect", [("behind", +1), ("ahead", -1)])
def test_the_grid_puts_the_learner_where_it_is_told(order, expect):
    env = _env(envs=16, spawn_order=order, spawn_gap=(3.0, 5.0))
    env.reset(seed=79)
    gap = env.signed_gaps(env.sim.s, env.sim.tid)[env.learner, 0]
    assert expect * float(gap.mean()) > 2.5, f"{order}: learner gap to the opponent {float(gap.mean()):+.2f} m"
    assert int(env.spawn_order_race[env.learner].unique().item()) == SPAWN_ORDER_ID[order]


def test_an_alongside_grid_is_actually_alongside_and_not_in_contact():
    env = _env(envs=32, track=WIDE, spawn_order="alongside")
    abreast = contacts = cars = 0
    for k in range(6):
        env.reset(seed=83 + k)
        lon = env.learner_view().lon[:, 0]
        abreast += int((lon[env.learner].abs() < 0.9).sum()); cars += int(env.learner.sum())
        _, _, _, _, info = env.step(torch.zeros(env.B, env.act_dim))
        contacts += int(info["car_collision"].sum())
    assert abreast == cars, f"only {abreast} of {cars} learners started alongside"
    assert contacts == 0, f"{contacts} cars spawned in contact"


def test_a_lane_with_room_for_one_car_staggers_instead_of_spawning_a_contact():
    """The whole point of consulting the distance field. A narrow track keeps some races staggered
    and the rest properly abreast; what it must never do is put two cars in one place."""
    env = _env(envs=32, track=TRACK, spawn_order="alongside", spawn_alongside_sep=1.5)
    contacts = 0
    for k in range(6):
        env.reset(seed=89 + k)
        _, _, _, _, info = env.step(torch.zeros(env.B, env.act_dim))
        contacts += int(info["car_collision"].sum())
    assert contacts == 0, f"{contacts} cars spawned in contact with a 1.5 m separation asked for"
    # The fallback is a *stagger*, so the test is on the arc the grid was laid out along, not on the
    # body-frame offset: a lane that doubles back puts a car 3 m away along the lane a few
    # centimetres ahead of you in your own frame, and that is the track's shape, not the grid's.
    arc = env.signed_gaps(env.sim.s, env.sim.tid)[env.learner, 0].abs()
    assert float(arc.min()) > 2.0, (
        f"a 1.5 m separation fitted a 1.6 m lane -- closest pair {float(arc.min()):.2f} m apart "
        f"along the lane; check the fallback")


@pytest.mark.parametrize("M", [2, 3])
def test_a_random_grid_draws_every_order_and_never_spawns_a_contact(M):
    env = _env(envs=12 * M, track=WIDE, race_size=M, spawn_order="random")
    seen, contacts = set(), 0
    for k in range(8):
        env.reset(seed=97 + k)
        seen |= set(env.spawn_order_race.unique().tolist())
        _, _, _, _, info = env.step(torch.zeros(env.B, env.act_dim))
        contacts += int(info["car_collision"].sum())
    assert seen == set(SPAWN_ORDER_ID.values()), f"grids drawn: {seen}"
    assert contacts == 0, f"{contacts} cars spawned in contact over 8 x {env.B} spawns"


def test_an_opponent_faster_than_its_profile_is_allowed_and_arrives():
    """`--opp-speed` above 1.0 is the only way the learner is ever the car being overtaken."""
    env = _env(envs=16, spawn_order="ahead", spawn_gap=(2.0, 3.0), opp_speed_range=(1.2, 1.4))
    env.reset(seed=101)
    a = _throttle(env, -0.6)                             # the leading learner crawls
    caught = torch.zeros(env.B, dtype=torch.bool)
    for _ in range(200):
        env.step(a)
        caught |= env.signed_gaps(env.sim.s, env.sim.tid)[:, 0] > 0
    n = int(env.learner.sum())
    assert int(caught[env.learner].sum()) >= 0.6 * n, (
        f"only {int(caught[env.learner].sum())} of {n} learners were passed by a 1.2-1.4x opponent")


# --------------------------------------------------------------------------- 5. the census
def test_the_census_is_reproducible_and_counts_the_situations_it_is_given(tmp_path):
    from f1sim.learn import opponent_census as oce
    argv = ["--teacher", "--tracks", TRACK, "--races", "4", "--steps", "60", "--device", "cpu",
            "--race-size", "2", "--opponent", "teacher", "--spawn-order", "behind",
            "--opp-events", "defend", "--opp-defend-prob", "1.0", "--seed", "4401"]
    a = oce.main(argv)
    b = oce.main(argv)
    for key in oce.SITUATION_KEYS:
        assert a["situations"][key]["seconds"] == b["situations"][key]["seconds"], (
            f"{key} is not reproducible from the seed")
    assert a["learner_seconds"] > 0
    s = a["situations"]
    assert s["contention"]["seconds"] > 0, "the learner never met the opponent at all"
    assert s["defended"]["seconds"] > 0, "the census reports no defending in a race built to defend"
    assert s["being_overtaken"]["seconds"] == 0.0, "a slower car overtook the learner"
    # the lateral-signal half is analysis of the same rollout, and must be populated
    ls = a["lateral_signal"]
    assert ls["car_proximity_reward_per_step"]["n"] > 0
    assert 0.0 <= ls["car_proximity_reward_per_step"]["zero_fraction"] <= 1.0


def test_the_census_situations_are_the_seven_the_contract_names():
    from f1sim.learn import opponent_census as oce
    named = ("behind_slower", "alongside", "being_overtaken", "defended", "yielded_to",
             "oblivious_behind", "two_in_range")
    assert oce.SITUATION_KEYS[:len(named)] == named


# --------------------------------------------------------------------------- 6. the flags
class _Captured(Exception):
    def __init__(self, env_cfg):
        self.env_cfg = env_cfg


def _env_cfg_for(monkeypatch, extra_argv):
    """Run `ppo.main` far enough to build its EnvConfig, then stop (see test_ppo_spawn_gap.py)."""
    argv = ["ppo", "--device", "cpu", "--envs", "2", "--race-size", "2", "--opponent", "teacher",
            "--tracks", "dummy", "--wandb", "disabled"] + list(extra_argv)
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(ppo.common, "track_names", lambda spec, **kw: ["dummy"])
    monkeypatch.setattr(ppo.common, "load_tracks", lambda names, **kw: ([object()], None))

    def capture(tracks, num_envs, device, env_cfg=None, **kw):
        raise _Captured(env_cfg)

    monkeypatch.setattr(ppo.common, "make_env", capture)
    with pytest.raises(_Captured) as caught:
        ppo.main()
    return caught.value.env_cfg


@pytest.mark.parametrize("flag,field,expected", [
    (["--spawn-order", "random"], "spawn_order", "random"),
    (["--spawn-alongside-sep", "0.2"], "spawn_alongside_sep", 0.2),
    (["--spawn-alongside-gap", "0.1", "0.3"], "spawn_alongside_gap", (0.1, 0.3)),
    (["--opp-speed", "1.0", "1.3"], "opp_speed_range", (1.0, 1.3)),
    (["--opp-events", "defend", "--opp-defend-prob", "0.4"], "opp_defend_prob", 0.4),
    (["--opp-events", "yield", "--opp-yield-prob", "0.3"], "opp_yield_prob", 0.3),
    (["--opp-events", "line", "--opp-line-prob", "0.6"], "opp_line_prob", 0.6),
    (["--opp-events", "oblivious", "--opp-oblivious-prob", "0.2"], "opp_oblivious_prob", 0.2),
    (["--opp-defend-offset", "0.1", "0.2"], "opp_defend_offset_range", (0.1, 0.2)),
    (["--opp-yield-offset", "0.1", "0.2"], "opp_yield_offset_range", (0.1, 0.2)),
    (["--opp-line-offset", "0.1", "0.2"], "opp_line_offset_range", (0.1, 0.2)),
    (["--opp-defend-range", "8.0"], "opp_defend_range", 8.0),
    (["--opp-defend-full", "2.0"], "opp_defend_full", 2.0),
    (["--opp-alongside-lon", "0.7"], "opp_alongside_lon", 0.7),
    (["--opp-alongside-lat", "1.0"], "opp_alongside_lat", 1.0),
    (["--opp-react-max", "0.3"], "opp_react_max", 0.3),
    (["--opp-react-slew", "0.9"], "opp_react_slew", 0.9),
    (["--opp-corner-kappa", "0.2"], "opp_corner_kappa", 0.2),
    (["--opp-corner-min-arc", "1.5"], "opp_corner_min_arc", 1.5),
    (["--opp-corner-smooth", "0.4"], "opp_corner_smooth", 0.4),
])
def test_each_flag_lands_on_its_env_config_field(monkeypatch, flag, field, expected):
    cfg = _env_cfg_for(monkeypatch, flag)
    assert getattr(cfg, field) == expected
    base = EnvConfig()
    changed = [f.name for f in dataclasses.fields(EnvConfig)
               if getattr(cfg, f.name) != getattr(base, f.name)]
    # the harness itself sets these; nothing else may move
    allowed = {field, "speed_cap", "reward_collision", "reward_steer_rate", "reward_proximity",
               "safe_dist", "reward_wrong_way", "reward_collision_speed", "proximity_speed_ref",
               "reward_plan_clearance", "plan_margin", "reward_lap", "reward_lap_time",
               "reward_overtake", "reward_car_contact", "reward_car_proximity", "reward_sideslip",
               "car_safe_gap", "max_steps", "scan_stack", "scan_stride", "hist_len", "race_size",
               "opponent", "compile_tracker", "opp_events", "opp_event_rate"}
    assert set(changed) <= allowed, f"{flag} also changed {sorted(set(changed) - allowed)}"


def test_an_unflagged_run_has_every_new_feature_off(monkeypatch):
    base = EnvConfig()
    assert base.opp_pool == () and base.spawn_order == "behind"
    assert all(getattr(base, f"opp_{n}_prob") == 0.0 for n in REACTIVE_NAMES)
    cfg = _env_cfg_for(monkeypatch, [])
    assert cfg.opp_pool == () and cfg.spawn_order == "behind"
    assert all(getattr(cfg, f"opp_{n}_prob") == 0.0 for n in REACTIVE_NAMES)


#: Every one of these is a command line that *looks* like it configures an opponent and would
#: silently run the unflagged configuration instead. `RACE` is prepended so that each case fails on
#: the check it is about rather than on "there is no other car".
RACE = ["--race-size", "2", "--opponent", "teacher"]


@pytest.mark.parametrize("argv,needle", [
    (RACE + ["--opp-events", "defend"], "--opp-defend-prob"),
    (RACE + ["--opp-events", "defend", "--opp-defend-prob", "0"], "--opp-defend-prob"),
    (RACE + ["--opp-events", "brake"], "--opp-event-rate"),
    (["--race-size", "2", "--opponent", "pool"], "--opp-pool"),
    (RACE + ["--opp-pool", "a.pt"], "without --opponent pool"),
    (["--race-size", "2", "--opponent", "pool", "--opp-pool", "/nope/a.pt"], "no such checkpoint"),
    (RACE + ["--opp-speed", "1.0", "0.5"], "--opp-speed"),
    (["--race-size", "2", "--opponent", "policy", "--opp-events", "defend",
      "--opp-defend-prob", "0.5"], "teacher-driven"),
    (["--race-size", "1", "--opponent", "pool", "--opp-pool", "self"], "--race-size 1"),
])
def test_a_configuration_that_would_silently_do_nothing_is_refused(argv, needle):
    ap = oc.add_arguments(__import__("argparse").ArgumentParser())
    a = ap.parse_args(argv)
    with pytest.raises(SystemExit) as exc:
        oc.validate(a)
    assert needle in str(exc.value), str(exc.value)


def test_a_pool_with_a_teacher_entry_accepts_the_behaviour_flags(tmp_path):
    ck = tmp_path / "x.pt"; ck.write_bytes(b"not a real checkpoint, only the path is checked here")
    ap = oc.add_arguments(__import__("argparse").ArgumentParser())
    a = ap.parse_args(["--race-size", "2", "--opponent", "pool", "--opp-pool", f"teacher,{ck}",
                       "--opp-events", "defend", "--opp-defend-prob", "0.5"])
    oc.validate(a)                              # must not raise: the pool holds a teacher
    assert oc.needs_racelines(a) and a.opp_pool == ("teacher", str(ck))
    kw = oc.env_kwargs(a)
    assert kw["opp_pool"] == ("teacher", str(ck)) and kw["opp_defend_prob"] == 0.5


def test_the_spawn_order_flag_needs_a_race():
    ap = oc.add_arguments(__import__("argparse").ArgumentParser())
    a = ap.parse_args(["--spawn-order", "alongside"])
    with pytest.raises(SystemExit) as exc:
        oc.validate(a)
    assert "--spawn-order" in str(exc.value)
