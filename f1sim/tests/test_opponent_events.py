"""Scripted opponent behaviour events (`f1sim.opponent_events`).

Five claims are worth testing and the rest is decoration:

1. **Off is off.** With `opp_events=()` a rollout is bit-identical to the same rollout on an env
   that has no event machinery at all -- the code path as it was before the feature existed. Every
   existing run, checkpoint comparison and benchmark number depends on that.
2. **On is on, and reproducible.** The events fire at about the configured rate, every configured
   kind appears, and the same seed produces the same schedule.
3. **The effects are the declared ones.** brake/stop slow the opponent (and only the opponent);
   shift/weave move it off the raceline by the declared amount and no further.
4. **They cannot hurt.** The lateral offset is clamped by the lane's own free space, and an event
   never raises an opponent's commanded speed above the `opp_follow_gap` cap.
5. **The flags reach the env.** Each `--opp-*` flag lands on the EnvConfig field of the same name
   and on nothing else.

The tracks are small and the LiDAR is cut to 36 beams (the convention in `test_sim_audit.py` and
friends): what is under test is the opponent's command, which does not read the scan at all.
"""
import dataclasses
import functools

import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig, F1VecEnv
from f1sim.learn import common, ppo
from f1sim.opponent_events import EVENT_ID, EVENT_NAMES, OpponentEvents, parse_events, raceline_offset_limit

TRACK = "gen:competition:0"          # a real lane with a centerline and a varying width
CONTROL_HZ = 40.0                    # Config().sim.control_rate, i.e. 0.025 s per step


@functools.lru_cache(maxsize=4)
def _track_and_raceline(name: str):
    from f1sim.raceline import Raceline
    track = maps.load(name)
    return track, Raceline.build_cached(track)


def _env(events=(), rate=0.0, envs=8, seed=7, track=TRACK, **cfg_kw):
    """A two-car race per track instance with teacher-driven opponents, small enough for CPU."""
    tr, rl = _track_and_raceline(track)
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(**{"race_size": 2, "opponent": "teacher", "max_steps": 4000, "hist_len": 0,
                        "opp_events": events, "opp_event_rate": rate, **cfg_kw})
    return common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl])


def _actions(steps: int, B: int, seed: int, act_dim: int = 2) -> torch.Tensor:
    """A fixed action tape, drawn from its own generator so no rollout can perturb another's."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(steps, B, act_dim, generator=g) * 2 - 1


def _rollout(env, tape, seed: int):
    """Drive the tape and snapshot everything a downstream consumer could read."""
    env.reset(seed=seed)
    frames = []
    for a in tape:
        obs, rew, term, trunc, info = env.step(a)
        frames.append((obs["scan"].clone(), obs["speed"].clone(), rew.clone(), term.clone(),
                       trunc.clone(), info["priv"].clone(), env.sim.state.clone(), env.last_cmd.clone()))
    return frames


# --------------------------------------------------------------------------- 1. off is off
def test_events_off_is_bit_identical_to_the_code_without_them():
    """200 steps with `opp_events=()` against 200 steps with the machinery absent entirely.

    `env.events = None` is the honest control: `_opponent_actions` then runs the exact instructions
    it ran before this feature, the teacher is called without an `offset` argument at all, and
    nothing is drawn from `sim.gen`. Bit-identical, not close -- a feature that is off must not cost
    an existing run so much as a rounding difference, because every checkpoint and benchmark number
    on the branch was measured on the other path.
    """
    steps, envs = 200, 8
    tape = _actions(steps, envs, seed=11)
    off = _env(envs=envs)
    assert off.events is not None and not off.events.enabled
    control = _env(envs=envs)
    control.events = None                                  # the feature flag, absent
    a = _rollout(off, tape, seed=5)
    b = _rollout(control, tape, seed=5)
    for i, (fa, fb) in enumerate(zip(a, b)):
        for name, ta, tb in zip(("scan", "speed", "reward", "terminated", "truncated", "priv",
                                 "state", "cmd"), fa, fb):
            assert torch.equal(ta, tb), (
                f"step {i}: {name} differs with events off -- max |delta| "
                f"{float((ta.float() - tb.float()).abs().max()):.3e}")


def test_events_off_draws_nothing_from_the_generator():
    """The other half of "off is off": the seeded stream itself must not move.

    Bit-identity over a rollout would still pass if the feature consumed randomness at a point
    nothing downstream happened to read. It would then desync any *future* draw, so an unflagged run
    would drift from the run it reproduces the moment anything else is added.
    """
    env = _env(envs=8)
    env.reset(seed=5)
    before = env.sim.gen.get_state().clone()
    env.events.step()
    env.events.reset(torch.arange(env.B))
    assert env.events.speed_scale() is None and env.events.lateral_offset() is None
    assert torch.equal(env.sim.gen.get_state(), before), "the disabled event machinery drew from sim.gen"


def test_info_reports_no_event_while_the_feature_is_off():
    env = _env(envs=8)
    env.reset(seed=5)
    _, _, _, _, info = env.step(torch.zeros(env.B, 2))
    ev = info["opp_event"]
    assert int(ev["id"].abs().sum()) == 0 and float(ev["time_left"].abs().sum()) == 0.0
    assert float(ev["offset"].abs().sum()) == 0.0


# --------------------------------------------------------------------------- 2. on, and reproducible
def _drive(env, steps: int, seed: int, speed: float = 0.2):
    """Drive a constant mild throttle and collect the per-step event ids and offsets.

    The learner's own driving is irrelevant here -- what is under test is what the *opponent* does --
    so a constant action keeps the rollout cheap and free of action-tape coupling.
    """
    env.reset(seed=seed)
    a = _throttle(env, speed)
    ids, offs, opp_speed = [], [], []
    for _ in range(steps):
        _, _, _, _, info = env.step(a)
        ids.append(info["opp_event"]["id"].clone())
        offs.append(info["opp_event"]["offset"].clone())
        opp_speed.append(env.sim.state[:, 3].clone())
    return torch.stack(ids), torch.stack(offs), torch.stack(opp_speed)


def _throttle(env, speed: float) -> torch.Tensor:
    """A constant mild throttle in whichever action space this env uses (direct, or a plan's speeds)."""
    a = torch.zeros(env.B, env.act_dim)
    a[:, 1 if env.act_dim == 2 else slice(-2, None)] = speed
    return a


def _starts(ids: torch.Tensor) -> torch.Tensor:
    """(T-1, B) mask of the steps where a car entered an event (id 0 -> id k)."""
    return (ids[1:] != ids[:-1]) & (ids[1:] != 0)


def test_every_configured_event_fires_at_about_the_configured_rate():
    steps, envs, rate = 500, 16, 2.0
    env = _env(events=("brake", "stop", "shift", "weave"), rate=rate, envs=envs, seed=13)
    ids, _, _ = _drive(env, steps, seed=13)
    start = _starts(ids)
    n_start = int(start.sum())
    per_kind = {n: int((start & (ids[1:] == EVENT_ID[n])).sum()) for n in EVENT_NAMES}
    assert all(v > 0 for v in per_kind.values()), f"an event kind never fired: {per_kind}"
    # Expected: rate/10 per opponent-second, over (opponents x seconds) of *idle* driving. Events do
    # not overlap, so the idle fraction is the correction; the bound is deliberately loose on both
    # sides -- this is a rate check, not a distribution test.
    opp_seconds = (steps / CONTROL_HZ) * (envs // 2)
    expected = rate / 10.0 * opp_seconds
    assert 0.4 * expected <= n_start <= 1.2 * expected, \
        f"{n_start} events over {opp_seconds:.0f} opponent-seconds, expected ~{expected:.0f}"


def test_the_same_seed_replays_the_same_schedule():
    kw = dict(events=("brake", "stop", "shift"), rate=3.0, envs=8, seed=21)
    a = _drive(_env(**kw), 200, seed=21)
    b = _drive(_env(**kw), 200, seed=21)
    assert torch.equal(a[0], b[0]), "the event schedule is not reproducible from the seed"
    assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    c = _drive(_env(**{**kw, "seed": 22}), 200, seed=22)
    assert not torch.equal(a[0], c[0]), "a different seed replayed the same schedule"


def test_events_never_touch_the_learner():
    env = _env(events=("brake", "stop", "shift", "weave"), rate=4.0, envs=8, seed=31)
    ids, offs, _ = _drive(env, 200, seed=31)
    learner = env.learner                                  # slot 0 of each race
    assert int(ids[:, learner].abs().sum()) == 0, "an event was scripted onto a learner-driven car"
    assert float(offs[:, learner].abs().sum()) == 0.0
    assert int(ids[:, ~learner].abs().sum()) > 0, "no opponent was scripted at all: the test proves nothing"


def test_mixed_mode_scripts_the_teacher_races_and_only_those():
    """With `--opponent mixed` the roles are redrawn at every race reset, so the gate has to be too.

    A stale gate would script a car the policy is driving -- the PPO buffers would then contain
    transitions whose action was not the policy's, which is the one thing the mixed-mode design is
    built to avoid.
    """
    env = _env(events=("brake", "stop", "shift"), rate=6.0, envs=16, seed=37,
               opponent="mixed", mixed_teacher_frac=0.5)
    env.reset(seed=37)
    a = torch.zeros(env.B, 2); a[:, 1] = 0.3
    scripted_policy_cars, scripted_teacher_cars, saw_selfplay = 0, 0, False
    for _ in range(300):
        _, _, _, _, info = env.step(a)
        on_policy, ids = info["on_policy"], info["opp_event"]["id"]
        scripted_policy_cars += int((ids != 0)[on_policy].sum())
        scripted_teacher_cars += int((ids != 0)[~on_policy].sum())
        saw_selfplay |= bool((on_policy & (env.slot > 0)).any())
    assert scripted_policy_cars == 0, f"{scripted_policy_cars} steps scripted a policy-driven car"
    assert scripted_teacher_cars > 100, f"only {scripted_teacher_cars} scripted opponent steps"
    assert saw_selfplay, "no self-play race was drawn: the mixed-mode half of the test proves nothing"


# --------------------------------------------------------------------------- 3. the declared effects
def test_stop_parks_the_opponent_and_brake_only_slows_it():
    """A stopped car reaches ~0 m/s; a braking one drops but keeps rolling."""
    stop = _env(events=("stop",), rate=6.0, envs=8, seed=41)
    ids_s, _, v_s = _drive(stop, 300, seed=41)
    opp = ~stop.learner
    parked = v_s[:, opp][ids_s[:, opp] == EVENT_ID["stop"]]
    assert parked.numel() > 50, f"only {parked.numel()} stopped-car steps to measure"
    assert float(parked.min()) < 0.05, f"no stopped opponent ever reached rest (min {float(parked.min()):.3f} m/s)"

    brake = _env(events=("brake",), rate=6.0, envs=8, seed=41, opp_brake_scale_range=(0.4, 0.5))
    ids_b, _, v_b = _drive(brake, 300, seed=41)
    opp_b = ~brake.learner
    free = v_b[:, opp_b][ids_b[:, opp_b] == 0]
    braking = v_b[:, opp_b][ids_b[:, opp_b] == EVENT_ID["brake"]]
    assert braking.numel() > 50, f"only {braking.numel()} braking steps to measure"
    assert float(braking.mean()) < float(free.mean()), \
        f"braking opponents were not slower ({float(braking.mean()):.2f} vs {float(free.mean()):.2f} m/s)"


def test_shift_moves_the_opponent_off_the_raceline_by_the_declared_amount():
    """The commanded offset ramps up, holds and ramps back, and never exceeds the configured range."""
    hi = 0.35
    env = _env(events=("shift",), rate=6.0, envs=8, seed=51,
               opp_shift_offset_range=(0.25, hi), opp_shift_hold_range=(1.0, 1.5))
    ids, offs, _ = _drive(env, 300, seed=51)
    opp = ~env.learner
    active = ids[:, opp] == EVENT_ID["shift"]
    assert int(active.sum()) > 50, f"only {int(active.sum())} shift steps to measure"
    o = offs[:, opp][active]
    assert float(o.abs().max()) <= hi + 1e-6, f"offset {float(o.abs().max()):.3f} m past the {hi} m range"
    assert float(o.abs().max()) > 0.15, "the shift never reached a useful offset"
    assert float(o.min()) < 0 < float(o.max()), "the offset sign is not being drawn: every shift went the same way"
    assert float(o.abs().min()) < 0.05, "no ramp: a shift jumped straight to its offset"


def test_the_shifted_opponent_actually_leaves_the_raceline():
    env = _env(events=("shift",), rate=6.0, envs=8, seed=53,
               opp_shift_offset_range=(0.30, 0.35), opp_shift_hold_range=(1.5, 2.0))
    env.reset(seed=53)
    a = torch.zeros(env.B, 2); a[:, 1] = 0.2
    opp = ~env.learner
    held, on_line = [], []
    for _ in range(300):
        _, _, _, _, info = env.step(a)
        _, dist = env.teacher.project(env.sim.state[:, :2], env.sim.tid)
        cmd_off = info["opp_event"]["offset"]
        held.append(dist[opp & (cmd_off.abs() > 0.25)])
        on_line.append(dist[opp & (info["opp_event"]["id"] == 0)])
    held = torch.cat(held); on_line = torch.cat(on_line)
    assert held.numel() > 30, f"only {held.numel()} fully-shifted steps to measure"
    assert float(held.mean()) > float(on_line.mean()) + 0.10, (
        f"a shifted opponent sat {float(held.mean()):.3f} m off the raceline against "
        f"{float(on_line.mean()):.3f} m on it: the offset is commanded but not driven")


def test_weave_oscillates_within_its_amplitude():
    amp = 0.25
    env = _env(events=("weave",), rate=6.0, envs=8, seed=61,
               opp_weave_amp_range=(0.2, amp), opp_weave_period_range=(2.0, 2.5))
    ids, offs, _ = _drive(env, 300, seed=61)
    opp = ~env.learner
    o = offs[:, opp][ids[:, opp] == EVENT_ID["weave"]]
    assert o.numel() > 50, f"only {o.numel()} weave steps to measure"
    assert float(o.abs().max()) <= amp + 1e-6, f"weave reached {float(o.abs().max()):.3f} m, past its {amp} m amplitude"
    assert float(o.min()) < -0.1 and float(o.max()) > 0.1, "the weave did not cross the line in both directions"


# --------------------------------------------------------------------------- 4. they cannot hurt
def test_the_offset_budget_leaves_the_car_clear_of_the_wall():
    """`raceline_offset_limit` is the free space at each raceline point minus the body and a margin."""
    env = _env(events=("shift",), rate=1.0, envs=4, seed=71)
    lim = env.teacher.offset_limit
    assert lim is not None and lim.shape == env.teacher.xy.shape[:2]
    tid = torch.zeros(lim.shape, dtype=torch.long)
    clearance = env.sim.track.sample_edt(env.teacher.xy, tid)
    half, margin = 0.5 * env.cfg.vehicle.width, env.ecfg.opp_event_margin
    assert float((lim - (clearance - half - margin).clamp_min(0.0)).abs().max()) < 1e-5
    assert float(lim.min()) >= 0.0
    # A car sitting at the limit still has `margin` of body-to-wall gap left, everywhere on the lane.
    assert float((clearance - lim - half).min()) >= margin - 1e-5


def test_a_narrow_section_clamps_the_commanded_offset():
    """A shift bigger than the lane is cut to what fits, not driven into the wall."""
    env = _env(events=("shift",), rate=6.0, envs=8, seed=73,
               opp_shift_offset_range=(3.0, 3.0), opp_shift_hold_range=(2.0, 2.0))
    env.reset(seed=73)
    a = torch.zeros(env.B, 2); a[:, 1] = 0.2
    opp = ~env.learner
    worst = 0.0
    for _ in range(200):
        env.step(a)
        idx, _ = env.teacher.project(env.sim.state[:, :2], env.sim.tid)
        want = env.events.lateral_offset()
        got = env.teacher.clamp_offset(want, env.sim.tid, idx)
        lim = env.teacher.offset_limit[env.sim.tid, idx]
        assert float((got.abs() - lim).max()) <= 1e-5, "a commanded offset was outside the lane's budget"
        worst = max(worst, float((want.abs() - got.abs())[opp].max()))
    assert worst > 0.5, "the clamp never bit: a 3 m shift should be cut hard on every section of this lane"


def test_an_event_never_lifts_an_opponent_over_its_follow_gap_cap():
    """An opponent already braking for the car ahead may not accelerate because an event said so.

    The event multiplier is in [0, 1] and is applied *before* the follow cap, so the command is the
    minimum of the two. This drives until cars are genuinely in each other's way and then checks the
    command the env actually produced against the cap it computed from the same state.
    """
    env = _env(events=("brake", "stop", "shift"), rate=6.0, envs=16, seed=81, spawn_gap=(1.6, 2.2))
    env.reset(seed=81)
    a = torch.zeros(env.B, 2); a[:, 1] = 0.9
    checked = 0
    for _ in range(200):
        follow, v_cap = env.follow_cap(env.sim.state)
        an = env._opponent_actions(a)                      # steps the machine, exactly as step() does
        v_cmd = (an[:, 1] + 1) * 0.5 * env.ecfg.v_max_policy
        gated = follow & ~env.on_policy
        if bool(gated.any()):
            assert float((v_cmd - v_cap)[gated].max()) <= 1e-4, \
                "a following opponent was commanded above its follow-gap cap"
            checked += int(gated.sum())
        env.step(a)
    assert checked > 20, f"only {checked} following-opponent steps seen: the test proves nothing"


def test_speed_scale_is_never_above_one():
    env = _env(events=("brake", "stop", "shift", "weave"), rate=8.0, envs=16, seed=83)
    env.reset(seed=83)
    a = torch.zeros(env.B, 2); a[:, 1] = 0.5
    for _ in range(150):
        env.step(a)
        s = env.events.speed_scale()
        assert float(s.max()) <= 1.0 and float(s.min()) >= 0.0, f"speed scale out of [0, 1]: {float(s.min())}..{float(s.max())}"


def test_events_do_not_add_opponent_wall_contacts():
    """The lane-change events are the ones that could put an opponent in a wall. They do not.

    Measured against the same seed with the events off, so track, spawn and the learner's tape are
    the same run: what changes is only whether the opponents move off the line.
    """
    steps, envs = 400, 16

    def opp_wall_hits(**kw):
        env = _env(envs=envs, seed=91, **kw)
        env.reset(seed=91)
        a = torch.zeros(env.B, 2); a[:, 1] = 0.3
        opp = ~env.learner
        hits = 0
        for _ in range(steps):
            _, _, _, _, info = env.step(a)
            car = info.get("car_collision")
            wall = env.last_result.collision.bool()
            if car is not None:
                wall = wall & ~car.bool()                  # a car-car contact is not a wall contact
            hits += int(wall[opp].sum())
        return hits

    base = opp_wall_hits()
    with_events = opp_wall_hits(events=("shift", "weave"), rate=4.0)
    assert with_events <= base + 2, \
        f"opponents hit the wall {with_events} times with lane-change events against {base} without"


# --------------------------------------------------------------------------- the plan action space
def _plan_env(events=(), rate=0.0, envs=4, seed=101):
    tr, rl = _track_and_raceline(TRACK)
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(race_size=2, opponent="teacher", max_steps=4000, hist_len=0,
                     action_mode="plan", compile_tracker=False,
                     opp_events=events, opp_event_rate=rate)
    return common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl])


def test_the_plan_action_space_carries_the_events_too():
    """`--action-mode plan` drives opponents through `plan_action`, a different function.

    Both halves matter: with the events off the plan path must be bit-identical (it is the path a
    plan-space run already trains on), and with them on the plan speeds and the planned line have to
    move, or the flags would silently do nothing for half the runs on this branch.
    """
    steps = 60
    off, control = _plan_env(), _plan_env()
    control.events = None
    assert off.act_dim > 2, "the plan action space collapsed to the direct one; this test is vacuous"
    tape = _actions(steps, 4, seed=103, act_dim=off.act_dim)
    for i, (fa, fb) in enumerate(zip(_rollout(off, tape, seed=103), _rollout(control, tape, seed=103))):
        for name, ta, tb in zip(("scan", "speed", "reward", "term", "trunc", "priv", "state", "cmd"), fa, fb):
            assert torch.equal(ta, tb), f"plan mode, step {i}: {name} differs with events off"

    on = _plan_env(events=("stop", "shift"), rate=15.0, seed=103)
    ids, offs, speed = _drive(on, 250, seed=103)
    opp = ~on.learner
    assert int((ids[:, opp] != 0).sum()) > 50, "no event fired in the plan action space"
    assert float(offs[:, opp].abs().max()) > 0.05, "no shift reached the plan"
    parked = speed[:, opp][ids[:, opp] == EVENT_ID["stop"]]
    free = speed[:, opp][ids[:, opp] == 0]
    assert parked.numel() > 0, "no stop event ran in the plan action space"
    # Not "reaches 0 m/s": in the plan action space the opponent's speed is a *reference* walked
    # down by the tracker, which delivers about 3.4 m/s^2 against its nominal bound (the same soft
    # speed weight `opp_follow_decel` is calibrated against), so a 1 s stop from 4 m/s ends before
    # the car does. What has to be true is that a stop is a hard, sustained deceleration.
    assert float(parked.min()) < 0.35 * float(free.mean()), (
        f"a stopped opponent only reached {float(parked.min()):.2f} m/s against a free-running "
        f"{float(free.mean()):.2f} m/s: the plan speeds are not being scaled")


# --------------------------------------------------------------------------- 5. the flags
def test_parse_events_normalizes_and_refuses_unknown_names():
    assert parse_events("brake,stop") == ("brake", "stop")
    assert parse_events(" shift , shift ") == ("shift",)          # deduplicated, whitespace ignored
    assert parse_events(()) == () and parse_events(None) == () and parse_events("") == ()
    with pytest.raises(ValueError) as exc:
        parse_events("brake,swerve")
    assert "swerve" in str(exc.value) and "brake" in str(exc.value), str(exc.value)


class _Captured(Exception):
    def __init__(self, env_cfg):
        self.env_cfg = env_cfg


def _env_cfg_for(monkeypatch, extra_argv):
    """Run `ppo.main` far enough to build its EnvConfig, then stop (see test_ppo_spawn_gap.py)."""
    argv = ["ppo", "--device", "cpu", "--envs", "2", "--race-size", "2", "--opponent", "teacher",
            "--tracks", "dummy", "--wandb", "disabled"] + list(extra_argv)
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(ppo.common, "track_names", lambda spec: ["dummy"])
    monkeypatch.setattr(ppo.common, "load_tracks", lambda names, **kw: ([object()], None))

    def capture(tracks, num_envs, device, env_cfg=None, **kw):
        raise _Captured(env_cfg)

    monkeypatch.setattr(ppo.common, "make_env", capture)
    with pytest.raises(_Captured) as caught:
        ppo.main()
    return caught.value.env_cfg


def test_an_unflagged_run_still_has_the_events_off(monkeypatch):
    assert EnvConfig().opp_events == () and EnvConfig().opp_event_rate == 0.0
    cfg = _env_cfg_for(monkeypatch, [])
    assert cfg.opp_events == () and cfg.opp_event_rate == 0.0


@pytest.mark.parametrize("flag,value,field,expected", [
    (["--opp-events", "brake,stop"], None, "opp_events", ("brake", "stop")),
    (["--opp-event-rate", "1.5"], None, "opp_event_rate", 1.5),
    (["--opp-brake-scale", "0.1", "0.3"], None, "opp_brake_scale_range", (0.1, 0.3)),
    (["--opp-brake-time", "0.2", "1.0"], None, "opp_brake_time_range", (0.2, 1.0)),
    (["--opp-stop-time", "2.0", "5.0"], None, "opp_stop_time_range", (2.0, 5.0)),
    (["--opp-shift-offset", "0.1", "0.2"], None, "opp_shift_offset_range", (0.1, 0.2)),
    (["--opp-shift-hold", "0.3", "0.9"], None, "opp_shift_hold_range", (0.3, 0.9)),
    (["--opp-shift-ramp", "0.6"], None, "opp_shift_ramp", 0.6),
    (["--opp-weave-amp", "0.05", "0.2"], None, "opp_weave_amp_range", (0.05, 0.2)),
    (["--opp-weave-period", "1.0", "3.0"], None, "opp_weave_period_range", (1.0, 3.0)),
    (["--opp-weave-time", "1.0", "2.0"], None, "opp_weave_time_range", (1.0, 2.0)),
    (["--opp-event-margin", "0.2"], None, "opp_event_margin", 0.2),
])
def test_each_flag_lands_on_its_field_and_nothing_else(monkeypatch, flag, value, field, expected):
    """The flag is more than documentation only if it reaches EnvConfig -- and moves nothing else."""
    base = ["--opp-events", "brake", "--opp-event-rate", "1.0"]
    control = _env_cfg_for(monkeypatch, base)
    treatment = _env_cfg_for(monkeypatch, base + flag)
    assert getattr(treatment, field) == expected
    differ = {f.name for f in dataclasses.fields(EnvConfig)
              if getattr(control, f.name) != getattr(treatment, f.name)}
    assert differ <= {field}, f"{flag[0]} moved more than {field}: {sorted(differ)}"


@pytest.mark.parametrize("argv,needle", [
    (["--opp-events", "swerve", "--opp-event-rate", "1.0"], "swerve"),
    (["--opp-events", "brake"], "--opp-event-rate"),
    (["--opp-events", "brake", "--opp-event-rate", "1.0", "--race-size", "1"], "--race-size"),
    (["--opp-events", "brake", "--opp-event-rate", "1.0", "--opponent", "policy"], "--opponent"),
])
def test_a_configuration_that_would_silently_do_nothing_is_refused(monkeypatch, argv, needle):
    with pytest.raises(SystemExit) as exc:
        _env_cfg_for(monkeypatch, argv)
    assert needle in str(exc.value), f"refused without saying why: {exc.value}"


# --------------------------------------------------------------------------- the machine in isolation
def test_the_state_machine_runs_one_event_at_a_time_for_its_declared_duration():
    """No env, no simulator: the timing contract of the machine itself."""
    ecfg = EnvConfig(opp_events=("stop",), opp_event_rate=40.0,   # ~1 start per 10 steps
                     opp_stop_time_range=(1.0, 1.0))
    gen = torch.Generator().manual_seed(3)
    ev = OpponentEvents(4, "cpu", ecfg, 1.0 / CONTROL_HZ, gen)
    ev.set_gate(torch.tensor([True, True, False, False]))
    trace = []
    for _ in range(400):
        ev.step()
        trace.append((ev.kind.clone(), ev.t.clone()))
    kinds = torch.stack([k for k, _ in trace])
    assert int(kinds[:, 2:].abs().sum()) == 0, "the machine scripted a car outside its gate"
    lengths, run = [], 0
    for i in (0, 1):                                       # `t == 0` marks a start, so back-to-back
        for kind, t in trace:                              # events are two runs and not one long one
            if int(kind[i]):
                if run and float(t[i]) == 0.0:
                    lengths.append(run); run = 0
                run += 1
            elif run:
                lengths.append(run); run = 0
        if run:
            lengths.append(run); run = 0                   # a run still going at the end is not a length
            lengths.pop()
    assert lengths, "no event ever ended"
    # 1.0 s at 40 Hz: the step that crosses the duration ends it, so the run is 40 or 41 steps.
    assert set(lengths) <= {40, 41}, f"stop events ran for {sorted(set(lengths))} steps, expected 40-41"
    assert len(lengths) > 10, f"only {len(lengths)} completed events in 400 steps at rate 40"


def test_event_ids_are_stable():
    """A logged id keeps its meaning: the mapping is part of the interface, not an implementation detail."""
    assert EVENT_NAMES == ("brake", "stop", "shift", "weave")
    assert EVENT_ID == {"brake": 1, "stop": 2, "shift": 3, "weave": 4}
