"""Per-opponent configuration: the spec, the env, the flag, the console widget.

The user asked for the other cars to be configurable one at a time -- driver kind, checkpoint, speed
profile, grip label, events, reactive probabilities, spawn. The claims that need a test are the ones
a demo cannot show:

1. **Off is off.** A table absent leaves the env the env it was: same rollout, tensor for tensor,
   and the same draws from the simulator's generator.
2. **The table is the grid.** Each slot's kind drives its car, each slot's band scales its car, each
   slot's label reaches the teacher's own per-car tensors, and each slot's `spawn` puts it where it
   says -- with the all-`ahead` table reproducing the grid `--spawn-order behind` builds.
3. **It refuses what it cannot build.** A race size that does not match the table, a kind this tree
   does not have, a checkpoint the loader will not open, events on a car that has no teacher.
4. **One spelling.** The JSON round trips through the flag, through `SessionConfig` and through the
   console widget, and what the widget emits is what the trainer parses.

Small tracks and a 36-beam LiDAR, the convention of `test_opponent_diversity.py`: what is under test
is where the cars are and what they are commanded, which does not read the scan.
"""
import argparse
import dataclasses
import functools
import json

import pytest
import torch

from f1sim import Config, maps
from f1sim import opponent_slots as osl
from f1sim.gym_env import (EnvConfig, OPP_DRIVER_POLICY, OPP_DRIVER_POOL, OPP_DRIVER_TEACHER)
from f1sim.learn import common, opponent_config as oc
from f1sim.opponent_events import EVENT_ID, REACTIVE_BIT
from f1sim.teacher import LABEL_GRIP_CODE

TRACK = "gen:competition:0"
WIDE = "real:blackbox2022_1"          # 1.6 m half-width: two cars fit abreast anywhere on it


@functools.lru_cache(maxsize=4)
def _track_and_raceline(name: str):
    from f1sim.raceline import Raceline
    track = maps.load(name)
    return track, Raceline.build_cached(track)


def _env(envs=8, seed=7, track=TRACK, **cfg_kw):
    tr, rl = _track_and_raceline(track)
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(**{"race_size": 2, "opponent": "teacher", "max_steps": 4000, "hist_len": 0,
                        **cfg_kw})
    return common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl])


def _slot_env(slots, envs=12, seed=7, track=TRACK, **cfg_kw):
    return _env(envs=envs, seed=seed, track=track, race_size=1 + len(slots), opponent="slots",
                opponent_slots=slots, **cfg_kw)


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


# =========================================================== 1. off is off
@pytest.mark.parametrize("opponent", ["teacher", "policy"])
def test_no_table_leaves_the_rollout_untouched(opponent):
    """The whole feature, off, is the env before it existed -- tensor for tensor.

    The control is an env built the same way whose `slots` attribute is forced to None, which is
    what every branch this change added tests. Any instruction that moved outside such a branch
    shows up here as a difference.
    """
    steps, envs = 60, 8
    tape = _actions(steps, envs, seed=11)
    off = _env(envs=envs, opponent=opponent)
    control = _env(envs=envs, opponent=opponent)
    assert off.ecfg.opponent_slots is None and off.slots is None
    for i, (fa, fb) in enumerate(zip(_rollout(off, tape, seed=5), _rollout(control, tape, seed=5))):
        for name, ta, tb in zip(("scan", "speed", "reward", "terminated", "truncated", "priv",
                                 "state", "cmd", "speed_cap"), fa, fb):
            assert torch.equal(ta, tb), f"step {i}: {name} differs with no slot table"


def test_no_table_draws_nothing_extra_from_the_generator():
    """The other half of "off is off": the seeded stream must not move.

    A feature that consumes randomness where nothing reads it still desyncs every *future* draw, so
    an unflagged run would drift from the run it reproduces as soon as anything else is added.
    """
    env = _env(envs=8)
    env.reset(seed=5)
    before = env.sim.gen.get_state().clone()
    env._reset_envs(torch.arange(env.B))
    after_a = env.sim.gen.get_state().clone()
    again = _env(envs=8)
    again.reset(seed=5)
    again._reset_envs(torch.arange(again.B))
    assert torch.equal(after_a, again.sim.gen.get_state())
    assert not torch.equal(before, after_a), "a full reset drew nothing at all: the test is inert"


def test_the_default_teacher_path_is_untouched_by_the_per_car_tensors():
    """`RacelineTeacher.grip_bin` keeps its scalar path: no slot table, no per-car codes."""
    env = _env(envs=8)
    assert env.teacher.label_grip_codes is None
    assert not torch.is_tensor(env.teacher.speed_scale)


# =========================================================== 2. the table is the grid
def test_each_slot_kind_drives_its_own_car():
    env = _slot_env([{"kind": "raceline"}, {"kind": "self"}], envs=12)
    env.reset(seed=5)
    drv = env.opp_driver.view(-1, 3)[0].tolist()
    assert drv == [OPP_DRIVER_POLICY, OPP_DRIVER_TEACHER, OPP_DRIVER_POLICY]
    assert env.teacher_driven.view(-1, 3)[0].tolist() == [False, True, False]
    # A `self` slot is the learner's own weights, so its rows belong in the PPO buffers; the
    # teacher's do not.
    assert env.learner.view(-1, 3)[0].tolist() == [True, False, True]


def test_a_table_of_teachers_only_keeps_the_buffers_narrow():
    env = _slot_env([{"kind": "raceline"}, {"kind": "raceline"}], envs=12)
    assert int(env.learner.sum()) == env.B // env.M


def test_the_speed_scale_of_each_slot_reaches_the_teacher_tensor():
    """Per-car, and from that car's own band -- the scale is on the teacher, not on the command.

    The band is degenerate here on purpose: a range would be a draw, and what is being pinned is
    which car gets which number.
    """
    env = _slot_env([{"kind": "raceline", "speed_scale": 0.5},
                     {"kind": "raceline", "speed_scale": 1.2}], envs=12)
    env.reset(seed=5)
    scale = env.teacher.speed_scale.view(-1, 3)
    assert torch.allclose(scale[:, 1], torch.full_like(scale[:, 1], 0.5))
    assert torch.allclose(scale[:, 2], torch.full_like(scale[:, 2], 1.2))
    # and exactly once: `opp_scale` must not apply it a second time on the way out of the env
    assert torch.allclose(env.opp_scale[env.teacher_driven],
                          torch.ones_like(env.opp_scale[env.teacher_driven]))


def test_a_ranged_speed_scale_is_redrawn_per_reset_inside_its_own_band():
    env = _slot_env([{"kind": "raceline", "speed_scale": [0.5, 0.6]},
                     {"kind": "raceline", "speed_scale": [1.0, 1.1]}], envs=12)
    seen = [set(), set()]
    for k in range(6):
        env.reset(seed=100 + k)
        sc = env.teacher.speed_scale.view(-1, 3)
        for j in (0, 1):
            col = sc[:, j + 1]
            lo, hi = (0.5, 0.6) if j == 0 else (1.0, 1.1)
            assert float(col.min()) >= lo - 1e-6 and float(col.max()) <= hi + 1e-6
            seen[j] |= {round(float(x), 4) for x in col}
    assert len(seen[0]) > 1 and len(seen[1]) > 1, "a range did not vary"


def test_the_grip_label_of_each_slot_reaches_the_teacher_and_moves_its_profile():
    """`label_grip` decides which speed profile that car plans on, so two labels are two speeds."""
    env = _slot_env([{"kind": "raceline", "label_grip": "nominal"},
                     {"kind": "raceline", "label_grip": "conservative"}], envs=12)
    env.reset(seed=5)
    codes = env.teacher.label_grip_codes.view(-1, 3)[0].tolist()
    assert codes[1] == LABEL_GRIP_CODE["nominal"] and codes[2] == LABEL_GRIP_CODE["conservative"]
    gb = env.teacher.grip_bin(env.sim.P, env.B, env.device).view(-1, 3)
    top = len(env.teacher.grip_levels) - 1
    assert torch.equal(gb[:, 1], torch.full_like(gb[:, 1], top))
    assert torch.equal(gb[:, 2], torch.zeros_like(gb[:, 2]))


def test_a_per_slot_speed_cap_binds_that_car_only():
    env = _slot_env([{"kind": "raceline", "speed_cap": 3.0}, {"kind": "raceline"}],
                    envs=12, speed_cap=8.0)
    env.reset(seed=5)
    cap = env.speed_cap.view(-1, 3)
    assert torch.allclose(cap[:, 1], torch.full_like(cap[:, 1], 3.0))
    assert torch.allclose(cap[:, 2], torch.full_like(cap[:, 2], 8.0))
    assert torch.allclose(cap[:, 0], torch.full_like(cap[:, 0], 8.0))
    a = torch.zeros(env.B, env.act_dim)
    for _ in range(80):
        env.step(a)
    v = env.sim.state[:, 3].view(-1, 3)
    assert float(v[:, 1].max()) <= 3.3, f"the capped car ran at {float(v[:, 1].max()):.2f} m/s"


@pytest.mark.parametrize("spawn,sign", [("ahead", +1), ("behind", -1)])
def test_a_slot_spawns_where_it_says_relative_to_the_learner(spawn, sign):
    env = _slot_env([{"kind": "raceline", "spawn": spawn}], envs=16, spawn_gap=(3.0, 3.0))
    env.reset(seed=13)
    gap = env.signed_gaps(env.sim.s, env.sim.tid).view(-1, 2, 1)[:, 0, 0]
    assert torch.all(sign * gap > 2.0), gap.tolist()


def test_an_all_ahead_table_builds_the_grid_spawn_order_behind_builds():
    """The default table has to be the default grid: same ranks, same order along the lane.

    Not the same arc numbers -- a per-slot band draws differently from one per race -- but the
    *ordering*, which is what the grid is: the learner last, the highest slot in front.
    """
    flagged = _env(envs=18, race_size=3, spawn_gap=(3.0, 3.0))
    flagged.reset(seed=17)
    tabled = _slot_env([{"kind": "raceline"}, {"kind": "raceline"}], envs=18, spawn_gap=(3.0, 3.0))
    tabled.reset(seed=17)
    for env in (flagged, tabled):
        s = env.sim.s.view(-1, 3)
        L = env.sim.track.length[env.sim.tid].view(-1, 3)[:, 0]
        ahead_1 = (s[:, 1] - s[:, 0] + L / 2) % L - L / 2
        ahead_2 = (s[:, 2] - s[:, 1] + L / 2) % L - L / 2
        assert float(ahead_1.min()) >= 2.4 and float(ahead_2.min()) >= 2.4, (
            f"{'flag' if env is flagged else 'table'} grid not monotone: "
            f"{ahead_1.tolist()} {ahead_2.tolist()}")


def test_an_alongside_slot_starts_level_with_the_learner_and_not_in_contact():
    env = _slot_env([{"kind": "raceline", "spawn": "alongside"}], envs=16, track=WIDE)
    env.reset(seed=19)
    st = env.sim.state.view(-1, 2, 8)
    d = (st[:, 1, :2] - st[:, 0, :2]).norm(dim=1)
    assert float(d.max()) < 1.2, f"not alongside: {d.tolist()}"
    assert float(d.min()) > env.cfg.vehicle.width, f"spawned in contact: {d.tolist()}"


def test_a_mixed_grid_puts_one_car_ahead_and_one_alongside():
    env = _slot_env([{"kind": "raceline", "spawn": "ahead"},
                     {"kind": "raceline", "spawn": "alongside"}], envs=18, track=WIDE,
                    spawn_gap=(3.0, 3.0))
    env.reset(seed=23)
    gaps = env.signed_gaps(env.sim.s, env.sim.tid).view(-1, 3, 2)[:, 0]       # learner -> the two
    assert torch.all(gaps[:, 0] > 2.0), f"slot 1 was not ahead: {gaps[:, 0].tolist()}"
    assert torch.all(gaps[:, 1].abs() < 1.0), f"slot 2 was not alongside: {gaps[:, 1].tolist()}"
    st = env.sim.state.view(-1, 3, 8)
    d = (st[:, 2, :2] - st[:, 0, :2]).norm(dim=1)
    assert float(d.min()) > env.cfg.vehicle.width, f"alongside slot spawned in contact: {d.tolist()}"


def test_a_random_spawn_slot_draws_every_placement_and_keeps_it_for_the_race():
    env = _slot_env([{"kind": "raceline", "spawn": "random"}], envs=32, track=WIDE)
    seen = set()
    for k in range(10):
        env.reset(seed=200 + k)
        seen |= set(env.slot_spawn_code.view(-1, 2)[:, 1].tolist())
    assert seen == {0, 1, 2}, f"a random spawn drew only {sorted(seen)}"
    before = env.slot_spawn_code.clone()
    env._reset_envs(torch.tensor([0]))                  # the learner alone: a crashed car rejoins
    assert torch.equal(env.slot_spawn_code, before), "a partial reset re-drew the race's grid"


def test_events_are_per_slot():
    """One car brakes and stops, the other never does anything -- from one `OpponentEvents`."""
    env = _slot_env([{"kind": "raceline", "events": ["brake", "stop"], "event_rate": 8.0},
                     {"kind": "raceline"}], envs=24)
    env.reset(seed=29)
    a = torch.zeros(env.B, env.act_dim)
    fired = torch.zeros(env.B, dtype=torch.bool)
    for _ in range(200):
        env.step(a)
        fired |= env.events.kind != 0
    f = fired.view(-1, 3)
    assert bool(f[:, 1].any()), "the slot with events never had one"
    assert not bool(f[:, 2].any()), "a slot with no events was given one"
    assert not bool(f[:, 0].any()), "the learner was given an event"
    kinds = {int(k) for k in env.events.kind.tolist()} | {0}
    assert kinds <= {0, EVENT_ID["brake"], EVENT_ID["stop"]}, kinds


def test_reactive_probabilities_are_per_slot():
    env = _slot_env([{"kind": "raceline", "reactive": {"defend": 1.0}},
                     {"kind": "raceline", "reactive": {"oblivious": 1.0}}], envs=24)
    env.reset(seed=31)
    disp = env.events.disp.view(-1, 3)
    assert torch.all((disp[:, 1] & REACTIVE_BIT["defend"]) != 0)
    assert torch.all((disp[:, 1] & REACTIVE_BIT["oblivious"]) == 0)
    assert torch.all((disp[:, 2] & REACTIVE_BIT["oblivious"]) != 0)
    assert torch.all((disp[:, 2] & REACTIVE_BIT["defend"]) == 0)
    assert torch.all(disp[:, 0] == 0), "the learner was given a disposition"


class _StubTeacher:
    """A teacher kind that is not the raceline teacher: it commands a constant, so which rows it
    drove is visible in the action itself.

    It takes the raceline teacher as its reference the way `InteractiveTeacher` does, which is the
    contract the registry's `teacher_factory` states.
    """
    LAST = None

    def __init__(self, base, env=None):
        self.base, self.env = base, env
        _StubTeacher.LAST = self

    def __call__(self, state, P=None, tid=None, offset=None):
        out = torch.zeros(state.shape[0], 2)
        out[:, 1] = 3.5                                # a speed nothing else would command
        return out

    def plan_action(self, state, P=None, tid=None, v_max=8.0, spec=None, offset=None):
        an = self.base.plan_action(state, P, tid, v_max, spec, offset=offset)
        return torch.full_like(an, 0.25)


def test_a_teacher_kind_that_is_not_the_raceline_teacher_drives_its_own_rows(monkeypatch):
    """The registry's whole point: a teacher kind the tree *has* must be driven by its own driver.

    Without this, `interactive` turning available the day worker 17's branch merges would silently be
    the raceline teacher wearing another name -- the failure the `available` machinery exists to
    prevent, arriving by the back door. Exercised here with a stub kind, because the real one is not
    on this branch.
    """
    kind = osl.DriverKind("stubteacher", "stub", "a test kind", teacher=True,
                          teacher_factory=f"{__name__}:_StubTeacher")
    monkeypatch.setitem(osl.KIND_BY_NAME, kind.name, kind)
    env = _slot_env([{"kind": "stubteacher"}, {"kind": "raceline"}], envs=12)
    env.reset(seed=61)
    assert env.alt_teacher_kinds == ("stubteacher",)
    assert isinstance(env.alt_teachers[0], _StubTeacher)
    assert env.alt_teachers[0].base is env.teacher, "the stub was not built from the raceline teacher"
    assert env.alt_teacher_mask["stubteacher"].view(-1, 3)[0].tolist() == [False, True, False]
    an = env._opponent_actions(torch.zeros(env.B, env.act_dim))
    a = an.view(-1, 3, env.act_dim)
    # the stub commands 3.5 m/s, which is `2 v / v_max - 1` once normalized and is a speed the
    # raceline teacher's own profile never lands on exactly
    want = 3.5 / env.ecfg.v_max_policy * 2 - 1
    assert torch.allclose(a[:, 1, 1], torch.full_like(a[:, 1, 1], want), atol=1e-5), \
        f"slot 1 was not driven by its own kind: {a[:, 1, 1].tolist()}"
    assert not torch.allclose(a[:, 2, 1], torch.full_like(a[:, 2, 1], want), atol=1e-5), \
        "slot 2 was driven by the stub"
    assert torch.allclose(a[:, 0], torch.zeros_like(a[:, 0])), "the learner was driven by a teacher"


def test_the_interactive_kind_names_a_factory_so_it_plugs_in_with_one_entry():
    """"One entry when it merges" -- and this tree is after the merge, so it is the real teacher.

    The registry's own claim was that the branch landing would cost nothing but the import existing.
    That is what is checked here: the kind is available because `f1sim.interactive_teacher` imports,
    and the factory it names resolves to that module's `InteractiveTeacher` -- not to the raceline
    teacher, which is the substitution the registry exists to prevent.
    """
    kind = osl.kind_of("interactive")
    assert kind.teacher and kind.teacher_factory.startswith(kind.module + ":")
    assert kind.available                           # merged: integrate/20260916
    from f1sim.interactive_teacher import InteractiveTeacher
    mod_name, _, cls_name = kind.teacher_factory.partition(":")
    import importlib
    assert getattr(importlib.import_module(mod_name), cls_name) is InteractiveTeacher
    osl.validate_slots(osl.parse_slots('[{"kind": "interactive"}]'), 2)   # no longer refused


def test_the_interactive_kind_actually_drives_its_own_rows():
    """The whole point of the factory: a slot that says `interactive` is driven by that teacher.

    Checked the way the stub-kind test checks a stub -- through `alt_teachers`, which is where a
    non-raceline teacher kind is built and asked for the rows it owns.
    """
    env = _slot_env([{"kind": "interactive"}, {"kind": "raceline"}], envs=12)
    env.reset(seed=11)
    from f1sim.interactive_teacher import InteractiveTeacher
    assert env.alt_teacher_kinds == ("interactive",)
    assert isinstance(env.alt_teachers[0], InteractiveTeacher)
    assert env.alt_teachers[0].base is env.teacher, "it was not built from the raceline teacher"
    assert env.alt_teacher_mask["interactive"].view(-1, 3)[0].tolist() == [False, True, False]


def test_a_checkpoint_slot_is_driven_by_its_checkpoint(tmp_path):
    """The command of a checkpoint-driven car comes from the pool, not from the caller's action."""
    probe = _env(envs=8)
    probe.reset(seed=37)
    path = _tiny_checkpoint(tmp_path / "opp.pt", probe, 101)
    env = _slot_env([{"kind": "policy", "checkpoint": path}], envs=8)
    env.reset(seed=37)
    assert env.pool is not None and len(env.pool) == 1
    assert env.pool.entries[0].arm == "legacy" and env.pool.entries[0].memory_kind == "feedforward"
    a = torch.zeros(env.B, env.act_dim)
    an = env._opponent_actions(a)
    pooled = env.pool_driven
    assert bool(pooled.any())
    assert not torch.allclose(an[pooled], a[pooled]), "a checkpoint car drove the caller's action"
    assert torch.equal(an[~pooled], a[~pooled]), "the pool overwrote a policy-driven car"


def _tiny_checkpoint(path, env, seed: int, extra=None):
    """A checkpoint that fits `env`'s observation, with its own weights."""
    from f1sim.learn.model import ActorCritic, save_checkpoint
    torch.manual_seed(seed)
    spec = common.obs_spec(env)
    model = ActorCritic(n_stack=env.ecfg.scan_stack, n_beams=env.n_beams,
                        proprio_dim=spec.proprio_dim, priv_dim=21, act_dim=env.act_dim)
    meta = {"spec": dataclasses.asdict(spec)}
    meta.update(extra or {})
    save_checkpoint(str(path), model, meta)
    return str(path)


def test_two_slots_may_name_the_same_checkpoint(tmp_path):
    probe = _env(envs=8)
    probe.reset(seed=41)
    path = _tiny_checkpoint(tmp_path / "opp.pt", probe, 102)
    env = _slot_env([{"kind": "policy", "checkpoint": path},
                     {"kind": "policy", "checkpoint": path}], envs=12)
    env.reset(seed=41)
    assert len(env.pool) == 1, "one file was loaded twice"
    d = env.opp_driver.view(-1, 3)[0].tolist()
    assert d == [OPP_DRIVER_POLICY, OPP_DRIVER_POOL, OPP_DRIVER_POOL]


# =========================================================== 3. what it refuses
def test_a_table_that_does_not_match_the_race_size_is_refused():
    with pytest.raises(ValueError) as exc:
        _env(envs=8, race_size=2, opponent="slots",
             opponent_slots=[{"kind": "raceline"}, {"kind": "raceline"}])
    assert "race_size" in str(exc.value)


def test_a_table_without_the_slots_mode_is_refused():
    with pytest.raises(ValueError) as exc:
        _env(envs=8, race_size=2, opponent="teacher", opponent_slots=[{"kind": "raceline"}])
    assert "opponent='slots'" in str(exc.value)


def test_the_slots_mode_without_a_table_is_refused():
    with pytest.raises(ValueError) as exc:
        _env(envs=8, race_size=2, opponent="slots")
    assert "opponent_slots" in str(exc.value)


def test_a_kind_this_tree_does_not_have_is_named_and_refused(monkeypatch):
    """A kind whose module is missing is listed, disabled, and refused with the reason -- never
    silently mapped onto the raceline teacher, which would be a run whose label is a lie.

    `interactive` was that kind until 2026-09-16 and is now present, so the machinery is exercised
    with a registry entry naming a module nothing provides. What is being tested is the rule, not
    which kind happens to be missing today.
    """
    kind = osl.DriverKind("branchteacher", "branch 티처", "a kind that lives on a branch",
                          teacher=True, module="f1sim.not_merged_yet",
                          teacher_factory="f1sim.not_merged_yet:BranchTeacher",
                          pending="아직 병합되지 않은 브랜치에 있습니다.")
    monkeypatch.setitem(osl.KIND_BY_NAME, kind.name, kind)
    assert not kind.available and "병합" in kind.unavailable_message()
    with pytest.raises(ValueError) as exc:
        osl.validate_slots(osl.parse_slots('[{"kind": "branchteacher"}]'), 2)
    assert "branchteacher" in str(exc.value)


def test_an_oracle_checkpoint_is_refused_with_the_loader_s_message(tmp_path):
    """A privileged-opponent checkpoint takes the simulator's true opponent state as an input, so
    nothing outside that experiment can drive it. The console says so before the session is built;
    the env says so through the loader."""
    probe = _env(envs=8)
    probe.reset(seed=43)
    path = _tiny_checkpoint(tmp_path / "oracle.pt", probe, 103)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["meta"]["opp_token"] = True
    torch.save(ck, path)
    with pytest.raises(ValueError) as exc:
        _slot_env([{"kind": "policy", "checkpoint": path}], envs=8)
    assert "opp_token" in str(exc.value)
    from f1sim.viewer.console.opponent_table import describe_checkpoint
    assert "특권" in describe_checkpoint(path)


def test_a_checkpoint_whose_arm_the_slot_does_not_name_is_refused(tmp_path):
    probe = _env(envs=8)
    probe.reset(seed=47)
    path = _tiny_checkpoint(tmp_path / "armed.pt", probe, 104,
                            extra={"experiment": {"controller": {"arm": "fixed_low"}}})
    with pytest.raises(ValueError) as exc:
        _slot_env([{"kind": "policy", "checkpoint": path}], envs=8)
    assert "fixed_low" in str(exc.value)
    env = _slot_env([{"kind": "policy", "checkpoint": path, "controller": "fixed_low"}], envs=8)
    assert env.pool.entries[0].arm == "fixed_low"


@pytest.mark.parametrize("table,needle", [
    ('[{"kind": "policy"}]', "checkpoint path"),
    ('[{"kind": "self", "events": ["brake"], "event_rate": 1.0}]', "teacher-driven"),
    ('[{"kind": "self", "label_grip": "nominal"}]', "label_grip"),
    ('[{"kind": "raceline", "events": ["brake"]}]', "event_rate"),
    ('[{"kind": "raceline", "event_rate": 2.0}]', "no events"),
    ('[{"kind": "raceline", "checkpoint": "/x.pt"}]', "not driven by one"),
])
def test_a_slot_that_would_silently_do_nothing_is_refused(table, needle):
    slots = osl.parse_slots(table)
    with pytest.raises(ValueError) as exc:
        osl.validate_slots(slots, 2, require_files=False)
    assert needle in str(exc.value), str(exc.value)


@pytest.mark.parametrize("text,needle", [
    ('[{"kind": "nope"}]', "unknown opponent kind"),
    ('[{"kind": "raceline", "speed_scale": [1.2, 0.8]}]', "lo <= hi"),
    ('[{"kind": "raceline", "reactive": {"blocks": 1.0}}]', "unknown reactive"),
    ('[{"kind": "raceline", "spd_scale": 1.0}]', "unknown field"),
    ('[]', "empty table"),
    ('not json', "not JSON"),
])
def test_a_malformed_table_says_what_is_wrong_with_it(text, needle):
    with pytest.raises(ValueError) as exc:
        osl.parse_slots(text)
    assert needle in str(exc.value), str(exc.value)


# =========================================================== 4. one spelling
def test_the_spec_round_trips_through_json():
    text = ('[{"kind": "raceline", "speed_scale": [0.6, 1.15], "label_grip": "nominal", '
            '"speed_cap": 4.5, "events": ["brake", "shift"], "event_rate": 1.5, '
            '"reactive": {"defend": 0.4, "oblivious": 0.1}, "spawn": "alongside", "seed": 12}, '
            '{"kind": "self", "speed_scale": 0.8, "spawn": "behind"}]')
    slots = osl.parse_slots(text)
    again = osl.parse_slots(osl.slots_json(slots))
    assert slots == again
    assert json.loads(osl.slots_json(slots))[0]["speed_cap"] == 4.5


def test_the_flag_takes_a_file_as_well_as_a_string(tmp_path):
    p = tmp_path / "slots.json"
    p.write_text('[{"kind": "raceline", "speed_scale": 0.7}]', encoding="utf-8")
    assert osl.parse_slots(f"@{p}") == osl.parse_slots('[{"kind": "raceline", "speed_scale": 0.7}]')


def _parsed(argv):
    ap = oc.add_arguments(argparse.ArgumentParser())
    return ap.parse_args(argv)


def test_the_flag_reaches_the_env_config_and_names_the_mode():
    a = _parsed(["--race-size", "3", "--opp-slots",
                 '[{"kind": "raceline", "speed_scale": 0.6}, {"kind": "self"}]'])
    oc.validate(a)
    kw = oc.env_kwargs(a)
    assert kw["opponent"] == "slots" and len(kw["opponent_slots"]) == 2
    assert kw["opponent_slots"][0].speed_scale == (0.6, 0.6)
    assert oc.needs_racelines(a)
    assert "per-slot" in oc.describe(a) and "car 2: self" in oc.describe(a)


def test_a_table_of_checkpoints_alone_needs_no_raceline(tmp_path):
    ck = tmp_path / "x.pt"; ck.write_bytes(b"only the path is checked by the flag")
    a = _parsed(["--race-size", "2", "--opp-slots",
                 json.dumps([{"kind": "policy", "checkpoint": str(ck)}])])
    oc.validate(a)
    assert not oc.needs_racelines(a)


@pytest.mark.parametrize("extra", [
    ["--opponent", "teacher"], ["--opp-speed", "0.5", "1.0"], ["--opp-events", "brake"],
    ["--spawn-order", "ahead"], ["--mixed-teacher-frac", "0.3"], ["--opp-defend-prob", "0.5"],
])
def test_a_flag_the_table_replaces_is_refused_rather_than_ignored(extra):
    a = _parsed(["--race-size", "2", "--opp-slots", '[{"kind": "raceline"}]'] + extra)
    with pytest.raises(SystemExit) as exc:
        oc.validate(a)
    assert extra[0] in str(exc.value), str(exc.value)


def test_the_shared_ranges_still_apply_alongside_a_table():
    """Only the flags that speak for *every* opponent are superseded. The ranges an event draws
    from are shared by construction and stay on the command line."""
    a = _parsed(["--race-size", "2", "--spawn-gap", "1.0", "4.0", "--opp-react-max", "0.3",
                 "--opp-slots", '[{"kind": "raceline", "reactive": {"defend": 1.0}}]'])
    oc.validate(a)
    kw = oc.env_kwargs(a)
    assert kw["spawn_gap"] == (1.0, 4.0) and kw["opp_react_max"] == 0.3


def test_a_race_size_mismatch_is_refused_at_the_flag():
    a = _parsed(["--race-size", "2", "--opp-slots",
                 '[{"kind": "raceline"}, {"kind": "raceline"}]'])
    with pytest.raises(SystemExit) as exc:
        oc.validate(a)
    assert "race_size" in str(exc.value)


def test_the_session_config_carries_the_table():
    from f1sim.viewer.console.protocol import SessionConfig
    table = [{"kind": "raceline", "speed_scale": [0.6, 1.0]}, {"kind": "self"}]
    cfg = SessionConfig(cars_per_race=3, opponent="slots", opponent_slots=table)
    back = SessionConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert back.opponent_slots == table
    assert [s.kind for s in back.slots()] == ["raceline", "self"]
    assert back.opponent_summary() == "1x raceline, 1x self"
    # a different table is a different simulation, not a live toggle
    other = SessionConfig(cars_per_race=3, opponent="slots",
                          opponent_slots=[{"kind": "self"}, {"kind": "self"}])
    assert cfg.affects_simulation(other)


# =========================================================== 5. a race that runs
def test_a_four_car_race_with_mixed_slots_runs_on_cpu(tmp_path):
    """The CPU smoke the contract asks for: four cars, one of each kind, 50 steps, nothing raises.

    Not a quality claim -- a fresh checkpoint is a bad driver. What it shows is that a table whose
    rows are genuinely different builds, steps and resets: four driver paths inside one batch.
    """
    probe = _env(envs=8)
    probe.reset(seed=53)
    path = _tiny_checkpoint(tmp_path / "opp.pt", probe, 105)
    slots = [{"kind": "raceline", "speed_scale": [0.6, 1.15], "events": ["brake", "stop"],
              "event_rate": 3.0, "reactive": {"defend": 0.5}},
             {"kind": "policy", "checkpoint": path, "speed_scale": 0.8, "spawn": "behind"},
             {"kind": "self", "speed_scale": 0.9, "spawn": "ahead"}]
    env = _slot_env(slots, envs=8, track=WIDE)
    obs, _info = env.reset(seed=53)
    a = torch.zeros(env.B, env.act_dim)
    for _ in range(50):
        obs, rew, term, trunc, info = env.step(a)
    assert torch.isfinite(rew).all() and torch.isfinite(env.sim.state).all()
    assert info["opp_event"]["id"].shape == (env.B,)


def test_the_census_reports_one_row_per_slot():
    from f1sim.learn.opponent_census import Census
    env = _slot_env([{"kind": "raceline", "speed_scale": 0.6,
                      "reactive": {"defend": 1.0}},
                     {"kind": "raceline", "speed_scale": 1.0}], envs=12)
    env.reset(seed=59)
    census = Census(env, attack_range_m=3.0)
    a = torch.zeros(env.B, env.act_dim)
    for _ in range(40):
        view = env.learner_view()
        st = env.sim.state.clone()
        obs, _r, term, trunc, info = env.step(a)
        census.observe(view, info, st, term, trunc)
    rows = census.report()["slots"]
    assert [r["slot"] for r in rows] == [1, 2]
    assert rows[0]["mean_speed_scale"] == pytest.approx(0.6, abs=1e-5)
    assert rows[1]["mean_speed_scale"] == pytest.approx(1.0, abs=1e-5)
    assert rows[0]["reactive_car_seconds"]["defend"] >= 0.0
    from f1sim.learn.opponent_census import format_report
    assert "per slot" in format_report(census.report())
