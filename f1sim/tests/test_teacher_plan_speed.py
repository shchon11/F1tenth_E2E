"""The teacher fits the same metric plan that the tracker decodes from incoming odometry."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from f1sim import mpc
from f1sim.gym_env import F1VecEnv
from f1sim.interactive_teacher import InteractiveTeacher
from f1sim.raceline import Raceline
from f1sim.teacher import RacelineTeacher


def scene():
    angle = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    xy = np.stack([8 * np.cos(angle), 5 * np.sin(angle)], 1)
    line = Raceline.from_xy(xy, np.full(len(xy), 5.0))
    teacher = RacelineTeacher(line)
    state = torch.zeros(2, 8)
    ids = torch.tensor([0, 37])
    state[:, :2] = teacher.xy[0, ids]
    state[:, 2] = torch.atan2(teacher.tan[0, ids, 1], teacher.tan[0, ids, 0])
    state[:, 3] = torch.tensor([2.0, 3.0])
    return teacher, state, torch.tensor([5.0, 6.0])


@pytest.mark.parametrize("interactive", [False, True])
def test_equal_measured_and_body_speed_preserves_standalone_bits(interactive):
    teacher, state, _ = scene()
    if interactive:
        teacher = InteractiveTeacher(teacher)
    old = teacher.plan_action(state)
    assert torch.equal(old, teacher.plan_action(state, plan_speed=None))
    assert torch.equal(old, teacher.plan_action(state, plan_speed=state[:, 3]))


def test_measured_speed_controls_fit_and_decoder_length_in_si(monkeypatch):
    teacher, state, measured = scene()
    original_state = state.clone()
    spec = mpc.PlanSpec()
    lengths = []
    original = mpc.plan_length

    def record(speed, plan_spec):
        length = original(speed, plan_spec)
        lengths.append((speed.clone(), length.clone()))
        return length

    monkeypatch.setattr(mpc, "plan_length", record)
    action = teacher.plan_action(state, spec=spec, plan_speed=measured)
    _, decoded_length, _, _ = mpc.decode(action, measured, 8.0, torch.full((2,), 8.0), spec)
    assert len(lengths) >= 2
    for speed, length in lengths:
        assert torch.equal(speed, measured), "wheel speed was renormalized or replaced by body speed"
        assert torch.equal(length, decoded_length)
    wheel_state = state.clone()
    wheel_state[:, 3] = measured
    geometry_reference = teacher.plan_action(wheel_state, spec=spec)
    assert torch.equal(action[:, :mpc.N_KNOTS], geometry_reference[:, :mpc.N_KNOTS])
    assert not torch.equal(action[:, :mpc.N_KNOTS],
                           teacher.plan_action(state, spec=spec)[:, :mpc.N_KNOTS])
    assert torch.equal(state, original_state)


def test_interactive_tiles_measured_speed_into_generation_and_scoring(monkeypatch):
    import f1sim.interactive_teacher as module
    base, state, measured = scene()
    teacher = InteractiveTeacher(base)
    generated, decoded = [], []
    generate, decode = teacher._gen.plan_action, module.decode

    def generation(*args, **kwargs):
        generated.append(kwargs.get("plan_speed"))
        return generate(*args, **kwargs)

    def decoding(action, speed, *args, **kwargs):
        decoded.append(speed.clone())
        return decode(action, speed, *args, **kwargs)

    monkeypatch.setattr(teacher._gen, "plan_action", generation)
    monkeypatch.setattr(module, "decode", decoding)
    teacher.plan_action(state, plan_speed=measured)
    assert len(generated) == len(decoded) == 1
    assert torch.equal(generated[0], measured.repeat(teacher.offsets.numel()))
    assert torch.equal(decoded[0], measured.repeat(teacher.n_candidates))


def test_override_keeps_privileged_profile_lookup_and_moves_only_plan_endpoint(monkeypatch):
    teacher, state, measured = scene()
    spec = mpc.PlanSpec()
    indices, grips = [], []
    original_speed_at, original_grip_bin = teacher.speed_at, teacher.grip_bin
    params = {"mu": torch.tensor([0.75, 1.05]), "mu_f_scale": torch.full((2,), 0.92)}

    def profile(tid, idx, grip_bin):
        indices.append(idx.clone())
        return original_speed_at(tid, idx, grip_bin)

    def grip(values, *args):
        grips.append(values)
        return original_grip_bin(values, *args)

    monkeypatch.setattr(teacher, "speed_at", profile)
    monkeypatch.setattr(teacher, "grip_bin", grip)
    teacher.plan_action(state, params, spec=spec, plan_speed=measured)
    tid = torch.zeros(2, dtype=torch.long)
    start, _ = teacher.project(state[:, :2], tid)
    expect_start = (start + (state[:, 3].abs() * spec.v_cmd_lead / teacher.ds[tid]).round().long()) % teacher.N
    expect_end = (start + (mpc.plan_length(measured, spec) / teacher.ds[tid]).round().long()) % teacher.N
    assert len(grips) == 1 and grips[0] is params
    assert torch.equal(indices[0], expect_start)
    assert torch.equal(indices[1], expect_end)


def env_shell(teacher, state, measured):
    env = F1VecEnv.__new__(F1VecEnv)
    env.B, env.M, env.act_dim = len(state), 2, mpc.ACT_DIM
    env.sim = SimpleNamespace(state=state, P=None, tid=torch.zeros(len(state), dtype=torch.long))
    base = getattr(teacher, 'base', teacher)
    env.cfg = SimpleNamespace(vehicle=base.vehicle)
    env.sim.track = SimpleNamespace(t_res=torch.tensor([.05]), has_props=False,
        sample_edt=lambda xy, tid: torch.full(xy.shape[:-1], 10.0))
    env.sim.other_idx = torch.tensor([[1], [0]])
    env.sim.car_rear = torch.zeros(len(state), 4)
    env.speed_cap = torch.full((len(state),), 8.0)
    def future(times, *, return_yaw=False, **kwargs):
        values = (torch.full((len(state), 1, len(times), 2), 100.0), torch.zeros(len(state), 1))
        return (*values, torch.zeros(len(state), 1, len(times))) if return_yaw else values
    env.opponent_future = future
    env.ecfg = SimpleNamespace(v_max_policy=8.0, opponent="teacher")
    env.tracker = SimpleNamespace(spec=mpc.PlanSpec())
    odom = torch.zeros(len(state), 4)
    odom[:, 3] = measured
    env.last_result = SimpleNamespace(odom=odom)
    env.events = env.pool = None
    env.teacher_any, env.teacher = True, teacher
    env.teacher_driven = torch.tensor([False, True])
    env.alt_teacher_kinds, env.alt_teachers = [], []
    env.slots = None                      # no slot table: `_raceline_teacher_needed` reads it
    env.opp_scale = torch.ones(len(state))
    env.follow_cap = lambda _state: (torch.zeros(len(state), dtype=torch.bool), torch.full((len(state),), 8.0))
    return env


@pytest.mark.parametrize("interactive", [False, True])
def test_env_labels_and_opponents_forward_incoming_tracker_odometry(monkeypatch, interactive):
    teacher, state, measured = scene()
    if interactive:
        teacher = InteractiveTeacher(teacher)
    env = env_shell(teacher, state, measured)
    seen = []
    original = teacher.plan_action

    def capture(*args, **kwargs):
        seen.append(kwargs.get("plan_speed"))
        return original(*args, **kwargs)

    monkeypatch.setattr(teacher, "plan_action", capture)
    label = env.teacher_label(teacher)
    offset_label = env.teacher_label(teacher, offset=torch.zeros(env.B))
    actions = env._opponent_actions(torch.zeros_like(label))
    assert len(seen) == 3
    for speed in seen:
        assert speed is not None, "production caller omitted the measured-speed contract"
        assert speed.data_ptr() == env.last_result.odom[:, 3].data_ptr()
        assert torch.equal(speed, measured)
    assert torch.equal(actions[1], label[1])
    assert torch.equal(actions[0], torch.zeros_like(actions[0]))
    assert torch.allclose(offset_label, label)
    assert torch.equal(env._tracker_plan_speed(), measured)
    env.last_result = None
    assert torch.equal(env._tracker_plan_speed(), state[:, 3])


@pytest.mark.parametrize("bad", [torch.tensor(3.0), torch.ones(2, 1), torch.ones(3)])
def test_plan_speed_shape_is_not_silently_broadcast(bad):
    teacher, state, _ = scene()
    with pytest.raises(ValueError, match="plan_speed"):
        teacher.plan_action(state, plan_speed=bad)


def test_interactive_per_car_slots_match_independent_candidate_generation():
    from copy import copy
    from f1sim.teacher import LABEL_GRIP_CODE
    base, state, measured = scene()
    base.speed_scale = torch.tensor([.65, .95])
    base.label_grip_codes = torch.tensor([LABEL_GRIP_CODE['true'], LABEL_GRIP_CODE['conservative']])
    params = {'mu': torch.tensor([.75, 1.05]), 'mu_f_scale': torch.full((2,), .92)}
    interactive = InteractiveTeacher(base)
    batched = interactive.plan_action(state, params, plan_speed=measured)
    separate = []
    for i in range(2):
        one = copy(base)
        one.speed_scale = base.speed_scale[i:i+1]
        one.label_grip_codes = base.label_grip_codes[i:i+1]
        separate.append(InteractiveTeacher(one).plan_action(state[i:i+1],
                        {k: v[i:i+1] for k, v in params.items()}, plan_speed=measured[i:i+1]))
    torch.testing.assert_close(batched, torch.cat(separate), atol=2e-5, rtol=2e-5)
    assert base.label_grip_codes.shape == base.speed_scale.shape == (2,)
    assert torch.isfinite(batched).all()
