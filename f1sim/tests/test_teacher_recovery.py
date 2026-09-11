"""The teacher has to be a usable recovery demonstrator, because DAgger teaches what it shows.

`plan_action` and `__call__` both scale the commanded speed by *lateral* error only, so a car sitting
on the raceline facing backwards was told to drive at the full profile speed: measured 3.06 m/s at
180 deg of heading error, on a commanded radius of 0.74 m. That needs 3.06^2 / 0.74 = 12.7 m/s^2 of
lateral acceleration against the ~6 m/s^2 the speed profile itself assumes -- the car cannot follow
it, so the label teaches "carry racing speed while pointing the wrong way".

There is no reward in DAgger to penalise that with; the only fix is for the teacher to demonstrate a
recovery. `recover_time` caps the speed at what can still be turned back onto the lane.
"""
import math

import numpy as np
import pytest
import torch

from f1sim.raceline import Raceline
from f1sim.teacher import RacelineTeacher

RADIUS = 8.0


def _teacher(**kw) -> RacelineTeacher:
    angle = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    xy = np.stack([RADIUS * np.cos(angle), RADIUS * np.sin(angle)], 1)
    return RacelineTeacher(Raceline.from_xy(xy, np.full(len(xy), 5.0)), **kw)


def _state_at(teacher: RacelineTeacher, heading_error_deg: float, speed: float = 1.5) -> torch.Tensor:
    """One car on the line, offset from the lane direction by the given heading error."""
    point = teacher.xy[0, 0]
    tangent = teacher.tan[0, 0]
    yaw = math.atan2(float(tangent[1]), float(tangent[0])) + math.radians(heading_error_deg)
    state = torch.zeros(1, 7)
    state[0, 0], state[0, 1], state[0, 2], state[0, 3] = point[0], point[1], yaw, speed
    return state


@pytest.mark.parametrize("heading_error_deg", [0.0, 45.0, 90.0, 180.0])
def test_disabled_by_default_so_existing_runs_are_bit_identical(heading_error_deg):
    teacher = _teacher()
    assert teacher.recover_time == 0.0
    assert teacher.heading_speed_cap(_state_at(teacher, heading_error_deg)[:, 2],
                                     torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)) is None


def test_speed_command_ignores_heading_error_when_disabled():
    # The defect, pinned: aligned and fully backwards get the same commanded speed.
    teacher = _teacher()
    tid = torch.zeros(1, dtype=torch.long)
    aligned = teacher(_state_at(teacher, 0.0), None, tid)[0, 1]
    backwards = teacher(_state_at(teacher, 180.0), None, tid)[0, 1]
    assert backwards == pytest.approx(aligned, rel=1e-6)


def test_recovery_cap_slows_a_backwards_car_but_not_an_aligned_one():
    teacher = _teacher(recover_time=0.7)
    tid = torch.zeros(1, dtype=torch.long)
    plain = _teacher()
    aligned = teacher(_state_at(teacher, 0.0), None, tid)[0, 1]
    assert aligned == pytest.approx(plain(_state_at(plain, 0.0), None, tid)[0, 1], rel=1e-6)

    speeds = [float(teacher(_state_at(teacher, d), None, tid)[0, 1]) for d in (0, 45, 90, 135, 180)]
    assert speeds == sorted(speeds, reverse=True)                 # monotone in heading error
    assert speeds[-1] < 0.6 * speeds[0]                           # 180 deg is meaningfully slower


def test_recovery_cap_matches_the_turn_it_has_to_make():
    # v <= a_lat * t / psi is the speed from which the heading can still be recovered in time t.
    teacher = _teacher(recover_time=0.7, a_lat_recover=6.0, v_recover_min=0.0)
    for deg in (90.0, 135.0, 180.0):
        cap = teacher.heading_speed_cap(_state_at(teacher, deg)[:, 2],
                                        torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
        assert float(cap) == pytest.approx(6.0 * 0.7 / math.radians(deg), rel=1e-3)


def test_recovery_cap_never_stops_the_car_dead():
    teacher = _teacher(recover_time=0.7, v_recover_min=0.6)
    cap = teacher.heading_speed_cap(_state_at(teacher, 180.0)[:, 2],
                                    torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    assert float(cap) >= 0.6


def test_plan_action_speeds_respect_the_cap():
    from f1sim.mpc import PlanSpec, decode

    spec = PlanSpec()
    tid = torch.zeros(1, dtype=torch.long)
    speeds = {}
    for name, kw in (("off", {}), ("on", dict(recover_time=0.7))):
        teacher = _teacher(**kw)
        state = _state_at(teacher, 180.0)
        action = teacher.plan_action(state, None, tid, 4.0, spec)
        _, _, v0, v1 = decode(action, state[:, 3], 4.0, torch.full((1,), 4.0), spec)
        speeds[name] = (float(v0), float(v1))
    assert speeds["on"][0] < speeds["off"][0] and speeds["on"][1] < speeds["off"][1]
