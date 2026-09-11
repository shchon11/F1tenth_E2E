"""The teacher's speed label must be a function of what the student can observe.

`RacelineTeacher.grip_bin` reads each env's randomized friction. That is legitimate for a privileged
baseline but it makes the *label* depend on something absent from the observation: identical scans
get speed labels up to 1/0.45 ~ 2.2x apart, and a Huber regression fits their conditional mean.
"""
import numpy as np
import torch

from f1sim.raceline import Raceline
from f1sim.teacher import RacelineTeacher


def _teacher(**kw) -> RacelineTeacher:
    angle = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    xy = np.stack([8 * np.cos(angle), 8 * np.sin(angle)], 1)
    line = Raceline.from_xy(xy, np.full(len(xy), 5.0))
    return RacelineTeacher(line, **kw)


def _params(mu: torch.Tensor) -> dict:
    n = mu.numel()
    return {"mu": mu, "mu_f_scale": torch.full((n,), 0.92), "cmd_delay": torch.zeros(n),
            "servo_tau": torch.zeros(n), "steer_bias": torch.zeros(n),
            "steer_gain": torch.ones(n), "speed_gain": torch.ones(n)}


def test_true_grip_makes_the_label_depend_on_unobservable_friction():
    teacher = _teacher()
    slippery, grippy = _params(torch.tensor([0.5, 1.0489]))["mu"], None
    params = _params(torch.tensor([0.5, 1.0489]))
    bins = teacher.grip_bin(params, 2, torch.device("cpu"))
    assert bins[0] != bins[1]                                   # same observation, different label

    # And the speed labels differ by roughly the grip ratio, which no observation distinguishes.
    idx = torch.zeros(2, dtype=torch.long)
    speeds = teacher.speed_at(torch.zeros(2, dtype=torch.long), idx, bins)
    assert speeds[1] / speeds[0] > 1.3
    assert slippery is not None and grippy is None              # keeps the intent of the fixture explicit


def test_nominal_and_conservative_grip_are_constant_across_envs():
    params = _params(torch.tensor([0.5, 0.75, 1.0489]))
    for mode, expected in (("nominal", 11), ("conservative", 0)):
        teacher = _teacher()
        teacher.label_grip = mode
        bins = teacher.grip_bin(params, 3, torch.device("cpu"))
        assert bins.tolist() == [expected] * 3                  # a function of the observation alone
        speeds = teacher.speed_at(torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long), bins)
        assert torch.allclose(speeds, speeds[0].expand(3))


def test_conservative_is_never_faster_than_nominal():
    params = _params(torch.tensor([1.0489]))
    speeds = {}
    for mode in ("conservative", "nominal"):
        teacher = _teacher()
        teacher.label_grip = mode
        bins = teacher.grip_bin(params, 1, torch.device("cpu"))
        speeds[mode] = teacher.speed_at(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), bins)
    assert speeds["conservative"] <= speeds["nominal"]


def test_default_stays_privileged_so_existing_runs_are_unchanged():
    teacher = _teacher()
    assert teacher.label_grip == "true"
    params = _params(torch.tensor([0.5, 1.0489]))
    assert teacher.grip_bin(params, 2, torch.device("cpu")).tolist() != [11, 11]


def test_missing_params_still_fall_back_to_nominal():
    teacher = _teacher()
    assert teacher.grip_bin(None, 4, torch.device("cpu")).tolist() == [11] * 4
