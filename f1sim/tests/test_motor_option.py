"""The opt-in real-car speed loop (`actuator.motor_ramp`, `actuator.motor_delay`): off by default,
and when on it caps the drive-side request and delays the speed channel only."""
import numpy as np
import pytest
import torch

from f1sim import Config, Simulator, Track
from f1sim import dynamics as dyn


@pytest.fixture(scope="module")
def track():
    n = 400
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return Track.from_occupancy(occ, 0.2, (-40.0, -40.0))


def run(track, secs, speed, **over):
    cfg = Config()
    cfg.rand.enabled = False
    cfg.actuator.cmd_delay = 0.0
    for k, v in over.items():
        setattr(cfg.actuator, k, v)
    sim = Simulator(track, cfg, num_envs=1, device="cpu")
    sim.reset(poses=torch.zeros(1, 3), speed=torch.full((1,), 2.0))
    vs = []
    for _ in range(int(round(secs / sim.control_dt))):
        sim.step(torch.tensor([[0.0, speed]]))
        vs.append(float(sim.state[0, dyn.IVX]))
    return np.array(vs), sim.control_dt


def test_off_by_default_is_the_old_loop(track):
    a, _ = run(track, 0.5, 5.0)
    b, _ = run(track, 0.5, 5.0, motor_ramp=0.0, motor_delay=0.0)
    assert np.array_equal(a, b)


def test_ramp_caps_the_drive_side(track):
    v, dt = run(track, 1.0, 5.0, motor_ramp=2.5)
    acc = np.diff(np.concatenate([[2.0], v])) / dt
    assert acc.max() <= 2.5 + 0.05
    free, _ = run(track, 1.0, 5.0)
    assert free[10] - 2.0 > 1.5 * (v[10] - 2.0)       # the default loop is far quicker


def test_delay_holds_the_speed_channel(track):
    v, dt = run(track, 0.5, 5.0, motor_delay=0.1)
    n_hold = int(round(0.1 / dt)) - 1
    # rolling resistance only until the delayed command arrives
    assert v[n_hold - 1] <= 2.0 + 1e-3
    assert v[-1] > 2.5
