"""The sprung-mass attitude model after the 2026-09-13 calibration: roll follows `roll_per_g`,
pitch is asymmetric (`pitch_per_g` squat under throttle, `dive_per_g` under braking), and the
road-tilt process puts `road_tilt` rms of random roll / pitch in while driving straight -- and
none when it is 0."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from f1sim import Config, Simulator                      # noqa: E402
from f1sim.raceline import Raceline                      # noqa: E402
from f1sim.teacher import RacelineTeacher                # noqa: E402
from f1sim.track import Track                            # noqa: E402


def _sim(track, n=2, **vehicle):
    cfg = Config()
    cfg.sim.compile = False
    cfg.rand.enabled = False
    cfg.lidar.n_beams = 31
    for k, v in vehicle.items():
        setattr(cfg.vehicle, k, v)
    sim = Simulator(track, cfg, num_envs=n, device="cpu")
    sim.reset(poses=sim.sample_spawn(n, lateral_std=0.0, yaw_std=0.0))
    return sim, cfg


@pytest.fixture(scope="module")
def track():
    return Track.generate_random(4)


def _drive(sim, track, seconds, cmd_fn):
    rows = []
    for _ in range(int(seconds / sim.control_dt)):
        r = sim.step(cmd_fn(sim))
        rows.append(np.concatenate([r.attitude[:, 0].numpy(), r.attitude[:, 1].numpy(),
                                    sim.ay.numpy(), sim.ax.numpy()]))
    a = np.array(rows)                       # (T, 4B)
    B = sim.B
    return a[:, :B], a[:, B:2 * B], a[:, 2 * B:3 * B], a[:, 3 * B:]


def test_roll_and_pitch_gains_are_the_configured_ones(track):
    sim, cfg = _sim(track, road_tilt=0.0, roll_per_g=0.05, pitch_per_g=0.06, dive_per_g=0.01)
    teacher = RacelineTeacher([Raceline.build_cached(track)], wheelbase=cfg.vehicle.lf + cfg.vehicle.lr, device="cpu")
    roll, pitch, ay, ax = _drive(sim, track, 30.0, lambda s: teacher(s.state, s.P, s.tid))
    G = 9.81
    roll, pitch, ay, ax = roll[80:].ravel(), pitch[80:].ravel(), ay[80:].ravel(), ax[80:].ravel()
    # quasi-static: the suspension is far faster than the teacher's cornering, so a regression of
    # the angle on the acceleration returns the gain
    k_roll = np.polyfit(ay / G, roll, 1)[0]
    assert k_roll == pytest.approx(0.05, rel=0.15), k_roll
    sq = ax > 0.5; dv = ax < -0.5
    assert sq.sum() > 50 and dv.sum() > 50, "the teacher must both accelerate and brake"
    k_squat = -np.polyfit(ax[sq] / G, pitch[sq], 1)[0]
    k_dive = -np.polyfit(ax[dv] / G, pitch[dv], 1)[0]
    assert k_squat == pytest.approx(0.06, rel=0.3), k_squat
    assert abs(k_dive) < 0.03, k_dive                          # near the 0.01 set, far from the 0.06 squat


def test_road_tilt_adds_its_rms_on_top_of_the_cornering_roll(track):
    def run(road_tilt):
        sim, cfg = _sim(track, road_tilt=road_tilt, road_tau=0.4)
        teacher = RacelineTeacher([Raceline.build_cached(track)], wheelbase=cfg.vehicle.lf + cfg.vehicle.lr, device="cpu")
        roll, _, _, _ = _drive(sim, track, 20.0, lambda s: teacher(s.state, s.P, s.tid))
        assert not bool(sim.collided.any()), "a frozen (collided) car has no attitude to measure"
        return roll[100:]
    r0, r1 = run(0.0), run(0.02)
    var_added = float(r1.var() - r0.var())
    assert 0.5 * 0.02 ** 2 < var_added < 1.6 * 0.02 ** 2, (float(r0.std()), float(r1.std()))
    assert abs(float(r1.mean() - r0.mean())) < 0.01                # zero-mean: no lasting lean


def test_attitude_state_survives_reset_and_freeze(track):
    sim, _ = _sim(track)
    assert sim.att.shape == (2, 6)
    r = sim.step(torch.tensor([[0.0, 2.0], [0.0, 2.0]]))
    assert r.attitude.shape == (2, 2)
    sim.reset(env_ids=torch.tensor([0]))
    assert torch.all(sim.att[0] == 0)
