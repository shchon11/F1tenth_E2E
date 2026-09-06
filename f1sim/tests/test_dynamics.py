import math
import numpy as np
import torch

from f1sim import Track, Config, Simulator


def open_field(size=60.0, res=0.1):
    n = int(size / res)
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return Track.from_occupancy(occ, res, (-size / 2, -size / 2))


def make_sim(n=1, **over):
    cfg = Config()
    cfg.rand.enabled = False
    cfg.actuator.cmd_delay = 0.0
    for k, v in over.items():
        grp, name = k.split(".")
        setattr(getattr(cfg, grp), name, v)
    sim = Simulator(open_field(), cfg, num_envs=n, device="cpu")
    sim.reset(poses=torch.zeros(n, 3))
    return sim


def test_straight_line_speed_tracking():
    sim = make_sim()
    for _ in range(int(4.0 / sim.control_dt)):
        r = sim.step(torch.tensor([[0.0, 3.0]]))
    st = r.state[0]
    assert abs(st[3].item() - 3.0) < 0.15, st          # speed tracked (drag costs a bit)
    assert abs(st[1].item()) < 0.02 and abs(st[2].item()) < 0.01   # no lateral drift
    assert not r.collision.any()


def test_low_speed_yaw_rate_is_kinematic():
    sim = make_sim()
    delta, v = 0.3, 1.0
    for _ in range(int(3.0 / sim.control_dt)):
        r = sim.step(torch.tensor([[delta, v]]))
    L = sim.cfg.vehicle.lf + sim.cfg.vehicle.lr
    vx = r.state[0, 3].item()
    beta = math.atan(sim.cfg.vehicle.lr / L * math.tan(delta))
    expected = vx * math.cos(beta) * math.tan(delta) / L
    assert abs(r.state[0, 5].item() - expected) / expected < 0.05


def test_high_speed_cornering_saturates_and_is_stable():
    """Moderate steer at a speed where the kinematic model would demand > mu*g: lateral accel
    must be friction-limited, the car must slide (understeer) rather than spin, stay finite."""
    sim = make_sim()
    for _ in range(int(6.0 / sim.control_dt)):
        r = sim.step(torch.tensor([[0.15, 6.0]]))
    st = r.state[0]
    assert torch.isfinite(st).all()
    v = math.hypot(st[3].item(), st[4].item())
    ay = v * st[5].item()
    L = sim.cfg.vehicle.lf + sim.cfg.vehicle.lr
    ay_kin = v ** 2 * math.tan(0.15) / L
    assert ay_kin > sim.cfg.vehicle.mu * 9.81, "test setup: kinematic demand must exceed grip"
    assert ay < sim.cfg.vehicle.mu * 9.81 * 1.05, ay
    assert ay > 0.6 * sim.cfg.vehicle.mu * 9.81, ay
    beta = math.degrees(math.atan2(st[4].item(), st[3].item()))
    assert abs(beta) < 25, beta


def test_full_lock_full_throttle_does_not_blow_up():
    sim = make_sim()
    for _ in range(int(6.0 / sim.control_dt)):
        r = sim.step(torch.tensor([[0.4189, 8.0]]))
    assert torch.isfinite(r.state).all()
    assert r.state[0, 5].abs() < 15


def test_servo_lag_and_delay():
    sim = make_sim(**{"actuator.cmd_delay": 0.05, "actuator.servo_tau": 0.05})
    # drive forward first so the car is moving, then step steering
    for _ in range(40):
        sim.step(torch.tensor([[0.0, 2.0]]))
    steers = []
    for _ in range(12):
        r = sim.step(torch.tensor([[0.3, 2.0]]))
        steers.append(r.state[0, 6].item())
    # after 25 ms (1 step) with 50 ms delay, steering must not have moved yet
    assert steers[0] < 1e-3, steers[:3]
    # rate limited at 3.2 rad/s -> after 300 ms (12 steps) about there
    assert steers[-1] > 0.25, steers


def test_randomization_changes_params_and_stays_finite():
    cfg = Config()
    sim = Simulator(open_field(), cfg, num_envs=64, device="cpu")
    assert sim.P["mu"].std() > 0.01
    assert sim.P["cmd_delay"].min() >= 0.0 and sim.P["cmd_delay"].max() <= 0.10
    for _ in range(80):
        r = sim.step(torch.stack([torch.rand(64) * 0.8 - 0.4, torch.rand(64) * 8], 1))
    assert torch.isfinite(r.state).all()
    assert torch.isfinite(r.scan[torch.isfinite(r.scan)]).all()


def test_soft_wall_contact_does_not_explode():
    cfg = Config(); cfg.rand.enabled = False; cfg.sim.terminate_on_collision = False
    tr = open_field(size=6.0, res=0.05)
    sim = Simulator(tr, cfg, num_envs=1, device="cpu")
    sim.reset(poses=torch.zeros(1, 3))
    for _ in range(200):
        r = sim.step(torch.tensor([[0.0, 4.0]]))
    assert torch.isfinite(r.state).all()
    assert r.state[0, 0] < 3.0        # still inside the room
    assert r.state[0, 3] < 1.0        # stopped against the wall
