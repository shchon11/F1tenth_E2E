"""Teacher feasibility agrees with simulator geometry, independently of soft costs."""
import math
from types import SimpleNamespace

import numpy as np
import torch

from f1sim.gym_env import F1VecEnv
from f1sim.interactive_teacher import InteractiveTeacher, OPP_HARD
from f1sim.params import Config
from f1sim.raceline import Raceline
from f1sim.sim import Simulator
from f1sim.teacher import RacelineTeacher
from f1sim.track import StaticProp, Track, TrackTensors


def teacher():
    a = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    line = Raceline.from_xy(np.stack([5 + 3 * np.cos(a), 5 + 3 * np.sin(a)], 1), np.full(80, 5.0))
    return InteractiveTeacher(RacelineTeacher(line))


def bare_track(prop=False):
    occ = np.zeros((200, 200), bool)
    occ[[0, -1]] = True
    occ[:, [0, -1]] = True
    a = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    cl = np.stack([5 + 3 * np.cos(a), 5 + 3 * np.sin(a)], 1)
    tr = Track.from_occupancy(occ, .05, (0., 0.), cl, "geometry-test")
    if prop:
        tr.props = (StaticProp("marker_post", 5., 5., 0., seed=0),)
    return TrackTensors(tr, "cpu")


def test_sat_matches_simulator_for_random_heading_and_rear_boxes():
    t = teacher()
    gen = torch.Generator().manual_seed(601)
    b = 256
    state = torch.zeros(b, 8)
    state[:, :2] = torch.rand(b, 2, generator=gen) * 1.1
    state[:, 2] = torch.rand(b, generator=gen) * 2 * math.pi
    sim = Simulator.__new__(Simulator)
    sim.device = torch.device("cpu")
    sim.M = 2
    sim.other_idx = (torch.arange(b) ^ 1)[:, None]
    sim.car_rear = torch.zeros(b, 4)
    sim.car_rear[:, 0] = torch.rand(b, generator=gen) * .12
    sim.corners = torch.tensor([[.29, .155], [.29, -.155], [-.29, .155], [-.29, -.155]])
    t.env = SimpleNamespace(sim=sim)
    world, yaw = state[None, :, None, :2], state[None, :, None, 2]
    future = state[sim.other_idx, :2][:, :, None]
    future_yaw = state[sim.other_idx, 2][:, :, None]
    actual = sim._car_contacts(state)
    predicted = t._body_overlap(world, yaw, future, future_yaw)[0, :, 0, 0]
    assert actual.any() and (~actual).any()
    assert torch.equal(actual, predicted)


def test_diagonal_rectangle_contact_missed_by_old_ellipse_and_safe_passing():
    t = teacher()
    world = torch.zeros(1, 1, 1, 2)
    yaw = torch.zeros(1, 1, 1)
    future = torch.tensor([[[[.55, .25]]]])
    assert ((future[..., 0] / OPP_HARD[0]) ** 2 + (future[..., 1] / OPP_HARD[1]) ** 2).item() > 1
    assert t._body_overlap(world, yaw, future, yaw).item()
    future[..., 0] = 0
    future[..., 1] = .32
    assert not t._body_overlap(world, yaw, future, yaw).item()


def test_oriented_nose_and_contained_asset_are_hard_violations():
    t = teacher()
    t.track = bare_track(prop=True)
    world = torch.tensor([[[[.24, 5.]]], [[[5., 5.]]], [[[5., 6.]]]])
    yaw = torch.zeros(3, 1, 1)
    wall, _, clear = t._wall_costs(world, torch.zeros(1, dtype=torch.long), yaw, True)
    assert clear[:, 0].tolist() == [False, False, True]
    assert (wall[:2] > 0).all()
    # Rotate parallel to the wall: the shorter half-width now fits.
    yaw[0] = math.pi / 2
    assert t._wall_costs(world, torch.zeros(1, dtype=torch.long), yaw, True)[2][0, 0]


def test_future_yaw_api_preserves_old_result_and_stopped_measured_heading():
    env = F1VecEnv.__new__(F1VecEnv)
    state = torch.zeros(2, 8)
    state[:, 2] = torch.tensor([1.1, -2.2])
    env.device = torch.device("cpu")
    env.M = 2
    env.sim = SimpleNamespace(state=state, other_idx=torch.tensor([[1], [0]]))
    env.ecfg = SimpleNamespace(opp_future_model="constv", overtake_range=20.)
    times = torch.tensor([0., .05, .2])
    old = env.opponent_future(times)
    new = env.opponent_future(times, return_yaw=True)
    assert torch.equal(old[0], new[0]) and torch.equal(old[1], new[1])
    assert torch.equal(new[2], state[env.sim.other_idx, 2, None].expand(2, 1, 3))


def test_tracker_yaw_interpolates_on_delay_clock_and_keeps_current_t0():
    env = F1VecEnv.__new__(F1VecEnv)
    state = torch.zeros(2, 8)
    state[:, 2] = 1.0
    env.device = torch.device("cpu")
    env.M = 2
    env.sim = SimpleNamespace(state=state, other_idx=torch.tensor([[1], [0]]), control_dt=.025)
    env.ecfg = SimpleNamespace(opp_future_model="pred", overtake_range=20.)
    z = torch.zeros(2, 3, 4)
    z[:, :, 2] = torch.tensor([0., .2, .4])
    env.tracker = SimpleNamespace(last_pred=z, spec=SimpleNamespace(delay=.025, dt=.1))
    env.tracker_delay = None
    env._plan_pose = torch.zeros(2, 3)
    env.teacher_driven = torch.zeros(2, dtype=torch.bool)
    times = torch.tensor([0., .05, .2, .4])
    _, _, yaw = env.opponent_future(times, return_yaw=True)
    assert torch.allclose(yaw[:, 0], torch.tensor([[1., .1, .4, .4]]).expand(2, -1), atol=1e-6)


def test_recorded_main032_contact_is_not_classified_clear():
    # Fixed evaluation main-032, trial 12, actual contact at 0.350 s.
    # Chassis-only SAT overlap was 7.18 mm while old ellipse reported 1.114.
    t = teacher()
    ego = torch.tensor([-4.042119979858398, .23770730197429657, -2.5079567432403564])
    opp = torch.tensor([-3.6644294261932373, .06576620042324066, -2.727229118347168])
    assert t._body_overlap(ego[:2].reshape(1, 1, 1, 2), ego[2].reshape(1, 1, 1),
                           opp[:2].reshape(1, 1, 1, 2), opp[2].reshape(1, 1, 1)).item()


def test_per_env_assets_keep_identity_when_candidates_and_times_are_tiled():
    t = teacher()
    t.track = bare_track(prop=True)
    original = t.track.props_near
    calls = []

    def own_layout(tid, xy, eid, reach):
        calls.append(eid.clone())
        values = list(original(tid, xy, eid, reach))
        values[0] = values[0].clone()
        # Same map, different layouts: row 1's post is 1 m away.
        values[0][:, :, 1] += eid[:, None]
        return tuple(values)

    t.track.props_near = own_layout
    world = torch.full((3, 2, 4, 2), 5.)
    _, _, clear = t._wall_costs(world, torch.zeros(2, dtype=torch.long), torch.zeros(3, 2, 4), True)
    assert clear.tolist() == [[False, True]] * 3
    assert calls[0].tolist() == [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4


def test_candidate_preview_matches_legacy_mpc_and_preserves_warm_state():
    from f1sim.mpc import PlanSpec, PlanTracker, solve
    t = teacher()
    spec = PlanSpec(dt=.05, N=12, delay=.05)
    tracker = PlanTracker(2, 'cpu', .33, .42, 9., spec, compile_solver=False)
    tracker.u_prev[:, 0] = torch.tensor([.05, -.08])
    tracker.u_seq[:, :, 0] = .03
    state = torch.zeros(2, 8)
    state[:, 3] = torch.tensor([3., 4.])
    delay = torch.tensor([.05, .1])
    t.env = SimpleNamespace(tracker=tracker, speed_cap=torch.full((2,), 9.),
                            last_result=None, tracker_delay=delay)
    cand = torch.zeros(2, 2, 8)
    cand[0, :, -2:] = -.5
    cand[1, :, 0] = .2
    saved_u, saved_warm = tracker.u_prev.clone(), tracker.u_seq.clone()
    world, yaw, speed = t._tracker_rollout(cand, state, 9., spec)
    for k in range(2):
        v = state[:, 3]
        rate = v * saved_u[:, 0].tan() / (tracker.wb + spec.k_us * v.square())
        _, expected, _ = solve(cand[k], v, t.env.speed_cap, rate, delay, saved_u,
                               torch.cat((saved_warm[:, 1:], saved_warm[:, -1:]), 1),
                               spec, tracker.wb, tracker.s_max, tracker.v_max)
        for b, offset in enumerate((1, 2)):
            assert torch.allclose(world[k, b, offset:offset+13], expected[b, :, :2], atol=1e-5)
            assert torch.allclose(yaw[k, b, offset:offset+13], expected[b, :, 2], atol=1e-5)
            assert torch.allclose(speed[k, b, offset:offset+13], expected[b, :, 3], atol=1e-5)
    assert torch.equal(tracker.u_prev, saved_u)
    assert torch.equal(tracker.u_seq, saved_warm)
    assert torch.equal(world[:, :, 0], state[None, :, :2].expand(2, -1, -1))
    assert t.last_preview_model == 'legacy_mpc'


def test_swept_guard_detects_between_sample_contact_with_clear_endpoints():
    t = teacher()
    world = torch.zeros(1, 1, 2, 2)
    yaw = torch.zeros(1, 1, 2)
    future = torch.tensor([[[[-.8, 0.], [.8, 0.]]]])
    assert not t._body_overlap(world, yaw, future, yaw).any()
    assert t._swept_body_overlap(world, yaw, future, yaw).all()


def test_swept_guard_preserves_same_velocity_following_and_side_pass():
    t = teacher()
    world = torch.tensor([[[[0., 0.], [.4, 0.]]]])
    yaw = torch.zeros(1, 1, 2)
    for relative in ([.65, 0.], [0., .32]):
        future = world[0, :, None] + torch.tensor(relative)
        assert not t._body_overlap(world, yaw, future, yaw).any()
        assert not t._swept_body_overlap(world, yaw, future, yaw).any()


def test_swept_rotating_body_contains_dense_intermediate_contacts():
    t = teacher()
    world = torch.zeros(1, 1, 2, 2)
    yaw = torch.tensor([[[0., math.pi / 2]]])
    future = torch.tensor([[[[.48, .25], [.48, .25]]]])
    other_yaw = torch.zeros(1, 1, 2)
    dense = torch.linspace(0, math.pi / 2, 101).reshape(1, 1, -1)
    hit = t._body_overlap(world[:, :, :1].expand(1, 1, 101, 2), dense,
                          future[:, :, :1].expand(1, 1, 101, 2), torch.zeros_like(dense))
    assert hit.any()
    assert t._swept_body_overlap(world, yaw, future, other_yaw).all()


def test_reset_rows_do_not_reuse_previous_episode_opponent_prediction():
    env = F1VecEnv.__new__(F1VecEnv)
    state = torch.zeros(2, 8)
    state[:, 2] = torch.tensor([0., math.pi / 2])
    state[:, 3] = 3.
    env.device = torch.device('cpu')
    env.M = 2
    env.sim = SimpleNamespace(state=state, other_idx=torch.tensor([[1], [0]]), control_dt=.025)
    env.ecfg = SimpleNamespace(opp_future_model='pred', overtake_range=20.)
    env.ep_step = torch.tensor([7, 0])
    z = torch.zeros(2, 3, 4)
    env.tracker = SimpleNamespace(last_pred=z, spec=SimpleNamespace(delay=.025, dt=.1))
    env.tracker_delay = None
    env._plan_pose = torch.zeros(2, 3)
    env.teacher_driven = torch.zeros(2, dtype=torch.bool)
    times = torch.tensor([0., .1, .2])
    future, _, yaw = env.opponent_future(times, return_yaw=True)
    # Ego row0 observes fresh row1: moving north, rather than the stale stationary prediction.
    assert torch.allclose(future[0, 0, :, 1], torch.tensor([0., .3, .6]))
    assert torch.allclose(yaw[0, 0], torch.full((3,), math.pi / 2))
    # Row0 has taken steps: preserve its real prediction.
    assert future[1, 0].count_nonzero() == 0


def test_current_motion_future_has_stable_straight_and_exact_turn_limits():
    t = teacher()
    t.env = SimpleNamespace(sim=SimpleNamespace(other_idx=torch.tensor([[1], [0]])))
    state = torch.zeros(2, 8)
    state[:, 3] = 2.
    state[0, 5] = math.pi / 2
    times = torch.tensor([0., .5, 1.])
    xy, yaw = t._current_motion_future(state, times)
    assert torch.isfinite(xy).all()
    assert torch.allclose(xy[0, 0, :, 0], 2 * times)
    assert xy[0, 0, :, 1].count_nonzero() == 0
    assert torch.allclose(xy[1, 0, -1], torch.full((2,), 4 / math.pi), atol=1e-6)
    assert torch.allclose(yaw[1, 0], times * math.pi / 2)


def test_clear_tracker_forecast_cannot_override_measured_closing_motion():
    from f1sim.mpc import PlanSpec
    t = teacher()
    spec = PlanSpec()
    times = t.horizon_times(spec)
    h = times.numel()
    state = torch.zeros(2, 8)
    state[1, 0] = 1.5
    state[1, 3] = -2.
    future = torch.zeros(2, 1, h, 2)
    future[0, 0, :, 0] = 1.5 + 2 * times  # optimistic prediction moves away
    future[1, 0, :, 0] = -5.
    t.env = SimpleNamespace(M=2, sim=SimpleNamespace(other_idx=torch.tensor([[1], [0]])),
        opponent_future=lambda *a, **kw: (future, torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 1, h)))
    world = torch.zeros(1, 2, h, 2)
    psi = torch.zeros(1, 2, h)
    assert not t._body_overlap(world, psi, future, torch.zeros(2, 1, h))[0, 0].any()
    _, clear = t._opp_cost(world, psi, state, spec, return_clear=True)
    assert not clear[0, 0]


def test_immediate_evasions_are_opposite_bounded_and_physically_projected():
    from f1sim.mpc import PlanSpec, decode
    t = teacher()
    spec = PlanSpec()
    state = torch.zeros(2, 8)
    state[:, 3] = torch.tensor([3., 7.])
    params = {'mu': torch.tensor([.8, .4]), 'mu_f_scale': torch.tensor([1., .9]),
              'mu_r_scale': torch.tensor([.9, 1.])}
    base = torch.zeros(2, 8)
    evasions = t._evasion_candidates(base, state, params, 10., spec)
    assert evasions.shape == (4, 2, 8)
    assert (evasions[:2, :, 0] > 0).all() and (evasions[2:, :, 0] < 0).all()
    assert torch.equal(evasions[:2, :, :6], -evasions[2:, :, :6])
    assert evasions[..., 2:6].count_nonzero() == 0
    assert evasions[0, 1, 0].abs() < evasions[0, 0, 0].abs()  # fast, slippery row
    assert (evasions[[1, 3], :, -2:] == -1).all()
    tiled = lambda x: x[None].expand(4, *x.shape).reshape(-1, *x.shape[1:])
    pp = {k: tiled(v) for k, v in params.items()}
    certified = t.base.project_plan_action(evasions.reshape(-1, 8), tiled(state), pp, 10., spec)
    again = t.base.project_plan_action(certified, tiled(state), pp, 10., spec)
    assert torch.allclose(certified, again, atol=1e-5)
    assert torch.isfinite(certified).all()


def test_zero_speed_evasions_are_finite_and_request_no_motion():
    from f1sim.mpc import PlanSpec
    t = teacher()
    state = torch.zeros(1, 8)
    actions = t._evasion_candidates(torch.zeros(1, 8), state, None, 10., PlanSpec())
    assert torch.isfinite(actions).all()
    assert (actions[..., -2:] == -1).all()
    assert actions.abs().max() <= 1


def test_custom_solver_does_not_claim_legacy_mpc_preview(monkeypatch):
    from f1sim.mpc import PlanSpec, PlanTracker
    t = teacher()
    spec = PlanSpec()
    tracker = PlanTracker(1, 'cpu', .33, .42, 9., spec, compile_solver=False)
    tracker._solver = lambda *a, **kw: None  # e.g. historical grip adapter
    t.env = SimpleNamespace(tracker=tracker)
    expected = (torch.ones(1), torch.ones(2), torch.ones(3))
    monkeypatch.setattr(t, 'rollout', lambda *a, **kw: expected)
    assert t._tracker_rollout(torch.zeros(1, 1, 8), torch.zeros(1, 8), 9., spec) is expected
    assert t.last_preview_model == 'reference'
