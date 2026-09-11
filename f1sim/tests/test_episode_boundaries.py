"""Episode boundaries preserve terminal targets and fresh next-episode state.

Speeds in the observation are normalised by `EnvConfig.v_max_policy`, so the tests derive the
scale from the config rather than repeating a literal. They used to hardcode 8, which was the
value of v_max_policy when they were written; it is 10 now and every one of those assertions was
off by exactly that ratio.
"""
import math
import torch
import pytest

from f1sim import Config, Track
from f1sim.gym_env import EnvConfig, F1VecEnv


@pytest.fixture(scope="module")
def tracks():
    return [Track.generate_random(seed) for seed in (11, 12)]


def make_env(tracks, **options):
    cfg = Config()
    cfg.sim.compile_mode = "none"
    cfg.lidar.n_beams = 36
    ecfg = EnvConfig(max_steps=20, spawn_lateral_std=0.0, spawn_yaw_std=0.0,
                     hist_len=2, resample_track_on_reset=False, **options)
    env = F1VecEnv(tracks, cfg, ecfg, num_envs=2, device="cpu")
    env.reset(seed=7)
    return env


def test_partial_reset_keeps_terminal_privileged_and_refreshes_current_state(tracks, monkeypatch):
    # Given one expiring row and a reset that switches its track and dynamics.
    env = make_env(tracks)
    env.ep_step[0] = 19
    terminal = {}
    original_reset = env._reset_envs
    original_step = env.sim.step

    def capture_step(command):
        result = original_step(command)
        terminal["priv"] = env.privileged(result).clone()
        terminal["odom"] = result.odom.clone()
        terminal["state"] = result.state.clone()
        terminal["t"] = env.sim.t
        return result

    def reset_on_other_track(ids):
        env.sim.tid[ids] = 1 - env.sim.tid[ids]
        return original_reset(ids)

    monkeypatch.setattr(env.sim, "step", capture_step)
    monkeypatch.setattr(env, "_reset_envs", reset_on_other_track)
    # When a step ends only the first episode.
    obs, _, term, trunc, info = env.step(torch.zeros(2, 2))
    # Then terminal values survive re-randomization and current rows are coherent.
    assert trunc.tolist() == [True, False] and not term.any()
    torch.testing.assert_close(info["final_priv"], terminal["priv"][:1])
    torch.testing.assert_close(info["final_obs"]["speed"], terminal["odom"][:1, 3:4] / env.ecfg.v_max_policy)
    r = env.last_result
    _, lateral, _ = env.sim.track.project(env.state[:, :2], env.sim.tid)
    torch.testing.assert_close(r.lateral, lateral)
    torch.testing.assert_close(r.wall_dist, env.sim._footprint_clearance(env.state))
    torch.testing.assert_close(r.odom[:, 3:4] / env.ecfg.v_max_policy, obs["speed"])
    torch.testing.assert_close(r.state[1], terminal["state"][1])
    torch.testing.assert_close(r.odom[1], terminal["odom"][1])
    assert env.sim.t == terminal["t"]
    assert r.progress[0] == 0 and not r.imu[0].any()
    assert not torch.equal(env.privileged(r)[0, 8:16], info["final_priv"][0, 8:16])


def test_plan_tracker_receives_reset_speed_and_imu(tracks, monkeypatch):
    # Given a plan car that just timed out.
    env = make_env(tracks, action_mode="plan")
    env.ep_step[0] = 19
    obs, *_ = env.step(torch.zeros(2, env.act_dim))
    measured = {}
    original_call = type(env.tracker).__call__

    def capture_tracker(self, action, speed, cap, yaw_rate, **kwargs):
        measured["speed"] = speed.clone()
        measured["yaw_rate"] = yaw_rate.clone()
        return original_call(self, action, speed, cap, yaw_rate, **kwargs)

    monkeypatch.setattr(type(env.tracker), "__call__", capture_tracker)
    # When the first plan of the new episode is tracked.
    env.step(torch.zeros(2, env.act_dim))
    # Then the tracker uses the same fresh sensors as the policy.
    torch.testing.assert_close(measured["speed"], obs["speed"][:, 0] * env.ecfg.v_max_policy)
    torch.testing.assert_close(measured["yaw_rate"], obs["imu"][:, 2] * env.ecfg.imu_gyro_scale)


def test_first_finish_crossing_starts_full_lap_timer(tracks, monkeypatch):
    # Given the first finish crossing after a mid-track spawn, and a second one a plausible lap later.
    #
    # The gap between the two crossings has to be long enough that the lap could physically have
    # been driven: gym_env drops any "lap" quicker than track length / v_max_policy, deliberately,
    # so that a wobble across the line is not reported as the best lap of the run. This fixture used
    # to put the crossings 100 steps (2.5 s) apart, which that filter rejects -- the test was
    # measuring the filter, not the first-crossing semantics it is named after.
    env = make_env(tracks)
    env.ecfg.max_steps = 100_000
    length = float(env.sim.track.length[env.sim.tid[0]])
    min_steps = int(math.ceil(length / env.ecfg.v_max_policy / env.sim.control_dt))
    gap = min_steps + 20                                   # comfortably a real lap, still short
    original_step = env.sim.step
    lap = 1

    def cross_finish(command):
        result = original_step(command)
        result.lap.fill_(lap)
        return result

    monkeypatch.setattr(env.sim, "step", cross_finish)
    env.ep_step.fill_(99)
    env.step(torch.zeros(2, 2))                            # first crossing: starts the timer only
    env.ep_step.fill_(99 + gap)
    lap = 2
    # When the next finish crossing occurs a full lap later.
    *_, info = env.step(torch.zeros(2, 2))
    # Then the reported lap is the gap between crossings, not the time since the mid-track spawn.
    assert info["lap_times"].shape == (2,), "a physically plausible lap must be reported"
    torch.testing.assert_close(info["lap_times"], torch.full((2,), gap * env.sim.control_dt))


def test_reset_scan_includes_opponents(tracks, monkeypatch):
    # Given a two-car race and a LiDAR spy on the actual reset scan call.
    env = make_env(tracks, race_size=2)
    scan_cars = []
    original_scan = env.sim.lidar.scan

    def capture_scan(*args, **kwargs):
        scan_cars.append(kwargs.get("cars"))
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(env.sim.lidar, "scan", capture_scan)
    # When one car respawns behind its opponent.
    env._reset_envs(torch.tensor([0]))
    # Then LiDAR receives the opponent's geometry at its actual pose.
    #
    # `_car_boxes` returns a list of entries whose first element is a tag: the mesh model emits
    # ("mesh", poses, scale, porosity). assert_close cannot compare the tag -- it raises
    # "No comparison pair was able to handle inputs of type str and str" -- so the tags are compared
    # structurally and only the tensors numerically.
    assert scan_cars[0] is not None
    expected = env.sim._car_boxes(env.state)
    assert len(scan_cars[0]) == len(expected)
    for got, want in zip(scan_cars[0], expected):
        assert len(got) == len(want)
        for a, b in zip(got, want):
            if isinstance(b, torch.Tensor):
                torch.testing.assert_close(a, b)
            else:
                assert a == b, f"geometry tag differs: {a!r} != {b!r}"


def test_speed_cap_gate_keeps_history_across_logging(tracks, monkeypatch, tmp_path):
    # Given four safe two-step rollouts, each ending a short episode.
    import sys
    from types import SimpleNamespace
    from f1sim.learn import ppo

    env = make_env(tracks)
    env.ecfg.max_steps = 2
    caps = []
    conditioning = []
    original_sample = ppo.sample_rollout_action

    def capture_conditioning(model, scan, proprio, cond=None):
        # `cond` is forwarded rather than dropped: the spy stands in for the real sampler, so its
        # signature has to track `ppo.sample_rollout_action(model, scan, proprio, cond=None)`.
        # Swallowing the argument would make this test pass while the conditioning vector silently
        # stopped reaching the policy.
        conditioning.append((proprio[:, 5].clone(), env.speed_cap.clone() / env.ecfg.v_max_policy))
        return original_sample(model, scan, proprio, cond)

    monkeypatch.setattr(ppo, "sample_rollout_action", capture_conditioning)
    original_cap = env.set_speed_cap

    def capture_cap(value):
        caps.append(value)
        original_cap(value)

    monkeypatch.setattr(env, "set_speed_cap", capture_cap)
    monkeypatch.setattr(env.sim, "warmup", lambda: None)
    monkeypatch.setattr(ppo.common, "load_tracks", lambda *args, **kwargs: (tracks, []))
    monkeypatch.setattr(ppo.common, "make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(ppo.common, "run_dir", lambda name: str(tmp_path))
    monkeypatch.setattr(ppo.common, "wandb_init", lambda *args, **kwargs: SimpleNamespace(log=lambda *args, **kwargs: None, finish=lambda: None))
    monkeypatch.setattr(ppo, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["ppo", "--device", "cpu", "--total", "16", "--horizon", "2",
                                    "--epochs", "1", "--minibatch", "4", "--cap0", "1", "--cap1", "2",
                                    "--cap-steps", "16", "--cap-gate", "0.5", "--wandb", "disabled"])
    # When training clears per-log episode statistics after updates one and two.
    ppo.main()
    # Then the gate still advances every safe rollout using retained history.
    assert caps == [1.0, 1.25, 1.5, 1.75]
    for observed, commanded in conditioning:
        torch.testing.assert_close(observed, commanded)


def test_transition_lap_is_not_cleared_by_autoreset(tracks, monkeypatch):
    # Given an expiring episode that just crossed the finish line.
    env = make_env(tracks)
    env.ep_step[0] = 19
    original_step = env.sim.step

    def cross_finish(command):
        result = original_step(command)
        result.lap[0] = 1
        return result

    monkeypatch.setattr(env.sim, "step", cross_finish)
    # When auto-reset clears the new episode's lap counter.
    *_, info = env.step(torch.zeros(2, 2))
    # Then transition info retains the completed episode's crossing.
    assert info["lap"][0] == 1
    assert env.last_result.lap[0] == 0


def test_terminal_snapshots_survive_the_next_autoreset(tracks):
    # Given retained terminal observations and a current post-reset sensor result.
    env = make_env(tracks)
    env.ecfg.max_steps = 1
    _, _, _, _, info = env.step(torch.zeros(2, 2))
    final_obs = {key: value.clone() for key, value in info["final_obs"].items()}
    final_priv = info["final_priv"].clone()
    current = env.last_result
    current_state = current.state.clone()
    current_odom = current.odom.clone()
    # When another episode steps and resets, replacing simulator buffers again.
    env.step(torch.zeros(2, 2))
    # Then both terminal targets and the retained current snapshot are immutable.
    torch.testing.assert_close(info["final_obs"], final_obs)
    torch.testing.assert_close(info["final_priv"], final_priv)
    torch.testing.assert_close(current.state, current_state)
    torch.testing.assert_close(current.odom, current_odom)
