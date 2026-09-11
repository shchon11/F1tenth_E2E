"""Contracts found broken (or worth pinning) by the 2026-09-10 simulator audit.

Every test here runs on the CPU and stays under a second. They lock four things:

* the role mask handed out with a transition describes *that* transition, not the episode that
  replaced it,
* an ended episode is attributed to the track it was driven on,
* the privileged opponent block has a declared unit, and the auxiliary loss masks on that unit,
* the IMU's internal clock agrees with the rate it claims to run at.
"""
import math
import sys
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from f1sim import Config, Track
from f1sim.gym_env import EnvConfig, F1VecEnv


@pytest.fixture(scope="module")
def tracks():
    return [Track.generate_random(seed) for seed in (11, 12)]


def make_env(tracks, num_envs=2, **options):
    cfg = Config()
    cfg.sim.compile_mode = "none"
    cfg.lidar.n_beams = 36
    defaults = dict(max_steps=20, spawn_lateral_std=0.0, spawn_yaw_std=0.0,
                    hist_len=2, resample_track_on_reset=False)
    env = F1VecEnv(tracks, cfg, EnvConfig(**{**defaults, **options}), num_envs=num_envs, device="cpu")
    env.reset(seed=7)
    return env


# --------------------------------------------------------------------------- T1: role mask alias
def test_on_policy_mask_describes_the_transition_not_the_next_episode(tracks):
    """`info["on_policy"]` weights the transition that was just produced.

    A race in "mixed" mode re-draws which cars the policy drives on every full reset. The mask is
    read by the trainer *after* env.step returns, so if the env hands out its live tensor the last
    transition of every episode is weighted by the roles of the episode that replaced it.
    """
    env = make_env(tracks, num_envs=2, race_size=2, opponent="policy", max_steps=2)
    obs, _, _, _, info = env.step(torch.zeros(2, env.act_dim))
    captured = info["on_policy"]
    assert captured is not env.on_policy, (
        "info['on_policy'] is the env's live tensor; _reset_envs mutates it in place later in the "
        "same step, so the trainer would weight the ended transition by the next episode's roles")

    # _reset_envs re-draws the roles with exactly this kind of in-place write (gym_env: on_policy
    # [ids] = ...). The snapshot handed out with the transition must not follow it.
    before = captured.clone()
    env.on_policy[:] = ~env.on_policy
    assert torch.equal(captured, before), "the published mask followed a later in-place role change"


# --------------------------------------------------------------------------- T2/T6: track attribution
def test_track_id_in_info_is_the_track_the_episode_was_driven_on(tracks, monkeypatch):
    """The trainer attributes an ended episode's collisions to a track for the per-track gate.

    `env.sim.tid` has already been re-drawn by the auto-reset by the time env.step returns, so the
    attribution has to use the pre-reset snapshot the env publishes.
    """
    env = make_env(tracks, num_envs=2, resample_track_on_reset=True)
    env.sim.tid[:] = 0
    env.ep_step[0] = 19                      # row 0 truncates on this step
    original_reset = env._reset_envs

    def reset_onto_the_other_track(ids):
        out = original_reset(ids)
        env.sim.tid[ids] = 1                 # the new episode runs on track 1
        return out

    monkeypatch.setattr(env, "_reset_envs", reset_onto_the_other_track)
    obs, _, term, trunc, info = env.step(torch.zeros(2, env.act_dim))

    assert "final" in info and info["final"]["ids"].tolist() == [0]
    ids = info["final"]["ids"]
    assert env.sim.tid[ids].tolist() == [1], "precondition: the reset moved row 0 to track 1"
    assert info["track_id"][ids].tolist() == [0], (
        "info['track_id'] must be the pre-reset snapshot so an ended episode is scored against the "
        "track it was actually driven on")


def test_ppo_attributes_ended_episodes_with_the_pre_reset_track():
    """Guard on the trainer's own attribution expression (secondary to the contract test above)."""
    import ast
    import inspect
    from f1sim.learn import ppo

    tree = ast.parse(inspect.getsource(ppo))
    uses = [ast.unparse(node) for node in ast.walk(tree)
            if isinstance(node, ast.For) and "track_hist" in ast.unparse(node)]
    assert uses, "could not find the per-track attribution loop"
    joined = "\n".join(uses)
    assert "env.sim.tid" not in joined, (
        "per-track attribution reads env.sim.tid, which the auto-reset has already re-drawn")
    assert "track_id" in joined, "per-track attribution should consume info['track_id']"


# --------------------------------------------------------------------------- T3/T4: opponent units
def test_privileged_opponent_block_is_metres_over_five(tracks):
    """Unit spec for privileged()[8:12] in a race: [dx/5, dy/5, dv/5, dist/5], all in metres/5."""
    env = make_env(tracks, num_envs=2, race_size=2, opponent="policy")
    env.step(torch.zeros(2, env.act_dim))
    pv = env.privileged(env.last_result)
    st = env.sim.state
    other = env.sim.other_idx
    d = st[other][:, :, :2] - st[:, None, :2]
    dist_m = d.norm(dim=2).min(1).values
    torch.testing.assert_close(pv[:, 11] * 5.0, dist_m, rtol=1e-5, atol=1e-5)


def test_aux_opponent_mask_selects_six_metres():
    """The auxiliary opponent loss is meant to score only cars close enough to be in the scan.

    priv[:, 11] is dist/5, so comparing it against 6.0 selects 30 m -- three times the LiDAR's own
    range, i.e. a mask that removes nothing.
    """
    from f1sim.learn import ppo

    scale = getattr(ppo, "PRIV_OPP_DIST_SCALE", None)
    limit = getattr(ppo, "AUX_OPP_RANGE_M", None)
    assert scale == 5.0, "the privileged distance scale must be declared, in metres"
    assert limit == 6.0, "the auxiliary opponent range must be declared, in metres"

    priv_col = torch.tensor([1.0, 5.9, 6.1, 29.0, 31.0]) / scale     # metres -> privileged units
    near = priv_col * scale < limit
    assert near.tolist() == [True, True, False, False, False]


# --------------------------------------------------------------------------- T5: run provenance
def test_wandb_config_records_track_names_when_resuming_with_optimizer_state(tracks, monkeypatch,
                                                                            tmp_path):
    """The run's config must identify the track set, including on the --init + optimizer path."""
    from f1sim.learn import ppo

    env = make_env(tracks)
    env.ecfg.max_steps = 2
    captured = {}

    def fake_wandb_init(name, config, **kwargs):
        captured["config"] = dict(config)
        return SimpleNamespace(id="test", log=lambda *a, **k: None, finish=lambda: None)

    def fake_load_checkpoint(path, device, override=None, **kw):
        # **kw: the real loader has grown `allow_conditional` and `priv_adapter`, and a fake
        # with a narrower signature fails as a TypeError that looks like a bug in the caller.
        from f1sim.learn.model import ActorCritic
        model = ActorCritic(override["n_stack"], override["n_beams"], override["proprio_dim"],
                            override["priv_dim"], act_dim=override["act_dim"]).to(device)
        names = [n for n, _ in model.named_parameters()]
        # a real Adam state_dict, so the restore branch runs the way a resume actually does
        opt_state = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-5).state_dict()
        return model, {"opt": opt_state, "opt_param_names": names}

    monkeypatch.setattr(env.sim, "warmup", lambda: None)
    monkeypatch.setattr(ppo.common, "load_tracks", lambda *a, **k: (tracks, []))
    monkeypatch.setattr(ppo.common, "make_env", lambda *a, **k: env)
    monkeypatch.setattr(ppo.common, "run_dir", lambda name: str(tmp_path))
    monkeypatch.setattr(ppo.common, "wandb_init", fake_wandb_init)
    monkeypatch.setattr(ppo.common, "track_names", lambda spec: ["real:alpha", "real:beta"])
    monkeypatch.setattr(ppo, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(ppo, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["ppo", "--device", "cpu", "--total", "8", "--horizon", "2",
                                      "--epochs", "1", "--minibatch", "4", "--init", str(tmp_path / "x.pt"),
                                      "--wandb", "disabled"])
    ppo.main()

    assert "config" in captured, "wandb_init was never called"
    assert captured["config"]["tracks"] == ["real:alpha", "real:beta"], (
        f"run config recorded {captured['config']['tracks'][:3]} instead of the track manifest")


# --------------------------------------------------------------------------- IMU clock
def test_imu_sample_period_matches_the_actual_spacing():
    """Whatever rate the emulator delivers, the dt it integrates with must be that rate's period.

    `sample()` uses this dt for the gyro bias random walk (sqrt(dt)), for integrating the gyro into
    the attitude estimate and for the complementary-filter gain (dt / tau). Handing it 1/50 s while
    emitting one sample every 1/40 s makes all three disagree with the simulation clock.
    """
    from f1sim.sim import Simulator

    cfg = Config()
    cfg.sim.compile_mode = "none"
    cfg.lidar.n_beams = 36
    sim = Simulator([Track.generate_random(3)], cfg, num_envs=1, device="cpu")
    assert sim.imu_ts == pytest.approx(1.0 / cfg.imu.imu_rate), (
        "the sensor period handed to sample() must be the sensor's own period; samples are emitted "
        "on that clock")


def test_imu_delivers_its_declared_rate_over_a_non_integer_ratio():
    """0.1 s of simulation at 50 Hz is five IMU events, not four.

    50 Hz against a 40 Hz control step is not an integer ratio, so a fixed count per control step
    cannot be right: the emitter has to alternate 1, 1, 1, 2 to average the declared rate.
    """
    from f1sim.sim import Simulator

    cfg = Config()
    cfg.sim.compile_mode = "none"
    cfg.lidar.n_beams = 36
    sim = Simulator([Track.generate_random(3)], cfg, num_envs=1, device="cpu")
    assert cfg.imu.imu_rate == 50.0 and sim.control_dt == pytest.approx(0.025)

    emitted = 0
    cmd = torch.zeros(1, 2)
    for _ in range(4):                       # 4 control steps = 0.1 s
        r = sim.step(cmd)
        emitted += 0 if r.imu is None else int(r.imu.shape[1])
    assert emitted == 5, f"0.1 s at 50 Hz must produce 5 IMU samples, got {emitted}"


def test_imu_sample_offsets_say_when_each_sample_was_taken():
    """A sample's timestamp is only recoverable if the step says how long before its end it fell.

    The ROS bridges stamp `now - offset`. With a free-running sensor the last sample of a control
    step is generally NOT at the step boundary, so assuming it is puts every sample at a time it
    was not taken.
    """
    from f1sim.sim import Simulator

    cfg = Config()
    cfg.sim.compile_mode = "none"
    cfg.lidar.n_beams = 36
    sim = Simulator([Track.generate_random(3)], cfg, num_envs=1, device="cpu")
    ts = 1.0 / cfg.imu.imu_rate

    seen = []
    cmd = torch.zeros(1, 2)
    for step in range(8):
        r = sim.step(cmd)
        assert r.imu_offsets is not None, "StepResult must carry per-sample offsets"
        assert r.imu_offsets.shape[0] == r.imu.shape[1], "one offset per emitted sample"
        end = (step + 1) * sim.control_dt
        for off in r.imu_offsets.tolist():
            assert 0.0 <= off < sim.control_dt + 1e-9, f"offset {off} outside the control step"
            seen.append(end - float(off))

    # every recovered timestamp must sit on the sensor's own grid, and they must be strictly
    # increasing and exactly one sensor period apart
    for i, t in enumerate(seen):
        assert t == pytest.approx((i + 1) * ts, abs=1e-9), (
            f"sample {i} recovered at t={t:.6f}s, expected {(i + 1) * ts:.6f}s")


def test_imu_schedule_is_periodic_and_bounded():
    """The emitter cycles through a fixed, small set of per-step layouts.

    Each distinct layout is a separate compiled graph on CUDA, so the period has to stay small
    enough not to thrash the compile cache.
    """
    from f1sim.sim import Simulator

    cfg = Config()
    cfg.sim.compile_mode = "none"
    cfg.lidar.n_beams = 36
    sim = Simulator([Track.generate_random(3)], cfg, num_envs=1, device="cpu")
    counts = [len(idx) for idx in sim.imu_schedule]
    assert sum(counts) * (1.0 / cfg.imu.imu_rate) == pytest.approx(len(counts) * sim.control_dt), (
        "one full cycle of the schedule must cover a whole number of both clocks")
    assert len(sim.imu_schedule) <= 8, "too many compiled variants"
    assert counts == [1, 1, 1, 2], f"50 Hz on a 40 Hz step should cycle 1,1,1,2; got {counts}"


def test_partial_reset_does_not_disturb_the_sensor_clock(tracks):
    """One car respawning is not a reason for the IMU's clock to jump."""
    from f1sim.sim import Simulator

    cfg = Config()
    cfg.sim.compile_mode = "none"
    cfg.lidar.n_beams = 36
    sim = Simulator(tracks, cfg, num_envs=2, device="cpu")
    cmd = torch.zeros(2, 2)
    sim.step(cmd)
    phase_before = sim._imu_phase
    sim.reset(torch.tensor([0]))
    assert sim._imu_phase == phase_before, "a per-env reset must not move the sensor phase"
    r = sim.step(cmd)
    assert r.imu.shape[1] == len(sim.imu_schedule[phase_before]), "the schedule kept advancing"


def test_warmup_leaves_the_sensor_clock_and_state_where_it_found_them(tracks):
    """warmup() exists to compile, not to advance anything.

    It steps the simulator and puts the state back. The IMU's phase is state: leaving it advanced
    slides the sample grid against `t` for the rest of the run, so a sample stamped from the step
    end would be attributed to a time the sensor never ticked at.
    """
    from f1sim.sim import Simulator

    cfg = Config()
    cfg.sim.compile_mode = "none"
    cfg.lidar.n_beams = 36
    sim = Simulator(tracks, cfg, num_envs=2, device="cpu")
    cmd = torch.zeros(2, 2)
    for start in range(sim.imu_period):          # warm up from every phase, not just phase 0
        while sim._imu_phase != start:
            sim.step(cmd)
        before = {k: getattr(sim, k).clone() for k in
                  ("state", "ax", "ay", "att", "s", "lap", "collided", "steps", "imu_state",
                   "cl_idx", "car_collision")}
        t_before, phase_before = sim.t, sim._imu_phase
        odom_before = sim.odom.state.clone()

        sim.warmup()

        assert sim._imu_phase == phase_before, f"warmup moved the sensor phase from {phase_before}"
        assert sim.t == t_before
        torch.testing.assert_close(sim.odom.state, odom_before)
        for k, v in before.items():
            torch.testing.assert_close(getattr(sim, k), v, msg=f"warmup changed {k}")
        # and the next sample really is the one the pre-warmup clock was due to produce
        r = sim.step(cmd)
        assert r.imu.shape[1] == len(sim.imu_schedule[phase_before])
        torch.testing.assert_close(r.imu_offsets, sim._imu_offsets[phase_before])


@pytest.mark.parametrize("rate,counts", [(50.0, [1, 1, 1, 2]), (100.0, [2, 3]), (200.0, [5]),
                                         (40.0, [1]), (80.0, [2])])
def test_known_good_rates_keep_exact_counts_and_offsets(rate, counts):
    """The rates this project actually uses stay exact, and their offsets land on the sensor grid."""
    from f1sim.imu import sample_schedule

    control_dt, dt, substeps = 0.025, 0.001, 25
    schedule, offsets, period = sample_schedule(rate, control_dt, dt, substeps)
    assert [len(s) for s in schedule] == counts
    assert sum(len(s) for s in schedule) / (period * control_dt) == pytest.approx(rate)
    flat = [(p + 1) * control_dt - off for p, offs in enumerate(offsets) for off in offs]
    for i, t in enumerate(sorted(flat)):
        assert t == pytest.approx((i + 1) / rate, abs=1e-9)


@pytest.mark.parametrize("rate,reason", [
    (25.0, "slower than the control loop leaves empty steps"),
    (0.0, "not a positive rate"),
    (-50.0, "not a positive rate"),
    (float("nan"), "not finite"),
    (float("inf"), "not finite"),
    (2000.0, "faster than the 1 kHz physics tick"),
    (57.0, "needs too long a cycle to stay phase-exact"),
])
def test_unsupported_rates_are_refused_not_silently_truncated(rate, reason):
    """An unsupported rate has to fail loudly; the old code floored it and carried on."""
    from f1sim.imu import sample_schedule

    with pytest.raises(ValueError):
        sample_schedule(rate, 0.025, 0.001, 25)


def test_warmup_does_not_change_a_seeded_run_cpu(tracks):
    """A seeded run must produce the same numbers whether or not warmup was called.

    The throw-away steps draw from two generators: `sim.gen` (spawn, command jitter) and the GLOBAL
    torch generator, which is what imu.sample's torch.randn and the lidar/odom `*_like` noise use.
    Restoring only the first left sensor noise shifted, so "warmup restores the state" was not true.

    CPU only: the CUDA global stream is restored by the same code but is not asserted here.
    """
    from f1sim.sim import Simulator

    def run(with_warmup):
        cfg = Config()
        cfg.sim.compile_mode = "none"
        cfg.lidar.n_beams = 36
        torch.manual_seed(4321)
        sim = Simulator(tracks, cfg, num_envs=4, device="cpu")
        sim.gen.manual_seed(99)
        sim.reset(torch.arange(4))
        torch.manual_seed(555)                       # the seed a caller would set before driving
        if with_warmup:
            sim.warmup()
        cmd = torch.zeros(4, 2); cmd[:, 1] = 2.0
        out = []
        for _ in range(6):
            r = sim.step(cmd)
            out.append((r.state.clone(), r.imu.clone(), r.scan.clone()))
        return out

    a, b = run(False), run(True)
    for i, ((sa, ia, ca), (sb, ib, cb)) in enumerate(zip(a, b)):
        torch.testing.assert_close(sa, sb, msg=f"state diverged at step {i} because of warmup")
        torch.testing.assert_close(ia, ib, msg=f"IMU noise stream shifted by warmup at step {i}")
        torch.testing.assert_close(ca, cb, msg=f"LiDAR noise stream shifted by warmup at step {i}")


# --------------------------------------------------------------------------- track grid dedup
def _two_tracks_differing_only_in_tall():
    """Same occupancy and duct, different tall. Reachable through the public constructor.

    `Track.from_occupancy` takes `duct` and `tall` as independent arrays, and `from_ros_map`'s
    unknown-floor handling can move `tall` while leaving occupancy and duct alone. So "tall always
    changes with occupancy" is a property of the two obstacle builders, not of the type.
    """
    import numpy as np
    from f1sim.track import Track

    occ = np.zeros((60, 60), bool)
    occ[:2] = occ[-2:] = True
    occ[:, :2] = occ[:, -2:] = True
    duct = np.zeros_like(occ)
    tall_a = np.zeros_like(occ)
    tall_b = np.zeros_like(occ)
    tall_b[26:32, 26:32] = True                      # a tall block only track B has
    a = Track.from_occupancy(occ, 0.05, (-1.5, -1.5), name="A", duct=duct, tall=tall_a)
    b = Track.from_occupancy(occ, 0.05, (-1.5, -1.5), name="B", duct=duct, tall=tall_b)
    return a, b


def test_grid_key_separates_tracks_that_differ_only_in_the_tall_layer():
    """The GPU grid cache is keyed on grid_key, so anything the LiDAR traces must be in that key.

    grid_key hashed occupancy and duct only. TrackTensors dedups on it and, on a hit, skips
    appending edt/edt_duct/edt_tall entirely -- so the second track silently reused the first's
    edt_tall and the LiDAR traced the wrong tall geometry for it.
    """
    a, b = _two_tracks_differing_only_in_tall()
    assert not np.array_equal(a.tall, b.tall), "precondition: the tall layers differ"
    assert np.array_equal(a.occupancy, b.occupancy) and np.array_equal(a.duct, b.duct)
    assert a.grid_key() != b.grid_key(), (
        "two tracks the LiDAR must trace differently share a grid-cache key")


def test_track_tensors_keeps_a_separate_tall_field_per_distinct_tall_layer():
    from f1sim.track import TrackTensors

    a, b = _two_tracks_differing_only_in_tall()
    tt = TrackTensors([a, b], "cpu")
    # sample edt_tall at the centre of B's block: A has no obstacle there, B does
    xy = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    tid = torch.tensor([0, 1])
    d = tt.sample_edt(xy, tid, field=tt.edt_tall)
    assert float(d[0]) != float(d[1]), (
        f"both tracks report tall-clearance {float(d[0]):.3f} m at the same point: the grid cache "
        f"handed track B track A's tall field")
    assert float(d[1]) == pytest.approx(0.0, abs=1e-6), "B has a tall block at the origin"
    assert float(d[0]) > 0.0, "A has none"


def test_reversed_track_still_shares_one_grid():
    """The dedup must keep working where it is meant to: same grids, reversed centerline."""
    from f1sim.track import Track, TrackTensors

    t = Track.generate_random(5)
    r = t.reversed()
    assert t.grid_key() == r.grid_key(), "a reversed lap must still share the GPU grid"
    tt = TrackTensors([t, r], "cpu")
    assert int(tt.t_off[0]) == int(tt.t_off[1]), "reversed track should reuse the same grid offset"
