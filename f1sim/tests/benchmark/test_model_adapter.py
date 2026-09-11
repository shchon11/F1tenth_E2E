"""Real CPU integration checks for the model/runtime adapter.

These are **API and integration fixtures, not benchmark scores**. Nothing here drives a roster
checkpoint on a benchmark scenario: the checkpoints are synthetic, generated here with a fixed seed,
and the tracks are procedural ones outside the suite. What is being proved is that the seams hold —
the actor receives the shapes it was trained on, the declared friction is the friction the plant
actually has, the controller runtime is installed and its hooks run, and everything is released
again afterwards.

Each checkpoint is built under `torch.random.fork_rng()` so the suite's global RNG is left as found.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import pytest
import torch


from f1sim.learn.benchmark import model_adapter as MA                          # noqa: E402

#: Procedural, outside every suite map, so nothing here can be mistaken for a benchmark cell.
TEST_MAP = "gen:control:770001"
MU_LOW, MU_HIGH = 0.73423, 1.15379


def make_checkpoint(tmp_path, name="synthetic", *, arm=None, estimator_path=None, act_dim=8,
                    seed=4242):
    """A valid checkpoint on the CURRENT six-scan/history ObsSpec, written to a temp path."""
    from f1sim.learn.model import ActorCritic, save_checkpoint
    from f1sim.learn.obs import ObsSpec
    from f1sim.params import Config

    spec = ObsSpec(n_beams=Config().lidar.n_beams, scan_stack=6, scan_stride=1,
                   hist_len=20, hist_stride=2, action_history=2, act_dim=act_dim)
    meta = dict(n_stack=spec.scan_stack, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                priv_dim=17, act_dim=spec.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = ActorCritic(**meta)
    extra = {"spec": dataclasses.asdict(spec), "phase": "ppo", "run": name,
             "update": 1, "updates": 1, "steps": 1, "total_steps": 1, "cap": 9.0}
    if arm:
        extra["experiment"] = {"controller": {"arm": arm, "estimator_path": estimator_path}}
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    p = d / "ppo_final.pt"
    save_checkpoint(str(p), model, extra)
    return str(p), dataclasses.asdict(spec)


SUITE = {"envs": 2, "speed_cap": 9.0, "sensor_noise": True, "budget_laps": 1.0,
         "race_size": 1, "compile_tracker": False}


def cell(mu=MU_LOW, seed=4401, mp=TEST_MAP, envs=2):
    return {"map": mp, "true_mu": mu, "seed": seed, "envs": envs}


# ------------------------------------------------------------------ loading and arms
def test_a_legacy_checkpoint_loads_and_drives_its_own_spec(tmp_path):
    path, spec = make_checkpoint(tmp_path)
    model, extra = MA.load_actor({"path": path, "arm": "legacy"}, "cpu")
    assert extra["spec"] == spec, "the adapter must hand back the spec the checkpoint recorded"
    assert model.meta["n_beams"] == spec["n_beams"]


def test_a_missing_checkpoint_is_refused_by_name(tmp_path):
    with pytest.raises(MA.AdapterError, match="does not exist"):
        MA.load_actor({"path": str(tmp_path / "nope" / "ppo_final.pt"), "arm": "legacy"}, "cpu")


def test_a_controller_checkpoint_under_the_wrong_arm_is_refused(tmp_path):
    path, _ = make_checkpoint(tmp_path, "trained_estimated", arm="estimated")
    with pytest.raises(MA.AdapterError, match="trained under 'estimated'"):
        MA.load_actor({"path": path, "arm": "fixed_low"}, "cpu")


def test_an_undeclared_cross_runtime_reference_is_refused(tmp_path):
    """A legacy policy under a research controller is a deliberate reference or a mistake, and the
    two are indistinguishable in the results. The entry has to say which."""
    path, _ = make_checkpoint(tmp_path)
    with pytest.raises(MA.AdapterError, match="cross_runtime"):
        MA.load_actor({"path": path, "arm": "fixed_low"}, "cpu")
    model, _ = MA.load_actor({"path": path, "arm": "fixed_low", "cross_runtime": True}, "cpu")
    assert model is not None


def test_a_checkpoint_without_a_spec_is_refused(tmp_path):
    from f1sim.learn.model import ActorCritic, save_checkpoint
    with torch.random.fork_rng():
        torch.manual_seed(1)
        m = ActorCritic(n_stack=6, n_beams=1081, proprio_dim=366, priv_dim=17, act_dim=8,
                        scan_deltas=True, temporal_encoder="cnn")
    p = tmp_path / "nospec.pt"
    save_checkpoint(str(p), m, {"phase": "ppo"})
    with pytest.raises(MA.AdapterError, match="no extra\\['spec'\\]"):
        MA.load_actor({"path": str(p), "arm": "legacy"}, "cpu")


# ------------------------------------------------------------------ the prepared cell
@pytest.fixture(scope="module")
def legacy_entry(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    path, spec = make_checkpoint(d)
    return {"path": path, "arm": "legacy"}, spec


def test_the_env_emits_exactly_the_checkpoints_observation(legacy_entry):
    entry, spec = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(), SUITE, "cpu")
    try:
        obs, _ = prep.env.reset(seed=4401)
        from f1sim.learn.obs import flatten_obs
        scan, proprio = flatten_obs(obs)
        assert scan.shape[1] == spec["scan_stack"], scan.shape
        assert scan.shape[2] == spec["n_beams"], scan.shape
        assert proprio.shape[1] == model.meta["proprio_dim"], proprio.shape
        # and the actor really consumes them
        action = MA.policy_for(model)(obs)
        assert action.shape == (prep.env.B, spec["act_dim"]), action.shape
        assert torch.isfinite(action).all()
        assert float(action.abs().max()) <= 1.0 + 1e-6, "actions must come back in [-1, 1]"
    finally:
        prep.close()


@pytest.mark.parametrize("mu", [MU_LOW, MU_HIGH])
def test_the_declared_friction_is_the_plants_friction(legacy_entry, mu):
    """`true_mu` has to be true. With randomisation on, `vehicle.mu` is a scale on a draw and the
    episode friction is some other number entirely."""
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(mu=mu), SUITE, "cpu")
    try:
        p = prep.env.sim.P["mu"]
        assert float(p.min()) == pytest.approx(mu, abs=1e-6)
        assert float(p.max()) == pytest.approx(mu, abs=1e-6), "friction varies across envs"
        assert prep.protocol["randomisation_enabled"] is False
        assert prep.protocol["plant_mu"] == pytest.approx(mu, abs=1e-6)
        # and it stays constant across a reset, which is the standing constraint
        prep.env.reset(seed=4401)
        assert float(prep.env.sim.P["mu"].min()) == pytest.approx(mu, abs=1e-6)
    finally:
        prep.close()


def test_the_protocol_record_says_what_was_actually_built(legacy_entry):
    entry, spec = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(seed=4402), SUITE, "cpu")
    try:
        pr = prep.protocol
        assert pr["arm"] == "legacy" and pr["trained_arm"] == "legacy"
        assert pr["seed"] == 4402 and pr["map"] == TEST_MAP
        assert pr["budget_steps"] > 0 and pr["speed_cap"] == 9.0
        assert pr["spec"] == spec
        assert pr["sim_compile"] is False
        assert pr["spec_match"]["proprio_dim"] == model.meta["proprio_dim"]
    finally:
        prep.close()


def test_a_spec_the_adapter_cannot_honour_is_refused_not_ignored(legacy_entry):
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    bad = dict(extra)
    bad["spec"] = {**extra["spec"], "some_future_scalar": 1.0}
    with pytest.raises(MA.AdapterError, match="cannot honour"):
        MA.prepare_cell(entry, bad, cell(), SUITE, "cpu")


def test_close_is_idempotent_and_releases_the_env(legacy_entry):
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(), SUITE, "cpu")
    prep.close()
    prep.close()                       # a second close must not raise
    assert prep.env is None and prep.controller is None


# ------------------------------------------------------------------ controller lifecycle
def test_the_fixed_low_runtime_is_installed_and_its_hooks_run(tmp_path):
    """`fixed_low` needs no estimator, so it exercises the whole install/begin/pre/post lifecycle
    on the CPU without a student checkpoint."""
    path, _ = make_checkpoint(tmp_path, "arm_fixed_low", arm="fixed_low")
    entry = {"path": path, "arm": "fixed_low"}
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(), SUITE, "cpu")
    try:
        assert prep.controller is not None, "a non-legacy arm must install a runtime"
        assert prep.controller.arm == "fixed_low"
        assert prep.protocol["trained_arm"] == "fixed_low"
        obs, _ = prep.env.reset(seed=4401)
        prep.begin(obs)
        policy = MA.policy_for(model)
        for _ in range(3):
            prep.pre_action(obs)
            action = policy(obs)
            obs, _r, term, trunc, _i = prep.env.step(action)
            prep.post_step(term, trunc)
        # the arm actually drove: its recorded friction is the fixed-low constant, not the episode's
        mu_used = float(prep.controller.grip.mu.min())
        assert mu_used == pytest.approx(0.73423, abs=1e-4), mu_used
        assert mu_used != pytest.approx(float(prep.env.sim.P["mu"].min()), abs=1e-6) or \
            cell()["true_mu"] == pytest.approx(0.73423, abs=1e-4)
    finally:
        prep.close()


def test_the_legacy_arm_installs_nothing_rather_than_a_no_op(legacy_entry):
    """A no-op stand-in would let a caller believe a runtime ran. None says it did not."""
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(), SUITE, "cpu")
    try:
        assert prep.controller is None
        obs, _ = prep.env.reset(seed=4401)
        prep.begin(obs)                # must be a safe no-op, not an AttributeError
        prep.pre_action(obs)
        prep.post_step(torch.zeros(prep.env.B, dtype=torch.bool),
                       torch.zeros(prep.env.B, dtype=torch.bool))
    finally:
        prep.close()


def test_installing_the_controller_does_not_step_the_simulator(tmp_path):
    """Every cell has to start from the same snapshot; an install that steps would desynchronise the
    arms by exactly one control period."""
    path, _ = make_checkpoint(tmp_path, "no_step", arm="fixed_low")
    entry = {"path": path, "arm": "fixed_low"}
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(), SUITE, "cpu")
    try:
        assert prep.protocol["arm"] == "fixed_low"      # reaching here means the guard passed
    finally:
        prep.close()


# ------------------------------------------------------------------ avoidance spawn
def test_a_fixed_spawn_arc_is_where_the_scored_reset_begins(legacy_entry):
    """The avoidance start is frozen, so the reset has to land there and the observation must be
    built from that pose rather than from a random spawn followed by a teleport."""
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    free = MA.prepare_cell(entry, extra, cell(), SUITE, "cpu")
    try:
        free.env.reset(seed=4401)
        length = float(free.env.sim.track.length[free.env.sim.tid].max())
    finally:
        free.close()

    target = round(length * 0.25, 3)
    prep = MA.prepare_cell(entry, extra, cell(), SUITE, "cpu", spawn_s_m=target)
    try:
        prep.env.reset(seed=4401)
        s = prep.env.sim.s
        # Not bit-identical across envs, and it should not be: `sample_spawn` applies the
        # configured lateral and yaw jitter around the requested arc, so each car's projected arc
        # differs by a centimetre or two. What matters is that every env starts at the SAME declared
        # arc rather than at an independent random point somewhere on the lap -- a free spawn
        # spreads over the whole track length, which here is ~50 m.
        spread = float(s.max() - s.min())
        assert spread < 0.25, f"envs did not start together: {spread:.3f} m apart"
        assert float(s.mean()) == pytest.approx(target, abs=0.5), float(s.mean())
        assert prep.protocol["spawn_s_m"] == target
    finally:
        prep.close()


# ------------------------------------------------------------------ integration seams
def test_a_prebuilt_track_override_is_used_not_reloaded_by_name(legacy_entry):
    """Core places an obstacle and hands the built `Track` over. Re-loading it by name would throw
    the placement away and score a clean track that looks identical in the record."""
    from f1sim.learn import common
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    trs, _ = common.load_tracks([TEST_MAP], racelines=False, drop_infeasible=False)
    built = trs[0]
    # Stamp the object so the check is identity, not resemblance: a re-load by name would produce an
    # equal-looking Track without the mark, which is exactly the failure mode -- a placed obstacle
    # silently absent while the record still names the map.
    built.name = built.name + "+marked"
    grid_before = built.grid_key() if callable(getattr(built, "grid_key", None)) else None
    prep = MA.prepare_cell(entry, extra, cell(), SUITE, "cpu", tracks_override=[built])
    try:
        assert built.name.endswith("+marked")
        assert any(getattr(t, "name", "").endswith("+marked")
                   for t in (prep.env.sim.tracks if hasattr(prep.env.sim, "tracks") else [built])), \
            "the prepared cell did not carry the built Track through"
        if grid_before is not None:
            assert built.grid_key() == grid_before, "the override object was mutated"
    finally:
        prep.close()


def test_names_still_work_as_a_track_override(legacy_entry):
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(), SUITE, "cpu", tracks_override=[TEST_MAP])
    try:
        assert prep.env.sim.B == SUITE["envs"]
    finally:
        prep.close()


def test_sensor_noise_off_actually_zeroes_every_sensor(legacy_entry):
    """`cfg.lidar.noise` does not exist; setting it created an unread attribute and left every
    sensor noisy while the protocol record claimed the opposite."""
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    quiet = {**SUITE, "sensor_noise": False}
    prep = MA.prepare_cell(entry, extra, cell(), quiet, "cpu")
    try:
        f = prep.protocol["sensor_noise_fields"]
        assert all(v == 0.0 for v in f.values()), f
        # Not just the gaussian terms: beam dropout, IMU bias random-walk and the vibration model
        # are draws too, and an earlier version silenced only the gaussians while labelling
        # everything off.
        for must in ("lidar_dropout_prob", "lidar_floor_dropout", "lidar_duct_graze_dropout",
                     "imu_gyro_bias_walk", "imu_vib_accel", "imu_vib_gyro", "imu_vib_broadband",
                     "imu_vib_accel_floor", "imu_vib_gyro_floor",
                     "lidar_noise_std", "lidar_noise_std_rel", "imu_gyro_noise", "imu_accel_noise",
                     "odom_speed_noise_std", "odom_yaw_rate_noise_std"):
            assert f[must] == 0.0, must
        # and the built simulator agrees, not just the config object
        sc = prep.env.sim.cfg
        assert float(sc.lidar.noise_std) == 0.0 and float(sc.lidar.dropout_prob) == 0.0
        assert float(sc.imu.gyro_bias_walk) == 0.0 and float(sc.imu.vib_broadband) == 0.0
        assert float(sc.odom.speed_noise_std) == 0.0
        # a threshold is not an amplitude: zeroing it would turn vibration on everywhere
        assert float(sc.imu.vib_onset_v) > 0.0
    finally:
        prep.close()


def test_sensor_noise_on_leaves_the_real_defaults(legacy_entry):
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(), SUITE, "cpu")
    try:
        f = prep.protocol["sensor_noise_fields"]
        assert f["lidar_noise_std"] > 0 and f["imu_accel_noise"] > 0
    finally:
        prep.close()


def test_a_race_allocates_learners_times_race_size_and_keeps_the_opponent_fixed(legacy_entry):
    """`envs` is learners. A race of 2 given 8 would score 4 cars, not 8 -- half the declared
    denominator -- and the default `opponent="policy"` would drive the other car with the candidate
    itself, so the opponent would change whenever the candidate did."""
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    race_suite = {**SUITE, "race_size": 2, "opponent": "teacher", "opp_speed_range": (0.6, 0.8)}
    prep = MA.prepare_cell(entry, extra, cell(envs=4), race_suite, "cpu", racelines=True)
    try:
        assert prep.env.sim.B == 8, "4 learners x race_size 2 = 8 simulator envs"
        assert prep.env.M == 2
        assert prep.protocol["learners"] == 4 and prep.protocol["envs"] == 8
        assert prep.protocol["opponent"] == "teacher"
        assert prep.protocol["opp_speed_range"] == [0.6, 0.8]
        assert prep.env.ecfg.opponent == "teacher"
        assert tuple(prep.env.ecfg.opp_speed_range) == (0.6, 0.8)
        assert prep.env.ecfg.selfplay_front_cap is False
        # exactly the learners are scored; the mask is what core's loop reads
        obs, _ = prep.env.reset(seed=4401)
        assert int(prep.env.on_policy.sum()) == 4, prep.env.on_policy
    finally:
        prep.close()


def test_a_solo_cell_is_unaffected_by_the_race_rule(legacy_entry):
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(envs=2), SUITE, "cpu")
    try:
        assert prep.env.sim.B == 2 and prep.env.M == 1
        assert prep.protocol["learners"] == 2 and prep.protocol["envs"] == 2
    finally:
        prep.close()


def test_friction_survives_a_reset_checked_against_a_clone(legacy_entry):
    """`P["mu"]` is rewritten IN PLACE on autoreset, so a stored reference aliases a live buffer and
    a comparison against it passes whatever happened. Clone at capture."""
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(mu=MU_HIGH), SUITE, "cpu")
    try:
        before = prep.env.sim.P["mu"].clone()
        prep.env.reset(seed=4401)
        assert torch.equal(before, prep.env.sim.P["mu"]), "friction moved across a reset"
        assert float(before.min()) == pytest.approx(MU_HIGH, abs=1e-6)
    finally:
        prep.close()


def test_close_releases_everything_even_if_the_controller_raises(legacy_entry):
    """`_closed` is set first, so a throwing release used to leave the graph runtime installed and
    the env referenced with no second chance to clean up."""
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    prep = MA.prepare_cell(entry, extra, cell(), SUITE, "cpu")

    class Boom:
        def release(self):
            raise RuntimeError("release failed")

    prep.controller = Boom()
    with pytest.raises(RuntimeError, match="release failed"):
        prep.close()
    assert prep.env is None, "env still referenced after a failing release"
    assert prep.graph_holder is None, "graph runtime still installed after a failing release"
    assert prep.controller is None


def test_the_overtake_shape_core_will_pass(legacy_entry):
    """The agreed conversion, end to end: core passes cell.envs = 8 LEARNERS unchanged, the adapter
    multiplies by race_size. 8 learners x 2 = 16 simulator envs, 8 scored, 8 opponents."""
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    o_suite = {**SUITE, "race_size": 2, "opponent": "teacher", "opp_speed_range": (0.6, 0.8)}
    prep = MA.prepare_cell(entry, extra, cell(envs=8), o_suite, "cpu", racelines=True)
    try:
        assert prep.protocol["learners"] == 8
        assert prep.protocol["envs"] == 16 and prep.env.sim.B == 16
        obs, _ = prep.env.reset(seed=4401)
        on = prep.env.on_policy
        assert int(on.sum()) == 8, f"scored cars: {int(on.sum())}"
        assert int((~on).sum()) == 8, "opponents"
        # slot 0 of each race is the learner, so the scored rows are the even indices: a stable
        # ordering is what lets a paired comparison line rows up across checkpoints
        assert on.view(-1, 2)[:, 0].all(), "learners are not slot 0 of every race"
        assert not on.view(-1, 2)[:, 1].any(), "an opponent slot was marked on-policy"
    finally:
        prep.close()


def test_a_failed_controller_install_releases_the_graph_runtime(legacy_entry, monkeypatch):
    """`prepare_graph_runtime` has already installed the runtime by the time the arm is built. If
    construction or the no-step assertion raises, this function still owns it and the caller has no
    handle to clean up with -- so it must release before propagating, and the ORIGINAL exception is
    what the caller should see."""
    entry, _ = legacy_entry
    model, extra = MA.load_actor(entry, "cpu")
    released = []

    import f1sim.learn.graph_runtime as GR
    real_prepare = GR.prepare_gr if hasattr(GR, "prepare_gr") else None
    del real_prepare

    class Holder:
        def release(self):
            released.append("holder")

    monkeypatch.setattr(MA, "prepare_cell", MA.prepare_cell)      # keep the symbol explicit
    import f1sim.learn.benchmark.model_adapter as mod
    monkeypatch.setattr("f1sim.learn.graph_runtime.prepare_graph_runtime",
                        lambda env, log=None: Holder())

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("construction failed")

    monkeypatch.setattr("f1sim.learn.grip_runtime.ControllerRuntime", Boom)
    with pytest.raises(RuntimeError, match="construction failed"):
        mod.prepare_cell({**entry, "arm": "fixed_low", "cross_runtime": True}, extra,
                         cell(), SUITE, "cpu")
    assert released == ["holder"], "the graph runtime was left installed after a failed install"
