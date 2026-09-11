"""Controller runtime: arm selection, pre-action timing, the reset spy, and B3's flags.

The stubs implement backend's published `estimator-api.md` surface exactly. They are test doubles,
not placeholders in production: `grip_runtime` imports the real module lazily, and these pin the
call sequence and argument content that the real one will receive.
"""
from __future__ import annotations

import pytest
import torch

from f1sim import mpc
from f1sim.learn import grip_control as gc
from f1sim.learn import grip_runtime as gr

B = 4
WB, SMAX, VMAX = 0.3302, 0.4189, 10.0


# ---------------------------------------------------------------- doubles
class StubHistory:
    """Records the call sequence and everything it was handed."""
    def __init__(self, batch, device, spec=None):
        self.B, self.device, self.spec = batch, device, spec
        self.calls = []
        self.seen_obs = []
        self.count = torch.zeros(batch, dtype=torch.long)

    def reset(self, ids, current_obs):
        self.calls.append(("reset", ids.tolist()))
        self.seen_obs.append(current_obs)
        self.count[ids] = 1

    def push(self, current_obs, previous_issued_cmd, reset_mask=None):
        self.calls.append(("push", previous_issued_cmd.clone(),
                           None if reset_mask is None else reset_mask.clone()))
        self.seen_obs.append(current_obs)
        self.count += 1

    def inputs(self):
        return torch.zeros(self.B, 40, 11), torch.ones(self.B, 40, dtype=torch.bool)

    def valid_count(self):
        return self.count

    def is_warm(self):
        return self.count >= 40


class StubEstimator:
    """Returns a scripted `used_mu` and the full diagnostics dict backend documented."""
    def __init__(self, mu=0.9, warm=True, fallback_reason=gr.FALLBACK_NONE, q10=None, q90=None):
        self.mu, self.warm, self.reason = mu, warm, fallback_reason
        self.q10, self.q90 = q10, q90
        self.reset_filter_calls = []
        self.saw = []
        self.spec = {"n_frames": 40, "n_features": 11}
        self.feature_spec = {"n_frames": 40, "n_features": 11, "control_dt": 0.025}
        self.mu_range = (0.73423, 1.15379)
        self.calibration = {"cal_delta": 0.03}
        self.meta = {"seed": 401}                  # the run record: deliberately no weight shape
        self.arch = {"width": 32, "hidden": 64}    # the weight shape lives here, per backend
        self.sha = "deadbeef"
        self.net = torch.nn.Linear(4, 3)           # backend keeps the weights on .net, not the wrapper

    def reset_filter(self, ids):
        self.reset_filter_calls.append(ids.tolist())

    def lower_mu(self, features, valid):
        self.saw.append((features, valid))
        n = features.shape[0]
        used = torch.full((n,), float(self.mu))
        q10 = torch.full((n,), self.q10 if self.q10 is not None else self.mu)
        q90 = torch.full((n,), self.q90 if self.q90 is not None else self.mu + 0.1)
        diag = {"q10": q10, "q50": q10 + 0.05, "q90": q90, "candidate": used, "used_mu": used,
                "warm": torch.full((n,), self.warm), "finite": torch.ones(n, dtype=torch.bool),
                "fallback_reason": torch.full((n,), self.reason, dtype=torch.int8),
                "q_gap": torch.full((n,), 0.05), "excitation_pass": torch.ones(n, dtype=torch.bool),
                "excitation_long": torch.ones(n, dtype=torch.bool),
                "excitation_lat": torch.zeros(n, dtype=torch.bool)}
        return used, diag


class FakeTracker:
    def __init__(self):
        self.spec = mpc.PlanSpec()
        self.wb, self.s_max, self.v_max = WB, SMAX, VMAX
        self._solver = None


class FakeSim:
    def __init__(self, mu):
        # `P` entries are views into a buffer that randomization rewrites in place -- reproduced here
        # because the runtime's clone is what protects against it.
        self.buf = torch.zeros(B, 1)
        self.buf[:, 0] = mu
        self.P = {"mu": self.buf[:, 0]}


class FakeEnv:
    """Just the surface the runtime touches, including the auto-reset that eats `last_cmd`."""
    def __init__(self, mu=0.9):
        self.B, self.device, self.M = B, torch.device("cpu"), 1
        self.tracker = FakeTracker()
        self.sim = FakeSim(mu)
        self.last_cmd = torch.zeros(B, 2)
        self.reset_log = []

    def _reset_envs(self, ids):
        """gym_env.py:431 -- zeroes the issued command and writes a spawn speed."""
        self.reset_log.append(ids.tolist())
        if ids.numel():
            self.last_cmd[ids] = 0.0
            self.last_cmd[ids, 1] = 3.0


def make(arm, **kw):
    env = FakeEnv(**kw)
    rt = gr.ControllerRuntime(env, arm, estimator_path="stub" if arm == "estimated" else None)
    return env, rt


def install_with_stubs(rt, est=None, hist=None):
    """Install without importing backend's module: the lazy import is what makes this possible."""
    rt.grip = gc.GripMPC(rt.env.tracker, rt.gspec, B, "cpu", WB, SMAX, VMAX)
    rt.grip.install(graph=False)
    rt.spy = gr.IssuedCommandSpy(rt.env).install()
    if rt.arm == "estimated":
        rt.estimator = est or StubEstimator()
        rt.net = rt.estimator.net
        rt.history = hist or StubHistory(B, "cpu")
    rt._installed = True
    return rt


# ---------------------------------------------------------------- arms
def test_legacy_installs_nothing():
    env, rt = make("legacy")
    rt.install()
    assert env.tracker._solver is None, "the default path must not acquire a solver hook"
    assert rt.grip is None and rt.spy is None
    rt.release()


def test_arm_validation():
    env = FakeEnv()
    with pytest.raises(ValueError, match="arm must be one of"):
        gr.ControllerRuntime(env, "nonsense")
    with pytest.raises(ValueError, match="needs --estimator"):
        gr.ControllerRuntime(env, "estimated")
    with pytest.raises(ValueError, match="only meaningful for the estimated arm"):
        gr.ControllerRuntime(env, "oracle", estimator_path="x")


def test_arm_to_grip_mode():
    assert gr.arm_to_grip_mode("fixed_low") == "fixed"
    assert gr.arm_to_grip_mode("oracle") == "oracle"
    assert gr.arm_to_grip_mode("estimated") == "estimated"


def test_estimated_mode_is_dynamic_and_fixed_is_not():
    g = gc.GripMPC(FakeTracker(), gc.GripSpec(mode="estimated"), B, "cpu", WB, SMAX, VMAX)
    g.update(torch.full((B,), 0.9))                       # must not raise
    assert torch.allclose(g.mu, torch.full((B,), 0.9))
    f = gc.GripMPC(FakeTracker(), gc.GripSpec(mode="fixed"), B, "cpu", WB, SMAX, VMAX)
    with pytest.raises(RuntimeError, match="oracle arm"):
        f.update(torch.full((B,), 0.9))


# ---------------------------------------------------------------- the spy
def test_spy_captures_the_command_before_auto_reset_destroys_it():
    """gym_env zeroes `last_cmd` for finished envs inside step(); reading it after is too late."""
    env = FakeEnv()
    spy = gr.IssuedCommandSpy(env).install()
    env.last_cmd = torch.tensor([[0.3, 7.0]] * B)          # what the tracker issued this step
    env._reset_envs(torch.tensor([0, 2]))                  # auto-reset, mid-step
    captured = spy.take()
    assert torch.allclose(captured, torch.tensor([[0.3, 7.0]] * B)), \
        "the spy returned post-reset commands"
    assert float(env.last_cmd[0, 1]) == 3.0, "the fake env did not actually clobber, so this proves nothing"
    spy.release()
    assert env._reset_envs.__name__ == "_reset_envs"


def test_spy_passes_through_when_no_reset_happened():
    env = FakeEnv()
    spy = gr.IssuedCommandSpy(env).install()
    env.last_cmd = torch.tensor([[0.1, 5.0]] * B)
    env._reset_envs(torch.tensor([], dtype=torch.long))    # empty reset is not a boundary
    assert torch.allclose(spy.take(), torch.tensor([[0.1, 5.0]] * B))
    spy.release()


def test_spy_release_restores_the_original_method():
    env = FakeEnv()
    orig = env._reset_envs
    spy = gr.IssuedCommandSpy(env).install()
    assert env._reset_envs is not orig
    spy.release()
    assert env._reset_envs == orig


# ---------------------------------------------------------------- timing
def test_history_is_pushed_then_cleared_for_finished_envs():
    env, rt = make("estimated")
    est, hist = StubEstimator(), StubHistory(B, "cpu")
    install_with_stubs(rt, est, hist)
    rt.begin({"speed": torch.zeros(B, 1)})
    assert hist.calls == [], "begin() must not write a row; the first pre_action pushes it once"
    rt.pre_action({"speed": torch.zeros(B, 1)})
    assert [c[0] for c in hist.calls] == ["push"], "the first observation went in twice"
    assert torch.equal(hist.calls[0][2], torch.ones(B, dtype=torch.bool)), \
        "the first push must carry a full reset mask so the buffer starts clean"
    rt.post_step(torch.tensor([True, False, False, False]), torch.zeros(B, dtype=torch.bool))
    hist.calls.clear(); est.reset_filter_calls.clear()
    rt.pre_action({"speed": torch.ones(B, 1)})
    kinds = [c[0] for c in hist.calls]
    assert kinds == ["push"], f"push(reset_mask) clears the old rows itself; got {kinds}"
    assert est.reset_filter_calls == [[0]], "the estimator's filter was not reset for the done env"
    assert torch.equal(hist.calls[0][2], torch.tensor([True, False, False, False])), \
        "push did not receive the reset mask"


def test_no_reset_means_no_clear_and_no_filter_reset():
    env, rt = make("estimated")
    est, hist = StubEstimator(), StubHistory(B, "cpu")
    install_with_stubs(rt, est, hist)
    rt.begin({"speed": torch.zeros(B, 1)})
    rt.pre_action({"speed": torch.zeros(B, 1)})
    rt.post_step(torch.zeros(B, dtype=torch.bool), torch.zeros(B, dtype=torch.bool))
    hist.calls.clear(); est.reset_filter_calls.clear()
    rt.pre_action({"speed": torch.ones(B, 1)})
    assert [c[0] for c in hist.calls] == ["push"]
    assert est.reset_filter_calls == [], "no episode ended, so no filter reset"


def test_the_issued_command_reaches_the_history_raw():
    """Backend normalises inside push, so it must receive physical units, not pre-divided ones."""
    env, rt = make("estimated")
    hist = StubHistory(B, "cpu")
    install_with_stubs(rt, StubEstimator(), hist)
    rt.begin({"speed": torch.zeros(B, 1)})
    env.last_cmd = torch.tensor([[0.4189, 9.0]] * B)       # full lock, 9 m/s
    rt.post_step(torch.zeros(B, dtype=torch.bool), torch.zeros(B, dtype=torch.bool))
    hist.calls.clear()
    rt.pre_action({"speed": torch.ones(B, 1)})
    got = hist.calls[0][1]
    assert torch.allclose(got, torch.tensor([[0.4189, 9.0]] * B)), \
        "the command was normalised before handing it over, or taken from the wrong place"


def test_inference_result_reaches_the_mpc():
    env, rt = make("estimated")
    install_with_stubs(rt, StubEstimator(mu=1.05), StubHistory(B, "cpu"))
    rt.begin({"speed": torch.zeros(B, 1)})
    rt.pre_action({"speed": torch.ones(B, 1)})
    assert torch.allclose(rt.grip.mu, torch.full((B,), 1.05)), "the MPC did not get the estimate"


def test_oracle_reads_truth_and_clones_it():
    """`P` entries are views; an un-cloned read would follow a later in-place rewrite."""
    env, rt = make("oracle", mu=0.8)
    install_with_stubs(rt)
    rt.begin({})
    mu = rt.pre_action({})
    assert torch.allclose(mu, torch.full((B,), 0.8))
    env.sim.buf[:, 0] = 1.1                                # randomization rewrites in place
    assert torch.allclose(mu, torch.full((B,), 0.8)), "the runtime handed out a live alias of P"
    assert torch.allclose(rt.grip.mu, torch.full((B,), 0.8))


def test_fixed_low_does_no_per_step_work():
    env, rt = make("fixed_low")
    install_with_stubs(rt)
    rt.begin({})
    assert rt.pre_action({"speed": torch.ones(B, 1)}) is None
    assert float(rt.grip.mu[0]) == pytest.approx(gc.MU_FIXED_LOW)


# ---------------------------------------------------------------- the estimator sees only safe fields
def test_privileged_keys_are_never_read_by_the_estimated_arm():
    """The obs dict is passed through; poison it and check nothing privileged reaches inference."""
    env, rt = make("estimated")
    hist = StubHistory(B, "cpu")
    install_with_stubs(rt, StubEstimator(), hist)
    poisoned = {"speed": torch.ones(B, 1), "imu": torch.zeros(B, 6), "imu_att": torch.zeros(B, 2),
                "priv": torch.full((B, 17), 99.0), "mu": torch.full((B,), 99.0),
                "speed_cap": torch.full((B, 1), 99.0)}
    rt.begin(poisoned)
    rt.pre_action(poisoned)
    # The runtime hands the dict straight to backend's narrow accessor; what it must never do is
    # read a privileged field itself and fold it into the features or the friction.
    feats, _ = rt.estimator.saw[0]
    assert torch.isfinite(feats).all() and float(feats.abs().max()) == 0.0, \
        "the runtime injected something into the features"
    assert torch.allclose(rt.grip.mu, torch.full((B,), 0.9)), "friction came from somewhere else"


# ---------------------------------------------------------------- metrics and flags
def _run_update(rt, steps=10, truth=None):
    """One update's worth of steps. `truth` mimics PPO handing over a pre-action clone."""
    rt.begin({"speed": torch.zeros(B, 1)})
    for _ in range(steps):
        if truth is not None:
            rt.observe_truth(torch.full((B,), float(truth)))
        rt.pre_action({"speed": torch.ones(B, 1)})
        rt.post_step(torch.zeros(B, dtype=torch.bool), torch.zeros(B, dtype=torch.bool))
    return rt.collect_metrics()


def test_fallback_reason_two_means_no_fallback():
    """0 cold, 1 non-finite, 2 none. Reading `!= 0` would count every healthy step as a fallback."""
    env, rt = make("estimated")
    install_with_stubs(rt, StubEstimator(fallback_reason=gr.FALLBACK_NONE), StubHistory(B, "cpu"))
    log, _ = _run_update(rt)
    assert log["controller/fallback_frac"] == pytest.approx(0.0)
    env, rt = make("estimated")
    install_with_stubs(rt, StubEstimator(fallback_reason=0), StubHistory(B, "cpu"))   # cold
    log, _ = _run_update(rt)
    assert log["controller/fallback_frac"] == pytest.approx(1.0)


def test_metrics_are_drained_each_update():
    env, rt = make("estimated")
    install_with_stubs(rt, StubEstimator(), StubHistory(B, "cpu"))
    _run_update(rt, steps=5)
    log2, _ = rt.collect_metrics()
    assert log2 == {}, "metrics carried over into the next update"


def test_degeneracy_flag_fires_after_exactly_five_consecutive_updates():
    env, rt = make("estimated")
    # Pinned at the floor and always falling back: both degeneracy conditions.
    install_with_stubs(rt, StubEstimator(mu=gc.MU_FIXED_LOW, fallback_reason=0), StubHistory(B, "cpu"))
    fired = []
    for _ in range(5):
        _, f = _run_update(rt, steps=3)
        fired.append(f)
    assert fired[:4] == [[], [], [], []], "the flag fired early"
    assert fired[4] == ["degeneracy"], "the flag did not fire on the fifth consecutive update"


def test_a_healthy_update_resets_the_degeneracy_streak():
    env, rt = make("estimated")
    install_with_stubs(rt, StubEstimator(mu=gc.MU_FIXED_LOW, fallback_reason=0), StubHistory(B, "cpu"))
    for _ in range(4):
        _run_update(rt, steps=3)
    rt.estimator.mu, rt.estimator.reason = 1.05, gr.FALLBACK_NONE      # recovered
    _, f = _run_update(rt, steps=3)
    assert f == [] and rt.flags.degeneracy_run == 0


def test_overestimate_flag_needs_both_the_rate_and_the_sample_size():
    env, rt = make("estimated", mu=0.75)
    install_with_stubs(rt, StubEstimator(mu=1.10), StubHistory(B, "cpu"))   # used >> true
    # Five consecutive over-rate updates but far too few transitions: the rate streak is there, the
    # evidence is not, so nothing fires.
    for _ in range(5):
        _, f = _run_update(rt, steps=2, truth=0.75)
    assert f == [], "fired on ~40 transitions, far below the 10k minimum"
    assert rt.flags.overestimate_run == 5, "the rate streak should still be counting"
    # Same rate, now with the window carrying enough transitions.
    fired = []
    for _ in range(5):
        _, f = _run_update(rt, steps=600, truth=0.75)     # 600 * B = 2400 each, 12000 over five
        fired.append(f)
    assert fired[4] == ["overestimate"], f"window evidence reached but no flag: {fired}"


def test_coverage_and_overestimate_use_truth_without_feeding_it_in():
    env, rt = make("estimated", mu=0.9)
    install_with_stubs(rt, StubEstimator(mu=0.85, q10=0.8, q90=1.0), StubHistory(B, "cpu"))
    log, _ = _run_update(rt, steps=4, truth=0.9)
    assert log["controller/coverage_q10_q90"] == pytest.approx(1.0), "0.9 lies in [0.8, 1.0]"
    assert log["controller/overestimate_frac"] == pytest.approx(0.0), "0.85 < 0.9 + 0.02"
    assert rt.estimator.saw, "inference never ran"


def test_estimated_inference_never_touches_P():
    """The onboard path must run on an env with no privileged state at all."""
    env, rt = make("estimated")
    install_with_stubs(rt, StubEstimator(), StubHistory(B, "cpu"))
    del env.sim.P["mu"]                            # no truth anywhere
    log, _ = _run_update(rt, steps=3)              # no observe_truth call either
    assert log["controller/mu_used_mean"] == pytest.approx(0.9)
    assert "controller/coverage_q10_q90" not in log, "coverage claimed without any truth"
    with pytest.raises(RuntimeError, match="must not read P"):
        rt._truth()


def test_realised_brake_bound_is_the_solver_minimum_not_the_straight_line_budget():
    """On a curved plan the path minimum is strictly below the rho=0 budget; log the former."""
    env, rt = make("fixed_low")
    install_with_stubs(rt)
    rt.begin({})
    rt.grip.last_bounds[:, 0] = 1.5                # what a curved plan actually left
    rt.post_step(torch.zeros(B, dtype=torch.bool), torch.zeros(B, dtype=torch.bool))
    log, _ = rt.collect_metrics()
    straight = float(gc.budgets(torch.tensor([[gc.MU_FIXED_LOW]]), torch.zeros(1, 1),
                                gc.GripSpec())[2])
    assert log["controller/a_brake_realised_mean"] == pytest.approx(1.5)
    assert straight > 2.9, "sanity: the straight-line budget really is much larger"


def test_used_friction_is_logged_and_separates_the_arms():
    env, rt = make("estimated")
    install_with_stubs(rt, StubEstimator(mu=gc.MU_FIXED_LOW), StubHistory(B, "cpu"))
    lo, _ = _run_update(rt, steps=3)
    env, rt = make("estimated")
    install_with_stubs(rt, StubEstimator(mu=1.15379), StubHistory(B, "cpu"))
    hi, _ = _run_update(rt, steps=3)
    assert hi["controller/mu_used_mean"] > lo["controller/mu_used_mean"] + 0.3


# ---------------------------------------------------------------- provenance
def test_checkpoint_meta_carries_the_estimator_identity():
    env, rt = make("estimated")
    install_with_stubs(rt, StubEstimator(), StubHistory(B, "cpu"))
    m = rt.checkpoint_meta()
    assert m["estimator_state"], "the student was not embedded"
    assert m["estimator_architecture"] == {"width": 32, "hidden": 64}
    assert "architecture" not in rt.estimator.meta, \
        "meta is the run record; reading the weight shape from it embeds nothing"
    assert m["arm"] == "estimated"
    assert m["estimator"]["sha256"] == "deadbeef"
    assert m["estimator"]["feature_spec"]["n_frames"] == 40
    assert m["estimator"]["calibration"]["cal_delta"] == pytest.approx(0.03)
    assert m["grip_spec"]["mode"] == "estimated"


def test_legacy_checkpoint_meta_is_inert():
    env, rt = make("legacy")
    rt.install()
    m = rt.checkpoint_meta()
    assert m["arm"] == "legacy" and m["estimator_path"] is None and "estimator" not in m


# ---------------------------------------------------------------- checkpoint gates (model.py)
def _tiny_ckpt(tmp_path, arm="legacy", priv_dim=17, estimator_path=None):
    from f1sim.learn.model import ActorCritic, save_checkpoint
    m = ActorCritic(1, 32, 8, priv_dim, act_dim=8)
    extra = {"experiment": {"controller": {"arm": arm, "estimator_path": estimator_path}}}
    p = tmp_path / f"{arm}.pt"
    save_checkpoint(p, m, extra)
    return p


def test_controller_arm_of_defaults_to_legacy_for_old_checkpoints(tmp_path):
    from f1sim.learn.model import controller_arm_of
    import torch as T
    p = _tiny_ckpt(tmp_path)
    assert controller_arm_of(T.load(p, weights_only=False)) == "legacy"
    assert controller_arm_of({"extra": {}}) == "legacy", "a pre-existing checkpoint is legacy"
    assert controller_arm_of({}) == "legacy"


def test_default_loader_refuses_a_controller_checkpoint(tmp_path):
    from f1sim.learn.model import load_checkpoint
    p = _tiny_ckpt(tmp_path, arm="estimated", estimator_path="/some/est.pt")
    with pytest.raises(ValueError, match="controller arm 'estimated'"):
        load_checkpoint(p, "cpu")
    m, _ = load_checkpoint(p, "cpu", allow_controller=True)      # explicit opt-in works
    assert m is not None


def test_legacy_checkpoints_still_load_without_the_flag(tmp_path):
    from f1sim.learn.model import load_checkpoint
    m, _ = load_checkpoint(_tiny_ckpt(tmp_path), "cpu")
    assert m is not None, "the default path must be unchanged for everything written before"


def test_conditioning_loader_refuses_a_controller_checkpoint_too(tmp_path):
    """The other door into a training job; leaving it open would let one in through the side."""
    from f1sim.learn.model import load_for_conditioning
    from f1sim.learn import conditioning as C
    p = _tiny_ckpt(tmp_path, arm="fixed_low")
    spec = C.spec_for("true_mu")
    with pytest.raises(ValueError, match="controller arm 'fixed_low'"):
        load_for_conditioning(p, "cpu", spec.dim, spec.to_meta())


def test_priv_adapter_reaches_the_model_not_just_the_width(tmp_path):
    """The bug: overriding priv_dim to 21 widened the critic but left no adapter, so the env's
    17-wide vector arrived unmapped."""
    from f1sim.learn.model import load_checkpoint
    from f1sim.learn import conditioning as C
    p = _tiny_ckpt(tmp_path, priv_dim=17)
    m, _ = load_checkpoint(p, "cpu", override={"priv_dim": 21},
                           priv_adapter=C.ADAPTER_ABSENT_OPPONENT)
    assert m.meta["priv_dim"] == 21
    assert m.meta.get("priv_adapter") == C.ADAPTER_ABSENT_OPPONENT, "adapter not recorded"
    assert getattr(m.critic, "priv_adapter", None) == C.ADAPTER_ABSENT_OPPONENT, \
        "the critic was widened without being told where the columns go"
    # And it actually accepts the env's narrower vector.
    out = m.critic(torch.zeros(2, 1, 32), torch.zeros(2, 8), torch.zeros(2, 17))
    assert out.shape[0] == 2 and torch.isfinite(out).all()


def test_omitting_priv_adapter_leaves_the_checkpoint_value_alone(tmp_path):
    from f1sim.learn.model import load_checkpoint
    p = _tiny_ckpt(tmp_path)
    m, _ = load_checkpoint(p, "cpu")
    assert m.meta.get("priv_adapter") in (None, ""), "a default load invented an adapter"


def test_a_missing_architecture_is_refused_not_embedded_empty():
    """Weights with no shape record cannot be rebuilt; that must fail where it happens."""
    env, rt = make("estimated")
    est = StubEstimator()
    est.arch = {}
    install_with_stubs(rt, est, StubHistory(B, "cpu"))
    with pytest.raises(RuntimeError, match="no architecture"):
        rt.checkpoint_meta()


# ---------------------------------------------------------------- live metadata and drain cadence
def test_flags_fired_is_live_not_a_startup_snapshot():
    """A snapshot taken before update 1 records `flags_fired: []` forever -- the one field whose
    entire job is to travel with the result."""
    env, rt = make("estimated")
    install_with_stubs(rt, StubEstimator(mu=gc.MU_FIXED_LOW, fallback_reason=0), StubHistory(B, "cpu"))
    at_startup = rt.checkpoint_meta()["flags_fired"]
    assert at_startup == [], "nothing has fired yet"
    for _ in range(5):
        _run_update(rt, steps=3)
    assert rt.flags.fired == ["degeneracy"], "the flag did not fire"
    assert rt.checkpoint_meta()["flags_fired"] == ["degeneracy"], \
        "checkpoint_meta re-read the startup value instead of the live one"
    assert at_startup == [], "the startup snapshot was mutated in place, which hides the difference"


def test_ppo_refreshes_controller_metadata_at_both_save_paths():
    """Both save sites must call the refreshing accessor, not the pre-training dict."""
    import inspect, re
    from f1sim.learn import ppo
    src = inspect.getsource(ppo.main)
    uses = re.findall(r'"experiment": (experiment_meta\w*)\(?\)?', src)
    assert uses, "no experiment metadata is written at all"
    assert set(uses) == {"experiment_meta_now"}, \
        f"a save path reuses the pre-training snapshot: {uses}"
    assert len(uses) >= 2, f"expected the periodic and final saves, found {len(uses)}"


def test_controller_metrics_are_drained_outside_the_log_every_branch():
    """`log_every` is a publication cadence. If the drain sits inside it, `--log-every 5` silently
    turns B3's five-consecutive-update rule into twenty-five real updates and throws away every
    intervening update's transitions."""
    import ast, inspect
    from f1sim.learn import ppo
    tree = ast.parse(inspect.getsource(ppo.main))

    def mentions(node, name):
        return any(isinstance(n, ast.Attribute) and n.attr == name for n in ast.walk(node))

    log_branches = [n for n in ast.walk(tree)
                    if isinstance(n, ast.If) and mentions(n.test, "log_every")]
    assert log_branches, "could not find the logging branch; this guard needs updating"
    for br in log_branches:
        for stmt in br.body:
            assert not mentions(stmt, "collect_metrics"), \
                "collect_metrics is inside the log_every branch: the flag rule now depends on the " \
                "logging cadence"
    assert mentions(tree, "collect_metrics"), "the drain disappeared entirely"


def test_controller_is_released_regardless_of_sim_backend():
    """The hook and the reset-spy are installed on every backend, so the release cannot be
    conditional on graphs having been captured."""
    import ast, inspect
    from f1sim.learn import ppo
    tree = ast.parse(inspect.getsource(ppo.main))
    guarded = []
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and any(isinstance(x, ast.Name) and x.id == "graph_rt"
                                         for x in ast.walk(n.test)):
            for stmt in n.body:
                if any(isinstance(c, ast.Attribute) and c.attr == "release"
                       for c in ast.walk(stmt)):
                    guarded.append(stmt)
    assert not guarded, "controller.release() sits inside a graph_rt guard; on --sim-backend " \
                        "compile the solver hook and the _reset_envs wrapper are never removed"
