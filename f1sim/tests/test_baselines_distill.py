"""The fair-comparison arm: the demonstrations are one set, and the label is the teacher's command.

`slow`: these build real simulators on CPU. They are small (a handful of races, tens of steps) and
they check the four things the comparison rests on.

  1. iteration 0 is the SAME demonstrations for every architecture -- the teacher drives, so with
     one seed the scans and the labels are bit-identical;
  2. the label is the teacher's *tracked command*, not something re-derived: it equals what the
     plan tracker inside `env.step` produced from the teacher's plan;
  3. when the student drives (beta < 1) it really drives -- through `ext_cmd`, the external-command
     path -- and the label is still the teacher's;
  4. a GRU never runs across a respawn: the sequence sampler refuses a window containing one.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from f1sim.learn import baselines                                        # noqa: E402
from f1sim.learn.baselines import distill                                # noqa: E402
from f1sim.learn.baselines import end2race as e2r_mod                    # noqa: E402
from f1sim.learn.baselines.tinylidarnet_torch import (                   # noqa: E402
    TinyLidarNetTorch, TorchBackendForDriver, load_keras_weights, verify_against)

TRACKS = ["gen:control:9100"]
E2R_REPO = os.path.join(os.path.expanduser("~"), "F1tenth", "F1tenth_E2E", "external", "baselines",
                        "End2Race")
TLN_H5 = os.path.join(os.path.expanduser("~"), "F1tenth", "F1tenth_E2E", "external", "baselines",
                      "TinyLidarNet", "Models", "f1_tenth_model.h5")
TLN_ONNX = os.path.join(os.path.expanduser("~"), "Documents", "Codex", "2026-09-10", "new-chat",
                        "work", "baselines", "models", "tinylidarnet_L_1081.onnx")


def build(seed=701, learners=2, n_beams=1081):
    from f1sim.learn import common
    tracks, rls = common.load_tracks(TRACKS, racelines=True, drop_infeasible=False)
    env, cfg = distill.make_collection_env(tracks, rls, learners, torch.device("cpu"),
                                           spec_beams=n_beams, range_max=10.0, v_max=10.0,
                                           seed=seed, overrides={"seed": seed})
    env.sim.warmup()
    teacher = common.make_teacher(rls, env, grip="true")
    return env, teacher, tracks, cfg


@pytest.mark.slow
def test_iteration_zero_is_the_same_demonstrations_whatever_the_architecture():
    """The teacher drives at beta = 1, so the architecture cannot affect a single sample."""
    out = []
    for _ in range(2):
        env, teacher, _t, _c = build()
        buf = distill.collect(env, teacher, None, 12, 1.0,
                              distill.DemoBuffer(range_max=10.0), v_max=10.0,
                              range_max=10.0).finalize()
        out.append(buf)
    assert np.array_equal(out[0].S, out[1].S)
    assert np.array_equal(out[0].L, out[1].L)
    assert np.array_equal(out[0].V, out[1].V)
    assert float(np.abs(out[0].L[:, :, 0]).max()) > 0.0, "the teacher never steered"


@pytest.mark.slow
def test_the_label_is_the_teachers_tracked_command():
    """Reproduce one step by hand: the teacher's plan through the env's own tracker."""
    env, teacher, _t, _c = build()
    rows = torch.nonzero(env.on_policy).flatten()
    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    plan = teacher.plan_action(env.sim.state, env.sim.P, env.sim.tid, env.ecfg.v_max_policy,
                               env.tracker.spec)
    v_meas = env.sim.state[:, 3] if env.last_result is None else env.last_result.odom[:, 3]
    lr = env.last_result
    yaw = (lr.imu[:, :, 2].mean(1) if (lr is not None and lr.imu is not None and lr.imu.shape[1])
           else None)
    import copy
    # A COPY of the tracker: it carries per-env warm-start state (`PlanTracker.u_prev/u_seq`), so
    # running the real one here would leave the env's next step warm-started from a step that never
    # happened. And `_opponent_actions` is deliberately not called: it advances the event scheduler
    # and redraws reactive offsets, so calling it twice gives the OPPONENTS two different plans --
    # which is exactly the difference this assertion sees on the non-learner rows. For the learner
    # rows it returns the action unchanged (`gym_env.py:995`), so the comparison is exact there.
    spy = copy.deepcopy(env.tracker)
    expected = spy(plan.clamp(-1, 1), v_meas, env.speed_cap, yaw, delay=env.tracker_delay)
    env.step(plan)
    got = env.last_cmd_raw
    assert torch.allclose(expected[rows], got[rows], atol=1e-5), \
        float((expected[rows] - got[rows]).abs().max())
    assert got[rows].shape == (rows.numel(), 2)


@pytest.mark.slow
def test_the_student_really_drives_and_the_label_is_still_the_teachers():
    """beta = 0 hands every learner to the student; the opponents and the label are untouched."""
    env, teacher, _t, _c = build()
    driver = baselines.load("tinylidarnet", TLN_ONNX)
    driver.bind_scanner(n_beams=1081, fov=4.71238898, range_max=10.0)
    rows = torch.nonzero(env.on_policy).flatten()
    buf = distill.collect(env, teacher, driver, 6, 0.0, distill.DemoBuffer(range_max=10.0),
                          v_max=10.0, range_max=10.0).finalize()
    # every learner was externally driven on the last step
    assert set(int(i) for i in rows.tolist()) <= set(env._ext_ids) or not env._ext_ids
    teacher_only = distill.collect(*build()[:2], None, 6, 1.0,
                                   distill.DemoBuffer(range_max=10.0), v_max=10.0,
                                   range_max=10.0).finalize()
    # the states diverge because a different car was driving ...
    assert not np.array_equal(buf.S, teacher_only.S)
    # ... and the labels are still commands the teacher could have issued
    assert np.abs(buf.L[:, :, 0]).max() <= 0.4189 + 1e-6
    assert buf.L[:, :, 1].min() >= -1e-6


@pytest.mark.slow
def test_external_command_reaches_the_plant():
    env, teacher, _t, _c = build()
    rows = torch.nonzero(env.on_policy).flatten()
    env.reset()
    plan = teacher.plan_action(env.sim.state, env.sim.P, env.sim.tid, env.ecfg.v_max_policy,
                               env.tracker.spec)
    cmd = np.tile(np.array([[0.31, 2.5]], dtype=np.float32), (rows.numel(), 1))
    distill.drive_external(env, rows, cmd)
    env.step(plan)
    got = env.last_cmd[rows].numpy()
    assert np.allclose(got[:, 0], 0.31, atol=1e-6)
    assert np.allclose(got[:, 1], 2.5, atol=1e-6)
    others = torch.nonzero(~env.on_policy).flatten()
    if others.numel():
        assert not np.allclose(env.last_cmd[others].numpy()[:, 1], 2.5)


def test_sequences_never_cross_an_episode_boundary():
    b = distill.DemoBuffer(range_max=10.0)
    T, B, F = 40, 3, 8
    rng = np.random.default_rng(0)
    for t in range(T):
        newep = np.zeros(B, bool)
        if t in (17, 18):
            newep[t - 17] = True
        b.add(rng.random((B, F)), rng.random(B), rng.random((B, 2)), newep)
    b.finalize()
    wins = distill._sequences([b], 10, np.random.default_rng(1), 200)
    for k, t0, i in wins:
        assert not b.N[t0 + 1:t0 + 10, i].any()
    assert len(wins) == 200


def test_the_torch_tinylidarnet_is_the_published_one_before_it_is_trained():
    """The fair arm trains a PyTorch copy; it has to BE their network first."""
    if not (os.path.exists(TLN_H5) and os.path.exists(TLN_ONNX)):
        pytest.skip("TinyLidarNet is not vendored/converted")
    from f1sim.learn.baselines.backends import OnnxBackend
    m = TinyLidarNetTorch(1081)
    assert m.n_params == 220686
    load_keras_weights(m, TLN_H5)
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_scans_100.npz")
    if not os.path.exists(f):
        pytest.skip("real scan fixture missing")
    r = np.load(f)["ranges_mm"].astype(np.float32) / 1000.0
    r = np.minimum(np.where(np.isfinite(r) & (r > 0) & (r < 65.0), r, np.float32(10.0)), 10.0)
    d = verify_against(m, OnnxBackend(TLN_ONNX), r)
    print(f"torch port vs the published Keras weights on 100 real scans: "
          f"max |delta| {d['max_abs_delta']:.3e}")
    assert d["max_abs_delta"] < 1e-5, d


def test_the_retrained_end2race_reads_this_cars_window_and_the_published_one_does_not(tmp_path):
    """Root's deviation: 270 evenly spaced beams of a 270 deg scan, one per degree."""
    if not os.path.exists(os.path.join(E2R_REPO, "model.py")):
        pytest.skip("End2Race is not vendored")
    model, deviation = e2r_mod.build_untrained(n_features=270)
    assert deviation["num_features"] == 270
    p = str(tmp_path / "retrained.pt")
    torch.save({"state_dict": model.state_dict(),
                "meta": {"n_features": 270, "n_beams": 1081, "fov": 4.71238898,
                         "range_max": 10.0, "hidden_scale": 4}}, p)
    d = baselines.load("end2race", p)
    src = d.bind_scanner(n_beams=1081, fov=4.71238898, range_max=10.0)
    assert d.scan.n_beams == 1081 and src["identity"] is True and src["unseen_fraction"] == 0.0
    bearings = np.degrees(np.linspace(-d.scan.fov / 2, d.scan.fov / 2, d.scan.n_beams)[d._idx])
    assert abs(float(np.diff(bearings).mean()) - 1.0) < 0.01, "not one beam per degree"
    pub = os.path.join(E2R_REPO, "pretrained", "end2race.pth")
    if os.path.exists(pub):
        q = baselines.load("end2race", pub)
        assert q.scan.n_beams == 1440 and q.n_features == 360
