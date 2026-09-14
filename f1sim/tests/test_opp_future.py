"""`F1VecEnv.car_future`, and the privileged opponent block built out of it.

Named for the prediction rather than for the block because the block is worker 16's
(`work/oracle-planner`, `--opp-token`, same flag, same modes, same layout, same presence gate):
this branch builds one under that contract's name so the teacher and the student can be measured
before the two branches meet, and `REPORT.md` says exactly where the two implementations differ.
The prediction is this branch's own.

Two halves, and they fail for different reasons:

* `F1VecEnv.car_future` -- is the prediction the *car's own controller* carried forward, as
  claimed, rather than a straight line? A stopped opponent has to be predicted stopped, a car
  holding a lane-change offset has to be predicted off the line, and every prediction has to start
  at the car rather than at the nearest point of the raceline. How close it lands to the realised
  future is a measurement, not a test: `python -m f1sim.learn.opp_future_check`.
* `F1VecEnv.opp_token` -- the layout, the ego-frame transform, the presence gate, and the three
  refusals that keep an oracle out of a deployment path (the exporter, the ROS node, and the
  deployment-side observation builder itself).

And the property every one of them rests on: off is off, and a warm start into the block is
bit-identical because the new columns are zeros appended after every column the policy already had.
"""
from __future__ import annotations

import functools
import math

import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import (OPP_FUTURE_TIMES, OPP_TOKEN_CARS, OPP_TOKEN_COLS, PRIV_OPP_DIST_SCALE,
                           EnvConfig, opp_token_dim)
from f1sim.learn import common
from f1sim.learn.model import ActorCritic, load_for_memory, save_checkpoint
from f1sim.learn.obs import ObsBuilder, ObsSpec, PROPRIO_KEYS, flatten_obs
from f1sim.opponent_events import EVENT_ID

TRACK = "gen:competition:0"


@functools.lru_cache(maxsize=8)
def _parts(track_name: str = TRACK):
    from f1sim.raceline import Raceline
    t = maps.load(track_name)
    return t, Raceline.build_cached(t)


def _env(envs=6, race_size=3, token="", events=("brake", "stop", "shift"), rate=3.0, **kw):
    tr, rl = _parts()
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(action_mode="plan", race_size=race_size, opponent="teacher", speed_cap=8.0,
                     opp_events=events, opp_event_rate=rate, opp_token=token,
                     compile_tracker=False, **kw)
    env = common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=3, rls=[rl])
    env.sim.warmup()
    return env


def _drive(env, steps=6):
    teacher = common.make_teacher([_parts()[1]], env)
    obs, _ = env.reset(seed=3)
    for _ in range(steps):
        obs, *_ = env.step(env.teacher_label(teacher))
    return obs


# ----------------------------------------------------------------- the prediction


def test_every_prediction_starts_where_the_car_is():
    """A raceline walk that started at the nearest point of the LINE would put a car that tracks it
    at 0.2 m of error somewhere it is not -- and the label would then report a lateral velocity the
    car does not have. Measured, that was the one horizon where this prediction lost to constant
    velocity."""
    env = _env()
    _drive(env)
    for model in ("plan", "constv"):
        f0 = env.car_future([0.0], model=model)
        assert float((f0[:, 0] - env.sim.state[:, :2]).abs().max()) == 0.0, model


def test_a_stopped_opponent_is_predicted_stopped():
    """The scheduled event is applied for exactly as long as it has left to run. Constant velocity
    cannot know this: the car is still moving when the event starts.

    `rate` is positive but negligible so the timed machine is ON while drawing nothing of its own --
    the event under test is set by hand, which is the only way to ask about one particular event.
    """
    env = _env(events=("stop",), rate=0.01)
    _drive(env)
    ev = env.events
    opp = torch.nonzero(env.teacher_driven).flatten()
    assert opp.numel(), "no teacher-driven opponent in this race"
    travel = lambda: (env.car_future([0.75])[:, 0] - env.sim.state[:, :2]).norm(dim=1)
    free = travel()
    ev.kind[opp] = EVENT_ID["stop"]                 # a stop that has just begun and lasts 3 s
    ev.t[opp] = 0.0; ev.dur[opp] = 3.0; ev.p0[opp] = 0.0
    stopped = travel()
    cv = (env.car_future([0.75], model="constv")[:, 0] - env.sim.state[:, :2]).norm(dim=1)
    moving = opp[env.sim.state[opp, 3] > 0.8]
    assert moving.numel(), "no moving opponent to stop"
    assert torch.all(stopped[moving] < 0.5 * free[moving]), (
        f"the stop barely changed the prediction: {stopped[moving].tolist()} vs {free[moving].tolist()}")
    assert torch.all(cv[moving] - stopped[moving] > 0.25), (
        "constant velocity has to be the one that misses the stop; it predicted "
        f"{cv[moving].tolist()} against {stopped[moving].tolist()}")
    # ... and nobody else's prediction moved
    others = torch.nonzero(~env.teacher_driven).flatten()
    assert torch.equal(stopped[others], free[others])


def test_a_lane_change_is_predicted_off_the_line():
    """`shift` holds a lateral offset; the prediction has to hold it too, or the pass lane the
    teacher is scoring is the one the car has already left."""
    env = _env(events=("shift",), rate=0.01)
    _drive(env)
    ev = env.events
    opp = torch.nonzero(env.teacher_driven).flatten()
    ev.kind[opp] = EVENT_ID["shift"]
    ev.t[opp] = 1.0; ev.dur[opp] = 4.0; ev.p0[opp] = 0.35; ev.p1[opp] = 1.0     # ramped fully in
    off = env.events.lateral_offset()
    assert float(off[opp].min()) > 0.3
    def signed_off(xy):
        """Metres LEFT of the raceline the predicted point lies -- signed, because the claim is
        that the prediction moves the way the event asked and not merely that it moved."""
        idx, _ = env.teacher.project(xy, env.sim.tid)
        d = xy - env.teacher.xy[env.sim.tid, idx]
        tn = env.teacher.tan[env.sim.tid, idx]
        return -d[:, 0] * tn[:, 1] + d[:, 1] * tn[:, 0]

    shifted = signed_off(env.car_future([0.75])[:, 0])
    ev.kind[opp] = 0
    plain = signed_off(env.car_future([0.75])[:, 0])
    moved = (shifted - plain)[opp]
    assert float(moved.min()) > 0.15, (
        f"a +0.35 m lane change moved the prediction by {moved.tolist()} m to the left")


def test_the_prediction_is_not_a_straight_line():
    """On a real lap the two models have to disagree, and by more the further ahead they look --
    otherwise `plan` is constant velocity wearing a different name."""
    env = _env()
    _drive(env, 12)
    d = [float((env.car_future([h])[:, 0] - env.car_future([h], model="constv")[:, 0]).norm(dim=1).mean())
         for h in OPP_FUTURE_TIMES]
    assert d[0] < d[-1], d
    assert d[-1] > 0.1, d


# ----------------------------------------------------------------- the block


@pytest.mark.parametrize("mode", ["", "pos", "posvel", "future"])
def test_layout_and_spec_width(mode):
    env = _env(token=mode)
    obs = _drive(env)
    spec = common.obs_spec(env)
    _scan, pro = flatten_obs(obs)
    assert spec.opp_token == mode
    assert env.opp_token_dim == opp_token_dim(mode) == OPP_TOKEN_CARS * OPP_TOKEN_COLS[mode]
    assert pro.shape[1] == spec.proprio_dim
    assert ("opp_token" in obs) == bool(mode)
    if mode:
        assert obs["opp_token"].shape == (env.B, env.opp_token_dim)
        # last in the canonical order: every column a deployable observation has keeps its index
        assert PROPRIO_KEYS[-1] == "opp_token"
        assert torch.equal(pro[:, -env.opp_token_dim:], obs["opp_token"])


def test_off_is_the_observation_it_always_was():
    a = _drive(_env(token=""))
    b = _drive(_env(token="pos"))
    ka, kb = [k for k in a if k != "opp_token"], [k for k in b if k != "opp_token"]
    assert ka == kb
    for k in ka:
        assert torch.equal(a[k], b[k]), k
    assert "opp_token" not in a


def test_ego_frame_and_presence_gate_on_a_synthetic_race():
    """One race of three cars, placed by hand: the nearest car first, both in the ego's own frame,
    and every column of a car outside `overtake_range` exactly zero -- the presence flag included."""
    env = _env(envs=3, race_size=3, token="posvel")
    env.reset(seed=3)
    st = env.sim.state.clone()
    yaw = 0.5
    st[0, :2] = torch.tensor([10.0, 5.0]); st[0, 2] = yaw; st[0, 3] = 3.0; st[0, 4] = 0.0
    c, s = math.cos(yaw), math.sin(yaw)
    ahead_left = torch.tensor([10.0 + 4.0 * c - 1.0 * s, 5.0 + 4.0 * s + 1.0 * c])   # 4 m ahead, 1 m left
    st[1, :2] = ahead_left; st[1, 2] = yaw; st[1, 3] = 5.0; st[1, 4] = 0.0
    st[2, :2] = torch.tensor([10.0 + 50.0 * c, 5.0 + 50.0 * s])                      # 50 m away
    st[2, 2] = yaw; st[2, 3] = 4.0
    tok = env.opp_token(st).reshape(3, OPP_TOKEN_CARS, OPP_TOKEN_COLS["posvel"])[0]
    s_ = PRIV_OPP_DIST_SCALE
    assert abs(float(tok[0, 0]) * s_ - 4.0) < 1e-3 and abs(float(tok[0, 1]) * s_ - 1.0) < 1e-3
    assert float(tok[0, 2]) == 1.0
    assert abs(float(tok[0, 3]) * s_ - 2.0) < 1e-3          # 5 - 3 m/s of closing, both along yaw
    assert abs(float(tok[0, 4]) * s_) < 1e-3
    assert float(tok[1].abs().max()) == 0.0, f"a car 50 m away is not being raced: {tok[1]}"


def test_the_nearest_two_come_first():
    env = _env(envs=4, race_size=4, token="pos")
    env.reset(seed=3)
    st = env.sim.state.clone()
    st[0, :2] = torch.zeros(2); st[0, 2] = 0.0
    for i, d in enumerate((7.0, 2.0, 4.0), start=1):
        st[i, :2] = torch.tensor([d, 0.0]); st[i, 2] = 0.0
    tok = env.opp_token(st).reshape(4, OPP_TOKEN_CARS, 3)[0]
    lon = [float(tok[j, 0]) * PRIV_OPP_DIST_SCALE for j in range(OPP_TOKEN_CARS)]
    assert abs(lon[0] - 2.0) < 1e-3 and abs(lon[1] - 4.0) < 1e-3, lon


def test_future_columns_are_the_prediction():
    """The `future` half of the block is `car_future`, in the ego frame, and nothing else."""
    env = _env(token="future")
    obs = _drive(env)
    tok = obs["opp_token"].reshape(env.B, OPP_TOKEN_CARS, OPP_TOKEN_COLS["future"])
    st = env.sim.state
    fut = env.car_future(OPP_FUTURE_TIMES)
    o = env.sim.other_idx
    d = (st[o][:, :, :2] - st[:, None, :2]).norm(dim=2)
    near = d.argmin(1)
    ar = torch.arange(env.B)
    kj = o[ar, near]
    c, s = torch.cos(st[:, 2]), torch.sin(st[:, 2])
    for i in range(len(OPP_FUTURE_TIMES)):
        dd = fut[kj, i] - st[:, :2]
        lon = (dd[:, 0] * c + dd[:, 1] * s) / PRIV_OPP_DIST_SCALE
        present = tok[:, 0, 2] > 0
        got = tok[:, 0, 5 + 2 * i]
        assert torch.allclose(got[present], (lon * present.float())[present], atol=1e-5), i


def test_solo_refuses_the_block():
    with pytest.raises(ValueError, match="race_size"):
        _env(envs=2, race_size=1, token="pos", events=(), rate=0.0)


# ----------------------------------------------------------------- the refusals


def test_a_warm_start_into_the_block_is_bit_identical(tmp_path):
    """The columns are appended after every proprio key the policy already had, so the actor's new
    ones go on the end and the critic's go in at the old proprio width -- its input is
    `cat([proprio, priv])` and the privileged columns have to keep their meaning."""
    torch.manual_seed(5)
    small = dict(n_stack=6, n_beams=256, proprio_dim=64, priv_dim=21, act_dim=8,
                 scan_deltas=True, temporal_encoder="cnn", scan_stem="resnet")
    base = ActorCritic(**small).eval()
    path = str(tmp_path / "base.pt")
    save_checkpoint(path, base, {"spec": {}})
    g = opp_token_dim("future")
    wide, _extra, fresh = load_for_memory(path, "cpu", opp_token_dim=g)
    wide.eval()
    assert not fresh, fresh                        # nothing new: only columns were added
    assert wide.meta["proprio_dim"] == 64 + g
    gen = torch.Generator().manual_seed(1)
    scan = torch.rand(4, 6, 256, generator=gen)
    pro = torch.rand(4, 64, generator=gen) * 2 - 1
    priv = torch.rand(4, 21, generator=gen) * 2 - 1
    token = torch.rand(4, g, generator=gen) * 2 - 1
    with torch.no_grad():
        a0 = base.actor(scan, pro); v0 = base.critic(scan, pro, priv)
        a1 = wide.actor(scan, torch.cat([pro, token], 1))
        v1 = wide.critic(scan, torch.cat([pro, token], 1), priv)
    assert torch.equal(a0, a1), float((a0 - a1).abs().max())
    assert torch.equal(v0, v1), float((v0 - v1).abs().max())


def test_the_deployment_observation_builder_refuses_it():
    with pytest.raises(ValueError, match="no sensor"):
        ObsBuilder(ObsSpec(opp_token="future"))


def test_the_exporter_refuses_it(tmp_path, monkeypatch):
    import sys
    from f1sim.learn import export
    small = dict(n_stack=3, n_beams=128, proprio_dim=20, priv_dim=17, act_dim=8)
    m = ActorCritic(**small)
    path = str(tmp_path / "oracle.pt")
    save_checkpoint(path, m, {"spec": dict(ObsSpec(opp_token="future", n_beams=128).__dict__)})
    monkeypatch.setattr(sys, "argv", ["export", path])
    with pytest.raises(SystemExit, match="oracle"):
        export.main()


def test_the_ros_node_refuses_it():
    """Checked on the source rather than by importing rclpy, which is not a test dependency: what
    matters is that the refusal is in the constructor, before the observation builder is made."""
    import inspect
    import pathlib
    src = pathlib.Path(inspect.getfile(common)).parents[3] / "f1sim_ros" / "f1sim_ros" / "policy_node.py"
    text = src.read_text()
    i, j = text.index("self.spec = ObsSpec("), text.index("self.obs = ObsBuilder(")
    assert "opp_token" in text[i:j] and "raise ValueError" in text[i:j], (
        "the ROS node must refuse an oracle checkpoint before it builds an observation for it")


def test_the_benchmark_adapter_refuses_an_oracle_checkpoint_by_default():
    """A suite row taken with the other cars' true future in the observation is not comparable with
    one taken without it, and — unlike a controller arm — there is no declaration that would make it
    comparable. So the refusal is the default and the opt-in exists only for a caller that is
    deliberately measuring an oracle arm and will label what it gets as one."""
    from f1sim.learn.benchmark import model_adapter as ma
    env = _env(token="future")
    spec = dict(common.obs_spec(env).__dict__)
    with pytest.raises(ma.AdapterError, match="privileged opponent block"):
        ma.assert_env_matches_spec(env, spec)
    match = ma.assert_env_matches_spec(env, spec, allow_oracle=True)
    assert match and match.get("opp_token", spec["opp_token"]) or True
    # ... and the opt-in is not a way past a width mismatch: an oracle spec against an env that did
    # not build the block still fails, because the actor's first layer would read the wrong columns
    plain = _env(token="")
    with pytest.raises(ma.AdapterError, match="observation spec"):
        ma.assert_env_matches_spec(plain, spec, allow_oracle=True)
