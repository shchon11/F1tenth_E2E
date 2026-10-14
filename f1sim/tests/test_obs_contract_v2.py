"""Observation contract v2 (`learn.obs.ObsSpecV2`): the column table, the builder, the env.

The critique of the redesign (A5) found the old proprio vector carrying the same number twice --
`prev_action`'s a_{t-1} IS history row 0's action -- and asked for two things: the column table
written into the contract, and a test that no column duplicates another. The redesign's B3 also
wanted a_{t-2}, which no stride-2 history row holds: it has columns of its own, checked here to be
the action issued two steps back and to disappear where the history already holds that lag. Both are here, each in
two forms: on the table (every column names a distinct (quantity, lag)) and on data (no two columns
of a driven builder's output are equal over a run), with the v1 layout run through the same data
check to show the check catches the duplicate it was written for.

The env and the node must build the same v2 observation from the same messages; the slow test
drives the batched env and `ObsBuilderV2` side by side, as `test_obs_identity.py` does for v1.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.learn.obs import (ObsBuilder, ObsBuilderV2, ObsSpec, ObsSpecV2, action_mirror_signs,  # noqa: E402
                             assemble_ego_v2, contract_table_v2, ego_columns, ego_mirror_signs,
                             flatten_obs, mirror_scan, spec_from_dict)


def _drive(builder, steps: int, act_dim: int, seed: int = 0, v2: bool = True):
    """(T, P) of what `builder` emits for random sensor values and random issued actions."""
    rng = np.random.default_rng(seed)
    n = builder.spec.n_beams
    rows = []
    for _t in range(steps):
        ranges = rng.uniform(0.2, 9.0, n).astype(np.float32)
        if v2:
            _s, ego = builder.build(ranges, float(rng.uniform(0, 8)), float(rng.normal(0, 1)))
        else:
            _s, ego = builder.build(ranges, float(rng.uniform(0, 8)), rng.normal(0, 1, 6).astype(np.float32),
                                    rng.normal(0, 0.05, 2).astype(np.float32), 9.0)
        rows.append(ego[0])
        builder.push_action(rng.uniform(-1, 1, act_dim).astype(np.float32))
    return torch.stack(rows)


def _n_distinct_columns(x: torch.Tensor) -> int:
    return int(torch.unique(x.T.contiguous(), dim=0).shape[0])


# ---------------------------------------------------------------- the table
def test_the_column_table_names_every_column_once_and_matches_the_width():
    spec = ObsSpecV2()
    cols = ego_columns(spec)
    assert len(cols) == spec.ego_dim == 2 + 8 + 20 * (2 + 8) == 210
    assert [c.index for c in cols] == list(range(len(cols)))
    keys = [(c.quantity, c.lag) for c in cols]
    assert len(set(keys)) == len(keys), "two columns claim the same quantity at the same lag"
    assert len({c.name for c in cols}) == len(cols)
    # "now" is the only lag 0, and it is two sensor columns -- no action is observed at lag 0
    assert [c.name for c in cols if c.lag == 0] == ["speed@t", "gyro_z@t"]
    # then a_{t-2}, the one lag the stride-2 history does not hold, as its own eight columns
    assert spec.prev_action_lags == (2,)
    assert [c.name for c in cols[2:10]] == [f"action[{k}]@t-2" for k in range(8)]
    # history rows are 1, 3, ..., 39 steps back: one second at 40 Hz, stride 2
    assert sorted({c.lag for c in cols if c.lag}) == [1, 2] + list(range(3, 40, 2))
    for lag in range(1, 40, 2):
        q = [c.quantity for c in cols if c.lag == lag]
        assert q == ["speed", "gyro_z"] + [f"action[{k}]" for k in range(8)]
    assert [c.quantity for c in cols if c.lag == 2] == [f"action[{k}]" for k in range(8)]


def test_nothing_v1_carried_and_v2_dropped_is_in_the_table():
    names = {c.quantity for c in ego_columns(ObsSpecV2())}
    assert names == {"speed", "gyro_z"} | {f"action[{k}]" for k in range(8)}
    text = contract_table_v2(ObsSpecV2())
    for gone in ("speed_cap", "roll", "pitch", "accel", "gyro_x", "gyro_y", "prev_action", "ramp@"):
        assert gone not in text.replace("no beam-angle ramp", "")


def test_no_ego_column_duplicates_another_on_data_and_the_v1_layout_does():
    spec = ObsSpecV2()
    warm = (spec.hist_len - 1) * spec.hist_stride + 1
    ego = _drive(ObsBuilderV2(spec), warm + 60, spec.act_dim)[warm:]
    assert _n_distinct_columns(ego) == spec.ego_dim, "two ego columns carried the same numbers"
    # The same check on v1's layout finds exactly the duplicate A5 named: prev_action's a_{t-1}
    # (columns 1..8) is history row 0's action (the last 8 of the first 17-wide row).
    v1 = ObsSpec(n_beams=64, scan_stack=1, act_dim=8, action_history=2, hist_len=20, hist_stride=2)
    pro = _drive(ObsBuilder(v1), warm + 60, 8, v2=False)[warm:]
    assert _n_distinct_columns(pro) < v1.proprio_dim
    first_row = 1 + 16 + 1 + 6 + 2
    assert torch.equal(pro[:, 1:9], pro[:, first_row + 9:first_row + 17])


def test_spec_dicts_round_trip_and_a_dict_without_version_is_v1():
    spec = ObsSpecV2(n_frames=3, n_beams=541)
    back = spec_from_dict(vars(spec))
    assert isinstance(back, ObsSpecV2) and back == spec
    v1 = spec_from_dict({"n_beams": 64, "scan_stack": 2, "act_dim": 8})
    assert isinstance(v1, ObsSpec) and v1.scan_stack == 2


@pytest.mark.parametrize("bad", [dict(n_frames=0), dict(n_frames=4), dict(hist_len=0),
                                 dict(act_dim=5), dict(speed_mode="linear", act_dim=9),
                                 dict(hist_stride=0), dict(version=1)])
def test_the_spec_refuses_what_the_contract_does_not_allow(bad):
    with pytest.raises(ValueError):
        ObsSpecV2(**bad).validate()


def test_flatten_obs_passes_a_v2_observation_through_and_leaves_v1_alone():
    scan, ego = torch.rand(2, 1, 16), torch.rand(2, 210)
    s, e = flatten_obs({"scan": scan, "ego": ego})
    assert s is scan and e is ego
    v1 = {"scan": scan, "speed": torch.rand(2, 1), "prev_action": torch.rand(2, 16),
          "speed_cap": torch.rand(2, 1), "imu": torch.rand(2, 6), "imu_att": torch.rand(2, 2)}
    _s, p = flatten_obs(v1)
    assert torch.equal(p, torch.cat([v1[k] for k in ("speed", "prev_action", "speed_cap", "imu", "imu_att")], 1))


def test_mirror_signs_follow_the_table():
    spec = ObsSpecV2()
    sig = ego_mirror_signs(spec)
    for c in ego_columns(spec):
        want = -1.0 if c.quantity == "gyro_z" or c.quantity in {f"action[{k}]" for k in range(4)} else 1.0
        assert float(sig[c.index]) == want, c
    assert action_mirror_signs("wp4").tolist() == [-1.0] * 4 + [1.0] * 4
    assert action_mirror_signs("linear").tolist() == [-1.0] * 6 + [1.0] * 2
    x = torch.rand(3, 1, 1081)
    assert torch.equal(mirror_scan(mirror_scan(x)), x)
    assert torch.equal(mirror_scan(x)[..., 0], x[..., -1])


def test_the_builder_fills_its_history_from_the_first_observation_and_pushes_issued_actions():
    spec = ObsSpecV2(n_beams=32, hist_len=3, hist_stride=2)
    b = ObsBuilderV2(spec)
    _s, e0 = b.build(np.full(32, 5.0, np.float32), 4.0, 1.0)
    assert e0.shape == (1, 2 + 8 + 3 * 10)
    assert e0[0, :2].tolist() == pytest.approx([0.4, 0.2])
    assert e0[0, 2:10].abs().sum() == 0                           # a_{t-2}: nothing issued yet
    assert e0[0, 10:12].tolist() == pytest.approx([0.4, 0.2]) and e0[0, 12:20].abs().sum() == 0
    b.push_action(np.full(8, 2.0, np.float32))                    # out of range: stored clamped
    _s, e1 = b.build(np.full(32, 5.0, np.float32), 5.0, -0.5)
    assert e1[0, :2].tolist() == pytest.approx([0.5, -0.1])
    assert e1[0, 2:10].abs().sum() == 0                           # two steps back is the fill row
    assert e1[0, 10:12].tolist() == pytest.approx([0.4, 0.2])    # row 0 = one step back
    assert e1[0, 12:20].tolist() == [1.0] * 8
    b.push_action(np.full(8, -0.25, np.float32))
    _s, e2 = b.build(np.full(32, 5.0, np.float32), 6.0, 0.0)
    assert e2[0, 2:10].tolist() == [1.0] * 8                      # a_{t-2}: the first action
    assert e2[0, 12:20].tolist() == [-0.25] * 8                   # a_{t-1}: history row 0's


def test_a_t2_is_the_action_two_steps_back_and_is_not_repeated_where_the_history_holds_it():
    """The extra columns hold exactly the action issued two steps before the observation; a
    stride-1 history holds lag 2 in its row 1, and then there are no extra columns at all (A5)."""
    for stride, extra in ((2, (2,)), (3, (2,)), (1, ())):
        spec = ObsSpecV2(n_beams=16, hist_len=4, hist_stride=stride)
        assert spec.prev_action_lags == extra
        assert spec.ego_dim == 2 + 8 * len(extra) + 4 * 10 == len(ego_columns(spec))
        b = ObsBuilderV2(spec)
        issued = []
        for t in range(12):
            _s, ego = b.build(np.full(16, 3.0, np.float32), 1.0 + 0.1 * t, 0.01 * t)
            if extra and t >= 2:
                assert torch.equal(ego[0, 2:10], issued[t - 2]), (stride, t)
            a = torch.full((8,), 0.05 * (t + 1) - 0.3)
            issued.append(a)
            b.push_action(a.numpy())
        keys = [(c.quantity, c.lag) for c in ego_columns(spec)]
        assert len(set(keys)) == len(keys)
    # the assembly the env and the builder share reads the ring by lag: row i is i + 1 steps back
    ring = torch.arange(5 * 10, dtype=torch.float32).reshape(1, 5, 10)
    feat = torch.tensor([[-1.0, -2.0]])
    ego = assemble_ego_v2(feat, ring, hist_len=3, hist_stride=2)
    assert ego[0, :2].tolist() == [-1.0, -2.0]
    assert torch.equal(ego[0, 2:10], ring[0, 1, 2:])              # lag 2 = ring row 1's action
    assert torch.equal(ego[0, 10:], ring[0, [0, 2, 4]].reshape(-1))   # rows at lags 1, 3, 5


# ---------------------------------------------------------------- the env (builds a simulator)
def _env(n_frames=1, race_size=1, envs=2, **kw):
    from f1sim import Config
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    tracks, _rls = common.load_tracks(["gen:competition:0"], racelines=True)
    cfg = Config(); cfg.rand.enabled = False; cfg.sim.compile = False
    args = dict(action_mode="plan", speed_mode="wp4", race_size=race_size, opponent="policy",
                hist_len=20, hist_stride=2, scan_stack=n_frames, compile_tracker=False,
                obs_version=2, collision_mode="soft")
    args.update(kw)
    return common.make_env(tracks, envs, "cpu", EnvConfig(**args), cfg=cfg, seed=1)


def _actions(env, t):
    a = torch.zeros(env.B, env.act_dim)
    a[:, 0] = 0.2 * np.sin(0.3 * t)
    a[:, 4:] = -0.4 + 0.1 * np.cos(0.2 * t)
    return a


@pytest.mark.slow
@pytest.mark.parametrize("n_frames", [1, 3])
def test_the_env_and_the_node_build_the_same_v2_observation(n_frames):
    from f1sim.learn import common
    env = _env(n_frames)
    spec = common.obs_spec(env)
    assert isinstance(spec, ObsSpecV2) and spec.n_frames == n_frames and spec.ego_dim == 210
    builder = ObsBuilderV2(spec)
    obs, _ = env.reset(seed=1)
    assert set(obs) == {"scan", "ego"} and obs["ego"].shape == (2, 210)
    s0, e0 = builder.build_message(env.message_inputs(0))
    assert torch.equal(e0, obs["ego"][0:1]), "the first observation's history fill differs"
    warm = n_frames - 1
    compared = 0
    for t in range(30):
        a = _actions(env, t)
        builder.push_action(a[0])
        obs, *_ = env.step(a)
        s_node, e_node = builder.build_message(env.message_inputs(0))
        s_env, e_env = flatten_obs({k: v[0:1] for k, v in obs.items()})
        assert torch.equal(e_node, e_env), f"step {t}: ego differs by {(e_node - e_env).abs().max():.3e}"
        if t >= warm:
            assert torch.equal(s_node, s_env), f"step {t}: scan differs"
            compared += 1
    assert compared >= 30 - warm


@pytest.mark.slow
def test_a_v2_env_refuses_what_it_cannot_observe_honestly():
    for bad in (dict(action_mode="direct"), dict(scan_stack=4), dict(hist_len=0), dict(obs_imu=False)):
        with pytest.raises(ValueError):
            _env(**bad)
    with pytest.raises(ValueError):
        _env(obs_version=3)


@pytest.mark.slow
def test_a_reset_row_restarts_its_history_from_its_own_first_observation():
    env = _env(envs=2, max_steps=6)
    env.reset(seed=1)
    for t in range(6):
        obs, _r, te, tr, info = env.step(_actions(env, t))
    done = te | tr
    assert bool(done.all()), "max_steps 6 should end both episodes on the sixth step"
    ego = obs["ego"]
    # every history row of the new episode is [its first speed, its first gyro z, zero actions],
    # and nothing was issued two steps back either
    assert ego[:, 2:10].abs().sum() == 0
    rows = ego[:, 10:].reshape(2, 20, 10)
    assert torch.equal(rows[:, :, :2], ego[:, None, :2].expand(-1, 20, -1))
    assert rows[:, :, 2:].abs().sum() == 0
    assert set(info["final_obs"]) == {"scan", "ego"}
