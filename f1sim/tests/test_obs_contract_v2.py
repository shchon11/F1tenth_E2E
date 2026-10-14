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

The contract without the issued actions (`ObsSpecV2.ego_actions` False, the copycat study's arm A,
docs/research/copycat-2026-10-14.md) is held to the same rules: its table names each column once, its
builder's data has no duplicate column, it is the full contract's ego with the action columns taken
out (one ring, two readings), the env and the node agree on it, and a spec dict written before the
field existed still reads as the contract it was.

It is the default since 2026-10-15 (`EGO_ACTIONS_DEFAULT`), so every test here whose subject is an
issued-action column -- a_{t-2}, the history rows' plans, `push_action`, their mirror signs, the A5
duplicate -- builds the full contract by name (`ObsSpecV2(ego_actions=True)`).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.learn.obs import (ObsBuilder, ObsBuilderV2, ObsSpec, ObsSpecV2, action_columns,  # noqa: E402
                             action_mirror_signs, assemble_ego_v2, contract_table_v2, ego_columns,
                             ego_mirror_signs, flatten_obs, mirror_scan, spec_from_dict)


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
    spec = ObsSpecV2(ego_actions=True)
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
    # the full contract (its issued actions are "action[k]", never v1's "prev_action") and the default
    for spec in (ObsSpecV2(ego_actions=True), ObsSpecV2()):
        names = {c.quantity for c in ego_columns(spec)}
        assert names == {"speed", "gyro_z"} | {f"action[{k}]" for k in range(8) if spec.ego_actions}
        text = contract_table_v2(spec)
        for gone in ("speed_cap", "roll", "pitch", "accel", "gyro_x", "gyro_y", "prev_action", "ramp@"):
            assert gone not in text.replace("no beam-angle ramp", "")


def test_no_ego_column_duplicates_another_on_data_and_the_v1_layout_does():
    # the duplicate A5 found was an issued action's, so this is the contract that carries them; the one
    # without them gets the same data check in the arm-A section below
    spec = ObsSpecV2(ego_actions=True)
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
    scan, ego = torch.rand(2, 1, 16), torch.rand(2, ObsSpecV2().ego_dim)
    s, e = flatten_obs({"scan": scan, "ego": ego})
    assert s is scan and e is ego
    v1 = {"scan": scan, "speed": torch.rand(2, 1), "prev_action": torch.rand(2, 16),
          "speed_cap": torch.rand(2, 1), "imu": torch.rand(2, 6), "imu_att": torch.rand(2, 2)}
    _s, p = flatten_obs(v1)
    assert torch.equal(p, torch.cat([v1[k] for k in ("speed", "prev_action", "speed_cap", "imu", "imu_att")], 1))


def test_mirror_signs_follow_the_table():
    spec = ObsSpecV2(ego_actions=True)              # the action columns' signs are the point
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
    spec = ObsSpecV2(n_beams=32, hist_len=3, hist_stride=2, ego_actions=True)
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
        spec = ObsSpecV2(n_beams=16, hist_len=4, hist_stride=stride, ego_actions=True)
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


# ---------------------------------------------------------------- without the issued actions (arm A)
def test_the_contract_without_issued_actions_reads_only_measured_columns():
    spec = ObsSpecV2(ego_actions=False).validate()
    cols = ego_columns(spec)
    assert len(cols) == spec.ego_dim == 2 + 20 * 2 == 42
    assert [c.index for c in cols] == list(range(42))
    assert {c.quantity for c in cols} == {"speed", "gyro_z"}
    assert spec.prev_action_lags == () and action_columns(spec) == ()
    # now, then the history rows 1, 3, ..., 39 steps back -- the same second the full contract reads
    assert sorted({c.lag for c in cols}) == [0] + list(range(1, 40, 2))
    keys = [(c.quantity, c.lag) for c in cols]
    assert len(set(keys)) == len(keys), "two columns claim the same quantity at the same lag"
    assert ego_mirror_signs(spec).tolist() == [1.0, -1.0] * 21
    # the full contract's issued actions are exactly the columns this one does not have
    full = ObsSpecV2(ego_actions=True)
    assert len(action_columns(full)) == full.ego_dim - spec.ego_dim == 8 + 20 * 8
    assert all(ego_columns(full)[i].quantity.startswith("action[") for i in action_columns(full))
    text = contract_table_v2(spec)
    assert "action[" not in text and "issued actions: none" in text
    assert spec.ring_len == full.ring_len and spec.row_dim == full.row_dim, "one ring for both contracts"


def test_the_two_contracts_read_one_ring_and_agree_on_every_measured_column():
    """The same sensor values and the same issued actions: the arm-A builder's ego is the full
    builder's with the action columns taken out, at every step from the first, and no two of its
    columns carry the same numbers."""
    full, a = ObsSpecV2(ego_actions=True), ObsSpecV2(ego_actions=False)
    warm = (full.hist_len - 1) * full.hist_stride + 1
    e_full = _drive(ObsBuilderV2(full), warm + 60, full.act_dim)
    e_a = _drive(ObsBuilderV2(a), warm + 60, a.act_dim)
    dropped = set(action_columns(full))
    keep = [i for i in range(full.ego_dim) if i not in dropped]
    assert e_a.shape[1] == 42 and torch.equal(e_a, e_full[:, keep])
    assert _n_distinct_columns(e_a[warm:]) == a.ego_dim
    # the shared assembly, on a ring whose row i is i + 1 steps back: the sensor half of each row
    ring = torch.arange(5 * 10, dtype=torch.float32).reshape(1, 5, 10)
    feat = torch.tensor([[-1.0, -2.0]])
    ego = assemble_ego_v2(feat, ring, hist_len=3, hist_stride=2, actions=False)
    assert ego[0].tolist() == [-1.0, -2.0] + ring[0, [0, 2, 4], :2].reshape(-1).tolist()


def test_a_spec_dict_older_than_ego_actions_reads_as_the_contract_it_was():
    """Every v2 checkpoint before the copycat study recorded its spec without `ego_actions`, and every
    one of them observed its issued actions: read back, that is the contract, whatever the default."""
    from f1sim.learn.model_v2 import ActorCriticV2
    old = {k: v for k, v in vars(ObsSpecV2()).items() if k != "ego_actions"}
    assert ObsSpecV2.from_dict(old).ego_actions is True and spec_from_dict(old).ego_dim == 210
    a = ObsSpecV2(ego_actions=False)
    assert spec_from_dict(vars(a)) == a and spec_from_dict(vars(a)).ego_dim == 42
    for bad in ("off", 0, None):
        with pytest.raises(ValueError):
            ObsSpecV2(ego_actions=bad).validate()
    # a network rebuilt from an old meta (which records its ego_dim, 210) reads 210 columns, and its
    # meta says so from now on
    old_meta = {k: v for k, v in vars(ObsSpecV2(n_beams=61)).items() if k != "ego_actions"}
    m = ActorCriticV2(n_beams=61, ego_dim=210, obs=old_meta)
    assert m.obs_spec.ego_actions is True and m.meta["obs"]["ego_actions"] is True and m.obs_spec.ego_dim == 210
    assert m.actor.ego[0].in_features == 210
    m_a = ActorCriticV2(n_beams=61, ego_dim=42, obs=dict(vars(ObsSpecV2(n_beams=61, ego_actions=False))))
    assert m_a.obs_spec.ego_dim == 42 and m_a.actor.ego[0].in_features == 42
    # a network that does not read the contract's width is refused, either way round
    for width, actions in ((210, False), (42, True)):
        with pytest.raises(ValueError):
            ActorCriticV2(n_beams=61, ego_dim=width, obs=dict(vars(ObsSpecV2(n_beams=61, ego_actions=actions))))
    with pytest.raises(ValueError):                    # an old meta is the 210-wide contract, not the default's 42
        ActorCriticV2(n_beams=61, ego_dim=42, obs=old_meta)


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
@pytest.mark.parametrize("n_frames,ego_actions", [(1, True), (3, True), (1, False)])
def test_the_env_and_the_node_build_the_same_v2_observation(n_frames, ego_actions):
    from f1sim.learn import common
    env = _env(n_frames, ego_actions=ego_actions)
    spec = common.obs_spec(env)
    width = 210 if ego_actions else 42
    assert isinstance(spec, ObsSpecV2) and spec.n_frames == n_frames and spec.ego_dim == width
    assert spec.ego_actions is ego_actions and env.single_observation_space["ego"].shape == (width,)
    builder = ObsBuilderV2(spec)
    obs, _ = env.reset(seed=1)
    assert set(obs) == {"scan", "ego"} and obs["ego"].shape == (2, width)
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
    for bad in (dict(action_mode="direct"), dict(scan_stack=4), dict(hist_len=0), dict(obs_imu=False),
                dict(ego_actions="off")):
        with pytest.raises(ValueError):
            _env(**bad)
    with pytest.raises(ValueError):
        _env(obs_version=3)


@pytest.mark.slow
def test_a_reset_row_restarts_its_history_from_its_own_first_observation():
    env = _env(envs=2, max_steps=6, ego_actions=True)       # the issued actions' restart is half the point
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


@pytest.mark.slow
def test_without_issued_actions_a_reset_row_restarts_from_its_first_observation_too():
    env = _env(envs=2, max_steps=6, ego_actions=False)
    env.reset(seed=1)
    for t in range(6):
        obs, _r, te, tr, info = env.step(_actions(env, t))
    assert bool((te | tr).all())
    ego = obs["ego"]
    assert ego.shape == (2, 42)
    rows = ego[:, 2:].reshape(2, 20, 2)
    assert torch.equal(rows, ego[:, None, :2].expand(-1, 20, -1))
