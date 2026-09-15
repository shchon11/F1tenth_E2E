"""The privileged opponent block: what it says, when it says nothing, and who refuses it.

`f1sim/opp_token.py` is an ORACLE -- the simulator's own description of the other cars, handed to
the planner as extra proprio columns. The experiment it exists for
(`docs/research/oracle-planner-2026-09-15.md`) only means anything if the block says what it claims
to say, so the geometry is pinned here on cases whose answer can be written down, and the refusals
are pinned because an oracle that leaks into a deployable checkpoint is the one failure mode that
does not announce itself.
"""
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim import Config, Track, dynamics as dyn
from f1sim.gym_env import EnvConfig, F1VecEnv, REWARD_COMPONENT_KEYS, TTC_FAR
from f1sim.opp_token import (OPP_TOKEN_CARS, OPP_TOKEN_HORIZONS_S, OPP_TOKEN_PLAN_SOURCE,
                             OPP_TOKEN_SCALE, opp_token_car_dim, opp_token_car_keys,
                             opp_token_dim, sample_body_traj, straight_traj, to_ego, to_world)

TTC = REWARD_COMPONENT_KEYS.index("ttc")
HOLD = REWARD_COMPONENT_KEYS.index("overtake_hold")


def _env(mode="future", m=2, n=None, **kw):
    """A tiny plan-mode race with the block on. `n` defaults to one race."""
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 24
    kw.setdefault("action_mode", "plan")
    ecfg = EnvConfig(race_size=m, opponent="policy", opp_token=mode,
                     compile_tracker=False, reward_progress=0.0, **kw)
    env = F1VecEnv(Track.generate_random(2), cfg, ecfg, num_envs=n or m, device="cpu")
    env.reset(seed=1)
    return env


def _state(rows):
    """(x, y, yaw, vx[, vy]) per car -> the simulator's state tensor."""
    st = torch.zeros(len(rows), dyn.STATE_DIM)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            st[i, j if j < 3 else j] = v
    return st


def _tokens(env, rows):
    """The block for a hand-placed scene, with every car's plan the stand-in straight line.

    `_plan_fallback` is what a just-respawned car gets, and it is the one plan whose future is
    closed-form: the car holds its current velocity. That is what lets the future columns below be
    checked against arithmetic instead of against the tracker.
    """
    st = _state(rows)
    env.sim.state[:] = st
    env._plan_fallback(torch.arange(env.B), st)
    return env.opp_tokens(SimpleNamespace(state=st)), st


# ---------------------------------------------------------------- layout and the ego frame
def test_the_layout_is_a_ladder_and_each_mode_is_a_prefix_of_the_next() -> None:
    assert opp_token_dim("off") == 0
    assert [opp_token_car_dim(m) for m in ("pos", "posvel", "future")] == [3, 5, 4 + 2 * len(OPP_TOKEN_HORIZONS_S) + 1]
    for a, b in (("pos", "posvel"), ("posvel", "future")):
        ka, kb = opp_token_car_keys(a), opp_token_car_keys(b)
        assert ka[:-1] == kb[:len(ka) - 1], (ka, kb)     # every column but `present` carries over
        assert ka[-1] == kb[-1] == "present"             # ... and `present` is last in both


def test_the_block_is_the_opponent_in_the_ego_frame_not_the_world() -> None:
    """A car 2 m ahead and 1 m to the left of an ego facing +90 deg reads (2, 1) in the ego frame,
    not (0, 3) in the world. Pinned on a yaw that is not zero, because that is the whole content of
    the transform and every wrong version of it agrees at yaw 0."""
    env = _env("posvel", m=2)
    yaw = math.pi / 2
    # ego at the origin facing +y; opponent 2 m along the ego's nose and 1 m to its left
    ox, oy = -1.0, 2.0
    tok, _st = _tokens(env, [(0.0, 0.0, yaw, 3.0), (ox, oy, yaw, 5.0)])
    per = opp_token_car_dim("posvel")
    dx, dy = tok[0, 0] * OPP_TOKEN_SCALE, tok[0, 1] * OPP_TOKEN_SCALE
    assert abs(float(dx) - 2.0) < 1e-5 and abs(float(dy) - 1.0) < 1e-5, (float(dx), float(dy))
    # both cars face +y, so a 2 m/s speed difference is purely longitudinal in the ego frame
    dvx, dvy = tok[0, 2] * OPP_TOKEN_SCALE, tok[0, 3] * OPP_TOKEN_SCALE
    assert abs(float(dvx) - 2.0) < 1e-5 and abs(float(dvy)) < 1e-5, (float(dvx), float(dvy))
    assert float(tok[0, per - 1]) == 1.0
    # and the opponent's own row sees the mirror image: 2 m behind, 1 m to its right
    assert abs(float(tok[1, 0]) * OPP_TOKEN_SCALE + 2.0) < 1e-5
    assert abs(float(tok[1, 1]) * OPP_TOKEN_SCALE + 1.0) < 1e-5


def test_the_future_columns_are_the_opponents_own_plan_carried_forward() -> None:
    """With the stand-in plan (a car holding its velocity) the answer is arithmetic: at horizon h the
    opponent is h + one control step further along its own heading, because the plan's clock starts
    one control step before the observation the block goes into."""
    env = _env("future", m=2)
    yaw_o = math.pi / 4
    v = 4.0
    tok, _st = _tokens(env, [(0.0, 0.0, 0.0, 3.0), (3.0, 0.0, yaw_o, v)])
    per = opp_token_car_dim("future")
    dt = env.sim.control_dt
    for i, h in enumerate(OPP_TOKEN_HORIZONS_S):
        fx = float(tok[0, 4 + 2 * i]) * OPP_TOKEN_SCALE
        fy = float(tok[0, 5 + 2 * i]) * OPP_TOKEN_SCALE
        want_x = 3.0 + v * (h + dt) * math.cos(yaw_o)
        want_y = 0.0 + v * (h + dt) * math.sin(yaw_o)
        assert abs(fx - want_x) < 2e-3 and abs(fy - want_y) < 2e-3, (h, fx, fy, want_x, want_y)
    assert float(tok[0, per - 1]) == 1.0


def test_slots_are_ordered_nearest_first() -> None:
    env = _env("pos", m=3)
    tok, _st = _tokens(env, [(0.0, 0.0, 0.0, 3.0), (5.0, 0.0, 0.0, 3.0), (2.0, 0.0, 0.0, 3.0)])
    per = opp_token_car_dim("pos")
    near, far = float(tok[0, 0]) * OPP_TOKEN_SCALE, float(tok[0, per]) * OPP_TOKEN_SCALE
    assert abs(near - 2.0) < 1e-5 and abs(far - 5.0) < 1e-5, (near, far)


# ---------------------------------------------------------------- presence gating
def test_a_car_beyond_the_overtake_range_reads_as_absent() -> None:
    """Not "far away": absent. A stale position leaking through an unmasked path is the "trained on
    an empty road" failure the future labels were already written to avoid."""
    env = _env("future", m=2)
    per = opp_token_car_dim("future")
    inside, _ = _tokens(env, [(0.0, 0.0, 0.0, 3.0), (env.ecfg.overtake_range - 0.5, 0.0, 0.0, 3.0)])
    outside, _ = _tokens(env, [(0.0, 0.0, 0.0, 3.0), (env.ecfg.overtake_range + 0.5, 0.0, 0.0, 3.0)])
    assert float(inside[0, per - 1]) == 1.0 and float(inside[0, :per].abs().max()) > 0
    assert float(outside[0, per - 1]) == 0.0
    assert float(outside[0, :per].abs().max()) == 0.0, outside[0, :per]


def test_a_race_with_one_opponent_leaves_the_second_slot_exactly_zero() -> None:
    env = _env("future", m=2)
    per = opp_token_car_dim("future")
    tok, _ = _tokens(env, [(0.0, 0.0, 0.0, 3.0), (2.0, 0.0, 0.0, 3.0)])
    assert tok.shape[1] == OPP_TOKEN_CARS * per
    assert float(tok[0, per:].abs().max()) == 0.0


# ---------------------------------------------------------------- the sampler itself
def test_the_sampler_extends_straight_past_the_end_of_the_plan() -> None:
    """The tracker plans 0.60 s and the block asks for 0.75. Past the end the car keeps going the
    way the plan left it -- the same extension `mpc.reference` applies past the end of the path."""
    traj = torch.tensor([[[0.0, 0.0, 0.0, 2.0], [0.2, 0.0, 0.0, 2.0], [0.4, 0.0, 0.0, 2.0]]])
    got = sample_body_traj(traj, 0.1, torch.tensor([[0.15, 0.2, 0.5]]))
    assert abs(float(got[0, 0, 0]) - 0.3) < 1e-6                 # interpolated inside the grid
    assert abs(float(got[0, 1, 0]) - 0.4) < 1e-6                 # the last grid point
    assert abs(float(got[0, 2, 0]) - (0.4 + 0.3 * 2.0)) < 1e-6   # 0.3 s past it at 2 m/s


def test_the_stand_in_plan_holds_the_cars_velocity() -> None:
    st = _state([(0.0, 0.0, 0.0, 3.0)])
    traj = straight_traj(st, 4, 0.05)
    assert torch.allclose(traj[0, :, 0], torch.tensor([0.0, 0.15, 0.30, 0.45, 0.60]), atol=1e-6)
    assert float(traj[0, :, 1].abs().max()) == 0.0
    shifted = straight_traj(st, 4, 0.05, t0=0.1)
    assert abs(float(shifted[0, 0, 0]) - 0.3) < 1e-6             # t0 moves the clock, not the speed


def test_world_and_ego_transforms_are_inverses() -> None:
    pose = torch.tensor([[1.0, -2.0, 0.7]])
    pts = torch.tensor([[[0.5, 0.25], [-1.0, 3.0]]])
    assert torch.allclose(to_ego(to_world(pts, pose), pose), pts, atol=1e-6)


# ---------------------------------------------------------------- time to contact
def test_time_to_contact_on_a_head_on_a_following_and_a_diverging_pair() -> None:
    """Three cases whose answer is a division. The gap is body-to-body, so it is centre distance
    minus the half-lengths the boxes already occupy."""
    env = _env("off", m=2, reward_ttc=1.0, ttc_safe=1.0)
    e = env.ecfg
    head_on = _state([(0.0, 0.0, 0.0, 2.0), (e.car_len + 2.0, 0.0, math.pi, 2.0)])
    follow = _state([(0.0, 0.0, 0.0, 5.0), (e.car_len + 2.0, 0.0, 0.0, 4.0)])
    apart = _state([(0.0, 0.0, 0.0, 3.0), (e.car_len + 2.0, 0.0, 0.0, 5.0)])
    assert abs(float(env.car_ttc(head_on)[0]) - 0.5) < 1e-3      # 2 m of gap closing at 4 m/s
    assert abs(float(env.car_ttc(follow)[0]) - 2.0) < 1e-3       # ... at 1 m/s
    assert float(env.car_ttc(apart)[0]) == TTC_FAR               # opening: no contact is coming


def test_the_ttc_penalty_is_zero_outside_the_horizon_and_full_at_contact() -> None:
    env = _env("off", m=2, reward_ttc=1.0, ttc_safe=1.0)
    e = env.ecfg
    far = _state([(0.0, 0.0, 0.0, 5.0), (e.car_len + 4.0, 0.0, 0.0, 4.0)])          # TTC 4 s
    near = _state([(0.0, 0.0, 0.0, 5.0), (e.car_len + 0.25, 0.0, 0.0, 4.0)])        # TTC 0.25 s
    touch = _state([(0.0, 0.0, 0.0, 5.0), (e.car_len, 0.0, 0.0, 4.0)])              # TTC 0
    pen = lambda st: max(0.0, (e.ttc_safe - float(env.car_ttc(st)[0])) / e.ttc_safe)
    assert pen(far) == 0.0
    assert abs(pen(near) - 0.75) < 1e-3
    assert abs(pen(touch) - 1.0) < 1e-3


def test_the_ttc_term_is_paid_in_a_real_step_and_is_zero_when_off() -> None:
    """The term reaches the reward vector of a real step, and the flag at 0 leaves it exactly 0."""
    on = _env("off", m=2, n=2, reward_ttc=1.0, ttc_safe=2.0)
    off = _env("off", m=2, n=2)
    for env, expect_zero in ((off, True), (on, False)):
        env.sim.state[:] = _state([(0.0, 0.0, 0.0, 5.0), (env.ecfg.car_len + 0.2, 0.0, 0.0, 3.0)])
        _obs, _r, _t, _tr, info = env.step(torch.zeros(env.B, env.act_dim))
        v = float(info["reward_components"]["ttc"][0])
        assert (v == 0.0) if expect_zero else (v < 0.0), (expect_zero, v)


# ---------------------------------------------------------------- the sustained-lead bonus
def test_the_sustained_bonus_fires_once_and_only_after_the_hold_time() -> None:
    """Not on the crossing instant: a lead of 1.5 m held for 1.0 s, and then nothing more until the
    lead is lost and taken again."""
    env = _env("off", m=2, n=2, reward_overtake_hold=3.0, overtake_hold_dist=1.5,
               overtake_hold_time=1.0)
    steps = int(round(1.0 / env.sim.control_dt))
    lead = torch.full((env.B, env.M - 1), -2.0)                  # 2 m ahead of the other car
    behind0 = torch.full((env.B, env.M - 1), +2.0)
    valid = torch.ones(env.B, dtype=torch.bool)
    none = torch.zeros(env.B, dtype=torch.bool)
    ls = torch.zeros(env.B, env.M - 1); lp = torch.ones_like(ls)
    def run(gap, n):
        nonlocal ls, lp
        out = []
        for _ in range(n):
            b, ls, lp = env.overtake_hold(gap, valid, none, ls, lp)
            out.append(float(b[0]))
        return out
    assert sum(run(behind0, 1)) == 0.0            # one step behind: the lead below is now earned
    fired = run(lead, steps + 5)
    assert sum(fired) == 1.0, fired
    assert fired.index(1.0) == steps - 1, (fired.index(1.0), steps)
    # lose the lead, take it again: it pays again, and again only once
    behind = torch.full((env.B, env.M - 1), +2.0)
    assert sum(run(behind, 3)) == 0.0
    again = run(lead, steps + 5)
    assert sum(again) == 1.0 and again.index(1.0) == steps - 1, again


def test_a_lead_shorter_than_the_distance_never_starts_the_clock() -> None:
    env = _env("off", m=2, n=2, reward_overtake_hold=3.0, overtake_hold_dist=1.5,
               overtake_hold_time=1.0)
    ls = torch.zeros(env.B, env.M - 1); lp = torch.zeros_like(ls)
    just_short = torch.full((env.B, env.M - 1), -1.4)
    valid = torch.ones(env.B, dtype=torch.bool); none = torch.zeros(env.B, dtype=torch.bool)
    total = 0.0
    for _ in range(200):
        b, ls, lp = env.overtake_hold(just_short, valid, none, ls, lp)
        total += float(b[0])
    assert total == 0.0
    assert float(ls.max()) == 0.0


def test_a_reset_in_the_race_clears_the_clock() -> None:
    """An opponent that crashes respawns behind the field in place. Without this the learner would
    be paid for a lead the other car's mistake handed it."""
    env = _env("off", m=2, n=2, reward_overtake_hold=3.0, overtake_hold_dist=1.5,
               overtake_hold_time=1.0)
    steps = int(round(1.0 / env.sim.control_dt))
    lead = torch.full((env.B, env.M - 1), -2.0)
    valid = torch.ones(env.B, dtype=torch.bool)
    none = torch.zeros(env.B, dtype=torch.bool); all_ = torch.ones(env.B, dtype=torch.bool)
    ls = torch.zeros(env.B, env.M - 1); lp = torch.zeros_like(ls)
    for _ in range(steps - 2):
        _b, ls, lp = env.overtake_hold(lead, valid, none, ls, lp)
    _b, ls, lp = env.overtake_hold(lead, valid, all_, ls, lp)    # a car of the race resets
    assert float(ls.max()) == 0.0
    fired = []
    for _ in range(3):
        b, ls, lp = env.overtake_hold(lead, valid, none, ls, lp)
        fired.append(float(b[0]))
    assert sum(fired) == 0.0, "the clock restarted from where it was"


# ---------------------------------------------------------------- off is off
def test_off_allocates_nothing_and_leaves_the_observation_alone() -> None:
    env = _env("off", m=2, n=2)
    obs, _info = env.reset(seed=3)
    assert "opp_token" not in obs and env.opp_token_dim == 0
    assert not hasattr(env, "_plan_traj")
    with pytest.raises(RuntimeError):
        env.opp_tokens()
    obs, _r, _t, _tr, info = env.step(torch.zeros(env.B, env.act_dim))
    assert float(info["reward_components"]["ttc"].abs().max()) == 0.0
    assert float(info["reward_components"]["overtake_hold"].abs().max()) == 0.0


@pytest.mark.parametrize("kw,msg", [
    ({"m": 1}, "race_size"),
    ({"action_mode": "direct"}, "action_mode"),
])
def test_the_block_refuses_a_configuration_it_cannot_mean_anything_in(kw, msg) -> None:
    m = kw.pop("m", 2)
    with pytest.raises(ValueError, match=msg):
        _env("future", m=m, **kw)


# ---------------------------------------------------------------- nobody deployable may open it
def _oracle_checkpoint(tmp_path):
    from f1sim.learn.model import ActorCritic, save_checkpoint
    from f1sim.learn.obs import ObsSpec
    spec = ObsSpec(n_beams=32, scan_stack=2, act_dim=2, opp_token="posvel")
    m = ActorCritic(2, 32, spec.proprio_dim, 12, act_dim=2, opp_token="posvel")
    p = str(tmp_path / "oracle.pt")
    save_checkpoint(p, m, {"spec": spec.__dict__})
    return p, spec


def test_a_token_checkpoint_is_refused_by_default_and_openable_by_a_simulator(tmp_path) -> None:
    from f1sim.learn.model import load_checkpoint, oracle_inputs_of
    p, _spec = _oracle_checkpoint(tmp_path)
    assert oracle_inputs_of(torch.load(p, map_location="cpu")) == "posvel"
    with pytest.raises(ValueError, match="ORACLE"):
        load_checkpoint(p, "cpu")
    load_checkpoint(p, "cpu", allow_oracle=True)                 # a caller that runs the simulator


def test_the_exporter_refuses_a_token_checkpoint(tmp_path, monkeypatch) -> None:
    """`export.main` has no flag to override it: an ONNX graph whose `proprio` input includes
    columns nothing on the car can fill is a graph nobody can feed."""
    import f1sim.learn.export as export
    p, _spec = _oracle_checkpoint(tmp_path)
    monkeypatch.setattr(sys, "argv", ["export", p, "--out", str(tmp_path / "x.onnx")])
    with pytest.raises(ValueError, match="ORACLE"):
        export.main()
    assert not os.path.exists(str(tmp_path / "x.onnx"))


def test_the_deployment_observation_builder_refuses_the_spec() -> None:
    """The second, independent refusal: even a checkpoint carrying only the spec and no `meta` key
    cannot reach the wheels, because the thing that builds the car's observation says it cannot
    build this one."""
    from f1sim.learn.obs import ObsBuilder, ObsSpec
    with pytest.raises(ValueError, match="not deployable"):
        ObsBuilder(ObsSpec(n_beams=32, opp_token="future"))
    ObsBuilder(ObsSpec(n_beams=32))                              # ... and the deployable one is fine


def test_the_viewer_declines_to_open_one(tmp_path) -> None:
    from f1sim.learn.watch import default_consumer_refusal
    p, _spec = _oracle_checkpoint(tmp_path)
    why = default_consumer_refusal(p)
    assert why and "opp_token" in why, why


# ---------------------------------------------------------------- the warm start
def test_the_new_proprio_columns_are_zero_and_cannot_move_the_action(tmp_path) -> None:
    """An oracle arm starts as the policy it warm-started from: the block's own input columns are
    exactly zero, so whatever is in them contributes exactly nothing until training moves them. That
    is bit-exact and is what the assertion checks -- two different blocks, one action.

    The residual against the ORIGINAL network is not bit-exact and is not claimed to be: a linear
    layer with a wider input reassociates its sum, which moves the result by ~1e-9 on a [-1, 1]
    action. Bounded here so a real change could not hide inside it.
    """
    from f1sim.learn.model import ActorCritic, load_for_memory, save_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(0)
    P, V = 64, 12
    base = ActorCritic(2, 32, P, V, act_dim=2).eval()
    path = str(tmp_path / "base.pt"); save_checkpoint(path, base)
    scan = torch.rand(4, 2, 32); pro = torch.rand(4, P) * 2 - 1; priv = torch.rand(4, V) * 2 - 1
    with torch.no_grad():
        a0, v0 = base.actor(scan, pro), base.critic(scan, pro, priv)
    for mode in ("pos", "posvel", "future"):
        K = opp_token_dim(mode)
        m, _extra, fresh = load_for_memory(path, "cpu", opp_token=mode,
                                           override={"proprio_dim": P + K, "priv_dim": V})
        m.eval()
        assert fresh == [], fresh                     # nothing new: this arm is columns, not modules
        assert m.meta["opp_token"] == mode
        assert float(m.actor.pro[0].weight.detach()[:, P:].abs().max()) == 0.0
        assert float(m.critic.pro[0].weight.detach()[:, P:P + K].abs().max()) == 0.0
        # the critic's privileged columns moved right by K and are otherwise untouched
        assert torch.equal(m.critic.pro[0].weight.detach()[:, P + K:],
                           base.critic.pro[0].weight.detach()[:, P:])
        with torch.no_grad():
            one = m.actor(scan, torch.cat([pro, torch.randn(4, K)], 1))
            two = m.actor(scan, torch.cat([pro, torch.randn(4, K) * 100], 1))
            vc = m.critic(scan, torch.cat([pro, torch.randn(4, K)], 1), priv)
        assert torch.equal(one, two), "the block moved the action at init"
        assert float((one - a0).abs().max()) < 1e-7, float((one - a0).abs().max())
        assert float((vc - v0).abs().max()) < 1e-6, float((vc - v0).abs().max())


def test_fresh_modules_are_seeded_from_their_names_so_the_arms_only_differ_by_width(tmp_path) -> None:
    """The confound `feat/motion-memory` measured, in the form this branch has it: a wider proprio
    layer draws more numbers, so without name-seeding the GRU of the `future` arm is not the GRU of
    the control and "A3 beats A0" could be an initialisation."""
    from f1sim.learn.memory import memory_spec
    from f1sim.learn.model import ActorCritic, load_for_memory, save_checkpoint
    torch.manual_seed(0)
    P, V = 64, 12
    base = ActorCritic(2, 32, P, V, act_dim=2)
    path = str(tmp_path / "base.pt"); save_checkpoint(path, base)

    def gru(mode, seed):
        K = opp_token_dim(mode) if mode != "off" else 0
        torch.manual_seed(11)                          # same ambient state for both arms
        m, _e, _f = load_for_memory(path, "cpu", memory_spec(hidden_size=32),
                                    opp_token=(None if mode == "off" else mode), init_seed=seed,
                                    override={"proprio_dim": P + K, "priv_dim": V})
        return m.actor.memory.gru.weight_ih_l0.clone()

    assert torch.equal(gru("off", 701), gru("future", 701))
    assert torch.equal(gru("pos", 701), gru("posvel", 701))
    assert not torch.equal(gru("off", 701), gru("off", 702))     # ... and the seed still decides

    def gru_unseeded(mode):
        K = opp_token_dim(mode) if mode != "off" else 0
        torch.manual_seed(11)
        m, _e, _f = load_for_memory(path, "cpu", memory_spec(hidden_size=32),
                                    opp_token=(None if mode == "off" else mode),
                                    override={"proprio_dim": P + K, "priv_dim": V})
        return m.actor.memory.gru.weight_ih_l0.clone()

    assert not torch.equal(gru_unseeded("off"), gru_unseeded("future")), \
        "the confound this flag exists for did not reproduce; the test no longer tests anything"


def test_a_lead_that_was_never_taken_is_never_paid() -> None:
    """The trap `--spawn-order random` sets: one race in three starts with the learner ahead. A
    bonus for holding a lead you were handed on the grid is a bonus for a grid position, and a
    crashed opponent respawning behind the field is the same thing arriving mid-race."""
    env = _env("off", m=2, n=2, reward_overtake_hold=3.0, overtake_hold_dist=1.5,
               overtake_hold_time=1.0)
    steps = int(round(1.0 / env.sim.control_dt))
    lead = torch.full((env.B, env.M - 1), -2.0)
    valid = torch.ones(env.B, dtype=torch.bool)
    none = torch.zeros(env.B, dtype=torch.bool)
    ls = torch.zeros(env.B, env.M - 1); lp = torch.ones_like(ls)       # the state a reset leaves
    total = 0.0
    for _ in range(4 * steps):
        b, ls, lp = env.overtake_hold(lead, valid, none, ls, lp)
        total += float(b[0])
    assert total == 0.0, "a lead held from the spawn was paid"
    # let the other car draw level, then take it back: now it is a pass, and it pays once
    level = torch.zeros(env.B, env.M - 1)
    for _ in range(3):
        _b, ls, lp = env.overtake_hold(level, valid, none, ls, lp)
    got = []
    for _ in range(steps + 3):
        b, ls, lp = env.overtake_hold(lead, valid, none, ls, lp)
        got.append(float(b[0]))
    assert sum(got) == 1.0, got


def test_the_step_after_a_reset_does_not_arm_the_bonus() -> None:
    """The gap is invalid for one step after a reset. Reading "no valid gap" as "not leading" would
    arm the payment on exactly the spawn it is meant to exclude."""
    env = _env("off", m=2, n=2, reward_overtake_hold=3.0, overtake_hold_dist=1.5,
               overtake_hold_time=1.0)
    steps = int(round(1.0 / env.sim.control_dt))
    lead = torch.full((env.B, env.M - 1), -2.0)
    none = torch.zeros(env.B, dtype=torch.bool)
    ls = torch.zeros(env.B, env.M - 1); lp = torch.ones_like(ls)
    invalid = torch.zeros(env.B, dtype=torch.bool)
    _b, ls, lp = env.overtake_hold(lead, invalid, none, ls, lp)        # the step after the reset
    total = 0.0
    for _ in range(2 * steps):
        b, ls, lp = env.overtake_hold(lead, torch.ones(env.B, dtype=torch.bool), none, ls, lp)
        total += float(b[0])
    assert total == 0.0


# ---------------------------------------------------------------- the ablation diagnostic
def test_the_ablation_zeroes_the_block_and_keeps_its_width() -> None:
    """`A3 ~ A0` has two causes that call for opposite next steps -- the planner ignored the block,
    or it used it and the block did not pay. Zeroing the block on a trained arm separates them."""
    env = _env("future", m=2, n=2, opp_token_ablate=True)
    tok, _ = _tokens(env, [(0.0, 0.0, 0.0, 3.0), (2.0, 0.0, 0.0, 5.0)])
    assert tok.shape[1] == opp_token_dim("future")      # the actor's first layer still fits
    assert float(tok.abs().max()) == 0.0
    live = _env("future", m=2, n=2)
    tok2, _ = _tokens(live, [(0.0, 0.0, 0.0, 3.0), (2.0, 0.0, 0.0, 5.0)])
    assert float(tok2.abs().max()) > 0.0, "the un-ablated env must actually report something"


def test_ablating_a_block_that_does_not_exist_is_refused() -> None:
    """It would silently be the control arm wearing another arm's name."""
    with pytest.raises(ValueError, match="control arm"):
        _env("off", m=2, n=2, opp_token_ablate=True)
