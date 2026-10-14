"""The race a v2 run trains in (UNIT 5b): the learner's unobserved random cap, the BC clearance mask,
the opponents' plan format, and the boot camp's league in the v2 trainers.

* the cap: drawn per episode for the policy-driven cars only, from a stream of its own (an env with it
  spawns exactly as one without), held under the curriculum's `set_speed_cap`, never stacked across
  episodes, off inside `learner_cap_suspended`, read by the critic (`privileged_v2`) and by nothing the
  actor sees; on `learner_cap_prob` of the episodes only, 1 drawing exactly the caps it always did;
* the clearance: the rectangle and prism SAT separations against known geometry and against the
  simulator's own contact test, and `plan_clearance` on a race -- a plan into a standing car is inside
  0.3 m, the same plan beside it is not, a car parked on a crate is inside its prop;
* the mask: a row that keeps no clearance carries no imitation loss and every auxiliary one, and a
  collection reports its masked share and its capped share;
* the command lines: v2 runs default to linear opponents in a race, the 3-9 m/s cap and the 0.3 m mask;
  v1 refuses all three; evaluate's "auto" picks linear opponents for a v2 checkpoint only;
* end to end (slow): v2 PPO races a v1 league seat that its own snapshot takes over, and v2 DAgger
  logs the masked and capped shares of its collection.
"""
from __future__ import annotations

import functools
import json
import math
import os
import shutil
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1sim import Config, maps, mpc  # noqa: E402
from f1sim.gym_env import PRIV_V2_KEYS, EnvConfig  # noqa: E402
from f1sim.learn import common  # noqa: E402

TRACK = "gen:competition:0"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = os.path.join(ROOT, "checkpoints", "iccas_bootcamp_safest.pt")


@functools.lru_cache(maxsize=1)
def _track():
    from f1sim.raceline import Raceline
    tr = maps.load(TRACK)
    return tr, Raceline.build_cached(tr)


def _env(envs=8, seed=3, beams=36, **kw):
    tr, rl = _track()
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = beams
    e = dict(action_mode="plan", race_size=2, opponent="teacher", opp_speed_range=(0.7, 0.9), collision_mode="soft",
             compile_tracker=False, max_steps=40, speed_cap=9.0, obs_version=2, speed_mode="wp4", scan_stack=1,
             hist_len=20, hist_stride=2)
    e.update(kw)
    env = common.make_env([tr], envs, "cpu", EnvConfig(**e), cfg=cfg, seed=seed, rls=[rl])
    env.reset(seed=seed)
    return env


# ===================================================================== the learner's random cap
def test_the_cap_is_drawn_per_episode_for_the_learner_only_and_moves_no_other_draw():
    off = _env()
    on = _env(learner_cap_range=(3.0, 9.0))
    assert off.learner_cap is None and on.learner_cap is not None
    assert torch.equal(off.sim.state, on.sim.state)                      # its own stream: the same spawns
    for k in off._last_obs:
        assert torch.equal(off._last_obs[k], on._last_obs[k])            # and the same first observation
    ego = on.learner
    cap = on.speed_cap
    assert bool(((cap[ego] >= 3.0) & (cap[ego] <= 9.0)).all()) and len(set(cap[ego].tolist())) == int(ego.sum())
    assert torch.equal(cap[~ego], off.speed_cap[~ego])                    # the teacher cars keep theirs
    # the critic is told (its last column), the actor's observation has no cap in it at all
    assert PRIV_V2_KEYS[-1] == "speed_cap"
    assert torch.allclose(on.privileged_v2()[:, -1], cap / on.ecfg.v_max_policy)
    assert set(on._last_obs) == {"scan", "ego"}
    # the curriculum moves under it: the draw holds, the lower of the two binds
    drawn = on.learner_cap.clone()
    on.set_speed_cap(5.0)
    assert torch.equal(on.speed_cap[ego], torch.minimum(torch.full_like(drawn[ego], 5.0), drawn[ego]))
    on.set_speed_cap(9.0)
    assert torch.equal(on.speed_cap[ego], drawn[ego])
    # a new episode draws anew and does not stack on the last draw
    lo = int(torch.nonzero(ego).flatten()[int(on.speed_cap[ego].argmin())])
    seen = []
    for _ in range(12):
        on._reset_envs(torch.tensor([lo]))
        seen.append(float(on.speed_cap[lo]))
    assert len(set(seen)) == 12 and max(seen) > drawn[lo]
    # a seeded reset names the caps too
    again = _env(learner_cap_range=(3.0, 9.0))
    assert torch.equal(again.learner_cap, _env(learner_cap_range=(3.0, 9.0)).learner_cap)


def test_a_share_of_the_episodes_draws_a_cap_and_every_episode_draws_the_caps_it_always_did():
    every = _env(learner_cap_range=(3.0, 9.0))
    assert torch.equal(every.learner_cap, _env(learner_cap_range=(3.0, 9.0), learner_cap_prob=1.0).learner_cap)
    off = _env()
    some = _env(learner_cap_range=(3.0, 9.0), learner_cap_prob=0.3)
    assert torch.equal(some.sim.state, off.sim.state)                   # the coin is the cap's own stream too
    ego = some.learner
    rows = torch.nonzero(ego).flatten()
    drawn, caps = [], []
    for _ in range(40):
        some._reset_envs(rows)
        drawn.append(torch.isfinite(some.learner_cap[rows]))
        caps.append(some.speed_cap[rows].clone())
    drawn, caps = torch.stack(drawn), torch.stack(caps)
    assert 0.2 < float(drawn.float().mean()) < 0.4                     # 320 episodes, 0.3 +- 0.026
    assert bool(((caps[drawn] >= 3.0) & (caps[drawn] <= 9.0)).all())
    assert bool((caps[~drawn] == 9.0).all())                             # the rest: the env's own cap
    assert torch.equal(some.speed_cap[~ego], off.speed_cap[~ego])        # the teacher cars never draw
    # the critic reads the cap in force either way, the actor nothing
    assert torch.allclose(some.privileged_v2()[:, -1], some.speed_cap / some.ecfg.v_max_policy)
    none = _env(learner_cap_range=(3.0, 9.0), learner_cap_prob=0.0)
    assert bool(torch.isinf(none.learner_cap).all()) and torch.equal(none.speed_cap, off.speed_cap)
    with pytest.raises(ValueError, match="learner_cap_prob"):
        _env(learner_cap_range=(3.0, 9.0), learner_cap_prob=1.5)


def test_a_measurement_inside_a_run_scores_at_the_fixed_cap():
    env = _env(learner_cap_range=(3.0, 9.0))
    with env.learner_cap_suspended():
        env.reset()
        assert bool(torch.isinf(env.learner_cap).all()) and bool((env.speed_cap == 9.0).all())
    env.reset()
    assert bool((env.speed_cap[env.learner] < 9.0).any())
    with pytest.raises(ValueError, match="learner_cap_range"):
        _env(learner_cap_range=(0.0, 9.0))


def test_the_step_says_which_learner_steps_the_cap_clipped():
    env = _env(learner_cap_range=(3.0, 3.5))
    a = torch.zeros(env.B, env.act_dim)
    a[:, mpc.plan_layout("wp4").speed] = 0.6                          # 8 m/s asked for everywhere
    _o, _r, _t, _tr, info = env.step(a)
    assert bool(info["speed_capped"][env.learner].all())


# ===================================================================== the clearance geometry
def test_rect_separation_is_the_gap_between_the_bodies():
    from f1sim.learn.train_v2 import rect_separation
    z = torch.zeros(())
    c = lambda x, y: torch.tensor([x, y])
    side = rect_separation(c(0, 0), z, 0.3, 0.15, c(0.0, 0.55), z, 0.3, 0.15)
    assert float(side) == pytest.approx(0.25, abs=1e-6)                 # abreast, 0.25 m between the sides
    behind = rect_separation(c(0, 0), z, 0.3, 0.15, c(1.0, 0.0), z, 0.3, 0.15)
    assert float(behind) == pytest.approx(0.4, abs=1e-6)
    over = rect_separation(c(0, 0), z, 0.3, 0.15, c(0.0, 0.2), z, 0.3, 0.15)
    assert float(over) == pytest.approx(-0.1, abs=1e-6)                 # 0.1 m of overlap
    turned = rect_separation(c(0, 0), z, 0.3, 0.15, c(0.0, 0.8), torch.tensor(math.pi / 2), 0.3, 0.15)
    assert float(turned) == pytest.approx(0.35, abs=1e-6)               # a crossing car, 0.35 m ahead of the side


def test_prism_separation_reads_prism_contacts_the_other_way():
    from f1sim.prop_math import prism_contacts, prism_separation, section_halfplanes
    import numpy as np
    sq = np.array([[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]])
    n, d = section_halfplanes(sq, 8)
    pn = torch.tensor(n, dtype=torch.float32)[None, None].expand(3, 1, 8, 2)
    pd = torch.tensor(d, dtype=torch.float32)[None, None].expand(3, 1, 8)
    poses = torch.tensor([[[1.0, 0.0, 0.0]], [[0.75, 0.0, 0.0]], [[0.5, 0.0, 0.0]]])
    zlo, zhi = torch.zeros(3, 1), torch.full((3, 1), 0.3)
    car = torch.tensor([[0.29, 0.155], [0.29, -0.155], [-0.29, -0.155], [-0.29, 0.155]])[None].expand(3, 4, 2)
    sep = prism_separation(car, poses, pn, pd, zlo, zhi, 0.2)
    depth, _n, _i = prism_contacts(car, poses, pn, pd, zlo, zhi, 0.2)
    assert sep.tolist() == pytest.approx([0.51, 0.26, 0.01], abs=1e-5)  # the box's face from the car's nose
    deep = prism_separation(car, poses - torch.tensor([0.4, 0.0, 0.0]), pn, pd, zlo, zhi, 0.2)
    depth2, _n, _i = prism_contacts(car, poses - torch.tensor([0.4, 0.0, 0.0]), pn, pd, zlo, zhi, 0.2)
    assert bool((depth == 0).all()) and torch.allclose(deep[depth2 > 0], -depth2[depth2 > 0])
    assert bool(torch.isinf(prism_separation(car, poses, pn, pd, zlo, zhi * 0, 0.2)).all())   # dead slots


def test_plan_clearance_sees_a_standing_car_in_the_plan_and_not_beside_it():
    from f1sim.learn.train_v2 import plan_clearance
    env = _env(envs=4)
    env._reset_envs(torch.arange(env.B))
    st = env.sim.state.clone()
    ego, opp = torch.tensor([0, 2]), torch.tensor([1, 3])
    pose = st[ego, :3].clone()
    fwd = torch.stack([pose[:, 2].cos(), pose[:, 2].sin()], 1)
    left = torch.stack([-pose[:, 2].sin(), pose[:, 2].cos()], 1)
    ahead = torch.cat([pose[:1, :2] + 1.0 * fwd[:1], pose[:1, 2:]], 1)    # race 0: 1 m in front, standing
    beside = torch.cat([pose[1:, :2] + 2.0 * left[1:], pose[1:, 2:]], 1)  # race 1: 2 m to the side
    env.sim.reset(ego, pose, torch.full((2,), 3.0))
    env.sim.reset(opp, torch.cat([ahead, beside]), torch.zeros(2))
    a = torch.zeros(env.B, env.act_dim)
    a[:, mpc.plan_layout("wp4").speed] = -0.2                           # straight on at 4 m/s
    cl = plan_clearance(env, a)
    assert float(cl["car"][0]) < 0.0                                     # it plans through the car
    assert 1.4 < float(cl["car"][2]) < 1.8                               # 2 m apart minus the two half widths
    assert bool(torch.isinf(cl["prop"]).all()) and torch.equal(cl["min"], cl["car"])


def test_plan_clearance_sees_a_crate_under_the_car():
    from f1sim.learn.train_v2 import plan_clearance
    from f1sim.prop_math import _slot_is_live
    env = _env(envs=4, procedural_obstacles=1.0, procedural_density=3.0)
    poses, n, d, zlo, zhi = env.procedural.slots()
    live = torch.stack([_slot_is_live(n[:, c], zlo[:, c], zhi[:, c]) for c in range(n.shape[1])], 1)
    i = int(torch.nonzero(live.any(1)).flatten()[0])
    c = int(torch.nonzero(live[i]).flatten()[0])
    on = torch.tensor([[float(poses[i, c, 0]), float(poses[i, c, 1]), 0.3]])
    env.sim.reset(torch.tensor([i]), on, torch.zeros(1))
    cl = plan_clearance(env, torch.zeros(env.B, env.act_dim))
    assert float(cl["prop"][i]) < 0.0 and float(cl["min"][i]) < 0.0


# ===================================================================== the mask in the loss
def test_a_masked_row_carries_no_imitation_and_every_auxiliary_loss():
    from f1sim.learn import dagger_v2
    from f1sim.learn.model_v2 import ActorCriticV2
    torch.manual_seed(0)
    m = ActorCriticV2(n_beams=61, aux={"b8": True})
    T, k = 8, 2
    g = torch.Generator().manual_seed(1)
    s = {"scan": torch.rand(T, k, 1, 61, generator=g), "ego": 0.3 * torch.rand(T, k, 210, generator=g),
         "c": torch.zeros(T, k, 1), "keep": torch.ones(T, k), "valid": torch.ones(T, k, dtype=torch.bool),
         "label": torch.rand(T, k, 8, generator=g) * 2 - 1, "b8_lab": 0.2 * torch.randn(T, k, 3, 3, generator=g),
         "b8_cand": torch.ones(T, k, 3, dtype=torch.bool), "b8_age": torch.zeros(T, k, 3, dtype=torch.long)}
    kw = dict(burn=2, plan_spec=mpc.PlanSpec(speed_mode="wp4"), v_max=10.0, coefs={"b8": 0.5}, beams=False)
    base = dagger_v2.sequence_losses(m, s, **kw)
    keep_all = dagger_v2.sequence_losses(m, {**s, "bcok": torch.ones(T, k, dtype=torch.bool)}, **kw)
    none = dagger_v2.sequence_losses(m, {**s, "bcok": torch.zeros(T, k, dtype=torch.bool)}, **kw)
    f = lambda t: float(t.detach())
    assert torch.equal(base["bc"], keep_all["bc"]) and f(base["bc"]) > 0
    assert f(none["bc"]) == 0.0 and torch.equal(none["b8"], base["b8"]) and f(none["b8"]) > 0
    half = torch.ones(T, k, dtype=torch.bool); half[:, 1] = False
    one = dagger_v2.sequence_losses(m, {**s, "bcok": half}, **kw)
    only = {key: (v[:, :1] if torch.is_tensor(v) and v.dim() >= 2 else v) for key, v in s.items()}
    assert f(one["bc"]) == pytest.approx(f(dagger_v2.sequence_losses(m, only, **kw)["bc"]), rel=1e-5)


def test_a_collection_reports_its_masked_and_capped_shares():
    from f1sim.learn import conditioning as cm
    from f1sim.learn import dagger_v2
    from f1sim.learn.model_v2 import ActorCriticV2
    env = _env(envs=8, beams=61, learner_cap_range=(3.0, 9.0), opponent_speed_mode="linear",
               spawn_gap=(0.8, 1.2))                                    # cars close: some plans run into one
    teacher = common.make_teacher([_track()[1]], env)
    torch.manual_seed(0)
    model = ActorCriticV2(n_beams=61, obs=dict(vars(common.obs_spec(env))))
    buf = dagger_v2.collect(env, model, dagger_v2.Wp4Labels(env, teacher, "env"), 30, 1.0, torch.device("cpu"),
                            dagger_v2.SequenceBuffer(1), noise=0.0, dial=cm.DialDraw(env, 0.3, 0.3),
                            cspec=cm.CondSpec.from_meta(model.meta["cond"]), b8=False, near=False, beam=False,
                            bc_clearance=0.3).finalize()
    st = buf.stats
    assert st["bc_valid_labels"] > 0 and 0.0 < st["bc_masked_frac"] <= 1.0
    assert st["bc_masked_frac"] == pytest.approx(1.0 - float(buf.data["bcok"][buf.data["valid"]].float().mean()))
    assert st["bc_masked_car_frac"] > 0 and st["bc_masked_prop_frac"] == 0.0
    assert 0.0 <= st["speed_capped_frac"] <= 1.0
    off = dagger_v2.collect(env, model, dagger_v2.Wp4Labels(env, teacher, "env"), 3, 1.0, torch.device("cpu"),
                            dagger_v2.SequenceBuffer(1), noise=0.0, dial=cm.DialDraw(env, 0.3, 0.3),
                            cspec=cm.CondSpec.from_meta(model.meta["cond"]), b8=False, near=False,
                            beam=False).finalize()
    assert "bcok" not in off.data and math.isnan(off.stats["bc_masked_frac"])


# ===================================================================== held-out auxiliary metrics
def test_the_heldout_metrics_read_the_heads_in_metres():
    from f1sim.learn.train_v2 import aux_heldout_metrics
    T, m = 4, 2
    near = torch.zeros(T, m, 5)
    near[..., 0] = 1.0; near[..., 1] = 2.0 / 5.0; near[..., 2] = 0.5 / 2.0; near[..., 3] = 1.0 / 5.0
    near[0, 0, 0] = 0.0; near[0, 0, 1:] = 0.0                                # no car seen there
    pred = near.clone(); pred[..., 0] = torch.where(near[..., 0] > 0, 4.0, -4.0)
    pred[..., 2] += 0.1 / 2.0                                                # 0.1 m off sideways
    beam_lab = torch.tensor([1, 1, 0, 0, -1], dtype=torch.int8).expand(T, m, 5)
    beam_lg = torch.tensor([3.0, -3.0, -3.0, 3.0, 3.0]).expand(T, m, 5)
    lab = torch.zeros(T, m, 3, 3); lab[..., 0, :] = torch.tensor([1.0 / 5.0, -0.4 / 2.0, 0.0])
    cand = torch.zeros(T, m, 3, dtype=torch.bool); cand[..., 0] = True
    b8 = torch.zeros(T, m, 3, 4); b8[..., 2, 0] = 5.0; b8[..., 2, 1:3] = torch.tensor([1.0 / 5.0, -0.3 / 2.0])
    b8[..., :2, 0] = -5.0                                                     # the car is in slot 2
    out = {"near_car": pred, "beam_car": beam_lg, "b8": b8}
    s = {"near": near, "beam": beam_lab, "b8_lab": lab, "b8_cand": cand, "b8_age": torch.zeros(T, m, 3, dtype=torch.long)}
    got = aux_heldout_metrics(out, s, torch.full((T, m), 10, dtype=torch.long))
    assert got["near_presence_auc"] == 1.0 and got["near_rows_3m"] == T * m - 1
    assert got["near_dy_p50_3m"] == pytest.approx(0.1, abs=1e-5) and got["near_dx_p50_3m"] == pytest.approx(0.0, abs=1e-6)
    assert got["beam_precision"] == pytest.approx(0.5) and got["beam_recall"] == pytest.approx(0.5)
    assert got["b8_dy_p50_3m"] == pytest.approx(0.1, abs=1e-5) and got["b8_dx_p50_3m"] == pytest.approx(0.0, abs=1e-6)
    assert aux_heldout_metrics({}, {}) == {}


# ===================================================================== the command lines
def _parser(module):
    from f1sim.viewer.console.training_schema_export import capture_parser
    return capture_parser(module)


@pytest.mark.parametrize("module,prep", [("f1sim.learn.dagger", "dagger_v2"), ("f1sim.learn.ppo", "ppo_v2")])
def test_a_v2_run_races_linear_opponents_under_a_random_cap_unless_told(module, prep):
    import importlib
    p = importlib.import_module(f"f1sim.learn.{prep}")
    ap = _parser(module)

    def run(*extra):
        argv = ["--arch", "v2", *extra]
        a = ap.parse_args(argv)
        applied = p.prepare_args(ap, a, argv)
        return a, applied
    a, applied = run("--race-size", "4", "--opponent", "teacher")
    assert a.opponent_speed_mode == "linear" and applied["opponent_speed_mode"] == "linear"
    assert a.learner_cap_range == (3.0, 9.0) and a.learner_cap_prob == 0.3
    # the perception gate's decision: one frame, every perception head on at the gate's coefficients
    assert (a.v2_frames, a.aux_b8, a.aux_near_car, a.aux_beam_car) == (1, 0.5, 0.5, 5.0)
    assert p.env_kwargs(a)["opp_visibility"] is True
    assert p.env_kwargs(a)["learner_cap_range"] == (3.0, 9.0) and p.env_kwargs(a)["learner_cap_prob"] == 0.3
    a, _ = run("--learner-cap-prob", "1")
    assert a.learner_cap_prob == 1.0 and p.env_kwargs(a)["learner_cap_prob"] == 1.0
    a, _ = run("--race-size", "4", "--opponent", "teacher", "--opponent-speed-mode", "wp4", "--learner-cap-range", "0", "0")
    assert a.opponent_speed_mode == "wp4" and a.learner_cap_range == ()
    a, applied = run()                                                    # alone: no other car to plan for
    assert a.opponent_speed_mode is None and "opponent_speed_mode" not in applied
    a, _ = run("--race-size", "2", "--opponent", "policy")             # self-play: every car the learner
    assert a.opponent_speed_mode is None
    for bad in (["--learner-cap-range", "5", "3"], ["--learner-cap-range", "0", "12"], ["--learner-cap-prob", "1.5"],
                ["--learner-cap-prob", "-0.1"], ["--learner-cap-range", "0", "0", "--learner-cap-prob", "0.5"]):
        argv = ["--arch", "v2", *bad]
        with pytest.raises(SystemExit):
            p.prepare_args(ap, ap.parse_args(argv), argv)
    if prep == "dagger_v2":
        a, _ = run()
        assert a.bc_clearance == 0.3
        a, _ = run("--bc-clearance", "0")
        assert a.bc_clearance == 0.0
    else:
        a, _ = run("--league-slots", "x.pt")                             # no longer refused (UNIT 4's refusal)
        assert a.league_slots == "x.pt"


@pytest.mark.parametrize("module,flag,val", [("f1sim.learn.dagger", "--bc-clearance", ["0.3"]),
                                             ("f1sim.learn.dagger", "--learner-cap-range", ["3", "9"]),
                                             ("f1sim.learn.ppo", "--learner-cap-range", ["3", "9"]),
                                             ("f1sim.learn.dagger", "--learner-cap-prob", ["0.3"]),
                                             ("f1sim.learn.ppo", "--learner-cap-prob", ["0.3"]),
                                             ("f1sim.learn.ppo", "--opponent-speed-mode", ["linear"]),
                                             ("f1sim.learn.dagger", "--opponent-speed-mode", ["wp4"])])
def test_a_v1_run_refuses_the_v2_race_flags(monkeypatch, module, flag, val):
    import importlib
    mod = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", [module, flag, *val])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "--arch v2" in str(e.value)


def test_evaluate_races_a_v2_checkpoint_against_linear_cars_and_leaves_v1_as_it_was():
    from f1sim.learn.evaluate import _opponent_format
    v2, v1 = {"version": 2, "n_frames": 1}, {"scan_stack": 6}
    assert _opponent_format("auto", v2, "plan", 4, "slots") == "linear"
    assert _opponent_format("auto", v2, "plan", 2, "teacher") == "linear"
    assert _opponent_format("auto", v2, "plan", 1, "policy") is None
    assert _opponent_format("auto", v2, "plan", 2, "policy") is None   # self-play: no other driver
    assert _opponent_format("auto", v1, "plan", 4, "slots") is None
    assert _opponent_format("wp4", v2, "plan", 4, "slots") == "wp4"
    assert _opponent_format(None, v2, "plan", 4, "slots") is None


# ===================================================================== end to end (CPU)
def _run_main(module, argv, runs, monkeypatch):
    import importlib
    monkeypatch.setattr(common, "RUNS_DIR", str(runs))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(sys, "argv", [module, *argv])
    importlib.import_module(module).main()


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(V1), reason="archived v1 checkpoint not present")
def test_v2_ppo_races_the_v1_league_and_its_snapshot_takes_the_seat(tmp_path, monkeypatch):
    from f1sim.learn.model import load_checkpoint
    from f1sim.learn.model_v2 import ActorCriticV2
    from v2_fixtures import generated_v2_checkpoint
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        seat = str(tmp_path / "league_seat.pt")
        shutil.copy(V1, seat)
        init = generated_v2_checkpoint(str(tmp_path / "student.pt"))
        slots = json.dumps([{"kind_mix": ["forzaeth", "lane_switch"], "ckpt_mix": [seat], "ckpt_p": 1.0,
                             "speed_scale": [0.8, 0.9]}])
        _run_main("f1sim.learn.ppo", ["--arch", "v2", "--name", "pl", "--init", init, "--envs", "4", "--race-size",
                                      "2", "--opp-slots", slots, "--tracks", TRACK, "--horizon", "16",
                                      "--minibatch", "32", "--grad-accum", "1", "--total", "64", "--epochs", "1",
                                      "--device", "cpu", "--sim-backend", "eager", "--wandb", "disabled",
                                      "--log-every", "1", "--save-every", "1", "--critic-warmup", "0",
                                      "--league-slots", seat, "--league-every", "1", "--collision-mode", "soft",
                                      "--car-contact-penalty", "1", "--cap0", "9", "--cap1", "9"],
                  tmp_path, monkeypatch)
        recs = [json.loads(x) for x in open(tmp_path / "pl" / "progress.jsonl")]
        assert [r["update"] for r in recs] == [1, 2]
        assert all(3.0 <= r["rollout/learner_cap_mean"] <= 9.0 and 0.0 <= r["rollout/speed_capped_frac"] <= 1.0
                   for r in recs)
        snap = tmp_path / "pl" / "league_0.pt"
        model, extra = load_checkpoint(str(snap))
        assert isinstance(model, ActorCriticV2) and "opt" not in extra and extra["speed_mode"] == "wp4"
        assert all(k in recs[0] for k in ("heldout/near_presence_auc", "heldout/beam_precision", "heldout/b8_dy_p50_3m"))
        _m, ex = load_checkpoint(str(tmp_path / "pl" / "ppo_final.pt"))
        assert ex["experiment"]["checkpoint_rule"] == "last"
        race = ex["experiment"]["race"]
        assert race == {"learner_cap_range": [3.0, 9.0], "learner_cap_prob": 0.3, "opponent_speed_mode": "linear",
                        "league_seats": [0], "league_every": 1}
    finally:
        torch.set_num_threads(threads)


@pytest.mark.slow
def test_v2_dagger_logs_the_masked_and_capped_shares_of_its_collection(tmp_path, monkeypatch):
    from f1sim.learn.model import load_checkpoint
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        _run_main("f1sim.learn.dagger", ["--arch", "v2", "--name", "dm", "--envs", "4", "--race-size", "2",
                                         "--opponent", "teacher", "--spawn-gap", "0.8", "1.2", "--tracks", TRACK,
                                         "--iters", "1", "--steps", "40", "--epochs", "1", "--batch", "64",
                                         "--device", "cpu", "--eager", "--wandb", "disabled", "--wp4-label", "env",
                                         "--eval-steps", "20", "--teacher-eval-steps", "0", "--speed-cap", "8",
                                         "--aux-b8", "0", "--log-every", "1"], tmp_path, monkeypatch)
        rec = [json.loads(x) for x in open(tmp_path / "dm" / "progress.jsonl")][-1]
        assert 0.0 <= rec["collect/bc_masked_frac"] <= 1.0 and rec["collect/bc_valid_labels"] > 0
        assert 0.0 <= rec["collect/speed_capped_frac"] <= 1.0
        _m, extra = load_checkpoint(str(tmp_path / "dm" / "student_latest.pt"))
        assert extra["race"] == {"bc_clearance_m": 0.3, "learner_cap_range": [3.0, 9.0], "learner_cap_prob": 0.3,
                                 "opponent_speed_mode": "linear"}
        assert extra["collect"]["bc_masked_frac"] == rec["collect/bc_masked_frac"]
        # the perception heads by default, scored held out every iteration; the last weights kept
        assert extra["checkpoint_rule"] == "last" and rec["heldout_scores_iter"] == -1
        assert all(k in rec for k in ("heldout/near_presence_auc", "heldout/beam_recall", "heldout/near_dy_p50_3m"))
        assert set(extra["heldout"]) == {k[len("heldout/"):] for k in rec if k.startswith("heldout/")}
    finally:
        torch.set_num_threads(threads)
