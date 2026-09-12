"""PPO fine-tuning with an asymmetric critic, initialized from the DAgger student.

- KL(pi_IL || pi) regularizer, decayed over the first `kl_decay` env steps, keeps the policy near
  the imitation policy while the critic warms up.
- Speed-cap curriculum: the commanded speed cap ramps from `cap0` to `cap1` over `cap_steps`
  env steps (the cap is part of the observation, so the policy stays consistent).
- Time-limit truncations bootstrap with the final observation value estimate.
"""
from __future__ import annotations

from collections import deque

import argparse
import copy
import math
import os
import time

import numpy as np
import torch

from ..gym_env import EnvConfig, PRIV_OPP_DIST_SCALE, REWARD_COMPONENT_KEYS
from ..params import Config
from . import common
from . import conditioning as cond_mod
from . import grip_runtime as grip_rt
from .model import ActorCritic, load_checkpoint, load_for_conditioning, save_checkpoint
from .obs import flatten_obs
from .returns import compute_gae

#: How close an opponent has to be for the auxiliary head to be scored on it [m]. This is a
#: deliberately conservative auxiliary range, NOT the sensor's reach: the LiDAR sees to 10 m, but
#: beyond a few metres an opponent is often occluded by the boundary or covered by too few beams to
#: read an offset from, so scoring the head out there mostly teaches it the conditional mean of an
#: empty road. Measured on a rollout, only ~47 % of frames have any beam on the opponent at all.
AUX_OPP_RANGE_M = 6.0


def sample_rollout_action(
    model: ActorCritic, scan: torch.Tensor, proprio: torch.Tensor, cond: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the sampled action and its matching log probability for PPO storage.

    `cond` is the frozen conditioning for this step. It is passed explicitly, never defaulted: the
    log probability stored here is the one the update ratio is measured against, so the recomputed
    log probability has to come from the identical input.
    """
    distribution = model.actor.dist(scan, proprio, cond)
    action = distribution.sample()
    return action, distribution.log_prob(action).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=f"ppo_{time.strftime('%m%d_%H%M')}")
    ap.add_argument("--init", default="", help="DAgger checkpoint to start from")
    ap.add_argument("--envs", type=int, default=2048); ap.add_argument("--tracks", default="train", help="'train', 'eval' or comma separated catalog names")
    ap.add_argument("--horizon", type=int, default=32); ap.add_argument("--total", type=float, default=100e6)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--lr-end", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.99); ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--vf", type=float, default=0.5); ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--sim-backend", default="compile", choices=("compile", "eager", "graphs"),
                    help="simulator runtime. 'compile' is the current default and what every run so "
                         "far used (torch.compile on the physics substep loop and the plan solver). "
                         "'eager' turns both off. 'graphs' turns both off and captures explicit CUDA "
                         "graphs instead, which starts in about a second rather than minutes. Opt in "
                         "only; nothing about the default changes.")
    ap.add_argument("--wandb-group", default="ppo",
                    help="W&B group. The pilot's arms share one so they can be read together.")
    ap.add_argument("--cond", default="none", choices=("none", "zero", "true_mu"),
                    help="Stage-1 friction conditioning arm. 'none' is the unmodified policy; 'zero' "
                         "(A0) and 'true_mu' (A1) both add the zero-initialised conditioning "
                         "projection and differ only in what is fed to it. 'true_mu' is a LAB "
                         "ORACLE: its input is privileged and its checkpoint refuses to load in the "
                         "viewer, export, watch, evaluate or the ROS node.")
    ap.add_argument("--fresh-opt", action="store_true",
                    help="start Adam from scratch instead of restoring the checkpoint's moments. "
                         "Required for the conditioning arms: adding a parameter shifts the "
                         "positional keys Adam's state is indexed by, and silently re-keying them "
                         "would give the two arms different optimiser state.")
    ap.add_argument("--controller", default="legacy", choices=grip_rt.ARMS,
                    help="plan-tracker controller arm. 'legacy' is the untouched path and is the "
                         "default, so an unflagged run is unchanged. 'fixed_low' is the control arm, "
                         "'oracle' reads the true friction (lab only), 'estimated' uses the frozen "
                         "estimator's filtered lower quantile from causal sensors.")
    ap.add_argument("--estimator", default="",
                    help="frozen grip-estimator checkpoint. Required by --controller estimated, and "
                         "rejected for every other arm.")
    ap.add_argument("--critic-priv-adapter", default="",
                    help="declared privileged-input adapter for the critic, e.g. "
                         "'absent_opponent_17_to_21' to run a race-trained critic in a solo env. "
                         "Empty is strict legacy.")
    ap.add_argument("--kl-coef", type=float, default=0.3)
    ap.add_argument("--kl-decay", type=float, default=None,
                    help="env steps over which the imitation-KL leash decays to zero (default: --total, so it "
                         "decays across the whole run). Shortening it to 20 %% was tried and reverted: on a 20M "
                         "run the policy was clean while the leash held (collision rate 0.00, lap 17.4 s) and "
                         "then traded safety for speed the moment it reached zero at 4M (0.33-0.47, lap 12.5 s), "
                         "with KL to the imitation policy rising 0.26 -> 6.8. The leash was holding a "
                         "speed/safety balance the reward alone does not")
    ap.add_argument("--cap0", type=float, default=4.0); ap.add_argument("--cap1", type=float, default=8.0); ap.add_argument("--cap-steps", type=float, default=40e6)
    ap.add_argument("--critic-warmup", type=int, default=10, help="updates with the actor frozen")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--wandb", default="online")
    ap.add_argument("--log-every", type=int, default=1, help="updates between W&B rows (1 = every update)")
    ap.add_argument("--save-every", type=int, default=25); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for network forward/backward (~2x faster)")
    ap.add_argument("--collision-penalty", type=float, default=10.0); ap.add_argument("--steer-penalty", type=float, default=0.05)
    ap.add_argument("--proximity-penalty", type=float, default=0.5, help="per-metre penalty at zero wall gap (0 = off)")
    ap.add_argument("--safe-dist", type=float, default=0.30, help="[m] body-to-wall gap where the proximity penalty starts")
    ap.add_argument("--wrong-way-penalty", type=float, default=0.2, help="per-step penalty while facing backwards along the lane")
    ap.add_argument("--collision-speed-penalty", type=float, default=0.0, help="extra collision penalty per m/s of impact speed")
    ap.add_argument("--cap-gate", type=float, default=0.0,
                    help=">0: [collisions/km] the speed cap only rises while the recent crash rate is "
                         "below this (safety before speed). Per km, not per episode: the fraction of "
                         "episodes that end in a crash saturates at 1.0 as soon as episodes are long "
                         "enough to almost always contain one, and a gate reading a saturated number "
                         "can never open again no matter how much the policy improves")
    ap.add_argument("--cap-gate-quantile", type=float, default=0.9,
                    help="which track holds the gate shut: 1.0 the worst, 0.9 the 90th percentile, "
                         "0.5 the median. The set average is what let a 12 coll/km map hide behind a "
                         "2.1 coll/km mean; the outright worst lets one pathological map freeze the "
                         "curriculum for the other 148")
    ap.add_argument("--cap-gate-min-km", type=float, default=1.0,
                    help="a track needs this many km driven before it can hold the gate shut. Distance, "
                         "not episodes: it is what a rate per km is estimated from, and with 149 tracks "
                         "sharing 1024 envs an episode count large enough to be meaningful takes longer "
                         "to reach than the whole curriculum")
    ap.add_argument("--proximity-speed-ref", type=float, default=0.0, help="[m/s] scale the proximity penalty by (1 + v/ref)")
    ap.add_argument("--plan-clearance-penalty", type=float, default=0.0, help="per-second penalty for a plan inside the wall margin")
    ap.add_argument("--plan-margin", type=float, default=0.15)
    ap.add_argument("--lap-time-bonus", type=float, default=0.0,
                    help="scale of the per-sector time reward. Progress already points at lap time, "
                         "but shaving 0.1 s off a 13.5 s lap moves the return by under 1 %% -- and "
                         "less the longer the track -- so the thing being optimised is nearly "
                         "invisible to the optimiser. This term scores each sector against the "
                         "track's own recent pace, in fractions of the headroom left to the "
                         "raceline, so a hundredth found near the limit outweighs a tenth found far "
                         "from it. At 2.0, and 1.5 s off the pace, 0.1 s of lap is worth about 0.8 "
                         "against 80 for a collision; at 0.1 s off the pace a hundredth is worth 1.3")
    ap.add_argument("--wandb-id", default=None, help="append to this W&B run id (default: the one the "
                                                     "--init checkpoint came from)")
    ap.add_argument("--wandb-new", action="store_true", help="start a fresh W&B run even when resuming")
    ap.add_argument("--steps-base", type=float, default=None,
                    help="x-axis offset for the resumed W&B run; defaults to the checkpoint's own count")
    ap.add_argument("--yield-to-viewer", type=float, default=1.0,
                    help="while a viewer is on screen, sleep this multiple of each update's duration "
                         "so it can reach the GPU. 1.0 halves the training rate and gives the viewer "
                         "a usable frame rate; 0 disables it")
    ap.add_argument("--overtake-bonus", type=float, default=0.0,
                    help="reward per metre of arc taken out of the nearest opponent ahead. Needs "
                         "--race-size > 1. Without it a collision penalty teaches avoidance and only "
                         "avoidance: sitting behind costs nothing, so there is never a reason to pass")
    ap.add_argument("--car-safe-gap", type=float, default=0.6,
                    help="[m] body gap below which the car proximity penalty starts. Larger = the "
                         "policy is asked to react earlier when a door is closing")
    ap.add_argument("--sideslip-penalty", type=float, default=0.0,
                    help="per metre driven, per radian of sideslip beyond ~3.5 deg: prices the grip "
                         "limit itself instead of only the wall it ends at")
    ap.add_argument("--aux-opp", type=float, default=0.0,
                    help="weight of an auxiliary loss making the actor's trunk predict the nearest "
                         "opponent's offset and closing speed from the scan stack (races only). 0 = off")
    ap.add_argument("--aux-grip", type=float, default=0.0,
                    help="weight of an auxiliary loss making the actor's trunk predict the car's true "
                         "friction from what it observes (IMU history). 0 = off")
    ap.add_argument("--car-proximity-penalty", type=float, default=0.0,
                    help="per metre driven with another car's body inside car_safe_gap (0.6 m), scaled "
                         "by closing speed. Without it the only gradient near a car is the overtake "
                         "term paying for closing, right down to contact")
    ap.add_argument("--car-contact-penalty", type=float, default=0.0,
                    help="extra penalty for touching another car, on top of ending the episode. "
                         "Contact and a wall otherwise score the same, so nothing tells the policy "
                         "that a car is something to negotiate with rather than more scenery")
    ap.add_argument("--lap-bonus", type=float, default=0.0,
                    help="reward per completed lap, multiplied by that lap's average speed (track length / lap "
                         "time). Dense progress reward already stands in for lap time, but it saturates where the "
                         "speed cap and not the corner sets the speed -- at a 4 m/s cap that is ~80 %% of these "
                         "tracks, and there every path scores the same. This term keeps paying for a faster lap")
    ap.add_argument("--init-log-std", type=float, default=None, help="reset the actor's exploration log-std at start (default: -1.8 in the plan space, whose curvature knots tolerate far less noise than steer/speed; unchanged otherwise)")
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--scan-stack", type=int, default=3); ap.add_argument("--scan-stride", type=int, default=1, help="control steps between stacked scans")
    ap.add_argument("--hist-len", type=int, default=0, help="proprio history rows in the observation (20 = the last second at stride 2)")
    ap.add_argument("--race-size", type=int, default=1, help="cars per track instance (>1: opponents in the LiDAR, car-car collisions)")
    ap.add_argument("--opponent", default="policy", choices=["policy", "teacher", "mixed"],
                    help="who drives cars 1..M-1: the policy (self-play), the raceline teacher, or per race "
                         "one or the other (mixed, share set by --mixed-teacher-frac)")
    ap.add_argument("--mixed-teacher-frac", type=float, default=0.5)
    ap.add_argument("--opp-speed", type=float, nargs=2, default=(0.6, 1.0), help="teacher opponents: speed scale range per race")
    ap.add_argument("--spawn-gap", type=float, nargs=2, default=(2.5, 6.0), metavar=("LOW", "HIGH"),
                    help="[m] arc along the lane between the cars of a race at spawn, drawn uniformly "
                         "per reset. The default is the range every race so far was trained at, so an "
                         "unflagged run is unchanged. Widening it is an opponent-diversity axis: a "
                         "fixed narrow band shows the policy one approach geometry")
    ap.add_argument("--action-mode", default="direct", choices=["direct", "plan"], help="plan: the policy outputs a local trajectory (f1sim.mpc)")
    ap.add_argument("--scan-deltas", action="store_true", help="append temporal scan differences for a new model without --init")
    ap.add_argument("--temporal-encoder", choices=["cnn", "gru"], default="cnn")
    ap.add_argument("--scan-stem", choices=["plain", "resnet"], default="resnet",
                    help="scan encoder for a new model without --init (an --init checkpoint keeps its own)")
    ap.add_argument("--raceline-margin", type=float, default=None,
                    help="[m] free space the opponents' raceline keeps from the boundary (default 0.40)")
    ap.add_argument("--teacher-grip", choices=["true", "nominal", "conservative"], default="true",
                    help="grip the teacher opponents' speed profile assumes")
    ap.add_argument("--teacher-recover-time", type=float, default=0.0,
                    help="[s] >0: cap the teacher's commanded speed at what can still be steered back onto the lane "
                         "(v <= a_lat * t / heading_error). 0 keeps the old behaviour, where a car facing "
                         "backwards on the line is told to carry full racing speed")
    a = ap.parse_args()
    _gap_lo, _gap_hi = (float(x) for x in a.spawn_gap)
    if not (math.isfinite(_gap_lo) and math.isfinite(_gap_hi)) or not (0.0 < _gap_lo <= _gap_hi):
        raise SystemExit(f"--spawn-gap {_gap_lo} {_gap_hi}: needs finite 0 < LOW <= HIGH [m]. The gap "
                         f"is an arc drawn uniformly from [LOW, HIGH] and subtracted per grid slot, so "
                         f"a non-positive or inverted range spawns cars on top of each other.")
    a.spawn_gap = (_gap_lo, _gap_hi)
    if a.kl_decay is None:
        a.kl_decay = a.total
    device = torch.device(a.device); torch.manual_seed(a.seed)
    names = common.track_names(a.tracks)
    if a.envs % a.race_size:
        raise SystemExit(f"--envs {a.envs} is not a multiple of --race-size {a.race_size}: the envs are "
                         f"dealt into races of that size, so a remainder leaves a race that is never "
                         f"complete. Use {a.envs - a.envs % a.race_size} or {a.envs + a.race_size - a.envs % a.race_size}.")
    if a.overtake_bonus > 0 and a.race_size < 2:
        raise SystemExit("--overtake-bonus needs --race-size > 1: with no opponent there is nothing to pass.")
    need_rl = (a.race_size > 1 and a.opponent in ("teacher", "mixed")) or a.lap_time_bonus > 0
    print(f"loading {len(names)} tracks{' + racelines' if need_rl else ''} ...", flush=True)
    tracks, rls = common.load_tracks(names, racelines=need_rl,
                                     **({} if a.raceline_margin is None else {"margin": a.raceline_margin}))
    # One switch sets both halves of the runtime. They are alternatives, not layers: `compile` and
    # `graphs` both end in a CUDA graph, and leaving `compile_tracker` True while asking for explicit
    # graphs would compile the solver anyway and make the choice a half-measure.
    sim_cfg = Config()
    sim_cfg.sim.compile = (a.sim_backend == "compile")
    _env_compile_tracker = (a.sim_backend == "compile")
    if a.sim_backend != "compile" and device.type != "cuda":
        print(f"--sim-backend {a.sim_backend} on {device.type}: physics/solver compile is off "
              f"(it is CUDA-only anyway)")
    env = common.make_env(tracks, a.envs, device, EnvConfig(speed_cap=a.cap0, reward_collision=-abs(a.collision_penalty),
                                                              reward_steer_rate=a.steer_penalty, reward_proximity=a.proximity_penalty,
                                                              safe_dist=a.safe_dist, reward_wrong_way=a.wrong_way_penalty,
                                                              reward_collision_speed=a.collision_speed_penalty, proximity_speed_ref=a.proximity_speed_ref,
                                                              reward_plan_clearance=a.plan_clearance_penalty, plan_margin=a.plan_margin,
                                                              reward_lap=a.lap_bonus,
                                                              reward_lap_time=a.lap_time_bonus,
                                                              reward_overtake=a.overtake_bonus,
                                                              reward_car_contact=a.car_contact_penalty,
                                                              reward_car_proximity=a.car_proximity_penalty,
                                                              reward_sideslip=a.sideslip_penalty,
                                                              car_safe_gap=a.car_safe_gap,
                                                              max_steps=int(a.episode_s * 40),
                                                              scan_stack=a.scan_stack, scan_stride=a.scan_stride, hist_len=a.hist_len,
                                                              race_size=a.race_size, opponent=a.opponent,
                                                              mixed_teacher_frac=a.mixed_teacher_frac,
                                                              opp_speed_range=tuple(a.opp_speed),
                                                              spawn_gap=tuple(a.spawn_gap), action_mode=a.action_mode,
                                                              compile_tracker=_env_compile_tracker), seed=a.seed, rls=rls,
                          cfg=sim_cfg,
                          teacher_grip=a.teacher_grip,
                          teacher_recover_time=a.teacher_recover_time)
    print(f"sim backend: {a.sim_backend} (cfg.sim.compile={sim_cfg.sim.compile}, "
          f"compile_tracker={_env_compile_tracker})")
    graph_rt = None
    if a.sim_backend == "graphs":
        from .graph_runtime import prepare_graph_runtime
        # On the training thread, which is the one that will replay them.
        graph_rt = prepare_graph_runtime(env)
    # After `prepare_graph_runtime`, never before: that captures `mpc.solve` and assigns
    # `tracker._solver`, so a controller installed earlier is silently overwritten and the run
    # becomes a legacy run wearing another arm's name.
    controller = grip_rt.ControllerRuntime(env, a.controller, a.estimator or None, device=device)
    # `graph_rt` carries the eleven real solver arguments recorded during capture; handing them over
    # lets the controller capture its own graph instead of installing an eager solver on top of a
    # run that just paid to capture a graphed one.
    controller.install(graph_rt=graph_rt)
    controller_on = a.controller != "legacy"
    if controller_on:
        if env.M > 1:
            raise SystemExit(f"--controller {a.controller} is validated solo only; this env has "
                             f"M={env.M} cars per race")
        print(f"controller arm: {a.controller}"
              + (f" | estimator {a.estimator}" if a.estimator else ""))
    spec = common.obs_spec(env)
    obs, info = env.reset(seed=a.seed)
    # After the seeded reset: the histories must start from the observations this run actually saw.
    controller.begin(obs)
    priv = env.privileged(env.last_result); priv_dim = priv.shape[1]
    lid = env.learner_ids                                   # races with teacher opponents: only the learners' data is used
    # Stage-1 conditioning. Both arms carry the same projection; only `--cond` differs.
    cond_spec = cond_mod.CondSpec() if a.cond == "none" else cond_mod.spec_for(a.cond)
    cond_dim = cond_spec.dim
    priv_adapter = a.critic_priv_adapter or None
    # The critic's own width comes from the checkpoint when an adapter reshapes the env's vector to
    # match it; without one it is the env's, as before.
    critic_priv_dim = priv_dim
    if priv_adapter == cond_mod.ADAPTER_ABSENT_OPPONENT:
        if priv_dim != 17:
            raise SystemExit(f"--critic-priv-adapter {priv_adapter} expects a solo env (priv 17), got {priv_dim}")
        critic_priv_dim = 21
        print(f"critic privileged adapter: {priv_adapter} (env {priv_dim} -> critic {critic_priv_dim}, "
              f"opponent slots zeroed); raw privileged storage and priv_mu_index={env.priv_mu_index} unchanged")
    if a.init and cond_dim:
        # By name, and nothing may be left fresh except the conditioning projection itself.
        model, extra, fresh = load_for_conditioning(
            a.init, device, cond_dim, cond_spec.to_meta(), priv_adapter=priv_adapter,
            override={"n_stack": spec.scan_stack, "n_beams": spec.n_beams,
                      "proprio_dim": spec.proprio_dim, "priv_dim": critic_priv_dim, "act_dim": env.act_dim})
        print(f"init from {a.init} as arm '{a.cond}' | {cond_spec.describe()} | fresh: {fresh}")
    elif a.init:
        # `priv_adapter` has to go to the loader as well, not just into the override. Overriding
        # `priv_dim` to 21 alone builds a 21-wide critic and loads the checkpoint's weights into it,
        # but leaves the model with no adapter -- so at the first forward the env still hands it a
        # 17-wide privileged vector. Both conditioning arms already pass it; the unconditional path
        # omitted it, which made `--controller X --critic-priv-adapter ...` (no `--cond`) the one
        # combination that widened the critic without teaching it where the columns went.
        model, extra = load_checkpoint(a.init, device, override={"n_stack": spec.scan_stack, "n_beams": spec.n_beams,
                                                                  "proprio_dim": spec.proprio_dim, "priv_dim": critic_priv_dim, "act_dim": env.act_dim},
                                       allow_conditional=False, priv_adapter=priv_adapter)
        print("init from", a.init, extra.get("metrics"), "| re-initialized:", extra.get("skipped") or "nothing")
        a.scan_deltas = bool(model.meta.get("scan_deltas", False))
        a.temporal_encoder = str(model.meta.get("temporal_encoder", "cnn"))
        a.scan_stem = str(model.meta.get("scan_stem", "plain"))
    else:
        model = ActorCritic(spec.scan_stack, spec.n_beams, spec.proprio_dim, critic_priv_dim, act_dim=env.act_dim,
                            scan_deltas=a.scan_deltas, temporal_encoder=a.temporal_encoder,
                            scan_stem=a.scan_stem, cond_dim=cond_dim,
                            cond=cond_spec.to_meta() if cond_dim else None,
                            priv_adapter=priv_adapter).to(device)
    # the plan default applies to a fresh actor only. Applied on every --init it silently undoes the
    # annealing each resume, and a resume that more than doubles the exploration noise crashes every
    # episode for the next ~40 updates before it claws back to where the checkpoint already was
    init_log_std = a.init_log_std if a.init_log_std is not None else (
        -1.8 if (a.action_mode == "plan" and not a.init) else None)
    if init_log_std is not None:
        with torch.no_grad(): model.actor.log_std.fill_(init_log_std)
        print(f"actor log_std reset to {init_log_std} (std {math.exp(init_log_std):.3f})")
    # The KL leash's reference is the frozen baseline actor. It is deep-copied here, before any
    # update, so its conditioning projection is still exactly zero -- which makes it the
    # *unconditional* policy the checkpoint arrived as, whatever the trained arm later learns to do
    # with `c`. If the reference moved with the conditioning, the two arms would be leashed to
    # different policies and the comparison would be between leashes, not conditioning.
    ref = copy.deepcopy(model.actor).eval()
    for p_ in ref.parameters(): p_.requires_grad_(False)
    if cond_dim and float(ref.cond.weight.abs().max()) != 0.0:
        # RuntimeError, not assert: `python -O` strips asserts, and this one is the only thing
        # standing between the two arms and a KL leash that moved with the conditioning.
        raise RuntimeError("the KL reference actor was captured after the conditioning projection "
                           "had trained; it must be the frozen baseline")
    #: What this run was, recorded in every checkpoint it writes so a result traces back to its arm,
    #: its normalization and its adapter without consulting a shell history.
    experiment_meta = {
        "stage": "stage1_current_mu_utility", "arm": a.cond, "cond": cond_spec.to_meta(),
        "critic_priv_adapter": priv_adapter, "env_priv_dim": int(priv_dim),
        "critic_priv_dim": int(critic_priv_dim), "priv_mu_index": int(env.priv_mu_index),
        "fresh_optimizer": bool(a.fresh_opt), "aux_grip": float(a.aux_grip), "aux_opp": float(a.aux_opp),
        "lab_oracle": bool(cond_spec.lab_oracle), "init": a.init, "seed": int(a.seed),
        "wandb_group": a.wandb_group,
        # The controller this policy was trained against, with the estimator's feature spec,
        # calibration and content hash. A consumer that cannot reproduce this refuses the checkpoint
        # rather than running the plans through a tracker they were not shaped for.
        "controller": controller.checkpoint_meta(),
    }

    def experiment_meta_now():
        """`experiment_meta` with the controller block re-read at save time.

        The controller's `flags_fired` grows during training. Snapshotting it once before the first
        update and reusing that dict means every checkpoint records `flags_fired: []` no matter what
        fired at update 40 -- the one field whose whole purpose is to travel with the result.
        """
        return {**experiment_meta, "controller": controller.checkpoint_meta()}
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    # Adam's moments are part of where training got to. Dropping them on every resume restarts the
    # step-size estimate from nothing, which is the other half of why a resume costs updates before
    # it is back where it was.
    #
    # Per parameter, because a resume is exactly where a layer can change shape: adding opponents
    # widens the critic's privileged input (383 -> 387 for a second car), load_checkpoint
    # re-initialises that weight, and a moment tensor of the old width sails through
    # load_state_dict and then fails inside the first opt.step() with a size mismatch -- past any
    # try/except around the load.
    # `--fresh-opt` skips the restore entirely. Adding the conditioning projection changes the
    # parameter list, and the name re-keying below is exactly the "silently restore by shifted
    # positional order" this experiment must not do: the two arms would start from optimiser state
    # that had been reconstructed differently. Both arms use a fresh Adam, and the run says so.
    if a.fresh_opt:
        print("optimizer: fresh Adam (--fresh-opt); checkpoint moments not restored")
    elif a.cond != "none":
        raise SystemExit("--cond requires --fresh-opt: restoring Adam moments across an added "
                         "parameter would give the arms different optimiser state")
    elif a.init and extra.get("opt"):
        sd = dict(extra["opt"]); st = dict(sd.get("state", {}))
        params = [p_ for g in opt.param_groups for p_ in g["params"]]
        # NOT `names`: that holds the track manifest, and the run config below still needs it
        param_names = [n_ for n_, _ in model.named_parameters()]
        saved_names = extra.get("opt_param_names")
        if saved_names and saved_names != param_names:
            # a layer was added (the grip head): Adam's state is indexed by position, so re-key the
            # saved moments by parameter *name* and leave the new tensors fresh -- a positional load
            # would either refuse (group size) or hand a critic layer the actor's moments
            by_name = {saved_names[i]: entry for i, entry in st.items() if isinstance(i, int) and i < len(saved_names)}
            st = {i: by_name[n_] for i, n_ in enumerate(param_names) if n_ in by_name}
            sd["param_groups"] = [dict(g, params=list(range(len(params)))) for g in sd["param_groups"][:1]]
        dropped = [i for i, entry in st.items()
                   if not (isinstance(i, int) and i < len(params)
                           and all(v.shape == params[i].shape
                                   for v in entry.values() if torch.is_tensor(v) and v.dim() > 0))]
        for i in dropped:
            st.pop(i, None)
        sd["state"] = st
        try:
            opt.load_state_dict(sd)
            print(f"optimizer state restored" + (f" ({len(dropped)} reshaped tensor(s) left fresh)" if dropped else ""))
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"optimizer state not restored ({exc}); starting Adam fresh")
    # Resume the same W&B run across legs. Every --init here carries the whole network over, so the
    # curves belong to one training history: a fresh run per leg restarts the x axis at zero and the
    # only question worth asking of the charts -- is this better than before the change? -- stops
    # being answerable without stitching tabs together by eye.
    steps_base = int(extra.get("total_steps", extra.get("steps", 0)) or 0) if a.init else 0
    if a.steps_base is not None:
        steps_base = int(a.steps_base)          # checkpoints written before total_steps existed only
                                                # carry their own leg's count, so the offset has to be
                                                # given by hand when stitching those onto a run
    wandb_id = (extra.get("wandb_id") if a.init else None) or (a.wandb_id or None)
    if a.wandb_new:
        steps_base, wandb_id = 0, None
    run = common.wandb_init(a.name, vars(a) | {"phase": "ppo", "tracks": names}, group=a.wandb_group, mode=a.wandb,
                            resume_id=wandb_id)
    wandb_id = getattr(run, "id", None) or wandb_id
    if steps_base:
        print(f"continuing W&B run {wandb_id} from {steps_base/1e6:.1f}M steps", flush=True)
    out = common.run_dir(a.name)
    env.sim.warmup()

    T, B = a.horizon, int(lid.numel())
    k, N, P = spec.scan_stack, spec.n_beams, spec.proprio_dim
    buf_scan = torch.zeros(T, B, k, N, device=device, dtype=torch.float16)
    buf_pro = torch.zeros(T, B, P, device=device); buf_priv = torch.zeros(T, B, priv_dim, device=device)
    buf_act = torch.zeros(T, B, env.act_dim, device=device); buf_logp = torch.zeros(T, B, device=device)
    buf_rew = torch.zeros(T, B, device=device); buf_done = torch.zeros(T, B, device=device); buf_trunc = torch.zeros(T, B, device=device)
    buf_val = torch.zeros(T + 1, B, device=device)
    buf_final_val = torch.zeros(T, B, device=device)
    buf_reward_components = torch.zeros(T, B, len(REWARD_COMPONENT_KEYS), device=device)
    # mixed opponents: a teacher-driven car's transitions sit in the buffer (its width is fixed) but
    # carry no weight in the loss -- its actions are the teacher's, not the policy's
    buf_mask = torch.ones(T, B, device=device)
    # The condition the sampled action was actually drawn under. Stored, never recomputed: the ratio
    # PPO clips compares a stored log probability against a recomputed one, so the recomputation has
    # to see the identical input. Reading `P["mu"]` again during SGD would silently use whatever the
    # env has been reset to since -- a different friction for the same transition.
    buf_cond = torch.zeros(T, B, max(1, cond_dim), device=device)

    steps_done = 0; update = 0; t_start = time.time(); last_log = {}; last_ctrl = {}; cap = a.cap0
    t_loop = 0.0; t_loop0 = time.time()        # wall time of the last whole update, for --yield-to-viewer
                                               # (t_upd below is the optimiser's own timing, for the log)
    ep_stats = {"return": [], "progress": [], "collided": [], "lap_time": [], "steps": []}
    collision_history: deque[tuple[float, float]] = deque(maxlen=500)      # (crashed, metres driven)
    # per track, so the gate cannot open on a good average while one map is failing. The set average
    # hid a 12 coll/km track behind a 2.1 coll/km mean once already; a speed curriculum keyed on the
    # mean would raise the cap exactly where the policy is least ready for it.
    track_hist = [deque(maxlen=64) for _ in range(env.sim.track.T)]

    def coll_per_km(h) -> float:
        m = sum(d for _, d in h)
        return 1000.0 * sum(c for c, _ in h) / m if m > 1e-6 else float("inf")

    def gate_rate():
        """(rate the gate reads, how many tracks it was estimated from). Falls back to the whole set
        while no single track has enough distance yet -- and says so, rather than looking per-track."""
        scored = [coll_per_km(h) for h in track_hist if sum(d for _, d in h) >= a.cap_gate_min_km * 1000]
        if scored:
            return float(np.quantile(scored, a.cap_gate_quantile)), len(scored)
        return (coll_per_km(collision_history) if collision_history else float("inf")), 0
    n_updates = int(a.total // (T * B))
    while steps_done < a.total:
        frac = steps_done / a.total
        if a.cap_gate > 0:                                   # gated: the cap climbs at the scheduled rate only while collisions are under the gate
            recent, _ = gate_rate()
            step_cap = (a.cap1 - a.cap0) * (T * B) / a.cap_steps
            cap = min(a.cap1, cap + step_cap) if (recent < a.cap_gate and update > 0) else cap
        else:
            cap = a.cap0 + (a.cap1 - a.cap0) * min(1.0, steps_done / a.cap_steps)
        env.set_speed_cap(cap)
        obs["speed_cap"] = (env.speed_cap / env.ecfg.v_max_policy)[:, None]
        priv = env.privileged(env.last_result)
        kl_coef = a.kl_coef * max(0.0, 1.0 - steps_done / a.kl_decay)
        lr = a.lr + (a.lr_end - a.lr) * frac
        for g in opt.param_groups: g["lr"] = lr
        # Hand the GPU back while someone is watching. A viewer next to a full-rate run measured
        # 1-2 fps and 125 ms per sim step no matter how few cars it drew: the cost was queueing
        # behind this process, not drawing. Sleeping a slice of each update leaves device gaps the
        # viewer can use, and training carries on at a fraction of the rate instead of stopping.
        if a.yield_to_viewer > 0 and t_loop > 0 and common.viewer_active():
            time.sleep(min(t_loop * a.yield_to_viewer, 1.0))
        t_loop0 = time.time()
        tm = common.Timer()
        # ---------------- rollout
        model.eval()
        ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and device.type == "cuda")
        with torch.no_grad():
            for t in range(T):
                scan, pro = flatten_obs(obs)
                # Frozen here, from the privileged vector belonging to THIS observation, before the
                # step advances the env or a reset re-draws `mu`.
                cond_t = (cond_mod.make_condition(a.cond, cond_spec, priv, env.priv_mu_index)
                          if cond_dim else None)
                # Pre-action, and before the policy is asked for anything: the friction this step's
                # plan will be tracked under has to be in the MPC before the plan exists.
                # The truth handed over here is for B3's safety metrics only and is read from the
                # privileged vector belonging to THIS observation, by this loop -- never from inside
                # the controller, whose estimated path has to work on a car that has no `P`.
                if controller_on:
                    controller.observe_truth(priv[:, env.priv_mu_index])
                controller.pre_action(obs)
                with ac:
                    act, logp = sample_rollout_action(model, scan, pro, cond_t)
                    val = model.critic(scan[lid], pro[lid], priv[lid]).float()
                act, logp = act.float(), logp.float()
                buf_scan[t] = scan[lid].half(); buf_pro[t] = pro[lid]; buf_priv[t] = priv[lid]; buf_act[t] = act[lid]; buf_logp[t] = logp[lid]; buf_val[t] = val
                if cond_dim:
                    buf_cond[t] = cond_t[lid].float()
                obs, rew, term, trunc, info = env.step(act.clamp(-1, 1))
                # Immediately: the issued command has to be taken from the spy's pre-reset snapshot,
                # and the episode boundaries recorded, before anything else advances the env.
                controller.post_step(term, trunc)
                priv = env.privileged(env.last_result)
                buf_rew[t] = rew[lid]; buf_done[t] = term[lid].float(); buf_trunc[t] = trunc[lid].float()
                if "on_policy" in info:
                    buf_mask[t] = info["on_policy"][lid].float()
                buf_reward_components[t] = torch.stack([info["reward_components"][key][lid] for key in REWARD_COMPONENT_KEYS], 1)
                ep_stats["lap_time"] += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
                buf_final_val[t].zero_()
                if "final" in info:
                    f = info["final"]; m = env.learner[f["ids"]]
                    final_scan, final_pro = flatten_obs(info["final_obs"])
                    with ac:
                        final_val = model.critic(final_scan, final_pro, info["final_priv"]).float()
                    all_final_val = torch.zeros(env.B, device=device)
                    all_final_val[f["ids"]] = final_val
                    buf_final_val[t] = all_final_val[lid]
                    crashed_l = f["collided"][m].float().tolist(); dist_l = f["progress"][m].tolist()
                    collision_history.extend(zip(crashed_l, dist_l))
                    # info["track_id"] is the pre-reset snapshot. env.sim.tid has already been
                    # re-drawn for these rows by the auto-reset inside env.step, so reading it here
                    # would score the finished episode against the track it never drove on.
                    ended_tid = info["track_id"][f["ids"]][m].tolist()
                    for tid_, crashed, dist in zip(ended_tid, crashed_l, dist_l):
                        track_hist[tid_].append((crashed, dist))
                    ep_stats["return"] += f["return"][m].tolist(); ep_stats["progress"] += f["progress"][m].tolist()
                    ep_stats["collided"] += f["collided"][m].float().tolist(); ep_stats["steps"] += f["steps"][m].tolist()
            scan, pro = flatten_obs(obs)
            with ac:
                buf_val[T] = model.critic(scan[lid], pro[lid], priv[lid]).float()
            adv = compute_gae(buf_rew, buf_val, buf_done, buf_trunc, buf_final_val, a.gamma, a.lam)
            ret = adv + buf_val[:T]
        t_roll = tm.lap()
        # ---------------- update
        model.train()
        n = T * B
        f_scan = buf_scan.reshape(n, k, N); f_pro = buf_pro.reshape(n, P); f_priv = buf_priv.reshape(n, priv_dim)
        f_act = buf_act.reshape(n, env.act_dim); f_logp = buf_logp.reshape(n); f_adv = adv.reshape(n); f_ret = ret.reshape(n); f_val = buf_val[:T].reshape(n)
        f_mask = buf_mask.reshape(n)
        f_cond = buf_cond.reshape(n, buf_cond.shape[-1]) if cond_dim else None
        # advantage statistics over the policy's own samples only; a teacher-driven car's advantages
        # are not the policy's and would otherwise set the scale everything else is normalised by
        w_all = f_mask / f_mask.sum().clamp_min(1.0)
        adv_mean = (f_adv * w_all).sum(); adv_std = ((f_adv - adv_mean) ** 2 * w_all).sum().sqrt()
        f_adv = (f_adv - adv_mean) / (adv_std + 1e-8)
        def wmean(x, w):
            return (x * w).sum() / w.sum().clamp_min(1.0)
        stats = {"pg": [], "vf": [], "ent": [], "kl_ref": [], "approx_kl": [], "clipfrac": []}
        freeze_actor = update < a.critic_warmup
        for ep in range(a.epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, a.minibatch):
                idx = perm[i:i + a.minibatch]
                scan = f_scan[idx].float(); pro = f_pro[idx]
                c_mb = f_cond[idx] if cond_dim else None       # the stored one, never recomputed
                with ac:
                    logp, ent, val, d, grip, opp_pred = model.evaluate_aux(scan, pro, f_priv[idx], f_act[idx], c_mb)
                logp, ent, val = logp.float(), ent.float(), val.float()
                w = f_mask[idx]
                aux = torch.zeros((), device=device)
                if a.aux_grip > 0:
                    mu_true = f_priv[idx][:, env.priv_mu_index] - 1.0
                    aux = wmean((grip - mu_true) ** 2, w)
                aux_o = torch.zeros((), device=device)
                if a.aux_opp > 0 and env.M > 1:
                    # priv[8:11] = nearest opponent (ahead offset, side offset, longitudinal speed
                    # difference), scaled to O(1). NOTE the third channel is other.vx - ego.vx, a
                    # difference of two body-frame longitudinal speeds -- NOT the line-of-sight
                    # closing speed (d/dt of the distance) that car_proximity_penalty computes. The
                    # name is kept honest here; changing what the channel *means* would silently
                    # reinterpret every checkpoint trained against it, so that is a separate change.
                    o_true = f_priv[idx][:, 8:11] / torch.tensor([3.0, 1.0, 2.0], device=device)
                    # priv[:, 11] is dist/PRIV_OPP_DIST_SCALE (gym_env owns that scale), so the comparison
                    # has to be made in metres. Comparing the scaled column against 6.0 directly
                    # selected 30 m -- three times the LiDAR's range, i.e. a mask that removed
                    # nothing and trained the head on an empty road.
                    near = (f_priv[idx][:, 11] * PRIV_OPP_DIST_SCALE < AUX_OPP_RANGE_M).to(w.dtype) * w
                    aux_o = wmean(((opp_pred - o_true) ** 2).mean(1), near)
                ratio = (logp - f_logp[idx]).exp()
                pg = -wmean(torch.min(ratio * f_adv[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * f_adv[idx]), w)
                v_clipped = f_val[idx] + (val - f_val[idx]).clamp(-a.clip, a.clip)
                vf = 0.5 * wmean(torch.max((val - f_ret[idx]) ** 2, (v_clipped - f_ret[idx]) ** 2), w)
                with torch.no_grad(), ac:
                    # The reference is fed the same condition, and its projection is still zero,
                    # so it evaluates as the frozen unconditional baseline. Passing `c` keeps the
                    # call valid for a conditional actor without letting the leash move with it.
                    d_ref = ref.dist(scan, pro, c_mb)
                d_ref = torch.distributions.Normal(d_ref.mean.float(), d_ref.stddev.float())
                d = torch.distributions.Normal(d.mean.float(), d.stddev.float())
                kl_ref = wmean(torch.distributions.kl_divergence(d_ref, d).sum(1), w)
                loss = a.vf * vf + (0.0 if freeze_actor else 1.0) * (pg - a.ent * wmean(ent, w) + kl_coef * kl_ref) + a.aux_grip * aux + a.aux_opp * aux_o
                opt.zero_grad()
                if cond_dim or controller_on:
                    # Scoped to the conditioning and controller arms so the legacy path keeps its
                    # exact behaviour.
                    # Without these, "the run finished" says nothing about whether a loss or a
                    # gradient ever went non-finite -- clip_grad_norm_ silently propagates a NaN, and
                    # a vacuously-empty loss dict looked like a passing finiteness check in the first
                    # smoke. With them, a completed run IS the evidence: every step's loss and total
                    # gradient norm were finite, or it stopped here.
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss at update {update} "
                                           f"(pg={float(pg):.4g} vf={float(vf):.4g} "
                                           f"kl_ref={float(kl_ref):.4g})")
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad,
                                                    error_if_nonfinite=bool(cond_dim or controller_on))
                opt.step()
                with torch.no_grad():
                    stats["pg"].append(pg.item()); stats["vf"].append(vf.item()); stats["ent"].append(ent.mean().item())
                    stats["kl_ref"].append(kl_ref.item()); stats["approx_kl"].append(((ratio - 1) - (logp - f_logp[idx])).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
                    if a.aux_grip > 0: stats.setdefault("aux_grip", []).append(aux.item())
                    if a.aux_opp > 0 and env.M > 1: stats.setdefault("aux_opp", []).append(aux_o.item())
        t_upd = tm.lap()
        steps_done += n; update += 1
        # B3, drained EVERY update, before the logging branch. "Five consecutive updates" is a
        # property of training, not of the logging cadence: evaluating it inside `log_every` would
        # make `--log-every 5` silently mean five consecutive *logged* updates, i.e. twenty-five real
        # ones, and would discard every intervening update's transitions from the evidence window.
        # Only publication below depends on the cadence. A fired flag prints immediately -- waiting
        # for a logging tick to report it is the same mistake in a smaller place.
        ctrl_log, ctrl_fired = controller.collect_metrics()
        if ctrl_log:
            # Into the checkpoint too, not only into W&B. `last_log` carries episode stats only, so
            # a `--wandb disabled` run would otherwise keep no record of what the controller did --
            # and B3's fallback/q-gap/used-mu evidence is exactly what has to travel with the result.
            last_ctrl = ctrl_log
        for flag in ctrl_fired:
            print(f"CONTROLLER FLAG '{flag}' at update {update}: "
                  f"{ {k: round(v, 4) for k, v in ctrl_log.items() if isinstance(v, float)} }. "
                  f"The run continues and nothing is retuned -- diagnose in a versioned iteration.",
                  flush=True)
        # ---------------- logging
        if update % a.log_every == 0 or update == 1:
            n_ep = len(ep_stats["return"])
            log = {"steps": steps_base + steps_done, "update": update, "speed_cap": cap, "kl_coef": kl_coef, "lr": lr,
                   "rollout/reward_per_step": buf_rew.mean().item(), "rollout/mean_speed": env.sim.state[:, 3].mean().item(),
                   "rollout/value_mean": buf_val[:T].mean().item(), "rollout/adv_std": adv.std().item(),
                   "loss/pg": np.mean(stats["pg"]), "loss/vf": np.mean(stats["vf"]), "loss/entropy": np.mean(stats["ent"]),
                   "loss/kl_ref": np.mean(stats["kl_ref"]), "loss/approx_kl": np.mean(stats["approx_kl"]), "loss/clipfrac": np.mean(stats["clipfrac"]),
                   **({"loss/aux_grip_mse": float(np.mean(stats["aux_grip"]))} if stats.get("aux_grip") else {}),
                   **({"loss/aux_opp_mse": float(np.mean(stats["aux_opp"]))} if stats.get("aux_opp") else {}),
                   "policy/log_std_steer": model.actor.log_std[0].item(), "policy/log_std_speed": model.actor.log_std[1].item(),
                   "time/rollout_s": t_roll, "time/update_s": t_upd, "time/env_steps_per_s": n / (t_roll + t_upd),
                   "time/elapsed_min": (time.time() - t_start) / 60}
            # over the policy's own cars only. With mixed opponents the teacher-driven car of a race
            # sits in the buffer too, and its overtake reward is the mirror image of the learner's,
            # so an unweighted mean of the pair reads ~0 no matter how much passing is going on.
            m_ = buf_mask[..., None]
            log.update({f"reward/{key}_per_step": ((buf_reward_components[:, :, index:index + 1] * m_).sum()
                                                   / m_.sum().clamp_min(1.0)).item()
                        for index, key in enumerate(REWARD_COMPONENT_KEYS)})
            if a.cap_gate > 0:
                # logged every time, including while it is still reading the whole set: a gate that
                # logs nothing until it has per-track data looks identical to a gate that is not
                # running, which is how this one sat dead for 34M steps without showing it
                rate, scored = gate_rate()
                per_track = [coll_per_km(h) for h in track_hist if sum(d for _, d in h) >= a.cap_gate_min_km * 1000]
                log.update({"curriculum/gate_coll_per_km": rate, "curriculum/tracks_scored": scored,
                            "curriculum/gate_open": float(rate < a.cap_gate)})
                if per_track:
                    log.update({"curriculum/worst_track_coll_per_km": max(per_track),
                                "curriculum/mean_track_coll_per_km": float(np.mean(per_track))})
            if n_ep:
                last_log = {"collision_rate": float(np.mean(ep_stats["collided"])), "progress_m": float(np.mean(ep_stats["progress"])),
                            "collisions_per_km": 1000.0 * float(np.sum(ep_stats["collided"])) / max(float(np.sum(ep_stats["progress"])), 1e-6),
                            "lap_time_s": float(np.mean(ep_stats["lap_time"])) if ep_stats["lap_time"] else float("nan")}
                dist = float(np.sum(ep_stats["progress"]))
                log.update({"episode/return": np.mean(ep_stats["return"]), "episode/progress_m": np.mean(ep_stats["progress"]),
                            "episode/collision_rate": np.mean(ep_stats["collided"]), "episode/len_steps": np.mean(ep_stats["steps"]),
                            # the hazard rate per metre driven. collision_rate saturates at 1.0 once
                            # episodes are long enough to almost always contain a crash, and stays
                            # there while the policy goes on getting better
                            "episode/collisions_per_km": 1000.0 * float(np.sum(ep_stats["collided"])) / max(dist, 1e-6),
                            "episode/count": n_ep})
                if ep_stats["lap_time"]:
                    laps = np.asarray(ep_stats["lap_time"], dtype=float)
                    # the mean alone hides the shape: a policy that gets faster on its good laps while
                    # its bad laps get worse looks flat, and the best lap is what a racing line is judged on
                    log.update({"episode/lap_time_s": float(laps.mean()), "episode/lap_time_best_s": float(laps.min()),
                                "episode/lap_time_p10_s": float(np.percentile(laps, 10)),
                                "episode/lap_time_median_s": float(np.median(laps)),
                                "episode/lap_time_p90_s": float(np.percentile(laps, 90)),
                                "episode/laps_completed": int(laps.size)})
                ep_stats = {kk: [] for kk in ep_stats}
            # Publication only -- the drain and the flag evaluation already happened above.
            log.update(ctrl_log)
            # the W&B step must be the cumulative count, not this leg's: a resume that logs from 0 again
            # has every point silently dropped ("step less than current step"), so the leg vanishes
            run.log(log, step=steps_base + steps_done)
            print(f"upd {update}/{n_updates} steps {(steps_base + steps_done)/1e6:.1f}M cap {cap:.1f} | rew/step {log['rollout/reward_per_step']:.3f} "
                  f"coll {log.get('episode/collisions_per_km', float('nan')):.1f}/km prog {log.get('episode/progress_m', float('nan')):.0f} m "
                  f"lap {log.get('episode/lap_time_s', float('nan')):.1f} s | gate {log.get('curriculum/gate_coll_per_km', float('nan')):.1f} "
                  f"({log.get('curriculum/tracks_scored', 0):.0f} tk) | kl_ref {log['loss/kl_ref']:.3f} | {log['time/env_steps_per_s']:.0f} steps/s", flush=True)
        t_loop = time.time() - t_loop0
        if update % a.save_every == 0:
            meta = {"spec": spec.__dict__, "phase": "ppo", "run": a.name, "update": update, "updates": n_updates,
                    "steps": steps_done, "total_steps": steps_base + steps_done, "wandb_id": wandb_id,
                    "cap": cap, "action_mode": a.action_mode, "metrics": {**last_log, **last_ctrl},
                    "opt": opt.state_dict(), "opt_param_names": [n_ for n_, _ in model.named_parameters()],
                    "experiment": experiment_meta_now()}
            save_checkpoint(os.path.join(out, f"ppo_u{update}.pt"), model, meta)
            save_checkpoint(os.path.join(out, "ppo_latest.pt"), model, meta)
    save_checkpoint(os.path.join(out, "ppo_final.pt"), model,
                    {"spec": spec.__dict__, "phase": "ppo", "run": a.name, "update": update, "updates": n_updates,
                     "steps": steps_done, "total_steps": steps_base + steps_done, "wandb_id": wandb_id,
                     "cap": cap, "action_mode": a.action_mode, "metrics": {**last_log, **last_ctrl},
                     "opt": opt.state_dict(), "opt_param_names": [n_ for n_, _ in model.named_parameters()],
                     "experiment": experiment_meta_now()})
    # Unconditionally, and before the graph runtime: the controller installs a solver hook and a
    # wrapper around `env._reset_envs` whatever the sim backend is, so releasing it only when graphs
    # were captured leaves both in place on the default `compile` backend.
    controller.release()
    if graph_rt is not None:
        from .graph_runtime import release_graph_runtime
        release_graph_runtime(graph_rt)         # puts sim._roll and the tracker's solver back
    run.finish()


if __name__ == "__main__":
    main()
