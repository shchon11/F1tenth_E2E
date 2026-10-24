"""PPO for the v2 actor-critic (`learn.model_v2`): `python -m f1sim.learn.ppo --arch v2 ...`.

docs/research/architecture-redesign-2026-10-13.md section 5 as corrected by appendix 2, and the
decisions that are not reopened here:

* **No KL term** (c7b1b10, the user's decision): `kl_coef` is 0 and a v2 run refuses another value.
  `loss/kl_ref` is still logged -- KL(frozen start actor || current) per step, the start actor
  walked over the rollout with ITS OWN recurrent state (critique B4: v1's reference was evaluated
  with the memory switched off) -- as a drift monitor, never a gradient.
* **Actor and critic clipped separately** (`--clip-split`, 8786cc0: one norm over both let a critic
  norm of 1.6k-31k set the actor's clip factor) with Adam eps 1e-8, each with its own learning rate
  (actor 2e-5 -> 1e-5, critic 3e-4, the redesign's numbers).
* **No value clip** (`--vf-clip 0`, e39b34c) and **value normalisation** (`train_v2.ValueNorm`): the
  critic learns normalised returns, the GAE reads de-normalised values, and every statistics update
  rescales the value head's last layer (PopArt) so the critic's de-normalised values survive it --
  checked live on the log updates (`value_norm/popart_max_dev`).
* **The auxiliary balance** (`train_v2.AuxBalance`, main loss = the policy gradient): every
  `--aux-balance-every` (10) updates the per-loss trunk gradient norms set each auxiliary's
  multiplier so it pulls the shared trunk at most `--aux-balance-rho` (0.5) x the policy gradient.
  UNIT 4's smoke measured B8 at 4.5x and the nearest-car head at 25x the policy gradient's pull.
  The measurement is one backward pass per loss on the update's first piece -- the piece the
  per-loss norms were already logged from.
* **One value stream plus "car contact within 1 s"** (G4): a BCE head on the critic, its labels built
  from the rollout buffer (`train_v2.contact_labels`: charged car-contact onsets in the next 40
  steps, masked where that second crosses the chunk end or a truncation).
* **The critic reads `privileged_v2`** (the explicit geometry included, +0.049 AUC), never the
  actor's features; the truncation bootstrap reads `info["final_priv_v2"]`.

The rollout carries the actor's GRU state per env and stores the state each env enters the chunk
with; the update replays every env's whole chunk from it (truncated BPTT over the horizon, as v1's
recurrent PPO). B8 trains on those replays with the stored-state window (`train_v2.b8_sequence_loss`:
a stored state has been carried since its episode began). The per-dimension `log_std` is the
model's own (B6: kappa_1, kappa_2 at -2.07, the rest -1.8; DAgger never moves it).

Logged besides v1's numbers: each loss's gradient norm on the trunk it shares (actor: pg, entropy,
B8, the other heads; critic: value, contact), the actor's relative movement per update (G5's
<= 2e-3), the share of speed-capped steps (critique B5: the cap is not observed and the decode clips
at it), the value normaliser's mean / std, the contact labels' positive rate, coverage and AUC.

The boot camp's race (UNIT 5b, 2026-10-14):

* **The league** (`--league-slots`): the v1 checkpoints of the boot camp's seats drive in their own
  observation contract and plan linear (`learn.seat_obs`, per-entry formats), and every
  `--league-every` updates the learner's own v2 snapshot takes a seat -- read in wp4, shown the env's
  own observation. Scripted seats plan linear (`--opponent-speed-mode`, the measured default).
* **The speed cap is the --cap0/--cap1 curriculum's** (a self-play front car's own below it), not
  observed by the actor; the share of learner steps whose plan it clipped is logged
  (`rollout/speed_capped_frac`).

Removed 2026-10-14: the learner's unobserved random cap (`--learner-cap-range` / `--learner-cap-prob`,
U(3, 9) m/s per episode, `rollout/learner_cap_mean`) and with it the critic's speed-cap column
(`privileged_v2` is 33 wide). The cap was a guess -- that a speed history under 9 m/s is out of
distribution at the car's 4 m/s (critique B5) -- that nothing measured; see `learn.dagger_v2`.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..gym_env import PRIV_V2_DIM, REWARD_COMPONENT_KEYS
from . import common
from . import conditioning as cond_mod
from .aux_v2 import aux_spec, nearest_car_loss
from .dagger_v2 import (AUX_TERMS, PERCEPTION_AUX_DEFAULTS, apply_race_defaults, check_balance_args, explicit_flags,
                        make_balance, resolve_ego_actions)
from .memory import reset_hidden
from .model import load_checkpoint, save_checkpoint
from .model_v2 import ARCH, ActorCriticV2, init_stems_from_v1
from .obs import ObsSpecV2
from .returns import compute_gae
from .train_v2 import (BALANCE_BETA, BALANCE_EVERY, BALANCE_RHO, CONTACT_HORIZON, ValueNorm,
                       actor_trunk_parameters, aux_heldout_metrics, b8_sequence_loss, binary_auc, contact_labels,
                       contact_onsets, critic_trunk_parameters, deployed_actor_parameters, flat_parameters,
                       loss_grad_norms, relative_movement, select_tracks)

#: v1 flags whose v2 default differs, applied when the command line does not set them: the
#: decisions above, and the redesign's optimiser section (actor lr 2e-5 -> 1e-5, horizon 128,
#: minibatch 4096 = 32 env chunks of 128 in 4 accumulated pieces).
V2_DEFAULTS = {"clip_split": True, "adam_eps": 1e-8, "vf_clip": 0.0, "kl_coef": 0.0,
               "lr": 2e-5, "lr_end": 1e-5, "horizon": 128, "minibatch": 4096, "grad_accum": 4}
#: The v2-only flags' defaults (None on the parser, so a v1 run can refuse them). The perception
#: auxiliaries are the perception gate's decision (1 frame, all three heads on, the gate's coefficients
#: before the adaptive balance; its numbers at `dagger_v2.PERCEPTION_AUX_DEFAULTS`).
V2_ONLY_DEFAULTS = {"v2_frames": 1, "v2_init_stem": "", "critic_lr": 3e-4, "value_norm_beta": 0.99,
                    "contact_bce": 1.0, **PERCEPTION_AUX_DEFAULTS,
                    "aux_balance_every": BALANCE_EVERY, "aux_balance_rho": BALANCE_RHO,
                    "aux_balance_beta": BALANCE_BETA}
#: Rows x steps of the rollout the PopArt check re-evaluates on a log update: two critic forwards of
#: this many rows, a few milliseconds beside an update of the whole buffer.
POPART_CHECK_ROWS = 32
POPART_CHECK_STEPS = 4
#: v1 options a v2 run cannot honour, at the value that means "not asked for".
V2_REFUSED = {"memory": "off", "scan_channels": "", "motion_memory": False, "aux_opp": 0.0, "aux_grip": 0.0,
              "aux_future": 0.0, "aux_floor": 0.0, "aux_opp_mask": 0.0, "aux_motion": 0.0, "opp_token": "off",
              "adaptation": "off", "algo": "ppo", "critic_priv_adapter": "", "maneuver_temp": None,
              "init_log_std": None, "fresh_lr_mult": 1.0, "name_seed_fresh": False,
              "scan_deltas": False, "hist_len": 0, "controller": "legacy", "cap_gate": 0.0,
              "reference": "", "research_estimator": False, "estimator": ""}


def prepare_args(ap: argparse.ArgumentParser, a, argv=None) -> dict:
    """Make `a` a v2 run before anything is built (see `dagger_v2.prepare_args`). Returns the v2
    defaults it applied."""
    given = explicit_flags(ap, argv)
    for k, v in V2_ONLY_DEFAULTS.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    bad = [k for k, off in V2_REFUSED.items() if getattr(a, k) != off]
    if bad:
        raise SystemExit(f"--arch v2: {', '.join('--' + k.replace('_', '-') for k in bad)} are v1's (its network, "
                         f"observation, controller arms or curriculum gate); not wired for the v2 actor")
    if "kl_coef" in given and float(a.kl_coef) != 0.0:
        raise SystemExit("--arch v2 has no KL leash (c7b1b10, the user's decision): kl_ref is logged as a "
                         "drift monitor only. Drop --kl-coef")
    if "action_mode" in given and a.action_mode != "plan":
        raise SystemExit("--arch v2 is the plan policy (its ego history holds WP4 plan actions)")
    if a.speed_mode not in (None, "wp4"):
        raise SystemExit("--arch v2 speaks WP4 (output-format-2026-10-14); --speed-mode is v1's")
    if a.cond not in ("none", "dial"):
        raise SystemExit("--arch v2 reads the grip dial (a deployable input column); no other conditioning")
    if a.init and a.v2_init_stem:
        raise SystemExit("--init continues a v2 policy; --v2-init-stem seeds a fresh one from a v1 stem")
    a.shield_aux = float(a.shield_aux or 0.0)
    if a.shield_aux < 0 or a.shield_brake <= 0 or a.shield_path_width <= 0 or a.shield_lag < 0 or a.shield_gap_min < 0:
        raise SystemExit("--shield-aux >= 0, --shield-brake / --shield-path-width > 0, --shield-lag / --shield-gap-min >= 0")
    if min(a.contact_bce, a.aux_b8, a.aux_near_car) < 0 or not 0 < a.value_norm_beta < 1:
        raise SystemExit("v2 loss weights are >= 0 and --value-norm-beta is in (0, 1)")
    check_balance_args(a)
    applied = apply_race_defaults(a)
    for k, v in V2_DEFAULTS.items():
        if k not in given:
            setattr(a, k, v)
            applied[k] = v
    a.action_mode, a.speed_mode, a.cond = "plan", "wp4", "dial"
    a.scan_stack, a.scan_stride = int(a.v2_frames), 1
    a.hist_len = ObsSpecV2().hist_len
    if a.minibatch < a.horizon or (a.minibatch // a.horizon) % a.grad_accum:
        raise SystemExit(f"--arch v2 updates whole env chunks of the horizon: --minibatch {a.minibatch} must hold "
                         f">= 1 chunk of {a.horizon} and its chunk count must divide by --grad-accum {a.grad_accum}")
    return applied


def env_kwargs(a) -> dict:
    """What a v2 run adds to the v1 EnvConfig fields `ppo.main` passes (the opponents' format travels
    with the opponent flags, `opponent_config.env_kwargs`). The ego vector is the one `--init`'s
    network reads -- with or without the issued actions (`ObsSpecV2.ego_actions`), the contract its
    DAgger run chose -- and the contract default for a fresh network."""
    return {"obs_version": 2, "hist_stride": ObsSpecV2().hist_stride,
            "opp_visibility": bool(a.aux_b8 > 0 or a.aux_near_car > 0),
            "ego_actions": resolve_ego_actions(a)}


def _per_km(count, dist_m: float) -> float:
    dist_m = float(dist_m)
    return 1000.0 * float(count) / dist_m if dist_m > 1.0 else float("nan")


def safe_speed(env, brake: float, width: float, lag: float, gap_min: float) -> torch.Tensor:
    """(B,) [m/s] the speed each car can still bring down to the speed of the car in its path ahead before the gap
    closes -- v_opp + sqrt(2 brake (gap - gap_min - (v_me - v_opp)_+ lag)_+) -- for the nearest such car of its race,
    inf where none is in the path (wholly ahead: body gap > 0.1 m along our heading, within 6 m, |y| < width). True states: `--shield-aux`'s target,
    training only."""
    from .. import dynamics as dyn
    st = env.sim.state
    o = env.sim.other_idx                                                   # (B, C)
    c, s_ = torch.cos(st[:, dyn.IYAW])[:, None], torch.sin(st[:, dyn.IYAW])[:, None]
    dxw, dyw = st[o, 0] - st[:, None, 0], st[o, 1] - st[:, None, 1]
    dx, dy = c * dxw + s_ * dyw, -s_ * dxw + c * dyw
    co, so = torch.cos(st[o, dyn.IYAW]), torch.sin(st[o, dyn.IYAW])
    vox = st[o, dyn.IVX] * co - st[o, dyn.IVY] * so
    voy = st[o, dyn.IVX] * so + st[o, dyn.IVY] * co
    v_opp = (vox * c + voy * s_).clamp_min(0.0)
    v_me = st[:, dyn.IVX][:, None]
    gap = (dx - float(env.ecfg.car_len)).clamp_min(0.0)
    room = (gap - gap_min - (v_me - v_opp).clamp_min(0.0) * lag).clamp_min(0.0)
    vs = v_opp + torch.sqrt(2.0 * brake * room)
    # clearly AHEAD: no longitudinal overlap. A car alongside (gap 0) is not "in our path": capping us to its speed there
    # keeps us stuck beside it mid-pass -- measured with the oracle shield (10-24): car contacts 9.0 -> 15.6 /km
    inpath = (gap > 0.1) & (dx < 6.0) & (dy.abs() < width)
    if getattr(env.sim, "absent_any", False):
        inpath = inpath & ~env.sim.absent[o]
    return torch.where(inpath, vs, torch.full_like(vs, float("inf"))).min(1).values


def train(a, *, env, obs, device, names, init_ck: dict, controller, graph_rt, applied: dict,
          league_seats=()):
    """The PPO loop for `--arch v2`, from the env `ppo.main` built and reset. `league_seats`: the pool
    entries `--league-slots` named, which the learner's own snapshots take over in turn."""
    from .ppo import lagrange_step, parse_lagrange, severity_onsets, severity_rates, SEVERITY_LOG
    spec = common.obs_spec(env)
    if not isinstance(spec, ObsSpecV2):
        raise SystemExit("--arch v2 needs the env built with obs_version 2")
    lid = env.learner_ids
    T, B = int(a.horizon), int(lid.numel())
    coefs = {"contact": float(a.contact_bce), "b8": float(a.aux_b8), "near": float(a.aux_near_car)}
    aux = aux_spec(b8=coefs["b8"] > 0, near_car=coefs["near"] > 0)
    priv_dim = int(env.privileged_v2(env.last_result).shape[1])
    if priv_dim != PRIV_V2_DIM:
        raise SystemExit(f"privileged_v2 is {priv_dim} wide, the v2 critic reads {PRIV_V2_DIM}")
    # ---------------------------------------------------------------- the network
    init_extra = dict(init_ck.get("extra") or {})
    if a.init:
        meta = dict(init_ck.get("meta") or {})
        if meta.get("arch") != ARCH:
            raise SystemExit(f"--init {a.init} is not a v2 checkpoint (a v1 stem seeds a fresh v2 policy through "
                             f"--v2-init-stem)")
        have = dict(meta.get("aux") or {})
        want = {k: bool(have.get(k)) or bool(v) for k, v in aux.items()}
        model, extra = load_checkpoint(a.init, device, override={"aux": want})
        fresh = [k for k, v in want.items() if v and not have.get(k)]
        print(f"init from {a.init} ({extra.get('phase')}, iter/update {extra.get('iter', extra.get('update'))}) "
              f"| heads {want}{' fresh: ' + str(fresh) if fresh else ''}", flush=True)
        if model.obs_spec != spec:
            raise SystemExit(f"--init's observation contract {model.obs_spec} is not this env's {spec}")
    else:
        model = ActorCriticV2(n_beams=spec.n_beams, n_frames=spec.n_frames, ego_dim=spec.ego_dim,
                              act_dim=spec.act_dim, speed_mode=spec.speed_mode, priv_dim=priv_dim,
                              obs=dict(vars(spec)), aux=aux).to(device)
        if a.v2_init_stem:
            rep = init_stems_from_v1(model, a.v2_init_stem, device)
            print(f"v2 stems seeded from {os.path.basename(a.v2_init_stem)}: actor {len(rep['actor'])} / "
                  f"critic {len(rep['critic'])} tensors", flush=True)
    has = {"b8": model.actor.b8 is not None, "near": model.actor.near_car is not None}
    on = {k: coefs[k] > 0 and has[k] for k in ("b8", "near")}
    for k in ("b8", "near"):
        if has[k] and not on[k]:
            print(f"head {k}: PRESENT BUT NOT TRAINED (its weight is 0)", flush=True)
    vnorm = ValueNorm(a.value_norm_beta)
    if init_extra.get("value_norm") and init_extra.get("phase") == "ppo":
        vnorm.load_state_dict(init_extra["value_norm"])
        print(f"value normaliser restored: mean {vnorm.mean:.3f} std {vnorm.std:.3f}", flush=True)
    # The auxiliary balance, against the policy gradient. A DAgger student's balance is not carried
    # over: it was measured against the imitation loss, a different main term.
    balance = make_balance(a, "pg", [k for k in AUX_TERMS if on[k]])
    if init_extra.get("aux_balance") and init_extra.get("phase") == "ppo":
        balance.load_state_dict(init_extra["aux_balance"])
        print(f"auxiliary balance restored: {balance.scale} after {balance.measurements} measurements", flush=True)
    # The drift monitor's reference: the actor this leg starts from, frozen, walked with its own state.
    ref = copy.deepcopy(model.actor).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref_std = ref.log_std.exp().float()
    # ---------------------------------------------------------------- optimiser
    named = list(model.named_parameters())
    actor_np = [(n, p) for n, p in named if not n.startswith("critic.")]
    critic_np = [(n, p) for n, p in named if n.startswith("critic.")]
    actor_params, critic_params = [p for _, p in actor_np], [p for _, p in critic_np]
    opt = torch.optim.Adam([{"params": actor_params, "lr": a.lr, "group": "actor"},
                            {"params": critic_params, "lr": a.critic_lr, "group": "critic"}], eps=a.adam_eps)
    opt_param_names = [n for n, _ in actor_np + critic_np]
    if (not a.fresh_opt and init_extra.get("opt") and init_extra.get("phase") == "ppo"
            and init_extra.get("opt_param_names") == opt_param_names):
        try:
            opt.load_state_dict(init_extra["opt"])
            print("optimizer state restored", flush=True)
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"optimizer state not restored ({exc}); starting Adam fresh", flush=True)
    for g in opt.param_groups:
        g["eps"] = a.adam_eps
    clip_sets = (actor_params, critic_params) if a.clip_split else None
    actor_trunk, critic_trunk = actor_trunk_parameters(model.actor), critic_trunk_parameters(model.critic)
    deployed = deployed_actor_parameters(model.actor)
    # ---------------------------------------------------------------- the run's records
    reward_meta = {k: getattr(env.ecfg, k) for k in (
        "reward_progress", "reward_collision", "reward_collision_speed", "reward_steer_rate", "reward_proximity",
        "safe_dist", "reward_plan_clearance", "plan_margin", "reward_wrong_way", "reward_lap", "reward_lap_time",
        "reward_alive", "reward_car_contact", "reward_overtake", "reward_car_proximity", "car_safe_gap",
        "reward_sideslip", "reward_ttc", "ttc_safe", "reward_overtake_hold", "overtake_hold_dist",
        "overtake_hold_time", "contact_severity_min")}
    experiment = {"arch": ARCH, "kl_reference": "leg_start_actor_recurrent (monitor only, kl_coef 0)",
                  "reward": reward_meta, "init": a.init, "seed": int(a.seed), "speed_mode": "wp4",
                  "speed_command": str(a.speed_command), "v2_defaults": applied,
                  "lagrange": ({n: {"target": t, "lambda_init": l0}
                                for n, (t, l0) in parse_lagrange(a.lagrange, a.lagrange_init).items()} if a.lagrange else None),
                  "shield_aux": ({"coef": float(a.shield_aux), "brake": float(a.shield_brake), "path_width": float(a.shield_path_width),
                                  "lag": float(a.shield_lag), "gap_min": float(a.shield_gap_min)} if a.shield_aux > 0 else None),
                  "losses": {"vf": float(a.vf), "vf_clip": float(a.vf_clip), "contact_bce": coefs["contact"],
                             "contact_horizon_steps": CONTACT_HORIZON,
                             "contact_label_delay_updates": (-(-(CONTACT_HORIZON - 1) // int(a.horizon))
                                                             if coefs["contact"] > 0 and int(a.horizon) < CONTACT_HORIZON else 0), "aux_b8": coefs["b8"],
                             "aux_near_car": coefs["near"], "ent": float(a.ent),
                             "b8_window": "since the stored state's episode began (train_v2.b8_sequence_loss)",
                             "aux_balance": {"main": "pg", "every": balance.every, "rho": balance.rho,
                                             "beta": balance.beta, "names": list(balance.names)},
                             "value_norm": "PopArt (the value head rescaled at every statistics update)"},
                  "optimizer": {"actor_lr": [float(a.lr), float(a.lr_end)], "critic_lr": float(a.critic_lr),
                                "adam_eps": float(a.adam_eps), "clip_split": bool(a.clip_split),
                                "max_grad": float(a.max_grad), "critic_warmup": int(a.critic_warmup)},
                  "value_norm_beta": float(a.value_norm_beta), "dial": {"margin": float(a.dial_margin),
                                                                        "exact": float(a.dial_exact)},
                  # held dial: the key `evaluate` falls back to, as v1 records it
                  **({"dial_constant": float(a.dial_constant)} if getattr(a, "dial_constant", None) is not None else {}),
                  # the LAST weights are the run's result, never ones picked by a loss (the perception
                  # gate: the risk head overfit at ~2.7 epochs while the perception heads kept improving)
                  "checkpoint_rule": "last",
                  "race": {"opponent_speed_mode": env.ecfg.opponent_speed_mode,
                           "league_seats": [int(j) for j in league_seats],
                           "league_every": int(a.league_every) if league_seats else None},
                  "controller": controller.checkpoint_meta()}
    steps_base = int(init_extra.get("total_steps", 0) or 0) if (a.init and init_extra.get("phase") == "ppo") else 0
    if a.steps_base is not None:
        steps_base = int(a.steps_base)
    wandb_id = (init_extra.get("wandb_id") if a.init else None) or (a.wandb_id or None)
    if a.wandb_new:
        steps_base, wandb_id = 0, None
    run = common.wandb_init(a.name, vars(a) | {"phase": "ppo", "arch": ARCH, "tracks": names, "v2_defaults": applied},
                            group=a.wandb_group, mode=a.wandb, resume_id=wandb_id)
    wandb_id = getattr(run, "id", None) or wandb_id
    out = common.run_dir(a.name)
    progress_log = common.ProgressLog(out)
    try:
        with open(os.path.join(out, "args.json"), "w") as fh:
            json.dump({k: v for k, v in vars(a).items()
                       if v is None or isinstance(v, (bool, int, float, str, list, tuple, dict))},
                      fh, indent=1, sort_keys=True, default=str)
    except OSError as exc:
        print(f"args.json not written: {exc}", flush=True)
    print(f"v2 PPO: horizon {T} x {B} learner rows, minibatch {a.minibatch} = {max(1, a.minibatch // T)} env chunk(s) "
          f"in {a.grad_accum} piece(s) | actor lr {a.lr:g}->{a.lr_end:g}, critic lr {a.critic_lr:g}, eps {a.adam_eps:g}, "
          f"clip-split {a.clip_split}, vf-clip {a.vf_clip:g}, kl_coef 0 | contact BCE {coefs['contact']:g}, "
          f"B8 {coefs['b8']:g}, near {coefs['near']:g}"
          + (f" balanced every {balance.every} updates to <= {balance.rho:g} x pg (EMA {balance.beta:g})"
             if balance.enabled else " (no auxiliary balance)")
          + f" | value normaliser with PopArt | v2 defaults applied: {applied or 'none'}", flush=True)
    print(f"v2 PPO race: speed cap the curriculum's ({a.cap0:g} -> {a.cap1:g} m/s), not observed"
          f" | opponents plan {env.ecfg.opponent_speed_mode or 'wp4 (as the learner)'}"
          + (f" | league: the learner's snapshot takes seat(s) {list(league_seats)} in turn every {a.league_every} "
             f"updates" if league_seats else ""), flush=True)
    league_next = 0
    torch.manual_seed(a.seed)
    env.sim.warmup()
    # ---------------------------------------------------------------- buffers (learner rows)
    K, N, E, A = spec.n_frames, spec.n_beams, spec.ego_dim, env.act_dim
    z = lambda *s, **kw: torch.zeros(*s, device=device, **kw)
    buf_scan = z(T, B, K, N, dtype=torch.float16)
    buf_ego, buf_c, buf_priv = z(T, B, E), z(T, B, 1), z(T, B, priv_dim)
    buf_act, buf_logp = z(T, B, A), z(T, B)
    buf_rew, buf_done, buf_trunc = z(T, B), z(T, B), z(T, B)
    buf_val, buf_final_val = z(T + 1, B), z(T, B)
    #: the critic's contact logit at rollout time: scored against the labels BEFORE the update has
    #: trained on them, so its AUC is out of sample (the G4 gate's held-out AUC, online)
    buf_logit = z(T, B)
    buf_mask, buf_keep = torch.ones(T, B, device=device), torch.ones(T, B, device=device)
    buf_onset = z(T, B, dtype=torch.bool)
    buf_capped = z(T, B, dtype=torch.bool)
    buf_since = z(T, B, dtype=torch.long)
    buf_ref_mu = z(T, B, A)
    buf_prog = z(T, B)
    buf_comp = z(T, B, len(REWARD_COMPONENT_KEYS))
    # --contact-severity-min: the learner's onsets both ways, all and at or above the threshold
    buf_sev = z(T, B, len(SEVERITY_LOG)) if env.ecfg.contact_severity_min > 0 else None
    buf_lvl = torch.full((T, B), -1, dtype=torch.long, device=device) if a.level_replay > 0 else None
    if on["b8"]:
        buf_b8_lab, buf_b8_cand, buf_b8_age = z(T, B, 3, 3), z(T, B, 3, dtype=torch.bool), z(T, B, 3, dtype=torch.long)
    if on["near"]:
        buf_near = z(T, B, 5)
    #: `--shield-aux`: the first-speed ceiling of each step in action units, and where a car is in our path
    shield_on = a.shield_aux > 0 and env.sim.other_idx is not None
    if shield_on:
        from ..mpc import plan_layout
        i_v1 = plan_layout("wp4").speed.start
        v_max_pol = float(env.ecfg.v_max_policy)
        buf_vsafe, buf_shield = z(T, B), z(T, B, dtype=torch.bool)
    #: The critic's contact labels need CONTACT_HORIZON steps of the future; a rollout shorter than that labels nothing
    #: on its own -- every window runs past the buffer. With the fine granularity (--horizon 32, since 2026-10-22) the
    #: contact head was never trained: label_valid_frac 0 in every run. So the critic's inputs and the onsets of the last
    #: `label_delay` rollouts are kept, and the oldest is labelled from the rollouts after it, that many updates late
    #: (2026-10-23). 0 when a rollout is long enough on its own: then nothing changes.
    label_delay = -(-(CONTACT_HORIZON - 1) // T) if (coefs["contact"] > 0 and T < CONTACT_HORIZON) else 0
    past = collections.deque(maxlen=label_delay) if label_delay else None
    if label_delay:
        print(f"contact labels: rollout {T} < horizon {CONTACT_HORIZON} steps -> labelled {label_delay} update(s) late",
              flush=True)
    h_actor = model.actor.initial_hidden(env.B, device=device)
    h_ref = ref.initial_hidden(env.B, device=device)
    buf_h0 = z(1, B, model.actor.hidden_size)
    since = torch.zeros(env.B, dtype=torch.long, device=device)          # steps since each row's episode began
    charge_prev = torch.zeros(env.B, dtype=torch.bool, device=device)
    lagrange = {}
    for name, (target, lam0) in parse_lagrange(a.lagrange, a.lagrange_init).items():
        key, weight = {"wall": ("collision", abs(env.ecfg.reward_collision)),
                       "car": ("car_contact", abs(env.ecfg.reward_car_contact))}[name]
        if weight <= 0:
            raise SystemExit(f"--lagrange {name}: its component '{key}' has weight 0")
        lagrange[name] = {"index": REWARD_COMPONENT_KEYS.index(key), "weight": weight,
                                  "target": target, "lam": lam0, "rate": float("nan"),
                                  "onset": name == "car",
                                  **({"anneal": {"frac": float(a.lagrange_anneal), "every": int(a.lagrange_anneal_every)}}
                                     if a.lagrange_anneal > 0 else {})}
    # The dial: drawn per episode, or held at one number for every car and episode (`--dial-constant`) -- then the
    # grip budget is the floor's own friction, as `ppo.main` does for v1 (30793bf, 7d63f92). Until 2026-10-22 v2 drew
    # the dial whatever --dial-constant said.
    held = getattr(a, "dial_constant", None)
    dial = (cond_mod.ConstantDial(env, held) if held is not None
            else cond_mod.DialDraw(env, a.dial_margin, a.dial_exact))
    cspec = cond_mod.CondSpec.from_meta(model.meta["cond"])
    dial_new = torch.ones(env.B, dtype=torch.bool, device=device)

    def condition_now():
        """c (B, 1) for the observation in hand; draws the dial of every episode that just began."""
        dial.redraw(dial_new); dial_new.zero_()
        d = dial.value()
        if a.grip_budget_penalty > 0:
            env.set_grip_budget(env.sim.P["mu"].reshape(-1) if held is not None else d)
        return cond_mod.mu_to_c(d, cspec)

    env_chunk = max(1, a.minibatch // T)
    accum = int(a.grad_accum)
    piece = env_chunk // accum
    n_updates = int(a.total // (T * B))
    steps_done = update = 0
    cap = a.cap0
    t_start = time.time(); t_loop = 0.0
    ep_stats = {"return": [], "progress": [], "contacts": [], "lap_time": [], "steps": []}
    last_log = {}

    def meta_now():
        """Everything a v2 PPO checkpoint carries besides the weights: the contract, where the run
        got to, Adam's moments by parameter name, and the value normaliser's statistics -- the critic
        was trained against them, so a resume without them would read every value on a new scale."""
        return {"spec": dict(vars(spec)), "phase": "ppo", "arch": ARCH, "run": a.name, "update": update,
                "updates": n_updates, "steps": steps_done, "total_steps": steps_base + steps_done,
                "wandb_id": wandb_id, "cap": cap, "action_mode": "plan", "speed_mode": "wp4",
                "speed_command": str(a.speed_command), "metrics": dict(last_log),
                "opt": opt.state_dict(), "opt_param_names": opt_param_names,
                "value_norm": vnorm.state_dict(), "aux_balance": balance.state_dict(),
                # the follower the plans were fitted to (`mpc.contract_of`), as v1 records it: present only for MAP.
                # Without it a v2 policy trained with MAP was evaluated on the iLQR (2026-10-22).
                **({"plan_follower": "map", "plan_follower_map": dict(a.plan_follower_map or {})}
                   if getattr(a, "plan_follower", None) == "map" else {}),
                "experiment": {**experiment, "controller": controller.checkpoint_meta()}}
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and device.type == "cuda")
    while steps_done < a.total:
        frac = steps_done / a.total
        cap = a.cap0 + (a.cap1 - a.cap0) * min(1.0, steps_done / a.cap_steps)
        env.set_speed_cap(cap)
        lr = a.lr + (a.lr_end - a.lr) * frac
        for g in opt.param_groups:
            g["lr"] = lr if g.get("group") == "actor" else a.critic_lr
        if a.yield_to_viewer > 0 and t_loop > 0 and common.viewer_active():
            time.sleep(min(t_loop * a.yield_to_viewer, 1.0))
        t_loop0 = time.time()
        tm = common.Timer()
        # The update this loop makes is logged (below, after the count moves on) when the NEW count
        # is a multiple of --log-every, or it is the first. Decided here, once, so what is measured
        # for the log (per-loss norms, the PopArt check) is measured on the update that is logged --
        # the two conditions used to differ by one, and with --log-every > 1 every measurement but
        # the first landed on an update that was never logged.
        log_now = (update + 1) % a.log_every == 0 or update == 0
        # ------------------------------------------------------------ rollout
        model.eval()
        buf_h0.copy_(h_actor.detach()[:, lid])
        buf_keep.fill_(1.0)
        with torch.no_grad():
            for t in range(T):
                scan, ego = obs["scan"], obs["ego"]
                c = condition_now().to(ego.dtype)
                priv = env.privileged_v2()
                controller.pre_action(obs)
                with ac:
                    d_act, h_next = model.actor.step_dist(scan, ego, c, h_actor)
                    act = d_act.sample()
                    logp = d_act.log_prob(act).sum(1)
                    v_n, logit_roll = model.critic(scan[lid], ego[lid], c[lid], priv[lid])
                    mu_ref, h_ref = ref.step(scan, ego, c, h_ref)
                buf_scan[t] = scan[lid].half(); buf_ego[t] = ego[lid]; buf_c[t] = c[lid]; buf_priv[t] = priv[lid]
                buf_act[t] = act[lid].float(); buf_logp[t] = logp[lid].float()
                buf_val[t] = vnorm.denormalize(v_n.float())
                buf_logit[t] = logit_roll.float()
                buf_ref_mu[t] = mu_ref[lid].float()
                buf_since[t] = since[lid]
                if on["b8"]:
                    tr = env.opponent_tracks_v2()
                    l3, c3, g3 = select_tracks(tr["lab"], tr["cand"], tr["age"], tr["dist"])
                    buf_b8_lab[t], buf_b8_cand[t], buf_b8_age[t] = l3[lid], c3[lid], g3[lid]
                if on["near"]:
                    buf_near[t] = env.nearest_visible_car_v2()[lid]
                if buf_lvl is not None:
                    buf_lvl[t] = env.level_seed[lid]
                if shield_on:
                    vs_ = safe_speed(env, a.shield_brake, a.shield_path_width, a.shield_lag, a.shield_gap_min)[lid]
                    buf_shield[t] = torch.isfinite(vs_) & (vs_ < v_max_pol)
                    buf_vsafe[t] = (vs_.clamp(max=v_max_pol) / v_max_pol * 2 - 1).clamp(-1.0, 1.0)
                obs, rew, term, trunc, info = env.step(act.clamp(-1, 1))
                h_actor = h_next.float()
                controller.post_step(term, trunc)
                done_now = term | trunc
                dial_new |= done_now
                buf_rew[t] = rew[lid]; buf_done[t] = term[lid].float(); buf_trunc[t] = trunc[lid].float()
                if "on_policy" in info:
                    buf_mask[t] = info["on_policy"][lid].float()
                buf_comp[t] = torch.stack([info["reward_components"][k][lid] for k in REWARD_COMPONENT_KEYS], 1)
                buf_prog[t] = info["progress"][lid]
                if buf_sev is not None:
                    buf_sev[t] = severity_onsets(info, lid)
                if "car_charge" in info:
                    onset, charge_prev = contact_onsets(info["car_charge"] > 0, charge_prev, done_now)
                    buf_onset[t] = onset[lid]
                if "speed_capped" in info:
                    buf_capped[t] = info["speed_capped"][lid]
                ep_stats["lap_time"] += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
                buf_final_val[t].zero_()
                if "final" in info:
                    f = info["final"]; ids = f["ids"]
                    fo = info["final_obs"]
                    with ac:
                        # the ended episode's own dial (`c` of this step): the next draw is for the new one
                        fv, _ = model.critic(fo["scan"], fo["ego"], c[ids], info["final_priv_v2"])
                    all_fv = torch.zeros(env.B, device=device)
                    all_fv[ids] = vnorm.denormalize(fv.float())
                    buf_final_val[t] = all_fv[lid]
                    mm = env.learner[ids] & ~f["staggered"]
                    ep_stats["return"] += f["return"][mm].tolist(); ep_stats["progress"] += f["progress"][mm].tolist()
                    ep_stats["contacts"] += f["contacts"][mm].float().tolist(); ep_stats["steps"] += f["steps"][mm].tolist()
                h_actor = reset_hidden(h_actor, done_now)
                h_ref = reset_hidden(h_ref, done_now)
                since = torch.where(done_now, torch.zeros_like(since), since + 1)
                if t + 1 < T:
                    buf_keep[t + 1] = (~done_now[lid]).float()
            c_boot = condition_now().to(obs["ego"].dtype)
            priv_boot = env.privileged_v2()
            with ac:
                vb, _ = model.critic(obs["scan"][lid], obs["ego"][lid], c_boot[lid], priv_boot[lid])
            buf_val[T] = vnorm.denormalize(vb.float())
            rew_used = buf_rew
            if lagrange:
                rew_used = lagrange_step(buf_rew, buf_comp, buf_prog, buf_mask, lagrange, a.lagrange_lr, a.lagrange_max)
            adv = compute_gae(rew_used, buf_val, buf_done, buf_trunc, buf_final_val, a.gamma, a.lam)
            ret = adv + buf_val[:T]
            popart = None
            if log_now:
                # PopArt, checked live: a few stored states' de-normalised values before and after
                # the statistics move, in float32 outside autocast. Rounding only (~1e-6 relative).
                rs, rb = slice(0, min(T, POPART_CHECK_STEPS)), slice(0, min(B, POPART_CHECK_ROWS))
                pick = lambda x: x[rs, rb].reshape(-1, *x.shape[2:])
                chk = (pick(buf_scan).float(), pick(buf_ego), pick(buf_c), pick(buf_priv))
                v_before = vnorm.denormalize(model.critic(*chk)[0].float())
            # the statistics move and the value head moves with them (PopArt): every value the critic
            # has learnt keeps its meaning, only the targets it is trained towards are re-expressed
            vnorm.update(ret, buf_mask, preserve=model.critic.value)
            if log_now:
                v_after = vnorm.denormalize(model.critic(*chk)[0].float())
                popart = (float((v_after - v_before).abs().max()), float(v_before.abs().max()))
            ret_n = vnorm.normalize(ret)
            y_c, y_valid = contact_labels(buf_onset, buf_done, buf_trunc, CONTACT_HORIZON)
            late = None
            if label_delay and len(past) == label_delay:
                cat_ = lambda k, cur: torch.cat([p_[k] for p_ in past] + [cur])
                yl_, vl_ = contact_labels(cat_("onset", buf_onset), cat_("done", buf_done), cat_("trunc", buf_trunc),
                                          CONTACT_HORIZON)
                late = dict(past[0], y=yl_[:T], valid=vl_[:T] & (past[0]["mask"] > 0))
            if buf_lvl is not None:
                seat = env.race[lid].cpu().numpy()
                lv = buf_lvl.cpu().numpy(); aa = adv.abs().float().cpu().numpy()
                seat_tb = np.broadcast_to(seat[None, :], lv.shape)
                ok = lv >= 0
                keys = np.stack([seat_tb[ok], lv[ok]], 1)
                if keys.shape[0]:
                    uk, inv = np.unique(keys, axis=0, return_inverse=True)
                    sums = np.bincount(inv.ravel(), weights=aa[ok]); cnts = np.bincount(inv.ravel())
                    for (st_, sd_), sm_, cn_ in zip(uk, sums, cnts):
                        env.level_replay.accumulate(int(st_), int(sd_), float(sm_), int(cn_))
        t_roll = tm.lap()
        # ------------------------------------------------------------ update
        model.train()
        w_all = buf_mask / buf_mask.sum().clamp_min(1.0)
        adv_mean = (adv * w_all).sum(); adv_std = ((adv - adv_mean) ** 2 * w_all).sum().sqrt()
        adv_n = (adv - adv_mean) / (adv_std + 1e-8)
        v_old_n = vnorm.normalize(buf_val[:T])
        freeze_actor = update < a.critic_warmup
        on_actor = 0.0 if freeze_actor else 1.0
        theta0 = flat_parameters(deployed)
        stats = {}
        due = balance.due()
        t_measure = 0.0
        for ep in range(a.epochs):
            perm = torch.randperm(B, device=device)
            for i in range(0, B, piece):
                cols = perm[i:i + piece]
                first = (i // piece) % accum == 0
                last = (i // piece) % accum == accum - 1 or i + piece >= B
                if first and accum > 1:
                    w_group = buf_mask[:, perm[i:i + env_chunk]].sum().clamp_min(1.0)
                m = cols.numel()
                fl = lambda x: x.reshape(T * m, *x.shape[2:])
                w = fl(buf_mask[:, cols])
                with ac:
                    seq = model.actor.sequence(buf_scan[:, cols].float(), buf_ego[:, cols], buf_c[:, cols],
                                               h0=buf_h0[:, cols].contiguous(), keep=buf_keep[:, cols])
                    v_n, logit = model.critic(fl(buf_scan[:, cols].float()), fl(buf_ego[:, cols]),
                                              fl(buf_c[:, cols]), fl(buf_priv[:, cols]))
                mu = fl(seq["mu"]).float()
                std = model.actor.log_std.exp().float().expand_as(mu)
                dist = torch.distributions.Normal(mu, std)
                act_mb = fl(buf_act[:, cols])
                logp = dist.log_prob(act_mb).sum(-1)
                ent = dist.entropy().sum(-1)
                wmean = lambda x, ww=w: (x * ww).sum() / ww.sum().clamp_min(1.0)
                ratio = (logp - fl(buf_logp[:, cols])).exp()
                adv_mb = fl(adv_n[:, cols])
                pg = -wmean(torch.min(ratio * adv_mb, ratio.clamp(1 - a.clip, 1 + a.clip) * adv_mb))
                v_n = v_n.float()
                tgt = fl(ret_n[:, cols]).float()
                if a.vf_clip > 0:
                    vo = fl(v_old_n[:, cols]).float()
                    vc = vo + (v_n - vo).clamp(-a.vf_clip, a.vf_clip)
                    vf = 0.5 * wmean(torch.max((v_n - tgt) ** 2, (vc - tgt) ** 2))
                else:
                    vf = 0.5 * wmean((v_n - tgt) ** 2)
                yv = fl(y_valid[:, cols]).float() * w
                bce_sum = (F.binary_cross_entropy_with_logits(logit.float(), fl(y_c[:, cols]), reduction="none") * yv).sum()
                bce_n = yv.sum()
                if late is not None:
                    with ac:
                        _v, logit_l = model.critic(fl(late["scan"][:, cols].float()), fl(late["ego"][:, cols]),
                                                   fl(late["c"][:, cols]), fl(late["priv"][:, cols]))
                    yv_l = fl(late["valid"][:, cols]).float()
                    bce_sum = bce_sum + (F.binary_cross_entropy_with_logits(logit_l.float(), fl(late["y"][:, cols]),
                                                                            reduction="none") * yv_l).sum()
                    bce_n = bce_n + yv_l.sum()
                bce = bce_sum / bce_n.clamp_min(1.0)
                ent_w = wmean(ent)
                zero = torch.zeros((), device=device)
                b8 = near = zero
                shield = zero
                if shield_on:
                    m_s = fl(buf_shield[:, cols]).float() * w
                    exc = (mu[:, i_v1] - fl(buf_vsafe[:, cols])).clamp_min(0.0)
                    shield = (exc ** 2 * m_s).sum() / m_s.sum().clamp_min(1.0)
                    stats.setdefault("shield_excess_mps", []).append(
                        float((exc * m_s).sum() / m_s.sum().clamp_min(1.0)) * v_max_pol / 2)
                    stats.setdefault("shield_active_frac", []).append(float(m_s.sum() / w.sum().clamp_min(1.0)))
                b8_know = None
                if on["b8"]:
                    b8, st_b8 = b8_sequence_loss(seq["b8"], buf_b8_lab[:, cols], buf_b8_cand[:, cols],
                                                 buf_b8_age[:, cols], buf_since[:, cols], weight=buf_mask[:, cols])
                    b8_know = st_b8["knowable"]
                if on["near"]:
                    near = wmean(nearest_car_loss(fl(seq["near_car"]).float(), fl(buf_near[:, cols])))
                with torch.no_grad():
                    d_ref = torch.distributions.Normal(fl(buf_ref_mu[:, cols]), ref_std.expand_as(mu))
                    kl_ref = wmean(torch.distributions.kl_divergence(d_ref, dist).sum(-1))
                    approx_kl = ((ratio - 1) - (logp - fl(buf_logp[:, cols]))).mean()
                    clipfrac = ((ratio - 1).abs() > a.clip).float().mean()
                if log_now and ep == 0 and i == 0:
                    # the heads on this rollout before any of this update's steps: held out for the
                    # weights that drove it (`train_v2.aux_heldout_metrics`; logged, never a selection
                    # rule -- the run keeps its last checkpoint, the perception gate's finding)
                    lab_ = {}
                    if on["near"]:
                        lab_["near"] = buf_near[:, cols]
                    if on["b8"]:
                        lab_.update(b8_lab=buf_b8_lab[:, cols], b8_cand=buf_b8_cand[:, cols], b8_age=buf_b8_age[:, cols])
                    for k_, v_ in aux_heldout_metrics({k: v.detach() for k, v in seq.items() if torch.is_tensor(v)},
                                                      lab_, buf_since[:, cols]).items():
                        stats.setdefault(f"heldout/{k_}", []).append(v_)
                if (log_now or due) and ep == 0 and i == 0:
                    # each loss's pull on the trunk it shares, before anything is clipped or stepped
                    # (no entropy term: the log std is state-independent, its trunk gradient is zero),
                    # the auxiliaries at their configured coefficient -- before the balance's
                    # multiplier, which is computed from exactly these numbers
                    t_m = time.perf_counter()
                    gn = loss_grad_norms({"pg": pg,
                                          "b8": coefs["b8"] * b8 if on["b8"] else None,
                                          "near": coefs["near"] * near if on["near"] else None,
                                          "shield": a.shield_aux * shield if shield_on else None}, actor_trunk)
                    if due:
                        balance.observe(gn)
                    t_measure += time.perf_counter() - t_m
                    if log_now:
                        gn.update({f"critic_{k}": v for k, v in loss_grad_norms(
                            {"vf": a.vf * vf, "contact": coefs["contact"] * bce}, critic_trunk).items()})
                    for k, v in gn.items():
                        stats.setdefault(f"grad/{k}", []).append(v)
                s_ = balance.scale
                actor_terms = (pg - a.ent * ent_w + coefs["b8"] * s_.get("b8", 1.0) * b8
                               + coefs["near"] * s_.get("near", 1.0) * near + a.shield_aux * shield)
                loss = a.vf * vf + coefs["contact"] * bce + on_actor * actor_terms
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite v2 PPO loss at update {update} (pg {float(pg):.4g} vf {float(vf):.4g} "
                                       f"bce {float(bce):.4g} b8 {float(b8):.4g})")
                if accum > 1:
                    loss = loss * (w.sum().clamp_min(1.0) / w_group)
                if first:
                    opt.zero_grad()
                loss.backward()
                if last:
                    if clip_sets is None:
                        g_a = torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad, error_if_nonfinite=True)
                        g_c = g_a
                    else:
                        g_a = torch.nn.utils.clip_grad_norm_(clip_sets[0], a.max_grad, error_if_nonfinite=True)
                        g_c = torch.nn.utils.clip_grad_norm_(clip_sets[1], a.max_grad, error_if_nonfinite=True)
                    opt.step()
                    stats.setdefault("grad_norm_actor", []).append(float(g_a))
                    stats.setdefault("grad_norm_critic", []).append(float(g_c))
                for k, v in (("pg", pg), ("vf", vf), ("contact_bce", bce), ("entropy", ent.mean()), ("shield", shield),
                             ("kl_ref", kl_ref), ("approx_kl", approx_kl), ("clipfrac", clipfrac),
                             ("b8", b8), ("near_car", near)):
                    stats.setdefault(k, []).append(float(v.detach()))
                if b8_know is not None:
                    stats.setdefault("b8_knowable", []).append(float(b8_know))
        move = relative_movement(theta0, deployed)
        balance.tick()
        t_upd = tm.lap()
        steps_done += T * B; update += 1
        # ------------------------------------------------------------ logging
        if log_now:                                  # (update % a.log_every == 0 or update == 1)
            n_pol = buf_mask.sum().clamp_min(1.0)
            km = float((buf_prog.clamp_min(0) * buf_mask).sum()) / 1000.0
            if label_delay:
                lab_y, lab_v, lab_logit = ((late["y"], late["valid"], late["logit"]) if late is not None else
                                           (y_c, torch.zeros_like(y_valid), buf_logit))
            else:
                lab_y, lab_v, lab_logit = y_c, y_valid, buf_logit
            sel = lab_v & (buf_mask > 0) if not label_delay else lab_v
            log = {"steps": steps_base + steps_done, "update": update, "speed_cap": cap, "lr": lr,
                   "lr_critic": a.critic_lr, "kl_coef": 0.0,
                   "rollout/reward_per_step": float((buf_rew * buf_mask).sum() / n_pol),
                   "rollout/mean_speed": env.sim.state[:, 3].mean().item(),
                   "rollout/value_mean": float(buf_val[:T].mean()), "rollout/adv_std": float(adv.std()),
                   "rollout/speed_capped_frac": float((buf_capped.float() * buf_mask).sum() / n_pol),
                   "value_norm/mean": vnorm.mean, "value_norm/std": vnorm.std,
                   "contact/label_rate": float((lab_y * lab_v).sum() / lab_v.sum().clamp_min(1)),
                   "contact/label_valid_frac": float(lab_v.float().mean()),
                   "contact/label_delay_updates": float(label_delay),
                   "contact/onsets_per_km": _per_km(float((buf_onset.float() * buf_mask).sum()), km * 1000.0),
                   "contact/auc_rollout": binary_auc(lab_logit[sel], lab_y[sel]) if bool(sel.any()) else float("nan"),
                   "policy/actor_move": move, "policy/frozen_actor": float(freeze_actor),
                   "time/rollout_s": t_roll, "time/update_s": t_upd,
                   "time/env_steps_per_s": T * B / max(t_roll + t_upd, 1e-9),
                   "time/elapsed_min": (time.time() - t_start) / 60}
            for k, v in stats.items():
                log[(k if k.startswith(("grad", "heldout/")) else f"loss/{k}")] = float(np.mean(v))
            if balance.enabled:
                log.update({f"balance/scale_{k}": v for k, v in balance.scale.items()})
                log.update({f"balance/pull_{k}": v for k, v in balance.pull().items()})
                log["balance/measurements"] = balance.measurements
            # the measurement's own cost this update (0 when nothing was measured), against the update
            log["time/grad_measure_s"] = t_measure
            if popart is not None:
                log["value_norm/popart_max_dev"], log["value_norm/popart_value_scale"] = popart
            ls = model.actor.log_std.detach()
            log["policy/log_std_kappa"] = float(ls[:4].mean()); log["policy/log_std_speed"] = float(ls[4:].mean())
            log.update({f"reward/{k}_per_step": float((buf_comp[:, :, i_] * buf_mask).sum() / n_pol)
                        for i_, k in enumerate(REWARD_COMPONENT_KEYS)})
            for n_, v_ in lagrange.items():
                log[f"lagrange/{n_}_lambda"] = v_["lam"]; log[f"lagrange/{n_}_rate_per_km"] = v_["rate"]
            sev_rates = severity_rates(buf_sev, buf_prog, buf_mask) if buf_sev is not None else {}
            log.update(sev_rates)
            if ep_stats["return"]:
                dist_m = float(np.sum(ep_stats["progress"]))
                last_log = {"collisions_per_km": _per_km(np.sum(ep_stats["contacts"]), dist_m),
                            "progress_m": float(np.mean(ep_stats["progress"])),
                            "lap_time_s": float(np.mean(ep_stats["lap_time"])) if ep_stats["lap_time"] else float("nan")}
                log.update({"episode/return": float(np.mean(ep_stats["return"])),
                            "episode/progress_m": last_log["progress_m"],
                            "episode/collisions_per_km": last_log["collisions_per_km"],
                            "episode/len_steps": float(np.mean(ep_stats["steps"])),
                            "episode/count": len(ep_stats["return"])})
                if ep_stats["lap_time"]:
                    log["episode/lap_time_s"] = last_log["lap_time_s"]
                ep_stats = {k: [] for k in ep_stats}
            run.log(log, step=steps_base + steps_done)
            if a.metrics_jsonl:
                with open(a.metrics_jsonl, "a") as f_:
                    f_.write(json.dumps({k_: (float(v_) if isinstance(v_, (int, float, np.floating)) else v_)
                                         for k_, v_ in log.items()}) + "\n")
            record = {"kind": "ppo", "arch": ARCH, "update": update, "total": n_updates,
                      "steps": steps_base + steps_done, "cap": cap,
                      "rew_per_step": log["rollout/reward_per_step"],
                      "coll_per_km": log.get("episode/collisions_per_km", float("nan")),
                      "prog_m": log.get("episode/progress_m", float("nan")),
                      "lap_s": log.get("episode/lap_time_s", float("nan")),
                      "kl_ref": log.get("loss/kl_ref", float("nan")), "sps": log["time/env_steps_per_s"],
                      "wall_s": time.time() - t_start}
            record.update({k_: v_ for k_, v_ in log.items() if k_ not in record})
            progress_log.write(record)
            print(f"upd {update}/{n_updates} steps {(steps_base + steps_done) / 1e6:.2f}M cap {cap:.1f} | "
                  f"rew/step {log['rollout/reward_per_step']:.3f} coll {log.get('episode/collisions_per_km', float('nan')):.1f}/km "
                  f"| pg {log.get('loss/pg', float('nan')):.4f} vf {log.get('loss/vf', float('nan')):.4f} "
                  f"bce {log.get('loss/contact_bce', float('nan')):.4f} b8 {log.get('loss/b8', float('nan')):.4f} "
                  f"kl_ref {log.get('loss/kl_ref', float('nan')):.4f} | move {move:.2e} | capped "
                  f"{log['rollout/speed_capped_frac']:.2f} | vnorm {vnorm.mean:.2f}+-{vnorm.std:.2f}"
                  + (f" (PopArt dev {popart[0]:.1e})" if popart is not None else "")
                  + (" | aux scale " + " ".join(f"{k} {v:.3g}" for k, v in balance.scale.items())
                     if balance.enabled else "")
                  + (f" | >={a.contact_severity_min:g} m/s wall {sev_rates['contacts/wall_hard_per_km']:.2f}"
                     f"/{sev_rates['contacts/wall_all_per_km']:.2f} car {sev_rates['contacts/car_onset_hard_per_km']:.2f}"
                     f"/{sev_rates['contacts/car_onset_all_per_km']:.2f}/km" if sev_rates else "")
                  + (f" | shield exc {log.get('loss/shield_excess_mps', float('nan')):.3f} m/s on "
                     f"{log.get('loss/shield_active_frac', float('nan')):.3f}" if shield_on else "")
                  + f" | {log['time/env_steps_per_s']:.0f} steps/s", flush=True)
        if label_delay:
            # this rollout, for the labels the next one(s) complete (see `label_delay`)
            past.append({"scan": buf_scan.clone(), "ego": buf_ego.clone(), "c": buf_c.clone(), "priv": buf_priv.clone(),
                         "onset": buf_onset.clone(), "done": buf_done.clone(), "trunc": buf_trunc.clone(),
                         "mask": buf_mask.clone(), "logit": buf_logit.clone()})
        t_loop = time.time() - t_loop0
        if getattr(env, "level_replay", None) is not None and update % 16 == 0:
            print("level replay: " + " ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                                              for k, v in env.level_replay.stats().items()), flush=True)
        if league_seats and update % a.league_every == 0:
            # the learner's current self takes the oldest league seat: a v2 entry, read in wp4 and shown
            # the env's own observation, beside the v1 seats in theirs (`opponent_pool`, `seat_obs`).
            # The weights and the contract only -- a seat drives, it is never resumed.
            k_ = league_next % len(league_seats); league_next += 1
            lp_ = os.path.join(out, f"league_{k_}.pt")
            snap = {key: v for key, v in meta_now().items() if key not in ("opt", "opt_param_names")}
            save_checkpoint(lp_, model, snap)
            env.pool.replace(league_seats[k_], lp_, env)
            print(f"league: update {update} -> seat {k_} (pool entry {league_seats[k_]})", flush=True)
        if update % a.save_every == 0:
            meta = meta_now()
            save_checkpoint(os.path.join(out, f"ppo_u{update}.pt"), model, meta)
            save_checkpoint(os.path.join(out, "ppo_latest.pt"), model, meta)
    save_checkpoint(os.path.join(out, "ppo_final.pt"), model, meta_now())
    controller.release()
    progress_log.close()
    if graph_rt is not None:
        from .graph_runtime import release_graph_runtime
        release_graph_runtime(graph_rt)
    run.finish()
