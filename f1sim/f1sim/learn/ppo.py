"""PPO fine-tuning with an asymmetric critic, initialized from the DAgger student.

- KL(pi_IL || pi) regularizer, decayed over the first `kl_decay` env steps, keeps the policy near
  the imitation policy while the critic warms up.
- Speed-cap curriculum: the commanded speed cap ramps from `cap0` to `cap1` over `cap_steps`
  env steps (the cap is part of the observation, so the policy stays consistent).
- Time-limit truncations bootstrap with the final observation value estimate.
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch

from ..gym_env import EnvConfig, FUTURE_LABEL_DIM, PRIV_OPP_DIST_SCALE, REWARD_COMPONENT_KEYS
from ..opp_token import OPP_TOKEN_MODES, describe as describe_opp_token
from ..params import Config
from . import common
from . import conditioning as cond_mod
from . import grip_runtime as grip_rt
from . import opponent_config as opp_cfg
from .future import (FUTURE_K, align_future_targets, future_labelled_fraction, future_loss,
                     future_spec)
from .memory import Hidden, memory_spec, reset_hidden
from .model import (ActorCritic, load_checkpoint, load_for_conditioning, load_for_memory,
                    save_checkpoint)
from .obs import SCAN_CHANNELS, ScanAugment, flatten_obs
from .returns import compute_gae

#: How close an opponent has to be for the auxiliary head to be scored on it [m]. This is a
#: deliberately conservative auxiliary range, NOT the sensor's reach: the LiDAR sees to 10 m, but
#: beyond a few metres an opponent is often occluded by the boundary or covered by too few beams to
#: read an offset from, so scoring the head out there mostly teaches it the conditional mean of an
#: empty road. Measured on a rollout, only ~47 % of frames have any beam on the opponent at all.
AUX_OPP_RANGE_M = 6.0


def sample_rollout_action(
    model: ActorCritic, scan: torch.Tensor, proprio: torch.Tensor, cond: torch.Tensor = None,
    hidden: torch.Tensor = None,
):
    """The sampled action, its matching log probability, and the actor's next hidden state.

    `cond` is the frozen conditioning for this step. It is passed explicitly, never defaulted: the
    log probability stored here is the one the update ratio is measured against, so the recomputed
    log probability has to come from the identical input. `hidden` is the same contract one step
    further: with `--memory`, the action depends on it, so the update has to replay from the state
    the rollout was actually in. It is `None`, in and out, for a feedforward actor -- and then
    `step_dist` is `dist`, so an unflagged run draws exactly the action it drew before.
    """
    distribution, hidden = model.actor.step_dist(scan, proprio, cond, hidden)
    action = distribution.sample()
    return action, distribution.log_prob(action).sum(1), hidden


@dataclass
class PPOHyper:
    """The scalars `minibatch_losses` needs, so the loss is a function of stated numbers."""
    clip: float
    vf: float
    ent: float
    kl_coef: float
    aux_grip: float = 0.0
    aux_opp: float = 0.0
    aux_future: float = 0.0
    priv_mu_index: int = 16
    aux_opp_range_m: float = AUX_OPP_RANGE_M
    m_gt_1: bool = False


def minibatch_losses(model: ActorCritic, ref, *, scan, pro, priv, act, logp_old, adv, ret,
                     val_old, w, cond=None, hyper: PPOHyper, autocast=None,
                     freeze_actor: bool = False, sequence=None,
                     future=None, future_valid=None) -> dict:
    """One PPO minibatch's loss terms, feedforward or recurrent.

    Lifted out of `main`'s inner loop unchanged so that (a) the recurrent path and the feedforward
    path cannot drift apart -- there is one copy of the arithmetic and only the model call differs
    -- and (b) `tests/test_ppo_memory.py` can hold it to `tests/data/ppo_loss_oracle.json`, which
    was recorded from the inlined version before any of this existed. `--memory off` has to be the
    run it was yesterday, and "the loss on a fixed batch is unchanged" is how that is checked.

    Feedforward: every tensor is (batch, ...) and `sequence` is None.
    Recurrent: `sequence` is `(h0, keep)`, `scan`/`pro`/`priv`/`act`/`cond` are (T, m, ...) blocks
    of whole env chunks, and the flat tensors (`logp_old`, `adv`, `ret`, `val_old`, `w`) are the
    matching (T * m,) rows in row-major (step, env) order. `keep` is 0 where the episode ended on
    the step before, which is where the hidden state is masked back to zero.

    `future` / `future_valid` are the auxiliary future head's aligned labels and their mask, shaped
    like `priv` and `w` respectively (so (T, m, D) and (T, m) in the recurrent path). They are
    optional and default to None, which -- together with `hyper.aux_future = 0` -- is what makes an
    unflagged run's loss the number the frozen oracle recorded.
    """
    ac = autocast if autocast is not None else nullcontext()
    with ac:
        if sequence is None:
            logp, ent, val, d, grip, opp_pred, fut_pred, h_next = model.evaluate_aux(
                scan, pro, priv, act, cond)
            ref_scan, ref_pro, ref_cond = scan, pro, cond
        else:
            h0, keep = sequence
            logp, ent, val, d, grip, opp_pred, fut_pred, h_next = model.evaluate_sequence(
                scan, pro, priv, act, cond, h0, keep)
            T, m = scan.shape[0], scan.shape[1]
            flat = lambda t: None if t is None else t.reshape(T * m, *t.shape[2:])
            ref_scan, ref_pro, ref_cond, priv = flat(scan), flat(pro), flat(cond), flat(priv)
            future, future_valid = flat(future), flat(future_valid)
    logp, ent, val = logp.float(), ent.float(), val.float()

    def wmean(x, w):
        return (x * w).sum() / w.sum().clamp_min(1.0)

    aux = torch.zeros((), device=logp.device)
    if hyper.aux_grip > 0:
        mu_true = priv[:, hyper.priv_mu_index] - 1.0
        aux = wmean((grip - mu_true) ** 2, w)
    aux_o = torch.zeros((), device=logp.device)
    if hyper.aux_opp > 0 and hyper.m_gt_1:
        # priv[8:11] = nearest opponent (ahead offset, side offset, longitudinal speed
        # difference), scaled to O(1). NOTE the third channel is other.vx - ego.vx, a
        # difference of two body-frame longitudinal speeds -- NOT the line-of-sight
        # closing speed (d/dt of the distance) that car_proximity_penalty computes. The
        # name is kept honest here; changing what the channel *means* would silently
        # reinterpret every checkpoint trained against it, so that is a separate change.
        o_true = priv[:, 8:11] / torch.tensor([3.0, 1.0, 2.0], device=priv.device)
        # priv[:, 11] is dist/PRIV_OPP_DIST_SCALE (gym_env owns that scale), so the comparison
        # has to be made in metres. Comparing the scaled column against 6.0 directly
        # selected 30 m -- three times the LiDAR's range, i.e. a mask that removed
        # nothing and trained the head on an empty road.
        near = (priv[:, 11] * PRIV_OPP_DIST_SCALE < hyper.aux_opp_range_m).to(w.dtype) * w
        aux_o = wmean(((opp_pred - o_true) ** 2).mean(1), near)
    aux_f = torch.zeros((), device=logp.device)
    aux_f_parts: dict = {}
    if hyper.aux_future > 0:
        if fut_pred is None or future is None or future_valid is None:
            raise ValueError(
                "--aux-future is on but this minibatch carries no future head / labels: the head "
                "is built from meta['future_head'] and the labels come from the rollout buffer, so "
                "a missing one means the run was assembled wrong rather than that the term is off.")
        aux_f, aux_f_parts = future_loss(fut_pred.float(), future.float(),
                                         future_valid.float(), w.float())
    ratio = (logp - logp_old).exp()
    pg = -wmean(torch.min(ratio * adv, ratio.clamp(1 - hyper.clip, 1 + hyper.clip) * adv), w)
    v_clipped = val_old + (val - val_old).clamp(-hyper.clip, hyper.clip)
    vf = 0.5 * wmean(torch.max((val - ret) ** 2, (v_clipped - ret) ** 2), w)
    with torch.no_grad(), ac:
        # The reference is fed the same condition, and its projection is still zero,
        # so it evaluates as the frozen unconditional baseline. Passing `c` keeps the
        # call valid for a conditional actor without letting the leash move with it.
        # `feedforward_dist` is the same call for a feedforward reference and, for a reference
        # copied from a memory actor, states what it is: the leash is the ORIGINAL policy, so
        # the recurrence is not run and its (zero) projection is not applied.
        d_ref = ref.feedforward_dist(ref_scan, ref_pro, ref_cond)
    d_ref = torch.distributions.Normal(d_ref.mean.float(), d_ref.stddev.float())
    d = torch.distributions.Normal(d.mean.float(), d.stddev.float())
    kl_ref = wmean(torch.distributions.kl_divergence(d_ref, d).sum(1), w)
    ent_w = wmean(ent, w)
    loss = (hyper.vf * vf + (0.0 if freeze_actor else 1.0) * (pg - hyper.ent * ent_w
            + hyper.kl_coef * kl_ref) + hyper.aux_grip * aux + hyper.aux_opp * aux_o
            + hyper.aux_future * aux_f)
    return {"pg": pg, "vf": vf, "ent": ent_w, "entropy_mean": ent.mean(), "kl_ref": kl_ref,
            "aux_grip": aux, "aux_opp": aux_o, "aux_future": aux_f,
            "aux_future_parts": aux_f_parts, "loss": loss, "hidden": h_next,
            "approx_kl": ((ratio - 1) - (logp - logp_old)).mean(),
            "clipfrac": ((ratio - 1).abs() > hyper.clip).float().mean()}


def warm_start_additions(init_meta: dict, memory, scan_channels, future_head, opp_token=None):
    """The architecture pieces `--init` does NOT already carry, i.e. what a warm start would add.

    A resume is not a warm start. `load_for_memory` refuses a checkpoint that already has the thing
    being asked for, and it is right to -- adding a second GRU or a second head to a trained path
    would re-initialise it. But the flag that *builds* a piece is also the flag that *keeps training*
    it (`--aux-future` is a coefficient), so leg two of a run passes the same command line as leg
    one and must not be sent down the warm-start path for it. This is the one place that decides:
    whatever the checkpoint already has, it keeps, and only the rest is new.
    """
    return (None if init_meta.get("memory") else memory,
            None if init_meta.get("scan_channels") else scan_channels,
            None if init_meta.get("future_head") else future_head,
            None if init_meta.get("opp_token") else opp_token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=f"ppo_{time.strftime('%m%d_%H%M')}")
    ap.add_argument("--init", default="", help="DAgger checkpoint to start from")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--tracks", default="train",
                    help="'train' / 'heldout' / 'heldout_obstacles' / 'heldout_all', or a comma "
                         "separated list in either grammar: 'real/bb22-1@rev#line:44' (new) or "
                         "'real:blackbox2022_1+rlobs44~rev' (loader). An open seed -- "
                         "'real/bb22-1#line:*' -- is drawn --obstacle-draws times.")
    ap.add_argument("--obstacle-draws", type=int, default=8,
                    help="how many random obstacle placements a '#<kind>:*' entry becomes. The draw "
                         "comes from --seed, so the same seed gives the same maps, and the concrete "
                         "names are recorded in the run manifest.")
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
                    help="controller arm between the policy and the wheels. 'legacy' is the "
                         "untouched path and is the default, so an unflagged run is unchanged. "
                         "'fixed_low' is the control arm, 'oracle' reads the true friction (lab "
                         "only), 'estimated' uses the frozen estimator's filtered lower quantile "
                         "from causal sensors. A '+tcs' suffix (or 'tcs' alone) additionally runs "
                         "the car's own traction guard on the speed command, fed from the simulated "
                         "ERPM odometry and IMU; it needs vehicle.wheel_model. 'fixed_low+tcs' is "
                         "the deployment default.")
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
    ap.add_argument("--metrics-jsonl", default="",
                    help="also append every logged metric dict to this file, one JSON object per "
                         "logged update. Opt-in and orthogonal to --wandb: a `--wandb disabled` run "
                         "otherwise leaves no machine-readable curve behind, which is exactly what a "
                         "smoke has to report")
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
    ap.add_argument("--aux-future", type=float, default=0.0,
                    help=f"weight of an auxiliary loss making the actor's RECURRENT STATE (the trunk "
                         f"features with --memory off) predict, {FUTURE_K} control steps "
                         f"({FUTURE_K * 0.025:.2f} s) ahead: the nearest opponent's relative position "
                         f"and velocity in the ego frame, a presence logit, and the ego's own speed "
                         f"and yaw rate. Privileged labels from the simulator, masked across episode "
                         f"boundaries and where no opponent is in range. 0 = off, and off the head is "
                         f"not built at all: same parameters, same state dict, same loss. Measure "
                         f"what it did with `python -m f1sim.learn.probe_hidden`")
    ap.add_argument("--aux-future-k", type=int, default=FUTURE_K,
                    help=f"lookahead of --aux-future in control steps (40 Hz). Default {FUTURE_K} "
                         f"= 0.5 s. Only the last k steps of each --horizon chunk go unlabelled, so "
                         f"a k near the horizon leaves almost nothing to train on")
    ap.add_argument("--aux-future-width", type=int, default=128,
                    help="hidden width of the future head's one layer")
    ap.add_argument("--ttc-penalty", type=float, default=0.0, metavar="LAMBDA",
                    help="per-step penalty at zero time to contact with the nearest car, ramping "
                         "linearly from 0 at --ttc-safe. The reward audit "
                         "(docs/research/reward-audit-2026-09-13.md) found nothing prices the "
                         "*approach* to another car: car_proximity is charged per metre driven and "
                         "only inside --car-safe-gap, so closing at 3 m/s from 2 m away -- 0.7 s "
                         "from a contact -- costs zero. 0 = off.")
    ap.add_argument("--ttc-safe", type=float, default=1.0, metavar="SECONDS",
                    help="[s] time to contact above which --ttc-penalty is 0")
    ap.add_argument("--overtake-sustained", type=float, nargs=3, default=None,
                    metavar=("DIST_M", "TIME_S", "BONUS"),
                    help="pay BONUS once when the learner has led an opponent by DIST_M metres "
                         "along the lane for TIME_S seconds continuously (1.5 1.0 <bonus> is the "
                         "contract's default). Unlike --overtake-bonus, which pays dense arc gained "
                         "and so pays the approach and the crossing instant, this pays only a pass "
                         "that stuck.")
    ap.add_argument("--opp-token", default="off", choices=OPP_TOKEN_MODES,
                    help="ORACLE. Append the simulator's own description of the nearest two "
                         "opponents to the observation (f1sim/opp_token.py): 'pos' = relative "
                         "position + presence, 'posvel' = + relative velocity, 'future' = + where "
                         "each opponent's own controller intends to be at 0.10/0.25/0.50/0.75 s. "
                         "The checkpoint records it and is refused by the exporter and the ROS "
                         "node. Its purpose is to answer whether the planner, handed the "
                         "opponent's state, races better than it does with LiDAR alone -- not to "
                         "produce a policy.")
    ap.add_argument("--name-seed-fresh", action="store_true",
                    help="re-initialise every tensor a warm start leaves fresh from its own module "
                         "name rather than from the ambient generator (learn.model."
                         "reinit_fresh_by_name, from branch feat/motion-memory). Two arms that "
                         "differ by an input width otherwise differ in every module built after "
                         "the layer that widened, because it consumes different draws.")
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
    # The race / opponent / behaviour / spawn flags. Shared with `learn.opponent_census`, which has
    # to be able to build exactly the configuration a run trains against -- see that module.
    opp_cfg.add_arguments(ap)
    ap.add_argument("--action-mode", default="direct", choices=["direct", "plan"], help="plan: the policy outputs a local trajectory (f1sim.mpc)")
    ap.add_argument("--memory", default="off", choices=("off", "gru"),
                    help="give the actor (and by default the critic) a recurrent memory over the "
                         "per-step trunk embedding, warm-started from --init so that step 0 is "
                         "bit-identical to it. 'off' is the untouched run in every respect: same "
                         "network, same rollout, same minibatching, same loss. The six-frame stack "
                         "stays the input; this extends the window past the 150 ms it covers, which "
                         "is what a box that has left the scan cannot otherwise survive")
    ap.add_argument("--memory-hidden", type=int, default=128,
                    help="GRU width (<= 256, the contract's ceiling). 128 over the 384-wide trunk "
                         "embedding measures at 1.07x the frozen actor's CPU forward time against "
                         "a 1.5x budget; see `python -m f1sim.learn.budget`")
    ap.add_argument("--memory-critic", default="own", choices=("own", "none"),
                    help="'own' gives the critic its own GRU (it already has its own stem, and it "
                         "reads the privileged vector the actor must never see, so sharing the "
                         "recurrence would be the one place a value gradient reached the actor's "
                         "trunk). 'none' leaves the critic feedforward")
    ap.add_argument("--scan-channels", default="",
                    help=f"comma separated extra scan channels, off by default "
                         f"({','.join(SCAN_CHANNELS)}). 'memory': the closest return seen at each "
                         f"bearing in the last --scan-memory-tau seconds, decayed -- explicit cheap "
                         f"memory the GRU does not have to learn. 'edges': |r[i]-r[i-1]| per beam, "
                         f"so the crack between two boxes in a row reads as two discontinuities "
                         f"rather than an opening. Both are pure arithmetic on the scan the env "
                         f"already emits (0.03 ms of a 25 ms step, measured) and both start as "
                         f"zeroed input columns, so a warm start is still bit-identical")
    ap.add_argument("--scan-memory-tau", type=float, default=2.0,
                    help="[s] time constant of the decayed scan-occupancy channel")
    ap.add_argument("--scan-deltas", action="store_true", help="append temporal scan differences for a new model without --init")
    ap.add_argument("--temporal-encoder", choices=["cnn", "gru"], default="cnn")
    ap.add_argument("--scan-stem", choices=["plain", "resnet"], default="resnet",
                    help="scan encoder for a new model without --init (an --init checkpoint keeps its own)")
    ap.add_argument("--procedural-obstacles", type=float, default=0.0, metavar="FRAC",
                    help="share of env resets that get a freshly drawn obstacle layout, placed as "
                         "analytic props from the hard-obstacle patterns (0 = off, and off is "
                         "byte-identical to a run without the flag)")
    ap.add_argument("--procedural-density", type=float, default=1.0, metavar="PER10M",
                    help="patterns per 10 m of lap when --procedural-obstacles is on")
    ap.add_argument("--procedural-max-props", type=int, default=0, metavar="N",
                    help="prop slots per env (0: from the density and the longest lap). Every slot "
                         "costs the beam tracer one pass per step, so this is the cost dial")
    ap.add_argument("--procedural-raceline-margin", type=float, default=0.25, metavar="M",
                    help="[m] kept clear either side of the raceline, beyond the car's half-width, "
                         "so the teacher opponents are never routed through a prop")
    ap.add_argument("--raceline-margin", type=float, default=None,
                    help="[m] free space the opponents' raceline keeps from the boundary (default 0.40)")
    ap.add_argument("--teacher-grip", choices=["true", "nominal", "conservative"], default="true",
                    help="grip the teacher opponents' speed profile assumes")
    ap.add_argument("--teacher-recover-time", type=float, default=0.0,
                    help="[s] >0: cap the teacher's commanded speed at what can still be steered back onto the lane "
                         "(v <= a_lat * t / heading_error). 0 keeps the old behaviour, where a car facing "
                         "backwards on the line is told to carry full racing speed")
    a = ap.parse_args()
    opp_cfg.validate(a)
    a.scan_channels = [c.strip() for c in str(a.scan_channels).split(",") if c.strip()]
    unknown = [c for c in a.scan_channels if c not in SCAN_CHANNELS]
    if unknown:
        raise SystemExit(f"--scan-channels {unknown}: known channels are {', '.join(SCAN_CHANNELS)}")
    if a.memory != "off" and a.cond != "none":
        raise SystemExit("--memory with --cond is not a supported combination: both migrate the "
                         "same checkpoint through a different loader, and nothing has measured the "
                         "two zero-initialised projections together.")
    if (a.memory != "off" or a.scan_channels) and a.controller != "legacy":
        raise SystemExit(
            f"--memory/--scan-channels with --controller {a.controller} is not a validated "
            f"combination: the memory work is measured against the legacy tracker only, and a "
            f"policy that learns to lean on a friction-limited controller *and* on memory would "
            f"have two untested changes in one result. Run --controller legacy.")
    if a.memory != "off" and a.minibatch < a.horizon:
        raise SystemExit(f"--memory {a.memory} needs --minibatch >= --horizon ({a.minibatch} < "
                         f"{a.horizon}): a recurrent update's minibatches are whole env chunks of "
                         f"the horizon, so a minibatch smaller than one chunk cannot be formed.")
    if a.procedural_obstacles:
        if not 0.0 < a.procedural_obstacles <= 1.0:
            raise SystemExit(f"--procedural-obstacles {a.procedural_obstacles}: it is the share of "
                             f"resets that get a layout, so it lives in (0, 1].")
        if not a.procedural_density > 0:
            raise SystemExit(f"--procedural-density {a.procedural_density}: at zero no pattern is "
                             f"ever placed and the run is silently the unflagged one.")
    elif a.procedural_density != 1.0 or a.procedural_max_props or a.procedural_raceline_margin != 0.25:
        raise SystemExit("--procedural-density / --procedural-max-props / "
                         "--procedural-raceline-margin without --procedural-obstacles: nothing "
                         "draws a layout, so these would silently do nothing.")
    if a.opp_token != "off":
        if a.race_size < 2:
            raise SystemExit(f"--opp-token {a.opp_token} needs --race-size > 1: with one car per "
                             f"race the block is the zeros an absent opponent already produces.")
        if a.action_mode != "plan":
            raise SystemExit(f"--opp-token {a.opp_token} needs --action-mode plan: the future "
                             f"columns are the opponent's own plan-tracker reference, and in "
                             f"'direct' mode there is no plan to read.")
    if a.overtake_sustained is not None:
        d_, t_, b_ = a.overtake_sustained
        if not (d_ > 0 and t_ > 0):
            raise SystemExit(f"--overtake-sustained {a.overtake_sustained}: the distance and the "
                             f"hold time are what make it 'sustained'; both must be positive.")
        if b_ > 0 and a.race_size < 2:
            raise SystemExit("--overtake-sustained needs --race-size > 1: with no opponent there is "
                             "nothing to lead.")
    if a.ttc_penalty > 0:
        if a.race_size < 2:
            raise SystemExit("--ttc-penalty needs --race-size > 1: time to contact is time to "
                             "contact with another car.")
        if not a.ttc_safe > 0:
            raise SystemExit(f"--ttc-safe {a.ttc_safe}: it is the horizon the penalty ramps over, "
                             f"so at 0 the term is either off or a step function at contact.")
    elif a.ttc_safe != 1.0:
        raise SystemExit("--ttc-safe without --ttc-penalty: nothing reads it, so it would silently "
                         "do nothing.")
    if a.kl_decay is None:
        a.kl_decay = a.total
    device = torch.device(a.device); torch.manual_seed(a.seed)
    names = common.track_names(a.tracks, draws=a.obstacle_draws, seed=a.seed)
    if a.envs % a.race_size:
        raise SystemExit(f"--envs {a.envs} is not a multiple of --race-size {a.race_size}: the envs are "
                         f"dealt into races of that size, so a remainder leaves a race that is never "
                         f"complete. Use {a.envs - a.envs % a.race_size} or {a.envs + a.race_size - a.envs % a.race_size}.")
    if a.overtake_bonus > 0 and a.race_size < 2:
        raise SystemExit("--overtake-bonus needs --race-size > 1: with no opponent there is nothing to pass.")
    need_rl = opp_cfg.needs_racelines(a) or a.lap_time_bonus > 0
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
                                                              reward_ttc=a.ttc_penalty, ttc_safe=a.ttc_safe,
                                                              **({} if a.overtake_sustained is None else
                                                                 {"overtake_hold_dist": a.overtake_sustained[0],
                                                                  "overtake_hold_time": a.overtake_sustained[1],
                                                                  "reward_overtake_hold": a.overtake_sustained[2]}),
                                                              opp_token=a.opp_token,
                                                              max_steps=int(a.episode_s * 40),
                                                              scan_stack=a.scan_stack, scan_stride=a.scan_stride, hist_len=a.hist_len,
                                                              action_mode=a.action_mode,
                                                              # every race / opponent / behaviour /
                                                              # spawn field, from the group the
                                                              # census shares (learn.opponent_config)
                                                              **opp_cfg.env_kwargs(a),
                                                              procedural_obstacles=a.procedural_obstacles,
                                                              procedural_density=a.procedural_density,
                                                              procedural_max_props=a.procedural_max_props,
                                                              procedural_raceline_margin=a.procedural_raceline_margin,
                                                              compile_tracker=_env_compile_tracker), seed=a.seed, rls=rls,
                          cfg=sim_cfg,
                          teacher_grip=a.teacher_grip,
                          teacher_recover_time=a.teacher_recover_time)
    print(f"sim backend: {a.sim_backend} (cfg.sim.compile={sim_cfg.sim.compile}, "
          f"compile_tracker={_env_compile_tracker})")
    print(f"opponents: {opp_cfg.describe(a)}", flush=True)
    if env.procedural is not None:
        print(env.procedural.describe())
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
        # The *tracker* arms are the ones validated solo; `tcs` shapes a speed command and is
        # indifferent to how many cars share the track, so it is not caught by this.
        if env.M > 1 and controller.base != "legacy":
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
    #: The memory / extra-channel configuration this run builds, in the form the model records.
    mem_cfg = memory_spec(kind=a.memory, hidden_size=a.memory_hidden,
                          critic=a.memory_critic) if a.memory != "off" else None
    chan_cfg = ({"channels": list(a.scan_channels), "memory_tau_s": float(a.scan_memory_tau)}
                if a.scan_channels else None)
    #: The future head is built when the term is on, and only then: `--aux-future 0` is the run it
    #: was, down to the state dict. A checkpoint that already carries one keeps it (the loaders read
    #: `meta`), so a resume does not have to repeat the flag to keep the head -- but it does have to
    #: repeat it to keep TRAINING the head, which is what the coefficient is.
    fut_cfg = (future_spec(k=a.aux_future_k, width=a.aux_future_width,
                           source=("memory" if mem_cfg else "trunk"))
               if a.aux_future > 0 else None)
    if fut_cfg and future_labelled_fraction(a.horizon, a.aux_future_k) <= 0.0:
        raise SystemExit(f"--aux-future-k {a.aux_future_k} needs --horizon > {a.aux_future_k - 1} "
                         f"(got {a.horizon}): the label for step t is the state at t + k, and a "
                         f"chunk shorter than the lookahead contains not one labelled step.")
    #: Only what the checkpoint does not already have is a warm start; the rest it keeps. Without
    #: this, leg two of a recurrent run -- same command line, `--init` now pointing at leg one's
    #: output -- goes down the warm-start path and is refused.
    init_meta = dict((torch.load(a.init, map_location="cpu").get("meta") or {})) if a.init else {}
    tok_cfg = a.opp_token if a.opp_token != "off" else None
    add_mem, add_chan, add_fut, add_tok = warm_start_additions(init_meta, mem_cfg, chan_cfg, fut_cfg, tok_cfg)
    if a.name_seed_fresh and not a.init:
        raise SystemExit("--name-seed-fresh without --init: with no warm start nothing is 'fresh' "
                         "and the whole model is drawn from --seed already.")
    if a.init and (add_mem or add_chan or add_fut or add_tok):
        # Warm start, not re-initialisation: every weight the checkpoint holds is copied by name,
        # the GRU's output projection is zero and any new scan-channel input column is zero, so the
        # actor's first action of this run is bit-identical to the one the original would have
        # produced. `tests/test_memory_model.py` is the check.
        model, extra, fresh = load_for_memory(
            a.init, device, add_mem, scan_channels=add_chan, priv_adapter=priv_adapter,
            future_head=add_fut, opp_token=add_tok,
            init_seed=(a.seed if a.name_seed_fresh else None),
            override={"n_stack": spec.scan_stack, "n_beams": spec.n_beams,
                      "proprio_dim": spec.proprio_dim, "priv_dim": critic_priv_dim,
                      "act_dim": env.act_dim})
        print(f"init from {a.init} with memory {add_mem} channels {add_chan} future {add_fut} "
              f"opp_token {add_tok} | {len(fresh)} fresh tensor(s), all zero-projected: {fresh[:4]}"
              + (f" | fresh tensors re-seeded from their own names (seed {a.seed})"
                 if a.name_seed_fresh else ""))
        a.scan_deltas = bool(model.meta.get("scan_deltas", False))
        a.temporal_encoder = str(model.meta.get("temporal_encoder", "cnn"))
        a.scan_stem = str(model.meta.get("scan_stem", "plain"))
    elif a.init and cond_dim:
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
        # allow_oracle: the trainer IS the simulator that produces the block, and this is the leg-two
        # path of an oracle run. Every other consumer still refuses it.
        model, extra = load_checkpoint(a.init, device, override={"n_stack": spec.scan_stack, "n_beams": spec.n_beams,
                                                                  "proprio_dim": spec.proprio_dim, "priv_dim": critic_priv_dim, "act_dim": env.act_dim},
                                       allow_conditional=False, priv_adapter=priv_adapter,
                                       allow_oracle=True)
        print("init from", a.init, extra.get("metrics"), "| re-initialized:", extra.get("skipped") or "nothing",
              "| resumed architecture:", {k_: v_ for k_, v_ in model.meta.items()
                                          if k_ in ("memory", "scan_channels", "future_head")} or "plain")
        a.scan_deltas = bool(model.meta.get("scan_deltas", False))
        a.temporal_encoder = str(model.meta.get("temporal_encoder", "cnn"))
        a.scan_stem = str(model.meta.get("scan_stem", "plain"))
    else:
        model = ActorCritic(spec.scan_stack, spec.n_beams, spec.proprio_dim, critic_priv_dim, act_dim=env.act_dim,
                            scan_deltas=a.scan_deltas, temporal_encoder=a.temporal_encoder,
                            scan_stem=a.scan_stem, cond_dim=cond_dim,
                            cond=cond_spec.to_meta() if cond_dim else None,
                            priv_adapter=priv_adapter, memory=mem_cfg,
                            scan_channels=chan_cfg, future_head=fut_cfg,
                            opp_token=tok_cfg).to(device)
    #: Read back from the model, never from the flags: an `--init` checkpoint that already carries
    #: memory keeps its own, and the rollout below has to agree with what was built.
    memory_on = bool(model.meta.get("memory"))
    scan_channels = list((model.meta.get("scan_channels") or {}).get("channels") or ())
    n_extra = len(scan_channels)
    roll_aug = (ScanAugment(scan_channels, spec.n_beams, env.B, device=device,
                            tau_s=float(model.meta["scan_channels"]["memory_tau_s"]))
                if n_extra else None)
    if memory_on or n_extra:
        from .memory import describe as _describe_memory
        print(f"policy memory: {_describe_memory(model.meta)}")
    #: Read back from the model for the same reason: a resumed checkpoint carries its own block, and
    #: the env has to be producing exactly the one the first layer's columns were laid out for.
    token_on = str(model.meta.get("opp_token") or "off")
    if token_on != env.opp_token:
        raise SystemExit(f"this checkpoint expects opponent tokens {token_on!r} and the env is "
                         f"producing {env.opp_token!r}: the proprio columns would mean different "
                         f"things on the two sides. Pass --opp-token {token_on}.")
    if token_on != "off":
        print(f"ORACLE: {describe_opp_token(token_on)}")
    if a.ttc_penalty > 0 or (a.overtake_sustained and a.overtake_sustained[2] > 0):
        print(f"reward additions: ttc {a.ttc_penalty} @ {a.ttc_safe} s | "
              f"sustained lead {a.overtake_sustained}")
    #: Read back from the model for the same reason `memory_on` is: a resumed checkpoint carries its
    #: own head, and the rollout has to collect labels exactly when there is something to score.
    future_cfg = dict(model.meta.get("future_head") or {})
    future_on = bool(future_cfg) and a.aux_future > 0
    future_k = int(future_cfg.get("k", a.aux_future_k))
    if future_cfg:
        from .future import describe as _describe_future
        frac = future_labelled_fraction(a.horizon, future_k)
        print(f"{_describe_future(model.meta)} | coefficient {a.aux_future} "
              f"({'training' if future_on else 'PRESENT BUT NOT TRAINED: --aux-future is 0'}) | "
              f"{frac:.0%} of each {a.horizon}-step chunk can carry a label")
        if future_on and env.M < 2:
            print("--aux-future on a solo env: the four opponent columns have no label and are "
                  "masked everywhere; only the ego speed / yaw-rate targets and a constant "
                  "presence logit train. This is an ablation, not the configuration the head is for.")
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
    if memory_on and ref.memory is not None and float(ref.memory.out.weight.detach().abs().max()) != 0.0:
        # Same argument as the conditioning check below, one projection further: the leash's
        # reference is the FEEDFORWARD original, and `minibatch_losses` evaluates it with the
        # recurrence switched off. That is only the original if the projection was still zero when
        # this copy was taken -- i.e. before any update.
        raise RuntimeError("the KL reference actor was captured after the memory projection had "
                           "trained; it must be the frozen feedforward baseline")
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
        "aux_future": float(a.aux_future), "future_head": dict(future_cfg) or None,
        "lab_oracle": bool(cond_spec.lab_oracle), "init": a.init, "seed": int(a.seed),
        "memory": dict(model.meta.get("memory") or {}) or None,
        "scan_channels": dict(model.meta.get("scan_channels") or {}) or None,
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
    #: What the POLICY sees on the channel axis: the stacked frames plus any extra channel. The
    #: channels are stored rather than recomputed during the update, because the occupancy one is
    #: itself a recurrence over the episode and a chunk replayed from a cleared memory would train
    #: the policy on an input it never saw.
    k_in = k + n_extra
    buf_scan = torch.zeros(T, B, k_in, N, device=device, dtype=torch.float16)
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
    # ---- the auxiliary future head's labels. `buf_fut_lab[t]` is the privileged label row for the
    # state at time t, for t = 0 .. T -- T + 1 rows, because the label for step t is the row at
    # t + k and the chunk's last state is the one the value bootstrap already visits. No extra
    # simulator step and no second rollout: the label is a read of the state the rollout is standing
    # in. `buf_fut_break[t]` marks the transition t -> t+1 as crossing an episode boundary for
    # SOMEBODY IN THE RACE, which is the condition under which the label at t + k belongs to a
    # different situation (`F1VecEnv.race_boundary`). `future.align_future_targets` turns the pair
    # into (target, valid) after the rollout.
    buf_fut_lab = torch.zeros(T + 1, B, FUTURE_LABEL_DIM, device=device) if future_cfg else None
    buf_fut_break = torch.zeros(T, B, device=device) if future_cfg else None
    # ---- recurrent state. `h_*` are the live states, full env width (the policy acts for every
    # car, including the self-play opponents); the buffers hold the learners' slice, which is what
    # the update replays. `buf_keep[t] = 0` marks a step whose predecessor ended an episode, so
    # truncated BPTT masks the hidden state back to zero exactly where the rollout did.
    h_actor = model.actor.initial_hidden(env.B, device=device) if memory_on else None
    h_critic = model.critic.initial_hidden(env.B, device=device) if memory_on else None
    buf_h0_actor = None if h_actor is None else torch.zeros(h_actor.shape[0], B, h_actor.shape[2], device=device)
    buf_h0_critic = None if h_critic is None else torch.zeros(h_critic.shape[0], B, h_critic.shape[2], device=device)
    buf_keep = torch.ones(T, B, device=device) if memory_on else None
    #: Envs per recurrent minibatch. `--minibatch` stays a sample count so that the two modes are
    #: comparable; a recurrent minibatch is whole env chunks, so it is that count divided by the
    #: horizon, and the number of samples per step is unchanged.
    env_chunk = max(1, a.minibatch // T) if memory_on else 0
    if memory_on:
        print(f"recurrent PPO: truncated BPTT over {T} steps, minibatches of {env_chunk} env "
              f"chunk(s) = {env_chunk * T} samples, hidden reset on term|trunc per env")

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
        if memory_on:
            # The state each env carries INTO this chunk, detached: truncated BPTT starts here and
            # the gradient does not run back into the last update's graph.
            buf_h0_actor.copy_(h_actor.detach()[:, lid])
            if buf_h0_critic is not None:
                buf_h0_critic.copy_(h_critic.detach()[:, lid])
            buf_keep.fill_(1.0)
        ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and device.type == "cuda")
        with torch.no_grad():
            for t in range(T):
                scan, pro = flatten_obs(obs)
                if roll_aug is not None:
                    scan = roll_aug(scan)                  # the extra channels, advanced one step
                # Frozen here, from the privileged vector belonging to THIS observation, before the
                # step advances the env or a reset re-draws `mu`.
                cond_t = (cond_mod.make_condition(a.cond, cond_spec, priv, env.priv_mu_index)
                          if cond_dim else None)
                # Pre-action, and before the policy is asked for anything: the friction this step's
                # plan will be tracked under has to be in the MPC before the plan exists.
                # The truth handed over here is for B3's safety metrics only and is read from the
                # privileged vector belonging to THIS observation, by this loop -- never from inside
                # the controller, whose estimated path has to work on a car that has no `P`.
                if controller.base != "legacy":
                    controller.observe_truth(priv[:, env.priv_mu_index])
                controller.pre_action(obs)
                with ac:
                    act, logp, h_actor_next = sample_rollout_action(model, scan, pro, cond_t, h_actor)
                    val, h_critic_next = model.critic.step(scan[lid], pro[lid], priv[lid],
                                                           None if h_critic is None else h_critic[:, lid])
                    val = val.float()
                act, logp = act.float(), logp.float()
                buf_scan[t] = scan[lid].half(); buf_pro[t] = pro[lid]; buf_priv[t] = priv[lid]; buf_act[t] = act[lid]; buf_logp[t] = logp[lid]; buf_val[t] = val
                if buf_fut_lab is not None:
                    # Before the step: `last_result` is still the state this action is taken from,
                    # which is the state `priv` above was read from.
                    buf_fut_lab[t] = env.future_labels()[lid]
                if cond_dim:
                    buf_cond[t] = cond_t[lid].float()
                obs, rew, term, trunc, info = env.step(act.clamp(-1, 1))
                # Advanced but NOT yet reset: the truncation bootstrap below reads the value of the
                # terminal observation, which belongs to the episode that just ended.
                if memory_on:
                    # Kept in float32 whatever `--amp` is doing. The hidden state is carried for a
                    # whole episode -- a thousand steps -- so accumulating it in bf16 would compound
                    # the rounding of every step into the one tensor the policy reads back; the
                    # GRU's own arithmetic is still autocast, because that is one step deep.
                    h_actor = h_actor_next.float()
                    if h_critic is not None:
                        # Only the learners' rows advanced (the critic is never run on the
                        # teacher-driven cars), so only those are written back.
                        h_critic = h_critic.index_copy(1, lid, h_critic_next.float())
                # Immediately: the issued command has to be taken from the spy's pre-reset snapshot,
                # and the episode boundaries recorded, before anything else advances the env.
                controller.post_step(term, trunc)
                priv = env.privileged(env.last_result)
                buf_rew[t] = rew[lid]; buf_done[t] = term[lid].float(); buf_trunc[t] = trunc[lid].float()
                if buf_fut_break is not None:
                    # Any car of the race, not just this one: an opponent that crashed was respawned
                    # behind the field inside this same step, so its pose half a second from now is
                    # not the continuation of the motion the head is being asked to extrapolate.
                    buf_fut_break[t] = env.race_boundary(term | trunc)[lid].float()
                if "on_policy" in info:
                    buf_mask[t] = info["on_policy"][lid].float()
                buf_reward_components[t] = torch.stack([info["reward_components"][key][lid] for key in REWARD_COMPONENT_KEYS], 1)
                ep_stats["lap_time"] += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
                buf_final_val[t].zero_()
                if "final" in info:
                    f = info["final"]; m = env.learner[f["ids"]]
                    final_scan, final_pro = flatten_obs(info["final_obs"])
                    if roll_aug is not None:
                        # `preview`, not `__call__`: a terminal observation is scored and never
                        # acted on, so advancing the occupancy memory for it would leave the next
                        # episode carrying a step it did not take.
                        final_scan = roll_aug.preview(final_scan, f["ids"])
                    with ac:
                        final_val = model.critic.step(
                            final_scan, final_pro, info["final_priv"],
                            None if h_critic is None else h_critic[:, f["ids"]])[0].float()
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
                if memory_on:
                    # The episode boundary, applied to everything that remembers: the two hidden
                    # states and the decayed scan-occupancy channel. `obs` is already the fresh
                    # episode's first observation (the env auto-resets inside `step`), so the state
                    # entering step t+1 has to be the state of a car that has just spawned.
                    done_now = term | trunc
                    h_actor = reset_hidden(h_actor, done_now)
                    h_critic = reset_hidden(h_critic, done_now)
                    if roll_aug is not None:
                        roll_aug.reset(done_now)
                    if t + 1 < T:
                        buf_keep[t + 1] = (~done_now[lid]).float()
            scan, pro = flatten_obs(obs)
            if roll_aug is not None:
                # The bootstrap value of the state the next chunk starts from: previewed, because
                # the next chunk's first step advances the channels itself.
                scan = roll_aug.preview(scan)
            if buf_fut_lab is not None:
                # The (T + 1)-th label row: the state the chunk ends in, the same one the bootstrap
                # value below is taken from. This is what makes step T - k the last labelled step
                # rather than T - k - 1.
                buf_fut_lab[T] = env.future_labels()[lid]
            with ac:
                buf_val[T] = model.critic.step(scan[lid], pro[lid], priv[lid],
                                               None if h_critic is None else h_critic[:, lid])[0].float()
            adv = compute_gae(buf_rew, buf_val, buf_done, buf_trunc, buf_final_val, a.gamma, a.lam)
            ret = adv + buf_val[:T]
        t_roll = tm.lap()
        # ---------------- update
        model.train()
        n = T * B
        f_scan = buf_scan.reshape(n, k_in, N); f_pro = buf_pro.reshape(n, P); f_priv = buf_priv.reshape(n, priv_dim)
        f_act = buf_act.reshape(n, env.act_dim); f_logp = buf_logp.reshape(n); f_adv = adv.reshape(n); f_ret = ret.reshape(n); f_val = buf_val[:T].reshape(n)
        f_mask = buf_mask.reshape(n)
        f_cond = buf_cond.reshape(n, buf_cond.shape[-1]) if cond_dim else None
        # The label for step t is the row recorded at t + k, dropped where that row is outside this
        # chunk or on the far side of a reset. Done once per update, here, so the minibatch loop
        # slices an aligned (T, B, D) block exactly as it slices `priv`.
        fut_tgt, fut_valid = (align_future_targets(buf_fut_lab, buf_fut_break, future_k)
                              if buf_fut_lab is not None else (None, None))
        f_fut = fut_tgt.reshape(n, FUTURE_LABEL_DIM) if fut_tgt is not None else None
        f_fut_valid = fut_valid.reshape(n) if fut_valid is not None else None
        # advantage statistics over the policy's own samples only; a teacher-driven car's advantages
        # are not the policy's and would otherwise set the scale everything else is normalised by
        w_all = f_mask / f_mask.sum().clamp_min(1.0)
        adv_mean = (f_adv * w_all).sum(); adv_std = ((f_adv - adv_mean) ** 2 * w_all).sum().sqrt()
        f_adv = (f_adv - adv_mean) / (adv_std + 1e-8)
        stats = {"pg": [], "vf": [], "ent": [], "kl_ref": [], "approx_kl": [], "clipfrac": []}
        freeze_actor = update < a.critic_warmup
        hyper = PPOHyper(clip=a.clip, vf=a.vf, ent=a.ent, kl_coef=kl_coef, aux_grip=a.aux_grip,
                         aux_opp=a.aux_opp, aux_future=(a.aux_future if future_on else 0.0),
                         priv_mu_index=int(env.priv_mu_index), m_gt_1=bool(env.M > 1))
        # Feedforward: minibatches are random SAMPLES, as they always were. Recurrent: minibatches
        # are whole env chunks of the horizon, because the update has to replay each env's chunk in
        # order from the hidden state the rollout was in -- a shuffled sample has no predecessor to
        # carry state from. `arange(T) * B + col` is the same flattening `(T, B).reshape(n)` uses,
        # so the flat buffers are indexed by exactly the rows the sequence block contains.
        units = B if memory_on else n
        step_units = env_chunk if memory_on else a.minibatch
        for ep in range(a.epochs):
            perm = torch.randperm(units, device=device)
            for i in range(0, units, step_units):
                sel = perm[i:i + step_units]
                seq = None
                if memory_on:
                    cols = sel
                    idx = (torch.arange(T, device=device)[:, None] * B + cols[None, :]).reshape(-1)
                    scan = buf_scan[:, cols].float(); pro = buf_pro[:, cols]
                    priv_mb = buf_priv[:, cols]; act_mb = buf_act[:, cols]
                    c_mb = buf_cond[:, cols] if cond_dim else None
                    seq = (Hidden(buf_h0_actor[:, cols],
                                  None if buf_h0_critic is None else buf_h0_critic[:, cols]),
                           buf_keep[:, cols])
                    fut_mb = fut_tgt[:, cols] if fut_tgt is not None else None
                    fut_valid_mb = fut_valid[:, cols] if fut_valid is not None else None
                else:
                    idx = sel
                    scan = f_scan[idx].float(); pro = f_pro[idx]
                    priv_mb = f_priv[idx]; act_mb = f_act[idx]
                    c_mb = f_cond[idx] if cond_dim else None   # the stored one, never recomputed
                    fut_mb = f_fut[idx] if f_fut is not None else None
                    fut_valid_mb = f_fut_valid[idx] if f_fut_valid is not None else None
                # NOT `out`: that is the run directory, twenty lines below, and shadowing it makes
                # a run that trains perfectly and then cannot write its checkpoint.
                terms = minibatch_losses(model, ref, scan=scan, pro=pro, priv=priv_mb, act=act_mb,
                                         logp_old=f_logp[idx], adv=f_adv[idx], ret=f_ret[idx],
                                         val_old=f_val[idx], w=f_mask[idx], cond=c_mb, hyper=hyper,
                                         autocast=ac, freeze_actor=freeze_actor, sequence=seq,
                                         future=fut_mb, future_valid=fut_valid_mb)
                pg, vf, kl_ref, aux, aux_o = (terms["pg"], terms["vf"], terms["kl_ref"],
                                              terms["aux_grip"], terms["aux_opp"])
                loss = terms["loss"]
                opt.zero_grad()
                if cond_dim or controller_on or memory_on:
                    # Scoped to the conditioning, controller and memory arms so the legacy path
                    # keeps its exact behaviour.
                    # Without these, "the run finished" says nothing about whether a loss or a
                    # gradient ever went non-finite -- clip_grad_norm_ silently propagates a NaN, and
                    # a vacuously-empty loss dict looked like a passing finiteness check in the first
                    # smoke. With them, a completed run IS the evidence: every step's loss and total
                    # gradient norm were finite, or it stopped here. A recurrent update is exactly
                    # where an exploding gradient through time would first show.
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss at update {update} "
                                           f"(pg={float(pg):.4g} vf={float(vf):.4g} "
                                           f"kl_ref={float(kl_ref):.4g})")
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad,
                                                    error_if_nonfinite=bool(cond_dim or controller_on or memory_on))
                opt.step()
                with torch.no_grad():
                    stats["pg"].append(pg.item()); stats["vf"].append(vf.item())
                    stats["ent"].append(terms["entropy_mean"].item())
                    stats["kl_ref"].append(kl_ref.item())
                    stats["approx_kl"].append(terms["approx_kl"].item())
                    stats["clipfrac"].append(terms["clipfrac"].item())
                    if memory_on: stats.setdefault("grad_norm", []).append(float(gn))
                    if a.aux_grip > 0: stats.setdefault("aux_grip", []).append(aux.item())
                    if a.aux_opp > 0 and env.M > 1: stats.setdefault("aux_opp", []).append(aux_o.item())
                    if future_on:
                        stats.setdefault("aux_future", []).append(terms["aux_future"].item())
                        for key, value in terms["aux_future_parts"].items():
                            stats.setdefault(f"aux_future/{key}", []).append(float(value))
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
                   **({"loss/grad_norm": float(np.mean(stats["grad_norm"]))} if stats.get("grad_norm") else {}),
                   **({"loss/aux_grip_mse": float(np.mean(stats["aux_grip"]))} if stats.get("aux_grip") else {}),
                   **({"loss/aux_opp_mse": float(np.mean(stats["aux_opp"]))} if stats.get("aux_opp") else {}),
                   # the total the coefficient multiplies, then every component of it: a head whose
                   # total falls because one easy column collapsed is not a head that learned the
                   # opponent, and only the split says which happened
                   **({"loss/aux_future_mse": float(np.mean(stats["aux_future"]))} if stats.get("aux_future") else {}),
                   **{f"loss/aux_future/{key.split('/', 1)[1]}": float(np.mean(v))
                      for key, v in stats.items() if key.startswith("aux_future/")},
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
            if a.metrics_jsonl:
                with open(a.metrics_jsonl, "a") as f_:
                    f_.write(json.dumps({k_: (float(v_) if isinstance(v_, (int, float, np.floating))
                                              else v_) for k_, v_ in log.items()}) + "\n")
            print(f"upd {update}/{n_updates} steps {(steps_base + steps_done)/1e6:.1f}M cap {cap:.1f} | rew/step {log['rollout/reward_per_step']:.3f} "
                  f"coll {log.get('episode/collisions_per_km', float('nan')):.1f}/km prog {log.get('episode/progress_m', float('nan')):.0f} m "
                  f"lap {log.get('episode/lap_time_s', float('nan')):.1f} s | gate {log.get('curriculum/gate_coll_per_km', float('nan')):.1f} "
                  f"({log.get('curriculum/tracks_scored', 0):.0f} tk) | kl_ref {log['loss/kl_ref']:.3f}"
                  + (f" | fut {log['loss/aux_future_mse']:.3f}" if 'loss/aux_future_mse' in log else "")
                  + f" | {log['time/env_steps_per_s']:.0f} steps/s", flush=True)
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
