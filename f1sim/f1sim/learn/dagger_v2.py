"""DAgger for the v2 actor (`learn.model_v2`): `python -m f1sim.learn.dagger --arch v2 ...`.

The redesign's stage 2 (docs/research/architecture-redesign-2026-10-13.md 6장 2) with its corrections
(appendix "완결성 비평" B1-B3, appendix 2) and the WP4 output (output-format-2026-10-14.md 5):

* **Recurrent sequences with burn-in.** Every update is a batch of contiguous runs of one env's
  steps, `--seq-burn` (40) steps that only warm the GRU state up -- no graph, no loss -- and then
  `--chunk-length` (80) steps that are scored. Without the burn-in, chunk 80 left 52.6 % of the memory
  labels unanswerable (judge-learnability-minimal/chunk_mask.json); burn-in 40 + chunk 80 is what G1
  ran (occluded-within-1-s position R^2 0.609, no memory -0.111). Critique B2: v1's `--seq-burn` was
  parsed and never used in training; here it is the burn-in.
* **The imitation loss in the decoded space** (`train_v2.bc_loss_wp4`): Huber on the four
  waypoints' lateral position [m] and speed [m/s] -- an error costs what it does to the plan the
  tracker is handed, not what it does to a normalised number whose authority differs 35x between
  kappa_1 and kappa_4.
* **B8 on the sequences** (critique B1: v1's recurrent DAgger set the auxiliary heads to None): the
  opponent-slot head on every scored step, with the window's knowability mask -- a car is a target
  only if the state, started at the burn-in's first step, has seen it within the last 3 s. The
  nearest-visible-car head behind its own coefficient (the per-beam car head is gone, see
  `PERCEPTION_AUX_DEFAULTS`).
* **Left-right mirroring**, p = `--mirror-prob` (0.5) per sequence (`train_v2.Mirror`; G1's
  appendix-2 correction: the flip took the visible lateral error p50 from 0.138 to 0.097 m).
* **Per-loss gradient norms** on the shared trunk (BC vs B8 vs the others) and the **auxiliary
  balance** (`train_v2.AuxBalance`): every `--aux-balance-every` (10) updates each auxiliary's
  multiplier is set from those norms so that it pulls the trunk at most `--aux-balance-rho` (0.5) x
  the imitation loss -- UNIT 4's smoke measured B8 at 1.8x and the nearest-car head at 4.8x the
  imitation's pull. A measured update is assembled from the per-loss gradients themselves
  (`train_v2.accumulate_grads`), so measuring costs one backward pass per auxiliary rather than one
  per loss plus the update's own (+7 % of the training compute with B8 at every 10, see
  `train_epochs`). With the balance off (`--aux-balance-every 0`) the norms are measured every
  `--log-every` updates, as before.

Teacher labels. A v2 run speaks WP4. `--wp4-label env` asks the env's teacher-label API
(`F1VecEnv.teacher_label`), which is where the wp4-speaking teachers of feat/v2 plug in; on a tree
without them that call raises, and the run says so. `--wp4-label linear-refit` is this branch's
stand-in for tests and smoke runs: the raceline teacher's own LINEAR plan, its first window refitted
to the four waypoints (`mpc.wp4_refit`) and its speed line read at them (`train_v2.linear_to_wp4`) --
the same refit recipe, so the decode, the refit and the mirror tables are all exercised, but not a
label to train the car's policy on.

What DAgger does not train: the critic (PPO's critic warm-up does) and `log_std` (imitation never
touches it; the per-dimension initial values of B6 reach PPO as they were built).

The race it is collected in (UNIT 5b, 2026-10-14):

* **Scripted opponents and the league plan LINEAR** (`--opponent-speed-mode`, default linear in a
  race): the measured choice of the wp4 label gates (RESULTS 7); a league checkpoint is shown its own
  observation contract (`learn.seat_obs`).
* **One speed cap, the fixed `--speed-cap`**, for the collection and the evaluations alike. The
  actor does not observe it; the share of learner steps whose executed plan it clipped is logged
  (`collect/speed_capped_frac`, `info["speed_capped"]`).

Removed 2026-10-14 (the code, not a switch), because no measurement stood behind either:

* **The learner's random speed cap** (`--learner-cap-range` / `--learner-cap-prob`: U(3, 9) m/s on a
  share of the episodes, not observed). Its premise -- that a policy trained under 9 m/s meets a speed
  history out of its distribution at the car's 4 m/s cap (critique B5) -- was a guess, never measured.
  It is the prime suspect for the copycat study's collapsed first iteration: the labels are the
  teacher's uncapped intent while 33-43 % of the learner's steps ran capped, so the actions the
  student had issued predicted a label better than the speed it measured.
* **The BC clearance mask** (`--bc-clearance`, 0.3 m: no imitation of a label planned within 0.3 m of
  another car). Its premise -- the teacher does not see the cars -- is false for the labels a v2 run
  imitates: `--wp4-label env` is the interactive guard over the raceline teacher
  (`F1VecEnv.teacher_label`), which reads the other cars and chooses to hold the line, slow down,
  stop or evade near them (v2 smoke, within 2 m of a car: line 46-59 %, slow 27-40 %, stop 5-10 %,
  evade 4-7 %). The mask took exactly those car-handling labels out of the imitation.

The copycat study (docs/research/copycat-2026-10-14.md): three iterations on e3781a7 (v2smoke_b) made
the student worse -- ICCAS solo completion 55 -> 11 -> 2 of 128 -- while its output speed followed the
speeds it had issued. Two remedies are options here: `--v2-ego-actions off` (the contract without the
issued actions, `ObsSpecV2.ego_actions`) and `--action-dropout P` (`train_v2.ActionDropout`, the action
columns of a sequence zeroed with probability P in training only). And labels are written only for
the rows the action drives (`label_rows`): the guard used to label all four cars of a race for the one
row a race stores.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
import torch

from . import common
from . import conditioning as cond_mod
from .aux_v2 import aux_spec, nearest_car_loss
from .model import load_checkpoint, save_checkpoint
from .model_v2 import ARCH, ActorCriticV2, init_stems_from_v1
from .obs import EGO_ACTIONS_DEFAULT, ObsSpecV2
from .train_v2 import (BALANCE_BETA, BALANCE_EVERY, BALANCE_RHO, BC_CONSTANTS, ActionDropout, AuxBalance, Mirror,
                       accumulate_grads, actor_trunk_parameters, aux_heldout_metrics, b8_sequence_loss, bc_loss_wp4,
                       linear_spec_of, linear_to_wp4, loss_grads, select_tracks, steps_since_window_start)

#: Where a v2 run's teacher labels come from (see the module docstring).
WP4_LABEL_SOURCES = ("env", "linear-refit", "policy")

#: The v2 defaults of flags v1 also has, applied when the command line does not set them: the
#: redesign's burn-in 40 + chunk 80 (G1), and the plan / dial contract every v2 actor is built for.
V2_DEFAULTS = {"chunk_length": 80, "seq_burn": 40}
#: The perception auxiliaries every v2 run trains unless told otherwise -- the perception gate's
#: decision (2026-10-14, scratchpad/arch/gates/percep/PREREG.md + results.txt; the coordinator's call)
#: as amended the same day: ONE frame, B8 and the nearest-visible-car head (auxiliary B). Boot-camp
#: data (1.06 M rows, 2 seeds x 4 folds, episode bootstrap 400), V3 (1 frame + B + B8) against V1
#: (1 frame + B8):
#:   nearest visible car |dy| p50 within 3 m   0.274 vs 0.422 m  (-0.148 [-0.161, -0.132])
#:   its distance error p50 within 3 m          0.345 vs 0.536 m  (-0.191 [-0.209, -0.176])
#:   B8 |dy| p50, seen within 3 m                0.331 vs 0.407 m  (-0.076 [-0.086, -0.066])
#:   contact-in-1-s AUC (BCE head)               0.887 vs 0.804    (+0.083 [+0.067, +0.101])
#: Frames bought nothing with the auxiliaries on (2 / 3 frames vs 1: every metric "no measurable
#: difference", V6 - V4, V7 - V4 -- measured with auxiliary A on as well), so 1 (`v2_frames`). No
#: variant passed the pre-registered bar (|dy| <= 0.15 m and AUC >= 0.85).
#:
#: The per-beam car / not-car head (auxiliary A, `--aux-beam-car`, default 5 until 2026-10-14) is
#: gone, code and flag. It was added on the premise that picking the cars out of the props is the
#: bottleneck, and the gate's own diagnosis refuted that premise: the nearest-car |dy| is the same with
#: and without a prop within 1.5 m (V1 0.423 vs 0.422 m), and of all its readouts within 3 m, 42-63 %
#: miss the car by more than 0.5 m and land on nothing, 4-5 % on a prop and 3-4 % on another car
#: (diag.py, V1-V7). On top of B -- which aims at the bottleneck that was measured, ranging -- it
#: bought -0.017 m [-0.024, -0.012] (V4 - V3, "no measurable difference").
#:
#: Coefficients are the gate's, applied BEFORE the adaptive balance (`AuxBalance`): B8 0.5 (w_pos 10,
#: w_vl 0.1), nearest car 0.5 (presence BCE + 10 pos^2 + 1 closing^2 + 0.1 v_lat^2). UNIT 3's heads
#: and losses (`aux_v2.NearestCarHead` / `b8_loss`, `nearest_car_loss`) are the gate's own
#: definitions, layer for layer and term for term, and its labels (>= 3 attributed valid HIT_CAR
#: returns, within 6 m of the LiDAR) are the env's (`F1VecEnv.nearest_visible_car_v2`): nothing in
#: them changed.
PERCEPTION_AUX_DEFAULTS = {"aux_b8": 0.5, "aux_near_car": 0.5}
#: Probability that a training sequence loses its issued-action columns (`--action-dropout`,
#: `train_v2.ActionDropout`) unless the command line says otherwise.
ACTION_DROPOUT = 0.0
#: The defaults of the v2-only flags (None on the parser, so a v1 run can refuse them).
#: `--v2-ego-actions` is not here: its default is --init's contract when there is one (`prepare_args`).
V2_ONLY_DEFAULTS = {"v2_frames": 1, "v2_init_stem": "", "wp4_label": "env", "mirror_prob": 0.5,
                    **PERCEPTION_AUX_DEFAULTS,
                    "aux_balance_every": BALANCE_EVERY, "aux_balance_rho": BALANCE_RHO,
                    "aux_balance_beta": BALANCE_BETA, "action_dropout": ACTION_DROPOUT}
#: The auxiliary losses, by the name `sequence_losses` gives them and the coefficient key they take.
AUX_TERMS = ("b8", "near")


def apply_race_defaults(a) -> dict:
    """The race a v2 run is collected / trained in, where the command line said nothing; returns what
    it applied. Shared by both v2 trainers (`ppo_v2.prepare_args` calls it too).

    * `opponent_speed_mode` "linear" in a race: the wp4 learner races scripted cars and league seats
      that plan linear -- the measured choice (wp4-label-gates RESULTS 7.4: against linear cars the
      wp4 labels' traffic contacts are level with linear's, 0.64 vs 0.61-0.69 per km pooled; with wp4
      cars every label touched more). Alone, or in self-play (every car the learner), there is no
      other driver, and no per-row formats are built."""
    applied = {}
    others = (int(getattr(a, "race_size", 1)) > 1
              and (getattr(a, "opponent", "policy") != "policy" or bool(getattr(a, "opp_slots", ""))))
    if getattr(a, "opponent_speed_mode", None) is None and others:
        a.opponent_speed_mode = applied["opponent_speed_mode"] = "linear"
    if a.opponent_speed_mode is not None and a.opponent_speed_mode not in ("linear", "wp4"):
        raise SystemExit("--opponent-speed-mode for a v2 run is 'linear' or 'wp4' (the per-row tracker's formats)")
    return applied


#: v1 options a v2 run cannot honour, with the value that means "not asked for". Refused rather than
#: ignored: a run that names an option and then runs without it looks like evidence it did nothing.
V2_REFUSED = {"memory": "off", "scan_channels": "", "maneuver": False, "opp_token": "off", "hard_frac": 0.0,
              "solo_fraction": 0.0, "speed_loss": "symmetric", "grip_quantile": 0.5, "aux_grip": 0.0,
              "aux_opp": 0.0, "aux_opp_visible": False, "seq_len": 0, "scan_deltas": False,
              "temporal_encoder": "cnn", "hist_len": 0}


def ego_actions_of(path: str) -> bool:
    """Whether the v2 checkpoint at `path` observes its issued actions (`ObsSpecV2.ego_actions`, a
    checkpoint older than the field: yes). Not a v2 checkpoint: the contract default -- the trainer
    refuses it for its own reasons, and a v1 stem seeds a fresh network either way."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    meta = ck.get("meta") or {}
    if meta.get("arch") != ARCH:
        return EGO_ACTIONS_DEFAULT
    return bool(ObsSpecV2.from_dict(meta.get("obs") or {}).ego_actions)


def resolve_ego_actions(a) -> bool:
    """`a.v2_ego_actions` ("on" / "off" / None) as the bool the env is built with: what the command line
    said, else the contract of the checkpoint `--init` continues, else `EGO_ACTIONS_DEFAULT`."""
    said = getattr(a, "v2_ego_actions", None)
    if said is not None:
        if said not in ("on", "off"):
            raise SystemExit("--v2-ego-actions is 'on' or 'off'")
        return said == "on"
    return ego_actions_of(a.init) if getattr(a, "init", "") else bool(EGO_ACTIONS_DEFAULT)


def explicit_flags(ap: argparse.ArgumentParser, argv=None) -> set:
    """The dests the command line set explicitly. argparse cannot tell "given, equal to the default"
    from "not given", and a v2 default must only replace the latter: parsed once more into a
    namespace pre-filled with a sentinel, argparse leaves every unset dest at the sentinel.

    An append-type option (`--rand-range`) copies what the dest holds and appends to the copy, so its
    sentinel is an empty list of its own: a bare `object()` there crashed every v2 run given such an
    option ("'object' object has no attribute 'append'", 2026-10-21). Given, the dest holds a new list."""
    import sys

    class _Unset(list):
        pass

    plain = object()
    sentinels = {x.dest: (_Unset() if isinstance(x, (argparse._AppendAction, argparse._AppendConstAction,
                                                      argparse._ExtendAction)) else plain)
                 for x in ap._actions if x.dest != argparse.SUPPRESS}
    ns = argparse.Namespace(**sentinels)
    ap.parse_args(sys.argv[1:] if argv is None else argv, namespace=ns)
    return {k for k, v in vars(ns).items() if v is not sentinels[k]}


def prepare_args(ap: argparse.ArgumentParser, a, argv=None) -> dict:
    """Make `a` a v2 run before anything is built: the v2 defaults where the command line said
    nothing, the plan / wp4 / dial contract forced, v1-only options refused. Returns the defaults
    it applied (recorded in every checkpoint)."""
    given = explicit_flags(ap, argv)
    for k, v in V2_ONLY_DEFAULTS.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    bad = [k for k, off in V2_REFUSED.items() if getattr(a, k) != off]
    if bad:
        raise SystemExit(f"--arch v2: {', '.join('--' + k.replace('_', '-') for k in bad)} belong to the v1 "
                         f"network or its observation; the v2 actor has its own memory (B5), auxiliary heads "
                         f"(--aux-b8 / --aux-near-car) and contract (learn.obs.ObsSpecV2)")
    if "action_mode" in given and a.action_mode != "plan":
        raise SystemExit("--arch v2 is the plan policy (its ego history holds WP4 plan actions)")
    if "speed_mode" in given:
        raise SystemExit("--arch v2 speaks WP4 (output-format-2026-10-14); --speed-mode is v1's")
    if a.cond == "true_mu":
        raise SystemExit("--arch v2 reads the grip dial (a deployable input); 'true_mu' is a lab oracle")
    if a.wp4_label not in WP4_LABEL_SOURCES:
        raise SystemExit(f"--wp4-label must be one of {WP4_LABEL_SOURCES}")
    if not 0.0 <= a.mirror_prob <= 1.0:
        raise SystemExit("--mirror-prob is a probability")
    if a.init and a.v2_init_stem:
        raise SystemExit("--init resumes a v2 student; --v2-init-stem seeds a fresh one from a v1 stem -- one or the other")
    if a.wp4_label == "policy" and not getattr(a, "label_policy", ""):
        raise SystemExit("--wp4-label policy refits a trained v1 policy's plans: name it with --label-policy CKPT")
    if a.wp4_label == "linear-refit" and a.teacher_kind != "raceline":
        raise SystemExit("--wp4-label linear-refit refits the RACELINE teacher's linear plan; the "
                         f"{a.teacher_kind} teacher's search would be silently dropped")
    if min(a.aux_b8, a.aux_near_car) < 0:
        raise SystemExit("auxiliary weights are >= 0")
    a.v2_ego_actions = "on" if resolve_ego_actions(a) else "off"
    if not 0.0 <= float(a.action_dropout) <= 1.0:
        raise SystemExit("--action-dropout is a probability")
    if a.updates is not None and int(a.updates) < 1:
        raise SystemExit("--updates is a number of optimiser updates per iteration (>= 1)")
    if float(a.action_dropout) > 0 and a.v2_ego_actions == "off":
        raise SystemExit("--action-dropout zeroes the ego's issued-action columns, and with --v2-ego-actions off "
                         "there are none: pick one of the two")
    check_balance_args(a)
    applied = apply_race_defaults(a)
    for k, v in V2_DEFAULTS.items():
        if k not in given:
            setattr(a, k, v)
            applied[k] = v
    a.action_mode, a.speed_mode, a.cond = "plan", "wp4", "dial"
    a.scan_stack, a.scan_stride = int(a.v2_frames), 1
    a.hist_len = ObsSpecV2().hist_len                   # the contract's 20 rows, stride 2
    if int(a.seq_burn) < 0 or int(a.chunk_length) < 1:
        raise SystemExit("--seq-burn must be >= 0 and --chunk-length >= 1")
    return applied


def check_balance_args(a) -> None:
    """`--aux-balance-*`, shared by both v2 trainers (`ppo_v2.prepare_args` calls it too)."""
    if int(a.aux_balance_every) < 0:
        raise SystemExit("--aux-balance-every is a number of updates (0 switches the balance off)")
    if not (math.isfinite(float(a.aux_balance_rho)) and float(a.aux_balance_rho) > 0):
        raise SystemExit("--aux-balance-rho is a positive ratio of gradient norms")
    if not 0.0 <= float(a.aux_balance_beta) < 1.0:
        raise SystemExit("--aux-balance-beta is an EMA weight in [0, 1)")


def make_balance(a, main: str, active) -> AuxBalance:
    """The run's `AuxBalance` over the auxiliaries that train (`active`, names as the trainer keys
    them), from the `--aux-balance-*` flags."""
    return AuxBalance(main, [k for k in AUX_TERMS if k in active], every=int(a.aux_balance_every),
                      rho=float(a.aux_balance_rho), beta=float(a.aux_balance_beta))


def env_kwargs(a) -> dict:
    """The EnvConfig fields a v2 run adds to the v1 ones `dagger.main` passes: contract v2 and the
    per-seat visibility the B8 / nearest-car labels are counted from. (`opponent_speed_mode` travels
    with the opponent flags, `opponent_config.env_kwargs`.)"""
    return {"obs_version": 2, "hist_stride": ObsSpecV2().hist_stride,
            "opp_visibility": bool(a.aux_b8 > 0 or a.aux_near_car > 0),
            "ego_actions": a.v2_ego_actions == "on"}


# ------------------------------------------------------------------------------- teacher labels
def label_rows(env) -> Optional[torch.Tensor]:
    """The env rows a label is needed for: the cars this step's action drives (`F1VecEnv.
    action_driven` -- the learners, and in self-play / mixed / pool races the seats the policy drives).
    A scripted, teacher or league car's command is the env's own, so its label would be thrown away.
    None when that is every row (solo, self-play). One host sync per step: which seats the policy
    drives is redrawn at race resets in mixed and pool races.

    Measured on v2smoke_b (512 cars, races of 4, e3781a7): labelling every car cost 0.118 s of each
    0.56 s collection step (21 %); one row in four is the learner's."""
    mine = env.action_driven()
    if bool(mine.all()):
        return None
    return torch.nonzero(mine).flatten()


class Wp4Labels:
    """(label (B, 8) normalised wp4, valid (B,) bool) for the env's current state, the teacher
    driving for friction `mu` (the dial's number, as `F1VecEnv.teacher_label` takes it)."""

    def __init__(self, env, teacher, source: str):
        if source not in WP4_LABEL_SOURCES:
            raise ValueError(f"wp4 label source must be one of {WP4_LABEL_SOURCES}, got {source!r}")
        self.env, self.teacher, self.source = env, teacher, source
        self.spec = env.tracker.spec
        if self.spec.speed_mode != "wp4":
            raise ValueError(f"a v2 run's env tracks 'wp4' plans; this one tracks {self.spec.speed_mode!r}")
        self.lin = linear_spec_of(self.spec)
        if source == "policy":
            from .policy_teacher import PolicyTeacher
            if not isinstance(teacher, PolicyTeacher):
                raise ValueError(f"--wp4-label policy reads a learn.policy_teacher.PolicyTeacher; got {type(teacher).__name__}")
            self.base = teacher
        if source == "linear-refit":
            from ..teacher import RacelineTeacher
            if not isinstance(teacher, RacelineTeacher):
                # not its `.base`: an interactive or rollout teacher's label is its search, and
                # refitting the raceline under it would drop the search without a word
                raise ValueError("--wp4-label linear-refit refits a raceline teacher's linear plan; "
                                 f"this teacher is {type(teacher).__name__}")
            self.base = teacher

    @torch.no_grad()
    def __call__(self, mu: Optional[torch.Tensor] = None, rows: Optional[torch.Tensor] = None):
        """`rows` (R,): label those env rows only (`label_rows`); the result is still (B, 8) / (B,),
        zero and invalid on every other row -- nothing reads them, and `step` overwrites their command."""
        env = self.env
        if rows is not None:
            lab_r, valid_r = self._label(mu, rows)
            lab = torch.zeros(env.B, lab_r.shape[1], device=lab_r.device, dtype=lab_r.dtype)
            valid = torch.zeros(env.B, dtype=torch.bool, device=lab_r.device)
            return lab.index_copy(0, rows.to(lab.device), lab_r), valid.index_copy(0, rows.to(lab.device), valid_r)
        return self._label(mu, None)

    def _label(self, mu, rows):
        env = self.env
        if self.source == "env":
            try:
                lab = env.teacher_label(self.teacher, mu=mu, rows=rows)
            except NotImplementedError as exc:
                raise SystemExit(f"the env's teacher does not speak wp4 on this tree ({exc}). The wp4 teachers "
                                 f"live on feat/v2; until it is merged run with --wp4-label linear-refit "
                                 f"(a stand-in for tests and smoke runs, see learn.dagger_v2).") from exc
            valid = getattr(self.teacher, "last_label_valid", None)
            if valid is not None and valid.shape[0] != lab.shape[0]:
                valid = None                    # a teacher that does not state validity per labelled row
        elif self.source == "policy":
            # a trained v1 policy's plan of this step (its own observation, recurrent state and held dial: `mu` is
            # not its business), refitted as the raceline's is below, under the same curvature bound
            v = env._tracker_plan_speed()
            a_lin = self.base.act()
            vp, P = env.cfg.vehicle, env.sim.P
            wb = P.get("lf", vp.lf) + P.get("lr", vp.lr)
            s_max = torch.as_tensor(P.get("s_max", vp.s_max), device=v.device, dtype=v.dtype).expand_as(v)
            k_lim = 0.85 * torch.minimum(torch.full_like(v, self.spec.kappa_max), torch.tan(s_max) / wb)
            lab = linear_to_wp4(a_lin.to(v.dtype), v, self.base.lin, self.spec, env.ecfg.v_max_policy, k_lim=k_lim)
            if rows is not None:
                lab = lab[rows]
            valid = None
        else:
            P = env.sim.P if mu is None else {**env.sim.P, "mu": mu.to(env.sim.P["mu"].dtype)}
            v = env._tracker_plan_speed()
            b = self.base
            a_lin = b.plan_action(env.sim.state, P, env.sim.tid, env.ecfg.v_max_policy, self.lin, plan_speed=v)
            # the teacher's own curvature bound (0.85 x steering-limited, `RacelineTeacher._plan_wp4`'s)
            wb = P.get("lf", b.vehicle.lf) + P.get("lr", b.vehicle.lr)
            s_max = torch.as_tensor(P.get("s_max", b.vehicle.s_max), device=v.device, dtype=v.dtype)
            s_max = s_max.clamp(max=b.steer_max).expand_as(v)
            k_lim = 0.85 * torch.minimum(torch.full_like(v, self.spec.kappa_max), torch.tan(s_max) / wb)
            lab = linear_to_wp4(a_lin, v, self.lin, self.spec, env.ecfg.v_max_policy, k_lim=k_lim)
            if rows is not None:
                lab = lab[rows]                 # the stand-in refits the whole batch and is cut
            valid = None
        ok = torch.isfinite(lab).all(1)
        valid = ok if valid is None else (valid.to(ok.device).bool() & ok)
        return torch.nan_to_num(lab), valid


# --------------------------------------------------------------------------------- the buffer
class SequenceBuffer:
    """Per-step storage of one collection, (steps, learners). A training sample is a contiguous run of
    one learner's steps; its frame stack is rebuilt from the stored newest frames at sampling time (the
    env's own history, which repeats an episode's first frame backwards, is what `frames_at` rebuilds).

    What one step stores: the newest frame (fp16), the ego vector (fp16 -- 42 numbers, 210 with the
    issued actions; the 1-s history in it), the teacher's label and its validity, whether the step began an episode, the
    dial column `c` the student was given, and -- when their heads train -- the B8 tracks (`select_tracks`:
    labels, candidates, AGE: knowability is decided per sampled window) and the nearest-visible-car
    target."""

    KEYS = ("scan", "ego", "label", "valid", "newep", "c", "b8_lab", "b8_cand", "b8_age", "near")

    def __init__(self, n_frames: int, frame_stride: int = 1):
        self.k, self.stride = int(n_frames), int(frame_stride)
        self.lists = {k: [] for k in self.KEYS}
        #: what the collection measured on its way (`collect`): the share of capped steps
        self.stats: dict = {}

    def add(self, **row):
        for k in self.KEYS:
            v = row.get(k)
            if v is None:
                continue
            v = v.detach()
            if k in ("scan", "ego", "b8_lab", "near"):
                v = v.to(torch.float16)
            elif k == "b8_age":
                v = v.clamp(max=32767).to(torch.int16)
            self.lists[k].append(v.cpu())

    def finalize(self):
        self.data = {k: torch.stack(v) for k, v in self.lists.items() if v}
        self.lists = {k: [] for k in self.KEYS}
        self.T, self.B = self.data["scan"].shape[:2]
        self.valid_count = int(self.data["valid"].sum())
        return self

    def __len__(self):
        return self.T * self.B

    def frames_at(self, t: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(n, K, N) frame stacks of rows (t, b): t, t - stride, ..., never reaching back past the
        step that began the episode (the env fills its history from an episode's first frame)."""
        S, NE = self.data["scan"], self.data["newep"]
        cur = t.clone()
        blocked = torch.zeros_like(t, dtype=torch.bool)
        out = []
        for _ in range(self.k):
            out.append(S[cur, b])
            for _ in range(self.stride):
                blocked = blocked | NE[cur, b] | (cur == 0)
                cur = torch.where(blocked, cur, cur - 1)
        return torch.stack(out, 1)

    def sample(self, m: int, length: int, device, generator: Optional[torch.Generator] = None) -> dict:
        """`m` runs of `length` contiguous steps, each from one learner: a dict of (length, m, ...)
        tensors on `device` (float32 where they were stored in half), plus `keep` (length, m) -- 0 at
        a step that began an episode, where the state must not be carried in. `generator`: a stream
        of the caller's (the held-out evaluation draws its windows without moving training's)."""
        length = max(1, min(int(length), self.T))
        t0 = torch.randint(self.T - length + 1, (m,), generator=generator)
        b = torch.randint(self.B, (m,), generator=generator)
        ts = (t0[None, :] + torch.arange(length)[:, None]).reshape(-1)
        bs = b[None, :].expand(length, m).reshape(-1)
        shape = lambda x: x.view(length, m, *x.shape[1:])
        out = {"scan": shape(self.frames_at(ts, bs)).to(device, torch.float32)}
        for k, v in self.data.items():
            if k == "scan":
                continue
            x = v[ts, bs]
            x = x.float() if x.dtype in (torch.float16, torch.float32) else x
            out[k] = shape(x).to(device)
        out["keep"] = (~out.pop("newep")).float()
        out["valid"] = out["valid"].bool()
        if "b8_age" in out:
            out["b8_age"] = out["b8_age"].long()
            out["b8_cand"] = out["b8_cand"].bool()
        return out


# ------------------------------------------------------------------------------- collection
def collect(env, model, labeler: Wp4Labels, steps: int, beta: float, device, buf: SequenceBuffer, noise: float,
            dial: cond_mod.DialDraw, cspec, b8: bool, near: bool) -> SequenceBuffer:
    """Roll the env for `steps`, label every state with the teacher, store the learners' rows.

    Iteration 0 (beta 1) drives the teacher; later ones mix the student in with probability
    1 - beta, as v1 does. The student runs from its own recurrent state at every step -- also on the
    steps the teacher's action is taken -- and the state is zeroed where an episode ends, so the
    state it acts from is the one it would have on the car. The dial is drawn per episode
    (`conditioning.DialDraw`: mu - U(0, 0.3), exact with p 0.3, the redesign's B4) and the teacher
    drives for that same number.

    Every valid label is imitated. A v2 run's labels (`--wp4-label env`) are the guard's over the
    raceline teacher, which sees the other cars, so its choices near them are the ones to copy (the BC
    clearance mask that dropped them is gone, see the module docstring). What the collection saw is
    left on the buffer as `buf.stats`: the share of the stored steps whose executed plan the env's
    speed cap clipped (`info["speed_capped"]`)."""
    obs, _info = env.reset()
    # the primary learner of each race, as v1's DAgger stores: the other seats' trajectories are
    # driven by their own population and are not what this intervention visited
    ids = torch.arange(0, env.B, env.M, device=env.device)
    new_ep = torch.ones(env.B, dtype=torch.bool, device=env.device)
    h = model.actor.initial_hidden(env.B, device=device)
    z = lambda: torch.zeros((), dtype=torch.float64, device=env.device)
    tally = {"capped": z(), "steps": z()}
    with torch.no_grad():
        for _t in range(steps):
            scan, ego = obs["scan"], obs["ego"]
            dial.redraw(new_ep)
            mu_dial = dial.value()
            c = cond_mod.mu_to_c(mu_dial, cspec).to(scan.dtype)
            # the stored rows are labelled, and any other row the action drives (self-play seats); a
            # scripted or league car's label would be computed for a command `step` overwrites
            label, valid = labeler(mu=mu_dial, rows=label_rows(env))
            student = None
            if beta < 1.0:
                student, h = model.actor.step(scan.to(device), ego.to(device), c.to(device), h)
            row = {"scan": scan[ids, 0], "ego": ego[ids], "label": label[ids], "valid": valid[ids],
                   "newep": new_ep[ids], "c": c[ids]}
            if b8:
                tr = env.opponent_tracks_v2()
                lab3, cand3, age3 = select_tracks(tr["lab"], tr["cand"], tr["age"], tr["dist"])
                row.update(b8_lab=lab3[ids], b8_cand=cand3[ids], b8_age=age3[ids])
            if near:
                row["near"] = env.nearest_visible_car_v2()[ids]
            buf.add(**row)
            if student is None:
                a = label
            else:
                a_s = student.to(label.device)
                if noise > 0:
                    a_s = (a_s + noise * torch.randn_like(a_s)).clamp(-1, 1)
                a = torch.where((torch.rand(env.B, device=label.device) < beta)[:, None], label, a_s)
            obs, _rew, term, trunc, info = env.step(a)
            if "speed_capped" in info:
                tally["capped"] += info["speed_capped"][ids].sum()
                tally["steps"] += ids.numel()
            new_ep = (term | trunc).to(env.device)
            h = h * (~new_ep).to(h.dtype).to(h.device)[None, :, None]
    n = {k: float(v) for k, v in tally.items()}
    buf.stats = {"speed_capped_frac": n["capped"] / n["steps"] if n["steps"] > 0 else float("nan")}
    return buf


# ---------------------------------------------------------------------------------- training
#: Windows of a fresh collection the auxiliary heads are scored on before training sees it.
HELDOUT_WINDOWS = 16


@torch.no_grad()
def heldout_aux(model, buf: SequenceBuffer, *, burn: int, chunk: int, device, coefs: dict, seed: int) -> dict:
    """The auxiliary heads' held-out metrics (`train_v2.aux_heldout_metrics`) on a collection the
    weights were not trained on: the iteration's fresh buffer, scored BEFORE `train_epochs` uses it, so
    the numbers are those of the checkpoint the previous iteration saved. `HELDOUT_WINDOWS` windows
    of burn-in + chunk, drawn from a generator of their own (training's draws do not move).

    Why per iteration and never a selection rule: the perception gate's risk head overfit after ~2.7
    epochs while its perception auxiliaries kept improving (nearest-car loss 0.95 -> 0.27 over 12), so
    a checkpoint picked by a loss would lose them; the run keeps its last checkpoint
    (`student_latest.pt`) and these say what its heads read."""
    if not any(float(coefs.get(k, 0.0)) > 0 for k in AUX_TERMS) or len(buf) == 0:
        return {}
    length = min(burn + chunk, buf.T)
    b = burn if burn + chunk <= buf.T else min(burn, int(round(length * burn / max(1, burn + chunk))))
    s = buf.sample(HELDOUT_WINDOWS, length, device, generator=torch.Generator().manual_seed(int(seed)))
    was = model.training
    model.eval()
    try:
        out = model.actor.sequence(s["scan"], s["ego"], s["c"], keep=s["keep"], burn_in=b)
    finally:
        model.train(was)
    since = steps_since_window_start(s["keep"])[b:]
    lab = {k: s[k][b:] for k in ("near", "b8_lab", "b8_cand", "b8_age") if k in s}
    return aux_heldout_metrics(out, lab, since)


def sequence_losses(model, s: dict, *, burn: int, plan_spec, v_max: float, coefs: dict):
    """The v2 imitation losses on one batch of sequences `s` (`SequenceBuffer.sample`, mirrored or
    not): {"total", "bc", "bc_lat", "bc_speed", "b8", "b8_knowable", "near"} -- every part a
    tensor (zero where its coefficient is off), the parts that train carrying their graph."""
    actor = model.actor
    out = actor.sequence(s["scan"], s["ego"], s["c"], keep=s["keep"], burn_in=burn)
    sl = slice(burn, None)
    T, m = out["mu"].shape[:2]
    flat = lambda x: x[sl].reshape(T * m, *x.shape[2:])
    v_meas = flat(s["ego"])[:, 0] * v_max                    # speed@t: the speed the plan is laid out from
    valid = flat(s["valid"]).float()
    bc, parts = bc_loss_wp4(out["mu"].reshape(T * m, -1), flat(s["label"]), v_meas, plan_spec, v_max, weight=valid)
    zero = torch.zeros((), device=bc.device)
    res = {"bc": bc, "bc_lat": parts["lat"].detach(), "bc_speed": parts["speed"].detach(),
           "b8": zero, "b8_knowable": zero, "near": zero}
    total = bc
    if coefs.get("b8", 0.0) > 0 and "b8" in out:
        since = steps_since_window_start(s["keep"])[sl]
        b8, st = b8_sequence_loss(out["b8"], s["b8_lab"][sl], s["b8_cand"][sl], s["b8_age"][sl], since)
        res["b8"], res["b8_knowable"] = b8, st["knowable"]
        total = total + coefs["b8"] * b8
    if coefs.get("near", 0.0) > 0 and "near_car" in out:
        near = nearest_car_loss(out["near_car"].float(), s["near"][sl]).mean()
        res["near"] = near
        total = total + coefs["near"] * near
    res["total"] = total
    return res


def train_epochs(model, bufs, *, epochs: float, batch: int, burn: int, chunk: int, device, opt, log,
                 plan_spec, v_max: float, coefs: dict, mirror: Optional[Mirror], mirror_prob: float,
                 log_every: int = 25, max_grad: float = 1.0, balance: Optional[AuxBalance] = None,
                 dropout: Optional[ActionDropout] = None, updates: Optional[int] = None):
    """Truncated BPTT over sequences of `burn + chunk` steps; `batch` scored steps per update (so
    `batch // chunk` sequences). Every buffer is drawn in proportion to its valid labels, as v1.
    Returns (the last updates' mean loss, {each part's and each gradient norm's mean over the
    iteration}) -- the summary goes into the iteration's progress record, so a run that logs nowhere
    else still says what B8 weighed against the imitation.

    `balance` (`train_v2.AuxBalance`, main loss "bc"): on the updates it is due, each loss's gradient
    is taken on its own (`train_v2.loss_grads`), the norms on the shared trunk set the multipliers,
    and the update's gradient is ASSEMBLED from those per-loss gradients with the multipliers now in
    force -- the same gradient `total.backward()` would give (up to the order of the float sums), one
    backward pass per loss and none more. Every other update is the plain backward of the total at
    the multipliers in force. Measured on the CPU at the deployed shapes (1081 beams, 4 sequences of
    40 + 80): a measured update costs 1.73x a plain one with B8, 2.76x with all four losses of the
    time (the per-beam car head's included, gone since 2026-10-14), so at `every` 10 the balance
    costs +7 % / +18 % of the training compute (UNIT 4's way of measuring -- norms with the graph
    kept, then the update's own backward -- 2.31x / 3.43x). With the balance on, its measurements are
    also the logged rows; with it off, `log_every` measures as before.

    `dropout` (`train_v2.ActionDropout`, `--action-dropout`): after the mirror, each sequence loses its
    issued-action columns with the dropout's probability; the share it dropped is in the summary
    (`action_dropped_frac`).

    `updates` (`--updates`): this many optimiser updates, whatever the buffers hold, in place of
    `epochs` passes over their valid labels."""
    n_valid = sum(b.valid_count for b in bufs)
    log({"dagger/valid_labels": n_valid, "dagger/invalid_labels": sum(len(b) for b in bufs) - n_valid})
    if not n_valid:
        log({"dagger/skipped_no_valid_labels": 1, "dagger/optimizer_updates": 0})
        return 0.0, {}
    T_min = min(b.T for b in bufs)
    if burn + chunk > T_min:
        # a collection shorter than one sequence (a smoke run): keep the burn-in's share
        length = T_min
        burn = min(burn, int(round(T_min * burn / max(1, burn + chunk))))
        chunk = length - burn
        print(f"  sequences shortened to the {T_min}-step collection: burn-in {burn} + chunk {chunk}", flush=True)
    m = max(1, int(batch) // max(1, chunk))
    steps = int(updates) if updates else max(1, int(epochs * n_valid / (m * chunk)))
    weights = torch.tensor([float(b.valid_count) for b in bufs])
    trunk = actor_trunk_parameters(model.actor)
    params = [p for p in model.actor.parameters() if p.requires_grad]
    active = [k for k in AUX_TERMS if coefs.get(k, 0.0) > 0]
    hist = {k: [] for k in ("total", "bc", "bc_lat", "bc_speed", "b8", "b8_knowable", "near")}
    grads = {}
    t_measured, t_plain, n_measured = 0.0, 0.0, 0
    n_drop, n_seq = 0.0, 0
    for i in range(steps):
        b = bufs[int(torch.multinomial(weights, 1))]
        s = b.sample(m, burn + chunk, device)
        if mirror is not None and mirror_prob > 0:
            s = mirror.apply(s, torch.rand(m, device=s["scan"].device) < mirror_prob)
        if dropout is not None and dropout.p > 0:
            drop = dropout.draw(m, s["ego"].device)
            s = dropout.apply(s, drop)
            n_drop += float(drop.float().sum()); n_seq += m
        t0 = time.perf_counter()
        eff = coefs if balance is None else balance.coefs(coefs)
        terms = sequence_losses(model, s, burn=burn, plan_spec=plan_spec, v_max=v_max, coefs=eff)
        loss = terms["total"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite DAgger v2 loss at step {i}: " +
                               ", ".join(f"{k} {float(v):.4g}" for k, v in terms.items()))
        due = balance is not None and balance.due()
        # With the balance on, its measurements are the logged ones too (a row every `every` updates):
        # a measured update costs 1.7x a plain one with B8 (2.8x with all four losses of the time, CPU
        # at 1081 beams), so measuring again on --log-every's own schedule would add a third of that again.
        measure = due if (balance is not None and balance.enabled) else i % log_every == 0
        gn = None
        opt.zero_grad()
        if measure:
            # each loss's pull on the trunk the losses share, before the optimiser step: the
            # auxiliaries at their configured coefficient, BEFORE the balance's multiplier -- the
            # number the multiplier is computed from, and the one UNIT 4's logs carry
            parts = {"bc": terms["bc"], **{k: coefs[k] * terms[k] for k in active if terms[k].requires_grad}}
            gn, per = loss_grads(parts, params, norm_params=trunk, retain=False)
            if due:
                balance.observe(gn)
            w = {"bc": 1.0, **{k: (1.0 if balance is None else balance.scale.get(k, 1.0)) for k in active}}
            accumulate_grads(params, per, w)
            loss = terms["bc"].detach() + sum(w[k] * parts[k].detach() for k in parts if k != "bc")
            terms["total"] = loss
        else:
            loss.backward()
        total_gn = torch.nn.utils.clip_grad_norm_(params, max_grad)
        opt.step()
        if balance is not None:
            balance.tick()
        if device.type == "cuda":
            # free: the loss read-out below waits for the device anyway, and without it a plain
            # update's clock would stop before its kernels ran
            torch.cuda.synchronize()
        if gn is not None:
            t_measured += time.perf_counter() - t0; n_measured += 1
        else:
            t_plain += time.perf_counter() - t0
        for k in hist:
            hist[k].append(float(terms[k].detach()))
        if gn is not None:
            gn["total_clip_input"] = float(total_gn)
            for k, v in gn.items():
                grads.setdefault(k, []).append(v)
            span = balance.every if (balance is not None and balance.enabled) else log_every
            row = {f"dagger/{k}": float(np.mean(v[-span:])) for k, v in hist.items()}
            row.update({f"dagger/grad_{k}": v for k, v in gn.items()})
            if balance is not None:
                row.update({f"dagger/aux_scale_{k}": v for k, v in balance.scale.items()})
                row.update({f"dagger/aux_pull_{k}": v for k, v in balance.pull().items()})
            row.update({"dagger/train_step": i, "dagger/lr": opt.param_groups[0]["lr"]})
            log(row)
    log({"dagger/optimizer_updates": steps})
    summary = {k: float(np.mean(v)) for k, v in hist.items() if v}
    summary.update({f"grad_{k}": float(np.mean(v)) for k, v in grads.items()})
    summary["updates"] = steps
    if n_seq:
        summary["action_dropped_frac"] = n_drop / n_seq
    if balance is not None:
        summary.update({f"aux_scale_{k}": v for k, v in balance.scale.items()})
        summary.update({f"aux_pull_{k}": v for k, v in balance.pull().items()})
    # what measuring cost, from this iteration's own clocks: a measured update against a plain one,
    # both from the forward to the optimiser step (sampling and mirroring are outside either)
    n_plain = steps - n_measured
    if n_measured and n_plain:
        summary["measure_update_s"] = t_measured / n_measured
        summary["plain_update_s"] = t_plain / n_plain
        summary["measure_overhead_frac"] = (t_measured - n_measured * t_plain / n_plain) / max(1e-9, t_measured + t_plain)
    return float(np.mean(hist["total"][-500:])), summary


# ------------------------------------------------------------------------------------- main
def run(a, *, env, teacher, teacher_desc: str, names, device, applied: dict):
    """The DAgger loop for `--arch v2`, from the env and teacher `dagger.main` built."""
    from .dagger import rng_island
    spec = common.obs_spec(env)
    if not isinstance(spec, ObsSpecV2):
        raise SystemExit("--arch v2 needs the env built with obs_version 2")
    coefs = {"b8": float(a.aux_b8), "near": float(a.aux_near_car)}
    aux = aux_spec(b8=coefs["b8"] > 0, near_car=coefs["near"] > 0)
    priv_dim = int(env.privileged_v2(env.reset()[1] and env.last_result).shape[1])
    balance = make_balance(a, "bc", [k for k in AUX_TERMS if coefs[k] > 0])
    if a.init:
        ck_meta = (torch.load(a.init, map_location="cpu", weights_only=True).get("meta") or {})
        if ck_meta.get("arch") != ARCH:
            raise SystemExit(f"--init {a.init} is not a v2 checkpoint; a v1 stem seeds a fresh v2 student "
                             f"through --v2-init-stem")
        have = dict(ck_meta.get("aux") or {})
        want = {k: bool(have.get(k)) or bool(v) for k, v in aux.items()}
        model, extra = load_checkpoint(a.init, device, override={"aux": want})
        print(f"resume v2 student {os.path.basename(a.init)} (iteration {extra.get('iter')}); "
              f"heads {want}, fresh: {[k for k, v in want.items() if v and not have.get(k)]}", flush=True)
        if model.obs_spec != spec:
            raise SystemExit(f"--init's observation contract {model.obs_spec} is not this env's {spec}")
        if extra.get("phase") == "dagger" and extra.get("aux_balance"):
            # the multipliers the student was being trained with; a fresh balance would re-measure
            # from 1 and let the auxiliaries pull at full weight until its first measurement
            balance.load_state_dict(extra["aux_balance"])
            print(f"auxiliary balance restored: {balance.scale} after {balance.measurements} measurements",
                  flush=True)
    else:
        model = ActorCriticV2(n_beams=spec.n_beams, n_frames=spec.n_frames, ego_dim=spec.ego_dim,
                              act_dim=spec.act_dim, speed_mode=spec.speed_mode, priv_dim=priv_dim,
                              obs=dict(vars(spec)), aux=aux).to(device)
        if a.v2_init_stem:
            rep = init_stems_from_v1(model, a.v2_init_stem, device)
            print(f"v2 stems seeded from {os.path.basename(a.v2_init_stem)}: actor {len(rep['actor'])} / "
                  f"critic {len(rep['critic'])} tensors", flush=True)
    cspec = cond_mod.CondSpec.from_meta(model.meta["cond"])
    dial = cond_mod.DialDraw(env, a.dial_margin, a.dial_exact)
    if a.wp4_label == "policy":
        from .policy_teacher import PolicyTeacher
        teacher = PolicyTeacher(env, a.label_policy, device, dial=getattr(a, "label_policy_dial", None))
        teacher_desc = teacher.describe()
        print(f"labels: {teacher_desc}", flush=True)
    labeler = Wp4Labels(env, teacher, a.wp4_label)
    mirror = Mirror(spec) if a.mirror_prob > 0 else None
    dropout = ActionDropout(spec, a.action_dropout) if float(a.action_dropout) > 0 else None
    plan_spec = env.tracker.spec
    v_max = float(env.ecfg.v_max_policy)
    opt = torch.optim.Adam(model.actor.parameters(), lr=a.lr)
    seq = {"burn_in": int(a.seq_burn), "chunk": int(a.chunk_length)}
    print(f"v2 DAgger: {spec.n_frames} frame(s), ego {spec.ego_dim} (issued actions "
          f"{'in it' if spec.ego_actions else 'NOT in it'}), action dropout p {float(a.action_dropout):g}, "
          + (f"{int(a.updates)} updates per iteration, " if a.updates else f"{a.epochs:g} epochs per iteration, ")
          + f"sequences burn-in {seq['burn_in']} + "
          f"chunk {seq['chunk']}, mirror p {a.mirror_prob:g}, labels '{a.wp4_label}', aux {coefs} "
          + (f"balanced every {balance.every} updates to <= {balance.rho:g} x the imitation's trunk gradient "
             f"(EMA {balance.beta:g}) " if balance.enabled else "(no auxiliary balance) ")
          + f"| v2 defaults applied: {applied or 'none'}", flush=True)
    race = {"opponent_speed_mode": a.opponent_speed_mode}
    print(f"v2 DAgger race: speed cap fixed {a.speed_cap:g} m/s, not observed | opponents plan "
          f"{a.opponent_speed_mode or 'wp4 (as the learner)'}", flush=True)
    run_ = common.wandb_init(a.name, vars(a) | {"phase": "dagger", "arch": ARCH, "tracks": names,
                                                "teacher_desc": teacher_desc, "v2_defaults": applied},
                             group="dagger", mode=a.wandb)
    out = common.run_dir(a.name)
    progress_log = common.ProgressLog(out)
    log = lambda d: run_.log(d)
    env.sim.warmup()
    want = {"b8": coefs["b8"] > 0, "near": coefs["near"] > 0}
    buffer_iters, teacher_metrics, m, m_iter = [], None, None, -1
    t0 = time.time()
    for it in range(a.start_iter, a.start_iter + a.iters):
        beta = 1.0 if it == 0 else a.beta0 * (0.5 ** (it - 1))
        tm = common.Timer()
        buf = collect(env, model, labeler, a.steps, beta, device, SequenceBuffer(spec.n_frames, spec.frame_stride),
                      noise=0.05 if it else 0.0, dial=dial, cspec=cspec, **want).finalize()
        buffer_iters = (buffer_iters + [buf])[-a.keep_iters:]
        t_col = tm.lap()
        log({f"collect/{k}": v for k, v in buf.stats.items()})
        # the heads that collected it, on data they have not trained on (`heldout_aux`)
        held = heldout_aux(model, buf, burn=seq["burn_in"], chunk=seq["chunk"], device=device, coefs=coefs,
                           seed=int(a.seed) * 1000 + it)
        if held:
            log({f"heldout/{k}": v for k, v in held.items()})
        model.train()
        loss, parts = train_epochs(model, buffer_iters, epochs=a.epochs, batch=a.batch, burn=seq["burn_in"],
                                   chunk=seq["chunk"], device=device, opt=opt, log=log, plan_spec=plan_spec,
                                   v_max=v_max, coefs=coefs, mirror=mirror, mirror_prob=a.mirror_prob,
                                   log_every=a.log_every, balance=balance, dropout=dropout, updates=a.updates)
        model.eval()
        t_tr = tm.lap()
        do_eval = (it % max(1, a.eval_every) == 0) or (it == a.start_iter + a.iters - 1)
        if do_eval:
            with rng_island(env):
                # the dial set exactly right (None: the true friction), the reference a supervisor is scored on,
                # at the fixed --speed-cap every DAgger table was measured at
                m = common.rollout_metrics(env, common.student_policy(model, env, device, deterministic=True),
                                           a.eval_steps, a.speed_cap, per_track=True)
            m_iter = it
        t_ev = tm.lap()
        if teacher_metrics is None:
            n_t = a.eval_steps if a.teacher_eval_steps is None else a.teacher_eval_steps
            if n_t:
                with rng_island(env):
                    # the cars the action drives only (`label_rows`); `step` overwrites every other
                    teacher_metrics = common.rollout_metrics(env, lambda o: labeler(rows=label_rows(env))[0],
                                                             n_t, a.speed_cap, per_track=True)
            else:
                nan = float("nan")
                teacher_metrics = {"collisions_per_km": nan, "collisions_per_km_worst": nan,
                                   "progress_rate_mps": nan, "lap_time_s": nan, "skipped": True}
        t_teach = tm.lap()
        scalar = lambda d: {k: v for k, v in d.items() if not isinstance(v, list)}
        n_samples = sum(len(b) for b in buffer_iters)
        log({"dagger/iter": it, "dagger/beta": beta, "dagger/samples": n_samples, "dagger/final_loss": loss,
             **({f"student/{k}": v for k, v in scalar(m).items()} if do_eval else {}),
             **{f"teacher/{k}": v for k, v in scalar(teacher_metrics).items()},
             "time/collect_s": t_col, "time/train_s": t_tr, "time/eval_s": t_ev, "time/teacher_eval_s": t_teach,
             "time/elapsed_min": (time.time() - t0) / 60})
        record = {"kind": "dagger", "arch": ARCH, "iter": it, "total": a.start_iter + a.iters, "beta": beta,
                  "samples": n_samples, "loss": loss,
                  "teacher_coll_per_km": teacher_metrics["collisions_per_km"],
                  "teacher_prog_mps": teacher_metrics["progress_rate_mps"],
                  "teacher_lap_s": teacher_metrics["lap_time_s"], "wall_s": time.time() - t0,
                  "collect_s": t_col, "train_s": t_tr, "eval_s": t_ev, "teacher_eval_s": t_teach,
                  "metrics_from_iter": m_iter, **{f"train/{k}": v for k, v in parts.items()},
                  **{f"collect/{k}": v for k, v in buf.stats.items()},
                  **{f"heldout/{k}": v for k, v in held.items()},
                  **({"heldout_scores_iter": it - 1} if held else {})}
        if do_eval:
            record.update({"student_coll_per_km": m["collisions_per_km"], "student_prog_mps": m["progress_rate_mps"],
                           "student_lap_s": m["lap_time_s"]})
            record.update({f"student/{k}": v for k, v in scalar(m).items()})
        progress_log.write(record)
        stud = (f"student {m['collisions_per_km']:.1f} coll/km prog {m['progress_rate_mps']:.2f} m/s "
                f"lap {m['lap_time_s']:.1f} s" if do_eval else f"student not measured (last at {m_iter})")
        print("  losses " + " ".join(f"{k} {parts[k]:.4f}" for k in ("bc_lat", "bc_speed", "b8", "near")
                                     if k in parts)
              + f" | B8 knowable {parts.get('b8_knowable', float('nan')):.2f} | trunk grad norms "
              + " ".join(f"{k[5:]} {v:.3g}" for k, v in parts.items() if k.startswith("grad_"))
              + (" | aux scale " + " ".join(f"{k[10:]} {v:.3g}" for k, v in parts.items() if k.startswith("aux_scale_"))
                 + " pull/bc " + " ".join(f"{k[9:]} {v:.2f}" for k, v in parts.items() if k.startswith("aux_pull_"))
                 if balance.enabled else "")
              + (f" | measuring {100 * parts['measure_overhead_frac']:.1f} % of training time"
                 if "measure_overhead_frac" in parts else ""), flush=True)
        if held:
            fmt = lambda k: f"{held[k]:.3f}" if k in held else "-"
            whose = "the initial weights" if it == a.start_iter and not a.init else f"the weights before iteration {it}"
            print(f"  held-out heads ({whose}, on this collection): nearest car |dy| p50 "
                  f"<3 m {fmt('near_dy_p50_3m')} m, dist {fmt('near_dist_p50_3m')} m, presence AUC "
                  f"{fmt('near_presence_auc')} | B8 |dy| p50 <3 m {fmt('b8_dy_p50_3m')} m", flush=True)
        print(f"  collection: speed-capped {100 * buf.stats.get('speed_capped_frac', float('nan')):.1f} % of "
              f"learner steps (the fixed {a.speed_cap:g} m/s cap clipped the executed plan)", flush=True)
        print(f"iter {it}: beta {beta:.2f} samples {n_samples} loss {loss:.4f} | {stud} | teacher "
              f"{teacher_metrics['collisions_per_km']:.1f} coll/km lap {teacher_metrics['lap_time_s']:.1f} s | "
              f"{t_col:.0f}+{t_tr:.0f}+{t_ev:.0f}+{t_teach:.0f} s", flush=True)
        meta = {"spec": dict(vars(spec)), "phase": "dagger", "arch": ARCH, "run": a.name, "iter": it,
                "iters": a.start_iter + a.iters, "start_iter": a.start_iter,
                "resumed_from": a.init if a.start_iter else None, "cap": a.speed_cap, "samples": n_samples,
                "metrics": m, "metrics_from_iter": m_iter, "teacher": teacher_metrics,
                "action_mode": "plan", "teacher_kind": a.teacher_kind, "teacher_desc": teacher_desc,
                "speed_mode": "wp4", "speed_command": a.speed_command,
                "tracker_speed_weight": a.tracker_speed_weight, "tracker_a_max": a.tracker_a_max,
                **({"tracker_profile_from_prediction": True} if a.tracker_profile_from_prediction else {}),
                **({"tracker_speed_ff": True} if a.tracker_speed_ff else {}),
                "wp4_label": a.wp4_label, "bc": dict(BC_CONSTANTS), "sequence": seq,
                "mirror_prob": float(a.mirror_prob), "action_dropout": float(a.action_dropout),
                "updates_per_iter": (int(a.updates) if a.updates else None), "epochs_per_iter": float(a.epochs),
                "aux_coefs": coefs, "aux_balance": balance.state_dict(),
                "dial": {"margin": float(a.dial_margin), "exact": float(a.dial_exact)},
                "v2_defaults": applied, "teacher_grip": a.teacher_grip, "teacher_speed": a.teacher_speed,
                "race": race, "collect": dict(buf.stats), "heldout": dict(held),
                # the LAST weights, always: never a checkpoint picked by a loss (`heldout_aux`)
                "checkpoint_rule": "last"}
        save_checkpoint(os.path.join(out, f"student_it{it}.pt"), model, meta)
        save_checkpoint(os.path.join(out, "student_latest.pt"), model, meta)
    progress_log.close()
    run_.finish()
