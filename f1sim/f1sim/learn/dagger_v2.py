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
  nearest-visible-car and per-beam heads behind their own coefficients.
* **Left-right mirroring**, p = `--mirror-prob` (0.5) per sequence (`train_v2.Mirror`; G1's
  appendix-2 correction: the flip took the visible lateral error p50 from 0.138 to 0.097 m).
* **Per-loss gradient norms** on the shared trunk (BC vs B8 vs the others), every `--log-every`
  updates.

Teacher labels. A v2 run speaks WP4. `--wp4-label env` asks the env's teacher-label API
(`F1VecEnv.teacher_label`), which is where the wp4-speaking teachers of feat/v2 plug in; on a tree
without them that call raises, and the run says so. `--wp4-label linear-refit` is this branch's
stand-in for tests and smoke runs: the raceline teacher's own LINEAR plan, its first window refitted
to the four waypoints (`mpc.wp4_refit`) and its speed line read at them (`train_v2.linear_to_wp4`) --
the same refit recipe, so the decode, the refit and the mirror tables are all exercised, but not a
label to train the car's policy on.

What DAgger does not train: the critic (PPO's critic warm-up does) and `log_std` (imitation never
touches it; the per-dimension initial values of B6 reach PPO as they were built).
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
from .aux_v2 import aux_spec, beam_car_loss, nearest_car_loss
from .model import load_checkpoint, save_checkpoint
from .model_v2 import ARCH, ActorCriticV2, init_stems_from_v1
from .obs import ObsSpecV2
from .train_v2 import (BC_CONSTANTS, Mirror, actor_trunk_parameters, b8_sequence_loss, bc_loss_wp4,
                       linear_spec_of, linear_to_wp4, loss_grad_norms, select_tracks,
                       steps_since_window_start)

#: Where a v2 run's teacher labels come from (see the module docstring).
WP4_LABEL_SOURCES = ("env", "linear-refit")

#: The v2 defaults of flags v1 also has, applied when the command line does not set them: the
#: redesign's burn-in 40 + chunk 80 (G1), and the plan / dial contract every v2 actor is built for.
V2_DEFAULTS = {"chunk_length": 80, "seq_burn": 40}
#: The defaults of the v2-only flags (None on the parser, so a v1 run can refuse them).
V2_ONLY_DEFAULTS = {"v2_frames": 1, "v2_init_stem": "", "wp4_label": "env", "mirror_prob": 0.5,
                    "aux_b8": 0.5, "aux_near_car": 0.0, "aux_beam_car": 0.0}

#: v1 options a v2 run cannot honour, with the value that means "not asked for". Refused rather than
#: ignored: a run that names an option and then runs without it looks like evidence it did nothing.
V2_REFUSED = {"memory": "off", "scan_channels": "", "maneuver": False, "opp_token": "off", "hard_frac": 0.0,
              "solo_fraction": 0.0, "speed_loss": "symmetric", "grip_quantile": 0.5, "aux_grip": 0.0,
              "aux_opp": 0.0, "aux_opp_visible": False, "seq_len": 0, "scan_deltas": False,
              "temporal_encoder": "cnn", "hist_len": 0}


def explicit_flags(ap: argparse.ArgumentParser, argv=None) -> set:
    """The dests the command line set explicitly. argparse cannot tell "given, equal to the default"
    from "not given", and a v2 default must only replace the latter: parsed once more into a
    namespace pre-filled with a sentinel, argparse leaves every unset dest at the sentinel."""
    import sys
    sentinel = object()
    ns = argparse.Namespace(**{x.dest: sentinel for x in ap._actions if x.dest != argparse.SUPPRESS})
    ap.parse_args(sys.argv[1:] if argv is None else argv, namespace=ns)
    return {k for k, v in vars(ns).items() if v is not sentinel}


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
                         f"(--aux-b8 / --aux-near-car / --aux-beam-car) and contract (learn.obs.ObsSpecV2)")
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
    if a.wp4_label == "linear-refit" and a.teacher_kind != "raceline":
        raise SystemExit("--wp4-label linear-refit refits the RACELINE teacher's linear plan; the "
                         f"{a.teacher_kind} teacher's search would be silently dropped")
    if min(a.aux_b8, a.aux_near_car, a.aux_beam_car) < 0:
        raise SystemExit("auxiliary weights are >= 0")
    applied = {}
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


def env_kwargs(a) -> dict:
    """The EnvConfig fields a v2 run adds to the v1 ones `dagger.main` passes: contract v2, and the
    per-seat visibility the B8 / nearest-car labels are counted from."""
    return {"obs_version": 2, "hist_stride": ObsSpecV2().hist_stride,
            "opp_visibility": bool(a.aux_b8 > 0 or a.aux_near_car > 0)}


# ------------------------------------------------------------------------------- teacher labels
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
        if source == "linear-refit":
            from ..teacher import RacelineTeacher
            if not isinstance(teacher, RacelineTeacher):
                # not its `.base`: an interactive or rollout teacher's label is its search, and
                # refitting the raceline under it would drop the search without a word
                raise ValueError("--wp4-label linear-refit refits a raceline teacher's linear plan; "
                                 f"this teacher is {type(teacher).__name__}")
            self.base = teacher

    @torch.no_grad()
    def __call__(self, mu: Optional[torch.Tensor] = None):
        env = self.env
        if self.source == "env":
            try:
                lab = env.teacher_label(self.teacher, mu=mu)
            except NotImplementedError as exc:
                raise SystemExit(f"the env's teacher does not speak wp4 on this tree ({exc}). The wp4 teachers "
                                 f"live on feat/v2; until it is merged run with --wp4-label linear-refit "
                                 f"(a stand-in for tests and smoke runs, see learn.dagger_v2).") from exc
            valid = getattr(self.teacher, "last_label_valid", None)
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
            valid = None
        ok = torch.isfinite(lab).all(1)
        valid = ok if valid is None else (valid.to(ok.device).bool() & ok)
        return torch.nan_to_num(lab), valid


# --------------------------------------------------------------------------------- the buffer
class SequenceBuffer:
    """Per-step storage of one collection, (steps, learners). A training sample is a contiguous run of
    one learner's steps; its frame stack is rebuilt from the stored newest frames at sampling time (the
    env's own history, which repeats an episode's first frame backwards, is what `frames_at` rebuilds).

    What one step stores: the newest frame (fp16), the ego vector (fp16 -- 210 numbers, the 1-s
    history in it), the teacher's label and its validity, whether the step began an episode, the
    dial column `c` the student was given, and -- when their heads train -- the B8 tracks (`select_tracks`:
    labels, candidates, AGE: knowability is decided per sampled window), the nearest-visible-car
    target and the per-beam car labels."""

    KEYS = ("scan", "ego", "label", "valid", "newep", "c", "b8_lab", "b8_cand", "b8_age", "near", "beam")

    def __init__(self, n_frames: int, frame_stride: int = 1):
        self.k, self.stride = int(n_frames), int(frame_stride)
        self.lists = {k: [] for k in self.KEYS}

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

    def sample(self, m: int, length: int, device) -> dict:
        """`m` runs of `length` contiguous steps, each from one learner: a dict of (length, m, ...)
        tensors on `device` (float32 where they were stored in half), plus `keep` (length, m) -- 0 at
        a step that began an episode, where the state must not be carried in."""
        length = max(1, min(int(length), self.T))
        t0 = torch.randint(self.T - length + 1, (m,))
        b = torch.randint(self.B, (m,))
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
            dial: cond_mod.DialDraw, cspec, b8: bool, near: bool, beam: bool) -> SequenceBuffer:
    """Roll the env for `steps`, label every state with the teacher, store the learners' rows.

    Iteration 0 (beta 1) drives the teacher; later ones mix the student in with probability
    1 - beta, as v1 does. The student runs from its own recurrent state at every step -- also on the
    steps the teacher's action is taken -- and the state is zeroed where an episode ends, so the
    state it acts from is the one it would have on the car. The dial is drawn per episode
    (`conditioning.DialDraw`: mu - U(0, 0.3), exact with p 0.3, the redesign's B4) and the teacher
    drives for that same number."""
    obs, _info = env.reset()
    # the primary learner of each race, as v1's DAgger stores: the other seats' trajectories are
    # driven by their own population and are not what this intervention visited
    ids = torch.arange(0, env.B, env.M, device=env.device)
    new_ep = torch.ones(env.B, dtype=torch.bool, device=env.device)
    h = model.actor.initial_hidden(env.B, device=device)
    with torch.no_grad():
        for _t in range(steps):
            scan, ego = obs["scan"], obs["ego"]
            dial.redraw(new_ep)
            mu_dial = dial.value()
            c = cond_mod.mu_to_c(mu_dial, cspec).to(scan.dtype)
            label, valid = labeler(mu=mu_dial)
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
            if beam:
                row["beam"] = env.beam_car_labels_v2()[ids]
            buf.add(**row)
            if student is None:
                a = label
            else:
                a_s = student.to(label.device)
                if noise > 0:
                    a_s = (a_s + noise * torch.randn_like(a_s)).clamp(-1, 1)
                a = torch.where((torch.rand(env.B, device=label.device) < beta)[:, None], label, a_s)
            obs, _rew, term, trunc, _info = env.step(a)
            new_ep = (term | trunc).to(env.device)
            h = h * (~new_ep).to(h.dtype).to(h.device)[None, :, None]
    return buf


# ---------------------------------------------------------------------------------- training
def sequence_losses(model, s: dict, *, burn: int, plan_spec, v_max: float, coefs: dict, beams: bool):
    """The v2 imitation losses on one batch of sequences `s` (`SequenceBuffer.sample`, mirrored or
    not): {"total", "bc", "bc_lat", "bc_speed", "b8", "b8_knowable", "near", "beam"} -- every part a
    tensor (zero where its coefficient is off), the parts that train carrying their graph."""
    actor = model.actor
    out = actor.sequence(s["scan"], s["ego"], s["c"], keep=s["keep"], burn_in=burn, beams=beams)
    sl = slice(burn, None)
    T, m = out["mu"].shape[:2]
    flat = lambda x: x[sl].reshape(T * m, *x.shape[2:])
    v_meas = flat(s["ego"])[:, 0] * v_max                    # speed@t: the speed the plan is laid out from
    valid = flat(s["valid"]).float()
    bc, parts = bc_loss_wp4(out["mu"].reshape(T * m, -1), flat(s["label"]), v_meas, plan_spec, v_max, weight=valid)
    zero = torch.zeros((), device=bc.device)
    res = {"bc": bc, "bc_lat": parts["lat"].detach(), "bc_speed": parts["speed"].detach(),
           "b8": zero, "b8_knowable": zero, "near": zero, "beam": zero}
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
    if coefs.get("beam", 0.0) > 0 and "beam_car" in out:
        beam = beam_car_loss(out["beam_car"].float().reshape(T * m, -1), flat(s["beam"]))
        res["beam"] = beam
        total = total + coefs["beam"] * beam
    res["total"] = total
    return res


def train_epochs(model, bufs, *, epochs: float, batch: int, burn: int, chunk: int, device, opt, log,
                 plan_spec, v_max: float, coefs: dict, mirror: Optional[Mirror], mirror_prob: float,
                 log_every: int = 25, max_grad: float = 1.0):
    """Truncated BPTT over sequences of `burn + chunk` steps; `batch` scored steps per update (so
    `batch // chunk` sequences). Every buffer is drawn in proportion to its valid labels, as v1.
    Returns (the last updates' mean loss, {each part's and each gradient norm's mean over the
    iteration}) -- the summary goes into the iteration's progress record, so a run that logs nowhere
    else still says what B8 weighed against the imitation."""
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
    steps = max(1, int(epochs * n_valid / (m * chunk)))
    weights = torch.tensor([float(b.valid_count) for b in bufs])
    trunk = actor_trunk_parameters(model.actor)
    params = [p for p in model.actor.parameters() if p.requires_grad]
    beams = coefs.get("beam", 0.0) > 0
    hist = {k: [] for k in ("total", "bc", "bc_lat", "bc_speed", "b8", "b8_knowable", "near", "beam")}
    grads = {}
    for i in range(steps):
        b = bufs[int(torch.multinomial(weights, 1))]
        s = b.sample(m, burn + chunk, device)
        if mirror is not None and mirror_prob > 0:
            s = mirror.apply(s, torch.rand(m, device=s["scan"].device) < mirror_prob)
        terms = sequence_losses(model, s, burn=burn, plan_spec=plan_spec, v_max=v_max, coefs=coefs, beams=beams)
        loss = terms["total"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite DAgger v2 loss at step {i}: " +
                               ", ".join(f"{k} {float(v):.4g}" for k, v in terms.items()))
        gn = None
        if i % log_every == 0:
            # each loss's pull on the trunk the losses share (before the optimiser step), so a
            # B8 that cannot move the memory -- or swamps the imitation -- shows as a number
            gn = loss_grad_norms({k: (coefs.get(k, 1.0) if k != "bc" else 1.0) * terms[k]
                                  for k in ("bc", "b8", "near", "beam") if terms[k].requires_grad}, trunk)
        opt.zero_grad()
        loss.backward()
        total_gn = torch.nn.utils.clip_grad_norm_(params, max_grad)
        opt.step()
        for k in hist:
            hist[k].append(float(terms[k].detach()))
        if gn is not None:
            gn["total_clip_input"] = float(total_gn)
            for k, v in gn.items():
                grads.setdefault(k, []).append(v)
            row = {f"dagger/{k}": float(np.mean(v[-log_every:])) for k, v in hist.items()}
            row.update({f"dagger/grad_{k}": v for k, v in gn.items()})
            row.update({"dagger/train_step": i, "dagger/lr": opt.param_groups[0]["lr"]})
            log(row)
    log({"dagger/optimizer_updates": steps})
    summary = {k: float(np.mean(v)) for k, v in hist.items() if v}
    summary.update({f"grad_{k}": float(np.mean(v)) for k, v in grads.items()})
    summary["updates"] = steps
    return float(np.mean(hist["total"][-500:])), summary


# ------------------------------------------------------------------------------------- main
def run(a, *, env, teacher, teacher_desc: str, names, device, applied: dict):
    """The DAgger loop for `--arch v2`, from the env and teacher `dagger.main` built."""
    from .dagger import rng_island
    spec = common.obs_spec(env)
    if not isinstance(spec, ObsSpecV2):
        raise SystemExit("--arch v2 needs the env built with obs_version 2")
    coefs = {"b8": float(a.aux_b8), "near": float(a.aux_near_car), "beam": float(a.aux_beam_car)}
    aux = aux_spec(b8=coefs["b8"] > 0, beam_car=coefs["beam"] > 0, near_car=coefs["near"] > 0)
    priv_dim = int(env.privileged_v2(env.reset()[1] and env.last_result).shape[1])
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
    labeler = Wp4Labels(env, teacher, a.wp4_label)
    mirror = Mirror(spec) if a.mirror_prob > 0 else None
    plan_spec = env.tracker.spec
    v_max = float(env.ecfg.v_max_policy)
    opt = torch.optim.Adam(model.actor.parameters(), lr=a.lr)
    seq = {"burn_in": int(a.seq_burn), "chunk": int(a.chunk_length)}
    print(f"v2 DAgger: {spec.n_frames} frame(s), ego {spec.ego_dim}, sequences burn-in {seq['burn_in']} + "
          f"chunk {seq['chunk']}, mirror p {a.mirror_prob:g}, labels '{a.wp4_label}', aux {coefs} "
          f"| v2 defaults applied: {applied or 'none'}", flush=True)
    run_ = common.wandb_init(a.name, vars(a) | {"phase": "dagger", "arch": ARCH, "tracks": names,
                                                "teacher_desc": teacher_desc, "v2_defaults": applied},
                             group="dagger", mode=a.wandb)
    out = common.run_dir(a.name)
    progress_log = common.ProgressLog(out)
    log = lambda d: run_.log(d)
    env.sim.warmup()
    want = {"b8": coefs["b8"] > 0, "near": coefs["near"] > 0, "beam": coefs["beam"] > 0}
    buffer_iters, teacher_metrics, m, m_iter = [], None, None, -1
    t0 = time.time()
    for it in range(a.start_iter, a.start_iter + a.iters):
        beta = 1.0 if it == 0 else a.beta0 * (0.5 ** (it - 1))
        tm = common.Timer()
        buf = collect(env, model, labeler, a.steps, beta, device, SequenceBuffer(spec.n_frames, spec.frame_stride),
                      noise=0.05 if it else 0.0, dial=dial, cspec=cspec, **want).finalize()
        buffer_iters = (buffer_iters + [buf])[-a.keep_iters:]
        t_col = tm.lap()
        model.train()
        loss, parts = train_epochs(model, buffer_iters, epochs=a.epochs, batch=a.batch, burn=seq["burn_in"],
                                   chunk=seq["chunk"], device=device, opt=opt, log=log, plan_spec=plan_spec,
                                   v_max=v_max, coefs=coefs, mirror=mirror, mirror_prob=a.mirror_prob,
                                   log_every=a.log_every)
        model.eval()
        t_tr = tm.lap()
        do_eval = (it % max(1, a.eval_every) == 0) or (it == a.start_iter + a.iters - 1)
        if do_eval:
            with rng_island(env):
                # the dial set exactly right (None: the true friction), the reference a supervisor is scored on
                m = common.rollout_metrics(env, common.student_policy(model, env, device, deterministic=True),
                                           a.eval_steps, a.speed_cap, per_track=True)
            m_iter = it
        t_ev = tm.lap()
        if teacher_metrics is None:
            n_t = a.eval_steps if a.teacher_eval_steps is None else a.teacher_eval_steps
            if n_t:
                with rng_island(env):
                    teacher_metrics = common.rollout_metrics(env, lambda o: labeler()[0], n_t, a.speed_cap,
                                                             per_track=True)
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
                  "metrics_from_iter": m_iter, **{f"train/{k}": v for k, v in parts.items()}}
        if do_eval:
            record.update({"student_coll_per_km": m["collisions_per_km"], "student_prog_mps": m["progress_rate_mps"],
                           "student_lap_s": m["lap_time_s"]})
            record.update({f"student/{k}": v for k, v in scalar(m).items()})
        progress_log.write(record)
        stud = (f"student {m['collisions_per_km']:.1f} coll/km prog {m['progress_rate_mps']:.2f} m/s "
                f"lap {m['lap_time_s']:.1f} s" if do_eval else f"student not measured (last at {m_iter})")
        print("  losses " + " ".join(f"{k} {parts[k]:.4f}" for k in ("bc_lat", "bc_speed", "b8", "near", "beam")
                                     if k in parts)
              + f" | B8 knowable {parts.get('b8_knowable', float('nan')):.2f} | trunk grad norms "
              + " ".join(f"{k[5:]} {v:.3g}" for k, v in parts.items() if k.startswith("grad_")), flush=True)
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
                "mirror_prob": float(a.mirror_prob), "aux_coefs": coefs,
                "dial": {"margin": float(a.dial_margin), "exact": float(a.dial_exact)},
                "v2_defaults": applied, "teacher_grip": a.teacher_grip, "teacher_speed": a.teacher_speed}
        save_checkpoint(os.path.join(out, f"student_it{it}.pt"), model, meta)
        save_checkpoint(os.path.join(out, "student_latest.pt"), model, meta)
    progress_log.close()
    run_.finish()
