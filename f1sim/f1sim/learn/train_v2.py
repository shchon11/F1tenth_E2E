"""What the v2 trainers share: DAgger v2 (`learn.dagger_v2`) and PPO v2 (`learn.ppo_v2`).

docs/research/architecture-redesign-2026-10-13.md sections 5-7 with its appendix 2 ("관문 결과와 설계
정정") and docs/research/output-format-2026-10-14.md sections 5 and 7. Each piece is a function of
stated numbers so the tests can hold it on its own (`tests/test_train_v2.py`):

* **`Mirror`** -- the left-right mirror of a training sequence, from the sign tables the observation
  and the env already state (`obs.ego_mirror_signs`, `obs.action_mirror_signs`, `obs.mirror_scan`,
  `gym_env.PRIV_V2_MIRROR`, `aux_v2.B8_LABEL_MIRROR`, `aux_v2.NEAR_MIRROR`). G1's appendix-2 result:
  the visible-opponent lateral error p50 0.138 m missed the 0.10 m bar, and the left-right flip with a
  longer run reached 0.097 -- so the flip is part of the recipe ("좌우 반전 증강 추가").
* **`bc_loss_wp4`** -- imitation in the DECODED space (output-format 5): Huber on the four
  waypoints' lateral position [m] and on their speed [m/s], through `mpc.decode_wp4` and
  `mpc.wp4_path`. In the normalised space a unit of kappa_1 moves the window's end by 4.76 m and a
  unit of kappa_4 by 0.40 m (output-format 4, the per-dimension authority table); an equal weight on
  the two would spend the loss on dimensions that barely move the car. Decoded, an error costs what
  it does to the plan the tracker is handed, and two actions the decode maps to one profile (the
  backward braking pass) cost the same.
* **the B8 window** (`steps_since_window_start`, `b8_sequence_loss`, `select_tracks`) -- which cars
  a recurrent state can know: seen by >= 3 beams within the last 3 s AND since its state began.
  A burn-in window's state begins at the burn-in's first step (DAgger); a stored rollout state
  (PPO) began at its episode's first observation (see `b8_sequence_loss`).
* **`ValueNorm`** -- running mean / std of the returns: the critic learns normalised targets and
  the GAE reads de-normalised values. G4: the value-normalised, unclipped critic lifted 0.5 s
  foresight R from 0.085 to 0.17 (raw returns spread with sd 125 against V's 7.5).
* **`contact_labels`** -- the "car contact within 1 s" labels of the critic's classifier from a
  rollout buffer (G4: R 0.21, AUC 0.82 with it).
* **`loss_grad_norms`** -- each loss's gradient norm on the parameters the losses share (the G5
  item "손실별 기울기 노름"): B8 is one small step (/ 3 slots, x 0.5) and has to be read against the
  imitation loss's to know whether it can move the memory at all (`aux_v2`'s CPU toy).
* **`AuxBalance`** -- what those norms showed, acted on: on UNIT 4's smoke runs the auxiliaries at
  their configured coefficients pulled the shared trunk 2-25x harder than the loss the policy is
  trained for, so each auxiliary gets a multiplier that holds its pull at most `rho` x the main loss's.
* **PopArt** (`ValueNorm.update(..., preserve=head)`) -- a statistics update rescales the value head's
  last layer so the de-normalised value of every state is what it was before the update.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gym_env import OPP_MEMORY_STEPS, OPP_NEVER_SEEN, PRIV_V2_MIRROR
from ..mpc import WP_K, PlanSpec, decode, decode_wp4, encode_wp4, plan_length, wp4_path, wp4_refit
from .aux_v2 import B8_LABEL_MIRROR, B8_SLOTS, NEAR_MIRROR, b8_know_mask, b8_loss
from .obs import ObsSpecV2, action_mirror_signs, ego_mirror_signs


# ---------------------------------------------------------------------------------- mirroring
class Mirror:
    """The left-right mirror of a block of training data, one sequence at a time.

    Everything a sequence carries is either mirror-invariant (speed, the dial, distances, presence,
    validity, the recurrence mask) or flips by a sign table / the beam order. Applied to whole
    sequences, never to single steps: a GRU walked through a sequence that flips half way is driving
    a world that does not exist.

    Exact for the scan because the beams are `linspace(-fov/2, fov/2, N)` about the car's x axis and
    the LiDAR sits on it (mount_y 0, mount_yaw 0 nominal); a run that randomises the mount laterally
    gets a mirror that is off by that offset, which is a few millimetres and a few tenths of a degree.
    """

    #: the (T, m, ...) keys `apply` knows, and what happens to each: a sign table or the beam order
    def __init__(self, spec: ObsSpecV2):
        self.signs = {"ego": ego_mirror_signs(spec),
                      "label": action_mirror_signs(spec.speed_mode),
                      "priv": torch.tensor(PRIV_V2_MIRROR, dtype=torch.float32),
                      "b8_lab": torch.tensor(B8_LABEL_MIRROR, dtype=torch.float32),
                      "near": torch.tensor(NEAR_MIRROR, dtype=torch.float32)}
        #: beam-indexed keys: reversed along the last axis (`obs.mirror_scan`'s rule)
        self.beams = ("scan", "beam")

    def mirrored(self, key: str, x: torch.Tensor) -> torch.Tensor:
        """`x` of the mirrored world, for every row (no selection)."""
        if key in self.beams:
            return x.flip(-1)
        s = self.signs[key].to(device=x.device, dtype=x.dtype)
        return x * s

    def apply(self, batch: Dict[str, torch.Tensor], flip: torch.Tensor) -> Dict[str, torch.Tensor]:
        """`batch` holds (T, m, ...) tensors; `flip` (m,) bool says which sequences to mirror. Keys
        `apply` does not know are mirror-invariant and come back as they are."""
        out = {}
        for k, v in batch.items():
            if v is None or (k not in self.signs and k not in self.beams):
                out[k] = v
                continue
            sel = flip.to(v.device).view(1, -1, *([1] * (v.dim() - 2)))
            out[k] = torch.where(sel, self.mirrored(k, v), v)
        return out


def mirror_state(state: torch.Tensor, x_lo: float, x_hi: float) -> torch.Tensor:
    """A car state of a map mirrored about its vertical axis (`Track.mirrored`: x -> x_lo + x_hi - x).

    Pose (x, y, yaw) -> (x_lo + x_hi - x, y, pi - yaw); in the body frame the forward speed stays,
    the lateral speed and the yaw rate flip. The steering angle (column 6, when the state has it)
    flips with the yaw rate. For the "mirrored label = label of the mirrored state" check."""
    import math
    s = state.clone()
    s[:, 0] = x_lo + x_hi - state[:, 0]
    s[:, 2] = math.pi - state[:, 2]
    s[:, 4] = -state[:, 4]
    s[:, 5] = -state[:, 5]
    if state.shape[1] > 6:
        s[:, 6] = -state[:, 6]
    return s


# ---------------------------------------------------------------------- imitation, decoded
#: Huber transition points of the decoded imitation loss: [m] for the waypoints' lateral position,
#: [m/s] for their speed. v1's Huber had beta 0.1 in normalised units; here 0.1 m and 0.1 m/s, one to
#: one -- a choice with no measurement behind it (the redesign's "기본값(요구 없음)"), recorded in
#: every checkpoint (`BC_CONSTANTS`) so a later change is a stated one.
BC_BETA_LAT = 0.1
BC_BETA_V = 0.1
BC_CONSTANTS = {"beta_lat_m": BC_BETA_LAT, "beta_v_mps": BC_BETA_V, "w_lat": 1.0, "w_speed": 1.0,
                "points": "the 4 waypoints: lateral position (wp4_path, body frame) and decoded speed"}


def wp4_decoded(action: torch.Tensor, v_meas: torch.Tensor, spec: PlanSpec, v_max: float):
    """(lateral position of the WP_K waypoints (B, WP_K) [m], their speed (B, WP_K) [m/s]): the plan
    `action` hands the tracker from measured speed `v_meas` (B,), through the tracker's own decode.

    Decoded WITHOUT the run's speed cap (the cap passed is v_max, the top of the box): the policy does
    not observe the cap (critique B5), and an imitation loss that went flat above it would leave the
    speed dimensions free to drift there -- the very failure B5 names for PPO."""
    k, L, v = decode_wp4(action, v_meas, v_max, torch.full_like(v_meas, float(v_max)), spec)
    _x, y, _psi, _s, _k = wp4_path(k, L, spec, extend=1.0)
    idx = torch.arange(1, WP_K + 1, device=action.device) * (int(spec.wp_grid) // WP_K)
    return y[:, idx], v


def bc_loss_wp4(mu: torch.Tensor, label: torch.Tensor, v_meas: torch.Tensor, spec: PlanSpec, v_max: float,
                weight: Optional[torch.Tensor] = None):
    """Imitation in the decoded space (output-format 5): (scalar, {"lat", "speed"}), each part a
    weighted mean over rows of the per-waypoint Huber's mean.

    `mu`, `label` (R, 8) normalised wp4 actions; `v_meas` (R,) the speed the plan is laid out from
    (the env's odometry speed -- the ego vector's speed@t times v_max); `weight` (R,) zero where the
    teacher's label is not to be learned from (invalid, or masked by the caller)."""
    ys, vs = wp4_decoded(mu, v_meas, spec, v_max)
    with torch.no_grad():
        yt, vt = wp4_decoded(label, v_meas, spec, v_max)
    lat = F.smooth_l1_loss(ys, yt, beta=BC_BETA_LAT, reduction="none").mean(-1)
    spd = F.smooth_l1_loss(vs, vt, beta=BC_BETA_V, reduction="none").mean(-1)
    w = torch.ones_like(lat) if weight is None else weight.to(lat.dtype)
    n = w.sum().clamp_min(1.0)
    lat_m, spd_m = (lat * w).sum() / n, (spd * w).sum() / n
    return BC_CONSTANTS["w_lat"] * lat_m + BC_CONSTANTS["w_speed"] * spd_m, {"lat": lat_m, "speed": spd_m}


def linear_to_wp4(a_lin: torch.Tensor, v_meas: torch.Tensor, lin_spec: PlanSpec, spec: PlanSpec, v_max: float,
                  k_lim=None) -> torch.Tensor:
    """A linear-format plan action -> the wp4 action that drives its first window: the curvature
    refitted to the four waypoints (`mpc.wp4_refit`, 4 Gauss-Newton iterations -- converged from the
    source's own curvature, feat/v2's measurement) and the linear plan's speed line (v0 at s = 0,
    linear in arc length to v1 at Lp, `mpc.reference`'s target) read at the waypoints' arcs.

    The stand-in label of this branch: the teachers here do not speak wp4 yet (feat/v2's
    `RacelineTeacher._plan_wp4` does, with its "refit" path chosen by the pre-registered replay). The
    path recipe is the same refit; the speeds are the linear plan's line rather than the profile at
    each waypoint, which is what tests and smoke runs need of it (the decode, the refit and the
    sign tables all exercised), not a label to train a policy for the car on."""
    k6, Lp, v0, v1 = decode(a_lin, v_meas, v_max, torch.full_like(v_meas, float(v_max)), lin_spec)
    k4 = wp4_refit(k6, Lp, v_meas, spec, iters=4, k_lim=k_lim)
    L = plan_length(v_meas, spec)
    s = L[:, None] * torch.arange(1, WP_K + 1, device=a_lin.device, dtype=a_lin.dtype) / WP_K
    vk = v0[:, None] + (v1 - v0)[:, None] * (s / Lp[:, None].clamp_min(1e-6)).clamp(0.0, 1.0)
    return encode_wp4(k4, vk, v_max, spec, v_meas=v_meas)


def linear_spec_of(spec: PlanSpec) -> PlanSpec:
    """The same tracker spec read as the linear format (what the teachers here plan in)."""
    return replace(spec, speed_mode="linear")


# ------------------------------------------------------------------------------- the B8 window
def select_tracks(lab: torch.Tensor, cand: torch.Tensor, age: torch.Tensor, dist: torch.Tensor,
                  slots: int = B8_SLOTS):
    """`F1VecEnv.opponent_tracks_v2`'s per-seat (L, C, ...) -> exactly `slots` tracks, as
    `aux_v2.b8_select` reduces them (the nearest `slots` CANDIDATES by LiDAR distance; fewer seats
    padded with absent ones), carrying the AGE along instead of a finished knowability mask -- which
    cars are knowable depends on the training window, and only the trainer knows that (DAgger's
    burn-in window is drawn at sampling time). Returns (lab (L, S, 3), cand (L, S), age (L, S))."""
    C = lab.shape[-2]
    if C < slots:
        pad = slots - C
        return (F.pad(lab, (0, 0, 0, pad)), F.pad(cand.to(torch.uint8), (0, pad)).bool(),
                F.pad(age, (0, pad), value=OPP_NEVER_SEEN))
    if C == slots:
        return lab, cand, age
    key = torch.where(cand, dist, torch.full_like(dist, float("inf")))
    idx = key.argsort(-1)[..., :slots]
    return (lab.gather(-2, idx[..., None].expand(*idx.shape, lab.shape[-1])), cand.gather(-1, idx),
            age.gather(-1, idx))


def steps_since_window_start(keep: torch.Tensor) -> torch.Tensor:
    """(T, m) long: steps since each row's recurrent state was last zero -- the window's first step
    (row 0) or a step that began an episode inside it (`keep` 0 there). What `aux_v2.b8_know_mask`
    calls `steps_into_window`: a car whose last sighting is older than this was seen by a state that
    has since been wiped, and asking for it is asking the impossible (chunk 80 without this rule:
    52.6 % unanswerable memory labels, judge-learnability-minimal/chunk_mask.json)."""
    out = torch.zeros(keep.shape, dtype=torch.long, device=keep.device)
    for t in range(1, keep.shape[0]):
        out[t] = torch.where(keep[t] > 0, out[t - 1] + 1, torch.zeros_like(out[t - 1]))
    return out


def b8_sequence_loss(pred: torch.Tensor, lab: torch.Tensor, cand: torch.Tensor, age: torch.Tensor,
                     since: torch.Tensor, weight: Optional[torch.Tensor] = None,
                     max_age: int = OPP_MEMORY_STEPS):
    """B8 over a (T, m) block: (scalar, {"knowable": share of candidate cars the window can know}).

    `pred` (T, m, S, 4) from the slot head; `lab`, `cand`, `age` from `select_tracks`; `since` (T, m)
    the steps between each row and the start of its state's window:

      DAgger (burn-in windows)  `steps_since_window_start(keep)` over the burn-in + chunk: the state
                                starts at zero on the burn-in's first step.
      PPO (stored states)       the steps since the row's episode began. The update replays a chunk
                                from the state the rollout carried into it, and that state has been
                                carried since its episode's first observation, so it CAN know a car
                                seen before the chunk began (within the 3 s age limit). It was
                                written by the rollout's parameters and is a constant of the update
                                (detached): a car seen only before the chunk trains the read-out of
                                the stored state, never the writing of it -- the stored-state
                                approximation (R2D2), close because one update moves the actor by
                                <= ~2e-3 relative.
    """
    know = b8_know_mask(cand, age, since, max_age)
    per = b8_loss(pred.float(), lab.float(), cand, know)
    w = torch.ones_like(per) if weight is None else weight.to(per.dtype)
    loss = (per * w).sum() / w.sum().clamp_min(1.0)
    n_cand = cand.sum()
    return loss, {"knowable": (know.sum().float() / n_cand.clamp_min(1)).detach(),
                  "candidates_per_row": (n_cand.float() / max(1, cand.shape[0] * cand.shape[1])).detach()}


# ---------------------------------------------------------------------------- value normaliser
class ValueNorm:
    """Running mean and std of the returns, by exponential moving average with Adam's debiasing.

    The critic predicts `normalize(G)`; the GAE and the truncation bootstrap read
    `denormalize(critic)`. Updated once per PPO update with that rollout's returns, weighted by the
    policy's own rows. `beta` per UPDATE: 0.99 keeps about the last hundred updates (a run of the
    redesign's size is 50-200 updates), so the scale follows a policy whose contact cost falls while
    one rollout cannot jerk it. Before the first update it is the identity.

    A statistics update changes the meaning of every critic output: the same normalised number
    de-normalises to a different value. `update(..., preserve=head)` is PopArt's answer (van Hasselt
    et al. 2016, "Learning values across many orders of magnitude"): the value head's last layer is
    rescaled in the same step so that sigma' (W' z + b') + mu' = sigma (W z + b) + mu for every z --
    the critic's de-normalised values are exactly what they were, and only the targets it is trained
    towards move. Without it, the first update alone (identity -> the batch's statistics, sd ~125 in
    G4's runs) re-meant every value the critic had learnt by two orders of magnitude.

    Why at all (G4, architecture-redesign appendix 2): the returns spread with sd ~125 under a contact
    price of 300-1000 while the value moved 0.2 per update under the old clip; unclipped and
    normalised the 0.5 s foresight R went 0.085 -> 0.17."""

    def __init__(self, beta: float = 0.99, eps: float = 1e-4):
        if not 0.0 < float(beta) < 1.0:
            raise ValueError(f"ValueNorm beta must be in (0, 1), got {beta}")
        self.beta, self.eps = float(beta), float(eps)
        self.m1 = torch.zeros((), dtype=torch.float64)
        self.m2 = torch.zeros((), dtype=torch.float64)
        self.debias = torch.zeros((), dtype=torch.float64)

    @property
    def mean(self) -> float:
        return float(self.m1 / self.debias) if float(self.debias) > 0 else 0.0

    @property
    def std(self) -> float:
        if float(self.debias) <= 0:
            return 1.0
        var = float(self.m2 / self.debias) - self.mean ** 2
        return max(var, self.eps) ** 0.5

    @torch.no_grad()
    def update(self, x: torch.Tensor, weight: Optional[torch.Tensor] = None,
               preserve: Optional[nn.Linear] = None) -> None:
        """Fold one rollout's returns in. `preserve`: the critic's value head (a one-output
        `nn.Linear`, the last layer), rescaled so its de-normalised outputs survive the update
        (`rescale_head`)."""
        if preserve is not None:
            self._check_head(preserve)                 # before anything moves: a refusal changes nothing
        x = x.detach().double().reshape(-1)
        w = torch.ones_like(x) if weight is None else weight.detach().double().reshape(-1)
        n = float(w.sum())
        if n <= 0:
            return
        old_mean, old_std = self.mean, self.std
        bm = float((x * w).sum()) / n
        bs = float((x * x * w).sum()) / n
        b = self.beta
        self.m1 = b * self.m1 + (1 - b) * bm
        self.m2 = b * self.m2 + (1 - b) * bs
        self.debias = b * self.debias + (1 - b)
        if preserve is not None:
            self.rescale_head(preserve, old_mean, old_std)

    @torch.no_grad()
    def rescale_head(self, head: nn.Linear, old_mean: float, old_std: float) -> None:
        """PopArt's output-preserving update of the value head for a move from (old_mean, old_std)
        to the current statistics: W' = W sigma / sigma', b' = (sigma b + mu - mu') / sigma'.

        Computed in float64 and stored in the head's dtype, so the de-normalised output moves by
        rounding only (`tests/test_train_v2.py` holds it to float32 precision through the whole
        critic). Adam's moments for the head are left as they are: they normalise each parameter's
        step by its own gradient scale, and at beta 0.99 the scale moves ~1 % per update."""
        self._check_head(head)
        new_mean, new_std = self.mean, self.std
        if new_mean == old_mean and new_std == old_std:
            return
        a = float(old_std) / float(new_std)
        head.weight.copy_(head.weight.double() * a)
        head.bias.copy_((head.bias.double() * float(old_std) + (float(old_mean) - float(new_mean)))
                        / float(new_std))

    @staticmethod
    def _check_head(head) -> None:
        if not isinstance(head, nn.Linear) or head.out_features != 1 or head.bias is None:
            raise ValueError("PopArt rescales a one-output nn.Linear value head with a bias (a shifted "
                             "mean has nowhere else to go)")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"beta": self.beta, "eps": self.eps, "m1": float(self.m1), "m2": float(self.m2),
                "debias": float(self.debias)}

    def load_state_dict(self, d: dict) -> None:
        self.beta, self.eps = float(d["beta"]), float(d["eps"])
        self.m1 = torch.tensor(float(d["m1"]), dtype=torch.float64)
        self.m2 = torch.tensor(float(d["m2"]), dtype=torch.float64)
        self.debias = torch.tensor(float(d["debias"]), dtype=torch.float64)


# ----------------------------------------------------------------------- contact within 1 s
#: [control steps] the classifier's horizon: 1 s at 40 Hz, the G4 gate's `y1 = win(40)`.
CONTACT_HORIZON = 40


def contact_onsets(charge: torch.Tensor, prev: torch.Tensor, done: torch.Tensor):
    """(onset, prev') of the CHARGED car contacts for one step. `charge` (B,) bool: the env's
    `info["car_charge"] > 0` (1 while a contact the car is charged for lasts -- latched at its first
    step, the car-behind exemption decided there, 8786cc0); `prev` the same of the step before in the
    same episode; `done` this step's term | trunc. A new episode starts with no contact under way,
    which is what the env's own reset does to its latch (`_car_prev`, `_car_resp`)."""
    onset = charge & ~prev
    return onset, charge & ~done


def contact_labels(onset: torch.Tensor, term: torch.Tensor, trunc: torch.Tensor,
                   horizon: int = CONTACT_HORIZON) -> Tuple[torch.Tensor, torch.Tensor]:
    """(label (T, B) float, valid (T, B) bool): does a charged car-contact onset fall in the next
    `horizon` transitions of the same episode -- steps t .. t + horizon - 1, the transition out of
    state t first -- and is that knowable from this buffer.

    The window ends early at the episode's end. Then:
      terminated  the episode is over; there is nothing more to wait for, so the label is complete
                  (and under `collision_mode terminate` the crash that ended it is the onset at that
                  very step -- masking these would mask every positive of a terminate run);
      truncated   the episode was cut, its future unknown -> masked, whatever the label says;
    and a window that runs past the buffer's last step is masked the same way. Masking by WHERE the
    window ends, never by what it holds: dropping only the uncertain negatives would bias the
    classifier's base rate towards contact. With a 128-step horizon, the last 39 steps of each chunk
    carry no label."""
    T, B = onset.shape
    H = int(horizon)
    dev = onset.device
    big = torch.full((B,), 10 ** 9, dtype=torch.long, device=dev)
    zero = torch.zeros(B, dtype=torch.long, device=dev)
    label = torch.zeros(T, B, device=dev)
    valid = torch.zeros(T, B, dtype=torch.bool, device=dev)
    od, ed, eterm = big.clone(), zero.clone(), torch.zeros(B, dtype=torch.bool, device=dev)
    for t in range(T - 1, -1, -1):
        done = term[t].bool() | trunc[t].bool()
        on = onset[t].bool()
        if t == T - 1:
            od = torch.where(on, zero, big)
            ed, eterm = zero.clone(), term[t].bool()
        else:
            od = torch.where(on, zero, torch.where(done, big, od + 1))
            ed = torch.where(done, zero, ed + 1)
            eterm = torch.where(done, term[t].bool(), eterm)
        label[t] = (od <= H - 1).float()
        valid[t] = (ed >= H - 1) | eterm
    return label, valid


def binary_auc(score: torch.Tensor, label: torch.Tensor) -> float:
    """ROC AUC of `score` for boolean `label` (ties averaged), NaN without both classes. For the log."""
    s, y = score.detach().double().reshape(-1), label.detach().bool().reshape(-1)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float64, device=s.device)
    uniq, inv, counts = torch.unique(s, return_inverse=True, return_counts=True)
    if uniq.numel() < s.numel():                       # average the ranks of tied scores
        sums = torch.zeros(uniq.numel(), dtype=torch.float64, device=s.device).index_add_(0, inv, ranks)
        ranks = (sums / counts.double())[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# ---------------------------------------------------------------------- per-loss gradient norms
def actor_trunk_parameters(actor) -> list:
    """The deployed actor's shared trunk, B1-B5 (stem, ego, fusion, LayerNorm, GRU): the parameters
    every actor-side loss reaches -- the imitation / policy loss through the head, B8 and the other
    train-time heads through `[e | y]`. Each head's own parameters are left out: they are not where
    the losses compete."""
    return [p for m in (actor.stem, actor.ego, actor.fuse, actor.ln, actor.gru) for p in m.parameters()]


def critic_trunk_parameters(critic) -> list:
    """The critic's shared trunk (its stem, the vector branch, the MLP): what the value loss and the
    contact classifier both reach."""
    return [p for m in (critic.stem, critic.inp, critic.mlp) for p in m.parameters()]


def deployed_actor_parameters(actor) -> list:
    """Every parameter of the deployed actor (the exported graph's), without the train-time heads."""
    heads = {id(p) for h in (actor.b8, actor.beam_car, actor.near_car) if h is not None for p in h.parameters()}
    return [p for p in actor.parameters() if id(p) not in heads]


def loss_grads(terms: Dict[str, Optional[torch.Tensor]], params: Iterable[torch.Tensor],
               norm_params: Optional[Iterable[torch.Tensor]] = None, retain: bool = True):
    """({name: L2 norm of d term / d norm_params}, {name: (d term / d p for p in params)}): each term's
    gradient on its own, without touching `.grad`.

    `params` are the tensors the gradients are taken for (None where a term does not reach one);
    the norm is over `norm_params`, a subset of them (all of them when None) -- the shared trunk,
    while the gradients cover the heads too, so a caller can assemble its update from these very
    gradients instead of paying for one more backward (`accumulate_grads`). `retain=False` frees the
    graph with the last term. A term that is None, not a tensor with a graph, or does not reach
    `norm_params` reads 0 and has no entry in the gradients."""
    params = [p for p in params if p.requires_grad]
    keep = None if norm_params is None else {id(p) for p in norm_params}
    names = [k for k, t in terms.items() if t is not None and torch.is_tensor(t) and t.requires_grad]
    norms = {k: 0.0 for k in terms}
    grads = {}
    for j, name in enumerate(names):
        g = torch.autograd.grad(terms[name], params, retain_graph=retain or j < len(names) - 1,
                                allow_unused=True)
        sq = [gi.detach().float().pow(2).sum() for gi, p in zip(g, params)
              if gi is not None and (keep is None or id(p) in keep)]
        norms[name] = float(torch.stack(sq).sum().sqrt()) if sq else 0.0
        grads[name] = g
    return norms, grads


def loss_grad_norms(terms: Dict[str, Optional[torch.Tensor]], params: Iterable[torch.Tensor]) -> Dict[str, float]:
    """{name: L2 norm of d term / d params} for each term, without touching `.grad`. The graph is
    retained, so the caller's own backward of the total still works afterwards. A term that does not
    reach `params` (or is None / not a tensor with a graph) reads 0."""
    return loss_grads(terms, params)[0]


@torch.no_grad()
def accumulate_grads(params: Sequence[torch.Tensor], grads: Dict[str, Sequence[Optional[torch.Tensor]]],
                     weights: Dict[str, float]) -> None:
    """`p.grad += sum_k weights[k] * grads[k][i]` for parameter i -- the backward of
    `sum_k weights[k] * term_k`, assembled from `loss_grads`' per-term gradients (same `params`,
    same order). Equal to that backward up to the order the floating-point sums are taken in."""
    params = [p for p in params if p.requires_grad]
    for i, p in enumerate(params):
        acc = None
        for k, g in grads.items():
            gi = g[i]
            if gi is None:
                continue
            w = float(weights.get(k, 1.0))
            gi = gi if w == 1.0 else gi * w
            acc = gi if acc is None else acc + gi
        if acc is not None:
            p.grad = acc.detach() if p.grad is None else p.grad + acc


# ----------------------------------------------------------------- auxiliary gradient balance
#: UNIT 5a's adaptive auxiliary scale, defaults (each one a flag on both trainers). What decided it,
#: the per-loss gradient norms on the shared actor trunk of UNIT 4's smoke runs, each auxiliary at its
#: configured coefficient:
#:     DAgger  bc 0.89   B8 1.62   nearest car 4.24   per beam 0.17
#:     PPO     pg 0.72   B8 3.24   nearest car 18.2
#: The trunk was being shaped mostly by the heads that never drive -- the A9/A10 pattern ("heads
#: learned better, drove worse": 188 vs 109 car contacts) in the making. rho 0.5 lets an auxiliary
#: pull at most half as hard as the loss the policy is trained for: enough to write the memory B8
#: exists for (its value), never enough to steer the trunk away from the action. `every` 10: the
#: measurement is a backward pass per loss, so it is taken on one update in ten -- measured on the CPU
#: at the deployed shapes, +7 % of DAgger's training compute with B8 (+18 % with all four losses; see
#: `learn.dagger_v2.train_epochs`), and in PPO one piece of the update's many. `beta` 0.8 per
#: measurement: about the last five measurements, 50 updates -- a minibatch's ratio moves with which
#: cars it happens to hold, and one noisy batch should not swing a coefficient.
BALANCE_EVERY = 10
BALANCE_RHO = 0.5
BALANCE_BETA = 0.8


class AuxBalance:
    """Each auxiliary loss's pull on the shared actor trunk, held at most `rho` x the main loss's.

    Every `every` optimiser updates the trainer measures each loss's gradient norm on the shared
    trunk (`actor_trunk_parameters`; the main loss -- imitation for DAgger, the policy gradient for
    PPO -- and each auxiliary already times its configured coefficient) and hands the norms to
    `observe`. Auxiliary i's multiplier is then

        s_i = min(1, rho |g_main| / |g_i|)

    smoothed by an EMA over measurements (the first measurement sets it outright: starting from 1
    would leave the imbalance in force for the EMA's whole time constant, which is the thing being
    fixed). The multiplier only ever turns an auxiliary DOWN; one that is already weaker than
    rho x the main loss keeps its configured coefficient. It applies on top of that coefficient
    (`coefs`). A measurement with no main gradient (nothing to measure against) changes nothing; an
    auxiliary with no gradient in this batch keeps its multiplier.

    `state_dict` goes into every checkpoint, so a resumed run continues with the multipliers it had
    rather than re-measuring from 1."""

    def __init__(self, main: str, names: Sequence[str], every: int = BALANCE_EVERY, rho: float = BALANCE_RHO,
                 beta: float = BALANCE_BETA):
        if int(every) < 0:
            raise ValueError(f"the balance period is >= 0 updates (0 = off), got {every}")
        if not (math.isfinite(float(rho)) and float(rho) > 0):
            raise ValueError(f"rho is a positive ratio, got {rho}")
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError(f"the balance EMA weight is in [0, 1), got {beta}")
        self.main, self.names = str(main), tuple(str(k) for k in names)
        self.every, self.rho, self.beta = int(every), float(rho), float(beta)
        self.scale = {k: 1.0 for k in self.names}
        self.seen = {k: 0 for k in self.names}
        #: optimiser updates counted by `tick`, and how many of them were measured
        self.updates = 0
        self.measurements = 0
        self.last_norms: Dict[str, float] = {}
        self.last_raw: Dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self.every > 0 and bool(self.names)

    def due(self) -> bool:
        """Is the coming update one whose gradients are measured (the first one always is)?"""
        return self.enabled and self.updates % self.every == 0

    def tick(self) -> None:
        """One optimiser update done."""
        self.updates += 1

    def observe(self, norms: Dict[str, float]) -> Dict[str, float]:
        """Fold one measurement in; returns this measurement's raw multipliers."""
        g_main = float(norms.get(self.main, 0.0))
        self.last_norms = {k: float(v) for k, v in norms.items()}
        raw = {}
        if not (math.isfinite(g_main) and g_main > 0.0):
            return raw
        for k in self.names:
            g = float(norms.get(k, 0.0))
            if not (math.isfinite(g) and g > 0.0):
                continue
            s = min(1.0, self.rho * g_main / g)
            raw[k] = s
            self.scale[k] = s if self.seen[k] == 0 else self.beta * self.scale[k] + (1.0 - self.beta) * s
            self.seen[k] += 1
        self.last_raw = raw
        self.measurements += 1
        return raw

    def coefs(self, base: Dict[str, float]) -> Dict[str, float]:
        """The coefficients to train with: each balanced one times its multiplier, the rest as given."""
        return {k: (float(v) * self.scale[k] if k in self.scale else float(v)) for k, v in base.items()}

    def pull(self) -> Dict[str, float]:
        """The last measurement's effective pull of each auxiliary relative to the main loss,
        s_i |g_i| / |g_main| with the multiplier now in force: <= rho once the EMA has settled."""
        g_main = float(self.last_norms.get(self.main, 0.0))
        if not g_main > 0.0:
            return {}
        return {k: self.scale[k] * float(self.last_norms.get(k, 0.0)) / g_main for k in self.names}

    def state_dict(self) -> dict:
        return {"main": self.main, "names": list(self.names), "every": self.every, "rho": self.rho,
                "beta": self.beta, "scale": dict(self.scale), "seen": dict(self.seen),
                "updates": self.updates, "measurements": self.measurements,
                "last_norms": dict(self.last_norms), "last_raw": dict(self.last_raw)}

    def load_state_dict(self, d: dict) -> None:
        """The multipliers and counters of a run being resumed. The configuration (`every`, `rho`,
        `beta`) stays this run's: a resume that changes a flag gets the flag it asked for. A head this
        run trains that the checkpoint's run did not starts at 1, unmeasured; a different main loss
        (a DAgger balance handed to PPO) is refused -- the ratios mean something else."""
        if str(d.get("main")) != self.main:
            raise ValueError(f"a balance measured against {d.get('main')!r} cannot continue against {self.main!r}")
        for k in self.names:
            if k in (d.get("scale") or {}):
                self.scale[k] = float(d["scale"][k])
                self.seen[k] = int((d.get("seen") or {}).get(k, 1))
        self.updates = int(d.get("updates", 0))
        self.measurements = int(d.get("measurements", 0))
        self.last_norms = {k: float(v) for k, v in (d.get("last_norms") or {}).items()}
        self.last_raw = {k: float(v) for k, v in (d.get("last_raw") or {}).items()}


@torch.no_grad()
def flat_parameters(params: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).float() for p in params])


def relative_movement(before: torch.Tensor, params: Iterable[torch.Tensor]) -> float:
    """||theta - theta_before|| / ||theta_before||: the redesign's "actor 이동" (G5: <= 2e-3 per update;
    every checkpoint that moved its actor more than that touched cars 32-54 times)."""
    after = flat_parameters(params)
    return float((after - before).norm() / before.norm().clamp_min(1e-12))
