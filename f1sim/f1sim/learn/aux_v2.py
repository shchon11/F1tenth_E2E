"""The v2 actor's train-time auxiliary heads, their targets and their losses.

None of these is ever called by the actor's forward, so none of them is in the exported graph or on
the car. Each is built only when its flag is on (`aux_spec`), after both networks, so turning one on
leaves every other weight drawn exactly as without it.

* **B8, opponent slots** (docs/research/architecture-redesign-2026-10-13.md, B8). Three slots x
  [presence logit, dx / 5, dy / 2, v_lat / 2] read from `[e | y]`, the features the action head
  reads -- including the memory's output, which is the point: the redesign's bisection found that
  with the stem's gradient flowing on every step a GRU does not learn to carry a cue, and that a
  dense target which cannot be met without memory makes all three GRU variants learn it, 3/3 seeds
  (`proposal-temporal/aux_*.json`; on this actor, `tests/test_model_v2.py`: learned at iteration
  205 with one, flat at chance for 500 without). G1 measured it on real driving: occluded-within-
  1-s position R^2 0.609 with the trainable stem and this memory, -0.111 without memory. The slots are an unordered set, matched to the cars by the cheapest of the
  slot permutations (`b8_loss`) -- a distance-ordered slot list relabels at every swap of rank,
  which a judge of the redesign pointed out.

  Slots and permutations (critique A6: "3 slots are 3! = 6 permutations, not 24"): 3 slots, so 6
  permutations, over at most 3 candidate cars. A race can hold more (race_size 5+). The rule then:
  **the three nearest candidates are the set** (`b8_select`); a fourth is outside it by
  definition, exactly as a car beyond 6 m is, and its slot is neither supervised nor penalised.
  Requirement P2 asks for the positions of up to three cars within 6 m; a fourth car within 6 m of
  a 0.5 m car is a pile-up, not a race.

  A caveat measured on a CPU toy (the 8-step recall with B8 as the ONLY auxiliary, fresh network):
  with the remembered car at one of two places the set loss has a hedge -- two slots on the two
  answers, presence 0.5 each, 2 ln 2 / 3 = 0.46 -- and the memory did not learn in 400 iterations;
  with a continuous position it did, slowly (action error 0.084 against a 0.333 floor after 1000
  iterations, 0.546 with the state cut; a direct dense regression learns the binary task in 205).
  Real positions are continuous, but one B8 step is a small gradient (/ 3 slots, x 0.5), so the
  training unit should log it against the imitation loss's (the redesign's G5 per-loss norms).

  Which candidates may be scored is the visibility mask (`b8_know_mask`): seen by >= 3 beams within
  the last 3 s (120 steps), AND that sighting inside the current training window -- the chunk or
  its burn-in -- because a state started at the window's beginning cannot know what it never saw
  (with chunk 80 and no such rule 52.6 % of the memory labels were unanswerable,
  judge-learnability-minimal/chunk_mask.json). A candidate that is NOT knowable still occupies a
  slot -- at zero cost, so the head is neither told it is there nor told it is not.

* **Per-beam car / not-car** (appendix 2's candidate, the perception gate's auxiliary A): a light
  head on the stem's neck map, one logit per beam, scored on valid returns only.

* **Nearest visible car** (appendix 2, perception gate auxiliary B): [exists, dx / 5, dy / 2,
  closing / 5, v_lat / 2] of the nearest car the newest scan sees, from `[e | y]`. The align gate
  found the true distance to that one car gives contact-risk AUC 0.902 by itself.

The loss weights are the gates' (w_pos 10, w_vl 0.1, B8 coefficient 0.5); they are defaults here and
the training unit owns the coefficients.
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gym_env import OPP_MEMORY_STEPS

#: The three auxiliary families, in the order they are attached.
AUX_HEADS = ("b8", "beam_car", "near_car")

B8_SLOTS = 3
#: Per slot: presence logit, dx / 5, dy / 2, v_lat / 2 (LiDAR frame), the align / perception gates'.
B8_OUT = 4
B8_W_POS = 10.0
B8_W_VL = 0.1
B8_COEF = 0.5
#: sign of each label column under a left-right mirror of the world: dx, dy, v_lat
B8_LABEL_MIRROR = (1.0, -1.0, -1.0)
#: [exists, dx / 5, dy / 2, closing / 5, v_lat / 2]
NEAR_OUT = 5
NEAR_W_POS = 10.0
NEAR_W_CLOSING = 1.0
NEAR_W_VL = 0.1
NEAR_MIRROR = (1.0, 1.0, -1.0, 1.0, -1.0)


def aux_spec(b8: bool = False, beam_car: bool = False, near_car: bool = False) -> dict:
    """The `meta["aux"]` block, validated: which train-time heads a v2 actor carries."""
    return {"b8": bool(b8), "beam_car": bool(beam_car), "near_car": bool(near_car)}


class OpponentSlotHead(nn.Module):
    """B8: `[e | y]` (384) -> 3 slots x [presence logit, dx / 5, dy / 2, v_lat / 2]."""

    def __init__(self, in_dim: int, width: int = 256, slots: int = B8_SLOTS):
        super().__init__()
        self.slots = int(slots)
        self.net = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Linear(width, self.slots * B8_OUT))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).reshape(*z.shape[:-1], self.slots, B8_OUT)


class NearestCarHead(nn.Module):
    """`[e | y]` -> [exists logit, dx / 5, dy / 2, closing / 5, v_lat / 2] of the nearest seen car."""

    def __init__(self, in_dim: int, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Linear(width, NEAR_OUT))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BeamCarHead(nn.Module):
    """Per-beam car logit from the stem's neck map (B, 48, L): two convolutions on the map, each cell
    unfolded into the `up` beams it covers (the trunk's total stride, 32), cropped to N beams. The
    perception gate's auxiliary A, on the map `fc` reads -- so the gradient reaches exactly the
    features the action is computed from."""

    def __init__(self, neck_channels: int, cells: int, n_beams: int, width: int = 64):
        super().__init__()
        self.n_beams, self.cells = int(n_beams), int(cells)
        self.up = int(math.ceil(self.n_beams / self.cells))
        self.net = nn.Sequential(nn.Conv1d(neck_channels, width, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(width, self.up, 1))

    def forward(self, neck: torch.Tensor) -> torch.Tensor:
        o = self.net(neck)                                           # (B, up, L)
        return o.permute(0, 2, 1).reshape(o.shape[0], self.cells * self.up)[:, :self.n_beams]


# ---------------------------------------------------------------- B8 targets and loss
_PERMS = {}


def _perms(n: int, device) -> torch.Tensor:
    key = (n, str(device))
    if key not in _PERMS:
        _PERMS[key] = torch.tensor(list(itertools.permutations(range(n))), device=device)
    return _PERMS[key]


def b8_know_mask(cand: torch.Tensor, age: torch.Tensor, steps_into_window: torch.Tensor,
                 max_age: int = OPP_MEMORY_STEPS) -> torch.Tensor:
    """Which candidates a recurrent state started at the window's first step can know.

    `cand`, `age` (..., C) from `F1VecEnv.opponent_tracks_v2` at each step; `steps_into_window`
    (...,) how many steps this row is past the first step of its training window (the burn-in's
    first step when there is one; 0 there). Known = a candidate seen within `max_age` steps whose
    last sighting is not before the window began: age <= max_age and age <= steps_into_window.
    The perception gate's `know = cand & (t - last <= 120) & (last >= window_start)`."""
    s = steps_into_window.to(age.dtype)[..., None]
    return cand & (age <= int(max_age)) & (age <= s)


def b8_select(lab: torch.Tensor, cand: torch.Tensor, know: torch.Tensor, dist: torch.Tensor,
              slots: int = B8_SLOTS):
    """(lab (..., slots, 3), cand (..., slots), know (..., slots)): the per-seat labels reduced or
    padded to exactly `slots` tracks.

    Fewer seats than slots (a solo or two-car race): padded with absent tracks, which the loss turns
    into "presence 0" for the spare slots. More (critique A6): the `slots` nearest CANDIDATES by
    LiDAR distance are the set; the rest leave the row's assignment entirely."""
    C = lab.shape[-2]
    if C < slots:
        pad = slots - C
        lab = torch.cat([lab, lab.new_zeros(*lab.shape[:-2], pad, lab.shape[-1])], -2)
        cand = torch.cat([cand, cand.new_zeros(*cand.shape[:-1], pad)], -1)
        know = torch.cat([know, know.new_zeros(*know.shape[:-1], pad)], -1)
        return lab, cand, know
    if C == slots:
        return lab, cand, know
    key = torch.where(cand, dist, torch.full_like(dist, float("inf")))
    idx = key.argsort(-1)[..., :slots]
    take = lambda t: t.gather(-1, idx)
    return lab.gather(-2, idx[..., None].expand(*idx.shape, lab.shape[-1])), take(cand), take(know)


def b8_loss(pred: torch.Tensor, lab: torch.Tensor, cand: torch.Tensor, know: torch.Tensor,
            w_pos: float = B8_W_POS, w_vl: float = B8_W_VL) -> torch.Tensor:
    """Per-row B8 loss (...,): the cheapest assignment of the slots to the tracks, over all slot
    permutations, divided by the slot count.

    `pred` (..., S, 4) from `OpponentSlotHead`; `lab` (..., S, 3), `cand`, `know` (..., S) from
    `b8_select`. A slot assigned to a track costs
      knowable candidate   softplus(-logit) + w_pos |dx, dy error|^2 + w_vl v_lat error^2
      unknowable candidate 0 -- the car is there and the window cannot know it
      no candidate         softplus(logit): presence 0
    (the perception gate's `b8_loss`, generalised from 3 to S slots)."""
    S = pred.shape[-2]
    if lab.shape[-2] != S:
        raise ValueError(f"{pred.shape[-2]} slots against {lab.shape[-2]} tracks: run b8_select first")
    ex, pos, vl = pred[..., 0], pred[..., 1:3], pred[..., 3]
    c_on = (F.softplus(-ex)[..., :, None]
            + w_pos * ((pos[..., :, None, :] - lab[..., None, :, :2]) ** 2).sum(-1)
            + w_vl * (vl[..., :, None] - lab[..., None, :, 2]) ** 2)          # (..., slot, track)
    c_off = F.softplus(ex)[..., :, None].expand_as(c_on)
    cost = torch.where(know[..., None, :], c_on,
                       torch.where(cand[..., None, :], torch.zeros_like(c_on), c_off))
    P = _perms(S, pred.device)                                               # (S!, S)
    slot = torch.arange(S, device=pred.device)
    tot = cost[..., slot[None, :], P].sum(-1)                                # (..., S!)
    return tot.min(-1).values / S


def nearest_car_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-row loss of the nearest-visible-car head (...,): presence BCE, and where a car is seen,
    10 |dx, dy error|^2 + 1 closing error^2 + 0.1 v_lat error^2 (the perception gate's `nv_loss`)."""
    ex = target[..., 0]
    bce = F.binary_cross_entropy_with_logits(pred[..., 0], ex, reduction="none")
    reg = (NEAR_W_POS * ((pred[..., 1] - target[..., 1]) ** 2 + (pred[..., 2] - target[..., 2]) ** 2)
           + NEAR_W_CLOSING * (pred[..., 3] - target[..., 3]) ** 2
           + NEAR_W_VL * (pred[..., 4] - target[..., 4]) ** 2)
    return bce + ex * reg


def beam_car_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Masked per-beam BCE (scalar): `labels` from `F1VecEnv.beam_car_labels_v2`, 1 car, 0 not a
    car, -1 no usable return (not scored)."""
    m = (labels >= 0).to(logits.dtype)
    y = (labels == 1).to(logits.dtype)
    l_ = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (l_ * m).sum() / m.sum().clamp_min(1.0)
