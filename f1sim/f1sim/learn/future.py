"""The auxiliary future head: what it predicts, the module, the label alignment and the loss.

Why (`docs/research/memory-policy-2026-09-13.md`, `opponent-diversity-2026-09-13.md`): the actor has
a GRU and it trains against interactive opponents, but nothing makes its hidden state *represent*
the opponent's motion. PPO rewards driving, the grip head asks for a friction, and the opponent head
asks for where the nearest car is *now* -- which six stacked LiDAR frames already almost answer. A
recurrent state that is never asked for anything the present observation does not contain is free to
stay a smoothed copy of it.

So this head asks for the one thing the present observation cannot contain: where the nearest car
will be, and how the ego will be moving, K control steps from now. The claim the paper wants to make
is "a latent race state you plan through"; a head that forces the state to carry the near future is
half of the evidence for it and `learn.probe_hidden` is the other half -- it measures, with ridge
regression on frozen hidden states, how much of the same targets the state explains with the head
trained and without it.

Three properties, each load-bearing:

* **The head reads the recurrent state.** With `--memory gru` its input is the actor GRU's hidden
  state after this step, not the trunk features the action comes from. That is deliberate: the thing
  being measured and the thing being trained have to be the same tensor, or the probe is evidence
  about something else. With memory off there is no such tensor and it reads the trunk features, so
  the feedforward ablation is still available.
* **Zero-initialised output layer.** The last linear's weight AND bias start at exactly zero, the
  same construction `Actor.cond` and `GRUMemory.out` use. The head's output feeds nothing else, so
  forward parity is free; what the zero buys is that `dL/dW_out = delta . a^T` is nonzero from the
  first update (the path trains immediately) while `dL/dW_hidden` is zero for exactly one update.
* **Absent unless asked for.** `future=None` builds no module, records no meta and consumes no RNG,
  so `--aux-future 0` is byte-for-byte the run it was -- which is what
  `tests/data/ppo_loss_oracle.json` pins.

The label itself is privileged and is produced by the simulator: `F1VecEnv.future_labels` returns the
row for the instant it is called, and `align_future_targets` here is what turns a chunk of those rows
into "the label for step t", with the masking discipline written down in one place.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..gym_env import FUTURE_LABEL_DIM, FUTURE_LABEL_KEYS, FUTURE_PRESENT_INDEX

#: Control steps of lookahead. 20 steps at the 40 Hz control rate is 0.5 s -- about a car length and
#: a half at racing speed, and past the 150 ms the six-frame LiDAR stack covers.
FUTURE_K = 20

#: Default width of the head's one hidden layer. The same 128 the grip and opponent heads use.
FUTURE_WIDTH = 128

#: The regression targets, in the order `FUTURE_LABEL_KEYS` puts them: everything but the presence
#: logit, which is trained with binary cross-entropy instead.
FUTURE_REGRESSION_KEYS = tuple(k for i, k in enumerate(FUTURE_LABEL_KEYS) if i != FUTURE_PRESENT_INDEX)

#: Which of the regression columns describe the opponent, and are therefore meaningless (and masked)
#: when no opponent is inside `overtake_range` at t+K. The remaining ones describe the ego and are
#: always scored.
FUTURE_OPPONENT_KEYS = ("opp_lon", "opp_lat", "opp_vlon", "opp_vlat")


def future_spec(k: int = FUTURE_K, width: int = FUTURE_WIDTH, source: Optional[str] = None,
                targets: Optional[Sequence[str]] = None) -> dict:
    """The `meta["future_head"]` block, validated. Written by the trainer, read by every loader.

    Idempotent -- `future_spec(**future_spec())` is the same dict -- because a checkpoint's recorded
    block is handed straight back to the constructor, and a validator that refused its own output
    would make "load what you saved" the one path nobody tested.

    `targets` is recorded rather than taken: a checkpoint whose head was trained on a different
    target list is not this head, and silently re-using its weights would compare two things.

    `source` may be None, meaning "whatever the actor this is attached to has" -- `Actor.
    attach_future` fills it in and refuses a spec that names the other one, so a checkpoint that
    recorded `memory` cannot be rebuilt on a feedforward actor without saying so.
    """
    k, width = int(k), int(width)
    if k < 0:
        raise ValueError(f"future head k {k} must be >= 0 control steps")
    if width <= 0:
        raise ValueError(f"future head width {width} must be positive")
    if source is not None and source not in ("memory", "trunk"):
        raise ValueError(f"future head source must be 'memory' or 'trunk', got {source!r}")
    if targets is not None and tuple(targets) != FUTURE_LABEL_KEYS:
        raise ValueError(f"this checkpoint's future head was trained on targets {list(targets)}, "
                         f"which are not this build's {list(FUTURE_LABEL_KEYS)}")
    return {"k": k, "width": width, "source": source, "targets": list(FUTURE_LABEL_KEYS)}


class FutureHead(nn.Module):
    """`in_dim -> width -> FUTURE_LABEL_DIM`, with the output layer at exactly zero.

    One hidden layer, like the grip and opponent heads: a linear read-out would be the probe, and a
    target the trunk can only reach linearly is a target the trunk is not being asked to represent.
    """

    def __init__(self, in_dim: int, width: int = FUTURE_WIDTH, out_dim: int = FUTURE_LABEL_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(int(in_dim), int(width)), nn.GELU(),
                                 nn.Linear(int(width), int(out_dim)))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------ labels
def align_future_targets(labels: torch.Tensor, boundary: torch.Tensor, k: int):
    """(target, valid) for every step of a rollout chunk, from the labels recorded during it.

    `labels` is (T + 1, B, D): the privileged label row for the state at time t, for t = 0 .. T. The
    extra row is the state the chunk ends in -- the same state the value bootstrap is taken from --
    so no extra simulator step is needed to get it.

    `boundary` is (T, B) and is 1 where the transition from t to t + 1 crossed an episode boundary
    for that row. In a race that has to mean *any car in the race reset*, not just this one: an
    opponent that crashes is respawned behind the field in place, so its position at t + k belongs
    to a different situation than the one at t. `F1VecEnv.race_boundary` builds it.

    Returns `target` (T, B, D), zero where there is no label, and `valid` (T, B) in {0, 1}:

    * `valid[t] = 0` for the last k steps of the chunk, because `labels[t + k]` does not exist. The
      chunk carries no state from the next rollout, so those steps are dropped rather than
      approximated -- with the default `--horizon 32` and k = 20 that leaves 13 of 32 steps labelled.
      Raising `--horizon` raises the fraction; `future_labelled_fraction` computes it.
    * `valid[t] = 0` wherever a boundary falls in t .. t + k - 1, so a label is never read across a
      reset.

    Presence masking is NOT applied here: `target[..., FUTURE_PRESENT_INDEX]` is the label the
    presence logit trains on, and `future_loss` is what drops the opponent columns where it is 0.
    """
    k = int(k)
    if labels.dim() != 3 or boundary.dim() != 2:
        raise ValueError(f"labels must be (T+1, B, D) and boundary (T, B), got {tuple(labels.shape)} "
                         f"and {tuple(boundary.shape)}")
    T, B = boundary.shape
    if labels.shape[0] != T + 1 or labels.shape[1] != B:
        raise ValueError(f"labels (T+1, B, D) must match boundary (T, B): {tuple(labels.shape)} vs "
                         f"{tuple(boundary.shape)}")
    target = torch.zeros(T, B, labels.shape[2], device=labels.device, dtype=labels.dtype)
    valid = torch.zeros(T, B, device=labels.device, dtype=labels.dtype)
    n = min(T, T + 1 - k)                        # steps whose label row exists inside this chunk
    if n <= 0:
        return target, valid
    target[:n] = labels[k:k + n]
    # cum[t] = boundaries strictly before t, so boundaries in [t, t + k) is cum[t + k] - cum[t].
    cum = torch.zeros(T + 1, B, device=labels.device, dtype=labels.dtype)
    cum[1:] = boundary.to(labels.dtype).cumsum(0)
    valid[:n] = (cum[k:k + n] - cum[:n] == 0).to(labels.dtype)
    return target, valid


def future_labelled_fraction(horizon: int, k: int) -> float:
    """The share of a chunk's steps that can carry a label at all, boundaries aside."""
    return max(0, min(int(horizon), int(horizon) + 1 - int(k))) / max(1, int(horizon))


# ------------------------------------------------------------------ loss
def _wmean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x * w).sum() / w.sum().clamp_min(1.0)


def future_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, w: torch.Tensor,
                keys: Sequence[str] = FUTURE_LABEL_KEYS):
    """(scalar to add to the loss, per-component dict) for one minibatch.

    `pred` and `target` are (n, D), `valid` and `w` are (n,). `w` is the sample weight the rest of
    the PPO loss uses (0 on a teacher-driven car's transitions); `valid` is the alignment mask from
    `align_future_targets`. Their product is the weight everything here is averaged under.

    Three groups, because they are masked differently:

    * the four opponent columns are additionally weighted by the presence label at t + k -- with no
      car inside `overtake_range` there is no relative position to predict, and scoring the head on
      a zero would teach it to answer "straight ahead, touching" for an empty road;
    * the two ego columns are scored wherever the label exists, opponent or not;
    * the presence logit is scored wherever the label exists, ALWAYS -- it is the column that says
      whether the others mean anything, so training it only on the frames where a car is present
      would leave it unable to say "no car".

    The per-component dict additionally carries `<key>_var` (the target's variance under the same
    weights) and `opp_present_bce_base` (the cross-entropy of predicting the base rate), so a
    reader can normalise both halves instead of reading a raw error against a moving distribution.

    The returned scalar is the mean of the six mean-squared errors plus the presence
    cross-entropy. Both are O(0.1-1) at the zero init (the logit starts at 0, i.e. p = 0.5, so the
    cross-entropy starts at log 2), so one coefficient in front of the sum is honest weighting
    rather than an unstated second hyper-parameter.
    """
    base = (valid * w).to(pred.dtype)
    present = target[:, FUTURE_PRESENT_INDEX].to(pred.dtype)
    per = {}
    total = torch.zeros((), device=pred.device, dtype=pred.dtype)
    n_reg = 0
    for i, key in enumerate(keys):
        if i == FUTURE_PRESENT_INDEX:
            continue
        wi = base * present if key in FUTURE_OPPONENT_KEYS else base
        term = _wmean((pred[:, i] - target[:, i]) ** 2, wi)
        per[key] = term
        # ... and the variance of the target under the SAME weights. A raw MSE against a target
        # whose distribution is still moving -- which it is, for the whole early part of a finetune,
        # because the policy is learning to go faster and further -- is not a learning curve. With
        # the variance beside it, `1 - mse / var` is the head's own explained variance, which is
        # both interpretable and directly comparable to what `probe_hidden` reports.
        mean_i = _wmean(target[:, i], wi)
        per[key + "_var"] = _wmean((target[:, i] - mean_i) ** 2, wi)
        total = total + term
        n_reg += 1
    total = total / max(1, n_reg)
    bce = _wmean(nn.functional.binary_cross_entropy_with_logits(
        pred[:, FUTURE_PRESENT_INDEX], present, reduction="none"), base)
    per[str(keys[FUTURE_PRESENT_INDEX]) + "_bce"] = bce
    # the cross-entropy a constant predictor of the base rate would pay: the floor the logit has to
    # beat. In a tight three-car field the presence label is nearly always 1 and this is near zero,
    # which is worth knowing before reading anything into a falling BCE.
    p_ = _wmean(present, base).clamp(1e-6, 1 - 1e-6)
    per[str(keys[FUTURE_PRESENT_INDEX]) + "_bce_base"] = -(p_ * p_.log() + (1 - p_) * (1 - p_).log())
    per["labelled_frac"] = base.gt(0).to(pred.dtype).mean()
    per["present_frac"] = _wmean(present, base)
    return total + bce, per


def describe(meta: Optional[dict]) -> str:
    """One line for a log or a checkpoint header."""
    f = (meta or {}).get("future_head")
    if not f:
        return "no future head"
    return (f"future head: k={f.get('k')} steps ({0.025 * int(f.get('k', 0)):.2f} s) from the "
            f"{f.get('source')} state, width {f.get('width')}, "
            f"{len(f.get('targets') or FUTURE_LABEL_KEYS)} targets")
