"""The train-time per-beam floor/solid auxiliary head, and its loss.

The geometric channel (`learn/floor.py`) is explicit: the car computes it and reads it. This is the
implicit half of the same idea -- the trunk is *asked* to know, per beam, whether a return is the
floor or a solid object, from a label the simulator gives away for free (`scan_type == HIT_GROUND`).
Nothing about it reaches the car: the head is never called by the actor's forward, never exported
(`learn/export.py` traces `actor.step`), and off by default, so an unflagged run is the run it was
down to the state dict.

Why a per-beam head rather than one more scalar on the trunk features
---------------------------------------------------------------------
"Is there floor in this scan" is a scalar the 256-D embedding can carry without representing *where*
the floor is, and where is the whole question -- a phantom wall matters because of the bearings it
occupies. A per-beam target cannot be satisfied by a summary.

What it reads
-------------
Two things, and both are needed:

* **the trunk's own feature map**, `(B, 128, N/32)` for the resnet stem -- whole-scan context, which
  is what says "this arc of returns lies on one line" rather than "this return is at 2.1 m";
* **the stem's input rows at full beam resolution** -- the six frames, the beam-angle ramp, the
  windowed nearest return and any extra channels. A label is per beam and the trunk's map is one
  position per 32 beams, so a head on context alone could not resolve the edge of the floor arc.

The context is upsampled to the beam axis and concatenated with a 5-tap convolution of the input
rows. That is a small U-net's decoder and nothing more.

The failure this is built to avoid
----------------------------------
`work/motion-memory/REPORT.md` §3: an equivalent per-beam head "collapses to 'no car' in both arms --
recall 0.19 -> 0.00 by update ~40, precision to 0 -- while its BCE falls 10 %. A loss curve alone
would have read as learning." Two consequences here:

* `floor_loss` returns **recall and precision alongside the loss**, every update, and `ppo` logs
  them. A head that has collapsed is visible in the line it prints.
* the positive weight is `negatives / positives` **from the batch**, and its clamp is reported, so a
  binding clamp -- which is what left that head's positives outweighed 1.6 : 1 at a 1.2 % rate -- is
  a number in the log rather than a thing to discover afterwards.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Ceiling on the positive weight. High on purpose. The floor's share of *returning* beams is a few
#: per cent while driving, so the honest reciprocal is 20-100 and the clamp does not bind; but a
#: rollout that starts at the line sees 0.1 %, which asks for 1000, and a clamp of 300 there would
#: leave positives outweighed 3.3 : 1. That is worse than the 1.6 : 1 the motion-memory mask head
#: collapsed under. So the ceiling is only a guard against a batch with a handful of positives,
#: and `floor_loss` reports `pos_weight_clamped` so a run where it bound is visible rather than
#: discovered afterwards.
POS_WEIGHT_MAX = 1000.0


def floor_head_spec(width: int = 32, kernel: int = 5) -> dict:
    """The validated head configuration recorded in `meta["floor_head"]`."""
    width, kernel = int(width), int(kernel)
    if width < 4:
        raise ValueError(f"floor head width {width} must be at least 4")
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError(f"floor head kernel {kernel} must be odd and positive")
    return {"width": width, "kernel": kernel}


class FloorHead(nn.Module):
    """(trunk feature map, stem input rows) -> one logit per beam."""

    def __init__(self, c_ctx: int, c_in: int, width: int = 32, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.ctx = nn.Conv1d(c_ctx, width, 1)
        self.local = nn.Conv1d(c_in, width, kernel, padding=pad)
        self.mix = nn.Sequential(nn.GELU(), nn.Conv1d(2 * width, width, kernel, padding=pad),
                                 nn.GELU(), nn.Conv1d(width, 1, 1))
        #: Zero output layer, the same construction every other addition in this project uses: the
        #: head predicts exactly 0.5 for every beam at init, so nothing about the first update
        #: depends on how it was drawn. It also means the head has to be *given a budget* -- see the
        #: future-head note, where 786 Adam steps left an output layer at an rms of 0.0067.
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, ctx: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """`ctx` (B, C, L) the trunk map, `rows` (B, C_in, N) the stem's input. -> (B, N) logits."""
        up = F.interpolate(self.ctx(ctx), size=rows.shape[-1], mode="linear", align_corners=False)
        return self.mix(torch.cat([up, self.local(rows)], 1))[:, 0]


def floor_loss(logits: torch.Tensor, label: torch.Tensor, valid: torch.Tensor,
               w: Optional[torch.Tensor] = None, pos_weight_max: float = POS_WEIGHT_MAX):
    """Weighted BCE over the beams that returned something, plus what it is actually doing.

    `logits` (M, N); `label` (M, N) in {0, 1} with 1 = floor; `valid` (M, N) in {0, 1}, 0 where the
    beam had no return (there is nothing to classify) ; `w` (M,) the PPO sample weights, so a
    teacher-driven row that carries no policy gradient carries no auxiliary gradient either.

    Returns `(loss, parts)` with `parts` carrying `recall`, `precision`, `rate` (the batch's
    positive share), `pos_weight` and `pos_weight_clamped` -- the five numbers that tell a collapsed
    head from a learning one.
    """
    m = valid.float()
    if w is not None:
        m = m * w.reshape(-1, *([1] * (valid.dim() - 1))).float()
    y = label.float()
    n_pos = (y * m).sum()
    n_tot = m.sum().clamp_min(1.0)
    rate = n_pos / n_tot
    raw = (n_tot - n_pos) / n_pos.clamp_min(1.0)
    pw = raw.clamp(1.0, float(pos_weight_max))
    per = F.binary_cross_entropy_with_logits(
        logits.float(), y, weight=None, reduction="none",
        pos_weight=None)
    # The positive weight applied by hand rather than through `pos_weight=`: that argument expects a
    # per-class tensor broadcast over the last dimension, and here the weight is one scalar over a
    # beam axis whose entries are all the same class problem.
    per = per * (1.0 + (pw - 1.0) * y)
    loss = (per * m).sum() / (m * (1.0 + (pw - 1.0) * y)).sum().clamp_min(1.0)
    with torch.no_grad():
        pred = (logits > 0).float()
        tp = (pred * y * m).sum()
        fp = (pred * (1 - y) * m).sum()
        fn = ((1 - pred) * y * m).sum()
        parts = {"recall": tp / (tp + fn).clamp_min(1.0),
                 "precision": tp / (tp + fp).clamp_min(1.0),
                 "rate": rate, "pos_weight": pw,
                 "pos_weight_clamped": (raw > pw).float()}
    return loss, parts


def describe(spec: Optional[dict]) -> str:
    if not spec:
        return "no floor head"
    return (f"floor head: per-beam solid/floor, width {spec.get('width')}, kernel "
            f"{spec.get('kernel')} (train-time only, never exported)")
