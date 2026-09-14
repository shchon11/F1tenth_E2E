"""A second, small recurrent state fed only by the aligned residual, and the losses that shape it.

Why a separate state (CONTRACT.md and its 05:20 addendum): `probe_hidden --current` says the actor's
GRU carries the ego's own motion and not the opponent's relative velocity. Handing the same GRU an
extra input and an extra loss would not fix that -- ego dynamics are the easy way to drive the loss
down and the state would go on spending itself on them. So the recurrence is **split**:

    current LiDAR ----- scan stem -------------> main GRU ----+
    aligned residual -- motion encoder -------> motion GRU ---+--> plan head -> curvatures, speeds
                             |                      |
                             +-- beam mask          +-- current Dv, and the 0.5 s future head

and the auxiliary losses attach to `h_dyn` and the motion encoder **only**. Nothing in `mask_loss`
or `dv_loss` reaches the main GRU or the scan stem: the motion encoder's input is the aligned rows
of the observation, which is a leaf as far as those modules are concerned, so the gradient has
nowhere else to go. That is the point of the split and it is the one property worth testing.

Three more, each load-bearing:

* **Zero-initialised projection.** `h_dyn` enters the actor's (and the critic's) first MLP
  preactivation through a bias-free projection initialised to zero -- the construction `Actor.cond`
  and `GRUMemory` already use. Concatenating `h_dyn` to the head's input and zeroing the new columns
  of the first layer is the same arithmetic; this spelling keeps the existing weights at their own
  indices, so a warm start is a copy plus zeros rather than a reshape.
* **The state is one tensor.** `h_dyn` is carried in the same `(layers, batch, H)` tensor as the
  main hidden state, concatenated on the feature axis, so every path that already carries a hidden
  state -- the rollout buffers, `reset_hidden`, the ROS node's callback state, the viewer's static
  CUDA-graph buffer, the ONNX export's `hidden` input -- carries this one too, unchanged, and
  nothing has to learn about a second state.
* **The heads are train-time only.** `mask` and `dv` are never called by `Actor.forward` or
  `Actor.step`, so they are absent from the exported graph by construction; deployment uses only
  what `h_dyn` projects into the plan head. `tests/test_motion_memory.py` checks the export.

The beam mask's label is the LiDAR's own `scan_type == HIT_CAR` (`f1sim.lidar`), which the simulator
has cast all along -- privileged, train-time, and free.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Width of `h_dyn`. The contract caps it at 64 and the point of the experiment is that the remedy
#: is inductive bias rather than capacity, so it is not a knob to turn up when a result disappoints.
MOTION_HIDDEN = 64

#: Output channels of the motion encoder's last convolution. The per-position features the beam-mask
#: head reads and, pooled, what the motion GRU is fed.
MOTION_CHANNELS = 32

#: Width of the two auxiliary heads' hidden layer.
MOTION_HEAD_WIDTH = 64

#: The upper bound on the positive-class weight the beam-mask BCE uses.
#:
#: **Raised from 50 to 1000 for phase 2**, and the reason is mechanical rather than aesthetic. With
#: weight `w`, a positive at logit 0 pulls with `w/2` and a negative with `1/2`, so the two classes'
#: total gradient mass is balanced exactly when `w = (1 - rate) / rate`, i.e. at the reciprocal. At
#: the measured 1.2 % positive rate that is about 83; clamped at 50 the negatives kept about
#: 1.6 times the pull, and in phase 1 both arms' heads collapsed to predicting no positive anywhere
#: by update 40 (`docs/research/motion-memory-2026-09-14.md`). The clamp was the binding constraint
#: and it bound in the direction of the collapse.
#:
#: What is left is a numerical bound, not a design choice: a minibatch is ~1.1 M scored beams, so a
#: single positive would ask for a weight of a million. 1000 catches that and nothing a real scene
#: produces. The loss reports the rate and the weight it used, so a clamp that binds again is
#: visible in the log rather than implicit.
MASK_POS_WEIGHT_MAX = 1000.0


def motion_spec(hidden_size: int = MOTION_HIDDEN, channels: int = MOTION_CHANNELS,
                width: int = MOTION_HEAD_WIDTH, critic: str = "own",
                rows: Optional[Sequence[str]] = None) -> dict:
    """The `meta["motion"]` block, validated. Written by the trainer, read by every loader.

    Idempotent, for the reason `memory_spec` and `future_spec` are: a checkpoint's recorded block is
    handed straight back to the constructor.

    `rows` is which scan channels the motion encoder reads, recorded rather than assumed: a
    checkpoint whose encoder was built over three rows cannot be rebuilt over one, and the first
    convolution's input width is the thing that would silently disagree.
    """
    hidden_size, channels, width = int(hidden_size), int(channels), int(width)
    if not 0 < hidden_size <= 64:
        raise ValueError(f"motion hidden_size {hidden_size} must be in (0, 64]: the contract caps "
                         f"the motion state at 64, and widening it is the remedy this experiment "
                         f"is testing an alternative to")
    if channels <= 0 or width <= 0:
        raise ValueError(f"motion channels {channels} and width {width} must be positive")
    if critic not in ("own", "none"):
        raise ValueError(f"motion critic must be 'own' or 'none', got {critic!r}")
    from .obs import ALIGNED_CHANNELS
    rows = tuple(rows) if rows is not None else ALIGNED_CHANNELS
    unknown = [r for r in rows if r not in ALIGNED_CHANNELS]
    if unknown:
        raise ValueError(f"the motion encoder reads aligned rows; {unknown} are not among "
                         f"{list(ALIGNED_CHANNELS)}")
    if not rows:
        raise ValueError("the motion encoder needs at least one aligned row to read")
    return {"hidden_size": hidden_size, "channels": channels, "width": width, "critic": critic,
            "rows": [r for r in ALIGNED_CHANNELS if r in rows]}


class MotionEncoder(nn.Module):
    """1D convolutions over the aligned rows: (B, R, N) -> per-position (B, C, L) and pooled (B, 2C).

    Deliberately small and deliberately separate from `ScanStem`. Separate because the whole design
    is that this branch is not allowed to become another way of encoding the corridor; small because
    what it has to read is a sparse, signed, mostly-zero row, not a range image.

    The pooled vector is `cat(mean, max)` over the beam axis. Mean because how much of the scan is
    moving is itself information, max because one car is a local event and an average over 1081
    beams would bury it.
    """

    STRIDES = (2, 2, 2)

    def __init__(self, in_rows: int, channels: int = MOTION_CHANNELS):
        super().__init__()
        self.in_rows, self.channels = int(in_rows), int(channels)
        self.net = nn.Sequential(
            nn.Conv1d(self.in_rows, 16, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(16, 24, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(24, self.channels, 5, stride=2, padding=2), nn.GELU())

    @property
    def stride(self) -> int:
        s = 1
        for v in self.STRIDES:
            s *= v
        return s

    def forward(self, rows: torch.Tensor):
        if rows.dim() != 3 or rows.shape[1] != self.in_rows:
            raise ValueError(f"motion encoder takes (B, {self.in_rows}, N), got {tuple(rows.shape)}")
        f = self.net(rows)
        return f, torch.cat([f.mean(2), f.amax(2)], 1)

    @property
    def out_dim(self) -> int:
        return 2 * self.channels


class MotionMemory(nn.Module):
    """The motion branch: encoder, GRU over its pooled features, zero-initialised projection out.

    `step` advances one control step and returns (the term added to the head's preactivation, the
    next `h_dyn`, the encoder's per-position features). The features come back because the beam-mask
    head reads them and because returning them is how the head gets a tensor whose gradient reaches
    the encoder and nothing else.
    """

    def __init__(self, in_rows: int, hidden_size: int, out_dim: int, channels: int = MOTION_CHANNELS):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.encoder = MotionEncoder(in_rows, channels)
        self.gru = nn.GRU(self.encoder.out_dim, self.hidden_size, num_layers=1, batch_first=True)
        self.out = nn.Linear(self.hidden_size, int(out_dim), bias=False)
        nn.init.zeros_(self.out.weight)

    def step(self, rows: torch.Tensor, h: Optional[torch.Tensor]):
        feat, pooled = self.encoder(rows)
        if h is None:
            h = torch.zeros(1, rows.shape[0], self.hidden_size, device=rows.device, dtype=rows.dtype)
        y, h_next = self.gru(pooled.unsqueeze(1), h)
        return self.out(y[:, 0]), h_next, feat


class OppMaskHead(nn.Module):
    """Per-beam "is this beam on another car", from the motion encoder's features alone.

    The encoder strides the beam axis by 8, so one logit per position would be a 2 deg answer to a
    per-beam question. Instead the head emits `up` logits per position and they are unfolded back
    along the beam axis -- an exact partition of the beams, no interpolation, and `up * L >= N` by
    construction with the tail trimmed. The cost is a 1x1 convolution on a 136-long sequence.
    """

    def __init__(self, channels: int, n_beams: int, stride: int, width: int = MOTION_HEAD_WIDTH):
        super().__init__()
        self.n_beams, self.up = int(n_beams), int(stride)
        self.net = nn.Sequential(nn.Conv1d(int(channels), int(width), 1), nn.GELU(),
                                 nn.Conv1d(int(width), self.up, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b = feat.shape[0]
        logit = self.net(feat)                                   # (B, up, L)
        logit = logit.transpose(1, 2).reshape(b, -1)             # (B, L * up), beam-major
        if logit.shape[1] < self.n_beams:
            raise ValueError(f"the mask head unfolds to {logit.shape[1]} beams, fewer than the "
                             f"{self.n_beams} the scan has")
        return logit[:, :self.n_beams]


class MotionDvHead(nn.Module):
    """`h_dyn -> (Dv_x, Dv_y)` of the nearest opponent NOW, on the label's own scale.

    One hidden layer, like the grip / opponent / future heads, and a zero output layer for the same
    reason: the head's output feeds nothing else, so forward parity is free, and the zero makes the
    output layer's gradient nonzero from the first update while the hidden layer's is zero for
    exactly one.
    """

    def __init__(self, in_dim: int, width: int = MOTION_HEAD_WIDTH):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(int(in_dim), int(width)), nn.GELU(),
                                 nn.Linear(int(width), 2))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ------------------------------------------------------------------ losses
def mask_loss(logit: torch.Tensor, label: torch.Tensor, w: torch.Tensor,
              pos_weight_max: float = MASK_POS_WEIGHT_MAX):
    """(scalar, parts) for the per-beam opponent mask. Weighted BCE, because the classes are not
    remotely balanced: a car covers a few percent of the beams when it is there at all.

    The positive weight balances the two classes' total gradient mass: at logit 0 a positive pulls
    with `w/2` and a negative with `1/2`, so `w = (1 - rate) / rate` equalises them. That is the
    reciprocal of the positive rate, bounded only numerically (`MASK_POS_WEIGHT_MAX`). Reported
    beside the loss are the rate it saw, the weight it used, and the recall and precision at a 0.5
    threshold -- because a BCE that falls while the head answers "no car" everywhere is exactly the
    failure this label is prone to, and phase 1 watched it happen in both arms.
    """
    if logit.shape != label.shape:
        raise ValueError(f"mask logits {tuple(logit.shape)} and labels {tuple(label.shape)} differ")
    ww = w.to(logit.dtype)[:, None].expand_as(logit)
    denom = ww.sum().clamp_min(1.0)
    rate = (label.to(logit.dtype) * ww).sum() / denom
    # (1 - rate) / rate, not 1 / rate: the negatives' share is what the positives have to balance,
    # and the two differ by a percent at these rates but by everything as rate approaches 1.
    pw = ((1.0 - rate) / rate.clamp_min(1e-6)).clamp(max=float(pos_weight_max))
    per = F.binary_cross_entropy_with_logits(logit, label.to(logit.dtype), reduction="none",
                                             pos_weight=pw)
    loss = (per * ww).sum() / denom
    with torch.no_grad():
        pred = logit > 0
        pos = label > 0.5
        tp = (pred & pos).to(logit.dtype) * ww
        parts = {"mask_pos_rate": rate, "mask_pos_weight": pw,
                 "mask_recall": tp.sum() / (pos.to(logit.dtype) * ww).sum().clamp_min(1.0),
                 "mask_precision": tp.sum() / (pred.to(logit.dtype) * ww).sum().clamp_min(1.0)}
    return loss, parts


def dv_loss(pred: torch.Tensor, target: torch.Tensor, present: torch.Tensor, w: torch.Tensor):
    """(scalar, parts) for the nearest opponent's CURRENT relative velocity.

    Masked by presence for the same reason `future_loss` masks its opponent columns: there is no
    relative velocity to a car that is not in range, and scoring the head on the zeros those rows
    carry would teach it to answer "not moving" for an empty road. The target's variance under the
    same weights travels with the loss, so `1 - mse / var` is available without a second pass.
    """
    ww = (w * present).to(pred.dtype)
    denom = ww.sum().clamp_min(1.0)
    per = ((pred - target.to(pred.dtype)) ** 2).mean(1)
    loss = (per * ww).sum() / denom
    parts = {"dv_labelled_frac": ww.gt(0).to(pred.dtype).mean()}
    for i, key in enumerate(("dv_x", "dv_y")):
        e = ((pred[:, i] - target[:, i]) ** 2 * ww).sum() / denom
        mean = (target[:, i] * ww).sum() / denom
        parts[f"{key}_mse"] = e
        parts[f"{key}_var"] = ((target[:, i] - mean) ** 2 * ww).sum() / denom
    return loss, parts


def describe(meta: Optional[dict]) -> str:
    """One line for a log or a checkpoint header."""
    m = (meta or {}).get("motion")
    if not m:
        return "no motion memory"
    return (f"motion memory: h_dyn={m.get('hidden_size')} over a {m.get('channels')}-channel "
            f"encoder of {','.join(m.get('rows') or ())} (critic: {m.get('critic')})")
