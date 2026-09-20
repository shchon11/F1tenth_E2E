"""Privileged opponent tokens: the layout, the normalisers, and the trajectory sampler.

**This is an oracle.** Everything in this module describes information a LiDAR-only car cannot
have: the exact relative position of another car, its exact velocity, and where its own controller
intends to put it up to three quarters of a second from now. A checkpoint trained with any of it is
not deployable, and the loaders refuse to hand one to the exporter or to the ROS node
(`learn.model.load_checkpoint(allow_oracle=...)`).

Why it exists (`docs/research/oracle-planner-2026-09-15.md`): every attempt to make the recurrent
state *carry* the opponent's velocity failed to move a linear read-out above R^2 0.3
(`motion-memory-2026-09-14.md`), and PPO shows no pressure to learn it. Before any more perception
work, the other end of the question has to be answered: handed the opponent's state for free, does
the planner race any better? If it does not, no motion encoder can help, because the planner is
what would have to use the output.

## The block

One block of `opp_token_dim(mode)` columns is appended to the proprio vector, after every existing
key (`learn.obs.PROPRIO_KEYS`). It covers the nearest `OPP_TOKEN_CARS` opponents, ordered by body
distance, in the ego's *current* body frame. Modes are a ladder, each a strict prefix of the next:

| mode | per car | columns |
|---|---|---|
| `pos` | 3 | dx, dy, present |
| `posvel` | 5 | dx, dy, dvx, dvy, present |
| `future` | 13 | dx, dy, dvx, dvy, then (fx, fy) at +0.10 / +0.25 / +0.50 / +0.75 s, present |

The contract writes `pos` as "dx, dy, presence"; presence is placed **last** here so that a mode's
columns are a strict prefix of the next mode's. That makes the three arms' input layouts nested --
`pos` reads columns 0..2 of what `future` reads -- which is one column list to document instead of
three, and it means the extra columns of a wider arm are exactly the extra columns.

`present` is 1 when that opponent slot is filled by a car inside `overtake_range` and 0 otherwise;
every other column of the slot is multiplied by it, so an absent or out-of-range car contributes an
all-zero slot rather than a stale position. A race of two cars leaves the second slot permanently
zero, which is the same thing an empty road leaves.

Positions and velocities are divided by `OPP_TOKEN_SCALE`, which is `gym_env.PRIV_OPP_DIST_SCALE` --
the normaliser `future_labels` and the critic's privileged opponent columns already use, so a
read-out or a probe comparing them does not have to carry two conventions.

## The future columns

`fx, fy` are **not** ground truth: a batched simulator cannot peek ahead. They are where the
opponent's own controller intends to be, taken from the plan it is being driven by this step. Every
car of a race -- teacher-driven or a pool checkpoint -- goes through the same plan tracker, so there
is one object to read rather than one per kind of opponent, and for a teacher-driven car it already
carries that car's scheduled event (brake / stop / shift / defend / yield / line), its speed scale
and its follow cap, because all of those act on the action before the tracker sees it.

The tracker produces two trajectories and `OPP_TOKEN_PLAN_SOURCE` says which one is the label:

* `pred` -- `PlanTracker.last_pred`, the **iLQR forward rollout**: where the tracker's own kinematic
  model says the car will be under the commands it just chose. Its clock starts at the
  latency-compensated pose, so it is read with that car's calibrated command delay subtracted.
* `ref` -- `PlanTracker.last_ref`, the reference the tracker is *chasing*: the raceline target (or a
  pool policy's emitted plan) walked at the speed the car can actually have.

`pred` is the default because it was measured to be the better description of where the car went:
RMSE against the realised position 0.024 / 0.056 / 0.199 / 0.409 m at the four horizons, against
0.046 / 0.125 / 0.300 / 0.487 m for `ref` (`work/oracle-planner/work/label_validation.py`, 3 tracks,
race size 3, reactive opponents). Both are far better than what a `posvel` policy can extrapolate
for itself -- 0.022 / 0.137 / 0.508 / 1.046 m -- from 0.25 s out, which is what makes `future` a
different arm from `posvel` rather than a re-parametrisation of it. The number to quote is the RMSE:
a pooled R^2 on position divides by the variance of how far away the cars are and reads +0.99 for
almost anything, the same trap `motion-memory-2026-09-14.md` documents for its distance split.

`sample_body_traj` is the sampler: a body-frame trajectory on a uniform time grid, read at
arbitrary times, extended past its end along its terminal heading at its terminal speed (which is
what `mpc.reference` itself does for arc past the end of the plan).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch

#: The modes `--opp-token` accepts. `off` allocates nothing and is bit-identical to the env without
#: this file.
OPP_TOKEN_MODES = ("off", "pos", "posvel", "future")

#: How many opponents get a slot. Two: race size 3 is the recipe, and a third car is the one that
#: makes "the nearest opponent" an insufficient description of the traffic.
OPP_TOKEN_CARS = 2

#: [s] the horizons the `future` mode reports, from the instant the observation describes.
OPP_TOKEN_HORIZONS_S = (0.10, 0.25, 0.50, 0.75)

#: Which of the plan tracker's two trajectories the `future` columns report. See above: `pred` is
#: the iLQR rollout and is the measured better label; `ref` is the reference it chases. A constant
#: and not a flag: it is part of what the label *means*, so changing it makes a different experiment
#: rather than a different setting, and the arms of one comparison have to agree on it.
OPP_TOKEN_PLAN_SOURCE = "pred"

#: Metres (and m/s) per unit. The same scale `gym_env.PRIV_OPP_DIST_SCALE` puts the privileged
#: opponent columns and `future_labels` on; kept as its own name here so this module does not
#: import the env.
OPP_TOKEN_SCALE = 5.0


def validate_opp_token(mode) -> str:
    """The mode, normalised, or a ValueError naming what is accepted."""
    m = "off" if mode is None else str(mode)
    if m not in OPP_TOKEN_MODES:
        raise ValueError(f"unknown opponent token mode {mode!r}; known: {', '.join(OPP_TOKEN_MODES)}")
    return m


def opp_token_car_keys(mode) -> Tuple[str, ...]:
    """Column names of ONE opponent slot, in order."""
    m = validate_opp_token(mode)
    if m == "off":
        return ()
    keys = ["dx", "dy"]
    if m in ("posvel", "future"):
        keys += ["dvx", "dvy"]
    if m == "future":
        for h in OPP_TOKEN_HORIZONS_S:
            keys += [f"fx_{h:.2f}", f"fy_{h:.2f}"]
    return tuple(keys + ["present"])


def opp_token_keys(mode) -> Tuple[str, ...]:
    """Column names of the whole block, in order: slot 0 (nearest) then slot 1."""
    per = opp_token_car_keys(mode)
    return tuple(f"opp{i}_{k}" for i in range(OPP_TOKEN_CARS) for k in per)


def opp_token_car_dim(mode) -> int:
    return len(opp_token_car_keys(mode))


def opp_token_dim(mode) -> int:
    """Width of the proprio block this mode appends. 0 for `off`."""
    return OPP_TOKEN_CARS * opp_token_car_dim(mode)


def describe(mode) -> str:
    """One line for a log or a checkpoint header."""
    m = validate_opp_token(mode)
    if m == "off":
        return "no opponent token"
    extra = ("" if m != "future" else
             " + plan at " + "/".join(f"{h:.2f}" for h in OPP_TOKEN_HORIZONS_S) + " s")
    return (f"opponent token '{m}': {OPP_TOKEN_CARS} nearest cars x {opp_token_car_dim(m)} columns "
            f"= {opp_token_dim(m)} (ORACLE, never deployable){extra}")


def sample_body_traj(traj: torch.Tensor, dt: float, t: torch.Tensor) -> torch.Tensor:
    """Sample a body-frame trajectory at arbitrary times. -> (B, H, 3) = x, y, speed.

    `traj` is (B, T, 4) = x, y, heading, speed on the uniform grid 0, dt, ..., (T-1) dt, in the
    body frame of whatever pose it was anchored at. `t` is (B, H) seconds on that same clock,
    clamped at 0.

    Inside the grid the pose is linearly interpolated. Past the end it continues straight at the
    terminal heading and the terminal speed -- the same extension `mpc.reference` applies to arc
    past the end of the plan, and the only honest one available: the plan says nothing about what
    happens after it ends, so the label says the car keeps going the way the plan left it. The
    fraction of each horizon that is extension rather than plan is a property of the tracker's
    horizon (12 x 50 ms = 0.60 s), and `label_validation.py` measures what it costs.
    """
    if traj.dim() != 3 or traj.shape[2] != 4:
        raise ValueError(f"trajectory must be (B, T, 4) x/y/heading/speed, got {tuple(traj.shape)}")
    if t.dim() != 2 or t.shape[0] != traj.shape[0]:
        raise ValueError(f"times must be (B, H) on the trajectory's own batch, got {tuple(t.shape)} "
                         f"against {tuple(traj.shape)}")
    if not dt > 0:
        raise ValueError(f"trajectory step {dt} s must be positive")
    T = traj.shape[1]
    t_end = (T - 1) * dt
    tc = t.clamp(0.0, t_end)
    pos = tc / dt
    i0 = pos.floor().clamp(max=T - 2).long()
    w = (pos - i0.to(traj.dtype)).unsqueeze(-1)
    g = lambda idx: torch.gather(traj, 1, idx.unsqueeze(-1).expand(-1, -1, 4))
    p = g(i0) * (1 - w) + g(i0 + 1) * w                       # (B, H, 4)
    over = (t - t_end).clamp(min=0.0)                          # seconds past the end
    psi_e, v_e = traj[:, -1, 2:3], traj[:, -1, 3:4]
    x = p[:, :, 0] + over * v_e * torch.cos(psi_e)
    y = p[:, :, 1] + over * v_e * torch.sin(psi_e)
    return torch.stack([x, y, p[:, :, 3]], 2)


def to_world(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Body-frame (B, H, 2) points anchored at pose (B, 3) x/y/yaw -> world (B, H, 2)."""
    c, s = torch.cos(pose[:, 2])[:, None], torch.sin(pose[:, 2])[:, None]
    x, y = points[:, :, 0], points[:, :, 1]
    return torch.stack([pose[:, 0:1] + x * c - y * s, pose[:, 1:2] + x * s + y * c], 2)


def to_ego(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """World (B, H, 2) points -> the body frame of pose (B, 3) x/y/yaw. Inverse of `to_world`."""
    c, s = torch.cos(pose[:, 2])[:, None], torch.sin(pose[:, 2])[:, None]
    dx = points[:, :, 0] - pose[:, 0:1]
    dy = points[:, :, 1] - pose[:, 1:2]
    return torch.stack([dx * c + dy * s, -dx * s + dy * c], 2)


def straight_traj(state: torch.Tensor, steps: int, dt: float,
                  t0: "torch.Tensor | float" = 0.0) -> torch.Tensor:
    """A (B, steps+1, 4) body-frame trajectory that holds the car's current velocity.

    The fallback for a row whose plan does not exist or does not belong to it any more: the first
    observation of an episode, and the step a car respawns on. Straight ahead at the body-frame
    longitudinal speed, which is what "no plan has been issued yet" can honestly be turned into.
    `state` is the simulator's (B, >=5) ground truth; only the longitudinal speed column is read.

    `t0` (a scalar or (B,)) shifts the trajectory's own clock forward, so a sampler that subtracts
    the same offset -- as it must for `pred`, whose rollout starts at the latency-compensated pose
    -- lands on the same instant. Without it the stand-in would sit one command latency behind the
    label it stands in for, on exactly the rows nobody looks at.
    """
    v = state[:, 3].abs()
    ts = torch.arange(steps + 1, device=state.device, dtype=state.dtype)[None] * dt
    off = t0 if torch.is_tensor(t0) else torch.as_tensor(float(t0), device=state.device,
                                                         dtype=state.dtype)
    ts = ts + (off[:, None] if off.dim() == 1 else off)
    x = v[:, None] * ts
    z = torch.zeros_like(x)
    return torch.stack([x, z, z, v[:, None].expand_as(x)], 2)
