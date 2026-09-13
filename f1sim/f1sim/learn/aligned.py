"""The ego-motion-aligned temporal scan: a residual that static geometry cancels out of.

Why (`docs/research/motion-memory-2026-09-14.md`): the policy's LiDAR stack is six frames of raw
ranges. Between two of them almost everything that moves is the *ego* -- a corridor sweeping past at
9 m/s -- and the other car is a handful of beams whose apparent motion is the sum of its own and the
ego's. Nothing in PPO's objective rewards separating the two, and `probe_hidden --current` measures
what that costs: the recurrent state carries the ego's own motion and not the opponent's relative
velocity. The remedy tried here is inductive bias, not capacity: subtract the ego's motion from the
observation before the network sees it, so that the part of the scan that still changes is the part
that moved by itself.

    R_t(theta) = r_t(theta) - r~_{t-k}(theta),   r~_{t-k} = rasterise(T_{t<-t-k} P_{t-k})

i.e. NOT `r_t - r_{t-k}`: the past *point cloud* is transformed into the current LiDAR frame and
re-rasterised onto the current angular bins, so a bin is compared against whatever surface actually
ends up along that ray rather than against whatever the same beam index used to see.

`warp` takes the scan from k control steps ago into the *current* sensor frame using only what the
car can measure about its own motion -- **wheel speed, yaw rate and the IMU's roll/pitch**. No pose,
no map, no odometry integration beyond those k steps, nothing privileged: this channel runs on the
car, and `f1sim_ros/policy_node.py` feeds it from the same three sensors it already reads.

## What the warp does, in order

1. Each return of the old scan becomes a 3D point in the old sensor's own yaw-aligned frame:
   `r * Ry(pitch) Rx(roll) (cos a, sin a, 0)`. That is the simulator's own beam geometry
   (`lidar.Lidar.rays`), so a beam that was tilted into the floor is placed where it actually hit.
2. The point is carried forward by the ego's planar motion over those k steps, composed from k
   per-step arcs (`step_increment`, `compose_increments`). Each arc is the exact constant-(v, omega)
   solution, from the trapezoidal average of the two endpoint measurements -- both of which are
   already in hand when the step is taken, so nothing here is acausal.
3. It is expressed in the *current* sensor plane by the inverse of the current tilt. A point whose
   out-of-plane offset then exceeds `z_tol` is dropped: the current scan cannot see it, so there is
   nothing to compare it against and pretending otherwise is how tilt turns into a false detection.
4. The surviving points are scattered into the current beam grid by bearing, nearest beam, keeping
   the closest -- the same "nearest return wins" the sensor itself applies.

Bins that receive no point are **unknown**, and the residual there is exactly 0. That distinction is
the whole safety of the channel: "I cannot tell" and "nothing moved" must not be the same number,
because the second one is a claim.

## The three channels

The addendum's channel set, one row each on the scan's channel axis, in this order:

* **`aligned`** -- the soft-thresholded residual `sign(R) * max(|R| - tau, 0)`, normalised by
  `range_max` like every other scan row. Soft, not hard: a shrinkage keeps the size of a real
  residual (a car 0.5 m out of place stays 0.5 m out of place, minus tau) where a hard gate would
  hand the network a step function at the threshold and nothing below it.
* **`aligned_prev`** -- the warped previous range itself, so the network can see what was predicted
  and not only how wrong it was. 1.0 (the scan encoding's "no return") where the warp predicted
  nothing.
* **`aligned_valid`** -- the warp-valid / visibility mask: 1 where a warped point reached this bin
  AND the current beam returned, 0 otherwise. This is what keeps "I cannot tell" from being spelled
  the same way as "nothing moved". The one-channel variant (`aligned` alone, invalid -> 0) is a
  legitimate ablation; the mask is the research-clean form.

The fourth channel of the addendum's set, the current range, is already the newest frame of the
stack and is not duplicated.

## Where tau comes from

The tilt is measured, not known. The plant has 1.7 deg/g of roll, a 1 deg rms floor wobble with a
0.4 s time constant (`docs/real_data_calibration.md` 6.1a), and a 2D scanner reports a 3D world, so
`Ry(pitch) Rx(roll)` is an approximation of the scan plane and never exactly it.

`tau` is **fixed before training** at 2-3 sigma of the residual measured on static scenes, and the
number is stated in `docs/research/motion-memory-2026-09-14.md` with the run that produced it
(`python -m f1sim.learn.aligned_floor`). It is not tuned on a result. `tau_rel` adds a
range-proportional term and is **0 by default**: the addendum asks for one number and gets one; the
knob exists because the warp's bearing error costs range in proportion to it, and the floor table
reports what that would buy.

On top of it the original contract's **two-frame consistency test**: a residual survives only if the
previous control step also had one of the same sign above `tau` within `consist_beams` of the same
bearing. A car is there in both frames a few beams apart; a tilt excursion or a grazing beam is not.
`consist_beams = 0` turns it off, and the floor table reports both.

## What it is not

* Not a tracker, not a detector, not a filter with state beyond the k scans and the k motion
  increments it needs. It is a function of the observation the policy already has.
* Not motion compensation of the *whole* observation: the six raw frames are untouched and the
  residual is one more channel beside them, appended after every column the original network had
  and zero-initialised in the first convolution, exactly as `memory` and `edges` are.
* Not tilt-exact. A 2D scan plane that rolls sees a different slice of the world, and no rotation of
  a 2D range image recovers the slice it did not measure. `z_tol` is the honest statement of that:
  points that the two planes do not share are dropped.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

#: Control steps of lag. 4 steps at 40 Hz is 100 ms -- long enough that a car closing at 3 m/s has
#: moved 0.3 m (tens of beams at racing distances, far outside the gate's bearing tolerance) and
#: short enough that the composed ego motion is a small extrapolation of two measured rates.
ALIGNED_K = 4

#: Soft threshold on the residual, metres: `sign(R) * max(|R| - tau, 0)`. Fixed BEFORE training at
#: 2-3 sigma of the static-world floor measured on the real bags (`learn.aligned_floor`, and the
#: number is stated in the research note). Not tuned on a result.
ALIGNED_TAU = 0.10

#: Range-proportional addition to tau. 0 by default: the addendum asks for one number. The knob
#: exists because a bearing error of e radians costs `r * e` of range on an oblique surface, and the
#: floor table reports what a nonzero value would buy.
ALIGNED_TAU_REL = 0.0

#: Bearing tolerance of the two-frame consistency test, in beams. The previous step's residual is
#: dilated by this much before its sign is required to agree, because the thing being detected has
#: moved between the two frames. 8 beams is 2 deg at this scanner's 0.25 deg spacing.
ALIGNED_CONSIST_BEAMS = 8

#: Out-of-plane tolerance [m]. A warped point further than this from the current scan plane is
#: dropped as not comparable.
ALIGNED_Z_TOL = 0.05

#: Beams a one-bin hole in the warped scan may be filled from. The scatter is a resampling of 1081
#: bearings onto 1081 bins and leaves holes wherever the old beams fanned out; a hole is an artefact
#: of the resampling, not a fact about the world. 0 disables the fill.
ALIGNED_GAP_FILL = 1

#: The rows this channel contributes, in the order `obs.SCAN_CHANNELS` lays them out. Named
#: separately so a caller can ask for the residual alone (the addendum's one-channel ablation) or
#: for the research-clean set, and so each row is still one entry of `SCAN_CHANNELS` with one
#: fixed column index.
ALIGNED_ROWS = ("aligned", "aligned_prev", "aligned_valid")

#: The scanner's field of view [rad]. Matches `params.LidarConfig.fov` and the ROS node's
#: `LIDAR_FOV`; recorded in the spec so a checkpoint says which window its channel was built for.
ALIGNED_FOV = 4.71238898

#: Bearing uncertainty of the warp, in beams: how far along the beam axis the current range is
#: allowed to match the warped scan before the difference counts as a residual. See `residual`.
ALIGNED_TOL_BEAMS = 4

#: "No point landed in this bin", in multiples of the scan's own range maximum. Finite on purpose --
#: see `warp_scan`.
MISS_RANGE = 8.0


def aligned_spec(k: int = ALIGNED_K, tau: float = ALIGNED_TAU, tau_rel: float = ALIGNED_TAU_REL,
                 consist_beams: int = ALIGNED_CONSIST_BEAMS, z_tol: float = ALIGNED_Z_TOL,
                 gap_fill: int = ALIGNED_GAP_FILL, fov: float = ALIGNED_FOV,
                 tol_beams: int = ALIGNED_TOL_BEAMS) -> dict:
    """The `meta["scan_channels"]["aligned"]` block, validated.

    Idempotent -- `aligned_spec(**aligned_spec())` is the same dict -- because a checkpoint's
    recorded block is handed straight back, and a validator that refused its own output would make
    "load what you saved" the one path nobody tested.
    """
    k, consist_beams, gap_fill, tol_beams = (int(k), int(consist_beams), int(gap_fill),
                                             int(tol_beams))
    tau, tau_rel, z_tol, fov = float(tau), float(tau_rel), float(z_tol), float(fov)
    if k < 1:
        raise ValueError(f"aligned k {k} must be >= 1 control step")
    if tau < 0 or tau_rel < 0:
        raise ValueError(f"aligned threshold ({tau} m, {tau_rel} /m) must be non-negative")
    if consist_beams < 0:
        raise ValueError(f"aligned consist_beams {consist_beams} must be >= 0")
    if gap_fill < 0:
        raise ValueError(f"aligned gap_fill {gap_fill} must be >= 0")
    if tol_beams < 0:
        raise ValueError(f"aligned tol_beams {tol_beams} must be >= 0")
    if not z_tol > 0:
        raise ValueError(f"aligned z_tol {z_tol} m must be positive")
    if not 0 < fov <= 2 * math.pi:
        raise ValueError(f"aligned fov {fov} rad must be in (0, 2 pi]")
    return {"k": k, "tau": tau, "tau_rel": tau_rel, "consist_beams": consist_beams,
            "z_tol": z_tol, "gap_fill": gap_fill, "fov": fov, "tol_beams": tol_beams}


# ------------------------------------------------------------------ ego motion over k steps
def step_increment(v0: torch.Tensor, w0: torch.Tensor, v1: torch.Tensor, w1: torch.Tensor,
                   dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """(translation (B,2) in the OLD frame, yaw change (B,)) of one control step.

    The exact constant-(v, omega) arc, from the trapezoidal average of the two endpoint
    measurements. Two properties are worth naming:

    * it is causal -- both endpoints are measured by the time the step is complete, so nothing is
      predicted;
    * it is the unicycle, not the bicycle. The measured yaw rate already contains whatever the
      steering, the slip angles and the friction did, so re-deriving it from a steering angle and a
      wheelbase would replace a measurement with a model. The lateral velocity the car actually has
      in a slide is the one thing left out, and it is not observable from these three sensors.
    """
    v, w = 0.5 * (v0 + v1), 0.5 * (w0 + w1)
    dyaw = w * dt
    small = dyaw.abs() < 1e-6
    # v/w * sin(w dt) and v/w * (1 - cos(w dt)), with the straight-line limit where w -> 0. The
    # `where` on the denominator, not only on the result: a division by zero produces a NaN whose
    # gradient and whose `where` both propagate it.
    wz = torch.where(small, torch.ones_like(w), w)
    dx = torch.where(small, v * dt, v * torch.sin(dyaw) / wz)
    dy = torch.where(small, torch.zeros_like(v), v * (1.0 - torch.cos(dyaw)) / wz)
    return torch.stack([dx, dy], -1), dyaw


def compose_increments(increments: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    """(A (B,2,2), b (B,2)) mapping a point's coordinates in the OLDEST frame to the newest.

    `increments` is oldest-first: `(p, dyaw)` for the step from frame j to frame j+1, where `p` is
    where the sensor moved to, in frame j. A world point at `x` in frame j is at
    `R(-dyaw) (x - p)` in frame j+1, and the composition of those is what carries a scan forward.
    """
    if not increments:
        raise ValueError("compose_increments needs at least one step")
    p0 = increments[0][0]
    B = p0.shape[0]
    a = torch.eye(2, device=p0.device, dtype=p0.dtype).expand(B, 2, 2).contiguous()
    b = torch.zeros(B, 2, device=p0.device, dtype=p0.dtype)
    for p, dyaw in increments:
        c, s = torch.cos(dyaw), torch.sin(dyaw)
        # R(-dyaw)
        r = torch.stack([torch.stack([c, s], -1), torch.stack([-s, c], -1)], -2)   # (B,2,2)
        a = r @ a
        b = (r @ (b - p).unsqueeze(-1)).squeeze(-1)
    return a, b


def tilt_matrix(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """(B,3,3) `Ry(pitch) Rx(roll)`: the sensor's own frame into the yaw-aligned level frame.

    The same composition `lidar.Lidar.rays` builds its beams with, in the same sign convention
    (+pitch is nose down, +roll is right side down), so a beam this function places is where the
    simulator's ray casting sent it.
    """
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    z, o = torch.zeros_like(cr), torch.ones_like(cr)
    rx = torch.stack([torch.stack([o, z, z], -1),
                      torch.stack([z, cr, -sr], -1),
                      torch.stack([z, sr, cr], -1)], -2)
    ry = torch.stack([torch.stack([cp, z, sp], -1),
                      torch.stack([z, o, z], -1),
                      torch.stack([-sp, z, cp], -1)], -2)
    return ry @ rx


def beam_angles(n_beams: int, fov: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """(N,) beam bearings, the same `linspace(-fov/2, fov/2, N)` the simulator and `urg_node` use."""
    return torch.linspace(-0.5 * fov, 0.5 * fov, int(n_beams), device=device, dtype=dtype)


# ------------------------------------------------------------------ the warp
def warp_scan(old_r: torch.Tensor, old_att: torch.Tensor, new_att: torch.Tensor,
              a: torch.Tensor, b: torch.Tensor, angles: torch.Tensor, range_max: float,
              z_tol: float = ALIGNED_Z_TOL, gap_fill: int = ALIGNED_GAP_FILL):
    """(warped range (B,N) [m], known (B,N) bool, bound (B,N) bool): the old scan from here.

    `old_r` is in metres, `old_att` / `new_att` are (B,2) roll/pitch in radians, `(a, b)` is the
    planar transform `compose_increments` returned, `angles` (N,) are the beam bearings.

    Three outputs because a range image carries two kinds of statement and they must not be warped
    into one:

    * a **return** at r says "there is a surface at r, that way". Warped, it stays an equality, and
      `bound` is False.
    * a **no return** says "nothing out to `range_max`, that way" -- a lower bound, not a point at
      `range_max`. Warped, it is still a lower bound (moving 0.9 m toward free space leaves at least
      `r - 0.9` of it), and `bound` is True. `residual` then only counts something that appeared
      CLOSER than it, never something further away. Treating it as an equality is a systematic
      artefact worth naming: on a 22 m straight every far beam reads the 10 m clamp, and a car
      driving 0.9 m up it would report +0.9 m of "motion" on every one of them.
    * a bin no point reached at all says nothing: `known` is False and the residual there is 0.

    Unknown bins arise where the scan rotated out of the window, where `z_tol` dropped a point the
    two scan planes do not share, and where the resampling simply left a hole.
    """
    if old_r.dim() != 2:
        raise ValueError(f"old_r must be (B, N), got {tuple(old_r.shape)}")
    b_, n = old_r.shape
    if angles.shape[0] != n:
        raise ValueError(f"angles must be ({n},), got {tuple(angles.shape)}")
    dtype = old_r.dtype
    ca, sa = torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)
    r = old_r.clamp(0.0, range_max)
    far = old_r >= range_max - 1e-4
    q = torch.stack([r * ca[None], r * sa[None], torch.zeros_like(r)], -1)          # (B,N,3)
    r_old = tilt_matrix(old_att[:, 0], old_att[:, 1]).to(dtype)                     # (B,3,3)
    p = torch.einsum("bij,bnj->bni", r_old, q)                                      # level frame, t-k
    xy = torch.einsum("bij,bnj->bni", a.to(dtype), p[..., :2]) + b.to(dtype)[:, None, :]
    p = torch.cat([xy, p[..., 2:]], -1)                                             # level frame, t
    r_new = tilt_matrix(new_att[:, 0], new_att[:, 1]).to(dtype)
    p = torch.einsum("bji,bnj->bni", r_new, p)                                      # transpose = inverse
    rho = torch.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2).clamp(1e-6, range_max)
    phi = torch.atan2(p[..., 1], p[..., 0])
    step = float(angles[1] - angles[0])
    idx = torch.round((phi - angles[0]) / step).long()
    ok = (idx >= 0) & (idx < n) & (p[..., 2].abs() <= z_tol)
    # A large FINITE sentinel, not an infinity: `max_pool1d` is the gap fill below and pooling over
    # infinities comes back as the dtype's largest finite value on some backends, which then reads
    # as a known bin holding an absurd range. `miss` is well clear of any range the scanner can
    # report and `< 0.5 * miss` is the one test for "did anything land here".
    miss = MISS_RANGE * max(1.0, float(range_max))
    full = torch.full_like(rho, miss)
    scatter = lambda keep: torch.full((b_, n), miss, device=old_r.device, dtype=dtype).scatter_reduce_(
        1, idx.clamp(0, n - 1), torch.where(ok & keep, rho, full), reduce="amin", include_self=True)
    near_r = scatter(~far)                       # bins a real return landed in
    far_r = scatter(far)                         # bins a free-space bound landed in
    if gap_fill > 0:
        # A hole is a bin the resampling skipped, between two bins it did not. Filling it from the
        # nearer of its neighbours keeps the warp a surface rather than a dotted line, and cannot
        # invent a return closer than one the old scan actually had.
        pool = lambda x: -F.max_pool1d(-x.unsqueeze(1), kernel_size=2 * gap_fill + 1, stride=1,
                                       padding=gap_fill).squeeze(1)
        near_r, far_r = torch.minimum(near_r, pool(near_r)), torch.minimum(far_r, pool(far_r))
    out = torch.minimum(near_r, far_r)           # nearest statement wins, as the sensor's does
    known = out < 0.5 * miss
    bound = known & (far_r < near_r)
    return torch.where(known, out, torch.zeros_like(out)), known, bound


def residual(now_r: torch.Tensor, warped_r: torch.Tensor, known: torch.Tensor,
             bound: Optional[torch.Tensor] = None, tol_beams: int = 0,
             now_known: Optional[torch.Tensor] = None):
    """(`L_t - warp(L_{t-k})` in metres, the mask of beams it could be computed for).

    Signed on purpose: a negative residual is something that came closer than the static world
    predicts (a car cutting in), a positive one is something that left (a car pulling away, or a
    return the old scan had and this one does not). A magnitude would throw away which.

    `bound` marks bins whose prediction is free space rather than a surface (`warp_scan`). There the
    residual is one-sided: something closer than the bound is news, something further away is the
    bound being what it always was.

    `now_known` is where the CURRENT scan actually returned. A beam that did not is not evidence
    that what the warp predicted has gone -- it is the sensor saying nothing, and on this hardware it
    says nothing often and in runs (0.4-32 % of beams per recording, median run 6-48 beams long,
    `docs/real_data_calibration.md` 2.4). Scored as a range of `range_max` it produces a residual the
    size of the whole scan: the largest false positives this channel can make, and the ones the
    consistency test is least able to reject, because a surface that flickers flickers in both
    frames. Where the current beam has no return the residual is 0 and the beam is unknown.

    `tol_beams` is the warp's own **bearing uncertainty**, and with it the comparison is against the
    warped scan's envelope over that window rather than against one bin:

        resid = now - lo   where now < lo = min warp over the window
                now - hi   where now > hi = max warp over the window (+inf if the window is bounded
                           only from below)
                0          where the current range is inside [lo, hi]

    i.e. zero wherever the current range is consistent with the static world anywhere the warp could
    plausibly have put it. It is not a smoothing: a range outside the envelope keeps its full signed
    distance from it, so a car does not get quietly shrunk.

    Why there has to be one, with the arithmetic: over k = 4 steps at 9 m/s the ego moves 0.9 m, so
    a 1 % speed error is 9 mm; a 0.05 rad/s yaw-rate error is 5 mrad, about one beam; and a 1 deg
    error in the measured tilt -- the plant's floor wobble, exactly (`docs/real_data_calibration.md`
    6.1a) -- moves an off-axis beam's footprint by up to a degree, four beams at this scanner's
    0.25 deg spacing. Against a wall seen at a shallow angle a bearing error of that size is metres
    of range, which is why the ungated residual's p99 is metres while its median is centimetres.
    `tol_beams = 0` is the literal one-bin difference, and the noise-floor table reports both.
    """
    if bound is None:
        bound = torch.zeros_like(known)
    if now_known is not None:
        known = known & now_known
    if int(tol_beams) <= 0:
        r = now_r - warped_r
        r = torch.where(bound, r.clamp(max=0.0), r)
        return torch.where(known, r, torch.zeros_like(r)), known
    kw = 2 * int(tol_beams) + 1
    pool = lambda x: F.max_pool1d(x.unsqueeze(1), kernel_size=kw, stride=1,
                                  padding=int(tol_beams)).squeeze(1)
    big = torch.full_like(warped_r, float(MISS_RANGE) * 1e3)
    lo = -pool(-torch.where(known, warped_r, big))
    surf = known & ~bound
    hi = pool(torch.where(surf, warped_r, torch.zeros_like(warped_r)))
    any_known = pool(known.to(warped_r.dtype)) > 0
    # A window that contains a free-space bound has no upper edge: beyond it the world may be empty
    # for any distance, so "further away than predicted" is not a statement there.
    capped = (pool(surf.to(warped_r.dtype)) > 0) & ~(pool(bound.to(warped_r.dtype)) > 0)
    r = torch.where(now_r < lo, now_r - lo,
                    torch.where(capped & (now_r > hi), now_r - hi, torch.zeros_like(now_r)))
    if now_known is not None:
        any_known = any_known & now_known
    return torch.where(any_known, r, torch.zeros_like(r)), any_known


def soft_threshold(raw: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """`sign(R) * max(|R| - tau, 0)`, the addendum's shrinkage.

    Soft rather than hard because the network reads the value and not only its presence: a hard gate
    would hand it a step discontinuity at tau and no gradient of size below it, while the shrinkage
    keeps a real residual's magnitude (minus tau) and takes the floor down to exactly zero.
    """
    return torch.sign(raw) * (raw.abs() - tau).clamp_min(0.0)


def gate(raw: torch.Tensor, now_r: torch.Tensor, prev_pos: Optional[torch.Tensor],
         prev_neg: Optional[torch.Tensor], tau: float, tau_rel: float, consist_beams: int):
    """(shrunk residual, this step's positive mask, this step's negative mask).

    The shrinkage is `soft_threshold`; on top of it the two-frame consistency test keeps only what
    the previous control step also saw with the same sign above tau, within `consist_beams` of the
    same bearing (`consist_beams = 0` turns the test off and the shrinkage stands alone).

    `prev_pos` / `prev_neg` are the previous control step's *pre-consistency* strong masks, or None
    on the first step after a reset, where nothing has a predecessor and the output is all zero.
    They are dilated before the sign has to agree, because whatever is being detected has moved
    between the two frames; the dilation is what makes the test a test of "the same thing is still
    there" rather than "the same beam".
    """
    t = tau + tau_rel * now_r
    pos = raw > t
    neg = raw < -t
    shrunk = soft_threshold(raw, t)
    if consist_beams <= 0:
        return shrunk, pos, neg
    if prev_pos is None or prev_neg is None:
        return torch.zeros_like(shrunk), pos, neg
    dil = lambda m: F.max_pool1d(m.to(raw.dtype).unsqueeze(1), kernel_size=2 * consist_beams + 1,
                                 stride=1, padding=consist_beams).squeeze(1) > 0
    keep = (pos & dil(prev_pos)) | (neg & dil(prev_neg))
    return torch.where(keep, shrunk, torch.zeros_like(shrunk)), pos, neg


# ------------------------------------------------------------------ the channel
class AlignedScan:
    """The aligned channels for one inference path, with its k scans and k motion increments.

    `__call__(scan_now, motion)` takes the newest normalised scan (B, N) and the four observed
    quantities the warp needs -- `motion` (B, 4) = speed [m/s], yaw rate [rad/s], roll [rad],
    pitch [rad], all SI, all from `obs.motion_from_proprio` so that the simulator and the car build
    them the same way -- and returns a dict of (B, N) rows keyed by `ALIGNED_ROWS`, all
    **normalised by `range_max` like every other scan channel** so the network's input columns are
    commensurate:

    * `aligned` -- the soft-thresholded residual, 0 where the warp is not valid;
    * `aligned_prev` -- the warped previous range, 1.0 (no return) where it is not valid;
    * `aligned_valid` -- the warp-valid / visibility mask, 1 or 0.

    Until an episode has produced `k + 1` scans the output is exactly zero: there is no scan to warp
    and no motion history to warp it with, and a partial answer there would be the channel's least
    reliable output arriving at the moment the policy has least other context.

    The state is episode state and is cleared exactly where the recurrent hidden state and the
    occupancy channel are (`learn.memory.PolicyRuntime`).
    """

    def __init__(self, n_beams: int, batch: int, spec: Optional[dict] = None, device="cpu",
                 dt: float = 0.025, range_max: float = 10.0, dtype=torch.float32):
        s = aligned_spec(**(spec or {}))
        self.k = s["k"]
        self.tau, self.tau_rel = s["tau"], s["tau_rel"]
        self.consist_beams, self.z_tol = s["consist_beams"], s["z_tol"]
        self.gap_fill, self.fov, self.tol_beams = s["gap_fill"], s["fov"], s["tol_beams"]
        self.spec = s
        self.n_beams, self.batch = int(n_beams), int(batch)
        self.device, self.dtype = torch.device(device), dtype
        self.dt, self.range_max = float(dt), float(range_max)
        self.angles = beam_angles(self.n_beams, self.fov, device=self.device, dtype=dtype)
        self.reset()

    def reset(self, done=None) -> None:
        """Clear every row (`done=None`) or the rows whose episode ended.

        A per-row clear zeroes that row's history and puts its step counter back to 0, so it spends
        the next `k` steps emitting zeros exactly as a fresh episode does. The buffers keep their
        shape; only the rows named move.
        """
        shape = (self.batch, self.n_beams)
        if done is None:
            self.scans = torch.ones(self.k + 1, *shape, device=self.device, dtype=self.dtype)
            self.att = torch.zeros(self.k + 1, self.batch, 2, device=self.device, dtype=self.dtype)
            self.vw = torch.zeros(self.batch, 2, device=self.device, dtype=self.dtype)
            self.inc_p = torch.zeros(self.k, self.batch, 2, device=self.device, dtype=self.dtype)
            self.inc_yaw = torch.zeros(self.k, self.batch, device=self.device, dtype=self.dtype)
            self.prev_pos = torch.zeros(*shape, device=self.device, dtype=torch.bool)
            self.prev_neg = torch.zeros(*shape, device=self.device, dtype=torch.bool)
            self.seen = torch.zeros(self.batch, device=self.device, dtype=torch.long)
            self.have_prev = torch.zeros(self.batch, device=self.device, dtype=torch.bool)
            self._raw = torch.zeros(*shape, device=self.device, dtype=self.dtype)
            self._known = torch.zeros(*shape, device=self.device, dtype=torch.bool)
            return
        d = done if torch.is_tensor(done) else torch.as_tensor(done, device=self.device)
        d = d.to(self.device)
        d = d.to(torch.bool) if d.dtype == torch.bool else d.to(self.dtype) > 0.5
        if d.dim() != 1 or d.shape[0] != self.batch:
            raise ValueError(f"episode-boundary mask must be ({self.batch},), got {tuple(d.shape)}")
        keep = (~d).to(self.dtype)
        self.scans = self.scans * keep[None, :, None] + (1.0 - keep)[None, :, None]
        self.att = self.att * keep[None, :, None]
        self.vw = self.vw * keep[:, None]
        self.inc_p = self.inc_p * keep[None, :, None]
        self.inc_yaw = self.inc_yaw * keep[None, :]
        self.prev_pos = self.prev_pos & ~d[:, None]
        self.prev_neg = self.prev_neg & ~d[:, None]
        self.seen = torch.where(d, torch.zeros_like(self.seen), self.seen)
        self.have_prev = self.have_prev & ~d

    def _advance(self, scan_now: torch.Tensor, motion: torch.Tensor, dt: Optional[float] = None):
        """Push this step's scan, attitude and motion increment into the ring buffers."""
        v, w = motion[:, 0], motion[:, 1]
        p, dyaw = step_increment(self.vw[:, 0], self.vw[:, 1], v, w,
                                 self.dt if dt is None else float(dt))
        self.vw = torch.stack([v, w], -1)
        self.inc_p = torch.roll(self.inc_p, -1, 0); self.inc_p[-1] = p
        self.inc_yaw = torch.roll(self.inc_yaw, -1, 0); self.inc_yaw[-1] = dyaw
        self.scans = torch.roll(self.scans, -1, 0); self.scans[-1] = scan_now
        self.att = torch.roll(self.att, -1, 0); self.att[-1] = motion[:, 2:4]
        self.seen = self.seen + 1

    def __call__(self, scan_now: torch.Tensor, motion: torch.Tensor,
                 dt: Optional[float] = None) -> dict:
        """The aligned rows for this control step, keyed by `ALIGNED_ROWS`.

        `dt` overrides the nominal control period, for an offline replay (a rosbag) whose scans do
        not arrive at exactly 40 Hz; the on-policy paths never pass it and get the period the
        channel was built with."""
        if scan_now.dim() != 2 or scan_now.shape != (self.batch, self.n_beams):
            raise ValueError(f"scan must be ({self.batch}, {self.n_beams}), got "
                             f"{tuple(scan_now.shape)}")
        if motion.dim() != 2 or motion.shape != (self.batch, 4):
            raise ValueError(f"motion must be ({self.batch}, 4) = speed, yaw rate, roll, pitch "
                             f"in SI units, got {tuple(motion.shape)}")
        scan_now = scan_now.to(self.dtype)
        motion = motion.to(self.dtype)
        self._advance(scan_now, motion, dt)
        a, b = compose_increments([(self.inc_p[i], self.inc_yaw[i]) for i in range(self.k)])
        warped, known, bound = warp_scan(self.scans[0] * self.range_max, self.att[0], self.att[-1],
                                         a, b, self.angles, self.range_max, self.z_tol,
                                         self.gap_fill)
        now_m = scan_now * self.range_max
        raw, known = residual(now_m, warped, known, bound, self.tol_beams,
                              now_known=now_m < self.range_max - 1e-4)
        #: The layers `parts` reports, kept from the last call rather than recomputed: a second
        #: implementation of the same arithmetic is a second thing that can be wrong.
        self._raw, self._known = raw / self.range_max, known
        out, pos, neg = gate(raw, now_m, self.prev_pos if self.have_prev.any() else None,
                             self.prev_neg if self.have_prev.any() else None,
                             self.tau, self.tau_rel, self.consist_beams)
        # Rows whose buffers are not yet full have warped a scan that belongs to a previous episode
        # (or to the ones() the reset wrote), so their residual is meaningless and is zeroed. The
        # masks are still recorded: the row is building the history the consistency test will use.
        ready = (self.seen > self.k)[:, None]
        valid = known & ready
        out = torch.where(valid, out, torch.zeros_like(out))
        self.prev_pos, self.prev_neg = pos & ready, neg & ready
        self.have_prev = self.have_prev | ready[:, 0]
        self._raw = torch.where(ready, self._raw, torch.zeros_like(self._raw))
        self._known = valid
        return {"aligned": out / self.range_max,
                # 1.0 is the scan encoding's "no return", which is the honest reading of a bin the
                # warp could not predict: nothing is asserted to be there. The mask says which is
                # which, and the one-channel ablation is the residual alone.
                "aligned_prev": torch.where(valid, warped / self.range_max,
                                            torch.ones_like(warped)),
                "aligned_valid": valid.to(self.dtype)}

    def parts(self, scan_now: torch.Tensor, motion: torch.Tensor, dt: Optional[float] = None):
        """(gated residual, raw residual, known mask, ready mask), all in NORMALISED scan units.

        The gated residual is what `__call__` returns; the other three are what the noise-floor
        measurement needs -- how much residual the warp leaves before either gate, where it had a
        prediction at all, and which rows have enough history to be scored. Advancing the state
        once and returning every layer is what keeps the measured floor a measurement OF the
        channel rather than of a re-implementation beside it.
        """
        before = self.seen.clone()
        rows = self(scan_now, motion, dt)
        return rows["aligned"], self._raw, self._known, (before + 1 > self.k)

    def preview(self, scan_now: torch.Tensor, motion: torch.Tensor, index=None,
                dt: Optional[float] = None) -> dict:
        """The rows this channel WOULD produce, without advancing any of its state.

        For a terminal observation, which is scored (the truncation bootstrap reads its value) and
        never acted on. Implemented by saving and restoring the buffers rather than by a second copy
        of the arithmetic, so the previewed value cannot drift from the acted one.
        """
        keep = (self.scans, self.att, self.vw, self.inc_p, self.inc_yaw, self.prev_pos,
                self.prev_neg, self.seen, self.have_prev, self.batch)
        try:
            if index is not None:
                idx = index if torch.is_tensor(index) else torch.as_tensor(index, device=self.device)
                self.scans = self.scans[:, idx]; self.att = self.att[:, idx]
                self.vw = self.vw[idx]; self.inc_p = self.inc_p[:, idx]
                self.inc_yaw = self.inc_yaw[:, idx]
                self.prev_pos = self.prev_pos[idx]; self.prev_neg = self.prev_neg[idx]
                self.seen = self.seen[idx]; self.have_prev = self.have_prev[idx]
                self.batch = int(self.scans.shape[1])
            return self(scan_now, motion, dt)
        finally:
            (self.scans, self.att, self.vw, self.inc_p, self.inc_yaw, self.prev_pos,
             self.prev_neg, self.seen, self.have_prev, self.batch) = keep


def describe(spec: Optional[dict]) -> str:
    """One line for a log or a checkpoint header."""
    if not spec:
        return "no aligned channel"
    s = aligned_spec(**spec)
    return (f"aligned: k={s['k']} steps ({0.025 * s['k']:.3f} s), bearing tol "
            f"+-{s['tol_beams']} beams, soft threshold tau={s['tau']:.3f} m"
            + (f" + {s['tau_rel']:.3f} r" if s['tau_rel'] else "")
            + f", consistency +-{s['consist_beams']} beams, z tol {s['z_tol']:.3f} m, "
              f"gap fill {s['gap_fill']}")
