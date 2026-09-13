"""The `clearance` controller arm: keep the *executed* plan off what the LiDAR can see.

Why this exists
---------------
Measured on the eight held-out proxy tracks (2026-09-13, `crash_attribution.py`): the privileged
raceline teacher completes 30-32 of 32 trials through the same iLQR tracker the policy uses, while
the policies complete 7-24. At the policies' collisions the **policy's own last plan**
(`tracker.last_ref`, body frame) had a median minimum body-edge clearance of 0.00-0.06 m; 85 % of
collisions had a plan margin under 0.10 m and 49 % of the plans passed *through* occupied cells. The
car still had ~0.2 m of clearance 25 ms before impact, and impact speed on the narrow real floors is
2.6-2.9 m/s against mu 0.93. The tracks are feasible, the tracker can follow safe plans, and the
policy plans with no margin. So this is a geometry clamp, exactly as `grip_control.py` is a friction
clamp: a runtime layer that needs no training.

Where it sits
-------------
**Between the decoded plan and the tracker**, on `PlanTracker._plan_hook`. Two consequences, and
both are the reason for that choice rather than a solver hook:

* `tracker.last_ref` -- the thing the attribution script measures, and the thing the tracker
  actually follows -- is built inside `mpc.solve` *from the action*. An arm that changes the action
  is therefore measured, and executed, as the plan it produced.
* `fixed_low` (`grip_control.GripMPC`) replaces `tracker._solver`. This replaces
  `tracker._plan_hook`. They are at different levels, so **neither can overwrite the other and the
  installation order cannot change the command**: the friction envelope is always computed on the
  adjusted geometry, which is the plan the car will drive.

What it may do
--------------
Only two things, and both are one-sided:

* **bend** -- one curvature offset added to the knots inside the tracker's own horizon, tapered to
  zero beyond it so the tail curvature (and therefore `fixed_low`'s speed envelope over the tail) is
  untouched;
* **slow** -- lower the plan's two speed targets.

It never raises a speed and never straightens a plan the policy bent. With the margin already met
everywhere, the action is returned **bit-identical** -- `legacy` behaviour is a no-op, not an
approximation of one.

What it may see
---------------
The current LiDAR frame and nothing else. No map, no `track.edt`, no pose, no privileged state: the
1081 returns become a coarse occupancy grid in the car's own frame, and a distance field on that
grid. A bearing with no return contributes nothing, and everything outside the grid is treated as
**free** -- the arm acts on what the sensor saw, and says so, rather than braking for the unknown
region behind the 270 deg window. That is the same discipline the policy's own inputs keep, and it
is what lets `f1sim_ros/policy_node.py` run this identical code off `/scan`.

Conventions
-----------
`margin` is a **body-edge** clearance, and `body_radius` (0.14 m) is the centre-to-edge distance the
attribution script subtracts, so the default 0.20 m margin is 0.34 m from a plan point to the
nearest return. The car's true half-width is 0.155 m and its half-length 0.29 m; 0.14 m is neither,
it is the disc the measurement uses, and matching it is what makes the before/after table a
comparison of the same quantity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn.functional as F

from .. import mpc as _mpc

#: Centre -> body edge [m]. The convention `crash_attribution.py` measures in (`edt - 0.14`), kept
#: here so "the plan margin is no longer at zero" is a statement about one quantity.
BODY_RADIUS = 0.14

#: The margin the arm defends, body-edge [m]. 0.20 m edge = 0.34 m from centre to the nearest return.
MARGIN = 0.20

#: [m/s] below this, a speed cap is float noise rather than a decision. See `adjust`.
V_EPS = 1e-3


@dataclass(frozen=True)
class ClearanceSpec:
    """Everything the arm needs besides the scan. Recorded with any result."""

    margin: float = MARGIN            # [m] body-edge clearance the adjusted plan is asked to keep
    body_radius: float = BODY_RADIUS  # [m] centre -> body edge

    # -- the local occupancy -----------------------------------------------------
    cell: float = 0.06                # [m] grid cell. A return is binned to the cell that contains
                                      # it, whose centre is the nearest centre, so the position
                                      # error is <= cell/2 per axis (0.042 m radial) and unbiased.
    x_min: float = -0.24              # [m] behind the base_link origin. Small: the rear 90 deg is
    x_max: float = 4.44               # outside the sensor's 270 deg window and is never observed.
    y_half: float = 2.04              # [m] half width of the grid
    d_clip: float = 0.60              # [m] the distance field saturates here. Everything the arm
                                      # decides is a function of clearance below `margin +
                                      # body_radius` = 0.34 m, so a 0.60 m ceiling loses nothing and
                                      # bounds the transform's cost at R = 10 offsets per pass.
    range_eps: float = 0.02           # a normalized range within this of 1.0 is "no return"

    # -- the plan ----------------------------------------------------------------
    n_path: int = 25                  # dense samples of the plan, as `grip_control.speed_envelope`
    horizon_s: float = 0.60           # [s] the tracker's own N*dt: the arc the reference walks, and
                                      # therefore the arc this arm is responsible for. Beyond it the
                                      # plan is replanned before it is ever executed.
    s_eval_min: float = 1.0           # [m] evaluation window floor, so a stopped car still looks
    s_eval_max: float = 4.2           # [m] and ceiling, inside the grid
    s_min: float = 0.50               # [m] where the arm's responsibility starts. base_link is the
                                      # rear axle and the car's nose is ~0.48 m ahead of it, so a
                                      # plan point closer than this is *inside the car's own
                                      # footprint*: its clearance is a fact about where the car
                                      # already is, which no bend can move (every candidate leaves
                                      # (0,0) straight ahead) and no speed can change. Counting it
                                      # would tie every candidate's score together on a narrow
                                      # floor and cap the speed for the wall the car is already
                                      # safely alongside.

    # -- the bend ----------------------------------------------------------------
    max_shift: float = 0.60           # [m] lateral displacement at the evaluation horizon
    n_shift: int = 6                  # candidate magnitudes each side (2n+1 candidates in all)
    dk_max: float = 1.0               # [1/m] ceiling on the curvature offset, so an adjustment
                                      # stays one and does not become a different plan
    shift_penalty: float = 0.05       # [m of clearance per m of deviation] the preference for the
                                      # smallest deviation that meets the margin

    # -- the cap -----------------------------------------------------------------
    v_stop: float = 0.60              # [m/s] the speed allowed where the plan has no clearance left
    a_brake: float = 3.3              # [m/s^2] what the backward pass assumes the car can shed --
                                      # deliberately NOT the 5.0 m/s^2 the tracker is allowed to
                                      # command. `grip_control`'s tracker_chain_probe measured the
                                      # deceleration the whole command chain actually delivers for a
                                      # steady request at low / mid / high grip as -3.300 / -3.977 /
                                      # -4.429 m/s^2. This arm's promise is that the car can still
                                      # shed the speed by the time it arrives, so the number that
                                      # keeps the promise is the one the chain delivers on the worst
                                      # floor.
                                      #
                                      # Under `fixed_low` the solver is clamped tighter than this:
                                      # its friction budget `q g lf / (L + q h)` is 2.97 m/s^2 on a
                                      # straight plan at mu 0.734 and less in a corner (1.72 m/s^2
                                      # mean over a measured held-out run). So under the composite
                                      # this backward pass is optimistic about how late it may start
                                      # slowing. That is a limitation, not a bug to paper over by
                                      # reading the other layer's bound -- doing so would make the
                                      # two layers' order matter, which is the one property that
                                      # makes them composable. The cap is a *target* re-issued at
                                      # 40 Hz and tightening as the obstacle nears, and the
                                      # deceleration actually commanded is the tracker's to bound;
                                      # like the grip envelope, this is a planning approximation and
                                      # not a stopping guarantee.

    def validate(self) -> "ClearanceSpec":
        if not self.margin > 0.0:
            raise ValueError(f"margin must be positive, got {self.margin}: a zero margin is a "
                             f"`legacy` run wearing this arm's name, and the speed rule divides "
                             f"by it")
        if not self.cell > 0.0:
            raise ValueError(f"cell must be positive, got {self.cell}")
        if not self.x_max > self.x_min or not self.y_half > 0.0:
            raise ValueError("the occupancy grid must have a positive extent")
        if not self.d_clip >= self.margin + self.body_radius:
            raise ValueError(
                f"d_clip {self.d_clip} saturates below the margin it has to measure "
                f"({self.margin} + {self.body_radius}); every clearance would read as met")
        if not 0.0 <= self.s_min < self.s_eval_min:
            raise ValueError(f"s_min {self.s_min} must be below the evaluation floor "
                             f"{self.s_eval_min}, or the window is empty for a stopped car")
        if self.n_path < 3 or self.n_shift < 1:
            raise ValueError("need at least 3 path samples and 1 shift magnitude")
        if not self.a_brake > 0.0 or not self.v_stop >= 0.0:
            raise ValueError("a_brake must be positive and v_stop non-negative")
        return self

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.cell))

    @property
    def ny(self) -> int:
        return int(round(2.0 * self.y_half / self.cell))

    @property
    def radius_cells(self) -> int:
        """Offsets each separable pass of the distance transform has to consider.

        The two passes are exact for any true distance within `radius_cells` cells on **each** axis,
        and every distance that survives the `d_clip` ceiling is, so the truncation is free.
        """
        return int(math.ceil(self.d_clip / self.cell))

    def to_meta(self) -> dict:
        d = asdict(self)
        d.update(nx=self.nx, ny=self.ny, radius_cells=self.radius_cells,
                 margin_centre=self.margin + self.body_radius)
        return d


def beam_angles(n_beams: int, fov: float, device=None, dtype=None) -> torch.Tensor:
    """The bearings of `n_beams` returns spread over `fov`, endpoints included -- `Lidar.angles`."""
    return torch.linspace(-fov / 2.0, fov / 2.0, n_beams, device=device, dtype=dtype)


def occupancy(scan_norm: torch.Tensor, angles: torch.Tensor, spec: ClearanceSpec,
              range_max: float, mount_x: float = 0.297, mount_y: float = 0.0) -> torch.Tensor:
    """Coarse occupancy in the car's frame from one LiDAR frame. (B, N) -> (B, ny, nx) in {0, 1}.

    `scan_norm` is the observation's own units: range / range_max, clamped to [0, 1], no return =
    1.0 (`obs.ObsBuilder.build`, `gym_env._norm_scan`). `mount_x` is the sensor's offset ahead of
    base_link -- 0.297 m from `/tf_static` in the recordings -- because the plan is in base_link and
    the returns are in the sensor's frame, and 0.297 m is one and a half margins.

    Beams with no return are dropped rather than placed at `range_max`: an unobserved bearing is
    unknown, and the cell the beam would have ended in is not occupied by anything that was seen.
    """
    if scan_norm.dim() != 2:
        raise ValueError(f"scan must be (B, N), got {tuple(scan_norm.shape)}")
    if angles.numel() != scan_norm.shape[1]:
        raise ValueError(f"{angles.numel()} bearings for {scan_norm.shape[1]} returns: the arm must "
                         f"be built with the beam geometry the scan it is fed actually has")
    B = scan_norm.shape[0]
    nx, ny = spec.nx, spec.ny
    ang = angles.to(scan_norm.device, scan_norm.dtype)[None]                  # (1, N)
    r = scan_norm * float(range_max)
    seen = scan_norm < (1.0 - spec.range_eps)
    px = mount_x + r * torch.cos(ang)
    py = mount_y + r * torch.sin(ang)
    ix = ((px - spec.x_min) / spec.cell).floor().long()
    iy = ((py + spec.y_half) / spec.cell).floor().long()
    inside = seen & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    # One spare column collects everything that is not a return inside the grid, and is then
    # dropped. `scatter_` is last-write-wins, so without a bin of its own a discarded beam whose
    # clamped index happened to be 0 would erase a real return in the first cell.
    flat = torch.where(inside, iy.clamp(0, ny - 1) * nx + ix.clamp(0, nx - 1),
                       torch.full_like(ix, nx * ny))
    grid = torch.zeros(B, nx * ny + 1, device=scan_norm.device, dtype=scan_norm.dtype)
    grid.scatter_(1, flat, torch.ones_like(flat, dtype=scan_norm.dtype))
    return grid[:, :nx * ny].view(B, ny, nx)


def distance_field(occ: torch.Tensor, spec: ClearanceSpec) -> torch.Tensor:
    """Euclidean distance [m] from each cell centre to the nearest occupied centre, clipped.

    Felzenszwalb's separable decomposition of the squared transform, with the 1-D min-plus
    convolutions written out as a fixed number of shifted minima: exact (not a chamfer
    approximation), fixed shapes, no data-dependent control flow, so it is as CUDA-graph-safe as
    anything else on this path and costs 2 x `radius_cells` elementwise passes over a grid of a few
    thousand cells. Truncating each pass at `radius_cells` is exact for every distance the `d_clip`
    ceiling keeps, because such a distance is within that many cells on each axis.
    """
    if occ.dim() != 3:
        raise ValueError(f"occupancy must be (B, ny, nx), got {tuple(occ.shape)}")
    c2 = spec.cell * spec.cell
    big = float((3.0 * spec.d_clip) ** 2)
    f = torch.where(occ > 0, torch.zeros_like(occ), torch.full_like(occ, big))
    R = spec.radius_cells
    g = f
    for k in range(1, R + 1):                       # along y (rows)
        cost = c2 * k * k
        up = F.pad(f[:, k:, :], (0, 0, 0, k), value=big)
        down = F.pad(f[:, :-k, :], (0, 0, k, 0), value=big)
        g = torch.minimum(g, torch.minimum(up, down) + cost)
    h = g
    for k in range(1, R + 1):                       # then along x (columns), over the first pass
        cost = c2 * k * k
        left = F.pad(g[:, :, k:], (0, k), value=big)
        right = F.pad(g[:, :, :-k], (k, 0), value=big)
        h = torch.minimum(h, torch.minimum(left, right) + cost)
    return h.clamp_max(float(spec.d_clip ** 2)).sqrt()


def sample_field(dist: torch.Tensor, px: torch.Tensor, py: torch.Tensor,
                 spec: ClearanceSpec) -> torch.Tensor:
    """Bilinear lookup of the distance field at body-frame points. dist (B, ny, nx), p (B, ..., n).

    Outside the grid the answer is `d_clip`, i.e. **free**. The arm is allowed to know only what the
    sensor saw; the alternative -- calling the unobserved region occupied -- would have the car brake
    for the 90 deg the LiDAR does not cover, every step, for ever.
    """
    B = dist.shape[0]
    shape = px.shape
    gx = 2.0 * (px - spec.x_min) / (spec.x_max - spec.x_min) - 1.0
    gy = 2.0 * (py + spec.y_half) / (2.0 * spec.y_half) - 1.0
    grid = torch.stack([gx.reshape(B, 1, -1), gy.reshape(B, 1, -1)], -1)      # (B, 1, K, 2)
    out = F.grid_sample(dist[:, None], grid, mode="bilinear", padding_mode="border",
                        align_corners=False)[:, 0, 0]                         # (B, K)
    inside = (gx.abs() <= 1.0) & (gy.abs() <= 1.0)
    return torch.where(inside, out.reshape(shape), torch.full_like(px, float(spec.d_clip)))


def _eval_window(v_ref: torch.Tensor, spec: ClearanceSpec) -> torch.Tensor:
    """How far along the plan this arm is responsible for [m], per env.

    The tracker's reference walks `N * dt` seconds of path and the iLQR follows that; everything
    beyond is replanned before it is executed. Constraining the whole plan instead would hold a
    0.20 m margin over the 6.8 m a 4.5 m/s plan spans -- far stricter than anything the car drives,
    and on a narrow floor it would simply cap the speed everywhere.
    """
    return (spec.horizon_s * v_ref.abs()).clamp(spec.s_eval_min, spec.s_eval_max)


def _candidates(spec: ClearanceSpec, s_eval: torch.Tensor, device, dtype) -> torch.Tensor:
    """Curvature offsets to try, (B, C). Ordered 0, -d1, +d1, -d2, +d2, ... by rising |offset|.

    Parameterised through the lateral displacement they produce at the evaluation horizon --
    `y(s) ~ dk s^2 / 2` -- so the candidate set means the same thing at 1.5 m/s and at 5 m/s. The
    displacement is the *parameterisation*; what is scored is the exact path each offset produces.
    """
    C = spec.n_shift
    mags = torch.arange(1, C + 1, device=device, dtype=dtype) * (spec.max_shift / C)   # (C,)
    order = torch.stack([-mags, mags], 1).reshape(-1)                                  # -d1,+d1,-d2,...
    delta = torch.cat([torch.zeros(1, device=device, dtype=dtype), order])              # (2C+1,)
    dk = 2.0 * delta[None] / (s_eval[:, None] ** 2).clamp_min(1e-3)
    return dk.clamp(-spec.dk_max, spec.dk_max)


def _taper(s_knot: torch.Tensor, s_eval: torch.Tensor, taper_len: torch.Tensor) -> torch.Tensor:
    """Knot weights: 1 inside the evaluation window, linearly to 0 one knot spacing past it.

    Without it a constant curvature offset would bend the whole plan, and the *tail* curvature is
    what `fixed_low`'s speed envelope reads -- so bending to clear an obstacle 2 m away would also
    quietly slow the car for a corner 6 m away that the policy never planned.
    """
    return ((s_eval[:, None] + taper_len[:, None] - s_knot) / taper_len[:, None]).clamp(0.0, 1.0)


@dataclass
class Adjustment:
    """What the arm did this step, for logging and for the tests. All (B,) unless noted."""
    action: torch.Tensor            # (B, A) the adjusted action
    dk: torch.Tensor                # [1/m] curvature offset chosen
    shift: torch.Tensor             # [m] lateral deviation at the evaluation horizon
    clear_before: torch.Tensor      # [m] min body-edge clearance of the policy's own plan
    clear_after: torch.Tensor       # [m] and of the adjusted one, over the evaluation window
    v0: torch.Tensor                # [m/s] capped plan speeds
    v1: torch.Tensor
    dv: torch.Tensor                # [m/s] total speed removed, (v0+v1) after - before
    s_eval: torch.Tensor            # [m] the window each env was judged over


def adjust(action: torch.Tensor, v_meas: torch.Tensor, speed_cap: torch.Tensor,
           dist: torch.Tensor, spec: _mpc.PlanSpec, cspec: ClearanceSpec,
           v_max: float) -> Adjustment:
    """The whole arm, given a distance field: bend the plan, then cap its speed.

    `action` is the policy's normalized plan. The returned action is **bit-identical** wherever the
    arm left something alone: the bend is applied in normalized curvature units (adding 0 is exact)
    and each speed is replaced through a `where` rather than a round trip through `decode`/`encode`.
    """
    n = cspec.n_path
    B = action.shape[0]
    dev, dt_ = action.device, action.dtype
    a = action.clamp(-1.0, 1.0)
    k, Lp, v0, v1 = _mpc.decode(a, v_meas, v_max, speed_cap, spec)
    s_eval = _eval_window(torch.maximum(torch.maximum(v_meas.abs(), v0), v1), cspec)

    # -- candidate plans, scored on the exact path each one produces -------------
    dk = _candidates(cspec, s_eval, dev, dt_)                                  # (B, C)
    C = dk.shape[1]
    xi = torch.linspace(0.0, 1.0, _mpc.N_KNOTS, device=dev, dtype=dt_)[None]    # (1, K)
    s_knot = xi * Lp[:, None]                                                  # (B, K)
    taper = _taper(s_knot, s_eval, (Lp / (_mpc.N_KNOTS - 1)).clamp_min(1e-3))  # (B, K)
    k_cand = (k[:, None, :] + dk[:, :, None] * taper[:, None, :]).clamp(-spec.kappa_max, spec.kappa_max)
    x, y, _psi, s = _mpc.path_points(k_cand.reshape(B * C, _mpc.N_KNOTS),
                                     Lp[:, None].expand(B, C).reshape(B * C), n=n)
    x, y = x.view(B, C, n), y.view(B, C, n)
    s = s.view(B, C, n)[:, 0]                                                  # (B, n), same for all
    in_eval = (s >= cspec.s_min) & (s <= s_eval[:, None])                      # (B, n)
    raw = sample_field(dist, x, y, cspec) - cspec.body_radius                  # (B, C, n) body edge
    # Everything from the first point the plan's body edge is *inside* something the scan saw is
    # void. The field is unsigned, so a point a metre past a wall reads as a metre of free space,
    # and a plan that drives through the wall and out the other side would otherwise score better
    # than one that only grazes it -- precisely backwards, because the car stops at the first thing
    # it hits.
    #
    # Void from the first *contact*, not a running minimum of the clearance. A running minimum is
    # pinned by the tightest point the plan has already passed, so on a narrow floor -- where the
    # window's first sample is routinely the tightest one, and no bend can move it -- every
    # candidate scores identically and the arm does nothing exactly where it is needed. Only an
    # actual contact makes what comes after it meaningless.
    hit = (raw < 0.0) & in_eval[:, None, :]
    blocked = hit.to(dt_).cumsum(-1) > 0
    eff = torch.where(blocked, torch.full_like(raw, -cspec.body_radius), raw)
    # The score is the **mean** over the window of that clearance, saturated at the margin:
    # how much of the arc ahead the plan keeps clear, and how clear. Saturated, because once a
    # candidate has enough room more room is not better, so the deviation penalty then picks the
    # smallest bend that is enough -- and a candidate that keeps the margin everywhere scores
    # exactly `margin`, which no other candidate can beat, so "meets the margin" is still identified
    # exactly.
    #
    # A worst-sample score was tried first and is wrong here: once a candidate penetrates at all,
    # the *unsigned* field pins its minimum at -body_radius plus grid quantisation, so between two
    # plans that both end up in the wall the minimum is noise while the mean still says which one
    # stayed clear longer. On a narrow floor that is every candidate, and the noise was choosing the
    # bend.
    room = eff.clamp(max=cspec.margin)
    w = in_eval[:, None, :].to(dt_)
    score = (room * w).sum(-1) / w.sum(-1).clamp_min(1.0)
    # The last term is a strict tie-break, so the choice cannot depend on which index an argmax
    # happens to return; candidates are ordered by rising deviation, so a tie goes to the smaller.
    idx = torch.arange(C, device=dev, dtype=dt_)[None]
    score = score - cspec.shift_penalty * (dk.abs() * s_eval[:, None] ** 2 * 0.5) - 1e-6 * idx
    best = score.argmax(1)                                                     # (B,)

    take = best[:, None, None]
    clear_b = eff.gather(1, take.expand(B, 1, n))[:, 0]                        # (B, n)
    y_b = y.gather(1, take.expand(B, 1, n))[:, 0]
    dk_b = dk.gather(1, best[:, None])[:, 0]
    #: Candidate 0 is the zero offset by construction, so this is the policy's own plan, scored on
    #: the same field and the same window as the adjusted one. The reported margin is the pointwise
    #: minimum of the raw body-edge clearance over the window -- the same quantity, measured the
    #: same way, as `crash_attribution.py`'s `plan_min_clear`.
    raw_b = raw.gather(1, take.expand(B, 1, n))[:, 0]
    raw_0 = raw[:, 0]
    y_0 = y[:, 0]

    # -- the cap: a speed the clearance allows, then braking anticipation --------
    cap = speed_cap.reshape(-1, 1)
    # Continuous and one-sided: at or above the margin this is the cap the caller already asked
    # for, so it binds nothing; at no clearance at all it is `v_stop`.
    v_tight = cspec.v_stop + (cap - cspec.v_stop).clamp_min(0.0) * (clear_b / cspec.margin).clamp(0.0, 1.0)
    free = torch.full_like(v_tight, float(v_max) * 4.0)
    v_tight = torch.where(in_eval, v_tight, free)
    ds = (Lp / (n - 1)).clamp_min(1e-3)
    cols = list(v_tight.unbind(1))
    for i in range(n - 2, -1, -1):        # backward: no faster than can still be shed by then
        cols[i] = torch.minimum(cols[i], torch.sqrt(cols[i + 1] ** 2 + 2.0 * cspec.a_brake * ds))
    v_allow = torch.stack(cols, 1)                                             # (B, n)

    # -- fitting the envelope with the two numbers a plan actually has --------
    # The plan's speed is linear in arc fraction, so the question is which line under the envelope
    # to pick. The constraint set runs to `s_eval` and includes the samples *before* the window: the
    # backward pass wrote the braking requirement into them, and that is exactly what makes the car
    # start slowing before it arrives.
    frac = (s / Lp[:, None].clamp_min(1e-3)).clamp(0.0, 1.0)                   # (B, n)
    rest = 1.0 - frac
    in_prof = s <= s_eval[:, None]
    big = torch.full_like(v_allow, float(v_max) * 4.0)

    # (a) the fastest near target, with the end target taken down to whatever keeps the line legal.
    #     `v0` also has to leave room for an end target of zero, or a line pinned at v1 = 0 would
    #     still cross the envelope on its way down.
    v0_cap = torch.where(in_prof & (rest > 1e-3), v_allow / rest.clamp_min(1e-3), big).amin(1)
    v0_a = torch.minimum(torch.minimum(v0, v_allow[:, 0]), v0_cap)
    head = torch.where(in_prof & (frac > 1e-3),
                       (v_allow - v0_a[:, None] * rest) / frac.clamp_min(1e-3), big)
    v1_a = torch.minimum(v1, head.amin(1)).clamp_min(0.0)

    # (b) the policy's own profile, scaled. Where the envelope is flat and low -- a car driving
    #     parallel to a wall it is already close to -- (a) spends everything on the near target and
    #     crushes the far one to nothing, which is a hard brake the geometry never asked for; a
    #     scaled profile keeps the shape the policy chose and simply lowers it.
    v_lin = v0[:, None] * rest + v1[:, None] * frac
    alpha = torch.where(in_prof, v_allow / v_lin.clamp_min(1e-3), big).amin(1).clamp(0.0, 1.0)
    v0_b, v1_b = v0 * alpha, v1 * alpha

    # Both are feasible by construction, so take the faster one. Neither can exceed what was asked.
    take_b = (v0_b + v1_b) > (v0_a + v1_a)
    v0_new = torch.where(take_b, v0_b, v0_a)
    v1_new = torch.where(take_b, v1_b, v1_a)
    # A cap of a millimetre per second is float noise in the envelope, not a decision: at full
    # clearance `v_tight` *is* the caller's own cap, and the arithmetic that carries it through the
    # backward pass and the line fit lands a few ULPs below. Without this the arm would report
    # itself as having slowed the car on every step of an empty straight.
    v0_new = torch.where(v0_new < v0 - V_EPS, v0_new, v0)
    v1_new = torch.where(v1_new < v1 - V_EPS, v1_new, v1)

    # -- back to a normalized action, changing only what was actually adjusted ---
    # Each branch below hands back the caller's own bits when this arm did nothing, so a plan that
    # already keeps the margin leaves through a bit-identical action rather than through a
    # round trip that happens to land close.
    bent = dk_b != 0.0
    k_norm = (a[:, :_mpc.N_KNOTS] + (dk_b[:, None] / spec.kappa_max) * taper).clamp(-1.0, 1.0)
    enc = lambda v: (v / v_max * 2.0 - 1.0).clamp(-1.0, 1.0)
    a_new = torch.cat([
        torch.where(bent[:, None], k_norm, action[:, :_mpc.N_KNOTS]),
        torch.where(v0_new < v0, enc(v0_new), action[:, _mpc.N_KNOTS])[:, None],
        torch.where(v1_new < v1, enc(v1_new), action[:, _mpc.N_KNOTS + 1])[:, None],
        action[:, _mpc.N_KNOTS + 2:],
    ], 1)

    # The largest *index* inside the window, not the count: the window is an interval that
    # starts at `s_min`, so counting its samples would point one short of where it ends.
    ar = torch.arange(n, device=dev)[None]
    last = torch.where(in_eval, ar, torch.zeros_like(ar)).amax(1, keepdim=True)
    guard = lambda c: torch.where(in_eval, c, torch.full_like(c, float(cspec.d_clip))).amin(1)
    return Adjustment(action=a_new, dk=dk_b, shift=(y_b - y_0).gather(1, last)[:, 0],
                      clear_before=guard(raw_0), clear_after=guard(raw_b),
                      v0=v0_new, v1=v1_new, dv=(v0_new - v0) + (v1_new - v1), s_eval=s_eval)


class ClearanceArm:
    """Installs `adjust` on the tracker's plan hook and keeps the current scan live.

    The scan is a **buffer the caller writes in place**, exactly as `GripMPC.mu` is: the hook closes
    over it, so a new frame reaches the installed arm without reinstalling anything, and a solver
    graph captured underneath keeps its static inputs.

    Order-free by construction. `GripMPC` replaces `tracker._solver`; this replaces
    `tracker._plan_hook`. Installing either first leaves the other untouched, and the friction
    envelope is computed from the adjusted action in both orders -- which is the only order that is
    right, because the adjusted plan is the one the car drives.
    """

    def __init__(self, tracker, cspec: ClearanceSpec, batch: int, device, v_max: float,
                 angles: torch.Tensor, range_max: float, mount_x: float = 0.297,
                 mount_y: float = 0.0):
        self.cspec = (cspec or ClearanceSpec()).validate()
        self.tracker = tracker
        self.B = int(batch)
        self.device = torch.device(device)
        self.v_max = float(v_max)
        self.range_max = float(range_max)
        self.mount_x, self.mount_y = float(mount_x), float(mount_y)
        self.angles = angles.detach().to(self.device).reshape(-1).clone()
        #: The newest LiDAR frame, in the observation's own units (range / range_max, no return =
        #: 1.0). All ones until the first `update_scan`, which is "nothing seen" -- the arm does
        #: nothing before it has been fed, rather than braking for a grid it has not been given.
        self.scan = torch.ones(self.B, self.angles.numel(), device=self.device)
        self.last: Optional[Adjustment] = None
        self._prev_hook = None
        self._installed = False
        #: Device-side sums, drained once per update by `metrics()`. A `float(t)` per step here is a
        #: host synchronise, and this arm runs inside the step.
        self._acc = torch.zeros(7, device=self.device, dtype=torch.float64)
        self._steps = 0

    # -- geometry ----------------------------------------------------------------
    def set_angles(self, angles: torch.Tensor) -> None:
        """Re-declare the beam bearings, for a sensor whose window is not the nominal one.

        The car's `/scan` carries `angle_min` / `angle_max`; a node that assumed +-135 deg against a
        driver publishing something else would build the grid from bearings the returns do not have,
        which is a silently rotated obstacle rather than an error.
        """
        a = angles.detach().to(self.device).reshape(-1).clone()
        if a.numel() != self.scan.shape[1]:
            self.scan = torch.ones(self.B, a.numel(), device=self.device)
        self.angles = a

    # -- per step ----------------------------------------------------------------
    def update_scan(self, scan_norm: torch.Tensor) -> None:
        """Hand the arm this control step's newest LiDAR frame, (B, N) normalized."""
        s = scan_norm.detach()
        if s.dim() == 3:                       # a stacked observation: frame 0 is the newest
            s = s[:, 0]
        if s.shape != self.scan.shape:
            raise ValueError(f"scan {tuple(s.shape)} is not this arm's {tuple(self.scan.shape)}; "
                             f"build one arm per inference path")
        self.scan.copy_(s)

    @torch.no_grad()
    def field(self) -> torch.Tensor:
        """The distance field for the frame currently held. (B, ny, nx) in metres."""
        occ = occupancy(self.scan, self.angles, self.cspec, self.range_max,
                        self.mount_x, self.mount_y)
        return distance_field(occ, self.cspec)

    @torch.no_grad()
    def shape(self, action: torch.Tensor, v_meas: torch.Tensor,
              speed_cap: torch.Tensor) -> torch.Tensor:
        """The plan hook: the policy's action in, the adjusted action out."""
        adj = adjust(action, v_meas, speed_cap, self.field(), self.tracker.spec, self.cspec,
                     self.v_max)
        self.last = adj
        self._record(adj)
        return adj.action

    def _record(self, adj: Adjustment) -> None:
        c = self.cspec
        self._steps += 1
        self._acc += torch.stack([
            torch.tensor(float(self.B), device=self.device, dtype=torch.float64),
            (adj.dk != 0).double().sum(),
            (adj.dv < 0).double().sum(),
            adj.shift.abs().double().sum(),
            (-adj.dv).double().sum(),
            adj.clear_before.clamp(max=c.margin).double().sum(),
            adj.clear_after.clamp(max=c.margin).double().sum(),
        ])

    def metrics(self, prefix="controller/") -> dict:
        """Drain the arm's sums. In env-transitions, like the rest of `grip_runtime`."""
        if self._steps == 0:
            return {}
        v = self._acc.tolist()                                   # the one host synchronise
        n = max(v[0], 1.0)
        d = {f"{prefix}clearance_bent_frac": v[1] / n,
             f"{prefix}clearance_slowed_frac": v[2] / n,
             f"{prefix}clearance_shift_mean": v[3] / n,
             f"{prefix}clearance_speed_cut_mean": v[4] / n,
             f"{prefix}clearance_plan_margin_before": v[5] / n,
             f"{prefix}clearance_plan_margin_after": v[6] / n}
        self._acc = torch.zeros_like(self._acc)
        self._steps = 0
        return d

    # -- install / release -------------------------------------------------------
    def install(self) -> "ClearanceArm":
        if self._installed:
            raise RuntimeError("already installed")
        if self.tracker is None:
            raise RuntimeError("the clearance arm needs the plan tracker (--action-mode plan)")
        if getattr(self.tracker, "_plan_hook", None) is not None:
            raise RuntimeError(f"tracker._plan_hook is already {self.tracker._plan_hook!r}; two "
                               f"plan shapers on one tracker is not something to resolve by "
                               f"ordering")
        self._prev_hook = getattr(self.tracker, "_plan_hook", None)
        self.tracker._plan_hook = self.shape
        self._installed = True
        return self

    def release(self) -> None:
        if self._installed:
            self.tracker._plan_hook = self._prev_hook
            self._installed = False
        self._prev_hook = None

    def meta(self) -> dict:
        """What a result has to carry to be reproducible: the spec and the beam geometry."""
        return {"spec": self.cspec.to_meta(), "n_beams": int(self.angles.numel()),
                "fov_deg": float((self.angles[-1] - self.angles[0]).abs() * 180.0 / math.pi)
                if self.angles.numel() > 1 else 0.0,
                "range_max": self.range_max, "mount_x": self.mount_x, "mount_y": self.mount_y}
