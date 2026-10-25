"""A speed limit for an obstacle on the policy's own planned path, from the LiDAR scan alone (2026-10-26).

The venue had boxes on the racing line (the user, 2026-10-25), and in races the other cars push boxes into our path:
with the props on the line, more than half of the hard contacts left after the speed limit for a car ahead are props
(docs/research/passing-risk-2026-10-24, sections 7 and 13). The policy brakes for a box that appears in its path but
rarely steers round it in time.

The rule: walk the plan's own path (its curvature knots over its own length) in `step` m steps, from the front bumper
on; at the first point within `corridor` of an obstacle return, hold the speed to the speed from which the car can still
stop there:

    limit = sqrt(2 brake max(0, s - front - gap_min - v lag))

An obstacle return is a return of a SMALL ISOLATED CLUSTER of the scan: a run of at least `min_beams` valid beams with no
range jump over `break_jump` between neighbours, under `max_extent` end to end, standing at least `stand` in front of
the beams beyond both of its ends (the farthest of the `stand_look` beams beyond each end that returned: a beam grazing the
object's own side face is not its background, and neither is a beam that returned nothing next to a wall seen at a
grazing angle; when none of them returned, that is open space). A wall is long, a box or a pole is not -- so no map and no localisation are needed,
which our map-less stack does not have. Returns within `car_exclude` of the policy's own nearest-car reading are left to
the speed limit for a car ahead (`opponent_speed_limit`). Nothing reads the simulator's truth: the same code runs in the
controller node (`f1sim_ros`, parameter `obstacle_speed_limit`), the console and `evaluate --obstacle-speed-limit`.

Measured on the Korea entry (`korea26_v2_ppo_s2421_u512_nearcar_slow`, finals race cells, props on the line, the car limit
on in both rows, same card per row, the rule as it stands since 300e051): all hard (>= 1.5 m/s) contacts 2.19 -> 1.74 /km
(seeds 81-84) and 1.95 -> 1.70 (fresh 91-94), hard prop contacts 1.24 -> 0.72 and 0.91 -> 0.75, race speed -1.1 / -1.6 %,
passes unchanged. Before 300e051 a missed return counted as background and walls seen at a grazing angle set half of its
holds: -32 % hard contacts at -4.4 % speed, the extra coming from slowing at walls, not from boxes. What it still misses
(prop_contact_anatomy.py, 51 hard prop contacts): pieces standing against a wall or another piece (47 %; a wall-attached
box is part of the wall's cluster), holds that came too late (35 %), the smallest 0.135 m box the 0.11 m scan plane passes
over at speed. On the real venue runs (open loop) it would have held the car on 3.9-6.8 % of the scans, on boxes and
objects; wall features 0-8 % of those.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ObstacleSpeedLimit:
    brake: float = 3.0           # [m/s^2] the deceleration the limit assumes
    lag: float = 0.2             # [s] drive lag at the current speed before braking bites
    gap_min: float = 0.1         # [m] left between the bumper and the obstacle
    front: float = 0.45          # [m] reference point -> front bumper; path points before it do not count
    corridor: float = 0.19       # [m] a return this close to a path point is in the path
    step: float = 0.1            # [m] path walk step
    reach: float = 6.0           # [m] the longest walk (the plan's own length bounds it first)
    break_jump: float = 0.15     # [m] a range jump between neighbouring beams that ends a cluster
    max_extent: float = 0.6      # [m] a cluster longer than this end to end is a wall
    stand: float = 0.3           # [m] a cluster stands at least this far in front of the beams beyond both its ends
    min_beams: int = 3
    car_exclude: float = 0.45    # [m] returns this close to the nearest-car reading are the car limit's
    # beams beyond each end the stand test looks at, the farthest of them counting (2026-10-26). Against the one beam next
    # to the end, a box was missed wherever a beam grazing its own side face (one beam, 0.1-0.2 m behind its front face:
    # a range jump that ends the cluster) stood next to it -- 12 % of the positions of an isolated 0.3 m box 0.8-4 m
    # ahead, 30 % at 4 m (synthetic scans), and in closed loop the limit let go and took hold again from scan to scan
    stand_look: int = 3
    # only beams that RETURNED count as what a cluster stands in front of; when none of the `stand_look` beams beyond an end
    # returned, that is open space (2026-10-26). A wall seen at a grazing angle misses returns (5-9 % of beams on the
    # competition track below |cos| 0.3, `LidarParams.duct_graze_dropout`), and a no-return read as far background made
    # the stretch of wall between two misses a small isolated cluster: measure-only in the race cells, walls set 60 % of
    # the limit's holds and half of its speed cut, 97 % of them cut out by a no-return beam; synthetic corridors with the
    # measured miss rates kept a wall piece in 59-67 % of scans, and none with this
    stand_valid_only: bool = True


DEFAULT = ObstacleSpeedLimit()


def cluster_keep(r: torch.Tensor, ok: torch.Tensor, px: torch.Tensor, py: torch.Tensor,
                 p: ObstacleSpeedLimit = DEFAULT) -> torch.Tensor:
    """(B, n) the valid returns `ok` of ranges `r` (points px, py) that belong to a small isolated cluster."""
    B, n = r.shape
    dev = r.device
    rf = torch.where(ok, r, torch.full_like(r, 30.0))                       # no return = far behind
    jump = (rf[:, 1:] - rf[:, :-1]).abs() > p.break_jump
    brk = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), jump | ~ok[:, 1:] | ~ok[:, :-1]], 1)
    seg = torch.cumsum(brk.long(), 1)                                       # the cluster id of every beam
    gid = (seg + torch.arange(B, device=dev)[:, None] * (n + 2)).flatten()
    idx = torch.arange(n, device=dev)[None].expand(B, n).flatten()
    first = torch.full((B * (n + 2),), n, dtype=torch.long, device=dev).scatter_reduce(0, gid, idx, "amin", include_self=True)
    last = torch.full((B * (n + 2),), -1, dtype=torch.long, device=dev).scatter_reduce(0, gid, idx, "amax", include_self=True)
    f, l = first[gid].view(B, n), last[gid].view(B, n)
    bi = torch.arange(B, device=dev)[:, None].expand(B, n)
    ext = torch.hypot(px[bi, l] - px[bi, f], py[bi, l] - py[bi, f])
    look = range(1, max(1, int(p.stand_look)) + 1)
    rs = torch.where(ok, r, torch.full_like(r, -1.0)) if p.stand_valid_only else rf   # -1: no return, no evidence
    r_left = torch.stack([rs[bi, (f - j).clamp_min(0)] for j in look]).max(0).values
    r_right = torch.stack([rs[bi, (l + j).clamp(max=n - 1)] for j in look]).max(0).values
    if p.stand_valid_only:                                                  # nothing beyond returned: open space
        r_left = torch.where(r_left < 0, torch.full_like(r_left, 30.0), r_left)
        r_right = torch.where(r_right < 0, torch.full_like(r_right, 30.0), r_right)
    standing = (f > 0) & (l < n - 1) & (r_left > rf[bi, f] + p.stand) & (r_right > rf[bi, l] + p.stand)
    return ok & ((l - f + 1) >= p.min_beams) & (ext < p.max_extent) & standing


def path_points(k: torch.Tensor, L: torch.Tensor, p: ObstacleSpeedLimit = DEFAULT):
    """(B, N, 2) the plan's path in the car frame (curvature k[:, i] over the i-th part of L, then the last knot) and
    (N,) the arc length of each point."""
    N = int(round(p.reach / p.step))
    s = torch.arange(1, N + 1, device=k.device, dtype=k.dtype) * p.step
    seg = (s[None, :] / (L[:, None] / k.shape[1]).clamp_min(1e-3)).floor().long().clamp(max=k.shape[1] - 1)
    th = torch.cumsum(k.gather(1, seg) * p.step, 1)
    return torch.stack([torch.cumsum(torch.cos(th) * p.step, 1), torch.cumsum(torch.sin(th) * p.step, 1)], -1), s


def speed_limit(ranges: torch.Tensor, angles: torch.Tensor, k: torch.Tensor, L: torch.Tensor, v_me: torch.Tensor,
                mount_x: float = 0.297, mount_y: float = 0.0, car_reading: Optional[torch.Tensor] = None,
                range_max: float = 30.0, p: ObstacleSpeedLimit = DEFAULT) -> torch.Tensor:
    """(B,) [m/s] the speed limit for the first obstacle on the plan's path, +inf where there is none.

    ranges (B, n) [m] the scan (non-finite or >= range_max: no return), angles (n,) or (B, n) its bearings from the LiDAR;
    k (B, K), L (B,) the plan's curvature knots and length (`mpc.decode_wp4` / `mpc.decode`); v_me (B,) our forward speed;
    car_reading (B, 5): the policy's nearest-car output (`opponent_speed_limit`), whose returns are left to the car limit."""
    r = ranges.float()
    ang = angles.float()
    if ang.dim() == 1:
        ang = ang[None].expand_as(r)
    ok = torch.isfinite(r) & (r > 0.05) & (r < range_max - 1e-3)
    rr = torch.where(ok, r, torch.zeros_like(r))
    px, py = mount_x + rr * torch.cos(ang), mount_y + rr * torch.sin(ang)        # car frame
    ok = cluster_keep(r, ok, px, py, p)
    if car_reading is not None:
        from .opponent_speed_limit import relative
        ex, cx, cy, _cl, _vl = relative(car_reading.to(r.device), mount_x, mount_y)
        ok = ok & ~(ex[:, None] & (torch.hypot(px - cx[:, None], py - cy[:, None]) < p.car_exclude))
    pts, s = path_points(k.float(), L.float(), p)                               # (B, N, 2)
    d2 = (pts[:, :, None, 0] - px[:, None, :]) ** 2 + (pts[:, :, None, 1] - py[:, None, :]) ** 2   # (B, N, n)
    d2 = torch.where(ok[:, None, :], d2, torch.full_like(d2, 1e9))
    near = (d2.min(2).values.sqrt() < p.corridor) & (s[None, :] <= L.float()[:, None]) & (s[None, :] > p.front)
    s_hit = torch.where(near, s[None, :].expand_as(near), torch.full_like(near, float("inf"), dtype=s.dtype)).min(1).values
    room = (s_hit - p.front - p.gap_min - v_me.float().clamp_min(0.0) * p.lag).clamp_min(0.0)
    return torch.where(torch.isfinite(s_hit), torch.sqrt(2.0 * p.brake * room), torch.full_like(s_hit, float("inf")))
