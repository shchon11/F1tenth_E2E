"""A mandatory obstacle, proven before any car drives at it.

`Track.with_lane_obstacles` is not usable for avoidance: it places boxes against a wall
(`off = side * (half_w - sy/2 - 0.05)`, track.py:330), so the racing line can stay entirely clear and
a car that never deviates still "passes" the map. We stamp a box centred on the line instead, and
then prove three things with no policy involved:

  1. blocking      -- the line really is obstructed over the swept longitudinal extent
  2. corridor      -- a gap at least one car wide plus margin survives on exactly one side
  3. connectivity  -- that gap joins the lane before and after, eroded by the car's own footprint,
                      so it is a route and not a pocket

The erosion is the part that is easy to get wrong: a corridor wide enough for a *point* is not wide
enough for a car, so connectivity is tested on free space eroded by the footprint, including the
rear box the contact test actually uses.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage

from .geom import REAR_BOX_MAX_M

#: Half-width of clear air required beside the car body, per side, for the alternative corridor.
CORRIDOR_MARGIN_M = 0.10

#: The corridor is at least this fraction of the lane, on top of the absolute minimum.
#:
#: Pinning it to the minimum instead made the obstacle as wide as the lane allowed, so a WIDER lane
#: demanded a LARGER sideways move into the same narrow slot -- backwards, and measured: 0.977 m of
#: displacement on gen:control:9100 against 0.715 m on the narrower gen:control:1400, with the
#: scripted expert clearing the second and hitting the first every trial. Scaling the corridor keeps
#: the manoeuvre proportionate to the lane while the obstacle still blocks the line.
CORRIDOR_LANE_FRACTION = 0.45


class GeometryError(ValueError):
    """A scenario that cannot be driven. Raised before the freeze, never scored around."""


@dataclass(frozen=True)
class Placement:
    """Where the obstacle went and what the proofs measured. Serialised into the suite file."""
    s_obs_m: float
    centre_xy: tuple
    size_m: tuple
    side: int
    corridor_width_m: float
    required_corridor_m: float
    blocked_span_m: float
    n_cells: int
    #: Half the lane width at s_obs, and the lateral offset of the free corridor's CENTRE from the
    #: centreline (signed: negative is left). A driver aiming at the corridor needs its centre, not
    #: its width -- aiming at half the width leaves the car still inside the obstacle.
    half_lane_m: float = 0.0
    corridor_centre_offset_m: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _cl_arc(cl: np.ndarray) -> np.ndarray:
    """Cumulative arc length of a closed centreline, starting at 0."""
    d = np.linalg.norm(np.diff(np.vstack([cl, cl[:1]]), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)[:-1]])


def _frame_at(cl: np.ndarray, i: int):
    """(point, tangent, normal) at centreline index i."""
    tang = cl[(i + 1) % len(cl)] - cl[i - 1]
    tang = tang / (np.linalg.norm(tang) + 1e-9)
    return cl[i], tang, np.array([-tang[1], tang[0]])


def _index_at_arc(cl: np.ndarray, s: float) -> int:
    arc = _cl_arc(cl)
    return int(np.argmin(np.abs(arc - (s % arc[-1]))))


def place_blocking_obstacle(track, s_obs_m: float, *, vehicle_length: float, vehicle_width: float,
                            side: int = 1, margin_m: float = CORRIDOR_MARGIN_M):
    """Return `(new_track, Placement)`, or raise GeometryError if the scenario is not drivable.

    The box is centred on the centreline and sized so the line is obstructed, offset toward `side`
    only far enough that the *other* side keeps a full corridor.
    """
    if track.centerline is None:
        raise GeometryError("a blocking obstacle needs a centreline")
    cl = track.centerline
    i = _index_at_arc(cl, s_obs_m)
    c, tang, nrm = _frame_at(cl, i)

    res, origin = track.resolution, track.origin
    H, W = track.occupancy.shape
    r0 = int(round((c[1] - origin[1]) / res))
    c0 = int(round((c[0] - origin[0]) / res))
    if not (0 <= r0 < H and 0 <= c0 < W):
        raise GeometryError(f"s={s_obs_m:.1f} m lies outside the map")

    half_lane = float(track.edt[r0, c0])                 # ~half the lane width here
    required = vehicle_width + 2 * margin_m
    # The box spans from one wall inward, leaving `required` on the far side. Its extent across the
    # lane is what makes the encounter mandatory; along the lane it need only be long enough to
    # obstruct the swept footprint.
    # Leave the required corridor PLUS rasterisation slack. The box and the walls are both snapped
    # to cells and the lane is not exactly symmetric about the centreline, so sizing to leave
    # exactly `required` reliably lands a cell or two short -- measured 0.45 m against a 0.51 m
    # requirement on gen:control:1400 before this slack was added.
    design_corridor = max(required + 2 * res, CORRIDOR_LANE_FRACTION * 2 * half_lane)
    across = 2 * half_lane - design_corridor
    # The box is pushed to one side, so its near edge sits at (half_lane - across) from the line.
    # To reach the centreline at all it needs across >= half_lane, i.e. half_lane >= required.
    # Below that the obstacle cannot both block the line and leave a corridor -- it would sit
    # against the wall, which is exactly the stock behaviour this module exists to avoid.
    if half_lane < design_corridor:
        raise GeometryError(
            f"lane at s={s_obs_m:.1f} m is {2*half_lane:.2f} m wide; it cannot hold an obstacle that "
            f"blocks the line and still leave the required {design_corridor:.2f} m corridor")
    along = vehicle_length * 0.5

    # centre of the box: pushed toward `side` so the free corridor is on the other side
    off = side * (half_lane - across / 2.0)
    centre = c + nrm * off

    occ = track.occupancy.copy()
    tall = track.tall.copy() if track.tall is not None else None
    ys = np.arange(H) * res + origin[1]
    xs = np.arange(W) * res + origin[0]
    gx, gy = np.meshgrid(xs, ys)
    dx, dy = gx - centre[0], gy - centre[1]
    u = dx * tang[0] + dy * tang[1]                      # along the lane
    v = dx * nrm[0] + dy * nrm[1]                        # across it
    box = (np.abs(u) <= along / 2.0) & (np.abs(v) <= across / 2.0)
    if not box.any():
        raise GeometryError("obstacle footprint rasterised to nothing")
    occ |= box
    if tall is not None:
        tall |= box

    new = type(track).from_occupancy(occ, res, origin, cl, f"{track.name}_block{int(s_obs_m)}",
                                     duct=track.duct, tall=tall, duct_height=track.duct_height)
    corridor_w = 2 * half_lane - across                  # what is left on the far side
    place = Placement(s_obs_m=float(s_obs_m), centre_xy=(float(centre[0]), float(centre[1])),
                      size_m=(float(along), float(across)), side=int(side),
                      half_lane_m=float(half_lane),
                      corridor_centre_offset_m=float(-side * (half_lane - corridor_w / 2.0)),
                      corridor_width_m=float(required), required_corridor_m=float(required),
                      # cells the box *adds*: part of it overlaps the wall it is pushed against,
                      # and those were never free space to begin with
                      blocked_span_m=float(along), n_cells=int((box & ~track.occupancy).sum()))
    prove(new, track, place, vehicle_length=vehicle_length, vehicle_width=vehicle_width,
          margin_m=margin_m)
    return new, place


# ---------------------------------------------------------------- the three proofs

def prove(new_track, base_track, place: Placement, *, vehicle_length: float, vehicle_width: float,
          margin_m: float = CORRIDOR_MARGIN_M) -> dict:
    """Run all three proofs. Raises GeometryError on the first failure; returns measurements."""
    return {"blocking": prove_blocking(new_track, base_track, place, vehicle_length=vehicle_length),
            "corridor": prove_corridor(new_track, place, vehicle_width=vehicle_width,
                                       base_track=base_track, margin_m=margin_m),
            "connectivity": prove_connectivity(new_track, place, vehicle_length=vehicle_length,
                                               vehicle_width=vehicle_width)}


def prove_blocking(new_track, base_track, place: Placement, *, vehicle_length: float) -> dict:
    """The centreline is obstructed over the swept longitudinal extent.

    The extent includes the rear box: `_car_contacts` appends it (sim.py:213), so the car sweeps
    `vehicle_length + REAR_BOX_MAX_M`, not `vehicle_length`.
    """
    cl = new_track.centerline
    arc = _cl_arc(cl)
    sweep = vehicle_length + REAR_BOX_MAX_M
    ds = np.abs((arc - place.s_obs_m + arc[-1] / 2) % arc[-1] - arc[-1] / 2)
    window = ds <= sweep / 2.0
    if not window.any():
        raise GeometryError("no centreline samples inside the sweep window")
    res, origin = new_track.resolution, new_track.origin
    H, W = new_track.occupancy.shape
    hit = 0
    for p in cl[window]:
        r = int(round((p[1] - origin[1]) / res))
        c = int(round((p[0] - origin[0]) / res))
        if 0 <= r < H and 0 <= c < W and new_track.occupancy[r, c]:
            hit += 1
    if hit == 0:
        raise GeometryError(
            f"obstacle at s={place.s_obs_m:.1f} m does not block the line: a car holding the "
            f"centreline never meets it, so the encounter is not mandatory")
    return {"samples_in_window": int(window.sum()), "blocked_samples": hit, "sweep_m": float(sweep)}


def prove_corridor(new_track, place: Placement, *, vehicle_width: float,
                   base_track=None, margin_m: float = CORRIDOR_MARGIN_M) -> dict:
    """A gap of at least `vehicle_width + 2*margin` survives on one side, INSIDE the lane.

    Bounding the scan to the lane is the whole difficulty. Track walls are thin and the obstacle is
    pushed up against one of them, so a ray cast outward from the centreline leaves the circuit
    entirely and finds the open space beyond -- metres of it. Two earlier versions of this function
    reported exactly that and called every map passable.

    So the lane is defined by the **base** track (free before the obstacle was added), and the
    corridor is the longest run of cells that are free in the new track and inside that lane.
    """
    cl = new_track.centerline
    i = _index_at_arc(cl, place.s_obs_m)
    c, _, nrm = _frame_at(cl, i)
    res, origin = new_track.resolution, new_track.origin
    H, W = new_track.occupancy.shape
    required = vehicle_width + 2 * margin_m
    tolerance = res          # one cell: box and walls are both rasterised. Recorded, not hidden.

    def cell(track, t):
        p = c + nrm * t
        r = int(round((p[1] - origin[1]) / res))
        cc = int(round((p[0] - origin[0]) / res))
        if not (0 <= r < H and 0 <= cc < W):
            return True
        return bool(track.occupancy[r, cc])

    widths = {}
    for name, sign in (("left", -1.0), ("right", 1.0)):
        run, best = 0.0, 0.0
        for t in np.arange(0.0, 6.0, res):
            if base_track is not None and cell(base_track, sign * t):
                break                                   # left the lane: stop at the wall
            if cell(new_track, sign * t):
                run = 0.0                               # inside the obstacle
                continue
            run += res
            best = max(best, run)
        widths[name] = float(best)

    passable = [k for k, v in widths.items() if v >= required - tolerance]
    if not passable:
        raise GeometryError(
            f"no side keeps a {required:.2f} m corridor at s={place.s_obs_m:.1f} m "
            f"(left {widths['left']:.2f}, right {widths['right']:.2f}): the obstacle is not passable")
    return {"widths_m": widths, "required_m": float(required),
            "tolerance_m": float(tolerance), "passable_sides": passable}


def prove_connectivity(new_track, place: Placement, *, vehicle_length: float,
                       vehicle_width: float, window_m: float = 8.0) -> dict:
    """A body-sized route passes the obstacle **locally**, beside it.

    A global connected-component test is wrong on a closed circuit. Every point of a lap is reachable
    from every other the long way round, so a corridor that is sealed just downstream still lands
    approach and exit in one component and all three proofs report success. Reproduced: blocking 7
    samples, corridor 0.70 m, component 1 -- on a scenario no car can drive through.

    So connectivity is restricted to a local window around the obstacle, and then the route is
    required to actually use the gap beside it: deleting the obstacle's own arc slice must
    disconnect approach from exit. If it does not, the local region contains a bypass and the
    encounter is not mandatory.
    """
    res = new_track.resolution
    radius = 0.5 * float(vehicle_width)
    free = ~new_track.occupancy
    passable = ndimage.binary_erosion(free, ndimage.generate_binary_structure(2, 2),
                                      iterations=max(1, int(round(radius / res))))

    cl = new_track.centerline
    arc = _cl_arc(cl)
    L = arc[-1]
    origin = new_track.origin
    H, W = new_track.occupancy.shape
    sweep = vehicle_length + REAR_BOX_MAX_M

    def band(centre_s: float, half_width_m: float) -> np.ndarray:
        """Cells near the centreline samples whose arc lies within +-half_width of centre_s."""
        d = np.abs((arc - centre_s + L / 2) % L - L / 2)
        idx = np.nonzero(d <= half_width_m)[0]
        m = np.zeros((H, W), bool)
        for j in idx:
            r = int(round((cl[j, 1] - origin[1]) / res))
            c = int(round((cl[j, 0] - origin[0]) / res))
            if 0 <= r < H and 0 <= c < W:
                m[r, c] = True
        # widen to the lane, not just the line
        return ndimage.binary_dilation(m, ndimage.generate_binary_structure(2, 2),
                                       iterations=max(1, int(round(3.0 / res))))

    local = band(place.s_obs_m, window_m)
    local_passable = passable & local

    def nearest(s, mask):
        i = _index_at_arc(cl, s)
        c, _, nrm = _frame_at(cl, i)
        for t in np.arange(0.0, 4.0, res):
            for sign in (1.0, -1.0):
                p = c + nrm * (sign * t)
                r = int(round((p[1] - origin[1]) / res))
                cc = int(round((p[0] - origin[0]) / res))
                if 0 <= r < H and 0 <= cc < W and mask[r, cc]:
                    return (r, cc)
        return None

    before = nearest(place.s_obs_m - window_m * 0.75, local_passable)
    after = nearest(place.s_obs_m + window_m * 0.75, local_passable)
    if before is None or after is None:
        raise GeometryError(
            f"no footprint-sized free cell {'before' if before is None else 'after'} the obstacle: "
            f"the approach or exit is narrower than the car")

    struct = ndimage.generate_binary_structure(2, 2)
    lab, _ = ndimage.label(local_passable, structure=struct)
    if lab[before] != lab[after]:
        raise GeometryError(
            f"the gap beside the obstacle at s={place.s_obs_m:.1f} m is a pocket: within {window_m} m "
            f"of it, approach and exit are not connected through footprint-sized free space")

    # The route must go past the obstacle, not around the lap and not through a local bypass.
    cut = local_passable & ~band(place.s_obs_m, sweep / 2.0)
    lab_cut, _ = ndimage.label(cut, structure=struct)
    if cut[before] and cut[after] and lab_cut[before] == lab_cut[after]:
        raise GeometryError(
            f"approach and exit stay connected at s={place.s_obs_m:.1f} m with the obstacle's own arc "
            f"slice removed: there is a bypass, so the encounter is not mandatory")

    return {"erosion_radius_m": float(radius), "component": int(lab[before]),
            "window_m": float(window_m), "swept_length_m": float(sweep),
            "local_cells": int(local_passable.sum()), "lane_length_m": float(L)}



def find_feasible_s(track, *, vehicle_length: float, vehicle_width: float,
                    start_m: float = 10.0, step_m: float = 2.0, max_tries: int = 60,
                    side: int = 1, margin_m: float = CORRIDOR_MARGIN_M):
    """First arc position whose placement passes all three proofs.

    Scanned in a fixed order from a fixed start, so the choice is deterministic and depends only on
    track geometry. No policy runs here and no outcome is visible, which is what makes selecting a
    position legitimate: a scenario may be moved because the lane is too narrow to drive, never
    because a system scored badly on it.
    """
    tried = []
    for k in range(max_tries):
        s_obs = start_m + k * step_m
        try:
            new, place = place_blocking_obstacle(
                track, s_obs, vehicle_length=vehicle_length, vehicle_width=vehicle_width,
                side=side, margin_m=margin_m)
            return new, place, {"s_obs_m": s_obs, "rejected": tried}
        except GeometryError as exc:
            tried.append({"s_obs_m": s_obs, "why": str(exc)})
    raise GeometryError(
        f"no feasible obstacle position on {track.name} after {max_tries} tries from {start_m} m; "
        f"the map cannot host this scenario -- drop it rather than weaken the margins")
