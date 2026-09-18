"""Local obstacle detours for an ordered, periodic offline racing-line seed."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import route_through_array


def _densify(points, spacing, closed=True):
    points = np.asarray(points, dtype=float)
    output = []
    count = len(points) if closed else len(points) - 1
    for i in range(count):
        a, b = points[i], points[(i + 1) % len(points)]
        steps = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        output.append(a + np.arange(steps)[:, None] / steps * (b - a))
    if not closed:
        output.append(points[-1:])
    return np.concatenate(output)


def obstacle_aware_seed(track, centerline=None, *, clearance: float, spacing=None):
    """Repair blocked stretches while retaining the original loop's order.

    ``track`` must include every static obstacle (``Track.for_planning()``). Clearance
    is the required center-to-solid distance in metres. Inflated-grid routes are only
    seeds, not steering-feasible trajectories; the trajectory solver must still check
    its full vehicle footprint. Unblocked input is returned unchanged, without smoothing.
    A blocked lap is rejected rather than replacing it with an unrelated global shortcut.
    """
    points = np.asarray(track.centerline if centerline is None else centerline, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('obstacle routing needs a finite closed centerline with at least three points')
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError('obstacle routing clearance must be finite and nonnegative')
    res = float(track.resolution)
    step = res * 0.4 if spacing is None else min(float(spacing), res * 0.4)
    if not np.isfinite(step) or step <= 0:
        raise ValueError('obstacle routing spacing must be finite and positive')
    origin = np.asarray(track.origin)
    # Distance is between cell centers. Subtract one cell diagonal to bound the
    # distance between arbitrary positions in the free and occupied cells.
    distance_to_solid = ndimage.distance_transform_edt(
        np.pad(~track.occupancy, 1, constant_values=False))[1:-1, 1:-1] * res
    free = distance_to_solid > clearance + np.sqrt(2) * res
    height, width = free.shape

    def clear(q):
        ij = np.floor((q - origin) / res).astype(int)
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < width) & (ij[:, 1] >= 0) & (ij[:, 1] < height)
        answer = np.zeros(len(q), dtype=bool)
        answer[inside] = free[ij[inside, 1], ij[inside, 0]]
        return answer

    dense = _densify(points, step)
    bad = ~clear(dense)
    if not bad.any():
        return points.copy()
    # Leave room to approach and leave a detour, then merge nearby blocked stretches.
    radius = max(1, int(np.ceil(max(0.5, 2 * clearance) / step)))
    extended = np.r_[bad, bad, bad]
    affected = ndimage.maximum_filter1d(extended.astype(np.uint8), 2 * radius + 1,
                                       mode='wrap')[len(bad):2 * len(bad)].astype(bool)
    if affected.all():
        raise ValueError(f'{track.name}: no clear anchors for an obstacle-aware lap at {clearance:.3f} m clearance')
    # Begin at a clear anchor; blocked runs across the original seam become ordinary runs.
    shift = int(np.flatnonzero(~affected)[0])
    dense = np.roll(dense, -shift, axis=0)
    affected = np.roll(affected, -shift)
    tree = cKDTree(dense)
    n = len(dense)
    starts = np.flatnonzero(affected & ~np.roll(affected, 1))
    ends = np.flatnonzero(affected & ~np.roll(affected, -1))
    output, cursor = [], 0
    for start, end in zip(starts, ends):
        first, last = start - 1, end + 1
        a, b = dense[first], dense[last % n]
        segment = dense[first:last + 1] if last < n else np.vstack([dense[first:], dense[:1]])
        route = None
        for padding in (max(2., 4 * clearance), max(4., 8 * clearance), max(8., 16 * clearance)):
            lo = np.maximum(np.floor((segment.min(0) - padding - origin) / res).astype(int), 0)
            hi = np.minimum(np.ceil((segment.max(0) + padding - origin) / res).astype(int), [width - 1, height - 1])
            rows, cols = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            world = np.stack([cols + .5, rows + .5], -1) * res + origin
            distance, index = tree.query(world.reshape(-1, 2))
            # Allow a little overlap at anchors, but never route via another part of the lap.
            progress = (index - first) % n
            allowed = (progress <= last - first + radius) | (progress >= n - radius)
            allowed = allowed.reshape(rows.shape) & free[rows, cols]
            costs = np.where(allowed, 1. + .15 * distance.reshape(rows.shape), np.inf)
            anchors = np.floor((np.array([a, b]) - origin) / res).astype(int) - lo
            try:
                indices, cost = route_through_array(costs, tuple(anchors[0, ::-1]),
                                                    tuple(anchors[1, ::-1]), fully_connected=False)
            except ValueError:
                continue
            if not np.isfinite(cost):
                continue
            route = (np.asarray(indices)[:, ::-1] + lo + .5) * res + origin
            route = np.vstack([a, route, b])
            break
        if route is None:
            raise ValueError(f'{track.name}: blocked corridor; no ordered obstacle detour at {clearance:.3f} m clearance')
        def edge_clear(a, b):
            samples = _densify(np.array([a, b]), step, closed=False)
            if not clear(samples).all():
                return False
            progress = (tree.query(samples)[1] - first) % n
            return bool(((progress <= last - first + radius) | (progress >= n - radius)).all())

        # String pulling removes grid stair steps without ever cutting occupied cells.
        smooth = [route[0]]
        i = 0
        while i < len(route) - 1:
            j = len(route) - 1
            while j > i + 1 and not edge_clear(route[i], route[j]):
                j -= 1
            if not edge_clear(route[i], route[j]):
                raise ValueError(f'{track.name}: obstacle detour has an unsafe grid transition')
            smooth.append(route[j])
            i = j
        output.extend([dense[cursor:first], _densify(np.asarray(smooth), step, closed=False)])
        cursor = last + 1
    output.append(dense[cursor:])
    result = np.concatenate(output)
    keep = np.r_[True, np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-10]
    result = result[keep]
    if np.linalg.norm(result[0] - result[-1]) < 1e-10:
        result = result[:-1]
    if not clear(_densify(result, step)).all():
        raise ValueError(f'{track.name}: obstacle-aware seed failed final edge-clearance validation')
    return result
