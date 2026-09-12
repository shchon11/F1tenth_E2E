"""Marching-squares contours of the occupancy layers, torch-free.

`track_contours` / `_runs_off_edge` lived in `server.py` and `track_contours_mask` in `native.py`;
both of those import torch at module level, and the environment editor draws contours inside the
console process, which never imports torch. The code is the same, moved: `mask_contours` is the
primitive over a boolean grid, and the two named wrappers keep their old signatures so the geometry
tests pass byte-identical. Anything with `.occupancy / .resolution / .origin` -- a `Track`, a
`SceneDoc`, a one-off duck -- is accepted where a track is.
"""
from __future__ import annotations

import numpy as np


def mask_contours(mask, resolution: float, origin, min_len: int = 8, outside_occupied: bool = True,
                  sigma_cells: float = 0.0, clip_canvas_edge: bool = False):
    """Closed contours (list of (K,2) float32 arrays in metres) of a boolean grid via marching
    squares. Row index increases with +y; `origin` is the lower-left corner of cell (0, 0).

    `outside_occupied` says what to assume lies beyond the grid's canvas. The padding ring exists
    so that a wall running off the edge still closes into a loop instead of ending mid-air, but
    filling it with *solid* also declares the canvas perimeter itself to be a wall, and the builders
    downstream then extrude that declaration: on Monza the largest duct contour was 8001 points
    tracing a 191.7 m square, every point on the canvas edge and 26 % of all duct vertices spent on
    it, drawn as a hose across the view. Passing False fills the ring with free space instead, so an
    edge-touching wall still closes -- along the real array edge, where it really ends -- and the
    canvas rectangle is simply never a contour. Legitimate walls that reach the edge are unaffected;
    only the synthetic ring goes.

    `clip_canvas_edge` goes one step further and drops the parts of a contour that run *along* the
    canvas edge, keeping the rest as open polylines -- see `_runs_off_edge`. That is the outward
    silhouette of a cropped SLAM surround, which is where the file ends rather than where anything
    physical is; the free-facing side of the same surround is a real wall and is kept.

    `sigma_cells` blurs the mask before contouring. On a binary grid marching squares can only place
    a vertex at a cell midpoint, so a diagonal comes out as treads and risers; a mild blur lets the
    0.5 crossing land between them. The displacement is sub-cell by construction and is measured
    against the unblurred contour rather than assumed -- see `gl_scene.smooth_contour`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mask = np.asarray(mask)
    occ = np.pad(mask.astype(float), 1, constant_values=1.0 if outside_occupied else 0.0)
    if sigma_cells > 0:
        from scipy import ndimage
        occ = ndimage.gaussian_filter(occ, sigma_cells, mode="nearest")
    fig = plt.figure(); ax = fig.add_subplot(111)
    cs = ax.contour(occ, levels=[0.5])
    segs = []
    paths = getattr(cs, "allsegs", None)
    if paths is not None and len(paths):
        segs = list(paths[0])
    else:                                   # matplotlib >= 3.8
        for p in cs.get_paths():
            for poly in p.to_polygons(closed_only=False):
                segs.append(np.asarray(poly))
    plt.close(fig)
    out = []
    H, W = mask.shape
    res = float(resolution)
    lo = np.array(origin, np.float64)
    hi = lo + np.array([W * res, H * res])
    for s in segs:
        if len(s) < min_len:
            continue
        xy = (np.asarray(s) - 1.0) * res + lo                                   # col->x, row->y
        if clip_canvas_edge:
            for run in _runs_off_edge(xy, lo, hi, res):
                if len(run) >= min_len:
                    out.append(run.astype(np.float32))
        elif len(xy) >= min_len:
            out.append(xy.astype(np.float32))
    return out


def track_contours(track, min_len: int = 8, outside_occupied: bool = True, sigma_cells: float = 0.0,
                   clip_canvas_edge: bool = False):
    """`mask_contours` over `track.occupancy` (anything with occupancy / resolution / origin)."""
    return mask_contours(track.occupancy, track.resolution, track.origin, min_len=min_len,
                         outside_occupied=outside_occupied, sigma_cells=sigma_cells,
                         clip_canvas_edge=clip_canvas_edge)


def track_contours_mask(track, mask, outside_occupied: bool = True, sigma_cells: float = 0.0,
                        clip_canvas_edge: bool = False):
    """Contours of an arbitrary boolean mask on the track's grid (same `resolution` / `origin`)."""
    return mask_contours(mask, track.resolution, track.origin, outside_occupied=outside_occupied,
                         sigma_cells=sigma_cells, clip_canvas_edge=clip_canvas_edge)


def _runs_off_edge(xy, lo, hi, res, tol=0.75):
    """Split a contour into the runs that are not lying along the map file's outer edge.

    A SLAM map's unknown surround is one occupied region that touches the border, so its contour has
    two halves that mean completely different things: the inner half is the real wall the car drives
    along, and the outer half is the rectangle where the *file* stops. Drawing both is what puts a
    crude rectangular box around the track and what pulled Korea's frame back to the canvas. Dropping
    the whole contour would take the real boundary with it, so only the vertices sitting on the edge
    go, and what is left stays as open polylines.

    `tol` is in cells; the contour of a border-adjacent cell sits half a cell in from the array edge,
    so anything under one cell has to count as on it."""
    on = ((np.abs(xy[:, 0] - lo[0]) < tol * res) | (np.abs(xy[:, 0] - hi[0]) < tol * res) |
          (np.abs(xy[:, 1] - lo[1]) < tol * res) | (np.abs(xy[:, 1] - hi[1]) < tol * res))
    if not on.any():
        return [xy]
    if on.all():
        return []
    # rotate so the split points are interior, then cut at the on-edge vertices
    start = int(np.argmin(on))
    rolled, mask = np.roll(xy, -start, 0), np.roll(on, -start)
    runs, cur = [], []
    for p, bad in zip(rolled, mask):
        if bad:
            if len(cur) > 1:
                runs.append(np.asarray(cur))
            cur = []
        else:
            cur.append(p)
    if len(cur) > 1:
        runs.append(np.asarray(cur))
    return runs
