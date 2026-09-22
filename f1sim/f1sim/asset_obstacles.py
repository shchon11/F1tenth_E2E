"""Asset-only scenario placement. Legacy raster scenarios remain available for replay.

The dimensions stored in StaticProp are the sole geometry contract: rendering, finite-height
LiDAR, contact and the teacher's projection all build the same authored asset at that size.
"""
from dataclasses import replace
import inspect
import math

import numpy as np

from . import props
from .track import StaticProp

_LINEAR_DIMS = {"width", "depth", "height", "radius", "scale"}
_FOOTPRINT_DIMS = {"width", "depth", "radius"}

#: [m] No placed asset is shorter than this. The LiDAR scans at 0.110 m above the floor, randomised
#: to 0.125 m (`params.py`, `lidar.mount_z`), and the body's pitch tilts that plane further; an
#: obstacle the scan passes over can only be found by hitting it. The high family used to shrink an
#: asset *uniformly* to fit its box, height included: ICCAS seed 1 stood a 0.108 m crate stack and two
#: 0.122 m blocks, and s912 and s913 hit that corner 29 times in a kilometre.
MIN_VISIBLE_HEIGHT = 0.16


def scaled_prop(prop, scale):
    """Materialize uniform scaling without scaling facets, taper or random appearance seeds."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("asset scale must be finite and positive")
    dims = dict(prop.dims)
    if prop.style == "mesh":
        dims["scale"] = float(dims.get("scale", 1.0)) * scale
    else:
        signature = inspect.signature(props.REGISTRY[prop.style])
        for key in _LINEAR_DIMS.intersection(signature.parameters):
            dims[key] = float(dims.get(key, signature.parameters[key].default)) * scale
    return replace(prop, dims=tuple(sorted(dims.items())))


def fitted_prop(prop, footprint, height):
    """`prop` with its footprint scaled by `footprint` and its height by `height`, never shorter than
    `MIN_VISIBLE_HEIGHT`. Fitting an asset into a box is a statement about the floor it covers; how
    tall it stands is what decides whether the LiDAR sees it, and that is not the box's to decide."""
    if not (math.isfinite(footprint) and footprint > 0 and math.isfinite(height) and height > 0):
        raise ValueError("asset scale must be finite and positive")
    if prop.style == "mesh":
        return scaled_prop(prop, footprint)
    dims = dict(prop.dims)
    signature = inspect.signature(props.REGISTRY[prop.style])
    for key in _FOOTPRINT_DIMS.intersection(signature.parameters):
        dims[key] = float(dims.get(key, signature.parameters[key].default)) * footprint
    if "height" in signature.parameters:
        h = float(dims.get("height", signature.parameters["height"].default)) * height
        dims["height"] = max(h, MIN_VISIBLE_HEIGHT)
    return replace(prop, dims=tuple(sorted(dims.items())))


def with_asset_obstacles(track, family="edge", seed=0, *, asset="mixed", scale=1.0, n=None):
    """Place existing asset presets; never invent raster walls or resize an asset to make it fit.

    Low hugs either boundary; medium alternates boundary and centerline placements. High uses
    the existing hard-layout recipe. Legacy pinch presets place opposing authored assets. Oversized requests fail loudly
    when no safe placement exists instead of silently launching an empty obstacle scenario.
    """
    if asset != "mixed" and asset not in props.STYLES:
        raise ValueError(f"unknown obstacle asset: {asset}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("asset scale must be finite and positive")
    existing = tuple(scaled_prop(p, scale) for p in track.props) if scale != 1 else tuple(track.props)
    out = replace(track, props=existing, name=f"{track.name}_assets_{asset}_{scale:g}")
    if not family:
        return out
    if family not in ("edge", "props", "line", "pinch", "hard"):
        raise ValueError(f"unknown asset placement: {family}")
    if family == "hard":
        from .hard_obstacles import with_hard_obstacles
        recipe = with_hard_obstacles(out.for_planning() if existing else track, seed=seed, n=n)
        rng = np.random.default_rng(seed)
        styles = props.STYLES if asset == "mixed" else (asset,)
        additions = []
        for x, y, yaw, sx, sy in recipe.hard_boxes:
            style = styles[int(rng.integers(len(styles)))]
            p = StaticProp(style, x, y, yaw, seed=int(rng.integers(1 << 30)))
            footprint = p.build().envelope.footprint
            span = np.ptp(footprint, axis=0)
            # Accepted legacy patterns prove their boxes passable. Assets fit inside each
            # accepted box, preserving that corridor while using real finite geometry.
            fit = min(sx / span[0], sy / span[1])
            jitter = float(rng.uniform(.85, 1.15))
            additions.append(fitted_prop(p, min(jitter * fit * scale, fit), jitter * scale))
        out.props = existing + tuple(additions)
        out.hard_patterns = recipe.hard_patterns
        if not additions:
            raise ValueError("no high-density asset pattern fits this track")
        return out
    if track.centerline is None:
        raise ValueError("asset placement requires a centerline")
    line = np.asarray(track.centerline, float)
    tangent = np.roll(line, -1, axis=0) - np.roll(line, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(1e-9)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    ds = np.linalg.norm(np.roll(line, -1, axis=0) - line, axis=1)
    arc = np.r_[0., np.cumsum(ds)][:-1]
    length = ds.sum()
    if n is None:
        n = int(np.clip(round(length / (18. if family == "line" else 35.)), 1 if family == "edge" else 2, 12))
    rng = np.random.default_rng(seed)
    styles = props.STYLES if asset == "mixed" else (asset,)
    keep = [(p.x, p.y, p.build().envelope.radius) for p in existing]
    sites, placed = [], []
    for _ in range(max(200, int(n) * 100)):
        if len(sites) >= n:
            break
        i = int(rng.integers(len(line)))
        if any(min(abs(arc[i] - arc[j]), length - abs(arc[i] - arc[j])) < 3.0 for j in sites):
            continue
        style = styles[int(rng.integers(len(styles)))]
        base = StaticProp(style, 0., 0., seed=int(rng.integers(1 << 30)))    # drawn before the size, as ever
        size = scale * float(rng.uniform(.65, 1.35))
        p = fitted_prop(base, size, size)
        radius = p.build().envelope.radius
        left = track.free_width_along(line[i], normal[i])
        right = track.free_width_along(line[i], -normal[i])
        side = float(rng.choice((-1., 1.)))
        center = family == "line" and len(sites) % 2 == 1
        if center:
            offsets = [0.]
            # At least one whole vehicle-width corridor must pass beside the full asset.
            if max(left, right) - radius < .70:
                continue
        elif family == "pinch":
            if left + right - 4 * radius - .10 < .70:
                continue
            offsets = [left - radius - .05, -right + radius + .05]
        else:
            if left + right - 2 * radius - .05 < .70:
                continue
            offsets = [side * ((left if side > 0 else right) - radius - .05)]
        candidates = []
        for off in offsets:
            xy = line[i] + normal[i] * off
            col, row = np.rint((xy - track.origin) / track.resolution).astype(int)
            if not (0 <= row < track.edt.shape[0] and 0 <= col < track.edt.shape[1]):
                break
            if track.edt[row, col] < radius + track.resolution:
                break
            if any(math.hypot(xy[0] - x, xy[1] - y) < radius + r + .10 for x, y, r in keep):
                break
            candidates.append(replace(p, x=float(xy[0]), y=float(xy[1]),
                                      yaw=math.atan2(tangent[i, 1], tangent[i, 0])))
        if len(candidates) != len(offsets):
            continue
        placed.extend(candidates)
        keep.extend((q.x, q.y, radius) for q in candidates)
        sites.append(i)
    if not placed and n:
        raise ValueError(f"{asset} at {scale:g}× does not fit this track with a passable gap")
    out.props = existing + tuple(placed)
    out.props_skipped = max(0, int(n) - len(sites))
    return out
