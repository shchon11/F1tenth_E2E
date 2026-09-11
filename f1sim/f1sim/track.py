"""Track / map representation: occupancy grid + Euclidean distance field + centerline.

Coordinates: world (x, y) in meters, grid (row, col) with row increasing along +y
(ROS map images are stored top-row-first, so they are flipped on load).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy import ndimage


@dataclass(frozen=True)
class StaticProp:
    """One placed obstacle: a style name and where it stands. Not a mesh and not grid cells.

    The alternative -- stamping the obstacle into `tall` the way `with_lane_obstacles` does -- was
    what made every obstacle a rotated rectangle of a single fixed height, and worse, an obstacle
    with no top: the tracer treats `tall` as unbounded above, so a 30 cm box drawn on screen was an
    infinitely high wall to every beam. Keeping the placement as a record instead lets the renderer,
    the LiDAR and the contact test each read the same finite convex sections."""
    style: str                                # one of f1sim.props.STYLES
    x: float
    y: float
    yaw: float = 0.0                          # [rad], CCW from world +x
    seed: int = 0
    dims: tuple = ()                          # ((key, value), ...) -- a tuple so Track stays hashable

    def build(self):
        from . import props as _props
        return _props.build(self.style, seed=self.seed, **dict(self.dims))


@dataclass
class Track:
    """Layered obstacle map.
    occupancy : every obstacle the car can hit (duct | tall)            -> collision, spawn checks
    duct      : low boundary objects (flexible duct hose, height duct_height) -> LiDAR beams pass over them when tilted
    tall      : tall objects (room walls, unknown space, clutter)        -> always block beams
    Cells that are neither are floor (drivable or not).
    props     : placed static obstacles, held analytically rather than rasterised (see StaticProp)."""
    occupancy: np.ndarray            # (H, W) bool
    resolution: float                # [m/cell]
    origin: tuple                    # (x, y) of cell (0, 0) lower-left corner
    edt: np.ndarray                  # (H, W) float32, distance [m] to nearest occupied cell center
    centerline: Optional[np.ndarray] = None   # (N, 2) closed polyline [m]
    name: str = "track"
    duct: Optional[np.ndarray] = None         # (H, W) bool
    tall: Optional[np.ndarray] = None         # (H, W) bool
    edt_duct: Optional[np.ndarray] = None
    edt_tall: Optional[np.ndarray] = None
    duct_height: float = 0.2                  # [m] duct hose diameter
    props: tuple = ()                         # (StaticProp, ...); empty on every existing track

    def __post_init__(self):
        if self.duct is None or self.tall is None:
            self.classify_shell()
        if self.edt_duct is None:
            self.edt_duct = ndimage.distance_transform_edt(~self.duct).astype(np.float32) * self.resolution
        if self.edt_tall is None:
            self.edt_tall = ndimage.distance_transform_edt(~self.tall).astype(np.float32) * self.resolution

    def classify_shell(self, shell_floor_depth: float = 1.0):
        """For maps that only know 'occupied': the occupied shell next to free space (within
        duct_height) is a duct hose; occupied cells deeper than shell_floor_depth are tall
        (unknown space / walls); in between is floor. Beams passing over the duct then travel
        up to shell_floor_depth before hitting something."""
        dist_to_free = ndimage.distance_transform_edt(self.occupancy).astype(np.float32) * self.resolution
        self.duct = self.occupancy & (dist_to_free <= self.duct_height + 0.5 * self.resolution)
        self.tall = self.occupancy & (dist_to_free > shell_floor_depth)
        self.edt_duct = self.edt_tall = None

    # ---- construction -------------------------------------------------------------------
    @staticmethod
    def from_occupancy(occ: np.ndarray, resolution: float, origin=(0.0, 0.0),
                       centerline: Optional[np.ndarray] = None, name="track",
                       duct: Optional[np.ndarray] = None, tall: Optional[np.ndarray] = None,
                       duct_height: float = 0.2) -> "Track":
        occ = occ.astype(bool)
        edt = ndimage.distance_transform_edt(~occ).astype(np.float32) * resolution
        return Track(occ, float(resolution), (float(origin[0]), float(origin[1])), edt, centerline, name,
                     duct=duct, tall=tall, duct_height=duct_height)

    @staticmethod
    def from_ros_map(yaml_path: str, centerline_csv: Optional[str] = None, duct_height: float = 0.33,
                     boundary: str = "duct", unknown_is_obstacle: bool = True, max_cells: float = 4.5e6,
                     name: Optional[str] = None, despeckle_m2: float = 0.03, crop_margin: float = 2.0,
                     unknown_floor_depth: float = 2.0, keep_region: bool = False, seed_xy=None,
                     outer_walls: bool = False) -> "Track":
        """Load a ROS map_server style map (yaml + pgm/png).
        boundary: "duct" -> occupied shell next to free space is a duct hose, deeper is tall
                  (SLAM maps of duct-hose tracks, f1tenth_racetracks drawings);
                  "wall" -> every occupied cell is a tall wall (buildings, hallways).
        unknown_is_obstacle: gray 'unknown' pixels (anything darker than free_min_value that is not
                  occupied: map_saver's 205, f1tenth_gym's 216, anti-aliased edges) count as tall obstacles.
        max_cells: downsample (block max) maps larger than this.
        crop_margin: SLAM canvases are mostly unknown space; crop to the free-space bounding box plus
                  this margin [m] (outside the grid counts as a wall) before any downsampling.
        unknown_floor_depth: unknown space within this distance [m] of the mapped lane is treated as
                  floor for the LiDAR (a beam tilted over the hose sees floor, then something tall
                  further out) while staying solid for collisions; deeper unknown space is tall.
        keep_region: SLAM clean-up: keep only the free region around seed_xy (largest region when
                  None) after a 0.15 m morphological opening that cuts the thin leaks through which
                  scan rays sprayed out of the hall; everything else becomes unknown.
        outer_walls: occupied cells that back onto unknown space are real walls (tall), only occupied
                  cells with free space behind them are duct hoses (fully mapped halls)."""
        import yaml
        from PIL import Image
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img_path = meta["image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(yaml_path), img_path)
        if not os.path.exists(img_path):                      # e.g. yaml says .png, file is .pgm
            base = os.path.splitext(img_path)[0]
            for ext in (".pgm", ".png", ".jpg"):
                if os.path.exists(base + ext):
                    img_path = base + ext; break
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        img = np.flipud(img)                      # row 0 -> bottom (y = origin_y)
        negate = int(meta.get("negate", 0))
        occ_th = float(meta.get("occupied_thresh", 0.65))
        free_th = float(meta.get("free_thresh", 0.196))
        p = img / 255.0 if negate else (255.0 - img) / 255.0   # p = probability occupied
        occ = p > occ_th
        unknown = ~occ & (p >= free_th)
        # gym-style maps mark unknown space with a gray the yaml thresholds would call free (levine: 216):
        # a single gray value covering > 20 % of the image is unknown, whatever the thresholds say
        vals, cnt = np.unique(img.astype(np.uint8), return_counts=True)
        for v, c in zip(vals, cnt):
            if 120 <= v <= 235 and c > 0.2 * img.size:
                unknown |= ~occ & (img == v)
        if unknown_is_obstacle:
            occ = occ | unknown
        res = float(meta["resolution"])
        if despeckle_m2 > 0:                       # SLAM scan noise: isolated specks inside the lane are not obstacles
            lab, n = ndimage.label(occ)
            sizes = np.bincount(lab.ravel())
            small = sizes < int(despeckle_m2 / res ** 2)
            small[0] = False
            occ = occ & ~small[lab]
            unknown = unknown & occ
        origin = list(meta.get("origin", [0.0, 0.0, 0.0]))
        if keep_region:                            # drop scan spray outside the hall and stray free specks
            free = ~occ
            r_open = max(1, int(round(0.15 / res)))
            opened = ndimage.binary_opening(free, structure=np.ones((3, 3), bool), iterations=r_open)
            lab, n = ndimage.label(opened)
            if n > 0:
                if seed_xy is not None:
                    c = int(round((seed_xy[0] - origin[0]) / res)); r = int(round((seed_xy[1] - origin[1]) / res))
                    ok = 0 <= r < lab.shape[0] and 0 <= c < lab.shape[1] and lab[r, c] > 0
                    if not ok:
                        pts = np.argwhere(lab > 0); j = np.argmin(np.hypot(pts[:, 0] - r, pts[:, 1] - c)); r, c = pts[j]
                    comp = lab == lab[r, c]
                else:
                    comp = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
                region = ndimage.binary_dilation(comp, structure=np.ones((3, 3), bool), iterations=r_open + 1) & free
                lost = free & ~region
                occ = occ | lost; unknown = unknown | lost
                lab2, n2 = ndimage.label(occ)          # occupied specks left floating inside the hall
                sizes = np.bincount(lab2.ravel()) * res * res
                small = sizes < max(despeckle_m2, 0.05); small[0] = False
                occ = occ & ~small[lab2]; unknown = unknown & occ
        if crop_margin is not None and unknown_is_obstacle:
            rr, cc = np.nonzero(~occ)
            if rr.size:
                m = int(np.ceil(crop_margin / res))
                r0, r1 = max(rr.min() - m, 0), min(rr.max() + m + 1, occ.shape[0])
                c0, c1 = max(cc.min() - m, 0), min(cc.max() + m + 1, occ.shape[1])
                occ = np.ascontiguousarray(occ[r0:r1, c0:c1]); unknown = np.ascontiguousarray(unknown[r0:r1, c0:c1])
                origin[0] += c0 * res; origin[1] += r0 * res
        f = 1
        while occ.size / (f * f) > max_cells:
            f += 1
        if f > 1:                                  # block-max downsample: conservative obstacles
            H, W = occ.shape
            Hc, Wc = H // f * f, W // f * f
            occ = occ[:Hc, :Wc].reshape(Hc // f, f, Wc // f, f).any(axis=(1, 3))
            unknown = unknown[:Hc, :Wc].reshape(Hc // f, f, Wc // f, f).any(axis=(1, 3))
            res *= f
        cl = None
        if centerline_csv is None:
            base = os.path.splitext(yaml_path)[0]
            for cand in (base + "_centerline.csv", base.replace("_map", "") + "_centerline.csv"):
                if os.path.exists(cand):
                    centerline_csv = cand; break
        if centerline_csv is not None:
            cl = load_centerline_csv(centerline_csv)
        name = name or os.path.splitext(os.path.basename(yaml_path))[0].replace("_map", "")
        duct = tall = None
        if boundary == "wall":
            tall = occ.copy(); duct = np.zeros_like(occ)
        elif boundary == "duct" and unknown_is_obstacle:
            # unknown space is never a duct hose: force it tall, classify the rest as a shell
            t = Track.from_occupancy(occ, res, (origin[0], origin[1]), cl, name, duct_height=duct_height)
            # unknown space comes in two kinds: the outside (never seen, one big region) and pockets
            # inside mapped objects (the unseen interior of a hose or box): pockets belong to the object
            # (a pocket may leak into the outside through a gap in the object's outline, so connectivity
            # is not enough: pockets are the small unknown patches inside the convex hull of the free space)
            from skimage.morphology import convex_hull_image
            hull = convex_hull_image(~occ) if (~occ).any() else np.zeros_like(occ)
            lab_u, n_u = ndimage.label(unknown & hull)
            areas = np.bincount(lab_u.ravel()) * res * res; areas[0] = np.inf
            pocket = unknown & hull & (areas[lab_u] < 3.0)
            outside = unknown & ~pocket
            d2free = ndimage.distance_transform_edt(occ).astype(np.float32) * res
            t.tall = (t.tall & ~unknown) | (outside & (d2free > unknown_floor_depth))
            t.duct = (t.duct & ~outside) | (pocket & ~t.tall)
            if outer_walls:                        # mapped walls back onto the outside; hoses have floor behind
                near_out = ndimage.binary_dilation(outside, iterations=max(1, int(round((duct_height + 0.1) / res))))
                wall = t.duct & near_out
                t.tall = t.tall | wall
                t.duct = t.duct & ~wall
            t.edt_duct = ndimage.distance_transform_edt(~t.duct).astype(np.float32) * res
            t.edt_tall = ndimage.distance_transform_edt(~t.tall).astype(np.float32) * res
            return t
        return Track.from_occupancy(occ, res, (origin[0], origin[1]), cl, name, duct=duct, tall=tall,
                                    duct_height=duct_height)

    def with_pinches(self, seed: int = 0, n: int = 3, keep: float = 0.45, run: float = 1.6) -> "Track":
        """Copy of the track with the boundary pushed inward at a few places along the lane.

        Measured against the real venues, every generator holds its width almost constant: the ratio
        of the local width to the narrowest spot near it runs 1.04-1.05 procedurally against 1.98 on
        the blackbox maps, where a 3 m section closes to 1.5 m and opens again. That is a different
        problem from a narrow track -- the car arrives carrying speed the gap will not take -- and
        nothing in the catalogue trains it except a handful of real maps.

        `keep` is the fraction of the local width left at the tightest point; `run` how many metres
        the squeeze extends over. The lane is never taken below the car's width plus a margin.

        0.45 rather than 0.55: with 0.55 the generated pinch ratio reached 1.53-1.65 while
        blackbox2022_3 -- the one held-out map the student still fails on, at 9-15 collisions/km
        against 0.1-5 everywhere else -- sits at the top of the measured range. Training against a
        milder version of the axis than the test set holds is how that gap stays open.
        """
        if self.centerline is None:
            raise ValueError("pinches need a centerline")
        rng = np.random.default_rng(seed + 4231)
        cl = self.centerline
        edt = ndimage.distance_transform_edt(~self.occupancy).astype(np.float32) * self.resolution
        occ = self.occupancy.copy(); duct = self.duct.copy() if self.duct is not None else None
        tall = self.tall.copy() if self.tall is not None else None
        H, W = occ.shape
        xs = np.arange(W) * self.resolution + self.origin[0]
        ys = np.arange(H) * self.resolution + self.origin[1]
        seg = np.linalg.norm(np.roll(cl, -1, 0) - cl, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        placed = []
        for _ in range(n * 12):
            if len(placed) >= n:
                break
            i = int(rng.integers(len(cl)))
            if any(abs(s[i] - s[j]) < 4.0 for j in placed):
                continue
            j = int(round((cl[i, 0] - self.origin[0]) / self.resolution))
            k = int(round((cl[i, 1] - self.origin[1]) / self.resolution))
            if not (0 <= j < W and 0 <= k < H):
                continue
            half = float(edt[k, j])
            target = max(0.55, half * keep)                 # never below the car plus a margin
            if half - target < 0.12:
                continue
            side = 1.0 if rng.uniform() < 0.5 else -1.0
            d = np.roll(cl, -1, 0) - np.roll(cl, 1, 0)
            t_ = d[i] / (np.linalg.norm(d[i]) + 1e-9)
            nrm = np.array([-t_[1], t_[0]]) * side
            # a smooth bump of blocked cells hugging one wall, tapering over `run` metres
            c = cl[i] + nrm * (half + target) / 2
            rad = (half - target) / 2 + 0.05
            gx, gy = np.meshgrid(xs, ys)
            along = ((gx - cl[i, 0]) * t_[0] + (gy - cl[i, 1]) * t_[1])
            across = ((gx - c[0]) * nrm[0] + (gy - c[1]) * nrm[1])
            taper = np.clip(1.0 - (np.abs(along) / (run / 2)) ** 2, 0.0, 1.0)
            hit = (np.abs(across) < rad * taper) & (np.abs(along) < run / 2)
            if not hit.any():
                continue
            # The raceline optimiser searches a corridor anchored to the centerline, so a pinch that
            # covers the centerline leaves no corridor at all and the line is returned running
            # straight through the blockage -- which then teaches the student to drive into it.
            # Keep the centerline itself clear by the car's half-width plus a margin.
            keep_free = np.zeros_like(occ)
            cj = np.clip(((cl[:, 0] - self.origin[0]) / self.resolution).astype(int), 0, W - 1)
            ci = np.clip(((cl[:, 1] - self.origin[1]) / self.resolution).astype(int), 0, H - 1)
            keep_free[ci, cj] = True
            keep_free = ndimage.binary_dilation(keep_free, iterations=int(round(0.28 / self.resolution)))
            if (hit & keep_free).any():
                continue
            occ |= hit
            if duct is not None: duct |= hit
            placed.append(i)
        t = Track.from_occupancy(occ, self.resolution, self.origin, self.centerline,
                                 f"{self.name}_pinch{seed}", duct=duct, tall=tall,
                                 duct_height=self.duct_height)
        t.props = self.props                       # placements survive a grid edit
        return t

    def with_lane_obstacles(self, seed: int = 0, n: int = 3, size=(0.25, 0.5), min_passage: float = 1.2,
                            min_spacing: float = 4.0, kind: str = "box") -> "Track":
        """Copy of the track with n tall boxes (competition 'static obstacles') dropped into the lane,
        placed against one side so at least min_passage of lane stays open, min_spacing apart."""
        rng = np.random.default_rng(seed)
        if self.centerline is None:
            raise ValueError("lane obstacles need a centerline")
        cl = self.centerline; N = len(cl)
        tang = np.roll(cl, -1, 0) - np.roll(cl, 1, 0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        tall = self.tall.copy(); occ = self.occupancy.copy()
        H, W = occ.shape
        xs = np.arange(W) * self.resolution + self.origin[0]; ys = np.arange(H) * self.resolution + self.origin[1]
        placed = []; tries = 0
        while len(placed) < n and tries < 200:
            tries += 1
            i = int(rng.integers(N)); side = rng.choice([-1.0, 1.0])
            if any(min(abs(i - j), N - abs(i - j)) * (np.linalg.norm(np.roll(cl, -1, 0) - cl, axis=1).mean()) < min_spacing for j in placed):
                continue
            # clearance on that side of the centerline (distance from the centerline point to the wall)
            c0 = int(round((cl[i, 0] - self.origin[0]) / self.resolution)); r0 = int(round((cl[i, 1] - self.origin[1]) / self.resolution))
            if not (0 <= r0 < H and 0 <= c0 < W): continue
            half_w = float(self.edt[r0, c0])                      # ~ half the lane width here
            sx, sy = rng.uniform(*size), rng.uniform(*size)
            if 2 * half_w - sy < min_passage + 0.1:                # lane too narrow for an obstacle
                continue
            off = side * (half_w - sy / 2 - 0.05)                 # hug one side
            cx, cy = cl[i] + nrm[i] * off
            c1 = int(round((cx - self.origin[0]) / self.resolution)); r1 = int(round((cy - self.origin[1]) / self.resolution))
            if not (0 <= r1 < H and 0 <= c1 < W) or occ[r1, c1]: continue
            cc0, cc1 = max(0, int((cx - 1 - self.origin[0]) / self.resolution)), min(W, int((cx + 1 - self.origin[0]) / self.resolution) + 1)
            rr0, rr1 = max(0, int((cy - 1 - self.origin[1]) / self.resolution)), min(H, int((cy + 1 - self.origin[1]) / self.resolution) + 1)
            gx, gy = np.meshgrid(xs[cc0:cc1], ys[rr0:rr1])
            # box aligned with the lane direction
            dx, dy = gx - cx, gy - cy
            u = dx * tang[i, 0] + dy * tang[i, 1]; v = dx * nrm[i, 0] + dy * nrm[i, 1]
            m = (np.abs(u) <= sx / 2) & (np.abs(v) <= sy / 2) if kind == "box" else (u ** 2 + v ** 2 <= (sx / 2) ** 2)
            tall[rr0:rr1, cc0:cc1] |= m; occ[rr0:rr1, cc0:cc1] |= m; placed.append(i)
        t = Track.from_occupancy(occ, self.resolution, self.origin, cl, f"{self.name}_obs{seed}", duct=self.duct, tall=tall,
                                 duct_height=self.duct_height)
        t.props = self.props                       # placements survive a grid edit
        return t

    def with_static_props(self, seed: int = 0, n: int = 3, styles=None, min_passage: float = 1.2,
                          min_spacing: float = 4.0, line: Optional[np.ndarray] = None) -> "Track":
        """Copy of the track carrying n placed props (see StaticProp). Grids are untouched.

        Placement mirrors `with_lane_obstacles` -- hug one side of the lane, leave `min_passage`
        open, keep `min_spacing` between props -- but the clearance test uses the prop's own
        circumradius from its declared footprint rather than a drawn box size, and nothing is
        stamped into `occupancy`/`tall`. A prop that will not fit is skipped and counted; it is never
        placed and silently left without physics."""
        from . import props as _props
        if styles is None:
            styles = _props.STYLES
        rng = np.random.default_rng(seed)
        cl = self.centerline if line is None else np.asarray(line, float)
        if cl is None:
            raise ValueError("static props need a centerline or an explicit line")
        N = len(cl)
        tang = np.roll(cl, -1, 0) - np.roll(cl, 1, 0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        ds = float(np.linalg.norm(np.roll(cl, -1, 0) - cl, axis=1).mean())
        placed, out, skipped, tries = [], [], 0, 0
        while len(out) < n and tries < 200:
            tries += 1
            i = int(rng.integers(N))
            if any(min(abs(i - j), N - abs(i - j)) * ds < min_spacing for j in placed):
                continue
            style = str(styles[int(rng.integers(len(styles)))])
            sp = StaticProp(style, 0.0, 0.0, 0.0, seed=int(rng.integers(1 << 30)))
            env = sp.build().envelope
            r = float(env.radius)
            c0 = int(round((cl[i, 0] - self.origin[0]) / self.resolution))
            r0 = int(round((cl[i, 1] - self.origin[1]) / self.resolution))
            if not (0 <= r0 < self.occupancy.shape[0] and 0 <= c0 < self.occupancy.shape[1]):
                continue
            half_w = float(self.edt[r0, c0])
            if 2 * half_w - 2 * r < min_passage + 0.1:            # lane too narrow for this prop
                skipped += 1
                continue
            side = float(rng.choice([-1.0, 1.0]))
            off = side * (half_w - r - 0.05)
            cx, cy = cl[i] + nrm[i] * off
            cc = int(round((cx - self.origin[0]) / self.resolution))
            rr = int(round((cy - self.origin[1]) / self.resolution))
            if not (0 <= rr < self.occupancy.shape[0] and 0 <= cc < self.occupancy.shape[1]):
                continue
            if self.occupancy[rr, cc] or float(self.edt[rr, cc]) < r:
                skipped += 1
                continue
            yaw = float(math.atan2(tang[i, 1], tang[i, 0]) + rng.uniform(-0.35, 0.35))
            out.append(StaticProp(style, float(cx), float(cy), yaw, seed=sp.seed))
            placed.append(i)
        t = Track.from_occupancy(self.occupancy, self.resolution, self.origin, self.centerline,
                                 f"{self.name}_props{seed}", duct=self.duct, tall=self.tall,
                                 duct_height=self.duct_height)
        t.edt_duct, t.edt_tall = self.edt_duct, self.edt_tall
        t.props = tuple(out)
        if skipped:
            t.props_skipped = skipped
        return t

    def free_width_along(self, point: np.ndarray, direction: np.ndarray, max_m: float = 6.0) -> float:
        """Distance from `point` along `direction` to the first occupied cell."""
        step = self.resolution * 0.5
        H, W = self.occupancy.shape
        for k in range(1, int(max_m / step)):
            q = point + direction * (k * step)
            c = int(round((q[0] - self.origin[0]) / self.resolution))
            r = int(round((q[1] - self.origin[1]) / self.resolution))
            if not (0 <= r < H and 0 <= c < W) or self.occupancy[r, c]:
                return k * step
        return max_m

    def with_line_obstacles(self, line: np.ndarray, seed: int = 0, n: int = 3, size=(0.25, 0.5),
                            min_passage: float = 1.0, min_spacing: float = 4.0,
                            lateral_jitter: float = 0.15, kind: str = "box",
                            speeds: Optional[np.ndarray] = None,
                            sight_bands=((0.0, 5.0), (5.0, 9.0), (9.0, 14.0), (14.0, 999.0))) -> "Track":
        """Boxes standing ON a given line (normally the raceline), not against the wall.

        `with_lane_obstacles` hugs one side so at least `min_passage` stays open *and the fast line
        stays clear*: the car can ignore the box. A competition box parked on the racing line is the
        case that actually forces a deviation, which is what makes it a test of seeing and planning
        rather than of staying on a memorised line. Placement still leaves `min_passage` free on one
        side, so every obstacle is passable and the track stays drivable.

        Situations, not counts. Drawing positions uniformly along the line puts every box where the
        track is open, because most of a track is: over the 84 boxes this used to produce, two thirds
        sat where the car could see more than 9 m ahead and none where it could see under 5. Boxes
        are drawn round-robin across `sight_bands` -- how many metres of view the approach gives --
        so the short-view cases are there by construction rather than by luck.

        `speeds`: speed profile matching `line`, kept for callers that want it; the bands themselves
        are distance, for the reason in the comment below."""
        rng = np.random.default_rng(seed)
        line = resample_closed(np.asarray(line, dtype=float), max(len(line), 400))
        N = len(line)
        tang = np.roll(line, -1, 0) - np.roll(line, 1, 0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        ds = float(np.linalg.norm(np.roll(line, -1, 0) - line, axis=1).mean())
        tall = self.tall.copy(); occ = self.occupancy.copy()
        H, W = occ.shape
        xs = np.arange(W) * self.resolution + self.origin[0]
        ys = np.arange(H) * self.resolution + self.origin[1]
        keep_free = np.zeros_like(occ)                   # cells the centerline needs, dilated by the car
        if self.centerline is not None:
            cj = np.clip(((self.centerline[:, 0] - self.origin[0]) / self.resolution).astype(int), 0, W - 1)
            ci = np.clip(((self.centerline[:, 1] - self.origin[1]) / self.resolution).astype(int), 0, H - 1)
            keep_free[ci, cj] = True
            keep_free = ndimage.binary_dilation(keep_free, iterations=int(round(0.28 / self.resolution)))
        def _sight(i_):
            """How far back along the line the point at i_ is still in view [m]."""
            out = 0.0
            for back in range(1, 260):
                a_ = line[(i_ - back) % N]; b_ = line[i_]
                d_ = b_ - a_
                k_ = max(2, int(np.linalg.norm(d_) / (self.resolution * 0.7)))
                q = a_[None] + np.linspace(0.0, 1.0, k_)[:, None] * d_[None]
                cj_ = np.clip(((q[:, 0] - self.origin[0]) / self.resolution).astype(int), 0, W - 1)
                ci_ = np.clip(((q[:, 1] - self.origin[1]) / self.resolution).astype(int), 0, H - 1)
                if occ0[ci_[1:-1], cj_[1:-1]].any():
                    break
                out = back * ds
            return out

        # Stratify on sight distance in metres, not on reaction time.
        #
        # Reaction time -- sight distance over the speed carried there -- is the number that decides
        # how hard an obstacle is, but the map cannot set it: the *policy* picks the speed. The
        # raceline profile that positions these boxes has already slowed for anywhere the view is
        # short (a corner exit is blind but also slow), so stratifying on reaction time returned
        # nothing under 0.7 s no matter how the candidates were drawn. Screens set beside the lane
        # were tried and measured: they move sight distance by 0.4-1.5 m, because a block hugging the
        # wall does not cross a sight line that runs along the corridor, and one that did would block
        # the passage.
        #
        # What the map can set is how many metres of view the car gets. A box 4 m beyond a corner
        # exit is an easy problem at the teacher's 3 m/s and a hard one at 8, so spreading over sight
        # distance is what puts the hard cases in reach as PPO raises the cap.
        occ0 = self.occupancy                             # sight against the bare track, not the boxes
        v_line = (np.asarray(speeds, float) if speeds is not None and len(speeds) == N
                  else np.full(N, 4.0))
        step = max(1, N // 90)                            # sample candidates, scoring is the expensive part
        cand = list(range(0, N, step))
        sight = {i_: _sight(i_) for i_ in cand}
        bands = [[i_ for i_ in cand if lo <= sight[i_] < hi] for lo, hi in sight_bands]
        for b in bands:
            rng.shuffle(b)
        order, b_i = [], 0
        while any(bands) and len(order) < 400:            # round-robin: every band contributes
            b = bands[b_i % len(bands)]
            if b:
                order.append(b.pop())
            b_i += 1
        order += [int(rng.integers(N)) for _ in range(200)]      # fall back to anywhere

        placed = []
        for i in order:
            if len(placed) >= n:
                break
            if any(min(abs(i - j), N - abs(i - j)) * ds < min_spacing for j in placed):
                continue
            sx, sy = rng.uniform(*size), rng.uniform(*size)
            jitter = rng.uniform(-lateral_jitter, lateral_jitter)
            centre = line[i] + nrm[i] * jitter
            w_left = self.free_width_along(line[i], nrm[i])
            w_right = self.free_width_along(line[i], -nrm[i])
            # the box spans [jitter - sy/2, jitter + sy/2] laterally; one side must stay passable
            if max(w_left - (jitter + sy / 2), w_right + (jitter - sy / 2)) < min_passage:
                continue
            c1 = int(round((centre[0] - self.origin[0]) / self.resolution))
            r1 = int(round((centre[1] - self.origin[1]) / self.resolution))
            if not (0 <= r1 < H and 0 <= c1 < W) or occ[r1, c1]:
                continue
            cc0 = max(0, int((centre[0] - 1 - self.origin[0]) / self.resolution))
            cc1 = min(W, int((centre[0] + 1 - self.origin[0]) / self.resolution) + 1)
            rr0 = max(0, int((centre[1] - 1 - self.origin[1]) / self.resolution))
            rr1 = min(H, int((centre[1] + 1 - self.origin[1]) / self.resolution) + 1)
            gx, gy = np.meshgrid(xs[cc0:cc1], ys[rr0:rr1])
            dx, dy = gx - centre[0], gy - centre[1]
            u = dx * tang[i, 0] + dy * tang[i, 1]; v = dx * nrm[i, 0] + dy * nrm[i, 1]
            m = (np.abs(u) <= sx / 2) & (np.abs(v) <= sy / 2) if kind == "box" else (u ** 2 + v ** 2 <= (sx / 2) ** 2)
            # A box on the *raceline* may still sit over the *centerline*, and the raceline optimiser
            # searches a corridor anchored to the centerline: cover that and there is no corridor
            # left, so the line comes back running straight through the box. Measured on the first
            # attempt at this set, 24 of 180 tracks had a raceline with zero clearance for exactly
            # this reason -- tracks that would have taught the student to drive into obstacles.
            if self.centerline is not None:
                blocked = m & keep_free[rr0:rr1, cc0:cc1]
                if blocked.any():
                    continue
            tall[rr0:rr1, cc0:cc1] |= m; occ[rr0:rr1, cc0:cc1] |= m
            placed.append(i)
        t = Track.from_occupancy(occ, self.resolution, self.origin, self.centerline,
                                 f"{self.name}_rlobs{seed}", duct=self.duct, tall=tall,
                                 duct_height=self.duct_height)
        t.props = self.props                       # placements survive a grid edit
        return t

    def mirrored(self) -> "Track":
        """Mirror the map about the vertical axis: a counter-clockwise loop becomes clockwise.
        Every procedural generator traces its loop counter-clockwise, so without this the policy
        would only ever see left-turning tracks."""
        H, W = self.occupancy.shape
        occ, duct, tall = self.occupancy[:, ::-1].copy(), self.duct[:, ::-1].copy(), self.tall[:, ::-1].copy()
        cl = None
        if self.centerline is not None:
            x_lo, x_hi = self.origin[0], self.origin[0] + (W - 1) * self.resolution
            cl = self.centerline.copy(); cl[:, 0] = x_lo + x_hi - cl[:, 0]
        t = Track.from_occupancy(occ, self.resolution, self.origin, cl, self.name + "m", duct=duct, tall=tall,
                                 duct_height=self.duct_height)
        # Props are placements, so mirroring the map has to mirror them too -- carrying them through
        # unchanged would leave a box floating where the lane no longer is, and dropping them would
        # make `rt:Monza+props3~mir` quietly a different track from `rt:Monza+props3`.
        if self.props:
            x_lo = self.origin[0]
            x_hi = self.origin[0] + (W - 1) * self.resolution
            t.props = tuple(StaticProp(p.style, x_lo + x_hi - p.x, p.y, math.pi - p.yaw, p.seed, p.dims)
                            for p in self.props)
        return t

    def grid_key(self):
        """Content hash of the obstacle layers (tracks that differ only in centerline share GPU grids).

        Every layer the LiDAR traces has to be in here. `tall` was missing, and `TrackTensors` skips
        appending edt/edt_duct/edt_tall entirely on a key hit, so two tracks with the same occupancy
        and duct but different tall silently shared the first one's tall field and the second was
        traced against geometry it does not have. That is reachable from the public constructor:
        `from_occupancy` takes `duct` and `tall` as independent arrays, and `from_ros_map`'s
        unknown-floor handling moves `tall` without touching occupancy or duct. The two obstacle
        builders happen to write `tall` and `occupancy` together, which is why it went unnoticed.
        """
        import hashlib
        h = hashlib.md5(np.packbits(self.occupancy).tobytes())
        h.update(np.packbits(self.duct).tobytes())
        h.update(np.packbits(self.tall).tobytes())
        return (self.occupancy.shape, round(self.resolution, 6), tuple(np.round(self.origin, 4)), float(self.duct_height), h.hexdigest())

    def reversed(self) -> "Track":
        """Same map, lap driven the other way round (centerline reversed). Grids are shared, not
        copied, so TrackTensors keeps a single GPU copy for both directions."""
        cl = None if self.centerline is None else np.ascontiguousarray(self.centerline[::-1])
        # Props stand where they stand: driving the lap the other way round does not move them.
        return Track(self.occupancy, self.resolution, self.origin, self.edt, cl, self.name + "r",
                     duct=self.duct, tall=self.tall, edt_duct=self.edt_duct, edt_tall=self.edt_tall,
                     duct_height=self.duct_height, props=self.props)

    @staticmethod
    def generate_random(seed: int = 0, style: str = "competition", resolution: float = 0.05, mirror="auto",
                        lane_obstacles: bool | int = False, **kw) -> "Track":
        """Procedural tracks. style:
        "competition": competition-hall layout (_gen_grid): the outline of a random blob of cells on a
                       2.5-3.3 m grid, so axis-aligned straights and rounded 90 deg corners, duct-hose
                       boundaries in a room with clutter (indoor RoboRacer/F1TENTH events). This is a
                       narrow parametric family -- more seeds add little new geometry
        "control":     control-point loop with hairpins, chicanes and varying width (1.6-2.6 m)
        "circuit":     smooth Fourier loop, wide, duct boundaries (the old generator)
        "hallway":     rectangular building corridor loop with 90 deg corners, tall walls, clutter
                       (Levine-style venues)
        "serpentine":  a hall with straight walls cut across it, so the lap folds back beside itself
                       behind a single hose -- the layout every real venue in the catalogue has and
                       no other generator produces
        mirror: True/False, or "auto" = odd seeds are mirrored (clockwise) so both turn directions
        appear equally often."""
        if style != "hallway":                      # duct-hose venues: 33 cm hoses laid in segments with gaps
            rng = np.random.default_rng(seed + 11)  # (CDC 2025), banner boards around some venues
            kw.setdefault("duct_height", 0.33)
            kw.setdefault("duct_gaps", 0.12)
            kw.setdefault("banner_fence", 0.0 if rng.uniform() < 0.4 else float(rng.uniform(1.0, 3.0)))
        # NOTE the names: "competition" is the grid/blob generator (_gen_grid), and the older
        # control-point ellipse generator (_gen_competition) is reachable as "control". Keeping the
        # mapping explicit -- it used to be an unlabelled else branch, which reads as a bug.
        builders = {"circuit": Track._gen_circuit, "hallway": Track._gen_hallway,
                    "control": Track._gen_competition, "competition": Track._gen_grid,
                    "serpentine": Track._gen_serpentine}
        if style not in builders:
            raise ValueError(f"unknown style {style!r}; expected one of {sorted(builders)}")
        t = builders[style](seed, resolution, **kw)
        if mirror == "auto":
            mirror = (seed % 2 == 1)                # odd seeds run clockwise
        t = t.mirrored() if mirror else t
        if lane_obstacles:
            n_obs = int(np.random.default_rng(seed + 7).integers(1, 5)) if lane_obstacles is True else int(lane_obstacles)
            t = t.with_lane_obstacles(seed=seed, n=n_obs)
        return t

    @staticmethod
    def from_waypoints(points, width=1.6, resolution=0.05, seed: int = 0, duct_height: float = 0.33, room_margin: float = 3.0,
                       n_clutter: int = 8, smooth: int = 2, name: str = "sketch", boundary: str = "duct",
                       duct_gaps: float = 0.0, banner_fence: float = 0.0) -> "Track":
        """Track from a hand-drawn closed centerline (list of (x, y) in metres, e.g. transcribed from a
        photo or sketch of a competition layout). Corners are rounded (Chaikin), the lane has the
        given width (float or per-point array). duct_gaps: probability per metre of a 0.15-0.35 m
        gap in the hose (LiDAR beams pass through, CDC 2025 rules). banner_fence: distance [m] of a
        tall banner-board fence around the track (0 = none), like the Korea venues."""
        rng = np.random.default_rng(seed)
        pts = _chaikin(_densify(np.asarray(points, float), 0.25), smooth)
        pts = resample_closed(pts, 800)
        w = np.full(len(pts), float(width)) if np.isscalar(width) else np.interp(np.linspace(0, 1, len(pts), endpoint=False), np.linspace(0, 1, len(width), endpoint=False), width)
        t = Track._rasterize_loop(pts, w, resolution, room_margin, duct_height, rng, n_clutter, (0.3, 1.0), boundary=boundary,
                                  name=name, duct_gaps=duct_gaps, banner_fence=banner_fence)
        return t

    @staticmethod
    def _rasterize_loop(pts, w, resolution, margin, duct_height, rng, n_clutter, clutter_size, boundary="duct",
                        wall_thickness=0.15, name="track", duct_gaps: float = 0.0, banner_fence: float = 0.0):
        """Free = union of discs along the loop; boundaries as ducts or walls; room + clutter."""
        lo = pts.min(0) - (w.max() + margin); hi = pts.max(0) + (w.max() + margin)
        W = int(math.ceil((hi[0] - lo[0]) / resolution)); H = int(math.ceil((hi[1] - lo[1]) / resolution))
        free = np.zeros((H, W), dtype=bool)
        xs = np.arange(W) * resolution + lo[0]; ys = np.arange(H) * resolution + lo[1]
        for (px, py), pw in zip(pts, w):
            rad = pw / 2
            c0 = max(0, int((px - rad - lo[0]) / resolution) - 1); c1 = min(W, int((px + rad - lo[0]) / resolution) + 2)
            r0 = max(0, int((py - rad - lo[1]) / resolution) - 1); r1 = min(H, int((py + rad - lo[1]) / resolution) + 2)
            gx, gy = np.meshgrid(xs[c0:c1], ys[r0:r1])
            free[r0:r1, c0:c1] |= ((gx - px) ** 2 + (gy - py) ** 2) <= rad ** 2
        dist_to_free = ndimage.distance_transform_edt(~free) * resolution
        if boundary == "duct":
            duct = ~free & (dist_to_free <= duct_height)
            tall = np.zeros((H, W), dtype=bool)
            if duct_gaps > 0:                       # gaps between hose segments: beams pass, car still can't (kept in occupancy? no: real gaps are drivable-through too -> remove)
                lab, n = ndimage.label(duct)
                for k in range(1, n + 1):
                    comp = np.argwhere(lab == k)
                    length_m = len(comp) * resolution ** 2 / duct_height
                    for _ in range(rng.poisson(duct_gaps * length_m)):
                        c = comp[int(rng.integers(len(comp)))]
                        r_gap = rng.uniform(0.15, 0.35) / 2
                        gy, gx = np.ogrid[:H, :W]
                        duct &= ~(((gx - c[1]) * resolution) ** 2 + ((gy - c[0]) * resolution) ** 2 <= r_gap ** 2)
            if banner_fence > 0:                    # tall banner boards around the track area
                band = (dist_to_free > banner_fence) & (dist_to_free <= banner_fence + 0.05)
                tall |= band
        else:                                       # building: walls right at the free region, unknown beyond
            duct = np.zeros((H, W), dtype=bool)
            tall = ~free & (dist_to_free <= wall_thickness)
            tall |= dist_to_free > wall_thickness + 0.6      # unknown space behind the walls
        wall_cells = max(1, int(round(0.15 / resolution)))
        tall[:wall_cells, :] = tall[-wall_cells:, :] = tall[:, :wall_cells] = tall[:, -wall_cells:] = True
        gx, gy = np.meshgrid(xs, ys)
        placed = 0; tries = 0
        while placed < n_clutter and tries < 300:
            tries += 1
            cx, cy = rng.uniform(lo[0] + 0.5, hi[0] - 0.5), rng.uniform(lo[1] + 0.5, hi[1] - 0.5)
            sx, sy = rng.uniform(*clutter_size), rng.uniform(*clutter_size)
            col = int((cx - lo[0]) / resolution); row = int((cy - lo[1]) / resolution)
            if boundary == "duct":
                if dist_to_free[row, col] < duct_height + 0.4 + max(sx, sy): continue
            else:                                   # inside the corridor along the walls, leave >= 1.2 m passage
                if free[row, col] is False or dist_to_free[row, col] > 0: continue
                d_wall = ndimage.distance_transform_edt(free)[row, col] * resolution
                if d_wall > 0.45 or w.min() - max(sx, sy) < 1.2: continue
                sx, sy = min(sx, 0.5), min(sy, 0.5)
            m = ((np.abs(gx - cx) <= sx / 2) & (np.abs(gy - cy) <= sy / 2)) if rng.random() < 0.6 \
                else ((gx - cx) ** 2 + (gy - cy) ** 2 <= (sx / 2) ** 2)
            tall |= m; placed += 1
        return Track.from_occupancy(duct | tall, resolution, (lo[0], lo[1]), pts, name=name,
                                    duct=duct, tall=tall, duct_height=duct_height)

    @staticmethod
    def _gen_circuit(seed=0, resolution=0.05, n_harmonics=4, base_radius=8.0, amplitude=0.35, width=2.2,
                     width_var=0.4, room_margin=4.0, n_points=800, duct_height=0.2, n_clutter=12,
                     clutter_size=(0.3, 1.2),
                  duct_gaps=0.0, banner_fence=0.0):
        rng = np.random.default_rng(seed)
        theta = np.linspace(0, 2 * math.pi, 2000, endpoint=False)
        r = np.full_like(theta, base_radius)
        for k in range(2, 2 + n_harmonics):
            r += rng.uniform(-amplitude, amplitude) * base_radius / k * np.cos(k * theta + rng.uniform(0, 2 * math.pi))
        pts = resample_closed(np.stack([r * np.cos(theta), r * np.sin(theta)], 1), n_points)
        w = width + width_var * _smooth_noise(rng, len(pts))
        return Track._rasterize_loop(pts, w, resolution, room_margin, duct_height, rng, n_clutter, clutter_size,
                                     name=f"circuit_{seed}", duct_gaps=duct_gaps, banner_fence=banner_fence)

    @staticmethod
    def _gen_competition(seed=0, resolution=0.05, n_control=None, radius=None, stretch=None, width=None,
                         width_var=0.5, min_width=1.5, r_min=0.7, n_chicanes=None, room_margin=3.5,
                         n_points=800, duct_height=0.2, n_clutter=10, clutter_size=(0.3, 1.2), max_tries=40,
                  duct_gaps=0.0, banner_fence=0.0):
        """Random control points around an ellipse, midpoint displacement, corner smoothing with a
        minimum turn radius, optional chicanes. Rejects self-intersecting layouts."""
        rng = np.random.default_rng(seed)
        for attempt in range(max_tries):
            n_c = n_control or int(rng.integers(7, 13))
            R = radius or rng.uniform(5.0, 9.0)
            st = stretch or rng.uniform(1.0, 1.8)
            ang = np.sort(rng.uniform(0, 2 * math.pi, n_c))
            ang += rng.uniform(0, 2 * math.pi)
            rad = R * rng.uniform(0.55, 1.0, n_c)
            ctrl = np.stack([rad * np.cos(ang) * st, rad * np.sin(ang)], 1)
            # midpoint displacement (adds hairpins / kinks)
            pts = []
            for i in range(n_c):
                a, b = ctrl[i], ctrl[(i + 1) % n_c]
                mid = (a + b) / 2; d = b - a; nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
                pts += [a, mid + nrm * rng.uniform(-0.45, 0.45) * np.linalg.norm(d)]
            pts = np.asarray(pts)
            pts = _chaikin(pts, 4)
            pts = resample_closed(pts, n_points)
            pts = _limit_curvature(pts, r_min)
            pts = resample_closed(pts, n_points)
            # chicanes: lateral S-bumps on the straighter parts
            nch = n_chicanes if n_chicanes is not None else int(rng.integers(0, 3))
            kap = np.abs(curvature_np(pts))
            for _ in range(nch):
                straight = np.where(kap < 0.15)[0]
                if len(straight) < 60: break
                i0 = int(rng.choice(straight)); L = int(rng.integers(50, 90))
                idx = (np.arange(L) + i0) % len(pts)
                tang = np.roll(pts, -1, 0) - np.roll(pts, 1, 0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
                nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
                bump = rng.uniform(0.5, 0.9) * np.sin(np.linspace(0, 2 * math.pi, L)) * np.sin(np.linspace(0, math.pi, L))
                pts[idx] += nrm[idx] * bump[:, None]
            pts = resample_closed(pts, n_points)
            pts = _limit_curvature(pts, r_min * 0.9)
            pts = resample_closed(pts, n_points)
            wbase = width or rng.uniform(1.7, 2.6)
            w = np.maximum(min_width, wbase + width_var * _smooth_noise(rng, len(pts)))
            if _self_clearance_ok(pts, w.max() + 0.5):
                break
        return Track._rasterize_loop(pts, w, resolution, room_margin, duct_height, rng, n_clutter, clutter_size,
                                     name=f"control_{seed}", duct_gaps=duct_gaps, banner_fence=banner_fence)

    @staticmethod
    def _gen_grid(seed=0, resolution=0.05, grid=None, pitch=None, width=None, fill=None, r_min=0.7,
                  room_margin=3.0, n_points=800, duct_height=0.2, n_clutter=8, clutter_size=(0.3, 1.2),
                  max_tries=30,
                  duct_gaps=0.0, banner_fence=0.0):
        """Competition-hall layout: the outline of a random simply-connected blob of cells on a coarse
        grid (pitch = lane spacing) is the centerline. Gives long straights along the blob edges,
        90-deg corners rounded to arcs, hairpins at one-cell peninsulas, and folded sections where
        two lanes run side by side separated only by the duct hose (the LiDAR sees the other lane
        over the hose when the car rolls)."""
        rng = np.random.default_rng(seed)
        for attempt in range(max_tries):
            gw, gh = grid or (int(rng.integers(5, 9)), int(rng.integers(4, 7)))
            p = pitch or rng.uniform(2.5, 3.3)
            w = width or rng.uniform(1.6, min(2.4, p - 0.7))
            target = fill or rng.uniform(0.35, 0.6)
            blob = np.zeros((gh, gw), dtype=bool)
            cy, cx = gh // 2, gw // 2
            blob[cy, cx] = True
            n_target = max(4, int(target * gw * gh))
            frontier = [(cy, cx)]
            while blob.sum() < n_target and frontier:
                y, x = frontier[int(rng.integers(len(frontier)))]
                nb = [(y + dy, x + dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if 0 <= y + dy < gh and 0 <= x + dx < gw and not blob[y + dy, x + dx]]
                if not nb:
                    frontier.remove((y, x)); continue
                yy, xx = nb[int(rng.integers(len(nb)))]
                blob[yy, xx] = True; frontier.append((yy, xx))
            blob = ndimage.binary_fill_holes(blob)
            if blob.sum() < 4:
                continue
            # outline polygon of the cell union (traced in order) -> centerline
            outline = _cell_union_outline(blob, p)
            if outline is None:
                continue
            pts = _chaikin(_densify(outline, 0.2), 3)
            pts = resample_closed(pts, n_points)
            pts = _limit_curvature(pts, r_min)
            pts = resample_closed(pts, n_points)
            wv = np.maximum(1.5, w + 0.2 * _smooth_noise(rng, len(pts)))
            if _self_clearance_ok(pts, wv.max() + 0.35, min_sep_frac=0.05):
                break
        return Track._rasterize_loop(pts, wv, resolution, room_margin, duct_height, rng, n_clutter, clutter_size,
                                     name=f"competition_{seed}", duct_gaps=duct_gaps, banner_fence=banner_fence)

    @staticmethod
    def _gen_serpentine(seed=0, resolution=0.05, n_lanes=None, pitch=None, width=None, lane_len=None,
                        room_margin=2.0, n_points=900, duct_height=0.33, n_clutter=6,
                        clutter_size=(0.3, 1.0), duct_gaps=0.0, banner_fence=0.0):
        """A hall with straight walls cut across it, so the lap folds back on itself.

        Every other generator produces a loop that stays away from itself -- _gen_grid even rejects
        candidates that do not, via _self_clearance_ok -- and measured that way the procedural tracks
        never come within 5.8 m of themselves while the real venues (korea_2025 2.4 m, icra2022 2.5 m,
        the 2026 competition hall 2.4 m) all do. That gap is not about lane width: when two parts of
        the lap twenty metres apart run side by side behind one duct hose, the scan contains open
        space the car may not drive into, sight lines end at a wall rather than at a corner, and two
        distant places on the track look alike to a LiDAR with no localisation to disambiguate them.

        Built as a boustrophedon: `n_lanes` parallel lanes joined by U-turns, closed by a return
        corridor along the top. `pitch` sets how far apart the lanes sit, so `pitch - width` is the
        thickness of the wall between them and controls directly how tightly the track folds.
        """
        rng = np.random.default_rng(seed + 7717)
        n = int(n_lanes or rng.choice([3, 3, 5]))                    # odd: the last lane ends at the top.
                                                                     # Two or three fold-backs, not five:
                                                                     # a comb of hairpins is as far from a
                                                                     # real hall as a plain oval is, just
                                                                     # in the other direction. The 2026
                                                                     # competition map folds twice.
        # Aim at the fold-back the real venues have (2.4-3.3 m), not at the tightest thing that will
        # rasterise. A first pass at 2.1-3.0 m of pitch folded to 1.33 m -- tighter than any real
        # track -- and left a 1.6 m lane with 0.4 m of error budget around a 1.05 m U-turn, which the
        # teacher could not hold: 20 collisions/km and not one completed lap.
        p = float(pitch or rng.uniform(3.0, 4.2))
        w = float(width if width is not None else rng.uniform(1.7, min(2.4, p - 0.9)))
        Ly = float(lane_len or rng.uniform(9.0, 17.0))               # longer lanes: straights between the turns
        ret_y = Ly + p / 2 + rng.uniform(1.6, 3.2)                   # return corridor above the block
        pts = []

        def arc(cx, cy, r, a0, a1, k=26):
            a = np.linspace(a0, a1, k)
            pts.extend(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1).tolist())

        # Each cross wall is its own length and the lanes are not evenly spaced: a comb with identical
        # teeth is one shape with a seed on it, which is the trap _gen_grid already fell into. Some
        # walls are dropped entirely, leaving a wide bay between two folded sections, so a lap mixes
        # tight fold-backs with open ground the way the real venues do.
        depth = Ly * rng.uniform(0.55, 1.0, size=n)                  # how far each wall cuts across
        gap = p * rng.uniform(0.85, 1.25, size=max(1, n - 1))        # spacing between lanes
        keep = rng.uniform(size=max(1, n - 1)) > 0.18                # 18 % of the walls are missing
        x = 0.0
        for i in range(n):
            up = (i % 2 == 0)
            Li = depth[i]
            y0, y1 = (0.0, Li) if up else (Li, 0.0)
            pts.append([x, y0]); pts.append([x, y1])
            if i == n - 1:
                break
            g = float(gap[i]) if keep[i] else float(gap[i]) * 2.1    # no wall: the two lanes join up
            r = g / 2
            if up:                                                   # U-turn over the top of the wall
                arc(x + r, Li, r, math.pi, 0.0)
            else:                                                    # U-turn under the bottom
                arc(x + r, 0.0, r, math.pi, 2 * math.pi)
            x += g
        # Close the loop with rounded corners, not right angles. 96 % of the teacher's collisions on
        # the first version landed within 10 m of s = 0 -- the seam where the last lane turned into
        # the return corridor and the corridor turned back down into the first lane. The lane is as
        # wide there as anywhere, so it was not geometry: two square corners joined by _limit_curvature
        # leave a curvature spike the tracker cannot hold, and every lap ended at the same two places.
        xr = x
        rc = min(p, ret_y - depth[n - 1]) / 2                    # corner radius that fits the corridor
        pts.append([xr, depth[n - 1]])
        pts.append([xr, ret_y - rc]); arc(xr - rc, ret_y - rc, rc, 0.0, math.pi / 2)
        pts.append([rc, ret_y]); arc(rc, ret_y - rc, rc, math.pi / 2, math.pi)
        pts.append([0.0, rc]); arc(rc, rc, rc, math.pi, 1.5 * math.pi)
        pts.append([rc, 0.0]); pts.append([0.0, 0.0])
        pts = np.asarray(pts, float)
        pts = _chaikin(_densify(pts, 0.2), 3)
        pts = resample_closed(pts, n_points)
        pts = _limit_curvature(pts, 0.75)
        pts = resample_closed(pts, n_points)
        wv = np.maximum(1.35, w + 0.18 * _smooth_noise(rng, len(pts)))
        return Track._rasterize_loop(pts, wv, resolution, room_margin, duct_height, rng, n_clutter,
                                     clutter_size, name=f"serpentine_{seed}", duct_gaps=duct_gaps,
                                     banner_fence=banner_fence)

    @staticmethod
    def _gen_hallway(seed=0, resolution=0.05, size=None, corridor=None, n_points=800, n_clutter=6,
                     clutter_size=(0.3, 0.6), room_margin=1.5):
        """Rectangular ring corridor (loop around a block) with optional jogs; walls are tall."""
        rng = np.random.default_rng(seed)
        Wd, Hd = size or (rng.uniform(12, 24), rng.uniform(8, 16))
        cw = corridor or rng.uniform(1.8, 3.0)
        # ring mid-line: rectangle of the outer size minus half corridor, corners rounded by chaikin
        ox, oy = Wd / 2 - cw / 2, Hd / 2 - cw / 2
        corners = np.array([[-ox, -oy], [ox, -oy], [ox, oy], [-ox, oy]])
        if rng.random() < 0.6:                       # a jog on one long side
            j = rng.uniform(0.8, 2.0) * rng.choice([-1, 1]); x1, x2 = sorted(rng.uniform(-ox * 0.6, ox * 0.6, 2))
            corners = np.array([[-ox, -oy], [x1, -oy], [x1, -oy + j], [x2, -oy + j], [x2, -oy], [ox, -oy], [ox, oy], [-ox, oy]])
        pts = _densify(corners, 0.25)
        pts = _chaikin(pts, 2)
        pts = resample_closed(pts, n_points)
        w = np.full(len(pts), cw) + rng.uniform(-0.15, 0.15)
        return Track._rasterize_loop(pts, w, resolution, room_margin, 0.2, rng, n_clutter, clutter_size,
                                     boundary="wall", name=f"hallway_{seed}")

    # ---- geometry -----------------------------------------------------------------------
    @property
    def shape(self):
        return self.occupancy.shape

    def world_to_grid(self, xy: np.ndarray):
        col = (xy[..., 0] - self.origin[0]) / self.resolution
        row = (xy[..., 1] - self.origin[1]) / self.resolution
        return row, col

    def centerline_from_free_space(self, n_points: int = 800, min_clearance: float = 0.5, smooth_iters: int = 3,
                                   direction: str = "ccw", seed_xy=None, min_hole_m2: float = 2.0) -> np.ndarray:
        """Extract the lane centerline of a loop track that has no centerline (SLAM / drawn maps).
        The drivable region of a loop is an annulus: one outer boundary and one infield hole.
        The centerline is the curve equidistant from both, i.e. the zero level set of
        d(outer boundary) - d(infield), extracted with marching squares. Small holes (pillars,
        noise) are filled first; seed_xy picks the free region when the map has several."""
        from skimage.morphology import remove_small_holes
        free = self.edt >= min_clearance
        lab, n = ndimage.label(free)
        if n == 0:
            raise ValueError("no free space with the requested clearance")
        if seed_xy is not None:
            c = int(round((seed_xy[0] - self.origin[0]) / self.resolution)); r = int(round((seed_xy[1] - self.origin[1]) / self.resolution))
            if not (0 <= r < lab.shape[0] and 0 <= c < lab.shape[1]) or lab[r, c] == 0:
                pts = np.argwhere(free); j = np.argmin(np.hypot(pts[:, 0] - r, pts[:, 1] - c)); r, c = pts[j]
            free = lab == lab[r, c]
        else:
            free = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        free = remove_small_holes(free, area_threshold=int(min_hole_m2 / self.resolution ** 2))
        filled = ndimage.binary_fill_holes(free)
        holes = filled & ~free
        hl, hn = ndimage.label(holes)
        if hn == 0:
            raise ValueError("free region has no infield hole: not a closed loop track (give seed_xy or a centerline csv)")
        infield = hl == (np.bincount(hl.ravel())[1:].argmax() + 1)
        free = free | (holes & ~infield)                          # other holes are pillars: part of the lane
        d_out = ndimage.distance_transform_edt(filled)             # distance to the outer boundary
        d_in = ndimage.distance_transform_edt(~infield)            # distance to the infield
        # outside the lane: negative on the outer side, positive in the infield, so the only zero
        # crossing is the mid-lane curve (not the lane boundary)
        phi = np.where(free, d_out - d_in, np.where(infield, 1e3, -1e3))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(); ax = fig.add_subplot(111)
        cs = ax.contour(phi, levels=[0.0])
        segs = list(cs.allsegs[0]) if getattr(cs, "allsegs", None) else [np.asarray(pp) for p_ in cs.get_paths() for pp in p_.to_polygons(closed_only=False)]
        plt.close(fig)
        segs = [sg for sg in segs if len(sg) > 10]
        if not segs:
            raise ValueError("equidistant contour not found")
        best = max(segs, key=lambda sg: abs(0.5 * np.sum(sg[:, 0] * np.roll(sg[:, 1], -1) - np.roll(sg[:, 0], -1) * sg[:, 1])))
        xy = np.stack([self.origin[0] + best[:, 0] * self.resolution, self.origin[1] + best[:, 1] * self.resolution], 1)
        xy = resample_closed(xy, n_points)
        for _ in range(smooth_iters):
            xy = 0.5 * xy + 0.25 * (np.roll(xy, 1, 0) + np.roll(xy, -1, 0))
        # the equidistant curve ignores pillars inside the lane (they were merged into it);
        # push points that ended up too close to any obstacle outwards along the distance gradient
        sdf = self.edt - ndimage.distance_transform_edt(self.occupancy) * self.resolution   # signed: <0 inside obstacles
        gy, gx = np.gradient(sdf, self.resolution)
        for it in range(200):
            rc = np.stack([(xy[:, 1] - self.origin[1]) / self.resolution, (xy[:, 0] - self.origin[0]) / self.resolution])
            d = ndimage.map_coordinates(sdf, rc, order=1, mode="nearest")
            bad = d < min_clearance
            if not bad.any():
                break
            g = np.stack([ndimage.map_coordinates(gx, rc, order=1, mode="nearest"),
                          ndimage.map_coordinates(gy, rc, order=1, mode="nearest")], 1)
            g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
            xy[bad] += g[bad] * np.minimum(min_clearance - d[bad], 2 * self.resolution)[:, None]
            if it % 5 == 4:
                xy = 0.5 * xy + 0.25 * (np.roll(xy, 1, 0) + np.roll(xy, -1, 0))
        area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
        if (direction == "ccw") != (area > 0):
            xy = xy[::-1]
        self.centerline = resample_closed(xy, n_points)
        return self.centerline

    def set_centerline(self, pts: np.ndarray, n_points: int = 800):
        self.centerline = resample_closed(np.asarray(pts, dtype=np.float64), n_points)

    def save_ros_map(self, out_dir: str, name: Optional[str] = None):
        """Write yaml + pgm (+ centerline csv) so RViz / map_server / the ROS bridge can use it."""
        import yaml
        from PIL import Image
        name = name or self.name
        os.makedirs(out_dir, exist_ok=True)
        img = np.where(self.occupancy, 0, 254).astype(np.uint8)
        Image.fromarray(np.flipud(img)).save(os.path.join(out_dir, name + ".pgm"))
        meta = {"image": name + ".pgm", "resolution": self.resolution,
                "origin": [float(self.origin[0]), float(self.origin[1]), 0.0],
                "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196}
        with open(os.path.join(out_dir, name + ".yaml"), "w") as f:
            yaml.safe_dump(meta, f)
        if self.centerline is not None:
            np.savetxt(os.path.join(out_dir, name + "_centerline.csv"), self.centerline,
                       delimiter=",", header="x_m,y_m", comments="# ")
        return os.path.join(out_dir, name + ".yaml")


def _merge_bands(bands, tol: float = 0.01):
    """Collapse consecutive section bands into their union hull while it costs less than `tol`.

    `props.sections` slices every prop into the same number of bands, but most of these props are
    extrusions with a small bevel: a cardboard box's four bands differ by the few millimetres of
    chamfer at its top and bottom, and each one costs the tracer a whole slot to answer the same
    question. Merging is an over-approximation, never an under-approximation -- the union hull
    contains both bands -- so the cost is bounded and measurable, and `tol` is exactly that bound:
    no merged band stands more than `tol` proud of any band it replaced.

    This matters for speed rather than tidiness. `ray_prisms_hits` loops over slots in Python, so
    four props at four bands each is sixteen passes over a (B, N, K) tensor; merging took a measured
    +95.7 % step-rate overhead down to a fraction of it."""
    out, sources = [], []
    for b in bands:
        p = _clean_polygon(b["polygon"])
        cur = out[-1] if out else None
        if cur is not None and abs(cur["z1"] - b["z0"]) <= 1e-9:
            src = sources[-1] + [p]
            u = _relax(_hull(np.vstack(src)), tol)
            # against every band this run has swallowed, not just the running union: checking only
            # the union lets a chain of merges each within tol drift the result past tol from the
            # band it started with.
            if all(_max_outside(u, s) <= tol for s in src):
                out[-1] = {"z0": cur["z0"], "z1": float(b["z1"]), "polygon": u}
                sources[-1] = src
                continue
        out.append({"z0": float(b["z0"]), "z1": float(b["z1"]), "polygon": p})
        sources.append([p])
    return out


def _hull(pts, eps: float = 1e-7):
    """Convex hull, CCW, with duplicate and collinear vertices removed.

    The cleanup is not tidiness. `ConvexHull` keeps points that lie on a hull edge, so hulling two
    copies of the same hexagon returns ten vertices describing six edges -- and four consecutive
    pairs of half-planes are then exactly parallel. `prop_math._polygon_vertices` recovers each
    vertex by intersecting adjacent planes, so a parallel pair divides by a zero determinant; the
    epsilon it substitutes makes the vertex finite but enormous, and it grows with the plane
    offsets, which are world coordinates.

    That is what a degenerate polygon costs: a marker post at the origin reported no contact with a
    car 3 cm off its flank, and the same post at (6, 3) reported 0.119 m of penetration. The bug is
    translation-dependent, so it never shows up in a fixture built at the origin."""
    from scipy.spatial import ConvexHull
    return _clean_polygon(pts[ConvexHull(pts).vertices], eps)


def _clean_polygon(poly, eps: float = 1e-7):
    """CCW, without duplicate or collinear vertices. Applied to every section, merged or not.

    `props.sections` hulls its own sampled cross-sections and keeps points that lie along an edge,
    so a plain hexagonal post arrives as a ten-vertex polygon describing six edges -- the extra four
    are vertices in the middle of edges, and the half-planes either side of each are exactly
    parallel. See `_hull` for what that costs downstream."""
    h = np.asarray(poly, np.float64)
    if _signed_area(h) < 0:
        h = h[::-1].copy()
    h = h[np.linalg.norm(h - np.roll(h, 1, 0), axis=1) > eps]
    while len(h) > 3:
        a, b, c = np.roll(h, 1, 0), h, np.roll(h, -1, 0)
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        flat = np.abs(cross) <= eps * max(1.0, float(np.abs(h).max()))
        if not flat.any() or int((~flat).sum()) < 3:
            break
        h = h[~flat]
    return h


def _relax(poly, tol):
    """Drop edges from a convex polygon while the result still contains it, within `tol`.

    Merging two bands hulls their vertices together, and the hull of two nearly-equal bevelled
    outlines keeps every chamfer corner from both -- a 36 cm box came out with 52 edges. Every edge
    is a half-plane the ray test evaluates for every beam, so those near-duplicates are paid for on
    each of them.

    Removing a vertex here means deleting its edge and letting the two neighbouring edges run on to
    meet, which can only grow the polygon; the new corner is accepted only while it stays within
    `tol` of the original. So this over-approximates, in the one direction that is safe, by a bounded
    and measured amount -- never the other way, which would let a beam pass through a drawn surface.
    """
    P = np.asarray(poly, np.float64)
    changed = True
    while changed and len(P) > 3:
        changed = False
        for i in range(len(P)):
            n = len(P)
            a0, a1 = P[(i - 1) % n], P[i]
            b0, b1 = P[(i + 1) % n], P[(i + 2) % n]
            u, v = a1 - a0, b1 - b0
            den = u[0] * v[1] - u[1] * v[0]
            if abs(den) < 1e-12:                                  # parallel: no corner to make
                continue
            t = ((b0[0] - a0[0]) * v[1] - (b0[1] - a0[1]) * v[0]) / den
            x = a0 + t * u
            cand = np.vstack([P[:i], x[None], P[i + 2:]]) if i + 2 <= n else np.vstack([x[None], P[1:i]])
            if len(cand) < 3 or _signed_area(cand) <= 0:
                continue
            if _max_outside(cand, P) <= tol:
                P = cand
                changed = True
                break
    return P


def _signed_area(p):
    return float(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _max_outside(hull, poly):
    """How far `hull` reaches outside `poly`: true Euclidean distance in metres, 0 where inside.

    Not the largest half-plane violation, which is what this used to compute. That quantity is a
    *lower* bound on the distance, not an upper one: for the unit square at [-0.5, 0.5] the point
    (0.51, 0.51) violates each of two planes by 0.01 while standing sqrt(2) * 0.01 away. A merge
    tolerance built on it would quietly admit merges up to sqrt(K) times looser than it claimed.

    K is a handful of vertices here, so the honest distance is affordable: point to each edge
    segment, taking the minimum, for the points that are outside at all."""
    from .prop_math import section_halfplanes
    H = np.asarray(hull, np.float64)
    P = np.asarray(poly, np.float64)
    n, d = section_halfplanes(P, len(P))
    outside = (H @ n.T - d[None]).max(1) > 0
    if not outside.any():
        return 0.0
    Q = H[outside]
    A, B = P, np.roll(P, -1, 0)
    seg = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-18)
    t = np.clip(((Q[:, None, :] - A[None]) * seg[None]).sum(-1) / L2[None], 0.0, 1.0)
    return float(np.linalg.norm(Q[:, None, :] - (A[None] + t[..., None] * seg[None]), axis=-1).min(1).max())


class TrackTensors:
    """GPU-resident copies of one or more tracks, batched: every env carries a track id (tid).
    Distance fields are concatenated into flat buffers with per-track offsets; centerlines are
    resampled to a common number of points so per-env gathers stay dense."""

    def __init__(self, tracks, device, n_cl: int = 1000, dtype=None):
        if isinstance(tracks, Track):
            tracks = [tracks]
        self.tracks = list(tracks)
        self.T = len(self.tracks)
        self.device = torch.device(device)
        offs, e, ed, et = [], [], [], []
        o = 0; seen = {}                                   # identical grids (reversed direction) share one copy
        for t in self.tracks:
            key = t.grid_key()
            if key in seen:
                offs.append(seen[key]); continue
            seen[key] = o; offs.append(o); o += t.occupancy.size
            e.append(torch.from_numpy(t.edt).reshape(-1)); ed.append(torch.from_numpy(t.edt_duct).reshape(-1)); et.append(torch.from_numpy(t.edt_tall).reshape(-1))
        self.dtype = dtype or (torch.float16 if o > 6e6 else torch.float32)
        self.edt = torch.cat(e).to(self.device, self.dtype)
        self.edt_duct = torch.cat(ed).to(self.device, self.dtype)
        self.edt_tall = torch.cat(et).to(self.device, self.dtype)
        dev = self.device
        self.t_off = torch.tensor(offs, dtype=torch.int64, device=dev)
        self.t_H = torch.tensor([t.occupancy.shape[0] for t in self.tracks], dtype=torch.int32, device=dev)
        self.t_W = torch.tensor([t.occupancy.shape[1] for t in self.tracks], dtype=torch.int32, device=dev)
        self.t_res = torch.tensor([t.resolution for t in self.tracks], dtype=torch.float32, device=dev)
        self.t_origin = torch.tensor([list(t.origin) for t in self.tracks], dtype=torch.float32, device=dev)
        self.t_duct_h = torch.tensor([t.duct_height for t in self.tracks], dtype=torch.float32, device=dev)
        # single-track conveniences (track 0): viewer, tests, scan_meta
        self.track = self.tracks[0]
        self.res = self.track.resolution
        self.origin = self.t_origin[0]
        self.H, self.W = self.track.occupancy.shape
        self.duct_height = float(self.track.duct_height)
        # centerlines: tracks without one get a dummy (progress/laps undefined there: cl_ok False)
        self.cl_ok = torch.tensor([t.centerline is not None for t in self.tracks], device=dev)
        if bool(self.cl_ok.any()):
            cls = [resample_closed(t.centerline, n_cl) if t.centerline is not None
                   else np.stack([np.cos(np.linspace(0, 2 * np.pi, n_cl, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, n_cl, endpoint=False))], 1) * 1e3
                   for t in self.tracks]
            cl = torch.tensor(np.stack(cls), dtype=torch.float32, device=dev)          # (T, N, 2)
            seg = torch.roll(cl, -1, 1) - cl
            seglen = seg.norm(dim=2)
            self.cl = cl
            self.cl_tangent = seg / seglen[..., None]
            self.cl_s = torch.cat([torch.zeros(self.T, 1, device=dev), torch.cumsum(seglen, 1)[:, :-1]], 1)
            self.length = torch.where(self.cl_ok, seglen.sum(1), torch.zeros(self.T, device=dev))   # (T,)
            self.n_cl = n_cl
        else:
            self.cl = None
            self.length = torch.zeros(self.T, device=dev)
        self._build_props()

    # ------------------------------------------------------------------ static props
    def _build_props(self, k_pad: Optional[int] = None, n_bands: int = 4):
        """Per-track prop section tensors, padded to a common slot count.

        Every prop contributes one slot per convex section band, so a tapered post is bounded far
        more tightly than one prism over its whole height would bound it. Bands whose polygon does
        not change are merged first, which costs a box nothing: its four identical bands collapse
        back to the single slot it deserves.

        `has_props` is False for every track that has none -- which is every existing track -- and
        the tracers check it to keep their old path exactly as it was."""
        self.has_props = any(len(t.props) for t in self.tracks)
        dev = self.device
        if not self.has_props:
            self.k_pad = k_pad = int(k_pad or 8)
            self.p_poses = torch.zeros(self.T, 0, 3, device=dev, dtype=torch.float32)
            self.p_n = torch.zeros(self.T, 0, k_pad, 2, device=dev, dtype=torch.float32)
            self.p_d = torch.zeros(self.T, 0, k_pad, device=dev, dtype=torch.float32)
            self.p_zlo = torch.zeros(self.T, 0, device=dev, dtype=torch.float32)
            self.p_zhi = torch.zeros(self.T, 0, device=dev, dtype=torch.float32)
            return
        from . import props as _props
        from .prop_math import section_halfplanes
        # Bands first, k_pad from what they turned out to need. A band polygon is the convex hull of
        # the cross-sections across that band, not the prop's base footprint, so it can carry more
        # edges than the footprint does -- a 16-sided drum with rolling ribs hulls to 32. Fixing the
        # pad at the footprint's budget made that a hard failure at map load.
        per_track = []
        for t in self.tracks:
            bands = []
            for sp in t.props:
                for band in _merge_bands(_props.sections(sp.build(), n_bands=n_bands)):
                    bands.append((sp, band))
            per_track.append(bands)
        need = max((len(b["polygon"]) for bs in per_track for _, b in bs), default=8)
        self.k_pad = k_pad = int(k_pad or max(8, need))
        per_track = [[((sp.x, sp.y, sp.yaw),
                       *section_halfplanes(np.asarray(b["polygon"], np.float64), k_pad),
                       b["z0"], b["z1"]) for sp, b in bs] for bs in per_track]
        C = max(len(s) for s in per_track)
        poses = np.zeros((self.T, C, 3), np.float32)
        pn = np.zeros((self.T, C, k_pad, 2), np.float32)
        pd = np.full((self.T, C, k_pad), np.inf, np.float32)
        zlo = np.zeros((self.T, C), np.float32)
        zhi = np.zeros((self.T, C), np.float32)          # z_hi <= z_lo marks a dead padding slot
        for ti, slots in enumerate(per_track):
            for ci, (pose, n_, d_, z0, z1) in enumerate(slots):
                poses[ti, ci] = pose
                pn[ti, ci] = n_
                pd[ti, ci] = d_
                zlo[ti, ci], zhi[ti, ci] = z0, z1
        self.p_poses = torch.from_numpy(poses).to(dev)
        self.p_n = torch.from_numpy(pn).to(dev)
        self.p_d = torch.from_numpy(pd).to(dev)
        self.p_zlo = torch.from_numpy(zlo).to(dev)
        self.p_zhi = torch.from_numpy(zhi).to(dev)

    def props_for(self, tid: torch.Tensor):
        """Gather each env's own track's prop slots: (poses, pn, pd, z_lo, z_hi), all leading (B,)."""
        return (self.p_poses[tid], self.p_n[tid], self.p_d[tid], self.p_zlo[tid], self.p_zhi[tid])

    # ------------------------------------------------------------------ grids
    def sample_edt(self, xy: torch.Tensor, tid: torch.Tensor, field: Optional[torch.Tensor] = None,
                   with_inside: bool = False):
        """Nearest-cell EDT lookup on each point's track; outside the grid counts as a wall (0).
        xy (..., 2), tid (...,) long (broadcastable to xy.shape[:-1]) -> (...)

        with_inside: also return the in-bounds mask. Callers that want both used to ask twice, the
        second time against a field of ones -- which allocates a tensor the size of every track's
        grid put together, on every call, to learn something four comparisons already know.
        """
        field = self.edt if field is None else field
        tid = torch.broadcast_to(tid, xy.shape[:-1])
        res = self.t_res[tid]; ox = self.t_origin[tid, 0]; oy = self.t_origin[tid, 1]
        H = self.t_H[tid].long(); W = self.t_W[tid].long()
        col = ((xy[..., 0] - ox) / res).round().long()
        row = ((xy[..., 1] - oy) / res).round().long()
        inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        idx = self.t_off[tid] + row.clamp(min=0).minimum(H - 1) * W + col.clamp(min=0).minimum(W - 1)
        d = field[idx].float()
        d = torch.where(inside, d, torch.zeros_like(d))
        return (d, inside) if with_inside else d

    def edt_gradient(self, xy: torch.Tensor, tid: torch.Tensor, field: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Unit vector pointing away from the nearest wall (central differences). xy (..., 2), tid (...)."""
        eps = self.t_res[tid][..., None]
        ex = torch.tensor([1.0, 0.0], device=self.device) * eps
        ey = torch.tensor([0.0, 1.0], device=self.device) * eps
        gx = self.sample_edt(xy + ex, tid, field) - self.sample_edt(xy - ex, tid, field)
        gy = self.sample_edt(xy + ey, tid, field) - self.sample_edt(xy - ey, tid, field)
        g = torch.stack([gx, gy], -1)
        return g / (g.norm(dim=-1, keepdim=True) + 1e-9)

    # ------------------------------------------------------------------ centerline
    def project(self, xy: torch.Tensor, tid: torch.Tensor, prev_idx: Optional[torch.Tensor] = None,
                window: float = 3.0):
        """Nearest centerline point on each env's track. xy (B,2), tid (B,) -> (s, lateral offset (left +), index)

        prev_idx: last step's index. Where the lap folds back on itself -- the layout every real
        venue has, and the one the serpentine generator is built for -- two parts of the lap run
        within a couple of metres of each other, and a plain global argmin will jump between them.
        Progress, lap counting and the wrong-way check all read `s`, so a jump of twenty metres is
        scored as a lap: the teacher measured 34 collisions/km and a 2 s "lap" on a 106 m track
        purely from this. Given the previous index, the search is restricted to a window of
        +-`window` metres of arc around it, which cannot straddle a fold. 3 m: a U-turn is about 4 m
        of arc, so a wider window reaches around it onto the next lane, while at 10 m/s the car
        covers only 0.25 m per control step.
        """
        cl = self.cl[tid]                                            # (B, N, 2)
        d2 = ((xy[:, None, :] - cl) ** 2).sum(-1)
        if prev_idx is not None:
            L = self.length[tid].clamp_min(1e-6)
            ds = (self.cl_s[tid] - self.cl_s[tid].gather(1, prev_idx[:, None]))
            ds = (ds + L[:, None] / 2) % L[:, None] - L[:, None] / 2      # signed arc distance, wrapped
            d2 = torch.where(ds.abs() <= window, d2, torch.full_like(d2, float("inf")))
        idx = d2.argmin(1)
        ar = torch.arange(xy.shape[0], device=xy.device)
        p, t = cl[ar, idx], self.cl_tangent[tid, idx]
        rel = xy - p
        along = (rel * t).sum(1)
        lateral = rel[:, 0] * -t[:, 1] + rel[:, 1] * t[:, 0]
        L = self.length[tid].clamp_min(1e-6)
        s = (self.cl_s[tid, idx] + along) % L
        s = torch.where(s >= L, s - L, s)
        ok = self.cl_ok[tid]
        return torch.where(ok, s, torch.zeros_like(s)), torch.where(ok, lateral, torch.zeros_like(lateral)), idx

    def pose_at_s(self, s: torch.Tensor, tid: torch.Tensor):
        """(x, y, yaw) on the centerline at arclength s (B,) of each env's track (B,)"""
        L = self.length[tid].clamp_min(1e-6)
        s = s % L
        cl_s = self.cl_s[tid]                                        # (B, N)
        idx = (torch.searchsorted(cl_s, s[:, None], right=True) - 1).clamp(0, self.n_cl - 1)[:, 0]
        ar = torch.arange(s.shape[0], device=s.device)
        p, t = self.cl[tid, idx], self.cl_tangent[tid, idx]
        xy = p + t * (s - cl_s[ar, idx])[:, None]
        yaw = torch.atan2(t[:, 1], t[:, 0])
        return xy, yaw


def resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    s_new = np.linspace(0, L, n, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    x = np.interp(s_new, s, closed[:, 0])
    y = np.interp(s_new, s, closed[:, 1])
    return np.stack([x, y], 1)


def load_centerline_csv(path: str) -> np.ndarray:
    """Accepts f1tenth_racetracks style csv (x_m, y_m, w_tr_right_m, w_tr_left_m) or plain x,y."""
    data = np.genfromtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data[None]
    return resample_closed(data[:, :2].astype(np.float64), max(400, len(data)))


def _smooth_noise(rng, n, k_max=3):
    v = np.zeros(n); t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    for k in range(1, k_max + 1):
        v += rng.uniform(-1, 1) / k * np.cos(k * t + rng.uniform(0, 2 * math.pi))
    return v / max(1e-6, np.abs(v).max())


def _chaikin(pts, iters):
    for _ in range(iters):
        nxt = np.roll(pts, -1, 0)
        out = np.empty((2 * len(pts), 2)); out[0::2] = 0.75 * pts + 0.25 * nxt; out[1::2] = 0.25 * pts + 0.75 * nxt
        pts = out
    return pts


def _densify(corners, step):
    out = []
    for i in range(len(corners)):
        a, b = corners[i], corners[(i + 1) % len(corners)]
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n, endpoint=False)]
    return np.asarray(out)


def curvature_np(pts):
    d = np.roll(pts, -1, 0) - np.roll(pts, 1, 0); dd = np.roll(pts, -1, 0) - 2 * pts + np.roll(pts, 1, 0)
    ds = np.linalg.norm(d, axis=1) / 2 + 1e-9
    d = d / (2 * ds[:, None]); dd = dd / (ds[:, None] ** 2)
    return d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]


def _limit_curvature(pts, r_min, iters=60):
    """Locally smooth (Laplacian) wherever |curvature| exceeds 1/r_min."""
    kmax = 1.0 / r_min
    for _ in range(iters):
        k = np.abs(curvature_np(pts))
        bad = k > kmax
        if not bad.any():
            break
        m = ndimage.binary_dilation(bad, iterations=3)
        lap = 0.5 * (np.roll(pts, -1, 0) + np.roll(pts, 1, 0)) - pts
        pts = pts + 0.5 * lap * m[:, None]
    return pts


def _self_clearance_ok(pts, min_dist, min_sep_frac=1 / 12):
    """No two points far apart along the loop may be closer than min_dist (lanes must not merge)."""
    n = len(pts)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    idx = np.arange(n)
    sep = np.minimum(np.abs(idx[:, None] - idx[None, :]), n - np.abs(idx[:, None] - idx[None, :]))
    mask = sep > int(n * min_sep_frac)
    return bool((d[mask] >= min_dist).all())


def _cell_union_outline(blob, pitch):
    """Ordered outline polygon (m) of a simply-connected set of grid cells, corners at cell vertices."""
    gh, gw = blob.shape
    # directed boundary edges (counter-clockwise around the blob): for each cell edge with no neighbour
    edges = {}
    for y in range(gh):
        for x in range(gw):
            if not blob[y, x]:
                continue
            c = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]      # CCW corners
            nbrs = [(y - 1, x), (y, x + 1), (y + 1, x), (y, x - 1)]  # bottom, right, top, left
            for i, (ny, nx) in enumerate(nbrs):
                if not (0 <= ny < gh and 0 <= nx < gw and blob[ny, nx]):
                    a, b = c[i], c[(i + 1) % 4]
                    edges[a] = b
    if not edges:
        return None
    start = next(iter(edges)); loop = [start]; cur = edges[start]
    while cur != start and len(loop) <= len(edges):
        loop.append(cur); cur = edges.get(cur)
        if cur is None:
            return None
    if len(loop) != len(edges):                       # more than one boundary loop (hole/disconnected)
        return None
    pts = np.array(loop, dtype=float) * pitch
    return pts - pts.mean(0)
