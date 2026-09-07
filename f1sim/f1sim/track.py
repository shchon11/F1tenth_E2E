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


@dataclass
class Track:
    """Layered obstacle map.
    occupancy : every obstacle the car can hit (duct | tall)            -> collision, spawn checks
    duct      : low boundary objects (flexible duct hose, height duct_height) -> LiDAR beams pass over them when tilted
    tall      : tall objects (room walls, unknown space, clutter)        -> always block beams
    Cells that are neither are floor (drivable or not)."""
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
        return Track.from_occupancy(occ, self.resolution, self.origin, cl, self.name + "m", duct=duct, tall=tall,
                                    duct_height=self.duct_height)

    def grid_key(self):
        """Content hash of the obstacle layers (tracks that differ only in centerline share GPU grids)."""
        import hashlib
        h = hashlib.md5(np.packbits(self.occupancy).tobytes()); h.update(np.packbits(self.duct).tobytes())
        return (self.occupancy.shape, round(self.resolution, 6), tuple(np.round(self.origin, 4)), float(self.duct_height), h.hexdigest())

    def reversed(self) -> "Track":
        """Same map, lap driven the other way round (centerline reversed). Grids are shared, not
        copied, so TrackTensors keeps a single GPU copy for both directions."""
        cl = None if self.centerline is None else np.ascontiguousarray(self.centerline[::-1])
        return Track(self.occupancy, self.resolution, self.origin, self.edt, cl, self.name + "r",
                     duct=self.duct, tall=self.tall, edt_duct=self.edt_duct, edt_tall=self.edt_tall,
                     duct_height=self.duct_height)

    @staticmethod
    def generate_random(seed: int = 0, style: str = "competition", resolution: float = 0.05, mirror="auto",
                        lane_obstacles: bool | int = False, **kw) -> "Track":
        """Procedural tracks. style:
        "competition": control-point loop with hairpins, chicanes and varying width (1.6-2.6 m),
                       duct-hose boundaries in a room with clutter (indoor RoboRacer/F1TENTH events)
        "circuit":     smooth Fourier loop, wide, duct boundaries (the old generator)
        "hallway":     rectangular building corridor loop with 90 deg corners, tall walls, clutter
                       (Levine-style venues)
        mirror: True/False, or "auto" = odd seeds are mirrored (clockwise) so both turn directions
        appear equally often."""
        if style != "hallway":                      # duct-hose venues: 33 cm hoses laid in segments with gaps
            rng = np.random.default_rng(seed + 11)  # (CDC 2025), banner boards around some venues
            kw.setdefault("duct_height", 0.33)
            kw.setdefault("duct_gaps", 0.12)
            kw.setdefault("banner_fence", 0.0 if rng.uniform() < 0.4 else float(rng.uniform(1.0, 3.0)))
        if style == "circuit":
            t = Track._gen_circuit(seed, resolution, **kw)
        elif style == "hallway":
            t = Track._gen_hallway(seed, resolution, **kw)
        elif style == "control":
            t = Track._gen_competition(seed, resolution, **kw)
        else:
            t = Track._gen_grid(seed, resolution, **kw)
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
                                     name=f"competition_{seed}", duct_gaps=duct_gaps, banner_fence=banner_fence)

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

    # ------------------------------------------------------------------ grids
    def sample_edt(self, xy: torch.Tensor, tid: torch.Tensor, field: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Nearest-cell EDT lookup on each point's track; outside the grid counts as a wall (0).
        xy (..., 2), tid (...,) long (broadcastable to xy.shape[:-1]) -> (...)"""
        field = self.edt if field is None else field
        tid = torch.broadcast_to(tid, xy.shape[:-1])
        res = self.t_res[tid]; ox = self.t_origin[tid, 0]; oy = self.t_origin[tid, 1]
        H = self.t_H[tid].long(); W = self.t_W[tid].long()
        col = ((xy[..., 0] - ox) / res).round().long()
        row = ((xy[..., 1] - oy) / res).round().long()
        inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        idx = self.t_off[tid] + row.clamp(min=0).minimum(H - 1) * W + col.clamp(min=0).minimum(W - 1)
        d = field[idx].float()
        return torch.where(inside, d, torch.zeros_like(d))

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
    def project(self, xy: torch.Tensor, tid: torch.Tensor):
        """Nearest centerline point on each env's track. xy (B,2), tid (B,) -> (s, lateral offset (left +), index)"""
        cl = self.cl[tid]                                            # (B, N, 2)
        d2 = ((xy[:, None, :] - cl) ** 2).sum(-1)
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
