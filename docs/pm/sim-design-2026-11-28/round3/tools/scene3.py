#!/usr/bin/env python3
"""Still stand-ins for the 3D scene in the round-3 mockups (CPU only, one process).

Draws the project's own car model and the team's measured maps as duct-hose courses, with
poses, boxes, scans and driven paths taken from recorded sessions (tools/world.py).
The real app draws these with Godot; this is only so the mockups have a believable backdrop.

    python3 -I tools/scene3.py [name ...] [--draft]     # no name = everything
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scene_base as sb          # noqa: E402  (Renderer, Mesh from round 2)
import world                      # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "assets")
SID = "20261129-000708-quali-a-1124"
HOSE = (0.66, 0.67, 0.69)
CARD = (0.62, 0.47, 0.30)
BG = np.array([0.050, 0.053, 0.060])
FLOOR = np.array([0.215, 0.222, 0.238])

CAR, PIV = world.load_car()


def V3(xy, h=0.0):
    """map (x, y) -> scene (x, up, -y)"""
    xy = np.asarray(xy, float)
    return np.stack([xy[..., 0], np.broadcast_to(np.asarray(h, float), xy.shape[:-1]), -xy[..., 1]], -1)


# ----------------------------------------------------------------- geometry
def add_car(m, x, y, yaw, steer=0.0):
    c, s = math.cos(yaw), math.sin(yaw)

    def place(P):
        return np.stack([x + P[..., 0] * c - P[..., 1] * s, P[..., 2], -(y + P[..., 0] * s + P[..., 1] * c)], -1)

    def rot(N):
        return np.stack([N[..., 0] * c - N[..., 1] * s, N[..., 2], -(N[..., 0] * s + N[..., 1] * c)], -1)

    for part, prims in CAR.items():
        for name, P, N, C in prims:
            if steer and part in ("wheel_fl", "wheel_fr"):
                piv, cs, sn = PIV[part], math.cos(steer), math.sin(steer)
                d = P - piv
                P = piv + np.stack([d[..., 0] * cs - d[..., 1] * sn, d[..., 0] * sn + d[..., 1] * cs, d[..., 2]], -1)
                N = np.stack([N[..., 0] * cs - N[..., 1] * sn, N[..., 0] * sn + N[..., 1] * cs, N[..., 2]], -1)
            m.add(place(P), rot(N), C)


def add_hoses(m, course, step=1, sides=10, r=0.15):
    for w in course.walls:
        p = w[::step]
        m.tube(np.stack([p[:, 0], -p[:, 1]], 1), r, HOSE, sides=sides)


def add_boxes(m, boxes):
    for b in boxes:
        m.box([b["x"], b["h"] / 2, -b["y"]], [b["w"], b["h"], b["d"]], yaw=b["yaw"], col=CARD)


def add_hall(m, lo, hi, margin=16.0, height=9.0, lamps=0.55):
    wall = (0.15, 0.155, 0.17)
    x0, x1, y0, y1 = lo[0] - margin, hi[0] + margin, lo[1] - margin, hi[1] + margin
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for c, sz in (([x0, height / 2, -cy], [0.3, height, y1 - y0]), ([x1, height / 2, -cy], [0.3, height, y1 - y0]),
                  ([cx, height / 2, -y0], [x1 - x0, height, 0.3]), ([cx, height / 2, -y1], [x1 - x0, height, 0.3])):
        m.box(c, sz, 0.0, wall)
    if lamps:
        for x in np.arange(x0 + 2.5, x1 - 1, 3.6):
            for y in np.arange(y0 + 3, y1 - 2, 5.0):
                m.box([x, height, -y], [0.28, 0.05, 3.2], 0.0, (-1.0 * lamps, -0.98 * lamps, -0.92 * lamps))


# ----------------------------------------------------------------- render
class Shot:
    """camera + buffers of a finished render: project() for overlays drawn in the page."""

    def __init__(self, r, W, H):
        self.r, self.W, self.H = r, W, H

    def project(self, P):
        sx, sy, z = self.r._project(np.asarray(P, float))
        return np.stack([sx / self.r.ss, sy / self.r.ss, z], -1)

    def visible(self, P, eps=0.03):
        """True where a point is not behind drawn geometry."""
        sx, sy, z = self.r._project(np.asarray(P, float))
        ix, iy = np.clip(sx.astype(int), 0, self.r.W - 1), np.clip(sy.astype(int), 0, self.r.H - 1)
        zb = self.r.z[iy, ix]
        ok = (z > 0.05) & (sx >= 0) & (sx < self.r.W) & (sy >= 0) & (sy < self.r.H)
        with np.errstate(divide="ignore", invalid="ignore"):
            return ok & ((zb <= 0) | (1.0 / np.maximum(z, 1e-6) >= zb * (1 - eps)))


def render(name, eye, tgt, fov, build, W=1920, H=1200, ss=2, fog=0.03, spot=None, spot_r=3.0, floor_fx=None,
           post=None, key=(0.55, 0.45), floor=FLOOR, bg=BG, joints=0.03, spot_gain=(0.62, 0.75)):
    eye, tgt = np.asarray(eye, float), np.asarray(tgt, float)
    r = sb.Renderer(W, H, ss)
    r.camera(eye, tgt, fov)
    flat = np.array([r.Rv[2][0], 0.0, r.Rv[2][2]])
    flat /= (np.linalg.norm(flat) + 1e-9)
    rgt = np.array([r.Rv[0][0], 0.0, r.Rv[0][2]])
    rgt /= np.linalg.norm(rgt)
    L = flat * key[0] + rgt * key[1] + np.array([0.0, -1.0, 0.0])
    L /= np.linalg.norm(L)
    mesh = sb.Mesh()
    build(mesh)
    P, N, A = mesh.arrays()
    N = N / (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12)
    Vv = eye - P
    dist = np.linalg.norm(Vv, axis=-1)
    Vv = Vv / dist[..., None]
    ndl = np.clip(-(N @ L), 0, 1)
    hemi = 0.30 + 0.25 * (N[..., 1] * 0.5 + 0.5)
    Hh = Vv - L
    Hh /= (np.linalg.norm(Hh, axis=-1, keepdims=True) + 1e-9)
    lum = A.mean(-1)
    spec = np.clip((N * Hh).sum(-1), 0, 1) ** 18 * (0.20 + 0.5 * np.clip(lum, 0, 1))
    rim = (1 - np.abs((N * Vv).sum(-1))) ** 3 * 0.16 * (N[..., 1] * 0.5 + 0.6)
    C = (A + 0.012) * (hemi + 1.05 * ndl)[..., None] + (spec + rim)[..., None]
    unlit = (A < 0).any(-1)
    C[unlit] = -A[unlit]
    fg = (1 - np.exp(-dist * fog))[..., None]
    C = C * (1 - fg) + bg * fg
    r.tris(P, C.astype(np.float32))
    lit = ~unlit.any(-1)
    shadow_total = None
    for Ld, blur, k in ((L, 5 * ss, 0.5), (np.array([0.0, -1.0, 0.0]), 9 * ss, 0.38)):
        r.shadow[:] = 0
        Pl = P[lit & (P[..., 1].max(-1) < 2.5)]
        t = Pl[..., 1] / -Ld[1]
        S = Pl + Ld * t[..., None]
        S[..., 1] = 0.0
        r.tris(S, shadow=True)
        sh = np.asarray(Image.fromarray((r.shadow * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur)),
                        np.float32) / 255
        shadow_total = sh * k if shadow_total is None else 1 - (1 - shadow_total) * (1 - sh * k)
    ys, xs = np.mgrid[0:r.H, 0:r.W].astype(np.float32)
    dx = (xs + 0.5 - r.cx) / r.f
    dy = -(ys + 0.5 - r.cy) / r.f
    D = dx[..., None] * r.Rv[0] + dy[..., None] * r.Rv[1] + r.Rv[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -eye[1] / D[..., 1]
    hit = (D[..., 1] < -1e-5) & (t > 0)
    fx = eye[0] + D[..., 0] * t
    fz = eye[2] + D[..., 2] * t
    fd = np.where(hit, t * np.linalg.norm(D, axis=-1), 1e6)
    fcol = np.broadcast_to(floor, (r.H, r.W, 3)).copy()
    if joints:
        px = fd / r.f * 1.6
        for g in (fx, fz):
            dgrid = np.abs(g - np.round(g))
            fcol += (np.clip(1 - dgrid / np.maximum(px, 0.004), 0, 1) * joints * np.clip(6 / np.maximum(fd, 1), 0, 1))[..., None]
    if spot is not None:
        sd = np.hypot(fx - spot[0], fz + spot[1])
        fcol *= (spot_gain[0] + spot_gain[1] * np.exp(-(sd / spot_r) ** 2))[..., None]
    if floor_fx is not None:
        fcol = floor_fx(fx, -fz, fcol, fd / r.f)          # map coordinates (x, y), metres per pixel
    fcol *= (1 - shadow_total)[..., None]
    ff = (1 - np.exp(-fd * fog))[..., None]
    fcol = fcol * (1 - ff) + bg * ff
    back = np.where(hit[..., None], fcol, bg)
    obj = r.z > 0
    img = np.clip(np.where(obj[..., None], r.col, back), 0, 1)
    out = Image.fromarray((img * 255).astype(np.uint8))
    shot = Shot(r, W, H)
    if post is not None:
        out = post(out, shot) or out
    out = out.resize((W, H), Image.LANCZOS)
    out.save(os.path.join(OUT, name + ".png"))
    print(name, out.size, len(P), "tris", flush=True)
    return shot


# ----------------------------------------------------------------- floor decals
def seg_dist(fx, fy, pts, closed=False):
    """distance from every floor pixel to a polyline (map coordinates), computed inside its bounding box."""
    pts = np.asarray(pts, float)
    d = np.full(fx.shape, 1e3, np.float32)
    lo, hi = pts.min(0) - 0.6, pts.max(0) + 0.6
    m = (fx > lo[0]) & (fx < hi[0]) & (fy > lo[1]) & (fy < hi[1])
    if not m.any():
        return d
    X, Y = fx[m], fy[m]
    best = np.full(X.shape, 1e3, np.float32)
    seq = list(zip(pts[:-1], pts[1:])) + ([(pts[-1], pts[0])] if closed else [])
    for a, b in seq:
        ab = b - a
        L2 = float(ab @ ab) + 1e-12
        tt = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0, 1)
        best = np.minimum(best, np.hypot(X - (a[0] + tt * ab[0]), Y - (a[1] + tt * ab[1])))
    d[m] = best
    return d


def line_decal(pts, width, gain=1.0, closed=False, col=(0.96, 0.96, 0.95), dots=None, dot_r=0.05):
    def fx_(fx, fy, fcol, mpp):
        d = seg_dist(fx, fy, pts, closed)
        a = np.clip((width / 2 - d) / np.maximum(mpp, 1e-4) + 0.5, 0, 1) * gain
        if dots is not None:
            for q in np.asarray(dots, float):
                dd = np.hypot(fx - q[0], fy - q[1])
                a = np.maximum(a, np.clip((dot_r - dd) / np.maximum(mpp, 1e-4) + 0.5, 0, 1))
        return fcol * (1 - a[..., None]) + np.asarray(col) * a[..., None]
    return fx_


def chain(*fns):
    def fx_(fx, fy, fcol, mpp):
        for f in fns:
            if f is not None:
                fcol = f(fx, fy, fcol, mpp)
        return fcol
    return fx_


def fan_decal(x, y, yaw, rmax=10.0, fov=math.radians(270), ranges=None, gain=0.16):
    """the LiDAR's field of view on the floor: a soft sector, its two edges, a ring every metre."""
    def fx_(fx, fy, fcol, mpp):
        dx, dy = fx - x, fy - y
        rr = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx) - yaw
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        inside = (np.abs(ang) < fov / 2) & (rr < rmax) & (rr > 0.03)
        if ranges is not None:
            k = np.clip(((ang + fov / 2) / fov * (len(ranges) - 1)).astype(int), 0, len(ranges) - 1)
            inside &= rr < np.nan_to_num(ranges, nan=rmax)[k]
        a = inside * gain * np.clip(1.15 - rr / rmax, 0.15, 1)
        ring = np.abs(rr - np.round(rr))
        a = a + inside * np.clip(1 - ring / np.maximum(mpp * 1.4, 1e-4), 0, 1) * 0.20 * (np.round(rr) >= 1)
        edge = np.abs(np.abs(ang) - fov / 2) * rr
        a = a + ((rr < rmax) & (rr > 0.05)) * np.clip(1 - edge / np.maximum(mpp * 1.5, 1e-4), 0, 1) * 0.55
        a = np.clip(a, 0, 1)
        return fcol * (1 - a[..., None]) + np.array([0.96, 0.96, 0.95]) * a[..., None]
    return fx_


def grid_decal(course, free=0.10, occ=0.95):
    """a map lying on the floor: occupied cells white, measured free space a little lighter."""
    def fx_(fx, fy, fcol, mpp):
        ij = course.to_px(np.stack([fx, fy], -1))
        ix, iy = ij[..., 0].astype(int), ij[..., 1].astype(int)
        ok = (ix >= 0) & (ix < course.W) & (iy >= 0) & (iy < course.H)
        g = np.full(fx.shape, 205, np.uint8)
        g[ok] = course.gray[iy[ok], ix[ok]]
        a = np.where(g < 90, occ, np.where(g > 250, free, 0.0)).astype(np.float32)
        return fcol * (1 - a[..., None]) + np.array([0.96, 0.96, 0.95]) * a[..., None]
    return fx_


def scan_points(course, x, y, yaw, boxes, every=3, h=0.16):
    ang, rng = course.scan(x + 0.27 * math.cos(yaw), y + 0.27 * math.sin(yaw), yaw, boxes)
    ok = ~np.isnan(rng)
    px = x + 0.27 * math.cos(yaw) + np.cos(ang) * rng
    py = y + 0.27 * math.sin(yaw) + np.sin(ang) * rng
    return V3(np.stack([px, py], 1)[ok][::every], h), rng


def draw_points(pts3, radius_m=0.016, min_px=1.6, col=(244, 244, 241)):
    def post(img, shot):
        r = shot.r
        sx, sy, z = r._project(pts3)
        vis = shot.visible(pts3, eps=0.06)
        d = ImageDraw.Draw(img)
        for X, Y, Z, ok in zip(sx, sy, z, vis):
            if not ok:
                continue
            rad = max(min_px * r.ss, radius_m * r.f / Z)
            d.ellipse([X - rad, Y - rad, X + rad, Y + rad], fill=col)
        return img
    return post


def write_overlay(name, **kw):
    def conv(v):
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (str, float, int)):
            return v
        return np.round(np.asarray(v, float), 1).tolist()
    with open(os.path.join(OUT, "ov_" + name + ".json"), "w") as f:
        json.dump(conv(kw), f)


def fit_camera(pts2d, az, elev, fov, rect, W=1920, H=1200, h=0.15):
    """eye, target so that map points (x, y) fill the screen rectangle (x0, y0, x1, y1), seen from
    azimuth az (the way the camera looks, map frame) and elev radians above the floor."""
    P = V3(pts2d, h)
    fwd = np.array([math.cos(elev) * math.cos(az), -math.sin(elev), -math.cos(elev) * math.sin(az)])
    gf = np.array([math.cos(az), 0.0, -math.sin(az)])
    gr = np.array([math.sin(az), 0.0, math.cos(az)])
    tgt = P.mean(0) * np.array([1, 0, 1])
    d = 20.0
    r = sb.Renderer(W, H, 1)
    for _ in range(80):
        r.camera(tgt - fwd * d, tgt, fov)
        sx, sy, z = r._project(P)
        bw, bh = sx.max() - sx.min(), sy.max() - sy.min()
        k = min((rect[2] - rect[0]) / bw, (rect[3] - rect[1]) / bh)
        d /= k ** 0.5
        ex = (rect[0] + rect[2]) / 2 - (sx.min() + sx.max()) / 2
        ey = (rect[1] + rect[3]) / 2 - (sy.min() + sy.max()) / 2
        tgt = tgt - gr * ex * d / r.f * 0.5 + gf * ey * d / r.f / max(math.sin(elev), 0.2) * 0.5
    return tgt - fwd * d, tgt


# ----------------------------------------------------------------- scenes
def main(argv):
    draft = "--draft" in argv
    which = set(a for a in argv if not a.startswith("--"))
    todo = lambda n: (not which) or (n in which)
    ss = 1 if draft else 2
    half = dict(W=960, H=600) if "--half" in argv else {}
    os.makedirs(OUT, exist_ok=True)
    ifac = world.Course("IFAC 2026")
    J, Z = world.session(SID)
    boxes = J["boxes"]
    lo, hi = ifac.lo, ifac.hi
    cx, cy = (lo + hi) / 2

    def pose(t):
        i = int(round(t / 0.025))
        return float(Z["x"][i]), float(Z["y"][i]), float(Z["yaw"][i]), float(Z["steer"][i])

    def path(t0, t1, every=2):
        i0, i1 = int(round(t0 / 0.025)), int(round(t1 / 0.025))
        return np.stack([Z["x"][i0:i1 + 1:every], Z["y"][i0:i1 + 1:every]], 1)

    def track(course, with_boxes=None, car=None, hall=True, step=1, sides=10, lamps=0.55):
        def build(m):
            if hall:
                add_hall(m, course.lo, course.hi, lamps=lamps)
            add_hoses(m, course, step=step, sides=sides)
            if with_boxes:
                add_boxes(m, with_boxes)
            for c in (car or []):
                add_car(m, *c)
        return build

    def rel(t, fwd, left, h, lead=0.165):
        x, y, yaw, _ = pose(t)
        f = np.array([math.cos(yaw), math.sin(yaw)])
        l = np.array([-f[1], f[0]])
        return V3(np.array([x, y]) + f * (lead + fwd) + l * left, h)

    # --- main: the car on the back straight, front three-quarter, the course running away behind it
    if todo("hero"):
        t = 30.6
        x, y, yaw, _ = pose(t)
        render("hero", rel(t, 0.88, 0.66, 0.23), rel(t, -0.80, -1.02, 0.085), 32,
               track(ifac, car=[(x, y, yaw, 0.0)], sides=16), ss=ss, fog=0.055, spot=(x, y), spot_r=3.2, **half)

    # --- race setup: the whole course, boxes on it, the car at the start line
    if todo("venue"):
        x, y, yaw, _ = pose(0.0)
        eye, tgt = fit_camera(np.vstack(ifac.walls), math.radians(62), math.radians(31), 40, (800, 190, 1850, 1010))
        s = render("venue", eye, tgt, 40, track(ifac, with_boxes=boxes, car=[(x, y, yaw, 0.0)], step=2, hall=False), ss=ss,
                   fog=0.008, spot=(cx, cy), spot_r=13.0, **half)

    # --- driving: chase camera
    def chase(name, t, with_boxes, layers=False, back=1.32, up=0.56, fov=62):
        x, y, yaw, st = pose(t)
        f = np.array([math.cos(yaw), math.sin(yaw)])
        c = np.array([x, y]) + f * 0.165
        fxs, post = None, None
        if layers:
            plan = path(t + 0.1, t + 1.0, every=4)                 # 10 points, 0.1 s apart
            fxs = line_decal(plan, 0.035, gain=0.85, dots=plan, dot_r=0.05)
            pts, _ = scan_points(ifac, x, y, yaw, with_boxes or [])
            post = draw_points(pts)
        return render(name, rel(t, -back, 0.0, up), rel(t, 2.6, 0.0, 0.04), fov,
                      track(ifac, with_boxes=with_boxes, car=[(x, y, yaw, st)], sides=14, lamps=0.4), ss=ss, fog=0.045,
                      spot=tuple(c + f * 2.0), spot_r=5.0, floor_fx=fxs, post=post, spot_gain=(0.70, 0.55), **half)

    if todo("drive"):
        chase("drive", 26.8, boxes)
    if todo("drive_layers"):
        chase("drive_layers", 26.8, boxes, layers=True)
    if todo("drive_clear"):                      # the moment the boxes are taken away: with and without, blended
        chase("drive_clear_a", 30.0, boxes)
        chase("drive_clear_b", 30.0, None)
        a = np.asarray(Image.open(os.path.join(OUT, "drive_clear_a.png")), np.float32)
        b = np.asarray(Image.open(os.path.join(OUT, "drive_clear_b.png")), np.float32)
        Image.fromarray((a * 0.45 + b * 0.55).astype(np.uint8)).save(os.path.join(OUT, "drive_clear.png"))
        os.remove(os.path.join(OUT, "drive_clear_a.png"))
        os.remove(os.path.join(OUT, "drive_clear_b.png"))
    if todo("drive_pause"):
        chase("drive_pause", 44.0, None)

    # --- result: the car where the session ended
    if todo("finish"):
        t = 59.975
        x, y, yaw, st = pose(t)
        render("finish", rel(t, -1.0, -0.86, 0.32), rel(t, 0.85, 1.05, 0.12), 34,
               track(ifac, car=[(x, y, yaw, st)], sides=16), ss=ss, fog=0.05, spot=(x, y), spot_r=3.4, **half)

    # --- the course from above (result laps): paths and contacts are drawn by the page
    if todo("top"):
        eye, tgt = fit_camera(np.vstack(ifac.walls), math.radians(0), math.radians(58), 30, (800, 250, 1850, 950))
        s = render("top", eye, tgt, 30, track(ifac, hall=False, step=2), ss=ss, fog=0.003, spot=(cx, cy), spot_r=16.0,
                   joints=0.02, **half)
        laps = {str(lap["n"]): s.project(V3(path(lap["t_start"], lap["t_end"], every=4), 0.02))[:, :2] for lap in J["laps"]}
        write_overlay("top", laps=laps, all=s.project(V3(path(0, 59.975, every=4), 0.02))[:, :2],
                      contacts=s.project(V3([[c["x"], c["y"]] for c in J["contacts"]], 0.05))[:, :2],
                      boxes=s.project(V3([[b["x"], b["y"]] for b in boxes], 0.15))[:, :2])

    # --- records: the focused session's course with the line it drove
    if todo("replay"):
        eye, tgt = fit_camera(np.vstack(ifac.walls), math.radians(118), math.radians(34), 40, (800, 200, 1850, 1010))
        x, y, yaw, st = pose(59.975)
        render("replay", eye, tgt, 40, track(ifac, hall=False, step=2, car=[(x, y, yaw, st)]), ss=ss, fog=0.008,
               spot=(cx, cy), spot_r=13.0, floor_fx=line_decal(path(0.0, 59.975, every=3), 0.045, gain=0.55), **half)

    # --- garage: the car alone on a dark floor
    def studio(steer=0.0):
        def build(m):
            add_car(m, 0.0, 0.0, 0.0, steer)
        return build

    if todo("garage"):
        render("garage", V3([0.99, 0.80], 0.34), V3([-0.10, -0.42], 0.075), 27, studio(), ss=ss, fog=0.05,
               spot=(0.17, 0.0), spot_r=1.5, joints=0.0, spot_gain=(0.36, 1.05), **half)
    if todo("garage_steer"):
        render("garage_steer", V3([1.34, 0.44], 0.22), V3([-0.30, -0.36], 0.065), 27, studio(0.412), ss=ss, fog=0.05,
               spot=(0.25, 0.0), spot_r=1.4, joints=0.0, spot_gain=(0.36, 1.05), **half)
    if todo("garage_sensor"):
        # from behind and above: the blind sector opens toward the camera, the field of view runs out to 10 m
        ring = np.array([[0.27 + 1.7 * math.cos(q), 1.7 * math.sin(q)] for q in np.linspace(0, 2 * math.pi, 24)])
        eye, tgt = fit_camera(ring, math.radians(14), math.radians(24), 42, (660, 470, 2000, 1250), h=0.0)
        render("garage_sensor", eye, tgt, 42, studio(), ss=ss, fog=0.0,
               spot=(2.0, 0.0), spot_r=9.0, joints=0.0, spot_gain=(0.50, 0.25), floor_fx=fan_decal(0.27, 0.0, 0.0, gain=0.24), **half)

    # --- course select: the focused course
    if todo("course_iccas"):
        c25 = world.Course("ICCAS 2025")
        eye, tgt = fit_camera(np.vstack(c25.walls), math.radians(105), math.radians(33), 40, (800, 230, 1850, 1000))
        p, yaw = c25.frame(0)
        c = (c25.lo + c25.hi) / 2
        render("course_iccas", eye, tgt, 40, track(c25, hall=False, step=2, car=[(p[0], p[1], yaw, 0.0)]), ss=ss, fog=0.008,
               spot=tuple(c), spot_r=14.0, **half)

    # --- course edit: map16x07 from straight above, control points drawn by the page
    if todo("edit"):
        c16 = world.Course("map16x07")
        c = (c16.lo + c16.hi) / 2
        eye, tgt = fit_camera(np.vstack(c16.walls), math.radians(90), math.radians(89.5), 30, (790, 300, 1840, 900))
        p, yaw = c16.frame(300)
        s = render("edit", eye, tgt, 30, track(c16, hall=False, car=[(p[0], p[1], yaw, 0.0)], sides=12), ss=ss, fog=0.0,
                   spot=tuple(c), spot_r=14.0, joints=0.05, spot_gain=(0.75, 0.35), **half)
        hand = [s.project(V3(w[::22], 0.30))[:, :2] for w in c16.walls]
        nrm = np.array([-math.sin(yaw), math.cos(yaw)])
        gate = s.project(V3([p + nrm * 0.75, p - nrm * 0.75], 0.02))[:, :2]
        write_overlay("edit", outer=hand[0], inner=hand[1], gate=gate)

    # --- bag import: the scan map of the venue laid on the floor
    if todo("bag"):
        eye, tgt = fit_camera(np.vstack(ifac.walls), math.radians(62), math.radians(40), 40, (800, 190, 1850, 1010))
        x, y, yaw, _ = pose(0.0)
        render("bag", eye, tgt, 40, lambda m: add_car(m, x, y, yaw), ss=ss, fog=0.008, spot=(cx, cy), spot_r=13.0,
               floor_fx=grid_decal(ifac), joints=0.0, **half)

    # --- training: the run's car on its course, low and far
    if todo("train"):
        t = 44.0
        x, y, yaw, st = pose(t)
        eye, tgt = fit_camera(np.vstack(ifac.walls), math.radians(152), math.radians(22), 34, (760, 780, 1880, 1150))
        render("train", eye, tgt, 34, track(ifac, with_boxes=boxes, car=[(x, y, yaw, st)], hall=False), ss=ss, fog=0.012,
               spot=(cx, cy), spot_r=12.0, floor_fx=line_decal(path(41.6, 44.0), 0.05, gain=0.7), **half)


if __name__ == "__main__":
    main(sys.argv[1:])
