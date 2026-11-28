#!/usr/bin/env python3
"""Still stand-ins for the 3D scene in the round-2 mockups.

A small CPU triangle rasteriser (numpy, z-buffer, per-vertex colour) that draws
a 1/10 car, duct-hose track walls and boxes on a hall floor.  No GPU.
Writes PNGs to ../assets and the top-down course outlines to ../assets/courses.js.

    python3 -I tools/scene.py [name ...]      # no name = everything
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "assets")


# ----------------------------------------------------------------- courses
def catmull(ctrl, step=0.08):
    """Closed Catmull-Rom spline through ctrl, resampled at ~step metres."""
    P = np.asarray(ctrl, float)
    n = len(P)
    pts = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        for t in np.linspace(0, 1, 40, endpoint=False):
            t2, t3 = t * t, t * t * t
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    pts = np.array(pts)
    d = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    m = max(8, int(s[-1] / step))
    u = np.linspace(0, s[-1], m, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    return np.stack([np.interp(u, s, closed[:, 0]), np.interp(u, s, closed[:, 1])], 1)


def offset(c, d):
    t = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    return c + nrm * d


def smooth(p, k=9):
    ker = np.ones(k) / k
    ext = np.vstack([p[-k:], p, p[:k]])
    return np.stack([np.convolve(ext[:, i], ker, "same")[k:-k] for i in range(2)], 1)


COURSE_CTRL = {
    # tall hall, long left straight, folded infield (the 2026 venue's general shape)
    "kr26": dict(name="2026 한국 대회장", width=2.0, ctrl=[
        (1.2, 2.2), (1.2, 7.0), (1.2, 12.0), (1.3, 16.6), (2.6, 19.4), (5.0, 20.3), (7.6, 19.3),
        (8.7, 16.8), (7.8, 14.3), (5.6, 13.2), (4.6, 11.0), (5.6, 8.9), (7.9, 7.9), (8.9, 5.6),
        (8.1, 3.0), (5.8, 1.5), (3.2, 0.9)]),
    "kr25": dict(name="2025 한국 대회장", width=2.1, ctrl=[
        (1.0, 1.4), (1.0, 6.0), (2.2, 8.6), (5.0, 8.8), (6.4, 10.6), (5.6, 13.0), (7.4, 14.6),
        (11.0, 14.4), (13.4, 12.6), (13.6, 9.0), (11.6, 7.2), (9.6, 5.6), (10.6, 3.0), (9.0, 1.0),
        (5.0, 0.8)]),
    "lab": dict(name="연구실 복도", width=1.7, ctrl=[
        (0.9, 1.7), (0.9, 4.0), (2.4, 5.4), (9.0, 5.4), (15.0, 5.4), (16.6, 4.0), (16.6, 1.9),
        (15.2, 0.75), (13.4, 1.3), (11.6, 2.7), (9.0, 3.0), (6.2, 2.7), (4.3, 1.3), (2.4, 0.75)]),
    "oval": dict(name="연습 타원", width=2.2, ctrl=[
        (1.2, 3.0), (1.2, 6.0), (3.0, 8.0), (9.0, 8.0), (10.8, 6.0), (10.8, 3.0), (9.0, 1.0), (3.0, 1.0)]),
}


def build_course(key):
    c = COURSE_CTRL[key]
    cen = catmull(c["ctrl"])
    left = smooth(offset(cen, c["width"] / 2))
    right = smooth(offset(cen, -c["width"] / 2))
    seg = np.linalg.norm(np.diff(np.vstack([cen, cen[:1]]), axis=0), axis=1)
    return dict(key=key, name=c["name"], cen=cen, left=left, right=right, length=float(seg.sum()),
                width=c["width"])


def write_courses_js():
    out = {}
    for key in COURSE_CTRL:
        c = build_course(key)
        allp = np.vstack([c["left"], c["right"]])
        mn, mx = allp.min(0) - 0.5, allp.max(0) + 0.5

        def path(p):
            # y flipped so +y is up on screen
            q = np.stack([p[:, 0] - mn[0], mx[1] - p[:, 1]], 1)[::3]
            return "M" + " L".join("%.2f %.2f" % (x, y) for x, y in q) + " Z"

        i0 = 4
        s = c["cen"][i0]
        t = c["cen"][i0 + 3] - c["cen"][i0]
        n = len(c["cen"])
        boxes = []
        for frac, side in ((0.22, 0.45), (0.51, -0.4), (0.78, 0.35)):
            i = int(n * frac)
            tt = c["cen"][(i + 2) % n] - c["cen"][i]
            tt /= np.linalg.norm(tt)
            b = c["cen"][i] + np.array([-tt[1], tt[0]]) * side * c["width"] / 2
            boxes.append([round(float(b[0] - mn[0]), 2), round(float(mx[1] - b[1]), 2)])
        out[key] = dict(name=c["name"], w=round(float(mx[0] - mn[0]), 2), h=round(float(mx[1] - mn[1]), 2),
                        length=round(c["length"], 1), cen=path(c["cen"]), left=path(c["left"]),
                        right=path(c["right"]),
                        start=[round(float(s[0] - mn[0]), 2), round(float(mx[1] - s[1]), 2),
                               round(math.degrees(math.atan2(-t[1], t[0])), 1)],
                        boxes=boxes)
    with open(os.path.join(OUT, "courses.js"), "w") as f:
        f.write("window.COURSES = " + json.dumps(out, ensure_ascii=False) + ";\n")
    print("courses.js", {k: v["length"] for k, v in out.items()})


# ----------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, W, H, ss=2):
        self.ss, self.W, self.H = ss, W * ss, H * ss
        self.col = np.zeros((self.H, self.W, 3), np.float32)
        self.z = np.zeros((self.H, self.W), np.float32)      # inverse depth, 0 = empty
        self.shadow = np.zeros((self.H, self.W), np.float32)

    def camera(self, eye, target, fov_deg):
        self.eye = np.asarray(eye, float)
        fwd = np.asarray(target, float) - self.eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 1, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.Rv = np.stack([right, up, fwd])
        self.f = 0.5 * self.H / math.tan(math.radians(fov_deg) / 2)
        self.cx, self.cy = self.W / 2, self.H / 2

    def _project(self, P):
        cam = (P - self.eye) @ self.Rv.T
        z = cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = self.cx + self.f * cam[..., 0] / z
            sy = self.cy - self.f * cam[..., 1] / z
        return sx, sy, z

    def tris(self, P, C=None, shadow=False):
        sx, sy, z = self._project(P)
        keep = (z > 0.05).all(1)
        sx, sy, z = sx[keep], sy[keep], z[keep]
        C = None if C is None else C[keep]
        iz = 1.0 / z
        W, H = self.W, self.H
        for i in range(len(sx)):
            x0, x1, x2 = sx[i]
            y0, y1, y2 = sy[i]
            minx = max(int(math.floor(min(x0, x1, x2))), 0)
            maxx = min(int(math.ceil(max(x0, x1, x2))), W - 1)
            miny = max(int(math.floor(min(y0, y1, y2))), 0)
            maxy = min(int(math.ceil(max(y0, y1, y2))), H - 1)
            if minx > maxx or miny > maxy:
                continue
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(area) < 1e-9:
                continue
            X = np.arange(minx, maxx + 1) + 0.5
            Y = (np.arange(miny, maxy + 1) + 0.5)[:, None]
            w0 = ((x1 - X) * (y2 - Y) - (x2 - X) * (y1 - Y)) / area
            w1 = ((x2 - X) * (y0 - Y) - (x0 - X) * (y2 - Y)) / area
            w2 = 1 - w0 - w1
            m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not m.any():
                continue
            if shadow:
                self.shadow[miny:maxy + 1, minx:maxx + 1][m] = 1.0
                continue
            zz = w0 * iz[i, 0] + w1 * iz[i, 1] + w2 * iz[i, 2]
            sub = self.z[miny:maxy + 1, minx:maxx + 1]
            m &= zz > sub
            if not m.any():
                continue
            sub[m] = zz[m]
            c = w0[..., None] * C[i, 0] + w1[..., None] * C[i, 1] + w2[..., None] * C[i, 2]
            self.col[miny:maxy + 1, minx:maxx + 1][m] = c[m]


# ----------------------------------------------------------------- geometry
class Mesh:
    """Triangles with per-vertex normals and albedo; lit at render time."""

    def __init__(self):
        self.P, self.N, self.A = [], [], []

    def add(self, P, N, A):
        P = np.asarray(P, float).reshape(-1, 3, 3)
        N = np.asarray(N, float).reshape(-1, 3, 3)
        A = np.broadcast_to(np.asarray(A, float), P.shape) if np.ndim(A) == 1 else np.asarray(A, float)
        self.P.append(P); self.N.append(N); self.A.append(np.array(A).reshape(-1, 3, 3))

    def arrays(self):
        return np.concatenate(self.P), np.concatenate(self.N), np.concatenate(self.A)

    def quad(self, a, b, c, d, col):
        a, b, c, d = map(np.asarray, (a, b, c, d))
        n = np.cross(b - a, d - a)
        n = n / (np.linalg.norm(n) + 1e-12)
        self.add([[a, b, c], [a, c, d]], np.tile(n, (2, 3, 1)), col)

    def box(self, center, size, yaw=0.0, col=(0.5, 0.5, 0.5), xf=None):
        cx, cy, cz = center
        sx, sy, sz = [s / 2 for s in size]
        c, s = math.cos(yaw), math.sin(yaw)
        v = []
        for dx in (-sx, sx):
            for dy in (-sy, sy):
                for dz in (-sz, sz):
                    v.append([cx + dx * c + dz * s, cy + dy, cz - dx * s + dz * c])
        v = np.array(v)
        if xf is not None:
            v = xf(v)
        f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        ctr = v.mean(0)
        for a, b, c2, d in f:
            n = np.cross(v[b] - v[a], v[d] - v[a])
            if np.dot(n, v[a] - ctr) < 0:
                a, b, c2, d = a, d, c2, b
            self.quad(v[a], v[b], v[c2], v[d], col)

    def cyl(self, p0, p1, r, col, n=18, caps=True, capcol=None, xf=None):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        if xf is not None:
            p0, p1 = xf(p0[None])[0], xf(p1[None])[0]
        ax = p1 - p0
        ax /= np.linalg.norm(ax)
        ref = np.array([0, 1, 0]) if abs(ax[1]) < 0.9 else np.array([1, 0, 0])
        u = np.cross(ax, ref); u /= np.linalg.norm(u)
        v = np.cross(ax, u)
        ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
        ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
        for i in range(n):
            j = (i + 1) % n
            a, b = p0 + r * ring[i], p0 + r * ring[j]
            c, d = p1 + r * ring[j], p1 + r * ring[i]
            self.add([[a, b, c], [a, c, d]],
                     [[ring[i], ring[j], ring[j]], [ring[i], ring[j], ring[i]]], col)
            if caps:
                cc = capcol if capcol is not None else col
                self.add([[p0, b, a]], np.tile(-ax, (1, 3, 1)), cc)
                self.add([[p1, d, c]], np.tile(ax, (1, 3, 1)), cc)

    def tube(self, line2d, r, col, y=None, sides=10):
        """Closed duct hose lying on the floor along a 2D polyline (x, z)."""
        y = r if y is None else y
        p = np.asarray(line2d, float)
        n = len(p)
        t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        side = np.stack([-t[:, 1], t[:, 0]], 1)
        ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)
        # ring[i, k] position and normal
        nx = side[:, None, 0] * np.cos(ang)[None]
        nz = side[:, None, 1] * np.cos(ang)[None]
        ny = np.broadcast_to(np.sin(ang)[None], nx.shape)
        Nn = np.stack([nx, ny, nz], -1)
        Pp = np.stack([p[:, None, 0] + r * nx, y + r * ny, p[:, None, 1] + r * nz], -1)
        base = np.asarray(col, float)
        rib = np.where(np.arange(n) % 2 == 0, 1.0, 0.80)          # the ribs of a flexible duct
        for i in range(n):
            j = (i + 1) % n
            ci, cj = base * rib[i], base * rib[j]
            for k in range(sides):
                l = (k + 1) % sides
                self.add([[Pp[i, k], Pp[j, k], Pp[j, l]], [Pp[i, k], Pp[j, l], Pp[i, l]]],
                         [[Nn[i, k], Nn[j, k], Nn[j, l]], [Nn[i, k], Nn[j, l], Nn[i, l]]],
                         [[ci, cj, cj], [ci, cj, ci]])


def car(mesh, pos, yaw, accent=(0.86, 0.16, 0.12)):
    """A bare 1/10 research car: 0.53 m long, sensors on a deck. +x is forward before yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    px, pz = pos

    def xf(v):
        v = np.asarray(v, float)
        return np.stack([px + v[:, 0] * c + v[:, 2] * s, v[:, 1], pz - v[:, 0] * s + v[:, 2] * c], 1)

    def B(center, size, col):
        mesh.box(center, size, 0.0, col, xf=xf)

    black, dark, mid, alu = (0.06, 0.06, 0.068), (0.13, 0.13, 0.145), (0.30, 0.31, 0.33), (0.70, 0.71, 0.74)
    # wheels + hubs
    for wx in (0.165, -0.165):
        for wz in (0.132, -0.132):
            sg = 1 if wz > 0 else -1
            mesh.cyl([wx, 0.055, wz - 0.024], [wx, 0.055, wz + 0.024], 0.055, black, n=28, capcol=(0.06, 0.06, 0.065), xf=xf)
            mesh.cyl([wx, 0.055, wz + sg * 0.0245], [wx, 0.055, wz + sg * 0.027], 0.034, alu, n=24, xf=xf)
            mesh.cyl([wx, 0.055, wz + sg * 0.027], [wx, 0.055, wz + sg * 0.029], 0.012, dark, n=10, xf=xf)
            mesh.cyl([wx, 0.055, wz - sg * 0.02], [wx, 0.055, wz - sg * 0.06], 0.008, mid, n=6, xf=xf)
    # chassis tub, bumpers
    B([0.0, 0.052, 0.0], [0.40, 0.014, 0.115], dark)
    B([0.255, 0.045, 0.0], [0.018, 0.03, 0.21], black)
    B([0.235, 0.05, 0.0], [0.03, 0.012, 0.10], dark)
    B([-0.25, 0.055, 0.0], [0.02, 0.03, 0.16], black)
    # shock towers + shocks
    for wx in (0.175, -0.175):
        B([wx, 0.095, 0.0], [0.008, 0.06, 0.10], dark)
        for wz in (0.075, -0.075):
            mesh.cyl([wx, 0.062, wz * 1.3], [wx, 0.122, wz * 0.62], 0.007, alu, n=8, xf=xf)
    # battery, motor, speed controller on the tub
    B([-0.02, 0.078, 0.03], [0.14, 0.036, 0.046], (0.12, 0.2, 0.5))
    mesh.cyl([-0.14, 0.085, -0.035], [-0.085, 0.085, -0.035], 0.02, alu, n=12, xf=xf)
    B([0.07, 0.074, -0.03], [0.07, 0.03, 0.045], dark)
    # sensor deck on four posts
    for dx in (0.12, -0.12):
        for dz in (0.065, -0.065):
            mesh.cyl([dx, 0.06, dz], [dx, 0.128, dz], 0.004, alu, n=6, xf=xf)
    B([0.0, 0.131, 0.0], [0.30, 0.006, 0.165], accent)
    # computer with heat-sink fins
    B([-0.045, 0.155, 0.0], [0.105, 0.042, 0.105], dark)
    for k in range(9):
        B([-0.045, 0.184, -0.044 + k * 0.011], [0.098, 0.016, 0.004], mid)
    # lidar: base + window + cap
    B([0.095, 0.151, 0.0], [0.05, 0.034, 0.05], black)
    mesh.cyl([0.095, 0.168, 0.0], [0.095, 0.196, 0.0], 0.022, (0.42, 0.2, 0.06), n=20, xf=xf)
    mesh.cyl([0.095, 0.196, 0.0], [0.095, 0.207, 0.0], 0.0235, black, n=20, xf=xf)
    # antenna
    mesh.cyl([-0.13, 0.134, 0.06], [-0.13, 0.29, 0.06], 0.002, mid, n=5, xf=xf)
    mesh.cyl([-0.13, 0.29, 0.06], [-0.13, 0.30, 0.06], 0.005, black, n=8, xf=xf)


def render(name, W, H, eye, target, fov, build, theme="dark", ss=2, alpha=False, spot=None,
           spot_r=3.0, fog=0.035, flip=False):
    if theme == "dark":
        bg = np.array([0.055, 0.058, 0.066]); floor = np.array([0.235, 0.242, 0.258])
        light_dir = np.array([-0.45, -1.0, 0.35]); amb_lo, amb_hi, diff = 0.30, 0.55, 0.95
        line = 0.03
    else:
        bg = np.array([0.925, 0.920, 0.905]); floor = np.array([0.800, 0.792, 0.772])
        light_dir = np.array([-0.4, -1.0, 0.45]); amb_lo, amb_hi, diff = 0.45, 0.72, 0.55
        line = -0.035
    r = Renderer(W, H, ss)
    r.camera(eye, target, fov)
    # key light over the camera's left shoulder, so the side we look at is lit
    flat = np.array([r.Rv[2][0], 0.0, r.Rv[2][2]]); flat /= np.linalg.norm(flat)
    rgt = np.array([r.Rv[0][0], 0.0, r.Rv[0][2]]); rgt /= np.linalg.norm(rgt)
    light_dir = flat * 0.55 + rgt * 0.45 + np.array([0.0, -1.0, 0.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    mesh = Mesh()
    build(mesh)
    P, N, A = mesh.arrays()
    N = N / (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12)
    ndl = np.clip(-(N @ light_dir), 0, 1)
    hemi = amb_lo + (amb_hi - amb_lo) * (N[..., 1] * 0.5 + 0.5)
    C = A * (hemi + diff * ndl)[..., None]
    unlit = (A < 0).any(-1)
    C[unlit] = -A[unlit]
    # distance fog on objects
    dist = np.linalg.norm(P - r.eye, axis=-1)
    fg = (1 - np.exp(-dist * fog))[..., None]
    if not alpha:
        C = C * (1 - fg) + bg * fg
    r.tris(P, C.astype(np.float32))
    # shadows: project onto the floor along the light, and straight down for contact
    for L, blur, k in ((light_dir, 5 * ss, 0.5), (np.array([0.0, -1.0, 0.0]), 9 * ss, 0.38)):
        r.shadow[:] = 0
        t = P[..., 1] / -L[1]
        S = P + L * t[..., None]
        S[..., 1] = 0.0
        r.tris(S, shadow=True)
        sh = np.asarray(Image.fromarray((r.shadow * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur)),
                        np.float32) / 255
        if L is light_dir:
            shadow_total = sh * k
        else:
            shadow_total = 1 - (1 - shadow_total) * (1 - sh * k)
    # floor per pixel
    ys, xs = np.mgrid[0:r.H, 0:r.W].astype(np.float32)
    dx = (xs + 0.5 - r.cx) / r.f
    dy = -(ys + 0.5 - r.cy) / r.f
    D = dx[..., None] * r.Rv[0] + dy[..., None] * r.Rv[1] + r.Rv[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -r.eye[1] / D[..., 1]
    hit = (D[..., 1] < -1e-5) & (t > 0)
    fx = r.eye[0] + D[..., 0] * t
    fz = r.eye[2] + D[..., 2] * t
    fd = np.where(hit, t * np.linalg.norm(D, axis=-1), 1e6)
    fcol = np.broadcast_to(floor, (r.H, r.W, 3)).copy()
    # 1 m floor joints, kept thin with distance
    px = fd / r.f * 1.6
    for g in (fx, fz):
        dgrid = np.abs(g - np.round(g))
        fcol += (np.clip(1 - dgrid / np.maximum(px, 0.004), 0, 1) * line * np.clip(6 / np.maximum(fd, 1), 0, 1))[..., None]
    if spot is not None:
        sd = np.hypot(fx - spot[0], fz - spot[1])
        lo, gain = (0.62, 0.75) if theme == "dark" else (0.90, 0.10)
        fcol *= (lo + gain * np.exp(-(sd / spot_r) ** 2))[..., None]
    fcol *= (1 - shadow_total)[..., None]
    ff = (1 - np.exp(-fd * fog))[..., None]
    fcol = fcol * (1 - ff) + bg * ff
    back = np.where(hit[..., None], fcol, bg)
    obj = r.z > 0
    img = np.where(obj[..., None], r.col, back)
    img = np.clip(img, 0, 1) ** (1 / 1.0)
    out = Image.fromarray((img * 255).astype(np.uint8))
    if alpha:
        a = np.clip(obj.astype(np.float32) + shadow_total * 0.9 * (~obj), 0, 1)
        rgb = np.where(obj[..., None], r.col, 0.0)
        out = Image.fromarray(np.dstack([np.clip(rgb, 0, 1) * 255, a * 255]).astype(np.uint8), "RGBA")
    out = out.resize((W, H), Image.LANCZOS)
    if flip:    # the 2D course outlines put +x on the right; a y-up view along +z puts it on the left
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    path = os.path.join(OUT, name)
    out.save(path)
    print(name, out.size, len(P), "tris")


# ----------------------------------------------------------------- scenes
HOSE_DARK = (0.66, 0.67, 0.69)
HOSE_LIGHT = (0.97, 0.97, 0.96)
CARD = (0.62, 0.47, 0.30)


def hall(m, theme):
    """Walls and ceiling light strips of an exhibition hall around the course."""
    wall = (0.34, 0.35, 0.38) if theme == "dark" else (0.93, 0.92, 0.90)
    lamp = (-1.0, -0.98, -0.92) if theme == "dark" else (-1.0, -1.0, -1.0)
    for c, sz in (([-9, 3.0, 11], [0.3, 6.0, 50]), ([19, 3.0, 11], [0.3, 6.0, 50]),
                  ([5, 3.0, -14], [28, 6.0, 0.3]), ([5, 3.0, 36], [28, 6.0, 0.3])):
        m.box(c, sz, 0.0, wall)
    for x in np.arange(-6.5, 17, 3.6):
        for z in np.arange(-12, 34, 5.0):
            m.box([x, 6.0, z], [0.28, 0.05, 3.2], 0.0, lamp)


def track_scene(course, hose, boxes=True, car_at=None, car_yaw=None, accent=(0.86, 0.16, 0.12), rival=None, step=2,
                hall_theme=None):
    def build(m):
        if hall_theme:
            hall(m, hall_theme)
        m.tube(course["left"][::step], 0.15, hose, sides=10 if step > 1 else 16)
        m.tube(course["right"][::step], 0.15, hose, sides=10 if step > 1 else 16)
        n = len(course["cen"])
        if boxes:
            for frac, side in ((0.22, 0.45), (0.51, -0.4), (0.78, 0.35)):
                i = int(n * frac)
                tt = course["cen"][(i + 2) % n] - course["cen"][i]
                tt /= np.linalg.norm(tt)
                b = course["cen"][i] + np.array([-tt[1], tt[0]]) * side * course["width"] / 2
                m.box([b[0], 0.16, b[1]], [0.34, 0.32, 0.30], yaw=math.atan2(tt[0], tt[1]) + 0.3, col=CARD)
        if car_at is not None:
            car(m, car_at, car_yaw, accent)
        if rival is not None:
            car(m, rival[0], rival[1], (0.10, 0.36, 0.78))
    return build


def frame_on_track(course, i, back=0.0, lateral=0.0):
    c = course["cen"]
    n = len(c)
    p = c[i % n]
    t = c[(i + 3) % n] - c[(i - 3) % n]
    t /= np.linalg.norm(t)
    nrm = np.array([-t[1], t[0]])
    pos = p - t * back + nrm * lateral
    yaw = math.atan2(-t[1], t[0])       # car +x forward; world (x, z)
    return pos, yaw, t, nrm


RED, YELLOW, ORANGE, INK = (0.80, 0.13, 0.09), (0.96, 0.74, 0.02), (0.95, 0.36, 0.06), (0.16, 0.16, 0.18)


def main(which):
    os.makedirs(OUT, exist_ok=True)
    kr26 = build_course("kr26")
    todo = lambda n: (not which) or (n in which)
    if todo("courses"):
        write_courses_js()
    pos, yaw, t, nrm = frame_on_track(kr26, 60)
    P3 = lambda p, y=0.0: np.array([p[0], y, p[1]])
    T, Nr = P3(t), P3(nrm)

    def hero(name, theme, hose, accent, fog):
        # from the track ahead of the car, low, looking back down the start straight
        eye = P3(pos) + T * 0.86 + Nr * 0.60 + [0, 0.23, 0]
        tgt = P3(pos, 0.17) + T * -0.75 + Nr * -0.50
        render(name, 1920, 1200, eye, tgt, 38, track_scene(kr26, hose, car_at=pos, car_yaw=yaw, accent=accent, step=1, hall_theme=theme),
               theme, spot=pos, spot_r=3.2, fog=fog)

    def venue(name, theme, hose, accent, fog):
        eye = np.array([-8.2, 8.0, -5.0]); tgt = np.array([-1.6, 0.0, 8.0])
        render(name, 1920, 1200, eye, tgt, 52, track_scene(kr26, hose, car_at=pos, car_yaw=yaw, accent=accent),
               theme, spot=(4.6, 9.5), spot_r=12.0, fog=fog, flip=True)

    if todo("hero_dark"):
        hero("hero_dark.png", "dark", HOSE_DARK, RED, 0.06)
    if todo("venue_dark"):
        venue("venue_dark.png", "dark", HOSE_DARK, RED, 0.012)
    if todo("hero_light"):
        hero("hero_light.png", "light", HOSE_LIGHT, YELLOW, 0.035)
    if todo("venue_light"):
        venue("venue_light.png", "light", HOSE_LIGHT, YELLOW, 0.008)
    for nm, acc in (("red", RED), ("yellow", YELLOW), ("orange", ORANGE), ("ink", INK)):
        if todo("car_" + nm):
            render("car_%s.png" % nm, 1400, 900, np.array([0.80, 0.34, -0.74]), np.array([0.0, 0.085, 0.02]), 30,
                   lambda m, acc=acc: car(m, (0, 0), 0.0, acc), "light", alpha=True)
        if todo("carside_" + nm):
            render("carside_%s.png" % nm, 1400, 620, np.array([0.10, 0.20, -1.3]), np.array([0.0, 0.10, 0.0]), 24,
                   lambda m, acc=acc: car(m, (0, 0), 0.0, acc), "light", alpha=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]))
