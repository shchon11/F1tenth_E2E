#!/usr/bin/env python3
"""Real data for the round-3 stand-in scenes. Read-only, numpy/cv2 only (no torch, no GPU).

  maps     the team's measured maps (occupancy grid + traced centre line) -> wall polylines
  car      the project's car model (f1tenth_car.glb) -> triangles with vertex colours
  session  a recorded TENTH session (result json + steps npz)
  scan     a 270 deg / 1081 beam / 10 m scan cast against the grid from a pose
"""
import glob
import json
import math
import os
import struct

import cv2
import numpy as np

REPO = "/home/shchon11/F1tenth_E2E"
ASSET = os.path.join(REPO, "f1sim/f1sim/assets")
CL_CACHE = os.path.expanduser("~/.cache/f1sim/centerlines")
RESULTS = "/data/tenth/results"

#: screen name -> (map yaml, centre-line cache prefix)
MAPS = {
    "IFAC 2026": (os.path.expanduser("~/.f1sim/maps/korea26_scan/korea26_scan.yaml"), "korea26_scan_"),
    "ICCAS 2025": (os.path.join(REPO, "external/korea_teams/KORA_K3/src/kora_k3/maps/real.yaml"), "korea_2025_iccas_"),
    "ICRA 2022": (os.path.join(REPO, "external/f1tenth-racing-stack-ICRA22/maps/icra_2.yaml"), "icra2022_"),
    "map12x16": (os.path.join(ASSET, "maps/map12x16.yaml"), "map12x16_"),
    "map16x07": (os.path.join(ASSET, "maps/map16x07.yaml"), "map16x07_"),
}


def _yaml(path):
    d = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    d["resolution"] = float(d["resolution"])
    d["origin"] = [float(x) for x in d["origin"].strip("[]").split(",")]
    d["image"] = d["image"].strip("'\"")
    return d


def _closed_len(p):
    return float(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum())


def resample(p, step):
    d = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    m = max(8, int(s[-1] / step))
    u = np.linspace(0, s[-1], m, endpoint=False)
    q = np.vstack([p, p[:1]])
    return np.stack([np.interp(u, s, q[:, 0]), np.interp(u, s, q[:, 1])], 1)


def smooth(p, k=7):
    ker = np.ones(k) / k
    ext = np.vstack([p[-k:], p, p[:k]])
    return np.stack([np.convolve(ext[:, i], ker, "same")[k:-k] for i in range(2)], 1)


class Course:
    """A measured map: free-space band the centre line runs in, its two bounding wall loops."""

    def __init__(self, name):
        self.name = name
        ypath, prefix = MAPS[name]
        y = _yaml(ypath)
        img = os.path.join(os.path.dirname(ypath), y["image"])
        if not os.path.exists(img):
            base = os.path.splitext(img)[0]
            img = next(base + e for e in (".pgm", ".png") if os.path.exists(base + e))
        g = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        self.res, self.origin = y["resolution"], np.array(y["origin"][:2])
        self.H, self.W = g.shape
        self.gray = g
        cl = sorted(glob.glob(os.path.join(CL_CACHE, prefix + "*.csv")))[0]
        self.cen = np.loadtxt(cl, delimiter=",")[:, :2]
        self.length = _closed_len(self.cen)
        occ = (g < 255 * (1 - 0.65)).astype(np.uint8)            # p_occ > occupied_thresh
        self.solid = cv2.dilate(occ, np.ones((3, 3), np.uint8)) > 0
        # walls: from every centre-line point, the first occupied cell to the left and to the right
        n = len(self.cen)
        t = np.roll(self.cen, -3, 0) - np.roll(self.cen, 3, 0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        r = np.arange(0.05, 4.0, 0.02)
        self.walls, dist = [], []
        for sg in (1.0, -1.0):
            pts = self.cen[:, None, :] + sg * nrm[:, None, :] * r[None, :, None]
            ij = self.to_px(pts).astype(int)
            ok = (ij[..., 0] >= 0) & (ij[..., 0] < self.W) & (ij[..., 1] >= 0) & (ij[..., 1] < self.H)
            hit = np.zeros(ok.shape, bool)
            hit[ok] = self.solid[ij[..., 1][ok], ij[..., 0][ok]]
            d = np.where(hit.any(1), r[hit.argmax(1)], np.nan)
            # drop jumps (a ray through a gap in the hose or into the next lane), then fill along the lap
            med = np.array([np.nanmedian(np.take(d, range(i - 12, i + 13), mode="wrap")) for i in range(n)])
            bad = np.isnan(d) | (np.abs(d - med) > 0.35)
            idx = np.arange(n)
            good = idx[~bad]
            d = np.interp(idx, np.concatenate([good - n, good, good + n]), np.tile(d[~bad], 3))
            dist.append(d)
            w = self.cen + sg * nrm * d[:, None]
            self.walls.append(smooth(resample(smooth(w, 5), 0.06), 7))
        self.width = dist[0] + dist[1]
        self.min_width = float(np.percentile(self.width, 2))
        allp = np.vstack(self.walls)
        self.lo, self.hi = allp.min(0), allp.max(0)
        self.size = self.hi - self.lo

    def to_px(self, xy):
        xy = np.asarray(xy, float)
        return np.stack([(xy[..., 0] - self.origin[0]) / self.res,
                         self.H - 1 - (xy[..., 1] - self.origin[1]) / self.res], -1)

    def to_world(self, ij):
        ij = np.asarray(ij, float)
        return np.stack([ij[..., 0] * self.res + self.origin[0],
                         (self.H - 1 - ij[..., 1]) * self.res + self.origin[1]], -1)

    def frame(self, i):
        """pose on the centre line: position, heading."""
        n = len(self.cen)
        p = self.cen[i % n]
        t = self.cen[(i + 4) % n] - self.cen[(i - 4) % n]
        return p, math.atan2(t[1], t[0])

    def scan(self, x, y, yaw, boxes=(), fov=math.radians(270), beams=1081, rmax=10.0, step=0.02):
        """Ranges of one scan from (x, y, yaw): ray march in the grid, boxes as oriented rectangles."""
        ang = yaw + np.linspace(-fov / 2, fov / 2, beams)
        r = np.arange(0.06, rmax, step)
        px = x + np.cos(ang)[:, None] * r[None]
        py = y + np.sin(ang)[:, None] * r[None]
        ij = self.to_px(np.stack([px, py], -1)).astype(int)
        inside = (ij[..., 0] >= 0) & (ij[..., 0] < self.W) & (ij[..., 1] >= 0) & (ij[..., 1] < self.H)
        hit = np.ones(px.shape, bool)
        hit[inside] = self.solid[ij[..., 1][inside], ij[..., 0][inside]]
        for b in boxes:
            c, s = math.cos(-b["yaw"]), math.sin(-b["yaw"])
            lx = (px - b["x"]) * c - (py - b["y"]) * s
            ly = (px - b["x"]) * s + (py - b["y"]) * c
            hit |= (np.abs(lx) < b["w"] / 2) & (np.abs(ly) < b["d"] / 2)
        first = hit.argmax(1)
        rng = np.where(hit.any(1), r[first], np.nan)
        return ang, rng


# ----------------------------------------------------------------- car model
MAT = {  # fallback albedo by node-name suffix when a primitive has no vertex colour
    "plastic": (0.08, 0.08, 0.09), "metal": (0.62, 0.63, 0.66), "blue": (0.12, 0.25, 0.62), "rubber": (0.05, 0.05, 0.055),
    "glass": (0.25, 0.14, 0.05), "paint": (0.8, 0.13, 0.09), "pcb": (0.1, 0.4, 0.2)}


def load_car():
    """-> dict part -> list of (name, P[n,3,3], N[n,3,3], C[n,3,3]) in the model frame
    (x forward from the rear axle, y left, z up, metres). Parts: chassis, lidar, wheel_fl/fr/rl/rr with
    their node offsets applied; wheels also return their pivot."""
    f = open(os.path.join(ASSET, "f1tenth_car.glb"), "rb").read()
    l0, _ = struct.unpack("<II", f[12:20])
    j = json.loads(f[20:20 + l0])
    b0 = 20 + l0
    l1, _ = struct.unpack("<II", f[b0:b0 + 8])
    B = f[b0 + 8:b0 + 8 + l1]

    def acc(i):
        a = j["accessors"][i]
        bv = j["bufferViews"][a["bufferView"]]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        dt = {5126: "<f4", 5123: "<u2", 5125: "<u4", 5121: "u1"}[a["componentType"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(B, dt, a["count"] * n, off).reshape(a["count"], n).astype(np.float64 if dt == "<f4" else np.int64)

    def prims(node):
        out = []
        for p in j["meshes"][node["mesh"]]["primitives"]:
            P = acc(p["attributes"]["POSITION"])
            N = acc(p["attributes"]["NORMAL"])
            if "COLOR_0" in p["attributes"]:
                C = acc(p["attributes"]["COLOR_0"])[:, :3]
                C = C / (255.0 if C.max() > 1.5 else 1.0)
            else:
                C = np.tile(MAT[node["name"].rsplit("_", 1)[-1]], (len(P), 1))
            idx = acc(p["indices"]).reshape(-1, 3) if "indices" in p else np.arange(len(P)).reshape(-1, 3)
            out.append((node["name"], P[idx], N[idx], C[idx]))
        return out

    parts = {"chassis": []}
    pivots = {}
    for n in j["nodes"]:
        if "children" in n:
            M = np.array(n["matrix"]).reshape(4, 4).T
            pivots[n["name"]] = M[:3, 3].copy()
            parts[n["name"]] = []
            for c in n["children"]:
                for name, P, N, C in prims(j["nodes"][c]):
                    parts[n["name"]].append((name, P + M[:3, 3], N, C))
    for i in j["scenes"][0]["nodes"]:
        n = j["nodes"][i]
        if "mesh" in n:
            parts["chassis"] += prims(n)
    return parts, pivots


# ----------------------------------------------------------------- sessions
def session(sid):
    j = json.load(open(os.path.join(RESULTS, sid + ".json")))
    z = np.load(os.path.join(RESULTS, sid + ".steps.npz"))
    return j, {k: z[k] for k in z.files}


def all_results():
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        out.append(json.load(open(f)))
    return out


if __name__ == "__main__":
    for k in MAPS:
        c = Course(k)
        print("%-11s length %.1f m  min width %.2f m  size %.1f x %.1f m  walls %s" % (
            k, c.length, c.min_width, c.size[0], c.size[1], [len(w) for w in c.walls]))
