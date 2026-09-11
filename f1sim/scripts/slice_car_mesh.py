"""Slice the viewer's car model (f1sim/assets/f1tenth_car.glb) into per-height outlines for the LiDAR.

The LiDAR then reflects off the same shape the viewer draws. Slices are closed outer loops at 1 cm
steps in the car's CoG frame (x forward, origin lr ahead of the rear axle); mesh detail below a
12 cm perimeter (bolts, cable ends) is dropped -- a 0.25 deg LiDAR cannot resolve it and every loop
kept costs a ray-segment test per beam per car.

    python scripts/slice_car_mesh.py        (from the f1sim directory)
"""
import os
import numpy as np
import trimesh
from shapely.geometry import LineString

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "f1sim", "assets")
GLB, OUT = os.path.join(ASSETS, "f1tenth_car.glb"), os.path.join(ASSETS, "f1tenth_car_slices.npz")
LF, LR = 0.15875, 0.17145           # sim vehicle geometry: CoG sits lr ahead of the rear axle
MIN_PERIMETER, TOL, STEP, S_MAX = 0.05, 0.006, 0.01, 256


def main() -> None:
    sc = trimesh.load(GLB)
    ax = {}
    for node in sc.graph.nodes_geometry:                          # rear axle from the wheel transforms
        T, g = sc.graph[node]
        if g.startswith("wheel_") and g.endswith("_rubber"):
            ax[g[6:8]] = trimesh.transform_points(sc.geometry[g].centroid[None], T)[0][0]
    xf, xr = np.mean([ax["fl"], ax["fr"]]), np.mean([ax["rl"], ax["rr"]])
    cog_x = xr + (xf - xr) * LR / (LF + LR)
    m = sc.to_geometry(); m.apply_translation([-cog_x, 0.0, 0.0])
    levels = np.arange(0.02, m.bounds[1][2] + 1e-6, STEP).astype(np.float32)
    segs = np.zeros((len(levels), S_MAX, 4), np.float32); valid = np.zeros((len(levels), S_MAX), bool)
    for i, z in enumerate(levels):
        sec = m.section(plane_origin=[0, 0, float(z)], plane_normal=[0, 0, 1])
        if sec is None:
            continue
        ss = []
        for path in sec.discrete:                                   # closed polylines, one per loop
            p = path[:, :2]
            if np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum() < MIN_PERIMETER:
                continue
            p = np.asarray(LineString(p).simplify(TOL).coords)
            if np.linalg.norm(p[0] - p[-1]) > 1e-6:
                p = np.vstack([p, p[:1]])
            ss += [(*a, *b) for a, b in zip(p[:-1], p[1:])]
        if not ss:
            continue
        ss = np.asarray(ss, np.float32)
        if len(ss) > S_MAX:
            raise SystemExit(f"z={z:.2f}: {len(ss)} segments > {S_MAX}; raise TOL or S_MAX")
        segs[i, :len(ss)] = ss; valid[i, :len(ss)] = True
        ext = ss[:, [0, 2]].max() - ss[:, [0, 2]].min(), ss[:, [1, 3]].max() - ss[:, [1, 3]].min()
        print(f"z={z:.2f}: {len(ss):3d} segments, outline {ext[0]:.3f} x {ext[1]:.3f} m")
    np.savez(OUT, z=levels, segs=segs, valid=valid, cog_x=float(cog_x), wheelbase=float(xf - xr))
    print(f"wrote {OUT}  (CoG {cog_x:.3f} m ahead of the rear axle, wheelbase {xf - xr:.3f} m)")


if __name__ == "__main__":
    main()
