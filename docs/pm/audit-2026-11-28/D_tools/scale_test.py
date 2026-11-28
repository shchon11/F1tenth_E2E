"""Does bag_map's scale check (scan_mapper.scale_from_scans) catch a map drawn 20 % small? Real scans + pf poses of one
competition bag against the bag's /map at true scale and shrunk to 0.8 / 0.9 (poses shrunk with it)."""
import sys, json, numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim import scan_mapper as SM
Z = np.load("npz/20260827-115713_map08x23_v9.2_161s_fastest-longest.npz")
m = Z["map"]; res, ox, oy, _ = Z["map_info"]
occ = m >= 65; free = (m >= 0) & (m < 25)
meta = Z["scan_meta"]
sc = Z["scan"].copy(); sc[sc > 29.9] = np.inf
scans = SM.Scans(t=Z["scan_t"], ranges=sc, angle_min=float(meta[0]), angle_inc=float(meta[2]), time_inc=float(meta[3]),
                 range_max=30.0, laser=(0.297, 0.0, 0.0))
o = Z["odom"]; pf = Z["pf"]
odom = SM.Odom(t=o[:, 0], x=o[:, 2], y=o[:, 3], yaw=o[:, 4], v=o[:, 5], w=o[:, 7])
out = {}
for s in (1.0, 0.9, 0.8):
    gm = SM.GridMap(free=free, occ=occ, x0=float(ox) * s, y0=float(oy) * s, res=float(res) * s)
    poses = np.stack([np.interp(scans.t, pf[:, 0], pf[:, 2]) * s, np.interp(scans.t, pf[:, 0], pf[:, 3]) * s,
                      np.interp(scans.t, pf[:, 0], np.unwrap(pf[:, 4]))], 1)
    r = SM.scale_from_scans(gm, scans, poses, odom)
    out[s] = r; print(s, r, flush=True)
json.dump({str(k): v for k, v in out.items()}, open("scale_test.json", "w"))
