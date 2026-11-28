"""Audit C: why each HARD wall / box contact happened -- did the plan drawn 0.5 s before already go into the obstacle, or did
the plan clear it and the car (MAP / physics) not follow? Clearance of the plan = min over the plan polyline (10 points,
densified) of the distance to the box polygon / to the wall (record's distance grid), minus the car half-width 0.155 m.
'plan clear' = that margin >= 0.05 m. Also: box seen before in the session (hit / passed earlier = repeat), speed 1 s
before. Args: rec.npz [rec.npz ...]"""
import sys
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec

HW = 0.155


def seg_dist(p, a, b):
    ab = b - a; t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(p - (a[None] + t[:, None] * ab[None]), axis=-1)


def poly_dist(pts, poly):
    d = np.full(len(pts), np.inf)
    for i in range(len(poly)):
        d = np.minimum(d, seg_dist(pts, poly[i], poly[(i + 1) % len(poly)]))
    return d


def dense(pl):
    out = [pl[0]]
    for a, b in zip(pl[:-1], pl[1:]):
        for f in np.linspace(0.1, 1, 10):
            out.append(a + f * (b - a))
    return np.array(out)


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        R = Rec(fn); st = R.st.astype(float); dt = R.dt
        res = {"box": [], "wall": []}
        for kind, col in (("wall", 3), ("box", 4)):
            for t, b in zip(*np.nonzero(R.d[:, :, col] > 0)):
                tp = max(0, t - 5)
                pl = dense(R.pl[tp, b].astype(float))
                pl = np.concatenate([st[tp:tp + 1, b, :2], pl], 0)
                v1 = abs(st[max(0, t - 10), b, 3])
                if kind == "box":
                    polys = R.boxes(t, b)
                    if not polys:
                        continue
                    car = st[t + 1, b, :2]
                    k = int(np.argmin([poly_dist(car[None], p)[0] for p in polys]))
                    m = float(poly_dist(pl, polys[k]).min()) - HW
                else:
                    m = float(R.wall_dist(int(R.tid[t, b]), pl).min()) - HW
                res[kind].append((m, v1))
        print(f"== {fn.split('/')[-1]}")
        for kind in ("wall", "box"):
            a = np.array(res[kind]) if res[kind] else np.zeros((0, 2))
            if len(a) == 0:
                print(f"  {kind}: 0"); continue
            clear = a[:, 0] >= 0.05
            print(f"  {kind}: hard {len(a)} | plan 0.5 s before cleared it by >= 5 cm: {clear.mean() * 100:.0f} % "
                  f"(margin median {np.median(a[:, 0]):.2f} m) | plan itself within 5 cm / through: {100 * (~clear).mean():.0f} % "
                  f"| speed 1 s before median {np.median(a[:, 1]):.1f} m/s")
