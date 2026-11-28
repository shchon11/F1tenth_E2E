"""Audit C: failure types counted over a whole record (the scene panels show examples; this counts them all).
Hard WALL contact: 'after/beside a box' = a live box within 2.0 m of the car; 'corner' = heading turned > 0.5 rad in the
1 s before; else 'straight'. 'start' = in the first 1.0 s (spawn artefact).
Hard BOX contact: 'plan through' = the plan drawn 0.5 s before passed within 5 cm (car half-width) of the box; 'repeat' =
this session already had a hard contact with a box at the same place (within 0.5 m) earlier.
STALL (>= 5 s with < 1 m along the track): at its onset 'pinned on box' (box polygon within 0.45 m of the car centre),
'pinned on wall' (wall within 0.35 m), 'driving the wrong way' (|v| > 1 m/s during it), else 'stopped in the open'.
Args: rec.npz [...]"""
import sys
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/C")
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec
from hit_split import poly_dist, dense, HW

for fn in sys.argv[1:]:
    R = Rec(fn); st = R.st.astype(float); dt = R.dt; T, B = R.T, R.B
    W = {"start": 0, "box near": 0, "corner": 0, "straight": 0}
    X = {"plan through": 0, "plan clear": 0, "repeat place": 0}
    for t, b in zip(*np.nonzero(R.d[:, :, 3] > 0)):
        if t * dt < 1.0:
            W["start"] += 1; continue
        car = st[t, b, :2]
        polys = R.boxes(t, b)
        if polys and min(poly_dist(car[None], p)[0] for p in polys) < 2.0:
            W["box near"] += 1; continue
        dyaw = abs(np.angle(np.exp(1j * (st[t, b, 2] - st[max(0, t - 10), b, 2]))))
        W["corner" if dyaw > 0.5 else "straight"] += 1
    hit_places = {}
    for t, b in sorted(zip(*np.nonzero(R.d[:, :, 4] > 0)), key=lambda x: x[0]):
        polys = R.boxes(t, b)
        if not polys:
            continue
        car = st[min(t + 1, T), b, :2]
        k = int(np.argmin([poly_dist(car[None], p)[0] for p in polys]))
        c = polys[k].mean(0)
        prev = hit_places.setdefault(b, [])
        if any(np.linalg.norm(c - q) < 0.5 and t - tq > 30 for q, tq in prev):
            X["repeat place"] += 1
        prev.append((c, t))
        tp = max(0, t - 5)
        pl = np.concatenate([st[tp:tp + 1, b, :2], dense(R.pl[tp, b].astype(float))], 0)
        X["plan through" if poly_dist(pl, polys[k]).min() - HW < 0.05 else "plan clear"] += 1
    rem = R.z["rem"].astype(float); L = float(R.z["lapL"][int(R.z["tid"][0, 0])])
    for t in range(1, rem.shape[0]):
        bad = ~np.isfinite(rem[t]); rem[t, bad] = rem[t - 1, bad]
    P = np.concatenate([np.zeros((1, B)), np.cumsum(np.mod(rem[:-1] - rem[1:] + 0.5 * L, L) - 0.5 * L, 0)], 0)
    Sx = {"pinned on box": 0, "pinned on wall": 0, "wrong way / circling": 0, "stopped in open": 0}
    n = 50
    for b in range(B):
        t = 10
        while t < T - n:
            if P[t + n, b] - P[t, b] < 1.0:
                car = st[t + 5, b, :2]
                polys = R.boxes(t + 5, b)
                vmean = np.abs(st[t:t + n, b, 3]).mean()
                if polys and min(poly_dist(car[None], p)[0] for p in polys) < 0.45:
                    Sx["pinned on box"] += 1
                elif float(R.wall_dist(int(R.tid[t, b]), car[None])[0]) < 0.35:
                    Sx["pinned on wall"] += 1
                elif vmean > 1.0:
                    Sx["wrong way / circling"] += 1
                else:
                    Sx["stopped in open"] += 1
                t += 200
            else:
                t += 1
    print(f"== {fn.split('/')[-1]}: hard wall {sum(W.values())} {W} | hard box {X['plan through'] + X['plan clear']} {X} | stalls>=5s {sum(Sx.values())} {Sx}")
