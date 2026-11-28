"""Audit C: failure scenes from exam_record.py records. Events = HARD contacts (wall / box / car) of the ego car, plus the
start of each long stall (>= 5 s with < 1 m along track). Picks events spread over sessions (one per session first),
hard car contacts and stalls first when asked. Each panel: map zoom +-3.5 m around the ego at the event; ego trail 3 s
before -> 0.5 s after (colour = time), ego body at -1.0 s / -0.5 s / event (blue), the plan drawn at -0.5 s (black dots),
boxes at the event (brown; boxes already removed are not drawn), opponent trail + body (red).
Args: out.png n_panels rec.npz[:label] [rec.npz[:label] ...]   env C_KINDS = comma list of wall,box,car,stall (default all)"""
import os, sys
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec, plt
from matplotlib.patches import Polygon

KINDS = os.environ.get("C_KINDS", "wall,box,car,stall").split(",")
out, n_pan = sys.argv[1], int(sys.argv[2])
events = []
recs = {}
for arg in sys.argv[3:]:
    fn, lab = (arg.split(":", 1) + [os.path.basename(arg)])[:2]
    R = Rec(fn); recs[lab] = R
    dt = R.dt; st = R.st.astype(float); T = R.T
    hard = R.d[:, :, 3:6]
    for k, kn in enumerate(("wall", "box", "car")):
        if kn not in KINDS:
            continue
        for t, b in zip(*np.nonzero(hard[:, :, k] > 0)):
            events.append((lab, kn, int(t), int(b)))
    if "stall" in KINDS:
        rem = R.z["rem"].astype(float); L = float(R.z["lapL"][int(R.z["tid"][0, 0])])
        for t in range(1, rem.shape[0]):
            bad = ~np.isfinite(rem[t]); rem[t, bad] = rem[t - 1, bad]
        dd = np.mod(rem[:-1] - rem[1:] + 0.5 * L, L) - 0.5 * L
        P = np.concatenate([np.zeros((1, R.B)), np.cumsum(dd, 0)], 0)
        n = 50
        for b in range(R.B):
            t = 10
            while t < T - n:
                if P[t + n, b] - P[t, b] < 1.0:
                    slow_ = np.nonzero(np.abs(st[t:t + n, b, 3]) < 0.5)[0]
                    events.append((lab, "stall", t + (int(slow_[0]) if len(slow_) else 0), b)); t += 200
                else:
                    t += 1
if os.environ.get("C_PACE"):                       # race4: only sessions against this copy pace
    _pz = {lab: R_.z["c_pace"] for lab, R_ in recs.items() if "c_pace" in R_.z.files}
    events = [e for e in events if e[0] in _pz and abs(float(_pz[e[0]][e[3]]) - float(os.environ["C_PACE"])) < 1e-3]
# one event per (record, session) first, interleaving kinds
rng = np.random.default_rng(1)
rng.shuffle(events)
order = {"car": 0, "stall": 1, "box": 2, "wall": 3}
seen, pick = set(), []
for kn in sorted(set(e[1] for e in events), key=lambda k: order[k]):
    pass
by_kind = {k: [e for e in events if e[1] == k] for k in order}
while len(pick) < n_pan and any(by_kind.values()):
    for k in sorted(by_kind, key=lambda k: order[k]):
        while by_kind[k]:
            e = by_kind[k].pop(0)
            if (e[0], e[3]) in seen:
                continue
            seen.add((e[0], e[3])); pick.append(e); break
        if len(pick) >= n_pan:
            break
print(f"events: {len(events)} ({', '.join(f'{k} {sum(e[1] == k for e in events)}' for k in order)}); drawn {len(pick)}")
cols = 4; rows = int(np.ceil(len(pick) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows), squeeze=False)
KN = {"wall": "벽 세게", "box": "상자 세게", "car": "차 세게", "stall": "멈춤 5 s"}
for ax, (lab, kn, t, b) in zip(axs.flat, pick):
    R = recs[lab]; st = R.st.astype(float); dt = R.dt
    k = int(R.tid[t, b]) if R.tid is not None else 0
    R.draw_map(ax, k)
    x0, y0 = st[t, b, 0], st[t, b, 1]
    t0, t1 = max(0, t - int(3 / dt)), min(R.T, t + int(0.5 / dt))
    ax.scatter(st[t0:t1, b, 0], st[t0:t1, b, 1], c=np.arange(t1 - t0), cmap="Blues", s=6, zorder=3)
    if R.on is not None and R.on[t0:t1, b].any():
        op = R.op.astype(float)
        ax.plot(op[t0:t1, b, 0], op[t0:t1, b, 1], color="#d62728", lw=1, zorder=3)
        ax.add_patch(Polygon(R.body(*op[t, b, :3]), closed=True, fc="#d62728", ec="k", alpha=0.6, zorder=4))
    for poly in R.boxes(t, b):
        ax.add_patch(Polygon(poly, closed=True, fc="#a0522d", ec="k", zorder=4))
    for dtk, a in ((int(1.0 / dt), 0.25), (int(0.5 / dt), 0.45), (0, 0.85)):
        tt = max(0, t - dtk)
        ax.add_patch(Polygon(R.body(*st[tt, b, :3]), closed=True, fc="#1f77b4", ec="k", alpha=a, zorder=5))
    tp = max(0, t - int(0.5 / dt))
    if R.pl is not None:
        pl = R.pl[tp, b].astype(float)
        ax.plot(pl[:, 0], pl[:, 1], "k.", ms=3, zorder=6)
    ax.set_xlim(x0 - 3.5, x0 + 3.5); ax.set_ylim(y0 - 3.5, y0 + 3.5)
    v1 = abs(st[max(0, t - int(1 / dt)), b, 3]); v0 = abs(st[t, b, 3])
    ax.set_title(f"{lab} · 세션 {b} · {t * dt:.1f} s\n{KN[kn]} · 속도 1 s 전 {v1:.1f} → {v0:.1f} m/s", fontsize=9)
for ax in axs.flat[len(pick):]:
    ax.axis("off")
fig.tight_layout(); fig.savefig(out, dpi=105); print("saved", out)
