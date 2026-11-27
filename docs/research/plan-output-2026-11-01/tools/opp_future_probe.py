"""(11-28) How far ahead does the policy's state know the other car? A linear probe (ridge) from the state the policy
acts on (feat + mem of an exam4 record saved with F1SIM_EXAM_SAVE_MEM=1) to the other car's place in OUR frame at the
decision, h s later (h = 0.5, 1, 1.5, 2, 2.5, 3), trained on 3/4 of the sessions, judged on the other 1/4, against a
straight-line guess (its heading and speed now). Decisions with the other car 1-6 m ahead (|lateral| < 1.5 m), split:
approaching the left hairpin (lap fraction 0.40-0.58) / elsewhere. A probe is a lower bound on what the state holds (a
linear read-out), and what a critic or the actor could use.  usage: opp_future_probe.py EXAM4.npz MEM.npz"""
import sys

import numpy as np

z = np.load(sys.argv[1]); m = np.load(sys.argv[2])
X = np.concatenate([m["feat"], m["mem"]], -1).astype(np.float32)             # (T, B, D)
T, B, D = X.shape; dt = float(z["dt"]); L = float(z["lapL"][0])
st, op, on, ors = z["st"][:T + 40].astype(float), z["op"][:T + 40].astype(float), z["on"], z["ors"]
H = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]; K = [int(round(h / dt)) for h in H]
rows, Y, base, grp, ses = [], [], [], [], []
for b in range(B):
    for t in range(T - max(K) - 1):
        if not on[t, b] or ors[t:t + max(K) + 1, b].any():
            continue
        x, y, yw = st[t, b, :3]; c, s = np.cos(yw), np.sin(yw)
        def ego(px, py):
            return (px - x) * c + (py - y) * s, -(px - x) * s + (py - y) * c
        ox, oy = ego(op[t, b, 0], op[t, b, 1])
        if not (1.0 <= ox <= 6.0 and abs(oy) < 1.5):
            continue
        yt, bt = [], []
        for k, h in zip(K, H):
            yt += list(ego(op[t + k, b, 0], op[t + k, b, 1]))
            bt += list(ego(op[t, b, 0] + np.cos(op[t, b, 2]) * op[t, b, 3] * h, op[t, b, 1] + np.sin(op[t, b, 2]) * op[t, b, 3] * h))
        for k, h in zip(K, H):                                  # along the track: its progress in h s
            yt.append(np.mod(z["orem"][t, b] - z["orem"][t + k, b] + 0.5 * L, L) - 0.5 * L)
            bt.append(op[t, b, 3] * h)
        rows.append(X[t, b]); Y.append(yt); base.append(bt)
        q = np.mod(z["orem"][t, b], L) / L
        grp.append(0.40 <= q < 0.58); ses.append(b)
Xs, Y, base, grp, ses = np.array(rows), np.array(Y), np.array(base), np.array(grp), np.array(ses)
te = (ses % 4) == 3; tr = ~te
mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-3
A = (Xs - mu) / sd; A = np.concatenate([A, np.ones((len(A), 1))], 1)
lam = 10.0
W = np.linalg.solve(A[tr].T @ A[tr] + lam * np.eye(A.shape[1]), A[tr].T @ Y[tr])
P = A @ W
print(f"== {sys.argv[1].split('/')[-1]}: decisions with the other car 1-6 m ahead {len(Y)} (test {int(te.sum())}, hairpin {int((te & grp).sum())})")
for gname, gm in (("헤어핀 진입", grp), ("그 밖", ~grp)):
    msk = te & gm
    if not msk.any():
        continue
    out = []
    for j, h in enumerate(H):
        e = np.hypot(P[msk, 2 * j] - Y[msk, 2 * j], P[msk, 2 * j + 1] - Y[msk, 2 * j + 1])
        eb = np.hypot(base[msk, 2 * j] - Y[msk, 2 * j], base[msk, 2 * j + 1] - Y[msk, 2 * j + 1])
        out.append(f"{h:g} s: 상태 {np.median(e):.2f} / 직진 {np.median(eb):.2f} m")
    print(f"  {gname} (n {int(msk.sum())}) | " + " | ".join(out))
    o2 = []
    for j, h in enumerate(H):
        c_ = 2 * len(H) + j
        e = np.abs(P[msk, c_] - Y[msk, c_]); eb = np.abs(base[msk, c_] - Y[msk, c_])
        o2.append(f"{h:g} s: 상태 {np.median(e):.2f} / 지금 속도 그대로 {np.median(eb):.2f} m")
    print(f"     트랙 따라 간 거리 오차 | " + " | ".join(o2))
