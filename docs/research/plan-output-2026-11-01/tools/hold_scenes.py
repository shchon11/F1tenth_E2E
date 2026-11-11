"""plan 하나를 N x 0.1 s 동안 그대로 달린 기록(`drive_record.py`, F1SIM_HOLD=N)에서 벽에 세게 닿은 장면 (11-11: 붙잡으면 왜 무너지나를
숫자가 아니라 장면으로). 건마다 닿기 전 2 s: 차가 간 길(검정), 새로 쓴 plan마다 10점(색 = 쓴 순서, 굵은 부분 = 실제로 달린 구간),
닿은 자리의 차(빨강). 오른쪽 칸: 같은 구간의 속도와 plan이 낸 속도.
인자: npz 그림 [건수, 기본 8] [경우: empty(기본) / all]"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False

D = np.load(sys.argv[1]); out = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
case = sys.argv[4] if len(sys.argv) > 4 else "empty"
st, pl, rs, sid, on, newp, cnt, tid = D["st"], D["pl"], D["rs"], D["sid"], D["on"], D["newp"], D["cnt"], D["tid"]
T, B = rs.shape; dt = float(D["dt"]); L, Wd = [float(v) for v in D["car"]]
edt, off, Hh, Ww, res, org = D["edt"], D["t_off"], D["t_H"].astype(int), D["t_W"].astype(int), D["t_res"], D["t_origin"]
m = np.ones(B, bool) if case == "all" else ~((sid >= 0).any(2).any(0)) & ~on.any(0)
d = np.diff(np.concatenate([np.zeros((1, B, 6)), cnt], 0), axis=0).clip(min=0)
ev = [(t, b) for t, b in np.argwhere((d[:, :, 3] > 0) & m[None, :]) if t >= 20]
hold = int(np.diff(np.nonzero(newp[:, 0])[0]).max()) if newp[:, 0].sum() > 1 else 1
rng = np.random.default_rng(0)
pick = [ev[i] for i in sorted(rng.choice(len(ev), min(n, len(ev)), replace=False))]
fig, axs = plt.subplots(len(pick), 2, figsize=(13, 4.6 * len(pick)), gridspec_kw={"width_ratios": [2.2, 1]}, squeeze=False)
cm = plt.get_cmap("viridis")
for r, (t, b) in enumerate(pick):
    ax, av = axs[r]
    k = int(tid[t, b]); g = edt[off[k]:off[k] + Hh[k] * Ww[k]].reshape(Hh[k], Ww[k])
    ext = [org[k][0], org[k][0] + Ww[k] * res[k], org[k][1], org[k][1] + Hh[k] * res[k]]
    ax.imshow(g <= 0, origin="lower", extent=ext, cmap="Greys", alpha=0.55, interpolation="nearest")
    t0 = t - 20
    P = st[t0:t + 2, b, :2]
    ax.plot(P[:, 0], P[:, 1], "k-", lw=1.2, zorder=3)
    ts = [u for u in range(t0, t + 1) if newp[u, b]]
    for i, u in enumerate(ts):
        c = cm(i / max(len(ts) - 1, 1))
        q = np.concatenate([st[u, b, :2][None], pl[u, b]], 0)
        ax.plot(q[:, 0], q[:, 1], "-", color=c, lw=0.9, marker="o", ms=2.2, zorder=4)
        ax.plot(q[:hold + 1, 0], q[:hold + 1, 1], "-", color=c, lw=3.0, alpha=0.8, zorder=4)
        # the plan's own speed per step
        sp = np.linalg.norm(np.diff(q, axis=0), axis=1) / dt
        av.plot((u - t) * dt + (np.arange(10) + 0.5) * dt, sp, "-", color=c, lw=1.0)
    x, y, yaw = st[t + 1, b, 0], st[t + 1, b, 1], st[t + 1, b, 2]
    cc, ss = np.cos(yaw), np.sin(yaw)
    box = np.array([[L / 2, Wd / 2], [L / 2, -Wd / 2], [-L / 2, -Wd / 2], [-L / 2, Wd / 2], [L / 2, Wd / 2]])
    ax.plot(x + box[:, 0] * cc - box[:, 1] * ss, y + box[:, 0] * ss + box[:, 1] * cc, "r-", lw=1.6, zorder=5)
    c0 = P.mean(0); half = max(np.ptp(P[:, 0]), np.ptp(P[:, 1])) / 2 + 2.0
    ax.set_xlim(c0[0] - half, c0[0] + half); ax.set_ylim(c0[1] - half, c0[1] + half); ax.set_aspect("equal")
    ax.set_title(f"차 {b}, {t * dt:.1f} s에 벽에 세게 닿음 (트랙 변형 {k}) — 굵은 부분이 실제로 달린 {hold * dt:.1f} s", fontsize=10)
    av.plot((np.arange(t0, t + 2) - t) * dt, np.abs(st[t0:t + 2, b, 3]), "k-", lw=1.8, label="실제 속도")
    av.set_xlabel("닿기 전 시간 s"); av.set_ylabel("속도 m/s"); av.grid(alpha=0.3); av.legend(fontsize=8)
    av.set_title("색 선 = 각 plan이 낸 속도", fontsize=9)
fig.suptitle(f"{D['name']} · plan 하나를 {hold * dt:.1f} s 동안 그대로 달림 · 벽에 세게 닿은 {len(ev)}건 중 {len(pick)}건", fontsize=12, y=1.0)
fig.tight_layout(); fig.savefig(out, dpi=70); print("저장", out, "건수", len(ev))
