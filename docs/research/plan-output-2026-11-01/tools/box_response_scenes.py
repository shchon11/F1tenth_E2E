"""box_response.py 기록에서 길을 막은 상자를 지나간 건의 원인 표와 장면.
표: 세게 닿는 비율을 (1) 상자 없는 plan이 상자 중심에서 얼마나 떨어져 지나나(0.8 s 전), (2) 상자가 스캔에 처음 보인 때, (3) 1 s 전 속도,
(4) 마지막 1 s 동안 실제 plan이 상자의 어느 쪽으로 지나는지가 바뀐 횟수 별로.
장면: 세게 닿은 건(과 비교용 깨끗이 지나간 건)을 지나가기 1.2 / 0.8 / 0.5 / 0.2 s 전 네 장으로 -- 실제 plan(파랑)과 상자 없는 plan(주황 점선).
인자: npz 출력폴더 [장면 수, 기본 10]"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False

D = np.load(sys.argv[1]); out = sys.argv[2]; NS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
os.makedirs(out, exist_ok=True)
st, sid, pose, cnt, rs, tid = D["st"], D["sid"], D["pose"], D["cnt"], D["rs"], D["tid"]
beams, cfhit, pl, plc = D["seen"], D["cfhit"], D["pl"].astype(np.float32), D["plc"].astype(np.float32)
dt = float(D["dt"]); T, B, P = sid.shape; W = 30
dprop = np.diff(np.concatenate([np.zeros((1, B, 6), np.float32), cnt], 0), axis=0).clip(min=0.0)
live = sid >= 0
xy = st[:, :, :2].astype(float)
dist = np.where(live, np.linalg.norm(pose[..., :2].astype(float) - xy[:, :, None], axis=3), 1e9)
near_j = dist.argmin(2)
_edt, _off, _H, _W, _res, _org = D["edt"], D["t_off"], D["t_H"].astype(int), D["t_W"].astype(int), D["t_res"], D["t_origin"]


def side_off(line, c):
    """signed distance from point c to the polyline `line` (n, 2): + = the point is left of the line's direction"""
    a, b_ = line[:-1], line[1:]
    ab = b_ - a; L2 = (ab ** 2).sum(1).clip(min=1e-9)
    u = (((c - a) * ab).sum(1) / L2).clip(0.0, 1.0)
    q = a + u[:, None] * ab
    dd = np.linalg.norm(c - q, axis=1); i = int(dd.argmin())
    cr = ab[i, 0] * (c[1] - a[i, 1]) - ab[i, 1] * (c[0] - a[i, 0])
    return float(dd[i]) * (1.0 if cr >= 0 else -1.0)


passes = []      # (b, j, t, hard, touch, offset of the no-box plan at 0.8 s, first-seen tau, speed 1 s before, side flips, isolated)
for b in range(B):
    if not live[:, b].any():
        continue
    start = np.concatenate([[0], np.nonzero(rs[:, b])[0] + 1])
    for j in range(P):
        if not live[:, b, j].any():
            continue
        d = dist[:, b, j]; t = 1
        while t < T - 1:
            lo, hi = max(t - 10, 0), min(t + 10, T - 1)
            if d[t] < 1.2 and d[t] <= d[lo: hi + 1].min():
                ep0 = int(start[start <= t].max())
                if t - ep0 >= W and cfhit[t - 12: t, b, j].any():
                    w0, w1 = max(t - 8, 0), min(t + 5, T - 1)
                    mine = near_j[w0: w1 + 1, b] == j
                    touch = bool((dprop[w0: w1 + 1, b, 1] * mine).sum() > 0.5); hard = bool((dprop[w0: w1 + 1, b, 4] * mine).sum() > 0.5)
                    c = pose[t, b, j, :2].astype(float)
                    line = lambda arr, k: np.concatenate([xy[k, b][None], arr[k, b].astype(float)], 0)
                    off = side_off(line(plc, t - 8), c)
                    sn = beams[t - W: t + 1, b, j] >= 2
                    u = W
                    while u > 0 and (sn[u - 1] or (u > 1 and sn[u - 2])):
                        u -= 1
                    sd = np.array([side_off(line(pl, k), c) for k in range(t - 10, t - 1)])
                    sd = np.sign(sd[np.abs(sd) > 0.05])
                    flips = int((np.diff(sd) != 0).sum()) if len(sd) > 1 else 0
                    passes.append((b, j, t, hard, touch, off, (u - W) * dt, abs(float(st[t - 10, b, 3])), flips))
                t = hi + 1
            else:
                t += 1
Pz = np.array([p[3:] for p in passes], float)
hard = Pz[:, 0] > 0.5
print(f"{D['name']}: 길을 막은 상자를 지나간 건 {len(Pz)}, 세게 닿음 {int(hard.sum())} ({hard.mean() * 100:.1f} %)")


def table(title, x, edges, fmt):
    print(f"\n[{title}]")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum():
            print(f"  {fmt(lo, hi)}: {int(m.sum()):5d}건 ({m.mean() * 100:4.1f} %) | 세게 닿음 {hard[m].mean() * 100:4.1f} % ({int(hard[m].sum())}건)")


table("상자 없는 plan이 상자 중심에서 떨어진 거리 (0.8 s 전)", np.abs(Pz[:, 2]), [0, 0.1, 0.2, 0.3, 0.4, 0.6, 9], lambda a, b_: f"{a * 100:3.0f}~{min(b_, 1) * 100:3.0f} cm")
table("상자가 스캔에 처음 보인 때 (지나가기 전)", -Pz[:, 3], [0, 0.6, 0.9, 1.2, 1.6, 2.2, 9], lambda a, b_: f"{a:.1f}~{min(b_, 3):.1f} s")
table("1 s 전 속도", Pz[:, 4], [0, 3.5, 4.3, 5.0, 5.6, 99], lambda a, b_: f"{a:.1f}~{min(b_, 9):.1f} m/s")
table("마지막 1 s 동안 실제 plan이 지나는 쪽이 바뀐 횟수", Pz[:, 5], [0, 1, 2, 3, 99], lambda a, b_: f"{int(a)}번" + ("" if b_ - a == 1 else " 넘게"))

# scenes
occ_c = {}


def occ(k):
    if k not in occ_c:
        occ_c[k] = _edt[_off[k]: _off[k] + _H[k] * _W[k]].reshape(_H[k], _W[k]) <= 0.0
    return occ_c[k]


def draw(ax, b, j, t, tau):
    k = t - int(round(tau / dt)); tk = int(tid[t, b])
    c = pose[t, b, j, :2].astype(float)
    o = occ(tk); ext = [_org[tk][0], _org[tk][0] + _W[tk] * _res[tk], _org[tk][1], _org[tk][1] + _H[tk] * _res[tk]]
    ax.imshow(np.where(o, 0.45, 1.0), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for q in range(P):
        if sid[k, b, q] < 0:
            continue
        fp = D[f"fp_{int(sid[k, b, q])}"]; x0, y0, th = pose[k, b, q]
        R_ = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pg = fp @ R_.T + [x0, y0]
        ax.fill(pg[:, 0], pg[:, 1], color="#8b5a2b" if q != j else "#c2410c", zorder=3)
    ax.plot(xy[max(k - 15, 0): k + 1, b, 0], xy[max(k - 15, 0): k + 1, b, 1], color="#6b7280", lw=1.2, zorder=4)
    for arr, col, ls, lab in ((plc, "#d97706", "--", "상자 없는 plan"), (pl, "#1d4ed8", "-", "실제 plan")):
        ln = np.concatenate([xy[k, b][None], arr[k, b].astype(float)], 0)
        ax.plot(ln[:, 0], ln[:, 1], ls, color=col, lw=1.8, marker="o", ms=2.5, zorder=5, label=lab)
    x0, y0, th = st[k, b, 0], st[k, b, 1], st[k, b, 2]
    ax.plot([x0], [y0], "s", color="#111827", ms=6, zorder=6)
    ax.arrow(x0, y0, 0.35 * np.cos(th), 0.35 * np.sin(th), color="#111827", head_width=0.12, zorder=6)
    ax.set_xlim(c[0] - 4.0, c[0] + 4.0); ax.set_ylim(c[1] - 4.0, c[1] + 4.0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{tau:.1f} s 전 · {abs(float(st[k, b, 3])):.1f} m/s · 상자에 닿은 빔 {int(beams[k, b, j])}", fontsize=9, loc="left")


TAUS = (1.2, 0.8, 0.5, 0.2)
for name, sel in (("hard", [p for p in passes if p[3]]), ("clean", [p for p in passes if not p[4] and abs(p[5]) < 0.15])):
    sel = sel[:: max(len(sel) // NS, 1)][:NS]
    fig, axs = plt.subplots(len(sel), 4, figsize=(15, 3.75 * len(sel)), dpi=80, squeeze=False)
    for r, p in enumerate(sel):
        for q, tau in enumerate(TAUS):
            draw(axs[r, q], p[0], p[1], p[2], tau)
        axs[r, 0].set_ylabel(f"차 {p[0]} · {p[2] * dt:.1f} s\n상자 없는 plan과 상자 중심 {abs(p[5]) * 100:.0f} cm", fontsize=9)
    axs[0, 0].legend(fontsize=8, loc="upper left", frameon=True)
    fig.suptitle(f"{D['name']} · 길을 막은 상자 · {'세게 닿은 건' if name == 'hard' else '깨끗이 지나간 건(상자가 길 한가운데)'} · 붉은 상자 = 그 상자, 검은 네모 = 차", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(f"{out}/scenes_{name}.png", facecolor="white"); plt.close(fig)
    print("저장", f"{out}/scenes_{name}.png")
