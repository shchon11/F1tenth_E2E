"""포팅본이 혼자 달린 기록(aurorus_solo.py)의 장면 (11-12; 사용자 규칙: 숫자보다 장면 먼저).
  <출력>_runs.png   속도 배율마다 차 몇 대의 달린 길 전체(색 = 속도), 상자(갈색), 접촉 시작(빨간 x = 세게, 주황 x = 가볍게), 걸려서 다시 출발(검은 +)
  <출력>_events.png 일(세게 벽 / 세게 상자 / 걸림 / 가볍게 닿음 순)마다 확대(±2.5 m): 3초 전부터 1초 뒤까지의 자취(옅음 -> 진함),
                    그 순간의 차체(빨강), 0.5초 전에 장애물로 본 스캔 점(주황)·고른 경로(파랑)·겨눈 점(파란 별), 상자(갈색)
                    제목 = 무슨 일, 속도 배율, 1초 전 속도 -> 그 순간 속도, 0.5초 전에 안 닿는 경로가 있었나
인자: 기록 npz 출력 접두 [일 최대 수, 기본 12]"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rec_lib as RL

fn, out = sys.argv[1], sys.argv[2]
N_EV = int(sys.argv[3]) if len(sys.argv) > 3 else 12
r = RL.Rec(fn); z = r.z
T, B, dt = r.T, r.B, r.dt
st = r.st.astype(float); v = np.abs(st[:T, :, 3])
ps = z["pscale"][:T]; scales = [float(x) for x in z["scales"]]
COLS = [str(c) for c in z["cols"]]
d = r.d                                                               # (T, B, 6) new contacts in a decision
stuck = z["stuck"][:T]
obs = z["obs"].astype(float); path = z["path"].astype(float); tgt = z["tgt"].astype(float); pfree = z["pfree"]

# ---- whole runs
per = 3
fig, axes = plt.subplots(len(scales), per, figsize=(5.2 * per, 4.9 * len(scales)), squeeze=False, constrained_layout=True)
for i, sc in enumerate(scales):
    cars = np.nonzero(np.abs(ps[0] - sc) < 1e-3)[0]
    # the first car, the one with the most contacts, the one that stood longest
    pick = [cars[0]] if len(cars) else []
    if len(cars) > 1:
        c2 = cars[np.argsort(-d[:, cars].sum((0, 2)))]
        pick += [c for c in c2 if c not in pick][:1]
        c3 = cars[np.argsort(-(v[:, cars] < 0.3).sum(0))]
        pick += [c for c in c3 if c not in pick][:1]
    for j in range(per):
        ax = axes[i, j]
        if j >= len(pick):
            ax.axis("off"); continue
        b = int(pick[j]); k = int(r.tid[0, b])
        r.draw_map(ax, k)
        for q in r.boxes(0, b):
            ax.add_patch(Polygon(q, closed=True, fc=RL.C_BOX, ec="k", lw=0.4, zorder=3))
        jump = np.concatenate([[False], np.linalg.norm(np.diff(st[:T, b, :2], axis=0), axis=1) > 2.0])
        xy = st[:T, b, :2].copy(); xy[jump] = np.nan
        scat = ax.scatter(xy[:, 0], xy[:, 1], c=v[:, b], s=3, cmap="viridis", vmin=0, vmax=8, zorder=4)
        for c_i, h_i in ((0, 3), (1, 4)):                               # light touches below, hard ones on top
            tt = np.nonzero((d[:, b, c_i] > 0) & (d[:, b, h_i] == 0))[0]
            ax.plot(st[tt, b, 0], st[tt, b, 1], "x", color="orange", ms=8, mew=1.6, zorder=6, ls="none")
            tt = np.nonzero(d[:, b, h_i] > 0)[0]
            ax.plot(st[tt, b, 0], st[tt, b, 1], "x", color="red", ms=9, mew=2, zorder=7, ls="none")
        tt = np.nonzero(stuck[:, b])[0]
        ax.plot(st[tt, b, 0], st[tt, b, 1], "+", color="k", ms=12, mew=2, zorder=8, ls="none")
        hw = int(d[:, b, 3].sum()); hb = int(d[:, b, 4].sum()); lw_ = int(d[:, b, 0].sum()) - hw; lb = int(d[:, b, 1].sum()) - hb
        ax.set_title(f"선 속도의 {sc:g}배 · {RL.trk_name(k)} · 평균 {v[:, b].mean():.1f} m/s\n세게: 벽 {hw} 상자 {hb} · 가볍게: 벽 {lw_} 상자 {lb} · "
                     f"걸림 {int(stuck[:, b].sum())} · 선 시간 {100 * (v[:, b] < 0.3).mean():.0f} %", fontsize=9)
        last = scat
    if len(pick):
        fig.colorbar(last, ax=list(axes[i]), fraction=0.02, pad=0.01, label="속도 m/s")
HOW = (f" · 주행 {dict(exact='정확 실행', physics='보통 물리')[str(z['drive'])]} · 바닥 점 "
       f"{dict(off='그대로', gate='자기 센서로 거름', oracle='정답으로 거름')[str(z['floor'])]}"
       + ((" · 경로 추종 " + ("pure pursuit(버린 옵션)" if '"race"' in str(z["aur_cfg"]) else "MAP 제어기" if '"map"' in str(z["aur_cfg"])
                              else "원본")) if "aur_cfg" in z.files else "")
       + ("" if "calibrated" in z.files or str(z["drive"]) != "physics" else " · 차 보정 없는 명령")) if "drive" in z.files else ""
fig.suptitle(f"{r.name}{HOW}: 혼자 달린 길 (차 {B}대 x {T * dt:.0f} s 가운데; 칸 = 첫 차 / 접촉 가장 많은 차 / 가장 오래 선 차)", fontsize=12)
fig.savefig(out + "_runs.png", dpi=95); plt.close(fig)

# ---- events
EV = []                                                               # (priority, t, b, label)
for pri, (c_i, lab) in enumerate(((3, "벽에 세게"), (4, "상자에 세게"))):
    for t, b in zip(*np.nonzero(d[..., c_i] > 0)):
        EV.append((pri, int(t), int(b), lab))
for t, b in zip(*np.nonzero(stuck)):
    EV.append((2, int(t), int(b), "걸려서 다시 출발"))
for pri, (c_i, h_i, lab) in enumerate(((0, 3, "벽에 가볍게"), (1, 4, "상자에 가볍게")), start=3):
    for t, b in zip(*np.nonzero((d[..., c_i] > 0) & (d[..., h_i] == 0))):
        EV.append((pri, int(t), int(b), lab))
n_kind = {}
for e in EV:
    n_kind[e[3]] = n_kind.get(e[3], 0) + 1
# a spread: per kind in turn, different cars first
rng = np.random.default_rng(0)
by = {}
for e in EV:
    by.setdefault(e[0], []).append(e)
for k_ in by:
    rng.shuffle(by[k_])
order, used = [], set()
share = {0: 4, 1: 4, 2: 3, 3: 1, 4: 1}
for k_ in sorted(by):
    took = 0
    for e in by[k_]:
        if took >= share.get(k_, 1) + max(0, N_EV - sum(min(share[q], len(by.get(q, []))) for q in share)) or (e[2], e[3]) in used:
            continue
        order.append(e); used.add((e[2], e[3])); took += 1
order = order[:N_EV]
print(f"{r.name} ({fn.split('/')[-1]}): " + " · ".join(f"{k_} {n}건" for k_, n in n_kind.items()) + f" | 그린 것 {len(order)}건")
if order:
    nc = min(4, len(order)); nr = int(np.ceil(len(order) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 5.1 * nr), squeeze=False, constrained_layout=True)
    for ax in axes.flat:
        ax.axis("off")
    W = 2.5
    for n, (pri, t, b, lab) in enumerate(order):
        ax = axes.flat[n]; ax.axis("on")
        k = int(r.tid[t, b])
        r.draw_map(ax, k)
        for q in r.boxes(t, b):
            ax.add_patch(Polygon(q, closed=True, fc=RL.C_BOX, ec="k", lw=0.5, zorder=3))
        a = max(int(r.k0[t, b]), t - 30); e = min(T, t + 11)
        if lab.startswith("걸려서"):
            e = t + 1
        seg = st[a:e + (0 if e >= T else 0), b, :2]
        keep = np.concatenate([[True], np.linalg.norm(np.diff(seg, axis=0), axis=1) < 2.0]).cumprod().astype(bool)
        seg = seg[keep]
        ax.scatter(seg[:, 0], seg[:, 1], c=np.arange(len(seg)), cmap="Greys", vmin=-len(seg) * 0.4, vmax=len(seg), s=10, zorder=5)
        t0 = max(a, t - 5)                                            # what it saw and chose 0.5 s before
        o = obs[t0, b]; o = o[np.isfinite(o[:, 0])]
        ax.plot(o[:, 0], o[:, 1], ".", color="darkorange", ms=5, zorder=6, ls="none")
        p = path[t0, b]; p = p[np.isfinite(p[:, 0])]
        if len(p):
            ax.plot(p[:, 0], p[:, 1], "-", color="tab:blue", lw=1.6, zorder=6)
        if np.isfinite(tgt[t0, b, 0]):
            ax.plot(tgt[t0, b, 0], tgt[t0, b, 1], "*", color="tab:blue", ms=11, mec="k", mew=0.5, zorder=7)
        ax.add_patch(Polygon(r.body(*st[t0, b, :3]), closed=True, fc="none", ec="tab:blue", lw=1.0, zorder=7))
        ax.add_patch(Polygon(r.body(*st[t, b, :3]), closed=True, fc="none", ec="red", lw=1.6, zorder=8))
        cx, cy = st[t, b, :2]
        ax.set_xlim(cx - W, cx + W); ax.set_ylim(cy - W, cy + W)
        v1 = v[max(a, t - 10), b]
        ax.set_title(f"{lab} · 선 속도의 {ps[t, b]:g}배 · {RL.trk_name(k)}\n1초 전 {v1:.1f} -> {v[t, b]:.1f} m/s · 0.5초 전: 장애물 점 {len(o)}개, "
                     + ("회피 안 함" if not len(p) else ("안 닿는 경로 있음" if pfree[t0, b] else "안 닿는 경로 없음")), fontsize=8.5)
    fig.suptitle(f"{r.name}{HOW}: 접촉·걸림 장면 (파랑 = 0.5초 전의 차·고른 경로·겨눈 점, 주황 점 = 그때 장애물로 본 스캔 점, 빨강 = 그 순간의 차)",
                 fontsize=11)
    fig.savefig(out + "_events.png", dpi=95); plt.close(fig)
