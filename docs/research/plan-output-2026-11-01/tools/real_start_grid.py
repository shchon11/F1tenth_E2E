"""대회장 기록(실차)에서 차가 멈췄다가 출발하는 순간마다 주변에 무엇이 있었나 (real_start_grid.py, 11-12).
물음: 학습의 출발 뽑기(상대 차가 앞 / 뒤 / 나란히 1/3씩, 나란히는 차체 사이 0.30 m)가 대회장에서 실제로 있던 출발인가.
입력: 10-25에 뽑아 둔 물체 후보 파일(real_bag_unexplained.py OBJS=..., GOOD=0.95: 자기 위치가 지도에 맞는 scan에서 지도로 설명 안 되는
      점 묶음; LiDAR 기준 앞 + / 왼쪽 + [m])과 그 움직임 분류(_cls.npy: 1 = 움직임, 2 = 서 있음, 0 = 모름). 참 위치는 없음: 전부 차의 자기 LiDAR·자기 위치 추정.
멈춤 = 속도 0.2 m/s 아래로 1.5 s 이상, 출발 = 그 뒤 1.0 m/s를 넘는 순간.
표: 멈춤마다 기록 · 멈춘 시각과 길이 · 멈추기 직전 감속(가장 큰 것 [m/s^2]) · 멈춘 동안 뒤로 간 속도 · 멈춘 동안 옮겨진 거리 ·
    출발 앞 1 s 동안 3 m 안에 있던 물체(앞 / 왼쪽, 점 수, 분류) · 출발 뒤 3 s 동안 가장 가까웠던 움직이는 물체.
그림: 멈춤마다 한 칸, 출발 앞 1 s의 물체 후보를 차 기준으로(위 = 앞). 주황 = 움직임, 갈색 = 서 있음, 회색 = 모름. 가운데 네모 = 우리 차(대략).
인자: 출력png 물체후보npz [npz ...]"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):       # same list as rec_lib (the CJK collection registers as JP here)
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False

V_STILL, V_GO, MIN_STILL = 0.2, 1.0, 1.5
NEAR, BEFORE, AFTER = 3.0, 1.0, 3.0
C_MOVE, C_STAT, C_UNK, INK, MUTED = "#d1601f", "#8a5a2b", "#a9a7a0", "#0b0b0b", "#898781"
BODY = (-0.40, 0.18, 0.155)                                           # car outline around the LiDAR (rear, front, half-width), approximate


def near(ts, t, tol=0.5):
    i = int(np.argmin(np.abs(ts - t)))
    return i if abs(ts[i] - t) <= tol else None


out_png, files = sys.argv[1], sys.argv[2:]
ev = []
for fn in files:
    z = np.load(fn)
    obj, scan, od = z["obj"], z["scan"], z["odom"]
    cls_fn = fn.replace(".npz", "_cls.npy")
    cls = np.load(cls_fn) if os.path.exists(cls_fn) else np.zeros(len(obj), int)
    name = os.path.basename(fn).replace("objs_", "").replace(".npz", "")
    t0 = min(od[0, 0], scan[0, 0]); t = od[:, 0] - t0; v = od[:, 1]
    ts = scan[:, 0] - t0; to = obj[:, 1] - t0
    i = 0
    while i < len(v):
        if abs(v[i]) >= V_STILL:
            i += 1; continue
        j = i
        while j + 1 < len(v) and v[j + 1] < V_GO:
            j += 1
        if j + 1 < len(v) and t[j] - t[i] >= MIN_STILL:
            ta, tb = float(t[i]), float(t[j + 1])
            pre = (t >= ta - 1.0) & (t < ta)
            dec = float(-np.gradient(v[pre], t[pre]).min()) if pre.sum() > 3 else float("nan")
            vpre = float(v[pre].max()) if pre.any() else 0.0
            back = float(v[i:j + 1].min())
            ia, ib = near(ts, ta), near(ts, tb)
            moved = float(np.hypot(*(scan[ib, 1:3] - scan[ia, 1:3]))) if ia is not None and ib is not None else float("nan")
            mb = (to >= tb - BEFORE) & (to <= tb) & (np.hypot(obj[:, 2], obj[:, 3]) <= NEAR)
            n_scan = int(((ts >= tb - BEFORE) & (ts <= tb)).sum())
            ma = (to > tb) & (to <= tb + AFTER) & (cls == 1)
            mv = None
            if ma.any():
                k = int(np.argmin(np.hypot(obj[ma, 2], obj[ma, 3])))
                mv = (float(to[ma][k] - tb), float(obj[ma, 2][k]), float(obj[ma, 3][k]))
            ev.append(dict(name=name, ta=ta, tb=tb, dec=dec, vpre=vpre, back=back, moved=moved, n_scan=n_scan,
                           pts=np.c_[obj[mb, 2], obj[mb, 3], obj[mb, 4], cls[mb]], mv=mv, first=ta < 1.0))
        i = j + 1


def groups(p):
    """object candidates of the last second merged where they stay within 0.3 m: (ahead, left, beams, class, share of scans)"""
    out = []
    left = np.ones(len(p), bool)
    while left.any():
        k = int(np.nonzero(left)[0][0])
        m = left & (np.hypot(p[:, 0] - p[k, 0], p[:, 1] - p[k, 1]) < 0.3)
        c = np.bincount(p[m, 3].astype(int), minlength=3)
        out.append((float(np.median(p[m, 0])), float(np.median(p[m, 1])), float(np.median(p[m, 2])), int(np.argmax(c)), int(m.sum())))
        left &= ~m
    return sorted(out, key=lambda g: np.hypot(g[0], g[1]))


CL = ("모름", "움직임", "서 있음")
n_abreast = n_ahead = n_any = n_data = 0
print("기록 | 멈춘 시각 · 길이 | 직전 속도 · 가장 큰 감속 | 멈춘 동안 뒤로 · 옮겨진 거리 | 출발 앞 1 s, 3 m 안의 물체(앞 / 왼쪽 [m], 점 수, 분류) | 출발 뒤 3 s 가장 가까운 움직이는 물체")
for e in ev:
    g = [x for x in groups(e["pts"]) if x[4] >= max(3, 0.2 * e["n_scan"])] if len(e["pts"]) else []
    e["g"] = g
    if e["n_scan"] >= 5:
        n_data += 1
        ab = [x for x in g if abs(x[0] + 0.11) <= 0.6 and 0.2 <= abs(x[1]) <= 1.0]
        ah = [x for x in g if 0.3 < x[0] <= NEAR and abs(x[1]) < 0.5]
        e["abreast"], e["ahead"] = bool(ab), bool(ah)
        n_abreast += bool(ab); n_ahead += bool(ah); n_any += bool(g)
    what = ("; ".join(f"{x[0]:+.1f} / {x[1]:+.1f}, {x[2]:.0f}점, {CL[x[3]]}" for x in g[:4]) or "없음") if e["n_scan"] >= 5 else "자기 위치가 맞는 scan 없음"
    mv = f"{e['mv'][0]:.1f} s 뒤 앞 {e['mv'][1]:+.1f} / 왼쪽 {e['mv'][2]:+.1f}" if e["mv"] else "없음"
    print(f"{e['name'][-6:]} | {e['ta']:5.1f} s · {e['tb'] - e['ta']:4.1f} s{' (기록 시작)' if e['first'] else ''} | {e['vpre']:.1f} m/s · {e['dec']:.1f} m/s^2 | "
          f"{e['back']:+.2f} m/s · {e['moved']:.2f} m | {what} | {mv}")
print(f"\n멈췄다 출발 {len(ev)}번 가운데 출발 앞 1 s에 자기 위치가 맞는 scan이 있는 것 {n_data}번: 3 m 안에 물체 있음 {n_any}번, "
      f"옆에 나란히(앞뒤 0.6 m 안, 옆 0.2-1.0 m) {n_abreast}번, 바로 앞(0.3-3 m, 옆 0.5 m 안) {n_ahead}번")
fst = [e for e in ev if e["first"] and e["n_scan"] >= 5]
dist = ", ".join("%.1f m" % np.hypot(e["g"][0][0], e["g"][0][1]) for e in fst if e["g"]) or "-"
n_mv = sum(int((e["pts"][:, 3] == 1).sum() >= 3) for e in fst)
print(f"기록 시작의 출발 {sum(e['first'] for e in ev)}번 가운데 scan이 있는 {len(fst)}번: 옆에 나란히 {sum(e['abreast'] for e in fst)}번 · "
      f"바로 앞에 서 있는 물체 {sum(e['ahead'] for e in fst)}번 · 3 m 안에 서 있는 물체 {sum(bool(e['g']) for e in fst)}번(가장 가까운 것까지 {dist}) · "
      f"출발 앞 1 s에 3 m 안을 지나간 움직이는 물체 {n_mv}번")
mid = [e for e in ev if not e["first"]]
if mid:
    dec = np.array([e["dec"] for e in mid]); mvd = np.array([e["moved"] for e in mid]); bk = np.array([e["back"] for e in mid])
    print(f"달리다 멈춘 것 {len(mid)}번: 멈춘 길이 중앙값 {np.median([e['tb'] - e['ta'] for e in mid]):.1f} s · 직전 감속 8 m/s^2 넘음 {int((dec > 8).sum())}번 · "
          f"멈춘 동안 뒤로 0.2 m/s 넘게 감 {int((bk < -0.2).sum())}번 · 멈춘 동안 0.3 m 넘게 옮겨짐 {int((mvd > 0.3).sum())}번(위치 모름 {int(np.isnan(mvd).sum())}번)")

n = len(ev); cols = 8; rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.55 * rows + 0.6), squeeze=False)
for k, ax in enumerate(axes.ravel()):
    if k >= n:
        ax.axis("off"); continue
    e = ev[k]; p = e["pts"]
    ax.add_patch(plt.Rectangle((-BODY[2], BODY[0]), 2 * BODY[2], BODY[1] - BODY[0], facecolor="#2a6fbf", edgecolor="white", lw=0.8, zorder=5))
    for c, col in ((0, C_UNK), (2, C_STAT), (1, C_MOVE)):
        m = p[:, 3] == c
        if m.any():
            ax.scatter(-p[m, 1], p[m, 0], s=9, color=col, lw=0, zorder=4)   # left of the car drawn on the left
    ax.set_xlim(-NEAR, NEAR); ax.set_ylim(-NEAR, NEAR); ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2]); ax.tick_params(labelsize=6.5, length=2, colors=MUTED)
    ax.grid(color="#e6e4dd", lw=0.6); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#d0cec6")
    note = "scan 없음" if e["n_scan"] < 5 else ("나란히" if e.get("abreast") else "바로 앞" if e.get("ahead") else "")
    ax.set_title(f"{e['name'][-6:]} · {e['ta']:.0f} s{' 시작' if e['first'] else ''} · 멈춤 {e['tb'] - e['ta']:.0f} s\n{note}", fontsize=7.5, loc="left", color=INK)
h = [plt.Line2D([], [], marker="o", ls="", color=c, ms=5, label=l) for c, l in ((C_MOVE, "움직이는 물체"), (C_STAT, "서 있는 물체"), (C_UNK, "분류 안 됨"))]
h.append(plt.Rectangle((0, 0), 1, 1, facecolor="#2a6fbf", label="우리 차(대략, 위 = 앞)"))
fig.legend(handles=h, loc="upper right", ncol=4, frameon=False, fontsize=9)
fig.suptitle("대회장 기록: 멈췄다 출발할 때마다, 출발 앞 1 s 동안 3 m 안에 보인 물체(차 기준, 눈금 m)", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (2.55 * rows + 0.6)))
fig.savefig(out_png, dpi=110); plt.close(fig)
print(f"그림: {out_png}")
