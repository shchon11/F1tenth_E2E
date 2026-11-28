"""세게 닿은 상자 사건 장면 그림. 칸마다:
  회색 선 = 같은 세션에서 그 상자를 앞서 지난 통과들의 실제 경로(바퀴마다), 파랑 선 = 이번 접근의 실제 경로,
  하늘색 삼각형 = 상자가 LiDAR에 처음 보인 자리(이어서 보인 구간의 시작), 하늘색 점선 = 그때 스캐너에서 상자까지,
  초록 점 = 처음 보인 결정의 계획, 주황 점 = 계획이 상자 자리에 처음 닿은 결정의 계획, 빨강 점 = 닿기 0.3 s 전 계획,
  파랑 사각형 = 닿은 순간 차체, 갈색 = 상자(다른 상자도).
인자: events.json out.png "lab:b:t,lab:b:t,..." [열 수]"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_lib import Rec, body_poly, dense_traj
from rec_lib import plt
from matplotlib.patches import Polygon

E = json.load(open(sys.argv[1]))
out = sys.argv[2]
picks = [p.split(":") for p in sys.argv[3].split(",")]
cols = int(sys.argv[4]) if len(sys.argv) > 4 else 4
ZOOM = float(os.environ.get('ZOOM', '0'))
recs = {}
rows_cls = {}
fn_rows = sys.argv[1].replace("events_ideal.json", "cls_rows.json")
if os.path.exists(fn_rows):
    for x in json.load(open(fn_rows)):
        rows_cls[(x["lab"], int(x["b"]), int(x["t"]))] = x
rows = int(np.ceil(len(picks) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.9 * rows), squeeze=False)
for ax, (lab, b, t) in zip(axs.flat, picks):
    b, t = int(b), int(t)
    r = E[lab]
    if lab not in recs:
        recs[lab] = Rec(r["file"])
    R = recs[lab]; st = R.st.astype(float)
    h = next(x for x in r["hard"] if x["b"] == b and x["t"] == t)
    R.draw_map(ax, 0)
    polys = R.boxes(t, b)
    for i, P in enumerate(polys):
        ax.add_patch(Polygon(P, closed=True, fc="#a0522d" if i == h["j"] else "#d2a679", ec="k", zorder=4))
    encs = sorted([e for e in r["enc"] if e["b"] == b and e["j"] == h["j"] and e["t1"] < t - 5], key=lambda e: e["t0"])
    for e in encs:
        tr, _ = dense_traj(st, max(0, e["t0"] - 15), min(R.T, e["t1"] + 5), b)
        ax.plot(tr[:, 0], tr[:, 1], color="#888888", lw=0.9, zorder=3)
    tv = h["t_vis"]
    tr, _ = dense_traj(st, max(0, tv - 5), min(R.T, t + 4), b)
    ax.plot(tr[:, 0], tr[:, 1], color="#1f77b4", lw=1.6, zorder=5)
    ax.plot(st[tv, b, 0], st[tv, b, 1], marker="^", color="#17becf", ms=9, mec="k", zorder=7)
    c = np.array(h["box_c"])
    mx = float(R.z["lidar_mount_x"])
    sx, sy = st[tv, b, 0] + mx * np.cos(st[tv, b, 2]), st[tv, b, 1] + mx * np.sin(st[tv, b, 2])
    ax.plot([sx, c[0]], [sy, c[1]], ls=":", color="#17becf", lw=1.2, zorder=6)
    def plan_dots(tt, col):
        if tt is None or tt < 0:
            return
        P = np.concatenate([st[tt, b, None, :2], R.pl[tt, b].astype(float)], 0)
        ax.plot(P[:, 0], P[:, 1], ".", color=col, ms=5, zorder=8)
    plan_dots(tv, "#2ca02c")
    plan_dots(h["reach_from"], "#ff7f0e")
    plan_dots(max(tv, t - 3), "#d62728")
    ax.add_patch(Polygon(body_poly(*st[t, b, :3], R.hl, R.hw), closed=True, fc="#1f77b4", ec="k", alpha=0.75, zorder=9))
    xs = np.concatenate([tr[:, 0], [c[0]]]); ys = np.concatenate([tr[:, 1], [c[1]]])
    cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + 1.0
    if ZOOM > 0:                                   # zoom on the box (the first-visible place may fall outside)
        cx, cy, half = c[0], c[1], ZOOM
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    x = rows_cls.get((lab, b, t), {})
    prev = h["prev"][-1] if h["prev"] else None
    ptxt = f"지난 통과 여유 {100 * prev['clear']:.0f} cm" + ("" if prev["side"] == h["enc_side"] else " (반대쪽)") if prev else "첫 만남"
    ax.set_title(f"{lab} 세션 {b} · {t * R.dt:.1f} s · {h['enc_no'] + 1}번째 만남\n"
                 f"{x.get('vis', '')}\n보임 {h['vis_s']:.1f} s 전 {h['d_vis']:.1f} m · {h['v_vis']:.1f}→{h['v_hit']:.1f} m/s · {ptxt}",
                 fontsize=8.5)
for ax in axs.flat[len(picks):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(out, dpi=100)
print("saved", out)
