"""세게 닿은 사건 장면: 차의 실제 자취(2 s 전 -> 0.3 s 뒤, 0.025 s마다), 1.0 / 0.7 / 0.4 / 0.1 s 전에 그린 plan(연한 -> 진한),
사건 순간 차체, 상자. 같은 세션의 R0(이상 실행) 자취를 회색으로 겹침(같은 자리를 이상 실행은 어떻게 지났나).
인자: out.png 기록이름 분석json [패널 수 8]"""
import sys, json, os
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec, plt
from matplotlib.patches import Polygon

O = "/data/pm-execgap-2026-11-28/rec"
out, name, anj = sys.argv[1:4]
npan = int(sys.argv[4]) if len(sys.argv) > 4 else 8
A = json.load(open(anj))[name]
R = Rec(f"{O}/{name}.npz"); xg = np.load(f"{O}/{name}_xg.npz")
R0 = Rec(f"{O}/R0_ideal.npz") if name != "R0_ideal" else None
st = xg["d_st"]
c, s = np.cos(st[..., 2])[..., None], np.sin(st[..., 2])[..., None]
X = st[..., 0:1] + xg["d_x"] * c - xg["d_y"] * s; Y = st[..., 1:2] + xg["d_x"] * s + xg["d_y"] * c
cst = xg["c_st"]
ev = A["events"]
# spread: alternate wall / box, one per session first
pick, seen = [], set()
rng = np.random.default_rng(int(os.environ.get("XG_SEED", "3")))
for kind in ("wall", "box"):
    E = [e for e in ev if e["kind"] == kind]
    for i in rng.permutation(len(E)):
        e = E[i]
        if e["b"] not in seen and len([p for p in pick if p["kind"] == kind]) < npan // 2:
            pick.append(e); seen.add(e["b"])
cols = 4; rows = int(np.ceil(len(pick) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows))
axs = np.atleast_1d(axs).ravel()
SH = ["#c6dcf6", "#8fbcef", "#4f94e4", "#1f5fb0"]
for ax, e in zip(axs, pick):
    t, b = e["t"], e["b"]
    k = int(R.tid[t, b])
    R.draw_map(ax, k)
    cx, cy = st[t, b, 0], st[t, b, 1]
    if R0 is not None:
        tr0 = R0.st[max(0, t - 30):t + 10, b, :2].astype(float)
        ax.plot(tr0[:, 0], tr0[:, 1], "-", color="#9a9993", lw=1.5, zorder=2, label="R0 같은 세션 자취")
    i0, i1 = max(0, 4 * (t - 20)), min(cst.shape[0], 4 * (t + 3))
    tr = cst[i0:i1, b]
    ax.scatter(tr[:, 0], tr[:, 1], c=np.linspace(0.2, 1, len(tr)), cmap="Oranges", s=6, zorder=4)
    for sh, kk in zip(SH, (t - 10, t - 7, t - 4, t - 1)):
        ax.plot(X[kk, b], Y[kk, b], "-", color=sh, lw=2, zorder=5)
        ax.plot(X[kk, b, 0], Y[kk, b, 0], "o", color=sh, ms=4, zorder=5)
    for p in R.boxes(t, b):
        ax.add_patch(Polygon(p, closed=True, fc="#a0703c", ec="#5a3b17", zorder=3))
    body = R.body(st[min(t + 1, st.shape[0] - 1), b, 0], st[min(t + 1, st.shape[0] - 1), b, 1], st[min(t + 1, st.shape[0] - 1), b, 2])
    ax.add_patch(Polygon(body, closed=True, fill=False, ec="#d03b2a", lw=2, zorder=6))
    ax.set_xlim(cx - 3, cx + 3); ax.set_ylim(cy - 3, cy + 3)
    ax.set_title(f"{e['kind']} s{b} {t * 0.1:.1f}s v{e['v']:.1f} | {e.get('class', '')[:12]}\n"
                 f"0.1 s 추종 최대 {100 * e['e1_lat_max1s']:.0f} cm · 의도 변화 최대 {100 * e['jump_max']:.0f} cm · 0.5 s 전 plan 여유 {100 * e['clear_t5']:.0f} cm",
                 fontsize=8)
for ax in axs[len(pick):]:
    ax.axis("off")
fig.suptitle(f"{name}: 주황 점 = 차 자취(진할수록 최근), 파랑 선 = 1.0/0.7/0.4/0.1 s 전 plan(진할수록 최근), 빨강 = 사건 직후 차체, 회색 = R0 같은 세션 자취",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=110); plt.close(fig)
print("saved", out, len(pick))
