"""시험 기록에서, 한 자리(기본 = 아래쪽 길로 튀어나온 짧은 벽의 끝 둘레, x -2..0, y -2..0)를 지날 때 벽까지 남긴 거리.
실제 달린 길(차 옆면에서 벽까지)과, 가장 가까워지기 0.5 s / 0.3 s 전에 낸 plan(0.1~0.5 s 점)이 남긴 거리를 통과마다 잼.
혼자 달리는 세션에서 상자를 치운 뒤(55 s~)만 = 빈 트랙 주행. 같은 policy의 checkpoint들, 실행 방식들을 나란히 놓고 보는 용도.
인자: 기록npz ...   (WALL_END_REG="x0,x1,y0,y1" 로 자리 바꿈; WALL_END_FIG=png 이면 기록마다 그 자리의 달린 길을 겹쳐 그린 그림도)"""
import os
import sys
import numpy as np
from rec_lib import Rec, plt

X0, X1, Y0, Y1 = (float(x) for x in os.environ.get("WALL_END_REG", "-2,0,-2,0").split(","))
FIG = os.environ.get("WALL_END_FIG", "")
MODE = dict(map="MAP", track="시간 추종기", ideal="plan대로 정확히(학습 조건)")
PAD = 2.5


def q(x):
    x = np.asarray(x, float)
    return f"중앙 {np.median(x) * 100:5.1f} cm · 아래 10 % {np.percentile(x, 10) * 100:5.1f} cm · 가장 작음 {x.min() * 100:5.1f} cm" if len(x) else "없음"


keep = []
for f in sys.argv[1:]:
    R = Rec(f); z = R.z; T, B, dt = R.T, R.B, R.dt
    st = R.st.astype(float); xy = st[:, :, :2]; V = np.abs(st[:, :, 3]); pl = R.pl.astype(float)
    opp = R.on[:T].any(0); k0 = int(round(55.0 / dt))
    hard = R.d[:, :, 3] > 0
    act, spd, p5, p3, hit, paths, marks = [], [], [], [], [], [], []
    for b in np.nonzero(~opp)[0]:
        tk = int(R.tid[0, b])
        ins = (xy[k0:T, b, 0] >= X0) & (xy[k0:T, b, 0] < X1) & (xy[k0:T, b, 1] >= Y0) & (xy[k0:T, b, 1] < Y1)
        idx = np.nonzero(ins)[0] + k0
        if len(idx) == 0:
            continue
        for seg in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
            if len(seg) < 2:
                continue
            d = R.wall_dist(tk, xy[seg, b]); j = int(d.argmin()); t = int(seg[j])
            h = hard[seg[0]:seg[-1] + 1, b]
            act.append(d[j] - R.hw); spd.append(V[t, b]); hit.append(bool(h.any()))
            for k, lst in ((5, p5), (3, p3)):
                tau = t - k
                lst.append(float(R.wall_dist(tk, pl[tau, b, :5]).min() - R.hw) if tau >= 0 else np.nan)
            w = np.arange(max(seg[0] - int(round(0.8 / dt)), 0), min(seg[-1] + int(round(0.8 / dt)), T) + 1)
            paths.append(xy[w, b])
            if h.any():
                marks.append(xy[seg[0] + int(h.argmax()), b])
    act, spd, p5, p3, hit = map(np.array, (act, spd, p5, p3, hit))
    name = f"{R.name.split('/')[0]} {R.name.split('_')[-1].split('.')[0]}"
    print(f"{z['exam']} · {MODE[str(z['mode'])]} · {name} — 혼자·빈 트랙(55 s~) 통과 {len(act)}번, 그 가운데 벽에 세게 닿음 {int(hit.sum())}번 ({100 * hit.mean():.1f} %)")
    print(f"   실제 달린 길: 차 옆면에서 벽까지 {q(act)} | 가장 가까울 때 속도 중앙 {np.median(spd):.2f} m/s")
    print(f"   0.5 s 전 plan(0.1~0.5 s 점): {q(p5[np.isfinite(p5)])}")
    print(f"   0.3 s 전 plan(0.1~0.5 s 점): {q(p3[np.isfinite(p3)])}")
    keep.append(dict(R=R, tk=int(R.tid[0, 0]), paths=paths, marks=marks,
                     title=f"{name} · {MODE[str(z['mode'])]}\n벽까지 중앙 {np.median(act) * 100:.0f} cm (아래 10 % {np.percentile(act, 10) * 100:.0f} cm)"
                           f" · {np.median(spd):.1f} m/s\n통과 {len(act)}번 가운데 세게 닿음 {int(hit.sum())}번 ({100 * hit.mean():.0f} %)"))
if FIG and keep:
    fig, axs = plt.subplots(1, len(keep), figsize=(5.6 * len(keep), 6.4), squeeze=False)
    for ax, k in zip(axs[0], keep):
        k["R"].draw_map(ax, k["tk"])
        for p in k["paths"]:
            ax.plot(p[:, 0], p[:, 1], "-", color="#1f77b4", lw=0.8, alpha=0.25, zorder=3)
        if k["marks"]:
            m = np.array(k["marks"]); ax.plot(m[:, 0], m[:, 1], "x", color="#d62728", ms=7, mew=1.8, zorder=5, label="벽에 세게 닿은 자리")
            ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.set_xlim(X0 - PAD, X1 + PAD); ax.set_ylim(Y0 - PAD, Y1 + PAD); ax.set_title(k["title"], fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.suptitle("한 자리를 지나는 길 (혼자 달리는 세션, 상자를 치운 뒤 55 s~; 파란 선 = 통과 하나하나)", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG, dpi=70); print("그림", FIG)
