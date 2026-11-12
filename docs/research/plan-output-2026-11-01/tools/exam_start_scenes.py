"""출발 직후(처음 5 s) 세게 닿은 세션의 장면 (exam_start_scenes.py, 11-12). 시험 기록(exam_record.py npz)에서 숫자보다 먼저 볼 것.
세션마다 한 칸: 출발부터 닿은 뒤 1.5 s까지 0.4 s마다 우리 차(파랑, 옅음 -> 짙음 = 시간)와 상대 차(주황, 같은 시각), 상자(갈색), 닿은 곳(검은 x),
닿기 0.5 s 전의 plan(검은 선). 제목 = 기록·차 번호·무엇에 닿았나·언제·그때 우리 속도 / 상대 속도·상대 차 종류와 속도 배율.
아래 칸 = 같은 세션의 속도(우리 파랑, 상대 주황)와 닿은 시각(세로선).
표: 기록 x 경우마다 세션 수와 출발 5 s 안에 세게 닿은 세션 수, 건마다 출발 때 상대 차가 놓인 자리(우리 차 기준 앞 + / 왼쪽 + [m])와 속도.
인자: 출력png 기록npz [기록npz ...] [--only 벽,상자,차 가운데 그릴 것(쉼표로; 기본 전부)]  (표는 늘 전부)"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, plt, C_BOX
from matplotlib.colors import LinearSegmentedColormap

N5, STEP, AFTER = 50, 4, 15                                            # decisions: the start phase, between drawn bodies, drawn after the hit
BLUE = LinearSegmentedColormap.from_list("b", ["#b9d3f1", "#0d3d7a"])
ORNG = LinearSegmentedColormap.from_list("o", ["#f6c9b3", "#a8380e"])
INK, MUTED = "#0b0b0b", "#898781"
WHAT = ("벽", "상자", "상대 차")

argv = sys.argv[1:]
ONLY = (0, 1, 2)
if "--only" in argv:
    i = argv.index("--only"); ONLY = tuple(dict(벽=0, 상자=1, 차=2)[w] for w in argv[i + 1].split(",")); del argv[i:i + 2]
out_png, files = argv[0], argv[1:]
ev = []
for fn in files:
    r = Rec(fn)
    tag = os.path.basename(fn).replace("exam1_", "").replace("_n256", "").replace(".npz", "")
    box = (r.sid[0] >= 0).any(1); opp = r.on[:N5].any(0)
    hard = r.d[..., 3:]
    print(f"== {tag} ({r.name}) ==")
    for lab, m in (("혼자·빈 트랙", ~box & ~opp), ("혼자·상자", box & ~opp), ("상대 차", ~box & opp), ("상대 차·상자", box & opp)):
        bs = np.nonzero(m & (hard[:N5].sum((0, 2)) > 0))[0]
        print(f"  [{lab}] 세션 {int(m.sum())}개 · 출발 5 s 안에 세게 닿은 세션 {len(bs)}개")
        for b in bs:
            t = int(np.nonzero(hard[:N5, b].sum(1) > 0)[0][0])
            k = int(np.argmax(hard[t, b]))
            s = r.st[:, b].astype(float); o = r.op[:, b].astype(float)
            line = f"    차 {b}: {t * r.dt:.1f} s에 {WHAT[k]} · 닿기 직전 속도 {s[max(t - 1, 0), 3]:.1f} m/s"
            if opp[b]:
                c, sn = np.cos(s[0, 2]), np.sin(s[0, 2]); dx, dy = o[0, 0] - s[0, 0], o[0, 1] - s[0, 1]
                line += (f" · 상대 {r.kinds[int(r.kind[0, b])]} 배율 {float(r.scale[0, b]):.2f}, 출발 때 우리 차 기준 앞 {c * dx + sn * dy:+.1f} m / 왼쪽 "
                         f"{-sn * dx + c * dy:+.1f} m, 닿을 때 상대 속도 {o[t, 3]:.1f} m/s")
            line += f" · 5 s 때 우리 속도 {s[N5, 3]:.1f} m/s"
            print(line)
            ev.append((r, tag, lab, int(b), t, k))
ev = [e for e in ev if e[5] in ONLY]
if not ev:
    print("그릴 세션 없음"); sys.exit(0)

n = len(ev); cols = min(n, 4); rows = int(np.ceil(n / cols))
fig = plt.figure(figsize=(5.4 * cols, 7.4 * rows + 0.5))
gs = fig.add_gridspec(3 * rows, cols, height_ratios=[3.2, 1.0, 0.55] * rows, hspace=0.12, wspace=0.12,
                      top=1 - 0.75 / (7.4 * rows + 0.5), bottom=0.0)
for i, (r, tag, lab, b, t, k) in enumerate(ev):
    ax = fig.add_subplot(gs[3 * (i // cols), i % cols])
    tid = int(r.tid[0, b]); r.draw_map(ax, tid)
    end = min(t + AFTER, r.T - 1)
    s = r.st[:, b].astype(float); o = r.op[:, b].astype(float); on = r.on[:, b].astype(bool)
    for q in r.boxes(0, b):
        ax.fill(q[:, 0], q[:, 1], color=C_BOX, zorder=2, lw=0)
    ts = list(range(0, end + 1, STEP))
    for j, tt in enumerate(ts):
        f = j / max(len(ts) - 1, 1)
        q = r.body(*s[tt, :3]); ax.fill(q[:, 0], q[:, 1], facecolor=BLUE(f), edgecolor="white", lw=0.6, zorder=5)
        if on[tt]:
            q = r.body(*o[tt, :3]); ax.fill(q[:, 0], q[:, 1], facecolor=ORNG(f), edgecolor="white", lw=0.6, zorder=4)
    tp = max(t - 5, 0)
    p = np.concatenate([s[tp, :2][None], r.pl[tp, b]], 0); ax.plot(p[:, 0], p[:, 1], "-", color=INK, lw=1.3, zorder=6)
    ax.plot(s[t, 0], s[t, 1], "x", color=INK, ms=11, mew=2.4, zorder=8)
    ax.plot(s[0, 0], s[0, 1], "o", color="white", mec=INK, ms=5, zorder=8)
    xs = np.r_[s[:end + 1, 0], o[:end + 1, 0][on[:end + 1]]]; ys = np.r_[s[:end + 1, 1], o[:end + 1, 1][on[:end + 1]]]
    cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max()); h = max(0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()) + 1.2, 2.5)
    ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy - h, cy + h)
    ttl = f"{tag} · 차 {b} · {lab}\n{t * r.dt:.1f} s에 {WHAT[k]} (우리 {s[max(t - 1, 0), 3]:.1f} m/s"
    if on[t]:
        ttl += f", 상대 {o[t, 3]:.1f} m/s; {r.kinds[int(r.kind[0, b])]} x{float(r.scale[0, b]):.2f}"
    ax.set_title(ttl + ")", fontsize=9, loc="left")
    a2 = fig.add_subplot(gs[3 * (i // cols) + 1, i % cols])
    tx = np.arange(end + 1) * r.dt
    a2.plot(tx, s[:end + 1, 3], "-", color=BLUE(0.75), lw=2, label="우리 차")
    if on[:end + 1].any():
        a2.plot(tx, np.where(on[:end + 1], o[:end + 1, 3], np.nan), "-", color=ORNG(0.75), lw=2, label="상대 차")
    a2.axvline(t * r.dt, color=INK, lw=1.2)
    a2.set_ylim(-0.3, 8.0); a2.set_xlim(0, tx[-1]); a2.grid(color="#e6e4dd", lw=0.8); a2.set_axisbelow(True)
    for sp_ in ("top", "right"):
        a2.spines[sp_].set_visible(False)
    a2.set_xlabel("출발 뒤 시간 [s]", fontsize=8.5, labelpad=1); a2.tick_params(labelsize=8)
    if i % cols == 0:
        a2.set_ylabel("속도 [m/s]", fontsize=9)
    if i == 0:
        a2.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
fig.suptitle("출발 5 s 안에 세게 닿은 세션: 0.4 s마다 우리 차(파랑)와 상대 차(주황), 옅음 -> 짙음 = 시간 · 흰 점 = 출발 자리 · 검은 x = 닿은 곳 · "
             "검은 선 = 닿기 0.5 s 전의 plan · 갈색 = 상자", fontsize=11, x=0.01, ha="left", y=1 - 0.15 / (7.4 * rows + 0.5))
fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"그림: {out_png}")
