"""평가 결과 그림 (11-16, 사용자: "이거 평가했던거만 계속 보내주면 될 것 같아. 어차피 그래프로 추이 팔로업 하고있으니 그래프랑 평가표랑 저 그림").
frozen exam(같은 128세션) 기록으로 한 장: 왼쪽 = run마다 가장 최근 시험의 세게 닿은 자리(벽 빨강 · 상자 주황 · 차 파랑),
오른쪽 = checkpoint별 세션당 세게 닿음(run마다 한 선, 이어 받은 run은 출발 checkpoint에서 시작).
인자: 출력png run [run ...]   (예: exam_spots_fig.py out.png kr26c kr26e kr26f)
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, plt  # noqa: E402
from run_chain import parent  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402

for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break

EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "exam")
COLS = {0: ("#e03131", "벽"), 1: ("#e8890c", "상자"), 2: ("#1971c2", "차")}
LINE = ["#868e96", "#2a7de1", "#e8590c", "#2f9e44", "#ae3ec9", "#f59f00"]


def exams(run):
    """{update: npz path} of a run's frozen exams"""
    out = {}
    for p in glob.glob(os.path.join(EX, f"exam1_{run}_u*_ideal_uniform_n256.npz")):
        m = re.search(rf"exam1_{run}_u(\d+)_ideal", p)
        if m:
            out[int(m.group(1))] = p
    return out


_cache = {}


def rec(p):
    if p not in _cache:
        _cache[p] = Rec(p)
    return _cache[p]


def hard_per_session(p):
    R = rec(p)
    return float(R.d[..., 3:6].sum()) / R.B


out, runs = sys.argv[1], sys.argv[2:]
latest = [(r, max(exams(r))) for r in runs if exams(r)]
fig = plt.figure(figsize=(5.0 * len(latest) + 7.5, 6.8))
gs = fig.add_gridspec(1, len(latest) + 1, width_ratios=[1] * len(latest) + [1.45])
for j, (r, u) in enumerate(latest):
    ax = fig.add_subplot(gs[0, j])
    R = rec(exams(r)[u])
    g, ext = R.grid(0)
    ax.imshow((g <= 0).astype(float), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
    hard = R.d[..., 3:6]
    st = R.st[:-1]
    tot = 0
    for k, (c, kl) in COLS.items():
        t, b = np.nonzero(hard[..., k] > 0)
        ax.scatter(st[t, b, 0], st[t, b, 1], s=22, c=c, alpha=0.65, edgecolors="none", label=f"{kl} {len(t)}")
        tot += len(t)
    ax.set_title(f"{r} u{u}\n세게 닿음 {tot}번 (같은 {R.B}세션)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ys, xs = np.nonzero(g > 0)
    ax.set_xlim(ext[0] + xs.min() * (ext[1] - ext[0]) / g.shape[1] - 0.5, ext[0] + xs.max() * (ext[1] - ext[0]) / g.shape[1] + 0.5)
    ax.set_ylim(ext[2] + ys.min() * (ext[3] - ext[2]) / g.shape[0] - 0.5, ext[2] + ys.max() * (ext[3] - ext[2]) / g.shape[0] + 0.5)
ax = fig.add_subplot(gs[0, -1])
for i, r in enumerate(runs):
    E = exams(r)
    if not E:
        continue
    pts = sorted(E.items())
    par, cut = parent(r)
    if par and cut in exams(par):                          # start the line at the checkpoint the run was continued from
        pts = [(cut, exams(par)[cut])] + pts
    x = [u for u, _ in pts]
    y = [hard_per_session(p) for _, p in pts]
    ax.plot(x, y, "-o", color=LINE[i % len(LINE)], lw=2, ms=5, label=r)
    ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points", xytext=(6, 0), fontsize=9, color=LINE[i % len(LINE)], va="center")
ax.set_xlabel("checkpoint (update)")
ax.set_ylabel("세게 닿음 / 세션 (90 s, 같은 세션)")
ax.grid(alpha=0.25)
ax.legend(fontsize=9, frameon=False)
ax.set_title("frozen exam: checkpoint별 세게 닿음", fontsize=11)
fig.suptitle("Korea 2026 frozen exam: 가장 최근 시험의 세게 닿은 자리 · checkpoint별 세게 닿음", fontsize=13)
fig.tight_layout()
fig.savefig(out, dpi=105)
print(out)
