"""시험 충돌 지도 한 장씩 (11-20, 사용자: "저 이미지 추이들을 여기다 계속 하지말고 차라리 사이트 거기 실시간에다가 저걸 계속 업로드를
해줘. 쭈르르륵 드래그하면 어디박았는지 볼 수 있게").
exam2(같은 128세션) 기록마다 세게 닿은 자리 한 장(사이트 실시간 페이지의 어두운 색)과 숫자를 out/spots/<run>/u<U>.png · u<U>.json으로,
run마다 index.json(update 순). 이미 그린 checkpoint는 건너뜀 -> spots_watch.sh가 1분마다 돌려 새 시험만 그림.
인자: [run ...]  (없으면 exam2 기록이 있는 모든 run, 최근에 바뀐 run부터)"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

OUT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
EX, SP = os.path.join(OUT, "exam"), os.path.join(OUT, "spots")
# live_page.html's tokens: the lane, the walls, the page; hits in its critical red / box amber / car blue
LANE, WALL, PAGE = "#22272e", "#0b0c0e", "#0d0f12"
COLS = {0: "#e66767", 1: "#fab219", 2: "#3987e5"}


def exams(run):
    out = {}
    for p in glob.glob(os.path.join(EX, f"exam2_{run}_u*_ideal_line3_n256.npz")):
        m = re.search(rf"exam2_{re.escape(run)}_u(\d+)_ideal", p)
        if m:
            out[int(m.group(1))] = p
    return out


def empty_m(R):
    """the empty-track sessions' distance along the track in 90 s (exam_spots_fig.py's)"""
    z = R.z
    T_ = z["cnt"].shape[0]
    r_ = z["rem"].astype(float).copy()
    for t_ in range(1, r_.shape[0]):
        bad = ~np.isfinite(r_[t_]); r_[t_, bad] = r_[t_ - 1, bad]
    r_[~np.isfinite(r_)] = 0.0
    L_ = float(z["lapL"][int(z["tid"][0, 0])])
    dd = np.mod(r_[:-1] - r_[1:] + 0.5 * L_, L_) - 0.5 * L_
    solo = ~z["on"][:T_].any(0) & ~(z["sid"][0] >= 0).any(-1)
    return float(dd.sum(0)[solo].mean())


def frame(run, u, p):
    R = Rec(p)
    g, ext = R.grid(0)
    hard = R.d[..., 3:6]
    st = R.st[:-1]
    ys, xs = np.nonzero(g > 0)
    x0 = ext[0] + xs.min() * (ext[1] - ext[0]) / g.shape[1] - 0.4
    x1 = ext[0] + xs.max() * (ext[1] - ext[0]) / g.shape[1] + 0.4
    y0 = ext[2] + ys.min() * (ext[3] - ext[2]) / g.shape[0] - 0.4
    y1 = ext[2] + ys.max() * (ext[3] - ext[2]) / g.shape[0] + 0.4
    h = 10.0
    fig = plt.figure(figsize=(h * (x1 - x0) / (y1 - y0), h), facecolor=PAGE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(PAGE)
    ax.imshow((g <= 0).astype(float), origin="lower", extent=ext, cmap=ListedColormap([LANE, WALL]), vmin=0, vmax=1,
              interpolation="nearest")
    n = {}
    for k, c in COLS.items():
        t, b = np.nonzero(hard[..., k] > 0)
        n[k] = int(len(t))
        if len(t):
            ax.scatter(st[t, b, 0], st[t, b, 1], s=26, c=c, alpha=0.8, edgecolors="none")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    d = os.path.join(SP, run); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"u{u}.png"), dpi=100, facecolor=PAGE)
    plt.close(fig)
    info = {"run": run, "u": int(u), "hard": n[0] + n[1] + n[2], "wall": n[0], "box": n[1], "car": n[2],
            "per_session": round((n[0] + n[1] + n[2]) / R.B, 3), "sessions": int(R.B), "empty_m": round(empty_m(R), 1),
            "mtime": round(os.path.getmtime(p), 1)}
    json.dump(info, open(os.path.join(d, f"u{u}.json"), "w"), ensure_ascii=False)
    return info


def update(run):
    ex = exams(run)
    if not ex:
        return 0
    d = os.path.join(SP, run)
    made = 0
    for u, p in sorted(ex.items()):
        js = os.path.join(d, f"u{u}.json")
        if os.path.exists(js) and os.path.exists(os.path.join(d, f"u{u}.png")) and json.load(open(js)).get("mtime") == round(os.path.getmtime(p), 1):
            continue
        try:
            frame(run, u, p); made += 1
        except Exception as exc:                          # a broken record must not stop the others
            print(f"{run} u{u}: {exc}", flush=True)
    rows = []
    for js in glob.glob(os.path.join(d, "u*.json")):
        try:
            rows.append(json.load(open(js)))
        except ValueError:
            pass
    rows.sort(key=lambda r: r["u"])
    tmp = os.path.join(d, "index.json.tmp")
    json.dump(rows, open(tmp, "w"), ensure_ascii=False); os.replace(tmp, os.path.join(d, "index.json"))
    return made


if __name__ == "__main__":
    runs = sys.argv[1:]
    if not runs:
        seen = {}
        for p in glob.glob(os.path.join(EX, "exam2_*_u*_ideal_line3_n256.npz")):
            m = re.search(r"exam2_(.+)_u\d+_ideal", os.path.basename(p))
            if m:
                seen[m.group(1)] = max(seen.get(m.group(1), 0), os.path.getmtime(p))
        runs = sorted(seen, key=lambda r: -seen[r])
    for r in runs:
        k = update(r)
        if k:
            print(f"{r}: 새로 그림 {k}장", flush=True)
