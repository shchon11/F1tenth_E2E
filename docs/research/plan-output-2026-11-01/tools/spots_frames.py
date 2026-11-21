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
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

OUT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
EX, SP = os.path.join(OUT, "exam"), os.path.join(OUT, "spots")
# live_page.html's tokens: the lane, the walls, the page; hits in its critical red / box amber / car blue
LANE, WALL, PAGE = "#22272e", "#0b0c0e", "#0d0f12"
COLS = {0: "#e66767", 1: "#fab219", 2: "#3987e5"}
# (11-21, the user: "이 그림에 빈 트랙에서 라인이랑 속도 어떻게 가져가는지도 같이 plot좀. 한바퀴 최고시간도") the empty track's
# fastest lap drawn under the hits, its speed from violet (slow) through blue to near white (fast), the same 2-9 m/s scale on every
# checkpoint; the page shows the map at about a third of its pixels on the phone, so line, labels and dots are drawn thick
# ("속도도 좀 빠르고 느린거 구분이 안되잖아": the one-blue ramp at 3 px read as one colour there). VER marks the frames drawn with
# the current look, older ones are drawn again
VER = 6
SPD = ["#7b3fe4", "#4f6ef7", "#2fa6f5", "#5fe0f0", "#e8fbff"]
V0, V1 = 2.0, 9.0
INK = "#e6edf3"


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


def best_lap(R):
    """the empty-track sessions' fastest lap, line crossing to line crossing as exam_table.py counts laps:
    (seconds, session, first decision, last decision) or None"""
    z = R.z
    lapsig = z["lap"]
    solo = ~z["on"][:R.T].any(0) & ~(z["sid"][0] >= 0).any(-1)
    net = np.cumsum(lapsig, 0)
    best = None
    for b in np.nonzero(solo)[0]:
        hi, t_prev = 0, None
        for t in np.nonzero(lapsig[:, b] > 0)[0]:
            if net[t, b] > hi:
                hi = net[t, b]
                if t_prev is not None and not R.rs[t_prev:t + 1, b].any():
                    sec = (t + 1 - t_prev) * R.dt
                    if best is None or sec < best[0]:
                        best = (sec, int(b), int(t_prev), int(t))
                t_prev = t + 1
    return best


def smooth_path(st, vy, dt, sub=8):
    """the driven path between the 0.1 s records as a cubic Hermite curve through each record's position, the velocity
    (heading, speed and side slip) as the tangent: (n, 2) points and (n,) speeds"""
    x, y, h, vx = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    mx, my = dt * (vx * np.cos(h) - vy * np.sin(h)), dt * (vx * np.sin(h) + vy * np.cos(h))
    v = np.hypot(vx, vy)
    s_ = np.linspace(0.0, 1.0, sub, endpoint=False)[:, None]
    h00, h10 = 2 * s_ ** 3 - 3 * s_ ** 2 + 1, s_ ** 3 - 2 * s_ ** 2 + s_
    h01, h11 = -2 * s_ ** 3 + 3 * s_ ** 2, s_ ** 3 - s_ ** 2
    X = (h00 * x[:-1] + h10 * mx[:-1] + h01 * x[1:] + h11 * mx[1:]).T.reshape(-1)
    Y = (h00 * y[:-1] + h10 * my[:-1] + h01 * y[1:] + h11 * my[1:]).T.reshape(-1)
    V = ((1 - s_) * v[:-1] + s_ * v[1:]).T.reshape(-1)
    return np.append(np.stack([X, Y], 1), [[x[-1], y[-1]]], 0), np.append(V, v[-1])


def lap_loop(R, b, t0, t1, tol=0.03, sub=8):
    """(11-22, the user: "대충 그리지 말고 똑바로좀 그리자 ... 아래쪽 말하는거야") the fastest lap as one closed line of driving.
    The timing line sits at the bottom U-turn's exit, where consecutive laps pass up to 1 m apart: joining the lap's end to
    its start there drew a V across the corner. The loop is closed where the next lap (the previous one when there is no
    clean next lap) passes this lap again within `tol` at the same distance to go: this lap from there to the line, then the
    next lap's start up to that point. Returns the curve's points and speeds and the records' points and speeds (labels)"""
    z, st = R.z, R.st[:, b].astype(float)
    rem = z["rem"][:, b].astype(float)
    vy = np.append(z["vy"][:, b], z["vy"][-1:, b]).astype(float)
    N, last = t1 - t0 + 1, R.st.shape[0] - 1

    def at_rem(js, r):
        """the first record j in js whose step passes distance to go r, and the point there"""
        for j in js:
            if 0 <= j < last and rem[j] >= r > rem[j + 1] and rem[j] - rem[j + 1] < 3.0:
                f = (rem[j] - r) / (rem[j] - rem[j + 1])
                return j, st[j, :2] + f * (st[j + 1, :2] - st[j, :2])
        return None, None

    best = None                                              # (gap [m], start record, end record, forward)
    w_ = N + 20
    if t1 + 1 + w_ <= last and not R.rs[t1 + 1: t1 + 1 + w_, b].any():
        for i in range(t0, t1 + 1):                          # this lap from i, the next lap up to i's distance to go
            j, q = at_rem(range(t1 + 1, t1 + 1 + w_), rem[i])
            if j is not None:
                d = float(np.hypot(*(q - st[i, :2])))
                if best is None or d < best[0]:
                    best = (d, i, j + 1, True)
                if d < tol:
                    break
    if (best is None or best[0] >= tol) and t0 - w_ >= 0 and not R.rs[t0 - w_: t0, b].any():
        for i in range(t1, t0 - 1, -1):                      # the previous lap from i's distance to go, this lap up to i
            j, q = at_rem(range(t0 - 1, t0 - w_ - 1, -1), rem[i])
            if j is not None:
                d = float(np.hypot(*(q - st[i, :2])))
                if best is None or d < best[0] - 1e-9:
                    best = (d, j, i, False)
                if d < tol:
                    break
    if best is None:                                         # no clean lap next to it: the timing lap, left open
        P, V = smooth_path(st[t0: t1 + 2], vy[t0: t1 + 2], R.dt, sub)
        return P, V, st[t0: t1 + 2, :2], np.hypot(st[t0: t1 + 2, 3], vy[t0: t1 + 2])
    _, a, c, fwd = best
    P, V = smooth_path(st[a: c + 1], vy[a: c + 1], R.dt, sub)
    if fwd:                                                  # cut the last step where it comes back to the start record
        k0 = (c - 1 - a) * sub
        k = k0 + int(np.argmin(np.hypot(*(P[k0:] - st[a, :2]).T)))
        P, V, rec = P[:k + 1], V[:k + 1], slice(a, c)
    else:                                                    # cut the first step where it leaves from the end record
        k = int(np.argmin(np.hypot(*(P[:sub + 1] - st[c, :2]).T)))
        P, V, rec = P[k:], V[k:], slice(a + 1, c + 1)
    P, V = np.vstack([P, P[:1]]), np.append(V, V[0])         # closed: the step left is the gap measured above
    return P, V, st[rec, :2], np.hypot(st[rec, 3], vy[rec])


def speed_marks(v, w=6, drop=0.8):
    """the slowest point of each corner (a local minimum the speed rises at least `drop` from on both sides; the lap is a loop,
    so the windows wrap around the line) and the fastest point"""
    n, out = len(v), []
    vv = np.concatenate([v, v, v])
    for i in range(n):
        j = i + n
        if v[i] <= vv[j - w: j + w + 1].min() + 1e-9 and vv[j - 3 * w: j + 1].max() - v[i] >= drop and \
                vv[j: j + 3 * w + 1].max() - v[i] >= drop:
            if not out or (i - out[-1] > w and out[0] + n - i > w):     # a flat bottom: its first point only
                out.append(i)
    return out + [int(np.argmax(v))]


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
    bl = best_lap(R)
    if bl is not None:
        _, b, t0, t1 = bl
        xc, vc, xy, v = lap_loop(R, b, t0, t1)
        lc = LineCollection(np.stack([xc[:-1], xc[1:]], 1), cmap=LinearSegmentedColormap.from_list("spd", SPD),
                            norm=plt.Normalize(V0, V1), linewidths=8.0, capstyle="round", joinstyle="round", zorder=2)
        lc.set_array(0.5 * (vc[:-1] + vc[1:]))
        ax.add_collection(lc)
        for i in speed_marks(v):
            ax.annotate(f"{v[i]:.1f}", xy[i], xytext=(9, 6), textcoords="offset points", fontsize=24, fontweight="bold",
                        color=INK, zorder=4, path_effects=[pe.withStroke(linewidth=5, foreground=PAGE)])
    n = {}
    for k, c in COLS.items():
        t, b = np.nonzero(hard[..., k] > 0)
        n[k] = int(len(t))
        if len(t):
            ax.scatter(st[t, b, 0], st[t, b, 1], s=60, c=c, alpha=0.9, edgecolors=PAGE, linewidths=0.8, zorder=3)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    d = os.path.join(SP, run); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"u{u}.png"), dpi=100, facecolor=PAGE)
    plt.close(fig)
    info = {"run": run, "u": int(u), "hard": n[0] + n[1] + n[2], "wall": n[0], "box": n[1], "car": n[2],
            "per_session": round((n[0] + n[1] + n[2]) / R.B, 3), "sessions": int(R.B), "empty_m": round(empty_m(R), 1),
            "mtime": round(os.path.getmtime(p), 1), "v": VER,
            "best_lap": round(bl[0], 2) if bl is not None else None}
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
        if os.path.exists(js) and os.path.exists(os.path.join(d, f"u{u}.png")) and json.load(open(js)).get("mtime") == round(os.path.getmtime(p), 1) \
                and json.load(open(js)).get("v") == VER:
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
