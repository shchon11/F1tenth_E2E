"""drive_record.py의 기록(npz)을 읽고 그리는 공통 부분: 지도(벽 거리 격자), 차·상자 외곽선, 결정마다 새로 닿은 수, 에피소드 시작 결정."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False

MAPC = ListedColormap(["#666666", "white"])
C_WALL, C_BOX, C_CAR = "#222222", "#a0522d", "#d62728"


class Rec:
    def __init__(self, fn):
        z = np.load(fn, allow_pickle=False)
        self.z = z
        for k in ("st", "op", "on", "pl", "rs", "cnt", "held", "sid", "pose", "tid", "kind", "scale"):
            setattr(self, k, z[k] if k in z.files else None)     # box_response.py records carry no opponent
        self.dt = float(z["dt"]); self.T, self.B = self.rs.shape
        car = z["car"] if "car" in z.files else (0.58, 0.31)                # box_response.py records: the same car
        self.hl, self.hw = 0.5 * float(car[0]), 0.5 * float(car[1])
        self.name = str(z["name"]); self.kinds = [str(k) for k in z["kinds"]] if "kinds" in z.files else []
        self._edt, self._off, self._H, self._W = z["edt"], z["t_off"], z["t_H"].astype(int), z["t_W"].astype(int)
        self._res, self._org = z["t_res"], z["t_origin"]
        self._grid, self._fp = {}, {}
        # the decision index each row's episode began at, and the events (new contacts in a decision that did not restart)
        self.k0 = np.zeros((self.T, self.B), int)
        cur = np.zeros(self.B, int)
        for t in range(self.T):
            self.k0[t] = cur
            cur = np.where(self.rs[t], t + 1, cur)
        prev = np.concatenate([np.zeros((1, self.B, 6), np.float32), self.cnt[:-1]], 0)
        prev[1:][self.rs[:-1]] = 0.0                                   # a restarted row counts from 0 again
        self.d = np.clip(self.cnt - prev, 0, None) * (~self.rs)[..., None]

    def grid(self, k):
        if k not in self._grid:
            g = self._edt[self._off[k]: self._off[k] + self._H[k] * self._W[k]].reshape(self._H[k], self._W[k])
            self._grid[k] = (g, [self._org[k][0], self._org[k][0] + self._W[k] * self._res[k],
                                 self._org[k][1], self._org[k][1] + self._H[k] * self._res[k]])
        return self._grid[k]

    def wall_dist(self, k, pts):
        """[m] distance to the nearest wall at each point (0 in a wall / off the grid)"""
        g, _ = self.grid(k)
        c = np.floor((pts[:, 0] - self._org[k][0]) / self._res[k]).astype(int)
        r = np.floor((pts[:, 1] - self._org[k][1]) / self._res[k]).astype(int)
        ok = (c >= 0) & (c < self._W[k]) & (r >= 0) & (r < self._H[k])
        out = np.zeros(len(pts))
        out[ok] = g[r[ok], c[ok]]
        return out

    def ray(self, k, x, y, yaw, maxd=15.0, step=0.05):
        """[m] from the car's nose along its heading to the first wall cell"""
        d = np.arange(self.hl, maxd, step)
        pts = np.stack([x + d * np.cos(yaw), y + d * np.sin(yaw)], 1)
        hit = self.wall_dist(k, pts) <= 0.0
        return float(d[hit.argmax()] - self.hl) if hit.any() else maxd

    def body(self, x, y, yaw):
        c, s_ = np.cos(yaw), np.sin(yaw)
        q = np.array([[self.hl, self.hw], [self.hl, -self.hw], [-self.hl, -self.hw], [-self.hl, self.hw]])
        return np.stack([x + q[:, 0] * c - q[:, 1] * s_, y + q[:, 0] * s_ + q[:, 1] * c], 1)

    def boxes(self, t, b):
        polys = []
        for j, k in enumerate(self.sid[t, b]):
            if k < 0:
                continue
            if int(k) not in self._fp:
                self._fp[int(k)] = self.z[f"fp_{int(k)}"].astype(float)
            q = self._fp[int(k)]; p = self.pose[t, b, j]; c, s_ = np.cos(p[2]), np.sin(p[2])
            polys.append(np.stack([p[0] + q[:, 0] * c - q[:, 1] * s_, p[1] + q[:, 0] * s_ + q[:, 1] * c], 1))
        return polys

    def draw_map(self, ax, k):
        g, ext = self.grid(k)
        ax.set_facecolor("#666666")
        ax.imshow(g > 0.0, cmap=MAPC, vmin=0, vmax=1, origin="lower", extent=ext, zorder=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def trk_name(t):
    return "원래 트랙" if t == 0 else f"변형 트랙 {t}"


def along_track(z, T):
    """(T, B) place along the track [m, 0 at the line] from the record's walls-only distance to go (the reward's field; a wall /
    off-grid sample holds the last value) and the lap index. For lining laps up in ANALYSIS only -- never a policy input."""
    L = float(z["lapL"][int(z["tid"][0, 0])])
    rem = z["rem"].astype(float)
    for t in range(1, rem.shape[0]):
        bad = ~np.isfinite(rem[t]); rem[t, bad] = rem[t - 1, bad]
    rem[~np.isfinite(rem)] = 0.0
    s = np.mod(L - rem[:T], L)
    d = np.mod(rem[:-1] - rem[1:] + 0.5 * L, L) - 0.5 * L
    B = rem.shape[1]
    lap = np.floor((s[0][None] + np.concatenate([np.zeros((1, B)), np.cumsum(d, 0)], 0)[:T]) / L).astype(int)
    return s, lap, L


def auc(score, y):
    """share of (positive, negative) pairs the score puts in the right order (0.5 = no information)"""
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def auc_se(a, n1, n0):
    """Hanley-McNeil standard error of an AUC from n1 positives and n0 negatives"""
    if not np.isfinite(a) or n1 < 2 or n0 < 2:
        return float("nan")
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return float(np.sqrt(max(a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a), 0.0) / (n1 * n0)))


def probe_auc(X, y, g, lam=0.1):
    """a linear read-out of y from X (ridge to +-1, classes balanced, inputs standardized), fit on the even groups `g` and
    scored on the odd ones and the other way round: (mean AUC of the two halves, its standard error). Groups = sessions, so
    nothing of a scored session was seen in the fit."""
    X = np.asarray(X, np.float64); y = np.asarray(y).astype(bool); g = np.asarray(g)
    res = []
    for h in (0, 1):
        tr, te = (g % 2 == h), (g % 2 != h)
        if X.shape[1] == 0 or y[tr].all() or (~y[tr]).all() or y[te].all() or (~y[te]).all():
            return float("nan"), float("nan")
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-4
        A = (X[tr] - mu) / sd
        w = np.where(y[tr], 0.5 / y[tr].mean(), 0.5 / (1 - y[tr].mean()))
        t = np.where(y[tr], 1.0, -1.0)
        coef = np.linalg.solve((A * w[:, None]).T @ A / w.sum() + lam * np.eye(A.shape[1]), (A * w[:, None]).T @ (t - (w * t).sum() / w.sum()) / w.sum())
        a = auc(((X[te] - mu) / sd) @ coef, y[te])
        res.append((a, auc_se(a, int(y[te].sum()), int((~y[te]).sum()))))
    return float(np.mean([r[0] for r in res])), float(np.sqrt(sum(r[1] ** 2 for r in res)) / 2)
