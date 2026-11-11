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
            setattr(self, k, z[k])
        self.dt = float(z["dt"]); self.T, self.B = self.rs.shape
        self.hl, self.hw = 0.5 * float(z["car"][0]), 0.5 * float(z["car"][1])
        self.name = str(z["name"]); self.kinds = [str(k) for k in z["kinds"]]
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
