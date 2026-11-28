"""이상 실행 충돌 분석 공통 부분 (2026-11-28). 기록 = exam_record.py의 npz (감사 C의 C_rec).

- 촘촘한 궤적: 결정 사이(0.1 s)를 10칸으로, 위치는 Hermite(접선 = 차 방향 x 현 길이), 방향은 선형.
- 차체 = 기록의 차 크기(0.58 x 0.31 m) 직사각형, 중심 = 상태 (x, y). 상자 = 기록의 외곽선.
- 부호 있는 여유: 떨어져 있으면 두 다각형 사이 최단거리, 겹치면 -(SAT 침투 깊이).
- LiDAR 가시성: 스캐너 = 차 중심 + mount_x 앞, 1081빔 / 270도 / 10 m. 벽은 기록의 벽 거리 격자(EDT)로 구 추적,
  상자는 외곽선과의 해석적 교차. 상자에 맞은 빔 수(다른 상자·벽보다 먼저 맞은 것)를 셈. 시뮬레이터 LiDAR의 잡음·빔 누락·
  바닥 맞음은 넣지 않음 (기하학적 가시성).
"""
import sys
import numpy as np

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec  # noqa: E402

N_SUB = 10
FOV = np.deg2rad(270.0)
NB = 1081
RMAX = 10.0
ANG = np.linspace(-FOV / 2, FOV / 2, NB)


def load_limits(fn="/data/pm-ideal-collisions-2026-11-28/limits_scan31b.npz"):
    L = np.load(fn)
    return {k: (L[k] if np.ndim(L[k]) else float(L[k])) for k in L.files}


def kmax(v, LIM):
    """이상 실행 계획의 곡률 한계 [1/m] (타이어 표 x 0.95, plan_steps.Limits.kappa_at)"""
    return np.interp(np.abs(v), LIM["v_grid"], LIM["k_tyre"])


# ---------------------------------------------------------------- geometry
def body_poly(x, y, yaw, hl, hw):
    """(..., 4, 2)"""
    x, y, yaw = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(yaw, float))
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    q = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return np.stack([x[..., None] + q[:, 0] * c - q[:, 1] * s, y[..., None] + q[:, 0] * s + q[:, 1] * c], -1)


def _pt_seg(p, a, b):
    """p (..., 2), a/b (..., 2) -> (...)"""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def signed_clear(A, Bp):
    """A (N, 4, 2) convex (car bodies), Bp (M, 2) convex polygon (a box) -> (N,) signed clearance [m]"""
    A = np.asarray(A, float); Bp = np.asarray(Bp, float)
    N = A.shape[0]
    ea = np.roll(A, -1, 1) - A                       # (N, 4, 2)
    eb = np.roll(Bp, -1, 0) - Bp                     # (M, 2)
    na = np.stack([-ea[..., 1], ea[..., 0]], -1); na /= np.linalg.norm(na, axis=-1, keepdims=True) + 1e-12
    nb = np.stack([-eb[:, 1], eb[:, 0]], -1); nb /= np.linalg.norm(nb, axis=-1, keepdims=True) + 1e-12
    axes = np.concatenate([na, np.broadcast_to(nb[None], (N,) + nb.shape)], 1)     # (N, 4+M, 2)
    pa = np.einsum("nvk,nak->nav", A, axes)                                          # (N, axes, 4)
    pb = np.einsum("vk,nak->nav", Bp, axes)                                          # (N, axes, M)
    gap = np.maximum(pb.min(-1) - pa.max(-1), pa.min(-1) - pb.max(-1))               # (N, axes)
    sat = gap.max(1)
    # exact distance when separated: vertex-edge both ways
    d1 = _pt_seg(A[:, :, None, :], Bp[None, None, :, :], np.roll(Bp, -1, 0)[None, None, :, :]).min((1, 2))
    d2 = _pt_seg(Bp[None, :, None, :], A[:, None, :, :], np.roll(A, -1, 1)[:, None, :, :]).min((1, 2))
    d = np.minimum(d1, d2)
    return np.where(sat > 0, d, sat)


def dense_traj(st, t0, t1, b):
    """decisions t0..t1 (states t0..t1 inclusive) -> (n, 3) x, y, yaw at N_SUB steps per decision, and (n,) time [decisions]"""
    S = st[t0:t1 + 1, b].astype(float)
    if len(S) < 2:
        return S[:, :3], np.arange(t0, t0 + len(S), dtype=float)
    p0, p1 = S[:-1, :2], S[1:, :2]
    y0 = S[:-1, 2]; y1 = y0 + np.angle(np.exp(1j * (S[1:, 2] - S[:-1, 2])))
    ch = np.linalg.norm(p1 - p0, axis=1)
    sgn = np.sign(np.where(np.abs(S[:-1, 3]) > 0.05, S[:-1, 3], 1.0))
    m0 = (sgn * ch)[:, None] * np.stack([np.cos(y0), np.sin(y0)], 1)
    m1 = (sgn * ch)[:, None] * np.stack([np.cos(y1), np.sin(y1)], 1)
    f = np.linspace(0, 1, N_SUB, endpoint=False)
    h00, h10, h01, h11 = 2 * f ** 3 - 3 * f ** 2 + 1, f ** 3 - 2 * f ** 2 + f, -2 * f ** 3 + 3 * f ** 2, f ** 3 - f ** 2
    P = (h00[None, :, None] * p0[:, None] + h10[None, :, None] * m0[:, None] + h01[None, :, None] * p1[:, None]
         + h11[None, :, None] * m1[:, None])
    Y = y0[:, None] + f[None] * (y1 - y0)[:, None]
    out = np.concatenate([P.reshape(-1, 2), Y.reshape(-1, 1)], 1)
    out = np.concatenate([out, S[-1:, :3]], 0)
    tt = np.concatenate([(np.arange(t0, t1)[:, None] + f[None]).reshape(-1), [t1]])
    return out, tt


def plan_poses(st_t, pl_t):
    """the plan drawn at a decision: car pose now + its 10 points, densified x5, heading from the polyline -> (n, 3)"""
    P = np.concatenate([st_t[None, :2], pl_t], 0).astype(float)
    seg = []
    for a, b_ in zip(P[:-1], P[1:]):
        for f in np.linspace(0, 1, 5, endpoint=False):
            seg.append(a + f * (b_ - a))
    seg.append(P[-1])
    Q = np.array(seg)
    d = np.diff(Q, axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    small = np.linalg.norm(d, axis=1) < 1e-3
    yaw[small] = st_t[2]
    yaw = np.concatenate([[st_t[2]], yaw[1:], [yaw[-1]]]) if len(yaw) > 1 else np.array([st_t[2]] * len(Q))
    yaw[0] = st_t[2]
    return np.concatenate([Q, yaw[:len(Q), None]], 1)


# ---------------------------------------------------------------- LiDAR
class Caster:
    def __init__(self, R: Rec):
        self.R = R
        self.k = 0
        self.g, ext = R.grid(0)
        self.org = np.array([ext[0], ext[2]]); self.res = float(R._res[0])
        self.H, self.W = self.g.shape

    def edt(self, p):
        c = np.floor((p[..., 0] - self.org[0]) / self.res).astype(int)
        r = np.floor((p[..., 1] - self.org[1]) / self.res).astype(int)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        out = np.zeros(p.shape[:-1])
        out[ok] = self.g[r[ok], c[ok]]
        return out

    def wall_range(self, o, ang):
        """o (2,), ang (n,) -> (n,) [m] distance to the first wall cell (RMAX if none)"""
        d = np.stack([np.cos(ang), np.sin(ang)], 1)
        s = np.zeros(len(ang)); done = np.zeros(len(ang), bool)
        for _ in range(400):
            p = o[None] + s[:, None] * d
            e = self.edt(p)
            hit = (e <= 0.006) & ~done
            done |= hit | (s >= RMAX)
            if done.all():
                break
            s = np.where(done, s, s + np.maximum(e - 0.003, 0.006))
        return np.minimum(s, RMAX)

    @staticmethod
    def poly_range(o, ang, poly):
        """(n,) ray-polygon first intersection [m] (inf if none)"""
        d = np.stack([np.cos(ang), np.sin(ang)], 1)                     # (n, 2)
        a = poly; b = np.roll(poly, -1, 0); e = b - a                    # (M, 2)
        ao = a[None] - o[None, None]                                     # (1, M, 2)
        den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        t = (ao[..., 0] * e[None, :, 1] - ao[..., 1] * e[None, :, 0]) / np.where(np.abs(den) > 1e-12, den, np.nan)
        u = (ao[..., 0] * d[:, None, 1] - ao[..., 1] * d[:, None, 0]) / np.where(np.abs(den) > 1e-12, den, np.nan)
        ok = (t > 0) & (u >= 0) & (u <= 1)
        t = np.where(ok, t, np.inf)
        return np.nanmin(t, 1)

    def beams_on(self, pose, polys, j, mount_x):
        """beams whose first return is box j (geometric; walls + other boxes occlude) from car pose (x, y, yaw)"""
        x, y, yaw = pose
        o = np.array([x + mount_x * np.cos(yaw), y + mount_x * np.sin(yaw)])
        P = polys[j]
        rel = np.angle(np.exp(1j * (np.arctan2(P[:, 1] - o[1], P[:, 0] - o[0]) - yaw)))
        dmin = np.linalg.norm(P - o, axis=1).min()
        if dmin > RMAX:
            return 0, dmin
        lo, hi = rel.min(), rel.max()
        if hi - lo > np.pi:                                              # straddles behind: not in a 270 deg front fov anyway
            return 0, dmin
        sel = (ANG >= lo - 1e-9) & (ANG <= hi + 1e-9)
        if not sel.any():
            return 0, dmin
        ang = yaw + ANG[sel]
        rb = self.poly_range(o, ang, P)
        rw = self.wall_range(o, ang)
        ro = np.full(len(ang), np.inf)
        for i, Q in enumerate(polys):
            if i != j:
                ro = np.minimum(ro, self.poly_range(o, ang, Q))
        vis = np.isfinite(rb) & (rb <= RMAX) & (rb < rw) & (rb < ro)
        return int(vis.sum()), dmin


def ray_free(C: Caster, p, n, maxd=2.0):
    """free distance from point p along unit direction n to the first wall cell [m]"""
    s = np.arange(0.0, maxd, 0.0125)
    q = p[None] + s[:, None] * n[None]
    w = C.edt(q) <= 0.0
    return float(s[w.argmax()]) if w.any() else maxd


def new_onsets(R):
    return R.d   # (T, B, 6) onsets wall, box, car, hard wall, hard box, hard car


def nearest_box(R, t, b, xy):
    polys = R.boxes(t, b)
    if not polys:
        return -1, np.inf, polys
    from_ = [float(np.min(np.linalg.norm(p - xy[None], axis=1))) for p in polys]
    body = body_poly(*R.st[t, b, :3].astype(float), R.hl, R.hw)[None]
    cl = [float(signed_clear(body, p)[0]) for p in polys]
    k = int(np.argmin(cl))
    return k, cl[k], polys
