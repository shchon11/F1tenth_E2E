"""MAP 기록: 실제 차가 plan의 길에서 벗어난 쪽이 굽은 길의 안쪽인가 바깥쪽인가(굽은 정도별), 벽 끝에서 세게 닿은 순간에는 어느 쪽이었나. 인자: npz"""
import sys
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec
R = Rec(sys.argv[1]); T, B, dt = R.T, R.B, R.dt
st = R.st.astype(float); xy = st[:, :, :2]; V = np.abs(st[:, :, 3]); pl = R.pl.astype(float)
hard = R.d[:, :, 3:6] > 0; any_h = hard.any(-1)
n10, n30 = 10, 30
recent = np.zeros((T, B), bool); soon = np.zeros((T, B), bool)
for k in range(1, n30 + 1):
    recent[k:] |= any_h[:-k]
for k in range(1, n10 + 1):
    soon[:-k] |= any_h[k:]


def signed(p, path):
    """signed distance of p from the polyline (left of the direction of travel = +)"""
    a, b = path[:, :-1], path[:, 1:]; ab = b - a
    u = np.clip(((p[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    off = p[:, None] - (a + u[..., None] * ab); dist = np.linalg.norm(off, axis=-1); j = dist.argmin(1); n = np.arange(len(p))
    cr = ab[n, j, 0] * off[n, j, 1] - ab[n, j, 1] * off[n, j, 0]
    return np.sign(cr) * dist[n, j]


def curv(path, i=3, j=6):
    """signed curvature of the circle through the car, plan point i and plan point j (left turn = +)"""
    a, b, c = path[:, 0], path[:, i], path[:, j]
    cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    la, lb, lc = np.linalg.norm(b - a, axis=-1), np.linalg.norm(c - b, axis=-1), np.linalg.norm(c - a, axis=-1)
    return 2 * cr / np.maximum(la * lb * lc, 1e-6)


ok = (V[:T] > 2.0) & ~recent & ~any_h & ~soon; ok[T - n10:] = False
t_, b_ = np.nonzero(ok); sel = np.random.default_rng(0).choice(len(t_), min(60000, len(t_)), replace=False); t_, b_ = t_[sel], b_[sel]
path = np.concatenate([xy[t_, b_][:, None], pl[t_, b_]], 1)
k = curv(path); v0 = V[t_, b_]
print(f"보통 주행 결정 {len(t_)}개 · {str(R.z['mode'])}")
for i in (2, 3, 5):
    s = signed(xy[t_ + i, b_], path) * np.sign(k)                   # + = inside of the plan's turn
    print(f" {i * 0.1:.1f} s 뒤, plan의 길에서 굽은 안쪽으로(+) 벗어난 거리:")
    for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 9)):
        m = (np.abs(k) >= lo) & (np.abs(k) < hi)
        if m.sum() < 50:
            continue
        rad = "거의 곧음" if hi <= 0.1 else f"반경 {1 / hi:.1f}~{1 / lo:.1f} m" if hi < 9 else f"반경 {1 / lo:.1f} m보다 급함"
        L1 = 0.15 + 0.3 * np.median(v0[m]); kk = np.median(np.abs(k[m]))
        geo = (1 / kk - np.sqrt(max(1 / kk ** 2 - (L1 / 2) ** 2, 0))) if kk > 1e-3 else 0.0
        print(f"   {rad:<18} {m.sum():6d}개 | 속도 중앙 {np.median(v0[m]):.1f} m/s | 중앙 {100 * np.median(s[m]):+5.1f} cm (10~90 %: {100 * np.percentile(s[m], 10):+5.1f} ~ {100 * np.percentile(s[m], 90):+5.1f}) | "
              f"안쪽 {100 * (s[m] > 0.02).mean():3.0f} % · 바깥쪽 {100 * (s[m] < -0.02).mean():3.0f} % | 앞을 보고 가는 방식이 기하학적으로 자르는 폭 {100 * geo:.0f} cm")
# hard wall hits while driving at the wall end (cell around (-1.5, -1.5))
hw = hard[:, :, 0] & ~recent
hw[:n10] = False
th, bh = np.nonzero(hw)
m = (np.abs(xy[th, bh, 0] + 1.5) < 0.5) & (np.abs(xy[th, bh, 1] + 1.5) < 0.5) & (V[np.maximum(th - n10, 0), bh] > 2.0)
th, bh = th[m], bh[m]
print(f"벽 끝 칸(-1.5,-1.5)에서 달리다 세게 닿음 {len(th)}번")
if len(th):
    for back in (3, 5):
        path = np.concatenate([xy[th - back, bh][:, None], pl[th - back, bh]], 1); k = curv(path)
        s = signed(xy[th, bh], path) * np.sign(k)
        pw = R.wall_dist(0, path[:, 1:back + 2].reshape(-1, 2)).reshape(len(th), -1).min(1)   # the plan's own closest wall distance up to the hit time (+0.1 s)
        print(f"  {back * 0.1:.1f} s 전 plan 기준: 닿은 순간 차는 plan의 굽은 안쪽으로 중앙 {100 * np.median(s):+5.1f} cm (10~90 %: {100 * np.percentile(s, 10):+5.1f} ~ {100 * np.percentile(s, 90):+5.1f}), "
              f"안쪽 {100 * (s > 0.02).mean():.0f} % · 바깥쪽 {100 * (s < -0.02).mean():.0f} % | 그 plan의 점들이 벽에서 가장 가까운 거리 중앙 {100 * np.median(pw):.0f} cm "
              f"(차 반폭 {100 * R.hw:.0f} cm보다 가까움 {100 * (pw < R.hw).mean():.0f} %, 25 cm보다 가까움 {100 * (pw < 0.25).mean():.0f} %) | plan의 굽은 정도 중앙 반경 {1 / max(np.median(np.abs(k)), 1e-3):.1f} m")
    # the same for passes of the same place without any hit: plan clearance and side
    okp = ok & (np.abs(xy[:T, :, 0] + 1.5) < 0.5) & (np.abs(xy[:T, :, 1] + 1.5) < 0.5)
    okp[:5] = False
    tp, bp = np.nonzero(okp)
    if len(tp):
        path = np.concatenate([xy[tp - 3, bp][:, None], pl[tp - 3, bp]], 1); k = curv(path)
        s = signed(xy[tp, bp], path) * np.sign(k)
        pw = R.wall_dist(0, path[:, 1:5].reshape(-1, 2)).reshape(len(tp), -1).min(1)
        print(f"  같은 칸을 안 닿고 지난 순간 {len(tp)}개 (0.3 s 전 plan 기준): 안쪽으로 중앙 {100 * np.median(s):+5.1f} cm (10~90 %: {100 * np.percentile(s, 10):+5.1f} ~ {100 * np.percentile(s, 90):+5.1f}) | "
              f"plan의 점들이 벽에서 가장 가까운 거리 중앙 {100 * np.median(pw):.0f} cm (25 cm보다 가까움 {100 * (pw < 0.25).mean():.0f} %)")
