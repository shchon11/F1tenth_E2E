"""기록(MAP 등): plan이 출발하는 방향과 차가 실제로 나아가는 방향의 차이가 차의 옆미끄럼(차체 방향 대비 진행 방향)인가.
옆가속도(v x 회전 속도)별로: 차체 방향 대비 실제 진행 방향, plan 첫 점 방향 대비 실제 진행 방향, 0.3 s 뒤 옆으로 벗어난 거리. 인자: npz"""
import sys
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec
R = Rec(sys.argv[1]); T, B, dt = R.T, R.B, R.dt
st = R.st.astype(float); xy = st[:, :, :2]; yaw = st[:, :, 2]; V = np.abs(st[:, :, 3]); pl = R.pl.astype(float)
any_h = (R.d[:, :, 3:6] > 0).any(-1)
recent = np.zeros((T, B), bool); soon = np.zeros((T, B), bool)
for k in range(1, 31):
    recent[k:] |= any_h[:-k]
for k in range(1, 11):
    soon[:-k] |= any_h[k:]
ok = (V[:T] > 2.0) & (st[:T, :, 3] > 0) & ~recent & ~any_h & ~soon; ok[T - 10:] = False
t_, b_ = np.nonzero(ok); sel = np.random.default_rng(0).choice(len(t_), min(80000, len(t_)), replace=False); t_, b_ = t_[sel], b_[sel]
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
ang = lambda d: np.arctan2(d[:, 1], d[:, 0])
p0, p1 = xy[t_, b_], xy[t_ + 1, b_]
y0, y1 = yaw[t_, b_], yaw[t_ + 1, b_]
r = wrap(y1 - y0) / dt; v = V[t_, b_]; alat = v * r; sg = np.sign(r)
beta_real = wrap(ang(p1 - p0) - (y0 + 0.5 * wrap(y1 - y0)))                 # direction of travel relative to the body's heading (mean over the step)
d_plan = wrap(ang(pl[t_, b_, 0] - p0) - ang(p1 - p0))                       # the plan's first chord relative to the actual chord
# the plan's own heading change over its first step, from its first two chords (kinematic: chord k+1 - chord k = r dt)
r_plan = wrap(ang(pl[t_, b_, 1] - pl[t_, b_, 0]) - ang(pl[t_, b_, 0] - p0)) / dt
path = np.concatenate([p0[:, None], pl[t_, b_]], 1)
a, b = path[:, :-1], path[:, 1:]; ab = b - a
p3 = xy[t_ + 3, b_]
u = np.clip(((p3[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
off = p3[:, None] - (a + u[..., None] * ab); dist = np.linalg.norm(off, axis=-1); j = dist.argmin(1); n = np.arange(len(p3))
lat3 = np.sign(ab[n, j, 0] * off[n, j, 1] - ab[n, j, 1] * off[n, j, 0]) * dist[n, j]
lr = 0.5 * 0.33                                                              # ~CoG to rear axle (only for the kinematic reference line)
print(f"{str(R.z['mode'])} · 보통 주행 {len(t_)}개 (부호: 도는 쪽 = +)")
print(" 옆가속도(m/s²)      개수 | 속도 | 차체 방향 대비 실제 진행 방향 | plan 첫 구간 방향 − 실제 진행 방향 | plan의 회전 속도 − 실제 | 0.3 s 뒤 plan의 길에서 (도는 쪽 +)")
for lo, hi in ((0, 1), (1, 2.5), (2.5, 4), (4, 5.5), (5.5, 7), (7, 20)):
    m = (np.abs(alat) >= lo) & (np.abs(alat) < hi)
    if m.sum() < 100:
        continue
    f = lambda x: f"{np.degrees(np.median(x[m] * sg[m])):+5.1f}°"
    print(f"  {lo:>4}~{hi:<4} {m.sum():9d} | {np.median(v[m]):.1f} | {f(beta_real)} (미끄럼 없는 차라면 {np.degrees(np.median(lr * np.abs(r[m]) / v[m])):+.1f}°) | "
          f"{f(d_plan)} (10~90 %: {np.degrees(np.percentile(d_plan[m] * sg[m], 10)):+.1f} ~ {np.degrees(np.percentile(d_plan[m] * sg[m], 90)):+.1f}) | "
          f"{np.median((r_plan[m] - r[m]) * sg[m]):+.2f} rad/s (실제 {np.median(np.abs(r[m])):.2f}) | {100 * np.median(lat3[m] * sg[m]):+5.1f} cm")
m = np.abs(alat) > 2.5
pred = v * 0.3 * np.sin(d_plan)                                              # drift if the direction offset stayed for 0.3 s
c = np.corrcoef(pred[m] * sg[m], -lat3[m] * sg[m])[0, 1]
print(f"옆가속도 2.5 이상에서: 방향 차이가 0.3 s 유지될 때의 벗어남(중앙 {100 * np.median(pred[m] * sg[m]):.1f} cm) 과 실제 벗어남(중앙 {-100 * np.median(lat3[m] * sg[m]):.1f} cm)의 상관 {c:.2f}")
