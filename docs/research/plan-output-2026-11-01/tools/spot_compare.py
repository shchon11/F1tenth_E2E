"""달리다 벽에 세게 닿은 건(exam_spots.py와 같은 정의: 앞 3 s에 세게 닿은 적 없고 1 s 전 2 m/s 넘음)을 자리별·세션당으로, 기록 여러 개 비교.
인자: 기록npz ..."""
import sys
import numpy as np
sys.path.insert(0, '/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools')
from rec_lib import Rec
rng = np.random.default_rng(0)
REG = (("짧은 벽 끝 둘레", (-2, 0, -2, 0)), ("섬 왼쪽 위 끝 둘레", (-4, -2, 4, 6)), ("오른쪽 위 바깥 벽(굽은 길을 나오며 가속하는 곳)", (4, 9, 0, 3)))
def ci(x):
    m = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return f"{x.mean():.2f} [{np.percentile(m, 2.5):.2f}-{np.percentile(m, 97.5):.2f}]"
for f in sys.argv[1:]:
    R = Rec(f); z = R.z; T, B, dt = R.T, R.B, R.dt
    st = R.st.astype(float); xy = st[:, :, :2]; V = np.abs(st[:, :, 3])
    opp = R.on[:T].any(0)
    hard = R.d[:, :, 3:6] > 0; any_h = hard.any(-1)
    n10, n30 = int(round(1.0 / dt)), int(round(3.0 / dt))
    recent = np.zeros((T, B), bool)
    for k in range(1, n30 + 1):
        recent[k:] |= any_h[:-k]
    fresh = np.zeros((T, B), bool); fresh[n10:] = (V[:T][:-n10] > 2.0)[: T - n10]
    fresh &= ~recent
    ev = hard[:, :, 0] & fresh                                   # (T, B)
    p = xy[:T]
    print(f"{z['exam']} · {z['mode']} · 세션 {B}개 ({f.split('/')[-1]})")
    rest = ev.copy()
    for name, (x0, x1, y0, y1) in REG:
        m = ev & (p[..., 0] >= x0) & (p[..., 0] < x1) & (p[..., 1] >= y0) & (p[..., 1] < y1)
        rest &= ~m
        n = m.sum(0).astype(float)
        print(f"   {name}: {int(n.sum())}건 | 세션당 전체 {ci(n)} · 혼자 {ci(n[~opp])} · 상대 차 {ci(n[opp])}")
    n = rest.sum(0).astype(float); a = ev.sum(0).astype(float)
    print(f"   그 밖: {int(n.sum())}건 | 세션당 전체 {ci(n)} · 혼자 {ci(n[~opp])} · 상대 차 {ci(n[opp])}")
    print(f"   모두: {int(a.sum())}건 | 세션당 전체 {ci(a)} · 혼자 {ci(a[~opp])} · 상대 차 {ci(a[opp])}")
