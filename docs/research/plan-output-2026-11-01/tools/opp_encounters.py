"""학습 세계에서 우리 차가 겪는 상대 차: 가까이(앞뒤 5 m 안) 있을 때 상대가 우리보다 빠른가, 누가 누구를 지나가나.
drive_record.py 기록에서 상대가 있는 경주만. '지나감' = 3 m 안에서 상대가 우리 차 좌표로 앞(+0.5 m)에서 뒤(-0.5 m)로 넘어감
(우리가 추월) 또는 뒤에서 앞으로(상대가 추월). 속도 비 = 상대 속도 / 우리 차 속도(둘 다 그 순간).
인자: 기록 npz [...]"""
import sys
import numpy as np

for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=False); dt = float(z["dt"]); T = z["cnt"].shape[0]
    st = z["st"][:T].astype(float); op = z["op"][:T].astype(float); on = z["on"][:T]
    d = op[..., :2] - st[..., :2]
    c, s = np.cos(st[..., 2]), np.sin(st[..., 2])
    dx, dy = d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c
    dist = np.hypot(dx, dy)
    jump = np.zeros_like(on)
    jump[1:] = (np.linalg.norm(st[1:, :, :2] - st[:-1, :, :2], axis=2) > 2.0) | (np.linalg.norm(op[1:, :, :2] - op[:-1, :, :2], axis=2) > 2.0)
    mins = on.sum() * dt / 60
    near = on & (dist < 5.0)
    vo, vs = np.abs(op[..., 3]), np.abs(st[..., 3])
    print(f"== {fn.split('/')[-1]}: 상대 있는 경주 {mins:.0f}분, 5 m 안에 상대가 있는 시간 {100 * near.sum() / on.sum():.0f} %")
    for name, m in (("앞 5 m 안(우리가 뒤)", near & (dx > 0.3)), ("뒤 5 m 안(우리가 앞)", near & (dx < -0.3))):
        if m.sum() == 0:
            continue
        r = vo[m] / np.maximum(vs[m], 0.5)
        print(f"  {name}: 경주 시간의 {100 * m.sum() / on.sum():4.1f} % | 상대 속도 평균 {vo[m].mean():.2f} · 우리 {vs[m].mean():.2f} m/s | "
              f"상대가 우리보다 빠른 시간 {100 * (vo[m] > vs[m]).mean():.0f} % | 상대 속도 < 우리의 70 %인 시간 {100 * (r < 0.7).mean():.0f} % | 상대가 2 m/s 아래 {100 * (vo[m] < 2.0).mean():.0f} %")
    # passes
    we = they = 0
    for b in range(on.shape[1]):
        side = 0
        for t in range(T):
            if not on[t, b] or jump[t, b] or dist[t, b] > 3.0:
                side = 0; continue
            s_now = 1 if dx[t, b] > 0.5 else (-1 if dx[t, b] < -0.5 else 0)
            if s_now == 0:
                continue
            if side == 1 and s_now == -1: we += 1
            if side == -1 and s_now == 1: they += 1
            side = s_now
    print(f"  지나감: 우리가 상대를 {we}번 ({we / mins:.2f}/분), 상대가 우리를 {they}번 ({they / mins:.2f}/분)")
