"""(11-25, plan_rl --pred-smooth) 상대 차 예측이 지그재그인가: exam4 기록(F1SIM_EXAM_PRED_PROBE=1 또는 ORACLE, `출력_pred.npz`가 같이 있음)에서
'있음' 결정마다 머리의 10점과 실제로 간 길(기록에서 나중에 본 상대 차 자리, plan_rl 타깃 규칙)을 비교.
꺾임 = 이웃한 두 구간 사이 방향 바뀜 |각도|의 평균(예측) - 같은 것(실제), 실제 차가 두 구간 다 5 cm 넘게 움직인 곳만 -- 학습 로그의
pred_zigzag와 같은 정의(plan_policy.pred_zigzag).
인자: 기록.npz [기록2.npz ...]   (각각 옆에 기록_pred.npz)"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.environ.get("F1SIM_SRC", "/home/shchon11/F1tenth_E2E/f1sim"))
from f1sim.learn.plan_policy import pred_zigzag


def realised(z):
    """(T, B, 10, 2) where the other car really was 0.1 .. 1.0 s after each decision in our frame at it, (T, B, 10) counts
    (pred_check.realised: no restart / 2 m jump in between, within 10 m)"""
    st, op = z["st"].astype(float), z["op"].astype(float)
    rs, ors = z["rs"].astype(bool), z["ors"].astype(bool)
    T, B = rs.shape
    jmp = np.linalg.norm(op[1:T + 1, :, :2] - op[:T, :, :2], axis=-1) > 2.0
    cb = np.cumsum(rs | ors | jmp, 0)
    tgt = np.full((T, B, 10, 2), np.nan); ok = np.zeros((T, B, 10), bool)
    for k in range(1, 11):
        n = min(T + 1 - k, T)
        d = op[k:k + n, :, :2] - st[:n, :, :2]
        c, s = np.cos(st[:n, :, 2]), np.sin(st[:n, :, 2])
        rel = np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], -1)
        before = np.concatenate([np.zeros((1, B)), cb[:n - 1]], 0)
        ok[:n, :, k - 1] = (cb[k - 1:k - 1 + n] - before == 0) & (np.linalg.norm(rel, axis=-1) <= 10.0)
        tgt[:n, :, k - 1] = rel
    return tgt, ok


for fn in sys.argv[1:]:
    z = np.load(fn); pp = np.load(fn[:-4] + "_pred.npz")
    tgt, ok = realised(z)
    sel = pp["there"].astype(bool) & ok.all(-1)
    H, R = pp["head"][sel].astype(np.float64), tgt[sel]
    e = np.linalg.norm(H - R, axis=-1)
    zz = pred_zigzag(torch.from_numpy(H), torch.from_numpy(R)).numpy()
    print(f"{os.path.basename(fn)}: '있음' + 실제 1 s 다 아는 결정 {int(sel.sum())}개 | 오차 0.5 s 중앙 {np.median(e[:, 4]):.2f} m, 1.0 s 중앙 "
          f"{np.median(e[:, 9]):.2f} · p90 {np.percentile(e[:, 9], 90):.2f} m | 꺾임 (예측 - 실제) 평균 {zz.mean():+.1f}° · 중앙 "
          f"{np.median(zz):+.1f}° · p90 {np.percentile(zz, 90):.1f}° /0.1 s")
