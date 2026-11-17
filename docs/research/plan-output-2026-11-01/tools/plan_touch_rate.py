"""(11-18) plan 벌점(--plan-hit-share)을 켜면 얼마나 무거울까 -- 시험 기록(평균 plan, 잡음 없음)에서 plan이 벽·상자에 걸리는 비율과
그때 매겨질 값을, 같은 결정의 진행(바퀴)과 비교. 학습 중 '박는 plan'(잡음 섞인 plan) 비율과 견줘 보려는 것.

plan_reward.plan_cost와 같은 판정: plan의 자세마다(지금 차 자리 제외) 차 네 모서리 중 하나가 벽(거리장 0) 또는 상자 안이면 걸림.
기록에는 plan 10점(0.1 s 간격)만 있어서, 점 사이를 SUB 등분해 잇고 방향은 이웃 점으로 잡음(근사). 값 = share x contact_scale x
(0.5 + 걸린 자리의 plan 속도) / 7.5 바퀴. 진행 = 그 결정 동안 트랙을 따라 간 거리 / 한 바퀴.
인자: 기록npz [기록npz ...]   (SHARE=0.1, SCALE=0.82)
"""
import os
import sys

import numpy as np

SHARE = float(os.environ.get("SHARE", 0.1))
SCALE = float(os.environ.get("SCALE", 0.82))
SUB = 4
T_REF, C0, C1 = 7.5, 0.5, 1.0

for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=True)
    st, pl, rs = z["st"].astype(float), z["pl"].astype(float), z["rs"]
    T, B = rs.shape
    dt = float(z["dt"])
    L = float(z["lapL"][0])
    H, W, res = int(z["t_H"][0]), int(z["t_W"][0]), float(z["t_res"][0])
    ox, oy = (float(v) for v in z["t_origin"][0])
    edt = z["edt"][int(z["t_off"][0]): int(z["t_off"][0]) + H * W].reshape(H, W)
    ln, wd = (float(v) for v in z["car"])
    corners = np.array([[ln / 2, wd / 2], [ln / 2, -wd / 2], [-ln / 2, wd / 2], [-ln / 2, -wd / 2]])
    fps = {int(k[3:]): z[k].astype(float) for k in z.files if k.startswith("fp_")}
    sid, pose = z["sid"], z["pose"].astype(float)
    rm = int(round(float(z["rm_t"]) / dt)) if "rm_t" in z.files else T
    # plan poses: the car now, then the 10 points, SUB pieces between
    p0 = st[:T, :, None, :2]
    P = np.concatenate([p0, pl], 2)                                       # (T, B, 11, 2)
    f = np.linspace(0, 1, SUB, endpoint=False)[1:]
    seg = P[:, :, :-1, None, :] + (P[:, :, 1:, None, :] - P[:, :, :-1, None, :]) * np.concatenate([[0.0], f])[None, None, None, :, None]
    Q = np.concatenate([seg.reshape(T, B, -1, 2), P[:, :, -1:, :]], 2)[:, :, 1:]   # without the car now
    d = np.diff(np.concatenate([P[:, :, :1], Q], 2), axis=2)
    psi = np.arctan2(d[..., 1], d[..., 0])
    vq = np.repeat(np.linalg.norm(np.diff(P, axis=2), axis=-1) / 0.1, SUB, axis=2)[:, :, :Q.shape[2]]
    c, s = np.cos(psi)[..., None], np.sin(psi)[..., None]
    cx = Q[..., 0, None] + corners[:, 0] * c - corners[:, 1] * s
    cy = Q[..., 1, None] + corners[:, 0] * s + corners[:, 1] * c
    ix = np.clip(((cx - ox) / res).astype(int), 0, W - 1); iy = np.clip(((cy - oy) / res).astype(int), 0, H - 1)
    inb = (cx >= ox) & (cy >= oy) & (cx < ox + W * res) & (cy < oy + H * res)
    wall = ((edt[iy, ix] <= 0.0) | ~inb).any(-1)                          # (T, B, n)
    box = np.zeros_like(wall)
    for b in range(B):
        for j in range(sid.shape[2]):
            k = int(sid[0, b, j])
            if k < 0 or k not in fps:
                continue
            lo, hi = fps[k].min(0), fps[k].max(0)
            bx, by, bth = pose[0, b, j]
            dx, dy = cx[:rm, b] - bx, cy[:rm, b] - by
            lx = np.cos(bth) * dx + np.sin(bth) * dy; ly = -np.sin(bth) * dx + np.cos(bth) * dy
            box[:rm, b] |= ((lx > lo[0]) & (lx < hi[0]) & (ly > lo[1]) & (ly < hi[1])).any(-1)
    hit = wall | box
    first = np.argmax(hit, 2)
    any_ = hit.any(2) & ~rs
    v_hit = np.take_along_axis(vq, first[..., None], 2)[..., 0]
    price = np.where(any_, SHARE * SCALE * (C0 + C1 * v_hit) / T_REF, 0.0)
    rem = z["rem"].astype(float)
    prog = np.mod(rem[:-1] - rem[1:] + 0.5 * L, L) - 0.5 * L
    prog = np.where(np.isfinite(prog) & ~rs, prog, 0.0) / L
    on = z["on"][:T].any(0); has = (sid[0] >= 0).any(-1)
    print(f"== {fn.split('/')[-1]} | share {SHARE} x {SCALE} | 차 {ln:.2f} x {wd:.2f} m")
    for nm, m in (("빈 트랙", ~on & ~has), ("상자 세션 (치우기 전)", ~on & has)):
        tm = np.zeros((T, B), bool); tm[:, m] = True
        if nm.startswith("상자"):
            tm[rm:] = False
        tm &= ~rs
        n = tm.sum()
        wf, bf = (wall.any(2) & tm).sum() / n, (box.any(2) & tm).sum() / n
        pp, pg = price[tm].mean(), prog[tm].mean()
        print(f"  {nm:<18} 결정 {n:>7} | plan이 벽에 걸림 {100 * wf:5.1f} % · 상자 {100 * bf:5.1f} % | 벌점 평균 {pp:.5f} 바퀴/결정"
              f" = 진행({pg:.5f})의 {100 * pp / max(pg, 1e-9):.0f} %")
        # where along the plan the first touch is (s ahead): the far end is what the car does not drive (the next plans differ)
        for kn, kk in (("벽", wall), ("상자", box)):
            km = kk.any(2) & tm
            if km.any():
                t_first = (np.argmax(kk, 2)[km] + 1) * 0.1 / SUB
                print(f"    {kn}에 처음 걸린 plan 자리: 중앙 {np.median(t_first):.2f} s 앞 | 0.5 s 안 {100 * (t_first <= 0.5).mean():.0f} %"
                      f" | 0.3 s 안 {100 * (t_first <= 0.3).mean():.0f} %")
