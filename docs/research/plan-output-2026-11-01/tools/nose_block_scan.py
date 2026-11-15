"""(11-16) 앞이 막혀 서는 것이 학습 세계에서 얼마나 자주 생기고, 그때 보상이 후진을 가르치는지 (plan_rl --dump-rollout 묶음, dump_kr26a.sh).

시험(exam1)에서 kr26a u600은 2 s 넘게 선 11번이 모두 벽·상자 0.1-0.3 m 앞에서였고 후진으로 풀린 건 4번, 46 s 끝까지 선 세션도 있었음.
여기서는 학습 rollout에서
  1. 2 s 넘게 선(|v| < 0.3 m/s) 구간: 차 1대 1시간당 몇 번, 학습 시간의 몇 %, 어떻게 풀렸나(후진 / 앞으로 / 재시작 / 묶음 끝)
     -- 시작할 때 앞 0.9 m 안(옆 0.45 m 안)에 상자가 있으면 '상자 앞', 아니면 '그 밖(벽·차)'
  2. 그렇게 서 있는 결정에서 첫 단계 속도 잡음(정규화, 음수 = 후진 쪽)과 advantage의 기울기: 음수면 보상이 후진을 칭찬함,
     0이면 결정 하나의 잡음으로는 빠져나오는 결과에 닿지 못해 배울 신호가 없음
인자: 묶음 폴더 [더 ...]
"""
import glob
import os
import sys

import numpy as np

STILL_V, STALL_S = 0.3, 2.0
BOX_AHEAD, BOX_SIDE = 0.9, 0.45


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "u*.npz")), key=lambda f: int(os.path.basename(f)[1:-4]))
    zs = [np.load(f) for f in fs]
    cat = lambda k: np.concatenate([z[k] for z in zs], 0)
    return len(fs), {k: cat(k) for k in ("pose", "u", "mean", "std", "adv", "done", "boxes")}


for d in sys.argv[1:]:
    n, Z = load(d)
    pose, adv, done = Z["pose"].astype(float), Z["adv"].astype(float), Z["done"].astype(bool)
    T, B = adv.shape
    v = pose[..., 3]
    step = np.hypot(np.diff(pose[..., 0], axis=0), np.diff(pose[..., 1], axis=0))
    mv = np.abs(v[:-1]) > 1.0
    dt = float(np.median(step[mv] / np.abs(v[:-1][mv]))) if mv.any() else 0.1
    k_stall = int(round(STALL_S / dt))
    bx = Z["boxes"].astype(float).reshape(T, B, -1, 4)
    near_box = (bx[..., 0] > 0.5) & (bx[..., 1] * 5 > 0) & (bx[..., 1] * 5 < BOX_AHEAD) & (np.abs(bx[..., 2] * 5) < BOX_SIDE)
    near_box = near_box.any(-1)
    segs = []
    stalled = np.zeros((T, B), bool)
    for b in range(B):
        still = np.abs(v[:, b]) < STILL_V
        t = 0
        while t < T:
            if still[t] and not done[t, b]:
                t0 = t
                while t < T and still[t] and not done[t, b]:
                    t += 1
                if t - t0 >= k_stall:
                    if t >= T:
                        how = "묶음 끝"
                    elif done[t, b]:
                        how = "재시작"
                    else:
                        how = "후진" if v[t, b] < 0 else "앞으로"
                    segs.append((b, t0, t - t0, how, bool(near_box[t0, b])))
                    stalled[t0:t, b] = True
            else:
                t += 1
    hours = T * B * dt / 3600.0
    print(f"== {d}: rollout {n}개, 차 {B}대, 결정 {T}개 x {dt:.3f} s = 차 1대당 {T * dt:.0f} s (모두 {hours:.1f} 차-시간)")
    print(f"  2 s 넘게 섬: {len(segs)}번 = 차-시간당 {len(segs) / hours:.1f}번, 학습 시간의 {100 * stalled.mean():.2f} %")
    for kind, sel in (("상자 앞", True), ("그 밖(벽·차)", False)):
        S = [s for s in segs if s[4] == sel]
        if not S:
            continue
        hows = {h: sum(1 for s in S if s[3] == h) for h in ("후진", "앞으로", "재시작", "묶음 끝")}
        dur = np.array([s[2] for s in S]) * dt
        print(f"  {kind}: {len(S)}번 · 길이 중앙 {np.median(dur):.1f} s (90 % {np.percentile(dur, 90):.1f} s) · 풀린 방식 {hows}")
    # 2. advantage against the first step's speed noise while stalled
    u, mu, sd = Z["u"].astype(float), Z["mean"].astype(float), Z["std"].astype(float)
    # the dump keeps only the dimensions RL samples (`rl_mask`): [0] the first step's steering, [1] its speed
    if u.ndim == 3 and u.shape[-1] == 2:
        z0 = (u[..., 1] - mu[..., 1]) / np.maximum(sd[..., 1], 1e-6)
        for name, zz in (("첫 단계", z0),):
            for kind, m in (("서 있는 결정", stalled), ("달리는 결정(비교)", (~stalled) & (np.abs(v) > 1.0))):
                x, y = zz[m], adv[m]
                if len(x) < 50:
                    continue
                x = x - x.mean()
                sl = (x * (y - y.mean())).sum() / (x * x).sum()
                res = y - y.mean() - sl * x
                se = np.sqrt((res * res).sum() / (len(x) - 2) / (x * x).sum())
                q = [y[zz[m] < -1].mean() if (zz[m] < -1).any() else np.nan, y[np.abs(zz[m]) <= 1].mean(),
                     y[zz[m] > 1].mean() if (zz[m] > 1).any() else np.nan]
                print(f"  속도 잡음({name}) vs advantage, {kind} {len(x)}개: 기울기 {sl:+.4f} ± {se:.4f} /σ "
                      f"| advantage 평균: 후진 쪽(<-1σ) {q[0]:+.4f} · 가운데 {q[1]:+.4f} · 앞쪽(>+1σ) {q[2]:+.4f}")
    else:
        print("  u 모양이 예상과 다름:", u.shape)
