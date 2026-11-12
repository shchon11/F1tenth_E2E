"""세게 닿는 상자 앞에서 plan이 상자의 어느 쪽으로 지나갈지 정하지 못하는가 (11-13). 가설: 왼쪽·오른쪽 두 갈래를 정규분포 하나로 내는
policy가 둘을 평균해 상자 정면을 계획하다가 늦게야 한쪽으로 정함(box_react.py: 닿는 경우 1.5 m 안에서야 크게 비킴, 어제 box_side.py:
세게 닿은 상자는 지난 바퀴와 다른 쪽으로 지나간 것).
기록(exam_record.py)의 실제 plan(10점)으로, 상자에 다가가는 마지막 2.5 s 동안 결정마다 plan이 상자 위치에서 상자 중심의 어느 쪽으로
지나가는지(차 좌표 옆 차, 0.05 m 안은 셈하지 않음):
  - 쪽이 바뀐 횟수, 마지막 쪽이 정해진 때 상자까지 앞 거리, 1.5 m 밖에서 plan이 상자 중심 0.3 m 안을 지나간 결정의 비율.
  견줌: 세게 닿음 vs 정면 상자(3 m 앞에서 옆 0.3 m 안)를 3 s 안에 안 닿고 지나감.
두 plan policy의 기록(br, pl_other, 11-13)이면 더: 1.5 m 밖에서 두 plan이 상자의 서로 다른 쪽을 지나가는 결정의 비율, 고른 plan이 바뀐 횟수.
상자 위치는 분석에서만 씀. 인자: 기록npz [기록npz ...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec

for fn in sys.argv[1:]:
    R = Rec(fn); z = R.z; T = R.T; st = R.st.astype(float); pl = R.pl.astype(float)
    two = "pl_other" in z.files
    plo = z["pl_other"].astype(float) if two else None; brr = z["br"] if two else None
    k_rm = min(int(round(float(z["rm_t"]) / R.dt)), T); box_w = (R.sid[0] >= 0).any(-1)
    hard = R.d[:T, :, 4] > 0; touch = R.d[:T, :, 1] > 0

    def rel(t, b):
        """(ahead, side of the nearest box in front, the plan's lateral offset from the box centre at the box)"""
        x, y, yaw = st[t, b, :3]; c, s = np.cos(yaw), np.sin(yaw); best = None
        for poly in R.boxes(t, b):
            dx, dy = poly.mean(0) - [x, y]; ax, ay = dx * c + dy * s, -dx * s + dy * c
            if 0.3 < ax < 5.0 and abs(ay) < 1.5 and (best is None or ax < best[0]):
                best = (ax, ay)
        if best is None:
            return None
        q = pl[t, b] - [x, y]; px, py = q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c
        if px.max() < best[0]:
            return None
        oo, bb = np.nan, -1
        if two:
            q = plo[t, b] - [x, y]; qx, qy = q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c
            if qx.max() >= best[0]:
                oo = float(np.interp(best[0], qx, qy)) - best[1]
            bb = int(brr[t, b])
        return best[0], best[1], float(np.interp(best[0], px, py)) - best[1], oo, bb

    def window(b, t_end):
        return [(t,) + r for t in range(max(t_end - 25, 0), t_end) for r in [rel(t, b)] if r is not None]

    ev = {"세게 닿음": [], "안 닿음(정면 상자)": []}
    for b in np.nonzero(box_w)[0]:
        for t in np.nonzero(hard[:k_rm, b])[0]:
            w = window(b, t)
            if len(w) >= 8:
                ev["세게 닿음"].append(w)
        t = 3
        while t < k_rm - 30:
            r = rel(t, b)
            if r is not None and 2.5 < r[0] < 3.2 and abs(r[1]) < 0.3 and not touch[t:t + 30, b].any():
                te = t
                while te < k_rm - 1:
                    r2 = rel(te, b)
                    if r2 is None or r2[0] < 0.8:
                        break
                    te += 1
                w = window(b, te)
                if len(w) >= 8:
                    ev["안 닿음(정면 상자)"].append(w)
                t = te + 20
            else:
                t += 1
    print(f"== {os.path.basename(fn)}", flush=True)
    for k, L in ev.items():
        if not L:
            print(f"  [{k}] 없음", flush=True); continue
        flips, late, ay0, cen, split, bflip = [], [], [], [], [], []
        for w in L:
            off = np.array([x[3] for x in w]); ax = np.array([x[1] for x in w]); ay0.append(abs(w[0][2]))
            sgn = np.sign(off[np.abs(off) > 0.05]); flips.append(int((np.diff(sgn) != 0).sum()) if len(sgn) > 1 else 0)
            fin = np.sign(off[-1]); same = (np.sign(off) == fin) & (np.abs(off) > 0.05)
            i = len(off) - 1
            while i >= 0 and same[i]:
                i -= 1
            late.append(float(ax[i + 1]) if i + 1 < len(ax) else np.nan)
            cen.append(float(np.mean(np.abs(off[ax > 1.5]) < 0.3)) if (ax > 1.5).any() else np.nan)
            if two:
                oo = np.array([x[4] for x in w]); bb = np.array([x[5] for x in w])
                m = (ax > 1.5) & np.isfinite(oo) & (np.abs(off) > 0.05) & (np.abs(oo) > 0.05)
                split.append(float(np.mean(np.sign(off[m]) != np.sign(oo[m]))) if m.any() else np.nan)
                bflip.append(int((np.diff(bb) != 0).sum()))
        flips = np.array(flips); late = np.array(late)
        print(f"  [{k}] {len(L)}건 | 처음 볼 때 상자의 옆 위치(차 정면에서) 중앙 {np.median(ay0):.2f} m | plan이 지나갈 쪽이 바뀐 횟수: 0번 "
              f"{np.mean(flips == 0) * 100:.0f} % · 1번 {np.mean(flips == 1) * 100:.0f} % · 2번 이상 {np.mean(flips >= 2) * 100:.0f} % | "
              f"마지막 쪽이 정해진 때 상자까지 중앙 {np.nanmedian(late):.2f} m | 1.5 m 밖에서 plan이 상자 중심 0.3 m 안을 지나는 결정 "
              f"{np.nanmean(cen) * 100:.0f} %", flush=True)
        if two:
            bflip = np.array(bflip)
            print(f"      두 plan: 1.5 m 밖에서 두 plan이 상자의 서로 다른 쪽을 지나가는 결정 {np.nanmean(split) * 100:.0f} % | "
                  f"고른 plan이 바뀐 횟수: 0번 {np.mean(bflip == 0) * 100:.0f} % · 1번 {np.mean(bflip == 1) * 100:.0f} % · "
                  f"2번 이상 {np.mean(bflip >= 2) * 100:.0f} %", flush=True)
