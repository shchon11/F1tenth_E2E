"""(11-14) exam_reward.py 기록들: 세션마다 학습 보상의 합(90 s, 할인 없이)과 그 조각, 세게 닿음 수, 달린 거리. 첫 기록을 기준으로 세션끼리 짝지은 차이.
보상 단위 = 바퀴(T_REF 7.5 s = 1바퀴): 진행(바퀴 + 진행 보상) - 접촉 값. plan 충돌 값은 plan26 학습들에서 꺼져 있어(--plan-hit-share 0) 뺌
(exam의 보상 계산은 기본값 0.1로 셈). 표에서는 x 7.5 해서 초로 보임.
인자: 이름=npz [...] (첫째가 기준)"""
import sys
import numpy as np
from rec_lib import Rec
T_REF = 7.5
rows = []
for a in sys.argv[1:]:
    nm, fn = a.split("=", 1)
    R = Rec(fn); z = R.z; rw = z["rw"].astype(float)           # (T, B, 5): lap, shaping, contact, plan_hit, reward
    S = rw.sum(0) * T_REF
    hard = (R.d[:, :, 3:6] > 0).sum(0)                          # (B, 3)
    rem = z["rem"].astype(float); L = float(z["lapL"][0])
    dist = np.where(np.abs(np.diff(rem, axis=0)) > 0.5 * L, 0.0, -np.diff(rem, axis=0)).sum(0)
    rows.append((nm, S, hard, dist))
nm0, S0, h0, d0 = rows[0]
print(f"{'':<14}{'학습 보상 합 [s]':>16}{'진행 [s]':>16}{'접촉 값 [s]':>16}{'세게 닿음/세션':>16}{'벽·상자·차':>18}{'거리 [m]':>14}")
for nm, S, h, d in rows:
    def f(x, x0):
        if nm == nm0:
            return f"{x.mean():.2f}"
        dd = x - x0
        return f"{dd.mean():+.2f}±{dd.std(ddof=1) / np.sqrt(len(dd)):.2f}"
    prog = S[:, 0] + S[:, 1]; prog0 = S0[:, 0] + S0[:, 1]
    ht = h.sum(1).astype(float); h0t = h0.sum(1).astype(float)
    tot, tot0 = prog + S[:, 2], prog0 + S0[:, 2]
    print(f"{nm:<14}{f(tot, tot0):>16}{f(prog, prog0):>16}{f(S[:, 2], S0[:, 2]):>16}{f(ht, h0t):>16}"
          f"{'   ' + ' · '.join(f'{x:.2f}' for x in h.mean(0)):>20}{f(d, d0):>14}")
