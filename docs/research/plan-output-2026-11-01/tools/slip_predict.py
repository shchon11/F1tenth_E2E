"""미끄럼 각(차체 방향 대비 실제 진행 방향 = atan2(vy, vx))을 차가 스스로 아는 값(속도, 회전 속도)만으로 얼마나 맞힐 수 있나.
기록: F1SIM_EXAM_SLIP_FRAME 기록(vy 저장). 절반 세션으로 표(옆가속도 x 속도 칸의 중앙값)를 만들고 나머지 절반에서 오차를 잼.
인자: 기록npz"""
import sys
import numpy as np
sys.path.insert(0, '/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools')
from rec_lib import Rec
R = Rec(sys.argv[1]); z = R.z; T, B, dt = R.T, R.B, R.dt
st = R.st.astype(float)
vx = st[:T, :, 3]; yaw = st[:, :, 2]
vy = np.asarray(z["vy"], float)                                  # (T, B) at the decision instant
r = np.angle(np.exp(1j * (yaw[1:T + 1] - yaw[:T]))) / dt         # mean yaw rate over the NEXT 0.1 s
r_prev = np.zeros_like(r); r_prev[1:] = r[:-1]                   # over the 0.1 s BEFORE the decision (what a gyro has seen)
beta = np.degrees(np.arctan2(vy, np.maximum(vx, 1e-3)))
hard = (R.d[:, :, 3:6] > 0).any(-1)
rec = np.zeros((T, B), bool)
for k in range(0, 21):
    rec[k:] |= hard[: T - k]
ok = (vx > 2.0) & ~rec; ok[:20] = False
ay = vx * r_prev
half = np.arange(B) % 2 == 0
print(f"{z['exam']} · {z['mode']} · 결정 {int(ok.sum())}개(2 m/s 넘고 앞 2 s에 세게 닿지 않음), 미끄럼 각 = 차체 방향 대비 실제 진행 방향")
sgn = np.sign(ay); sgn[sgn == 0] = 1
b_in = beta * sgn                                                 # + = toward the inside of the turn
print(f"  미끄럼 각(도는 쪽 +): 중앙 {np.median(b_in[ok]):+.1f}°, 10~90 % {np.percentile(b_in[ok], 10):+.1f} ~ {np.percentile(b_in[ok], 90):+.1f}°, 절댓값 중앙 {np.median(np.abs(beta[ok])):.1f}°")
# table on half the sessions: median slip by (lateral acceleration, speed)
ea = np.array([-20, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 20.0]); ev = np.array([2, 3.5, 4.5, 5.5, 6.5, 12.0])
ia = np.clip(np.digitize(ay, ea) - 1, 0, len(ea) - 2); iv = np.clip(np.digitize(vx, ev) - 1, 0, len(ev) - 2)
tab = np.zeros((len(ea) - 1, len(ev) - 1))
fit = ok & half[None]; test = ok & ~half[None]
for i in range(len(ea) - 1):
    for j in range(len(ev) - 1):
        m = fit & (ia == i) & (iv == j)
        tab[i, j] = np.median(beta[m]) if m.sum() >= 20 else np.nan
col = np.array([np.nanmedian(tab[i]) for i in range(len(ea) - 1)])
tab = np.where(np.isnan(tab), col[:, None], tab)
pred = tab[ia, iv]
# linear steady-state form: beta = a * r/v + b * ay
X = np.stack([(r_prev / np.maximum(vx, 1.0))[fit], ay[fit]], 1); coef, *_ = np.linalg.lstsq(X, np.radians(beta[fit]), rcond=None)
pred_lin = np.degrees(coef[0] * r_prev / np.maximum(vx, 1.0) + coef[1] * ay)
print(f"  직선 꼴: 미끄럼 각 = {coef[0]:+.3f} m x (회전 속도/속도) {coef[1]:+.4f} s²/m x 옆가속도")
def show(name, e, m):
    e = e[m]
    print(f"  {name}: 오차 중앙 {np.median(e):+.2f}°, 절댓값 중앙 {np.median(np.abs(e)):.2f}°, 90 % {np.percentile(np.abs(e), 90):.2f}°, 97 % {np.percentile(np.abs(e), 97):.2f}°")
show("아무것도 모름(0°로 봄, 지금 plan의 가정에 가까움)", beta, test)
show("표(옆가속도 x 속도; 직전 0.1 s 회전 속도 사용)", beta - pred, test)
show("직선 꼴", beta - pred_lin, test)
big = test & (np.abs(ay) > 4)
show("옆가속도 4 넘는 곳만 — 0°로 봄", beta, big); show("옆가속도 4 넘는 곳만 — 표", beta - pred, big)
# transients: where the yaw rate changes fast
dr = np.zeros_like(r); dr[1:] = (r_prev[1:] - r_prev[:-1]) / dt
tr = test & (np.abs(dr) > 3.0)
show(f"회전 속도가 빨리 바뀌는 곳({100 * tr.sum() / test.sum():.0f} %) — 표", beta - pred, tr)
print("  표(행 = 옆가속도 칸 가운데, 열 = 속도 2~3.5 / 3.5~4.5 / 4.5~5.5 / 5.5~6.5 / 6.5~ m/s) [°]:")
for i in range(len(ea) - 1):
    c = 0.5 * (max(ea[i], -9) + min(ea[i + 1], 9))
    print(f"    {c:+5.1f}: " + "  ".join(f"{x:+5.1f}" for x in tab[i]))
# what an error of e deg means: sideways drift after 0.3 s at the speed
v_ = vx[test]; e_ = np.radians(np.abs((beta - pred)[test]))
print(f"  표의 오차가 0.3 s 동안 그대로면 옆으로 벗어남: 중앙 {100 * np.median(v_ * 0.3 * np.sin(e_)):.1f} cm, 90 % {100 * np.percentile(v_ * 0.3 * np.sin(e_), 90):.1f} cm (0°로 보면 중앙 {100 * np.median(v_ * 0.3 * np.sin(np.radians(np.abs(beta[test])))):.1f} cm, 90 % {100 * np.percentile(v_ * 0.3 * np.sin(np.radians(np.abs(beta[test]))), 90):.1f} cm)")
