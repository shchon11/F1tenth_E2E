"""횡방향 비용을 둔다면 상자 구간에서 무엇을 얼마나 세게 되나 (11-12; 사용자 생각: 횡방향 주행 안정성 비용으로 '미리 꺾기'를 간접적으로 이끌 수 있나).
시험 기록(정확히 실행)에서, 닿지 않고 지난 길 위 상자마다 그 둘레 구간에서 더한 값을 같은 세션에서 상자를 치운 뒤 같은 구간의 값과 비교:
 - 횡가속도의 변화(= 결정 사이 횡가속도 차 / 0.1 s)의 제곱 합
 - 그 변화의 변화(= 조향을 돌리는 빠르기가 뒤집히는 정도; plan_rl --smooth-reg가 세는 것과 같은 종류)의 제곱 합
 - 둘의 최댓값
넓은 구간(상자 8 m 전 ~ 6 m 뒤)과 좁은 구간(3 m 전 ~ 2 m 뒤) 둘 다. 상자를 옆으로 비킨 거리별, 그리고 상자가 얼마나 일찍 보였는지별로 나눔
(늦게 보인 상자에서 비용이 더 크면, 그 비용 아래에서는 '미리 아는 것'(기억)이 버는 자리가 생김).
빈 트랙을 달리는 것만으로 내는 값(결정마다의 분포, 한 바퀴 합)도 같이 적음: 비용의 대부분이 평소 굽은 길에서 나오면 상자보다 평소 주행을 먼저 바꿈.
인자: 기록npz 통과행npy"""
import sys
import numpy as np
from rec_lib import Rec, along_track

IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
ZONES = ((-8.0, 6.0), (-3.0, 2.0))
R = Rec(sys.argv[1]); z = R.z; T, B, dt = R.T, R.B, R.dt; rows = np.load(sys.argv[2])
st = R.st.astype(float); V = np.abs(st[:, :, 3]); k_rm = min(int(round(float(z["rm_t"]) / dt)), T)
hard_any = R.d[:, :, 3:6].sum(-1) > 0
s, lap, L = along_track(z, T); u = lap * L + s
dyaw = np.diff(st[:, :, 2], axis=0); dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
alat = V[:T] * dyaw / dt
jerk = np.diff(alat, axis=0) / dt; snap = np.diff(jerk, axis=0) / dt


def cross(ub, c, k_lo, k_hi):
    k = np.nonzero(ub[k_lo: k_hi + 1] >= c)[0]
    if len(k) == 0 or k[0] == 0:
        return np.nan
    k = k_lo + int(k[0]); d = ub[k] - ub[k - 1]
    return k - 1 + (c - ub[k - 1]) / d if d > 1e-6 else float(k)


def seg(b, c0, c1, k_lo, k_hi):
    """over the places [c0, c1]: summed jerk^2, summed snap^2 (x dt), their peaks, the time [s]; None when the stretch has a hard contact, a stop or a restart"""
    a, e = cross(u[:, b], c0, k_lo, k_hi), cross(u[:, b], c1, k_lo, k_hi)
    if not (np.isfinite(a) and np.isfinite(e)) or e <= a:
        return None
    i0, i1 = int(np.floor(a)), int(np.ceil(e))
    if hard_any[i0: i1 + 1, b].any() or V[i0: i1 + 1, b].min() < 1.0 or R.rs[i0: i1 + 1, b].any():
        return None
    j = jerk[i0: max(i1 - 1, i0 + 1), b]; sn = snap[i0: max(i1 - 2, i0 + 1), b]
    return np.array([(j ** 2).sum() * dt, (sn ** 2).sum() * dt, np.abs(j).max(), np.abs(sn).max(), (e - a) * dt])


sel = (rows[:, IGAP] < 0.40) & (rows[:, IOPP] == 0) & (rows[:, ITOUCH] == 0) & (rows[:, IT] >= 70) & (rows[:, IT] < k_rm)
q = lambda x: f"{np.nanmedian(x):.2f} [{np.nanpercentile(x, 25):.2f}~{np.nanpercentile(x, 75):.2f}]"
print(f"{z['exam']} · {R.name} · 실행 {z['mode']} · 상자 {z['count']} — 혼자 달리는 세션, 닿지 않고 지난 길 위 상자 {int(sel.sum())}번")
for c0, c1 in ZONES:
    res = []
    for r in rows[sel]:
        b, t = int(r[IB]), int(r[IT])
        w = seg(b, u[t, b] + c0, u[t, b] + c1, max(t - 90, 0), k_rm - 1)
        rf = [x for x in (seg(b, u[t, b] + m * L + c0, u[t, b] + m * L + c1, k_rm + 30, T - 3) for m in range(1, 9)) if x is not None]
        if w is None or len(rf) < 2:
            continue
        res.append(np.concatenate([[r[ISHIFT], r[ISEE]], w, np.median(np.array(rf), 0)]))
    A = np.array(res); sh, see = A[:, 0], A[:, 1]; W, E = A[:, 2: 7], A[:, 7: 12]
    print(f"구간: 상자 {-c0:.0f} m 전 ~ {c1:.0f} m 뒤 (통과 {len(A)}) — 상자 있을 때 / 치운 뒤 같은 구간 (중앙 [25~75 %]; 1 = 같음)")
    big = sh >= 0.30
    groups = [("비킨 거리 0.15 m 미만", sh < 0.15), ("0.15~0.30 m", (sh >= 0.15) & (sh < 0.30)), ("0.30~0.60 m", (sh >= 0.30) & (sh < 0.60)), ("0.60 m 넘게", sh >= 0.60)]
    if big.sum() >= 15:
        c_lo, c_hi = np.percentile(see[big], [33.3, 66.7])
        groups += [(f"0.30 m 넘게 비킨 것 가운데 늦게 보인 것 (가장 가까워지기 {c_lo:.1f} s 전 이후)", big & (see <= c_lo)),
                   (f"  〃 중간 ({c_lo:.1f}~{c_hi:.1f} s 전)", big & (see > c_lo) & (see <= c_hi)), (f"  〃 일찍 보인 것 ({c_hi:.1f} s 전보다 먼저)", big & (see > c_hi))]
    for name, m in groups:
        if m.sum() < 5:
            continue
        print(f"  {name} ({int(m.sum())}) | 횡가속도 변화² 합 {q(W[m, 0] / E[m, 0])} | 변화의 변화² 합 {q(W[m, 1] / E[m, 1])} | 변화 최댓값 {q(W[m, 2] / E[m, 2])} | "
              f"변화의 변화 최댓값 {q(W[m, 3] / E[m, 3])} | 더 걸린 시간 {np.nanmedian(W[m, 4] - E[m, 4]):+.2f} s | 더 낸 변화² {np.nanmedian(W[m, 0] - E[m, 0]):+.0f} (m/s³)²·s")
    print(f"  치운 뒤 같은 구간의 값: 변화² 합 중앙 {np.nanmedian(E[:, 0]):.0f} (m/s³)²·s · 변화 최댓값 {np.nanmedian(E[:, 2]):.0f} m/s³ · 걸린 시간 {np.nanmedian(E[:, 4]):.2f} s")
    if big.sum() >= 12:
        print(f"  0.30 m 넘게 비킨 통과에서 '보인 시간'과의 상관: 변화² 비 {np.corrcoef(see[big], W[big, 0] / E[big, 0])[0, 1]:+.2f} · 더 걸린 시간 {np.corrcoef(see[big], W[big, 4] - E[big, 4])[0, 1]:+.2f} "
              f"(음수 = 늦게 보일수록 큼)")

solo = np.nonzero(~R.on[:T].any(0))[0]
J = np.abs(jerk[k_rm + 30: T - 3][:, solo]); ok = (V[k_rm + 30: T - 3][:, solo] > 1.0)
lap_s = float(np.median([(T - 4 - k_rm - 30) * dt / max((u[T - 4, b] - u[k_rm + 30, b]) / L, 1e-6) for b in solo]))
per_lap = float(np.median((J ** 2 * ok).sum(0) * dt / ((T - 3 - k_rm - 30) * dt) * lap_s))
print(f"빈 트랙(상자를 치운 뒤, 혼자): 결정마다 |횡가속도 변화| 50/75/90/95/99 % = " + " / ".join(f"{x:.0f}" for x in np.percentile(J[ok], [50, 75, 90, 95, 99])) + " m/s³"
      f" · 한 바퀴({lap_s:.1f} s)에 내는 변화² 합 중앙 {per_lap:.0f} (m/s³)²·s")
