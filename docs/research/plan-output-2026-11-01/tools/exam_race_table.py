"""경주 시험표 (exam4, 11-24): exam_record.py 상자 개수 "race4" 기록(npz)을 조건마다 세션 단위로 채점.
조건 = 출발(반 바퀴 떨어져 / 5 m 뒤) x 상대 페이스(0.6 / 0.8 / 0.95 / 제 페이스) x 상자(줄 위 3개, 50 s에 치움 / 없음). 상대 차 = 기준 policy
(scan31b u500)의 얼린 복사본. 세션 = 90 s.
  - 바퀴: 90 s 동안 트랙을 따라 간 거리 / 한 바퀴 길이 (보상의 벽 기준 거리장, 중심선 없음); 상대 차도 같이
  - 앞지름: 상대 차 뒤에 있다가 앞으로 나간 횟수(거리 차가 바퀴 길이의 배수를 1 m 넘게 넘어간 때, exam_table.py와 같은 셈), 앞지름 당함은 반대.
    반 바퀴 출발은 상대가 반 바퀴 앞에서 시작한 것으로 셈
  - 우리 책임 차 접촉: 차끼리 닿기 시작한 때 env 규칙(뒤 차, 또는 차 길이 안에 나란히)이 우리 탓으로 보는 것
  - 세게 닿음(벽 · 상자 · 차), 멈춤(2 s 동안 트랙을 따라 1 m도 못 간 적 있는 세션, 그렇게 보낸 시간)
모든 평균·비율에 세션 다시 뽑기 95 % 구간.
인자: 기록npz"""
import os, sys
import numpy as np
# (exam_table.py's helpers, copied: importing that script runs its whole table on argv)
RNG = np.random.default_rng(0)
HYST = 1.0          # [m] a pass counts once the gap has changed sign by this much (exam_table.py's)
STALL_S = 2.0
STUCK_M = 1.0


def ci(v, stat=np.mean, n=2000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(v), (n, len(v)))
    s = np.array([stat(v[i]) for i in idx]) if stat is not np.mean else v[idx].mean(1)
    return tuple(np.percentile(s, [2.5, 97.5]))


def pct(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "  -"
    lo, hi = ci(v)
    return f"{100 * v.mean():3.0f} % [{100 * lo:.0f}-{100 * hi:.0f}]"


def mean(v, f="{:.2f}"):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "-"
    lo, hi = ci(v)
    return (f + " [" + f + "-" + f + "]").format(v.mean(), lo, hi)


def progress(rem, L, skip=None):
    """(T+1, B) distance to go -> (T+1, B) cumulative distance driven along the track (a wall / off-grid sample holds the last value;
    `skip` (T, B): decisions whose step is not driving, e.g. a re-placed car)"""
    r = rem.astype(float).copy()
    for t in range(1, r.shape[0]):                       # carry the last valid value forward
        bad = ~np.isfinite(r[t]); r[t, bad] = r[t - 1, bad]
    r[~np.isfinite(r)] = 0.0
    d = r[:-1] - r[1:]
    d = np.mod(d + 0.5 * L, L) - 0.5 * L
    if skip is not None:
        d[skip] = 0.0
    return np.concatenate([np.zeros((1, r.shape[1])), np.cumsum(d, 0)], 0), r


def runs(mask, n_min):
    """per column: the number of runs of True at least n_min long, and the longest run"""
    T, B = mask.shape
    cnt = np.zeros(B, int); longest = np.zeros(B, int); cur = np.zeros(B, int)
    for t in range(T):
        cur = np.where(mask[t], cur + 1, 0)
        cnt += (cur == n_min)
        longest = np.maximum(longest, cur)
    return cnt, longest



fn = sys.argv[1]
z = np.load(fn, allow_pickle=False)
assert "c_pace" in z.files, "exam4 (상자 개수 race4) 기록이 아님"
dt = float(z["dt"]); cnt = z["cnt"].astype(float); T, B = cnt.shape[:2]
secs = T * dt; rm_t = float(z["rm_t"])
L = float(z["lapL"][int(z["tid"][0, 0])])
c_start, c_pace, c_box = z["c_start"][:B].astype(int), z["c_pace"][:B].astype(float), z["c_box"][:B].astype(bool)
d = np.clip(np.diff(np.concatenate([np.zeros((1, B, 6)), cnt], 0), axis=0), 0, None)
hard = d[:, :, 3:6].sum(0)                                     # (B, 3) wall, box, car
fault = z["cfault"][:T].astype(float).sum(0)
P, _ = progress(z["rem"], L)
ors = z["ors"][:T]
Po, _ = progress(z["orem"], L, skip=ors)
# the gap, + = we are ahead: from 5 m behind as measured, from half a lap as the other car half a lap ahead
g0 = np.mod(z["orem"][0].astype(float) - z["rem"][0].astype(float) + 0.5 * L, L) - 0.5 * L
g0 = np.where(c_start == 0, -0.5 * L, np.nan_to_num(g0))
Gp = g0[None, :] + P - Po
made = np.zeros(B, int); lost = np.zeros(B, int); cell = np.floor(Gp[0] / L)
for t in range(1, T + 1):
    re_ = ors[t - 1]                                           # a re-placed opponent: count on from where it now is
    cell = np.where(re_, np.floor(Gp[t] / L), cell)
    up = Gp[t] >= (cell + 1) * L + HYST; dn = Gp[t] <= cell * L - HYST
    made += up & ~re_; lost += dn & ~re_
    cell = cell + up.astype(int) - dn.astype(int)
n1 = int(round(1.0 / dt)); n_st = int(round(STALL_S / dt))
adv = np.full((T, B), np.inf); adv[:T - n_st] = P[n_st:T] - P[:T - n_st]
slow = adv < STUCK_M; slow[:n1] = False
stalls, longest = runs(slow, 1)
stuck_s = slow.sum(0) * dt
laps, olaps = P[-1] / L, Po[-1] / L

print(f"== exam4 경주 시험 · {dict(map='MAP', track='시간 추종기', ideal='plan대로 정확히')[str(z['mode'])]} · {z['name']} · 상대 = "
      f"{os.path.basename(os.path.dirname(str(z['ref'])))}/{os.path.basename(str(z['ref']))}의 복사본 "
      f"(세션 {B}개 x {secs:.0f} s, 상자는 {rm_t:g} s에 치움, 시드 {int(z['seed'])}, 상대 차 다시 출발 {int(ors.sum())}번) ==")
hdr = (f"{'조건':<22} | {'세션':>3} | {'바퀴 (상대)':>13} | {'앞지름':>15} | {'앞지름 당함':>6} | {'우리 책임 차 접촉':>14} | "
       f"{'세게 닿음 벽 · 상자 · 차':>20} | {'멈춘 세션':>14}")


def line(label, m):
    n = int(m.sum())
    if n == 0:
        return
    print(f"{label:<22} | {n:>3} | {laps[m].mean():5.2f} ({olaps[m].mean():4.2f}) | {mean(made[m]):>15} | {lost[m].mean():6.2f} | "
          f"{mean(fault[m]):>14} | {hard[m, 0].mean():5.2f} · {hard[m, 1].mean():4.2f} · {hard[m, 2].mean():4.2f} | {pct(stalls[m] > 0):>14}")


START = {0: "반 바퀴", 1: "5 m 뒤"}
pace_lab = lambda p: "제 페이스" if p >= 0.999 else f"{p:.2f}배"
print("[전체]")
print(hdr)
line("전체", np.ones(B, bool))
print("[출발별 / 상대 페이스별 / 상자별]")
print(hdr)
for k, v in START.items():
    line(f"출발 {v}", c_start == k)
for p in sorted(set(c_pace.tolist())):
    line(f"상대 {pace_lab(p)}", np.isclose(c_pace, p))
line("상자 있음", c_box); line("상자 없음", ~c_box)
print("[조건마다]")
print(hdr)
for k, v in START.items():
    for p in sorted(set(c_pace.tolist())):
        for bx in (True, False):
            line(f"{v} · {pace_lab(p)} · {'상자' if bx else '빈 트랙'}", (c_start == k) & np.isclose(c_pace, p) & (c_box == bx))
print(f"멈춤 = {STALL_S:g} s 동안 트랙을 따라 {STUCK_M:g} m도 못 간 적 있는 세션 (그렇게 보낸 시간: 평균 {stuck_s.mean():.1f} s/세션, 가장 긴 것 "
      f"{(np.where(longest > 0, longest + n_st - 1, 0) * dt).max():.1f} s)")
