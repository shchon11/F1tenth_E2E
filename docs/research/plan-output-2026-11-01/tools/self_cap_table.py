"""복사본이 모는 상대 차에 최고 속도를 줬을 때 (11-12): 그 차가 깨끗하게 달리나, 우리 차와 지나가는 상황이 생기나.
같은 pace의 복사본끼리는 지나감이 0.00~0.02번/분뿐이었음(out/self_meas_summary.txt) -> 경주마다 상대 차의 최고 속도를 뽑아 줌
(`plan_rl.SelfOpponents(vcap=)`, drive_record.py 덮어쓰기 "self_vcap").
drive_record.py 기록(`cap` = 그 경주 상대 차의 최고 속도, 1000 = 없음; `copy` = 복사본이 모는가)에서 상대가 있는 경주만,
복사본은 최고 속도 구간별로 · 복사본이 아닌 차는 종류별로:
  경주 분 | 상대 차: 평균 속도 · 한 바퀴(중앙값) · 선 시간(0.3 m/s 아래) · 벽/상자 접촉과 그중 세게 /분
  우리 차: 평균 속도 · 한 바퀴(중앙값) · 벽+상자 세게 /분
  지나감: 우리가 상대를 / 상대가 우리를 (/분; opp_encounters.py와 같은 정의: 3 m 안에서 앞 +0.5 m <-> 뒤 -0.5 m)
  차끼리: 접촉 /분 · 세게 /분 · 우리가 지나간 한 번당 세게 닿은 수
한 바퀴 = 같은 자리(0.5 m)·같은 방향으로 돌아올 때까지, 그 사이 다시 놓임이 없는 것만 (트랙 크기 변형이 섞여 있으니 같은 줄의
우리 차 한 바퀴와 견줄 것).
인자: 기록 npz [구간 경계 m/s ...; 기본 3.5 4.5]"""
import sys
import numpy as np

fn = sys.argv[1]
EDGES = [float(v) for v in sys.argv[2:]] or [3.5, 4.5]
z = np.load(fn, allow_pickle=False); dt = float(z["dt"]); T = z["cnt"].shape[0]
st = z["st"].astype(float); op = z["op"].astype(float); on = z["on"][:T]
B = on.shape[1]
copy = z["copy"][:T] >= 0; cap = z["cap"][:T].astype(float); kind = z["kind"][:T]; kinds = [str(k) for k in z["kinds"]]
COLS = [str(c) for c in z["cols"]]
inc = lambda c: np.clip(np.diff(np.concatenate([np.zeros((1,) + c.shape[1:], np.float32), c], 0), axis=0), 0, None)
dc, do = inc(z["cnt"]), inc(z["ocnt"])
jump_s = np.zeros((T, B), bool); jump_o = np.zeros((T, B), bool)
jump_s[1:] = np.linalg.norm(st[1:T, :, :2] - st[:T - 1, :, :2], axis=2) > 2.0
jump_o[1:] = np.linalg.norm(op[1:T, :, :2] - op[:T - 1, :, :2], axis=2) > 2.0
d = op[:T, :, :2] - st[:T, :, :2]
c, s = np.cos(st[:T, :, 2]), np.sin(st[:T, :, 2])
dx = d[..., 0] * c + d[..., 1] * s
dist = np.linalg.norm(d, axis=2)
vo, vs = np.abs(op[:T, :, 3]), np.abs(st[:T, :, 3])

# passes: (decision, car, +1 we pass / -1 we are passed)
PASS = []
for b in range(B):
    side = 0
    for t in range(T):
        if not on[t, b] or jump_s[t, b] or jump_o[t, b] or dist[t, b] > 3.0:
            side = 0; continue
        s_now = 1 if dx[t, b] > 0.5 else (-1 if dx[t, b] < -0.5 else 0)
        if s_now == 0:
            continue
        if side == 1 and s_now == -1: PASS.append((t, b, 1))
        if side == -1 and s_now == 1: PASS.append((t, b, -1))
        side = s_now
PASS = np.array(PASS).reshape(-1, 3)


def laps(S, ok):
    """lap times [s] with the decision each began at: the first return (>= 3 s later) to within 0.5 m with the same heading,
    `ok` holding all the way"""
    xy = S[:T, :, :2]; yaw = S[:T, :, 2]
    bad = np.concatenate([np.zeros((1, B)), np.cumsum(~ok, 0)], 0)
    out = []
    for b in range(B):
        if ok[:, b].sum() < 40:
            continue
        dd = np.linalg.norm(xy[:, None, b] - xy[None, :, b], axis=2)
        dy = np.abs((yaw[:, None, b] - yaw[None, :, b] + np.pi) % (2 * np.pi) - np.pi)
        t = 0
        while t < T - 30:
            if not ok[t, b]:
                t += 1; continue
            cnd = np.nonzero((dd[t, t + 30:] < 0.5) & (dy[t, t + 30:] < 0.6))[0]
            if len(cnd) == 0:
                break
            j = t + 30 + int(cnd[0])
            while j + 1 < T and dd[t, j + 1] < dd[t, j]:
                j += 1
            if bad[j + 1, b] - bad[t, b] == 0:
                out.append((t, b, (j - t) * dt))
            t = j
    return np.array(out).reshape(-1, 3)


LAP_S = laps(st, ~jump_s); LAP_O = laps(op, on & ~jump_o)
med = lambda L, m: (f"{np.median(L[m[L[:, 0].astype(int), L[:, 1].astype(int)], 2]):.1f} s"
                    if len(L) and m[L[:, 0].astype(int), L[:, 1].astype(int)].sum() >= 5 else "-")


def row(label, m):
    mins = m.sum() * dt / 60
    if mins < 3:
        return
    pm = m[PASS[:, 0], PASS[:, 1]] if len(PASS) else np.zeros(0, bool)
    we = int((PASS[pm, 2] == 1).sum()) if len(PASS) else 0; they = int((PASS[pm, 2] == -1).sum()) if len(PASS) else 0
    o = do[m].sum(0) / mins; l = dc[m].sum(0) / mins
    hard_car = dc[m][:, COLS.index("hard_car")].sum()
    print(f"{label} | {mins:.0f}분 | 상대 차: 평균 {vo[m].mean():.2f} m/s · 한 바퀴 {med(LAP_O, m)} · 선 시간 {100 * (vo[m] < 0.3).mean():.1f} % · "
          f"벽 {o[0]:.2f} (세게 {o[3]:.2f}) · 상자 {o[1]:.2f} (세게 {o[4]:.2f}) /분 | 우리 차: 평균 {vs[m].mean():.2f} m/s · 한 바퀴 {med(LAP_S, m)} · "
          f"벽+상자 세게 {l[3] + l[4]:.2f}/분 | 지나감: 우리가 {we}번 ({we / mins:.2f}/분) · 상대가 {they}번 ({they / mins:.2f}/분) | "
          f"차끼리: 접촉 {l[2]:.2f}/분 · 세게 {int(hard_car)}건 ({l[5]:.2f}/분)" + (f" · 우리가 지나간 한 번당 {hard_car / we:.2f}" if we else ""))


print(f"== {str(z['name'])} ({fn.split('/')[-1]}): 우리 차 {B}대 x {T * dt:.0f} s, 상대 있는 경주 {on.sum() * dt / 60:.0f}분 "
      f"(복사본이 모는 것 {(on & copy).sum() * dt / 60:.0f}분) ==")
print("누가 모는 상대 차 | 경주 시간 | 상대 차 | 우리 차 | 지나감 | 차끼리 접촉")
cm = on & copy
if cm.any():
    lo = [0.0] + EDGES; hi = EDGES + [1e2]
    for a, b_ in zip(lo, hi):
        row(f"복사본, 최고 속도 {a:g}~{b_:g} m/s" if b_ < 1e2 and a > 0 else (f"복사본, 최고 속도 {b_:g} m/s 아래" if a == 0 else f"복사본, 최고 속도 {a:g} m/s 위"),
            cm & (cap >= a) & (cap < b_))
    row("복사본, 최고 속도 없음", cm & (cap >= 1e2))
    row("복사본 전체", cm)
for k, name in enumerate(kinds):
    row(f"기존 상대 차 {name}", on & ~copy & (kind == k))
row("기존 상대 차 전체", on & ~copy)
