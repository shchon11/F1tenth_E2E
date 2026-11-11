"""상자가 없을 때 상대 차가 얼마나 잘 달리나 (11-12 사용자: "장애물이 있든 없든 그냥 존나 못달린다", "그냥 주행이 된다~ 수준").
opp_kind_table.py는 서 버리는 비율·접촉만 봐서 '상자 없으면 멀쩡'이라고 잘못 읽었음 -> 여기서는 달리는 질을 봄.
drive_record.py 기록(npz)에서 상자가 하나도 없는 동안만, 종류별(+ 속도 배율 구간)로:
  한 바퀴 시간(같은 자리·같은 방향으로 돌아올 때까지; 그 사이 상자·다시 놓임이 없는 바퀴만), 평균 속도, 직선 속도(상위 10 %),
  옆 가속도(속도 x 방향 바뀌는 빠르기)의 상위 10 % / 1 %, 굽은 방향이 바뀐 횟수/s(좌우 흔들림), 방향 바뀌는 빠르기의 떨림(둘째 차분 RMS),
  벽까지 거리(차 중심) 하위 5 %, 벽·우리 차에 닿은 횟수/분.
같은 표에 우리 차(혼자 / 상대 있는 경주)를 같은 방식으로 넣어 기준으로 삼음. `--free`: 상대 차는 우리 차가 4 m 밖에 있는 결정만.
인자: 기록 npz [--free] [--tid N: 그 트랙만]"""
import sys
import numpy as np
from rec_lib import Rec

fn = sys.argv[1]
FREE = "--free" in sys.argv
TID = int(sys.argv[sys.argv.index("--tid") + 1]) if "--tid" in sys.argv else None
rc = Rec(fn); z = rc.z
T, B, dt = rc.T, rc.B, rc.dt
kinds = rc.kinds
on = z["on"][:T]; kind = z["kind"][:T]; scale = z["scale"][:T]; tid = z["tid"][:T]
nobox = ~(z["sid"][:T] >= 0).any(-1)
if TID is not None:
    nobox = nobox & (tid == TID)
gap = np.linalg.norm(z["st"][:T, :, :2].astype(float) - z["op"][:T, :, :2].astype(float), axis=2)
print(f"== {rc.name} ({fn.split('/')[-1]}, {B}대 x {T * dt:.0f} s), 상자 없는 동안만" + (", 우리 차가 4 m 밖일 때만" if FREE else "")
      + (f", 트랙 {TID}만" if TID is not None else "") + " ==")


def laps(st, ok):
    """lap times [s]: from each decision, the first return (>= 3 s later) to within 0.5 m with the same heading; `ok` (T, B) must hold all the way"""
    xy = st[:T, :, :2].astype(float); yaw = st[:T, :, 2].astype(float)
    bad = np.concatenate([np.zeros((1, B)), np.cumsum(~ok, 0)], 0)
    out = [[] for _ in range(B)]
    for b in range(B):
        if ok[:, b].sum() < 40:
            continue
        d = np.linalg.norm(xy[:, None, b] - xy[None, :, b], axis=2)
        dy = np.abs((yaw[:, None, b] - yaw[None, :, b] + np.pi) % (2 * np.pi) - np.pi)
        t = 0
        while t < T - 30:
            if not ok[t, b]:
                t += 1; continue
            c = np.nonzero((d[t, t + 30:] < 0.5) & (dy[t, t + 30:] < 0.6))[0]
            if len(c) == 0:
                break
            t2 = t + 30 + int(c[0])
            # the closest decision of that pass, not the first inside 0.5 m
            j = t2
            while j + 1 < T and d[t, j + 1] < d[t, j]:
                j += 1
            if bad[j + 1, b] - bad[t, b] == 0:
                out[b].append((j - t) * dt)
            t = j
    return out


def rows(st, cnt, mask, label, lap_ok):
    if mask.sum() * dt < 30.0:
        return
    xy = st[:, :, :2].astype(float); v = st[:T, :, 3].astype(float)
    yaw = st[:, :, 2].astype(float)
    r = ((yaw[1:T + 1] - yaw[:T] + np.pi) % (2 * np.pi) - np.pi) / dt
    step = np.linalg.norm(xy[1:T + 1] - xy[:T], axis=2)
    m = mask & (step < 2.0)
    mv = m & (np.abs(v) > 1.0)
    ay = np.abs(v * r)
    # sign changes of the turning direction (dead band 0.15 rad/s), per second of driving
    sg = np.where(r > 0.15, 1, np.where(r < -0.15, -1, 0))
    flips = 0; secs = 0.0
    for b in np.nonzero(mv.any(0))[0]:
        s = sg[mv[:, b], b]; s = s[s != 0]
        flips += int((s[1:] != s[:-1]).sum()); secs += mv[:, b].sum() * dt
    d2 = np.zeros_like(r); d2[1:-1] = r[2:] - 2 * r[1:-1] + r[:-2]
    m2 = mv.copy(); m2[0] = m2[-1] = False; m2[1:-1] &= mv[2:] & mv[:-2]
    wd = np.concatenate([rc.wall_dist(int(k), xy[:T][m & (tid == k)]) for k in np.unique(tid[m])])
    d = np.clip(np.diff(np.concatenate([np.zeros((1,) + cnt.shape[1:], np.float32), cnt], 0), axis=0), 0, None)
    d[step > 2.0] = 0.0
    c = d[m].sum(0) / (m.sum() * dt / 60.0)
    lp = [x for b in np.nonzero(mask.any(0))[0] for x in lap_ok[b]]
    lt = f"{np.median(lp):5.2f} s ({len(lp):3d}바퀴, 빠른 10 % {np.percentile(lp, 10):5.2f})" if len(lp) >= 3 else f"   -    ({len(lp):3d}바퀴)              "
    print(f"  {label:<28} {m.sum() * dt / 60:5.1f}분 | 한 바퀴 {lt} | 속도 평균 {np.abs(v[m]).mean():4.2f} 상위10% {np.percentile(np.abs(v[m]), 90):4.2f} m/s | "
          f"옆가속 상위10% {np.percentile(ay[mv], 90):4.1f} 상위1% {np.percentile(ay[mv], 99):4.1f} m/s² | 좌우 바뀜 {flips / max(secs, 1e-9):4.2f}/s | "
          f"떨림 {np.sqrt((d2[m2] ** 2).mean()):4.2f} | 벽 거리 하위5% {np.percentile(wd, 5):4.2f} m | 닿음/분 벽 {c[0]:4.2f} 차 {c[2]:4.2f} (세게 벽 {c[3]:4.2f} 차 {c[5]:4.2f})")


op, st = z["op"], z["st"]
mo = on & nobox & ((gap > 4.0) if FREE else True)
lap_o = laps(op, mo)
for k, kn in enumerate(kinds):
    m = mo & (kind == k)
    if m.sum() == 0:
        continue
    sel = lambda mm: [lap_o[b] if mm[:, b].any() else [] for b in range(B)]
    rows(op, z["ocnt"], m, f"상대 {kn} ({int(m.any(0).sum())}대)", sel(m))
    for lo, hi in ((0.0, 0.7), (0.7, 0.95), (0.95, 9.9)):
        mm = m & (scale >= lo) & (scale < hi)
        rows(op, z["ocnt"], mm, f"    배율 {max(lo, 0.4):.2f}-{min(hi, 1.2):.2f} ({int(mm.any(0).sum())}대)", sel(mm))
ms = ~on & nobox
rows(st, z["cnt"], ms, f"우리 차 혼자 ({int(ms.any(0).sum())}대)", laps(st, ms))
mr = on & nobox
rows(st, z["cnt"], mr, f"우리 차, 상대 있는 경주 ({int(mr.any(0).sum())}대)", laps(st, mr))
