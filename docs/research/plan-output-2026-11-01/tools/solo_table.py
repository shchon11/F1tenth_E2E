"""혼자 달린 기록 견주기 (11-12): 상대 차 후보(aurorus_solo.py의 포팅본 기록)와 우리 policy(drive_record.py에 같은 덮어쓰기,
opp_present 0)를 같은 세계·같은 시드에서. 사용자 기준: "안 걸린다"가 아니라 혼자 달린 한 바퀴 시간을 우리 policy와 견줌.
기록마다 한 묶음, 포팅본 기록은 속도 배율(pscale)별로 한 줄씩:
  달린 분 | 평균 속도 | 한 바퀴(중앙값; 같은 자리·같은 방향으로 돌아올 때까지, 그 사이 다시 놓임 없음) · 가장 빠른 10 % | 바퀴 수
  선 시간(0.3 m/s 아래) | 벽 접촉과 그중 세게 /분 | 상자 접촉과 그중 세게 /분
  걸림 /분: 2초 동안 0.5 m도 못 간 일의 시작 (두 기록에서 같은 식; 포팅본 기록은 그때 다시 출발시키므로 선 시간은 그만큼 짧게 나옴)
  포팅본만: 회피 중이던 시간 · 속도를 줄이던 시간(gain < 0.99) · 멈추려던 시간(남은 거리 유한) · 선에서 1.5 m 넘게 벗어난 일
  포팅본의 둘째 줄(기록에 있을 때): 횡가속도(속도 x 요 각속도) 50 / 90 / 99 % · 옆 미끄러짐 속도 90 % ·
    장애물로 본 스캔 점 수와 그중 실제로는 바닥인 것 · 바닥이라고 버린 점 수 · 차체 기울기 짐작의 오차(roll, pitch; 중앙 / 95 %)
트랙이 여러 개인 기록은 "원래 트랙"만의 줄을 따로 냄 (변형 트랙은 길이가 달라 한 바퀴를 섞어 읽으면 안 됨).
인자: 기록 npz [기록 npz ...]"""
import sys
import numpy as np

K_ST = 20                                                             # 2 s in decisions


def laps(st, ok, dt):
    """(n, 3) decision it began at, car, lap time [s]: the first return (>= 3 s later) to within 0.5 m with the same heading,
    `ok` holding all the way (self_cap_table.py's)"""
    T, B = ok.shape
    xy = st[:T, :, :2]; yaw = st[:T, :, 2]
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
            if len(cnd) == 0:                                         # e.g. the way from the start spot onto its line
                t += 1; continue
            j = t + 30 + int(cnd[0])
            while j + 1 < T and dd[t, j + 1] < dd[t, j]:
                j += 1
            if bad[j + 1, b] - bad[t, b] == 0:
                out.append((t, b, (j - t) * dt))
            t = j
    return np.array(out).reshape(-1, 3)


def one(fn):
    z = np.load(fn, allow_pickle=False); dt = float(z["dt"]); T = z["cnt"].shape[0]
    st = z["st"].astype(float); rs = z["rs"][:T]; B = rs.shape[1]
    tid = z["tid"][:T].astype(int)
    COLS = [str(c) for c in z["cols"]]
    prev = np.concatenate([np.zeros((1, B, len(COLS)), np.float32), z["cnt"][:-1]], 0)
    prev[1:][rs[:-1]] = 0.0                                           # a restarted car counts from 0 again
    inc = np.clip(z["cnt"] - prev, 0, None) * (~rs)[..., None]
    jump = np.zeros((T, B), bool)
    jump[1:] = np.linalg.norm(st[1:T, :, :2] - st[:T - 1, :, :2], axis=2) > 2.0
    v = np.abs(st[:T, :, 3])
    # stuck: less than 0.5 m of net movement over 2 s without a restart in it; the onset = not stuck the decision before
    brk = np.cumsum(jump | np.concatenate([np.zeros((1, B), bool), rs[:-1]], 0), 0)
    stuck = np.zeros((T, B), bool)
    stuck[K_ST:] = (np.linalg.norm(st[K_ST:T, :, :2] - st[:T - K_ST, :, :2], axis=2) < 0.5) & (brk[K_ST:] == brk[:-K_ST])
    onset = stuck.copy(); onset[1:] &= ~stuck[:-1]
    LAP = laps(st, ~jump & ~np.concatenate([np.zeros((1, B), bool), rs[:-1]], 0), dt)
    port = "pscale" in z.files
    name = str(z["name"])
    box = bool((z["sid"] >= 0).any())
    print(f"== {name} ({fn.split('/')[-1]}): 차 {B}대 x {T * dt:.0f} s, 트랙 {len(np.unique(tid))}개, 상자 {'있음' if box else '없음'}"
          + (f", 걸리면 {float(z['stuck_s']):g}초 뒤 다시 출발" if port and float(z["stuck_s"]) > 0 else "")
          + (f", 주행 {dict(exact='정확 실행', physics='보통 물리')[str(z['drive'])]}, 바닥 점 "
             f"{dict(off='그대로', gate='자기 센서로 거름', oracle='정답으로 거름')[str(z['floor'])]}" if "drive" in z.files else "") + " ==")
    print("줄 | 달린 분 | 평균 속도 | 한 바퀴 중앙값 · 빠른 10 % (바퀴 수) | 선 시간 | 벽 접촉 (세게) /분 | 상자 접촉 (세게) /분 | 걸림 /분"
          + (" | 회피 중 · 속도 줄임 · 멈추려 함 · 선에서 벗어남" if port else ""))

    def row(label, m):
        mins = m.sum() * dt / 60
        if mins < 1:
            return
        c = inc[m].sum(0) / mins
        lm = m[LAP[:, 0].astype(int), LAP[:, 1].astype(int)] if len(LAP) else np.zeros(0, bool)
        lt = LAP[lm, 2] if len(LAP) else np.zeros(0)
        lap = f"{np.median(lt):.2f} s · {np.percentile(lt, 10):.2f} s ({len(lt)})" if len(lt) >= 5 else f"- ({len(lt)})"
        s = (f"{label} | {mins:.0f} | {v[m].mean():.2f} m/s | {lap} | {100 * (v[m] < 0.3).mean():.1f} % | {c[0]:.2f} ({c[3]:.2f}) | "
             f"{c[1]:.2f} ({c[4]:.2f}) | {onset[m].sum() / mins:.2f}")
        if port:
            lost = z["lost"][:T]; lo_on = lost.copy(); lo_on[1:] &= ~lost[:-1]
            s += (f" | {100 * z['avoid'][:T][m].mean():.0f} % · {100 * (z['gain'][:T][m] < 0.99).mean():.0f} % · "
                  f"{100 * np.isfinite(z['reach'][:T][m]).mean():.0f} % · {int(lo_on[m].sum())}번")
        print(s)
        if port and "yr" in z.files:
            ay = np.abs(st[:T, :, 3] * z["yr"][:T])[m]
            s2 = (f"    횡가속도 {np.percentile(ay, 50):.1f} / {np.percentile(ay, 90):.1f} / {np.percentile(ay, 99):.1f} m/s² · "
                  f"옆 미끄러짐 90 % {np.percentile(np.abs(z['vy'][:T][m]), 90):.2f} m/s · 장애물로 본 점 {int(z['nob'][:T][m].sum())}개 "
                  f"(그중 바닥 {int(z['nobf'][:T][m].sum())}개) · 바닥이라고 버린 점 {int(z['nfl'][:T][m].sum())}개")
            tl = z["tilt"][:T]
            if np.isfinite(tl).any():
                e = np.degrees(np.abs(tl - z["att"][:T]))[m]
                s2 += (f" · 기울기 짐작 오차 roll {np.nanmedian(e[:, 0]):.2f}° / {np.nanpercentile(e[:, 0], 95):.2f}°, "
                       f"pitch {np.nanmedian(e[:, 1]):.2f}° / {np.nanpercentile(e[:, 1], 95):.2f}°")
            print(s2)

    groups = [("모든 트랙", np.ones((T, B), bool))]
    if len(np.unique(tid)) > 1:
        groups.append(("원래 트랙", tid == 0))
    for gl, gm in groups:
        if port:
            ps = z["pscale"][:T]
            for sc in [float(x) for x in z["scales"]]:
                row(f"{gl} · 선 속도의 {sc:g}배", gm & (np.abs(ps - sc) < 1e-3))
        row(f"{gl} · 전체", gm)


for fn in sys.argv[1:]:
    one(fn)
