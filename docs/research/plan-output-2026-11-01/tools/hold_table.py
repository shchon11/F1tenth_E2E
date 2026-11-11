"""plan 하나를 N x 0.1 s 동안 그대로 달린 주행 기록(`drive_record.py`, F1SIM_HOLD=N)들의 표 (11-11: 제어기가 못 따라가는 것이 plan
출력 자체 때문인가 -- 제어기 없이 plan만 본다). 기록마다 네 경우(40 s 동안 상자·상대 차가 한 번이라도 있었나)로 나눠: 바퀴/분, 한 바퀴
중앙, 세게 닿음 /분(벽·상자·상대 차), 선 시간 %, 다시 출발 횟수.
인자: 이름=npz ..."""
import sys
import numpy as np

print("기록 | 경우 | 차 | 바퀴/분 | 한 바퀴 중앙 s | 세게 닿음 /분 벽 · 상자 · 차 (합) | 닿음 전체 /분 | 섬 % | 다시 출발")
for a in sys.argv[1:]:
    name, f = a.split("=", 1)
    D = np.load(f)
    st, lap, cnt, rs, sid, on = D["st"], D["lap"], D["cnt"], D["rs"], D["sid"], D["on"]
    T, B = rs.shape; dt = float(D["dt"])
    d = np.diff(np.concatenate([np.zeros((1, B, cnt.shape[2]), cnt.dtype), cnt], 0), axis=0).clip(min=0)   # a restart starts the counts again
    boxes = (sid >= 0).any(2).any(0); opp = on.any(0)
    for lab, m in (("빈 트랙", ~boxes & ~opp), ("상대 차", ~boxes & opp), ("상자", boxes & ~opp), ("상자+상대", boxes & opp), ("전체", np.ones(B, bool))):
        n = int(m.sum())
        if n == 0:
            continue
        mins = n * T * dt / 60.0
        lt = []
        for i in np.nonzero(m)[0]:
            t = np.nonzero(lap[:, i] > 0)[0]
            lt += list(np.diff(t) * dt)
        hard = d[:, m, 3:6].sum((0, 1)) / mins
        print(f"{name} | {lab} | {n} | {(lap[:, m] > 0).sum() / mins:.2f} | {np.median(lt) if lt else float('nan'):.2f} | "
              f"{hard[0]:.2f} · {hard[1]:.2f} · {hard[2]:.2f} ({hard.sum():.2f}) | {d[:, m, :3].sum() / mins:.2f} | "
              f"{(np.abs(st[:T, m, 3]) < 0.3).mean() * 100:.0f} | {int(rs[:, m].sum())}", flush=True)
