"""느린 차를 따라잡을 때 일찍 비키거나 줄이면 덜 닿나 (11-15, 상자 쪽은 nudge_value.py: 상자 1.5-4 m 앞에서 2도 x 0.3 s 비키면 세게 절반 아래).
입력: box_replay.py 기록 (BOX_REPLAY_OPP=1: 세계마다 선 따라가는 느린 차 하나, 상자 없음; 갈래 A 그대로 / L·R 조향 +-2도 x 0.3 s /
S plan 속도 x0.8 x 1 s). 창 = 3 s 복사 주기. 고르는 창 = 복사 순간 원본 차 앞(차 좌표 앞 0.5-8 m, 옆 1.5 m 안)에 상대 차가 있고 우리가
0.3 m/s 넘게 빠름. 복사 순간 앞 거리별로 갈래마다: 창 안에서 차와 세게 닿은 복제 비율 / 차와 닿은 비율 / 진행(앞으로 간 거리 m) / 끝 속도.
'비키기' = 복사 순간 상대 차가 있는 쪽의 반대로 튼 갈래, '다가가기' = 그쪽으로 튼 갈래.
인자: 출력앞머리 기록npz [기록npz ...]"""
import sys
import numpy as np

OUT = sys.argv[1]
LOG = open(OUT + ".txt", "w")
BINS = [(0.5, 2.0), (2.0, 4.0), (4.0, 8.0)]


def say(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


rows = []          # [bin, class, hard car, any car, progress m, end speed, hard wall]
for fn in sys.argv[2:]:
    z = np.load(fn)
    st = z["st"].astype(float); op = z["op"].astype(float); inc = z["inc"].astype(float); rs = z["rs"].astype(bool)
    grp = z["grp"].astype(int); arm = z["arm"].astype(int); PER = int(z["period"]); T = inc.shape[0]
    cols = list(z["cols"]) if "cols" in z.files else None
    i_hc = 5; i_c = 2; i_hw = 3                      # inc columns: onsets wall / prop / car, hard wall / prop / car
    for g in range(int(grp.max()) + 1):
        rr = np.nonzero(grp == g)[0]; s = rr[0]
        for w in range(T // PER):
            t0, t1 = w * PER, (w + 1) * PER
            if t1 >= T or rs[t0:t1, rr].any() or (t0 > 0 and rs[t0 - 1, s]):
                continue
            x, y, yaw, v = st[t0, s]; ox, oy, _, ov = op[t0, s]
            c, sn = np.cos(yaw), np.sin(yaw)
            dx = (ox - x) * c + (oy - y) * sn; dy = -(ox - x) * sn + (oy - y) * c
            if not (0.5 <= dx < 8.0 and abs(dy) < 1.5 and abs(v) > abs(ov) + 0.3):
                continue
            bi = next((i for i, (lo, hi) in enumerate(BINS) if lo <= dx < hi), None)
            if bi is None:
                continue
            side = np.sign(dy) or 1.0
            for a in np.unique(arm[rr]):
                if a == 0:
                    continue
                ra = rr[arm[rr] == a]
                cls = int(a)
                if a in (4, 5):
                    # arm 4 = +steer (left), 5 = -steer (right): away = to the side the other car is NOT on
                    cls = 4 if (a == 5) == (side > 0) else 5
                prog = np.hypot(np.diff(st[t0:t1, ra, 0], axis=0), np.diff(st[t0:t1, ra, 1], axis=0)).sum(0)
                for k, r_ in enumerate(ra):
                    rows.append((bi, cls, float(inc[t0:t1, r_, i_hc].sum() > 0), float(inc[t0:t1, r_, i_c].sum() > 0), prog[k],
                                 abs(st[t1 - 1, r_, 3]), float(inc[t0:t1, r_, i_hw].sum() > 0)))
A = np.array(rows, float)
NAME = {1: "그대로", 4: "상대 차 반대쪽으로 틂", 5: "상대 차 쪽으로 틂", 6: "잠깐 느리게"}
say(f"창(복제) {len(A)}개 · 기록 {len(sys.argv) - 2}개")
say(f"  {'앞 거리 [m]':>10} {'갈래':<16} {'복제 수':>6} {'차 세게 %':>9} {'차 닿음 %':>9} {'벽 세게 %':>9} {'진행 [m]':>9} {'그대로와 진행 차':>14}")
for bi, (lo, hi) in enumerate(BINS):
    base = A[(A[:, 0] == bi) & (A[:, 1] == 1)]
    for cls in (1, 4, 5, 6):
        m = (A[:, 0] == bi) & (A[:, 1] == cls)
        if not m.any():
            continue
        B_ = A[m]
        dprog = B_[:, 4].mean() - base[:, 4].mean() if len(base) else np.nan
        say(f"  {lo:>4.1f}-{hi:<4.1f}  {NAME[cls]:<16} {int(m.sum()):>6} {100 * B_[:, 2].mean():>9.1f} {100 * B_[:, 3].mean():>9.1f} {100 * B_[:, 6].mean():>9.1f}"
            f" {B_[:, 4].mean():>9.2f} {dprog:>+14.2f}")
