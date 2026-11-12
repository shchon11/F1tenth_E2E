"""출발 직후의 차끼리 접촉 (11-12): 상대 차가 있는 경주의 출발을 놓인 모양별(나란히 / 상대가 앞 / 상대가 뒤)·상대 차를 누가 모는가별로.
11-11 측정(plan25r, 기존 상대 차): 상대 있는 출발의 30 %가 나란히(옆면 사이 0.40 m)였고 2초 안 접촉이 14 % 대 그 밖 2.4 %.
drive_record.py 기록에서 출발(우리 차가 새 경주를 시작한 결정)마다:
  놓인 모양 = 우리 차 기준 상대 차 위치: 앞뒤 차이 0.5 m 안 + 옆 차이 1.0 m 안 = 나란히, 그 밖은 상대가 앞 / 뒤
  2초·3초 안에 차끼리 닿았나, 세게 닿았나 / 닿기 전 0.5초 동안 옆으로 다가간 쪽(우리 / 상대: 각자의 옆 방향 이동이 더 큰 쪽)
  견줄 값: 같은 기록의 출발 5초 뒤부터의 접촉(분당)을 2초 창으로 바꾼 것
인자: 기록 npz [기록 npz ...]"""
import sys
import numpy as np

W2, W3 = 20, 30                                                       # 2 s, 3 s in decisions (0.1 s)


def one(fn):
    z = np.load(fn, allow_pickle=False); dt = float(z["dt"]); T = z["cnt"].shape[0]
    st = z["st"].astype(float); op = z["op"].astype(float); on = z["on"][:T]; rs = z["rs"][:T]
    B = on.shape[1]
    copy = z["copy"][:T] >= 0 if "copy" in z else np.zeros((T, B), bool)
    kind = z["kind"][:T]; kinds = [str(k) for k in z["kinds"]]
    COLS = [str(c) for c in z["cols"]]; ic, ih = COLS.index("onsets_car"), COLS.index("hard_car")
    inc = np.clip(np.diff(np.concatenate([np.zeros((1, B, len(COLS)), np.float32), z["cnt"]], 0), axis=0), 0, None)
    car, hard = inc[..., ic] > 0, inc[..., ih] > 0
    start = np.zeros((T, B), bool); start[0] = True; start[1:] = rs[:-1]
    age = np.zeros((T, B), int)                                       # decisions since the race began
    for t in range(1, T):
        age[t] = np.where(start[t], 0, age[t - 1] + 1)
    d = op[:T, :, :2] - st[:T, :, :2]
    c, s = np.cos(st[:T, :, 2]), np.sin(st[:T, :, 2])
    dx = d[..., 0] * c + d[..., 1] * s; dy = -d[..., 0] * s + d[..., 1] * c
    rows = []
    for t, b in zip(*np.nonzero(start & on)):
        if t + W3 >= T or rs[t:t + W3, b].any() or not on[t:t + W3, b].all():
            continue
        shape = 0 if (abs(dx[t, b]) < 0.5 and abs(dy[t, b]) < 1.0) else (1 if dx[t, b] > 0 else 2)
        if np.hypot(dx[t, b], dy[t, b]) > 8.0:
            shape = 3                                                 # too far apart to meet within 3 s
        hit = np.nonzero(car[t:t + W3, b])[0]
        t_hit = int(hit[0]) if len(hit) else -1
        who = 0
        if 0 <= t_hit:
            a = max(t, t + t_hit - 5); e = t + t_hit
            # sideways (across the line joining the two cars at the start of the window) movement of each car toward the other
            u = d[a, b] / max(np.linalg.norm(d[a, b]), 1e-6)
            mv_s = float((st[e, b, :2] - st[a, b, :2]) @ u); mv_o = float(-(op[e, b, :2] - op[a, b, :2]) @ u)
            who = 1 if mv_s > mv_o else 2
        rows.append((shape, int(copy[t, b]), int(kind[t, b]), t_hit, int(hard[t:t + W2, b].any()), int(hard[t:t + W3, b].any()), who,
                     abs(dy[t, b])))
    rows = np.array(rows, float).reshape(-1, 8)
    late = on & (age >= 50)
    base = dict(mins=late.sum() * dt / 60, car=car[late].sum(), hard=hard[late].sum(),
                copy_mins=(late & copy).sum() * dt / 60, copy_car=car[late & copy].sum(), copy_hard=hard[late & copy].sum())
    return rows, kinds, base, str(z["name"])


ALL, base, kinds, name = [], None, None, ""
for fn in sys.argv[1:]:
    r, kinds, b, name = one(fn)
    ALL.append(r)
    base = b if base is None else {k: base[k] + b[k] for k in b}
R = np.concatenate(ALL)
print(f"== {name} ({len(sys.argv) - 1}개 기록): 상대 차가 있고 3초 동안 이어진 출발 {len(R)}번 ==")
m = max(base["mins"], 1e-9)
print(f"견줄 값 (출발 5초 뒤부터, {base['mins']:.0f}분): 차끼리 접촉 {base['car'] / m:.2f}/분 = 2초 창에 {100 * base['car'] / m / 30:.1f} % · "
      f"세게 {base['hard'] / m:.2f}/분 = 2초 창에 {100 * base['hard'] / m / 30:.1f} %")
print("놓인 모양 · 상대 차를 모는 것 | 출발 수 | 2초 안 접촉 | 2초 안 세게 | 3초 안 세게 | 닿기 전 다가간 쪽 (우리 / 상대)")
SH = ("나란히 (앞뒤 0.5 m · 옆 1.0 m 안)", "상대가 앞", "상대가 뒤", "8 m 넘게 떨어짐")


def row(label, mk):
    n = int(mk.sum())
    if n < 20:
        return
    h2 = (R[mk, 3] >= 0) & (R[mk, 3] < W2)
    print(f"{label} | {n} | {int(h2.sum())} ({100 * h2.mean():.1f} %) | {int(R[mk, 4].sum())} ({100 * R[mk, 4].mean():.1f} %) | "
          f"{int(R[mk, 5].sum())} ({100 * R[mk, 5].mean():.1f} %) | {int((R[mk, 6][h2] == 1).sum())} / {int((R[mk, 6][h2] == 2).sum())}")


for k, lab in enumerate(SH):
    row(lab + " · 전체", R[:, 0] == k)
    row(lab + " · 복사본", (R[:, 0] == k) & (R[:, 1] == 1))
    for j, kn in enumerate(kinds):
        row(lab + f" · 기존 {kn}", (R[:, 0] == k) & (R[:, 1] == 0) & (R[:, 2] == j))
ab = R[:, 0] == 0
if ab.sum() >= 20:
    print(f"나란히 출발의 옆 간격(중심 사이): 중앙 {np.median(R[ab, 7]):.2f} m, 가장 좁은 10 % {np.percentile(R[ab, 7], 10):.2f} m · "
          f"상대 있는 출발 가운데 나란히 {100 * ab.mean():.0f} %")
