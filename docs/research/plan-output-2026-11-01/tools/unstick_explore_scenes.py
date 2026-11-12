"""unstick_explore.py 기록의 선 차를 묶음별로 크게 봄 (11-12): 묶음마다 벗어난 차와 못 벗어난 차를 몇 대씩.
건마다 왼쪽 = 장면(±2.5 m; 진한 파랑 = 선 순간의 차(흰 선 = 앞), 옅은 파랑 = 15 s 뒤(또는 3 m 벗어난 순간), 파란 선 = 선 3 s 전부터의 자취
(빨간 구간 = 후진), 검은 선 = 선 순간의 plan, 갈색 = 상자, 초록 = 상대 차), 오른쪽 = 속도(파랑, 음수 = 후진), policy의 속도 추정(회색),
첫 0.1 s 속도 명령의 평균(-1..1, 보라), 앞 ±10도 스캔의 가장 가까운 거리(초록 점선, m).
인자: 기록npz png [묶음마다 건수, 기본 3]"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rec_lib as RL
plt = RL.plt
R = RL.Rec(sys.argv[1]); dt, T = R.dt, R.T
z = np.load(sys.argv[1])
ev, vhat, front, asp = z["ev"], z["vhat"], z["front"], z["asp"]
ARMS = [str(a) for a in z["arms"]]
xy = R.st[:T, :, :2].astype(float); v = R.st[:T, :, 3].astype(float)
n15 = int(15.0 / dt); per = int(sys.argv[3]) if len(sys.argv) > 3 else 3
cand = {a: [] for a in range(len(ARMS))}
for b, k, arm in ev:
    if k + n15 >= T or R.rs[k:k + n15, b].any():
        continue
    d = np.linalg.norm(xy[k:k + n15 + 1, b] - xy[k, b], axis=1)
    j = np.nonzero(d >= 3.0)[0]
    cand[int(arm)].append((0 if len(j) else 1, int(j[0]) if len(j) else n15, int(b), int(k)))
rows = []
for a in cand:
    c = sorted(cand[a]); free = [r for r in c if r[0] == 0]; stay = [r for r in c if r[0] == 1]
    take = free[:max(per - min(len(stay), per // 2 if free else per), 0)]
    take += stay[:per - len(take)]
    rows += [(a,) + r for r in take]
    print(f"{ARMS[a]}: 볼 수 있는 선 차 {len(c)}대 가운데 벗어남 {len(free)}, 그림에 {len(take)}대")
n = max(len(rows), 1)
fig, axs = plt.subplots(n, 2, figsize=(15, 5.0 * n), gridspec_kw={"width_ratios": [1, 1.5]}, squeeze=False)
for (arm, stay, je, b, k), (ax, a2) in zip(rows, axs):
    tk = int(R.tid[k, b]); R.draw_map(ax, tk)
    for q in R.boxes(k, b):
        ax.fill(q[:, 0], q[:, 1], color=RL.C_BOX, zorder=2)
    e = min(k + je, T - 1)
    if R.on[k, b]:
        for tt, al in ((k, 0.8), (e, 0.3)):
            o = R.op[tt, b]; q = R.body(o[0], o[1], o[2]); ax.fill(q[:, 0], q[:, 1], color="#2ca02c", alpha=al, zorder=3)
    a0 = max(k - 30, 0)
    ax.plot(xy[a0:e + 1, b, 0], xy[a0:e + 1, b, 1], color="#1f77b4", lw=1.0, zorder=3)
    back = v[a0:e + 1, b] < -0.1
    ax.plot(np.where(back, xy[a0:e + 1, b, 0], np.nan), np.where(back, xy[a0:e + 1, b, 1], np.nan), color="#d62728", lw=2.0, zorder=4)
    for tt, al in ((k, 0.9), (e, 0.35)):
        q = R.body(*R.st[tt, b, :3]); ax.fill(q[:, 0], q[:, 1], color="#1f77b4", alpha=al, zorder=4)
        hd = R.st[tt, b]; ax.plot([hd[0], hd[0] + 0.35 * np.cos(hd[2])], [hd[1], hd[1] + 0.35 * np.sin(hd[2])], color="white", lw=2, zorder=5)
    p = np.concatenate([xy[k, b][None], R.pl[k, b]], 0); ax.plot(p[:, 0], p[:, 1], "-o", color="k", ms=3, lw=1.5, zorder=6)
    ax.set_xlim(xy[k, b, 0] - 2.5, xy[k, b, 0] + 2.5); ax.set_ylim(xy[k, b, 1] - 2.5, xy[k, b, 1] + 2.5)
    c6 = R.d[max(k - 30, 0):k + 1, b].sum(0)
    ax.set_title(f"[{ARMS[arm]}] 차 {b}: 출발 {k * dt:.1f} s 뒤에 섬 ({RL.trk_name(tk)})\n"
                 f"서기 전 3 s에 새로 닿음: 벽 {int(c6[0])} · 상자 {int(c6[1])} · 상대 차 {int(c6[2])}", fontsize=10)
    ts = (np.arange(a0, e + 1) - k) * dt
    a2.plot(ts, v[a0:e + 1, b], color="#1f77b4", label="속도 (m/s, 음수 = 후진)")
    a2.plot(ts, vhat[a0:e + 1, b], color="#7f7f7f", lw=1.0, label="policy의 속도 추정 (m/s)")
    a2.plot(ts, asp[a0:e + 1, b], color="#9467bd", lw=1.0, label="첫 0.1 s 속도 명령의 평균 (-1..1)")
    a2.plot(ts, np.clip(front[a0:e + 1, b], 0, 4), color="#2ca02c", ls="--", lw=1.0, label="앞 ±10도 스캔의 가장 가까운 거리 (m)")
    a2.axhline(0, color="k", lw=0.5); a2.axvline(0, color="#d62728", lw=0.8); a2.set_xlabel("선 뒤 시간 (s)")
    a2.legend(fontsize=9, loc="upper right"); a2.set_ylim(-2.5, 5)
    c6 = R.d[k:e + 1, b].sum(0)
    a2.set_title((f"{je * dt:.1f} s 뒤 3 m 넘게 벗어남" if not stay else "15 s 동안 못 벗어남")
                 + f" — 새로 닿음 벽 {int(c6[0])} · 상자 {int(c6[1])} · 상대 차 {int(c6[2])}번, 서 있던 시간 "
                 f"{(np.abs(v[k:e + 1, b]) < 0.3).sum() * dt:.1f} s, 후진한 시간 {(v[k:e + 1, b] < -0.2).sum() * dt:.1f} s", fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=60); print("그림", sys.argv[2])
