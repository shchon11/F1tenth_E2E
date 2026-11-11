"""세게 닿은 뒤 끝까지 못 벗어난 건을 크게 봄 (11-12: 닿은 뒤 후진해서 다시 출발하는 것이 안 되는 건이 무엇인가). recovery.py와 같은 건 정의.
건마다 왼쪽 = 장면(±1.6 m; 진한 파랑 = 닿는 순간의 차, 옅은 파랑 = 10 s 뒤, 검은 선 = 닿고 1 s 뒤의 plan, 분홍 = 5 s 뒤, 주황 = 20 s 뒤, 갈색 = 상자,
초록 = 상대 차), 오른쪽 = 닿기 2 s 전부터의 속도(파랑, 음수 = 후진)와 plan이 1.0 s 뒤에 차를 차 앞쪽으로 얼마나 보내려 하나(주황, 음수 = 뒤로).
인자: drive_record npz, png"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rec_lib as RL
plt = RL.plt
R = RL.Rec(sys.argv[1]); dt, T, B = R.dt, R.T, R.B
KN = ("벽", "상자", "상대 차")
xy = R.st[:T, :, :2].astype(float); v = R.st[:T, :, 3].astype(float)
rows = []
for b in range(B):
    ev = np.nonzero(R.d[:, b, 3:6].sum(1) > 0.5)[0]
    for t0 in ev:
        if t0 < 3:
            continue
        disp = np.linalg.norm(xy[t0:, b] - xy[t0, b], axis=1)
        if (disp >= 3.0).any() or T - t0 < 80:
            continue
        rows.append((b, int(t0), int(R.d[t0, b, 3:6].argmax()))); break
print(f"{R.name}: 세게 닿고 8 s 넘게 3 m를 못 벗어난 차 {len(rows)}대 / {B}대 | 닿은 때(출발 뒤 s): " + ", ".join(f"{t0 * dt:.1f}" for _, t0, _ in rows)
      + " | 닿은 것: " + ", ".join(KN[k] for _, _, k in rows))
n = min(len(rows), 8)
fig, axs = plt.subplots(n, 2, figsize=(15, 4.6 * n), gridspec_kw={"width_ratios": [1, 1.6]}, squeeze=False)
for (b, t0, k), (ax, a2) in zip(rows[:n], axs):
    tr = int(R.tid[t0, b]); R.draw_map(ax, tr)
    for q in R.boxes(t0, b):
        ax.fill(q[:, 0], q[:, 1], color=RL.C_BOX, zorder=2)
    if R.on[t0, b]:
        for tt, al in ((t0, 0.8), (min(t0 + 100, T - 1), 0.3)):
            o = R.op[tt, b]; q = R.body(o[0], o[1], o[2]); ax.fill(q[:, 0], q[:, 1], color="#2ca02c", alpha=al, zorder=3)
    a0 = max(t0 - 20, 0)
    ax.plot(xy[a0:, b, 0], xy[a0:, b, 1], color="#1f77b4", lw=1.0, zorder=3)
    for tt, al in ((t0, 0.9), (min(t0 + 100, T - 1), 0.35)):
        q = R.body(*R.st[tt, b, :3]); ax.fill(q[:, 0], q[:, 1], color="#1f77b4", alpha=al, zorder=4)
        hd = R.st[tt, b]; ax.plot([hd[0], hd[0] + 0.35 * np.cos(hd[2])], [hd[1], hd[1] + 0.35 * np.sin(hd[2])], color="white", lw=2, zorder=5)
    for tt, c in ((t0 + 10, "k"), (t0 + 50, "#e377c2"), (t0 + 200, "#ff7f0e")):
        if tt < T:
            p = np.concatenate([xy[tt, b][None], R.pl[tt, b]], 0); ax.plot(p[:, 0], p[:, 1], "-o", color=c, ms=3, lw=1.5, zorder=6)
    ax.plot(*xy[t0, b], marker="*", color="#ffd400", ms=18, mec="k", zorder=7)
    ax.set_xlim(xy[t0, b, 0] - 1.6, xy[t0, b, 0] + 1.6); ax.set_ylim(xy[t0, b, 1] - 1.6, xy[t0, b, 1] + 1.6)
    ax.set_title(f"차 {b}: 출발 {t0 * dt:.1f} s 뒤 {KN[k]}에 세게 닿음 ({RL.trk_name(tr)})", fontsize=11)
    ts = (np.arange(a0, T) - t0) * dt
    yaw = R.st[a0:T, b, 2].astype(float)
    fwd = (R.pl[a0:T, b, 9, 0] - xy[a0:T, b, 0]) * np.cos(yaw) + (R.pl[a0:T, b, 9, 1] - xy[a0:T, b, 1]) * np.sin(yaw)
    a2.plot(ts, v[a0:T, b], color="#1f77b4", label="속도 (m/s, 음수 = 후진)")
    a2.plot(ts, fwd, color="#ff7f0e", lw=1.0, label="plan의 1.0 s 점이 차 앞으로 몇 m (음수 = 뒤)")
    a2.axhline(0, color="k", lw=0.5); a2.axvline(0, color="#d62728", lw=0.8); a2.set_xlabel("닿은 뒤 시간 (s)"); a2.legend(fontsize=9, loc="upper right"); a2.set_ylim(-2.5, 5)
    c6 = R.d[t0:, b].sum(0)
    a2.set_title(f"닿은 뒤 기록 끝까지: 새로 닿음 벽 {int(c6[0])} · 상자 {int(c6[1])} · 상대 차 {int(c6[2])}번, 서 있던 시간 {(np.abs(v[t0:, b]) < 0.3).sum() * dt:.0f} s, "
                 f"후진한 시간 {(v[t0:, b] < -0.2).sum() * dt:.1f} s", fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=60); print("그림", sys.argv[2])
