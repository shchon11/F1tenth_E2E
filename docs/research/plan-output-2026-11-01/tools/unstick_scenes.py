"""선 뒤 8 s 안에 3 m를 못 벗어난 차를 크게 봄 (11-12: 닿은 뒤 스스로 빠져나오기 -- 후진을 시켜 줘도 못 나오는 차는 무엇에 걸렸나).
unstick_probe.py 기록의 '선 차' 목록에서 못 벗어난 건만: 먼저 그대로 둔 차, 다음 후진을 시킨 차.
건마다 왼쪽 = 장면(±2 m; 진한 파랑 = 선 순간의 차(흰 선 = 앞), 옅은 파랑 = 8 s 뒤, 파란 선 = 선 3 s 전부터 8 s 뒤까지의 자취, 검은 선 = 선 순간의 plan,
분홍 = 2 s 뒤, 주황 = 6 s 뒤, 갈색 = 상자, 초록 = 상대 차), 오른쪽 = 속도(파랑, 음수 = 후진), policy의 속도 추정(회색), plan의 1.0 s 점이 차 앞으로
몇 m(주황, 음수 = 뒤), 앞 ±10도 스캔의 가장 가까운 거리(초록 점선, m).
인자: unstick_probe npz, png [건수, 기본 8]"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rec_lib as RL
plt = RL.plt
R = RL.Rec(sys.argv[1]); dt, T = R.dt, R.T
z = np.load(sys.argv[1])
ev, vhat, front = z["ev"], z["vhat"], z["front"]
xy = R.st[:T, :, :2].astype(float); v = R.st[:T, :, 3].astype(float)
n8 = int(8.0 / dt)
rows = []
for b, k, tr in ev:
    if k + n8 >= T:
        continue
    if (np.linalg.norm(xy[k:k + n8 + 1, b] - xy[k, b], axis=1) >= 3.0).any():
        continue
    rows.append((int(tr), int(b), int(k)))
rows.sort()
print(f"{R.name}: 선 뒤 8 s 안에 3 m를 못 벗어난 차 {len(rows)}대 (그대로 둠 {sum(r[0] == 0 for r in rows)}, 후진시킴 {sum(r[0] == 1 for r in rows)})")
n = min(len(rows), int(sys.argv[3]) if len(sys.argv) > 3 else 8)
fig, axs = plt.subplots(n, 2, figsize=(15, 5.0 * n), gridspec_kw={"width_ratios": [1, 1.5]}, squeeze=False)
for (tr, b, k), (ax, a2) in zip(rows[:n], axs):
    tk = int(R.tid[k, b]); R.draw_map(ax, tk)
    for q in R.boxes(k, b):
        ax.fill(q[:, 0], q[:, 1], color=RL.C_BOX, zorder=2)
    e = min(k + n8, T - 1)
    if R.on[k, b]:
        for tt, al in ((k, 0.8), (e, 0.3)):
            o = R.op[tt, b]; q = R.body(o[0], o[1], o[2]); ax.fill(q[:, 0], q[:, 1], color="#2ca02c", alpha=al, zorder=3)
    a0 = max(k - 30, 0)
    ax.plot(xy[a0:e + 1, b, 0], xy[a0:e + 1, b, 1], color="#1f77b4", lw=1.0, zorder=3)
    for tt, al in ((k, 0.9), (e, 0.35)):
        q = R.body(*R.st[tt, b, :3]); ax.fill(q[:, 0], q[:, 1], color="#1f77b4", alpha=al, zorder=4)
        hd = R.st[tt, b]; ax.plot([hd[0], hd[0] + 0.35 * np.cos(hd[2])], [hd[1], hd[1] + 0.35 * np.sin(hd[2])], color="white", lw=2, zorder=5)
    for tt, c in ((k, "k"), (k + 20, "#e377c2"), (k + 60, "#ff7f0e")):
        if tt < T:
            p = np.concatenate([xy[tt, b][None], R.pl[tt, b]], 0); ax.plot(p[:, 0], p[:, 1], "-o", color=c, ms=3, lw=1.5, zorder=6)
    ax.set_xlim(xy[k, b, 0] - 2.0, xy[k, b, 0] + 2.0); ax.set_ylim(xy[k, b, 1] - 2.0, xy[k, b, 1] + 2.0)
    c6 = R.d[max(k - 30, 0):k + 1, b].sum(0)
    ax.set_title(f"차 {b}: 출발 {k * dt:.1f} s 뒤에 섬 ({RL.trk_name(tk)}) — {'후진을 시킴' if tr else '그대로 둠'}\n"
                 f"서기 전 3 s에 새로 닿음: 벽 {int(c6[0])} · 상자 {int(c6[1])} · 상대 차 {int(c6[2])}", fontsize=10)
    ts = (np.arange(a0, e + 1) - k) * dt
    yaw = R.st[a0:e + 1, b, 2].astype(float)
    fwd = (R.pl[a0:e + 1, b, 9, 0] - xy[a0:e + 1, b, 0]) * np.cos(yaw) + (R.pl[a0:e + 1, b, 9, 1] - xy[a0:e + 1, b, 1]) * np.sin(yaw)
    a2.plot(ts, v[a0:e + 1, b], color="#1f77b4", label="속도 (m/s, 음수 = 후진)")
    a2.plot(ts, vhat[a0:e + 1, b], color="#7f7f7f", lw=1.0, label="policy의 속도 추정 (m/s)")
    a2.plot(ts, fwd, color="#ff7f0e", lw=1.0, label="plan의 1.0 s 점이 차 앞으로 몇 m (음수 = 뒤)")
    a2.plot(ts, np.clip(front[a0:e + 1, b], 0, 4), color="#2ca02c", ls="--", lw=1.0, label="앞 ±10도 스캔의 가장 가까운 거리 (m)")
    a2.axhline(0, color="k", lw=0.5); a2.axvline(0, color="#d62728", lw=0.8); a2.set_xlabel("선 뒤 시간 (s)")
    a2.legend(fontsize=9, loc="upper right"); a2.set_ylim(-2.5, 5)
    c6 = R.d[k:e + 1, b].sum(0)
    a2.set_title(f"선 뒤 8 s: 새로 닿음 벽 {int(c6[0])} · 상자 {int(c6[1])} · 상대 차 {int(c6[2])}번, 서 있던 시간 "
                 f"{(np.abs(v[k:e + 1, b]) < 0.3).sum() * dt:.1f} s, 후진한 시간 {(v[k:e + 1, b] < -0.2).sum() * dt:.1f} s, "
                 f"가장 멀리 {np.linalg.norm(xy[k:e + 1, b] - xy[k, b], axis=1).max():.1f} m", fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=60); print("그림", sys.argv[2])
