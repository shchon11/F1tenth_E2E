"""같은 차가 같은 상자를 다시 지날 때 무엇이 달라지나를 장면으로 (11-12: 지난 바퀴에 본·닿은 상자를 다음 바퀴에 미리 대비하나). edge_cases.py가
저장한 통과 표에서 '이번 통과에서 세게 닿았고 그 전에 이미 지난 적 있는' 건을 골라, 건마다 왼쪽 = 장면(초록 = 앞 통과의 자취, 파랑 = 이번 통과,
점 간격 0.1 s, 굵은 갈색 = 그 상자, 동그라미 = 이번 통과에서 그 상자가 스캔에 처음 보인 때의 차 위치, 검은 선 = 그때의 plan, 분홍 = 0.5 s 전 plan,
별 = 가장 가까워진 곳), 오른쪽 = 두 통과의 속도(가장 가까워지기 전 2.5 s ~ 뒤 1 s).
먼저 앞 통과에서도 세게 닿았던 건(또 닿음), 다음 늦게(0.9 s 안) 보인 건, 나머지는 라인을 크게 바꾼 건 순.
인자: box_response npz, 통과 표 npy, png [묶음 0/1/2만] [건수]"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rec_lib as RL
plt = RL.plt
R = RL.Rec(sys.argv[1]); E = np.load(sys.argv[2]); dt, T = R.dt, R.T
IB, IJ, IEP, IN, IT, IWAY, ITOUCH, IHARD, ITAU, IVIN, IVMIN, IDROP, IAL, IPT, ISHIFT, IO1, IBX, IBY = range(18)
key = {(int(r[IB]), int(r[IJ]), int(r[IEP]), int(r[IN])): i for i, r in enumerate(E)}
cand = []
for i, r in enumerate(E):
    if r[IHARD] < 0.5 or r[IN] < 0.5:
        continue
    k = (int(r[IB]), int(r[IJ]), int(r[IEP]), int(r[IN]) - 1)
    if k not in key:
        continue
    p = E[key[k]]
    grp = 0 if p[IHARD] > 0.5 else (1 if -r[ITAU] < 0.9 else 2)
    cand.append((grp, -r[ISHIFT], i, key[k]))
cand.sort()
print(f"{R.name}: 이미 지난 적 있는 상자에 세게 닿은 건 {len(cand)} | 앞 통과에서도 세게 닿음 {sum(c[0] == 0 for c in cand)}, 늦게 보임 {sum(c[0] == 1 for c in cand)}, 그 밖 {sum(c[0] == 2 for c in cand)}")
if len(sys.argv) > 4:                                         # one group only (0 hit before too / 1 seen late / 2 the rest)
    cand = [c for c in cand if c[0] == int(sys.argv[4])]
n = min(len(cand), int(sys.argv[5]) if len(sys.argv) > 5 else 12)
fig, axs = plt.subplots(n, 2, figsize=(15, 5.2 * n), gridspec_kw={"width_ratios": [1, 1.4]}, squeeze=False)
GL = ("앞 통과에서도 세게 닿음", "늦게 보인 상자", "미리 보인 상자")
for (grp, _, i, ip), (ax, a2) in zip(cand[:n], axs):
    r, p = E[i], E[ip]; b, j, t1, t0 = int(r[IB]), int(r[IJ]), int(r[IT]), int(p[IT])
    tr = int(R.tid[t1, b]); R.draw_map(ax, tr)
    for jj, q in zip([x for x in range(R.sid.shape[2]) if R.sid[t1, b, x] >= 0], R.boxes(t1, b)):
        ax.fill(q[:, 0], q[:, 1], color=RL.C_BOX, alpha=1.0 if jj == j else 0.45, zorder=2, ec="k" if jj == j else None, lw=2)
    for tt, c, lab in ((t0, "#2ca02c", "앞 통과"), (t1, "#1f77b4", "이번 통과")):
        s = R.st[max(tt - 25, 0): min(tt + 9, T), b]
        ax.plot(s[:, 0], s[:, 1], "-o", color=c, ms=3.5, lw=1.0, zorder=4, label=lab)
        q = R.body(*R.st[tt, b, :3]); ax.fill(q[:, 0], q[:, 1], color=c, alpha=0.35, zorder=3)
    ts = max(t1 + int(round(r[ITAU] / dt)), 0)
    ax.plot(*R.st[ts, b, :2], "o", mfc="none", mec="k", ms=14, mew=2, zorder=6)
    for tt, c in ((ts, "k"), (t1 - 5, "#e377c2")):
        pl = np.concatenate([R.st[tt, b, :2][None], R.pl[tt, b]], 0); ax.plot(pl[:, 0], pl[:, 1], "-", color=c, lw=2.2, zorder=5)
    ax.plot(*R.st[t1, b, :2], marker="*", color="#ffd400", ms=16, mec="k", zorder=7)
    bx = R.pose[t1, b, j, :2]; ax.set_xlim(bx[0] - 3, bx[0] + 3); ax.set_ylim(bx[1] - 3, bx[1] + 3); ax.legend(fontsize=9, loc="upper right")
    ax.set_title(f"차 {b}, {int(r[IN]) + 1}번째 통과에서 세게 닿음 ({GL[grp]}) | 이번에 처음 보인 때 {r[ITAU]:+.1f} s, 앞 통과 {p[ITAU]:+.1f} s", fontsize=11)
    for tt, c, lab in ((t0, "#2ca02c", "앞 통과" + (" (세게 닿음)" if p[IHARD] > 0.5 else " (안 닿음)" if p[ITOUCH] < 0.5 else " (약하게 닿음)")), (t1, "#1f77b4", "이번 통과 (세게 닿음)")):
        w = np.arange(max(tt - 25, 0), min(tt + 11, T))
        a2.plot((w - tt) * dt, np.abs(R.st[w, b, 3]), "-o", ms=3, color=c, label=lab)
    a2.axvline(r[ITAU], color="k", ls="--", lw=1); a2.axvline(0, color="#d62728", lw=0.8)
    a2.set_xlabel("가장 가까워진 때 기준 시간 (s) — 점선 = 이번 통과에서 상자가 처음 보인 때"); a2.set_ylabel("속도 (m/s)"); a2.set_ylim(0, 8); a2.legend(fontsize=9); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=62); print("그림", sys.argv[3])
