"""상대 차와 세게 닿은 장면을 기록(drive_record.py npz)에서 그림 (11-12: 느린 line 차는 60-80 % 속도에서 85분에 3-4건인데 forzaeth /
forzaeth_pred / lane_switch는 같은 60 %에서 56-59건, 77-100 %가 우리가 들이받음 -> 숫자보다 장면 먼저).
건마다 왼쪽 = 지도 확대(±3 m): 닿기 2 s 전부터 두 차의 자취(점, 진할수록 최근), 닿기 1.0 s 전·0.5 s 전·닿는 순간의 차체(파랑 = 우리,
빨강 = 상대), 닿기 1.0 s 전(검정)·0.5 s 전(분홍) 우리 plan. 오른쪽 = 닿기 3 s 전부터: 우리 속도, 상대 속도, 우리 차 좌표로 상대가 앞으로 몇 m /
옆으로 몇 m, plan 끝점(1.0 s)까지의 거리(= plan이 요구하는 속도의 눈금).
맨 위에 종류별 요약: 닿기 1.0 s 전 상대가 우리 차 앞 ±20도 안에 있었나, 그때 거리, 닿기 전 1 s 동안 우리 속도가 얼마나 줄었나.
인자: 기록npz 출력png [장면 수 12] [종류 이름(쉼표), 기본 전부]"""
import sys
import numpy as np
import rec_lib as RL
from rec_lib import Rec, plt

R = Rec(sys.argv[1]); T, dt = R.T, R.dt
n_show = int(sys.argv[3]) if len(sys.argv) > 3 else 12
want = sys.argv[4].split(",") if len(sys.argv) > 4 else None
st = R.st.astype(float); op = R.op.astype(float)
n05, n10, n20, n30 = (int(round(x / dt)) for x in (0.5, 1.0, 2.0, 3.0))


def rel(t, b):
    """the other car in our car's frame at decision t: (ahead m, left m)"""
    dx, dy = op[t, b, 0] - st[t, b, 0], op[t, b, 1] - st[t, b, 1]
    c, s_ = np.cos(st[t, b, 2]), np.sin(st[t, b, 2])
    return dx * c + dy * s_, -dx * s_ + dy * c


ts, bs = np.nonzero((R.d[:, :, 5] > 0) & R.on[:T])
ev = []
for t, b in zip(ts, bs):
    if t - n10 < R.k0[t, b]:                                           # the race began under a second ago: a start scene
        continue
    kn = R.kinds[int(R.kind[t, b])] if R.kinds else "?"
    if want and kn not in want:
        continue
    a1, l1 = rel(t - n10, b)
    a0, l0 = rel(t, b)
    ev.append(dict(t=int(t), b=int(b), kind=kn, ahead1=a1, left1=l1, ahead0=a0, left0=l0,
                   in_view=bool(a1 > 0 and abs(np.arctan2(l1, a1)) < np.radians(20)), dist1=float(np.hypot(a1, l1)),
                   v_us1=st[t - n10, b, 3], v_us0=st[t, b, 3], v_op1=op[t - n10, b, 3], v_op0=op[t, b, 3],
                   op_side=float(rel(t, b)[1] - l1)))
print(f"{R.name} ({sys.argv[1].split('/')[-1]}): 차끼리 세게 닿음 {len(ev)}건 (경주 시작 1 s 안은 뺌)")
for kn in sorted({e["kind"] for e in ev}):
    g = [e for e in ev if e["kind"] == kn]
    f = lambda k_: np.array([e[k_] for e in g], float)
    print(f"  {kn:<14} {len(g):3d}건 | 1.0 s 전 상대가 앞 ±20도 안 {100 * f('in_view').mean():3.0f} % · 그때 거리(중앙) {np.median(f('dist1')):.2f} m | "
          f"우리 속도 1.0 s 전 {np.median(f('v_us1')):.1f} -> 닿을 때 {np.median(f('v_us0')):.1f} m/s (1 m/s 넘게 줄인 건 "
          f"{100 * ((f('v_us1') - f('v_us0')) > 1.0).mean():.0f} %) | 상대 {np.median(f('v_op1')):.1f} -> {np.median(f('v_op0')):.1f} m/s "
          f"(1 m/s 넘게 줄인 건 {100 * ((f('v_op1') - f('v_op0')) > 1.0).mean():.0f} %) | 닿을 때 상대가 앞 {np.median(f('ahead0')):.2f} · "
          f"옆 {np.median(np.abs(f('left0'))):.2f} m")

if not ev:
    sys.exit(0)
pick = [ev[i] for i in np.linspace(0, len(ev) - 1, min(n_show, len(ev))).round().astype(int)]
n = len(pick)
fig, axs = plt.subplots(n, 2, figsize=(15, 5.0 * n), gridspec_kw={"width_ratios": [1, 1.5]}, squeeze=False)
for e, (ax, a2) in zip(pick, axs):
    t, b = e["t"], e["b"]; tk = int(R.tid[t, b]); R.draw_map(ax, tk)
    for q in R.boxes(t, b):
        ax.fill(q[:, 0], q[:, 1], color=RL.C_BOX, zorder=2)
    a = max(t - n20, int(R.k0[t, b]))
    for tt in range(a, t + 1):
        al = 0.15 + 0.85 * (tt - a) / max(t - a, 1)
        ax.plot(st[tt, b, 0], st[tt, b, 1], ".", color="#1f77b4", alpha=al, ms=5, zorder=3)
        ax.plot(op[tt, b, 0], op[tt, b, 1], ".", color=RL.C_CAR, alpha=al, ms=5, zorder=3)
    for tt, al in ((t - n10, 0.25), (t - n05, 0.5), (t, 0.9)):
        q = R.body(*st[tt, b, :3]); ax.fill(q[:, 0], q[:, 1], color="#1f77b4", alpha=al, zorder=4)
        q = R.body(*op[tt, b, :3]); ax.fill(q[:, 0], q[:, 1], color=RL.C_CAR, alpha=al, zorder=4)
    for tt, c in ((t - n10, "k"), (t - n05, "#e377c2")):
        p = np.concatenate([st[tt, b, :2][None], R.pl[tt, b]], 0); ax.plot(p[:, 0], p[:, 1], "-o", color=c, ms=3, lw=1.5, zorder=6)
    ax.set_xlim(st[t, b, 0] - 3.0, st[t, b, 0] + 3.0); ax.set_ylim(st[t, b, 1] - 3.0, st[t, b, 1] + 3.0)
    ax.set_title(f"차 {b}, 경주 {((t - R.k0[t, b]) * dt):.1f} s째 ({RL.trk_name(tk)}) — 상대 {e['kind']} (속도 배율 {float(R.scale[t, b]):.2f})\n"
                 f"닿을 때 우리 {e['v_us0']:.1f} · 상대 {e['v_op0']:.1f} m/s, 상대가 우리 앞 {e['ahead0']:.2f} m · 옆 {e['left0']:+.2f} m", fontsize=10)
    a3 = max(t - n30, int(R.k0[t, b])); e3 = min(t + n05, T - 1)
    tt = np.arange(a3, e3 + 1); x = (tt - t) * dt
    ah, lf = np.array([rel(k, b) for k in tt]).T
    reach = np.linalg.norm(R.pl[tt, b, 9] - st[tt, b, :2], axis=1)
    a2.plot(x, st[tt, b, 3], color="#1f77b4", lw=2, label="우리 속도 (m/s)")
    a2.plot(x, op[tt, b, 3], color=RL.C_CAR, lw=2, label="상대 속도 (m/s)")
    a2.plot(x, reach, color="#1f77b4", lw=1, ls=":", label="plan 1.0 s 점까지의 거리 (m) = plan이 요구하는 평균 속도")
    a2.plot(x, np.clip(ah, -3, 6), color="#2ca02c", lw=1.5, label="상대가 우리 앞으로 몇 m")
    a2.plot(x, np.clip(lf, -3, 3), color="#ff7f0e", lw=1.5, label="상대가 우리 왼쪽으로 몇 m (음수 = 오른쪽)")
    a2.axhline(0, color="k", lw=0.5); a2.axvline(0, color="#d62728", lw=0.8); a2.set_xlabel("닿기까지 시간 (s)")
    a2.set_ylim(-3, 7); a2.legend(fontsize=9, loc="upper left"); a2.grid(alpha=0.3)
    a2.set_title(f"1.0 s 전: 상대가 앞 {e['ahead1']:.2f} m · 옆 {e['left1']:+.2f} m ({'앞 ±20도 안' if e['in_view'] else '앞 ±20도 밖'}), "
                 f"우리 {e['v_us1']:.1f} · 상대 {e['v_op1']:.1f} m/s", fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=60); print("그림", sys.argv[2])
