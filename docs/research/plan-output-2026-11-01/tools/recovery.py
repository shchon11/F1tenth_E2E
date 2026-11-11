"""세게 닿은 뒤 다시 달리기까지 (11-12 사용자: 장애물에 닿고 뒤로 갔다가 다시 출발하는 것에 대한 분석이 빠져 있음). drive_record.py 기록에서
학습 차가 벽·상자·상대 차에 세게 닿은 건마다(같은 차가 1 s 안에 또 닿으면 한 건으로 묶음): 닿은 곳에서 3 m 벗어나기까지 걸린 시간, 그동안
후진했나(-0.2 m/s 아래), 서 있던 시간(0.3 m/s 아래), 또 세게 닿은 횟수, 잃은 시간(걸린 시간 - 닿기 전 속도로 3 m 가는 시간). 기록이 끝나
끝을 못 본 건은 '못 벗어남'으로 따로 셈. 상대 차도 같은 식으로(상대 차의 접촉 기록) 나란히. 장면: 오래 걸린 순 12장 + 중간쯤 6장
(파랑 = 전진, 빨강 = 후진 자취, 노란 별 = 닿은 곳, 갈색 = 상자, 초록 = 상대 차의 그때 위치).
인자: drive_record npz [장면 png]"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rec_lib as RL
plt = RL.plt

R = RL.Rec(sys.argv[1]); z = R.z
dt, T, B = R.dt, R.T, R.B
KN = ("벽", "상자", "상대 차")


def incidents(st, cnt, rs, alive=None):
    v = st[:T, :, 3].astype(float); xy = st[:T, :, :2].astype(float)
    prev = np.concatenate([np.zeros((1, B, 6), np.float32), cnt[:-1]], 0)
    prev[1:][rs[:-1]] = 0.0
    d = np.clip(cnt - prev, 0, None) * (~rs)[..., None]
    rows = []
    for b in range(B):
        ev = np.nonzero(d[:, b, 3:6].sum(1) > 0.5)[0]
        if alive is not None:
            ev = ev[alive[ev, b]]
        i = 0
        while i < len(ev):
            t0 = int(ev[i])
            if t0 < 10:
                i += 1; continue
            kind = int(d[t0, b, 3:6].argmax())
            nx = np.nonzero(rs[t0:, b])[0]; t_end = t0 + int(nx[0]) if len(nx) else T - 1
            disp = np.linalg.norm(xy[t0: t_end + 1, b] - xy[t0, b], axis=1)
            far = np.nonzero(disp >= 3.0)[0]
            t1 = t0 + int(far[0]) if len(far) else t_end
            cleared = len(far) > 0
            seg = v[t0: t1 + 1, b]
            v0 = float(np.abs(v[t0 - 5: t0, b]).mean())
            rehit = int((d[t0 + 1: t1 + 1, b, 3:6].sum(1) > 0.5).sum())
            rows.append((b, t0, kind, (t1 - t0) * dt, float(cleared), float((seg < -0.2).any()), float((np.abs(seg) < 0.3).sum() * dt), rehit, v0,
                         (t1 - t0) * dt - 3.0 / max(v0, 1.0), float((seg < -0.2).sum() * dt)))
            while i < len(ev) and ev[i] <= t1:
                i += 1
    return np.array(rows, float).reshape(-1, 11)


def table(E, mins, who):
    print(f"[{who}] {mins:.0f}대·분, 세게 닿은 건 {len(E)} ({len(E) / mins:.2f}/분)")
    lost_all = 0.0
    for k in range(3):
        e = E[E[:, 2] == k]
        if len(e) == 0:
            print(f"  {KN[k]} | 0건"); continue
        c = e[e[:, 4] > 0.5]; un = e[e[:, 4] < 0.5]
        lost_all += e[:, 9].clip(min=0).sum()
        q = (lambda p: float(np.percentile(c[:, 3], p))) if len(c) else (lambda p: float("nan"))
        print(f"  {KN[k]} | {len(e)}건 ({len(e) / mins:.2f}/분) | 3 m 벗어나기까지: 중앙 {q(50):.1f} s, 75 % {q(75):.1f} s, 90 % {q(90):.1f} s | 3 s 넘게 걸림 "
              f"{((e[:, 3] > 3.0) | (e[:, 4] < 0.5)).mean() * 100:.0f} % | 기록 끝까지 못 벗어남 {len(un)}건 (그때까지 중앙 {np.median(un[:, 3]) if len(un) else 0:.0f} s) | "
              f"후진함 {e[:, 5].mean() * 100:.0f} % (후진 시간 중앙 {np.median(e[e[:, 5] > 0.5, 10]) if (e[:, 5] > 0.5).any() else 0:.1f} s) | 서 있던 시간 중앙 "
              f"{np.median(e[:, 6]):.1f} s | 벗어나기 전에 또 세게 닿음 {(e[:, 7] > 0).mean() * 100:.0f} % | 닿기 전 속도 중앙 {np.median(e[:, 8]):.1f} m/s | "
              f"잃은 시간 중앙 {np.median(e[:, 9]):.1f} s, 합 {e[:, 9].clip(min=0).sum():.0f} s")
    print(f"  세게 닿은 뒤 잃은 시간 합 {lost_all:.0f} s = 주행 시간의 {lost_all / (mins * 60) * 100:.1f} %")


E = incidents(R.st, R.cnt, R.rs)
print(f"{R.name}: 학습 차 {B}대 x {T * dt:.0f} s")
table(E, B * T * dt / 60.0, "우리 차")
if "ocnt" in z.files:
    on = z["on"][:T]
    Eo = incidents(R.op, z["ocnt"], R.rs, alive=on)
    table(Eo, on.sum() * dt / 60.0, "상대 차")
# standing or wedged without getting anywhere: less than 0.5 m of net movement over the next 2 s
xy = R.st[:T, :, :2].astype(float); n2 = int(2.0 / dt)
net = np.linalg.norm(xy[n2:] - xy[:-n2], axis=2)
ok = np.ones((T - n2, B), bool)
for b in range(B):
    for t in np.nonzero(R.rs[:, b])[0]:
        ok[max(t - n2 + 1, 0): t + 1, b] = False
stuck = (net < 0.5) & ok
print(f"[우리 차] 2 s 동안 0.5 m도 못 간 시간 {stuck.sum() / max(ok.sum(), 1) * 100:.1f} %", end="")
if "ocnt" in z.files:
    oxy = R.op[:T, :, :2].astype(float); onet = np.linalg.norm(oxy[n2:] - oxy[:-n2], axis=2); oo = ok & on[:-n2] & on[n2:]
    print(f" | [상대 차] {((onet < 0.5) & oo).sum() / max(oo.sum(), 1) * 100:.1f} %", end="")
print()
if len(sys.argv) > 2 and len(E):
    order = np.argsort(-(E[:, 3] + 100 * (E[:, 4] < 0.5)))
    mid = order[len(order) // 2 - 3: len(order) // 2 + 3]
    pick = list(order[:12]) + list(mid)
    fig, axs = plt.subplots(3, 6, figsize=(24, 12.5))
    for ax, i in zip(axs.ravel(), pick):
        b, t0, k = int(E[i, 0]), int(E[i, 1]), int(E[i, 2]); t1 = min(t0 + int(round(E[i, 3] / dt)), T - 1)
        tr = int(R.tid[t0, b]); R.draw_map(ax, tr)
        a0 = max(t0 - 15, 0); seg = R.st[a0: t1 + 1, b]
        for q in R.boxes(t0, b):
            ax.fill(q[:, 0], q[:, 1], color=RL.C_BOX, zorder=2)
        if R.on[t0, b]:
            o = R.op[t0, b]; q = R.body(o[0], o[1], o[2]); ax.fill(q[:, 0], q[:, 1], color="#2ca02c", alpha=0.8, zorder=3)
            o = R.op[t1, b]; q = R.body(o[0], o[1], o[2]); ax.fill(q[:, 0], q[:, 1], color="#2ca02c", alpha=0.3, zorder=3)
        ax.plot(seg[:, 0], seg[:, 1], color="#888888", lw=0.8, zorder=3)
        fw = seg[:, 3] >= -0.2
        ax.scatter(seg[fw, 0], seg[fw, 1], s=9, color="#1f77b4", zorder=4); ax.scatter(seg[~fw, 0], seg[~fw, 1], s=14, color="#d62728", zorder=5)
        for tt, al in ((t0, 0.9), (t1, 0.4)):
            q = R.body(*R.st[tt, b, :3]); ax.fill(q[:, 0], q[:, 1], color="#1f77b4", alpha=al, zorder=4)
        ax.plot(*R.st[t0, b, :2], marker="*", color="#ffd400", ms=16, mec="k", zorder=6)
        c = seg[:, :2].mean(0); h = max(np.ptp(seg[:, 0]), np.ptp(seg[:, 1]), 3.0) / 2 + 0.8
        ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h)
        ax.set_title(f"{KN[k]}에 세게 닿음, 닿기 전 {E[i, 8]:.1f} m/s | " + (f"{E[i, 3]:.1f} s 뒤 벗어남" if E[i, 4] > 0.5 else f"{E[i, 3]:.0f} s 지나도 못 벗어남")
                     + f"\n후진 {E[i, 10]:.1f} s, 서 있음 {E[i, 6]:.1f} s, 또 닿음 {int(E[i, 7])}번", fontsize=10)
    fig.suptitle(f"{R.name}: 세게 닿은 뒤 다시 달리기까지 — 윗 두 줄 = 오래 걸린 순 12건, 아랫줄 = 중간쯤 6건 (파랑 점 = 전진, 빨강 점 = 후진, 별 = 닿은 곳, 초록 = 상대 차)", fontsize=13)
    fig.tight_layout(); fig.savefig(sys.argv[2], dpi=70); print("그림", sys.argv[2])
