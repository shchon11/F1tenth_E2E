"""기록(MAP 등)에서: plan의 길과 실제로 달린 길이 어긋나는 모습. 위 = 벽 끝에서 세게 닿은 장면 3개(닿기 전 0.8 s 동안 0.1 s마다 낸 plan과
실제로 달린 길), 아래 = 옆가속도별 (왼) 차체 방향 대비 실제 진행 방향·plan 출발 방향 (오) 0.3 s 뒤 plan의 길에서 벗어난 거리.
인자: npz 출력png [벽 끝 칸 x y, 기본 -1.5 -1.5]"""
import sys
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
import rec_lib as RL
from rec_lib import Rec, plt
R = Rec(sys.argv[1]); T, B, dt = R.T, R.B, R.dt
cx, cy = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (-1.5, -1.5)
st = R.st.astype(float); xy = st[:, :, :2]; yaw = st[:, :, 2]; V = np.abs(st[:, :, 3]); pl = R.pl.astype(float)
hard = R.d[:, :, 3:6] > 0; any_h = hard.any(-1)
recent = np.zeros((T, B), bool); soon = np.zeros((T, B), bool)
for k in range(1, 31):
    recent[k:] |= any_h[:-k]
for k in range(1, 11):
    soon[:-k] |= any_h[k:]
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
ang = lambda d: np.arctan2(d[..., 1], d[..., 0])
dlt = R.z["dlt"].astype(float) if "dlt" in R.z.files else np.zeros((T, B))     # the frame the plan was drawn in (slip-frame records)
MODE = dict(map="MAP", track="시간 추종기", ideal="plan대로 정확히")[str(R.z["mode"])]

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1.0])
# ---- scenes: fresh hard wall hits while driving, in the 1 m cell round (cx, cy)
ev = hard[:, :, 0] & ~recent & (V[:T] > 2.0)
t_e, b_e = np.nonzero(ev)
m = (np.abs(xy[t_e, b_e, 0] - cx) < 0.5) & (np.abs(xy[t_e, b_e, 1] - cy) < 0.5) & (t_e >= 10)
t_e, b_e = t_e[m], b_e[m]
print(f"{MODE}: 벽 끝 칸({cx:g}, {cy:g})에서 달리다 세게 닿음 {len(t_e)}건")
pick = np.linspace(0, len(t_e) - 1, min(3, len(t_e))).round().astype(int) if len(t_e) else []
for q, i in enumerate(pick):
    ax = fig.add_subplot(gs[0, 2 * q:2 * q + 2]); t, b = int(t_e[i]), int(b_e[i]); tk = int(R.tid[0, b])
    R.draw_map(ax, tk)
    for bx in R.boxes(t, b):
        ax.fill(bx[:, 0], bx[:, 1], color=RL.C_BOX, zorder=2)
    w = np.arange(t - 10, min(t + 3, T) + 1)
    for n_, tt in enumerate(range(t - 8, t)):
        p = np.concatenate([xy[tt, b][None], pl[tt, b, :6]], 0)
        ax.plot(p[:, 0], p[:, 1], "-", color=plt.cm.autumn_r(0.25 + 0.75 * n_ / 7), lw=1.3, zorder=5,
                label="0.1 s마다 낸 plan (앞 0.6 s; 진할수록 나중)" if n_ == 7 else None)
        ax.plot(p[0, 0], p[0, 1], "o", color="k", ms=2.5, zorder=7)
    ax.plot(xy[w, b, 0], xy[w, b, 1], "-", color="#1f77b4", lw=2.4, zorder=6, label="실제로 달린 길")
    for tt, al in ((t - 8, 0.2), (t - 4, 0.35), (t, 0.6)):
        bd = R.body(*st[tt, b, :3]); ax.fill(bd[:, 0], bd[:, 1], color="#1f77b4", alpha=al, zorder=4)
    c = 0.5 * (xy[t - 5, b] + xy[t, b])
    ax.set_xlim(c[0] - 1.9, c[0] + 1.9); ax.set_ylim(c[1] - 1.9, c[1] + 1.9); ax.set_aspect("equal")
    ax.set_title(f"세션 {b}, {t * dt:.1f} s째 벽에 세게 닿음 (속도 {V[t - 3, b]:.1f} m/s)", fontsize=10)
    if q == 0:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
# ---- normal driving, by lateral acceleration
ok = (V[:T] > 2.0) & (st[:T, :, 3] > 0) & ~recent & ~any_h & ~soon; ok[T - 10:] = False
t_, b_ = np.nonzero(ok); sel = np.random.default_rng(0).choice(len(t_), min(80000, len(t_)), replace=False); t_, b_ = t_[sel], b_[sel]
p0, p1 = xy[t_, b_], xy[t_ + 1, b_]; y0, y1 = yaw[t_, b_], yaw[t_ + 1, b_]
r = wrap(y1 - y0) / dt; v = V[t_, b_]; alat = np.abs(v * r); sg = np.sign(r)
chord = ang(p1 - p0)
beta_real = wrap(chord - (y0 + 0.5 * wrap(y1 - y0))) * sg
plan_dir = wrap(ang(pl[t_, b_, 0] - p0) - (y0 + 0.5 * wrap(y1 - y0))) * sg
path = np.concatenate([p0[:, None], pl[t_, b_]], 1); a, bb = path[:, :-1], path[:, 1:]; ab = bb - a
p3 = xy[t_ + 3, b_]
u = np.clip(((p3[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
off = p3[:, None] - (a + u[..., None] * ab); dist = np.linalg.norm(off, axis=-1); j = dist.argmin(1); n = np.arange(len(p3))
lat3 = np.sign(ab[n, j, 0] * off[n, j, 1] - ab[n, j, 1] * off[n, j, 0]) * dist[n, j] * sg
edges = [0, 1, 2, 3, 4, 5, 6, 7, 9]
mid, rows = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    mm = (alat >= lo) & (alat < hi)
    if mm.sum() < 200:
        continue
    mid.append(0.5 * (lo + hi))
    rows.append([np.percentile(x[mm], q_) for x in (np.degrees(beta_real), np.degrees(plan_dir), 100 * lat3) for q_ in (10, 50, 90)])
mid, rows = np.array(mid), np.array(rows)
a1 = fig.add_subplot(gs[1, 0:3]); a2 = fig.add_subplot(gs[1, 3:6])
for k, (col, lab) in enumerate((("#1f77b4", "실제 진행 방향 (0.1 s 동안 간 방향)"), ("#d95f02", "plan의 첫 0.1 s 방향"))):
    a1.fill_between(mid, rows[:, 3 * k], rows[:, 3 * k + 2], color=col, alpha=0.18, lw=0)
    a1.plot(mid, rows[:, 3 * k + 1], "-o", color=col, lw=2, ms=5, label=lab)
a1.axhline(0, color="k", lw=0.8); a1.grid(alpha=0.3); a1.legend(fontsize=10, loc="upper left")
a1.set_xlabel("옆가속도 (m/s²) = 속도 × 회전 속도"); a1.set_ylabel("차체 방향에서 잰 각 (°; + = 도는 쪽 안쪽, − = 바깥쪽)")
a1.set_title("차체 방향을 0°로 했을 때 실제 진행 방향과 plan의 출발 방향 (선 = 중앙, 띠 = 10~90 %)"
             + (" — policy가 진행 방향 기준으로 보게 한 실행" if "dlt" in R.z.files else ""), fontsize=10)
a2.fill_between(mid, rows[:, 6], rows[:, 8], color="#1f77b4", alpha=0.18, lw=0)
a2.plot(mid, rows[:, 7], "-o", color="#1f77b4", lw=2, ms=5)
a2.axhline(0, color="k", lw=0.8); a2.axhline(-100 * R.hw, color="#888888", ls=":", lw=1); a2.grid(alpha=0.3)
a2.text(mid[0], -100 * R.hw - 0.6, "차 반폭 15.5 cm", fontsize=9, color="#666666", va="top")
a2.set_xlabel("옆가속도 (m/s²)"); a2.set_ylabel("0.3 s 뒤 차의 자리 − plan의 길 (cm; − = 바깥쪽)")
a2.set_title("0.3 s 뒤 차가 plan의 길에서 벗어난 거리 (선 = 중앙, 띠 = 10~90 %)", fontsize=10)
fig.suptitle(f"{R.z['exam']} · {MODE} · {R.name} — plan의 길과 실제로 달린 길 (보통 주행 {len(t_)}개 결정)", fontsize=12, y=0.995)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=70); print("그림", sys.argv[2])
for k_, m_ in enumerate(mid):
    print(f"  옆가속도 {m_:.1f}: 실제 진행 {rows[k_, 1]:+.1f}°, plan 출발 {rows[k_, 4]:+.1f}°, 0.3 s 뒤 {rows[k_, 7]:+.1f} cm")
