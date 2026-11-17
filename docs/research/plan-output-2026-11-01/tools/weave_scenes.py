"""(11-18, the user: "0.3보다 빠르게 와리가리치는장면들이 종종 보이네") 제자리 와리가리 장면: 2 s 동안 트랙을 따라 1 m도 못 가면서
|v| 평균 > 0.3 m/s이고 좌우(요 각속도 부호가 3번 넘게 바뀜) 또는 앞뒤(속도 부호가 2번 넘게 바뀜)로 흔드는 구간.
frozen exam 기록(exam_record.py npz)에서 그런 구간을 세고, 세션마다 처음 것을 그림: 왼쪽 = 상자 둘레 지도에 그 구간 앞뒤 3 s의 길(시간 색),
차체(구간 시작·끝), 그 구간의 plan들(회색), 세게 닿은 곳(x); 오른쪽 = 속도·요 각속도 시간축(구간 음영, 세게 닿은 시각 세로선).
인자: 기록npz 출력png [장면 수 6]
"""
import sys

import numpy as np

import rec_lib as RL  # noqa: F401
from rec_lib import Rec, plt

R = Rec(sys.argv[1]); z = R.z
out, n_show = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6
st, rs, rem, pl = z["st"].astype(float), z["rs"], z["rem"].astype(float), z["pl"].astype(float)
T, B = rs.shape; dt = float(z["dt"]); L = float(z["lapL"][0])
W = int(round(2.0 / dt))
yaw = np.unwrap(st[:T + 1, :, 2], axis=0); yr = np.diff(yaw, axis=0) / dt
v = st[:T, :, 3]
rr = rem.copy()
for t in range(1, rr.shape[0]):
    bad = ~np.isfinite(rr[t]); rr[t, bad] = rr[t - 1, bad]
rr = np.nan_to_num(rr)
hard = R.d[..., 3:6].sum(-1)                                   # (T, B) hard contacts per decision


def flips(x, thr):
    s = np.sign(np.where(np.abs(x) > thr, x, 0)); s = s[s != 0]
    return int((s[1:] != s[:-1]).sum()) if len(s) > 1 else 0


found = []
for b in range(B):
    for t0 in range(0, T - W, 5):
        t1 = t0 + W
        if rs[t0:t1, b].any():
            continue
        prog = np.mod(rr[t0, b] - rr[t1, b] + 0.5 * L, L) - 0.5 * L
        if prog < 1.0 and np.abs(v[t0:t1, b]).mean() > 0.3 and (flips(yr[t0:t1, b], 0.3) >= 3 or flips(v[t0:t1, b], 0.3) >= 2):
            found.append((b, t0, t1)); break
print(f"제자리 와리가리 세션 {len(found)}개")
if not found:
    sys.exit(0)
pick = [found[i] for i in np.linspace(0, len(found) - 1, min(n_show, len(found))).round().astype(int)]
fig, axs = plt.subplots(len(pick), 2, figsize=(14, 4.6 * len(pick)), gridspec_kw={"width_ratios": [1, 1.5]}, squeeze=False)
g, ext = R.grid(0)
fp = {int(k[3:]): z[k].astype(float) for k in z.files if k.startswith("fp_")}
for (b, t0, t1), (ax, a2) in zip(pick, axs):
    a, e = max(0, t0 - 30), min(T, t1 + 30)
    ax.imshow((g <= 0).astype(float), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
    for j in range(z["sid"].shape[2]):
        k = int(z["sid"][t0, b, j])
        if k >= 0 and k in fp:
            x, y, th = z["pose"][t0, b, j]
            c, s = np.cos(th), np.sin(th)
            q = np.array([[x + p * c - r * s, y + p * s + r * c] for p, r in fp[k]])
            ax.fill(q[:, 0], q[:, 1], color="#8b5a2b")
    ax.scatter(st[a:e, b, 0], st[a:e, b, 1], c=np.arange(a, e), cmap="viridis", s=10, zorder=3)
    for t in range(t0, t1, 3):
        ax.plot(pl[t, b, :, 0], pl[t, b, :, 1], color="0.55", lw=0.8, zorder=2)
    for t in (t0, t1 - 1):
        x, y, th = st[t, b, :3]; c, s = np.cos(th), np.sin(th)
        q = np.array([[x + p * c - r * s, y + p * s + r * c] for p, r in ((0.29, 0.155), (0.29, -0.155), (-0.29, -0.155), (-0.29, 0.155), (0.29, 0.155))])
        ax.plot(q[:, 0], q[:, 1], color="#1f77b4", lw=1.5, zorder=4)
    th_ = np.nonzero(hard[a:e, b] > 0)[0] + a
    ax.scatter(st[th_, b, 0], st[th_, b, 1], marker="x", c="red", s=60, zorder=5)
    cx, cy = st[t0:t1, b, 0].mean(), st[t0:t1, b, 1].mean()
    ax.set_xlim(cx - 2.5, cx + 2.5); ax.set_ylim(cy - 2.5, cy + 2.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"세션 {b}, {t0 * dt:.1f}-{t1 * dt:.1f} s (앞뒤 3 s, 색 = 시간)", fontsize=10)
    tt = np.arange(a, e) * dt
    a2.plot(tt, v[a:e, b], label="속도 [m/s]"); a2.plot(tt, yr[a:e, b], label="요 각속도 [rad/s]")
    a2.axvspan(t0 * dt, t1 * dt, color="0.85"); a2.axhline(0.3, color="0.5", ls=":"); a2.axhline(-0.3, color="0.5", ls=":")
    for t in th_:
        a2.axvline(t * dt, color="red", lw=1)
    a2.legend(fontsize=8, loc="upper right"); a2.grid(alpha=0.3); a2.set_xlabel("시각 [s]")
fig.suptitle(f"{sys.argv[1].split('/')[-1]}: 2 s 동안 1 m도 못 가며 좌우·앞뒤로 흔든 구간 (세션마다 처음 것)", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=90); print(out)
