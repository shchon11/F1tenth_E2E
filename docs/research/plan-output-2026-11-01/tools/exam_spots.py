"""시험 기록(exam_record.py npz)에서 세게 닿은 자리와 그 장면. 숫자보다 먼저 볼 것.
그림 1 (앞머리_spots.png): 경우 4개(혼자·빈 트랙 / 혼자·상자 / 상대 차·빈 트랙 / 상대 차·상자)마다 지도 위에 세게 닿은 자리
(x = 벽, 네모 = 상자, 세모 = 차; 달리다 처음 닿은 것만 진하게, 앞선 접촉 3 s 안에 이어진 것은 흐리게), 검은 동그라미 = 2 s 넘게 선 자리.
그림 2 (앞머리_wall.png): 달리다 벽에 세게 닿은 건이 가장 많이 몰린 자리(1 m 칸) 몇 곳의 확대 장면. 장면마다 왼쪽 = ±2.5 m:
닿기 2 s 전부터 1 s 뒤까지 차의 길(점, 진할수록 최근), 닿기 1.0 s 전(검정)·0.5 s 전(분홍) plan, 닿는 순간 차체(빨강).
오른쪽 = 닿기 3 s 전부터의 속도와, plan(1.0 s 전 것)의 점과 그 시각 실제 자리의 거리.
출력(글): 달리다 벽에 세게 닿은 건의 자리별 몫, 닿기 직전 속도, plan과 실제의 거리.
인자: 기록npz 출력앞머리 [몰린 자리 수 3] [자리마다 장면 수 3]"""
import sys
import numpy as np
import rec_lib as RL
from rec_lib import Rec, plt

R = Rec(sys.argv[1]); z = R.z; T, B, dt = R.T, R.B, R.dt
n_spot = int(sys.argv[3]) if len(sys.argv) > 3 else 3
n_per = int(sys.argv[4]) if len(sys.argv) > 4 else 3
st = R.st.astype(float); xy = st[:, :, :2]; V = np.abs(st[:, :, 3])
box = (R.sid[0] >= 0).any(-1); opp = R.on[:T].any(0)
hard = R.d[:, :, 3:6] > 0
n05, n10, n20, n30 = (int(round(x / dt)) for x in (0.5, 1.0, 2.0, 3.0))
MODE = dict(map="MAP", track="시간 추종기", ideal="plan대로 정확히")[str(z["mode"])]
any_h = hard.any(-1)
# a hit "while driving": no hard contact in the 3 s before it and faster than 2 m/s one second before
recent = np.zeros((T, B), bool)
for k in range(1, n30 + 1):
    recent[k:] |= any_h[:-k]
fresh = np.zeros((T, B), bool); fresh[n10:] = (V[:T][:-n10] > 2.0)[: T - n10]
fresh &= ~recent
slow = V[:T] < 0.3; slow[:n10] = False
run = np.zeros((T, B), int)
for t in range(1, T):
    run[t] = np.where(slow[t], run[t - 1] + 1, 0)
stall = run == n20
MK = (("x", "벽"), ("s", "상자"), ("^", "차"))

fig, axs = plt.subplots(2, 2, figsize=(16, 14))
for ax, (label, m) in zip(axs.ravel(), (("혼자 · 빈 트랙", ~box & ~opp), ("혼자 · 상자", box & ~opp), ("상대 차 · 빈 트랙", ~box & opp), ("상대 차 · 상자", box & opp))):
    R.draw_map(ax, 0); ids = np.nonzero(m)[0]; txt = []
    for j, (mk, nm) in enumerate(MK):
        for fr, al, ms in ((False, 0.25, 6), (True, 0.9, 9)):
            t_, b_ = np.nonzero(hard[:, ids, j] & (fresh[:, ids] == fr))
            if len(t_):
                ax.plot(xy[t_, ids[b_], 0], xy[t_, ids[b_], 1], mk, color="k" if j == 0 else (RL.C_BOX if j == 1 else RL.C_CAR), mfc="none",
                        ms=ms, mew=1.5, alpha=al, ls="none", zorder=5)
        txt.append(f"{nm} {int((hard[:, ids, j] & fresh[:, ids]).sum())}(+이어진 {int((hard[:, ids, j] & ~fresh[:, ids]).sum())})")
    t_, b_ = np.nonzero(stall[:, ids])
    ax.plot(xy[t_, ids[b_], 0], xy[t_, ids[b_], 1], "o", color="k", mfc="none", ms=15, mew=1.2, ls="none", zorder=6)
    ax.set_title(f"{label} — 세션 {len(ids)}개 x {T * dt:g} s | 달리다 세게 닿음: {' · '.join(txt)} | 2 s 넘게 섬 {len(t_)}번", fontsize=11)
fig.suptitle(f"{z['exam']} · {MODE} · {R.name} · 상자 {z['count']} — 세게 닿은 자리 (진한 표시 = 달리다 처음 닿음, 흐린 표시 = 3 s 안에 이어진 접촉, 동그라미 = 선 자리)", fontsize=13, y=0.995)
fig.tight_layout(); fig.savefig(f"{sys.argv[2]}_spots.png", dpi=60); plt.close(fig); print("그림", f"{sys.argv[2]}_spots.png")

# where the wall hits while driving pile up (1 m cells), all sessions
t_, b_ = np.nonzero(hard[:, :, 0] & fresh)
print(f"{z['exam']} · {MODE} · {R.name}: 달리다 벽에 세게 닿음 {len(t_)}건 (이어진 접촉 {int((hard[:, :, 0] & ~fresh).sum())}건은 뺌), 세션 {B}개")
if len(t_) == 0:
    sys.exit(0)
cell = np.floor(xy[t_, b_] / 1.0).astype(int)
uc, inv, cnt = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
order = np.argsort(-cnt)


def plan_gap(t, b):
    """[m] between the points of the plan made 1.0 s before decision t and where the car was at those times (0.5 s, 1.0 s)"""
    t0 = t - n10
    if t0 < 0:
        return np.nan, np.nan
    return (float(np.linalg.norm(R.pl[t0, b, n05 - 1] - xy[t0 + n05, b])), float(np.linalg.norm(R.pl[t0, b, n10 - 1] - xy[t0 + n10, b])))


for r_, c in enumerate(order[:8]):
    me = np.nonzero(inv == c)[0]
    g = np.array([plan_gap(t_[i], b_[i]) for i in me])
    print(f"  자리 {r_ + 1}: ({uc[c][0] + 0.5:+.1f}, {uc[c][1] + 0.5:+.1f}) m 둘레 1 m 칸 — {cnt[c]}건 = {100 * cnt[c] / len(t_):.0f} % (세션 {len(set(b_[me]))}개) | "
          f"닿기 1 s 전 속도(중앙) {np.median(V[t_[me] - n10, b_[me]]):.1f} m/s | 1.0 s 전 plan과 실제 자리의 거리(중앙): 0.5 s 뒤 "
          f"{100 * np.nanmedian(g[:, 0]):.0f} cm · 1.0 s 뒤 {100 * np.nanmedian(g[:, 1]):.0f} cm")
ga = np.array([plan_gap(t, b) for t, b in zip(t_, b_)])
print(f"  전체: 가장 많은 3곳이 {100 * cnt[order[:3]].sum() / len(t_):.0f} %, 8곳이 {100 * cnt[order[:8]].sum() / len(t_):.0f} % | plan과 실제 자리의 거리(중앙): "
      f"0.5 s 뒤 {100 * np.nanmedian(ga[:, 0]):.0f} cm · 1.0 s 뒤 {100 * np.nanmedian(ga[:, 1]):.0f} cm")
# the same distance over all normal driving, for comparison (every 7th decision, faster than 2 m/s, no hit within 3 s)
tt, bb = np.nonzero(~recent[n10:T - n10:7] & ~any_h[n10:T - n10:7] & (V[n10:T - n10:7] > 2.0))
tt = tt * 7 + n10
gn = np.array([plan_gap(t + n10, b) for t, b in zip(tt[:20000], bb[:20000])])
print(f"  비교(닿지 않은 보통 주행 {len(gn)}곳): plan과 실제 자리의 거리(중앙) 0.5 s 뒤 {100 * np.nanmedian(gn[:, 0]):.0f} cm · 1.0 s 뒤 {100 * np.nanmedian(gn[:, 1]):.0f} cm "
      f"(90 %: {100 * np.nanpercentile(gn[:, 0], 90):.0f} / {100 * np.nanpercentile(gn[:, 1], 90):.0f} cm)")

pick = []
for c in order[:n_spot]:
    me = np.nonzero(inv == c)[0]
    pick += [(int(t_[i]), int(b_[i]), int(np.nonzero(order == c)[0][0]) + 1) for i in me[np.linspace(0, len(me) - 1, min(n_per, len(me))).round().astype(int)]]
fig, axs = plt.subplots(len(pick), 2, figsize=(15, 5.0 * len(pick)), gridspec_kw={"width_ratios": [1, 1.4]}, squeeze=False)
for (t, b, rank), (ax, a2) in zip(pick, axs):
    R.draw_map(ax, 0)
    for q in R.boxes(t, b):
        ax.fill(q[:, 0], q[:, 1], color=RL.C_BOX, zorder=2)
    a, e = max(t - n20, 0), min(t + n10, T)
    for k in range(a, e + 1):
        ax.plot(xy[k, b, 0], xy[k, b, 1], ".", color="#1f77b4", alpha=0.15 + 0.85 * (k - a) / max(e - a, 1), ms=6, zorder=3)
    for k, col in ((t - n10, "k"), (t - n05, "#e377c2")):
        if k >= 0:
            p = np.concatenate([xy[k, b][None], R.pl[k, b]], 0); ax.plot(p[:, 0], p[:, 1], "-o", color=col, ms=3, lw=1.5, zorder=6)
            q = R.body(*st[k, b, :3]); ax.fill(q[:, 0], q[:, 1], color=col, alpha=0.25, zorder=4)
    q = R.body(*st[t, b, :3]); ax.fill(q[:, 0], q[:, 1], color="#d62728", alpha=0.8, zorder=5)
    if R.on[t, b]:
        q = R.body(*R.op[t, b, :3].astype(float)); ax.fill(q[:, 0], q[:, 1], color="#8c564b", alpha=0.6, zorder=5)
    ax.set_xlim(xy[t, b, 0] - 2.5, xy[t, b, 0] + 2.5); ax.set_ylim(xy[t, b, 1] - 2.5, xy[t, b, 1] + 2.5)
    case = ("상대 차" if opp[b] else "혼자") + " · " + ("상자" if box[b] else "빈 트랙")
    ax.set_title(f"자리 {rank} — 세션 {b} ({case}), {t * dt:.1f} s째 | 검정 = 1.0 s 전 plan, 분홍 = 0.5 s 전 plan, 빨강 = 닿는 순간", fontsize=10)
    w = np.arange(max(t - n30, 0), min(t + n10, T) + 1); x = (w - t) * dt
    a2.plot(x, V[w, b], color="#1f77b4", lw=2, label="속도 (m/s)")
    k0 = t - n10
    if k0 >= 0:
        gap = [np.linalg.norm(R.pl[k0, b, i] - xy[k0 + i + 1, b]) for i in range(n10)]
        a2.plot((np.arange(n10) + 1) * dt - 1.0, np.array(gap) * 10, "-o", color="k", ms=3, lw=1.2, label="1.0 s 전 plan의 점과 그 시각 실제 자리의 거리 (x 10 cm)")
    a2.axvline(0, color="#d62728", lw=0.8); a2.set_ylim(0, 8.5); a2.grid(alpha=0.3); a2.legend(fontsize=9, loc="upper left")
    a2.set_xlabel("닿기까지 시간 (s)")
    g = plan_gap(t, b)
    a2.set_title(f"닿기 1 s 전 속도 {V[max(t - n10, 0), b]:.1f} m/s | 1.0 s 전 plan과 실제: 0.5 s 뒤 {100 * g[0]:.0f} cm · 1.0 s 뒤(닿는 순간) {100 * g[1]:.0f} cm", fontsize=10)
fig.suptitle(f"{z['exam']} · {MODE} · {R.name} — 달리다 벽에 세게 닿은 건이 몰린 자리 {n_spot}곳의 장면", fontsize=12, y=1.0)
fig.tight_layout(); fig.savefig(f"{sys.argv[2]}_wall.png", dpi=60); print("그림", f"{sys.argv[2]}_wall.png")
