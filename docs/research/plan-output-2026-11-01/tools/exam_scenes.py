"""시험 기록(exam_record.py npz)의 세션을 그림으로: 숫자(exam_table.py)보다 먼저 볼 것.
세션마다 왼쪽 = 지도 전체: 상자(갈색, 치우기 전 자리), 우리 차가 달린 길(파랑 = 상자 있는 동안, 주황 = 치운 뒤; 상자 없는 세션은 같은 시각으로
나눔), 상대 차가 달린 길(빨강 가는 선), 세게 닿은 곳(x = 벽, 네모 = 상자, 세모 = 차), 2 s 넘게 선 곳(검은 동그라미).
오른쪽 = 시간축: 바퀴마다 가로 막대(시작~끝, 높이 = 그 바퀴 시간; 치우는 시각에 걸친 바퀴는 회색), 세로 점선 = 상자 치운 시각, 아래 표시 = 세게 닿은 시각.
세션 고르기: 경우마다 세게 닿은 횟수로 줄 세워 고르게 뽑음(좋은 세션만·나쁜 세션만 고르지 않음).
인자: 기록npz 출력png앞머리 [경우마다 세션 수 4]  -> 앞머리_solo.png, _box.png, _opp.png, _oppbox.png"""
import sys
import numpy as np
import rec_lib as RL
from rec_lib import Rec, plt

R = Rec(sys.argv[1]); z = R.z; T, B, dt = R.T, R.B, R.dt
n_show = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rm_t = float(z["rm_t"]); k_rm = int(round(rm_t / dt)); secs = T * dt
st = R.st.astype(float); op = R.op.astype(float); on = R.on[:T]
box = (R.sid[0] >= 0).any(-1); opp = on.any(0)
hard = R.d[:, :, 3:6]; lapsig = z["lap"]
v = np.abs(st[:T, :, 3]); slow = v < 0.3; slow[:int(round(1.0 / dt))] = False
n_st = int(round(2.0 / dt))
C_US, C_AFTER = "#1f77b4", "#ff7f0e"
MK = (("x", "벽"), ("s", "상자"), ("^", "차"))
MODE = dict(map="MAP", track="시간 추종기", ideal="plan대로 정확히")[str(z["mode"])]


def laps_of(b):
    net = np.cumsum(lapsig[:, b]); hi, t_prev, out = 0, None, []
    for t in np.nonzero(lapsig[:, b] > 0)[0]:
        if net[t] > hi:
            hi = net[t]
            if t_prev is not None:
                out.append((t_prev * dt, (t + 1) * dt))
            t_prev = t + 1
    return out


def stall_starts(b):
    cur, out = 0, []
    for t in range(T):
        cur = cur + 1 if slow[t, b] else 0
        if cur == n_st:
            out.append(t)
    return out


for tag, label, m in (("solo", "혼자 · 빈 트랙", ~box & ~opp), ("box", "혼자 · 상자", box & ~opp), ("opp", "상대 차 · 빈 트랙", ~box & opp),
                      ("oppbox", "상대 차 · 상자", box & opp)):
    ids = np.nonzero(m)[0]
    if len(ids) == 0:
        continue
    order = ids[np.argsort(hard[:, ids].sum((0, 2)), kind="stable")]
    pick = order[np.linspace(0, len(order) - 1, min(n_show, len(order))).round().astype(int)]
    fig, axs = plt.subplots(len(pick), 2, figsize=(15, 5.2 * len(pick)), gridspec_kw={"width_ratios": [1, 1.3]}, squeeze=False)
    for b, (ax, a2) in zip(pick, axs):
        tk = int(R.tid[0, b]); R.draw_map(ax, tk)
        for q in R.boxes(0, b):
            ax.fill(q[:, 0], q[:, 1], color=RL.C_BOX, zorder=2)
        if opp[b]:
            ax.plot(op[:T, b, 0], op[:T, b, 1], "-", color=RL.C_CAR, lw=0.6, alpha=0.6, zorder=3, label="상대 차가 달린 길")
        ax.plot(st[:k_rm + 1, b, 0], st[:k_rm + 1, b, 1], "-", color=C_US, lw=1.0, alpha=0.8, zorder=4,
                label=f"우리 차 0-{rm_t:g} s" + (" (상자 있음)" if box[b] else ""))
        ax.plot(st[k_rm:, b, 0], st[k_rm:, b, 1], "-", color=C_AFTER, lw=1.0, alpha=0.8, zorder=4,
                label=f"우리 차 {rm_t:g}-{secs:g} s" + (" (치운 뒤)" if box[b] else ""))
        for j, (mk, nm) in enumerate(MK):
            ts = np.nonzero(hard[:, b, j] > 0)[0]
            if len(ts):
                ax.plot(st[ts, b, 0], st[ts, b, 1], mk, color="k", mfc="none", ms=11, mew=2, ls="none", zorder=6, label=f"세게 닿음: {nm} {len(ts)}번")
                a2.plot(ts * dt, np.full(len(ts), 0.4 + 0.5 * j), mk, color="k", mfc="none", ms=9, mew=1.5, ls="none")
        ss = stall_starts(b)
        if ss:
            ax.plot(st[ss, b, 0], st[ss, b, 1], "o", color="k", ms=14, mfc="none", mew=1.5, ls="none", zorder=6, label=f"2 s 넘게 섬 {len(ss)}번")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        kn = f" — 상대 {R.kinds[int(R.kind[0, b])]}, 속도 배율 {float(R.scale[0, b]):.2f}" if opp[b] and R.kinds else ""
        ax.set_title(f"세션 {b}{kn}", fontsize=10)
        L = laps_of(b)
        for s, e in L:
            a2.plot([s, e], [e - s, e - s], "-", color=C_US if e <= rm_t else (C_AFTER if s >= rm_t else "#999999"), lw=3, solid_capstyle="butt")
        lb = [e - s for s, e in L if e <= rm_t]; la = [e - s for s, e in L if s >= rm_t]
        a2.axvline(rm_t, color="k", ls="--", lw=0.8)
        top = max([e - s for s, e in L] + [10.0])
        a2.set_xlim(0, secs); a2.set_ylim(0, min(top * 1.1, 30)); a2.grid(alpha=0.3)
        a2.set_xlabel("세션 시간 (s)"); a2.set_ylabel("한 바퀴 시간 (s)")
        f = lambda x: f"{np.median(x):.2f} s ({len(x)}바퀴)" if x else "바퀴 없음"
        a2.set_title(f"바퀴: 0-{rm_t:g} s 중앙 {f(lb)} · {rm_t:g} s 뒤 중앙 {f(la)} | 세게 닿음 벽 {int(hard[:, b, 0].sum())} · 상자 "
                     f"{int(hard[:, b, 1].sum())} · 차 {int(hard[:, b, 2].sum())}", fontsize=10)
    fig.suptitle(f"{z['exam']} · {MODE} · {R.name} — {label} (세션 {len(ids)}개 가운데 세게 닿은 횟수 순으로 고르게 {len(pick)}개)", fontsize=12, y=1.0)
    fig.tight_layout(); out = f"{sys.argv[2]}_{tag}.png"; fig.savefig(out, dpi=60); plt.close(fig); print("그림", out)
