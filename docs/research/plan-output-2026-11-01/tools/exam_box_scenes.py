"""시험 기록의 상자 통과 장면: exam_boxes.py가 저장한 통과별 행(EXAM_ROWS)에서 골라 그림. 숫자보다 먼저 볼 것.
장면마다 왼쪽 = 지도 확대(상자 둘레 ±3 m): 상자(갈색), 그 세션에서 그 상자를 지난 모든 통과의 길(회색 = 다른 통과, 파랑 = 이 통과,
숫자 = 몇 번째 통과), 이 통과의 1.0 s 전·0.5 s 전·가장 가까운 순간의 차체, 1.0 s 전(검정)·0.5 s 전(분홍) plan, 초록 점선 = 상자를 치운 뒤
같은 세션에서 달린 길. 오른쪽 = 상자까지 남은 거리(가로)에 대한 속도: 통과마다 한 줄(이 통과 = 파랑 굵게, 치운 뒤 = 초록 점선),
세로 점선 = 상자 중심이 스캔에 보이기 시작한 거리.
인자: 기록npz 행npy 출력png [고르기: hard(세게 닿은 길 위 통과, 기본) | late(0.9 s 안에야 보인 길 위 통과) | repeat(세게 닿은 다음 통과)] [장면 수 10]"""
import sys
import numpy as np
import rec_lib as RL
from rec_lib import Rec, plt

R = Rec(sys.argv[1]); z = R.z; T, dt = R.T, R.dt
rows = np.load(sys.argv[2])
sel = sys.argv[4] if len(sys.argv) > 4 else "hard"
n_show = int(sys.argv[5]) if len(sys.argv) > 5 else 10
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
st = R.st.astype(float); xy = st[:, :, :2]
rm_t = float(z["rm_t"]); k_rm = min(int(round(rm_t / dt)), T)
n05, n10, n30 = (int(round(x / dt)) for x in (0.5, 1.0, 3.0))
way = rows[:, IGAP] < 0.40
m = dict(hard=way & (rows[:, IHARD] > 0), late=way & (rows[:, ISEE] < 0.9), repeat=way & (rows[:, IN] >= 2) & (rows[:, IPREV] > 0))[sel]
NAME = dict(hard="길 위의 상자에 세게 닿은 통과", late="0.9 s 안에야 보인 길 위의 상자", repeat="지난번에 세게 닿은 상자의 다음 통과")[sel]
ids = np.nonzero(m)[0]
print(f"{NAME}: {len(ids)}번")
if len(ids) == 0:
    sys.exit(0)
pick = ids[np.linspace(0, len(ids) - 1, min(n_show, len(ids))).round().astype(int)]
MODE = dict(map="MAP", track="시간 추종기", ideal="plan대로 정확히")[str(z["mode"])]
fig, axs = plt.subplots(len(pick), 2, figsize=(15, 5.0 * len(pick)), gridspec_kw={"width_ratios": [1, 1.4]}, squeeze=False)
for i, (ax, a2) in zip(pick, axs):
    r = rows[i]; b, j, t = int(r[IB]), int(r[IJ]), int(r[IT]); tk = int(R.tid[0, b]); c = r[[IBX, IBY]]
    R.draw_map(ax, tk)
    for q in R.boxes(max(t - n30, 0), b):
        ax.fill(q[:, 0], q[:, 1], color=RL.C_BOX, zorder=2)
    d_all = np.linalg.norm(xy[:, b] - c[None], axis=1)
    after = np.arange(k_rm + n30, T + 1)
    near = after[d_all[after] < 4.0] if len(after) else after
    if len(near):
        cut = np.nonzero(np.diff(near) > 1)[0] + 1
        for seg in np.split(near, cut):
            ax.plot(xy[seg, b, 0], xy[seg, b, 1], "--", color="#2ca02c", lw=1.2, zorder=3)
            sgn = np.where(np.arange(len(seg)) <= d_all[seg].argmin(), 1.0, -1.0)
            a2.plot(sgn * d_all[seg], np.abs(st[seg, b, 3]), "--", color="#2ca02c", lw=1.0)
        ax.plot([], [], "--", color="#2ca02c", lw=1.2, label="상자를 치운 뒤 달린 길")
    same = np.nonzero((rows[:, IB] == b) & (rows[:, IJ] == j))[0]
    for k in same:
        tt = int(rows[k, IT]); w = np.arange(max(tt - n30, 0), min(tt + n10, T) + 1); me = k == i
        ax.plot(xy[w, b, 0], xy[w, b, 1], "-", color="#1f77b4" if me else "#999999", lw=2.0 if me else 1.0, zorder=5 if me else 4)
        ins = w[np.abs(xy[w, b] - c[None]).max(1) < 2.8]            # label where the path enters the picture
        if len(ins):
            ax.text(xy[ins[0], b, 0], xy[ins[0], b, 1], f"{int(rows[k, IN])}", fontsize=11, color="#1f77b4" if me else "#666666", zorder=7,
                    clip_on=True)
        sgn = np.where(w <= tt, 1.0, -1.0)
        a2.plot(sgn * d_all[w], np.abs(st[w, b, 3]), "-", color="#1f77b4" if me else "#999999", lw=2.5 if me else 1.0,
                label=f"{int(rows[k, IN])}번째 통과 ({tt * dt:.0f} s째)" + (" — 세게 닿음" if rows[k, IHARD] > 0 else (" — 닿음" if rows[k, ITOUCH] > 0 else "")))
    for tt, al in ((t - n10, 0.25), (t - n05, 0.5), (t, 0.9)):
        if tt >= 0:
            q = R.body(*st[tt, b, :3]); ax.fill(q[:, 0], q[:, 1], color="#1f77b4", alpha=al, zorder=6)
    for tt, col in ((t - n10, "k"), (t - n05, "#e377c2")):
        if 0 <= tt < T:
            p = np.concatenate([xy[tt, b][None], R.pl[tt, b]], 0); ax.plot(p[:, 0], p[:, 1], "-o", color=col, ms=3, lw=1.5, zorder=8)
    if R.on[:T, b].any():
        w = np.arange(max(t - n30, 0), min(t + n10, T) + 1)
        ax.plot(R.op[w, b, 0], R.op[w, b, 1], "-", color=RL.C_CAR, lw=1.0, alpha=0.7, zorder=4, label="상대 차가 달린 길")
        q = R.body(*R.op[t, b, :3].astype(float)); ax.fill(q[:, 0], q[:, 1], color=RL.C_CAR, alpha=0.6, zorder=6)
    ax.set_xlim(c[0] - 3.0, c[0] + 3.0); ax.set_ylim(c[1] - 3.0, c[1] + 3.0)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_title(f"세션 {b}, {t * dt:.1f} s째 — 그 상자 {int(r[IN])}번째 통과 | 달리던 길에서 상자 중심까지 {r[IGAP]:.2f} m, 실제로 비킨 폭 {r[ISHIFT]:.2f} m", fontsize=10)
    if r[ISEED] > 0:
        a2.axvline(r[ISEED], color="k", ls=":", lw=1.0)
    a2.axvline(0, color="#d62728", lw=0.8); a2.set_xlim(8.0, -2.0); a2.set_ylim(0, 8.5); a2.grid(alpha=0.3)
    a2.set_xlabel("상자까지 거리 (m; 음수 = 지나간 뒤)"); a2.set_ylabel("속도 (m/s)"); a2.legend(fontsize=8, loc="lower left")
    a2.set_title(f"상자 중심이 보이기 시작: 지나가기 {r[ISEE]:.1f} s 전, {r[ISEED]:.1f} m 앞 | 1 s 전 속도 {r[IVIN]:.1f} m/s, 그 뒤 1 s 동안 {r[IDROP]:.1f} m/s 줄임"
                 + (" | 세게 닿음" if r[IHARD] > 0 else (" | 닿음" if r[ITOUCH] > 0 else " | 안 닿음")), fontsize=10)
fig.suptitle(f"{z['exam']} · {MODE} · {R.name} — {NAME} {len(ids)}번 가운데 고르게 {len(pick)}개", fontsize=12, y=1.0)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=60); print("그림", sys.argv[3])
