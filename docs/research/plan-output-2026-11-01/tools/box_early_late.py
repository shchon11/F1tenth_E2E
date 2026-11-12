"""상자를 늦게 봤을 때와 일찍 봤을 때의 통과 비교 (11-12, 사용자: "횡방향 주행안정성 보상 ... 다와가서 꺾지 않고 미리미리 꺾이게
간접적으로 유도되도록 하면 기억을 써먹을 수 있게 되지 않을까"). 미리 아는 것이 지금 얼마를 벌 수 있는지, 횡방향 비용이 물 자리가 있는지를
시험 기록(plan대로 정확히 실행)에서 잼. 혼자 달리는 상자 세션의 '길 위의 상자'(달리던 길이 상자 중심 0.40 m 안) 통과마다:
 - 보이기 시작한 때: 상자에 가장 가까워지기 몇 s 전부터 스캔에 보였나 (exam_boxes.py의 통과 행)
 - 비키기 시작한 때: 같은 세션에서 상자를 치운 뒤 달린 길에서 0.10 m 넘게 벗어나기 시작한 때 (가장 가까워지기 몇 s 전)
   -> 둘의 차 = 보고 나서 비키기까지. 보이기 전에 비키기 시작했으면 '미리 비킴'
 - 꺾는 세기: 상자 8 m 전 ~ 3 m 뒤 구간의 횡가속도(속도 x 요 각속도) 최대와 0.1 s 사이 변화 최대, 같은 세션 빈 트랙 같은 구간과의 차
 - 잃은 시간: 그 구간을 지나는 시간 - 같은 세션에서 상자를 치운 뒤 같은 구간을 지나는 시간(바퀴들의 중앙)
자리 맞춤은 보상이 쓰는 벽 기준 거리장(분석에만 씀). 닿은 통과는 꺾는 세기·잃은 시간 통계에서 빼고 비율만 따로 셈.
인자: 기록npz 통과행npy [그림png]"""
import sys
import warnings
import numpy as np
from rec_lib import Rec, plt, along_track

IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
BEFORE, AFTER, DEV = 8.0, 3.0, 0.10
R = Rec(sys.argv[1]); z = R.z; T, B, dt = R.T, R.B, R.dt
rows = np.load(sys.argv[2]); fig_out = sys.argv[3] if len(sys.argv) > 3 else ""
st = R.st.astype(float); xy = st[:, :, :2]; V = np.abs(st[:, :, 3])
rate = np.diff(st[:, :, 2], axis=0) / dt                        # (T, B) yaw rate over each decision
alat = V[:T] * rate                                             # lateral acceleration [m/s^2]
jerk = np.abs(np.diff(alat, axis=0)) / dt                       # (T-1, B) its change [m/s^3]
rm_t = float(z["rm_t"]); k_rm = min(int(round(rm_t / dt)), T)
s, lap, L = along_track(z, T); u = lap * L + s                  # unwrapped place along the track
hard_any = R.d[:, :, 3:6].sum(-1) > 0
opp = R.on[:T].any(0)
live = R.sid >= 0; pose = R.pose.astype(float)
MODE = dict(map="MAP", track="시간 추종기", ideal="plan대로 정확히")[str(z["mode"])]


def cross(ub, c, k_lo, k_hi):
    """the (fractional) decision at which the place first reaches c inside [k_lo, k_hi], or nan"""
    k = np.nonzero(ub[k_lo: k_hi + 1] >= c)[0]
    if len(k) == 0 or k[0] == 0:
        return np.nan
    k = k_lo + int(k[0]); d = ub[k] - ub[k - 1]
    return k - 1 + (c - ub[k - 1]) / d if d > 1e-6 else float(k)


def stretch(b, c, k_lo, k_hi):
    """(time through, peak |lateral acceleration|, peak change of it) over the places [c - BEFORE, c + AFTER], or None"""
    a, e = cross(u[:, b], c - BEFORE, k_lo, k_hi), cross(u[:, b], c + AFTER, k_lo, k_hi)
    if not (np.isfinite(a) and np.isfinite(e)) or e <= a:
        return None
    i0, i1 = int(np.floor(a)), int(np.ceil(e))
    if hard_any[i0: i1 + 1, b].any() or V[i0: i1 + 1, b].min() < 1.0:
        return None
    return (e - a) * dt, float(np.abs(alat[i0: i1, b]).max()), float(jerk[i0: max(i1 - 1, i0 + 1), b].max())


def to_poly(p, A, D, LL):
    w = (((p[:, None] - A[None]) * D[None]).sum(2) / LL[None]).clip(0.0, 1.0)
    return np.linalg.norm(p[:, None] - (A[None] + w[..., None] * D[None]), axis=2).min(1)


out = []
for r in rows:
    b, j, t = int(r[IB]), int(r[IJ]), int(r[IT])
    if opp[b] or r[IGAP] >= 0.40 or t < 30:
        continue
    kk = np.arange(k_rm + 50, T - 1)                           # this session's own path once the boxes are gone
    kk = kk[(V[kk, b] > 2.0) & (V[kk + 1, b] > 2.0)]
    if len(kk) < 100:
        continue
    A = xy[kk, b]; D = xy[kk + 1, b] - A; LL = (D ** 2).sum(1).clip(min=1e-9)
    w = np.arange(max(t - 35, 0), t + 1)
    dev = to_poly(xy[w, b], A, D, LL)
    k0 = len(w) - 1
    while k0 > 0 and dev[k0 - 1] > DEV:
        k0 -= 1
    t_dev = (len(w) - 1 - k0) * dt if dev[-1] > DEV else 0.0   # s before the closest approach the car left its usual path
    shift = float(dev[-5:].max())
    others = np.nonzero(live[t, b] & (np.arange(live.shape[2]) != j))[0]
    alone = not len(others) or float(np.linalg.norm(pose[t, b, others, :2] - pose[t, b, j, :2], axis=1).min()) > 6.0
    w_ = stretch(b, u[t, b], max(t - 80, 0), min(t + 60, k_rm - 1)) if r[ITOUCH] < 0.5 else None
    refs = [x for x in (stretch(b, u[t, b] + m * L, k_rm + 30, T - 2) for m in range(1, 9)) if x is not None]
    if w_ is not None and len(refs) >= 2:
        ref = np.median(np.array(refs), 0)
        kin = [w_[0] - ref[0], w_[1], w_[1] - ref[1], w_[2], w_[2] - ref[2]]
    else:
        kin = [np.nan] * 5
    out.append([b, t, r[IN], r[ISEE], t_dev, shift, r[ITOUCH], r[IHARD], r[IVIN], float(alone)] + kin)
O = np.array(out, float).reshape(-1, 15)
JB, JT, JN, JSEE, JDEV, JSH, JTOUCH, JHARD, JV, JAL, JLOSS, JA, JDA, JJ, JDJ = range(15)
print(f"{z['exam']} · {MODE} · {R.name} · 상자 {z['count']} — 혼자 달리는 상자 세션의 길 위 상자 통과 {len(O)}번 "
      f"(닿음 {int((O[:, JTOUCH] > 0).sum())}, 그중 세게 {int((O[:, JHARD] > 0).sum())}; 꺾는 세기·잃은 시간을 잰 통과 {int(np.isfinite(O[:, JLOSS]).sum())})")
if len(O) < 8:
    sys.exit(0)


def med(x, q=50):
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else float("nan")


def line(name, m):
    n = int(m.sum())
    if n == 0:
        return f"  {name} | 0번"
    k = O[m]; ok = np.isfinite(k[:, JLOSS]); moved = k[:, JSH] > DEV
    early = moved & (k[:, JSEE] > 0) & (k[:, JDEV] > k[:, JSEE] + 0.05)
    return (f"  {name} | {n}번 | 닿음 {100 * (k[:, JTOUCH] > 0).mean():.0f} % (세게 {100 * (k[:, JHARD] > 0).mean():.0f} %) | 잃은 시간 중앙 {med(k[:, JLOSS]):+.2f} s "
            f"(75 % {med(k[:, JLOSS], 75):+.2f}) | 횡가속도 최대 {med(k[:, JA]):.1f} m/s² (빈 트랙 같은 구간보다 {med(k[:, JDA]):+.1f}) | 그 변화 최대 "
            f"{med(k[:, JJ]):.0f} m/s³ ({med(k[:, JDJ]):+.0f}) | 비킨 거리 중앙 {med(k[:, JSH]):.2f} m | 보인 때 {med(k[:, JSEE]):.1f} s 전 · 비키기 시작 "
            f"{med(k[moved, JDEV]):.1f} s 전 | 보이기 전에 비키기 시작 {int(early.sum())}번 ({100 * early.sum() / max(int(moved.sum()), 1):.0f} %) | 들어가는 속도 {med(k[:, JV]):.1f} m/s"
            f" | 잰 통과 {int(ok.sum())}")


see = O[:, JSEE]
print("(가) 보이기 시작한 때별 (가장 가까워지기 몇 s 전부터 보였나)")
for name, m in (("0.9 s 안", (see > 0) & (see < 0.9)), ("0.9~1.5 s", (see >= 0.9) & (see < 1.5)), ("1.5~2.2 s", (see >= 1.5) & (see < 2.2)),
                ("2.2 s 넘게", see >= 2.2), ("끝까지 안 보임", see <= 0)):
    print(line(name, m))
print("(나) 실제 비킨 거리별 (상자 옆을 지날 때 평소 길에서 벗어난 거리)")
for name, m in (("0.10 m 안(안 비킴)", O[:, JSH] <= 0.10), ("0.10~0.30 m", (O[:, JSH] > 0.10) & (O[:, JSH] <= 0.30)),
                ("0.30~0.60 m", (O[:, JSH] > 0.30) & (O[:, JSH] <= 0.60)), ("0.60 m 넘게", O[:, JSH] > 0.60)):
    print(line(name, m))
print("(다) 그 세션에서 그 상자를 몇 번째 지나나 (둘째부터는 이미 한 번 지나간 상자)")
for name, m in (("첫째", O[:, JN] == 1), ("둘째", O[:, JN] == 2), ("셋째", O[:, JN] == 3), ("넷째부터", O[:, JN] >= 4)):
    print(line(name, m))
print("(라) 크게 비켜야 했던 통과(0.30 m 넘게)만, 보이기 시작한 때별")
big = O[:, JSH] > 0.30
for name, m in (("1.2 s 안", big & (see > 0) & (see < 1.2)), ("1.2~1.8 s", big & (see >= 1.2) & (see < 1.8)), ("1.8 s 넘게", big & (see >= 1.8))):
    print(line(name, m))
ok = np.isfinite(O[:, JLOSS])
tot = float(np.nansum(O[ok, JLOSS])); laps_box = sum(int(lap[k_rm - 1, b] - lap[0, b]) for b in np.unique(O[:, JB]).astype(int))
print(f"(마) 잃은 시간 합 {tot:.1f} s / 상자 있는 동안 달린 바퀴 {laps_box} = 바퀴당 {tot / max(laps_box, 1):+.2f} s (닿지 않은 통과만; 상자 8 m 전 ~ 3 m 뒤 구간)")
for name, m in (("  그중 1.2 s 안에 보인 통과", ok & (see > 0) & (see < 1.2)), ("  그중 1.2 s 넘게 전에 보인 통과", ok & (see >= 1.2))):
    print(f"{name}: {int(m.sum())}번, 합 {float(O[m, JLOSS].sum()):+.1f} s, 통과당 {float(O[m, JLOSS].mean()) if m.any() else float('nan'):+.2f} s")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    c1 = np.corrcoef(see[ok & (see > 0)], O[ok & (see > 0), JLOSS])[0, 1]; c2 = np.corrcoef(see[ok & (see > 0)], O[ok & (see > 0), JDJ])[0, 1]
print(f"(바) 보인 때(s 전)와의 상관: 잃은 시간 {c1:+.2f}, 횡가속도 변화 최대(빈 트랙 대비) {c2:+.2f}  (음수 = 늦게 볼수록 큼)")

if fig_out:
    BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#333333"
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.2))
    hit = O[:, JTOUCH] > 0; seen = see > 0
    panels = ((JLOSS, "잃은 시간 (s; 상자 8 m 전 ~ 3 m 뒤 구간, 빈 트랙 같은 구간 대비)", "① 늦게 본 상자일수록 시간을 더 잃나"),
              (JDJ, "횡가속도 변화 최대 (m/s³; 빈 트랙 같은 구간 대비)", "② 늦게 본 상자일수록 더 급하게 꺾나"))
    edges = np.array([0.0, 0.9, 1.2, 1.5, 1.8, 2.2, 3.01])
    for ax, (col, yl, title) in zip(axs[:2], panels):
        m = seen & np.isfinite(O[:, col])
        ax.scatter(see[m], O[m, col], s=22, color=BLUE, alpha=0.45, lw=0, label="통과 하나 (닿지 않은 것)")
        xs, ys = [], []
        for a, e in zip(edges[:-1], edges[1:]):
            mm = m & (see >= a) & (see < e)
            if mm.sum() >= 5:
                xs.append(0.5 * (a + e)); ys.append(np.median(O[mm, col]))
        ax.plot(xs, ys, "-o", color=ORANGE, lw=2.5, ms=8, label="구간 중앙값")
        ax.axhline(0, color=INK, lw=1)
        ax.set_xlabel("상자가 보이기 시작한 때 (가장 가까워지기 몇 s 전)"); ax.set_ylabel(yl); ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10, frameon=False); ax.grid(alpha=0.25)
        if col == JLOSS:
            ax.set_ylim(-0.3, max(1.2, float(np.nanpercentile(O[m, col], 98))))
    ax = axs[2]
    m = seen & (O[:, JSH] > DEV)
    first = O[:, JN] == 1
    ax.scatter(see[m & first], O[m & first, JDEV], s=30, color=ORANGE, alpha=0.7, lw=0, label="첫째 통과")
    ax.scatter(see[m & ~first], O[m & ~first, JDEV], s=22, color=BLUE, alpha=0.45, lw=0, label="둘째 통과부터 (이미 지나간 상자)")
    ax.plot([0, 3.2], [0, 3.2], "--", color=INK, lw=1.5)
    ax.text(0.25, 2.9, "선 위쪽 = 보이기 전에\n비키기 시작(미리 비킴)", fontsize=10, color=INK, va="top")
    ax.text(3.15, 0.15, "선 아래쪽 = 보고 나서 비킴", fontsize=10, color=INK, ha="right")
    ax.set_xlim(0, 3.2); ax.set_ylim(0, 3.2)
    ax.set_xlabel("상자가 보이기 시작한 때 (가장 가까워지기 몇 s 전)"); ax.set_ylabel("평소 길에서 벗어나기 시작한 때 (몇 s 전)")
    ax.set_title("③ 이미 지나간 상자를 보이기 전에 미리 비키나", fontsize=12); ax.legend(fontsize=10, frameon=False, loc="lower right"); ax.grid(alpha=0.25)
    fig.suptitle(f"{z['exam']} · {R.name} · {MODE} · 상자 {z['count']} — 혼자 달리는 세션, 길 위의 상자 통과 {len(O)}번", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(fig_out, dpi=70); print("그림", fig_out)
