"""시험 기록의 길 위 상자 통과에서, plan이 상자 옆에 둔 간격과 실제로 지나간 간격의 차이 (실행에서 간격이 얼마나 사라지나).
통과마다: 가장 가까워지기 0.5 s / 0.3 s 전에 낸 plan의 길이 상자 중심에서 떨어진 거리(plan이 상자를 지나칠 만큼 길 때만),
실제로 달린 길이 상자 중심에서 떨어진 거리, 그리고 그 0.3 s 뒤 실제 위치가 plan의 0.3 s 점에서 상자 쪽으로 얼마나 가 있는지.
인자: 기록npz 행npy [기록npz 행npy ...]   (행 = exam_boxes.py의 EXAM_ROWS; EXAM_BOX_FOLLOW_FIG=png 이면 그림도)"""
import os
import sys
import numpy as np
from rec_lib import Rec, plt

FIG = os.environ.get("EXAM_BOX_FOLLOW_FIG", "")
COL = ("#555555", "#1f77b4", "#e6820e", "#2ca02c", "#9467bd")
BINS = ((0, 0.25), (0.25, 0.35), (0.35, 0.50), (0.50, 9))
keep = []

IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)


KS, NEAR = (9, 7, 5, 4, 3, 2, 1), 0.35      # NEAR: 차 반폭 0.15 + 상자 반폭 ~0.2 = 이보다 가까운 길은 상자에 닿음


def seg_dist(c, p):
    """점 c에서 꺾은선 p까지 거리와, 가장 가까운 점이 꺾은선의 끝점이 아닌지"""
    a, b = p[:-1], p[1:]; ab = b - a
    u = np.clip(((c - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    d = np.linalg.norm(a + u[:, None] * ab - c, axis=1); k = int(d.argmin())
    return float(d[k]), not (k == len(a) - 1 and u[k] > 0.999)


def q(x):
    x = np.asarray(x, float)
    return f"중앙 {np.median(x) * 100:+5.1f} cm, 10~90 % {np.percentile(x, 10) * 100:+5.1f}~{np.percentile(x, 90) * 100:+5.1f}" if len(x) else "없음"


for fn, fr in zip(sys.argv[1::2], sys.argv[2::2]):
    R = Rec(fn); z = R.z; T, dt = R.T, R.dt
    rows = np.load(fr); st = R.st.astype(float); xy = st[:, :, :2]; pl = R.pl.astype(float)
    out = {}
    for r in rows[rows[:, IGAP] < 0.40]:
        b, t = int(r[IB]), int(r[IT]); c = r[[IBX, IBY]]
        if t < 12 or t + 3 > T:
            continue
        w = np.arange(t - 10, min(t + 5, T) + 1)
        d_act = seg_dist(c, xy[w, b])[0]
        rec = dict(act=d_act, hard=r[IHARD] > 0, vin=r[IVIN])
        for name, k in (("p5", 5), ("p3", 3)):
            tau = t - k; p = np.concatenate([xy[tau, b][None], pl[tau, b]], 0)
            d, inner = seg_dist(c, p); rec[name] = d if inner else np.nan
            # 0.3 s later: actual position against the plan's 0.3 s point, measured toward the box
            e = xy[tau + 3, b] - p[3]; n = p[3] - p[2]; n = np.array([-n[1], n[0]]) / max(np.linalg.norm(n), 1e-6)
            sgn = np.sign(((c - p[3]) * n).sum()) or 1.0
            rec["e" + name] = float((e * n).sum() * sgn)
        for k in KS:                                         # when does the plan clear the box
            tau = t - k; d, inner = seg_dist(c, np.concatenate([xy[tau, b][None], pl[tau, b]], 0))
            rec[f"k{k}"] = d if inner else np.nan
        rec["see"] = r[ISEE]
        for k_, v in rec.items():
            out.setdefault(k_, []).append(v)
    o = {k_: np.array(v) for k_, v in out.items()}; hard = o["hard"].astype(bool)
    print(f"{z['exam']} · {z['mode']} · {R.name} — 길 위 상자 통과 {len(hard)}번 (세게 닿음 {int(hard.sum())})")
    for name, lab in (("p5", "0.5 s 전 plan"), ("p3", "0.3 s 전 plan")):
        ok = np.isfinite(o[name]); loss = o[name] - o["act"]
        print(f"  {lab}이 상자 중심에서 둔 거리: {q(o[name][ok])} | 실제 길의 거리: {q(o['act'][ok])}")
        print(f"    사라진 간격(plan - 실제): 전체 {q(loss[ok])} | 세게 닿은 통과 {q(loss[ok & hard])} | 안 닿은 통과 {q(loss[ok & ~hard])}")
        print(f"    그 plan을 낸 0.3 s 뒤 실제 위치가 plan 점에서 상자 쪽으로: 전체 {q(o['e' + name])} | 세게 닿은 통과 {q(o['e' + name][hard])}")
    seen = o["see"] >= 1.2
    print(f"  plan의 길이 상자 중심에서 {NEAR:.2f} m 안을 지나는 통과의 비율(plan이 상자를 지나칠 만큼 길 때; 괄호 = 1.2 s 넘게 전부터 보인 상자만): "
          + ", ".join(f"{0.1 * k:.1f} s 전 {100 * (o[f'k{k}'][np.isfinite(o[f'k{k}'])] < NEAR).mean():.0f} % ({100 * (o[f'k{k}'][np.isfinite(o[f'k{k}']) & seen] < NEAR).mean():.0f} %)" for k in KS))
    ok = np.isfinite(o["p5"])
    cls = []
    for lo, hi in BINS:
        s = ok & (o["p5"] >= lo) & (o["p5"] < hi); cls.append((int(s.sum()), int(hard[s].sum())))
        if s.sum():
            print(f"  0.5 s 전 plan의 거리 {lo:.2f}~{hi:.2f} m: 통과 {int(s.sum())}번, 세게 닿음 {100 * hard[s].mean():.0f} %")
    ex = str(z["exam"])
    lab = dict(ideal="plan대로 정확히(학습 조건)", map="MAP", track="시간 추종기")[str(z["mode"])]
    lab += (" + 미끄럼 각(참값)" if "slip_oracle" in ex else "") + (" + 미끄럼 각(표)" if "slip_table" in ex else "") + (f" + 속도 앞당김 {ex.split('lead')[1]} s" if "lead" in ex else "")
    keep.append(dict(lab=f"{lab} · {R.name.split('/')[0]} {R.name.split('_')[-1].split('.')[0]}", n=len(hard), cls=cls,
                     near=[100 * (o[f"k{k}"][np.isfinite(o[f"k{k}"])] < NEAR).mean() for k in KS]))
if FIG and keep:
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
    ax = axs[0]
    for k, c in zip(keep, COL):
        ax.plot([0.1 * q_ for q_ in KS], k["near"], "-o", color=c, lw=2, ms=5, label=f"{k['lab']} (통과 {k['n']}번)")
    ax.invert_xaxis(); ax.set_ylim(0, None); ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="upper right")
    ax.set_xlabel("상자를 지나가기 몇 초 전에 낸 plan인가 (s)"); ax.set_ylabel(f"그 plan의 길이 상자 중심 {NEAR:.2f} m 안을 지나는 통과 (%)")
    ax.set_title("길 위 상자: 지나가기 직전까지도 상자를 지나가는 plan을 냄", fontsize=11)
    ax = axs[1]; wbar = 0.8 / len(keep)
    for i, (k, c) in enumerate(zip(keep, COL)):
        xs = np.arange(len(BINS)) - 0.4 + wbar * (i + 0.5); ys = [100 * h / n if n else 0 for n, h in k["cls"]]
        ax.bar(xs, ys, wbar * 0.92, color=c, label=k["lab"])
        for x_, y_, (n, h) in zip(xs, ys, k["cls"]):
            ax.text(x_, y_ + 1.0, f"{h}/{n}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(np.arange(len(BINS))); ax.set_xticklabels([f"{lo:.2f}~{hi:.2f} m" if hi < 9 else f"{lo:.2f} m 넘게" for lo, hi in BINS])
    ax.grid(alpha=0.25, axis="y"); ax.legend(fontsize=9, loc="upper right")
    ax.set_xlabel("지나가기 0.5 s 전 plan의 길이 상자 중심에서 떨어진 거리"); ax.set_ylabel("세게 닿은 통과 (%)  [막대 위: 세게 닿음/통과]")
    ax.set_title("0.5 s 전 plan이 상자를 지나가면: 정확한 실행은 마지막에 비키고, MAP은 못 비킴", fontsize=11)
    fig.suptitle("시험(exam1)의 길 위 상자 통과 — plan이 언제 상자를 비키는가, 그리고 늦게 비키는 통과의 결과", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG, dpi=70); print("그림", FIG)
