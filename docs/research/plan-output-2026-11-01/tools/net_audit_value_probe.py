"""Can the coming hard box hit be told ahead of time, and by which part of the network?  (11-13, structure audit)

Input: `net_audit_hits.py` in NET_HITS_VALUE mode (`OUT.value.npz`): for every hard box hit and for every pass of a box within
0.8 m centre to centre without any contact ("near pass"), the decisions of the 3 s before, with the critic's value, the value
with that box taken out of the critic's inputs, the actor's first-0.1 s change caused by the box, the actor's trunk input +
memory, and the critic's inputs.

Per time before (one decision an event): how well each of these separates "this ends in a hard hit" from "this passes" (AUC:
0.5 = cannot tell, 1 = always). The critic's value and the box's share of it are used as they are; the read-outs are logistic
regressions fitted on other events (5 folds x 3 splits), a ceiling for what the inputs hold, not something the policy has.

    python3 net_audit_value_probe.py OUT.value.npz OUT_PREFIX
"""
import sys
import numpy as np
import rec_lib  # noqa: F401  (sets the Korean font)
import matplotlib.pyplot as plt
import torch
from scipy.stats import rankdata

z = np.load(sys.argv[1]); OUT = sys.argv[2]
LOG = open(OUT + ".txt", "w")


def say(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


H, N, C = z["hit"], z["near"], z["clean"]
# decisions are lined up by how far ahead the box is (the hit happens ~0.5 m before the car's centre is level with the box, a
# near pass is timed at the closest approach, so equal times before would compare different distances)
LAGS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
say(f"세게 닿음 {len(np.unique(H[:, 10]))}건 · 가까이 안 닿고 지나감 {len(np.unique(N[:, 10]))}건 · 앞에 놓인 상자를 안 닿고 지나감 {len(np.unique(C[:, 10]))}건")


_EV = {}


def at(A, dist):
    """one decision an event: the one with the box closest to `dist` m ahead (within 0.3 m)"""
    if id(A) not in _EV:
        _EV[id(A)] = [np.nonzero(A[:, 10] == e)[0] for e in np.unique(A[:, 10])]
    out = []
    for i in _EV[id(A)]:
        k = i[np.argmin(np.abs(A[i, 1] - dist))]
        if abs(A[k, 1] - dist) < 0.3:
            out.append(k)
    return np.array(out, int)


def roc_auc_score(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit_fit(X, y, l2=20.0):
    """class-balanced logistic regression with an L2 price (the weights' squared size x l2 / the number of rows)"""
    Xt = torch.from_numpy(X).double(); yt = torch.from_numpy(y).double()
    w = torch.zeros(X.shape[1], dtype=torch.double, requires_grad=True); b = torch.zeros(1, dtype=torch.double, requires_grad=True)
    cw = torch.where(yt > 0.5, 0.5 / yt.sum(), 0.5 / (len(yt) - yt.sum()))
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (cw * torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt, reduction="none")).sum() + l2 / len(yt) * (w ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    return w.detach().numpy(), float(b)


def auc_cv(X, y, g):
    """AUC of out-of-fold scores (5 folds x 3 shuffles; a session `g` is never split between fitting and scoring, so a box hit
    on several laps of one session cannot be told by recognising the place)"""
    X = np.asarray(X, float); sc = np.zeros(len(y)); gs = np.unique(g)
    for seed in range(3):
        rng = np.random.default_rng(seed); fold = (rng.permutation(len(gs)) % 5)[np.searchsorted(gs, g)]
        for f in range(5):
            tr, te = fold != f, fold == f
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            w, b = logit_fit((X[tr] - mu) / sd, y[tr])
            sc[te] += ((X[te] - mu) / sd) @ w + b
    return roc_auc_score(y, sc)


ROWS = (("critic의 값 그대로", "#7048e8"), ("그 상자가 critic 값에서 깎는 크기", "#1c7ed6"), ("actor의 첫 0.1 s가 상자 때문에 바뀐 크기", "#e8590c"),
        ("읽어내기: 상자의 옆 위치·속도·도는 정도만", "#868e96"), ("읽어내기: critic이 받는 숫자 입력(상자 위치 포함, 스캔 제외)", "#0b7285"),
        ("읽어내기: actor의 LiDAR 인코딩 + 차 상태", "#74b816"), ("읽어내기: actor의 LiDAR 인코딩 + 차 상태 + 기억", "#2f9e44"))
AUC = {k: [] for k, _ in ROWS}; NN = []
for lag in LAGS:
    ih, ino = at(H, lag), at(N, lag)
    y = np.r_[np.ones(len(ih)), np.zeros(len(ino))]; A = np.concatenate([H[ih], N[ino]])
    NN.append((len(ih), len(ino)))
    AUC[ROWS[0][0]].append(roc_auc_score(y, -A[:, 4]))
    AUC[ROWS[1][0]].append(roc_auc_score(y, -A[:, 5]))
    AUC[ROWS[2][0]].append(roc_auc_score(y, np.abs(A[:, 7]) / 3.0 + np.abs(A[:, 8])))
    geo = np.stack([A[:, 11], np.abs(A[:, 11]), A[:, 12], A[:, 13], A[:, 13] * A[:, 11], A[:, 12] ** 2 * np.abs(A[:, 13])], 1)
    g = A[:, 14]
    AUC[ROWS[3][0]].append(auc_cv(geo, y, g))
    AUC[ROWS[4][0]].append(auc_cv(np.concatenate([z["hit_xc"][ih], z["near_xc"][ino]]).astype(np.float32), y, g))
    xa = np.concatenate([z["hit_xa"][ih], z["near_xa"][ino]]).astype(np.float32)
    AUC[ROWS[5][0]].append(auc_cv(xa[:, :292], y, g))
    AUC[ROWS[6][0]].append(auc_cv(xa, y, g))
if "near_total" in z.files:
    say(f"가까이(중심 사이 0.8 m 안) 지나간 것 전체 {int(z['near_total'])}건 -> 그 가운데 세게 닿는 비율 {len(np.unique(H[:, 10])) / (len(np.unique(H[:, 10])) + int(z['near_total'])) * 100:.1f} %")
say("닿는 쪽과 지나가는 쪽을 가르는 정도 (0.5 = 못 가름, 1 = 늘 가름), 상자가 몇 m 앞일 때의 결정:")
say("  (읽어내기는 세션 단위로 나눠 가르치고 잼: 같은 세션의 같은 상자를 외워서 맞히는 것은 안 통함)")
say("  " + " " * 44 + " ".join(f"{l:>5.1f}" for l in LAGS))
say("  " + f"{'건수 (닿음/지나감)':<40}" + " ".join(f"{a:>3d}/{b:<3d}"[:7] for a, b in NN))
for k, _ in ROWS:
    say(f"  {k:<44}" + " ".join(f"{v:>5.2f}" for v in AUC[k]))


def band(ax, A, col, color, label, absval=False, scale=1.0):
    xs, md, lo, hi = [], [], [], []
    for lag in LAGS:
        i = at(A, lag)
        if len(i) < 8:
            continue
        v = (np.abs(A[i, col]) if absval else A[i, col]) * scale
        xs.append(lag); md.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0); ax.plot(xs, md, color=color, lw=2, marker="o", ms=4, label=label)
    return xs, md


for tag, A in (("세게 닿음", H), ("가까이 안 닿고 지나감", N), ("앞에 놓인 상자를 안 닿고 지나감", C)):
    say(f"  [{tag}] 상자가 몇 m 앞: 상자가 critic 값에서 깎는 크기 중앙 [s] / actor 첫 0.1 s 조향 변화 중앙 |도| / 줄인 속도 중앙 [m/s] / 그때 속도 중앙 [m/s] / 스캔에 보임 % / 건수")
    say("     " + " · ".join(f"{lag:.1f} m {np.median(A[i, 5]):+.2f}/{np.median(np.abs(A[i, 7])):.1f}/{np.median(A[i, 8]):+.2f}/{np.median(A[i, 12]):.1f}/{np.mean(A[i, 2] >= 2) * 100:.0f}/{len(i)}"
                             for lag in LAGS for i in [at(A, lag)] if len(i) >= 8))

fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
CH, CN = "#e03131", "#1c7ed6"
band(ax[0], H, 5, CH, f"세게 닿은 경우 ({len(np.unique(H[:, 10]))}건)"); band(ax[0], N, 5, CN, f"가까이 안 닿고 지나간 경우 ({len(np.unique(N[:, 10]))}건)")
ax[0].axhline(0, color="#868e96", lw=0.8); ax[0].set_ylabel("그 상자 때문에 critic의 값이 낮아진 크기 [s]")
ax[0].set_title("critic (상자 위치를 직접 받음):\n닿을 상자를 언제부터 더 비싸게 치나", fontsize=11)
band(ax[1], H, 7, CH, "세게 닿은 경우", absval=True); band(ax[1], N, 7, CN, "가까이 안 닿고 지나간 경우", absval=True)
ax[1].set_ylabel("상자 때문에 바뀐 첫 0.1 s 조향 [도]"); ax[1].set_title("actor (LiDAR만 받음):\n상자 때문에 조향을 언제부터 바꾸나", fontsize=11)
for k, col in ROWS:
    ax[2].plot(LAGS, AUC[k], color=col, lw=2 if "읽어내기" not in k else 1.6, ls="-" if "읽어내기" not in k else "--", marker="o", ms=4, label=k)
ax[2].axhline(0.5, color="#868e96", lw=0.8); ax[2].set_ylim(0.3, 1.02); ax[2].set_ylabel("닿는 쪽 / 지나가는 쪽을 가르는 정도 (0.5 = 못 가름)")
ax[2].set_title("닿을지를 얼마나 멀리서 가를 수 있나\n(점선 = 같은 입력에서 따로 가르친 읽어내기 = 입력에 든 정보의 상한)", fontsize=11)
for a in ax:
    a.set_xlim(6.3, 0.7); a.set_xlabel("상자가 차 앞 몇 m일 때 (5.5 m/s면 1 m = 0.18 s)"); a.grid(alpha=0.25); a.legend(fontsize=8, loc="best")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle(sys.argv[3] if len(sys.argv) > 3 else "", fontsize=13)
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=110)
say("그림", OUT + ".png")
