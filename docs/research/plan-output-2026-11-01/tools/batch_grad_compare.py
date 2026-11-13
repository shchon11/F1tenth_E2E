"""One figure, errors taken ACROSS rollouts: what the batch teaches and how alike the rollouts' gradients are (11-13).

`batch_compare.py` took a slope's error from the rows of a rollout as if they were independent. With exploration noise that
stays over decisions they are not (the noise and the advantage both last ~1 s), so its "seen in one update" was too large for
those runs. Here every rollout gives one number and the spread between rollouts is the error; nothing is assumed about rows.

    python3 batch_grad_compare.py OUT_PREFIX "LABEL=RHO=batch_signal txt=DUMP_DIR[,DUMP_DIR..]=GRAD_DIR" [...]

Left: how far 1 degree of steering noise moves the car sideways (5-6.5 m/s; from the txt).
Middle: per rollout (one dot each) the advantage a noise of 1 SD earns [s]; the bar is the mean.
Right: the cosine between the policy gradients of two different rollouts (one dot a pair; grad_dump.sh), for the steering and
the speed output apart. 1 = every rollout pushes the weights the same way, 0 = unrelated, below 0 = opposite.
"""
import sys, re, os, glob, itertools
import numpy as np
import rec_lib  # noqa: F401  (Korean font)
import matplotlib.pyplot as plt
import batch_cells as BC

OUT = sys.argv[1]
LOG = open(OUT + ".txt", "w")
COL = ["#868e96", "#4dabf7", "#1864ab"]
FIRST = 4


def say(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


CASES = (("앞에 상자 없음:\n더 빠르게", lambda q: ~q["has"] & q["full"] & (q["speed"] > 3.0), "zv"),
         ("가까이 지나갈 상자가\n1.5~6 m 앞: 더 빠르게", lambda q: q["has"] & q["full"] & (q["dist"] >= 1.5) & (q["dist"] < 6.0) & (q["dmin"] < 0.8), "zv"),
         ("모든 결정:\n조향을 더 왼쪽으로", lambda q: q["full"], "z"))
res = []
for label, rho, txt, dumps, gdir in [a.split("=") for a in sys.argv[2:]]:
    rho = float(rho)
    T = open(txt).read(); T = T[T.index("[5~6.5 m/s]"):]
    lat = [(float(a), float(b)) for a, b in re.findall(r"([\d.]+) s 뒤: 옆으로 ([+-][\d.]+) cm", T)[:5]]
    D = []
    for d in dumps.split(","):
        D += BC.load(d, rho)
    sl = [np.array([BC.slope(q["adv"][m(q)], q[zk][m(q)])[0] for q in D]) for _, m, zk in CASES]
    fs = sorted(glob.glob(os.path.join(gdir, "g*.npz")), key=lambda f: int(os.path.basename(f)[1:-4]))
    Z = [np.load(f) for f in fs if int(os.path.basename(f)[1:-4]) >= FIRST]
    cos = {}
    for key in ("steer", "speed"):
        G = np.stack([z[key].astype(np.float64) for z in Z]); G /= np.linalg.norm(G, axis=1, keepdims=True)
        cos[key] = np.array([G[i] @ G[j] for i, j in itertools.combinations(range(len(G)), 2)])
    res.append((label, lat, sl, cos))
    say(f"== {label}: batch rollout {len(D)}개, gradient rollout {len(Z)}개")
    for (nm, _, _), v in zip(CASES, sl):
        se = v.std(ddof=1) / np.sqrt(len(v))
        say(f"   {nm.replace(chr(10), ' '):<40} 평균 {v.mean():+.4f} ± {se:.4f} s (rollout 사이 표준오차)  ·  rollout 하나의 흩어짐 {v.std(ddof=1):.4f}  ·  평균/흩어짐 {v.mean() / v.std(ddof=1):+.2f}  ·  부호가 평균과 같은 rollout {int((np.sign(v) == np.sign(v.mean())).sum())}/{len(v)}")
    for key, kn in (("steer", "조향"), ("speed", "속도")):
        c = cos[key]
        say(f"   서로 다른 rollout 둘의 gradient cosine ({kn}): 평균 {c.mean():+.3f}, 가장 낮은 쌍 {c.min():+.2f}, 가장 높은 쌍 {c.max():+.2f}, 0보다 큰 쌍 {int((c > 0).sum())}/{len(c)}")

fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1.0, 1.5, 1.2]})
for i, (label, lat, sl, cos) in enumerate(res):
    ax[0].plot([a for a, _ in lat], [b for _, b in lat], color=COL[i], lw=2.2, marker="o", ms=5, label=label)
    ax[0].annotate(f"{lat[2][1]:.0f} cm" if lat[2][1] >= 10 else f"{lat[2][1]:.1f} cm", (lat[2][0], lat[2][1]), textcoords="offset points", xytext=(6, 6), fontsize=9, color="#343a40")
ax[0].axhspan(10, 20, color="#e03131", alpha=0.08, lw=0); ax[0].text(1.52, 10.6, "세게 닿은 경우에 모자랐던 폭 (10~20 cm)", fontsize=9, color="#c92a2a", va="bottom", ha="right")
ax[0].set_xlabel("결정 뒤 지난 시간 [s]"); ax[0].set_ylabel("조향 잡음 1도가 차를 옆으로 옮긴 거리 [cm]")
ax[0].set_title("잡음 한 번이 차를 실제로 얼마나 옮기나\n(5~6.5 m/s)", fontsize=11); ax[0].legend(fontsize=9, loc="upper left")
w = 0.8 / len(res); rng = np.random.default_rng(0)
for i, (label, lat, sl, cos) in enumerate(res):
    for k, v in enumerate(sl):
        x = k + (i - (len(res) - 1) / 2) * w
        ax[1].bar(x, v.mean(), w * 0.9, color=COL[i], alpha=0.35, label=label if k == 0 else None)
        ax[1].scatter(x + rng.uniform(-w * 0.3, w * 0.3, len(v)), v, s=16, color=COL[i], edgecolor="white", lw=0.5, zorder=3)
        ax[1].text(x, max(v.max(), 0) + 0.006, f"{v.mean():+.3f}", ha="center", va="bottom", fontsize=8.5, color="#343a40")
    for k, key in enumerate(("steer", "speed")):
        x = k + (i - (len(res) - 1) / 2) * w; c = cos[key]
        ax[2].scatter(x + rng.uniform(-w * 0.3, w * 0.3, len(c)), c, s=16, color=COL[i], edgecolor="white", lw=0.5, zorder=3, label=label if k == 0 else None)
        ax[2].plot([x - w * 0.4, x + w * 0.4], [c.mean()] * 2, color="#212529", lw=2, zorder=4)
        ax[2].text(x, 1.04, f"{c.mean():+.2f}", ha="center", va="bottom", fontsize=9, color="#343a40")
ax[1].axhline(0, color="#495057", lw=0.8); ax[1].set_xticks(range(len(CASES))); ax[1].set_xticklabels([c[0] for c in CASES], fontsize=9)
ax[1].set_ylabel("잡음 1 표준편차가 받은 advantage [s]"); ax[1].legend(fontsize=9, loc="lower left")
ax[1].set_title("batch가 가르쳐 주는 것: 점 하나 = rollout 하나, 막대 = 평균\n(점들이 0의 한쪽에 모여 있어야 update마다 같은 쪽을 가리킴)", fontsize=11)
ax[2].axhline(0, color="#495057", lw=0.8); ax[2].set_ylim(-1.05, 1.15); ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["조향 출력을 미는 gradient", "속도 출력을 미는 gradient"], fontsize=9)
ax[2].set_ylabel("서로 다른 rollout 둘의 gradient cosine"); ax[2].legend(fontsize=9, loc="lower left")
ax[2].set_title("rollout마다 actor 가중치를 같은 쪽으로 미나\n(점 하나 = rollout 한 쌍, 검은 선 = 평균; 1 = 같은 쪽, 0 = 무관, 음수 = 반대)", fontsize=11)
for a in ax:
    a.grid(alpha=0.25, axis="y")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("plan25y u150, update 없이 rollout만: 탐색 잡음을 결정마다 새로 뽑을 때와 이어지게 뽑을 때 (오차는 rollout 사이 흩어짐으로)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=110); say("그림", OUT + ".png")
