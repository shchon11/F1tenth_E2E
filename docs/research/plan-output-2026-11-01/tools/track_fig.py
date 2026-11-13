"""Figure: real updates from the same checkpoint, steps per minibatch vs one step per epoch (11-14; --track-steps records).

    python3 track_fig.py OUT.png BASE_EXAM.npz "LABEL=RUN_DIR=EXAM_U8.npz" [...]
Left: distance moved along the measured good direction, summed over updates (random direction dotted). Middle: how far the actor
moved in all, per update. Right: the rollout's policy gradient (first epoch) against the good direction, per update (cosine).
Bottom line in the title: the frozen exam after 8 updates, paired with the start checkpoint.
"""
import sys, os
import numpy as np
import rec_lib  # noqa: F401
import matplotlib.pyplot as plt
import grad_step_table as GT

OUT, BASE = sys.argv[1], sys.argv[2]
COL = ["#868e96", "#1864ab", "#e8590c", "#2b8a3e"]
b0 = GT.load(BASE); es = ~b0["box"] & ~b0["opp"]
fig, ax = plt.subplots(1, 4, figsize=(21, 5.4), gridspec_kw={"width_ratios": [1.1, 1.0, 1.1, 1.0]})
ex = []
for i, a in enumerate(sys.argv[3:]):
    lab, run, exn = a.split("=")
    Z = [np.load(os.path.join(run, "track", f"t{u}.npz")) for u in range(1, 9)]
    u = np.arange(0, 9)
    ax[0].plot(u, np.r_[0, np.cumsum([float(z["update_w"]) for z in Z])], color=COL[i], lw=2.2, marker="o", ms=5, label=lab)
    ax[0].plot(u, np.r_[0, np.cumsum([float(z["update_rand"]) for z in Z])], color=COL[i], lw=1.2, ls=":")
    ax[1].plot(u[1:], [float(z["update_norm"]) for z in Z], color=COL[i], lw=2.2, marker="o", ms=5, label=lab)
    pc = [float(z["term_policy_w"].sum() / np.linalg.norm(z["term_policy_sum"].astype(np.float64))) for z in Z]
    ax[2].plot(u[1:], pc, color=COL[i], lw=2.2, marker="o", ms=5, label=lab)
    q = GT.load(exn); dd = q["dist"] - b0["dist"]; dh = (q["hard"] - b0["hard"]).sum(1)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    ex.append((lab, dd[es].mean(), se(dd[es]), dh.mean(), se(dh)))
ax[0].set_title("좋은 방향으로 간 거리 (update를 거듭하며 쌓임)\n점선 = 무작위 방향", fontsize=11)
ax[0].set_xlabel("update"); ax[0].set_ylabel("u150에서 좋은 방향으로 간 거리"); ax[0].legend(fontsize=9, loc="upper left")
ax[0].text(8.1, 0.03, "← 학습 없이 0.03 옮긴 것:\n    빈 트랙 +9.4 m", fontsize=8.5, color="#495057", va="center", ha="right")
ax[0].axhline(0.03, color="#adb5bd", lw=0.8, ls="--")
ax[1].set_title("가중치 전체가 움직인 거리 (update 하나)", fontsize=11); ax[1].set_xlabel("update"); ax[1].set_ylim(0, None)
ax[2].set_title("그 update의 rollout 전체 policy gradient가\n좋은 방향을 가리키는 정도 (cosine)", fontsize=11); ax[2].set_xlabel("update"); ax[2].axhline(0, color="#495057", lw=0.8)
ax[2].set_ylim(-1, 1)
x = np.arange(len(ex))
ax[3].bar(x, [e[1] for e in ex], 0.6, yerr=[1.96 * e[2] for e in ex], color=COL[:len(ex)], capsize=4)
for k, e in enumerate(ex):
    ax[3].text(k, e[1] + (1.0 if e[1] >= 0 else -1.0), f"{e[1]:+.1f} m\n세게 닿음 {e[3]:+.2f}±{e[4]:.2f}/세션", ha="center", va="bottom" if e[1] >= 0 else "top", fontsize=9)
ax[3].axhline(0, color="#495057", lw=0.8); ax[3].set_xticks(x); ax[3].set_xticklabels([e[0] for e in ex], fontsize=9)
ax[3].set_ylabel("빈 트랙 혼자 거리 [m / 90 s], u150과의 차"); ax[3].set_title("8 update 뒤 frozen exam\n(u150과 같은 세계, 128 세션; 막대 = 95 % 구간)", fontsize=11)
lo, hi = ax[3].get_ylim(); ax[3].set_ylim(min(lo, -6), hi + 3)
for a in ax:
    a.grid(alpha=0.25)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("plan25y u150에서 실제 학습 8 update: minibatch마다 한 걸음(지금 설정, 걸음 64번) vs 16 minibatch를 합쳐 한 걸음(epoch마다 1번, 걸음 4번). 나머지 설정·첫 rollout 같음", fontsize=12.5)
fig.tight_layout(); fig.savefig(OUT, dpi=105); print("그림", OUT)
