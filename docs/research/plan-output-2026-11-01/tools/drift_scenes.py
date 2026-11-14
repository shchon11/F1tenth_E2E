"""(11-14, 사용자: "분석을 좀 심도깊게 하고 정해") 같은 frozen exam 세션에서 checkpoint 둘(A = 앞, B = 뒤)을 비교: B가 세게 닿은 곳마다
A는 그 자리를 어떻게 지났나. 숫자보다 먼저 그림을 볼 것.
표: B의 세게 닿음 하나마다 -- B가 그 자리에 들어온 속도, 같은 세션의 A가 그 자리(0.4 m 안)를 지난 속도·가장 가까이 간 거리, A도 그 근처(1 m)에서 세게
닿았나, B가 닿기 전 1 s 동안 이 자리까지 몇 번째 통과였나(같은 자리를 이미 지나 본 적이 있나).
그림: 종류마다(벽·상자) 고르게 뽑은 장면 -- 확대 지도(±2.5 m), 상자(닿은 시각의 자리), B가 지난 길(닿기 2 s 전 ~ 0.3 s 뒤, 빨강; 점 = 0.1 s, 굵은
외곽선 = 닿은 순간의 차체, 가는 선 = 그때의 plan), A가 같은 세션에서 이 자리를 지난 모든 통과(파랑), 가장 가까운 통과의 차체(파랑 외곽선).
인자: A.npz B.npz 이름A 이름B 출력앞머리 [종류마다 장면 수 6]"""
import sys
import numpy as np
import rec_lib as RL
from rec_lib import Rec, plt

A, B = Rec(sys.argv[1]), Rec(sys.argv[2]); nA, nB = sys.argv[3], sys.argv[4]; OUT = sys.argv[5]
n_show = int(sys.argv[6]) if len(sys.argv) > 6 else 6
T, NB, dt = min(A.T, B.T), B.B, B.dt
KN = ("벽", "상자", "차")
LOG = open(OUT + ".txt", "w")


def say(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


def case_of(R):
    box = (R.sid[0] >= 0).any(-1); opp = R.on[:R.T].any(0)
    return np.where(box & opp, "상자+상대", np.where(box, "상자", np.where(opp, "상대", "빈 트랙")))


cA, cB = case_of(A), case_of(B)
assert (cA == cB).all(), "같은 세계가 아님"
for R in (A, B):
    R.v = np.abs(R.st[:T, :, 3].astype(float)); R.xy = R.st[:T, :, :2].astype(float)

ev = []
for t, b, k in zip(*np.nonzero(B.d[:T, :, 3:6] > 0)):
    p = B.xy[t, b]
    dA = np.linalg.norm(A.xy[:, b] - p, axis=1)
    j = int(dA.argmin())
    near = dA < 0.4
    vA = float(np.median(A.v[near, b])) if near.any() else np.nan
    # A's speed coming into the spot, like B's: the fastest in the second before each of A's passes (median over passes)
    ent = np.nonzero(np.diff(near.astype(int)) == 1)[0] + 1
    vinA = float(np.median([A.v[max(0, i - int(round(1.0 / dt))):i + 1, b].max() for i in ent])) if len(ent) else np.nan
    hitA = bool(((A.d[:T, b, 3:6].sum(1) > 0) & (dA < 1.0)).any())
    t0 = max(0, t - int(round(1.0 / dt)))
    vin = float(B.v[t0:t + 1, b].max())                          # the fastest B was in the second before
    inside = (np.linalg.norm(B.xy[:t0, b] - p, axis=1) < 0.4).astype(int)
    passes = int((np.diff(inside) == 1).sum() + (inside[:1] == 1).sum()) if t0 > 1 else 0
    ev.append(dict(t=int(t), b=int(b), k=int(k), case=cB[b], vB=float(B.v[max(0, t - 1), b]), vin=vin, dmin=float(dA[j]), jA=j,
                   vA=vA, vinA=vinA, hitA=hitA, passes=passes))
say(f"{nA} -> {nB}: 같은 128 세션, {nB}의 세게 닿음 {len(ev)}번 ({nA} {int((A.d[:T, :, 3:6] > 0).sum())}번)")
for k in range(3):
    E = [e for e in ev if e["k"] == k]
    if not E:
        continue
    passed = [e for e in E if e["dmin"] < 0.4]
    say(f"  [{KN[k]}] {len(E)}번 | 그 자리를 {nA}도 0.4 m 안으로 지난 경우 {len(passed)}번 ({100 * len(passed) / len(E):.0f} %), "
        f"그중 {nA}도 1 m 안에서 세게 닿은 경우 {sum(e['hitA'] for e in passed)}번")
    if passed:
        P2 = [e for e in passed if np.isfinite(e["vinA"])]
        vb = np.array([e["vin"] for e in P2]); va = np.array([e["vinA"] for e in P2])
        say(f"      그 자리에 들어오기 전 1 s의 최고 속도: {nB} 중앙 {np.median(vb):.2f} m/s · 같은 자리 {nA} 중앙 {np.median(va):.2f} m/s · "
            f"{nB}가 0.2 m/s 넘게 더 빠른 경우 {int((vb > va + 0.2).sum())}/{len(vb)}, 더 느린 경우 {int((vb < va - 0.2).sum())}/{len(vb)}")
        vb2 = np.array([e["vB"] for e in passed]); va2 = np.array([e["vA"] for e in passed])
        say(f"      그 자리에서의 속도: {nB} 중앙 {np.median(vb2):.2f} m/s (닿기 직전) · {nA} 중앙 {np.median(va2):.2f} m/s")
    far = [e for e in E if e["dmin"] >= 0.4]
    if far:
        say(f"      {nA}가 그 자리를 지나지 않은 경우 {len(far)}번: {nA}가 가장 가까이 간 거리 중앙 {np.median([e['dmin'] for e in far]):.2f} m")
    for c in ("빈 트랙", "상대", "상자", "상자+상대"):
        nb_ = sum(1 for e in E if e["case"] == c)
        na_ = int((A.d[:T, cA == c, 3 + k] > 0).sum())
        say(f"      {c:<6}: {nA} {na_}번 -> {nB} {nb_}번 (세션 {int((cA == c).sum())}개)")
    rep = np.array([e["passes"] for e in E])
    say(f"      같은 자리를 이미 지나 본 뒤에 닿은 경우 {int((rep > 0).sum())}/{len(E)}")


def draw(ax, e):
    t, b = e["t"], e["b"]; p = B.xy[t, b]; kk = int(B.tid[t, b]) if B.tid.ndim == 2 else int(B.tid[b])
    B.draw_map(ax, kk)
    for poly in B.boxes(t, b):
        ax.fill(poly[:, 0], poly[:, 1], color=RL.C_BOX, alpha=0.85, zorder=2)
    dA = np.linalg.norm(A.xy[:, b] - p, axis=1); inn = dA < 2.5
    seg = np.split(np.arange(T), np.nonzero(np.diff(inn.astype(int)))[0] + 1)
    for s_ in seg:
        if inn[s_[0]] and len(s_) > 1:
            ax.plot(A.xy[s_, b, 0], A.xy[s_, b, 1], "-", color="#1c7ed6", lw=1.3, alpha=0.8, zorder=3)
    j = e["jA"]; bd = A.body(A.xy[j, b, 0], A.xy[j, b, 1], float(A.st[j, b, 2]))
    ax.plot(*np.vstack([bd, bd[:1]]).T, color="#1c7ed6", lw=1.6, zorder=4)
    t0, t1 = max(0, t - int(round(2.0 / dt))), min(T, t + int(round(0.3 / dt)) + 1)
    ax.plot(B.xy[t0:t1, b, 0], B.xy[t0:t1, b, 1], "-o", color="#e03131", lw=1.6, ms=2.2, zorder=5)
    bd = B.body(B.xy[t, b, 0], B.xy[t, b, 1], float(B.st[t, b, 2]))
    ax.plot(*np.vstack([bd, bd[:1]]).T, color="#c92a2a", lw=2.4, zorder=6)
    pl = B.z["pl"][max(0, t - 1), b].astype(float)
    ax.plot(pl[:, 0], pl[:, 1], ".-", color="#f783ac", lw=1.0, ms=3, zorder=6)
    ax.set_xlim(p[0] - 2.5, p[0] + 2.5); ax.set_ylim(p[1] - 2.5, p[1] + 2.5)
    va = "지나지 않음" if not np.isfinite(e["vinA"]) else f"{e['vinA']:.1f} m/s"
    ax.set_title(f"세션 {b} · {e['case']} · {t * dt:.1f} s · {KN[e['k']]}\n{nB} {e['vin']:.1f} m/s 들어옴 / {nA} 같은 자리 {va}"
                 f"{' (A도 닿음)' if e['hitA'] else ''}", fontsize=9)


for k in (0, 1):
    E = sorted([e for e in ev if e["k"] == k], key=lambda e: (e["b"], e["t"]))
    if not E:
        continue
    pick = [E[i] for i in np.unique(np.linspace(0, len(E) - 1, min(n_show, len(E))).round().astype(int))]
    fig, axs = plt.subplots(2, (len(pick) + 1) // 2, figsize=(4.6 * ((len(pick) + 1) // 2), 9.6), squeeze=False)
    for ax, e in zip(axs.flat, pick):
        draw(ax, e)
    for ax in list(axs.flat)[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"{nB}가 {KN[k]}에 세게 닿은 장면 (고르게 뽑음): 빨강 = {nB}가 닿기 2 s 전부터 지난 길(점 0.1 s), 분홍 = 그때 plan, "
                 f"파랑 = 같은 세션 {nA}가 이 자리를 지난 길", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}_{'wall' if k == 0 else 'box'}.png", dpi=95); plt.close(fig)
    say("그림", f"{OUT}_{'wall' if k == 0 else 'box'}.png")
