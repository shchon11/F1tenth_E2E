"""같은 경주를 다시 냈을 때 세게 닿음이 되풀이되나 (11-12: 어려웠던 경주를 다시 내는 학습을 하기 전에 그 전제를 재는 시험).
drive_record.py를 같은 "levels" 시드로 돌린 기록 둘 이상(탐색 잡음 시드만 다름 / 평균 행동)을 받아, 첫 기록 A와 나머지 각각을 견줌:
 - 종류별(벽 / 상자 / 상대 차 / 아무거나)로, 그 종류가 있을 수 있는 경주만(상자: 상자 있는 경주, 차: 상대 있는 경주) 놓고
   A에서 세게 닿은 경주가 B에서도 세게 닿는 비율 vs A에서 안 닿은 경주가 B에서 닿는 비율 (Fisher 정확 검정), B의 세게 닿음/분도 같이
 - 둘 다 세게 닿은 경주에서 닿은 자리가 같은가 (1.5 m 안)
 - A의 점수(세게 닿은 횟수 + 서 있던 시간/2 s)로 경주를 줄 세웠을 때 위 10 / 20 / 30 % 경주가 B의 세게 닿음의 몇 %를 갖고 있나
 - 서 있던 시간, 달린 거리의 순위 상관 (상자·상대 유무가 같은 경주끼리)
인자: 기록 A npz, 기록 B npz [기록 C npz ...]"""
import sys, os
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec

NAMES = ("벽", "상자", "상대 차")


def load(fn):
    r = Rec(fn); z = r.z
    T = r.T
    st = z["st"].astype(float)
    hard = r.d[:, :, 3:6]                                              # (T, B, 3) new hard contacts per decision
    o = dict(fn=fn.split("/")[-1], level=z["level"], tid=z["tid"][0], box=(z["sid"][0] >= 0).any(-1), opp=z["on"][0].astype(bool),
             hard=hard, n=hard.sum(0), stand=(np.abs(st[:T, :, 3]) < 0.3).sum(0) * r.dt,
             dist=np.linalg.norm(st[1:T + 1, :, :2] - st[:T, :, :2], axis=2).clip(max=2.0).sum(0), xy=st[1:T + 1, :, :2], st0=st[0], T=T, dt=r.dt)
    return o


def pair(A, B):
    assert (A["level"] == B["level"]).all(), "경주 시드가 다름"
    same0 = np.abs(A["st0"][:, :3] - B["st0"][:, :3]).max(1) < 1e-4
    Bn = len(A["level"]); mins = A["T"] * A["dt"] / 60.0
    print(f"\n=== A = {A['fn']}  vs  B = {B['fn']} ===  경주 {Bn}개 x {A['T'] * A['dt']:.0f} s, 출발 자세 같은 경주 {100 * same0.mean():.0f} %, "
          f"상자 있는 경주 {int(A['box'].sum())}, 상대 있는 경주 {int(A['opp'].sum())}")
    print(f"세게 닿음 합계  A: 벽 {int(A['n'][:, 0].sum())} 상자 {int(A['n'][:, 1].sum())} 차 {int(A['n'][:, 2].sum())}  |  "
          f"B: 벽 {int(B['n'][:, 0].sum())} 상자 {int(B['n'][:, 1].sum())} 차 {int(B['n'][:, 2].sum())}")
    print("종류    | 놓고 본 경주 | A에서 닿은 경주 | 그 경주가 B에서도 닿음 | A에서 안 닿은 경주가 B에서 닿음 | 배수 | p     | B 세게 닿음/분: A에서 닿은 경주 / 안 닿은 경주")
    for i, nm in list(enumerate(NAMES)) + [(-1, "아무거나")]:
        el = (np.ones(Bn, bool), A["box"], A["opp"])[i] if i >= 0 else np.ones(Bn, bool)
        a = (A["n"][:, i] if i >= 0 else A["n"].sum(1))[el] > 0; bn = (B["n"][:, i] if i >= 0 else B["n"].sum(1))[el]
        b = bn > 0
        n11, n10, n01, n00 = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
        p1 = n11 / max(n11 + n10, 1); p0 = n01 / max(n01 + n00, 1)
        pv = stats.fisher_exact([[n11, n10], [n01, n00]], alternative="greater")[1]
        print(f"{nm:<6} | {int(el.sum()):5d} | {n11 + n10:4d} | {n11:3d}/{n11 + n10:<3d} = {100 * p1:4.1f} % | {n01:3d}/{n01 + n00:<3d} = {100 * p0:4.1f} % | "
              f"x{p1 / max(p0, 1e-9):4.1f} | {pv:.3f} | {bn[a].sum() / max(a.sum() * mins, 1e-9):.2f} / {bn[~a].sum() / max((~a).sum() * mins, 1e-9):.2f}")
    # the same place?
    for i, nm in enumerate(NAMES):
        both = np.nonzero((A["n"][:, i] > 0) & (B["n"][:, i] > 0))[0]
        same = 0
        for b in both:
            pa = A["xy"][A["hard"][:, b, i] > 0, b]; pb = B["xy"][B["hard"][:, b, i] > 0, b]
            same += bool((np.linalg.norm(pa[:, None] - pb[None], axis=2) < 1.5).any())
        print(f"둘 다 {nm}에 세게 닿은 경주 {len(both)}개 중 닿은 자리가 1.5 m 안으로 같은 경주: {same}개")
    # lift: races ranked by A's score hold what share of B's hard hits
    score = A["n"].sum(1) + A["stand"] / 2.0
    order = np.argsort(-score, kind="stable"); tot = B["n"].sum()
    s = " | ".join(f"위 {int(100 * q)} %: {100 * B['n'][order[:int(q * Bn)]].sum() / max(tot, 1):.0f} %" for q in (0.1, 0.2, 0.3))
    print(f"A 점수로 줄 세운 경주가 가진 B의 세게 닿음 몫 -> {s}   (점수 > 0인 경주 {100 * (score > 0).mean():.0f} %)")
    stB = B["stand"]; sc2 = B["n"].sum(1) + B["stand"] / 2.0
    print(f"   같은 방식으로 B의 서 있던 시간 몫 -> " + " | ".join(f"위 {int(100 * q)} %: {100 * stB[order[:int(q * Bn)]].sum() / max(stB.sum(), 1e-9):.0f} %" for q in (0.1, 0.2, 0.3)))
    for nm, m in (("전체", np.ones(Bn, bool)), ("상자+상대", A["box"] & A["opp"]), ("상자만", A["box"] & ~A["opp"]), ("상대만", ~A["box"] & A["opp"]), ("빈 트랙", ~A["box"] & ~A["opp"])):
        if m.sum() < 20:
            continue
        print(f"   {nm:<6} ({int(m.sum()):3d}경주): 세게 닿음/분 A {A['n'][m].sum() / (m.sum() * mins):.2f} B {B['n'][m].sum() / (m.sum() * mins):.2f} | "
              f"순위 상관(A, B) 달린 거리 {stats.spearmanr(A['dist'][m], B['dist'][m])[0]:+.2f}, 점수 {stats.spearmanr(score[m], sc2[m])[0]:+.2f}")


recs = [load(f) for f in sys.argv[1:]]
for B in recs[1:]:
    pair(recs[0], B)
if len(recs) > 2:
    pair(recs[1], recs[2])
