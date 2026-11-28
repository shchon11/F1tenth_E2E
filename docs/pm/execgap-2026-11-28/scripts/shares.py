"""격차의 몫: 같은 64세션 짝지은 부트스트랩(2000번)으로 R0 -> 단계들 -> R4 사이 각 구간이 전체 격차(R0 - R4)의 몇 %인지.
지표: 무접촉 세션 비율, 세게 닿음/세션(벽+상자), 120 s 거리. 인자: 출력json 라벨=기록 ... (첫 라벨 = R0, 마지막 = R4; 사이는 차례)"""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_table import per_session, O

out = sys.argv[1]
items = [a.split("=", 1) for a in sys.argv[2:]]
D = {lab: per_session(f"{O}/{nm}.npz") for lab, nm in items}
labs = [l for l, _ in items]
rng = np.random.default_rng(0)
n = len(D[labs[0]]["clean"])
res = {}
for met, f in (("clean", lambda d: d["clean"]), ("hard", lambda d: d["wall_hard"] + d["box_hard"]), ("dist", lambda d: d["dist"])):
    V = np.stack([f(D[l]) for l in labs])                       # (stages, n)
    idx = rng.integers(0, n, (2000, n))
    M = V[:, idx].mean(-1)                                      # (stages, 2000)
    m0 = V.mean(-1)
    gap = m0[0] - m0[-1]; gapb = M[0] - M[-1]
    r = {}
    for i in range(len(labs) - 1):
        sh = (m0[i] - m0[i + 1]) / gap
        shb = (M[i] - M[i + 1]) / gapb
        r[f"{labs[i]}->{labs[i + 1]}"] = [float(sh), float(np.percentile(shb, 2.5)), float(np.percentile(shb, 97.5)),
                                          float(m0[i] - m0[i + 1])]
    res[met] = {"means": dict(zip(labs, m0.tolist())), "gap": float(gap), "steps": r}
    print(f"== {met}: " + " | ".join(f"{l} {m:.3f}" for l, m in zip(labs, m0)) + f" | 격차 {gap:.3f}")
    for k, v in r.items():
        print(f"   {k}: 몫 {100 * v[0]:.0f} % [{100 * v[1]:.0f}, {100 * v[2]:.0f}] (차 {v[3]:+.3f})")
json.dump(res, open(out, "w"), ensure_ascii=False)
