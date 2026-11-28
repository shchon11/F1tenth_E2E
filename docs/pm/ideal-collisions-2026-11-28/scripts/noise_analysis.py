"""H1: 학습 잡음(noise_probe.py)으로 몬 기록에서 결정 하나의 조향 잡음이 상자 통과 여유를 얼마나 옮기나.
만남마다 가장 가까운 순간(세게 닿았으면 그 결정) te. lag k = 1..15 (te-k 결정)의 첫 단계 조향 잡음(표준화 e0)을 '상자에서 멀어지는 쪽'
부호로 바꿈: away = -sign(y_box) * e0 (y_box = 그 결정 차 틀에서 상자 중심 옆 위치, + 왼쪽; 조향 + = 왼쪽). 여유(부호 있음, m)를 lag별 away에
한꺼번에 회귀(상자마다 평균을 뺀 고정효과). 계수 = 그 lag에서 잡음 1 sd가 여유를 옮긴 cm. 같은 세션 평균 행동 기록(probe32)과 접촉률 비교.
인자: events_noise32.json noise32_noise.npz events_probe32.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_lib import Rec

E = json.load(open(sys.argv[1]))["noise32"]
N = np.load(sys.argv[2]); E0 = json.load(open(sys.argv[3]))["probe32"]
R = Rec(E["file"]); st = R.st.astype(float)
e0, sd0, zone = N["e0"], N["sd0"], N["zone"]
print(f"조향 잡음 sd(pre-tanh) 중앙 {np.median(sd0):.3f}, 속도 잡음 sd {np.median(N['sd10']):.3f}, 이어 간(상자 1.5-4 m) 결정 {100 * zone.mean():.1f} %")
hard_at = {}
for h in E["hard"]:
    hard_at.setdefault((h["b"], h["j"]), []).append(h["t"])
K = 15
X, Y, G, H = [], [], [], []
for e in E["enc"]:
    b, j = e["b"], e["j"]
    hits = [t for t in hard_at.get((b, j), []) if e["t0"] - 1 <= t <= e["t1"] + 1]
    te = hits[0] if hits else int(round(e["tmin"]))
    if te - K < 0 or (e["v"] < 1.0 and not hits):
        continue
    c = R.boxes(te, b)[j].mean(0)
    tt = np.arange(te - K, te)[::-1]                          # lag 1..K
    yb = -(c[0] - st[tt, b, 0]) * np.sin(st[tt, b, 2]) + (c[1] - st[tt, b, 1]) * np.cos(st[tt, b, 2])
    away = -np.sign(yb) * e0[tt, b]
    X.append(away); Y.append(e["clear"]); G.append(f"{b}:{j}"); H.append(bool(hits) or e["any"] > 0)
X, Y, G, H = np.array(X), np.array(Y), np.array(G), np.array(H)
# box fixed effects: demean within box
Xd, Yd = X.copy(), Y.copy()
for g in np.unique(G):
    m = G == g
    Xd[m] -= X[m].mean(0); Yd[m] -= Y[m].mean()
coef = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
res = Yd - Xd @ coef
rng = np.random.default_rng(0); ug = np.unique(G); idx = {g: np.nonzero(G == g)[0] for g in ug}; bs = []
for _ in range(500):
    s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
    bs.append(np.linalg.lstsq(Xd[s], Yd[s], rcond=None)[0])
bs = np.array(bs)
print(f"만남 {len(Y)}개 (닿음 {H.sum()}), 상자 {len(ug)}개 | 여유 = sum_k c_k * (lag k 잡음, 멀어지는 쪽 +) + 상자 평균")
for k in range(K):
    print(f"  lag {k + 1:2d} ({0.1 * (k + 1):.1f} s 전): 잡음 1 sd당 여유 {100 * coef[k]:+.2f} cm [{100 * np.percentile(bs[:, k], 2.5):+.2f}, {100 * np.percentile(bs[:, k], 97.5):+.2f}]")
r2 = 1 - (res ** 2).sum() / (Yd ** 2).sum()
print(f"  상자 안 여유 분산 중 15결정 잡음이 설명하는 몫 R2 = {r2:.3f}; 상자 안 여유 sd {100 * Yd.std():.1f} cm")
# contact rates: sampled vs mean action on the same sessions
for nm, EE in (("평균 행동(probe32)", E0), ("학습 잡음(noise32)", E)):
    enc = [e for e in EE["enc"] if e["v"] > 1.0 or e["hard"] > 0]
    cl = np.array([e["clear"] for e in enc if e["any"] == 0 and e["v"] > 1.0])
    print(f"  {nm}: 만남 {len(enc)}, 세게 닿은 만남 {sum(e['hard'] > 0 for e in enc)} ({100 * np.mean([e['hard'] > 0 for e in enc]):.2f} %), "
          f"닿음 {sum(e['any'] > 0 for e in enc)} ({100 * np.mean([e['any'] > 0 for e in enc]):.2f} %), 안 닿은 통과 여유 중앙 {100 * np.median(cl):.0f} cm, 10 cm 안 {100 * np.mean(cl < 0.1):.0f} %")
