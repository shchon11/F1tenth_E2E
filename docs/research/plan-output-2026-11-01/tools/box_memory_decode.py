"""(11-19, feature_decode.py 다음) 기억이 '이 앞에 상자가 있다'를 자리와 함께 들고 있나: 상자가 스캔에 보이기 0.3 s 전(아직 안 보임),
같은 세션에서 그 상자를 이미 한 번 이상 지나친 통과(2번째 이후)의 결정 = 양성. 음성 = 다른 상자 세션의 같은 자리(한 바퀴 위치 ±0.3 m),
상자를 치우기 전, 출발 10 s 뒤, 그 세션에는 앞 10 m 안에 상자가 없는 결정. 둘 다 '이 세션엔 상자가 있다'는 같고 다른 것은 '이 자리 앞에
지금 상자가 있나'뿐 -> 기억(mem)에서 가를 수 있으면 자리에 묶인 상자 정보를 들고 있는 것. 그 순간 스캔 값(feat)은 상자를 못 보므로 비교 기준.
decoder = 로지스틱 회귀(입력 표준화, L2), 세션으로 나눠(4번째마다 시험) AUC.
인자: 기록npz(+_mem.npz) 행npy [보이기 몇 결정 전, 3 = 0.3 s] [둘레 m 안에 (다른) 상자 없을 때만, 0 = 끔]"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec

IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
rec, rows_p = sys.argv[1], sys.argv[2]
LEAD = int(sys.argv[3]) if len(sys.argv) > 3 else 3          # decisions before the box centre is first seen
R = Rec(rec); z = R.z; dt = R.dt
M = np.load(rec[:-4] + "_mem.npz")
feat, mem = M["feat"].astype(np.float32), M["mem"].astype(np.float32)
T, B = feat.shape[:2]
rows = np.load(rows_p)
r_ = z["rem"].astype(float)[:T].copy()
for t in range(1, T):
    bad = ~np.isfinite(r_[t]); r_[t, bad] = r_[t - 1, bad]
L = float(z["lapL"][0]); s = np.mod(L - r_, L)                                   # lap position of each decision
k_rm = int(round(float(z["rm_t"]) / dt))
opp = z["on"][:T].any(0); box = (z["sid"][0] >= 0).any(-1); bs = ~opp & box
ok = ~R.rs[:T].astype(bool)
# each session's box lap positions (from its passes: the car's lap position at the closest approach)
box_s = {}
for r in rows:
    b, j = int(r[IB]), int(r[IJ]); box_s.setdefault(b, {})[j] = s[int(r[IT]), b]
CLEAR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0          # > 0: no (other) box within this many m of the car at the moment
BXY = {b: z["pose"][0, b, z["sid"][0, b] >= 0, :2].astype(float) for b in range(B)}


def near_box(t, b, skip=-1):
    # along the lap, either way: walls block the view across the track, a straight Euclidean radius would count the far side
    # 0 if another box lies in the forward view window (3 m behind .. CLEAR m ahead along the lap), else 1e9
    for j, bj in box_s.get(b, {}).items():
        if j == skip:
            continue
        rel = ((bj - s[t, b]) + 0.5 * L) % L - 0.5 * L
        if -3.0 <= rel <= CLEAR:
            return 0.0
    return 1e9


pos = []
for r in rows:
    b = int(r[IB])
    if r[IN] < 2 or not bs[b]:
        continue
    t0 = int(r[IT]) - int(round(r[ISEE] / dt)) - LEAD
    if t0 < int(10.0 / dt) or t0 >= k_rm or not ok[t0, b]:
        continue
    if CLEAR and near_box(t0, b, skip=int(r[IJ])) < CLEAR:                       # another box in sight range: the scan would tell
        continue
    pos.append((t0, b))
pos = np.array(pos)
print(f"양성(2번째 이후 통과, 보이기 0.3 s 전) {len(pos)}개", flush=True)


def ahead_clear(b, s0, d=10.0):
    return all(((bj - s0) % L) > d for bj in box_s.get(b, {}).values())


rng = np.random.default_rng(0)
neg = []
cand_t = np.arange(int(10.0 / dt), k_rm)
bs_idx = np.nonzero(bs)[0]
for (t0, b) in pos:
    s0 = s[t0, b]
    for _ in range(3):                                                           # up to 3 matched negatives a positive
        for _try in range(40):
            b2 = int(rng.choice(bs_idx))
            if b2 == b:
                continue
            m = cand_t[(np.abs(((s[cand_t, b2] - s0 + 0.5 * L) % L) - 0.5 * L) < 0.3) & ok[cand_t, b2]]
            if len(m) and ahead_clear(b2, s0):
                t2 = int(rng.choice(m))
                if CLEAR and near_box(t2, b2) < CLEAR:
                    continue
                neg.append((t2, b2)); break
neg = np.array(neg)
print(f"음성(다른 상자 세션의 같은 자리, 앞 10 m에 상자 없음) {len(neg)}개", flush=True)
idx = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
test = (idx[:, 1] % 4) == 0


def auc(score, yy):
    o = np.argsort(score); rnk = np.empty(len(o)); rnk[o] = np.arange(len(o))
    n1 = yy.sum(); n0 = len(yy) - n1
    return (rnk[yy > 0].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1)


for name, X in (("feat (그 순간 스캔 값)", feat[idx[:, 0], idx[:, 1]]), ("mem (기억 상태)", mem[idx[:, 0], idx[:, 1]]),
                ("feat+mem", np.concatenate([feat[idx[:, 0], idx[:, 1]], mem[idx[:, 0], idx[:, 1]]], 1))):
    mu, sd = X[~test].mean(0), X[~test].std(0) + 1e-4
    Xt = torch.tensor((X - mu) / sd); yt = torch.tensor(y, dtype=torch.float32)
    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        w = torch.zeros(X.shape[1], requires_grad=True); b0 = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b0], max_iter=200)
        tr = torch.tensor(~test)
        def closure():
            opt.zero_grad()
            l = torch.nn.functional.binary_cross_entropy_with_logits(Xt[tr] @ w + b0, yt[tr]) + lam * (w ** 2).sum() / tr.sum()
            l.backward(); return l
        opt.step(closure)
        with torch.no_grad():
            sc = (Xt @ w + b0).numpy()
        a_tr, a_te = auc(sc[~test], y[~test]), auc(sc[test], y[test])
        if best is None or a_te > best[1]:
            best = (lam, a_te, a_tr)
    print(f"[{name}] 시험 세션 AUC {best[1]:.2f} (학습 {best[2]:.2f}, L2 {best[0]:g}) | 양성 {int(y[test].sum())} · 음성 {int((1 - y[test]).sum())}", flush=True)
