"""(11-19, gru_bind_test.py의 큰 데이터판) 이 크기의 GRU가 policy 자신의 특징(feat)만 읽고 '지난 바퀴에 본, 지금은 안 보이는 앞 상자'를
자리에 묶어 들고 다닐 수 있나 -- 강화학습 없이 직접 가르침. 기록 여러 개(F1SIM_EXAM_SAVE_MEM=1, 상자 안 치움 권장)를 세션으로 이어 씀.
정답 y(t) = 이 세션에서 이미 한 번 지나친 상자가 트랙 따라 앞 3-10 m에 있고 이번 통과에서 아직 안 보임 (상자 세션만, 치우기 전).
채점(시험 세션, 4번째마다):
 A. 같은 상자: 2번째 이후 통과 대 첫 통과, 보이기 0.5-1.5 s 전 -- 그 상자를 기억해야만 가름. 대조: 상자 없는 자리의 둘째 바퀴 이후 대 첫 바퀴
    (점수가 세션 뒤쪽이라서 오르는 몫).
 B. 짝 맞춘 비교: 둘째 바퀴 이후, 보이기 0.5-1.5 s 전(양성) 대 다른 세션의 같은 자리(±0.3 m)·'방금 지난 상자로부터 걸린 시간' ±0.5 s·
    앞 10 m 안 상자 없음(음성) -- 상자 간격 규칙(바퀴 3분의 1마다 하나) 같은 단서를 지움. 기억 없는 신경망은 여기서 0.5 근처여야 함.
모델: GRU 128(정책과 같은 크기·같은 오래 들고 있기 시작값) + 선형, 세션 통째 BPTT / 기억 없는 MLP.
인자: 기록npz:행npy [기록npz:행npy ...]
(11-19) STEPS=100,300,... 이면 그 걸음 수마다 채점(학습 걸음 수에 따른 곡선), LR=학습률(기본 1e-3), 변형 gru128p = policy checkpoint
(POLICY_CK)의 GRU 가중치에서 시작하고 policy와 같은 날 특징(표준화 안 함)을 읽음 -- 강화학습 중 기억 보조 문제가 받는 조건에 가깝게."""
import sys, math, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec

IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)
FEAT, Ys, Ws, S_, TL, SIGHTW, FIRST, NOB, LAP, KD, LAYK = [], [], [], [], [], [], [], [], [], [], []
for arg in sys.argv[1:]:
    rec, rows_p = arg.split(":")
    R = Rec(rec); z = R.z; dt = R.dt
    feat = np.load(rec[:-4] + "_mem.npz")["feat"].astype(np.float32)
    T, B, F = feat.shape
    rows = np.load(rows_p)
    r_ = z["rem"].astype(float)[:T].copy()
    for t in range(1, T):
        bad = ~np.isfinite(r_[t]); r_[t, bad] = r_[t - 1, bad]
    L = float(z["lapL"][0]); s = np.mod(L - r_, L)
    k_rm = min(int(round(float(z["rm_t"]) / dt)), T)
    opp = z["on"][:T].any(0); box = (z["sid"][0] >= 0).any(-1)
    info = {}
    for r in rows:
        b, j = int(r[IB]), int(r[IJ])
        d = info.setdefault((b, j), dict(s=s[int(r[IT]), b], passes=[]))
        d["passes"].append((int(r[IT]), int(r[IT]) - int(round(r[ISEE] / dt))))
    for b in np.nonzero(~opp & box)[0]:
        y = np.zeros(T, np.float32); w = np.zeros(T, np.float32); w[:k_rm] = 1.0
        sightw = np.zeros(T, np.int8)                 # 1 = 0.5-1.5 s before a box is first seen on this pass
        first = np.zeros(T, np.int8)                  # 1 = that pass is the box's first pass
        nob = np.zeros(T, bool)                       # nothing within 10 m ahead
        tl = np.full(T, 99.0, np.float32)             # s since the last closest approach to any box
        kd = np.zeros(T, np.int8)                     # box 3-10 m ahead and not yet seen on this pass: 1 = seen on an earlier pass, 2 = first pass
        boxes = [d for (bb, j), d in info.items() if bb == b]
        last_it = sorted(it for d in boxes for it, _ in d["passes"])
        for t in range(k_rm):
            prev = [it for it in last_it if it < t]
            if prev:
                tl[t] = (t - prev[-1]) * dt
            near = False
            for d in boxes:
                ps = sorted(d["passes"]); ahead = (d["s"] - s[t, b]) % L
                if ahead <= 10.0:
                    near = True
                nxt = [(it, sg) for it, sg in ps if it >= t]
                if not nxt:
                    continue
                it, sg = nxt[0]
                if sg - 15 <= t <= sg - 5:
                    sightw[t] = 1; first[t] = 1 if it == ps[0][0] else 0
                if 3.0 <= ahead <= 10.0 and t < sg:
                    if any(i < t for i, _ in ps):
                        y[t] = 1.0; kd[t] = 1
                    elif kd[t] == 0:
                        kd[t] = 2
            nob[t] = not near
        FEAT.append(feat[:, b]); Ys.append(y); Ws.append(w); S_.append(s[:, b]); TL.append(tl)
        lv = z["sid"][0, b] >= 0
        LAYK.append(tuple(sorted(map(tuple, np.round(z["pose"][0, b, lv, :2], 1).tolist()))))
        SIGHTW.append(sightw); FIRST.append(first); NOB.append(nob); LAP.append(np.floor(np.arange(T) * dt / 8.0)); KD.append(kd)
feat = np.stack(FEAT); Y = np.stack(Ys); W = np.stack(Ws); Sx = np.stack(S_); TLx = np.stack(TL)
SW = np.stack(SIGHTW); FI = np.stack(FIRST); NB = np.stack(NOB); LP = np.stack(LAP); KDx = np.stack(KD)
N, T, F = feat.shape
# split by box LAYOUT, not by session: the exam reuses its 64 layouts across sessions and records, and a net could learn a layout
# in training and recall it in a test session without having seen its boxes there
lay_ids = {k: i for i, k in enumerate(sorted(set(LAYK)))}
lay = np.array([lay_ids[k] for k in LAYK])
test = (lay % 4) == 0
print(f"배치 {len(lay_ids)}가지, 시험 배치 {int(len({l for l in lay[test]}))}가지(학습 세션과 겹치지 않음)", flush=True)
print(f"상자 세션 {N} (학습 {int((~test).sum())}, 시험 {int(test.sum())}) | 정답 1인 결정 {Y[W > 0].mean() * 100:.1f} %", flush=True)
mu = feat[~test].reshape(-1, F).mean(0); sd = feat[~test].reshape(-1, F).std(0) + 1e-4
import os
_want_raw = any(k.endswith("p") for k in os.environ.get("MEMV", "mlp,gru128").split(","))
_want_std = any(not k.endswith("p") for k in os.environ.get("MEMV", "mlp,gru128").split(","))
X = torch.tensor((feat - mu) / sd, device=dev) if _want_std else None
Xraw = torch.tensor(feat, device=dev) if _want_raw else None
Yt = torch.tensor(Y, device=dev); Wt = torch.tensor(W, device=dev)
pw = float((1 - Y[~test][W[~test] > 0]).sum() / max(Y[~test][W[~test] > 0].sum(), 1))


class Mem(nn.Module):
    def __init__(self, n=128, kind="gru", drop=0.0):
        super().__init__()
        self.kind = kind
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(F, n, batch_first=True); self.out = nn.Linear(n, 1)
        self.drop = nn.Dropout(drop)
        if kind == "gru":
            t_max = 30.0 / 0.1
            with torch.no_grad():
                tk = torch.exp(torch.empty(n).uniform_(0.0, math.log(t_max))).clamp_min(1.0 + 1e-3)
                self.rnn.bias_ih_l0[n:2 * n] = torch.log(tk - 1.0); self.rnn.bias_hh_l0[n:2 * n] = 0.0

    def forward(self, x):
        return self.out(self.rnn(self.drop(x))[0])[..., 0]


class MemP(Mem):
    """the policy's own GRU (checkpoint POLICY_CK) and a fresh read-out"""
    def __init__(self):
        super().__init__(128)
        sd_ = torch.load(os.environ["POLICY_CK"], map_location="cpu", weights_only=False)["actor"]
        with torch.no_grad():
            for a in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(self.rnn, a + "_l0").copy_(sd_["gru." + a])


class NoMem(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(F, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.f(x)[..., 0]


def auc(score, yy):
    if len(yy) == 0 or yy.sum() == 0 or yy.sum() == len(yy):
        return float("nan")
    o = np.argsort(score); rnk = np.empty(len(o)); rnk[o] = np.arange(len(o))
    n1 = yy.sum(); n0 = len(yy) - n1
    return (rnk[yy > 0].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1)


# B: matched negatives for every positive moment (lap >= 2, 0.5-1.5 s before sight) in the test sessions
te_idx = np.nonzero(test)[0]; rng = np.random.default_rng(0)
posB = [(i, t) for i in te_idx for t in np.nonzero((KDx[i] == 1) & (LP[i] >= 1) & (W[i] > 0))[0]]
cand = {i: np.nonzero(NB[i] & (LP[i] >= 1) & (W[i] > 0))[0] for i in te_idx}
negB = []
for (i, t) in posB:
    for _ in range(30):
        i2 = int(rng.choice(te_idx))
        if i2 == i:
            continue
        c = cand[i2]
        m = c[(np.abs(((Sx[i2, c] - Sx[i, t]) + 0.5 * L) % L - 0.5 * L) < 0.3) & (np.abs(TLx[i2, c] - TLx[i, t]) < 0.5)]
        if len(m):
            negB.append((i2, int(rng.choice(m)))); break
posB, negB = np.array(posB), np.array(negB)
print(f"짝 맞춘 비교 B: 양성 {len(posB)} · 음성 {len(negB)}", flush=True)
tr = torch.tensor(np.nonzero(~test)[0], device=dev); te = torch.tensor(te_idx, device=dev)
import os
VAR = {"mlp": ("기억 없는 MLP", lambda: NoMem(), 0.0), "gru128": ("GRU 128", lambda: Mem(128), 0.0),
       "gru128r": ("GRU 128 + 입력 dropout 0.3 + 가중치 감쇠", lambda: Mem(128, drop=0.3), 1e-4),
       "gru256": ("GRU 256", lambda: Mem(256), 0.0), "gru256r": ("GRU 256 + dropout + 감쇠", lambda: Mem(256, drop=0.3), 1e-4),
       "lstm128": ("LSTM 128", lambda: Mem(128, kind="lstm"), 0.0),
       "gru128p": ("policy의 GRU에서 시작", lambda: MemP(), 0.0)}
STEPS = sorted(int(x) for x in os.environ.get("STEPS", "").split(",") if x)
LR = float(os.environ.get("LR", "1e-3"))


def evaluate(net, Xin, name, tag, t0):
    with torch.no_grad():
        P = np.concatenate([net(Xin[c]).cpu().numpy() for c in torch.arange(N, device=dev).split(64)])   # (N, T), in chunks
    ww = (W[test] > 0)
    a_all = auc(P[test][ww], Y[test][ww]); a_tr = auc(P[~test][W[~test] > 0], Y[~test][W[~test] > 0])
    kk = (KDx[test] > 0) & ww                      # the same boxes 3-10 m ahead, hidden: seen before (1) vs first pass (2)
    a_pass = auc(P[test][kk], (KDx[test][kk] == 1).astype(float))
    nb = NB[test] & ww
    a_time = auc(P[test][nb], (LP[test][nb] >= 1).astype(float))
    sc = np.r_[P[posB[:, 0], posB[:, 1]], P[negB[:, 0], negB[:, 1]]] if len(negB) else np.array([])
    a_B = auc(sc, np.r_[np.ones(len(posB)), np.zeros(len(negB))]) if len(negB) else float("nan")
    print(f"[{name}]{tag} 전체 AUC 시험 {a_all:.2f} (학습 {a_tr:.2f}) | A 같은 상자 이후 대 첫 통과 {a_pass:.2f} (시간 효과 대조 {a_time:.2f}) | "
          f"B 짝 맞춤 {a_B:.2f} | {time.time() - t0:.0f} s", flush=True)

for key in os.environ.get("MEMV", "mlp,gru128").split(","):
    name, mk, wd = VAR[key]; net = mk()
    Xin = Xraw if key.endswith("p") else X
    net = net.to(dev); opt = torch.optim.Adam(net.parameters(), LR, weight_decay=wd)
    t0 = time.time(); step = 0; last = STEPS[-1] if STEPS else None
    for ep in range(200 if not STEPS else 10 ** 6):
        for idx in tr[torch.randperm(len(tr), device=dev)].split(32):
            lg = net(Xin[idx])
            l = (nn.functional.binary_cross_entropy_with_logits(lg, Yt[idx], pos_weight=torch.tensor(pw, device=dev), reduction="none") * Wt[idx]).sum() / Wt[idx].sum()
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            step += 1
            if step in STEPS:
                evaluate(net, Xin, name, f" 걸음 {step} (lr {LR:g})", t0)
            if last is not None and step >= last:
                break
        if last is not None and step >= last:
            break
    if not STEPS:
        evaluate(net, Xin, name, "", t0)
