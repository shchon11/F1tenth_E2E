"""(11-19, box_memory_decode.py 다음) 이 크기의 GRU가 policy 자신의 특징(feat)만 읽고 '지난 바퀴에 본, 지금은 안 보이는 앞 상자'를
자리에 묶어 들고 다닐 수 있나 -- 강화학습 없이 그 일만 직접 가르쳐 봄(지도 학습). 되면 = 구조는 되고 강화학습이 그 신호를 못 줬던 것
(보조 학습 신호가 답), 안 되면 = 구조를 바꿔야 함.
데이터: F1SIM_EXAM_SAVE_MEM=1 시험 기록의 feat(결정마다 스캔 인코더 출력 + 정규화 상태) 세션 통째(90 s), 상자 세션만, 상자 치우기 전까지.
정답 y(t) = 이 세션에서 이미 한 번 지나친 상자가 트랙 따라 앞 3-10 m에 있고 이번 통과에서 아직 안 보임. 학습 96세션 / 시험 32세션(4번째마다).
모델: GRU 128(정책과 같은 크기, 같은 오래 들고 있기 시작값 0.2-30 s) + 선형 출력, 세션 통째 BPTT. 비교: 같은 feat로 기억 없는 MLP.
채점: 시험 세션 AUC -- 전체 결정, 그리고 상자가 보이기 0.5-1.5 s 전 결정(양성) 대 같은 세션의 '앞 10 m 안 상자 없음' 결정(음성).
인자: 기록npz(+_mem.npz) 행npy"""
import sys, math, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec

IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
rec, rows_p = sys.argv[1], sys.argv[2]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)
R = Rec(rec); z = R.z; dt = R.dt
feat = np.load(rec[:-4] + "_mem.npz")["feat"].astype(np.float32)               # (T, B, F)
T, B, F = feat.shape
rows = np.load(rows_p)
r_ = z["rem"].astype(float)[:T].copy()
for t in range(1, T):
    bad = ~np.isfinite(r_[t]); r_[t, bad] = r_[t - 1, bad]
L = float(z["lapL"][0]); s = np.mod(L - r_, L)
k_rm = int(round(float(z["rm_t"]) / dt))
opp = z["on"][:T].any(0); box = (z["sid"][0] >= 0).any(-1)
sess = np.nonzero(~opp & box)[0]
# per session and box: its lap position, and its passes (closest-approach step, first-sight step)
info = {}
for r in rows:
    b, j = int(r[IB]), int(r[IJ])
    d = info.setdefault((b, j), dict(s=s[int(r[IT]), b], passes=[]))
    d["passes"].append((int(r[IT]), int(r[IT]) - int(round(r[ISEE] / dt))))
y = np.zeros((T, B), np.float32); w = np.zeros((T, B), np.float32)              # label and which steps count
win = np.zeros((T, B), np.int8)                                                  # 1 = 0.5-1.5 s before sight (2nd+ pass), -1 = nothing within 10 m ahead
for b in sess:
    w[:k_rm, b] = 1.0
    boxes = [(j, d) for (bb, j), d in info.items() if bb == b]
    for t in range(k_rm):
        near = False
        for j, d in boxes:
            ps = sorted(d["passes"])
            ahead = (d["s"] - s[t, b]) % L
            if ahead <= 10.0:
                near = True
            seen_before = any(it < t for it, _ in ps)
            nxt = [(it, sg) for it, sg in ps if it >= t]
            if not nxt or not seen_before:
                continue
            it, sg = nxt[0]
            if 3.0 <= ahead <= 10.0 and t < sg:
                y[t, b] = 1.0
                if sg - 15 <= t <= sg - 5:
                    win[t, b] = 1
        if not near:
            win[t, b] = -1
test_s = sess[(sess % 4) == 0]; train_s = sess[(sess % 4) != 0]
print(f"상자 세션 {len(sess)} (학습 {len(train_s)}, 시험 {len(test_s)}) | 정답 1인 결정 {y[:, sess].sum() / w[:, sess].sum() * 100:.1f} %", flush=True)

mu = feat[:, train_s].reshape(-1, F).mean(0); sd = feat[:, train_s].reshape(-1, F).std(0) + 1e-4
X = torch.tensor((feat - mu) / sd, device=dev).permute(1, 0, 2).contiguous()      # (B, T, F)
Y = torch.tensor(y, device=dev).t().contiguous(); W = torch.tensor(w, device=dev).t().contiguous()
pw = float((1 - y[:, train_s]).sum() / max(y[:, train_s].sum(), 1))


class Mem(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(F, 128, batch_first=True); self.out = nn.Linear(128, 1)
        n = 128; t_max = 30.0 / dt                                               # the policy's keep init: 0.2 .. 30 s, log-spread
        with torch.no_grad():
            tk = torch.exp(torch.empty(n).uniform_(0.0, math.log(t_max))).clamp_min(1.0 + 1e-3)
            self.gru.bias_ih_l0[n:2 * n] = torch.log(tk - 1.0); self.gru.bias_hh_l0[n:2 * n] = 0.0

    def forward(self, x):
        h, _ = self.gru(x)
        return self.out(h)[..., 0]


class NoMem(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(F, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.f(x)[..., 0]


def auc(score, yy):
    o = np.argsort(score); rnk = np.empty(len(o)); rnk[o] = np.arange(len(o))
    n1 = yy.sum(); n0 = len(yy) - n1
    return (rnk[yy > 0].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1)


tr = torch.tensor(train_s, device=dev); te = torch.tensor(test_s, device=dev)
for name, net in (("기억 없는 MLP (그 순간 feat만)", NoMem()), ("GRU 128 (정책 크기)", Mem())):
    net = net.to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    t0 = time.time()
    for ep in range(300):
        for idx in tr[torch.randperm(len(tr), device=dev)].split(16):
            lg = net(X[idx])
            l = (nn.functional.binary_cross_entropy_with_logits(lg, Y[idx], pos_weight=torch.tensor(pw, device=dev), reduction="none") * W[idx]).sum() / W[idx].sum()
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():
        p = net(X[te]).cpu().numpy().T                                          # (T, nb)
    yy, ww, wn = y[:, test_s], w[:, test_s] > 0, win[:, test_s]
    a_all = auc(p[ww], yy[ww])
    m = ww & (wn != 0)
    a_win = auc(p[m], (wn[m] > 0).astype(float))
    with torch.no_grad():
        ptr = net(X[tr]).cpu().numpy().T
    a_tr = auc(ptr[w[:, train_s] > 0], y[:, train_s][w[:, train_s] > 0])
    # first pass vs a later pass of the same boxes, 0.5-1.5 s before sight: only a memory of THIS box separates them -- unless the
    # score just grows with time in the session; the same contrast at places with nothing 10 m ahead (lap 1 vs later) checks that
    fp, lp, n1, n2 = [], [], [], []
    for jb, b in enumerate(test_s):
        for (bb, j), d in info.items():
            if bb != b:
                continue
            for it, sg in sorted(d["passes"]):
                for t in range(max(sg - 15, 0), max(sg - 5, 0) + 1):
                    if t >= k_rm:
                        continue
                    (fp if it == min(i for i, _ in d["passes"]) else lp).append(p[t, jb])
        lapn = np.floor(np.arange(T) * dt / 8.0)
        for t in np.nonzero(wn[:, jb] < 0)[0]:
            (n1 if lapn[t] == 0 else n2).append(p[t, jb])
    a_pass = auc(np.r_[lp, fp], np.r_[np.ones(len(lp)), np.zeros(len(fp))])
    a_time = auc(np.r_[n2, n1], np.r_[np.ones(len(n2)), np.zeros(len(n1))])
    print(f"    같은 상자: 2번째 이후 통과 대 첫 통과(보이기 0.5-1.5 s 전) AUC {a_pass:.2f} ({len(lp)} · {len(fp)}) | "
          f"상자 없는 자리: 둘째 바퀴 이후 대 첫 바퀴 {a_time:.2f} ({len(n2)} · {len(n1)})", flush=True)
    print(f"[{name}] 시험 AUC 전체 {a_all:.2f} (학습 {a_tr:.2f}) | 보이기 0.5-1.5 s 전 대 앞 10 m 비었음 {a_win:.2f} "
          f"(양성 {int((wn[m] > 0).sum())} · 음성 {int((wn[m] < 0).sum())}) | {time.time() - t0:.0f} s", flush=True)
