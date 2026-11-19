"""(11-19, qual26s --mem-aux-seen) policy 자신의 기억 보조 출력(aux, 학습 중 배운 그대로)이 시험 기록의 기억 상태에서 무엇을 아는지.
학습 정답과 같은 정의로 결정마다 라벨: 이 세션에서 이미 지나친(가장 가까이 간 때가 지난) 상자가 트랙 따라 -1..12 m 앞에 있음.
 (가) 학습 정답 그대로, 출발 10 s 뒤 모든 결정: 보조 출력 '가장 가까운 상자 있음' 값의 AUC (보이는 상자 포함 -- 가장 쉬운 판)
 (나) 같은 상자가 3-10 m 앞, 이번 통과에서 아직 안 보임: 전에 지나친 통과 대 첫 통과 (결합 시험 A; 정답은 앞이 1, 뒤가 0)
 (다) (나)의 전에 지나친 통과 대 앞 12 m 안 상자 없음 (시간이 지나면 '있음'이라 말하는 것만으로도 오르는 판)
상자 세션만(상대 차 없음), 상자 치우기 전. 인자: 기록npz(+_mem.npz) 행npy checkpoint"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import Rec

IB, IJ, IT, IN, IGAP, ISEE = range(6)
rec, rows_p, ckp = sys.argv[1], sys.argv[2], sys.argv[3]
R = Rec(rec); z = R.z; dt = R.dt
mem = np.load(rec[:-4] + "_mem.npz")["mem"].astype(np.float32)          # (T, B, 128)
T, B = mem.shape[:2]
ck = torch.load(ckp, map_location="cpu", weights_only=False)
W_, b_ = ck["actor"]["aux.weight"].float().numpy(), ck["actor"]["aux.bias"].float().numpy()
m0 = int(ck["args"].get("memory", 0) or 0) if int(ck["args"].get("box_memory", 0) or 0) else 0
lg = mem[..., m0:] @ W_[0] + b_[0]                                         # (T, B) the nearest listed box: present (logit); the box memory's part
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


def auc(score, yy):
    o = np.argsort(score); rnk = np.empty(len(o)); rnk[o] = np.arange(len(o))
    n1 = yy.sum(); n0 = len(yy) - n1
    return (rnk[yy > 0].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1)


A_s, A_y, K_s, K_y, N_s = [], [], [], [], []
t0 = int(round(10.0 / dt))
for b in np.nonzero(~opp & box)[0]:
    boxes = [d for (bb, j), d in info.items() if bb == b]
    for t in range(t0, k_rm):
        known_ahead, kd, near = False, 0, False
        for d in boxes:
            ps = sorted(d["passes"]); ahead = (d["s"] - s[t, b]) % L
            ahead_signed = ahead if ahead <= 0.5 * L else ahead - L
            if ahead <= 12.0:
                near = True
            passed = any(i < t for i, _ in ps)
            if passed and -1.0 <= ahead_signed <= 12.0:
                known_ahead = True
            nxt = [(it, sg) for it, sg in ps if it >= t]
            if nxt and 3.0 <= ahead <= 10.0 and t < nxt[0][1]:
                kd = 1 if passed else (kd or 2)
        A_s.append(lg[t, b]); A_y.append(known_ahead)
        if kd:
            K_s.append(lg[t, b]); K_y.append(kd == 1)
        elif not near:
            N_s.append(lg[t, b])
A_s, A_y, K_s, K_y, N_s = map(np.array, (A_s, A_y, K_s, K_y, N_s))
print(f"(가) 학습 정답 그대로: AUC {auc(A_s, A_y.astype(float)):.2f} | 정답 있음 {A_y.mean() * 100:.0f} % | '있음'이라 말함: 정답 있음일 때 "
      f"{(A_s[A_y] > 0).mean() * 100:.0f} % · 없음일 때 {(A_s[~A_y] > 0).mean() * 100:.0f} %")
print(f"(나) 3-10 m 앞 안 보이는 상자, 전에 지나침 대 첫 통과: AUC {auc(K_s, K_y.astype(float)):.2f} ({int(K_y.sum())} · {int((~K_y).sum())})")
kp = K_s[K_y]
print(f"(다) 전에 지나친 숨은 상자 대 앞 12 m 안 상자 없음: AUC {auc(np.r_[kp, N_s], np.r_[np.ones(len(kp)), np.zeros(len(N_s))]):.2f} "
      f"({len(kp)} · {len(N_s)})")
