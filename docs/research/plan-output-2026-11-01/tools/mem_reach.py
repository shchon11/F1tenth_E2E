"""(11-17, 사용자: "기억이 1~2초밖에 안된다는게 뭔소리야") 지금 policy의 GRU 기억이 얼마나 오래 가나를 그대로 잼 (예전 측정을 믿지 않고).
exam_record F1SIM_EXAM_SAVE_MEM=1 기록(결정마다 GRU가 읽은 입력 feat과 그 뒤 상태 mem)으로, 같은 입력을 그대로 다시 넣어:
 1) 값마다 '그대로 남기는 몫'(update gate z): 혼자 두면 한 결정에 z만큼 남음 -> 남는 시간 1/(1-z) 결정 = 0.1 s 단위
 2) 지운 기억과 안 지운 기억을 같은 입력으로 함께 굴려, 두 상태의 차가 처음의 50 % / 10 % / 1 %로 줄 때까지 몇 초 (= 지난 일이 남는 한계)
인자: checkpoint 기록npz(…_mem.npz가 옆에 있어야 함)"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn.plan_policy import load_actor2

ck_path, rec = sys.argv[1], sys.argv[2]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
actor = load_actor2(ck, torch.device("cpu"), check_mount=False).eval()
gru = actor.gru; n = gru.hidden_size
M = np.load(rec[:-4] + "_mem.npz"); Z = np.load(rec, allow_pickle=True)
feat = torch.tensor(M["feat"].astype(np.float32)); mem = torch.tensor(M["mem"].astype(np.float32))
rs = torch.tensor(Z["rs"]); T, B = rs.shape
DT = float(Z["dt"])
Wi, Wh, bi, bh = gru.weight_ih.detach(), gru.weight_hh.detach(), gru.bias_ih.detach(), gru.bias_hh.detach()
with torch.no_grad():
    # 1) update gates along the recorded drive (decisions with no restart in between)
    zs = []
    for t in range(1, T, 3):
        ok = ~rs[t - 1]
        x, h = feat[t][ok], mem[t - 1][ok]
        gi, gh = x @ Wi.T + bi, h @ Wh.T + bh
        zs.append(torch.sigmoid(gi[:, n:2 * n] + gh[:, n:2 * n]))
    zz = torch.cat(zs).mean(0).numpy()                              # per value, the mean share kept a decision
    tau = DT / np.clip(1.0 - zz, 1e-6, None)
    print(f"{ck_path.split('/')[-2]}/{ck_path.split('/')[-1]}: 기억 값 {n}개가 혼자 두면 남는 시간(1/(1-z) x 0.1 s): 중앙 {np.median(tau):.2f} s, "
          f"90 % {np.percentile(tau, 90):.2f} s, 최대 {tau.max():.1f} s | 1 s 넘게 {int((tau > 1).sum())}개 · 3 s 넘게 {int((tau > 3).sum())}개 · "
          f"7 s(한 바퀴) 넘게 {int((tau > 7).sum())}개")
    # 2) wiped vs kept, the same inputs from then on
    H = int(round(12.0 / DT))
    out = []
    for t0 in (150, 300, 450):
        ok = ~rs[t0:t0 + H].any(0)
        ha, hb = mem[t0 - 1][ok].clone(), torch.zeros_like(mem[t0 - 1][ok])
        d0 = (ha - hb).norm(dim=1)
        r = []
        for k in range(H):
            x = feat[t0 + k][ok]
            ha, hb = gru(x, ha), gru(x, hb)
            r.append(((ha - hb).norm(dim=1) / d0.clamp_min(1e-6)).median().item())
        out.append(r)
    r = np.median(np.array(out), 0)
    def when(f):
        k = np.nonzero(r < f)[0]
        return f"{(k[0] + 1) * DT:.1f} s" if len(k) else f"> {H * DT:.0f} s"
    print(f"  지운 기억 vs 그대로 둔 기억, 같은 입력으로: 차이가 50 %로 {when(0.5)}, 10 %로 {when(0.1)}, 1 %로 {when(0.01)} | "
          f"1 s 뒤 {100 * r[int(1 / DT) - 1]:.0f} %, 3 s 뒤 {100 * r[int(3 / DT) - 1]:.0f} %, 7 s 뒤 {100 * r[int(7 / DT) - 1]:.1f} %")
