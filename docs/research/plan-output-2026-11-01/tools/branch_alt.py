"""두 plan policy(--branches 2)가 주행 때 plan을 번갈아 고르는가, 그것이 조향 흔들림을 만드는가, 무엇이 고르기를 정하는가 (11-13).
u25에서 본 것: 고른 plan이 앞 결정과 달라진 결정 65 %, 요 각속도 변화가 뒤집힌 순간의 99 %가 plan 바뀐 직후, 고르기를 정하는 입력은
지난 plan의 점(평균값으로 바꾸면 고르기 60 % 바뀜) -> 지난 plan과 반대쪽을 고르는 좌우 번갈음.
인자: checkpoint 기록npz 기록_mem.npz (exam_record.py, F1SIM_EXAM_SAVE_MEM=1)"""
import sys
import numpy as np
import torch
from f1sim.learn.plan_policy import load_actor2

ck_path, rec, mem = sys.argv[1:4]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
a = load_actor2(ck, torch.device("cpu"), check_mount=False).eval()
z = np.load(rec); m = np.load(mem)
F = torch.tensor(m["feat"]).float(); H = torch.tensor(m["mem"]).float()
T, B, _ = F.shape
br = z["br"][:T].astype(int); rs = z["rs"][:T]; st = z["st"][:T].astype(float); lg = z["logit"][:T]; dt = float(z["dt"])
yaw, v = st[..., 2], st[..., 3]
good = (v > 1.0) & ~rs
sw = br[1:] != br[:-1]
r = np.angle(np.exp(1j * (yaw[1:] - yaw[:-1]))) / dt
d = r[1:] - r[:-1]
flip = (np.sign(d[1:]) != np.sign(d[:-1])) & (np.abs(d[1:]) > 0.05) & (np.abs(d[:-1]) > 0.05)
okf = good[:-3] & good[1:-2] & good[2:-1] & good[3:]
S = sw[1:T - 2] | sw[:T - 3]
p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
print(f"{ck_path.split('/')[-2]}/{ck_path.split('/')[-1]}: 주행 결정 {good.sum()}개 (1 m/s 넘게)")
print(f"  고른 plan이 앞 결정과 달라짐 {sw[good[:-1] & good[1:]].mean() * 100:.1f} % | 고른 plan의 확률 중앙 {np.median(p.max(-1)[good]) * 100:.0f} %")
print(f"  요 각속도 변화 뒤집힘 {flip[okf].mean() * 100:.1f} % | 앞 두 결정 안에 plan 바뀜: {flip[okf & S].mean() * 100:.1f} % / 안 바뀜: "
      f"{flip[okf & ~S].mean() * 100:.1f} % | 뒤집힘 중 plan 바뀜 직후 {(flip & S & okf).sum() / max((flip & okf).sum(), 1) * 100:.1f} % "
      f"(우연히 겹칠 몫 {S[okf].mean() * 100:.1f} %)")
H0 = torch.cat([torch.zeros(1, B, H.shape[2]), H[:-1]], 0)
mu, sd = a.norm.mean, torch.sqrt(a.norm.var + 1e-8)


def choose(Fx):
    with torch.no_grad():
        f = Fx.reshape(T * B, -1); h1 = a.gru(f, H0.reshape(T * B, -1))
        a.heads(a.trunk(torch.cat([f, h1], -1)))
        return a.branch_out.reshape(T, B)


b0 = choose(F)
ok = torch.tensor(good); ok[0] = False
print(f"  다시 계산한 고르기 = 기록 {float((b0.numpy() == br).mean()) * 100:.1f} %")
E = F.shape[2] - mu.numel()
groups = {"IMU": range(0, 6), "지난 조향 명령": [6], "지난 속도 명령": [7], "지난 plan 점": range(8, 26), "지난 plan 속도": range(26, 36),
          "지난번 고른 plan": [36]}
row = []
for nm, idx in groups.items():
    F2 = F.clone()
    for i in idx:
        F2[..., E + i] = 0.0                                 # the input at its training mean
    row.append(f"{nm} {float((choose(F2) != b0)[ok].float().mean()) * 100:.0f} %")
F2 = F.clone(); F2[..., :E] = F[..., :E].mean((0, 1), keepdim=True)
row.append(f"스캔 {float((choose(F2) != b0)[ok].float().mean()) * 100:.0f} %")
print("  입력을 학습 평균값으로 바꿀 때 고르기가 바뀌는 결정: " + " · ".join(row))
