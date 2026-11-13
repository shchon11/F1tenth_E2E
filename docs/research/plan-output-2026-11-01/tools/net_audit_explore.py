"""탐색 잡음이 상자 앞에서 실제로 해 보는 조향·속도 변화의 크기 (11-13 구조 점검).
근거: nudge_value.py -- 상자 1.5~6 m 앞에서 조향을 2도, 0.3 s 동안 상자 반대쪽으로 틀어 주면 지금 보상으로 +0.24~+0.36 s 이득인데 학습은
그것을 못 찾음. 학습이 해 보는 것은 출력(평균 + 표준편차 x 잡음)의 tanh를 그 결정의 조향 범위(조향 속도 한계와 타이어 한계 안)에 편 값이라,
평균이 tanh의 끝(+-1 근처)에 있으면 같은 잡음이 조향각으로는 거의 안 움직임.
물음: 닿는 사건 / 가까이 지나간 사건의 결정마다, 학습 때의 잡음이 첫 0.1 s 조향을 몇 도 움직이나, 2도 넘게 상자 반대쪽으로 가는 일이
얼마나 되나, 평균이 tanh의 끝에 얼마나 붙어 있나.
입력: net_audit_hits.py 값 추적의 OUT.value.npz (actor 본체 입력 + 기억이 들어 있음).   인자: checkpoint value.npz 출력앞머리"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_lib import limits_for, dev
from f1sim.learn.plan_policy import load_actor2
from f1sim import plan_steps as PS

ck = torch.load(sys.argv[1], map_location=dev, weights_only=False)
z = np.load(sys.argv[2]); OUT = sys.argv[3]
LOG = open(OUT + ".txt", "w")
lim = limits_for(ck)
actor = load_actor2(ck, dev, check_mount=False).eval()
NM, NS = actor.norm.mean.clone(), torch.sqrt(actor.norm.var + 1e-8)
STD = torch.maximum(actor.log_std, actor.log_std_floor).exp().detach() if hasattr(actor, "log_std_floor") else actor.log_std.exp().detach()
NS_ = 400                                    # noise draws a decision
G = torch.Generator().manual_seed(0)


def say(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


def measure(xa):
    """per decision: [0 mean of the first turn output before tanh, 1 its tanh, 2 low end of the steering range [deg], 3 high end,
    4 steering of the mean [deg], 5 std of the steering the noise gives [deg], 6 share of draws > 1 deg to the left, 7 > 2 deg left,
    8 > 1 deg right, 9 > 2 deg right, 10 speed of the mean at 0.1 s, 11 std of the speed the noise gives, 12 the range is set by the
    tyres (1) or the servo rate (0) on the low side, 13 same on the high side, 14 speed now]"""
    xa = torch.from_numpy(xa.astype(np.float32)); out = []
    with torch.no_grad():
        for i in range(0, len(xa), 512):
            x = xa[i: i + 512]; n = len(x)
            mean, _, v_hat = actor.heads(actor.trunk(x))
            raw = x[:, 256:292] * NS + NM
            v0 = v_hat.clamp(-PS.V_REV, PS.V_FWD); d0 = raw[:, 6]
            a = torch.tanh(mean)
            d_m, v_m, _ = PS.step_targets(a, v0, d0, lim)
            lo_a, hi_a = a.clone(), a.clone(); lo_a[:, 0] = -1.0; hi_a[:, 0] = 1.0
            d_lo = PS.step_targets(lo_a, v0, d0, lim)[0][:, 0]; d_hi = PS.step_targets(hi_a, v0, d0, lim)[0][:, 0]
            u = mean[:, None, :] + STD[None, None, :] * torch.randn(n, NS_, mean.shape[1], generator=G)
            d_s, v_s, _ = PS.step_targets(torch.tanh(u).reshape(n * NS_, -1), v0.repeat_interleave(NS_), d0.repeat_interleave(NS_), lim)
            dd = torch.rad2deg(d_s[:, 0].reshape(n, NS_) - d_m[:, :1]); dv = v_s[:, 0].reshape(n, NS_) - v_m[:, :1]
            ds = lim.sv_max * PS.DT
            out.append(torch.stack([mean[:, 0], a[:, 0], torch.rad2deg(d_lo), torch.rad2deg(d_hi), torch.rad2deg(d_m[:, 0]), dd.std(1),
                                    (dd > 1).float().mean(1), (dd > 2).float().mean(1), (dd < -1).float().mean(1), (dd < -2).float().mean(1),
                                    v_m[:, 0], dv.std(1), (d_lo > d0 - ds + 1e-4).float(), (d_hi < d0 + ds - 1e-4).float(), v0], 1))
    return torch.cat(out).numpy()


say(f"첫 0.1 s 출력의 표준편차(tanh 앞): 조향 {float(STD[0]):.2f}, 속도 {float(STD[PS.N_STEP]):.2f} · 조향 속도 한계 {np.degrees(lim.sv_max * PS.DT):.1f} 도/결정")
BINS = ((1.0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 6.0), (6.0, 10.0))
RES = {}
for key, tag in (("hit", "세게 닿은 사건"), ("near", "가까이 안 닿고 지나간 사건"), ("clean", "앞에 놓인 상자를 안 닿고 지나간 사건")):
    A = z[key]; M = measure(z[key + "_xa"]); RES[key] = (A, M)
    say(f"[{tag}] 결정 {len(A)}개")
    say("   상자까지 | 결정 수 | 속도 | 조향 범위 폭(도) | 평균의 |tanh| 중앙 · 0.9 넘는 비율 | 잡음이 움직이는 조향 표준편차 중앙(25~75 %) [도] | 0.5도 아래인 비율 |"
        " 상자 반대쪽으로 1도 넘게 / 2도 넘게 가는 잡음의 비율 | 0.3 s 내리 2도 넘게 | 잡음이 움직이는 속도 표준편차 [m/s]")
    for lo, hi in BINS:
        m = (A[:, 1] >= lo) & (A[:, 1] < hi) & (np.abs(A[:, 11]) > 0.03)
        if m.sum() < 20:
            continue
        left = A[m, 11] < 0                                                   # the box is on the right: away = to the left
        p1 = np.where(left, M[m, 6], M[m, 8]); p2 = np.where(left, M[m, 7], M[m, 9])
        say(f"   {lo:.1f}~{hi:.1f} m | {int(m.sum()):5d} | {np.median(M[m, 14]):.1f} | {np.median(M[m, 3] - M[m, 2]):.1f} | {np.median(np.abs(M[m, 1])):.2f} · {np.mean(np.abs(M[m, 1]) > 0.9) * 100:.0f} % |"
            f" {np.median(M[m, 5]):.2f} ({np.percentile(M[m, 5], 25):.2f}~{np.percentile(M[m, 5], 75):.2f}) | {np.mean(M[m, 5] < 0.5) * 100:.0f} % |"
            f" {p1.mean() * 100:.1f} % / {p2.mean() * 100:.1f} % | {(p2 ** 3).mean() * 100:.2f} % | {np.median(M[m, 11]):.2f}")
np.savez_compressed(OUT + ".npz", **{f"{k}_m": v[1] for k, v in RES.items()})
