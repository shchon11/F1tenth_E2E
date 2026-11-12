"""기억(GRU)을 거쳐 학습 신호가 얼마나 먼 과거까지 닿는가 (11-12; gru_forget.py는 '값이 얼마나 오래 남나', 이것은 '배울 수 있나').
학습은 지금의 출력이 좋았는지 나빴는지를 기억을 거슬러 과거로 보내서(시간을 거슬러 미분) "그때 그것을 기억해 뒀어야 했다"를 가르침.
그 신호는 결정 하나를 거슬러 갈 때마다 GRU의 한 걸음 미분만큼 곱해져 줄어듦. 여기서는 기록된 입력을 그대로 넣고,
 지금의 기억(무작위 방향 하나)을 k 결정 전의 기억으로 미분한 크기 = k 결정 전으로 보낸 신호가 몇 배로 줄어드나
를 k마다 잼. 세 가지 GRU에 같은 입력을 넣어 비교:
 - 학습된 것 (checkpoint의 GRU; 지금의 plan 출력에서 k 결정 전 기억까지도 따로 잼)
 - 학습을 시작할 때의 것: PyTorch 기본 초기값의 새 GRU (씨앗 3개; 기억은 0에서 시작해 60 결정을 먼저 돌림)
 - 붙드는 문의 bias만 다르게 초기화한 새 GRU: 값마다 붙드는 시간이 1 ~ TMAX 결정에 고루 퍼지게 (Tallec & Ollivier 2018, chrono init)
입력은 세 경우 모두 학습된 policy의 기록(시험 기록의 _mem.npz)이라, 새 GRU의 숫자는 '학습 첫날 그대로'가 아니라 같은 입력에서의 비교임.
인자: checkpoint 기록npz [그림png]"""
import sys
import numpy as np
import torch
from torch import nn
from f1sim.learn.plan_policy import load_actor2
from rec_lib import Rec, plt, along_track

KS = [1, 3, 5, 10, 20, 40, 70, 100, 140]
K, BURN, NV, TMAX, NMAX = max(KS), 60, 4, 300, 500
torch.manual_seed(0); rng = np.random.default_rng(0)
ck_path, npz = sys.argv[1], sys.argv[2]
fig_out = sys.argv[3] if len(sys.argv) > 3 else ""
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
actor = load_actor2(ck, torch.device("cpu"), check_mount=False)
for p_ in actor.parameters():
    p_.requires_grad_(False)
assert actor.mem_front, "기억이 몸통 앞에 있는 policy만"
R = Rec(npz); z = R.z; T, B, dt = R.T, R.B, R.dt
mm = np.load(npz[:-4] + "_mem.npz")
F = torch.tensor(mm["feat"].astype(np.float32)); H = torch.tensor(mm["mem"].astype(np.float32))
NU, NI = actor.memory, F.shape[2]
tb = np.array([(t, b) for t in np.arange(K + BURN + 2, T, 29) for b in range(B) if R.k0[t, b] <= t - K - BURN - 1])
tb = tb[rng.permutation(len(tb))[:NMAX]]
tt, bb = torch.tensor(tb[:, 0]), torch.tensor(tb[:, 1]); N = len(tb)
s, lap, L = along_track(z, T)
solo = ~R.on[:T].any(0)
lap_s = float(np.median([(T - 101) * dt / max(((lap[T - 1, b] * L + s[T - 1, b]) - (lap[100, b] * L + s[100, b])) / L, 1e-6) for b in np.nonzero(solo)[0]]))


def keep_gate(gru, x, h):
    gi = x @ gru.weight_ih.T + gru.bias_ih; gh = h @ gru.weight_hh.T + gru.bias_hh
    return torch.sigmoid(gi[:, NU: 2 * NU] + gh[:, NU: 2 * NU])


def back(gru, h0, out=False):
    """feed the K decisions before each sample from the state h0; returns (K+1, N): |d(a random direction of the memory now) /
    d(memory k decisions ago)|, with `out` the same from a random direction of the plan's 20 outputs; and the keep gates now"""
    hs = [h0.clone().requires_grad_(True)]
    for i in range(K):
        hs.append(gru(F[tt - K + 1 + i, bb], hs[-1]))
    res = np.zeros((K + 1, N))
    for _ in range(NV):
        if out:
            o = torch.tanh(actor.heads(actor.trunk(torch.cat([F[tt, bb], hs[-1]], 1)))[0])
            v = torch.randn_like(o); v = v / v.norm(dim=1, keepdim=True); y = (o * v).sum()
        else:
            v = torch.randn(N, NU); v = v / v.norm(dim=1, keepdim=True); y = (hs[-1] * v).sum()
        gs = torch.autograd.grad(y, hs, retain_graph=True)
        res += np.stack([g.norm(dim=1).numpy() for g in gs])[::-1] / NV          # index = decisions back
    with torch.no_grad():
        zk = keep_gate(gru, F[tt, bb], hs[-2].detach())
    return res, zk.numpy()


def burn(gru):
    with torch.no_grad():
        h = torch.zeros(N, NU)
        for i in range(BURN):
            h = gru(F[tt - K - BURN + 1 + i, bb], h)
    return h


def fresh(seed, chrono):
    torch.manual_seed(1000 + seed)
    g = nn.GRUCell(NI, NU)
    if chrono:
        with torch.no_grad():
            g.bias_ih[NU: 2 * NU] = torch.log(torch.empty(NU).uniform_(1.0, TMAX - 1.0)); g.bias_hh[NU: 2 * NU] = 0.0
    for p_ in g.parameters():
        p_.requires_grad_(False)
    return g


rows = {}
rows["학습된 GRU: 기억 → 기억"], zk_t = back(actor.gru, H[tt - K, bb])
rows["학습된 GRU: 지금의 plan 출력 → 기억"], _ = back(actor.gru, H[tt - K, bb], out=True)
for name, chrono in (("새 GRU, 기본 초기값: 기억 → 기억", False), (f"새 GRU, 붙드는 문 bias를 1~{TMAX} 결정에 퍼지게: 기억 → 기억", True)):
    acc, zs = [], []
    for seed in range(3):
        g = fresh(seed, chrono); r_, zk = back(g, burn(g)); acc.append(r_); zs.append(zk)
    rows[name] = np.mean(acc, 0); rows[name + "|z"] = np.concatenate(zs)
rows["학습된 GRU: 기억 → 기억|z"] = zk_t

print(f"{z['exam']} · {R.name} · 실행 {z['mode']} · 상자 {z['count']} — 표본 {N}개 (재시작이 {(K + BURN) * dt:.0f} s 안에 없는 것), 기억 값 {NU}개, "
      f"혼자 달리는 세션의 한 바퀴 중앙 {lap_s:.1f} s = {lap_s / dt:.0f} 결정")
print("(1) k 결정 전으로 보낸 신호가 몇 배로 남나 (표본 중앙; 1 = 그대로)")
print("  " + " | ".join(["무엇"] + [f"{k * dt:.1f} s" for k in KS]))
for name, r_ in rows.items():
    if name.endswith("|z"):
        continue
    md = np.median(r_, 1)
    print("  " + " | ".join([name] + [f"{md[k]:.1e}" for k in KS]) + (f"   (지금의 기억에 대한 출력의 민감도 {md[0]:.2f})" if "출력" in name else ""))
print("(2) 붙드는 문(1에 가까울수록 값을 그대로 둠)")
for name in rows:
    if not name.endswith("|z"):
        continue
    zk = rows[name]; per = np.median(zk.reshape(-1, NU), 0)          # each value's usual gate
    tau = 1.0 / np.clip(1.0 - per, 1e-6, None) * dt
    print(f"  {name[:-2].split(':')[0]}: 문 중앙 {np.median(zk):.2f} · 값 {NU}개 가운데 붙드는 시간이 1 s 넘는 것 {int((tau > 1.0).sum())}, 4 s 넘는 것 {int((tau > 4.0).sum())}, "
          f"한 바퀴({lap_s:.1f} s) 넘는 것 {int((tau > lap_s).sum())}")
md_t = np.median(rows["학습된 GRU: 기억 → 기억"], 1)
below = np.nonzero(md_t < 1e-3)[0]
print(f"(3) 학습된 GRU에서 신호가 1/1000 아래로 떨어지는 때: {below[0] * dt:.1f} s 전" if len(below) else "(3) 학습된 GRU에서 신호가 14 s 안에 1/1000 아래로 안 떨어짐")

if fig_out:
    BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#777777", "#333333"
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.4))
    x = np.arange(K + 1) * dt
    style = (("학습된 GRU: 기억 → 기억", BLUE, "-", "학습된 GRU"), ("학습된 GRU: 지금의 plan 출력 → 기억", BLUE, ":", "학습된 GRU (지금의 plan 출력에서 출발)"),
             ("새 GRU, 기본 초기값: 기억 → 기억", GRAY, "-", "새 GRU, 기본 초기값 (학습을 시작할 때)"),
             (f"새 GRU, 붙드는 문 bias를 1~{TMAX} 결정에 퍼지게: 기억 → 기억", ORANGE, "-", f"새 GRU, 붙드는 문 bias를 1~{TMAX * dt:.0f} s에 퍼지게 초기화"))
    for name, col, ls, lab in style:
        r_ = rows[name]; md = np.median(r_, 1); lo, hi = np.percentile(r_, [25, 75], axis=1)
        if ls == "-":
            ax.fill_between(x[1:], np.clip(lo[1:], 1e-30, None), np.clip(hi[1:], 1e-30, None), color=col, alpha=0.15, lw=0)
        ax.plot(x[1:], np.clip(md[1:], 1e-30, None), ls, color=col, lw=2.5, label=lab)
    ax.set_yscale("log"); ax.set_ylim(1e-12, 3.0)
    ax.axvline(lap_s, color=INK, lw=1.2, ls="--"); ax.text(lap_s + 0.1, 1.2, f"한 바퀴 전 ({lap_s:.1f} s)", fontsize=10, color=INK, va="top")
    ax.axhline(1e-3, color=INK, lw=0.8, ls=":"); ax.text(13.9, 1.3e-3, "1/1000", fontsize=9, color=INK, ha="right", va="bottom")
    ax.set_xlabel("몇 s 전의 기억으로 보내는가"); ax.set_ylabel("남는 신호의 크기 (1 = 그대로; 로그 눈금)")
    ax.set_title(f"학습 신호가 기억을 거슬러 얼마나 먼 과거까지 닿나 — {R.name} (선 = 표본 중앙, 띠 = 25~75 %)", fontsize=12)
    ax.legend(fontsize=10, frameon=False, loc="lower left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_out, dpi=80); print("그림", fig_out)
