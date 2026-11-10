"""기억(GRU) 들여다보기 (11-09, 사용자: "gru 네트워크같은거 시각화 가능?", "초반이라 추이 확인하는게 엄청 중요해보여서 시각화등등 분석 많이
해봐"). 학습 세계, 원래 트랙(변형 없음), 학습 조건(plan대로 정확히)에서 기억 policy로 몰며 -- 상자 있는 세계 절반, 그중 절반은 15 s에 상자를
치움(대회처럼), 상대 차 절반 -- 결정(0.1 s)마다 학습 차의 기억 상태(128), 몸통 특징(지금 보는 것), 첫 단계 속도 출력, 기억을 지웠을 때(새 에피소드
처럼 0에서 한 번)의 속도 출력, 위치를 기록. 그림:
 (1) 기억이 아는 것: 선형 판별기(로지스틱)가 '지금 상자가 있나'를 기억 상태로 / 지금 특징으로 맞히는 비율, 에피소드 시작부터 시간별
 (2) 치운 걸 아나: 상자를 치운 세계에서 '치웠나'를 기억 / 지금 특징으로, 치운 때부터 시간별
 (3) 기억 상태를 주성분 2개로: 빈 트랙 / 상자 그대로 / 상자 치움(치우기 전·후) 차들의 궤적
 (4) 상자 치운 차 하나의 기억 128개 값 열지도 (세로선 = 치운 때)
 (5) 기억이 몸통 특징을 얼마나 바꾸나: |기억이 더한 것| / |특징|, 상황별 시간 추이
 (6) 기억을 지웠을 때 첫 단계 속도 출력이 바뀌는 크기, 상황별
인자: checkpoint 출력png [초] [환경 수]"""
import os, sys, math
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn import state_rl as SR
from f1sim.learn.plan_policy import PlanObs2, load_actor2
from f1sim.learn.plan_reward import PlanReward
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
ck_path, out = sys.argv[1], sys.argv[2]
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
n_env = int(sys.argv[4]) if len(sys.argv) > 4 else 256
T_RM = 15.0
dev = torch.device("cuda")
env, lim, world = PR.build(PR.WORLD, n_env, 21, dev, overrides={"warps": 0, "episode_s": secs + 10.0}, fmt="v2")
lid = env.learner_ids; Bl = int(lid.numel())
ck = torch.load(ck_path, map_location=dev, weights_only=False)
actor = load_actor2(ck, dev)
if not actor.memory:
    raise SystemExit("이 checkpoint에는 기억이 없습니다")
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
obs.mem_h = torch.zeros(Bl, actor.memory, device=dev)
drv = PR.Driver(env, lim, pr, obs)
remover = SR.PropRemoval(env, 0.5, T_RM, T_RM, seed=3)
proc = env.procedural
others = env.sim.other_idx[lid] if env.M > 1 else None
race_of = (lid // env.M) if env.M > 1 else torch.arange(Bl, device=dev)
Hs, Fs, SP, SP0, CR, HP, RM, OP, RS, XY = [], [], [], [], [], [], [], [], [], []
for kd in range(int(secs / PS.DT)):
    f = actor.features(obs.scans, obs.state())
    fm_, h1 = actor.remember(f, obs.mem_h)
    mean, _, v_hat = actor.heads(fm_)
    f0, _ = actor.remember(f, torch.zeros_like(obs.mem_h))           # the same moment with the memory wiped
    mean0, _, _ = actor.heads(f0)
    contrib = (fm_ - f).norm(dim=1) / f.norm(dim=1).clamp_min(1e-6)
    p = PS.roll(torch.tanh(mean), v_hat.clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim)
    obs.mem_h = h1
    out_, _, _ = drv.decision(torch.tanh(mean), p)
    remover.step()
    has_p = (proc.p_sid[env.sim.eid[lid]] >= 0).any(-1)
    absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
    has_o = (~absent[others]).any(-1) if others is not None else torch.zeros(Bl, dtype=torch.bool, device=dev)
    Hs.append(h1.half().cpu()); Fs.append(f.half().cpu())
    SP.append(torch.tanh(mean[:, PS.N_STEP]).cpu()); SP0.append(torch.tanh(mean0[:, PS.N_STEP]).cpu()); CR.append(contrib.cpu())
    HP.append(has_p.cpu()); RM.append(remover.removed[race_of].cpu()); OP.append(has_o.cpu()); RS.append(out_["restarted"].cpu())
    XY.append(env.sim.state[lid, :2].cpu())
H = torch.stack(Hs, 0).float().numpy(); F = torch.stack(Fs, 0).float().numpy()          # (T, Bl, d)
SP, SP0, CR = (torch.stack(z, 0).numpy() for z in (SP, SP0, CR))
HP, RM, OP, RS = (torch.stack(z, 0).numpy() for z in (HP, RM, OP, RS))
XY = torch.stack(XY, 0).numpy()
T = H.shape[0]
t_s = (np.arange(T) + 1) * PS.DT
# the first episode of each car only (until its first restart)
first_end = np.array([int(np.argmax(RS[:, b])) if RS[:, b].any() else T for b in range(Bl)])
valid = np.arange(T)[:, None] < first_end[None, :]
box_world = HP[0].copy()                                    # boxes at the start (a removal in an empty world removes nothing)
rm_world = RM.any(0) & box_world
cond = np.where(~box_world, 0, np.where(rm_world, 2, 1))   # 0 empty, 1 boxes kept, 2 boxes removed
names = {0: "빈 트랙", 1: "상자 그대로", 2: f"상자 {T_RM:.0f} s에 치움"}
cols = {0: "tab:blue", 1: "saddlebrown", 2: "tab:orange"}


def probe(X, y, tr, te):
    """a linear classifier (logistic, standardized inputs) fit on rows `tr`, (its predictions on `te`)"""
    Xt = torch.tensor(X[tr], dtype=torch.float32, device=dev); yt = torch.tensor(y[tr], dtype=torch.float32, device=dev)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-4
    Xt = (Xt - mu) / sd
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True); b = torch.zeros((), device=dev, requires_grad=True)
    pos = float(yt.mean().clamp(0.01, 0.99))
    wt = torch.where(yt > 0.5, 0.5 / pos, 0.5 / (1 - pos))   # balanced
    opt = torch.optim.Adam([w, b], lr=0.05)
    with torch.enable_grad():
        for _ in range(300):
            z = Xt @ w + b
            loss = (wt * torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")).mean() + 1e-3 * w.pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    Xe = (torch.tensor(X[te], dtype=torch.float32, device=dev) - mu) / sd
    return ((Xe @ w + b) > 0).cpu().numpy()


# 11-09, the user: "그 두번째 시각화는 뭘 어떻게 해석해야하는지 모르겠네" -- four questions, each with how to read it, and a one-line
# verdict on top
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
cars = np.arange(Bl)
tr_c, te_c = cars[0::2], cars[1::2]
READ = dict(fontsize=9, color="0.25", ha="left", va="top", transform=None)


def note(ax, text):
    ax.text(0.01, -0.13, "읽는 법: " + text, transform=ax.transAxes, fontsize=9.5, color="0.2", ha="left", va="top")


# A: does the memory know more than the view now?
y_now = HP.astype(np.float32)
idx = np.argwhere(valid)
trm = np.isin(idx[:, 1], tr_c); tem = ~trm
Xh = H[idx[:, 0], idx[:, 1]]; Xf = F[idx[:, 0], idx[:, 1]]; yy = y_now[idx[:, 0], idx[:, 1]]
ph = probe(Xh, yy, trm, tem); pf = probe(Xf, yy, trm, tem)
tt = idx[tem, 0] * PS.DT; ye = yy[tem]
acc_h, acc_f = float(np.mean(ph == ye)), float(np.mean(pf == ye))
ax = axs[0, 0]
bins = np.arange(0, secs + 0.01, 2.0)
for pred, lab, c in ((ph, f"기억으로 맞힘 (평균 {acc_h * 100:.0f} %)", "tab:red"), (pf, f"지금 보이는 것으로 맞힘 (평균 {acc_f * 100:.0f} %)", "tab:gray")):
    acc = [np.mean(pred[(tt >= lo) & (tt < lo + 2)] == ye[(tt >= lo) & (tt < lo + 2)]) if np.any((tt >= lo) & (tt < lo + 2)) else np.nan
           for lo in bins[:-1]]
    ax.plot(bins[:-1] + 1, np.array(acc) * 100, "-o", color=c, ms=3, label=lab)
ax.set_ylim(40, 101); ax.axhline(50, color="0.7", lw=0.8)
ax.set_title("A. 기억이 '지금 보이는 것'보다 더 아나? -- 이 트랙에 상자가 있는지 맞히기", fontsize=11)
ax.set_xlabel("에피소드 시작부터 [s]"); ax.set_ylabel("맞힌 비율 [%]"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
note(ax, "빨강이 회색보다 위면 기억이 시야 밖(지나온 곳)의 상자를 기억하는 것. 둘이 겹치면 기억이 더 아는 게 없음")
# B: does it notice the boxes were taken away?
ax = axs[0, 1]
sel = valid & rm_world[None, :]
idx2 = np.argwhere(sel)
if len(idx2) > 50:
    y2 = (idx2[:, 0] * PS.DT >= T_RM).astype(np.float32)
    trm2 = np.isin(idx2[:, 1], tr_c); tem2 = ~trm2
    ph2 = probe(H[idx2[:, 0], idx2[:, 1]], y2, trm2, tem2); pf2 = probe(F[idx2[:, 0], idx2[:, 1]], y2, trm2, tem2)
    tt2 = idx2[tem2, 0] * PS.DT - T_RM; ye2 = y2[tem2]
    b2 = np.arange(-T_RM, secs - T_RM + 0.01, 2.0)
    for pred, lab, c in ((ph2, "기억으로", "tab:red"), (pf2, "지금 보이는 것으로", "tab:gray")):
        acc = [np.mean(pred[(tt2 >= lo) & (tt2 < lo + 2)] == ye2[(tt2 >= lo) & (tt2 < lo + 2)]) if np.any((tt2 >= lo) & (tt2 < lo + 2)) else np.nan
               for lo in b2[:-1]]
        ax.plot(b2[:-1] + 1, np.array(acc) * 100, "-o", color=c, ms=3, label=lab)
    ax.axvline(0, color="k", lw=1, ls="--"); ax.set_ylim(0, 101); ax.legend(fontsize=9)
ax.set_title("B. 상자를 치운 걸 알아차리나? -- '이미 치웠나' 맞히기 (상자 치운 차들)", fontsize=11)
ax.set_xlabel("치운 때부터 [s] (점선 = 치운 순간)"); ax.set_ylabel("맞힌 비율 [%]"); ax.grid(alpha=0.3)
note(ax, "점선 뒤에 빨강이 회색보다 빨리, 높이 오르면 기억이 '치워졌다'를 알아차리는 것")
# C: does the memory change the driving?
ax = axs[1, 0]


def ma_(y, n=10):
    """the 1 s running mean (10 decisions), centred, NaN-safe (11-10, the user: \"C 좀 smoothing해서 좀 그려봐\")"""
    v = np.where(np.isfinite(y), y, 0.0); w = np.isfinite(y).astype(float); k = np.ones(n)
    num, den = np.convolve(v, k, "same"), np.convolve(w, k, "same")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


means, c_series = {}, {}
for c_ in (0, 1, 2):
    m = valid & (cond[None, :] == c_)
    y6 = np.array([np.mean(np.abs(SP[t] - SP0[t])[m[t]]) if m[t].any() else np.nan for t in range(T)])
    means[c_] = float(np.nanmean(y6))
    c_series[c_] = y6
    ax.plot(t_s, y6, color=cols[c_], lw=0.6, alpha=0.3)
    ax.plot(t_s, ma_(y6), color=cols[c_], lw=2.2, label=f"{names[c_]} (평균 {means[c_]:.3f})")
try:                                                        # the C panel's series, to redraw it in a second (11-10)
    np.savez(os.path.splitext(sys.argv[2])[0] + "_C.npz", t=t_s, **{f"c{k}": v for k, v in c_series.items()})
except Exception:
    pass
ax.axvline(T_RM, color="k", lw=1, ls="--")
ax.set_title("C. 기억이 운전을 바꾸나? -- 기억을 지웠을 때 첫 단계 속도 출력(-1..1)이 바뀌는 크기 (굵은 선 = 1초 평균)", fontsize=11)
ax.set_xlabel("시간 [s] (점선 = 치운 순간)"); ax.set_ylabel("|바뀐 크기|"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
note(ax, "0이면 기억을 운전에 안 씀. 클수록 기억이 운전을 바꿈(0.1 = 최대 가속 대비 5 %). 색끼리 다르면 상황에 따라 다르게 씀")
# D: the memory's values for one car whose boxes were taken away
ax = axs[1, 1]
rcars = np.nonzero((cond == 2) & (first_end >= T))[0]
if len(rcars) == 0:
    rcars = np.nonzero(cond == 2)[0]
if len(rcars):
    b = rcars[0]
    e = first_end[b]
    im = ax.imshow(H[:e, b].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, extent=[0, e * PS.DT, H.shape[2], 0])
    ax.axvline(T_RM, color="k", lw=1.5, ls="--")
    plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("D. 기억 128칸의 값 (상자 치운 차 1대, 점선 = 치운 순간)", fontsize=11)
ax.set_xlabel("시간 [s]"); ax.set_ylabel("기억 칸 번호")
note(ax, "세로 줄무늬가 바퀴마다 되풀이되면 트랙 위 위치를 따라가는 것. 점선 뒤 무늬가 바뀌면 기억이 치운 걸 반영하는 것")
name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
cr_all = float(np.mean(CR[valid]))
dsp = float(np.mean(np.abs(SP - SP0)[valid]))
verdict = ("기억이 시야보다 더 앎" if acc_h - acc_f >= 0.03 else "기억이 시야보다 더 아는 게 거의 없음") + " · " + \
          ("기억이 운전을 바꿈" if dsp >= 0.03 else ("기억이 운전을 조금 바꿈" if dsp >= 0.01 else "기억이 운전을 거의 안 바꿈"))
fig.suptitle(f"{name} 기억(GRU {actor.memory}) -- 한 줄 요약: {verdict}\n"
             f"학습 세계 원래 트랙, 학습 조건, 학습 차 {Bl}대 x {secs:.0f} s (빈 트랙 {np.sum(cond == 0)}대, 상자 그대로 {np.sum(cond == 1)}대, "
             f"상자 {T_RM:.0f} s에 치움 {np.sum(cond == 2)}대)", fontsize=13)
plt.tight_layout(rect=(0, 0.02, 1, 0.93)); plt.subplots_adjust(hspace=0.42)
plt.savefig(out, dpi=70)
print(f"{name}: '상자 있나' 맞힘 기억 {acc_h * 100:.0f} % / 지금 특징 {acc_f * 100:.0f} % | 기억이 특징을 바꾸는 크기 평균 {cr_all * 100:.1f} % | "
      f"기억을 지우면 첫 단계 속도 출력 평균 {dsp:.3f} 바뀜 | 그림 {out}", flush=True)
