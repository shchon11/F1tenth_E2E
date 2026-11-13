"""네트워크 구조 점검 2 (11-13): plan의 뒷부분(0.1~1.0 s)은 차가 실제로 갈 길을 얼마나 맞히나, 못 맞히면 어디가 모자란가.
지금 구조: 첫 0.1 s만 RL의 행동이고, 뒷부분 18개 숫자는 "다음 plan이 같은 순간에 대해 말하는 것"을 출력층 혼자(본체는 건드리지 않고) 배움.
견줌 (세션 반으로 가르치고 나머지 반에서 채점, 모두 차 기준 좌표의 10개 점):
  (가) policy가 낸 plan 그대로
  (나) 본체 마지막 층(256)에서 새로 맞춘 선형 읽기 -> 실제로 간 길   (본체 특징에 앞날이 들어 있나)
  (다) 본체가 받는 입력(LiDAR 인코딩 + 상태 + 기억, 420)에서 같은 크기의 층 2개를 새로 가르침 -> 실제로 간 길  (입력에는 들어 있나)
  (라) LiDAR 인코딩을 뺀 입력(상태 + 기억)만으로 (다)와 같이   (LiDAR 인코딩이 보태는 몫)
묶음: 아무 결정 / 상자가 가는 길 위에 보이는 결정(net_audit.py가 고른 것: 상자를 지운 스캔으로 낸 plan이 상자를 지나감), 상자까지 거리별.
인자: checkpoint 기록npz net_audit의_pairs.npz 출력앞머리"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, plt
from policy_lib import limits_for, dev

ck_path, npz, pairs, OUT = sys.argv[1:5]
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
RNG = np.random.default_rng(0); torch.manual_seed(0)
from f1sim.learn.plan_policy import load_actor2, N_SCANS, ScanEncoder
DV = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # the freshly taught nets only; the policy stays on `dev`
from f1sim import plan_steps as PS
ck = torch.load(ck_path, map_location=dev, weights_only=False)
lim = limits_for(ck)
actor = load_actor2(ck, dev, check_mount=False).eval()
for p_ in actor.parameters():
    p_.requires_grad_(False)
R = Rec(npz); z = R.z
mm = np.load(npz[:-4] + "_mem.npz")
F = mm["feat"].astype(np.float32); H = mm["mem"].astype(np.float32)
T, B = F.shape[:2]
SC = np.load(npz[:-4] + "_scan.npz")["scan"]


def stack3(tt, bb):
    """the three scans the policy read at decisions (tt, bb), newest first (float16)"""
    k0 = R.k0[tt, bb]
    return np.stack([SC[np.maximum(tt - i, k0), bb] for i in range(N_SCANS)], 1)


H0 = np.zeros_like(H); H0[1:] = H[:-1]; H0[1:][R.rs[:T - 1].astype(bool)] = 0.0
st = R.st.astype(float); V = np.abs(st[:T, :, 3])
k_rm = min(int(round(float(z["rm_t"]) / R.dt)), T)
NM, NS = actor.norm.mean.clone(), torch.sqrt(actor.norm.var + 1e-8)
N = PS.N_STEP
HALF_W = 0.155
name = f"{os.path.basename(os.path.dirname(ck_path))} u{int(ck.get('update', 0))}"
BINS = [(0.8, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 5.5), (5.5, 7.0)]
BL = [f"{lo:g}~{hi:g} m" for lo, hi in BINS]
log = open(OUT + ".txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); log.write(s + "\n"); log.flush()


def poly_dist(pts, poly):
    d = np.full(len(pts), np.inf); n = len(poly); inside = np.ones(len(pts), bool); c = poly.mean(0)
    for i in range(n):
        a, b_ = poly[i], poly[(i + 1) % n]; e = b_ - a
        u = np.clip(((pts - a) @ e) / (e @ e), 0.0, 1.0)
        d = np.minimum(d, np.hypot(*(pts - (a + u[:, None] * e)).T))
        nrm = np.array([e[1], -e[0]]) / (np.hypot(*e) + 1e-9)
        if np.dot(c - a, nrm) > 0:
            nrm = -nrm
        inside &= ((pts - a) @ nrm) <= 0.0
    return np.where(inside, 0.0, d)


def gather(tt, bb):
    """for decisions (tt, bb): trunk input (420), trunk last layer (256), the plan's 10 points and what the car drove (car frame)"""
    f = torch.from_numpy(F[tt, bb]); h0 = torch.from_numpy(H0[tt, bb])
    with torch.no_grad():
        h1 = actor.gru(f, h0)
        zin = torch.cat([f, h1], 1)
        t2 = actor.trunk(zin)
        mean, _, v_hat = actor.heads(t2)
        raw = f[:, 256:] * NS + NM
        p = PS.roll(torch.tanh(mean), v_hat.clamp(-PS.V_REV, PS.V_FWD), raw[:, 6], lim)
    x, y, yaw = st[tt, bb, 0], st[tt, bb, 1], st[tt, bb, 2]
    c, s_ = np.cos(yaw), np.sin(yaw)
    drv = np.zeros((len(tt), N, 2), np.float32)
    for k in range(N):
        dx, dy = st[tt + k + 1, bb, 0] - x, st[tt + k + 1, bb, 1] - y
        drv[:, k, 0] = dx * c + dy * s_; drv[:, k, 1] = -dx * s_ + dy * c
    return zin.numpy(), t2.numpy(), p["pts"].numpy().astype(np.float32), drv


# decisions whose next 1.0 s is one unbroken drive
ok = np.zeros((T, B), bool)
norst = ~R.rs[:T].astype(bool)
run = np.ones((T - N, B), bool)
for k in range(N):
    run &= norst[k: T - N + k]
ok[: T - N] = run & (V[: T - N] > 1.0)
ok[max(0, k_rm - N): k_rm + 1] = False                                 # the boxes are taken away inside the second
P = np.load(pairs)
tb = P["tb"]; keep = ok[tb[:, 0], tb[:, 1]]
on = P["ONP"] & keep
tt_all, bb_all = np.nonzero(ok)
pick = RNG.choice(len(tt_all), min(len(tt_all), 40000), replace=False)
tt, bb = tt_all[pick], bb_all[pick]
Zg, Tg, PLg, DRg = gather(tt, bb)
Zb, Tb, PLb, DRb = gather(tb[:, 0], tb[:, 1])
say(f"== {name}: 아무 결정 {len(tt)}개 · 상자가 가는 길 위에 보이는 결정 {int(on.sum())}개 (다음 1.0 s가 끊기지 않은 것)")


def fit_ridge(X, Y, lam=1e-2):
    mu, sd = X.mean(0), X.std(0) + 1e-4
    A = np.concatenate([(X - mu) / sd, np.ones((len(X), 1), np.float32)], 1)
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1], dtype=np.float32), A.T @ Y)
    return lambda Xn: np.concatenate([(Xn - mu) / sd, np.ones((len(Xn), 1), np.float32)], 1) @ W


def fit_mlp(X, Y, epochs=int(os.environ.get("NET_AUDIT_EPOCHS", "40"))):
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Xt = torch.from_numpy((X - mu) / sd); Yt = torch.from_numpy(Y)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for j in range(0, len(perm), 256):
            idx = perm[j: j + 256]
            loss = nn.functional.smooth_l1_loss(net(Xt[idx]), Yt[idx], beta=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()

    def f(Xn):
        with torch.no_grad():
            return net(torch.from_numpy((Xn - mu) / sd)).numpy()
    return f


def fit_scan(S, X, Y, w, epochs=int(os.environ.get("NET_AUDIT_EPOCHS", "40"))):
    """the same LiDAR encoder structure, fresh, + the state and memory -> two layers -> the driven path; rows weighted by w"""
    mu, sd = X.mean(0), X.std(0) + 1e-4
    St = torch.from_numpy(S); Xt = torch.from_numpy((X - mu) / sd); Yt = torch.from_numpy(Y); wt = torch.from_numpy(w.astype(np.float32))
    enc = ScanEncoder(S.shape[2]).to(DV)
    head = nn.Sequential(nn.Linear(256 + X.shape[1], 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, Y.shape[1])).to(DV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for j in range(0, len(perm), 256):
            idx = perm[j: j + 256]
            out = head(torch.cat([enc(St[idx].float().to(DV)), Xt[idx].to(DV)], 1))
            loss = (nn.functional.smooth_l1_loss(out, Yt[idx].to(DV), beta=0.05, reduction="none").mean(1) * wt[idx].to(DV)).sum() / wt[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    enc.eval(); head.eval()

    def f(Sn, Xn):
        with torch.no_grad():
            Xn_ = torch.from_numpy((Xn - mu) / sd)
            return torch.cat([head(torch.cat([enc(torch.from_numpy(Sn[j: j + 2048]).float().to(DV)), Xn_[j: j + 2048].to(DV)], 1)).cpu()
                              for j in range(0, len(Xn), 2048)]).numpy()
    return f


# fit on half the sessions (general decisions + the box decisions of those sessions), score on the other half
METH = ["(가) policy의 plan", "(나) 본체 마지막 층에서 선형 읽기", "(다) 본체 입력에서 새로 가르침", "(라) LiDAR 인코딩 없이 새로 가르침",
        "(마) 스캔에서 인코더까지 새로 가르침"]
Sg, Sb = stack3(tt, bb), stack3(tb[:, 0], tb[:, 1])
pred_g = {m: np.zeros_like(DRg) for m in METH}; pred_b = {m: np.zeros_like(DRb) for m in METH}
pred_g[METH[0]], pred_b[METH[0]] = PLg, PLb
for h in (0, 1):
    trg, teg = bb % 2 == h, bb % 2 != h
    trb, teb = (tb[:, 1] % 2 == h) & keep, tb[:, 1] % 2 != h
    Ytr = np.concatenate([DRg[trg], DRb[trb]]).reshape(-1, 2 * N)
    for m, Xg_, Xb_, fit in ((METH[1], Tg, Tb, fit_ridge), (METH[2], Zg, Zb, fit_mlp), (METH[3], Zg[:, 256:], Zb[:, 256:], fit_mlp)):
        f = fit(np.concatenate([Xg_[trg], Xb_[trb]]), Ytr)
        pred_g[m][teg] = f(Xg_[teg]).reshape(-1, N, 2); pred_b[m][teb] = f(Xb_[teb]).reshape(-1, N, 2)
    # (마): the box decisions weigh 5x (they are 1 in 20 of the rows)
    f = fit_scan(np.concatenate([Sg[trg], Sb[trb]]), np.concatenate([Zg[trg, 256:], Zb[trb, 256:]]), Ytr,
                 np.concatenate([np.ones(int(trg.sum())), 5.0 * np.ones(int(trb.sum()))]))
    pred_g[METH[4]][teg] = f(Sg[teg], Zg[teg, 256:]).reshape(-1, N, 2); pred_b[METH[4]][teb] = f(Sb[teb], Zb[teb, 256:]).reshape(-1, N, 2)

# the boxes of each box decision in the car's frame
polys = []
for t, b in tb:
    x, y, yaw = st[t, b, :3]; c, s_ = np.cos(yaw), np.sin(yaw)
    polys.append([np.stack([(q[:, 0] - x) * c + (q[:, 1] - y) * s_, -(q[:, 0] - x) * s_ + (q[:, 1] - y) * c], 1) for q in R.boxes(t, b)])


def through(pts):
    """does the line through the 10 points pass through a box (car half-width), per decision"""
    out = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        line = np.concatenate([np.zeros((1, 2)), pts[i].astype(float)], 0)
        dense = np.concatenate([line[j] + (line[j + 1] - line[j]) * np.linspace(0, 1, 5)[:, None] for j in range(N)], 0)
        out[i] = min(float(poly_dist(dense, q).min()) for q in polys[i]) < HALF_W
    return out


say("  실제로 간 길과의 차 [cm]: 옆 방향, 0.5 s 뒤 / 1.0 s 뒤, 중앙 (90 %)")
res = {}
for m in METH:
    e = np.abs(pred_g[m][..., 1] - DRg[..., 1]) * 100
    res[m] = [np.median(e[:, 4]), np.median(e[:, 9])]
    say(f"    아무 결정        {m}: {np.median(e[:, 4]):.0f} ({np.quantile(e[:, 4], 0.9):.0f}) / {np.median(e[:, 9]):.0f} ({np.quantile(e[:, 9], 0.9):.0f})")
say("  (가) policy의 plan, 시점별 옆 방향 차 중앙 [cm] (0.1 s ... 1.0 s): " + " ".join(f"{np.median(np.abs(PLg[:, k, 1] - DRg[:, k, 1])) * 100:.1f}" for k in range(N)))
say("  (가) 같은 것, 앞뒤 방향: " + " ".join(f"{np.median(np.abs(PLg[:, k, 0] - DRg[:, k, 0])) * 100:.1f}" for k in range(N)))
say("  (마) 시점별 옆 방향 차 중앙 [cm]: " + " ".join(f"{np.median(np.abs(pred_g[METH[4]][:, k, 1] - DRg[:, k, 1])) * 100:.1f}" for k in range(N)))
drv_thr = through(DRb)
say("  상자가 가는 길 위에 보이는 결정: 옆 방향 차 1.0 s 뒤 중앙 [cm] · 그 선이 상자를 지나가는 비율 (실제로 간 길이 지나간 = 닿은 비율)")
thr = {m: through(pred_b[m]) for m in METH}
tab = np.full((len(METH) + 1, len(BINS)), np.nan)
for i in range(len(BINS)):
    msk = on & (P["BIN"] == i)
    if msk.sum() < 15:
        continue
    say(f"    {BL[i]} ({int(msk.sum())}개, 실제로 닿음 {100 * drv_thr[msk].mean():.0f} %)")
    tab[-1, i] = 100 * drv_thr[msk].mean()
    for j, m in enumerate(METH):
        e = np.abs(pred_b[m][msk][:, 9, 1] - DRb[msk][:, 9, 1]) * 100
        tab[j, i] = 100 * thr[m][msk].mean()
        say(f"       {m}: {np.median(e):.0f} cm · {tab[j, i]:.0f} %")
np.savez(OUT + ".npz", tab=tab, res=np.array([res[m] for m in METH]))
log.close()
