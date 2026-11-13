"""네트워크 구조 점검 (11-13 19:40, 사용자: "네트워크 구조 자체에서 분석 좀 하라고. 계속 설정 좀 바꾸고 깔짝대다가 안되네만 도대체 언제까지").
보상·설정이 아니라 policy 네트워크(LiDAR 인코더 -> 기억 GRU -> 층 2개 -> plan 20개 숫자)의 어느 자리에서 상자가 늦게 읽히는지를 잰다.
입력: checkpoint, 그 checkpoint의 시험 기록(exam_record.py, F1SIM_EXAM_SAVE_MEM=1 + F1SIM_EXAM_SAVE_SCAN=1).
고르는 결정: 상자 세션, 치우기 전, 1 m/s 넘게 달림, 상자가 차 앞 0.8~7 m · 옆 1.5 m 안, 지금 스캔에 그 상자에 맞은 빔이 있음.
결정마다 "같은 장면에서 상자만 지운 스캔"(상자 빔을 지도 벽까지 거리로)을 만들어 짝으로 견줌 (box_react.py와 같은 방법).
  1) 층마다 상자의 자취: (상자 있음 - 없음)의 크기 / 그 층이 0.1 s 동안 평소 바뀌는 크기. 층마다 선형 읽기로 있음/없음 가르기
     (세션 반으로 맞추고 나머지 반에서 채점). 견줌: 같은 인코더 구조를 "상자 있음/없음"만 직접 가르쳤을 때(구조가 할 수 있는 한계).
  2) plan을 움직이는 입력: 입력 묶음(LiDAR 인코딩 / IMU / 직전 명령 / 직전 plan 점 / 직전 plan 속도 / 기억)별 기울기 x 흔들림 폭.
     직전 plan 입력의 되먹임: 직전 plan 끝점을 옆으로 30 cm 민 입력 -> 새 plan의 같은 점이 따라가는 비율(1이면 직전 plan을 베낌).
     스캔을 그대로 두고 자기 plan을 직전 plan 입력으로 다시 넣기를 되풀이: 상자 때문에 비키는 거리가 몇 번째에 얼마나 커지나.
  3) (NET_AUDIT_CKS=이름=경로,...) checkpoint마다 죽은 단위·포화·특징의 유효 차원·GRU 유지 문, 첫 층이 입력 묶음에서 받는 크기.
상자 위치·모양·지도는 분석에서만 씀(policy 입력 아님).
인자: checkpoint 기록npz 출력앞머리"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, plt, auc
from policy_lib import limits_for, dev

ck_path, npz, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
RNG = np.random.default_rng(0)
torch.manual_seed(0)
PER_BIN = int(os.environ.get("NET_AUDIT_PER_BIN", "800"))
BINS = [(0.8, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 5.5), (5.5, 7.0)]
BL = [f"{lo:g}~{hi:g} m" for lo, hi in BINS]
HALF_W = 0.155

from f1sim.learn.plan_policy import load_actor2, N_SCANS, ScanEncoder
from f1sim import plan_steps as PS
ck = torch.load(ck_path, map_location=dev, weights_only=False)
lim = limits_for(ck)
actor = load_actor2(ck, dev, check_mount=False).eval()
assert actor.memory and actor.mem_front and actor.branches == 1
for p_ in actor.parameters():
    p_.requires_grad_(False)
R = Rec(npz); z = R.z
mm = np.load(npz[:-4] + "_mem.npz"); sc = np.load(npz[:-4] + "_scan.npz")
F = mm["feat"].astype(np.float32); H = mm["mem"].astype(np.float32)
SC = sc["scan"]; RMAX = float(sc["range_max"]); ANG = sc["angles"].astype(float)
T, B = F.shape[0], F.shape[1]
H0 = np.zeros_like(H); H0[1:] = H[:-1]; H0[1:][R.rs[:T - 1].astype(bool)] = 0.0
dt = R.dt
k_rm = min(int(round(float(z["rm_t"]) / dt)), T)
mount = float(z["lidar_mount_x"]) if "lidar_mount_x" in z.files and float(z["lidar_mount_x"]) > 0 else 0.297
st = R.st.astype(float)
name = f"{os.path.basename(os.path.dirname(ck_path))} u{int(ck.get('update', 0))}"
NM, NS = actor.norm.mean.clone(), torch.sqrt(actor.norm.var + 1e-8)
GROUPS = [("LiDAR 인코딩", 0, 256), ("IMU", 256, 262), ("직전 명령", 262, 264), ("직전 plan 점", 264, 282), ("직전 plan 속도", 282, 292)]
log = open(OUT + ".txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); log.write(s + "\n"); log.flush()


def stack3(t, b):
    k0 = R.k0[t, b]
    return np.stack([SC[max(t - i, k0), b] for i in range(N_SCANS)], 0)


def in_poly(pts, poly, pad=0.05):
    c = poly.mean(0); inside = np.ones(len(pts), bool)
    n = len(poly)
    for i in range(n):
        a, b_ = poly[i], poly[(i + 1) % n]
        e = b_ - a; nrm = np.array([e[1], -e[0]]) / (np.hypot(*e) + 1e-9)
        if np.dot(c - a, nrm) > 0:
            nrm = -nrm
        inside &= ((pts - a) @ nrm) <= pad
    return inside


def poly_dist(pts, poly):
    d = np.full(len(pts), np.inf); n = len(poly)
    for i in range(n):
        a, b_ = poly[i], poly[(i + 1) % n]; e = b_ - a
        u = np.clip(((pts - a) @ e) / (e @ e), 0.0, 1.0)
        d = np.minimum(d, np.hypot(*(pts - (a + u[:, None] * e)).T))
    return np.where(in_poly(pts, poly, 0.0), 0.0, d)


def nobox(t, b):
    """frames t, t-1, t-2 with the box beams set to the map's wall distance: (with, without, box-beam mask of frame t)"""
    k0 = R.k0[t, b]; tk = int(R.tid[0, b])
    w, wo, m0 = [], [], None
    for i in range(N_SCANS):
        tf = max(t - i, k0)
        r = SC[tf, b].astype(float) * RMAX
        x, y, yaw = st[tf, b, 0], st[tf, b, 1], st[tf, b, 2]
        sx, sy = x + mount * np.cos(yaw), y + mount * np.sin(yaw)
        a = yaw + ANG
        end = np.stack([sx + r * np.cos(a), sy + r * np.sin(a)], 1)
        m = np.zeros(len(r), bool)
        for poly in R.boxes(min(tf, k_rm - 1), b):
            m |= in_poly(end, poly) & (r < RMAX * 0.999)
        r2 = r.copy()
        if m.any():
            d = np.arange(0.05, RMAX, 0.03)
            idx = np.nonzero(m)[0]
            px = sx + d[None, :] * np.cos(a[idx])[:, None]; py = sy + d[None, :] * np.sin(a[idx])[:, None]
            wd = R.wall_dist(tk, np.stack([px.ravel(), py.ravel()], 1)).reshape(len(idx), len(d))
            hit = wd <= 0.0
            r2[idx] = np.where(hit.any(1), d[hit.argmax(1)], RMAX)
        w.append(r / RMAX); wo.append(np.clip(r2 / RMAX, 0.0, 1.0))
        if i == 0:
            m0 = m
    return np.stack(w, 0), np.stack(wo, 0), m0


def acts(S3, Fs, h0):
    """every layer's output for scans S3 (n, 3, beams), normalised state Fs (n, 36), memory before h0 (n, 128)"""
    a = {}
    x = S3; j = 0
    for layer in actor.enc.conv:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            j += 1; a[f"합성곱 {j}"] = x.flatten(1)
    e = actor.enc.fc(x); a["인코더 출력(256)"] = e
    f = torch.cat([e, Fs], 1)
    h1 = actor.gru(f, h0); a["기억(128)"] = h1
    t1 = actor.trunk[1](actor.trunk[0](torch.cat([f, h1], 1))); a["본체 층 1"] = t1
    t2 = actor.trunk[3](actor.trunk[2](t1)); a["본체 층 2"] = t2
    a["출력(20)"] = actor.pi(t2)
    return a


def full_plan(e, raw, h0):
    """the plan as driving makes it: scan encoding e, raw state `raw` (36), memory before h0 -> (plan dict, memory after)"""
    f = torch.cat([e, ((raw - NM) / NS).clamp(-actor.norm.clip, actor.norm.clip)], 1)
    h1 = actor.gru(f, h0)
    mean, _, v_hat = actor.heads(actor.trunk(torch.cat([f, h1], 1)))
    return PS.roll(torch.tanh(mean), v_hat.clamp(-PS.V_REV, PS.V_FWD), raw[:, 6], lim), h1


def state_from_plan(raw, p):
    """the state inputs a decision later if the car drove plan p for 0.1 s (`PlanObs2.set_plan2`); the IMU stays"""
    i1 = PS.SUB
    x1, y1, h1 = p["x"][:, i1], p["y"][:, i1], p["psi"][:, i1]
    idx = torch.arange(2, PS.N_STEP + 1) * PS.SUB
    dx = p["x"][:, idx] - x1[:, None]; dy = p["y"][:, idx] - y1[:, None]
    c, s_ = torch.cos(h1)[:, None], torch.sin(h1)[:, None]
    rel = torch.stack([dx * c + dy * s_, -dx * s_ + dy * c], -1)
    out = raw.clone()
    out[:, 6] = p["delta"][:, i1]; out[:, 7] = p["v"][:, i1]
    out[:, 8:26] = rel.reshape(raw.shape[0], -1); out[:, 26:36] = p["pv"]
    return out


def ridge_auc(X, y, g, binid, lam=1.0):
    """linear read-out of y (ridge to +-1, standardised, dual form), fit on half the sessions `g`, scored on the other half and
    the other way round; AUC per distance bin (mean of the two halves)"""
    X = np.asarray(X, np.float32); res = np.zeros((2, len(BINS)))
    for h in (0, 1):
        tr, te = (g % 2 == h), (g % 2 != h)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-4
        A = (X[tr] - mu) / sd; Bt = (X[te] - mu) / sd
        t = np.where(y[tr], 1.0, -1.0).astype(np.float32)
        if A.shape[1] <= A.shape[0]:
            w = np.linalg.solve(A.T @ A + lam * A.shape[0] * np.eye(A.shape[1], dtype=np.float32), A.T @ t)
            s = Bt @ w
        else:
            al = np.linalg.solve(A @ A.T + lam * A.shape[0] * np.eye(A.shape[0], dtype=np.float32), t)
            s = Bt @ (A.T @ al)
        for i in range(len(BINS)):
            m = binid[te] == i
            res[h, i] = auc(s[m], y[te][m]) if m.sum() > 20 else np.nan
    return res.mean(0)


# ------------------------------------------------------------------ the decisions
with torch.no_grad():
    tt = RNG.integers(5, T, 64); bb = RNG.integers(0, B, 64)
    S3 = torch.from_numpy(np.stack([stack3(t, b) for t, b in zip(tt, bb)]).astype(np.float32))
    err = float((actor.enc(S3) - torch.from_numpy(F[tt, bb, :256])).abs().max())
say(f"== {name} ({os.path.basename(npz)}): 결정 {T} x 차 {B} | 다시 쌓은 스캔의 인코더 출력과 기록된 특징의 차 최대 {err:.4f}")
box_w = (R.sid[0] >= 0).any(-1)
V = np.abs(st[:T, :, 3])
cand = [[] for _ in BINS]
for b in np.nonzero(box_w)[0]:
    for t in range(N_SCANS + 1, k_rm):
        if V[t, b] < 1.0 or R.k0[t, b] > t - N_SCANS - 1:
            continue
        x, y, yaw = st[t, b, :3]
        c, s_ = np.cos(yaw), np.sin(yaw)
        best = None
        for poly in R.boxes(t, b):
            cx, cy = poly.mean(0) - [x, y]
            ax_, ay_ = cx * c + cy * s_, -cx * s_ + cy * c
            if 0.8 < ax_ < 7.0 and abs(ay_) < 1.5 and (best is None or ax_ < best[0]):
                best = (ax_, ay_)
        if best is not None:
            cand[next(i for i, (lo, hi) in enumerate(BINS) if lo <= best[0] < hi or i == len(BINS) - 1)].append((t, b, best[0], best[1]))
rows = []
for i, cs in enumerate(cand):
    pick = RNG.choice(len(cs), min(len(cs), PER_BIN * 2), replace=False) if len(cs) else []
    n = 0
    for j in pick:
        t, b, ax_, ay_ = cs[j]
        w3, wo3, m0 = nobox(t, b)
        if m0.sum() < 2:
            continue                                                   # the box is not in the scan now
        rows.append((t, b, ax_, ay_, w3, wo3, int(m0.sum()))); n += 1
        if n >= PER_BIN:
            break
tb = np.array([(r[0], r[1]) for r in rows]); AX = np.array([r[2] for r in rows]); AY = np.array([r[3] for r in rows])
NBEAM = np.array([r[6] for r in rows])
BIN = np.array([next(i for i, (lo, hi) in enumerate(BINS) if lo <= a < hi or i == len(BINS) - 1) for a in AX])
n = len(rows)
W3 = torch.from_numpy(np.stack([r[4] for r in rows]).astype(np.float32)); WO3 = torch.from_numpy(np.stack([r[5] for r in rows]).astype(np.float32))
P3 = torch.from_numpy(np.stack([stack3(t - 1, b) for t, b in tb]).astype(np.float32))
Fs = torch.from_numpy(F[tb[:, 0], tb[:, 1], 256:]); h0 = torch.from_numpy(H0[tb[:, 0], tb[:, 1]])
Fsp = torch.from_numpy(F[tb[:, 0] - 1, tb[:, 1], 256:]); h0p = torch.from_numpy(H0[tb[:, 0] - 1, tb[:, 1]])
say(f"  상자가 지금 스캔에 보이는 결정 {n}개: " + " · ".join(f"{BL[i]} {int((BIN == i).sum())}개(상자에 맞은 빔 중앙 {int(np.median(NBEAM[BIN == i])) if (BIN == i).any() else 0}개)" for i in range(len(BINS))))
del rows

# ------------------------------------------------------------------ 1) the box's trace layer by layer
with torch.no_grad():
    Aw, Ao, Ap = acts(W3, Fs, h0), acts(WO3, Fs, h0), acts(P3, Fsp, h0p)
LAY = list(Aw.keys())
ratio = np.zeros((len(LAY), len(BINS))); rd = np.zeros((len(LAY), len(BINS)))
y2 = np.concatenate([np.ones(n, bool), np.zeros(n, bool)]); g2 = np.concatenate([tb[:, 1], tb[:, 1]]); b2 = np.concatenate([BIN, BIN])
say("  1) 층마다 상자의 자취 -- (가) 상자 있음과 없음의 차이가 그 층이 0.1 s 동안 평소 바뀌는 양의 몇 배인가(중앙) · (나) 선형 읽기로 있음/없음 가르기(0.5 = 못 가름, 1 = 완벽)")
for li, k in enumerate(LAY):
    dbox = (Aw[k] - Ao[k]).norm(dim=1).numpy(); dstep = (Aw[k] - Ap[k]).norm(dim=1).numpy()
    for i in range(len(BINS)):
        m = BIN == i
        ratio[li, i] = np.median(dbox[m] / (dstep[m] + 1e-9)) if m.any() else np.nan
    rd[li] = ridge_auc(np.concatenate([Aw[k].numpy(), Ao[k].numpy()], 0), y2, g2, b2)
    say(f"    {k:<12s} (가) " + " · ".join(f"{BL[i]} {ratio[li, i]:.2f}" for i in range(len(BINS))) + "   (나) " + " · ".join(f"{rd[li, i]:.2f}" for i in range(len(BINS))))
# the same encoder structure taught "box / no box" directly (what the structure can do with these scans)
Xd = torch.cat([W3, WO3], 0); yd = torch.from_numpy(y2.astype(np.float32))
direct = np.zeros((2, len(BINS)))
for hh in (0, 1):
    tr = torch.from_numpy(g2 % 2 == hh); te = ~tr
    net = nn.Sequential(ScanEncoder(W3.shape[2]), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    itr = torch.nonzero(tr)[:, 0]
    for ep in range(int(os.environ.get("NET_AUDIT_EPOCHS", "30"))):
        perm = itr[torch.randperm(len(itr))]
        for j in range(0, len(perm), 128):
            idx = perm[j: j + 128]
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xd[idx])[:, 0], yd[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = torch.cat([net(Xd[j: j + 512])[:, 0] for j in range(0, len(Xd), 512)]).numpy()
    for i in range(len(BINS)):
        m = (te.numpy()) & (b2 == i)
        direct[hh, i] = auc(s[m], y2[m]) if m.sum() > 20 else np.nan
direct = direct.mean(0)
say("    같은 인코더 구조를 '상자 있음/없음'만 직접 가르쳤을 때(배운 적 없는 세션에서 채점): " + " · ".join(f"{BL[i]} {direct[i]:.2f}" for i in range(len(BINS))))

# ------------------------------------------------------------------ 2) what moves the plan
RAW = Fs * NS + NM
clipped = float((Fs.abs() >= actor.norm.clip - 1e-3).float().mean())
with torch.no_grad():
    Ew, Eo = actor.enc(W3), actor.enc(WO3)
    p0, _ = full_plan(Ew, RAW, h0)
# check: the plan recomputed from the record's inputs against the recorded plan (world frame)
x0, y0, yaw0 = (torch.from_numpy(st[tb[:, 0], tb[:, 1], i]).float() for i in range(3))
c0, s0 = torch.cos(yaw0)[:, None], torch.sin(yaw0)[:, None]
wx = x0[:, None] + p0["pts"][..., 0] * c0 - p0["pts"][..., 1] * s0; wy = y0[:, None] + p0["pts"][..., 0] * s0 + p0["pts"][..., 1] * c0
perr = torch.hypot(wx - torch.from_numpy(R.pl[tb[:, 0], tb[:, 1], :, 0]), wy - torch.from_numpy(R.pl[tb[:, 0], tb[:, 1], :, 1])).max(1).values.numpy()
say(f"  2) 확인: 기록된 입력에서 다시 계산한 plan과 기록된 plan의 차 -- 중앙 {np.median(perr) * 100:.1f} cm · 90 % {np.quantile(perr, 0.9) * 100:.1f} cm (상태 입력이 잘린 비율 {100 * clipped:.2f} %)")
# 2a) gradient x spread per input group: the 1.0 s point's sideways place, the first 0.1 s steering
sd_f = torch.from_numpy(F.reshape(-1, F.shape[2])[::7].std(0)); sd_h = torch.from_numpy(H0.reshape(-1, H.shape[2])[::7].std(0))


def shares(e, fs, hh):
    fg = torch.cat([e, fs], 1).clone().requires_grad_(True); hg = hh.clone().requires_grad_(True)
    h1 = actor.gru(fg, hg)
    mean, _, v_hat = actor.heads(actor.trunk(torch.cat([fg, h1], 1)))
    p = PS.roll(torch.tanh(mean), v_hat.clamp(-PS.V_REV, PS.V_FWD), torch.zeros_like(v_hat), lim)
    out = {}
    for nm_, q in (("1.0 s 점 옆 위치", p["pts"][:, -1, 1]), ("0.5 s 점 옆 위치", p["pts"][:, 4, 1]), ("1.0 s 속도", p["pv"][:, -1])):
        gf, gh = torch.autograd.grad(q.sum(), [fg, hg], retain_graph=True)
        parts = [(gf[:, a:b_] * sd_f[a:b_]).norm(dim=1) for _, a, b_ in GROUPS] + [(gh * sd_h).norm(dim=1)]
        tot = sum(parts) + 1e-12
        out[nm_] = [float((q_ / tot).median()) for q_ in parts]
    return out


GN = [g_[0] for g_ in GROUPS] + ["기억"]
sh_box = shares(Ew, Fs, h0)
ti = RNG.integers(5, T, 3000); bi = RNG.integers(0, B, 3000); mv = V[ti, bi] > 1.0; ti, bi = ti[mv], bi[mv]
sh_all = shares(torch.from_numpy(F[ti, bi, :256]), torch.from_numpy(F[ti, bi, 256:]), torch.from_numpy(H0[ti, bi]))
say("     입력 묶음별 몫(기울기 x 그 입력이 평소 흔들리는 폭, 합 1) -- 상자가 보이는 결정 / 아무 결정")
for q in sh_box:
    say(f"       {q}: " + " · ".join(f"{GN[i]} {100 * sh_box[q][i]:.0f} %/{100 * sh_all[q][i]:.0f} %" for i in range(len(GN))))
# 2b) the gain of the last-plan input: its points pushed sideways (30 cm at the end, growing with the square of the step)
with torch.no_grad():
    gains = []
    for sgn in (1.0, -1.0):
        raw2 = RAW.clone()
        rel = raw2[:, 8:26].reshape(n, PS.N_STEP - 1, 2).clone()
        rel[:, :, 1] += sgn * 0.3 * (torch.arange(1, PS.N_STEP) / (PS.N_STEP - 1.0)) ** 2
        raw2[:, 8:26] = rel.reshape(n, -1)
        p2, _ = full_plan(Ew, raw2, h0)
        gains.append(sgn * (p2["pts"][:, 8, 1] - p0["pts"][:, 8, 1]) / 0.3)
    gain = (0.5 * (gains[0] + gains[1])).numpy()
say("     직전 plan 입력의 되먹임: 직전 plan 끝점을 옆으로 30 cm 민 입력 -> 새 plan의 같은 점(0.9 s)이 따라간 비율 중앙 "
    + " · ".join(f"{BL[i]} {np.median(gain[BIN == i]):.2f}" for i in range(len(BINS))) + f" | 전체 {np.median(gain):.2f} (25~75 %: {np.quantile(gain, 0.25):.2f}~{np.quantile(gain, 0.75):.2f})")
# 2c) the loop: the scan held, its own plan fed back as the last plan, again and again
KS = [0, 1, 2, 3, 5, 8]
side = -np.sign(AY)
# the nearest box in the car's frame, for the plan's clearance
clr = {}
polys = []
for i, (t, b) in enumerate(tb):
    x, y, yaw = st[t, b, :3]; c, s_ = np.cos(yaw), np.sin(yaw)
    ps_ = []
    for poly in R.boxes(t, b):
        dx, dy = poly[:, 0] - x, poly[:, 1] - y
        ps_.append(np.stack([dx * c + dy * s_, -dx * s_ + dy * c], 1))
    polys.append(ps_)


def clear(p):
    pts = p["pts"].numpy().astype(float)
    return np.array([min(float(poly_dist(pts[i], q).min()) for q in polys[i]) - HALF_W for i in range(n)])


away = {}; v1 = {}
with torch.no_grad():
    pw, hw_ = full_plan(Ew, RAW, h0); po, ho_ = full_plan(Eo, RAW, h0)
    rw, ro = RAW, RAW
    for k in range(max(KS) + 1):
        if k == 0:
            # the first 0.1 s (the part that drives): steering away from the box [deg] and speed taken off [m/s] because of the box
            d_first = np.degrees(side * (pw["delta"][:, PS.SUB] - po["delta"][:, PS.SUB]).numpy())
            v_first = (po["v"][:, PS.SUB] - pw["v"][:, PS.SUB]).numpy()
            d_room = np.degrees((pw["delta"][:, PS.SUB] - RAW[:, 6]).abs().numpy())
            clr_o = clear(po)                                          # the plan without the box: through the box = the box is on the car's way
        if k in KS:
            away[k] = side * (pw["pts"][:, -1, 1] - po["pts"][:, -1, 1]).numpy()
            v1[k] = (po["pv"][:, -1] - pw["pv"][:, -1]).numpy()
            clr[k] = clear(pw)
        rw, ro = state_from_plan(rw, pw), state_from_plan(ro, po)
        pw, hw_ = full_plan(Ew, rw, hw_); po, ho_ = full_plan(Eo, ro, ho_)
say("     스캔을 그대로 두고 자기 plan을 직전 plan 입력으로 다시 넣기를 k번: 상자 때문에 1.0 s 점이 상자 반대쪽으로 비킨 거리 중앙 [cm] / plan의 선이 상자를 지나가는 비율")
for i in range(len(BINS)):
    m = BIN == i
    if m.sum() < 20:
        continue
    say(f"       {BL[i]}: " + " · ".join(f"k={k} {100 * np.median(away[k][m]):+.0f} cm/{100 * np.mean(clr[k][m] < 0):.0f} %" for k in KS)
        + f" | 줄인 1.0 s 속도 k=0 {np.median(v1[0][m]):+.2f} -> k={KS[-1]} {np.median(v1[KS[-1]][m]):+.2f} m/s")
say("     그 가운데 상자가 가는 길 위에 있는 결정(상자를 지운 스캔으로 낸 plan의 선이 상자를 지나감)만: 상자를 보고 낸 plan도 여전히 상자를 지나가는 비율 · 비킨 거리 · 줄인 속도")
ONP = clr_o < 0
onp_through = np.full(len(BINS), np.nan); onp_n = np.zeros(len(BINS), int)
for i in range(len(BINS)):
    m = (BIN == i) & ONP; onp_n[i] = int(m.sum())
    if m.sum() < 15:
        say(f"       {BL[i]}: {int(m.sum())}개 -- 너무 적음"); continue
    onp_through[i] = 100 * np.mean(clr[0][m] < 0)
    say(f"       {BL[i]} ({int(m.sum())}개): 여전히 지나감 {onp_through[i]:.0f} % (되풀이 {KS[-1]}번 뒤 {100 * np.mean(clr[KS[-1]][m] < 0):.0f} %) · 1.0 s 점이 비킨 거리 중앙 {100 * np.median(away[0][m]):+.0f} cm · "
        f"상자에서 떨어진 여유 중앙 {100 * np.median(clr[0][m]):+.0f} cm · 줄인 1.0 s 속도 {np.median(v1[0][m]):+.2f} m/s")
say("     같은 결정에서 실제로 차를 움직이는 첫 0.1 s: 상자 때문에 상자 반대쪽으로 더 꺾은 조향 [도] (중앙 · 25~75 %) · 1도 넘게 꺾은 비율 · 줄인 속도")
for i in range(len(BINS)):
    m = (BIN == i) & ONP
    if m.sum() < 15:
        continue
    say(f"       {BL[i]}: {np.median(d_first[m]):+.1f}도 ({np.quantile(d_first[m], 0.25):+.1f}~{np.quantile(d_first[m], 0.75):+.1f}) · {100 * np.mean(d_first[m] > 1.0):.0f} % · {np.median(v_first[m]):+.2f} m/s")
np.savez(OUT + "_pairs.npz", tb=tb, AX=AX, AY=AY, BIN=BIN, ONP=ONP, away=away[0], clr=clr[0], clr_o=clr_o, d_first=d_first, v_first=v_first)

# ------------------------------------------------------------------ figure
C = ["#d9480f", "#f08c00", "#2b8a3e", "#1971c2", "#6741d9"]
fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
xs = np.arange(len(LAY))
for i in range(len(BINS)):
    ax[0].plot(xs, rd[:, i], "-o", color=C[i], label=f"상자까지 {BL[i]}")
    ax[0].plot([len(LAY) + 0.3], [direct[i]], "*", color=C[i], ms=14)
    ax[1].plot(xs, ratio[:, i], "-o", color=C[i], label=f"상자까지 {BL[i]}")
ax[0].set_xticks(list(xs) + [len(LAY) + 0.3]); ax[0].set_xticklabels(LAY + ["같은 인코더를\n직접 가르침(★)"], rotation=40, ha="right", fontsize=8)
ax[0].axhline(0.5, color="#868e96", lw=0.8); ax[0].set_ylim(0.4, 1.02); ax[0].legend(fontsize=8)
ax[0].set_title("층마다: 상자 있음/없음을 선형으로 가를 수 있나\n(0.5 = 못 가름, 1 = 완벽, 배운 적 없는 세션)", fontsize=10)
ax[1].set_xticks(xs); ax[1].set_xticklabels(LAY, rotation=40, ha="right", fontsize=8); ax[1].set_yscale("log"); ax[1].axhline(1.0, color="#868e96", lw=0.8)
ax[1].set_title("층마다: 상자 때문에 생긴 차이 / 그 층이 0.1 s 동안 평소 바뀌는 양", fontsize=10); ax[1].legend(fontsize=8)
for i in range(len(BINS)):
    m = BIN == i
    if m.sum() >= 20:
        ax[2].plot(KS, [100 * np.median(away[k][m]) for k in KS], "-o", color=C[i], label=f"상자까지 {BL[i]}")
ax[2].set_xlabel("자기 plan을 직전 plan 입력으로 다시 넣은 횟수(스캔은 그대로)"); ax[2].set_ylabel("[cm]"); ax[2].legend(fontsize=8); ax[2].axhline(0, color="#868e96", lw=0.8)
ax[2].set_title("상자 때문에 plan(1.0 s 점)이 상자 반대쪽으로 비킨 거리", fontsize=10)
w_ = 0.38
ax[3].bar(np.arange(len(GN)) - w_ / 2, [100 * v for v in sh_box["1.0 s 점 옆 위치"]], w_, color="#d9480f", label="상자가 보이는 결정")
ax[3].bar(np.arange(len(GN)) + w_ / 2, [100 * v for v in sh_all["1.0 s 점 옆 위치"]], w_, color="#1971c2", label="아무 결정")
ax[3].set_xticks(np.arange(len(GN))); ax[3].set_xticklabels(GN, rotation=30, ha="right", fontsize=9); ax[3].set_ylabel("[%]"); ax[3].legend(fontsize=8)
ax[3].set_title("plan의 옆 위치(1.0 s 점)를 움직이는 입력의 몫", fontsize=10)
fig.suptitle(f"{name}: 네트워크 구조 점검 (시험 기록 {os.path.basename(npz)})", fontsize=11)
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=100); plt.close(fig)
say("그림", OUT + ".png")

# ------------------------------------------------------------------ 3) health per checkpoint
cks = [s.split("=") for s in os.environ.get("NET_AUDIT_CKS", "").split(",") if s]
if cks:
    NB = 24
    bs = RNG.choice(B, NB, replace=False)
    S_all = torch.from_numpy(np.stack([[stack3(t, b) for b in bs] for t in range(T)]).astype(np.float32))      # (T, NB, 3, beams)
    RAW_all = torch.from_numpy(F[:, bs, 256:]) * NS + NM
    rs = torch.from_numpy(R.rs[:T, bs].astype(np.float32))

    def srank(X, keep=0.99):
        s = torch.linalg.svdvals(X - X.mean(0))
        return int((torch.cumsum(s, 0) / s.sum() < keep).sum()) + 1

    say("  3) checkpoint마다 (같은 스캔·상태 기록을 각 네트워크에 넣음; 차 24대 x 900 결정)")
    say("     이름 | 합성곱 채널 가운데 거의 안 켜지는 것(1~4층) | 인코더 출력 256 중 안 켜지는 것 · 유효 차원 | 본체 층1/층2 포화(|값| > 0.97) 비율 · 층2 유효 차원 | 기억: 유지 문 중앙 · 1 s 넘게/5 s 넘게 쥐는 값 수 | 본체 첫 층이 받는 크기 몫: " + "/".join(GN))
    for nm_, path in cks:
        c2 = torch.load(path, map_location=dev, weights_only=False)
        a2 = load_actor2(c2, dev, check_mount=False).eval()
        with torch.no_grad():
            x = S_all.reshape(-1, N_SCANS, S_all.shape[-1]); dorm = []
            for layer in a2.enc.conv:
                x = torch.cat([layer(x[j: j + 2048]) for j in range(0, len(x), 2048)])
                if isinstance(layer, nn.ReLU):
                    m_ = x.mean((0, 2)); dorm.append(int((m_ / (m_.mean() + 1e-12) <= 0.025).sum()))
            e = a2.enc.fc(x)
            fdead = int((e.mean(0) / (e.mean() + 1e-12) <= 0.025).sum())
            fs_ = ((RAW_all - a2.norm.mean) / torch.sqrt(a2.norm.var + 1e-8)).clamp(-a2.norm.clip, a2.norm.clip)
            f = torch.cat([e.reshape(T, NB, -1), fs_], 2)
            h = torch.zeros(NB, a2.memory); Hs = []; Z = []
            n_ = a2.memory
            for t in range(T):
                gi = f[t] @ a2.gru.weight_ih.T + a2.gru.bias_ih; gh = h @ a2.gru.weight_hh.T + a2.gru.bias_hh
                Z.append(torch.sigmoid(gi[:, n_: 2 * n_] + gh[:, n_: 2 * n_]))
                h = a2.gru(f[t], h); Hs.append(h)
                h = h * (1.0 - rs[t])[:, None]
            Hs = torch.stack(Hs); Zm = torch.stack(Z).reshape(-1, n_).mean(0)
            hold = 0.1 / (1.0 - Zm).clamp_min(1e-6)
            zin = torch.cat([f, Hs], 2).reshape(T * NB, -1)
            t1 = a2.trunk[1](a2.trunk[0](zin)); t2 = a2.trunk[3](a2.trunk[2](t1))
            W = a2.trunk[0].weight
            grp = GROUPS + [("기억", 292, 292 + n_)]
            drive = torch.stack([(zin[:, a:b_] - zin[:, a:b_].mean(0)) @ W[:, a:b_].T for _, a, b_ in grp]).pow(2).mean((1, 2)).sqrt()
            drive = drive / drive.sum()
        say(f"     {nm_}: {dorm[0]}/32 · {dorm[1]}/64 · {dorm[2]}/64 · {dorm[3]}/64 | {fdead}/256 · {srank(e)} | {100 * float((t1.abs() > 0.97).float().mean()):.0f} %/{100 * float((t2.abs() > 0.97).float().mean()):.0f} % · {srank(t2)} | "
            f"{float(Zm.median()):.2f} · {int((hold > 1.0).sum())}/{int((hold > 5.0).sum())} | " + "/".join(f"{100 * float(d_):.0f}" for d_ in drive))
log.close()
