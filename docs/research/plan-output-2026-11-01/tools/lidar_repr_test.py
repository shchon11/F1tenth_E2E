"""LiDAR 입력 표현 시험 (11-25, 사용자와 정한 지도 학습 시험 하나). 질문: policy가 움직이는 상대 차를 서 있는 상자처럼 다룸 -- 스캔마다
1D conv로 256개로 줄인 뒤에야 GRU가 시간을 봐서 0.1 s에 30 cm 움직인 게 사라지고, 극좌표에서는 우리 움직임이 모든 점을 밀어 남의
움직임과 엉킴 -> 우리 움직임을 지운 위에서 본 격자(BEV, 나이마다 한 채널)면 움직임이 바로 보여 상대 차의 앞날을 훨씬 잘 맞히나?

기록: exam_record.py race4 (F1SIM_EXAM_SAVE_SCAN=1, F1SIM_EXAM_PRED_SAVE=1, F1SIM_EXAM4_PACES, 시험과 다른 시드). 결정(0.1 s)마다
policy가 읽은 그대로의 스캔(1081 빔 / 10 m), 두 차의 참 자세, 우리 속도·회전 속도, '있음'(학습 타깃 규칙), 머리의 예측.

모형 (같은 데이터·같은 세션 나눔·같은 예산; 출력은 머리와 같은 형식 = 0.1..1.0 s 뒤 10점, 지금 우리 틀, + '있음' logit):
  polar_gru   : policy의 스캔 인코더 구조(plan_policy.ScanEncoder, 새 가중치; 3장 겹친 입력) -> 마지막 10결정 GRU (+ 속도·회전 속도)
  polar_stackN: 마지막 N장을 conv 채널로 (GRU 없음) -- '시간 전에 압축' 효과만 따로
  polar_wide10: polar_stack10의 채널 4배 (계산량을 BEV 쪽에 맞춘 대조: 이기는 게 표현 때문인지 크기 때문인지)
  bevN        : 마지막 N장을 점으로 바꿔 우리 참 자세로 지금 틀에 옮겨 격자(0.125 m, 앞 -9.5..10.5 m x 옆 ±10 m)에 나이마다 한 채널
  bev10_noise : 상대 자세에 주행 거리계 잡음(0.1 s마다 2 cm·0.5° 무작위 걸음)을 얹어 (학습·평가 둘 다)
모든 모형이 마지막 10결정의 속도·회전 속도를 따로 읽음. 좌우 거울 뒤집기 증강(모두 같게).
평가는 따로 둔 세션의 '있음' 결정만. 기준 줄: 라이브 머리, 참 위치 그대로(서 있다고 봄), 참 위치 + 참 속도 직진, 복사본 자기 plan(오라클).

사용: python lidar_repr_test.py opp MODEL OUT_DIR BASE1 [BASE2 ...]   (BASE = 기록 .npz 경로에서 .npz 뺀 것)
      python lidar_repr_test.py box MODEL OUT_DIR BASE1 [...]          (상자 기억 시험, 아래 BOX 부분)
      python lidar_repr_test.py table OUT_DIR                          (한국어 요약)
환경: REPR_EPOCHS(기본 20), REPR_BATCH(256), REPR_LR(1e-3), REPR_GPU_FRAC(0.3: 랩의 시험이 GPU에서 쫓겨나지 않게)."""
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANGE = 10.0                       # [m] LiDAR range_max (the scan is / this, clipped to 1)
PRED_N, PRED_SCALE, PRED_RANGE = 10, 5.0, 10.0
HIST = 12                          # decisions of clean history every sample needs (10 GRU steps x a 3-scan stack)
GX0, GY0, RES, GH, GW = -9.5, -10.0, 0.125, 160, 160     # the BEV grid: x ahead [GX0, GX0 + GH RES), y left
SIG_P, SIG_TH = 0.02, math.radians(0.5)                  # odometry noise per 0.1 s (random walk)
EPOCHS = int(os.environ.get("REPR_EPOCHS", "20"))
BATCH = int(os.environ.get("REPR_BATCH", "256"))
LR = float(os.environ.get("REPR_LR", "1e-3"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------- shared nets
class Enc1D(nn.Module):
    """plan_policy.ScanEncoder's layers (same kernels / strides / widths by default), `in_ch` scans as channels"""

    def __init__(self, in_ch: int, beams: int, out: int = 256, w=(32, 64, 64, 64)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, w[0], 7, stride=3, padding=3), nn.ReLU(),
            nn.Conv1d(w[0], w[1], 5, stride=3, padding=2), nn.ReLU(),
            nn.Conv1d(w[1], w[2], 5, stride=3, padding=2), nn.ReLU(),
            nn.Conv1d(w[2], w[3], 3, stride=2, padding=1), nn.ReLU())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_ch, beams)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n, out), nn.ReLU())

    def forward(self, x):
        return self.fc(self.conv(x))


class Enc2D(nn.Module):
    """a small strided 2-D CNN over a (C, 160, 160) grid -> 256"""

    def __init__(self, in_ch: int, out: int = 256, hw=(GH, GW)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(),      # 80
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),         # 40
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),        # 20
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),       # 10
            nn.Conv2d(128, 64, 1), nn.ReLU())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_ch, *hw)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n, out), nn.ReLU())

    def forward(self, x):
        return self.fc(self.conv(x))


def mlp_head(n_in: int, n_out: int, hidden: int = 256):
    """OppPredictor's MLP shape"""
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out))


def se2_compose(a, b):
    """poses (..., 3) x, y, th: a o b (b expressed in a's frame -> in a's parent)"""
    c, s = torch.cos(a[..., 2]), torch.sin(a[..., 2])
    return torch.stack([a[..., 0] + c * b[..., 0] - s * b[..., 1], a[..., 1] + s * b[..., 0] + c * b[..., 1], a[..., 2] + b[..., 2]], -1)


def se2_rel(ref, p):
    """pose p (..., 3) in the frame of pose ref (..., 3)"""
    c, s = torch.cos(ref[..., 2]), torch.sin(ref[..., 2])
    dx, dy = p[..., 0] - ref[..., 0], p[..., 1] - ref[..., 1]
    return torch.stack([c * dx + s * dy, -s * dx + c * dy, p[..., 2] - ref[..., 2]], -1)


def rel_chain(poses, noise: float = 0.0, gen=None):
    """poses (n, N, 3) world poses at ages 0..N-1 (age 0 = now) -> (n, N, 3) each age's pose in the frame of age 0,
    built by composing the 0.1 s increments, each with odometry noise x `noise` (SIG_P per axis, SIG_TH) when > 0"""
    n, N, _ = poses.shape
    out = [torch.zeros(n, 3, device=poses.device)]
    T = out[0]
    for j in range(1, N):
        inc = se2_rel(poses[:, j - 1], poses[:, j])                  # age j in the frame of age j-1
        if noise > 0:
            e = torch.randn(n, 3, device=poses.device, generator=gen)
            inc = se2_compose(inc, e * torch.tensor([SIG_P, SIG_P, SIG_TH], device=poses.device) * noise)
        T = se2_compose(T, inc)
        out.append(T)
    return torch.stack(out, 1)


class Geo:
    """the beams' directions and the mount, to turn a scan into points in the car's frame"""

    def __init__(self, angles: np.ndarray):
        self.ang = torch.as_tensor(angles, dtype=torch.float32, device=DEV)

    def points(self, sc, mount):
        """sc (n, N, beams) normalised ranges, mount (n, 3) -> x, y (n, N, beams) in the car frame, valid (a return)"""
        r = sc.float() * RANGE
        a = self.ang[None, None] + mount[:, None, None, 2]
        x = mount[:, None, None, 0] + r * torch.cos(a)
        y = mount[:, None, None, 1] + r * torch.sin(a)
        return x, y, sc < 0.999


def raster(x, y, ok, H=GH, W=GW, x0=GX0, y0=GY0, res=RES):
    """points (n, C, P) -> (n, C, H, W) occupancy 0/1"""
    n, C, P = x.shape
    ix = torch.floor((x - x0) / res).long(); iy = torch.floor((y - y0) / res).long()
    ok = ok & (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
    g = torch.zeros(n * C * H * W, device=x.device)
    base = (torch.arange(n * C, device=x.device).view(n, C, 1) * (H * W))
    g[(base + ix * W + iy)[ok]] = 1.0
    return g.view(n, C, H, W)


def auc(score: np.ndarray, lab: np.ndarray) -> float:
    lab = lab.astype(bool)
    if lab.all() or (~lab).all():
        return float("nan")
    o = np.argsort(score, kind="mergesort"); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    # ties: average ranks
    s_sorted = score[o]; i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def flops_of(model, *inp) -> float:
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        model(*inp)
    return float(fc.get_total_flops())


def latency_ms(fn, n: int = 200) -> float:
    with torch.no_grad():
        for _ in range(20):
            fn()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return 1000.0 * (time.time() - t0) / n


# ------------------------------------------------------------------------------------------------------- opponent probe
def load_opp(bases):
    """the records' arrays, sessions of all records side by side (time axis 0 for the per-decision ones)"""
    per_sess = ("mount", "pace", "run", "idx")
    L = {}
    angles = None
    for r, b in enumerate(bases):
        m = np.load(b + ".npz"); s = np.load(b + "_scan.npz"); p = np.load(b + "_pred.npz")
        T, B = s["scan"].shape[:2]
        d = dict(scan=s["scan"], ego=s["ego"], mount=s["mount"], st=m["st"][:T + 1, :, :3], op=m["op"][:T + 1], on=m["on"][:T + 1],
                 rs=m["rs"][:T], ors=m["ors"][:T], there=p["there"][:T], dist=p["dist"][:T], head=p["head"][:T],
                 head_p=p["head_p"][:T], orc=p["orc"][:T], pace=np.asarray(m["c_pace"], np.float32),
                 run=np.full(B, r, np.int16), idx=np.arange(B, dtype=np.int16))
        assert d["pace"].shape[0] == B and d["st"].shape[0] == T + 1, (d["pace"].shape, d["st"].shape, T, B)
        if angles is None:
            angles = s["angles"]
        assert np.allclose(angles, s["angles"]) and (not L or L["scan"][0].shape[0] == T), "records of different lengths"
        for k, v in d.items():
            L.setdefault(k, []).append(v)
        log(f"기록 {b}: 결정 {T} x 세션 {B}")
    return {k: np.concatenate(v, 0 if k in per_sess else 1) for k, v in L.items()}, angles


def opp_samples(D):
    """the sample decisions (t, b) with HIST clean decisions behind them, targets (10, 2) in our frame at t + mask"""
    scan = D["scan"]; T, B = scan.shape[:2]
    st = D["st"].astype(np.float32); op = D["op"].astype(np.float32)
    jump = np.linalg.norm(op[1:T + 1, :, :2] - op[:T, :, :2], axis=-1) > 2.0
    brk = D["rs"].astype(bool) | D["ors"].astype(bool) | jump                    # (T, B): a break between t and t+1
    cb = np.concatenate([np.zeros((1, B), np.int64), np.cumsum(brk, 0)], 0)      # cb[t] = breaks in [0, t-1]
    t = np.arange(HIST, T)
    clean = (cb[t] - cb[t - HIST]) == 0                                           # no break in [t-HIST, t-1]
    tt, bb = np.nonzero(clean); tt = t[tt]
    tgt = np.zeros((len(tt), PRED_N, 2), np.float32); msk = np.zeros((len(tt), PRED_N), bool)
    c, s = np.cos(st[tt, bb, 2]), np.sin(st[tt, bb, 2])
    for k in range(1, PRED_N + 1):
        tk = np.minimum(tt + k, T)
        d = op[tk, bb, :2] - st[tt, bb, :2]
        rel = np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], -1)
        ok = (tt + k <= T) & ((cb[tk, bb] - cb[tt, bb]) == 0) & D["on"][tk, bb].astype(bool) & (np.linalg.norm(rel, axis=-1) <= PRED_RANGE)
        tgt[:, k - 1] = rel; msk[:, k - 1] = ok
    # references, same frame: the other car where it is now (standing), now + its true speed straight on, the copy's plan
    d0 = op[tt, bb, :2] - st[tt, bb, :2]
    now = np.stack([d0[:, 0] * c + d0[:, 1] * s, -d0[:, 0] * s + d0[:, 1] * c], -1)
    hy = op[tt, bb, 2] - st[tt, bb, 2]; v = op[tt, bb, 3]
    kk = np.arange(1, PRED_N + 1, dtype=np.float32) * 0.1
    cv = now[:, None, :] + v[:, None, None] * kk[None, :, None] * np.stack([np.cos(hy), np.sin(hy)], -1)[:, None, :]
    ref = {"head": D["head"][tt, bb], "still": np.repeat(now[:, None, :], PRED_N, 1), "cv": cv, "oracle": D["orc"][tt, bb]}
    meta = dict(t=tt, b=bb, there=D["there"][tt, bb].astype(bool), dist=D["dist"][tt, bb], pace=D["pace"][bb],
                head_p=D["head_p"][tt, bb], run=D["run"][bb], idx=D["idx"][bb])
    return tgt, msk, ref, meta


class OppNet(nn.Module):
    def __init__(self, kind: str, beams: int, geo: Geo):
        super().__init__()
        self.kind, self.geo = kind, geo
        ego_d = 2 * 10
        if kind == "polar_gru":
            self.enc = Enc1D(3, beams); self.gru = nn.GRU(256 + 2, 256, batch_first=True)
            self.head = mlp_head(256 + 256 + ego_d, 2 * PRED_N + 1)
        elif kind.startswith("polar_stack") or kind.startswith("polar_wide"):
            self.N = int(kind.replace("polar_stack", "").replace("polar_wide", ""))
            w = (128, 256, 256, 256) if kind.startswith("polar_wide") else (32, 64, 64, 64)
            self.enc = Enc1D(self.N, beams, w=w); self.head = mlp_head(256 + ego_d, 2 * PRED_N + 1)
        elif kind.startswith("bev"):
            self.N = int(kind[3:].split("_")[0]); self.noise = 1.0 if kind.endswith("_noise") else 0.0
            self.enc = Enc2D(self.N); self.head = mlp_head(256 + ego_d, 2 * PRED_N + 1)
        else:
            raise ValueError(kind)

    def forward(self, x, ego):
        """x: polar_gru (n, 10, 3, beams); polar stack (n, N, beams); bev (n, N, H, W). ego (n, 10, 2) newest first"""
        e = ego.reshape(ego.shape[0], -1)
        if self.kind == "polar_gru":
            n = x.shape[0]
            f = self.enc(x.reshape(n * 10, 3, -1).float()).view(n, 10, 256)
            h, _ = self.gru(torch.cat([f, ego.flip(1)], -1))              # oldest first
            return self.head(torch.cat([h[:, -1], f[:, -1], e], 1))
        return self.head(torch.cat([self.enc(x.float()), e], 1))


class OppData:
    """everything on the GPU; `batch(rows)` builds a model's inputs"""

    def __init__(self, D, angles, tgt, msk, meta):
        self.scan = torch.as_tensor(D["scan"], device=DEV)                           # (T, B, beams) f16
        self.st = torch.as_tensor(D["st"], dtype=torch.float32, device=DEV)         # (T+1, B, 3)
        ego = D["ego"][..., [0, 2]].astype(np.float32) / np.array([5.0, 2.0], np.float32)
        self.ego = torch.as_tensor(ego, device=DEV)                                  # (T, B, 2)
        self.mount = torch.as_tensor(D["mount"], dtype=torch.float32, device=DEV)
        self.t = torch.as_tensor(meta["t"], device=DEV).long(); self.b = torch.as_tensor(meta["b"], device=DEV).long()
        self.tgt = torch.as_tensor(tgt, device=DEV); self.msk = torch.as_tensor(msk, device=DEV)
        self.there = torch.as_tensor(meta["there"], device=DEV)
        self.geo = Geo(angles)

    def batch(self, net: OppNet, rows, mirror=None, gen=None):
        t, b = self.t[rows], self.b[rows]
        n = rows.numel()
        if mirror is None:
            mirror = torch.zeros(n, dtype=torch.bool, device=DEV)
        ages = torch.arange(10, device=DEV)
        ego = self.ego[(t[:, None] - ages[None]), b[:, None]]                     # (n, 10, 2) newest first
        ego = torch.where(mirror[:, None, None], ego * torch.tensor([1.0, -1.0], device=DEV), ego)
        k = net.kind
        if k == "polar_gru":
            # step s (oldest first) = the decision 9 - s back; its 3-scan stack as the policy reads it (newest channel first)
            a = torch.arange(9, -1, -1, device=DEV)[:, None] + torch.arange(3, device=DEV)[None]       # (10, 3)
            x = self.scan[t[:, None, None] - a[None], b[:, None, None]]                               # (n, 10, 3, beams)
            x = torch.where(mirror[:, None, None, None], x.flip(-1), x)
        elif k.startswith("polar"):
            x = self.scan[(t[:, None] - torch.arange(net.N, device=DEV)[None]), b[:, None]]       # (n, N, beams)
            x = torch.where(mirror[:, None, None], x.flip(-1), x)
        else:
            N = net.N
            sc = self.scan[(t[:, None] - torch.arange(N, device=DEV)[None]), b[:, None]]          # (n, N, beams)
            poses = self.st[(t[:, None] - torch.arange(N, device=DEV)[None]), b[:, None]]         # (n, N, 3) world
            rel = rel_chain(poses, net.noise, gen)                                               # (n, N, 3) in frame now
            px, py, ok = self.geo.points(sc, self.mount[b])
            c, s = torch.cos(rel[..., 2])[..., None], torch.sin(rel[..., 2])[..., None]
            X = rel[..., 0:1] + c * px - s * py; Y = rel[..., 1:2] + s * px + c * py
            Y = torch.where(mirror[:, None, None], -Y, Y)
            x = raster(X, Y, ok)
        tg = self.tgt[rows]
        tg = torch.where(mirror[:, None, None], tg * torch.tensor([1.0, -1.0], device=DEV), tg)
        return x, ego, tg, self.msk[rows], self.there[rows]


def opp_loss(out, tg, msk, there):
    pts = out[:, :2 * PRED_N].float().view(-1, PRED_N, 2)
    m = (msk & there[:, None]).float()
    hub = F.huber_loss(pts, tg / PRED_SCALE, delta=0.2, reduction="none").sum(-1)
    lp = (hub * m).sum() / m.sum().clamp_min(1.0)
    lv = F.binary_cross_entropy_with_logits(out[:, -1].float(), there.float())
    return lp + 0.2 * lv, lp, lv


def opp_predict(net, data, rows, bs=256, gen=None):
    net.eval(); P, L = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, rows.numel(), bs):
            r = rows[i:i + bs]
            x, ego, tg, msk, th = data.batch(net, r, gen=gen)
            o = net(x, ego).float()
            P.append((o[:, :2 * PRED_N].view(-1, PRED_N, 2) * PRED_SCALE).cpu()); L.append(o[:, -1].cpu())
    net.train()
    return torch.cat(P).numpy(), torch.cat(L).numpy()


def err_at(pred, tgt, msk, k):
    e = np.linalg.norm(pred[:, k] - tgt[:, k], axis=-1)
    return np.where(msk[:, k], e, np.nan)


def run_opp(kind, out_dir, bases):
    torch.manual_seed(0); np.random.seed(0)
    D, angles = load_opp(bases)
    tgt, msk, ref, meta = opp_samples(D)
    run, idx = meta["run"].astype(int), meta["idx"].astype(int)
    rep = idx // 16                                                          # the race's replicate within its condition (0..7)
    val = (rep == 7) | ((rep == 3) & (run == 1))
    if os.environ.get("REPR_VAL_MOD"):                                       # smoke tests on a few sessions
        val = (idx % int(os.environ["REPR_VAL_MOD"])) == 0
    log(f"표본 {len(val)}개 (따로 둔 세션 {val.mean() * 100:.1f} %), '있음' {meta['there'].mean() * 100:.1f} %")
    data = OppData(D, angles, tgt, msk, meta)
    del D
    beams = data.scan.shape[-1]
    net = OppNet(kind, beams, data.geo).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    tr = torch.as_tensor(np.nonzero(~val)[0], device=DEV); va = torch.as_tensor(np.nonzero(val)[0], device=DEV)
    g_tr = torch.Generator(device=DEV); g_tr.manual_seed(1)
    sub_tr = tr[torch.randperm(tr.numel(), generator=g_tr, device=DEV)[:20000]]
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps_ep = tr.numel() // BATCH; total = EPOCHS * steps_ep
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.05, final_div_factor=100.0)
    log(f"[{kind}] 매개변수 {n_par:,}, 학습 표본 {tr.numel()}, 검증 {va.numel()}, {EPOCHS} epoch x {steps_ep} 걸음")
    curves = []
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = tr[torch.randperm(tr.numel(), device=DEV)]
        tl = []
        for i in range(steps_ep):
            r = perm[i * BATCH:(i + 1) * BATCH]
            mir = torch.rand(r.numel(), device=DEV) < 0.5
            x, ego, tg, m, th = data.batch(net, r, mir)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x, ego)
            loss, lp, lv = opp_loss(out, tg, m, th)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
            tl.append(float(loss))
        row = {"epoch": ep + 1, "train_loss": float(np.mean(tl)), "s": time.time() - t0}
        for nm, rows, msk_ in (("val", va, val), ("trsub", sub_tr, None)):
            g = torch.Generator(device=DEV); g.manual_seed(7)
            P, Lg = opp_predict(net, data, rows, gen=g)
            ri = rows.cpu().numpy()
            th = meta["there"][ri]
            for k, kn in ((4, "05"), (9, "10")):
                e = err_at(P, tgt[ri], msk[ri], k)[th]
                row[f"{nm}_med{kn}"] = float(np.nanmedian(e))
            row[f"{nm}_auc"] = auc(Lg, th)
        curves.append(row)
        log(f"[{kind}] epoch {ep + 1}: 학습 손실 {row['train_loss']:.4f} | 검증 중앙 오차 0.5 s {row['val_med05']:.3f} m, 1.0 s {row['val_med10']:.3f} m, "
            f"AUC {row['val_auc']:.3f} | 학습 일부 1.0 s {row['trsub_med10']:.3f} m | {row['s']:.0f} s")
    # final predictions on the held-out sessions (noise: fixed seed)
    g = torch.Generator(device=DEV); g.manual_seed(11)
    P, Lg = opp_predict(net, data, va, gen=g)
    res = dict(pred=P, logit=Lg)
    if kind == "bev10":
        # the noise-free model read with noisy odometry (robustness without training on it)
        net.noise = 1.0; g.manual_seed(11)
        res["pred_noisy_in"], _ = opp_predict(net, data, va, gen=g); net.noise = 0.0
    # cost: params, FLOPs, ms at batch 1 (incl. building the input from the scans for the BEV)
    r1 = va[:1]
    x1, e1, _, _, _ = data.batch(net, r1)
    net.eval()
    fl = flops_of(net, x1, e1)
    if kind == "polar_gru" and flops_of(net.gru, torch.zeros(1, 10, 258, device=DEV)) == 0:
        fl += 10 * 2 * 3 * (256 + 2 + 256) * 256        # the GRU's gates (FlopCounterMode does not count cudnn's GRU)
    fl_stream = fl
    if kind == "polar_gru":
        # run as the policy runs it: one encoder pass and one GRU step a decision (the older steps' features kept)
        fl_stream = (flops_of(net.enc, x1[:, -1].float()) + 2 * 3 * (256 + 2 + 256) * 256
                     + flops_of(net.head, torch.zeros(1, 256 + 256 + 20, device=DEV)))
    ms_net = latency_ms(lambda: net(x1, e1))
    ms_all = latency_ms(lambda: net(*data.batch(net, r1)[:2]))
    np.savez_compressed(os.path.join(out_dir, f"opp_{kind}.npz"), rows=va.cpu().numpy(), tgt=tgt[val], msk=msk[val],
                        **res, **{f"ref_{k}": v[val] for k, v in ref.items()}, **{f"meta_{k}": v[val] for k, v in meta.items()})
    json.dump(dict(kind=kind, params=n_par, flops=fl, flops_stream=fl_stream, ms_net=ms_net, ms_with_input=ms_all, curves=curves, epochs=EPOCHS,
                   batch=BATCH, lr=LR, n_train=int(tr.numel()), n_val=int(va.numel()),
                   peak_gpu_mb=torch.cuda.max_memory_allocated() / 2 ** 20),
              open(os.path.join(out_dir, f"opp_{kind}.json"), "w"), indent=1)
    log(f"[{kind}] 끝: FLOPs {fl / 1e6:.1f} M, batch 1 {ms_net:.2f} ms (입력 만들기 포함 {ms_all:.2f} ms), GPU 최고 {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB")


# ------------------------------------------------------------------------------------------------------------ box probe
# (11-25, the user's addition) do the inputs keep STATIC boxes seen earlier (a lap ago) that are hidden now? Solo sessions with
# boxes (exam_record.py line3, F1SIM_EXAM_ALL_BOXES=1, F1SIM_EXAM_SAVE_SCAN=1, F1SIM_EXAM_SAVE_BOXVIS=1, 60 s, no removal).
# Target each decision: the boxes within BOX_AHEAD m ahead ALONG THE TRACK (the reward's walls-only distance field -- only to
# define the label, never an input), not in view now (plan_rl.boxes_in_view), seen at least once since the session (re)started.
# Output: a heatmap of those boxes' centres in our car's frame (OX0.., 0.25 m cells), CenterNet focal loss.
# Models: box_gru   = the policy's encoder (3-scan stack) + speed / yaw rate -> a GRU over the WHOLE session (TBPTT chunks)
#         box_mem   = an ego-centric spatial memory: every return written into a grid carried by dead-reckoned ego motion
#                     (kept in the dead-reckoned start frame and read out in the car's frame each decision = the same as a grid
#                     shifted / turned every step, without re-sampling it 600 times), channels: ever hit / how recently / the
#                     scan now; read by a small U-Net. Odometry noise x0 (true motion) or x1 (2 cm, 0.5 deg per 0.1 s random walk)
#         box_now   = the same U-Net, the scan now only (no memory): what the map shape alone gives away (the floor)
# No global map, no scan matching, no lap position, no line: only the ego motion integrated.
BOX_AHEAD = 8.0
OX0, OY0, ORES, OH, OW = -8.0, -8.0, 0.25, 64, 64
IX0, IY0, IRES, IH, IW = -8.0, -8.0, 0.125, 128, 128
MEM_HALF, MEM_RES = 32.0, 0.125
MEM_N = int(2 * MEM_HALF / MEM_RES)
BOX_EPOCHS = int(os.environ.get("REPR_BOX_EPOCHS", "12"))
GROUP_MEM = int(os.environ.get("REPR_BOX_GROUP", "32"))   # sessions streamed together (U-Net models)
GROUP_GRU = 8                                            # box_gru: more, smaller truncated-BPTT steps for the same data
CHUNK = 80
AGE_EDGES = (2.0, 5.0, 10.0)


def load_box(bases):
    per_sess = ("mount", "run", "idx")
    L = {}
    angles = None
    for r, b in enumerate(bases):
        m = np.load(b + ".npz"); s = np.load(b + "_scan.npz"); v = np.load(b + "_boxvis.npz")
        T, B = s["scan"].shape[:2]
        d = dict(scan=s["scan"], ego=s["ego"], mount=s["mount"], st=m["st"][:T + 1, :, :3].astype(np.float32), rs=m["rs"][:T],
                 pose=m["pose"][:T, :, :, :2].astype(np.float32), live=v["live"][:T], view=v["view"][:T],
                 ds=v["ds"][:T].astype(np.float32), run=np.full(B, r, np.int16), idx=np.arange(B, dtype=np.int16))
        if angles is None:
            angles = s["angles"]
        assert not L or L["scan"][0].shape[0] == T
        for k, x in d.items():
            L.setdefault(k, []).append(x)
        log(f"기록 {b}: 결정 {T} x 세션 {B}, 상자 칸 {d['live'].shape[-1]}")
    return {k: np.concatenate(x, 0 if k in per_sess else 1) for k, x in L.items()}, angles


def box_labels(D):
    """targets (T, B, C) bool, their age [s] since last in view, 'neutral' boxes (live, near, not targets), xy in our frame"""
    view, live, ds, rs = D["view"].astype(bool), D["live"].astype(bool), D["ds"], D["rs"].astype(bool)
    T, B, C = view.shape
    last = np.full((T, B, C), -1, np.int32)
    cur = np.full((B, C), -1, np.int32)
    for t in range(T):
        if t > 0:
            cur = np.where(view[t - 1], t - 1, cur)
            cur[rs[t - 1]] = -1                                   # a restart: the memory starts over
        last[t] = cur
    tgt = live & ~view & (last >= 0) & (ds > 0.0) & (ds <= BOX_AHEAD)
    age = np.where(last >= 0, (np.arange(T)[:, None, None] - last) * 0.1, np.inf).astype(np.float32)
    st = D["st"][:T]
    d = D["pose"] - st[:, :, None, :2]
    c, s = np.cos(st[..., 2])[..., None], np.sin(st[..., 2])[..., None]
    rel = np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], -1).astype(np.float32)
    inmap = (rel[..., 0] >= OX0) & (rel[..., 0] < OX0 + OH * ORES) & (rel[..., 1] >= OY0) & (rel[..., 1] < OY0 + OW * ORES)
    neutral = live & ~tgt & inmap
    return tgt, age, neutral, rel


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU())

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.c(x if skip is None else torch.cat([x, skip], 1))


class VecDecoder(nn.Module):
    """a feature vector -> (OH, OW) heatmap logits"""

    def __init__(self, n_in):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(n_in, 64 * 8 * 8), nn.ReLU())
        self.u1, self.u2, self.u3 = Up(64, 64), Up(64, 32), Up(32, 16)
        self.out = nn.Conv2d(16, 1, 1)
        nn.init.constant_(self.out.bias, -4.0)

    def forward(self, f):
        x = self.fc(f).view(-1, 64, 8, 8)
        return self.out(self.u3(self.u2(self.u1(x))))[:, 0]


class UNet(nn.Module):
    """(n, 3, IH, IW) at IRES -> (n, OH, OW) logits at ORES (= IRES x 2)"""

    def __init__(self, cin=3):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(cin, 16, 3, padding=1), nn.ReLU())                 # 128
        self.c2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())        # 64
        self.c3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())        # 32
        self.c4 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),        # 16
                                nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.ReLU())
        self.u1, self.u2 = Up(128, 64), Up(96, 32)
        self.out = nn.Conv2d(32, 1, 1)
        nn.init.constant_(self.out.bias, -4.0)

    def forward(self, x):
        x1 = self.c1(x); x2 = self.c2(x1); x3 = self.c3(x2); x4 = self.c4(x3)
        return self.out(self.u2(self.u1(x4, x3), x2))[:, 0]


class SpatialMem:
    """per session: the last decision each cell of a grid in the dead-reckoned start frame was hit, and the dead-reckoned pose"""

    def __init__(self, n):
        self.last = torch.full((n, MEM_N * MEM_N), -1e9, device=DEV)
        self.pose = torch.zeros(n, 3, device=DEV)
        ii = (torch.arange(IH, device=DEV).float() + 0.5) * IRES + IX0
        jj = (torch.arange(IW, device=DEV).float() + 0.5) * IRES + IY0
        self.cx, self.cy = torch.meshgrid(ii, jj, indexing="ij")         # (IH, IW) the crop's cell centres, car frame

    def reset(self, rows):
        self.last[rows] = -1e9; self.pose[rows] = 0.0

    def to_cell(self, X, Y):
        """car-frame points (n, ...) -> flat cell index in the dead-reckoned grid, in-grid mask"""
        p = self.pose
        c, s = torch.cos(p[:, 2]), torch.sin(p[:, 2])
        sh = (-1,) + (1,) * (X.dim() - 1)
        wx = p[:, 0].view(sh) + c.view(sh) * X - s.view(sh) * Y
        wy = p[:, 1].view(sh) + s.view(sh) * X + c.view(sh) * Y
        ix = torch.floor((wx + MEM_HALF) / MEM_RES).long(); iy = torch.floor((wy + MEM_HALF) / MEM_RES).long()
        ok = (ix >= 0) & (ix < MEM_N) & (iy >= 0) & (iy < MEM_N)
        return ix.clamp(0, MEM_N - 1) * MEM_N + iy.clamp(0, MEM_N - 1), ok

    def write(self, X, Y, ok, t: int):
        cell, inb = self.to_cell(X, Y)
        ok = ok & inb
        n = X.shape[0]
        flat = (torch.arange(n, device=DEV)[:, None] * (MEM_N * MEM_N) + cell)[ok]
        self.last.view(-1)[flat] = float(t)

    def read(self, t: int):
        """(n, 2, IH, IW): hit before now (0/1), recency exp(-(t - last) / 50 decisions)"""
        n = self.pose.shape[0]
        cell, inb = self.to_cell(self.cx[None].expand(n, -1, -1), self.cy[None].expand(n, -1, -1))
        last = torch.gather(self.last, 1, cell.view(n, -1)).view(n, IH, IW)
        hit = (last > -1e8) & (last < t) & inb
        rec = torch.where(hit, torch.exp(-(t - last) / 50.0), torch.zeros_like(last))
        return torch.stack([hit.float(), rec], 1)


class BoxData:
    def __init__(self, D, angles, tgt, age, neutral, rel):
        self.scan = torch.as_tensor(D["scan"], device=DEV)
        self.st = torch.as_tensor(D["st"], device=DEV)
        self.ego = torch.as_tensor(D["ego"][..., [0, 2]].astype(np.float32) / np.array([5.0, 2.0], np.float32), device=DEV)
        self.mount = torch.as_tensor(D["mount"], dtype=torch.float32, device=DEV)
        self.rs = torch.as_tensor(D["rs"].astype(bool), device=DEV)
        self.tgt = torch.as_tensor(tgt, device=DEV); self.rel = torch.as_tensor(rel, device=DEV)
        self.geo = Geo(angles)
        self.T = self.scan.shape[0]
        ci = (torch.arange(OH, device=DEV).float() + 0.5) * ORES + OX0
        cj = (torch.arange(OW, device=DEV).float() + 0.5) * ORES + OY0
        self.ox, self.oy = torch.meshgrid(ci, cj, indexing="ij")

    def stack3(self, t: int, b):
        """the policy's input at decision t: scans t, t-1, t-2 (held at the session's / a restart's first scan)"""
        return torch.stack([self.scan[max(t - k, 0), b] for k in range(3)], 1)

    def heat_target(self, t: int, b):
        """(n, OH, OW) Gaussian splats (sigma = one cell) at the targets, exactly 1 at each target's cell"""
        m = self.tgt[t, b]; r = self.rel[t, b]                                         # (n, C), (n, C, 2)
        dx = self.ox[None, None] - r[:, :, 0, None, None]; dy = self.oy[None, None] - r[:, :, 1, None, None]
        g = torch.exp(-(dx * dx + dy * dy) / (2 * ORES ** 2)) * m[:, :, None, None]
        y = g.amax(1)
        ix = torch.floor((r[..., 0] - OX0) / ORES).long(); iy = torch.floor((r[..., 1] - OY0) / ORES).long()
        ok = m & (ix >= 0) & (ix < OH) & (iy >= 0) & (iy < OW)
        n = b.numel()
        flat = (torch.arange(n, device=DEV)[:, None] * (OH * OW) + ix.clamp(0, OH - 1) * OW + iy.clamp(0, OW - 1))[ok]
        y.view(-1)[flat] = 1.0
        return y

    def scan_now(self, t: int, b):
        """(n, 1, IH, IW) the current scan's returns in the car's frame"""
        X, Y, ok = self.geo.points(self.scan[t, b][:, None], self.mount[b])
        return raster(X, Y, ok, IH, IW, IX0, IY0, IRES)


def focal(logit, y):
    p = torch.sigmoid(logit.float()).clamp(1e-4, 1 - 1e-4)
    pos = (y >= 0.999).float()
    lp = -(pos * (1 - p) ** 2 * torch.log(p)).sum()
    ln = -((1 - pos) * (1 - y) ** 4 * p ** 2 * torch.log(1 - p)).sum()
    return (lp + ln) / pos.sum().clamp_min(1.0)


class BoxNet(nn.Module):
    def __init__(self, kind, beams):
        super().__init__()
        self.kind = kind
        if kind == "box_gru":
            self.enc = Enc1D(3, beams); self.cell = nn.GRUCell(256 + 2, 256); self.dec = VecDecoder(512)
        else:
            self.unet = UNet(3)


def mem_inc(data, t, b, noise, gen):
    """the 0.1 s increment t-1 -> t of each session's pose (true, or with odometry noise x `noise`)"""
    inc = se2_rel(data.st[t - 1, b], data.st[t, b])
    if noise > 0:
        e = torch.randn(b.numel(), 3, device=DEV, generator=gen) * torch.tensor([SIG_P, SIG_P, SIG_TH], device=DEV) * noise
        inc = se2_compose(inc, e)
    return inc


def stream(net, data, sess, train, noise=0.0, opt=None, sch=None, gen=None, keep_thr=0.05, stride=2):
    """run the sessions `sess` (a group, all in parallel) through the whole record. train: losses on every `stride`-th
    decision, a step per chunk (box_gru: truncated BPTT over CHUNK decisions; box_mem / box_now: the chunk's crops in
    minibatches); else: every decision's peaks above keep_thr -> list of (t, session, x, y, p) rows"""
    n = sess.numel(); T = data.T
    kind = net.kind
    h = torch.zeros(n, 256, device=DEV)
    mem = SpatialMem(n) if kind != "box_gru" else None
    out_rows, losses = [], []
    for t0 in range(0, T, CHUNK):
        ts = list(range(t0, min(t0 + CHUNK, T)))
        with torch.set_grad_enabled(train and kind == "box_gru"):
            if kind == "box_gru":
                x = torch.stack([data.stack3(t, sess) for t in ts], 1)                     # (n, L, 3, beams)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = net.enc(x.reshape(-1, 3, x.shape[-1]).float()).view(n, len(ts), 256)
                feats = []
                for j, t in enumerate(ts):
                    if t > 0:
                        h = torch.where(data.rs[t - 1, sess][:, None], torch.zeros_like(h), h)
                    h = net.cell(torch.cat([f[:, j].float(), data.ego[t, sess]], 1), h)
                    feats.append(torch.cat([h, f[:, j].float()], 1))
                sel = [j for j, t in enumerate(ts) if (t % stride == 0 or not train)]
                fs = torch.stack([feats[j] for j in sel], 1).reshape(-1, 512)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = torch.cat([net.dec(fs[k:k + 512]).float() for k in range(0, fs.shape[0], 512)]).view(n, len(sel), OH, OW)
                if train:
                    y = torch.stack([data.heat_target(ts[j], sess) for j in sel], 1)
                    loss = focal(lg, y)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
                    losses.append(float(loss)); h = h.detach()
                else:
                    for jj, j in enumerate(sel):
                        out_rows.append(peaks_of(lg[:, jj], ts[j], sess, keep_thr))
            else:
                crops, tsel = [], []
                with torch.no_grad():
                    for t in ts:
                        if t > 0:
                            rr = data.rs[t - 1, sess]
                            if bool(rr.any()):
                                mem.reset(rr.nonzero()[:, 0])
                            mem.pose = torch.where(rr[:, None], mem.pose, se2_compose(mem.pose, mem_inc(data, t, sess, noise, gen)))
                        cur = data.scan_now(t, sess)
                        if t % stride == 0 or not train:
                            m2 = mem.read(t) if kind == "box_mem" else torch.zeros(n, 2, IH, IW, device=DEV)
                            crops.append(torch.cat([m2, cur], 1).to(torch.bfloat16)); tsel.append(t)
                        X, Y, ok = data.geo.points(data.scan[t, sess][:, None], data.mount[sess])
                        mem.write(X[:, 0], Y[:, 0], ok[:, 0], t)
                if train:
                    order = torch.randperm(len(tsel) * n, device=DEV)
                    allc = torch.stack(crops, 1).reshape(-1, 3, IH, IW)
                    ally = torch.stack([data.heat_target(t, sess) for t in tsel], 1).reshape(-1, OH, OW)
                    for k in range(0, order.numel(), 128):
                        o = order[k:k + 128]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            lg = net.unet(allc[o].float())
                        loss = focal(lg, ally[o])
                        opt.zero_grad(set_to_none=True); loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
                        losses.append(float(loss))
                else:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for c_, t in zip(crops, tsel):
                            out_rows.append(peaks_of(net.unet(c_.float()).float(), t, sess, keep_thr))
    return losses, out_rows


def peaks_of(lg, t, sess, thr):
    """(n, OH, OW) logits -> rows (t, session, x, y, p): local maxima (3x3) above thr, position = the 3x3 weighted mean"""
    p = torch.sigmoid(lg.float())
    mx = F.max_pool2d(p[:, None], 3, 1, 1)[:, 0]
    pk = (p >= mx) & (p > thr)
    if not bool(pk.any()):
        return np.zeros((0, 5), np.float32)
    nn_, ii, jj = pk.nonzero(as_tuple=True)
    pp = F.pad(p, (1, 1, 1, 1))
    w = torch.stack([pp[nn_, ii + di, jj + dj] for di in range(3) for dj in range(3)], 1)
    oi = torch.tensor([di - 1 for di in range(3) for dj in range(3)], device=DEV, dtype=torch.float32)
    oj = torch.tensor([dj - 1 for di in range(3) for dj in range(3)], device=DEV, dtype=torch.float32)
    x = OX0 + (ii.float() + 0.5 + (w * oi).sum(1) / w.sum(1)) * ORES
    y = OY0 + (jj.float() + 0.5 + (w * oj).sum(1) / w.sum(1)) * ORES
    return torch.stack([torch.full_like(x, float(t)), sess[nn_].float(), x, y, p[nn_, ii, jj]], 1).cpu().numpy()


def box_score(rows, tgt, age, neutral, rel, sess, thr, r_match=0.5):
    """match peaks (p > thr) to targets per decision: (recall per target instance, matched error, ages, TP, FP)"""
    P = rows[rows[:, 4] > thr] if len(rows) else rows
    T = tgt.shape[0]
    key = {}
    for r in P:
        key.setdefault((int(r[0]), int(r[1])), []).append(r[2:4])
    hit_age, miss_age, errs, err_age = [], [], [], []
    tp = fp = 0
    for b in sess:
        for t in range(T):
            pk = np.array(key.get((t, int(b)), np.zeros((0, 2))), np.float32).reshape(-1, 2)
            tg = np.nonzero(tgt[t, b])[0]
            used = np.zeros(len(pk), bool)
            for c in sorted(tg, key=lambda c: age[t, b, c]):
                if len(pk):
                    d = np.linalg.norm(pk - rel[t, b, c], axis=1); d[used] = np.inf
                    j = int(np.argmin(d))
                    if d[j] <= r_match:
                        used[j] = True; hit_age.append(age[t, b, c]); errs.append(d[j]); err_age.append(age[t, b, c]); continue
                miss_age.append(age[t, b, c])
            tp += int(used.sum())
            if len(pk) and (~used).any():
                nb = rel[t, b][neutral[t, b]]
                for q in pk[~used]:
                    if len(nb) and np.linalg.norm(nb - q, axis=1).min() <= r_match:
                        continue
                    fp += 1
    return dict(hit_age=np.array(hit_age), miss_age=np.array(miss_age), err=np.array(errs), err_age=np.array(err_age), tp=tp, fp=fp)


def run_box(kind, out_dir, bases):
    torch.manual_seed(0); np.random.seed(0)
    noise_tr = 1.0 if kind.endswith("_noise") else 0.0
    base_kind = kind.replace("_noise", "")
    D, angles = load_box(bases)
    tgt, age, neutral, rel = box_labels(D)
    B = tgt.shape[1]
    lay = D["idx"].astype(int) % 64                                # ALL_BOXES: session i got bank layout i % 64
    val = (lay % 5) == 4
    if os.environ.get("REPR_VAL_MOD"):
        val = (D["idx"].astype(int) % int(os.environ["REPR_VAL_MOD"])) == 0
    log(f"[{kind}] 세션 {B} (검증 = 배치 {sorted(set(lay[val].tolist()))[:20]}.. {val.sum()}개), 타깃(숨은·본 적 있는 8 m 안 상자) "
        f"결정당 {tgt.sum() / tgt.shape[0] / B:.3f}개, 나이 중앙 {np.median(age[tgt]):.1f} s")
    data = BoxData(D, angles, tgt, age, neutral, rel)
    del D
    net = BoxNet(base_kind, data.scan.shape[-1]).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    tr_s = np.nonzero(~val)[0]; va_s = np.nonzero(val)[0]
    GROUP = GROUP_GRU if base_kind == "box_gru" else GROUP_MEM
    groups_ep = math.ceil(len(tr_s) / GROUP)
    if base_kind == "box_gru":
        steps_group = math.ceil(data.T / CHUNK)
    else:
        steps_group = sum(math.ceil(GROUP * len([t for t in range(t0, min(t0 + CHUNK, data.T)) if t % 2 == 0]) / 128)
                          for t0 in range(0, data.T, CHUNK))
    total = BOX_EPOCHS * groups_ep * steps_group + 10
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.05, final_div_factor=100.0)
    log(f"[{kind}] 매개변수 {n_par:,}, 학습 세션 {len(tr_s)}, 검증 {len(va_s)}, {BOX_EPOCHS} epoch, 걸음 약 {total}")
    gen = torch.Generator(device=DEV); gen.manual_seed(3)
    rng = np.random.default_rng(0)
    curves = []; t0 = time.time()
    sub_tr = rng.choice(tr_s, size=min(len(tr_s), 64), replace=False)

    def evaluate(sessions, noise, seed=11):
        g = torch.Generator(device=DEV); g.manual_seed(seed)
        rows = []
        net.eval()
        for k in range(0, len(sessions), 2 * GROUP_MEM):
            ss = torch.as_tensor(sessions[k:k + 2 * GROUP_MEM], device=DEV)
            rows += stream(net, data, ss, False, noise=noise, gen=g)[1]
        net.train()
        return np.concatenate(rows, 0) if rows else np.zeros((0, 5), np.float32)

    def best_thr(rows, sessions):
        best = (0.3, -1.0)
        for th in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            sc = box_score(rows, tgt, age, neutral, rel, sessions, th)
            n_t = len(sc["hit_age"]) + len(sc["miss_age"])
            rec = len(sc["hit_age"]) / max(n_t, 1); prec = sc["tp"] / max(sc["tp"] + sc["fp"], 1)
            f1 = 2 * rec * prec / max(rec + prec, 1e-9)
            if f1 > best[1]:
                best = (th, f1)
        return best[0]

    for ep in range(BOX_EPOCHS):
        perm = rng.permutation(tr_s); ls = []
        for k in range(0, len(perm), GROUP):
            ss = torch.as_tensor(perm[k:k + GROUP], device=DEV)
            ls += stream(net, data, ss, True, noise=noise_tr, opt=opt, sch=sch, gen=gen)[0]
        row = {"epoch": ep + 1, "train_loss": float(np.mean(ls)), "s": time.time() - t0}
        if (ep + 1) % 3 == 0 or ep + 1 == BOX_EPOCHS:
            for nm, ses in (("val", va_s), ("trsub", sub_tr)):
                rows = evaluate(ses, noise_tr)
                sc = box_score(rows, tgt, age, neutral, rel, ses, 0.3)
                n_t = len(sc["hit_age"]) + len(sc["miss_age"])
                row[f"{nm}_recall"] = len(sc["hit_age"]) / max(n_t, 1); row[f"{nm}_prec"] = sc["tp"] / max(sc["tp"] + sc["fp"], 1)
            log(f"[{kind}] epoch {ep + 1}: 손실 {row['train_loss']:.4f} | 검증 재현율 {row['val_recall']:.3f} 정밀도 {row['val_prec']:.3f} | "
                f"학습 일부 재현율 {row['trsub_recall']:.3f} 정밀도 {row['trsub_prec']:.3f} (문턱 0.3) | {row['s']:.0f} s")
        else:
            log(f"[{kind}] epoch {ep + 1}: 손실 {row['train_loss']:.4f} | {row['s']:.0f} s")
        curves.append(row)
    thr = best_thr(evaluate(sub_tr, noise_tr), sub_tr)
    res = {}
    for nz in ((0.0, 0.5, 1.0, 2.0) if base_kind == "box_mem" else (0.0,)):
        rows = evaluate(va_s, nz)
        sc = box_score(rows, tgt, age, neutral, rel, va_s, thr)
        res[f"n{nz:g}"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sc.items()}
        n_t = len(sc["hit_age"]) + len(sc["miss_age"])
        log(f"[{kind}] 검증 (거리계 잡음 x{nz:g}, 문턱 {thr}): 재현율 {len(sc['hit_age']) / max(n_t, 1):.3f}, "
            f"정밀도 {sc['tp'] / max(sc['tp'] + sc['fp'], 1):.3f}, 오차 중앙 {np.median(sc['err']) if len(sc['err']) else float('nan'):.3f} m")
    # cost at batch 1
    net.eval()
    if base_kind == "box_gru":
        x1 = data.stack3(5, torch.as_tensor([0], device=DEV)).float()
        fl = (flops_of(net.enc, x1) + 2 * 3 * (256 + 2 + 256) * 256 + flops_of(net.dec, torch.zeros(1, 512, device=DEV)))
        ms = latency_ms(lambda: net.dec(torch.cat([net.cell(torch.cat([net.enc(x1), torch.zeros(1, 2, device=DEV)], 1),
                                                            torch.zeros(1, 256, device=DEV)), net.enc(x1)], 1)))
    else:
        x1 = torch.zeros(1, 3, IH, IW, device=DEV)
        fl = flops_of(net.unet, x1)
        m1 = SpatialMem(1); s1 = torch.as_tensor([0], device=DEV)

        def one():
            m1.pose = se2_compose(m1.pose, mem_inc(data, 5, s1, 0.0, None))
            cur = data.scan_now(5, s1)
            X, Y, ok = data.geo.points(data.scan[5, s1][:, None], data.mount[s1])
            m1.write(X[:, 0], Y[:, 0], ok[:, 0], 5)
            return net.unet(torch.cat([m1.read(6), cur], 1))
        ms = latency_ms(one)
    json.dump(dict(kind=kind, params=n_par, flops=fl, ms=ms, thr=thr, curves=curves, res=res, epochs=BOX_EPOCHS,
                   n_train_sess=int(len(tr_s)), n_val_sess=int(len(va_s)), noise_train=noise_tr,
                   peak_gpu_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                   age_all_targets=np.percentile(age[tgt], [10, 25, 50, 75, 90]).tolist()),
              open(os.path.join(out_dir, f"box_{kind}.json"), "w"), indent=1)
    log(f"[{kind}] 끝: FLOPs {fl / 1e6:.1f} M, batch 1 {ms:.2f} ms, GPU 최고 {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB")


# ---------------------------------------------------------------------------------------------------------------- table
OPP_NAMES = [("polar_gru", "극좌표 스캔 + GRU 10결정 (지금 policy 방식)"), ("polar_stack5", "극좌표 5장 겹침 (GRU 없음)"),
             ("polar_stack10", "극좌표 10장 겹침 (GRU 없음)"), ("polar_wide10", "극좌표 10장 겹침, 채널 4배 (계산량 대조)"),
             ("bev5", "위에서 본 격자 5장 (참 자세)"), ("bev10", "위에서 본 격자 10장 (참 자세)"),
             ("bev10_noise", "위에서 본 격자 10장 (거리계 잡음으로 학습·평가)")]
REF_NAMES = [("head", "라이브 policy 머리 (race33b u925, 같은 결정)"), ("still", "[참값] 지금 자리에 그대로 선다고 봄"),
             ("cv", "[참값] 지금 자리 + 참 속도로 직진"), ("oracle", "[참값] 상대 복사본 자신의 plan")]
BOX_NAMES = [("box_gru", "극좌표 스캔 + GRU (세션 전체, 지금 policy 방식)"), ("box_now", "지금 스캔만 격자로 (기억 없음: 바닥선)"),
             ("box_mem", "자기 움직임으로 옮기는 격자 기억 (참 움직임으로 학습)"), ("box_mem_noise", "같은 격자 기억 (거리계 잡음으로 학습)")]


def q(e):
    e = e[np.isfinite(e)]
    return (np.median(e), np.percentile(e, 90), len(e)) if len(e) else (np.nan, np.nan, 0)


def table(out_dir):
    L = []
    pr = lambda *a: L.append(" ".join(str(x) for x in a))
    have = [(k, n) for k, n in OPP_NAMES if os.path.exists(os.path.join(out_dir, f"opp_{k}.npz"))]
    if have:
        R = {k: np.load(os.path.join(out_dir, f"opp_{k}.npz")) for k, _ in have}
        J = {k: json.load(open(os.path.join(out_dir, f"opp_{k}.json"))) for k, _ in have}
        z = R[have[0][0]]
        tgt, msk, th = z["tgt"], z["msk"], z["meta_there"].astype(bool)
        pace, dist = z["meta_pace"], z["meta_dist"]
        rows = [(n, z[f"ref_{k}"], z["meta_head_p"] if k == "head" else None) for k, n in REF_NAMES]
        rows += [(n, R[k]["pred"], R[k]["logit"]) for k, n in have]
        if "bev10" in R and "pred_noisy_in" in R["bev10"].files:
            rows.append(("위에서 본 격자 10장, 참 자세로 학습 -> 잡음 자세로 평가", R["bev10"]["pred_noisy_in"], None))
        pr(f"(1) 상대 차 위치 예측 오차 [m] -- 따로 둔 세션의 '있음' 결정만 (0.5 s: {int((th & msk[:, 4]).sum())}개, 1.0 s: {int((th & msk[:, 9]).sum())}개)")
        pr(f"{'입력 / 모형':<52} | 0.5 s 중앙 / p90 | 1.0 s 중앙 / p90 | '있음' AUC")
        for n, P, sc in rows:
            e5 = q(err_at(P, tgt, msk, 4)[th]); e10 = q(err_at(P, tgt, msk, 9)[th])
            a = auc(np.asarray(sc, np.float64), th) if sc is not None else float("nan")
            pr(f"{n:<52} | {e5[0]:5.2f} / {e5[1]:5.2f}    | {e10[0]:5.2f} / {e10[1]:5.2f}    | {a:.3f}" if np.isfinite(a) else
               f"{n:<52} | {e5[0]:5.2f} / {e5[1]:5.2f}    | {e10[0]:5.2f} / {e10[1]:5.2f}    |   -")
        for k_, kn in ((9, "1.0 s"), (4, "0.5 s")):
            pr("")
            pr(f"{kn} 뒤 오차 중앙 / p90 [m], 상대 페이스별과 지금 거리별")
            pb = [("페이스<0.7", pace < 0.7), ("0.7-0.9", (pace >= 0.7) & (pace < 0.9)), ("≥0.9", pace >= 0.9),
                  ("거리<3m", dist < 3), ("3-6m", (dist >= 3) & (dist < 6)), ("6-10m", dist >= 6)]
            pr(f"{'':<52} | " + " | ".join(f"{b:^13}" for b, _ in pb))
            pr(f"{'(결정 수)':<52} | " + " | ".join(f"{int((th & msk[:, k_] & m).sum()):^13}" for _, m in pb))
            for n, P, _ in rows:
                e = err_at(P, tgt, msk, k_)
                cells = []
                for _, m in pb:
                    a_ = q(e[th & m])
                    cells.append(f"{a_[0]:5.2f} / {a_[1]:5.2f}")
                pr(f"{n:<52} | " + " | ".join(cells))
        pr("")
        pr("계산량 (batch 1; 랩 RTX 4060 Ti fp32, 입력 만들기 포함 = 스캔에서 격자/겹침을 만드는 시간까지)")
        pr(f"{'모형':<52} | 매개변수 | 결정당 연산(MFLOP) | ms (망만 / 입력 포함) | 학습 GPU 최고 MB")
        for k, n in have:
            j = J[k]
            pr(f"{n:<52} | {j['params'] / 1e6:6.2f} M | {j.get('flops_stream', j['flops']) / 1e6:8.1f} "
               f"{'(10결정 창 통째 ' + format(j['flops'] / 1e6, '.0f') + ')' if k == 'polar_gru' else ''} | {j['ms_net']:.2f} / {j['ms_with_input']:.2f} | {j['peak_gpu_mb']:.0f}")
        pr("")
        pr("학습 곡선 (epoch: 검증 1.0 s 중앙 / 학습 일부 1.0 s 중앙)")
        for k, n in have:
            c = J[k]["curves"]
            pick = [r for r in c if r["epoch"] in (1, 2, 5, 10, 15, len(c))]
            best = min(c, key=lambda r: r["val_med10"])
            pr(f"  {n}: " + ", ".join(f"{r['epoch']}: {r['val_med10']:.2f}/{r['trsub_med10']:.2f}" for r in pick)
               + f" | 가장 좋은 검증 {best['val_med10']:.2f} (epoch {best['epoch']})")
    hb = [(k, n) for k, n in BOX_NAMES if os.path.exists(os.path.join(out_dir, f"box_{k}.json"))]
    if hb:
        pr("")
        pr(f"(2) 지금 안 보이지만 전에 본 상자 (트랙 따라 {BOX_AHEAD:g} m 안, 따로 둔 상자 배치의 세션) -- 0.5 m 안에 맞히면 찾음")
        edges = (0.0,) + AGE_EDGES + (np.inf,)
        bn = [f"{edges[i]:g}-{edges[i + 1]:g}s" if np.isfinite(edges[i + 1]) else f"≥{edges[i]:g}s" for i in range(len(edges) - 1)]
        pr(f"{'모형 (평가 때 거리계 잡음)':<60} | 재현율 " + " ".join(f"{b:>7}" for b in bn) + " | 전체 | 정밀도 | 위치 오차 중앙/p90 m")
        for k, n in hb:
            j = json.load(open(os.path.join(out_dir, f"box_{k}.json")))
            for nz, r in j["res"].items():
                ha, ma = np.asarray(r["hit_age"]), np.asarray(r["miss_age"])
                cells = []
                for i in range(len(edges) - 1):
                    h_ = ((ha >= edges[i]) & (ha < edges[i + 1])).sum(); m_ = ((ma >= edges[i]) & (ma < edges[i + 1])).sum()
                    cells.append(f"{h_ / max(h_ + m_, 1):7.2f}")
                rec = len(ha) / max(len(ha) + len(ma), 1); prec = r["tp"] / max(r["tp"] + r["fp"], 1)
                e = np.asarray(r["err"])
                pr(f"{n + ' (x' + nz[1:] + ')':<60} | 재현율 " + " ".join(cells) + f" | {rec:4.2f} | {prec:5.2f} | "
                   + (f"{np.median(e):.2f} / {np.percentile(e, 90):.2f}" if len(e) else "-"))
        j0 = json.load(open(os.path.join(out_dir, f"box_{hb[0][0]}.json")))
        r0 = j0["res"]["n0"]
        ages = np.concatenate([np.asarray(r0["hit_age"]), np.asarray(r0["miss_age"])])
        pr("(타깃 수 나이별: " + ", ".join(f"{bn[i]} {int(((ages >= edges[i]) & (ages < edges[i + 1])).sum())}" for i in range(len(edges) - 1)) + ")")
        pr(f"{'모형':<60} | 매개변수 | 결정당 연산(MFLOP) | ms batch 1 (기억 쓰기·읽기 포함)")
        for k, n in hb:
            j = json.load(open(os.path.join(out_dir, f"box_{k}.json")))
            pr(f"{n:<60} | {j['params'] / 1e6:6.2f} M | {j['flops'] / 1e6:8.1f} | {j['ms']:.2f}")
    txt = "\n".join(L)
    print(txt)
    return txt


if __name__ == "__main__":
    mode = sys.argv[1]
    if DEV.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("REPR_GPU_FRAC", "0.3")))
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if mode == "opp":
        os.makedirs(sys.argv[3], exist_ok=True)
        run_opp(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif mode == "box":
        os.makedirs(sys.argv[3], exist_ok=True)
        run_box(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif mode == "table":
        table(sys.argv[2])
    else:
        raise SystemExit(f"unknown mode {mode}")
