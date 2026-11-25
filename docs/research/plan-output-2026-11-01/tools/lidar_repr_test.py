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
환경: REPR_EPOCHS(기본 20), REPR_BATCH(128), REPR_LR(1e-3), REPR_GPU_FRAC(0.22: 랩의 시험이 GPU에서 쫓겨나지 않게)."""
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
BATCH = int(os.environ.get("REPR_BATCH", "128"))
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
                 head_p=p["head_p"][:T], orc=p["orc"][:T], pace=np.asarray(m["c_pace"], np.float32), pl=m["pl"][:T],
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



# ------------------------------------------------------------------------------------------- point tokens (second test)
# (11-25, net_redesign_v2.md §3 A / §5) the front end A/C: every return of the last K frames as an exact point in the frame now
# (the dead-reckoned chain, no raster), grouped into runs of consecutive beams (8 in the newest frame, 16 in older ones), a
# per-point MLP on Fourier(x, y) at 0.05 / 0.2 / 1 / 5 m + age + range, max-pooled per run -> N ~ 400 tokens (K = 5). Read by
# 10 time-aligned queries anchored at plan points + 1 car query: 2 blocks of [cross-attention query -> tokens with the
# RELATIVE position Fourier(token - anchor) and age in keys and values, self-attention among the queries, FFN], width 128,
# 4 heads, pre-LN, no token-token attention.
FOURIER_S = (0.05, 0.2, 1.0, 5.0)
PAD_BEAMS, RUN_NEW, RUN_OLD = 1088, 8, 16


def fourier(xy):
    """(..., 2) [m] -> (..., 16): sin / cos of 2 pi x / s and 2 pi y / s at the four scales"""
    sc = torch.tensor([2 * math.pi / v for v in FOURIER_S], device=xy.device, dtype=xy.dtype)
    a = xy[..., None, :] * sc[:, None]                                       # (..., 4, 2)
    return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)


def point_set(geo, sc, rel, mount, mirror):
    """sc (n, K, beams) normalised ranges, rel (n, K, 3) each frame's pose in the frame now -> X, Y [m], r (range / max),
    ok (n, K, PAD_BEAMS) in the frame now (padded beams invalid); mirror (n,) bool flips y"""
    px, py, ok = geo.points(sc, mount)
    c, s_ = torch.cos(rel[..., 2])[..., None], torch.sin(rel[..., 2])[..., None]
    X = rel[..., 0:1] + c * px - s_ * py; Y = rel[..., 1:2] + s_ * px + c * py
    Y = torch.where(mirror[:, None, None], -Y, Y)
    pad = PAD_BEAMS - X.shape[-1]
    return F.pad(X, (0, pad)), F.pad(Y, (0, pad)), F.pad(sc.float(), (0, pad), value=1.0), F.pad(ok, (0, pad), value=False)


class PtsBlock(nn.Module):
    def __init__(self, d: int, H: int, rel_d: int = 17):
        super().__init__()
        self.H = H
        self.ln_q, self.ln_t = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d) for _ in range(4))
        self.rk, self.rv = nn.Linear(rel_d, d), nn.Linear(rel_d, d)
        self.ln_s = nn.LayerNorm(d); self.sa = nn.MultiheadAttention(d, H, batch_first=True)
        self.ln_f = nn.LayerNorm(d); self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, tok, rel, valid):
        n, Q, d = q.shape; N = tok.shape[1]; H = self.H; dh = d // H
        t = self.ln_t(tok)
        qq = self.wq(self.ln_q(q)).view(n, Q, H, dh)
        k = (self.wk(t)[:, None] + self.rk(rel)).view(n, Q, N, H, dh)       # keys / values: token + relative position
        v = (self.wv(t)[:, None] + self.rv(rel)).view(n, Q, N, H, dh)
        lg = torch.einsum("nqhd,nqjhd->nhqj", qq, k).float() / math.sqrt(dh)
        lg = lg.masked_fill(~valid[:, None, None, :], float("-inf"))
        att = torch.softmax(lg, -1).to(v.dtype)
        o = torch.einsum("nhqj,nqjhd->nqhd", att, v).reshape(n, Q, d)
        q = q + self.wo(o)
        h = self.ln_s(q)
        q = q + self.sa(h, h, h, need_weights=False)[0]
        return q + self.ffn(self.ln_f(q))


class PtsNet(nn.Module):
    """x = (X, Y, r, ok, anchors (n, 10, 2)), ego (n, 10, 2) -> (n, 10 q_out [+ ego_out])"""

    def __init__(self, K: int, q_out: int = 2, ego_out: int = 1, d: int = 128, heads: int = 4, blocks: int = 2, n_q: int = 10,
                 anc_extra: int = 0):
        super().__init__()
        self.q_extra = nn.Linear(anc_extra, d) if anc_extra else None      # e.g. the query pose's heading (cos, sin)
        self.K, self.N, self.n_q, self.q_out, self.ego_out = K, K, n_q, q_out, ego_out
        self.pmlp = nn.Sequential(nn.Linear(18, 64), nn.ReLU(), nn.Linear(64, 64))
        self.tok = nn.Sequential(nn.Linear(64 + 16 + 1, d), nn.ReLU(), nn.Linear(d, d))
        self.q0 = nn.Parameter(torch.randn(n_q + 1, d) * 0.02)               # learned embedding = the query's time slot
        self.q_anchor = nn.Linear(16, d); self.q_ego = nn.Linear(20, d)
        self.blocks = nn.ModuleList([PtsBlock(d, heads) for _ in range(blocks)])
        self.ln = nn.LayerNorm(d)
        self.out_q = nn.Linear(d, q_out)
        self.out_e = nn.Linear(d, ego_out) if ego_out else None

    def tokens(self, X, Y, r, ok):
        n, K, P = X.shape
        age = (torch.arange(K, device=X.device, dtype=X.dtype) * 0.1)[None, :, None].expand(n, K, P)
        xy = torch.stack([X, Y], -1)
        h = self.pmlp(torch.cat([fourier(xy), age[..., None], r[..., None]], -1))            # (n, K, P, 64)
        h = h.masked_fill(~ok[..., None], float("-inf"))
        okf = ok.to(X.dtype)

        def runs(hh, oo, pp, aa, L):            # (n, F, P, ...) -> runs of L beams
            nF = hh.shape[1]
            hh = hh.reshape(n, nF, P // L, L, -1).amax(3)
            cnt = oo.reshape(n, nF, P // L, L).sum(3)
            pos = (pp * oo[..., None]).reshape(n, nF, P // L, L, 2).sum(3) / cnt.clamp_min(1.0)[..., None]
            return hh.reshape(n, -1, hh.shape[-1]), pos.reshape(n, -1, 2), aa[:, :, ::L].reshape(n, -1), (cnt > 0).reshape(n, -1)
        parts = [runs(h[:, :1], okf[:, :1], xy[:, :1], age[:, :1], RUN_NEW)]
        if K > 1:
            parts.append(runs(h[:, 1:], okf[:, 1:], xy[:, 1:], age[:, 1:], RUN_OLD))
        hh, pos, ag, val = (torch.cat([p_[i] for p_ in parts], 1) for i in range(4))
        hh = torch.where(val[..., None], hh, torch.zeros_like(hh))
        val = val.clone(); val[:, 0] = val[:, 0] | ~val.any(1)                 # never an all-masked row
        tok = self.tok(torch.cat([hh, fourier(pos), ag[..., None]], -1))
        return tok, pos, ag, val

    def forward(self, x, ego):
        X, Y, r, ok, anc = x
        X, Y, r, anc = X.float(), Y.float(), r.float(), anc.float()
        ext = anc[..., 2:]; anc = anc[..., :2]
        tok, pos, ag, val = self.tokens(X, Y, r, ok)
        n = X.shape[0]
        a = torch.cat([anc, torch.zeros(n, 1, 2, device=anc.device)], 1)                     # the car query sits at the car
        qa = self.q_anchor(fourier(anc))
        if self.q_extra is not None:
            qa = qa + self.q_extra(ext)
        q = self.q0[None] + torch.cat([qa, self.q_ego(ego.reshape(n, -1).float())[:, None]], 1)
        Q, N = a.shape[1], pos.shape[1]
        rel = torch.cat([fourier(pos[:, None] - a[:, :, None]), ag[:, None, :, None].expand(n, Q, N, 1)], -1)
        for blk in self.blocks:
            q = blk(q, tok, rel, val)
        q = self.ln(q)
        out = self.out_q(q[:, :-1]).reshape(n, -1)
        return torch.cat([out, self.out_e(q[:, -1])], 1) if self.out_e is not None else out


def make_opp_net(kind, beams, geo):
    if kind.startswith("pts"):
        net = PtsNet(int(kind[3:].split("_")[0]))
        net.kind, net.noise = kind, (1.0 if kind.endswith("_noise") else 0.0)
        return net
    return OppNet(kind, beams, geo)


def anchors_of(pl_w, st_now, mirror):
    """plan points in the world (n, 10, 2) -> in the car frame now (n, 10, 2), y flipped where mirror"""
    c, s_ = torch.cos(st_now[:, 2])[:, None], torch.sin(st_now[:, 2])[:, None]
    d = pl_w - st_now[:, None, :2]
    a = torch.stack([d[..., 0] * c + d[..., 1] * s_, -d[..., 0] * s_ + d[..., 1] * c], -1)
    return torch.where(mirror[:, None, None], a * torch.tensor([1.0, -1.0], device=a.device), a)


class OppData:
    """everything on the GPU; `batch(rows)` builds a model's inputs"""

    def __init__(self, D, angles, tgt, msk, meta):
        # the scans stay in (pinned) host memory, gathered per batch: the lab GPU is shared with the race exams
        self.scan = torch.as_tensor(D["scan"]).pin_memory() if DEV.type == "cuda" else torch.as_tensor(D["scan"])
        self.t_c = torch.as_tensor(meta["t"]).long(); self.b_c = torch.as_tensor(meta["b"]).long()
        self.st = torch.as_tensor(D["st"], dtype=torch.float32, device=DEV)         # (T+1, B, 3)
        ego = D["ego"][..., [0, 2]].astype(np.float32) / np.array([5.0, 2.0], np.float32)
        self.ego = torch.as_tensor(ego, device=DEV)                                  # (T, B, 2)
        self.mount = torch.as_tensor(D["mount"], dtype=torch.float32, device=DEV)
        self.t = torch.as_tensor(meta["t"], device=DEV).long(); self.b = torch.as_tensor(meta["b"], device=DEV).long()
        self.tgt = torch.as_tensor(tgt, device=DEV); self.msk = torch.as_tensor(msk, device=DEV)
        self.there = torch.as_tensor(meta["there"], device=DEV) if "there" in meta else None
        self.pl = torch.as_tensor(D["pl"], dtype=torch.float32, device=DEV) if "pl" in D else None     # (T, B, 10, 2) world
        # pts anchors / query points: "prev" = the previous plan moved on (opponent probe), "plan" = this decision's plan (clearance),
        # "future" = our own realised poses at t + 0.1 k with heading (car-clearance probe)
        self.anchor_mode = "prev"
        self.geo = Geo(angles)

    def gather(self, t_c, b_c, ages):
        """scans at decisions t - age of sessions b (host gather, then to the GPU): (n, *ages.shape, beams)"""
        sh = (-1,) + (1,) * ages.dim()
        return self.scan[t_c.view(sh) - ages[None], b_c.view(sh)].to(DEV, non_blocking=True)

    def batch(self, net: OppNet, rows, mirror=None, gen=None):
        t, b = self.t[rows], self.b[rows]
        rc = rows.cpu(); t_c, b_c = self.t_c[rc], self.b_c[rc]
        n = rows.numel()
        if mirror is None:
            mirror = torch.zeros(n, dtype=torch.bool, device=DEV)
        ages = torch.arange(10, device=DEV)
        ego = self.ego[(t[:, None] - ages[None]), b[:, None]]                     # (n, 10, 2) newest first
        ego = torch.where(mirror[:, None, None], ego * torch.tensor([1.0, -1.0], device=DEV), ego)
        k = net.kind
        if k == "polar_gru":
            # step s (oldest first) = the decision 9 - s back; its 3-scan stack as the policy reads it (newest channel first)
            a = torch.arange(9, -1, -1)[:, None] + torch.arange(3)[None]                               # (10, 3)
            x = self.gather(t_c, b_c, a)                                                              # (n, 10, 3, beams)
            x = torch.where(mirror[:, None, None, None], x.flip(-1), x)
        elif k.startswith("polar"):
            x = self.gather(t_c, b_c, torch.arange(net.N))                                       # (n, N, beams)
            x = torch.where(mirror[:, None, None], x.flip(-1), x)
        elif k.startswith("pts"):
            N = net.K
            sc = self.gather(t_c, b_c, torch.arange(N))
            poses = self.st[(t[:, None] - torch.arange(N, device=DEV)[None]), b[:, None]]
            rel = rel_chain(poses, net.noise, gen)
            X, Y, r, ok = point_set(self.geo, sc, rel, self.mount[b], mirror)
            if self.anchor_mode == "prev":  # the previous plan's points k = 1..9 are now 0.1 k ahead; the 10th extrapolated
                pp = self.pl[t - 1, b]
                pw = torch.cat([pp[:, 1:], 2.0 * pp[:, 9:10] - pp[:, 8:9]], 1)
                x = (X, Y, r, ok, anchors_of(pw, self.st[t, b], mirror))
            else:
                x = (X, Y, r, ok, self.qpoints(rows, mirror))
        else:
            N = net.N
            sc = self.gather(t_c, b_c, torch.arange(N))                                          # (n, N, beams)
            poses = self.st[(t[:, None] - torch.arange(N, device=DEV)[None]), b[:, None]]         # (n, N, 3) world
            rel = rel_chain(poses, net.noise, gen)                                               # (n, N, 3) in frame now
            px, py, ok = self.geo.points(sc, self.mount[b])
            c, s = torch.cos(rel[..., 2])[..., None], torch.sin(rel[..., 2])[..., None]
            X = rel[..., 0:1] + c * px - s * py; Y = rel[..., 1:2] + s * px + c * py
            Y = torch.where(mirror[:, None, None], -Y, Y)
            x = raster(X, Y, ok)
        if self.anchor_mode != "prev":      # the clearance probes: their own targets / mask (mirror-invariant)
            return x, ego, self.tgt[rows], self.msk[rows], None
        tg = self.tgt[rows]
        tg = torch.where(mirror[:, None, None], tg * torch.tensor([1.0, -1.0], device=DEV), tg)
        return x, ego, tg, self.msk[rows], self.there[rows]

    def qpoints(self, rows, mirror=None):
        """this decision's plan points in the car frame now (n, 10, 2) -- the clearance probe's query points"""
        t, b = self.t[rows], self.b[rows]
        if mirror is None:
            mirror = torch.zeros(rows.numel(), dtype=torch.bool, device=DEV)
        if self.anchor_mode == "future":    # our realised poses at t + 0.1 k in the car frame now: (n, 10, 4) x, y, cos, sin
            k = torch.arange(1, 11, device=DEV)
            tk = (t[:, None] + k[None]).clamp(max=self.st.shape[0] - 1)
            rel = se2_rel(self.st[t, b][:, None].expand(-1, 10, -1), self.st[tk, b[:, None]])
            q = torch.stack([rel[..., 0], rel[..., 1], torch.cos(rel[..., 2]), torch.sin(rel[..., 2])], -1)
            return torch.where(mirror[:, None, None], q * torch.tensor([1.0, -1.0, 1.0, -1.0], device=DEV), q)
        return anchors_of(self.pl[t, b], self.st[t, b], mirror)


def opp_loss(out, tg, msk, there):
    pts = out[:, :2 * PRED_N].float().view(-1, PRED_N, 2)
    m = (msk & there[:, None]).float()
    hub = F.huber_loss(pts, tg / PRED_SCALE, delta=0.2, reduction="none").sum(-1)
    lp = (hub * m).sum() / m.sum().clamp_min(1.0)
    lv = F.binary_cross_entropy_with_logits(out[:, -1].float(), there.float())
    return lp + 0.2 * lv, lp, lv


def opp_predict(net, data, rows, bs=128, gen=None):
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
    net = make_opp_net(kind, beams, data.geo).to(DEV)
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
    if kind in ("bev10", "pts5"):
        # the noise-free model read with noisy odometry (robustness without training on it)
        net.noise = 1.0; g.manual_seed(11)
        res["pred_noisy_in"], _ = opp_predict(net, data, va, gen=g); net.noise = 0.0
    # cost: params, FLOPs, ms at batch 1 (incl. building the input from the scans for the BEV)
    r1 = va[:1]
    x1, e1, _, _, _ = data.batch(net, r1)
    net.eval()
    with torch.no_grad():
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



# ---------------------------------------------------------------------------------------------- clearance precision probe
# (11-25, net_redesign_v2 §5.2) how exactly does each front end read free space AT the plan's points? For each decision and
# each plan point k = 1..10 (the plan this decision drew, `pl`, in the car frame now -- with exact execution the car is there
# at t + 0.1 k for k = 1; later points are re-planned, see the note in the table), the signed gap from the point to the
# nearest box edge (footprints from the record's fp_<shape>, poses static) and to the nearest wall (the walls-only EDT,
# bilinear on its 1.25 cm grid), both clipped to [-0.5, 2] m. Solo box sessions (line3), split by box layout as the box probe.
# Models (query points given as input): clr_polar_gru (today: pooled vector + points), clr_bev10 (pooled vector + points),
# clr_bev10b (the same CNN read bilinearly at the points: 0.25 / 0.5 / 1 m feature maps), clr_pts5 (queries at the points).
CLR_CLIP = (-0.5, 2.0)


def load_clr(bases):
    per_sess = ("mount", "run", "idx", "bxy", "byaw", "bhs", "blive")
    L = {}; angles = wall = None
    for r, b in enumerate(bases):
        m = np.load(b + ".npz"); s = np.load(b + "_scan.npz"); v = np.load(b + "_boxvis.npz")
        T, B = s["scan"].shape[:2]
        sid0, pose0 = m["sid"][0], m["pose"][0]
        assert (m["sid"][-1] == sid0).all() and np.allclose(m["pose"][-1], pose0, atol=1e-4), "boxes moved"
        hs = np.zeros(sid0.shape + (2,), np.float32)
        for k in np.unique(sid0[sid0 >= 0]):
            hs[sid0 == k] = np.abs(m[f"fp_{k}"]).max(0)
        d = dict(scan=s["scan"], ego=s["ego"], mount=s["mount"], st=m["st"][:T + 1, :, :3].astype(np.float32), rs=m["rs"][:T],
                 pl=m["pl"][:T].astype(np.float32), view=v["view"][:T], bxy=pose0[..., :2].astype(np.float32),
                 byaw=pose0[..., 2].astype(np.float32), bhs=hs, blive=sid0 >= 0,
                 run=np.full(B, r, np.int16), idx=np.arange(B, dtype=np.int16))
        if angles is None:
            angles = s["angles"]
            wall = (m["edt"].astype(np.float32), float(m["t_res"][0]), float(m["t_origin"][0, 0]), float(m["t_origin"][0, 1]),
                    int(m["t_H"][0]), int(m["t_W"][0]))
            assert int(m["t_off"].shape[0]) == 1, "one track"
        for k, x in d.items():
            L.setdefault(k, []).append(x)
        log(f"기록 {b}: 결정 {T} x 세션 {B}")
    return {k: np.concatenate(x, 0 if k in per_sess else 1) for k, x in L.items()}, angles, wall


def edt_bilinear(P, wall):
    """walls-only distance [m] at world points P (..., 2), bilinear between the EDT's cell centres (outside the grid = 0)"""
    edt, res, ox, oy, H, W = wall
    fx = (P[..., 0] - ox) / res; fy = (P[..., 1] - oy) / res
    c0 = np.floor(fx).astype(np.int64); r0 = np.floor(fy).astype(np.int64)
    wx = (fx - c0).astype(np.float32); wy = (fy - r0).astype(np.float32)

    def at(r_, c_):
        ins = (r_ >= 0) & (r_ < H) & (c_ >= 0) & (c_ < W)
        return np.where(ins, edt[np.clip(r_, 0, H - 1) * W + np.clip(c_, 0, W - 1)], 0.0)
    return ((1 - wx) * (1 - wy) * at(r0, c0) + wx * (1 - wy) * at(r0, c0 + 1)
            + (1 - wx) * wy * at(r0 + 1, c0) + wx * wy * at(r0 + 1, c0 + 1)).astype(np.float32)


def clr_samples(D, wall):
    scan = D["scan"]; T, B = scan.shape[:2]
    brk = D["rs"].astype(bool)
    cb = np.concatenate([np.zeros((1, B), np.int64), np.cumsum(brk, 0)], 0)
    t = np.arange(HIST, T)
    clean = (cb[t] - cb[t - HIST]) == 0
    tt, bb = np.nonzero(clean); tt = t[tt]
    P = D["pl"][tt, bb]                                                       # (S, 10, 2) world
    gb = np.empty(P.shape[:2], np.float32); cn = np.empty(P.shape[:2], np.int64)
    for i0 in range(0, len(tt), 20000):                                       # in pieces (S x 10 x C)
        sl = slice(i0, i0 + 20000); b_ = bb[sl]
        d = P[sl, :, None, :] - D["bxy"][b_][:, None]
        c, s_ = np.cos(D["byaw"][b_])[:, None], np.sin(D["byaw"][b_])[:, None]
        lx = np.abs(d[..., 0] * c + d[..., 1] * s_) - D["bhs"][b_][:, None, :, 0]
        ly = np.abs(-d[..., 0] * s_ + d[..., 1] * c) - D["bhs"][b_][:, None, :, 1]
        sd = np.hypot(np.maximum(lx, 0), np.maximum(ly, 0)) + np.minimum(np.maximum(lx, ly), 0)
        sd = np.where(D["blive"][b_][:, None], sd, np.inf)
        gb[sl] = sd.min(2); cn[sl] = sd.argmin(2)
    gw = edt_bilinear(P, wall)
    vis = D["view"][tt[:, None], bb[:, None], cn]                              # the nearest box in view now
    tgt = np.stack([np.clip(gb, *CLR_CLIP), np.clip(gw, *CLR_CLIP)], -1).astype(np.float32)
    msk = np.ones(P.shape[:2], bool)
    st = D["st"]
    real = np.stack([np.linalg.norm(st[np.minimum(tt + k, T), bb, :2] - P[:, k - 1], axis=-1) for k in range(1, 11)], 1)
    meta = dict(t=tt, b=bb, run=D["run"][bb], idx=D["idx"][bb], vis=vis, real=real.astype(np.float32),
                gb_raw=gb, gw_raw=gw)
    return tgt, msk, meta


class ClrNet(nn.Module):
    """clr_<base>: (box gap, wall gap) at the plan's 10 points (query = x, y); car_<base>: the gap to the other car at our own
    poses t + 0.1 k (query = x, y, cos, sin of the heading). base: polar_gru / bev10 (pooled) / bev10b (bilinear) / pts5"""
    BASE = {"polar_gru": "polar_gru", "bev10": "bev10", "bev10b": "bev10", "pts5": "pts5"}

    def __init__(self, ckind, beams):
        super().__init__()
        pre, base = ckind.split("_", 1)
        self.ckind, self.base, self.kind, self.noise = ckind, base, self.BASE[base], 0.0
        qd, out = (4, 1) if pre == "car" else (2, 2)
        self.qd, self.out = qd, out
        if base == "polar_gru":
            self.enc = Enc1D(3, beams); self.gru = nn.GRU(256 + 2, 256, batch_first=True)
            self.head = mlp_head(256 + 256 + 20 + 10 * qd, 10 * out)
        elif base == "bev10":
            self.N = 10; self.enc = Enc2D(10); self.head = mlp_head(256 + 20 + 10 * qd, 10 * out)
        elif base == "bev10b":
            self.N = 10; self.enc = Enc2D(10)
            self.pt = mlp_head(32 + 64 + 128 + 16 + 20 + (qd - 2), out)
        else:
            self.pts = PtsNet(5, q_out=out, ego_out=0, anc_extra=qd - 2); self.K = self.N = 5

    def forward(self, x, ego, qp):
        n = ego.shape[0]; e = ego.reshape(n, -1).float(); qf = qp.float().reshape(n, -1) / 5.0
        if self.base == "polar_gru":
            f = self.enc(x.reshape(n * 10, 3, -1).float()).view(n, 10, 256)
            h, _ = self.gru(torch.cat([f, ego.flip(1).float()], -1))
            return self.head(torch.cat([h[:, -1], f[:, -1], e, qf], 1)).view(n, 10, self.out)
        if self.base == "bev10":
            return self.head(torch.cat([self.enc(x.float()), e, qf], 1)).view(n, 10, self.out)
        if self.base == "bev10b":
            maps, z = [], x.float()
            for i, layer in enumerate(self.enc.conv):
                z = layer(z)
                if i in (1, 5, 7):                     # 0.25 m (32 ch), 0.5 m (64 ch), 1 m (128 ch) feature maps
                    maps.append(z)
                if i == 7:
                    break
            qq = qp.float()
            g = torch.stack([2 * (qq[..., 1] - GY0) / (GW * RES) - 1, 2 * (qq[..., 0] - GX0) / (GH * RES) - 1], -1)[:, :, None]  # (n,10,1,2)
            fs = [F.grid_sample(mp, g.to(mp.dtype), mode="bilinear", align_corners=False)[..., 0].transpose(1, 2) for mp in maps]
            z = torch.cat(fs + [fourier(qq[..., :2]).to(fs[0].dtype), e[:, None].expand(n, 10, 20).to(fs[0].dtype),
                           qq[..., 2:].to(fs[0].dtype)], -1)
            return self.pt(z)
        return self.pts(x, ego).view(n, 10, self.out)


def run_clr(kind, out_dir, bases):
    torch.manual_seed(0); np.random.seed(0)
    D, angles, wall = load_clr(bases)
    tgt, msk, meta = clr_samples(D, wall)
    lay = meta["idx"].astype(int) % 64
    val = (lay % 5) == 4
    if os.environ.get("REPR_VAL_MOD"):
        val = (meta["idx"].astype(int) % int(os.environ["REPR_VAL_MOD"])) == 0
    log(f"[{kind}] 표본 {len(val)} (검증 {val.mean() * 100:.1f} %), 상자 틈 중앙 {np.median(tgt[..., 0]):.2f} m (< 0.3 m {np.mean(tgt[..., 0] < 0.3) * 100:.1f} %), "
        f"벽 틈 중앙 {np.median(tgt[..., 1]):.2f} m (< 0.3 m {np.mean(tgt[..., 1] < 0.3) * 100:.1f} %), plan 점과 실제 자리 차 중앙 k=1 "
        f"{np.median(meta['real'][:, 0]) * 100:.1f} cm / k=10 {np.median(meta['real'][:, 9]) * 100:.1f} cm")
    data = OppData(D, angles, tgt, msk, meta); data.anchor_mode = "plan"
    del D
    net = ClrNet(kind, data.scan.shape[-1]).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    tr = torch.as_tensor(np.nonzero(~val)[0], device=DEV); va = torch.as_tensor(np.nonzero(val)[0], device=DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps_ep = tr.numel() // BATCH; total = EPOCHS * steps_ep
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.05, final_div_factor=100.0)
    log(f"[{kind}] 매개변수 {n_par:,}, 학습 {tr.numel()}, 검증 {va.numel()}, {EPOCHS} epoch x {steps_ep}")
    sub_tr = tr[torch.randperm(tr.numel(), device=DEV)[:20000]]

    def predict(rows, bs=128):
        net.eval(); P_ = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, rows.numel(), bs):
                r = rows[i:i + bs]
                x, ego, _, _, _ = data.batch(net, r)
                P_.append(net(x, ego, data.qpoints(r)).float().cpu())
        net.train()
        return torch.cat(P_).numpy()

    def errs(P_, ri):
        e = np.abs(P_ - tgt[ri])
        o = {}
        for j, nm in ((0, "box"), (1, "wall")):
            g = tgt[ri][..., j]
            o[f"{nm}_med"] = float(np.median(e[..., j][g < CLR_CLIP[1]])); o[f"{nm}_near"] = float(np.median(e[..., j][g < 0.3]))
        return o
    curves = []; t0 = time.time()
    for ep in range(EPOCHS):
        perm = tr[torch.randperm(tr.numel(), device=DEV)]; tl = []
        for i in range(steps_ep):
            r = perm[i * BATCH:(i + 1) * BATCH]
            mir = torch.rand(r.numel(), device=DEV) < 0.5
            x, ego, tg, _, _ = data.batch(net, r, mir)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x, ego, data.qpoints(r, mir))
            loss = F.huber_loss(out.float(), tg, delta=0.05)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
            tl.append(float(loss))
        row = {"epoch": ep + 1, "train_loss": float(np.mean(tl)), "s": time.time() - t0}
        row.update({f"val_{k}": v for k, v in errs(predict(va), va.cpu().numpy()).items()})
        row.update({f"trsub_{k}": v for k, v in errs(predict(sub_tr), sub_tr.cpu().numpy()).items()})
        curves.append(row)
        log(f"[{kind}] epoch {ep + 1}: 손실 {row['train_loss']:.5f} | 검증 상자 {row['val_box_med'] * 100:.1f} cm (가까움 "
            f"{row['val_box_near'] * 100:.1f}), 벽 {row['val_wall_med'] * 100:.1f} cm (가까움 {row['val_wall_near'] * 100:.1f}) | 학습 일부 상자 "
            f"{row['trsub_box_med'] * 100:.1f} 벽 {row['trsub_wall_med'] * 100:.1f} cm | {row['s']:.0f} s")
    P_ = predict(va)
    r1 = va[:1]
    x1, e1, _, _, _ = data.batch(net, r1); q1 = data.qpoints(r1)
    net.eval()
    with torch.no_grad():
        fl = flops_of(net, x1, e1, q1)
    if kind == "clr_polar_gru" and flops_of(net.gru, torch.zeros(1, 10, 258, device=DEV)) == 0:
        fl += 10 * 2 * 3 * (256 + 2 + 256) * 256
    ms_net = latency_ms(lambda: net(x1, e1, q1))
    ms_all = latency_ms(lambda: net(*data.batch(net, r1)[:2], data.qpoints(r1)))
    vi = val.nonzero()[0]
    np.savez_compressed(os.path.join(out_dir, f"{kind}.npz"), pred=P_, tgt=tgt[vi], vis=meta["vis"][vi], real=meta["real"][vi])
    json.dump(dict(kind=kind, params=n_par, flops=fl, ms_net=ms_net, ms_with_input=ms_all, curves=curves, epochs=EPOCHS,
                   batch=BATCH, lr=LR, n_train=int(tr.numel()), n_val=int(va.numel()),
                   peak_gpu_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                   real_dev_med_cm=[float(np.median(meta["real"][:, k]) * 100) for k in range(10)]),
              open(os.path.join(out_dir, f"{kind}.json"), "w"), indent=1)
    log(f"[{kind}] 끝: FLOPs {fl / 1e6:.1f} M, batch 1 {ms_net:.2f} ms (입력 포함 {ms_all:.2f} ms), GPU 최고 {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB")



# ----------------------------------------------------------------------------------------- time-aligned gap to a moving car
# (11-25, the coordinator: "the one that actually tests the query design on MOVING cars") on the race records: for each
# decision with the other car 'there' and k = 1..10, the signed gap between OUR car at our realised pose at t + 0.1 k and the
# OTHER car at the same time t + 0.1 k, both 0.58 x 0.31 m boxes -- exact box-box signed distance (separated: the nearest
# vertex-to-box distance; overlapping: minus the smallest separating-axis overlap), clipped to [-1, 3] m. A step counts when
# no restart / jump lies between t and t + 0.1 k and the other car is on track (the opponent targets' rule). Queries = our
# poses at t + 0.1 k in the car frame now (x, y, cos, sin): with exact execution that is the plan, except that the car
# re-plans every decision. References (truth): the other car standing where it is now; now + its true speed straight on.
CAR_LW = (0.58, 0.31)
CAR_CLIP = (-1.0, 3.0)


def rect_corners(x, y, th, L=CAR_LW[0], W=CAR_LW[1]):
    c, s_ = np.cos(th), np.sin(th)
    ox = np.array([1, -1, -1, 1], np.float32) * L / 2; oy = np.array([1, 1, -1, -1], np.float32) * W / 2
    return np.stack([x[..., None] + c[..., None] * ox - s_[..., None] * oy, y[..., None] + s_[..., None] * ox + c[..., None] * oy], -1)


def point_rect_sd(px, py, x, y, th, L=CAR_LW[0], W=CAR_LW[1]):
    """signed distance (negative inside) from points (..., P) to boxes (...) broadcast"""
    c, s_ = np.cos(th)[..., None], np.sin(th)[..., None]
    dx, dy = px - x[..., None], py - y[..., None]
    lx = np.abs(dx * c + dy * s_) - L / 2; ly = np.abs(-dx * s_ + dy * c) - W / 2
    return np.hypot(np.maximum(lx, 0), np.maximum(ly, 0)) + np.minimum(np.maximum(lx, ly), 0)


def rect_rect_sd(a, b):
    """a, b (..., 3) poses (x, y, yaw) of two 0.58 x 0.31 boxes -> exact signed distance (...)"""
    ca, cb = rect_corners(a[..., 0], a[..., 1], a[..., 2]), rect_corners(b[..., 0], b[..., 1], b[..., 2])
    sep = np.minimum(point_rect_sd(ca[..., 0], ca[..., 1], b[..., 0], b[..., 1], b[..., 2]).min(-1),
                     point_rect_sd(cb[..., 0], cb[..., 1], a[..., 0], a[..., 1], a[..., 2]).min(-1))
    pen = np.full(a.shape[:-1], np.inf, np.float32); over = np.ones(a.shape[:-1], bool)
    for th in (a[..., 2], a[..., 2] + np.pi / 2, b[..., 2], b[..., 2] + np.pi / 2):
        u = np.stack([np.cos(th), np.sin(th)], -1)[..., None, :]
        pa, pb = (ca * u).sum(-1), (cb * u).sum(-1)
        ov = np.minimum(pa.max(-1), pb.max(-1)) - np.maximum(pa.min(-1), pb.min(-1))
        over &= ov > 0; pen = np.minimum(pen, ov)
    return np.where(over, -pen, np.maximum(sep, 0.0)).astype(np.float32)


def car_samples(D):
    scan = D["scan"]; T, B = scan.shape[:2]
    st = D["st"].astype(np.float32); op = D["op"].astype(np.float32)
    jump = np.linalg.norm(op[1:T + 1, :, :2] - op[:T, :, :2], axis=-1) > 2.0
    brk = D["rs"].astype(bool) | D["ors"].astype(bool) | jump
    cb = np.concatenate([np.zeros((1, B), np.int64), np.cumsum(brk, 0)], 0)
    t = np.arange(HIST, T)
    keep = ((cb[t] - cb[t - HIST]) == 0) & D["there"][t].astype(bool)
    tt, bb = np.nonzero(keep); tt = t[tt]
    gap = np.zeros((len(tt), 10), np.float32); msk = np.zeros((len(tt), 10), bool)
    still = np.zeros_like(gap); cv = np.zeros_like(gap)
    v0 = op[tt, bb, 3]; o0 = op[tt, bb, :3]
    for k in range(1, 11):
        tk = np.minimum(tt + k, T)
        msk[:, k - 1] = (tt + k <= T) & ((cb[tk, bb] - cb[tt, bb]) == 0) & D["on"][tk, bb].astype(bool)
        me = st[tk, bb, :3]
        gap[:, k - 1] = rect_rect_sd(me, op[tk, bb, :3])
        still[:, k - 1] = rect_rect_sd(me, o0)
        ocv = o0.copy(); ocv[:, 0] += v0 * 0.1 * k * np.cos(o0[:, 2]); ocv[:, 1] += v0 * 0.1 * k * np.sin(o0[:, 2])
        cv[:, k - 1] = rect_rect_sd(me, ocv)
    clip = lambda g: np.clip(g, *CAR_CLIP).astype(np.float32)
    meta = dict(t=tt, b=bb, there=np.ones(len(tt), bool), dist=D["dist"][tt, bb], pace=D["pace"][bb], run=D["run"][bb],
                idx=D["idx"][bb], ref_still=clip(still), ref_cv=clip(cv))
    return clip(gap), msk, meta


def run_car(kind, out_dir, bases):
    torch.manual_seed(0); np.random.seed(0)
    D, angles = load_opp(bases)
    tgt, msk, meta = car_samples(D)
    run, idx = meta["run"].astype(int), meta["idx"].astype(int)
    rep_ = idx // 16
    val = (rep_ == 7) | ((rep_ == 3) & (run == 1))
    if os.environ.get("REPR_VAL_MOD"):
        val = (idx % int(os.environ["REPR_VAL_MOD"])) == 0
    m_ = msk
    log(f"[{kind}] '있음' 결정 {len(val)} (검증 {val.mean() * 100:.1f} %), 틈 중앙 {np.median(tgt[m_]):.2f} m, < 0.5 m {np.mean(tgt[m_] < 0.5) * 100:.1f} %, "
        f"겹침 {np.mean(tgt[m_] < 0) * 100:.2f} %")
    data = OppData(D, angles, tgt, msk, meta); data.anchor_mode = "future"
    del D
    net = ClrNet(kind, data.scan.shape[-1]).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    tr = torch.as_tensor(np.nonzero(~val)[0], device=DEV); va = torch.as_tensor(np.nonzero(val)[0], device=DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps_ep = tr.numel() // BATCH; total = EPOCHS * steps_ep
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.05, final_div_factor=100.0)
    log(f"[{kind}] 매개변수 {n_par:,}, 학습 {tr.numel()}, 검증 {va.numel()}, {EPOCHS} epoch x {steps_ep}")
    sub_tr = tr[torch.randperm(tr.numel(), device=DEV)[:20000]]

    def predict(rows, bs=128):
        net.eval(); P_ = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, rows.numel(), bs):
                r = rows[i:i + bs]
                x, ego, _, _, _ = data.batch(net, r)
                P_.append(net(x, ego, data.qpoints(r)).float()[..., 0].cpu())
        net.train()
        return torch.cat(P_).numpy()

    def errs(P_, ri):
        e = np.abs(P_ - tgt[ri]); m = msk[ri]; g = tgt[ri]
        return {"med": float(np.median(e[m])), "near": float(np.median(e[m & (g < 0.5)]))}
    curves = []; t0 = time.time()
    for ep in range(EPOCHS):
        perm = tr[torch.randperm(tr.numel(), device=DEV)]; tl = []
        for i in range(steps_ep):
            r = perm[i * BATCH:(i + 1) * BATCH]
            mir = torch.rand(r.numel(), device=DEV) < 0.5
            x, ego, tg, mk, _ = data.batch(net, r, mir)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x, ego, data.qpoints(r, mir))[..., 0]
            hub = F.huber_loss(out.float(), tg, delta=0.05, reduction="none")
            loss = (hub * mk).sum() / mk.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
            tl.append(float(loss))
        row = {"epoch": ep + 1, "train_loss": float(np.mean(tl)), "s": time.time() - t0}
        row.update({f"val_{k}": v for k, v in errs(predict(va), va.cpu().numpy()).items()})
        row.update({f"trsub_{k}": v for k, v in errs(predict(sub_tr), sub_tr.cpu().numpy()).items()})
        curves.append(row)
        log(f"[{kind}] epoch {ep + 1}: 손실 {row['train_loss']:.5f} | 검증 틈 오차 중앙 {row['val_med'] * 100:.1f} cm (< 0.5 m {row['val_near'] * 100:.1f}) | "
            f"학습 일부 {row['trsub_med'] * 100:.1f} cm | {row['s']:.0f} s")
    P_ = predict(va)
    r1 = va[:1]
    x1, e1, _, _, _ = data.batch(net, r1); q1 = data.qpoints(r1)
    net.eval()
    with torch.no_grad():
        fl = flops_of(net, x1, e1, q1)
    if net.base == "polar_gru" and flops_of(net.gru, torch.zeros(1, 10, 258, device=DEV)) == 0:
        fl += 10 * 2 * 3 * (256 + 2 + 256) * 256
    ms_net = latency_ms(lambda: net(x1, e1, q1))
    ms_all = latency_ms(lambda: net(*data.batch(net, r1)[:2], data.qpoints(r1)))
    vi = val.nonzero()[0]
    np.savez_compressed(os.path.join(out_dir, f"{kind}.npz"), pred=P_, tgt=tgt[vi], msk=msk[vi], pace=meta["pace"][vi],
                        dist=meta["dist"][vi], ref_still=meta["ref_still"][vi], ref_cv=meta["ref_cv"][vi])
    json.dump(dict(kind=kind, params=n_par, flops=fl, ms_net=ms_net, ms_with_input=ms_all, curves=curves, epochs=EPOCHS,
                   batch=BATCH, lr=LR, n_train=int(tr.numel()), n_val=int(va.numel()),
                   peak_gpu_mb=torch.cuda.max_memory_allocated() / 2 ** 20),
              open(os.path.join(out_dir, f"{kind}.json"), "w"), indent=1)
    log(f"[{kind}] 끝: FLOPs {fl / 1e6:.1f} M, batch 1 {ms_net:.2f} ms (입력 포함 {ms_all:.2f} ms), GPU 최고 {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB")


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
#         box_bevgru= (c) the scan now as a grid -> a 2-D CNN -> the box_gru GRU and decoder (spatial features, latent memory)
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

    def advance(self, inc, rr):
        """the 0.1 s increment (n, 3) of each session's dead-reckoned pose; rr (n,) rows just restarted keep pose 0"""
        self.pose = torch.where(rr[:, None], self.pose, se2_compose(self.pose, inc))

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



class PointMem:
    """(11-25, net_redesign_v2 §5.3) a dead-reckoned POINT memory: M points per session kept in the car frame and moved by
    every 0.1 s increment (exact SE(2), no re-sampling, never matched against the scan), written FIFO from the newest scan
    sub-sampled 1/8 (every 8th beam's return) with no learned score. `dedup` > 0 (diagnostic variant): a new point within
    `dedup` m of a stored one refreshes that one instead of taking a FIFO slot. Read out into the same two channels as the
    grid memory (any point in the cell; recency exp(-(t - written) / 50 decisions)) so the decoder and scoring are identical"""

    def __init__(self, n, M=1024, sub=8, dedup=0.0):
        self.M, self.sub, self.dedup = M, sub, dedup
        self.p = torch.zeros(n, M, 2, device=DEV)
        self.tw = torch.full((n, M), -1e9, device=DEV)
        self.ptr = torch.zeros(n, dtype=torch.long, device=DEV)

    def reset(self, rows):
        self.tw[rows] = -1e9; self.ptr[rows] = 0

    def advance(self, inc, rr):
        c, s_ = torch.cos(inc[:, 2])[:, None], torch.sin(inc[:, 2])[:, None]
        d = self.p - inc[:, None, :2]
        new = torch.stack([c * d[..., 0] + s_ * d[..., 1], -s_ * d[..., 0] + c * d[..., 1]], -1)
        self.p = torch.where(rr[:, None, None], self.p, new)

    def write(self, X, Y, ok, t: int):
        P = torch.stack([X[:, ::self.sub], Y[:, ::self.sub]], -1); ok = ok[:, ::self.sub].clone()
        n = P.shape[0]
        rows = torch.arange(n, device=DEV)[:, None].expand(ok.shape)
        if self.dedup > 0:
            live = self.tw > -1e8
            d2 = torch.cdist(P, self.p).masked_fill(~live[:, None, :], float("inf"))
            dmin, j = d2.min(2)
            near = ok & (dmin < self.dedup)
            self.p[rows[near], j[near]] = P[near]; self.tw[rows[near], j[near]] = float(t)
            ok = ok & ~near
        rank = torch.cumsum(ok.long(), 1) - 1
        slot = (self.ptr[:, None] + rank) % self.M
        self.p[rows[ok], slot[ok]] = P[ok]; self.tw[rows[ok], slot[ok]] = float(t)
        self.ptr = (self.ptr + ok.sum(1)) % self.M

    def read(self, t: int):
        n = self.p.shape[0]
        live = (self.tw > -1e8) & (self.tw < t)
        ix = torch.floor((self.p[..., 0] - IX0) / IRES).long(); iy = torch.floor((self.p[..., 1] - IY0) / IRES).long()
        ok = live & (ix >= 0) & (ix < IH) & (iy >= 0) & (iy < IW)
        flat = (torch.arange(n, device=DEV)[:, None] * (IH * IW) + ix * IW + iy)[ok]
        hit = torch.zeros(n * IH * IW, device=DEV); hit[flat] = 1.0
        rec = torch.zeros(n * IH * IW, device=DEV).scatter_reduce_(0, flat, torch.exp(-(t - self.tw[ok]) / 50.0), "amax")
        return torch.stack([hit.view(n, IH, IW), rec.view(n, IH, IW)], 1)


def make_mem(kind, n):
    if kind in ("box_ptmem", "box_ptmemd"):
        return PointMem(n, dedup=0.15 if kind == "box_ptmemd" else 0.0)
    return SpatialMem(n) if kind in ("box_mem", "box_now") else None


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
        elif kind == "box_bevgru":
            # (c) the scan now as a grid in the car's frame -> a 2-D CNN -> the same GRU and decoder as box_gru
            self.enc = Enc2D(1, hw=(IH, IW)); self.cell = nn.GRUCell(256 + 2, 256); self.dec = VecDecoder(512)
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
    mem = make_mem(kind, n)
    out_rows, losses = [], []
    for t0 in range(0, T, CHUNK):
        ts = list(range(t0, min(t0 + CHUNK, T)))
        with torch.set_grad_enabled(train):
            if kind in ("box_gru", "box_bevgru"):
                if kind == "box_gru":
                    x = torch.stack([data.stack3(t, sess) for t in ts], 1).reshape(n * len(ts), 3, -1)   # (n L, 3, beams)
                else:
                    x = torch.stack([data.scan_now(t, sess)[:, 0] for t in ts], 1).reshape(n * len(ts), 1, IH, IW)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = torch.cat([net.enc(x[k:k + 1024].float()) for k in range(0, x.shape[0], 1024)]).view(n, len(ts), 256)
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
                            mem.advance(mem_inc(data, t, sess, noise, gen), rr)
                        cur = data.scan_now(t, sess)
                        if t % stride == 0 or not train:
                            m2 = mem.read(t) if kind != "box_now" else torch.zeros(n, 2, IH, IW, device=DEV)
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
        f"결정당 {tgt.sum() / tgt.shape[0] / B:.3f}개, 나이 중앙 {np.median(age[tgt]) if tgt.any() else float('nan'):.1f} s")
    data = BoxData(D, angles, tgt, age, neutral, rel)
    del D
    net = BoxNet(base_kind, data.scan.shape[-1]).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    tr_s = np.nonzero(~val)[0]; va_s = np.nonzero(val)[0]
    GROUP = GROUP_GRU if base_kind in ("box_gru", "box_bevgru") else GROUP_MEM
    groups_ep = math.ceil(len(tr_s) / GROUP)
    if base_kind in ("box_gru", "box_bevgru"):
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
        for k in range(0, len(sessions), GROUP_MEM):
            ss = torch.as_tensor(sessions[k:k + GROUP_MEM], device=DEV)
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
    for nz in ((0.0, 0.5, 1.0, 2.0) if base_kind in ("box_mem", "box_ptmem", "box_ptmemd") else (0.0,)):
        rows = evaluate(va_s, nz)
        sc = box_score(rows, tgt, age, neutral, rel, va_s, thr)
        res[f"n{nz:g}"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sc.items()}
        n_t = len(sc["hit_age"]) + len(sc["miss_age"])
        log(f"[{kind}] 검증 (거리계 잡음 x{nz:g}, 문턱 {thr}): 재현율 {len(sc['hit_age']) / max(n_t, 1):.3f}, "
            f"정밀도 {sc['tp'] / max(sc['tp'] + sc['fp'], 1):.3f}, 오차 중앙 {np.median(sc['err']) if len(sc['err']) else float('nan'):.3f} m")
    # cost at batch 1
    net.eval()
    if base_kind in ("box_gru", "box_bevgru"):
        x1 = (data.stack3(5, torch.as_tensor([0], device=DEV)).float() if base_kind == "box_gru"
              else data.scan_now(5, torch.as_tensor([0], device=DEV)))
        fl = (flops_of(net.enc, x1) + 2 * 3 * (256 + 2 + 256) * 256 + flops_of(net.dec, torch.zeros(1, 512, device=DEV)))
        ms = latency_ms(lambda: net.dec(torch.cat([net.cell(torch.cat([net.enc(x1), torch.zeros(1, 2, device=DEV)], 1),
                                                            torch.zeros(1, 256, device=DEV)), net.enc(x1)], 1)))
    else:
        x1 = torch.zeros(1, 3, IH, IW, device=DEV)
        fl = flops_of(net.unet, x1)
        m1 = make_mem(base_kind, 1) or SpatialMem(1); s1 = torch.as_tensor([0], device=DEV)

        def one():
            m1.advance(mem_inc(data, 5, s1, 0.0, None), torch.zeros(1, dtype=torch.bool, device=DEV))
            cur = data.scan_now(5, s1)
            X, Y, ok = data.geo.points(data.scan[5, s1][:, None], data.mount[s1])
            m1.write(X[:, 0], Y[:, 0], ok[:, 0], 5)
            return net.unet(torch.cat([m1.read(6), cur], 1))
        ms = latency_ms(one)
    json.dump(dict(kind=kind, params=n_par, flops=fl, ms=ms, thr=thr, curves=curves, res=res, epochs=BOX_EPOCHS,
                   n_train_sess=int(len(tr_s)), n_val_sess=int(len(va_s)), noise_train=noise_tr,
                   peak_gpu_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                   age_all_targets=np.percentile(age[tgt], [10, 25, 50, 75, 90]).tolist() if tgt.any() else []),
              open(os.path.join(out_dir, f"box_{kind}.json"), "w"), indent=1)
    log(f"[{kind}] 끝: FLOPs {fl / 1e6:.1f} M, batch 1 {ms:.2f} ms, GPU 최고 {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB")


# ---------------------------------------------------------------------------------------------------------------- table
OPP_NAMES = [("polar_gru", "극좌표 스캔 + GRU 10결정 (지금 policy 방식)"), ("polar_stack5", "극좌표 5장 겹침 (GRU 없음)"),
             ("polar_stack10", "극좌표 10장 겹침 (GRU 없음)"), ("polar_wide10", "극좌표 10장 겹침, 채널 4배 (계산량 대조)"),
             ("bev5", "위에서 본 격자 5장 (참 자세)"), ("bev10", "위에서 본 격자 10장 (참 자세)"),
             ("bev10_noise", "위에서 본 격자 10장 (거리계 잡음으로 학습·평가)"),
             ("pts5", "점 토큰 5장 + 시간 query (참 자세)"), ("pts10", "점 토큰 10장 + 시간 query (참 자세)"),
             ("pts5_noise", "점 토큰 5장 + 시간 query (거리계 잡음으로 학습·평가)")]
REF_NAMES = [("head", "라이브 policy 머리 (race33b u925, 같은 결정)"), ("still", "[참값] 지금 자리에 그대로 선다고 봄"),
             ("cv", "[참값] 지금 자리 + 참 속도로 직진"), ("oracle", "[참값] 상대 복사본 자신의 plan")]
BOX_NAMES = [("box_ptmem_noise", "점 기억 1024점, 새 스캔 1/8 FIFO (거리계 잡음으로 학습)"),
             ("box_ptmemd_noise", "점 기억 1024점, 0.15 m 안 점은 갱신 + FIFO (진단, 잡음으로 학습)"),
             ("box_gru", "극좌표 스캔 + GRU (세션 전체, 지금 policy 방식)"), ("box_bevgru", "지금 스캔 격자 CNN + GRU (세션 전체)"), ("box_now", "지금 스캔만 격자로 (기억 없음: 바닥선)"),
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
        r0 = R[have[0][0]]["rows"]
        have = [(k, n) for k, n in have if np.array_equal(R[k]["rows"], r0)]       # the same held-out decisions only
        J = {k: json.load(open(os.path.join(out_dir, f"opp_{k}.json"))) for k, _ in have}
        z = R[have[0][0]]
        tgt, msk, th = z["tgt"], z["msk"], z["meta_there"].astype(bool)
        pace, dist = z["meta_pace"], z["meta_dist"]
        rows = [(n, z[f"ref_{k}"], z["meta_head_p"] if k == "head" else None) for k, n in REF_NAMES]
        rows += [(n, R[k]["pred"], R[k]["logit"]) for k, n in have]
        if "bev10" in R and "pred_noisy_in" in R["bev10"].files:
            rows.append(("위에서 본 격자 10장, 참 자세로 학습 -> 잡음 자세로 평가", R["bev10"]["pred_noisy_in"], None))
        if "pts5" in R and "pred_noisy_in" in R["pts5"].files:
            rows.append(("점 토큰 5장, 참 자세로 학습 -> 잡음 자세로 평가", R["pts5"]["pred_noisy_in"], None))
        pr(f"(1) 상대 차 위치 예측 오차 [m] -- 따로 둔 세션의 '있음' 결정만 (0.5 s: {int((th & msk[:, 4]).sum())}개, 1.0 s: {int((th & msk[:, 9]).sum())}개)")
        pr(f"{'입력 / 모형':<52} | 0.1 s 중앙 / p90 | 0.5 s 중앙 / p90 | 1.0 s 중앙 / p90 | '있음' AUC")
        for n, P, sc in rows:
            e1 = q(err_at(P, tgt, msk, 0)[th]); e5 = q(err_at(P, tgt, msk, 4)[th]); e10 = q(err_at(P, tgt, msk, 9)[th])
            a = auc(np.asarray(sc, np.float64), th) if sc is not None else float("nan")
            pr(f"{n:<52} | {e1[0]:5.2f} / {e1[1]:5.2f}    | {e5[0]:5.2f} / {e5[1]:5.2f}    | {e10[0]:5.2f} / {e10[1]:5.2f}    | "
               + (f"{a:.3f}" if np.isfinite(a) else "  -"))
        pr("")
        pr("움직임만: 0.1 s 점에서 1.0 s 점까지 예측한 이동(벡터)과 실제 이동의 차 [m] (지금 위치를 잘못 잡은 만큼은 빠짐), 같은 결정")
        mm = th & msk[:, 0] & msk[:, 9]
        dT = tgt[:, 9] - tgt[:, 0]
        pr(f"{'':<52} | 전체 중앙 / p90 | 페이스<0.7 | 0.7-0.9 | ≥0.9 (중앙)  | 실제 이동 중앙 {np.median(np.linalg.norm(dT[mm], axis=-1)):.2f} m")
        for n, P, _ in rows:
            em = np.linalg.norm((P[:, 9] - P[:, 0]) - dT, axis=-1)
            a_ = q(em[mm])
            pc = [np.median(em[mm & m]) for m in (pace < 0.7, (pace >= 0.7) & (pace < 0.9), pace >= 0.9)]
            pr(f"{n:<52} | {a_[0]:5.2f} / {a_[1]:5.2f}  | " + " | ".join(f"{x:5.2f}" for x in pc))
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
    CLR_NAMES = [("clr_polar_gru", "극좌표 스캔 + GRU (지금 방식; 묶은 벡터 + 점 좌표)"), ("clr_bev10", "격자 10장, 묶은 벡터 + 점 좌표"),
                 ("clr_bev10b", "격자 10장, plan 점에서 특징 지도 쌍선형 읽기"), ("clr_pts5", "점 토큰 5장, plan 점에 query")]
    hc = [(k, n) for k, n in CLR_NAMES if os.path.exists(os.path.join(out_dir, f"{k}.npz"))]
    if hc:
        pr("")
        pr("(3) plan 점의 빈 공간 정밀도 -- 이 결정의 plan 10점에서 가장 가까운 상자 모서리 / 벽까지의 부호 있는 틈 [cm], 따로 둔 상자 배치의 세션")
        pr(f"{'모형':<46} | 상자 중앙/p90 | 상자 <0.3m 중앙/p90 | 상자(지금 보임) 중앙 | 벽 중앙/p90 | 벽 <0.3m 중앙/p90 | MFLOP | ms 망/입력 포함")
        for k, n in hc:
            z = np.load(os.path.join(out_dir, f"{k}.npz")); j = json.load(open(os.path.join(out_dir, f"{k}.json")))
            e = np.abs(z["pred"] - z["tgt"]) * 100; g = z["tgt"]
            def mp(m_, jj):
                v = e[..., jj][m_]
                return f"{np.median(v):5.1f} / {np.percentile(v, 90):5.1f}" if v.size else "   -  "
            cl = g[..., 0] < CLR_CLIP[1]
            pr(f"{n:<46} | {mp(cl, 0)} | {mp(g[..., 0] < 0.3, 0):>17} | {np.median(e[..., 0][cl & z['vis']]):8.1f}           | "
               f"{mp(g[..., 1] < CLR_CLIP[1], 1)} | {mp(g[..., 1] < 0.3, 1):>15} | {j['flops'] / 1e6:6.1f} | {j['ms_net']:.2f} / {j['ms_with_input']:.2f}")
        z = np.load(os.path.join(out_dir, f"{hc[0][0]}.npz")); g = z["tgt"]
        pr(f"(점 수: 상자 틈 < 2 m {int((g[..., 0] < CLR_CLIP[1]).sum())}, < 0.3 m {int((g[..., 0] < 0.3).sum())}; 벽 < 0.3 m {int((g[..., 1] < 0.3).sum())}. "
           f"plan 점과 실제로 그 시각 차가 있던 자리의 차 중앙: k=1 {np.median(z['real'][:, 0]) * 100:.1f} cm, k=5 {np.median(z['real'][:, 4]) * 100:.1f} cm, "
           f"k=10 {np.median(z['real'][:, 9]) * 100:.1f} cm -- 매 결정 다시 그리므로 k ≥ 2는 실제 자리와 다름: 타깃은 plan 점 자체)")
    CAR_NAMES = [("car_polar_gru", "극좌표 스캔 + GRU (지금 방식; 묶은 벡터 + 자세)"), ("car_bev10b", "격자 10장, 우리 자세에서 쌍선형 읽기"),
                 ("car_pts5", "점 토큰 5장, 우리 자세에 query")]
    hk = [(k, n) for k, n in CAR_NAMES if os.path.exists(os.path.join(out_dir, f"{k}.npz"))]
    if hk:
        z = np.load(os.path.join(out_dir, f"{hk[0][0]}.npz")); g, m = z["tgt"], z["msk"]; pc = z["pace"][:, None].repeat(10, 1)
        rows_k = [("[참값] 상대가 지금 자리에 그대로", z["ref_still"]), ("[참값] 상대 지금 자리 + 참 속도로 직진", z["ref_cv"])]
        rows_k += [(n, np.load(os.path.join(out_dir, f"{k}.npz"))["pred"]) for k, n in hk]
        pr("")
        pr(f"(4) 움직이는 차와의 시간 맞춘 빈틈 -- 0.1 k s 뒤 우리 차(그 시각 실제 자세)와 같은 시각 상대 차 사이의 부호 있는 틈 [cm] "
           f"(두 차 0.58 x 0.31 m 상자, 정확한 상자-상자 거리, [-1, 3] m로 자름), 따로 둔 세션의 '있음' 결정, k = 1..10 모두")
        mu = m & (g < CAR_CLIP[1])                     # not clipped: the other car within 3 m of our body at t + 0.1 k
        pr(f"(점 수 {int(m.sum())}, 그중 3 m 안(잘리지 않음) {int(mu.sum())}, 틈 < 0.5 m {int((m & (g < 0.5)).sum())}, 겹침 {int((m & (g < 0)).sum())}; "
           f"< 0.5 m 점 수 페이스별 " + ", ".join(f"{int((m & (g < 0.5) & (pc >= lo) & (pc < hi)).sum())}" for lo, hi in ((0, 0.7), (0.7, 0.9), (0.9, 9))) + ")")
        pr(f"{'모형':<46} | 3 m 안 중앙/p90 | <0.5m 중앙/p90 | 페이스<0.7 | 0.7-0.9 | ≥0.9 (3 m 안 중앙; <0.5m 중앙) | <0.5m 부호 맞음 | 겹침 맞힘 | 헛겹침")
        for n, P_ in rows_k:
            e = np.abs(P_ - g) * 100
            def mq(mm):
                v = e[mm]
                return f"{np.median(v):5.1f} / {np.percentile(v, 90):5.1f}" if v.size else "   -   "
            near = m & (g < 0.5)
            pcs = []
            for lo, hi in ((0, 0.7), (0.7, 0.9), (0.9, 9)):
                mm = mu & (pc >= lo) & (pc < hi); mn = near & (pc >= lo) & (pc < hi)
                pcs.append(f"{np.median(e[mm]) if mm.any() else float('nan'):5.1f};{np.median(e[mn]) if mn.any() else float('nan'):5.1f}")
            sg = np.mean((P_[near] < 0) == (g[near] < 0)) * 100
            ov = near & (g < 0); cl = near & (g >= 0)
            pr(f"{n:<46} | {mq(mu)} | {mq(near):>14} | " + " | ".join(pcs) + f" | {sg:5.1f} % | {np.mean(P_[ov] < 0) * 100:5.1f} % | {np.mean(P_[cl] < 0) * 100:5.1f} %")
        pr(f"{'모형':<46} | 매개변수 | MFLOP | ms 망 / 입력 포함")
        for k, n in hk:
            j = json.load(open(os.path.join(out_dir, f"{k}.json")))
            pr(f"{n:<46} | {j['params'] / 1e6:5.2f} M | {j['flops'] / 1e6:6.1f} | {j['ms_net']:.2f} / {j['ms_with_input']:.2f}")
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
                    cells.append(f"{h_ / (h_ + m_):7.2f}" if h_ + m_ else f"{'-':>7}")
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
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("REPR_GPU_FRAC", "0.22")))
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if mode == "opp":
        os.makedirs(sys.argv[3], exist_ok=True)
        run_opp(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif mode == "box":
        os.makedirs(sys.argv[3], exist_ok=True)
        run_box(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif mode == "clr":
        os.makedirs(sys.argv[3], exist_ok=True)
        run_clr(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif mode == "car":
        os.makedirs(sys.argv[3], exist_ok=True)
        run_car(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif mode == "table":
        table(sys.argv[2])
    else:
        raise SystemExit(f"unknown mode {mode}")
