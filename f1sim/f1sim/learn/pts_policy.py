"""(11-25) The point-token policy (plan_rl --arch pts; net_redesign_v2.md section 3 A/C and section 4, approved by the user).

What it reads -- the same sensors as `plan_policy.Actor2` (v2), no map, no line, no lap position, no scan matching:
  * the last K = 5 LiDAR scans (0 .. 0.4 s old), every return turned into a point (x, y) in the car's frame NOW through the
    chain of dead-reckoned SE(2) increments of the car's own odometry (`GymEnv.sim.odom`, the VESC's wheel speed and the gyro --
    what the real car's `vesc_to_odom` gives), with its age and range. No raster. Training adds the probe's odometry noise
    (ODOM_SIG: 2 cm and 0.5 deg per 0.1 s random walk) on top of the odometry model's own errors.
  * the v2 state (IMU, the last commands, the last plan in its own frame) and the last odometry increment.
  * the previous plan: its point k + 1 moved into the frame now is the anchor a_k of plan query k (t_k = 0.1 k s).

The network (per decision):
  points -> MLP(Fourier(x, y) at 0.05 / 0.2 / 1 / 5 m, age, range) -> max-pool over runs of consecutive beams (8 in the newest
  frame, 16 in older ones), each token keeping its run's mean position: N = 136 + 4 x 68 = 408 tokens of width 96, padded
  to a fixed N with a mask.
  11 queries = 10 plan queries (learned slot + time code of t_k + Fourier(a_k) + the anchor's heading / speed / the anchor
  action) + 1 ego query (normalised state, last increment). Block 1 = [query -> token cross-attention with the RELATIVE
  position Fourier(token - a_k) and the token's age in keys and values, self-attention among the 11 queries, FFN], width 128,
  4 heads, pre-LN, no token-token attention. The relative terms are computed exactly but factorised (`RelCross`): a
  Fourier feature of a difference is a rotation of the two Fourier features, so no (n, Q, N, d) tensor is ever built.
  A GRU (128) reads the ego query's output after block 1; its state is added to all queries (the place code).
  The free-space head's answer (detached, see below) is added to each plan query. Block 2, then each plan query gives its
  (turn, speed) share of the 20 plan numbers, decoded by `plan_steps.roll` as `Actor2`'s; the speed estimate and the
  exploration noise as `Actor2`'s.
  The plan output is a RESIDUAL on the anchor action (the previous mean, one step on, the last step neutral): a zero-initialised
  head drives the anchor -- the previous plan continued. Without the neutral last step the shift would integrate the last
  step's output over the whole episode.

The free-space head (`FreeHead`, its own front end and query blocks, its own optimizer in plan_rl, nothing of the policy's loss
reaches it): for each plan query, the signed clearance [m] between our car's footprint at the anchor pose at t_k and the
nearest knowable obstacle (walls; boxes seen this episode; the other car where it really is at t + t_k, if 'there' at t),
clipped to [-1, 3], and a Laplace scale (its confidence). The policy reads (clearance / 3, log scale / 3) detached, added
to its plan query before block 2: there the queries' self-attention lets a blocked point k move the others and the FFN reads
it nonlinearly (fed to the plan head alone it could only shift its own point's output linearly)."""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import plan_steps as PS
from .plan_policy import RunningNorm, PlanObs2, STATE_DIM2, PT_IDX, Actor2

ARCH = "pts"
K_FRAMES = 5
RUN_NEW, RUN_OLD = 8, 16
FOURIER_S = (0.05, 0.2, 1.0, 5.0)              # [m]
TIME_S = (0.25, 0.5, 1.0, 2.0)                 # [s] the time code's periods
D_PT, D_TOK, D_Q, HEADS = 64, 96, 128, 4
N_PQ = PS.N_STEP                               # plan queries (the ego query is the last one)
ODOM_SIG = (0.02, 0.02, math.radians(0.5))     # [m, m, rad] per 0.1 s: the probes' odometry noise (x1)
INC_SCALE = (1.0, 0.1, 0.3)                    # the last increment as the ego query reads it
FS_LO, FS_HI, FS_SCALE = -1.0, 3.0, 3.0        # [m] the free-space target's clip; the input's scale
FS_LOGB = (-5.0, 1.0)                          # the head's log Laplace scale (log m)
FS_ORACLE_LOGB = -5.0                          # --free-oracle: 'certain'
FS_NEAR = 0.5                                  # [m] the monitored band
FS_THERE_RANGE = 10.0                          # [m] the other car counts within this (and seen by a beam within 1 s)


# ------------------------------------------------------------------------------------------------------------- geometry
def se2_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """poses (..., 3): a o b (b expressed in a's frame -> in a's parent)"""
    c, s = torch.cos(a[..., 2]), torch.sin(a[..., 2])
    return torch.stack([a[..., 0] + c * b[..., 0] - s * b[..., 1], a[..., 1] + s * b[..., 0] + c * b[..., 1], a[..., 2] + b[..., 2]], -1)


def se2_rel(ref: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """pose p (..., 3) in the frame of pose ref (..., 3)"""
    c, s = torch.cos(ref[..., 2]), torch.sin(ref[..., 2])
    dx, dy = p[..., 0] - ref[..., 0], p[..., 1] - ref[..., 1]
    return torch.stack([c * dx + s * dy, -s * dx + c * dy, p[..., 2] - ref[..., 2]], -1)


_CONST = {}


def _const(key, values, like: torch.Tensor) -> torch.Tensor:
    """a small constant tensor on `like`'s device / dtype, made once (a `torch.tensor(list, device=cuda)` per call is a pageable
    host copy: a device sync every time, and not allowed inside a CUDA graph capture)"""
    k = (key, like.device, like.dtype)
    t = _CONST.get(k)
    if t is None:
        t = _CONST[k] = torch.tensor(values, device=like.device, dtype=like.dtype)
    return t


def fourier(xy: torch.Tensor) -> torch.Tensor:
    """(..., 2) [m] -> (..., 16): per scale s (FOURIER_S) sin x', sin y', cos x', cos y' of (x, y) 2 pi / s -- the probe's order"""
    sc = _const("fourier", [2.0 * math.pi / v for v in FOURIER_S], xy)
    a = xy[..., None, :] * sc[:, None]                                       # (..., 4, 2)
    return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)


def time_code() -> torch.Tensor:
    """(N_PQ, 8): sin / cos of t_k 2 pi / T for T in TIME_S, t_k = 0.1 k s"""
    t = torch.arange(1, N_PQ + 1, dtype=torch.float32)[:, None] * PS.DT
    w = torch.tensor([2.0 * math.pi / v for v in TIME_S])[None]
    return torch.cat([torch.sin(t * w), torch.cos(t * w)], 1)


def pad_beams(beams: int) -> int:
    """beams padded to whole runs of both lengths (1081 -> 1088)"""
    L = RUN_OLD if RUN_OLD % RUN_NEW == 0 else RUN_NEW * RUN_OLD
    return int(math.ceil(beams / L) * L)


def n_tokens(beams: int, K: int = K_FRAMES) -> int:
    P = pad_beams(beams)
    return P // RUN_NEW + (K - 1) * (P // RUN_OLD)


class PtsIn(NamedTuple):
    """one decision's inputs besides the state: frames (n, K, beams) ranges / range_max in [0, 1] (newest first); rel (n, K, 3)
    each frame's pose in the frame now; fok (n, K) frame there; anc (n, 10, 3) the anchors (x, y, heading) in the frame now;
    anc_v (n, 10) their speeds; anc_ok (n,) a previous plan exists; u_anc (n, 20) the anchor action (pre-tanh)"""
    frames: torch.Tensor
    rel: torch.Tensor
    fok: torch.Tensor
    anc: torch.Tensor
    anc_v: torch.Tensor
    anc_ok: torch.Tensor
    u_anc: torch.Tensor


class PtsGeom(nn.Module):
    """the beams' directions, the scanner's NOMINAL mount (x, y, yaw in the car's frame) and its range: a scan -> points. The
    simulator's per-car mount error is not known to the policy (domain randomisation), as on the real car"""

    def __init__(self, angles: torch.Tensor, mount=(0.297, 0.0, 0.0), rmax: float = 10.0):
        super().__init__()
        self.register_buffer("angles", torch.as_tensor(angles, dtype=torch.float32).clone())
        self.register_buffer("mount", torch.tensor([float(m) for m in mount], dtype=torch.float32))
        self.register_buffer("rmax", torch.tensor(float(rmax)))
        self.beams = int(self.angles.numel())
        self.P = pad_beams(self.beams)

    def points(self, frames: torch.Tensor, rel: torch.Tensor, fok: torch.Tensor):
        """frames (n, K, beams), rel (n, K, 3), fok (n, K) -> X, Y [m] in the frame now, r (range / max), ok (n, K, P)"""
        sc = frames.float()
        r = sc * self.rmax
        a = self.angles + self.mount[2]
        px = self.mount[0] + r * torch.cos(a)
        py = self.mount[1] + r * torch.sin(a)
        c, s_ = torch.cos(rel[..., 2])[..., None], torch.sin(rel[..., 2])[..., None]
        X = rel[..., 0:1] + c * px - s_ * py
        Y = rel[..., 1:2] + s_ * px + c * py
        ok = (sc < 0.999) & fok[..., None]
        pad = self.P - self.beams
        return (F.pad(X, (0, pad)), F.pad(Y, (0, pad)), F.pad(sc, (0, pad), value=1.0), F.pad(ok, (0, pad), value=False))


class PtsFront(nn.Module):
    """points -> tokens (the probe's `PtsNet.tokens`): (n, N, D_TOK), their mean positions (n, N, 2), ages (n, N), valid (n, N)"""

    def __init__(self, K: int = K_FRAMES):
        super().__init__()
        self.K = int(K)
        self.pmlp = nn.Sequential(nn.Linear(18, D_PT), nn.ReLU(), nn.Linear(D_PT, D_PT))
        self.tok = nn.Sequential(nn.Linear(D_PT + 16 + 1, D_TOK), nn.ReLU(), nn.Linear(D_TOK, D_TOK))

    def forward(self, X, Y, r, ok):
        """frame by frame (the per-point layer is the memory: 1.4 MB a decision for all five frames at once), the tokens in the
        probe's order: the newest frame's runs, then each older frame's"""
        n, K, P = X.shape
        hs, ps, ags, vals = [], [], [], []
        for j in range(K):
            L = RUN_NEW if j == 0 else RUN_OLD
            xy = torch.stack([X[:, j], Y[:, j]], -1)                                         # (n, P, 2)
            a = torch.full_like(X[:, j], j * PS.DT)
            h = self.pmlp(torch.cat([fourier(xy), a[..., None], r[:, j, :, None]], -1))     # (n, P, D_PT)
            h = h.masked_fill(~ok[:, j, :, None], float("-inf"))
            okf = ok[:, j].to(X.dtype)
            hs.append(h.reshape(n, P // L, L, -1).amax(2))
            cnt = okf.reshape(n, P // L, L).sum(2)
            ps.append((xy * okf[..., None]).reshape(n, P // L, L, 2).sum(2) / cnt.clamp_min(1.0)[..., None])
            ags.append(a[:, ::L])
            vals.append(cnt > 0)
        hh, pos, ag, val = torch.cat(hs, 1), torch.cat(ps, 1), torch.cat(ags, 1), torch.cat(vals, 1)
        hh = torch.where(val[..., None], hh, torch.zeros_like(hh))
        val = val.clone()
        val[:, 0] = val[:, 0] | ~val.any(1)                                                  # never an all-masked row
        tok = self.tok(torch.cat([hh, fourier(pos), ag[..., None]], -1))
        return tok, pos, ag, val


def _rot_terms(fa: torch.Tensor):
    """fourier(anchor) (..., 16) -> its sines / cosines (..., 4, 2) each"""
    f = fa.unflatten(-1, (len(FOURIER_S), 4))
    return f[..., 0:2], f[..., 2:4]


class RelCross(nn.Module):
    """query -> token cross-attention whose keys and values carry the RELATIVE position Fourier(token - anchor) and the token's
    age: k_qj = Wk t_j + Rk [Fourier(p_j - a_q), age_j], v_qj = Wv t_j + Rv [...] -- the probe's `PtsBlock` -- computed without
    the (n, Q, N, d) tensor: sin w(p - a) = sin wp cos wa - cos wp sin wa, cos w(p - a) = cos wp cos wa + sin wp sin wa, so the
    relative logit is the query side's rotated Rk^T q against Fourier(p_j), and the relative value Rv applied to the attention-
    weighted Fourier(p) rotated back by the anchor (the biases of Wk and Rk add the same number to every token's logit and
    cancel in the softmax). The same numbers as the direct form (`rel_cross_naive`; tests/test_pts_arch.py)."""

    def __init__(self, d: int = D_Q, d_tok: int = D_TOK, H: int = HEADS):
        super().__init__()
        self.H = int(H)
        self.ln_q, self.ln_t = nn.LayerNorm(d), nn.LayerNorm(d_tok)
        self.wq, self.wo = nn.Linear(d, d), nn.Linear(d, d)
        self.wk, self.wv = nn.Linear(d_tok, d), nn.Linear(d_tok, d)
        self.rk, self.rv = nn.Linear(17, d), nn.Linear(17, d)

    def forward(self, q, anc, tok, pos, age, valid):
        """q (n, Q, d), anc (n, Q, 2), tok (n, N, d_tok), pos (n, N, 2), age (n, N), valid (n, N) -> the attention's output (n, Q, d)"""
        n, Q, d = q.shape
        N = tok.shape[1]; H = self.H; dh = d // H
        t = self.ln_t(tok)
        qq = self.wq(self.ln_q(q)).view(n, Q, H, dh)
        k = self.wk(t).view(n, N, H, dh)
        v = self.wv(t).view(n, N, H, dh)
        fp = fourier(pos)                                                     # (n, N, 16)
        sa, ca = _rot_terms(fourier(anc))                                     # (n, Q, 4, 2)
        g = torch.einsum("nqhd,hdm->nhqm", qq, self.rk.weight.view(H, dh, 17))   # (n, H, Q, 17)
        gS, gC = _rot_terms(g[..., :16])                                      # (n, H, Q, 4, 2)
        sa_, ca_ = sa[:, None], ca[:, None]
        psi = torch.cat([gS * ca_ + gC * sa_, gC * ca_ - gS * sa_], -1).flatten(-2)   # coefficients of sin wp / cos wp
        lg = torch.einsum("nqhd,njhd->nhqj", qq, k) + torch.einsum("nhqm,njm->nhqj", psi, fp) + g[..., 16:17] * age[:, None, None, :]
        lg = lg / math.sqrt(dh)
        lg = lg.masked_fill(~valid[:, None, None, :], float("-inf"))
        att = torch.softmax(lg, -1)
        o = torch.einsum("nhqj,njhd->nqhd", att, v)
        Sp = torch.einsum("nhqj,njm->nhqm", att, fp)                          # sum_j att [sin wp, cos wp]
        Ss, Sc = _rot_terms(Sp)
        rel_f = torch.cat([Ss * ca_ - Sc * sa_, Sc * ca_ + Ss * sa_], -1).flatten(-2)   # sum_j att Fourier(p_j - a)
        rel = torch.cat([rel_f, torch.einsum("nhqj,nj->nhq", att, age)[..., None]], -1)   # (n, H, Q, 17)
        o = o + torch.einsum("nhqm,hdm->nqhd", rel, self.rv.weight.view(H, dh, 17)) + self.rv.bias.view(H, dh)
        return self.wo(o.reshape(n, Q, d))


def rel_cross_naive(m: RelCross, q, anc, tok, pos, age, valid):
    """the probe's direct form of `RelCross` (tests): keys / values built per (query, token) pair"""
    n, Q, d = q.shape
    N = tok.shape[1]; H = m.H; dh = d // H
    t = m.ln_t(tok)
    rel = torch.cat([fourier(pos[:, None] - anc[:, :, None]), age[:, None, :, None].expand(n, Q, N, 1)], -1)
    qq = m.wq(m.ln_q(q)).view(n, Q, H, dh)
    k = (m.wk(t)[:, None] + m.rk(rel)).view(n, Q, N, H, dh)
    v = (m.wv(t)[:, None] + m.rv(rel)).view(n, Q, N, H, dh)
    lg = torch.einsum("nqhd,nqjhd->nhqj", qq, k) / math.sqrt(dh)
    lg = lg.masked_fill(~valid[:, None, None, :], float("-inf"))
    att = torch.softmax(lg, -1)
    o = torch.einsum("nhqj,nqjhd->nqhd", att, v).reshape(n, Q, d)
    return m.wo(o)


class PtsBlock(nn.Module):
    """[cross-attention to the tokens, self-attention among the queries, FFN], pre-LN, residual"""

    def __init__(self, d: int = D_Q, d_tok: int = D_TOK, H: int = HEADS):
        super().__init__()
        self.cross = RelCross(d, d_tok, H)
        self.ln_s = nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, H, batch_first=True)
        self.ln_f = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, anc, tok, pos, age, valid):
        q = q + self.cross(q, anc, tok, pos, age, valid)
        h = self.ln_s(q)
        q = q + self.sa(h, h, h, need_weights=False)[0]
        return q + self.ffn(self.ln_f(q))


class PtsCore(nn.Module):
    """tokens + the 11 queries + two blocks (the actor's and the free-space head's are two of these, each its own weights)"""

    def __init__(self, K: int = K_FRAMES, state_dim: int = STATE_DIM2):
        super().__init__()
        self.front = PtsFront(K)
        self.q0 = nn.Parameter(torch.randn(N_PQ + 1, D_Q) * 0.02)          # the query's slot (its t_k for a plan query)
        self.q_time = nn.Linear(8, D_Q)
        self.q_anc = nn.Linear(16, D_Q)
        self.q_ext = nn.Linear(6, D_Q)                                       # the anchor's heading, speed, there, its action
        self.q_ego = nn.Linear(state_dim + 4, D_Q)                           # normalised state, last increment, its flag
        self.blk1, self.blk2 = PtsBlock(), PtsBlock()
        self.ln = nn.LayerNorm(D_Q)
        self.register_buffer("tcode", time_code(), persistent=False)
        self.register_buffer("inc_scale", torch.tensor(INC_SCALE), persistent=False)

    def stage1(self, geom: PtsGeom, P: PtsIn, s_n: torch.Tensor):
        """-> (queries after block 1 (n, 11, D_Q), the context block 2 reads: tokens, positions, ages, valid, anchors (n, 11, 2))"""
        X, Y, r, ok = geom.points(P.frames, P.rel, P.fok)
        tok, pos, age, val = self.front(X, Y, r, ok)
        n = s_n.shape[0]
        anc = P.anc.float()
        u = torch.tanh(P.u_anc.float())
        okf = P.anc_ok.float()[:, None].expand(n, N_PQ)
        ext = torch.stack([torch.cos(anc[..., 2]), torch.sin(anc[..., 2]), P.anc_v.float() / 5.0, okf, u[:, :N_PQ], u[:, N_PQ:]], -1)
        qp = self.q_time(self.tcode)[None] + self.q_anc(fourier(anc[..., :2])) + self.q_ext(ext)
        inc = P.rel[:, 1].float() / self.inc_scale
        qe = self.q_ego(torch.cat([s_n, inc, P.fok[:, 1:2].float()], 1))
        q = self.q0[None] + torch.cat([qp, qe[:, None]], 1)
        a2 = torch.cat([anc[..., :2], torch.zeros(n, 1, 2, device=anc.device, dtype=anc.dtype)], 1)
        q = self.blk1(q, a2, tok, pos, age, val)
        return q, (tok, pos, age, val, a2)

    def stage2(self, q, ctx):
        tok, pos, age, val, a2 = ctx
        return self.ln(self.blk2(q, a2, tok, pos, age, val))


class PtsPlanHead(nn.Module):
    """`Actor2.pi`'s place: the features `fc` = (the 11 final queries flattened, the anchor action (20)) -> the plan's 20 numbers
    (pre-tanh; [turn 1..10, speed 1..10]): each plan query its own (turn, speed), one shared linear map, zero at the start; +
    the anchor action (`residual`)"""

    def __init__(self, residual: bool = True):
        super().__init__()
        self.residual = bool(residual)
        self.lin = nn.Linear(D_Q, 2)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, fc: torch.Tensor) -> torch.Tensor:
        n = fc.shape[0]
        q = fc[:, :(N_PQ + 1) * D_Q].reshape(n, N_PQ + 1, D_Q)
        o = self.lin(q[:, :N_PQ])
        u = torch.cat([o[..., 0], o[..., 1]], 1)
        return u + fc[:, (N_PQ + 1) * D_Q:] if self.residual else u


class FreeHead(nn.Module):
    """the free-space head: its own `PtsCore` -> per plan query (clearance [m], log Laplace scale)"""

    def __init__(self, K: int = K_FRAMES, state_dim: int = STATE_DIM2):
        super().__init__()
        self.core = PtsCore(K, state_dim)
        self.out = nn.Linear(D_Q, 2)
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.copy_(torch.tensor([1.0, math.log(0.3)]))

    def forward(self, geom: PtsGeom, P: PtsIn, s_n: torch.Tensor):
        q1, ctx = self.core.stage1(geom, P, s_n.detach())
        o = self.out(self.core.stage2(q1, ctx)[:, :N_PQ])
        return o[..., 0], o[..., 1].clamp(*FS_LOGB)


def fs_feed(m: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """what the policy reads of the free-space head, detached: (n, 10, 2) = (clearance / 3, log scale / 3)"""
    return torch.stack([m.detach().clamp(FS_LO, FS_HI) / FS_SCALE, log_b.detach() / FS_SCALE], -1)


def fs_oracle_feed(c: torch.Tensor) -> torch.Tensor:
    """--free-oracle: the true clearance (n, 10) [m] in the head's place, 'certain'"""
    return torch.stack([c.clamp(FS_LO, FS_HI) / FS_SCALE, torch.full_like(c, FS_ORACLE_LOGB / FS_SCALE)], -1)


def free_loss(m, log_b, y, msk):
    """the head's loss on the points that count (`msk`): Huber on the clearance (beta 5 cm), and the Laplace scale's negative
    log-likelihood on the clearance's detached error (so the scale never slows the clearance's own learning)"""
    mf = msk.float()
    n = mf.sum().clamp_min(1.0)
    hub = F.smooth_l1_loss(m, y, beta=0.05, reduction="none")
    nll = (m.detach() - y).abs() / log_b.exp() + log_b
    return ((hub + 0.1 * nll) * mf).sum() / n


@torch.no_grad()
def free_stats(m, y, msk) -> dict:
    """(sums for the log) |error| median overall / where the true gap < FS_NEAR, the sign right there, counts"""
    e = (m - y).abs()[msk]
    yy = y[msk]
    near = yy < FS_NEAR
    out = {"n": int(msk.sum()), "n_near": int(near.sum())}
    out["med"] = float(e.median()) if e.numel() else None
    out["med_near"] = float(e[near].median()) if bool(near.any()) else None
    out["sign_near"] = float(((m[msk][near] < 0) == (yy[near] < 0)).float().mean()) if bool(near.any()) else None
    return out


def free_head_step(actor, opt, P: PtsIn, s_n: torch.Tensor, y: torch.Tensor, msk: torch.Tensor, reduce_grads=None,
                   max_grad: float = 0.5, run=None):
    """one step of the free-space head's own optimizer: its loss on (P, s_n) against the targets y (n, 10) where msk; the
    gradient handed to `reduce_grads(params)` (plan_rl's all-reduce: every process calls this once per minibatch, with or
    without a target in it); `run(P, s_n)` the head's forward (plan_rl: in recomputed pieces). -> (loss, stats)"""
    params = actor.free_parameters()
    m, lb = (run(P, s_n) if run is not None else actor.free(actor.geom, P, s_n))
    loss = free_loss(m, lb, y, msk)
    gs = torch.autograd.grad(loss, params, allow_unused=True)
    for p_, g_ in zip(params, gs):
        p_.grad = g_ if g_ is not None else torch.zeros_like(p_)
    if reduce_grads is not None:
        reduce_grads(params)
    nn.utils.clip_grad_norm_(params, max_grad)
    opt.step()
    for p_ in params:
        p_.grad = None
    return float(loss.detach()), free_stats(m.detach(), y, msk)


def shift_anchor(u: torch.Tensor) -> torch.Tensor:
    """the anchor action: the previous plan's mean (n, 20) one step on -- steps 2..10 -> 1..9 -- and the last step neutral (0:
    the middle of what is reachable; repeating the last step would integrate it over the episode)"""
    N = PS.N_STEP
    z = torch.zeros_like(u[:, :1])
    return torch.cat([u[:, 1:N], z, u[:, N + 1:], z], 1)


class PtsActor(nn.Module):
    """plan_rl --arch pts: `Actor2`'s interface where plan_rl / the runners read it (memory, heads, pi, v_est, log_std,
    log_std_floor, h_out, mem_cell, memory_alone, vmem_read, policy_parameters); `forward(P, state, h, fs)`."""

    def __init__(self, angles, mount=(0.297, 0.0, 0.0), rmax: float = 10.0, state_dim: int = STATE_DIM2, memory: int = 128,
                 std0: float = 0.6, residual: bool = True, mem_value: bool = False, mem_keep_s: float = 0.0,
                 mem_keep_log: bool = False, K: int = K_FRAMES):
        super().__init__()
        self.arch, self.K = ARCH, int(K)
        self.geom = PtsGeom(angles, mount, rmax)
        self.beams = self.geom.beams
        self.norm = RunningNorm(state_dim)
        self.core = PtsCore(K, state_dim)
        self.memory = self.memory_main = int(memory)
        if self.memory <= 0:
            raise ValueError("the pts policy keeps its GRU: memory > 0")
        self.gru = nn.GRUCell(D_Q, self.memory)
        self.h_in = nn.Linear(self.memory, D_Q)
        self.fs_in = nn.Linear(2, D_Q)
        nn.init.zeros_(self.fs_in.weight); nn.init.zeros_(self.fs_in.bias)
        self.pi = PtsPlanHead(residual)
        self.v_est = nn.Linear(D_Q, 1)
        self.log_std = nn.Parameter(torch.full((PS.ACT_DIM,), math.log(std0)))
        self.register_buffer("log_std_floor", torch.full((PS.ACT_DIM,), -math.inf), persistent=False)
        self.free = FreeHead(K, state_dim)
        if mem_value:
            self.v_mem = nn.Linear(self.memory, 1)
        if mem_keep_s > 0:
            Actor2.keep_init(self, float(mem_keep_s), bool(mem_keep_log))
        # what plan_rl / the runners ask an Actor2 for
        self.box_memory, self.mem_front, self.branches, self.pace_levels = 0, True, 1, 0
        self.pred_opp = self.pred_clear = self.aux_in = False
        self.aux = self.pred = self.pred_out = self.pc_out = None
        self.h_out = self.fs_out = self.free_out = None

    # -- the pieces plan_rl's update calls one by one
    def stage1(self, P: PtsIn, state: torch.Tensor):
        """-> (queries after block 1, context, the normalised state): the GRU's input is the ego query, `q1[:, -1]`"""
        s_n = self.norm(state)
        q1, ctx = self.core.stage1(self.geom, P, s_n)
        return q1, ctx, s_n

    def mem_cell(self, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.gru(f, h)

    def trunk(self, q1, ctx, h1, fs, u_anc) -> torch.Tensor:
        """the GRU's state added to every query, the free-space input to each plan query, block 2 -> `fc` (n, 11 D_Q + 20)"""
        n = q1.shape[0]
        q = q1 + self.h_in(h1)[:, None]
        q = q + torch.cat([self.fs_in(fs.to(q.dtype)), torch.zeros(n, 1, D_Q, device=q.device, dtype=q.dtype)], 1)
        q = self.core.stage2(q, ctx)
        return torch.cat([q.reshape(n, -1), u_anc.to(q.dtype)], 1)

    def heads(self, fc: torch.Tensor):
        log_std = torch.maximum(self.log_std, self.log_std_floor)
        v = self.v_est(fc[:, N_PQ * D_Q:(N_PQ + 1) * D_Q])[:, 0] * 5.0
        return self.pi(fc), log_std.expand(fc.shape[0], -1), v

    @torch.no_grad()
    def free_feed(self, P: PtsIn, s_n: torch.Tensor) -> torch.Tensor:
        m, lb = self.free(self.geom, P, s_n)
        self.free_out = (m, lb)
        return fs_feed(m, lb)

    def forward(self, P: PtsIn, state: torch.Tensor, h: Optional[torch.Tensor] = None, fs: Optional[torch.Tensor] = None):
        """one decision: (mean (n, 20), log_std, speed estimate (n,)); `h_out` the GRU's new state, `fs_out` the free-space input
        read (`fs` None: the head's own answer), `free_out` the head's raw (clearance, log scale) when it ran"""
        self.free_out = None
        q1, ctx, s_n = self.stage1(P, state)
        h0 = h if h is not None else torch.zeros(q1.shape[0], self.memory, device=q1.device, dtype=q1.dtype)
        h1 = self.gru(q1[:, -1], h0)
        if fs is None:
            fs = self.free_feed(P, s_n)
        fc = self.trunk(q1, ctx, h1, fs, P.u_anc)
        self.h_out, self.fs_out = h1, fs
        return self.heads(fc)

    @torch.no_grad()
    def gru_inputs(self, P: PtsIn, state: torch.Tensor) -> torch.Tensor:
        """--mem-prev: what the GRU reads at these decisions (no gradient)"""
        return self.stage1(P, state)[0][:, -1]

    def memory_alone(self, f_seq, h, done, h_start=None):
        return Actor2.memory_alone(self, f_seq, h, done, h_start)

    def vmem_read(self, hs: torch.Tensor) -> torch.Tensor:
        return self.v_mem(hs)

    def policy_parameters(self):
        """every weight but the free-space head's (it has its own optimizer)"""
        return [p for n_, p in self.named_parameters() if not n_.startswith("free.")]

    def free_parameters(self):
        return list(self.free.parameters())

    def config(self) -> dict:
        """what a checkpoint needs to build this actor again"""
        return {"arch": ARCH, "K": self.K, "beams": self.beams, "memory": self.memory, "residual": self.pi.residual,
                "state_dim": int(self.norm.mean.numel()), "mem_value": hasattr(self, "v_mem"),
                "mount": [float(x) for x in self.geom.mount], "rmax": float(self.geom.rmax)}


def build_pts_actor(cfg: dict, std0: float = 0.6, angles=None) -> PtsActor:
    """a `PtsActor` from `PtsActor.config()` (the geometry buffers come with the state dict)"""
    beams = int(cfg["beams"])
    ang = torch.zeros(beams) if angles is None else angles
    return PtsActor(ang, tuple(cfg.get("mount", (0.297, 0.0, 0.0))), float(cfg.get("rmax", 10.0)),
                    state_dim=int(cfg.get("state_dim", STATE_DIM2)), memory=int(cfg["memory"]), std0=std0,
                    residual=bool(cfg.get("residual", True)), mem_value=bool(cfg.get("mem_value", False)), K=int(cfg.get("K", K_FRAMES)))


def is_pts(ck: dict) -> bool:
    return (ck.get("pts") or {}).get("arch") == ARCH


def load_pts_actor(ck: dict, device) -> PtsActor:
    actor = build_pts_actor(ck["pts"], std0=float((ck.get("args") or {}).get("std0", 0.6))).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor


# ------------------------------------------------------------------------------------------------------------- inputs
class PtsObs(PlanObs2):
    """`PlanObs2` + what the point tokens need: the last K scans with their poses in the frame now (dead-reckoned from the
    car's own odometry, `env.sim.odom`), the previous plan's points for the anchors, the anchor action. `odom_noise` x ODOM_SIG
    per 0.1 s added to each increment (training; 0 on the console / car)."""

    #: what a stored moment (scene replay) keeps; the newest N_SCANS frames are the scan stack's own (`scans`), so only the
    #: older ones are kept apart (`pts_frames_old`)
    PTS_KEYS = ("pts_frames_old", "pts_rel", "pts_fok", "pl_pose", "pl_v", "anc_ok", "u_prev")

    def __init__(self, env, lim, goal=None, rows: Optional[torch.Tensor] = None, K: int = K_FRAMES, odom_noise: float = 0.0):
        lid = env.learner_ids if rows is None else rows
        Bl = int(lid.numel())
        dev = env.device
        beams = int(env.sim.lidar.angles.numel())
        self.K, self.odom_noise = int(K), float(odom_noise)
        from .plan_policy import N_SCANS
        if self.K < N_SCANS:
            raise ValueError(f"pts frames K >= the scan stack's {N_SCANS}")
        self.pts_frames = torch.zeros(Bl, self.K, beams, device=dev)
        self.pts_rel = torch.zeros(Bl, self.K, 3, device=dev)
        self.pts_fok = torch.zeros(Bl, self.K, dtype=torch.bool, device=dev)
        self.odom_last = torch.zeros(Bl, 3, device=dev)
        self.pl_pose = torch.zeros(Bl, PS.N_STEP, 3, device=dev)    # the last plan's 10 points (x, y, heading) in its own frame
        self.pl_v = torch.zeros(Bl, PS.N_STEP, device=dev)
        self.anc_ok = torch.zeros(Bl, dtype=torch.bool, device=dev)
        self.u_prev = torch.zeros(Bl, PS.ACT_DIM, device=dev)        # the anchor action for the next decision
        self.u_pending = None                                        # this decision's mean, committed by set_plan2
        super().__init__(env, lim, goal, rows)

    def _odom(self) -> torch.Tensor:
        return self.env.sim.odom.state[self.lid, :3].float()

    def reset_rows(self, rows: torch.Tensor) -> None:
        super().reset_rows(rows)
        if rows.numel() == 0:
            return
        self.pts_frames[rows] = self._scan()[rows][:, None, :].expand(-1, self.K, -1)
        self.pts_rel[rows] = 0.0
        self.pts_fok[rows] = False
        self.pts_fok[rows, 0] = True
        self.odom_last[rows] = self._odom()[rows]
        self.pl_pose[rows] = 0.0; self.pl_v[rows] = 0.0; self.anc_ok[rows] = False; self.u_prev[rows] = 0.0

    def push_scan(self, rows: Optional[torch.Tensor] = None) -> None:
        super().push_scan(rows)
        od = self._odom()
        d = se2_rel(od, self.odom_last)                               # where the frame at the last push is, in the frame now
        if self.odom_noise > 0:
            e = torch.randn_like(d) * _const("odom_sig", list(ODOM_SIG), d) * self.odom_noise
            d = se2_compose(d, e)
        K = self.K
        f = torch.cat([self._scan()[:, None, :], self.pts_frames[:, :-1]], 1)
        r = torch.cat([torch.zeros_like(d)[:, None], se2_compose(d[:, None].expand(-1, K - 1, -1), self.pts_rel[:, :-1])], 1)
        k = torch.cat([torch.ones_like(self.pts_fok[:, :1]), self.pts_fok[:, :-1]], 1)
        if rows is None:
            self.pts_frames, self.pts_rel, self.pts_fok, self.odom_last = f, r, k, od
        else:
            self.pts_frames = torch.where(rows[:, None, None], f, self.pts_frames)
            self.pts_rel = torch.where(rows[:, None, None], r, self.pts_rel)
            self.pts_fok = torch.where(rows[:, None], k, self.pts_fok)
            self.odom_last = torch.where(rows[:, None], od, self.odom_last)

    def set_plan2(self, p, rows: Optional[torch.Tensor] = None) -> None:
        super().set_plan2(p, rows)
        ii = PT_IDX.to(p["x"].device)
        pose = torch.stack([p["x"][:, ii], p["y"][:, ii], p["psi"][:, ii]], -1).float()
        u = shift_anchor(self.u_pending.float()) if self.u_pending is not None else torch.zeros_like(self.u_prev)
        if rows is None:
            self.pl_pose, self.pl_v, self.u_prev = pose, p["pv"].float().clone(), u
            self.anc_ok = torch.ones_like(self.anc_ok)
        else:
            self.pl_pose = torch.where(rows[:, None, None], pose, self.pl_pose)
            self.pl_v = torch.where(rows[:, None], p["pv"].float(), self.pl_v)
            self.u_prev = torch.where(rows[:, None], u, self.u_prev)
            self.anc_ok = self.anc_ok | rows
        self.u_pending = None

    def pts_input(self) -> PtsIn:
        """this decision's inputs: the anchors are the last plan's points 2..10 (t_1..t_9 now) and its 10th carried on 0.1 s at its
        speed and heading, moved from the frame the plan was drawn in (the frame one push ago) into the frame now"""
        P = self.pl_pose
        ext = torch.stack([P[:, -1, 0] + self.pl_v[:, -1] * PS.DT * torch.cos(P[:, -1, 2]),
                           P[:, -1, 1] + self.pl_v[:, -1] * PS.DT * torch.sin(P[:, -1, 2]), P[:, -1, 2]], -1)
        prev = torch.cat([P[:, 1:], ext[:, None]], 1)
        anc = se2_compose(self.pts_rel[:, 1][:, None].expand(-1, PS.N_STEP, -1), prev)
        ok = self.anc_ok & self.pts_fok[:, 1]
        z = lambda a: torch.where(ok.view(-1, *([1] * (a.dim() - 1))), a, torch.zeros_like(a))
        anc_v = torch.cat([self.pl_v[:, 1:], self.pl_v[:, -1:]], 1)
        return PtsIn(self.pts_frames, self.pts_rel, self.pts_fok, z(anc), z(anc_v), ok, z(self.u_prev))

    def pts_snapshot(self, rows=None) -> dict:
        sel = (lambda a: a.clone()) if rows is None else (lambda a: a[rows].clone())
        out = {k: sel(getattr(self, k)) for k in self.PTS_KEYS if k != "pts_frames_old"}
        out["pts_frames_old"] = sel(self.pts_frames[:, self.scans.shape[1]:])
        return out

    def pts_restore(self, rows: torch.Tensor, d: dict) -> None:
        """a stored moment's inputs back on `rows` -- after its scan stack (`scans[rows]`, the newest frames) is back; the
        odometry's reference is the car where it is now (just put there)"""
        for k in self.PTS_KEYS:
            if k != "pts_frames_old":
                getattr(self, k)[rows] = d[k].to(getattr(self, k).dtype)
        self.pts_frames[rows] = torch.cat([self.scans[rows], d["pts_frames_old"].to(self.pts_frames.dtype)], 1)
        self.odom_last[rows] = self._odom()[rows]


def plan_pts(actor: PtsActor, obs: PtsObs, lim, sample: bool = False, eps: Optional[torch.Tensor] = None,
             cap: Optional[torch.Tensor] = None, fs: Optional[torch.Tensor] = None):
    """one decision of a pts policy (`plan_policy.plan_v2`'s place): (u, mean, log_std, v_hat, plan). Its GRU state is
    `obs.mem_h`; the mean is left for the next decision's anchor (`obs.u_pending`, committed by `set_plan2`)"""
    Bl = obs.scans.shape[0]
    h = getattr(obs, "mem_h", None)
    if h is None or h.shape != (Bl, actor.memory):
        h = obs.mem_h = torch.zeros(Bl, actor.memory, device=obs.scans.device)
    mean, log_std, v_hat = actor(obs.pts_input(), obs.state(), h, fs)
    obs.mem_h = actor.h_out.detach()
    obs.u_pending = mean.detach()
    u = (mean + log_std.exp() * (torch.randn_like(mean) if eps is None else eps)) if sample else mean
    p = PS.roll(torch.tanh(u), v_hat.detach().clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim, cap)
    return u, mean, log_std, v_hat, p


# ------------------------------------------------------------------------------------------------------------- the rollout's store
class PtsStore:
    """what the update needs to build every decision's tokens again -- no tokens are kept. Per decision: each frame's pose in the
    frame then (the dead-reckoned chain AS the rollout composed it, noise included: 12 floats, read back bit for bit), the frame
    flags, the anchors, the anchor action, the free-space input the policy read, and where the older frames' scans are: the
    newest scan of each decision is already stored (`plan_rl`'s X[:, 0]); a frame from before a decision's 'segment start' (the
    rollout's first decision, or a scene put back mid-rollout) comes from the anchor table, filled at those moments. A
    restart starts a segment with no older frame (its frames are not there)."""

    def __init__(self, Bl: int, K: int, device):
        self.Bl, self.K, self.dev = int(Bl), int(K), device
        self.seg = torch.zeros(self.Bl, dtype=torch.long, device=device)
        self.src = torch.full((self.Bl,), -1, dtype=torch.long, device=device)
        self.cols = None

    def begin(self, obs: PtsObs) -> None:
        """the rollout's first decision is about to be taken (the last rollout's tables are let go)"""
        self.X = self.TAB = self.D = None
        self.cols = {k: [] for k in ("REL", "FOK", "ANC", "ANV", "AOK", "UAN", "FSIN", "SEG", "SRC")}
        self.tab = [obs.pts_frames.clone()]
        self.n_tab = self.Bl
        self.seg.zero_()
        self.src = torch.arange(self.Bl, device=self.dev)

    def record(self, P: PtsIn, fs: torch.Tensor) -> None:
        c = self.cols
        c["REL"].append(P.rel.clone()); c["FOK"].append(P.fok.clone()); c["ANC"].append(P.anc.clone())
        c["ANV"].append(P.anc_v.clone()); c["AOK"].append(P.anc_ok.clone()); c["UAN"].append(P.u_anc.clone())
        c["FSIN"].append(fs.detach().clone()); c["SEG"].append(self.seg.clone()); c["SRC"].append(self.src.clone())

    def restarted(self, rows: torch.Tensor, t: int) -> None:
        """rows (bool (Bl,)) restarted in decision t: decision t + 1 starts a segment without older frames"""
        self.seg = torch.where(rows, torch.full_like(self.seg, t + 1), self.seg)
        self.src = torch.where(rows, torch.full_like(self.src, -1), self.src)

    def loaded(self, idx: torch.Tensor, obs: PtsObs, t: int) -> None:
        """rows `idx` had a stored moment put back after decision t: decision t + 1's older frames are the moment's"""
        if idx.numel() == 0:
            return
        self.tab.append(obs.pts_frames[idx].clone())
        self.seg[idx] = t + 1
        self.src[idx] = self.n_tab + torch.arange(idx.numel(), device=self.dev)
        self.n_tab += int(idx.numel())

    def finish(self, X: torch.Tensor) -> None:
        """the rollout is over: X (T Bl, >= 1, beams) the stored scan stacks (newest first)"""
        self.X = X
        self.TAB = torch.cat(self.tab, 0)
        self.D = {k: torch.cat(v, 0) for k, v in self.cols.items()}
        self.cols = None

    def frames(self, rows: torch.Tensor) -> torch.Tensor:
        """(n, K, beams) the K frames of decisions `rows` (flat t Bl + b)"""
        Bl = self.Bl
        t, b = rows // Bl, rows % Bl
        seg, src = self.D["SEG"][rows], self.D["SRC"][rows]
        out = []
        for j in range(self.K):
            # after a restart the frames older than the segment are copies of its first scan (`PtsObs.reset_rows`; masked, but
            # read back as they were all the same)
            tj = torch.where(src < 0, torch.maximum(t - j, seg), t - j)
            inr = tj >= seg
            a = (j - (t - seg)).clamp(0, self.K - 1)
            out.append(torch.where(inr[:, None], self.X[tj.clamp_min(0) * Bl + b, 0], self.TAB[src.clamp_min(0), a]))
        return torch.stack(out, 1)

    def gather(self, rows: torch.Tensor) -> PtsIn:
        D = self.D
        return PtsIn(self.frames(rows), D["REL"][rows], D["FOK"][rows], D["ANC"][rows], D["ANV"][rows], D["AOK"][rows],
                     D["UAN"][rows])

    def fs_in(self, rows: torch.Tensor) -> torch.Tensor:
        return self.D["FSIN"][rows]


# ------------------------------------------------------------------------------------------------------------- free-space targets
def rect_corners(pose: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """pose (..., 3), half (..., 2) -> (..., 4, 2)"""
    c, s = torch.cos(pose[..., 2])[..., None], torch.sin(pose[..., 2])[..., None]
    ox = torch.stack([half[..., 0], -half[..., 0], -half[..., 0], half[..., 0]], -1)
    oy = torch.stack([half[..., 1], half[..., 1], -half[..., 1], -half[..., 1]], -1)
    return torch.stack([pose[..., 0:1] + c * ox - s * oy, pose[..., 1:2] + s * ox + c * oy], -1)


def point_rect_sd(p: torch.Tensor, pose: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """signed distance (negative inside) from points p (..., P, 2) to rectangles (pose (..., 3), half (..., 2))"""
    c, s = torch.cos(pose[..., 2])[..., None], torch.sin(pose[..., 2])[..., None]
    dx, dy = p[..., 0] - pose[..., 0:1], p[..., 1] - pose[..., 1:2]
    lx = (dx * c + dy * s).abs() - half[..., 0:1]
    ly = (-dx * s + dy * c).abs() - half[..., 1:2]
    return torch.hypot(lx.clamp_min(0), ly.clamp_min(0)) + torch.maximum(lx, ly).clamp_max(0)


def rect_rect_sd(a: torch.Tensor, ha: torch.Tensor, b: torch.Tensor, hb: torch.Tensor) -> torch.Tensor:
    """exact signed distance between rectangles (poses (..., 3), halves (..., 2)): apart, the nearest vertex-to-rectangle
    distance either way; overlapping, minus the smallest separating-axis overlap (the probe's `rect_rect_sd`)"""
    ca, cb = rect_corners(a, ha), rect_corners(b, hb)
    sep = torch.minimum(point_rect_sd(ca, b, hb).amin(-1), point_rect_sd(cb, a, ha).amin(-1))
    pen = None; over = None
    for th in (a[..., 2], a[..., 2] + math.pi / 2, b[..., 2], b[..., 2] + math.pi / 2):
        u = torch.stack([torch.cos(th), torch.sin(th)], -1)[..., None, :]
        pa, pb = (ca * u).sum(-1), (cb * u).sum(-1)
        ov = torch.minimum(pa.amax(-1), pb.amax(-1)) - torch.maximum(pa.amin(-1), pb.amin(-1))
        over = ov > 0 if over is None else over & (ov > 0)
        pen = ov if pen is None else torch.minimum(pen, ov)
    return torch.where(over, -pen, sep.clamp_min(0.0))


def footprint_samples(pose: torch.Tensor, L: float, W: float, n_long: int = 3, n_short: int = 1) -> torch.Tensor:
    """(..., 3) -> (..., S, 2): the L x W footprint's 4 corners, n_long points inside each long side and n_short inside each
    short side (12 by default: at most 0.145 m apart on a 0.58 x 0.31 car)"""
    hx, hy = 0.5 * L, 0.5 * W
    pts = [(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]
    for i in range(1, n_long + 1):
        x = -hx + 2.0 * hx * i / (n_long + 1)
        pts += [(x, hy), (x, -hy)]
    for i in range(1, n_short + 1):
        y = -hy + 2.0 * hy * i / (n_short + 1)
        pts += [(hx, y), (-hx, y)]
    o = _const(("fp", L, W, n_long, n_short), pts, pose)                     # (S, 2)
    c, s = torch.cos(pose[..., 2])[..., None], torch.sin(pose[..., 2])[..., None]
    return torch.stack([pose[..., 0:1] + c * o[:, 0] - s * o[:, 1], pose[..., 1:2] + s * o[:, 0] + c * o[:, 1]], -1)


def wall_clear(sdf_at, poses: torch.Tensor, L: float, W: float) -> torch.Tensor:
    """(B, K) [m]: the smallest signed wall distance over the footprint's samples; `sdf_at(xy (B, K, S, 2)) -> (B, K, S)`"""
    return sdf_at(footprint_samples(poses, L, W)).amin(-1)


def box_clear(poses: torch.Tensor, L: float, W: float, box_pose: torch.Tensor, box_half: torch.Tensor,
              box_ok: torch.Tensor) -> torch.Tensor:
    """(B, K) [m]: our footprint at poses (B, K, 3) against the boxes that count (box_pose (B, C, 3), box_half (B, C, 2), box_ok
    (B, C) bool); +inf where none counts"""
    B, K = poses.shape[:2]
    C = box_pose.shape[1]
    if C == 0:
        return torch.full((B, K), float("inf"), device=poses.device)
    me_h = _const(("half", L, W), [0.5 * L, 0.5 * W], poses)
    sd = rect_rect_sd(poses[:, :, None].expand(B, K, C, 3), me_h.expand(B, K, C, 2),
                      box_pose[:, None].expand(B, K, C, 3), box_half[:, None].expand(B, K, C, 2))
    sd = torch.where(box_ok[:, None, :], sd, torch.full_like(sd, float("inf")))
    return sd.amin(-1)


def future_known(brk: torch.Tensor, K: int = N_PQ) -> torch.Tensor:
    """brk (T, B) bool: a restart / the other car put elsewhere DURING decision t -> (T, B, K): t + k is inside the rollout (index
    T = after its last decision) with no break in decisions t .. t + k - 1"""
    T, B = brk.shape
    cb = torch.cumsum(brk.long(), 0)
    before = cb - brk.long()
    out = torch.zeros(T, B, K, dtype=torch.bool, device=brk.device)
    for k in range(1, K + 1):
        n = T - k + 1
        if n <= 0:
            continue
        out[:n, :, k - 1] = (cb[k - 1:k - 1 + n] - before[:n]) == 0
    return out


def car_clear(anc_w: torch.Tensor, op: torch.Tensor, op_ok: torch.Tensor, know: torch.Tensor, brk: torch.Tensor,
              L: float, W: float, kofs=None):
    """the other car at t + t_k against our footprint at the anchors, time-aligned. anc_w (T, B, K, 3) world anchor poses;
    op (T+1, B, 3) the other car's pose at each decision and after the last; op_ok (T+1, B) it is on the track; know (T, B) it is
    'there' at t (within range, seen by a beam in the last 1 s); brk (T, B). -> (clearance (T, B, K), +inf where it does not
    count; unknown (T, B, K): it counts but its pose at t + t_k is not in this rollout -- the target is left out there)"""
    T, B, K = anc_w.shape[:3]
    kn = future_known(brk, K)
    opf = torch.zeros(T, B, K, 3, device=anc_w.device, dtype=anc_w.dtype)
    okf = torch.zeros(T, B, K, dtype=torch.bool, device=anc_w.device)
    for j in range(K):
        k = (j + 1) if kofs is None else int(kofs[j])
        n = T - k + 1
        if n <= 0:
            continue
        opf[:n, :, j] = op[k:k + n]
        okf[:n, :, j] = op_ok[k:k + n]
    h = _const(("half", L, W), [0.5 * L, 0.5 * W], anc_w)
    sd = rect_rect_sd(anc_w, h.expand(T, B, K, 2), opf, h.expand(T, B, K, 2))
    counts = know[..., None] & kn & okf
    car = torch.where(counts, sd, torch.full_like(sd, float("inf")))
    unknown = know[..., None] & ~kn
    return car, unknown


def free_target(static: torch.Tensor, car: torch.Tensor, unknown: torch.Tensor):
    """(target clipped to [FS_LO, FS_HI], mask): the nearest of the walls / knowable boxes (static) and the other car"""
    y = torch.minimum(static, car)
    y = torch.where(torch.isfinite(y), y, torch.full_like(y, FS_HI)).clamp(FS_LO, FS_HI)
    return y, ~unknown


def box_local_rects(procedural) -> torch.Tensor:
    """(S, 4) per catalogue shape: its footprint's bounding rectangle in the prop's own frame (centre x, y, half x, half y), from
    the half-planes' vertices (`prop_math._polygon_vertices`); the real boxes are rectangles (bevel <= 3 cm)"""
    from ..prop_math import _polygon_vertices
    n_, d_ = procedural.sh_n, procedural.sh_d
    valid = n_.pow(2).sum(-1) > 0.5
    v = _polygon_vertices(n_, d_, valid)                                       # (S, k_pad, 2)
    big = torch.full_like(v[..., 0], 1e6)
    lo_x = torch.where(valid, v[..., 0], big).amin(-1); hi_x = torch.where(valid, v[..., 0], -big).amax(-1)
    lo_y = torch.where(valid, v[..., 1], big).amin(-1); hi_y = torch.where(valid, v[..., 1], -big).amax(-1)
    return torch.stack([0.5 * (lo_x + hi_x), 0.5 * (lo_y + hi_y), 0.5 * (hi_x - lo_x), 0.5 * (hi_y - lo_y)], -1)


def boxes_as_rects(procedural, rows: torch.Tensor, rects: torch.Tensor):
    """the box slots of env rows `rows`: (pose (B, C, 3) of each bounding rectangle in the world, half (B, C, 2), live (B, C))"""
    sid = procedural.p_sid[rows]
    live = sid >= 0
    r = rects[sid.clamp_min(0)]                                                # (B, C, 4)
    pose = procedural.p_poses[rows].float()
    c, s = torch.cos(pose[..., 2]), torch.sin(pose[..., 2])
    cx = pose[..., 0] + c * r[..., 0] - s * r[..., 1]
    cy = pose[..., 1] + s * r[..., 0] + c * r[..., 1]
    return torch.stack([cx, cy, pose[..., 2]], -1), r[..., 2:4], live


def signed_wall_field(track_set) -> torch.Tensor:
    """the track set's walls-only distance field made signed (negative inside walls: minus the distance to free space), on the
    same cells (`Track.sample_edt(..., field=)` reads it). Built once on the CPU (scipy), as `Track` builds its `sdf`"""
    import numpy as np
    from scipy import ndimage
    edt = track_set.edt
    out = edt.detach().float().cpu().numpy().copy()
    for k in range(int(track_set.t_H.numel())):
        o, H, W = int(track_set.t_off[k]), int(track_set.t_H[k]), int(track_set.t_W[k])
        res = float(track_set.t_res[k])
        g = out[o:o + H * W].reshape(H, W)
        occ = g <= 0.0
        if occ.any():
            inside = ndimage.distance_transform_edt(occ).astype(np.float32) * res
            g -= inside
        out[o:o + H * W] = g.reshape(-1)
    return torch.from_numpy(out).to(edt.device)


def car_future_guess(op_state: torch.Tensor, K: int = N_PQ) -> torch.Tensor:
    """--free-oracle's stand-in for the other car's next second when no copy's plan is known: its pose (B, 3), speed and yaw rate
    held -- op_state (B, >= 6): x, y, yaw, vx, vy, yaw_rate (`dynamics` order) -> (B, K, 3)"""
    from .. import dynamics as dyn
    x, y, th = op_state[:, 0], op_state[:, 1], op_state[:, 2]
    v = torch.hypot(op_state[:, dyn.IVX], op_state[:, dyn.IVY])
    r = op_state[:, dyn.IR]
    t = torch.arange(1, K + 1, device=op_state.device, dtype=op_state.dtype) * PS.DT
    h = th[:, None] + r[:, None] * t
    # a constant-turn arc; straight where the turn is tiny
    rr = r[:, None]
    small = rr.abs() < 1e-3
    dx = torch.where(small, v[:, None] * t * torch.cos(th[:, None]), v[:, None] / rr.where(~small, torch.ones_like(rr)) * (torch.sin(h) - torch.sin(th[:, None])))
    dy = torch.where(small, v[:, None] * t * torch.sin(th[:, None]), v[:, None] / rr.where(~small, torch.ones_like(rr)) * (torch.cos(th[:, None]) - torch.cos(h)))
    return torch.stack([x[:, None] + dx, y[:, None] + dy, h], -1)
