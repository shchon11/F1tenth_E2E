"""A local path planner for the privileged teacher (2026-10-30): the GEOMETRY of getting round props and cars is
computed, and the RL policy only drives the given path at the car's limit.

Why. Two days of RL teachers learned the empty track at the limit (7.5-7.8 s, no hits) but learned to avoid props and cars
only slowly and partly from the sparse hit price (props 3-5/km; the grid teacher's actor barely reacted to the prop
channel after 700 updates while its critic had learned it), and every race stage slowed the empty-track lap. The planner
that knew the layout (layout_line, 09-24) touched nothing in 42 km; rlc6, which imitated it, hits 1.15 props/km.

What. Frenet coordinates along the track centerline (s, d: left +). The reference is the minimum-time raceline's offset
d_rl(s). A plan is an offset from that reference, e(s) = e_from + (e_to - e_from) * smoothstep((s - s_from) / L): it starts
from where the PREVIOUS plan was at the car's station (not from the car -- else a deviation from the plan would always be
zero), and ramps over L to a held offset e_to. Every decision the candidates (e_to on a grid across the track, L in a few
lengths) are checked along the next `HORIZON` m against
    props   -- their true shapes (local half-planes), inflated by the car's half width + `PROP_MARGIN`,
    walls   -- the distance field at the path, less the half width + `WALL_MARGIN`,
    cars    -- each other car's position when we would reach that station (constant velocity), `CAR_R` apart,
and the cheapest is kept: blocked ones pay by how soon they are blocked, free ones by how far they leave the raceline and
how far they move from the current plan. Outputs: the path at any stations (for the policy's track-ahead block), the
path offset at the car's station (for the tracking reward), and the free distance along the plan (where it is blocked).
"""
from __future__ import annotations

import math

import numpy as np
import torch

HORIZON = 15.0            # [m] of arc checked ahead
STEP = 0.4                # [m] spacing of the checks along the path
HALF_W = 0.16             # [m] the car's half width (0.31 m body)
PROP_MARGIN = 0.15        # [m] clearance kept from a prop beyond the half width
WALL_MARGIN = 0.15        # [m] clearance kept from a wall beyond the half width
CAR_R = 0.55              # [m] centre distance under which two cars touch (0.5 m long, a little more)
OFFSETS = tuple(float(x) for x in np.linspace(-1.2, 1.2, 13))   # [m] candidate held offsets from the raceline
RAMPS = (2.5, 5.0)        # [m] candidate ramp lengths
NEAR_M = 1.2              # [m] checks start here: nearer is committed whatever the plan


def _smoothstep(x):
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class PathPlanner:
    """One plan per learner row of `env` (rows `lid`); `racelines` = one `Raceline` per track of the env."""

    def __init__(self, env, lid, racelines, res=0.1):
        self.env, self.lid = env, lid
        tr = env.sim.track
        dev = env.device
        self.dev = dev
        T = len(racelines)
        L = tr.length.float()
        n = int(math.ceil(float(L.max()) / res)) + 2
        self.res = res
        self.d_rl = torch.zeros(T, n, device=dev)           # raceline offset from the centerline, per res of s
        self.n_s = (L / res).long().clamp_min(1)
        for t, rl in enumerate(racelines):
            xy = torch.as_tensor(np.asarray(rl.xy), dtype=torch.float32, device=dev)
            s, lat = self._project_line(xy, t)
            # resample on the s grid (periodic): sort by s, interpolate
            o = torch.argsort(s); s, lat = s[o], lat[o]
            Lt = float(L[t]); m = int(self.n_s[t])
            grid = torch.arange(m, device=dev, dtype=torch.float32) * res
            s_ext = torch.cat([s[-1:] - Lt, s, s[:1] + Lt]); l_ext = torch.cat([lat[-1:], lat, lat[:1]])
            i = torch.searchsorted(s_ext, grid).clamp(1, s_ext.numel() - 1)
            w = ((grid - s_ext[i - 1]) / (s_ext[i] - s_ext[i - 1]).clamp_min(1e-6)).clamp(0, 1)
            self.d_rl[t, :m] = l_ext[i - 1] * (1 - w) + l_ext[i] * w
        # the raceline's speed profile on the same grid (for a plain tracker driving the plan)
        self.v_rl = torch.zeros(T, n, device=dev)
        for t, rl in enumerate(racelines):
            xy = torch.as_tensor(np.asarray(rl.xy), dtype=torch.float32, device=dev)
            s, _ = self._project_line(xy, t)
            v = torch.as_tensor(np.asarray(rl.v), dtype=torch.float32, device=dev)
            m = int(self.n_s[t])
            idx = (s / res).long().clamp(0, m - 1)
            acc = torch.zeros(m, device=dev); cnt = torch.zeros(m, device=dev)
            acc.index_add_(0, idx, v); cnt.index_add_(0, idx, torch.ones_like(v))
            filled = (cnt > 0).cpu().numpy()
            vv = (acc / cnt.clamp_min(1)).cpu().numpy()
            # fill the gaps forward from the last filled cell (cyclic: start after the last filled one)
            if filled.any():
                idx = np.where(filled, np.arange(m), -1)
                ff = np.maximum.accumulate(idx)
                ff[ff < 0] = np.flatnonzero(filled)[-1]
                vv = vv[ff]
            else:
                vv[:] = 5.0
            self.v_rl[t, :m] = torch.as_tensor(vv, device=dev)
        B = lid.numel()
        self.e_from = torch.zeros(B, device=dev); self.e_to = torch.zeros(B, device=dev)
        self.s_from = torch.zeros(B, device=dev); self.ramp = torch.full((B,), 4.0, device=dev)
        self.free = torch.full((B,), HORIZON, device=dev)
        self.fresh = torch.ones(B, dtype=torch.bool, device=dev)
        ds = torch.arange(1, int(HORIZON / STEP) + 1, device=dev, dtype=torch.float32) * STEP
        self.ds = ds                                        # (K,) stations ahead checked
        oo, rr = torch.meshgrid(torch.tensor(OFFSETS, device=dev), torch.tensor(RAMPS, device=dev), indexing="ij")
        self.cand_o, self.cand_r = oo.reshape(-1), rr.reshape(-1)       # (C,)

    def _project_line(self, xy, t):
        """the line's points onto track t's centerline IN ORDER: a plain nearest-point projection jumps to the next lane
        where the lap folds back on itself (a U-turn's two sides are ~1.5 m apart); here each point searches only the
        centerline points within +-3 m of arc of where the previous point landed (a walk along the lap, on the host)"""
        tr = self.env.sim.track
        cl = tr.cl[t].detach().cpu().numpy().astype(np.float64)               # (N, 2)
        cs = tr.cl_s[t].detach().cpu().numpy().astype(np.float64)
        tg = tr.cl_tangent[t].detach().cpu().numpy().astype(np.float64)
        ok = np.isfinite(cl).all(1)
        N = int(ok.sum()); cl, cs, tg = cl[:N], cs[:N], tg[:N]
        L = float(tr.length[t])
        P = xy.detach().cpu().numpy().astype(np.float64)
        d2 = ((P[0][None] - cl) ** 2).sum(1)
        k = int(d2.argmin())
        step = max(cs[1] - cs[0], 1e-3)
        w = int(math.ceil(3.0 / step))
        S = np.zeros(len(P)); LAT = np.zeros(len(P))
        for j, p in enumerate(P):
            idx = (k + np.arange(-w, w + 1)) % N
            d2 = ((p[None] - cl[idx]) ** 2).sum(1)
            k = int(idx[int(d2.argmin())])
            rel = p - cl[k]
            S[j] = (cs[k] + rel @ tg[k]) % L
            LAT[j] = rel[0] * -tg[k, 1] + rel[1] * tg[k, 0]
        return torch.as_tensor(S, dtype=torch.float32, device=xy.device), torch.as_tensor(LAT, dtype=torch.float32, device=xy.device)

    def set_reference(self, table):
        """use another reference line than the raceline: `table` (T, n) lateral offsets from the centerline per `res` of s
        (10-30: the fast solo policy's OWN line, recorded by `record_line`; mode 2 moves that line, so the planner must
        check the line it moves -- checking the raceline let the policy's own line + the offset run into a chicane wall)"""
        t = torch.as_tensor(table, dtype=torch.float32, device=self.dev)[: self.d_rl.shape[0]]
        m = min(t.shape[1], self.d_rl.shape[1])
        self.d_rl[: t.shape[0], :m] = t[:, :m]

    # --- the reference and a plan's offset ---------------------------------------------------------------------------
    def rl_speed(self, s, tid):
        L = self.env.sim.track.length[tid]
        i = (torch.remainder(s, L) / self.res).long()
        return self.v_rl[tid, torch.remainder(i, self.n_s[tid])]

    def rl_offset(self, s, tid):
        L = self.env.sim.track.length[tid]
        s = torch.remainder(s, L)
        i = (s / self.res)
        i0 = i.floor().long(); w = i - i0.float()
        m = self.n_s[tid]
        i0 = torch.remainder(i0, m); i1 = torch.remainder(i0 + 1, m)
        return self.d_rl[tid, i0] * (1 - w) + self.d_rl[tid, i1] * w

    def _arc(self, s, s0, tid):
        """s - s0 wrapped to (-L/2, L/2]"""
        L = self.env.sim.track.length[tid]
        return torch.remainder(s - s0 + L / 2, L) - L / 2

    def plan_offset(self, s, tid, rows=None):
        """the current plan's offset from the raceline at stations s (B, ...)"""
        sl = slice(None) if rows is None else rows
        e_from, e_to, s_from, ramp = self.e_from[sl], self.e_to[sl], self.s_from[sl], self.ramp[sl]
        shp = [-1] + [1] * (s.dim() - 1)
        a = self._arc(s, s_from.view(shp), tid.view(shp) if tid.dim() == 1 and s.dim() > 1 else tid)
        return e_from.view(shp) + (e_to - e_from).view(shp) * _smoothstep(a / ramp.view(shp))

    def path_xy(self, s, tid):
        """world points of the plan at stations s (B, K) -> (B, K, 2), and the path offset from the centerline (B, K)"""
        tr = self.env.sim.track
        B, K = s.shape
        tk = tid[:, None].expand(B, K)
        d = self.rl_offset(s, tk) + self.plan_offset(s, tid)
        xy, yaw = tr.pose_at_s(s.reshape(-1), tk.reshape(-1))
        xy = xy.reshape(B, K, 2); yaw = yaw.reshape(B, K)
        nx, ny = -torch.sin(yaw), torch.cos(yaw)
        return torch.stack([xy[..., 0] + d * nx, xy[..., 1] + d * ny], -1), d

    # --- checking candidates -----------------------------------------------------------------------------------------
    def _blocked(self, pts, tid, t_reach):
        """pts (B, C, K, 2) world, t_reach (B, K) seconds until we get there -> (B, C, K) bool"""
        env = self.env
        B, C, K, _ = pts.shape
        lid = self.lid
        tr = env.sim.track
        edt = tr.sample_edt(pts, tid[:, None, None].expand(B, C, K))
        bad = edt < HALF_W + 0.05                      # walls: the clamp keeps the margin; this catches a curve's inside
        po = env.procedural
        if po is not None and po.C > 0:
            pp_all = po.p_poses[lid]; live_all = po.p_sid[lid] >= 0
            nrm_all, dd_all = po.p_n[lid], po.p_d[lid]
            px, py = pts[..., 0], pts[..., 1]
            for ci in range(po.C):
                live = live_all[:, ci]
                if not bool(live.any()):
                    continue
                pp = pp_all[:, ci]
                dx = px - pp[:, None, None, 0]; dy = py - pp[:, None, None, 1]
                cc, ss = torch.cos(pp[:, 2])[:, None, None], torch.sin(pp[:, 2])[:, None, None]
                lx, ly = dx * cc + dy * ss, -dx * ss + dy * cc
                nrm, dd = nrm_all[:, ci], dd_all[:, ci]
                fin = torch.isfinite(dd)
                sd = (nrm[:, None, None, :, 0] * lx[..., None] + nrm[:, None, None, :, 1] * ly[..., None]
                      - torch.where(fin, dd, torch.zeros_like(dd))[:, None, None, :])
                sd = torch.where(fin[:, None, None, :], sd, torch.full_like(sd, -1e9)).amax(-1)
                bad |= (sd < HALF_W + PROP_MARGIN) & live[:, None, None]
        if env.M > 1 and env.sim.other_idx is not None:
            st = env.last_result.state
            so = st[env.sim.other_idx[lid]]                                  # (B, M-1, S)
            pres = (~env.sim.absent[env.sim.other_idx[lid]]) if env.sim.absent_any else torch.ones(so.shape[:2], dtype=torch.bool, device=pts.device)
            for q in range(so.shape[1]):
                c, s_ = torch.cos(so[:, q, 2]), torch.sin(so[:, q, 2])
                vx = so[:, q, 3] * c - so[:, q, 4] * s_; vy = so[:, q, 3] * s_ + so[:, q, 4] * c
                fx = so[:, q, None, 0] + vx[:, None] * t_reach                # (B, K) where it will be
                fy = so[:, q, None, 1] + vy[:, None] * t_reach
                d2 = (pts[..., 0] - fx[:, None]) ** 2 + (pts[..., 1] - fy[:, None]) ** 2
                bad |= (d2 < CAR_R ** 2) & pres[:, q, None, None]
        return bad

    @torch.no_grad()
    def update(self):
        """re-plan every learner row from its current state (call once per decision, after the env step)"""
        env, lid = self.env, self.lid
        tr = env.sim.track
        res = env.last_result
        st = res.state[lid]
        tid = env.sim.tid[lid]
        s_car = res.s[lid]
        B = lid.numel()
        # rows whose episode just began: the plan starts at the car's own offset and ramps to the raceline
        fresh = (env.ep_step[lid] <= 1) | self.fresh
        if bool(fresh.any()):
            e_car = res.lateral[lid] - self.rl_offset(s_car, tid)           # the sim's own (fold-safe) projection
            self.e_from = torch.where(fresh, e_car, self.e_from); self.e_to = torch.where(fresh, torch.zeros_like(e_car), self.e_to)
            self.s_from = torch.where(fresh, s_car, self.s_from); self.ramp = torch.where(fresh, torch.full_like(e_car, 4.0), self.ramp)
            self.fresh = torch.zeros_like(self.fresh)
        e_now = self.plan_offset(s_car, tid)                                   # where the plan is at our station
        C = self.cand_o.numel(); K = self.ds.numel()
        s_k = s_car[:, None] + self.ds[None]                                  # (B, K)
        tk = tid[:, None].expand(B, K)
        base_d = self.rl_offset(s_k, tk)                                      # (B, K)
        e_c = e_now[:, None, None] + (self.cand_o[None, :, None] - e_now[:, None, None]) * \
            _smoothstep(self.ds[None, None, :] / self.cand_r[None, :, None])  # (B, C, K)
        # keep every candidate inside the track: clamp the held offset to the half width there (less the margins)
        half = tr.sample_edt(tr.pose_at_s(s_k.reshape(-1), tk.reshape(-1))[0].reshape(B, K, 2), tk)
        lim = (half - HALF_W - WALL_MARGIN - 0.10).clamp_min(0.0)   # a buffer: a clamped candidate is not "at the wall"
        d_c = (base_d[:, None] + e_c).clamp(-lim[:, None], lim[:, None])
        xy, yaw = tr.pose_at_s(s_k.reshape(-1), tk.reshape(-1))
        xy = xy.reshape(B, K, 2); yaw = yaw.reshape(B, K)
        nx, ny = -torch.sin(yaw), torch.cos(yaw)
        pts = torch.stack([xy[:, None, :, 0] + d_c * nx[:, None], xy[:, None, :, 1] + d_c * ny[:, None]], -1)  # (B, C, K, 2)
        v = st[:, 3].abs().clamp_min(1.5)
        t_reach = self.ds[None] / v[:, None]
        bad = self._blocked(pts, tid, t_reach)                                # (B, C, K)
        # the first metre is committed (a car beside a wall would find every candidate blocked at 0.4 m)
        bad = bad & (self.ds >= NEAR_M)[None, None]
        first = torch.where(bad, self.ds[None, None].expand_as(bad), torch.full_like(bad, HORIZON + 1.0, dtype=torch.float32)).amin(-1)
        free_c = first.clamp(max=HORIZON)                                     # (B, C)
        blocked = first <= HORIZON
        cost = (self.cand_o.abs()[None] + 0.5 * (self.cand_o[None] - self.e_to[:, None]).abs()
                + 0.2 * (self.cand_r[None] < 4.0).float())
        cost = torch.where(blocked, 1000.0 + (HORIZON - free_c) * 10.0, cost)
        best = cost.argmin(1)
        ar = torch.arange(B, device=self.dev)
        # adopt: a new plan from where the old one is now, unless the chosen target is the current one (keep its ramp)
        new_o = self.cand_o[best]; new_r = self.cand_r[best]
        same = (new_o - self.e_to).abs() < 1e-4
        self.e_from = torch.where(same, self.e_from, e_now)
        self.s_from = torch.where(same, self.s_from, s_car)
        self.ramp = torch.where(same, self.ramp, new_r)
        self.e_to = new_o
        self.free = free_c[ar, best]
        self.last_candidates = (pts, bad, best)                                # for drawing

    def deviation(self):
        """(B,) the car's lateral distance from its plan at its own station [m]"""
        env, lid = self.env, self.lid
        res = env.last_result
        tid = env.sim.tid[lid]
        s_car = res.s[lid]
        return res.lateral[lid] - (self.rl_offset(s_car, tid) + self.plan_offset(s_car, tid))


@torch.no_grad()
def record_line(env, act_fn, n_steps, res=0.1, skip_steps=80):
    """(T, n) the lateral offset from the centerline at which `act_fn` (obs-free callable env -> (B, 2) actions for the
    learner rows, decided every 2 steps) drives each track, per `res` m of s, averaged over every learner on that track and
    every pass; cells never visited are filled from their neighbours along s"""
    tr = env.sim.track
    lid = env.learner_ids
    T = int(tr.length.numel())
    L = tr.length.float()
    n = int(math.ceil(float(L.max()) / res)) + 2
    acc = torch.zeros(T, n, device=env.device); cnt = torch.zeros(T, n, device=env.device)
    act = torch.zeros(env.B, env.act_dim, device=env.device)
    for t in range(int(n_steps)):
        if t % 2 == 0:
            act[lid] = act_fn(env)
        env.step(act)
        if t >= skip_steps:
            res_ = env.last_result
            tid = env.sim.tid[lid]; s = res_.s[lid]
            i = (s / res).long().clamp(0, n - 1)
            ok = res_.state[lid, 3] > 1.0
            flat = tid * n + i
            acc.view(-1).index_add_(0, flat[ok], res_.lateral[lid][ok])
            cnt.view(-1).index_add_(0, flat[ok], torch.ones_like(s[ok]))
    out = np.zeros((T, n), np.float32)
    A = acc.cpu().numpy(); C = cnt.cpu().numpy()
    for t in range(T):
        m = int(math.ceil(float(L[t]) / res))
        v = np.where(C[t, :m] > 0, A[t, :m] / np.maximum(C[t, :m], 1), np.nan)
        good = np.flatnonzero(np.isfinite(v))
        if good.size:
            v = np.interp(np.arange(m), good, v[good], period=m)
        else:
            v = np.zeros(m)
        out[t, :m] = v
    return out, (C > 0).mean()
