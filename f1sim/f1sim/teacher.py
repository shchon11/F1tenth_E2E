"""Privileged teacher: pure pursuit on the optimized raceline + speed profile (batched torch).
Uses ground-truth state; only valid inside the simulator. Serves as IL teacher and as the
baseline the RL student must beat."""
from __future__ import annotations

import math
from typing import Optional

import torch

import numpy as np

from .raceline import Raceline, curvature
from .track import resample_closed


class RacelineTeacher:
    """mode "pp" (default): pure pursuit on the raceline with understeer compensation
    (effective wheelbase L + k_us v^2; the simulated car turns ~20 % less than kinematic at
    4 m/s^2 lateral, ~35 % at 6) and privileged latency / servo / calibration / grip compensation.
    mode "stanley": curvature feed-forward + Stanley feedback at the front axle; more accurate
    without latency but loses to pure pursuit under domain randomization (delays up to 80 ms):
    on the 26-track set at 6 m/s, collisions per env per 15 s: pp 0.13 vs stanley 0.39."""

    def __init__(self, raceline, wheelbase: float = 0.3302, device="cpu", mode: str = "pp",
                 lookahead_gain: float = 0.35, lookahead_min: float = 0.6, lookahead_max: float = 2.5,
                 speed_lookahead_time: float = 0.35, lateral_slowdown: float = 0.6,
                 speed_scale: float = 1.0, steer_max: float = 0.4189,
                 k_e: float = 2.5, k_psi: float = 1.0, v_soft: float = 1.0, k_us: float = 0.003,
                 ff_time: float = 0.05, k_e_pp: float = 0.0,
                 mu_nominal: float = 1.0489, mu_f_scale_nominal: float = 0.92):
        self.device = torch.device(device)
        rls = [raceline] if isinstance(raceline, Raceline) else list(raceline)     # one per track id
        N = max(len(r.xy) for r in rls)
        xy, v, kap = [], [], []
        for r in rls:
            xr = resample_closed(r.xy, N)
            s_old = r.s; s_new = np.linspace(0, r.length, N, endpoint=False)
            xy.append(xr); v.append(np.interp(s_new, np.concatenate([s_old, [r.length]]), np.concatenate([r.v, r.v[:1]])))
            kap.append(curvature(xr))
        self.xy = torch.tensor(np.stack(xy), dtype=torch.float32, device=self.device)     # (T, N, 2)
        self.v = torch.tensor(np.stack(v), dtype=torch.float32, device=self.device)       # (T, N)
        self.kappa = torch.tensor(np.stack(kap), dtype=torch.float32, device=self.device) # (T, N) left +
        tan = torch.roll(self.xy, -1, 1) - torch.roll(self.xy, 1, 1)
        self.tan = tan / tan.norm(dim=2, keepdim=True).clamp_min(1e-9)                     # (T, N, 2)
        self.length = torch.tensor([r.length for r in rls], dtype=torch.float32, device=self.device)
        self.N = N
        self.ds = self.length / N                                                          # (T,)
        self.L = wheelbase
        self.mode = mode
        self.k_ld, self.ld_min, self.ld_max = lookahead_gain, lookahead_min, lookahead_max
        self.k_e, self.k_psi, self.v_soft, self.k_us, self.ff_time, self.k_e_pp = k_e, k_psi, v_soft, k_us, ff_time, k_e_pp
        self.t_v = speed_lookahead_time
        self.lat_slow = lateral_slowdown
        self.speed_scale = speed_scale
        self.steer_max = steer_max
        self.mu_nom, self.mu_f_nom = mu_nominal, mu_f_scale_nominal

    @torch.no_grad()
    def plan_action(self, state: torch.Tensor, P=None, tid: Optional[torch.Tensor] = None, v_max: float = 8.0,
                    spec=None) -> torch.Tensor:
        """The teacher as a *planner*: the raceline segment ahead of the car, expressed in the
        plan action space (3 lateral offsets at the stations of a plan of length L_p + start/end
        speeds, normalized), see f1sim.mpc. This is what a plan-space student imitates."""
        from .mpc import PlanSpec, encode, plan_length, M_INV
        spec = spec or PlanSpec()
        xy, yaw, vx = state[:, :2], state[:, 2], state[:, 3]
        B = xy.shape[0]
        tid = torch.zeros(B, dtype=torch.long, device=xy.device) if tid is None else tid
        idx, _ = self.project(xy, tid)
        ds = self.ds[tid]
        Lp = plan_length(vx, spec)
        # raceline points at 6 arc distances up to L_p ahead, in the body frame
        fr = torch.linspace(1.0 / 6.0, 1.0, 6, device=xy.device)
        pidx = (idx[:, None] + ((Lp[:, None] * fr[None]) / ds[:, None]).round().long()) % self.N        # (B,6)
        pts = self.xy[tid[:, None].expand_as(pidx), pidx] - xy[:, None, :]
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        bx = pts[..., 0] * c[:, None] + pts[..., 1] * s_[:, None]
        by = -pts[..., 0] * s_[:, None] + pts[..., 1] * c[:, None]
        # least squares for g(xi) = b2 xi^2 + b3 xi^3 + b4 xi^4 through (bx/Lp, by/Lp)
        xi = (bx / Lp[:, None]).clamp(0.0, 1.5)
        A = torch.stack([xi ** 2, xi ** 3, xi ** 4], 2)                                                  # (B,6,3)
        AtA = A.transpose(1, 2) @ A + 1e-4 * torch.eye(3, device=xy.device)                              # ridge: points behind the
        b = torch.linalg.solve(AtA, A.transpose(1, 2) @ (by / Lp[:, None])[..., None]).squeeze(-1)       # car collapse the fit
        XI = torch.tensor([1 / 3, 2 / 3, 1.0], device=xy.device)
        offsets = Lp[:, None] * (b[:, 0:1] * XI ** 2 + b[:, 1:2] * XI ** 3 + b[:, 2:3] * XI ** 4)
        # speeds from the profile: 0.15 s ahead and at the end of the plan
        v_idx0 = (idx + ((vx.abs() * spec.v_cmd_lead) / ds).round().long()) % self.N
        v_idx1 = (idx + (Lp / ds).round().long()) % self.N
        v0 = self.v[tid, v_idx0] * self.speed_scale; v1 = self.v[tid, v_idx1] * self.speed_scale
        if P is not None:
            grip = torch.sqrt(((P["mu"] * P["mu_f_scale"]) / (self.mu_nom * self.mu_f_nom)).clamp(0.3, 1.5))
            v0 = v0 * grip; v1 = v1 * grip
        return encode(offsets, v0, v1, v_max, spec)

    def project(self, xy: torch.Tensor, tid: Optional[torch.Tensor] = None):
        tid = torch.zeros(xy.shape[0], dtype=torch.long, device=xy.device) if tid is None else tid
        d2 = ((xy[:, None, :] - self.xy[tid]) ** 2).sum(-1)
        idx = d2.argmin(1)
        return idx, d2[torch.arange(len(idx)), idx].sqrt()

    def __call__(self, state: torch.Tensor, P=None, tid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """state (B,7) ground truth -> action (B,2) = (steer [rad], speed [m/s]).
        P: the simulator's per-env parameter dict (privileged). When given, the teacher
        compensates command latency, servo lag, actuator calibration and scales speed with grip.
        tid: per-env track id (which raceline to follow)."""
        xy, yaw, vx = state[:, :2], state[:, 2], state[:, 3]
        tid = torch.zeros(xy.shape[0], dtype=torch.long, device=xy.device) if tid is None else tid
        if P is not None:
            # latency compensation: predict the pose at the time the command takes effect
            dt_c = P["cmd_delay"] + 0.5 * P["servo_tau"]
            r = state[:, 5]
            yaw_c = yaw + r * dt_c
            xy = xy + torch.stack([vx * torch.cos(yaw_c), vx * torch.sin(yaw_c)], 1) * dt_c[:, None]
            yaw = yaw_c
        idx, lat_err = self.project(xy, tid)
        ds = self.ds[tid]
        if self.mode == "pp":
            ld = (self.k_ld * vx.abs()).clamp(self.ld_min, self.ld_max)
            tgt_idx = (idx + (ld / ds).round().long()) % self.N
            tgt = self.xy[tid, tgt_idx]
            dx, dy = tgt[:, 0] - xy[:, 0], tgt[:, 1] - xy[:, 1]
            alpha = torch.atan2(dy, dx) - yaw
            alpha = torch.remainder(alpha + math.pi, 2 * math.pi) - math.pi
            ld_act = torch.sqrt(dx ** 2 + dy ** 2).clamp_min(1e-3)
            L_eff = self.L + self.k_us * vx ** 2                                             # understeer
            steer = torch.atan(2 * L_eff * torch.sin(alpha) / ld_act)
            if self.k_e_pp > 0:                                                             # small lateral-error term
                t = self.tan[tid, idx]; p = self.xy[tid, idx]
                e = t[:, 0] * (xy[:, 1] - p[:, 1]) - t[:, 1] * (xy[:, 0] - p[:, 0])
                steer = steer - torch.atan(self.k_e_pp * e / (vx.abs() + self.v_soft))
        else:
            front = xy + self.L * torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
            fidx, _ = self.project(front, tid)
            t = self.tan[tid, fidx]; p = self.xy[tid, fidx]
            e = t[:, 0] * (front[:, 1] - p[:, 1]) - t[:, 1] * (front[:, 0] - p[:, 0])      # left of the line +
            psi = torch.atan2(t[:, 1], t[:, 0]) - yaw
            psi = torch.remainder(psi + math.pi, 2 * math.pi) - math.pi
            k_idx = (fidx + ((vx.abs() * self.ff_time) / ds).round().long()) % self.N
            kap = self.kappa[tid, k_idx]
            ff = torch.atan(kap * (self.L + self.k_us * vx ** 2))                          # + understeer
            steer = ff + self.k_psi * psi - torch.atan(self.k_e * e / (vx.abs() + self.v_soft))
        steer = steer.clamp(-self.steer_max, self.steer_max)
        v_idx = (idx + ((vx.abs() * self.t_v) / ds).round().long()) % self.N
        v_cmd = self.v[tid, v_idx] * self.speed_scale
        v_cmd = v_cmd * (1.0 - self.lat_slow * lat_err).clamp(0.3, 1.0)   # slow down when off-line
        if P is not None:
            grip = (P["mu"] * P["mu_f_scale"]) / (self.mu_nom * self.mu_f_nom)
            v_cmd = v_cmd * torch.sqrt(grip.clamp(0.3, 1.5)) / P["speed_gain"]
            steer = ((steer - P["steer_bias"]) / P["steer_gain"]).clamp(-self.steer_max, self.steer_max)
        return torch.stack([steer, v_cmd], 1)
