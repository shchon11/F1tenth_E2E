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
                 mu_nominal: float = 1.0489, mu_f_scale_nominal: float = 0.92,
                 recover_time: float = 0.0, v_recover_min: float = 0.6, a_lat_recover: float = 6.0,
                 v_max_profile: float = 10.0, a_lat: float = 6.0, a_acc: float = 6.0, a_brake: float = 3.0):
        self.device = torch.device(device)
        rls = [raceline] if isinstance(raceline, Raceline) else list(raceline)     # one per track id
        N = max(len(r.xy) for r in rls)
        # speed profiles per grip level: the teacher is privileged, so it brakes and corners for the
        # friction *this* car has (a_lat and a_acc scale with grip; braking does not, see below)
        from .raceline import speed_profile
        self.grip_levels = np.linspace(0.45, 1.0, 12)      # never faster than the nominal profile: above nominal grip the
                                                            # limit is tracking error, not the tyres (measured: faster = more crashes)
        xy, v, kap = [], [], []
        for r in rls:
            xr = resample_closed(r.xy, N)
            xy.append(xr); kap.append(curvature(xr))
            # Measured on the real car (02_pre-competition bags, IMU a_y/a_x smoothed over 200 ms to
            # drop vibration spikes -- the raw p99 reads 15.9 m/s^2 and is not sustained for even
            # 0.02 s): lateral 9.2-11.5, longitudinal +6 to +10, braking -5.4 (IMU) / -6.3 (wheel
            # speed). mu*g in the simulator is 9.5, so the physics already matched; the profile was
            # planning at 6.0 * grip, i.e. 30-60 % of what the car can do, which also put the
            # braking point outside the 10 m the LiDAR can see.
            # a_lat and a_acc scale with grip; braking does not. Measured on the car, the hardest
            # 200 ms of braking always coincides with the regen current at 95-100 % of its
            # configured limit, at -4.2 to -5.7 m/s^2 -- well inside mu*g, so the VESC binds first
            # and the surface never gets a vote. Scaling it by grip made the teacher plan 2.25 m/s^2
            # of braking on a low-grip draw where the car can still do 5, costing lap time for
            # nothing.
            v.append(np.stack([speed_profile(xr, v_max_profile, a_lat * g, a_acc * g, a_brake)
                               for g in self.grip_levels]))   # (K, N)
        self.xy = torch.tensor(np.stack(xy), dtype=torch.float32, device=self.device)     # (T, N, 2)
        self.v_grip = torch.tensor(np.stack(v), dtype=torch.float32, device=self.device)  # (T, K, N)
        self.v = self.v_grip[:, -1]                                                        # nominal grip (T, N)
        self.grip_levels_t = torch.tensor(self.grip_levels, dtype=torch.float32, device=self.device)
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
        # recover_time > 0: cap the commanded speed by what can still be turned back onto the lane.
        # The existing slowdown reads *lateral* error only, so a car sitting on the line facing
        # backwards is told to drive at full profile speed: measured 3.06 m/s at 180 deg of heading
        # error, on a commanded radius of 0.74 m -- 12.7 m/s^2 of lateral acceleration against the
        # ~6 m/s^2 the profile itself assumes. That is not a recovery demonstration, and DAgger can
        # only teach what the teacher shows, so weighting those samples harder would teach it harder.
        self.recover_time, self.v_recover_min, self.a_lat_recover = recover_time, v_recover_min, a_lat_recover

    @torch.no_grad()
    def plan_action(self, state: torch.Tensor, P=None, tid: Optional[torch.Tensor] = None, v_max: float = 8.0,
                    spec=None, iters: int = 6) -> torch.Tensor:
        """The teacher as a *planner*: the raceline segment ahead of the car expressed in the plan
        action space (f1sim.mpc: curvature knots along the next L_p of arc + start/end speeds).
        Gauss-Newton fits the knots so the integrated path passes through the raceline points
        ahead (in the body frame), starting from the raceline's own curvature there; this is what
        a plan-space student imitates."""
        from .mpc import N_KNOTS, PlanSpec, encode, path_points, plan_length
        spec = spec or PlanSpec()
        xy, yaw, vx = state[:, :2], state[:, 2], state[:, 3]
        B = xy.shape[0]; dev = xy.device
        tid = torch.zeros(B, dtype=torch.long, device=dev) if tid is None else tid
        idx, _ = self.project(xy, tid)
        ds = self.ds[tid]
        Lp = plan_length(vx, spec)
        # raceline points at 6 arc distances between 0.4 and 1.0 L_p ahead, in the body frame (the near
        # points are skipped on purpose: like pure pursuit, an off-line car should rejoin gently)
        M = 6
        fr = torch.linspace(0.4, 1.0, M, device=dev)
        pidx = (idx[:, None] + ((Lp[:, None] * fr[None]) / ds[:, None]).round().long()) % self.N        # (B,M)
        pts = self.xy[tid[:, None].expand_as(pidx), pidx] - xy[:, None, :]
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        tx = pts[..., 0] * c[:, None] + pts[..., 1] * s_[:, None]
        ty = -pts[..., 0] * s_[:, None] + pts[..., 1] * c[:, None]
        # initial knots: the raceline curvature at the knot arc distances
        kidx = (idx[:, None] + ((Lp[:, None] * torch.linspace(0, 1, N_KNOTS, device=dev)[None]) / ds[:, None]).round().long()) % self.N
        k_rl = self.kappa[tid[:, None].expand_as(kidx), kidx].clone()
        # initial guess = what pure pursuit would do (its lookahead grows with speed, which is exactly the
        # gain scheduling the direct teacher's rejoin has) blended into the raceline's own curvature ahead:
        # the fit below only refines this, so an off-line car neither snaps to the line at full lock
        # (over-correction crashes) nor drifts along beside it (under-correction crashes)
        ld = (self.k_ld * vx.abs()).clamp(self.ld_min, self.ld_max)
        tgt = self.xy[tid, (idx + (ld / ds).round().long()) % self.N] - xy
        alpha = torch.atan2(-tgt[:, 0] * s_ + tgt[:, 1] * c, tgt[:, 0] * c + tgt[:, 1] * s_)
        k_pp = (2.0 * torch.sin(alpha) / ld).clamp(-spec.kappa_max, spec.kappa_max)
        w = torch.clamp(1.0 - torch.linspace(0, 1, N_KNOTS, device=dev) * Lp[:, None] / ld[:, None], 0.0, 1.0)   # PP weight fades over the lookahead
        k = w * k_pp[:, None] + (1 - w) * k_rl
        k_init = k.clone()
        gb = self.grip_bin(P, B, dev)
        k_lim = 0.85 * spec.kappa_max

        def resid(kk):                                             # path samples at the target arc fractions vs targets
            x, y, _, _ = path_points(kk, Lp, 25)
            j = (fr * 24).round().long()
            return torch.cat([x[:, j] - tx, y[:, j] - ty], 1)      # (B,2M)
        lam, mu, eps = 1e-2, 0.3, 0.02                             # GN damping, ridge towards the pure-pursuit / raceline guess
        for _ in range(iters):
            r0 = resid(k)
            J = torch.stack([(resid(k + eps * torch.nn.functional.one_hot(torch.tensor(j, device=dev), N_KNOTS).to(k.dtype)[None]) - r0) / eps
                             for j in range(N_KNOTS)], 2)          # (B,2M,4)
            A = J.transpose(1, 2) @ J + (lam + mu) * torch.eye(N_KNOTS, device=dev)
            g = J.transpose(1, 2) @ r0[..., None] + mu * (k - k_init)[..., None]
            step = torch.linalg.solve(A, g).squeeze(-1)
            k = (k - step).clamp(-k_lim, k_lim)
        # speeds from the profile: 0.15 s ahead and at the end of the plan
        v_idx0 = (idx + ((vx.abs() * spec.v_cmd_lead) / ds).round().long()) % self.N
        v_idx1 = (idx + (Lp / ds).round().long()) % self.N
        _, lat_err = self.project(xy, tid)
        slow = (1.0 - self.lat_slow * lat_err).clamp(0.3, 1.0)     # off the line: slow down, like the direct teacher
        v0 = self.speed_at(tid, v_idx0, gb) * slow; v1 = self.speed_at(tid, v_idx1, gb) * slow
        cap = self.heading_speed_cap(yaw, tid, idx)
        if cap is not None:
            v0 = torch.minimum(v0, cap); v1 = torch.minimum(v1, cap)
        return encode(k, v0, v1, v_max, spec)

    label_grip = "true"          # "true": per-env grip (privileged); "nominal"/"conservative": constant

    def heading_speed_cap(self, yaw: torch.Tensor, tid: torch.Tensor, idx: torch.Tensor) -> Optional[torch.Tensor]:
        """Speed from which the car can still turn back onto the lane within `recover_time`.

        Heading change available over a time t at the grip-limited curvature a_lat / v^2 is
        a_lat * t / v, so recovering a heading error psi needs v <= a_lat * t / psi. Returns None
        when disabled, leaving the commanded speed exactly as before."""
        if self.recover_time <= 0:
            return None
        tan = self.tan[tid, idx]
        psi = torch.remainder(torch.atan2(tan[:, 1], tan[:, 0]) - yaw + math.pi, 2 * math.pi) - math.pi
        return (self.a_lat_recover * self.recover_time / psi.abs().clamp_min(1e-3)).clamp_min(self.v_recover_min)

    def grip_bin(self, P, B: int, device):
        """Index of the speed profile this teacher drives on.

        "true" reads each env's randomized mu. That makes the teacher fast, but it also makes the
        *label* a function of something the student cannot see: two identical scans get speed labels
        up to 1/0.45 ~ 2.2x apart, and a Huber regression can only learn their conditional mean --
        too fast in low grip, too slow in high grip. "nominal" and "conservative" pin the profile so
        the label is a function of the observation alone (the price is a slower target, and a teacher
        that can over-drive a low-grip car, which is why collection uses `speed_scale` < 1).
        """
        if self.label_grip == "nominal" or P is None:
            return torch.full((B,), len(self.grip_levels) - 1, dtype=torch.long, device=device)
        if self.label_grip == "conservative":
            return torch.zeros(B, dtype=torch.long, device=device)
        g = ((P["mu"] * P["mu_f_scale"]) / (self.mu_nom * self.mu_f_nom)).clamp(max=1.0)
        return (g[:, None] - self.grip_levels_t[None]).abs().argmin(1)

    speed_mode = "grip"          # "grip": per-grip profiles (braking points move too); "sqrt": nominal profile x sqrt(grip)

    def speed_at(self, tid: torch.Tensor, idx: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
        if self.speed_mode == "sqrt":
            g = self.grip_levels_t[gb]
            return self.v[tid, idx] * torch.sqrt(g) * self.speed_scale
        return self.v_grip[tid, gb, idx] * self.speed_scale

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
        v_cmd = self.speed_at(tid, v_idx, self.grip_bin(P, xy.shape[0], xy.device))
        v_cmd = v_cmd * (1.0 - self.lat_slow * lat_err).clamp(0.3, 1.0)   # slow down when off-line
        cap = self.heading_speed_cap(yaw, tid, idx)
        if cap is not None:
            v_cmd = torch.minimum(v_cmd, cap)
        if P is not None:
            v_cmd = v_cmd / P["speed_gain"]
            steer = ((steer - P["steer_bias"]) / P["steer_gain"]).clamp(-self.steer_max, self.steer_max)
        return torch.stack([steer, v_cmd], 1)
