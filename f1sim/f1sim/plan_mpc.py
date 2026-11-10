"""MPC following a `plan_steps` plan (11-10, the user: "제어기좀 똑바로 만들어라").

Why not `plan_follow.PlanFollower`: its feedforward is MAP's steering table read at the plan's instant. The table is the
STEADY-STATE steering for a lateral acceleration, and near the tyres' limit it is steep -- the front tyre (B 2, C 1.8) needs
~3 deg more slip per m/s^2 at 7 m/s^2, against 0.5 deg of kinematic steering at 6 m/s. A plan's small jitter between
decisions (1 deg of kinematic steering = 1.9 m/s^2 at 6 m/s) came out of the table 4-7x larger, the car's yaw oscillated
(60 deg/s against 16 for the plan driven exactly), and the policy -- which only ever saw a car that drives its plan exactly
-- re-planned against that oscillation twice as jittery (1.84 deg a decision against 0.92, u300). Following one plan a
whole second (no re-planning) its steering still flipped 41 % of the time: the follower wobbles on its own. Six filters
on the table only traded the wobble for the corners (11-10).

This follower predicts the car with its own dynamics -- the simulator's single-track model: Pacejka lateral tyres, load
transfer, the rear friction circle shared with the drive force, the drag of a steered front tyre, the speed loop (the same
speed law as PlanFollower's, through the VESC's first-order loop and its limits), the servo's first-order lag and rate
limit, the command delay, the nominal car's constants -- and chooses the steering commands of the next `horizon` control
steps that keep the car on the plan's PATH (the distance to its nearest point) and its course (the direction the car
moves, not where its body points: at the tyres' limit the two differ by up to 10 deg, and the plan is a kinematic
bicycle), with a price on the steering's change. A jitter of the plan's curvature hardly moves its path (1.9 m/s^2 for
0.1 s is 1 cm), so it is not chased; a corner shows up in the path ahead of it, so the turn-in starts in time; how much
steering a lateral acceleration takes, and what more steering costs (speed, then the rear's grip once the speed loop
pushes), is the model's. First version (11-10): the speed held to the plan's in the model and the body heading as the
heading -- it steered to the stop (24 deg) for a second at the limit while the car's rear slid out (sideslip -15 deg).

One iLQR step a control step from the last solution shifted (real-time iteration), derivatives by finite differences of
the model, all rows batched. The speed command is PlanFollower's. The lateral velocity and yaw rate are read from the
state (the gyro measures the yaw rate; the lateral velocity needs an estimate on the real car); the servo angle is the
model's own prediction from the commands sent (no servo feedback on the car)."""
import math
import torch

from . import dynamics as dyn
from .plan_follow import PlanFollower

NZ = 9                                     # x, y, psi, vx, vy, r, servo angle, longitudinal specific force, last command
IX, IY, IPSI, IVX, IVY, IR, ID, IAX, IU = range(NZ)


class PlanMPC(PlanFollower):
    def __init__(self, n: int, device, nominal: dict, s_max: float = 0.4189, k_v: float = 2.0, horizon: int = 20,
                 sub: int = 4, w_lat: float = 400.0, w_head: float = 100.0, w_yaw: float = 0.0, w_rate: float = 2500.0,
                 w_end: float = 2.0):
        f = lambda k: float(nominal[k])
        super().__init__(n, device, motor_tau=f("motor_tau"), servo_tau=f("servo_tau"), delay=f("cmd_delay"), k_v=k_v,
                         k_lat=0.0, k_head=0.0, s_max=s_max, wheelbase=f("lf") + f("lr"))
        self.m, self.Iz, self.lf, self.lr, self.hcg = f("m"), f("Iz"), f("lf"), f("lr"), f("h")
        self.muf, self.mur = f("mu") * f("mu_f_scale"), f("mu") * f("mu_r_scale")
        kB = (f("mu_ref") / f("mu")) ** (1.0 - f("mu_stiffness_exp"))
        self.Bf, self.Cf, self.Ef = f("B_f") * kB, f("C_f"), f("E_f")
        self.Br, self.Cr, self.Er = f("B_r") * kB, f("C_r"), f("E_r")
        self.sv_max, self.vb0, self.vb1 = f("sv_max"), f("v_blend_min"), f("v_blend_max")
        self.a_max, self.a_brake, self.v_switch = f("a_max"), f("a_brake"), f("v_switch")
        self.c_roll, self.c_drag = f("c_roll"), f("c_drag")
        self.H, self.sub = int(horizon), int(sub)
        self.w_lat, self.w_head, self.w_yaw, self.w_rate, self.w_end = w_lat, w_head, w_yaw, w_rate, w_end
        self.U = torch.zeros(n, self.H, device=device)                # the last solution's commands (warm start)
        self.hist = torch.zeros(n, 4, device=device)                  # steering commands sent 1, 2, 3, 4 steps ago
        self.servo = torch.zeros(n, device=device)                    # the model's servo angle now
        self.axf = torch.zeros(n, device=device)                      # the model's longitudinal specific force now
        self.fresh = torch.ones(n, dtype=torch.bool, device=device)   # no command sent yet: start from the car's angle
        self.eps = torch.full((NZ,), 1e-3, device=device)
        # the last forward pass's states (positions about the car's position then): the next step's linearisation, shifted
        self.Zl, self.xy_l, self.have_l = None, None, False

    # ---------------------------------------------------------------- the model
    def _kin(self, d, vx):
        L = self.lf + self.lr
        td = torch.tan(d)
        beta = torch.atan(self.lr / L * td)
        return vx * torch.cos(beta) * td / L, vx * torch.tan(beta)

    def _f(self, z, u, vref, vff, dt: float, nsub: int):
        """the car `nsub` x `dt` on, the steering command u held, the speed command vff + k_v (vref - vx) through the
        speed loop. z (..., NZ); u, vref, vff (...)"""
        x, y, psi, vx, vy, r, d, axf = (z[..., i] for i in range(IU))
        L = self.lf + self.lr
        for _ in range(nsub):
            # the speed loop (the VESC's first order, its drive / brake limits) on PlanFollower's speed law
            v_cmd = vff + self.k_v * (vref - vx)
            a_lim = self.a_max * (self.v_switch / vx.abs().clamp_min(1e-3)).clamp(max=1.0)
            a_cmd = torch.minimum(torch.maximum((v_cmd - vx) / self.motor_tau, torch.full_like(vx, -self.a_brake)), a_lim)
            a_res = self.c_roll * torch.tanh(vx / 0.05) + self.c_drag * vx * vx.abs()
            rk0, vyk0 = self._kin(d, vx)
            rate = ((u - d) / self.servo_tau).clamp(-self.sv_max, self.sv_max)
            d = (d + rate * dt).clamp(-self.s_max, self.s_max)
            vxs = torch.where(vx.abs() < 0.3, torch.full_like(vx, 0.3), vx)
            af = d - torch.atan2(vy + self.lf * r, vxs)
            ar = -torch.atan2(vy - self.lr * r, vxs)
            Fzf = ((self.m * dyn.G * self.lr - self.m * self.hcg * axf) / L).clamp_min(0.0)
            Fzr = ((self.m * dyn.G * self.lf + self.m * self.hcg * axf) / L).clamp_min(0.0)
            Fyf = self.muf * Fzf * dyn.pacejka(af, self.Bf, self.Cf, self.Ef)
            Fx_req = self.m * a_cmd
            Fy_req = self.mur * Fzr * dyn.pacejka(ar, self.Br, self.Cr, self.Er)
            sc = (self.mur * Fzr / (Fx_req * Fx_req + Fy_req * Fy_req).sqrt().clamp_min(1e-6)).clamp(max=1.0)
            Fxr, Fyr = Fx_req * sc, Fy_req * sc
            cd, sd = torch.cos(d), torch.sin(d)
            axd = (Fxr - Fyf * sd) / self.m - a_res + vy * r
            vyd = (Fyr + Fyf * cd) / self.m - vx * r
            rd = (self.lf * Fyf * cd - self.lr * Fyr) / self.Iz
            # low speed: the simulator's blend toward the kinematic bicycle (its rate + a 50 ms relaxation)
            w = ((vx.abs() - self.vb0) / (self.vb1 - self.vb0)).clamp(0.0, 1.0)
            rk1, vyk1 = self._kin(d, vx)
            axd = w * axd + (1 - w) * (a_cmd - a_res)
            vyd = w * vyd + (1 - w) * ((vyk1 - vyk0) / dt + 20.0 * (vyk1 - vy))
            rd = w * rd + (1 - w) * ((rk1 - rk0) / dt + 20.0 * (rk1 - r))
            vx = vx + axd * dt
            vy = vy + vyd * dt
            r = r + rd * dt
            axf = axd - vy * r
            c, s = torch.cos(psi), torch.sin(psi)
            x = x + (vx * c - vy * s) * dt
            y = y + (vx * s + vy * c) * dt
            psi = psi + r * dt
        return torch.stack([x, y, psi, vx, vy, r, d, axf, torch.broadcast_to(u, d.shape)], -1)

    def _step(self, z, u, vref, vff):
        return self._f(z, u, vref, vff, self.h / self.sub, self.sub)

    def _inflight(self, T: float):
        """[(the history column, seconds)] of the commands in effect over [now, now + T), T <= delay: the one sent j steps
        ago lands j h - delay from now (the simulator's command history)"""
        segs = []
        for j in range(len(self.hist[0]), 0, -1):
            a, b = max(0.0, self.delay - j * self.h), min(T, max(0.0, self.delay - (j - 1) * self.h))
            if b - a > 1e-6:
                segs.append((j - 1, b - a))
        return segs

    # ---------------------------------------------------------------- the plan as a reference
    def _ref(self, f):
        """each row's plan at fractional grid indices f (n, K): x, y, course (the direction its CoG moves: the body's
        heading + the kinematic bicycle's slip), speed, speed command feedforward (+ motor_tau x acceleration), yaw rate;
        past the plan's end, on along its last curvature at its last speed"""
        last = self.X.shape[1] - 1
        fc = f.clamp(0.0, float(last))
        i0 = fc.floor().long().clamp(max=last - 1)
        w = fc - i0.float()
        g = lambda A: A.gather(1, i0) * (1 - w) + A.gather(1, i0 + 1) * w
        x, y, v, d = g(self.X), g(self.Y), g(self.V), g(self.D)
        a = (self.V.gather(1, i0 + 1) - self.V.gather(1, i0)) / self.h
        psi0, psi1 = self.PSI.gather(1, i0), self.PSI.gather(1, i0 + 1)
        dpsi = torch.remainder(psi1 - psi0 + math.pi, 2 * math.pi) - math.pi
        psi = psi0 + dpsi * w
        r = dpsi / self.h
        over = (f - float(last)).clamp_min(0.0) * self.h               # seconds past the end
        r_end = (torch.remainder(self.PSI[:, -1] - self.PSI[:, -2] + math.pi, 2 * math.pi) - math.pi)[:, None] / self.h
        v_end, p_end = self.V[:, -1:], self.PSI[:, -1:]
        pe = p_end + r_end * over
        safe = r_end.abs() > 1e-4
        rs = torch.where(safe, r_end, torch.ones_like(r_end))
        dx = torch.where(safe, v_end * (torch.sin(pe) - torch.sin(p_end)) / rs, v_end * torch.cos(p_end) * over)
        dy = torch.where(safe, -v_end * (torch.cos(pe) - torch.cos(p_end)) / rs, v_end * torch.sin(p_end) * over)
        beyond = over > 0
        x = torch.where(beyond, self.X[:, -1:] + dx, x)
        y = torch.where(beyond, self.Y[:, -1:] + dy, y)
        psi = torch.where(beyond, pe, psi)
        v = torch.where(beyond, v_end.expand_as(v), v)
        a = torch.where(beyond, torch.zeros_like(a), a)
        r = torch.where(beyond, r_end.expand_as(r), r)
        d = torch.where(beyond, self.D[:, -1:].expand_as(d), d)
        L = self.lf + self.lr
        course = psi + torch.atan(self.lr / L * torch.tan(d))
        return x, y, course, v, v + self.motor_tau * a, r

    # ---------------------------------------------------------------- cost
    def _track(self, z, P, wk):
        """path cost of states z (n, K, NZ) against the reference polyline P = (x, y, course, yaw rate) each (n, M): the
        nearest point within 8 grid steps of the state's own time (the polyline starts 2 steps before now); gradient and
        Gauss-Newton Hessian in z"""
        px, py, pc, pr = P
        K, M = z.shape[1], px.shape[1]
        d2 = (z[..., IX, None] - px[:, None, :]) ** 2 + (z[..., IY, None] - py[:, None, :]) ** 2          # (n, K, M)
        i_t = (torch.arange(1, K + 1, device=z.device) + 2.0 + self.delay / self.h)[None, :, None]      # the time's index
        far = (torch.arange(M, device=z.device)[None, None, :] - i_t).abs() > 8.0
        i = torch.where(far, torch.full_like(d2, 1e9), d2).argmin(-1)                                    # (n, K)
        xr, yr, cr, rr = px.gather(1, i), py.gather(1, i), pc.gather(1, i), pr.gather(1, i)
        sn, cs = torch.sin(cr), torch.cos(cr)
        e_lat = -sn * (z[..., IX] - xr) + cs * (z[..., IY] - yr)
        vx, vy = z[..., IVX], z[..., IVY]
        vv = (vx * vx + vy * vy).clamp_min(0.09)
        e_c = torch.remainder(z[..., IPSI] + torch.atan2(vy, vx.clamp_min(0.3)) - cr + math.pi, 2 * math.pi) - math.pi
        e_r = z[..., IR] - rr
        J = torch.zeros(*z.shape[:2], 3, NZ, device=z.device)                                            # d(e_lat, e_c, e_r)/dz
        J[..., 0, IX], J[..., 0, IY] = -sn, cs
        J[..., 1, IPSI], J[..., 1, IVX], J[..., 1, IVY] = 1.0, -vy / vv, vx / vv
        J[..., 2, IR] = 1.0
        Wd = torch.tensor([self.w_lat, self.w_head, self.w_yaw], device=z.device)
        e = torch.stack([e_lat, e_c, e_r], -1)                                                            # (n, K, 3)
        we = wk[..., None] * Wd * e
        g = 2 * (J.transpose(-1, -2) @ we[..., None])[..., 0]                                             # (n, K, NZ)
        Hm = 2 * J.transpose(-1, -2) @ ((wk[..., None] * Wd)[..., None] * J)                             # (n, K, NZ, NZ)
        return e_lat, g, Hm

    # ---------------------------------------------------------------- one control step
    def command(self, st: torch.Tensor) -> torch.Tensor:
        """st (n, >= 7): the car's x, y, yaw, vx, vy, yaw rate (the servo angle column is read once, at the start). (n, 2)
        steering, speed."""
        out_dev = st.device
        st = st.to(self.device).float()
        n, H, h, dev = self.n, self.H, self.h, st.device
        kf = self.k.float()
        # the speed: PlanFollower's (the speed loop's inversion at the command's landing + feedback on the speed now)
        f_lead = kf + (self.delay + 0.5 * h) / h
        v_ff, a_ff = self._at(self.V, f_lead)
        v_p, _ = self._at(self.V, kf)
        speed = v_ff + self.motor_tau * a_ff + self.k_v * (v_p - st[:, 3])
        # the servo the model believes in: the car's own angle before any command of ours
        self.servo = torch.where(self.fresh, st[:, dyn.ISTEER], self.servo)
        self.hist = torch.where(self.fresh[:, None], self.servo[:, None].expand_as(self.hist), self.hist)
        self.U = torch.where(self.fresh[:, None], self.servo[:, None].expand_as(self.U), self.U)
        self.axf = torch.where(self.fresh, torch.zeros_like(self.axf), self.axf)
        new_rows = self.fresh.clone()
        self.fresh[:] = False
        x0, y0 = st[:, 0], st[:, 1]
        z = torch.stack([torch.zeros_like(x0), torch.zeros_like(y0), st[:, 2], st[:, 3], st[:, 4], st[:, 5], self.servo,
                         self.axf, self.hist[:, 0]], 1)
        # the plan from 2 steps before now to past the horizon's end, on the control grid (the path the cost reads)
        M = H + 12
        fi = kf[:, None] - 2.0 + torch.arange(M, device=dev, dtype=torch.float32)[None, :]
        px, py, pc, pv, pvff, pr = self._ref(fi)
        P = (px - x0[:, None], py - y0[:, None], pc, pr)
        # the speed law's inputs at each model step's time (now + delay + k h)
        tk = kf[:, None] + (self.delay + h * torch.arange(H + 1, device=dev, dtype=torch.float32))[None, :] / h
        _, _, _, vref, vff, _ = self._ref(tk)
        # through the command delay with the commands already sent (the one sent j steps ago lands j h - delay from now)
        for j, T in self._inflight(self.delay):
            nsub = max(1, int(math.ceil(T / 0.005 - 1e-9)))
            z = self._f(z, self.hist[:, j], vref[:, 0], vff[:, 0], T / nsub, nsub)
        z[:, IU] = self.hist[:, 0]
        # warm start: the last solution one step on; linearised along the last forward pass one step on (real-time
        # iteration) -- a fresh start rolls the model out instead
        U = torch.cat([self.U[:, 1:], self.U[:, -1:]], 1)
        if self.have_l and not bool(new_rows.any()):
            Z = torch.cat([self.Zl[:, 1:], self._step(self.Zl[:, -1], U[:, -1], vref[:, H - 1], vff[:, H - 1])[:, None]], 1)
            Z[..., IX] += (self.xy_l[:, 0] - x0)[:, None]
            Z[..., IY] += (self.xy_l[:, 1] - y0)[:, None]
            # the car's yaw is wrapped to (-pi, pi], the stored states run on continuously: the yaw now on their branch
            z[:, IPSI] = Z[:, 0, IPSI] + torch.remainder(z[:, IPSI] - Z[:, 0, IPSI] + math.pi, 2 * math.pi) - math.pi
        else:
            Z = [z]
            for k in range(H):
                Z.append(self._step(Z[-1], U[:, k], vref[:, k], vff[:, k]))
            Z = torch.stack(Z, 1)                                      # (n, H + 1, NZ)
        # derivatives along (Z, U) by forward differences, every step at once
        E = torch.cat([torch.diag(self.eps), torch.zeros(2, NZ, device=dev)], 0)        # NZ state + nominal + control
        Zp = Z[:, :H, None, :] + E[None, None]                                          # (n, H, NZ + 2, NZ)
        du_p = torch.zeros(NZ + 2, device=dev); du_p[-1] = 1e-3
        Up = U[:, :, None] + du_p[None, None]
        F = self._step(Zp, Up, vref[:, :H, None].expand_as(Up), vff[:, :H, None].expand_as(Up))
        f0 = F[:, :, NZ]
        A = ((F[:, :, :NZ] - f0[:, :, None]) / self.eps[None, None, :, None]).transpose(2, 3)   # (n, H, NZ, NZ): d f / d z
        Bv = (F[:, :, NZ + 1] - f0) / 1e-3                                                   # (n, H, NZ)
        # path cost at z_1 .. z_H, the end weighted
        wk = torch.ones(1, H, device=dev); wk[:, -1] = self.w_end
        _, gz, Hz = self._track(Z[:, 1:], P, wk)
        # backward pass (one control: the Riccati recursion without a matrix inverse)
        Vz, Vzz = gz[:, -1], Hz[:, -1]
        kff = torch.zeros(n, H, device=dev); Kfb = torch.zeros(n, H, NZ, device=dev)
        for k in reversed(range(H)):
            Ak, Bk = A[:, k], Bv[:, k]
            du = U[:, k] - Z[:, k, IU]
            lu = 2 * self.w_rate * du
            luz = torch.zeros(n, NZ, device=dev); luz[:, IU] = -2 * self.w_rate
            if k > 0:
                lz, lzz = gz[:, k - 1].clone(), Hz[:, k - 1].clone()
            else:
                lz, lzz = torch.zeros(n, NZ, device=dev), torch.zeros(n, NZ, NZ, device=dev)
            lz[:, IU] = lz[:, IU] - 2 * self.w_rate * du
            lzz[:, IU, IU] = lzz[:, IU, IU] + 2 * self.w_rate
            Qz = lz + (Ak.transpose(1, 2) @ Vz[..., None])[..., 0]
            Qu = lu + (Bk * Vz).sum(1)
            Qzz = lzz + Ak.transpose(1, 2) @ Vzz @ Ak
            VB = (Vzz @ Bk[..., None])[..., 0]                                           # (n, NZ)
            Quu = 2 * self.w_rate + (Bk * VB).sum(1) + 1e-6
            Quz = luz + (Ak.transpose(1, 2) @ VB[..., None])[..., 0]
            kff[:, k] = -Qu / Quu
            Kfb[:, k] = -Quz / Quu[:, None]
            Vz = Qz - Quz * (Qu / Quu)[:, None]
            Vzz = Qzz - Quz[:, :, None] * Quz[:, None, :] / Quu[:, None, None]
            Vzz = 0.5 * (Vzz + Vzz.transpose(1, 2))
        # forward pass from the predicted state at the command's landing
        zn = z; Un = torch.empty_like(U); Zn = [z]
        for k in range(H):
            dz = zn - Z[:, k]
            dz[:, IPSI] = torch.remainder(dz[:, IPSI] + math.pi, 2 * math.pi) - math.pi
            uk = (U[:, k] + kff[:, k] + (Kfb[:, k] * dz).sum(1)).clamp(-self.s_max, self.s_max)
            Un[:, k] = uk
            zn = self._step(zn, uk, vref[:, k], vff[:, k])
            Zn.append(zn)
        self.U = Un
        self.Zl, self.xy_l, self.have_l = torch.stack(Zn, 1), torch.stack([x0, y0], 1), True
        steer = Un[:, 0]
        # the model's own servo angle and drive force one control step on, under the commands in effect over it
        zs = torch.stack([torch.zeros_like(x0), torch.zeros_like(y0), st[:, 2], st[:, 3], st[:, 4], st[:, 5], self.servo,
                          self.axf, self.hist[:, 0]], 1)
        for j, T in self._inflight(min(h, self.delay)):
            nsub = max(1, int(math.ceil(T / 0.005 - 1e-9)))
            zs = self._f(zs, self.hist[:, j], vref[:, 0], vff[:, 0], T / nsub, nsub)
        self.servo, self.axf = zs[:, ID], zs[:, IAX]
        self.hist = torch.cat([steer[:, None], self.hist[:, :-1]], 1)
        self.k += 1
        return torch.stack([steer, speed], 1).to(out_dev)

    def set_plan(self, rows, x, y, psi, v, delta, cmd_d=None, cmd_v=None) -> None:
        if torch.is_tensor(rows):
            rows = rows.to(self.device)
        to = lambda a: a.to(self.device) if a is not None else None
        super().set_plan(rows, to(x), to(y), to(psi), to(v), to(delta), to(cmd_d), to(cmd_v))

    def reset_rows(self, rows) -> None:
        """a row restarted (a respawn, the console's reset): the servo and the commands start again from the car's"""
        if torch.is_tensor(rows):
            rows = rows.to(self.device)
        self.fresh[rows] = True
