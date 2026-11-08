"""Following a `plan_steps` plan in time (11-08, the user: "제어기를 예측가능하게끔"): the car should be where the plan
says, when it says -- the training's assumption (`Sim.ideal_rows`) made true on the physics.

MAP follows a plan as a path (the nearest waypoint, a lookahead) and commands the first 0.1 s segment's mean speed. On
the VESC's first-order speed loop (`actuators.vesc_accel`: a = (v_cmd - v) / motor_tau, motor_tau 0.20 s) a plan that
accelerates at a from v asks for v + a * 0.05 and gets a / 4 -- every plan, so the car is always behind it, and a stop
that plans to back out moves a few cm before the next plan turns it round.

This follower reads the plan on the control grid (the dense poses `plan_steps.roll` makes, 0.025 s apart) and inverts
the two actuator lags the simulator models -- the speed loop and the servo (`dynamics`: first order, `servo_tau`, rate
`sv_max`) -- after the command delay: the speed command is the plan's speed plus motor_tau x its acceleration, the
steering command the plan's angle plus servo_tau x its rate, both read at (now + delay + half a step). The plan keeps
inside the motor, brakes, servo rate and tyre table (`plan_steps.Limits`), so on the nominal car these are commands the
car can carry out. A small feedback on the error to the plan's pose at this instant (lateral, heading, speed) takes up
what the inversion misses (a randomised car, tyre slip, contacts). Gains chosen by the error to a whole plan followed
1 s (11-09, u1500, 64 cars x 40 s): (k_v, k_lat, k_head) 0,0,0 -> 23.1 cm at 1 s, 1,0,0 15.2, 1,.2,.2 14.5, 1,.5,.5 14.0,
2,1,1 12.1 (MAP 46.2). The lateral gains barely change the closed-loop lap: the weave there is the policy re-planning
every 0.1 s for an instant car, not this follower. The constants are the nominal car's, not the
randomised ones: the real car's are not known to it either."""
import math
import torch

from . import plan_steps as PS


class PlanFollower:
    """(steering, speed) commands for `n` rows from their current plans. `set_plan(rows, x, y, psi, v, delta)` hands
    rows a new plan (the dense world-frame poses from `plan_policy.plan_world` and the plan's `v` / `delta`, (n_r, N SUB
    + 1)); `command(state)` is called once per control step and returns (n, 2), stepping each row's clock."""

    def __init__(self, n: int, device, motor_tau: float = 0.20, servo_tau: float = 0.08, delay: float = 0.015,
                 k_v: float = 2.0, k_lat: float = 1.0, k_head: float = 1.0, s_max: float = 0.4189,
                 lut=None, grip: int = 0, wheelbase: float = 0.3302):
        self.n, self.device = n, device
        self.h = PS.DT / PS.SUB
        self.motor_tau, self.servo_tau, self.delay = motor_tau, servo_tau, delay
        self.k_v, self.k_lat, self.k_head, self.s_max = k_v, k_lat, k_head, s_max
        # MAP's steering table (11-09): the plan's steering is the kinematic bicycle's; the car understeers (at 7 m/s it
        # turns 30 % of that), so the feedforward is the table's steering for the plan's curvature, as MAP's is
        self.lut, self.grip, self.L = lut, grip, wheelbase
        m = PS.N_STEP * PS.SUB + 1
        z = lambda: torch.zeros(n, m, device=device)
        self.X, self.Y, self.PSI, self.V, self.D = z(), z(), z(), z(), z()
        self.k = torch.zeros(n, dtype=torch.long, device=device)      # control steps since each row's plan began

    def set_plan(self, rows, x, y, psi, v, delta) -> None:
        self.X[rows], self.Y[rows], self.PSI[rows] = x.float(), y.float(), psi.float()
        self.V[rows], self.D[rows] = v.float(), delta.float()
        self.k[rows] = 0

    def _at(self, A: torch.Tensor, f: torch.Tensor):
        """the value and the slope of each row's dense profile A at fractional grid index f (clamped to the plan)"""
        last = A.shape[1] - 1
        f = f.clamp(0.0, float(last))
        i0 = f.floor().long().clamp(max=last - 1)
        w = (f - i0.float())[:, None]
        a0 = A.gather(1, i0[:, None]); a1 = A.gather(1, (i0 + 1)[:, None])
        val = (a0 * (1 - w) + a1 * w)[:, 0]
        slope = ((a1 - a0)[:, 0] / self.h)
        past = f >= float(last)                                        # beyond the plan's end: hold, no rate
        return val, torch.where(past, torch.zeros_like(slope), slope)

    def _table_steer(self, d_kin: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """the steering the car needs for the curvature the kinematic steering d_kin means, at speed v (MAP's table;
        below 0.5 m/s, the table's first column, the kinematic one)"""
        acc = v * v * torch.tan(d_kin) / self.L
        g = torch.full_like(v, self.grip, dtype=torch.long)
        d = self.lut.lookup(acc, v.abs().clamp_min(0.5), g, interp="linear")
        slow = v.abs() < 0.5
        return torch.where(slow, d_kin, torch.nan_to_num(d, nan=0.0)).clamp(-self.s_max, self.s_max)

    def command(self, st: torch.Tensor) -> torch.Tensor:
        """st (n, >= 4): x, y, yaw, vx (the car's estimate of itself). (n, 2) steering [rad], speed [m/s]."""
        kf = self.k.float()
        f_lead = kf + (self.delay + 0.5 * self.h) / self.h
        v_ff, a_ff = self._at(self.V, f_lead)
        d_ff, w_ff = self._at(self.D, f_lead)
        if self.lut is not None:
            d_ff, w_ff = self._table_steer(d_ff, v_ff), (self._table_steer(d_ff + w_ff * self.h, v_ff + a_ff * self.h)
                                                       - self._table_steer(d_ff, v_ff)) / self.h
        # the error to where the plan is now, in the plan's frame
        x_p, _ = self._at(self.X, kf); y_p, _ = self._at(self.Y, kf); psi_p, _ = self._at(self.PSI, kf)
        v_p, _ = self._at(self.V, kf)
        ex, ey = st[:, 0] - x_p, st[:, 1] - y_p
        e_lat = -torch.sin(psi_p) * ex + torch.cos(psi_p) * ey          # + = the car is left of the plan
        e_head = torch.remainder(st[:, 2] - psi_p + math.pi, 2 * math.pi) - math.pi
        sgn = torch.where(v_p < 0, -torch.ones_like(v_p), torch.ones_like(v_p))   # backwards, steering turns the other way
        speed = v_ff + self.motor_tau * a_ff + self.k_v * (v_p - st[:, 3])
        steer = d_ff + self.servo_tau * w_ff - sgn * (self.k_lat * e_lat + self.k_head * e_head)
        self.k += 1
        return torch.stack([steer.clamp(-self.s_max, self.s_max), speed], 1)


def to_car(env, rows: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    """(steering, speed) commands meant for the car -> what goes to its servo / VESC, through the car's calibration as MAP's
    commands go (`GymEnv.tracker_cal`: the servo's trim and gain, the VESC's speed gain, measured once). 11-09: sent raw,
    the simulator's trim error (+-0.03 rad) bent every plan 40-70 cm off in 1 s at 4 m/s."""
    cal = getattr(env, "tracker_cal", None)
    if cal is None:
        return cmd
    c = cal[rows]
    return torch.stack([((cmd[:, 0] - c[:, 0]) / c[:, 1]).clamp(-float(env.s_max), float(env.s_max)), cmd[:, 1] / c[:, 2]], 1)
