"""Plans that know the car's control delay (11-09, the user: "저만큼 제어딜레이 가정해서 출력하게하면 되잖아").

`plan_steps` plans the car's motion as if every command took effect at once: the steering and the speed at each step's
end, reached linearly. The car does not: the command reaches the VESC 15 ms later, the servo follows its target as a
first-order lag (`servo_tau` 0.08 s, at most `sv_max`), and the VESC's speed loop drives the motor at
(target - speed) / `motor_tau` (0.20 s), inside the current limit (`a_max`, falling as v_switch / v above `v_switch`) and
the brakes (`a_brake`), against rolling resistance and drag (`actuators.vesc_accel`, `dynamics.motor_accel` /
`step_dynamics`). A policy that re-plans every 0.1 s for an instant car over-corrects a lagging one.

Here the policy's 20 numbers are COMMANDS, two per 0.1 s step, and the plan is the car's response to them through that
delay and those lags (the nominal car), integrated every 5 ms, the poses by the kinematic bicycle about the CoG:

  steering command  the last steering command +- what the servo turns in a step (sv_max x 0.1 s), inside the steering
                    the tyres hold at the step's start speed (MAP's table), spread over that range as v2's steering is
                    (11-09: as a_turn x the grip, mean-reverting to straight, an untrained policy drove into the corners'
                    walls 40x as often as v2 and learned to stand still)
  speed command     the last speed command - a_brake x 0.1 s ... + a_max x 0.1 s (reversed when backing), inside [-V_REV,
                    V_FWD], spread over that range: v2's speed step on the COMMAND (11-09: from scratch every v3 form
                    learned to stand still -- the lagging car crashed while learning to drive; with v2's step sizes on
                    the commands a v2 policy's outputs mean the same commands, and v3 starts from it)

The plan starts from the car's state: its speed, its servo angle (the previous plan's, carried: nothing measures it) and
the last command sent, which is still the one in effect for the first 15 ms. The commands per control step (`cmd_d`,
`cmd_v`) are what goes to the car -- in training through the same model on the car's true state
(`Sim.ideal_lag`), on the car to the VESC with a small feedback on the pose error (`plan_follow.CommandFollower`)."""
import math
from dataclasses import dataclass
from typing import Dict

import torch

from .plan_steps import N_STEP, DT, SUB, V_FWD, V_REV, ACT_DIM, Limits, steer_to_kappa, kappa_to_steer   # noqa: F401

H = 0.005                 # [s] integration step
DV_CMD = 1.5              # [m/s] the speed command's span around the step's start speed
N_H = int(round(DT / SUB / H))   # integration steps per control step (5)


@dataclass
class Actuators:
    """the nominal car's command path (`params.ActuatorParams`, `VehicleParams` defaults)"""
    delay: float = 0.015         # [s] command -> VESC / servo
    servo_tau: float = 0.08      # [s]
    motor_tau: float = 0.20      # [s]
    a_max: float = 5.6           # [m/s^2] current limit
    v_switch: float = 6.0        # [m/s] above: a_max x v_switch / v
    a_brake: float = 5.0         # [m/s^2]
    c_roll: float = 0.1          # [m/s^2]
    c_drag: float = 0.01         # [1/m]
    sv_max: float = 3.2          # [rad/s] servo rate (the simulator's ideal rows; the plan reads `Limits`, the same)
    s_max: float = 0.4189        # [rad] full lock

    @staticmethod
    def nominal(nominal: Dict) -> "Actuators":
        a = Actuators()
        for k in ("a_max", "v_switch", "a_brake", "c_roll", "c_drag"):
            if k in nominal:
                setattr(a, k, float(nominal[k]))
        return a


class Turning:
    """How much the car turns for a steering angle at a speed: MAP's steering table (the simulator's nominal car, steady
    state, nominal grip), not the kinematic bicycle -- the car understeers: at 0.1 rad it turns 97 % of the bicycle's
    curvature at 1.4 m/s, 57 % at 4.3, 30 % at 7.2 (11-09). `eff(d, v)` is the kinematic steering that makes the table's
    curvature, so the poses stay a kinematic bicycle on it; `max_steer(v)` the steering of the table's largest settled
    lateral acceleration (the grip)."""

    def __init__(self, lut, grip: int, wheelbase: float, device):
        steers = lut.steers.float().to(device); vels = lut.vels.float().to(device)
        A = lut.table[grip].float().to(device)                          # (S, V) lateral acceleration, NaN unsettled
        ok = ~torch.isnan(A)
        A0 = torch.where(ok, A, torch.full_like(A, -1.0))
        top = A0.argmax(0)                                               # per speed: the steering of the most grip
        S = steers.numel()
        k = torch.arange(S, device=device)[:, None]
        # past the top a column saturates (more steering, no more turn): the table held flat
        A_f = torch.where(k <= top[None, :], torch.cummax(A0.clamp_min(0.0), 0).values, A0.gather(0, top[None, :]).expand_as(A0))
        self.steers, self.vels, self.A, self.L = steers, vels, A_f, wheelbase
        self.top_steer = steers[top]                                     # (V,)
        # host floats: the simulator's ideal rows call `eff` inside a CUDA graph capture, where reading a tensor syncs
        self.v_lo, self.v_hi, self.s_hi = float(vels[0]), float(vels[-1]), float(steers[-1])

    def _iv(self, v: torch.Tensor):
        V = self.vels.numel()
        vc = v.abs().clamp(self.v_lo, self.v_hi)
        hi = torch.searchsorted(self.vels, vc.contiguous()).clamp(1, V - 1)
        lo = hi - 1
        w = ((vc - self.vels[lo]) / (self.vels[hi] - self.vels[lo])).clamp(0.0, 1.0)
        return vc, lo, hi, w

    def max_steer(self, v: torch.Tensor) -> torch.Tensor:
        _, lo, hi, w = self._iv(v)
        return self.top_steer[lo] * (1 - w) + self.top_steer[hi] * w

    def eff(self, d: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        vc, lo, hi, w = self._iv(v)
        S = self.steers.numel()
        da = d.abs().clamp(max=self.s_hi)
        sh = torch.searchsorted(self.steers, da.contiguous()).clamp(1, S - 1)
        sl = sh - 1
        u = ((da - self.steers[sl]) / (self.steers[sh] - self.steers[sl])).clamp(0.0, 1.0)
        a_lo = self.A[sl, lo] * (1 - u) + self.A[sh, lo] * u
        a_hi = self.A[sl, hi] * (1 - u) + self.A[sh, hi] * u
        acc = a_lo * (1 - w) + a_hi * w
        kappa = acc / (vc * vc)
        return torch.sign(d) * torch.atan(kappa * self.L)


def servo_step(d: torch.Tensor, target: torch.Tensor, act: Actuators, lim: Limits, dt: float) -> torch.Tensor:
    rate = ((target - d) / act.servo_tau).clamp(-lim.sv_max, lim.sv_max)
    return (d + rate * dt).clamp(-lim.s_max, lim.s_max)


def motor_step(v: torch.Tensor, target: torch.Tensor, act: Actuators, dt: float) -> torch.Tensor:
    a = (target - v) / act.motor_tau
    a_lim = act.a_max * torch.clamp(act.v_switch / v.abs().clamp_min(1e-3), max=1.0)
    a = torch.maximum(torch.minimum(a, a_lim), torch.full_like(a, -act.a_brake))
    res = act.c_roll * torch.tanh(v / 0.05) + act.c_drag * v * v.abs()
    res = torch.sign(v) * torch.minimum(res.abs(), v.abs() / dt)
    return v + (a - res) * dt


def kin_step(x, y, psi, v, d, lim: Limits, dt: float, turn: "Turning" = None):
    tan_d = torch.tan(turn.eff(d, v) if turn is not None else d)
    beta = torch.atan(lim.lr / lim.wheelbase * tan_d)
    r = v * torch.cos(beta) * tan_d / lim.wheelbase
    vy = v * torch.tan(beta)
    mid = psi + 0.5 * r * dt
    return x + (v * torch.cos(mid) - vy * torch.sin(mid)) * dt, y + (v * torch.sin(mid) + vy * torch.cos(mid)) * dt, psi + r * dt


def grip_steer(v: torch.Tensor, lim: Limits, turn: "Turning" = None) -> torch.Tensor:
    """the largest steering angle the tyres hold at speed v (full lock at low speed)"""
    if turn is not None:
        return torch.minimum(turn.max_steer(v), torch.full_like(v, lim.s_max))
    return torch.minimum(kappa_to_steer(lim.kappa_at(v), lim), torch.full_like(v, lim.s_max))


def roll(action: torch.Tensor, v0: torch.Tensor, d0: torch.Tensor, c_prev: torch.Tensor, lim: Limits,
         act: Actuators, turn: "Turning" = None) -> Dict[str, torch.Tensor]:
    """(B, 20) action, the car's speed v0 and servo angle d0 (B,), the command in effect c_prev (B, 2) (steering, speed)
    -> the plan in the car's frame now: dense x, y, psi, v, delta, t (B, N SUB + 1) on the control grid, the 10 points
    `pts` (B, N, 2) and their speeds `pv`, the commands per control step `cmd_d` / `cmd_v` (B, N SUB)"""
    a_turn, a_speed = action[:, :N_STEP].clamp(-1, 1), action[:, N_STEP:].clamp(-1, 1)
    B = action.shape[0]
    dev = action.device
    x = torch.zeros(B, device=dev); y = torch.zeros(B, device=dev); psi = torch.zeros(B, device=dev)
    v, d = v0.float(), d0.float()
    prev_d, prev_v = c_prev[:, 0].float(), c_prev[:, 1].float()
    n_delay = int(round(act.delay / H))
    X, Y, P, V, D, CD, CV = [x], [y], [psi], [v], [d], [], []
    dd = lim.sv_max * DT
    for k in range(N_STEP):
        fwd = prev_v >= 0
        v_lo = torch.where(fwd, prev_v - lim.a_brake * DT, prev_v - lim.a_acc * DT).clamp(min=-V_REV)
        v_hi = torch.where(fwd, prev_v + lim.a_acc * DT, prev_v + lim.a_brake * DT).clamp(max=V_FWD)
        cv = v_lo + (a_speed[:, k] + 1.0) * 0.5 * (v_hi - v_lo)      # the step's speed command
        g = grip_steer(v, lim, turn)
        lo = torch.maximum(prev_d - dd, -g); hi = torch.minimum(prev_d + dd, g)
        bad = lo > hi                                     # the last command is past the grip now: unwind to its edge
        edge = torch.where(prev_d > 0, g, -g)
        lo = torch.where(bad, edge, lo); hi = torch.where(bad, edge, hi)
        cd_k = lo + (a_turn[:, k] + 1.0) * 0.5 * (hi - lo)
        for j in range(SUB):
            cd = cd_k                                          # the command sent at this control step
            for i in range(N_H):
                in_d, in_v = (prev_d, prev_v) if i < n_delay else (cd, cv)
                d = servo_step(d, in_d, act, lim, H)
                v = motor_step(v, in_v, act, H)
                x, y, psi = kin_step(x, y, psi, v, d, lim, H, turn)
            prev_d, prev_v = cd, cv
            CD.append(cd); CV.append(cv)
            X.append(x); Y.append(y); P.append(psi); V.append(v); D.append(d)
    out = {"x": torch.stack(X, 1), "y": torch.stack(Y, 1), "psi": torch.stack(P, 1), "v": torch.stack(V, 1),
           "delta": torch.stack(D, 1), "cmd_d": torch.stack(CD, 1), "cmd_v": torch.stack(CV, 1)}
    out["t"] = torch.arange(N_STEP * SUB + 1, device=dev, dtype=torch.float32) * (DT / SUB)
    idx = torch.arange(1, N_STEP + 1, device=dev) * SUB
    out["pts"] = torch.stack([out["x"][:, idx], out["y"][:, idx]], -1)
    out["pv"] = out["v"][:, idx]
    return out
