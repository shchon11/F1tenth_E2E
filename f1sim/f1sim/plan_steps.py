"""The planner's output (11-07 redesign, agreed with the user step by step): a plan of 10 positions, 0.1 s apart over
1.0 s, that the car can actually drive.

The policy outputs 20 numbers in [-1, 1], two per 0.1 s step: how to turn (the steering angle at the step's end) and how
fast to go (the signed speed at the step's end). Each number is spread linearly over what is REACHABLE from the previous
step under the car's limits, so every output is a plan the car can drive, and different outputs are different plans
(no clamp: a clamped output range is a dead zone where the plan stops changing and the gradient with it):

  steering   within the servo's rate (`sv_max` x 0.1 s) of the last step, inside the full lock `s_max`, and inside the
             tyres' grip at the slowest speed the step can brake to (MAP's steering table: the steady-state lateral
             acceleration a_max(v), curvature a_max / v^2) -- turn first, so the speed range below is never empty
  speed      within the motor's acceleration / the brakes' deceleration of the last step (reversing the roles when
             backing), inside [-V_REV, V_FWD], and no faster than the tyres hold the chosen curvature

Within a step speed and steering change linearly; the poses are integrated on the control grid (0.025 s, a kinematic
bicycle), so the first step's four poses are exactly where the car is at the next four control steps when the plan is
executed as given (training, "ideal execution"); on the car MAP follows the 10 points and their speeds.

The start of every plan is the car's state now: its signed speed and its steering angle."""
from dataclasses import dataclass
from typing import Dict

import torch

N_STEP = 10               # plan points
DT = 0.1                  # [s] between points
SUB = 4                   # control steps per point (0.025 s)
V_FWD = 10.0              # [m/s] top planned speed (`EnvConfig.v_max_policy`)
V_REV = 2.0               # [m/s] top reverse speed (`EnvConfig.reverse_speed`)
ACT_DIM = 2 * N_STEP      # per step: turn, speed


@dataclass
class Limits:
    """The car's limits, from the simulator's nominal car and MAP's steering table (no other car parameter)."""
    s_max: float            # [rad] full lock
    sv_max: float           # [rad/s] servo rate
    a_acc: float            # [m/s^2] motor
    a_brake: float          # [m/s^2] brakes
    wheelbase: float        # [m]
    v_grid: torch.Tensor    # (G,) [m/s] speeds of the tyre table
    k_tyre: torch.Tensor    # (G,) [1/m] curvature the tyres hold at each speed, non-increasing

    @staticmethod
    def from_nominal(nominal: Dict, lut, grip_level: int, device) -> "Limits":
        """`nominal` the car's parameter dict, `lut` MAP's `SteerLUT`, `grip_level` the table to read (nominal grip)"""
        L = float(nominal["lf"]) + float(nominal["lr"])
        s_max = float(nominal["s_max"])
        k_lock = float(torch.tan(torch.tensor(s_max))) / L
        a_lat = torch.nan_to_num(lut.table[grip_level].float(), nan=-1.0).amax(0).clamp_min(0.1).cpu()   # (V,)
        vt = lut.vels.float().cpu()
        v_grid = torch.linspace(0.0, V_FWD + 2.0, 241)
        a_g = torch.from_numpy(__import__("numpy").interp(v_grid.numpy(), vt.numpy(), a_lat.numpy())).float()
        k = torch.minimum(torch.full_like(v_grid, k_lock), a_g / v_grid.clamp_min(1e-3) ** 2)
        k = torch.cummin(k, 0).values                       # non-increasing in speed, so the inverse below is one-valued
        return Limits(s_max, float(nominal["sv_max"]), float(nominal["a_max"]), float(nominal["a_brake"]), L,
                      v_grid.to(device), k.to(device))

    def kappa_at(self, v: torch.Tensor) -> torch.Tensor:
        """curvature the tyres hold at speed |v| (linear in the table)"""
        s = v.abs().clamp(max=float(self.v_grid[-1]))
        g = (s / float(self.v_grid[1] - self.v_grid[0]))
        i0 = g.floor().long().clamp(max=self.v_grid.numel() - 2)
        w = g - i0.to(g.dtype)
        return self.k_tyre[i0] * (1 - w) + self.k_tyre[i0 + 1] * w

    def speed_for(self, kappa: torch.Tensor) -> torch.Tensor:
        """the highest speed at which the tyres hold |kappa| (inf below the curvature of the fastest table entry)"""
        k = kappa.abs()
        kt = self.k_tyre.flip(0)                             # non-decreasing
        vg = self.v_grid.flip(0)
        j = torch.searchsorted(kt.contiguous(), k.contiguous().reshape(-1)).reshape(k.shape)
        j = j.clamp(1, kt.numel() - 1)
        k0, k1, v0, v1 = kt[j - 1], kt[j], vg[j - 1], vg[j]
        w = ((k - k0) / (k1 - k0).clamp_min(1e-9)).clamp(0.0, 1.0)
        v = v0 + (v1 - v0) * w
        return torch.where(k <= kt[0], torch.full_like(v, float("inf")), v)


def steer_to_kappa(delta: torch.Tensor, lim: Limits) -> torch.Tensor:
    return torch.tan(delta) / lim.wheelbase


def kappa_to_steer(kappa: torch.Tensor, lim: Limits) -> torch.Tensor:
    return torch.atan(kappa * lim.wheelbase)


def _spread(a: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo + (a.clamp(-1.0, 1.0) + 1.0) * 0.5 * (hi - lo)


def step_targets(action: torch.Tensor, v0: torch.Tensor, d0: torch.Tensor, lim: Limits):
    """(B, 20) action, start speed v0 (B,) [m/s, signed] and steering d0 (B,) [rad] -> the steering and speed at the end
    of every step, (B, N) each, and how far each step's steering had to unwind past its range (B, N) [rad] (0 unless the
    start was already outside the tyres' grip)"""
    a_turn, a_speed = action[:, :N_STEP], action[:, N_STEP:]
    ds = lim.sv_max * DT
    v, d = v0.float(), d0.float()
    ds_, vs_, forced = [], [], []
    for k in range(N_STEP):
        # the slowest the car can be at the step's end, and so the most turn the tyres can hold there
        fwd = v >= 0
        v_lo = torch.where(fwd, v - lim.a_brake * DT, v - lim.a_acc * DT).clamp(min=-V_REV)
        v_hi = torch.where(fwd, v + lim.a_acc * DT, v + lim.a_brake * DT).clamp(max=V_FWD)
        slowest = torch.where((v_lo <= 0) & (v_hi >= 0), torch.zeros_like(v), torch.minimum(v_lo.abs(), v_hi.abs()))
        d_grip = torch.minimum(kappa_to_steer(lim.kappa_at(slowest), lim), torch.full_like(v, lim.s_max))
        lo = torch.maximum(d - ds, -d_grip)
        hi = torch.minimum(d + ds, d_grip)
        # a start outside the grip it can reach (the car turned sharply at speed): unwind at the servo's rate
        bad = lo > hi
        edge = torch.where(d > 0, d - ds, d + ds)
        lo = torch.where(bad, edge, lo); hi = torch.where(bad, edge, hi)
        forced.append(torch.where(bad, (d.abs() - ds - d_grip).clamp_min(0), torch.zeros_like(d)))
        d = _spread(a_turn[:, k], lo, hi)
        # speed: inside the motor / brakes and no faster than the tyres hold this turn
        # (never below `slowest`: d is inside the grip at `slowest` by construction, and the table is flat at low speed
        # -- full lock, ~1.24 1/m from 0.5 to ~2 m/s -- where inverting it is ill-posed: a rounding error of tan/atan
        # turned a cap of 2 m/s into 0.5 and forced a 1.5 m/s speed jump in one step)
        vcap = torch.maximum(lim.speed_for(steer_to_kappa(d, lim)), slowest)
        hi_v = torch.minimum(v_hi, vcap)
        lo_v = torch.maximum(v_lo, -vcap)
        hi_v = torch.maximum(hi_v, lo_v)                       # a forced unwind can leave no room: take the slowest
        v = _spread(a_speed[:, k], lo_v, hi_v)
        ds_.append(d); vs_.append(v)
    return torch.stack(ds_, 1), torch.stack(vs_, 1), torch.stack(forced, 1)


def roll(action: torch.Tensor, v0: torch.Tensor, d0: torch.Tensor, lim: Limits) -> Dict[str, torch.Tensor]:
    """The plan in the car's frame now: dense poses on the control grid, (B, N SUB + 1) each of x, y, psi, v, delta, t
    (index 0 = now), and the 10 points `pts` (B, N, 2) / their speeds `pv` (B, N)."""
    d_end, v_end, forced = step_targets(action, v0, d0, lim)
    B = action.shape[0]
    dev = action.device
    h = DT / SUB
    x = torch.zeros(B, device=dev); y = torch.zeros(B, device=dev); psi = torch.zeros(B, device=dev)
    v_prev, d_prev = v0.float(), d0.float()
    X, Y, P, V, D = [x], [y], [psi], [v_prev], [d_prev]
    for k in range(N_STEP):
        for j in range(1, SUB + 1):
            w0, w1 = (j - 1) / SUB, j / SUB
            va = v_prev + (v_end[:, k] - v_prev) * (0.5 * (w0 + w1))       # the substep's mean speed and steering
            da = d_prev + (d_end[:, k] - d_prev) * (0.5 * (w0 + w1))
            kap = steer_to_kappa(da, lim)
            dpsi = kap * va * h
            mid = psi + 0.5 * dpsi
            x = x + va * h * torch.cos(mid); y = y + va * h * torch.sin(mid); psi = psi + dpsi
            X.append(x); Y.append(y); P.append(psi)
            V.append(v_prev + (v_end[:, k] - v_prev) * w1); D.append(d_prev + (d_end[:, k] - d_prev) * w1)
        v_prev, d_prev = v_end[:, k], d_end[:, k]
    out = {"x": torch.stack(X, 1), "y": torch.stack(Y, 1), "psi": torch.stack(P, 1), "v": torch.stack(V, 1),
           "delta": torch.stack(D, 1), "forced": forced}
    out["t"] = torch.arange(N_STEP * SUB + 1, device=dev, dtype=torch.float32) * h
    idx = torch.arange(1, N_STEP + 1, device=dev) * SUB
    out["pts"] = torch.stack([out["x"][:, idx], out["y"][:, idx]], -1)
    out["pv"] = out["v"][:, idx]
    return out
