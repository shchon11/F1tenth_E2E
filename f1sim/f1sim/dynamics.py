"""Batched single-track vehicle dynamics (torch).

State layout (B, 7): x, y, yaw, vx, vy, yaw_rate, steer
  x, y, yaw   : rear-axle-centered? No -- CoG position/heading in world frame
  vx, vy      : body-frame velocity at CoG
  steer       : actual front wheel angle (servo state)

Model: dynamic single-track with Pacejka lateral tire forces, longitudinal load
transfer, rear-wheel longitudinal force with friction ellipse, drag and rolling
resistance. Blended with a kinematic bicycle at low speed where the slip-angle
formulation degenerates.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

G = 9.81
IX, IY, IYAW, IVX, IVY, IR, ISTEER = range(7)
STATE_DIM = 7


def pacejka(alpha: torch.Tensor, B: torch.Tensor, C: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Normalized magic formula (peak = 1 at optimal slip)."""
    Ba = B * alpha
    return torch.sin(C * torch.atan(Ba - E * (Ba - torch.atan(Ba))))


FRICTION_MARGIN = 1.1     # slack for load transfer and one-substep transients


def step_dynamics(state: torch.Tensor, steer_target: torch.Tensor, a_cmd: torch.Tensor,
                  ax_prev: torch.Tensor, P: Dict[str, torch.Tensor], servo_tau: torch.Tensor,
                  dt: float):
    """One physics substep. All per-env params in P are (B,) tensors.
    Returns (new_state, ax, ay) with body-frame *specific forces* for IMU emulation, load
    transfer and the pitch/roll model -- not the state derivatives (see ax_f below)."""
    x, y, yaw = state[:, IX], state[:, IY], state[:, IYAW]
    vx, vy, r, delta = state[:, IVX], state[:, IVY], state[:, IR], state[:, ISTEER]
    m, Iz, lf, lr, h, mu = P["m"], P["Iz"], P["lf"], P["lr"], P["h"], P["mu"]
    L = lf + lr

    # ---- steering servo: first-order lag with rate limit ----
    d_rate = ((steer_target - delta) / servo_tau).clamp(-P["sv_max"], P["sv_max"])
    delta = (delta + d_rate * dt).clamp(-P["s_max"], P["s_max"])

    # ---- longitudinal drive force (rear axle) with power limit ----
    a_lim = P["a_max"] * torch.clamp(P["v_switch"] / vx.abs().clamp_min(1e-3), max=1.0)
    a_cmd = torch.maximum(torch.minimum(a_cmd, a_lim), -P["a_brake"])
    # resistance (never reverses sign of vx within a step)
    a_res = P["c_roll"] * torch.tanh(vx / 0.05) + P["c_drag"] * vx * vx.abs()
    a_res = torch.sign(vx) * torch.minimum(a_res.abs(), vx.abs() / dt)

    # ---- dynamic model ----
    vx_safe = torch.where(vx.abs() < 0.3, torch.full_like(vx, 0.3) * torch.sign(vx + 1e-9), vx)
    alpha_f = delta - torch.atan2(vy + lf * r, vx_safe)
    alpha_r = -torch.atan2(vy - lr * r, vx_safe)
    Fzf = m * G * lr / L - m * h * ax_prev / L
    Fzr = m * G * lf / L + m * h * ax_prev / L
    Fzf, Fzr = Fzf.clamp_min(0.0), Fzr.clamp_min(0.0)
    Fyf = mu * P["mu_f_scale"] * Fzf * pacejka(alpha_f, P["B_f"], P["C_f"], P["E_f"])
    # rear tire: combined slip via friction circle -- requested (Fx, Fy) keeps its direction
    # and is capped in magnitude at mu*Fz (sliding tire), so hard throttle in a corner costs
    # lateral grip (power oversteer) but does not zero it out.
    Fx_req = m * a_cmd
    Fy_req = mu * P["mu_r_scale"] * Fzr * pacejka(alpha_r, P["B_r"], P["C_r"], P["E_r"])
    F_lim = mu * P["mu_r_scale"] * Fzr
    F_mag = torch.sqrt(Fx_req ** 2 + Fy_req ** 2).clamp_min(1e-6)
    scale = torch.clamp(F_lim / F_mag, max=1.0)
    Fx_r, Fyr = Fx_req * scale, Fy_req * scale

    ax_dyn = (Fx_r - Fyf * torch.sin(delta)) / m - a_res + vy * r
    ay_dyn = (Fyr + Fyf * torch.cos(delta)) / m - vx * r
    rdot_dyn = (lf * Fyf * torch.cos(delta) - lr * Fyr) / Iz

    # ---- kinematic model (derivatives, so the blend below mixes rates, not states) ----
    ax_kin = a_cmd - a_res
    k = lr / L
    tan_d = torch.tan(delta)
    beta = torch.atan(k * tan_d)
    sec2_d = 1.0 + tan_d * tan_d
    beta_dot = k * sec2_d * d_rate / (1.0 + (k * tan_d) ** 2)
    cb, sb = torch.cos(beta), torch.sin(beta)
    r_kin = vx * cb * tan_d / L
    vy_kin = vx * torch.tan(beta)
    rdot_kin = (ax_kin * cb * tan_d - vx * sb * beta_dot * tan_d + vx * cb * sec2_d * d_rate) / L
    vydot_kin = ax_kin * torch.tan(beta) + vx * beta_dot / (cb * cb)
    # weak relaxation of the integrated states toward the kinematic solution (tau = 50 ms),
    # only active in the low-speed regime; keeps them consistent after contacts / resets
    relax = 20.0
    rdot_kin = rdot_kin + relax * (r_kin - r)
    vydot_kin = vydot_kin + relax * (vy_kin - vy)

    # ---- blend derivatives ----
    w = ((vx.abs() - P["v_blend_min"]) / (P["v_blend_max"] - P["v_blend_min"])).clamp(0.0, 1.0)
    ax = w * ax_dyn + (1 - w) * ax_kin
    vydot = w * ay_dyn + (1 - w) * vydot_kin
    rdot = w * rdot_dyn + (1 - w) * rdot_kin

    # ---- friction bound on the blended result ----
    # The kinematic branch is a constraint, not a force model, and its relaxation term is pure
    # numerics: caught mid-slide at vx 1.5 m/s with vy -5.2, relax*(vy_kin - vy) alone asks for
    # 103 m/s^2 of lateral acceleration. Measured, that put 53 m/s^2 (5.5 g) on the emulated
    # accelerometer where the floor's limit is 8.8, and it wiped vy from -6.3 to -0.4 in 0.4 s --
    # so a simulated spin recovered on its own while the real car (map16x07 SPIN-AT-LIMIT) stayed
    # out of shape for four seconds. Whichever branch produced them, four tyre contact patches
    # carrying mg between them cannot pull more than mu*g, so bound the blended derivatives by it.
    # The dynamic branch is already inside this circle; the clamp only bites where the model lies.
    lim = mu * G * FRICTION_MARGIN
    fx_t, fy_t = ax - vy * r, vydot + vx * r                # provisional specific forces
    scale = (lim / torch.sqrt(fx_t * fx_t + fy_t * fy_t).clamp_min(1e-6)).clamp(max=1.0)
    ax = fx_t * scale + vy * r
    vydot = fy_t * scale - vx * r
    # same argument about the yaw moment: |Mz| <= mu*m*g*max(lf, lr)
    rdot_lim = mu * m * G * torch.maximum(lf, lr) / Iz * FRICTION_MARGIN
    rdot = rdot.clamp(-rdot_lim, rdot_lim)
    vx_n = (vx + ax * dt).clamp(P["v_min"], P["v_max"])
    vy_n = vy + vydot * dt
    r_n = r + rdot * dt
    # What the IMU (and the load transfer / pitch model) sees is specific force, not the state
    # derivative: in the rotating body frame f_x = vx_dot - vy*r and f_y = vy_dot + vx*r. Returning
    # vx_dot as "ax" left the longitudinal channel missing -vy*r, which is ~1.5 m/s^2 at 0.5 m/s of
    # sideslip and 3 rad/s of yaw -- exactly the cornering states the policy reads to infer grip.
    ax_f = ax - vy * r_n             # body longitudinal specific force
    ay = vydot + vx * r_n            # body lateral acceleration as an IMU would measure it

    # ---- pose integration (use updated body velocity, semi-implicit) ----
    c, s = torch.cos(yaw), torch.sin(yaw)
    x_n = x + (vx_n * c - vy_n * s) * dt
    y_n = y + (vx_n * s + vy_n * c) * dt
    yaw_n = wrap_angle(yaw + r_n * dt)

    new = torch.stack([x_n, y_n, yaw_n, vx_n, vy_n, r_n, delta], 1)
    return new, ax_f, ay


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi
