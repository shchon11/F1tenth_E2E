"""Batched single-track vehicle dynamics (torch).

State layout (B, 8): x, y, yaw, vx, vy, yaw_rate, steer, omega_r
  x, y, yaw   : rear-axle-centered? No -- CoG position/heading in world frame
  vx, vy      : body-frame velocity at CoG
  steer       : actual front wheel angle (servo state)
  omega_r     : rear-axle angular speed [rad/s]. The first seven columns are exactly what they
                were; this one was appended, never inserted, so every `state[:, :3]`-style reader
                keeps working.

Model: dynamic single-track with Pacejka lateral tire forces, longitudinal load
transfer, rear-wheel longitudinal force with friction ellipse, drag and rolling
resistance. Blended with a kinematic bicycle at low speed where the slip-angle
formulation degenerates.

Two longitudinal models live here, chosen by `vehicle.wheel_model`:

* **off** (the model before 2026-09-13) -- the rear longitudinal force is set directly from the
  commanded acceleration, `Fx_req = m * a_cmd`, capped on the friction circle. The wheel turns at
  whatever speed the body does (`omega_r = vx / r_w`, carried so the column means the same thing in
  both modes), so the wheel can neither spin nor lock and `odom.py`'s wheel speed is the body speed.
* **on** -- the rear axle is a rotating body. `kappa = (omega_r*r_w - vx) / max(|vx|, v_slip_eps)`
  drives a longitudinal magic formula that shares the friction circle with `Fy_r`, and the wheel
  carries `I_w * omega_dot = T_motor - r_w * Fx` with `T_motor = m * a_cmd * r_w` (so `a_max` /
  `a_brake` stay the *commanded* limits and the actuator interface does not change). The wheel
  equation is stiff -- its time constant is `I_w * max(|vx|, v_slip_eps) / (r_w^2 * dFx/dkappa)`,
  about 0.25 ms at a standstill on this car against a 1 ms substep -- so `omega_r` is advanced
  semi-implicitly on the local slope of the tyre curve (see `pacejka_slope`). Backward Euler on
  that one term is unconditionally stable and, because the slope goes to zero where the tyre is
  sliding, it does not damp the lock and spin transients this whole model exists to produce.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

G = 9.81
IX, IY, IYAW, IVX, IVY, IR, ISTEER, IOMEGA = range(8)
STATE_DIM = 8


def pacejka(alpha: torch.Tensor, B: torch.Tensor, C: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Normalized magic formula (peak = 1 at optimal slip)."""
    Ba = B * alpha
    return torch.sin(C * torch.atan(Ba - E * (Ba - torch.atan(Ba))))


def pacejka_slope(alpha: torch.Tensor, B: torch.Tensor, C: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """d/d(alpha) of `pacejka`, in closed form.

    Only used to damp the stiff wheel equation implicitly. It has to be the *local* slope, not the
    peak stiffness: a locked or spinning tyre is past the peak, where the true slope is ~0, and
    damping the wheel with the peak value there would quietly suppress exactly the transient the
    wheel model was added to produce. It can go negative past the peak (the curve falls away);
    the caller clamps at zero, because negative damping is not something to integrate implicitly.
    """
    Ba = B * alpha
    z = Ba - E * (Ba - torch.atan(Ba))
    dz = B * (1.0 - E) + E * B / (1.0 + Ba * Ba)
    return torch.cos(C * torch.atan(z)) * C / (1.0 + z * z) * dz


FRICTION_MARGIN = 1.1     # slack for load transfer and one-substep transients
#: How far past `v_max` the rear wheel's surface speed may run before it is clamped [m/s]. A
#: spinning wheel is bounded by the motor torque and the tyre anyway (and by the VESC loop, which
#: closes on it), so this only stops a pathological parameter draw from producing a wheel speed no
#: finite integrator can follow; it is never reached in normal operation.
WHEEL_OVERSPEED = 4.0


def motor_accel(a_cmd: torch.Tensor, v_pwr: torch.Tensor, P: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The acceleration the VESC actually applies, after its current and regen limits.

    `a_max` scaled by the power limit above `v_switch`, floored at `-a_brake`. Split out of
    `step_dynamics` because `sim.py` needs the same number to emulate `/sensors/core`
    `current_motor`: the current is proportional to the torque that is *applied*, and the raw
    output of the speed loop is not that -- a step command asks for 30 m/s^2 where the VESC will
    deliver 7.
    """
    a_lim = P["a_max"] * torch.clamp(P["v_switch"] / v_pwr.abs().clamp_min(1e-3), max=1.0)
    return torch.maximum(torch.minimum(a_cmd, a_lim), -P["a_brake"])


def step_dynamics(state: torch.Tensor, steer_target: torch.Tensor, a_cmd: torch.Tensor,
                  ax_prev: torch.Tensor, P: Dict[str, torch.Tensor], servo_tau: torch.Tensor,
                  dt: float, wheel: bool = False):
    """One physics substep. All per-env params in P are (B,) tensors.
    Returns (new_state, ax, ay) with body-frame *specific forces* for IMU emulation, load
    transfer and the pitch/roll model -- not the state derivatives (see ax_f below).

    `wheel` selects the longitudinal model (`vehicle.wheel_model`; see the module docstring). It is
    a Python bool rather than a tensor because it changes which forces exist, not their values, and
    because it is read at CUDA-graph capture time -- `viewer/graph_fastpath.py` guards it."""
    x, y, yaw = state[:, IX], state[:, IY], state[:, IYAW]
    vx, vy, r, delta = state[:, IVX], state[:, IVY], state[:, IR], state[:, ISTEER]
    omega_r = state[:, IOMEGA]
    m, Iz, lf, lr, h, mu = P["m"], P["Iz"], P["lf"], P["lr"], P["h"], P["mu"]
    L = lf + lr
    r_w = P["r_w"]
    v_wheel = omega_r * r_w

    # ---- steering servo: first-order lag with rate limit ----
    d_rate = ((steer_target - delta) / servo_tau).clamp(-P["sv_max"], P["sv_max"])
    delta = (delta + d_rate * dt).clamp(-P["s_max"], P["s_max"])

    # ---- longitudinal drive force (rear axle) with power limit ----
    # The VESC's current limit falls with *motor* speed, which is the wheel's, not the body's. With
    # the wheel model off the two are identical, so this reads exactly as it did.
    a_cmd = motor_accel(a_cmd, v_wheel if wheel else vx, P)
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
    mu_r = mu * P["mu_r_scale"]
    if wheel:
        # Longitudinal slip ratio. The denominator is regularised at `v_slip_eps` the way the
        # lateral branch blends below `v_blend_min`: kappa is a ratio to the ground speed and is
        # meaningless as that goes to zero, which is also where the wheel equation is stiffest.
        v_den = vx.abs().clamp_min(P["v_slip_eps"])
        kappa = (v_wheel - vx) / v_den
        # The rear axle's own longitudinal force, from its own slip ratio on its own normal load.
        # The car is 4WD off one motor, so the front axle makes the complementary share of the
        # drivetrain's force at the same drivetrain speed; `drive_split_r` is how the torque
        # divides. Modelling the rear alone -- every newton of drive and brake on an axle carrying
        # 48 % of the weight -- caps braking at mu*Fzr/m = 4.5 m/s^2, just under the 5.0 `a_brake`
        # allows, so with a wheel state every full brake command locked and the simulated slip rate
        # came out several times the recordings'. Treating the drivetrain as rigidly sharing the
        # whole car's grip removes the lock entirely, which is just as wrong. The split is what is
        # actually true: a fixed torque division against a load division that moves under braking.
        share_r = P["drive_split_r"]
        Fx_req = mu_r * Fzr * pacejka(kappa, P["B_x"], P["C_x"], P["E_x"])
    else:
        Fx_req = m * a_cmd
        share_r = torch.ones_like(Fx_req)
    Fy_req = mu_r * Fzr * pacejka(alpha_r, P["B_r"], P["C_r"], P["E_r"])
    F_lim = mu_r * Fzr
    F_mag = torch.sqrt(Fx_req ** 2 + Fy_req ** 2).clamp_min(1e-6)
    scale = torch.clamp(F_lim / F_mag, max=1.0)
    Fx_r, Fyr = Fx_req * scale, Fy_req * scale
    # What the body feels is the whole drivetrain's force; `Fx_r` is the rear axle's share of it,
    # after the circle. Identical to `Fx_r` with the wheel model off, where share_r is 1.
    # Two things about the front axle are approximated, both in the optimistic direction and both
    # stated rather than hidden: its share of the longitudinal force is not subtracted from `Fyf`
    # (a real 4WD car does lose front grip under braking), and its own longitudinal capacity is not
    # checked (under a hard launch the load leaves the front, so `mu*mu_f_scale*Fzf` can fall below
    # the share it is assumed to deliver, which makes the simulated launch slightly stronger and
    # its wheel spin slightly rarer than the car's). The lateral balance this file was calibrated to
    # is the rear axle's, and coupling the front would move the understeer gradient the existing
    # tests pin; both are recorded in docs/research/wheel-model-2026-09-13.md section 7.
    Fx_long = Fx_r / share_r

    ax_dyn = (Fx_long - Fyf * torch.sin(delta)) / m - a_res + vy * r
    ay_dyn = (Fyr + Fyf * torch.cos(delta)) / m - vx * r
    rdot_dyn = (lf * Fyf * torch.cos(delta) - lr * Fyr) / Iz

    # ---- kinematic model (derivatives, so the blend below mixes rates, not states) ----
    # The kinematic branch exists because the slip *angle* degenerates at low speed, not because
    # the longitudinal force does. With the wheel model on it therefore keeps the tyre's own Fx:
    # otherwise a launch below v_blend_min (which is where every launch starts, and where every
    # spin in the recordings lives) would push the body at the commanded acceleration no matter
    # what the wheel was doing, and the spin would be visible in the odometry while being invisible
    # in the body acceleration the guard compares it against.
    ax_kin = (Fx_long / m if wheel else a_cmd) - a_res
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

    # ---- rear axle ----
    if wheel:
        # `a_max` / `a_brake` stay the commanded limits (the actuator model is untouched); the
        # torque that realises them is m*a*r_w, i.e. the torque that would produce that body
        # acceleration if the tyre held. Whether it holds is what the line below decides.
        T_motor = m * a_cmd * r_w
        wdot = (T_motor - r_w * Fx_long) / P["I_w"]
        # Backward Euler on the one stiff term. d(wdot)/d(omega) = -r_w^2 * dFx/dv_wheel / I_w, and
        # dFx/dv_wheel = mu*Fzr*pacejka'(kappa)*scale / v_den. `scale` is carried because a tyre
        # held at the friction circle has no slope left, and `clamp_min(0)` because past the peak
        # the slope turns negative -- which is a real feature of the curve and a terrible thing to
        # feed an implicit step.
        slope = mu_r * Fzr * pacejka_slope(kappa, P["B_x"], P["C_x"], P["E_x"]) * scale / share_r
        lam = (r_w * r_w * slope / (P["I_w"] * v_den)).clamp_min(0.0)
        omega_n = omega_r + wdot * dt / (1.0 + lam * dt)
        omega_n = omega_n.clamp(P["v_min"] / r_w, (P["v_max"] + WHEEL_OVERSPEED) / r_w)
    else:
        # No wheel dynamics: the wheel rolls at the body speed by definition, so the column still
        # means "rear-axle angular speed" and `odom.py` reads the same number either way.
        omega_n = vx_n / r_w

    new = torch.stack([x_n, y_n, yaw_n, vx_n, vy_n, r_n, delta, omega_n], 1)
    return new, ax_f, ay


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi
