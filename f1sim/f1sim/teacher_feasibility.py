"""Teacher-only speed certification on the final curvature plan.

This bounds a desired spatial profile, not reachability from an overspeed/sliding
state. It never installs or changes the student controller.
"""
from __future__ import annotations

import torch


def project_speeds(k, length, v0, v1, parameters, vehicle, spec, mu_front, mu_rear):
    """Return certified endpoints and their minimum retained request fraction.

Curvature knots and speed are linear in arc. Interval extrema conservatively
bound forces, positive axle loads and steering slew between samples, including
both sides of every knot. A bounded grid followed by bracket refinement chooses
a feasible scale; no global time-optimality claim is made for this projection.
"""
    def param(name):
        value = parameters.get(name, getattr(vehicle, name)) if parameters is not None else getattr(vehicle, name)
        return torch.as_tensor(value, device=k.device, dtype=k.dtype).expand_as(v0)[:, None, None]

    lf, lr, height = param('lf'), param('lr'), param('h')
    wb = lf + lr
    split = param('drive_split_r') if vehicle.wheel_model else torch.ones_like(lf)
    roll, drag = param('c_roll'), param('c_drag')
    acceleration = param('a_max').clamp(max=spec.a_max)
    braking = param('a_brake').clamp(max=spec.a_brake)
    switch = param('v_switch')
    steer_max, steer_rate = param('s_max'), param('sv_max')
    length = length.clamp_min(1e-6)
    # Eight intervals per curvature segment: every knot appears on both sides.
    u = torch.linspace(0, 1, 9, device=k.device, dtype=k.dtype)
    segments = k.shape[1] - 1
    fractions = (torch.arange(segments, device=k.device, dtype=k.dtype)[:, None] + u[None]) / segments
    curve = k[:, :-1, None] + (k[:, 1:] - k[:, :-1])[:, :, None] * u
    curve_lo = curve[:, :, :-1].reshape(k.shape[0], -1)[:, None]
    curve_hi = curve[:, :, 1:].reshape(k.shape[0], -1)[:, None]
    k_abs = torch.maximum(curve_lo.abs(), curve_hi.abs())
    k_min = torch.where(curve_lo * curve_hi <= 0, torch.zeros_like(curve_lo),
                        torch.minimum(curve_lo.abs(), curve_hi.abs()))
    dk = ((k[:, 1:] - k[:, :-1]).abs() * segments / length[:, None])
    dk = dk[:, :, None].expand(-1, -1, 8).reshape(k.shape[0], -1)[:, None]
    f_lo = fractions[:, :-1].reshape(1, 1, -1)
    f_hi = fractions[:, 1:].reshape(1, 1, -1)
    target0, target1 = v0.clamp_min(0), v1.clamp_min(0)
    front = mu_front[:, None, None]
    rear = mu_rear[:, None, None]

    def feasible(scale, end_scale=None):
        a = (target0[:, None] * scale)[:, :, None]
        b = (target1[:, None] * (scale if end_scale is None else end_scale))[:, :, None]
        dv = (b - a) / length[:, None, None]
        vl, vr = a + (b - a) * f_lo, a + (b - a) * f_hi
        vlo, vhi = torch.minimum(vl, vr), torch.maximum(vl, vr)
        al, ar = vl * dv, vr * dv
        alo, ahi = torch.minimum(al, ar), torch.maximum(al, ar)
        rlo = roll * torch.tanh(vlo / .05) + drag * vlo.square()
        rhi = roll * torch.tanh(vhi / .05) + drag * vhi.square()
        longitudinal = torch.maximum((alo + rlo).abs(), (ahi + rhi).abs())
        lateral = vhi.square() * k_abs
        valid = (ahi + rhi <= acceleration * (switch / vhi.clamp_min(1e-6)).clamp(max=1) + 1e-6)
        valid = valid & (alo + rlo >= -braking - 1e-6)
        # Tracker authority is net acceleration; motor/current bounds above are
        # force-equivalent acceleration before drag. They are not interchangeable.
        valid = valid & (ahi <= spec.a_max + 1e-6) & (alo >= -spec.a_brake - 1e-6)
        for share, load, transfer, grip in (
                (1 - split, lr / wb, -height / wb, front),
                (split, lf / wb, height / wb, rear)):
            normal = 9.81 * load + torch.minimum(transfer * alo, transfer * ahi)
            force2 = (share * longitudinal).square() + (load * lateral).square()
            valid = valid & (normal > 0) & (force2 <= (grip * normal).square() * (1 + 1e-6))
        effective = wb + spec.k_us * vhi.square()
        valid = valid & (torch.atan(effective * k_abs) <= steer_max + 1e-6)
        # Absolute derivative bound for atan((L+k_us*v²)*k), times ds/dt=v.
        denominator = 1 + ((wb + spec.k_us * vlo.square()) * k_min).square()
        slew = vhi * (effective * dk + 2 * spec.k_us * vhi * dv.abs() * k_abs) / denominator
        valid = valid & (slew <= steer_rate + 1e-6)
        return valid.all(-1)

    grid = torch.linspace(0, 1, 17, device=k.device, dtype=k.dtype)[None].expand(k.shape[0], -1)
    ok = feasible(grid)
    low = torch.where(ok, grid, torch.zeros_like(grid)).amax(1)
    high = (low + 1 / 16).clamp(max=1)
    for _ in range(8):
        middle = (low + high) * .5
        good = feasible(middle[:, None])[:, 0]
        low = torch.where(good, middle, low)
        high = torch.where(good, high, middle)
    scale = torch.where(feasible(low[:, None])[:, 0], low, torch.zeros_like(low))
    # Uniform scaling can needlessly slow the near endpoint solely because a
    # later bend limits the far endpoint. Recover each endpoint independently,
    # accepting only certified, pointwise-faster profiles. This bounded five-
    # point search adds no control knob or iterative online optimizer.
    recovery = torch.linspace(0, 1, 5, device=k.device, dtype=k.dtype)[None]
    choices = scale[:, None] + (1 - scale[:, None]) * recovery
    near_ok = feasible(choices, scale[:, None])
    near = torch.where(near_ok, choices, scale[:, None]).amax(1)
    far_ok = feasible(near[:, None], choices)
    far = torch.where(far_ok, choices, scale[:, None]).amax(1)
    certified = feasible(near[:, None], far[:, None])[:, 0]
    near = torch.where(certified, near, scale)
    far = torch.where(certified, far, scale)
    return target0 * near, target1 * far, torch.minimum(near, far)
