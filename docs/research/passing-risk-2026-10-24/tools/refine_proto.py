"""prototype: the nearest-car reading's position refined by the scan segment at it (vectorised)"""
import torch


def refine(reading, ranges, angles, mount_x=0.297, mount_y=0.0, range_max=30.0, gate=0.8, half_len=0.25,
           break_jump=0.15, max_extent=0.8):
    from f1sim.opponent_speed_limit import relative
    r = ranges.float()
    B, n = r.shape
    ang = angles.float()
    if ang.dim() == 1:
        ang = ang[None].expand_as(r)
    ex, cx, cy, _cl, _vl = relative(reading, mount_x, mount_y)
    ok = torch.isfinite(r) & (r > 0.05) & (r < range_max - 1e-3)
    rr = torch.where(ok, r, torch.zeros_like(r))
    px, py = mount_x + rr * torch.cos(ang), mount_y + rr * torch.sin(ang)
    rf = torch.where(ok, r, torch.full_like(r, 30.0))
    brk = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=r.device),
                     ((rf[:, 1:] - rf[:, :-1]).abs() > break_jump) | ~ok[:, 1:] | ~ok[:, :-1]], 1)
    seg = torch.cumsum(brk.long(), 1)
    d = torch.where(ok, torch.hypot(px - cx[:, None], py - cy[:, None]), torch.full_like(r, 1e9))
    dmin, i = d.min(1)
    same = (seg == seg.gather(1, i[:, None])) & ok
    big = torch.full_like(px, 1e9)
    xmin = torch.where(same, px, big).min(1).values
    xmax = torch.where(same, px, -big).max(1).values
    ymin = torch.where(same, py, big).min(1).values
    ymax = torch.where(same, py, -big).max(1).values
    ext = torch.hypot(xmax - xmin, ymax - ymin)
    found = ex & (dmin < gate) & (ext < max_extent)
    out = reading.clone().float()
    out[:, 1] = torch.where(found, (xmin + half_len - mount_x) / 5.0, out[:, 1])
    out[:, 2] = torch.where(found, (0.5 * (ymin + ymax) - mount_y) / 2.0, out[:, 2])
    return out, found
