"""prototype (10-26): a box standing against a wall is the free END of a long wall segment that sticks out from the wall line"""
import torch
from f1sim.obstacle_speed_limit import DEFAULT as P


def protrusion_keep(r, ok, px, py, end_len=0.6, fit_len=0.6, depth=0.15, p=P):
    """(B, n): beams of the free end (<= end_len from the end, standing in front of what is beyond it) of a segment longer than
    p.max_extent whose end part lies >= depth off the line through the segment's next fit_len of points, toward the car."""
    B, n = r.shape
    keep = torch.zeros_like(ok)
    rf = torch.where(ok, r, torch.full_like(r, 30.0))
    for b in range(B):                               # prototype: a loop; the shipped form would be batched
        brk = torch.cat([torch.ones(1, dtype=torch.bool), ((rf[b, 1:] - rf[b, :-1]).abs() > p.break_jump) | ~ok[b, 1:] | ~ok[b, :-1]])
        seg = torch.cumsum(brk.long(), 0)
        for sid in torch.unique(seg[ok[b]]).tolist():
            idx = ((seg == sid) & ok[b]).nonzero().flatten()
            if idx.numel() < 6:
                continue
            f, l = int(idx[0]), int(idx[-1])
            x, y = px[b, idx], py[b, idx]
            if float(torch.hypot(x[-1] - x[0], y[-1] - y[0])) < p.max_extent:
                continue                              # a small cluster: the existing rule's business
            for end, order in ((l, torch.arange(idx.numel() - 1, -1, -1)), (f, torch.arange(idx.numel()))):
                beyond = [end + j if end == l else end - j for j in range(1, p.stand_look + 1)]
                beyond = [j for j in beyond if 0 <= j < n]
                vals = [float(r[b, j]) for j in beyond if bool(ok[b, j])]
                bg = max(vals) if vals else 30.0
                if not bg > float(rf[b, end]) + p.stand:
                    continue                          # this end does not stand in front of anything
                xo, yo = x[order], y[order]
                dist_from_end = torch.hypot(xo - xo[0], yo - yo[0])
                E = dist_from_end <= end_len
                F = (dist_from_end > end_len) & (dist_from_end <= end_len + fit_len)
                if int(F.sum()) < 4 or int(E.sum()) < 3:
                    continue
                fx, fy = xo[F], yo[F]
                cx, cy = fx.mean(), fy.mean()
                u, s_, v = torch.linalg.svd(torch.stack([fx - cx, fy - cy], 1), full_matrices=False)
                nx, ny = float(v[1, 0]), float(v[1, 1])            # the wall line's normal
                if nx * (0.0 - float(cx)) + ny * (0.0 - float(cy)) < 0:  # toward the car (the origin)
                    nx, ny = -nx, -ny
                off = (xo[E] - cx) * nx + (yo[E] - cy) * ny
                if float(off.max()) >= depth:
                    sel = idx[order][E][off >= 0.5 * depth]
                    keep[b, sel] = True
    return keep & ok
