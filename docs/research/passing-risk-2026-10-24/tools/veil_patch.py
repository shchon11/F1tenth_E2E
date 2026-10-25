"""Research (2026-10-25): mixed pixels at object edges in the simulator's scanner (imported by the research runners when VEIL is set).

The real venue boxes (the user: they were boxes) are read by the policy's nearest-car output as a car about half the time and
planned around far less than the simulator's boxes (plans cross them 43 % vs 4-6 % at 1.5-2.5 m). One difference the scanner
model lacks (`lidar.Lidar._post`: noise, spikes, dropouts): a beam that grazes an object's edge returns a range BETWEEN the
object and what is behind it. On the venue bags 21-41 % of the boxes show such a veil, 0.25-0.61 beams per edge
(real_object_edges.py). Here: at every edge of a prop or car hit (HIT_TALL / HIT_CAR) whose neighbour beam returns from at
least JUMP m further back, that neighbour is, with probability VEIL, replaced by a range uniformly 20-80 % of the way from
the object to the background. Patches `Lidar._post` (the class, so it must be imported before the env is built)."""
import os

import torch

from f1sim import lidar as _lidar

PV = float(os.environ.get("VEIL", "0.5"))
JUMP = float(os.environ.get("VEIL_JUMP", "0.3"))
_orig = _lidar.Lidar._post


def _post(self, r_true, typ, origin, dh, k, tid, P, car_poro=None):
    r, rt, ty = _orig(self, r_true, typ, origin, dh, k, tid, P, car_poro)
    obj = (ty == _lidar.HIT_TALL) | (ty == _lidar.HIT_CAR)
    rmax = P["range_max"][:, None]
    valid = torch.isfinite(r) & (r < rmax - 1e-2) & (r > 0.05)
    out = r
    for shift in (1, -1):
        # beam j = i + shift is background behind an object edge at beam i
        r_obj = torch.roll(r, shift, 1)                     # at j: the range of its neighbour i
        obj_i = torch.roll(obj & valid, shift, 1)
        edge = obj_i & ~obj & valid & (r > r_obj + JUMP)
        u = torch.rand_like(r)
        frac = 0.2 + 0.6 * torch.rand_like(r)
        out = torch.where(edge & (u < PV), r_obj + frac * (r - r_obj), out)
    return out, rt, ty


_lidar.Lidar._post = _post
