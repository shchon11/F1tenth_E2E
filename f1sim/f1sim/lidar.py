"""Batched 2D LiDAR (Hokuyo UST-10LX class) with 3D beam geometry.

The scan plane follows the sprung body: roll/pitch (from the suspension model) and the
mounting misalignment tilt every beam, so beams hit the floor a few metres ahead under
braking, or pass over the duct hoses on the inside of a corner and return objects outside
the track. Ranges are the true 3D beam lengths, as the sensor reports them.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .track import TrackTensors
from .lidar_triton import HAVE_TRITON, trace3d_triton

HIT_NONE, HIT_DUCT, HIT_TALL, HIT_GROUND, HIT_CAR = 0, 1, 2, 3, 4


_CAR_SLICES = {}


def car_slices(device):
    """Outlines of the F1TENTH car mesh (assets/f1tenth_car.glb) at 1 cm height steps, in the car's
    CoG frame: z (L,), segments (L, S, 4) x0 y0 x1 y1, valid (L, S). Built by
    scripts/slice_car_mesh.py from the same model the viewer draws, so what the LiDAR hits is what
    is on screen -- the hand-placed part boxes read 0.60 m long from a rear quarter where the real
    car (competition bags) reads 0.33 m."""
    key = str(device)
    if key not in _CAR_SLICES:
        import os
        import numpy as np
        d = np.load(os.path.join(os.path.dirname(__file__), "assets", "f1tenth_car_slices.npz"))
        _CAR_SLICES[key] = (torch.as_tensor(d["z"], device=device), torch.as_tensor(d["segs"], device=device),
                            torch.as_tensor(d["valid"], device=device))
    return _CAR_SLICES[key]


def ray_slices_hits(origin: torch.Tensor, dh: torch.Tensor, k: torch.Tensor, poses: torch.Tensor,
                    scale: torch.Tensor, chunk: int = 32):
    """Beams against the car mesh outlines. origin (B,N,3), dh (B,N,2), k (B,N); poses (B,C,3) x, y,
    yaw of each car in view; scale (B,C) build size. Returns (3D range (B,N), hit (B,N)).

    The outline used for a beam is the one at the height the beam has where it reaches the car's
    centre, so a tilted scan plane sees the wheels low and the LiDAR tower high, as a level one at
    11 cm sees wheels and chassis side by side. Segments are streamed in chunks against every beam
    with a level mask -- gathering each beam's own outline would build a (B, N, S, 4) tensor, 4.5 GB
    at training batch size -- and the loop has no data-dependent branches, so it captures into the
    simulator's CUDA graph like the box path does."""
    z_lv, segs, valid = car_slices(k.device)
    B, N = k.shape; L, S, _ = segs.shape
    dz = float(z_lv[1] - z_lv[0]); z0 = float(z_lv[0])
    best = torch.full((B, N), float("inf"), device=k.device); hit = torch.zeros(B, N, dtype=torch.bool, device=k.device)
    slen = torch.sqrt(1.0 + k * k)
    for c in range(poses.shape[1]):
        px, py, pyaw = poses[:, c, 0:1], poses[:, c, 1:2], poses[:, c, 2:3]
        cy, sy = torch.cos(pyaw), torch.sin(pyaw)
        ox, oy = origin[..., 0] - px, origin[..., 1] - py
        lx, ly = ox * cy + oy * sy, -ox * sy + oy * cy                       # ray origin in the car frame
        dx, dy = dh[..., 0] * cy + dh[..., 1] * sy, -dh[..., 0] * sy + dh[..., 1] * cy
        t_c = (-lx * dx - ly * dy).clamp_min(0.0)                            # along-ray distance to the car centre
        z_c = origin[..., 2] + k * t_c
        lvl = ((z_c - z0) / dz).round().long()
        in_band = (lvl >= 0) & (lvl < L)
        sc_ = scale[:, c, None]                                              # (B,1)
        t_best = torch.full((B, N), float("inf"), device=k.device)
        for l in range(L):
            n_l = int(valid[l].sum())
            if n_l == 0:
                continue
            on_level = in_band & (lvl == l)
            for s0 in range(0, n_l, chunk):
                sg = segs[l, s0:s0 + chunk]                                  # (S',4), same for every beam
                ax, ay = sg[None, None, :, 0] * sc_[..., None], sg[None, None, :, 1] * sc_[..., None]
                bx, by = sg[None, None, :, 2] * sc_[..., None], sg[None, None, :, 3] * sc_[..., None]
                ex, ey = bx - ax, by - ay
                den = dx[..., None] * ey - dy[..., None] * ex
                den = torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
                wx, wy = ax - lx[..., None], ay - ly[..., None]
                t = (wx * ey - wy * ex) / den
                u = (wx * dy[..., None] - wy * dx[..., None]) / den
                ok = (t > 0.0) & (u >= 0.0) & (u <= 1.0) & on_level[..., None]
                t = torch.where(ok, t, torch.full_like(t, float("inf"))).min(-1).values
                t_best = torch.minimum(t_best, t)
        r = t_best * slen
        closer = torch.isfinite(t_best) & (r < best)
        best = torch.where(closer, r, best); hit = hit | torch.isfinite(t_best)
    return best, hit


def ray_box_hits(origin: torch.Tensor, dh: torch.Tensor, k: torch.Tensor, boxes: torch.Tensor, dims: torch.Tensor,
                 zrange: Optional[torch.Tensor] = None):
    """Oriented boxes (other cars, their rear detection boxes). origin (B,N,3), dh (B,N,2) unit
    horizontal, k (B,N) dz/ds; boxes (B,C,3) x, y, yaw of the C boxes each env can see; dims (B,C,3)
    length, width, height; zrange (B,C,2) bottom/top above the floor (default 0..height). Returns
    (3D range to the nearest box (B,N), hit (B,N) bool). A beam above the top or below the bottom
    where it reaches the box passes it (tilted scan planes)."""
    B, N = k.shape
    best = torch.full((B, N), float("inf"), device=k.device); hit = torch.zeros(B, N, dtype=torch.bool, device=k.device)
    slen = torch.sqrt(1.0 + k * k)
    eps = 1e-9
    for c in range(boxes.shape[1]):
        bx, by, byaw = boxes[:, c, 0:1], boxes[:, c, 1:2], boxes[:, c, 2:3]
        hl, hw = 0.5 * dims[:, c, 0:1], 0.5 * dims[:, c, 1:2]
        z_lo, z_hi = (torch.zeros_like(hl), dims[:, c, 2:3]) if zrange is None else (zrange[:, c, 0:1], zrange[:, c, 1:2])
        cy, sy = torch.cos(byaw), torch.sin(byaw)
        ox, oy = origin[..., 0] - bx, origin[..., 1] - by
        lx, ly = ox * cy + oy * sy, -ox * sy + oy * cy                       # ray origin in the box frame
        dx, dy = dh[..., 0] * cy + dh[..., 1] * sy, -dh[..., 0] * sy + dh[..., 1] * cy
        idx_ = 1.0 / torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
        idy_ = 1.0 / torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)
        tx1, tx2 = (-hl - lx) * idx_, (hl - lx) * idx_
        ty1, ty2 = (-hw - ly) * idy_, (hw - ly) * idy_
        t_in = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
        t_out = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
        z_in = origin[..., 2] + k * t_in
        ok = (t_out >= t_in) & (t_in > 0.0) & (z_in >= z_lo) & (z_in <= z_hi)
        r = t_in * slen
        closer = ok & (r < best)
        best = torch.where(closer, r, best); hit = hit | ok
    return best, hit


class Lidar:
    def __init__(self, track: TrackTensors, n_beams: int, fov: float, device, max_iters: int = 64, post_mode: str = "default",
                 compile: bool = True):
        self.track = track
        self.n = n_beams
        self.fov = fov
        self.device = torch.device(device)
        self.max_iters = max_iters
        self.post_mode = post_mode
        self._post_fast = self._post
        if post_mode == "reduce-overhead" and self.device.type == "cuda":
            try:
                self._post_fast = torch.compile(self._post, dynamic=False, mode="reduce-overhead")
            except Exception:
                self._post_fast = self._post
        self.angles = torch.linspace(-fov / 2, fov / 2, n_beams, device=self.device)   # (N,)
        # beam i is emitted at fraction time_frac[i] of the scan period before the scan timestamp
        self.time_frac = (fov / (2 * math.pi)) * (1.0 - torch.arange(n_beams, device=self.device) / (n_beams - 1))
        self._rays = self.rays
        if compile and self.device.type == "cuda":
            try:
                self._rays = torch.compile(self.rays, dynamic=False)
            except Exception:
                self._rays = self.rays

    @property
    def angle_increment(self) -> float:
        return self.fov / (self.n - 1)

    # ------------------------------------------------------------------ ray casting
    def trace(self, origin: torch.Tensor, direction_h: torch.Tensor, k: torch.Tensor, range_max: torch.Tensor,
              tid: Optional[torch.Tensor] = None, duct_scale: Optional[torch.Tensor] = None):
        """origin (B,N,3), direction_h (B,N,2) unit horizontal, k (B,N) dz/ds, range_max (B,), tid (B,),
        duct_scale (B,) per-env multiplier on the hose diameter -> (ranges, types)."""
        tr = self.track
        if tid is None:
            tid = torch.zeros(origin.shape[0], dtype=torch.long, device=origin.device)
        if HAVE_TRITON and origin.is_cuda:
            r, typ = trace3d_triton(origin, direction_h, k, range_max, tr, tid, self.max_iters, duct_scale)
        else:
            r, typ = self._trace_torch(origin, direction_h, k, range_max, tid, duct_scale)
        # Props merge HERE, at the common return, not inside `_trace_torch`: on CUDA the call above
        # goes straight to the Triton kernel and never reaches it, so merging there would have given
        # the two backends different worlds.
        if getattr(tr, "has_props", False):
            r, typ = self._merge_props(origin, direction_h, k, range_max, tid, r, typ)
        return r, typ

    def _merge_props(self, origin, direction_h, k, range_max, tid, r, typ):
        """Nearest-hit merge of the finite prop sections, the way cars are merged below.

        Props report as HIT_TALL rather than a type of their own. A box in the lane is a static solid
        obstacle, which is exactly what that type already means, and adding a sixth value would
        change what every consumer of `typ` sees -- observation encodings included -- for a rebuild
        of how obstacles are *drawn*. The prop's finiteness is in its geometry, not in its label."""
        from .prop_math import ray_prisms_hits
        tr = self.track
        poses, pn, pd, z_lo, z_hi = tr.props_for(tid)
        r_p, hit_p = ray_prisms_hits(origin, direction_h, k, poses, pn, pd, z_lo, z_hi)
        closer = hit_p & (r_p < r) & (r_p <= range_max[:, None])
        return (torch.where(closer, r_p.to(r.dtype), r),
                torch.where(closer, torch.full_like(typ, HIT_TALL), typ))

    def _trace_torch(self, origin, direction_h, k, range_max, tid, duct_scale=None):
        tr = self.track
        tidN = tid[:, None].expand(k.shape)
        res = tr.t_res[tidN]; half, minstep = 0.5 * res, 0.25 * res; duct_h = tr.t_duct_h[tidN]
        if duct_scale is not None:
            duct_h = duct_h * duct_scale[:, None]
        ox, oy, oz = origin[..., 0], origin[..., 1], origin[..., 2]
        dx, dy = direction_h[..., 0], direction_h[..., 1]
        slen = torch.sqrt(1.0 + k * k)
        s_max = range_max[:, None] / slen
        desc = k < -1e-6
        s_ground = torch.where(desc & (oz > 0), -oz / torch.where(desc, k, -torch.ones_like(k)), torch.full_like(k, 1e9))
        s = torch.zeros_like(k); done = torch.zeros_like(k, dtype=torch.bool); typ = torch.zeros_like(k, dtype=torch.int32)
        for it in range(self.max_iters):
            p = torch.stack([ox + dx * s, oy + dy * s], -1)
            z = oz + k * s
            d_duct, inside = tr.sample_edt(p, tidN, tr.edt_duct, with_inside=True)
            d_tall = tr.sample_edt(p, tidN, tr.edt_tall)
            d_duct = torch.where(inside, d_duct, torch.full_like(k, 1e9))
            above = z > duct_h
            d_eff = torch.where(above, d_tall, torch.minimum(d_duct, d_tall))
            hit = (d_eff <= half) & ~done
            hit_type = torch.where(above | (d_tall <= d_duct), 2, 1).to(torch.int32)
            step = (d_eff - half).clamp_min(minstep)
            s_desc = torch.where(desc, (z - duct_h) / torch.where(desc, -k, torch.ones_like(k)), torch.full_like(k, 1e9))
            step = torch.where(above, torch.minimum(step, s_desc.clamp_min(minstep)), step)
            s_next = s + step
            ground = ~done & ~hit & (s_next >= s_ground)
            far = ~done & ~hit & ~ground & (s_next >= s_max)
            typ = torch.where(hit, hit_type, typ); typ = torch.where(ground, torch.full_like(typ, 3), typ); typ = torch.where(far, torch.zeros_like(typ), typ)
            s_new = torch.where(hit, s, torch.where(ground, s_ground, torch.where(far, s_max, s_next)))
            s = torch.where(done, s, s_new)
            done = done | hit | ground | far
            # every iteration on the CPU, where reading the flag is free and the beams that are still
            # running are a shrinking minority. On CUDA the read is a device sync, so it stays coarse.
            if (done.all() if not done.is_cuda else (it % 8 == 7 and bool(done.all()))):
                break
        R = 0.5 * duct_h
        zh = oz + k * s
        w = torch.sqrt((R * R - (zh - R) ** 2).clamp_min(0.0))
        s = torch.where(typ == 1, s + (R - w), s)
        s = torch.minimum(s, s_max)
        typ = torch.where(done, typ, torch.zeros_like(typ))
        return s * slen, typ

    # ------------------------------------------------------------------ scan
    def rays(self, pose: torch.Tensor, att: torch.Tensor, P: Dict[str, torch.Tensor], per_beam: bool):
        """Build 3D rays. pose (B,N,3) or (B,3): x, y, yaw; att (B,N,2)/(B,2): roll, pitch.
        Returns origin (B,N,3), horizontal unit dir (B,N,2), slope k (B,N)."""
        B = pose.shape[0]
        if not per_beam:
            pose = pose[:, None, :].expand(B, self.n, 3); att = att[:, None, :].expand(B, self.n, 2)
        yaw = pose[..., 2]
        phi = att[..., 0] + P["mount_roll"][:, None]
        th = att[..., 1] + P["mount_pitch"][:, None]
        a = self.angles[None] + P["mount_yaw"][:, None]
        ca, sa = torch.cos(a), torch.sin(a)
        cph, sph, cth, sth = torch.cos(phi), torch.sin(phi), torch.cos(th), torch.sin(th)
        # body-frame beam after Rx(roll) then Ry(pitch); ROS: +pitch = nose down, +roll = right side down
        bx = ca * cth + sa * sph * sth
        by = sa * cph
        bz = -ca * sth + sa * sph * cth
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        wx = bx * cy - by * sy; wy = bx * sy + by * cy
        hn = torch.sqrt(wx * wx + wy * wy).clamp_min(1e-6)
        dh = torch.stack([wx / hn, wy / hn], -1)
        k = bz / hn
        mx, my, mz = P["mount_x"][:, None], P["mount_y"][:, None], P["mount_z"][:, None]
        t = my * sph + mz * cph
        ox_b = mx * cth + t * sth; oy_b = my * cph - mz * sph; oz_b = -mx * sth + t * cth
        origin = torch.stack([pose[..., 0] + ox_b * cy - oy_b * sy, pose[..., 1] + ox_b * sy + oy_b * cy, oz_b], -1)
        return origin, dh, k

    def scan(self, pose: torch.Tensor, pose_prev: Optional[torch.Tensor], P: Dict[str, torch.Tensor],
             motion_distortion: bool = True, noisy: bool = True, att: Optional[torch.Tensor] = None,
             att_prev: Optional[torch.Tensor] = None, tid: Optional[torch.Tensor] = None,
             compiled: bool = True, cars=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """pose (B,3) base_link (x, y, yaw) at scan end time; pose_prev one scan period earlier;
        att (B,2) body roll/pitch (rad) at scan end, att_prev earlier. Returns
        cars: optional list of box sets [(boxes (B,C,3), dims (B,C,3), zrange (B,C,2), porosity (B,)), ...]
              = other cars (porous bodies) and their rear detection boxes (solid) in the scan.
        (ranges_noisy (B,N), ranges_true (B,N), hit_type (B,N) int32)."""
        B = pose.shape[0]
        if att is None:
            att = torch.zeros(B, 2, device=pose.device)
        if motion_distortion and pose_prev is not None:
            f = self.time_frac[None, :, None]                                  # (1,N,1)
            dyaw = torch.remainder(pose[:, 2] - pose_prev[:, 2] + math.pi, 2 * math.pi) - math.pi
            xy = pose[:, None, :2] - f * (pose[:, None, :2] - pose_prev[:, None, :2])
            yaw = pose[:, None, 2] - f[..., 0] * dyaw[:, None]
            pb = torch.cat([xy, yaw[..., None]], -1)
            ap = att_prev if att_prev is not None else att
            ab = att[:, None, :] - f * (att[:, None, :] - ap[:, None, :])
            origin, dh, k = self._rays(pb, ab, P, per_beam=True)
        else:
            # partial resets call this with varying batch sizes: use the eager path (no recompiles)
            origin, dh, k = (self._rays if compiled else self.rays)(pose, att, P, per_beam=False)
        rmax = P["range_max"]
        r_true, typ = self.trace(origin, dh, k, rmax, tid, P.get("duct_scale"))
        car_poro = None
        if cars is not None:
            car_poro = torch.zeros_like(r_true)                   # per-beam porosity of whatever car part was hit
            for entry in cars:
                if entry[0] == "mesh":                                       # ("mesh", poses, scale, porosity)
                    _, poses_, scale_, poro = entry
                    r_car, hit_car = ray_slices_hits(origin, dh, k, poses_, scale_)
                else:
                    boxes, dims, zr, poro = entry
                    r_car, hit_car = ray_box_hits(origin, dh, k, boxes, dims, zr)
                closer = hit_car & (r_car < r_true)
                r_true = torch.where(closer, r_car, r_true)
                typ = torch.where(closer, torch.full_like(typ, HIT_CAR), typ)
                car_poro = torch.where(closer, poro[:, None].expand_as(car_poro), car_poro)
        if not noisy:
            return r_true, r_true, typ
        tid_ = torch.zeros(B, dtype=torch.long, device=pose.device) if tid is None else tid
        if compiled and self._post_fast is not self._post and car_poro is None:   # one CUDA graph for the whole post-processing (full batches only)
            try:
                r, r_true2, typ2 = self._post_fast(r_true, typ, origin, dh, k, tid_, P)
                return r.clone(), r_true2.clone(), typ2.clone()
            except Exception:
                self._post_fast = self._post                        # inductor/Triton failure: eager from now on
        return self._post(r_true, typ, origin, dh, k, tid_, P, car_poro)

    def _post(self, r_true, typ, origin, dh, k, tid, P, car_poro=None):
        """Noise, spikes, dropouts, incidence-dependent returns -> (ranges, ranges_true, types)."""
        B = r_true.shape[0]
        rm = P["range_max"][:, None]
        r = r_true + torch.randn_like(r_true) * (P["noise_std"][:, None] + P["noise_std_rel"][:, None] * r_true)
        u = torch.rand_like(r)
        spike = u < P["spike_prob"][:, None]
        r = torch.where(spike, torch.rand_like(r) * rm, r)
        u = torch.rand_like(r)
        no_return = (u < P["dropout_prob"][:, None]) | (typ == HIT_NONE) | (r_true >= rm - 1e-4)
        if "floor_dropout" in P:
            theta = torch.atan(k.abs())
            p_floor = P["floor_dropout"][:, None] * (1.0 - theta / P["floor_graze"][:, None]).clamp(0.0, 1.0)
            tid_ = torch.zeros(B, dtype=torch.long, device=r.device) if tid is None else tid
            hit_xy = origin[..., :2] + dh * (r_true / torch.sqrt(1.0 + k * k))[..., None]
            nrm = self.track.edt_gradient(hit_xy, tid_[:, None].expand(k.shape), self.track.edt_duct)
            cos_inc = (nrm * dh).sum(-1).abs()
            p_duct = P["duct_graze_dropout"][:, None] * (1.0 - cos_inc / P["duct_graze_cos"][:, None]).clamp(0.0, 1.0)
            u2 = torch.rand_like(r)
            no_return = no_return | ((typ == HIT_GROUND) & (u2 < p_floor)) | ((typ == HIT_DUCT) & (u2 < p_duct))
        if car_poro is not None:
            no_return = no_return | ((typ == HIT_CAR) & (torch.rand_like(r) < car_poro))
        r = r.clamp_min(P["range_min"][:, None])
        r = torch.where(no_return, P["dropout_value"][:, None].expand_as(r), r)
        return r, r_true, typ
