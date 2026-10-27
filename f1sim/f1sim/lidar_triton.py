"""Triton kernels for the 2D LiDAR: 3D-aware sphere tracing on layered distance fields.

Each ray is a straight line with horizontal unit direction (dx, dy) and slope k = dz/ds.
Obstacles: duct hoses (height duct_h, beams pass over them), tall objects (always block),
and the floor (z = 0). One thread per ray, the whole march inside the kernel.
Hit types: 0 none/max range, 1 duct, 2 tall, 3 ground.

`car_mesh_hits_triton` is the other cars' mesh outlines (`lidar.ray_slices_hits`) in one kernel.
"""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

if HAVE_TRITON:
    @triton.jit
    def _trace3d_kernel(ox_ptr, oy_ptr, oz_ptr, dx_ptr, dy_ptr, k_ptr, rmax_ptr, out_r_ptr, out_t_ptr,
                        duct_ptr, tall_ptr, tid_ptr, t_off_ptr, t_ox_ptr, t_oy_ptr, t_res_ptr, t_H_ptr, t_W_ptr,
                        t_duct_ptr, dscale_ptr, n_rays, n_beams, max_iters, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_rays
        ox = tl.load(ox_ptr + offs, mask=mask, other=0.0)
        oy = tl.load(oy_ptr + offs, mask=mask, other=0.0)
        oz = tl.load(oz_ptr + offs, mask=mask, other=0.0)
        dx = tl.load(dx_ptr + offs, mask=mask, other=0.0)
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0)
        k = tl.load(k_ptr + offs, mask=mask, other=0.0)
        env = offs // n_beams
        rmax = tl.load(rmax_ptr + env, mask=mask, other=0.0)
        tid = tl.load(tid_ptr + env, mask=mask, other=0)
        off = tl.load(t_off_ptr + tid, mask=mask, other=0)
        origin_x = tl.load(t_ox_ptr + tid, mask=mask, other=0.0)
        origin_y = tl.load(t_oy_ptr + tid, mask=mask, other=0.0)
        res = tl.load(t_res_ptr + tid, mask=mask, other=1.0)
        inv_res = 1.0 / res
        H = tl.load(t_H_ptr + tid, mask=mask, other=1)
        W = tl.load(t_W_ptr + tid, mask=mask, other=1)
        duct_h = tl.load(t_duct_ptr + tid, mask=mask, other=0.2) * tl.load(dscale_ptr + env, mask=mask, other=1.0)
        slen = tl.sqrt(1.0 + k * k)                 # 3D length per horizontal meter
        s_max = rmax / slen
        desc = k < -1e-6
        s_ground = tl.where(desc & (oz > 0.0), -oz / tl.where(desc, k, -1.0), 1e9)
        s = tl.zeros([BLOCK], dtype=tl.float32)
        done = mask == 0                            # padding lanes count as finished
        typ = tl.zeros([BLOCK], dtype=tl.int32)
        half = 0.5 * res
        minstep = 0.25 * res
        it = 0
        n_done = tl.sum(done.to(tl.int32), axis=0)
        while (it < max_iters) & (n_done < BLOCK):   # early exit once every ray in the block finished
            it += 1
            px = ox + dx * s
            py = oy + dy * s
            z = oz + k * s
            col = tl.floor((px - origin_x) * inv_res + 0.5).to(tl.int32)
            row = tl.floor((py - origin_y) * inv_res + 0.5).to(tl.int32)
            inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
            colc = tl.minimum(tl.maximum(col, 0), W - 1)
            rowc = tl.minimum(tl.maximum(row, 0), H - 1)
            idx = off + (rowc * W + colc).to(tl.int64)
            d_duct = tl.load(duct_ptr + idx, mask=mask & inside, other=1e9).to(tl.float32)
            d_tall = tl.load(tall_ptr + idx, mask=mask & inside, other=0.0).to(tl.float32)
            d_duct = tl.where(inside, d_duct, 1e9)
            d_tall = tl.where(inside, d_tall, 0.0)
            above = z > duct_h
            d_eff = tl.where(above, d_tall, tl.minimum(d_duct, d_tall))
            hit = (d_eff <= half) & (done == 0)
            hit_type = tl.where(above | (d_tall <= d_duct), 2, 1)
            step = tl.maximum(d_eff - half, minstep)
            s_desc = tl.where(desc, (z - duct_h) / tl.where(desc, -k, 1.0), 1e9)
            step = tl.where(above, tl.minimum(step, tl.maximum(s_desc, minstep)), step)
            s_next = s + step
            ground = (done == 0) & (hit == 0) & (s_next >= s_ground)
            far = (done == 0) & (hit == 0) & (ground == 0) & (s_next >= s_max)
            typ = tl.where(hit, hit_type, typ)
            typ = tl.where(ground, 3, typ)
            typ = tl.where(far, 0, typ)
            s_new = tl.where(hit, s, tl.where(ground, s_ground, tl.where(far, s_max, s_next)))
            s = tl.where(done, s, s_new)
            done = done | hit | ground | far
            n_done = tl.sum(done.to(tl.int32), axis=0)
        # duct hose is round: at heights off its centre the surface is further in
        R = 0.5 * duct_h
        zh = oz + k * s
        w = tl.sqrt(tl.maximum(R * R - (zh - R) * (zh - R), 0.0))
        s = tl.where(typ == 1, s + (R - w), s)
        s = tl.minimum(s, s_max)
        typ = tl.where(done, typ, 0)
        tl.store(out_r_ptr + offs, s * slen, mask=mask)
        tl.store(out_t_ptr + offs, typ, mask=mask)


def trace3d_triton(origin, direction_h, k, range_max, tt, tid, max_iters, duct_scale=None):
    """origin (B,N,3), direction_h (B,N,2) unit horizontal, k (B,N) slope, range_max (B,), tt: TrackTensors,
    tid (B,) track id per env -> (ranges (B,N), types (B,N) int32)."""
    B, N = k.shape
    f = lambda t: t.contiguous().reshape(-1)
    ox, oy, oz = f(origin[..., 0]), f(origin[..., 1]), f(origin[..., 2])
    dx, dy = f(direction_h[..., 0]), f(direction_h[..., 1])
    kk = f(k)
    rmax = range_max.reshape(-1).contiguous()
    out_r = torch.empty(B * N, device=k.device, dtype=torch.float32)
    out_t = torch.empty(B * N, device=k.device, dtype=torch.int32)
    n = B * N
    BLOCK = 256
    tid32 = tid.to(torch.int32).contiguous()
    dscale = (torch.ones(B, device=k.device, dtype=torch.float32) if duct_scale is None else duct_scale.to(torch.float32)).contiguous()
    _trace3d_kernel[(triton.cdiv(n, BLOCK),)](
        ox, oy, oz, dx, dy, kk, rmax, out_r, out_t, tt.edt_duct, tt.edt_tall, tid32, tt.t_off,
        tt.t_origin[:, 0].contiguous(), tt.t_origin[:, 1].contiguous(), tt.t_res, tt.t_H, tt.t_W, tt.t_duct_h, dscale,
        n, N, int(max_iters), BLOCK=BLOCK)
    return out_r.view(B, N), out_t.view(B, N)


# ---------------------------------------------------------------------------------------------
# Other cars: beams against the car mesh outlines (`lidar.ray_slices_hits`).
#
# `_car_mesh_kernel` computes exactly what `lidar._ray_slices_hits_gathered` computes, operation
# for operation in float32 -- the same products, sums and quotients in the same order, each rounded
# once: every rounded operation is an explicit IEEE intrinsic (`_mul`/`_add`/`_sub` below, `div_rn`,
# `sqrt_rn` -- not Triton's approximate division and square root), cos/sin of the car yaw come from
# torch, the level is `rint` (torch.round rounds half to even) of `(.. - z0) * inv_dz` (torch on
# CUDA divides by a Python scalar as a multiply by its float32 reciprocal), and the NaN-propagating
# max/min stand where torch's clamp_min/minimum propagate NaN. One program per BLOCK rays; per car,
# the segments of each ray's own level are gathered SBLOCK at a time.
#
# Why intrinsics and not `a * b - c * d` with `enable_fp_fusion=False`: that was bit-exact on sm_89,
# but on sm_120 (RTX 5060, Blackwell) LLVM packs neighbouring operations into `mul.rn.f32x2` /
# `sub.rn.f32x2`, and ptxas then fuses those into FFMA despite the `.rn` and `--fmad=false` -- e.g.
# `ex = fl(x1*sc) - fl(x0*sc)` came out as one rounding, `fma(x1, sc, -fl(x0*sc))`, in 29 % of the
# segment pairs of a probe (2026-10-27). The torch reference itself is IEEE per operation on both
# GPUs. The scalar NVVM intrinsics behind `libdevice.mul_rn` etc. are neither packed nor fused, and
# the launch turns off Triton's default flush-to-zero for libdevice (`enable_reflect_ftz=False`), so
# they are `mul.rn.f32`, not `mul.rn.ftz.f32`: torch's kernels keep subnormals, and so does this.
#
# Early-out (EARLY_OUT). A ray skips a car when its LINE (both directions -- see below) passes the
# car centre farther than every outline endpoint can reach; a block skips the car's segment loop
# when all of its rays do. Proof that a skipped ray had no segment with `ok` in the reference:
#   In the car frame, L = (lx, ly) is the ray origin and D = (dx, dy) its direction -- the same
#   float32 numbers in both implementations. Every valid scaled endpoint P = (fl(x*sc), fl(y*sc))
#   has |P| <= R := r_bound*|sc|, as r_bound >= (1 + 1e-6) * max |(x, y)| over the asset's valid
#   endpoints and a rounded product adds at most 2^-24 relative. The function g(P) = cross(P - L, D)
#   is affine, so on the disc |P| <= R it keeps one sign and |g| >= H - R|D|, H = |cross(L, D)|.
#   For a segment A->B the reference takes u = fl(num/den_c), num = fl(cross(W, D)) with
#   W = fl(A - L); in exact arithmetic num = a := g(A) and den = cross(D, B - A) = a - b, b := g(B).
#   The rounding errors are |e1| <= 3*2^-24*(2R + |lx| + |ly|)*|D| on num and |e2| <= 12*2^-24*R*|D|
#   on den. Whenever min(|a|, |b|) > |e1| + |e2| + 1e-9 + 2e-5*|D| the test 0 <= u <= 1 fails: with
#   a, b > 0 (the other sign is the mirror image) num > 2e-5*|D| > 0, so a den_c < 0 gives u < 0,
#   the 1e-9 clamp gives u = num/1e-9 >> 1, and a den_c > 0 gives num - den_c = b + e1 - e2 > 0,
#   i.e. num/den_c > 1 by a relative (|b| - |e1| - |e2|)/|den_c| >= 2e-5/(2.00001*R) > 9.9e-8 > 2^-24
#   (|den_c| <= |D|*|B - A| + |e2|, R < 100), so the rounded quotient is > 1 too; and no |u| here
#   is below 9.9e-8, far from a subnormal that could round to -0.
#   The kernel skips when hc > thr, dl > 0.5 and R < 100, with hc = |fl(lx*dy - ly*dx)| (error <=
#   3*2^-24*(|lx| + |ly|)*|D| against H), dl = sqrt_rn(dx^2 + dy^2) (within 2e-7 relative of |D|)
#   and thr = (R*(1 + 1e-5) + 1e-4 + 1e-5*(|lx| + |ly|))*dl (within 1e-6 relative after its own
#   rounding). Then H - R|D| >= |D|*(8.9e-6*R + 0.99e-4 + 9.8e-6*(|lx| + |ly|)), which beats
#   |e1| + |e2| + 1e-9 + 2e-5*|D| <= |D|*(1.1e-6*R + 1.8e-7*(|lx| + |ly|) + 2e-5) + 1e-9 term by
#   term (|D| > 0.49). A non-finite H, thr or dl makes the comparison false: never skipped.
# The test is on the whole line, not the half-line t > 0: with a ray parallel to a segment the
# reference clamps den to 1e-9, and then u and t are rounding noise -- t > 0 can pass for a segment
# BEHIND the origin. On the line such a segment still lies within R + 1e-9, so it is never skipped.
# ---------------------------------------------------------------------------------------------
if HAVE_TRITON:
    # One IEEE rounding each, on every architecture: scalar NVVM intrinsics, never packed into
    # f32x2 and never contracted into an FMA (see the note above `_car_mesh_kernel`).
    @triton.jit
    def _mul(a, b):
        return libdevice.mul_rn(a, b)

    @triton.jit
    def _add(a, b):
        return libdevice.add_rn(a, b)

    @triton.jit
    def _sub(a, b):
        return libdevice.sub_rn(a, b)

    @triton.jit
    def _car_mesh_kernel(o_ptr, d_ptr, k_ptr, pose_ptr, cy_ptr, sy_ptr, sc_ptr,
                         x0_ptr, y0_ptr, x1_ptr, y1_ptr, valid_ptr, count_ptr,
                         out_r_ptr, out_h_ptr, n_rays, n_beams, n_cars, z0, inv_dz, r_bound,
                         LEVELS: tl.constexpr, SEGS: tl.constexpr, BLOCK: tl.constexpr,
                         SBLOCK: tl.constexpr, EARLY_OUT: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_rays
        ox = tl.load(o_ptr + offs * 3, mask=mask, other=0.0)
        oy = tl.load(o_ptr + offs * 3 + 1, mask=mask, other=0.0)
        oz = tl.load(o_ptr + offs * 3 + 2, mask=mask, other=0.0)
        hx = tl.load(d_ptr + offs * 2, mask=mask, other=1.0)
        hy = tl.load(d_ptr + offs * 2 + 1, mask=mask, other=0.0)
        k = tl.load(k_ptr + offs, mask=mask, other=0.0)
        env = offs // n_beams
        slen = tl.sqrt_rn(_add(1.0, _mul(k, k)))
        best = tl.full([BLOCK], float("inf"), tl.float32)
        for c in range(n_cars):
            pc = env * n_cars + c
            px = tl.load(pose_ptr + pc * 3, mask=mask, other=0.0)
            py = tl.load(pose_ptr + pc * 3 + 1, mask=mask, other=0.0)
            cy = tl.load(cy_ptr + pc, mask=mask, other=1.0)
            sy = tl.load(sy_ptr + pc, mask=mask, other=0.0)
            sc = tl.load(sc_ptr + pc, mask=mask, other=1.0)
            rx = _sub(ox, px)
            ry = _sub(oy, py)
            lx = _add(_mul(rx, cy), _mul(ry, sy))                    # ray origin in the car frame
            ly = _add(_mul(-rx, sy), _mul(ry, cy))
            dx = _add(_mul(hx, cy), _mul(hy, sy))
            dy = _add(_mul(-hx, sy), _mul(hy, cy))
            t_c = tl.maximum(_sub(_mul(-lx, dx), _mul(ly, dy)), 0.0, propagate_nan=tl.PropagateNan.ALL)
            lvl = libdevice.rint(_mul(_sub(_add(oz, _mul(k, t_c)), z0), inv_dz))
            # float compares = the reference's int64 ones: NaN/inf -> out of band there as well
            in_band = (lvl >= 0.0) & (lvl < LEVELS)
            safe = tl.minimum(tl.maximum(lvl, 0.0), LEVELS - 1).to(tl.int32)
            active = mask & in_band
            if EARLY_OUT:
                rs = _mul(r_bound, tl.abs(sc))
                dl = tl.sqrt_rn(_add(_mul(dx, dx), _mul(dy, dy)))
                hc = tl.abs(_sub(_mul(lx, dy), _mul(ly, dx)))
                thr = _mul(_add(_add(_mul(rs, 1.00001), 1e-4), _mul(1e-5, _add(tl.abs(lx), tl.abs(ly)))), dl)
                active = active & (((hc > thr) & (dl > 0.5) & (rs < 100.0)) == 0)
            # segments past the level's last valid one are invalid: the loop stops there
            n_eff = tl.where(active, tl.load(count_ptr + safe, mask=mask, other=0), 0)
            n_max = tl.max(n_eff, axis=0)
            t_best = tl.full([BLOCK], float("inf"), tl.float32)
            base = safe * SEGS
            for s0 in range(0, n_max, SBLOCK):
                s = s0 + tl.arange(0, SBLOCK)
                m2 = s[None, :] < n_eff[:, None]
                idx = base[:, None] + s[None, :]
                sx0 = tl.load(x0_ptr + idx, mask=m2, other=0.0)
                sy0 = tl.load(y0_ptr + idx, mask=m2, other=0.0)
                sx1 = tl.load(x1_ptr + idx, mask=m2, other=0.0)
                sy1 = tl.load(y1_ptr + idx, mask=m2, other=0.0)
                v = tl.load(valid_ptr + idx, mask=m2, other=0)
                scb = sc[:, None]
                ax = _mul(sx0, scb)
                ay = _mul(sy0, scb)
                bx = _mul(sx1, scb)
                by = _mul(sy1, scb)
                ex = _sub(bx, ax)
                ey = _sub(by, ay)
                dxb = dx[:, None]
                dyb = dy[:, None]
                den = _sub(_mul(dxb, ey), _mul(dyb, ex))
                den = tl.where(tl.abs(den) < 1e-9, 1e-9, den)
                wx = _sub(ax, lx[:, None])
                wy = _sub(ay, ly[:, None])
                t = tl.div_rn(_sub(_mul(wx, ey), _mul(wy, ex)), den)
                u = tl.div_rn(_sub(_mul(wx, dyb), _mul(wy, dxb)), den)
                ok = (t > 0.0) & (u >= 0.0) & (u <= 1.0) & m2 & (v != 0)
                t_best = tl.minimum(t_best, tl.min(tl.where(ok, t, float("inf")), axis=1))
            best = tl.minimum(best, _mul(t_best, slen), propagate_nan=tl.PropagateNan.ALL)
        tl.store(out_r_ptr + offs, best, mask=mask)
        tl.store(out_h_ptr + offs, tl.abs(best) < float("inf"), mask=mask)   # isfinite


def car_mesh_hits_triton(origin, dh, k, poses, scale, tables, early_out=True,
                         block=16, sblock=8, num_warps=1):
    """origin (B,N,3), dh (B,N,2), k (B,N); poses (B,C,3) x, y, yaw; scale (B,C), all float32 CUDA;
    tables: (x0, y0, x1, y1 (L,S) float32, valid (L,S) uint8, count (L,) int32 = last valid
    segment + 1, z0, inv_dz, r_bound) from `lidar._car_mesh_tables` -> (3D range (B,N), hit (B,N)).

    Small tiles, because the early-out is per block: 16 beams are 4 deg of a 1081-beam scan, so
    a block rarely runs a car it does not look at. RTX 4070 SUPER, N=1081, C=3, beside a training
    job: 16/8/1 was the fastest of the configs tried on real race states (0.10-0.32 ms at B=256,
    0.42-0.73 ms at B=1024) and on a dense case with cars 0.5-1.5 m away (0.37 / 1.19 ms); the
    eager tiles it replaces took 113-118 ms and ~470 ms."""
    B, N = k.shape
    C = poses.shape[1]
    n = B * N
    if n == 0 or C == 0:
        return (torch.full((B, N), float("inf"), device=k.device, dtype=torch.float32),
                torch.zeros((B, N), device=k.device, dtype=torch.bool))
    out_r = torch.empty((B, N), device=k.device, dtype=torch.float32)
    out_h = torch.empty((B, N), device=k.device, dtype=torch.bool)
    x0, y0, x1, y1, valid, count, z0, inv_dz, r_bound = tables
    p = poses.contiguous()
    cy, sy = torch.cos(p[..., 2]).contiguous(), torch.sin(p[..., 2]).contiguous()
    with torch.cuda.device(k.device):                    # Triton launches on the current device
        _car_mesh_kernel[(triton.cdiv(n, block),)](
            origin.contiguous(), dh.contiguous(), k.contiguous(), p, cy, sy, scale.contiguous(),
            x0, y0, x1, y1, valid, count, out_r, out_h, n, N, C, z0, inv_dz, r_bound,
            LEVELS=x0.shape[0], SEGS=x0.shape[1], BLOCK=block, SBLOCK=sblock, EARLY_OUT=early_out,
            num_warps=num_warps, enable_fp_fusion=False, enable_reflect_ftz=False)
    return out_r, out_h
