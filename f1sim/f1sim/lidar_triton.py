"""Triton kernels for the 2D LiDAR: 3D-aware sphere tracing on layered distance fields.

Each ray is a straight line with horizontal unit direction (dx, dy) and slope k = dz/ds.
Obstacles: duct hoses (height duct_h, beams pass over them), tall objects (always block),
and the floor (z = 0). One thread per ray, the whole march inside the kernel.
Hit types: 0 none/max range, 1 duct, 2 tall, 3 ground.
"""
import torch

try:
    import triton
    import triton.language as tl
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
