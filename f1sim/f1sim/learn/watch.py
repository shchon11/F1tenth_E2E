"""Watch a policy while it trains: runs N agents on a track with the latest checkpoint of a run
(auto-reloads when the file changes) in the native viewer, with the network's inner state on screen:
  * the focus car's LiDAR points colored by saliency |d action / d beam| (what the policy looks at)
  * hidden-layer activations (256 units) as a heat grid, action mean/std gauges, value estimate
  * checkpoint step count in the HUD
Headless recording: --record out.mp4 (needs ffmpeg) or --frames dir/ for PNGs.
"""
from __future__ import annotations

# ---------------------------------------------------------------- GUI route, before the heavy imports
# `python -m f1sim.learn.watch` with no arguments is how people open the console, and everything
# below this block -- torch, gym_env, common, the checkpoint loader -- belongs to the headless and
# recording paths, not to it. Left where they were, the console process paid `import torch` (1.3 s
# measured) before it could paint anything, and the "the GUI process never imports torch" property
# held only for the explicit `-m f1sim.viewer.console` spelling. Routing here makes the claim true
# of the entry point people actually type.
#
# Deliberately narrow: only the no-argument and `--console` / `--gui` spellings, only when this
# module is __main__. Every other invocation, and every programmatic `watch.main(argv)` call, falls
# through to the unchanged module below -- `main()` keeps its own copy of this routing so the
# library entry point behaves the same way.
if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) == 1 or (len(_sys.argv) == 2 and _sys.argv[1] in ("--console", "--gui")):
        from ..viewer.console import launch as _launch          # no torch on this path
        _sys.exit(_launch([_sys.argv[0]]))

import argparse
import collections
import glob
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..gym_env import EnvConfig
from ..params import Config
from . import common
from .model import load_checkpoint
from .obs import flatten_obs


class Introspector:
    """Forward hooks on the actor: hidden activations + saliency of the mean action w.r.t. the scan."""

    def __init__(self, model):
        self.model = model; self.h = {}
        model.actor.mlp.register_forward_hook(lambda m, i, o: self.h.__setitem__("hidden", o.detach()))
        model.actor.stem.fc.register_forward_hook(lambda m, i, o: self.h.__setitem__("stem", o.detach()))

    def saliency(self, scan, pro, i):
        s = scan[i:i + 1].clone().requires_grad_(True); p = pro[i:i + 1]
        mu = self.model.actor(s, p)
        g = torch.autograd.grad(mu.abs().sum(), s)[0][0]              # (k, N)
        sal = g.abs().sum(0)                                          # per beam
        return (sal / (sal.max() + 1e-9)).cpu().numpy(), mu[0].detach().cpu().numpy()


def sal_colors(sal):
    """Per-beam saliency as brightness on one hue, not as a hue ramp.

    Hue is spoken for: the plan line is coloured by its speed profile, and that ramp already runs
    blue to red. A second ramp ending in red put the two on the same word -- a red patch meant
    "the network is looking here" or "the car is flat out" depending on which mark you were reading.
    Attention is a scalar with no natural direction, so brightness carries it: dim slate for beams
    that do not move the action, bright white for the ones that do.
    """
    sal = np.asarray(sal, np.float32)
    t = np.clip(sal, 0.0, 1.0)[:, None]
    lo = np.array([0.26, 0.34, 0.46], np.float32)          # slate: present but quiet
    hi = np.array([1.00, 1.00, 0.98], np.float32)          # white hot
    rgb = lo + (hi - lo) * (t ** 0.7)                      # gamma: mid saliency still reads as lit
    a = 0.45 + 0.55 * t
    return np.concatenate([rgb, a], 1).astype(np.float32)


def panel_internals(hidden, stem):
    """Raw activations (--internals): hidden layer as a 16x16 heat grid, scan features as a strip."""
    W, H = 480, 130
    img = Image.new("RGBA", (W, H), (12, 15, 22, 190)); d = ImageDraw.Draw(img)
    from ..viewer.gl_scene import _font
    fs = _font(13)
    hgrid = hidden.reshape(16, 16); hg = (hgrid - hgrid.min()) / (hgrid.max() - hgrid.min() + 1e-9)
    for r in range(16):
        for c in range(16):
            v = float(hg[r, c]); d.rectangle((10 + c * 6, 8 + r * 6, 15 + c * 6, 13 + r * 6), fill=(int(255 * v), int(120 * v), int(255 * (1 - v)), 255))
    d.text((10, 108), "hidden layer, 256 neurons", font=fs, fill=(180, 190, 210, 255))
    sg = (stem - stem.min()) / (stem.max() - stem.min() + 1e-9)
    for i in range(256):
        v = float(sg[i]); d.rectangle((130 + (i % 64) * 5, 8 + (i // 64) * 10, 134 + (i % 64) * 5, 16 + (i // 64) * 10), fill=(int(255 * v), int(200 * v), 60, 255))
    d.text((130, 52), "scan features after the 1D conv, 256", font=fs, fill=(180, 190, 210, 255))
    return img


_BEV_BG = {}


def _bev_background(size: int, span: float, v_max: float):
    """Rings, labels and legend: fixed for a given scale, so draw them once and keep the array."""
    from PIL import Image, ImageDraw
    key = (size, round(span, 2), round(v_max, 2))
    if key in _BEV_BG:
        return _BEV_BG[key]
    W = H = size
    px_per_m = (size * 0.44) / span
    cx, cy = W * 0.5, H * 0.55
    img = Image.new("RGBA", (W, H), (10, 13, 20, 205))
    d = ImageDraw.Draw(img)
    ring = 1 if span <= 5 else 2
    for r_ in range(ring, int(span) + 1, ring):
        rr = r_ * px_per_m
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(70, 80, 100, 120))
        d.text((cx + 3, cy - rr - 12), f"{r_}m", fill=(120, 132, 155, 190))
    d.text((10, H - 34), "points = saliency: dim quiet, bright drives the action",
           fill=(150, 160, 180, 210))
    d.text((10, H - 18), f"plan line = its speed profile, blue 0 to red {v_max:.1f} m/s",
           fill=(150, 160, 180, 210))
    _BEV_BG[key] = np.asarray(img, np.uint8).copy()
    if len(_BEV_BG) > 8:
        _BEV_BG.pop(next(iter(_BEV_BG)))
    return _BEV_BG[key]


def bev_image(scan, range_max: float, fov: float = 4.71238898, span: float = 8.0,
              size: int = 360, plan_ref=None, sal=None, v_max: float = 10.0, title: str = ""):
    """The one view of what the policy is doing: its newest scan from above, coloured by how much
    each beam moves the action, with the plan it produced drawn on the same axes.

    Only the newest scan. The stack is what the temporal encoder consumes, but drawing six frames
    made a smear that hid the thing worth seeing -- where the network is looking right now, and
    whether the plan it drew goes anywhere near the gap it was looking at.

    True scale, metres to pixels. `span` is the half-width in metres: 8 m covers the field a driving
    decision is actually made on, and anything further is clipped rather than compressed, so a
    distance read off the picture is the distance.

    Channels do not overlap: hue is the plan's speed, brightness is per-beam saliency.

    The beams are splatted with numpy into a cached background rather than drawn one PIL ellipse at
    a time -- a thousand ellipse calls cost 5 ms, a third of a 60 Hz frame, which is why this panel
    used to run on its own 2 Hz timer and drift out of step with the scene behind it.

    scan: (k, N) or (N,) normalised range / range_max. sal: (N,) per-beam saliency in [0, 1].
    """
    from PIL import Image, ImageDraw
    sc = np.asarray(scan, np.float32)
    r = (sc[0] if sc.ndim > 1 else sc) * range_max
    N = len(r)
    span = float(span or range_max)
    W = H = size
    px_per_m = (size * 0.44) / span
    cx, cy = W * 0.5, H * 0.55
    buf = _bev_background(size, span, v_max).copy()

    ang = np.linspace(-fov / 2, fov / 2, N)         # same geometry as f1sim.lidar.Lidar.rays
    keep = (r < range_max * 0.999) & (r <= span)    # a miss reads range_max: not a return
    if keep.any():
        # Body frame per f1sim.lidar.Lidar.rays: forward is +x, left is +y, the beam angle measured
        # from forward. On screen forward is up and left is left, so both axes are negated.
        rp = r[keep] * px_per_m
        xi = np.rint(cx - rp * np.sin(ang[keep])).astype(np.int32)
        yi = np.rint(cy - rp * np.cos(ang[keep])).astype(np.int32)
        cols = (sal_colors(np.asarray(sal, np.float32))[keep] if sal is not None and len(sal) == N
                else np.tile(np.array([0.55, 0.72, 0.95, 0.9], np.float32), (int(keep.sum()), 1)))
        rgba = (cols * 255).astype(np.uint8)
        for dx in (-1, 0, 1):                       # 3x3 splat: one pixel per beam is invisible
            for dy in (-1, 0, 1):
                x2, y2 = xi + dx, yi + dy
                ok = (x2 >= 0) & (x2 < W) & (y2 >= 0) & (y2 < H)
                buf[y2[ok], x2[ok]] = rgba[ok]
    img = Image.fromarray(buf, "RGBA")
    d = ImageDraw.Draw(img)
    if plan_ref is not None and len(plan_ref) > 1:  # the plan, in the same body frame and scale
        from ..viewer.gl_scene import speed_colors
        pr = np.asarray(plan_ref, np.float32)
        pts = [(cx - pr[i, 1] * px_per_m, cy - pr[i, 0] * px_per_m) for i in range(len(pr))]
        if pr.shape[1] > 3:                         # coloured by its own speed profile, on the same
            pc = (speed_colors(pr[:, 3], v_max) * 255).astype(int)   # ramp the 3D plan line uses
            for i in range(len(pts) - 1):
                d.line([pts[i], pts[i + 1]], fill=tuple(pc[i]), width=4)
        else:
            d.line(pts, fill=(245, 245, 250, 235), width=4)
    d.polygon([(cx, cy - 8), (cx - 6, cy + 7), (cx + 6, cy + 7)], fill=(255, 255, 255, 235))
    d.text((10, 8), title or f"POLICY INPUT + PLAN   {N} beams   {span:.0f} m",
           fill=(235, 238, 245, 255))
    return img


def dash_image(v, v_cmd, v_cap, steer, steer_cmd, v_max=8.0, steer_max=0.4189,
               gg=None, mu_g=None):
    """Bottom-centre dash: a speedometer (needle = measured speed, orange tick = commanded speed,
    red zone above the cap), a steering wheel turned by the actual steering angle (x3 for
    visibility, ghost tick = commanded angle), and a g-g diagram.

    gg: recent (a_lat, a_lon) in m/s^2, oldest first -- the body-frame specific forces the
    accelerometer reads, so the trace is what the car felt rather than what was asked of it.
    mu_g: the friction limit [m/s^2] to draw the circle at. A car using its tyres traces a full
    circle; one that only ever brakes and accelerates in a straight line traces a cross, which is
    what a policy leaving lap time on the table looks like.
    """
    from ..viewer.gl_scene import _font
    W, H = 540 if gg is None else 720, 180
    img = Image.new("RGBA", (W, H), (10, 12, 18, 170)); d = ImageDraw.Draw(img)
    f, fs, fb = _font(13), _font(11), _font(22)
    # ---- speedometer
    cx, cy, R = 110, 105, 78
    a0, a1 = 210.0, -30.0                                          # degrees, clockwise sweep of 240
    def ang(val): return math.radians(a0 + (a1 - a0) * min(max(val / v_max, 0.0), 1.0))
    d.arc((cx - R, cy - R, cx + R, cy + R), start=-a0, end=-a1, fill=(90, 100, 120, 255), width=10)
    if v_cap < v_max:                                              # red zone: beyond the speed cap
        d.arc((cx - R, cy - R, cx + R, cy + R), start=-math.degrees(ang(v_cap)), end=-a1, fill=(200, 60, 60, 255), width=10)
    d.arc((cx - R, cy - R, cx + R, cy + R), start=-a0, end=-math.degrees(ang(v)), fill=(80, 200, 255, 255), width=10)
    for k in range(int(v_max) + 1):
        t = ang(k); x1, y1 = cx + (R - 14) * math.cos(t), cy - (R - 14) * math.sin(t); x2, y2 = cx + (R - 6) * math.cos(t), cy - (R - 6) * math.sin(t)
        d.line((x1, y1, x2, y2), fill=(200, 205, 215, 255), width=2)
        tx, ty = cx + (R - 26) * math.cos(t), cy - (R - 26) * math.sin(t); d.text((tx - 4, ty - 7), str(k), font=fs, fill=(170, 180, 195, 255))
    t = ang(v_cmd); d.polygon([(cx + (R + 2) * math.cos(t), cy - (R + 2) * math.sin(t)), (cx + (R + 12) * math.cos(t + 0.06), cy - (R + 12) * math.sin(t + 0.06)),
                               (cx + (R + 12) * math.cos(t - 0.06), cy - (R + 12) * math.sin(t - 0.06))], fill=(255, 160, 40, 255))
    t = ang(v); d.line((cx, cy, cx + (R - 18) * math.cos(t), cy - (R - 18) * math.sin(t)), fill=(255, 255, 255, 255), width=3)
    d.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(255, 255, 255, 255))
    d.text((cx - 34, cy + 22), f"{v:4.2f}", font=fb, fill=(255, 255, 255, 255)); d.text((cx + 22, cy + 30), "m/s", font=fs, fill=(200, 205, 215, 255))
    d.text((cx - 62, H - 22), "speed", font=f, fill=(200, 205, 215, 255)); d.text((cx - 10, H - 22), f"cmd {v_cmd:.1f}   cap {v_cap:.0f}", font=fs, fill=(255, 160, 40, 255))
    # ---- steering wheel
    wx, wy, r = 340, 92, 58
    wheel = Image.new("RGBA", (2 * r + 20, 2 * r + 20), (0, 0, 0, 0)); wd = ImageDraw.Draw(wheel); c = r + 10
    wd.ellipse((c - r, c - r, c + r, c + r), outline=(225, 228, 235, 255), width=9)
    for a_ in (90, 210, 330):
        t = math.radians(a_); wd.line((c, c, c + (r - 4) * math.cos(t), c - (r - 4) * math.sin(t)), fill=(200, 205, 215, 255), width=7)
    wd.ellipse((c - 12, c - 12, c + 12, c + 12), fill=(150, 155, 170, 255))
    wd.rectangle((c - 3, c - r - 4, c + 3, c - r + 10), fill=(255, 160, 40, 255))           # top marker
    gain = 3.0
    wheel = wheel.rotate(math.degrees(steer) * gain, resample=Image.BICUBIC)               # left turn = counter-clockwise
    img.alpha_composite(wheel, (wx - c, wy - c))
    t = math.radians(90 + math.degrees(steer_cmd) * gain)                                  # ghost: commanded angle
    d.line((wx + (r + 6) * math.cos(t), wy - (r + 6) * math.sin(t), wx + (r + 16) * math.cos(t), wy - (r + 16) * math.sin(t)), fill=(255, 160, 40, 200), width=3)
    d.text((wx + r + 22, wy - 24), f"{math.degrees(steer):+5.1f}°", font=fb, fill=(255, 255, 255, 255))
    d.text((wx + r + 22, wy + 4), f"cmd {math.degrees(steer_cmd):+.1f}°", font=fs, fill=(255, 160, 40, 255))
    # ---- g-g diagram
    if gg is not None:
        gx, gy, gr = 620, 92, 66
        lim = float(mu_g or 9.81)
        px = gr / lim
        for frac in (0.5, 1.0):                                    # half and full friction circle
            rr = gr * frac
            d.ellipse((gx - rr, gy - rr, gx + rr, gy + rr),
                      outline=(200, 70, 60, 255) if frac == 1.0 else (70, 80, 100, 255),
                      width=2 if frac == 1.0 else 1)
        d.line((gx - gr, gy, gx + gr, gy), fill=(70, 80, 100, 200))
        d.line((gx, gy - gr, gx, gy + gr), fill=(70, 80, 100, 200))
        a = np.asarray(gg, np.float32).reshape(-1, 2)
        for k, (alat, alon) in enumerate(a):
            age = 1.0 - k / max(1, len(a) - 1)                     # newest brightest
            x = gx + float(np.clip(alat, -lim * 1.3, lim * 1.3)) * px
            y = gy - float(np.clip(alon, -lim * 1.3, lim * 1.3)) * px
            rad = 3.0 if k == len(a) - 1 else 1.6
            c_ = (255, 255, 255, 255) if k == len(a) - 1 else (110, 190, 255, int(40 + 180 * (1 - age)))
            d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=c_)
        cur = a[-1] if len(a) else np.zeros(2)
        g_now = float(np.hypot(*cur)) / 9.81
        d.text((gx - gr, gy + gr + 6), f"g-g   {g_now:4.2f} g of {lim / 9.81:.2f}",
               font=f, fill=(200, 205, 215, 255))
        d.text((gx + gr - 26, gy - gr - 16), "accel", font=fs, fill=(140, 150, 170, 255))
        d.text((gx + gr - 20, gy + gr - 2), "brake", font=fs, fill=(140, 150, 170, 255))
    d.text((wx - 30, H - 22), "steering", font=f, fill=(200, 205, 215, 255)); d.text((wx + 34, H - 22), "(wheel turned x3)", font=fs, fill=(150, 160, 180, 255))
    return img


class EpisodeRecorder:
    """Stores what the viewer needs for every agent and step (scans as uint8, 4 cm resolution) so
    any agent's run can be replayed exactly, and ranks agents by their first episode."""

    def __init__(self, env):
        self.env = env; self.B = env.B; self.range_max = env.range_max
        self.state, self.att, self.lap, self.coll, self.s, self.wall, self.scan, self.typ, self.pro, self.t = ([] for _ in range(10))
        self.done_step = np.full(self.B, -1); self.progress = np.zeros(self.B); self.collided = np.zeros(self.B, bool)
        self.lap_time = np.full(self.B, np.nan)

    def add(self, r, obs, info, step):
        scan, pro = flatten_obs(obs)
        self.state.append(r.state.cpu().numpy()); self.att.append(r.attitude.cpu().numpy()); self.lap.append(r.lap.cpu().numpy())
        self.coll.append(r.collision.cpu().numpy()); self.s.append(r.s.cpu().numpy()); self.wall.append(r.wall_dist.cpu().numpy())
        sc = torch.where(torch.isfinite(r.scan), r.scan, torch.full_like(r.scan, self.range_max))
        self.scan.append((sc / self.range_max * 255).clamp(0, 255).to(torch.uint8).cpu().numpy())
        self.typ.append(r.scan_type.to(torch.uint8).cpu().numpy()); self.pro.append(pro.cpu().numpy()); self.t.append(r.t)
        for i, lt in zip(info["lap_ids"].tolist(), info["lap_times"].tolist()):
            if self.done_step[i] < 0 and np.isnan(self.lap_time[i]): self.lap_time[i] = lt
        if "final" in info:
            f = info["final"]
            for j, i in enumerate(f["ids"].tolist()):
                if self.done_step[i] < 0:
                    self.done_step[i] = step; self.progress[i] = float(f["progress"][j]); self.collided[i] = bool(f["collided"][j])

    def finalize(self):
        T = len(self.state); self.T = T
        live = self.done_step < 0
        self.progress[live] = self.env.ep_progress.cpu().numpy()[live]
        self.done_step[live] = T
        for k in ("state", "att", "lap", "coll", "s", "wall", "scan", "typ", "pro"):
            setattr(self, k, np.stack(getattr(self, k)))
        # rank: no collision first, then progress
        order = np.lexsort((-self.progress, self.collided.astype(int)))
        self.ranking = order
        return self

    def frame(self, t):
        """StepResult-like object for the viewer at step t (all agents)."""
        import types
        dev = self.env.device
        r = types.SimpleNamespace()
        r.state = torch.from_numpy(self.state[t]).to(dev); r.attitude = torch.from_numpy(self.att[t]).to(dev)
        r.lap = torch.from_numpy(self.lap[t]).to(dev); r.collision = torch.from_numpy(self.coll[t]).to(dev)
        r.s = torch.from_numpy(self.s[t]).to(dev); r.wall_dist = torch.from_numpy(self.wall[t]).to(dev)
        r.scan = torch.from_numpy(self.scan[t].astype(np.float32) / 255 * self.range_max).to(dev)
        r.scan_type = torch.from_numpy(self.typ[t].astype(np.int32)).to(dev); r.t = self.t[t]
        return r

    def obs_at(self, t, i, k):
        idx = [max(0, t - j) for j in range(k)]
        scan = torch.from_numpy(np.stack([self.scan[j, i] for j in idx]).astype(np.float32) / 255).to(self.env.device)[None]
        return scan, torch.from_numpy(self.pro[t, i]).to(self.env.device)[None]

    def leaderboard(self, n=5):
        lines = []
        for rank, i in enumerate(self.ranking[:n]):
            lt = f"lap {self.lap_time[i]:.1f} s" if not np.isnan(self.lap_time[i]) else "no lap"
            lines.append(f"  #{rank + 1} agent {i:3d}  {self.progress[i]:6.1f} m  {lt}  {'CRASH' if self.collided[i] else 'clean'}")
        return lines


def run_episode(env, model, steps, stochastic, rec=None):
    obs, info = env.reset()
    rec = rec or EpisodeRecorder(env)
    with torch.no_grad():
        for t in range(steps):
            scan, pro = flatten_obs(obs)
            act, _ = model.act(scan, pro, deterministic=not stochastic)
            obs, rew, term, trunc, info = env.step(act)
            rec.add(env.last_result, obs, info, t)
    return rec.finalize()


def replay_best(v, rec, model, intro, ep, sink, speed_cap, fps=30, rank=0, realtime=True):
    """Play back agent `rec.ranking[rank]` from the recording with saliency + panel + leaderboard."""
    i = int(rec.ranking[rank]); v.focus = int(rec.env_ids[i]) if hasattr(rec, "env_ids") else i
    T = int(rec.done_step[i]); k = model.meta["n_stack"]
    sim_per_frame = max(1, round(1.0 / rec.env.sim.control_dt / fps))
    banner = [f"REPLAY  best of episode {ep}: agent {i}  {rec.progress[i]:.1f} m  " + (f"lap {rec.lap_time[i]:.1f} s" if not np.isnan(rec.lap_time[i]) else "") + ("  CRASHED" if rec.collided[i] else "  clean"),
              "leaderboard:"] + rec.leaderboard()
    v.prev_frame = None; v.frame = None
    t_wall = time.perf_counter()
    for t in range(0, T, sim_per_frame):
        v.update(rec.frame(t))
        scan, pro = rec.obs_at(t, i, k)
        sal, mu = intro.saliency(scan, pro, 0)
        v.point_colors = sal_colors(sal)
        v.range_max = float(rec.env.range_max)
        v.bev = bev_image(scan[0].float().cpu().numpy(), v.range_max, sal=sal)   # same panel as live
        v.extra_hud = banner
        v.mode = 0
        v.render()
        if sink: sink(v)
        if realtime:
            t_wall += 1.0 / fps; d = t_wall - time.perf_counter()
            if d > 0: time.sleep(d)


def checkpoint_speed_cap(extra: dict, ckpt_path: str, fallback: float = 6.0) -> float:
    """The speed cap the checkpoint was trained at.

    PPO writes it as `cap`; DAgger runs from before that field fall back to the run's W&B config,
    which records `speed_cap`. Watching a policy above its training cap shows it failing at a problem
    it was never asked to solve, which reads as a policy bug and is not one."""
    if extra.get("cap"):
        return float(extra["cap"])
    run_dir = os.path.dirname(ckpt_path)
    for cfg_path in sorted(glob.glob(os.path.join(run_dir, "wandb", "*", "files", "config.yaml")), reverse=True):
        try:
            import yaml
            value = (yaml.safe_load(open(cfg_path)) or {}).get("speed_cap")
            if isinstance(value, dict):
                value = value.get("value")
            if value:
                return float(value)
        except Exception:
            pass
    return fallback


def describe_checkpoint_line(extra: dict, path: str) -> str:
    """One plain line about a checkpoint: which run, which iteration / update, how it was doing.

    Was a closure inside `main`; the viewer console's worker process needs the same sentence when
    it only wants to *describe* a checkpoint, without building an environment around it. Same text
    as before -- `main` calls this now rather than its own copy.
    """
    run = extra.get("run") or os.path.basename(os.path.dirname(path))
    m = extra.get("metrics") or {}
    if extra.get("phase") == "ppo" or "update" in extra or os.path.basename(path).startswith("ppo"):
        u, n = extra.get("update"), extra.get("updates")
        # A resumed run keeps counting the current leg in `steps` while `total_steps` carries
        # everything since the first launch, and the two can be a hundred million apart. Quoting
        # the leg as if it were the run's history makes a long-trained policy look barely started,
        # so prefer the total and label the fallback for what it is.
        total = extra.get("total_steps")
        steps = (f"  {float(total) / 1e6:.1f}M steps" if total
                 else f"  {extra.get('steps', 0) / 1e6:.1f}M steps (this leg)")
        core = (f"PPO {run}" + (f"  update {u}" + (f" of {n}" if n else "") if u is not None else "")
                + steps
                + (f"  cap {extra['cap']:.1f} m/s" if "cap" in extra else ""))
    else:
        it, n = extra.get("iter"), extra.get("iters")
        core = (f"DAgger {run}" + (f"  iteration {it + 1}" + (f" of {n}" if n else "") if it is not None else "")
                + (f"  {extra['samples'] / 1e6:.1f}M samples" if "samples" in extra else ""))
    if m:
        core += (f"   |  at save: collision {m.get('collision_rate', float('nan')):.2f}"
                 + (f", {m['progress_rate_mps']:.1f} m/s" if "progress_rate_mps" in m else "")
                 + (f", lap {m['lap_time_s']:.1f} s"
                    if m.get("lap_time_s") == m.get("lap_time_s") and m.get("lap_time_s") else ""))
    return core


def latest_run() -> str:
    """The most recently updated run directory in ~/f1sim_runs that holds a checkpoint."""
    best, t_best = "", -1.0
    for d in glob.glob(os.path.join(common.RUNS_DIR, "*")):
        for ck in ("ppo_latest.pt", "student_latest.pt"):
            p = os.path.join(d, ck)
            if os.path.exists(p) and os.path.getmtime(p) > t_best:
                best, t_best = d, os.path.getmtime(p)
    return best


def default_consumer_refusal(path: str) -> Optional[str]:
    """Why `load_checkpoint(path, ...)` with no opt-in flag would refuse, or None if the **metadata**
    is supported.

    A metadata-only mirror of the guards in `model.py`, for callers that must *choose* a checkpoint
    before trying to load one:

      * a checkpoint-shaped file at all -- a mapping carrying `meta` and `state_dict`;
      * `residual_plan` (`model.py:372-374`);
      * `_refuse_controller` (`model.py:324-341`) -- a non-`legacy` recorded controller arm;
      * the conditional gate (`model.py:384`) -- `cond_dim > 0` or a lab-oracle source, including the
        `CondSpec.from_meta` parse and its dim-consistency check;
      * an `act_dim` this viewer cannot drive (`2` or the current plan width).

    **`None` means the metadata is supported, not that the weights are valid.** Shapes, migration and
    strictness are the loader's business and it still decides; nothing here relaxes any guard, and a
    caller that ignores this and loads directly gets exactly the same refusal. This exists so that
    "open the newest run" can mean "the newest run this viewer can open".

    `mmap=True, weights_only=True` keeps it cheap and safe: tensor storages are never faulted in, so
    probing every run reads metadata rather than hundreds of megabytes, and nothing in the file is
    executed. **Fails closed** -- unreadable, truncated, mid-write or unexpectedly shaped returns a
    reason, never None.
    """
    import torch

    from .model import ActorCritic
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as exc:                       # truncated, mid-write, or not a checkpoint
        return f"unreadable ({type(exc).__name__})"
    try:
        if not hasattr(ck, "get") or not hasattr(ck, "__contains__"):
            return "not a checkpoint (no mapping)"
        if "meta" not in ck or "state_dict" not in ck:
            missing = [k for k in ("meta", "state_dict") if k not in ck]
            return f"not a checkpoint (missing {', '.join(missing)})"
        meta = ck.get("meta")
        if not hasattr(meta, "get") or not hasattr(meta, "items"):
            return f"malformed meta ({type(meta).__name__})"
        if not hasattr(ck.get("state_dict"), "keys"):
            return f"malformed state_dict ({type(ck.get('state_dict')).__name__})"
        if meta.get("residual_plan"):
            return "experimental residual-plan checkpoint"
        # The constructor arguments have to be present and acceptable, checked the way the loader
        # would build them -- `dict(meta)` minus `residual_plan` -- but WITHOUT building the network.
        # `signature.bind` catches both a missing required argument (`meta={}` names none of
        # n_stack/n_beams/proprio_dim/priv_dim) and an unknown key, which would be a TypeError deep
        # inside `load_checkpoint` rather than a decision this selector could have made.
        import inspect
        ctor = dict(meta)
        ctor.pop("residual_plan", None)
        try:
            inspect.signature(ActorCritic).bind(**ctor)
        except TypeError as exc:
            return f"metadata does not fit the model constructor ({exc})"
        exp = ((ck.get("extra") or {}).get("experiment") or {})
        arm = str(((exp.get("controller") or {}).get("arm")) or "legacy")
        if arm != "legacy":
            return f"controller arm {arm!r}"
        from .conditioning import CondSpec
        cond_meta = CondSpec.from_meta(meta.get("cond")).to_meta()   # raises on an inconsistent spec
        cond_dim = meta.get("cond_dim", 0)
        if cond_dim is None or isinstance(cond_dim, bool) or not isinstance(cond_dim, int):
            return f"malformed cond_dim ({cond_dim!r})"     # None must not read as 0
        if cond_dim != int(cond_meta.get("dim", 0) or 0):
            return (f"conditioning metadata disagrees (cond_dim={cond_dim}, "
                    f"spec dim={cond_meta.get('dim')})")
        if cond_dim:
            return f"conditional (cond_dim={cond_dim})"
        if cond_meta.get("lab_oracle"):
            return "conditional (lab-oracle input)"
        from ..mpc import ACT_DIM as PLAN_DIM
        act_dim = meta.get("act_dim", 2)
        if isinstance(act_dim, bool) or not isinstance(act_dim, int):
            return f"malformed act_dim ({act_dim!r})"       # 8.5 must not truncate to 8
        if act_dim not in (2, PLAN_DIM):
            return f"action dim {act_dim} (this viewer drives 2 or {PLAN_DIM})"
    except Exception as exc:                       # any malformed metadata fails closed
        return f"unsupported metadata ({type(exc).__name__}: {exc})"
    return None


def latest_compatible_run() -> tuple:
    """(newest run dir the default consumer can open, [(run, why skipped), ...] newest first).

    **Ordering note.** `latest_run()` ranks runs by the newest of `ppo_latest.pt` and
    `student_latest.pt`. This ranks each run by the mtime of the file that would actually be
    **opened** -- `ppo_latest.pt` when it exists, otherwise `student_latest.pt` -- and probes that
    same file. The two therefore disagree for a run holding a fresh student beside a stale ppo
    checkpoint, and this one matches what a caller will really load.

    `latest_run()` itself is untouched and still means "newest", because the training side depends
    on that meaning. Callers are expected to **say** which run they opened and which newer ones they
    passed over: a silent substitution would let someone believe they are watching the policy they
    just trained.
    """
    cands = []
    for d in glob.glob(os.path.join(common.RUNS_DIR, "*")):
        files = [os.path.join(d, ck) for ck in ("ppo_latest.pt", "student_latest.pt")]
        files = [p for p in files if os.path.exists(p)]
        if files:
            # Rank by the mtime of the file that will ACTUALLY be opened -- `ppo_latest.pt` when it
            # exists. Ranking by the newest file in the directory lets a recent `student_latest.pt`
            # promote a run whose stale `ppo_latest.pt` is the one that gets loaded.
            cands.append((os.path.getmtime(files[0]), d, files[0]))
    skipped = []
    for _, d, p in sorted(cands, key=lambda c: c[0], reverse=True):
        why = default_consumer_refusal(p)
        if why is None:
            return d, skipped
        skipped.append((os.path.basename(d), why))
    return "", skipped


def viewer_config(compile_enabled: bool, randomize: bool = True) -> Config:
    config = Config()
    config.sim.compile = compile_enabled
    config.sim.compile_mode = "default" if compile_enabled else "none"
    # Randomisation is what training sees, so it is the honest default. But it is per car, per
    # reset -- friction 0.74 on one car and 1.15 on the next -- and a viewer with one car on screen
    # cannot show that the slide you are watching is a low-grip draw and not the policy.
    config.rand.enabled = randomize
    return config


def actor_runner(model, device: torch.device, compile_enabled: bool):
    import copy as _copy
    actor = _copy.deepcopy(model.actor).eval()
    for module in actor.modules():
        module._forward_hooks.clear(); module._forward_pre_hooks.clear()
    if not compile_enabled or device.type != "cuda":
        return actor.forward
    compiled = torch.compile(actor.forward, dynamic=False, mode="reduce-overhead")
    state = {"call": compiled}

    def run(scan, proprio):
        try:
            return state["call"](scan, proprio)
        except RuntimeError:
            # Falling back keeps the session alive, which is right -- but silently, a session that
            # reports `compile: true` would go on running eager and its timings would be quoted as
            # compiled ones. The behaviour is unchanged; only the fact is now recorded.
            state["call"] = actor.forward
            run.fell_back_to_eager = True
            return actor(scan, proprio)

    run.compiled = True
    run.fell_back_to_eager = False
    return run


def viewer_threaded(compile_enabled: bool) -> bool:
    return not compile_enabled


def main(argv=None):
    # No arguments: open the driving console. It replaces the old three-piece interactive path (a
    # modal Tk launcher, a second always-on-top Tk map panel in its own process talking through a
    # file in ~/.cache, and a GLFW window whose only controls were undocumented single keys).
    #
    # Everything below this is unchanged and still reached the same way: `--record`, `--frames`,
    # `--bench`, `--episodes`, `--highlights` and every flag they use behave exactly as before.
    # `--legacy-launcher` still opens the old Tk picker for anyone who wants it.
    if argv is None and len(sys.argv) == 1:
        from ..viewer.console import launch
        return launch()
    if argv is None and len(sys.argv) == 2 and sys.argv[1] in ("--console", "--gui"):
        from ..viewer.console import launch
        return launch([sys.argv[0]])
    if argv is None and len(sys.argv) == 2 and sys.argv[1] == "--legacy-launcher":
        from .watch_gui import ask
        argv = ask()
        if argv is None:
            return
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="latest", help="run dir (uses ppo_latest.pt / student_latest.pt), a .pt file, or 'latest' = the newest run")
    ap.add_argument("--map", default="gen:competition:2",
                    help="one map, or several comma separated: the viewer then switches between them with M / N "
                         "without restarting (every car moves to the shown track)")
    ap.add_argument("--map-sets", default="",
                    help="named sets over the loaded maps, 'name=a,b,c;name2=d,e'. M / N walks the "
                         "active set only and G cycles sets, so a held-out map and a training map "
                         "are never one keypress apart -- they answer different questions and the "
                         "picker exists to keep them separable. Maps named here are loaded too")
    ap.add_argument("--map-set", default="", help="which of --map-sets starts active")
    ap.add_argument("--cars", type=int, default=1,
                    help="parallel simulations drawn at once. These are independent runs, not "
                         "opponents -- they share nothing with the watched car -- so more of them "
                         "is more clutter around the one being looked at. Opponents are --race-size")
    ap.add_argument("--speed-cap", type=float, default=None,
                    help="[m/s] default: the cap the checkpoint was trained at (PPO/DAgger store it). Watching a "
                         "policy above its training cap shows a problem it was never asked to solve")
    ap.add_argument("--device", default="auto",
                    help="'auto' is the GPU whenever there is one. Measured beside a trainer at "
                         "100 %% utilisation, one car: 19.8 ms per step with CUDA graphs, against "
                         "56 ms on the CPU. An earlier version of this flag fell back to the CPU "
                         "when the GPU was busy, on a measurement taken with the graphs off -- what "
                         "it had actually measured was launch overhead for a thousand tiny kernels, "
                         "which is what the graphs remove, and not contention at all")
    ap.add_argument("--stochastic", action="store_true", help="sample actions like during training")
    ap.add_argument("--record", default="", help="headless: write an mp4 (ffmpeg)"); ap.add_argument("--frames", default="", help="headless: write PNG frames here")
    ap.add_argument("--seconds", type=float, default=20.0, help="recording length"); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--episodes", type=int, default=0, help="highlight mode: run this many episodes, replay the best agent of each")
    ap.add_argument("--episode-s", type=float, default=3600.0,
                    help="[s] episode time limit in the viewer. Training truncates at 40 s, and the same "
                         "limit here made the car vanish and respawn mid-lap for no visible reason; "
                         "watching wants the run to go on until it actually hits something")
    ap.add_argument("--replay-top", type=int, default=1, help="replay the top-k agents")
    ap.add_argument("--highlights", default="", help="headless highlight mode: directory for one mp4 per episode")
    ap.add_argument("--internals", action="store_true", help="also show the raw hidden-layer / conv-feature activations")
    ap.add_argument("--no-bev", dest="bev", action="store_false",
                    help="hide the BEV panel (top-left). It shows the policy's newest scan from above,"
                         " coloured by per-beam saliency, with its plan on the same axes -- what the "
                         "network was handed and what it did with it, in one picture")
    ap.set_defaults(bev=True)
    ap.add_argument("--bev-span", type=float, default=8.0,
                    help="[m] half-width of the BEV panel, true scale (10 = the full sensor range)")
    ap.add_argument("--panel-every", type=int, default=3, help="recompute saliency + the panel every k sim steps (GPU launches)")
    ap.add_argument("--bench", type=float, default=0.0, help="headless: run the interactive loop (threaded, real-time paced) for this many seconds and report the sim rate")
    ap.add_argument("--gl", default="nvidia", choices=["nvidia", "amd"], help="GPU for the window's OpenGL: 'amd' renders on the integrated Radeon through Mesa (PRIME offload) and leaves the NVIDIA GPU to the simulation")
    ap.add_argument("--race-size", type=int, default=1, help="cars per race (>1: opponents in the scan)")
    ap.add_argument("--opponent", default="teacher", choices=["teacher", "policy"], help="who drives the other cars of a race")
    ap.add_argument("--no-dr", dest="randomize", action="store_false",
                    help="nominal vehicle parameters for every car (friction 1.05, no delay/gain "
                         "randomisation). Default: randomised per car and per reset, as in training")
    ap.set_defaults(randomize=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false",
                    help="skip the CUDA graphs. They cost ~17 s at startup and are worth it: a 1-car "
                         "step is 19.8 ms with them and 71 ms without, because at this batch size "
                         "every kernel is far smaller than the cost of launching it. Beside a "
                         "trainer as well -- the launches are the viewer's own CPU time, not a queue "
                         "behind anyone")
    ap.add_argument("--compile", dest="compile", action="store_true",
                    help=argparse.SUPPRESS)      # now the default; still accepted so older launchers work
    ap.set_defaults(compile=True)
    a = ap.parse_args(argv)
    if a.cars < a.race_size:
        raise SystemExit(f"--cars {a.cars} is fewer than --race-size {a.race_size}: a race needs at least one full "
                         f"grid of cars. Raise cars to a multiple of {a.race_size}.")
    if a.gl == "amd":                                            # must be set before the first GLX call (window creation)
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "mesa"; os.environ["DRI_PRIME"] = "1"
        os.environ.pop("__NV_PRIME_RENDER_OFFLOAD", None)
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {a.device}", flush=True)
    device = torch.device(a.device)
    # Nothing to compile on the CPU: inductor leaves this graph of hundreds of tiny ops alone (0.8 s
    # of "warmup" and 64 ms per step either way), and claiming otherwise would only cost the viewer
    # its render thread, which the flag also controls.
    a.compile = a.compile and device.type == "cuda"
    if a.run == "latest":
        # "latest" means the newest run this viewer can actually open. `latest_run()` means the
        # newest run, full stop, and keeps that meaning for the training side -- but once
        # controller-arm or conditional training is running, the newest run is one the default
        # loader refuses, and the default invocation would fail on the checkpoint the user is least
        # surprised by. Announce the choice: silently opening an older run would let someone believe
        # they are watching the policy they just trained.
        run, skipped = latest_compatible_run()
        for name, why in skipped:
            print(f"skipping {name}: {why} -- this viewer runs the default controller", flush=True)
        if not run and skipped:
            raise SystemExit(
                f"no run under {common.RUNS_DIR} can be opened with the default controller.\n"
                + "\n".join(f"  {n}: {w}" for n, w in skipped)
                + "\n\nPass --run <name> (or --run /path/to/ppo_final.pt) to choose one "
                  "explicitly; an incompatible one will say why.")
    else:
        run = a.run
    if not run:
        raise SystemExit(f"no run with a checkpoint under {common.RUNS_DIR}")
    if run.endswith(".pt"):
        ckpt_path = run
    else:
        # A bare name is resolved under RUNS_DIR, like every other consumer does it. The old
        # `next(...)` genexp raised a bare `StopIteration` when neither file was there -- for a
        # mistyped `--run`, the user got a traceback with no name in it.
        base = run if os.path.isabs(run) else os.path.join(common.RUNS_DIR, run)
        ckpt_path = next((p for p in (os.path.join(base, "ppo_latest.pt"),
                                      os.path.join(base, "student_latest.pt")) if os.path.exists(p)),
                         "")
        if not ckpt_path:
            raise SystemExit(
                f"no checkpoint in {base}: expected ppo_latest.pt or student_latest.pt.\n"
                f"Pass --run with a run name under {common.RUNS_DIR}, an absolute run directory, "
                f"or a path to a .pt file.")
    if a.run == "latest":
        print(f"run {os.path.basename(run)}  ({os.path.basename(ckpt_path)})", flush=True)
    model, extra = load_checkpoint(ckpt_path, device); model.eval(); intro = Introspector(model)
    act_dim = model.meta.get("act_dim", 2)
    from ..mpc import ACT_DIM as PLAN_DIM
    if act_dim not in (2, PLAN_DIM):
        raise SystemExit(f"{ckpt_path}: action dim {act_dim} is from an older plan representation (the current one has {PLAN_DIM}); pick a newer run")
    if a.speed_cap is None:
        a.speed_cap = checkpoint_speed_cap(extra, ckpt_path)
        print(f"speed cap {a.speed_cap:.1f} m/s (from the checkpoint)", flush=True)
    mode = "plan" if act_dim == PLAN_DIM else "direct"                       # plan-space policies drive through the tracker
    sp = extra.get("spec") or {}                                              # the observation layout the policy was trained with
    need_rl = a.race_size > 1 and a.opponent == "teacher"
    map_names = [m.strip() for m in a.map.split(",") if m.strip()]
    # named sets, and the flat load order they index into
    map_sets = {}
    for part in (p_ for p_ in a.map_sets.split(";") if p_.strip()):
        label, _, body = part.partition("=")
        names = [m.strip() for m in body.split(",") if m.strip()]
        if names:
            map_sets[label.strip()] = names
    for names in map_sets.values():
        map_names += [n for n in names if n not in map_names]
    # drop_infeasible=False so the loaded order matches map_names one for one, which is what the set
    # indices below address. Dropping is right for training -- a track whose raceline does not fit
    # teaches the student to drive into it -- but here the raceline is only drawn, and a picker whose
    # sets silently renumber themselves is worse than a badly drawn line.
    tracks, rls = common.load_tracks(map_names, racelines=need_rl, drop_infeasible=False)
    order = {n: i for i, n in enumerate(map_names)}
    track_groups = {k: [order[n] for n in v if n in order] for k, v in map_sets.items()}
    track_groups = {k: v for k, v in track_groups.items() if v}
    if rls is None:                                  # the viewer draws the raceline even without teacher opponents
        try:
            from ..raceline import Raceline
            rls = [Raceline.build_cached(t) for t in tracks]
        except Exception:
            rls = None
    cfg = viewer_config(a.compile, randomize=a.randomize)
    n_cars = a.cars - a.cars % a.race_size
    env = common.make_env(tracks, n_cars, device, EnvConfig(speed_cap=a.speed_cap, action_mode=mode, race_size=a.race_size, opponent=a.opponent,
                                                             max_steps=int(a.episode_s * 40),
                                                             # opponents at the policy's own cap and the teacher at full
                                                             # raceline pace: the viewer shows the policy against an equal
                                                             selfplay_front_cap=False, opp_speed_range=(1.0, 1.0),
                                                             scan_stack=sp.get("scan_stack", 3), scan_stride=sp.get("scan_stride", 1),
                                                             hist_len=sp.get("hist_len", 0), hist_stride=sp.get("hist_stride", 2),
                                                             compile_tracker=a.compile,
                                                             # a switch moves every car itself; re-drawing tracks on
                                                             # reset would scatter them back over the whole set
                                                             resample_track_on_reset=False),
                          cfg=cfg, rls=rls if need_rl else None)
    # every car starts on the map that was asked for: the simulator otherwise deals the envs round
    # robin over the whole loaded set, which with ten maps loaded for M / N switching leaves one car
    # per map on screen. Switching moves them all again (viewer.pending_track).
    env.sim.tid.fill_(0)
    print(f"{len(map_names)} map(s), {n_cars} cars, cap {a.speed_cap} m/s"
          + (f", races of {a.race_size} vs {a.opponent}" if a.race_size > 1 else "")
          + (f"   |  M / N switches between: {', '.join(map_names)}"
             if len(map_names) > 1 and not track_groups else ""), flush=True)
    describe = describe_checkpoint_line                       # module level now; see its docstring
    mtime = os.path.getmtime(ckpt_path); step_info = describe(extra, ckpt_path)
    obs, info = env.reset()
    env.sim.warmup()
    from ..viewer.native import NativeViewer
    headless = bool(a.record or a.frames or a.highlights or a.bench)
    v = NativeViewer(env.sim, headless=headless, max_cars=a.cars, width=1600, height=900)
    if track_groups:
        v.track_groups = track_groups
        v.set_group(a.map_set if a.map_set in track_groups else next(iter(track_groups)))
        print("map sets: " + "   ".join(f"{k} ({len(idx)})" for k, idx in track_groups.items())
              + f"   |  active: {v.track_group_name}   (M / N within the set, G switches set)", flush=True)
    if not headless:
        common.viewer_heartbeat()                    # tell any training job to give back a slice
        import atexit; atexit.register(common.viewer_gone)
    state = {"obs": obs, "last_reload": time.time(), "model": model, "extra": extra}

    # ---------------- highlight mode: episodes -> replay of the best agent(s)
    if a.episodes or a.highlights:
        n_ep = a.episodes or 1; steps = int(a.episode_s / env.sim.control_dt)
        if a.highlights: os.makedirs(a.highlights, exist_ok=True)
        for ep in range(n_ep):
            rec = run_episode(env, model, steps, a.stochastic)
            print(f"episode {ep}: best agent {rec.ranking[0]} progress {rec.progress[rec.ranking[0]]:.1f} m, "
                  f"{int((~rec.collided).sum())}/{env.B} clean, mean progress {rec.progress.mean():.1f} m", flush=True)
            for rank in range(a.replay_top):
                proc = None
                if a.highlights:
                    i = int(rec.ranking[rank]); tag = f"ep{ep:03d}_rank{rank + 1}_agent{i:03d}_{rec.progress[i]:.0f}m"
                    out = os.path.join(a.highlights, tag + ".mp4")
                    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{v.width}x{v.height}",
                                             "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], stdin=subprocess.PIPE)
                def sink(vv, proc=proc):
                    if proc is None: return
                    if vv.scene.fbo_ms is not None: vv.scene.ctx.copy_framebuffer(vv.scene.fbo, vv.scene.fbo_ms)
                    proc.stdin.write(Image.frombytes("RGB", (vv.width, vv.height), vv.scene.fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM).tobytes())
                replay_best(v, rec, model, intro, ep, sink if proc else (lambda vv: None), a.speed_cap, a.fps, rank, realtime=not headless)
                if proc is not None:
                    proc.stdin.close(); proc.wait(); print("wrote", out, flush=True)
            if not headless and not v.alive: break
        if not headless: v.close()
        return

    state["act_fn"] = actor_runner(model, device, a.compile)

    def step():
        nonlocal mtime, step_info
        if v.pending_track is not None:                  # M / N in the viewer: move every car to the shown track
            env.sim.tid.fill_(v.pending_track)
            state["obs"], _ = env.reset()
            v.pending_track = None
        if time.time() - state["last_reload"] > 5.0:
            state["last_reload"] = time.time()
            try:
                mt = os.path.getmtime(ckpt_path)
                if mt != mtime:
                    m2, ex = load_checkpoint(ckpt_path, device); m2.eval(); state["model"], state["extra"] = m2, ex
                    intro.__init__(m2); mtime = mt; step_info = describe(ex, ckpt_path); state["act_fn"] = actor_runner(m2, device, a.compile)
            except Exception:
                pass
        m = state["model"]
        prof = state.setdefault("prof", {}); tp = time.perf_counter
        def lap(name, t0):
            if a.bench:
                prof[name] = prof.get(name, 0.0) + tp() - t0        # CPU time only: a sync here would itself stall the sim thread
            return tp()
        t0 = tp()
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad():
            mu_all = state["act_fn"](scan, pro).clone()
            act = mu_all if not a.stochastic else (mu_all + m.actor.log_std.exp() * torch.randn_like(mu_all)).clamp(-1, 1)
        t0 = lap("act", t0)
        state["k"] = state.get("k", 0) + 1
        t0 = lap("saliency", t0)
        state["obs"], rew, term, trunc, info = env.step(act)
        t0 = lap("env.step", t0)
        fc = v.focus
        # everything the overlays need, as GPU clones (no host sync on this thread); the render thread
        # runs the saliency backward pass, the critic and the PIL drawing at its own pace
        if state["k"] % 2 == 0:
            # Everything here stays a GPU tensor. Reading one scalar with float() or .item() drains
            # the whole device queue first, and next to a training job that queue is deep: measured,
            # the viewer took 125 ms per step whether it drew 1 car or 64, which is a wait, not work.
            # Two of those scalars were added with the g-g diagram and cost more than the diagram.
            src = {"scan": scan[fc:fc + 1].clone(), "pro": pro[fc:fc + 1].clone(),
                   "dash": torch.cat([env.sim.state[fc, [3, 6]], env.last_cmd[fc], env.speed_cap[fc:fc + 1],
                                      env.sim.ay[fc:fc + 1], env.sim.ax[fc:fc + 1],
                                      env.sim.P["mu"][fc:fc + 1] * 9.81]).clone(),
                   "plan": env.tracker.last_ref[fc].clone() if mode == "plan" else None,
                   "pose": env.sim.state[fc, :3].clone() if mode == "plan" else None,
                   "plan_pred": env.tracker.last_pred[fc].clone() if mode == "plan" and env.tracker.last_pred is not None else None,
                   "priv": env.privileged(env.last_result)[fc:fc + 1].clone() if state["k"] % (2 * a.panel_every) == 0 else None,
                   "model": m, "info": step_info}
            v.overlay_src = src
        lap("overlay data", t0)
        age = time.time() - mtime
        v.extra_hud = ["", f"POLICY  {step_info}",
                       f"        file {os.path.basename(ckpt_path)}, saved {age / 60:.0f} min ago (auto-reloads)   output: " + ("local plan -> iLQR tracker" if mode == "plan" else "steer + speed")]
        # The world-frame plan lines and the colour scale used to be built here with .cpu() and
        # float(), i.e. three more device drains per step. They are derived on the render thread now,
        # from the tensors already in overlay_src.
        return env.last_result

    # capture every CUDA graph (actor, tracker, LiDAR post-processing) in the main thread: the sim thread
    # cannot record new graphs and would silently fall back to eager kernels (3x slower). Pump the
    # window between steps: this takes ten to twenty seconds, and a window that never drains its
    # event queue is one the desktop paints "not responding" over.
    for k in range(8):
        v.keep_alive(f"compiling CUDA graphs ... {k + 1}/8   (first run of a checkpoint, ~15 s)")
        step()
    v.keep_alive("")
    vmax, smax = env.ecfg.v_max_policy, env.s_max
    cache = {}
    gg_trail = collections.deque(maxlen=90)                   # ~4 s of g-g history at the dash rate
    def dash_fn():
        src = getattr(v, "overlay_src", None)
        if src is None or time.perf_counter() - cache.get("t_dash", 0.0) < 0.1: return v.dash
        cache["t_dash"] = time.perf_counter()
        d = src["dash"].cpu().numpy()                         # one transfer for every scalar the
        cache["dash"] = d                                     # overlays need, on this thread
        gg_trail.append((float(d[5]), float(d[6])))           # (lateral, longitudinal) specific force
        # the world-frame plan lines and the plan's colour scale, derived here rather than on the
        # sim thread where each of them cost a device drain
        if src.get("plan") is not None and src.get("pose") is not None:
            pose = src["pose"].cpu().numpy()
            c_, s_ = math.cos(pose[2]), math.sin(pose[2])
            def to_world(ref):
                r = ref.cpu().numpy()
                return np.stack([pose[0] + r[:, 0] * c_ - r[:, 1] * s_,
                                 pose[1] + r[:, 0] * s_ + r[:, 1] * c_, r[:, 3]], 1)
            v.plan = to_world(src["plan"])
            v.plan_pred = to_world(src["plan_pred"]) if src.get("plan_pred") is not None else None
            v.color_v_max = float(d[4])                       # the cap in force
        return dash_image(float(d[0]), float(d[3]), float(d[4]), float(d[1]), float(d[2]), vmax, smax,
                          gg=list(gg_trail), mu_g=float(d[7]))
    def bev_fn():
        """The policy panel: its newest scan from above, coloured by saliency, with its plan on top.

        Rebuilt whenever the sim hands over a new frame, not on a wall-clock timer, so it steps in
        time with the scene behind it instead of drifting at its own rate. Only the saliency pass is
        throttled -- it is an autograd backward and costs far more than the drawing does; the beams
        and the plan move every frame and reuse the last attention map.
        """
        src = getattr(v, "overlay_src", None)
        if src is None: return v.bev
        if src is cache.get("bev_src"): return v.bev          # same sim frame: nothing has moved
        if time.perf_counter() - cache.get("t_bev_draw", 0.0) < 0.06: return v.bev   # 16 Hz is plenty
        cache["t_bev_draw"] = time.perf_counter()
        cache["bev_src"] = src
        now = time.perf_counter()
        if now - cache.get("t_sal", 0.0) > 0.5:               # attention map at ~2 Hz, the picture at
            cache["t_sal"] = now                              # the sim's rate
            sal, _ = intro.saliency(src["scan"], src["pro"], 0)
            cache["sal"] = sal
            v.point_colors = sal_colors(sal)                  # the 3D cloud uses the same colours
        pr_ = src["plan"].cpu().numpy() if src.get("plan") is not None else None
        return bev_image(src["scan"][0].float().cpu().numpy(), v.range_max, span=a.bev_span,
                         plan_ref=pr_, sal=cache.get("sal"), v_max=v.speed_scale())

    def internals_fn():
        src = getattr(v, "overlay_src", None)
        if src is None or cache.get("sal") is None: return v.panel
        if time.perf_counter() - cache.get("t_int", 0.0) < 0.5: return v.panel
        cache["t_int"] = time.perf_counter()
        return panel_internals(intro.h["hidden"][0].float().cpu().numpy(),
                               intro.h["stem"][0].float().cpu().numpy())

    v.dash_fn = dash_fn
    # A launcher panel (f1sim.learn.watch_gui --panel) stays open next to the window and writes what
    # it wants here; the viewer picks it up on its own thread. It has to be another process -- Tk and
    # the GL window each want the main loop -- and a file rather than a socket keeps that trivial.
    # start from the file as it is now: a command left behind by the previous session would otherwise
    # fire on the first frame and quietly move the viewer off the set that was just picked
    ctrl = {"t": os.path.getmtime(common.VIEWER_CONTROL) if os.path.exists(common.VIEWER_CONTROL) else 0.0}
    def on_frame():
        common.viewer_heartbeat()
        cmd, ctrl["t"] = common.viewer_poll_command(ctrl["t"])
        if not cmd:
            return
        if cmd.get("set") and cmd["set"] != v.track_group_name:
            v.set_group(cmd["set"])
        if cmd.get("map"):                                   # jump straight to a named map
            names = [m.strip() for m in a.map.split(",") if m.strip()]
            names += [n for grp in map_sets.values() for n in grp if n not in names]
            if cmd["map"] in names[:env.sim.track.T]:
                v.set_track(names.index(cmd["map"]))
    v.on_frame = on_frame
    v.range_max = float(env.range_max)
    if a.bev:
        v.bev_fn = bev_fn
    if a.internals:                                          # top-left, only when asked for
        v.panel_fn = internals_fn
    if a.bench:
        t_end = time.time() + a.bench; n = {"f": 0}; orig_render = v.render
        def timed_render():
            orig_render(); n["f"] += 1
            if time.time() > t_end: v.alive = False
        v.render = timed_render; t0 = time.time(); s0 = env.sim.t
        v.run(step, realtime=not a.fast, threaded=viewer_threaded(a.compile))
        dt = time.time() - t0
        print(f"bench: sim {(env.sim.t - s0) / dt:.2f}x real time, render {n['f'] / dt:.1f} fps, {a.cars} cars, {dt:.0f} s", flush=True)
        k = max(1, state.get("k", 1)); print("  per step:", {kk: f"{vv / k * 1e3:.1f} ms" for kk, vv in state.get("prof", {}).items()}, flush=True)
        v.close(); torch.cuda.synchronize() if device.type == "cuda" else None; os._exit(0)
    if not headless:
        try:
            v.run(step, realtime=not a.fast, threaded=viewer_threaded(a.compile))
            v.close(); torch.cuda.synchronize() if device.type == "cuda" else None
            os._exit(0)                                          # skip interpreter teardown: CUDA-graph state from the sim thread crashes it
        except KeyboardInterrupt:
            pass
        v.close(); return
    # ---- headless recording
    n_frames = int(a.seconds * a.fps); sim_per_frame = max(1, round(1.0 / env.sim.control_dt / a.fps))
    proc = None
    if a.record:
        proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{v.width}x{v.height}",
                                 "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.record], stdin=subprocess.PIPE)
    if a.frames:
        os.makedirs(a.frames, exist_ok=True)
    for i in range(n_frames):
        for _ in range(sim_per_frame):
            v.update(step())
        if i % (a.fps * 4) == 0: v.mode = (v.mode + 1) % 5
        v.render()
        if proc is not None or a.frames:
            from PIL import Image as _I
            if v.scene.fbo_ms is not None: v.scene.ctx.copy_framebuffer(v.scene.fbo, v.scene.fbo_ms)
            data = v.scene.fbo.read(components=3)
            img = _I.frombytes("RGB", (v.width, v.height), data).transpose(_I.FLIP_TOP_BOTTOM)
            if proc is not None: proc.stdin.write(img.tobytes())
            if a.frames and i % a.fps == 0: img.save(os.path.join(a.frames, f"f{i:05d}.png"))
    if proc is not None:
        proc.stdin.close(); proc.wait(); print("wrote", a.record)


if __name__ == "__main__":
    main()
