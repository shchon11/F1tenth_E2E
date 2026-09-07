"""Watch a policy while it trains: runs N agents on a track with the latest checkpoint of a run
(auto-reloads when the file changes) in the native viewer, with the network's inner state on screen:
  * the focus car's LiDAR points colored by saliency |d action / d beam| (what the policy looks at)
  * hidden-layer activations (256 units) as a heat grid, action mean/std gauges, value estimate
  * checkpoint step count in the HUD
Headless recording: --record out.mp4 (needs ffmpeg) or --frames dir/ for PNGs.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
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
    c = np.zeros((len(sal), 4), np.float32)
    c[:, 0] = np.clip(0.2 + 1.2 * sal, 0, 1); c[:, 1] = np.clip(1.0 - 1.4 * sal, 0, 1) * 0.9; c[:, 2] = 0.9 * (1 - sal); c[:, 3] = 1.0
    return c


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


def panel_image(sal, mu, std, value, speed_cap, step_info, plan_ref=None, cmd=None, angles=None, fov=4.71238898):
    """The policy panel. Left: where the network is looking (saliency of the scan, per direction).
    Right: what it decided -- the planned path + speed profile (plan action space) or the steer /
    speed command (direct), with the exploration noise. Bottom: what it was given."""
    W, H = 480, 300
    img = Image.new("RGBA", (W, H), (12, 15, 22, 190)); d = ImageDraw.Draw(img)
    from ..viewer.gl_scene import _font
    f, fs, ft = _font(15), _font(13), _font(11)
    d.text((10, 6), "where the policy looks", font=f, fill=(255, 255, 255, 255))
    # polar attention plot: car in the centre, bar length = strongest saliency in that 5 deg sector
    cx, cy, R = 105, 130, 85
    n = len(sal); ang = np.linspace(-fov / 2, fov / 2, n) if angles is None else np.asarray(angles)
    bins = 54; edges = np.linspace(-fov / 2, fov / 2, bins + 1)
    d.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(70, 80, 100, 255))
    d.ellipse((cx - R / 2, cy - R / 2, cx + R / 2, cy + R / 2), outline=(50, 58, 75, 255))
    for k in range(bins):
        m = (ang >= edges[k]) & (ang < edges[k + 1])
        if not m.any(): continue
        v = float(sal[m].max()); a_ = 0.5 * (edges[k] + edges[k + 1])
        ex, ey = cx + (14 + v * (R - 16)) * np.sin(a_), cy - (14 + v * (R - 16)) * np.cos(a_)   # forward = up
        col = (int(255 * min(1, 0.2 + 1.2 * v)), int(230 * max(0, 1 - 1.4 * v)), int(200 * (1 - v)), 255)
        d.line((cx + 12 * np.sin(a_), cy - 12 * np.cos(a_), ex, ey), fill=col, width=3)
    d.polygon([(cx, cy - 10), (cx - 6, cy + 7), (cx + 6, cy + 7)], fill=(235, 235, 240, 255))   # the car, nose up
    d.text((10, 222), "red sectors: beams that change the decision most", font=ft, fill=(255, 160, 130, 255))
    d.text((10, 236), "(same colours on the 3D scan points)", font=ft, fill=(170, 180, 200, 255))
    # decision
    d.text((220, 6), "what it decided", font=f, fill=(255, 255, 255, 255))
    if plan_ref is not None:                                   # planned path, top-down, forward = up
        px, py, pw, ph = 225, 28, 130, 150
        d.rectangle((px, py, px + pw, py + ph), outline=(70, 80, 100, 255))
        xs, ys, vs = plan_ref[:, 0], plan_ref[:, 1], plan_ref[:, 3]
        scale = (ph - 20) / max(1.0, float(np.abs(xs).max()) + 0.3)
        pts = [(px + pw / 2 - y * scale, py + ph - 10 - x * scale) for x, y in zip(xs, ys)]
        for i in range(len(pts) - 1):
            t = min(1.0, vs[i] / 8.0)
            col = (int(255 * t) if t > 0.5 else int(51 + 0 * t), int(140 + 110 * min(1, 2 * t)) if t < 0.5 else int(250 - 80 * (t - 0.5) * 2), int(255 * (1 - 2 * t)) if t < 0.5 else 40, 255)
            d.line((pts[i], pts[i + 1]), fill=col, width=4)
        d.polygon([(px + pw / 2, py + ph - 16), (px + pw / 2 - 5, py + ph - 4), (px + pw / 2 + 5, py + ph - 4)], fill=(235, 235, 240, 255))
        d.text((px, py + ph + 4), f"planned path, next {max(0.1, float(np.linalg.norm(plan_ref[-1, :2]))):.1f} m; colour = speed", font=ft, fill=(200, 210, 225, 255))
        d.text((365, 30), "planned speed", font=fs, fill=(230, 230, 235, 255))
        d.text((365, 48), f"now  {vs[3]:.1f} m/s", font=fs, fill=(230, 230, 235, 255))
        d.text((365, 66), f"end  {vs[-1]:.1f} m/s", font=fs, fill=(230, 230, 235, 255))
        d.text((365, 84), f"cap  {speed_cap:.1f} m/s", font=fs, fill=(160, 170, 190, 255))
        d.text((365, 112), "noise (exploration)", font=ft, fill=(160, 170, 190, 255))
        d.text((365, 126), f"curvature +-{float(np.mean(std[:-2])) * 1.6:.2f} 1/m", font=ft, fill=(160, 170, 190, 255))
        d.text((365, 140), f"speed +-{float(np.mean(std[-2:])) * 4:.1f} m/s", font=ft, fill=(160, 170, 190, 255))
    else:
        def gauge(y, name, m, s_, txt):
            d.text((225, y), f"{name}: {txt}", font=fs, fill=(230, 230, 235, 255))
            d.rectangle((225, y + 18, 455, y + 26), fill=(40, 45, 60, 255))
            cxg = 340 + m * 115; d.rectangle((cxg - max(2, s_ * 115), y + 18, cxg + max(2, s_ * 115), y + 26), fill=(80, 140, 255, 180))
            d.rectangle((cxg - 2, y + 16, cxg + 2, y + 28), fill=(255, 255, 255, 255))
        gauge(34, "steer command", mu[0], std[0], f"{math.degrees(mu[0] * 0.4189):+.1f} deg")
        gauge(84, "speed command", mu[1], std[1], f"{(mu[1] + 1) * 0.5 * 8.0:.2f} m/s  (cap {speed_cap:.1f})")
        d.text((225, 134), "blue band = exploration noise around the mean", font=ft, fill=(160, 170, 190, 255))
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        d.text((225, 208), f"critic's value estimate {value:6.1f}", font=ft, fill=(200, 210, 225, 255))
    if step_info:
        d.text((225, 224), step_info[:64], font=ft, fill=(180, 190, 210, 255))
    d.text((10, 258), "given: last 3 LiDAR scans, VESC speed, IMU, roll/pitch estimate, its last 2 actions", font=ft, fill=(150, 160, 180, 255))
    d.text((10, 274), "not given: map, position, opponents' positions  (LiDAR-only, end to end)", font=ft, fill=(150, 160, 180, 255))
    return img


def dash_image(v, v_cmd, v_cap, steer, steer_cmd, v_max=8.0, steer_max=0.4189):
    """Bottom-centre dash: a speedometer (needle = measured speed, orange tick = commanded speed,
    red zone above the cap) and a steering wheel turned by the actual steering angle (x3 for
    visibility, ghost tick = commanded angle)."""
    from ..viewer.gl_scene import _font
    W, H = 540, 180
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
        with torch.no_grad():
            std = model.actor.log_std.exp().cpu().numpy()
        v.panel = panel_image(sal, mu, std, None, speed_cap, f"replay t={rec.t[t]:.1f}s")
        v.extra_hud = banner
        v.mode = 0
        v.render()
        if sink: sink(v)
        if realtime:
            t_wall += 1.0 / fps; d = t_wall - time.perf_counter()
            if d > 0: time.sleep(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(common.RUNS_DIR, "ppo_v3"), help="run dir (uses ppo_latest.pt / student_latest.pt) or a .pt file")
    ap.add_argument("--map", default="gen:competition:2"); ap.add_argument("--cars", type=int, default=64)
    ap.add_argument("--speed-cap", type=float, default=6.0); ap.add_argument("--device", default="cuda")
    ap.add_argument("--stochastic", action="store_true", help="sample actions like during training")
    ap.add_argument("--record", default="", help="headless: write an mp4 (ffmpeg)"); ap.add_argument("--frames", default="", help="headless: write PNG frames here")
    ap.add_argument("--seconds", type=float, default=20.0, help="recording length"); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--episodes", type=int, default=0, help="highlight mode: run this many episodes, replay the best agent of each")
    ap.add_argument("--episode-s", type=float, default=40.0); ap.add_argument("--replay-top", type=int, default=1, help="replay the top-k agents")
    ap.add_argument("--highlights", default="", help="headless highlight mode: directory for one mp4 per episode")
    ap.add_argument("--internals", action="store_true", help="also show the raw hidden-layer / conv-feature activations")
    ap.add_argument("--panel-every", type=int, default=3, help="recompute saliency + the panel every k sim steps (GPU launches)")
    a = ap.parse_args()
    device = torch.device(a.device)
    ckpt_path = a.run if a.run.endswith(".pt") else next(p for p in (os.path.join(a.run, "ppo_latest.pt"), os.path.join(a.run, "student_latest.pt")) if os.path.exists(p))
    tracks, _ = common.load_tracks([a.map])
    model, extra = load_checkpoint(ckpt_path, device); model.eval(); intro = Introspector(model)
    mode = "plan" if model.meta.get("act_dim", 2) >= 5 else "direct"        # plan-space policies drive through the tracker
    cfg = Config(); cfg.sim.compile_mode = "reduce-overhead"                  # CUDA graphs: the sim step is one launch
    env = common.make_env(tracks, a.cars, device, EnvConfig(speed_cap=a.speed_cap, action_mode=mode), cfg=cfg)
    def describe(ex, path):
        """One plain line about the checkpoint: which run, which iteration / update, how it was doing."""
        run = ex.get("run") or os.path.basename(os.path.dirname(path)); m = ex.get("metrics") or {}
        if ex.get("phase") == "ppo" or "update" in ex or os.path.basename(path).startswith("ppo"):
            u, n = ex.get("update"), ex.get("updates")
            core = f"PPO {run}" + (f"  update {u}" + (f" of {n}" if n else "") if u is not None else "") + f"  {ex.get('steps', 0) / 1e6:.1f}M steps" + (f"  cap {ex['cap']:.1f} m/s" if "cap" in ex else "")
        else:
            it, n = ex.get("iter"), ex.get("iters")
            core = f"DAgger {run}" + (f"  iteration {it + 1}" + (f" of {n}" if n else "") if it is not None else "") + (f"  {ex['samples'] / 1e6:.1f}M samples" if "samples" in ex else "")
        if m:
            core += f"   |  at save: collision {m.get('collision_rate', float('nan')):.2f}" + (f", {m['progress_rate_mps']:.1f} m/s" if "progress_rate_mps" in m else "") + (f", lap {m['lap_time_s']:.1f} s" if m.get("lap_time_s") == m.get("lap_time_s") and m.get("lap_time_s") else "")
        return core
    mtime = os.path.getmtime(ckpt_path); step_info = describe(extra, ckpt_path)
    obs, info = env.reset()
    env.sim.warmup()
    from ..viewer.native import NativeViewer
    headless = bool(a.record or a.frames or a.highlights)
    v = NativeViewer(env.sim, headless=headless, max_cars=a.cars, width=1600, height=900)
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

    def step():
        nonlocal mtime, step_info
        if time.time() - state["last_reload"] > 5.0:
            state["last_reload"] = time.time()
            try:
                mt = os.path.getmtime(ckpt_path)
                if mt != mtime:
                    m2, ex = load_checkpoint(ckpt_path, device); m2.eval(); state["model"], state["extra"] = m2, ex
                    intro.__init__(m2); mtime = mt; step_info = describe(ex, ckpt_path)
            except Exception:
                pass
        m = state["model"]
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad():
            act, _ = m.act(scan, pro, deterministic=not a.stochastic)
        state["k"] = state.get("k", 0) + 1
        if state["k"] % a.panel_every == 1 or a.panel_every == 1:      # saliency (a backward pass) + panel: not every frame
            with torch.no_grad():
                priv = env.privileged(env.last_result)
                val = m.critic(scan[v.focus:v.focus + 1], pro[v.focus:v.focus + 1], priv[v.focus:v.focus + 1]).item()
            sal, mu = intro.saliency(scan, pro, v.focus)
            v.point_colors = sal_colors(sal)
            std = m.actor.log_std.exp().detach().cpu().numpy()
            state["panel_args"] = (sal, mu, std, val)
        state["obs"], rew, term, trunc, info = env.step(act)
        if "panel_args" in state and (state["k"] % a.panel_every == 1 or a.panel_every == 1):
            sal, mu, std, val = state["panel_args"]
            plan_ref = env.tracker.last_ref[v.focus].cpu().numpy() if mode == "plan" else None
            v.panel = panel_image(sal, mu, std, val, a.speed_cap, step_info, plan_ref=plan_ref, cmd=env.last_cmd[v.focus].cpu().numpy() if mode == "plan" else None)
            if a.internals:
                pi = panel_internals(intro.h["hidden"][0].float().cpu().numpy(), intro.h["stem"][0].float().cpu().numpy())
                both = Image.new("RGBA", (480, 300 + 130), (0, 0, 0, 0)); both.paste(v.panel, (0, 0)); both.paste(pi, (0, 300)); v.panel = both
        fc = v.focus
        st_ = torch.cat([env.sim.state[fc, [3, 6]], env.last_cmd[fc]]).cpu().numpy()           # speed, steer, cmd steer, cmd speed
        v.dash = dash_image(float(st_[0]), float(st_[3]), float(env.speed_cap[fc]), float(st_[1]), float(st_[2]), env.ecfg.v_max_policy, env.s_max)
        age = time.time() - mtime
        v.extra_hud = ["", f"POLICY  {step_info}",
                       f"        file {os.path.basename(ckpt_path)}, saved {age / 60:.0f} min ago (auto-reloads)   output: " + ("local plan -> iLQR tracker" if mode == "plan" else "steer + speed")]
        if mode == "plan":                                       # the focus car's plan, coloured by its speed profile
            v.plan = env.plan_world(v.focus); v.plan_pred = env.plan_world(v.focus, predicted=True)
        return env.last_result

    if not headless:
        try:
            v.run(step, realtime=not a.fast)
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
