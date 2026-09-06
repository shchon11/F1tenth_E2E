"""Watch a policy while it trains: runs N agents on a track with the latest checkpoint of a run
(auto-reloads when the file changes) in the native viewer, with the network's inner state on screen:
  * the focus car's LiDAR points colored by saliency |d action / d beam| (what the policy looks at)
  * hidden-layer activations (256 units) as a heat grid, action mean/std gauges, value estimate
  * checkpoint step count in the HUD
Headless recording: --record out.mp4 (needs ffmpeg) or --frames dir/ for PNGs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..gym_env import EnvConfig
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


def panel_image(hidden, stem, mu, std, value, speed_cap, step_info):
    W, H = 480, 300
    img = Image.new("RGBA", (W, H), (12, 15, 22, 190)); d = ImageDraw.Draw(img)
    from ..viewer.gl_scene import _font
    f, fs = _font(15), _font(13)
    d.text((10, 6), "policy internals", font=f, fill=(255, 255, 255, 255))
    d.text((10, 232), step_info, font=fs, fill=(180, 190, 210, 255))
    # hidden layer 256 -> 16x16 heat grid
    hgrid = hidden.reshape(16, 16); hg = (hgrid - hgrid.min()) / (hgrid.max() - hgrid.min() + 1e-9)
    for r in range(16):
        for c in range(16):
            v = float(hg[r, c]); col = (int(255 * v), int(120 * v), int(255 * (1 - v)), 255)
            d.rectangle((10 + c * 12, 30 + r * 12, 20 + c * 12, 40 + r * 12), fill=col)
    d.text((10, 215), "hidden 256 (MLP)", font=fs, fill=(180, 190, 210, 255))
    # stem features 256 -> strip
    sg = (stem - stem.min()) / (stem.max() - stem.min() + 1e-9)
    for i in range(256):
        v = float(sg[i]); d.rectangle((210 + (i % 64) * 3, 30 + (i // 64) * 8, 212 + (i % 64) * 3, 36 + (i // 64) * 8), fill=(int(255 * v), int(200 * v), 60, 255))
    d.text((210, 64), "scan features 256 (conv)", font=fs, fill=(180, 190, 210, 255))
    # action gauges
    def gauge(y, name, m, s_):
        d.text((210, y), f"{name} {m:+.2f} ± {s_:.2f}", font=fs, fill=(230, 230, 235, 255))
        d.rectangle((210, y + 18, 400, y + 26), fill=(40, 45, 60, 255))
        cx = 305 + m * 95; d.rectangle((cx - max(2, s_ * 95), y + 18, cx + max(2, s_ * 95), y + 26), fill=(80, 140, 255, 180))
        d.rectangle((cx - 2, y + 16, cx + 2, y + 28), fill=(255, 255, 255, 255))
    gauge(90, "steer", mu[0], std[0]); gauge(130, "speed", mu[1], std[1])
    d.text((210, 172), f"value  {value:7.2f}", font=fs, fill=(230, 230, 235, 255))
    d.text((210, 190), f"speed cap {speed_cap:.1f} m/s", font=fs, fill=(230, 230, 235, 255))
    d.text((210, 226), "scan points: red = high saliency", font=fs, fill=(255, 150, 120, 255))
    d.text((10, 256), "input: 3 scans, VESC speed, IMU, roll/pitch est., last 2 actions", font=fs, fill=(150, 160, 180, 255))
    d.text((10, 276), "no map, no localization", font=fs, fill=(150, 160, 180, 255))
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
        v.panel = panel_image(intro.h["hidden"][0].float().cpu().numpy(), intro.h["stem"][0].float().cpu().numpy(), mu, std, float("nan"), speed_cap, f"replay t={rec.t[t]:.1f}s")
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
    a = ap.parse_args()
    device = torch.device(a.device)
    ckpt_path = a.run if a.run.endswith(".pt") else next(p for p in (os.path.join(a.run, "ppo_latest.pt"), os.path.join(a.run, "student_latest.pt")) if os.path.exists(p))
    tracks, _ = common.load_tracks([a.map])
    env = common.make_env(tracks, a.cars, device, EnvConfig(speed_cap=a.speed_cap))
    model, extra = load_checkpoint(ckpt_path, device); model.eval(); intro = Introspector(model)
    mtime = os.path.getmtime(ckpt_path); step_info = f"ckpt {extra.get('steps', extra.get('iter', 0)) / 1e6:.1f}M steps"
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
                    intro.__init__(m2); mtime = mt; step_info = f"ckpt {ex.get('steps', 0) / 1e6:.1f}M steps (reloaded)"
            except Exception:
                pass
        m = state["model"]
        scan, pro = flatten_obs(state["obs"])
        with torch.no_grad():
            act, _ = m.act(scan, pro, deterministic=not a.stochastic)
            priv = env.privileged(env.last_result)
            val = m.critic(scan[v.focus:v.focus + 1], pro[v.focus:v.focus + 1], priv[v.focus:v.focus + 1]).item()
        sal, mu = intro.saliency(scan, pro, v.focus)
        v.point_colors = sal_colors(sal)
        std = m.actor.log_std.exp().detach().cpu().numpy()
        v.panel = panel_image(intro.h["hidden"][v.focus].float().cpu().numpy(), intro.h["stem"][v.focus].float().cpu().numpy(), mu, std, val, a.speed_cap, step_info)
        v.extra_hud = [f"WATCH {os.path.basename(ckpt_path)}  {step_info}   agents {a.cars}"]
        state["obs"], rew, term, trunc, info = env.step(act)
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
