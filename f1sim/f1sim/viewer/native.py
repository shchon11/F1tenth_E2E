"""Native OpenGL viewer (moderngl): runs in the simulator process, instanced car rendering with
shadow mapping, duct-hose track, LiDAR hit points colored by what they hit, HUD.

    viewer = NativeViewer(sim, raceline=rl)          # opens a window (or headless=True for EGL offscreen)
    while viewer.alive:
        r = sim.step(a); viewer.update(r); viewer.render()
    viewer.run(step_fn)                              # or: paced real-time loop with interpolation

Keys: C camera (chase/top/orbit/overview/closeup)  [ ] focus car  L lidar  T trails  S screenshot
      P pause  Esc quit.  Mouse: orbit (drag), zoom (wheel).  Driving keys (manual mode) are read
      by polling: W/S A/D or arrows, Shift boost, Space handbrake, R reset.
"""
from __future__ import annotations

import math
import os
import time
from typing import Optional

import numpy as np
import torch

from . import gl_scene as G
from .server import track_contours

MODES = ["chase", "top", "orbit", "overview", "closeup"]


def _fix_prime_env():
    """NVIDIA PRIME render-offload variables (__NV_PRIME_RENDER_OFFLOAD / __GLX_VENDOR_LIBRARY_NAME)
    are meant for laptops whose *integrated* GPU drives the display. On a machine where the NVIDIA
    GPU already is the primary X GPU (prime-select nvidia) they make every GLX window present
    black while rendering fine. Detect that and drop them before libGLX is loaded."""
    import shutil, subprocess
    keys = ("__NV_PRIME_RENDER_OFFLOAD", "__GLX_VENDOR_LIBRARY_NAME", "__VK_LAYER_NV_optimus")
    if not any(k in os.environ for k in keys) or os.environ.get("F1SIM_KEEP_PRIME_ENV"):
        return
    nvidia_primary = None
    if shutil.which("prime-select"):
        try:
            nvidia_primary = subprocess.run(["prime-select", "query"], capture_output=True, text=True, timeout=5).stdout.strip() == "nvidia"
        except Exception:
            pass
    if nvidia_primary is None and shutil.which("xrandr"):
        try:
            out = subprocess.run(["xrandr", "--listproviders"], capture_output=True, text=True, timeout=5).stdout
            first = next((l for l in out.splitlines() if l.startswith("Provider 0")), "")
            nvidia_primary = "NVIDIA" in first and "Source Output" in first
        except Exception:
            pass
    if nvidia_primary:
        for k in keys:
            os.environ.pop(k, None)
        print("[f1sim viewer] NVIDIA is the primary GPU: dropped PRIME render-offload env vars "
              "(they blank GLX windows here; set F1SIM_KEEP_PRIME_ENV=1 to keep them)")


class NativeViewer:
    def __init__(self, sim, raceline=None, width: int = 1600, height: int = 900, title: str = "f1sim",
                 headless: bool = False, max_cars: int = 64, duct_diameter: Optional[float] = None,
                 vsync: bool = True, msaa: int = 4, shadows: bool = True, lidar_height: Optional[float] = None,
                 max_fps: float = 120.0, track_index: Optional[int] = None):
        """Multi-track sims: shows track `track_index` (default: the focus car's track) and only the cars on it."""
        self.sim = sim
        self.focus = 0
        self.track_index = int(sim.tid[self.focus]) if track_index is None else int(track_index)
        self.max_fps = max_fps
        self._last_render = 0.0
        self.headless = headless
        self.max_cars = max_cars
        self.width, self.height = width, height
        self.alive = True
        self.paused = False
        self.extra_hud = []                    # lines appended to the HUD (teleop bars etc.)
        self.point_colors = None               # (N,4) override for the focus car's scan points (e.g. saliency)
        self.plan = None                       # (K,3) world x, y, speed: the focus car's current plan (plan action space)
        self.plan_pred = None                  # (K,3) the tracker's predicted motion along it
        self.panel = None                      # PIL RGBA image drawn top-right (activations etc.)
        self.mode = 0
        self.focus = 0
        self.show_lidar = self.show_race = self.show_trails = True
        self.frame: Optional[dict] = None           # latest sim state snapshot (numpy)
        self.prev_frame: Optional[dict] = None
        self._t_frame = 0.0
        self._t_wall0 = None; self._t_sim0 = None
        self._orbit = [math.radians(-35), math.radians(30), 4.0]   # azimuth, elevation, distance
        self._drag = None
        self._fps = 0.0; self._fps_n = 0; self._fps_t = time.perf_counter()
        self._hud_t = 0.0
        tr = sim.tracks[self.track_index]
        self.duct_d = duct_diameter if duct_diameter is not None else tr.duct_height
        self.lidar_h = lidar_height if lidar_height is not None else sim.cfg.lidar.mount_z

        # ---- context / window
        if headless:
            import moderngl
            self.window = None
            self.ctx = moderngl.create_standalone_context(backend="egl")
        else:
            _fix_prime_env()
            import glfw
            import moderngl
            if not glfw.init():
                raise RuntimeError("glfw init failed")
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3); glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE); glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
            glfw.window_hint(glfw.SAMPLES, msaa)
            self.window = glfw.create_window(width, height, title, None, None)
            if not self.window:
                glfw.terminate(); raise RuntimeError("glfw window creation failed")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1 if vsync else 0)
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
            glfw.set_cursor_pos_callback(self.window, self._on_cursor)
            glfw.set_scroll_callback(self.window, self._on_scroll)
            glfw.set_framebuffer_size_callback(self.window, self._on_resize)
            self.ctx = moderngl.create_context()
            self.width, self.height = glfw.get_framebuffer_size(self.window)
        self.scene = G.Scene(self.ctx, self.width, self.height, headless=headless, msaa=msaa, shadows=shadows)

        # ---- static geometry
        contours_duct = track_contours_mask(tr, tr.duct)
        contours_tall = track_contours_mask(tr, tr.tall)
        H, W = tr.shape
        self.bounds = (tr.origin[0], tr.origin[1], tr.origin[0] + W * tr.resolution, tr.origin[1] + H * tr.resolution)
        self.scene.build_floor(self.bounds)
        self.scene.build_ducts(contours_duct, self.duct_d)
        def is_solid(xy):
            col = int(round((xy[0] - tr.origin[0]) / tr.resolution)); row = int(round((xy[1] - tr.origin[1]) / tr.resolution))
            return 0 <= row < H and 0 <= col < W and bool(tr.tall[row, col])
        self.scene.build_walls(contours_tall, height=1.0, is_solid=is_solid)
        if tr.centerline is not None:
            self.scene.add_line("centerline", np.vstack([tr.centerline, tr.centerline[:1]]), (0.35, 0.55, 0.9, 0.35), z=0.008)
        if isinstance(raceline, (list, tuple)):
            raceline = raceline[self.track_index]
        self.raceline = raceline
        if raceline is not None:
            v = raceline.v; t = (v - v.min()) / (v.max() - v.min() + 1e-6)
            cols = np.stack([t, 1 - np.abs(2 * t - 1), 1 - t, np.ones_like(t)], 1)
            self.scene.add_line("raceline", np.vstack([raceline.xy, raceline.xy[:1]]), np.vstack([cols, cols[:1]]), z=0.012)
        glb = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "f1tenth_car.glb")
        self.scene.load_car(glb, max_cars)
        self.n_beams = sim.cfg.lidar.n_beams
        self.scene.alloc_points(self.n_beams)
        self.scene.alloc_trails(min(max_cars, 64), 600)
        vp = sim.cfg.vehicle
        self.cog_x, self.cog_z = vp.lr, 0.06
        self.wheel_r = sim.cfg.vehicle.width * 0.0 + 0.056
        self._spin = np.zeros(max_cars)

    # ------------------------------------------------------------------ state
    def update(self, r):
        """Snapshot a StepResult (torch) into numpy for rendering."""
        sel = torch.nonzero(self.sim.tid == self.track_index).flatten()[: self.max_cars]
        n = sel.numel()
        if n == 0:
            return
        self.env_ids = sel.cpu().numpy()
        st = r.state[sel].detach().cpu().numpy()
        att = r.attitude[sel].detach().cpu().numpy()
        # focus is an env id; map to its position among the cars on this track
        f = int((self.env_ids == self.focus).nonzero()[0][0]) if self.focus in self.env_ids else 0
        fe = int(self.env_ids[f])
        fr = {"t": r.t, "n": n, "x": st[:, 0], "y": st[:, 1], "yaw": st[:, 2], "vx": st[:, 3], "steer": st[:, 6],
              "roll": att[:, 0], "pitch": att[:, 1], "lap": r.lap[sel].cpu().numpy(), "coll": r.collision[sel].cpu().numpy(),
              "s": r.s[sel].cpu().numpy(), "scan": r.scan[fe].cpu().numpy(), "scan_type": r.scan_type[fe].cpu().numpy(),
              "wall": r.wall_dist[sel].cpu().numpy(), "focus": f, "focus_env": fe, "ids": self.env_ids,
              "rear": self.sim.car_rear[sel].cpu().numpy(), "len": self.sim.car_dims[sel, 0].cpu().numpy(),
              "P": {k: float(self.sim.P[k][fe]) for k in ("mount_x", "mount_y", "mount_z", "mount_yaw", "mount_roll", "mount_pitch")}}
        self.prev_frame, self.frame = self.frame, fr
        self._t_frame = time.perf_counter()
        self.scene.push_trail_points(fr["x"], fr["y"])

    def sync(self):
        """Pace the calling loop to wall-clock real time (one call per sim step)."""
        t_sim = self.sim.t
        if self._t_wall0 is None or t_sim < self._t_sim0:
            self._t_wall0, self._t_sim0 = time.perf_counter(), t_sim; return
        target = self._t_wall0 + (t_sim - self._t_sim0); now = time.perf_counter()
        if target > now:
            time.sleep(target - now)
        elif now - target > 0.5:
            self._t_wall0, self._t_sim0 = now, t_sim

    def run(self, step_fn, realtime: bool = True, max_catchup: int = 4):
        """Drive the sim from the render loop: step_fn() -> StepResult. Renders every vsync with
        interpolation; steps the sim as often as wall-clock demands (or every frame if not realtime).
        Call sim.warmup() before creating the viewer, otherwise the first step's JIT compile
        (10-20 s) stalls the window and the desktop reports it as not responding."""
        self.render()                                   # show something before the first step
        t0 = time.perf_counter(); sim_t0 = self.sim.t
        while self.alive:
            if not self.paused:
                if realtime:
                    n = 0
                    while (self.sim.t - sim_t0) < (time.perf_counter() - t0) and n < max_catchup:
                        self.update(step_fn()); n += 1
                    if n == max_catchup:            # can't keep up: re-anchor instead of spiralling
                        t0 = time.perf_counter(); sim_t0 = self.sim.t
                else:
                    self.update(step_fn())
            else:
                t0 = time.perf_counter(); sim_t0 = self.sim.t
            self.render()

    # ------------------------------------------------------------------ rendering
    def _interp(self):
        """Interpolated car poses between the two latest frames (render-side smoothing)."""
        fr, pf = self.frame, self.prev_frame
        if fr is None:
            return None
        if pf is None or pf["n"] != fr["n"]:
            return fr
        dt_frames = max(1e-3, self.sim.control_dt)
        a = float(np.clip((time.perf_counter() - self._t_frame) / dt_frames, 0.0, 1.0))
        out = dict(fr)
        for k in ("x", "y", "vx", "steer", "roll", "pitch"):
            out[k] = pf[k] + (fr[k] - pf[k]) * a
        dyaw = (fr["yaw"] - pf["yaw"] + math.pi) % (2 * math.pi) - math.pi
        out["yaw"] = pf["yaw"] + dyaw * a
        return out

    def render(self):
        if getattr(self, "_last_mode", None) != self.mode:
            self._last_mode = self.mode; self._hud_t = 0.0        # refresh the HUD on camera change
        if not self.headless:
            import glfw
            glfw.poll_events()
            if glfw.window_should_close(self.window):
                self.alive = False; return
        fr = self._interp()
        if fr is not None:
            self._draw(fr)
        else:
            self.scene.clear()
        if not self.headless:
            import glfw
            glfw.swap_buffers(self.window)
            if self.max_fps:                       # drivers/compositors often ignore vsync: don't spin
                now = time.perf_counter(); wait = self._last_render + 1.0 / self.max_fps - now
                if wait > 0: time.sleep(wait)
                self._last_render = time.perf_counter()
        self._fps_n += 1
        now = time.perf_counter()
        if now - self._fps_t > 0.5:
            self._fps = self._fps_n / (now - self._fps_t); self._fps_n = 0; self._fps_t = now

    def _draw(self, fr):
        n, f = fr["n"], fr["focus"]
        sc = self.scene
        # camera
        eye, target, up = self._camera(fr)
        view = G.look_at(eye, target, up)
        proj = G.perspective(55.0, self.width / max(1, self.height), 0.05, 300.0)
        # car instances
        mats = np.zeros((n, 7, 4, 4), np.float32)      # per car: chassis, lidar, wheel_fl, fr, rl, rr, rear box
        for i in range(n):
            x, y, yaw = fr["x"][i], fr["y"][i], fr["yaw"][i]
            c, s = math.cos(yaw), math.sin(yaw)
            rear = np.array([x - self.cog_x * c, y - self.cog_x * s, 0.0])
            M = G.trans(rear) @ G.rot_z(yaw)
            tilt = G.trans([self.cog_x, 0, self.cog_z]) @ G.rot_y(fr["pitch"][i]) @ G.rot_x(fr["roll"][i]) @ G.trans([-self.cog_x, 0, -self.cog_z])
            mats[i, 0] = M @ tilt
            mats[i, 1] = M @ tilt @ sc.car_pivots["lidar"]
            self._spin[i] += fr["vx"][i] * self.sim.control_dt / self.wheel_r if self.prev_frame is not None else 0.0
            spin = G.rot_y(self._spin[i])
            for j, w in enumerate(("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")):
                steer = G.rot_z(fr["steer"][i]) if j < 2 else np.eye(4, dtype=np.float32)
                mats[i, 2 + j] = M @ sc.car_pivots[w] @ steer @ spin
            d, w_, zlo, zhi = fr["rear"][i]                               # detection box behind the body, on the chassis
            bx = self.cog_x - 0.5 * fr["len"][i] - 0.5 * d
            S = np.diag([d, w_, zhi - zlo, 1.0]).astype(np.float32)
            mats[i, 6] = M @ tilt @ G.trans([bx, 0.0, 0.5 * (zlo + zhi)]) @ S
        tint = np.ones((n, 4), np.float32)
        tint[fr["coll"] > 0.5] = (1.0, 0.35, 0.3, 1.0)
        tint[f] *= (1.0, 1.0, 1.0, 1.0)
        sc.set_car_instances(mats, tint, labels=fr.get("ids"))
        # lidar points of the focus car
        if self.show_lidar:
            pts, cols = self._scan_points(fr)
            sc.set_points(pts, cols)
        for slot, src in ((0, self.plan), (1, self.plan_pred)):
            if src is None or len(src) < 2:
                sc.set_plan(None, slot=slot); continue
            p_ = np.asarray(src, np.float32)
            v = p_[:, 2] if p_.shape[1] > 2 else np.zeros(len(p_), np.float32)
            t = np.clip(v / 8.0, 0.0, 1.0)[:, None]                        # speed profile: blue (slow) -> green -> yellow (fast)
            cols = (1 - t) * np.array([0.2, 0.55, 1.0, 0.95], np.float32) + t * np.array([1.0, 0.9, 0.15, 0.95], np.float32)
            cols = np.where(t < 0.5, (1 - 2 * t) * np.array([0.2, 0.55, 1.0, 0.95], np.float32) + 2 * t * np.array([0.2, 1.0, 0.45, 0.95], np.float32),
                            (2 - 2 * t) * np.array([0.2, 1.0, 0.45, 0.95], np.float32) + (2 * t - 1) * np.array([1.0, 0.9, 0.15, 0.95], np.float32))
            if slot == 1: cols[:, 3] = 0.6; cols[:, :3] = 0.5 * cols[:, :3] + 0.5
            sc.set_plan(np.concatenate([p_[:, :2], np.full((len(p_), 1), 0.05 if slot == 0 else 0.03, np.float32)], 1), cols, slot=slot)
        sc.draw_frame(view, proj, eye, light_center=np.array([fr["x"][f], fr["y"][f], 0.0]),
                      show_points=self.show_lidar, show_race=self.show_race, show_trails=self.show_trails, n_cars=n,
                      focus_xy=(fr["x"][f], fr["y"][f]))
        # hud (10 Hz)
        now = time.perf_counter()
        if now - self._hud_t > 0.1:
            self._hud_t = now
            coll = int((fr["coll"] > 0.5).sum())
            lines = [f"f1sim  {self.sim.tracks[self.track_index].name}   {n} cars" + (f"  (track {self.track_index}/{self.sim.track.T})" if self.sim.track.T > 1 else ""),
                     f"sim time  {fr['t']:7.2f} s      render {self._fps:4.0f} fps",
                     f"focus car {fr.get('focus_env', f):3d}   speed {fr['vx'][f]:5.2f} m/s   steer {math.degrees(fr['steer'][f]):5.1f} deg",
                     f"roll {math.degrees(fr['roll'][f]):5.2f} deg   pitch {math.degrees(fr['pitch'][f]):5.2f} deg",
                     f"lap {int(fr['lap'][f])}   s {fr['s'][f]:6.1f} m   wall {fr['wall'][f]:4.2f} m",
                     f"collided {coll}/{n}   camera {MODES[self.mode]}" + ("   PAUSED" if self.paused else "")]
            sc.set_hud(lines + list(self.extra_hud))
            if self.panel is not None:
                sc.set_panel(self.panel)
        sc.draw_hud()
        sc.draw_panel()

    def _scan_points(self, fr):
        """3D LiDAR hit points of the focus car (same beam geometry as f1sim.lidar.Lidar.rays)."""
        f = fr["focus"]; P = fr["P"]
        x, y, yaw = fr["x"][f], fr["y"][f], fr["yaw"][f]
        phi, th = fr["roll"][f] + P["mount_roll"], fr["pitch"][f] + P["mount_pitch"]
        lp = self.sim.cfg.lidar
        a = np.linspace(-lp.fov / 2, lp.fov / 2, self.n_beams) + P["mount_yaw"]
        ca, sa = np.cos(a), np.sin(a)
        cph, sph, cth, sth = math.cos(phi), math.sin(phi), math.cos(th), math.sin(th)
        bx = ca * cth + sa * sph * sth; by = sa * cph; bz = -ca * sth + sa * sph * cth
        cy, sy = math.cos(yaw), math.sin(yaw)
        wx = bx * cy - by * sy; wy = bx * sy + by * cy
        mx, my, mz = P["mount_x"], P["mount_y"], P["mount_z"]
        t = my * sph + mz * cph
        ox_b = mx * cth + t * sth; oy_b = my * cph - mz * sph; oz_b = -mx * sth + t * cth
        o = np.array([x + ox_b * cy - oy_b * sy, y + ox_b * sy + oy_b * cy, oz_b])
        r = fr["scan"].astype(np.float64); typ = fr["scan_type"]
        ok = np.isfinite(r)
        r = np.where(ok, r, 0.0)
        pts = o[None, :] + np.stack([wx, wy, bz], 1) * r[:, None]
        pts[~ok] = (0, 0, -100)
        cols = np.zeros((self.n_beams, 4), np.float32)
        cols[typ == 1] = (1.0, 0.55, 0.15, 1.0)     # duct
        cols[typ == 2] = (1.0, 0.2, 0.9, 1.0)       # tall object outside
        cols[typ == 3] = (0.2, 0.9, 1.0, 1.0)       # floor
        cols[typ == 4] = (1.0, 0.95, 0.2, 1.0)      # another car
        cols[typ == 0] = (0.5, 0.5, 0.5, 1.0)
        if self.point_colors is not None and len(self.point_colors) == self.n_beams:
            cols = np.asarray(self.point_colors, np.float32)
        return pts.astype(np.float32), cols

    def _camera(self, fr):
        f = fr["focus"]; x, y, yaw = fr["x"][f], fr["y"][f], fr["yaw"][f]
        m = MODES[self.mode]
        up = np.array([0, 0, 1.0])
        if m == "chase":
            eye = np.array([x - 1.7 * math.cos(yaw), y - 1.7 * math.sin(yaw), 0.75])
            target = np.array([x + 1.2 * math.cos(yaw), y + 1.2 * math.sin(yaw), 0.1])
        elif m == "top":
            eye = np.array([x, y, 14.0]); target = np.array([x, y, 0.0]); up = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        elif m == "overview":
            b = self.bounds; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            eye = np.array([cx, cy, max(b[2] - b[0], b[3] - b[1]) * 1.05]); target = np.array([cx, cy, 0.0]); up = np.array([0, 1.0, 0])
        elif m == "closeup":
            a = time.perf_counter() * 0.4
            eye = np.array([x + 1.1 * math.cos(a), y + 1.1 * math.sin(a), 0.45]); target = np.array([x, y, 0.1])
        else:  # orbit around the focus car
            az, el, d = self._orbit
            eye = np.array([x + d * math.cos(el) * math.cos(az), y + d * math.cos(el) * math.sin(az), 0.1 + d * math.sin(el)])
            target = np.array([x, y, 0.1])
        if not hasattr(self, "_cam_eye") or m in ("orbit", "closeup", "overview", "top"):
            self._cam_eye, self._cam_tgt = eye, target
        else:
            k = 1 - math.exp(-0.15 * 60 / max(self._fps, 30))
            self._cam_eye = self._cam_eye + (eye - self._cam_eye) * k
            self._cam_tgt = self._cam_tgt + (target - self._cam_tgt) * k
        return self._cam_eye, self._cam_tgt, up

    def keys_held(self) -> dict:
        """Polled driving keys (see f1sim.teleop.KEYS); empty when headless."""
        if self.headless or self.window is None:
            return {}
        import glfw
        from ..teleop import KEYS
        out = {}
        for name, keys in KEYS.items():
            out[name] = any(glfw.get_key(self.window, getattr(glfw, "KEY_" + k)) == glfw.PRESS for k in keys)
        return out

    # ------------------------------------------------------------------ io
    def screenshot(self, path: str):
        self.scene.screenshot(path)

    def close(self):
        self.alive = False
        if self.window is not None:
            import glfw
            glfw.destroy_window(self.window); glfw.terminate(); self.window = None

    # ------------------------------------------------------------------ input
    def _on_key(self, win, key, scancode, action, mods):
        import glfw
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE: self.alive = False
        elif key == glfw.KEY_C: self.mode = (self.mode + 1) % len(MODES); self._hud_t = 0.0
        elif key == glfw.KEY_RIGHT_BRACKET: self._cycle_focus(1)
        elif key == glfw.KEY_LEFT_BRACKET: self._cycle_focus(-1)
        elif key == glfw.KEY_L: self.show_lidar = not self.show_lidar
        elif key == glfw.KEY_V: self.show_race = not self.show_race
        elif key == glfw.KEY_T: self.show_trails = not self.show_trails
        elif key == glfw.KEY_P: self.paused = not self.paused
        elif key == glfw.KEY_S: self.screenshot(f"f1sim_{int(time.time())}.png")

    def _cycle_focus(self, step):
        ids = list(getattr(self, "env_ids", [])) or [0]
        i = ids.index(self.focus) if self.focus in ids else 0
        self.focus = int(ids[(i + step) % len(ids)])

    def _on_mouse_button(self, win, button, action, mods):
        import glfw
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._drag = glfw.get_cursor_pos(win) if action == glfw.PRESS else None
            if action == glfw.PRESS and MODES[self.mode] != "orbit":
                self.mode = MODES.index("orbit")

    def _on_cursor(self, win, xpos, ypos):
        if self._drag is None:
            return
        dx, dy = xpos - self._drag[0], ypos - self._drag[1]; self._drag = (xpos, ypos)
        self._orbit[0] -= dx * 0.005
        self._orbit[1] = float(np.clip(self._orbit[1] + dy * 0.005, math.radians(2), math.radians(89)))

    def _on_scroll(self, win, xoff, yoff):
        self._orbit[2] = float(np.clip(self._orbit[2] * (0.9 ** yoff), 0.5, 80.0))

    def _on_resize(self, win, w, h):
        self.width, self.height = max(1, w), max(1, h)
        self.scene.resize(self.width, self.height)


def track_contours_mask(track, mask):
    """Contours of an arbitrary boolean mask on the track grid (reuses server.track_contours)."""
    class _T:  # duck-typed view with the mask as occupancy
        occupancy = mask; resolution = track.resolution; origin = track.origin
    return track_contours(_T)
