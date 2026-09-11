"""Native OpenGL viewer (moderngl): runs in the simulator process, instanced car rendering with
shadow mapping, duct-hose track, LiDAR hit points colored by what they hit, HUD.

    viewer = NativeViewer(sim, raceline=rl)          # opens a window (or headless=True for EGL offscreen)
    while viewer.alive:
        r = sim.step(a); viewer.update(r); viewer.render()
    viewer.run(step_fn)                              # or: paced real-time loop with interpolation

Keys: C camera (chase/top/orbit/overview/closeup)  [ ] focus car  M N track  L lidar  T trails  S screenshot
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
                 max_fps: float = 120.0, track_index: Optional[int] = None, sim_yield: float = 0.0005):
        """Multi-track sims: shows track `track_index` (default: the focus car's track) and only the cars on it."""
        self.sim = sim
        self.focus = 0
        self.track_index = int(sim.tid[self.focus]) if track_index is None else int(track_index)
        self.track_groups: Optional[dict] = None   # {set name: [track index, ...]}; M / N walks one set
        self.track_group_name: str = ""
        self.max_fps = max_fps
        self.vsync = vsync
        self.sim_yield = sim_yield        # [s] yielded per step *only when the sim is behind real time*.
                                          # An unconditional sleep here was 2 ms on every step, which
                                          # slows a sim that is already struggling -- exactly when it
                                          # can least afford it. While it is ahead the lead logic
                                          # below already hands the GL thread all the slack it has.
        self._last_render = 0.0
        self.headless = headless
        self.max_cars = max_cars
        self.width, self.height = width, height
        self.alive = True
        self.paused = False
        self.extra_hud = []                    # lines appended to the HUD (teleop bars etc.)
        self.point_colors = None               # (N,4) override for the focus car's scan points (e.g. saliency)
        self.plan = None                       # (K,3) world x, y, speed: the focus car's current plan (plan action space)
        self.color_v_max = None                # top of the plan's speed colour ramp; None = the policy's v_max
        self.plan_pred = None                  # (K,3) the tracker's predicted motion along it
        self.panel = None                      # PIL RGBA image drawn top-right (policy panel)
        self.dash = None                       # PIL RGBA image drawn bottom-centre (speedometer + steering wheel)
        self.bev = None                        # PIL RGBA image drawn top-right: the policy's scan from above
        self.range_max = 10.0                  # [m] scale the BEV panel denormalises the scan with
        self.panel_fn = None; self.dash_fn = None; self.bev_fn = None   # callables building those images, run in the render thread
        self.on_frame = None                   # called ~1 Hz from the render loop: the viewer's
                                               # "I am on screen" heartbeat, which a training job
                                               # reads to hand back a slice of the GPU
        self.mode = 0
        self.focus = 0
        self.show_lidar = self.show_race = self.show_trails = True
        self.frame: Optional[dict] = None           # latest sim state snapshot (numpy)
        self.prev_frame: Optional[dict] = None
        self._t_frame = 0.0
        import threading
        self._lock = threading.RLock()
        self._worker = None; self._worker_err = None
        self._snap = None; self._sel = None; self.env_ids = None
        from collections import deque
        self._frames = deque(maxlen=32); self._play_t = None; self._play_wall = None; self._play_rate = 1.0
        self._t_wall0 = None; self._t_sim0 = None
        self._orbit = [math.radians(-35), math.radians(30), 4.0]   # azimuth, elevation, distance
        self._drag = None
        self._fps = 0.0; self._fps_n = 0; self._fps_t = time.perf_counter()
        self._hud_t = 0.0
        self._duct_d_override = duct_diameter
        self.duct_d = duct_diameter if duct_diameter is not None else sim.tracks[self.track_index].duct_height
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

        # ---- static geometry (rebuilt whenever the shown track changes)
        self._racelines = raceline
        self.pending_track = None              # set on a switch; the sim loop moves the cars over
        self.build_track_geometry()
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
        """Hand the latest StepResult to the render thread: no GPU work at all on the sim thread (the
        result's tensors are fresh objects every step, so reading them later is safe)."""
        with self._lock:
            self._snap = (r, r.t)

    def _pull(self):
        """Render thread: gather the newest snapshot on the GPU (its own CUDA stream, so the sim's stream
        is not held up), one device->host copy, numpy frame dict."""
        with self._lock:
            snap = self._snap; self._snap = None
        if snap is None:
            return
        if self.sim.device.type == "cuda":
            if getattr(self, "_rstream", None) is None:
                self._rstream = torch.cuda.Stream()
            with torch.cuda.stream(self._rstream):
                self._rstream.wait_stream(torch.cuda.current_stream())
                return self._pull_gather(snap)
        return self._pull_gather(snap)

    def _pull_gather(self, snap):
        r, t = snap
        sel = torch.nonzero(self.sim.tid == self.track_index).flatten()[: self.max_cars] if self._sel is None else self._sel
        if self._sel is None and self.sim.track.T == 1:
            self._sel = sel
        if sel.numel() == 0:
            return
        fe = int(self.focus) if 0 <= int(self.focus) < r.scan.shape[0] else 0
        Pk = ("mount_x", "mount_y", "mount_z", "mount_yaw", "mount_roll", "mount_pitch")
        pack_t = torch.cat([r.state[sel], r.attitude[sel], r.lap[sel, None].float(), r.collision[sel, None].float(),
                            r.s[sel, None], r.wall_dist[sel, None], self.sim.car_rear[sel], self.sim.car_dims[sel, 0:1]], 1)
        nb = r.scan.shape[1]
        flat = torch.cat([pack_t.reshape(-1), r.scan[fe], r.scan_type[fe].float(), torch.stack([self.sim.P[k][fe] for k in Pk])]).cpu().numpy()
        pshape = pack_t.shape; n = pshape[0]
        pack = flat[:n * pshape[1]].reshape(pshape)
        scan = flat[n * pshape[1]:n * pshape[1] + 2 * nb].reshape(2, nb); pv = flat[n * pshape[1] + 2 * nb:]
        env_ids = sel.cpu().numpy() if self.env_ids is None or len(self.env_ids) != n else self.env_ids
        f = int((env_ids == fe).nonzero()[0][0]) if fe in env_ids else 0
        fr = {"t": t, "n": n, "x": pack[:, 0], "y": pack[:, 1], "yaw": pack[:, 2], "vx": pack[:, 3], "steer": pack[:, 6],
              "roll": pack[:, 7], "pitch": pack[:, 8], "lap": pack[:, 9], "coll": pack[:, 10], "s": pack[:, 11], "wall": pack[:, 12],
              "rear": pack[:, 13:17], "len": pack[:, 17], "scan": scan[0], "scan_type": scan[1].astype(np.int32),
              "focus": f, "focus_env": fe, "ids": env_ids, "P": {k: float(pv[i]) for i, k in enumerate(Pk)}}
        # cars sharing the watched car's race, as opposed to unrelated parallel runs: only these are
        # something it can actually hit
        if self.sim.other_idx is not None:
            rivals = set(self.sim.other_idx[fe].tolist())
            fr["opponent"] = np.array([int(e) in rivals for e in env_ids], bool)
        now = time.perf_counter()
        with self._lock:
            self.env_ids = env_ids
            self._dt_frame = max(1e-3, now - self._t_frame) if self.frame is not None else self.sim.control_dt
            self.prev_frame, self.frame = self.frame, fr
            self._t_frame = now
            self._trail_pending = True
            self._frames.append(fr)                                # playback queue (sim time in fr["t"])

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

    def keep_alive(self, text: str = ""):
        """Poll events and paint a frame while the caller is busy compiling.

        The window is created before the CUDA graphs are captured, and capturing them takes ten to
        twenty seconds during which nothing polls the event queue -- so the window manager decides
        the application has hung and paints its "not responding" banner over it. Calling this
        between the warm-up steps keeps the queue drained and shows what is happening instead.
        """
        if self.headless or self.window is None:
            return
        import glfw
        glfw.poll_events()
        if glfw.window_should_close(self.window):
            self.alive = False
            return
        sc = self.scene
        sc.target = sc.ctx.screen                 # clear() paints the offscreen target otherwise
        sc.clear()
        if text:
            sc.set_hud([text])
            sc.draw_hud()
        glfw.swap_buffers(self.window)

    def run(self, step_fn, realtime: bool = True, max_catchup: int = 4, threaded: bool = True):
        """Drive the sim from the viewer: step_fn() -> StepResult. threaded (default): the sim runs in a
        worker thread paced to wall-clock real time (as fast as it can otherwise) while this thread
        renders every vsync with interpolation, so the picture stays smooth even when a training job
        hogs the GPU and a sim step takes 100 ms. Call sim.warmup() before creating the viewer,
        otherwise the first step's JIT compile (10-20 s) stalls the window."""
        self.render()                                   # show something before the first step
        if threaded:
            import sys, threading
            # Pacing has to come from exactly one place. vsync already blocks swap_buffers for a whole
            # refresh, and there used to be two more sleeps on top of it -- render()'s own max_fps cap
            # and the loop's -- with max_fps clamped to 30. On a 60 Hz screen that asks for a frame
            # every 33.3 ms from a device that can only deliver on 16.7 ms boundaries, so frames land
            # on irregular multiples of the refresh and the picture judders. With vsync on, let the
            # swap alone set the rate.
            sys.setswitchinterval(0.001)                # long enough to finish a frame uninterrupted,
                                                        # short enough that the sim is not left waiting
            if self.vsync:                              # a cap well above any refresh never fights the
                self.max_fps = max(self.max_fps or 0.0, 240.0)   # swap, but still stops a spin if the
                                                        # compositor turned vsync off behind our back
            self._sim_rate = 0.0

            # torch's CUDA-graph trees keep their state in thread-local storage that only autograd threads
            # inherit: hand the main thread's state to the sim thread, or every graphed call fails there
            try:
                import torch._inductor.cudagraph_trees as _ct
                _tls = (_ct.local.tree_manager_containers, _ct.local.tree_manager_locks)
            except Exception:
                _tls = None

            def worker():
                if _tls is not None:
                    try:
                        import torch
                        torch._C._stash_obj_in_tls("tree_manager_containers", _tls[0]); torch._C._stash_obj_in_tls("tree_manager_locks", _tls[1])
                    except Exception:
                        pass
                t0 = time.perf_counter(); sim_t0 = self.sim.t
                try:
                    n_ = 0; t_rate = time.perf_counter(); sim_rate0 = self.sim.t
                    while self.alive:
                        if self.paused:
                            time.sleep(0.02); t0 = time.perf_counter(); sim_t0 = self.sim.t; continue
                        self.update(step_fn()); n_ += 1
                        if n_ % 10 == 0:                     # sim speed relative to real time, for the HUD
                            now_ = time.perf_counter(); self._sim_rate = (self.sim.t - sim_rate0) / max(1e-3, now_ - t_rate); t_rate, sim_rate0 = now_, self.sim.t
                        if realtime:
                            lead = (self.sim.t - sim_t0) - (time.perf_counter() - t0)
                            self._lead = lead
                            if lead > 0: time.sleep(min(lead, 0.05))     # ahead: give the GL thread the slack
                            elif self.sim_yield: time.sleep(self.sim_yield)  # behind: a token yield only
                            elif lead < -0.5: t0 = time.perf_counter(); sim_t0 = self.sim.t   # can't keep up: re-anchor
                except Exception as e:                  # surface in the main thread
                    self._worker_err = e; self.alive = False
            self._worker = threading.Thread(target=worker, daemon=True); self._worker.start()
            self._lead = 0.0
            hb = 0.0
            while self.alive:
                # a steady frame rate comes first: the picture plays back at the sim's measured pace
                # (smoothly, one or two sim frames behind) even when the sim cannot keep real time
                now_ = time.perf_counter()
                if self.on_frame is not None and now_ - hb > 1.0:
                    hb = now_; self.on_frame()
                self.render()                           # render() owns the frame rate: vsync blocks in
                                                        # swap_buffers, and its max_fps guard only bites
                                                        # if the compositor ignored vsync. Capping here
                                                        # as well was the second of three pacers.
            self._worker.join(timeout=10.0)                          # let the sim step in flight finish before exit
            if self._worker_err is not None:
                raise self._worker_err
            return
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
    def speed_scale(self) -> float:
        """Top of the plan's colour ramp. watch.py sets color_v_max to the cap in force, so red means
        "at the cap" and the hues re-spread themselves as the curriculum raises it."""
        return float(getattr(self, "color_v_max", None) or 10.0)

    def _speed_colors(self, v):
        from .gl_scene import speed_colors
        return speed_colors(v, self.speed_scale())

    def _interp(self):
        """Frame to draw: a playback clock in sim time runs at the sim's measured pace and stays a
        little behind the newest frame, so irregular sim frames (a busy GPU) still play smoothly."""
        frames = list(self._frames)
        if not frames:
            return self.frame
        if len(frames) < 3 or self._worker is None:
            return frames[-1]
        now = time.perf_counter()
        t_new = frames[-1]["t"]; t_old = frames[0]["t"]
        span = max(1e-3, (t_new - t_old) / max(1, len(frames) - 1))       # sim seconds per pulled frame
        if self._play_t is None:
            self._play_t, self._play_wall = t_new - 2 * span, now
            return frames[-1]
        # advance at the measured sim rate, nudged to keep ~2 frames of buffer
        lag = t_new - self._play_t
        rate = max(0.05, self._sim_rate or 1.0) * float(np.clip(1.0 + 0.5 * (lag - 2 * span) / (2 * span), 0.5, 1.5))
        self._play_t = min(self._play_t + (now - self._play_wall) * rate, t_new)
        self._play_wall = now
        # bracketing frames
        j = len(frames) - 1
        while j > 0 and frames[j - 1]["t"] > self._play_t:
            j -= 1
        if j == 0:
            return frames[0]
        f0, f1 = frames[j - 1], frames[j]
        if f0["n"] != f1["n"]:
            return f1
        a = float(np.clip((self._play_t - f0["t"]) / max(1e-6, f1["t"] - f0["t"]), 0.0, 1.0))
        out = dict(f1)
        for k in ("x", "y", "vx", "steer", "roll", "pitch"):
            out[k] = f0[k] + (f1[k] - f0[k]) * a
        dyaw = (f1["yaw"] - f0["yaw"] + math.pi) % (2 * math.pi) - math.pi
        out["yaw"] = f0["yaw"] + dyaw * a
        out["t"] = self._play_t
        return out

    def render(self):
        if getattr(self, "_last_mode", None) != self.mode:
            self._last_mode = self.mode; self._hud_t = 0.0        # refresh the HUD on camera change
        if not self.headless:
            import glfw
            glfw.poll_events()
            if glfw.window_should_close(self.window):
                self.alive = False; return
        self._pull()
        with self._lock:
            fr = self._interp()
            if getattr(self, "_trail_pending", False) and self.frame is not None:
                self.scene.push_trail_points(self.frame["x"], self.frame["y"]); self._trail_pending = False
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
        # The watched car has to be findable at a glance. This used to multiply its tint by 1.0 --
        # a no-op -- so it was drawn exactly like every other car and the only way to tell which one
        # the panels were describing was to guess. Everything else is dimmed instead, and an
        # opponent sharing its track is marked apart from an unrelated parallel run.
        tint = np.full((n, 4), 0.45, np.float32); tint[:, 3] = 1.0
        opp = fr.get("opponent")
        if opp is not None and len(opp) == n:
            tint[np.asarray(opp, bool)] = (0.95, 0.62, 0.25, 1.0)      # racing this car: amber
        tint[fr["coll"] > 0.5] = (1.0, 0.35, 0.3, 1.0)                 # crashed: red
        tint[f] = (0.35, 0.95, 1.0, 1.0)                               # the watched car: cyan
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
            cols = self._speed_colors(v)
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
            rate = getattr(self, "_sim_rate", 0.0)
            lines = [f"{self.sim.tracks[self.track_index].name}   {n} cars   t = {fr['t']:.1f} s   {self._fps:.0f} fps" + (f"   sim {rate:.2f}x real time" if rate else "") + self._track_hud(),
                     f"watching car {fr.get('focus_env', f)}      [ ] other car   C camera ({MODES[self.mode]})   L lidar"
                     + ("   M N track" if self.sim.track.T > 1 else "") + ("   PAUSED" if self.paused else ""),
                     f"  speed {fr['vx'][f]:4.2f} m/s   steering {math.degrees(fr['steer'][f]):+5.1f} deg   body roll {math.degrees(fr['roll'][f]):+4.1f} deg  pitch {math.degrees(fr['pitch'][f]):+4.1f} deg",
                     f"  lap {int(fr['lap'][f])}, {fr['s'][f]:.1f} m into it      nearest wall {fr['wall'][f]:.2f} m",
                     f"crashed: {coll} of {n} cars"]
            sc.set_hud(lines + list(self.extra_hud))
            if self.panel_fn is not None:
                self.panel = self.panel_fn()
            if self.panel is not None:
                sc.set_panel(self.panel, slot=2)
        if self.dash_fn is not None:
            self.dash = self.dash_fn()
        if self.dash is not None:
            sc.set_panel(self.dash, slot=1)
        if self.bev_fn is not None:
            self.bev = self.bev_fn()
        if self.bev is not None:
            sc.set_panel(self.bev, slot=0)
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
            # Lay the map's long axis along the screen's long axis, and frame it for the real aspect
            # ratio. Fixed north-up on a 22.6 x 8.4 m track left most of a widescreen window empty
            # and the car four pixels across; the distance was also picked from the larger extent
            # alone, which over-frames whenever the two do not match.
            b = self.bounds; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            w, h = b[2] - b[0], b[3] - b[1]
            aspect = self.width / max(1, self.height)
            long_map, short_map = max(w, h), max(1e-6, min(w, h))
            turn = long_map / short_map > 1.1 and (h > w) != (aspect < 1.0)   # long meets long, but
                                                                              # never spin a square map
            up = np.array([1.0, 0, 0]) if turn else np.array([0, 1.0, 0])
            vert, horiz = (w, h) if turn else (h, w)       # extents as they land on screen
            t = math.tan(math.radians(55.0) / 2)
            d = max(0.5 * vert / t, 0.5 * horiz / (t * aspect)) * 1.08
            eye = np.array([cx, cy, d]); target = np.array([cx, cy, 0.0])
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
    def build_track_geometry(self):
        """(Re)build everything that depends on which track is shown. Safe to call again: the scene
        builders overwrite their meshes and `add_line` is keyed by name."""
        tr = self.sim.tracks[self.track_index]
        self.scene.clear_static()                              # the builders append: without this the
        self.scene.clear_line("centerline")                    # previous track stays on screen
        self.scene.clear_line("raceline")
        self.duct_d = tr.duct_height if self._duct_d_override is None else self._duct_d_override
        H, W = tr.shape
        self.bounds = (tr.origin[0], tr.origin[1], tr.origin[0] + W * tr.resolution, tr.origin[1] + H * tr.resolution)
        self.scene.build_floor(self.bounds)
        def _occ(xy, mask=tr.duct, tr=tr):
            j = np.clip(((xy[:, 0] - tr.origin[0]) / tr.resolution).astype(int), 0, mask.shape[1] - 1)
            i = np.clip(((xy[:, 1] - tr.origin[1]) / tr.resolution).astype(int), 0, mask.shape[0] - 1)
            return mask[i, j]
        self.scene.build_ducts(track_contours_mask(tr, tr.duct), self.duct_d, occupied=_occ)

        def is_solid(xy):
            col = int(round((xy[0] - tr.origin[0]) / tr.resolution)); row = int(round((xy[1] - tr.origin[1]) / tr.resolution))
            return 0 <= row < H and 0 <= col < W and bool(tr.tall[row, col])

        self.scene.build_walls(track_contours_mask(tr, tr.tall), height=1.0, is_solid=is_solid)
        if tr.centerline is not None:
            self.scene.add_line("centerline", np.vstack([tr.centerline, tr.centerline[:1]]), (0.35, 0.55, 0.9, 0.35), z=0.008)
        rl = self._racelines
        if isinstance(rl, (list, tuple)):
            rl = rl[self.track_index] if self.track_index < len(rl) else None
        self.raceline = rl
        if rl is not None:
            v = rl.v; t = (v - v.min()) / (v.max() - v.min() + 1e-6)
            cols = np.stack([t, 1 - np.abs(2 * t - 1), 1 - t, np.ones_like(t)], 1)
            self.scene.add_line("raceline", np.vstack([rl.xy, rl.xy[:1]]), np.vstack([cols, cols[:1]]), z=0.012)

    def _track_hud(self) -> str:
        """Which map, out of which set -- the set matters as much as the number: 3 of 24 held-out
        maps and 3 of 24 training maps are answers to different questions."""
        idx = self.group_indices()
        if self.sim.track.T <= 1:
            return ""
        here = idx.index(self.track_index) + 1 if self.track_index in idx else 0
        where = f"   (track {here} of {len(idx)}"
        return where + (f" in \"{self.track_group_name}\", G switches set)" if self.track_group_name else ")")

    def set_group(self, name: str) -> bool:
        """Restrict M / N to one named set of the loaded tracks, and show its first map.

        The point of the sets is that a held-out map answers a different question from a training
        map. M / N walking every loaded track regardless of which set was picked makes it easy to
        read the answer off the wrong one, which is the whole failure the picker exists to prevent.
        """
        idx = (self.track_groups or {}).get(name)
        if not idx:
            return False
        self.track_group_name = name
        if self.track_index not in idx:
            self.set_track(idx[0])
        else:
            self._hud_t = 0.0
        return True

    def group_indices(self):
        """Track indices M / N walks: the active set, or everything loaded when no set is named."""
        idx = (self.track_groups or {}).get(self.track_group_name)
        return idx if idx else list(range(self.sim.track.T))

    def step_track(self, delta: int):
        """Next / previous track *within the active set*."""
        idx = self.group_indices()
        if len(idx) <= 1 and self.sim.track.T <= 1:
            return
        here = idx.index(self.track_index) if self.track_index in idx else 0
        self.set_track(idx[(here + delta) % len(idx)])

    def set_track(self, index: int):
        """Show another of the sim's tracks without restarting. Called from the key callback, which
        runs on the thread holding the GL context, so rebuilding the meshes here is safe."""
        n = self.sim.track.T
        if n <= 1:
            return
        self.track_index = int(index) % n
        self.build_track_geometry()
        self._sel = None                       # re-pick the cars to show on the new track
        self.focus = 0
        self.pending_track = self.track_index  # the sim loop moves every car onto it and resets
        self._hud_t = 0.0

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
        elif key == glfw.KEY_M: self.step_track(1)
        elif key == glfw.KEY_N: self.step_track(-1)
        elif key == glfw.KEY_G and self.track_groups:          # cycle the set M / N walks
            names = list(self.track_groups)
            here = names.index(self.track_group_name) if self.track_group_name in names else -1
            self.set_group(names[(here + 1) % len(names)])

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


def track_contours_mask(track, mask, outside_occupied: bool = True, sigma_cells: float = 0.0,
                        clip_canvas_edge: bool = False):
    """Contours of an arbitrary boolean mask on the track grid (reuses server.track_contours)."""
    class _T:  # duck-typed view with the mask as occupancy
        occupancy = mask; resolution = track.resolution; origin = track.origin
    return track_contours(_T, outside_occupied=outside_occupied, sigma_cells=sigma_cells,
                          clip_canvas_edge=clip_canvas_edge)
