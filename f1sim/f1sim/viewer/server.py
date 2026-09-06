"""Lightweight 3D viewer: the simulator streams car states + one LiDAR scan over a WebSocket
to a three.js page served from this package. Open http://localhost:<port>/ in a browser.

    viewer = Viewer(sim, raceline=rl)        # starts http + ws servers in a background thread
    ...
    r = sim.step(a); viewer.publish(r)       # non-blocking, drops frames when the browser lags
    viewer.sync()                            # optional: pace the loop to wall-clock real time
"""
from __future__ import annotations

import asyncio
import functools
import http.server
import json
import os
import struct
import threading
import time
from typing import Optional

import numpy as np
import torch

from ..sim import Simulator, StepResult

STATIC = os.path.join(os.path.dirname(__file__), "static")
ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


def track_contours(track, min_len: int = 8):
    """Closed wall contours (list of (K,2) arrays in meters) via marching squares on the occupancy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    occ = np.pad(track.occupancy.astype(float), 1, constant_values=1.0)
    fig = plt.figure(); ax = fig.add_subplot(111)
    cs = ax.contour(occ, levels=[0.5])
    segs = []
    paths = getattr(cs, "allsegs", None)
    if paths is not None and len(paths):
        segs = list(paths[0])
    else:                                   # matplotlib >= 3.8
        for p in cs.get_paths():
            for poly in p.to_polygons(closed_only=False):
                segs.append(np.asarray(poly))
    plt.close(fig)
    out = []
    for s in segs:
        if len(s) < min_len:
            continue
        xy = (np.asarray(s) - 1.0) * track.resolution + np.array(track.origin)   # col->x, row->y
        out.append(xy.astype(np.float32))
    return out


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, extra_dirs=None, **kw):
        self.extra_dirs = extra_dirs or {}
        super().__init__(*a, directory=STATIC, **kw)

    def translate_path(self, path):
        for prefix, d in self.extra_dirs.items():
            if path.startswith(prefix):
                return os.path.join(d, path[len(prefix):].lstrip("/"))
        return super().translate_path(path)

    def log_message(self, *a):
        pass


class Viewer:
    def __init__(self, sim: Simulator, raceline=None, port: int = 8765, ws_port: Optional[int] = None,
                 max_cars: int = 64, fps: float = 30.0, focus: int = 0, host: str = "0.0.0.0",
                 duct_diameter: float = 0.2, lidar_height: float = 0.15):
        """duct_diameter: the flexible duct hose used as track boundary [m]; lidar_height: scan plane [m]."""
        self.sim = sim
        self.duct_diameter, self.lidar_height = duct_diameter, lidar_height
        self.port, self.ws_port = port, ws_port or port + 1
        self.max_cars, self.fps, self.focus = max_cars, fps, focus
        self._latest: Optional[bytes] = None
        self._lock = threading.Lock()
        self._clients = set()
        self._t_wall0 = None; self._t_sim0 = None
        self.init_msg = self._build_init(raceline)
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_ws, daemon=True).start()
        self._httpd = http.server.ThreadingHTTPServer(
            (host, port), functools.partial(_Handler, extra_dirs={"/assets/": ASSETS}))
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        print(f"[f1sim viewer] open http://localhost:{port}/  (ws :{self.ws_port})")

    # ------------------------------------------------------------------ static scene
    def _build_init(self, raceline):
        sim, tr = self.sim, self.sim.tracks[0]
        lp, vp = sim.cfg.lidar, sim.cfg.vehicle
        msg = {
            "type": "init", "ws_port": self.ws_port,
            "track": {"name": tr.name, "contours": [c.tolist() for c in track_contours(tr)],
                      "origin": list(tr.origin), "size": [tr.shape[1] * tr.resolution, tr.shape[0] * tr.resolution],
                      "duct_diameter": self.duct_diameter},
            "centerline": tr.centerline.tolist() if tr.centerline is not None else None,
            "raceline": np.column_stack([raceline.xy, raceline.v]).tolist() if raceline is not None else None,
            "car": {"url": "/assets/f1tenth_car.glb", "wheelbase": vp.lf + vp.lr, "width": vp.width,
                    "length": vp.length, "cog_x": vp.lr},
            "lidar": {"mount_x": lp.mount_x, "mount_y": lp.mount_y, "fov": lp.fov, "n_beams": lp.n_beams,
                      "range_max": lp.range_max, "height": self.lidar_height},
            "n_envs": min(sim.B, self.max_cars), "control_dt": sim.control_dt,
        }
        return json.dumps(msg)

    # ------------------------------------------------------------------ publishing
    def publish(self, r: StepResult, focus: Optional[int] = None):
        if focus is not None:
            self.focus = focus
        n = min(self.sim.B, self.max_cars)
        f = min(self.focus, n - 1)
        st = r.state[:n]
        # per car: x, y, yaw(world, CoG), steer, vx, lap, collided, s
        cars = torch.stack([st[:, 0], st[:, 1], st[:, 2], st[:, 6], st[:, 3], r.lap[:n].float(),
                            r.collision[:n].float(), r.s[:n]], 1).cpu().numpy().astype(np.float32)
        scan = r.scan[f].cpu().numpy().astype(np.float32)
        scan = np.where(np.isfinite(scan), scan, -1.0).astype(np.float32)
        head = struct.pack("<4f", float(r.t), float(n), float(scan.shape[0]), float(f))
        buf = head + cars.tobytes() + scan.tobytes()
        with self._lock:
            self._latest = buf

    def sync(self):
        """Sleep so that sim time advances at wall-clock rate (call once per step)."""
        t_sim = self.sim.t
        if self._t_wall0 is None or t_sim < self._t_sim0:
            self._t_wall0, self._t_sim0 = time.perf_counter(), t_sim
            return
        target = self._t_wall0 + (t_sim - self._t_sim0)
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)
        elif now - target > 1.0:                  # fell far behind: re-anchor instead of racing
            self._t_wall0, self._t_sim0 = now, t_sim

    # ------------------------------------------------------------------ websocket
    def _run_ws(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        import websockets
        async def handler(ws):
            self._clients.add(ws)
            try:
                await ws.send(self.init_msg)
                async for msg in ws:            # client messages (focus change)
                    try:
                        m = json.loads(msg)
                        if "focus" in m:
                            self.focus = int(m["focus"])
                    except Exception:
                        pass
            except websockets.ConnectionClosed:
                pass
            finally:
                self._clients.discard(ws)
        async with websockets.serve(handler, "0.0.0.0", self.ws_port, max_size=None):
            last = None
            while True:
                await asyncio.sleep(1.0 / self.fps)
                with self._lock:
                    buf = self._latest
                if buf is None or buf is last or not self._clients:
                    continue
                last = buf
                websockets.broadcast(self._clients, buf)
