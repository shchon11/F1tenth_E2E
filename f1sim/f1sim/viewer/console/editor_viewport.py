"""The environment editor's 3D view: the session viewport's GL path with a free camera and picking.

What is shared with `ViewportWidget` and what is not
----------------------------------------------------
Everything about *owning* the GL context is inherited unchanged: the moderngl context, the
`Scene`, the borrowed Qt framebuffer (`_resolve_target` / `_ForeignFramebuffer`), the deferred
geometry upload (`set_geometry` -> `_apply_geometry` on the GL thread), the screenshot path and the
`upload_timed` / `draw_timed` / `gl_failed` signals. Those are the parts that took care to get
right and there is one correct way to do them.

What is replaced is everything that assumes a running session. The base class draws whatever
frame its `FrameBuffer` holds and places its camera relative to a car; an editor has no frames and
no cars. So `paintGL` here draws sky + static meshes + map lines (`Scene.draw_static`) with a
pivot camera the user moves directly, then a set of overlays (brush cursor, selection outlines,
tool preview, gizmo) on top, and turns mouse input into *world-space* events for the page.

Conventions
-----------
* The viewport never mutates document state. It reports where the user pressed, moved and
  released on the ground plane and draws what the page tells it to; the page owns the document.
* Mouse coordinates arrive in logical widget pixels; the GL side works in physical pixels. The
  two only meet in `ground_point` / `project`, which take and return logical pixels and use the
  physical size for the aspect ratio only, exactly as the base class does.
* Picking goes through the same matrices the paint uses (`_matrices`), so a point projected on
  screen and unprojected again lands where it started -- there is no second, approximate camera.
* Overlays are drawn with the depth test off. A brush ring behind a wall is still a brush ring;
  an editor that hides its own cursor behind the geometry being edited is unusable.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

from .. import gl_scene as G
from .theme import C
from .viewport import TrackGeometry, ViewportWidget, _gl_msaa_state

#: Vertical field of view of the perspective camera, degrees. The same as the session viewport, so
#: a map looks the same size in both pages at the same distance.
FOV_DEG = 55.0
#: Elevation of the top view. Not 90: `look_at` needs a view direction that is not parallel to any
#: up vector it might be handed, and 89.9 leaves the picture indistinguishable from straight down.
TOP_EL = math.radians(89.9)
#: Orbit elevation limits. The floor is a single-sided quad and the walls have no underside, so the
#: camera stays above the ground; the ceiling keeps the view direction away from the up vector.
EL_MIN, EL_MAX = math.radians(2.0), math.radians(89.0)
DIST_MIN, DIST_MAX = 0.3, 600.0
#: Ground-plane height at which overlay lines are placed. Depth test is off for them, so this only
#: matters for a screenshot's appearance, not for visibility.
OVERLAY_Z = 0.02

#: Layers `ViewportWidget._upload_static` tags meshes with, and what the visibility switches map
#: to. The backdrop disc is part of the ground as far as the user is concerned.
_LAYER_GROUPS = {"floor": ("floor", "backdrop"), "ducts": ("ducts",), "walls": ("walls",),
                 "props": ("props",)}


def _rgba(hex_colour: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)) + (float(alpha),)


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Even-odd test of (x, y) against a closed polygon (K, 2). Works for concave polygons."""
    P = np.asarray(poly, np.float64)
    if P.ndim != 2 or len(P) < 3:
        return False
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = (x1 - x0) * (y - y0) / (y1 - y0) + x0
    return bool(np.count_nonzero(crosses & (x < xi)) % 2 == 1)


class EditorViewport(ViewportWidget):
    """Draws a TrackGeometry with a free camera and turns mouse input into world-space events."""

    ground_pressed = QtCore.pyqtSignal(float, float, int, int, str)    # x, y, button, modifiers, hit id
    ground_moved = QtCore.pyqtSignal(float, float, int, int)           # x, y, buttons, modifiers
    ground_released = QtCore.pyqtSignal(float, float, int, int)        # x, y, button, modifiers
    hovered = QtCore.pyqtSignal(float, float, str)                     # x, y, hit id (no button down)
    double_clicked = QtCore.pyqtSignal(float, float)                   # x, y (left button)
    left_widget = QtCore.pyqtSignal()
    wheel_edit = QtCore.pyqtSignal(int, int)                           # angleDelta().y(), modifiers
    key_pressed = QtCore.pyqtSignal(int, int)                          # key, modifiers
    camera_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._empty.setText("환경을 새로 만들거나 목록에서 고르세요.")
        # The chrome labels sit on top of the GL surface. They must never take a mouse event the
        # page is waiting for: a hover over the empty-state text is still a hover.
        for lab in (self._empty, self._badge, self._corner):
            lab.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        # ---- camera: pivot on the ground, eye on a sphere around it
        self.pivot = np.zeros(2, np.float64)
        self.az = math.radians(-90.0)          # eye south of the pivot: +y runs up the screen
        self.el = math.radians(55.0)
        self.dist = 12.0
        self.top = False
        #: top view uses an orthographic projection. On for a map editor: painting a grid under
        #: perspective foreshortening is a guessing game, and the brush ring only *is* a circle
        #: of the stated radius on the ground when the projection has no vanishing point.
        self.ortho_top = True
        self._saved_view: Tuple[float, float] = (self.az, self.el)
        self._auto_frame = True

        # ---- input state
        self._drag_mode: Optional[str] = None      # "orbit" | "pan" | "edit"
        self._drag_pos: Optional[Tuple[float, float]] = None
        self._space_down = False
        self._last_ground: Optional[Tuple[float, float]] = None
        self._edit_button = 0

        # ---- picking
        self._pickables: List[Tuple[str, np.ndarray, Tuple[float, float, float, float]]] = []

        # ---- overlays (CPU side; rebuilt into one LINES buffer when dirty)
        self._ov: Dict[str, List[tuple]] = {"cursor": [], "selection": [], "hover": [],
                                            "preview": [], "gizmo": []}
        self._ov_dirty = True
        self._ov_vbo = None
        self._ov_vao = None
        self._ov_ranges: List[Tuple[float, int, int]] = []     # (line width, first, count)
        self._ov_capacity = 0

        # ---- layers
        self._hidden: set = set()
        #: static meshes drawn at the last paint -- what `set_layer_visibility` is measured by
        self.last_static_drawn = 0

    # ---------------------------------------------------------------- lifecycle
    def _load_car(self):
        """No cars in an editor. The car GLB is a few MB of buffers that would never be drawn."""
        self._car_loaded = False

    def _release_overlay_gl(self):
        for attr in ("_ov_vao", "_ov_vbo"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
            setattr(self, attr, None)
        self._ov_capacity = 0
        self._ov_dirty = True

    def _on_context_destroyed(self):
        self._release_overlay_gl()
        super()._on_context_destroyed()

    def teardown(self):
        if self.scene is not None:
            try:
                self.makeCurrent()
                self._release_overlay_gl()
            except Exception:
                pass
            finally:
                try:
                    self.doneCurrent()
                except Exception:
                    pass
        super().teardown()

    # ---------------------------------------------------------------- geometry
    def set_geometry(self, geom: Optional[TrackGeometry]):
        if geom is None:
            self._auto_frame = True
        super().set_geometry(geom)

    def _apply_geometry(self):
        had = self.geometry_data is not None
        pending = self._pending_geometry
        super()._apply_geometry()
        if pending is not None and self.geometry_data is pending and (self._auto_frame or not had):
            # The first map after a blank view is framed for the user; later uploads of the same
            # scene (every edit re-sends geometry) must leave the camera exactly where it is.
            self._auto_frame = False
            self.frame_all()

    def _framing_geometry(self) -> Optional[TrackGeometry]:
        return self._pending_geometry if self._pending_geometry is not None else self.geometry_data

    # ---------------------------------------------------------------- camera
    def camera_state(self) -> dict:
        """What the page shows in its corner chip."""
        return {"pivot": (float(self.pivot[0]), float(self.pivot[1])),
                "az_deg": math.degrees(self.az), "el_deg": math.degrees(self.el),
                "dist": float(self.dist), "top": bool(self.top),
                "ortho": bool(self.top and self.ortho_top)}

    def _camera_moved(self):
        self._ov_dirty = True            # the cross cursor is sized by distance
        self.camera_changed.emit()
        self.update()

    def _pose(self):
        """(eye, target, up) of the free camera."""
        el = TOP_EL if self.top else float(np.clip(self.el, EL_MIN, EL_MAX))
        ce, se = math.cos(el), math.sin(el)
        px, py = float(self.pivot[0]), float(self.pivot[1])
        eye = np.array([px + self.dist * ce * math.cos(self.az),
                        py + self.dist * ce * math.sin(self.az),
                        self.dist * se], np.float64)
        target = np.array([px, py, 0.0], np.float64)
        up = np.array([0.0, 1.0, 0.0]) if self.top else np.array([0.0, 0.0, 1.0])
        return eye, target, up

    def _matrices(self):
        """(view, proj, eye, target, (fog0, fog1)) for the current camera and widget size.

        Computed from the camera state every time rather than cached from the last paint, so a
        pick made between a wheel event and the repaint it scheduled already sees the new camera.
        It is a handful of 4x4 products; caching it would buy nothing and cost the guarantee.
        """
        eye, target, up = self._pose()
        w, h = self.fb_size()
        aspect = w / max(1, h)
        view = G.look_at(eye, target, up)
        near, far, fog0, fog1 = self._depth_range(eye, target)
        if self.top and self.ortho_top:
            hh = self.dist * math.tan(math.radians(FOV_DEG) / 2)
            proj = G.ortho(-hh * aspect, hh * aspect, -hh, hh, near, far)
        else:
            proj = G.perspective(FOV_DEG, aspect, near, far)
        return view, proj, eye, target, (fog0, fog1)

    def frame_all(self):
        """Look at the whole map: pivot on its centre, distance so every corner is on screen.

        Framed on what is drawn (`drawn_bounds`: content bounds on a v2 payload, the canvas on
        v1), the same rule as the session viewport's 전체보기. The distance is found by bisection
        on the actual projection of the four corners rather than a closed form, because the fit
        depends on elevation, on whether the top view is orthographic and on the aspect ratio,
        and one exact search is simpler than four approximate formulas.
        """
        geom = self._framing_geometry()
        if geom is None:
            return
        b = geom.drawn_bounds()
        if b is None:
            return
        x0, y0, x1, y1 = (float(v) for v in b)
        self.pivot = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], np.float64)
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        R = 0.5 * math.hypot(x1 - x0, y1 - y0)
        w, h = self.fb_size()
        half = math.atan(math.tan(math.radians(FOV_DEG) / 2) * min(1.0, w / max(1, h)))
        # a sphere of radius R about the pivot is inside the frustum at this distance: it fits
        hi = max(DIST_MIN, R / max(1e-6, math.sin(half)) * 1.05)
        lo = max(DIST_MIN, 0.05 * hi)

        def fits(d):
            self.dist = d
            for cx, cy in corners:
                p = self._ndc(cx, cy, 0.0)
                if p is None or abs(p[0]) > 0.92 or abs(p[1]) > 0.92:
                    return False
            return True

        if not fits(hi):
            self.dist = hi
        else:
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if fits(mid):
                    hi = mid
                else:
                    lo = mid
            self.dist = float(np.clip(hi, DIST_MIN, DIST_MAX))
        self._camera_moved()

    def set_top_view(self, on: bool):
        on = bool(on)
        if on == self.top:
            return
        if on:
            self._saved_view = (self.az, self.el)
        else:
            self.az, self.el = self._saved_view
        self.top = on
        self._camera_moved()

    def look_at_point(self, x: float, y: float):
        self.pivot = np.array([float(x), float(y)], np.float64)
        self._camera_moved()

    def _zoom(self, steps: float, at: Optional[Tuple[float, float]] = None):
        """Scale the distance by 0.9 per step, keeping the ground point under `at` fixed."""
        anchor = self.ground_point(*at) if at is not None else None
        new = float(np.clip(self.dist * (0.9 ** steps), DIST_MIN, DIST_MAX))
        k = new / self.dist
        if anchor is not None:
            g = np.asarray(anchor, np.float64)
            self.pivot = g + (self.pivot - g) * k
        self.dist = new
        self._camera_moved()

    # ---------------------------------------------------------------- picking
    def _ndc(self, x: float, y: float, z: float = 0.0) -> Optional[Tuple[float, float]]:
        view, proj, _eye, _t, _fog = self._matrices()
        clip = (np.asarray(proj, np.float64) @ np.asarray(view, np.float64)) @ np.array([x, y, z, 1.0])
        if clip[3] <= 1e-9:
            return None
        return float(clip[0] / clip[3]), float(clip[1] / clip[3])

    def project(self, x: float, y: float, z: float = 0.0) -> Optional[Tuple[float, float]]:
        """World -> logical widget pixels, or None when behind the camera."""
        p = self._ndc(x, y, z)
        if p is None:
            return None
        return (0.5 * (p[0] + 1.0) * self.width(), 0.5 * (1.0 - p[1]) * self.height())

    def ground_point(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """Logical widget pixel -> world (x, y) on z = 0, or None when the ray misses the plane."""
        w, h = max(1, self.width()), max(1, self.height())
        nx = 2.0 * float(px) / w - 1.0
        ny = 1.0 - 2.0 * float(py) / h
        view, proj, _eye, _t, _fog = self._matrices()
        vp = np.asarray(proj, np.float64) @ np.asarray(view, np.float64)
        try:
            inv = np.linalg.inv(vp)
        except np.linalg.LinAlgError:
            return None
        p0 = inv @ np.array([nx, ny, -1.0, 1.0])
        p1 = inv @ np.array([nx, ny, 1.0, 1.0])
        if abs(p0[3]) < 1e-12 or abs(p1[3]) < 1e-12:
            return None
        p0, p1 = p0[:3] / p0[3], p1[:3] / p1[3]
        d = p1 - p0
        if abs(d[2]) < 1e-12:
            return None
        t = -p0[2] / d[2]
        if t < 0.0:
            return None                          # the plane is behind the near plane: sky
        return float(p0[0] + t * d[0]), float(p0[1] + t * d[1])

    def set_pickables(self, items):
        """[(id, (K,2) world polygon)] for hover/press hit testing. Later items win overlaps."""
        out = []
        for pid, poly in items or ():
            P = np.asarray(poly, np.float64).reshape(-1, 2)
            if len(P) < 3:
                continue
            out.append((str(pid), P, (float(P[:, 0].min()), float(P[:, 1].min()),
                                      float(P[:, 0].max()), float(P[:, 1].max()))))
        self._pickables = out

    def hit_test(self, x: float, y: float) -> str:
        for pid, P, (bx0, by0, bx1, by1) in reversed(self._pickables):
            if bx0 <= x <= bx1 and by0 <= y <= by1 and point_in_polygon(x, y, P):
                return pid
        return ""

    # ---------------------------------------------------------------- overlays
    def _set_overlay(self, name: str, items: List[tuple]):
        self._ov[name] = items
        self._ov_dirty = True
        self.update()

    def set_cursor(self, kind: str, x: float, y: float, radius: float = 0.0, yaw: float = 0.0):
        """kind in {"none", "brush", "cross", "place"}. brush = ring of `radius`; place = ring +
        heading tick; cross = a small cross sized by the camera distance."""
        items: List[tuple] = []
        col = _rgba(C["text.0"], 0.9)
        x, y, radius = float(x), float(y), float(radius)
        if kind == "cross" or (kind == "brush" and radius <= 0.0):
            items.append(("cross", x, y, col, 1.5))
        elif kind in ("brush", "place"):
            a = np.linspace(0.0, 2 * math.pi, 64, endpoint=False)
            ring = np.stack([x + radius * np.cos(a), y + radius * np.sin(a)], 1)
            items.append((ring, col, 1.5, True))
            tick = min(0.12, 0.25 * radius)
            items.append((np.array([[x - tick, y], [x + tick, y]]), col, 1.0, False))
            items.append((np.array([[x, y - tick], [x, y + tick]]), col, 1.0, False))
            if kind == "place":
                c, s = math.cos(yaw), math.sin(yaw)
                items.append((np.array([[x + 0.6 * radius * c, y + 0.6 * radius * s],
                                        [x + 1.5 * radius * c, y + 1.5 * radius * s]]), col, 2.0, False))
        self._set_overlay("cursor", items)

    def set_selection(self, polys, colour: Optional[str] = None):
        """Closed outlines on the ground for the selected props. Colour is a theme hex string."""
        col = _rgba(colour or C["accent"], 0.95)
        items = [(np.asarray(p, np.float64).reshape(-1, 2), col, 2.0, True)
                 for p in (polys or ()) if len(p) >= 2]
        self._set_overlay("selection", items)

    def set_hover(self, poly):
        items = []
        if poly is not None and len(poly) >= 2:
            items.append((np.asarray(poly, np.float64).reshape(-1, 2), _rgba(C["text.1"], 0.9), 1.0, True))
        self._set_overlay("hover", items)

    def set_preview(self, pts, width: float, closed: bool = False):
        """In-progress polyline/polygon tool: centre line plus the two edges of the stroke band."""
        items: List[tuple] = []
        P = np.asarray(pts, np.float64).reshape(-1, 2) if pts is not None else np.zeros((0, 2))
        col = _rgba(C["warn"], 0.95)
        edge = _rgba(C["warn"], 0.55)
        half = 0.5 * float(width)
        if len(P) == 1:
            a = np.linspace(0.0, 2 * math.pi, 32, endpoint=False)
            r = max(half, 0.03)
            items.append((np.stack([P[0, 0] + r * np.cos(a), P[0, 1] + r * np.sin(a)], 1), edge, 1.0, True))
        elif len(P) >= 2:
            items.append((P, col, 1.5, bool(closed)))
            if half > 0.0:
                if closed and len(P) >= 3:
                    d = np.roll(P, -1, 0) - np.roll(P, 1, 0)
                else:
                    d = np.gradient(P, axis=0)
                d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
                n = np.stack([-d[:, 1], d[:, 0]], 1) * half
                items.append((P + n, edge, 1.0, bool(closed)))
                items.append((P - n, edge, 1.0, bool(closed)))
        self._set_overlay("preview", items)

    def set_gizmo(self, x: float, y: float, yaw: float, radius: float = 0.4, on: bool = True):
        """Move handle (cross) and rotation ring with a heading tick, at the selection."""
        items: List[tuple] = []
        if on:
            x, y, radius = float(x), float(y), float(radius)
            ring_col = _rgba(C["ego"], 0.95)
            arm_col = _rgba(C["text.0"], 0.9)
            a = np.linspace(0.0, 2 * math.pi, 72, endpoint=False)
            items.append((np.stack([x + radius * np.cos(a), y + radius * np.sin(a)], 1), ring_col, 2.0, True))
            c, s = math.cos(yaw), math.sin(yaw)
            items.append((np.array([[x + 0.85 * radius * c, y + 0.85 * radius * s],
                                    [x + 1.35 * radius * c, y + 1.35 * radius * s]]), ring_col, 2.0, False))
            arm = 0.7 * radius
            items.append((np.array([[x - arm, y], [x + arm, y]]), arm_col, 1.5, False))
            items.append((np.array([[x, y - arm], [x, y + arm]]), arm_col, 1.5, False))
        self._set_overlay("gizmo", items)

    def set_layer_visibility(self, ducts: bool = True, walls: bool = True, props: bool = True,
                             floor: bool = True):
        hidden = set()
        for key, on in (("ducts", ducts), ("walls", walls), ("props", props), ("floor", floor)):
            if not on:
                hidden.update(_LAYER_GROUPS[key])
        self._hidden = hidden
        self.update()

    def _overlay_segments(self):
        """All overlay polylines as LINES vertex data (N, 7) plus (width, first, count) ranges."""
        by_width: Dict[float, List[np.ndarray]] = {}
        cross = max(0.05, min(3.0, 0.02 * self.dist))
        for items in self._ov.values():
            for it in items:
                if isinstance(it[0], str) and it[0] == "cross":
                    _, x, y, col, width = it
                    strips = [(np.array([[x - cross, y], [x + cross, y]]), col, width, False),
                              (np.array([[x, y - cross], [x, y + cross]]), col, width, False)]
                else:
                    strips = [it]
                for P, col, width, closed in strips:
                    P = np.asarray(P, np.float64)
                    if len(P) < 2:
                        continue
                    a = P if not closed else np.vstack([P, P[:1]])
                    seg = np.stack([a[:-1], a[1:]], 1).reshape(-1, 2)
                    data = np.empty((len(seg), 7), np.float32)
                    data[:, :2] = seg
                    data[:, 2] = OVERLAY_Z
                    data[:, 3:] = np.asarray(col, np.float32)
                    by_width.setdefault(float(width), []).append(data)
        ranges, chunks, first = [], [], 0
        for width in sorted(by_width):
            block = np.vstack(by_width[width])
            ranges.append((width, first, len(block)))
            chunks.append(block)
            first += len(block)
        return (np.vstack(chunks) if chunks else np.zeros((0, 7), np.float32)), ranges

    def _draw_overlays(self, vp):
        import moderngl
        sc, ctx = self.scene, self.scene.ctx
        if self._ov_dirty:
            data, self._ov_ranges = self._overlay_segments()
            self._ov_dirty = False
            if len(data):
                nbytes = int(data.nbytes)
                if self._ov_vbo is None or self._ov_capacity < nbytes:
                    self._release_overlay_gl()
                    self._ov_dirty = False
                    self._ov_capacity = max(nbytes, 256 * 7 * 4)
                    self._ov_vbo = ctx.buffer(reserve=self._ov_capacity, dynamic=True)
                    self._ov_vao = ctx.vertex_array(sc.line_prog, [(self._ov_vbo, "3f 4f", "in_pos", "in_col")])
                self._ov_vbo.write(np.ascontiguousarray(data).tobytes())
        if not self._ov_ranges or self._ov_vao is None:
            return
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        sc.line_prog["u_vp"].write(G._u(vp))
        for width, first, count in self._ov_ranges:
            ctx.line_width = float(width)
            self._ov_vao.render(moderngl.LINES, vertices=count, first=first)
        ctx.line_width = 1.0
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)

    # ---------------------------------------------------------------- painting
    def paintGL(self):
        t0 = self._t_paint_start = time.perf_counter()
        if not self._resolve_target():
            return
        if self.msaa_actual is None:
            self.msaa_actual = _gl_msaa_state()
        self._apply_geometry()
        try:
            if self.geometry_data is None:
                self.scene.clear()
                self.last_static_drawn = 0
                self._empty.setVisible(True)
            else:
                self._empty.setVisible(False)
                self._draw_editor()
        except Exception as exc:
            self._gl_error = f"{type(exc).__name__}: {exc}"
            self.gl_failed.emit(self._gl_error)
            try:
                self.scene.clear()
            except Exception:
                pass
        self._last_paint = time.perf_counter()
        draw_ms = (self._last_paint - t0) * 1e3
        self._draw_ms.append(draw_ms)
        if len(self._draw_ms) > 240:
            del self._draw_ms[:len(self._draw_ms) - 240]
        self.draw_timed.emit(draw_ms)
        dt = self.rate.tick()
        if dt is not None:
            self.frame_timed.emit(dt)

    def _draw_editor(self):
        sc = self.scene
        view, proj, eye, target, (fog0, fog1) = self._matrices()
        sc.set_fog(fog0, fog1)
        t_gl = time.perf_counter()
        self._cpu_ms.append((t_gl - self._t_paint_start) * 1e3)
        self.last_static_drawn = sc.draw_static(view, proj, eye, light_center=np.array([target[0], target[1], 0.0]),
                                                hidden=self._hidden)
        self._draw_overlays(np.asarray(proj, np.float64) @ np.asarray(view, np.float64))
        self._gl_ms.append((time.perf_counter() - t_gl) * 1e3)
        for buf in (self._cpu_ms, self._gl_ms):
            if len(buf) > 240:
                del buf[:len(buf) - 240]

    # ---------------------------------------------------------------- input
    @staticmethod
    def _mods(ev) -> int:
        return int(ev.modifiers())

    @staticmethod
    def _pos(ev) -> Tuple[float, float]:
        """Logical widget position as floats. `ev.x()` truncates to whole pixels, and at a wide
        zoom one pixel is several centimetres of ground; the float position costs nothing."""
        p = ev.localPos() if hasattr(ev, "localPos") else (ev.position() if hasattr(ev, "position") else ev.posF())
        return float(p.x()), float(p.y())

    def _camera_drag_mode(self, ev) -> Optional[str]:
        b, m = ev.button(), ev.modifiers()
        if b == QtCore.Qt.MiddleButton or (b == QtCore.Qt.LeftButton and m & QtCore.Qt.AltModifier):
            return "orbit"
        if b == QtCore.Qt.RightButton or (b == QtCore.Qt.LeftButton and (m & QtCore.Qt.ShiftModifier or self._space_down)):
            return "pan"
        return None

    def mousePressEvent(self, ev):
        self.setFocus(QtCore.Qt.MouseFocusReason)
        if self._drag_mode is not None:
            return                                    # a second button during a drag: ignored
        pos = self._pos(ev)
        mode = self._camera_drag_mode(ev)
        if mode is not None:
            self._drag_mode, self._drag_pos = mode, pos
            if mode == "orbit" and self.top:
                # dragging out of the top view: start from just below straight-down so the tilt
                # is continuous, and forget the saved perspective -- this *is* the new one
                self.top = False
                self.el = EL_MAX
            return
        g = self.ground_point(*pos)
        if g is None:
            return
        self._drag_mode, self._drag_pos = "edit", pos
        self._edit_button = int(ev.button())
        self._last_ground = g
        self.ground_pressed.emit(g[0], g[1], int(ev.button()), self._mods(ev), self.hit_test(*g))

    def mouseMoveEvent(self, ev):
        pos = self._pos(ev)
        if self._drag_mode == "orbit":
            dx, dy = pos[0] - self._drag_pos[0], pos[1] - self._drag_pos[1]
            self._drag_pos = pos
            self.az -= dx * 0.005
            self.el = float(np.clip(self.el + dy * 0.005, EL_MIN, EL_MAX))
            self._camera_moved()
            return
        if self._drag_mode == "pan":
            g0 = self.ground_point(*self._drag_pos)
            g1 = self.ground_point(*pos)
            self._drag_pos = pos
            if g0 is not None and g1 is not None:
                self.pivot = self.pivot + (np.asarray(g0) - np.asarray(g1))
                self._camera_moved()
            return
        g = self.ground_point(*pos)
        if self._drag_mode == "edit":
            if g is not None:
                self._last_ground = g
                self.ground_moved.emit(g[0], g[1], int(ev.buttons()), self._mods(ev))
            return
        if g is not None and int(ev.buttons()) == 0:
            self.hovered.emit(g[0], g[1], self.hit_test(*g))

    def mouseReleaseEvent(self, ev):
        mode = self._drag_mode
        if mode in ("orbit", "pan"):
            self._drag_mode = self._drag_pos = None
            return
        if mode == "edit" and int(ev.button()) == self._edit_button:
            self._drag_mode = self._drag_pos = None
            g = self.ground_point(*self._pos(ev)) or self._last_ground
            if g is not None:
                self.ground_released.emit(g[0], g[1], int(ev.button()), self._mods(ev))

    def mouseDoubleClickEvent(self, ev):
        """Qt replaces the second press of a double-click with this event, so a double-click is
        press / release / double_clicked / release: no second `ground_pressed`, and the trailing
        release is dropped because no press opened an edit drag."""
        if ev.button() != QtCore.Qt.LeftButton or self._camera_drag_mode(ev) is not None:
            return
        g = self.ground_point(*self._pos(ev))
        if g is not None:
            self.double_clicked.emit(g[0], g[1])

    def wheelEvent(self, ev):
        m = ev.modifiers()
        if m & (QtCore.Qt.ControlModifier | QtCore.Qt.AltModifier):
            self.wheel_edit.emit(int(ev.angleDelta().y()), int(m))
            return
        steps = ev.angleDelta().y() / 120.0
        if steps == 0.0:
            return
        self._zoom(steps, at=self._pos(ev))

    def leaveEvent(self, ev):
        super().leaveEvent(ev)
        self.left_widget.emit()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key in (QtCore.Qt.Key_F, QtCore.Qt.Key_Home):
            self.frame_all()
        elif key == QtCore.Qt.Key_5:
            self.set_top_view(not self.top)
        elif key == QtCore.Qt.Key_Space:
            self._space_down = True                   # a pan modifier, not a command
        else:
            self.key_pressed.emit(int(key), self._mods(ev))
            ev.ignore()                               # let the window's own shortcuts see it too
            return
        ev.accept()

    def keyReleaseEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Space and not ev.isAutoRepeat():
            self._space_down = False
        super().keyReleaseEvent(ev)
