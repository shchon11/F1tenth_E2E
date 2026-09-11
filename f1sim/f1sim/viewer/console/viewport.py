"""The 3D viewport: a QOpenGLWidget that draws with the existing moderngl `Scene`.

Qt owns the framebuffer, we own everything in it
-----------------------------------------------
A QOpenGLWidget does *not* render to framebuffer 0. It renders to an FBO Qt created, and the Qt
documentation is explicit that an application must not rebind 0 -- so `Scene.target`, which
defaults to `ctx.screen`, has to be pointed at `defaultFramebufferObject()` instead.

Getting a moderngl handle for someone else's FBO is `ctx.detect_framebuffer(glo)`, and the object
it returns needs care. In moderngl 5.12.0 `detect_framebuffer` sets `_is_reference = True`, and
`Framebuffer.__del__` honours that flag and skips the delete -- but `Framebuffer.release()` does
**not** check it and calls straight through to the C release, which deletes a non-zero framebuffer
name. Calling `.release()` on a detected framebuffer would therefore delete Qt's FBO out from
under it. `_ForeignFramebuffer` exists so that cannot happen by accident: it forwards everything
the scene needs and raises on `release`. Dropping the Python reference is the correct disposal.

The cache key is `(glo, width_px, height_px)`, not the name alone: a framebuffer name can be
reused after a resize, so an ID match does not imply the same surface. Sizes here are always
*physical* pixels (`devicePixelRatioF()` applied); Qt widget coordinates are logical, and the two
are kept apart deliberately -- GL viewport and camera aspect use physical, mouse events use
logical.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import gl_scene as G
from . import theme
from .frames import FrameBuffer, RateMeter
from .theme import C

#: Clickable camera modes. Every one of them is a button in the UI; the digit is a shortcut, not
#: the only way to reach it.
CAMERA_MODES: List[Tuple[str, str, str]] = [
    ("overview", "전체보기", "맵 전체를 화면에 맞춰 위에서 본다  (단축키 1)"),
    ("chase", "추격", "주시 차량 뒤에서 따라간다  (단축키 2)"),
    ("top", "위에서", "주시 차량 바로 위, 차량 방향 기준  (단축키 3)"),
    ("orbit", "궤도", "주시 차량 주위를 돈다. 드래그로 회전, 휠로 확대  (단축키 4)"),
    ("closeup", "근접", "주시 차량을 가까이서 천천히 돈다  (단축키 5)"),
]
CAMERA_KEYS = [k for k, _, _ in CAMERA_MODES]

#: The scene's label atlas holds 64 numbers and the trail buffer is allocated for 64 cars, so 64 is
#: the honest ceiling on what can be drawn. A session may simulate more; the UI says how many of
#: how many are on screen rather than quietly showing a subset.
MAX_RENDER_CARS = 64


def _gl_msaa_state():
    """`GL_SAMPLE_BUFFERS`, `GL_SAMPLES` and the `GL_MULTISAMPLE` enable, for whatever framebuffer
    is bound right now. Must be called with the context current and the target bound.

    Why ask GL rather than Qt: `QOpenGLWidget::format().samples()` is 0 by construction. Qt 5.15
    keeps the requested sample count in `requestedSamples` and deliberately zeroes the *context*
    format's samples, because the widget renders into an FBO it builds from `requestedSamples`
    separately (`qopenglwidget.cpp`, the requestedSamples handling and `recreateFbo`). Reading the
    widget or context format therefore tells you nothing about whether the framebuffer you are
    drawing into is multisampled -- only the bound framebuffer can answer that.
    """
    import ctypes
    import ctypes.util
    try:
        name = ctypes.util.find_library("GL") or "libGL.so.1"
        gl = ctypes.CDLL(name)
        out = ctypes.c_int(0)
        gl.glGetIntegerv(ctypes.c_uint(0x80A8), ctypes.byref(out))   # GL_SAMPLE_BUFFERS
        buffers = out.value
        gl.glGetIntegerv(ctypes.c_uint(0x80A9), ctypes.byref(out))   # GL_SAMPLES
        samples = out.value
        gl.glIsEnabled.restype = ctypes.c_ubyte
        enabled = bool(gl.glIsEnabled(ctypes.c_uint(0x809D)))        # GL_MULTISAMPLE
        return {"sample_buffers": buffers, "samples": samples, "multisample_enabled": enabled}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


class ForeignObjectError(RuntimeError):
    """Raised when something tries to free a GL object this process did not create."""


class _ForeignFramebuffer:
    """A framebuffer owned by Qt, wrapped so the scene can draw into it and nothing can free it."""

    def __init__(self, fbo, glo: int, size: Tuple[int, int]):
        self._fbo = fbo
        self.glo = int(glo)
        self.size = size

    def use(self):
        self._fbo.use()

    def clear(self, *a, **kw):
        self._fbo.clear(*a, **kw)

    def read(self, *a, **kw):
        return self._fbo.read(*a, **kw)

    @property
    def viewport(self):
        return self._fbo.viewport

    @viewport.setter
    def viewport(self, value):
        self._fbo.viewport = value

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def release(self):
        raise ForeignObjectError(
            "Qt가 소유한 기본 프레임버퍼는 해제할 수 없습니다. moderngl 5.12.0의 "
            "Framebuffer.release()는 _is_reference를 무시하고 glDeleteFramebuffers를 호출하므로 "
            "Qt의 FBO가 삭제됩니다. 파이썬 참조만 버리면 __del__이 올바르게 아무것도 하지 않습니다.")


@dataclass
class TrackGeometry:
    """Everything needed to draw one map, built by the worker, uploaded here.

    The arrays arrive ready to upload: the marching-squares contours and the tube/wall vertex
    generation happen in the worker process, so switching maps costs this thread an upload rather
    than a full geometry pass.
    """
    name: str
    bounds: Tuple[float, float, float, float]
    duct_height: float
    floor: Optional[tuple] = None          # (pos, nrm, col, idx) -- v2: an apron turned to the track
    ducts: Optional[tuple] = None
    walls: Optional[tuple] = None
    centerline: Optional[np.ndarray] = None
    raceline_xy: Optional[np.ndarray] = None
    raceline_v: Optional[np.ndarray] = None
    build_ms: float = 0.0                  # what the worker spent producing this

    # -- geometry v2. Every one of these is optional: a payload without `geometry_version` is v1 by
    # definition, and an older worker paired with this console must keep working.
    geometry_version: int = 1
    #: a disc under the apron whose rim fades into the clear colour, so the ground has no hard edge
    backdrop: Optional[tuple] = None
    #: bbox of what is actually drawn. `bounds` stays the map file's canvas, which on a SLAM map
    #: with `unknown_is_obstacle` can be far larger than anything visible.
    content_bounds: Optional[Tuple[float, float, float, float]] = None
    #: camera framing only -- {"up_angle_deg", "center", "extent"}. World coordinates, the PGM and
    #: everything the policy sees are unaffected; this says how to *look* at them.
    presentation: Optional[dict] = None
    #: Placed obstacles, already transformed into world space and **merged by material** in the
    #: worker: a tuple of {"material", "pos", "nrm", "col", "idx", "n_props", "n_tris"}. Merged
    #: because the alternative is one draw call per box on the track, and built in the worker
    #: because building them is numpy work that has no business on the thread that owns the GL
    #: context. There are four materials in the whole prop catalogue, so this is four uploads
    #: however many props a map places.
    props: Optional[tuple] = None

    def prop_counts(self):
        """(props, triangles, batches) -- what the status line and the tests both want."""
        if not self.props:
            return (0, 0, 0)
        return (sum(int(b.get("n_props", 0)) for b in self.props),
                sum(int(b.get("n_tris", 0)) for b in self.props),
                len(self.props))

    def frame(self):
        """(up_angle_rad, centre (2,), extent (2,)) for the camera, or None when v1.

        The worker measures the track's own heading from the centerline and its size from
        everything that will be drawn. Falling back to the axis-aligned `bounds` is what left a
        7-degree tilt on Korea and framed Monza against a canvas 2.6x the area of its content.
        """
        p = self.presentation or {}
        c, e = p.get("center"), p.get("extent")
        if c is None or e is None:
            return None
        return (math.radians(float(p.get("up_angle_deg", 0.0))),
                np.asarray(c, np.float64), np.asarray(e, np.float64))

    def drawn_bounds(self):
        """What is on screen: content bounds when the worker measured them, else the canvas."""
        return self.content_bounds if self.content_bounds is not None else self.bounds

    def nbytes(self) -> int:
        total = 0
        for arrays in (self.floor, self.ducts, self.walls, self.backdrop):
            if arrays:
                total += sum(int(a.nbytes) for a in arrays)
        for batch in (self.props or ()):
            for key in ("pos", "nrm", "col", "idx"):
                a = batch.get(key)
                if a is not None:
                    total += int(np.asarray(a).nbytes)
        for a in (self.centerline, self.raceline_xy, self.raceline_v):
            if a is not None:
                total += int(a.nbytes)
        return total


class ViewportWidget(QtWidgets.QOpenGLWidget):
    """Draws the scene and nothing else.

    Everything that used to share this thread -- PIL panel drawing, saliency backward passes,
    device syncs -- now happens in the worker process or in a plain Qt widget. What is left here is
    matrix maths, buffer writes and draw calls, which is the only way a frame time stays where it
    should be.
    """

    frame_timed = QtCore.pyqtSignal(float)         # ms *between* frames -- pacing, what is felt
    draw_timed = QtCore.pyqtSignal(float)          # ms *inside* paintGL -- how long drawing took
    gl_ready = QtCore.pyqtSignal(str)              # renderer string
    gl_failed = QtCore.pyqtSignal(str)
    upload_timed = QtCore.pyqtSignal(str, float)   # (what, ms) -- geometry upload cost, measured
    focus_change_requested = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(360, 260)
        self.setAttribute(QtCore.Qt.WA_AlwaysStackOnTop, False)

        self.ctx = None
        self.scene: Optional[G.Scene] = None
        self._target: Optional[_ForeignFramebuffer] = None
        self._target_key: Optional[Tuple[int, int, int]] = None
        self._gl_error: Optional[str] = None
        self._car_loaded = False
        self._n_beams = 0

        self.buffer: Optional[FrameBuffer] = None
        # Two different questions, deliberately kept apart. `rate` is the interval between painted
        # frames: that is the pacing a person sees. `draw` is how long the paint body itself takes.
        # A large interval with a small draw means nothing asked us to repaint -- a throttled or
        # unexposed window -- and calling that "slow rendering" would send someone optimising the
        # wrong thing. It is exactly what happens offscreen under Xvfb.
        self.rate = RateMeter()
        self._draw_ms: List[float] = []
        #: paintGL split into the part we are responsible for (picking a frame, building matrices,
        #: writing buffers -- Python and numpy) and the part the driver is (draw calls, which under
        #: a software rasteriser include the rasterisation itself). Without the split, "the frame
        #: took 57 ms" cannot be acted on: it might be our per-car loop or it might be llvmpipe.
        self._cpu_ms: List[float] = []
        self._gl_ms: List[float] = []
        #: How many times the keepalive had to restart a stalled pacing chain. Reported, not
        #: hidden: it is the difference between "the renderer is slow" and "nothing asked it to
        #: draw", and those need different fixes.
        self.stalls = 0
        #: Multisample state of the framebuffer actually drawn into, sampled once inside paintGL.
        #: Not derived from the widget or context format -- see `_gl_msaa_state`.
        self.msaa_actual = None
        #: Swap notifications actually received. If this stays at zero the display never tells us a
        #: frame went out, and the swap-driven pacing chain can never start -- so we pace ourselves.
        self.swaps = 0
        self.geometry_data: Optional[TrackGeometry] = None
        self._pending_geometry: Optional[TrackGeometry] = None
        self._clear_geometry = False

        self.camera = "overview"
        self.show_lidar = True
        self.show_raceline = True
        self.show_trails = True
        self.show_labels = True
        self.interpolate = True
        self.paused = False
        #: wall clock at the last camera update, so the chase easing can be expressed in time
        self._cam_wall: Optional[float] = None
        #: the heading 추격 is currently looking along, eased toward the car's
        self._chase_heading: Optional[float] = None
        #: (generation, focused car, x, y) at the last chase update -- see `_chase_discontinuity`
        self._chase_ref: Optional[tuple] = None
        self.plan: Optional[np.ndarray] = None
        self.plan_pred: Optional[np.ndarray] = None
        self.point_colors: Optional[np.ndarray] = None
        self.color_v_max = 6.0
        self.lidar_cfg: Dict[str, float] = {}
        self.vehicle_cfg: Dict[str, float] = {"lr": 0.15, "cog_z": 0.06, "wheel_r": 0.056}

        self._orbit = [math.radians(-35), math.radians(30), 4.0]
        self._drag: Optional[Tuple[float, float]] = None
        self._cam_eye = None
        self._cam_tgt = None
        self._spin = np.zeros(MAX_RENDER_CARS)
        self._last_frame_t: Optional[float] = None
        self._trail_seq = -1

        # Overlay chrome is made of ordinary child widgets rather than textures baked into the GL
        # frame: text drawn by Qt is text the OS can render in Korean, it costs the render thread
        # nothing, and it does not have to be rebuilt whenever a number changes.
        self._empty = QtWidgets.QLabel(self)
        self._empty.setAlignment(QtCore.Qt.AlignCenter)
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet(f"color: {C['text.2']}; background: transparent; font-size: 14px;")
        self._empty.setText("맵과 정책 런을 고르고 <b>시작</b>을 누르세요.")

        self._badge = QtWidgets.QLabel(self)
        self._badge.setVisible(False)
        self._badge.setAlignment(QtCore.Qt.AlignCenter)

        self._corner = QtWidgets.QLabel(self)
        self._corner.setStyleSheet(
            f"color: {C['text.1']}; background: rgba(13,17,23,190); border: 1px solid {C['line']};"
            f"border-radius: 6px; padding: 5px 9px; font-family: '{theme.MONO_FONT}', monospace;"
            f"font-size: {theme.SIZE['hint']}px;")
        self._corner.setVisible(False)

        # One pacer. `frameSwapped` fires after the buffer swap, so with vsync on this schedules the
        # next frame exactly when the display is ready for it; the cap timer only bites if the
        # driver or compositor ignored vsync, which would otherwise spin a core for nothing.
        self._cap_timer = QtCore.QTimer(self)
        self._cap_timer.setSingleShot(True)
        self._cap_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._cap_timer.timeout.connect(self.update)
        self.frameSwapped.connect(self._on_swapped)
        self.max_fps = 120.0
        self._running = False
        self._last_paint = 0.0
        self._t_paint_start = 0.0

        # Safety net for the pacing chain. `frameSwapped -> schedule -> update` is the right way to
        # drive a QOpenGLWidget, but it is a *chain*: if one swap notification never arrives the
        # animation stops for good and the viewer looks frozen while the simulation runs on. It
        # really does happen -- measured offscreen under Xvfb with no window manager, paints fell
        # to about 1 per second while each paint itself took 25 ms, which reads as a hopelessly
        # slow renderer and is nothing of the sort.
        #
        # This only fires when the chain has already starved, so on a healthy display it costs a
        # timer callback that finds nothing to do and never competes with vsync for pacing.
        self._keepalive = QtCore.QTimer(self)
        self._keepalive.setSingleShot(True)
        self._keepalive.timeout.connect(self._keepalive_tick)

    # ---------------------------------------------------------------- lifecycle
    def initializeGL(self):
        import moderngl
        try:
            self.ctx = moderngl.create_context()
            w, h = self.fb_size()
            # `msaa` only sizes the offscreen renderbuffers Scene builds for its headless path; on
            # the Qt path the target is the widget's own framebuffer and this value is unused. It is
            # 0 here by construction -- Qt zeroes the context format's samples and keeps the real
            # count on the FBO it builds itself (verified: the bound FBO reports GL_SAMPLES 4 when 4
            # was requested, 0 when 0 was). Left as 1 so the headless fallback stays valid.
            self.scene = G.Scene(self.ctx, w, h, headless=False, msaa=self.format().samples() or 1)
            # No clear here. `Scene.__init__` leaves `target = ctx.screen`, which is framebuffer 0 --
            # not the FBO a QOpenGLWidget draws into, and one Qt documents must not be rebound.
            # `paintGL` resolves the real target first and clears through that.
            self._target_key = None
            self._resolve_target()
            self._load_car()
            self._gl_error = None
            info = (f"{self.ctx.info.get('GL_RENDERER', '?')} · GL "
                    f"{self.ctx.info.get('GL_VERSION', '?')}")
            self.gl_ready.emit(info)
        except Exception as exc:                       # a viewer that cannot open GL must say so
            self.ctx = None
            self.scene = None
            self._gl_error = f"{type(exc).__name__}: {exc}"
            self.gl_failed.emit(self._gl_error)
            return
        ctx_obj = self.context()
        if ctx_obj is not None:
            ctx_obj.aboutToBeDestroyed.connect(self._on_context_destroyed)

    def _load_car(self):
        glb = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "assets", "f1tenth_car.glb")
        t0 = time.perf_counter()
        self.scene.load_car(glb, MAX_RENDER_CARS)
        self.scene.alloc_trails(MAX_RENDER_CARS, 600)
        self._car_loaded = True
        self.upload_timed.emit("car mesh", (time.perf_counter() - t0) * 1e3)

    def _on_context_destroyed(self):
        """Release what we created, while the context is still current.

        Explicit rather than left to refcounting: if anything at all still holds the Scene -- a
        stored traceback, a debugger, a future controller keeping a reference -- its moderngl
        objects would be finalised later, after the GL context is gone, and `glDelete*` would run
        against a dead context.

        Qt's own framebuffer is not in that list. `self._target` only ever holds a reference
        obtained from `detect_framebuffer`, and dropping the Python name is its correct disposal.
        """
        self._target = None                 # drop the reference; __del__ is a no-op for a reference
        self._target_key = None
        scene, self.scene = self.scene, None
        if scene is not None:
            try:
                scene.release()
            except Exception:
                pass
        self._car_loaded = False
        self._n_beams = 0
        self.geometry_data = None
        self._pending_geometry = None
        self.ctx = None

    def teardown(self):
        """Release our GL objects deliberately, with the context made current for it.

        `_on_context_destroyed` covers the path where Qt tears the context down and tells us. This
        covers the other one: the widget being destroyed by ordinary Python lifetime, where that
        signal may never arrive and the objects would otherwise be finalised with no current
        context. Calling it twice is safe.
        """
        if self.scene is None:
            self.ctx = None
            return
        try:
            self.makeCurrent()
            self.scene.release()
        except Exception:
            pass
        finally:
            try:
                self.doneCurrent()
            except Exception:
                pass
        self.scene = None
        self._target = None
        self._target_key = None
        self._car_loaded = False
        self._n_beams = 0
        self.geometry_data = None
        self.ctx = None

    def fb_size(self) -> Tuple[int, int]:
        """Framebuffer size in physical pixels (what GL wants), not logical widget units."""
        dpr = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
        return max(1, int(round(self.width() * dpr))), max(1, int(round(self.height() * dpr)))

    def resizeGL(self, w: int, h: int):
        # w/h arrive in physical pixels already, but the FBO name may change and may also be
        # *reused*, so the cache is invalidated on size as well as on name.
        self._target_key = None
        if self.scene is not None:
            self.scene.resize(max(1, w), max(1, h))

    def _resolve_target(self) -> bool:
        """Point the scene at Qt's current default framebuffer. True when a target is available."""
        if self.ctx is None or self.scene is None:
            return False
        glo = int(self.defaultFramebufferObject())
        w, h = self.fb_size()
        key = (glo, w, h)
        if key != self._target_key or self._target is None:
            fbo = self.ctx.detect_framebuffer(glo)
            self._target = _ForeignFramebuffer(fbo, glo, (w, h))
            self._target_key = key
        self.scene.target = self._target
        self.scene.width, self.scene.height = w, h
        return True

    # ---------------------------------------------------------------- driving
    def attach(self, buffer: FrameBuffer):
        self.buffer = buffer

    def set_running(self, running: bool):
        """Continuous repaint only while there is something moving to draw."""
        self._running = bool(running)
        if self._running:
            self._arm_keepalive()
            self.update()
        else:
            self._keepalive.stop()

    def _keepalive_tick(self):
        """The swap chain has gone quiet for longer than a frame should take. Ask for one.

        Armed after each paint and cancelled by the next, so on a healthy display it never fires.
        It used to poll every 16 ms instead, which cost more than it was worth: measured on the
        RTX 4060 Ti with a window manager, polling left the frame rate unchanged (72.4 against 70.2
        fps, the same 121 paints) while pushing input latency from 2.2 ms to 9.0 ms at the median.
        A safety net that runs constantly is not a safety net, it is overhead.

        It still earns its place offscreen: under Xvfb with llvmpipe and no window manager the swap
        chain does not sustain itself at all, and without this the picture fell to about a third of
        its frame rate.
        """
        if not self._running or self.ctx is None:
            return
        self.stalls += 1
        self.update()

    def _arm_keepalive(self):
        """(Re)start the stall deadline. One frame's grace at the target rate, floor 40 ms."""
        if not self._running:
            return
        target = 1.0 / max(1.0, min(self.max_fps or 60.0, 60.0))
        self._keepalive.start(int(max(40.0, 2.5 * target * 1e3)))

    def set_geometry(self, geom: Optional[TrackGeometry]):
        """Queue new map geometry; it is uploaded on the GL thread at the next paint."""
        if geom is None:
            self._clear_geometry = True
            self._pending_geometry = None
        else:
            self._pending_geometry = geom
            self._clear_geometry = False
        self.update()

    def _apply_geometry(self):
        if self.scene is None:
            return
        if self._clear_geometry:
            self._clear_geometry = False
            self.scene.clear_static()
            for name in ("centerline", "raceline"):
                self.scene.clear_line(name)
            self.geometry_data = None
            self._cam_eye = self._cam_tgt = None
            self._cam_wall = self._chase_heading = self._chase_ref = None
        geom = self._pending_geometry
        if geom is None:
            return
        self._pending_geometry = None
        t0 = time.perf_counter()
        self.scene.clear_static()
        self.scene.clear_line("centerline")
        self.scene.clear_line("raceline")
        if geom.backdrop is not None:
            # Before the apron, so the apron draws over it. No grid: the shader's grid is
            # `fract(world.xy)`, a 1 m lattice, and the backdrop is a disc hundreds of metres
            # across -- at overview distance that lattice is pure moire. Its rim is already the
            # scene's clear colour, which is the whole point: the ground stops having a hard edge.
            self.scene.add_static_mesh(*geom.backdrop, material="rubber",
                                       grid=False, casts=False, cull=False)
        if geom.floor is not None:
            self.scene.add_static_mesh(*geom.floor, material="rubber", grid=True, casts=False)
        if geom.ducts is not None:
            self.scene.add_static_mesh(*geom.ducts, material="metal", stripes=True)
        if geom.walls is not None:
            self.scene.add_static_mesh(*geom.walls, material="plastic", cull=False)
        for batch in (geom.props or ()):
            # Closed solids standing on the floor, so unlike the walls they cull and cast: a
            # cardboard box with no shadow reads as a decal rather than an object. The arrays
            # arrive world-transformed and merged, so this is an upload, not a build.
            arrays = tuple(batch[k] for k in ("pos", "nrm", "col", "idx"))
            self.scene.add_static_mesh(*arrays, material=str(batch.get("material", "plastic")))
        if geom.centerline is not None and len(geom.centerline) > 1:
            cl = np.vstack([geom.centerline, geom.centerline[:1]])
            self.scene.add_line("centerline", cl, (0.35, 0.55, 0.9, 0.35), z=0.008)
        if geom.raceline_xy is not None and len(geom.raceline_xy) > 1:
            v = geom.raceline_v if geom.raceline_v is not None else np.zeros(len(geom.raceline_xy), np.float32)
            t = (v - v.min()) / (v.max() - v.min() + 1e-6)
            cols = np.stack([t, 1 - np.abs(2 * t - 1), 1 - t, np.ones_like(t)], 1).astype(np.float32)
            self.scene.add_line("raceline", np.vstack([geom.raceline_xy, geom.raceline_xy[:1]]),
                                np.vstack([cols, cols[:1]]), z=0.012)
        self.geometry_data = geom
        self._cam_eye = self._cam_tgt = None
        self.scene.trail_fill = 0
        self._trail_seq = -1
        self.upload_timed.emit(f"map:{geom.name}", (time.perf_counter() - t0) * 1e3)

    def ensure_points(self, n_beams: int):
        """(Re)allocate the LiDAR point buffers for this beam count.

        `alloc_points` overwrites the attributes, so the previous VAO/VBO have to be released here
        or they stay on the GPU for the life of the process."""
        if self.scene is None or not n_beams or n_beams == self._n_beams:
            return
        for attr in ("pts_vao", "pts_vbo"):
            old = getattr(self.scene, attr, None)
            if old is not None:
                try:
                    old.release()
                except Exception:
                    pass
        self.scene.alloc_points(n_beams)
        self._n_beams = n_beams

    def set_camera(self, mode: str):
        if mode in CAMERA_KEYS and mode != self.camera:
            self.camera = mode
            self._cam_eye = self._cam_tgt = None
            self._cam_wall = self._chase_heading = self._chase_ref = None
            self.update()

    def set_badge(self, text: str, colour: str = ""):
        if not text:
            self._badge.setVisible(False)
            return
        colour = colour or C["warn"]
        self._badge.setText(text)
        self._badge.setStyleSheet(
            f"color: {colour}; background: rgba(13,17,23,215); border: 1px solid {colour};"
            f"border-radius: 8px; padding: 6px 12px; font-weight: 600; font-size: {theme.SIZE['label']}px;")
        self._badge.adjustSize()
        self._badge.setVisible(True)
        self._place_overlays()

    def set_corner_text(self, text: str):
        self._corner.setVisible(bool(text))
        if text:
            self._corner.setText(text)
            self._corner.adjustSize()
            self._place_overlays()

    def set_empty_text(self, html: str):
        self._empty.setText(html)

    # ---------------------------------------------------------------- painting
    def _on_swapped(self):
        self.swaps += 1
        if not self._running:
            return
        if self.max_fps and self.max_fps > 0:
            self._cap_timer.start(max(0, int(1000.0 / self.max_fps) - 1))
        else:
            self.update()

    def paintGL(self):
        t0 = self._t_paint_start = time.perf_counter()
        if not self._resolve_target():
            return
        if self.msaa_actual is None:
            # once, with Qt's framebuffer bound: this is the only place the answer is meaningful
            self.msaa_actual = _gl_msaa_state()
        self._apply_geometry()
        # One guard around everything that touches frame data. Picking the frame reads keys off it
        # too, so a malformed frame must not be able to throw out of paintGL -- an exception
        # escaping a paint handler takes the window with it, and the whole point of the split is
        # that a sick worker cannot kill the UI.
        try:
            frame = None
            if self.buffer is not None:
                frame = self.buffer.frame_to_draw(interpolate=self.interpolate and not self.paused)
            if frame is None or self.geometry_data is None:
                self.scene.clear()
                self._empty.setVisible(self.geometry_data is None)
            else:
                self._empty.setVisible(False)
                self._draw(frame)
        except Exception as exc:
            self._gl_error = f"{type(exc).__name__}: {exc}"
            self.gl_failed.emit(self._gl_error)
            try:
                self.scene.clear()
            except Exception:
                pass
        self._last_paint = time.perf_counter()
        self._arm_keepalive()          # cancels the pending deadline and sets the next one
        draw_ms = (self._last_paint - t0) * 1e3
        self._draw_ms.append(draw_ms)
        if len(self._draw_ms) > 240:
            del self._draw_ms[:len(self._draw_ms) - 240]
        self.draw_timed.emit(draw_ms)
        dt = self.rate.tick()
        if dt is not None:
            self.frame_timed.emit(dt)

    def _draw(self, fr: dict):
        sc = self.scene
        n, f = int(fr["n"]), int(fr["focus"])
        w, h = self.fb_size()
        eye, target, up = self._camera_pose(fr)
        view = G.look_at(eye, target, up)
        near, far, fog0, fog1 = self._depth_range(eye, target)
        proj = G.perspective(55.0, w / max(1, h), near, far)
        sc.set_fog(fog0, fog1)

        cog_x = float(self.vehicle_cfg.get("lr", 0.15))
        cog_z = float(self.vehicle_cfg.get("cog_z", 0.06))
        wheel_r = float(self.vehicle_cfg.get("wheel_r", 0.056))
        mats = np.zeros((n, 7, 4, 4), np.float32)
        dt_sim = float(fr.get("control_dt", 0.025))
        for i in range(n):
            x, y, yaw = fr["x"][i], fr["y"][i], fr["yaw"][i]
            c, s = math.cos(yaw), math.sin(yaw)
            rear = np.array([x - cog_x * c, y - cog_x * s, 0.0])
            M = G.trans(rear) @ G.rot_z(yaw)
            tilt = (G.trans([cog_x, 0, cog_z]) @ G.rot_y(fr["pitch"][i]) @ G.rot_x(fr["roll"][i])
                    @ G.trans([-cog_x, 0, -cog_z]))
            mats[i, 0] = M @ tilt
            mats[i, 1] = M @ tilt @ sc.car_pivots["lidar"]
            if i < len(self._spin):
                self._spin[i] += fr["vx"][i] * dt_sim / wheel_r
                spin = G.rot_y(self._spin[i])
            else:
                spin = np.eye(4, dtype=np.float32)
            for j, wname in enumerate(("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")):
                steer = G.rot_z(fr["steer"][i]) if j < 2 else np.eye(4, dtype=np.float32)
                mats[i, 2 + j] = M @ sc.car_pivots[wname] @ steer @ spin
            d, w_, zlo, zhi = fr["rear"][i]
            bx = cog_x - 0.5 * fr["len"][i] - 0.5 * d
            S = np.diag([d, w_, zhi - zlo, 1.0]).astype(np.float32)
            mats[i, 6] = M @ tilt @ G.trans([bx, 0.0, 0.5 * (zlo + zhi)]) @ S

        # Same colour vocabulary as the panels: cyan is the car being watched, amber a rival in its
        # race, red a collision, everything else dimmed so the watched car is findable at a glance.
        tint = np.full((n, 4), 0.45, np.float32)
        tint[:, 3] = 1.0
        opp = fr.get("opponent")
        if opp is not None and len(opp) == n:
            tint[np.asarray(opp, bool)] = (0.95, 0.62, 0.25, 1.0)
        tint[fr["coll"] > 0.5] = (1.0, 0.35, 0.3, 1.0)
        if 0 <= f < n:
            tint[f] = (0.35, 0.95, 1.0, 1.0)
        sc.set_car_instances(mats, tint, labels=fr.get("ids") if self.show_labels else None)

        # `draw_frame(show_points=...)` renders the point VAO, and that VAO only exists once a
        # frame carrying a scan has allocated it. A session with the LiDAR overlay on but no scan
        # in the frame yet -- which is every frame before the first one arrives -- would otherwise
        # draw a buffer that was never created.
        if self.show_lidar and fr.get("scan") is not None:
            pts, cols = self._scan_points(fr)
            if pts is not None:
                self.ensure_points(len(pts))
                sc.set_points(pts, cols)
        draw_points = self.show_lidar and self._n_beams > 0

        for slot, src in ((0, self.plan), (1, self.plan_pred)):
            if src is None or len(src) < 2:
                sc.set_plan(None, slot=slot)
                continue
            p_ = np.asarray(src, np.float32)
            v = p_[:, 2] if p_.shape[1] > 2 else np.zeros(len(p_), np.float32)
            cols = G.speed_colors(v, max(1e-3, self.color_v_max))
            if slot == 1:
                cols[:, 3] = 0.6
                cols[:, :3] = 0.5 * cols[:, :3] + 0.5
            z = 0.05 if slot == 0 else 0.03
            sc.set_plan(np.concatenate([p_[:, :2], np.full((len(p_), 1), z, np.float32)], 1), cols, slot=slot)

        seq = int(fr.get("seq", -1))
        if self.show_trails and seq != self._trail_seq:
            self._trail_seq = seq
            sc.push_trail_points(fr["x"], fr["y"])

        fx = fr["x"][f] if 0 <= f < n else 0.0
        fy = fr["y"][f] if 0 <= f < n else 0.0
        t_gl = time.perf_counter()
        self._cpu_ms.append((t_gl - self._t_paint_start) * 1e3)
        sc.draw_frame(view, proj, eye, light_center=np.array([fx, fy, 0.0]),
                      show_points=draw_points, show_race=self.show_raceline,
                      show_trails=self.show_trails, n_cars=n, focus_xy=(fx, fy))
        self._gl_ms.append((time.perf_counter() - t_gl) * 1e3)
        for buf in (self._cpu_ms, self._gl_ms):
            if len(buf) > 240:
                del buf[:len(buf) - 240]

    def _scan_points(self, fr):
        """LiDAR hit points of the focus car, same beam geometry as f1sim.lidar.Lidar.rays."""
        P = fr.get("P")
        if P is None or fr.get("scan") is None:
            return None, None
        f = int(fr["focus"])
        n = int(fr["n"])
        if not (0 <= f < n):
            return None, None
        x, y, yaw = fr["x"][f], fr["y"][f], fr["yaw"][f]
        phi = fr["roll"][f] + P["mount_roll"]
        th = fr["pitch"][f] + P["mount_pitch"]
        fov = float(self.lidar_cfg.get("fov", 4.71238898))
        r = np.asarray(fr["scan"], np.float64)
        nb = len(r)
        a = np.linspace(-fov / 2, fov / 2, nb) + P["mount_yaw"]
        ca, sa = np.cos(a), np.sin(a)
        cph, sph, cth, sth = math.cos(phi), math.sin(phi), math.cos(th), math.sin(th)
        bx = ca * cth + sa * sph * sth
        by = sa * cph
        bz = -ca * sth + sa * sph * cth
        cy, sy = math.cos(yaw), math.sin(yaw)
        wx = bx * cy - by * sy
        wy = bx * sy + by * cy
        mx, my, mz = P["mount_x"], P["mount_y"], P["mount_z"]
        t = my * sph + mz * cph
        ox_b = mx * cth + t * sth
        oy_b = my * cph - mz * sph
        oz_b = -mx * sth + t * cth
        o = np.array([x + ox_b * cy - oy_b * sy, y + ox_b * sy + oy_b * cy, oz_b])
        ok = np.isfinite(r)
        r = np.where(ok, r, 0.0)
        pts = o[None, :] + np.stack([wx, wy, bz], 1) * r[:, None]
        pts[~ok] = (0, 0, -100)
        typ = np.asarray(fr.get("scan_type", np.zeros(nb, np.int32)), np.int32)
        cols = np.zeros((nb, 4), np.float32)
        cols[typ == 1] = (1.0, 0.55, 0.15, 1.0)     # duct hose
        cols[typ == 2] = (1.0, 0.2, 0.9, 1.0)       # tall object
        cols[typ == 3] = (0.2, 0.9, 1.0, 1.0)       # floor
        cols[typ == 4] = (1.0, 0.95, 0.2, 1.0)      # another car
        cols[typ == 0] = (0.5, 0.5, 0.5, 1.0)
        if self.point_colors is not None and len(self.point_colors) == nb:
            cols = np.asarray(self.point_colors, np.float32)
        return pts.astype(np.float32), cols

    #: Time constant of the 추격 camera's lag, in seconds. At a steady 60 Hz this reproduces the
    #: per-paint fraction the previous fps-derived expression produced, so the feel is unchanged;
    #: what changes is that a long frame now moves the camera proportionally further.
    CAM_TAU_S = 0.112
    #: Continuity of the chase subject is tracked as (generation, focused car, x, y): a change of
    #: session or of focused car, or a jump no speed explains, resets the eased heading.

    #: Depth range for a camera sitting a few metres from its subject. Everything below scales up
    #: from here; nothing scales below it, so close cameras render exactly as they always did.
    NEAR_MIN, FAR_MIN = 0.05, 300.0
    FOG_MIN = (50.0, 200.0)

    def _depth_range(self, eye, target):
        """(near, far, fog start, fog end) for this camera, from its distance and the map's size.

        Three constants used to be baked in: near 0.05, far 300, fog 50-200 m. They are right for a
        chase camera and wrong for every camera that has to pull back. `전체보기` places the eye at
        whatever height frames the whole map -- 22 m on a 23 m indoor map, 199 m on Monza -- so on a
        large map every surface in the scene was past the end of the fog and the map rendered as one
        flat colour with the walls gone. A map larger than 300 m across would additionally have been
        clipped by the far plane, and holding near at 0.05 m while the eye is 200 m up spends the
        depth buffer's precision on space nothing occupies.

        So all three are derived rather than assumed, and each is clamped so that it can only ever
        grow past the old fixed value.
        """
        dist = float(np.linalg.norm(np.asarray(eye) - np.asarray(target)))
        gd = self.geometry_data
        diag = 0.0
        if gd is not None:
            # What has to fit in front of the far plane is what is drawn. On a SLAM map with
            # `unknown_is_obstacle` the file's canvas can be several times the content, and sizing
            # the far plane off it wastes depth range on space nothing occupies. v1 payloads have
            # no content bounds and fall back to the canvas, exactly as before.
            b = gd.drawn_bounds()
            if b is not None:
                diag = float(math.hypot(b[2] - b[0], b[3] - b[1]))
        # far has to clear the far corner of the map, not just the point being looked at
        far = max(self.FAR_MIN, dist + diag * 1.2 + 20.0)
        # Near rises with far, to hold far/near inside what a 24-bit depth buffer can separate --
        # but never past a twentieth of the distance to what the camera is looking at, so raising it
        # can never clip the subject out of its own view.
        near = min(max(far / 4000.0, self.NEAR_MIN), max(self.NEAR_MIN, dist * 0.05))
        # Fog scales with the camera distance alone, deliberately: tie it to map size as well and a
        # close camera on a big map would lose the haze that gives the picture depth. Every camera
        # that was already inside the old range keeps exactly the old range.
        fog0 = max(self.FOG_MIN[0], dist * 0.90)
        fog1 = max(self.FOG_MIN[1], dist * 2.60)
        return near, far, fog0, fog1

    def _chase_discontinuity(self, fr: dict, x: float, y: float, dt: float) -> bool:
        """Is the pose we are about to look at continuous with the one we last looked at?

        Three things break continuity, and none of them is an angle:

        * the user picked a **different car** -- a new subject entirely;
        * a **new session** -- generation changed under us;
        * the car was **put back on the grid** -- a reset moves it further than it could drive.

        A large turn is not itself a reset signal: a car really can spin, and easing through that
        is the right thing to do. Conversely a reset can leave the heading almost unchanged while
        the car crosses the map, which an angle test would sail straight through.
        """
        gen, focus = fr.get("gen"), int(fr.get("focus", 0))
        ref = self._chase_ref
        self._chase_ref = (gen, focus, x, y)
        if ref is None or ref[0] != gen or ref[1] != focus:
            return True
        if dt <= 0.0:
            return False
        moved = math.hypot(x - ref[2], y - ref[3])
        try:
            v = abs(float(np.asarray(fr["vx"])[focus]))
        except Exception:
            v = 0.0
        # the same generous rule the interpolator uses: catch a reset, not the physics
        return moved > 4.0 * (v + 1.0) * dt + 0.5

    def _chase_yaw(self, yaw: float, fr: dict, x: float, y: float) -> float:
        """The heading 추격 looks along: the car's, eased in time, wrap-safe.

        `dt` is measured, not assumed. A fixed fraction per paint eases by the same amount whether
        16 ms or 90 ms elapsed, which is what made the view lurch on a long frame.
        """
        now = time.perf_counter()
        dt = 0.0 if self._cam_wall is None else min(0.25, max(0.0, now - self._cam_wall))
        self._cam_wall = now
        if self._chase_discontinuity(fr, x, y, dt) or self._chase_heading is None or dt <= 0.0:
            self._chase_heading = yaw
            return yaw
        d = (yaw - self._chase_heading + math.pi) % (2 * math.pi) - math.pi
        self._chase_heading += d * (1.0 - math.exp(-dt / self.CAM_TAU_S))
        return self._chase_heading

    def _camera_pose(self, fr):
        f = int(fr["focus"])
        n = int(fr["n"])
        if 0 <= f < n:
            x, y, yaw = float(fr["x"][f]), float(fr["y"][f]), float(fr["yaw"][f])
        else:
            x = y = yaw = 0.0
        up = np.array([0, 0, 1.0])
        m = self.camera
        w, h = self.fb_size()
        if m == "chase":
            # Translation is rigid to the pose being drawn; only the heading is damped.
            #
            # The camera used to be a first-order filter on its own *world position*, chasing a
            # moving endpoint. Even with the time constant expressed correctly in seconds, that
            # leaves a lag proportional to speed and to how the frames happen to fall, so the car
            # drifts around inside the frame and every long paint shifts it. Anchoring the eye to
            # the interpolated car pose removes the translation lag entirely: the car sits where
            # the chase anchor says it sits, and what is smoothed is the direction the camera looks
            # from, which is the part that should ease rather than snap.
            yaw_c = self._chase_yaw(yaw, fr, x, y)
            eye = np.array([x - 1.7 * math.cos(yaw_c), y - 1.7 * math.sin(yaw_c), 0.75])
            target = np.array([x + 1.2 * math.cos(yaw_c), y + 1.2 * math.sin(yaw_c), 0.1])
        elif m == "top":
            eye = np.array([x, y, 14.0])
            target = np.array([x, y, 0.0])
            up = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        elif m == "overview":
            aspect = w / max(1, h)
            t = math.tan(math.radians(55.0) / 2)
            frame = self.geometry_data.frame() if self.geometry_data else None
            if frame is not None:
                # geometry v2: the worker measured which way the track lies and how big the drawn
                # content is. Standing the long axis up the screen is what takes Korea's 7-degree
                # tilt out of the picture, and framing on the content rather than the canvas is
                # what stops Monza being shown inside a rectangle 2.6x its own area.
                ang, centre, extent = frame
                cx, cy = float(centre[0]), float(centre[1])
                # `content_frame` measures extent[0] ALONG `ang` and extent[1] across it
                # (`u, v = e @ t, e @ n` with `t = (cos ang, sin ang)`). So the screen-vertical
                # size is whichever of the two lies along whatever we choose for `up`.
                along, across = float(extent[0]), float(extent[1])
                # Two ways to stand the frame up. Pick the one that needs the camera closer, which
                # is the one that wastes less of the viewport -- a landscape map laid across a
                # landscape screen should stay laid across it.
                d_along = max(along, across / aspect)          # up along the track's axis
                d_across = max(across, along / aspect)         # up across it
                if d_along <= d_across:
                    up = np.array([math.cos(ang), math.sin(ang), 0.0])
                    need = d_along
                else:
                    up = np.array([-math.sin(ang), math.cos(ang), 0.0])
                    need = d_across
                d = 0.5 * need / t * 1.08
            else:
                # geometry v1, or a worker that sent no presentation: the previous behaviour
                b = self.geometry_data.bounds if self.geometry_data else (x - 5, y - 5, x + 5, y + 5)
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                bw, bh = b[2] - b[0], b[3] - b[1]
                long_map, short_map = max(bw, bh), max(1e-6, min(bw, bh))
                turn = long_map / short_map > 1.1 and (bh > bw) != (aspect < 1.0)
                up = np.array([1.0, 0, 0]) if turn else np.array([0, 1.0, 0])
                vert, horiz = (bw, bh) if turn else (bh, bw)
                d = max(0.5 * vert / t, 0.5 * horiz / (t * aspect)) * 1.08
            eye = np.array([cx, cy, d])
            target = np.array([cx, cy, 0.0])
        elif m == "closeup":
            ang = time.perf_counter() * 0.4
            eye = np.array([x + 1.1 * math.cos(ang), y + 1.1 * math.sin(ang), 0.45])
            target = np.array([x, y, 0.1])
        else:
            az, el, d = self._orbit
            eye = np.array([x + d * math.cos(el) * math.cos(az), y + d * math.cos(el) * math.sin(az),
                            0.1 + d * math.sin(el)])
            target = np.array([x, y, 0.1])
        # Every mode now places the camera directly. `chase` used to be the exception, filtered in
        # world space; its easing lives in `_chase_yaw` instead, where it cannot become a lag on
        # where the car appears.
        self._cam_eye, self._cam_tgt = eye, target
        return self._cam_eye, self._cam_tgt, up

    # ---------------------------------------------------------------- input
    def mousePressEvent(self, ev):
        """A click never changes the camera mode.

        Pressing anywhere in the picture used to switch to 궤도 -- so glancing at the window, or
        clicking it to give it focus, silently threw away the view the user had chosen. The camera
        mode is chosen by its button or its shortcut, and by nothing else; dragging rotates only
        when 궤도 is already the mode.
        """
        if ev.button() == QtCore.Qt.LeftButton and self.camera == "orbit":
            self._drag = (ev.x(), ev.y())
        super().mousePressEvent(ev)

    def parent_camera_sync(self):
        """Keep the window's camera buttons in step with `self.camera`."""
        w = self.window()
        if hasattr(w, "sync_camera_buttons"):
            w.sync_camera_buttons(self.camera)

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag = None
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag is None or self.camera != "orbit":
            return
        dx, dy = ev.x() - self._drag[0], ev.y() - self._drag[1]
        self._drag = (ev.x(), ev.y())
        self._orbit[0] -= dx * 0.005
        self._orbit[1] = float(np.clip(self._orbit[1] + dy * 0.005, math.radians(2), math.radians(89)))
        self.update()

    def wheelEvent(self, ev):
        """Zoom is 궤도's, and stays there: the wheel must not switch modes either.

        The orbit distance is still updated in other modes so that turning the wheel and *then*
        choosing 궤도 lands where the user expected, but nothing about the current view changes.
        """
        steps = ev.angleDelta().y() / 120.0
        self._orbit[2] = float(np.clip(self._orbit[2] * (0.9 ** steps), 0.5, 80.0))
        if self.camera == "orbit":
            self.update()

    # ---------------------------------------------------------------- chrome placement
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._place_overlays()

    def _place_overlays(self):
        m = 12
        self._empty.setGeometry(m, 0, max(1, self.width() - 2 * m), self.height())
        if self._badge.isVisible():
            self._badge.adjustSize()
            self._badge.move(max(m, self.width() - self._badge.width() - m), m)
        if self._corner.isVisible():
            self._corner.adjustSize()
            self._corner.move(m, m)

    def draw_percentiles(self):
        """(p50, p95, max) ms spent inside paintGL, or None."""
        if not self._draw_ms:
            return None
        s = sorted(self._draw_ms)
        return (s[len(s) // 2], s[min(len(s) - 1, int(0.95 * len(s)))], s[-1])

    def paint_split(self):
        """Median ms of the paint we are responsible for, and of the GL draw itself."""
        def med(v):
            return sorted(v)[len(v) // 2] if v else None
        return {"our_cpu_ms_p50": med(self._cpu_ms), "gl_draw_ms_p50": med(self._gl_ms),
                "our_cpu_ms_max": max(self._cpu_ms) if self._cpu_ms else None,
                "gl_draw_ms_max": max(self._gl_ms) if self._gl_ms else None,
                "n": len(self._gl_ms)}

    def reset_timing(self):
        self.rate.reset()
        self._draw_ms.clear()
        self.stalls = 0
        self.swaps = 0
        self._cpu_ms.clear()
        self._gl_ms.clear()

    def grab_png(self, path: str) -> bool:
        """Screenshot through Qt, which reads the widget's own framebuffer correctly."""
        try:
            img = self.grabFramebuffer()
            return bool(img.save(path))
        except Exception:
            return False
