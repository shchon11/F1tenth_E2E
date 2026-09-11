"""The policy-input panel and the dash, as Qt widgets.

These used to be PIL images rebuilt on the render thread and uploaded as GL textures every frame --
a thousand draw primitives and a device-to-host copy sitting between `poll_events` and
`swap_buffers`. They carry information worth keeping, so the *content* is preserved and only the
place it is drawn has moved: ordinary widgets painted by Qt, on their own modest timer, where a
slow repaint costs the 3D frame nothing.

What they draw is unchanged in meaning:
  * scan points coloured by per-beam saliency -- brightness, not hue, because hue is already spent
    on the plan's speed profile and one colour must not mean two things;
  * the plan the policy produced, on the same axes and the same speed ramp as the 3D line;
  * a speedometer with the commanded speed and the cap marked, a steering wheel, and a g-g trace
    against the friction circle.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import gl_scene as G
from . import theme
from .theme import C


def _qc(rgb, alpha: float = 1.0) -> QtGui.QColor:
    r, g, b = (int(255 * float(v)) for v in rgb[:3])
    return QtGui.QColor(r, g, b, int(255 * alpha))


class PolicyInputPanel(QtWidgets.QWidget):
    paint_ms: list = []          # instrumentation: how long each repaint of this panel took

    """The policy's newest scan from above, with its plan on the same axes.

    Only the newest scan: the temporal stack is what the encoder consumes, but drawing six frames
    made a smear that hid the thing worth seeing -- where the network is looking right now, and
    whether the plan it drew goes anywhere near the gap it was looking at.

    True scale. `span` is the half-width in metres, so a distance read off the picture is the
    distance; anything beyond is clipped rather than compressed.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(230, 230)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.span = 8.0
        self.range_max = 10.0
        self.v_max = 6.0
        self.fov = 4.71238898
        self._scan: Optional[np.ndarray] = None
        self._sal: Optional[np.ndarray] = None
        self._plan: Optional[np.ndarray] = None
        self._sal_age: Optional[float] = None
        self.setToolTip("정책이 방금 받은 LiDAR 스캔을 위에서 본 그림.\n"
                        "점 밝기 = 그 빔이 행동을 얼마나 움직이는가(saliency).\n"
                        "선 = 정책이 만든 로컬 플랜, 색은 그 속도 프로파일.")

    def set_data(self, scan: Optional[np.ndarray], sal: Optional[np.ndarray],
                 plan_ref: Optional[np.ndarray], v_max: float,
                 sal_age: Optional[float] = None):
        """`scan`: ranges in **metres**, as the frame carries them. Not normalised.

        The panel used to multiply by `range_max`, which is right for the normalised observation the
        network sees and wrong for the frame, where the range is already a distance. Multiplied, a
        0.9 m wall landed at 9 m and every beam was clipped out of the view -- the panel drew an
        empty circle beside a scene full of LiDAR points."""
        self._scan = None if scan is None else np.asarray(scan, np.float32)
        self._sal = None if sal is None else np.asarray(sal, np.float32)
        self._plan = None if plan_ref is None else np.asarray(plan_ref, np.float32)
        self.v_max = float(v_max or 6.0)
        self._sal_age = sal_age
        self.update()

    def clear(self):
        self._scan = self._sal = self._plan = None
        self.update()

    def paintEvent(self, _ev):
        import time as _t
        _t0 = _t.perf_counter()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QtGui.QColor(C["bg.viewport"]))
        legend_h = 18                                   # reserved, so the caption is never clipped
        plot_h = max(40, h - legend_h)
        size = min(w, plot_h)
        px_per_m = (size * 0.44) / max(1e-3, self.span)
        cx, cy = w * 0.5, plot_h * 0.55

        # range rings, with their distance written on them: the panel is only useful if a distance
        # can actually be read off it
        p.setPen(QtGui.QPen(QtGui.QColor(C["line"]), 1))
        step = 1 if self.span <= 5 else 2
        for r_ in range(step, int(self.span) + 1, step):
            rr = r_ * px_per_m
            p.drawEllipse(QtCore.QRectF(cx - rr, cy - rr, 2 * rr, 2 * rr))
        p.setPen(QtGui.QColor(C["text.2"]))
        f = p.font(); f.setPointSizeF(7.5); p.setFont(f)
        for r_ in range(step, int(self.span) + 1, step):
            p.drawText(QtCore.QPointF(cx + 3, cy - r_ * px_per_m - 3), f"{r_}m")

        if self._scan is None:
            p.setPen(QtGui.QColor(C["text.2"]))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "데이터 없음")
            self._draw_car(p, cx, cy)
            return

        sc = self._scan
        r = sc[0] if sc.ndim > 1 else sc          # metres already; see set_data
        nb = len(r)
        ang = np.linspace(-self.fov / 2, self.fov / 2, nb)
        keep = np.isfinite(r) & (r < self.range_max * 0.999) & (r <= self.span)
        if keep.any():
            rp = r[keep] * px_per_m
            # body frame: forward +x, left +y; on screen forward is up and left is left
            xs = cx - rp * np.sin(ang[keep])
            ys = cy - rp * np.cos(ang[keep])
            if self._sal is not None and len(self._sal) == nb:
                t = np.clip(self._sal[keep], 0.0, 1.0)
            else:
                t = np.full(int(keep.sum()), 0.55, np.float32)
            # brightness carries attention; hue stays reserved for the plan's speed
            lo = np.array([0.26, 0.34, 0.46], np.float32)
            hi = np.array([1.00, 1.00, 0.98], np.float32)
            rgb = lo + (hi - lo) * (t[:, None] ** 0.7)
            alpha = 0.45 + 0.55 * t
            # One image, splatted with numpy -- not a thousand QPainter ellipses.
            #
            # Measured on hardware before this change: this panel repainted 25 times a second at
            # 11.7 ms each, about 29% of the GUI thread, against 2.1 ms for the entire 3D scene.
            # That is the same mistake the old viewer made with PIL, moved into Qt: a diagnostic
            # readout costing six times the picture it annotates. The geometry is unchanged; only
            # the number of draw calls is.
            self._splat = np.zeros((h, w, 4), np.uint8)
            xi = np.rint(xs).astype(np.int32)
            yi = np.rint(ys).astype(np.int32)
            rgba = np.concatenate([rgb * 255.0, alpha[:, None] * 255.0], 1).astype(np.uint8)
            for dx in (-1, 0, 1):                 # 3x3, so one beam is visible
                for dy in (-1, 0, 1):
                    x2, y2 = xi + dx, yi + dy
                    ok = (x2 >= 0) & (x2 < w) & (y2 >= 0) & (y2 < h)
                    self._splat[y2[ok], x2[ok]] = rgba[ok]
            img = QtGui.QImage(self._splat.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
            p.drawImage(0, 0, img)

        if self._plan is not None and len(self._plan) > 1:
            pr = self._plan
            pts = [QtCore.QPointF(cx - float(pr[i, 1]) * px_per_m, cy - float(pr[i, 0]) * px_per_m)
                   for i in range(len(pr))]
            if pr.shape[1] > 3:
                cols = G.speed_colors(pr[:, 3], max(1e-3, self.v_max))
                for i in range(len(pts) - 1):
                    p.setPen(QtGui.QPen(_qc(cols[i]), 3.0, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                    p.drawLine(pts[i], pts[i + 1])
            else:
                p.setPen(QtGui.QPen(QtGui.QColor(245, 245, 250), 3.0))
                p.drawPolyline(QtGui.QPolygonF(pts))
        self._draw_car(p, cx, cy)

        p.setPen(QtGui.QColor(C["text.2"]))
        f = p.font(); f.setPointSizeF(7.0); p.setFont(f)
        legend = f"{nb}빔 · 반경 {self.span:.0f}m · 플랜색 = 속도 0→{self.v_max:.1f}"
        if self._sal is None:
            legend += " · saliency 꺼짐"
        elif self._sal_age is not None:
            legend += f" · saliency {self._sal_age:.1f}s 전"
        m = QtGui.QFontMetrics(p.font())
        p.drawText(QtCore.QRectF(5, h - legend_h + 2, w - 10, legend_h - 3),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   m.elidedText(legend, QtCore.Qt.ElideRight, int(w - 12)))
        PolicyInputPanel.paint_ms.append((_t.perf_counter() - _t0) * 1e3)
        if len(PolicyInputPanel.paint_ms) > 400:
            del PolicyInputPanel.paint_ms[:200]

    def _draw_car(self, p, cx, cy):
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(C["accent"]))
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(cx, cy - 8), QtCore.QPointF(cx - 6, cy + 7),
                                       QtCore.QPointF(cx + 6, cy + 7)]))


class DashPanel(QtWidgets.QWidget):
    paint_ms: list = []          # instrumentation: how long each repaint of this panel took

    """Speedometer, steering wheel and g-g trace.

    The g-g trace is the body-frame specific force the accelerometer reads, so it is what the car
    felt rather than what was asked of it. A car using its tyres traces a circle; one that only
    ever brakes and accelerates in a straight line traces a cross, which is what a policy leaving
    lap time on the table looks like.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.v = self.v_cmd = self.steer = self.steer_cmd = 0.0
        self.v_cap = 6.0
        self.v_max = 8.0
        self.mu_g: Optional[float] = None
        self._gg: List[Tuple[float, float]] = []
        self._have = False
        self.setToolTip("바깥 눈금은 정책이 명령할 수 있는 최대 속도, 빨간 구간은 속도 상한 초과.\n"
                        "주황 표시는 명령값, 흰 바늘/각도는 실제 측정값.\n\n"
                        "g-g는 시뮬레이터가 측정한 차체 가속도이고 원은 시뮬이 이 차에 적용한 마찰\n"
                        "한계입니다. 둘 다 시뮬 참값이며, 정책의 관측이나 접지력 추정이 아닙니다.")

    def set_data(self, v, v_cmd, v_cap, steer, steer_cmd, v_max, gg, mu_g):
        self.v, self.v_cmd, self.v_cap = float(v), float(v_cmd), float(v_cap)
        self.steer, self.steer_cmd = float(steer), float(steer_cmd)
        self.v_max = max(1.0, float(v_max))
        self._gg = list(gg or [])
        self.mu_g = mu_g
        self._have = True
        self.update()

    def clear(self):
        self._have = False
        self._gg = []
        self.update()

    # Each gauge owns a column with the same four bands -- caption, dial, value, name -- so nothing
    # can land on top of anything else regardless of how tall the panel is stretched.
    CAP_H, VAL_H, NAME_H = 15, 21, 14

    #: Below this a gauge starts eliding its own labels, which makes it useless. When three will
    #: not fit side by side they wrap onto another row and the panel grows, rather than all three
    #: becoming unreadable at once.
    MIN_GAUGE_W = 152
    ROW_H = 178

    def columns(self, width: Optional[int] = None) -> int:
        w = self.width() if width is None else width
        return max(1, min(3, int((w - 10) // self.MIN_GAUGE_W)))

    def rows_needed(self, width: Optional[int] = None) -> int:
        cols = self.columns(width)
        return (3 + cols - 1) // cols

    def sizeHint(self):
        return QtCore.QSize(3 * self.MIN_GAUGE_W + 24, self.ROW_H)

    def minimumSizeHint(self):
        return QtCore.QSize(self.MIN_GAUGE_W + 24, self.ROW_H)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.setMinimumHeight(self.rows_needed() * self.ROW_H)

    def paintEvent(self, _ev):
        import time as _t
        _t0 = _t.perf_counter()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QtGui.QColor(C["bg.card"]))
        w, h = self.width(), self.height()
        pad = 10
        cols = self.columns()
        rows = self.rows_needed()
        cw = (w - pad * (cols + 1)) / cols
        ch = (h - pad * (rows + 1)) / rows
        cells = []
        for i in range(3):
            r_, c_ = divmod(i, cols)
            cells.append(QtCore.QRectF(pad + c_ * (cw + pad), pad + r_ * (ch + pad), cw, ch))
        self._speedo(p, cells[0])
        self._wheel(p, cells[1])
        self._gg_plot(p, cells[2])
        # Same instrumentation as the policy panel, for the same reason: these gauges were redrawn
        # to look like instruments, and a redesign whose cost nobody measured is how the old viewer
        # ended up spending a third of its GUI thread on a readout.
        DashPanel.paint_ms.append((_t.perf_counter() - _t0) * 1e3)
        if len(DashPanel.paint_ms) > 400:
            del DashPanel.paint_ms[:200]

    def _bands(self, cell: QtCore.QRectF):
        """(caption rect, dial centre, dial radius, value rect, name rect) for one column."""
        dial_top = cell.top() + self.CAP_H
        dial_h = cell.height() - self.CAP_H - self.VAL_H - self.NAME_H
        r = max(18.0, min(cell.width() * 0.42, dial_h * 0.5))
        cx = cell.center().x()
        cy = dial_top + dial_h * 0.5
        return (QtCore.QRectF(cell.left(), cell.top(), cell.width(), self.CAP_H),
                (cx, cy), r,
                QtCore.QRectF(cell.left(), dial_top + dial_h, cell.width(), self.VAL_H),
                QtCore.QRectF(cell.left(), dial_top + dial_h + self.VAL_H, cell.width(), self.NAME_H))

    def _line(self, p, rect, text, colour=None, size=7.5, mono=False, bold=False):
        f = p.font()
        f.setFamily(theme.MONO_FONT if mono else theme.UI_FONT)
        f.setPointSizeF(size)
        f.setBold(bold)
        p.setFont(f)
        p.setPen(QtGui.QColor(colour or C["text.1"]))
        m = QtGui.QFontMetrics(f)
        p.drawText(rect, QtCore.Qt.AlignCenter,
                   m.elidedText(text, QtCore.Qt.ElideRight, int(rect.width())))

    def _speedo(self, p, cell):
        cap_r, (cx, cy), r, val_r, name_r = self._bands(cell)
        # Qt angles are degrees x 16, counter-clockwise from 3 o'clock. The dial sweeps clockwise
        # from 210 deg (bottom-left) to -30 deg (bottom-right), so every span here is negative.
        a0, a1 = 210.0, -30.0

        def deg(val):
            return a0 + (a1 - a0) * min(max(val / self.v_max, 0.0), 1.0)

        def ang(val):
            return math.radians(deg(val))

        rect = QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)
        p.setBrush(QtCore.Qt.NoBrush)          # arcs are strokes; a stale brush would fill them

        # -- dial face: a shallow well, so the ring reads as sitting in something
        face = QtGui.QRadialGradient(QtCore.QPointF(cx, cy - r * 0.25), r * 1.35)
        face.setColorAt(0.0, QtGui.QColor(C["bg.dial.hi"]))
        face.setColorAt(1.0, QtGui.QColor(C["bg.dial.lo"]))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(face)
        p.drawEllipse(QtCore.QPointF(cx, cy), r + 5, r + 5)

        # -- graduations. The dial is worth reading off, so it gets numbers, not just a sweep.
        step = 1.0 if self.v_max <= 12 else 2.0
        label_every = 2.0 if self.v_max <= 12 else 4.0
        fm_font = QtGui.QFont(theme.MONO_FONT)
        fm_font.setPointSizeF(6.0)
        p.setFont(fm_font)
        fm = QtGui.QFontMetrics(fm_font)
        v_t = 0.0
        while v_t <= self.v_max + 1e-6:
            t = ang(v_t)
            ct, st = math.cos(t), math.sin(t)
            major = abs(v_t / label_every - round(v_t / label_every)) < 1e-6
            r_in = r - (11 if major else 8)
            # No session, no cap: the caption reads "상한 —", so the dial must not colour a
            # limit it does not have.
            over = self._have and v_t > self.v_cap + 1e-6
            col = QtGui.QColor(C["danger"] if over else C["line.strong"])
            p.setPen(QtGui.QPen(col, 1.6 if major else 1.0))
            p.drawLine(QtCore.QPointF(cx + r_in * ct, cy - r_in * st),
                       QtCore.QPointF(cx + (r - 4.5) * ct, cy - (r - 4.5) * st))
            if major and r >= 34:
                s = f"{v_t:g}"
                lr = r - 20
                p.setPen(QtGui.QColor(C["text.2"]))
                p.drawText(QtCore.QRectF(cx + lr * ct - 11, cy - lr * st - fm.height() / 2,
                                         22, fm.height()),
                           QtCore.Qt.AlignCenter, s)
            v_t += step

        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(C["line.strong"]), 6, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
        p.drawArc(rect, int(a0 * 16), int((a1 - a0) * 16))
        if self._have and self.v_cap < self.v_max:   # beyond the cap in force, in red
            p.setPen(QtGui.QPen(QtGui.QColor(C["danger"]), 6, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
            p.drawArc(rect, int(deg(self.v_cap) * 16), int((a1 - deg(self.v_cap)) * 16))
        if self._have:
            # the swept arc, brightening towards the current reading
            g = QtGui.QConicalGradient(QtCore.QPointF(cx, cy), a0)
            span = abs(a1 - a0)
            frac = max(1e-3, abs(deg(self.v) - a0) / span)
            g.setColorAt(0.0, QtGui.QColor(C["accent.deep"]))
            g.setColorAt(max(0.0, min(1.0, frac * 0.999)), QtGui.QColor(C["accent"]))
            pen = QtGui.QPen(QtGui.QBrush(g), 6)
            pen.setCapStyle(QtCore.Qt.FlatCap)
            p.setPen(pen)
            p.drawArc(rect, int(a0 * 16), int((deg(self.v) - a0) * 16))

            t = ang(self.v_cmd)                # commanded speed: a marker outside the ring
            ct, st = math.cos(t), math.sin(t)
            tri = QtGui.QPolygonF([
                QtCore.QPointF(cx + (r + 2) * ct, cy - (r + 2) * st),
                QtCore.QPointF(cx + (r + 9) * ct - 3.4 * st, cy - (r + 9) * st - 3.4 * ct),
                QtCore.QPointF(cx + (r + 9) * ct + 3.4 * st, cy - (r + 9) * st + 3.4 * ct)])
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(C["rival"]))
            p.drawPolygon(tri)

            t = ang(self.v)                    # measured speed: a tapered needle with a tail
            ct, st = math.cos(t), math.sin(t)
            tip, tail, half = r - 9, 9.0, 2.6
            needle = QtGui.QPolygonF([
                QtCore.QPointF(cx + tip * ct, cy - tip * st),
                QtCore.QPointF(cx - tail * ct - half * st, cy + tail * st - half * ct),
                QtCore.QPointF(cx - tail * ct + half * st, cy + tail * st + half * ct)])
            p.setBrush(QtGui.QColor(0, 0, 0, 90))
            p.drawPolygon(needle.translated(1.0, 1.5))
            p.setBrush(QtGui.QColor(C["needle"]))
            p.drawPolygon(needle)

        # hub: a ring around a filled centre, so the needle looks pinned rather than glued
        p.setPen(QtGui.QPen(QtGui.QColor(C["line.strong"]), 1.5))
        p.setBrush(QtGui.QColor(C["bg.dial.lo"]))
        p.drawEllipse(QtCore.QPointF(cx, cy), 5.2, 5.2)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(C["text.0"] if self._have else C["text.2"]))
        p.drawEllipse(QtCore.QPointF(cx, cy), 2.4, 2.4)
        self._line(p, cap_r,
                   f"명령 {self.v_cmd:.1f} · 상한 {self.v_cap:.1f}" if self._have else "명령 — · 상한 —",
                   C["rival"])
        self._line(p, val_r, f"{self.v:4.2f}" if self._have else "—",
                   C["text.0"] if self._have else C["text.2"], size=12.5, mono=True, bold=True)
        self._line(p, name_r, "속도 m/s", C["text.2"])

    def _wheel(self, p, cell):
        cap_r, (cx, cy), r, val_r, name_r = self._bands(cell)
        r *= 0.88
        live = self._have

        # -- fixed reference behind the wheel: the arc the rim turns through, and centre
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(C["line"]), 1))
        arc_r = r + 7
        p.drawArc(QtCore.QRectF(cx - arc_r, cy - arc_r, 2 * arc_r, 2 * arc_r), int(40 * 16), int(100 * 16))
        p.setPen(QtGui.QPen(QtGui.QColor(C["line.strong"]), 1.4))
        p.drawLine(QtCore.QPointF(cx, cy - arc_r - 3), QtCore.QPointF(cx, cy - arc_r + 3))
        if live:                                           # travelled angle, centre to now
            sweep = -math.degrees(self.steer) * 3.0
            p.setPen(QtGui.QPen(QtGui.QColor(C["accent"]), 2.5, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
            p.drawArc(QtCore.QRectF(cx - arc_r, cy - arc_r, 2 * arc_r, 2 * arc_r),
                      int(90 * 16), int(sweep * 16))

        p.save()
        p.translate(cx, cy)
        p.rotate(-math.degrees(self.steer) * 3.0)          # x3 so a 5-degree input is visible

        # -- rim: a flat-bottom wheel drawn as a filled ring, not a stroked circle
        rim_w = max(3.5, r * 0.17)
        outer = QtGui.QPainterPath()
        outer.addEllipse(QtCore.QPointF(0, 0), r, r)
        inner = QtGui.QPainterPath()
        inner.addEllipse(QtCore.QPointF(0, 0), r - rim_w, r - rim_w)
        ring = outer.subtracted(inner)
        # A shallow flat bottom, F1 style. Shallow on purpose: cut deep and the rim stops reading
        # as a wheel and starts reading as a broken ring, which is worse than no flat at all.
        flat = QtGui.QPainterPath()
        flat.addRect(QtCore.QRectF(-r - 2, r * 0.86, 2 * r + 4, r))
        ring = ring.subtracted(flat)
        grad = QtGui.QLinearGradient(0, -r, 0, r)
        grad.setColorAt(0.0, QtGui.QColor(C["rim.hi"] if live else C["text.2"]))
        grad.setColorAt(1.0, QtGui.QColor(C["rim.lo"] if live else C["line.strong"]))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(grad)
        p.drawPath(ring)
        p.setPen(QtGui.QPen(QtGui.QColor(C["rim.edge"]), 1))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPath(ring)

        # -- grips: the thickened sections a driver's hands sit on, at 10 and 2 o'clock. Clipped to
        # the rim band so they read as part of it rather than as blobs stuck on top.
        band = outer.subtracted(inner)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(C["rim.grip"] if live else C["line.strong"]))
        for a_ in (150.0, 30.0):
            t = math.radians(a_)
            g = QtGui.QPainterPath()
            g.addEllipse(QtCore.QPointF((r - rim_w * 0.5) * math.cos(t),
                                        -(r - rim_w * 0.5) * math.sin(t)),
                         r * 0.30, r * 0.30)
            p.drawPath(g.intersected(band))

        # -- spokes: tapered, so they read as structure rather than three lines from a point. The
        # lower spoke stops at the flat, which is where the rim it would meet has been cut away.
        p.setBrush(QtGui.QColor(C["spoke"] if live else C["text.2"]))
        for a_, wid, reach in ((180.0, 0.15, 1.0), (0.0, 0.15, 1.0), (270.0, 0.12, 0.86)):
            t = math.radians(a_)
            ct, st = math.cos(t), math.sin(t)
            hw = max(2.0, r * wid)
            end = (r - rim_w * 0.4) * reach
            sp = QtGui.QPolygonF([
                QtCore.QPointF(-hw * 0.55 * st, -hw * 0.55 * ct),
                QtCore.QPointF(hw * 0.55 * st, hw * 0.55 * ct),
                QtCore.QPointF(end * ct + hw * 0.32 * st, -end * st + hw * 0.32 * ct),
                QtCore.QPointF(end * ct - hw * 0.32 * st, -end * st - hw * 0.32 * ct)])
            p.drawPolygon(sp)

        # -- hub
        p.setBrush(QtGui.QColor(C["bg.dial.lo"]))
        p.setPen(QtGui.QPen(QtGui.QColor(C["rim.edge"]), 1))
        p.drawEllipse(QtCore.QPointF(0, 0), r * 0.26, r * 0.26)

        # -- top-centre marker: which way the wheel is actually pointing
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(C["accent"] if live else C["text.2"]))
        mk = QtGui.QPainterPath()
        mk.moveTo(0, -r + rim_w * 0.15)
        mk.lineTo(-rim_w * 0.42, -r + rim_w * 0.95)
        mk.lineTo(rim_w * 0.42, -r + rim_w * 0.95)
        mk.closeSubpath()
        p.drawPath(mk)
        p.restore()

        if live:                                           # ghost tick: the commanded angle
            t = math.radians(90 + math.degrees(self.steer_cmd) * 3.0)
            ct, st = math.cos(t), math.sin(t)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(C["rival"]))
            p.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(cx + (r + 4) * ct, cy - (r + 4) * st),
                QtCore.QPointF(cx + (r + 11) * ct - 3.0 * st, cy - (r + 11) * st - 3.0 * ct),
                QtCore.QPointF(cx + (r + 11) * ct + 3.0 * st, cy - (r + 11) * st + 3.0 * ct)]))
        self._line(p, cap_r,
                   f"명령 {math.degrees(self.steer_cmd):+.1f}°" if self._have else "명령 —", C["rival"])
        self._line(p, val_r, f"{math.degrees(self.steer):+5.1f}°" if self._have else "—",
                   C["text.0"] if self._have else C["text.2"], size=12.5, mono=True, bold=True)
        self._line(p, name_r, "조향 (휠 x3 과장)", C["text.2"])

    def _gg_plot(self, p, cell):
        cap_r, (cx, cy), r, val_r, name_r = self._bands(cell)
        lim = float(self.mu_g or 9.81)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(C["line.strong"]), 1))
        p.drawEllipse(QtCore.QPointF(cx, cy), r * 0.5, r * 0.5)
        p.setPen(QtGui.QPen(QtGui.QColor(C["danger"]), 1.5))       # the friction circle itself
        p.drawEllipse(QtCore.QPointF(cx, cy), r, r)
        p.setPen(QtGui.QPen(QtGui.QColor(C["line"]), 1))
        p.drawLine(QtCore.QPointF(cx - r, cy), QtCore.QPointF(cx + r, cy))
        p.drawLine(QtCore.QPointF(cx, cy - r), QtCore.QPointF(cx, cy + r))
        px = r / max(1e-3, lim)
        p.setPen(QtCore.Qt.NoPen)
        n = len(self._gg)
        for k, (alat, alon) in enumerate(self._gg):
            x = cx + float(np.clip(alat, -lim * 1.3, lim * 1.3)) * px
            y = cy - float(np.clip(alon, -lim * 1.3, lim * 1.3)) * px
            if k == n - 1:                                          # newest, brightest
                p.setBrush(QtGui.QColor("#ffffff"))
                p.drawEllipse(QtCore.QPointF(x, y), 2.8, 2.8)
            else:
                p.setBrush(QtGui.QColor(110, 190, 255, int(30 + 150 * (k / max(1, n - 1)))))
                p.drawEllipse(QtCore.QPointF(x, y), 1.5, 1.5)
        cur = self._gg[-1] if self._gg else (0.0, 0.0)
        g_now = float(np.hypot(*cur)) / 9.81
        self._line(p, cap_r, "↑가속  ↓제동  ↔횡가속", C["text.2"], size=7.0)
        self._line(p, val_r, f"{g_now:4.2f} g" if self._have else "—",
                   C["text.0"] if self._have else C["text.2"], size=12.5, mono=True, bold=True)
        # The circle is the friction limit the simulator applied to this car, not a value the
        # policy estimated or observed; without the tag it reads as a grip estimate.
        self._line(p, name_r, f"시뮬 참값 · 마찰 한계 {lim / 9.81:.2f} g" if self._have else "g-g (시뮬 참값)",
                   C["text.2"])


class ActivationPanel(QtWidgets.QWidget):
    """Raw hidden-layer and conv-feature activations, when they are asked for.

    Off by default: it is a debugging view, it costs a device transfer per update, and it is the
    kind of thing that quietly slows a session down while looking harmless.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._hidden: Optional[np.ndarray] = None
        self._stem: Optional[np.ndarray] = None

    def set_data(self, hidden, stem):
        self._hidden = None if hidden is None else np.asarray(hidden, np.float32)
        self._stem = None if stem is None else np.asarray(stem, np.float32)
        self.update()

    def clear(self):
        self._hidden = self._stem = None
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(C["bg.viewport"]))
        if self._hidden is None:
            p.setPen(QtGui.QColor(C["text.2"]))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "활성화 데이터 없음")
            return
        hg = self._hidden.reshape(16, 16)
        hg = (hg - hg.min()) / (hg.max() - hg.min() + 1e-9)
        cell = max(3, min(int(self.height() * 0.9) // 16, 10))
        p.setPen(QtCore.Qt.NoPen)
        for r_ in range(16):
            for c_ in range(16):
                v = float(hg[r_, c_])
                p.setBrush(QtGui.QColor(int(255 * v), int(120 * v), int(255 * (1 - v))))
                p.drawRect(8 + c_ * cell, 8 + r_ * cell, cell - 1, cell - 1)
        x0 = 8 + 16 * cell + 14
        if self._stem is not None:
            sg = self._stem
            sg = (sg - sg.min()) / (sg.max() - sg.min() + 1e-9)
            cols = max(16, min(64, (self.width() - x0 - 10) // 5))
            for i in range(min(len(sg), 256)):
                v = float(sg[i])
                p.setBrush(QtGui.QColor(int(255 * v), int(200 * v), 60))
                p.drawRect(int(x0 + (i % cols) * 5), int(8 + (i // cols) * 9), 4, 7)
        p.setPen(QtGui.QColor(C["text.2"]))
        f = p.font(); f.setPointSizeF(7.5); p.setFont(f)
        p.drawText(QtCore.QRectF(8, self.height() - 16, self.width() - 16, 14), QtCore.Qt.AlignLeft,
                   "왼쪽: hidden 256 뉴런  ·  오른쪽: 1D conv 스캔 특징 256")
