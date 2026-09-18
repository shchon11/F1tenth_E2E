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


def _as_bool(value, default: bool = False) -> bool:
    """Normalize bools crossing the JSON snapshot boundary, including 0.0/1.0 diagnostics."""
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "0", "false", "no", "off", "none", "null"):
            return False
        if text in ("1", "true", "yes", "on"):
            return True
        try:
            return bool(float(text))
        except ValueError:
            return default
    try:
        return bool(value)
    except (TypeError, ValueError):
        try:
            return bool(float(value))
        except (TypeError, ValueError):
            return default


def _finite_mu(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


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
        p.setBrush(QtGui.QColor(C["ego"]))
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
        self._grip_estimate = None
        self._grip_fault = False
        self._grip_used_mu = None
        self._grip_focus = None
        self._grip_status = "추정 대기"
        self.setToolTip("바깥 눈금은 정책이 명령할 수 있는 최대 속도, 빨간 구간은 속도 상한 초과.\n"
                        "주황 표시는 명령값, 흰 바늘/각도는 실제 측정값.\n\n"
                        "g-g는 시뮬레이터가 측정한 차체 가속도입니다. 빨강 원은 시뮬 참값 마찰 한계,\n"
                        "청록 원은 센서 기반 추정 q50이며 반투명 띠는 q10–q90 범위입니다.\n"
                        "적용 μ는 컨트롤러에 사용한 값입니다.")

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
        self._reset_grip()
        self._grip_focus = None
        self.update()

    def clear_grip_estimate(self):
        """Forget the previous estimate when a new session has no evidence yet."""
        self._reset_grip()
        self._grip_focus = None
        self.update()

    def _reset_grip(self):
        self._grip_estimate = None
        self._grip_fault = False
        self._grip_used_mu = None
        self._grip_status = "추정 대기"

    @property
    def grip_status(self) -> str:
        return self._grip_status

    def set_focus(self, focus: int):
        focus = int(focus)
        changed = self._grip_focus is not None and focus != self._grip_focus
        self._grip_focus = focus
        if changed:
            self._reset_grip()
            self.update()

    def set_grip_estimate(self, estimate):
        """Set a new estimate, retaining evidence while an incoming frame is uninformative."""
        if not isinstance(estimate, dict):
            return
        warm = _as_bool(estimate.get("warm"))
        finite = _as_bool(estimate.get("finite"))
        has_evidence = _as_bool(estimate.get("has_evidence"), default=warm and finite)
        informative = _as_bool(estimate.get("informative"), default=True)
        fault = _as_bool(estimate.get("fault"))
        if fault:
            used_mu = _finite_mu(estimate.get("used_mu"))
            self._grip_estimate = None
            self._grip_fault = True
            self._grip_used_mu = used_mu
            self._grip_status = f"센서 확인 · 적용 {used_mu:.2f}" if used_mu is not None else "추정 오류"
            self.update()
            return
        # Cold or explicitly evidence-free packets clear stale evidence before the retention path.
        if not has_evidence or not warm or not finite:
            self._reset_grip()
            self.update()
            return
        if not informative and (self._grip_estimate is not None or self._grip_fault):
            used_mu = _finite_mu(estimate.get("used_mu"))
            if used_mu is not None:
                self._grip_used_mu = used_mu
            if self._grip_fault:
                self._grip_status = (f"센서 확인 · 적용 {self._grip_used_mu:.2f}"
                                     if self._grip_used_mu is not None else "추정 오류")
            elif self._grip_estimate is not None:
                if used_mu is not None:
                    self._grip_estimate["used_mu"] = used_mu
                self._grip_status = f"추정 μ {self._grip_estimate['q50']:.2f} · 적용 {self._grip_estimate['used_mu']:.2f}"
            self.update()
            return
        if not informative:
            used_mu = _finite_mu(estimate.get("used_mu"))
            self._reset_grip()
            self._grip_used_mu = used_mu
            if used_mu is not None:
                self._grip_status = f"추정 대기 · 적용 {used_mu:.2f}"
            self.update()
            return
        try:
            q10 = float(estimate["q10"])
            q50 = float(estimate["q50"])
            q90 = float(estimate["q90"])
            used_mu = float(estimate["used_mu"])
            valid = all(math.isfinite(v) and v >= 0.0 for v in (q10, q50, q90, used_mu))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            self._grip_estimate = None
            self._grip_fault = False
            self._grip_status = "추정 대기"
        else:
            self._grip_estimate = {"used_mu": used_mu, "q10": q10, "q50": q50, "q90": q90}
            self._grip_fault = False
            self._grip_used_mu = used_mu
            self._grip_status = f"추정 μ {q50:.2f} · 적용 {used_mu:.2f}"
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
        p.setPen(QtGui.QPen(QtGui.QColor(C["line"]), 1))
        for i in range(1, cols):
            x = pad + i * (cw + pad) - pad / 2
            p.drawLine(QtCore.QPointF(x, pad + 4), QtCore.QPointF(x, h - pad - 4))
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
        """Flat ring gauge: a thin track arc, the measured speed swept in the accent colour, the
        cap zone in red, the commanded speed as a marker inside the ring, the number in the band
        below. No dial face, no rim, no needle -- an instrument cluster, not a dashboard prop."""
        cap_r, (cx, cy), r, val_r, name_r = self._bands(cell)
        a0, a1 = 210.0, -30.0

        def deg(val):
            return a0 + (a1 - a0) * min(max(val / self.v_max, 0.0), 1.0)

        def ang(val):
            return math.radians(deg(val))

        rect = QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(C["line.strong"]), 5, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
        p.drawArc(rect, int(a0 * 16), int((a1 - a0) * 16))
        if self._have and self.v_cap < self.v_max:   # beyond the cap in force
            p.setPen(QtGui.QPen(QtGui.QColor(C["danger"]), 5, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
            p.drawArc(rect, int(deg(self.v_cap) * 16), int((a1 - deg(self.v_cap)) * 16))
        step = 2.0 if self.v_max <= 12 else 4.0
        fm_font = QtGui.QFont(theme.MONO_FONT)
        fm_font.setPointSizeF(6.0)
        p.setFont(fm_font)
        fm = QtGui.QFontMetrics(fm_font)
        v_t = 0.0
        while v_t <= self.v_max + 1e-6:
            t = ang(v_t)
            ct, st = math.cos(t), math.sin(t)
            p.setPen(QtGui.QPen(QtGui.QColor(C["line.strong"]), 1.0))
            p.drawLine(QtCore.QPointF(cx + (r + 4) * ct, cy - (r + 4) * st),
                       QtCore.QPointF(cx + (r + 8) * ct, cy - (r + 8) * st))
            if r >= 34:
                lr = r + 15
                p.setPen(QtGui.QColor(C["text.2"]))
                p.drawText(QtCore.QRectF(cx + lr * ct - 11, cy - lr * st - fm.height() / 2, 22, fm.height()),
                           QtCore.Qt.AlignCenter, f"{v_t:g}")
            v_t += step
        if self._have:
            g = QtGui.QConicalGradient(QtCore.QPointF(cx, cy), a0)
            span = abs(a1 - a0)
            frac = max(1e-3, abs(deg(self.v) - a0) / span)
            g.setColorAt(0.0, QtGui.QColor(C["accent.deep"]))
            g.setColorAt(max(0.0, min(1.0, frac * 0.999)), QtGui.QColor(C["accent"]))
            pen = QtGui.QPen(QtGui.QBrush(g), 5)
            pen.setCapStyle(QtCore.Qt.FlatCap)
            p.setPen(pen)
            p.drawArc(rect, int(a0 * 16), int((deg(self.v) - a0) * 16))
            t = ang(self.v_cmd)                # commanded speed: a marker just inside the ring
            ct, st = math.cos(t), math.sin(t)
            tri = QtGui.QPolygonF([
                QtCore.QPointF(cx + (r - 4) * ct, cy - (r - 4) * st),
                QtCore.QPointF(cx + (r - 11) * ct - 3.2 * st, cy - (r - 11) * st - 3.2 * ct),
                QtCore.QPointF(cx + (r - 11) * ct + 3.2 * st, cy - (r - 11) * st + 3.2 * ct)])
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(C["rival"]))
            p.drawPolygon(tri)
        # the reading sits inside the ring, its unit under it
        self._line(p, QtCore.QRectF(cx - r, cy - r * 0.46, 2 * r, r * 0.62),
                   f"{self.v:4.2f}" if self._have else "—",
                   C["text.0"] if self._have else C["text.2"], size=max(9.0, r * 0.30), mono=True, bold=True)
        self._line(p, QtCore.QRectF(cx - r, cy + r * 0.14, 2 * r, r * 0.34), "m/s", C["text.2"], size=6.5)
        self._line(p, cap_r,
                   f"명령 {self.v_cmd:.1f} · 상한 {self.v_cap:.1f}" if self._have else "명령 — · 상한 —",
                   C["rival"])
        self._line(p, val_r, "", C["text.2"])
        self._line(p, name_r, "속도", C["text.2"])

    def _wheel(self, p, cell):
        """Steering as a centred bar: fill from the centre to the measured angle, a marker for the
        commanded angle. Left steer fills to the left. The drawn wheel it replaces looked like a
        wheel and read like nothing -- a bar reads in a glance from across the room."""
        cap_r, (cx, cy), r, val_r, name_r = self._bands(cell)
        live = self._have
        max_deg = 25.0
        bw, bh = min(cell.width() * 0.74, 2 * r * 1.7), max(7.0, r * 0.17)
        x0, y0 = cx - bw / 2, cy - bh / 2
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(C["line.strong"]))
        p.drawRoundedRect(QtCore.QRectF(x0, y0, bw, bh), bh / 2, bh / 2)
        p.setPen(QtGui.QPen(QtGui.QColor(C["text.2"]), 1.2))
        p.drawLine(QtCore.QPointF(cx, y0 - 6), QtCore.QPointF(cx, y0 + bh + 6))
        f = QtGui.QFont(theme.MONO_FONT)
        f.setPointSizeF(6.0)
        p.setFont(f)
        p.setPen(QtGui.QColor(C["text.2"]))
        p.drawText(QtCore.QRectF(x0 - 4, y0 + bh + 8, 48, 12), QtCore.Qt.AlignLeft, f"L {max_deg:.0f}°")
        p.drawText(QtCore.QRectF(x0 + bw - 44, y0 + bh + 8, 48, 12), QtCore.Qt.AlignRight, f"R {max_deg:.0f}°")
        if live:
            def px(rad):
                return cx - (bw / 2) * max(-1.0, min(1.0, math.degrees(rad) / max_deg))

            xm = px(self.steer)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(C["accent"]))
            lft, rgt = min(cx, xm), max(cx, xm)
            p.drawRoundedRect(QtCore.QRectF(lft, y0, max(2.0, rgt - lft), bh), bh / 2, bh / 2)
            xc = px(self.steer_cmd)
            p.setBrush(QtGui.QColor(C["rival"]))
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(xc, y0 - 3), QtCore.QPointF(xc - 4, y0 - 10),
                                           QtCore.QPointF(xc + 4, y0 - 10)]))
        self._line(p, cap_r,
                   f"명령 {math.degrees(self.steer_cmd):+.1f}°" if live else "명령 —", C["rival"])
        self._line(p, val_r, f"{math.degrees(self.steer):+5.1f}°" if live else "—",
                   C["text.0"] if live else C["text.2"], size=12.5, mono=True, bold=True)
        self._line(p, name_r, "조향", C["text.2"])

    def _gg_radii(self, radius):
        """Return the shared plot scale, truth radius, estimate radii, and pixels per m/s²."""
        truth_lim = max(1e-3, float(self.mu_g or 9.81))
        estimate = self._grip_estimate
        if estimate is None:
            plot_lim = truth_lim
            estimate_radii = None
        else:
            estimate_lim = max(estimate["q10"], estimate["q50"], estimate["q90"]) * 9.81
            plot_lim = max(truth_lim, estimate_lim)
            px = radius / plot_lim
            estimate_radii = tuple(estimate[key] * 9.81 * px for key in ("q10", "q50", "q90"))
        px = radius / plot_lim
        return plot_lim, truth_lim, radius * truth_lim / plot_lim, estimate_radii, px

    def _gg_plot(self, p, cell):
        cap_r, (cx, cy), r, val_r, name_r = self._bands(cell)
        lim, truth_lim, truth_r, estimate_radii, px = self._gg_radii(r)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(C["line.strong"]), 1))
        p.drawEllipse(QtCore.QPointF(cx, cy), r * 0.5, r * 0.5)
        if estimate_radii is not None:
            q10_r, q50_r, q90_r = estimate_radii
            if q90_r > q10_r + 1e-3:
                band = QtGui.QPainterPath()
                band.setFillRule(QtCore.Qt.OddEvenFill)
                band.addEllipse(QtCore.QPointF(cx, cy), q90_r, q90_r)
                band.addEllipse(QtCore.QPointF(cx, cy), q10_r, q10_r)
                p.setPen(QtCore.Qt.NoPen)
                band_colour = QtGui.QColor(C["ego"])
                band_colour.setAlpha(32)
                p.setBrush(band_colour)
                p.drawPath(band)
            p.setBrush(QtCore.Qt.NoBrush)
            p.setPen(QtGui.QPen(QtGui.QColor(C["ego"]), 1.5))
            p.drawEllipse(QtCore.QPointF(cx, cy), q50_r, q50_r)
        p.setPen(QtGui.QPen(QtGui.QColor(C["danger"]), 1.5))       # the friction circle itself
        p.drawEllipse(QtCore.QPointF(cx, cy), truth_r, truth_r)
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
                p.setBrush(QtGui.QColor(90, 169, 255, int(30 + 150 * (k / max(1, n - 1)))))
                p.drawEllipse(QtCore.QPointF(x, y), 1.5, 1.5)
        cur = self._gg[-1] if self._gg else (0.0, 0.0)
        g_now = float(np.hypot(*cur)) / 9.81
        self._line(p, cap_r, self._grip_status,
                   C["ego"] if estimate_radii is not None else
                   C["warn"] if self._grip_fault else C["text.2"], size=7.0)
        self._line(p, val_r, f"{g_now:4.2f} g" if self._have else "—",
                   C["text.0"] if self._have else C["text.2"], size=12.5, mono=True, bold=True)
        # The circle is the friction limit the simulator applied to this car, not a value the
        # policy estimated or observed; without the tag it reads as a grip estimate.
        self._line(p, name_r, f"시뮬 참값 · 마찰 한계 {truth_lim / 9.81:.2f} g" if self._have else "g-g (시뮬 참값)",
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
