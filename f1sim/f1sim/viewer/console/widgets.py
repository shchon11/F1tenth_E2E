"""Reusable pieces of the console UI.

Two rules run through this file.

* **A control never claims a state the backend has not confirmed.** `PendingButton` and
  `PendingToggle` draw a distinct "적용 중" look between the click and the worker's ack; only the
  ack moves them to the confirmed state. Controls whose truth lives entirely in the GUI (camera,
  overlay visibility) are built with `local_truth=True` and settle immediately -- the difference is
  explicit rather than accidental.
* **A number that is not known is drawn as `—`, never as 0.** `MetricTile.set_unknown()` exists so
  that "no data yet" and "measured zero" cannot look the same.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from . import theme
from .theme import C, SP


# ---------------------------------------------------------------- small helpers
def label(text: str, kind: str = "body", parent=None) -> QtWidgets.QLabel:
    """A QLabel wired to one of the stylesheet's named roles."""
    lb = QtWidgets.QLabel(text, parent)
    lb.setObjectName({"body": "", "section": "SectionLabel", "field": "FieldLabel", "hint": "Hint",
                      "hint.warn": "HintWarn", "hint.danger": "HintDanger", "mono": "Mono"}.get(kind, ""))
    if kind in ("hint", "hint.warn", "hint.danger"):
        lb.setWordWrap(True)
    return lb


def hline(parent=None) -> QtWidgets.QFrame:
    f = QtWidgets.QFrame(parent)
    f.setFrameShape(QtWidgets.QFrame.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {C['line']}; border: none;")
    return f


class Card(QtWidgets.QFrame):
    """Titled container. The title is optional -- some cards are pure content."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(SP[2], SP[2], SP[2], SP[2])
        outer.setSpacing(SP[1])
        if title:
            head = QtWidgets.QHBoxLayout()
            head.setSpacing(SP[1])
            self.title_label = label(title, "section", self)
            head.addWidget(self.title_label)
            head.addStretch(1)
            self.head_layout = head
            outer.addLayout(head)
        else:
            self.title_label = None
            self.head_layout = None
        self.body = QtWidgets.QVBoxLayout()
        self.body.setSpacing(SP[1])
        outer.addLayout(self.body)
        self._outer = outer

    def add(self, w):
        if isinstance(w, QtWidgets.QLayout):
            self.body.addLayout(w)
        else:
            self.body.addWidget(w)
        return w


class Collapsible(QtWidgets.QWidget):
    """A section that folds away. Advanced options live here so they do not crowd the normal path."""

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(SP[1])
        self.toggle = QtWidgets.QPushButton(("▾  " if expanded else "▸  ") + title, self)
        self.toggle.setObjectName("GhostButton")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setStyleSheet("text-align: left;")
        self.toggle.setAccessibleName(f"{title} 접기/펼치기")
        self._title = title
        # The header row can carry a control or two. On a small window, giving a single setting a
        # row of its own costs more vertical space than the setting is worth.
        self._header = QtWidgets.QHBoxLayout()
        self._header.setContentsMargins(0, 0, 0, 0)
        self._header.setSpacing(SP[1])
        self._header.addWidget(self.toggle, 1)
        v.addLayout(self._header)
        self.content = QtWidgets.QWidget(self)
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, SP[0])
        self.content_layout.setSpacing(SP[1])
        self.content.setVisible(expanded)
        v.addWidget(self.content)
        self.toggle.toggled.connect(self._on_toggle)

    def _on_toggle(self, on: bool):
        self.content.setVisible(on)
        self.toggle.setText(("▾  " if on else "▸  ") + self._title)

    def add(self, w):
        if isinstance(w, QtWidgets.QLayout):
            self.content_layout.addLayout(w)
        else:
            self.content_layout.addWidget(w)
        return w

    def add_header(self, w):
        """Put a control on the title row instead of giving it a row of its own."""
        self._header.addWidget(w)
        return w


class FieldRow(QtWidgets.QWidget):
    """Label above control, with an optional hint line underneath that can turn into an error.

    The hint slot is always reserved once used, so validation text appearing does not make the
    whole panel jump.
    """

    def __init__(self, text: str, widget: QtWidgets.QWidget, hint: str = "", parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3)
        self.label = label(text, "field", self)
        v.addWidget(self.label)
        v.addWidget(widget)
        self.widget = widget
        self.hint = label(hint, "hint", self)
        self.hint.setVisible(bool(hint))
        v.addWidget(self.hint)
        self._base_hint = hint

    def set_hint(self, text: str, kind: str = "hint"):
        self.hint.setObjectName({"hint": "Hint", "warn": "HintWarn", "danger": "HintDanger"}[kind])
        self.hint.setText(text)
        self.hint.setVisible(bool(text))
        self.hint.style().unpolish(self.hint)
        self.hint.style().polish(self.hint)

    def reset_hint(self):
        self.set_hint(self._base_hint, "hint")


# ---------------------------------------------------------------- pending-aware controls
class _PendingMixin:
    """Shared pending bookkeeping.

    `local_truth` controls are settled by the GUI itself (a camera angle is true the moment the
    button is pressed, because nothing else decides it). Everything else stays pending until
    `ack()`; `pending_since` lets the status bar report how long a worker has been quiet without
    ever blocking the event loop.
    """
    pending_since: Optional[float] = None
    local_truth: bool = False

    def _set_pending(self, on: bool):
        self.setProperty("pending", "true" if on else "false")
        self.pending_since = QtCore.QDateTime.currentMSecsSinceEpoch() / 1000.0 if on else None
        self.style().unpolish(self)
        self.style().polish(self)

    def mark_pending(self):
        if not self.local_truth:
            self._set_pending(True)

    def ack(self):
        self._set_pending(False)


class PendingButton(QtWidgets.QPushButton, _PendingMixin):
    """A momentary command button (시작 / 리셋 / 정지 ...)."""

    def __init__(self, text: str, parent=None, role: str = "", local_truth: bool = False):
        super().__init__(text, parent)
        if role:
            self.setObjectName(role)
        self.local_truth = local_truth
        self._set_pending(False)


class PendingToggle(QtWidgets.QPushButton, _PendingMixin):
    """A checkable control whose checked state must match what the backend confirmed.

    Clicking does *not* flip the visual state for a backend-owned toggle: the click is turned back
    and the request is emitted, and only `settle(state)` (driven by the ack) moves it. That is the
    whole point -- pressing 일시정지 must never draw "일시정지" while the sim is still stepping.
    """
    requested = QtCore.pyqtSignal(bool)

    def __init__(self, text: str, parent=None, local_truth: bool = False, role: str = ""):
        super().__init__(text, parent)
        if role:
            self.setObjectName(role)
        self.setCheckable(True)
        self.local_truth = local_truth
        self._confirmed = False
        self._set_pending(False)
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self, checked: bool):
        if self.local_truth:
            self._confirmed = checked
            self.requested.emit(checked)
            return
        self.setChecked(self._confirmed)      # do not show the new state until it is confirmed
        self.mark_pending()
        self.requested.emit(checked)

    def settle(self, state: bool):
        self._confirmed = bool(state)
        self.setChecked(self._confirmed)
        self.ack()

    def is_confirmed(self) -> bool:
        return self._confirmed


class SegmentedButtons(QtWidgets.QWidget):
    """Mutually exclusive buttons in a row (the camera picker).

    Every mode is a visible, clickable, tab-reachable button. The keyboard shortcut is a
    convenience shown in the tooltip, never the only way in.
    """
    selected = QtCore.pyqtSignal(str)

    def __init__(self, options: Sequence[Tuple[str, str, str]], parent=None):
        """options: (key, label, tooltip)"""
        super().__init__(parent)
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(SP[0])
        self._buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        for key, text, tip in options:
            b = QtWidgets.QPushButton(text, self)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setAccessibleName(text)
            b.clicked.connect(lambda _=False, k=key: self._pick(k))
            self._group.addButton(b)
            self._buttons[key] = b
            h.addWidget(b)
        self._current: Optional[str] = None

    def _pick(self, key: str):
        self.set_current(key)
        self.selected.emit(key)

    def set_current(self, key: str):
        self._current = key
        for k, b in self._buttons.items():
            b.setChecked(k == key)

    def current(self) -> Optional[str]:
        return self._current

    def button(self, key: str) -> Optional[QtWidgets.QPushButton]:
        return self._buttons.get(key)


# ---------------------------------------------------------------- metrics
class MetricTile(QtWidgets.QWidget):
    """One number with its name and unit.

    `set_unknown()` is not cosmetic: a viewer that draws 0.00 m/s when it has never received a
    frame is lying about a measurement, and 0 is a perfectly plausible speed.
    """

    def __init__(self, name: str, unit: str = "", small: bool = False, tooltip: str = "", parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(1)
        self.name_label = label(name, "field", self)
        if tooltip:
            self.name_label.setToolTip(tooltip)
            self.setToolTip(tooltip)
        v.addWidget(self.name_label)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.value = QtWidgets.QLabel("—", self)
        self.value.setObjectName("MetricSmall" if small else "Metric")
        row.addWidget(self.value)
        self.unit = label(unit, "hint", self)
        self.unit.setObjectName("MetricUnit")
        row.addWidget(self.unit, 0, QtCore.Qt.AlignBottom)
        row.addStretch(1)
        v.addLayout(row)
        self._colour: Optional[str] = None

    def set_value(self, text: str, colour: Optional[str] = None):
        self.value.setText(text)
        if colour != self._colour:
            self._colour = colour
            self.value.setStyleSheet(f"color: {colour};" if colour else "")

    def set_unknown(self):
        self.set_value("—", C["text.2"])


class KeyValueList(QtWidgets.QWidget):
    """Compact label/value rows for facts that are text, not measurements."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(SP[2])
        self.grid.setVerticalSpacing(3)
        self.grid.setColumnStretch(1, 1)
        self._rows: Dict[str, QtWidgets.QLabel] = {}

    def set(self, key: str, value: str, mono: bool = True, tooltip: str = ""):
        if key not in self._rows:
            r = self.grid.rowCount()
            k = label(key, "field", self)
            k.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            self.grid.addWidget(k, r, 0)
            v = label("—", "mono" if mono else "body", self)
            v.setWordWrap(True)
            v.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self.grid.addWidget(v, r, 1)
            self._rows[key] = v
            if tooltip:
                k.setToolTip(tooltip)
                v.setToolTip(tooltip)
        self._rows[key].setText(value if value else "—")

    def value_label(self, key: str) -> Optional[QtWidgets.QLabel]:
        return self._rows.get(key)


class StateBadge(QtWidgets.QLabel):
    """The session state, in one place, with one colour per state.

    Every other part of the UI that shows state derives it from the same `SessionState`, so the
    badge and the buttons cannot disagree.
    """
    STYLES = {
        "IDLE": ("○", "대기", C["text.2"]),
        "PREPARING": ("◐", "준비 중", C["warn"]),
        "RUNNING": ("●", "주행 중", C["ok"]),
        "PAUSED": ("‖", "일시정지", C["accent"]),
        "STOPPING": ("◌", "정지 중", C["text.2"]),
        "FAILED": ("✕", "오류", C["danger"]),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Badge")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.set_state("IDLE")

    def set_state(self, state: str, detail: str = ""):
        glyph, text, colour = self.STYLES.get(state, self.STYLES["IDLE"])
        self.setText(f"{glyph}  {text}" + (f" · {detail}" if detail else ""))
        self.setStyleSheet(
            f"color: {colour}; background: {C['bg.card']}; border: 1px solid {colour};"
            f"border-radius: 12px; padding: 3px 12px; font-weight: 600; font-size: {theme.SIZE['label']}px;")


# ---------------------------------------------------------------- filtered pickers
class FilterList(QtWidgets.QWidget):
    """Search box + list, optionally grouped under non-selectable headers.

    Hundreds of maps are unusable as a flat combo box, which is what the old launcher had. Typing
    narrows; the group a name came from stays visible because that is the fact that decides what a
    result means.
    """
    activated = QtCore.pyqtSignal(str)          # selection changed to this key
    committed = QtCore.pyqtSignal(str)          # double-click / Enter

    def __init__(self, placeholder: str = "검색", parent=None, rows: int = 8):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(SP[0])
        self.search = QtWidgets.QLineEdit(self)
        self.search.setObjectName("SearchBox")
        self.search.setPlaceholderText(placeholder)
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName(placeholder)
        v.addWidget(self.search)
        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)                 # name, then a short muted meta column
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(10)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tree.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        # Middle elision, never the end: `ppo_race_0910` and `ppo_race_0910b` have to stay
        # distinguishable, and it is the tail that distinguishes them. The full name is on the
        # tooltip and in the selection card either way.
        self.tree.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tree.setMinimumHeight(rows * 25)
        v.addWidget(self.tree, 1)
        self.status = label("", "hint", self)
        v.addWidget(self.status)
        self._items: List[Tuple[str, str, str, str]] = []      # (group, key, title, subtitle)
        self._selected: Optional[str] = None
        self.search.textChanged.connect(lambda _: self._rebuild())
        self.tree.currentItemChanged.connect(self._on_current)
        self.tree.itemActivated.connect(self._on_activated)
        self.tree.itemDoubleClicked.connect(self._on_activated)

    # -- data
    def set_items(self, items: Iterable[Tuple[str, str, str, str]], keep_selection: bool = True):
        self._items = list(items)
        self._rebuild(keep_selection=keep_selection)

    def set_status(self, text: str, kind: str = "hint"):
        self.status.setObjectName({"hint": "Hint", "warn": "HintWarn", "danger": "HintDanger"}[kind])
        self.status.setText(text)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def selected(self) -> Optional[str]:
        return self._selected

    def select(self, key: str) -> bool:
        it = self._find(key)
        if it is None:
            return False
        self.tree.setCurrentItem(it)
        return True

    # -- internals
    def _find(self, key: str) -> Optional[QtWidgets.QTreeWidgetItem]:
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            item = it.value()
            if item.data(0, QtCore.Qt.UserRole) == key:
                return item
            it += 1
        return None

    def _rebuild(self, keep_selection: bool = True):
        want = self._selected if keep_selection else None
        needle = self.search.text().strip().lower()
        self.tree.blockSignals(True)
        self.tree.clear()
        groups: Dict[str, QtWidgets.QTreeWidgetItem] = {}
        shown = 0
        for group, key, title, subtitle in self._items:
            if needle and needle not in key.lower() and needle not in title.lower():
                continue
            parent = self.tree
            if group:
                if group not in groups:
                    g = QtWidgets.QTreeWidgetItem(self.tree, [group])
                    g.setFlags(QtCore.Qt.ItemIsEnabled)
                    f = g.font(0)
                    f.setBold(True)
                    f.setPointSizeF(max(8.0, f.pointSizeF() - 0.5))
                    g.setFont(0, f)
                    g.setForeground(0, QtGui.QColor(C["text.2"]))
                    g.setExpanded(True)
                    groups[group] = g
                parent = groups[group]
            item = QtWidgets.QTreeWidgetItem(parent, [title, subtitle])
            item.setData(0, QtCore.Qt.UserRole, key)
            tip = f"{key}\n{subtitle}" if subtitle else key
            item.setToolTip(0, tip)
            item.setToolTip(1, tip)
            if subtitle:
                item.setForeground(1, QtGui.QColor(C["text.2"]))
                f = item.font(1)
                f.setPointSizeF(max(7.0, f.pointSizeF() - 1.2))
                item.setFont(1, f)
            shown += 1
        self.tree.expandAll()
        self.tree.blockSignals(False)
        if want is not None and self.select(want):
            pass
        elif shown:
            first = self._first_selectable()
            if first is not None:
                self.tree.setCurrentItem(first)
        else:
            self._selected = None
        total = len({k for _, k, _, _ in self._items})
        self.set_status(f"{shown} / {total} 개 표시" if needle else f"{total} 개")

    def _first_selectable(self):
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            item = it.value()
            if item.data(0, QtCore.Qt.UserRole) is not None:
                return item
            it += 1
        return None

    def _on_current(self, cur, _prev):
        key = cur.data(0, QtCore.Qt.UserRole) if cur is not None else None
        if key is None or key == self._selected:
            return
        self._selected = key
        self.activated.emit(key)

    def _on_activated(self, item, _col=0):
        key = item.data(0, QtCore.Qt.UserRole)
        if key is not None:
            self.committed.emit(key)


class FlowLayout(QtWidgets.QLayout):
    """A layout that wraps its children onto a new line instead of squeezing them.

    The control bar holds a dozen buttons. In a fixed row, a narrow window shrinks every one of
    them below its text -- "일시정지" became "시정" and the camera buttons turned into unreadable
    slivers. Wrapping keeps every control at its natural size and readable at any width, which is
    the difference between a resizable window and one that merely does not crash when resized.
    """

    def __init__(self, parent=None, margin: int = 0, h_spacing: int = 6, v_spacing: int = 6):
        super().__init__(parent)
        self._items: List[QtWidgets.QLayoutItem] = []
        self._h = h_spacing
        self._v = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    @staticmethod
    def _expands(it) -> bool:
        """A widget marked `flowExpand` grows to fill what is left of its row.

        Without it a panel keeps its hint width even when the row is much wider, and the spare
        space goes to the margin instead of to the gauges that were being truncated for want of it.
        """
        w = it.widget()
        return bool(w is not None and w.property("flowExpand"))

    def _layout(self, rect, apply: bool) -> int:
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        rows: List[List[QtWidgets.QLayoutItem]] = [[]]
        x, line_h = eff.x(), 0
        placed = []
        for it in self._items:
            hint = it.sizeHint()
            if line_h > 0 and x + hint.width() > eff.right() + 1:
                rows.append([])
                x = eff.x()
                line_h = 0
            rows[-1].append(it)
            x += hint.width() + self._h
            line_h = max(line_h, hint.height())
        y = eff.y()
        for row in rows:
            if not row:
                continue
            used = sum(it.sizeHint().width() for it in row) + self._h * (len(row) - 1)
            spare = max(0, eff.width() - used)
            growers = [it for it in row if self._expands(it)]
            share = spare // len(growers) if growers else 0
            x = eff.x()
            row_h = max(it.sizeHint().height() for it in row)
            for it in row:
                w_ = it.sizeHint().width() + (share if self._expands(it) else 0)
                if apply:
                    it.setGeometry(QtCore.QRect(x, y, w_, row_h))
                placed.append((x, y, w_, row_h))
                x += w_ + self._h
            y += row_h + self._v
        return y - self._v - rect.y() + m.bottom()


class Sparkbar(QtWidgets.QWidget):
    """Frame-time distribution, drawn as it is.

    A single averaged fps number is exactly how a stuttering viewer looks fine on paper, so the
    recent frame times are drawn individually with the p95 marked. Nothing here is smoothed.
    """

    def __init__(self, parent=None, capacity: int = 120):
        super().__init__(parent)
        self.setMinimumHeight(34)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._values: List[float] = []
        self._capacity = capacity
        self._budget_ms = 16.7
        self.setToolTip("최근 프레임 시간. 가로선은 화면 주사 예산, 주황 선은 p95.")

    def push(self, ms: float):
        self._values.append(float(ms))
        if len(self._values) > self._capacity:
            del self._values[:len(self._values) - self._capacity]
        self.update()

    def clear(self):
        self._values.clear()
        self.update()

    def set_budget(self, ms: float):
        self._budget_ms = max(1.0, float(ms))

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QtGui.QColor(C["bg.window"]))
        if not self._values:
            p.setPen(QtGui.QColor(C["text.2"]))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "측정 없음")
            return
        vals = self._values
        top = max(self._budget_ms * 2.0, max(vals) * 1.05)
        n = len(vals)
        bw = max(1.0, w / max(self._capacity, n))
        budget_y = h - int(h * min(1.0, self._budget_ms / top))
        for i, v in enumerate(vals):
            x = int(w - (n - i) * bw)
            bh = max(1, int(h * min(1.0, v / top)))
            over = v > self._budget_ms * 1.5
            p.fillRect(QtCore.QRect(x, h - bh, max(1, int(bw) - 1), bh),
                       QtGui.QColor(C["danger"] if over else C["accent"]))
        p.setPen(QtGui.QPen(QtGui.QColor(C["line.strong"]), 1, QtCore.Qt.DashLine))
        p.drawLine(0, budget_y, w, budget_y)
        srt = sorted(vals)
        p95 = srt[min(len(srt) - 1, int(0.95 * len(srt)))]
        y95 = h - int(h * min(1.0, p95 / top))
        p.setPen(QtGui.QPen(QtGui.QColor(C["warn"]), 1))
        p.drawLine(0, y95, w, y95)
