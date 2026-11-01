"""The console's top-bar pickers and HUD tiles (2026-11-02).

The user, on the old run list: "정책 선택하는 칸 너무 불편해 ... 좌우로 늘리기도 안 되고 ... 정책 선택 박스 칸 누르면은
쪼르르르 그냥 떠서 거기서 고를 수 있게끔 해줘. ... 뭐라고 써있는지도 모르겠고". A 340 px panel list elided every name
to "…" and every note mid-word.

`RichPicker` is a top-bar button that shows what is chosen -- a small caption over the full value -- and opens a
`RichPopup`: a wide list of two-line rows (a title and a grey subtitle), grouped under small headers, filtered as you
type, keyboard-driven (arrows, Enter, Esc). Plain PyQt5: a Qt.Popup frame, a QListView over a QStandardItemModel, and
one delegate that paints the rows. `Tile` is a HUD card: a caption, a big fixed-width number and a unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from . import theme
from .theme import C

ROLE_KEY = QtCore.Qt.UserRole + 1
ROLE_SUB = QtCore.Qt.UserRole + 2
ROLE_HEADER = QtCore.Qt.UserRole + 3
ROLE_BADGE = QtCore.Qt.UserRole + 4
ROLE_CURRENT = QtCore.Qt.UserRole + 5


@dataclass
class PickItem:
    key: str
    title: str
    subtitle: str = ""
    group: str = ""
    badge: str = ""


def _font(px: int, weight: int = QtGui.QFont.Normal, mono: bool = False) -> QtGui.QFont:
    f = QtGui.QFont(theme.MONO_FONT if mono else theme.UI_FONT)
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


class _RowDelegate(QtWidgets.QStyledItemDelegate):
    """Paints a group header (small grey caps) or a row: title over subtitle, a badge on the right, the current
    choice marked by an accent bar. Hover and selection are rounded fills inside the row, not edge-to-edge bars."""
    ROW_H = 56
    ROW_H_1 = 38
    HEAD_H = 30

    def sizeHint(self, option, index):
        if index.data(ROLE_HEADER):
            h = self.HEAD_H
        else:
            h = self.ROW_H if index.data(ROLE_SUB) else self.ROW_H_1
        return QtCore.QSize(max(1, option.rect.width()), h)

    def paint(self, p: QtGui.QPainter, option, index):
        p.save()
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        r = option.rect
        if index.data(ROLE_HEADER):
            p.setFont(_font(11, QtGui.QFont.DemiBold))
            p.setPen(QtGui.QColor(C["text.2"]))
            p.drawText(r.adjusted(16, 6, -16, 0), int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter),
                       str(index.data(QtCore.Qt.DisplayRole) or ""))
            p.restore()
            return
        sel = bool(option.state & QtWidgets.QStyle.State_Selected)
        hov = bool(option.state & QtWidgets.QStyle.State_MouseOver)
        box = QtCore.QRectF(r.adjusted(6, 2, -6, -2))
        if sel or hov:
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(C["accent.soft"] if sel else C["bg.hover"]))
            p.drawRoundedRect(box, 8, 8)
        if index.data(ROLE_CURRENT):
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(C["accent"]))
            p.drawRoundedRect(QtCore.QRectF(box.left() + 4, box.top() + 10, 3, box.height() - 20), 1.5, 1.5)
        title = str(index.data(QtCore.Qt.DisplayRole) or "")
        sub = str(index.data(ROLE_SUB) or "")
        badge = str(index.data(ROLE_BADGE) or "")
        x0 = r.left() + 20
        right = r.right() - 16
        if badge:
            fb = _font(11, QtGui.QFont.DemiBold)
            bw = QtGui.QFontMetrics(fb).horizontalAdvance(badge) + 16
            br = QtCore.QRectF(right - bw, r.top() + (r.height() - 20) / 2, bw, 20)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(C["bg.raised"]))
            p.drawRoundedRect(br, 10, 10)
            p.setFont(fb)
            p.setPen(QtGui.QColor(C["text.1"]))
            p.drawText(br, int(QtCore.Qt.AlignCenter), badge)
            right = int(br.left()) - 10
        w = max(10, right - x0)
        ft = _font(14, QtGui.QFont.DemiBold)
        p.setFont(ft)
        p.setPen(QtGui.QColor(C["text.0"]))
        title_e = QtGui.QFontMetrics(ft).elidedText(title, QtCore.Qt.ElideRight, w)
        if sub:
            p.drawText(QtCore.QRect(x0, r.top() + 9, w, 20), int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter), title_e)
            fs = _font(12)
            p.setFont(fs)
            p.setPen(QtGui.QColor(C["text.2"]))
            p.drawText(QtCore.QRect(x0, r.top() + 30, w, 18), int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter),
                       QtGui.QFontMetrics(fs).elidedText(sub, QtCore.Qt.ElideRight, w))
        else:
            p.drawText(QtCore.QRect(x0, r.top(), w, r.height()), int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter),
                       title_e)
        p.restore()


class RichPopup(QtWidgets.QFrame):
    """The list a `RichPicker` opens. Typing filters (title, subtitle, key and group); arrows move; Enter chooses."""
    chosen = QtCore.pyqtSignal(str)
    MAX_H = 600

    def __init__(self, parent=None, searchable: bool = True, placeholder: str = "검색"):
        super().__init__(parent, QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.setObjectName("PickerPopup")
        self._items: List[PickItem] = []
        self._current: Optional[str] = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.search = QtWidgets.QLineEdit()
        self.search.setObjectName("PickerSearch")
        self.search.setPlaceholderText(placeholder)
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda t: self._rebuild(t))
        self.search.installEventFilter(self)
        self.search.setVisible(searchable)
        lay.addWidget(self.search)
        self.view = QtWidgets.QListView()
        self.view.setObjectName("PickerList")
        self.view.setItemDelegate(_RowDelegate(self.view))
        self.view.setMouseTracking(True)
        self.view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.view.setUniformItemSizes(False)
        self.model = QtGui.QStandardItemModel(self)
        self.view.setModel(self.model)
        self.view.clicked.connect(self._choose)
        self.view.activated.connect(self._choose)
        lay.addWidget(self.view, 1)

    # -- content
    def set_items(self, items: List[PickItem], current: Optional[str] = None):
        self._items = list(items)
        self._current = current
        self._rebuild(self.search.text())

    def _rebuild(self, query: str):
        q = (query or "").strip().casefold()
        words = q.split()
        self.model.clear()
        group = None
        first = cur = None
        for it in self._items:
            hay = f"{it.title} {it.subtitle} {it.key} {it.group} {it.badge}".casefold()
            if words and not all(w in hay for w in words):
                continue
            if it.group and it.group != group:
                h = QtGui.QStandardItem(it.group)
                h.setData(True, ROLE_HEADER)
                h.setFlags(QtCore.Qt.ItemIsEnabled)
                self.model.appendRow(h)
                group = it.group
            row = QtGui.QStandardItem(it.title)
            row.setData(it.key, ROLE_KEY)
            row.setData(it.subtitle, ROLE_SUB)
            row.setData(it.badge, ROLE_BADGE)
            row.setData(it.key == self._current, ROLE_CURRENT)
            row.setToolTip("\n".join(x for x in (it.title, it.subtitle, it.key) if x))
            row.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.model.appendRow(row)
            if first is None:
                first = row
            if it.key == self._current:
                cur = row
        pick = cur if (cur is not None and not q) else first
        if pick is not None:
            self.view.setCurrentIndex(pick.index())
            self.view.scrollTo(pick.index(), QtWidgets.QAbstractItemView.PositionAtCenter)

    def content_height(self) -> int:
        h = 0
        for i in range(self.model.rowCount()):
            it = self.model.item(i)
            h += _RowDelegate.HEAD_H if it.data(ROLE_HEADER) else (
                _RowDelegate.ROW_H if it.data(ROLE_SUB) else _RowDelegate.ROW_H_1)
        return h

    # -- behaviour
    def _choose(self, index: QtCore.QModelIndex):
        key = index.data(ROLE_KEY) if index.isValid() else None
        if key is None:
            return
        self.hide()
        self.chosen.emit(str(key))

    def _move(self, step: int):
        n = self.model.rowCount()
        if not n:
            return
        i = self.view.currentIndex().row()
        for _ in range(n):
            i = (i + step) % n
            if not self.model.item(i).data(ROLE_HEADER):
                self.view.setCurrentIndex(self.model.index(i, 0))
                return

    def eventFilter(self, obj, ev):
        if obj is self.search and ev.type() == QtCore.QEvent.KeyPress:
            k = ev.key()
            if k in (QtCore.Qt.Key_Down, QtCore.Qt.Key_Up):
                self._move(1 if k == QtCore.Qt.Key_Down else -1)
                return True
            if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self._choose(self.view.currentIndex())
                return True
            if k == QtCore.Qt.Key_Escape:
                self.hide()
                return True
        return super().eventFilter(obj, ev)

    def open_below(self, anchor: QtWidgets.QWidget, width: int):
        self.search.clear()
        self._rebuild("")
        search_h = self.search.sizeHint().height() + 6 if self.search.isVisible() else 0
        h = min(self.MAX_H, self.content_height() + search_h + 22)
        screen = (anchor.screen().availableGeometry() if hasattr(anchor, "screen") and anchor.screen()
                  else QtWidgets.QApplication.desktop().availableGeometry(anchor))
        pos = anchor.mapToGlobal(QtCore.QPoint(0, anchor.height() + 6))
        x = min(pos.x(), screen.right() - width - 8)
        self.setGeometry(max(screen.left() + 8, x), pos.y(), width, max(80, h))
        self.show()
        self.raise_()
        (self.search if self.search.isVisible() else self.view).setFocus(QtCore.Qt.PopupFocusReason)


class RichPicker(QtWidgets.QPushButton):
    """A top-bar button: a small caption over the chosen value, a chevron, and a `RichPopup` on click."""
    chosen = QtCore.pyqtSignal(str)

    def __init__(self, caption: str, placeholder: str = "선택하세요", popup_width: int = 640,
                 searchable: bool = True, search_hint: str = "검색", parent=None):
        super().__init__(parent)
        self.setObjectName("Picker")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.caption = caption
        self.placeholder = placeholder
        self.value = ""
        self.popup_width = int(popup_width)
        self._items: List[PickItem] = []
        self._current: Optional[str] = None
        self.popup = RichPopup(self, searchable=searchable, placeholder=search_hint)
        self.popup.chosen.connect(self._on_chosen)
        self.clicked.connect(self.open_popup)
        self.setMinimumHeight(46)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def sizeHint(self):
        return QtCore.QSize(260, 46)

    def set_items(self, items: List[PickItem]):
        self._items = list(items)
        if self._current is not None:
            self.set_current(self._current)

    def items(self) -> List[PickItem]:
        return list(self._items)

    def set_current(self, key: Optional[str], title: Optional[str] = None):
        self._current = key
        it = next((x for x in self._items if x.key == key), None) if key is not None else None
        self.value = title if title is not None else (it.title if it is not None else (key or ""))
        tip = "\n".join(x for x in ((it.title, it.subtitle, it.key) if it is not None else (self.value,)) if x)
        self.setToolTip(tip)
        self.update()

    def current(self) -> Optional[str]:
        return self._current

    def open_popup(self):
        self.popup.set_items(self._items, self._current)
        self.popup.open_below(self, max(self.width(), self.popup_width))

    def _on_chosen(self, key: str):
        self.set_current(key)
        self.chosen.emit(key)

    def paintEvent(self, ev):
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text = ""
        sp = QtWidgets.QStylePainter(self)
        sp.drawControl(QtWidgets.QStyle.CE_PushButton, opt)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        r = self.rect()
        x0, w = 14, r.width() - 14 - 30
        p.setFont(_font(11, QtGui.QFont.DemiBold))
        p.setPen(QtGui.QColor(C["text.2"]))
        p.drawText(QtCore.QRect(x0, 6, w, 15), int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter), self.caption)
        fv = _font(14, QtGui.QFont.DemiBold)
        p.setFont(fv)
        p.setPen(QtGui.QColor(C["text.0"] if (self.value and self.isEnabled()) else C["text.2"]))
        p.drawText(QtCore.QRect(x0, 21, w, 20), int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter),
                   QtGui.QFontMetrics(fv).elidedText(self.value or self.placeholder, QtCore.Qt.ElideRight, w))
        # chevron
        cx, cy = r.width() - 18, r.height() / 2 + 1
        pen = QtGui.QPen(QtGui.QColor(C["text.1"]), 1.6)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(cx - 4, cy - 2), QtCore.QPointF(cx, cy + 2),
                                        QtCore.QPointF(cx + 4, cy - 2)]))
        p.end()


class Tile(QtWidgets.QFrame):
    """A HUD card: caption, a fixed-width value (big or normal), a unit, an optional grey line under it."""

    def __init__(self, caption: str, unit: str = "", big: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("Tile")
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(2)
        self.caption = QtWidgets.QLabel(caption)
        self.caption.setObjectName("TileCaption")
        v.addWidget(self.caption)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.value = QtWidgets.QLabel("—")
        self.value.setObjectName("TileValueBig" if big else "TileValue")
        row.addWidget(self.value, 0, QtCore.Qt.AlignBottom)
        self.unit = QtWidgets.QLabel(unit)
        self.unit.setObjectName("TileUnit")
        row.addWidget(self.unit, 0, QtCore.Qt.AlignBottom)
        row.addStretch(1)
        v.addLayout(row)
        self.sub = QtWidgets.QLabel("")
        self.sub.setObjectName("TileSub")
        self.sub.setWordWrap(True)
        self.sub.setVisible(False)
        v.addWidget(self.sub)

    def set_value(self, text: str, colour: Optional[str] = None):
        self.value.setText(text)
        self.value.setStyleSheet(f"color: {colour};" if colour else "")

    def set_sub(self, text: str):
        self.sub.setText(text)
        self.sub.setVisible(bool(text))
