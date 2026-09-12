"""Design tokens and the Qt stylesheet built from them.

One place decides colour, type and spacing, so a value has the same meaning everywhere. The 3D
scene already spends colour on facts -- cyan is the watched car, amber a rival in the same race,
red a collision -- and the panels reuse exactly those, so a colour never means two things in one
window.

No Qt import happens at module import time beyond QtGui/QtCore, which are cheap; nothing here
touches torch or a GPU.
"""
from __future__ import annotations

# ---------------------------------------------------------------- colour
# Graphite neutrals, the way Omniverse / Isaac Sim and Unreal's editor chrome are built: no hue in
# the panels, so the only colour on screen is the colour the scene spends on facts. One calm blue
# accent for selection and focus; NVIDIA-style green only for "running" and the start action.
C = {
    "bg.window": "#161616",
    "bg.panel": "#1c1c1c",
    "bg.card": "#222222",
    "bg.raised": "#2c2c2c",
    "bg.viewport": "#121316",
    "line": "#303030",
    "line.strong": "#454545",
    "text.0": "#e6e6e6",
    "text.1": "#a9a9a9",
    "text.2": "#747474",
    "accent": "#5aa9ff",
    "accent.deep": "#22415f",
    "primary": "#3d6f0b",
    "primary.hover": "#4c8a10",
    "primary.line": "#76b900",
    "ego": "#59e6ff",
    "ok": "#76b900",
    "warn": "#f0a030",
    "danger": "#ff5c5c",
    "rival": "#f2a044",
    # -- instrument shading. The gauges are drawn objects, not charts, so they get their own few
    # tones: a dial well that is slightly lighter at the top, a rim with a top-lit gradient, and a
    # needle that stays white against both.
    "bg.dial.hi": "#2a2a2a",
    "bg.dial.lo": "#1a1a1a",
    "needle": "#f2f2f2",
    "rim.hi": "#5a5a5a",
    "rim.lo": "#333333",
    "rim.edge": "#171717",
    "rim.grip": "#767676",
    "spoke": "#4d4d4d",
}

# ---------------------------------------------------------------- type
UI_FONT = "Noto Sans CJK KR"
MONO_FONT = "DejaVu Sans Mono"
FONT_FALLBACK = ["Noto Sans CJK KR", "NanumGothic", "Noto Sans", "DejaVu Sans"]

SIZE = {"title": 14, "section": 11, "body": 12, "label": 11, "hint": 11, "metric": 24, "metric.sm": 14}

# ---------------------------------------------------------------- spacing
SP = (4, 8, 12, 16, 24)
RADIUS_CTL = 3
RADIUS_CARD = 4
MIN_CTL_H = 26
MAIN_BTN_H = 32


def clear_color():
    """GL clear colour as floats, from the same token the panels are drawn against."""
    h = C["bg.viewport"].lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def qss() -> str:
    """The application stylesheet. Everything a widget needs to look right without per-widget code.

    Object names used as hooks:
      #Header #HeaderTitle #Panel #Card #SectionLabel #FieldLabel #Hint #Metric #MetricSmall
      #PrimaryButton #DangerButton #GhostButton #StatusBar #Badge #SearchBox #Viewport
    A button carries dynamic property `pending=true` while a command it sent is unacknowledged --
    that state is drawn differently on purpose: the UI must never claim a change the worker has
    not confirmed.
    """
    c = C
    return f"""
/* Type and colour are inherited by everything; a *background* is not.
   Painting every QWidget with the window colour draws a dark rectangle behind every label sitting
   inside a lighter card, which is what the first screenshot showed. Containers name their own
   background below; leaf widgets stay transparent so they take the colour of whatever they are on. */
QWidget {{
    color: {c['text.0']};
    font-family: "{UI_FONT}", "NanumGothic", "Noto Sans", sans-serif;
    font-size: {SIZE['body']}px;
}}
QMainWindow, QDialog {{ background: {c['bg.window']}; }}
QLabel, QCheckBox, QSplitter, QGroupBox {{ background: transparent; border: none; }}
QToolTip {{
    background: {c['bg.card']}; color: {c['text.0']};
    border: 1px solid {c['line.strong']}; padding: 6px 8px;
}}

/* ---------------------------------------------------------------- structure */
#Header {{ background: {c['bg.panel']}; border-bottom: 1px solid {c['line']}; }}
#HeaderTitle {{ font-size: {SIZE['title']}px; font-weight: 600; color: {c['text.0']}; }}
#HeaderSub {{ font-family: "{MONO_FONT}", monospace; font-size: {SIZE['hint']}px; color: {c['text.2']}; }}
#Panel {{ background: {c['bg.panel']}; }}
#PanelScroll {{ background: {c['bg.panel']}; border: none; }}
#PanelScroll > QWidget > QWidget {{ background: {c['bg.panel']}; }}
#Centre {{ background: {c['bg.window']}; }}
#Card {{
    background: {c['bg.card']}; border: 1px solid {c['line']};
    border-radius: {RADIUS_CARD}px;
}}
#SectionLabel {{
    font-size: {SIZE['section']}px; font-weight: 600; color: {c['text.2']};
    letter-spacing: 1.1px;
}}
#FieldLabel {{ font-size: {SIZE['label']}px; font-weight: 500; color: {c['text.1']}; }}
#Hint {{ font-size: {SIZE['hint']}px; color: {c['text.2']}; }}
#HintWarn {{ font-size: {SIZE['hint']}px; color: {c['warn']}; }}
#HintDanger {{ font-size: {SIZE['hint']}px; color: {c['danger']}; }}
#Mono {{ font-family: "{MONO_FONT}", monospace; font-size: {SIZE['label']}px; color: {c['text.1']}; }}
#Metric {{
    font-family: "{MONO_FONT}", monospace; font-size: {SIZE['metric']}px;
    font-weight: 600; color: {c['text.0']};
}}
#MetricSmall {{
    font-family: "{MONO_FONT}", monospace; font-size: {SIZE['metric.sm']}px;
    font-weight: 600; color: {c['text.0']};
}}
#MetricUnit {{ font-size: {SIZE['hint']}px; color: {c['text.2']}; }}

/* ---------------------------------------------------------------- buttons */
QPushButton {{
    background: {c['bg.card']}; color: {c['text.0']};
    border: 1px solid {c['line.strong']}; border-radius: {RADIUS_CTL}px;
    padding: 6px 12px; min-height: {MIN_CTL_H - 14}px;
}}
QPushButton:hover:!disabled {{ background: {c['bg.raised']}; border-color: {c['accent']}; }}
QPushButton:pressed:!disabled {{ background: {c['accent.deep']}; }}
QPushButton:disabled {{ color: {c['text.2']}; border-color: {c['line']}; background: {c['bg.panel']}; }}
QPushButton:checked {{
    background: {c['accent.deep']}; border-color: {c['accent']}; color: {c['text.0']};
}}
QPushButton:focus {{ outline: none; border: 1px solid {c['accent']}; }}
QPushButton[pending="true"] {{
    border: 1px dashed {c['warn']}; color: {c['warn']}; background: {c['bg.panel']};
}}
#PrimaryButton {{
    background: {c['primary']}; border: 1px solid {c['primary.line']};
    color: #ffffff; font-weight: 600; min-height: {MAIN_BTN_H - 14}px;
}}
#PrimaryButton:hover:!disabled {{ background: {c['primary.hover']}; }}
#PrimaryButton:disabled {{ background: {c['bg.panel']}; border-color: {c['line']}; color: {c['text.2']}; }}
#DangerButton {{ border-color: #5a2a2a; color: {c['danger']}; }}
#DangerButton:hover:!disabled {{ background: #3a1c1c; border-color: {c['danger']}; }}
#GhostButton {{ background: transparent; border: 1px solid {c['line']}; color: {c['text.1']}; }}
#GhostButton:hover:!disabled {{ background: {c['bg.raised']}; color: {c['text.0']}; }}

/* ---------------------------------------------------------------- inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {c['bg.window']}; color: {c['text.0']};
    border: 1px solid {c['line.strong']}; border-radius: {RADIUS_CTL}px;
    padding: 5px 8px; min-height: {MIN_CTL_H - 12}px;
    selection-background-color: {c['accent.deep']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {c['accent']}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {c['text.2']}; }}
#SearchBox {{ font-family: "{MONO_FONT}", monospace; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {c['bg.card']}; color: {c['text.0']};
    border: 1px solid {c['line.strong']}; selection-background-color: {c['accent.deep']};
    outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {c['bg.card']}; border-left: 1px solid {c['line']}; width: 16px;
}}
QCheckBox {{ spacing: 8px; color: {c['text.0']}; padding: 3px 0; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid {c['line.strong']}; background: {c['bg.window']};
}}
QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; }}
QCheckBox:disabled {{ color: {c['text.2']}; }}
QCheckBox:focus {{ outline: none; }}

/* ---------------------------------------------------------------- lists */
QListWidget, QTreeWidget {{
    background: {c['bg.window']}; border: 1px solid {c['line']};
    border-radius: {RADIUS_CTL}px; outline: none;
}}
/* right padding leaves room for the scrollbar so a long name is never printed under it */
QListWidget::item, QTreeWidget::item {{ padding: 4px 14px 4px 6px; border-radius: 4px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {c['accent.deep']}; color: #ffffff;
}}
QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected {{ background: {c['bg.raised']}; }}
QTreeWidget::branch {{ background: transparent; }}
QHeaderView::section {{
    background: {c['bg.panel']}; color: {c['text.1']};
    border: none; border-bottom: 1px solid {c['line']}; padding: 4px 6px;
}}

/* ---------------------------------------------------------------- misc */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {c['line.strong']}; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {c['text.2']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {c['line.strong']}; border-radius: 5px; min-width: 28px; }}
QSplitter::handle {{ background: {c['line']}; }}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:hover {{ background: {c['accent']}; }}
QProgressBar {{
    background: {c['bg.window']}; border: 1px solid {c['line']};
    border-radius: 4px; height: 6px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}
#StatusBar {{ background: {c['bg.panel']}; border-top: 1px solid {c['line']}; }}
QFrame[frameShape="4"] {{ color: {c['line']}; max-height: 1px; }}
"""
