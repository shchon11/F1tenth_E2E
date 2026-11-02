"""Design tokens and the Qt stylesheet built from them.

One place decides colour, type and spacing, so a value has the same meaning everywhere. The 3D
scene already spends colour on facts -- cyan is the watched car, amber a rival in the same race, and a
contact is coloured by what was touched: orange a wall or duct hose, yellow a crate, red another car --
and the panels reuse exactly those, so a colour never means two things in one window.

No Qt import happens at module import time beyond QtGui/QtCore, which are cheap; nothing here
touches torch or a GPU.
"""
from __future__ import annotations

# ---------------------------------------------------------------- colour
# Graphite neutrals, the way Omniverse / Isaac Sim and Unreal's editor chrome are built: no hue in
# the panels, so the only colour on screen is the colour the scene spends on facts. One calm blue
# accent for selection and focus; NVIDIA-style green only for "running" and the start action.
C = {
    # Four surfaces, far enough apart to be told apart at a glance: the window behind everything,
    # the panel columns, a card on a panel, and a control raised off a card. They used to span 22
    # points of grey in total, so a card was invisible against the panel it sat on and the whole
    # window read as one flat sheet with text scattered over it.
    # 11-02 (the user: "UI 너무 쓰레기야 ... 쌈뽕하게 직관적으로"): a blue-black ramp in the manner of Linear / Raycast /
    # VS Code dark -- four surfaces a few points apart with a faint cool cast, hairline borders instead of boxes,
    # one blue accent, green only for the start action and "running".
    "bg.window": "#0a0c10",
    "bg.panel": "#0f1218",
    "bg.card": "#151922",
    "bg.raised": "#1c212c",
    "bg.hover": "#252b38",
    "bg.sunken": "#080a0e",          # inputs and lists: the well a value sits in
    "bg.viewport": "#0b0d11",
    "line": "#232834",
    "line.strong": "#323949",
    "text.0": "#eef1f6",
    "text.1": "#a7b0c0",
    "text.2": "#8a909d",             # 4.6:1 on every surface (was #6b7485, under 4:1)
    "accent": "#4c8dff",
    "accent.soft": "#172744",        # a selected row / segment
    "accent.deep": "#1d3f7a",
    "accent.fill": "#2f6feb",        # a filled accent button: white text at 4.6:1
    "accent.fill.hover": "#3d7ef5",
    "best": "#b98cff",               # the session's best lap: F1's purple
    "primary": "#1fb866",
    "primary.hover": "#25cc72",
    "primary.line": "#1fb866",
    "ego": "#59e6ff",
    "ok": "#34d399",
    "warn": "#f5a524",
    "danger": "#f04848",
    "rival": "#f2a044",
    # -- contacts, by what was touched (`console.contacts`): the marker at the contact point, the
    # flash on the car and the kind's name on the panel. Saturated and brighter than `rival`, and the
    # car's flash is over-bright and fades, so a car lit by a wall does not read as a rival.
    "contact.wall": "#ff7a1a",
    "contact.prop": "#ffd426",
    "contact.car": "#ff3344",
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
#: Bundled (f1sim/assets/fonts, SIL OFL; registered by `app.register_fonts`): Pretendard for the Korean UI, JetBrains
#: Mono for every number that changes while you watch it (fixed-width digits do not jitter).
UI_FONT = "Pretendard"
MONO_FONT = "JetBrains Mono"
FONT_FALLBACK = ["Pretendard", "Noto Sans CJK KR", "NanumGothic", "Noto Sans", "DejaVu Sans"]

#: Type scale. `section` is small on purpose -- a heading reads as a heading through weight,
#: letter-spacing and colour, not through size, and making it big would compete with the values
#: underneath it, which are the thing being read.
SIZE = {"title": 15, "section": 11, "body": 13, "label": 12, "hint": 12,
        "metric": 27, "metric.sm": 15, "hud": 40, "hud.sm": 22}

# ---------------------------------------------------------------- spacing
SP = (4, 8, 12, 16, 24)
RADIUS_CTL = 8
RADIUS_CARD = 12
MIN_CTL_H = 30
MAIN_BTN_H = 40


def _asset(name: str) -> str:
    """A branding file as a URL a stylesheet can use, or "" when it is not installed.

    Resolved through this module's own location, so it is right in a checkout, a wheel and an
    AppImage alike. Qt wants forward slashes in a stylesheet url() on every platform.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))                     # .../f1sim/viewer/console
    p = os.path.join(os.path.dirname(os.path.dirname(here)), "assets", "branding", name)
    return p.replace(os.sep, "/") if os.path.exists(p) else ""


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
    # A filled square with nothing in it reads as a colour swatch, not as "on". The mark is an
    # image because a stylesheet cannot put a glyph in an indicator; without the file the square
    # is still the checked state, just less obviously so.
    _check = _asset("check-15.png")
    _CHECK_IMAGE = f"image: url({_check});" if _check else ""
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
/* A heading reads as a heading through weight, spacing and brightness. At text.2 it was dimmer
   than the hints underneath it, so the panel had no visible structure at all. */
#SectionLabel {{
    font-size: {SIZE['section']}px; font-weight: 700; color: #dcdce4;
    letter-spacing: 1.3px; padding-bottom: 2px;
}}
#FieldLabel {{ font-size: {SIZE['label']}px; font-weight: 600; color: {c['text.1']}; }}
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
    background: {c['bg.raised']}; color: {c['text.0']};
    border: 1px solid {c['line.strong']}; border-radius: {RADIUS_CTL}px;
    padding: 6px 13px; min-height: {MIN_CTL_H - 14}px; font-weight: 500;
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
    background: {c['bg.sunken']}; color: {c['text.0']};
    border: 1px solid {c['line']}; border-radius: {RADIUS_CTL}px;
    padding: 6px 9px; min-height: {MIN_CTL_H - 12}px;
    selection-background-color: {c['accent.deep']};
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {c['line.strong']};
}}
/* Focus is two pixels of accent, not one: a one-pixel border change on a dark input is not
   visible from a normal viewing distance, and "where am I typing" is the one thing a keyboard
   user needs the window to answer. */
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 2px solid {c['accent']}; padding: 5px 8px;
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {c['text.2']}; }}
#SearchBox {{ font-family: "{MONO_FONT}", monospace; }}
#CardStrong {{
    background: {c['bg.card']}; border: 1px solid {c['line.strong']};
    border-left: 3px solid {c['accent']}; border-radius: {RADIUS_CARD}px;
}}
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
    border: 1px solid {c['line.strong']}; background: {c['bg.sunken']};
}}
QCheckBox::indicator:checked {{
    background: {c['accent']}; border-color: {c['accent']};
    {_CHECK_IMAGE}
}}
QCheckBox::indicator:hover:!checked {{ border-color: {c['accent']}; }}
QCheckBox:disabled {{ color: {c['text.2']}; }}
QCheckBox:focus {{ outline: none; }}

/* ---------------------------------------------------------------- lists */
QListWidget, QTreeWidget {{
    background: {c['bg.sunken']}; border: 1px solid {c['line']};
    border-radius: {RADIUS_CTL}px; outline: none;
}}
/* right padding leaves room for the scrollbar so a long name is never printed under it */
QListWidget::item, QTreeWidget::item {{ padding: 4px 14px 4px 6px; border-radius: 4px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {c['accent.deep']}; color: #ffffff;
}}
QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected {{ background: {c['bg.raised']}; }}
QTreeWidget::branch {{ background: transparent; }}
/* Tabs. Without a rule the platform style paints an unselected tab's label in a colour meant for a
   light palette, and on this graphite it is invisible -- which is how a three-tab picker reads as
   one tab and two blank slabs. */
QTabWidget::pane {{ border: 1px solid {c['line']}; border-radius: {RADIUS_CTL}px; top: -1px; }}
QTabBar::tab {{
    background: {c['bg.window']}; color: {c['text.1']};
    border: 1px solid {c['line']}; border-bottom: none;
    border-top-left-radius: {RADIUS_CTL}px; border-top-right-radius: {RADIUS_CTL}px;
    padding: 5px 12px; margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {c['bg.card']}; color: {c['text.0']}; border-color: {c['line.strong']}; }}
QTabBar::tab:hover:!selected {{ background: {c['bg.raised']}; color: {c['text.0']}; }}
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
/* A vertical rule for grouping a row of buttons (transport | camera | capture). Without one the
   centre bar is twelve identical grey rectangles and nothing says which belong together. */
#BarSep {{ background: {c['line']}; max-width: 1px; min-width: 1px; }}
QFrame[frameShape="4"] {{ color: {c['line']}; max-height: 1px; }}

/* ---------------------------------------------------------------- 11-02 layout: top bar, pickers, HUD */
#TopBar {{ background: {c['bg.panel']}; border-bottom: 1px solid {c['line']}; }}
#Brand {{ font-size: 15px; font-weight: 700; color: {c['text.0']}; letter-spacing: 0.3px; }}
#BrandMark {{ color: {c['accent']}; font-size: 15px; font-weight: 800; }}
QPushButton#Picker {{
    background: {c['bg.raised']}; border: 1px solid {c['line']}; border-radius: 10px;
    padding: 0px; text-align: left;
}}
QPushButton#Picker:hover:!disabled {{ background: {c['bg.hover']}; border-color: {c['line.strong']}; }}
QPushButton#Picker:pressed:!disabled {{ background: {c['bg.hover']}; border-color: {c['accent']}; }}
QPushButton#Picker:focus {{ border: 1px solid {c['accent']}; }}
QPushButton#Picker:disabled {{ background: {c['bg.panel']}; border-color: {c['line']}; }}
#PickerPopup {{ background: {c['bg.card']}; border: 1px solid {c['line.strong']}; border-radius: 12px; }}
QLineEdit#PickerSearch {{
    background: {c['bg.sunken']}; border: 1px solid {c['line']}; border-radius: 8px;
    padding: 8px 12px; font-size: 13px;
}}
QLineEdit#PickerSearch:focus {{ border: 1px solid {c['accent']}; padding: 8px 12px; }}
QListView#PickerList {{ background: transparent; border: none; outline: none; }}
QPushButton#StartButton {{
    background: {c['accent.fill']}; border: 1px solid {c['accent.fill']}; color: #ffffff;
    font-size: 14px; font-weight: 700; border-radius: 10px; padding: 0px 22px; min-height: 26px;
}}
QPushButton#StartButton:hover:!disabled {{ background: {c['accent.fill.hover']}; border-color: {c['accent.fill.hover']}; }}
QPushButton#StartButton:disabled {{ background: {c['bg.raised']}; border-color: {c['line']}; color: {c['text.2']}; }}
/* Neutral, not red: in this window red means a car contact. */
QPushButton#StopButton {{
    background: {c['bg.raised']}; border: 1px solid {c['line.strong']}; color: {c['text.0']};
    font-size: 14px; font-weight: 700; border-radius: 10px; padding: 0px 18px; min-height: 26px;
}}
QPushButton#StopButton:hover:!disabled {{ background: {c['bg.hover']}; border-color: {c['text.2']}; }}
QPushButton#StopButton:disabled {{ color: {c['text.2']}; border-color: {c['line']}; }}
QPushButton#IconButton {{
    background: {c['bg.raised']}; border: 1px solid {c['line']}; border-radius: 10px;
    padding: 0px 12px; font-size: 13px; font-weight: 600; color: {c['text.1']}; min-height: 26px;
}}
QPushButton#IconButton:hover:!disabled {{ background: {c['bg.hover']}; color: {c['text.0']}; }}
QPushButton#IconButton:checked {{ background: {c['accent.soft']}; border-color: {c['accent']}; color: {c['text.0']}; }}
QPushButton#IconButton:disabled {{ color: {c['text.2']}; background: {c['bg.panel']}; }}
#Hud {{ background: {c['bg.panel']}; border-left: 1px solid {c['line']}; }}
#Tile {{ background: {c['bg.card']}; border: 1px solid {c['line']}; border-radius: 12px; }}
#TileCaption {{ font-size: 11px; font-weight: 600; color: {c['text.2']}; letter-spacing: 0.8px; }}
#TileValue {{ font-family: "{MONO_FONT}", monospace; font-size: {SIZE['hud.sm']}px; font-weight: 600; color: {c['text.0']}; }}
#TileValueBig {{ font-family: "{MONO_FONT}", monospace; font-size: {SIZE['hud']}px; font-weight: 600; color: {c['text.0']}; }}
#TileUnit {{ font-size: 12px; color: {c['text.2']}; }}
#TileSub {{ font-family: "{MONO_FONT}", monospace; font-size: 12px; color: {c['text.1']}; }}
/* 11-03: the condition bar under the top bar -- the obstacle seed / layout and the opponent of the chosen condition */
#CondBar {{ background: {c['bg.panel']}; border-bottom: 1px solid {c['line']}; }}
#CondBar QLabel#CondLabel {{ font-size: 11px; font-weight: 700; color: {c['text.2']}; letter-spacing: 0.8px; }}
#CondBar QLabel#CondChip {{ font-size: 12px; color: {c['text.1']}; }}
#Viewport {{ border: none; }}
#Chip {{ font-family: "{MONO_FONT}", monospace; font-size: 12px; color: {c['text.1']}; }}
#StatusBar QLabel {{ font-size: 12px; color: {c['text.2']}; }}
"""
