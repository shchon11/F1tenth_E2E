"""Process bootstrap for the console.

The surface format has to be chosen before the first QOpenGLWidget exists -- Qt reads the default
format when it creates the context, and a widget built before `setDefaultFormat` gets whatever the
platform hands out (often a compatibility profile, where the 3.3-core shaders fail to compile).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from . import theme

#: Matches what `gl_scene` asks for: `#version 330` shaders in a core profile.
GL_MAJOR, GL_MINOR = 3, 3


def configure_surface_format(samples: int = 4, vsync: bool = True) -> QtGui.QSurfaceFormat:
    """Must run before any QOpenGLWidget is constructed (and before QApplication, in practice)."""
    fmt = QtGui.QSurfaceFormat()
    fmt.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
    fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    fmt.setVersion(GL_MAJOR, GL_MINOR)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(0)
    fmt.setSamples(int(samples))
    fmt.setSwapBehavior(QtGui.QSurfaceFormat.DoubleBuffer)
    # One pacer. The viewport schedules its next frame from `frameSwapped`, so with vsync on the
    # swap itself sets the rate; its fps cap only matters when the driver ignores vsync.
    fmt.setSwapInterval(1 if vsync else 0)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    return fmt


def create_app(argv: Optional[list] = None, samples: int = 4, vsync: bool = True) -> QtWidgets.QApplication:
    configure_surface_format(samples=samples, vsync=vsync)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    # what we asked the platform for, so a report can say "asked 4x, got N" rather than assuming
    app.setProperty("f1sim_msaa_requested", int(samples))
    # ASCII, on purpose. Qt publishes the window title twice on X11: `_NET_WM_NAME` in UTF-8 and
    # the legacy `WM_NAME` in Latin-1. Plenty of panels, docks and alt-tab switchers still read the
    # legacy one, and a Korean name comes out of it as "f1sim ì£¼í–‰ ì½˜ì†”". The Korean name is not
    # lost: it is on the header inside the window, and in the desktop entry's `Name[ko]`, which is
    # UTF-8 by specification and handled correctly.
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName("f1sim")
    # Lets a dock match the running window to its .desktop file, and take the icon and the
    # localised name from there. Without it a dock falls back on WM_CLASS and the legacy title.
    if hasattr(app, "setDesktopFileName"):
        app.setDesktopFileName(DESKTOP_ID)
    icon = app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    apply_theme(app)
    return app


#: What the window manager, the taskbar and the desktop entry call this program. See
#: `create_app` for why it is not the Korean name.
APP_NAME = "f1sim Console"

#: The desktop entry's basename, without `.desktop`. Reverse-DNS so it cannot collide.
DESKTOP_ID = "io.f1sim.Console"


def icon_dir() -> str:
    """Where the icon files live, resolved through the imported package rather than a repo path,
    so it works the same from a checkout, a wheel and an AppImage."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "assets", "branding")


def app_icon() -> QtGui.QIcon:
    """The application icon at every size that ships, or a null icon if none is installed.

    Every size is added rather than only the largest: Qt downscales a 512 px tile into a 16 px
    tray slot as mush, and the small PNGs are drawn for the size they are.
    """
    icon = QtGui.QIcon()
    d = icon_dir()
    for n in (16, 24, 32, 48, 64, 128, 256, 512):
        p = os.path.join(d, f"f1sim-{n}.png")
        if os.path.exists(p):
            icon.addFile(p, QtCore.QSize(n, n))
    if icon.isNull():
        svg = os.path.join(d, "f1sim.svg")
        if os.path.exists(svg):
            icon.addFile(svg)
    return icon


def apply_theme(app: QtWidgets.QApplication) -> None:
    font = QtGui.QFont()
    # Korean text needs a font that has it; DejaVu (the scene's HUD font) does not. Ask for the
    # families we checked are installed and let fontconfig fall back through the rest.
    for family in theme.FONT_FALLBACK:
        if family in QtGui.QFontDatabase().families():
            font.setFamily(family)
            break
    font.setPointSizeF(10.0)
    font.setHintingPreference(QtGui.QFont.PreferFullHinting)
    app.setFont(font)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    pal = app.palette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(theme.C["bg.window"]))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(theme.C["bg.window"]))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor(theme.C["text.0"]))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(theme.C["text.0"]))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(theme.C["accent.deep"]))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(theme.C["bg.card"]))
    pal.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(theme.C["text.0"]))
    app.setPalette(pal)
    app.setStyleSheet(theme.qss())


def software_gl_requested() -> bool:
    """True when the environment has asked for llvmpipe (the Xvfb test fixture does)."""
    return os.environ.get("LIBGL_ALWAYS_SOFTWARE", "") not in ("", "0")
