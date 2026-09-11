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
    app.setApplicationName("f1sim 주행 콘솔")
    app.setOrganizationName("f1sim")
    apply_theme(app)
    return app


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
