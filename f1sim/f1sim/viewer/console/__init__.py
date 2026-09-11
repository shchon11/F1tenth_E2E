"""The f1sim driving console: one window to pick a policy and a map, watch it drive, and judge
whether what is on screen can be believed.

    python -m f1sim.viewer.console          # or: python -m f1sim.learn.watch  (no arguments)

Replaces the old three-piece interactive path -- a modal Tk launcher, a second always-on-top Tk
map panel in its own process talking through a file in ~/.cache, and a GLFW window whose only
controls were undocumented single keys. The headless paths (`--record`, `--frames`, `--bench`,
`--episodes`, `--highlights`) are untouched and still go through `f1sim.learn.watch`.

Importing this module does not import torch; the simulation lives in a worker process.

Runtime prerequisite
--------------------
The console needs **PyQt5**, which is installed in this environment but is *not* in the project's
`[viewer]` extra (that lists `moderngl`, `glfw`, `trimesh`). Nothing here installs it or changes the
dependency sets -- that is a packaging decision, not this module's to make. What it does do is fail
in a way someone can act on: `require_qt()` says what is missing, how the console is normally
started, and that `--legacy-launcher` still opens the old Tk picker without Qt.

Importing `f1sim.learn.watch` for a headless run does not import Qt: the console is only imported
inside the no-argument branch, so `--record` / `--frames` / `--bench` work on a machine with no Qt
at all.
"""
from __future__ import annotations

import sys
from typing import List, Optional

__all__ = ["launch", "main", "require_qt", "MissingQtError"]

#: Shown when PyQt5 is absent. Names the alternative rather than only the problem.
_MISSING_QT = """f1sim 주행 콘솔을 열려면 PyQt5가 필요합니다.

PyQt5는 이 프로젝트의 [viewer] extra(moderngl, glfw, trimesh)에 들어 있지 않습니다.
이 프로그램은 패키지를 설치하지 않습니다 — 아래 중 하나를 선택하세요.

  1) 콘솔을 쓰려면 실행 환경에 PyQt5를 마련한 뒤 다시 실행하세요.
       python -m f1sim.learn.watch
       python -m f1sim.viewer.console

  2) Qt 없이 예전 Tk 런처로 실행:
       python -m f1sim.learn.watch --legacy-launcher

  3) Qt 없이 화면 없는 기록/벤치 (원래도 Qt를 쓰지 않습니다):
       python -m f1sim.learn.watch --record out.mp4 --seconds 20
       python -m f1sim.learn.watch --frames dir/ --seconds 5
       python -m f1sim.learn.watch --bench 30
"""


class MissingQtError(RuntimeError):
    """PyQt5 is not importable. Carries the actionable message, not just the module name."""


def require_qt():
    """Raise `MissingQtError` with something useful if PyQt5 is missing.

    A bare `ModuleNotFoundError: No module named 'PyQt5'` tells someone what broke and nothing
    about what to do -- least of all that the old launcher and every headless path still work
    without it.
    """
    try:
        from PyQt5 import QtCore, QtWidgets                      # noqa: F401
    except ImportError as exc:
        raise MissingQtError(_MISSING_QT) from exc


def launch(argv: Optional[List[str]] = None) -> int:
    """Open the console. Returns the Qt exit code, or 2 if Qt is not available."""
    try:
        require_qt()
    except MissingQtError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from PyQt5 import QtCore, QtWidgets

    from .app import create_app
    from .catalog import list_runs
    from .session import SessionController
    from .window import ConsoleWindow

    app = create_app(argv if argv is not None else sys.argv)
    window = ConsoleWindow()
    controller = SessionController(window)

    # Runs are a directory listing, so the picker is filled before the window is even shown; the
    # map catalog needs f1sim.maps and arrives from the worker a moment later.
    window.set_runs(list_runs())
    window.set_maps(controller._maps)
    window.show()

    # Start the worker after the first paint. The window is interactive from the first frame and
    # the several seconds of `import torch` happen somewhere that cannot block it.
    QtCore.QTimer.singleShot(0, controller.start_worker)

    # The controller wires window closure itself. Keep a last resort for app-level exits.
    app.aboutToQuit.connect(controller.shutdown)
    _install_sigint(app, window, controller)
    return app.exec_()


def _install_sigint(app, window, controller):
    """Make Ctrl+C in the terminal close the console.

    Qt's event loop sits in C, so a Python signal handler set with `signal.signal` does not run
    until the interpreter is next entered -- which, in an idle GUI, can be never. The usual remedy
    is a timer that does nothing except give Python a chance to notice. Measured before this: the
    process kept running after SIGINT and had to be killed, which for a program people start from
    a shell is a bug, not a quirk.
    """
    import signal

    from PyQt5 import QtCore

    def finish():
        # Close the window rather than quitting outright, so the GL objects are released while the
        # context still exists and the reader threads are joined. Quitting straight from the signal
        # handler left both to Qt's teardown and the process aborted.
        window.allow_close()
        app.quit()

    def handler(_signum, _frame):
        window.status_text.setText("종료 신호를 받았습니다. 시뮬레이터를 정리합니다…")
        controller.shutdown_finished.connect(finish)
        controller.begin_shutdown()

    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except ValueError:
        return                      # not the main thread; nothing to install
    waker = QtCore.QTimer(app)
    waker.setInterval(200)
    waker.timeout.connect(lambda: None)     # hands control back to Python often enough to see it
    waker.start()
    app._sigint_waker = waker               # keep it alive


def main(argv: Optional[List[str]] = None) -> int:
    return launch(argv)
