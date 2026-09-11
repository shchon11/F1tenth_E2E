"""Viewer package: the Qt driving console, the moderngl scene it draws with, and the older
WebSocket `Viewer`.

`Viewer` is resolved lazily for the same reason as in `f1sim/__init__.py`: `server.py` imports
torch, and the console process imports `f1sim.viewer.gl_scene` (numpy and PIL only) without ever
wanting a tensor. `from f1sim.viewer import Viewer` is unchanged.
"""
__all__ = ["Viewer"]


def __getattr__(name):
    if name != "Viewer":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .server import Viewer
    globals()["Viewer"] = Viewer
    return Viewer


def __dir__():
    return sorted(set(globals()) | set(__all__))
