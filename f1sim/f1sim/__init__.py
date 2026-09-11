import os as _os
# torch.compile caches live in /tmp by default and vanish on reboot (60-90 s cold compile each time);
# keep them under the user's cache dir instead. Must be set before the first compile.
_os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _os.path.join(_os.path.expanduser("~"), ".cache", "f1sim", "inductor"))
_os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

# The names below are resolved on first use (PEP 562) rather than imported here.
#
# What this buys, precisely: `import f1sim` no longer costs an `import torch`. Measured, the eager
# version was 1.59 s and `import torch` alone is 1.33 s of it. That matters because the viewer
# console is a Qt process which never touches a tensor -- it talks to a worker process that does --
# and every module it needs (`f1sim.viewer.gl_scene`, `f1sim.viewer.console.*`) imports without
# torch. It does *not* mean torch is avoided in general: `from f1sim import Track` still imports it,
# because `track.py` uses it. Only `params` is genuinely torch-free.
#
# This is the documented module-__getattr__ hook, not a sys.modules or import-path trick: the
# module object is the real one, submodules resolve through the normal import machinery, and
# `f1sim.Track` is the same class it always was.
_LAZY = {
    "Config": ".params", "VehicleParams": ".params", "ActuatorParams": ".params",
    "LidarParams": ".params", "OdomParams": ".params", "ImuParams": ".params",
    "SimParams": ".params", "RandomizationConfig": ".params",
    "Track": ".track",
    "Simulator": ".sim", "StepResult": ".sim",
}

# Eager `from .params import ...` also left `f1sim.params`, `f1sim.track` and `f1sim.sim` as
# attributes of the package, and outside code is entitled to rely on that: `import f1sim` followed
# by `f1sim.track.Track` used to work. Resolving these here restores it, without which the
# attribute would exist or not depending on which name someone happened to touch first.
_LAZY_SUBMODULES = ("params", "track", "sim")

__all__ = ["Config", "VehicleParams", "ActuatorParams", "LidarParams", "OdomParams", "ImuParams", "SimParams",
           "RandomizationConfig", "Track", "Simulator", "StepResult"]


def __getattr__(name):
    import importlib
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    where = _LAZY.get(name)
    if where is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(where, __name__), name)
    globals()[name] = value          # resolve once; later lookups skip this hook entirely
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(_LAZY_SUBMODULES))
