import os as _os
# torch.compile caches live in /tmp by default and vanish on reboot (60-90 s cold compile each time);
# keep them under the user's cache dir instead. Must be set before the first compile.
_os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _os.path.join(_os.path.expanduser("~"), ".cache", "f1sim", "inductor"))
_os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

from .params import Config, VehicleParams, ActuatorParams, LidarParams, OdomParams, ImuParams, SimParams, RandomizationConfig
from .track import Track
from .sim import Simulator, StepResult

__all__ = ["Config", "VehicleParams", "ActuatorParams", "LidarParams", "OdomParams", "ImuParams", "SimParams",
           "RandomizationConfig", "Track", "Simulator", "StepResult"]
