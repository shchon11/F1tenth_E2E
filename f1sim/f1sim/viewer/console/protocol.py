"""The contract between the console (Qt) and the simulation worker (torch).

Shared by both processes, and free of both Qt and torch so either side can import it cheaply and a
test can exercise it with neither.

Three channel rules matter more than the message list.

**Generations.** Anything that changes *what is being simulated* -- the run, the map, the number of
cars, domain randomisation -- carries a `gen`. The worker stamps every frame and status with the
generation it belongs to, and the console throws away anything from a generation it is no longer
showing. Without that, clicking through three maps quickly leaves the third one's geometry on
screen with the first one's cars still arriving, and the user's last request appears to have been
ignored.

**Acks.** Commands that take effect immediately (pause, reset, focus) carry a `seq`, and the worker
replies with the same `seq`. Until that reply lands the UI draws the control as pending, never as
done. A pause button that says "일시정지" while the simulation is still stepping is worse than a
slow one.

**Bounded, drop-oldest frames.** The frame channel is not a queue to be drained in order; it is a
window onto the newest state. The worker drops old frames rather than block the simulation, and
reports how many, so a backlog is a number on screen instead of a mystery.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = 1

# ---------------------------------------------------------------- console -> worker
CMD_HELLO = "hello"                # clock handshake + capability report
CMD_LIST_MAPS = "list_maps"        # map catalog (needs f1sim.maps, hence the worker)
CMD_DESCRIBE = "describe"          # read a checkpoint's metadata without starting anything
CMD_START = "start"                # build and run a session (new generation)
CMD_STOP = "stop"                  # tear the session down, stay alive
CMD_PAUSE = "pause"                # {"paused": bool}
CMD_RESET = "reset"                # re-draw the episode on the current map
CMD_FOCUS = "focus"                # {"env": int} which car the overlays describe
CMD_OVERLAY = "overlay"            # {"saliency": bool, "internals": bool, "plan": bool}
CMD_CANCEL = "cancel"              # abandon an in-flight start (by generation)
CMD_SHUTDOWN = "shutdown"

# ---------------------------------------------------------------- worker -> console
MSG_HELLO = "hello"                # {"pid", "monotonic", "torch", "cuda", "device", "runs_dir"}
MSG_MAPS = "maps"                  # {"groups": {name: [map, ...]}}
MSG_DESCRIBED = "described"        # {"ok", "info": {...}} for one checkpoint
MSG_STAGE = "stage"                # progress inside PREPARING
MSG_READY = "ready"                # session is up: static geometry + session facts
MSG_GEOMETRY = "geometry"          # a map's static meshes (also sent on map switch)
MSG_ACK = "ack"                    # {"seq", "command", "state": {...}}
MSG_STATE = "state"                # unsolicited state change (e.g. worker paused itself)
MSG_ERROR = "error"                # {"gen", "where", "message", "detail", "retryable"}
MSG_STOPPED = "stopped"
MSG_LOG = "log"                    # plain text the console shows in the log drawer
MSG_BYE = "bye"

# ---------------------------------------------------------------- session states
STATE_IDLE = "IDLE"
STATE_PREPARING = "PREPARING"
STATE_RUNNING = "RUNNING"
STATE_PAUSED = "PAUSED"
STATE_STOPPING = "STOPPING"
STATE_FAILED = "FAILED"

#: Ordered stages of PREPARING. The console shows the current one by name, because "준비 중" for
#: twenty seconds with no detail is indistinguishable from a hang.
STAGES = [
    ("checkpoint", "체크포인트 읽는 중"),
    ("map", "맵 로드 중"),
    ("raceline", "레이싱 라인 준비 중"),
    ("env", "시뮬레이터 만드는 중"),
    ("warmup", "예열 중"),
    # `compile` is inductor: minutes, opt-in. `graph` is the explicit CUDA-graph capture the viewer
    # does by default: about a second. The old label called the first one "그래프 캡처", which is
    # what the second one actually is.
    ("compile", "torch.compile 컴파일 중"),
    ("geometry", "맵 지오메트리 만드는 중"),
    ("graph", "CUDA 그래프 캡처 중"),
    ("done", "시작"),
]
STAGE_TEXT = dict(STAGES)


@dataclass
class SessionConfig:
    """Everything the worker needs to build a session.

    `races` x `cars_per_race` replaces the old `--cars` / `--race-size` pair. Those two were both
    counted in cars and related by a division, so `--cars 3 --race-size 4` was an error the old
    launcher only discovered after its window had closed, and `--cars 10 --race-size 4` silently
    became 8. Asking for the two numbers that multiply removes the invalid combination instead of
    validating it.
    """
    run: str = "latest"                  # run directory, a .pt path, or "latest"
    map_name: str = ""
    races: int = 1
    cars_per_race: int = 1
    speed_cap: Optional[float] = None    # None = whatever the checkpoint was trained at
    device: str = "auto"
    #: `torch.compile` the policy and the physics. Off by default: this is a *startup cost*, not
    #: the GPU switch -- `device` decides that, and a session runs on CUDA either way. Compiling
    #: pays for itself over a long run with many cars; for opening the viewer to look at one car it
    #: costs minutes and returns nothing. Headless and training paths set it themselves and are
    #: unaffected by this default.
    compile: bool = False
    randomize: bool = True               # domain randomisation, as in training
    stochastic: bool = False
    opponent: str = "teacher"
    episode_s: float = 3600.0
    saliency: bool = False
    internals: bool = False
    max_render_cars: int = 64

    @property
    def total_cars(self) -> int:
        return max(1, int(self.races)) * max(1, int(self.cars_per_race))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SessionConfig":
        known = {f for f in SessionConfig.__dataclass_fields__}
        return SessionConfig(**{k: v for k, v in d.items() if k in known})

    def affects_simulation(self, other: "SessionConfig") -> bool:
        """True when moving to `other` needs a rebuild rather than a live command.

        Saliency and internals are live toggles; everything else changes what is being simulated
        and therefore starts a new generation.
        """
        a, b = self.to_dict(), other.to_dict()
        for k in ("saliency", "internals", "max_render_cars"):
            a.pop(k, None)
            b.pop(k, None)
        return a != b


def message(kind: str, **fields) -> Dict[str, Any]:
    """A message with its send time already on it, so the far side can measure the trip."""
    fields["kind"] = kind
    fields.setdefault("monotonic", time.monotonic())
    return fields


class LatestSlot:
    """Bounded, drop-oldest hand-off between a producer and a slower consumer.

    The producer must never block: a simulation thread that waits for a pipe is a simulation that
    runs at the speed of the display, which defeats the point of separating them. `put` therefore
    discards the oldest item when full and counts it.

    Kept here rather than in the worker so the console can use the same class for its own inbound
    side, and so it can be tested without starting either process.
    """

    def __init__(self, capacity: int = 2):
        import threading
        self.capacity = max(1, int(capacity))
        self._items: List[Any] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self.dropped = 0

    def put(self, item: Any) -> bool:
        """Returns False when an older item had to be discarded to make room."""
        with self._not_empty:
            ok = True
            while len(self._items) >= self.capacity:
                self._items.pop(0)
                self.dropped += 1
                ok = False
            self._items.append(item)
            self._not_empty.notify()
            return ok

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        with self._not_empty:
            if not self._items and not self._closed:
                self._not_empty.wait(timeout)
            if self._items:
                return self._items.pop(0)
            return None

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
