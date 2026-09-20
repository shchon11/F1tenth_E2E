"""Per-opponent configuration: one specification per car of a race, instead of one per race.

Everything about the other cars has been a property of the *race* so far. `--opponent teacher`
makes every opponent a teacher; `--opp-speed 0.6 1.0` gives all of them the same band; `--opp-events
brake` scripts all of them; `--spawn-order behind` places the whole grid. With two cars that is the
same thing as configuring the opponent. With three it stops being: "a slow car ahead and a defending
car alongside" is not a configuration this vocabulary can say, and it is the configuration a race is.

The user asked for exactly that (2026-09-15): *"대상차를 티쳐/정책 선택여부 뿐만 아니라 각 대상차에서
적용할 속도 프로파일링 및 체크포인트 등의 설정이 가능했으면 해."*

An `OpponentSlot` is that specification for one car. A race of M cars is the learner (slot 0) plus
M-1 slots, and slot i of *every* race in the batch is built from spec i. What the spec fixes is
deterministic per slot -- who drives it, which checkpoint, which events it can have, where it starts
-- and what it leaves as a range is still drawn per reset, from the simulator's own generator, so a
seed still reproduces the race.

Three things this module is built around.

**A driver-kind registry, not an if-chain.** `KINDS` is the table; adding a kind is one entry in it,
and whether a kind can be *used* is asked of the tree (`DriverKind.available` imports its module)
rather than written down. That is what made worker 17's `InteractiveTeacher` cost exactly one entry:
while it lived on a branch the kind was listed with `available = False` and a message saying so --
a UI that offers it greys it out and says why, a flag that names it is refused with the same
sentence -- and on the day the branch merged (`integrate/20260916`) the same entry started building
the real teacher through `teacher_factory`. What it is *not*, in either state, is silently mapped
onto the raceline teacher: a run that says `interactive` and drives a raceline is a result nobody
can read.

**Off is off.** `EnvConfig.opponent_slots = None` is the whole feature switched off, and the env then
runs the instructions it ran before this module existed. Every existing checkpoint and benchmark
number was measured on that path.

**One spelling.** The same JSON goes into `--opp-slots` for a training run and into the console's
session config, and the console's table is a view of it. A table that emitted something the trainer
could not parse would be a second configuration language nobody could diff against the first.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from . import opponent_event_contract as opp_ev

#: Where a slot's car starts **relative to the learner**. This is the mirror of `--spawn-order`,
#: which says where the *learner* starts relative to everyone: a per-car table reads "this car
#: starts ahead of me", so that is the word this field uses. `ahead` is the default and reproduces
#: `--spawn-order behind` (the learner at the back, with a pass to make) exactly.
SLOT_SPAWNS = ("ahead", "behind", "alongside", "random")
SLOT_SPAWN_ID = {name: i for i, name in enumerate(SLOT_SPAWNS[:3])}

#: The grip the teacher of a slot plans its speed profile on -- `RacelineTeacher.label_grip`.
GRIP_LABELS = ("true", "nominal", "conservative")

#: Korean labels for the console, one place, so the table and the header line agree.
GRIP_LABEL_TEXT = {"true": "참값 (privileged: 이 차의 실제 마찰로 계획)",
                   "nominal": "공칭 (고정된 차량 공칭 마찰로 계획)",
                   "conservative": "보수적 (항상 최저 그립 프로파일)"}
#: The same three, short enough for a table cell.
GRIP_LABEL_SHORT = {"true": "참값", "nominal": "공칭", "conservative": "보수적"}
SPAWN_TEXT = {"ahead": "앞", "behind": "뒤", "alongside": "나란히", "random": "무작위"}


@dataclass(frozen=True)
class DriverKind:
    """One kind of driver a slot may name.

    `teacher` and `checkpoint` are what the rest of the system has to know: a teacher-driven car
    needs a raceline and can carry `f1sim.opponent_events`, a checkpoint-driven car needs an
    `OpponentPool` entry, and a `self` car is the learner's own current weights and needs neither.
    `module` names an import that must succeed for the kind to be usable at all, which is how a kind
    that lives on a branch is *listed and refused* rather than missing or silently substituted.
    """
    name: str
    label: str                       # what the console's combo shows
    note: str                        # the one-line explanation under it
    teacher: bool = False            # driven by a raceline-style teacher (raceline, events, grip label)
    checkpoint: bool = False         # needs `checkpoint=` -- a policy .pt
    policy: bool = False             # driven by the learner's own current weights
    module: str = ""                 # import that has to exist; "" = always available
    pending: str = ""                # why it is not here yet, when `module` is missing
    #: `"module:Class"` for a teacher kind that is *not* the raceline teacher. The env builds it as
    #: `Class(raceline_teacher, env=env)` and asks it for the rows this kind drives, next to the
    #: raceline teacher's own. Empty means "this kind IS the raceline teacher".
    #:
    #: This field is what makes the registry a registry rather than a list of names. Without it a
    #: teacher kind the tree happens to have would be driven by the raceline teacher -- which is the
    #: silent substitution the whole `available` machinery exists to prevent, arriving by the back
    #: door the moment the branch merges.
    teacher_factory: str = ""

    @property
    def available(self) -> bool:
        if not self.module:
            return True
        import importlib.util
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            return False

    def unavailable_message(self) -> str:
        return (f"상대차 종류 '{self.name}' 은(는) 이 트리에 아직 없습니다: {self.pending}"
                if self.pending else
                f"상대차 종류 '{self.name}' 은(는) 이 트리에서 사용할 수 없습니다 ({self.module} 없음).")


#: The registry. Order is the order the console lists them in.
KINDS: Tuple[DriverKind, ...] = (
    DriverKind("raceline", "raceline 티처",
               "최적 레이싱 라인을 따라가는 특권 티처. 속도 프로파일·그립 라벨·이벤트를 모두 받습니다.",
               teacher=True),
    DriverKind("interactive", "interactive 티처",
               "상대 차의 예측된 미래 위치까지 보고 학생의 행동 공간에서 계획을 고르는 티처 "
               "(`f1sim.interactive_teacher`). 추월을 시범 보일 수 있는 쪽입니다.",
               teacher=True, module="f1sim.interactive_teacher",
               teacher_factory="f1sim.interactive_teacher:InteractiveTeacher",
               pending="`f1sim/interactive_teacher.py` 가 이 트리에 없습니다 "
                       "(2026-09-16 병합 이후로는 있어야 정상입니다)."),
    DriverKind("forzaeth", "ForzaETH spliner",
               "ForzaETH race stack (arXiv:2403.11784) 의 spliner local planner. 상대차 현재 위치를 "
               "기준으로 7점 spline 을 그려 옆으로 비켜 갑니다 (`f1sim.spliner_teacher`). "
               "우리 정책의 추월을 견줄 외부 기준선입니다.",
               teacher=True, module="f1sim.spliner_teacher",
               teacher_factory="f1sim.spliner_teacher:SplinerTeacher",
               pending="`f1sim/spliner_teacher.py` 가 이 트리에 없습니다."),
    DriverKind("forzaeth_pred", "ForzaETH predictive spliner",
               "같은 planner 를 상대차의 0.5 초 뒤 예측 위치에 겨눕니다. 원 논문은 상대 속도를 "
               "Gaussian process 로 추정하지만 여기서는 시뮬레이터가 참값을 알므로, 그 추정의 "
               "오차가 없는 낙관적인 상한으로 읽어야 합니다.",
               teacher=True, module="f1sim.spliner_teacher",
               teacher_factory="f1sim.spliner_teacher:PredictiveSplinerTeacher",
               pending="`f1sim/spliner_teacher.py` 가 이 트리에 없습니다."),
    DriverKind("lane_switch", "lane-switch (고정 레인)",
               "레이싱 라인에서 일정 간격으로 떨어진 고정 레인들 중 비어 있는 가장 싼 레인을 골라 "
               "옮겨 갑니다 (`f1sim.lane_teacher`). UNIST UNICORN 이 공개한 구성(Lane Change "
               "Planner + Frenet 추종 + L1)과 같은 계열이지만, 공개된 것이 구성요소 이름뿐이라 "
               "**재현이 아니라 같은 계열의 구현**입니다. 수치는 우리 것이며 모듈에 그렇게 적어 "
               "두었습니다. spline 계열과 달리 이산 선택 + hysteresis 라 비교 축이 다릅니다.",
               teacher=True, module="f1sim.lane_teacher",
               teacher_factory="f1sim.lane_teacher:LaneSwitchTeacher",
               pending="`f1sim/lane_teacher.py` 가 이 트리에 없습니다."),
    DriverKind("policy", "정책 체크포인트",
               "저장된 정책이 스스로 주행합니다. 자기 라인을 잡고 자기 실수를 합니다.",
               checkpoint=True),
    DriverKind("self", "자기 자신",
               "학습 중인 정책의 현재 가중치 (self-play).", policy=True),
)

KIND_NAMES = tuple(k.name for k in KINDS)
KIND_BY_NAME = {k.name: k for k in KINDS}


def build_teacher(kind: DriverKind, base, env):
    """The driver object a non-raceline teacher kind needs, built from the raceline teacher.

    One factory call, named by the registry entry: the per-car tensors (`speed_scale`,
    `label_grip_codes`) stay on `base`, which every such teacher keeps as its reference, so a slot's
    speed profile and grip label reach it without this function knowing what it is.
    """
    import importlib
    mod_name, _, cls_name = kind.teacher_factory.partition(":")
    if not mod_name or not cls_name:
        raise ValueError(f"opponent kind {kind.name!r} has no teacher factory to build")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(base, env=env)


def kind_of(name: str) -> DriverKind:
    try:
        return KIND_BY_NAME[str(name)]
    except KeyError:
        raise ValueError(f"unknown opponent kind {name!r}: choose from {list(KIND_NAMES)}") from None


def available_kinds() -> Tuple[DriverKind, ...]:
    return tuple(k for k in KINDS if k.available)


# --------------------------------------------------------------------------- the spec
def _speed_pair(value) -> Tuple[float, float]:
    """`1.0`, `[0.6, 1.15]` or `(lo, hi)` -> a (lo, hi) pair. A single number is a degenerate range."""
    if isinstance(value, (int, float)):
        lo = hi = float(value)
    else:
        seq = list(value)
        if len(seq) == 1:
            lo = hi = float(seq[0])
        elif len(seq) == 2:
            lo, hi = float(seq[0]), float(seq[1])
        else:
            raise ValueError(f"speed_scale {value!r}: one number or two (lo, hi)")
    if not (math.isfinite(lo) and math.isfinite(hi)) or not (0.0 < lo <= hi):
        raise ValueError(f"speed_scale {value!r}: needs finite 0 < lo <= hi (above 1.0 is a car "
                         f"faster than its profile, which is allowed)")
    return (lo, hi)


@dataclass(frozen=True)
class OpponentSlot:
    """Everything about one opponent car of a race.

    `speed_scale` is a multiplier, not a speed. On a teacher-driven car it scales the raceline speed
    *profile*, exactly as `--opp-speed` did for all of them at once. On a checkpoint or a `self` car
    there is no profile to scale -- a policy drives at whatever pace it drives -- so it scales that
    car's **speed cap** against `selfplay_pace_ref`, which is the handle `--opp-speed` already had on
    a pool car. `speed_cap` (m/s, absolute) pins the cap directly and wins over both; it is the one
    handle that means the same thing for every kind.

    `controller` is the plan-controller arm the *checkpoint* was trained under (`learn.grip_runtime`
    names them). It is checked against what the file records, not installed: the env has one plan
    tracker shared by every car, so a slot cannot run its own arm, and the honest thing a slot can do
    is refuse a checkpoint whose plans were fitted to a controller this session is not running.

    `seed` gives this slot its own draw stream for its per-reset ranges. Without it the draws come
    from the simulator's generator like everything else -- reproducible, but shared, so re-writing
    slot 2 moves slot 1's numbers. With it slot 1 replays whatever else changes.
    """
    kind: str = "raceline"
    checkpoint: Optional[str] = None
    controller: str = "legacy"
    speed_scale: Tuple[float, float] = (1.0, 1.0)
    label_grip: str = "true"
    speed_cap: Optional[float] = None
    events: Tuple[str, ...] = ()
    event_rate: float = 0.0
    reactive: Dict[str, float] = field(default_factory=dict)
    spawn: str = "ahead"
    seed: Optional[int] = None

    # ---------------------------------------------------------------- construction
    @staticmethod
    def from_dict(d: Dict) -> "OpponentSlot":
        """One slot from a JSON object. Unknown keys are an error, not a shrug.

        A typo in a per-car table is exactly the mistake that produces a run which looks like the one
        that was asked for; `--opp-speed-scale` silently ignored is a race at 1.0.
        """
        if not isinstance(d, dict):
            raise ValueError(f"an opponent slot is a JSON object, got {type(d).__name__}")
        known = set(OpponentSlot.__dataclass_fields__)
        unknown = sorted(k for k in d if k not in known)
        if unknown:
            raise ValueError(f"opponent slot: unknown field(s) {unknown}; known fields are "
                             f"{sorted(known)}")
        kind = str(d.get("kind", "raceline"))
        kind_of(kind)                                        # raises on a name that is not a kind
        events = opp_ev.parse_events(d.get("events", ()))
        reactive = dict(d.get("reactive") or {})
        # A reactive name written in `events` is accepted and moved where it belongs, because that
        # is how `--opp-events` spells it and someone will write it that way.
        timed, react = opp_ev.split_events(events)
        for name in react:
            reactive.setdefault(name, 0.0)
        bad = sorted(set(reactive) - set(opp_ev.REACTIVE_NAMES))
        if bad:
            raise ValueError(f"opponent slot: unknown reactive behaviour(s) {bad}; choose from "
                             f"{list(opp_ev.REACTIVE_NAMES)}")
        for name, p in reactive.items():
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError(f"opponent slot: reactive {name} p={p}: a probability is in [0, 1]")
        grip = str(d.get("label_grip", "true"))
        if grip not in GRIP_LABELS:
            raise ValueError(f"opponent slot: label_grip {grip!r} is not one of {list(GRIP_LABELS)}")
        spawn = str(d.get("spawn", "ahead"))
        if spawn not in SLOT_SPAWNS:
            raise ValueError(f"opponent slot: spawn {spawn!r} is not one of {list(SLOT_SPAWNS)}")
        cap = d.get("speed_cap", None)
        if cap is not None:
            cap = float(cap)
            if not (math.isfinite(cap) and cap > 0):
                raise ValueError(f"opponent slot: speed_cap {cap}: a cap is a positive speed in m/s "
                                 f"(leave it out for 'no cap of its own')")
        rate = float(d.get("event_rate", 0.0))
        if not (math.isfinite(rate) and rate >= 0.0):
            raise ValueError(f"opponent slot: event_rate {rate}: events per 10 s, >= 0")
        seed = d.get("seed", None)
        return OpponentSlot(
            kind=kind,
            checkpoint=(str(d["checkpoint"]) if d.get("checkpoint") else None),
            controller=str(d.get("controller", "legacy")),
            speed_scale=_speed_pair(d.get("speed_scale", 1.0)),
            label_grip=grip, speed_cap=cap,
            events=tuple(timed), event_rate=rate,
            reactive={k: float(v) for k, v in reactive.items() if float(v) > 0.0},
            spawn=spawn, seed=(None if seed is None else int(seed)))

    def to_dict(self) -> Dict:
        """The JSON object this slot came from -- only the fields that are not the default.

        Round trip, and readable: a table of four cars whose every field were spelled out is 40 lines
        of JSON in which the one number that matters cannot be found.
        """
        base = OpponentSlot()
        out: Dict = {"kind": self.kind}
        lo, hi = self.speed_scale
        if self.checkpoint:
            out["checkpoint"] = self.checkpoint
        if self.controller != base.controller:
            out["controller"] = self.controller
        if self.speed_scale != base.speed_scale:
            out["speed_scale"] = ([lo, hi] if lo != hi else lo)
        if self.label_grip != base.label_grip:
            out["label_grip"] = self.label_grip
        if self.speed_cap is not None:
            out["speed_cap"] = self.speed_cap
        if self.events:
            out["events"] = list(self.events)
        if self.event_rate:
            out["event_rate"] = self.event_rate
        if self.reactive:
            out["reactive"] = {k: self.reactive[k] for k in opp_ev.REACTIVE_NAMES
                               if self.reactive.get(k)}
        if self.spawn != base.spawn:
            out["spawn"] = self.spawn
        if self.seed is not None:
            out["seed"] = int(self.seed)
        return out

    # ---------------------------------------------------------------- what it is
    @property
    def driver(self) -> DriverKind:
        return kind_of(self.kind)

    @property
    def teacher_driven(self) -> bool:
        return self.driver.teacher

    @property
    def policy_driven(self) -> bool:
        return self.driver.policy

    def describe(self) -> str:
        """One line for a log header, a facts strip and the census."""
        lo, hi = self.speed_scale
        parts = [self.kind if not self.checkpoint else f"{self.kind}:{os.path.basename(self.checkpoint)}"]
        parts.append(f"x{lo:g}" if lo == hi else f"x{lo:g}-{hi:g}")
        if self.teacher_driven:
            parts.append(f"grip {self.label_grip}")
        if self.speed_cap is not None:
            parts.append(f"cap {self.speed_cap:g} m/s")
        if self.events:
            parts.append(f"{'+'.join(self.events)}@{self.event_rate:g}/10s")
        if self.reactive:
            parts.append("+".join(f"{n} {p:g}" for n, p in sorted(self.reactive.items())))
        parts.append(f"spawn {self.spawn}")
        return " · ".join(parts)


# --------------------------------------------------------------------------- parsing a table
def parse_slots(value) -> Optional[Tuple[OpponentSlot, ...]]:
    """`None`/`""` -> None (off). A JSON array, an `@file.json`, or a list of dicts/slots -> a table.

    `None` and `()` are different answers: `None` is "this feature is off", `()` is "a race of one
    car, with a slot table that happens to be empty", which is a contradiction and is refused.
    """
    if value is None:
        return None
    if isinstance(value, OpponentSlot):
        return (value,)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("@"):
            path = os.path.expanduser(text[1:])
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except OSError as exc:
                raise ValueError(f"--opp-slots {text}: cannot read {path}: {exc}") from None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--opp-slots: not JSON ({exc}). Pass a JSON array of slot objects, or "
                             f"@path/to/slots.json") from None
    else:
        data = value
    if isinstance(data, dict):
        data = [data]                       # one car, written without the brackets
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"--opp-slots: a JSON array of slot objects, got {type(data).__name__}")
    if not data:
        raise ValueError("--opp-slots []: an empty table is not 'no slots', it is a race with no "
                         "other car. Leave the flag out for that.")
    return tuple(s if isinstance(s, OpponentSlot) else OpponentSlot.from_dict(s) for s in data)


def slots_json(slots: Sequence[OpponentSlot], indent: Optional[int] = None) -> str:
    """The table as JSON -- what `--opp-slots` takes and what a session config stores."""
    return json.dumps([s.to_dict() for s in slots], ensure_ascii=False, indent=indent)


def slots_dicts(slots: Optional[Sequence[OpponentSlot]]) -> Optional[list]:
    return None if slots is None else [s.to_dict() for s in slots]


def validate_slots(slots: Sequence[OpponentSlot], race_size: int, *,
                   require_files: bool = True) -> None:
    """Refuse a table that cannot be built, with the reason. Raises `ValueError`.

    Called by the flag, by the env and by the console, so "this is not a race" has one wording.
    """
    if race_size != 1 + len(slots):
        raise ValueError(
            f"opponent_slots names {len(slots)} car(s) but race_size is {race_size}: the table is "
            f"the *other* cars of a race, so it has race_size - 1 rows. Use race_size "
            f"{1 + len(slots)}, or {race_size - 1} row(s).")
    for i, s in enumerate(slots, start=1):
        kind = s.driver
        where = f"opponent slot {i} ({s.kind})"
        if not kind.available:
            raise ValueError(f"{where}: {kind.unavailable_message()}")
        if kind.checkpoint and not s.checkpoint:
            raise ValueError(f"{where}: needs a checkpoint path -- that is what drives the car.")
        if not kind.checkpoint and s.checkpoint:
            raise ValueError(f"{where}: carries a checkpoint but a '{s.kind}' car is not driven by "
                             f"one, so it would be ignored. Use kind 'policy', or drop the path.")
        if kind.checkpoint and require_files and not os.path.exists(s.checkpoint):
            raise ValueError(f"{where}: no such checkpoint {s.checkpoint!r}.")
        if not kind.teacher:
            if s.events or s.reactive:
                raise ValueError(
                    f"{where}: events / reactive behaviour are the teacher's script and a "
                    f"'{s.kind}' car is not teacher-driven, so naming them here would run the "
                    f"unscripted car anyway. Drop them, or make this slot a teacher.")
            if s.label_grip != "true":
                raise ValueError(f"{where}: label_grip is the teacher's speed profile and a "
                                 f"'{s.kind}' car has none.")
        if s.events and not s.event_rate > 0:
            raise ValueError(
                f"{where}: events {list(s.events)} with event_rate {s.event_rate}: the rate is how "
                f"many events this car gets per 10 s, so at 0 they never fire and the car is "
                f"silently the unscripted one. Give it a positive rate or drop the events.")
        if s.event_rate > 0 and not s.events:
            raise ValueError(f"{where}: event_rate {s.event_rate} with no events to fire.")
        for name, p in s.reactive.items():
            if not 0.0 < float(p) <= 1.0:
                raise ValueError(f"{where}: reactive {name} p={p}: 0 < p <= 1, or leave it out.")


def describe(slots: Optional[Sequence[OpponentSlot]]) -> str:
    """The whole table on one line, in slot order."""
    if slots is None:
        return "opponent slots off"
    return " | ".join(f"car {i}: {s.describe()}" for i, s in enumerate(slots, start=1))


def mix_summary(slots: Optional[Sequence[OpponentSlot]]) -> str:
    """The short form for a header: how many of each kind. `2x raceline, 1x policy`."""
    if not slots:
        return ""
    order, seen = [], {}
    for s in slots:
        if s.kind not in seen:
            seen[s.kind] = 0
            order.append(s.kind)
        seen[s.kind] += 1
    return ", ".join(f"{seen[k]}x {k}" for k in order)


# --------------------------------------------------------------------------- presets
#: The console's preset buttons, and what a script can name. Each maps a row count to a table, so a
#: preset applies whatever the race size is.
def _repeat(spec: Dict, n: int) -> Tuple[OpponentSlot, ...]:
    return tuple(OpponentSlot.from_dict(dict(spec)) for _ in range(n))


def preset_basic(n: int) -> Tuple[OpponentSlot, ...]:
    """What the viewer has always run: raceline teachers at full profile speed."""
    return _repeat({"kind": "raceline", "speed_scale": 1.0}, n)


def preset_interactive(n: int) -> Tuple[OpponentSlot, ...]:
    """`preset_basic` with the UPGRADED teacher: `f1sim.interactive_teacher.InteractiveTeacher`.

    The user asked whether the visualiser's teacher is the upgraded one
    (*"비쥬얼라이저의 티쳐는 그 업그레이드된 티쳐로 동작하는거지?"*). It is, when this is chosen --
    and `기본` above is still the raceline teacher, so it is a selection and not a substitution.
    Everything else is deliberately identical to `preset_basic`, so the two differ by the teacher
    and by nothing else: same speed profile, same grip label, no events, no dispositions.

    Refused with the registry's own sentence on a tree that does not carry the module
    (`validate_slots` -> `DriverKind.unavailable_message`), never quietly mapped onto the raceline
    teacher.
    """
    return _repeat({"kind": "interactive", "speed_scale": 1.0}, n)


#: What the training recipe's population is made of, in the order `--opp-pool` named them. Used by
#: `preset_recipe` when the caller has not picked checkpoints of its own; entries that are not on
#: this machine are skipped rather than written into a table that cannot start.
RECIPE_POOL_CHECKPOINTS = (
    "~/f1sim_runs/_baselines/frozen_original_48cc698f.pt",      # the frozen original
    "~/f1sim_runs/cl_origrecipe_legacy_s701/ppo_final.pt",      # A701
    "~/f1sim_runs/cl_mem_s701/ppo_latest.pt",                   # a memory policy (mem_u8)
)


def default_recipe_checkpoints() -> Tuple[str, ...]:
    """The recipe pool's checkpoints that actually exist here."""
    out = []
    for c in RECIPE_POOL_CHECKPOINTS:
        path = os.path.expanduser(c)
        if os.path.isfile(path):
            out.append(path)
    return tuple(out)


def preset_recipe(n: int, checkpoints: Sequence[str] = ()) -> Tuple[OpponentSlot, ...]:
    """The training recipe's population, spread over the slots: teacher, self, then checkpoints.

    `--opponent pool` drew one of those per race and gave it to every opponent of that race; a slot
    table puts the whole population on the grid at once, which is the thing a pool could never do.
    The band and the event list are the recipe's (`0.6-1.15`, all four timed events, the four
    reactive behaviours), because the point of the preset is to be that configuration.
    """
    cks = list(checkpoints) or list(default_recipe_checkpoints())
    out = []
    for i in range(n):
        j = i % (2 + len(cks))
        if j == 0:
            out.append(OpponentSlot.from_dict({
                "kind": "raceline", "speed_scale": [0.6, 1.15],
                "events": list(opp_ev.EVENT_NAMES), "event_rate": 1.0,
                "reactive": {"defend": 0.4, "yield": 0.3, "line": 0.5, "oblivious": 0.1}}))
        elif j == 1:
            out.append(OpponentSlot.from_dict({"kind": "self", "speed_scale": [0.6, 1.15]}))
        else:
            out.append(OpponentSlot.from_dict({"kind": "policy", "checkpoint": cks[j - 2],
                                               "speed_scale": [0.6, 1.15]}))
    return tuple(out)


def preset_slow_leader(n: int) -> Tuple[OpponentSlot, ...]:
    """A slow car on the line, ahead: the pass the benchmark's traffic family is about."""
    return _repeat({"kind": "raceline", "speed_scale": 0.6, "spawn": "ahead"}, n)


def preset_blocker(n: int) -> Tuple[OpponentSlot, ...]:
    """A car that defends every time. The situation `--opp-defend-prob 1.0` produced by chance."""
    return _repeat({"kind": "raceline", "speed_scale": 0.8, "spawn": "ahead",
                    "reactive": {"defend": 1.0}}, n)


PRESETS = (
    ("basic", "기본 (raceline 티처 1.0)", preset_basic),
    ("interactive", "업그레이드 티처 (interactive 1.0)", preset_interactive),
    ("recipe", "학습 레시피 (0.6–1.15 · 전체 이벤트)", preset_recipe),
    ("slow", "느린 선두 (0.6)", preset_slow_leader),
    ("blocker", "막는 상대 (defend 1.0)", preset_blocker),
)
PRESET_BY_KEY = {k: (title, fn) for k, title, fn in PRESETS}


def preset(key: str, n: int, checkpoints: Sequence[str] = ()) -> Tuple[OpponentSlot, ...]:
    try:
        _title, fn = PRESET_BY_KEY[key]
    except KeyError:
        raise ValueError(f"unknown opponent preset {key!r}: "
                         f"choose from {[k for k, _, _ in PRESETS]}") from None
    try:
        return fn(n, checkpoints)
    except TypeError:
        return fn(n)
