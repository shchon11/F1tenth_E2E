"""The opponent / race / spawn command line, in one place, for every tool that has to agree on it.

`ppo.py` owns the flags that describe *the learner*. The flags that describe the other cars --
which population drives them, what behaviour they have, where the grid puts them -- are shared,
because three tools have to build the same configuration out of them:

* `ppo` trains against it,
* `opponent_census` measures whether the learner actually experiences each situation in it,
* anything later that wants to reproduce one of those numbers.

A census that took its own flags would eventually measure a configuration the trainer cannot build,
and the first sign of it would be a number nobody could reproduce. So there is one parser group,
one validator and one `env_kwargs`, and `--opp-defend-prob 0.4` means the same thing everywhere.

`add_arguments` reproduces the flags `ppo.py` carried before this module existed, name for name,
default for default, help for help, so an existing command line and an existing W&B config are
unchanged.
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Optional, Sequence

from .. import opponent_events as opp_ev
from .. import opponent_slots as opp_sl
from ..gym_env import OPPONENT_MODES, POOL_SELF, POOL_TEACHER, SPAWN_ORDERS, pool_entries

#: The flags a slot table replaces. Each one answers, for every opponent at once, a question the
#: table answers per car; passing both would leave two answers in the configuration and only one of
#: them in the run. Listed as (flag, attribute, default) so `validate` can say which one was set.
SLOT_SUPERSEDED = (
    ("--opponent", "opponent", "policy"),
    ("--opp-speed", "opp_speed", (0.6, 1.0)),
    ("--opp-pool", "opp_pool", ""),
    ("--opp-events", "opp_events", ""),
    ("--opp-event-rate", "opp_event_rate", 0.0),
    ("--spawn-order", "spawn_order", "behind"),
    ("--mixed-teacher-frac", "mixed_teacher_frac", 0.5),
    ("--opp-defend-prob", "opp_defend_prob", 0.0),
    ("--opp-yield-prob", "opp_yield_prob", 0.0),
    ("--opp-line-prob", "opp_line_prob", 0.0),
    ("--opp-oblivious-prob", "opp_oblivious_prob", 0.0),
)


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the race / opponent / behaviour / spawn flags. Every one maps 1:1 onto an EnvConfig field."""
    ap.add_argument("--race-size", type=int, default=1, help="cars per track instance (>1: opponents in the LiDAR, car-car collisions)")
    ap.add_argument("--opponent", default="policy", choices=list(OPPONENT_MODES),
                    help="who drives cars 1..M-1: the policy (self-play), the raceline teacher, per race "
                         "one or the other (mixed, share set by --mixed-teacher-frac), or per race one "
                         "entry drawn from --opp-pool (pool)")
    ap.add_argument("--mixed-teacher-frac", type=float, default=0.5)
    ap.add_argument("--opp-speed", type=float, nargs=2, default=(0.6, 1.0), help="teacher opponents: speed scale range per race")
    ap.add_argument("--opp-pool", default="", metavar="A.pt,B.pt",
                    help=f"comma-separated opponent population for --opponent pool: checkpoint "
                         f"paths, {POOL_SELF!r} (the learner's own current weights, i.e. self-play) "
                         f"and {POOL_TEACHER!r} (the raceline teacher, carrying --opp-events). One "
                         f"entry is drawn per race from the simulator's generator. Empty = off. A "
                         f"checkpoint is the only opponent that takes its own line, defends its own "
                         f"position and makes its own mistakes; every scripted behaviour is still "
                         f"the same tracker on the same line being told what to do")
    ap.add_argument("--spawn-gap", type=float, nargs=2, default=(2.5, 6.0), metavar=("LOW", "HIGH"),
                    help="[m] arc along the lane between the cars of a race at spawn, drawn uniformly "
                         "per reset. The default is the range every race so far was trained at, so an "
                         "unflagged run is unchanged. Widening it is an opponent-diversity axis: a "
                         "fixed narrow band shows the policy one approach geometry")
    ap.add_argument("--spawn-order", default="behind", choices=list(SPAWN_ORDERS),
                    help="where the learner starts on the grid of a race whose other cars are not "
                         "itself: behind (every race trained so far -- a pass to make), ahead (a "
                         "place to defend, and the only way to be overtaken), alongside (side by "
                         "side, which is where the contacts measured in traffic actually happen) or "
                         "random (drawn per race)")
    ap.add_argument("--spawn-alongside-sep", type=float, default=0.15, metavar="M",
                    help="[m] body-to-body lateral gap asked for on an alongside grid; what the lane "
                         "has room for wins, and a lane with room for one car spawns a stagger")
    ap.add_argument("--spawn-alongside-gap", type=float, nargs=2, default=(0.0, 0.2), metavar=("LOW", "HIGH"),
                    help="[m] arc stagger inside an alongside grid, drawn independently per car. "
                         "Small on purpose: a grid's lateral offsets are taken along the centerline "
                         "normal at each car's own arc position, and those normals stop agreeing a "
                         "few tens of centimetres apart on a tight corner")
    # Scripted opponent behaviour (f1sim.opponent_events). Every flag maps 1:1 onto the EnvConfig
    # field of the same name; the defaults are the EnvConfig defaults, so an unflagged run is the
    # run it was before these existed.
    ap.add_argument("--opp-events", default="", metavar="A,B",
                    help=f"comma-separated behaviour for the teacher-driven opponents. Timed events "
                         f"({','.join(opp_ev.EVENT_NAMES)}) fire at --opp-event-rate; reactive "
                         f"behaviours ({','.join(opp_ev.REACTIVE_NAMES)}) read the learner's position "
                         f"every step and each takes its own --opp-<name>-prob. Empty = off. A "
                         f"teacher opponent otherwise only ever presents a slower car on the racing "
                         f"line, which is the one overtaking and avoidance problem the policy has "
                         f"already been trained on")
    ap.add_argument("--opp-event-rate", type=float, default=0.0, metavar="PER10S",
                    help="expected timed events per teacher opponent per 10 s of driving (events do "
                         "not overlap, so the realized rate is a little below this)")
    ap.add_argument("--opp-brake-scale", type=float, nargs=2, default=(0.0, 0.5), metavar=("LOW", "HIGH"),
                    help="brake: fraction of its profile speed the opponent drops to")
    ap.add_argument("--opp-brake-time", type=float, nargs=2, default=(0.5, 2.5), metavar=("LOW", "HIGH"),
                    help="[s] how long a brake event holds")
    ap.add_argument("--opp-stop-time", type=float, nargs=2, default=(1.0, 4.0), metavar=("LOW", "HIGH"),
                    help="[s] how long a stopped opponent stays stopped")
    ap.add_argument("--opp-shift-offset", type=float, nargs=2, default=(0.0, 0.35), metavar=("LOW", "HIGH"),
                    help="[m] |lateral offset| of a lane change; the sign is drawn separately")
    ap.add_argument("--opp-shift-hold", type=float, nargs=2, default=(0.5, 2.0), metavar=("LOW", "HIGH"),
                    help="[s] time held at the offset, between the ramp in and the ramp out")
    ap.add_argument("--opp-shift-ramp", type=float, default=1.0, metavar="S",
                    help="[s] ramp in / ramp out of a shift")
    ap.add_argument("--opp-weave-amp", type=float, nargs=2, default=(0.1, 0.25), metavar=("LOW", "HIGH"),
                    help="[m] sinusoidal lateral amplitude of a weave")
    ap.add_argument("--opp-weave-period", type=float, nargs=2, default=(2.0, 4.0), metavar=("LOW", "HIGH"),
                    help="[s] weave period")
    ap.add_argument("--opp-weave-time", type=float, nargs=2, default=(2.0, 6.0), metavar=("LOW", "HIGH"),
                    help="[s] how long a weave lasts")
    ap.add_argument("--opp-event-margin", type=float, default=0.10, metavar="M",
                    help="[m] free space kept beyond the car's half-width when an event moves an "
                         "opponent off the raceline; the offset is clamped per raceline point "
                         "against the track's distance field, so it can never reach a wall")
    # Reactive behaviour. A probability per teacher-driven car per race, not a rate: these are not
    # events a car falls into, they are how it drives when the learner turns up.
    ap.add_argument("--opp-defend-prob", type=float, default=0.0, metavar="P",
                    help="P(a teacher-driven opponent defends): while a learner is within "
                         "--opp-defend-range behind, it moves its line toward the side that learner "
                         "is coming down, at full amplitude inside --opp-defend-full")
    ap.add_argument("--opp-yield-prob", type=float, default=0.0, metavar="P",
                    help="P(a teacher-driven opponent yields): with a learner alongside it moves away")
    ap.add_argument("--opp-line-prob", type=float, default=0.0, metavar="P",
                    help="P(a teacher-driven opponent drives its own corner line): an out-in or "
                         "in-out offset sweep, drawn per corner")
    ap.add_argument("--opp-oblivious-prob", type=float, default=0.0, metavar="P",
                    help="P(a teacher-driven opponent's follow-gap slowdown is off): it does not "
                         "brake for a car ahead or beside it. This is the car that rams -- the "
                         "situation the training distribution has never contained on purpose")
    ap.add_argument("--opp-defend-offset", type=float, nargs=2, default=(0.15, 0.35), metavar=("LOW", "HIGH"),
                    help="[m] how far across a defending opponent moves, drawn per race")
    ap.add_argument("--opp-yield-offset", type=float, nargs=2, default=(0.15, 0.35), metavar=("LOW", "HIGH"),
                    help="[m] ... a yielding one")
    ap.add_argument("--opp-line-offset", type=float, nargs=2, default=(0.15, 0.35), metavar=("LOW", "HIGH"),
                    help="[m] ... each end of a corner-line sweep, drawn per corner")
    ap.add_argument("--opp-defend-range", type=float, default=0.0, metavar="M",
                    help="[m] how far behind a learner starts being defended against; 0 = the env's "
                         "own overtake_range (12 m), the window the overtake reward calls a race")
    ap.add_argument("--opp-defend-full", type=float, default=3.0, metavar="M",
                    help="[m] inside this the block is at full amplitude, ramping to zero at "
                         "--opp-defend-range")
    ap.add_argument("--opp-alongside-lon", type=float, default=0.8, metavar="M",
                    help="[m] |longitudinal offset| within which two cars count as alongside "
                         "(bodies are 0.58 m long). Read by `yield` and by the census, which must "
                         "measure the situation the behaviour reacts to")
    ap.add_argument("--opp-alongside-lat", type=float, default=1.2, metavar="M",
                    help="[m] ... and the lateral bound, so a car across a hairpin is not alongside")
    ap.add_argument("--opp-react-max", type=float, default=0.45, metavar="M",
                    help="[m] cap on the reactive lateral offset, before the lane's own clamp")
    ap.add_argument("--opp-react-slew", type=float, default=0.6, metavar="MPS",
                    help="[m/s] rate limit on it: a target that jumps sideways asks the pure-pursuit "
                         "teacher for a step steer input")
    ap.add_argument("--opp-corner-kappa", type=float, default=0.15, metavar="INVM",
                    help="[1/m] smoothed raceline curvature above which a point is in a corner, for "
                         "the `line` behaviour")
    ap.add_argument("--opp-corner-min-arc", type=float, default=1.0, metavar="M",
                    help="[m] shortest run of such points that counts as one corner")
    ap.add_argument("--opp-corner-smooth", type=float, default=0.6, metavar="M",
                    help="[m] window the curvature is averaged over first")
    ap.add_argument("--opp-slots", default="", metavar="JSON|@FILE",
                    help="per-opponent configuration: a JSON array with one object per car of a "
                         "race (race_size - 1 of them), or @path/to/slots.json. Each object may "
                         f"carry kind ({'|'.join(opp_sl.KIND_NAMES)}), checkpoint, controller, "
                         f"speed_scale (a number or [lo, hi]), label_grip "
                         f"({'|'.join(opp_sl.GRIP_LABELS)}), speed_cap [m/s], events "
                         f"({','.join(opp_ev.EVENT_NAMES)}), event_rate, reactive "
                         f"({{{','.join(n + ': p' for n in opp_ev.REACTIVE_NAMES)}}}), spawn "
                         f"({'|'.join(opp_sl.SLOT_SPAWNS)}, where that car starts relative to the "
                         f"learner) and seed. Slot i of every race in the batch is built from spec "
                         f"i, so 'a slow car ahead and a defending car alongside' is one run rather "
                         f"than two. Replaces --opponent / --opp-speed / --opp-pool / --opp-events "
                         f"/ --spawn-order, which say one thing about every opponent at once; the "
                         f"remaining --opp-* flags are the shared *ranges* an event draws from and "
                         f"still apply. Empty = off, and off is the env this flag was added to")
    return ap


def _differs(value, default) -> bool:
    """Whether a flag was moved off its default. Tuples compare elementwise, floats exactly."""
    if isinstance(default, tuple):
        return tuple(float(x) for x in value) != tuple(float(x) for x in default)
    return value != default


def opp_slots_of(a):
    """The parsed `--opp-slots` table, or None. Raises SystemExit with the parser's own reason."""
    raw = getattr(a, "opp_slots", "")
    if raw is None or isinstance(raw, tuple):
        return raw
    try:
        return opp_sl.parse_slots(raw)
    except ValueError as exc:
        raise SystemExit(str(exc))


def validate(a) -> None:
    """Check the combination and raise SystemExit with the reason. Called by every entry point.

    Every check here exists because the alternative is a run that looks like the one that was asked
    for and is silently the unflagged one.
    """
    a.opp_slots = opp_slots_of(a)
    if a.opp_slots is not None:
        set_flags = [flag for flag, attr, default in SLOT_SUPERSEDED
                     if _differs(getattr(a, attr, default), default)]
        if set_flags:
            raise SystemExit(
                f"--opp-slots with {' '.join(set_flags)}: the slot table already says who drives "
                f"each car, how fast it is, what it does and where it starts, one car at a time. "
                f"Those flags say it for every opponent at once, so the run would carry two answers "
                f"and use one. Put the value in the table, or drop the table.")
        a.opponent = "slots"
        try:
            opp_sl.validate_slots(a.opp_slots, a.race_size)
        except ValueError as exc:
            raise SystemExit(f"--opp-slots: {exc}")
    lo, hi = (float(x) for x in a.spawn_gap)
    if not (math.isfinite(lo) and math.isfinite(hi)) or not (0.0 < lo <= hi):
        raise SystemExit(f"--spawn-gap {lo} {hi}: needs finite 0 < LOW <= HIGH [m]. The gap "
                         f"is an arc drawn uniformly from [LOW, HIGH] and subtracted per grid slot, so "
                         f"a non-positive or inverted range spawns cars on top of each other.")
    a.spawn_gap = (lo, hi)
    slo, shi = (float(x) for x in a.opp_speed)
    if not (math.isfinite(slo) and math.isfinite(shi)) or not (0.0 < slo <= shi):
        raise SystemExit(f"--opp-speed {slo} {shi}: needs finite 0 < LOW <= HIGH. Above 1.0 is "
                         f"allowed and means an opponent faster than its raceline profile -- the "
                         f"learner is then the car being overtaken. The teacher's profile already "
                         f"plans at the grip limit, so much above ~1.2 is a car that leaves the "
                         f"road; `python -m f1sim.learn.opponent_census` reports opponent wall "
                         f"contacts, which is how to find this configuration's ceiling.")
    a.opp_speed = (slo, shi)
    a.opp_pool = pool_entries(a.opp_pool)
    try:
        a.opp_events = opp_ev.parse_events(a.opp_events)
    except ValueError as exc:
        raise SystemExit(f"--opp-events: {exc}")
    timed, react = opp_ev.split_events(a.opp_events)

    if a.opponent == "pool":
        if a.race_size < 2:
            raise SystemExit(f"--opponent pool with --race-size {a.race_size}: with one car per "
                             f"race there is no other car for the population to drive.")
        if not a.opp_pool:
            raise SystemExit("--opponent pool needs --opp-pool: the population is what drives the "
                             f"other car, so an empty one is no opponent at all. Name entries (a "
                             f"checkpoint path, {POOL_SELF!r} or {POOL_TEACHER!r}).")
        missing = [p for p in a.opp_pool if p not in (POOL_SELF, POOL_TEACHER) and not os.path.exists(p)]
        if missing:
            raise SystemExit(f"--opp-pool: no such checkpoint {missing}. Entries are file paths, "
                             f"{POOL_SELF!r} or {POOL_TEACHER!r}.")
    elif a.opp_pool:
        raise SystemExit(f"--opp-pool {','.join(a.opp_pool)} without --opponent pool: nothing would "
                         f"read the population, so the run would silently be a "
                         f"--opponent {a.opponent} run.")

    teacher_cars = a.race_size > 1 and (a.opponent in ("teacher", "mixed")
                                        or (a.opponent == "pool" and POOL_TEACHER in a.opp_pool))
    if a.opp_events and not teacher_cars:
        raise SystemExit(f"--opp-events {','.join(a.opp_events)} needs a teacher-driven car to act "
                         f"on and there is none here (--race-size {a.race_size}, --opponent "
                         f"{a.opponent}"
                         + (f", --opp-pool without {POOL_TEACHER!r}" if a.opponent == "pool" else "")
                         + "). Use --opponent teacher|mixed, or put "
                         f"{POOL_TEACHER!r} in the pool.")
    if timed and not a.opp_event_rate > 0:
        raise SystemExit(f"--opp-events {','.join(timed)} with --opp-event-rate {a.opp_event_rate}: "
                         f"the rate is how many events an opponent gets per 10 s, so at 0 the named "
                         f"events never fire and the run is silently the unflagged one. Pass a "
                         f"positive rate or drop them.")
    for name in react:
        field = opp_ev.REACTIVE_PROB_FIELD[name]
        flag = "--" + field.replace("_", "-")
        p = float(getattr(a, field))
        if not 0.0 < p <= 1.0:
            raise SystemExit(f"--opp-events {name} with {flag} {p}: the probability is how often a "
                             f"teacher-driven car is given that behaviour for a race, so at 0 it is "
                             f"never given and the run is silently the unflagged one. Pass "
                             f"0 < P <= 1 or drop {name}.")
    if a.spawn_order != "behind" and a.race_size < 2:
        raise SystemExit(f"--spawn-order {a.spawn_order} with --race-size {a.race_size}: the grid "
                         f"order is where the learner starts relative to the other cars of its "
                         f"race, and there are none.")


def env_kwargs(a) -> dict:
    """The EnvConfig fields these flags set. One dict, so no caller can set a subset by accident."""
    return dict(race_size=a.race_size, opponent=a.opponent,
                opponent_slots=getattr(a, "opp_slots", None) or None,
                mixed_teacher_frac=a.mixed_teacher_frac,
                opp_speed_range=tuple(a.opp_speed),
                opp_pool=tuple(a.opp_pool),
                spawn_gap=tuple(a.spawn_gap),
                spawn_order=a.spawn_order,
                spawn_alongside_sep=a.spawn_alongside_sep,
                spawn_alongside_gap=tuple(a.spawn_alongside_gap),
                opp_events=a.opp_events, opp_event_rate=a.opp_event_rate,
                opp_brake_scale_range=tuple(a.opp_brake_scale),
                opp_brake_time_range=tuple(a.opp_brake_time),
                opp_stop_time_range=tuple(a.opp_stop_time),
                opp_shift_offset_range=tuple(a.opp_shift_offset),
                opp_shift_hold_range=tuple(a.opp_shift_hold),
                opp_shift_ramp=a.opp_shift_ramp,
                opp_weave_amp_range=tuple(a.opp_weave_amp),
                opp_weave_period_range=tuple(a.opp_weave_period),
                opp_weave_time_range=tuple(a.opp_weave_time),
                opp_event_margin=a.opp_event_margin,
                opp_defend_prob=a.opp_defend_prob, opp_yield_prob=a.opp_yield_prob,
                opp_line_prob=a.opp_line_prob, opp_oblivious_prob=a.opp_oblivious_prob,
                opp_defend_offset_range=tuple(a.opp_defend_offset),
                opp_yield_offset_range=tuple(a.opp_yield_offset),
                opp_line_offset_range=tuple(a.opp_line_offset),
                opp_defend_range=a.opp_defend_range, opp_defend_full=a.opp_defend_full,
                opp_alongside_lon=a.opp_alongside_lon, opp_alongside_lat=a.opp_alongside_lat,
                opp_react_max=a.opp_react_max, opp_react_slew=a.opp_react_slew,
                opp_corner_kappa=a.opp_corner_kappa, opp_corner_min_arc=a.opp_corner_min_arc,
                opp_corner_smooth=a.opp_corner_smooth)


def slots_config(a) -> str:
    """The slot table as the JSON string a config record should carry, or "" when there is none.

    A tuple of dataclasses is not what a W&B config or a recorded argv wants: what someone reading
    the run needs is the text they can paste back into `--opp-slots`.
    """
    slots = getattr(a, "opp_slots", None)
    return opp_sl.slots_json(slots) if slots else ""


def needs_racelines(a) -> bool:
    """Whether the track set has to be loaded with racelines: something drives one."""
    slots = getattr(a, "opp_slots", None)
    if slots:
        return a.race_size > 1 and any(s.teacher_driven for s in slots)
    return a.race_size > 1 and (a.opponent in ("teacher", "mixed")
                                or (a.opponent == "pool" and POOL_TEACHER in pool_entries(a.opp_pool)))


def describe(a) -> str:
    """One line naming the opponent configuration, for a log header and a report."""
    if a.race_size < 2:
        return "solo (race_size 1): no opponent"
    slots = getattr(a, "opp_slots", None)
    if slots:
        return (f"race {a.race_size} | per-slot ({opp_sl.mix_summary(slots)}) | "
                f"spawn gap {a.spawn_gap[0]:g}-{a.spawn_gap[1]:g} m\n  "
                + "\n  ".join(f"car {i}: {s.describe()}" for i, s in enumerate(slots, start=1)))
    who = {"policy": "self-play", "teacher": "raceline teacher", "mixed": "teacher / self-play per race",
           "pool": "pool " + ",".join(a.opp_pool)}[a.opponent]
    probs = {n: float(getattr(a, opp_ev.REACTIVE_PROB_FIELD[n]))
             for n in opp_ev.split_events(a.opp_events)[1]}
    return (f"race {a.race_size} | {who} | speed x{a.opp_speed[0]:g}-{a.opp_speed[1]:g} | "
            f"spawn {a.spawn_order} gap {a.spawn_gap[0]:g}-{a.spawn_gap[1]:g} m | "
            f"{opp_ev.describe(a.opp_events, float(a.opp_event_rate), probs)}")
