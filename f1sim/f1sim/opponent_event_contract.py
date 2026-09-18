"""Pure opponent event names and parsing shared by the simulator and console.

Keeping this contract separate lets configuration widgets load without importing Torch.
"""
from __future__ import annotations

#: Timed event names, in id order. Fixed for the life of the feature so a logged id keeps its
#: meaning; the reactive names are appended after them for the same reason.
EVENT_NAMES = ("brake", "stop", "shift", "weave")

#: Reactive behaviour names, in bit order.
REACTIVE_NAMES = ("defend", "yield", "line", "oblivious")

ALL_EVENT_NAMES = EVENT_NAMES + REACTIVE_NAMES

#: Event ids as reported in `info["opp_event"]["id"]`. 0 is "no event"; the rest are these names'
#: 1-based positions in `ALL_EVENT_NAMES`. Only a timed event ever appears in `id` -- a reactive
#: behaviour is not a state the car is "in" to the exclusion of others, so those are reported as a
#: bitmask in `info["opp_event"]["react"]` instead, under `REACTIVE_BIT`.
EVENT_ID = {name: i + 1 for i, name in enumerate(ALL_EVENT_NAMES)}
NO_EVENT = 0

#: Bit of each reactive behaviour in the `disposition` / `react` masks.
REACTIVE_BIT = {name: 1 << i for i, name in enumerate(REACTIVE_NAMES)}

#: `opp_<name>_prob` is the per-race probability a teacher-driven car is given that disposition.
REACTIVE_PROB_FIELD = {name: f"opp_{name}_prob" for name in REACTIVE_NAMES}


def parse_events(events) -> tuple:
    """Normalize an `opp_events` value (tuple/list, or a comma-separated string) and check the names."""
    if events is None:
        return ()
    if isinstance(events, str):
        events = [e for e in events.replace(" ", "").split(",") if e]
    names = tuple(str(e) for e in events)
    bad = [n for n in names if n not in EVENT_ID]
    if bad:
        raise ValueError(f"unknown opponent event(s) {bad}: choose from {list(ALL_EVENT_NAMES)}")
    seen = set()
    return tuple(n for n in names if not (n in seen or seen.add(n)))


def split_events(events) -> tuple:
    """(timed names, reactive names) of a parsed or unparsed `opp_events` value, in id order."""
    names = parse_events(events)
    return (tuple(n for n in names if n in EVENT_NAMES),
            tuple(n for n in names if n in REACTIVE_NAMES))


