"""What the console remembers between launches.

The setup sidebar is a dozen controls -- map, direction, obstacles, cars, speed cap, device,
friction, controller arm, the grip dial, the overlays -- and every one of them was reset to its
default each time the window opened. Anyone driving the same checkpoint on the same track twice in
a row set all of them twice.

What is remembered is what a person *chose*, and nothing else. Deliberately not remembered:

* the session seed, which is drawn per session on purpose, so a remembered one would quietly turn
  every run into a replay of the last;
* anything the worker reports rather than the person sets (the dial a checkpoint arrives with, the
  map list, a run's own speed cap);
* a checkpoint or map that has since disappeared -- restoring a path that no longer exists puts the
  window in a state whose Start button fails, so those are dropped on load and the field is left
  empty, which is what a first launch shows.

A corrupt or unreadable file is not an error worth a dialog: the console opens with its defaults,
which is exactly where it was before any of this existed.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

#: Not `~/.cache`: a cache is something a tool may delete, and losing these is the complaint this
#: file exists to answer. Alongside `slam_map.USER_MAPS`, which is user data for the same reason.
#: `$F1SIM_CONSOLE_PREFS` overrides it -- a test needs its own, and a window that read the running
#: user's real preferences would pass or fail depending on how that person last left the console.
PREFS_PATH = os.path.join(os.path.expanduser("~"), ".f1sim", "console.json")


def path() -> str:
    """The file in force, resolved per call. Never a default argument: bound at import, a default
    cannot be pointed somewhere else, which is how the first version of this leaked into `~`."""
    return os.environ.get("F1SIM_CONSOLE_PREFS") or PREFS_PATH

#: Bumped when a key's *meaning* changes. A file from a different version is ignored rather than
#: half-applied, because a control restored from a number that used to mean something else is worse
#: than a control at its default.
VERSION = 1


def load(file: str = "") -> Dict[str, Any]:
    try:
        with open(file or path()) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or int(data.get("version", 0)) != VERSION:
        return {}
    got = data.get("console")
    return dict(got) if isinstance(got, dict) else {}


def save(values: Dict[str, Any], file: str = "") -> bool:
    """Write, atomically. Returns False rather than raising: failing to remember a preference must
    never be what stops a window from closing."""
    try:
        p = file or path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"version": VERSION, "console": values}, f, indent=1, sort_keys=True)
        os.replace(tmp, p)
        return True
    except (OSError, TypeError, ValueError):
        return False
