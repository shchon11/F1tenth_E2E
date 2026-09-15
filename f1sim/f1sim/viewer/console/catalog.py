"""What the GUI can discover on its own, without importing torch.

The console process deliberately never imports torch: the first window has to be on screen in
well under a second, and `import torch` alone is seconds. Runs are a directory listing, so the GUI
reads them itself and the picker is populated on the first paint. Map names need `f1sim.maps`
(which pulls in torch through `track.py`), so those come from the worker over the control channel
and the list fills in when they arrive.

`RUNS_DIR` is duplicated from `f1sim.learn.common` rather than imported for exactly that reason;
`check_runs_dir_matches()` in the worker asserts the two agree, so the copy cannot drift silently.

The *names* are no longer in that bargain. `f1sim.tracks` -- the registry and the scenario grammar
-- is torch-free, so the console formats, parses and groups a scenario itself; what still comes
from the worker is the catalogue as the worker can actually load it (which racetrack directories
and gym maps exist on its disk), which is a fact about the worker's filesystem, not about naming.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ... import tracks

RUNS_DIR = os.path.join(os.path.expanduser("~"), "f1sim_runs")
CHECKPOINT_NAMES = ("ppo_latest.pt", "student_latest.pt")


@dataclass
class RunInfo:
    """One run directory that holds at least one checkpoint the viewer can load."""
    name: str
    path: str
    checkpoint: str                 # absolute path of the newest checkpoint in the run
    kind: str                       # "PPO" or "DAgger", from the file name
    mtime: float
    size_mb: float

    @property
    def age_text(self) -> str:
        return format_age(time.time() - self.mtime)

    @property
    def subtitle(self) -> str:
        return f"{self.kind} · {self.age_text} 전"


def format_age(seconds: float) -> str:
    """Human-readable elapsed time, Korean, coarse on purpose: nobody needs 'saved 4013 s ago'."""
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.0f}초"
    if s < 3600:
        return f"{s / 60:.0f}분"
    if s < 86400:
        return f"{s / 3600:.1f}시간"
    return f"{s / 86400:.1f}일"


def list_runs(runs_dir: str = RUNS_DIR) -> List[RunInfo]:
    """Runs holding a checkpoint, newest first. Pure filesystem: safe on the GUI thread."""
    out: List[RunInfo] = []
    try:
        entries = sorted(os.scandir(runs_dir), key=lambda e: e.name)
    except OSError:
        return out
    for entry in entries:
        if not entry.is_dir():
            continue
        best, best_t = None, -1.0
        for ck in CHECKPOINT_NAMES:
            p = os.path.join(entry.path, ck)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if st.st_mtime > best_t:
                best, best_t = (p, st.st_size), st.st_mtime
        if best is None:
            continue
        path, size = best
        out.append(RunInfo(name=entry.name, path=entry.path, checkpoint=path,
                           kind="PPO" if os.path.basename(path).startswith("ppo") else "DAgger",
                           mtime=best_t, size_mb=size / 1e6))
    out.sort(key=lambda r: r.mtime, reverse=True)
    return out


@dataclass
class MapCatalog:
    """The base tracks the worker can load, in the project's three groups.

    What changed and why: this used to be five groups of *loader names*, and a loader name is a map
    with a direction, an obstacle family and a placement seed glued to it
    (`real:korea_2026_competition+rlobs213~mir~rev`). Two hundred of those in one list is not a list
    of maps. The groups are now three, they hold **base track ids** (`real/korea26`), and the
    direction / obstacle / seed choices are three little controls next to the list.

    The three groups answer one question -- has the policy seen this floor? -- which is the only
    thing a group label here can honestly claim. It is a claim about the *project's* split, not
    about the checkpoint you happen to have selected: runs here were resumed with different
    `--tracks`, so `GROUP_CAVEAT` says so wherever a label appears.

    `entries` carries the display name, family and note for every id in `groups` *and* for every
    other track in the catalogue, so the search box can reach a map that is in neither split
    (`rt/silverstone`, `gym/levine`) without a group for it.

    `ready` stays False until the worker has replied, so the UI can say "목록 읽는 중" instead of
    pretending the catalog is empty.
    """
    groups: Dict[str, List[str]] = field(default_factory=dict)
    entries: Dict[str, dict] = field(default_factory=dict)
    ready: bool = False
    error: Optional[str] = None

    @property
    def total(self) -> int:
        return len(self.entries) or len({n for names in self.groups.values() for n in names})

    def ids(self) -> List[str]:
        """Every track, group members first (in group order), then the rest of the catalogue."""
        out = [n for g in GROUP_ORDER for n in self.groups.get(g, [])]
        out += [n for g, names in self.groups.items() if g not in GROUP_ORDER for n in names]
        out += [i for i in self.entries if i not in set(out)]
        return list(dict.fromkeys(out))

    def entry(self, track_id: str) -> dict:
        e = self.entries.get(track_id)
        if e:
            return e
        return {"id": track_id, "family": track_id.split("/", 1)[0], "display": track_id,
                "legacy": track_id, "note": "", "obstacles": list(tracks.OBSTACLES), "props": 0}

    def display(self, track_id: str) -> str:
        return self.entry(track_id).get("display") or track_id

    def obstacle_options(self, track_id: str) -> List[str]:
        return list(self.entry(track_id).get("obstacles") or ("",))

    def authored_props(self, track_id: str) -> int:
        """How many obstacles this map's author placed. 0 for every map that is not a scene.

        A scene is read live rather than from the entry: the editor is in the same process, and a
        count that lagged a save would put the wrong number on the 장애물 labels -- which are
        exactly the labels this number exists to make honest.
        """
        if str(track_id).startswith("scene/"):
            n = tracks.scene_props(track_id)
            if n is not None:
                return int(n)
        return int(self.entry(track_id).get("props") or 0)

    def group_of(self, track_id: str) -> Optional[str]:
        for g, names in self.groups.items():
            if track_id in names:
                return g
        return None

    def first(self) -> Optional[str]:
        for g in GROUP_ORDER:
            if self.groups.get(g):
                return self.groups[g][0]
        for names in self.groups.values():
            if names:
                return names[0]
        return None


#: The three groups, their order, the one sentence each has to earn, and the caveat that goes with
#: any of them. Defined in `f1sim.tracks` next to the split rules that generate them -- that module
#: is torch-free for exactly this reason -- and re-exported here because every console module
#: already imports this one.
SCENES_GROUP = tracks.GROUP_SCENES
TRAIN_GROUP = tracks.GROUP_TRAIN
HELDOUT_GROUP = tracks.GROUP_HELDOUT
GROUP_ORDER = list(tracks.GROUP_ORDER)
GROUP_HINT = dict(tracks.GROUP_HINT)
GROUP_CAVEAT = tracks.GROUP_CAVEAT

#: What the search box says it does. The groups are three; the catalogue is larger than the three,
#: and typing is how the rest of it is reached.
SEARCH_HINT = "검색하면 세 그룹 밖의 맵(다른 레이스트랙, gym 맵, 생성 시드)까지 전부 찾습니다."


def list_scenes() -> List[dict]:
    """The editor's saved scenes, newest first -- `f1sim.scene.list_scenes`, re-exported so the GUI
    can fill the `내 환경` group itself. Pure filesystem, torch-free."""
    from ...scene import list_scenes as _list_scenes
    return _list_scenes()


def scene_ids() -> List[str]:
    return [f"scene/{s['name']}" for s in list_scenes()]
