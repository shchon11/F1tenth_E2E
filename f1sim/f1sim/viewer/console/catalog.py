"""What the GUI can discover on its own, without importing torch.

The console process deliberately never imports torch: the first window has to be on screen in
well under a second, and `import torch` alone is seconds. Runs are a directory listing, so the GUI
reads them itself and the picker is populated on the first paint. Map names need `f1sim.maps`
(which pulls in torch through `track.py`), so those come from the worker over the control channel
and the list fills in when they arrive.

`RUNS_DIR` is duplicated from `f1sim.learn.common` rather than imported for exactly that reason;
`check_runs_dir_matches()` in the worker asserts the two agree, so the copy cannot drift silently.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    """Named map groups, delivered by the worker.

    The groups come straight from the project's own split constants in `f1sim.learn.common`
    (`EVAL_TRACKS`, `EVAL_OBSTACLE_TRACKS`, `TRAIN_TRACKS`). They are worth showing because a map
    from the evaluation split and one from the training split answer different questions, and a
    flat list of several hundred names makes it easy to read an answer off the wrong one.

    What they are *not* is a statement about the checkpoint you happen to have selected. Runs in
    this project were resumed with different splits and some W&B track metadata is broken, so the
    only thing the label can honestly claim is "this is the project's default split", never "this
    policy has never seen this map". The UI names them accordingly and says so in the tooltip.

    `ready` stays False until the worker has replied, so the UI can say "목록 읽는 중" instead of
    pretending the catalog is empty.
    """
    groups: Dict[str, List[str]] = field(default_factory=dict)
    ready: bool = False
    error: Optional[str] = None

    @property
    def total(self) -> int:
        seen = set()
        for names in self.groups.values():
            seen.update(names)
        return len(seen)

    def group_of(self, name: str) -> Optional[str]:
        for g, names in self.groups.items():
            if name in names:
                return g
        return None

    def first(self) -> Optional[str]:
        for names in self.groups.values():
            if names:
                return names[0]
        return None


#: Group order and the one-line explanation shown under each. Order matters: the first group is
#: the default selection. The names say "프로젝트 정의" rather than "미학습" on purpose -- see the
#: MapCatalog docstring. Keep these in sync with `sim_worker.map_catalog()`.
GROUP_ORDER = [
    "기본 평가셋",
    "장애물 (상자·궤짝·드럼)",
    "기본 학습셋",
    "이전 실험 재현 (격자 장애물)",
    "전체 카탈로그",
]
GROUP_HINT = {
    "기본 평가셋": "프로젝트 정의 평가 분할 (common.EVAL_TRACKS)",
    "장애물 (상자·궤짝·드럼)": ("평가 맵에 상자·나무궤짝·드럼 같은 입체 장애물을 놓은 변형입니다. "
                        "차가 실제로 부딪히고 LiDAR 에도 잡힙니다. 이름 뒤 숫자는 배치 seed 입니다."),
    "기본 학습셋": "프로젝트 정의 학습 분할 (common.TRAIN_TRACKS)",
    "이전 실험 재현 (격자 장애물)": ("이전 실험이 학습·평가에 쓰던 장애물 세트입니다 "
                            "(common.EVAL_OBSTACLE_TRACKS). 점유 격자에 찍어 넣는 방식이라 "
                            "높이가 하나인 회전 사각형이고 윗면이 없습니다 — 위 입체 장애물과 "
                            "다른 것입니다. 예전 결과를 재현할 때 쓰세요."),
    "전체 카탈로그": "로드 가능한 모든 이름",
}
#: Shown wherever a group label appears. The split is a project convention, not a per-checkpoint
#: fact, and the UI must not let the two be confused.
GROUP_CAVEAT = ("이 분류는 프로젝트가 정의한 기본 분할입니다. 선택한 체크포인트가 실제로 어떤 맵으로 "
                "학습됐는지는 해당 run의 학습 manifest를 봐야 알 수 있습니다.")
