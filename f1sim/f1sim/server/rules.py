"""대회 형식 채점(M1: 예선). 시뮬 시간 기준(설계서 §0-5): 서버가 실시간보다 느려도 결과는 같다.

예선(kr-event-quali, roboracer-2025-quali)
  바퀴     출발선을 처음 지날 때부터 센다(`lap_timing.from: first_crossing`). 통과 시각은 한 제어 주기(25 ms) 안에서
           선형 보간한다.
  무충돌   그 바퀴 동안 규칙이 세는 접촉(`clean_lap.kinds`: 벽·상자)이 하나도 없음(`breaks_on: any`).
           `breaks_on: hard`면 `hard_mps` 이상만 셈. 세게 닿음(≥ hard_mps)은 결과에 따로 적는다.
  점수     ① 무충돌 연속 바퀴의 최댓값(클수록 좋음) ② 최고 랩(작을수록 좋음). 감사 C의 `comp_metrics.py`와
           `f1sim.compbench.score`의 `streak` 규칙(접촉 바퀴에서 0으로 끊김)과 같다.
  끝       세션 시간이 다 되면 끝. 그때 달리던 바퀴는 완주하지 않았으므로 세지 않는다.

같은 기록으로 독립 계산해 대조하는 것은 `verify.py`(기록 파일 -> compbench 채점 함수).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

KIND_NAMES = {1: "wall", 2: "box", 3: "car"}
KIND_LABEL = {"wall": "벽", "box": "상자", "car": "차"}


def severity_label(sev: float, hard_mps: float) -> str:
    return "세게 닿음" if sev >= hard_mps else "약하게 닿음"


def streak(bad: List[float]) -> int:
    """연속으로 0인 항목 수의 최댓값(compbench.score.streak과 같은 규칙)."""
    try:
        from ..compbench.score import streak as _cb_streak      # 다른 작업자의 채점 함수(있으면 그대로 쓴다)
        return int(_cb_streak(bad))
    except Exception:
        best = cur = 0
        for x in bad:
            cur = 0 if x > 0 else cur + 1
            best = max(best, cur)
        return best


@dataclass
class Lap:
    n: int
    t_start: float
    t_end: float
    contacts: int = 0             # 규칙이 세는 접촉
    hard: int = 0                 # 그중 세게 닿음
    sectors: List[float] = field(default_factory=list)
    boxes_on: bool = True         # 이 바퀴를 시작할 때 상자가 있었나

    @property
    def time(self) -> float:
        return self.t_end - self.t_start

    def clean(self, breaks_on: str) -> bool:
        return (self.hard if breaks_on == "hard" else self.contacts) == 0


class QualiScorer:
    """한 차의 예선 채점. 서버가 매 제어 주기에 `on_step`을 부르고, 끝에 `result()`."""

    def __init__(self, rules: dict, lap_length: float, sector_s: List[float]):
        self.rules = rules
        cl = rules.get("clean_lap") or {}
        self.breaks_on = str(cl.get("breaks_on", "any"))
        self.kinds = set(cl.get("kinds", ["wall", "box"]))
        self.hard_mps = float(cl.get("hard_mps", 1.0))
        self.L = float(lap_length)
        self.sector_s = list(sector_s)          # 구간 경계(코스 s, 출발선 = 0)
        self.laps: List[Lap] = []
        self.contacts: List[dict] = []          # 모든 접촉(규칙이 안 세는 것도: counted 표시)
        self.t_lap0: Optional[float] = None     # 지금 바퀴 시작 시각(첫 통과 전 None)
        self.cur_contacts = 0
        self.cur_hard = 0
        self.cur_sectors: List[float] = []
        self.cur_boxes_on = True
        self.t_sector0: Optional[float] = None
        self.best_lap: Optional[float] = None
        self.best_sectors: List[Optional[float]] = [None] * (len(sector_s) + 1)
        self.clean_streak = 0
        self.best_streak = 0
        self.best_streak_end_lap = 0

    # ------------------------------------------------------------------ 입력
    def on_cross(self, t: float, boxes_on: bool) -> Optional[Lap]:
        """출발선 통과(시뮬 시각, 보간됨). 직전 바퀴가 있으면 확정해서 돌려준다."""
        done = None
        if self.t_lap0 is not None:
            secs = self.cur_sectors + [t - (self.t_sector0 if self.t_sector0 is not None else self.t_lap0)]
            done = Lap(len(self.laps) + 1, self.t_lap0, t, self.cur_contacts, self.cur_hard, secs, self.cur_boxes_on)
            self.laps.append(done)
            if done.clean(self.breaks_on):
                self.clean_streak += 1
            else:
                self.clean_streak = 0
            if self.clean_streak > self.best_streak:
                self.best_streak = self.clean_streak
                self.best_streak_end_lap = done.n
            if self.best_lap is None or done.time < self.best_lap:
                self.best_lap = done.time
            for i, v in enumerate(secs):
                if i < len(self.best_sectors) and (self.best_sectors[i] is None or v < self.best_sectors[i]):
                    self.best_sectors[i] = v
        self.t_lap0 = t
        self.t_sector0 = t
        self.cur_contacts = 0
        self.cur_hard = 0
        self.cur_sectors = []
        self.cur_boxes_on = boxes_on
        return done

    def on_sector(self, t: float) -> Optional[float]:
        if self.t_lap0 is None or self.t_sector0 is None:
            return None
        v = t - self.t_sector0
        self.cur_sectors.append(v)
        self.t_sector0 = t
        return v

    def on_contact(self, t: float, kind: int, severity: float, x: float, y: float, s: float) -> dict:
        name = KIND_NAMES.get(int(kind), "wall")
        counted = name in self.kinds
        hard = severity >= self.hard_mps
        sector = self.sector_of(s)
        ev = {"t": float(t), "kind": name, "kind_label": KIND_LABEL.get(name, name), "severity": float(severity),
              "hard": bool(hard), "severity_label": severity_label(severity, self.hard_mps), "x": float(x), "y": float(y),
              "lap": len(self.laps) + 1 if self.t_lap0 is not None else 0, "sector": sector, "counted": counted}
        self.contacts.append(ev)
        if counted:
            self.cur_contacts += 1
            self.cur_hard += int(hard)
            if self.breaks_on == "any" or hard:
                self.clean_streak = 0          # 지금 바퀴는 이미 무충돌이 아님 -> 이어가는 중인 연속도 끊김
        return ev

    def sector_of(self, s: float) -> int:
        k = 1
        for b in self.sector_s:
            if s >= b:
                k += 1
        return k

    # ------------------------------------------------------------------ 출력
    def live(self) -> dict:
        """주행 HUD용 요약(매 상태 프레임)."""
        cur_clean = self.t_lap0 is not None and (self.cur_hard if self.breaks_on == "hard" else self.cur_contacts) == 0
        return {"laps": len(self.laps), "clean_streak": self.clean_streak, "best_streak": self.best_streak,
                "best_lap": self.best_lap, "last_lap": self.laps[-1].time if self.laps else None,
                "cur_clean": bool(cur_clean), "cur_contacts": self.cur_contacts,
                "best_sectors": self.best_sectors, "cur_sectors": self.cur_sectors,
                "contacts_total": sum(1 for c in self.contacts if c["counted"])}

    def lap_rows(self) -> List[dict]:
        rows = []
        for lp in self.laps:
            rows.append({"n": lp.n, "time": lp.time, "t_start": lp.t_start, "t_end": lp.t_end,
                         "clean": lp.clean(self.breaks_on), "contacts": lp.contacts, "hard": lp.hard,
                         "sectors": lp.sectors, "boxes_on": lp.boxes_on,
                         "is_best": self.best_lap is not None and abs(lp.time - self.best_lap) < 1e-9})
        return rows

    def result(self) -> dict:
        rows = self.lap_rows()
        clean_rows = [r for r in rows if r["clean"]]
        before = [r["time"] for r in rows if r["boxes_on"]]
        after = [r["time"] for r in rows if not r["boxes_on"]]
        return {
            "score": {"best_clean_streak": int(self.best_streak), "best_lap": self.best_lap},
            "streak_check": streak([0.0 if r["clean"] else 1.0 for r in rows]),
            "best_streak_end_lap": self.best_streak_end_lap,
            "laps": rows, "n_laps": len(rows), "n_clean": len(clean_rows),
            "best_clean_lap": min((r["time"] for r in clean_rows), default=None),
            "best_lap_boxes": min(before, default=None), "best_lap_no_boxes": min(after, default=None),
            "contacts": self.contacts,
            "contacts_counted": sum(1 for c in self.contacts if c["counted"]),
            "contacts_hard": sum(1 for c in self.contacts if c["counted"] and c["hard"]),
            "breaks_on": self.breaks_on, "hard_mps": self.hard_mps,
        }


def rank(results: List[dict], scoring: List[dict]) -> List[int]:
    """여러 결과의 순위(0부터). 점수 항목 순서대로 비교, 없는 값은 가장 나쁨. 완전히 같으면 같은 순위."""
    def key(r):
        k = []
        for item in scoring:
            v = (r.get("score") or {}).get(item["key"])
            if v is None:
                v = math.inf if item.get("order") == "asc" else -math.inf
            k.append(v if item.get("order") == "asc" else -v)
        return tuple(k)
    keys = [key(r) for r in results]
    order = sorted(range(len(results)), key=lambda i: keys[i])
    ranks = [0] * len(results)
    for pos, i in enumerate(order):
        if pos > 0 and keys[i] == keys[order[pos - 1]]:
            ranks[i] = ranks[order[pos - 1]]
        else:
            ranks[i] = pos
    return ranks


def describe_rules(rules: dict, duration_s: float, n_boxes: int, remove_at: Optional[float]) -> str:
    box = "상자 없음" if n_boxes <= 0 else (f"상자 {n_boxes}개 · {remove_at:g}초에 치움" if remove_at and remove_at < duration_s
                                         else f"상자 {n_boxes}개")
    return f"{rules.get('name')} · {duration_s:g}초 · {box}"
