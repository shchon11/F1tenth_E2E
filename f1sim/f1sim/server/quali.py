"""예선 세션(M1): 엔진(감싼 실행기) + 규칙 채점 + 세션 시계(남은 시간, 상자 치우는 시각) + 기록 + 결과.

세션 구성(spec, 앱이 `session.start`로 보냄)
  course   코스 id                 driver  드라이버 id           exec  real | ideal
  physics  물리 프로필 id          rules   규칙 id               duration_s, boxes{source, index|seed, count}, remove_at_s

결과(`result`): 대회 채점 + 바퀴 표 + 접촉 목록 + 표기(실행 방식·물리 프로필·규칙) + 측정(배속·장치).
`/data/tenth/results/<id>.json`(결과)과 `<id>.steps.npz`(제어 주기마다의 기록: 독립 검산용, `verify.py`)를 쓴다.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import layouts as LY
from .library import DATA_ROOT, Library, LibraryError
from .rules import QualiScorer, describe_rules

RESULTS = os.path.join(DATA_ROOT, "results")


class SpecError(ValueError):
    pass


class Progress:
    """코스 중심선 위 위치 s [m]를 연속으로 잰다. 시뮬레이터의 `s`는 한 주기에 1~2 m씩 튀는 곳이 있어(11-28 측정: 60 s에
    0.4 m 넘는 점프 65번) 랩 계시에 쓰지 않는다. 중심선을 5 cm로 다시 나누고, 직전 위치 앞뒤 3 m 안에서 선분에 투영한다."""

    def __init__(self, centerline, step: float = 0.05):
        cl = np.asarray(centerline, float)
        pts = np.vstack([cl, cl[:1]])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        self.L = float(s[-1])
        n = max(16, int(round(self.L / step)))
        q = np.linspace(0.0, self.L, n, endpoint=False)
        self.xy = np.stack([np.interp(q, s, pts[:, 0]), np.interp(q, s, pts[:, 1])], 1)
        self.s = q
        self.n = n
        self.idx: Optional[int] = None
        self.win = int(math.ceil(3.0 / (self.L / n)))

    def locate(self, x: float, y: float) -> float:
        p = np.array([x, y])
        if self.idx is None:
            cand = np.arange(self.n)
        else:
            cand = (self.idx + np.arange(-self.win, self.win + 1)) % self.n
        a = self.xy[cand]
        b = self.xy[(cand + 1) % self.n]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
        proj = a + ab * t[:, None]
        d = np.linalg.norm(proj - p, axis=1)
        j = int(np.argmin(d))
        self.idx = int(cand[j])
        return float((self.s[self.idx] + t[j] * np.linalg.norm(ab[j])) % self.L)


def resolve_spec(lib: Library, spec: dict) -> dict:
    """앱이 보낸 구성을 자료와 맞춰 완성한다(없는 값은 기본값). 틀리면 한국어 문장으로 SpecError."""
    try:
        course = lib.get("course", spec.get("course") or lib.courses[0]["id"])
        driver = lib.get("driver", spec.get("driver") or lib.drivers[0]["id"])
        rules = lib.get("rules", spec.get("rules") or lib.default("rules")["id"])
        physics = lib.get("physics", spec.get("physics") or lib.default("physics")["id"])
    except LibraryError as exc:
        raise SpecError(str(exc)) from exc
    if rules.get("format") != "qualifier":
        raise SpecError(f"'{rules['name']}'는 {rules.get('available_from', '다음 단계')}에서 달릴 수 있습니다. 지금은 예선 형식만 됩니다.")
    exec_mode = spec.get("exec", "real")
    if exec_mode not in ("real", "ideal"):
        raise SpecError(f"실행 방식 {exec_mode!r}를 모릅니다(real 또는 ideal)")
    dur = float(spec.get("duration_s") or rules["duration_s"]["v"])
    bx = dict(rules.get("boxes") or {})
    b = dict(spec.get("boxes") or {})
    count = int(b.get("count", bx.get("count", 0)))
    source = b.get("source") or ("saved" if (course.get("layouts") or {}).get("saved") else
                                 ("random" if (course.get("layouts") or {}).get("random") else "none"))
    if count > 0 and source == "none":
        count = 0
    rm = spec.get("remove_at_s", bx.get("remove_at_s"))
    rm = float(rm) if rm not in (None, "") else None
    return {"course": course, "driver": driver, "rules": rules, "physics": physics, "exec": exec_mode,
            "duration_s": dur, "boxes": {"source": source, "count": count, "index": int(b.get("index", 0)),
                                          "seed": int(b.get("seed", 1))},
            "remove_at_s": rm if (rm is not None and count > 0 and rm < dur) else None}


class Gate:
    """계시선: 코스를 가로지르는 선분(차로 양쪽 벽까지). 실제 대회의 계시 고리처럼 차의 위치가 선분을 지나는 순간을 잰다.
    중심선 투영(진행 거리)은 굽은 곳에서 튀므로(차가 중심선에서 멀면 투영이 한 주기에 1.5 m 넘게 뛴다) 쓰지 않는다."""

    def __init__(self, x: float, y: float, yaw: float, wl: float, wr: float):
        self.c = np.array([x, y], float)
        self.t = np.array([math.cos(yaw), math.sin(yaw)])
        self.n = np.array([-self.t[1], self.t[0]])
        self.wl, self.wr = float(wl), float(wr)

    def cross(self, p0, p1) -> Optional[tuple]:
        """(+1 앞으로 | -1 뒤로, 주기 안 비율 f) 또는 None."""
        d0 = float(np.dot(np.asarray(p0) - self.c, self.t))
        d1 = float(np.dot(np.asarray(p1) - self.c, self.t))
        if (d0 < 0.0) == (d1 < 0.0):
            return None
        f = -d0 / (d1 - d0)
        q = np.asarray(p0) + f * (np.asarray(p1) - np.asarray(p0))
        lat = float(np.dot(q - self.c, self.n))
        if not (-self.wr - 0.15 <= lat <= self.wl + 0.15):
            return None
        return (1 if d1 >= 0.0 else -1), f

    def as_dict(self, k: int, s_rel: float) -> dict:
        yaw = math.atan2(self.t[1], self.t[0])
        return {"k": k, "s": float(s_rel), "x": float(self.c[0]), "y": float(self.c[1]), "yaw": yaw,
                "wl": self.wl, "wr": self.wr}


def lane_half_widths(track, x: float, y: float, yaw: float, step: float = 0.01, max_w: float = 4.0):
    """(왼쪽, 오른쪽) 벽까지 거리 [m]: 그 점에서 진행 방향의 법선을 따라 막힌 칸을 만날 때까지."""
    occ = np.asarray(track.occupancy, bool)
    res = float(track.resolution)
    n = np.array([-math.sin(yaw), math.cos(yaw)])
    out = []
    for sign in (1.0, -1.0):
        w = 0.0
        while w < max_w:
            q = np.array([x, y]) + sign * (w + step) * n
            j = int(math.floor((q[0] - track.origin[0]) / res)); i = int(math.floor((q[1] - track.origin[1]) / res))
            if not (0 <= i < occ.shape[0] and 0 <= j < occ.shape[1]) or occ[i, j]:
                break
            w += step
        out.append(w)
    return out[0], out[1]


def gate_ok(track, x: float, y: float, yaw: float, max_half: float = 3.0) -> Optional[tuple]:
    """그 자리에 계시선을 놓을 수 있으면 (왼쪽, 오른쪽 폭): 선분 양끝이 모두 벽에 닿아야 한다(굽은 곳의 넓은 구석은 안 됨)."""
    wl, wr = lane_half_widths(track, x, y, yaw, max_w=max_half + 0.05)
    if wl >= max_half or wr >= max_half or wl + wr > 3.6:
        return None
    return wl, wr


def place_gate_s(track, target_s: float, box_s: List[float], behind: float, ahead: float, max_shift: float) -> Optional[float]:
    """target_s에서 가장 가까운, (1) 곧은 곳(앞뒤 1.5 m 방향 차이 < 15°) (2) 선분이 차로를 막는 곳
    (3) 상자가 [-behind, +ahead] m 안에 없는 곳의 중심선 s. 없으면 None."""
    cl = np.asarray(track.centerline, float)
    pts = np.vstack([cl, cl[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    sc = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(sc[-1])
    bs = np.asarray(box_s, float)

    def pose(sv):
        sv = sv % L
        j = int(np.searchsorted(sc, sv, side="right") - 1) % len(cl)
        a, b = pts[j], pts[j + 1]
        f = (sv - sc[j]) / max(seg[j], 1e-9)
        return a + f * (b - a), math.atan2(b[1] - a[1], b[0] - a[0])
    for k in range(int(max_shift / 0.25) + 1):
        for sv in ((target_s + k * 0.25), (target_s - k * 0.25)):
            if len(bs):
                d = (bs - sv + L / 2) % L - L / 2
                if ((d > -behind) & (d < ahead)).any():
                    continue
            (x, y), yaw = pose(sv)
            _, y1 = pose(sv - 1.5)
            _, y2 = pose(sv + 1.5)
            if abs((y2 - y1 + math.pi) % (2 * math.pi) - math.pi) > math.radians(15):
                continue
            if gate_ok(track, x, y, yaw) is None:
                continue
            return float(sv % L)
    return None


class QualiSession:
    def __init__(self, lib: Library, spec: dict, device: str, stage: Optional[Callable[[str, str], None]] = None):
        self.lib = lib
        self.spec_in = dict(spec)
        self.r = resolve_spec(lib, spec)
        self.stage = stage or (lambda a, b: None)
        r = self.r
        course, driver = r["course"], r["driver"]
        self.stage("layout", "상자 배치")
        from .. import maps
        track_for_layout = maps.load(maps.smooth_name(course["track"]))
        lay = LY.resolve(course, r["boxes"], track_for_layout) if r["boxes"]["count"] > 0 else \
            {"pieces": [], "source": "none", "label": "상자 없음"}
        self.layout = lay
        # 출발선(= 계시선): 코스 중심선 s = 0에서 가장 가까운, 곧고 선분이 차로를 막으며 상자가 근처(뒤 2.5 m ~ 앞 3.5 m)에
        # 없는 곳. 시험 세트의 상자는 바퀴 전체에 퍼져 있어(64배치 192개) 모든 배치에 맞는 고정선이 없다 -> 배치마다 고른다
        self.track0 = track_for_layout
        self.box_s = LY.centerline_s(track_for_layout, lay["pieces"]) if lay["pieces"] else []
        self.start_s = None
        for behind, ahead in ((2.5, 3.5), (1.5, 2.0), (1.0, 1.0)):
            self.start_s = place_gate_s(track_for_layout, 0.0, self.box_s, behind, ahead, 60.0)
            if self.start_s is not None:
                break
        if self.start_s is None:
            self.start_s = 0.0
        if driver["kind"] == "learned":
            ckpt = os.path.expanduser(driver["checkpoint"]["path"])
            ego = None
        else:
            ckpt = os.path.expanduser(driver.get("host_checkpoint") or "")
            bl = driver.get("baseline") or {}
            st = bl.get("setting") or {}
            opts = {}
            if st.get("from") == "baseline_settings":
                from ..learn import baseline_settings as BS
                found = BS.lookup(st.get("track", ""), bl.get("kind", "ftg"), solo=bool(st.get("solo", True)))
                opts = dict((found or {}).get("options") or {})
            ego = {"kind": bl.get("kind", "ftg"), "options": opts, "tuned": False}
        if not os.path.isfile(ckpt):
            raise SpecError(f"{driver['name']}: 저장 파일이 이 PC에 없습니다")
        from .engine import Engine
        self.stage("engine", "세션 만들기")
        self.engine = Engine(ckpt=ckpt, track=course["track"], device=device, exec_mode=r["exec"], ego=ego,
                             profile=r["physics"].get("profile"), pieces=lay["pieces"],
                             start_behind_m=float((r["rules"].get("start") or {}).get("behind_m", 1.0)),
                             start_s=self.start_s, stage=self.stage)
        self.stage("geometry", "코스 그리기")
        from . import course_geometry as CG
        dd = float((course.get("duct_diameter") or {}).get("v", 0.30))
        self.geom = CG.build(self.engine.track, duct_diameter=dd, n_sectors=int(course.get("sectors", 3)))
        self.L = Progress(self.engine.track.centerline).L
        # 구간 경계: 출발선부터 바퀴를 3등분(코스 기하의 구간 표시도 출발선 기준으로 다시 놓는다)
        nsec = int(course.get("sectors", 3))
        self.sector_s = []
        for k in range(1, nsec):
            want = (self.start_s + self.L * k / nsec) % self.L
            got = place_gate_s(self.track0, want, [], 0.0, 0.0, 5.0)
            self.sector_s.append(((got if got is not None else want) - self.start_s) % self.L)
        self.gates = self._gates()
        self.geom["sectors"] = [g.as_dict(k, ([0.0] + self.sector_s)[k]) for k, g in enumerate(self.gates)]
        self.boxes = LY.describe(lay["pieces"])
        self.id = time.strftime("%Y%m%d-%H%M%S") + f"-{driver['id']}"
        self.check = self.engine.applied_check()
        self.restart()

    def _gates(self) -> List[Gate]:
        """계시선 0(출발·결승) + 구간 경계선들. 중심선 위 점에서 진행 방향에 수직, 양쪽 벽까지."""
        import torch
        tr = self.engine.sim.track
        out = []
        for srel in [0.0] + self.sector_s:
            xy, yaw = tr.pose_at_s(torch.tensor([(self.start_s + srel) % self.L], device=self.engine.sim.device),
                                   torch.zeros(1, dtype=torch.long, device=self.engine.sim.device))
            x, y, a = float(xy[0, 0]), float(xy[0, 1]), float(yaw[0])
            wl, wr = lane_half_widths(self.engine.track, x, y, a)
            out.append(Gate(x, y, a, wl, wr))
        return out

    # ------------------------------------------------------------------ 세션 시계
    def restart(self) -> None:
        self.engine.reset()
        self.prog = Progress(self.engine.track.centerline)
        self.engine.last["s"] = self.prog.locate(self.engine.last["x"], self.engine.last["y"])
        self.scorer = QualiScorer(self.r["rules"], self.L, self.sector_s)
        self.events: List[dict] = []
        self.done = False
        self.removed_at: Optional[float] = None
        self.prev = self.engine.last
        self.log: Dict[str, list] = {k: [] for k in ("t", "x", "y", "yaw", "v", "steer", "cmd_steer", "cmd_v", "s",
                                                     "lap_sim", "counts", "c_kind", "c_sev", "boxes_on")}
        self._log(self.prev)
        self.wall_steps: List[float] = []
        self.sectors_passed = 0          # 이번 바퀴에 지난 구간 경계선 수(바퀴는 모든 경계를 지나야 인정)

    def _log(self, st: dict) -> None:
        for k in self.log:
            if k == "boxes_on":
                self.log[k].append(bool(self.engine.boxes_on))
            elif k == "counts":
                self.log[k].append(np.asarray(st["counts"], np.float32))
            else:
                self.log[k].append(st[k])

    @property
    def t(self) -> float:
        return float(self.engine.last["t"])

    def t_left(self) -> float:
        return max(0.0, self.r["duration_s"] - self.t)

    def step(self) -> List[dict]:
        """한 제어 주기. 새 사건 목록(랩·구간·접촉·상자 치움·끝)을 돌려준다."""
        if self.done:
            return []
        evs: List[dict] = []
        rm = self.r["remove_at_s"]
        if rm is not None and self.engine.boxes_on and self.t >= rm:
            self.engine.remove_boxes()
            self.removed_at = self.t
            evs.append({"type": "boxes_removed", "t": self.t})
        a = time.perf_counter()
        prev = self.engine.last
        st = self.engine.step()
        st["s"] = self.prog.locate(st["x"], st["y"])
        self.wall_steps.append(time.perf_counter() - a)
        dt = st["t"] - prev["t"]
        # 계시선 통과: 차 위치(무게중심)가 이번 주기에 지난 선. 통과 시각 = 주기 안에서 선형 보간
        p0, p1 = (prev["x"], prev["y"]), (st["x"], st["y"])
        hits = []
        for k, g in enumerate(self.gates):
            c = g.cross(p0, p1)
            if c is not None and c[0] > 0:
                hits.append((c[1], k))
        for f, k in sorted(hits):
            tc = prev["t"] + f * dt
            if k == 0:
                if self.scorer.t_lap0 is None:
                    self.scorer.on_cross(tc, self.engine.boxes_on)
                    self.sectors_passed = 0
                    evs.append({"type": "timing_start", "t": tc})
                elif self.sectors_passed >= len(self.sector_s):
                    done = self.scorer.on_cross(tc, self.engine.boxes_on)
                    self.sectors_passed = 0
                    lv = self.scorer.live()
                    evs.append({"type": "lap", "t": tc, "lap": done.n, "time": done.time, "clean": done.clean(self.scorer.breaks_on),
                                "contacts": done.contacts, "hard": done.hard, "sectors": done.sectors,
                                "is_best": abs(done.time - (self.scorer.best_lap or -1)) < 1e-9,
                                "clean_streak": lv["clean_streak"], "best_streak": lv["best_streak"]})
            elif self.scorer.t_lap0 is not None and k == self.sectors_passed + 1:
                v = self.scorer.on_sector(tc)
                self.sectors_passed = k
                if v is not None:
                    best = self.scorer.best_sectors[k - 1] if k - 1 < len(self.scorer.best_sectors) else None
                    evs.append({"type": "sector", "t": tc, "sector": k, "time": v, "best": best})
        # 접촉: 이번 주기에 새로 시작된 것(종류별 세기: 이번 주기의 가장 센 접촉)
        new = st["new"]
        for j, kind in enumerate((1, 2, 3)):
            for _ in range(int(round(new[j]))):
                ev = self.scorer.on_contact(st["t"], kind, float(st["c_sev"]), float(st["c_x"]) or st["x"],
                                            float(st["c_y"]) or st["y"], float(self.sector_s[self.sectors_passed - 1]) if self.sectors_passed else 0.0)
                evs.append(dict(ev, type="contact"))
        self._log(st)
        if self.t >= self.r["duration_s"] - 1e-9:
            self.done = True
            evs.append({"type": "session_end", "t": self.t})
        self.events += evs
        return evs

    # ------------------------------------------------------------------ 앱에 보낼 것
    def labels(self) -> dict:
        r = self.r
        execs = {"real": "실차 방식", "ideal": "이상 실행 (참고용)"}
        drv = r["driver"]
        exec_label = execs[r["exec"]] if drv["kind"] == "learned" else "직접 명령"
        return {"course": r["course"]["name"], "driver": drv["name"], "driver_kind": drv["kind"],
                "driver_summary": drv.get("summary", ""), "exec": r["exec"], "exec_label": exec_label,
                "physics": r["physics"]["name"], "car": r["physics"].get("car_label", r["physics"]["name"]),
                "rules": r["rules"]["name"], "rules_short": r["rules"].get("short", ""),
                "rules_line": describe_rules(r["rules"], r["duration_s"], len(self.boxes), r["remove_at_s"]),
                "layout": self.layout.get("label", ""), "layout_unchecked": bool(self.layout.get("unchecked")),
                "walls": "매끄러운 판" if self.engine.facts()["walls"] == "smooth" else "격자"}

    def session_message(self) -> dict:
        r = self.r
        return {"id": self.id, "labels": self.labels(), "duration_s": r["duration_s"], "remove_at_s": r["remove_at_s"],
                "boxes": self.boxes, "scoring": r["rules"].get("scoring", []), "clean_lap": r["rules"].get("clean_lap", {}),
                "facts": self.engine.facts(), "physics_check": self.check,
                "detail": {"driver": {"run": (r["driver"].get("checkpoint") or {}).get("run", ""),
                                      "update": (r["driver"].get("checkpoint") or {}).get("update"),
                                      "path": os.path.expanduser((r["driver"].get("checkpoint") or {}).get("path", "")
                                                                 or r["driver"].get("host_checkpoint", ""))},
                           "course_track": r["course"]["track"], "profile": r["physics"].get("profile"),
                           "rules_file": r["rules"].get("_file")},
                "spec": {"course": r["course"]["id"], "driver": r["driver"]["id"], "rules": r["rules"]["id"],
                         "physics": r["physics"]["id"], "exec": r["exec"], "duration_s": r["duration_s"],
                         "remove_at_s": r["remove_at_s"], "boxes": r["boxes"]}}

    def track_message(self) -> dict:
        g = self.geom
        tr = self.engine.track
        cl = np.asarray(tr.centerline, np.float32)
        lp = self.engine.sim.cfg.lidar
        vp = self.engine.sim.cfg.vehicle
        a, b = cl[0], cl[1]
        return {"name": self.r["course"]["name"], "course": self.r["course"]["id"], "length_m": float(self.L),
                "hoses": g["hoses"], "walls_poly": g["walls"], "outline": g["outline"], "lane": g["lane"],
                "sectors": g["sectors"], "duct_diameter": g["duct_diameter"], "duct_height": float(tr.duct_height),
                "centerline": cl, "bounds": g["content_bounds"],
                "start": {"x": float(a[0]), "y": float(a[1]), "yaw": float(math.atan2(b[1] - a[1], b[0] - a[0]))},
                "lidar": {"n": int(lp.n_beams), "fov": float(lp.fov), "angle_min": -float(lp.fov) / 2.0,
                          "range_max": float(lp.range_max),
                          "mount": [float(self.engine.sim.P["mount_x"][0]), float(lp.mount_y), float(self.engine.sim.P["mount_z"][0])],
                          "rate": float(lp.rate)},
                "car": {"length": float(vp.length), "width": float(vp.width), "wheelbase": float(vp.lf + vp.lr),
                        "lr": float(vp.lr)},
                "boxes": self.boxes, "control_rate": float(1.0 / self.engine.sim.control_dt)}

    def car_state(self, snap: Optional[dict] = None) -> dict:
        st = self.engine.last
        lv = self.scorer.live()
        lap_t = (st["t"] - self.scorer.t_lap0) if self.scorer.t_lap0 is not None else None
        sec_t = (st["t"] - self.scorer.t_sector0) if self.scorer.t_sector0 is not None else None
        c = {"id": 0, "x": st["x"], "y": st["y"], "yaw": st["yaw"], "v": st["v"], "vy": st["vy"], "wz": st["wz"],
             "steer": st["steer"], "cmd_steer": st["cmd_steer"], "cmd_v": st["cmd_v"], "ax": st["ax"], "ay": st["ay"],
             "s": st["s"], "lap": lv["laps"], "lap_t": lap_t, "sector_t": sec_t, "sector": len(lv["cur_sectors"]) + 1,
             "last_lap": lv["last_lap"], "best_lap": lv["best_lap"], "clean_streak": lv["clean_streak"],
             "best_streak": lv["best_streak"], "cur_clean": lv["cur_clean"], "contacts": lv["contacts_total"],
             "contact": int(st["c_kind"]) > 0, "cur_sectors": lv["cur_sectors"], "best_sectors": lv["best_sectors"],
             "driver": self.r["driver"]["id"]}
        if snap is not None:
            mu = snap.get("P") or {}
            c["roll"] = float(np.asarray(snap.get("roll", [0.0]))[0])
            c["pitch"] = float(np.asarray(snap.get("pitch", [0.0]))[0])
        return c

    def scan_message(self, snap: dict) -> dict:
        P_ = snap.get("P") or {}
        x, y, yaw = self.engine.last["x"], self.engine.last["y"], self.engine.last["yaw"]
        mx, my = float(P_.get("mount_x", 0.0)), float(P_.get("mount_y", 0.0))
        sx = x + mx * math.cos(yaw) - my * math.sin(yaw)
        sy = y + mx * math.sin(yaw) + my * math.cos(yaw)
        rng = np.asarray(snap["scan"], np.float32)
        rng = np.where(np.isfinite(rng), rng, 0.0).astype(np.float32)
        return {"car": 0, "pose": [sx, sy, yaw + float(P_.get("mount_yaw", 0.0))], "ranges": rng,
                "hit": np.asarray(snap["scan_type"]).astype(np.uint8)}

    def plan_message(self, snap: dict) -> Optional[np.ndarray]:
        p = snap.get("plan")
        if p is None:
            return None
        return np.asarray(p, np.float32)

    # ------------------------------------------------------------------ 결과
    def result(self, measured: Optional[dict] = None) -> dict:
        res = self.scorer.result()
        r = self.r
        out = {"schema": "tenth.result/1", "id": self.id, "format": "qualifier", "time_base": "sim",
               "finished": bool(self.done), "t_end": self.t, "labels": self.labels(), "spec": self.session_message()["spec"],
               "detail": self.session_message()["detail"], "boxes": self.boxes, "remove_at_s": r["remove_at_s"],
               "removed_at": self.removed_at, "duration_s": r["duration_s"], "physics_check": self.check,
               "facts": self.engine.facts(), "scoring": r["rules"].get("scoring", []), "measured": measured or {}}
        out.update(res)
        return out

    def save(self, result: dict) -> Dict[str, str]:
        os.makedirs(RESULTS, exist_ok=True)
        base = os.path.join(RESULTS, self.id)
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1, default=_json_default)
        np.savez_compressed(base + ".steps.npz", dt=np.array(float(self.engine.sim.control_dt)), L=np.array(self.L),
                            **{k: np.asarray(v) for k, v in self.log.items()},
                            sector_s=np.asarray(self.sector_s, float), start_s=np.array(self.start_s),
                            gate0=np.array([*self.gates[0].c, math.atan2(self.gates[0].t[1], self.gates[0].t[0]), self.gates[0].wl, self.gates[0].wr]),
                            track=np.array(self.r["course"]["track"]))
        return {"json": base + ".json", "steps": base + ".steps.npz"}

    def close(self) -> None:
        self.engine.close()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)
