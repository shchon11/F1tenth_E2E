"""자료 서비스(M1): 저장소 `library/`의 YAML(코스·규칙·드라이버·물리 프로필)을 읽고 검사해 앱에 보낼 목록을 만든다.

원칙(설계서 §2.1, PM 결정 1·2·6)
- 파이썬 서버만 이 파일들을 읽고 쓴다. 앱(Godot)은 `catalog` 메시지로 받는다.
- 물리·센서 값은 **양(quantity)** 사전 하나로 적는다: {v, u, err | lo/hi, src, ref, note}. 맨 숫자는 거부한다.
- 출처 배지(src): measured 실측 / estimated 추정 / spec 규격 / assumed 가정.
  식별 상태(status): measured 측정됨 / lower_bound 하한만 / not_identified 식별 안 됨.
- 화면에 나가는 이름은 사람이 읽는 이름이다. 실행 이름·절대 경로는 `detail`(상세 보기)에만 넣는다.

파일 형식(스키마 이름 → 필수 키)
  tenth.course/1  : id, name, track, layouts
  tenth.rules/1   : id, name, format, duration_s, boxes, scoring|win
  tenth.driver/1  : id, name, kind(learned|baseline), checkpoint|baseline
  tenth.physics/1 : id, name, profile(f1sim/profiles 이름 또는 null = params.py 기본값)
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
LIBRARY = os.environ.get("TENTH_LIBRARY", os.path.join(REPO, "library"))
DATA_ROOT = os.environ.get("TENTH_DATA", "/data/tenth")

SOURCES = {"measured": "실측", "estimated": "추정", "spec": "규격", "assumed": "가정", "lower_bound": "하한만"}
STATUSES = {"measured": "측정됨", "lower_bound": "하한만", "not_identified": "식별 안 됨", "not_applied": "시뮬에 없음"}
KINDS = {"course": "tenth.course/1", "rules": "tenth.rules/1", "driver": "tenth.driver/1", "physics": "tenth.physics/1"}
FOLDERS = {"course": "courses", "rules": "rules", "driver": "drivers", "physics": "physics"}
REQUIRED = {
    "course": ("id", "name", "track", "layouts"),
    "rules": ("id", "name", "format", "duration_s", "boxes"),
    "driver": ("id", "name", "kind"),
    "physics": ("id", "name", "profile"),
}


class LibraryError(ValueError):
    """자료 파일이 틀림. 문장은 한국어, 파일 이름을 앞에 붙인다."""


def check_quantity(q: Any, where: str) -> List[str]:
    """양 사전 하나를 검사해 문제 문장 목록을 돌려준다(없으면 빈 목록)."""
    if not isinstance(q, dict) or "v" not in q:
        return [f"{where}: 값은 {{v, u, err, src, ...}} 사전이어야 합니다(맨 숫자 거부)"]
    bad = []
    if q.get("src") not in SOURCES:
        bad.append(f"{where}: 출처 src {q.get('src')!r}는 {'/'.join(SOURCES)} 중 하나여야 합니다")
    if "status" in q and q["status"] not in STATUSES:
        bad.append(f"{where}: 식별 상태 {q['status']!r}는 {'/'.join(STATUSES)} 중 하나여야 합니다")
    if q.get("src") == "estimated" and not any(k in q for k in ("err", "lo", "hi", "range")):
        bad.append(f"{where}: 추정 값은 오차(err) 또는 구간(lo/hi)이 있어야 합니다")
    return bad


def _quantities(d: Any, path: str = ""):
    """사전 안에서 양 사전(`v`와 `src`를 가진 것)을 모두 찾는다."""
    if isinstance(d, dict):
        if "v" in d and "src" in d:
            yield path, d
            return
        for k, v in d.items():
            yield from _quantities(v, f"{path}.{k}" if path else str(k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _quantities(v, f"{path}[{i}]")


def validate(kind: str, doc: dict, fname: str = "?") -> None:
    bad = []
    if doc.get("schema") != KINDS[kind]:
        bad.append(f"schema가 {KINDS[kind]!r}이 아닙니다({doc.get('schema')!r})")
    for k in REQUIRED[kind]:
        if k not in doc:
            bad.append(f"필수 키 {k!r}가 없습니다")
    for p, q in _quantities(doc):
        bad += check_quantity(q, p)
    if kind == "course":
        d = doc.get("duct_diameter")
        if d is not None:
            bad += check_quantity(d, "duct_diameter")
    if kind == "rules":
        if not isinstance(doc.get("duration_s"), dict) or "v" not in doc["duration_s"]:
            bad.append("duration_s는 {v, choices} 사전이어야 합니다")
    if kind == "driver" and doc.get("kind") not in ("learned", "baseline"):
        bad.append(f"드라이버 종류 {doc.get('kind')!r}는 learned 또는 baseline이어야 합니다")
    if kind == "driver" and doc.get("kind") == "learned" and not (doc.get("checkpoint") or {}).get("path"):
        bad.append("학습된 드라이버에 checkpoint.path가 없습니다")
    if bad:
        raise LibraryError(f"{fname}: " + "; ".join(bad))


def load_dir(kind: str, root: str = LIBRARY) -> List[dict]:
    out = []
    for fn in sorted(glob.glob(os.path.join(root, FOLDERS[kind], f"*.{kind}.yaml"))):
        with open(fn, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        validate(kind, doc, os.path.relpath(fn, REPO))
        doc["_file"] = os.path.relpath(fn, REPO)
        out.append(doc)
    out.sort(key=lambda d: (d.get("order", 99), d["id"]))
    ids = [d["id"] for d in out]
    if len(set(ids)) != len(ids):
        raise LibraryError(f"{FOLDERS[kind]}: id가 겹칩니다 {ids}")
    return out


def dump(doc: dict) -> str:
    """서버가 쓸 때는 키 순서를 고정한다(git diff가 작게). `_`로 시작하는 키는 쓰지 않는다."""
    clean = {k: v for k, v in doc.items() if not k.startswith("_")}
    return yaml.safe_dump(clean, allow_unicode=True, sort_keys=False, width=120)


# ------------------------------------------------------------------ 물리 프로필 요약(출처 배지)
def profile_summary(profile_name: Optional[str]) -> dict:
    """f1sim.profiles의 값마다 출처·식별 상태를 세고, 화면에 보일 핵심 값 몇 개를 고른다."""
    if not profile_name:
        return {"counts": {}, "status": {}, "rows": [], "n": 0,
                "note": "params.py 기본값. 값마다 출처가 기록돼 있지 않습니다(코드 주석에만 있음)."}
    from ..profiles.loader import GROUPS, load_profile, overrides
    prof = load_profile(profile_name)
    counts: Dict[str, int] = {}
    status: Dict[str, int] = {}
    rows = []
    for grp in GROUPS:
        for k, e in (prof.get(grp) or {}).items():
            src, st = e.get("source"), e.get("status")
            badge = "lower_bound" if st == "lower_bound" else src
            counts[badge] = counts.get(badge, 0) + 1
            status[st] = status.get(st, 0) + 1
            rows.append({"key": f"{grp}.{k}", "v": e.get("value"), "u": e.get("unit", ""), "err": e.get("err"),
                         "src": src, "badge": badge, "status": st, "tier": e.get("tier", ""),
                         "note": e.get("note", ""), "applied": e.get("apply") is not False and st != "not_applied"})
    _, skipped = overrides(prof)
    head_keys = {"vehicle.mu": "그립(최대 마찰)", "actuator.cmd_delay": "명령 지연", "actuator.servo_tau": "조향 응답",
                 "vehicle.a_max": "가속 한계", "vehicle.a_brake": "제동 한계", "vehicle.m": "질량"}
    head = []
    for r in rows:
        if r["key"] in head_keys:
            head.append(dict(r, label=head_keys[r["key"]]))
    return {"counts": counts, "status": status, "rows": rows, "n": len(rows), "head": head,
            "unapplied": skipped, "path": prof.get("_path", "")}


# ------------------------------------------------------------------ 목록
class Library:
    def __init__(self, root: str = LIBRARY):
        self.root = root
        self.courses = load_dir("course", root)
        self.rules = load_dir("rules", root)
        self.drivers = load_dir("driver", root)
        self.physics = load_dir("physics", root)

    def get(self, kind: str, ident: str) -> dict:
        lst = {"course": self.courses, "rules": self.rules, "driver": self.drivers, "physics": self.physics}[kind]
        for d in lst:
            if d["id"] == ident:
                return d
        raise LibraryError(f"{FOLDERS[kind]}에 {ident!r}가 없습니다 (있는 것: {', '.join(x['id'] for x in lst)})")

    def default(self, kind: str) -> dict:
        lst = {"rules": self.rules, "physics": self.physics, "course": self.courses, "driver": self.drivers}[kind]
        for d in lst:
            if d.get("default"):
                return d
        return lst[0]

    # 앱에 보내는 목록: 사람용 이름·요약, 상세 보기용 원래 식별자(detail)
    def catalog(self) -> dict:
        from . import layouts as LY
        courses = []
        for c in self.courses:
            lay = c.get("layouts") or {}
            saved = lay.get("saved")
            n_saved = LY.bank_count(saved["bank"]) if saved else 0
            courses.append({
                "id": c["id"], "name": c["name"], "summary": c.get("summary", ""),
                "walls": c.get("walls", "smooth"), "sectors": int(c.get("sectors", 3)),
                "saved_layouts": n_saved, "saved_name": (saved or {}).get("name", ""),
                "saved_note": (saved or {}).get("note", ""),
                "random_layouts": bool(lay.get("random")), "random_note": (lay.get("random") or {}).get("note", ""),
                "no_boxes_reason": lay.get("none_reason", ""),
                "trained_here": list(c.get("trained_here") or []),
                "detail": {"track": c["track"], "file": c["_file"], "bank": (saved or {}).get("bank", "")},
            })
        drivers = []
        for d in self.drivers:
            ck = d.get("checkpoint") or {}
            path = os.path.expanduser(ck.get("path") or d.get("host_checkpoint") or "")
            ok = os.path.isfile(path) if path else False
            drivers.append({
                "id": d["id"], "name": d["name"], "kind": d["kind"],
                "kind_label": "학습된 드라이버" if d["kind"] == "learned" else "기준선",
                "summary": d.get("summary", ""), "note": d.get("note", ""),
                "output": (d.get("output") or {}).get("kind", "direct"),
                "trained_on": list(d.get("trained_on") or []),
                "available": ok, "why_not": "" if ok else "저장 파일이 이 PC에 없습니다",
                "detail": {"run": ck.get("run", ""), "update": ck.get("update"), "path": path,
                           "sha256": ck.get("sha256", ""), "file": d["_file"],
                           "baseline": d.get("baseline")},
            })
        rules = []
        for r in self.rules:
            rules.append({
                "id": r["id"], "name": r["name"], "short": r.get("short", r["name"]), "format": r["format"],
                "summary": r.get("summary", ""), "available": r.get("available_from") in (None, "M1"),
                "available_from": r.get("available_from", ""),
                "duration_s": r["duration_s"], "boxes": r["boxes"], "start": r.get("start", {}),
                "clean_lap": r.get("clean_lap", {}), "scoring": r.get("scoring", []), "win": r.get("win"),
                "source": r.get("source", {}), "default": bool(r.get("default")),
                "detail": {"file": r["_file"]},
            })
        physics = []
        for p in self.physics:
            s = profile_summary(p.get("profile"))
            physics.append({
                "id": p["id"], "name": p["name"], "car_label": p.get("car_label", p["name"]),
                "summary": p.get("summary", ""), "default": bool(p.get("default")),
                "counts": s["counts"], "status": s["status"], "n": s["n"], "head": s.get("head", []),
                "note": s.get("note", ""),
                "detail": {"profile": p.get("profile"), "file": p["_file"], "ref": p.get("ref", ""),
                           "unapplied": s.get("unapplied", [])},
            })
        execs = [
            {"id": "real", "name": "실차 방식", "long": "실차 방식", "default": True,
             "summary": "드라이버가 낸 궤적을 실차와 같은 추종기(지도 기반 추종, MAP)가 따라가고, 차는 타이어·지연·서보가 있는 동역학으로 움직입니다."},
            {"id": "ideal", "name": "이상 실행 (참고용)", "long": "이상 실행 (참고용)", "default": False,
             "reference_only": True,
             "summary": "궤적을 계획대로 정확히 달립니다. 훈련 조건이며 실차에서는 일어나지 않으므로 참고용입니다."},
        ]
        return {"courses": courses, "drivers": drivers, "rules": rules, "physics": physics, "execs": execs}
