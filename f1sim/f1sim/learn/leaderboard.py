"""Offline leaderboard report from already-measured benchmark results.

This module renders; it never measures. Every cohort in the manifest is pushed through
`benchmark.report.validate_results` with the full strict argument set -- suite object, declared
cells, declared trial counts, declared system set, roster and freeze -- and one cohort at a time, so
two sets of results taken under different code can never end up in one ranking. Nothing here
weakens that gate: the only thing added is presentation.

Cohorts are separate on purpose. The recipe study and the initial study were recorded under
different benchmark source digests, and `validate_results` already refuses to pool rows across
digests; the leaderboard keeps that separation visible by ranking inside a cohort and never across
cohorts.

Rank metrics are computed from exact numerators and denominators derived from the validated outcome
arrays, never from the rounded values the table prints. Paired pace is descriptive and carries its
own n: it compares lap times only on the trials where both the system and the cohort's reference
system completed, which is a different sample for every pair.

Evidence is recorded as a path plus the sha256 of the bytes on disk. Validation stamps derived
counts onto the rows it checks, so a validated row is no longer the file's row; it is never
serialised as original evidence.

    cd f1sim
    python -m f1sim.learn.leaderboard --manifest ../docs/leaderboard/data/manifest.json \
        --out-dir ../docs/leaderboard
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

from .benchmark import report as report_mod
from .benchmark import roster as roster_mod
from .benchmark import suite as suite_mod

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard_assets")

SCHEMA = "f1sim.leaderboard/1"


class LeaderboardError(ValueError):
    """A manifest or a result set that must not be rendered."""


#: The six rankable metrics, in table order. `better` is the meaningful direction and is shown in
#: the header; `pct` cells also print the exact numerator and denominator they were derived from.
#: There is deliberately no composite score: the categories have different units and different
#: denominators, and an average of them would be an invented number.
METRICS = (
    {"key": "solo_completion", "label": "Solo completion", "unit": "%", "better": "up",
     "kind": "pct", "source": "suite S, all declared friction levels",
     "description": "Trials completed over all solo cells, pooled over the three declared "
                    "friction levels."},
    {"key": "low_mu_completion", "label": "Low-mu completion", "unit": "%", "better": "up",
     "kind": "pct", "source": "suite S, minimum declared solo mu",
     "description": "Trials completed at the minimum declared solo friction level. This is "
                    "completion at that level, not friction-estimation accuracy."},
    {"key": "avoidance", "label": "Avoidance", "unit": "%", "better": "up", "kind": "pct",
     "source": "suite A",
     "description": "Obstacle approaches cleared over pre-validated avoidance trials."},
    {"key": "overtaking", "label": "Overtaking", "unit": "%", "better": "up", "kind": "pct",
     "source": "suite O",
     "description": "Passes held to the end of the race over pre-validated race trials."},
    {"key": "collisions_km", "label": "Collisions/km", "unit": "/km", "better": "down",
     "kind": "rate", "source": "all suites",
     "description": "Collisions per kilometre travelled, pooled over every suite."},
    {"key": "slip_km", "label": "Large-slip s/km", "unit": "s/km", "better": "down",
     "kind": "rate", "source": "all suites",
     "description": "Seconds spent above the large-slip threshold per kilometre travelled, "
                    "pooled over every suite."},
)

#: Aggregate column names this module reads. Kept in one place so a rename in `report.aggregate`
#: fails loudly here instead of silently dropping a metric.
AGG_COLUMNS = {"driving": "completion ↑", "surface": "completion ↑",
               "avoidance": "cleared/pre-validated ↑", "overtaking": "passes held/race ↑",
               "collisions": "collisions/km ↓", "slip": "large-slip s/km ↓"}

#: Percentages are derived twice -- from the validated outcome arrays and by `report.aggregate` --
#: and must agree to the last bit of the division. A disagreement means the two are not measuring
#: the same thing and the report is refused rather than printed.
PARITY_TOL = 1e-12

METHOD = (
    "Each cohort's raw per-cell records are read and pushed through "
    "f1sim.learn.benchmark.report.validate_results with the frozen suite object, the declared cell "
    "grid, the declared trial counts, the declared system set and the roster. A missing cell, a "
    "duplicate, a short denominator, an edited row, an unpinned checkpoint, a mixed source digest "
    "or an unpaired start refuses the whole cohort, and nothing partial is written.",
    "Metrics come from report.aggregate over the validated cells. Percentages are additionally "
    "derived from the validated outcome arrays as exact counts, and the two must agree exactly or "
    "the report is refused.",
    "Ranks are competition ranks over the exact values, per cohort and per metric: equal exact "
    "values share a rank and the next rank skips. Rounded display values are never ranked. A "
    "missing measurement is N/A with its reason, carries no rank and sorts last.",
    "Paired pace is the mean solo lap-time difference against the cohort reference over the trials "
    "both systems completed, reported with its n. It is descriptive and never ranked.",
    "There is no composite or overall score: the six metrics have different units and different "
    "denominators.",
)

LIMITS = (
    "A measured, representative development subset: three reused development maps, two seeds, "
    "eight trials per cell.",
    "Static per-episode friction. Each episode runs at one fixed friction level; no level changes "
    "inside an episode.",
    "No unseen-map, generalisation or on-car claim attaches to any number here. All maps were "
    "already in use during development.",
    "Low-mu completion is completion at the minimum declared solo friction level. It is not a "
    "measurement of friction-estimation accuracy.",
    "Paired pace is descriptive, not a rank metric: it is conditional on the trials where both "
    "systems completed, so its sample differs for every pair.",
    "Cohorts were recorded under different benchmark source digests. They are never ranked "
    "together.",
)


# --------------------------------------------------------------------------- manifest


def _resolve(base: str, rel: str) -> str:
    return os.path.normpath(os.path.join(base, rel))


def load_manifest(path: str) -> tuple[dict, str]:
    with open(path) as fh:
        man = json.load(fh)
    if not isinstance(man, dict):
        raise LeaderboardError(f"{path}: manifest is not an object")
    cohorts = man.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        raise LeaderboardError(f"{path}: manifest declares no cohorts")
    ids = []
    for c in cohorts:
        for k in ("id", "label", "suite", "roster", "results", "reference_system"):
            if not c.get(k):
                raise LeaderboardError(f"{path}: cohort {c.get('id')!r} has no {k}")
        if not isinstance(c["results"], list):
            raise LeaderboardError(f"{path}: cohort {c['id']}: results must be a list")
        ids.append(c["id"])
    if len(set(ids)) != len(ids):
        raise LeaderboardError(f"{path}: duplicate cohort ids in {ids}")
    default = man.get("default_cohort") or ids[0]
    if default not in ids:
        raise LeaderboardError(f"{path}: default_cohort {default!r} is not one of {ids}")
    man["default_cohort"] = default
    return man, os.path.dirname(os.path.abspath(path))


def _read_cells(base: str, rel_paths: list, out_dir: str | None) -> tuple[list, list]:
    """Raw cells plus the evidence record for the files they came from.

    `path` stays the manifest-relative path, which is the provenance the manifest declared. `href`
    is the same file rebased on the output directory, because the report links from there and a
    manifest-relative path would be a broken link in the rendered page.
    """
    cells, evidence = [], []
    for rel in rel_paths:
        path = _resolve(base, rel)
        with open(path, "rb") as fh:
            blob = fh.read()
        got = [json.loads(ln) for ln in blob.decode().splitlines() if ln.strip()]
        if not got:
            raise LeaderboardError(f"{path}: no result cells")
        cells += got
        ev = {"path": rel, "sha256": hashlib.sha256(blob).hexdigest(), "n_cells": len(got)}
        if out_dir:
            ev["href"] = os.path.relpath(path, os.path.abspath(out_dir)).replace(os.sep, "/")
        evidence.append(ev)
    return cells, evidence


# --------------------------------------------------------------------------- metrics


def _derived(cell: dict) -> dict:
    d = cell.get("_derived") or (cell.get("result") or {}).get("_derived")
    if d is None:                                     # only reachable if validation was skipped
        raise LeaderboardError(f"{cell.get('system_id')} {cell.get('cell_id')}: not validated")
    return d


def exact_counts(cells: list, low_mu: float) -> dict:
    """Numerator and denominator per system, straight from the validated outcome arrays.

    The counts the table prints are these, not a rate read back off a rounded percentage.
    """
    out: dict = {}
    for c in cells:
        d = _derived(c)
        b = out.setdefault(c["system_id"],
                           {"S": [0, 0], "A": [0, 0], "O": [0, 0], "low": [0, 0], "fail": {}})
        slot = b.get(c["suite"])
        if slot is None:
            raise LeaderboardError(f"{c['system_id']}: unknown suite {c['suite']!r}")
        slot[0] += d["successes"]
        slot[1] += d["n"]
        if c["suite"] == "S" and float(c["mu"]) == float(low_mu):
            b["low"][0] += d["successes"]
            b["low"][1] += d["n"]
        for reason, n in d["failures"].items():
            key = str(reason) if reason is not None else "unspecified"
            b["fail"][key] = b["fail"].get(key, 0) + n
    return out


def _agg_index(agg: dict, low_mu: float) -> dict:
    """`report.aggregate` output, keyed by system id, for the six leaderboard metrics."""
    idx: dict = {}

    def slot(sid):
        return idx.setdefault(sid, {})

    for r in agg.get("driving", []):
        slot(r["system_id"])["solo_completion"] = r["values"][AGG_COLUMNS["driving"]]
    for r in agg.get("avoidance", []):
        slot(r["system_id"])["avoidance"] = r["values"][AGG_COLUMNS["avoidance"]]
    for r in agg.get("overtaking", []):
        slot(r["system_id"])["overtaking"] = r["values"][AGG_COLUMNS["overtaking"]]
    for r in agg.get("stability", []):
        s = slot(r["system_id"])
        s["collisions_km"] = r["values"][AGG_COLUMNS["collisions"]]
        s["slip_km"] = r["values"][AGG_COLUMNS["slip"]]
        s["distance_km"] = r["values"]["distance km"]
    for r in agg.get("surface", []):
        # `aggregate` labels surface rows "<arm> · mu=<level>"; the level is parsed back rather
        # than matched as a string so a formatting change cannot silently drop the low-mu metric.
        tail = str(r["runtime"]).rsplit("mu=", 1)
        if len(tail) == 2 and float(tail[1]) == float(low_mu):
            slot(r["system_id"])["low_mu_completion"] = r["values"][AGG_COLUMNS["surface"]]
    return idx


def _round(value, places: int) -> str:
    """Half-up decimal rounding, computed once so every view prints the same string.

    Python's `%.1f` and JavaScript's `toFixed(1)` disagree on an exact half -- 56.25 prints as
    56.2 and 56.3 respectively -- so the display string is produced here and the page reuses it
    rather than re-formatting the number. Ranking never touches these strings.
    """
    q = Decimal(1).scaleb(-places)
    return str(Decimal(value).quantize(q, rounding=ROUND_HALF_UP))


def _pct_display(num: int, den: int) -> str:
    return _round(Decimal(num) * 100 / Decimal(den), 1) + "%"


def _pct(num: int, den: int, agg_cell, tag: str) -> dict:
    """An exact rate, cross-checked against the pooled figure `report.aggregate` computed."""
    if not den:
        return {"value": None, "num": num, "den": den, "rank": None,
                "reason": "no pre-validated trials"}
    value = num / den
    got = agg_cell.get("value") if isinstance(agg_cell, dict) else agg_cell
    if got is None:
        return {"value": None, "num": num, "den": den, "rank": None,
                "reason": (agg_cell or {}).get("reason") or "not aggregated"}
    if abs(float(got) - value) > PARITY_TOL:
        raise LeaderboardError(f"{tag}: exact {num}/{den} = {value!r} disagrees with the "
                               f"aggregated {got!r}; the two are not one measurement")
    return {"value": value, "num": num, "den": den, "rank": None, "reason": None,
            "display": _pct_display(num, den), "counts": f"{num}/{den}",
            "exact": Fraction(num, den)}


def _rate(agg_cell, tag: str) -> dict:
    v = agg_cell.get("value") if isinstance(agg_cell, dict) else agg_cell
    if v is None:
        return {"value": None, "rank": None,
                "reason": (agg_cell or {}).get("reason") or "not measured in every cell"}
    if not isinstance(v, (int, float)):
        raise LeaderboardError(f"{tag}: non-numeric aggregate {v!r}")
    return {"value": float(v), "rank": None, "reason": None,
            "display": _round(repr(float(v)), 2), "exact": float(v)}


def paired_pace(cells: list, reference_id: str) -> dict:
    """Mean solo lap-time difference against the reference, on jointly completed trials only.

    Trial index is a shared start: `validate_results` has already refused the cohort if any cell's
    systems did not begin from the same physical state, so index i is the same start for both.
    A pair with no jointly completed trial is N/A -- never a zero, which would read as "same pace".
    """
    by_cell: dict = {}
    for c in cells:
        if c["suite"] != "S":
            continue
        by_cell.setdefault(str(c["cell_id"]), {})[c["system_id"]] = c
    tot: dict = {}
    for _cid, group in sorted(by_cell.items()):
        ref = group.get(reference_id)
        if ref is None:
            continue
        ref_out, ref_lap = ref["result"]["outcomes"], ref["result"]["lap_time_s"]
        for sid, c in sorted(group.items()):
            if sid == reference_id:
                continue
            out, lap = c["result"]["outcomes"], c["result"]["lap_time_s"]
            t = tot.setdefault(sid, {"sum": 0.0, "n": 0})
            for i in range(min(len(out), len(ref_out))):
                if out[i].get("success") and ref_out[i].get("success"):
                    t["sum"] += float(lap[i]) - float(ref_lap[i])
                    t["n"] += 1
    pace = {reference_id: {"value": None, "n": None, "reason": "reference system",
                           "display": "N/A (reference system)"}}
    for sid, t in tot.items():
        if not t["n"]:
            pace[sid] = {"value": None, "n": 0,
                         "reason": "no trial completed by both this system and the reference",
                         "display": "N/A (no jointly completed trial)"}
            continue
        mean = t["sum"] / t["n"]
        text = _round(repr(mean), 3)
        pace[sid] = {"value": mean, "n": t["n"], "reason": None,
                     "display": f"{'+' if mean >= 0 else ''}{text} s (n={t['n']})"}
    return pace


def competition_ranks(values: dict, better: str) -> tuple[dict, list]:
    """Shared rank for exactly equal values (1, 2, 2, 4), stable order, N/A last with no rank.

    Ranking uses the exact value -- a Fraction for a rate, the full float for a per-km figure --
    so two systems tie only when their measurements really are equal, never because the table
    rounds them to the same string.
    """
    avail = [(sid, v) for sid, v in values.items() if v is not None]
    avail.sort(key=lambda t: t[0])                              # stable tie order: system id
    avail.sort(key=lambda t: t[1], reverse=(better == "up"))
    ranks, order, prev, prev_rank = {}, [], None, None
    for i, (sid, v) in enumerate(avail, 1):
        rank = prev_rank if (prev is not None and v == prev) else i
        prev, prev_rank = v, rank
        ranks[sid] = rank
        order.append(sid)
    order += sorted(sid for sid, v in values.items() if v is None)
    return ranks, order


# --------------------------------------------------------------------------- view model


def build_cohort(cohort: dict, base: str, out_dir: str | None = None) -> dict:
    """Validate one cohort strictly, then reduce it to the view model the renderers print."""
    cid = cohort["id"]
    suite, freeze = suite_mod.load(_resolve(base, cohort["suite"]))
    entries = roster_mod.load(_resolve(base, cohort["roster"]))
    cells, evidence = _read_cells(base, cohort["results"], out_dir)
    roster = {e.system_id: e for e in entries}
    # Shallow copies, exactly as the benchmark CLI does: `result` stays shared so the derived
    # counts stamped during validation are visible on the cells aggregation consumes.
    rows = [dict(c) for c in cells]
    try:
        info = report_mod.validate_results(
            rows, suite_freeze=freeze,
            expected_systems=set(roster),
            expected_trials=suite.expected_trials(),
            expected_cells=suite.cells(),
            suite=suite,
            roster=roster)
    except report_mod.ReportError as exc:
        raise LeaderboardError(f"cohort {cid}: {exc}") from exc

    reference = cohort["reference_system"]
    if reference not in roster:
        raise LeaderboardError(f"cohort {cid}: reference_system {reference!r} is not in the roster")

    agg = report_mod.aggregate(cells)
    low_mu = min(float(m) for m in suite.solo_mus)
    counts = exact_counts(cells, low_mu)
    idx = _agg_index(agg, low_mu)
    pace = paired_pace(cells, reference)
    labels = cohort.get("labels") or {}

    systems, exact = [], {m["key"]: {} for m in METRICS}
    for sid in sorted(roster):
        e, b, a = roster[sid], counts[sid], idx.get(sid, {})
        lab = labels.get(sid) or {}
        m = {
            "solo_completion": _pct(b["S"][0], b["S"][1], a.get("solo_completion"), f"{cid} {sid}"),
            "low_mu_completion": _pct(b["low"][0], b["low"][1], a.get("low_mu_completion"),
                                      f"{cid} {sid}"),
            "avoidance": _pct(b["A"][0], b["A"][1], a.get("avoidance"), f"{cid} {sid}"),
            "overtaking": _pct(b["O"][0], b["O"][1], a.get("overtaking"), f"{cid} {sid}"),
            "collisions_km": _rate(a.get("collisions_km"), f"{cid} {sid}"),
            "slip_km": _rate(a.get("slip_km"), f"{cid} {sid}"),
        }
        for key, cellv in m.items():
            exact[key][sid] = cellv.pop("exact", None)
        systems.append({
            "system_id": sid,
            "name": str(lab.get("name") or sid),
            "training": str(lab.get("training") or "not declared"),
            "note": str(lab.get("note") or ""),
            "is_reference": sid == reference,
            "controller_arm": e.controller_arm,
            "cross_runtime": bool(e.cross_runtime),
            "checkpoint_sha256": e.checkpoint_sha256,
            "estimator_sha256": e.estimator_sha256,
            "roster_note": e.note,
            "metrics": m,
            "pace": pace.get(sid, {"value": None, "n": 0, "reason": "no paired solo cells"}),
            "counts": {"solo": b["S"], "low_mu": b["low"], "avoidance": b["A"],
                       "overtaking": b["O"]},
            "distance_km": a.get("distance_km"),
            "failures": dict(sorted(b["fail"].items())),
        })

    order = {}
    for spec in METRICS:
        ranks, order[spec["key"]] = competition_ranks(exact[spec["key"]], spec["better"])
        for s in systems:
            s["metrics"][spec["key"]]["rank"] = ranks.get(s["system_id"])

    trials = sum(int(r["n_envs"]) for r in rows)
    return {
        "id": cid, "label": cohort["label"], "description": cohort.get("description", ""),
        "reference_system": reference,
        # `scenario_cells` and `trials_per_system` are the grid every system ran; `n_cells` and
        # `n_trials` are the pooled system-cells and trial outcomes behind the whole cohort. They
        # are different counts and are labelled differently everywhere they are printed.
        "n_systems": info["n_systems"], "n_cells": info["n_rows"], "n_trials": trials,
        "scenario_cells": len(suite.cells()),
        "trials_per_system": sum(int(v) for v in suite.expected_trials().values()),
        "suite": {"version": suite.version, "freeze_sha256": info["suite_freeze_sha256"],
                  "solo_mus": [float(x) for x in suite.solo_mus],
                  "paired_mus": [float(x) for x in suite.paired_mus],
                  "low_mu": low_mu,
                  "solo_maps": list(suite.solo_maps), "seeds": [int(s) for s in suite.seeds],
                  "envs_per_cell": int(suite.envs), "speed_cap": float(suite.speed_cap),
                  "budget_laps": float(suite.budget_laps),
                  "sensor_noise": bool(suite.sensor_noise), "backend": suite.backend,
                  "reused_maps_note": suite.reused_maps_note},
        # The digest is identical for every row in the cohort -- `validate_results` refuses a
        # cohort that mixes two -- so one hash names the code every number here was measured under.
        "source_digest_sha256": suite_mod.identity_hash(rows[0]["source_digest"]),
        "expected_trials": {k: int(v) for k, v in sorted(suite.expected_trials().items())},
        # Paths and byte hashes: the files, not the rows. Validation stamps derived counts onto the
        # rows, so a validated row is no longer what the file holds and is never republished here.
        "evidence": evidence,
        "order": order,
        "systems": systems,
    }


def build(manifest_path: str, out_dir: str | None = None) -> dict:
    man, base = load_manifest(manifest_path)
    return {
        "schema": SCHEMA,
        "title": str(man.get("title") or "Checkpoint leaderboard"),
        "as_of": str(man.get("as_of") or ""),
        "default_cohort": man["default_cohort"],
        "manifest": os.path.basename(manifest_path),
        "default_rank_metric": METRICS[0]["key"],
        "metrics": [dict(m) for m in METRICS],
        "method": list(METHOD),
        "limitations": list(LIMITS),
        "command": ("cd f1sim && python -m f1sim.learn.leaderboard "
                    "--manifest ../docs/leaderboard/data/" + os.path.basename(manifest_path) +
                    " --out-dir ../docs/leaderboard"),
        "cohorts": [build_cohort(c, base, out_dir) for c in man["cohorts"]],
    }


# --------------------------------------------------------------------------- rendering


def fmt_metric(cell: dict) -> str:
    """The one display string, built when the value was derived. Never re-rounded here."""
    if cell.get("value") is None:
        return f"N/A ({cell.get('reason') or 'unmeasured'})"
    return cell["display"]


def fmt_pace(pace: dict) -> str:
    return pace.get("display") or f"N/A ({pace.get('reason') or 'unmeasured'})"


def _embed_json(obj) -> str:
    """JSON safe to sit inside a <script type="application/json"> block.

    `<`, `>` and `&` cannot appear outside a JSON string, so escaping them everywhere keeps the
    data valid JSON while making `</script>` and `<!--` impossible to form. U+2028/U+2029 are
    legal in JSON strings but are line terminators to a JavaScript parser.
    """
    s = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    for bad, safe in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                      ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        s = s.replace(bad, safe)
    return s


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _asset(name: str) -> str:
    with open(os.path.join(ASSETS, name), encoding="utf-8") as fh:
        return fh.read()


def render_html(model: dict) -> str:
    """The whole page, offline: no network, no CDN, no font fetch, no dependency.

    Dynamic values are never interpolated into markup. They travel as escaped JSON and the page
    writes them with `textContent`.
    """
    page = _asset("index_template.html")
    for token, value in (("{{TITLE}}", _esc(model["title"])),
                         ("{{AS_OF}}", _esc(model["as_of"])),
                         ("{{STYLE}}", _asset("style.css")),
                         ("{{SCRIPT}}", _asset("app.js")),
                         ("{{DATA}}", _embed_json(model))):
        if token not in page:
            raise LeaderboardError(f"template lost its {token} placeholder")
        page = page.replace(token, value)
    return page


def _md(text) -> str:
    """One markdown table cell: no pipes, no line breaks, no raw HTML."""
    s = " ".join(str(text).split())
    return (s.replace("\\", "\\\\").replace("|", "\\|")
            .replace("<", "&lt;").replace(">", "&gt;"))


def render_markdown(model: dict) -> str:
    """The GitHub-readable view: one combined six-metric table per cohort, sorted by solo
    completion, with exact counts, per-checkpoint details and links to the raw evidence."""
    key0 = model["default_rank_metric"]
    key0_label = next(m["label"] for m in model["metrics"] if m["key"] == key0)
    out = [f"# {_md(model['title'])}", "",
           f"As of `{_md(model['as_of'])}`. Rendered offline from results that were already "
           f"measured: no new trials, no weights, no GPU.", "",
           "Each cohort is validated separately and ranked separately. The cohorts were recorded "
           "under different benchmark source digests, so their rows are never pooled and their "
           "ranks are never shared.", "",
           "This file is the GitHub-readable view. `index.html` in this directory is the same "
           "report with a cohort switch, search, sortable metric columns and per-checkpoint "
           "details; it is a self-contained offline page, so download it and open it in a browser "
           "-- GitHub does not run it for you.", ""]
    for co in model["cohorts"]:
        s_ = co["suite"]
        out += [f"## {_md(co['label'])} (`{_md(co['id'])}`)", ""]
        if co["description"]:
            out += [_md(co["description"]), ""]
        out += [
            f"- {co['n_systems']} systems, each on the same grid of {co['scenario_cells']} "
            f"scenario cells = {co['trials_per_system']} trials per system",
            f"- {co['n_cells']} system-cells and {co['n_trials']} trial outcomes in total "
            f"(pooled over systems, not distinct scenarios)",
            f"- Suite `{_md(s_['version'])}` freeze `{s_['freeze_sha256']}`",
            f"- Benchmark source digest `{co['source_digest_sha256']}`",
            f"- Solo friction levels {' · '.join(str(m) for m in s_['solo_mus'])}; the low-mu "
            f"column is completion at the minimum of these ({s_['low_mu']}), which is not a "
            f"measurement of friction-estimation accuracy",
            f"- Reference system for paired pace: `{_md(co['reference_system'])}`",
            "",
            f"Sorted by {_md(key0_label)}, highest first. Percentages carry the exact "
            f"numerator/denominator they were derived from. Exactly equal values share a "
            f"competition rank.", "",
        ]
        header = ["Rank", "Checkpoint"]
        for m in model["metrics"]:
            arrow = "↑" if m["better"] == "up" else "↓"
            header.append(f"{_md(m['label'])} {arrow}")
        out += ["| " + " | ".join(header) + " |",
                "| ---: | :--- | " + " | ".join(["---:"] * len(model["metrics"])) + " |"]
        by_id = {x["system_id"]: x for x in co["systems"]}
        for sid in co["order"][key0]:
            row = by_id[sid]
            rank = row["metrics"][key0]["rank"]
            name = _md(row["name"]) + (" **(reference)**" if row["is_reference"] else "")
            cells = [str(rank) if rank is not None else "—", f"{name}<br>`{_md(sid)}`"]
            for m in model["metrics"]:
                cell = row["metrics"][m["key"]]
                if cell.get("value") is None:
                    cells.append(f"N/A ({_md(cell.get('reason') or 'unmeasured')})")
                elif m["kind"] == "pct":
                    cells.append(f"{fmt_metric(cell)}<br>{cell['counts']}")
                else:
                    cells.append(fmt_metric(cell))
            out.append("| " + " | ".join(cells) + " |")
        out += ["", "<details>",
                "<summary>Checkpoint details, paired pace and provenance</summary>", "",
                "| Checkpoint | Training | Controller | Paired pace vs reference (n) | "
                "Checkpoint sha256 | Estimator sha256 |",
                "| --- | --- | --- | --- | --- | --- |"]
        for sid in co["order"][key0]:
            row = by_id[sid]
            out.append(f"| `{_md(sid)}` | {_md(row['training'])} | {_md(row['controller_arm'])} | "
                       f"{_md(fmt_pace(row['pace']))} | `{row['checkpoint_sha256']}` | "
                       f"`{row['estimator_sha256'] or 'none'}` |")
        out += ["",
                "Paired pace is the mean solo lap-time difference against the reference over the "
                "trials both systems completed. Negative is faster than the reference. It is "
                "descriptive with its own n, not a rank metric, and its sample differs for every "
                "pair.", "",
                "Notes as declared in the manifest:", ""]
        for sid in co["order"][key0]:
            row = by_id[sid]
            if row["note"]:
                out.append(f"- `{_md(sid)}` — {_md(row['note'])}")
        out += ["", "Evidence (raw per-cell records, hashed as read):", ""]
        for ev in co["evidence"]:
            href = ev.get("href") or ev["path"]
            out.append(f"- [`{_md(ev['path'])}`]({href}) — {ev['n_cells']} cells, sha256 "
                       f"`{ev['sha256']}`")
        out += ["", "</details>", ""]

    out += ["## Method", ""]
    out += [f"{i}. {_md(t)}" for i, t in enumerate(model["method"], 1)]
    out += ["", "## Scope and limitations", ""]
    out += [f"- {_md(x)}" for x in model["limitations"]]
    out += ["", "## Regenerating", "", "```sh", model["command"], "```", "",
            "Deterministic: the same manifest and the same result files produce byte-identical "
            "`index.html`, `README.md` and `leaderboard.json`. Nothing is written unless every "
            "cohort validates.", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- CLI


def _write_all(out_dir: str, files: dict) -> list:
    """Every file written only after every cohort validated, each through a temporary file.

    A refusal leaves the previous report untouched rather than half-replaced.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, text in sorted(files.items()):
        path = os.path.join(out_dir, name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
        written.append(path)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m f1sim.learn.leaderboard",
                                description="Render the offline checkpoint leaderboard.")
    p.add_argument("--manifest", required=True, help="leaderboard manifest JSON")
    p.add_argument("--out-dir", required=True, help="directory for index.html, README.md, "
                                                    "leaderboard.json")
    a = p.parse_args(argv)
    try:
        model = build(a.manifest, a.out_dir)
        files = {"index.html": render_html(model),
                 "README.md": render_markdown(model),
                 "leaderboard.json": json.dumps(model, indent=1, sort_keys=True,
                                                ensure_ascii=False) + "\n"}
    except (LeaderboardError, report_mod.ReportError, json.JSONDecodeError) as exc:
        print(f"refusing to render: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"refusing to render: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for path in _write_all(a.out_dir, files):
        print(f"wrote {path}")
    for co in model["cohorts"]:
        print(f"  {co['id']}: {co['n_systems']} systems · {co['n_cells']} cells · "
              f"{co['n_trials']} trials · digest {co['source_digest_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
