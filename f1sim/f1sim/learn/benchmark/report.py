"""Results validation and leaderboard rendering.

The renderer refuses more than it prints. Integrity is enforced on structure, not on truthiness: a
row is rejected for a missing pin, a suite-hash mismatch, a short trial count or an unevaluated
category -- not merely for being empty. The SGR lesson is that a report which silently renders
whatever it was handed will eventually render something wrong with full confidence.

N/A is `{"value": None, "reason": ...}` and prints as `N/A (reason)`. A category with no measurement
never becomes 0.
"""
from __future__ import annotations
import json

REQUIRED_PINS = ("system_id", "checkpoint_sha256", "controller_arm", "suite_freeze_sha256",
                 "suite_version", "map_id", "mu", "seed", "n_envs", "source_digest")

CATEGORIES = ("driving", "stability", "surface", "avoidance", "overtaking")


class ReportError(ValueError):
    """A result set that must not be rendered."""


def validate_row(row: dict) -> None:
    missing = [k for k in REQUIRED_PINS if row.get(k) in (None, "")]
    if missing:
        raise ReportError(f"row missing required pins: {', '.join(missing)}")
    if row["controller_arm"] == "estimated" and not row.get("estimator_sha256"):
        raise ReportError(f"{row['system_id']}: estimated arm without an estimator pin")
    path = str(row.get("checkpoint_path", ""))
    # Same rule as the roster: an unresolved alias is refused, a substring is not. An immutable
    # `ppo_latest_frozen.pt` is a real pinned file.
    if path:
        import os
        if os.path.basename(path) in ("ppo_latest.pt", "latest.pt", "last.pt") or \
                "latest" in os.path.abspath(os.path.expanduser(path)).split(os.sep):
            raise ReportError(f"{row['system_id']}: unresolved alias in checkpoint path {path!r}")


#: Per-trial arrays the producer emits. Every one is checked for length; the optional ones may be
#: absent, and absent means N/A, never 0.
REQUIRED_TRIAL_ARRAYS = ("outcomes", "progress_m", "distance_m", "route_progress_fraction",
                         "lap_time_s")
OPTIONAL_TRIAL_ARRAYS = ("cross_track_abs_mean_m", "cross_track_rms_m", "cross_track_samples")
#: Scalars `runner.py` writes only when an accumulator was attached. Absent is a real state.
OPTIONAL_SCALARS = ("spin_events", "large_slip_seconds", "wrong_way_seconds", "max_abs_yaw_rate")
#: Quantities that cannot be negative. Route progress is deliberately NOT here: it is signed, and an
#: abs() would silently reward driving backwards.
NONNEGATIVE_SCALARS = ("spin_events", "large_slip_seconds", "wrong_way_seconds", "max_abs_yaw_rate")
NONNEGATIVE_ARRAYS = ("distance_m", "lap_time_s", "cross_track_abs_mean_m", "cross_track_rms_m",
                      "cross_track_samples")


#: Continuous protocol fields survive a float32 round-trip through the plant; discrete ones do not
#: and are compared exactly. 1e-6 is far tighter than any meaningful difference in mu or speed.
FLOAT_PROTOCOL_REL_TOL = 1e-6


def _finite(x) -> bool:
    import math
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _derive(res: dict) -> dict:
    """Counts rebuilt from the outcome array. Never read from a supplied tally.

    A tally and an outcome array can be wrong in the same way; only one of them is the measurement.
    """
    outcomes = res.get("outcomes") or []
    completed = [i for i, o in enumerate(outcomes) if o.get("success")]
    failures: dict = {}
    for o in outcomes:
        if not o.get("success"):
            failures[o.get("reason")] = failures.get(o.get("reason"), 0) + 1
    return {"n": len(outcomes), "successes": len(completed), "completed_idx": completed,
            "failures": failures}


def validate_row_identity(row: dict, expected: dict) -> None:
    """The row is the protocol it claims, and the cell it claims.

    `cell_id` is recomputed from the row's own map/mu/seed rather than trusted, so a row cannot
    carry one cell's measurements under another cell's name.
    """
    for k, want in expected.items():
        got = row.get(k)
        if got != want:
            raise ReportError(f"{row.get('system_id')} {row.get('cell_id')}: {k} is {got!r}, "
                              f"expected {want!r}")
    want_id = f"{row.get('suite')}:{row.get('map_id')}:{row.get('mu')}:{row.get('seed')}"
    if str(row.get("cell_id")) != want_id:
        raise ReportError(f"{row.get('system_id')}: cell_id {row.get('cell_id')!r} does not match "
                          f"its own metadata, which describes {want_id!r}")


def validate_cell(row: dict, expected_cell, suite=None, roster_entry=None) -> dict:
    """The one shared gate: a row is admissible evidence for `expected_cell`, or it is refused.

    Called by the report before aggregation and by the runner's resume before marking a cell done,
    so a resume can never skip a cell whose row the report would later reject.

    `expected_cell` is REQUIRED. A validator whose expectations are optional is a validator that
    runs without them the first time a caller forgets, and defects 2 and 3 were exactly that.
    """
    validate_raw(row, expected_cell)
    res = row["result"]
    derived = _derive(res)
    tag = f"{row.get('system_id')} {row.get('cell_id')}"

    # -- effective configuration against what the cell and suite declared
    eff = res.get("effective") or row.get("effective")
    if eff is None:
        raise ReportError(f"{tag}: no effective configuration recorded; the declared protocol "
                          f"cannot be checked against what actually ran")
    # Continuous protocol fields are compared with a TOLERANCE, discrete ones exactly.
    #
    # The plant stores mu as float32, so a declared float64 level never comes back bit-identical:
    # 0.73423 -> 0.7342299818992615 (1.8e-08), 0.94401 -> 0.9440100193023682 (1.9e-08),
    # 1.15379 -> 1.15378999710083 (2.9e-09). An exact comparison refuses every real cell at every
    # friction level while passing synthetic rows whose `effective` values are float64 literals --
    # so it fails only on the measurements it exists to protect. rel_tol=1e-6 is orders of
    # magnitude tighter than any physically meaningful difference in mu, speed cap or budget, and
    # comfortably absorbs a float32 round-trip.
    import math
    # PRESENCE, then equality. A field that is merely absent used to pass every comparison, so a
    # row could omit the friction it ran at, or its whole declared protocol, and be accepted -- the
    # same vacuous-check shape as an equality test that never runs.
    REQUIRED_EFFECTIVE = ("true_mu", "plant_mu")
    missing_eff = [k for k in REQUIRED_EFFECTIVE if eff.get(k) is None]
    if missing_eff:
        raise ReportError(f"{tag}: effective configuration records no {missing_eff}; the friction "
                          f"the run actually used cannot be checked against the declared cell")
    REQUIRED_PROTOCOL = ("speed_cap", "budget_laps", "sensor_noise", "backend")
    missing_proto = [k for k in REQUIRED_PROTOCOL
                     if eff.get(k, row.get(k, None)) is None]
    if missing_proto:
        raise ReportError(f"{tag}: row records no {missing_proto}; the declared protocol cannot be "
                          f"checked against what ran")
    declared_mu = getattr(expected_cell, "true_mu", getattr(expected_cell, "mu", None))
    numeric = [("true_mu", declared_mu), ("plant_mu", declared_mu)]
    for key in ("speed_cap", "budget_laps"):
        numeric.append((key, getattr(suite, key, None) if suite is not None else None))
    for key, want in numeric:
        got = eff.get(key, row.get(key))
        if want is None:
            continue                       # nothing declared for this field
        if got is None:
            raise ReportError(f"{tag}: no effective {key} recorded to compare against the "
                              f"declared {want!r}")
        if not math.isclose(float(got), float(want), rel_tol=FLOAT_PROTOCOL_REL_TOL,
                            abs_tol=0.0):
            raise ReportError(f"{tag}: effective {key} {got!r} != declared {want!r} "
                              f"(rel_tol {FLOAT_PROTOCOL_REL_TOL})")
    # Discrete: a backend or a noise policy is either the declared one or it is not.
    for key in ("sensor_noise", "backend"):
        want = getattr(suite, key, None) if suite is not None else None
        got = eff.get(key, row.get(key))
        if want is not None and got is None:
            raise ReportError(f"{tag}: no effective {key} recorded to compare against the "
                              f"declared {want!r}")
        if want is not None and got is not None and got != want:
            raise ReportError(f"{tag}: effective {key} {got!r} != declared {want!r}")

    # -- runtime is derived from the validated controller identity, never from the free-text field
    arm = row.get("controller_arm")
    if roster_entry is not None and getattr(roster_entry, "controller_arm", arm) != arm:
        raise ReportError(f"{tag}: controller_arm {arm!r} != roster "
                          f"{getattr(roster_entry, 'controller_arm', None)!r}")
    if row.get("runtime") is not None and row.get("runtime") != arm:
        raise ReportError(f"{tag}: runtime {row.get('runtime')!r} contradicts the validated "
                          f"controller arm {arm!r}; runtime is not an independent field")
    # Stamped on the RESULT, not the row. `_read_results` returns `rows` as shallow copies and
    # `cells` as the originals; `result` is the same object in both, so a stamp there survives that
    # split while a top-level one is visible to only one of them.
    res["_derived"] = derived
    # -- reproducibility metadata: the start this cell was measured from.
    #
    # ONE validator, in `fingerprint`, called from here. This used to be a ~60-line copy of the
    # same rules, and the copy drifted: it cast `schema_version` with `int()`, so 1.9 truncated to
    # 1 and `True` read as 1, and both passed here while the shared helper refused them. Two
    # implementations of one contract diverge silently, and the one that is wrong is whichever the
    # caller happens to reach.
    fp = res.get("start_fingerprint") or row.get("start_fingerprint")
    from . import fingerprint as _fp
    try:
        _fp.validate_fingerprint(fp, where=tag, expected_cell=expected_cell, suite=suite)
    except ValueError as exc:
        raise ReportError(str(exc)) from exc
    row["_derived"] = derived
    return derived


def validate_raw(row: dict, expected_cell=None) -> None:
    """The per-trial arrays must agree with the counts derived from them, the declared cell, and
    physical reality.

    Metadata presence is not evidence. A row whose tally says 8 trials while its outcome array holds
    6, whose outcomes are not mutually exclusive, whose lap times are negative or whose progress is
    NaN is a broken measurement that would otherwise render as a clean number.
    """
    res = row.get("result")
    tag = f"{row.get('system_id')} {row.get('cell_id')}"
    if not isinstance(res, dict):
        raise ReportError(f"{tag}: row carries no raw result")
    n = int(res.get("n", 0))
    outcomes = res.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != n:
        raise ReportError(f"{tag}: {len(outcomes or [])} outcomes for n={n}")
    if any(o is None for o in outcomes):
        raise ReportError(f"{tag}: unfinished trial in outcomes")

    # -- cardinality against the DECLARED cell, not just internal agreement
    if expected_cell is not None:
        want = int(getattr(expected_cell, "envs", n))
        rec = row.get("n_envs")
        if rec is not None and int(rec) != want:
            raise ReportError(f"{tag}: recorded n_envs {rec} != declared {want}")
        if n != want:
            raise ReportError(f"{tag}: result.n {n} != declared envs {want}")

    # -- outcomes exclusive and complete
    if any(o.get("success") and o.get("reason") for o in outcomes):
        raise ReportError(f"{tag}: a success carries a failure reason; success and failure are "
                          f"exclusive")
    if any(not o.get("success") and not o.get("reason") for o in outcomes):
        raise ReportError(f"{tag}: a failure carries no reason")

    d = _derive(res)
    t = res.get("tally") or {}
    if int(t.get("successes", -1)) != d["successes"]:
        raise ReportError(f"{tag}: tally says {t.get('successes')} successes, the array holds "
                          f"{d['successes']}")
    if int(t.get("denominator", -1)) != n:
        raise ReportError(f"{tag}: denominator {t.get('denominator')} against {n} trials")
    # Rebuilt from the outcomes, then required to match. A supplied count that disagrees with the
    # measurement it summarises is not a rounding difference, it is a different claim.
    supplied = {k: v for k, v in (t.get("failures") or {}).items()}
    if supplied != d["failures"]:
        raise ReportError(f"{tag}: failure counts {supplied} do not match the outcomes "
                          f"{d['failures']}")

    # -- every per-trial array, not only the two that used to be checked
    for key in REQUIRED_TRIAL_ARRAYS:
        arr = res.get(key)
        if arr is None:
            raise ReportError(f"{tag}: required per-trial array {key!r} is absent")
        if not isinstance(arr, list) or len(arr) != n:
            raise ReportError(f"{tag}: {key} has {len(arr) if isinstance(arr, list) else '?'} "
                              f"entries for n={n}")
    for key in OPTIONAL_TRIAL_ARRAYS:
        arr = res.get(key)
        if arr is not None and (not isinstance(arr, list) or len(arr) != n):
            raise ReportError(f"{tag}: optional {key} has "
                              f"{len(arr) if isinstance(arr, list) else '?'} entries for n={n}")

    # -- lap time, and ONLY for the suite that has laps.
    #
    # `outcome.success` is TASK success, not lap completion. runner.py:196 sets
    # `completed[i] = success and self.suite == "S"`, so on the A and O suites `completed` is always
    # False and `lap_time_s` is all None even for a successful trial. Demanding a lap time on every
    # success refused every real A/O row, and the alternative -- inventing a number to satisfy the
    # check -- would be worse. Task duration is already recorded separately as `elapsed_s`.
    suite_id = str(getattr(expected_cell, "suite", None) or row.get("suite") or "")
    laps = res["lap_time_s"]
    successes = [bool(o.get("success")) for o in outcomes]
    if suite_id == "S":
        for i, (lt, ok) in enumerate(zip(laps, successes)):
            if ok and lt is None:
                raise ReportError(f"{tag}: S trial {i} completed a lap but carries no lap time")
            if not ok and lt is not None:
                raise ReportError(f"{tag}: S trial {i} did not complete but carries lap time {lt}")
    elif suite_id:
        # A/O: a lap time here would be a fabricated number for a lap that was never run.
        bad = [i for i, lt in enumerate(laps) if lt is not None]
        if bad:
            raise ReportError(f"{tag}: suite {suite_id} has no laps, but trials {bad[:3]} carry a "
                              f"lap time; task duration belongs in elapsed_s, not lap_time_s")

    # -- finite and physically possible
    for key in REQUIRED_TRIAL_ARRAYS + OPTIONAL_TRIAL_ARRAYS:
        arr = res.get(key)
        if key == "outcomes" or arr is None:
            continue
        for i, v in enumerate(arr):
            if v is None:
                continue
            if isinstance(v, dict):          # the producer's {"value": None, "reason": ...} form
                v = v.get("value")
                if v is None:
                    continue
            if not _finite(v):
                raise ReportError(f"{tag}: {key}[{i}] is not finite ({v!r})")
            if key in NONNEGATIVE_ARRAYS and float(v) < 0:
                raise ReportError(f"{tag}: {key}[{i}] is negative ({v!r})")
    for key in OPTIONAL_SCALARS:
        v = res.get(key)
        if v is None:
            continue
        if isinstance(v, dict):
            v = v.get("value")
            if v is None:
                continue
        if not _finite(v):
            raise ReportError(f"{tag}: {key} is not finite ({v!r})")
        if key in NONNEGATIVE_SCALARS and float(v) < 0:
            raise ReportError(f"{tag}: {key} is negative ({v!r})")


def validate_results(rows: list[dict], *, suite_freeze: str, expected_systems: set[str],
                     expected_trials: dict, expected_cells=None, roster=None, suite=None) -> dict:
    """Every row pinned, every row on the frozen suite, every declared system present and complete."""
    # Checked FIRST: this is a defect in the call, not in the data, and reporting it as a row
    # problem would send the reader looking in the wrong place. Without the Suite object,
    # validate_cell silently skips the declared speed cap, budget, noise policy and backend -- the
    # "an optional expectation is one a caller eventually omits" failure I argued to core for and
    # then committed at my own call site.
    if expected_cells is not None and suite is None:
        raise ReportError("validate_results needs the Suite object to check effective "
                          "configuration against the declared protocol; pass suite=<Suite>. "
                          "Validating cells without it would skip speed_cap, budget_laps, "
                          "sensor_noise and backend entirely.")
    if not rows:
        raise ReportError("no results to render")
    for r in rows:
        validate_row(r)
        validate_raw(r)
        if roster is not None:
            e = roster.get(r["system_id"])
            if e is None:
                raise ReportError(f"{r['system_id']} is not in the roster")
            if r["checkpoint_sha256"] != e.checkpoint_sha256:
                raise ReportError(f"{r['system_id']}: row SHA {r['checkpoint_sha256'][:12]} does "
                                  f"not match the roster's {e.checkpoint_sha256[:12]}")
            if r["controller_arm"] != e.controller_arm:
                raise ReportError(f"{r['system_id']}: row arm {r['controller_arm']!r} does not "
                                  f"match the roster's {e.controller_arm!r}")
            if r.get("estimator_sha256") != e.estimator_sha256:
                raise ReportError(f"{r['system_id']}: estimator pin disagrees with the roster")
        if r.get("identity_sha256"):
            from .suite import identity_hash
            # Mirrors `protocol_identity`: the hash covers the constant protocol, not per-cell
            # facts. `effective` is evidence and is validated separately, not hashed.
            body = {k: v for k, v in r.items()
                    if k not in ("identity_sha256", "suite", "map_id", "mu", "seed", "n_envs",
                                 "runtime", "cell_id", "wall_seconds", "result", "label",
                                 "obstacle", "effective")}
            if identity_hash(body) != r["identity_sha256"]:
                raise ReportError(f"{r['system_id']} {r.get('cell_id')}: identity hash does not "
                                  f"match the fields it covers; the row was edited after the run")
        if r["suite_freeze_sha256"] != suite_freeze:
            raise ReportError(
                f"{r['system_id']} was run against suite {r['suite_freeze_sha256'][:12]} but the "
                f"report is for {suite_freeze[:12]}: different protocols cannot share one table")

    # One set of weights per system id. A table row that silently mixes two checkpoints is the
    # worst failure available here, because it still looks like a measurement.
    by_sys = {}
    for r in rows:
        by_sys.setdefault(r["system_id"], set()).add(r["checkpoint_sha256"])
    mixed = {k: sorted(v) for k, v in by_sys.items() if len(v) > 1}
    if mixed:
        raise ReportError(f"rows for one system carry different checkpoint SHAs: {mixed}")

    # Each declared cell exactly once: a duplicate double-counts and a missing one shrinks a
    # denominator, and both survive every other check.
    counts = {}
    for r in rows:
        key = (r["system_id"], r["suite"], r["map_id"], float(r["mu"]), int(r["seed"]))
        counts[key] = counts.get(key, 0) + 1
    dupes = {k: v for k, v in counts.items() if v > 1}
    if dupes:
        raise ReportError(f"duplicate cells: {sorted(dupes)[:4]}")
    if expected_cells is not None:
        # One shared gate per row, so the report and the runner's resume admit exactly the same
        # evidence. This also stamps the derived counts aggregation consumes.
        by_cell = {(c.suite, c.map_id, float(c.mu), int(c.seed)): c for c in expected_cells}
        for r in rows:
            cell = by_cell.get((r.get("suite"), r.get("map_id"),
                                float(r.get("mu")), int(r.get("seed"))))
            if cell is None:
                raise ReportError(f"{r.get('system_id')} {r.get('cell_id')}: row is not on the "
                                  f"declared grid")
            validate_cell(r, cell, suite=suite,
                          roster_entry=(roster or {}).get(r.get("system_id")))
        # Paired starts and one observation layout, both checked BEFORE any cross-system number
        # exists.  refuses a row carrying no fingerprint rather than skipping it, so
        # an unfingerprinted row cannot pass the check that exists to catch it.
        from . import fingerprint as _fp
        by_id = {}
        for r in rows:
            by_id.setdefault(str(r.get("cell_id")), []).append(r)
        paired = {}
        for cid, group in sorted(by_id.items()):
            try:
                # Refuses a mixed layout by name rather than pooling it. Two systems can agree on
                # every observation tensor and still expect different stacking, which is a real way
                # to be unpaired while looking paired.
                _fp.assert_single_obs_spec(group, cell_id=cid)
            except ValueError as exc:
                raise ReportError(str(exc)) from exc
            if len(group) < 2:
                continue                      # one system on a cell: nothing to pair against
            try:
                paired[cid] = _fp.assert_paired(group, cell_id=cid)
            except ValueError as exc:
                raise ReportError(str(exc)) from exc
        want = {(sid, c.suite, c.map_id, float(c.mu), int(c.seed))
                for sid in expected_systems for c in expected_cells}
        missing = want - set(counts)
        extra = set(counts) - want
        if missing or extra:
            raise ReportError(f"cell roster mismatch: {len(missing)} missing, {len(extra)} "
                              f"unexpected; e.g. missing {sorted(missing)[:2]}")

    digests = {json.dumps(r["source_digest"], sort_keys=True) for r in rows}
    if len(digests) > 1:
        raise ReportError(
            f"rows were measured under {len(digests)} different runtime source digests; results "
            f"from different code are not one table")

    seen = {r["system_id"] for r in rows}
    if seen != expected_systems:
        missing, extra = expected_systems - seen, seen - expected_systems
        raise ReportError(f"roster mismatch: missing {sorted(missing)}, unexpected {sorted(extra)}")

    for sid in sorted(seen):
        got = {}
        for r in rows:
            if r["system_id"] == sid:
                got[r["suite"]] = got.get(r["suite"], 0) + int(r["n_envs"])
        for suite, want in expected_trials.items():
            if got.get(suite, 0) != want:
                raise ReportError(
                    f"{sid}: suite {suite} has {got.get(suite, 0)} of {want} declared trials; a "
                    f"short denominator is a different measurement, not a lower score")
    return {"n_rows": len(rows), "n_systems": len(seen), "suite_freeze_sha256": suite_freeze}


#: Labels by value, so a suite declaring any subset of levels still renders correctly. Indexing
#: `solo_mus` positionally crashed on a suite with one level.
MU_NAMES = {0.73423: "low (MU_MIN)", 0.94401: "mid (range midpoint)", 1.15379: "high (MU_MAX)"}


def mu_line(suite) -> str:
    mus = sorted(set(tuple(suite.solo_mus) + tuple(suite.paired_mus)))
    return " · ".join(f"{m} {MU_NAMES.get(m, 'unlabelled')}" for m in mus)


def fmt(cell) -> str:
    """A measurement, or an honest N/A. Never a zero standing in for 'not measured'."""
    if cell is None:
        return "N/A (not reported)"
    if isinstance(cell, dict):
        if cell.get("value") is None:
            return f"N/A ({cell.get('reason', 'unmeasured')})"
        v = cell["value"]
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    return f"{cell:.3f}" if isinstance(cell, float) else str(cell)


def render_markdown(summary: dict, *, suite, roster_note: str = "") -> str:
    """Category tables, explicit arrows, sample counts, no composite score."""
    sv = summary["suite_freeze_sha256"]
    out = [f"# Checkpoint benchmark {suite.version}", "",
           f"Suite freeze `{sv[:16]}` · {summary['n_systems']} systems · {summary['n_rows']} cells",
           "",
           "**Scope.** " + suite.reused_maps_note,
           "",
           "**Friction levels.** " + mu_line(suite) +
           " Nominal vehicle mu is 1.0489; the mid level is the *range midpoint*, not that nominal.",
           "",
           "**No composite score.** Categories are reported separately with their own units and "
           "directions; rows are not ranked across suites.", ""]
    if roster_note:
        out += [roster_note, ""]

    for cat in CATEGORIES:
        rows = summary.get(cat)
        if not rows:
            out += [f"## {cat.title()}", "", "N/A (category not evaluated)", ""]
            continue
        cols = rows[0]["columns"]
        out += [f"## {cat.title()}", "",
                "| system | runtime | " + " | ".join(cols) + " | n |",
                "| --- | --- | " + " | ".join(["---"] * len(cols)) + " | --- |"]
        for r in rows:
            vals = " | ".join(fmt(r["values"].get(c)) for c in cols)
            out.append(f"| `{r['system_id']}` | {r['runtime']} | {vals} | {r['n']} |")
        out.append("")
    return "\n".join(out)


def na(reason: str) -> dict:
    """A missing measurement. Never rendered as 0. Mirrors `tally.na`."""
    return {"value": None, "reason": reason}


def aggregate(cells: list[dict]) -> dict:
    """Raw per-cell records -> the category blocks the renderer prints.

    Rates are pooled over cells by summing successes and denominators, never by averaging rates:
    cells can differ in size, and a mean of rates would weight a short cell like a full one.
    """
    by, by_mu = {}, {}
    for c in cells:
        key = (c["system_id"], c["runtime"], c["suite"])
        b = by.setdefault(key, {"succ": 0, "den": 0, "n": 0, "prog": [], "enc": 0, "interr": 0,
                                "reasons": {}, "dist": 0.0, "coll": 0, "lap_times": [],
                                "spins": 0, "slip_s": 0.0, "xt_sq": 0.0, "xt_n": 0.0,
                                "yaw": 0.0})
        # Derived from the outcome arrays by `validate_cell`, not read from the supplied tally.
        d = c.get("_derived") or (c.get("result") or {}).get("_derived")
        if d is None:
            raise ReportError(f"{c.get('system_id')} {c.get('cell_id')}: aggregated before "
                              f"validation; call validate_cell first")
        b["succ"] += d["successes"]
        b["den"] += d["n"]
        b["n"] += c["result"]["n"]
        b["prog"] += c["result"].get("route_progress_fraction", [])
        b["enc"] += c["result"].get("encountered", 0)
        b["interr"] += sum(c["result"].get("hold_interruptions", []) or [])
        for k, v in d["failures"].items():
            b["reasons"][k] = b["reasons"].get(k, 0) + v
        # Travelled distance only. The producer always emits `distance_m` as the accumulated
        # abs(speed)*dt, so there is nothing to fall back to -- and |net progress| is a different
        # quantity that would report a car oscillating in place as having covered no ground.
        b["dist"] += sum(c["result"]["distance_m"])
        # Lap times on completed trials, by index. `if x` would also drop a legitimate 0.0 and
        # would accept a lap time sitting on a failed trial.
        laps = c["result"]["lap_time_s"]
        b["lap_times"] += [laps[i] for i in d["completed_idx"]]
        # Absent optional measurements make the pooled figure UNAVAILABLE, not smaller. Summing the
        # cells that happen to carry it would silently report a partial total as a complete one.
        for key, slot in (("max_abs_yaw_rate", "yaw"), ("spin_events", "spins"),
                          ("large_slip_seconds", "slip_s")):
            v = c["result"].get(key)
            if isinstance(v, dict):
                v = v.get("value")
            if v is None:
                b[slot] = None
            elif b[slot] is not None:
                b[slot] = (max(b[slot], float(v)) if slot == "yaw"
                           else b[slot] + (int(v) if slot == "spins" else float(v)))
        # RMS pooled over SAMPLES, so it is a real RMS rather than an RMS of per-trial RMSs.
        rms = c["result"].get("cross_track_rms_m") or []
        cnt = c["result"].get("cross_track_samples") or []
        for v, k in zip(rms, cnt):
            if isinstance(v, dict):
                v = v.get("value")
            if v is None or not k:
                continue
            b["xt_sq"] += float(v) ** 2 * float(k)
            b["xt_n"] += float(k)
        b["coll"] += sum(1 for o in c["result"].get("outcomes", [])
                         if o and o.get("reason") in ("collision", "hit", "approach_collision",
                                                      "contact"))
        if c["suite"] == "S" and c.get("mu") is not None:
            m = by_mu.setdefault((c["system_id"], c["runtime"], float(c["mu"])),
                                 {"succ": 0, "den": 0, "prog": [], "n": 0, "slip_s": 0.0,
                                  "dist": 0.0, "xt_sq": 0.0, "xt_n": 0.0})
            d = c.get("_derived") or c["result"]["_derived"]
            m["succ"] += d["successes"]
            m["den"] += d["n"]
            m["n"] += d["n"]
            m["prog"] += c["result"]["route_progress_fraction"]
            v = c["result"].get("large_slip_seconds")
            if isinstance(v, dict):
                v = v.get("value")
            m["slip_s"] = None if v is None else (
                None if m["slip_s"] is None else m["slip_s"] + float(v))
            m["dist"] += sum(c["result"]["distance_m"])
            for rv, k in zip(c["result"].get("cross_track_rms_m") or [],
                             c["result"].get("cross_track_samples") or []):
                if isinstance(rv, dict):
                    rv = rv.get("value")
                if rv is None or not k:
                    continue
                m["xt_sq"] += float(rv) ** 2 * float(k)
                m["xt_n"] += float(k)

    def rate(b):
        return {"value": b["succ"] / b["den"], "reason": None} if b["den"] else \
            {"value": None, "reason": "no pre-validated trials"}

    def pct(xs, q):
        if not xs:
            return {"value": None, "reason": "no trials"}
        s = sorted(xs)
        return {"value": s[min(len(s) - 1, int(q * len(s)))], "reason": None}

    out = {}
    # Stability draws on every suite: a collision is a collision wherever it happened.
    stab = {}
    for (sid, runtime, _suite), b in by.items():
        t = stab.setdefault((sid, runtime), {"dist": 0.0, "coll": 0, "n": 0, "spins": 0,
                                             "slip_s": 0.0, "yaw": 0.0})
        # `yaw` starts at 0.0 and is a MAX, so an absent measurement must null it rather than leave
        # a floor of zero standing in for "never observed".
        t["dist"] += b["dist"]
        t["coll"] += b["coll"]
        t["n"] += b["n"]
        # An absent optional anywhere makes the pooled figure unavailable rather than partial.
        t["spins"] = None if (b["spins"] is None or t["spins"] is None) else t["spins"] + b["spins"]
        t["slip_s"] = None if (b["slip_s"] is None or t["slip_s"] is None) else \
            t["slip_s"] + b["slip_s"]
        t["yaw"] = None if (b["yaw"] is None or t["yaw"] is None) else max(t["yaw"], b["yaw"])
    for (sid, runtime), t in sorted(stab.items()):
        km = t["dist"] / 1000.0
        out.setdefault("stability", []).append({
            "system_id": sid, "runtime": runtime, "n": t["n"],
            "columns": ["collisions/km ↓", "distance km", "spins ↓", "large-slip s/km ↓",
                        "max yaw rate rad/s (diagnostic)"],
            "values": {"collisions/km ↓": {"value": t["coll"] / km, "reason": None} if km > 0
                                          else {"value": None, "reason": "no distance travelled"},
                       "distance km": round(km, 4),
                       # Absent optional measurements are N/A. Rendering them as 0 asserts "no
                       # spins observed" when the truth is "spins were never measured".
                       "spins ↓": na("not measured in every cell") if t["spins"] is None
                                  else {"value": t["spins"], "reason": None},
                       "large-slip s/km ↓": na("not measured in every cell")
                                            if t["slip_s"] is None else
                                            ({"value": t["slip_s"] / km, "reason": None} if km > 0
                                             else na("no distance travelled")),
                       # Diagnostic: a high yaw rate on a tight corner is the corner, not a fault.
                       "max yaw rate rad/s (diagnostic)":
                           na("not measured in every cell") if t["yaw"] is None
                           else {"value": t["yaw"], "reason": None}}})

    # Retention is measured against the MIDPOINT level, which is the reference the design names --
    # not the vehicle nominal, and not the best level observed.
    mid_mu = 0.94401
    ref = {(sid, rt): m for (sid, rt, mu), m in by_mu.items() if abs(mu - mid_mu) < 1e-9}

    def _loss(m, base, key):
        if base is None or not m["den"] or not base["den"]:
            return na("no midpoint reference at this runtime")
        if key == "completion":
            return {"value": m["succ"] / m["den"] - base["succ"] / base["den"], "reason": None}
        if not m["prog"] or not base["prog"]:
            return na("no trials")
        return {"value": sum(m["prog"]) / len(m["prog"]) - sum(base["prog"]) / len(base["prog"]),
                "reason": None}

    for (sid, runtime, mu), m in sorted(by_mu.items()):
        base = ref.get((sid, runtime))
        km = m["dist"] / 1000.0
        out.setdefault("surface", []).append({
            "system_id": sid, "runtime": f"{runtime} · mu={mu}", "n": m["n"],
            "columns": ["completion ↑", "progress mean ↑", "completion vs midpoint ↑",
                        "progress vs midpoint ↑", "large-slip s/km ↓", "centreline offset RMS m"],
            "values": {"completion ↑": {"value": m["succ"] / m["den"], "reason": None} if m["den"]
                                       else na("no trials"),
                       "progress mean ↑": {"value": sum(m["prog"]) / len(m["prog"]), "reason": None}
                                          if m["prog"] else na("no trials"),
                       "completion vs midpoint ↑": _loss(m, base, "completion"),
                       "progress vs midpoint ↑": _loss(m, base, "progress"),
                       "large-slip s/km ↓": na("not measured in every cell")
                                            if m["slip_s"] is None else
                                            ({"value": m["slip_s"] / km, "reason": None} if km > 0
                                             else na("no distance travelled")),
                       # Descriptive, and deliberately without a direction: a racing or avoiding car
                       # leaves the centreline on purpose, so neither more nor less is better, and
                       # this is not evidence about friction estimation.
                       # A REAL RMS: the producer emits per-trial RMS and the sample count it was
                       # taken over, so pooling weights by samples. An unweighted mean of per-trial
                       # RMSs is not the RMS of anything.
                       "centreline offset RMS m": {"value": (m["xt_sq"] / m["xt_n"]) ** 0.5,
                                                   "reason": None}
                                                  if m["xt_n"] else na("no samples")}})

    for (sid, runtime, suite), b in sorted(by.items()):
        if suite == "S":
            out.setdefault("driving", []).append({
                "system_id": sid, "runtime": runtime, "n": b["n"],
                "columns": ["completion ↑", "progress mean ↑", "progress p10 ↑", "progress p90 ↑",
                            "lap time s ↓ (own completions only)"],
                "values": {"lap time s ↓ (own completions only)":
                               {"value": sum(b["lap_times"]) / len(b["lap_times"]), "reason": None}
                               if b["lap_times"] else
                               na("no completed laps; pace is conditional on this system's own "
                                  "successes and is not comparable across different success sets"),
                           "completion ↑": rate(b),
                           "progress mean ↑": {"value": sum(b["prog"]) / len(b["prog"]),
                                               "reason": None} if b["prog"] else
                                              {"value": None, "reason": "no trials"},
                           "progress p10 ↑": pct(b["prog"], 0.10),
                           "progress p90 ↑": pct(b["prog"], 0.90)}})
        elif suite == "A":
            out.setdefault("avoidance", []).append({
                "system_id": sid, "runtime": runtime, "n": b["n"],
                "columns": ["cleared/pre-validated ↑", "encountered", "approach failures ↓"],
                "values": {"cleared/pre-validated ↑": rate(b),
                           "encountered": b["enc"],
                           "approach failures ↓": b["reasons"].get("approach_collision", 0) +
                                                  b["reasons"].get("approach_timeout", 0)}})
        elif suite == "O":
            out.setdefault("overtaking", []).append({
                "system_id": sid, "runtime": runtime, "n": b["n"],
                "columns": ["passes held/race ↑", "hold interruptions", "contact ↓"],
                "values": {"passes held/race ↑": rate(b),
                           "hold interruptions": b["interr"],
                           "contact ↓": b["reasons"].get("contact", 0)}})
    return out
