"""Focused tests for the offline leaderboard report.

The real published evidence is the fixture: the point of this report is that it refuses anything
the benchmark validator refuses, so the tests check parity against the measured numbers and then
mutate real inputs to prove each refusal. No checkpoint is loaded and no GPU is touched -- the
roster used here carries portable identifiers, and rendering never reads weights.
"""
from __future__ import annotations

import json
import os
import shutil
from fractions import Fraction
from pathlib import Path

import pytest

from f1sim.learn import leaderboard as lb

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs" / "leaderboard" / "data"
MANIFEST = DATA / "manifest.json"

#: Independently audited values for the published cohorts: (exact counts, collisions/km,
#: large-slip s/km). A drift in any of them means the report no longer prints the measurement.
AUDIT = {
    "recipe-study": {
        "cl_origrecipe_legacy_s702@estimated": ((136, 144), (43, 48), (43, 64), (45, 64),
                                                5.253903756572392, 2.9662664958981604),
        "frozen_original@estimated": ((134, 144), (41, 48), (41, 64), (42, 64),
                                      6.0669109476388705, 3.4195316250328145),
        "cl_sgr_base_restoreopt_s701@estimated": ((130, 144), (37, 48), (36, 64), (37, 64),
                                                  7.696276343286761, 5.2089290613259625),
    },
    "initial-study": {
        "frozen_original@legacy": ((110, 144), (16, 48), (18, 64), (44, 64),
                                   12.06892275067982, 7.162905652528469),
    },
}


@pytest.fixture(scope="module")
def model():
    return lb.build(str(MANIFEST))


@pytest.fixture
def data_copy(tmp_path):
    """A writable copy of the published data, including the research files it points outside to."""
    root = tmp_path / "repo"
    shutil.copytree(DATA, root / "docs" / "leaderboard" / "data")
    shutil.copytree(REPO / "docs" / "research", root / "docs" / "research")
    return root / "docs" / "leaderboard" / "data"


def cohort(model, cid):
    return next(c for c in model["cohorts"] if c["id"] == cid)


def system(co, sid):
    return next(s for s in co["systems"] if s["system_id"] == sid)


# --------------------------------------------------------------- real data, real numbers


def test_published_cohorts_validate_with_the_audited_shape(model):
    recipe, initial = cohort(model, "recipe-study"), cohort(model, "initial-study")
    assert (recipe["n_systems"], recipe["n_cells"], recipe["n_trials"]) == (7, 238, 1904)
    assert (initial["n_systems"], initial["n_cells"], initial["n_trials"]) == (3, 102, 816)
    for co in (recipe, initial):
        # The grid each system ran, which is not the pooled system-cell count.
        assert (co["scenario_cells"], co["trials_per_system"]) == (34, 272)
        assert co["n_cells"] == co["scenario_cells"] * co["n_systems"]
        assert co["n_trials"] == co["trials_per_system"] * co["n_systems"]
    # Different recorder code: the cohorts must never be pooled, and must never share a ranking.
    assert recipe["source_digest_sha256"] != initial["source_digest_sha256"]
    assert set(recipe["order"]) == set(initial["order"]) == {m["key"] for m in model["metrics"]}


def test_metrics_match_the_independent_audit(model):
    for cid, expected in AUDIT.items():
        co = cohort(model, cid)
        for sid, (solo, low, avoid, over, coll, slip) in expected.items():
            s = system(co, sid)
            m = s["metrics"]
            assert (m["solo_completion"]["num"], m["solo_completion"]["den"]) == solo
            assert (m["low_mu_completion"]["num"], m["low_mu_completion"]["den"]) == low
            assert (m["avoidance"]["num"], m["avoidance"]["den"]) == avoid
            assert (m["overtaking"]["num"], m["overtaking"]["den"]) == over
            assert m["collisions_km"]["value"] == pytest.approx(coll, abs=1e-12)
            assert m["slip_km"]["value"] == pytest.approx(slip, abs=1e-12)
            assert s["counts"]["solo"] == list(solo)


def test_percentages_are_the_exact_counts_not_a_rounded_rate(model):
    for co in model["cohorts"]:
        for s in co["systems"]:
            for key in ("solo_completion", "low_mu_completion", "avoidance", "overtaking"):
                cell = s["metrics"][key]
                assert cell["value"] == cell["num"] / cell["den"]
                assert cell["counts"] == f"{cell['num']}/{cell['den']}"
                assert cell["den"] > 0


# --------------------------------------------------------------- refusals


def _write_manifest(data_dir: Path, mutate) -> Path:
    man = json.loads((data_dir / "manifest.json").read_text())
    mutate(man)
    path = data_dir / "mutated-manifest.json"
    path.write_text(json.dumps(man))
    return path


def test_an_incomplete_cohort_is_refused(data_copy):
    """One dropped cell is a short denominator, not a lower score."""
    src = data_copy / "recipe-study-additional-cells.jsonl"
    lines = [ln for ln in src.read_text().splitlines() if ln.strip()]
    src.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(lb.LeaderboardError) as exc:
        lb.build(str(data_copy / "manifest.json"))
    assert "recipe-study" in str(exc.value)


def test_mixing_two_source_digests_in_one_cohort_is_refused(data_copy):
    """The cohorts were recorded under different benchmark code; one table cannot hold both."""
    path = _write_manifest(data_copy, lambda m: m["cohorts"][0]["results"].append(
        "../../research/benchmark-v1-2026-09-12-derived-cells.jsonl"))
    with pytest.raises(lb.LeaderboardError) as exc:
        lb.build(str(path))
    assert "recipe-study" in str(exc.value)


def test_an_unknown_reference_system_is_refused(data_copy):
    path = _write_manifest(data_copy,
                           lambda m: m["cohorts"][0].update(reference_system="not_a_system"))
    with pytest.raises(lb.LeaderboardError):
        lb.build(str(path))


def test_a_cohort_short_of_a_declared_system_is_refused(data_copy):
    """A roster of six against results for seven is a roster mismatch, not a smaller table."""
    roster = json.loads((data_copy / "recipe-study-roster.json").read_text())
    roster["systems"] = [e for e in roster["systems"]
                         if e["system_id"] != "cl_r10_base_s702@estimated"]
    (data_copy / "short-roster.json").write_text(json.dumps(roster))
    path = _write_manifest(data_copy, lambda m: m["cohorts"][0].update(roster="short-roster.json"))
    with pytest.raises(lb.LeaderboardError):
        lb.build(str(path))


def test_a_manifest_without_cohorts_is_refused(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps({"title": "x", "cohorts": []}))
    with pytest.raises(lb.LeaderboardError):
        lb.build(str(bad))


# --------------------------------------------------------------- ranking


def test_competition_ranks_share_a_rank_and_skip_the_next():
    values = {"a": Fraction(9, 10), "b": Fraction(8, 10), "c": Fraction(8, 10),
              "d": Fraction(7, 10)}
    ranks, order = lb.competition_ranks(values, "up")
    assert ranks == {"a": 1, "b": 2, "c": 2, "d": 4}
    assert order == ["a", "b", "c", "d"]                 # ties ordered stably by system id


def test_lower_is_better_metrics_rank_the_other_way():
    ranks, order = lb.competition_ranks({"a": 7.0, "b": 5.0, "c": 5.0}, "down")
    assert ranks == {"b": 1, "c": 1, "a": 3}
    assert order == ["b", "c", "a"]


def test_missing_values_get_no_rank_and_sort_last():
    ranks, order = lb.competition_ranks({"a": None, "b": Fraction(1, 2), "c": None}, "up")
    assert ranks == {"b": 1}
    assert order == ["b", "a", "c"]


def test_ranking_is_on_exact_values_never_on_the_printed_string():
    """1/3 and 333/1000 both print as 33.3% and must not tie; 9/16 and 36/64 must."""
    ranks, _ = lb.competition_ranks({"third": Fraction(1, 3), "almost": Fraction(333, 1000)}, "up")
    assert ranks == {"third": 1, "almost": 2}
    assert lb._pct_display(1, 3) == lb._pct_display(333, 1000) == "33.3%"
    tied, _ = lb.competition_ranks({"a": Fraction(9, 16), "b": Fraction(36, 64)}, "up")
    assert tied == {"a": 1, "b": 1}


def test_one_display_string_is_shared_by_both_views(model):
    """36/64 is exactly 56.25%: Python's %.1f gives 56.2 and JavaScript's toFixed gives 56.3, so
    the string is produced once, half-up, and reused."""
    assert lb._pct_display(36, 64) == "56.3%"
    assert lb._round("5.253903756572392", 2) == "5.25"
    assert lb._round("2.965", 2) == "2.97"
    co = cohort(model, "recipe-study")
    cell = system(co, "cl_r10_base_s702@estimated")["metrics"]["avoidance"]
    assert (cell["num"], cell["den"], cell["display"]) == (36, 64, "56.3%")
    html, md = lb.render_html(model), lb.render_markdown(model)
    assert "56.3%" in md and "56.2%" not in md
    assert '"56.3%"' in html


def test_published_ranks_are_competition_ranks_per_cohort(model):
    co = cohort(model, "recipe-study")
    solo = {s["system_id"]: s["metrics"]["solo_completion"]["rank"] for s in co["systems"]}
    assert sorted(solo.values()) == [1, 2, 2, 2, 5, 5, 7]
    assert solo["cl_origrecipe_legacy_s702@estimated"] == 1
    coll = {s["system_id"]: s["metrics"]["collisions_km"]["rank"] for s in co["systems"]}
    assert coll["cl_origrecipe_legacy_s702@estimated"] == 1        # fewest collisions ranks first
    assert co["order"]["collisions_km"][0] == "cl_origrecipe_legacy_s702@estimated"


# --------------------------------------------------------------- paired pace


def _solo_cell(sid, cid, successes, laps):
    return {"system_id": sid, "suite": "S", "cell_id": cid,
            "result": {"outcomes": [{"success": bool(x)} for x in successes], "lap_time_s": laps}}


def test_paired_pace_counts_only_jointly_completed_trials():
    cells = [_solo_cell("ref", "S:m:1:1", [1, 1, 0], [10.0, 10.0, 0.0]),
             _solo_cell("cand", "S:m:1:1", [1, 0, 1], [11.0, 0.0, 9.0])]
    pace = lb.paired_pace(cells, "ref")
    assert pace["cand"] == {"value": pytest.approx(1.0), "n": 1, "reason": None,
                            "display": "+1.000 s (n=1)"}
    assert pace["ref"]["value"] is None and pace["ref"]["reason"] == "reference system"


def test_paired_pace_with_no_shared_completion_is_na_not_zero():
    cells = [_solo_cell("ref", "S:m:1:1", [1, 0], [10.0, 0.0]),
             _solo_cell("cand", "S:m:1:1", [0, 1], [0.0, 9.0])]
    pace = lb.paired_pace(cells, "ref")
    assert pace["cand"]["value"] is None and pace["cand"]["n"] == 0
    assert "jointly" in pace["cand"]["display"]


def test_published_pace_carries_its_own_n(model):
    co = cohort(model, "recipe-study")
    for s in co["systems"]:
        pace = s["pace"]
        if s["is_reference"]:
            assert pace["value"] is None and pace["reason"] == "reference system"
        else:
            assert pace["n"] and pace["n"] <= co["trials_per_system"]
            assert f"n={pace['n']}" in pace["display"]
    assert system(co, "cl_origrecipe_legacy_s702@estimated")["pace"]["display"] == \
        "+0.157 s (n=134)"


# --------------------------------------------------------------- untrusted label text


HOSTILE = '</script><img src=x onerror="alert(1)"> A & B \u2028 javascript:alert(2)'


def test_a_hostile_label_cannot_escape_the_embedded_json(data_copy):
    path = _write_manifest(data_copy, lambda m: m["cohorts"][0]["labels"].__setitem__(
        "frozen_original@estimated", {"name": HOSTILE, "training": HOSTILE, "note": HOSTILE}))
    model = lb.build(str(path))
    html = lb.render_html(model)
    assert "</script><img" not in html
    assert "onerror" not in html.replace("\\u003e", ">").split("<script")[0]
    assert "\\u003c/script\\u003e\\u003cimg" in html
    assert "\u2028" not in html
    # The block is still one parseable JSON payload, and the label survived unmangled inside it.
    blob = html.split('<script id="leaderboard-data" type="application/json">', 1)[1]
    parsed = json.loads(blob.split("</script>", 1)[0])
    assert parsed["cohorts"][0]["systems"][0]["name"] == HOSTILE or any(
        s["name"] == HOSTILE for s in parsed["cohorts"][0]["systems"])
    # Only the page's own two inline scripts exist; the label did not create a third.
    assert html.count("<script") == 2


def test_a_hostile_label_cannot_break_the_markdown_table(data_copy):
    path = _write_manifest(data_copy, lambda m: m["cohorts"][0]["labels"].__setitem__(
        "frozen_original@estimated", {"name": "a | b", "training": "<b>x</b>", "note": HOSTILE}))
    md = lb.render_markdown(lb.build(str(path)))
    assert "a \\| b" in md
    assert "&lt;b&gt;x&lt;/b&gt;" in md and "<b>x</b>" not in md
    assert "</script>" not in md and "<img" not in md
    for line in md.splitlines():
        if line.startswith("| ") and "frozen_original@estimated" in line:
            assert line.count("|") == line.rstrip().count("|")     # no cell split by a raw pipe


# --------------------------------------------------------------- CLI


def test_cli_regenerates_byte_identical_output(tmp_path):
    out = tmp_path / "out"
    assert lb.main(["--manifest", str(MANIFEST), "--out-dir", str(out)]) == 0
    first = {p.name: p.read_bytes() for p in sorted(out.iterdir())}
    assert set(first) == {"index.html", "README.md", "leaderboard.json"}
    assert lb.main(["--manifest", str(MANIFEST), "--out-dir", str(out)]) == 0
    assert {p.name: p.read_bytes() for p in sorted(out.iterdir())} == first
    assert b"http://" not in first["index.html"] and b"https://" not in first["index.html"]
    assert b"<script src" not in first["index.html"]
    assert not list(out.glob("*.tmp"))


def test_cli_links_evidence_relative_to_the_out_dir(tmp_path):
    """The manifest path is provenance; the link has to work from where the report was written."""
    out = tmp_path / "elsewhere" / "deep"
    assert lb.main(["--manifest", str(MANIFEST), "--out-dir", str(out)]) == 0
    model = json.loads((out / "leaderboard.json").read_text())
    for co in model["cohorts"]:
        for ev in co["evidence"]:
            assert os.path.exists(os.path.join(out, ev["href"])), ev
            assert not os.path.isabs(ev["href"])
    default = lb.build(str(MANIFEST), str(REPO / "docs" / "leaderboard"))
    hrefs = [e["href"] for e in cohort(default, "recipe-study")["evidence"]]
    assert hrefs == ["../research/a702-replication-2026-09-12-raw-cells.jsonl",
                     "data/recipe-study-additional-cells.jsonl"]


def test_a_refused_cohort_writes_nothing_at_all(tmp_path, data_copy, capsys):
    out = tmp_path / "out"
    good = lb.main(["--manifest", str(MANIFEST), "--out-dir", str(out)])
    before = {p.name: p.read_bytes() for p in sorted(out.iterdir())}
    assert good == 0
    src = data_copy / "recipe-study-additional-cells.jsonl"
    lines = [ln for ln in src.read_text().splitlines() if ln.strip()]
    src.write_text("\n".join(lines[:-1]) + "\n")
    assert lb.main(["--manifest", str(data_copy / "manifest.json"), "--out-dir", str(out)]) == 1
    assert "refusing to render" in capsys.readouterr().err
    # The previous report is intact: a refusal never half-replaces it.
    assert {p.name: p.read_bytes() for p in sorted(out.iterdir())} == before


def test_cli_refuses_a_missing_manifest(tmp_path, capsys):
    out = tmp_path / "out"
    assert lb.main(["--manifest", str(tmp_path / "nope.json"), "--out-dir", str(out)]) == 1
    assert "refusing to render" in capsys.readouterr().err
    assert not out.exists()


def test_the_page_is_self_contained_and_names_its_limits(model):
    html = lb.render_html(model)
    assert "cdn" not in html.lower() and "@import" not in html and "fonts.googleapis" not in html
    for needle in ("Scenario cells", "reference", "Solo completion", "Low-mu completion",
                   "Collisions/km", "Large-slip s/km"):
        assert needle in html
    md = lb.render_markdown(model)
    for needle in ("not a measurement of friction-estimation accuracy",
                   "No unseen-map, generalisation or on-car claim",
                   "Static per-episode friction", "never ranked"):
        assert needle in md
    # Nothing may read as a deployment promotion.
    for text in (html, md):
        low = text.lower()
        assert "production-ready" not in low and "state of the art" not in low
        assert "overall score" not in low or "no composite or overall score" in low
