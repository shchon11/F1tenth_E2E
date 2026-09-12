"""Random track generation (`f1sim.trackgen`): closed, clear of itself, honouring the recipe."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim import trackgen as T                                          # noqa: E402
from f1sim.scene import SceneDoc, sample_path                            # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL = T.TrackRecipe(runs={k: -1 for k in T.RUN_FEATURES}, turns={k: -1 for k in T.TURN_FEATURES}, size_m=20)


def _closed_and_clear(g):
    cl = sample_path(g.points, True, step=0.1)
    assert len(cl) > 50
    # no two samples farther apart along the lap than 1.5 clearances come closer than a lane + hose
    clearance = g.lane_width + g.hose
    d = np.linalg.norm(cl[:, None] - cl[None, :], axis=2)
    seg = np.linalg.norm(np.roll(cl, -1, 0) - cl, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    da = np.abs(arc[:, None] - arc[None, :])
    da = np.minimum(da, seg.sum() - da)
    far = da >= 1.5 * clearance
    assert d[far].min() > clearance, f"track comes within {d[far].min():.2f} m of itself"


@pytest.mark.parametrize("seed", range(12))
def test_every_seed_gives_a_closed_clear_lap_of_the_requested_size(seed):
    g = T.generate(ALL, seed)
    _closed_and_clear(g)
    x0, y0, x1, y1 = g.bounds
    ext = max(x1 - x0, y1 - y0) - 2 * (ALL.margin_m + 0.5 * ALL.lane_width + ALL.hose)
    assert abs(ext - ALL.size_m) < 0.5
    assert sum(1 for f in g.features if f in T.TURN_FEATURES) == sum(1 for f in g.features if f in T.RUN_FEATURES)
    assert abs(sum(T.TURN_FEATURES[f]["angle"] for f in g.features if f in T.TURN_FEATURES) - 360.0) < 1e-6
    assert any(sm for _x, _y, sm in g.points), "arcs are curve vertices"


def test_same_recipe_and_seed_is_reproducible():
    a, b = T.generate(ALL, 7), T.generate(ALL, 7)
    assert a.points == b.points and a.features == b.features and a.props == b.props
    assert T.generate(ALL, 8).points != a.points


def test_fixed_counts_are_honoured():
    r = T.TrackRecipe(runs={"straight": -1, "chicane": 2, "cone_slalom": 1},
                      turns={"corner": -1, "hairpin": 1, "sweeper": -1, "reverse": -1}, size_m=24)
    for seed in range(4):
        g = T.generate(r, seed)
        assert g.features.count("chicane") == 2 and g.features.count("cone_slalom") == 1
        assert g.features.count("hairpin") == 1
        assert "slalom_fast" not in g.features and "slalom_slow" not in g.features
        assert all(p["style"] == "marker_post" for p in g.props) and len(g.props) >= 3
        _closed_and_clear(g)


def test_two_hairpins_make_an_oval_and_four_corners_a_rectangle():
    g = T.generate(T.TrackRecipe(runs={"straight": -1}, turns={"hairpin": 2}, size_m=16), 1)
    assert [f for f in g.features if f in T.TURN_FEATURES] == ["hairpin", "hairpin"]
    _closed_and_clear(g)
    g = T.generate(T.TrackRecipe(runs={"straight": -1}, turns={"corner": 4}, size_m=12), 2)
    assert g.features.count("corner") == 4 and g.features.count("straight") == 4
    assert sum(1 for _x, _y, sm in g.points if not sm) == 5      # four run ends plus the start


def test_impossible_turn_set_is_refused_loudly():
    with pytest.raises(RuntimeError):
        T.generate(T.TrackRecipe(runs={"straight": -1}, turns={"sweeper": 1}), 0)


def test_scene_from_a_generated_track_drives_the_editor_path_model(tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path))
    g = T.generate(ALL, 3)
    doc = g.to_scene("gen3")
    assert doc.track_path is not None and doc.centerline is not None
    assert doc.duct.any() and doc.tall.any()                    # hoses and the border wall
    assert len(doc.props) == len(g.props)
    assert doc.source["generator"] == "trackgen" and doc.source["seed"] == 3
    issues = [i for i in doc.quick_issues() if i["level"] == "error"]
    assert issues == [], issues
    d = doc.save()
    back = SceneDoc.load(d)
    assert back.track_path is not None and np.allclose(back.centerline, doc.centerline)


def test_cli_dry_run_and_scene_output(tmp_path):
    env = {**os.environ, "F1SIM_SCENES": str(tmp_path)}
    r = subprocess.run([sys.executable, "-m", "f1sim.trackgen", "--seed", "5", "--dry-run",
                        "--runs", "straight,slalom_fast=1", "--turns", "corner,sweeper"],
                       cwd=HERE, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["features"].count("slalom_fast") == 1 and "dir" not in out
    r = subprocess.run([sys.executable, "-m", "f1sim.trackgen", "--name", "batch", "--seed", "10", "--count", "2",
                        "--size", "14"], cwd=HERE, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert [o["name"] for o in out] == ["batch_10", "batch_11"]
    assert all(os.path.isfile(os.path.join(o["dir"], "scene.json")) for o in out)
