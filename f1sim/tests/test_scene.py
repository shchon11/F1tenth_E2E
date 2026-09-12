"""Scene documents (`f1sim.scene`): the on-disk format, editing, the track round trip, the loader.

The scene is what the 환경 page edits; the simulator only ever sees the `Track` it becomes. The
properties worth pinning are therefore the ones a user would be burned by silently: a save that
does not load back to the same grids, a paint that touches the other layer, a scene that drives
differently from the catalogue map it was imported from, a cached track that ignores an edit.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim import scene as S                                            # noqa: E402
from f1sim.scene import SceneDoc                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    return tmp_path / "scenes"


def ring_scene(name="ring", size=12.0, res=0.05):
    """A walled square with a circular duct island: a loop track a centerline can be found in."""
    doc = SceneDoc.new_blank(name, size, size, resolution=res)
    doc.paint_border("tall", 0.1)
    doc.paint_disc("duct", size / 2, size / 2, 2.0)
    a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    doc.centerline = np.stack([size / 2 + 4.0 * np.cos(a), size / 2 + 4.0 * np.sin(a)], 1)
    return doc


# ----------------------------------------------------------------- torch stays out
@pytest.mark.parametrize("module", ["f1sim.scene", "f1sim.viewer.geometry", "f1sim.viewer.contours",
                                    "f1sim.viewer.console.catalog"])
def test_the_editor_modules_do_not_import_torch(module):
    code = f"import sys, {module}; assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    r = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ----------------------------------------------------------------- paths and listing
def test_scene_dir_resolves_names_and_paths(root):
    assert S.scene_dir("hall") == os.path.join(str(root), "hall")
    assert S.scene_dir("/abs/dir") == "/abs/dir"
    assert S.scene_dir("/abs/dir/scene.json") == "/abs/dir"
    with pytest.raises(ValueError):
        S.scene_dir("")


def test_list_scenes_is_newest_first_and_ignores_junk(root):
    assert S.list_scenes() == []
    a = SceneDoc.new_blank("a", 2, 2); a.save()
    os.makedirs(root / "not_a_scene")
    b = SceneDoc.new_blank("b", 2, 2); b.add_prop("cardboard_box", 1, 1)
    b.modified = ""                                   # save() stamps it
    import time; time.sleep(1.05)                     # the stamp has 1 s resolution
    b.save()
    names = [s["name"] for s in S.list_scenes()]
    assert names == ["b", "a"]
    assert S.list_scenes()[0]["props"] == 1 and S.list_scenes()[0]["shape"] == [40, 40]


# ----------------------------------------------------------------- round trip
def test_save_and_load_round_trip(root):
    doc = ring_scene()
    doc.notes = "메모"
    p = doc.add_prop("steel_drum", 3.0, 4.0, yaw=0.7, seed=5, dims={"radius": 0.2})
    d = doc.save()
    assert os.path.isfile(os.path.join(d, "scene.json")) and os.path.isfile(os.path.join(d, "layers.npz"))
    back = SceneDoc.load("ring")
    assert back.dir == d and back.name == "ring" and back.notes == "메모"
    assert np.array_equal(back.duct, doc.duct) and np.array_equal(back.tall, doc.tall)
    assert back.resolution == doc.resolution and back.origin == doc.origin
    assert np.allclose(back.centerline, doc.centerline)
    assert len(back.props) == 1
    q = back.props[0]
    assert (q.id, q.style, q.x, q.y, q.yaw, q.seed, q.dims) == (p.id, "steel_drum", 3.0, 4.0, 0.7, 5, {"radius": 0.2})
    assert q.doc is back, "loaded placements must be bound to their document"
    meta = json.load(open(os.path.join(d, "scene.json"), encoding="utf-8"))
    assert meta["schema"] == S.SCENE_SCHEMA_VERSION and meta["shape"] == [240, 240]
    assert SceneDoc.load(os.path.join(d, "scene.json")).name == "ring"


def test_copy_is_independent(root):
    doc = ring_scene()
    doc.add_prop("cardboard_box", 2, 2)
    c = doc.copy()
    c.paint_rect("tall", 1, 1, 2, 2)
    c.props[0].x = 9.0
    assert not doc.tall[doc.world_to_cell(1.5, 1.5)]
    assert doc.props[0].x == 2.0 and c.props[0].doc is c


# ----------------------------------------------------------------- grid editing
def test_each_paint_op_touches_only_its_layer():
    doc = SceneDoc.new_blank("p", 5, 5)
    ops = [
        ("paint_disc", ("duct", 2.5, 2.5, 0.5)),
        ("paint_polyline", ("tall", [(0.5, 0.5), (4.5, 0.5), (4.5, 4.5)], 0.2)),
        ("paint_rect", ("duct", 1.0, 3.0, 2.0, 4.0)),
        ("fill_polygon", ("tall", [(3.0, 3.0), (4.5, 3.0), (4.5, 4.5)])),
        ("paint_border", ("tall", 0.1)),
    ]
    for name, args in ops:
        before_d, before_t = doc.duct.copy(), doc.tall.copy()
        assert getattr(doc, name)(*args) is True, name
        layer = args[0]
        other_before = before_t if layer == "duct" else before_d
        other_after = doc.tall if layer == "duct" else doc.duct
        assert np.array_equal(other_before, other_after), f"{name} changed the other layer"
        own = doc.duct if layer == "duct" else doc.tall
        assert own.sum() > (before_d if layer == "duct" else before_t).sum(), f"{name} painted nothing"
        assert getattr(doc, name)(*args) is False, f"{name} reported a change on a no-op repaint"
    # erasing one layer clears only that layer; "free" clears both
    # `duct` / `tall` are derived from the painted layers (plus any paths): a direct edit goes
    # to `painted_*` and is followed by `rebuild_layers()`
    r, c = doc.world_to_cell(2.5, 2.5)
    doc.painted_tall[r, c] = True
    doc.rebuild_layers()
    assert doc.paint_disc("duct", 2.5, 2.5, 0.05, value=False)
    assert not doc.duct[r, c] and doc.tall[r, c]
    doc.painted_duct[r, c] = True
    doc.rebuild_layers()
    assert doc.paint_disc("free", 2.5, 2.5, 0.05)
    assert not doc.duct[r, c] and not doc.tall[r, c]


def test_disc_and_rect_use_cell_centres():
    doc = SceneDoc.new_blank("g", 2, 2, resolution=0.1)
    doc.paint_rect("tall", 0.5, 0.5, 1.0, 1.0)
    rows, cols = np.nonzero(doc.tall)
    assert rows.min() == 5 and rows.max() == 9 and cols.min() == 5 and cols.max() == 9
    doc2 = SceneDoc.new_blank("g", 2, 2, resolution=0.1)
    doc2.paint_disc("duct", 1.05, 1.05, 0.11)              # centre cell plus its 4 neighbours
    assert doc2.duct.sum() == 5 and doc2.duct[10, 10]
    assert doc.cell_to_world(*doc.world_to_cell(1.23, 0.41)) == pytest.approx((1.25, 0.45))


def test_thin_strokes_still_leave_a_trace():
    doc = SceneDoc.new_blank("t", 3, 3, resolution=0.1)
    assert doc.paint_polyline("tall", [(0.5, 1.5), (2.5, 1.5)], width=0.02)
    assert doc.tall.sum() >= 20, "a hairline stroke must not fall between cell centres"


def test_resize_keeps_content_in_place_and_fit_shrinks_to_content():
    doc = SceneDoc.new_blank("r", 6, 6)
    doc.paint_rect("tall", 2.0, 2.0, 3.0, 3.0)
    p = doc.add_prop("cardboard_box", 2.5, 4.0)
    before = doc.tall.sum()
    doc.resize_canvas(-1.0, -1.0, 8.0, 7.0)
    assert doc.origin == (-1.0, -1.0) and doc.shape == (160, 180)
    assert doc.tall.sum() == before and doc.tall[doc.world_to_cell(2.5, 2.5)]
    assert p.x == 2.5, "props are world-anchored, a resize does not move them"
    doc.fit_canvas(margin=0.5)
    x0, y0, x1, y1 = doc.bounds()
    assert x0 == pytest.approx(1.5, abs=0.06) and y0 == pytest.approx(1.5, abs=0.06)
    assert x1 >= 3.5 - 1e-9 and y1 >= 4.0 + 0.15 + 0.5 - 0.06
    assert doc.tall.sum() == before


# ----------------------------------------------------------------- props
def test_props_get_fresh_ids_and_bake_covers_the_footprint():
    doc = SceneDoc.new_blank("b", 4, 4)
    a = doc.add_prop("wooden_crate", 1.0, 1.0)
    b = doc.add_prop("marker_post", 3.0, 3.0, yaw=0.3)
    assert a.id != b.id and doc.get_prop(b.id) is b
    doc.remove_prop(a.id)
    c = doc.add_prop("barrier_block", 2.0, 2.0)
    assert c.id not in (a.id, b.id)
    ids = [i for i, _ in doc.prop_footprints()]
    assert ids == [b.id, c.id]
    fp = c.footprint_world()
    assert doc.bake_prop(c.id, "tall")
    assert doc.get_prop(c.id) is None
    inside = doc._polygon_mask(fp)
    assert inside.sum() > 0 and doc.tall[inside].all(), "every cell under the footprint is baked"
    assert doc.duct.sum() == 0
    # a post thinner than a cell still leaves cells behind
    assert doc.bake_prop(b.id, "duct") and doc.duct.sum() >= 1
    with pytest.raises(ValueError):
        doc.add_prop("no_such_style", 0, 0)


def test_quick_issues_flags_what_the_page_needs_to_show():
    doc = SceneDoc.new_blank("q", 4, 4)
    doc.paint_border("tall", 0.1)
    levels = {i["msg"]: i["level"] for i in doc.quick_issues()}
    assert any("중심선" in m for m in levels) and all(v == "warn" for v in levels.values())
    out = doc.add_prop("cardboard_box", 3.95, 2.0)          # hanging over the border wall / edge
    over = doc.add_prop("steel_drum", 2.0, 2.0)
    over2 = doc.add_prop("steel_drum", 2.05, 2.0)
    issues = doc.quick_issues()
    by_prop = {}
    for i in issues:
        by_prop.setdefault(i["prop"], []).append(i)
    assert any(i["level"] == "error" for i in by_prop[out.id])
    assert any("겹칩" in i["msg"] and i["prop"] == over.id for i in issues)
    for i in issues:
        assert set(i) == {"level", "msg", "x", "y", "prop"}
    assert over2.id in {i["prop"] for i in issues} or over.id in {i["prop"] for i in issues}
    full = SceneDoc.new_blank("full", 1, 1)
    full.paint_rect("tall", -1, -1, 2, 2)
    assert any(i["level"] == "error" and "빈 공간" in i["msg"] for i in full.quick_issues())


# ----------------------------------------------------------------- track round trip and loader
def test_from_track_to_track_is_the_same_track():
    from f1sim import maps
    t = maps.load("gen:competition:0+props3")
    assert len(t.props) > 0
    doc = SceneDoc.from_track(t, "comp0", source_map="gen:competition:0+props3")
    assert doc.source["map"] == "gen:competition:0+props3"
    back = doc.to_track()
    assert np.array_equal(back.duct, t.duct) and np.array_equal(back.tall, t.tall)
    assert np.array_equal(back.occupancy, t.occupancy)
    assert back.grid_key() == t.grid_key()
    assert back.duct_height == t.duct_height and np.allclose(back.centerline, t.centerline)
    assert back.name == "scene:comp0"
    assert [(p.style, p.x, p.y, p.yaw, p.seed, tuple(p.dims)) for p in back.props] == \
           [(p.style, p.x, p.y, p.yaw, p.seed, tuple(p.dims)) for p in t.props]


def test_maps_load_scene_prefix_with_modifiers_and_edits(root):
    from f1sim import maps
    from f1sim.viewer.console import catalog as C
    t = maps.load("gen:competition:0")
    SceneDoc.from_track(t, "comp0").save()
    assert "comp0" in maps.scene_names() and "scene:comp0" in maps.catalog()
    assert [s["name"] for s in C.list_scenes()] == ["comp0"]
    s = maps.load("scene:comp0")
    assert s.name == "scene:comp0" and s.grid_key() == t.grid_key()
    r = maps.load("scene:comp0~rev")
    assert np.allclose(r.centerline, s.centerline[::-1]) and r.edt is s.edt
    m = maps.load("scene:comp0~mir")
    assert not np.array_equal(m.occupancy, s.occupancy)
    p = maps.load("scene:comp0+props3")
    assert len(p.props) > 0
    d = maps.load(f"scene:{S.scene_dir('comp0')}")
    assert d.grid_key() == s.grid_key()
    # an edit on disk is seen by the next load, even though `maps.load` caches per process
    doc = SceneDoc.load("comp0")
    doc.paint_rect("tall", *[v + 0.3 * i for i, v in enumerate((t.origin[0], t.origin[1], t.origin[0], t.origin[1]))])
    import time; time.sleep(0.02)
    doc.save()
    os.utime(os.path.join(doc.dir, "scene.json"), None)
    s2 = maps.load("scene:comp0")
    assert s2.grid_key() != s.grid_key(), "a saved edit must not be served from the cache"
    with pytest.raises(ValueError):
        maps.load("scene:comp0+obs3")
    with pytest.raises(FileNotFoundError):
        maps.load("scene:nope")


def test_map_catalog_lists_the_editor_group_first_only_when_there_are_scenes(root, monkeypatch):
    from f1sim.viewer import sim_worker as SW
    from f1sim.viewer.console.catalog import SCENES_GROUP
    w = SW.SimWorker.__new__(SW.SimWorker)
    groups = SW.SimWorker.map_catalog(w)
    assert SCENES_GROUP not in groups
    SceneDoc.new_blank("mine", 3, 3).save()
    groups = SW.SimWorker.map_catalog(w)
    assert list(groups)[0] == SCENES_GROUP and groups[SCENES_GROUP] == ["scene:mine"]
    assert "scene:mine" in groups["전체 카탈로그"]


# ----------------------------------------------------------------- geometry for the viewport
def test_build_track_geometry_accepts_a_scene_and_a_track():
    from f1sim.viewer.geometry import GEOMETRY_VERSION, build_track_geometry
    from f1sim.viewer import sim_worker as SW
    doc = ring_scene()
    doc.add_prop("cardboard_box", 2.0, 2.0)
    doc.add_prop("steel_drum", 10.0, 2.0, yaw=1.0)
    doc.wall_height = 0.7
    g = build_track_geometry(doc)
    assert g["geometry_version"] == GEOMETRY_VERSION == SW.GEOMETRY_VERSION
    assert g["name"] == "ring" and g["bounds"] == doc.bounds() and g["duct_height"] == doc.duct_height
    assert g["ducts"] is not None and g["walls"] is not None and g["floor"] is not None
    assert {b["material"] for b in g["props"]} == {"rubber", "metal"}
    assert float(g["walls"][0][:, 2].max()) == pytest.approx(0.7, abs=1e-5), "wall_height reaches the wall mesh"
    fast = build_track_geometry(doc, only_props=True)
    assert [b["n_tris"] for b in fast] == [b["n_tris"] for b in g["props"]]
    # a Track gives the same keys, through the worker's method and the module function alike
    t = doc.to_track()
    gt = SW.SimWorker.build_geometry(SW.SimWorker.__new__(SW.SimWorker), t)
    assert set(gt) == set(g)
    assert float(gt["walls"][0][:, 2].max()) == pytest.approx(1.0, abs=1e-5)
    for a, b in zip(g["props"], gt["props"]):
        assert np.array_equal(a["pos"], b["pos"]) and a["material"] == b["material"]


def test_contours_moved_without_changing():
    from f1sim.viewer import contours as Cn
    from f1sim.viewer.native import track_contours_mask
    from f1sim.viewer.server import _runs_off_edge, track_contours
    assert track_contours is Cn.track_contours and _runs_off_edge is Cn._runs_off_edge
    assert track_contours_mask is Cn.track_contours_mask
    doc = ring_scene()
    a = Cn.track_contours(doc, outside_occupied=False, sigma_cells=0.5)
    b = Cn.mask_contours(doc.occupancy, doc.resolution, doc.origin, outside_occupied=False, sigma_cells=0.5)
    assert len(a) == len(b) >= 2 and all(np.array_equal(x, y) for x, y in zip(a, b))
    t = doc.to_track()
    c = Cn.track_contours(t, outside_occupied=False, sigma_cells=0.5)
    assert all(np.array_equal(x, y) for x, y in zip(a, c))


# ----------------------------------------------------------------- CLI
def _run_cli(*args):
    r = subprocess.run([sys.executable, "-m", "f1sim.scene", *args], cwd=HERE, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout must be exactly one JSON object, got {r.stdout!r}\n{r.stderr}"
    return r.returncode, json.loads(lines[0]), r.stderr


def test_cli_import_and_validate_a_catalogue_map(root):
    code, out, err = _run_cli("import", "gen:competition:0", "comp")
    assert code == 0, err
    assert out["dir"] == os.path.join(str(root), "comp") and out["shape"][0] > 10 and out["props"] == 0
    doc = SceneDoc.load("comp")
    assert doc.centerline is not None
    doc.centerline = None
    doc.save()
    code, rep, err = _run_cli("validate", "comp", "--centerline", "auto")
    assert code == 0, (rep, err)
    assert rep["ok"] and rep["stats"]["length_m"] > 10 and rep["stats"]["min_width_m"] >= S.MIN_CORRIDOR_M
    assert SceneDoc.load("comp").centerline is not None, "the extracted centerline is written back"
    assert all(i["level"] != "error" for i in rep["issues"])
    # a scene that cannot be driven says so and exits 1
    bad = SceneDoc.new_blank("bad", 3, 3)
    bad.paint_border("tall", 0.1)
    bad.save()
    code, rep, err = _run_cli("validate", "bad")
    assert code == 1 and not rep["ok"] and any(i["level"] == "error" for i in rep["issues"])
    code, out, err = _run_cli("export-ros", "comp", str(root / "ros"))
    assert code == 0 and os.path.isfile(out["yaml"]) and os.path.isfile(os.path.join(str(root / "ros"), "comp.pgm"))
    code, out, err = _run_cli("validate", "does_not_exist")
    assert code == 1 and "error" in out
