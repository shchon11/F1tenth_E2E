"""Imported mesh assets: `props.mesh_asset`, `SceneDoc.import_asset`, and the mesh as an obstacle
the LiDAR and the geometry builder see.

The simulator sees a mesh only as convex prisms per z-band, so the checks here are about the
envelope being an honest over-approximation (hull of every vertex, budgeted to 24 sides), the
sections existing, and a beam at prop height stopping at it while one above it passes.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim import props as P                                            # noqa: E402
from f1sim.scene import SceneDoc                                        # noqa: E402

trimesh = pytest.importorskip("trimesh")

#: `sections()` output for the six built-ins at their default seeds, before the vertex cap existed.
#: If this changes, the cap (or anything else) has re-cut a prop every trained checkpoint saw.
SECTION_HASHES = {
    ("cardboard_box", 4): "f261b3f7be355975e8b9bc94ac0e1574", ("cardboard_box", 8): "6ccf1fa33b6fc8ef516c63b386460484",
    ("wooden_crate", 4): "d5970cb1121c1539cd9e1eed95d48363", ("wooden_crate", 8): "f9e5c4ae3ac7acadf15a23f0e41cc238",
    ("steel_drum", 4): "5c6d0d11ef94bdb842d868edb5aaf3e7", ("steel_drum", 8): "9a31bc0cd4db300d89ffedb4e4e40e98",
    ("crate_stack_low", 4): "5cc338ff56506586dadd97c08d495e10", ("crate_stack_low", 8): "1599cedb75529f819722079943d6d667",
    ("barrier_block", 4): "bb5ea33df1d65af11533b28a12da617c", ("barrier_block", 8): "cb60fecd0b4fa229125473f623e5f21a",
    ("marker_post", 4): "9aad0bbd0356b820f8f503dd519dff80", ("marker_post", 8): "9a306909bda2274b07451db793b96eea",
}


def _hash_sections(prop, n_bands):
    h = hashlib.md5()
    for b in P.sections(prop, n_bands=n_bands):
        h.update(np.float64(b["z0"]).tobytes()); h.update(np.float64(b["z1"]).tobytes())
        h.update(np.ascontiguousarray(b["polygon"], np.float64).tobytes())
    return h.hexdigest()


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_SCENES", str(tmp_path / "scenes"))
    return tmp_path / "scenes"


@pytest.fixture
def box_glb(tmp_path):
    p = tmp_path / "tire_stack.glb"                      # y-up file: 0.45 m tall along +Y
    trimesh.creation.box(extents=[0.6, 0.45, 0.5]).export(str(p))
    return str(p)


@pytest.fixture
def cyl_stl(tmp_path):
    p = tmp_path / "steel_pillar.stl"                    # z-up, 96-gon: more than 24 hull vertices
    trimesh.creation.cylinder(radius=0.2, height=0.7, sections=96).export(str(p))
    return str(p)


# ----------------------------------------------------------------- built-ins are untouched
@pytest.mark.parametrize("style", P.STYLES)
def test_sections_of_the_builtins_are_byte_identical(style):
    prop = P.build(style, seed=P.STYLES.index(style))
    for nb in (4, 8):
        assert _hash_sections(prop, nb) == SECTION_HASHES[(style, nb)], f"{style} sections changed at {nb} bands"
    assert "mesh" not in P.STYLES and "mesh" in P.DIMS
    assert len(P.build_all()) == 6


def test_support_polygon_encloses_and_meets_the_budget():
    hull = P._hull2d(np.random.default_rng(0).normal(size=(500, 2)))
    sp = P.support_polygon(hull, 24)
    assert 3 <= len(sp) <= 24
    e = np.roll(sp, -1, 0) - sp
    n = np.stack([e[:, 1], -e[:, 0]], 1)
    d = np.einsum("ij,ij->i", n, sp)
    assert (hull @ n.T - d[None, :] <= 1e-9).all(), "the support polygon must contain the hull"
    assert P._polygon_area(sp) > 0 and P._polygon_area(sp) < 1.15 * P._polygon_area(hull)


# ----------------------------------------------------------------- props.mesh_asset
def test_mesh_asset_from_a_y_up_glb(box_glb):
    prop = P.build("mesh", seed=0, path=box_glb, scale=1.0, up="y")
    env = prop.envelope
    assert env.height == pytest.approx(0.45, abs=1e-5), "+Y of the file must have become +Z"
    x0, y0, _, x1, y1, _ = env.bounds()
    assert (x1 - x0, y1 - y0) == pytest.approx((0.6, 0.5), abs=1e-5)
    assert len(env.footprint) == 4 and env.bevel_tolerance == 0.0
    assert np.allclose(P._centroid(env.footprint), 0.0, atol=1e-9)
    part = prop.parts[0]
    assert part.material == "rubber" and part.n_tris == 12
    assert part.pos[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(part.col[:, 3], 1.0) and np.allclose(part.col[0, :3], P.MESH_GREY)
    assert np.allclose(np.linalg.norm(part.nrm, axis=1), 1.0, atol=1e-5)
    secs = P.sections(prop)
    assert secs and all(len(b["polygon"]) <= P.MAX_FOOTPRINT_VERTS for b in secs)
    assert prop.spec["max_section_verts"] == 24 and prop.spec["tris"] == 12
    assert P.build("mesh", path=box_glb, scale=1.0, up="y") is prop, "builds are cached by file/scale/up"
    assert P.build("mesh", path=box_glb, scale=2.0, up="y").envelope.height == pytest.approx(0.9, abs=1e-5)


def test_mesh_asset_from_an_stl_reduces_a_round_hull(cyl_stl):
    prop = P.mesh_asset(cyl_stl, up="z")
    env = prop.envelope
    assert env.height == pytest.approx(0.7, abs=1e-5)
    assert len(env.footprint) <= P.MAX_FOOTPRINT_VERTS
    r = np.linalg.norm(env.footprint, axis=1)
    assert r.min() >= 0.2 - 1e-6 and r.max() <= 0.2 / math.cos(math.pi / 24) + 1e-6, \
        "the footprint must enclose the cylinder without exceeding the support polygon bound"
    assert prop.parts[0].material == "metal"
    secs = P.sections(prop, n_bands=4)
    assert len(secs) == 4 and all(len(b["polygon"]) <= 24 for b in secs)
    from f1sim.prop_math import section_halfplanes
    for b in secs:
        section_halfplanes(b["polygon"], 24)             # convex, CCW, fits the pad


def test_mesh_build_validates_its_dims(box_glb):
    with pytest.raises(FileNotFoundError, match="nowhere.glb"):
        P.build("mesh", path="/nowhere/nowhere.glb")
    with pytest.raises(ValueError):
        P.build("mesh", path=box_glb, up="x")
    with pytest.raises(ValueError):
        P.build("mesh", path=box_glb, scale=-1)
    with pytest.raises(TypeError):
        P.build("mesh", path=box_glb, radius=0.2)
    with pytest.raises(ValueError):
        P.build("cardboard_box", width="wide")


def test_a_large_mesh_is_decimated_and_keeps_its_colours(tmp_path):
    m = trimesh.creation.icosphere(subdivisions=6, radius=0.3)          # 81920 triangles
    m.visual.vertex_colors = np.tile(np.array([200, 30, 30, 255], np.uint8), (len(m.vertices), 1))
    p = tmp_path / "ball.ply"
    m.export(str(p))
    prop = P.mesh_asset(str(p))
    assert prop.n_tris <= P.MAX_MESH_TRIS and prop.spec["source_tris"] == 81920
    assert prop.envelope.height == pytest.approx(0.6, abs=0.02)
    col = prop.parts[0].col
    assert np.allclose(col[:, 0], 200 / 255, atol=1e-3) and np.allclose(col[:, 3], 1.0)


# ----------------------------------------------------------------- SceneDoc assets
def test_import_asset_copies_measures_and_round_trips(root, box_glb, cyl_stl):
    doc = SceneDoc.new_blank("hall", 10, 10)
    with pytest.raises(ValueError):
        doc.import_asset(box_glb)                          # no dir yet
    doc.save()
    a = doc.import_asset(box_glb)
    assert a.up == "y" and a.file == "assets/tire_stack.glb" and a.tris == 12
    assert a.size == pytest.approx([0.6, 0.5, 0.45], abs=1e-5)
    assert os.path.isfile(doc.asset_path(a.id))
    b = doc.import_asset(cyl_stl, name="기둥", target_height=1.4)
    assert b.up == "z" and b.scale == pytest.approx(2.0, abs=1e-6) and b.size[2] == pytest.approx(1.4, abs=1e-6)
    c = doc.import_asset(box_glb)                          # same file name twice: kept apart
    assert c.file != a.file and os.path.isfile(doc.asset_path(c.id))
    with pytest.raises(ValueError):
        doc.import_asset(__file__)                         # not a mesh
    p = doc.add_prop("mesh", 5.0, 5.0, yaw=0.5, asset=b.id, dims={"scale": 0.5})
    prop = p.build()
    assert prop.envelope.height == pytest.approx(0.7, abs=1e-5), "asset scale x placement scale"
    sp = p.to_static_prop()
    assert sp.style == "mesh" and dict(sp.dims)["path"] == doc.asset_path(b.id) and dict(sp.dims)["up"] == "z"
    with pytest.raises(ValueError):
        doc.remove_asset(b.id)                             # still placed
    with pytest.raises(ValueError):
        doc.add_prop("mesh", 1, 1, asset="a99")
    doc.save()
    back = SceneDoc.load("hall")
    assert [x.to_json() for x in back.assets] == [x.to_json() for x in doc.assets]
    q = back.get_prop(p.id)
    assert q.asset == b.id and q.dims == {"scale": 0.5}
    assert np.allclose(q.footprint_world(), p.footprint_world())
    assert not back.quick_issues() or all(i["level"] == "warn" for i in back.quick_issues())
    back.remove_prop(p.id)
    back.remove_asset(b.id)
    assert not os.path.exists(os.path.join(back.dir, b.file)) and back.get_asset(b.id) is None
    # save-as carries the asset files along
    d2 = back.save("hall_copy")
    assert os.path.isfile(os.path.join(d2, a.file)) and back.name == "hall_copy"
    # a missing file is an error the page can show
    os.remove(os.path.join(d2, a.file))
    assert any(i["level"] == "error" and "메시" in i["msg"] for i in back.quick_issues())


def test_bake_a_mesh_prop_covers_its_projection(root, box_glb):
    doc = SceneDoc.new_blank("bake", 4, 4)
    doc.save()
    a = doc.import_asset(box_glb)
    p = doc.add_prop("mesh", 2.0, 2.0, yaw=0.4, asset=a.id)
    fp = p.footprint_world()
    assert doc.bake_prop(p.id, "tall") and doc.get_prop(p.id) is None
    inside = doc._polygon_mask(fp)
    assert inside.sum() > 50 and doc.tall[inside].all()
    assert doc.tall.sum() < 3 * inside.sum(), "the bake is the box, not a blob"


# ----------------------------------------------------------------- the simulator sees it
def test_geometry_and_lidar_see_the_imported_mesh(root, box_glb, cyl_stl):
    import torch
    from f1sim.params import Config
    from f1sim.sim import Simulator
    from f1sim.lidar import HIT_TALL
    from f1sim.viewer.geometry import build_track_geometry

    doc = SceneDoc.new_blank("lidar", 12, 12)
    doc.paint_border("tall", 0.1)
    a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    doc.centerline = np.stack([6.0 + 3.0 * np.cos(a - np.pi / 2), 6.0 + 3.0 * np.sin(a - np.pi / 2)], 1)
    doc.save()
    box = doc.import_asset(box_glb)
    pillar = doc.import_asset(cyl_stl)
    doc.add_prop("mesh", 6.0, 3.0, yaw=0.0, asset=box.id)          # 0.45 m tall, 0.6 m wide in x
    doc.add_prop("mesh", 9.0, 9.0, asset=pillar.id)

    g = build_track_geometry(doc)
    by_mat = {b["material"]: b for b in g["props"]}
    assert by_mat["rubber"]["n_tris"] == 12 and by_mat["metal"]["n_tris"] == 96 * 4
    assert sum(b["n_props"] for b in g["props"]) == 2

    track = doc.to_track()
    assert len(track.props) == 2
    cfg = Config()
    cfg.sim.device = "cpu"
    cfg.rand.enabled = False
    cfg.lidar.motion_distortion = False
    cfg.lidar.noise_std = 0.0; cfg.lidar.dropout_prob = 0.0; cfg.lidar.spike_prob = 0.0
    cfg.lidar.n_beams = 25; cfg.lidar.mount_x = 0.0; cfg.lidar.mount_z = 0.15
    cfg.imu.enabled = False
    sim = Simulator(track, cfg, num_envs=1, device="cpu")
    pose = torch.tensor([[3.0, 3.0, 0.0]])
    sim.reset(poses=pose)
    mid = cfg.lidar.n_beams // 2
    level = torch.zeros(1, 2)
    _, r0, t0 = sim.lidar.scan(pose, None, sim.P, False, noisy=False, att=level)
    assert int(t0[0, mid]) == HIT_TALL, "a beam at 0.15 m must stop at the 0.45 m box"
    assert float(r0[0, mid]) == pytest.approx(3.0 - 0.3, abs=0.02)
    # nose up 8 deg: at 2.7 m the beam is at 0.15 + 2.7 tan 8 = 0.53 m, above the box -> it passes
    att = torch.tensor([[0.0, -math.radians(8.0)]])
    _, r1, t1 = sim.lidar.scan(pose, None, sim.P, False, noisy=False, att=att)
    assert float(r1[0, mid]) > 6.0, "a beam above the mesh's height must pass over it"
    # and the car actually collides with it
    sim.reset(poses=torch.tensor([[6.0, 3.0, 0.0]]))
    assert float(sim._prop_contact(sim.state)[0].max()) > 0
