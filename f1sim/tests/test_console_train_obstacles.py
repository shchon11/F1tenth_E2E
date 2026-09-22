"""장애물 "학습과 같음" -- the pieces that are not the window or the worker process.

Where the generator's settings come from (a checkpoint does not carry them), that the console's
prop leaves are captured for a per-env layout rather than left eager, that the catalogue's meshes
pass the same checks as placed ones, and that the scene puts each piece where the frame says.
"""
import json
import math

import numpy as np
import pytest
import torch

from f1sim import procedural_obstacles as po
from f1sim.viewer.console.session import PropPayloadError, _dyn_prop_shapes
from f1sim.viewer.geometry import shape_meshes
from f1sim.viewer.graph_fastpath import SimGraphFastPath
from f1sim.viewer.sim_worker import PROCEDURAL_DEFAULTS, procedural_settings


def _run(tmp_path, args=None, name="run"):
    d = tmp_path / name
    d.mkdir()
    ck = d / "ppo_latest.pt"
    ck.write_bytes(b"")
    if args is not None:
        (d / "args.json").write_text(json.dumps(args))
    return str(ck)


def test_a_run_that_recorded_its_command_line_is_copied(tmp_path):
    ck = _run(tmp_path, {"procedural_obstacles": 0.8, "procedural_density": 1.2, "procedural_max_props": 12,
                         "procedural_raceline_corridor": "on", "spawn_runway": 2.0, "movable_obstacles": False})
    kw, src = procedural_settings(ck, "soft")
    assert src == "런의 args.json"
    assert (kw["procedural_obstacles"], kw["procedural_density"], kw["procedural_max_props"]) == (0.8, 1.2, 12)
    assert kw["procedural_raceline_corridor"] == "on" and kw["spawn_runway"] == 2.0
    assert kw["movable_obstacles"] is False and kw["procedural_shared"] is True


def test_no_record_or_no_generator_falls_back_to_the_recipe_and_says_so(tmp_path):
    kw, src = procedural_settings(_run(tmp_path), "soft")
    assert "기본값" in src and "기록 없음" in src
    assert all(kw[k] == v for k, v in PROCEDURAL_DEFAULTS.items())
    kw2, src2 = procedural_settings(_run(tmp_path, {"procedural_obstacles": 0.0}, name="off"), "soft")
    assert "없이 학습" in src2 and kw2["procedural_density"] == PROCEDURAL_DEFAULTS["procedural_density"]


def test_terminate_mode_cannot_shove_a_crate(tmp_path):
    kw, _ = procedural_settings(_run(tmp_path), "terminate")
    assert kw["movable_obstacles"] is False


def test_a_per_env_layout_counts_as_props_for_the_graph_leaves():
    """A map with no props of its own and a training layout attached has every prop there is in
    that layout; the leaves used to be refused there and ran eager every substep."""
    class Bare:
        p_poses = torch.zeros(1, 0, 3)
        env_props = None
    assert not SimGraphFastPath.has_props(Bare())
    with_layout = Bare()
    with_layout.env_props = object()
    assert SimGraphFastPath.has_props(with_layout)


def test_the_catalogue_meshes_pass_the_placed_props_checks():
    shapes, _, _ = po.build_catalogue(po.catalogue_k_pad())
    meshes = shape_meshes(shapes)
    assert len(meshes) == len(shapes)
    checked = _dyn_prop_shapes(meshes)
    assert len(checked) == len(shapes) and all(len(parts) > 0 for parts in checked)
    with pytest.raises(PropPayloadError):
        _dyn_prop_shapes([[]])                         # a shape with nothing to draw is refused
    assert _dyn_prop_shapes(None) is None


def test_the_scene_places_each_piece_where_the_frame_says():
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(backend="egl")
    except Exception as exc:
        pytest.skip(f"no headless GL: {exc}")
    from f1sim.viewer import gl_scene as G
    sc = G.Scene(ctx, 64, 48, headless=True, shadows=False)
    shapes, _, _ = po.build_catalogue(po.catalogue_k_pad())
    sc.set_dyn_props(shape_meshes(shapes), max_instances=4)
    rows = np.array([[0, 1.0, 2.0, 0.0], [0, -3.0, 0.5, math.pi / 2], [3, 5.0, 5.0, 0.3]], np.float32)
    sc.set_dyn_prop_poses(rows)
    assert [m.n_inst for m in sc.dyn_props[0]] == [2] * len(sc.dyn_props[0])
    assert [m.n_inst for m in sc.dyn_props[3]] == [1] * len(sc.dyn_props[3])
    assert all(m.n_inst == 0 for k, ms in enumerate(sc.dyn_props) if k not in (0, 3) for m in ms)
    inst = np.frombuffer(sc.dyn_props[0][0].inst.read(), np.float32).reshape(-1, 20)[:2]
    M = inst[:, :16].reshape(2, 4, 4).transpose(0, 2, 1)             # stored column-major
    np.testing.assert_allclose(M[:, :2, 3], [[1.0, 2.0], [-3.0, 0.5]], atol=1e-6)
    np.testing.assert_allclose(M[1, :2, :2], [[0.0, -1.0], [1.0, 0.0]], atol=1e-6)
    sc.set_dyn_prop_poses(None)
    assert all(m.n_inst == 0 for ms in sc.dyn_props for m in ms)
    sc.release()
    assert sc.dyn_props == []
