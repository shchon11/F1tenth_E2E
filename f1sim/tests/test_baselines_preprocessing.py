"""The baselines' preprocessing, against the vendored upstream code that defines it.

CONTRACT.md: "Each model's preprocessing is reproduced from its own repo code, not re-derived".
A test that restates the paper would prove nothing, so this **executes the vendored source**:

* TinyLidarNet -- `zarrar/tiny_lidarnet.py` is imported with `tensorflow`, `numba` and
  `f1tenth_benchmarks.utils.BasePlanner` stubbed out, so its real `plan()` runs on plain numpy in
  this venv (which has no TensorFlow). The stub interpreter records the tensor their code feeds the
  network and returns a fixed output, which pins both halves at once: the input the network sees and
  the (steer, speed) their code derives from an output.
* End2Race -- `eval_singleagent.py` is a script that imports gym, f110_gym, imageio and their lattice
  planner, so it cannot be imported here. Its preprocessing lines are instead **extracted from the
  vendored file by content** and executed: the test asserts the exact source text is still there, so
  an upstream change breaks the test rather than being silently skipped.

Everything is checked against `f1sim.learn.baselines`, which is the single object the ROS node and
the batched benchmark adapter both drive.
"""
import json
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.learn import baselines                                        # noqa: E402
from f1sim.learn.baselines import end2race as e2r_mod                    # noqa: E402
from f1sim.learn.baselines import tinylidarnet as tln_mod                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EXTERNAL = os.path.join(os.path.expanduser("~"), "F1tenth", "F1tenth_E2E", "external", "baselines")
TLN_REPO = os.path.join(EXTERNAL, "TinyLidarNet")
TLN_H5 = os.path.join(TLN_REPO, "Models", "f1_tenth_model.h5")
E2R_REPO = os.path.join(EXTERNAL, "End2Race")
#: Converted once by `work/baselines/scripts/tln_to_onnx.py`; see `baselines/backends.py` for why.
MODELS = os.path.join(os.path.expanduser("~"), "Documents", "Codex", "2026-09-10", "new-chat",
                      "work", "baselines", "models")
TLN_ONNX = os.path.join(MODELS, "tinylidarnet_L_1081.onnx")
#: The three published sizes, by the `skip_n` each was trained with
#: (`generate_benchmark_results.py:168,156,147`). L is the Keras model CONTRACT.md names; M is what
#: their real-car `inference.py` loads.
TLN_BY_SKIP = {1: TLN_ONNX, 2: os.path.join(MODELS, "tinylidarnet_M_541.onnx"),
               4: os.path.join(MODELS, "tinylidarnet_S_271.onnx")}
E2R_WEIGHTS = os.path.join(E2R_REPO, "pretrained", "end2race.pth")


def real_scans():
    f = os.path.join(DATA, "real_scans_100.npz")
    if not os.path.exists(f):
        pytest.skip("real scan fixture missing")
    return np.load(f)["ranges_mm"].astype(np.float32) / 1000.0


def saturated(r, range_max=10.0):
    """`baseline_node.saturate` + the driver's own clip, written out (see the module docstring)."""
    r = np.where(np.isfinite(r) & (r > 0) & (r < 65.0), r, np.float32(range_max))
    return np.minimum(r, np.float32(range_max)).astype(np.float32)


# --------------------------------------------------------------------------- TinyLidarNet
class _RecordingInterpreter:
    """Stands in for `tf.lite.Interpreter` inside their `plan()`. Records, returns a fixed output."""

    def __init__(self, model_path=None, **_kw):
        self.model_path = model_path
        self.fed = None
        self.output = np.array([[0.25, 0.6]], dtype=np.float32)

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"index": 0, "shape": np.array([1, 1081, 1])}]

    def get_output_details(self):
        return [{"index": 1}]

    def set_tensor(self, index, value):
        self.fed = np.array(value, copy=True)

    def invoke(self):
        pass

    def get_tensor(self, index):
        return self.output


def load_vendored_tinylidarnet():
    """Their `zarrar/tiny_lidarnet.py`, executed here, with only the TF/numba imports stubbed."""
    src = os.path.join(TLN_REPO, "Benchmark", "f1tenth_benchmarks", "zarrar", "tiny_lidarnet.py")
    if not os.path.exists(src):
        pytest.skip(f"TinyLidarNet is not vendored at {TLN_REPO}")
    tf = types.ModuleType("tensorflow")
    tf.lite = types.SimpleNamespace(Interpreter=_RecordingInterpreter)
    numba = types.ModuleType("numba")
    numba.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    pkg = types.ModuleType("f1tenth_benchmarks")
    utils = types.ModuleType("f1tenth_benchmarks.utils")
    bp = types.ModuleType("f1tenth_benchmarks.utils.BasePlanner")

    class BasePlanner:                       # their base class only does bookkeeping we do not want
        def __init__(self, *a, **k):
            pass

    bp.BasePlanner = BasePlanner
    utils.BasePlanner = bp
    pkg.utils = utils
    saved = {k: sys.modules.get(k) for k in
             ("tensorflow", "numba", "f1tenth_benchmarks", "f1tenth_benchmarks.utils",
              "f1tenth_benchmarks.utils.BasePlanner")}
    sys.modules.update({"tensorflow": tf, "numba": numba, "f1tenth_benchmarks": pkg,
                        "f1tenth_benchmarks.utils": utils,
                        "f1tenth_benchmarks.utils.BasePlanner": bp})
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_vendored_tiny_lidarnet", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


@pytest.mark.parametrize("skip_n", [1, 2, 4])
def test_tinylidarnet_input_tensor_matches_vendored_plan(skip_n):
    """The tensor our driver feeds the network is the tensor their `plan()` feeds it.

    Their `plan()` adds `N(0, 0.5)` metres to every beam first (`tiny_lidarnet.py:46-47`). That is a
    sensor model, not preprocessing -- this project's suites declare their own noise per cell -- so
    it is neutralised here and declared in `TinyLidarNet.describe()['upstream_noise_applied']`.
    """
    mod = load_vendored_tinylidarnet()
    mod.np.random = types.SimpleNamespace(normal=lambda *a, **k: np.zeros(a[2] if len(a) > 2
                                                                         else k["size"]))
    planner = mod.TinyLidarNet("parity", skip_n, 0, "stub.tflite")
    scans = saturated(real_scans())

    weights = TLN_BY_SKIP[skip_n]
    if not os.path.exists(weights):
        pytest.skip(f"{os.path.basename(weights)} has not been converted")
    ours = baselines.load("tinylidarnet", weights, skip_n=skip_n)
    assert ours.describe()["upstream_noise_applied"] is False
    for row in scans[:20]:
        planner.plan({"scan": row.astype(np.float64).copy()})
        theirs = planner.interpreter.fed                        # (1, N, 1) float32
        mine = ours._prepare(row[None, :])
        assert mine.shape == theirs.shape, (mine.shape, theirs.shape)
        assert np.array_equal(mine, theirs), float(np.abs(mine - theirs).max())


def test_tinylidarnet_output_mapping_matches_vendored_plan():
    """steer straight through in radians, speed through their `linear_map` into 1..8 m/s."""
    mod = load_vendored_tinylidarnet()
    mod.np.random = types.SimpleNamespace(normal=lambda *a, **k: np.zeros(a[2] if len(a) > 2
                                                                         else k["size"]))
    planner = mod.TinyLidarNet("parity", 1, 0, "stub.tflite")
    for raw in ([[0.25, 0.6]], [[-0.4, 0.05]], [[0.0, 1.0]], [[0.31, -0.2]]):
        planner.interpreter.output = np.array(raw, dtype=np.float32)
        theirs = planner.plan({"scan": np.full(1081, 5.0)})
        mine_speed = tln_mod.linear_map(np.float32(raw[0][1]), 0.0, 1.0, *tln_mod.SPEED_MAPS["sim"][:2])
        assert theirs[0] == pytest.approx(raw[0][0], abs=1e-7)          # steer is radians, unmapped
        assert theirs[1] == pytest.approx(float(mine_speed), abs=1e-5)


def test_the_tensorflow_branch_exists_and_says_which_one_ran():
    """CONTRACT.md: "load in TF if available, otherwise convert to ONNX once ... record which".

    Both halves are pinned here. In this venv TensorFlow is not importable, so a `.h5` is refused
    with the conversion in the message rather than silently falling back, and the ONNX driver's
    `describe()` names onnxruntime and carries the conversion's provenance. (The TF branch itself is
    exercised where TF exists -- `work/baselines/venv-tf` -- and reproduces the same commands; that
    is what `tests/data/tln_tf_reference.json` is.)
    """
    if not os.path.exists(TLN_H5):
        pytest.skip("TinyLidarNet is not vendored")
    try:
        import tensorflow                                              # noqa: F401
    except ImportError:
        with pytest.raises(baselines.BaselineError, match="tln_to_onnx"):
            baselines.load("tinylidarnet", TLN_H5)
    d = baselines.load("tinylidarnet", TLN_ONNX)
    b = d.describe()["backend"]
    assert b["backend"] == "onnxruntime"
    assert b["conversion"]["source"].endswith("f1_tenth_model.h5")
    assert b["conversion"]["n_params"] == 220686
    assert b["threads"] == 1, "a multi-threaded session reorders reductions; parity is checked to 1e-5"


def test_tinylidarnet_speed_maps_are_the_two_upstream_ones():
    """Both mappings exist upstream and disagree; neither may be picked silently."""
    assert tln_mod.SPEED_MAPS["sim"][:2] == (1.0, 8.0)
    assert tln_mod.SPEED_MAPS["car"][:2] == (-0.5, 7.0)
    with pytest.raises(baselines.BaselineError):
        baselines.load("tinylidarnet", TLN_ONNX, speed_map="whatever")


def test_tinylidarnet_matches_tensorflow_on_100_real_scans():
    """CONTRACT.md's TF-vs-onnxruntime check, on real bag scans, reported as max |delta|."""
    ref_path = os.path.join(DATA, "tln_tf_reference.json")
    if not os.path.exists(ref_path):
        pytest.skip("TensorFlow reference missing (work/baselines/scripts/tln_tf_reference.py)")
    with open(ref_path) as fh:
        ref = json.load(fh)
    d = baselines.load("tinylidarnet", TLN_ONNX)
    d.bind_scanner(n_beams=1081, fov=baselines.OUR_FOV, range_max=10.0)
    cmd = d.command(d.adapt(saturated(real_scans())))
    dsteer = float(np.abs(cmd[:, 0] - np.asarray(ref["steer_rad"])).max())
    dspeed = float(np.abs(cmd[:, 1] - np.asarray(ref["speed_mps"])).max())
    print(f"TF {ref['tensorflow']} vs onnxruntime on 100 real scans: "
          f"max |dsteer| {dsteer:.3e} rad, max |dspeed| {dspeed:.3e} m/s")
    assert dsteer < 1e-5 and dspeed < 1e-5, (dsteer, dspeed)


def test_tinylidarnet_batch_width_does_not_change_the_command():
    """Batch 1 and batch 100 must agree, or node-vs-adapter parity is about the batch size."""
    d = baselines.load("tinylidarnet", TLN_ONNX)
    d.bind_scanner(n_beams=1081, fov=baselines.OUR_FOV, range_max=10.0)
    r = d.adapt(saturated(real_scans()))
    whole = d.command(r)
    one = np.concatenate([d.command(r[i:i + 1]) for i in range(len(r))])
    assert np.abs(whole - one).max() == 0.0


# --------------------------------------------------------------------------- End2Race
#: `eval_singleagent.py:104-107`, the beam selection, quoted exactly. Asserted present in the
#: vendored file and then executed, so a change upstream fails this test instead of passing it.
E2R_SELECT_SRC = """        lidar = np.array(obs["scans"][0]).flatten()
        if len(lidar) > num_features:
            indices = np.linspace(0, len(lidar)-1, num_features, dtype=int)
            lidar = lidar[indices]
"""


def vendored_eval_source():
    f = os.path.join(E2R_REPO, "eval_singleagent.py")
    if not os.path.exists(f):
        pytest.skip(f"End2Race is not vendored at {E2R_REPO}")
    with open(f) as fh:
        return fh.read()


def test_end2race_beam_selection_is_the_vendored_one():
    src = vendored_eval_source()
    assert E2R_SELECT_SRC in src, ("eval_singleagent.py no longer contains the beam selection this "
                                   "driver reproduces; re-read it before trusting any result")
    ns = {"np": np, "num_features": e2r_mod.N_FEATURES,
          "obs": {"scans": [np.arange(e2r_mod.RAW_BEAMS, dtype=np.float32)]}}
    exec(E2R_SELECT_SRC.replace("        ", ""), ns)          # their four lines, dedented
    theirs = ns["lidar"]
    d = baselines.load("end2race", E2R_WEIGHTS)
    assert np.array_equal(theirs, np.arange(e2r_mod.RAW_BEAMS, dtype=np.float32)[d._idx])


def test_end2race_steer_clip_and_unclipped_speed_are_the_vendored_ones():
    src = vendored_eval_source()
    assert "ego_steer = np.clip(ego_steer, -0.52, 0.52)" in src
    assert "prev_speed = obs['linear_vels_x'][0]" in src
    d = baselines.load("end2race", E2R_WEIGHTS)
    assert d.steer_clip == pytest.approx(0.52)
    assert d.describe()["speed_output_clipped"] is False


def test_end2race_pressure_token_is_their_module_not_a_copy():
    """The token comes out of their `model.py` forward, so this checks we import rather than copy."""
    up = e2r_mod.import_upstream(e2r_mod.DEFAULT_REPO)
    src = open(os.path.join(e2r_mod.DEFAULT_REPO, "model.py")).read()
    assert "processed_lidar = (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2" in src
    import torch
    m = up.End2Race(mask_prob=0.0, hidden_scale=4)
    k = m.k.detach()
    x = torch.tensor([[[0.0] * e2r_mod.N_FEATURES]])
    token = (-1 / (1 + torch.exp(-k * x)) + 1) * 2
    assert float(token.max()) == pytest.approx(1.0, abs=1e-6)    # 0 m reads as "right here"


def test_end2race_num_features_deviation_is_one_line_and_declared():
    """The fair-comparison arm's single documented change to their model code."""
    up = e2r_mod.import_upstream(e2r_mod.DEFAULT_REPO, num_features=270)
    assert up._f1sim_deviation["num_features"] == 270
    m = up.End2Race(mask_prob=0.0, hidden_scale=4)
    assert tuple(m.k.shape) == (270,)
    assert m.gru.input_size == 270 + 270 // 6
    assert m.gru.hidden_size == (270 + 270 // 6) * 4
    # ... and the untouched build is still theirs
    up0 = e2r_mod.import_upstream(e2r_mod.DEFAULT_REPO)
    assert up0._f1sim_deviation is None
    assert tuple(up0.End2Race(mask_prob=0.0).k.shape) == (e2r_mod.N_FEATURES,)


# --------------------------------------------------------------------------- scan mapping
def test_map_scan_is_identity_when_the_windows_agree():
    r = saturated(real_scans())
    out = baselines.map_scan(r, src_fov=baselines.OUR_FOV, src_range_max=10.0,
                             dst_n=1081, dst_fov=baselines.OUR_FOV, dst_range_max=10.0)
    assert np.array_equal(out, r)


def test_map_scan_fills_exactly_the_bearings_a_270_degree_scanner_cannot_see():
    """Root's decision: 270 of End2Race's 360 features carry a measurement, 90 carry the fill."""
    r = saturated(real_scans()[:3])
    for fill in (30.0, 0.0):
        out = baselines.map_scan(r, src_fov=baselines.OUR_FOV, src_range_max=10.0,
                                 dst_n=1440, dst_fov=e2r_mod.RAW_FOV, dst_range_max=30.0, fill=fill)
        idx = np.linspace(0, 1439, 360, dtype=int)
        filled = (out[:, idx] == fill).sum(1)
        assert set(filled.tolist()) == {90}, filled
    assert baselines.unseen_fraction(src_fov=baselines.OUR_FOV, dst_n=1440,
                                     dst_fov=e2r_mod.RAW_FOV) == pytest.approx(0.25, abs=0.005)


def test_map_scan_places_a_return_at_the_bearing_it_was_measured_at():
    """A single close return must land at its own bearing, not at its own index."""
    src_n, src_fov = 1081, baselines.OUR_FOV
    r = np.full((1, src_n), 10.0, dtype=np.float32)
    j = 800                                                   # a beam right of centre
    r[0, j] = 1.0
    bearing = -0.5 * src_fov + j * src_fov / (src_n - 1)
    out = baselines.map_scan(r, src_fov=src_fov, src_range_max=10.0, dst_n=1440,
                             dst_fov=e2r_mod.RAW_FOV, dst_range_max=30.0)
    dst = np.linspace(-0.5 * e2r_mod.RAW_FOV, 0.5 * e2r_mod.RAW_FOV, 1440)
    landed = dst[int(np.argmin(out[0]))]
    assert abs(landed - bearing) < 2 * (e2r_mod.RAW_FOV / 1439), (landed, bearing)
