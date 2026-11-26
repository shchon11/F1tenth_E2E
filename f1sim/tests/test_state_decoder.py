"""The state decoder file (11-26, f1sim/learn/state_decoder.py): one `.npz` read by the console in torch and by the site in numpy
(docs/research/plan-output-2026-11-01/tools/dec_replay.py) -- both must give the same answer, in the same units."""
import importlib.util
import math
import os

import numpy as np
import torch

from f1sim.learn import state_decoder as SD

TOOLS = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "research", "plan-output-2026-11-01", "tools")


def _random_decoders(tmp_path, F=12, H=6, hidden=64):
    torch.manual_seed(0)
    nets = {s: SD.DecoderNet({"feat": F, "mem": H, "feat+mem": F + H}[s], hidden) for s in SD.SETS}
    norms = {s: (np.random.RandomState(1).randn(n.towers["pe"][0].in_features).astype(np.float32),
                 np.random.RandomState(2).rand(n.towers["pe"][0].in_features).astype(np.float32) + 0.5)
             for s, n in nets.items()}
    path = str(tmp_path / "x_dec.npz")
    SD.save(path, nets, norms, [-10.0, 30.0, -5.0, 25.0], [2.0, 0.2, 1.0, 0.1], F, H, record="r", checkpoint="run/plan_u5.pt",
            hidden=hidden, has_opp=True)
    return path, nets, norms


def test_save_load_round_trip(tmp_path):
    path, nets, norms = _random_decoders(tmp_path)
    assert path.endswith("_dec.npz") and os.path.isfile(path)
    dec = SD.StateDecoder.load(path)
    assert set(dec.nets) == set(SD.SETS) and dec.feat_dim == 12 and dec.mem_dim == 6 and dec.checkpoint == "run/plan_u5.pt"
    feat, mem = np.random.RandomState(3).randn(7, 12).astype(np.float32), np.random.RandomState(4).randn(7, 6).astype(np.float32)
    for s in SD.SETS:
        x = SD.inputs_of(torch.as_tensor(feat), torch.as_tensor(mem), s)
        mu, sd = norms[s]
        with torch.no_grad():
            ref = nets[s].half().float()((x - torch.as_tensor(mu)) / torch.as_tensor(sd))     # the file keeps fp16 weights
        got = dec.raw(feat, mem, s)
        for k in ref:
            assert torch.allclose(ref[k], got[k], atol=1e-4), (s, k)
    out = dec.run(feat, mem)
    assert out["pose"].shape == (7, 3) and out["wall"].shape == (7, SD.NX, SD.NY) and out["opp_fut"].shape == (7, 2, 2)
    assert np.all((out["box"] >= 0) & (out["box"] <= 1))


def test_site_numpy_matches_torch(tmp_path):
    spec = importlib.util.spec_from_file_location("dec_replay", os.path.join(TOOLS, "dec_replay.py"))
    DR = importlib.util.module_from_spec(spec); spec.loader.exec_module(DR)
    path, _, _ = _random_decoders(tmp_path)
    dec, npd = SD.StateDecoder.load(path), DR.NPDecoder(path)
    feat, mem = np.random.RandomState(5).randn(9, 12).astype(np.float32), np.random.RandomState(6).randn(9, 6).astype(np.float32)
    for s in SD.SETS:
        a, b = dec.raw(feat, mem, s), npd.run(feat, mem, s)
        for k in a:
            assert np.allclose(a[k].numpy(), b[k], atol=1e-3), (s, k)


def test_grid_and_paths():
    c = SD.cell_centres()
    assert c.shape == (SD.N_CELL, 2)
    # cell (ix, iy) at ix * NY + iy: x ahead from GRID_X0, y left from GRID_Y0
    assert np.allclose(c[0], [SD.GRID_X0 + SD.GRID_RES / 2, SD.GRID_Y0 + SD.GRID_RES / 2])
    assert np.allclose(c[1], [SD.GRID_X0 + SD.GRID_RES / 2, SD.GRID_Y0 + 1.5 * SD.GRID_RES])
    assert SD.decoder_path_of("/a/b/plan_u950.pt") == "/a/b/plan_u950_dec.npz"
    assert abs(SD.wrap(3 * math.pi) - math.pi) < 1e-9 or abs(SD.wrap(3 * math.pi) + math.pi) < 1e-9
