"""The v2 actor on the car path: ONNX at 1081 beams, parity with torch, the contract in the sidecar.

Requirement X1 of the redesign: the policy runs in the real node and as ONNX at 1081 beams with
|onnx - torch| <= 1e-5. v1's resnet stem could not be exported until its sector bypass was rewritten
(stage 0); the v2 stem has no bypass at all. The graph takes the dial in friction units and applies
the checkpoint's normalisation itself, carries the GRU state as an input and an output, and holds
none of the train-time heads.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from f1sim.learn.export import ActorV2Only, export_v2  # noqa: E402
from f1sim.learn.model_v2 import ActorCriticV2  # noqa: E402


@pytest.mark.parametrize("n_frames", [1, 3])
def test_the_v2_actor_exports_at_1081_beams_within_1e_5_of_torch(tmp_path, n_frames):
    torch.manual_seed(0)
    m = ActorCriticV2(n_beams=1081, n_frames=n_frames, aux={"b8": True, "near_car": True}).eval()
    path = str(tmp_path / "actor_v2.onnx")
    rep = export_v2(m, path, steps=20)
    assert rep["max_abs_err"] is not None and rep["max_abs_err"] <= 1e-5, rep
    g = onnx.load(path)
    assert [i.name for i in g.graph.input] == ["scan", "ego", "dial", "hidden"]
    assert [o.name for o in g.graph.output] == ["action", "hidden_next"]
    names = " ".join(i.name for i in g.graph.initializer)
    assert not any(h in names for h in ("b8", "near_car")), "a train-time head is in the graph"
    side = json.load(open(str(tmp_path / "actor_v2.json")))
    assert side["arch"] == "v2" and len(side["ego_columns"]) == 210
    assert side["inputs"]["scan"] == [1, n_frames, 1081] and side["inputs"]["hidden"] == [1, 1, 128]
    assert any("dial" in line for line in side["contract"])


def test_the_contract_without_issued_actions_exports_its_42_columns(tmp_path):
    """Arm A of the copycat study (`ObsSpecV2.ego_actions` False): the graph's `ego` input is 42 wide
    and the sidecar's table -- what the car builds -- names no issued action."""
    from f1sim.learn.obs import ObsSpecV2
    torch.manual_seed(4)
    spec = ObsSpecV2(ego_actions=False)
    m = ActorCriticV2(n_beams=1081, ego_dim=spec.ego_dim, obs=dict(vars(spec))).eval()
    path = str(tmp_path / "actor_a.onnx")
    rep = export_v2(m, path, steps=5)
    assert rep["max_abs_err"] is not None and rep["max_abs_err"] <= 1e-5, rep
    side = json.load(open(str(tmp_path / "actor_a.json")))
    assert side["inputs"]["ego"] == [1, 42] and len(side["ego_columns"]) == 42
    assert not any("action[" in c for c in side["ego_columns"]) and side["spec"]["ego_actions"] is False
    assert any("issued actions: none" in line for line in side["contract"])


def test_the_graph_reads_the_dial_in_mu_and_runs_a_batch(tmp_path):
    from f1sim.learn.conditioning import CondSpec, mu_to_c
    torch.manual_seed(1)
    m = ActorCriticV2(n_beams=1081).eval()
    path = str(tmp_path / "a.onnx")
    export_v2(m, path, steps=2)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    g = torch.Generator().manual_seed(2)
    scan = torch.rand(3, 1, 1081, generator=g); ego = 0.3 * torch.randn(3, 210, generator=g)
    dial = torch.tensor([[0.55], [0.75], [1.0]])
    h = torch.zeros(1, 3, 128)
    a_o, h_o = sess.run(None, {"scan": scan.numpy(), "ego": ego.numpy(), "dial": dial.numpy(), "hidden": h.numpy()})
    with torch.no_grad():
        c = mu_to_c(dial, CondSpec.from_meta(m.meta["cond"]))
        assert torch.allclose(c[:, 0], (dial[:, 0] - 1.0) / 0.25)
        a_t, h_t = m.actor.step(scan, ego, c, h)
    assert np.abs(a_o - a_t.numpy()).max() <= 1e-5 and np.abs(h_o - h_t.numpy()).max() <= 1e-5
    a_hi, _ = sess.run(None, {"scan": scan.numpy(), "ego": ego.numpy(), "dial": (dial + 0.2).numpy(), "hidden": h.numpy()})
    assert np.abs(a_hi - a_o).max() > 0, "the dial must reach the action through the graph"


def test_the_exporter_cli_takes_a_v2_checkpoint(tmp_path, monkeypatch, capsys):
    from f1sim.learn import export
    from f1sim.learn.model import save_checkpoint
    torch.manual_seed(3)
    m = ActorCriticV2(n_beams=1081).eval()
    ck = str(tmp_path / "v2.pt")
    save_checkpoint(ck, m, {"spec": vars(m.obs_spec)})
    monkeypatch.setattr(sys, "argv", ["export", ck])
    export.main()
    out = capsys.readouterr().out
    assert "onnx ok" in out and "v2" in out and os.path.isfile(str(tmp_path / "v2.onnx"))


def test_the_wrapper_is_the_actors_step():
    torch.manual_seed(4)
    m = ActorCriticV2(n_beams=271).eval()
    from f1sim.learn.conditioning import CondSpec
    w = ActorV2Only(m.actor, CondSpec.from_meta(m.meta["cond"]))
    scan, ego, dial = torch.rand(2, 1, 271), torch.randn(2, 210), torch.full((2, 1), 0.8)
    with torch.no_grad():
        a, h = w(scan, ego, dial, m.actor.initial_hidden(2))
        b, k = m.actor.step(scan, ego, (dial - 1.0) / 0.25, m.actor.initial_hidden(2))
    assert torch.equal(a, b) and torch.equal(h, k)


def test_the_budget_table_runs_and_leaves_no_files(tmp_path, monkeypatch):
    """`python -m f1sim.learn.budget --v2`: the numbers are a CPU proxy for the Orin; what is
    pinned is that the table is produced and its temporary graphs are gone afterwards."""
    import tempfile
    from f1sim.learn.budget import v2_budget
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    threads = torch.get_num_threads()
    try:
        rows = v2_budget([1], n_beams=271, iters=3, repeats=1)
    finally:
        torch.set_num_threads(threads)
    r = rows[0]
    assert r["params"]["actor (deployed)"] == 861_616 and r["torch_ms"] > 0      # 271 beams: a 9-cell neck
    assert r["onnx_max_abs_err"] <= 1e-5 and r["onnxruntime_ms"] > 0
    assert os.listdir(str(tmp_path)) == []
