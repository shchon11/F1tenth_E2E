"""The current best policy on the real-car path: an exportable sector profile and the grip dial.

Two things blocked it, both found by the 2026-10-13 architecture audit (stage 0):

* the resnet stem's sector profile was `F.adaptive_max_pool1d(-x, 36)`, which the ONNX exporter
  refuses for 1081 beams (not a multiple of 36). It is a gather over the pool's own bins now, and
  what is pinned here is that the change is invisible: bit-identical values for every beam count,
  and an unchanged action from a real checkpoint;
* a grip-dial checkpoint needs `c` at every forward, and neither the export wrapper nor the ROS
  node passed one. The graph now takes `dial` as an input when the checkpoint has one, and the node
  refuses to start without its `dial` parameter instead of assuming a friction.
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from f1sim.learn import conditioning as C
from f1sim.learn.model import ActorCritic, ScanStem, save_checkpoint, sector_index

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAFEST = os.path.join(ROOT, "checkpoints", "iccas_bootcamp_safest.pt")


def _pool_reference(x, sectors=ScanStem.SECTORS):
    """What `_sector_profile` was before the change, verbatim."""
    return -F.adaptive_max_pool1d(-x[:, :1], sectors).flatten(1)


# ---------------------------------------------------------------- sector profile
@pytest.mark.parametrize("beams", [20, 36, 37, 64, 71, 100, 288, 1079, 1080, 1081, 1082, 1440])
def test_sector_profile_is_bit_identical_to_adaptive_max_pool(beams):
    """Every count, including fewer beams than sectors (overlapping bins) and the deployed 1081."""
    stem = ScanStem(2, beams, scan_stem="resnet")
    g = torch.Generator().manual_seed(beams)
    x = torch.rand(5, 2, beams, generator=g)
    # Ties are the common case on a real scan (every beam past range_max reads exactly 1.0), and a
    # quantised copy makes them common here too; -0.0 / 0.0 and an exact 0 included.
    q = (x * 8).floor() / 8
    q[0, 0, :3] = 0.0
    for v in (x, q):
        assert torch.equal(stem._sector_profile(v), _pool_reference(v))
    # The bins themselves are the pool's: [floor(i L/S), ceil((i+1) L/S)).
    idx = sector_index(beams, ScanStem.SECTORS)
    for i, row in enumerate(idx.tolist()):
        lo, hi = (i * beams) // 36, -(-((i + 1) * beams) // 36)
        assert set(row) == set(range(lo, hi))


def test_sector_profile_gradient_matches_the_pool_without_ties():
    stem = ScanStem(2, 1081, scan_stem="resnet")
    x = torch.rand(3, 2, 1081, requires_grad=True)
    w = torch.randn(3, 36)
    (stem._sector_profile(x) * w).sum().backward()
    g_new = x.grad.clone(); x.grad = None
    (_pool_reference(x) * w).sum().backward()
    assert torch.equal(g_new, x.grad)


def test_the_index_is_not_in_the_state_dict():
    """A buffer rebuilt from `n_beams`, so every existing checkpoint loads exactly as before."""
    stem = ScanStem(6, 1081, scan_stem="resnet")
    assert "sector_index" not in stem.state_dict()


@pytest.mark.skipif(not os.path.exists(SAFEST), reason="the deployed checkpoint is not on this disk")
def test_the_deployed_checkpoint_drives_exactly_as_before(monkeypatch):
    """Same weights, same inputs, the old pooling patched back in: the same action, bit for bit."""
    from f1sim.learn.model import load_checkpoint
    model, _ = load_checkpoint(SAFEST, "cpu"); model.eval()
    k, N, P = model.meta["n_stack"], model.meta["n_beams"], model.meta["proprio_dim"]
    torch.manual_seed(0)
    scan, pro = torch.rand(8, k, N), torch.randn(8, P) * 0.3
    scan[:, :, 500:540] = 1.0                                 # a block of saturated beams: ties
    c = C.mu_to_c(torch.linspace(0.6, 1.15, 8), C.CondSpec.from_meta(model.meta["cond"]))
    h = model.actor.initial_hidden(8)
    with torch.no_grad():
        new = model.actor.step(scan, pro, c, h)
        monkeypatch.setattr(ScanStem, "_sector_profile", lambda self, x: _pool_reference(x))
        old = model.actor.step(scan, pro, c, h)
    assert torch.equal(new[0], old[0]) and torch.equal(new[1], old[1])


# ---------------------------------------------------------------- export
def _dial_checkpoint(tmp_path, memory=True, source="dial", n_beams=1081):
    """A small resnet-stem, plan-output dial checkpoint at the deployed beam count, with the cond
    projection made nonzero so that the dial visibly reaches the action."""
    from f1sim.learn.memory import memory_spec
    spec = C.spec_for(source)
    torch.manual_seed(3)
    m = ActorCritic(n_stack=2, n_beams=n_beams, proprio_dim=12, priv_dim=9, act_dim=8,
                    scan_stem="resnet", cond_dim=spec.dim, cond=spec.to_meta(),
                    memory=(memory_spec(hidden_size=16) if memory else None))
    with torch.no_grad():
        m.actor.cond.weight.normal_(0, 1.0)
        if memory:
            m.actor.memory.out.weight.normal_(0, 0.3)
    p = str(tmp_path / f"dial_{source}_{int(memory)}.pt")
    save_checkpoint(p, m, {"spec": {}})
    return p, m.eval()


def _export(path, out, monkeypatch):
    from f1sim.learn import export
    monkeypatch.setattr("sys.argv", ["export", path, "--out", out])
    export.main()


@pytest.mark.parametrize("memory", [True, False])
def test_a_dial_checkpoint_exports_with_a_dial_input_and_matches_torch(tmp_path, monkeypatch, memory):
    ort = pytest.importorskip("onnxruntime")
    path, model = _dial_checkpoint(tmp_path, memory=memory)
    out = str(tmp_path / "m.onnx")
    _export(path, out, monkeypatch)
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    assert names == (["scan", "proprio", "hidden", "dial"] if memory else ["scan", "proprio", "dial"])
    side = json.load(open(os.path.splitext(out)[0] + ".json"))
    assert side["inputs"]["dial"] == [1] and side["inputs"]["cond"]["source"] == "dial"
    assert side["inputs"]["cond"]["offset"] == 1.0 and side["inputs"]["cond"]["scale"] == 0.25
    torch.manual_seed(5)
    B = 4
    scan, pro = torch.rand(B, 2, 1081), torch.randn(B, 12)
    h = model.actor.initial_hidden(B) if memory else None
    spec = C.CondSpec.from_meta(model.meta["cond"])
    actions = []
    for mu in (0.6, 0.73423, 1.0489, 1.15):
        dial = torch.full((B, 1), mu)
        feed = {"scan": scan.numpy(), "proprio": pro.numpy(), "dial": dial.numpy()}
        if memory:
            feed["hidden"] = h.numpy()
        got = sess.run(None, feed)
        with torch.no_grad():
            ref = model.actor.step(scan, pro, C.mu_to_c(dial, spec), h)
        assert np.abs(got[0] - ref[0].numpy()).max() < 1e-5
        if memory:
            assert np.abs(got[1] - ref[1].numpy()).max() < 1e-5
        actions.append(got[0])
    assert np.abs(actions[0] - actions[-1]).max() > 1e-3, "the dial input must reach the action"


def test_an_unconditional_checkpoint_exports_as_before(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    m = ActorCritic(n_stack=2, n_beams=1081, proprio_dim=12, priv_dim=9, act_dim=8, scan_stem="resnet")
    p = str(tmp_path / "u.pt"); save_checkpoint(p, m, {"spec": {}})
    out = str(tmp_path / "u.onnx")
    _export(p, out, monkeypatch)
    assert [i.name for i in ort.InferenceSession(out).get_inputs()] == ["scan", "proprio"]
    side = json.load(open(os.path.splitext(out)[0] + ".json"))
    assert side["inputs"]["dial"] is None and side["inputs"]["cond"]["dim"] == 0


def test_the_zero_arm_exports_with_its_constant_zero_and_no_dial(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    path, model = _dial_checkpoint(tmp_path, memory=False, source="zero")
    out = str(tmp_path / "z.onnx")
    _export(path, out, monkeypatch)
    sess = ort.InferenceSession(out)
    assert [i.name for i in sess.get_inputs()] == ["scan", "proprio"]
    scan, pro = torch.rand(2, 2, 1081), torch.randn(2, 12)
    with torch.no_grad():
        ref = model.actor(scan, pro, torch.zeros(2, 1))
    got = sess.run(None, {"scan": scan.numpy(), "proprio": pro.numpy()})[0]
    assert np.abs(got - ref.numpy()).max() < 1e-5


# ---------------------------------------------------------------- the ROS node's dial
def _deploy():
    pytest.importorskip("f1sim_ros")
    from f1sim_ros import deploy
    return deploy


def test_the_node_refuses_a_dial_checkpoint_without_a_dial(tmp_path):
    deploy = _deploy()
    _p, model = _dial_checkpoint(tmp_path, memory=False)
    for missing in (None, float("nan")):
        with pytest.raises(ValueError, match="needs the parameter 'dial'.*no default"):
            deploy.policy_condition(model, missing, "cpu")
    for outside in (0.40, 1.30):
        with pytest.raises(ValueError, match="outside the range"):
            deploy.policy_condition(model, outside, "cpu")
    c = deploy.policy_condition(model, 0.73423, "cpu")
    assert c.shape == (1, 1)
    assert torch.equal(c, C.mu_to_c(torch.tensor([0.73423]), C.CondSpec.from_meta(model.meta["cond"])))


def test_the_node_dial_is_ignored_by_an_unconditional_checkpoint_and_zero_for_a0(tmp_path):
    deploy = _deploy()
    m = ActorCritic(n_stack=2, n_beams=64, proprio_dim=12, priv_dim=9, act_dim=8)
    assert deploy.policy_condition(m, float("nan"), "cpu") is None
    assert deploy.policy_condition(m, 0.9, "cpu") is None
    _p, a0 = _dial_checkpoint(tmp_path, memory=False, source="zero", n_beams=64)
    assert torch.equal(deploy.policy_condition(a0, float("nan"), "cpu"), torch.zeros(1, 1))


def test_the_shipped_graph_yaml_dial_is_inside_the_trained_range():
    yaml = pytest.importorskip("yaml")
    from f1sim.learn.grip_estimator import MU_MAX, MU_MIN
    cfg = yaml.safe_load(open(os.path.join(ROOT, "f1sim_ros", "config", "graph.yaml")))
    dial = cfg["/**/f1sim_policy"]["ros__parameters"]["dial"]
    assert isinstance(dial, float) and C.DIAL_FLOOR <= dial <= MU_MAX
    assert dial == MU_MIN, "documented as the bottom of the training friction range"


def test_the_node_drives_a_dial_checkpoint_and_the_dial_reaches_the_plan(tmp_path):
    """`on_scan` passes `c`: the hand-assembled node of `test_policy_node_memory`, a dial
    checkpoint, two dials -- two different plans, and no ValueError from `_require_cond`."""
    pytest.importorskip("rclpy")
    from test_policy_node_memory import feed, make_node
    from f1sim.learn.obs import ObsSpec
    deploy = _deploy()
    spec = ObsSpec(n_beams=64, scan_stack=2, scan_stride=1, action_history=2, act_dim=8, hist_len=0,
                   range_max=10.0, v_max=8.0)
    torch.manual_seed(3)
    cs = C.spec_for("dial")
    m = ActorCritic(n_stack=2, n_beams=64, proprio_dim=spec.proprio_dim, priv_dim=9, act_dim=8,
                    cond_dim=1, cond=cs.to_meta()).eval()
    with torch.no_grad():
        m.actor.cond.weight.normal_(0, 1.0)
    plans = []
    for mu in (0.6, 1.15):
        node = make_node(m, spec)
        node.cond = deploy.policy_condition(m, mu, "cpu")
        feed(node, 2)
        assert len(node.pub_plan.msgs) == 2
        plans.append(np.array(node.pub_plan.msgs[-1].plan))
    assert np.abs(plans[0] - plans[1]).max() > 1e-3
    node = make_node(m, spec)                     # the class default: no condition at all
    with pytest.raises(ValueError, match="requires an explicit condition"):
        feed(node, 1)


def test_the_running_node_refuses_a_live_dial_change(tmp_path):
    """`c` is computed once at start, so `ros2 param set ... dial` on a running node must fail
    loudly rather than report success while the policy keeps driving the old dial."""
    rclpy = pytest.importorskip("rclpy")
    _deploy()
    from rclpy.parameter import Parameter
    from f1sim_ros.policy_node import PolicyNode
    from f1sim.learn.obs import ObsSpec
    spec = ObsSpec()                              # what the node builds for a checkpoint without one
    cs = C.spec_for("dial")
    m = ActorCritic(n_stack=spec.scan_stack, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                    priv_dim=9, act_dim=spec.act_dim, cond_dim=cs.dim, cond=cs.to_meta())
    path = str(tmp_path / "dial_node.pt")
    save_checkpoint(path, m, {"spec": {}})
    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init(args=["--ros-args", "-p", f"checkpoint:={path}", "-p", "device:=cpu",
                     "-p", "dial:=1.0489"])
    try:
        node = PolicyNode()
        c0 = node.cond.clone()
        for v in (0.8, 1.6, float("nan")):
            r = node.set_parameters([Parameter("dial", Parameter.Type.DOUBLE, v)])[0]
            assert not r.successful and "read-only" in r.reason
        assert node.get_parameter("dial").value == 1.0489 and torch.equal(node.cond, c0)
        node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_a_held_dial_checkpoint_is_told_its_own_dial():
    """`deploy.node_dial`: the launch's `dial:=`, else the dial the checkpoint was trained with held, else
    graph.yaml's. Until 2026-10-23 graph.yaml's 0.73423 reached the Korea entries, trained at 1.0489 only."""
    deploy = _deploy()
    held = {"spec": {}, "experiment": {"dial_constant": 1.0489}}
    assert deploy.node_dial("", 0.73423, held) == (1.0489, "the checkpoint (trained with it held)")
    assert deploy.node_dial("", 0.73423, {"dial_constant": 1.0489})[0] == 1.0489          # an archived copy
    assert deploy.node_dial(" 0.9 ", 0.73423, held) == (0.9, "launch dial:=")
    assert deploy.node_dial("", 0.73423, {"spec": {}}) == (0.73423, "graph.yaml")         # drawn dials


def test_the_node_tells_a_held_dial_checkpoint_its_dial_over_graph_yaml(tmp_path):
    rclpy = pytest.importorskip("rclpy")
    _deploy()
    from f1sim_ros.policy_node import PolicyNode
    from f1sim.learn.obs import ObsSpec
    spec = ObsSpec()
    cs = C.spec_for("dial")
    m = ActorCritic(n_stack=spec.scan_stack, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                    priv_dim=9, act_dim=spec.act_dim, cond_dim=cs.dim, cond=cs.to_meta())
    path = str(tmp_path / "held_dial_node.pt")
    save_checkpoint(path, m, {"spec": {}, "experiment": {"dial_constant": 1.0489}})
    for args, want in ((["-p", "dial:=0.73423"], 1.0489), (["-p", "dial:=0.73423", "-p", "dial_override:=0.9"], 0.9)):
        if rclpy.ok():
            rclpy.shutdown()
        rclpy.init(args=["--ros-args", "-p", f"checkpoint:={path}", "-p", "device:=cpu", *args])
        try:
            node = PolicyNode()
            assert node.dial == pytest.approx(want)
            assert torch.allclose(node.cond, C.mu_to_c(torch.tensor([want]), cs))
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
