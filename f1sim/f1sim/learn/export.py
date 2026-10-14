"""Export the actor to ONNX (and TensorRT if available) for the Jetson; verify against torch.

A recurrent checkpoint (`--memory gru`) exports as `(scan, proprio, hidden) -> (action,
hidden_next)`: the state is an input and an output, never module state, because that is the only
form a graph runtime can carry. Extra scan channels widen the `scan` input; the sidecar JSON
records exactly what to feed and what to keep.

A grip-dial checkpoint (`meta["cond"]`, source `dial`) takes the dial as one more input, `dial`
(batch, 1), in friction units -- the number an operator sets, the ROS node's `dial` parameter. The
graph applies the checkpoint's own `(dial - offset) / scale` (`conditioning.mu_to_c`), so a consumer
never re-implements the normalisation; the sidecar records the spec. There is no default for it in
the graph: a conditional actor is never run on an implied number (`Actor._require_cond`).

A v2 checkpoint (`learn.model_v2`, `meta["arch"] == "v2"`) exports as `(scan, ego, dial, hidden) ->
(action, hidden_next)` -- `export_v2` -- with the observation contract's column table in the sidecar
(`learn.obs.contract_table_v2`), so the car builds exactly the vector the network was trained on.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .conditioning import CondSpec, mu_to_c
from .model import load_checkpoint


def _condition(spec: CondSpec, proprio, dial):
    """`c` for the actor from the graph's `dial` input: None for an unconditional checkpoint, zeros
    for the A0 arm (trained on c = 0, so it has no dial), the checkpoint's normalisation otherwise."""
    if not spec.dim:
        return None
    if spec.source == "zero":
        return proprio.new_zeros(proprio.shape[0], spec.dim)
    return mu_to_c(dial, spec).to(proprio.dtype)


def takes_dial(spec: CondSpec) -> bool:
    """Whether the exported graph has a `dial` input. A lab-oracle spec never reaches here
    (`load_checkpoint` refuses it), and the A0 arm's `c` is a constant zero."""
    return bool(spec.dim) and spec.source == "dial"


class ActorOnly(torch.nn.Module):
    def __init__(self, actor, cond: CondSpec = CondSpec()):
        super().__init__(); self.actor = actor; self.cond = cond

    def forward(self, scan, proprio, dial=None):
        return self.actor(scan, proprio, _condition(self.cond, proprio, dial))   # deterministic action in [-1, 1]


class ActorV2Only(torch.nn.Module):
    """The deployed v2 graph (`learn.model_v2.ActorV2`): (scan, ego, dial, hidden) -> (action,
    hidden_next).

    `dial` is in friction units, the number the ROS node's `dial` parameter holds; the graph applies
    the checkpoint's `(dial - offset) / scale` itself (`conditioning.mu_to_c`), so a consumer never
    re-implements the normalisation. No default anywhere: the v2 actor has no implied dial. The
    min-pool channel is computed in the graph from `scan`; the auxiliary heads are not in it."""

    def __init__(self, actor, cond: CondSpec):
        super().__init__(); self.actor = actor; self.cond = cond

    def forward(self, scan, ego, dial, hidden):
        return self.actor.step(scan, ego, mu_to_c(dial, self.cond).to(ego.dtype), hidden)


def export_v2(model, out: str, steps: int = 20, seed: int = 0) -> dict:
    """Export a v2 actor to ONNX (opset 17), write the sidecar, and check it against torch.

    The check runs `steps` control steps with the hidden state fed back, as the car runs it, on a
    corridor-like scan, and reports the worst |onnx - torch| over the action and the state. The
    redesign's X1 requirement is <= 1e-5 at 1081 beams."""
    import json
    from .obs import contract_table_v2, ego_columns
    model = model.eval()
    meta = model.meta
    cond = CondSpec.from_meta(meta["cond"])
    spec = model.obs_spec
    g = torch.Generator().manual_seed(seed)
    n, k, E = spec.n_beams, spec.n_frames, spec.ego_dim
    ramp = torch.linspace(0.15, 1.0, n)

    def scan_at(t):
        return (ramp[None, None, :] * (0.7 + 0.3 * torch.rand(1, k, 1, generator=g))
                + 0.02 * torch.rand(1, k, n, generator=g)).clamp(0, 1)

    m = ActorV2Only(model.actor, cond).eval()
    args = (scan_at(0), 0.3 * torch.randn(1, E, generator=g), torch.full((1, 1), 0.85),
            model.actor.initial_hidden(1))
    names, outs = ["scan", "ego", "dial", "hidden"], ["action", "hidden_next"]
    dyn = {"scan": {0: "batch"}, "ego": {0: "batch"}, "dial": {0: "batch"}, "hidden": {1: "batch"},
           "action": {0: "batch"}, "hidden_next": {1: "batch"}}
    torch.onnx.export(m, args, out, input_names=names, output_names=outs, opset_version=17,
                      dynamic_axes=dyn, dynamo=False)
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump({"arch": meta["arch"], "meta": meta, "spec": dict(vars(spec)),
                   "contract": contract_table_v2(spec).splitlines(),
                   "ego_columns": [c.name for c in ego_columns(spec)],
                   "inputs": {"scan": [1, k, n], "ego": [1, E], "dial": [1, 1],
                              "hidden": list(args[3].shape), "cond": cond.to_meta()},
                   "outputs": outs}, f, indent=1)
    report = {"onnx": out, "max_abs_err": None, "steps": int(steps)}
    try:
        import onnxruntime as ort
    except ImportError:
        return report
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(out, so, providers=["CPUExecutionProvider"])
    h_t, h_o, err = args[3], args[3].numpy(), 0.0
    for t in range(int(steps)):
        sc, eg = scan_at(t), 0.3 * torch.randn(1, E, generator=g)
        dial = torch.full((1, 1), 0.6 + 0.04 * (t % 10))
        with torch.no_grad():
            a_t, h_t = m(sc, eg, dial, h_t)
        a_o, h_o = sess.run(None, {"scan": sc.numpy(), "ego": eg.numpy(), "dial": dial.numpy(), "hidden": h_o})
        err = max(err, float(np.abs(a_o - a_t.numpy()).max()), float(np.abs(h_o - h_t.numpy()).max()))
    report["max_abs_err"] = err
    return report


def ort_step_ms(onnx_path: str, model, iters: int = 200, repeats: int = 5, threads: int = 1) -> float:
    """Milliseconds per control step of an exported v2 actor under onnxruntime, batch 1, the fastest
    of `repeats` blocks of `iters` (`learn.budget`'s protocol, for the graph the car would run)."""
    import time
    import onnxruntime as ort
    spec = model.obs_spec
    so = ort.SessionOptions(); so.intra_op_num_threads = int(threads); so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    feed = {"scan": np.random.rand(1, spec.n_frames, spec.n_beams).astype(np.float32),
            "ego": np.random.randn(1, spec.ego_dim).astype(np.float32),
            "dial": np.full((1, 1), 0.85, np.float32),
            "hidden": np.zeros((1, 1, model.actor.hidden_size), np.float32)}
    for _ in range(20):
        sess.run(None, feed)
    best = float("inf")
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        for _ in range(iters):
            sess.run(None, feed)
        best = min(best, (time.perf_counter() - t0) / iters)
    return best * 1e3


class RecurrentActorOnly(torch.nn.Module):
    """A memory actor as a pure function: (scan, proprio, hidden) -> (action, next hidden).

    The hidden state is an input and an output rather than module state, because that is the only
    form a graph runtime -- ONNX Runtime, TensorRT, the CUDA graph in the viewer -- can carry: the
    caller keeps one tensor and hands it back each step. `scan` here already includes any extra
    channel the checkpoint declares (`meta["scan_channels"]`); the deployment side builds those
    from the raw scan with `learn.obs.ScanAugment` before the forward, exactly as the ROS node
    does, and the sidecar JSON records which ones and in what order.
    """

    def __init__(self, actor, cond: CondSpec = CondSpec()):
        super().__init__(); self.actor = actor; self.cond = cond

    def forward(self, scan, proprio, hidden, dial=None):
        return self.actor.step(scan, proprio, _condition(self.cond, proprio, dial), hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--out", default=""); ap.add_argument("--trt", action="store_true")
    a = ap.parse_args()
    # No `allow_oracle`: a checkpoint trained with privileged opponent tokens
    # (`f1sim.opp_token`) has input columns the car cannot fill, and an ONNX graph that takes them
    # as part of `proprio` is a graph nobody can feed. `load_checkpoint` refuses it here by
    # default, with the reason; there is deliberately no flag on this tool to override that.
    model, extra = load_checkpoint(a.ckpt, "cpu"); model.eval()
    if (getattr(model, "meta", None) or {}).get("arch") == "v2":
        out = a.out or os.path.splitext(a.ckpt)[0] + ".onnx"
        rep = export_v2(model, out)
        err = rep["max_abs_err"]
        print(f"onnx ok: {out}  (v2: scan, ego, dial, hidden -> action, hidden_next)"
              + ("" if err is None else f"  max |onnx - torch| over {rep['steps']} recurrent steps = {err:.2e}"))
        if a.trt:
            raise SystemExit("--trt is not wired for the v2 graph yet; build the engine from the .onnx")
        return
    spec = extra.get("spec", {})
    if str(spec.get("opp_token") or "off") != "off":
        # An oracle is not a deployment target. The block is the other cars' exact position,
        # velocity and future, read out of the simulator; nothing on the car produces it, so an
        # exported graph would either be fed zeros -- a policy driving on an input that is always
        # "no car" -- or fed something invented. Refused here rather than at the ROS node alone,
        # because an .onnx outlives the checkpoint it came from.
        raise SystemExit(
            f"{os.path.basename(a.ckpt)} was trained with the privileged opponent block "
            f"(opp_token={spec['opp_token']!r}, future model "
            f"{spec.get('opp_future_model', 'plan')!r}). It is an oracle: the simulator's own "
            f"ground truth about the other cars, which no sensor on the car can measure. This "
            f"checkpoint is a measurement, not a deployable policy, and will not be exported.")
    k, N, P = model.meta["n_stack"], model.meta["n_beams"], model.meta["proprio_dim"]
    out = a.out or os.path.splitext(a.ckpt)[0] + ".onnx"
    channels = list((model.meta.get("scan_channels") or {}).get("channels") or ())
    k_in = k + len(channels)                       # what the forward takes, extra channels included
    scan, pro = torch.rand(1, k_in, N), torch.rand(1, P)
    recurrent = model.actor.has_memory
    cond = CondSpec.from_meta(model.meta.get("cond"))
    if recurrent:
        m = RecurrentActorOnly(model.actor, cond).eval()
        h = model.actor.initial_hidden(1)
        args, names = (scan, pro, h), ["scan", "proprio", "hidden"]
        outs = ["action", "hidden_next"]
        dyn = {"scan": {0: "batch"}, "proprio": {0: "batch"}, "hidden": {1: "batch"},
               "action": {0: "batch"}, "hidden_next": {1: "batch"}}
    else:
        m = ActorOnly(model.actor, cond).eval()
        args, names = (scan, pro), ["scan", "proprio"]
        outs = ["action"]
        dyn = {"scan": {0: "batch"}, "proprio": {0: "batch"}, "action": {0: "batch"}}
    dial = None
    if takes_dial(cond):
        # Any value in the trained range traces the same graph; this one is only the probe below.
        dial = torch.full((1, 1), 0.9)
        args, names = args + (dial,), names + ["dial"]
        dyn["dial"] = {0: "batch"}
    # dynamo=False: torch>=2.6 defaults to the dynamo exporter, which needs `onnxscript` and is not
    # installed here. The TorchScript path handles both scan stems (GroupNorm, adaptive pooling included).
    torch.onnx.export(m, args, out, input_names=names, output_names=outs, opset_version=17,
                      dynamic_axes=dyn, dynamo=False)
    import json
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        # `inputs` is what the deployment side has to build and carry, spelled out: a consumer that
        # reads only `meta` would not know that the scan it must feed is wider than `n_stack`, nor
        # that there is a state to keep between steps.
        json.dump({"spec": spec, "meta": model.meta,
                   "inputs": {"scan_channels": ["frame"] * k + channels,
                              "scan_width": k_in, "proprio": P,
                              "hidden": (list(h.shape) if recurrent else None),
                              # (batch, 1) friction the policy is told to use; the graph applies
                              # `cond`'s (dial - offset) / scale itself. None: no dial input.
                              "dial": ([1] if dial is not None else None),
                              "cond": cond.to_meta()},
                   "outputs": outs}, f, indent=1)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        feed = {"scan": scan.numpy(), "proprio": pro.numpy()}
        if recurrent:
            feed["hidden"] = h.numpy()
        if dial is not None:
            feed["dial"] = dial.numpy()
        y = sess.run(None, feed)
        ref = m(*args)
        ref = (ref,) if torch.is_tensor(ref) else ref
        err = max(float(np.abs(yi - ri.detach().numpy()).max()) for yi, ri in zip(y, ref))
        print(f"onnx ok: {out}  max |onnx - torch| = {err:.2e}"
              + ("  (recurrent: hidden in, hidden_next out)" if recurrent else "")
              + (f"  (grip dial input: {cond.describe()})" if dial is not None else ""))
    except ImportError:
        print(f"onnx written: {out} (install onnxruntime to verify)")
    if a.trt:
        import torch_tensorrt
        inputs = [torch_tensorrt.Input((1, k_in, N)), torch_tensorrt.Input((1, P))]
        if recurrent:
            inputs.append(torch_tensorrt.Input(tuple(h.shape)))
        if dial is not None:
            inputs.append(torch_tensorrt.Input((1, 1)))
        trt = torch_tensorrt.compile(m, inputs=inputs, enabled_precisions={torch.float16})
        torch.jit.save(trt, os.path.splitext(out)[0] + "_trt.ts"); print("tensorrt module saved")


if __name__ == "__main__":
    main()
