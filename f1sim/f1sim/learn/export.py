"""Export the actor to ONNX (and TensorRT if available) for the Jetson; verify against torch.

A recurrent checkpoint (`--memory gru`) exports as `(scan, proprio, hidden) -> (action,
hidden_next)`: the state is an input and an output, never module state, because that is the only
form a graph runtime can carry. Extra scan channels widen the `scan` input; the sidecar JSON
records exactly what to feed and what to keep.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .model import load_checkpoint


class ActorOnly(torch.nn.Module):
    def __init__(self, actor):
        super().__init__(); self.actor = actor

    def forward(self, scan, proprio):
        return self.actor(scan, proprio)          # deterministic action in [-1, 1]


class RecurrentActorOnly(torch.nn.Module):
    """A memory actor as a pure function: (scan, proprio, hidden) -> (action, next hidden).

    The hidden state is an input and an output rather than module state, because that is the only
    form a graph runtime -- ONNX Runtime, TensorRT, the CUDA graph in the viewer -- can carry: the
    caller keeps one tensor and hands it back each step. `scan` here already includes any extra
    channel the checkpoint declares (`meta["scan_channels"]`); the deployment side builds those
    from the raw scan with `learn.obs.ScanAugment` before the forward, exactly as the ROS node
    does, and the sidecar JSON records which ones and in what order.
    """

    def __init__(self, actor):
        super().__init__(); self.actor = actor

    def forward(self, scan, proprio, hidden):
        return self.actor.step(scan, proprio, None, hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--out", default=""); ap.add_argument("--trt", action="store_true")
    a = ap.parse_args()
    # No `allow_oracle`: a checkpoint trained with privileged opponent tokens
    # (`f1sim.opp_token`) has input columns the car cannot fill, and an ONNX graph that takes them
    # as part of `proprio` is a graph nobody can feed. `load_checkpoint` refuses it here by
    # default, with the reason; there is deliberately no flag on this tool to override that.
    model, extra = load_checkpoint(a.ckpt, "cpu"); model.eval()
    spec = extra.get("spec", {})
    k, N, P = model.meta["n_stack"], model.meta["n_beams"], model.meta["proprio_dim"]
    out = a.out or os.path.splitext(a.ckpt)[0] + ".onnx"
    channels = list((model.meta.get("scan_channels") or {}).get("channels") or ())
    k_in = k + len(channels)                       # what the forward takes, extra channels included
    scan, pro = torch.rand(1, k_in, N), torch.rand(1, P)
    recurrent = model.actor.has_memory
    if recurrent:
        m = RecurrentActorOnly(model.actor).eval()
        h = model.actor.initial_hidden(1)
        args, names = (scan, pro, h), ["scan", "proprio", "hidden"]
        outs = ["action", "hidden_next"]
        dyn = {"scan": {0: "batch"}, "proprio": {0: "batch"}, "hidden": {1: "batch"},
               "action": {0: "batch"}, "hidden_next": {1: "batch"}}
    else:
        m = ActorOnly(model.actor).eval()
        args, names = (scan, pro), ["scan", "proprio"]
        outs = ["action"]
        dyn = {"scan": {0: "batch"}, "proprio": {0: "batch"}, "action": {0: "batch"}}
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
                              "hidden": (list(h.shape) if recurrent else None)},
                   "outputs": outs}, f, indent=1)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        feed = {"scan": scan.numpy(), "proprio": pro.numpy()}
        if recurrent:
            feed["hidden"] = h.numpy()
        y = sess.run(None, feed)
        ref = m(*args)
        ref = (ref,) if torch.is_tensor(ref) else ref
        err = max(float(np.abs(yi - ri.detach().numpy()).max()) for yi, ri in zip(y, ref))
        print(f"onnx ok: {out}  max |onnx - torch| = {err:.2e}"
              + ("  (recurrent: hidden in, hidden_next out)" if recurrent else ""))
    except ImportError:
        print(f"onnx written: {out} (install onnxruntime to verify)")
    if a.trt:
        import torch_tensorrt
        inputs = [torch_tensorrt.Input((1, k_in, N)), torch_tensorrt.Input((1, P))]
        if recurrent:
            inputs.append(torch_tensorrt.Input(tuple(h.shape)))
        trt = torch_tensorrt.compile(m, inputs=inputs, enabled_precisions={torch.float16})
        torch.jit.save(trt, os.path.splitext(out)[0] + "_trt.ts"); print("tensorrt module saved")


if __name__ == "__main__":
    main()
