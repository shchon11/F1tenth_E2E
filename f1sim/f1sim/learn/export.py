"""Export the actor to ONNX (and TensorRT if available) for the Jetson; verify against torch."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--out", default=""); ap.add_argument("--trt", action="store_true")
    a = ap.parse_args()
    model, extra = load_checkpoint(a.ckpt, "cpu"); model.eval()
    spec = extra.get("spec", {})
    k, N, P = model.meta["n_stack"], model.meta["n_beams"], model.meta["proprio_dim"]
    out = a.out or os.path.splitext(a.ckpt)[0] + ".onnx"
    m = ActorOnly(model.actor).eval()
    scan, pro = torch.rand(1, k, N), torch.rand(1, P)
    torch.onnx.export(m, (scan, pro), out, input_names=["scan", "proprio"], output_names=["action"], opset_version=17,
                      dynamic_axes={"scan": {0: "batch"}, "proprio": {0: "batch"}, "action": {0: "batch"}})
    # torch's exporter writes the weights beside the graph as <out>.data; a Jetson that only got the
    # .onnx would load a model with no weights. Fold them back in so the file is self-contained.
    try:
        import onnx
        g = onnx.load(out)                                   # reads the external data next to it
        onnx.save_model(g, out, save_as_external_data=False)
        if os.path.exists(out + ".data"):
            os.remove(out + ".data")
        need = sum(p.numel() for p in m.parameters()) * 4 * 0.9
        if os.path.getsize(out) < need:
            raise RuntimeError(f"{out} is {os.path.getsize(out)} bytes, too small to hold the weights")
    except ImportError:
        print("onnx not installed: weights may sit in a separate .data file, copy it along with the .onnx")
    import json
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump({"spec": spec, "meta": model.meta}, f, indent=1)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"scan": scan.numpy(), "proprio": pro.numpy()})[0]
        err = np.abs(y - m(scan, pro).detach().numpy()).max()
        print(f"onnx ok: {out}  max |onnx - torch| = {err:.2e}")
    except ImportError:
        print(f"onnx written: {out} (install onnxruntime to verify)")
    if a.trt:
        import torch_tensorrt
        trt = torch_tensorrt.compile(m, inputs=[torch_tensorrt.Input((1, k, N)), torch_tensorrt.Input((1, P))], enabled_precisions={torch.float16})
        torch.jit.save(trt, os.path.splitext(out)[0] + "_trt.ts"); print("tensorrt module saved")


if __name__ == "__main__":
    main()
