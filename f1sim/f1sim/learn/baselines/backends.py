"""How a vendored network is actually executed here, and the record of which way it went.

Two backends, and the choice is recorded rather than inferred:

* **onnx** -- `onnxruntime`, for TinyLidarNet. Its weights are a Keras `.h5`, and TensorFlow is not
  in this project's venv. Installing it there is not a neutral act: TF pins protobuf and numpy
  ranges and that venv is shared with other workers' running training jobs. So the conversion runs
  once, in an isolated venv (`work/baselines/scripts/tln_to_onnx.py`), and the runtime here loads
  the `.onnx`. CONTRACT.md asks which of the two branches ran; `describe()` answers it.
* **torch** -- End2Race is PyTorch, so its own `model.py` is imported from the vendored checkout and
  the published `state_dict` is loaded into it. Nothing is transcribed.

A backend is deliberately thin. It owns loading, the forward call and the provenance string. It
owns no preprocessing at all -- that is the driver's, and it is the thing the parity test is about.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np

from .common import BaselineError


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class OnnxBackend:
    """One ONNX graph on the CPU execution provider, batch dimension dynamic.

    Single-threaded by default. Not a performance choice: it is the *determinism* choice. The intra
    -op thread pool changes reduction order, so a multi-threaded session can move the last bits of
    an output between runs, and the node/adapter parity claim is checked to 1e-5 -- close enough to
    that noise to be worth removing. It is also what `learn/budget.py` times the project's own actor
    under, so the two millisecond figures are comparable.
    """

    def __init__(self, path: str, threads: int = 1):
        try:
            import onnxruntime as ort
        except ImportError as exc:                                     # pragma: no cover
            raise BaselineError(
                "onnxruntime is not installed, and TinyLidarNet's weights are a Keras model this "
                "project cannot load natively. Install onnxruntime, or point the entry at a "
                "TensorFlow-capable interpreter.") from exc
        if not os.path.exists(path):
            raise BaselineError(f"ONNX model does not exist: {path}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        self.path = os.path.abspath(path)
        self.sha256 = sha256_file(self.path)
        self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
        self.version = ort.__version__
        ins = self.sess.get_inputs()
        if len(ins) != 1:
            raise BaselineError(f"{path}: {len(ins)} inputs, expected 1")
        self.input_name = ins[0].name
        self.input_shape = list(ins[0].shape)
        #: The conversion's provenance, if it is lying next to the model. Carried into the protocol
        #: block of every benchmark row: "which TF produced these weights" is part of the result.
        self.provenance = None
        side = os.path.splitext(self.path)[0] + ".json"
        if os.path.exists(side):
            with open(side) as fh:
                p = json.load(fh)
            if p.get("onnx_sha256") and p["onnx_sha256"] != self.sha256:
                raise BaselineError(
                    f"{side} records onnx_sha256 {p['onnx_sha256'][:16]} but {os.path.basename(self.path)} "
                    f"hashes to {self.sha256[:16]}: the conversion record does not describe this file")
            self.provenance = {k: p[k] for k in
                               ("source", "source_sha256", "n_beams", "n_params", "opset",
                                "tensorflow", "tf2onnx") if k in p}
        self.threads = int(threads)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: x})[0]

    def describe(self) -> dict:
        return {"backend": "onnxruntime", "version": self.version, "providers": ["CPUExecutionProvider"],
                "threads": self.threads, "path": self.path, "sha256": self.sha256,
                "input_shape": self.input_shape, "conversion": self.provenance,
                "why": "TensorFlow is not installed in this venv; the Keras model was converted once "
                       "with tf2onnx and is run through onnxruntime"}


class KerasBackend:
    """TinyLidarNet's `.h5` (or `.tflite`) through TensorFlow, when TensorFlow is importable.

    CONTRACT.md's first branch: *"load in TF if available, otherwise convert to ONNX once"*. It is
    implemented rather than assumed absent, so `describe()["backend"]` is a fact about what ran and
    not about what was installed on one machine. In this project's venv the import fails and
    `tinylidarnet.load` takes the ONNX branch; `work/baselines/scripts/tln_to_onnx.py` is the
    "once", and the conversion is checked against TensorFlow's own outputs on 100 real bag scans.

    Not used for training: the fair-comparison arm rebuilds the same architecture in PyTorch
    (`tinylidarnet_torch.py`) and proves it reproduces these weights first.
    """

    def __init__(self, path: str, threads: int = 1):
        import os as _os

        _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        try:
            import tensorflow as tf
        except ImportError as exc:                                     # pragma: no cover
            raise BaselineError(
                f"{path} is a TensorFlow model and TensorFlow is not importable here. Convert it "
                f"once with work/baselines/scripts/tln_to_onnx.py and pass the .onnx instead -- "
                f"installing TensorFlow into this venv would repin protobuf and numpy under other "
                f"jobs sharing it.") from exc
        if not _os.path.exists(path):
            raise BaselineError(f"model does not exist: {path}")
        self.path = _os.path.abspath(path)
        self.sha256 = sha256_file(self.path)
        self.version = tf.__version__
        self.threads = int(threads)
        self.provenance = None
        self.tflite = self.path.endswith(".tflite")
        if self.tflite:
            self._interp = tf.lite.Interpreter(model_path=self.path)
            self._interp.allocate_tensors()
            self._in = self._interp.get_input_details()[0]
            self._out = self._interp.get_output_details()[0]
            self.input_shape = [int(d) for d in self._in["shape"]]
            self._model = None
        else:
            self._model = tf.keras.models.load_model(self.path, compile=False)
            self.input_shape = [None if d is None else int(d) for d in self._model.input_shape]
            self._interp = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._model is not None:
            return np.asarray(self._model(x, training=False))
        # TFLite freezes the batch at 1 unless resized; resize rather than loop, and only when the
        # width actually changes, because `allocate_tensors` is not free.
        if int(self._in["shape"][0]) != x.shape[0]:
            self._interp.resize_tensor_input(self._in["index"], list(x.shape))
            self._interp.allocate_tensors()
            self._in = self._interp.get_input_details()[0]
            self._out = self._interp.get_output_details()[0]
        self._interp.set_tensor(self._in["index"], x)
        self._interp.invoke()
        return self._interp.get_tensor(self._out["index"])

    def describe(self) -> dict:
        return {"backend": "tensorflow" + ("/tflite" if self.tflite else "/keras"),
                "version": self.version, "providers": ["CPU"], "threads": self.threads,
                "path": self.path, "sha256": self.sha256, "input_shape": self.input_shape,
                "conversion": None,
                "why": "TensorFlow was importable, so the Keras model is loaded natively"}


class TorchBackend:
    """A `torch.nn.Module` from the vendored repo, with the published `state_dict` loaded strictly.

    Strict: `load_state_dict(..., strict=True)`. A tolerant load is how "the architecture drifted"
    and "the baseline is just bad" come to look the same.
    """

    def __init__(self, module, path: str, device="cpu", threads: Optional[int] = 1):
        import torch

        if not os.path.exists(path):
            raise BaselineError(f"checkpoint does not exist: {path}")
        self.path = os.path.abspath(path)
        self.sha256 = sha256_file(self.path)
        sd = torch.load(self.path, map_location="cpu", weights_only=True)
        if not isinstance(sd, dict):
            raise BaselineError(f"{path}: expected a state_dict, got {type(sd).__name__}")
        # A file this project wrote wraps the weights with the provenance that makes the row
        # readable (`baselines/__main__.py:_save`); a file the upstream repo wrote is the bare
        # state_dict (`End2Race/train.py:148`). Both are accepted, and the wrapped metadata is kept.
        self.meta = sd.get("meta") if "state_dict" in sd else None
        module.load_state_dict(sd.get("state_dict", sd), strict=True)
        self.device = torch.device(device)
        self.module = module.to(self.device).eval()
        self.version = torch.__version__
        self.threads = threads
        self.n_params = int(sum(p.numel() for p in module.parameters()))

    def describe(self) -> dict:
        return {"backend": "torch", "version": self.version, "device": str(self.device),
                "threads": self.threads, "path": self.path, "sha256": self.sha256,
                "n_params": self.n_params, "trained_here": self.meta}
