"""TinyLidarNet's architecture in PyTorch, for the fair-comparison arm — and proof it is theirs.

The published weights are Keras and run here through ONNX (`backends.OnnxBackend`), which is fine
for *evaluating* them and useless for *training* a new one: the DAgger loop has to put the student
back in the car after every iteration, and routing that through a TensorFlow venv and a tf2onnx
export each time is three moving parts where one will do.

So the architecture is rebuilt here, layer for layer from `train.py:170-181`, and then **checked
rather than asserted**: `load_keras_weights` transplants the published `.h5` weights into this
module and `verify_against_onnx` runs both on the 100 real bag scans. If the two agree to float
noise, this is their network; if they do not, the port is wrong and the fair-comparison arm would
be measuring something else.

Their training hyperparameters, all repo defaults (`train.py:58-62,187-196`):

    Adam(5e-5), loss 'huber', batch 64, 20 epochs, 85/15 split, shuffle(random_state=62)

with three details that are Keras defaults rather than written in their file, and that PyTorch does
not share -- so they are set explicitly here and listed in the checkpoint's provenance:

  * `glorot_uniform` kernels and zero biases on every Conv1D and Dense (Keras layer defaults);
    PyTorch would use kaiming-uniform with a fan-in bias;
  * `Adam(epsilon=1e-7)`; PyTorch's default is 1e-8;
  * Keras `huber` is delta = 1.0, which is `torch.nn.HuberLoss(delta=1.0)` -- NOT
    `SmoothL1Loss(beta=1.0)`, which is the same curve divided by delta.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

#: `train.py:171-181`, in order: (filters, kernel, stride) then the dense widths.
CONV = ((24, 10, 4), (36, 8, 4), (48, 4, 2), (64, 3, 1), (64, 3, 1))
DENSE = (100, 50, 10)
N_ACTIONS = 2


class TinyLidarNetTorch(nn.Module):
    """`(B, n_beams, 1)` metres -> `(B, 2)` in [-1, 1]. The tensor layout is Keras's, not PyTorch's.

    Channels-last in, channels-last out, because that is the tensor the driver's `_prepare` already
    builds for the ONNX backend: one `_prepare`, two backends, and nothing between them that could
    transpose differently.
    """

    def __init__(self, n_beams: int = 1081):
        super().__init__()
        self.n_beams = int(n_beams)
        layers, c_in, length = [], 1, self.n_beams
        for c_out, k, s in CONV:
            layers += [nn.Conv1d(c_in, c_out, k, stride=s), nn.ReLU()]
            length = (length - k) // s + 1              # Keras 'valid' padding
            c_in = c_out
        if length <= 0:
            raise ValueError(f"{n_beams} beams is too short for this convolution stack")
        self.conv = nn.Sequential(*layers)
        self.flat_dim = c_in * length
        d = []
        prev = self.flat_dim
        for w in DENSE:
            d += [nn.Linear(prev, w), nn.ReLU()]
            prev = w
        d += [nn.Linear(prev, N_ACTIONS), nn.Tanh()]    # `train.py:180`
        self.head = nn.Sequential(*d)
        self.reset_parameters()

    def reset_parameters(self):
        """Keras layer defaults: glorot_uniform kernel, zero bias."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[2] != 1:
            raise ValueError(f"expected (B, {self.n_beams}, 1), got {tuple(x.shape)}")
        # `.transpose(1, 2)` back to (B, L, C) BEFORE flattening, and it is not cosmetic: Keras
        # `Flatten` on a channels-last `(B, L, C)` produces index `l * C + c`, PyTorch's native
        # `(B, C, L)` flatten produces `c * L + l`. The first dense layer's 1792 weights are laid
        # out for the first order, so flattening the other way transposes its input and the
        # transplanted network drives like a different one -- which it did: 0.99 of a +-1 output.
        return self.head(self.conv(x.transpose(1, 2)).transpose(1, 2).flatten(1))

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def load_keras_weights(model: TinyLidarNetTorch, h5_path: str) -> dict:
    """Transplant a Keras `.h5`'s weights into this module, without TensorFlow.

    A Keras HDF5 file stores each layer's kernel as `(kernel_size, in_channels, out_channels)` for
    Conv1D and `(in, out)` for Dense; PyTorch wants `(out, in, kernel_size)` and `(out, in)`. Read
    with h5py so this works in the project venv; returns what was moved, for the record.
    """
    import h5py

    moved = {}
    with h5py.File(h5_path, "r") as f:
        weights = f["model_weights"] if "model_weights" in f else f
        tensors = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                tensors.append((name, np.array(obj)))

        weights.visititems(visit)
        kernels = [(n, a) for n, a in tensors if n.endswith(("kernel:0", "kernel"))]
        biases = [(n, a) for n, a in tensors if n.endswith(("bias:0", "bias"))]
    if len(kernels) != len(biases):
        raise ValueError(f"{h5_path}: {len(kernels)} kernels but {len(biases)} biases")
    target = [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    if len(target) != len(kernels):
        raise ValueError(f"{h5_path} holds {len(kernels)} weighted layers; this module has "
                         f"{len(target)}. The architectures do not match.")
    order = sorted(range(len(kernels)), key=lambda i: kernels[i][0])
    with torch.no_grad():
        for slot, i in enumerate(order):
            m = target[slot]
            k = kernels[i][1]
            b = biases[i][1]
            if isinstance(m, nn.Conv1d):
                w = np.transpose(k, (2, 1, 0))          # (kw, cin, cout) -> (cout, cin, kw)
            else:
                w = np.transpose(k, (1, 0))             # (in, out) -> (out, in)
            if tuple(w.shape) != tuple(m.weight.shape):
                raise ValueError(f"layer {slot} ({type(m).__name__}): h5 gives {w.shape}, module "
                                 f"wants {tuple(m.weight.shape)}")
            m.weight.copy_(torch.as_tensor(w, dtype=m.weight.dtype))
            m.bias.copy_(torch.as_tensor(b, dtype=m.bias.dtype))
            moved[kernels[i][0]] = list(w.shape)
    return moved


def verify_against(model: TinyLidarNetTorch, reference, scans_m: np.ndarray) -> dict:
    """Max |delta| between this module and a callable that maps `(B, N, 1)` metres to `(B, 2)`."""
    model = model.eval()
    x = np.asarray(scans_m, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, :, None]
    with torch.no_grad():
        mine = model(torch.as_tensor(x)).numpy().astype(np.float64)
    theirs = np.asarray(reference(x), dtype=np.float64)
    d = np.abs(mine - theirs)
    return {"max_abs_delta": float(d.max()), "max_abs_delta_out0": float(d[:, 0].max()),
            "max_abs_delta_out1": float(d[:, 1].max()), "n": int(x.shape[0])}


class TorchBackendForDriver:
    """Adapter so `tinylidarnet.TinyLidarNet` can drive a torch module as it drives an ONNX one."""

    def __init__(self, model: TinyLidarNetTorch, path: str = "", device="cpu", threads: int = 1):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.path = path
        self.threads = int(threads)
        self.input_shape = [None, model.n_beams, 1]
        self.sha256 = None
        self.provenance = None
        if path:
            from .backends import sha256_file
            self.sha256 = sha256_file(path)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.model(torch.as_tensor(x, device=self.device)).cpu().numpy()

    def describe(self) -> dict:
        return {"backend": "torch", "version": torch.__version__, "device": str(self.device),
                "threads": self.threads, "path": self.path, "sha256": self.sha256,
                "n_params": self.model.n_params, "input_shape": self.input_shape,
                "conversion": None,
                "why": "a retrained TinyLidarNet: the architecture of train.py:170-181 in PyTorch, "
                       "verified against the published Keras weights before it was ever trained"}
