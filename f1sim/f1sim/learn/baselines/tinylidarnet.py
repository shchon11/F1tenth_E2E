"""TinyLidarNet (IROS 2024, CSL-KU/TinyLidarNet @ 516800231b), transcribed line by line.

The upstream repo carries **no licence file**; it is vendored read-only and used for evaluation and
citation only (CONTRACT.md).

Everything below is a quotation, with the file and line it comes from. Nothing here is re-derived
from the paper, because the paper does not fix the two things that decide the numbers -- which beams
are fed in and what the second output means.

  network      `train.py:170-181`  Conv1D(24,10,s4) - Conv1D(36,8,s4) - Conv1D(48,4,s2) -
                                   Conv1D(64,3) - Conv1D(64,3) - Flatten - 100 - 50 - 10 - 2, relu
                                   throughout and **tanh** on the output.
                                   `Models/f1_tenth_model.h5` is that model at 1081 beams:
                                   220 686 parameters, which is the "220 k" the contract quotes.
  input        `inference.py:33`   `l.ranges[::subsample_lidar]`, `subsample_lidar = 2` on the car
               `train.py:57,86`    `ranges[::down_sample_param]`, `down_sample_param = 2`
               `zarrar/tiny_lidarnet.py:59`  `scans[::self.skip_n]`; the paper's three sizes are
                                   skip 1 / 2 / 4 (`generate_benchmark_results.py:168,156,147`)
                                   over a 1081-beam 270 deg scan
                                   (`Benchmark/params/simulator_params.yaml:6-8`: fov 4.7,
                                   num_beams 1081).
               **no normalisation** anywhere: the network eats metres.
  clip         `zarrar/tiny_lidarnet.py:63`  `scans[scans>10] = 10`
  shape        `inference.py:34-35` `(1, N, 1)` float32
  steer        `inference.py:120`  `msg.drive.steering_angle = servo`, i.e. output[0] is **radians**
  speed        `zarrar/tiny_lidarnet.py:77-79`  `linear_map(out[1], 0, 1, 1, 8)` -- their SIMULATOR
               `inference.py:126`  `linear_map(out[1], 0, 1, -0.5, 7.0)` -- their CAR
               The two disagree, so the mapping is a declared parameter (`speed_map`) rather than a
               silent choice, and it is recorded in every result.
  rate         `inference.py:26`   `hz = 40` -- the same 40 Hz this project's control loop runs at.

Two deliberate departures, both declared:

* `zarrar/tiny_lidarnet.py:46-47` adds `N(0, 0.5)` metres of noise to every beam inside `plan()`.
  That is a sensor model, not preprocessing: this project's suites declare their own noise policy
  per cell (`suite.sensor_noise`, `model_adapter.NOISE_FIELDS`) and applying a second one inside the
  driver would make the cell's declared policy false. Not applied; stated here and in the row.
* The 1081-beam variant needs no resampling at all on this car -- 1081 beams over 270 deg is exactly
  this scanner (`params.LidarParams:173-174`) and exactly the scanner their benchmark assumed.
"""
from __future__ import annotations

import numpy as np

from .backends import OnnxBackend
from .common import BaselineDriver, BaselineError, ScanContract, map_scan

#: `zarrar/tiny_lidarnet.py:63`.
CLIP_M = 10.0

#: `inference.py:26`.
CONTROL_RATE_HZ = 40.0

#: The scan their code assumes before its own `[::skip_n]`: 1081 beams over 270 deg.
#:
#: The beam count is stated three times upstream and agrees each time
#: (`Benchmark/params/simulator_params.yaml:8`, `zarrar/test_model.py:27` `np.random.rand(1081)`,
#: and the 1081-wide input of `Models/f1_tenth_model.h5`). The WINDOW is the Hokuyo UST-10LX's
#: 270 deg (`Readme.md`, "Hokuyo Lidar UST10-LX"), which the weights were trained on from real-car
#: bags -- 4.71238898 rad, the same constant `params.LidarParams:174` carries. Their *simulator*
#: config rounds it to 4.7 (`simulator_params.yaml:6`), 0.26 % narrower; that rounding is theirs and
#: is not reproduced, because reproducing it would resample this car's real 270 deg scan onto a
#: window it does not have, for no reason but to copy a typo.
RAW_BEAMS = 1081
RAW_FOV = 4.71238898

#: The two output speed mappings that exist in the upstream repo, by the file each is in.
SPEED_MAPS = {
    "sim": (1.0, 8.0, "zarrar/tiny_lidarnet.py:77-79 (their f1tenth_benchmarks evaluation)"),
    "car": (-0.5, 7.0, "inference.py:126 (their real-car node)"),
}


def linear_map(x, x_min, x_max, y_min, y_max):
    """`train.py:25-27` / `inference.py:88-89`, verbatim."""
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min


class TinyLidarNet(BaselineDriver):
    """1-D CNN, LiDAR only. Stateless: `pre=0`, the configuration all three paper sizes use."""

    KIND = "tinylidarnet"
    needs_speed = False

    def __init__(self, backend, *, skip_n: int, speed_map: str = "sim",
                 raw_beams: int = RAW_BEAMS, raw_fov: float = RAW_FOV, clip_m: float = CLIP_M):
        if speed_map not in SPEED_MAPS:
            raise BaselineError(f"speed_map must be one of {sorted(SPEED_MAPS)}, got {speed_map!r}; "
                                f"the two mappings are both in the upstream repo and disagree, so "
                                f"one has to be named")
        self.skip_n = int(skip_n)
        if self.skip_n < 1:
            raise BaselineError(f"skip_n must be >= 1, got {skip_n}")
        self.speed_map = speed_map
        self.clip_m = float(clip_m)
        self._idx = np.arange(0, int(raw_beams), self.skip_n)          # `scans[::skip_n]`
        n_in = self._model_beams(backend)
        if n_in != self._idx.size:
            raise BaselineError(
                f"the model takes {n_in} beams but {raw_beams} beams stepped by {self.skip_n} gives "
                f"{self._idx.size}. Which beams a 1-D CNN reads is part of the trained network; pick "
                f"the skip_n its weights were trained with (1 -> 1081, 2 -> 541, 4 -> 271).")
        super().__init__(ScanContract(n_beams=int(raw_beams), fov=float(raw_fov),
                                      range_max=self.clip_m, control_rate=CONTROL_RATE_HZ),
                         backend,
                         note=f"skip_n={self.skip_n}, speed_map={speed_map} "
                              f"({SPEED_MAPS[speed_map][2]})")

    @staticmethod
    def _model_beams(backend) -> int:
        shape = list(getattr(backend, "input_shape", []) or [])
        if len(shape) != 3 or shape[2] != 1:
            raise BaselineError(f"TinyLidarNet expects an (batch, beams, 1) input, got {shape}")
        return int(shape[1])

    # `zarrar/tiny_lidarnet.py:59-66` + `inference.py:33-35`
    def _prepare(self, ranges_m: np.ndarray) -> np.ndarray:
        s = ranges_m[:, self._idx]                     # scans[::skip_n]
        s = np.minimum(s, np.float32(self.clip_m))     # scans[scans>10] = 10
        return s[:, :, None].astype(np.float32)        # expand_dims(-1), batch is already axis 0

    def _forward(self, ranges_m, speed_mps):
        out = np.asarray(self.backend(self._prepare(ranges_m)), dtype=np.float32)
        steer = out[:, 0]                              # `inference.py:120`: radians, as published
        lo, hi, _src = SPEED_MAPS[self.speed_map]
        speed = linear_map(out[:, 1], 0.0, 1.0, lo, hi)
        return np.stack([steer, speed], 1)

    def describe(self) -> dict:
        d = super().describe()
        lo, hi, src = SPEED_MAPS[self.speed_map]
        d.update({"skip_n": self.skip_n, "model_beams": int(self._idx.size),
                  "clip_m": self.clip_m, "speed_map": self.speed_map,
                  "speed_range_mps": [lo, hi], "speed_map_source": src,
                  "input_normalisation": "none (metres, as trained)",
                  "upstream_noise_applied": False})
        return d


def load(weights: str, *, skip_n: int | None = None, speed_map: str = "sim", threads: int = 1,
         **_unused) -> TinyLidarNet:
    """`weights` is the converted `.onnx` (see `work/baselines/scripts/tln_to_onnx.py`).

    `skip_n` defaults to the value implied by the model's own input width over a 1081-beam scan, so
    an entry that pins `tinylidarnet_L_1081.onnx` does not have to restate it -- but a model whose
    width is not one of 1081/541/271 has to say what it wants, because guessing at that is guessing
    at which beams the network was trained on.
    """
    backend = OnnxBackend(weights, threads=threads)
    if skip_n is None:
        n_in = TinyLidarNet._model_beams(backend)
        implied = {RAW_BEAMS: 1, 541: 2, 271: 4}.get(n_in)
        if implied is None:
            raise BaselineError(
                f"{weights} takes {n_in} beams, which is not 1081/541/271; pass skip_n explicitly")
        skip_n = implied
    return TinyLidarNet(backend, skip_n=skip_n, speed_map=speed_map)
