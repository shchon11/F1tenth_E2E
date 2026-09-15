"""End2Race (arXiv 2509.16894, michigan-traffic-lab/End2Race @ c563dd07a8, MIT), transcribed.

Their network is PyTorch, so it is *imported* from the vendored checkout rather than copied:
`model.End2Race`. Only the code that sits around it is written here, and every piece of it is a
quotation.

  network     `model.py:5-100`   per-beam learnable `k` -> pressure token -> concat with a 60-wide
                                 speed embedding -> GRU(420 -> 1680) -> 1680-420-2. The published
                                 `pretrained/end2race.pth` is **11 301 482** parameters; the fitted
                                 `k` spans 0.241-1.335 (initialised at 0.529 for every beam).
  token       `model.py:81`      `(-1 / (1 + exp(-k * x)) + 1) * 2` -- 1.0 at 0 m, ~0.01 at 10 m
                                 with the initial k. **Metres in, no normalisation.**
  input scan  their fork's default `Simulator(num_beams=1440, fov=6.28)`
                                 (`f1tenth_gym/gym/f110_gym/envs/base_classes.py:459`) -- a
                                 **360 degree, 1440-beam** scan, `max_range=30.0`
                                 (`.../laser_models.py:362`).
              `eval_singleagent.py:104-107`  `indices = linspace(0, len-1, 360, dtype=int)`,
                                 `lidar = lidar[indices]` -- 360 of those beams, 1 deg apart.
              (`demonstration.py:215` recorded the TRAINING csvs with `downsample_lidar(...,
                                 original_points=1440, target_points=360)` = `[::4]`. The two index
                                 sets drift by up to 3 beams near the ends of the scan. Their
                                 inconsistency; the evaluation path is the one reproduced, because
                                 it is the one their published numbers come from.)
  speed in    `eval_singleagent.py:118,126`  the **previous** step's measured `linear_vels_x`,
                                 seeded at `initial_speed * 0.9` (`:79`).
  outputs     `:121-125`         `actions[:, -1, :]`; steer clipped to +-0.52 rad; **speed is not
                                 clipped at all**.
  rate        `:38,136`          `timestep=0.01` with one `env.step` per inference: **100 Hz**.
                                 Their training CSVs were sampled at 10 Hz (`demonstration.py:213`,
                                 `sample_interval = 0.1`). Neither is this project's 40 Hz, and a
                                 GRU's recurrence is a function of the step it is run at, so the
                                 rate is recorded with every row rather than treated as detail.

Departures, declared:

* `eval_singleagent.py:110-114` optionally zeroes a fraction of the beams (`--noise`). That is an
  ablation of theirs, defaulted off upstream; off here too.
* the speed input. Their training fed the *commanded* speed of the previous step
  (`train.py:87`, the `desired_speed` label) while their evaluation feeds the *measured* one
  (`eval_singleagent.py:126`). The evaluation behaviour is reproduced.
* the first step of an episode has no previous speed. Upstream seeds it from the raceline speed at
  the spawn point; there is no raceline in this interface, so the first step uses the speed measured
  with its own scan -- one 25 ms step's difference, once per episode, identical in the node and in
  the adapter because both reach it through this file.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

from .backends import TorchBackend
from .common import BaselineDriver, BaselineError, ScanContract

#: `f110_gym/envs/base_classes.py:459` and `laser_models.py:362`.
RAW_BEAMS = 1440
RAW_FOV = 6.28
RAW_RANGE_MAX = 30.0

#: `eval_singleagent.py:35` / `model.py:11`.
N_FEATURES = 360

#: `eval_singleagent.py:38` `timestep=0.01`, one step per inference.
CONTROL_RATE_HZ = 100.0

#: `eval_singleagent.py:125`.
STEER_CLIP_RAD = 0.52

#: Where the vendored checkout lives. Overridable so a test can point at a copy.
DEFAULT_REPO = os.path.join(os.path.expanduser("~"), "F1tenth", "F1tenth_E2E", "external",
                            "baselines", "End2Race")


#: The one line of their `model.py` that this project is ever allowed to change, and only for the
#: fair-comparison arm: the feature width. Their per-beam `k` is indexed by beam, so a 270 deg
#: scanner resampled to 270 evenly spaced beams (one per degree) maps each `k` to a bearing, which
#: is the point of the deviation. Everything else -- token, GRU, output head, initialisation -- is
#: byte-identical, and `import_upstream` proves it rather than claiming it.
NUM_FEATURES_LINE = "        num_features = 360"


def import_upstream(repo: str, num_features: int | None = None):
    """`model.End2Race` from the vendored checkout, without leaving the path on `sys.path`.

    Their `model.py` sits at the repo root under the very generic name `model`, and this project
    has a `f1sim.learn.model` of its own. Importing it by file location keeps the two apart; leaving
    the repo root on `sys.path` would let a later `import utils` or `import model` anywhere in the
    process pick up theirs.

    `num_features` rewrites exactly one line of their source (`NUM_FEATURES_LINE`) and refuses if
    that line does not occur exactly once or if anything else would change. That is a deliberate,
    checkable deviation rather than a fork: `module_source_diff()` returns the one-line diff, and it
    goes into the protocol of every row built this way.
    """
    import importlib.util

    path = os.path.join(repo, "model.py")
    if not os.path.exists(path):
        raise BaselineError(
            f"End2Race is not vendored at {repo} (no model.py). Clone it under external/baselines/ "
            f"-- their model code is imported, never transcribed.")
    with open(path) as fh:
        src = fh.read()
    deviation = None
    if num_features is not None and int(num_features) != N_FEATURES:
        n = int(num_features)
        if src.count(NUM_FEATURES_LINE) != 1:
            raise BaselineError(
                f"{path}: expected exactly one {NUM_FEATURES_LINE.strip()!r} line to rewrite, found "
                f"{src.count(NUM_FEATURES_LINE)}. The upstream file has changed; re-read it before "
                f"assuming which line sets the feature width.")
        replacement = NUM_FEATURES_LINE.replace("360", str(n))
        src = src.replace(NUM_FEATURES_LINE, replacement)
        deviation = {"line": NUM_FEATURES_LINE.strip(), "replaced_with": replacement.strip(),
                     "num_features": n}
    name = f"_end2race_upstream_model_{int(num_features or N_FEATURES)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        exec(compile(src, path, "exec"), mod.__dict__)
    except BaseException:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved
        raise
    if not hasattr(mod, "End2Race"):
        raise BaselineError(f"{path} has no End2Race class")
    mod._f1sim_deviation = deviation
    mod._f1sim_source_sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return mod


class End2Race(BaselineDriver):
    """GRU over pressure tokens. Stateful: one hidden state and one previous speed per row."""

    KIND = "end2race"
    needs_speed = True

    def __init__(self, backend, *, raw_beams: int = RAW_BEAMS, raw_fov: float = RAW_FOV,
                 range_max: float = RAW_RANGE_MAX, steer_clip: float = STEER_CLIP_RAD,
                 control_rate: float = CONTROL_RATE_HZ, repo: str = "",
                 n_features: int = N_FEATURES, scan_fill=None, tick_hz=None,
                 caller_rate_hz: float = 40.0, deviation=None):
        import torch

        self.torch = torch
        self.steer_clip = float(steer_clip)
        self.n_features = int(n_features)
        self.deviation = deviation
        n_raw = int(raw_beams)
        if n_raw < self.n_features:
            raise BaselineError(f"End2Race takes {self.n_features} of at least that many beams, "
                                f"got {n_raw}")
        # `eval_singleagent.py:106`, verbatim -- NOT `[::4]`, which is what their data collector
        # used. The two differ by up to 3 beam indices and this is the evaluation path.
        self._idx = np.linspace(0, n_raw - 1, self.n_features, dtype=int)
        # Their recurrence rate. Upstream `eval_singleagent.py` calls the GRU once per 10 ms step;
        # this project's control loop runs at 40 Hz, so one call per control step advances the GRU
        # at 40 Hz and its memory spans 2.5x more wall time per state than it was evaluated with.
        # `tick_hz` restores their rate WITHOUT changing the plant: the model is ticked
        # `tick_hz / caller_rate_hz` times per command on the same held scan and speed, and the last
        # tick's output is the command. 2.5 is not an integer, so a fractional accumulator alternates
        # 3, 2, 3, 2 ... and the long-run rate is exactly 100 Hz. None = one tick per call.
        self.tick_hz = None if tick_hz is None else float(tick_hz)
        self.caller_rate_hz = float(caller_rate_hz)
        self._ticks_per_call = (1.0 if self.tick_hz is None
                                else self.tick_hz / max(1e-9, self.caller_rate_hz))
        if self._ticks_per_call < 1.0:
            raise BaselineError(f"tick_hz {self.tick_hz} is below the caller's "
                                f"{self.caller_rate_hz} Hz: a model cannot be run less often than "
                                f"it is asked for a command")
        self._tick_debt = 0.0
        self._hidden = None
        self._prev_speed = None
        super().__init__(ScanContract(n_beams=n_raw, fov=float(raw_fov),
                                      range_max=float(range_max), control_rate=float(control_rate)),
                         backend, scan_fill=scan_fill,
                         note=f"{self.n_features} of {n_raw} beams over "
                              f"{np.degrees(float(raw_fov)):.0f} deg"
                              + (f", ticked at {self.tick_hz:.0f} Hz against a "
                                 f"{self.caller_rate_hz:.0f} Hz caller" if self.tick_hz else "")
                              + (f", model.py num_features -> {self.n_features}" if deviation else ""))

    # -- per-row state ---------------------------------------------------------------------------
    def _rebuild(self, batch: int) -> None:
        h = int(self.backend.module.gru.hidden_size)
        self._hidden = self.torch.zeros((1, batch, h), dtype=self.torch.float32,
                                        device=self.backend.device)
        self._prev_speed = np.full(batch, np.nan, dtype=np.float32)      # nan = "no previous step"
        self._tick_debt = 0.0

    def reset(self, done=None) -> None:
        if self._hidden is None:
            return
        if done is None:
            self._hidden.zero_()
            self._prev_speed[:] = np.nan
            self._tick_debt = 0.0
            return
        m = np.asarray(_to_numpy(done), dtype=bool).reshape(-1)
        if m.shape[0] != self._hidden.shape[1]:
            raise BaselineError(f"reset mask of {m.shape[0]} for a batch of {self._hidden.shape[1]}")
        if not m.any():
            return
        idx = self.torch.as_tensor(np.nonzero(m)[0], device=self.backend.device)
        self._hidden[:, idx, :] = 0.0
        self._prev_speed[m] = np.nan

    # -- one step --------------------------------------------------------------------------------
    def _forward(self, ranges_m, speed_mps):
        torch = self.torch
        lidar = ranges_m[:, self._idx]                                   # `eval_singleagent.py:107`
        # First step of an episode: no previous speed exists. See the module docstring.
        prev = np.where(np.isnan(self._prev_speed), speed_mps, self._prev_speed).astype(np.float32)
        # How many GRU steps this command is worth. One, unless `tick_hz` says the model is to run
        # at its own rate against a slower caller; then the debt accumulates so the long-run rate is
        # exact rather than rounded up every step.
        self._tick_debt += self._ticks_per_call
        n_tick = int(self._tick_debt)
        self._tick_debt -= n_tick
        with torch.no_grad():
            x = torch.as_tensor(lidar, device=self.backend.device).unsqueeze(1)       # (B, 1, 360)
            v = torch.as_tensor(prev, device=self.backend.device).reshape(-1, 1, 1)   # (B, 1, 1)
            if n_tick > 1:
                # The held scan and speed repeated: the plant has not moved between the ticks, and
                # feeding it a scan it has not taken would be inventing a measurement.
                x = x.expand(-1, n_tick, -1)
                v = v.expand(-1, n_tick, -1)
            actions, self._hidden = self.backend.module(x, v, self._hidden)
            a = actions[:, -1, :].cpu().numpy()                          # `:121`
        self._prev_speed = np.asarray(speed_mps, dtype=np.float32).copy()  # `:126`
        steer = np.clip(a[:, 0], -self.steer_clip, self.steer_clip)      # `:125`
        return np.stack([steer.astype(np.float32), a[:, 1].astype(np.float32)], 1)

    def describe(self) -> dict:
        d = super().describe()
        d.update({"model_beams": self.n_features,
                  "beam_indices": "linspace(0, n_raw-1, n_features, int) (eval_singleagent.py:106)",
                  "steer_clip_rad": self.steer_clip,
                  "speed_output_clipped": False,
                  "input_normalisation": "learnable per-beam sigmoid pressure token (metres in)",
                  "speed_input": "previous step's measured speed (eval_singleagent.py:126)",
                  "upstream_rate_hz": CONTROL_RATE_HZ,
                  "upstream_training_rate_hz": 10.0,
                  "tick_hz": self.tick_hz, "caller_rate_hz": self.caller_rate_hz,
                  "ticks_per_call": self._ticks_per_call,
                  "upstream_model_deviation": self.deviation,
                  "upstream_noise_applied": False})
        return d


def _to_numpy(x):
    cpu = getattr(x, "cpu", None)
    return x.cpu().numpy() if callable(cpu) else np.asarray(x)


def load(weights: str, *, repo: str = "", hidden_scale: int = 4, device: str = "cpu",
         raw_beams: int = RAW_BEAMS, raw_fov: float = RAW_FOV, range_max: float = RAW_RANGE_MAX,
         n_features: int = N_FEATURES, scan_fill=None, tick_hz=None,
         caller_rate_hz: float = 40.0, **_unused) -> End2Race:
    """`weights` is their `pretrained/end2race.pth` (or one retrained on our demonstrations).

    `hidden_scale` is their CLI default (`eval_singleagent.py:21`, `train.py:24`).

    `raw_beams` / `raw_fov` / `range_max` describe the scan the driver's own preprocessing runs on
    -- their fork's 1440-beam 360 deg 30 m scanner by default. What the *caller* has is declared
    separately with `bind_scanner()`, and on this car the two differ: a 270 deg Hokuyo covers 270 of
    those 360 bearings and the other 90 are filled with `scan_fill` (default `range_max`, which is
    what their own ray tracer returns for a beam that hits nothing).

    `n_features` is the fair-comparison deviation: 270 evenly spaced beams, one per degree, so each
    learned `k` still indexes a bearing. It rewrites one line of their `model.py` and records it.

    `tick_hz` restores their 100 Hz recurrence against this project's 40 Hz control loop.
    """
    repo = repo or DEFAULT_REPO
    up = import_upstream(repo, num_features=(None if int(n_features) == N_FEATURES
                                             else int(n_features)))
    module = up.End2Race(mask_prob=0.0, hidden_scale=int(hidden_scale))
    backend = TorchBackend(module, weights, device=device)
    d = End2Race(backend, raw_beams=raw_beams, raw_fov=raw_fov, range_max=range_max, repo=repo,
                 n_features=int(n_features), scan_fill=scan_fill, tick_hz=tick_hz,
                 caller_rate_hz=caller_rate_hz, deviation=getattr(up, "_f1sim_deviation", None))
    d.hidden_scale = int(hidden_scale)
    d.upstream_source_sha256 = getattr(up, "_f1sim_source_sha256", None)
    return d


def build_untrained(*, repo: str = "", hidden_scale: int = 4, n_features: int = N_FEATURES,
                    mask_prob: float = 0.1, device: str = "cpu"):
    """Their architecture at their initialisation, for the fair-comparison arm's training.

    `mask_prob` is their `train.py:25` default; it only does anything in `.train()` mode, which is
    why the evaluation driver builds with 0.0 and this does not.
    """
    up = import_upstream(repo or DEFAULT_REPO,
                         num_features=(None if int(n_features) == N_FEATURES else int(n_features)))
    module = up.End2Race(mask_prob=float(mask_prob), hidden_scale=int(hidden_scale))
    return module.to(device), getattr(up, "_f1sim_deviation", None)
