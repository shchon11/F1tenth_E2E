"""The published baselines, as one implementation with two call sites.

`f1sim_ros.baseline_node` drives one car in real time off `/scan`; `learn.benchmark.model_adapter`
drives a batch of them inside the simulator. Both reach the model through the SAME `BaselineDriver`
object here, and both go through `command()`. That is the whole reason this module exists: a
preprocessing pipeline written twice is a preprocessing pipeline that differs, and the node/adapter
parity test would then be measuring two transcriptions of a paper rather than one program. The
parity test still runs -- it is what proves the message plumbing around this agrees too -- but the
thing it compares is one function called from two places.

What a driver owns, in the units the upstream repo uses:

  * the raw scan it wants (`n_beams`, `fov`, `range_max`) -- before its own downsampling;
  * its preprocessing, transcribed from the upstream file and cited line by line;
  * its network, run through whatever backend is available (see `backends.py`);
  * its output mapping to (steering [rad], speed [m/s]).

What a driver does NOT own: anything of ours. No plan tracker, no controller arm, no clearance
layer, no traction guard, no observation stack. These models emit a steering angle and a speed and
that is the entire interface -- which is exactly the comparison the paper wants to make.

Batch and state. `command()` takes `(B, n_beams)` ranges in metres and `(B,)` speeds in m/s and
returns `(B, 2)` = (steer [rad], speed [m/s]). A driver with memory (End2Race's GRU, TinyLidarNet's
`pre=4/5/6` scan buffers) keeps it per row and clears it in `reset(done)`, where `done` is None
(clear everything) or a boolean mask of rows whose episode just ended. The node calls `reset()` with
no argument on `/f1sim/reset` and when the scan stream comes back; the benchmark runner calls it at
the cell's seeded reset and per row at every episode boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


class BaselineError(RuntimeError):
    """A refusal this module is certain about. Never raised to mean 'probably fine'."""


#: Our LiDAR, from `f1sim.params.LidarParams`: Hokuyo UST-10LX, 1081 beams over 270 deg at 40 Hz,
#: 10 m usable. Restated here as the number the baselines are *checked against* rather than imported,
#: because a driver declares what its upstream repo assumed and the check is between two independent
#: statements. Both published baselines happen to assume a 1081-beam 270 deg scanner too
#: (TinyLidarNet: `Benchmark/params/simulator_params.yaml:6-8`, fov 4.7, num_beams 1081;
#: End2Race: see `end2race.py`), which is why this comparison is possible at all.
OUR_N_BEAMS = 1081
OUR_FOV = 4.71238898


@dataclass(frozen=True)
class ScanContract:
    """The raw scan a driver wants, before any of its own downsampling.

    `range_max` is the saturation value the upstream code assumed -- what a miss reads as. It is NOT
    a normalisation constant unless the driver says so; TinyLidarNet clips at 10 m and feeds metres,
    End2Race maps metres through a sigmoid.
    """
    n_beams: int
    fov: float
    range_max: float
    control_rate: float


class BaselineDriver:
    """One published baseline, batched. Subclasses implement `_forward`.

    Subclasses set `KIND`, `scan` (a `ScanContract`), `needs_speed`, and are constructed by their
    module's `load()`.
    """

    KIND = "?"
    #: Whether `command()` reads the `speed` argument at all. TinyLidarNet does not (it is a
    #: LiDAR-only network); End2Race does. The node uses this to decide whether a stale `/odom` is a
    #: reason to stop the car -- demanding a sensor the network never reads would stop it for
    #: nothing.
    needs_speed = False

    def __init__(self, scan: ScanContract, backend, *, note: str = "", scan_fill: Optional[float] = None):
        self.scan = scan
        self.backend = backend
        self.note = note
        #: What a bearing this scanner cannot see is filled with. None = the model's own no-return
        #: value (`scan.range_max`). Held on the driver rather than passed at each call site so the
        #: node and the batched adapter cannot drift apart on it.
        self.scan_fill = scan_fill
        #: The scanner the caller has declared it will hand over (`bind_scanner`).
        self.source: Optional[dict] = None
        self._batch: Optional[int] = None

    # -- the scanner the caller actually has -----------------------------------------------------
    def bind_scanner(self, *, n_beams: int, fov: float, range_max: float) -> dict:
        """Declare the real scanner once, and work out what has to happen to its returns.

        Both call sites do this before their first `command()`: `baseline_node` from the first
        `LaserScan` header, the benchmark adapter from the cell's `ObsSpec` and the simulator's own
        LiDAR config. `adapt()` then applies whatever this decided. Two call sites, one decision.
        """
        fill = self.scan.range_max if self.scan_fill is None else float(self.scan_fill)
        self.source = {
            "n_beams": int(n_beams), "fov": float(fov), "range_max": float(range_max),
            "identity": scan_is_identity(src_n=n_beams, src_fov=fov, src_range_max=range_max,
                                         dst_n=self.scan.n_beams, dst_fov=self.scan.fov,
                                         dst_range_max=self.scan.range_max),
            "unseen_fraction": unseen_fraction(src_fov=fov, dst_n=self.scan.n_beams,
                                               dst_fov=self.scan.fov),
            "fill_m": fill,
        }
        return self.source

    def adapt(self, ranges_m) -> np.ndarray:
        """The bound scanner's returns, in the shape and units the model was trained on.

        The first thing done is a clamp at the SOURCE scanner's own `range_max`, and it is
        load-bearing rather than tidy. A Hokuyo UST-10LX is usable to 10 m but still emits returns
        out to ~30 m, and the bags contain them; the simulator, meanwhile, saturates its scan at
        `cfg.lidar.range_max` before the observation is built (`gym_env._norm_scan`). Without the
        clamp the node would hand End2Race a 14 m return where the batched adapter hands it 10, and
        the two would not be comparing the same measurement -- which is exactly what the parity test
        caught: a 4.4 m disagreement on single beams, amplified by the GRU into 8 mm/s of speed.
        Beyond a scanner's declared range there is no measurement to preserve.
        """
        if self.source is None:
            raise BaselineError(f"{self.KIND}: bind_scanner() has not been called; the driver does "
                                f"not know what scanner it is reading")
        r = np.minimum(np.asarray(ranges_m, dtype=np.float32),
                       np.float32(self.source["range_max"]))
        if self.source["identity"]:
            return np.minimum(r, np.float32(self.scan.range_max))
        return map_scan(r, src_fov=self.source["fov"], src_range_max=self.source["range_max"],
                        dst_n=self.scan.n_beams, dst_fov=self.scan.fov,
                        dst_range_max=self.scan.range_max, fill=self.source["fill_m"])

    # -- the interface both call sites use -------------------------------------------------------
    def command(self, ranges_m, speed_mps=None) -> np.ndarray:
        """`(B, n_beams)` metres (+ `(B,)` m/s) -> `(B, 2)` = (steering [rad], speed [m/s]).

        The ranges are what the scanner reported, in metres, already saturated at `range_max` by the
        caller (the node saturates the 0xFFFF sentinel; the simulator's `_norm_scan` clamps). Every
        further transformation belongs to the model and lives in `_prepare`.
        """
        r = np.asarray(ranges_m, dtype=np.float32)
        if r.ndim == 1:
            r = r[None, :]
        if r.ndim != 2:
            raise BaselineError(f"ranges must be (B, n_beams) or (n_beams,), got {r.shape}")
        if r.shape[1] != self.scan.n_beams:
            raise BaselineError(
                f"{self.KIND} wants {self.scan.n_beams} beams and was given {r.shape[1]}. Resample "
                f"before calling: which beams a 1-D CNN reads is part of the trained model, so "
                f"silently accepting another count would score a different network.")
        if not np.isfinite(r).all():
            raise BaselineError(f"{self.KIND}: non-finite range in the scan handed to the model; "
                                f"saturate misses at range_max before calling")
        v = None
        if self.needs_speed:
            if speed_mps is None:
                raise BaselineError(f"{self.KIND} reads the measured speed and none was given")
            v = np.asarray(speed_mps, dtype=np.float32).reshape(-1)
            if v.shape[0] != r.shape[0]:
                raise BaselineError(f"{self.KIND}: {v.shape[0]} speeds for {r.shape[0]} scans")
            if not np.isfinite(v).all():
                raise BaselineError(f"{self.KIND}: non-finite speed")
        self._ensure_batch(r.shape[0])
        out = self._forward(r, v)
        out = np.asarray(out, dtype=np.float32).reshape(r.shape[0], 2)
        if not np.isfinite(out).all():
            raise BaselineError(f"{self.KIND}: the network produced a non-finite command")
        return out

    def reset(self, done=None) -> None:
        """Clear per-row memory. `None` clears every row; a boolean mask clears those rows."""

    # -- subclass hooks ---------------------------------------------------------------------------
    def _forward(self, ranges_m: np.ndarray, speed_mps: Optional[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def _ensure_batch(self, batch: int) -> None:
        """A width change rebuilds (and so clears) any per-row state.

        Same rule as `learn.memory.policy_fn`: a hidden state is per row and cannot be carried from
        one cell's rows to another's.
        """
        if self._batch != batch:
            self._batch = batch
            self._rebuild(batch)

    def _rebuild(self, batch: int) -> None:
        """Allocate per-row state for a batch of this width. Stateless drivers need nothing."""

    # -- provenance ------------------------------------------------------------------------------
    def describe(self) -> dict:
        """What actually ran, for the protocol block of a benchmark row and the node's log line."""
        return {"kind": self.KIND, "backend": self.backend.describe(),
                "n_beams": self.scan.n_beams, "fov": self.scan.fov,
                "range_max": self.scan.range_max, "control_rate": self.scan.control_rate,
                "needs_speed": bool(self.needs_speed), "note": self.note,
                "scan_source": self.source}


def map_scan(r: np.ndarray, *, src_fov: float, src_range_max: float,
             dst_n: int, dst_fov: float, dst_range_max: float,
             fill: Optional[float] = None) -> np.ndarray:
    """One scanner's ranges as the scanner a driver was trained on would have reported them.

    Interpolation is by **bearing**, not by index, and that distinction is the whole point. Both
    windows are `linspace(-fov/2, +fov/2, n)` -- ours (`f1sim/lidar.py:141`), f1tenth_gym's
    (`laser_models.py:165-175`) and a `LaserScan`'s (`angle_min`..`angle_max`) all use that
    convention -- so when the two fields of view agree this is exactly the index interpolation
    `ObsBuilder.build` does, and when they do not it is the only mapping that puts each return at
    the bearing it was measured at. Interpolating by index across a wider or narrower window would
    rotate and stretch the world silently, which is the failure mode that looks like a working
    policy driving badly.

    Bearings the source scanner cannot see (End2Race wants 360 deg; this car's Hokuyo spans 270)
    are filled with `fill`, defaulting to `dst_range_max` -- the value the model's own simulator
    returns for a beam that hits nothing (`f1tenth_gym/.../laser_models.py:143-144`), i.e. "nothing
    out there as far as I can see". That is a declared substitution, not a measurement:
    `baseline_node` logs it once and the adapter records it in the protocol, and the opposite
    convention (`fill=0.0`, the value `eval_singleagent.py:114` writes into beams it masks out) is
    run as a second variant so the choice is measured rather than argued.

    Ranges beyond the destination's own saturation are clipped to it, because a model trained
    against a 30 m scanner reads 30 m as "far", not as a number to extrapolate past. The clip is
    applied to the interpolated returns only -- a `fill` below 0 or above `dst_range_max` would be
    a different declared substitution and is left as given.
    """
    r = np.asarray(r, dtype=np.float32)
    single = r.ndim == 1
    if single:
        r = r[None, :]
    src_n = r.shape[1]
    src = np.linspace(-0.5 * float(src_fov), 0.5 * float(src_fov), src_n)
    dst = np.linspace(-0.5 * float(dst_fov), 0.5 * float(dst_fov), int(dst_n))
    inside = (dst >= src[0]) & (dst <= src[-1])
    fill_v = float(dst_range_max) if fill is None else float(fill)
    out = np.full((r.shape[0], int(dst_n)), np.float32(fill_v), dtype=np.float32)
    if inside.any():
        xi = dst[inside]
        seen = np.empty((r.shape[0], int(inside.sum())), dtype=np.float32)
        for i in range(r.shape[0]):
            seen[i] = np.interp(xi, src, r[i])
        np.clip(seen, 0.0, float(dst_range_max), out=seen)
        out[:, inside] = seen
    return out[0] if single else out


def scan_is_identity(*, src_n: int, src_fov: float, src_range_max: float,
                     dst_n: int, dst_fov: float, dst_range_max: float) -> bool:
    """Would `map_scan` be a no-op (up to the clip)? True on this car for TinyLidarNet."""
    return (int(src_n) == int(dst_n)
            and math.isclose(float(src_fov), float(dst_fov), rel_tol=1e-9, abs_tol=1e-9)
            and float(src_range_max) <= float(dst_range_max))


def unseen_fraction(*, src_fov: float, dst_n: int, dst_fov: float) -> float:
    """Fraction of the driver's beams that the source scanner cannot see at all."""
    src_half, dst = 0.5 * float(src_fov), np.linspace(-0.5 * float(dst_fov), 0.5 * float(dst_fov),
                                                      int(dst_n))
    return float(np.mean((dst < -src_half) | (dst > src_half)))
