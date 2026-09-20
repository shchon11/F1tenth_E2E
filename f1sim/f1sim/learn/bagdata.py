"""rosbag2 -> dataset: the observations a node WOULD have built, and the commands it published.

    python3 -m f1sim.learn.bagdata <bag> --out run.npz --spec-from ~/f1sim_runs/.../ppo_latest.pt
    python3 -m f1sim.learn.bagdata <bag> --summary

One bag in, one `BagDataset` out: for every `/scan` in the recording, the observation
`learn/obs.ObsBuilder` builds from that scan and the `/sensors/imu/raw` and `/odom` messages around
it, plus the `/drive` that answered it and the pose if the bag has one. A simulator bag and a car
bag produce the same object, because `config/record.yaml` makes them the same bag.

What it is for
--------------
* **DAgger relabelling.** Replay the recorded poses in the simulator and label the observations
  with `env.teacher_label`; the observation the learner is corrected on is then the one it actually
  had, not one re-synthesised from a reconstructed state.
* **Probes.** `probe_hidden` and friends take (observation, target) pairs; this is where recorded
  observations come from.
* **Calibration.** `docs/real_data_calibration.md` works in message units; `.bag` (the underlying
  `BagData`) is carried on the dataset so a calibration tool can reach the raw series without a
  second read.
* **Baseline parity.** Worker 18's check is "feed the recorded scans to the node, compare the
  commands to the recorded `/drive`" -- which is `scan`/`proprio` in and `command` out, for the
  first hundred scans of any bag.

The observation is rebuilt, not recorded
----------------------------------------
Nothing in the bag holds a 6 x 1081 stacked scan; it holds the scans. So the observation is
reconstructed by driving `ObsBuilder` over the recording exactly as `policy_node` drives it --
`resample_ranges`, the mean of the IMU samples since the previous scan, the last `/odom` wheel
speed, the roll and pitch of the `Imu` quaternion. Two consequences worth knowing before trusting a
row:

* **The action history has to come from somewhere.** The observation carries the policy's previous
  actions, and they are only in the bag if `/f1sim/plan` was recorded (`action_source="plan"`) or
  the run was a direct-action checkpoint whose `/drive` inverts back to one
  (`action_source="drive"`). Otherwise those channels are zeros and `action_source="zeros"` says
  so -- the rest of the observation is still exact, and for a probe on the scan encoder that is
  enough, but it is not the observation the policy saw.
* **Episode boundaries are honoured.** A `/f1sim/reset` clears the builder, because a history
  stitched across a reset describes a run that never happened. `episode` numbers the segments.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..calib import bagread
from .obs import ObsBuilder, ObsSpec, resample_ranges

#: The topics a dataset is built from. `/f1sim/plan` and `/f1sim/reset` are optional -- a raw car
#: recording has neither -- and `/ego_racecar/odom` is simulator-only ground truth.
SCAN, IMU, ODOM, DRIVE = "/scan", "/sensors/imu/raw", "/odom", "/drive"
PLAN, RESET, GT = "/f1sim/plan", "/f1sim/reset", "/ego_racecar/odom"


@dataclass
class BagDataset:
    """One recording, as the arrays a learner or a probe consumes. `T` rows, one per `/scan`."""
    name: str
    t: np.ndarray                    # (T,) scan record times [s], from the bag's own clock
    scan: np.ndarray                 # (T, k, N) the observation's stacked scan
    proprio: np.ndarray              # (T, P)
    command: np.ndarray              # (T, 2) the /drive that answered each scan: steer [rad], speed [m/s]
    command_valid: np.ndarray        # (T,) bool: was there one inside `command_window`?
    speed: np.ndarray                # (T,) /odom twist.linear.x [m/s]
    att: np.ndarray                  # (T, 2) roll, pitch [rad]
    imu: np.ndarray                  # (T, 6) the per-scan IMU mean, SI
    episode: np.ndarray              # (T,) int, incremented at every /f1sim/reset
    spec: ObsSpec
    meta: dict
    pose: Optional[np.ndarray] = None       # (T, 3) x, y, yaw from /odom -- DRIFTING dead reckoning
    pose_gt: Optional[np.ndarray] = None    # (T, 3) ground truth; simulator bags only
    plan: Optional[np.ndarray] = None       # (T, 8) the recorded /f1sim/plan, when there was one
    bag: Optional[bagread.BagData] = None   # the raw series, for a calibration tool

    def __len__(self):
        return int(self.t.shape[0])

    def summary(self) -> str:
        m = self.meta
        pose = "odom" + ("+gt" if self.pose_gt is not None else "") if self.pose is not None else "none"
        return (f"{self.name}: {len(self)} frames over {self.t[-1] - self.t[0]:.1f} s "
                f"({m['scan_rate_hz']:.1f} Hz), {int(self.episode.max()) + 1} episode(s)\n"
                f"  scan {self.scan.shape}  proprio {self.proprio.shape}  "
                f"commands {int(self.command_valid.sum())}/{len(self)}  pose {pose}\n"
                f"  action history: {m['action_source']}   spec: {m['spec_source']}   "
                f"imu {m['imu_per_scan']:.2f}/scan")

    def save(self, path: str) -> str:
        """`.npz`. `bag` is dropped: it is the raw recording, which is already on disk."""
        d = {k: v for k, v in dataclasses.asdict(self).items()
             if isinstance(v, np.ndarray)}
        d["spec"] = np.asarray(json.dumps(dataclasses.asdict(self.spec)))
        d["meta"] = np.asarray(json.dumps(self.meta))
        d["name"] = np.asarray(self.name)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(path, **d)
        return path


def load(path: str) -> BagDataset:
    d = np.load(path, allow_pickle=False)
    fields = {k for k in BagDataset.__dataclass_fields__ if k not in ("spec", "meta", "name", "bag")}
    kw = {k: d[k] for k in fields if k in d.files}
    return BagDataset(name=str(d["name"]), spec=ObsSpec(**json.loads(str(d["spec"]))),
                      meta=json.loads(str(d["meta"])), **kw)


def _last_at_or_before(t_src: np.ndarray, v_src: np.ndarray, t_at: np.ndarray):
    """The most recent sample at or before each time, and whether there was one.

    Held, not interpolated: this is what the node does with `/odom` -- it keeps the last value it
    was given until a new one arrives -- and an interpolated wheel speed is a speed the car never
    reported.
    """
    idx = np.searchsorted(t_src, t_at, side="right") - 1
    ok = idx >= 0
    return v_src[np.clip(idx, 0, len(v_src) - 1)], ok


def _first_at_or_after(t_src: np.ndarray, v_src: np.ndarray, t_at: np.ndarray, window: float):
    """The first sample at or after each time, within `window`. The command that ANSWERS a scan."""
    idx = np.searchsorted(t_src, t_at, side="left")
    ok = (idx < len(t_src))
    safe = np.clip(idx, 0, max(0, len(t_src) - 1))
    if len(t_src):
        ok &= (t_src[safe] - t_at) <= window
    return v_src[safe], ok


def _mean_between(t_src: np.ndarray, v_src: np.ndarray, t_lo: float, t_hi: float,
                  fallback: np.ndarray):
    """Mean of the samples in (t_lo, t_hi] -- the node's per-scan IMU mean, over the same window.

    With no sample in the window the previous mean is held, which is the node's "a dropped sample
    is tolerated" rule; a run of them is visible as a low `imu_per_scan` in the metadata.
    """
    lo, hi = np.searchsorted(t_src, t_lo, side="right"), np.searchsorted(t_src, t_hi, side="right")
    if hi <= lo:
        return fallback, 0
    return v_src[lo:hi].mean(0), hi - lo


def _attitude(quat: np.ndarray) -> Optional[tuple]:
    """(roll, pitch) from (qw, qx, qy, qz, cov0), or None -- `deploy.attitude_from_orientation`
    without importing `f1sim_ros`, which is a separate package and not a dependency of training."""
    w, x, y, z, cov0 = (float(c) for c in quat)
    if cov0 == -1.0 or not all(math.isfinite(c) for c in (w, x, y, z)):
        return None
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if abs(n - 1.0) > 1e-3:
        return None
    sinr = 2 * (w * x + y * z); cosr = 1 - 2 * (x * x + y * y)
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    return math.atan2(sinr, cosr), math.asin(sinp)


def spec_from_checkpoint(path: str) -> ObsSpec:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    d = dict((ck.get("extra") or {}).get("spec") or {})
    return ObsSpec(**d) if d else ObsSpec()


def from_bag(path: str, spec: Optional[ObsSpec] = None, *, speed_cap: float = 4.0,
             max_frames: Optional[int] = None, command_window: float = 0.05,
             action_source: str = "auto", steer_max: float = 0.4189,
             keep_bag: bool = False) -> BagDataset:
    """Build the dataset. `spec` defaults to `ObsSpec()`; pass the checkpoint's own when there is one.

    `command_window` is how long after a scan the `/drive` answering it may arrive. The default is
    about two control periods: wider and a row is labelled with the command for the NEXT scan,
    which for a DAgger label is an off-by-one in the direction that teaches the wrong thing.
    """
    spec = spec or ObsSpec()
    b = bagread.read(path, want_scan=True, want_orientation=True)
    for topic in (SCAN, IMU, ODOM):
        if not b.has(topic):
            raise ValueError(f"{os.path.basename(path.rstrip('/'))} has no {topic}; a dataset needs "
                             f"the three topics config/record.yaml marks required "
                             f"({SCAN}, {IMU}, {ODOM})")
    ori_key = IMU + bagread.ORIENTATION_SUFFIX
    t_scan = b.t[SCAN]
    if max_frames:
        t_scan = t_scan[:max_frames]
    n = len(t_scan)

    plan = b.v[PLAN][:, :8] if b.has(PLAN) else None
    if action_source == "auto":
        action_source = "plan" if (plan is not None and spec.act_dim == 8) else (
            "drive" if (spec.act_dim == 2 and b.has(DRIVE)) else "zeros")

    builder = ObsBuilder(spec, "cpu")
    resets = b.t.get(RESET, np.zeros(0))
    scans, pros, imus, atts, speeds, eps = [], [], [], [], [], []
    imu_counts = 0
    last_mean = np.zeros(6, dtype=np.float32)
    last_att = (0.0, 0.0)
    ep = 0
    prev_t = -np.inf
    v_hold, v_ok = _last_at_or_before(b.t[ODOM], b.v[ODOM][:, 3], t_scan)
    for i in range(n):
        ts = float(t_scan[i])
        if np.any((resets > prev_t) & (resets <= ts)):
            # A reset between the previous scan and this one. The builder's history describes a
            # segment that is over; the node clears it here and so does this.
            builder.reset()
            ep += 1
        mean, k = _mean_between(b.t[IMU], b.v[IMU], prev_t, ts, last_mean)
        last_mean = mean; imu_counts += k
        if ori_key in b.v:
            rows = b.v[ori_key][np.searchsorted(b.t[ori_key], prev_t, side="right"):
                                np.searchsorted(b.t[ori_key], ts, side="right")]
            for row in reversed(rows):                  # the newest usable one, as the node does
                got = _attitude(row)
                if got is not None:
                    last_att = got
                    break
        # `resample_ranges` first, exactly as the node does: the bag holds the driver's own beam
        # count, and the observation is defined at the spec's.
        s, p = builder.build(resample_ranges(b.v[SCAN][i], spec.range_max, spec.n_beams),
                             float(v_hold[i]) if v_ok[i] else 0.0, mean, last_att, speed_cap)
        if action_source == "plan" and plan is not None and i < len(plan):
            builder.push_action(plan[i])
        elif action_source == "drive" and b.has(DRIVE):
            d, ok = _first_at_or_after(b.t[DRIVE], b.v[DRIVE][:, :2], np.array([ts]),
                                       command_window)
            if ok[0]:
                builder.push_action(np.array([d[0, 0] / steer_max,
                                              2.0 * d[0, 1] / spec.v_max - 1.0], dtype=np.float32))
            else:
                builder.push_action(np.zeros(spec.act_dim, dtype=np.float32))
        else:
            builder.push_action(np.zeros(spec.act_dim, dtype=np.float32))
        scans.append(s[0].numpy()); pros.append(p[0].numpy())
        imus.append(mean); atts.append(last_att); speeds.append(float(v_hold[i]) if v_ok[i] else 0.0)
        eps.append(ep)
        prev_t = ts

    cmd, cmd_ok = ((np.zeros((n, 2), np.float32), np.zeros(n, bool)) if not b.has(DRIVE) else
                   _first_at_or_after(b.t[DRIVE], b.v[DRIVE][:, :2].astype(np.float32), t_scan,
                                      command_window))
    pose = pose_gt = None
    if b.has(ODOM):
        p_hold, p_ok = _last_at_or_before(b.t[ODOM], b.v[ODOM][:, :3], t_scan)
        pose = np.where(p_ok[:, None], p_hold, np.nan).astype(np.float32)
    if b.has(GT):
        g_hold, g_ok = _last_at_or_before(b.t[GT], b.v[GT][:, :3], t_scan)
        pose_gt = np.where(g_ok[:, None], g_hold, np.nan).astype(np.float32)
    span = float(t_scan[-1] - t_scan[0]) if n > 1 else 0.0
    meta = {"bag": os.path.abspath(path), "topics": sorted(b.t),
            "scan_rate_hz": (n - 1) / span if span > 0 else 0.0,
            "imu_per_scan": imu_counts / max(1, n),
            "action_source": action_source, "speed_cap": float(speed_cap),
            "command_window_s": float(command_window),
            "spec_source": "checkpoint" if spec != ObsSpec() else "ObsSpec() defaults",
            "origin_record_ns": b.origin_record_ns}
    return BagDataset(name=os.path.basename(path.rstrip("/")),
                      t=np.asarray(t_scan, dtype=np.float64),
                      scan=np.asarray(scans, dtype=np.float32),
                      proprio=np.asarray(pros, dtype=np.float32),
                      command=np.asarray(cmd, dtype=np.float32),
                      command_valid=np.asarray(cmd_ok, dtype=bool),
                      speed=np.asarray(speeds, dtype=np.float32),
                      att=np.asarray(atts, dtype=np.float32),
                      imu=np.asarray(imus, dtype=np.float32),
                      episode=np.asarray(eps, dtype=np.int32),
                      spec=spec, meta=meta, pose=pose, pose_gt=pose_gt,
                      plan=(None if plan is None else np.asarray(plan[:n], dtype=np.float32)),
                      bag=(b if keep_bag else None))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag")
    ap.add_argument("--out", default="")
    ap.add_argument("--spec-from", default="", help="a checkpoint whose observation spec to use")
    ap.add_argument("--speed-cap", type=float, default=4.0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--command-window", type=float, default=0.05)
    ap.add_argument("--action-source", default="auto", choices=("auto", "plan", "drive", "zeros"))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    spec = spec_from_checkpoint(a.spec_from) if a.spec_from else None
    d = from_bag(a.bag, spec, speed_cap=a.speed_cap, max_frames=(a.frames or None),
                 command_window=a.command_window, action_source=a.action_source)
    print(d.summary())
    if a.out:
        print("wrote " + d.save(a.out) + f"  ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
