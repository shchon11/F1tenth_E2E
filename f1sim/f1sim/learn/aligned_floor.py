"""What the `aligned` channel reports when NOTHING is moving: its noise floor, measured twice.

The channel's claim is that `L_t - warp(L_{t-k})` is zero for static geometry and nonzero for a car
that moved by itself. The first half of that is an approximation, and this is where its error is
measured rather than asserted:

* **in simulation**, on scenes with no other car and the attitude randomisation on -- 1.7 deg/g of
  roll, a 1 deg rms floor wobble on a 0.4 s time constant, +-4 deg of IMU misalignment
  (`docs/real_data_calibration.md` 6.1a). Every residual there is error;
* **on the real bags** (`/home/shchon11/F1tenth/real_data`, 22 recordings), replayed through the
  same channel from the same three sensors the car will run it on: `/odom` wheel speed,
  `/sensors/imu/raw` gyro z, and roll/pitch from that message's orientation quaternion -- the very
  path `f1sim_ros/policy_node.attitude_from_orientation` takes. The real floor is what the plant, the
  scanner and the driver's own attitude estimate actually leave behind, and it includes whatever the
  simulator does not model.

The bags are not free of moving objects -- a competition recording has people beside the track and,
in some, another car -- so the real numbers are an UPPER bound on the floor, not the floor. They are
reported as such: the point is that the gate's survivors are rare and structured there too, not that
every one of them is noise.

What is reported, over beams where the warp had a prediction at all:

    known        share of beams the warp could predict (the rest are `unknown` and read exactly 0)
    |raw| pXX    percentiles of the ungated residual [m] -- the warp's own error
    over eps     share of beams the small-value threshold does NOT reject
    gated        share of beams that ALSO survive the two-frame consistency test: the channel's
                 false-positive rate on a scene where the right answer is zero everywhere
    |gated| pXX  how large the survivors are

    python -m f1sim.learn.aligned_floor sim --ckpt ~/f1sim_runs/_baselines/frozen_original_48cc698f.pt
    python -m f1sim.learn.aligned_floor bags --out floor_bags.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional

import numpy as np
import torch

from ..gym_env import EnvConfig
from ..params import Config
from . import common
from .aligned import AlignedScan, aligned_spec
from .obs import ObsSpec, flatten_obs, motion_from_proprio, motion_index_spec
from .model import load_checkpoint
from .memory import policy_fn

BAG_ROOT = "/home/shchon11/F1tenth/real_data"
PCTS = (50.0, 68.27, 90.0, 99.0, 99.9)

#: Candidate thresholds [m] the floor reports a survivor rate for. tau is chosen off this curve and
#: then fixed before any training -- see `docs/research/motion-memory-2026-09-14.md`.
TAU_GRID = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00)


# ------------------------------------------------------------------ statistics
class Floor:
    """Accumulates the residual distribution without keeping every beam of every step.

    Percentiles come from a histogram rather than a sorted array: a 400-step rollout of 63 cars over
    1081 beams is 27 million values, and the interesting part of the distribution is 0-2 m with
    millimetre structure. The bin width is stated with the numbers so a reader knows the resolution
    of the quantiles they are being shown.
    """

    RUN_MAX = 64

    def __init__(self, bins: int = 4000, hi: float = 20.0, label: str = ""):
        self.bins, self.hi, self.label = int(bins), float(hi), label
        self.raw = np.zeros(self.bins + 1, dtype=np.int64)
        self.gated = np.zeros(self.bins + 1, dtype=np.int64)
        self.n_beams = 0            # scored beams (ready rows)
        self.n_known = 0
        self.n_over = 0             # over the small-value threshold
        self.n_gated = 0            # ... and through the consistency test
        self.speed = []
        self.yaw = []
        self.tilt = []
        #: Run lengths of consecutive flagged beams. The single most interpretable statistic here:
        #: a car three metres away covers 30-80 beams in one block, so a floor made of isolated
        #: beams and short runs is one the stem can tell apart from an object, and a floor made of
        #: long runs is not. Counted per row per step, capped so the histogram stays small.
        self.runs = np.zeros(self.RUN_MAX + 2, dtype=np.int64)

    #: Beams per row, for the "runs per scan" figure. Set by the caller once.
    n_per_row = 1081

    def _add(self, h, v):
        idx = np.clip((np.abs(v) / self.hi * self.bins).astype(np.int64), 0, self.bins)
        np.add.at(h, idx, 1) if idx.size < 4096 else h.__iadd__(
            np.bincount(idx, minlength=self.bins + 1))

    def add_runs(self, flagged: np.ndarray):
        """Count the runs of consecutive flagged beams in each (row of) a (B, N) bool array."""
        f = np.atleast_2d(flagged)
        pad = np.zeros((f.shape[0], 1), dtype=bool)
        d = np.diff(np.concatenate([pad, f, pad], 1).astype(np.int8), axis=1)
        for row in range(f.shape[0]):
            starts = np.flatnonzero(d[row] == 1)
            ends = np.flatnonzero(d[row] == -1)
            if starts.size:
                np.add.at(self.runs, np.clip(ends - starts, 0, self.RUN_MAX + 1), 1)

    def add(self, raw_m: np.ndarray, gated_m: np.ndarray, known: np.ndarray, eps: np.ndarray,
            motion: Optional[np.ndarray] = None):
        self.n_beams += int(known.size)
        self.n_known += int(known.sum())
        r = raw_m[known]
        self._add(self.raw, r)
        over = np.abs(r) > eps[known]
        self.n_over += int(over.sum())
        g = gated_m[known]
        nz = g != 0
        self.n_gated += int(nz.sum())
        if nz.any():
            self._add(self.gated, g[nz])
        if motion is not None:
            self.speed.append(motion[:, 0]); self.yaw.append(motion[:, 1])
            self.tilt.append(np.hypot(motion[:, 2], motion[:, 3]))

    def _over(self, tau: float) -> float:
        """Share of scored beams whose |raw residual| exceeds `tau`, from the histogram."""
        n = self.raw.sum()
        if n == 0:
            return float("nan")
        i = min(self.bins, int(np.ceil(tau / self.hi * self.bins)))
        return float(self.raw[i:].sum() / n)

    def _pct(self, h, pcts=PCTS):
        n = h.sum()
        if n == 0:
            return {f"p{p:g}": float("nan") for p in pcts}
        c = np.cumsum(h) / n
        edges = np.arange(self.bins + 1) * (self.hi / self.bins)
        return {f"p{p:g}": float(edges[int(np.searchsorted(c, p / 100.0))]) for p in pcts}

    def summary(self) -> dict:
        # sigma_static, two ways. The plain standard deviation is dominated by the tail -- edges,
        # disocclusions and, on a real recording, whatever actually moved -- and 2-3 of it would be
        # a threshold set by the outliers rather than by the noise. The robust estimate is the one
        # tau is taken from: for a zero-mean normal, median|x| = 0.6745 sigma, so
        # sigma = 1.4826 median|x|, and p68.27 of |x| is sigma directly. The two are reported side
        # by side, and the note states which number tau came from.
        r = self._pct(self.raw)
        out = {"label": self.label, "beams": int(self.n_beams),
               "sigma_robust_m": 1.4826 * r["p50"], "sigma_p68_m": r["p68.27"],
               # What each candidate tau would leave behind on a scene where the right answer is
               # zero everywhere. This is the curve tau is read off: "2-3 sigma" is a statement
               # about a normal distribution, the residual's is not normal, and the survivor rate
               # is the quantity that actually matters to a channel feeding a policy.
               "tau_curve": {f"{t:g}": self._over(t) for t in TAU_GRID},
               "known_frac": self.n_known / max(1, self.n_beams),
               "over_eps_frac": self.n_over / max(1, self.n_known),
               "gated_frac": self.n_gated / max(1, self.n_known),
               "bin_m": self.hi / self.bins,
               "raw_abs": self._pct(self.raw), "gated_abs": self._pct(self.gated)}
        n_runs = int(self.runs.sum())
        beams_in = float((self.runs * np.arange(self.runs.size)).sum())
        long_ = lambda L: float((self.runs[L:] * np.arange(L, self.runs.size)).sum()
                                / max(1.0, beams_in))
        c = np.cumsum(self.runs) / max(1, n_runs)
        out["runs"] = {"count": n_runs, "per_scan": n_runs / max(1, self.n_beams / max(1, self.n_per_row)),
                       "p50": float(np.searchsorted(c, 0.5)), "p90": float(np.searchsorted(c, 0.9)),
                       "max": int(np.flatnonzero(self.runs)[-1]) if n_runs else 0,
                       "share_in_runs_ge_5": long_(5), "share_in_runs_ge_10": long_(10),
                       "share_in_runs_ge_20": long_(20)}
        if self.speed:
            s, y, t = (np.concatenate(v) for v in (self.speed, self.yaw, self.tilt))
            out["motion"] = {"speed_mean": float(s.mean()), "speed_p90": float(np.percentile(s, 90)),
                             "yaw_abs_p90": float(np.percentile(np.abs(y), 90)),
                             "tilt_deg_rms": float(np.degrees(np.sqrt((t ** 2).mean()))),
                             "tilt_deg_p99": float(np.degrees(np.percentile(t, 99)))}
        return out


def line(s: dict) -> str:
    r, g = s["raw_abs"], s["gated_abs"]
    return (f"{s['label']:<28s} known {s['known_frac']:6.1%}  sigma "
            f"{s['sigma_robust_m']:.3f}/{s['sigma_p68_m']:.3f} m  |raw| p50 {r['p50']:.3f} "
            f"p90 {r['p90']:.3f} p99 {r['p99']:.3f} p99.9 {r['p99.9']:.3f} m  |  "
            f"over eps {s['over_eps_frac']:7.3%}  gated {s['gated_frac']:8.4%}  "
            f"|gated| p50 {g['p50']:.3f} p99 {g['p99']:.3f} m  |  runs p50 "
            f"{s['runs']['p50']:.0f} p90 {s['runs']['p90']:.0f} max {s['runs']['max']}, "
            f"{s['runs']['share_in_runs_ge_20']:.0%} of flagged beams in runs >= 20")


def tau_line(s: dict) -> str:
    return ("  survivors by tau [m]: "
            + "  ".join(f"{t}:{v:.2%}" for t, v in s["tau_curve"].items()))


# ------------------------------------------------------------------ simulation
@torch.no_grad()
def measure_sim(a) -> dict:
    """Drive a checkpoint solo on `--tracks` with no opponent and score every residual as error."""
    device = torch.device(a.device)
    names = common.track_names(a.tracks, draws=a.obstacle_draws, seed=a.seed)
    tracks, rls = common.load_tracks(names, racelines=(a.race_size > 1 and a.opponent == "teacher"))
    cfg = Config(); cfg.sim.seed = a.seed; cfg.sim.compile = (a.sim_backend == "compile")
    model, extra = load_checkpoint(a.ckpt, device, allow_controller=True)
    model.eval()
    sp = dict(extra.get("spec") or {})
    env = common.make_env(tracks, a.envs, device,
                          EnvConfig(speed_cap=a.speed_cap, resample_track_on_reset=True,
                                    action_mode=("plan" if int(model.meta.get("act_dim", 2)) >= 5
                                                 else "direct"),
                                    max_steps=int(a.episode_s * 40),
                                    scan_stack=int(sp.get("scan_stack", 6)),
                                    scan_stride=int(sp.get("scan_stride", 1)),
                                    hist_len=int(sp.get("hist_len", 20)),
                                    hist_stride=int(sp.get("hist_stride", 2)),
                                    procedural_obstacles=a.procedural_obstacles,
                                    race_size=a.race_size, opponent=a.opponent,
                                    opp_speed_range=(0.6, 1.15), spawn_order="random"),
                          cfg=cfg, seed=a.seed, rls=rls)
    env.sim.warmup()
    spec = common.obs_spec(env)
    idx = motion_index_spec(spec)
    s = aligned_spec(k=a.aligned_k, tau=a.tau, tau_rel=a.tau_rel,
                     consist_beams=a.consist_beams, z_tol=a.z_tol, gap_fill=a.gap_fill,
                     tol_beams=a.tol_beams)
    arms = {"aligned (tilt on)": AlignedScan(env.n_beams, env.B, s, device=device,
                                             range_max=spec.range_max)}
    if a.no_tilt_arm:
        # The same channel told the car is level. Its floor is what the roll/pitch input buys, which
        # is the one number that says whether reading the IMU into the warp was worth doing.
        arms["aligned (tilt ignored)"] = AlignedScan(env.n_beams, env.B, s, device=device,
                                                     range_max=spec.range_max)
    floors = {k: Floor(label=k) for k in arms}
    policy = policy_fn(model, env.B, device=device, deterministic=True)
    obs, _ = env.reset(seed=a.seed)
    policy.reset()
    for arm in arms.values():
        arm.reset()
    for step in range(a.steps):
        scan, pro = flatten_obs(obs)
        motion = motion_from_proprio(pro, idx)
        for name, arm in arms.items():
            m = motion if "tilt on" in name else torch.cat([motion[:, :2],
                                                            torch.zeros_like(motion[:, 2:])], 1)
            gated, raw, known, ready = arm.parts(scan[:, 0], m)
            if not bool(ready.any()):
                continue
            r = ready
            eps = (s["tau"] + s["tau_rel"] * scan[r, 0] * spec.range_max).cpu().numpy()
            floors[name].n_per_row = env.n_beams
            floors[name].add((raw[r] * spec.range_max).cpu().numpy(),
                             (gated[r] * spec.range_max).cpu().numpy(),
                             known[r].cpu().numpy(), eps,
                             motion[r].cpu().numpy() if "tilt on" in name else None)
            floors[name].add_runs((gated[r] != 0).cpu().numpy())
        act = policy(obs)
        obs, _rew, term, trunc, _info = env.step(act.clamp(-1, 1))
        done = term | trunc
        policy.reset(done)
        for arm in arms.values():
            arm.reset(done)
    return {"mode": "sim", "tracks": names, "envs": a.envs, "steps": a.steps,
            "race_size": a.race_size, "opponent": a.opponent if a.race_size > 1 else None,
            "ckpt": a.ckpt, "spec": s, "arms": [f.summary() for f in floors.values()]}


# ------------------------------------------------------------------ the real bags
def _quat_to_rp(q: np.ndarray) -> np.ndarray:
    """(roll, pitch) from (N, 4) w x y z, the same formula `policy_node.quat_to_rp` uses."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return np.stack([roll, pitch], 1)


#: A yaw rate above this is not a 1/10-scale car turning; it is the raw gyro channel the dataset
#: README warns about, whose |gz| p99 reads 49-255 rad/s in some recordings against a normal 3-4.
#: The README's instruction is to validate the units PER BAG, so this validates them rather than
#: trusting a folder-name tag -- a bag with a good gyro and a `GYRO-BAD` name would be used, and one
#: with a bad gyro and no tag would not.
MAX_YAW_RATE = 10.0

#: Likewise for the attitude: a body roll or pitch past this is not a car on a flat floor, it is an
#: orientation estimate that is not measuring what its field says.
MAX_TILT_RAD = 0.35


def read_bag_streams(path: str, n_beams: int, range_max: float,
                     max_yaw: float = MAX_YAW_RATE, max_tilt: float = MAX_TILT_RAD):
    """((t, scan (T, N) normalised, motion (T, 4) SI), reason) for one bag; the data is None when
    the bag cannot be replayed and `reason` says why.

    Everything is sampled onto the scan's own timestamps: the scan is what the policy runs on, and
    interpolating the scan onto somebody else's clock would smear ranges across a discontinuity.
    The other three are interpolated onto it, which is what the node's freshness window does in
    effect anyway.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    need = ("/scan", "/odom", "/sensors/imu/raw")
    if any(t not in types for t in need):
        return None, "missing " + ", ".join(t for t in need if t not in types)
    cls, acc = {}, {t: ([], []) for t in need}
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if topic not in acc:
            continue
        if topic not in cls:
            cls[topic] = get_message(types[topic])
        m = deserialize_message(raw, cls[topic])
        if topic == "/scan":
            r = np.asarray(m.ranges, dtype=np.float32)
            # The 0xFFFF mm sentinel is finite; interpolating across it invents ranges (the ROS
            # node saturates first for the same reason), so saturate then resample.
            r = np.where(np.isfinite(r) & (r > 0) & (r < 65.0), r, np.float32(m.range_max))
            if r.size != n_beams:
                r = np.interp(np.linspace(0, r.size - 1, n_beams), np.arange(r.size), r)
            acc[topic][1].append(r)
        elif topic == "/odom":
            acc[topic][1].append([m.twist.twist.linear.x])
        else:
            q = m.orientation
            acc[topic][1].append([m.angular_velocity.z, q.w, q.x, q.y, q.z,
                                  float(np.asarray(m.orientation_covariance).reshape(-1)[0])])
        acc[topic][0].append(int(stamp))
    if any(not acc[t][0] for t in need):
        return None, "a needed topic is present but empty"
    out = {}
    for t in need:
        ns = np.asarray(acc[t][0], dtype=np.int64)
        order = np.argsort(ns, kind="stable")
        out[t] = (ns[order] * 1e-9, np.asarray(acc[t][1], dtype=np.float64)[order])
    ts, scan = out["/scan"]
    ts = ts - ts[0]
    to, vo = out["/odom"]; to = to - out["/scan"][0][0]
    ti, vi = out["/sensors/imu/raw"]; ti = ti - out["/scan"][0][0]
    if float(np.median(vi[:, 5])) == -1.0:
        return None, "orientation_covariance[0] = -1: the publisher states it has no orientation"
    speed = np.interp(ts, to, vo[:, 0])
    yaw = np.interp(ts, ti, vi[:, 0])
    q = np.stack([np.interp(ts, ti, vi[:, j]) for j in range(1, 5)], 1)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.where(n > 1e-6, n, 1.0)
    rp = _quat_to_rp(q)
    # The attitude the car reports is absolute, including whatever constant mounting tilt the
    # scanner has. The warp only ever uses the DIFFERENCE between two instants, so a constant offset
    # cancels -- but subtracting the recording's own median makes that explicit and keeps the tilt
    # statistics reported below about the car's motion rather than about its bolts.
    rp = rp - np.median(rp, 0, keepdims=True)
    yaw_p99 = float(np.percentile(np.abs(yaw), 99))
    if yaw_p99 > max_yaw:
        return None, f"gyro z |p99| = {yaw_p99:.1f} rad/s, past the {max_yaw:g} a 1/10 car can turn"
    tilt_p99 = float(np.percentile(np.hypot(rp[:, 0], rp[:, 1]), 99))
    if tilt_p99 > max_tilt:
        return None, (f"attitude |p99| = {math.degrees(tilt_p99):.0f} deg, past the "
                      f"{math.degrees(max_tilt):.0f} a car on a flat floor reaches")
    motion = np.concatenate([speed[:, None], yaw[:, None], rp], 1)
    return (ts, np.clip(scan / range_max, 0.0, 1.0), motion), ""


def measure_bags(a) -> dict:
    root = a.bag_root
    folders = [os.path.join(root, d) for d in sorted(os.listdir(root))
               if os.path.isdir(os.path.join(root, d))]
    bags = []
    for f in folders:
        bags += [os.path.join(f, d) for d in sorted(os.listdir(f))
                 if os.path.isdir(os.path.join(f, d))
                 and any(x.endswith(".db3") for x in os.listdir(os.path.join(f, d)))]
    if a.bags:
        bags = [b for b in bags if any(k in os.path.basename(b) for k in a.bags.split(","))]
    s = aligned_spec(k=a.aligned_k, tau=a.tau, tau_rel=a.tau_rel,
                     consist_beams=a.consist_beams, z_tol=a.z_tol, gap_fill=a.gap_fill,
                     tol_beams=a.tol_beams)
    total = Floor(label="all bags")
    per_bag, skipped = [], []
    for path in bags:
        name = os.path.basename(path)
        try:
            got, why = read_bag_streams(path, a.n_beams, a.range_max, a.max_yaw, a.max_tilt)
        except Exception as exc:                                   # noqa: BLE001 - reported, not hidden
            skipped.append((name, f"{type(exc).__name__}: {exc}")); continue
        if got is None:
            skipped.append((name, why)); continue
        ts, scan, motion = got
        arm = AlignedScan(a.n_beams, 1, s, range_max=a.range_max)
        floor = Floor(label=name[:28])
        moving = 0
        for i in range(1, len(ts)):
            dt = float(ts[i] - ts[i - 1])
            if not 0.005 < dt < 0.2:            # a gap in the recording is not a control step
                arm.reset(); continue
            if a.min_speed > 0 and abs(motion[i, 0]) < a.min_speed:
                arm.reset(); continue
            moving += 1
            m = torch.tensor(motion[i], dtype=torch.float32)[None]
            gated, raw, known, ready = arm.parts(
                torch.tensor(scan[i], dtype=torch.float32)[None], m, dt)
            if not bool(ready[0]):
                continue
            eps = (s["tau"] + s["tau_rel"] * scan[i] * a.range_max)
            for f in (floor, total):
                f.n_per_row = a.n_beams
                f.add((raw[0] * a.range_max).numpy(), (gated[0] * a.range_max).numpy(),
                      known[0].numpy(), eps, motion[i][None])
                f.add_runs((gated[0] != 0).numpy())
        if floor.n_beams:
            per_bag.append({**floor.summary(), "steps": moving})
    return {"mode": "bags", "root": root, "spec": s, "n_beams": a.n_beams,
            "used": [b["label"] for b in per_bag], "skipped": skipped,
            "arms": [total.summary()], "per_bag": per_bag}


# ------------------------------------------------------------------ cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("sim", "bags"))
    ap.add_argument("--out", default="", help="write the full result as JSON here")
    ap.add_argument("--aligned-k", type=int, default=4, dest="aligned_k")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="soft threshold. 0 measures the UNGATED floor, which is what sigma_static "
                         "is read off; the declared tau is then 2-3 of those sigma")
    ap.add_argument("--tau-rel", type=float, default=0.0)
    ap.add_argument("--consist-beams", type=int, default=8)
    ap.add_argument("--z-tol", type=float, default=0.05)
    ap.add_argument("--gap-fill", type=int, default=1)
    ap.add_argument("--tol-beams", type=int, default=4,
                    help="bearing uncertainty of the warp, in beams (0 = the literal one-bin "
                         "difference)")
    # sim
    ap.add_argument("--ckpt", default="/home/shchon11/f1sim_runs/_baselines/frozen_original_48cc698f.pt",
                    help="[sim] whoever drives. The floor depends on how the car moves, so it is "
                         "measured under a policy rather than under a fixed command")
    ap.add_argument("--tracks", default="real:blackbox2022_1,gen:control:1400,real:korea_2026_competition+rlobs211")
    ap.add_argument("--obstacle-draws", type=int, default=8)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--speed-cap", type=float, default=9.0)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--procedural-obstacles", type=float, default=1.0,
                    help="[sim] static obstacle layouts. They are part of the static world the warp "
                         "has to cancel, and a bare corridor would make the floor look better than "
                         "the scenes the policy trains in")
    ap.add_argument("--race-size", type=int, default=1,
                    help="[sim] 1 is the FLOOR -- no other car, so every residual is error. A "
                         "larger value is not a floor: it is the same channel with something real "
                         "to find, and the pair of numbers is the only 'signal over floor' this "
                         "measurement can honestly produce")
    ap.add_argument("--opponent", default="teacher", choices=("policy", "teacher"),
                    help="[sim] who drives the other cars when --race-size > 1")
    ap.add_argument("--no-tilt-arm", action="store_true", default=True,
                    help="[sim] also measure the same channel with the roll/pitch input zeroed")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sim-backend", default="compile", choices=("compile", "eager"))
    # bags
    ap.add_argument("--bag-root", default=BAG_ROOT)
    ap.add_argument("--bags", default="", help="[bags] comma-separated substrings to select bags")
    ap.add_argument("--n-beams", type=int, default=1081)
    ap.add_argument("--range-max", type=float, default=10.0)
    ap.add_argument("--max-yaw", type=float, default=MAX_YAW_RATE,
                    help="[bags] refuse a recording whose gyro z |p99| is past this [rad/s]")
    ap.add_argument("--max-tilt", type=float, default=MAX_TILT_RAD,
                    help="[bags] refuse a recording whose |roll,pitch| p99 is past this [rad]")
    ap.add_argument("--min-speed", type=float, default=0.5,
                    help="[bags] skip control steps below this speed. A parked car has no ego "
                         "motion to cancel, so its residual measures the scanner and not the warp")
    a = ap.parse_args()
    res = measure_sim(a) if a.mode == "sim" else measure_bags(a)
    print()
    for s in res["arms"]:
        print(line(s))
        print(tau_line(s))
    if res.get("per_bag"):
        print()
        for s in res["per_bag"]:
            print(line(s))
    if res.get("skipped"):
        print()
        for name, why in res["skipped"]:
            print(f"skipped {name}: {why}")
    for s in res["arms"]:
        if s.get("motion"):
            print()
            print(f"{s['label']}: speed mean {s['motion']['speed_mean']:.2f} m/s (p90 "
                  f"{s['motion']['speed_p90']:.2f}), |yaw rate| p90 "
                  f"{s['motion']['yaw_abs_p90']:.2f} rad/s, tilt rms "
                  f"{s['motion']['tilt_deg_rms']:.2f} deg (p99 {s['motion']['tilt_deg_p99']:.2f})")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
