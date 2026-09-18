"""Resumable, TRAIN-map-only policy-domain grip collection and fixed-epoch fitting.

Each split owns its base tracks and simulator seeds before any windows exist. Shards
are atomic, fsynced per chunk, and content-verified on resume. An interrupted job is
replayed from its initial seed; already durable chunks are compared, never overwritten.
FINAL is opened only by the separate, single-use ``final`` command.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

from .adaptive_grip import (AdaptiveConsumption, AdaptiveGripNet, adaptive_feature_spec,
                            load_adaptive_grip_estimator, save_adaptive_grip_estimator)
from .adaptive_grip_train import adaptive_loss, evaluate, predictions, privileged_targets, _source_hashes
from .grip_estimator import FeatureSpec, SensorHistory, file_sha256

SPLITS = ("train", "cal", "dev", "final")
DATA_FORMAT = "policy_grip_dataset_v1"
RECIPE = "adaptive_grip_policy_refit_v1"



def collector_source_hashes():
    """Pin action/observation/graph sources without changing the historical V2 pilot contract."""
    root = Path(__file__).resolve().parents[1]
    hashes = _source_hashes()
    names = ("learn/policy_grip_data.py", "learn/common.py", "learn/obs.py",
             "learn/grip_runtime.py", "learn/grip_control.py", "learn/graph_runtime.py",
             "learn/model.py", "learn/memory.py", "mpc.py", "teacher.py", "gym_env.py",
             "tracks.py", "opponent_events.py", "opponent_event_contract.py",
             "opponent_slots.py", "viewer/graph_fastpath.py")
    hashes.update({name: file_sha256(root / name) for name in names})
    return hashes


def _atomic_file(path, writer, binary=False):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"symlink output refused: {path.name}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    _atomic_file(path, lambda f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False))


def dataset_root(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("symlink dataset root refused")
    return root.resolve()


def safe_dataset_path(root, name):
    root = dataset_root(root)
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"noncanonical dataset path: {name!r}")
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError(f"symlink or escaping dataset path: {name!r}")
    return path


def _safe_bytes(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        return f.read()


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("format") != DATA_FORMAT:
        raise ValueError("unsupported dataset manifest format")
    spec = manifest.get("specification")
    if not isinstance(spec, dict) or not isinstance(spec.get("jobs"), list) or not spec["jobs"]:
        raise ValueError("manifest specification must declare jobs")
    jobs = spec["jobs"]
    if any(not isinstance(j, dict) or type(j.get("id")) is not int or j["id"] < 0
           or j.get("split") not in SPLITS for j in jobs):
        raise ValueError("invalid manifest job identity/split")
    ids = {j["id"] for j in jobs}
    if len(ids) != len(jobs):
        raise ValueError("duplicate manifest job identity")
    completed, shards = manifest.get("completed_jobs"), manifest.get("shards")
    if (not isinstance(completed, list) or any(type(i) is not int or i not in ids for i in completed)
            or len(set(completed)) != len(completed) or not isinstance(shards, dict)):
        raise ValueError("invalid completed jobs or shard manifest")
    for name, record in shards.items():
        match = re.fullmatch(r"job([0-9]{5,})_chunk([0-9]{5,})\.pt", name)
        if not match:
            raise ValueError(f"noncanonical shard filename: {name!r}")
        job, chunk = map(int, match.groups())
        if name != f"job{job:05d}_chunk{chunk:05d}.pt":
            raise ValueError(f"noncanonical shard filename: {name!r}")
        if (not isinstance(record, dict) or record.get("job_id") != job or job not in ids
                or type(record.get("rows")) is not int or record["rows"] < 0
                or any(not isinstance(record.get(key), str)
                       or not re.fullmatch(r"[0-9a-f]{64}", record[key])
                       for key in ("sha256", "tensor_digest"))):
            raise ValueError(f"invalid shard record: {name}")
    return manifest


def read_manifest(root):
    manifest = validate_manifest(json.loads(_safe_bytes(safe_dataset_path(root, "manifest.json"))))
    for name in manifest["shards"]:
        safe_dataset_path(root, name)
    return manifest


def read_shard(root, name, record=None):
    blob = _safe_bytes(safe_dataset_path(root, name))
    if record is not None and hashlib.sha256(blob).hexdigest() != record["sha256"]:
        raise ValueError(f"shard digest mismatch: {name}")
    data = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not all(isinstance(v, torch.Tensor) for v in data.values()):
        raise ValueError(f"shard must contain only named tensors: {name}")
    if record is not None:
        if ("mu" not in data or data["mu"].ndim == 0 or len(data["mu"]) != record["rows"]
                or any(v.ndim == 0 or len(v) != record["rows"] for v in data.values())
                or tensor_digest(data) != record["tensor_digest"]):
            raise ValueError(f"shard tensor schema/digest mismatch: {name}")
    return data


def verify_asset_pins(specification):
    for key in ("policy", "estimator"):
        path = Path(specification[key])
        expected = specification.get(f"{key}_sha256")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{key} asset must be a regular nonsymlink file")
        actual = hashlib.sha256(_safe_bytes(path)).hexdigest()
        if actual != expected:
            raise ValueError(f"{key} asset SHA changed from dataset pin")


def tensor_digest(data):
    h = hashlib.sha256()
    for key, tensor in sorted(data.items()):
        x = tensor.detach().cpu().contiguous()
        h.update(f"{key}:{x.dtype}:{tuple(x.shape)}".encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


class DatasetStore:
    def __init__(self, root, specification):
        self.root = dataset_root(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = safe_dataset_path(self.root, "manifest.json")
        # JSON normalization also ensures tuples compare identically on resume.
        specification = json.loads(json.dumps(specification, sort_keys=True))
        if self.path.exists():
            self.manifest = read_manifest(self.root)
            if self.manifest["specification"] != specification:
                raise ValueError("resume specification/source/checkpoint differs from frozen dataset")
        else:
            if any(self.root.iterdir()):
                raise ValueError("new dataset directory must be empty")
            self.manifest = {"format": DATA_FORMAT, "specification": specification,
                             "shards": {}, "completed_jobs": []}
            validate_manifest(self.manifest)
            atomic_json(self.path, self.manifest)

    def write(self, job_id, chunk, data):
        if type(job_id) is not int or job_id not in {j["id"] for j in self.manifest["specification"]["jobs"]} or type(chunk) is not int or chunk < 0:
            raise ValueError("invalid shard job/chunk identity")
        if "policy" in self.manifest["specification"] or "estimator" in self.manifest["specification"]:
            verify_asset_pins(self.manifest["specification"])
        name = f"job{job_id:05d}_chunk{chunk:05d}.pt"
        path = safe_dataset_path(self.root, name)
        digest = tensor_digest(data)
        if name in self.manifest["shards"]:
            record = self.manifest["shards"][name]
            if not path.exists() or hashlib.sha256(_safe_bytes(path)).hexdigest() != record["sha256"]:
                raise ValueError(f"corrupt durable shard: {name}")
            if digest != record["tensor_digest"]:
                raise ValueError(f"non-deterministic partial-job replay: {name}")
            return
        # Recover a crash after shard commit but before manifest commit, validating replay.
        if path.exists():
            if tensor_digest(read_shard(self.root, name)) != digest:
                raise ValueError(f"orphan shard disagrees with replay: {name}")
        else:
            _atomic_file(path, lambda f: torch.save(data, f), binary=True)
        self.manifest["shards"][name] = {"job_id": job_id, "rows": len(data["mu"]),
                                         "sha256": file_sha256(path), "tensor_digest": digest}
        atomic_json(self.path, self.manifest)

    def complete(self, job_id):
        if type(job_id) is not int or job_id not in {j["id"] for j in self.manifest["specification"]["jobs"]}:
            raise ValueError("invalid completed job identity")
        if job_id not in self.manifest["completed_jobs"]:
            self.manifest["completed_jobs"].append(job_id)
            atomic_json(self.path, self.manifest)


def split_tracks(names, seeds):
    from . import common
    if len(seeds) != 4 or len(set(seeds)) != 4 or any(s < 0 for s in seeds):
        raise ValueError("four distinct nonnegative split seeds are required")
    allowed_bases = {common.base_map(t) for t in common.TRAIN_TRACKS}
    groups = {}
    for name in names:
        base = common.base_map(name)
        if base not in allowed_bases:
            raise ValueError(f"collector only admits TRAIN base maps: {name}")
        groups.setdefault(base, []).append(name)
    if len(groups) < 4:
        raise ValueError("at least four TRAIN base tracks required for disjoint train/cal/dev/final")
    result = {name: [] for name in SPLITS}
    for i, base in enumerate(sorted(groups)):
        result[SPLITS[i % 4]].extend(sorted(set(groups[base])))
    return result


def make_specification(args):
    from . import common
    names = common.track_names(args.tracks)
    seeds = [int(s) for s in args.split_seeds.split(",")]
    partitions = split_tracks(names, seeds)
    jobs = []
    # Every job includes an entire independent stream; episode ids additionally track autoresets.
    used_seeds = set()
    for split, seed in zip(SPLITS, seeds):
        for n in range(args.jobs_per_split):
            job_seed = seed + n
            if job_seed in used_seeds:
                raise ValueError("split seed ranges overlap; space split seeds farther apart")
            used_seeds.add(job_seed)
            jobs.append({"id": len(jobs), "split": split, "seed": job_seed,
                         "track": partitions[split][n % len(partitions[split])],
                         "mode": ("policy_solo", "policy_traffic", "scripted_solo")[n % 3],
                         "cohort": ("randomized", "actuator_lag", "sensor_bias", "sensor_noise")[(n // 3) % 4]})
    return {"recipe": RECIPE, "policy": str(args.policy.resolve()), "policy_sha256": file_sha256(args.policy),
            "estimator": str(args.estimator.resolve()), "estimator_sha256": file_sha256(args.estimator),
            "controller": args.controller, "research_estimator": args.research_estimator,
            "device": args.device, "envs": args.envs, "steps": args.steps, "stride": args.stride,
            "chunk_steps": args.chunk_steps, "speed_cap": args.speed_cap, "jobs": jobs,
            "partitions": partitions, "feature_spec": adaptive_feature_spec().to_meta(),
            "source_sha256": collector_source_hashes(), "torch_version": str(torch.__version__),
            "sim_backend": "cuda_graphs" if torch.device(args.device).type == "cuda" else "eager",
            "compile_tracker": False,
            "labels": "endpoint tire utilization >=.92, >=3/last8; negatives <=.50, >=6/last8",
            "boundary": "terminal old observation/issued command/old targets saved before resetting history",
            "selection": "fixed final epoch; CAL admitted-distribution calibration; DEV report only; FINAL once",
            "limitations": ["simulated TRAIN base maps only", "uniform friction per episode", "FINAL is estimator diagnostic on TRAIN-domain tracks, not policy heldout",
                            "scripted command excitation is marked separately from frozen-policy driving"]}


def observation_for_estimator(obs, ecfg, spec):
    from .obs import ATT_SCALE
    source = (ecfg.v_max_policy,) + (ecfg.imu_gyro_scale,) * 3 + (ecfg.imu_accel_scale,) * 3 + (ATT_SCALE,) * 2
    return {key: obs[key] * torch.tensor([source[i] / spec.scales[i] for i in range(start, stop)],
                                        device=obs[key].device, dtype=obs[key].dtype)
            for key, start, stop in (("speed", 0, 1), ("imu", 1, 7), ("imu_att", 7, 9))}


def terminal_observation(obs, info):
    """Undo only the autoreset observation replacement, without touching environment state."""
    result = {k: v.clone() for k, v in obs.items() if k in ("speed", "imu", "imu_att")}
    if "final" in info:
        ids = info["final"]["ids"]
        for key in result:
            result[key][ids] = info["final_obs"][key]
    return result


class WindowStream:
    def __init__(self, batch, device, spec, job_id):
        self.history = SensorHistory(batch, device, spec)
        self.positive = torch.zeros(batch, 8, dtype=torch.bool, device=device)
        self.low = torch.zeros_like(self.positive)
        self.episode = torch.arange(batch, device=device, dtype=torch.long) + job_id * 10**9
        self.batch = batch
        self.step = torch.zeros(batch, dtype=torch.long, device=device)

    def begin(self, obs):
        self.history.push(obs, None)

    def append(self, obs, issued, targets, done, reset_obs):
        self.history.push(obs, issued)
        mu, positive, low, lower, utilization = targets
        self.positive = self.positive.roll(1, 1)
        self.low = self.low.roll(1, 1)
        self.positive[:, 0], self.low[:, 0] = positive, low
        self.step += 1
        values = {"features": self.history.features, "valid": self.history.valid,
                  "mu": mu, "informative": self.positive.sum(1) >= 3,
                  "low_excitation": self.low.sum(1) >= 6, "lower_bound": lower,
                  "utilization": utilization, "episode": self.episode, "step": self.step,
                  "terminal": done}
        rows = {k: v.detach().cpu().clone() for k, v in values.items()}
        ids = done.nonzero().flatten()
        if ids.numel():
            self.history.reset(ids, reset_obs)
            self.positive[ids] = False
            self.low[ids] = False
            self.episode[ids] += self.batch
            self.step[ids] = 0
        return rows


def cohort_config(cohort, seed, device):
    from ..params import Config
    cfg = Config()
    cfg.sim.device, cfg.sim.compile, cfg.sim.seed = device, False, seed
    cfg.vehicle.wheel_model, cfg.imu.enabled = True, True
    spec = adaptive_feature_spec()
    cfg.rand.ranges["vehicle.mu"] = (spec.mu_min / cfg.vehicle.mu, spec.mu_max / cfg.vehicle.mu)
    overrides = {
        "randomized": {},
        "actuator_lag": {"actuator.servo_tau": (.05, .06), "actuator.motor_tau": (.25, .30),
                         "actuator.cmd_delay": (.025, .03)},
        "sensor_bias": {"imu.accel_bias_x": (.30, .45), "imu.accel_bias_y": (-.45, -.30),
                        "imu.imu_roll": (.05, .07), "odom.speed_scale_err": (-.08, -.05)},
        "sensor_noise": {"imu.gyro_noise": (1.7, 2.), "imu.accel_noise": (1.7, 2.),
                         "imu.vib_accel": (1.7, 2.), "imu.shock_rate": (2., 2.5)},
    }
    cfg.rand.ranges.update(overrides[cohort])
    return cfg



def load_collection_policy(path, device, cfg):
    """Require exact frozen weights and the numerical observation contract before collection."""
    from .model import load_checkpoint
    from .obs import ATT_SCALE, ObsSpec
    model, metadata = load_checkpoint(path, device, strict_names=True)
    saved = metadata.get("spec")
    if not isinstance(saved, dict) or not saved:
        raise ValueError("policy checkpoint has no observation specification")
    spec = ObsSpec(**{key: value for key, value in saved.items()
                      if key in ObsSpec.__dataclass_fields__})
    if not np.isfinite(float(spec.att_scale)) or float(spec.att_scale) != float(ATT_SCALE):
        raise ValueError(f"policy attitude scale {spec.att_scale!r} differs from runtime {ATT_SCALE}")
    if int(spec.n_beams) != spec.n_beams or spec.n_beams < 1:
        raise ValueError("policy n_beams must be a positive integer")
    if not np.isfinite(float(spec.range_max)) or spec.range_max <= 0:
        raise ValueError("policy range_max must be finite and positive")
    cfg.lidar.n_beams = int(spec.n_beams)
    cfg.lidar.range_max = float(spec.range_max)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, saved


@torch.no_grad()
def collect_job(store, job):
    from . import common
    from ..gym_env import EnvConfig
    from .memory import policy_fn as memory_policy_fn
    from .grip_runtime import ControllerRuntime, IssuedCommandSpy
    s = store.manifest["specification"]
    verify_asset_pins(s)
    seed, device = job["seed"], s["device"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = cohort_config(job["cohort"], seed, device)
    model, policy_spec = load_collection_policy(s["policy"], device, cfg)
    verify_asset_pins(s)
    race_size = 2 if job["mode"] == "policy_traffic" else 1
    tracks, rls = common.load_tracks([job["track"]], racelines=race_size > 1)
    ecfg = EnvConfig(action_mode="plan", race_size=race_size, opponent="teacher",
                     compile_tracker=False,
                     max_steps=s["steps"], speed_cap=s["speed_cap"],
                     scan_stack=policy_spec.get("scan_stack", 3), scan_stride=policy_spec.get("scan_stride", 1),
                     hist_len=policy_spec.get("hist_len", 0), hist_stride=policy_spec.get("hist_stride", 2),
                     action_history=policy_spec.get("action_history", EnvConfig.action_history),
                     v_max_policy=policy_spec.get("v_max", 10.0),
                     imu_gyro_scale=policy_spec.get("gyro_scale", 5.0),
                     imu_accel_scale=policy_spec.get("accel_scale", 10.0))
    env = common.make_env(tracks, s["envs"], device, ecfg, cfg, seed, rls=rls)
    common.validate_policy_observation(policy_spec, env)
    kwargs = {"research_estimator": True} if s["research_estimator"] else {}
    runtime = ControllerRuntime(env, s["controller"], s["estimator"], device=device, **kwargs)
    graph_rt = None
    if torch.device(device).type == "cuda":
        from .graph_runtime import prepare_graph_runtime
        env.sim.warmup()
        graph_rt = prepare_graph_runtime(env)
    try:
        runtime.install(graph_rt=graph_rt)
        verify_asset_pins(s)
    except BaseException:
        runtime.release()
        if graph_rt is not None:
            graph_rt.release()
        raise
    backend = {"physics_graphs": graph_rt is not None, "controller_graph": runtime.graphed,
               "compile_tracker": False}
    previous_backend = store.manifest.setdefault("job_backends", {}).get(str(job["id"]))
    if previous_backend is not None and previous_backend != backend:
        runtime.release()
        if graph_rt is not None:
            graph_rt.release()
        raise ValueError("resume graph backend differs from original collection")
    store.manifest["job_backends"][str(job["id"])] = backend
    atomic_json(store.path, store.manifest)
    spy = IssuedCommandSpy(env).install()
    policy = memory_policy_fn(model, env.B, device=device, deterministic=True)
    spec = FeatureSpec.from_meta(s["feature_spec"])
    if abs(spec.control_dt - env.sim.control_dt) > 1e-9:
        raise ValueError("collector/runtime control timestep differs from feature contract")
    stream = WindowStream(env.B, device, spec, job["id"])
    latest = []
    original_step = env.sim.step
    def capture(cmd):
        result = original_step(cmd)
        latest[:] = [v.detach().clone() for v in privileged_targets(env.sim)]
        return result
    env.sim.step = capture
    original_shaper = env.cmd_shaper
    tick = [0]
    if job["mode"] == "scripted_solo":
        def excite(cmd):
            # Recorded command intervention, independent of simulator truth. Keep policy
            # geometry, alternate acceleration/braking and bounded steering excitation.
            t = tick[0] * spec.control_dt
            out = cmd.clone()
            phase = (tick[0] // 32) % 4
            out[:, 1] = (2.0, 8.0, 3.0, 9.0)[phase]
            out[:, 0] = (out[:, 0] + .07 * np.sin(2.1 * t)).clamp(-env.s_max, env.s_max)
            return out
        env.cmd_shaper = excite
    buffered = []
    try:
        obs, _ = env.reset(seed=seed)
        runtime.begin(obs)
        spy.discard()
        if hasattr(policy, "reset"):
            policy.reset()
        stream.begin(observation_for_estimator(obs, ecfg, spec))
        for step in range(s["steps"]):
            tick[0] = step
            runtime.pre_action(obs)
            learner_ids = env.learner.detach().cpu().clone()
            obs, _, term, trunc, info = env.step(policy(obs))
            issued = spy.take()
            runtime.post_step(term, trunc)
            done = term | trunc
            rows = stream.append(observation_for_estimator(terminal_observation(obs, info), ecfg, spec),
                                 issued, latest, done, observation_for_estimator(obs, ecfg, spec))
            rows["job"] = torch.full_like(rows["episode"], job["id"])
            if hasattr(policy, "reset"):
                policy.reset(done)
            if step % s["stride"] == 0 or bool(done.any()):
                mask = rows["valid"].sum(1) >= spec.warm_frames
                if step % s["stride"] != 0:
                    mask &= rows["terminal"]
                # Opponents are teacher driven; only frozen-policy learner rows belong here.
                learner = torch.zeros(env.B, dtype=torch.bool)
                learner[learner_ids] = True
                mask &= learner
                buffered.append({k: v[mask] for k, v in rows.items()})
            if (step + 1) % s["chunk_steps"] == 0 or step + 1 == s["steps"]:
                if buffered:
                    data = {k: torch.cat([b[k] for b in buffered]) for k in buffered[0]}
                    store.write(job["id"], step // s["chunk_steps"], data)
                    buffered.clear()
        if collector_source_hashes() != s["source_sha256"]:
            raise ValueError("source changed within collection job; job remains incomplete")
        verify_asset_pins(s)
        store.complete(job["id"])
    finally:
        env.sim.step = original_step
        env.cmd_shaper = original_shaper
        spy.release()
        runtime.release()
        if graph_rt is not None:
            graph_rt.release()
        if hasattr(env, "close"):
            env.close()


def load_split(root, split):
    root = dataset_root(root)
    manifest = read_manifest(root)
    jobs = {j["id"] for j in manifest["specification"]["jobs"] if j["split"] == split}
    if not jobs.issubset(set(manifest["completed_jobs"])):
        raise ValueError(f"split {split} collection incomplete")
    chunks = []
    for name, record in sorted(manifest["shards"].items()):
        if record["job_id"] not in jobs:
            continue
        chunks.append(read_shard(root, name, record))
    if not chunks or sum(len(c["mu"]) for c in chunks) == 0:
        raise ValueError(f"no windows in {split}")
    return {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}



def policy_rows(data, manifest):
    """Scripted excitation augments TRAIN, but CAL must represent deployed policy admission."""
    policy_jobs = {j["id"] for j in manifest["specification"]["jobs"]
                   if j.get("mode", "policy_solo").startswith("policy_")}
    if "job" not in data:
        # Data written by the early collector has collision-free job-prefixed episode ids.
        job_ids = data["episode"] // 10**9
    else:
        job_ids = data["job"]
    mask = torch.zeros(len(job_ids), dtype=torch.bool)
    for job in policy_jobs:
        mask |= job_ids == job
    if not bool(mask.any()):
        raise ValueError("split has no frozen-policy rows")
    return {k: v[mask] for k, v in data.items()}


def calibrate_admitted(q, confidence, data, threshold=.65):
    selected = confidence >= threshold
    if not bool(selected.any()):
        raise ValueError("CAL has no admitted windows; cannot calibrate deployed selection")
    delta = max(0., float(torch.quantile(q[selected, 0] - data["mu"][selected], .90, interpolation="higher")))
    return {"cal_delta": delta, "target_exceedance": .10, "method": "CAL_admitted_downward_only_empirical_quantile",
            "confidence_threshold": threshold, "admitted_windows": int(selected.sum()),
            "admitted_episodes": int(data["episode"][selected].unique().numel()),
            "selection": "frozen-policy rows only; confidence >= frozen threshold, including false positive admissions"}


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def replay_batch_sizes(batch_size, fraction, replay_enabled):
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("replay_fraction must be finite and strictly between 0 and 1")
    if not replay_enabled:
        if fraction != .5:
            raise ValueError("replay_fraction requires replay_data")
        return batch_size, 0
    replay_size = round(batch_size * fraction)
    if not 0 < replay_size < batch_size:
        raise ValueError("replay fraction must give at least one row from each TRAIN source")
    return batch_size - replay_size, replay_size


def validate_training_tensors(data, spec, label):
    required = ("features", "valid", "mu", "informative", "lower_bound")
    if not isinstance(data, dict) or any(not isinstance(data.get(k), torch.Tensor) for k in required):
        raise ValueError(f"{label} must contain canonical TRAIN feature/target tensors")
    features, valid, mu = data["features"], data["valid"], data["mu"]
    if mu.ndim != 1 or len(mu) < 1 or mu.dtype != torch.float32:
        raise ValueError(f"{label} mu must be a nonempty float32 vector")
    n = len(mu)
    if features.shape != (n, spec.n_frames, spec.n_features) or features.dtype != torch.float32:
        raise ValueError(f"{label} features do not match the canonical feature shape/float32 dtype")
    if valid.shape != features.shape[:2] or valid.dtype != torch.bool:
        raise ValueError(f"{label} valid must be a boolean feature-row mask")
    if bool(((~valid[:, :-1]) & valid[:, 1:]).any()) or bool((valid.sum(1) < spec.warm_frames).any()):
        raise ValueError(f"{label} valid must be newest-first contiguous warm history")
    if data["informative"].shape != (n,) or data["informative"].dtype != torch.bool:
        raise ValueError(f"{label} informative must be a boolean vector")
    if data["lower_bound"].shape != (n,) or data["lower_bound"].dtype != torch.float32:
        raise ValueError(f"{label} lower_bound must be a float32 vector")
    if any(not bool(torch.isfinite(data[k]).all()) for k in required):
        raise ValueError(f"{label} contains nonfinite values")
    if bool(((mu < spec.mu_min) | (mu > spec.mu_max)).any()) or bool((data["lower_bound"] < 0).any()):
        raise ValueError(f"{label} target values exceed the feature contract support")


def fit_asset(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("fit assets must be regular nonsymlink files")
    blob = _safe_bytes(path)
    pin = {"path": str(path.resolve()), "sha256": hashlib.sha256(blob).hexdigest()}
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True), pin


def verify_fit_pins(pins):
    for name, pin in pins.items():
        path = Path(pin["path"])
        if path.is_symlink() or hashlib.sha256(_safe_bytes(path)).hexdigest() != pin["sha256"]:
            raise ValueError(f"{name} fit asset changed from the predeclared pin")


def epoch_training_batches(train, batch_size, replay=None, replay_fraction=.5):
    current_size, replay_size = replay_batch_sizes(batch_size, replay_fraction, replay is not None)
    order = torch.randperm(len(train["mu"]))
    if replay is None:
        # Preserve the historical scratch-fit RNG order and partial final minibatch exactly.
        for ix in order.split(batch_size):
            yield {k: v[ix] for k, v in train.items()}
        return
    steps = math.ceil(len(order) / current_size)
    padding = steps * current_size - len(order)
    if padding:
        order = torch.cat([order, torch.randint(len(train["mu"]), (padding,))])
    needed = steps * replay_size
    replay_orders = []
    while needed:
        permutation = torch.randperm(len(replay["mu"]))[:needed]
        replay_orders.append(permutation)
        needed -= len(permutation)
    replay_order = torch.cat(replay_orders)
    for current, previous in zip(order.split(current_size), replay_order.split(replay_size)):
        yield {k: torch.cat([train[k][current], replay[k][previous]])
               for k in ("features", "valid", "mu", "informative", "lower_bound")}


def fit_dataset(data_dir, out, seed, epochs, batch_size=128, hidden=64, learning_rate=.002,
                init_estimator=None, replay_data=None, replay_fraction=.5):
    if min(epochs, batch_size, hidden) < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive finite training hyperparameters required")
    current_size, replay_size = replay_batch_sizes(batch_size, replay_fraction, replay_data is not None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("fit output must be new; a new fit is a new preregistered candidate")
    manifest = read_manifest(data_dir)
    spec = FeatureSpec.from_meta(manifest["specification"]["feature_spec"])
    pins, initial, replay = {}, None, None
    if init_estimator is not None:
        initial, pins["init_estimator"] = fit_asset(init_estimator)
        from .adaptive_grip import FORMAT
        if (not isinstance(initial, dict) or initial.get("format") != FORMAT
                or FeatureSpec.from_meta(initial["feature_spec"]) != spec):
            raise ValueError("init estimator format/feature spec differs from current dataset")
        if (set(initial.get("arch", {})) != {"hidden", "wheelbase"}
                or initial["arch"]["hidden"] != hidden):
            raise ValueError("init estimator architecture/hidden differs from requested fit")
        if initial.get("consumption") != asdict(AdaptiveConsumption()):
            raise ValueError("init estimator consumption must match the unchanged fit consumption")
        if not all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all())
                   for v in initial.get("state_dict", {}).values()):
            raise ValueError("init estimator weights must be finite tensors")
    if replay_data is not None:
        replay, pins["replay_data"] = fit_asset(replay_data)
        validate_training_tensors(replay, spec, "replay TRAIN")
    declaration = {"recipe": RECIPE if not pins else "adaptive_grip_policy_refit_warm_replay_v1",
                   "dataset_manifest_sha256": file_sha256(Path(data_dir) / "manifest.json"),
                   "source_sha256": collector_source_hashes(), "seed": seed, "epochs": epochs, "batch_size": batch_size,
                   "hidden": hidden, "learning_rate": learning_rate, "selection": "fixed final epoch; no DEV selection",
                   "consumption_version": "adaptive_grip_v2_q10_evidence_hold", "deployment_approved": False}
    if pins:
        declaration["fit_assets"] = pins
        declaration["initialization"] = "weights only; CAL refitted from current policy rows" if initial is not None else "scratch"
    if replay is not None:
        declaration["replay"] = {"training_data_only": True, "requested_fraction": replay_fraction,
                                 "effective_fraction": replay_size / batch_size, "rows_per_batch": replay_size,
                                 "current_rows_per_batch": current_size, "rows": len(replay["mu"]),
                                 "sampling": "current TRAIN shuffled each epoch, final batch padded uniformly; replay shuffled without replacement, cycling fresh permutations if needed"}
    atomic_json(out / "predeclared.json", declaration)
    train = load_split(data_dir, "train")
    cal = policy_rows(load_split(data_dir, "cal"), manifest)
    validate_training_tensors(train, spec, "current TRAIN")
    if not bool(train["informative"].any()):
        raise ValueError("TRAIN has no informative windows")
    verify_fit_pins(pins)
    torch.manual_seed(seed)
    net = AdaptiveGripNet(spec, **initial["arch"]) if initial is not None else AdaptiveGripNet(spec, hidden=hidden)
    if initial is not None:
        net.load_state_dict(initial["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    curve = []
    for epoch in range(epochs):
        verify_fit_pins(pins)
        net.train()
        losses = []
        for batch in epoch_training_batches(train, batch_size, replay, replay_fraction):
            q, logits = net.distribution(batch["features"], batch["valid"])
            loss = adaptive_loss(q, logits, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
            optimizer.step()
            losses.append(float(loss.detach()))
        curve.append(float(np.mean(losses)))
        print(f"epoch {epoch + 1}/{epochs} loss={curve[-1]:.6f}", flush=True)
    verify_fit_pins(pins)
    if file_sha256(Path(data_dir) / "manifest.json") != declaration["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest changed during fitting")
    if collector_source_hashes() != declaration["source_sha256"]:
        raise ValueError("source changed during fitting")
    q, confidence = predictions(net, cal)
    calibration = calibrate_admitted(q, confidence, cal, AdaptiveConsumption().confidence_threshold)
    prior = float(train["mu"][train["informative"]].median())
    meta = {"deployment_approved": False, "promoted": False, "pilot_only": True,
            "training": declaration, "training_prior_mu": prior,
            "training_domain": DATA_FORMAT, "consumption_version": declaration["consumption_version"]}
    sha = save_adaptive_grip_estimator(out / "candidate.pt", net, spec, calibration, meta)
    dev = policy_rows(load_split(data_dir, "dev"), manifest)
    dq, dc = predictions(net, dev)
    report = {"candidate_sha256": sha, "promoted": False, "training_loss": curve,
              "calibration": calibration, "cal": evaluate(q, confidence, cal, prior, calibration["cal_delta"]),
              "dev": evaluate(dq, dc, dev, prior, calibration["cal_delta"]), "final_opened": False}
    atomic_json(out / "report.json", finite_json(report))
    return report


def score_final(data, candidate, out):
    root, out = Path(data), Path(out)
    if out.exists():
        raise ValueError("final report already exists")
    est = load_adaptive_grip_estimator(candidate)
    expected = est.meta.get("training", {}).get("dataset_manifest_sha256")
    if expected != file_sha256(root / "manifest.json"):
        raise ValueError("candidate training dataset does not match FINAL manifest")
    claim = root / "final_opened.json"
    # A crash after access leaves the marker, so no silent repeated candidate selection.
    with claim.open("x") as f:
        json.dump({"candidate_sha256": file_sha256(candidate), "report": str(out.resolve())}, f)
        f.flush()
        os.fsync(f.fileno())
    manifest = read_manifest(root)
    final = policy_rows(load_split(root, "final"), manifest)
    q, confidence = predictions(est.net, final)
    report = evaluate(q, confidence, final, est.meta["training_prior_mu"], est.cal_delta)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, finite_json({"candidate_sha256": file_sha256(candidate), "metrics": report, "promoted": False}))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--out", type=Path, required=True)
    collect.add_argument("--policy", type=Path, required=True)
    collect.add_argument("--estimator", type=Path, required=True)
    collect.add_argument("--controller", choices=["auto", "estimated"], default="auto")
    collect.add_argument("--research-estimator", action="store_true")
    collect.add_argument("--tracks", required=True)
    collect.add_argument("--split-seeds", default="18101,19101,20101,21101")
    collect.add_argument("--jobs-per-split", type=int, default=12)
    collect.add_argument("--envs", type=int, default=4)
    collect.add_argument("--steps", type=int, default=400)
    collect.add_argument("--stride", type=int, default=4)
    collect.add_argument("--chunk-steps", type=int, default=32)
    collect.add_argument("--speed-cap", type=float, default=10.)
    collect.add_argument("--device", default="cpu")
    fit = sub.add_parser("fit")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--epochs", type=int, required=True)
    fit.add_argument("--batch-size", type=int, default=128)
    fit.add_argument("--hidden", type=int, default=64)
    fit.add_argument("--learning-rate", type=float, default=.002)
    fit.add_argument("--init-estimator", type=Path,
                     help="optional compatible adaptive estimator weights; current CAL calibration is refitted")
    fit.add_argument("--replay-data", type=Path,
                     help="optional controlled TRAIN-only canonical sensor/target tensor dictionary; never CAL/DEV/FINAL")
    fit.add_argument("--replay-fraction", type=float, default=.5,
                     help="replay share per minibatch in (0,1), rounded to rows; requires --replay-data when changed")
    final = sub.add_parser("final")
    final.add_argument("--data", type=Path, required=True)
    final.add_argument("--candidate", type=Path, required=True)
    final.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "collect":
        if min(args.envs, args.jobs_per_split, args.stride, args.chunk_steps) < 1 or args.steps < 8:
            parser.error("positive collection sizes and >=8 steps required")
        store = DatasetStore(args.out, make_specification(args))
        for job in store.manifest["specification"]["jobs"]:
            if job["id"] not in store.manifest["completed_jobs"]:
                if collector_source_hashes() != store.manifest["specification"]["source_sha256"]:
                    raise ValueError("source changed during collection; freeze source before collecting")
                collect_job(store, job)
                print(f"completed {job}", flush=True)
    elif args.command == "fit":
        if min(args.epochs, args.batch_size, args.hidden) < 1 or args.learning_rate <= 0:
            parser.error("positive training hyperparameters required")
        try:
            replay_batch_sizes(args.batch_size, args.replay_fraction, args.replay_data is not None)
        except ValueError as error:
            parser.error(str(error))
        fit_dataset(args.data, args.out, args.seed, args.epochs, args.batch_size, args.hidden, args.learning_rate,
                    init_estimator=args.init_estimator, replay_data=args.replay_data,
                    replay_fraction=args.replay_fraction)
    else:
        score_final(args.data, args.candidate, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
