"""Small JSON checkpoint inspection worker. Torch is imported only in the child process."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys


def _plain(value, depth=0):
    if depth > 12:
        raise ValueError("checkpoint metadata is too deeply nested")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint metadata contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 4096:
        return [_plain(item, depth + 1) for item in value]
    if isinstance(value, dict) and len(value) <= 512 and all(isinstance(key, str) for key in value):
        return {key: _plain(item, depth + 1) for key, item in value.items()}
    raise ValueError("checkpoint metadata must contain only JSON values, not tensors")


def inspect_checkpoint(path):
    import torch
    path = Path(path).expanduser().resolve(strict=True)
    before = path.stat()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    extra = checkpoint.get("extra") or {}
    experiment = extra.get("experiment") or {}
    reference = experiment.get("original_reference") or {}
    controller = experiment.get("controller") or {}
    model = checkpoint.get("meta") or {}
    observation = extra.get("spec") or {}
    model_fields = ("n_stack", "n_beams", "proprio_dim", "priv_dim", "act_dim", "scan_deltas",
                    "scan_stem", "temporal_encoder", "memory", "scan_channels", "opp_token",
                    "cond_dim", "cond", "priv_adapter", "motion", "motion_heads", "future_head", "floor_head")
    proprio_dim = None
    if observation:
        from ...learn.obs import ObsSpec
        proprio_dim = ObsSpec(**observation).proprio_dim
    result = _plain({
        "format": "f1sim-checkpoint-metadata-v1", "path": str(path),
        "phase": extra.get("phase"), "run": extra.get("run"), "iter": extra.get("iter"),
        "adaptation": experiment.get("adaptation", "off"),
        "original_reference": {key: reference[key] for key in ("path", "sha256") if key in reference},
        "stage_schedule": extra.get("stage_schedule"),
        "collection_mix": extra.get("collection_mix"),
        "controller": controller.get("arm", "legacy"),
        "embedded_estimator": bool(controller.get("estimator")),
        "observation_spec": observation,
        "observation_proprio_dim": proprio_dim,
        "model_contract": {key: model[key] for key in model_fields if key in model},
        "action_mode": extra.get("action_mode"),
    })
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("checkpoint changed while reading metadata")
    result["file_size"], result["mtime_ns"] = after.st_size, after.st_mtime_ns
    payload = json.dumps(result, allow_nan=False, ensure_ascii=False)
    if len(payload.encode()) > 131072:
        raise ValueError("checkpoint metadata exceeds the panel limit")
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValueError("one checkpoint path is required")
        print(json.dumps(inspect_checkpoint(args[0]), allow_nan=False, ensure_ascii=False))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
