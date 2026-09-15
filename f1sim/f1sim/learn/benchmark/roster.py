"""Explicitly pinned roster entries. No auto-aliases, no symlink following.

`run="latest"` currently resolves to the SGR job that is running right now, whose recorded controller
arm the default loader refuses -- backend hit exactly that while running the publication suite. Every
entry here carries an absolute path and the sha256 it must hash to, and a mismatch fails closed.

What is rejected is an **unresolved alias**, not the substring "latest": a symlink, a path segment
that is literally `latest`, or a basename that is one of the moving pointers training rewrites. An
immutable baseline copy named `ppo_latest_frozen.pt` is a real pinned file and must be accepted --
the bytes are what settle it, and they are hashed.

Row identity is (checkpoint_sha256, controller_arm, estimator_sha): the same weights under two arms
are two evaluated systems, which is why the key is not the checkpoint alone.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os

#: Derived from the runtime's list rather than restated, so the roster can never pin an arm no
#: `ControllerRuntime` will build. The `oracle` arms are dropped: their friction is privileged, so a
#: number they produce is not a benchmark result. The `+tcs` arms are in, because
#: `fixed_low+tcs` is what the car deploys.
from f1sim.learn.grip_runtime import ARMS as _RUNTIME_ARMS, split_arm       # noqa: E402

#: The arm an **external** baseline declares. It is not a `grip_runtime` arm and cannot be one:
#: those name what is installed on the plan tracker, and a published baseline that emits (steer,
#: speed) never reaches a plan tracker. Spelled `none` rather than `legacy` because `legacy` means
#: "the tracker, with nothing on it", which is a different system.
EXTERNAL_ARM = "none"

#: Baselines `f1sim.learn.baselines.load` knows. Restated rather than imported so that verifying a
#: roster does not drag in onnxruntime/torch backends; `load_actor` is where the real load happens.
EXTERNAL_KINDS = ("tinylidarnet", "end2race")

ARMS = tuple(a for a in _RUNTIME_ARMS if not split_arm(a)[0] == "oracle")

#: Filenames training rewrites in place. A roster may never point at one of these directly.
MOVING_POINTERS = ("ppo_latest.pt", "latest.pt", "last.pt")


def sha256_file(path: str, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Entry:
    system_id: str
    path: str
    checkpoint_sha256: str
    controller_arm: str
    estimator_path: str | None = None
    estimator_sha256: str | None = None
    #: An **external** published baseline instead of an f1sim checkpoint: `"tinylidarnet"` or
    #: `"end2race"`. `path` is then the weights file (the JSON may spell it `weights`, which `load`
    #: maps onto `path` so every pin check below -- symlink, moving pointer, `latest` segment, sha
    #: -- applies unchanged). `controller_arm` must be `"none"`.
    kind: str | None = None
    #: Driver options for an external entry: `speed_map`, `skip_n`, `scan_fill`, `tick_hz`,
    #: `n_features`, `hidden_scale`, `repo`. They change what the system IS -- End2Race with the
    #: unseen quarter of its scan filled at 30 m and the same weights filled at 0 m are two systems
    #: -- so they are part of `key()`.
    options: dict = None
    #: A legacy-trained checkpoint evaluated under a non-legacy arm. The frozen original under
    #: `estimated` is the deliberate case; it has to be declared so it cannot happen by accident.
    cross_runtime: bool = False
    note: str = ""

    def key(self) -> tuple:
        """Row identity. For an external entry the driver options are part of it: the same weights
        under two preprocessing choices are two evaluated systems, exactly as the same checkpoint
        under two controller arms is."""
        return (self.checkpoint_sha256, self.controller_arm, self.estimator_sha256,
                self.kind, json.dumps(self.options or {}, sort_keys=True))

    def resolved(self) -> str:
        """Canonical absolute path. A relative path is a fine way to name a real file."""
        return os.path.abspath(os.path.expanduser(self.path))

    def verify(self) -> None:
        """Fail closed on anything unpinned, missing, drifted, or arm-inconsistent."""
        if self.kind is not None:
            if self.kind not in EXTERNAL_KINDS:
                raise ValueError(f"{self.system_id}: unknown external kind {self.kind!r}; known: "
                                 f"{', '.join(EXTERNAL_KINDS)}")
            if self.controller_arm != EXTERNAL_ARM:
                raise ValueError(
                    f"{self.system_id}: an external baseline declares arm "
                    f"{self.controller_arm!r}. These models publish (steer, speed) directly -- "
                    f"there is no plan for a tracker to follow and no solver for an arm to wrap -- "
                    f"so the only honest declaration is {EXTERNAL_ARM!r}.")
            if self.estimator_path or self.estimator_sha256:
                raise ValueError(f"{self.system_id}: an external baseline has no controller arm, so "
                                 f"it cannot have a friction estimator either")
            if self.cross_runtime:
                raise ValueError(f"{self.system_id}: cross_runtime declares a checkpoint evaluated "
                                 f"under an arm it did not train under; an external baseline has no "
                                 f"arm at all")
        elif self.controller_arm == EXTERNAL_ARM:
            raise ValueError(f"{self.system_id}: arm {EXTERNAL_ARM!r} names the absence of a plan "
                             f"tracker and only an external baseline (`kind`) can declare it")
        elif self.controller_arm not in ARMS:
            raise ValueError(f"{self.system_id}: unknown arm {self.controller_arm!r}")
        if self.kind is not None and self.options is not None and not isinstance(self.options, dict):
            raise ValueError(f"{self.system_id}: options must be an object, got "
                             f"{type(self.options).__name__}")
        if os.path.basename(self.path) in MOVING_POINTERS:
            raise ValueError(
                f"{self.system_id}: {os.path.basename(self.path)} is rewritten in place by training; "
                f"pin an immutable copy instead")
        if "latest" in self.resolved().split(os.sep):
            raise ValueError(f"{self.system_id}: unresolved 'latest' path segment in {self.path!r}")
        if os.path.islink(self.path):
            raise ValueError(f"{self.system_id}: path is a symlink; pin the real file")
        if not os.path.exists(self.resolved()):
            raise FileNotFoundError(f"{self.system_id}: {self.resolved()}")
        actual = sha256_file(self.resolved())
        if actual != self.checkpoint_sha256:
            raise ValueError(f"{self.system_id}: sha mismatch\n  declared {self.checkpoint_sha256}"
                             f"\n  actual   {actual}")
        if self.kind is not None:
            return                       # everything below is about tracker arms; there is none
        base = split_arm(self.controller_arm).base
        if (base == "estimated") != bool(self.estimator_sha256):
            raise ValueError(f"{self.system_id}: arm {self.controller_arm} and estimator pin disagree")
        if base == "estimated":
            # A declared sha with no resolvable file is not a pin. The runtime cannot load it, so
            # the entry must fail here rather than at GPU time.
            if not self.estimator_path:
                raise ValueError(f"{self.system_id}: estimated arm needs an estimator_path, not a "
                                 f"bare sha")
            if not os.path.exists(os.path.abspath(os.path.expanduser(self.estimator_path))):
                raise FileNotFoundError(f"{self.system_id}: estimator {self.estimator_path}")
            if os.path.islink(self.estimator_path):
                raise ValueError(f"{self.system_id}: estimator path is a symlink; pin the real file")
        if self.estimator_path:
            actual_e = sha256_file(os.path.abspath(os.path.expanduser(self.estimator_path)))
            if actual_e != self.estimator_sha256:
                raise ValueError(f"{self.system_id}: estimator sha mismatch\n  declared "
                                 f"{self.estimator_sha256}\n  actual   {actual_e}")


def probe(path: str) -> dict:
    """Read a checkpoint's metadata without executing anything in it.

    `weights_only=True` on CPU: these files are tensors plus plain dicts (`state_dict`, `meta`,
    `extra`), which is exactly the format that restriction allows, so nothing is given up by using
    it. The frozen production loader is untouched -- this is a read-only probe for roster
    verification, not a runtime path.
    """
    import torch
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{path}: not a readable checkpoint ({type(exc).__name__}). A roster "
                         f"entry must point at a real checkpoint.") from exc
    if not isinstance(d, dict):
        raise ValueError(f"{path}: checkpoint is {type(d).__name__}, expected a dict")
    meta = d.get("meta") or {}
    extra = d.get("extra") or {}
    spec = extra.get("spec") or {}
    # The production contract, matching `model.controller_arm_of`: the arm lives at
    # extra.experiment.controller.arm and defaults to "legacy" when absent. Reading
    # meta.controller_arm returned None for every real controller-trained checkpoint, so
    # `check_arm_matches_record` was failing OPEN -- it accepted an estimated checkpoint declared
    # as legacy because it had nothing to compare against.
    experiment = extra.get("experiment") or {}
    controller = experiment.get("controller") or {}
    recorded_arm = str(controller.get("arm") or "legacy")
    return {"recorded_arm": recorded_arm,
            "recorded_estimator_path": controller.get("estimator_path"),
            "act_dim": meta.get("act_dim") or spec.get("act_dim"),
            "n_beams": meta.get("n_beams") or spec.get("n_beams"),
            "obs_spec": spec,
            "n_params": sum(int(v.numel()) for v in d.get("state_dict", {}).values()
                            if hasattr(v, "numel")),
            "top_keys": sorted(d.keys())}


def probe_external(entry: "Entry") -> dict:
    """What an external entry pins, without executing the weights.

    Deliberately shallow: the file is hashed by `verify`, and what it *is* -- beam count, parameter
    count, backend -- is reported by the driver itself at load time and lands in the row's protocol
    block. Opening an ONNX graph or a torch `state_dict` here to restate that would be a second,
    divergeable description of the same file.
    """
    return {"recorded_arm": EXTERNAL_ARM, "recorded_estimator_path": None,
            "kind": entry.kind, "options": dict(entry.options or {}),
            "weights": entry.resolved(), "external": True}


def check_arm_matches_record(entry: "Entry") -> dict:
    """Declared arm vs the arm the checkpoint records. Same rule as `model_adapter.load_actor`.

    A policy is only comparable under the controller it trained against, so a non-legacy checkpoint
    may only run under its own arm. A legacy-trained checkpoint under a non-legacy arm is a
    legitimate cross-runtime reference -- the frozen original under `estimated` is exactly that --
    but it must be declared, so it cannot arise from a typo.
    """
    if entry.kind is not None:
        return probe_external(entry)
    info = probe(entry.resolved())
    rec = info["recorded_arm"]
    if rec is None:
        return info
    if rec != "legacy" and rec != entry.controller_arm:
        raise ValueError(f"{entry.system_id}: trained under {rec!r} but would run under "
                         f"{entry.controller_arm!r}; a policy is only comparable under the "
                         f"controller it trained against")
    if rec == "legacy" and entry.controller_arm != "legacy" and not entry.cross_runtime:
        raise ValueError(f"{entry.system_id}: legacy-trained under {entry.controller_arm!r} is a "
                         f"cross-runtime reference and must set cross_runtime=true")
    return info


def _from_json(e: dict) -> Entry:
    """One roster record -> `Entry`. `weights` is an accepted spelling of `path` for an external
    entry, because that is what CONTRACT.md writes and because "the weights" is what it is; mapping
    it onto `path` here means every pin check downstream applies to it unchanged."""
    e = dict(e)
    if "weights" in e:
        if e.get("path") and e["path"] != e["weights"]:
            raise ValueError(f"{e.get('system_id')}: both `path` and `weights` are set and differ; "
                             f"they name the same file")
        e["path"] = e.pop("weights")
    unknown = sorted(set(e) - set(Entry.__dataclass_fields__))
    if unknown:
        raise ValueError(f"{e.get('system_id')}: unknown roster field(s) {unknown}. A field this "
                         f"loader drops is a pin nobody is checking.")
    return Entry(**e)


def load(path: str) -> list[Entry]:
    with open(path) as fh:
        raw = json.load(fh)
    entries = [_from_json(e) for e in raw["systems"]]
    # Unique system_id BEFORE any lookup dict is built. Callers construct {e.system_id: e}, so a
    # repeat silently replaces the earlier entry and the expected-system set collapses -- one
    # declared system would vanish from the roster and from the results without an error.
    ids = {}
    for e in entries:
        sid = str(e.system_id or "").strip()
        if not sid:
            raise ValueError(f"roster entry with no system_id: {e}")
        if sid in ids:
            raise ValueError(f"duplicate system_id {sid!r}: two entries share it "
                             f"({ids[sid]} and {e.checkpoint_sha256[:16]}); system ids address "
                             f"results and must be unique")
        ids[sid] = e.checkpoint_sha256[:16]
    seen = {}
    for e in entries:
        if e.key() in seen:
            raise ValueError(f"duplicate system identity: {e.system_id} and {seen[e.key()]}")
        seen[e.key()] = e.system_id
    return entries


def verify_all(entries: list[Entry]) -> dict:
    for e in entries:
        e.verify()
    return {"n_systems": len(entries),
            "n_unique_weights": len({e.checkpoint_sha256 for e in entries}),
            "systems": [e.system_id for e in entries]}
