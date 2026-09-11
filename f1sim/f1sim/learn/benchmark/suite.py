"""Suite definition and its freeze hash.

The freeze is the whole point: scenario definitions are fixed before any system is scored, so a map,
seed or obstacle placement cannot be adjusted once outcomes are visible. `run` records the hash in
every result row and refuses a mismatch.

Geometry and the non-candidate feasibility smoke run *before* the freeze -- they involve no evaluated
checkpoint, so they cannot leak an outcome into the scenario choice.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib
import json

SUITE_VERSION = "v1"

#: Fixed friction levels. 0.94401 is the MIDPOINT of [MU_MIN, MU_MAX], not the vehicle's nominal mu
#: (params.py:25 = 1.0489). Labelled so the midpoint is never read as the design point.
MU_LOW, MU_MID, MU_HIGH = 0.73423, 0.94401, 1.15379
MU_LABELS = {MU_LOW: "low (MU_MIN)", MU_MID: "mid (range midpoint)", MU_HIGH: "high (MU_MAX)"}
MU_NOMINAL_VEHICLE = 1.0489

SEEDS = (4401, 4402)
HOLD_SECONDS = 1.0
OPP_SPEED_RANGE = (0.6, 0.8)
S_START_OFFSET_M = -15.0          # avoidance start, relative to the obstacle


@dataclass(frozen=True)
class Cell:
    suite: str
    map_id: str
    mu: float
    seed: int
    envs: int

    def cell_id(self) -> str:
        return f"{self.suite}:{self.map_id}:{self.mu}:{self.seed}"


@dataclass
class Suite:
    """The frozen scenario definition. Everything that can change an outcome lives here."""
    version: str = SUITE_VERSION
    solo_maps: tuple = ("gen:control:1400", "real:korea_2026_competition", "gen:control:9100")
    obstacle_maps: tuple = ("gen:control:1400", "gen:control:9100")
    race_maps: tuple = ("gen:control:1400", "gen:control:9100")
    solo_mus: tuple = (MU_LOW, MU_MID, MU_HIGH)
    paired_mus: tuple = (MU_LOW, MU_MID)
    seeds: tuple = SEEDS
    envs: int = 8
    race_size: int = 2
    hold_seconds: float = HOLD_SECONDS
    opp_speed_range: tuple = OPP_SPEED_RANGE
    s_start_offset_m: float = S_START_OFFSET_M
    #: Protocol fields the adapter reads. They change outcomes, so they are hashed with the rest --
    #: a benchmark whose speed cap or noise policy drifted silently is not the same benchmark.
    speed_cap: float = 9.0
    budget_laps: float = 3.0
    sensor_noise: bool = True
    backend: str = "f1sim"
    obstacle_window_m: float = 4.0
    #: filled by the geometry stage, before the freeze
    placements: dict = field(default_factory=dict)
    #: measured by the non-candidate feasibility smoke, before the freeze
    calibration: dict = field(default_factory=dict)
    #: all three maps are reused development tracks; no unseen-map claim attaches to any result
    reused_maps_note: str = ("gen:control:1400 is an SGR training map; korea_2026_competition and "
                             "gen:control:9100 are existing diagnostic maps. No unseen-map or "
                             "generalisation claim attaches to any result.")

    def cells(self) -> list[Cell]:
        out = []
        for m in self.solo_maps:
            for mu in self.solo_mus:
                for s in self.seeds:
                    out.append(Cell("S", m, mu, s, self.envs))
        for m in self.obstacle_maps:
            for mu in self.paired_mus:
                for s in self.seeds:
                    out.append(Cell("A", m, mu, s, self.envs))
        for m in self.race_maps:
            for mu in self.paired_mus:
                for s in self.seeds:
                    out.append(Cell("O", m, mu, s, self.envs))
        return out

    def adapter_suite(self) -> dict:
        """The subset `model_adapter.prepare_cell` reads, by its contract's key names."""
        return {"envs": self.envs, "speed_cap": self.speed_cap, "sensor_noise": self.sensor_noise,
                "race_size": self.race_size, "budget_laps": self.budget_laps,
                # The opponent must be the fixed raceline teacher at the declared speed range. The
                # EnvConfig default is `policy`, which would make the opponent a copy of the
                # candidate -- the exact coupling the independence gate exists to rule out.
                "opponent": "teacher", "opp_speed_range": list(self.opp_speed_range)}

    def expected_trials(self) -> dict:
        c = self.cells()
        return {s: sum(x.envs for x in c if x.suite == s) for s in ("S", "A", "O")}

    def as_dict(self) -> dict:
        return asdict(self)

    def freeze_hash(self) -> str:
        """Stable hash over everything that can change an outcome.

        `calibration` is excluded: it records how long cells took, which is a property of the machine
        rather than of the scenario, and must not invalidate a freeze when re-measured.
        """
        d = self.as_dict()
        d.pop("calibration", None)
        d.pop("reused_maps_note", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def save(self, path: str) -> str:
        h = self.freeze_hash()
        with open(path, "w") as fh:
            json.dump({"suite": self.as_dict(), "freeze_sha256": h}, fh, indent=2, sort_keys=True)
        return h


#: Runtime modules whose contents can change a measurement: physics, sensing, control, the model
#: and the observation layout. Truncating these or omitting the sensor models would let a changed
#: LiDAR or dynamics pass as the same protocol.
RUNTIME_MODULES = ("sim.py", "gym_env.py", "mpc.py", "params.py", "track.py", "dynamics.py",
                   "lidar.py", "imu.py", "odom.py", "teacher.py",
                   "learn/model.py", "learn/obs.py", "learn/common.py", "learn/evaluate.py",
                   "learn/graph_runtime.py", "learn/grip_runtime.py", "learn/grip_control.py",
                   "learn/grip_estimator.py", "learn/evaluation_metrics.py")


def source_digest() -> dict:
    """Full sha256 of every module a measurement depends on, plus this benchmark's own code.

    A result is only comparable with another taken under the same code, and that includes the code
    doing the measuring: a changed recorder or obstacle placer changes the number as surely as a
    changed simulator. Hashes are full length -- a 16-character prefix is a weaker claim than the
    comparison it is used for.
    """
    import hashlib as _h
    import os as _os
    out = {"runtime": {}, "benchmark": {}, "missing": []}
    try:
        import f1sim
        root = _os.path.dirname(f1sim.__file__)
    except Exception:
        root = None
    if root:
        for rel in RUNTIME_MODULES:
            p_ = _os.path.join(root, rel)
            if _os.path.exists(p_):
                out["runtime"][rel] = _h.sha256(open(p_, "rb").read()).hexdigest()
            else:
                out["missing"].append(rel)
    here = _os.path.dirname(_os.path.abspath(__file__))
    for fn in sorted(_os.listdir(here)):
        if fn.endswith(".py"):
            out["benchmark"][fn] = _h.sha256(open(_os.path.join(here, fn), "rb").read()).hexdigest()
    return out


def identity_hash(d: dict) -> str:
    """One digest over everything that makes two rows comparable."""
    import hashlib as _h
    return _h.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def protocol_identity(suite: "Suite", entry=None, effective: dict | None = None) -> dict:
    """Everything a row must carry to be comparable with another row.

    `effective` is what the adapter reports the cell actually turned out to be -- the ObsSpec it
    built, the friction it applied, the budget it derived. Declared and effective are both recorded
    because a silent difference between them is exactly the failure this is meant to catch.
    """
    d = {"suite_version": suite.version, "suite_freeze_sha256": suite.freeze_hash(),
         "source_digest": source_digest(), "speed_cap": suite.speed_cap,
         "budget_laps": suite.budget_laps, "sensor_noise": suite.sensor_noise,
         "backend": suite.backend}
    # `effective` is recorded as evidence but deliberately NOT hashed: it is per-cell (its map,
    # friction and budget differ by construction), while the identity hash exists to prove that two
    # rows share one protocol. Hashing it would make every cell a different identity and refuse
    # every legitimate resume.
    if entry is not None:
        d.update({"system_id": entry.system_id,
                  "checkpoint_sha256": entry.checkpoint_sha256,
                  "controller_arm": entry.controller_arm,
                  "estimator_sha256": entry.estimator_sha256,
                  "checkpoint_path": entry.resolved()})
    d["identity_sha256"] = identity_hash(d)
    if effective:
        d["effective"] = effective
    return d


def load(path: str) -> tuple[Suite, str]:
    """Load and verify: a suite whose recorded hash no longer matches its content is refused."""
    with open(path) as fh:
        raw = json.load(fh)
    s = Suite(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw["suite"].items()})
    actual = s.freeze_hash()
    if actual != raw["freeze_sha256"]:
        raise ValueError(f"suite file was edited after freeze\n  recorded {raw['freeze_sha256']}"
                         f"\n  actual   {actual}")
    return s, actual
