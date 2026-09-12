"""Suite definition and its freeze hash.

The freeze is the whole point: scenario definitions are fixed before any system is scored, so a map,
seed or obstacle placement cannot be adjusted once outcomes are visible. `run` records the hash in
every result row and refuses a mismatch.

Geometry and the non-candidate feasibility smoke run *before* the freeze -- they involve no evaluated
checkpoint, so they cannot leak an outcome into the scenario choice.

Two scenario sets live here. `Suite()` is v1, unchanged; `v2()` is the held-out set. They share
every protocol field -- friction levels, seeds, envs, race size, speed cap, lap budget, sensor
noise, opponent -- and differ only in which maps those fields are applied to, so a v1 row and a v2
row are comparable as measurements and differ in the one thing the version names. `of(version)`
picks between them; the CLI's `geometry --version` is the only caller that needs to.
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
    #: Where these maps come from, and therefore what a score on them may be called. Prose, and
    #: excluded from the freeze hash for that reason -- but it is the field that decides whether a
    #: number can be reported as generalisation, so every suite has to fill it in.
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


#: Suite v2's solo family: every held-out base map that has a raceline the car fits through
#: (measured, see the held-out suite report). `HELDOUT_TRACKS` also carries each real floor's
#: `~rev`, which is the same geometry driven the other way -- the solo family takes the base map
#: once, because a benchmark cell is a scenario and not a direction.
V2_SOLO_MAPS = ("real:korea_2025_iccas", "real:blackbox2022_3", "rt:Monza", "gen:competition:0",
                "gen:control:9100", "gen:competition:9200+pinch9200",
                "real:map16x07", "real:map12x16")

#: Avoidance: the two unseen real floors plus `gen:control:9100`, the one v1 map that was already
#: outside training. All three host a proven blocking obstacle, and the scripted avoidance expert
#: clears every one at both friction levels.
V2_OBSTACLE_MAPS = ("real:map16x07", "real:map12x16", "gen:control:9100")

#: Overtaking: `gen:control:9100` alone. Both real floors were declared, measured, and DROPPED --
#: `feasibility` could not show a pass on either, 0/4 at both friction levels, the failures being
#: collisions with the track rather than contact with the car being passed. Neither reversing the
#: reference driver's side nor cutting its lateral offset from 0.40 m to 0.25 m recovered a single
#: trial on either map, so this is not a side-of-the-track accident.
#:
#: What it is, is not settled, and the honest version matters for whether this comes back. Two cars
#: abreast do fit everywhere on both floors (narrowest lane 1.000 m and 1.082 m against 0.620 m of
#: car), so the floors are not provably too narrow for a pass; what could not be demonstrated is a
#: pass by THIS reference driver, whose 0.40 m offset already exceeds the lane over 2.0 % of
#: map16x07 and whose 7 m engage window is a fifth of that map's 33 m lap. A stronger reference
#: driver may well restore these cells. Until one exists and shows a pass, the cells stay out: a
#: scenario nobody has driven is not a scenario, and the alternative -- trimming the proof until
#: the map fits -- is the thing this suite exists to refuse.
V2_RACE_MAPS = ("gen:control:9100",)

V2_MAPS_NOTE = (
    "Held out. real:map16x07 and real:map12x16 are floors this car drove on that no training set "
    "contains, in any direction and with any obstacle suffix; the rest of the solo family is "
    "common.HELDOUT_TRACKS, guarded by tests/test_heldout_split.py. korea_2026_competition is "
    "absent by design: it is trained through twenty obstacle variants (common.KOREA26_TRAIN) and "
    "belongs to v1's in-distribution set. The overtaking family is gen:control:9100 alone: a pass "
    "on either real floor could not be shown by the scripted reference driver (0/4 at both mu "
    "levels), so those cells were dropped rather than admitted undemonstrated. Adoption decisions "
    "read v2.")


def v2() -> Suite:
    """The held-out suite. Same protocol as v1, applied to maps no training set contains.

    v1 scores `gen:control:1400` (a training map) and `real:korea_2026_competition` (whose geometry
    is trained through twenty obstacle variants), so its headline number measures how well the
    training distribution was fitted. That is worth knowing and it is not generalisation, and while
    v1 was the only suite there was no number for the other question at all.
    """
    return Suite(version="v2", solo_maps=V2_SOLO_MAPS, obstacle_maps=V2_OBSTACLE_MAPS,
                 race_maps=V2_RACE_MAPS, reused_maps_note=V2_MAPS_NOTE)


#: Every scenario set this CLI can freeze, by version.
VERSIONS = {"v1": Suite, "v2": v2}


def of(version: str) -> Suite:
    """The suite definition for a version name. Unknown names are refused rather than defaulted:
    freezing v1's maps under a file called v2 is exactly the confusion the version exists to stop."""
    try:
        factory = VERSIONS[version]
    except KeyError:
        raise ValueError(f"unknown suite version {version!r}; known: {sorted(VERSIONS)}") from None
    return factory()


#: Runtime modules whose contents can change a measurement: physics, sensing, control, the model
#: and the observation layout. Truncating these or omitting the sensor models would let a changed
#: LiDAR or dynamics pass as the same protocol.
RUNTIME_MODULES = ("sim.py", "gym_env.py", "mpc.py", "params.py", "track.py", "dynamics.py",
                   "lidar.py", "imu.py", "odom.py", "teacher.py",
                   "learn/model.py", "learn/obs.py", "learn/common.py", "learn/evaluate.py",
                   "learn/graph_runtime.py", "learn/grip_runtime.py", "learn/grip_control.py",
                   "learn/grip_estimator.py", "learn/evaluation_metrics.py",
                   # The `tcs` arm sits in the command path and its thresholds decide what the car
                   # is allowed to do; a changed guard changes a measurement as surely as a changed
                   # tyre model. `actuators.py` joins for the same reason -- it is the one line that
                   # decides whether the VESC loop closes on the wheel or on the body.
                   "learn/traction_arm.py", "actuators.py")


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
