"""Suite definition and its freeze hash.

The freeze is the whole point: scenario definitions are fixed before any system is scored, so a map,
seed or obstacle placement cannot be adjusted once outcomes are visible. `run` records the hash in
every result row and refuses a mismatch.

Geometry and the non-candidate feasibility smoke run *before* the freeze -- they involve no evaluated
checkpoint, so they cannot leak an outcome into the scenario choice.

Three scenario sets live here. `Suite()` is v1, unchanged; `v2()` is the held-out set; `v2_1()` is
v2 plus the T (traffic) family. v1 and v2 share every protocol field -- friction levels, seeds,
envs, race size, speed cap, lap budget, sensor noise, opponent -- and differ only in which maps
those fields are applied to, so a v1 row and a v2 row are comparable as measurements and differ in
the one thing the version names. v2.1 keeps every one of v2's S/A/O cells bit-identical and only
adds T, so a v2 row and a v2.1 row of the same family are the same measurement.  `of(version)`
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
    #: Which scenario within a family, for a family that has more than one. Empty for S/A/O, whose
    #: single scenario is the family, and one of `TrafficScenario.id` for T. It is part of the cell
    #: id because four T cells can otherwise share a map, a friction and a seed and differ only in
    #: what the opponent is doing -- which is the entire point of them, and which every duplicate
    #: check in the report would otherwise read as the same cell measured four times.
    variant: str = ""
    #: Cars per race, INCLUDING the learner. None means "not stated by the cell"; `declared_race_size`
    #: then falls back to the suite, which is how a hand-built O cell keeps working.
    race_size: int | None = None

    def cell_id(self) -> str:
        head = f"{self.suite}:{self.variant}" if self.variant else self.suite
        return f"{head}:{self.map_id}:{self.mu}:{self.seed}"


def declared_race_size(cell, suite=None) -> int:
    """Cars per race for a cell, from the cell if it says and from the suite if it does not.

    Every cell `Suite.cells()` builds states it. The fallback covers a `Cell` constructed by hand --
    tests and the odd tool do that -- where O has always meant the suite's `race_size` and
    everything else has always meant a solo car.
    """
    rs = getattr(cell, "race_size", None)
    if rs:
        return int(rs)
    if suite is not None and getattr(cell, "suite", None) == "O":
        return int(getattr(suite, "race_size", 1))
    return 1


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
    #: The T (traffic) family, or empty for a suite that has none. One field rather than a dozen,
    #: and hashed only when it is non-empty -- see `HASH_OMIT_WHEN_EMPTY` on `freeze_hash`.
    #: Shape: {"contention_range_m", "attack_range_m", "maps", "mus", "scenarios": [...]},
    #: each scenario {"id", "opp_speed_range", "race_size", "opp_events", "opp_event_rate", "note"}.
    traffic: dict = field(default_factory=dict)
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
                    out.append(Cell("S", m, mu, s, self.envs, race_size=1))
        for m in self.obstacle_maps:
            for mu in self.paired_mus:
                for s in self.seeds:
                    out.append(Cell("A", m, mu, s, self.envs, race_size=1))
        for m in self.race_maps:
            for mu in self.paired_mus:
                for s in self.seeds:
                    out.append(Cell("O", m, mu, s, self.envs, race_size=self.race_size))
        # T iterates scenario-outermost so the printed matrix and the run order group by what the
        # opponent is doing, which is the axis a reader of the table is comparing along.
        for sc in self.traffic_scenarios():
            for m in self.traffic.get("maps", ()):
                for mu in self.traffic.get("mus", self.paired_mus):
                    for s in self.seeds:
                        out.append(Cell("T", m, float(mu), s, self.envs, variant=str(sc["id"]),
                                        race_size=int(sc["race_size"])))
        return out

    def traffic_scenarios(self) -> tuple:
        """The declared T scenarios, or nothing at all. Never a default: a suite that does not
        declare the family does not have it, and inventing one here would put unfrozen scenarios
        into a frozen suite's cell list."""
        return tuple(self.traffic.get("scenarios", ()))

    def traffic_scenario(self, variant: str) -> dict:
        for sc in self.traffic_scenarios():
            if str(sc["id"]) == str(variant):
                return dict(sc)
        raise KeyError(f"suite {self.version} declares no traffic scenario {variant!r}; known: "
                       f"{[sc['id'] for sc in self.traffic_scenarios()]}")

    def adapter_suite(self, cell: Cell | None = None) -> dict:
        """The subset `model_adapter.prepare_cell` reads, by its contract's key names.

        With a T cell, the scenario's own opponent count, speed range and event schedule replace the
        suite-wide ones. They come from the FROZEN scenario record, not from a CLI flag: what the
        opponent does is the scenario, so a run that could change it from the command line would not
        be measuring the frozen cell.
        """
        d = {"envs": self.envs, "speed_cap": self.speed_cap, "sensor_noise": self.sensor_noise,
             "race_size": self.race_size, "budget_laps": self.budget_laps,
             # The opponent must be the fixed raceline teacher at the declared speed range. The
             # EnvConfig default is `policy`, which would make the opponent a copy of the
             # candidate -- the exact coupling the independence gate exists to rule out.
             "opponent": "teacher", "opp_speed_range": list(self.opp_speed_range),
             "opp_events": [], "opp_event_rate": 0.0}
        if cell is not None and cell.suite == "T":
            sc = self.traffic_scenario(cell.variant)
            d.update({"race_size": int(sc["race_size"]),
                      "opp_speed_range": list(sc["opp_speed_range"]),
                      "opp_events": list(sc.get("opp_events", ())),
                      "opp_event_rate": float(sc.get("opp_event_rate", 0.0))})
        return d

    #: Every family this definition can carry. A family with no cells reports 0 trials rather than
    #: disappearing, so a suite that dropped one says so instead of looking like it never had it.
    FAMILIES = ("S", "A", "O", "T")

    def expected_trials(self) -> dict:
        c = self.cells()
        fams = [f for f in self.FAMILIES if f != "T" or self.traffic_scenarios()]
        return {s: sum(x.envs for x in c if x.suite == s) for s in fams}

    def as_dict(self) -> dict:
        return asdict(self)

    #: Additive extension fields that contribute to the hash only when they are non-empty.
    #:
    #: An empty `traffic` declares no scenario, adds no cell and can change no outcome, so hashing
    #: it would have re-hashed v1 and v2 -- freezes that already exist, are published, and describe
    #: measurements taken against them -- for a change that is not one to them. `suite-v2.example.json`
    #: would then have failed `load()`, which is the exact failure the freeze exists to raise and
    #: would here have been raised by adding a feature that suite does not use. Non-empty, it is
    #: hashed in full like every other field: the scenarios ARE the suite.
    HASH_OMIT_WHEN_EMPTY = ("traffic",)

    def freeze_hash(self) -> str:
        """Stable hash over everything that can change an outcome.

        `calibration` is excluded: it records how long cells took, which is a property of the machine
        rather than of the scenario, and must not invalidate a freeze when re-measured.
        """
        d = self.as_dict()
        d.pop("calibration", None)
        d.pop("reused_maps_note", None)
        for k in self.HASH_OMIT_WHEN_EMPTY:
            if not d.get(k):
                d.pop(k, None)
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


# ------------------------------------------------------------------ suite v2.1: v2 + traffic
#
# What T measures, and why it is not O.
#
# O asks "was a pass completed and held?" and answers with a binary. That question is only
# answerable where a pass is something a driver has been shown to do, and on the two unseen real
# floors it is not: the held-out worker measured 0/4 at both friction levels with the scripted
# reference driver, so O was dropped there and v2's overtaking number comes from one generated map.
# A number from one map, 32 trials, cannot tell whether a recipe improved overtaking.
#
# The narrowness is real -- `real:map16x07` has a 0.702 m median half-width against 0.620 m of two
# cars abreast -- so the fix is not a bigger denominator on the same binary. It is a metric whose
# floor is not zero. T measures what happens in traffic:
#
#   clean traffic runs   the trial's own outcome: came through the stint with no wall collision and
#                        no contact with another car. Achievable on a 0.70 m half-lane, and the
#                        thing a car in traffic has to do before anything else counts.
#   passes / race        completed, HELD passes, counted rather than latched, and allowed to be 0.
#   pace vs opponent     the learner's arc over the opponents' arc across the same steps. This is
#                        what refuses the obvious way to game "clean": a car that hangs back and
#                        never tries scores 1.00 on cleanliness and well under 1.0 here.
#   following / attacking seconds   whether the learner engaged at all, at two ranges.
#   contact, wall collisions        counted separately, because "hit the car" and "hit the wall"
#                        are different failures with different fixes.
#
# No composite. The panel is the metric, and each column is reported with its own direction, which
# is the same rule the rest of this benchmark follows.

#: The env's own `overtake_range` (gym_env.py:92). Two cars further apart than this along the lane
#: are not racing each other, which is the definition the reward already uses.
T_CONTENTION_RANGE_M = 12.0

#: The tight window, and the reason there are two. `map16x07`'s lap is 33.2 m, so |gap| <= 12 m is
#: three quarters of every gap two cars on it can be at: the wide window saturates there and
#: separates nobody. 3.0 m is about five car lengths -- close enough that a pass is actually on --
#: and it does not saturate on any map in the family. Both are reported; neither is chosen after
#: the fact.
T_ATTACK_RANGE_M = 3.0

#: The held-out maps the traffic family runs on, exactly as the contract names them. Every one is a
#: base map of `common.HELDOUT_TRACKS`; `assert_heldout_maps` proves it before the suite is built,
#: and `tests/test_heldout_split.py` proves it again from outside.
V21_TRAFFIC_MAPS = ("real:map16x07", "real:map12x16", "gen:control:9100",
                    "real:korea_2025_iccas", "gen:competition:0")

#: Events per opponent per 10 s for the event scenario. Events do not overlap, so the realized rate
#: is this times the idle fraction. Chosen so that an event lands inside the learner's contention
#: window in essentially every trial rather than in some of them -- MEASURED, and the measurement is
#: in the report and in `calibration`, because a scenario whose defining feature only shows up half
#: the time is two scenarios sharing a name.
T_EVENT_RATE_PER_10S = 3.0

#: The four scenarios. `slow` and `pace` vary the one axis a teacher opponent has always had -- how
#: fast it is -- at the two ends the contract names. `event` and `pair` hold that axis at v2's own
#: `OPP_SPEED_RANGE` and vary behaviour and count instead, so each of those two cells differs from a
#: comparable baseline in exactly one thing.
#:
#: `weave` is deliberately absent: it is a small continuous oscillation, not a decision the learner
#: has to react to, and the contract names brake / stop / shift.
V21_TRAFFIC_SCENARIOS = (
    {"id": "slow", "opp_speed_range": [0.5, 0.7], "race_size": 2,
     "opp_events": [], "opp_event_rate": 0.0,
     "note": "a clearly slower car on the racing line: the pass is available, the question is "
             "whether it is taken and taken cleanly"},
    {"id": "pace", "opp_speed_range": [0.8, 0.95], "race_size": 2,
     "opp_events": [], "opp_event_rate": 0.0,
     "note": "a car at nearly the learner's own pace: passes are rare by construction, so this "
             "cell is carried by pace-vs-opponent and by whether the learner stays clean while "
             "spending a whole stint in close company"},
    {"id": "event", "opp_speed_range": list(OPP_SPEED_RANGE), "race_size": 2,
     "opp_events": ["brake", "stop", "shift"], "opp_event_rate": T_EVENT_RATE_PER_10S,
     "note": "the scripted behaviours of f1sim.opponent_events: a car that brakes, a car that has "
             "stopped, a car that moves across the lane. Nothing else in the benchmark evaluates "
             "against them"},
    {"id": "pair", "opp_speed_range": list(OPP_SPEED_RANGE), "race_size": 3,
     "opp_events": [], "opp_event_rate": 0.0,
     "note": "two opponents. A second car removes the assumption that the lane beside the car "
             "ahead is empty, which every one-opponent cell quietly grants"},
)

V21_TRAFFIC = {
    "contention_range_m": T_CONTENTION_RANGE_M,
    "attack_range_m": T_ATTACK_RANGE_M,
    "maps": list(V21_TRAFFIC_MAPS),
    "mus": [MU_LOW, MU_MID],
    "scenarios": [dict(sc) for sc in V21_TRAFFIC_SCENARIOS],
}

V21_MAPS_NOTE = (
    V2_MAPS_NOTE.replace("Adoption decisions read v2.", "").strip() + " The traffic family T adds "
    "five held-out maps -- real:map16x07, real:map12x16, gen:control:9100, real:korea_2025_iccas, "
    "gen:competition:0 -- under four opponent scenarios (slower, at pace, brake/stop/shift events, "
    "two opponents). T does NOT ask whether a pass was completed, because on the two real floors no "
    "reference driver has completed one and a binary that reads zero everywhere measures nothing; "
    "it reports clean traffic runs, passes per race, pace against the opponent, engagement time and "
    "contacts, each with its own direction and none combined. Adoption decisions read v2.1; its "
    "S/A/O cells are v2's, unchanged.")


def assert_heldout_maps(maps, *, where: str = "suite") -> list:
    """Every map is a base map of `common.HELDOUT_TRACKS`, or this raises.

    The guard the T family needs and the reason it is a guard rather than a convention: T exists to
    put a *real-floor* traffic number in the benchmark, and a real-floor number measured on a floor
    something trained on is not one. Comparison is by base map through `common.base_map`, so a
    `+obs`/`~rev`/`~mir` variant of a training venue cannot slip in under a different string.

    Raises rather than warning, and raises on an import failure too: a leakage guard that quietly
    does not run is worse than none, because the suite then carries a claim nobody checked.
    """
    try:
        from f1sim.learn import common
    except Exception as exc:                       # pragma: no cover - environment dependent
        raise RuntimeError(
            f"cannot check the held-out split: f1sim.learn.common did not import ({exc}). The T "
            f"family claims its maps are held out; refusing to build it unchecked.") from exc
    held = {common.base_map(n) for n in common.HELDOUT_TRACKS}
    leaked = [m for m in maps if common.base_map(m) not in held]
    if leaked:
        raise ValueError(
            f"{where}: {leaked} are not held-out base maps. A traffic score on a floor that appears "
            f"in any training list, in any direction and with any obstacle suffix, is not a "
            f"generalisation number. Known held-out base maps: {sorted(held)}")
    return list(maps)


def v2_1() -> Suite:
    """v2 plus the T (traffic) family. Every S, A and O cell of v2 is unchanged.

    The freeze hash differs from v2's -- it has to, because the scenario set differs -- but the
    cells the two share are the same cells, so a v2 avoidance row and a v2.1 avoidance row measure
    the same thing and only the traffic family is new.
    """
    s = v2()
    assert_heldout_maps(V21_TRAFFIC["maps"], where="suite v2.1 traffic family")
    s.version = "v2.1"
    s.traffic = {k: (list(v) if isinstance(v, list) else v) for k, v in V21_TRAFFIC.items()}
    s.reused_maps_note = V21_MAPS_NOTE
    return s


#: Every scenario set this CLI can freeze, by version.
VERSIONS = {"v1": Suite, "v2": v2, "v2.1": v2_1}


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
                   "learn/traction_arm.py", "actuators.py",
                   # The published baselines' evaluation path. These files decide what an `external`
                   # roster entry actually does with a scan -- which beams it reads, what fills the
                   # bearings this car cannot see, how the output becomes a command -- so a change
                   # to one of them changes a measurement exactly as a changed tyre model does.
                   # `learn/baselines/distill.py` and `__main__.py` are deliberately NOT here: they
                   # train a model and never run inside a scored cell, and listing a file whose
                   # edits cannot move a number would invalidate every existing result for nothing.
                   "learn/baselines/__init__.py", "learn/baselines/common.py",
                   "learn/baselines/backends.py", "learn/baselines/tinylidarnet.py",
                   "learn/baselines/tinylidarnet_torch.py", "learn/baselines/end2race.py")


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
