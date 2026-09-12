"""The track catalogue: short ids for the base maps, and a grammar for the scenario driven on one.

Why this module exists
----------------------
The catalogue the user saw was the loader's own strings, and the loader's strings bake every choice
into one name: `real:korea_2026_competition+rlobs213~mir~rev` is a map, a direction, a mirroring, an
obstacle family and a placement seed, glued together with punctuation that only `maps.load` reads.
Two hundred of those in one list is not a list of maps -- it is a list of *runs*, and picking a map
out of it means reading the same forty-character prefix over and over.

So there are two names now, and they say different things:

* a **track** is a base map -- a floor, a circuit, a generated layout. It has a short canonical id
  (`real/korea26`), a Korean display name (`Korea 2026`), and nothing else in it. This is what a
  list shows and what a user picks.
* a **scenario** is a track plus the three choices that used to be baked into the name: direction,
  obstacle family, obstacle seed. `real/korea26@rev#line:213` is one, and it is *built* by the
  picker's three little controls rather than found in a list.

`legacy()` turns a scenario back into the loader's string, so nothing downstream changes meaning:
`tracks.resolve("real/korea26@mir+rev#line:213") == "real:korea_2026_competition+rlobs213~mir~rev"`.
`parse()` reads either grammar, which is what lets a frozen benchmark suite, an old manifest and a
new picker all talk about the same thing.

Torch-free on purpose, like `console/catalog.py`: the console process must be able to name and
format a scenario without paying for `import torch` (`maps.py` pulls it in through `track.py`).
Everything here is string and path arithmetic; the only filesystem access is the directory listing
that discovers racetracks, gym maps and editor scenes.
"""
from __future__ import annotations

import glob
import os
import random
import re
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # <repo>/f1sim
EXTERNAL = os.path.normpath(os.path.join(_HERE, "..", "external"))
RACETRACKS = os.path.join(EXTERNAL, "f1tenth_racetracks")
GYM_MAPS = os.path.join(EXTERNAL, "f1tenth_gym", "gym", "f110_gym", "envs", "maps")

#: The families an id can start with. `real` and `rt` are measured floors and circuits, `gen` is
#: the procedural generator, `gym` the f1tenth_gym drawn maps, `scene` the environment editor's own.
FAMILIES = ("real", "rt", "gen", "gym", "scene")

#: Family chip text in the UI. Short by design: it sits next to the id in a narrow list.
FAMILY_LABEL = {"real": "실측", "rt": "레이스트랙", "gen": "생성", "gym": "gym", "scene": "에디터"}

# ---------------------------------------------------------------- the two option axes
#: Direction of travel. The key is what goes after `@`; `""` is the map as recorded.
#: `mir+rev` is one choice, not two, because the loader applies mirror-then-reverse and the
#: opposite order is a different track.
DIRECTIONS: Tuple[str, ...] = ("", "rev", "mir", "mir+rev")
DIRECTION_LABEL = {"": "정방향", "rev": "역방향", "mir": "거울", "mir+rev": "거울+역방향"}

#: Obstacle families. The key is what goes after `#`; `""` is a clean lap.
#: The names are what the obstacles *do*, which is the thing a user is choosing between --
#: `+obs` vs `+rlobs` said which function stamped them.
OBSTACLES: Tuple[str, ...] = ("", "edge", "line", "pinch", "props")
OBSTACLE_LABEL = {"": "없음", "edge": "가장자리", "line": "주행선 위", "pinch": "좁아짐", "props": "입체"}
OBSTACLE_HINT = {
    "": "장애물 없이 빈 트랙을 그대로 달립니다.",
    "edge": "차선 가장자리에 상자를 세웁니다. 주행선은 비어 있습니다 (옛 이름 +obs).",
    "line": "주행선 위에 상자를 세웁니다. 피해서 계획해야 합니다 (옛 이름 +rlobs).",
    "pinch": "몇 군데에서 차선 폭을 좁힙니다 (옛 이름 +pinch).",
    "props": "상자·궤짝·드럼을 입체로 세웁니다. 점유 격자가 아니라 유한한 볼록 단면입니다 (옛 이름 +props).",
}
#: new name -> the loader's suffix. The loader is not renamed: every checkpoint, manifest and
#: frozen benchmark file in this repository names its tracks in the old grammar.
LEGACY_OBSTACLE = {"edge": "obs", "line": "rlobs", "pinch": "pinch", "props": "props"}
OBSTACLE_FROM_LEGACY = {v: k for k, v in LEGACY_OBSTACLE.items()}

#: Which obstacle families a family of tracks can actually carry, from `maps._load_base`:
#: racetracks and editor scenes only understand `+props`.
OBSTACLES_BY_FAMILY = {
    "real": ("", "edge", "line", "pinch", "props"),
    "gen": ("", "edge", "line", "pinch", "props"),
    "rt": ("", "props"),
    "scene": ("", "props"),
    "gym": ("",),
}


# ---------------------------------------------------------------- the registry
@dataclass(frozen=True)
class TrackEntry:
    """One base map. `legacy` is the exact string `maps.load` already accepts for it."""
    id: str
    family: str
    display: str
    legacy: str
    source: str              # bag | racetracks | gym | generator | editor
    note: str = ""

    @property
    def slug(self) -> str:
        return self.id.split("/", 1)[1]

    @property
    def family_label(self) -> str:
        return FAMILY_LABEL.get(self.family, self.family)

    def obstacle_options(self) -> Tuple[str, ...]:
        return OBSTACLES_BY_FAMILY.get(self.family, ("",))


#: The measured floors and circuits, in catalogue order. `note` is the one-line description
#: `maps.REAL` already carries; `test_tracks.py` asserts the two agree so they cannot drift.
_REAL: Tuple[Tuple[str, str, str, str], ...] = (
    # (slug, legacy name, display, note)
    ("icra22", "icra2022", "ICRA 2022",
     "ICRA 2022 F1TENTH Grand Prix track (Philadelphia), team SLAM map"),
    ("bb21-1", "blackbox2021_1", "Blackbox 2021 #1", "TU Wien BlackBox race 2021"),
    ("bb21-2", "blackbox2021_2", "Blackbox 2021 #2", "TU Wien BlackBox race 2021"),
    ("bb21-3", "blackbox2021_3", "Blackbox 2021 #3", "TU Wien BlackBox race 2021"),
    ("bb22-1", "blackbox2022_1", "Blackbox 2022 #1", "TU Wien BlackBox race 2022"),
    ("bb22-2", "blackbox2022_2", "Blackbox 2022 #2", "TU Wien BlackBox race 2022"),
    ("bb22-3", "blackbox2022_3", "Blackbox 2022 #3", "TU Wien BlackBox race 2022"),
    ("korea26", "korea_2026_competition", "Korea 2026",
     "2026 competition venue, from the team's own recordings"),
    ("lab16x07", "map16x07", "실측 16×7 m",
     "pre-competition floor, 15.5 x 7.0 m hairpin loop, from the team's own recordings"),
    ("lab12x16", "map12x16", "실측 12×16 m",
     "pre-competition floor, 12.3 x 16.4 m loop, from the team's own recordings"),
    ("iccas25", "korea_2025_iccas", "ICCAS 2025",
     "4th F1TENTH Korea Championship 2025 (ICCAS) race track, team SLAM map"),
)

#: The racetrack directories of `external/f1tenth_racetracks`, so an id resolves with the submodule
#: absent (tests, a console that has not seen the worker yet). Discovery adds anything new.
_RT_KNOWN: Tuple[str, ...] = (
    "Austin", "BrandsHatch", "Budapest", "Catalunya", "Hockenheim", "IMS", "Melbourne",
    "Mexico City", "Montreal", "Monza", "MoscowRaceway", "Nuerburgring", "Oschersleben", "Sakhir",
    "SaoPaulo", "Sepang", "Shanghai", "Silverstone", "Sochi", "Spa", "Spielberg", "YasMarina",
    "Zandvoort",
)
#: Slugs stay at 12 characters or fewer, so the id column of a narrow list never elides.
#: Only the names that do not fit the lowercase-alphanumeric rule need an entry here.
_RT_SLUG_OVERRIDE = {"MoscowRaceway": "moscow"}
_GYM_SLUG_OVERRIDE = {"stata_basement": "stata"}

#: Generator style <-> id alias. The seed is part of the slug for `gen` because a generated map has
#: no other identity: `gen/control-1400` *is* seed 1400 of the control style.
GEN_STYLE_ALIAS = {"competition": "comp", "control": "control", "circuit": "circuit",
                   "hallway": "hall", "serpentine": "serp"}
GEN_ALIAS_STYLE = {v: k for k, v in GEN_STYLE_ALIAS.items()}
#: Shown in the starter catalogue, so every generator style is reachable without typing a seed.
GEN_STARTER_SEEDS = (0, 1, 2, 3)


class TrackError(ValueError):
    """A spec that names no track, or names one option that cannot be honoured."""


def _rt_slug(name: str) -> str:
    return _RT_SLUG_OVERRIDE.get(name) or re.sub(r"[^a-z0-9]", "", name.lower())


def _gym_slug(name: str) -> str:
    return _GYM_SLUG_OVERRIDE.get(name, name)


def racetrack_dirs() -> Tuple[str, ...]:
    """Racetrack directory names: whatever is on disk, else the known list."""
    found = sorted(os.path.basename(os.path.dirname(y))
                   for y in glob.glob(os.path.join(RACETRACKS, "*", "*_map.yaml")))
    if not found:
        return _RT_KNOWN
    return tuple(dict.fromkeys(list(found) + [n for n in _RT_KNOWN if n not in found]))


def gym_map_names() -> Tuple[str, ...]:
    found = sorted(os.path.splitext(os.path.basename(y))[0]
                   for y in glob.glob(os.path.join(GYM_MAPS, "*.yaml")))
    return tuple(found) if found else ("berlin", "levine", "skirk", "stata_basement", "vegas")


def scene_names() -> Tuple[str, ...]:
    """Environment-editor scenes, newest first. `f1sim.scene` is torch-free."""
    try:
        from .scene import list_scenes
        return tuple(s["name"] for s in list_scenes())
    except Exception:
        return ()


def _real_entries() -> List[TrackEntry]:
    return [TrackEntry(id=f"real/{slug}", family="real", display=display, legacy=f"real:{legacy}",
                       source="bag", note=note)
            for slug, legacy, display, note in _REAL]


def _rt_entry(name: str) -> TrackEntry:
    return TrackEntry(id=f"rt/{_rt_slug(name)}", family="rt", display=name, legacy=f"rt:{name}",
                      source="racetracks", note="f1tenth_racetracks, 1:10 scale of a real circuit")


def _gym_entry(name: str) -> TrackEntry:
    return TrackEntry(id=f"gym/{_gym_slug(name)}", family="gym", display=name.replace("_", " "),
                      legacy=f"gym:{name}", source="gym", note="f1tenth_gym SLAM/drawn map")


def _gen_entry(style: str, seed: int) -> TrackEntry:
    alias = GEN_STYLE_ALIAS[style]
    return TrackEntry(id=f"gen/{alias}-{seed}", family="gen", display=f"생성 {style} {seed}",
                      legacy=f"gen:{style}:{seed}", source="generator",
                      note=f"절차 생성 {style} 스타일, 시드 {seed}")


def _scene_entry(name: str) -> TrackEntry:
    return TrackEntry(id=f"scene/{name}", family="scene", display=f"{name} (에디터)",
                      legacy=f"scene:{name}", source="editor",
                      note="환경 에디터에서 만든 장면")


#: Static part of the registry: the measured floors, keyed by id. Everything else is derived by
#: rule, because `rt`, `gym` and `scene` are directory listings and `gen` is an infinite family.
REGISTRY: Dict[str, TrackEntry] = {e.id: e for e in _real_entries()}
_LEGACY_TO_ID: Dict[str, str] = {e.legacy: e.id for e in REGISTRY.values()}


def get(track_id: str) -> TrackEntry:
    """The entry for a canonical id. Derives `rt` / `gym` / `gen` / `scene` ids on the spot."""
    track_id = (track_id or "").strip()
    if track_id in REGISTRY:
        return REGISTRY[track_id]
    family, _, slug = track_id.partition("/")
    if not slug:
        raise TrackError(f"트랙 id 가 아닙니다: {track_id!r} (예: real/bb22-1)")
    if family == "rt":
        for name in racetrack_dirs():
            if _rt_slug(name) == slug:
                return _rt_entry(name)
        raise TrackError(f"레이스트랙을 찾지 못했습니다: {track_id!r}")
    if family == "gym":
        for name in gym_map_names():
            if _gym_slug(name) == slug:
                return _gym_entry(name)
        raise TrackError(f"gym 맵을 찾지 못했습니다: {track_id!r}")
    if family == "gen":
        alias, _, seed = slug.rpartition("-")
        if alias in GEN_ALIAS_STYLE and seed.isdigit():
            return _gen_entry(GEN_ALIAS_STYLE[alias], int(seed))
        raise TrackError(f"생성 트랙 id 를 읽지 못했습니다: {track_id!r} "
                         f"(예: gen/control-1400, 스타일 {'/'.join(sorted(GEN_ALIAS_STYLE))})")
    if family == "scene":
        return _scene_entry(slug)
    if family == "real":
        raise TrackError(f"실측 맵을 찾지 못했습니다: {track_id!r} "
                         f"(있는 것: {', '.join(sorted(e.id for e in REGISTRY.values()))})")
    raise TrackError(f"알 수 없는 트랙 계열입니다: {family!r} ({'/'.join(FAMILIES)})")


def id_of_legacy(legacy: str) -> Optional[str]:
    """The canonical id for a *base* legacy name, or None when it names no catalogue track."""
    legacy = (legacy or "").strip()
    if legacy in _LEGACY_TO_ID:
        return _LEGACY_TO_ID[legacy]
    family, _, rest = legacy.partition(":")
    if not rest:
        return None
    if family == "rt":
        return f"rt/{_rt_slug(rest)}" if rest in racetrack_dirs() else None
    if family == "gym":
        return f"gym/{_gym_slug(rest)}"
    if family == "scene":
        return None if rest.startswith("/") else f"scene/{rest}"
    if family == "gen":
        style, _, seed = rest.partition(":")
        if style in GEN_STYLE_ALIAS and seed.isdigit():
            return f"gen/{GEN_STYLE_ALIAS[style]}-{int(seed)}"
    return None


def catalog(scenes: bool = True, gen_starters: bool = True,
            extra: Sequence[str] = ()) -> List[TrackEntry]:
    """Every base track worth offering, in list order: measured floors, circuits, generated,
    gym, and the editor's own scenes. `extra` pins ids that must appear (the split members)."""
    out: List[TrackEntry] = list(REGISTRY.values())
    out += [_rt_entry(n) for n in racetrack_dirs()]
    gen_ids: List[str] = []
    if gen_starters:
        gen_ids += [f"gen/{GEN_STYLE_ALIAS[s]}-{i}"
                    for s in ("competition", "control", "circuit", "hallway", "serpentine")
                    for i in GEN_STARTER_SEEDS]
    gen_ids += [i for i in extra if i.startswith("gen/")]
    seen = {e.id for e in out}
    for tid in gen_ids:
        if tid not in seen:
            seen.add(tid)
            out.append(get(tid))
    out += [_gym_entry(n) for n in gym_map_names()]
    for tid in extra:
        if tid not in seen and not tid.startswith("gen/"):
            seen.add(tid)
            try:
                out.append(get(tid))
            except TrackError:
                continue
    if scenes:
        out += [_scene_entry(n) for n in scene_names()]
    return out


# ---------------------------------------------------------------- the scenario grammar
_SPEC = re.compile(r"^(?P<track>[A-Za-z0-9_\-./]+?)(?:@(?P<dir>[a-z+]+))?(?:#(?P<obs>[a-z]+):(?P<seed>\*|\d+))?$")


@dataclass(frozen=True)
class Scenario:
    """A track plus the three choices that used to be glued into its name.

    `seed is None` with an obstacle set means "무작위": a concrete seed has not been drawn yet.
    `legacy()` refuses to produce a loader name in that state rather than inventing one, because a
    silently invented seed is exactly the thing that makes two runs look like the same track.
    """
    track: str = ""
    reverse: bool = False
    mirror: bool = False
    obstacle: str = ""
    seed: Optional[int] = None
    #: Set when the name is not one this module can spell. `legacy()` and `short()` then return it
    #: verbatim -- rebuilding a string we do not understand is how a name silently becomes a
    #: different map. `track` may still be filled in: an *unknown obstacle family on a known map*
    #: (another branch is training with `+hard<seed>`) is still that map, and a list of them should
    #: group and count as that map rather than as one new map per entry.
    raw: str = ""

    # -- the option axes
    @property
    def direction(self) -> str:
        return ("mir+rev" if self.reverse else "mir") if self.mirror else ("rev" if self.reverse else "")

    @property
    def random_seed(self) -> bool:
        return bool(self.obstacle) and self.seed is None

    @property
    def entry(self) -> TrackEntry:
        return get(self.track)

    def with_seed(self, seed: int) -> "Scenario":
        return replace(self, seed=int(seed))

    def with_options(self, direction: Optional[str] = None, obstacle: Optional[str] = None,
                     seed: Optional[int] = None, random: bool = False) -> "Scenario":
        """The picker's edit: change one axis and keep the rest."""
        sc = self
        if direction is not None:
            if direction not in DIRECTIONS:
                raise TrackError(f"방향을 알 수 없습니다: {direction!r} ({', '.join(d or '정방향' for d in DIRECTIONS)})")
            sc = replace(sc, mirror="mir" in direction, reverse="rev" in direction)
        if obstacle is not None:
            if obstacle not in OBSTACLES:
                raise TrackError(f"장애물 종류를 알 수 없습니다: {obstacle!r} ({', '.join(o or '없음' for o in OBSTACLES)})")
            sc = replace(sc, obstacle=obstacle, seed=None if not obstacle else sc.seed)
        if random:
            sc = replace(sc, seed=None)
        elif seed is not None:
            sc = replace(sc, seed=int(seed))
        return sc

    # -- the two spellings
    def short(self) -> str:
        """The new grammar: `real/bb22-1@mir+rev#line:44`. What every user-facing list shows."""
        if self.raw:
            return self.raw
        s = self.track
        if self.direction:
            s += f"@{self.direction}"
        if self.obstacle:
            s += f"#{self.obstacle}:{'*' if self.seed is None else self.seed}"
        return s

    def legacy(self) -> str:
        """The loader's grammar: `real:blackbox2022_1+rlobs44~mir~rev`. Suffix order is the
        loader's (`maps._split_obstacle_suffix` then `maps.MODIFIERS`), not a preference."""
        if self.raw:
            return self.raw
        s = self.entry.legacy
        if self.obstacle:
            if self.seed is None:
                raise TrackError(f"{self.short()}: 시드가 무작위입니다. expand()/with_seed() 로 "
                                 f"먼저 하나 뽑아야 로더 이름이 됩니다.")
            s += f"+{LEGACY_OBSTACLE[self.obstacle]}{int(self.seed)}"
        if self.mirror:
            s += "~mir"
        if self.reverse:
            s += "~rev"
        return s

    def display(self) -> str:
        """One human line: `Blackbox 2022 #1 · 역방향 · 주행선 위 (시드 44)`."""
        if self.raw:
            # A known map with a suffix we cannot name: say the map, then the suffix as it stands.
            if self.track:
                return f"{self.entry.display} · {self.raw[len(self.entry.legacy):].lstrip('+')}"
            return self.raw
        parts = [self.entry.display]
        if self.direction:
            parts.append(DIRECTION_LABEL[self.direction])
        if self.obstacle:
            seed = "무작위" if self.seed is None else f"시드 {self.seed}"
            parts.append(f"{OBSTACLE_LABEL[self.obstacle]} ({seed})")
        return " · ".join(parts)

    def validate(self) -> "Scenario":
        """Raise when the track cannot carry the obstacle family asked for."""
        if self.raw or not self.obstacle:
            return self
        allowed = self.entry.obstacle_options()
        if self.obstacle not in allowed:
            raise TrackError(
                f"{self.entry.display} ({self.track}) 은 '{OBSTACLE_LABEL[self.obstacle]}' 장애물을 "
                f"지원하지 않습니다. 가능한 것: "
                f"{', '.join(OBSTACLE_LABEL[o] for o in allowed)}")
        return self


def is_spec(spec: str) -> bool:
    """True when `spec` is written in the new grammar (`<family>/...`)."""
    s = (spec or "").strip()
    return bool(s) and not s.startswith("/") and s.split("/", 1)[0] in FAMILIES and "/" in s


def parse(spec: str) -> Scenario:
    """Read either grammar. Anything that is neither (an absolute map path, an unknown name) comes
    back as a passthrough `Scenario` whose `legacy()` and `short()` are the string itself, so a
    caller can hand any name to this module without first asking what kind it is."""
    s = (spec or "").strip()
    if not s:
        raise TrackError("빈 트랙 이름입니다.")
    if is_spec(s):
        return _parse_spec(s).validate()
    return _parse_legacy(s)


def _parse_spec(s: str) -> Scenario:
    m = _SPEC.match(s)
    if not m:
        raise TrackError(f"시나리오 문법을 읽지 못했습니다: {s!r} "
                         f"(형식: <트랙>[@방향][#장애물:시드], 예 real/bb22-1@rev#line:44)")
    track = m.group("track")
    get(track)                                              # raises when the track is unknown
    direction = m.group("dir") or ""
    if direction not in DIRECTIONS:
        raise TrackError(f"방향을 알 수 없습니다: {direction!r} "
                         f"({', '.join(d for d in DIRECTIONS if d)})")
    obstacle = m.group("obs") or ""
    if obstacle and obstacle not in OBSTACLES:
        raise TrackError(f"장애물 종류를 알 수 없습니다: {obstacle!r} "
                         f"({', '.join(o for o in OBSTACLES if o)})")
    raw_seed = m.group("seed")
    seed = None if (raw_seed is None or raw_seed == "*") else int(raw_seed)
    return Scenario(track=track, mirror="mir" in direction, reverse="rev" in direction,
                    obstacle=obstacle, seed=seed)


#: The loader's modifier suffixes, in the order `maps.load` peels them.
_MODIFIERS = ("~rev", "~mir")
#: `<base>+<something><digits>` where `<something>` is an obstacle family this version does not
#: know. Deliberately permissive: the point is to recognise the *map*, not the suffix.
_UNKNOWN_SUFFIX = re.compile(r"^(?P<base>.+?)\+(?P<kind>[a-z_]+)(?P<seed>\d*)$")


def _parse_legacy(s: str) -> Scenario:
    name = s
    reverse = mirror = False
    while name.endswith(_MODIFIERS):
        if name.endswith("~rev"):
            reverse, name = True, name[:-4]
        elif name.endswith("~mir"):
            mirror, name = True, name[:-4]
    obstacle, seed = "", None
    for tag in ("+rlobs", "+obs", "+pinch", "+props"):      # `+rlobs` first: see maps._split_obstacle_suffix
        if tag in name:
            base, _, spec = name.partition(tag)
            obstacle = OBSTACLE_FROM_LEGACY[tag[1:]]
            seed = int(spec) if spec.isdigit() else None
            name = base
            break
    tid = id_of_legacy(name)
    if tid is not None:
        return Scenario(track=tid, reverse=reverse, mirror=mirror, obstacle=obstacle, seed=seed)
    # Not a name this module can spell. Before giving up on it entirely, see whether it is a known
    # map carrying an obstacle family we have not heard of -- `real:icra2022+hard1~rev`. Naming the
    # map is most of what a caller wanted; the string still comes back verbatim.
    m = _UNKNOWN_SUFFIX.match(name)
    if m:
        tid = id_of_legacy(m.group("base"))
        if tid is not None:
            return Scenario(track=tid, raw=s)
    return Scenario(raw=s)


def resolve(spec: str) -> str:
    """`spec` (either grammar) -> the string `maps.load` accepts. Identity on legacy names."""
    return parse(spec).legacy()


def short(spec: str) -> str:
    """`spec` (either grammar) -> the short grammar. What a user-facing list prints."""
    return parse(spec).short()


def display(spec: str) -> str:
    return parse(spec).display()


# ---------------------------------------------------------------- random seeds
#: Obstacle seeds are drawn from this range. Four digits, so a seed printed in the facts strip or a
#: manifest is short enough to read out loud and type back in as `#line:4417`.
SEED_RANGE = (1000, 9999)


def draw_seed(rng: random.Random) -> int:
    return rng.randint(*SEED_RANGE)


def expand(spec: str, n: int = 1, rng: Optional[random.Random] = None) -> List[str]:
    """`spec` -> `n` concrete legacy names.

    A scenario with a fixed seed (or no obstacle at all) is one track however large `n` is: there
    is nothing to draw. `#line:*` is the interesting case -- it means "this map, with obstacles,
    any placement" -- and it becomes `n` distinct rasterised variants, drawn from `rng` so the same
    run seed gives the same set and a manifest can record the concrete names.
    """
    sc = parse(spec)
    if not sc.random_seed:
        return [sc.legacy()]
    rng = rng or random.Random(0)
    seeds: List[int] = []
    seen = set()
    for _ in range(max(1, int(n)) * 20):
        if len(seeds) >= max(1, int(n)):
            break
        s = draw_seed(rng)
        if s not in seen:
            seen.add(s)
            seeds.append(s)
    return [sc.with_seed(s).legacy() for s in seeds]


def expand_all(specs: Iterable[str], n: int = 1, seed: int = 0) -> List[str]:
    """`expand` over a list, one `random.Random(seed)` shared by the whole list so the draw depends
    on the run seed and the order of the list, and on nothing else."""
    rng = random.Random(seed)
    out: List[str] = []
    for s in specs:
        out.extend(expand(s, n, rng))
    return out


# ================================================================ the three splits
# What a split *is* has to be readable, because the only thing a user can do with a group label is
# believe it. The old console offered five groups ("기본 평가셋", "장애물", "기본 학습셋", "이전 실험
# 재현", "전체 카탈로그") whose names described where the list came from rather than what it means,
# and two of them were the same maps with different obstacles stamped in.
#
# There are three, and the line between them is one question: has the policy seen this floor?
#
#   학습 (train)    the maps a policy trained on, in every direction and with every obstacle family
#                   that was trained against. A score here says how well a known circuit was learned.
#   검증 (held-out) maps that have never been used for training in any form. The generalisation
#                   number comes from here and nowhere else.
#   내 환경         whatever the user built in the environment editor (`scene:<name>`).
#
# The direction and obstacle variants are *not* groups any more. They are the two little controls
# next to the map list, and the split a variant belongs to is the split its base track belongs to.

@dataclass(frozen=True)
class SplitRule:
    """One base track and the variant policy applied to it.

    Expansion order is `for seed in seeds for direction in directions`, which is the order the
    literal lists in `learn/common.py` were written in; `tests/test_tracks.py` holds those lists as
    a frozen oracle and compares string for string, so the order is part of the contract rather
    than an accident of how this loop is written.
    """
    track: str
    directions: Tuple[str, ...] = ("",)
    obstacle: str = ""
    seeds: Tuple[Optional[int], ...] = (None,)

    def expand(self) -> List[Scenario]:
        seeds = self.seeds if self.obstacle else (None,)
        return [Scenario(track=self.track, mirror="mir" in d, reverse="rev" in d,
                         obstacle=self.obstacle, seed=s)
                for s in seeds for d in self.directions]

    def legacy_names(self) -> List[str]:
        return [sc.legacy() for sc in self.expand()]


def _rules(tracks_: Sequence[str], directions: Tuple[str, ...] = ("",), obstacle: str = "",
           seeds=(None,)) -> List[SplitRule]:
    """One rule per track, same policy on each -- the shape most of the lists below have."""
    return [SplitRule(t, directions, obstacle, tuple(seeds)) for t in tracks_]


#: Every direction. A policy that has driven a floor one way round has not driven the mirror of it,
#: and the four are cheap: one loaded grid, four `Track` views (`maps.load` shares the base).
ALL_DIRECTIONS: Tuple[str, ...] = ("", "rev", "mir", "mir+rev")
BOTH_DIRECTIONS: Tuple[str, ...] = ("", "rev")

#: The measured venues a policy trains on. `real/bb22-3` is deliberately absent -- it is the pinched
#: real venue the held-out split probes -- and so is `real/iccas25`.
REAL_TRAIN_IDS = ("real/icra22", "real/bb21-1", "real/bb21-2", "real/bb21-3",
                  "real/bb22-1", "real/bb22-2")
RT_TRAIN_IDS = ("rt/spielberg", "rt/oschersleben")

# The procedural mix is chosen from what was measured to be missing, not from how many seeds a
# generator can produce. Against the real venues the old set had three holes:
#
#   fold-back   how close the lap comes to itself while being far away along it: every generator
#               stayed 5.8-7.0 m away, every real venue folds to 2.4-3.3 m behind a single hose.
#               `serpentine` was written for this and lands at 1.3-2.5 m.
#   pinch       local width over the narrowest spot near it: 1.04-1.05 procedurally against 1.35-1.98
#               on the blackbox maps. `#pinch` closes the lane down at a few places on any style.
#   width var   0.07-0.15 against 0.26. `control` (hairpins, chicanes, varying width) was in the
#               code and in no track set at all.
#
# `competition` is cut back rather than grown: its own docstring says more seeds add little new
# geometry, and it was 32 of 112 tracks. `hallway` is cut hardest -- its scans are so self-similar
# (aliasing 0.087 against 0.28-0.58 everywhere else) that far-apart places are indistinguishable to
# a LiDAR-only policy, which is teaching one observation two answers.
# `serpentine` is written and measured but held back from training for now: it delivers the fold-back
# (1.3-2.5 m against 5.8-7.0 for every other generator) and with the loop seam rounded the teacher's
# episode failures dropped tenfold, but it still runs at 18 collisions/km against 0.2-1.4 elsewhere.
# The residual cause is not the generator: two lanes 3 m apart make the centerline projection
# ambiguous, and progress, lap counting and the wrong-way check all read that projection. Windowing
# the search around the previous index fixed the *real* folded venues (korea_2026 went to 0.35
# collisions/km, s no longer jumping) but not a track that folds this often. Fold-back exposure comes
# from those real maps until the projection is solid enough to carry procedural folds too.
GEN_TRAIN_IDS = ([f"gen/control-{s}" for s in range(1400, 1408)]         # hairpins, chicanes, width
                 + [f"gen/comp-{s}" for s in range(1000, 1008)]
                 + [f"gen/circuit-{s}" for s in range(1200, 1204)]
                 + [f"gen/hall-{s}" for s in range(1100, 1104)])

#: The venue this car actually raced at. TRAINING ONLY, and it must stay that way: twenty-five
#: obstacle variants of this floor are trained on, so its geometry, its folds and its sight lines
#: are all in the weights, and a score on it -- with or without a box it has not seen -- says how
#: well a known circuit was learned. The held-out split holds the floors that answer the other
#: question.
KOREA26_ID = "real/korea26"

# Obstacles are half the point of the exercise, so they get their own share of the set rather than
# one variant per real map. `#edge` sits boxes against a lane edge; `#line` puts them ON the racing
# line, which is the case that actually has to be avoided -- the teacher itself goes from 0.22 to
# 1.67 collisions/km on those, and there were none in any track set.
TRAIN_RULES: List[SplitRule] = (
    _rules(REAL_TRAIN_IDS, ALL_DIRECTIONS)
    + _rules(RT_TRAIN_IDS, ALL_DIRECTIONS)
    + _rules(GEN_TRAIN_IDS, BOTH_DIRECTIONS)
    # the lane closing down, on the styles that have the width to lose
    + [SplitRule(f"gen/control-{s}", ("",), "pinch", (s,)) for s in range(1408, 1414)]
    + [SplitRule(f"gen/comp-{s}", ("",), "pinch", (s,)) for s in range(1008, 1012)]
    # the raced venue, every obstacle family, both directions and mirrored
    + [SplitRule(KOREA26_ID, ("", "rev", "mir"), "edge", (201, 202, 203)),
       SplitRule(KOREA26_ID, ("", "rev", "mir"), "line", (211, 212, 213, 214)),
       SplitRule(KOREA26_ID, ("", "rev"), "pinch", (221, 222))]
    + [SplitRule(t, BOTH_DIRECTIONS, "edge", (i,)) for i, t in enumerate(REAL_TRAIN_IDS)]
    + [SplitRule(t, BOTH_DIRECTIONS, "line", (i + 40,)) for i, t in enumerate(REAL_TRAIN_IDS)]
    + [SplitRule(f"gen/control-{s}", ("",), "line", (s,)) for s in range(1414, 1420)]
    + [SplitRule(f"gen/comp-{s}", ("",), "line", (s,)) for s in range(1012, 1016)]
)

# Held out entirely. Grouped by the axis each one probes, so a failure says which kind of novelty
# broke it rather than only that something did.
#
#   real/lab16x07   15.5 x 7.0 m, the pre-competition hairpin loop. Median half-width 0.702 m,
#                   the narrowest median lane in the catalog: the tightest training venue by that
#                   measure is real/icra22 at 0.873 m, and the raced floor is 0.960 m.
#   real/lab12x16   12.3 x 16.4 m loop, median half-width 0.939 m
#
# Both come from `real_data/02_pre-competition` via `scripts/extract_bag_map.py`; see
# `docs/benchmark.md` for the boundary evidence. They are the only entries in any list here whose
# geometry has never been seen in any form, which is what makes a generalisation number possible.
HELDOUT_BASE_IDS = ("real/lab16x07", "real/lab12x16")

HELDOUT_RULES: List[SplitRule] = [
    SplitRule("real/iccas25", BOTH_DIRECTIONS),        # folded real venue, never seen
    SplitRule("real/bb22-3", BOTH_DIRECTIONS),         # pinched real venue
    SplitRule("rt/monza"),                             # long, fast, smooth
    SplitRule("gen/comp-0"),                           # the familiar family, unseen seed
    SplitRule("gen/control-9100"),                     # hairpins and chicanes
    SplitRule("gen/comp-9200", ("",), "pinch", (9200,)),   # sudden narrowing
    SplitRule("real/lab16x07", BOTH_DIRECTIONS),       # unseen real floor, tight hairpin loop
    SplitRule("real/lab12x16", BOTH_DIRECTIONS),       # unseen real floor
]

#: The obstacle half of the held-out split: the same unseen floors with boxes on them. Kept
#: separate because a failure here and a failure above mean different things -- one is unseen
#: geometry, the other is unseen geometry *and* something standing in the way.
HELDOUT_OBSTACLE_RULES: List[SplitRule] = (
    [SplitRule(t, BOTH_DIRECTIONS, "edge", (s,))
     for t, s in (("real/iccas25", 101), ("real/bb22-3", 102), ("gen/comp-0", 103))]
    + [SplitRule(t, BOTH_DIRECTIONS, "line", (s,))
       for t, s in (("real/iccas25", 111), ("real/bb22-3", 112), ("gen/control-9102", 113))]
)

SPLIT_RULES: Dict[str, List[SplitRule]] = {
    "train": TRAIN_RULES,
    "heldout": HELDOUT_RULES,
    "heldout_obstacles": HELDOUT_OBSTACLE_RULES,
}
#: The aliases `track_names` has always taken, mapped onto the three rule sets.
SPLIT_ALIASES = {"train": "train", "eval": "heldout", "heldout": "heldout",
                 "eval_obstacles": "heldout_obstacles", "heldout_obstacles": "heldout_obstacles"}

#: The three groups the UI is allowed to show, and the one sentence each one has to earn.
GROUP_TRAIN = "학습"
GROUP_HELDOUT = "검증"
GROUP_SCENES = "내 환경"
GROUP_ORDER = (GROUP_TRAIN, GROUP_HELDOUT, GROUP_SCENES)
GROUP_SPLIT = {GROUP_TRAIN: "train", GROUP_HELDOUT: "heldout"}
GROUP_HINT = {
    GROUP_TRAIN: "정책이 학습한 맵입니다. 여기 점수는 아는 코스를 얼마나 잘 배웠는지를 말합니다.",
    GROUP_HELDOUT: "어떤 형태로도 학습에 쓰인 적 없는 맵입니다 — 일반화 점수는 여기서만 나옵니다.",
    GROUP_SCENES: "환경 페이지에서 직접 만들거나 고친 환경입니다 (~/f1sim_scenes, 또는 $F1SIM_SCENES).",
}
#: Shown wherever a group label appears. The split is a project convention, not a per-checkpoint
#: fact: runs here were resumed with different `--tracks`, so the honest claim is "this is the
#: project's split", never "this policy has never seen this map".
GROUP_CAVEAT = ("이 분류는 프로젝트가 정의한 기본 분할입니다. 선택한 체크포인트가 실제로 어떤 맵으로 "
                "학습됐는지는 해당 run의 학습 manifest를 봐야 알 수 있습니다.")


def split_rules(name: str) -> List[SplitRule]:
    key = SPLIT_ALIASES.get((name or "").strip())
    if key is None:
        raise TrackError(f"분할 이름을 알 수 없습니다: {name!r} ({', '.join(sorted(SPLIT_ALIASES))})")
    return SPLIT_RULES[key]


def split_names(name: str) -> List[str]:
    """The split as loader names, in the order the literal lists used to be written in."""
    if (name or "").strip() in ("eval_all", "heldout_all"):
        return split_names("heldout") + split_names("heldout_obstacles")
    return [n for rule in split_rules(name) for n in rule.legacy_names()]


def split_specs(name: str) -> List[str]:
    """The split in the short grammar, for anything that shows it to a user."""
    if (name or "").strip() in ("eval_all", "heldout_all"):
        return split_specs("heldout") + split_specs("heldout_obstacles")
    return [sc.short() for rule in split_rules(name) for sc in rule.expand()]


def split_tracks(name: str) -> List[str]:
    """The *base* tracks of a split, in first-appearance order. This is what a picker lists."""
    if (name or "").strip() in ("eval_all", "heldout_all"):
        seen = list(dict.fromkeys(split_tracks("heldout") + split_tracks("heldout_obstacles")))
        return seen
    return list(dict.fromkeys(r.track for r in split_rules(name)))


def split_summary(name: str) -> dict:
    """What the UI needs to explain a split in a sentence and a few numbers.

    `directions` / `obstacles` are the policies actually used across the split's rules, not a
    wish: if no rule mirrors a track, "거울" does not appear.
    """
    if (name or "").strip() in ("eval_all", "heldout_all"):
        rules = split_rules("heldout") + split_rules("heldout_obstacles")
        key = "heldout_all"
    else:
        key = SPLIT_ALIASES[(name or "").strip()] if (name or "").strip() in SPLIT_ALIASES else None
        rules = split_rules(name)
    tracks_ = list(dict.fromkeys(r.track for r in rules))
    directions = list(dict.fromkeys(d for r in rules for d in r.directions))
    obstacles = list(dict.fromkeys(r.obstacle for r in rules))
    families: Dict[str, int] = {}
    for t in tracks_:
        families[t.split("/", 1)[0]] = families.get(t.split("/", 1)[0], 0) + 1
    return {
        "split": key or name,
        "group": next((g for g, s in GROUP_SPLIT.items() if s == key), None),
        "hint": GROUP_HINT.get(next((g for g, s in GROUP_SPLIT.items() if s == key), ""), ""),
        "tracks": tracks_,
        "displays": [get(t).display for t in tracks_],
        "families": families,
        "directions": directions,
        "obstacles": obstacles,
        "n_tracks": len(tracks_),
        "n_variants": sum(len(r.expand()) for r in rules),
        "direction_text": " / ".join(DIRECTION_LABEL[d] for d in directions),
        "obstacle_text": " / ".join(OBSTACLE_LABEL[o] for o in obstacles),
    }


def group_of(track_id: str) -> Optional[str]:
    """Which of the three groups a base track belongs to, or None when it is in neither split."""
    if track_id.startswith("scene/"):
        return GROUP_SCENES
    if track_id in split_tracks("train"):
        return GROUP_TRAIN
    if track_id in split_tracks("heldout") or track_id in split_tracks("heldout_obstacles"):
        return GROUP_HELDOUT
    return None


def groups(scene_ids: Sequence[str] = ()) -> "Dict[str, List[str]]":
    """The three groups as base-track ids. Scenes come from the caller (the console reads the
    folder itself; the worker passes `scene_names()`), and the group is omitted when empty."""
    out: Dict[str, List[str]] = {
        GROUP_TRAIN: split_tracks("train"),
        GROUP_HELDOUT: list(dict.fromkeys(split_tracks("heldout") + split_tracks("heldout_obstacles"))),
    }
    if scene_ids:
        out[GROUP_SCENES] = list(scene_ids)
    return out
