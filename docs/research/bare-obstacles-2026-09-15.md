# 기본 vs 없음: what the obstacle selector was claiming, and what it says now (2026-09-15)

Branch `feat/opponent-slots`, base `df60b44`. Worker 21, deliverable 5.

## The report

> 맵 고르고 장애물 정도 여부 선택할 때 없음을 선택하면 기본 맵이 되는데, 내가 커스텀해서 장애물을 놨으면
> 그 맵 자체가 기본으로 나와서 장애물 없음이라는 말과 안맞아. 그래서 장애물 있게 렌더링 해도 없음 기본
> 이렇게 더 나눠야 할 듯.

`OBSTACLE_LABEL[""]` was `없음` and it meant **no procedural family**. On a measured floor or a
generated layout those are the same sentence — there is nothing on the map to begin with. On an
editor scene they are not: the scene carries the props its author placed, and `Track.props` survives
every loader path. So "없음" produced a map full of boxes, and the one word was doing two jobs.

## What changed

**One axis became two.** `Scenario.obstacle` now holds a *procedural family* or nothing, and
`Scenario.bare` is a separate flag. The control works in a single **choice key** that covers both --
`""`, `"bare"`, a family, or `"bare+<family>"` -- so a combo, a checkbox row and a command line can
all hold the same value without either page knowing how the two fields are arranged.

| choice | UI | short grammar | loader | what it builds |
| --- | --- | --- | --- | --- |
| `""` | 기본 | `scene/hall` | `scene:hall` | the map as authored |
| `bare` | 없음 | `scene/hall#bare` | `scene:hall+bare` | the placed props removed, walls only |
| `hard` | 극단 | `scene/hall#hard:3` | `scene:hall+hard3` | the map as authored, plus the family |
| `bare+hard` | 없음 + 극단 | `scene/hall#bare+hard:3` | `scene:hall+bare+hard3` | removed first, then the family |

**`bare` takes no seed.** It places nothing. `#bare:3` is refused with that sentence, and a family
without a seed is refused with the opposite one — the two mistakes are different and say so.

**`+bare` is applied to the base map, before any family.** That is the whole reason it is peeled off
the name in `maps._load_base` and threaded down rather than applied to the result: applied last it
would remove the props the family had just placed and be `+bare` wearing a family's name.

**What it removes, precisely.** `Track.props` -- the poses an author placed -- and nothing else. The
grids are untouched, so a painted wall is still a wall. A loader cannot honestly guess which painted
cells were meant as scenery: a wall and a box are the same cells in the same layer. `Track.bare()`
returns `self` when there is nothing to remove, which is what makes the suffix free on every map that
is not an editor scene.

**The families really do add now.** `Track.with_static_props` *replaced* a track's props
(`t.props = tuple(out)`), so `scene:hall+props3` silently threw away the author's boxes -- the same
lie, one control over. It appends now, and places the new props clear of the old ones. Only an editor
scene can reach that code carrying props, so `scene:<name>+props<seed>` is the only kind of id whose
meaning moved; nothing in any split, recipe or benchmark suite names one.

## The console

The 주행 page's 장애물 combo shows 기본 / 없음 / families. On a scene with three placed obstacles the
first two read `기본 (배치된 장애물 3개)` and `없음 (배치 장애물 제거)`; on a map with none, 기본
carries no count and 없음 is **greyed** with `이 맵은 배치 장애물이 없음 — '기본'과 같습니다.`. Greyed
rather than hidden, because "this map has nothing on it" is an answer and a missing entry would leave
기본 looking like the only thing there is. Selecting a map with nothing placed while 없음 is selected
falls back to 기본 rather than leaving a selection that means nothing.

Beside it, **배치 장애물 먼저 제거** is the `+bare` composition, enabled exactly where both halves are
true: the map has placed obstacles and a family is selected. The session header names what was
actually built (`장애물 없음`, `장애물 기본 (3개)`) from the worker's facts, not from the control.

The 학습 page's picker is the same two words as two checkboxes, which is what that multi-select row
has always meant: tick both and every selected map is built twice.

## Byte identity

Required for every id that does not carry `+bare`, and measured rather than asserted. The same 28
ids were loaded in this tree and in a clean extract of `df60b44`, hashing the occupancy, duct and
tall grids, the EDT, the centreline and every prop pose:

`real:blackbox2022_1` (plain, `~rev`, `~mir~rev`, `+obs3`, `+rlobs3`, `+pinch3`, `+props3`,
`+hard3`), `real:map16x07`, `real:korea_2026_competition`, `gen:competition:0` (plain and all five
families), `gen:control:1400`, `gen:hallway:2`, `gen:recipe:7`, `gen:recipe:7+props2`, `gym:levine`,
`rt:Austin` (plain, `+props3`, `+hard3`), and a propless editor scene (plain, `+props3`, `+hard3`,
`~rev`).

**Identical in all 28.** The `scene:<name>+props<seed>` case whose meaning deliberately moved is a
scene *with* authored props, which no id in that list is; the sweep includes a propless scene through
the same code path to show the path itself did not move.

The env-side rollout hashes from the slot work were re-run unchanged as well (nine opponent
configurations, generator state included), since `Track`, `maps` and `tracks` sit under the env.

## Tests

* `tests/test_bare_obstacles.py` (28) — the registry says two different things; both grammars round
  trip a bare choice, including composed; a seed is demanded exactly where something is placed;
  `+bare` drops the props and leaves the grids; it is a no-op where nothing is placed; a family adds
  to what the author placed; `+bare` composes with a prop family and with a grid family and is
  applied first; it survives `~rev`/`~mir`; the short grammar reaches the loader.
* `tests/test_console_obstacle_choice.py` (14, xvfb) — the counts on the labels, the greyed 없음 and
  its reason, the seedless spec, the composition checkbox and where it is disabled, the hint that
  says a family adds, the fallback when the map changes under 없음, the header line, and the 학습
  page's two checkboxes.
* `tests/test_sim_worker_smoke.py` (+1) — a real worker builds a scene twice, `기본` and `없음`, and
  the facts *and the geometry it sends the console* carry two obstacles and none.
* `tests/test_tracks.py` — its round-trip generator now walks choice keys rather than family names,
  and the "not every map carries every family" assertions name 기본/없음 as universal.
