# REPORT-A — Scene document, asset import, loader, CLI

Worker A, 2026-09-12. Everything in section A of `CONTRACT.md` is implemented; nothing was committed.
All paths below are under `/home/shchon11/F1tenth/F1tenth_E2E/f1sim/`.

## What was built

| File | Status | Contents |
|---|---|---|
| `f1sim/scene.py` | **new** (1004 lines) | `SCENE_SCHEMA_VERSION`, `SCENES_ROOT` / `scenes_root()` / `scene_dir()` / `list_scenes()`; `AssetInfo`, `PropPlacement` (with `doc` back-reference so `build()` takes no argument), `SceneDoc` with every method of A.2 (`new_blank`, `load`, `save`, `copy`, `from_track`, `to_track`, `shape`/`occupancy`/`bounds`, `world_to_cell`/`cell_to_world`, `paint_disc`/`paint_polyline`/`paint_rect`/`fill_polygon`/`paint_border`, `resize_canvas`/`fit_canvas`, `add_prop`/`remove_prop`/`get_prop`/`bake_prop`/`prop_footprints`, `import_asset`/`remove_asset`/`asset_path`, `quick_issues`); `lane_seed()` (see deviations); `validate()`; the CLI `main()` (`import`, `validate [--centerline auto\|keep] [--raceline] [--seed X Y]`, `export-ros`). Torch-free; `Track` is imported inside `to_track` / `to_static_prop` / the CLI only. |
| `f1sim/viewer/contours.py` | **new** | `mask_contours(mask, resolution, origin, ...)` primitive; `track_contours`, `track_contours_mask`, `_runs_off_edge` moved verbatim (same signatures, any object with `.occupancy/.resolution/.origin`). |
| `f1sim/viewer/geometry.py` | **new** | `GEOMETRY_VERSION = 2`, `SMOOTH_SIGMA`, `_bbox`, `placement_key`, `PropBuildError`, `prop_batches`, `build_track_geometry(track_like, raceline=None, only_props=False)` — the body of `SimWorker.build_geometry` moved verbatim, plus `wall_height` from the doc (default 1.0) passed to `wall_mesh_arrays`, and `only_props=True` returning just the prop batches. |
| `f1sim/props.py` | edited (additive) | `DIMS["mesh"] = ("path","scale","up")`, `STRING_DIMS`, `build()` accepts string `path`/`up` (validated) and dispatches `"mesh"` to `mesh_asset()`; `MAX_MESH_TRIS = 20000`, `MESH_GREY`, `_MESH_CACHE` keyed by `(abspath, mtime, scale, up)`; `mesh_asset(path, scale, up, seed)` (trimesh `force="mesh"`, y-up rotation, scale, centroid/`min z = 0` recentring, `fast_simplification` decimation with colour carry-over, flat shading, `col[:,3]=1`, material from the file name, hull envelope with `bevel_tolerance = 0`, `FileNotFoundError` naming the path); `support_polygon(hull, k)` (k-direction support polygon via half-plane clipping); `sections(..., max_verts=None)` applies the cap when asked or when `prop.spec["max_section_verts"]` is set (mesh props set it to 24). `STYLES` stays the six built-ins on purpose (`build_all`, `Track.with_static_props` iterate it). |
| `f1sim/maps.py` | edited | `scene:<name>` / `scene:/abs/dir` → `SceneDoc.load(n).to_track()`, track named `scene:<name>`; `+props<seed>` and `~rev`/`~mir` work through the existing parsers; other `+obs`-style suffixes raise `ValueError`. `scene_names()`; `catalog()` lists `scene:<n>`. The per-process `_BASE_CACHE` key for `scene:` names includes the mtimes of `scene.json`/`layers.npz`, so an edited-and-saved scene is reloaded, not served stale. |
| `f1sim/viewer/server.py`, `native.py` | edited | function bodies replaced by `from .contours import ...`; same names exported, `native.track_contours_mask` is the contours module's function. |
| `f1sim/viewer/sim_worker.py` | edited | re-exports `GEOMETRY_VERSION`, `SMOOTH_SIGMA`, `_bbox`, `build_track_geometry`; `prop_batches` and `SimWorker.build_geometry` delegate to `geometry.py` and re-raise `PropBuildError` as `StartConfigError` (existing tests keep their exception type and Korean message); `map_catalog()` adds `"내 환경 (에디터)"` first when non-empty and lists scenes in `전체 카탈로그` too; `load_track()` never serves a `scene:` name from its LRU (edited between sessions). |
| `f1sim/viewer/console/catalog.py` | edited | `SCENES_GROUP = "내 환경 (에디터)"` first in `GROUP_ORDER`, a `GROUP_HINT` entry, torch-free `list_scenes()` re-export. |
| `tests/test_scene.py` | **new** (19 tests) | torch-free import check for `f1sim.scene`, `viewer.geometry`, `viewer.contours`, `console.catalog` (subprocess); `scene_dir`; `list_scenes` order; save/load round trip (grids, props, assets, centerline, notes, schema); `copy` independence; every paint op changes only its layer, no-op repaint returns False, erase semantics incl. `"free"`; cell-centre convention; thin strokes; `resize_canvas`/`fit_canvas`; fresh ids, `bake_prop` covering the footprint (and a sub-cell post); `quick_issues`; `from_track(gen:competition:0+props3)` → `to_track()` identical duct/tall/props and the same `grid_key()`; `maps.load("scene:…")` with `~rev`, `~mir`, `+props3`, an absolute dir, an on-disk edit invalidating the cache, bad suffix, missing scene; `map_catalog()` group only when scenes exist; `build_track_geometry` on a `SceneDoc` and a `Track` (same keys, `wall_height` reaches the wall mesh, `only_props` fast path); contours moved without change; CLI `import` + `validate --centerline auto` (centerline written back, exit 0) + a failing scene (exit 1) + `export-ros` + a missing scene (one JSON object, exit 1). |
| `tests/test_scene_assets.py` | **new** (14 tests) | `sections()` of the six built-ins byte-identical to pre-change md5s at 4 and 8 bands; `support_polygon` encloses the hull within budget; y-up GLB box (height/size/hull/material/colours/normals/cache/scale); STL 96-gon cylinder (hull ≤ 24, sections ≤ 24, `section_halfplanes` accepts them); dim validation incl. `FileNotFoundError` with the path; an 81 920-triangle coloured PLY is decimated to ≤ 20 000 and keeps its colour; `import_asset` copy/measure/`target_height`/duplicate names/round trip/`remove_asset` refusal/save-as carrying assets/missing-file issue; baking a mesh prop; **the simulator sees it**: `build_track_geometry(doc)` has the right triangle counts, and through `Simulator` on CPU a level beam stops at the 0.45 m box at the right range with `HIT_TALL`, an 8° nose-up beam passes over it, and the car standing on it reports a prop contact. |
| `tests/test_sim_worker_smoke.py` | edited (1 assertion) | see deviations. |

`scene.json` follows A.1 exactly (schema 1, `origin` lower-left of cell (0,0), row index up with +y, closed CCW centerline not repeated, unique prop ids, assets with `up/scale/tris/size/collision`, `source`, `modified`). Cell centres are at `origin + (col+0.5, row+0.5)·res`; every paint op and `quick_issues` rasterise on cell centres.

## Test commands and results

```
source /home/shchon11/F1tenth/activate.sh && cd /home/shchon11/F1tenth/F1tenth_E2E/f1sim
pytest tests/test_scene.py tests/test_scene_assets.py -q -x
#   33 passed
pytest tests/test_scene.py tests/test_scene_assets.py tests/test_console_geometry_v2.py \
       tests/test_console_props.py tests/test_prop_sim.py tests/test_track_modifiers.py \
       tests/test_lidar3d.py tests/test_map_sets.py tests/test_sim_worker_smoke.py -q -x
#   136 passed, 2 skipped, 14 warnings in 146.55s
#   skips: tests/test_lidar3d.py::test_triton_matches_torch_on_random_track (needs cuda; GPU busy, never used)
#          tests/test_map_sets.py:88 ("only meaningful under the software-GL CPU run contract")
```
CPU only throughout (`device="cpu"`, no CUDA use). `pyflakes` clean on every touched file.

(Root re-ran the suites after integration: console/viewport/scene/worker 310 passed, 1 skipped under Xvfb; a second CPU run of the simulator-side suites was cut off by a 1700 s timeout at load average 33 with 36 passed, 2 skipped and no failures; worker A's own run of the same files recorded 136 passed, 2 skipped.)

Worker A's sanity run of console tests outside the requested list that touch the catalogue group
order (on the real `:0` display `test_console_session`'s Qt/GL harness segfaults in fixture setup
before any assertion — environmental; under Xvfb it is fine):
```
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a pytest tests/test_console_session.py tests/test_console_layout.py \
       tests/test_sim_worker_lifecycle.py tests/test_console_startup.py -q
#   91 passed, 1 failed -- test_console_startup.py::test_worker_and_console_agree_on_the_group_names, whose file
#   root rewrote at 21:42:01 while the run was in flight (old literal collected, new source reported); on the current files:
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a pytest tests/test_console_startup.py -q
#   13 passed
```

## Deviations from the contract, and why

1. **`sections()` vertex cap is per-prop, not unconditional.** The contract says "built-ins never exceed 24 vertices" — they do: `steel_drum`'s ribbed bands hull to **32** vertices at 4 bands (what `TrackTensors._build_props` uses; its own comment says so) and **60** at 8. An unconditional cap would have re-cut the drum's collision/LiDAR prisms (k_pad 32 → 24) on every existing `+props` map and made the byte-identical test impossible. So `sections(prop, ..., max_verts=None)` caps only when asked, or when `prop.spec["max_section_verts"]` is set — `mesh_asset` sets it to 24, the six built-ins never do. The test pins md5s of all six built-ins' sections at 4 and 8 bands, recorded before the change.
2. **`test_sim_worker_smoke.py::test_map_catalog_uses_the_group_names_the_console_expects`** asserted `list(groups) == GROUP_ORDER`. With the editor group first in `GROUP_ORDER` but (per contract) only sent when non-empty, the assertion became "same order as `GROUP_ORDER`, every group except the optional editor group present" (7 lines). It passes with or without scenes on the machine.
3. **`lane_seed()` and `--seed X Y` were added to `validate`.** `Track.centerline_from_free_space` without a seed takes the *largest* free region, which on a duct ring inside a hall (every editor scene, and `gen:competition:0` once its centerline is dropped) is the floor *outside* the ring: it returned a line 5.8 m off the lane with a 0.18 m "corridor". `lane_seed` picks the innermost annulus (a region with an infield hole whose hole contains no other holed region) and passes it as `seed_xy`; an explicit `--seed`, then the previous centerline's first point, take precedence. Verified on gen:competition 0/3, hallway, circuit, serpentine, real:korea and on walled/open editor rings (min clearance 0.33–1.03 m). It is torch-free so the page can call it.
4. **`validate` stats keys**: `length_m`, `min_width_m`, `props`, `seed_xy`, and with `--raceline` `raceline_s` (build seconds), `raceline_length_m`, `raceline_min_clearance_m`. Corridor threshold `MIN_CORRIDOR_M = 0.6`, spawn clearance 0.35 m along the centerline, also outside every prop footprint.
5. **Painting one layer does not clear the other** (the contract's "changes only the intended layer" was read literally); a cell can be both `duct` and `tall`, as `Track` allows. `"free"` clears both. `bake_prop` into `tall` likewise leaves `duct`.
6. `PropPlacement.build()` takes an optional `doc` but normally uses the back-reference set by `SceneDoc` (`__post_init__`, `add_prop`, `copy`, `load`), which is the option the contract offered. `prop_batches` keys its build cache with `placement_key()` (sorted dict items + asset) because `tuple(dict)` would have been the keys only.
7. `.gltf` with external `.bin`/textures: only the `.gltf` is copied (the contract says "copied in verbatim"); `.glb`, `.obj`, `.stl`, `.ply` are self-contained. `import_asset` loads the file once through `mesh_asset` so a broken file fails at import, not at first placement, and removes the copy on failure.
8. `map_catalog()` also lists `scene:<n>` under `전체 카탈로그` (harmless, useful for the picker's flat search).

## Notes for integration (C)

- The page can do everything at interaction time with `SceneDoc` + `viewer.geometry.build_track_geometry(doc)` (or `only_props=True` while dragging) + `viewer.contours`; `quick_issues()` after each edit; `python -m f1sim.scene validate <name> [--seed x y]` as a subprocess for the centerline/corridor/spawn/raceline checks (stdout is one JSON object, exit 0/1; progress on stderr).
- `catalog.list_scenes()` fills `내 환경 (에디터)` without the worker; the worker's `map_catalog()` sends the same group first when non-empty.
- `maps.load("scene:<name>")` and the worker's `load_track` both re-read an edited scene (mtime-keyed cache / no LRU), so 편집 → 저장 → 주행 → 편집 → 주행 works inside one worker lifetime.
