# Environment editor

**2026-09-12.** The console's third page, **환경**, builds and edits driving environments in 3D:
paint duct hoses and tall walls, draw wall polylines, place the built-in props, **import mesh
assets** (GLB / glTF / OBJ / STL / PLY) as obstacles, save the scene, validate it, and drive on it
from the 주행 page. Making a map no longer needs a generator seed, a PGM edited by hand, or a
session with someone doing it for you.

```bash
cd f1sim
python3 -m f1sim.learn.watch          # header: 주행 · 학습 · 환경
```

[![the environment editor](media/f1tenth-environment-editor.png)](media/f1tenth-environment-editor.png)

<sub>A hall drawn with the duct-line tool, a pillar from the rectangle tool, three built-in props and
two placements of an imported STL tyre stack (the selected one carries the gizmo). Captured from the
real console on software GL by `work/env-editor/capture_editor.py`; the scene, prop placements and
file hashes are in [`media/environment-editor-manifest-2026-09-12.json`](media/environment-editor-manifest-2026-09-12.json).
The [top view](media/f1tenth-environment-editor-top.png) shows the wall brush cursor.</sub>

## What a scene is

A scene is a folder under `~/f1sim_scenes/<name>/` (root overridable with `$F1SIM_SCENES`):

```
scene.json        name, resolution, origin, duct height, centerline, props, assets, notes
layers.npz        duct: bool (H,W), tall: bool (H,W)
assets/           imported meshes, copied in verbatim
```

`f1sim/f1sim/scene.py` owns the format (`SceneDoc`), and `maps.py` loads it as
**`scene:<name>`** — a first-class catalogue entry, so `~rev`, `~mir` and `+props<seed>` compose
with it, `--tracks scene:my_hall` trains on it, and the driving console lists every saved scene under
**내 환경 (에디터)** in its map picker. `python -m f1sim.scene export-ros <name> <dir>` writes the
ROS `map.yaml` + PGM if another stack needs it.

The layers map directly onto what the simulator can see. The LiDAR traces two distance fields —
`duct` (a hose of `duct_height`, beams pass over it) and `tall` (blocks every beam) — and props as
finite convex prisms. Nothing in the editor produces geometry outside those three kinds, which is
why a scene that looks right in the editor is what the policy is trained and evaluated against.

## Tools

| key | tool | what it does |
| --- | --- | --- |
| `V` | 선택·이동 | click a prop to select (Shift adds), drag to move (grid snap unless Alt), `R` / Ctrl+wheel rotates, `Delete` removes, Ctrl+D duplicates |
| `B` / `W` | 덕트 · 벽 브러시 | paint a duct hose / tall wall along the mouse path; Ctrl+wheel changes the radius; one undo step per stroke |
| `E` | 지우개 | clears both layers |
| `L` / `K` | 덕트 선 · 벽 선 | click vertices, double-click / Enter / right-click commits a band of `2 × brush` width; `선 닫기` joins the last vertex to the first |
| `M` | 사각형 | drag a rectangle of tall wall (Shift: duct) |
| `P` | 다각형 채우기 | click vertices, commit fills the polygon with tall wall |
| `A` | 배치 | put the asset chosen in the library at the click; Ctrl+wheel or `R` turns it first |

Camera: middle-drag or Alt+left orbits, right-drag or Shift+left pans on the ground, the wheel
zooms toward the cursor, `F` frames everything, `5` toggles a top-down view. Ctrl+Z / Ctrl+Y undo
and redo up to 60 steps; every edit is a snapshot of the document, so undo is exact.

The inspector on the right edits the selected prop numerically — position, heading, the style's
own dimensions (`width/depth/height`, `radius`, `scale` for meshes), its appearance seed — and
offers **벽으로 굽기**: stamp the prop's true footprint into the tall layer and remove the prop. That
is the way to turn a non-convex imported mesh into an exact wall, at the cost of the top: the grid
has no height, so a baked object blocks beams at every height.

## Assets

**메쉬 가져오기…** copies the file into the scene's `assets/` folder and reads it with trimesh:
glTF/GLB are taken as Y-up, everything else as Z-up (the asset record keeps the choice), the model
is centred on its footprint and set on the floor, and anything over 20 000 triangles is decimated
for the renderer. It then appears in the library and is placed like a built-in prop, with a `scale`
dimension.

The collision shape is derived, not authored: the LiDAR and the contact test see the **convex hull
of each of a few height bands** of the mesh (`props.sections`), the same path the built-in props
use. A drum, a cone, a tyre stack or a stack of boxes is represented tightly; an L-shaped piece
becomes its hull per band — bake it into the grid if that matters. The editor marks a prop whose
footprint overlaps a wall, and the validator refuses one the renderer cannot build, so a collider
you cannot see is not something a scene can contain.

## Validate and drive

**검증** saves the scene and runs `python -m f1sim.scene validate` in a subprocess (this is the
one step that imports torch): it extracts a centerline from the free space when the scene has none,
checks the loop closes, measures the corridor width along it, builds every prop, and reports what it
found in the panel — click an issue to fly the camera to it. **이 환경으로 주행 →** validates, then
switches to the 주행 page with `scene:<name>` selected; press 시작 as usual. The worker loads the
scene through the ordinary map path, so a raceline is built and cached on first use like any map —
on the scene above that took about three minutes on the CPU (`validate --raceline` does it ahead of
time from a shell), after which it is cached under `~/.cache/f1sim/racelines/`.

Editing a catalogue map starts from **기존 맵에서 시작**: the same grouped picker as the driving
page (filled once the worker has listed the maps; a name can also be typed into its search box), and
**이 맵 복제해서 편집** runs `python -m f1sim.scene import <map> <name>`, copying the layers,
centerline and props into a new scene — `real:korea_2026_competition` to add a chicane to the real
venue, or `gen:competition:3+props7` to move the props around.

## Limits

* The simulator is not updated live: a change on the 환경 page reaches a running session only
  through 시작 (a rebuild — seconds without `torch.compile`).
* Walls have one height class each; a mesh baked into the grid loses its top. Props stay convex
  per band. There is no per-cell height field and no arbitrary triangle collision, by design of the
  tracer (see [Architecture](architecture.md#maps)).
* Scenes are local files, not catalogue code: `learn/common.py`'s named track sets do not include
  them unless you list `scene:<name>` in a `--tracks` argument.

## Code

| | |
| --- | --- |
| `f1sim/f1sim/scene.py` | `SceneDoc`, `PropPlacement`, `AssetInfo`; paint/fill/bake operations; asset import; `python -m f1sim.scene import / validate / export-ros` |
| `f1sim/f1sim/props.py` | the `mesh` prop style (`path`, `scale`, `up`) beside the six built-ins |
| `f1sim/f1sim/maps.py` | the `scene:` prefix and `scene_names()` |
| `f1sim/f1sim/viewer/contours.py`, `geometry.py` | torch-free marching-squares contours and the geometry build the worker and the editor share |
| `f1sim/f1sim/viewer/console/editor_viewport.py` | free camera, ground picking, overlays (cursor, selection, preview band, gizmo) |
| `f1sim/f1sim/viewer/console/env_editor.py` | the page: tools, undo stack, geometry builder thread, subprocess jobs, inspector, asset library |
| `f1sim/tests/test_scene*.py`, `test_editor_viewport.py`, `test_env_editor.py` | round-trips, LiDAR against imported meshes, picking, tools and undo, the hand-off |
