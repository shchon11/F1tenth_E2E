# 환경 에디터 (Environment editor) — integration contract

Date: 2026-09-12. Coordinator: root Claude session. Repo: `/home/shchon11/F1tenth/F1tenth_E2E`,
package `f1sim/` (run Python from `f1sim/`: `cd f1sim && python ...`; venv is
`/home/shchon11/F1tenth/.venv`, activate with `source /home/shchon11/F1tenth/activate.sh`).

Goal: a third console page, **환경** (next to 주행 / 학습), where the user builds and edits driving
environments interactively in 3D — paint/erase duct hoses and tall walls, draw wall polylines,
place/move/rotate/delete the six built-in props, **import mesh assets (GLB/glTF/OBJ/STL)** and place
them as obstacles, save the scene, validate it, and start driving on it from the 주행 page. No
Claude session should be needed to make a map.

Hard simulator constraints (do not fight them; the contract routes around them):

* The LiDAR/contact code sees exactly two kinds of static geometry: (1) boolean cells on one uniform
  grid, layers `duct` (height `duct_height`, beams pass over) and `tall` (infinitely high), and
  (2) props = convex prisms per z-band (`props.sections` → `TrackTensors._build_props`,
  `prop_math.ray_prisms_hits`). Imported meshes therefore become **props whose collision is the
  per-band convex hull** of their cross-sections, optionally **baked** into the grid instead.
* The console process never imports torch (`f1sim/viewer/console/__init__.py`). Everything the
  page uses at interaction time must be torch-free: `f1sim.scene`, `f1sim.props`,
  `f1sim.viewer.gl_scene`, `f1sim.viewer.contours`, `f1sim.viewer.geometry`. Anything needing
  `Track` (centerline extraction, raceline, validation, catalogue import) runs as a **subprocess**
  (`python -m f1sim.scene ...`).
* All UI strings Korean, theme tokens from `console/theme.py`, widgets from `console/widgets.py`.

Three work packages. A and B are independent; C (root) integrates.

---

## A. Scene document, asset import, loader, CLI  (`f1sim/f1sim/scene.py` + friends)

Owner: worker A. Files: **new** `f1sim/f1sim/scene.py`, `f1sim/f1sim/__main__`-style CLI inside
`scene.py` (`python -m f1sim.scene`), **new** `f1sim/f1sim/viewer/contours.py`, **new**
`f1sim/f1sim/viewer/geometry.py`; **edit** `f1sim/f1sim/props.py` (add `mesh` style),
`f1sim/f1sim/maps.py` (add `scene:` prefix + `scene_names()`), `f1sim/f1sim/viewer/server.py` and
`native.py` (re-export contours from the new module), `f1sim/f1sim/viewer/sim_worker.py`
(`build_geometry` delegates to `viewer/geometry.py`; `map_catalog()` adds group `"내 환경 (에디터)"`
listing `scene:<name>`), `f1sim/f1sim/viewer/console/catalog.py` (torch-free `list_scenes()`).
Tests: **new** `f1sim/tests/test_scene.py`, `test_scene_assets.py`, plus keep every existing test
green (`pytest tests/test_console_geometry_v2.py tests/test_console_props.py tests/test_prop_sim.py
tests/test_track_modifiers.py tests/test_lidar3d.py -q` at minimum).

### A.1 On-disk format

```
~/f1sim_scenes/<name>/            # root overridable by $F1SIM_SCENES
    scene.json
    layers.npz                    # duct: bool (H,W), tall: bool (H,W)   (np.savez_compressed)
    assets/<file>                 # imported meshes, copied in verbatim
```

`scene.json` (schema 1):

```json
{
  "schema": 1, "name": "my_hall", "notes": "",
  "resolution": 0.05, "origin": [x0, y0], "shape": [H, W],
  "duct_height": 0.33, "wall_height": 1.0,
  "centerline": [[x, y], ...] | null,
  "props": [
    {"id": "p1", "style": "steel_drum", "x": 1.0, "y": 2.0, "yaw": 0.0, "seed": 2, "dims": {"radius": 0.15}},
    {"id": "p2", "style": "mesh", "asset": "a1", "x": 3.0, "y": 1.0, "yaw": 1.2, "seed": 0, "dims": {"scale": 1.0}}
  ],
  "assets": [
    {"id": "a1", "file": "assets/tire_stack.glb", "name": "tire_stack", "up": "y",
     "scale": 1.0, "tris": 4200, "size": [0.6, 0.6, 0.45], "collision": "bands"}
  ],
  "source": {"map": "real:korea_2026_competition" | null, "created": "2026-09-12T20:00:00"},
  "modified": "2026-09-12T20:05:00"
}
```

Row index increases with +y (same as `Track`). `origin` is the lower-left corner of cell (0,0).
`centerline` is closed, CCW, metres, not repeated at the end. Prop `id`s are unique strings.

### A.2 Python API (torch-free module `f1sim.scene`)

```python
SCENE_SCHEMA_VERSION = 1
SCENES_ROOT = os.environ.get("F1SIM_SCENES", "~/f1sim_scenes")   # expanded on use
def scenes_root() -> str
def scene_dir(name_or_path: str) -> str      # "name" -> <root>/name ; "/abs/dir" or "/abs/dir/scene.json" -> dir
def list_scenes() -> list[dict]              # [{"name","dir","modified","props","shape","source"}], newest first

@dataclass
class PropPlacement:     # mirrors track.StaticProp + id/asset, torch-free
    id: str; style: str; x: float; y: float; yaw: float = 0.0; seed: int = 0
    dims: dict = field(default_factory=dict); asset: str | None = None
    def build(self, doc) -> props.Prop           # resolves asset -> path for style "mesh"
    def footprint_world(self, doc) -> np.ndarray # (K,2) envelope footprint rotated+translated (cached by (style,seed,dims,asset))
    def to_static_prop(self, doc)                # -> track.StaticProp (local import of f1sim.track)

@dataclass
class AssetInfo: id, file, name, up ("y"|"z"), scale, tris, size, collision ("bands"|"hull")

@dataclass
class SceneDoc:
    name: str; resolution: float; origin: tuple[float,float]
    duct: np.ndarray; tall: np.ndarray            # bool (H,W), always same shape
    duct_height: float = 0.33; wall_height: float = 1.0
    centerline: np.ndarray | None = None
    props: list[PropPlacement]; assets: list[AssetInfo]
    notes: str = ""; source: dict; dir: str | None    # dir set after load/save
    # --- construction / persistence
    @staticmethod
    def new_blank(name, width_m, height_m, resolution=0.05, duct_height=0.33) -> SceneDoc   # empty free space, origin (0,0)
    @staticmethod
    def load(name_or_path) -> SceneDoc
    def save(self, name_or_path=None) -> str      # writes scene.json + layers.npz, returns dir; updates modified
    def copy(self) -> SceneDoc                    # deep copy (undo snapshots)
    @staticmethod
    def from_track(track, name, source_map=None) -> SceneDoc   # keeps duct/tall/duct_height/centerline/props
    def to_track(self)                            # -> f1sim.track.Track via Track.from_occupancy(..., duct=, tall=, duct_height=) + props tuple; local import of torch/track
    # --- derived
    @property shape -> (H, W); @property occupancy -> duct | tall
    def bounds(self) -> (x0, y0, x1, y1)
    def world_to_cell(self, x, y) -> (row, col) (floor); def cell_to_world(row, col) -> centre (x, y)
    # --- grid editing (in place, value True paints, False erases; layer in {"duct","tall"}; erasing "duct" or "tall" clears that layer only; `layer="free"` clears both)
    def paint_disc(self, layer, x, y, radius, value=True) -> bool        # returns whether anything changed
    def paint_polyline(self, layer, pts, width, value=True, closed=False) -> bool   # stroke of given width (m)
    def paint_rect(self, layer, x0, y0, x1, y1, value=True) -> bool
    def fill_polygon(self, layer, pts, value=True) -> bool
    def paint_border(self, layer, thickness) -> bool                     # wall ring around the canvas
    def resize_canvas(self, x0, y0, x1, y1) -> None                      # keeps content aligned to the grid, pads with free
    def fit_canvas(self, margin=2.0) -> None                             # shrink/grow to content bbox (grid + props) + margin
    # --- props
    def add_prop(self, style, x, y, yaw=0.0, seed=0, dims=None, asset=None) -> PropPlacement   # fresh unique id
    def remove_prop(self, pid) -> None; def get_prop(self, pid) -> PropPlacement | None
    def bake_prop(self, pid, layer="tall") -> bool     # rasterises the prop's true footprint (union of band polygons for built-ins; projected triangles for meshes) into the layer and removes the prop
    def prop_footprints(self) -> list[tuple[str, np.ndarray]]   # [(id, (K,2) world polygon)], for picking
    # --- assets (torch-free, trimesh)
    def import_asset(self, path, name=None, up="auto", scale=1.0, target_height=None) -> AssetInfo
        # copies file into <dir>/assets/, loads via trimesh (force="mesh"), records tris/size; `up="auto"`: y-up for .glb/.gltf, z-up otherwise; target_height scales the asset so its height is that many metres. Requires self.dir (save first) — raise ValueError otherwise.
    def remove_asset(self, aid) -> None      # refuses (ValueError) while props reference it
    def asset_path(self, aid) -> str          # absolute
    # --- validation that needs no torch (cheap, run on every edit by the page)
    def quick_issues(self) -> list[dict]     # [{"level":"warn"|"error","msg":str,"x":float|None,"y":float|None,"prop":id|None}]
        # checks: prop outside canvas; prop footprint overlapping occupied cells; two props overlapping; asset missing; canvas has no free space; missing centerline (warn)
```

### A.3 `props.py` — the `mesh` style

```python
DIMS["mesh"] = ("path", "scale", "up")          # path: str absolute; scale: float>0; up: "y"|"z"
def mesh_asset(path: str, scale: float = 1.0, up: str = "z", seed: int = 0) -> Prop
```
* `build()` must accept string values for `path`/`up` (extend the validator; every other key stays
  numeric-positive as today).
* Load with `trimesh.load(path, force="mesh")`; apply `up` (y-up → rotate so +Y becomes +Z),
  scale; translate so the XY centroid of the footprint hull is at the origin and `min z = 0`.
* Decimate to at most `MAX_MESH_TRIS = 20000` triangles with `fast_simplification` (installed in
  the venv) when larger; keep vertex colours if the mesh has them, else neutral grey
  `(0.62, 0.63, 0.66)`; flat-shade (per-face vertices, `col[:,3] = 1.0`), material `"plastic"`
  unless the file name contains `metal`/`steel`/`drum` (→ `"metal"`) or `rubber`/`tire`/`tyre`
  (→ `"rubber"`).
* Envelope: convex hull of all XY vertices, height = max z, `bevel_tolerance = 0`. If the hull
  has more than `MAX_FOOTPRINT_VERTS` (24) vertices, replace it by the **k-direction support
  polygon** (intersection of 24 half-planes tangent to the hull) so it stays an over-approximation.
  Apply the same reduction inside `sections()` for any band hull exceeding 24 vertices (built-ins
  never do, so their output is unchanged — add a test that proves byte-identical output for the six
  built-ins).
* Cache builds by `(path, mtime, scale, up)` in a module-level dict (meshes are loaded per prop
  placement otherwise).
* A prop whose file is missing raises `FileNotFoundError` with the path in the message.

### A.4 `maps.py`

* `scene:<name>` → `SceneDoc.load(name).to_track()`; `scene:/abs/path` likewise. Name the track
  `scene:<name>`. Modifiers `~rev`/`~mir` and `+props<seed>` etc. keep working through the existing
  suffix parser (base first).
* `scene_names() -> list[str]` (torch-free; from `f1sim.scene.list_scenes`), included in
  `catalog()`.
* `sim_worker.map_catalog()` adds group `"내 환경 (에디터)"` = `[f"scene:{n}" ...]` first when
  non-empty. `console/catalog.py` gets `list_scenes()` re-export so the GUI can list them without
  the worker.

### A.5 `viewer/contours.py` and `viewer/geometry.py` (torch-free)

* Move `track_contours` / `_runs_off_edge` from `server.py` and `track_contours_mask` from
  `native.py` into `viewer/contours.py` with the same signatures, but taking either a track or any
  object with `.occupancy/.resolution/.origin`; `mask_contours(mask, resolution, origin, ...)` is the
  primitive. `server.py` / `native.py` import from there (no behaviour change; the geometry-v2 tests
  must pass byte-identical).
* `viewer/geometry.py`: `build_track_geometry(track_like, raceline=None) -> dict` — the body of
  `SimWorker.build_geometry` moved verbatim (same keys, `GEOMETRY_VERSION = 2`, `SMOOTH_SIGMA`,
  `prop_batches` moved here too and re-exported from `sim_worker`). `track_like` needs
  `shape, resolution, origin, duct, tall, duct_height, centerline, name, props` where each prop has
  `.build()` — both `Track` and `SceneDoc` satisfy this (`SceneDoc.props` are `PropPlacement`, whose
  `build()` needs the doc: give `PropPlacement` a back-reference `doc` set by `SceneDoc` so
  `.build()` takes no args, or make `SceneDoc.props_for_geometry()` return bound objects — your
  choice, but `build_track_geometry(scene_doc)` must just work).
* `build_track_geometry(doc, only_props=True)` returns only the `props` batches (fast path for
  dragging).
* `wall_height` from the doc (default 1.0) is passed to `wall_mesh_arrays`.

### A.6 CLI (`python -m f1sim.scene`, imports torch lazily; used by the page as a subprocess)

```
python -m f1sim.scene import <map_name> <scene_name|dir>       # catalogue map -> scene (from_track), prints JSON {"dir":..., "props":n, "shape":[H,W]}
python -m f1sim.scene validate <scene_name|dir> [--centerline auto|keep] [--raceline]
    # loads -> to_track(); if centerline missing or --centerline auto: Track.centerline_from_free_space(), written back to scene.json
    # checks (JSON on stdout, exit 0 ok / 1 errors): centerline found & closed; min corridor width along centerline >= 0.6 m (from edt); every prop builds; props not inside occupied cells; a spawn exists (Simulator not needed: use edt at centerline points); with --raceline also Raceline.build_cached(track) and report length/min clearance
    # {"ok": bool, "issues": [{"level","msg","x","y","prop"}], "stats": {"length_m", "min_width_m", "props", "raceline_s": ...}}
python -m f1sim.scene export-ros <scene_name|dir> <out_dir>    # Track.save_ros_map
```
Progress lines go to stderr; stdout is exactly one JSON object.

### A.7 Tests (A)
* round-trip save/load equality (grids, props, assets, centerline);
* `from_track(gen:competition:0+props3)` → `to_track()` gives identical duct/tall/props and the same
  `grid_key()`;
* `maps.load("scene:<tmp>")` works with `~rev`;
* every paint op changes only the intended layer; `bake_prop` covers the footprint;
* import a trimesh-generated box/cylinder GLB (y-up) and an STL: envelope height/size right, hull
  ≤ 24 verts, `sections()` non-empty, and `build_track_geometry(doc)` returns prop batches with the
  right triangle count; a LiDAR beam at `z < height` hits it, one above passes (use
  `Lidar`/`TrackTensors` on CPU like `tests/test_prop_sim.py` does);
* `import f1sim.scene`, `f1sim.viewer.geometry`, `f1sim.viewer.contours` do not import torch
  (subprocess check with `-c "import sys, f1sim.scene; assert 'torch' not in sys.modules"`);
* CLI `import` + `validate` on `gen:competition:0` in a tmp `F1SIM_SCENES` root.

---

## B. Editor viewport  (`f1sim/f1sim/viewer/console/editor_viewport.py`)

Owner: worker B. New file plus tests `f1sim/tests/test_editor_viewport.py`. May make small,
backward-compatible changes to `viewport.py` (e.g. factor `_apply_geometry`/`_resolve_target` so a
subclass can reuse them) and `gl_scene.py` (e.g. an `add_line` that accepts open polylines with
per-vertex colour — it already does — or a `draw_lines_only` helper); every existing console test
must stay green (`pytest tests/test_console_*.py tests/test_editor_viewport.py -q`, run under
`xvfb-run -a` with `LIBGL_ALWAYS_SOFTWARE=1` when a display is needed; the offscreen QApplication
fixture pattern is in `tests/test_console_layout.py`).

```python
class EditorViewport(ViewportWidget):
    """Draws a TrackGeometry with a free camera and turns mouse input into world-space events."""
    # ---- camera (free, pivot-based; no cars)
    pivot: np.ndarray (2,) ; az, el, dist : floats ; ortho_top: bool
    def frame_all(self)                       # from geometry presentation/content bounds (like 전체보기)
    def set_top_view(self, on: bool)          # el -> ~89.9 deg, up = +y ; off -> restore last perspective
    def look_at_point(self, x, y)             # move pivot, keep az/el/dist
    # mouse: middle-drag or Alt+left = orbit; right-drag or Shift+left(or Space+left) = pan on the ground plane;
    # wheel = zoom toward the cursor's ground point; Ctrl+wheel is NOT consumed (emitted as wheel_edit)
    # keys handled here: F (frame_all), Home (same), 5 toggles top view; everything else → key_pressed
    # ---- picking
    def ground_point(self, px, py) -> Optional[tuple[float,float]]   # widget px -> world (x,y,0) via stored view/proj; None when the ray misses the plane
    def set_pickables(self, items: list[tuple[str, np.ndarray]])     # [(id, (K,2) world polygon)] for hover/press hit testing (point-in-polygon, last wins)
    def hit_test(self, x, y) -> str                                  # id or ""
    # ---- overlays (all optional; drawn every frame after the scene, above the ground, no depth write)
    def set_cursor(self, kind: str, x, y, radius=0.0, yaw=0.0)  # kind in {"none","brush","cross","place"}; brush = ring of `radius`, place = ring + heading tick
    def set_selection(self, polys: list[np.ndarray], colour=theme accent)   # outlines on the ground for selected props (closed)
    def set_hover(self, poly: Optional[np.ndarray])                          # thinner outline
    def set_preview(self, pts: np.ndarray, width: float, closed=False)       # in-progress polyline/polygon tool: centre line + offset band edges
    def set_gizmo(self, x, y, yaw, radius=0.4, on=True)                     # move handle (cross) + rotation ring with heading tick at the selection
    def set_layer_visibility(self, ducts=True, walls=True, props=True, floor=True)   # skips the corresponding static meshes at draw time (tag meshes on upload)
    # ---- signals (world coordinates; buttons/modifiers are Qt ints)
    ground_pressed  = pyqtSignal(float, float, int, int, str)   # x, y, button, modifiers, hit id ("" when none)
    ground_moved    = pyqtSignal(float, float, int, int)         # x, y, buttons, modifiers  (only while a button is down and not consumed by the camera)
    ground_released = pyqtSignal(float, float, int, int)         # x, y, button, modifiers
    hovered         = pyqtSignal(float, float, str)              # x, y, hit id   (mouse move with no button)
    left_widget     = pyqtSignal()                               # leaveEvent → page hides cursor
    wheel_edit      = pyqtSignal(int, int)                       # angleDelta().y(), modifiers — only when Ctrl or Alt is held
    key_pressed     = pyqtSignal(int, int)                       # key, modifiers — every key the viewport does not consume
    camera_changed  = pyqtSignal()                               # after orbit/pan/zoom (page updates the corner chip)
    # ---- drawing
    # paintGL: resolve target, apply pending geometry, draw sky + static (respecting layer visibility) + overlays with the free camera. No frames/cars/lidar. Empty-state label text: "환경을 새로 만들거나 목록에서 고르세요." when geometry is None.
    # Keep `set_geometry`, `grab screenshot`, `upload_timed`, `draw_timed` from the base class working.
    # Corner chip text is set by the page via `set_corner_text(str)` (reuse `_corner`).
```

Conventions: the viewport never mutates document state; it only reports events and draws what it
is told. The page calls `update()` after changing overlays (or the setters call it). The ground grid
in the shader stays. Selection accent colour `theme.C["accent"]`, hover `text.1`, preview `warn`,
gizmo ring `ego`.

Tests (offscreen QApplication + real GL under xvfb): construct, `set_geometry` with the fake-worker
style geometry from `tests/console_fake_worker.py::_geometry`, `frame_all`, a projection round-trip
(`ground_point` of the projected pixel of (x, y, 0) returns (x, y) within 1 cm at a few camera
poses), `hit_test` on a square, a synthetic `QMouseEvent` press/move/release emits the three
signals with sensible world coordinates, orbit/pan change `az`/`pivot`, and `set_layer_visibility`
changes what is drawn (count draw calls or compare framebuffer pixels).

---

## C. Editor page + wiring (root)

`f1sim/f1sim/viewer/console/env_editor.py` (`EnvEditorPage(QWidget)`, `set_active(bool)`, signals
`drive_requested(str map_name)`, `scenes_changed()`), header segmented button `("edit", "환경", …)`,
`window.set_mode`, `session._connect_window`, docs (`docs/environment_editor.md`, README section,
`docs/local_pc.md`), integration tests. Root also reviews and merges A and B.

Sequence when the user presses **주행**: page saves → `drive_requested("scene:<name>")` → window
switches to 주행 with that map selected in the picker under `내 환경 (에디터)` (list refreshed from
`catalog.list_scenes()`) → user presses 시작.

---

## Working rules for A and B

* Work directly in the repo (no worktree); do not touch files outside your package list; do not
  commit. Root integrates. If you must change a shared file (`viewport.py`, `gl_scene.py`,
  `props.py`), keep the change additive and backward-compatible.
* Korean strings for anything user-facing; docstrings in English like the rest of the code.
* Never run GPU jobs (a training run holds the GPU). CPU-only tests; `device="cpu"`.
* When done write `work/env-editor/REPORT-<A|B>.md`: what was built, the exact test commands and
  their pass counts, anything deviating from this contract and why.
