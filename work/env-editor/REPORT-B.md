# REPORT-B — Editor viewport (`f1sim/viewer/console/editor_viewport.py`)

Date: 2026-09-12. Worker B. Nothing committed; root integrates.

## What was built

### New: `f1sim/f1sim/viewer/console/editor_viewport.py` — `EditorViewport(ViewportWidget)`

Inherits the GL ownership path unchanged (context, `Scene`, `_resolve_target` / `_ForeignFramebuffer`,
deferred `set_geometry` → `_apply_geometry`, `grab_png`, `upload_timed` / `draw_timed` / `gl_failed`).
Replaces everything that assumes a session:

* **Camera** — `pivot (2,)`, `az`, `el`, `dist`, `top`, `ortho_top` (default `True`: the top view is
  orthographic, so a brush ring on the ground is a true circle of its radius). `frame_all()` frames
  `drawn_bounds()` (content bounds on v2, canvas on v1) by bisection on the real projection of the four
  corners, so it is tight for any elevation / aspect / projection. `set_top_view(on)` → el 89.9°, up +y,
  restores the saved az/el on off. `look_at_point(x, y)`. `camera_state()` returns a dict for the corner
  chip. The first geometry after a blank view is auto-framed; re-uploads (every edit) leave the camera alone.
* **Mouse** — middle-drag or Alt+left = orbit (orbiting out of the top view exits it continuously from
  el = 89°); right-drag, Shift+left or Space+left = pan on the ground plane (the ground point under the
  cursor stays under the cursor); wheel = zoom toward the cursor's ground point; **Ctrl/Alt+wheel is not
  consumed** → `wheel_edit(delta, modifiers)`. Everything else is an edit gesture → `ground_pressed(x, y,
  button, modifiers, hit)`, `ground_moved(x, y, buttons, modifiers)` (only during an edit drag),
  `ground_released(x, y, button, modifiers)`, `hovered(x, y, hit)` (no button; mouse tracking is on),
  `left_widget()`. **Coordinator addition:** `double_clicked(x, y)` on a left double-click; Qt replaces
  the second press with the double-click event, so there is no second `ground_pressed`, and the trailing
  release is dropped because no press opened an edit drag (press / release / double_clicked / —).
* **Keys** — F and Home → `frame_all`, 5 → toggle top view, Space → pan modifier (consumed); every other
  key → `key_pressed(key, modifiers)` and the event is *ignored* so window shortcuts still see it.
* **Picking** — `ground_point(px, py)` unprojects through the same matrices the paint uses (`_matrices()`,
  computed from camera state on demand, so a pick between a wheel event and its repaint already sees the
  new camera); returns `None` when the ray misses the plane (sky). `project(x, y, z=0)` is the inverse
  (extra, for the page). `set_pickables([(id, (K,2))])` / `hit_test(x, y)` — even-odd point-in-polygon
  with a bbox prefilter, last wins. Event positions use `localPos()` (float), not `ev.x()` (int): at a
  wide zoom one pixel is centimetres of ground.
* **Overlays** — `set_cursor(kind, x, y, radius, yaw)` (`none | brush | cross | place`), `set_selection(polys,
  colour)` (accent), `set_hover(poly)` (`text.1`, thin), `set_preview(pts, width, closed)` (warn: centre line +
  two offset band edges; a single point draws a small ring), `set_gizmo(x, y, yaw, radius, on)` (ring +
  heading tick in `ego`, move cross in `text.0`). All overlay polylines are packed into **one dynamic
  `LINES` buffer**, grouped by line width (1.0 / 1.5 / 2.0 px), rebuilt only when dirty, drawn after the
  scene with **depth test off** and blending on. Buffers are released in `teardown` /
  `_on_context_destroyed` before the scene's program goes.
* **Layers** — `set_layer_visibility(ducts, walls, props, floor)`; `floor=False` also hides the backdrop
  disc. `last_static_drawn` records how many static meshes the last paint drew.
* **paintGL** — resolve target → apply pending geometry → `Scene.draw_static(...)` with the free camera
  (fog / near / far from the inherited `_depth_range`) → overlays; empty-state text
  `"환경을 새로 만들거나 목록에서 고르세요."` when geometry is `None`. `_load_car` is a no-op (no cars are ever drawn).
  `set_corner_text` is inherited. The overlay chrome labels are `WA_TransparentForMouseEvents` so a hover
  over the empty-state text still reaches the page.

### Shared-file changes (additive, backward-compatible)

* `f1sim/viewer/console/viewport.py` — new `ViewportWidget._upload_static(layer, arrays, **kw)`; the five
  `add_static_mesh` calls in `_apply_geometry` go through it and tag each `Mesh.layer` with
  `backdrop | floor | ducts | walls | props`. `draw_frame` and every session path ignore the tag.
* `f1sim/viewer/gl_scene.py` — new `Scene.draw_static(view, proj, eye, light_center, hidden=(), show_lines=True)`:
  shadow pass + sky + static meshes (skipping layers in `hidden` in both passes) + named lines; returns the
  number of meshes drawn. `draw_frame` is untouched.

### Tests: `f1sim/tests/test_editor_viewport.py` (23 tests)

Offscreen `QApplication` + real GL under Xvfb, same fixture pattern as `test_console_gl_lifecycle.py`. The
widget fixture connects `gl_failed` and **fails any test whose paint raised** (a caught paint exception
clears the frame, which would otherwise pass a "pixels changed" check — this caught a real bug during
development). Covers: construction/empty text; upload tags layers and auto-frames; re-upload keeps the
camera; `frame_all` puts all corners on screen and fills > 0.6 of the half-extent (perspective, ortho top,
perspective top); projection round trip within 1 cm at five poses; sky miss; top-view toggle restores az/el;
`hit_test` on squares (last wins); synthetic press/move/release with world coordinates and hit id; hover;
double-click without a second press; orbit/pan by middle/right/Shift/Alt with no edit signals; wheel zoom
keeps the cursor's ground point and Ctrl/Alt+wheel is forwarded; keys F/Home/5/Space and forwarding;
layer visibility by draw count **and** framebuffer pixel diff (and exact restoration); overlays change pixels
where the brush ring is and clearing them restores the bare picture exactly; screenshot + timing signals.

### Screenshot

`work/env-editor/viewport-b.png` (1280×800, `grabFramebuffer` under Xvfb/llvmpipe): fake oval hoses, wall
ring + pillar, one box prop, centerline, brush cursor, selection outline, hover outline, preview band and
gizmo. Script: `/tmp/claude-1000/-home-shchon11-F1tenth/1ba4bd0e-d409-4b02-9dc7-edc80cb2df59/scratchpad/shot_editor.py`
(scratch; not part of the repo).

## Test commands and results

All run from `f1sim/` with the venv active (`source /home/shchon11/F1tenth/activate.sh`). `QT_QPA_PLATFORM=xcb`
is what the GL tests need under `xvfb-run` (the offscreen platform gives no usable GL context for
`QOpenGLWidget`; the layout/props fixtures `setdefault` offscreen and therefore keep xcb when it is set).

```
xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 QT_QPA_PLATFORM=xcb python -m pytest \
    tests/test_editor_viewport.py tests/test_console_gl_lifecycle.py tests/test_console_geometry_v2.py \
    tests/test_console_layout.py tests/test_console_props.py -q
→ 119 passed in 54.13s
```

```
xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 QT_QPA_PLATFORM=xcb python -m pytest tests/test_editor_viewport.py -q
→ 23 passed in 12.24s   (rerun after the final lint cleanup: 23 passed in 12.99s)
```

```
xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 QT_QPA_PLATFORM=xcb python -m pytest \
    tests/test_console_frames.py tests/test_console_session.py tests/test_console_startup.py \
    tests/test_console_launch_smoke.py tests/test_sim_worker_console_integration.py -q
→ 102 passed, 1 failed in 144.32s
   FAILED tests/test_console_startup.py::test_worker_and_console_agree_on_the_group_names
```

The one failure is **not from B**: the test string-searches `f1sim/viewer/sim_worker.py` for the literal
`return {\n            "기본 평가셋"` block, and `sim_worker.py` and `console/catalog.py` were both rewritten at
21:26:27 (worker A's `map_catalog()` / `list_scenes()` change under contract A.4), after my last edit to any
shared file and outside my package. B never touched either file. A (or root) should update that test to the
new `map_catalog` shape.

Baseline before any B change: `tests/test_console_gl_lifecycle.py tests/test_console_geometry_v2.py` →
59 passed (same command), so those 59 are unchanged by the `viewport.py` / `gl_scene.py` additions.

`pyflakes` on the new files is clean. `import f1sim.viewer.console.editor_viewport` does not import torch.

## Deviations from the contract, and why

* `ortho_top` defaults to **True** (contract only lists the attribute). A map editor paints on a grid; under
  perspective the brush ring is only a circle at the centre of the screen. The page can set it `False`.
* `wheel_edit` is emitted for **Ctrl or Alt** + wheel and the wheel is then not consumed, exactly as
  specified; a plain wheel zooms toward the cursor's ground point.
* `ground_released` uses the current ground point, or the **last on-plane point** of the drag if the release
  happens over the sky, so a stroke that ends off-plane still ends. A release that follows a double-click
  (no open edit drag) is not emitted.
* Orbit-dragging while in the top view **exits** the top view (starting from el = 89° so the tilt is
  continuous) and forgets the saved perspective — that drag *is* the new perspective. `set_top_view(False)`
  by key/button restores the saved one as specified.
* Extra public helpers not in the contract, all harmless: `project(x, y, z=0)`, `camera_state()`,
  `last_static_drawn`, module-level `point_in_polygon`. `_load_car` is overridden to skip the car GLB.
* The coordinator's mid-task addition `double_clicked(float, float)` is implemented (see above).
